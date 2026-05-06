use std::io::{Seek, SeekFrom, Write};

use crate::error::{Error, Result};
use crate::format::messages::data_layout::{ChunkIndexType, LayoutClass};
use crate::format::messages::filter_pipeline::{FILTER_DEFLATE, FILTER_FLETCHER32, FILTER_SHUFFLE};

use super::MutableFile;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WritableChunkIndexKind {
    BTreeV1,
    FixedArray,
    ExtensibleArray,
    BTreeV2,
}

impl MutableFile {
    /// Write a full uncompressed chunk and update the dataset's chunk index.
    ///
    /// This supports chunked datasets written by this crate where the chunk
    /// index variant is one of the currently mutable writer-side subsets.
    pub fn write_chunk(
        &mut self,
        dataset_path: &str,
        chunk_coords: &[u64],
        data: &[u8],
    ) -> Result<()> {
        let ds = self.dataset(dataset_path)?;
        let info = ds.info()?;

        if info.layout.layout_class != LayoutClass::Chunked {
            return Err(Error::InvalidFormat(
                "write_chunk only supports chunked datasets".into(),
            ));
        }
        if info.layout.version > 3
            && !matches!(
                info.layout.chunk_index_type,
                Some(ChunkIndexType::BTreeV1)
                    | Some(ChunkIndexType::FixedArray)
                    | Some(ChunkIndexType::ExtensibleArray)
                    | Some(ChunkIndexType::BTreeV2)
            )
        {
            return Err(Error::Unsupported(
                "write_chunk currently supports only v1 B-tree, fixed-array, simple extensible-array, and simple v2 B-tree chunk indexes".into(),
            ));
        }

        let chunk_data_dims = Self::chunk_data_dims(&info)?;
        Self::validate_chunk_coords(chunk_coords, &chunk_data_dims)?;

        let element_size = Self::u64_to_usize(u64::from(info.datatype.size), "datatype size")?;
        let expected_len = Self::expected_chunk_len(&chunk_data_dims, element_size)?;
        Self::validate_chunk_write_len(data.len(), expected_len)?;
        let filtered = Self::encode_chunk_write_data(&info, data, element_size)?;

        let index_addr = info
            .layout
            .chunk_index_addr
            .ok_or_else(|| Error::InvalidFormat("chunked dataset missing B-tree address".into()))?;

        let chunk_addr = self.write_handle.seek(SeekFrom::End(0))?;
        self.write_handle.write_all(&filtered)?;
        self.rewrite_chunk_index(
            index_addr,
            &info,
            chunk_coords,
            &chunk_data_dims,
            Self::usize_to_u64(filtered.len(), "filtered chunk size")?,
            chunk_addr,
            expected_len,
            element_size,
        )?;
        self.write_handle.flush()?;
        self.reopen_reader()?;

        Ok(())
    }

    fn chunk_data_dims(info: &crate::hl::dataset::DatasetInfo) -> Result<Vec<u64>> {
        let chunk_dims = info
            .layout
            .chunk_dims
            .as_ref()
            .ok_or_else(|| Error::InvalidFormat("chunked layout missing chunk dims".into()))?;
        if chunk_dims.len() == info.dataspace.dims.len() + 1 {
            Ok(chunk_dims[..info.dataspace.dims.len()].to_vec())
        } else if chunk_dims.len() == info.dataspace.dims.len() {
            Ok(chunk_dims.clone())
        } else {
            Err(Error::InvalidFormat(format!(
                "chunk dimension rank {} does not match dataset rank {}",
                chunk_dims.len(),
                info.dataspace.dims.len()
            )))
        }
    }

    fn validate_chunk_coords(chunk_coords: &[u64], chunk_data_dims: &[u64]) -> Result<()> {
        if chunk_coords.len() != chunk_data_dims.len() {
            return Err(Error::InvalidFormat(format!(
                "chunk coordinate rank {} does not match dataset rank {}",
                chunk_coords.len(),
                chunk_data_dims.len()
            )));
        }
        for ((idx, &coord), &chunk) in chunk_coords.iter().enumerate().zip(chunk_data_dims) {
            if chunk == 0 || coord % chunk != 0 {
                return Err(Error::InvalidFormat(format!(
                    "chunk coordinate {idx}={coord} is not aligned to chunk size {chunk}"
                )));
            }
        }
        Ok(())
    }

    fn expected_chunk_len(chunk_data_dims: &[u64], element_size: usize) -> Result<usize> {
        let chunk_elements = chunk_data_dims.iter().try_fold(1usize, |acc, &dim| {
            let dim = Self::u64_to_usize(dim, "chunk dimension")?;
            acc.checked_mul(dim)
                .ok_or_else(|| Error::InvalidFormat("chunk element count overflow".into()))
        })?;
        chunk_elements
            .checked_mul(element_size)
            .ok_or_else(|| Error::InvalidFormat("chunk byte size overflow".into()))
    }

    fn validate_chunk_write_len(actual_len: usize, expected_len: usize) -> Result<()> {
        if actual_len != expected_len {
            return Err(Error::InvalidFormat(format!(
                "chunk data has {actual_len} bytes, expected {expected_len}",
            )));
        }
        Ok(())
    }

    fn encode_chunk_write_data(
        info: &crate::hl::dataset::DatasetInfo,
        data: &[u8],
        element_size: usize,
    ) -> Result<Vec<u8>> {
        let mut filtered = data.to_vec();
        if let Some(ref pipeline) = info.filter_pipeline {
            for filter in &pipeline.filters {
                match filter.id {
                    FILTER_SHUFFLE => {
                        filtered = crate::filters::shuffle::shuffle(&filtered, element_size)?;
                    }
                    FILTER_DEFLATE => {
                        let level = filter.client_data.first().copied().unwrap_or(6);
                        filtered = crate::filters::deflate::compress(&filtered, level)?;
                    }
                    FILTER_FLETCHER32 => {
                        filtered = crate::filters::fletcher32::append_checksum(&filtered)?;
                    }
                    other => {
                        return Err(Error::Unsupported(format!(
                            "write_chunk cannot encode filter {other}"
                        )));
                    }
                }
            }
        }
        Ok(filtered)
    }

    fn rewrite_chunk_index(
        &mut self,
        index_addr: u64,
        info: &crate::hl::dataset::DatasetInfo,
        chunk_coords: &[u64],
        chunk_data_dims: &[u64],
        filtered_len: u64,
        chunk_addr: u64,
        expected_len: usize,
        element_size: usize,
    ) -> Result<()> {
        match Self::writable_chunk_index_kind(info) {
            WritableChunkIndexKind::FixedArray => self.rewrite_fixed_array_chunk(
                index_addr,
                &info,
                chunk_coords,
                chunk_data_dims,
                filtered_len,
                chunk_addr,
                expected_len,
            ),
            WritableChunkIndexKind::ExtensibleArray => self.rewrite_extensible_array_chunk(
                index_addr,
                &info,
                chunk_coords,
                chunk_data_dims,
                filtered_len,
                chunk_addr,
                expected_len,
            ),
            WritableChunkIndexKind::BTreeV2 => self.rewrite_btree_v2_chunk(
                index_addr,
                &info,
                chunk_coords,
                chunk_data_dims,
                filtered_len,
                chunk_addr,
                expected_len,
            ),
            WritableChunkIndexKind::BTreeV1 => self.rewrite_leaf_chunk_btree(
                index_addr,
                chunk_coords,
                Self::u64_to_u32(filtered_len, "filtered chunk size")?,
                chunk_addr,
                Self::usize_to_u32(element_size, "datatype size")?,
            ),
        }
    }

    fn writable_chunk_index_kind(info: &crate::hl::dataset::DatasetInfo) -> WritableChunkIndexKind {
        match info.layout.chunk_index_type {
            Some(ChunkIndexType::FixedArray) => WritableChunkIndexKind::FixedArray,
            Some(ChunkIndexType::ExtensibleArray) => WritableChunkIndexKind::ExtensibleArray,
            Some(ChunkIndexType::BTreeV2) => WritableChunkIndexKind::BTreeV2,
            _ => WritableChunkIndexKind::BTreeV1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_chunk_len_rejects_element_count_overflow() {
        let err = MutableFile::expected_chunk_len(&[u64::MAX, 2], 1).unwrap_err();
        assert!(
            err.to_string().contains("chunk element count"),
            "unexpected error: {err}"
        );
    }
}
