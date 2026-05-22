use std::io::{Read, Seek, SeekFrom, Write};

use crate::error::{Error, Result};
use crate::format::messages::data_layout::{ChunkIndexType, LayoutClass};
use crate::format::messages::fill_value::FILL_TIME_NEVER;
use crate::format::object_header::{
    self, HDR_ATTR_STORE_PHASE_CHANGE, HDR_CHUNK0_SIZE_MASK, HDR_STORE_TIMES, HDR_V2_KNOWN_FLAGS,
};

use super::MutableFile;

struct ObjectHeaderMessageLocation {
    msg_data_offset: u64,
    msg_data_len: usize,
    v2_checksum: Option<(u64, usize)>,
}

impl MutableFile {
    /// Resize a chunked dataset to new dimensions.
    ///
    /// The new dimensions must not exceed the dataset's maximum dimensions.
    /// Whole v2-B-tree chunks outside a shrunken extent are pruned; other
    /// chunk data is left in place.
    /// New regions are filled with the default fill value (zeros).
    pub fn resize_dataset(&mut self, path: &str, new_dims: &[u64]) -> Result<()> {
        let ds = self.dataset(path)?;
        let info = ds.info()?;

        if info.layout.layout_class != LayoutClass::Chunked {
            return Err(Error::InvalidFormat(
                "can only resize chunked datasets".into(),
            ));
        }

        if new_dims.len() != info.dataspace.dims.len() {
            return Err(Error::InvalidFormat(format!(
                "dimension count mismatch: dataset has {} dims, new shape has {}",
                info.dataspace.dims.len(),
                new_dims.len()
            )));
        }

        if let Some(ref max_dims) = info.dataspace.max_dims {
            for (i, (&new_d, &max_d)) in new_dims.iter().zip(max_dims.iter()).enumerate() {
                if max_d != u64::MAX && new_d > max_d {
                    return Err(Error::InvalidFormat(format!(
                        "dimension {i}: new size {new_d} exceeds max {max_d}"
                    )));
                }
            }
        }

        let shrinks = new_dims
            .iter()
            .zip(&info.dataspace.dims)
            .any(|(&new_dim, &old_dim)| new_dim < old_dim);
        if shrinks
            && (info.layout.version <= 3
                || matches!(
                    info.layout.chunk_index_type,
                    Some(ChunkIndexType::BTreeV1)
                        | Some(ChunkIndexType::BTreeV2)
                        | Some(ChunkIndexType::FixedArray)
                        | Some(ChunkIndexType::ExtensibleArray)
                ))
        {
            let chunk_dims = Self::chunk_data_dims(&info)?.to_vec();
            let index_addr = info.layout.chunk_index_addr.ok_or_else(|| {
                Error::InvalidFormat("chunked dataset missing chunk index address".into())
            })?;
            self.scrub_partial_shrink_chunks(path, &ds, &info, new_dims, &chunk_dims)?;
            if info.layout.chunk_index_type == Some(ChunkIndexType::BTreeV2) {
                self.prune_btree_v2_chunks_outside_extent(
                    index_addr,
                    &info,
                    new_dims,
                    &chunk_dims,
                )?;
            }
            let physical_eof = self.write_handle.seek(SeekFrom::End(0))?;
            let logical_eof = physical_eof
                .checked_sub(self.superblock.base_addr)
                .ok_or_else(|| {
                    Error::InvalidFormat("file EOF is before HDF5 base address".into())
                })?;
            self.rewrite_superblock_eof(logical_eof)?;
            self.reopen_reader()?;
        }

        let ds_addr = ds.addr();
        let location = self.find_message_in_oh(ds_addr, object_header::MSG_DATASPACE)?;

        let mut new_space = info.dataspace.clone();
        new_space.dims = new_dims.to_vec();
        let new_ds_bytes = new_space.encode()?;

        if new_ds_bytes.len() != location.msg_data_len {
            return Err(Error::InvalidFormat(format!(
                "dataspace message size changed ({} -> {}); in-place resize not possible",
                location.msg_data_len,
                new_ds_bytes.len()
            )));
        }

        let msg_data_offset = self.physical_addr(location.msg_data_offset, "dataspace message")?;
        self.write_handle.seek(SeekFrom::Start(msg_data_offset))?;
        self.write_handle.write_all(&new_ds_bytes)?;
        if let Some((oh_start, oh_check_len)) = location.v2_checksum {
            self.rewrite_oh_checksum(oh_start, oh_check_len)?;
        }
        self.write_handle.flush()?;
        self.reopen_reader()?;

        Ok(())
    }

    fn scrub_partial_shrink_chunks(
        &mut self,
        path: &str,
        ds: &crate::hl::dataset::Dataset,
        info: &crate::hl::dataset::DatasetInfo,
        new_dims: &[u64],
        chunk_dims: &[u64],
    ) -> Result<()> {
        if new_dims.len() != info.dataspace.dims.len() || chunk_dims.len() != new_dims.len() {
            return Ok(());
        }
        if chunk_dims.iter().any(|&dim| dim == 0) {
            return Ok(());
        }
        let old_dims = &info.dataspace.dims;
        let has_partial_boundary = old_dims.iter().zip(new_dims).zip(chunk_dims).any(
            |((&old_dim, &new_dim), &chunk_dim)| new_dim < old_dim && new_dim % chunk_dim != 0,
        );
        if !has_partial_boundary {
            return Ok(());
        }

        let element_size = Self::u64_to_usize(u64::from(info.datatype.size), "datatype size")?;
        let old_elements = Self::checked_product_u64(old_dims, "dataset element count")?;
        let total_bytes = Self::u64_to_usize(old_elements, "dataset element count")?
            .checked_mul(element_size)
            .ok_or_else(|| Error::InvalidFormat("dataset byte size overflow".into()))?;
        let raw = ds.read_raw()?;
        if raw.len() != total_bytes {
            return Err(Error::InvalidFormat(format!(
                "raw dataset read returned {} bytes, expected {total_bytes}",
                raw.len()
            )));
        }

        let chunk_elements = Self::checked_product_u64(chunk_dims, "chunk element count")?;
        let chunk_bytes = Self::u64_to_usize(chunk_elements, "chunk element count")?
            .checked_mul(element_size)
            .ok_or_else(|| Error::InvalidFormat("chunk byte size overflow".into()))?;
        let fill = Self::fill_value_bytes(info, element_size)?;
        let old_strides = Self::row_major_strides(old_dims)?;
        let chunk_strides = Self::row_major_strides(chunk_dims)?;
        let mut starts = vec![0; new_dims.len()];
        self.scrub_partial_shrink_chunks_recursive(
            path,
            0,
            &mut starts,
            old_dims,
            new_dims,
            chunk_dims,
            &old_strides,
            &chunk_strides,
            &raw,
            chunk_bytes,
            element_size,
            &fill,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn scrub_partial_shrink_chunks_recursive(
        &mut self,
        path: &str,
        dim: usize,
        starts: &mut [u64],
        old_dims: &[u64],
        new_dims: &[u64],
        chunk_dims: &[u64],
        old_strides: &[u64],
        chunk_strides: &[u64],
        raw: &[u8],
        chunk_bytes: usize,
        element_size: usize,
        fill: &[u8],
    ) -> Result<()> {
        if dim < starts.len() {
            let mut start = 0;
            while start < old_dims[dim] {
                starts[dim] = start;
                self.scrub_partial_shrink_chunks_recursive(
                    path,
                    dim + 1,
                    starts,
                    old_dims,
                    new_dims,
                    chunk_dims,
                    old_strides,
                    chunk_strides,
                    raw,
                    chunk_bytes,
                    element_size,
                    fill,
                )?;
                start = start.checked_add(chunk_dims[dim]).ok_or_else(|| {
                    Error::InvalidFormat("chunk start coordinate overflow".into())
                })?;
            }
            return Ok(());
        }

        if starts
            .iter()
            .zip(new_dims)
            .any(|(&start, &new_dim)| start >= new_dim)
        {
            return Ok(());
        }
        let boundary = starts
            .iter()
            .zip(old_dims)
            .zip(new_dims)
            .zip(chunk_dims)
            .any(|(((&start, &old_dim), &new_dim), &chunk_dim)| {
                new_dim < old_dim
                    && start < new_dim
                    && new_dim < start.saturating_add(chunk_dim).min(old_dim)
            });
        if !boundary {
            return Ok(());
        }

        let mut chunk = vec![0; chunk_bytes];
        for slot in chunk.chunks_exact_mut(element_size) {
            slot.copy_from_slice(fill);
        }
        let mut local = vec![0; starts.len()];
        Self::copy_retained_chunk_elements(
            0,
            starts,
            &mut local,
            old_dims,
            new_dims,
            chunk_dims,
            old_strides,
            chunk_strides,
            raw,
            &mut chunk,
            element_size,
        )?;
        self.write_chunk(path, starts, &chunk)
    }

    #[allow(clippy::too_many_arguments)]
    fn copy_retained_chunk_elements(
        dim: usize,
        starts: &[u64],
        local: &mut [u64],
        old_dims: &[u64],
        new_dims: &[u64],
        chunk_dims: &[u64],
        old_strides: &[u64],
        chunk_strides: &[u64],
        raw: &[u8],
        chunk: &mut [u8],
        element_size: usize,
    ) -> Result<()> {
        if dim < local.len() {
            for coord in 0..chunk_dims[dim] {
                local[dim] = coord;
                Self::copy_retained_chunk_elements(
                    dim + 1,
                    starts,
                    local,
                    old_dims,
                    new_dims,
                    chunk_dims,
                    old_strides,
                    chunk_strides,
                    raw,
                    chunk,
                    element_size,
                )?;
            }
            return Ok(());
        }

        let mut old_index = 0u64;
        let mut chunk_index = 0u64;
        for idx in 0..local.len() {
            let global = starts[idx]
                .checked_add(local[idx])
                .ok_or_else(|| Error::InvalidFormat("chunk coordinate overflow".into()))?;
            if global >= old_dims[idx] {
                return Ok(());
            }
            if global >= new_dims[idx] {
                return Ok(());
            }
            old_index =
                old_index
                    .checked_add(global.checked_mul(old_strides[idx]).ok_or_else(|| {
                        Error::InvalidFormat("dataset linear index overflow".into())
                    })?)
                    .ok_or_else(|| Error::InvalidFormat("dataset linear index overflow".into()))?;
            chunk_index =
                chunk_index
                    .checked_add(local[idx].checked_mul(chunk_strides[idx]).ok_or_else(|| {
                        Error::InvalidFormat("chunk linear index overflow".into())
                    })?)
                    .ok_or_else(|| Error::InvalidFormat("chunk linear index overflow".into()))?;
        }

        let old_byte = Self::u64_to_usize(old_index, "dataset linear index")?
            .checked_mul(element_size)
            .ok_or_else(|| Error::InvalidFormat("dataset byte offset overflow".into()))?;
        let chunk_byte = Self::u64_to_usize(chunk_index, "chunk linear index")?
            .checked_mul(element_size)
            .ok_or_else(|| Error::InvalidFormat("chunk byte offset overflow".into()))?;
        let old_end = old_byte
            .checked_add(element_size)
            .ok_or_else(|| Error::InvalidFormat("dataset byte range overflow".into()))?;
        let chunk_end = chunk_byte
            .checked_add(element_size)
            .ok_or_else(|| Error::InvalidFormat("chunk byte range overflow".into()))?;
        chunk[chunk_byte..chunk_end].copy_from_slice(&raw[old_byte..old_end]);
        Ok(())
    }

    fn checked_product_u64(values: &[u64], what: &str) -> Result<u64> {
        values.iter().try_fold(1u64, |acc, &value| {
            acc.checked_mul(value)
                .ok_or_else(|| Error::InvalidFormat(format!("{what} overflow")))
        })
    }

    fn row_major_strides(dims: &[u64]) -> Result<Vec<u64>> {
        let mut strides = vec![1; dims.len()];
        let mut stride = 1u64;
        for idx in (0..dims.len()).rev() {
            strides[idx] = stride;
            stride = stride
                .checked_mul(dims[idx])
                .ok_or_else(|| Error::InvalidFormat("row-major stride overflow".into()))?;
        }
        Ok(strides)
    }

    fn fill_value_bytes(
        info: &crate::hl::dataset::DatasetInfo,
        element_size: usize,
    ) -> Result<Vec<u8>> {
        let Some(fill) = &info.fill_value else {
            return Ok(vec![0; element_size]);
        };
        if fill.fill_time == FILL_TIME_NEVER {
            return Ok(vec![0; element_size]);
        }
        let Some(value) = fill.value.as_deref() else {
            return Ok(vec![0; element_size]);
        };
        if value.len() != element_size {
            return Err(Error::Unsupported(format!(
                "fill value size {} does not match element size {}",
                value.len(),
                element_size
            )));
        }
        Ok(value.to_vec())
    }

    /// Find a message of the given type in an object header.
    fn find_message_in_oh(
        &self,
        oh_addr: u64,
        target_msg_type: u16,
    ) -> Result<ObjectHeaderMessageLocation> {
        let mut guard = self.inner.lock();
        let reader = &mut guard.reader;
        reader.seek(oh_addr)?;

        let mut first_bytes = [0u8; 4];
        reader.read_bytes_into(&mut first_bytes)?;
        if first_bytes != [b'O', b'H', b'D', b'R'] {
            reader.seek(oh_addr)?;
            return Self::find_message_in_v1_oh(reader, target_msg_type);
        }

        let version = reader.read_u8()?;
        if version != 2 {
            return Err(Error::Unsupported(
                "resize only supported for v2 object headers".into(),
            ));
        }

        let flags = reader.read_u8()?;
        if flags & !HDR_V2_KNOWN_FLAGS != 0 {
            return Err(Error::InvalidFormat(format!(
                "object header v2 flags contain reserved bits: {flags:#04x}"
            )));
        }
        if flags & HDR_STORE_TIMES != 0 {
            reader.skip(16)?;
        }
        if flags & HDR_ATTR_STORE_PHASE_CHANGE != 0 {
            reader.skip(4)?;
        }

        let chunk0_size_bytes = 1u8 << (flags & HDR_CHUNK0_SIZE_MASK);
        let chunk0_data_size = reader.read_uint(chunk0_size_bytes)?;
        let chunk0_data_start = reader.position()?;
        let chunk0_data_end = chunk0_data_start
            .checked_add(chunk0_data_size)
            .ok_or_else(|| Error::InvalidFormat("object-header chunk range overflow".into()))?;
        let oh_check_len = usize::try_from(chunk0_data_end - oh_addr)
            .map_err(|_| Error::InvalidFormat("object-header checksum range overflow".into()))?;

        while reader.position()? < chunk0_data_end {
            let msg_header_pos = reader.position()?;
            if msg_header_pos
                .checked_add(4)
                .is_none_or(|end| end > chunk0_data_end)
            {
                break;
            }

            let msg_type = u16::from(reader.read_u8()?);
            let msg_size = usize::from(reader.read_u16()?);
            let _msg_flags = reader.read_u8()?;

            if flags & object_header::HDR_ATTR_CRT_ORDER_TRACKED != 0 {
                reader.skip(2)?;
            }

            let msg_data_offset = reader.position()?;
            let msg_size_u64 = Self::usize_to_u64(msg_size, "object-header message size")?;
            if msg_data_offset
                .checked_add(msg_size_u64)
                .is_none_or(|end| end > chunk0_data_end)
            {
                return Err(Error::InvalidFormat(
                    "object-header message payload exceeds chunk".into(),
                ));
            }

            if msg_type == target_msg_type {
                return Ok(ObjectHeaderMessageLocation {
                    msg_data_offset,
                    msg_data_len: msg_size,
                    v2_checksum: Some((oh_addr, oh_check_len)),
                });
            }

            reader.skip(msg_size_u64)?;
        }

        Err(Error::InvalidFormat(format!(
            "message type {target_msg_type:#06x} not found in object header"
        )))
    }

    fn find_message_in_v1_oh<R: Read + Seek>(
        reader: &mut crate::io::reader::HdfReader<R>,
        target_msg_type: u16,
    ) -> Result<ObjectHeaderMessageLocation> {
        let version = reader.read_u8()?;
        if version != 1 {
            return Err(Error::Unsupported(format!(
                "resize only supports v1/v2 object headers, got version {version}"
            )));
        }
        reader.skip(1)?;
        let _num_messages = reader.read_u16()?;
        let _refcount = reader.read_u32()?;
        let chunk_data_size = u64::from(reader.read_u32()?);
        reader.skip(4)?;

        let chunk_start = reader.position()?;
        let chunk_end = chunk_start
            .checked_add(chunk_data_size)
            .ok_or_else(|| Error::InvalidFormat("object header v1 chunk range overflow".into()))?;
        let mut continuations = Vec::new();
        if let Some(location) = Self::find_message_in_v1_message_chunk(
            reader,
            chunk_end,
            target_msg_type,
            &mut continuations,
        )? {
            return Ok(location);
        }

        while let Some((addr, len)) = continuations.pop() {
            reader.seek(addr)?;
            let chunk_end = addr.checked_add(len).ok_or_else(|| {
                Error::InvalidFormat("object header v1 continuation range overflow".into())
            })?;
            if let Some(location) = Self::find_message_in_v1_message_chunk(
                reader,
                chunk_end,
                target_msg_type,
                &mut continuations,
            )? {
                return Ok(location);
            }
        }

        Err(Error::InvalidFormat(format!(
            "message type {target_msg_type:#06x} not found in object header"
        )))
    }

    fn find_message_in_v1_message_chunk<R: Read + Seek>(
        reader: &mut crate::io::reader::HdfReader<R>,
        chunk_end: u64,
        target_msg_type: u16,
        continuations: &mut Vec<(u64, u64)>,
    ) -> Result<Option<ObjectHeaderMessageLocation>> {
        while reader.position()? < chunk_end {
            let msg_header_pos = reader.position()?;
            if msg_header_pos
                .checked_add(8)
                .is_none_or(|end| end > chunk_end)
            {
                break;
            }

            let msg_type = reader.read_u16()?;
            let msg_size = usize::from(reader.read_u16()?);
            let _msg_flags = reader.read_u8()?;
            reader.skip(3)?;
            let msg_data_offset = reader.position()?;
            let aligned_size = Self::align_u64(
                Self::usize_to_u64(msg_size, "object-header message size")?,
                8,
                "object-header message alignment",
            )?;
            let msg_end = msg_data_offset.checked_add(aligned_size).ok_or_else(|| {
                Error::InvalidFormat("object-header message range overflow".into())
            })?;
            if msg_end > chunk_end {
                return Err(Error::InvalidFormat(
                    "object-header message payload exceeds chunk".into(),
                ));
            }

            if msg_type == target_msg_type {
                return Ok(Some(ObjectHeaderMessageLocation {
                    msg_data_offset,
                    msg_data_len: msg_size,
                    v2_checksum: None,
                }));
            }

            if msg_type == object_header::MSG_HEADER_CONTINUATION {
                let used = u64::from(reader.sizeof_addr())
                    .checked_add(u64::from(reader.sizeof_size()))
                    .ok_or_else(|| {
                        Error::InvalidFormat("object-header continuation width overflow".into())
                    })?;
                let msg_size_u64 = Self::usize_to_u64(msg_size, "object-header message size")?;
                if msg_size_u64 < used {
                    return Err(Error::InvalidFormat(
                        "object-header continuation message is truncated".into(),
                    ));
                }
                let cont_addr = reader.read_addr()?;
                let cont_len = reader.read_length()?;
                continuations.push((cont_addr, cont_len));
            }

            reader.seek(msg_end)?;
        }
        Ok(None)
    }

    fn align_u64(value: u64, align: u64, what: &str) -> Result<u64> {
        let mask = align
            .checked_sub(1)
            .ok_or_else(|| Error::InvalidFormat(format!("{what} overflow")))?;
        value
            .checked_add(mask)
            .map(|value| value & !mask)
            .ok_or_else(|| Error::InvalidFormat(format!("{what} overflow")))
    }
}

#[cfg(test)]
mod tests {
    use crate::format::messages::dataspace::{DataspaceMessage, DataspaceType};

    #[test]
    fn v1_dataspace_encode_preserves_message_layout() {
        let space = DataspaceMessage {
            version: 1,
            space_type: DataspaceType::Simple,
            ndims: 1,
            dims: vec![322],
            max_dims: Some(vec![u64::MAX]),
        };
        let encoded = space.encode().unwrap();
        assert_eq!(encoded.len(), 24);
        assert_eq!(DataspaceMessage::decode(&encoded).unwrap(), space);
    }
}
