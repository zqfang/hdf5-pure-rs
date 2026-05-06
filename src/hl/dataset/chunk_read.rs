use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::filters;
use crate::format::messages::data_layout::ChunkIndexType;
use crate::io::reader::HdfReader;

use super::{u64_from_usize, usize_from_u64, Dataset, DatasetInfo};

#[derive(Debug, Clone)]
pub(super) struct ChunkBTreeRecord {
    pub(super) coords: Vec<u64>,
    pub(super) chunk_addr: u64,
    pub(super) chunk_size: u64,
    pub(super) filter_mask: u32,
}

pub(super) struct ChunkReadContext<'a> {
    pub(super) idx_addr: u64,
    pub(super) data_dims: &'a [u64],
    pub(super) chunk_dims: &'a [u64],
    pub(super) chunk_bytes: usize,
    pub(super) element_size: usize,
    pub(super) total_bytes: usize,
}

impl Dataset {
    pub(super) fn read_chunked<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        info: &DatasetInfo,
        total_bytes: usize,
    ) -> Result<Vec<u8>> {
        let element_size = usize_from_u64(u64::from(info.datatype.size), "datatype size")?;
        let data_dims = &info.dataspace.dims;
        let chunk_dims = info
            .layout
            .chunk_dims
            .as_ref()
            .ok_or_else(|| Error::InvalidFormat("chunked dataset missing chunk dims".into()))?;
        let chunk_data_dims = Self::chunk_data_dims(data_dims, chunk_dims)?;
        let chunk_bytes = Self::chunk_byte_len(chunk_dims, chunk_data_dims, element_size)?;
        let idx_addr = info
            .layout
            .chunk_index_addr
            .ok_or_else(|| Error::InvalidFormat("chunked dataset missing index address".into()))?;

        if crate::io::reader::is_undef_addr(idx_addr) {
            return Self::filled_data(total_bytes / element_size, element_size, info);
        }

        let chunk_ctx = ChunkReadContext {
            idx_addr,
            data_dims,
            chunk_dims: chunk_data_dims,
            chunk_bytes,
            element_size,
            total_bytes,
        };

        Self::read_chunked_with_index(reader, info, &chunk_ctx)
    }

    fn read_chunked_with_index<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        info: &DatasetInfo,
        chunk_ctx: &ChunkReadContext<'_>,
    ) -> Result<Vec<u8>> {
        match info.layout.chunk_index_type.clone() {
            Some(ChunkIndexType::SingleChunk) => Self::read_single_chunk(
                reader,
                chunk_ctx.idx_addr,
                info,
                chunk_ctx.chunk_bytes,
                chunk_ctx.element_size,
                chunk_ctx.total_bytes,
            ),
            Some(ChunkIndexType::BTreeV1) => Self::read_chunked_btree_v1(reader, info, chunk_ctx),
            Some(ChunkIndexType::Implicit) => Self::read_chunked_implicit(reader, info, chunk_ctx),
            Some(ChunkIndexType::FixedArray) => {
                Self::read_chunked_fixed_array(reader, info, chunk_ctx)
            }
            Some(ChunkIndexType::ExtensibleArray) => {
                Self::read_chunked_extensible_array(reader, info, chunk_ctx)
            }
            Some(ChunkIndexType::BTreeV2) => Self::read_chunked_btree_v2(reader, info, chunk_ctx),
            None if info.layout.version <= 3 => {
                Self::read_chunked_btree_v1(reader, info, chunk_ctx)
            }
            None => Err(Error::InvalidFormat(
                "chunked dataset missing chunk index type".into(),
            )),
        }
    }

    pub(super) fn chunk_data_dims<'a>(
        data_dims: &[u64],
        chunk_dims: &'a [u64],
    ) -> Result<&'a [u64]> {
        let ndims = data_dims.len();
        if chunk_dims.len() == ndims + 1 {
            Ok(&chunk_dims[..ndims])
        } else if chunk_dims.len() == ndims {
            Ok(chunk_dims)
        } else {
            Err(Error::InvalidFormat(format!(
                "chunk dims rank {} does not match dataspace rank {}",
                chunk_dims.len(),
                ndims
            )))
        }
    }

    pub(super) fn chunk_byte_len(
        chunk_dims: &[u64],
        chunk_data_dims: &[u64],
        element_size: usize,
    ) -> Result<usize> {
        if chunk_dims.len() == chunk_data_dims.len() + 1 {
            let bytes = chunk_dims
                .iter()
                .copied()
                .try_fold(1u64, |a, b| a.checked_mul(b))
                .ok_or_else(|| Error::InvalidFormat("chunk byte size overflow".into()))?;
            return usize_from_u64(bytes, "chunk byte size");
        }

        let chunk_elements: u64 = chunk_data_dims
            .iter()
            .copied()
            .try_fold(1u64, |a, b| a.checked_mul(b))
            .ok_or_else(|| Error::InvalidFormat("chunk dimension product overflow".into()))?;
        usize_from_u64(chunk_elements, "chunk element count")?
            .checked_mul(element_size)
            .ok_or_else(|| Error::InvalidFormat("chunk byte size overflow".into()))
    }

    fn read_single_chunk<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        chunk_addr: u64,
        info: &DatasetInfo,
        chunk_bytes: usize,
        element_size: usize,
        total_bytes: usize,
    ) -> Result<Vec<u8>> {
        reader.seek(chunk_addr)?;
        let read_size = usize_from_u64(
            info.layout
                .single_chunk_filtered_size
                .unwrap_or(u64_from_usize(chunk_bytes, "single-chunk size")?),
            "single-chunk size",
        )?;
        let mut raw = reader.read_bytes(read_size)?;

        if let Some(ref pipeline) = info.filter_pipeline {
            if !pipeline.filters.is_empty() {
                raw = filters::apply_pipeline_reverse_with_mask_expected(
                    &raw,
                    pipeline,
                    element_size,
                    info.layout.single_chunk_filter_mask.unwrap_or(0),
                    chunk_bytes,
                )?;
            }
        }

        if raw.len() < total_bytes {
            return Err(Error::InvalidFormat(
                "single-chunk data shorter than dataset size".into(),
            ));
        }
        Ok(raw[..total_bytes].to_vec())
    }

    pub(super) fn read_chunked_implicit<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        info: &DatasetInfo,
        chunk_ctx: &ChunkReadContext<'_>,
    ) -> Result<Vec<u8>> {
        if let Some(ref pipeline) = info.filter_pipeline {
            if !pipeline.filters.is_empty() {
                return Err(Error::Unsupported(
                    "v4 implicit chunk index with filters is not implemented".into(),
                ));
            }
        }

        let ndims = chunk_ctx.data_dims.len();
        let chunks_per_dim = Self::chunks_per_dim(chunk_ctx.data_dims, chunk_ctx.chunk_dims)?;
        let total_chunks: usize = chunks_per_dim
            .iter()
            .try_fold(1usize, |acc, &count| acc.checked_mul(count))
            .ok_or_else(|| Error::InvalidFormat("chunk count overflow".into()))?;

        let mut output = if Self::can_skip_chunk_prefill_for_implicit_1d(chunk_ctx)? {
            // Optimization opportunity: this path should fully overwrite the buffer, but proving
            // every fallback/error edge is safe needs a deeper audit before using uninitialized memory.
            Self::scratch_output(chunk_ctx.total_bytes)
        } else {
            Self::filled_data(
                chunk_ctx.total_bytes / chunk_ctx.element_size,
                chunk_ctx.element_size,
                info,
            )?
        };
        let mut compressed_scratch = Vec::new();
        for chunk_index in 0..total_chunks {
            let coords =
                Self::implicit_chunk_coords(chunk_index, chunk_ctx.chunk_dims, &chunks_per_dim)?;
            let chunk_index_u64 = u64_from_usize(chunk_index, "implicit chunk index")?;
            let chunk_bytes_u64 =
                u64_from_usize(chunk_ctx.chunk_bytes, "implicit chunk byte size")?;
            let offset = chunk_index_u64
                .checked_mul(chunk_bytes_u64)
                .and_then(|off| chunk_ctx.idx_addr.checked_add(off))
                .ok_or_else(|| Error::InvalidFormat("implicit chunk address overflow".into()))?;
            if Self::try_read_full_chunk_1d_into_output(
                reader,
                info,
                chunk_ctx,
                &coords,
                offset,
                chunk_ctx.chunk_bytes,
                0,
                &mut output,
                &mut compressed_scratch,
            )? {
                continue;
            }
            reader.seek(offset)?;
            let raw = reader.read_bytes(chunk_ctx.chunk_bytes)?;
            Self::copy_chunk_to_output(
                &raw,
                &coords,
                chunk_ctx.data_dims,
                chunk_ctx.chunk_dims,
                chunk_ctx.element_size,
                &mut output,
            )?;
        }

        if ndims == 0 {
            output.truncate(chunk_ctx.total_bytes.min(chunk_ctx.chunk_bytes));
        }
        Ok(output)
    }

    pub(super) fn has_full_chunk_coverage_1d(
        chunk_records: &[ChunkBTreeRecord],
        chunk_ctx: &ChunkReadContext<'_>,
    ) -> Result<bool> {
        if chunk_ctx.data_dims.len() != 1 || chunk_ctx.chunk_dims.len() != 1 {
            return Ok(false);
        }

        let data_size = usize_from_u64(chunk_ctx.data_dims[0], "dataset dimension")?;
        let chunk_size = usize_from_u64(chunk_ctx.chunk_dims[0], "chunk dimension")?;
        if chunk_size == 0 {
            return Ok(false);
        }
        let expected_chunks = data_size.div_ceil(chunk_size);
        if chunk_records.len() != expected_chunks {
            return Ok(false);
        }

        for (index, record) in chunk_records.iter().enumerate() {
            if record.coords.len() != 1 {
                return Ok(false);
            }
            let expected_start = index
                .checked_mul(chunk_size)
                .ok_or_else(|| Error::InvalidFormat("chunk coordinate overflow".into()))?;
            if usize_from_u64(record.coords[0], "chunk coordinate")? != expected_start {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub(super) fn can_skip_chunk_prefill_for_implicit_1d(
        chunk_ctx: &ChunkReadContext<'_>,
    ) -> Result<bool> {
        Ok(Self::expected_chunk_count_1d(chunk_ctx)?.is_some())
    }

    pub(super) fn expected_chunk_count_1d(
        chunk_ctx: &ChunkReadContext<'_>,
    ) -> Result<Option<usize>> {
        if chunk_ctx.data_dims.len() != 1 || chunk_ctx.chunk_dims.len() != 1 {
            return Ok(None);
        }
        let data_size = usize_from_u64(chunk_ctx.data_dims[0], "dataset dimension")?;
        let chunk_size = usize_from_u64(chunk_ctx.chunk_dims[0], "chunk dimension")?;
        if chunk_size == 0 {
            return Ok(None);
        }
        Ok(Some(data_size.div_ceil(chunk_size)))
    }

    pub(super) fn scratch_output(total_bytes: usize) -> Vec<u8> {
        // This intentionally avoids the fill-value prefill cost, not allocation initialization.
        // Replacing it with uninitialized memory is an optimization opportunity that needs a
        // full proof that every selected chunk path writes all bytes before observation.
        vec![0; total_bytes]
    }

    pub(super) fn filtered_chunk_size_len(
        info: &DatasetInfo,
        chunk_bytes: usize,
        sizeof_size: usize,
    ) -> Result<usize> {
        if info.layout.version > 4 {
            return Ok(sizeof_size);
        }

        let bits = if chunk_bytes == 0 {
            0
        } else {
            usize::try_from(usize::BITS - chunk_bytes.leading_zeros())
                .map_err(|_| Error::InvalidFormat("chunk size bit count overflow".into()))?
        };
        Ok((1 + ((bits + 8) / 8)).min(8))
    }

    pub(super) fn chunks_per_dim(data_dims: &[u64], chunk_dims: &[u64]) -> Result<Vec<usize>> {
        if data_dims.len() != chunk_dims.len() {
            return Err(Error::InvalidFormat(
                "dataset rank does not match chunk rank".into(),
            ));
        }
        data_dims
            .iter()
            .zip(chunk_dims)
            .map(|(&dim, &chunk)| {
                if chunk == 0 {
                    return Err(Error::InvalidFormat("zero chunk dimension".into()));
                }
                let count = dim
                    .checked_add(chunk - 1)
                    .ok_or_else(|| Error::InvalidFormat("chunk count overflow".into()))?
                    / chunk;
                usize_from_u64(count, "chunks per dimension")
            })
            .collect()
    }

    pub(super) fn implicit_chunk_coords(
        chunk_index: usize,
        chunk_dims: &[u64],
        chunks_per_dim: &[usize],
    ) -> Result<Vec<u64>> {
        if chunk_dims.len() != chunks_per_dim.len() {
            return Err(Error::InvalidFormat(
                "chunk dimension rank does not match chunk-count rank".into(),
            ));
        }
        let ndims = chunk_dims.len();
        let mut remaining = chunk_index;
        let mut coords = vec![0u64; ndims];

        for dim in (0..ndims).rev() {
            if chunks_per_dim[dim] == 0 {
                return Err(Error::InvalidFormat(
                    "chunks per dimension contains zero".into(),
                ));
            }
            let chunk_coord = remaining % chunks_per_dim[dim];
            remaining /= chunks_per_dim[dim];
            coords[dim] = u64_from_usize(chunk_coord, "implicit chunk coordinate")?
                .checked_mul(chunk_dims[dim])
                .ok_or_else(|| Error::InvalidFormat("implicit chunk coordinate overflow".into()))?;
        }

        Ok(coords)
    }
}
