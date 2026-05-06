use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::filters;
use crate::io::reader::HdfReader;

use super::chunk_read::ChunkReadContext;
use super::{u64_from_usize, usize_from_u64, Dataset, DatasetInfo};

impl Dataset {
    pub(super) fn read_chunked_fixed_array<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        info: &DatasetInfo,
        chunk_ctx: &ChunkReadContext<'_>,
    ) -> Result<Vec<u8>> {
        let filtered = info
            .filter_pipeline
            .as_ref()
            .map(|pipeline| !pipeline.filters.is_empty())
            .unwrap_or(false);
        let chunk_size_len = if filtered {
            Self::filtered_chunk_size_len(
                info,
                chunk_ctx.chunk_bytes,
                usize::from(reader.sizeof_size()),
            )?
        } else {
            0
        };

        let elements = crate::format::fixed_array::read_fixed_array_chunks(
            reader,
            chunk_ctx.idx_addr,
            filtered,
            chunk_size_len,
        )?;
        let chunks_per_dim = Self::chunks_per_dim(chunk_ctx.data_dims, chunk_ctx.chunk_dims)?;
        let mut output = if Self::has_full_linear_chunk_coverage_1d(elements.iter(), chunk_ctx)? {
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

        for (chunk_index, element) in elements.iter().enumerate() {
            Self::trace_linear_chunk_lookup(
                "hdf5.chunk_index.fixed_array.lookup",
                chunk_ctx.idx_addr,
                u64_from_usize(chunk_index, "fixed-array chunk index")?,
                element.addr,
                element.nbytes.unwrap_or(u64_from_usize(
                    chunk_ctx.chunk_bytes,
                    "fixed-array chunk size",
                )?),
                element.filter_mask,
            );

            if crate::io::reader::is_undef_addr(element.addr) {
                continue;
            }

            let coords =
                Self::implicit_chunk_coords(chunk_index, chunk_ctx.chunk_dims, &chunks_per_dim)?;
            let read_size = usize_from_u64(
                element.nbytes.unwrap_or(u64_from_usize(
                    chunk_ctx.chunk_bytes,
                    "fixed-array chunk size",
                )?),
                "fixed-array chunk size",
            )?;
            if Self::try_read_full_chunk_1d_into_output(
                reader,
                info,
                chunk_ctx,
                &coords,
                element.addr,
                read_size,
                element.filter_mask,
                &mut output,
                &mut compressed_scratch,
            )? {
                continue;
            }
            reader.seek(element.addr)?;
            let mut raw = reader.read_bytes(read_size).map_err(|err| {
                Error::InvalidFormat(format!(
                    "failed to read fixed-array chunk {chunk_index} at address {} with size {read_size}: {err}",
                    element.addr
                ))
            })?;

            if let Some(ref pipeline) = info.filter_pipeline {
                if !pipeline.filters.is_empty() {
                    raw = filters::apply_pipeline_reverse_with_mask_expected(
                        &raw,
                        pipeline,
                        chunk_ctx.element_size,
                        element.filter_mask,
                        chunk_ctx.chunk_bytes,
                    )?;
                }
            }

            Self::copy_chunk_to_output(
                &raw,
                &coords,
                chunk_ctx.data_dims,
                chunk_ctx.chunk_dims,
                chunk_ctx.element_size,
                &mut output,
            )?;
        }

        Ok(output)
    }

    pub(super) fn read_chunked_extensible_array<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        info: &DatasetInfo,
        chunk_ctx: &ChunkReadContext<'_>,
    ) -> Result<Vec<u8>> {
        let filtered = info
            .filter_pipeline
            .as_ref()
            .map(|pipeline| !pipeline.filters.is_empty())
            .unwrap_or(false);
        let chunk_size_len = if filtered {
            Self::filtered_chunk_size_len(
                info,
                chunk_ctx.chunk_bytes,
                usize::from(reader.sizeof_size()),
            )?
        } else {
            0
        };

        let elements = crate::format::extensible_array::read_extensible_array_chunks(
            reader,
            chunk_ctx.idx_addr,
            filtered,
            chunk_size_len,
        )?;
        let chunks_per_dim = Self::chunks_per_dim(chunk_ctx.data_dims, chunk_ctx.chunk_dims)?;
        let mut output = if Self::has_full_linear_chunk_coverage_1d(elements.iter(), chunk_ctx)? {
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

        for (chunk_index, element) in elements.iter().enumerate() {
            Self::trace_linear_chunk_lookup(
                "hdf5.chunk_index.extensible_array.lookup",
                chunk_ctx.idx_addr,
                u64_from_usize(chunk_index, "extensible-array chunk index")?,
                element.addr,
                element.nbytes.unwrap_or(u64_from_usize(
                    chunk_ctx.chunk_bytes,
                    "extensible-array chunk size",
                )?),
                element.filter_mask,
            );

            if crate::io::reader::is_undef_addr(element.addr) {
                continue;
            }

            let coords =
                Self::implicit_chunk_coords(chunk_index, chunk_ctx.chunk_dims, &chunks_per_dim)?;
            let read_size = usize_from_u64(
                element.nbytes.unwrap_or(u64_from_usize(
                    chunk_ctx.chunk_bytes,
                    "extensible-array chunk size",
                )?),
                "extensible-array chunk size",
            )?;
            if Self::try_read_full_chunk_1d_into_output(
                reader,
                info,
                chunk_ctx,
                &coords,
                element.addr,
                read_size,
                element.filter_mask,
                &mut output,
                &mut compressed_scratch,
            )? {
                continue;
            }
            reader.seek(element.addr)?;
            let mut raw = reader.read_bytes(read_size).map_err(|err| {
                Error::InvalidFormat(format!(
                    "failed to read extensible-array chunk {chunk_index} at address {} with size {read_size}: {err}",
                    element.addr
                ))
            })?;

            if let Some(ref pipeline) = info.filter_pipeline {
                if !pipeline.filters.is_empty() {
                    raw = filters::apply_pipeline_reverse_with_mask_expected(
                        &raw,
                        pipeline,
                        chunk_ctx.element_size,
                        element.filter_mask,
                        chunk_ctx.chunk_bytes,
                    )?;
                }
            }

            Self::copy_chunk_to_output(
                &raw,
                &coords,
                chunk_ctx.data_dims,
                chunk_ctx.chunk_dims,
                chunk_ctx.element_size,
                &mut output,
            )?;
        }

        Ok(output)
    }

    pub(super) fn has_full_linear_chunk_coverage_1d<'a, I>(
        elements: I,
        chunk_ctx: &ChunkReadContext<'_>,
    ) -> Result<bool>
    where
        I: IntoIterator<Item = &'a crate::format::fixed_array::FixedArrayElement>,
    {
        let Some(expected_chunks) = Self::expected_chunk_count_1d(chunk_ctx)? else {
            return Ok(false);
        };
        let mut count = 0usize;
        for element in elements {
            if crate::io::reader::is_undef_addr(element.addr) {
                return Ok(false);
            }
            count = count
                .checked_add(1)
                .ok_or_else(|| Error::InvalidFormat("chunk count overflow".into()))?;
        }
        Ok(count == expected_chunks)
    }

    #[cfg(feature = "tracehash")]
    fn trace_linear_chunk_lookup(
        function: &'static str,
        index_addr: u64,
        chunk_index: u64,
        addr: u64,
        nbytes: u64,
        filter_mask: u32,
    ) {
        let mut th = tracehash::Call::new(function, file!(), line!());
        th.input_u64(index_addr);
        th.input_u64(chunk_index);
        th.output_value(&(true));
        th.output_u64(addr);
        th.output_u64(if crate::io::reader::is_undef_addr(addr) {
            0
        } else {
            nbytes
        });
        th.output_u64(u64::from(filter_mask));
        th.finish();
    }

    #[cfg(not(feature = "tracehash"))]
    fn trace_linear_chunk_lookup(
        _function: &'static str,
        _index_addr: u64,
        _chunk_index: u64,
        _addr: u64,
        _nbytes: u64,
        _filter_mask: u32,
    ) {
    }
}
