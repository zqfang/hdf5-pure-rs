use crate::error::{Error, Result};
use crate::filters;
use crate::format::fixed_array::FixedArrayElement;
use crate::io::reader::HdfReader;
use std::io::{Read, Seek};
use std::thread;

use super::chunk_read::ChunkReadContext;
use super::{u64_from_usize, usize_from_u64, Dataset, DatasetInfo};

const MIN_PARALLEL_LINEAR_DEFLATE_CHUNKS_1D: usize = 8;
const MIN_PARALLEL_LINEAR_DEFLATE_BYTES_1D: usize = 64 * 1024;

impl Dataset {
    pub(super) fn read_chunked_fixed_array<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        info: &DatasetInfo,
        chunk_ctx: &ChunkReadContext<'_>,
    ) -> Result<Vec<u8>> {
        let mut output = Self::scratch_output(chunk_ctx.total_bytes);
        Self::read_chunked_fixed_array_into(reader, info, chunk_ctx, &mut output)?;
        Ok(output)
    }

    pub(super) fn read_chunked_fixed_array_into<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        info: &DatasetInfo,
        chunk_ctx: &ChunkReadContext<'_>,
        output: &mut [u8],
    ) -> Result<()> {
        if output.len() != chunk_ctx.total_bytes {
            return Err(Error::InvalidFormat(format!(
                "fixed-array chunk output buffer has {} bytes, expected {}",
                output.len(),
                chunk_ctx.total_bytes
            )));
        }
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

        let mut elements = Vec::new();
        crate::format::fixed_array::read_fixed_array_chunks_into(
            reader,
            chunk_ctx.idx_addr,
            filtered,
            chunk_size_len,
            &mut elements,
        )?;
        let chunks_per_dim = Self::chunks_per_dim(chunk_ctx.data_dims, chunk_ctx.chunk_dims)?;
        if !Self::has_full_linear_chunk_coverage_1d(elements.iter(), chunk_ctx)? {
            Self::filled_data_into(
                chunk_ctx.total_bytes / chunk_ctx.element_size,
                chunk_ctx.element_size,
                info,
                output,
            )?;
        }
        let mut compressed_scratch = Vec::new();
        let mut shuffle_scratch = Vec::new();
        let mut raw_scratch = Vec::new();
        let mut coords = Vec::with_capacity(chunk_ctx.data_dims.len());
        let mut handled = if Self::linear_index_uses_unfiltered_coalescing(info) {
            Self::try_read_coalesced_linear_index_unfiltered_chunks_1d(
                reader,
                info,
                chunk_ctx,
                elements.iter(),
                "fixed-array chunk size",
                output,
            )?
        } else {
            Vec::new()
        };
        let logical_chunks = Self::expected_chunk_count_1d(chunk_ctx)?.unwrap_or(elements.len());
        if handled.is_empty() {
            handled.resize(logical_chunks.min(elements.len()), false);
        }
        Self::try_read_parallel_deflate_linear_chunks_1d(
            reader,
            info,
            chunk_ctx,
            &elements,
            output,
            &mut handled,
            "fixed-array chunk size",
        )?;

        for (chunk_index, element) in elements.iter().take(logical_chunks).enumerate() {
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

            if handled.get(chunk_index).copied().unwrap_or(false) {
                continue;
            }
            Self::implicit_chunk_coords_into(
                chunk_index,
                chunk_ctx.chunk_dims,
                &chunks_per_dim,
                &mut coords,
            )?;
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
                output,
                &mut compressed_scratch,
                &mut shuffle_scratch,
            )? {
                continue;
            }
            Self::read_linear_chunk_payload_into_scratch(
                reader,
                element.addr,
                read_size,
                "fixed-array",
                chunk_index,
                &mut raw_scratch,
            )?;

            if let Some(ref pipeline) = info.filter_pipeline {
                if !pipeline.filters.is_empty() {
                    let filtered = filters::apply_pipeline_reverse_with_mask_expected(
                        &raw_scratch,
                        pipeline,
                        chunk_ctx.element_size,
                        element.filter_mask,
                        chunk_ctx.chunk_bytes,
                    )?;
                    Self::copy_chunk_to_output(
                        &filtered,
                        &coords,
                        chunk_ctx.data_dims,
                        chunk_ctx.chunk_dims,
                        chunk_ctx.element_size,
                        output,
                    )?;
                    continue;
                }
            }

            Self::copy_chunk_to_output(
                &raw_scratch,
                &coords,
                chunk_ctx.data_dims,
                chunk_ctx.chunk_dims,
                chunk_ctx.element_size,
                output,
            )?;
        }

        Ok(())
    }

    pub(super) fn read_chunked_extensible_array<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        info: &DatasetInfo,
        chunk_ctx: &ChunkReadContext<'_>,
    ) -> Result<Vec<u8>> {
        let mut output = Self::scratch_output(chunk_ctx.total_bytes);
        Self::read_chunked_extensible_array_into(reader, info, chunk_ctx, &mut output)?;
        Ok(output)
    }

    pub(super) fn read_chunked_extensible_array_into<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        info: &DatasetInfo,
        chunk_ctx: &ChunkReadContext<'_>,
        output: &mut [u8],
    ) -> Result<()> {
        if output.len() != chunk_ctx.total_bytes {
            return Err(Error::InvalidFormat(format!(
                "extensible-array chunk output buffer has {} bytes, expected {}",
                output.len(),
                chunk_ctx.total_bytes
            )));
        }
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

        let mut elements = Vec::new();
        crate::format::extensible_array::read_extensible_array_chunks_into(
            reader,
            chunk_ctx.idx_addr,
            filtered,
            chunk_size_len,
            &mut elements,
        )?;
        let chunks_per_dim = Self::chunks_per_dim(chunk_ctx.data_dims, chunk_ctx.chunk_dims)?;
        if !Self::has_full_linear_chunk_coverage_1d(elements.iter(), chunk_ctx)? {
            Self::filled_data_into(
                chunk_ctx.total_bytes / chunk_ctx.element_size,
                chunk_ctx.element_size,
                info,
                output,
            )?;
        }
        let mut compressed_scratch = Vec::new();
        let mut shuffle_scratch = Vec::new();
        let mut raw_scratch = Vec::new();
        let mut coords = Vec::with_capacity(chunk_ctx.data_dims.len());
        let mut handled = if Self::linear_index_uses_unfiltered_coalescing(info) {
            Self::try_read_coalesced_linear_index_unfiltered_chunks_1d(
                reader,
                info,
                chunk_ctx,
                elements.iter(),
                "extensible-array chunk size",
                output,
            )?
        } else {
            Vec::new()
        };
        let logical_chunks = Self::expected_chunk_count_1d(chunk_ctx)?.unwrap_or(elements.len());
        if handled.is_empty() {
            handled.resize(logical_chunks.min(elements.len()), false);
        }
        Self::try_read_parallel_deflate_linear_chunks_1d(
            reader,
            info,
            chunk_ctx,
            &elements,
            output,
            &mut handled,
            "extensible-array chunk size",
        )?;

        for (chunk_index, element) in elements.iter().take(logical_chunks).enumerate() {
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

            if handled.get(chunk_index).copied().unwrap_or(false) {
                continue;
            }
            Self::implicit_chunk_coords_into(
                chunk_index,
                chunk_ctx.chunk_dims,
                &chunks_per_dim,
                &mut coords,
            )?;
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
                output,
                &mut compressed_scratch,
                &mut shuffle_scratch,
            )? {
                continue;
            }
            Self::read_linear_chunk_payload_into_scratch(
                reader,
                element.addr,
                read_size,
                "extensible-array",
                chunk_index,
                &mut raw_scratch,
            )?;

            if let Some(ref pipeline) = info.filter_pipeline {
                if !pipeline.filters.is_empty() {
                    let filtered = filters::apply_pipeline_reverse_with_mask_expected(
                        &raw_scratch,
                        pipeline,
                        chunk_ctx.element_size,
                        element.filter_mask,
                        chunk_ctx.chunk_bytes,
                    )?;
                    Self::copy_chunk_to_output(
                        &filtered,
                        &coords,
                        chunk_ctx.data_dims,
                        chunk_ctx.chunk_dims,
                        chunk_ctx.element_size,
                        output,
                    )?;
                    continue;
                }
            }

            Self::copy_chunk_to_output(
                &raw_scratch,
                &coords,
                chunk_ctx.data_dims,
                chunk_ctx.chunk_dims,
                chunk_ctx.element_size,
                output,
            )?;
        }

        Ok(())
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

    fn linear_index_uses_unfiltered_coalescing(info: &DatasetInfo) -> bool {
        info.filter_pipeline
            .as_ref()
            .map(|pipeline| pipeline.filters.is_empty())
            .unwrap_or(true)
    }

    fn try_read_parallel_deflate_linear_chunks_1d<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        info: &DatasetInfo,
        chunk_ctx: &ChunkReadContext<'_>,
        elements: &[FixedArrayElement],
        output: &mut [u8],
        handled: &mut [bool],
        size_context: &'static str,
    ) -> Result<()> {
        let Some(pipeline) = info.filter_pipeline.as_ref() else {
            return Ok(());
        };
        if !Self::is_deflate_only_pipeline(pipeline) || chunk_ctx.data_dims.len() != 1 {
            return Ok(());
        }
        if chunk_ctx.chunk_dims.len() != 1 || output.len() != chunk_ctx.total_bytes {
            return Ok(());
        }

        let data_size = usize_from_u64(chunk_ctx.data_dims[0], "dataset dimension")?;
        let chunk_size = usize_from_u64(chunk_ctx.chunk_dims[0], "chunk dimension")?;
        if chunk_size == 0 {
            return Ok(());
        }
        let full_chunk_count = data_size / chunk_size;
        if full_chunk_count < MIN_PARALLEL_LINEAR_DEFLATE_CHUNKS_1D
            || elements.len() < full_chunk_count
        {
            return Ok(());
        }
        let chunk_bytes = chunk_size
            .checked_mul(chunk_ctx.element_size)
            .ok_or_else(|| Error::InvalidFormat("chunk byte size overflow".into()))?;
        if chunk_bytes == 0 || chunk_bytes != chunk_ctx.chunk_bytes {
            return Ok(());
        }
        let full_output_len = full_chunk_count
            .checked_mul(chunk_bytes)
            .ok_or_else(|| Error::InvalidFormat("parallel chunk output length overflow".into()))?;
        if full_output_len > output.len() || full_output_len < MIN_PARALLEL_LINEAR_DEFLATE_BYTES_1D
        {
            return Ok(());
        }

        let worker_count = super::support::parallel_deflate_worker_count(full_chunk_count);
        if worker_count <= 1 {
            return Ok(());
        }

        let mut payloads = Vec::with_capacity(full_chunk_count);
        for (chunk_index, element) in elements.iter().take(full_chunk_count).enumerate() {
            if handled.get(chunk_index).copied().unwrap_or(false)
                || element.filter_mask != 0
                || crate::io::reader::is_undef_addr(element.addr)
            {
                return Ok(());
            }
            let read_size = usize_from_u64(
                element
                    .nbytes
                    .unwrap_or(u64_from_usize(chunk_bytes, size_context)?),
                size_context,
            )?;
            let mut payload = vec![0u8; read_size];
            reader.seek(element.addr)?;
            reader.read_bytes_into(&mut payload).map_err(|err| {
                Error::InvalidFormat(format!(
                    "failed to read {size_context} chunk {chunk_index} at address {} with size {read_size}: {err}",
                    element.addr
                ))
            })?;
            payloads.push(payload);
        }

        let chunks_per_worker = full_chunk_count.div_ceil(worker_count);
        let full_output = &mut output[..full_output_len];
        let parallel_result: Result<()> = thread::scope(|scope| {
            let mut handles = Vec::new();
            let mut remaining_output = full_output;
            let mut payload_start = 0usize;
            while payload_start < full_chunk_count {
                let payload_end = (payload_start + chunks_per_worker).min(full_chunk_count);
                let group_chunks = payload_end - payload_start;
                let group_bytes = group_chunks * chunk_bytes;
                let (group_output, next_output) = remaining_output.split_at_mut(group_bytes);
                let group_payloads = &payloads[payload_start..payload_end];
                handles.push(scope.spawn(move || -> Result<()> {
                    for (chunk_offset, payload) in group_payloads.iter().enumerate() {
                        let start = chunk_offset * chunk_bytes;
                        let end = start + chunk_bytes;
                        crate::filters::deflate::decompress_exact_into(
                            payload,
                            &mut group_output[start..end],
                        )?;
                    }
                    Ok(())
                }));
                remaining_output = next_output;
                payload_start = payload_end;
            }
            for handle in handles {
                handle.join().map_err(|_| {
                    Error::InvalidFormat("parallel deflate worker panicked".into())
                })??;
            }
            Ok(())
        });
        parallel_result?;
        super::support::record_parallel_deflate_chunks_handled(full_chunk_count);

        for flag in handled.iter_mut().take(full_chunk_count) {
            *flag = true;
        }
        Ok(())
    }

    fn read_linear_chunk_payload_into_scratch<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        addr: u64,
        read_size: usize,
        index_name: &'static str,
        chunk_index: usize,
        scratch: &mut Vec<u8>,
    ) -> Result<()> {
        reader.seek(addr)?;
        scratch.resize(read_size, 0);
        reader.read_bytes_into(scratch).map_err(|err| {
            Error::InvalidFormat(format!(
                "failed to read {index_name} chunk {chunk_index} at address {addr} with size {read_size}: {err}"
            ))
        })?;
        Ok(())
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
