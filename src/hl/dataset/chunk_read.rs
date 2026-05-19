use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::filters;
use crate::format::fixed_array::FixedArrayElement;
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

pub(super) struct BorrowedChunkPayloadRead<'a> {
    pub(super) coords: &'a [u64],
    pub(super) addr: u64,
    pub(super) read_size: usize,
    pub(super) filter_mask: u32,
}

impl Dataset {
    pub(super) fn read_chunked_into<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        info: &DatasetInfo,
        total_bytes: usize,
        out: &mut [u8],
    ) -> Result<()> {
        if out.len() != total_bytes {
            return Err(Error::InvalidFormat(format!(
                "chunked output buffer has {} bytes, expected {total_bytes}",
                out.len()
            )));
        }
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
            return Self::filled_data_into(total_bytes / element_size, element_size, info, out);
        }

        let chunk_ctx = ChunkReadContext {
            idx_addr,
            data_dims,
            chunk_dims: chunk_data_dims,
            chunk_bytes,
            element_size,
            total_bytes,
        };

        match info.layout.chunk_index_type {
            Some(ChunkIndexType::SingleChunk) => Self::read_single_chunk_into(
                reader,
                chunk_ctx.idx_addr,
                info,
                chunk_ctx.chunk_bytes,
                chunk_ctx.element_size,
                chunk_ctx.total_bytes,
                out,
            ),
            Some(ChunkIndexType::BTreeV1) => {
                Self::read_chunked_btree_v1_into(reader, info, &chunk_ctx, out)
            }
            Some(ChunkIndexType::Implicit) => {
                Self::read_chunked_implicit_into(reader, info, &chunk_ctx, out)
            }
            Some(ChunkIndexType::FixedArray) => {
                Self::read_chunked_fixed_array_into(reader, info, &chunk_ctx, out)
            }
            Some(ChunkIndexType::ExtensibleArray) => {
                Self::read_chunked_extensible_array_into(reader, info, &chunk_ctx, out)
            }
            Some(ChunkIndexType::BTreeV2) => {
                Self::read_chunked_btree_v2_into(reader, info, &chunk_ctx, out)
            }
            None if info.layout.version <= 3 => {
                Self::read_chunked_btree_v1_into(reader, info, &chunk_ctx, out)
            }
            _ => {
                let data = Self::read_chunked_with_index(reader, info, &chunk_ctx)?;
                out.copy_from_slice(&data);
                Ok(())
            }
        }
    }

    fn read_chunked_with_index<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        info: &DatasetInfo,
        chunk_ctx: &ChunkReadContext<'_>,
    ) -> Result<Vec<u8>> {
        match info.layout.chunk_index_type {
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
        let mut raw = vec![0u8; read_size];
        reader.read_bytes_into(&mut raw)?;

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
        raw.truncate(total_bytes);
        Ok(raw)
    }

    fn read_single_chunk_into<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        chunk_addr: u64,
        info: &DatasetInfo,
        chunk_bytes: usize,
        element_size: usize,
        total_bytes: usize,
        out: &mut [u8],
    ) -> Result<()> {
        reader.seek(chunk_addr)?;
        let read_size = usize_from_u64(
            info.layout
                .single_chunk_filtered_size
                .unwrap_or(u64_from_usize(chunk_bytes, "single-chunk size")?),
            "single-chunk size",
        )?;

        let filtered = info
            .filter_pipeline
            .as_ref()
            .map(|pipeline| !pipeline.filters.is_empty())
            .unwrap_or(false);
        if !filtered && read_size >= total_bytes {
            reader.read_exact(out)?;
            if read_size > total_bytes {
                reader.skip(u64_from_usize(
                    read_size - total_bytes,
                    "single-chunk trailing bytes",
                )?)?;
            }
            return Ok(());
        }

        let mut raw = vec![0u8; read_size];
        reader.read_bytes_into(&mut raw)?;
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
        out.copy_from_slice(&raw[..total_bytes]);
        Ok(())
    }

    pub(super) fn read_chunked_implicit<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        info: &DatasetInfo,
        chunk_ctx: &ChunkReadContext<'_>,
    ) -> Result<Vec<u8>> {
        let mut output = Self::scratch_output(chunk_ctx.total_bytes);
        Self::read_chunked_implicit_into(reader, info, chunk_ctx, &mut output)?;
        Ok(output)
    }

    pub(super) fn read_chunked_implicit_into<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        info: &DatasetInfo,
        chunk_ctx: &ChunkReadContext<'_>,
        output: &mut [u8],
    ) -> Result<()> {
        if output.len() != chunk_ctx.total_bytes {
            return Err(Error::InvalidFormat(format!(
                "implicit chunk output buffer has {} bytes, expected {}",
                output.len(),
                chunk_ctx.total_bytes
            )));
        }
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

        if !Self::can_skip_chunk_prefill_for_implicit_1d(chunk_ctx)? {
            Self::filled_data_into(
                chunk_ctx.total_bytes / chunk_ctx.element_size,
                chunk_ctx.element_size,
                info,
                output,
            )?;
        }
        let chunk_bytes_u64 = u64_from_usize(chunk_ctx.chunk_bytes, "implicit chunk byte size")?;
        let mut compressed_scratch = Vec::new();
        let mut raw_scratch = Vec::new();
        let mut coords = Vec::with_capacity(ndims);
        let handled = if Self::implicit_index_uses_unfiltered_coalescing(info, chunk_ctx) {
            Self::try_read_coalesced_implicit_unfiltered_chunks_1d(
                reader,
                info,
                chunk_ctx,
                total_chunks,
                chunk_bytes_u64,
                output,
            )?
        } else {
            Vec::new()
        };

        for chunk_index in 0..total_chunks {
            if handled.get(chunk_index).copied().unwrap_or(false) {
                continue;
            }
            Self::implicit_chunk_coords_into(
                chunk_index,
                chunk_ctx.chunk_dims,
                &chunks_per_dim,
                &mut coords,
            )?;
            let offset =
                Self::implicit_chunk_offset(chunk_ctx.idx_addr, chunk_index, chunk_bytes_u64)?;
            if Self::try_read_full_chunk_1d_into_output(
                reader,
                info,
                chunk_ctx,
                &coords,
                offset,
                chunk_ctx.chunk_bytes,
                0,
                output,
                &mut compressed_scratch,
            )? {
                continue;
            }
            reader.seek(offset)?;
            raw_scratch.resize(chunk_ctx.chunk_bytes, 0);
            reader.read_bytes_into(&mut raw_scratch)?;
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

    fn implicit_index_uses_unfiltered_coalescing(
        info: &DatasetInfo,
        chunk_ctx: &ChunkReadContext<'_>,
    ) -> bool {
        chunk_ctx.data_dims.len() == 1
            && chunk_ctx.chunk_dims.len() == 1
            && info
                .filter_pipeline
                .as_ref()
                .map(|pipeline| pipeline.filters.is_empty())
                .unwrap_or(true)
    }

    pub(super) fn try_read_coalesced_linear_index_unfiltered_chunks_1d<'a, R, I>(
        reader: &mut HdfReader<R>,
        info: &DatasetInfo,
        chunk_ctx: &ChunkReadContext<'_>,
        elements: I,
        size_context: &'static str,
        output: &mut [u8],
    ) -> Result<Vec<bool>>
    where
        R: Read + Seek,
        I: IntoIterator<Item = &'a FixedArrayElement>,
    {
        let mut handled = Vec::new();
        if !Self::can_direct_read_linear_index_unfiltered_chunks(info, chunk_ctx) {
            return Ok(handled);
        }

        let default_read_size = u64_from_usize(chunk_ctx.chunk_bytes, size_context)?;
        let mut run: Option<(u64, usize, usize)> = None;
        let mut run_start_index = 0usize;
        let mut run_end_index = 0usize;

        for (chunk_index, element) in elements.into_iter().enumerate() {
            handled.push(false);
            let read_size =
                usize_from_u64(element.nbytes.unwrap_or(default_read_size), size_context)?;
            let dst_range =
                if crate::io::reader::is_undef_addr(element.addr) || element.filter_mask != 0 {
                    None
                } else {
                    Self::linear_index_chunk_1d_output_range(chunk_index, chunk_ctx, output.len())?
                };

            let Some(dst_range) = dst_range else {
                Self::flush_linear_coalesced_run_and_mark(
                    reader,
                    output,
                    run.take(),
                    &mut handled,
                    run_start_index,
                    run_end_index,
                )?;
                continue;
            };
            if read_size != dst_range.len() {
                Self::flush_linear_coalesced_run_and_mark(
                    reader,
                    output,
                    run.take(),
                    &mut handled,
                    run_start_index,
                    run_end_index,
                )?;
                continue;
            }

            let can_extend = run
                .as_ref()
                .and_then(|&(file_start, dst_start, len)| {
                    let next_file = file_start.checked_add(u64::try_from(len).ok()?)?;
                    let next_dst = dst_start.checked_add(len)?;
                    Some(element.addr == next_file && dst_range.start == next_dst)
                })
                .unwrap_or(false);

            if can_extend {
                if let Some((_, _, len)) = run.as_mut() {
                    *len = len.checked_add(read_size).ok_or_else(|| {
                        Error::InvalidFormat("coalesced chunk read overflow".into())
                    })?;
                }
            } else {
                Self::flush_linear_coalesced_run_and_mark(
                    reader,
                    output,
                    run.take(),
                    &mut handled,
                    run_start_index,
                    run_end_index,
                )?;
                run = Some((element.addr, dst_range.start, read_size));
                run_start_index = chunk_index;
            }
            run_end_index = chunk_index + 1;
        }

        Self::flush_linear_coalesced_run_and_mark(
            reader,
            output,
            run.take(),
            &mut handled,
            run_start_index,
            run_end_index,
        )?;
        Ok(handled)
    }

    fn try_read_coalesced_implicit_unfiltered_chunks_1d<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        info: &DatasetInfo,
        chunk_ctx: &ChunkReadContext<'_>,
        total_chunks: usize,
        chunk_bytes_u64: u64,
        output: &mut [u8],
    ) -> Result<Vec<bool>> {
        if !Self::can_direct_read_linear_index_unfiltered_chunks(info, chunk_ctx) {
            return Ok(Vec::new());
        }

        let mut handled = vec![false; total_chunks];
        let mut run: Option<(u64, usize, usize)> = None;
        let mut run_start_index = 0usize;
        let mut run_end_index = 0usize;

        for chunk_index in 0..total_chunks {
            let offset =
                Self::implicit_chunk_offset(chunk_ctx.idx_addr, chunk_index, chunk_bytes_u64)?;
            let Some(dst_range) =
                Self::linear_index_chunk_1d_output_range(chunk_index, chunk_ctx, output.len())?
            else {
                Self::flush_linear_coalesced_run_and_mark(
                    reader,
                    output,
                    run.take(),
                    &mut handled,
                    run_start_index,
                    run_end_index,
                )?;
                continue;
            };
            if chunk_ctx.chunk_bytes != dst_range.len() {
                Self::flush_linear_coalesced_run_and_mark(
                    reader,
                    output,
                    run.take(),
                    &mut handled,
                    run_start_index,
                    run_end_index,
                )?;
                continue;
            }

            let can_extend = run
                .as_ref()
                .and_then(|&(file_start, dst_start, len)| {
                    let next_file = file_start.checked_add(u64::try_from(len).ok()?)?;
                    let next_dst = dst_start.checked_add(len)?;
                    Some(offset == next_file && dst_range.start == next_dst)
                })
                .unwrap_or(false);

            if can_extend {
                if let Some((_, _, len)) = run.as_mut() {
                    *len = len.checked_add(chunk_ctx.chunk_bytes).ok_or_else(|| {
                        Error::InvalidFormat("coalesced chunk read overflow".into())
                    })?;
                }
            } else {
                Self::flush_linear_coalesced_run_and_mark(
                    reader,
                    output,
                    run.take(),
                    &mut handled,
                    run_start_index,
                    run_end_index,
                )?;
                run = Some((offset, dst_range.start, chunk_ctx.chunk_bytes));
                run_start_index = chunk_index;
            }
            run_end_index = chunk_index + 1;
        }

        Self::flush_linear_coalesced_run_and_mark(
            reader,
            output,
            run.take(),
            &mut handled,
            run_start_index,
            run_end_index,
        )?;
        Ok(handled)
    }

    fn can_direct_read_linear_index_unfiltered_chunks(
        info: &DatasetInfo,
        chunk_ctx: &ChunkReadContext<'_>,
    ) -> bool {
        chunk_ctx.data_dims.len() == 1
            && chunk_ctx.chunk_dims.len() == 1
            && info
                .filter_pipeline
                .as_ref()
                .map(|pipeline| pipeline.filters.is_empty())
                .unwrap_or(true)
    }

    pub(super) fn try_read_coalesced_borrowed_unfiltered_chunks_1d<'a, R, I>(
        reader: &mut HdfReader<R>,
        info: &DatasetInfo,
        chunk_ctx: &ChunkReadContext<'_>,
        total_chunks: usize,
        reads: I,
        output: &mut [u8],
    ) -> Result<Vec<bool>>
    where
        R: Read + Seek,
        I: IntoIterator<Item = Result<BorrowedChunkPayloadRead<'a>>>,
    {
        if !Self::can_direct_read_borrowed_unfiltered_chunks(info, chunk_ctx) {
            return Ok(Vec::new());
        }

        let mut handled = vec![false; total_chunks];
        let mut run: Option<(u64, usize, usize)> = None;
        let mut run_start_index = 0usize;
        let mut run_end_index = 0usize;

        for (chunk_index, read) in reads.into_iter().enumerate() {
            let read = read?;
            let dst_range = if crate::io::reader::is_undef_addr(read.addr) || read.filter_mask != 0
            {
                None
            } else {
                Self::borrowed_chunk_1d_output_range(read.coords, chunk_ctx, output.len())?
            };
            let Some(dst_range) = dst_range else {
                Self::flush_linear_coalesced_run_and_mark(
                    reader,
                    output,
                    run.take(),
                    &mut handled,
                    run_start_index,
                    run_end_index,
                )?;
                continue;
            };
            if read.read_size != dst_range.len() {
                Self::flush_linear_coalesced_run_and_mark(
                    reader,
                    output,
                    run.take(),
                    &mut handled,
                    run_start_index,
                    run_end_index,
                )?;
                continue;
            }

            let can_extend = run
                .as_ref()
                .and_then(|&(file_start, dst_start, len)| {
                    let next_file = file_start.checked_add(u64::try_from(len).ok()?)?;
                    let next_dst = dst_start.checked_add(len)?;
                    Some(read.addr == next_file && dst_range.start == next_dst)
                })
                .unwrap_or(false);

            if can_extend {
                if let Some((_, _, len)) = run.as_mut() {
                    *len = len.checked_add(read.read_size).ok_or_else(|| {
                        Error::InvalidFormat("coalesced chunk read overflow".into())
                    })?;
                }
            } else {
                Self::flush_linear_coalesced_run_and_mark(
                    reader,
                    output,
                    run.take(),
                    &mut handled,
                    run_start_index,
                    run_end_index,
                )?;
                run = Some((read.addr, dst_range.start, read.read_size));
                run_start_index = chunk_index;
            }
            run_end_index = chunk_index + 1;
        }

        Self::flush_linear_coalesced_run_and_mark(
            reader,
            output,
            run.take(),
            &mut handled,
            run_start_index,
            run_end_index,
        )?;
        Ok(handled)
    }

    fn can_direct_read_borrowed_unfiltered_chunks(
        info: &DatasetInfo,
        chunk_ctx: &ChunkReadContext<'_>,
    ) -> bool {
        chunk_ctx.data_dims.len() == 1
            && chunk_ctx.chunk_dims.len() == 1
            && info
                .filter_pipeline
                .as_ref()
                .map(|pipeline| pipeline.filters.is_empty())
                .unwrap_or(true)
    }

    fn borrowed_chunk_1d_output_range(
        coords: &[u64],
        chunk_ctx: &ChunkReadContext<'_>,
        output_len: usize,
    ) -> Result<Option<std::ops::Range<usize>>> {
        if coords.len() != 1 || chunk_ctx.data_dims.len() != 1 || chunk_ctx.chunk_dims.len() != 1 {
            return Ok(None);
        }

        let start = usize_from_u64(coords[0], "chunk coordinate")?;
        let chunk_size = usize_from_u64(chunk_ctx.chunk_dims[0], "chunk dimension")?;
        let data_size = usize_from_u64(chunk_ctx.data_dims[0], "dataset dimension")?;
        if start >= data_size {
            return Ok(None);
        }
        let end = start
            .checked_add(chunk_size)
            .ok_or_else(|| Error::InvalidFormat("chunk coordinate overflow".into()))?;
        if end > data_size {
            return Ok(None);
        }

        let dst_offset = start
            .checked_mul(chunk_ctx.element_size)
            .ok_or_else(|| Error::InvalidFormat("chunk copy offset overflow".into()))?;
        let dst_len = chunk_size
            .checked_mul(chunk_ctx.element_size)
            .ok_or_else(|| Error::InvalidFormat("chunk copy size overflow".into()))?;
        let dst_end = dst_offset
            .checked_add(dst_len)
            .ok_or_else(|| Error::InvalidFormat("chunk copy end overflow".into()))?;
        if dst_end > output_len {
            return Ok(None);
        }
        Ok(Some(dst_offset..dst_end))
    }

    fn linear_index_chunk_1d_output_range(
        chunk_index: usize,
        chunk_ctx: &ChunkReadContext<'_>,
        output_len: usize,
    ) -> Result<Option<std::ops::Range<usize>>> {
        if chunk_ctx.data_dims.len() != 1 || chunk_ctx.chunk_dims.len() != 1 {
            return Ok(None);
        }

        let chunk_size = usize_from_u64(chunk_ctx.chunk_dims[0], "chunk dimension")?;
        let data_size = usize_from_u64(chunk_ctx.data_dims[0], "dataset dimension")?;
        let start = chunk_index
            .checked_mul(chunk_size)
            .ok_or_else(|| Error::InvalidFormat("chunk coordinate overflow".into()))?;
        if start >= data_size {
            return Ok(None);
        }
        let end = start
            .checked_add(chunk_size)
            .ok_or_else(|| Error::InvalidFormat("chunk coordinate overflow".into()))?;
        if end > data_size {
            return Ok(None);
        }

        let dst_offset = start
            .checked_mul(chunk_ctx.element_size)
            .ok_or_else(|| Error::InvalidFormat("chunk copy offset overflow".into()))?;
        let dst_len = chunk_size
            .checked_mul(chunk_ctx.element_size)
            .ok_or_else(|| Error::InvalidFormat("chunk copy size overflow".into()))?;
        let dst_end = dst_offset
            .checked_add(dst_len)
            .ok_or_else(|| Error::InvalidFormat("chunk copy end overflow".into()))?;
        if dst_end > output_len {
            return Ok(None);
        }
        Ok(Some(dst_offset..dst_end))
    }

    fn flush_linear_coalesced_run_and_mark<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        output: &mut [u8],
        run: Option<(u64, usize, usize)>,
        handled: &mut [bool],
        run_start_index: usize,
        run_end_index: usize,
    ) -> Result<()> {
        let Some((addr, dst_start, len)) = run else {
            return Ok(());
        };
        let dst_end = dst_start
            .checked_add(len)
            .ok_or_else(|| Error::InvalidFormat("coalesced chunk output overflow".into()))?;
        reader.seek(addr)?;
        reader.read_exact(&mut output[dst_start..dst_end])?;
        for is_handled in &mut handled[run_start_index..run_end_index] {
            *is_handled = true;
        }
        Ok(())
    }

    fn implicit_chunk_offset(idx_addr: u64, chunk_index: usize, chunk_bytes: u64) -> Result<u64> {
        u64_from_usize(chunk_index, "implicit chunk index")?
            .checked_mul(chunk_bytes)
            .and_then(|off| idx_addr.checked_add(off))
            .ok_or_else(|| Error::InvalidFormat("implicit chunk address overflow".into()))
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

    pub(super) fn implicit_chunk_coords_into(
        chunk_index: usize,
        chunk_dims: &[u64],
        chunks_per_dim: &[usize],
        coords: &mut Vec<u64>,
    ) -> Result<()> {
        if chunk_dims.len() != chunks_per_dim.len() {
            return Err(Error::InvalidFormat(
                "chunk dimension rank does not match chunk-count rank".into(),
            ));
        }
        let ndims = chunk_dims.len();
        let mut remaining = chunk_index;
        coords.clear();
        coords.resize(ndims, 0);

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

        Ok(())
    }
}
