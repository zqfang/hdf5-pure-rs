use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::format::messages::filter_pipeline::FilterPipelineMessage;
use crate::io::reader::HdfReader;

use super::chunk_read::ChunkReadContext;
use super::{usize_from_u64, Dataset, DatasetInfo};

struct ChunkCopyPlan {
    out_strides: Vec<usize>,
    chunk_strides: Vec<usize>,
    data_dims: Vec<usize>,
    chunk_dims: Vec<usize>,
    #[cfg(test)]
    chunk_suffix_products: Vec<usize>,
    #[cfg(test)]
    total_chunk_elements: usize,
}

impl Dataset {
    pub(super) fn is_deflate_only_pipeline(pipeline: &FilterPipelineMessage) -> bool {
        pipeline.filters.len() == 1
            && pipeline.filters[0].id == crate::format::messages::filter_pipeline::FILTER_DEFLATE
    }

    pub(super) fn is_shuffle_deflate_pipeline(pipeline: &FilterPipelineMessage) -> bool {
        pipeline.filters.len() == 2
            && pipeline.filters[0].id == crate::format::messages::filter_pipeline::FILTER_SHUFFLE
            && pipeline.filters[1].id == crate::format::messages::filter_pipeline::FILTER_DEFLATE
    }

    /// Copy chunk data into the output buffer at the correct position.
    pub(super) fn copy_chunk_to_output(
        chunk_data: &[u8],
        coords: &[u64],
        data_dims: &[u64],
        chunk_dims: &[u64],
        element_size: usize,
        output: &mut [u8],
    ) -> Result<()> {
        let ndims = data_dims.len();

        if ndims == 1 {
            return Self::copy_chunk_1d(
                chunk_data,
                coords,
                data_dims,
                chunk_dims,
                element_size,
                output,
            );
        }

        let copy_plan = Self::build_chunk_copy_plan(data_dims, chunk_dims, element_size)?;
        Self::copy_chunk_nd(
            chunk_data,
            coords,
            data_dims,
            element_size,
            output,
            &copy_plan,
        )
    }

    fn copy_chunk_1d(
        chunk_data: &[u8],
        coords: &[u64],
        data_dims: &[u64],
        chunk_dims: &[u64],
        element_size: usize,
        output: &mut [u8],
    ) -> Result<()> {
        let start = usize_from_u64(coords[0], "chunk coordinate")?;
        let chunk_size = usize_from_u64(chunk_dims[0], "chunk dimension")?;
        let data_size = usize_from_u64(data_dims[0], "dataset dimension")?;
        if start >= data_size {
            return Ok(());
        }

        let n_copy = chunk_size.min(data_size - start);
        let src_bytes = n_copy
            .checked_mul(element_size)
            .ok_or_else(|| Error::InvalidFormat("chunk copy size overflow".into()))?;
        let dst_offset = start
            .checked_mul(element_size)
            .ok_or_else(|| Error::InvalidFormat("chunk copy offset overflow".into()))?;

        let Some(dst) =
            checked_window_mut(output, dst_offset, src_bytes, "chunk copy output range")?
        else {
            return Ok(());
        };
        let Some(src) = checked_window(chunk_data, 0, src_bytes, "chunk copy input range")? else {
            return Ok(());
        };
        dst.copy_from_slice(src);
        Ok(())
    }

    pub(super) fn try_read_full_chunk_1d_into_output<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        info: &DatasetInfo,
        chunk_ctx: &ChunkReadContext<'_>,
        coords: &[u64],
        addr: u64,
        read_size: usize,
        filter_mask: u32,
        output: &mut [u8],
        compressed_scratch: &mut Vec<u8>,
        shuffle_scratch: &mut Vec<u8>,
    ) -> Result<bool> {
        let Some(dst_range) = Self::full_chunk_1d_output_range(coords, chunk_ctx, output.len())?
        else {
            return Ok(false);
        };

        match info.filter_pipeline.as_ref() {
            None => {
                if read_size != dst_range.len() {
                    return Ok(false);
                }
                reader.seek(addr)?;
                reader.read_exact(&mut output[dst_range])?;
                Ok(true)
            }
            Some(pipeline) if pipeline.filters.is_empty() => {
                if read_size != dst_range.len() {
                    return Ok(false);
                }
                reader.seek(addr)?;
                reader.read_exact(&mut output[dst_range])?;
                Ok(true)
            }
            Some(pipeline) if Self::is_deflate_only_pipeline(pipeline) && filter_mask == 0 => {
                Self::read_chunk_into_scratch(reader, addr, read_size, compressed_scratch)?;
                crate::filters::deflate::decompress_exact_into(
                    compressed_scratch,
                    &mut output[dst_range],
                )?;
                Ok(true)
            }
            Some(pipeline) if Self::is_shuffle_deflate_pipeline(pipeline) && filter_mask == 0 => {
                Self::read_chunk_into_scratch(reader, addr, read_size, compressed_scratch)?;
                shuffle_scratch.resize(dst_range.len(), 0);
                crate::filters::deflate::decompress_exact_into(
                    compressed_scratch,
                    shuffle_scratch,
                )?;
                crate::filters::shuffle::unshuffle_into(
                    shuffle_scratch,
                    chunk_ctx.element_size,
                    &mut output[dst_range],
                )?;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    fn read_chunk_into_scratch<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        addr: u64,
        read_size: usize,
        scratch: &mut Vec<u8>,
    ) -> Result<()> {
        reader.seek(addr)?;
        scratch.resize(read_size, 0);
        reader.read_exact(scratch)?;
        Ok(())
    }

    fn full_chunk_1d_output_range(
        coords: &[u64],
        chunk_ctx: &ChunkReadContext<'_>,
        output_len: usize,
    ) -> Result<Option<std::ops::Range<usize>>> {
        if chunk_ctx.data_dims.len() != 1 || chunk_ctx.chunk_dims.len() != 1 || coords.len() != 1 {
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

    fn build_chunk_copy_plan(
        data_dims: &[u64],
        chunk_dims: &[u64],
        element_size: usize,
    ) -> Result<ChunkCopyPlan> {
        let ndims = data_dims.len();

        let data_dims_usize = data_dims
            .iter()
            .map(|&dim| usize_from_u64(dim, "dataset dimension"))
            .collect::<Result<Vec<_>>>()?;
        let chunk_dims_usize = chunk_dims
            .iter()
            .map(|&dim| usize_from_u64(dim, "chunk dimension"))
            .collect::<Result<Vec<_>>>()?;

        let mut out_strides = vec![0usize; ndims];
        out_strides[ndims - 1] = element_size;
        for i in (0..ndims - 1).rev() {
            out_strides[i] = out_strides[i + 1]
                .checked_mul(data_dims_usize[i + 1])
                .ok_or_else(|| Error::InvalidFormat("chunk output stride overflow".into()))?;
        }

        let mut chunk_strides = vec![0usize; ndims];
        chunk_strides[ndims - 1] = element_size;
        for i in (0..ndims - 1).rev() {
            chunk_strides[i] = chunk_strides[i + 1]
                .checked_mul(chunk_dims_usize[i + 1])
                .ok_or_else(|| Error::InvalidFormat("chunk stride overflow".into()))?;
        }

        #[cfg(test)]
        let chunk_suffix_products = {
            let mut products = vec![1usize; ndims];
            for d in (0..ndims - 1).rev() {
                products[d] = products[d + 1]
                    .checked_mul(chunk_dims_usize[d + 1])
                    .ok_or_else(|| Error::InvalidFormat("chunk suffix product overflow".into()))?;
            }
            products
        };

        #[cfg(test)]
        let total_chunk_elements = chunk_dims_usize.iter().try_fold(1usize, |acc, &dim| {
            acc.checked_mul(dim)
                .ok_or_else(|| Error::InvalidFormat("chunk element count overflow".into()))
        })?;

        Ok(ChunkCopyPlan {
            out_strides,
            chunk_strides,
            data_dims: data_dims_usize,
            chunk_dims: chunk_dims_usize,
            #[cfg(test)]
            chunk_suffix_products,
            #[cfg(test)]
            total_chunk_elements,
        })
    }

    fn copy_chunk_nd(
        chunk_data: &[u8],
        coords: &[u64],
        _data_dims: &[u64],
        element_size: usize,
        output: &mut [u8],
        copy_plan: &ChunkCopyPlan,
    ) -> Result<()> {
        let coords = coords
            .iter()
            .map(|&coord| usize_from_u64(coord, "chunk coordinate"))
            .collect::<Result<Vec<_>>>()?;
        let copy_counts = Self::chunk_copy_counts(&coords, copy_plan)?;
        if copy_counts.iter().any(|&count| count == 0) {
            return Ok(());
        }

        let span_start = Self::chunk_copy_span_start(&copy_counts, copy_plan);
        let span_elements = copy_counts[span_start..]
            .iter()
            .try_fold(1usize, |acc, &count| {
                acc.checked_mul(count)
                    .ok_or_else(|| Error::InvalidFormat("chunk copy span overflow".into()))
            })?;
        let span_bytes = span_elements
            .checked_mul(element_size)
            .ok_or_else(|| Error::InvalidFormat("chunk copy span byte overflow".into()))?;

        if span_start == 0 {
            let out_offset = Self::chunk_output_offset(&coords, copy_plan)?;
            Self::copy_chunk_span(chunk_data, output, out_offset, 0, span_bytes, element_size)?;
            return Ok(());
        }

        let mut idx = vec![0usize; span_start];
        loop {
            let out_offset = Self::chunk_outer_offset(&coords, &idx, span_start, copy_plan)?;
            let chunk_offset = Self::chunk_inner_offset(&idx, span_start, copy_plan)?;

            Self::copy_chunk_span(
                chunk_data,
                output,
                out_offset,
                chunk_offset,
                span_bytes,
                element_size,
            )?;

            if !Self::increment_index(&mut idx, &copy_counts[..span_start]) {
                break;
            }
        }
        Ok(())
    }

    fn chunk_copy_counts(coords: &[usize], copy_plan: &ChunkCopyPlan) -> Result<Vec<usize>> {
        coords
            .iter()
            .zip(&copy_plan.data_dims)
            .zip(&copy_plan.chunk_dims)
            .map(|((&coord, &data_dim), &chunk_dim)| {
                if coord >= data_dim {
                    Ok(0)
                } else {
                    Ok(chunk_dim.min(data_dim - coord))
                }
            })
            .collect()
    }

    fn chunk_copy_span_start(copy_counts: &[usize], copy_plan: &ChunkCopyPlan) -> usize {
        let mut span_start = copy_counts.len() - 1;
        while span_start > 0 {
            let dim = span_start;
            if copy_counts[dim] != copy_plan.chunk_dims[dim]
                || copy_counts[dim] != copy_plan.data_dims[dim]
            {
                break;
            }
            span_start -= 1;
        }
        span_start
    }

    fn chunk_output_offset(coords: &[usize], copy_plan: &ChunkCopyPlan) -> Result<usize> {
        coords
            .iter()
            .zip(&copy_plan.out_strides)
            .try_fold(0usize, |offset, (&coord, &stride)| {
                offset
                    .checked_add(coord.checked_mul(stride).ok_or_else(|| {
                        Error::InvalidFormat("chunk output offset overflow".into())
                    })?)
                    .ok_or_else(|| Error::InvalidFormat("chunk output offset overflow".into()))
            })
    }

    fn chunk_outer_offset(
        coords: &[usize],
        idx: &[usize],
        span_start: usize,
        copy_plan: &ChunkCopyPlan,
    ) -> Result<usize> {
        let mut offset = 0usize;
        for d in 0..span_start {
            let global = coords[d]
                .checked_add(idx[d])
                .ok_or_else(|| Error::InvalidFormat("chunk coordinate overflow".into()))?;
            offset = offset
                .checked_add(
                    global
                        .checked_mul(copy_plan.out_strides[d])
                        .ok_or_else(|| {
                            Error::InvalidFormat("chunk output offset overflow".into())
                        })?,
                )
                .ok_or_else(|| Error::InvalidFormat("chunk output offset overflow".into()))?;
        }
        offset = offset
            .checked_add(
                coords[span_start]
                    .checked_mul(copy_plan.out_strides[span_start])
                    .ok_or_else(|| Error::InvalidFormat("chunk output offset overflow".into()))?,
            )
            .ok_or_else(|| Error::InvalidFormat("chunk output offset overflow".into()))?;
        Ok(offset)
    }

    fn chunk_inner_offset(
        idx: &[usize],
        span_start: usize,
        copy_plan: &ChunkCopyPlan,
    ) -> Result<usize> {
        let mut offset = 0usize;
        for d in 0..span_start {
            offset =
                offset
                    .checked_add(idx[d].checked_mul(copy_plan.chunk_strides[d]).ok_or_else(
                        || Error::InvalidFormat("chunk input offset overflow".into()),
                    )?)
                    .ok_or_else(|| Error::InvalidFormat("chunk input offset overflow".into()))?;
        }
        Ok(offset)
    }

    fn copy_chunk_span(
        chunk_data: &[u8],
        output: &mut [u8],
        out_offset: usize,
        chunk_offset: usize,
        span_bytes: usize,
        element_size: usize,
    ) -> Result<()> {
        let Some(dst) =
            checked_window_mut(output, out_offset, span_bytes, "chunk copy output range")?
        else {
            return Self::copy_chunk_span_elementwise(
                chunk_data,
                output,
                out_offset,
                chunk_offset,
                span_bytes,
                element_size,
            );
        };
        let Some(src) = checked_window(
            chunk_data,
            chunk_offset,
            span_bytes,
            "chunk copy input range",
        )?
        else {
            return Self::copy_chunk_span_elementwise(
                chunk_data,
                output,
                out_offset,
                chunk_offset,
                span_bytes,
                element_size,
            );
        };
        dst.copy_from_slice(src);
        Ok(())
    }

    fn copy_chunk_span_elementwise(
        chunk_data: &[u8],
        output: &mut [u8],
        out_offset: usize,
        chunk_offset: usize,
        span_bytes: usize,
        element_size: usize,
    ) -> Result<()> {
        for offset in (0..span_bytes).step_by(element_size) {
            let Some(dst) = checked_window_mut(
                output,
                out_offset
                    .checked_add(offset)
                    .ok_or_else(|| Error::InvalidFormat("chunk output offset overflow".into()))?,
                element_size,
                "chunk copy output range",
            )?
            else {
                continue;
            };
            let Some(src) = checked_window(
                chunk_data,
                chunk_offset
                    .checked_add(offset)
                    .ok_or_else(|| Error::InvalidFormat("chunk input offset overflow".into()))?,
                element_size,
                "chunk copy input range",
            )?
            else {
                continue;
            };
            dst.copy_from_slice(src);
        }
        Ok(())
    }

    fn increment_index(idx: &mut [usize], limits: &[usize]) -> bool {
        for d in (0..idx.len()).rev() {
            idx[d] += 1;
            if idx[d] < limits[d] {
                return true;
            }
            idx[d] = 0;
        }
        false
    }

    #[cfg(test)]
    fn copy_chunk_nd_elementwise_reference(
        chunk_data: &[u8],
        coords: &[u64],
        data_dims: &[u64],
        element_size: usize,
        output: &mut [u8],
        copy_plan: &ChunkCopyPlan,
    ) -> Result<()> {
        let ndims = data_dims.len();
        let mut idx = vec![0usize; ndims];

        for elem_idx in 0..copy_plan.total_chunk_elements {
            let mut remaining = elem_idx;
            for d in 0..ndims {
                idx[d] = remaining / copy_plan.chunk_suffix_products[d];
                remaining %= copy_plan.chunk_suffix_products[d];
            }

            let mut in_bounds = true;
            let mut out_offset = 0usize;
            let mut chunk_offset = 0usize;
            for d in 0..ndims {
                let global = usize_from_u64(coords[d], "chunk coordinate")?
                    .checked_add(idx[d])
                    .ok_or_else(|| Error::InvalidFormat("chunk coordinate overflow".into()))?;
                if global >= usize_from_u64(data_dims[d], "dataset dimension")? {
                    in_bounds = false;
                    break;
                }
                out_offset =
                    out_offset
                        .checked_add(global.checked_mul(copy_plan.out_strides[d]).ok_or_else(
                            || Error::InvalidFormat("chunk output offset overflow".into()),
                        )?)
                        .ok_or_else(|| {
                            Error::InvalidFormat("chunk output offset overflow".into())
                        })?;
                chunk_offset = chunk_offset
                    .checked_add(idx[d].checked_mul(copy_plan.chunk_strides[d]).ok_or_else(
                        || Error::InvalidFormat("chunk input offset overflow".into()),
                    )?)
                    .ok_or_else(|| Error::InvalidFormat("chunk input offset overflow".into()))?;
            }

            if in_bounds {
                let Some(dst) = checked_window_mut(
                    output,
                    out_offset,
                    element_size,
                    "chunk copy output range",
                )?
                else {
                    continue;
                };
                let Some(src) = checked_window(
                    chunk_data,
                    chunk_offset,
                    element_size,
                    "chunk copy input range",
                )?
                else {
                    continue;
                };
                dst.copy_from_slice(src);
            }
        }
        Ok(())
    }
}

fn checked_window<'a>(
    data: &'a [u8],
    offset: usize,
    len: usize,
    context: &str,
) -> Result<Option<&'a [u8]>> {
    let end = offset
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} overflow")))?;
    Ok(data.get(offset..end))
}

fn checked_window_mut<'a>(
    data: &'a mut [u8],
    offset: usize,
    len: usize,
    context: &str,
) -> Result<Option<&'a mut [u8]>> {
    let end = offset
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} overflow")))?;
    Ok(data.get_mut(offset..end))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_window_rejects_offset_overflow() {
        let err = checked_window(&[], usize::MAX, 1, "test range").unwrap_err();
        assert!(
            err.to_string().contains("test range overflow"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn checked_window_mut_rejects_offset_overflow() {
        let mut data = [];
        let err = checked_window_mut(&mut data, usize::MAX, 1, "test range").unwrap_err();
        assert!(
            err.to_string().contains("test range overflow"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn copy_chunk_nd_matches_elementwise_for_middle_2d_chunk() {
        assert_copy_matches_reference(&[6, 7], &[2, 3], &[2, 3], 2, 0, 0);
    }

    #[test]
    fn copy_chunk_nd_matches_elementwise_for_edge_2d_chunk() {
        assert_copy_matches_reference(&[5, 5], &[3, 3], &[3, 3], 4, 0, 0);
    }

    #[test]
    fn copy_chunk_nd_matches_elementwise_for_full_suffix_span() {
        assert_copy_matches_reference(&[4, 3, 5], &[2, 0, 0], &[2, 3, 5], 1, 0, 0);
    }

    #[test]
    fn copy_chunk_nd_matches_elementwise_with_truncated_input() {
        assert_copy_matches_reference(&[4, 6], &[1, 2], &[3, 3], 1, 2, 0);
    }

    #[test]
    fn copy_chunk_nd_matches_elementwise_with_truncated_output() {
        assert_copy_matches_reference(&[4, 6], &[1, 2], &[3, 3], 1, 0, 3);
    }

    fn assert_copy_matches_reference(
        data_dims: &[u64],
        coords: &[u64],
        chunk_dims: &[u64],
        element_size: usize,
        truncate_input: usize,
        truncate_output: usize,
    ) {
        let plan = Dataset::build_chunk_copy_plan(data_dims, chunk_dims, element_size).unwrap();
        let chunk_len = plan.total_chunk_elements * element_size;
        let output_len = data_dims
            .iter()
            .try_fold(element_size, |acc, &dim| acc.checked_mul(dim as usize))
            .unwrap();
        let chunk_data = (0..chunk_len)
            .map(|byte| (byte % 251) as u8)
            .collect::<Vec<_>>();
        let input_len = chunk_len.saturating_sub(truncate_input);
        let output_len = output_len.saturating_sub(truncate_output);
        let mut optimized = vec![0xA5; output_len];
        let mut reference = optimized.clone();

        Dataset::copy_chunk_nd(
            &chunk_data[..input_len],
            coords,
            data_dims,
            element_size,
            &mut optimized,
            &plan,
        )
        .unwrap();
        Dataset::copy_chunk_nd_elementwise_reference(
            &chunk_data[..input_len],
            coords,
            data_dims,
            element_size,
            &mut reference,
            &plan,
        )
        .unwrap();

        assert_eq!(optimized, reference);
    }
}
