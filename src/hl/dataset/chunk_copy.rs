use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::format::messages::filter_pipeline::FilterPipelineMessage;
use crate::io::reader::HdfReader;

use super::chunk_read::ChunkReadContext;
use super::{usize_from_u64, Dataset, DatasetInfo};

struct ChunkCopyPlan {
    out_strides: Vec<usize>,
    chunk_strides: Vec<usize>,
    chunk_suffix_products: Vec<usize>,
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
                let mut shuffled = vec![0u8; dst_range.len()];
                crate::filters::deflate::decompress_exact_into(compressed_scratch, &mut shuffled)?;
                crate::filters::shuffle::unshuffle_into(
                    &shuffled,
                    chunk_ctx.element_size,
                    &mut output[dst_range],
                )?;
                shuffled.clear();
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

        let mut out_strides = vec![0usize; ndims];
        out_strides[ndims - 1] = element_size;
        for i in (0..ndims - 1).rev() {
            out_strides[i] = out_strides[i + 1]
                .checked_mul(usize_from_u64(data_dims[i + 1], "dataset dimension")?)
                .ok_or_else(|| Error::InvalidFormat("chunk output stride overflow".into()))?;
        }

        let mut chunk_strides = vec![0usize; ndims];
        chunk_strides[ndims - 1] = element_size;
        for i in (0..ndims - 1).rev() {
            chunk_strides[i] = chunk_strides[i + 1]
                .checked_mul(usize_from_u64(chunk_dims[i + 1], "chunk dimension")?)
                .ok_or_else(|| Error::InvalidFormat("chunk stride overflow".into()))?;
        }

        let mut chunk_suffix_products = vec![1usize; ndims];
        for d in (0..ndims - 1).rev() {
            chunk_suffix_products[d] = chunk_suffix_products[d + 1]
                .checked_mul(usize_from_u64(chunk_dims[d + 1], "chunk dimension")?)
                .ok_or_else(|| Error::InvalidFormat("chunk suffix product overflow".into()))?;
        }

        let total_chunk_elements = chunk_dims.iter().try_fold(1usize, |acc, &dim| {
            acc.checked_mul(usize_from_u64(dim, "chunk dimension")?)
                .ok_or_else(|| Error::InvalidFormat("chunk element count overflow".into()))
        })?;

        Ok(ChunkCopyPlan {
            out_strides,
            chunk_strides,
            chunk_suffix_products,
            total_chunk_elements,
        })
    }

    fn copy_chunk_nd(
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
}
