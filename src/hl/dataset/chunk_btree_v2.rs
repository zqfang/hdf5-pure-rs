use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::filters;
use crate::io::reader::HdfReader;

use super::chunk_read::{BorrowedChunkPayloadRead, ChunkReadContext};
use super::{read_le_u32_at, read_le_uint_at, usize_from_u64, Dataset, DatasetInfo};

struct DecodedBTreeV2Chunk {
    coords: Vec<u64>,
    addr: u64,
    nbytes: u64,
    read_size: usize,
    filter_mask: u32,
}

impl Dataset {
    pub(super) fn read_chunked_btree_v2<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        info: &DatasetInfo,
        chunk_ctx: &ChunkReadContext<'_>,
    ) -> Result<Vec<u8>> {
        let mut output = Self::scratch_output(chunk_ctx.total_bytes);
        Self::read_chunked_btree_v2_into(reader, info, chunk_ctx, &mut output)?;
        Ok(output)
    }

    pub(super) fn read_chunked_btree_v2_into<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        info: &DatasetInfo,
        chunk_ctx: &ChunkReadContext<'_>,
        output: &mut [u8],
    ) -> Result<()> {
        if output.len() != chunk_ctx.total_bytes {
            return Err(Error::InvalidFormat(format!(
                "v2-B-tree chunk output buffer has {} bytes, expected {}",
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
        let mut records = Vec::new();
        crate::format::btree_v2::collect_all_records_into(
            reader,
            chunk_ctx.idx_addr,
            &mut records,
        )?;
        Self::filled_data_into(
            chunk_ctx.total_bytes / chunk_ctx.element_size,
            chunk_ctx.element_size,
            info,
            output,
        )?;
        let mut compressed_scratch = Vec::new();
        let mut raw_scratch = Vec::new();

        if !(Self::btree_v2_uses_unfiltered_coalescing(info)
            && chunk_ctx.data_dims.len() == 1
            && chunk_ctx.chunk_dims.len() == 1)
        {
            let mut coords = Vec::with_capacity(chunk_ctx.data_dims.len());
            let mut filtered_scratch = Vec::new();
            for record in &records {
                let (addr, nbytes, filter_mask) = Self::decode_btree_v2_chunk_record_into(
                    record,
                    filtered,
                    chunk_size_len,
                    usize::from(reader.sizeof_addr()),
                    chunk_ctx.data_dims.len(),
                    chunk_ctx.chunk_bytes,
                    &mut coords,
                )?;
                Self::trace_btree2_chunk_lookup(
                    chunk_ctx.idx_addr,
                    &coords,
                    addr,
                    nbytes,
                    filter_mask,
                );
                if crate::io::reader::is_undef_addr(addr) {
                    continue;
                }
                Self::scale_btree_v2_chunk_coords(&mut coords, chunk_ctx.chunk_dims)?;
                let read_size = usize_from_u64(nbytes, "v2-B-tree chunk size")?;
                if Self::try_read_full_chunk_1d_into_output(
                    reader,
                    info,
                    chunk_ctx,
                    &coords,
                    addr,
                    read_size,
                    filter_mask,
                    output,
                    &mut compressed_scratch,
                )? {
                    continue;
                }
                Self::read_btree_v2_chunk_payload_into_scratch(
                    reader,
                    addr,
                    nbytes,
                    read_size,
                    &mut raw_scratch,
                )?;

                if let Some(ref pipeline) = info.filter_pipeline {
                    if !pipeline.filters.is_empty() {
                        filters::apply_pipeline_reverse_with_mask_expected_into(
                            &raw_scratch,
                            pipeline,
                            chunk_ctx.element_size,
                            filter_mask,
                            chunk_ctx.chunk_bytes,
                            &mut filtered_scratch,
                        )?;
                        Self::copy_chunk_to_output(
                            &filtered_scratch,
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
            return Ok(());
        }

        let mut chunks = Vec::with_capacity(records.len());
        for record in &records {
            let (addr, nbytes, filter_mask, mut coords) = Self::decode_btree_v2_chunk_record(
                record,
                filtered,
                chunk_size_len,
                usize::from(reader.sizeof_addr()),
                chunk_ctx.data_dims.len(),
                chunk_ctx.chunk_bytes,
            )?;
            Self::trace_btree2_chunk_lookup(chunk_ctx.idx_addr, &coords, addr, nbytes, filter_mask);
            if crate::io::reader::is_undef_addr(addr) {
                continue;
            }

            Self::scale_btree_v2_chunk_coords(&mut coords, chunk_ctx.chunk_dims)?;
            let read_size = usize_from_u64(nbytes, "v2-B-tree chunk size")?;
            chunks.push(DecodedBTreeV2Chunk {
                coords,
                addr,
                nbytes,
                read_size,
                filter_mask,
            });
        }

        let handled = {
            Self::try_read_coalesced_borrowed_unfiltered_chunks_1d(
                reader,
                info,
                chunk_ctx,
                chunks.len(),
                chunks.iter().map(|chunk| {
                    Ok(BorrowedChunkPayloadRead {
                        coords: &chunk.coords,
                        addr: chunk.addr,
                        read_size: chunk.read_size,
                        filter_mask: chunk.filter_mask,
                    })
                }),
                output,
            )?
        };

        for (chunk_index, chunk) in chunks.iter().enumerate() {
            if handled.get(chunk_index).copied().unwrap_or(false) {
                continue;
            }
            if Self::try_read_full_chunk_1d_into_output(
                reader,
                info,
                chunk_ctx,
                &chunk.coords,
                chunk.addr,
                chunk.read_size,
                chunk.filter_mask,
                output,
                &mut compressed_scratch,
            )? {
                continue;
            }
            Self::read_btree_v2_chunk_payload_into_scratch(
                reader,
                chunk.addr,
                chunk.nbytes,
                chunk.read_size,
                &mut raw_scratch,
            )?;

            if let Some(ref pipeline) = info.filter_pipeline {
                if !pipeline.filters.is_empty() {
                    raw_scratch = filters::apply_pipeline_reverse_with_mask_expected(
                        &raw_scratch,
                        pipeline,
                        chunk_ctx.element_size,
                        chunk.filter_mask,
                        chunk_ctx.chunk_bytes,
                    )?;
                }
            }

            Self::copy_chunk_to_output(
                &raw_scratch,
                &chunk.coords,
                chunk_ctx.data_dims,
                chunk_ctx.chunk_dims,
                chunk_ctx.element_size,
                output,
            )?;
        }

        Ok(())
    }

    fn scale_btree_v2_chunk_coords(coords: &mut [u64], chunk_dims: &[u64]) -> Result<()> {
        for (coord, &chunk) in coords.iter_mut().zip(chunk_dims) {
            *coord = (*coord).checked_mul(chunk).ok_or_else(|| {
                Error::InvalidFormat("v2-B-tree chunk coordinate overflow".into())
            })?;
        }
        Ok(())
    }

    fn read_btree_v2_chunk_payload_into_scratch<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        addr: u64,
        nbytes: u64,
        read_size: usize,
        scratch: &mut Vec<u8>,
    ) -> Result<()> {
        reader.seek(addr).map_err(|err| {
            Error::InvalidFormat(format!(
                "failed to seek to v2-B-tree chunk address {addr}: {err}",
            ))
        })?;
        scratch.resize(read_size, 0);
        reader.read_bytes_into(scratch).map_err(|err| {
            Error::InvalidFormat(format!(
                "failed to read v2-B-tree chunk at address {addr} with size {nbytes}: {err}"
            ))
        })
    }

    fn btree_v2_uses_unfiltered_coalescing(info: &DatasetInfo) -> bool {
        info.filter_pipeline
            .as_ref()
            .map(|pipeline| pipeline.filters.is_empty())
            .unwrap_or(true)
    }

    #[cfg(feature = "tracehash")]
    fn trace_btree2_chunk_lookup(
        index_addr: u64,
        scaled: &[u64],
        addr: u64,
        nbytes: u64,
        filter_mask: u32,
    ) {
        let mut th = tracehash::th_call!("hdf5.chunk_index.btree2.lookup");
        th.input_u64(index_addr);
        for coord in scaled {
            th.input_u64(*coord);
        }
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
    fn trace_btree2_chunk_lookup(
        _index_addr: u64,
        _scaled: &[u64],
        _addr: u64,
        _nbytes: u64,
        _filter_mask: u32,
    ) {
    }

    #[cfg(feature = "tracehash")]
    fn trace_btree2_record_decode(
        record: &[u8],
        addr: u64,
        nbytes: u64,
        filter_mask: u32,
        scaled: &[u64],
    ) {
        let mut th = tracehash::th_call!("hdf5.chunk_index.btree2.record_decode");
        th.input_bytes(record);
        th.output_value(&(true));
        th.output_u64(addr);
        th.output_u64(nbytes);
        let Ok(scaled_len) = u64::try_from(scaled.len()) else {
            return;
        };
        th.output_u64(u64::from(filter_mask));
        th.output_u64(scaled_len);
        for coord in scaled {
            th.output_u64(*coord);
        }
        th.finish();
    }

    #[cfg(not(feature = "tracehash"))]
    fn trace_btree2_record_decode(
        _record: &[u8],
        _addr: u64,
        _nbytes: u64,
        _filter_mask: u32,
        _scaled: &[u64],
    ) {
    }

    pub(super) fn decode_btree_v2_chunk_record(
        record: &[u8],
        filtered: bool,
        chunk_size_len: usize,
        sizeof_addr: usize,
        ndims: usize,
        chunk_bytes: usize,
    ) -> Result<(u64, u64, u32, Vec<u64>)> {
        let mut scaled = Vec::with_capacity(ndims);
        let (addr, nbytes, filter_mask) = Self::decode_btree_v2_chunk_record_into(
            record,
            filtered,
            chunk_size_len,
            sizeof_addr,
            ndims,
            chunk_bytes,
            &mut scaled,
        )?;
        Ok((addr, nbytes, filter_mask, scaled))
    }

    pub(super) fn decode_btree_v2_chunk_record_into(
        record: &[u8],
        filtered: bool,
        chunk_size_len: usize,
        sizeof_addr: usize,
        ndims: usize,
        chunk_bytes: usize,
        scaled: &mut Vec<u64>,
    ) -> Result<(u64, u64, u32)> {
        let mut pos = 0usize;
        let addr = read_le_uint_at(record, &mut pos, sizeof_addr)?;

        let (nbytes, filter_mask) = if filtered {
            let nbytes = read_le_uint_at(record, &mut pos, chunk_size_len)?;
            let filter_mask = read_le_u32_at(record, &mut pos).map_err(|err| {
                if err.to_string().contains("truncated u32 field") {
                    Error::InvalidFormat("truncated v2-B-tree filter mask".into())
                } else {
                    err
                }
            })?;
            (nbytes, filter_mask)
        } else {
            (
                u64::try_from(chunk_bytes)
                    .map_err(|_| Error::InvalidFormat("v2-B-tree chunk size overflow".into()))?,
                0,
            )
        };

        scaled.clear();
        scaled.reserve(ndims);
        for _ in 0..ndims {
            scaled.push(read_le_uint_at(record, &mut pos, 8)?);
        }

        Self::trace_btree2_record_decode(record, addr, nbytes, filter_mask, scaled);

        Ok((addr, nbytes, filter_mask))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_btree_v2_record_rejects_truncated_filter_mask() {
        let record = [1u8; 8 + 1 + 3];
        let err = Dataset::decode_btree_v2_chunk_record(&record, true, 1, 8, 0, 16).unwrap_err();
        assert!(
            err.to_string().contains("truncated v2-B-tree filter mask"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn decode_btree_v2_record_rejects_truncated_scaled_coordinate() {
        let record = [1u8; 8 + 7];
        let err = Dataset::decode_btree_v2_chunk_record(&record, false, 0, 8, 1, 16).unwrap_err();
        assert!(
            err.to_string().contains("truncated integer field")
                || err
                    .to_string()
                    .contains("invalid little-endian integer size"),
            "unexpected error: {err}"
        );
    }
}
