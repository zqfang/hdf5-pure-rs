use crate::error::{Error, Result};
use crate::filters;
use crate::io::reader::HdfReader;
use std::io::{Read, Seek};
use std::thread;

use super::chunk_read::{BorrowedChunkPayloadRead, ChunkReadContext};
use super::{read_le_u32_at, read_le_uint_at, usize_from_u64, Dataset, DatasetInfo};

struct DecodedBTreeV2Chunk {
    coords: Vec<u64>,
    addr: u64,
    nbytes: u64,
    read_size: usize,
    filter_mask: u32,
}

const MIN_PARALLEL_BTREE_V2_DEFLATE_CHUNKS_1D: usize = 8;
const MIN_PARALLEL_BTREE_V2_DEFLATE_BYTES_1D: usize = 64 * 1024;

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
        let mut shuffle_scratch = Vec::new();
        let mut raw_scratch = Vec::new();

        if !(Self::btree_v2_uses_unfiltered_coalescing(info)
            && chunk_ctx.data_dims.len() == 1
            && chunk_ctx.chunk_dims.len() == 1)
        {
            let mut coords = Vec::with_capacity(chunk_ctx.data_dims.len());
            let mut filtered_scratch = Vec::new();
            let mut chunks = Vec::with_capacity(records.len());
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
                chunks.push(DecodedBTreeV2Chunk {
                    coords: coords.clone(),
                    addr,
                    nbytes,
                    read_size,
                    filter_mask,
                });
            }

            let mut handled = vec![false; chunks.len()];
            Self::try_read_parallel_deflate_btree_v2_chunks_1d(
                reader,
                info,
                chunk_ctx,
                &chunks,
                output,
                &mut handled,
            )?;

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
                    &mut shuffle_scratch,
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
                        filters::apply_pipeline_reverse_with_mask_expected_into(
                            &raw_scratch,
                            pipeline,
                            chunk_ctx.element_size,
                            chunk.filter_mask,
                            chunk_ctx.chunk_bytes,
                            &mut filtered_scratch,
                        )?;
                        Self::copy_chunk_to_output(
                            &filtered_scratch,
                            &chunk.coords,
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
                    &chunk.coords,
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
                &mut shuffle_scratch,
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

    fn try_read_parallel_deflate_btree_v2_chunks_1d<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        info: &DatasetInfo,
        chunk_ctx: &ChunkReadContext<'_>,
        chunks: &[DecodedBTreeV2Chunk],
        output: &mut [u8],
        handled: &mut [bool],
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
        if full_chunk_count < MIN_PARALLEL_BTREE_V2_DEFLATE_CHUNKS_1D
            || chunks.len() < full_chunk_count
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
        if full_output_len > output.len()
            || full_output_len < MIN_PARALLEL_BTREE_V2_DEFLATE_BYTES_1D
        {
            return Ok(());
        }

        let worker_count = super::support::parallel_deflate_worker_count(full_chunk_count);
        if worker_count <= 1 {
            return Ok(());
        }

        let mut payloads = Vec::with_capacity(full_chunk_count);
        for (chunk_index, chunk) in chunks.iter().take(full_chunk_count).enumerate() {
            let expected_coord = u64::try_from(chunk_index)
                .ok()
                .and_then(|index| index.checked_mul(chunk_ctx.chunk_dims[0]))
                .ok_or_else(|| {
                    Error::InvalidFormat("v2-B-tree chunk coordinate overflow".into())
                })?;
            if handled.get(chunk_index).copied().unwrap_or(false)
                || chunk.filter_mask != 0
                || chunk.coords.len() != 1
                || chunk.coords[0] != expected_coord
                || crate::io::reader::is_undef_addr(chunk.addr)
            {
                return Ok(());
            }
            let mut payload = vec![0u8; chunk.read_size];
            reader.seek(chunk.addr)?;
            reader.read_bytes_into(&mut payload).map_err(|err| {
                Error::InvalidFormat(format!(
                    "failed to read v2-B-tree chunk {chunk_index} at address {} with size {}: {err}",
                    chunk.addr, chunk.nbytes
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
    use crate::format::messages::data_layout::{ChunkIndexType, DataLayoutMessage, LayoutClass};
    use crate::format::messages::dataspace::{DataspaceMessage, DataspaceType};
    use crate::format::messages::datatype::{DatatypeClass, DatatypeMessage};
    use crate::format::messages::filter_pipeline::{
        FilterDesc, FilterPipelineMessage, FILTER_DEFLATE,
    };
    use std::io::Cursor;

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

    fn deflate_info_and_context<'a>(
        data_dims: &'a [u64],
        chunk_dims: &'a [u64],
        chunk_bytes: usize,
        total_bytes: usize,
    ) -> (DatasetInfo, ChunkReadContext<'a>) {
        let info = DatasetInfo {
            dataspace: DataspaceMessage {
                version: 2,
                space_type: DataspaceType::Simple,
                ndims: u8::try_from(data_dims.len()).unwrap(),
                dims: data_dims.to_vec(),
                max_dims: None,
            },
            datatype: DatatypeMessage {
                version: 4,
                class: DatatypeClass::FixedPoint,
                class_bits: [0, 0, 0],
                size: 4,
                properties: vec![0; 4],
            },
            layout: DataLayoutMessage {
                version: 4,
                layout_class: LayoutClass::Chunked,
                compact_data: None,
                contiguous_addr: None,
                contiguous_size: None,
                chunk_dims: Some(chunk_dims.to_vec()),
                chunk_index_addr: Some(0),
                chunk_index_type: Some(ChunkIndexType::BTreeV2),
                chunk_element_size: None,
                chunk_flags: None,
                chunk_encoded_dims: Some(chunk_dims.to_vec()),
                single_chunk_filtered_size: None,
                single_chunk_filter_mask: None,
                data_addr: None,
                virtual_heap_addr: None,
                virtual_heap_index: None,
            },
            filter_pipeline: Some(FilterPipelineMessage {
                version: 2,
                filters: vec![FilterDesc {
                    id: FILTER_DEFLATE,
                    name: None,
                    flags: 0,
                    client_data: vec![4],
                }],
            }),
            fill_value: None,
            external_file_list: None,
        };
        let chunk_ctx = ChunkReadContext {
            idx_addr: 0,
            data_dims,
            chunk_dims,
            chunk_bytes,
            element_size: 4,
            total_bytes,
        };
        (info, chunk_ctx)
    }

    fn synthetic_btree_v2_deflate_payloads(
        chunk_elems: usize,
        full_chunk_count: usize,
    ) -> (Vec<u8>, Vec<DecodedBTreeV2Chunk>, Vec<u8>) {
        let mut image = Vec::new();
        let mut chunks = Vec::new();
        let mut expected = Vec::new();

        for chunk_index in 0..full_chunk_count {
            let start = chunk_index * chunk_elems;
            let mut raw = Vec::with_capacity(chunk_elems * 4);
            for value in start..start + chunk_elems {
                let value = value as i32 * 13 - 23;
                raw.extend_from_slice(&value.to_le_bytes());
            }
            expected.extend_from_slice(&raw);

            let addr = image.len() as u64;
            let mut compressed = Vec::new();
            crate::filters::deflate::compress_into(&raw, 4, &mut compressed).unwrap();
            let read_size = compressed.len();
            image.extend_from_slice(&compressed);

            chunks.push(DecodedBTreeV2Chunk {
                coords: vec![(start as u64)],
                addr,
                nbytes: read_size as u64,
                read_size,
                filter_mask: 0,
            });
        }

        (image, chunks, expected)
    }

    #[test]
    fn parallel_deflate_btree_v2_helper_decodes_full_prefix() {
        const CHUNK_ELEMS: usize = 2048;
        const FULL_CHUNKS: usize = 8;
        const TAIL_ELEMS: usize = 17;
        let data_dims = [(CHUNK_ELEMS * FULL_CHUNKS + TAIL_ELEMS) as u64];
        let chunk_dims = [CHUNK_ELEMS as u64];
        let chunk_bytes = CHUNK_ELEMS * 4;
        let total_bytes = (CHUNK_ELEMS * FULL_CHUNKS + TAIL_ELEMS) * 4;
        let (info, chunk_ctx) =
            deflate_info_and_context(&data_dims, &chunk_dims, chunk_bytes, total_bytes);
        let (image, chunks, expected_prefix) =
            synthetic_btree_v2_deflate_payloads(CHUNK_ELEMS, FULL_CHUNKS);
        let mut reader = HdfReader::new(Cursor::new(image));
        let mut output = vec![0u8; total_bytes];
        let mut handled = vec![false; chunks.len()];

        super::super::support::set_parallel_deflate_worker_override(2);
        super::super::support::reset_parallel_deflate_chunks_handled();
        Dataset::try_read_parallel_deflate_btree_v2_chunks_1d(
            &mut reader,
            &info,
            &chunk_ctx,
            &chunks,
            &mut output,
            &mut handled,
        )
        .unwrap();

        assert_eq!(
            super::super::support::parallel_deflate_chunks_handled(),
            FULL_CHUNKS
        );
        assert!(handled.iter().all(|handled| *handled));
        assert_eq!(&output[..expected_prefix.len()], expected_prefix);
        assert!(output[expected_prefix.len()..]
            .iter()
            .all(|byte| *byte == 0));
        super::super::support::set_parallel_deflate_worker_override(0);
    }

    #[test]
    fn parallel_deflate_btree_v2_helper_skips_masked_chunk() {
        const CHUNK_ELEMS: usize = 2048;
        const FULL_CHUNKS: usize = 8;
        let data_dims = [(CHUNK_ELEMS * FULL_CHUNKS) as u64];
        let chunk_dims = [CHUNK_ELEMS as u64];
        let chunk_bytes = CHUNK_ELEMS * 4;
        let total_bytes = CHUNK_ELEMS * FULL_CHUNKS * 4;
        let (info, chunk_ctx) =
            deflate_info_and_context(&data_dims, &chunk_dims, chunk_bytes, total_bytes);
        let (image, mut chunks, _) = synthetic_btree_v2_deflate_payloads(CHUNK_ELEMS, FULL_CHUNKS);
        chunks[3].filter_mask = 1;
        let mut reader = HdfReader::new(Cursor::new(image));
        let mut output = vec![0u8; total_bytes];
        let mut handled = vec![false; chunks.len()];

        super::super::support::set_parallel_deflate_worker_override(2);
        super::super::support::reset_parallel_deflate_chunks_handled();
        Dataset::try_read_parallel_deflate_btree_v2_chunks_1d(
            &mut reader,
            &info,
            &chunk_ctx,
            &chunks,
            &mut output,
            &mut handled,
        )
        .unwrap();

        assert_eq!(super::super::support::parallel_deflate_chunks_handled(), 0);
        assert!(handled.iter().all(|handled| !*handled));
        assert!(output.iter().all(|byte| *byte == 0));
        super::super::support::set_parallel_deflate_worker_override(0);
    }
}
