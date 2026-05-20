use std::io::{Read, Seek};
use std::thread;

use crate::error::{Error, Result};
use crate::filters;
use crate::io::reader::HdfReader;

use super::chunk_read::{BorrowedChunkPayloadRead, ChunkBTreeRecord, ChunkReadContext};
use super::{usize_from_u64, Dataset, DatasetInfo};

const MAX_CHUNK_BTREE_V1_RECURSION: usize = 64;
const MIN_PARALLEL_DEFLATE_CHUNKS_1D: usize = 8;
const MIN_PARALLEL_DEFLATE_BYTES_1D: usize = 64 * 1024;

/// Output of `decode_chunk_btree_node`: either a leaf node's chunk
/// records (coords, addr, size, filter_mask) or an internal node's
/// child-pointer addresses. Mirrors the structural distinction libhdf5
/// makes via the `H5B_t` `level == 0` test.
enum ChunkBTreeNode {
    Leaf {
        level: u8,
        records: Vec<ChunkBTreeRecord>,
    },
    Internal {
        level: u8,
        child_addrs: Vec<u64>,
    },
}

impl Dataset {
    pub(super) fn read_chunked_btree_v1<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        info: &DatasetInfo,
        chunk_ctx: &ChunkReadContext<'_>,
    ) -> Result<Vec<u8>> {
        let mut output = Self::scratch_output(chunk_ctx.total_bytes);
        Self::read_chunked_btree_v1_into(reader, info, chunk_ctx, &mut output)?;
        Ok(output)
    }

    pub(super) fn read_chunked_btree_v1_into<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        info: &DatasetInfo,
        chunk_ctx: &ChunkReadContext<'_>,
        output: &mut [u8],
    ) -> Result<()> {
        if output.len() != chunk_ctx.total_bytes {
            return Err(Error::InvalidFormat(format!(
                "v1 B-tree chunk output buffer has {} bytes, expected {}",
                output.len(),
                chunk_ctx.total_bytes
            )));
        }
        let ndims = chunk_ctx.data_dims.len();
        let chunk_records = Self::collect_btree_v1_chunks(reader, chunk_ctx.idx_addr, ndims)?;
        if !Self::has_full_chunk_coverage_1d(&chunk_records, chunk_ctx)? {
            Self::filled_data_into(
                chunk_ctx.total_bytes / chunk_ctx.element_size,
                chunk_ctx.element_size,
                info,
                output,
            )?;
        }
        let mut compressed_scratch = Vec::new();
        let mut raw_scratch = Vec::new();
        let mut scaled_scratch = Vec::new();
        let mut handled = if Self::btree_v1_uses_unfiltered_coalescing(info) {
            Self::try_read_coalesced_borrowed_unfiltered_chunks_1d(
                reader,
                info,
                chunk_ctx,
                chunk_records.len(),
                chunk_records.iter().map(|record| {
                    Ok(BorrowedChunkPayloadRead {
                        coords: &record.coords,
                        addr: record.chunk_addr,
                        read_size: usize_from_u64(record.chunk_size, "v1 B-tree chunk size")?,
                        filter_mask: record.filter_mask,
                    })
                }),
                output,
            )?
        } else {
            Vec::new()
        };
        if handled.is_empty() {
            handled.resize(chunk_records.len(), false);
        }
        Self::try_read_parallel_deflate_btree_v1_chunks_1d(
            reader,
            info,
            chunk_ctx,
            &chunk_records,
            output,
            &mut handled,
        )?;

        for (chunk_index, chunk_record) in chunk_records.iter().enumerate() {
            if handled.get(chunk_index).copied().unwrap_or(false) {
                continue;
            }
            if Self::try_read_full_chunk_1d_into_output(
                reader,
                info,
                chunk_ctx,
                &chunk_record.coords,
                chunk_record.chunk_addr,
                usize_from_u64(chunk_record.chunk_size, "v1 B-tree chunk size")?,
                chunk_record.filter_mask,
                output,
                &mut compressed_scratch,
            )? {
                continue;
            }
            Self::process_btree_v1_chunk_record(
                reader,
                chunk_ctx.idx_addr,
                chunk_record,
                info,
                chunk_ctx.data_dims,
                chunk_ctx.chunk_dims,
                chunk_ctx.chunk_bytes,
                chunk_ctx.element_size,
                output,
                &mut raw_scratch,
                &mut scaled_scratch,
            )?;
        }

        Ok(())
    }

    fn btree_v1_uses_unfiltered_coalescing(info: &DatasetInfo) -> bool {
        info.filter_pipeline
            .as_ref()
            .map(|pipeline| pipeline.filters.is_empty())
            .unwrap_or(true)
    }

    fn try_read_parallel_deflate_btree_v1_chunks_1d<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        info: &DatasetInfo,
        chunk_ctx: &ChunkReadContext<'_>,
        chunk_records: &[ChunkBTreeRecord],
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
        if full_chunk_count < 2 || chunk_records.len() < full_chunk_count {
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
        if full_output_len > output.len() {
            return Ok(());
        }
        if full_chunk_count < MIN_PARALLEL_DEFLATE_CHUNKS_1D
            || full_output_len < MIN_PARALLEL_DEFLATE_BYTES_1D
        {
            return Ok(());
        }

        let worker_count = thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
            .min(full_chunk_count);
        if worker_count <= 1 {
            return Ok(());
        }

        let mut payloads = Vec::with_capacity(full_chunk_count);
        for (chunk_index, record) in chunk_records.iter().take(full_chunk_count).enumerate() {
            if handled.get(chunk_index).copied().unwrap_or(false) {
                return Ok(());
            }
            if record.filter_mask != 0
                || record.coords.len() != 1
                || crate::io::reader::is_undef_addr(record.chunk_addr)
            {
                return Ok(());
            }
            let expected_start = chunk_index
                .checked_mul(chunk_size)
                .ok_or_else(|| Error::InvalidFormat("chunk coordinate overflow".into()))?;
            if usize_from_u64(record.coords[0], "chunk coordinate")? != expected_start {
                return Ok(());
            }

            let read_size = usize_from_u64(record.chunk_size, "v1 B-tree chunk size")?;
            let mut payload = vec![0u8; read_size];
            reader.seek(record.chunk_addr)?;
            reader.read_bytes_into(&mut payload)?;
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

        for flag in handled.iter_mut().take(full_chunk_count) {
            *flag = true;
        }
        Ok(())
    }

    #[cfg(feature = "tracehash")]
    fn trace_btree1_chunk_lookup(
        index_addr: u64,
        scaled: &[u64],
        addr: u64,
        nbytes: u64,
        filter_mask: u32,
    ) {
        let mut th = tracehash::th_call!("hdf5.chunk_index.btree1.lookup");
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
    fn trace_btree1_chunk_lookup(
        _index_addr: u64,
        _scaled: &[u64],
        _addr: u64,
        _nbytes: u64,
        _filter_mask: u32,
    ) {
    }

    fn scaled_chunk_coords_into(
        coords: &[u64],
        chunk_dims: &[u64],
        out: &mut Vec<u64>,
    ) -> Result<()> {
        if coords.len() != chunk_dims.len() {
            return Err(Error::InvalidFormat(
                "chunk coordinate rank does not match chunk dimensions".into(),
            ));
        }
        out.clear();
        out.reserve(coords.len());
        for (&coord, &dim) in coords.iter().zip(chunk_dims) {
            if dim == 0 {
                return Err(Error::InvalidFormat("chunk dimension is zero".into()));
            }
            if coord % dim != 0 {
                return Err(Error::InvalidFormat(
                    "chunk coordinate is not aligned to chunk dimension".into(),
                ));
            }
            out.push(coord / dim);
        }
        Ok(())
    }

    /// Pure deserializer for one v1 chunk-index B-tree node — returns
    /// either the leaf chunk records or the list of child addresses,
    /// depending on the node level. Mirrors libhdf5's
    /// `H5B__cache_deserialize` for the chunk-index node type. No I/O
    /// after the read; no recursion.
    fn decode_chunk_btree_node<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        addr: u64,
        ndims: usize,
    ) -> Result<ChunkBTreeNode> {
        reader.seek(addr)?;

        let mut magic = [0u8; 4];
        reader.read_bytes_into(&mut magic)?;
        if magic != [b'T', b'R', b'E', b'E'] {
            return Err(Error::InvalidFormat("invalid chunk B-tree magic".into()));
        }

        let node_type = reader.read_u8()?;
        if node_type != 1 {
            return Err(Error::InvalidFormat(format!(
                "expected raw data B-tree (type 1), got type {node_type}"
            )));
        }

        let level = reader.read_u8()?;
        let entries_used = usize::from(reader.read_u16()?);
        let _left_sibling = reader.read_addr()?;
        let _right_sibling = reader.read_addr()?;

        if level == 0 {
            let mut records = Vec::with_capacity(entries_used);
            for _ in 0..entries_used {
                records.push(Self::decode_chunk_btree_leaf_record(reader, ndims)?);
            }
            Self::skip_chunk_btree_final_key(reader, ndims)?;
            Ok(ChunkBTreeNode::Leaf { level, records })
        } else {
            let mut child_addrs = Vec::with_capacity(entries_used);
            for _ in 0..entries_used {
                child_addrs.push(Self::decode_chunk_btree_child_addr(reader, ndims)?);
            }
            Self::skip_chunk_btree_final_key(reader, ndims)?;
            Ok(ChunkBTreeNode::Internal { level, child_addrs })
        }
    }

    /// Walk a v1 chunk-index B-tree, depth-first. Mirrors libhdf5's
    /// `H5D__btree_idx_iterate` / `H5B__iterate_helper`: the actual
    /// node decoding lives in `decode_chunk_btree_node`.
    pub(super) fn collect_btree_v1_chunks<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        addr: u64,
        ndims: usize,
    ) -> Result<Vec<ChunkBTreeRecord>> {
        let mut visited = Vec::new();
        Self::collect_btree_v1_chunks_inner(reader, addr, ndims, 0, None, &mut visited)
    }

    fn collect_btree_v1_chunks_inner<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        addr: u64,
        ndims: usize,
        depth: usize,
        expected_level: Option<u8>,
        visited: &mut Vec<u64>,
    ) -> Result<Vec<ChunkBTreeRecord>> {
        if depth > MAX_CHUNK_BTREE_V1_RECURSION {
            return Err(Error::InvalidFormat(
                "v1 chunk B-tree recursion depth exceeded".into(),
            ));
        }
        if crate::io::reader::is_undef_addr(addr) {
            return Err(Error::InvalidFormat(
                "v1 chunk B-tree node address is undefined".into(),
            ));
        }
        if visited.contains(&addr) {
            return Err(Error::InvalidFormat(
                "v1 chunk B-tree traversal cycle detected".into(),
            ));
        }
        visited.push(addr);

        let node = Self::decode_chunk_btree_node(reader, addr, ndims)?;
        let node_level = match &node {
            ChunkBTreeNode::Leaf { level, .. } | ChunkBTreeNode::Internal { level, .. } => *level,
        };
        if let Some(expected_level) = expected_level {
            if node_level != expected_level {
                visited.pop();
                return Err(Error::InvalidFormat(format!(
                    "v1 chunk B-tree child level {node_level} does not match expected {expected_level}"
                )));
            }
        }
        let result = match node {
            ChunkBTreeNode::Leaf { records, .. } => Ok(records),
            ChunkBTreeNode::Internal { level, child_addrs } => {
                let mut all_chunks = Vec::new();
                for child_addr in child_addrs {
                    let mut child_chunks = Self::collect_btree_v1_chunks_inner(
                        reader,
                        child_addr,
                        ndims,
                        depth.checked_add(1).ok_or_else(|| {
                            Error::InvalidFormat("v1 chunk B-tree recursion depth overflow".into())
                        })?,
                        Some(level - 1),
                        visited,
                    )?;
                    all_chunks.append(&mut child_chunks);
                }
                Ok(all_chunks)
            }
        };

        visited.pop();
        result
    }

    fn decode_chunk_btree_leaf_record<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        ndims: usize,
    ) -> Result<ChunkBTreeRecord> {
        let chunk_size = u64::from(reader.read_u32()?);
        let filter_mask = reader.read_u32()?;
        let mut coords = Vec::with_capacity(ndims);
        for _ in 0..ndims {
            coords.push(reader.read_u64()?);
        }
        let _extra = reader.read_u64()?;
        let chunk_addr = reader.read_addr()?;
        Ok(ChunkBTreeRecord {
            coords,
            chunk_addr,
            chunk_size,
            filter_mask,
        })
    }

    fn decode_chunk_btree_child_addr<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        ndims: usize,
    ) -> Result<u64> {
        let _chunk_size = reader.read_u32()?;
        let _filter_mask = reader.read_u32()?;
        for _ in 0..=ndims {
            let _ = reader.read_u64()?;
        }
        let child_addr = reader.read_addr()?;
        if crate::io::reader::is_undef_addr(child_addr) {
            return Err(Error::InvalidFormat(
                "v1 chunk B-tree child address is undefined".into(),
            ));
        }
        Ok(child_addr)
    }

    fn skip_chunk_btree_final_key<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        ndims: usize,
    ) -> Result<()> {
        let _final_chunk_size = reader.read_u32()?;
        let _final_filter_mask = reader.read_u32()?;
        for _ in 0..=ndims {
            let _ = reader.read_u64()?;
        }
        Ok(())
    }

    fn process_btree_v1_chunk_record<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        btree_addr: u64,
        chunk_record: &ChunkBTreeRecord,
        info: &DatasetInfo,
        data_dims: &[u64],
        chunk_dims: &[u64],
        chunk_bytes: usize,
        element_size: usize,
        output: &mut [u8],
        raw_scratch: &mut Vec<u8>,
        scaled_scratch: &mut Vec<u64>,
    ) -> Result<()> {
        Self::scaled_chunk_coords_into(&chunk_record.coords, chunk_dims, scaled_scratch)?;
        Self::trace_btree1_chunk_lookup(
            btree_addr,
            scaled_scratch,
            chunk_record.chunk_addr,
            chunk_record.chunk_size,
            chunk_record.filter_mask,
        );

        if crate::io::reader::is_undef_addr(chunk_record.chunk_addr) {
            return Ok(());
        }

        let raw = Self::read_btree_v1_chunk_payload_into_scratch(
            reader,
            chunk_record,
            info,
            chunk_bytes,
            element_size,
            raw_scratch,
        )?;
        Self::copy_chunk_to_output(
            raw,
            &chunk_record.coords,
            data_dims,
            chunk_dims,
            element_size,
            output,
        )
    }

    fn read_btree_v1_chunk_payload_into_scratch<'a, R: Read + Seek>(
        reader: &mut HdfReader<R>,
        chunk_record: &ChunkBTreeRecord,
        info: &DatasetInfo,
        chunk_bytes: usize,
        element_size: usize,
        raw_scratch: &'a mut Vec<u8>,
    ) -> Result<&'a [u8]> {
        reader.seek(chunk_record.chunk_addr)?;
        let read_size = usize_from_u64(chunk_record.chunk_size, "v1 B-tree chunk size")?;
        raw_scratch.resize(read_size, 0);
        reader.read_bytes_into(raw_scratch)?;

        if let Some(ref pipeline) = info.filter_pipeline {
            if !pipeline.filters.is_empty() {
                *raw_scratch = filters::apply_pipeline_reverse_with_mask_expected(
                    raw_scratch,
                    pipeline,
                    element_size,
                    chunk_record.filter_mask,
                    chunk_bytes,
                )?;
            }
        }

        Ok(raw_scratch)
    }
}
