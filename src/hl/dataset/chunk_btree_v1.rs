use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::filters;
use crate::io::reader::HdfReader;

use super::chunk_read::{ChunkBTreeRecord, ChunkReadContext};
use super::{usize_from_u64, Dataset, DatasetInfo};

const MAX_CHUNK_BTREE_V1_RECURSION: usize = 64;

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
        let ndims = chunk_ctx.data_dims.len();
        let chunk_records = Self::collect_btree_v1_chunks(reader, chunk_ctx.idx_addr, ndims)?;
        let mut output = if Self::has_full_chunk_coverage_1d(&chunk_records, chunk_ctx)? {
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

        for chunk_record in &chunk_records {
            if Self::try_read_full_chunk_1d_into_output(
                reader,
                info,
                chunk_ctx,
                &chunk_record.coords,
                chunk_record.chunk_addr,
                usize_from_u64(chunk_record.chunk_size, "v1 B-tree chunk size")?,
                chunk_record.filter_mask,
                &mut output,
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
                &mut output,
            )?;
        }

        Ok(output)
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

    fn scaled_chunk_coords(coords: &[u64], chunk_dims: &[u64]) -> Result<Vec<u64>> {
        if coords.len() != chunk_dims.len() {
            return Err(Error::InvalidFormat(
                "chunk coordinate rank does not match chunk dimensions".into(),
            ));
        }
        coords
            .iter()
            .zip(chunk_dims)
            .map(|(&coord, &dim)| {
                if dim == 0 {
                    return Err(Error::InvalidFormat("chunk dimension is zero".into()));
                }
                if coord % dim != 0 {
                    return Err(Error::InvalidFormat(
                        "chunk coordinate is not aligned to chunk dimension".into(),
                    ));
                }
                Ok(coord / dim)
            })
            .collect()
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

        let magic = reader.read_bytes(4)?;
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
    ) -> Result<()> {
        let scaled = Self::scaled_chunk_coords(&chunk_record.coords, chunk_dims)?;
        Self::trace_btree1_chunk_lookup(
            btree_addr,
            &scaled,
            chunk_record.chunk_addr,
            chunk_record.chunk_size,
            chunk_record.filter_mask,
        );

        if crate::io::reader::is_undef_addr(chunk_record.chunk_addr) {
            return Ok(());
        }

        let raw = Self::read_btree_v1_chunk_payload(
            reader,
            chunk_record,
            info,
            chunk_bytes,
            element_size,
        )?;
        Self::copy_chunk_to_output(
            &raw,
            &chunk_record.coords,
            data_dims,
            chunk_dims,
            element_size,
            output,
        )
    }

    fn read_btree_v1_chunk_payload<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        chunk_record: &ChunkBTreeRecord,
        info: &DatasetInfo,
        chunk_bytes: usize,
        element_size: usize,
    ) -> Result<Vec<u8>> {
        reader.seek(chunk_record.chunk_addr)?;
        let read_size = usize_from_u64(chunk_record.chunk_size, "v1 B-tree chunk size")?;
        let mut raw = reader.read_bytes(read_size)?;

        if let Some(ref pipeline) = info.filter_pipeline {
            if !pipeline.filters.is_empty() {
                raw = filters::apply_pipeline_reverse_with_mask_expected(
                    &raw,
                    pipeline,
                    element_size,
                    chunk_record.filter_mask,
                    chunk_bytes,
                )?;
            }
        }

        Ok(raw)
    }
}
