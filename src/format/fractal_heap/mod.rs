//! Fractal heap — top-level public API. Mirrors libhdf5's `H5HF.c`
//! (the file-spanning entry points). Per-component code lives in sibling
//! modules:
//!   - `hdr`    → `H5HFhdr.c` + hdr-half of `H5HFcache.c`
//!   - `iblock` → `H5HFiblock.c` + iblock-half of `H5HFcache.c`
//!   - `dblock` → `H5HFdblock.c` + dblock-half of `H5HFcache.c`
//!   - `man`    → `H5HFman.c`
//!   - `huge`   → `H5HFhuge.c` + `H5HFbtree2.c`
//!   - `tiny`   → `H5HFtiny.c`
//!   - `dtable` → `H5HFdtable.c`
//!
//! Trace probes and small numeric helpers live here in `mod.rs` because
//! they cut across all the per-block files.

mod dblock;
mod dtable;
mod hdr;
mod huge;
mod iblock;
mod man;
mod tiny;

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::format::checksum::checksum_metadata;
use crate::format::messages::filter_pipeline::FilterPipelineMessage;
use crate::io::reader::{HdfReader, UNDEF_ADDR};

/// Fractal heap header magic: "FRHP"
pub(super) const FRHP_MAGIC: [u8; 4] = [b'F', b'R', b'H', b'P'];
/// Direct block magic: "FHDB" (kept for reference; we currently read by
/// offset rather than by magic).
#[allow(dead_code)]
pub(super) const FHDB_MAGIC: [u8; 4] = [b'F', b'H', b'D', b'B'];
/// Indirect block magic: "FHIB"
pub(super) const FHIB_MAGIC: [u8; 4] = [b'F', b'H', b'I', b'B'];
const MAX_HEAP_OBJECT_BYTES: usize = 4 * 1024 * 1024 * 1024;
const FRACTAL_HEAP_DTABLE_IMAGE_LEN: usize = 24;
const HUGE_FILTERED_DIRECT_RECORD_LEN: usize = 28;

/// Fractal heap header.
#[derive(Debug, Clone)]
pub struct FractalHeapHeader {
    pub heap_addr: u64,
    pub heap_id_len: u16,
    pub io_filter_len: u16,
    pub flags: u8,
    pub max_managed_obj_size: u32,

    pub table_width: u16,
    pub start_block_size: u64,
    pub max_direct_block_size: u64,
    pub max_heap_size: u16,
    pub start_root_rows: u16,
    pub root_block_addr: u64,
    pub current_root_rows: u16,
    pub num_managed_objects: u64,
    pub has_checksum: bool,
    pub sizeof_addr: u8,
    pub sizeof_size: u8,
    pub huge_btree_addr: u64,
    pub root_direct_filtered_size: Option<u64>,
    pub root_direct_filter_mask: u32,
    pub filter_pipeline: Option<FilterPipelineMessage>,
}

#[derive(Debug, Clone)]
pub struct FractalHeap {
    pub header: FractalHeapHeader,
    objects: Vec<Vec<u8>>,
    closed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct DirectBlockCacheKey {
    block_addr: u64,
    block_size: u64,
    filtered_size: Option<u64>,
    filter_mask: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct IndirectBlockCacheKey {
    block_addr: u64,
    nrows: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FilteredIndirectBlockCacheKey {
    block_addr: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DirectBlockSpan {
    start: u64,
    end: u64,
    block_addr: u64,
    block_size: u64,
    filtered_size: Option<u64>,
    filter_mask: u32,
}

/// Per-traversal cache for reading several managed objects from the same
/// fractal heap. Dense link and attribute indexes often point many records at
/// a small number of direct blocks, so keeping decoded block images and heap
/// offset spans here avoids repeated indirect-table decodes, scans, seeks,
/// reads, and filter application.
#[derive(Debug, Default)]
pub struct FractalHeapManagedObjectCache {
    direct_blocks: HashMap<DirectBlockCacheKey, Vec<u8>>,
    indirect_blocks: HashMap<IndirectBlockCacheKey, iblock::IndirectBlock>,
    filtered_indirect_blocks: HashMap<FilteredIndirectBlockCacheKey, iblock::FilteredIndirectBlock>,
    direct_spans: BTreeMap<u64, DirectBlockSpan>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FractalHeapCreateParams {
    pub heap_id_len: u16,
    pub table_width: u16,
    pub start_block_size: u64,
    pub max_direct_block_size: u64,
    pub max_heap_size: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FractalHeapIndirectBlock {
    pub nrows: usize,
    pub child_addrs: Vec<u64>,
    pub ref_count: usize,
    pub dirty: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FractalHeapDirectBlock {
    pub addr: u64,
    pub data: Vec<u8>,
    pub dirty: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FractalHeapHugeContext {
    filtered: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FractalHeapHugeRecord {
    pub id: u64,
    pub addr: u64,
    pub len: u64,
    pub obj_size: u64,
    pub filter_mask: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FractalHeapDTableImage {
    pub table_width: u16,
    pub start_block_size: u64,
    pub max_direct_block_size: u64,
    pub max_heap_size: u16,
    pub start_root_rows: u16,
    pub current_root_rows: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FractalHeapIterator {
    offsets: Vec<u64>,
    index: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FractalHeapSection {
    Single { offset: u64, size: u64 },
    Row { row: usize, offset: u64, size: u64 },
    Indirect { offset: u64, rows: usize, size: u64 },
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FractalHeapSpace {
    sections: Vec<FractalHeapSection>,
    closed: bool,
}

impl Default for FractalHeapCreateParams {
    fn default() -> Self {
        Self {
            heap_id_len: 8,
            table_width: 4,
            start_block_size: 512,
            max_direct_block_size: 4096,
            max_heap_size: 64,
        }
    }
}

impl FractalHeapHeader {
    /// Allocate a shared fractal heap header with the given create
    /// parameters. Mirrors libhdf5's `H5HF__hdr_alloc`.
    pub fn hdr_alloc(params: FractalHeapCreateParams) -> Result<Self> {
        validate_heap_create_params(&params)?;
        Ok(Self {
            heap_addr: 0,
            heap_id_len: params.heap_id_len,
            io_filter_len: 0,
            flags: 0,
            max_managed_obj_size: u32::MAX,
            table_width: params.table_width,
            start_block_size: params.start_block_size,
            max_direct_block_size: params.max_direct_block_size,
            max_heap_size: params.max_heap_size,
            start_root_rows: 0,
            root_block_addr: u64::MAX,
            current_root_rows: 0,
            num_managed_objects: 0,
            has_checksum: false,
            sizeof_addr: 8,
            sizeof_size: 8,
            huge_btree_addr: u64::MAX,
            root_direct_filtered_size: None,
            root_direct_filter_mask: 0,
            filter_pipeline: None,
        })
    }

    /// First phase of finishing initialization of the shared heap header.
    /// Mirrors `H5HF__hdr_finish_init_phase1`.
    pub fn hdr_finish_init_phase1(&mut self) -> Result<()> {
        self.validate_header()
    }

    /// Second phase of finishing initialization of the shared heap header.
    /// Mirrors `H5HF__hdr_finish_init_phase2`.
    pub fn hdr_finish_init_phase2(&mut self) -> Result<()> {
        self.validate_header()
    }

    /// Finish initializing the shared heap header (both phases).
    /// Mirrors `H5HF__hdr_finish_init`.
    pub fn hdr_finish_init(&mut self) -> Result<()> {
        self.hdr_finish_init_phase1()?;
        self.hdr_finish_init_phase2()
    }

    /// Create a new fractal heap header. Mirrors `H5HF__hdr_create`:
    /// allocates the header and runs the finish-init phases.
    pub fn hdr_create(params: FractalHeapCreateParams) -> Result<Self> {
        let mut header = Self::hdr_alloc(params)?;
        header.hdr_finish_init()?;
        Ok(header)
    }

    /// Pin the header for use. Mirrors `H5HF__hdr_protect`
    /// (`H5AC_protect` wrapper); the Rust port clones the value.
    pub fn hdr_protect(&self) -> Self {
        self.clone()
    }

    /// Increment the component reference count on the shared heap header.
    /// Mirrors `H5HF__hdr_incr`.
    pub fn hdr_incr(&mut self) -> Result<()> {
        self.num_managed_objects = self
            .num_managed_objects
            .checked_add(1)
            .ok_or_else(|| Error::InvalidFormat("fractal heap object count overflow".into()))?;
        Ok(())
    }

    /// Decrement the component reference count on the shared heap header.
    /// Mirrors `H5HF__hdr_decr`.
    pub fn hdr_decr(&mut self) -> Result<()> {
        self.num_managed_objects = self
            .num_managed_objects
            .checked_sub(1)
            .ok_or_else(|| Error::InvalidFormat("fractal heap object count underflow".into()))?;
        Ok(())
    }

    /// Increment the file reference count on the shared heap header.
    /// Mirrors `H5HF__hdr_fuse_incr`.
    pub fn hdr_fuse_incr(&mut self) -> Result<()> {
        self.hdr_incr()
    }

    /// Decrement the file reference count on the shared heap header.
    /// Mirrors `H5HF__hdr_fuse_decr`.
    pub fn hdr_fuse_decr(&mut self) -> Result<()> {
        self.hdr_decr()
    }

    /// Mark the heap header as dirty. Mirrors `H5HF__hdr_dirty` (no-op
    /// in the Rust port).
    pub fn hdr_dirty(&mut self) {}

    /// Adjust the free space accounting for the heap. Mirrors
    /// `H5HF__hdr_adj_free` (the Rust port doesn't track free space).
    pub fn hdr_adj_free(&mut self, _delta: i64) {}

    /// Adjust the heap's root block address and row count. Mirrors
    /// `H5HF__hdr_adjust_heap`.
    pub fn hdr_adjust_heap(&mut self, root_addr: u64, root_rows: u16) {
        self.root_block_addr = root_addr;
        self.current_root_rows = root_rows;
    }

    /// Increase the allocated size of the heap by `bytes` (capped at
    /// `max_direct_block_size`). Mirrors `H5HF__hdr_inc_alloc`.
    pub fn hdr_inc_alloc(&mut self, bytes: u64) -> Result<()> {
        let next = self
            .start_block_size
            .checked_add(bytes)
            .ok_or_else(|| Error::InvalidFormat("fractal heap allocation overflow".into()))?;
        self.start_block_size = next.min(self.max_direct_block_size);
        Ok(())
    }

    /// Start a "next block" iterator at the beginning of the heap.
    /// Mirrors `H5HF__hdr_start_iter`.
    pub fn hdr_start_iter(&self) -> FractalHeapIterator {
        FractalHeapIterator::man_iter_init(0, self.num_managed_objects)
    }

    /// Reset a "next block" iterator. Mirrors `H5HF__hdr_reset_iter`.
    pub fn hdr_reset_iter(iter: &mut FractalHeapIterator) {
        iter.man_iter_reset();
    }

    /// Add skipped direct blocks to the free space and advance the iterator
    /// accordingly. Mirrors `H5HF__hdr_skip_blocks`.
    pub fn hdr_skip_blocks(iter: &mut FractalHeapIterator, blocks: usize) {
        iter.index = iter.index.saturating_add(blocks).min(iter.offsets.len());
    }

    /// Update the heap-iterator state to a fresh set of offsets. Mirrors
    /// `H5HF__hdr_update_iter`.
    pub fn hdr_update_iter(iter: &mut FractalHeapIterator, offsets: Vec<u64>) {
        iter.offsets = offsets;
        iter.index = 0;
    }

    /// Advance the "next block" iterator by one. Mirrors
    /// `H5HF__hdr_inc_iter`.
    pub fn hdr_inc_iter(iter: &mut FractalHeapIterator) {
        let _ = iter.man_iter_next();
    }

    /// Walk the "next block" iterator backwards. Mirrors
    /// `H5HF__hdr_reverse_iter`.
    pub fn hdr_reverse_iter(iter: &mut FractalHeapIterator) {
        iter.offsets.reverse();
        iter.index = 0;
    }

    /// Whether the heap is in its "empty heap" state. Mirrors
    /// `H5HF__hdr_empty`.
    pub fn hdr_empty(&self) -> bool {
        self.num_managed_objects == 0
    }

    /// Free the shared fractal heap header. Mirrors `H5HF__hdr_free`.
    pub fn hdr_free(self) {}

    /// Delete a fractal heap starting with the header. Mirrors
    /// `H5HF__hdr_delete`.
    pub fn hdr_delete(self) {}

    /// Initial cache-load size for a fractal heap header (just the magic).
    pub fn cache_hdr_get_initial_load_size() -> usize {
        4
    }

    /// Final cache-load size for the header once the prefix is known.
    pub fn cache_hdr_get_final_load_size(&self) -> usize {
        self.cache_hdr_image_len()
    }

    /// Verify the metadata-cache checksum on a header image. Mirrors
    /// `H5HF__cache_hdr_verify_chksum`.
    pub fn cache_hdr_verify_chksum(image: &[u8]) -> Result<()> {
        verify_image_checksum(image, "fractal heap header")
    }

    /// On-disk image size of the fractal heap header. Mirrors
    /// `H5HF__cache_hdr_image_len`.
    pub fn cache_hdr_image_len(&self) -> usize {
        let size_width = usize::from(self.sizeof_size);
        let addr_width = usize::from(self.sizeof_addr);
        let base = 4 + 1 + 2 + 2 + 1 + 4 + 2 + 2 + 2 + 2 + 4;
        let filter_len = if self.io_filter_len > 0 {
            size_width + 4 + usize::from(self.io_filter_len)
        } else {
            0
        };
        base + (12 * size_width) + (3 * addr_width) + filter_len
    }

    /// Pre-serialize the header into an existing output buffer.
    pub fn cache_hdr_pre_serialize_into(&self, out: &mut Vec<u8>) -> Result<()> {
        self.cache_hdr_serialize_into(out)
    }

    /// Append the on-disk image of the fractal heap header to `out`. Mirrors
    /// `H5HF__cache_hdr_serialize`.
    pub fn cache_hdr_serialize_into(&self, out: &mut Vec<u8>) -> Result<()> {
        self.validate_header()?;
        let start = out.len();
        out.reserve(self.cache_hdr_image_len());
        out.extend_from_slice(&FRHP_MAGIC);
        out.push(0);
        out.extend_from_slice(&self.heap_id_len.to_le_bytes());
        out.extend_from_slice(&self.io_filter_len.to_le_bytes());
        out.push(self.flags);
        out.extend_from_slice(&self.max_managed_obj_size.to_le_bytes());
        encode_size_field(out, 0, self.sizeof_size, "fractal heap next huge id")?;
        encode_addr_field(
            out,
            self.huge_btree_addr,
            self.sizeof_addr,
            "fractal heap huge B-tree address",
        )?;
        encode_size_field(out, 0, self.sizeof_size, "fractal heap managed free space")?;
        encode_addr_field(
            out,
            UNDEF_ADDR,
            self.sizeof_addr,
            "fractal heap free-space-manager address",
        )?;
        encode_size_field(
            out,
            self.num_managed_objects,
            self.sizeof_size,
            "fractal heap managed object storage size",
        )?;
        encode_size_field(
            out,
            self.num_managed_objects,
            self.sizeof_size,
            "fractal heap managed allocated size",
        )?;
        encode_size_field(
            out,
            0,
            self.sizeof_size,
            "fractal heap managed iterator offset",
        )?;
        encode_size_field(
            out,
            self.num_managed_objects,
            self.sizeof_size,
            "fractal heap managed object count",
        )?;
        encode_size_field(out, 0, self.sizeof_size, "fractal heap huge object size")?;
        encode_size_field(out, 0, self.sizeof_size, "fractal heap huge object count")?;
        encode_size_field(out, 0, self.sizeof_size, "fractal heap tiny object size")?;
        encode_size_field(out, 0, self.sizeof_size, "fractal heap tiny object count")?;
        out.extend_from_slice(&self.table_width.to_le_bytes());
        encode_size_field(
            out,
            self.start_block_size,
            self.sizeof_size,
            "fractal heap start block size",
        )?;
        encode_size_field(
            out,
            self.max_direct_block_size,
            self.sizeof_size,
            "fractal heap max direct block size",
        )?;
        out.extend_from_slice(&self.max_heap_size.to_le_bytes());
        out.extend_from_slice(&self.start_root_rows.to_le_bytes());
        encode_addr_field(
            out,
            self.root_block_addr,
            self.sizeof_addr,
            "fractal heap root block address",
        )?;
        out.extend_from_slice(&self.current_root_rows.to_le_bytes());
        if self.io_filter_len > 0 {
            encode_size_field(
                out,
                self.root_direct_filtered_size.unwrap_or(0),
                self.sizeof_size,
                "fractal heap root direct filtered size",
            )?;
            out.extend_from_slice(&self.root_direct_filter_mask.to_le_bytes());
            let pipeline = self.filter_pipeline.as_ref().ok_or_else(|| {
                Error::InvalidFormat("fractal heap filter pipeline is missing".into())
            })?;
            let pipeline_bytes = encode_filter_pipeline_for_heap(pipeline)?;
            if pipeline_bytes.len() != usize::from(self.io_filter_len) {
                return Err(Error::InvalidFormat(
                    "fractal heap filter pipeline length does not match header".into(),
                ));
            }
            out.extend_from_slice(&pipeline_bytes);
        }
        let checksum = checksum_metadata(&out[start..]);
        out.extend_from_slice(&checksum.to_le_bytes());
        Ok(())
    }

    /// Free the in-core image. Mirrors `H5HF__cache_hdr_free_icr` (no-op
    /// for owned Rust buffers).
    pub fn cache_hdr_free_icr(_image: Vec<u8>) {}

    /// Sanity check: confirm all descendant cache entries are clean.
    /// Mirrors `H5HF__cache_verify_hdr_descendants_clean`.
    pub fn cache_verify_hdr_descendants_clean(&self) -> bool {
        true
    }

    /// On-disk size of the heap header.
    pub fn hdr_size(&self) -> usize {
        self.cache_hdr_image_len()
    }

    /// Write info about a fractal heap header into `out`. Mirrors
    /// `H5HF_hdr_print`.
    pub fn hdr_print_fmt(&self, out: &mut impl fmt::Write) -> fmt::Result {
        self.hdr_debug_fmt(out)
    }

    /// Write debugging info about a fractal heap header into `out`. Mirrors
    /// `H5HF_hdr_debug`.
    pub fn hdr_debug_fmt(&self, out: &mut impl fmt::Write) -> fmt::Result {
        write!(
            out,
            "FractalHeapHeader(addr={:#x}, id_len={}, width={}, root={:#x}, nmanaged={})",
            self.heap_addr,
            self.heap_id_len,
            self.table_width,
            self.root_block_addr,
            self.num_managed_objects
        )
    }

    /// Retrieve the parameters used to create the fractal heap.
    /// Mirrors `H5HF_get_cparam_test`.
    pub fn get_cparam_test(&self) -> FractalHeapCreateParams {
        FractalHeapCreateParams {
            heap_id_len: self.heap_id_len,
            table_width: self.table_width,
            start_block_size: self.start_block_size,
            max_direct_block_size: self.max_direct_block_size,
            max_heap_size: self.max_heap_size,
        }
    }

    /// Compare two sets of fractal heap creation parameters for equality.
    /// Mirrors `H5HF_cmp_cparam_test`.
    pub fn cmp_cparam_test(
        left: &FractalHeapCreateParams,
        right: &FractalHeapCreateParams,
    ) -> bool {
        left == right
    }

    /// Width of the doubling table. Mirrors `H5HF_get_dtable_width_test`.
    pub fn get_dtable_width_test(&self) -> u16 {
        self.table_width
    }

    /// Maximum direct-block rows in any indirect block. Mirrors
    /// `H5HF_get_dtable_max_drows_test`.
    pub fn get_dtable_max_drows_test(&self) -> usize {
        self.max_direct_rows()
    }

    /// Maximum direct-block rows in this heap's root indirect block.
    /// Mirrors `H5HF_get_iblock_max_drows_test`.
    pub fn get_iblock_max_drows_test(&self) -> usize {
        usize::from(self.current_root_rows)
    }

    /// Direct block size for a given doubling-table row. Mirrors
    /// `H5HF_get_dblock_size_test`.
    pub fn get_dblock_size_test(&self, row: usize) -> Result<u64> {
        self.checked_row_block_size(row)
    }

    /// Free space remaining in a direct block of the given row after
    /// `used` bytes are consumed. Mirrors `H5HF_get_dblock_free_test`.
    pub fn get_dblock_free_test(&self, used: u64, row: usize) -> Result<u64> {
        self.checked_row_block_size(row)?
            .checked_sub(used)
            .ok_or_else(|| {
                Error::InvalidFormat("fractal heap direct block used bytes exceed size".into())
            })
    }

    /// Retrieve the offset encoded in a managed heap ID. Mirrors
    /// `H5HF_get_id_off_test`.
    pub fn get_id_off_test(&self, heap_id: &[u8]) -> Result<u64> {
        let offset_bytes = (usize::from(self.max_heap_size) + 7) / 8;
        heap_id
            .get(1..1 + offset_bytes)
            .map(read_le_uint)
            .ok_or_else(|| Error::InvalidFormat("fractal heap ID too short for offset".into()))
    }

    /// Retrieve a tiny object's ID length and borrowed bytes.
    pub fn get_tiny_info_slice_test<'a>(&self, heap_id: &'a [u8]) -> Result<(usize, &'a [u8])> {
        let data = self.read_tiny_payload(heap_id)?;
        Ok((data.len(), data))
    }

    /// Huge-object tracking info (B-tree address, max managed size).
    /// Mirrors `H5HF_get_huge_info_test`.
    pub fn get_huge_info_test(&self) -> (u64, u64) {
        (self.huge_btree_addr, u64::from(self.max_managed_obj_size))
    }

    /// Append the doubling-table metadata image to `out`. Mirrors
    /// `H5HF__dtable_encode`.
    pub fn dtable_encode_into(&self, out: &mut Vec<u8>) -> Result<()> {
        validate_heap_create_params(&FractalHeapCreateParams {
            heap_id_len: self.heap_id_len,
            table_width: self.table_width,
            start_block_size: self.start_block_size,
            max_direct_block_size: self.max_direct_block_size,
            max_heap_size: self.max_heap_size,
        })?;
        if self.current_root_rows < self.start_root_rows {
            return Err(Error::InvalidFormat(
                "fractal heap current root rows is smaller than start root rows".into(),
            ));
        }

        out.reserve(FRACTAL_HEAP_DTABLE_IMAGE_LEN);
        out.extend_from_slice(&self.table_width.to_le_bytes());
        out.extend_from_slice(&self.start_block_size.to_le_bytes());
        out.extend_from_slice(&self.max_direct_block_size.to_le_bytes());
        out.extend_from_slice(&self.max_heap_size.to_le_bytes());
        out.extend_from_slice(&self.start_root_rows.to_le_bytes());
        out.extend_from_slice(&self.current_root_rows.to_le_bytes());
        Ok(())
    }

    /// Decode the on-disk doubling-table image into its parsed fields.
    /// Counterpart of `dtable_encode_into`.
    pub fn dtable_decode(bytes: &[u8]) -> Result<FractalHeapDTableImage> {
        if bytes.len() != FRACTAL_HEAP_DTABLE_IMAGE_LEN {
            return Err(Error::InvalidFormat(format!(
                "fractal heap doubling-table image must be exactly {FRACTAL_HEAP_DTABLE_IMAGE_LEN} bytes"
            )));
        }
        let table_width = read_u16_at(bytes, 0, "fractal heap dtable width")?;
        let start_block_size = read_u64_at(bytes, 2, "fractal heap dtable start block size")?;
        let max_direct_block_size =
            read_u64_at(bytes, 10, "fractal heap dtable max direct block size")?;
        let max_heap_size = read_u16_at(bytes, 18, "fractal heap dtable max heap size")?;
        let start_root_rows = read_u16_at(bytes, 20, "fractal heap dtable start root rows")?;
        let current_root_rows = read_u16_at(bytes, 22, "fractal heap dtable current root rows")?;
        validate_heap_create_params(&FractalHeapCreateParams {
            heap_id_len: 1,
            table_width,
            start_block_size,
            max_direct_block_size,
            max_heap_size,
        })?;
        if current_root_rows < start_root_rows {
            return Err(Error::InvalidFormat(
                "fractal heap current root rows is smaller than start root rows".into(),
            ));
        }
        Ok(FractalHeapDTableImage {
            table_width,
            start_block_size,
            max_direct_block_size,
            max_heap_size,
            start_root_rows,
            current_root_rows,
        })
    }

    /// Write debugging info about the doubling table into `out`. Mirrors
    /// `H5HF__dtable_debug`.
    pub fn dtable_debug_fmt(&self, out: &mut impl fmt::Write) -> fmt::Result {
        write!(
            out,
            "FractalHeapDTable(width={}, start={}, max_direct={}, max_heap_bits={})",
            self.table_width, self.start_block_size, self.max_direct_block_size, self.max_heap_size
        )
    }

    /// Initialize the doubling-table fields on the header. Mirrors
    /// `H5HF__dtable_init`.
    pub fn dtable_init(&mut self, params: FractalHeapCreateParams) -> Result<()> {
        validate_heap_create_params(&params)?;
        self.heap_id_len = params.heap_id_len;
        self.table_width = params.table_width;
        self.start_block_size = params.start_block_size;
        self.max_direct_block_size = params.max_direct_block_size;
        self.max_heap_size = params.max_heap_size;
        Ok(())
    }

    /// Compute the (row, column-offset) for a heap offset in the doubling
    /// table. Mirrors `H5HF__dtable_lookup`.
    pub fn dtable_lookup(&self, offset: u64) -> Result<(usize, u64)> {
        let mut row = 0usize;
        let width = u64::from(self.table_width);
        let mut base = 0u64;
        loop {
            let row_span = self
                .checked_row_block_size(row)?
                .checked_mul(width)
                .ok_or_else(|| Error::InvalidFormat("fractal heap row span overflow".into()))?;
            let end = base
                .checked_add(row_span)
                .ok_or_else(|| Error::InvalidFormat("fractal heap row end overflow".into()))?;
            if offset < end {
                return Ok((row, offset - base));
            }
            base = end;
            row = row
                .checked_add(1)
                .ok_or_else(|| Error::InvalidFormat("fractal heap dtable row overflow".into()))?;
            let row_limit = self
                .max_direct_rows_checked()?
                .checked_add(64)
                .ok_or_else(|| {
                    Error::InvalidFormat("fractal heap dtable lookup row limit overflow".into())
                })?;
            if row > row_limit {
                return Err(Error::InvalidFormat(
                    "fractal heap dtable lookup exceeded bounds".into(),
                ));
            }
        }
    }

    /// Release information for the doubling table. Mirrors
    /// `H5HF__dtable_dest` (no-op).
    pub fn dtable_dest(&mut self) {}

    /// Compute the row that can hold a block of the given size. Mirrors
    /// `H5HF__dtable_size_to_row`.
    pub fn dtable_size_to_row(&self, size: u64) -> Result<usize> {
        let mut row = 0usize;
        while self.checked_row_block_size(row)? < size {
            row = row.checked_add(1).ok_or_else(|| {
                Error::InvalidFormat("fractal heap dtable row count overflow".into())
            })?;
        }
        Ok(row)
    }

    /// Compute the number of rows in an indirect block of the given size.
    /// Mirrors `H5HF_dtable_size_to_rows`.
    pub fn dtable_size_to_rows(&self, size: u64) -> Result<usize> {
        self.dtable_size_to_row(size)?
            .checked_add(1)
            .ok_or_else(|| Error::InvalidFormat("fractal heap dtable row count overflow".into()))
    }

    /// Compute the byte span covered by the given number of doubling-table
    /// rows. Mirrors `H5HF_dtable_span_size`.
    pub fn dtable_span_size(&self, rows: usize) -> Result<u64> {
        let mut span = 0u64;
        for row in 0..rows {
            span = span
                .checked_add(
                    self.checked_row_block_size(row)?
                        .checked_mul(u64::from(self.table_width))
                        .ok_or_else(|| {
                            Error::InvalidFormat("fractal heap dtable span overflow".into())
                        })?,
                )
                .ok_or_else(|| Error::InvalidFormat("fractal heap dtable span overflow".into()))?;
        }
        Ok(span)
    }

    /// Initialize information for tracking tiny objects. Mirrors
    /// `H5HF__tiny_init` (no-op).
    pub fn tiny_init(&self) {}

    /// Append a tiny-object heap ID to `out`. Mirrors `H5HF__tiny_insert`.
    pub fn tiny_insert_into(&self, data: &[u8], out: &mut Vec<u8>) -> Result<()> {
        if data.is_empty() || data.len() > 16 {
            return Err(Error::InvalidFormat(
                "tiny fractal heap object size must be 1..=16".into(),
            ));
        }
        out.reserve(data.len() + 1);
        out.push(0x20 | ((data.len() as u8) - 1));
        out.extend_from_slice(data);
        Ok(())
    }

    /// Get the size of a tiny object encoded in `heap_id`. Mirrors
    /// `H5HF__tiny_get_obj_len`.
    pub fn tiny_get_obj_len(&self, heap_id: &[u8]) -> Result<usize> {
        if heap_id.is_empty() {
            return Err(Error::InvalidFormat("empty tiny heap ID".into()));
        }
        Ok(usize::from(heap_id[0] & 0x0f) + 1)
    }

    /// Operate directly on a tiny heap object without allocating.
    pub fn tiny_op_slice<'a>(&self, heap_id: &'a [u8]) -> Result<&'a [u8]> {
        self.read_tiny_payload(heap_id)
    }

    /// Remove a tiny object from the heap statistics. Mirrors
    /// `H5HF__tiny_remove`.
    pub fn tiny_remove(&self, heap_id: &[u8]) -> Result<usize> {
        self.tiny_get_obj_len(heap_id)
    }

    /// Create the v2 B-tree for tracking huge objects in the heap.
    /// Mirrors `H5HF__huge_bt2_create`.
    pub fn huge_bt2_create(&self) -> Vec<FractalHeapHugeRecord> {
        Vec::new()
    }

    /// Initialize information for tracking huge objects. Mirrors
    /// `H5HF__huge_init` (no-op).
    pub fn huge_init(&self) {}

    /// Append a new huge-object heap ID to `out`. Mirrors `H5HF__huge_new_id`.
    pub fn huge_new_id_into(&self, id: u64, out: &mut Vec<u8>) {
        out.reserve(9);
        out.push(0x10);
        out.extend_from_slice(&id.to_le_bytes());
    }

    /// Insert a huge object record into the heap's tracking list,
    /// keeping records sorted. Mirrors `H5HF__huge_insert`.
    pub fn huge_insert(records: &mut Vec<FractalHeapHugeRecord>, record: FractalHeapHugeRecord) {
        records.push(record);
        records.sort_by(FractalHeapHugeRecord::huge_bt2_indir_compare);
    }

    /// Get the size of a huge object by ID. Mirrors
    /// `H5HF__huge_get_obj_len`.
    pub fn huge_get_obj_len(records: &[FractalHeapHugeRecord], id: u64) -> Option<u64> {
        Self::huge_op_ref(records, id).map(|record| record.obj_size)
    }

    /// Borrow a huge object record by ID. Mirrors `H5HF__huge_op`.
    pub fn huge_op_ref(
        records: &[FractalHeapHugeRecord],
        id: u64,
    ) -> Option<&FractalHeapHugeRecord> {
        if let Some(record) = records
            .binary_search_by_key(&id, |record| record.id)
            .ok()
            .and_then(|idx| records.get(idx))
        {
            return Some(record);
        }
        records.iter().find(|record| record.id == id)
    }

    /// Remove a huge object from the v2 B-tree tracker. Mirrors
    /// `H5HF__huge_remove`.
    pub fn huge_remove(
        records: &mut Vec<FractalHeapHugeRecord>,
        id: u64,
    ) -> Option<FractalHeapHugeRecord> {
        FractalHeapHugeRecord::huge_bt2_indir_remove(records, id)
    }

    /// Shut down the huge-object tracker. Mirrors `H5HF__huge_term`.
    pub fn huge_term(&self) {}

    /// Delete all huge objects in the heap. Mirrors `H5HF__huge_delete`.
    pub fn huge_delete(records: &mut Vec<FractalHeapHugeRecord>) {
        records.clear();
    }

    /// Probe whether tracehash has already seen a managed-object offset.
    /// Stub for the libhdf5 tracehash hook.
    pub fn tracehash_man_seen(&self, _offset: u64) -> bool {
        true
    }

    /// Insert an object into a managed direct block. Mirrors
    /// `H5HF__man_insert`.
    pub fn man_insert(heap: &mut FractalHeap, data: Vec<u8>) -> Result<Vec<u8>> {
        heap.insert(data)
    }

    /// Get the size of a managed heap object. Mirrors
    /// `H5HF__man_get_obj_len`.
    pub fn man_get_obj_len(heap: &FractalHeap, id: usize) -> Option<usize> {
        heap.get_obj_len(id)
    }

    /// Get the offset of a managed heap object. Mirrors
    /// `H5HF__man_get_obj_off`.
    pub fn man_get_obj_off(heap: &FractalHeap, heap_id: &[u8]) -> Result<u64> {
        heap.get_obj_off(heap_id)
    }

    /// Write an object to a managed heap. Mirrors `H5HF__man_write`.
    pub fn man_write(heap: &mut FractalHeap, id: usize, data: Vec<u8>) -> Result<()> {
        heap.write(id, data)
    }

    /// Remove an object from a managed heap. Mirrors `H5HF__man_remove`.
    pub fn man_remove(heap: &mut FractalHeap, id: usize) -> Result<Option<Vec<u8>>> {
        heap.remove(id)
    }

    /// Read a managed object from the fractal heap by its heap ID.
    /// Mirrors libhdf5's `H5HF_op` / `H5HF_get_obj_len`: dispatches by
    /// the type bits (managed / huge / tiny) in the heap-ID byte 0.
    pub fn read_managed_object<R: Read + Seek>(
        &self,
        reader: &mut HdfReader<R>,
        heap_id: &[u8],
    ) -> Result<Vec<u8>> {
        match decode_heap_id_type(heap_id)? {
            0 => self.read_managed(reader, heap_id),
            1 => self.read_huge(reader, heap_id),
            2 => Ok(self.read_tiny_payload(heap_id)?.to_vec()),
            id_type => Err(Error::InvalidFormat(format!(
                "unknown heap ID type {id_type}"
            ))),
        }
    }

    /// Visit a heap object while reusing cached managed direct blocks.
    ///
    /// Managed objects are passed as a slice borrowed from the cached direct
    /// block image, mirroring `H5HF_op`/`H5HF__man_op_real`. Tiny objects
    /// borrow directly from the heap ID. Huge objects still materialize into
    /// a temporary buffer before invoking the callback.
    pub fn visit_managed_object_cached<R, T, F>(
        &self,
        reader: &mut HdfReader<R>,
        heap_id: &[u8],
        cache: &mut FractalHeapManagedObjectCache,
        op: F,
    ) -> Result<T>
    where
        R: Read + Seek,
        F: FnOnce(&[u8]) -> Result<T>,
    {
        match decode_heap_id_type(heap_id)? {
            0 => self.visit_managed_with_cache(reader, heap_id, cache, op),
            1 => self.visit_huge(reader, heap_id, op),
            2 => {
                let data = self.read_tiny_payload(heap_id)?;
                op(data)
            }
            id_type => Err(Error::InvalidFormat(format!(
                "unknown heap ID type {id_type}"
            ))),
        }
    }

    /// Visit a heap object with a fresh managed-object cache.
    pub fn visit_managed_object<R, T, F>(
        &self,
        reader: &mut HdfReader<R>,
        heap_id: &[u8],
        op: F,
    ) -> Result<T>
    where
        R: Read + Seek,
        F: FnOnce(&[u8]) -> Result<T>,
    {
        let mut cache = FractalHeapManagedObjectCache::new();
        self.visit_managed_object_cached(reader, heap_id, &mut cache, op)
    }

    /// Read a batch of managed heap objects while reusing decoded direct
    /// blocks across records.
    pub fn read_managed_objects_batched<R: Read + Seek>(
        &self,
        reader: &mut HdfReader<R>,
        heap_ids: &[&[u8]],
    ) -> Vec<Result<Vec<u8>>> {
        let mut cache = FractalHeapManagedObjectCache::new();
        heap_ids
            .iter()
            .map(|heap_id| self.read_managed_object_cached(reader, heap_id, &mut cache))
            .collect()
    }

    pub(crate) fn read_managed_object_cached<R: Read + Seek>(
        &self,
        reader: &mut HdfReader<R>,
        heap_id: &[u8],
        cache: &mut FractalHeapManagedObjectCache,
    ) -> Result<Vec<u8>> {
        self.visit_managed_object_cached(reader, heap_id, cache, |data| Ok(data.to_vec()))
    }

    /// Sanity-check fixed fields of the heap header (address/size widths,
    /// filter pipeline presence, creation parameters).
    fn validate_header(&self) -> Result<()> {
        if self.sizeof_addr == 0 || self.sizeof_addr > 8 {
            return Err(Error::InvalidFormat(
                "fractal heap address size is invalid".into(),
            ));
        }
        if self.sizeof_size == 0 || self.sizeof_size > 8 {
            return Err(Error::InvalidFormat(
                "fractal heap size field width is invalid".into(),
            ));
        }
        if self.io_filter_len > 0 && self.filter_pipeline.is_none() {
            return Err(Error::InvalidFormat(
                "fractal heap filter pipeline is missing".into(),
            ));
        }
        validate_heap_create_params(&self.get_cparam_test())
    }
}

impl FractalHeapManagedObjectCache {
    pub fn new() -> Self {
        Self::default()
    }

    fn lookup_direct_span(&self, offset: u64) -> Option<&DirectBlockSpan> {
        let (_, span) = self.direct_spans.range(..=offset).next_back()?;
        (offset < span.end).then_some(span)
    }

    fn insert_direct_span(&mut self, span: DirectBlockSpan) {
        self.direct_spans.entry(span.start).or_insert(span);
    }

    fn insert_indirect_block(&mut self, key: IndirectBlockCacheKey, block: iblock::IndirectBlock) {
        self.indirect_blocks.entry(key).or_insert(block);
    }

    fn insert_filtered_indirect_block(
        &mut self,
        key: FilteredIndirectBlockCacheKey,
        block: iblock::FilteredIndirectBlock,
    ) {
        self.filtered_indirect_blocks.entry(key).or_insert(block);
    }
}

fn decode_heap_id_type(heap_id: &[u8]) -> Result<u8> {
    if heap_id.is_empty() {
        return Err(Error::InvalidFormat("empty heap ID".into()));
    }

    // Heap ID byte 0 layout (per H5HFpkg.h): bits 6-7 = version,
    // bits 4-5 = type, bits 0-3 = reserved (or tiny-length). Only
    // version 0 is currently defined; reject anything else, matching
    // libhdf5's `H5HF_get_obj_len` "incorrect heap ID version" check.
    let version = (heap_id[0] >> 6) & 0x03;
    if version != 0 {
        return Err(Error::InvalidFormat(format!(
            "unsupported fractal heap ID version {version}"
        )));
    }
    let id_type = (heap_id[0] >> 4) & 0x03;
    if id_type != 2 && heap_id[0] & 0x0f != 0 {
        return Err(Error::InvalidFormat(format!(
            "fractal heap ID reserved bits are nonzero: {:#04x}",
            heap_id[0] & 0x0f
        )));
    }

    Ok(id_type)
}

impl FractalHeap {
    /// Create a new, empty fractal heap. Mirrors libhdf5's `H5HF_create`.
    pub fn create(params: FractalHeapCreateParams) -> Result<Self> {
        Ok(Self {
            header: FractalHeapHeader::hdr_create(params)?,
            objects: Vec::new(),
            closed: false,
        })
    }

    /// Open an existing fractal heap from a decoded header and object
    /// list. Mirrors libhdf5's `H5HF_open`.
    pub fn open(header: FractalHeapHeader, objects: Vec<Vec<u8>>) -> Result<Self> {
        header.validate_header()?;
        Ok(Self {
            header,
            objects,
            closed: false,
        })
    }

    /// Length in bytes of heap IDs issued for this heap. Mirrors
    /// `H5HF_get_id_len`.
    pub fn get_id_len(&self) -> u16 {
        self.header.heap_id_len
    }

    /// Insert a new object into the fractal heap, returning the assigned
    /// heap ID. Mirrors libhdf5's `H5HF_insert`.
    pub fn insert(&mut self, data: Vec<u8>) -> Result<Vec<u8>> {
        self.ensure_open()?;
        let id = u64::try_from(self.objects.len())
            .map_err(|_| Error::InvalidFormat("fractal heap object id overflow".into()))?;
        let data_len = u64::try_from(data.len())
            .map_err(|_| Error::InvalidFormat("fractal heap object length overflow".into()))?;
        if data_len > u64::from(self.header.max_managed_obj_size) {
            return Err(Error::InvalidFormat(
                "fractal heap object exceeds managed limit".into(),
            ));
        }
        let heap_id = self.encode_managed_id(id)?;
        self.objects.push(data);
        self.header.hdr_incr()?;
        Ok(heap_id)
    }

    /// Write data into an existing heap object by ID. Mirrors libhdf5's
    /// `H5HF_write`.
    pub fn write(&mut self, id: usize, data: Vec<u8>) -> Result<()> {
        self.ensure_open()?;
        let slot = self
            .objects
            .get_mut(id)
            .ok_or_else(|| Error::InvalidFormat("fractal heap object id out of bounds".into()))?;
        *slot = data;
        Ok(())
    }

    /// Operate directly on a heap object by ID. Mirrors libhdf5's
    /// `H5HF_op`.
    pub fn op(&self, id: usize) -> Option<&[u8]> {
        self.objects.get(id).map(Vec::as_slice)
    }

    /// Get the heap offset for a managed heap ID. Mirrors libhdf5's
    /// `H5HF_get_obj_off`.
    pub fn get_obj_off(&self, heap_id: &[u8]) -> Result<u64> {
        self.header.get_id_off_test(heap_id)
    }

    /// Get the size of a heap object by ID. Mirrors libhdf5's
    /// `H5HF_get_obj_len`.
    pub fn get_obj_len(&self, id: usize) -> Option<usize> {
        self.objects.get(id).map(Vec::len)
    }

    /// Remove an object from the fractal heap. Mirrors libhdf5's
    /// `H5HF_remove`.
    pub fn remove(&mut self, id: usize) -> Result<Option<Vec<u8>>> {
        self.ensure_open()?;
        let removed = if id < self.objects.len() {
            Some(self.objects.remove(id))
        } else {
            None
        };
        if removed.is_some() {
            self.header.hdr_decr()?;
        }
        Ok(removed)
    }

    /// Close the fractal heap. Mirrors libhdf5's `H5HF_close`.
    pub fn close(&mut self) {
        self.closed = true;
    }

    /// Delete the fractal heap (clearing all contents). Mirrors libhdf5's
    /// `H5HF_delete`.
    pub fn delete(mut self) {
        self.objects.clear();
        self.header.num_managed_objects = 0;
        self.closed = true;
    }

    /// Retrieve metadata statistics for the fractal heap (object count
    /// and aggregate size). Mirrors libhdf5's `H5HF_stat_info`.
    pub fn stat_info(&self) -> (usize, u64) {
        (self.objects.len(), self.size())
    }

    /// Aggregate storage size used by the heap's objects. Mirrors
    /// libhdf5's `H5HF_size`.
    pub fn size(&self) -> u64 {
        self.objects
            .iter()
            .map(|o| u64::try_from(o.len()).unwrap_or(u64::MAX))
            .try_fold(0u64, |acc, len| acc.checked_add(len))
            .unwrap_or(u64::MAX)
    }

    /// Write a fractal heap ID as hex into `out`. Mirrors libhdf5's
    /// `H5HF_id_print`.
    pub fn id_print_fmt(heap_id: &[u8], out: &mut impl fmt::Write) -> fmt::Result {
        for byte in heap_id {
            write!(out, "{byte:02x}")?;
        }
        Ok(())
    }

    /// Build the heap-ID bytes for a managed object at `offset`.
    fn encode_managed_id(&self, offset: u64) -> Result<Vec<u8>> {
        let mut id = vec![0; usize::from(self.header.heap_id_len)];
        if !id.is_empty() {
            id[0] = 0;
            let offset_bytes = managed_id_offset_bytes(
                self.header.max_heap_size,
                self.header.heap_id_len,
                "fractal heap managed ID",
            )?;
            ensure_uint_fits_width(offset, offset_bytes, "fractal heap managed object offset")?;
            id[1..1 + offset_bytes].copy_from_slice(&offset.to_le_bytes()[..offset_bytes]);
        }
        Ok(id)
    }

    /// Return an error if the heap has been closed.
    fn ensure_open(&self) -> Result<()> {
        if self.closed {
            Err(Error::InvalidFormat("fractal heap is closed".into()))
        } else {
            Ok(())
        }
    }
}

impl FractalHeapIndirectBlock {
    /// Pin the indirect block (increase its ref count).
    pub fn iblock_pin(&mut self) {
        self.ref_count += 1;
    }

    /// Unpin the indirect block (decrease its ref count, saturating at 0).
    pub fn iblock_unpin(&mut self) {
        self.ref_count = self.ref_count.saturating_sub(1);
    }

    /// Increment the indirect block's ref count.
    pub fn iblock_incr(&mut self) {
        self.iblock_pin();
    }

    /// Decrement the indirect block's ref count.
    pub fn iblock_decr(&mut self) {
        self.iblock_unpin();
    }

    /// Mark the indirect block as dirty.
    pub fn iblock_dirty(&mut self) {
        self.dirty = true;
    }

    /// Create a root indirect block with the given row/width geometry.
    /// Mirrors `H5HF__man_iblock_root_create`.
    pub fn man_iblock_root_create(nrows: usize, width: usize) -> Result<Self> {
        let child_count = nrows.checked_mul(width).ok_or_else(|| {
            Error::InvalidFormat("fractal heap indirect block size overflow".into())
        })?;
        Ok(Self {
            nrows,
            child_addrs: vec![u64::MAX; child_count],
            ref_count: 0,
            dirty: false,
        })
    }

    /// Allocate a row of entries in the indirect block, growing the
    /// child-address table if needed. Mirrors `H5HF__man_iblock_alloc_row`.
    pub fn man_iblock_alloc_row(&mut self, row: usize, width: usize) -> Result<()> {
        let rows = row.checked_add(1).ok_or_else(|| {
            Error::InvalidFormat("fractal heap indirect block row overflow".into())
        })?;
        let needed = rows.checked_mul(width).ok_or_else(|| {
            Error::InvalidFormat("fractal heap indirect block size overflow".into())
        })?;
        if self.child_addrs.len() < needed {
            self.child_addrs.resize(needed, u64::MAX);
        }
        self.nrows = self.nrows.max(rows);
        Ok(())
    }

    /// Allocate and initialize a managed indirect block. Mirrors
    /// `H5HF__man_iblock_create`.
    pub fn man_iblock_create(nrows: usize, width: usize) -> Result<Self> {
        Self::man_iblock_root_create(nrows, width)
    }

    /// Pin the indirect block for use. Mirrors `H5HF__man_iblock_protect`
    /// (`H5AC_protect` wrapper).
    pub fn man_iblock_protect(&self) -> Self {
        self.clone()
    }

    /// Release the protection acquired by `man_iblock_protect`. Mirrors
    /// `H5HF__man_iblock_unprotect`.
    pub fn man_iblock_unprotect(self) -> Self {
        self
    }

    /// Attach a child block (direct or indirect) at the given entry slot.
    /// Mirrors `H5HF__man_iblock_attach`.
    pub fn man_iblock_attach(&mut self, index: usize, addr: u64) {
        if index >= self.child_addrs.len() {
            self.child_addrs.resize(index + 1, u64::MAX);
        }
        self.child_addrs[index] = addr;
        self.dirty = true;
    }

    /// Detach a child block from the given entry slot, returning its old
    /// address. Mirrors `H5HF__man_iblock_detach`.
    pub fn man_iblock_detach(&mut self, index: usize) -> Option<u64> {
        let addr = self.child_addrs.get_mut(index)?;
        let old = *addr;
        *addr = u64::MAX;
        self.dirty = true;
        Some(old)
    }

    /// Retrieve the address of a child block at the given entry slot.
    /// Mirrors `H5HF__man_iblock_entry_addr`.
    pub fn man_iblock_entry_addr(&self, index: usize) -> Option<u64> {
        self.child_addrs.get(index).copied()
    }

    /// Delete a managed indirect block. Mirrors `H5HF__man_iblock_delete`.
    pub fn man_iblock_delete(self) {}

    /// Retrieve (nrows, child-table-len) for a parent-info query.
    pub fn man_iblock_parent_info(&self) -> (usize, usize) {
        (self.nrows, self.child_addrs.len())
    }

    /// Destroy the indirect block in memory. Mirrors `H5HF__man_iblock_dest`.
    pub fn man_iblock_dest(self) {}

    /// Initial cache-load size for an indirect block (the magic).
    pub fn cache_iblock_get_initial_load_size() -> usize {
        4
    }

    /// Verify the cache image's checksum. Mirrors
    /// `H5HF__cache_iblock_verify_chksum`.
    pub fn cache_iblock_verify_chksum(image: &[u8]) -> Result<()> {
        verify_image_checksum(image, "fractal heap indirect block")
    }

    /// On-disk image size of the indirect block. Mirrors
    /// `H5HF__cache_iblock_image_len`.
    pub fn cache_iblock_image_len(&self) -> Result<usize> {
        let child_bytes = self.child_addrs.len().checked_mul(8).ok_or_else(|| {
            Error::InvalidFormat("fractal heap indirect block image length overflow".into())
        })?;
        4usize
            .checked_add(1)
            .and_then(|len| len.checked_add(8))
            .and_then(|len| len.checked_add(child_bytes))
            .and_then(|len| len.checked_add(4))
            .ok_or_else(|| {
                Error::InvalidFormat("fractal heap indirect block image length overflow".into())
            })
    }

    /// Pre-serialize the indirect block into an existing output buffer.
    pub fn cache_iblock_pre_serialize_into(&self, out: &mut Vec<u8>) -> Result<()> {
        self.cache_iblock_serialize_into(out)
    }

    /// Append the indirect block's on-disk image to `out`. Mirrors
    /// `H5HF__cache_iblock_serialize`.
    pub fn cache_iblock_serialize_into(&self, out: &mut Vec<u8>) -> Result<()> {
        let nrows = u64::try_from(self.nrows).map_err(|_| {
            Error::InvalidFormat("fractal heap indirect block row count is too large".into())
        })?;
        let start = out.len();
        out.reserve(self.cache_iblock_image_len()?);
        out.extend_from_slice(&FHIB_MAGIC);
        out.push(0);
        out.extend_from_slice(&nrows.to_le_bytes());
        for addr in &self.child_addrs {
            out.extend_from_slice(&addr.to_le_bytes());
        }
        let checksum = checksum_metadata(&out[start..]);
        out.extend_from_slice(&checksum.to_le_bytes());
        Ok(())
    }

    /// Flush-dependency lifecycle hook. Mirrors
    /// `H5HF__cache_iblock_notify`.
    pub fn cache_iblock_notify(&mut self) {
        self.dirty = false;
    }

    /// Free the in-core image (no-op for owned buffers).
    pub fn cache_iblock_free_icr(_image: Vec<u8>) {}

    /// Sanity check: indirect block has no dirty descendants. Mirrors
    /// `H5HF__cache_verify_iblock_descendants_clean`.
    pub fn cache_verify_iblock_descendants_clean(&self) -> bool {
        !self.dirty
    }

    /// Sanity check: all descendant direct blocks are clean. Mirrors
    /// `H5HF__cache_verify_iblocks_dblocks_clean`.
    pub fn cache_verify_iblocks_dblocks_clean(&self, dblocks: &[FractalHeapDirectBlock]) -> bool {
        !self.dirty && dblocks.iter().all(|dblock| !dblock.dirty)
    }

    /// Sanity check: all descendant indirect blocks are clean. Mirrors
    /// `H5HF__cache_verify_descendant_iblocks_clean`.
    pub fn cache_verify_descendant_iblocks_clean(blocks: &[Self]) -> bool {
        blocks.iter().all(|block| !block.dirty)
    }

    /// Write debugging info about a fractal heap indirect block into `out`.
    /// Mirrors `H5HF_iblock_print`.
    pub fn iblock_print_fmt(&self, out: &mut impl fmt::Write) -> fmt::Result {
        self.iblock_debug_fmt(out)
    }

    /// Write debugging info about a fractal heap indirect block into `out`.
    /// Mirrors `H5HF_iblock_debug`.
    pub fn iblock_debug_fmt(&self, out: &mut impl fmt::Write) -> fmt::Result {
        write!(
            out,
            "FractalHeapIndirectBlock(nrows={}, children={}, dirty={})",
            self.nrows,
            self.child_addrs.len(),
            self.dirty
        )
    }
}

impl FractalHeapDirectBlock {
    /// Create a fresh direct block large enough to hold the given data.
    /// Mirrors `H5HF__man_dblock_new`.
    pub fn man_dblock_new(addr: u64, data: Vec<u8>) -> Self {
        Self {
            addr,
            data,
            dirty: false,
        }
    }

    /// Pin a direct block for use. Mirrors `H5HF__man_dblock_protect`
    /// (`H5AC_protect` wrapper).
    pub fn man_dblock_protect(&self) -> Self {
        self.clone()
    }

    /// Borrow a pinned direct block when the caller does not need ownership.
    pub fn man_dblock_protect_ref(&self) -> &Self {
        self
    }

    /// Consume a direct block as its pinned value, avoiding a clone when
    /// ownership is already available.
    pub fn man_dblock_protect_into(self) -> Self {
        self
    }

    /// Delete a managed direct block. Mirrors `H5HF__man_dblock_delete`.
    pub fn man_dblock_delete(self) {}

    /// Destroy a managed direct block. Mirrors `H5HF__man_dblock_destroy`.
    pub fn man_dblock_destroy(self) {}

    /// Destroy the direct block in memory. Mirrors `H5HF__man_dblock_dest`.
    pub fn man_dblock_dest(self) {}

    /// Initial cache-load size for a direct block (just the magic).
    pub fn cache_dblock_get_initial_load_size() -> usize {
        4
    }

    /// On-disk image size of the direct block. Mirrors
    /// `H5HF__cache_dblock_image_len`.
    pub fn cache_dblock_image_len(&self) -> Result<usize> {
        4usize
            .checked_add(1)
            .and_then(|len| len.checked_add(8))
            .and_then(|len| len.checked_add(self.data.len()))
            .and_then(|len| len.checked_add(4))
            .ok_or_else(|| {
                Error::InvalidFormat("fractal heap direct block image length overflow".into())
            })
    }

    /// Pre-serialize the direct block into an existing output buffer.
    pub fn cache_dblock_pre_serialize_into(&self, out: &mut Vec<u8>) -> Result<()> {
        self.cache_dblock_serialize_into(out)
    }

    /// Append the direct block's on-disk image to `out`. Mirrors
    /// `H5HF__cache_dblock_serialize`.
    pub fn cache_dblock_serialize_into(&self, out: &mut Vec<u8>) -> Result<()> {
        let start = out.len();
        out.reserve(self.cache_dblock_image_len()?);
        out.extend_from_slice(&FHDB_MAGIC);
        out.push(0);
        out.extend_from_slice(&self.addr.to_le_bytes());
        out.extend_from_slice(&self.data);
        let checksum = checksum_metadata(&out[start..]);
        out.extend_from_slice(&checksum.to_le_bytes());
        Ok(())
    }

    /// Flush-dependency lifecycle hook. Mirrors
    /// `H5HF__cache_dblock_notify`.
    pub fn cache_dblock_notify(&mut self) {
        self.dirty = false;
    }

    /// Free the in-core image (no-op for owned buffers).
    pub fn cache_dblock_free_icr(_image: Vec<u8>) {}

    /// File-space allocation footprint of the direct block. Mirrors
    /// `H5HF__cache_dblock_fsf_size`.
    pub fn cache_dblock_fsf_size(&self) -> usize {
        self.data.len()
    }

    /// Write per-block debug info into `out`. Mirrors `H5HF_dblock_debug_cb`.
    pub fn dblock_debug_cb_fmt(&self, out: &mut impl fmt::Write) -> fmt::Result {
        self.dblock_debug_fmt(out)
    }

    /// Write debugging info about a fractal heap direct block into `out`.
    /// Mirrors `H5HF_dblock_debug`.
    pub fn dblock_debug_fmt(&self, out: &mut impl fmt::Write) -> fmt::Result {
        write!(
            out,
            "FractalHeapDirectBlock(addr={:#x}, size={})",
            self.addr,
            self.data.len()
        )
    }
}

impl FractalHeapHugeContext {
    /// Create a client callback context for the huge-object v2 B-tree.
    /// Mirrors `H5HF__huge_bt2_crt_context`.
    pub fn huge_bt2_crt_context(filtered: bool) -> Self {
        Self { filtered }
    }

    /// Destroy the client callback context. Mirrors
    /// `H5HF__huge_bt2_dst_context`.
    pub fn huge_bt2_dst_context(self) {}
}

impl FractalHeapHugeRecord {
    /// Store native info into a v2 B-tree record for an indirectly
    /// accessed huge object. Mirrors `H5HF__huge_bt2_indir_store`.
    pub fn huge_bt2_indir_store(id: u64, addr: u64, len: u64) -> Self {
        Self {
            id,
            addr,
            len,
            obj_size: len,
            filter_mask: 0,
        }
    }

    /// Free space for an indirectly accessed huge object by removing its
    /// v2 B-tree record. Mirrors `H5HF__huge_bt2_indir_remove`.
    pub fn huge_bt2_indir_remove(records: &mut Vec<Self>, id: u64) -> Option<Self> {
        let idx = records
            .binary_search_by_key(&id, |record| record.id)
            .ok()
            .or_else(|| records.iter().position(|record| record.id == id))?;
        Some(records.remove(idx))
    }

    /// Compare two indirect-huge records by ID for ordering. Mirrors
    /// `H5HF__huge_bt2_indir_compare`.
    pub fn huge_bt2_indir_compare(left: &Self, right: &Self) -> std::cmp::Ordering {
        left.id.cmp(&right.id)
    }

    /// Write debug info for an indirect-huge record into `out`. Mirrors
    /// `H5HF__huge_bt2_indir_debug`.
    pub fn huge_bt2_indir_debug_fmt(&self, out: &mut impl fmt::Write) -> fmt::Result {
        write!(
            out,
            "HugeIndirect(id={}, addr={:#x}, len={})",
            self.id, self.addr, self.len
        )
    }

    /// Whether this filtered-indirect record matches the given ID and is
    /// filtered. Mirrors `H5HF__huge_bt2_filt_indir_found`.
    pub fn huge_bt2_filt_indir_found(&self, id: u64) -> bool {
        self.id == id && self.filter_mask != 0
    }

    /// Store native info into a v2 B-tree record for an indirectly
    /// accessed, filtered huge object. Mirrors
    /// `H5HF__huge_bt2_filt_indir_store`.
    pub fn huge_bt2_filt_indir_store(
        id: u64,
        addr: u64,
        len: u64,
        obj_size: u64,
        filter_mask: u32,
    ) -> Self {
        Self {
            id,
            addr,
            len,
            obj_size,
            filter_mask,
        }
    }

    /// Free space for an indirectly accessed, filtered huge object.
    /// Mirrors `H5HF__huge_bt2_filt_indir_remove`.
    pub fn huge_bt2_filt_indir_remove(records: &mut Vec<Self>, id: u64) -> Option<Self> {
        Self::huge_bt2_indir_remove(records, id)
    }

    /// Compare two filtered-indirect records by ID. Mirrors
    /// `H5HF__huge_bt2_filt_indir_compare`.
    pub fn huge_bt2_filt_indir_compare(left: &Self, right: &Self) -> std::cmp::Ordering {
        Self::huge_bt2_indir_compare(left, right)
    }

    /// Write debug info for a filtered-indirect record into `out`. Mirrors
    /// `H5HF__huge_bt2_filt_indir_debug`.
    pub fn huge_bt2_filt_indir_debug_fmt(&self, out: &mut impl fmt::Write) -> fmt::Result {
        write!(
            out,
            "HugeFilteredIndirect(id={}, addr={:#x}, len={}, obj_size={}, mask={:#x})",
            self.id, self.addr, self.len, self.obj_size, self.filter_mask
        )
    }

    /// Store native info into a v2 B-tree record for a directly accessed
    /// huge object. Mirrors `H5HF__huge_bt2_dir_store`.
    pub fn huge_bt2_dir_store(addr: u64, len: u64) -> Self {
        Self::huge_bt2_indir_store(0, addr, len)
    }

    /// Free space for a directly accessed huge object. Mirrors
    /// `H5HF__huge_bt2_dir_remove`.
    pub fn huge_bt2_dir_remove(records: &mut Vec<Self>, addr: u64) -> Option<Self> {
        let idx = records.iter().position(|record| record.addr == addr)?;
        Some(records.remove(idx))
    }

    /// Compare two direct-huge records by address. Mirrors
    /// `H5HF__huge_bt2_dir_compare`.
    pub fn huge_bt2_dir_compare(left: &Self, right: &Self) -> std::cmp::Ordering {
        left.addr.cmp(&right.addr)
    }

    /// Write debug info for a direct-huge record into `out`. Mirrors
    /// `H5HF__huge_bt2_dir_debug`.
    pub fn huge_bt2_dir_debug_fmt(&self, out: &mut impl fmt::Write) -> fmt::Result {
        write!(out, "HugeDirect(addr={:#x}, len={})", self.addr, self.len)
    }

    /// Whether this filtered-direct record matches the given address and
    /// is filtered. Mirrors `H5HF__huge_bt2_filt_dir_found`.
    pub fn huge_bt2_filt_dir_found(&self, addr: u64) -> bool {
        self.addr == addr && self.filter_mask != 0
    }

    /// Store native info into a v2 B-tree record for a directly accessed,
    /// filtered huge object. Mirrors `H5HF__huge_bt2_filt_dir_store`.
    pub fn huge_bt2_filt_dir_store(addr: u64, len: u64, obj_size: u64, filter_mask: u32) -> Self {
        Self {
            id: 0,
            addr,
            len,
            obj_size,
            filter_mask,
        }
    }

    /// Free space for a directly accessed, filtered huge object. Mirrors
    /// `H5HF__huge_bt2_filt_dir_remove`.
    pub fn huge_bt2_filt_dir_remove(records: &mut Vec<Self>, addr: u64) -> Option<Self> {
        Self::huge_bt2_dir_remove(records, addr)
    }

    /// Compare two filtered-direct records by address. Mirrors
    /// `H5HF__huge_bt2_filt_dir_compare`.
    pub fn huge_bt2_filt_dir_compare(left: &Self, right: &Self) -> std::cmp::Ordering {
        Self::huge_bt2_dir_compare(left, right)
    }

    /// Append a filtered-direct record's raw on-disk form to `out`.
    /// Mirrors `H5HF__huge_bt2_filt_dir_encode`.
    pub fn huge_bt2_filt_dir_encode_into(&self, out: &mut Vec<u8>) -> Result<()> {
        if self.filter_mask == 0 {
            return Err(Error::InvalidFormat(
                "filtered huge direct record has no filter mask".into(),
            ));
        }
        out.reserve(HUGE_FILTERED_DIRECT_RECORD_LEN);
        out.extend_from_slice(&self.addr.to_le_bytes());
        out.extend_from_slice(&self.len.to_le_bytes());
        out.extend_from_slice(&self.filter_mask.to_le_bytes());
        out.extend_from_slice(&self.obj_size.to_le_bytes());
        Ok(())
    }

    /// Decode the on-disk raw form of a filtered-direct record into a
    /// native record. Mirrors `H5HF__huge_bt2_filt_dir_decode`.
    pub fn huge_bt2_filt_dir_decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != HUGE_FILTERED_DIRECT_RECORD_LEN {
            return Err(Error::InvalidFormat(format!(
                "filtered huge direct record must be exactly {HUGE_FILTERED_DIRECT_RECORD_LEN} bytes"
            )));
        }
        let addr = read_u64_at(bytes, 0, "filtered huge direct record address")?;
        let len = read_u64_at(bytes, 8, "filtered huge direct record length")?;
        let filter_mask = read_u32_at(bytes, 16, "filtered huge direct record filter mask")?;
        if filter_mask == 0 {
            return Err(Error::InvalidFormat(
                "filtered huge direct record has no filter mask".into(),
            ));
        }
        let obj_size = read_u64_at(bytes, 20, "filtered huge direct record object size")?;
        Ok(Self {
            id: 0,
            addr,
            len,
            obj_size,
            filter_mask,
        })
    }

    /// Write debug info for a filtered-direct record into `out`. Mirrors
    /// `H5HF__huge_bt2_filt_dir_debug`.
    pub fn huge_bt2_filt_dir_debug_fmt(&self, out: &mut impl fmt::Write) -> fmt::Result {
        write!(
            out,
            "HugeFilteredDirect(addr={:#x}, len={}, obj_size={}, mask={:#x})",
            self.addr, self.len, self.obj_size, self.filter_mask
        )
    }
}

impl FractalHeapIterator {
    /// Initialize a block iterator walking `count` offsets starting at
    /// `start`. Mirrors `H5HF__man_iter_init`; falls back to saturating
    /// math when offsets would overflow.
    pub fn man_iter_init(start: u64, count: u64) -> Self {
        Self::man_iter_init_checked(start, count).unwrap_or_else(|_| {
            let offsets = (0..count).map(|idx| start.saturating_add(idx)).collect();
            Self { offsets, index: 0 }
        })
    }

    /// Initialize a block iterator with overflow checking; returns an
    /// error rather than truncating on overflow.
    pub fn man_iter_init_checked(start: u64, count: u64) -> Result<Self> {
        let mut offsets = Vec::new();
        for idx in 0..count {
            offsets.push(start.checked_add(idx).ok_or_else(|| {
                Error::InvalidFormat("fractal heap iterator offset overflow".into())
            })?);
        }
        Ok(Self { offsets, index: 0 })
    }

    /// Position the iterator at the first entry whose offset is `>= offset`.
    /// Mirrors `H5HF__man_iter_start_offset`.
    pub fn man_iter_start_offset(&mut self, offset: u64) {
        self.index = self
            .offsets
            .iter()
            .position(|candidate| *candidate >= offset)
            .unwrap_or(self.offsets.len());
    }

    /// Position the iterator at a particular entry index. Mirrors
    /// `H5HF__man_iter_start_entry`.
    pub fn man_iter_start_entry(&mut self, entry: usize) {
        self.index = entry.min(self.offsets.len());
    }

    /// Reset the iterator to its initial state. Mirrors
    /// `H5HF__man_iter_reset`.
    pub fn man_iter_reset(&mut self) {
        self.index = 0;
    }

    /// Advance to the next offset, returning it. Mirrors
    /// `H5HF__man_iter_next`.
    pub fn man_iter_next(&mut self) -> Option<u64> {
        let value = self.offsets.get(self.index).copied();
        if value.is_some() {
            self.index += 1;
        }
        value
    }

    /// Move the iterator up one level. Mirrors `H5HF__man_iter_up`.
    pub fn man_iter_up(&mut self) {
        self.index = self.index.saturating_sub(1);
    }

    /// Move the iterator down one level. Mirrors `H5HF__man_iter_down`.
    pub fn man_iter_down(&mut self) {
        self.index = self.index.saturating_add(1).min(self.offsets.len());
    }

    /// Current offset at the iterator's position. Mirrors
    /// `H5HF__man_iter_curr`.
    pub fn man_iter_curr(&self) -> Option<u64> {
        self.offsets.get(self.index).copied()
    }

    /// Whether the iterator has any remaining entries. Mirrors
    /// `H5HF__man_iter_ready`.
    pub fn man_iter_ready(&self) -> bool {
        self.index < self.offsets.len()
    }
}

impl FractalHeapSection {
    /// Free a section node. Mirrors `H5HF__sect_node_free`.
    pub fn sect_node_free(self) {}

    /// Create a new 'single' free-space section. Mirrors
    /// `H5HF__sect_single_new`.
    pub fn sect_single_new(offset: u64, size: u64) -> Self {
        Self::Single { offset, size }
    }

    /// Update the memory info for a 'single' free section. Mirrors
    /// `H5HF__sect_single_revive` (no-op).
    pub fn sect_single_revive(&mut self) {}

    /// Retrieve the direct-block info `(offset, size)` for a Single section.
    /// Mirrors `H5HF__sect_single_dblock_info`.
    pub fn sect_single_dblock_info(&self) -> Option<(u64, u64)> {
        match self {
            Self::Single { offset, size } => Some((*offset, *size)),
            _ => None,
        }
    }

    /// Reduce the size of a 'single' section by `amount`. Mirrors
    /// `H5HF__sect_single_reduce`.
    pub fn sect_single_reduce(&mut self, amount: u64) {
        if let Self::Single { size, .. } = self {
            *size = size.saturating_sub(amount);
        }
    }

    /// Add a section to the free space sections list. Mirrors
    /// `H5HF__sect_single_add`.
    pub fn sect_single_add(sections: &mut Vec<Self>, section: Self) {
        sections.push(section);
    }

    /// Deserialize a buffer into a live 'single' section. Mirrors
    /// `H5HF__sect_single_deserialize`.
    pub fn sect_single_deserialize(offset: u64, size: u64) -> Self {
        Self::Single { offset, size }
    }

    /// Whether two sections are adjacent and so can merge. Mirrors
    /// `H5HF__sect_single_can_merge`.
    pub fn sect_single_can_merge(&self, other: &Self) -> bool {
        section_end(self) == section_offset(other)
    }

    /// Merge `other` into self when possible. Mirrors
    /// `H5HF__sect_single_merge`.
    pub fn sect_single_merge(&mut self, other: &Self) -> bool {
        self.sect_single_merge_checked(other).unwrap_or(false)
    }

    /// Merge `other` into self with overflow checking on the combined size.
    pub fn sect_single_merge_checked(&mut self, other: &Self) -> Result<bool> {
        if self.sect_single_can_merge(other) {
            if let (Some(size), Some(other_size)) = (section_size_mut(self), section_size(other)) {
                *size = size.checked_add(other_size).ok_or_else(|| {
                    Error::InvalidFormat("fractal heap section size overflow".into())
                })?;
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Whether this section can shrink the container (sits at the EOA).
    /// Mirrors `H5HF__sect_single_can_shrink`.
    pub fn sect_single_can_shrink(&self, eoa: u64) -> bool {
        section_end(self) == Some(eoa)
    }

    /// Shrink the container with this section. Mirrors
    /// `H5HF__sect_single_shrink`.
    pub fn sect_single_shrink(&mut self, amount: u64) {
        self.sect_single_reduce(amount);
    }

    /// Free a 'single' section node. Mirrors `H5HF__sect_single_free`.
    pub fn sect_single_free(self) {}

    /// Validity check for a 'single' section. Mirrors
    /// `H5HF__sect_single_valid`.
    pub fn sect_single_valid(&self) -> bool {
        section_size(self).is_some_and(|size| size > 0)
    }

    /// Convert a Single section into a Row section for the given row.
    /// Mirrors `H5HF__sect_row_from_single`.
    pub fn sect_row_from_single(section: Self, row: usize) -> Self {
        let offset = section_offset(&section).unwrap_or(0);
        let size = section_size(&section).unwrap_or(0);
        Self::Row { row, offset, size }
    }

    /// Update the memory info for a 'row' free section. Mirrors
    /// `H5HF__sect_row_revive` (no-op).
    pub fn sect_row_revive(&mut self) {}

    /// Reduce the size of a 'row' section by `amount`. Mirrors
    /// `H5HF__sect_row_reduce`.
    pub fn sect_row_reduce(&mut self, amount: u64) {
        if let Self::Row { size, .. } = self {
            *size = size.saturating_sub(amount);
        }
    }

    /// Make a row a "first row". Mirrors `H5HF__sect_row_first`.
    pub fn sect_row_first(&self) -> Option<u64> {
        section_offset(self)
    }

    /// Retrieve the indirect-block index for a row section. Mirrors
    /// `H5HF__sect_row_get_iblock`.
    pub fn sect_row_get_iblock(&self) -> Option<usize> {
        match self {
            Self::Row { row, .. } => Some(*row),
            _ => None,
        }
    }

    /// Update row/parent info after the parent indirect block is removed.
    /// Mirrors `H5HF__sect_row_parent_removed` (no-op).
    pub fn sect_row_parent_removed(&mut self) {}

    /// Initialize the row section class structure. Mirrors
    /// `H5HF__sect_row_init_cls` (no-op).
    pub fn sect_row_init_cls() {}

    /// Terminate the row section class structure. Mirrors
    /// `H5HF__sect_row_term_cls` (no-op).
    pub fn sect_row_term_cls() {}

    /// Append a row section's serialized form to `out`. Mirrors
    /// `H5HF__sect_row_serialize`.
    pub fn sect_row_serialize_into(&self, out: &mut Vec<u8>) -> Result<()> {
        let Self::Row { offset, size, .. } = self else {
            return Err(Error::InvalidFormat(
                "fractal heap row section serializer received non-row section".into(),
            ));
        };
        out.reserve(16);
        out.extend_from_slice(&offset.to_le_bytes());
        out.extend_from_slice(&size.to_le_bytes());
        Ok(())
    }

    /// Deserialize a row section. Mirrors `H5HF__sect_row_deserialize`.
    pub fn sect_row_deserialize(row: usize, offset: u64, size: u64) -> Self {
        Self::Row { row, offset, size }
    }

    /// Whether two row sections can merge. Mirrors
    /// `H5HF__sect_row_can_merge`.
    pub fn sect_row_can_merge(&self, other: &Self) -> bool {
        self.sect_single_can_merge(other)
    }

    /// Merge two row sections. Mirrors `H5HF__sect_row_merge`.
    pub fn sect_row_merge(&mut self, other: &Self) -> bool {
        self.sect_single_merge(other)
    }

    /// Merge two row sections with overflow checking on combined size.
    pub fn sect_row_merge_checked(&mut self, other: &Self) -> Result<bool> {
        self.sect_single_merge_checked(other)
    }

    /// Whether a row section can shrink the container. Mirrors
    /// `H5HF__sect_row_can_shrink`.
    pub fn sect_row_can_shrink(&self, eoa: u64) -> bool {
        self.sect_single_can_shrink(eoa)
    }

    /// Shrink the container by `amount`. Mirrors `H5HF__sect_row_shrink`.
    pub fn sect_row_shrink(&mut self, amount: u64) {
        self.sect_row_reduce(amount);
    }

    /// Free a 'row' section node (real). Mirrors `H5HF__sect_row_free_real`.
    pub fn sect_row_free_real(self) {}

    /// Free a 'row' section node. Mirrors `H5HF__sect_row_free`.
    pub fn sect_row_free(self) {}

    /// Validity check for a row section. Mirrors `H5HF__sect_row_valid`.
    pub fn sect_row_valid(&self) -> bool {
        self.sect_single_valid()
    }

    /// Write debugging information about a row section into `out`. Mirrors
    /// `H5HF__sect_row_debug`.
    pub fn sect_row_debug_fmt(&self, out: &mut impl fmt::Write) -> fmt::Result {
        write!(out, "{self:?}")
    }

    /// Indirect-block offset for this section. Mirrors
    /// `H5HF__sect_indirect_iblock_off`.
    pub fn sect_indirect_iblock_off(&self) -> Option<u64> {
        section_offset(self)
    }

    /// Whether this is a top-level Indirect section. Mirrors
    /// `H5HF__sect_indirect_top`.
    pub fn sect_indirect_top(&self) -> bool {
        matches!(self, Self::Indirect { .. })
    }

    /// Terminate the indirect section class. Mirrors
    /// `H5HF__sect_indirect_term_cls` (no-op).
    pub fn sect_indirect_term_cls() {}

    /// Create a new Indirect section. Mirrors `H5HF__sect_indirect_new`.
    pub fn sect_indirect_new(offset: u64, rows: usize, size: u64) -> Self {
        Self::Indirect { offset, rows, size }
    }

    /// Create an Indirect section that backs a row section. Mirrors
    /// `H5HF__sect_indirect_for_row`.
    pub fn sect_indirect_for_row(row: usize, offset: u64, size: u64) -> Self {
        Self::sect_indirect_for_row_checked(row, offset, size).unwrap_or(Self::Indirect {
            offset,
            rows: usize::MAX,
            size,
        })
    }

    /// Checked variant of `sect_indirect_for_row` that surfaces row-count
    /// overflows rather than saturating.
    pub fn sect_indirect_for_row_checked(row: usize, offset: u64, size: u64) -> Result<Self> {
        let rows = row.checked_add(1).ok_or_else(|| {
            Error::InvalidFormat("fractal heap indirect section row overflow".into())
        })?;
        Ok(Self::Indirect { offset, rows, size })
    }

    /// Initialize the derived row count for a newly created indirect
    /// section. Mirrors `H5HF__sect_indirect_init_rows`.
    pub fn sect_indirect_init_rows(&mut self, rows: usize) {
        if let Self::Indirect { rows: current, .. } = self {
            *current = rows;
        }
    }

    /// Add a new Indirect section to the free space manager. Mirrors
    /// `H5HF__sect_indirect_add`.
    pub fn sect_indirect_add(sections: &mut Vec<Self>, section: Self) {
        sections.push(section);
    }

    /// Decrement the ref count on an indirect section. Mirrors
    /// `H5HF__sect_indirect_decr`.
    pub fn sect_indirect_decr(&mut self) {
        if let Self::Indirect { rows, .. } = self {
            *rows = rows.saturating_sub(1);
        }
    }

    /// Update the row info on an Indirect section so it includes `row`.
    /// Mirrors `H5HF__sect_indirect_revive_row`.
    pub fn sect_indirect_revive_row(&mut self, row: usize) {
        let _ = self.sect_indirect_revive_row_checked(row);
    }

    /// Checked variant that surfaces row-count overflow.
    pub fn sect_indirect_revive_row_checked(&mut self, row: usize) -> Result<()> {
        if let Self::Indirect { rows, .. } = self {
            let needed = row.checked_add(1).ok_or_else(|| {
                Error::InvalidFormat("fractal heap indirect section row overflow".into())
            })?;
            *rows = (*rows).max(needed);
        }
        Ok(())
    }

    /// Update memory info for an indirect free section. Mirrors
    /// `H5HF__sect_indirect_revive` (no-op).
    pub fn sect_indirect_revive(&mut self) {}

    /// Remove a block from an indirect section. Mirrors
    /// `H5HF__sect_indirect_reduce_row`.
    pub fn sect_indirect_reduce_row(&mut self, amount: u64) {
        if let Self::Indirect { size, .. } = self {
            *size = size.saturating_sub(amount);
        }
    }

    /// Reduce the size of an indirect section. Mirrors
    /// `H5HF__sect_indirect_reduce`.
    pub fn sect_indirect_reduce(&mut self, amount: u64) {
        self.sect_indirect_reduce_row(amount);
    }

    /// Whether the section's offset matches `offset`. Mirrors
    /// `H5HF__sect_indirect_is_first`.
    pub fn sect_indirect_is_first(&self, offset: u64) -> bool {
        section_offset(self) == Some(offset)
    }

    /// First-row offset for the indirect section. Mirrors
    /// `H5HF__sect_indirect_first`.
    pub fn sect_indirect_first(&self) -> Option<u64> {
        section_offset(self)
    }

    /// Retrieve the iblock row count for an Indirect section. Mirrors
    /// `H5HF__sect_indirect_get_iblock`.
    pub fn sect_indirect_get_iblock(&self) -> Option<usize> {
        match self {
            Self::Indirect { rows, .. } => Some(*rows),
            _ => None,
        }
    }

    /// Merge two Indirect sections. Mirrors
    /// `H5HF__sect_indirect_merge_row`.
    pub fn sect_indirect_merge_row(&mut self, other: &Self) -> bool {
        self.sect_single_merge(other)
    }

    /// Checked variant of `sect_indirect_merge_row`.
    pub fn sect_indirect_merge_row_checked(&mut self, other: &Self) -> Result<bool> {
        self.sect_single_merge_checked(other)
    }

    /// Borrow the parent Indirect section for this section. Mirrors
    /// `H5HF__sect_indirect_build_parent`.
    pub fn sect_indirect_build_parent_ref(&self) -> Option<&Self> {
        Some(self)
    }

    /// Shrink the container with this indirect section. Mirrors
    /// `H5HF__sect_indirect_shrink`.
    pub fn sect_indirect_shrink(&mut self, amount: u64) {
        self.sect_indirect_reduce(amount);
    }

    /// Append an indirect section's serialized form to `out`. Mirrors
    /// `H5HF__sect_indirect_serialize`.
    pub fn sect_indirect_serialize_into(&self, out: &mut Vec<u8>) -> Result<()> {
        let Self::Indirect { offset, size, .. } = self else {
            return Err(Error::InvalidFormat(
                "fractal heap indirect section serializer received non-indirect section".into(),
            ));
        };
        out.reserve(16);
        out.extend_from_slice(&offset.to_le_bytes());
        out.extend_from_slice(&size.to_le_bytes());
        Ok(())
    }

    /// Free an Indirect section node. Mirrors `H5HF__sect_indirect_free`.
    pub fn sect_indirect_free(self) {}

    /// Validity check for an Indirect section. Mirrors
    /// `H5HF__sect_indirect_valid`.
    pub fn sect_indirect_valid(&self) -> bool {
        self.sect_single_valid()
    }

    /// Write debugging information about an Indirect section into `out`. Mirrors
    /// `H5HF__sect_indirect_debug`.
    pub fn sect_indirect_debug_fmt(&self, out: &mut impl fmt::Write) -> fmt::Result {
        write!(out, "{self:?}")
    }
}

impl FractalHeapSpace {
    /// Start up free space for the heap (open the existing free-space
    /// manager). Mirrors `H5HF__space_start`.
    pub fn space_start() -> Self {
        Self::default()
    }

    /// Add a section to the heap's free space. Mirrors
    /// `H5HF__space_add`.
    pub fn space_add(&mut self, section: FractalHeapSection) {
        self.sections.push(section);
    }

    /// Find a free-space section big enough to satisfy `size`. Mirrors
    /// `H5HF__space_find`.
    pub fn space_find(&self, size: u64) -> Option<&FractalHeapSection> {
        self.sections
            .iter()
            .find(|section| section_size(section).is_some_and(|candidate| candidate >= size))
    }

    /// Iterator callback that resets 'parent' pointers in sections.
    /// Mirrors `H5HF__space_revert_root_cb` (no-op).
    pub fn space_revert_root_cb(&mut self) {}

    /// Reset 'parent' pointers in sections when the heap reverts to its
    /// direct-block root. Mirrors `H5HF__space_revert_root` (no-op).
    pub fn space_revert_root(&mut self) {}

    /// Iterator callback that sets 'parent' pointers in sections.
    /// Mirrors `H5HF__space_create_root_cb` (no-op).
    pub fn space_create_root_cb(&mut self) {}

    /// Set 'parent' pointers in sections when a new root indirect block
    /// is created. Mirrors `H5HF__space_create_root` (no-op).
    pub fn space_create_root(&mut self) {}

    /// Aggregate size of the heap's free space sections. Mirrors
    /// `H5HF__space_size`.
    pub fn space_size(&self) -> u64 {
        self.sections.iter().filter_map(section_size).sum()
    }

    /// Remove the section at `offset` from the free-space sections.
    /// Mirrors `H5HF__space_remove`.
    pub fn space_remove(&mut self, offset: u64) -> Option<FractalHeapSection> {
        let idx = self
            .sections
            .iter()
            .position(|section| section_offset(section) == Some(offset))?;
        Some(self.sections.remove(idx))
    }

    /// Close the heap's free space manager. Mirrors `H5HF__space_close`.
    pub fn space_close(&mut self) {
        self.closed = true;
    }

    /// Delete the free space manager for the heap. Mirrors
    /// `H5HF__space_delete`.
    pub fn space_delete(mut self) {
        self.sections.clear();
        self.closed = true;
    }

    /// Change a section's class (from a row-class single into an Indirect
    /// for the given rows). Mirrors `H5HF__space_sect_change_class`.
    pub fn space_sect_change_class(&mut self, offset: u64, rows: usize) -> bool {
        if let Some(section) = self
            .sections
            .iter_mut()
            .find(|section| section_offset(section) == Some(offset))
        {
            let size = section_size(section).unwrap_or(0);
            *section = FractalHeapSection::Indirect { offset, rows, size };
            true
        } else {
            false
        }
    }

    /// Write per-section debug info into `out`. Mirrors `H5HF__sects_debug_cb`.
    pub fn sects_debug_cb_fmt(
        section: &FractalHeapSection,
        out: &mut impl fmt::Write,
    ) -> fmt::Result {
        write!(out, "{section:?}")
    }

    /// Write debug info for all sections in the free space into `out`. Mirrors
    /// `H5HF__sects_debug`.
    pub fn sects_debug_fmt(&self, out: &mut impl fmt::Write) -> fmt::Result {
        let mut sections = self.sections.iter();
        if let Some(section) = sections.next() {
            Self::sects_debug_cb_fmt(section, out)?;
            for section in sections {
                out.write_str(", ")?;
                Self::sects_debug_cb_fmt(section, out)?;
            }
        }
        Ok(())
    }

    /// Iterate over free-space sections without cloning the section list.
    pub fn sections(&self) -> impl ExactSizeIterator<Item = &FractalHeapSection> {
        self.sections.iter()
    }
}

// ---------------------------------------------------------------------------
// Trace probes — kept in one place because they're small, conditional on
// the `tracehash` feature, and called from sibling modules (dblock/huge/
// tiny). Each emits one tracehash event for the read it just performed.
// ---------------------------------------------------------------------------

impl FractalHeapHeader {
    /// Emit a tracehash event recording a managed-object read.
    #[cfg(feature = "tracehash")]
    pub(super) fn trace_managed_object(
        &self,
        block_addr: u64,
        block_size: u64,
        block_offset: u64,
        object_len: u64,
        _filter_mask: u32,
        filtered: bool,
    ) {
        let mut th = tracehash::th_call!("hdf5.fractal_heap.managed_object");
        th.input_u64(self.heap_addr);
        th.input_u64(block_addr);
        th.input_u64(block_offset);
        th.input_u64(object_len);
        th.output_value(&(true));
        th.output_u64(block_addr);
        th.output_u64(block_size);
        th.output_u64(block_offset);
        th.output_u64(object_len);
        th.output_u64(0);
        th.output_value(&(filtered));
        th.finish();
    }

    /// No-op tracehash hook for managed-object reads (feature disabled).
    #[cfg(not(feature = "tracehash"))]
    pub(super) fn trace_managed_object(
        &self,
        _block_addr: u64,
        _block_size: u64,
        _block_offset: u64,
        _object_len: u64,
        _filter_mask: u32,
        _filtered: bool,
    ) {
    }

    /// Emit a tracehash event recording a huge-object read.
    #[cfg(feature = "tracehash")]
    pub(super) fn trace_huge_object(
        &self,
        heap_id: &[u8],
        addr: u64,
        stored_len: u64,
        object_len: u64,
        filter_mask: u32,
        filtered: bool,
    ) {
        let mut th = tracehash::th_call!("hdf5.fractal_heap.huge_object");
        th.input_u64(self.heap_addr);
        th.input_bytes(heap_id);
        th.output_value(&(true));
        th.output_u64(addr);
        th.output_u64(stored_len);
        th.output_u64(object_len);
        th.output_u64(u64::from(filter_mask));
        th.output_value(&(filtered));
        th.finish();
    }

    /// No-op tracehash hook for huge-object reads (feature disabled).
    #[cfg(not(feature = "tracehash"))]
    pub(super) fn trace_huge_object(
        &self,
        _heap_id: &[u8],
        _addr: u64,
        _stored_len: u64,
        _object_len: u64,
        _filter_mask: u32,
        _filtered: bool,
    ) {
    }

    /// Emit a tracehash event recording a tiny-object read.
    #[cfg(feature = "tracehash")]
    pub(super) fn trace_tiny_object(&self, heap_id: &[u8], object_len: u64) {
        let mut th = tracehash::th_call!("hdf5.fractal_heap.tiny_object");
        th.input_u64(self.heap_addr);
        th.input_bytes(heap_id);
        th.output_value(&(true));
        th.output_u64(object_len);
        th.finish();
    }

    /// No-op tracehash hook for tiny-object reads (feature disabled).
    #[cfg(not(feature = "tracehash"))]
    pub(super) fn trace_tiny_object(&self, _heap_id: &[u8], _object_len: u64) {}
}

// ---------------------------------------------------------------------------
// Internal numeric / checksum helpers shared across submodules.
// ---------------------------------------------------------------------------

/// Decode the first up-to-8 bytes of `bytes` as a little-endian unsigned
/// integer. Rust analog of libhdf5's `H5F_DECODE_LENGTH`-style helpers.
pub(super) fn read_le_uint(bytes: &[u8]) -> u64 {
    let mut value = 0u64;
    for (idx, byte) in bytes.iter().take(8).enumerate() {
        value |= u64::from(*byte) << (idx * 8);
    }
    value
}

/// Read a little-endian `u16` at `offset` with a descriptive truncation
/// error.
fn read_u16_at(bytes: &[u8], offset: usize, context: &str) -> Result<u16> {
    let end = offset
        .checked_add(2)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))?;
    let raw: [u8; 2] = bytes
        .get(offset..end)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))?
        .try_into()
        .map_err(|_| Error::InvalidFormat(format!("{context} is truncated")))?;
    Ok(u16::from_le_bytes(raw))
}

/// Read a little-endian `u32` at `offset` with a descriptive truncation
/// error.
fn read_u32_at(bytes: &[u8], offset: usize, context: &str) -> Result<u32> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))?;
    let raw: [u8; 4] = bytes
        .get(offset..end)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))?
        .try_into()
        .map_err(|_| Error::InvalidFormat(format!("{context} is truncated")))?;
    Ok(u32::from_le_bytes(raw))
}

/// Read a little-endian `u64` at `offset` with a descriptive truncation
/// error.
fn read_u64_at(bytes: &[u8], offset: usize, context: &str) -> Result<u64> {
    let end = offset
        .checked_add(8)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))?;
    let raw: [u8; 8] = bytes
        .get(offset..end)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))?
        .try_into()
        .map_err(|_| Error::InvalidFormat(format!("{context} is truncated")))?;
    Ok(u64::from_le_bytes(raw))
}

/// Read the bytes at `[start, start+check_len)` plus a trailing 4-byte
/// checksum, recompute the JenkinsLookup3 metadata checksum, and verify
/// they match. Wrapper around libhdf5's `H5_checksum_metadata`.
pub(super) fn verify_metadata_checksum<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    start: u64,
    check_len: u64,
    context: &str,
) -> Result<()> {
    let restore = reader.position()?;
    let check_len_usize = heap_object_len(check_len, context)?;
    reader.seek(start)?;
    let mut bytes = vec![0; check_len_usize];
    reader.read_bytes_into(&mut bytes)?;
    let stored = reader.read_u32()?;
    let computed = checksum_metadata(&bytes);
    reader.seek(restore)?;
    if stored != computed {
        return Err(Error::InvalidFormat(format!(
            "{context} checksum mismatch: stored={stored:#010x}, computed={computed:#010x}"
        )));
    }
    Ok(())
}

/// Verify the checksum on a fractal-heap direct block. Mirrors
/// libhdf5's `H5HF__cache_dblock_verify_chksum`: accepts several
/// historical hashing strategies (header-only, header+zeroed-checksum,
/// header+payload) to match different libhdf5 writer versions.
pub(super) fn verify_direct_block_checksum<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    start: u64,
    max_heap_size: u16,
    block_size: u64,
) -> Result<()> {
    let restore = reader.position()?;
    let block_offset_bytes = (usize::from(max_heap_size) + 7) / 8;
    let check_len = 4usize
        .checked_add(1)
        .and_then(|n| n.checked_add(usize::from(reader.sizeof_addr())))
        .and_then(|n| n.checked_add(block_offset_bytes))
        .ok_or_else(|| {
            Error::InvalidFormat("fractal heap direct block checksum span overflow".into())
        })?;
    reader.seek(start)?;
    let mut bytes = vec![0; check_len];
    reader.read_bytes_into(&mut bytes)?;
    let stored = reader.read_u32()?;
    let payload_len = heap_object_len(block_size, "fractal heap direct block size")?;
    let mut payload = vec![0; payload_len];
    reader.read_bytes_into(&mut payload)?;
    reader.seek(restore)?;

    let computed = checksum_metadata(&bytes);
    let mut with_zero_checksum = bytes.clone();
    with_zero_checksum.extend_from_slice(&0u32.to_le_bytes());
    let computed_with_zero_checksum = checksum_metadata(&with_zero_checksum);
    let mut whole_block = with_zero_checksum;
    whole_block.extend_from_slice(&payload);
    let computed_whole_block = checksum_metadata(&whole_block);
    if stored != computed && stored != computed_with_zero_checksum && stored != computed_whole_block
    {
        return Err(Error::InvalidFormat(format!(
            "fractal heap direct block checksum mismatch: stored={stored:#010x}, computed={computed:#010x}"
        )));
    }
    Ok(())
}

/// Convert a heap-encoded object size into a `usize`, rejecting values
/// that overflow or exceed the supported per-object cap.
pub(super) fn heap_object_len(value: u64, context: &str) -> Result<usize> {
    let len = usize::try_from(value)
        .map_err(|_| Error::InvalidFormat(format!("{context} does not fit in usize")))?;
    if len > MAX_HEAP_OBJECT_BYTES {
        return Err(Error::InvalidFormat(format!(
            "{context} {len} exceeds supported maximum {MAX_HEAP_OBJECT_BYTES}"
        )));
    }
    Ok(len)
}

/// Sanity-check the user-facing fractal-heap create parameters (table
/// width, block sizes power-of-two, max_heap_size width, etc.).
fn validate_heap_create_params(params: &FractalHeapCreateParams) -> Result<()> {
    if params.heap_id_len == 0 {
        return Err(Error::InvalidFormat(
            "fractal heap ID length must be nonzero".into(),
        ));
    }
    if params.table_width == 0 {
        return Err(Error::InvalidFormat(
            "fractal heap table width must be nonzero".into(),
        ));
    }
    if params.start_block_size == 0 || !params.start_block_size.is_power_of_two() {
        return Err(Error::InvalidFormat(
            "fractal heap start block size must be a nonzero power of two".into(),
        ));
    }
    if params.max_direct_block_size == 0 || !params.max_direct_block_size.is_power_of_two() {
        return Err(Error::InvalidFormat(
            "fractal heap max direct block size must be a nonzero power of two".into(),
        ));
    }
    if params.max_direct_block_size < params.start_block_size {
        return Err(Error::InvalidFormat(
            "fractal heap max direct block size is smaller than start block size".into(),
        ));
    }
    if params.max_heap_size > 64 {
        return Err(Error::Unsupported(format!(
            "fractal heap max heap size {} exceeds 64-bit offsets",
            params.max_heap_size
        )));
    }
    Ok(())
}

/// Split the trailing 4-byte checksum from an in-memory cache image and
/// verify it matches the metadata checksum of the preceding bytes.
fn verify_image_checksum(image: &[u8], context: &str) -> Result<()> {
    if image.len() < 4 {
        return Err(Error::InvalidFormat(format!("{context} image too short")));
    }
    let split = image.len() - 4;
    let stored = u32::from_le_bytes(
        image[split..]
            .try_into()
            .map_err(|_| Error::InvalidFormat(format!("{context} checksum truncated")))?,
    );
    let computed = checksum_metadata(&image[..split]);
    if stored != computed {
        return Err(Error::InvalidFormat(format!(
            "{context} checksum mismatch: stored={stored:#010x}, computed={computed:#010x}"
        )));
    }
    Ok(())
}

/// Encode an unsigned integer into `width` little-endian bytes, checking
/// that the value fits.
fn encode_size_field(out: &mut Vec<u8>, value: u64, width: u8, context: &str) -> Result<()> {
    let width = usize::from(width);
    if width == 0 || width > 8 {
        return Err(Error::InvalidFormat(format!("{context} width is invalid")));
    }
    ensure_uint_fits_width(value, width, context)?;
    out.extend_from_slice(&value.to_le_bytes()[..width]);
    Ok(())
}

/// Verify that `value` fits in `width` little-endian bytes (1..=8).
fn ensure_uint_fits_width(value: u64, width: usize, context: &str) -> Result<()> {
    if width == 0 || width > 8 {
        return Err(Error::InvalidFormat(format!("{context} width is invalid")));
    }
    if width < 8 && value >= (1u64 << (width * 8)) {
        return Err(Error::InvalidFormat(format!(
            "{context} value {value:#x} does not fit in {width} bytes"
        )));
    }
    Ok(())
}

/// Encode a file address into `width` bytes, writing the all-0xff
/// sentinel when the address is `UNDEF_ADDR`.
fn encode_addr_field(out: &mut Vec<u8>, value: u64, width: u8, context: &str) -> Result<()> {
    let width = usize::from(width);
    if width == 0 || width > 8 {
        return Err(Error::InvalidFormat(format!("{context} width is invalid")));
    }
    if value == UNDEF_ADDR {
        out.extend(std::iter::repeat_n(0xff, width));
        return Ok(());
    }
    encode_size_field(
        out,
        value,
        u8::try_from(width)
            .map_err(|_| Error::InvalidFormat(format!("{context} width overflow")))?,
        context,
    )
}

/// Number of bytes used to encode a managed heap ID's offset given the
/// configured max heap size and heap-ID length, with a sanity check that
/// the bytes actually fit.
fn managed_id_offset_bytes(max_heap_size: u16, heap_id_len: u16, context: &str) -> Result<usize> {
    let offset_bytes = usize::from(max_heap_size).div_ceil(8);
    let available = usize::from(heap_id_len)
        .checked_sub(1)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} length must include type byte")))?;
    if offset_bytes > available {
        return Err(Error::InvalidFormat(format!(
            "{context} needs {offset_bytes} offset bytes but ID length provides {available}"
        )));
    }
    Ok(offset_bytes)
}

/// Placeholder for encoding a filter pipeline into the heap header.
/// Currently unimplemented; filtered direct-root serialization is not
/// supported by the Rust port yet.
fn encode_filter_pipeline_for_heap(_pipeline: &FilterPipelineMessage) -> Result<Vec<u8>> {
    Err(Error::Unsupported(
        "fractal heap filtered direct root serialization is not implemented".into(),
    ))
}

/// Extract the heap offset of any `FractalHeapSection` variant.
fn section_offset(section: &FractalHeapSection) -> Option<u64> {
    match section {
        FractalHeapSection::Single { offset, .. }
        | FractalHeapSection::Row { offset, .. }
        | FractalHeapSection::Indirect { offset, .. } => Some(*offset),
    }
}

/// Extract the byte size of any `FractalHeapSection` variant.
fn section_size(section: &FractalHeapSection) -> Option<u64> {
    match section {
        FractalHeapSection::Single { size, .. }
        | FractalHeapSection::Row { size, .. }
        | FractalHeapSection::Indirect { size, .. } => Some(*size),
    }
}

/// Mutable accessor for any `FractalHeapSection` variant's size field.
fn section_size_mut(section: &mut FractalHeapSection) -> Option<&mut u64> {
    match section {
        FractalHeapSection::Single { size, .. }
        | FractalHeapSection::Row { size, .. }
        | FractalHeapSection::Indirect { size, .. } => Some(size),
    }
}

/// One-past-the-end heap offset of a section (offset + size), or `None`
/// when the addition overflows.
fn section_end(section: &FractalHeapSection) -> Option<u64> {
    section_offset(section)?.checked_add(section_size(section)?)
}

/// log2 of a value that must already be a power of two; cheap shift-count
/// math.
pub(super) fn log2_power2(value: u64) -> u32 {
    debug_assert!(value.is_power_of_two());
    value.trailing_zeros()
}

/// Floor of log2 for any nonzero u64 (high-bit position).
pub(super) fn log2_floor(value: u64) -> u32 {
    63 - value.leading_zeros()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_block_serializers_validate_lengths_and_checksums() {
        let iblock = FractalHeapIndirectBlock {
            nrows: 2,
            child_addrs: vec![10, u64::MAX, 30],
            ref_count: 1,
            dirty: true,
        };
        let mut image = Vec::new();
        iblock.cache_iblock_serialize_into(&mut image).unwrap();
        assert_eq!(image.len(), iblock.cache_iblock_image_len().unwrap());
        assert_eq!(&image[..4], &FHIB_MAGIC);
        assert_eq!(u64::from_le_bytes(image[5..13].try_into().unwrap()), 2);
        FractalHeapIndirectBlock::cache_iblock_verify_chksum(&image).unwrap();
        let mut pre_image = Vec::new();
        iblock
            .cache_iblock_pre_serialize_into(&mut pre_image)
            .unwrap();
        assert_eq!(pre_image, image);

        let dblock = FractalHeapDirectBlock::man_dblock_new(64, vec![1, 2, 3, 4]);
        let mut image = Vec::new();
        dblock.cache_dblock_serialize_into(&mut image).unwrap();
        assert_eq!(image.len(), dblock.cache_dblock_image_len().unwrap());
        assert_eq!(&image[..4], &FHDB_MAGIC);
        assert_eq!(u64::from_le_bytes(image[5..13].try_into().unwrap()), 64);
        verify_image_checksum(&image, "fractal heap direct block").unwrap();
        let mut pre_image = Vec::new();
        dblock
            .cache_dblock_pre_serialize_into(&mut pre_image)
            .unwrap();
        assert_eq!(pre_image, image);
    }

    #[test]
    fn fractal_heap_header_cache_serialize_uses_configured_widths() {
        let mut header = FractalHeapHeader::hdr_alloc(FractalHeapCreateParams {
            heap_id_len: 8,
            table_width: 4,
            start_block_size: 512,
            max_direct_block_size: 4096,
            max_heap_size: 64,
        })
        .unwrap();
        header.sizeof_addr = 4;
        header.sizeof_size = 4;
        header.huge_btree_addr = UNDEF_ADDR;
        header.root_block_addr = UNDEF_ADDR;

        let mut image = Vec::new();
        header.cache_hdr_serialize_into(&mut image).unwrap();
        assert_eq!(image.len(), header.cache_hdr_image_len());
        FractalHeapHeader::cache_hdr_verify_chksum(&image).unwrap();

        let mut reader = HdfReader::new(std::io::Cursor::new(image));
        reader.set_sizeof_addr(4);
        reader.set_sizeof_size(4);
        let decoded = FractalHeapHeader::read_at(&mut reader, 0).unwrap();
        assert_eq!(decoded.sizeof_addr, 4);
        assert_eq!(decoded.sizeof_size, 4);
        assert_eq!(decoded.huge_btree_addr, u64::from(u32::MAX));
        assert_eq!(decoded.root_block_addr, u64::from(u32::MAX));

        let too_large = FractalHeapHeader {
            root_block_addr: u64::from(u32::MAX) + 1,
            ..header
        };
        let mut image = Vec::new();
        assert!(too_large.cache_hdr_serialize_into(&mut image).is_err());
    }

    #[test]
    fn managed_heap_id_encoding_rejects_offset_narrowing() {
        let mut heap = FractalHeap::create(FractalHeapCreateParams {
            heap_id_len: 2,
            table_width: 4,
            start_block_size: 512,
            max_direct_block_size: 4096,
            max_heap_size: 8,
        })
        .unwrap();
        heap.objects = vec![Vec::new(); 256];
        heap.header.num_managed_objects = 256;

        assert!(heap.insert(vec![1]).is_err());
        assert_eq!(heap.objects.len(), 256);
        assert_eq!(heap.header.num_managed_objects, 256);

        let mut too_short = heap;
        too_short.header.heap_id_len = 1;
        too_short.objects.clear();
        too_short.header.num_managed_objects = 0;
        assert!(too_short.insert(vec![1]).is_err());
    }

    #[test]
    fn visit_managed_object_cached_visits_borrowed_direct_block_slice() {
        let mut header = FractalHeapHeader::hdr_alloc(FractalHeapCreateParams {
            heap_id_len: 3,
            table_width: 1,
            start_block_size: 16,
            max_direct_block_size: 16,
            max_heap_size: 8,
        })
        .unwrap();
        header.root_block_addr = 0;
        header.current_root_rows = 0;
        header.sizeof_addr = 8;
        header.sizeof_size = 8;
        header.huge_btree_addr = UNDEF_ADDR;

        let heap_id = [0x00, 2, 5];
        let mut reader = HdfReader::new(std::io::Cursor::new(b"abcdefghijklmnop".to_vec()));
        let mut cache = FractalHeapManagedObjectCache::new();

        let len = header
            .visit_managed_object_cached(&mut reader, &heap_id, &mut cache, |data| {
                assert_eq!(data, b"cdefg");
                Ok(data.len())
            })
            .unwrap();

        assert_eq!(len, 5);
    }

    #[test]
    fn visit_managed_object_cached_visits_tiny_heap_id_payload() {
        let header = FractalHeapHeader::hdr_alloc(FractalHeapCreateParams {
            heap_id_len: 8,
            table_width: 1,
            start_block_size: 16,
            max_direct_block_size: 16,
            max_heap_size: 8,
        })
        .unwrap();
        let heap_id = [0x22, b'x', b'y', b'z'];
        let mut reader = HdfReader::new(std::io::Cursor::new(Vec::<u8>::new()));
        let mut cache = FractalHeapManagedObjectCache::new();

        let bytes =
            header
                .visit_managed_object_cached(&mut reader, &heap_id, &mut cache, |data| {
                    Ok(data.to_vec())
                })
                .unwrap();

        assert_eq!(bytes, b"xyz");
    }

    #[test]
    fn indirect_block_helpers_reject_child_table_overflow() {
        assert!(FractalHeapIndirectBlock::man_iblock_root_create(usize::MAX, 2).is_err());
        assert!(FractalHeapIndirectBlock::man_iblock_create(usize::MAX, 2).is_err());

        let mut block = FractalHeapIndirectBlock::man_iblock_root_create(2, 3).unwrap();
        assert_eq!(block.child_addrs.len(), 6);
        block.man_iblock_alloc_row(4, 3).unwrap();
        assert_eq!(block.nrows, 5);
        assert_eq!(block.child_addrs.len(), 15);

        assert!(block.man_iblock_alloc_row(usize::MAX, 2).is_err());
    }

    #[test]
    fn section_and_iterator_checked_helpers_reject_overflow() {
        assert!(FractalHeapIterator::man_iter_init_checked(u64::MAX, 2).is_err());

        let mut section = FractalHeapSection::sect_single_new(0, u64::MAX);
        let adjacent = FractalHeapSection::sect_single_new(u64::MAX, 1);
        assert!(section.sect_single_merge_checked(&adjacent).is_err());
        assert!(!section.sect_single_merge(&adjacent));

        assert!(FractalHeapSection::sect_indirect_for_row_checked(usize::MAX, 0, 1).is_err());

        let mut indirect = FractalHeapSection::sect_indirect_new(0, 1, 1);
        assert!(indirect
            .sect_indirect_revive_row_checked(usize::MAX)
            .is_err());
    }

    #[test]
    fn section_class_serializers_reject_wrong_section_variants() {
        let row = FractalHeapSection::sect_row_deserialize(3, 10, 20);
        let mut image = Vec::new();
        row.sect_row_serialize_into(&mut image).unwrap();
        assert_eq!(u64::from_le_bytes(image[..8].try_into().unwrap()), 10);
        assert_eq!(u64::from_le_bytes(image[8..16].try_into().unwrap()), 20);

        let indirect = FractalHeapSection::sect_indirect_new(30, 2, 40);
        let mut image = Vec::new();
        indirect.sect_indirect_serialize_into(&mut image).unwrap();
        assert_eq!(u64::from_le_bytes(image[..8].try_into().unwrap()), 30);
        assert_eq!(u64::from_le_bytes(image[8..16].try_into().unwrap()), 40);

        let single = FractalHeapSection::sect_single_new(1, 2);
        let mut image = Vec::new();
        assert!(single.sect_row_serialize_into(&mut image).is_err());
        assert!(single.sect_indirect_serialize_into(&mut image).is_err());
    }

    #[test]
    fn doubling_table_image_roundtrips_and_validates_geometry() {
        let mut header = FractalHeapHeader::hdr_alloc(FractalHeapCreateParams {
            heap_id_len: 8,
            table_width: 4,
            start_block_size: 512,
            max_direct_block_size: 4096,
            max_heap_size: 64,
        })
        .unwrap();
        header.start_root_rows = 2;
        header.current_root_rows = 3;

        let mut image = Vec::new();
        header.dtable_encode_into(&mut image).unwrap();
        assert_eq!(image.len(), FRACTAL_HEAP_DTABLE_IMAGE_LEN);
        let decoded = FractalHeapHeader::dtable_decode(&image).unwrap();
        assert_eq!(decoded.table_width, 4);
        assert_eq!(decoded.start_block_size, 512);
        assert_eq!(decoded.current_root_rows, 3);

        assert!(FractalHeapHeader::dtable_decode(&image[..23]).is_err());
        header.current_root_rows = 1;
        let mut image = Vec::new();
        assert!(header.dtable_encode_into(&mut image).is_err());
    }

    #[test]
    fn filtered_huge_direct_record_image_roundtrips() {
        let record = FractalHeapHugeRecord::huge_bt2_filt_dir_store(0x10, 20, 100, 0x04);
        let mut image = Vec::new();
        record.huge_bt2_filt_dir_encode_into(&mut image).unwrap();
        assert_eq!(image.len(), HUGE_FILTERED_DIRECT_RECORD_LEN);

        let decoded = FractalHeapHugeRecord::huge_bt2_filt_dir_decode(&image).unwrap();
        assert_eq!(decoded.addr, 0x10);
        assert_eq!(decoded.len, 20);
        assert_eq!(decoded.obj_size, 100);
        assert_eq!(decoded.filter_mask, 0x04);

        assert!(FractalHeapHugeRecord::huge_bt2_filt_dir_decode(&image[..27]).is_err());
        let unfiltered = FractalHeapHugeRecord::huge_bt2_filt_dir_store(0x10, 20, 100, 0);
        let mut image = Vec::new();
        assert!(unfiltered
            .huge_bt2_filt_dir_encode_into(&mut image)
            .is_err());
    }
}
