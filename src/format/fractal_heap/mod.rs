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

use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::format::checksum::checksum_metadata;
use crate::format::messages::filter_pipeline::FilterPipelineMessage;
use crate::io::reader::HdfReader;

/// Fractal heap header magic: "FRHP"
pub(super) const FRHP_MAGIC: [u8; 4] = [b'F', b'R', b'H', b'P'];
/// Direct block magic: "FHDB" (kept for reference; we currently read by
/// offset rather than by magic).
#[allow(dead_code)]
pub(super) const FHDB_MAGIC: [u8; 4] = [b'F', b'H', b'D', b'B'];
/// Indirect block magic: "FHIB"
pub(super) const FHIB_MAGIC: [u8; 4] = [b'F', b'H', b'I', b'B'];
const MAX_HEAP_OBJECT_BYTES: usize = 4 * 1024 * 1024 * 1024;

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

    pub fn hdr_finish_init_phase1(&mut self) -> Result<()> {
        self.validate_header()
    }

    pub fn hdr_finish_init_phase2(&mut self) -> Result<()> {
        self.validate_header()
    }

    pub fn hdr_finish_init(&mut self) -> Result<()> {
        self.hdr_finish_init_phase1()?;
        self.hdr_finish_init_phase2()
    }

    pub fn hdr_create(params: FractalHeapCreateParams) -> Result<Self> {
        let mut header = Self::hdr_alloc(params)?;
        header.hdr_finish_init()?;
        Ok(header)
    }

    pub fn hdr_protect(&self) -> Self {
        self.clone()
    }

    pub fn hdr_incr(&mut self) -> Result<()> {
        self.num_managed_objects = self
            .num_managed_objects
            .checked_add(1)
            .ok_or_else(|| Error::InvalidFormat("fractal heap object count overflow".into()))?;
        Ok(())
    }

    pub fn hdr_decr(&mut self) -> Result<()> {
        self.num_managed_objects = self
            .num_managed_objects
            .checked_sub(1)
            .ok_or_else(|| Error::InvalidFormat("fractal heap object count underflow".into()))?;
        Ok(())
    }

    pub fn hdr_fuse_incr(&mut self) -> Result<()> {
        self.hdr_incr()
    }

    pub fn hdr_fuse_decr(&mut self) -> Result<()> {
        self.hdr_decr()
    }

    pub fn hdr_dirty(&mut self) {}

    pub fn hdr_adj_free(&mut self, _delta: i64) {}

    pub fn hdr_adjust_heap(&mut self, root_addr: u64, root_rows: u16) {
        self.root_block_addr = root_addr;
        self.current_root_rows = root_rows;
    }

    pub fn hdr_inc_alloc(&mut self, bytes: u64) -> Result<()> {
        let next = self
            .start_block_size
            .checked_add(bytes)
            .ok_or_else(|| Error::InvalidFormat("fractal heap allocation overflow".into()))?;
        self.start_block_size = next.min(self.max_direct_block_size);
        Ok(())
    }

    pub fn hdr_start_iter(&self) -> FractalHeapIterator {
        FractalHeapIterator::man_iter_init(0, self.num_managed_objects)
    }

    pub fn hdr_reset_iter(iter: &mut FractalHeapIterator) {
        iter.man_iter_reset();
    }

    pub fn hdr_skip_blocks(iter: &mut FractalHeapIterator, blocks: usize) {
        iter.index = iter.index.saturating_add(blocks).min(iter.offsets.len());
    }

    pub fn hdr_update_iter(iter: &mut FractalHeapIterator, offsets: Vec<u64>) {
        iter.offsets = offsets;
        iter.index = 0;
    }

    pub fn hdr_inc_iter(iter: &mut FractalHeapIterator) {
        let _ = iter.man_iter_next();
    }

    pub fn hdr_reverse_iter(iter: &mut FractalHeapIterator) {
        iter.offsets.reverse();
        iter.index = 0;
    }

    pub fn hdr_empty(&self) -> bool {
        self.num_managed_objects == 0
    }

    pub fn hdr_free(self) {}

    pub fn hdr_delete(self) {}

    pub fn cache_hdr_get_initial_load_size() -> usize {
        4
    }

    pub fn cache_hdr_get_final_load_size(&self) -> usize {
        self.cache_hdr_image_len()
    }

    pub fn cache_hdr_verify_chksum(image: &[u8]) -> Result<()> {
        verify_image_checksum(image, "fractal heap header")
    }

    pub fn cache_hdr_image_len(&self) -> usize {
        4 + 1 + 2 + 2 + 1 + 4 + 8 + 8 + 8 + 8 + 8 + 8 + 2 + 8 + 8 + 2 + 2 + 8 + 2 + 4
    }

    pub fn cache_hdr_pre_serialize(&self) -> Result<Vec<u8>> {
        self.cache_hdr_serialize()
    }

    pub fn cache_hdr_serialize(&self) -> Result<Vec<u8>> {
        self.validate_header()?;
        let mut out = Vec::with_capacity(self.cache_hdr_image_len());
        out.extend_from_slice(&FRHP_MAGIC);
        out.push(0);
        out.extend_from_slice(&self.heap_id_len.to_le_bytes());
        out.extend_from_slice(&self.io_filter_len.to_le_bytes());
        out.push(self.flags);
        out.extend_from_slice(&self.max_managed_obj_size.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        out.extend_from_slice(&self.huge_btree_addr.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        out.extend_from_slice(&u64::from(self.num_managed_objects > 0).to_le_bytes());
        out.extend_from_slice(&self.num_managed_objects.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        out.extend_from_slice(&self.table_width.to_le_bytes());
        out.extend_from_slice(&self.start_block_size.to_le_bytes());
        out.extend_from_slice(&self.max_direct_block_size.to_le_bytes());
        out.extend_from_slice(&self.max_heap_size.to_le_bytes());
        out.extend_from_slice(&self.start_root_rows.to_le_bytes());
        out.extend_from_slice(&self.root_block_addr.to_le_bytes());
        out.extend_from_slice(&self.current_root_rows.to_le_bytes());
        let checksum = checksum_metadata(&out);
        out.extend_from_slice(&checksum.to_le_bytes());
        Ok(out)
    }

    pub fn cache_hdr_free_icr(_image: Vec<u8>) {}

    pub fn cache_verify_hdr_descendants_clean(&self) -> bool {
        true
    }

    pub fn hdr_size(&self) -> usize {
        self.cache_hdr_image_len()
    }

    pub fn hdr_print(&self) -> String {
        self.hdr_debug()
    }

    pub fn hdr_debug(&self) -> String {
        format!(
            "FractalHeapHeader(addr={:#x}, id_len={}, width={}, root={:#x}, nmanaged={})",
            self.heap_addr,
            self.heap_id_len,
            self.table_width,
            self.root_block_addr,
            self.num_managed_objects
        )
    }

    pub fn get_cparam_test(&self) -> FractalHeapCreateParams {
        FractalHeapCreateParams {
            heap_id_len: self.heap_id_len,
            table_width: self.table_width,
            start_block_size: self.start_block_size,
            max_direct_block_size: self.max_direct_block_size,
            max_heap_size: self.max_heap_size,
        }
    }

    pub fn cmp_cparam_test(
        left: &FractalHeapCreateParams,
        right: &FractalHeapCreateParams,
    ) -> bool {
        left == right
    }

    pub fn get_dtable_width_test(&self) -> u16 {
        self.table_width
    }

    pub fn get_dtable_max_drows_test(&self) -> usize {
        self.max_direct_rows()
    }

    pub fn get_iblock_max_drows_test(&self) -> usize {
        usize::from(self.current_root_rows)
    }

    pub fn get_dblock_size_test(&self, row: usize) -> Result<u64> {
        self.checked_row_block_size(row)
    }

    pub fn get_dblock_free_test(&self, used: u64, row: usize) -> Result<u64> {
        self.checked_row_block_size(row)?
            .checked_sub(used)
            .ok_or_else(|| {
                Error::InvalidFormat("fractal heap direct block used bytes exceed size".into())
            })
    }

    pub fn get_id_off_test(&self, heap_id: &[u8]) -> Result<u64> {
        let offset_bytes = (usize::from(self.max_heap_size) + 7) / 8;
        heap_id
            .get(1..1 + offset_bytes)
            .map(read_le_uint)
            .ok_or_else(|| Error::InvalidFormat("fractal heap ID too short for offset".into()))
    }

    pub fn get_tiny_info_test(&self, heap_id: &[u8]) -> Result<(usize, Vec<u8>)> {
        let data = self.read_tiny(heap_id)?;
        Ok((data.len(), data))
    }

    pub fn get_huge_info_test(&self) -> (u64, u64) {
        (self.huge_btree_addr, u64::from(self.max_managed_obj_size))
    }

    pub fn dtable_encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(28);
        out.extend_from_slice(&self.table_width.to_le_bytes());
        out.extend_from_slice(&self.start_block_size.to_le_bytes());
        out.extend_from_slice(&self.max_direct_block_size.to_le_bytes());
        out.extend_from_slice(&self.max_heap_size.to_le_bytes());
        out.extend_from_slice(&self.start_root_rows.to_le_bytes());
        out.extend_from_slice(&self.current_root_rows.to_le_bytes());
        out
    }

    pub fn dtable_debug(&self) -> String {
        format!(
            "FractalHeapDTable(width={}, start={}, max_direct={}, max_heap_bits={})",
            self.table_width, self.start_block_size, self.max_direct_block_size, self.max_heap_size
        )
    }

    pub fn dtable_init(&mut self, params: FractalHeapCreateParams) -> Result<()> {
        validate_heap_create_params(&params)?;
        self.heap_id_len = params.heap_id_len;
        self.table_width = params.table_width;
        self.start_block_size = params.start_block_size;
        self.max_direct_block_size = params.max_direct_block_size;
        self.max_heap_size = params.max_heap_size;
        Ok(())
    }

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
            row += 1;
            if row > self.max_direct_rows().saturating_add(64) {
                return Err(Error::InvalidFormat(
                    "fractal heap dtable lookup exceeded bounds".into(),
                ));
            }
        }
    }

    pub fn dtable_dest(&mut self) {}

    pub fn dtable_size_to_row(&self, size: u64) -> Result<usize> {
        let mut row = 0usize;
        while self.checked_row_block_size(row)? < size {
            row += 1;
        }
        Ok(row)
    }

    pub fn dtable_size_to_rows(&self, size: u64) -> Result<usize> {
        Ok(self.dtable_size_to_row(size)?.saturating_add(1))
    }

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

    pub fn tiny_init(&self) {}

    pub fn tiny_insert(&self, data: &[u8]) -> Result<Vec<u8>> {
        if data.is_empty() || data.len() > 16 {
            return Err(Error::InvalidFormat(
                "tiny fractal heap object size must be 1..=16".into(),
            ));
        }
        let mut id = Vec::with_capacity(data.len() + 1);
        id.push(0x20 | ((data.len() as u8) - 1));
        id.extend_from_slice(data);
        Ok(id)
    }

    pub fn tiny_get_obj_len(&self, heap_id: &[u8]) -> Result<usize> {
        if heap_id.is_empty() {
            return Err(Error::InvalidFormat("empty tiny heap ID".into()));
        }
        Ok(usize::from(heap_id[0] & 0x0f) + 1)
    }

    pub fn tiny_op(&self, heap_id: &[u8]) -> Result<Vec<u8>> {
        self.read_tiny(heap_id)
    }

    pub fn tiny_remove(&self, heap_id: &[u8]) -> Result<usize> {
        self.tiny_get_obj_len(heap_id)
    }

    pub fn huge_bt2_create(&self) -> Vec<FractalHeapHugeRecord> {
        Vec::new()
    }

    pub fn huge_init(&self) {}

    pub fn huge_new_id(&self, id: u64) -> Vec<u8> {
        let mut out = vec![0x10];
        out.extend_from_slice(&id.to_le_bytes());
        out
    }

    pub fn huge_insert(records: &mut Vec<FractalHeapHugeRecord>, record: FractalHeapHugeRecord) {
        records.push(record);
        records.sort_by(FractalHeapHugeRecord::huge_bt2_indir_compare);
    }

    pub fn huge_get_obj_len(records: &[FractalHeapHugeRecord], id: u64) -> Option<u64> {
        records
            .iter()
            .find(|record| record.id == id)
            .map(|record| record.obj_size)
    }

    pub fn huge_op(records: &[FractalHeapHugeRecord], id: u64) -> Option<FractalHeapHugeRecord> {
        records.iter().find(|record| record.id == id).cloned()
    }

    pub fn huge_remove(
        records: &mut Vec<FractalHeapHugeRecord>,
        id: u64,
    ) -> Option<FractalHeapHugeRecord> {
        FractalHeapHugeRecord::huge_bt2_indir_remove(records, id)
    }

    pub fn huge_term(&self) {}

    pub fn huge_delete(records: &mut Vec<FractalHeapHugeRecord>) {
        records.clear();
    }

    pub fn tracehash_man_seen(&self, _offset: u64) -> bool {
        true
    }

    pub fn man_insert(heap: &mut FractalHeap, data: Vec<u8>) -> Result<Vec<u8>> {
        heap.insert(data)
    }

    pub fn man_get_obj_len(heap: &FractalHeap, id: usize) -> Option<usize> {
        heap.get_obj_len(id)
    }

    pub fn man_get_obj_off(heap: &FractalHeap, heap_id: &[u8]) -> Result<u64> {
        heap.get_obj_off(heap_id)
    }

    pub fn man_write(heap: &mut FractalHeap, id: usize, data: Vec<u8>) -> Result<()> {
        heap.write(id, data)
    }

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

        match id_type {
            0 => self.read_managed(reader, heap_id),
            1 => self.read_huge(reader, heap_id),
            2 => self.read_tiny(heap_id),
            _ => Err(Error::InvalidFormat(format!(
                "unknown heap ID type {id_type}"
            ))),
        }
    }

    fn validate_header(&self) -> Result<()> {
        validate_heap_create_params(&self.get_cparam_test())
    }
}

impl FractalHeap {
    pub fn create(params: FractalHeapCreateParams) -> Result<Self> {
        Ok(Self {
            header: FractalHeapHeader::hdr_create(params)?,
            objects: Vec::new(),
            closed: false,
        })
    }

    pub fn open(header: FractalHeapHeader, objects: Vec<Vec<u8>>) -> Result<Self> {
        header.validate_header()?;
        Ok(Self {
            header,
            objects,
            closed: false,
        })
    }

    pub fn get_id_len(&self) -> u16 {
        self.header.heap_id_len
    }

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
        self.objects.push(data);
        self.header.hdr_incr()?;
        Ok(self.encode_managed_id(id))
    }

    pub fn write(&mut self, id: usize, data: Vec<u8>) -> Result<()> {
        self.ensure_open()?;
        let slot = self
            .objects
            .get_mut(id)
            .ok_or_else(|| Error::InvalidFormat("fractal heap object id out of bounds".into()))?;
        *slot = data;
        Ok(())
    }

    pub fn op(&self, id: usize) -> Option<&[u8]> {
        self.objects.get(id).map(Vec::as_slice)
    }

    pub fn get_obj_off(&self, heap_id: &[u8]) -> Result<u64> {
        self.header.get_id_off_test(heap_id)
    }

    pub fn get_obj_len(&self, id: usize) -> Option<usize> {
        self.objects.get(id).map(Vec::len)
    }

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

    pub fn close(&mut self) {
        self.closed = true;
    }

    pub fn delete(mut self) {
        self.objects.clear();
        self.header.num_managed_objects = 0;
        self.closed = true;
    }

    pub fn stat_info(&self) -> (usize, u64) {
        (self.objects.len(), self.size())
    }

    pub fn size(&self) -> u64 {
        self.objects
            .iter()
            .map(|o| u64::try_from(o.len()).unwrap_or(u64::MAX))
            .try_fold(0u64, |acc, len| acc.checked_add(len))
            .unwrap_or(u64::MAX)
    }

    pub fn id_print(heap_id: &[u8]) -> String {
        heap_id
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<Vec<_>>()
            .join("")
    }

    fn encode_managed_id(&self, offset: u64) -> Vec<u8> {
        let mut id = vec![0; usize::from(self.header.heap_id_len)];
        if !id.is_empty() {
            id[0] = 0;
            let end = id.len().min(9);
            id[1..end].copy_from_slice(&offset.to_le_bytes()[..end - 1]);
        }
        id
    }

    fn ensure_open(&self) -> Result<()> {
        if self.closed {
            Err(Error::InvalidFormat("fractal heap is closed".into()))
        } else {
            Ok(())
        }
    }
}

impl FractalHeapIndirectBlock {
    pub fn iblock_pin(&mut self) {
        self.ref_count += 1;
    }

    pub fn iblock_unpin(&mut self) {
        self.ref_count = self.ref_count.saturating_sub(1);
    }

    pub fn iblock_incr(&mut self) {
        self.iblock_pin();
    }

    pub fn iblock_decr(&mut self) {
        self.iblock_unpin();
    }

    pub fn iblock_dirty(&mut self) {
        self.dirty = true;
    }

    pub fn man_iblock_root_create(nrows: usize, width: usize) -> Self {
        Self {
            nrows,
            child_addrs: vec![u64::MAX; nrows.saturating_mul(width)],
            ref_count: 0,
            dirty: false,
        }
    }

    pub fn man_iblock_alloc_row(&mut self, row: usize, width: usize) {
        let needed = row.saturating_add(1).saturating_mul(width);
        if self.child_addrs.len() < needed {
            self.child_addrs.resize(needed, u64::MAX);
        }
        self.nrows = self.nrows.max(row + 1);
    }

    pub fn man_iblock_create(nrows: usize, width: usize) -> Self {
        Self::man_iblock_root_create(nrows, width)
    }

    pub fn man_iblock_protect(&self) -> Self {
        self.clone()
    }

    pub fn man_iblock_unprotect(self) -> Self {
        self
    }

    pub fn man_iblock_attach(&mut self, index: usize, addr: u64) {
        if index >= self.child_addrs.len() {
            self.child_addrs.resize(index + 1, u64::MAX);
        }
        self.child_addrs[index] = addr;
        self.dirty = true;
    }

    pub fn man_iblock_detach(&mut self, index: usize) -> Option<u64> {
        let addr = self.child_addrs.get_mut(index)?;
        let old = *addr;
        *addr = u64::MAX;
        self.dirty = true;
        Some(old)
    }

    pub fn man_iblock_entry_addr(&self, index: usize) -> Option<u64> {
        self.child_addrs.get(index).copied()
    }

    pub fn man_iblock_delete(self) {}

    pub fn man_iblock_parent_info(&self) -> (usize, usize) {
        (self.nrows, self.child_addrs.len())
    }

    pub fn man_iblock_dest(self) {}

    pub fn cache_iblock_get_initial_load_size() -> usize {
        4
    }

    pub fn cache_iblock_verify_chksum(image: &[u8]) -> Result<()> {
        verify_image_checksum(image, "fractal heap indirect block")
    }

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

    pub fn cache_iblock_pre_serialize(&self) -> Result<Vec<u8>> {
        self.cache_iblock_serialize()
    }

    pub fn cache_iblock_serialize(&self) -> Result<Vec<u8>> {
        let nrows = u64::try_from(self.nrows).map_err(|_| {
            Error::InvalidFormat("fractal heap indirect block row count is too large".into())
        })?;
        let mut out = Vec::with_capacity(self.cache_iblock_image_len()?);
        out.extend_from_slice(&FHIB_MAGIC);
        out.push(0);
        out.extend_from_slice(&nrows.to_le_bytes());
        for addr in &self.child_addrs {
            out.extend_from_slice(&addr.to_le_bytes());
        }
        let checksum = checksum_metadata(&out);
        out.extend_from_slice(&checksum.to_le_bytes());
        Ok(out)
    }

    pub fn cache_iblock_notify(&mut self) {
        self.dirty = false;
    }

    pub fn cache_iblock_free_icr(_image: Vec<u8>) {}

    pub fn cache_verify_iblock_descendants_clean(&self) -> bool {
        !self.dirty
    }

    pub fn cache_verify_iblocks_dblocks_clean(&self, dblocks: &[FractalHeapDirectBlock]) -> bool {
        !self.dirty && dblocks.iter().all(|dblock| !dblock.dirty)
    }

    pub fn cache_verify_descendant_iblocks_clean(blocks: &[Self]) -> bool {
        blocks.iter().all(|block| !block.dirty)
    }

    pub fn iblock_print(&self) -> String {
        self.iblock_debug()
    }

    pub fn iblock_debug(&self) -> String {
        format!(
            "FractalHeapIndirectBlock(nrows={}, children={}, dirty={})",
            self.nrows,
            self.child_addrs.len(),
            self.dirty
        )
    }
}

impl FractalHeapDirectBlock {
    pub fn man_dblock_new(addr: u64, data: Vec<u8>) -> Self {
        Self {
            addr,
            data,
            dirty: false,
        }
    }

    pub fn man_dblock_protect(&self) -> Self {
        self.clone()
    }

    pub fn man_dblock_delete(self) {}

    pub fn man_dblock_destroy(self) {}

    pub fn man_dblock_dest(self) {}

    pub fn cache_dblock_get_initial_load_size() -> usize {
        4
    }

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

    pub fn cache_dblock_pre_serialize(&self) -> Result<Vec<u8>> {
        self.cache_dblock_serialize()
    }

    pub fn cache_dblock_serialize(&self) -> Result<Vec<u8>> {
        let mut out = Vec::with_capacity(self.cache_dblock_image_len()?);
        out.extend_from_slice(&FHDB_MAGIC);
        out.push(0);
        out.extend_from_slice(&self.addr.to_le_bytes());
        out.extend_from_slice(&self.data);
        let checksum = checksum_metadata(&out);
        out.extend_from_slice(&checksum.to_le_bytes());
        Ok(out)
    }

    pub fn cache_dblock_notify(&mut self) {
        self.dirty = false;
    }

    pub fn cache_dblock_free_icr(_image: Vec<u8>) {}

    pub fn cache_dblock_fsf_size(&self) -> usize {
        self.data.len()
    }

    pub fn dblock_debug_cb(&self) -> String {
        self.dblock_debug()
    }

    pub fn dblock_debug(&self) -> String {
        format!(
            "FractalHeapDirectBlock(addr={:#x}, size={})",
            self.addr,
            self.data.len()
        )
    }
}

impl FractalHeapHugeContext {
    pub fn huge_bt2_crt_context(filtered: bool) -> Self {
        Self { filtered }
    }

    pub fn huge_bt2_dst_context(self) {}
}

impl FractalHeapHugeRecord {
    pub fn huge_bt2_indir_store(id: u64, addr: u64, len: u64) -> Self {
        Self {
            id,
            addr,
            len,
            obj_size: len,
            filter_mask: 0,
        }
    }

    pub fn huge_bt2_indir_remove(records: &mut Vec<Self>, id: u64) -> Option<Self> {
        let idx = records.iter().position(|record| record.id == id)?;
        Some(records.remove(idx))
    }

    pub fn huge_bt2_indir_compare(left: &Self, right: &Self) -> std::cmp::Ordering {
        left.id.cmp(&right.id)
    }

    pub fn huge_bt2_indir_debug(&self) -> String {
        format!(
            "HugeIndirect(id={}, addr={:#x}, len={})",
            self.id, self.addr, self.len
        )
    }

    pub fn huge_bt2_filt_indir_found(&self, id: u64) -> bool {
        self.id == id && self.filter_mask != 0
    }

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

    pub fn huge_bt2_filt_indir_remove(records: &mut Vec<Self>, id: u64) -> Option<Self> {
        Self::huge_bt2_indir_remove(records, id)
    }

    pub fn huge_bt2_filt_indir_compare(left: &Self, right: &Self) -> std::cmp::Ordering {
        Self::huge_bt2_indir_compare(left, right)
    }

    pub fn huge_bt2_filt_indir_debug(&self) -> String {
        format!(
            "HugeFilteredIndirect(id={}, addr={:#x}, len={}, obj_size={}, mask={:#x})",
            self.id, self.addr, self.len, self.obj_size, self.filter_mask
        )
    }

    pub fn huge_bt2_dir_store(addr: u64, len: u64) -> Self {
        Self::huge_bt2_indir_store(0, addr, len)
    }

    pub fn huge_bt2_dir_remove(records: &mut Vec<Self>, addr: u64) -> Option<Self> {
        let idx = records.iter().position(|record| record.addr == addr)?;
        Some(records.remove(idx))
    }

    pub fn huge_bt2_dir_compare(left: &Self, right: &Self) -> std::cmp::Ordering {
        left.addr.cmp(&right.addr)
    }

    pub fn huge_bt2_dir_debug(&self) -> String {
        format!("HugeDirect(addr={:#x}, len={})", self.addr, self.len)
    }

    pub fn huge_bt2_filt_dir_found(&self, addr: u64) -> bool {
        self.addr == addr && self.filter_mask != 0
    }

    pub fn huge_bt2_filt_dir_store(addr: u64, len: u64, obj_size: u64, filter_mask: u32) -> Self {
        Self {
            id: 0,
            addr,
            len,
            obj_size,
            filter_mask,
        }
    }

    pub fn huge_bt2_filt_dir_remove(records: &mut Vec<Self>, addr: u64) -> Option<Self> {
        Self::huge_bt2_dir_remove(records, addr)
    }

    pub fn huge_bt2_filt_dir_compare(left: &Self, right: &Self) -> std::cmp::Ordering {
        Self::huge_bt2_dir_compare(left, right)
    }

    pub fn huge_bt2_filt_dir_encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(28);
        out.extend_from_slice(&self.addr.to_le_bytes());
        out.extend_from_slice(&self.len.to_le_bytes());
        out.extend_from_slice(&self.filter_mask.to_le_bytes());
        out.extend_from_slice(&self.obj_size.to_le_bytes());
        out
    }

    pub fn huge_bt2_filt_dir_debug(&self) -> String {
        format!(
            "HugeFilteredDirect(addr={:#x}, len={}, obj_size={}, mask={:#x})",
            self.addr, self.len, self.obj_size, self.filter_mask
        )
    }
}

impl FractalHeapIterator {
    pub fn man_iter_init(start: u64, count: u64) -> Self {
        let offsets = (0..count).map(|idx| start.saturating_add(idx)).collect();
        Self { offsets, index: 0 }
    }

    pub fn man_iter_start_offset(&mut self, offset: u64) {
        self.index = self
            .offsets
            .iter()
            .position(|candidate| *candidate >= offset)
            .unwrap_or(self.offsets.len());
    }

    pub fn man_iter_start_entry(&mut self, entry: usize) {
        self.index = entry.min(self.offsets.len());
    }

    pub fn man_iter_reset(&mut self) {
        self.index = 0;
    }

    pub fn man_iter_next(&mut self) -> Option<u64> {
        let value = self.offsets.get(self.index).copied();
        if value.is_some() {
            self.index += 1;
        }
        value
    }

    pub fn man_iter_up(&mut self) {
        self.index = self.index.saturating_sub(1);
    }

    pub fn man_iter_down(&mut self) {
        self.index = self.index.saturating_add(1).min(self.offsets.len());
    }

    pub fn man_iter_curr(&self) -> Option<u64> {
        self.offsets.get(self.index).copied()
    }

    pub fn man_iter_ready(&self) -> bool {
        self.index < self.offsets.len()
    }
}

impl FractalHeapSection {
    pub fn sect_node_free(self) {}

    pub fn sect_single_new(offset: u64, size: u64) -> Self {
        Self::Single { offset, size }
    }

    pub fn sect_single_revive(&mut self) {}

    pub fn sect_single_dblock_info(&self) -> Option<(u64, u64)> {
        match self {
            Self::Single { offset, size } => Some((*offset, *size)),
            _ => None,
        }
    }

    pub fn sect_single_reduce(&mut self, amount: u64) {
        if let Self::Single { size, .. } = self {
            *size = size.saturating_sub(amount);
        }
    }

    pub fn sect_single_add(sections: &mut Vec<Self>, section: Self) {
        sections.push(section);
    }

    pub fn sect_single_deserialize(offset: u64, size: u64) -> Self {
        Self::Single { offset, size }
    }

    pub fn sect_single_can_merge(&self, other: &Self) -> bool {
        section_end(self) == section_offset(other)
    }

    pub fn sect_single_merge(&mut self, other: &Self) -> bool {
        if self.sect_single_can_merge(other) {
            if let (Some(size), Some(other_size)) = (section_size_mut(self), section_size(other)) {
                *size = size.saturating_add(other_size);
                return true;
            }
        }
        false
    }

    pub fn sect_single_can_shrink(&self, eoa: u64) -> bool {
        section_end(self) == Some(eoa)
    }

    pub fn sect_single_shrink(&mut self, amount: u64) {
        self.sect_single_reduce(amount);
    }

    pub fn sect_single_free(self) {}

    pub fn sect_single_valid(&self) -> bool {
        section_size(self).is_some_and(|size| size > 0)
    }

    pub fn sect_row_from_single(section: Self, row: usize) -> Self {
        let offset = section_offset(&section).unwrap_or(0);
        let size = section_size(&section).unwrap_or(0);
        Self::Row { row, offset, size }
    }

    pub fn sect_row_revive(&mut self) {}

    pub fn sect_row_reduce(&mut self, amount: u64) {
        if let Self::Row { size, .. } = self {
            *size = size.saturating_sub(amount);
        }
    }

    pub fn sect_row_first(&self) -> Option<u64> {
        section_offset(self)
    }

    pub fn sect_row_get_iblock(&self) -> Option<usize> {
        match self {
            Self::Row { row, .. } => Some(*row),
            _ => None,
        }
    }

    pub fn sect_row_parent_removed(&mut self) {}

    pub fn sect_row_init_cls() {}

    pub fn sect_row_term_cls() {}

    pub fn sect_row_serialize(&self) -> Result<Vec<u8>> {
        let Self::Row { offset, size, .. } = self else {
            return Err(Error::InvalidFormat(
                "fractal heap row section serializer received non-row section".into(),
            ));
        };
        let mut out = Vec::with_capacity(16);
        out.extend_from_slice(&offset.to_le_bytes());
        out.extend_from_slice(&size.to_le_bytes());
        Ok(out)
    }

    pub fn sect_row_deserialize(row: usize, offset: u64, size: u64) -> Self {
        Self::Row { row, offset, size }
    }

    pub fn sect_row_can_merge(&self, other: &Self) -> bool {
        self.sect_single_can_merge(other)
    }

    pub fn sect_row_merge(&mut self, other: &Self) -> bool {
        self.sect_single_merge(other)
    }

    pub fn sect_row_can_shrink(&self, eoa: u64) -> bool {
        self.sect_single_can_shrink(eoa)
    }

    pub fn sect_row_shrink(&mut self, amount: u64) {
        self.sect_row_reduce(amount);
    }

    pub fn sect_row_free_real(self) {}

    pub fn sect_row_free(self) {}

    pub fn sect_row_valid(&self) -> bool {
        self.sect_single_valid()
    }

    pub fn sect_row_debug(&self) -> String {
        format!("{self:?}")
    }

    pub fn sect_indirect_iblock_off(&self) -> Option<u64> {
        section_offset(self)
    }

    pub fn sect_indirect_top(&self) -> bool {
        matches!(self, Self::Indirect { .. })
    }

    pub fn sect_indirect_term_cls() {}

    pub fn sect_indirect_new(offset: u64, rows: usize, size: u64) -> Self {
        Self::Indirect { offset, rows, size }
    }

    pub fn sect_indirect_for_row(row: usize, offset: u64, size: u64) -> Self {
        Self::Indirect {
            offset,
            rows: row.saturating_add(1),
            size,
        }
    }

    pub fn sect_indirect_init_rows(&mut self, rows: usize) {
        if let Self::Indirect { rows: current, .. } = self {
            *current = rows;
        }
    }

    pub fn sect_indirect_add(sections: &mut Vec<Self>, section: Self) {
        sections.push(section);
    }

    pub fn sect_indirect_decr(&mut self) {
        if let Self::Indirect { rows, .. } = self {
            *rows = rows.saturating_sub(1);
        }
    }

    pub fn sect_indirect_revive_row(&mut self, row: usize) {
        if let Self::Indirect { rows, .. } = self {
            *rows = (*rows).max(row + 1);
        }
    }

    pub fn sect_indirect_revive(&mut self) {}

    pub fn sect_indirect_reduce_row(&mut self, amount: u64) {
        if let Self::Indirect { size, .. } = self {
            *size = size.saturating_sub(amount);
        }
    }

    pub fn sect_indirect_reduce(&mut self, amount: u64) {
        self.sect_indirect_reduce_row(amount);
    }

    pub fn sect_indirect_is_first(&self, offset: u64) -> bool {
        section_offset(self) == Some(offset)
    }

    pub fn sect_indirect_first(&self) -> Option<u64> {
        section_offset(self)
    }

    pub fn sect_indirect_get_iblock(&self) -> Option<usize> {
        match self {
            Self::Indirect { rows, .. } => Some(*rows),
            _ => None,
        }
    }

    pub fn sect_indirect_merge_row(&mut self, other: &Self) -> bool {
        self.sect_single_merge(other)
    }

    pub fn sect_indirect_build_parent(&self) -> Option<Self> {
        Some(self.clone())
    }

    pub fn sect_indirect_shrink(&mut self, amount: u64) {
        self.sect_indirect_reduce(amount);
    }

    pub fn sect_indirect_serialize(&self) -> Result<Vec<u8>> {
        let Self::Indirect { offset, size, .. } = self else {
            return Err(Error::InvalidFormat(
                "fractal heap indirect section serializer received non-indirect section".into(),
            ));
        };
        let mut out = Vec::with_capacity(16);
        out.extend_from_slice(&offset.to_le_bytes());
        out.extend_from_slice(&size.to_le_bytes());
        Ok(out)
    }

    pub fn sect_indirect_free(self) {}

    pub fn sect_indirect_valid(&self) -> bool {
        self.sect_single_valid()
    }

    pub fn sect_indirect_debug(&self) -> String {
        format!("{self:?}")
    }
}

impl FractalHeapSpace {
    pub fn space_start() -> Self {
        Self::default()
    }

    pub fn space_add(&mut self, section: FractalHeapSection) {
        self.sections.push(section);
    }

    pub fn space_find(&self, size: u64) -> Option<&FractalHeapSection> {
        self.sections
            .iter()
            .find(|section| section_size(section).is_some_and(|candidate| candidate >= size))
    }

    pub fn space_revert_root_cb(&mut self) {}

    pub fn space_revert_root(&mut self) {}

    pub fn space_create_root_cb(&mut self) {}

    pub fn space_create_root(&mut self) {}

    pub fn space_size(&self) -> u64 {
        self.sections.iter().filter_map(section_size).sum()
    }

    pub fn space_remove(&mut self, offset: u64) -> Option<FractalHeapSection> {
        let idx = self
            .sections
            .iter()
            .position(|section| section_offset(section) == Some(offset))?;
        Some(self.sections.remove(idx))
    }

    pub fn space_close(&mut self) {
        self.closed = true;
    }

    pub fn space_delete(mut self) {
        self.sections.clear();
        self.closed = true;
    }

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

    pub fn sects_debug_cb(section: &FractalHeapSection) -> String {
        format!("{section:?}")
    }

    pub fn sects_debug(&self) -> String {
        self.sections
            .iter()
            .map(Self::sects_debug_cb)
            .collect::<Vec<_>>()
            .join(", ")
    }
}

// ---------------------------------------------------------------------------
// Trace probes — kept in one place because they're small, conditional on
// the `tracehash` feature, and called from sibling modules (dblock/huge/
// tiny). Each emits one tracehash event for the read it just performed.
// ---------------------------------------------------------------------------

impl FractalHeapHeader {
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

    #[cfg(feature = "tracehash")]
    pub(super) fn trace_tiny_object(&self, heap_id: &[u8], object_len: u64) {
        let mut th = tracehash::th_call!("hdf5.fractal_heap.tiny_object");
        th.input_u64(self.heap_addr);
        th.input_bytes(heap_id);
        th.output_value(&(true));
        th.output_u64(object_len);
        th.finish();
    }

    #[cfg(not(feature = "tracehash"))]
    pub(super) fn trace_tiny_object(&self, _heap_id: &[u8], _object_len: u64) {}
}

// ---------------------------------------------------------------------------
// Internal numeric / checksum helpers shared across submodules.
// ---------------------------------------------------------------------------

pub(super) fn read_le_uint(bytes: &[u8]) -> u64 {
    let mut value = 0u64;
    for (idx, byte) in bytes.iter().take(8).enumerate() {
        value |= u64::from(*byte) << (idx * 8);
    }
    value
}

pub(super) fn verify_metadata_checksum<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    start: u64,
    check_len: u64,
    context: &str,
) -> Result<()> {
    let restore = reader.position()?;
    let check_len_usize = heap_object_len(check_len, context)?;
    reader.seek(start)?;
    let bytes = reader.read_bytes(check_len_usize)?;
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
    let bytes = reader.read_bytes(check_len)?;
    let stored = reader.read_u32()?;
    let payload_len = heap_object_len(block_size, "fractal heap direct block size")?;
    let payload = reader.read_bytes(payload_len)?;
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

fn section_offset(section: &FractalHeapSection) -> Option<u64> {
    match section {
        FractalHeapSection::Single { offset, .. }
        | FractalHeapSection::Row { offset, .. }
        | FractalHeapSection::Indirect { offset, .. } => Some(*offset),
    }
}

fn section_size(section: &FractalHeapSection) -> Option<u64> {
    match section {
        FractalHeapSection::Single { size, .. }
        | FractalHeapSection::Row { size, .. }
        | FractalHeapSection::Indirect { size, .. } => Some(*size),
    }
}

fn section_size_mut(section: &mut FractalHeapSection) -> Option<&mut u64> {
    match section {
        FractalHeapSection::Single { size, .. }
        | FractalHeapSection::Row { size, .. }
        | FractalHeapSection::Indirect { size, .. } => Some(size),
    }
}

fn section_end(section: &FractalHeapSection) -> Option<u64> {
    section_offset(section)?.checked_add(section_size(section)?)
}

pub(super) fn log2_power2(value: u64) -> u32 {
    debug_assert!(value.is_power_of_two());
    value.trailing_zeros()
}

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
        let image = iblock.cache_iblock_serialize().unwrap();
        assert_eq!(image.len(), iblock.cache_iblock_image_len().unwrap());
        assert_eq!(&image[..4], &FHIB_MAGIC);
        assert_eq!(u64::from_le_bytes(image[5..13].try_into().unwrap()), 2);
        FractalHeapIndirectBlock::cache_iblock_verify_chksum(&image).unwrap();
        assert_eq!(iblock.cache_iblock_pre_serialize().unwrap(), image);

        let dblock = FractalHeapDirectBlock::man_dblock_new(64, vec![1, 2, 3, 4]);
        let image = dblock.cache_dblock_serialize().unwrap();
        assert_eq!(image.len(), dblock.cache_dblock_image_len().unwrap());
        assert_eq!(&image[..4], &FHDB_MAGIC);
        assert_eq!(u64::from_le_bytes(image[5..13].try_into().unwrap()), 64);
        verify_image_checksum(&image, "fractal heap direct block").unwrap();
        assert_eq!(dblock.cache_dblock_pre_serialize().unwrap(), image);
    }

    #[test]
    fn section_class_serializers_reject_wrong_section_variants() {
        let row = FractalHeapSection::sect_row_deserialize(3, 10, 20);
        let image = row.sect_row_serialize().unwrap();
        assert_eq!(u64::from_le_bytes(image[..8].try_into().unwrap()), 10);
        assert_eq!(u64::from_le_bytes(image[8..16].try_into().unwrap()), 20);

        let indirect = FractalHeapSection::sect_indirect_new(30, 2, 40);
        let image = indirect.sect_indirect_serialize().unwrap();
        assert_eq!(u64::from_le_bytes(image[..8].try_into().unwrap()), 30);
        assert_eq!(u64::from_le_bytes(image[8..16].try_into().unwrap()), 40);

        let single = FractalHeapSection::sect_single_new(1, 2);
        assert!(single.sect_row_serialize().is_err());
        assert!(single.sect_indirect_serialize().is_err());
    }
}
