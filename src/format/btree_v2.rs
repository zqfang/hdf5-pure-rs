use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::format::checksum::checksum_metadata;
use crate::io::reader::{is_undef_addr, HdfReader};

/// v2 B-tree header magic: "BTHD"
const B2HD_MAGIC: [u8; 4] = [b'B', b'T', b'H', b'D'];
/// v2 B-tree leaf node magic: "BTLF"
const B2LF_MAGIC: [u8; 4] = [b'B', b'T', b'L', b'F'];
/// v2 B-tree internal node magic: "BTIN"
const B2IN_MAGIC: [u8; 4] = [b'B', b'T', b'I', b'N'];
const B2_METADATA_PREFIX_SIZE: usize = 10;

/// v2 B-tree header.
#[derive(Debug, Clone)]
pub struct BTreeV2Header {
    pub tree_type: u8,
    pub node_size: u32,
    pub record_size: u16,
    pub depth: u16,
    pub split_pct: u8,
    pub merge_pct: u8,
    pub root_addr: u64,
    pub root_nrecords: u16,
    pub total_records: u64,
}

impl BTreeV2Header {
    pub fn hdr_alloc(
        tree_type: u8,
        node_size: u32,
        record_size: u16,
        split_pct: u8,
        merge_pct: u8,
    ) -> Result<Self> {
        let header = Self {
            tree_type,
            node_size,
            record_size,
            depth: 0,
            split_pct,
            merge_pct,
            root_addr: 0,
            root_nrecords: 0,
            total_records: 0,
        };
        header.validate()?;
        Ok(header)
    }

    pub fn hdr_create(
        tree_type: u8,
        node_size: u32,
        record_size: u16,
        split_pct: u8,
        merge_pct: u8,
    ) -> Result<Self> {
        Self::hdr_alloc(tree_type, node_size, record_size, split_pct, merge_pct)
    }

    pub fn read_at<R: Read + Seek>(reader: &mut HdfReader<R>, addr: u64) -> Result<Self> {
        reader.seek(addr).map_err(|err| {
            Error::InvalidFormat(format!("failed to seek to v2 B-tree header {addr}: {err}"))
        })?;

        let magic = reader.read_bytes(4)?;
        if magic != B2HD_MAGIC {
            return Err(Error::InvalidFormat(
                "invalid v2 B-tree header magic".into(),
            ));
        }

        let version = reader.read_u8()?;
        if version != 0 {
            return Err(Error::Unsupported(format!(
                "v2 B-tree header version {version}"
            )));
        }

        let tree_type = reader.read_u8()?;
        let node_size = reader.read_u32()?;
        let record_size = reader.read_u16()?;
        let depth = reader.read_u16()?;
        let split_pct = reader.read_u8()?;
        let merge_pct = reader.read_u8()?;
        if split_pct == 0 || split_pct > 100 {
            return Err(Error::InvalidFormat(format!(
                "v2 B-tree split percent {split_pct} must be in 1..=100"
            )));
        }
        if merge_pct == 0 || merge_pct > 100 {
            return Err(Error::InvalidFormat(format!(
                "v2 B-tree merge percent {merge_pct} must be in 1..=100"
            )));
        }
        let root_addr = reader.read_addr()?;
        let root_nrecords = reader.read_u16()?;
        let total_records = reader.read_length()?;
        if total_records == 0 && root_nrecords != 0 {
            return Err(Error::InvalidFormat(
                "v2 B-tree empty tree declares root records".into(),
            ));
        }
        if total_records > 0 && is_undef_addr(root_addr) {
            return Err(Error::InvalidFormat(
                "v2 B-tree root address is undefined for non-empty tree".into(),
            ));
        }
        if u64::from(root_nrecords) > total_records {
            return Err(Error::InvalidFormat(
                "v2 B-tree root record count exceeds total records".into(),
            ));
        }

        verify_checksum(reader, addr, "v2 B-tree header")?;

        let header = Self {
            tree_type,
            node_size,
            record_size,
            depth,
            split_pct,
            merge_pct,
            root_addr,
            root_nrecords,
            total_records,
        };
        header.validate()?;
        Ok(header)
    }

    pub fn validate(&self) -> Result<()> {
        let min_node_size = u32::try_from(B2_METADATA_PREFIX_SIZE)
            .map_err(|_| Error::InvalidFormat("v2 B-tree metadata prefix too large".into()))?;
        if self.node_size <= min_node_size {
            return Err(Error::InvalidFormat("invalid v2 B-tree node sizing".into()));
        }
        if self.record_size == 0 {
            return Err(Error::InvalidFormat(
                "v2 B-tree record size must be positive".into(),
            ));
        }
        if self.split_pct == 0 || self.split_pct > 100 {
            return Err(Error::InvalidFormat(format!(
                "v2 B-tree split percent {} must be in 1..=100",
                self.split_pct
            )));
        }
        if self.merge_pct == 0 || self.merge_pct > 100 {
            return Err(Error::InvalidFormat(format!(
                "v2 B-tree merge percent {} must be in 1..=100",
                self.merge_pct
            )));
        }
        if self.total_records == 0 && self.root_nrecords != 0 {
            return Err(Error::InvalidFormat(
                "v2 B-tree empty tree declares root records".into(),
            ));
        }
        if u64::from(self.root_nrecords) > self.total_records {
            return Err(Error::InvalidFormat(
                "v2 B-tree root record count exceeds total records".into(),
            ));
        }
        Ok(())
    }

    pub fn cache_hdr_get_initial_load_size() -> usize {
        4
    }

    pub fn cache_hdr_image_len(&self) -> usize {
        B2_METADATA_PREFIX_SIZE + 2 + 2 + 1 + 1 + 8 + 2 + 8 + 4
    }

    pub fn cache_hdr_serialize(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let mut image = Vec::with_capacity(self.cache_hdr_image_len());
        image.extend_from_slice(&B2HD_MAGIC);
        image.push(0);
        image.push(self.tree_type);
        image.extend_from_slice(&self.node_size.to_le_bytes());
        image.extend_from_slice(&self.record_size.to_le_bytes());
        image.extend_from_slice(&self.depth.to_le_bytes());
        image.push(self.split_pct);
        image.push(self.merge_pct);
        image.extend_from_slice(&self.root_addr.to_le_bytes());
        image.extend_from_slice(&self.root_nrecords.to_le_bytes());
        image.extend_from_slice(&self.total_records.to_le_bytes());
        let checksum = checksum_metadata(&image);
        image.extend_from_slice(&checksum.to_le_bytes());
        Ok(image)
    }

    pub fn cache_hdr_notify(&mut self, action: BTreeV2CacheAction) {
        if matches!(action, BTreeV2CacheAction::Dirtied) {
            self.hdr_dirty();
        }
    }

    pub fn cache_hdr_free_icr(_image: Vec<u8>) {}

    pub fn hdr_debug(&self) -> String {
        format!(
            "BTreeV2Header(type={}, node_size={}, record_size={}, depth={}, root={:#x}, nrec={})",
            self.tree_type,
            self.node_size,
            self.record_size,
            self.depth,
            self.root_addr,
            self.total_records
        )
    }

    pub fn hdr_incr(&mut self, nrecords: u64) -> Result<()> {
        self.total_records = self
            .total_records
            .checked_add(nrecords)
            .ok_or_else(|| Error::InvalidFormat("v2 B-tree record count overflow".into()))?;
        Ok(())
    }

    pub fn hdr_decr(&mut self, nrecords: u64) -> Result<()> {
        self.total_records = self
            .total_records
            .checked_sub(nrecords)
            .ok_or_else(|| Error::InvalidFormat("v2 B-tree record count underflow".into()))?;
        Ok(())
    }

    pub fn hdr_fuse_incr(&mut self) -> Result<()> {
        self.hdr_incr(1)
    }

    pub fn hdr_fuse_decr(&mut self) -> Result<()> {
        self.hdr_decr(1)
    }

    pub fn hdr_dirty(&mut self) {}

    pub fn hdr_protect(&self) -> Self {
        self.clone()
    }

    pub fn hdr_unprotect(self) -> Self {
        self
    }

    pub fn hdr_free(self) {}

    pub fn hdr_delete(self) {}

    pub fn size(&self) -> u64 {
        let header_len = u64::try_from(self.cache_hdr_image_len()).unwrap_or(u64::MAX);
        header_len.saturating_add(
            self.total_records
                .saturating_mul(u64::from(self.record_size)),
        )
    }

    pub fn node_size(&self) -> u32 {
        self.node_size
    }

    pub fn create_flush_depend(&self) {}

    pub fn update_flush_depend(&self) {}

    pub fn update_child_flush_depends(&self) {}

    pub fn destroy_flush_depend(&self) {}

    pub fn get_root_addr_test(&self) -> u64 {
        self.root_addr
    }

    pub fn get_node_info_test(&self, sizeof_addr: usize) -> Result<Vec<(usize, u64, usize)>> {
        compute_node_info(self, sizeof_addr).map(|infos| {
            infos
                .into_iter()
                .map(|info| (info.max_nrec, info.cum_max_nrec, info.cum_max_nrec_size))
                .collect()
        })
    }

    pub fn get_node_depth_test(&self) -> u16 {
        self.depth
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BTreeV2CacheAction {
    Loaded,
    Dirtied,
    Flushed,
    Evicted,
}

fn verify_checksum<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    start: u64,
    context: &str,
) -> Result<()> {
    let checksum_pos = reader.position()?;
    let stored_checksum = reader.read_u32()?;
    let check_len = usize::try_from(checksum_pos - start)
        .map_err(|_| Error::InvalidFormat(format!("{context} checksum span is too large")))?;
    reader.seek(start)?;
    let check_data = reader.read_bytes(check_len)?;
    let computed = checksum_metadata(&check_data);
    if stored_checksum != computed {
        return Err(Error::InvalidFormat(format!(
            "{context} checksum mismatch: stored={stored_checksum:#010x}, computed={computed:#010x}"
        )));
    }
    reader.seek(checksum_pos.checked_add(4).ok_or_else(|| {
        Error::InvalidFormat(format!("{context} checksum end offset overflow"))
    })?)?;
    Ok(())
}

/// Collect all records from a v2 B-tree as raw byte arrays.
pub fn collect_all_records<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header_addr: u64,
) -> Result<Vec<Vec<u8>>> {
    let header = BTreeV2Header::read_at(reader, header_addr)?;

    if header.total_records == 0 {
        return Ok(Vec::new());
    }

    let mut records = Vec::new();
    if header.depth == 0 {
        let node_info = compute_node_info(&header, usize::from(reader.sizeof_addr()))?;
        read_leaf_records(
            reader,
            &header,
            &node_info,
            header.root_addr,
            header.root_nrecords,
            &mut records,
        )?;
    } else {
        let node_info = compute_node_info(&header, usize::from(reader.sizeof_addr()))?;
        read_internal_records(
            reader,
            &header,
            &node_info,
            header.root_addr,
            header.root_nrecords,
            header.depth,
            &mut records,
        )?;
    }

    Ok(records)
}

#[derive(Debug, Clone)]
struct NodeInfo {
    max_nrec: usize,
    cum_max_nrec: u64,
    cum_max_nrec_size: usize,
}

fn compute_node_info(header: &BTreeV2Header, sizeof_addr: usize) -> Result<Vec<NodeInfo>> {
    let node_size = btree_u32_to_usize(header.node_size, "v2 B-tree node size")?;
    let record_size = usize::from(header.record_size);
    if node_size <= B2_METADATA_PREFIX_SIZE || record_size == 0 {
        return Err(Error::InvalidFormat("invalid v2 B-tree node sizing".into()));
    }

    let leaf_max = (node_size - B2_METADATA_PREFIX_SIZE) / record_size;
    if leaf_max == 0 {
        return Err(Error::InvalidFormat(
            "v2 B-tree leaf cannot hold any records".into(),
        ));
    }

    let leaf_max_u64 = btree_usize_to_u64(leaf_max, "v2 B-tree leaf record capacity")?;
    let max_nrec_size = bytes_needed(leaf_max_u64);
    let depth = usize::from(header.depth);
    let depth_count = depth
        .checked_add(1)
        .ok_or_else(|| Error::InvalidFormat("v2 B-tree depth count overflow".into()))?;
    let mut node_info = Vec::with_capacity(depth_count);
    node_info.push(NodeInfo {
        max_nrec: leaf_max,
        cum_max_nrec: leaf_max_u64,
        cum_max_nrec_size: 0,
    });

    for depth in 1..=depth {
        let pointer_size = checked_usize_sum(
            &[
                sizeof_addr,
                max_nrec_size,
                node_info[depth - 1].cum_max_nrec_size,
            ],
            "v2 B-tree pointer size",
        )?;
        let prefix_and_pointer = checked_usize_sum(
            &[B2_METADATA_PREFIX_SIZE, pointer_size],
            "v2 B-tree internal node prefix",
        )?;
        if node_size <= prefix_and_pointer {
            return Err(Error::InvalidFormat(
                "v2 B-tree internal node cannot hold records".into(),
            ));
        }

        let record_slot = record_size.checked_add(pointer_size).ok_or_else(|| {
            Error::InvalidFormat("v2 B-tree internal record slot size overflow".into())
        })?;
        let max_nrec = (node_size - prefix_and_pointer) / record_slot;
        if max_nrec == 0 {
            return Err(Error::InvalidFormat(
                "v2 B-tree internal node cannot hold records".into(),
            ));
        }

        let prev_cum = node_info[depth - 1].cum_max_nrec;
        let max_nrec_u64 = btree_usize_to_u64(max_nrec, "v2 B-tree internal record capacity")?;
        let cum_max_nrec = max_nrec_u64
            .checked_add(1)
            .and_then(|n| n.checked_mul(prev_cum))
            .and_then(|n| n.checked_add(max_nrec_u64))
            .ok_or_else(|| {
                Error::InvalidFormat("v2 B-tree cumulative record count overflow".into())
            })?;
        node_info.push(NodeInfo {
            max_nrec,
            cum_max_nrec,
            cum_max_nrec_size: bytes_needed(cum_max_nrec),
        });
    }

    Ok(node_info)
}

/// Decoded v2 B-tree internal node — magic, version, the record array,
/// and the child-pointer table. Mirrors `H5B2__cache_int_deserialize`:
/// pure parsing, no traversal of children.
#[derive(Debug, Clone)]
pub struct BTreeV2InternalNode {
    pub records: Vec<Vec<u8>>,
    /// One entry per child pointer: `(child_addr, child_nrecords)`.
    /// `records.len() + 1` entries total.
    pub children: Vec<(u64, u16)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BTreeV2LeafNode {
    pub records: Vec<Vec<u8>>,
}

impl BTreeV2LeafNode {
    pub fn create_leaf(records: Vec<Vec<u8>>, record_size: usize) -> Result<Self> {
        validate_records(&records, record_size)?;
        Ok(Self { records })
    }

    pub fn protect_leaf(&self) -> Self {
        self.clone()
    }

    pub fn neighbor_leaf(&self, record: &[u8], direction: BTreeV2Neighbor) -> Option<Vec<u8>> {
        neighbor_record(&self.records, record, direction)
    }

    pub fn insert_leaf(&mut self, record: Vec<u8>, record_size: usize) -> Result<bool> {
        insert_sorted_unique(&mut self.records, record, record_size)
    }

    pub fn update_leaf(&mut self, record: Vec<u8>, record_size: usize) -> Result<bool> {
        validate_record(&record, record_size)?;
        match self
            .records
            .binary_search_by(|probe| probe.as_slice().cmp(&record))
        {
            Ok(idx) => {
                self.records[idx] = record;
                Ok(true)
            }
            Err(_) => Ok(false),
        }
    }

    pub fn swap_leaf(&mut self, idx_a: usize, idx_b: usize) -> Result<()> {
        if idx_a >= self.records.len() || idx_b >= self.records.len() {
            return Err(Error::InvalidFormat(
                "v2 B-tree leaf swap index out of bounds".into(),
            ));
        }
        self.records.swap(idx_a, idx_b);
        Ok(())
    }

    pub fn remove_leaf(&mut self, record: &[u8]) -> Option<Vec<u8>> {
        let idx = self
            .records
            .binary_search_by(|probe| probe.as_slice().cmp(record))
            .ok()?;
        Some(self.records.remove(idx))
    }

    pub fn remove_leaf_by_idx(&mut self, index: usize) -> Option<Vec<u8>> {
        if index < self.records.len() {
            Some(self.records.remove(index))
        } else {
            None
        }
    }

    pub fn assert_leaf(&self, record_size: usize) -> Result<()> {
        validate_records(&self.records, record_size)?;
        if !self.records.windows(2).all(|pair| pair[0] <= pair[1]) {
            return Err(Error::InvalidFormat(
                "v2 B-tree leaf records are not sorted".into(),
            ));
        }
        Ok(())
    }

    pub fn assert_leaf2(&self, record_size: usize) -> Result<()> {
        self.assert_leaf(record_size)
    }

    pub fn cache_leaf_get_initial_load_size() -> usize {
        6
    }

    pub fn cache_leaf_verify_chksum(image: &[u8]) -> Result<()> {
        verify_image_checksum(image, "v2 B-tree leaf")
    }

    pub fn cache_leaf_image_len(&self, record_size: usize) -> usize {
        6 + self.records.len() * record_size + 4
    }

    pub fn cache_leaf_serialize(&self, tree_type: u8, record_size: usize) -> Result<Vec<u8>> {
        self.assert_leaf(record_size)?;
        let mut image = Vec::with_capacity(self.cache_leaf_image_len(record_size));
        image.extend_from_slice(&B2LF_MAGIC);
        image.push(0);
        image.push(tree_type);
        for record in &self.records {
            image.extend_from_slice(record);
        }
        let checksum = checksum_metadata(&image);
        image.extend_from_slice(&checksum.to_le_bytes());
        Ok(image)
    }

    pub fn cache_leaf_notify(&mut self, _action: BTreeV2CacheAction) {}

    pub fn cache_leaf_free_icr(_image: Vec<u8>) {}
}

impl BTreeV2InternalNode {
    pub fn create_internal(
        records: Vec<Vec<u8>>,
        children: Vec<(u64, u16)>,
        record_size: usize,
    ) -> Result<Self> {
        validate_records(&records, record_size)?;
        if children.len() != records.len().saturating_add(1) {
            return Err(Error::InvalidFormat(
                "v2 B-tree internal child count must be record count + 1".into(),
            ));
        }
        Ok(Self { records, children })
    }

    pub fn protect_internal(&self) -> Self {
        self.clone()
    }

    pub fn neighbor_internal(&self, record: &[u8], direction: BTreeV2Neighbor) -> Option<Vec<u8>> {
        neighbor_record(&self.records, record, direction)
    }

    pub fn insert_internal(
        &mut self,
        record: Vec<u8>,
        right_child: (u64, u16),
        record_size: usize,
    ) -> Result<bool> {
        validate_record(&record, record_size)?;
        match self
            .records
            .binary_search_by(|probe| probe.as_slice().cmp(&record))
        {
            Ok(_) => Ok(false),
            Err(idx) => {
                self.records.insert(idx, record);
                self.children.insert(idx + 1, right_child);
                Ok(true)
            }
        }
    }

    pub fn update_internal(&mut self, record: Vec<u8>, record_size: usize) -> Result<bool> {
        validate_record(&record, record_size)?;
        match self
            .records
            .binary_search_by(|probe| probe.as_slice().cmp(&record))
        {
            Ok(idx) => {
                self.records[idx] = record;
                Ok(true)
            }
            Err(_) => Ok(false),
        }
    }

    pub fn shadow_internal(&self) -> Self {
        self.clone()
    }

    pub fn remove_internal(&mut self, record: &[u8]) -> Option<Vec<u8>> {
        let idx = self
            .records
            .binary_search_by(|probe| probe.as_slice().cmp(record))
            .ok()?;
        self.children.remove(idx + 1);
        Some(self.records.remove(idx))
    }

    pub fn remove_internal_by_idx(&mut self, index: usize) -> Option<Vec<u8>> {
        if index >= self.records.len() {
            return None;
        }
        self.children.remove(index + 1);
        Some(self.records.remove(index))
    }

    pub fn internal_free(self) {}

    pub fn assert_internal(&self, record_size: usize) -> Result<()> {
        validate_records(&self.records, record_size)?;
        if self.children.len() != self.records.len().saturating_add(1) {
            return Err(Error::InvalidFormat(
                "v2 B-tree internal child count must be record count + 1".into(),
            ));
        }
        if !self.records.windows(2).all(|pair| pair[0] <= pair[1]) {
            return Err(Error::InvalidFormat(
                "v2 B-tree internal records are not sorted".into(),
            ));
        }
        Ok(())
    }

    pub fn assert_internal2(&self, record_size: usize) -> Result<()> {
        self.assert_internal(record_size)
    }

    pub fn cache_int_get_initial_load_size() -> usize {
        6
    }

    pub fn cache_int_verify_chksum(image: &[u8]) -> Result<()> {
        verify_image_checksum(image, "v2 B-tree internal")
    }

    pub fn cache_int_image_len(&self, header: &BTreeV2Header, sizeof_addr: usize) -> Result<usize> {
        let infos = compute_node_info(header, sizeof_addr)?;
        let record_size = usize::from(header.record_size);
        let nrec_size = bytes_needed(btree_usize_to_u64(
            infos[0].max_nrec,
            "v2 B-tree leaf record capacity",
        )?);
        let record_bytes = self.records.len().checked_mul(record_size).ok_or_else(|| {
            Error::InvalidFormat("v2 B-tree internal record bytes overflow".into())
        })?;
        let child_entry_size = sizeof_addr.checked_add(nrec_size).ok_or_else(|| {
            Error::InvalidFormat("v2 B-tree internal child entry size overflow".into())
        })?;
        let child_bytes = self
            .children
            .len()
            .checked_mul(child_entry_size)
            .ok_or_else(|| {
                Error::InvalidFormat("v2 B-tree internal child bytes overflow".into())
            })?;
        checked_usize_sum(
            &[6, record_bytes, child_bytes, 4],
            "v2 B-tree internal image",
        )
    }

    pub fn cache_int_serialize(
        &self,
        header: &BTreeV2Header,
        sizeof_addr: usize,
    ) -> Result<Vec<u8>> {
        self.assert_internal(usize::from(header.record_size))?;
        let infos = compute_node_info(header, sizeof_addr)?;
        let nrec_size = bytes_needed(btree_usize_to_u64(
            infos[0].max_nrec,
            "v2 B-tree leaf record capacity",
        )?);
        let mut image = Vec::with_capacity(self.cache_int_image_len(header, sizeof_addr)?);
        image.extend_from_slice(&B2IN_MAGIC);
        image.push(0);
        image.push(header.tree_type);
        for record in &self.records {
            image.extend_from_slice(record);
        }
        for (addr, nrecords) in &self.children {
            write_fixed_le(&mut image, *addr, sizeof_addr)?;
            write_fixed_le(&mut image, u64::from(*nrecords), nrec_size)?;
        }
        let checksum = checksum_metadata(&image);
        image.extend_from_slice(&checksum.to_le_bytes());
        Ok(image)
    }

    pub fn cache_int_notify(&mut self, _action: BTreeV2CacheAction) {}

    pub fn cache_int_free_icr(_image: Vec<u8>) {}

    pub fn int_debug(&self) -> String {
        format!(
            "BTreeV2InternalNode(records={}, children={})",
            self.records.len(),
            self.children.len()
        )
    }
}

/// Pure deserializer for a v2 B-tree internal node — mirrors libhdf5's
/// `H5B2__cache_int_deserialize`.
fn decode_internal_node<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header: &BTreeV2Header,
    node_info: &[NodeInfo],
    addr: u64,
    nrecords: u16,
    depth: u16,
) -> Result<BTreeV2InternalNode> {
    let depth_index = usize::from(depth);
    if depth == 0 || depth_index >= node_info.len() {
        return Err(Error::InvalidFormat(
            "v2 B-tree internal node depth is invalid".into(),
        ));
    }

    reader.seek(addr).map_err(|err| {
        Error::InvalidFormat(format!(
            "failed to seek to v2 B-tree internal node {addr}: {err}"
        ))
    })?;

    let magic = reader.read_bytes(4)?;
    if magic != B2IN_MAGIC {
        return Err(Error::InvalidFormat(
            "invalid v2 B-tree internal magic".into(),
        ));
    }

    validate_node_prefix(reader, header.tree_type, "v2 B-tree internal")?;

    let nrecords_usize = usize::from(nrecords);
    if nrecords_usize > node_info[depth_index].max_nrec {
        return Err(Error::InvalidFormat(format!(
            "v2 B-tree internal node has too many records: {} > {}",
            nrecords, node_info[depth_index].max_nrec
        )));
    }

    let record_size = usize::from(header.record_size);
    let mut records = Vec::with_capacity(nrecords_usize);
    for _ in 0..nrecords {
        records.push(reader.read_bytes(record_size)?);
    }

    let max_nrec_size = bytes_needed(btree_usize_to_u64(
        node_info[0].max_nrec,
        "v2 B-tree leaf record capacity",
    )?);
    let child_all_nrec_size = if depth > 1 {
        node_info[depth_index - 1].cum_max_nrec_size
    } else {
        0
    };

    let child_count = nrecords_usize
        .checked_add(1)
        .ok_or_else(|| Error::InvalidFormat("v2 B-tree child count overflow".into()))?;
    let mut children = Vec::with_capacity(child_count);
    for _ in 0..=nrecords {
        let child_addr = reader.read_addr()?;
        let child_nrecords = btree_u64_to_u16(
            read_var_uint(reader, max_nrec_size)?,
            "v2 B-tree child record count",
        )?;
        let child_all_records = if child_all_nrec_size > 0 {
            read_var_uint(reader, child_all_nrec_size)?
        } else {
            u64::from(child_nrecords)
        };
        if is_undef_addr(child_addr) {
            return Err(Error::InvalidFormat(
                "v2 B-tree internal child address is undefined".into(),
            ));
        }
        let child_max_nrecords = node_info[depth_index - 1].max_nrec;
        if usize::from(child_nrecords) > child_max_nrecords {
            return Err(Error::InvalidFormat(format!(
                "v2 B-tree internal child has too many records: {} > {}",
                child_nrecords, child_max_nrecords
            )));
        }
        if child_all_records < u64::from(child_nrecords) {
            return Err(Error::InvalidFormat(
                "v2 B-tree internal child total record count is inconsistent".into(),
            ));
        }
        if matches!(header.tree_type, 10 | 11) {
            trace_internal_child(
                depth,
                children.len(),
                child_addr,
                child_nrecords,
                child_all_records,
            );
        }
        children.push((child_addr, child_nrecords));
    }

    verify_checksum(reader, addr, "v2 B-tree internal")?;

    Ok(BTreeV2InternalNode { records, children })
}

/// Drive the decoded internal-node into the depth-first record stream —
/// mirrors libhdf5's `H5B2_iterate` for an internal node.
fn read_internal_records<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header: &BTreeV2Header,
    node_info: &[NodeInfo],
    addr: u64,
    nrecords: u16,
    depth: u16,
    records: &mut Vec<Vec<u8>>,
) -> Result<()> {
    let node = decode_internal_node(reader, header, node_info, addr, nrecords, depth)?;

    for idx in 0..node.records.len() {
        read_child_records(
            reader,
            header,
            node_info,
            node.children[idx],
            depth - 1,
            records,
        )?;
        records.push(node.records[idx].clone());
    }
    read_child_records(
        reader,
        header,
        node_info,
        node.children[node.records.len()],
        depth - 1,
        records,
    )?;

    Ok(())
}

#[cfg(feature = "tracehash")]
fn trace_internal_child(
    depth: u16,
    child_index: usize,
    child_addr: u64,
    child_nrecords: u16,
    child_all_records: u64,
) {
    let mut th = tracehash::th_call!("hdf5.chunk_index.btree2.internal_traverse");
    let Ok(child_index_u64) = u64::try_from(child_index) else {
        return;
    };
    th.input_u64(u64::from(depth));
    th.input_u64(child_index_u64);
    th.input_u64(child_addr);
    th.output_value(&(true));
    th.output_u64(u64::from(child_nrecords));
    th.output_u64(child_all_records);
    th.finish();
}

#[cfg(not(feature = "tracehash"))]
fn trace_internal_child(
    _depth: u16,
    _child_index: usize,
    _child_addr: u64,
    _child_nrecords: u16,
    _child_all_records: u64,
) {
}

fn read_child_records<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header: &BTreeV2Header,
    node_info: &[NodeInfo],
    child: (u64, u16),
    depth: u16,
    records: &mut Vec<Vec<u8>>,
) -> Result<()> {
    if depth == 0 {
        read_leaf_records(reader, header, node_info, child.0, child.1, records)
    } else {
        read_internal_records(reader, header, node_info, child.0, child.1, depth, records)
    }
}

fn read_leaf_records<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header: &BTreeV2Header,
    node_info: &[NodeInfo],
    addr: u64,
    nrecords: u16,
    records: &mut Vec<Vec<u8>>,
) -> Result<()> {
    reader.seek(addr).map_err(|err| {
        Error::InvalidFormat(format!("failed to seek to v2 B-tree leaf {addr}: {err}"))
    })?;

    let magic = reader.read_bytes(4)?;
    if magic != B2LF_MAGIC {
        return Err(Error::InvalidFormat("invalid v2 B-tree leaf magic".into()));
    }

    validate_node_prefix(reader, header.tree_type, "v2 B-tree leaf")?;

    if header.record_size == 0 {
        return Err(Error::InvalidFormat(
            "v2 B-tree leaf record size must be positive".into(),
        ));
    }
    let nrecords_usize = usize::from(nrecords);
    if nrecords_usize > node_info[0].max_nrec {
        return Err(Error::InvalidFormat(format!(
            "v2 B-tree leaf has too many records: {} > {}",
            nrecords, node_info[0].max_nrec
        )));
    }

    let record_size = usize::from(header.record_size);
    for _ in 0..nrecords {
        let record = reader.read_bytes(record_size)?;
        records.push(record);
    }

    verify_checksum(reader, addr, "v2 B-tree leaf")?;

    Ok(())
}

fn validate_node_prefix<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    expected_type: u8,
    context: &str,
) -> Result<()> {
    let version = reader.read_u8()?;
    if version != 0 {
        return Err(Error::InvalidFormat(format!("{context} version {version}")));
    }
    let node_type = reader.read_u8()?;
    if node_type != expected_type {
        return Err(Error::InvalidFormat(format!(
            "{context} type {node_type} does not match header type {expected_type}"
        )));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BTreeV2Neighbor {
    Less,
    Greater,
}

#[derive(Debug, Clone)]
pub struct BTreeV2Tree {
    pub header: BTreeV2Header,
    records: Vec<Vec<u8>>,
    closed: bool,
}

impl BTreeV2Tree {
    pub fn create(header: BTreeV2Header) -> Result<Self> {
        header.validate()?;
        Ok(Self {
            header,
            records: Vec::new(),
            closed: false,
        })
    }

    pub fn open(header: BTreeV2Header, mut records: Vec<Vec<u8>>) -> Result<Self> {
        header.validate()?;
        validate_records(&records, usize::from(header.record_size))?;
        records.sort();
        let mut tree = Self {
            header,
            records,
            closed: false,
        };
        tree.sync_header_count()?;
        Ok(tree)
    }

    pub fn insert(&mut self, record: Vec<u8>) -> Result<bool> {
        self.ensure_open()?;
        let inserted = insert_sorted_unique(
            &mut self.records,
            record,
            usize::from(self.header.record_size),
        )?;
        if inserted {
            self.sync_header_count()?;
        }
        Ok(inserted)
    }

    pub fn insert_tree(&mut self, record: Vec<u8>) -> Result<bool> {
        self.insert(record)
    }

    pub fn update(&mut self, record: Vec<u8>) -> Result<bool> {
        self.ensure_open()?;
        validate_record(&record, usize::from(self.header.record_size))?;
        match self
            .records
            .binary_search_by(|probe| probe.as_slice().cmp(&record))
        {
            Ok(idx) => {
                self.records[idx] = record;
                Ok(true)
            }
            Err(_) => Ok(false),
        }
    }

    pub fn find(&self, record: &[u8]) -> Option<Vec<u8>> {
        self.records
            .binary_search_by(|probe| probe.as_slice().cmp(record))
            .ok()
            .map(|idx| self.records[idx].clone())
    }

    pub fn index(&self, index: usize) -> Option<&[u8]> {
        self.records.get(index).map(Vec::as_slice)
    }

    pub fn remove(&mut self, record: &[u8]) -> Result<Option<Vec<u8>>> {
        self.ensure_open()?;
        let removed = match self
            .records
            .binary_search_by(|probe| probe.as_slice().cmp(record))
        {
            Ok(idx) => Some(self.records.remove(idx)),
            Err(_) => None,
        };
        if removed.is_some() {
            self.sync_header_count()?;
        }
        Ok(removed)
    }

    pub fn remove_by_idx(&mut self, index: usize) -> Result<Option<Vec<u8>>> {
        self.ensure_open()?;
        let removed = if index < self.records.len() {
            Some(self.records.remove(index))
        } else {
            None
        };
        if removed.is_some() {
            self.sync_header_count()?;
        }
        Ok(removed)
    }

    pub fn get_nrec(&self) -> u64 {
        self.header.total_records
    }

    pub fn get_addr(&self) -> u64 {
        self.header.root_addr
    }

    pub fn neighbor(&self, record: &[u8], direction: BTreeV2Neighbor) -> Option<Vec<u8>> {
        neighbor_record(&self.records, record, direction)
    }

    pub fn modify<F>(&mut self, record: &[u8], mut update: F) -> Result<bool>
    where
        F: FnMut(&mut Vec<u8>),
    {
        self.ensure_open()?;
        let idx = match self
            .records
            .binary_search_by(|probe| probe.as_slice().cmp(record))
        {
            Ok(idx) => idx,
            Err(_) => return Ok(false),
        };
        update(&mut self.records[idx]);
        validate_record(&self.records[idx], usize::from(self.header.record_size))?;
        self.records.sort();
        Ok(true)
    }

    pub fn close(&mut self) {
        self.closed = true;
    }

    pub fn delete(mut self) {
        self.records.clear();
        self.header.total_records = 0;
        self.closed = true;
    }

    pub fn depend(&self) {}

    pub fn patch_file(&mut self, root_addr: u64) {
        self.header.root_addr = root_addr;
    }

    pub fn locate_record(&self, record: &[u8]) -> std::result::Result<usize, usize> {
        self.records
            .binary_search_by(|probe| probe.as_slice().cmp(record))
    }

    pub fn split1(&self) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
        split_records(&self.records)
    }

    pub fn split_root(&self) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
        self.split1()
    }

    pub fn redistribute2(left: &mut Vec<Vec<u8>>, right: &mut Vec<Vec<u8>>) {
        rebalance_two(left, right);
    }

    pub fn redistribute3(
        left: &mut Vec<Vec<u8>>,
        middle: &mut Vec<Vec<u8>>,
        right: &mut Vec<Vec<u8>>,
    ) {
        let mut all = Vec::new();
        all.append(left);
        all.append(middle);
        all.append(right);
        all.sort();
        let one = all.len() / 3;
        let two = (all.len() * 2) / 3;
        *left = all[..one].to_vec();
        *middle = all[one..two].to_vec();
        *right = all[two..].to_vec();
    }

    pub fn merge2(left: Vec<Vec<u8>>, right: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
        let mut merged = left;
        merged.extend(right);
        merged.sort();
        merged
    }

    pub fn merge3(left: Vec<Vec<u8>>, middle: Vec<Vec<u8>>, right: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
        let mut merged = left;
        merged.extend(middle);
        merged.extend(right);
        merged.sort();
        merged
    }

    pub fn delete_node(records: &mut Vec<Vec<u8>>, record: &[u8]) -> Option<Vec<u8>> {
        let idx = records
            .binary_search_by(|probe| probe.as_slice().cmp(record))
            .ok()?;
        Some(records.remove(idx))
    }

    fn ensure_open(&self) -> Result<()> {
        if self.closed {
            Err(Error::InvalidFormat("v2 B-tree is closed".into()))
        } else {
            Ok(())
        }
    }

    fn sync_header_count(&mut self) -> Result<()> {
        self.header.total_records =
            btree_usize_to_u64(self.records.len(), "v2 B-tree total record count")?;
        self.header.root_nrecords = u16::try_from(self.records.len())
            .map_err(|_| Error::InvalidFormat("v2 B-tree root record count overflow".into()))?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BTreeV2TestContext {
    pub record_size: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct BTreeV2TestRecord {
    pub key: u64,
    pub value: u64,
}

impl BTreeV2TestContext {
    pub fn test_crt_context(record_size: usize) -> Result<Self> {
        if record_size < 8 {
            return Err(Error::InvalidFormat(
                "v2 B-tree test context record size must be at least 8".into(),
            ));
        }
        Ok(Self { record_size })
    }

    pub fn test_dst_context(self) {}
}

impl BTreeV2TestRecord {
    pub fn test_store(key: u64, value: u64) -> Self {
        Self { key, value }
    }

    pub fn test_compare(left: &Self, right: &Self) -> std::cmp::Ordering {
        left.cmp(right)
    }

    pub fn test_encode(&self, context: &BTreeV2TestContext) -> Result<Vec<u8>> {
        let mut out = vec![0; context.record_size];
        checked_window_mut(&mut out, 0, 8, "v2 B-tree test key")?
            .copy_from_slice(&self.key.to_le_bytes());
        if context.record_size >= 16 {
            checked_window_mut(&mut out, 8, 8, "v2 B-tree test value")?
                .copy_from_slice(&self.value.to_le_bytes());
        }
        Ok(out)
    }

    pub fn test_decode(context: &BTreeV2TestContext, image: &[u8]) -> Result<Self> {
        if image.len() < context.record_size || context.record_size < 8 {
            return Err(Error::InvalidFormat(
                "v2 B-tree test record image is truncated".into(),
            ));
        }
        let key = read_u64_le_at(image, 0, "v2 B-tree test key")?;
        let value = if context.record_size >= 16 {
            read_u64_le_at(image, 8, "v2 B-tree test value")?
        } else {
            0
        };
        Ok(Self { key, value })
    }

    pub fn test_debug(&self) -> String {
        format!("BTreeV2TestRecord(key={}, value={})", self.key, self.value)
    }

    pub fn test2_store(key: u64, value: u64) -> Self {
        Self::test_store(key, value)
    }

    pub fn test2_compare(left: &Self, right: &Self) -> std::cmp::Ordering {
        right.cmp(left)
    }

    pub fn test2_encode(&self, context: &BTreeV2TestContext) -> Result<Vec<u8>> {
        let mut out = vec![0; context.record_size];
        checked_window_mut(&mut out, 0, 8, "v2 B-tree test2 key")?
            .copy_from_slice(&self.key.to_be_bytes());
        if context.record_size >= 16 {
            checked_window_mut(&mut out, 8, 8, "v2 B-tree test2 value")?
                .copy_from_slice(&self.value.to_be_bytes());
        }
        Ok(out)
    }

    pub fn test2_decode(context: &BTreeV2TestContext, image: &[u8]) -> Result<Self> {
        if image.len() < context.record_size || context.record_size < 8 {
            return Err(Error::InvalidFormat(
                "v2 B-tree test2 record image is truncated".into(),
            ));
        }
        let key = read_u64_be_at(image, 0, "v2 B-tree test2 key")?;
        let value = if context.record_size >= 16 {
            read_u64_be_at(image, 8, "v2 B-tree test2 value")?
        } else {
            0
        };
        Ok(Self { key, value })
    }

    pub fn test2_debug(&self) -> String {
        format!("BTreeV2Test2Record(key={}, value={})", self.key, self.value)
    }
}

fn checked_window<'a>(data: &'a [u8], pos: usize, len: usize, context: &str) -> Result<&'a [u8]> {
    let end = pos
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))?;
    data.get(pos..end)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))
}

fn checked_window_mut<'a>(
    data: &'a mut [u8],
    pos: usize,
    len: usize,
    context: &str,
) -> Result<&'a mut [u8]> {
    let end = pos
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))?;
    data.get_mut(pos..end)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))
}

fn read_u64_le_at(data: &[u8], pos: usize, context: &str) -> Result<u64> {
    let bytes = checked_window(data, pos, 8, context)?;
    Ok(u64::from_le_bytes(bytes.try_into().map_err(|_| {
        Error::InvalidFormat(format!("{context} is truncated"))
    })?))
}

fn read_u64_be_at(data: &[u8], pos: usize, context: &str) -> Result<u64> {
    let bytes = checked_window(data, pos, 8, context)?;
    Ok(u64::from_be_bytes(bytes.try_into().map_err(|_| {
        Error::InvalidFormat(format!("{context} is truncated"))
    })?))
}

fn read_var_uint<R: Read + Seek>(reader: &mut HdfReader<R>, size: usize) -> Result<u64> {
    if size == 0 || size > 8 {
        return Err(Error::InvalidFormat(format!(
            "invalid v2 B-tree variable integer size {size}"
        )));
    }

    let bytes = reader.read_bytes(size)?;
    let mut value = 0u64;
    for (idx, byte) in bytes.iter().enumerate() {
        value |= u64::from(*byte) << (idx * 8);
    }
    Ok(value)
}

fn validate_record(record: &[u8], record_size: usize) -> Result<()> {
    if record.len() != record_size {
        return Err(Error::InvalidFormat(format!(
            "v2 B-tree record has size {}, expected {record_size}",
            record.len()
        )));
    }
    Ok(())
}

fn validate_records(records: &[Vec<u8>], record_size: usize) -> Result<()> {
    for record in records {
        validate_record(record, record_size)?;
    }
    Ok(())
}

fn insert_sorted_unique(
    records: &mut Vec<Vec<u8>>,
    record: Vec<u8>,
    record_size: usize,
) -> Result<bool> {
    validate_record(&record, record_size)?;
    match records.binary_search_by(|probe| probe.as_slice().cmp(&record)) {
        Ok(_) => Ok(false),
        Err(idx) => {
            records.insert(idx, record);
            Ok(true)
        }
    }
}

fn neighbor_record(
    records: &[Vec<u8>],
    record: &[u8],
    direction: BTreeV2Neighbor,
) -> Option<Vec<u8>> {
    let idx = match records.binary_search_by(|probe| probe.as_slice().cmp(record)) {
        Ok(idx) => idx,
        Err(idx) => idx,
    };
    match direction {
        BTreeV2Neighbor::Less => idx.checked_sub(1).and_then(|pos| records.get(pos)).cloned(),
        BTreeV2Neighbor::Greater => {
            let pos = if records
                .get(idx)
                .is_some_and(|found| found.as_slice() == record)
            {
                idx + 1
            } else {
                idx
            };
            records.get(pos).cloned()
        }
    }
}

fn split_records(records: &[Vec<u8>]) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let mid = records.len() / 2;
    (records[..mid].to_vec(), records[mid..].to_vec())
}

fn rebalance_two(left: &mut Vec<Vec<u8>>, right: &mut Vec<Vec<u8>>) {
    let mut all = Vec::new();
    all.append(left);
    all.append(right);
    all.sort();
    let mid = all.len() / 2;
    *left = all[..mid].to_vec();
    *right = all[mid..].to_vec();
}

fn verify_image_checksum(image: &[u8], context: &str) -> Result<()> {
    if image.len() < 4 {
        return Err(Error::InvalidFormat(format!("{context} image too short")));
    }
    let split = image.len() - 4;
    let stored = u32::from_le_bytes(
        image[split..]
            .try_into()
            .map_err(|_| Error::InvalidFormat(format!("{context} checksum is truncated")))?,
    );
    let computed = checksum_metadata(&image[..split]);
    if stored != computed {
        return Err(Error::InvalidFormat(format!(
            "{context} checksum mismatch: stored={stored:#010x}, computed={computed:#010x}"
        )));
    }
    Ok(())
}

fn write_fixed_le(out: &mut Vec<u8>, value: u64, width: usize) -> Result<()> {
    if width == 0 || width > 8 {
        return Err(Error::InvalidFormat(format!(
            "invalid v2 B-tree integer width {width}"
        )));
    }
    out.extend_from_slice(&value.to_le_bytes()[..width]);
    Ok(())
}

fn bytes_needed(mut value: u64) -> usize {
    let mut bytes = 1usize;
    while value > 0xff {
        value >>= 8;
        bytes += 1;
    }
    bytes
}

fn btree_u32_to_usize(value: u32, context: &str) -> Result<usize> {
    usize::try_from(value)
        .map_err(|_| Error::InvalidFormat(format!("{context} does not fit in usize")))
}

fn btree_u64_to_u16(value: u64, context: &str) -> Result<u16> {
    u16::try_from(value).map_err(|_| Error::InvalidFormat(format!("{context} overflow")))
}

fn btree_usize_to_u64(value: usize, context: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| Error::InvalidFormat(format!("{context} overflow")))
}

fn checked_usize_sum(parts: &[usize], context: &str) -> Result<usize> {
    let mut sum = 0usize;
    for part in parts {
        sum = sum
            .checked_add(*part)
            .ok_or_else(|| Error::InvalidFormat(format!("{context} overflow")))?;
    }
    Ok(sum)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use crate::io::HdfReader;

    use super::{
        checked_usize_sum, checked_window, checked_window_mut, compute_node_info,
        read_leaf_records, BTreeV2Header, BTreeV2InternalNode, BTreeV2LeafNode, BTreeV2Neighbor,
        BTreeV2TestContext, BTreeV2TestRecord, BTreeV2Tree,
    };

    #[test]
    fn btree_v2_usize_sum_rejects_overflow() {
        let err = checked_usize_sum(&[usize::MAX, 1], "btree").unwrap_err();
        assert!(err.to_string().contains("overflow"));
    }

    #[test]
    fn btree_v2_tree_tracks_sorted_records() {
        let header = BTreeV2Header::hdr_create(10, 256, 2, 100, 40).unwrap();
        let mut tree = BTreeV2Tree::create(header).unwrap();

        assert!(tree.insert(vec![2, 0]).unwrap());
        assert!(tree.insert(vec![1, 0]).unwrap());
        assert!(!tree.insert(vec![1, 0]).unwrap());

        assert_eq!(tree.get_nrec(), 2);
        assert_eq!(tree.index(0), Some([1, 0].as_slice()));
        assert_eq!(tree.find(&[2, 0]), Some(vec![2, 0]));
        assert_eq!(
            tree.neighbor(&[1, 0], BTreeV2Neighbor::Greater),
            Some(vec![2, 0])
        );
        assert_eq!(tree.remove(&[1, 0]).unwrap(), Some(vec![1, 0]));
        assert_eq!(tree.get_nrec(), 1);
    }

    #[test]
    fn btree_v2_cache_images_have_valid_checksums() {
        let header = BTreeV2Header::hdr_create(10, 256, 2, 100, 40).unwrap();
        let header_image = header.cache_hdr_serialize().unwrap();
        super::verify_image_checksum(&header_image, "header").unwrap();

        let leaf = BTreeV2LeafNode::create_leaf(vec![vec![1, 0], vec![2, 0]], 2).unwrap();
        let leaf_image = leaf.cache_leaf_serialize(10, 2).unwrap();
        BTreeV2LeafNode::cache_leaf_verify_chksum(&leaf_image).unwrap();

        let internal =
            BTreeV2InternalNode::create_internal(vec![vec![2, 0]], vec![(10, 1), (20, 1)], 2)
                .unwrap();
        let internal_image = internal.cache_int_serialize(&header, 8).unwrap();
        BTreeV2InternalNode::cache_int_verify_chksum(&internal_image).unwrap();
    }

    #[test]
    fn btree_v2_header_rejects_inconsistent_root_record_counts() {
        fn refresh_header_checksum(image: &mut [u8]) {
            let checksum_offset = image.len() - 4;
            let checksum = super::checksum_metadata(
                checked_window(image, 0, checksum_offset, "test v2 B-tree header payload")
                    .expect("header payload should be in range"),
            );
            checked_window_mut(image, checksum_offset, 4, "test v2 B-tree header checksum")
                .expect("header checksum field should be in range")
                .copy_from_slice(&checksum.to_le_bytes());
        }

        let header = BTreeV2Header::hdr_create(10, 256, 2, 100, 40).unwrap();

        let mut empty_with_root_records = header.cache_hdr_serialize().unwrap();
        checked_window_mut(
            &mut empty_with_root_records,
            24,
            2,
            "test v2 B-tree root records",
        )
        .expect("root record field should be in range")
        .copy_from_slice(&1u16.to_le_bytes());
        refresh_header_checksum(&mut empty_with_root_records);
        let mut reader = HdfReader::new(Cursor::new(empty_with_root_records));
        let err = BTreeV2Header::read_at(&mut reader, 0)
            .expect_err("empty v2 B-tree with root records should fail");
        assert!(err.to_string().contains("empty tree"));

        let mut root_exceeds_total = header.cache_hdr_serialize().unwrap();
        checked_window_mut(
            &mut root_exceeds_total,
            16,
            8,
            "test v2 B-tree record count",
        )
        .expect("record count field should be in range")
        .copy_from_slice(&0x100u64.to_le_bytes());
        checked_window_mut(
            &mut root_exceeds_total,
            24,
            2,
            "test v2 B-tree root records",
        )
        .expect("root record field should be in range")
        .copy_from_slice(&2u16.to_le_bytes());
        checked_window_mut(
            &mut root_exceeds_total,
            26,
            8,
            "test v2 B-tree root node address",
        )
        .expect("root address field should be in range")
        .copy_from_slice(&1u64.to_le_bytes());
        refresh_header_checksum(&mut root_exceeds_total);
        let mut reader = HdfReader::new(Cursor::new(root_exceeds_total));
        let err = BTreeV2Header::read_at(&mut reader, 0)
            .expect_err("v2 B-tree root record count above total should fail");
        assert!(err.to_string().contains("root record count"));
    }

    #[test]
    fn btree_v2_leaf_decode_rejects_bad_prefix_and_checksum() {
        let header = BTreeV2Header::hdr_create(10, 256, 2, 100, 40).unwrap();
        let node_info = compute_node_info(&header, 8).unwrap();
        let leaf = BTreeV2LeafNode::create_leaf(vec![vec![1, 0]], 2).unwrap();
        let image = leaf.cache_leaf_serialize(10, 2).unwrap();

        let mut bad_version = image.clone();
        bad_version[4] = 1;
        let mut reader = HdfReader::new(Cursor::new(bad_version));
        let err = read_leaf_records(&mut reader, &header, &node_info, 0, 1, &mut Vec::new())
            .expect_err("bad leaf version should fail");
        assert!(err.to_string().contains("version"));

        let mut bad_type = image.clone();
        bad_type[5] = 11;
        let mut reader = HdfReader::new(Cursor::new(bad_type));
        let err = read_leaf_records(&mut reader, &header, &node_info, 0, 1, &mut Vec::new())
            .expect_err("bad leaf type should fail");
        assert!(err.to_string().contains("type"));

        let mut bad_checksum = image;
        let last = bad_checksum.len() - 1;
        bad_checksum[last] ^= 0xff;
        let mut reader = HdfReader::new(Cursor::new(bad_checksum));
        let err = read_leaf_records(&mut reader, &header, &node_info, 0, 1, &mut Vec::new())
            .expect_err("bad leaf checksum should fail");
        assert!(err.to_string().contains("checksum"));
    }

    #[test]
    fn btree_v2_internal_decode_rejects_bad_prefix_and_checksum() {
        let header = BTreeV2Header {
            depth: 1,
            ..BTreeV2Header::hdr_create(10, 256, 2, 100, 40).unwrap()
        };
        let node_info = compute_node_info(&header, 8).unwrap();
        let internal =
            BTreeV2InternalNode::create_internal(vec![vec![2, 0]], vec![(10, 1), (20, 1)], 2)
                .unwrap();
        let image = internal.cache_int_serialize(&header, 8).unwrap();

        let mut bad_version = image.clone();
        bad_version[4] = 1;
        let mut reader = HdfReader::new(Cursor::new(bad_version));
        let err = super::decode_internal_node(&mut reader, &header, &node_info, 0, 1, 1)
            .expect_err("bad internal version should fail");
        assert!(err.to_string().contains("version"));

        let mut bad_type = image.clone();
        bad_type[5] = 11;
        let mut reader = HdfReader::new(Cursor::new(bad_type));
        let err = super::decode_internal_node(&mut reader, &header, &node_info, 0, 1, 1)
            .expect_err("bad internal type should fail");
        assert!(err.to_string().contains("type"));

        let mut bad_checksum = image;
        let last = bad_checksum.len() - 1;
        bad_checksum[last] ^= 0xff;
        let mut reader = HdfReader::new(Cursor::new(bad_checksum));
        let err = super::decode_internal_node(&mut reader, &header, &node_info, 0, 1, 1)
            .expect_err("bad internal checksum should fail");
        assert!(err.to_string().contains("checksum"));

        let mut bad_child_addr = internal.cache_int_serialize(&header, 8).unwrap();
        checked_window_mut(&mut bad_child_addr, 8, 8, "test v2 B-tree child address")
            .expect("child address field should be in range")
            .copy_from_slice(&u64::MAX.to_le_bytes());
        let checksum_offset = bad_child_addr.len() - 4;
        let checksum = super::checksum_metadata(
            checked_window(
                &bad_child_addr,
                0,
                checksum_offset,
                "test v2 B-tree internal payload",
            )
            .expect("internal payload should be in range"),
        );
        checked_window_mut(
            &mut bad_child_addr,
            checksum_offset,
            4,
            "test v2 B-tree internal checksum",
        )
        .expect("internal checksum field should be in range")
        .copy_from_slice(&checksum.to_le_bytes());
        let mut reader = HdfReader::new(Cursor::new(bad_child_addr));
        let err = super::decode_internal_node(&mut reader, &header, &node_info, 0, 1, 1)
            .expect_err("undefined internal child address should fail");
        assert!(err.to_string().contains("child address"));
    }

    #[test]
    fn btree_v2_test_record_codecs_round_trip() {
        let context = BTreeV2TestContext::test_crt_context(16).unwrap();
        let record = BTreeV2TestRecord::test_store(7, 11);
        let image = record.test_encode(&context).unwrap();
        assert_eq!(
            BTreeV2TestRecord::test_decode(&context, &image).unwrap(),
            record
        );

        let image2 = record.test2_encode(&context).unwrap();
        assert_eq!(
            BTreeV2TestRecord::test2_decode(&context, &image2).unwrap(),
            record
        );
    }
}
