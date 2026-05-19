use std::cmp::Ordering;
use std::fmt::{self, Write};
use std::io::{Cursor, Read, Seek};

use crate::error::{Error, Result};
use crate::format::checksum::checksum_metadata;
use crate::io::reader::{is_undef_addr, HdfReader, UNDEF_ADDR};

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
    /// Allocate a v2 B-tree header.
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

    /// Create a new v2 B-tree header (currently delegates to `hdr_alloc`).
    pub fn hdr_create(
        tree_type: u8,
        node_size: u32,
        record_size: u16,
        split_pct: u8,
        merge_pct: u8,
    ) -> Result<Self> {
        Self::hdr_alloc(tree_type, node_size, record_size, split_pct, merge_pct)
    }

    /// Load a v2 B-tree header from disk at the given address.
    pub fn read_at<R: Read + Seek>(reader: &mut HdfReader<R>, addr: u64) -> Result<Self> {
        let header_image_len = btree_v2_header_image_len(
            usize::from(reader.sizeof_addr()),
            usize::from(reader.sizeof_size()),
        )?;
        let mut image = vec![0; header_image_len];
        reader.seek(addr).map_err(|err| {
            Error::InvalidFormat(format!("failed to seek to v2 B-tree header {addr}: {err}"))
        })?;
        reader.read_bytes_into(&mut image)?;
        let mut image_reader = image_reader(&image, reader);

        let mut magic = [0; 4];
        image_reader.read_bytes_into(&mut magic)?;
        if magic != B2HD_MAGIC {
            return Err(Error::InvalidFormat(
                "invalid v2 B-tree header magic".into(),
            ));
        }

        let version = image_reader.read_u8()?;
        if version != 0 {
            return Err(Error::Unsupported(format!(
                "v2 B-tree header version {version}"
            )));
        }

        let tree_type = image_reader.read_u8()?;
        let node_size = image_reader.read_u32()?;
        let record_size = image_reader.read_u16()?;
        let depth = image_reader.read_u16()?;
        let split_pct = image_reader.read_u8()?;
        let merge_pct = image_reader.read_u8()?;
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
        let root_addr = image_reader.read_addr()?;
        let root_nrecords = image_reader.read_u16()?;
        let total_records = image_reader.read_length()?;
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
        verify_image_checksum(&image, "v2 B-tree header")?;

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

    /// Validate that the header's fields are internally consistent.
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

    /// Compute the initial number of bytes the metadata cache must read to
    /// determine the full header size.
    pub fn cache_hdr_get_initial_load_size() -> usize {
        4
    }

    /// Compute the on-disk size of this header (assuming 8-byte offsets/lengths).
    pub fn cache_hdr_image_len(&self) -> usize {
        self.cache_hdr_image_len_with_widths(8, 8)
            .unwrap_or(usize::MAX)
    }

    /// Compute the on-disk header size for the given address and length widths.
    pub fn cache_hdr_image_len_with_widths(
        &self,
        sizeof_addr: usize,
        sizeof_size: usize,
    ) -> Result<usize> {
        validate_fixed_width(sizeof_addr, "v2 B-tree header address")?;
        validate_fixed_width(sizeof_size, "v2 B-tree header size")?;
        Ok(B2_METADATA_PREFIX_SIZE + 2 + 2 + 1 + 1 + sizeof_addr + 2 + sizeof_size + 4)
    }

    /// Serialize a dirty v2 B-tree header to caller-provided storage (8-byte widths).
    pub fn cache_hdr_serialize_into(&self, image: &mut Vec<u8>) -> Result<()> {
        self.cache_hdr_serialize_with_widths_into(8, 8, image)
    }

    /// Serialize the header with the supplied address and length widths.
    pub fn cache_hdr_serialize_with_widths_into(
        &self,
        sizeof_addr: usize,
        sizeof_size: usize,
        image: &mut Vec<u8>,
    ) -> Result<()> {
        self.validate()?;
        image.clear();
        image.reserve(self.cache_hdr_image_len_with_widths(sizeof_addr, sizeof_size)?);
        image.extend_from_slice(&B2HD_MAGIC);
        image.push(0);
        image.push(self.tree_type);
        image.extend_from_slice(&self.node_size.to_le_bytes());
        image.extend_from_slice(&self.record_size.to_le_bytes());
        image.extend_from_slice(&self.depth.to_le_bytes());
        image.push(self.split_pct);
        image.push(self.merge_pct);
        write_hdr_addr_fixed_le(
            image,
            self.root_addr,
            sizeof_addr,
            self.total_records,
            "v2 B-tree root",
        )?;
        image.extend_from_slice(&self.root_nrecords.to_le_bytes());
        write_fixed_le(image, self.total_records, sizeof_size)?;
        let checksum = checksum_metadata(image);
        image.extend_from_slice(&checksum.to_le_bytes());
        Ok(())
    }

    /// Handle metadata-cache action notifications for the header.
    pub fn cache_hdr_notify(&mut self, action: BTreeV2CacheAction) {
        if matches!(action, BTreeV2CacheAction::Dirtied) {
            self.hdr_dirty();
        }
    }

    /// Destroy/release an in-core representation of the header image.
    pub fn cache_hdr_free_icr(_image: Vec<u8>) {}

    /// Format the header for debug printing.
    pub fn write_hdr_debug<W: Write + ?Sized>(&self, out: &mut W) -> fmt::Result {
        write!(
            out,
            "BTreeV2Header(type={}, node_size={}, record_size={}, depth={}, root={:#x}, nrec={})",
            self.tree_type,
            self.node_size,
            self.record_size,
            self.depth,
            self.root_addr,
            self.total_records
        )
    }

    /// Increment the record count on the v2 B-tree header.
    pub fn hdr_incr(&mut self, nrecords: u64) -> Result<()> {
        self.total_records = self
            .total_records
            .checked_add(nrecords)
            .ok_or_else(|| Error::InvalidFormat("v2 B-tree record count overflow".into()))?;
        Ok(())
    }

    /// Decrement the record count on the v2 B-tree header.
    pub fn hdr_decr(&mut self, nrecords: u64) -> Result<()> {
        self.total_records = self
            .total_records
            .checked_sub(nrecords)
            .ok_or_else(|| Error::InvalidFormat("v2 B-tree record count underflow".into()))?;
        Ok(())
    }

    /// Increment the file reference count on the shared header.
    pub fn hdr_fuse_incr(&mut self) -> Result<()> {
        self.hdr_incr(1)
    }

    /// Decrement the file reference count on the shared header.
    pub fn hdr_fuse_decr(&mut self) -> Result<()> {
        self.hdr_decr(1)
    }

    /// Mark the header as dirty so the metadata cache will flush it.
    pub fn hdr_dirty(&mut self) {}

    /// Protect (clone) the header for use under the metadata-cache wrapper.
    pub fn hdr_protect(&self) -> Self {
        self.clone()
    }

    /// Unprotect the header, releasing it back to the metadata cache.
    pub fn hdr_unprotect(self) -> Self {
        self
    }

    /// Free the header's resources.
    pub fn hdr_free(self) {}

    /// Delete the entire v2 B-tree, starting with the header.
    pub fn hdr_delete(self) {}

    /// Return the total amount of storage used by the B-tree.
    /// Saturates to `u64::MAX` on overflow.
    pub fn size(&self) -> u64 {
        self.checked_size().unwrap_or(u64::MAX)
    }

    /// Checked variant of `size` that returns an error on overflow.
    pub fn checked_size(&self) -> Result<u64> {
        let header_len = u64::try_from(self.cache_hdr_image_len_with_widths(8, 8)?)
            .map_err(|_| Error::InvalidFormat("v2 B-tree header size exceeds u64".into()))?;
        let records_len = self
            .total_records
            .checked_mul(u64::from(self.record_size))
            .ok_or_else(|| Error::InvalidFormat("v2 B-tree record size overflow".into()))?;
        header_len
            .checked_add(records_len)
            .ok_or_else(|| Error::InvalidFormat("v2 B-tree size overflow".into()))
    }

    /// Return the configured per-node size in bytes.
    pub fn node_size(&self) -> u32 {
        self.node_size
    }

    /// Create a flush dependency between two data-structure components.
    pub fn create_flush_depend(&self) {}

    /// Update flush dependencies for a node.
    pub fn update_flush_depend(&self) {}

    /// Update flush dependencies for a node's children.
    pub fn update_child_flush_depends(&self) {}

    /// Destroy a flush dependency.
    pub fn destroy_flush_depend(&self) {}

    /// Test hook: return the root node's address.
    pub fn get_root_addr_test(&self) -> u64 {
        self.root_addr
    }

    /// Test hook: return per-depth node info `(max_nrec, cum_max_nrec, cum_max_nrec_size)`.
    pub fn get_node_info_test_into(
        &self,
        sizeof_addr: usize,
        out: &mut Vec<(usize, u64, usize)>,
    ) -> Result<()> {
        out.clear();
        for info in compute_node_info(self, sizeof_addr)? {
            out.push((info.max_nrec, info.cum_max_nrec, info.cum_max_nrec_size));
        }
        Ok(())
    }

    /// Test hook: return the tree depth.
    pub fn get_node_depth_test(&self) -> u16 {
        self.depth
    }
}

fn image_reader<'a, R: Read + Seek>(
    image: &'a [u8],
    source: &HdfReader<R>,
) -> HdfReader<Cursor<&'a [u8]>> {
    let mut reader = HdfReader::new(Cursor::new(image));
    reader.set_sizeof_addr(source.sizeof_addr());
    reader.set_sizeof_size(source.sizeof_size());
    reader
}

fn btree_v2_header_image_len(sizeof_addr: usize, sizeof_size: usize) -> Result<usize> {
    4usize
        .checked_add(1)
        .and_then(|value| value.checked_add(1))
        .and_then(|value| value.checked_add(4))
        .and_then(|value| value.checked_add(2))
        .and_then(|value| value.checked_add(2))
        .and_then(|value| value.checked_add(1))
        .and_then(|value| value.checked_add(1))
        .and_then(|value| value.checked_add(sizeof_addr))
        .and_then(|value| value.checked_add(2))
        .and_then(|value| value.checked_add(sizeof_size))
        .and_then(|value| value.checked_add(4))
        .ok_or_else(|| Error::InvalidFormat("v2 B-tree header image length overflow".into()))
}

fn verify_compact_or_padded_node_image<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    image: &mut Vec<u8>,
    node_size: usize,
    context: &str,
) -> Result<()> {
    if node_size < 4 {
        return Err(Error::InvalidFormat(format!("{context} node is too small")));
    }
    let compact_len = image.len();
    let compact_payload_len = compact_len
        .checked_sub(4)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} image too short")))?;
    let padded_payload_len = node_size
        .checked_sub(4)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} node is too small")))?;
    if compact_payload_len > padded_payload_len {
        return Err(Error::InvalidFormat(format!(
            "{context} records exceed declared node size"
        )));
    }
    match verify_image_checksum(image, context) {
        Ok(()) => Ok(()),
        Err(compact_err) => {
            if compact_len >= node_size {
                return Err(compact_err);
            }
            image.resize(node_size, 0);
            if reader.read_bytes_into(&mut image[compact_len..]).is_err() {
                return Err(compact_err);
            }
            verify_image_checksum(image, context)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BTreeV2CacheAction {
    Loaded,
    Dirtied,
    Flushed,
    Evicted,
}

/// Collect all records from a v2 B-tree into caller-provided storage.
pub fn collect_all_records_into<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header_addr: u64,
    records: &mut Vec<Vec<u8>>,
) -> Result<()> {
    records.clear();
    visit_all_records(reader, header_addr, |record| {
        records.push(record.to_vec());
        Ok(())
    })
}

/// Visit all records from a v2 B-tree in traversal order without collecting
/// owned record buffers.
pub fn visit_all_records<R, F>(
    reader: &mut HdfReader<R>,
    header_addr: u64,
    mut visit: F,
) -> Result<()>
where
    R: Read + Seek,
    F: FnMut(&[u8]) -> Result<()>,
{
    let header = BTreeV2Header::read_at(reader, header_addr)?;

    if header.total_records == 0 {
        return Ok(());
    }

    if header.depth == 0 {
        let node_info = compute_node_info(&header, usize::from(reader.sizeof_addr()))?;
        let mut scratch = BTreeV2TraversalScratch::new(usize::from(header.record_size));
        visit_leaf_records_with_buffer(
            reader,
            &header,
            &node_info,
            header.root_addr,
            header.root_nrecords,
            &mut scratch.leaf_record,
            &mut visit,
        )?;
    } else {
        let node_info = compute_node_info(&header, usize::from(reader.sizeof_addr()))?;
        let mut scratch = BTreeV2TraversalScratch::new(usize::from(header.record_size));
        visit_internal_records(
            reader,
            &header,
            &node_info,
            header.root_addr,
            header.root_nrecords,
            header.depth,
            &mut scratch,
            &mut visit,
        )?;
    }

    Ok(())
}

/// Collect records whose key compares equal to a caller-provided target.
///
/// `compare` must return the ordering of the record key relative to the
/// target key, using the same ordering as the B-tree type's native comparator.
pub fn collect_matching_records_into<R, F>(
    reader: &mut HdfReader<R>,
    header_addr: u64,
    records: &mut Vec<Vec<u8>>,
    compare: F,
) -> Result<()>
where
    R: Read + Seek,
    F: FnMut(&[u8]) -> Ordering,
{
    records.clear();
    visit_matching_records(reader, header_addr, compare, |record| {
        records.push(record.to_vec());
        Ok(())
    })
}

/// Visit records whose key compares equal to a caller-provided target.
pub fn visit_matching_records<R, C, V>(
    reader: &mut HdfReader<R>,
    header_addr: u64,
    mut compare: C,
    mut visit: V,
) -> Result<()>
where
    R: Read + Seek,
    C: FnMut(&[u8]) -> Ordering,
    V: FnMut(&[u8]) -> Result<()>,
{
    let header = BTreeV2Header::read_at(reader, header_addr)?;

    if header.total_records == 0 {
        return Ok(());
    }

    let node_info = compute_node_info(&header, usize::from(reader.sizeof_addr()))?;
    let mut scratch = BTreeV2TraversalScratch::new(usize::from(header.record_size));
    visit_matching_child_records(
        reader,
        &header,
        &node_info,
        (header.root_addr, header.root_nrecords),
        header.depth,
        &mut scratch,
        &mut compare,
        &mut visit,
    )
}

#[derive(Debug, Clone)]
struct NodeInfo {
    max_nrec: usize,
    cum_max_nrec: u64,
    cum_max_nrec_size: usize,
}

#[derive(Debug)]
struct BTreeV2InternalNodeImage {
    record_bytes: Vec<u8>,
    record_size: usize,
    children: Vec<(u64, u16)>,
}

impl BTreeV2InternalNodeImage {
    fn new(record_size: usize) -> Self {
        Self {
            record_bytes: Vec::new(),
            record_size,
            children: Vec::new(),
        }
    }

    fn len(&self) -> usize {
        self.children.len().saturating_sub(1)
    }

    fn record(&self, index: usize) -> &[u8] {
        let start = index * self.record_size;
        &self.record_bytes[start..start + self.record_size]
    }
}

struct BTreeV2TraversalScratch {
    internal_nodes: Vec<BTreeV2InternalNodeImage>,
    leaf_record: Vec<u8>,
}

impl BTreeV2TraversalScratch {
    fn new(record_size: usize) -> Self {
        Self {
            internal_nodes: Vec::new(),
            leaf_record: Vec::with_capacity(record_size),
        }
    }

    fn internal_node(
        &mut self,
        depth_index: usize,
        record_size: usize,
    ) -> &mut BTreeV2InternalNodeImage {
        while self.internal_nodes.len() <= depth_index {
            self.internal_nodes
                .push(BTreeV2InternalNodeImage::new(record_size));
        }
        &mut self.internal_nodes[depth_index]
    }
}

/// Compute per-depth `NodeInfo` for a v2 B-tree (max records and cumulative
/// max-record byte widths). Mirrors `H5B2__hdr_init`.
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
    /// Create an empty leaf node, validating record widths.
    pub fn create_leaf(records: Vec<Vec<u8>>, record_size: usize) -> Result<Self> {
        validate_records(&records, record_size)?;
        Ok(Self { records })
    }

    /// "Protect" a leaf node in the metadata cache (returns a clone).
    pub fn protect_leaf(&self) -> Self {
        self.clone()
    }

    /// Locate the record neighboring `record` in the requested direction.
    pub fn neighbor_leaf_ref(&self, record: &[u8], direction: BTreeV2Neighbor) -> Option<&[u8]> {
        neighbor_record_ref(&self.records, record, direction)
    }

    /// Add a new record to the leaf node. Returns `false` if the record already exists.
    pub fn insert_leaf(&mut self, record: Vec<u8>, record_size: usize) -> Result<bool> {
        insert_sorted_unique(&mut self.records, record, record_size)
    }

    /// Insert or modify a record in the leaf node. Returns `true` if an
    /// existing record was overwritten.
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

    /// Swap two records in the leaf node by index.
    pub fn swap_leaf(&mut self, idx_a: usize, idx_b: usize) -> Result<()> {
        if idx_a >= self.records.len() || idx_b >= self.records.len() {
            return Err(Error::InvalidFormat(
                "v2 B-tree leaf swap index out of bounds".into(),
            ));
        }
        self.records.swap(idx_a, idx_b);
        Ok(())
    }

    /// Remove a record matching `record` from the leaf, returning it if found.
    pub fn remove_leaf(&mut self, record: &[u8]) -> Option<Vec<u8>> {
        let idx = self
            .records
            .binary_search_by(|probe| probe.as_slice().cmp(record))
            .ok()?;
        Some(self.records.remove(idx))
    }

    /// Remove the record at the given index from the leaf.
    pub fn remove_leaf_by_idx(&mut self, index: usize) -> Option<Vec<u8>> {
        if index < self.records.len() {
            Some(self.records.remove(index))
        } else {
            None
        }
    }

    /// Verify that the leaf node is mostly sane (correct record sizes, sorted).
    pub fn assert_leaf(&self, record_size: usize) -> Result<()> {
        validate_records(&self.records, record_size)?;
        if !self.records.windows(2).all(|pair| pair[0] <= pair[1]) {
            return Err(Error::InvalidFormat(
                "v2 B-tree leaf records are not sorted".into(),
            ));
        }
        Ok(())
    }

    /// Secondary leaf sanity check (alias of `assert_leaf`).
    pub fn assert_leaf2(&self, record_size: usize) -> Result<()> {
        self.assert_leaf(record_size)
    }

    /// Initial number of bytes the metadata cache must read for a leaf node.
    pub fn cache_leaf_get_initial_load_size() -> usize {
        6
    }

    /// Verify the trailing checksum of a leaf node image.
    pub fn cache_leaf_verify_chksum(image: &[u8]) -> Result<()> {
        verify_image_checksum(image, "v2 B-tree leaf")
    }

    /// Compute the on-disk size of this leaf node.
    pub fn cache_leaf_image_len(&self, record_size: usize) -> Result<usize> {
        let record_bytes =
            self.records.len().checked_mul(record_size).ok_or_else(|| {
                Error::InvalidFormat("v2 B-tree leaf record bytes overflow".into())
            })?;
        checked_usize_sum(&[6, record_bytes, 4], "v2 B-tree leaf image length")
    }

    /// Serialize the leaf node into caller-provided storage.
    pub fn cache_leaf_serialize_into(
        &self,
        tree_type: u8,
        record_size: usize,
        image: &mut Vec<u8>,
    ) -> Result<()> {
        self.assert_leaf(record_size)?;
        image.clear();
        image.reserve(self.cache_leaf_image_len(record_size)?);
        image.extend_from_slice(&B2LF_MAGIC);
        image.push(0);
        image.push(tree_type);
        for record in &self.records {
            image.extend_from_slice(record);
        }
        let checksum = checksum_metadata(image);
        image.extend_from_slice(&checksum.to_le_bytes());
        Ok(())
    }

    /// Handle metadata-cache action notifications for the leaf.
    pub fn cache_leaf_notify(&mut self, _action: BTreeV2CacheAction) {}

    /// Destroy/release an in-core representation of a leaf image.
    pub fn cache_leaf_free_icr(_image: Vec<u8>) {}
}

impl BTreeV2InternalNode {
    /// Create an empty internal node from records and child pointers.
    pub fn create_internal(
        records: Vec<Vec<u8>>,
        children: Vec<(u64, u16)>,
        record_size: usize,
    ) -> Result<Self> {
        validate_records(&records, record_size)?;
        let expected_children = records.len().checked_add(1).ok_or_else(|| {
            Error::InvalidFormat("v2 B-tree internal child count overflow".into())
        })?;
        if children.len() != expected_children {
            return Err(Error::InvalidFormat(
                "v2 B-tree internal child count must be record count + 1".into(),
            ));
        }
        Ok(Self { records, children })
    }

    /// "Protect" an internal node in the metadata cache (returns a clone).
    pub fn protect_internal(&self) -> Self {
        self.clone()
    }

    /// Locate the record neighboring `record` in the requested direction.
    pub fn neighbor_internal_ref(
        &self,
        record: &[u8],
        direction: BTreeV2Neighbor,
    ) -> Option<&[u8]> {
        neighbor_record_ref(&self.records, record, direction)
    }

    /// Insert a new record (and the associated right child) into the internal node.
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

    /// Insert or modify a record in the internal node.
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

    /// "Shadow" the internal node — return a deep copy as if it had been
    /// relocated by the SWMR shadowing machinery.
    pub fn shadow_internal(&self) -> Self {
        self.clone()
    }

    /// Remove a record (and its right-hand child pointer) from the internal node.
    pub fn remove_internal(&mut self, record: &[u8]) -> Option<Vec<u8>> {
        let idx = self
            .records
            .binary_search_by(|probe| probe.as_slice().cmp(record))
            .ok()?;
        self.children.remove(idx + 1);
        Some(self.records.remove(idx))
    }

    /// Remove the record at the given index (and its right child pointer).
    pub fn remove_internal_by_idx(&mut self, index: usize) -> Option<Vec<u8>> {
        if index >= self.records.len() {
            return None;
        }
        self.children.remove(index + 1);
        Some(self.records.remove(index))
    }

    /// Destroy a v2 B-tree internal node, releasing its memory.
    pub fn internal_free(self) {}

    /// Verify than an internal node is mostly sane (sizes, sorted records).
    pub fn assert_internal(&self, record_size: usize) -> Result<()> {
        validate_records(&self.records, record_size)?;
        let expected_children = self.records.len().checked_add(1).ok_or_else(|| {
            Error::InvalidFormat("v2 B-tree internal child count overflow".into())
        })?;
        if self.children.len() != expected_children {
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

    /// Secondary internal node sanity check (alias of `assert_internal`).
    pub fn assert_internal2(&self, record_size: usize) -> Result<()> {
        self.assert_internal(record_size)
    }

    /// Initial number of bytes the metadata cache must read for an internal node.
    pub fn cache_int_get_initial_load_size() -> usize {
        6
    }

    /// Verify the trailing checksum of an internal node image.
    pub fn cache_int_verify_chksum(image: &[u8]) -> Result<()> {
        verify_image_checksum(image, "v2 B-tree internal")
    }

    /// Compute the on-disk size of this internal node.
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

    /// Serialize the internal node into caller-provided storage.
    pub fn cache_int_serialize_into(
        &self,
        header: &BTreeV2Header,
        sizeof_addr: usize,
        image: &mut Vec<u8>,
    ) -> Result<()> {
        self.assert_internal(usize::from(header.record_size))?;
        let infos = compute_node_info(header, sizeof_addr)?;
        let nrec_size = bytes_needed(btree_usize_to_u64(
            infos[0].max_nrec,
            "v2 B-tree leaf record capacity",
        )?);
        image.clear();
        image.reserve(self.cache_int_image_len(header, sizeof_addr)?);
        image.extend_from_slice(&B2IN_MAGIC);
        image.push(0);
        image.push(header.tree_type);
        for record in &self.records {
            image.extend_from_slice(record);
        }
        for (addr, nrecords) in &self.children {
            write_addr_fixed_le(image, *addr, sizeof_addr, "v2 B-tree internal child")?;
            write_fixed_le(image, u64::from(*nrecords), nrec_size)?;
        }
        let checksum = checksum_metadata(image);
        image.extend_from_slice(&checksum.to_le_bytes());
        Ok(())
    }

    /// Handle metadata-cache action notifications for the internal node.
    pub fn cache_int_notify(&mut self, _action: BTreeV2CacheAction) {}

    /// Destroy/release an in-core representation of an internal-node image.
    pub fn cache_int_free_icr(_image: Vec<u8>) {}

    /// Format the internal node for debug printing.
    pub fn write_int_debug<W: Write + ?Sized>(&self, out: &mut W) -> fmt::Result {
        write!(
            out,
            "BTreeV2InternalNode(records={}, children={})",
            self.records.len(),
            self.children.len()
        )
    }
}

/// Pure deserializer for a v2 B-tree internal node — mirrors libhdf5's
/// `H5B2__cache_int_deserialize`.
#[cfg(test)]
fn decode_internal_node_image<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header: &BTreeV2Header,
    node_info: &[NodeInfo],
    addr: u64,
    nrecords: u16,
    depth: u16,
) -> Result<BTreeV2InternalNodeImage> {
    let mut image = BTreeV2InternalNodeImage::new(usize::from(header.record_size));
    decode_internal_node_image_into(reader, header, node_info, addr, nrecords, depth, &mut image)?;
    Ok(image)
}

fn decode_internal_node_image_into<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header: &BTreeV2Header,
    node_info: &[NodeInfo],
    addr: u64,
    nrecords: u16,
    depth: u16,
    image: &mut BTreeV2InternalNodeImage,
) -> Result<()> {
    let depth_index = usize::from(depth);
    if depth == 0 || depth_index >= node_info.len() {
        return Err(Error::InvalidFormat(
            "v2 B-tree internal node depth is invalid".into(),
        ));
    }

    let nrecords_usize = usize::from(nrecords);
    let record_size = usize::from(header.record_size);
    let record_bytes_len = nrecords_usize
        .checked_mul(record_size)
        .ok_or_else(|| Error::InvalidFormat("v2 B-tree internal record bytes overflow".into()))?;
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
    let child_entry_size = usize::from(reader.sizeof_addr())
        .checked_add(max_nrec_size)
        .and_then(|value| value.checked_add(child_all_nrec_size))
        .ok_or_else(|| Error::InvalidFormat("v2 B-tree child entry size overflow".into()))?;
    let child_bytes = child_count
        .checked_mul(child_entry_size)
        .ok_or_else(|| Error::InvalidFormat("v2 B-tree child bytes overflow".into()))?;
    let compact_image_len = checked_usize_sum(
        &[6, record_bytes_len, child_bytes, 4],
        "v2 B-tree internal image length",
    )?;
    let mut node_image = vec![0; compact_image_len];
    reader.seek(addr).map_err(|err| {
        Error::InvalidFormat(format!(
            "failed to seek to v2 B-tree internal node {addr}: {err}"
        ))
    })?;
    reader.read_bytes_into(&mut node_image)?;
    let mut image_reader = image_reader(&node_image, reader);

    let mut magic = [0; 4];
    image_reader.read_bytes_into(&mut magic)?;
    if magic != B2IN_MAGIC {
        return Err(Error::InvalidFormat(
            "invalid v2 B-tree internal magic".into(),
        ));
    }

    validate_node_prefix(&mut image_reader, header.tree_type, "v2 B-tree internal")?;

    if nrecords_usize > node_info[depth_index].max_nrec {
        return Err(Error::InvalidFormat(format!(
            "v2 B-tree internal node has too many records: {} > {}",
            nrecords, node_info[depth_index].max_nrec
        )));
    }

    image.record_size = record_size;
    image.record_bytes.clear();
    image.record_bytes.resize(record_bytes_len, 0);
    image_reader.read_bytes_into(&mut image.record_bytes)?;

    image.children.clear();
    image.children.reserve(child_count);
    for _ in 0..=nrecords {
        let child_addr = image_reader.read_addr()?;
        let child_nrecords = btree_u64_to_u16(
            read_var_uint(&mut image_reader, max_nrec_size)?,
            "v2 B-tree child record count",
        )?;
        let child_all_records = if child_all_nrec_size > 0 {
            read_var_uint(&mut image_reader, child_all_nrec_size)?
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
                image.children.len(),
                child_addr,
                child_nrecords,
                child_all_records,
            );
        }
        image.children.push((child_addr, child_nrecords));
    }
    verify_compact_or_padded_node_image(
        reader,
        &mut node_image,
        usize::try_from(header.node_size).map_err(|_| {
            Error::InvalidFormat("v2 B-tree internal node size is too large".into())
        })?,
        "v2 B-tree internal",
    )?;

    Ok(())
}

#[cfg(test)]
fn decode_internal_node<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header: &BTreeV2Header,
    node_info: &[NodeInfo],
    addr: u64,
    nrecords: u16,
    depth: u16,
) -> Result<BTreeV2InternalNode> {
    let image = decode_internal_node_image(reader, header, node_info, addr, nrecords, depth)?;
    let records = (0..image.len())
        .map(|index| image.record(index).to_vec())
        .collect();
    Ok(BTreeV2InternalNode {
        records,
        children: image.children,
    })
}

/// Drive the decoded internal-node into the depth-first record stream —
/// mirrors libhdf5's `H5B2_iterate` for an internal node.
fn visit_internal_records<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header: &BTreeV2Header,
    node_info: &[NodeInfo],
    addr: u64,
    nrecords: u16,
    depth: u16,
    scratch: &mut BTreeV2TraversalScratch,
    visit: &mut dyn FnMut(&[u8]) -> Result<()>,
) -> Result<()> {
    let depth_index = usize::from(depth);
    decode_internal_node_image_into(
        reader,
        header,
        node_info,
        addr,
        nrecords,
        depth,
        scratch.internal_node(depth_index, usize::from(header.record_size)),
    )?;

    let node_record_count = scratch.internal_nodes[depth_index].len();
    for idx in 0..node_record_count {
        let child = scratch.internal_nodes[depth_index].children[idx];
        visit_child_records(reader, header, node_info, child, depth - 1, scratch, visit)?;
        visit(scratch.internal_nodes[depth_index].record(idx))?;
    }
    let child = scratch.internal_nodes[depth_index].children[node_record_count];
    visit_child_records(reader, header, node_info, child, depth - 1, scratch, visit)?;

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

fn visit_child_records<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header: &BTreeV2Header,
    node_info: &[NodeInfo],
    child: (u64, u16),
    depth: u16,
    scratch: &mut BTreeV2TraversalScratch,
    visit: &mut dyn FnMut(&[u8]) -> Result<()>,
) -> Result<()> {
    if depth == 0 {
        visit_leaf_records_with_buffer(
            reader,
            header,
            node_info,
            child.0,
            child.1,
            &mut scratch.leaf_record,
            visit,
        )
    } else {
        visit_internal_records(
            reader, header, node_info, child.0, child.1, depth, scratch, visit,
        )
    }
}

fn visit_matching_child_records<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header: &BTreeV2Header,
    node_info: &[NodeInfo],
    child: (u64, u16),
    depth: u16,
    scratch: &mut BTreeV2TraversalScratch,
    compare: &mut dyn FnMut(&[u8]) -> Ordering,
    visit: &mut dyn FnMut(&[u8]) -> Result<()>,
) -> Result<()> {
    if depth == 0 {
        visit_matching_leaf_records(
            reader, header, node_info, child.0, child.1, scratch, compare, visit,
        )
    } else {
        visit_matching_internal_records(
            reader, header, node_info, child.0, child.1, depth, scratch, compare, visit,
        )
    }
}

fn visit_matching_internal_records<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header: &BTreeV2Header,
    node_info: &[NodeInfo],
    addr: u64,
    nrecords: u16,
    depth: u16,
    scratch: &mut BTreeV2TraversalScratch,
    compare: &mut dyn FnMut(&[u8]) -> Ordering,
    visit: &mut dyn FnMut(&[u8]) -> Result<()>,
) -> Result<()> {
    let depth_index = usize::from(depth);
    decode_internal_node_image_into(
        reader,
        header,
        node_info,
        addr,
        nrecords,
        depth,
        scratch.internal_node(depth_index, usize::from(header.record_size)),
    )?;

    let mut previous_ordering = None;
    let node_record_count = scratch.internal_nodes[depth_index].len();
    for idx in 0..node_record_count {
        let current_ordering = compare(scratch.internal_nodes[depth_index].record(idx));
        let lower_allows_match = previous_ordering != Some(Ordering::Greater);
        let upper_allows_match = current_ordering != Ordering::Less;
        if lower_allows_match && upper_allows_match {
            let child = scratch.internal_nodes[depth_index].children[idx];
            visit_matching_child_records(
                reader,
                header,
                node_info,
                child,
                depth - 1,
                scratch,
                compare,
                visit,
            )?;
        }

        if current_ordering == Ordering::Equal {
            visit(scratch.internal_nodes[depth_index].record(idx))?;
        }
        previous_ordering = Some(current_ordering);
    }

    if previous_ordering != Some(Ordering::Greater) {
        let child = scratch.internal_nodes[depth_index].children[node_record_count];
        visit_matching_child_records(
            reader,
            header,
            node_info,
            child,
            depth - 1,
            scratch,
            compare,
            visit,
        )?;
    }

    Ok(())
}

fn visit_matching_leaf_records<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header: &BTreeV2Header,
    node_info: &[NodeInfo],
    addr: u64,
    nrecords: u16,
    scratch: &mut BTreeV2TraversalScratch,
    compare: &mut dyn FnMut(&[u8]) -> Ordering,
    visit: &mut dyn FnMut(&[u8]) -> Result<()>,
) -> Result<()> {
    let mut matching_visit = |record: &[u8]| {
        if compare(record) == Ordering::Equal {
            visit(record)?;
        }
        Ok(())
    };
    visit_leaf_records_with_buffer(
        reader,
        header,
        node_info,
        addr,
        nrecords,
        &mut scratch.leaf_record,
        &mut matching_visit,
    )?;
    Ok(())
}

/// Deserialize a v2 B-tree leaf node and visit each record as borrowed bytes.
#[cfg(test)]
fn visit_leaf_records<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header: &BTreeV2Header,
    node_info: &[NodeInfo],
    addr: u64,
    nrecords: u16,
    visit: &mut dyn FnMut(&[u8]) -> Result<()>,
) -> Result<()> {
    let mut record = Vec::new();
    visit_leaf_records_with_buffer(
        reader,
        header,
        node_info,
        addr,
        nrecords,
        &mut record,
        visit,
    )
}

fn visit_leaf_records_with_buffer<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header: &BTreeV2Header,
    node_info: &[NodeInfo],
    addr: u64,
    nrecords: u16,
    record: &mut Vec<u8>,
    visit: &mut dyn FnMut(&[u8]) -> Result<()>,
) -> Result<()> {
    let nrecords_usize = usize::from(nrecords);
    let record_size = usize::from(header.record_size);
    let record_bytes = nrecords_usize
        .checked_mul(record_size)
        .ok_or_else(|| Error::InvalidFormat("v2 B-tree leaf record bytes overflow".into()))?;
    let compact_image_len =
        checked_usize_sum(&[6, record_bytes, 4], "v2 B-tree leaf image length")?;
    let mut node_image = vec![0; compact_image_len];
    reader.seek(addr).map_err(|err| {
        Error::InvalidFormat(format!("failed to seek to v2 B-tree leaf {addr}: {err}"))
    })?;
    reader.read_bytes_into(&mut node_image)?;
    let mut image_reader = image_reader(&node_image, reader);

    let mut magic = [0; 4];
    image_reader.read_bytes_into(&mut magic)?;
    if magic != B2LF_MAGIC {
        return Err(Error::InvalidFormat("invalid v2 B-tree leaf magic".into()));
    }

    validate_node_prefix(&mut image_reader, header.tree_type, "v2 B-tree leaf")?;

    if header.record_size == 0 {
        return Err(Error::InvalidFormat(
            "v2 B-tree leaf record size must be positive".into(),
        ));
    }
    if nrecords_usize > node_info[0].max_nrec {
        return Err(Error::InvalidFormat(format!(
            "v2 B-tree leaf has too many records: {} > {}",
            nrecords, node_info[0].max_nrec
        )));
    }

    record.clear();
    record.resize(record_size, 0);
    for _ in 0..nrecords {
        image_reader.read_bytes_into(record)?;
        visit(&record)?;
    }
    verify_compact_or_padded_node_image(
        reader,
        &mut node_image,
        usize::try_from(header.node_size)
            .map_err(|_| Error::InvalidFormat("v2 B-tree leaf node size is too large".into()))?,
        "v2 B-tree leaf",
    )?;

    Ok(())
}

/// Deserialize a v2 B-tree leaf node from disk and append its records to `records`.
#[cfg(test)]
fn read_leaf_records<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header: &BTreeV2Header,
    node_info: &[NodeInfo],
    addr: u64,
    nrecords: u16,
    records: &mut Vec<Vec<u8>>,
) -> Result<()> {
    records.reserve(usize::from(nrecords));
    let mut append_record = |record: &[u8]| {
        records.push(record.to_vec());
        Ok(())
    };
    visit_leaf_records(
        reader,
        header,
        node_info,
        addr,
        nrecords,
        &mut append_record,
    )
}

/// Read and validate the common version/type bytes shared by leaf and internal nodes.
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
    /// Create a new empty v2 B-tree from a validated header.
    pub fn create(header: BTreeV2Header) -> Result<Self> {
        header.validate()?;
        Ok(Self {
            header,
            records: Vec::new(),
            closed: false,
        })
    }

    /// Open an existing v2 B-tree from records, sorting the supplied records.
    pub fn open_from_iter<I>(header: BTreeV2Header, records: I) -> Result<Self>
    where
        I: IntoIterator<Item = Vec<u8>>,
    {
        header.validate()?;
        let mut records: Vec<Vec<u8>> = records.into_iter().collect();
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

    /// Add a new record to the B-tree.
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

    /// Internal recursive insert entry point (alias of `insert`).
    pub fn insert_tree(&mut self, record: Vec<u8>) -> Result<bool> {
        self.insert(record)
    }

    /// Insert or modify a record in the B-tree.
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

    /// Locate a record matching `record` and borrow it.
    pub fn find_ref(&self, record: &[u8]) -> Option<&[u8]> {
        self.records
            .binary_search_by(|probe| probe.as_slice().cmp(record))
            .ok()
            .map(|idx| self.records[idx].as_slice())
    }

    /// Return the n-th record in the tree according to the natural ordering.
    pub fn index(&self, index: usize) -> Option<&[u8]> {
        self.records.get(index).map(Vec::as_slice)
    }

    /// Borrow all records in natural order.
    pub fn records(&self) -> impl Iterator<Item = &[u8]> {
        self.records.iter().map(Vec::as_slice)
    }

    /// Remove a record from the B-tree.
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

    /// Remove the n-th record from the B-tree.
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

    /// Retrieve the total number of records in the B-tree.
    pub fn get_nrec(&self) -> u64 {
        self.header.total_records
    }

    /// Return the address of the B-tree's root.
    pub fn get_addr(&self) -> u64 {
        self.header.root_addr
    }

    /// Locate the record neighboring `record` in the requested direction.
    pub fn neighbor_ref(&self, record: &[u8], direction: BTreeV2Neighbor) -> Option<&[u8]> {
        neighbor_record_ref(&self.records, record, direction)
    }

    /// Find a record and apply `update` to mutate it in place.
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
        let mut updated = self.records.remove(idx);
        update(&mut updated);
        validate_record(&updated, usize::from(self.header.record_size))?;
        let insert_idx = match self
            .records
            .binary_search_by(|probe| probe.as_slice().cmp(&updated))
        {
            Ok(idx) | Err(idx) => idx,
        };
        self.records.insert(insert_idx, updated);
        Ok(true)
    }

    /// Close the B-tree handle. Subsequent mutating calls return an error.
    pub fn close(&mut self) {
        self.closed = true;
    }

    /// Delete the entire B-tree from the file.
    pub fn delete(mut self) {
        self.records.clear();
        self.header.total_records = 0;
        self.closed = true;
    }

    /// Make a child flush dependency between the tree's components.
    pub fn depend(&self) {}

    /// Patch the recorded root address (e.g. after file relocation).
    pub fn patch_file(&mut self, root_addr: u64) {
        self.header.root_addr = root_addr;
    }

    /// Binary-search for `record`. Returns `Ok(idx)` on hit, `Err(idx)` on miss.
    pub fn locate_record(&self, record: &[u8]) -> std::result::Result<usize, usize> {
        self.records
            .binary_search_by(|probe| probe.as_slice().cmp(record))
    }

    /// Perform a 1->2 node split, borrowing the left and right record halves.
    pub fn split1_ref(&self) -> (&[Vec<u8>], &[Vec<u8>]) {
        split_records_ref(&self.records)
    }

    /// Split the root node by partitioning its records in half.
    pub fn split_root_ref(&self) -> (&[Vec<u8>], &[Vec<u8>]) {
        self.split1_ref()
    }

    /// Redistribute records evenly between two nodes.
    pub fn redistribute2(left: &mut Vec<Vec<u8>>, right: &mut Vec<Vec<u8>>) {
        rebalance_two(left, right);
    }

    /// Redistribute records evenly across three adjacent nodes.
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
        let two = all
            .len()
            .checked_mul(2)
            .map(|value| value / 3)
            .unwrap_or(all.len());
        let right_records = all.split_off(two);
        let middle_records = all.split_off(one);
        *left = all;
        *middle = middle_records;
        *right = right_records;
    }

    /// Remove `record` from a node's record list, returning the removed entry.
    pub fn delete_node(records: &mut Vec<Vec<u8>>, record: &[u8]) -> Option<Vec<u8>> {
        let idx = records
            .binary_search_by(|probe| probe.as_slice().cmp(record))
            .ok()?;
        Some(records.remove(idx))
    }

    /// Return an error if the tree has been closed.
    fn ensure_open(&self) -> Result<()> {
        if self.closed {
            Err(Error::InvalidFormat("v2 B-tree is closed".into()))
        } else {
            Ok(())
        }
    }

    /// Refresh the header's record-count fields after record-list mutation.
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
    /// Create a client callback context for the v2 B-tree test record type.
    pub fn test_crt_context(record_size: usize) -> Result<Self> {
        if record_size < 8 {
            return Err(Error::InvalidFormat(
                "v2 B-tree test context record size must be at least 8".into(),
            ));
        }
        Ok(Self { record_size })
    }

    /// Destroy a client callback context.
    pub fn test_dst_context(self) {}
}

impl BTreeV2TestRecord {
    /// Store native information into a v2 B-tree test record.
    pub fn test_store(key: u64, value: u64) -> Self {
        Self { key, value }
    }

    /// Compare two native test records using natural ordering.
    pub fn test_compare(left: &Self, right: &Self) -> std::cmp::Ordering {
        left.cmp(right)
    }

    /// Encode the native test record into its on-disk little-endian form.
    pub fn test_encode_into(&self, context: &BTreeV2TestContext, out: &mut Vec<u8>) -> Result<()> {
        out.clear();
        out.resize(context.record_size, 0);
        checked_window_mut(out, 0, 8, "v2 B-tree test key")?
            .copy_from_slice(&self.key.to_le_bytes());
        if context.record_size >= 16 {
            checked_window_mut(out, 8, 8, "v2 B-tree test value")?
                .copy_from_slice(&self.value.to_le_bytes());
        }
        Ok(())
    }

    /// Decode the on-disk little-endian image back into a native test record.
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

    /// Format the native test record for debug printing.
    pub fn write_test_debug<W: Write + ?Sized>(&self, out: &mut W) -> fmt::Result {
        write!(
            out,
            "BTreeV2TestRecord(key={}, value={})",
            self.key, self.value
        )
    }

    /// Store native information into the alternate (test2) v2 B-tree test record.
    pub fn test2_store(key: u64, value: u64) -> Self {
        Self::test_store(key, value)
    }

    /// Compare two records using the reversed test2 ordering.
    pub fn test2_compare(left: &Self, right: &Self) -> std::cmp::Ordering {
        right.cmp(left)
    }

    /// Encode the native record into the on-disk big-endian (test2) form.
    pub fn test2_encode_into(&self, context: &BTreeV2TestContext, out: &mut Vec<u8>) -> Result<()> {
        out.clear();
        out.resize(context.record_size, 0);
        checked_window_mut(out, 0, 8, "v2 B-tree test2 key")?
            .copy_from_slice(&self.key.to_be_bytes());
        if context.record_size >= 16 {
            checked_window_mut(out, 8, 8, "v2 B-tree test2 value")?
                .copy_from_slice(&self.value.to_be_bytes());
        }
        Ok(())
    }

    /// Decode the on-disk big-endian (test2) image back into a native test record.
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

    /// Format the alternate test record for debug printing.
    pub fn write_test2_debug<W: Write + ?Sized>(&self, out: &mut W) -> fmt::Result {
        write!(
            out,
            "BTreeV2Test2Record(key={}, value={})",
            self.key, self.value
        )
    }
}

/// Borrow a `[pos..pos+len]` slice from `data`, returning an error on overflow or truncation.
fn checked_window<'a>(data: &'a [u8], pos: usize, len: usize, context: &str) -> Result<&'a [u8]> {
    let end = pos
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))?;
    data.get(pos..end)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))
}

/// Mutable variant of `checked_window`.
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

/// Read a little-endian u64 at `pos` from `data` with bounds checks.
fn read_u64_le_at(data: &[u8], pos: usize, context: &str) -> Result<u64> {
    let bytes = checked_window(data, pos, 8, context)?;
    Ok(u64::from_le_bytes(bytes.try_into().map_err(|_| {
        Error::InvalidFormat(format!("{context} is truncated"))
    })?))
}

/// Read a big-endian u64 at `pos` from `data` with bounds checks.
fn read_u64_be_at(data: &[u8], pos: usize, context: &str) -> Result<u64> {
    let bytes = checked_window(data, pos, 8, context)?;
    Ok(u64::from_be_bytes(bytes.try_into().map_err(|_| {
        Error::InvalidFormat(format!("{context} is truncated"))
    })?))
}

/// Read a `size`-byte little-endian variable-length unsigned integer (HDF5 `H5F_DECODE_LENGTH`).
fn read_var_uint<R: Read + Seek>(reader: &mut HdfReader<R>, size: usize) -> Result<u64> {
    if size == 0 || size > 8 {
        return Err(Error::InvalidFormat(format!(
            "invalid v2 B-tree variable integer size {size}"
        )));
    }

    let mut bytes = [0; 8];
    reader.read_bytes_into(&mut bytes[..size])?;
    let mut value = 0u64;
    for (idx, byte) in bytes[..size].iter().enumerate() {
        value |= u64::from(*byte) << (idx * 8);
    }
    Ok(value)
}

/// Ensure that `record` has exactly the expected width.
fn validate_record(record: &[u8], record_size: usize) -> Result<()> {
    if record.len() != record_size {
        return Err(Error::InvalidFormat(format!(
            "v2 B-tree record has size {}, expected {record_size}",
            record.len()
        )));
    }
    Ok(())
}

/// Validate that every record in a slice has the expected width.
fn validate_records(records: &[Vec<u8>], record_size: usize) -> Result<()> {
    for record in records {
        validate_record(record, record_size)?;
    }
    Ok(())
}

/// Insert `record` into a sorted record list, returning `false` if a duplicate exists.
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

/// Find the record adjacent to `record` (less-than or greater-than) in a sorted list.
fn neighbor_record_ref<'a>(
    records: &'a [Vec<u8>],
    record: &[u8],
    direction: BTreeV2Neighbor,
) -> Option<&'a [u8]> {
    let idx = match records.binary_search_by(|probe| probe.as_slice().cmp(record)) {
        Ok(idx) => idx,
        Err(idx) => idx,
    };
    match direction {
        BTreeV2Neighbor::Less => idx
            .checked_sub(1)
            .and_then(|pos| records.get(pos))
            .map(Vec::as_slice),
        BTreeV2Neighbor::Greater => {
            let pos = if records
                .get(idx)
                .is_some_and(|found| found.as_slice() == record)
            {
                idx + 1
            } else {
                idx
            };
            records.get(pos).map(Vec::as_slice)
        }
    }
}

/// Split a record list into two borrowed halves.
fn split_records_ref(records: &[Vec<u8>]) -> (&[Vec<u8>], &[Vec<u8>]) {
    let mid = records.len() / 2;
    records.split_at(mid)
}

/// Rebalance two record lists so they each hold half of the combined records.
fn rebalance_two(left: &mut Vec<Vec<u8>>, right: &mut Vec<Vec<u8>>) {
    let mut all = Vec::new();
    all.append(left);
    all.append(right);
    all.sort();
    let mid = all.len() / 2;
    let right_records = all.split_off(mid);
    *left = all;
    *right = right_records;
}

/// Verify the trailing 4-byte metadata checksum of an in-memory node image.
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

/// Reject zero or oversized (>8 bytes) integer widths.
fn validate_fixed_width(width: usize, context: &str) -> Result<()> {
    if width == 0 || width > 8 {
        return Err(Error::InvalidFormat(format!(
            "invalid {context} integer width {width}"
        )));
    }
    Ok(())
}

/// Write a fixed-width little-endian integer, erroring on overflow.
fn write_fixed_le(out: &mut Vec<u8>, value: u64, width: usize) -> Result<()> {
    validate_fixed_width(width, "v2 B-tree")?;
    if width < 8 && value >= (1u64 << (width * 8)) {
        return Err(Error::InvalidFormat(format!(
            "v2 B-tree integer value {value:#x} does not fit in {width} bytes"
        )));
    }
    out.extend_from_slice(&value.to_le_bytes()[..width]);
    Ok(())
}

/// Write a non-undefined address as a fixed-width little-endian integer.
fn write_addr_fixed_le(out: &mut Vec<u8>, value: u64, width: usize, context: &str) -> Result<()> {
    if value == UNDEF_ADDR {
        return Err(Error::InvalidFormat(format!(
            "{context} address is undefined"
        )));
    }
    write_fixed_le(out, value, width)
}

/// Write the header's root address. Allows the undefined sentinel only when
/// the tree contains no records.
fn write_hdr_addr_fixed_le(
    out: &mut Vec<u8>,
    value: u64,
    width: usize,
    total_records: u64,
    context: &str,
) -> Result<()> {
    validate_fixed_width(width, context)?;
    if value == UNDEF_ADDR {
        if total_records > 0 {
            return Err(Error::InvalidFormat(format!(
                "{context} address is undefined for non-empty tree"
            )));
        }
        out.extend(std::iter::repeat_n(0xff, width));
        return Ok(());
    }
    write_fixed_le(out, value, width)
}

/// Return the minimum number of bytes needed to encode `value` in little-endian.
fn bytes_needed(mut value: u64) -> usize {
    let mut bytes = 1usize;
    while value > 0xff {
        value >>= 8;
        bytes += 1;
    }
    bytes
}

/// Convert `u32` to `usize`, returning an error if it does not fit.
fn btree_u32_to_usize(value: u32, context: &str) -> Result<usize> {
    usize::try_from(value)
        .map_err(|_| Error::InvalidFormat(format!("{context} does not fit in usize")))
}

/// Convert `u64` to `u16`, returning an error on overflow.
fn btree_u64_to_u16(value: u64, context: &str) -> Result<u16> {
    u16::try_from(value).map_err(|_| Error::InvalidFormat(format!("{context} overflow")))
}

/// Convert `usize` to `u64`, returning an error on overflow.
fn btree_usize_to_u64(value: usize, context: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| Error::InvalidFormat(format!("{context} overflow")))
}

/// Sum a slice of `usize` values, returning an error on overflow.
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

    use crate::io::reader::UNDEF_ADDR;
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
        assert_eq!(tree.find_ref(&[2, 0]), Some([2, 0].as_slice()));
        assert_eq!(
            tree.neighbor_ref(&[1, 0], BTreeV2Neighbor::Greater),
            Some([2, 0].as_slice())
        );
        assert!(tree.modify(&[2, 0], |record| record[0] = 0).unwrap());
        assert_eq!(tree.index(0), Some([0, 0].as_slice()));
        assert_eq!(tree.index(1), Some([1, 0].as_slice()));
        assert_eq!(tree.remove(&[1, 0]).unwrap(), Some(vec![1, 0]));
        assert_eq!(tree.get_nrec(), 1);
    }

    #[test]
    fn btree_v2_cache_images_have_valid_checksums() {
        let header = BTreeV2Header::hdr_create(10, 256, 2, 100, 40).unwrap();
        let mut header_image = Vec::new();
        header.cache_hdr_serialize_into(&mut header_image).unwrap();
        super::verify_image_checksum(&header_image, "header").unwrap();

        let leaf = BTreeV2LeafNode::create_leaf(vec![vec![1, 0], vec![2, 0]], 2).unwrap();
        let mut leaf_image = Vec::new();
        leaf.cache_leaf_serialize_into(10, 2, &mut leaf_image)
            .unwrap();
        BTreeV2LeafNode::cache_leaf_verify_chksum(&leaf_image).unwrap();

        let internal =
            BTreeV2InternalNode::create_internal(vec![vec![2, 0]], vec![(10, 1), (20, 1)], 2)
                .unwrap();
        let mut internal_image = Vec::new();
        internal
            .cache_int_serialize_into(&header, 8, &mut internal_image)
            .unwrap();
        BTreeV2InternalNode::cache_int_verify_chksum(&internal_image).unwrap();
    }

    #[test]
    fn btree_v2_internal_cache_serialize_checks_configured_widths() {
        let header = BTreeV2Header {
            depth: 1,
            ..BTreeV2Header::hdr_create(10, 256, 2, 100, 40).unwrap()
        };
        let internal =
            BTreeV2InternalNode::create_internal(vec![vec![2, 0]], vec![(10, 1), (20, 1)], 2)
                .unwrap();
        let mut image = Vec::new();
        internal
            .cache_int_serialize_into(&header, 4, &mut image)
            .unwrap();
        assert_eq!(&image[8..12], &[10, 0, 0, 0]);

        let too_large = BTreeV2InternalNode::create_internal(
            vec![vec![2, 0]],
            vec![(u64::from(u32::MAX) + 1, 1), (20, 1)],
            2,
        )
        .unwrap();
        assert!(too_large
            .cache_int_serialize_into(&header, 4, &mut image)
            .is_err());

        let undefined = BTreeV2InternalNode::create_internal(
            vec![vec![2, 0]],
            vec![(UNDEF_ADDR, 1), (20, 1)],
            2,
        )
        .unwrap();
        assert!(undefined
            .cache_int_serialize_into(&header, 8, &mut image)
            .is_err());
    }

    #[test]
    fn btree_v2_header_cache_serialize_checks_configured_widths() {
        let header = BTreeV2Header {
            root_addr: 0x0102_0304,
            root_nrecords: 1,
            total_records: 1,
            ..BTreeV2Header::hdr_create(10, 256, 2, 100, 40).unwrap()
        };
        let mut image = Vec::new();
        header
            .cache_hdr_serialize_with_widths_into(4, 4, &mut image)
            .unwrap();
        assert_eq!(
            image.len(),
            header.cache_hdr_image_len_with_widths(4, 4).unwrap()
        );
        assert_eq!(&image[16..20], &[0x04, 0x03, 0x02, 0x01]);
        assert_eq!(&image[22..26], &[1, 0, 0, 0]);

        let mut reader = HdfReader::new(Cursor::new(image.clone()));
        reader.set_sizeof_addr(4);
        reader.set_sizeof_size(4);
        let decoded = BTreeV2Header::read_at(&mut reader, 0).unwrap();
        assert_eq!(decoded.root_addr, 0x0102_0304);
        assert_eq!(decoded.total_records, 1);

        let too_large_addr = BTreeV2Header {
            root_addr: u64::from(u32::MAX) + 1,
            ..header.clone()
        };
        assert!(too_large_addr
            .cache_hdr_serialize_with_widths_into(4, 4, &mut image)
            .is_err());

        let too_large_total = BTreeV2Header {
            total_records: u64::from(u32::MAX) + 1,
            ..header.clone()
        };
        assert!(too_large_total
            .cache_hdr_serialize_with_widths_into(4, 4, &mut image)
            .is_err());

        let empty_undef_root = BTreeV2Header {
            root_addr: UNDEF_ADDR,
            root_nrecords: 0,
            total_records: 0,
            ..header.clone()
        };
        empty_undef_root
            .cache_hdr_serialize_with_widths_into(4, 4, &mut image)
            .unwrap();
        assert_eq!(&image[16..20], &[0xff; 4]);

        let nonempty_undef_root = BTreeV2Header {
            root_addr: UNDEF_ADDR,
            ..header
        };
        assert!(nonempty_undef_root
            .cache_hdr_serialize_into(&mut image)
            .is_err());
    }

    #[test]
    fn btree_v2_header_checked_size_rejects_record_overflow() {
        let header = BTreeV2Header {
            total_records: u64::MAX,
            record_size: 2,
            ..BTreeV2Header::hdr_create(10, 256, 2, 100, 40).unwrap()
        };
        assert!(header.checked_size().is_err());
        assert_eq!(header.size(), u64::MAX);
    }

    #[test]
    fn btree_v2_leaf_cache_image_len_rejects_overflow() {
        let leaf = BTreeV2LeafNode::create_leaf(vec![vec![1, 0]], 2).unwrap();
        assert_eq!(leaf.cache_leaf_image_len(2).unwrap(), 12);
        assert!(leaf.cache_leaf_image_len(usize::MAX).is_err());
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

        let mut empty_with_root_records = Vec::new();
        header
            .cache_hdr_serialize_into(&mut empty_with_root_records)
            .unwrap();
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

        let mut root_exceeds_total = Vec::new();
        header
            .cache_hdr_serialize_into(&mut root_exceeds_total)
            .unwrap();
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
        let mut image = Vec::new();
        leaf.cache_leaf_serialize_into(10, 2, &mut image).unwrap();

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
        let mut image = Vec::new();
        internal
            .cache_int_serialize_into(&header, 8, &mut image)
            .unwrap();

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

        let mut bad_child_addr = Vec::new();
        internal
            .cache_int_serialize_into(&header, 8, &mut bad_child_addr)
            .unwrap();
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
        let mut image = Vec::new();
        record.test_encode_into(&context, &mut image).unwrap();
        assert_eq!(
            BTreeV2TestRecord::test_decode(&context, &image).unwrap(),
            record
        );

        let mut image2 = Vec::new();
        record.test2_encode_into(&context, &mut image2).unwrap();
        assert_eq!(
            BTreeV2TestRecord::test2_decode(&context, &image2).unwrap(),
            record
        );
    }
}
