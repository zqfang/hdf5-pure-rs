use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::io::reader::{is_undef_addr, HdfReader, UNDEF_ADDR};

/// v1 B-tree node magic: "TREE"
const BTREE_MAGIC: [u8; 4] = [b'T', b'R', b'E', b'E'];
const MAX_GROUP_BTREE_RECURSION: usize = 64;

/// B-tree node types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BTreeType {
    /// Type 0: Group nodes (symbol table nodes).
    Group,
    /// Type 1: Raw data chunks.
    RawData,
}

/// A v1 B-tree node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BTreeV1Node {
    pub node_type: BTreeType,
    pub level: u8,
    pub entries_used: u16,
    pub left_sibling: u64,
    pub right_sibling: u64,
    /// For group B-trees: child node addresses.
    pub children: Vec<u64>,
    /// For group B-trees: keys (symbol name offsets in local heap).
    pub keys: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BTreeV1Info {
    pub node_count: usize,
    pub record_count: usize,
    pub depth: u8,
}

impl BTreeV1Node {
    /// Deserialize a v1 B-tree node from disk at the given address.
    pub fn read_at<R: Read + Seek>(reader: &mut HdfReader<R>, addr: u64) -> Result<Self> {
        reader.seek(addr)?;

        let magic = reader.read_bytes(4)?;
        if magic != BTREE_MAGIC {
            return Err(Error::InvalidFormat("invalid v1 B-tree magic".into()));
        }

        let node_type_val = reader.read_u8()?;
        let node_type = match node_type_val {
            0 => BTreeType::Group,
            1 => BTreeType::RawData,
            _ => {
                return Err(Error::Unsupported(format!(
                    "B-tree node type {node_type_val}"
                )))
            }
        };

        let level = reader.read_u8()?;
        let entries_used = reader.read_u16()?;
        let left_sibling = reader.read_addr()?;
        let right_sibling = reader.read_addr()?;

        let mut keys = Vec::new();
        let mut children = Vec::new();

        match node_type {
            BTreeType::Group => {
                // Group B-tree: keys are (heap_offset, obj_header_addr) pairs for internal nodes,
                // or just heap_offset for comparison.
                // Actually for group B-trees, keys are just the symbol name heap offset (sizeof_size).
                // Structure: key[0], child[0], key[1], child[1], ..., key[n]
                // So there are entries_used children and entries_used+1 keys.

                let entries_used_usize = usize::from(entries_used);
                let key_count = entries_used_usize
                    .checked_add(1)
                    .ok_or_else(|| Error::InvalidFormat("v1 B-tree key count overflow".into()))?;
                keys.reserve(key_count);
                children.reserve(entries_used_usize);

                for _i in 0..entries_used_usize {
                    // Key
                    let key = reader.read_length()?;
                    keys.push(key);

                    // Child pointer
                    let child = reader.read_addr()?;
                    if is_undef_addr(child) {
                        return Err(Error::InvalidFormat(
                            "v1 group B-tree child address is undefined".into(),
                        ));
                    }
                    children.push(child);
                }
                // Final key
                let final_key = reader.read_length()?;
                keys.push(final_key);
            }
            BTreeType::RawData => {
                return Err(Error::Unsupported(
                    "raw data v1 B-tree nodes require dataset chunk key context".into(),
                ));
            }
        }

        Ok(Self {
            node_type,
            level,
            entries_used,
            left_sibling,
            right_sibling,
            children,
            keys,
        })
    }

    /// Collect all leaf-level symbol table node addresses from a group B-tree.
    /// Recursively traverses internal nodes to reach leaves.
    pub fn collect_symbol_table_addrs<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        btree_addr: u64,
    ) -> Result<Vec<u64>> {
        let mut visited = Vec::new();
        Self::collect_symbol_table_addrs_inner(reader, btree_addr, 0, &mut visited)
    }

    /// Recursive helper that walks internal B-tree nodes down to leaves while
    /// tracking visited addresses to detect cycles and runaway depth.
    fn collect_symbol_table_addrs_inner<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        btree_addr: u64,
        depth: usize,
        visited: &mut Vec<u64>,
    ) -> Result<Vec<u64>> {
        if depth > MAX_GROUP_BTREE_RECURSION {
            return Err(Error::InvalidFormat(
                "v1 group B-tree recursion depth exceeded".into(),
            ));
        }
        if is_undef_addr(btree_addr) {
            return Err(Error::InvalidFormat(
                "v1 group B-tree address is undefined".into(),
            ));
        }
        if visited.contains(&btree_addr) {
            return Err(Error::InvalidFormat(
                "v1 group B-tree traversal cycle detected".into(),
            ));
        }
        visited.push(btree_addr);

        let node = Self::read_at(reader, btree_addr)?;

        let result = if node.node_type != BTreeType::Group {
            Err(Error::InvalidFormat("expected group B-tree".into()))
        } else if node.level == 0 {
            // Leaf node: children are symbol table node addresses
            Ok(node.children)
        } else {
            // Internal node: recurse into children
            let mut result = Vec::new();
            for &child_addr in &node.children {
                let mut child_addrs = Self::collect_symbol_table_addrs_inner(
                    reader,
                    child_addr,
                    depth.checked_add(1).ok_or_else(|| {
                        Error::InvalidFormat("v1 group B-tree recursion depth overflow".into())
                    })?,
                    visited,
                )?;
                result.append(&mut child_addrs);
            }
            Ok(result)
        };

        visited.pop();
        result
    }

    /// Compute the on-disk size of a v1 B-tree node prefix (used by the metadata cache).
    pub fn cache_get_initial_load_size(sizeof_addr: usize) -> Result<usize> {
        checked_usize_sum(&[4, 1, 1, 2, sizeof_addr, sizeof_addr], "v1 B-tree prefix")
    }

    /// Compute the on-disk size of this B-tree node.
    pub fn cache_image_len(&self, sizeof_addr: usize, sizeof_size: usize) -> Result<usize> {
        let prefix = Self::cache_get_initial_load_size(sizeof_addr)?;
        let per_entry = sizeof_size
            .checked_add(sizeof_addr)
            .ok_or_else(|| Error::InvalidFormat("v1 B-tree entry size overflow".into()))?;
        let entries = per_entry
            .checked_mul(self.children.len())
            .ok_or_else(|| Error::InvalidFormat("v1 B-tree entry bytes overflow".into()))?;
        prefix
            .checked_add(entries)
            .and_then(|len| len.checked_add(sizeof_size))
            .ok_or_else(|| Error::InvalidFormat("v1 B-tree image length overflow".into()))
    }

    /// Serialize the B-tree node into its on-disk image.
    pub fn cache_serialize(&self, sizeof_addr: usize, sizeof_size: usize) -> Result<Vec<u8>> {
        self.verify_structure()?;
        let mut out = Vec::with_capacity(self.cache_image_len(sizeof_addr, sizeof_size)?);
        out.extend_from_slice(&BTREE_MAGIC);
        out.push(match self.node_type {
            BTreeType::Group => 0,
            BTreeType::RawData => 1,
        });
        out.push(self.level);
        out.extend_from_slice(&self.entries_used.to_le_bytes());
        write_addr_le(
            &mut out,
            self.left_sibling,
            sizeof_addr,
            "v1 B-tree left sibling",
        )?;
        write_addr_le(
            &mut out,
            self.right_sibling,
            sizeof_addr,
            "v1 B-tree right sibling",
        )?;
        for index in 0..self.children.len() {
            write_var_le(&mut out, self.keys[index], sizeof_size)?;
            if self.children[index] == UNDEF_ADDR {
                return Err(Error::InvalidFormat(
                    "v1 B-tree child address is undefined".into(),
                ));
            }
            write_addr_le(
                &mut out,
                self.children[index],
                sizeof_addr,
                "v1 B-tree child",
            )?;
        }
        write_var_le(&mut out, *self.keys.last().unwrap_or(&0), sizeof_size)?;
        Ok(out)
    }

    /// Destroy/release an in-core representation of the B-tree node.
    pub fn cache_free_icr(self) {}

    /// Format the node for debug printing (B-tree debug dump).
    pub fn debug(&self) -> String {
        format!(
            "BTreeV1Node(type={:?}, level={}, entries={}, children={})",
            self.node_type,
            self.level,
            self.entries_used,
            self.children.len()
        )
    }

    /// Verify that the node is internally consistent (correct child count,
    /// matching key count, and sorted keys).
    pub fn verify_structure(&self) -> Result<()> {
        if self.children.len() != usize::from(self.entries_used) {
            return Err(Error::InvalidFormat(
                "v1 B-tree child count does not match entries_used".into(),
            ));
        }
        let expected_keys = self
            .children
            .len()
            .checked_add(1)
            .ok_or_else(|| Error::InvalidFormat("v1 B-tree key count overflow".into()))?;
        if self.keys.len() != expected_keys {
            return Err(Error::InvalidFormat(
                "v1 B-tree key count must be entries_used + 1".into(),
            ));
        }
        if self.keys.windows(2).any(|pair| pair[0] > pair[1]) {
            return Err(Error::InvalidFormat("v1 B-tree keys are not sorted".into()));
        }
        Ok(())
    }

    /// Create a new empty B-tree leaf node.
    pub fn create(node_type: BTreeType, level: u8) -> Self {
        Self {
            node_type,
            level,
            entries_used: 0,
            left_sibling: u64::MAX,
            right_sibling: u64::MAX,
            children: Vec::new(),
            keys: vec![0],
        }
    }

    /// Locate the child whose key range contains `key`, returning its address.
    /// Returns `None` if the key falls outside this node's range.
    pub fn find(&self, key: u64) -> Result<Option<u64>> {
        self.verify_structure()?;
        Ok(self.find_helper(key))
    }

    /// Unchecked find helper that returns the child whose key range contains `key`.
    pub fn find_helper(&self, key: u64) -> Option<u64> {
        self.children
            .iter()
            .enumerate()
            .find(|(index, _)| key >= self.keys[*index] && key < self.keys[*index + 1])
            .map(|(_, child)| *child)
    }

    /// Split this full node into two, returning the new right-hand sibling.
    /// This node keeps the left children; the returned node holds the right children.
    pub fn split(&mut self) -> Result<Self> {
        self.verify_structure()?;
        if self.children.len() < 2 {
            return Err(Error::InvalidFormat(
                "cannot split v1 B-tree node with fewer than two children".into(),
            ));
        }
        let split_at = self.children.len() / 2;
        let boundary_key = self.keys[split_at];
        let right_children = self.children.split_off(split_at);
        let right_keys = self.keys.split_off(split_at);
        self.keys.push(boundary_key);
        self.entries_used = u16::try_from(self.children.len())
            .map_err(|_| Error::InvalidFormat("v1 B-tree entry count overflow".into()))?;
        Ok(Self {
            node_type: self.node_type,
            level: self.level,
            entries_used: u16::try_from(right_children.len())
                .map_err(|_| Error::InvalidFormat("v1 B-tree entry count overflow".into()))?,
            left_sibling: u64::MAX,
            right_sibling: self.right_sibling,
            children: right_children,
            keys: right_keys,
        })
    }

    /// Add a new item to the B-tree node.
    pub fn insert(&mut self, key: u64, child: u64, upper_key: u64) -> Result<()> {
        self.insert_helper(key, child, upper_key)
    }

    /// Insert a child at the given position, updating the surrounding keys.
    pub fn insert_child(
        &mut self,
        index: usize,
        key: u64,
        child: u64,
        upper_key: u64,
    ) -> Result<()> {
        if index > self.children.len() {
            return Err(Error::InvalidFormat(
                "v1 B-tree insert index out of bounds".into(),
            ));
        }
        self.children.insert(index, child);
        self.keys.insert(index, key);
        self.keys[index + 1] = upper_key;
        self.entries_used = u16::try_from(self.children.len())
            .map_err(|_| Error::InvalidFormat("v1 B-tree entry count overflow".into()))?;
        self.verify_structure()
    }

    /// Recursive insert helper: locate the correct slot for `key` and add the child.
    pub fn insert_helper(&mut self, key: u64, child: u64, upper_key: u64) -> Result<()> {
        let index = self.keys.partition_point(|&existing| existing <= key);
        self.insert_child(
            index.saturating_sub(1).min(self.children.len()),
            key,
            child,
            upper_key,
        )
    }

    /// Call `f(key, child)` once for each entry in this leaf node.
    pub fn iterate_helper<F: FnMut(u64, u64)>(&self, mut f: F) -> Result<()> {
        self.verify_structure()?;
        for (index, &child) in self.children.iter().enumerate() {
            f(self.keys[index], child);
        }
        Ok(())
    }

    /// Recursive removal helper: removes the child whose key range contains `key`.
    pub fn remove_helper(&mut self, key: u64) -> Result<Option<u64>> {
        self.verify_structure()?;
        let Some(index) = self
            .children
            .iter()
            .enumerate()
            .position(|(index, _)| key >= self.keys[index] && key < self.keys[index + 1])
        else {
            return Ok(None);
        };
        let child = self.children.remove(index);
        self.keys.remove(index);
        self.entries_used = u16::try_from(self.children.len())
            .map_err(|_| Error::InvalidFormat("v1 B-tree entry count overflow".into()))?;
        Ok(Some(child))
    }

    /// Remove an item from the B-tree node. The tree is not rebalanced on removal.
    pub fn remove(&mut self, key: u64) -> Result<Option<u64>> {
        self.remove_helper(key)
    }

    /// Delete the entire B-tree node, clearing all children and keys.
    pub fn delete(&mut self) {
        self.children.clear();
        self.keys.clear();
        self.keys.push(0);
        self.entries_used = 0;
    }

    /// Allocate and construct a shared v1 B-tree node for a client.
    pub fn shared_new(node_type: BTreeType, level: u8) -> Self {
        Self::create(node_type, level)
    }

    /// Free the shared B-tree info.
    pub fn shared_free(self) {}

    /// Return a deep copy of the node.
    pub fn copy(&self) -> Self {
        self.clone()
    }

    /// Walk this node and gather node/record/depth information.
    /// On overflow returns a saturated value rather than an error (matches the
    /// historical `H5B__get_info_helper` lossy behavior).
    pub fn get_info_helper(&self) -> BTreeV1Info {
        self.get_info_helper_checked().unwrap_or(BTreeV1Info {
            node_count: 1,
            record_count: self.children.len(),
            depth: u8::MAX,
        })
    }

    /// Checked variant of `get_info_helper` that returns an error on depth overflow.
    pub fn get_info_helper_checked(&self) -> Result<BTreeV1Info> {
        Ok(BTreeV1Info {
            node_count: 1,
            record_count: self.children.len(),
            depth: self
                .level
                .checked_add(1)
                .ok_or_else(|| Error::InvalidFormat("v1 B-tree depth overflow".into()))?,
        })
    }

    /// Return the amount of storage used for this B-tree node.
    pub fn get_info(&self) -> BTreeV1Info {
        self.get_info_helper()
    }

    /// Returns `true` if the node passes structural validation.
    pub fn valid(&self) -> bool {
        self.verify_structure().is_ok()
    }

    /// Destroy/release the B-tree node, consuming it.
    pub fn node_dest(self) {}
}

/// Sum a slice of `usize` values, returning an error on overflow.
fn checked_usize_sum(values: &[usize], context: &str) -> Result<usize> {
    values.iter().try_fold(0usize, |acc, &value| {
        acc.checked_add(value)
            .ok_or_else(|| Error::InvalidFormat(format!("{context} size overflow")))
    })
}

/// Write a variable-width little-endian integer, validating that `value` fits.
fn write_var_le(out: &mut Vec<u8>, value: u64, width: usize) -> Result<()> {
    if width == 0 || width > 8 {
        return Err(Error::Unsupported(format!(
            "v1 B-tree integer width {width} exceeds u64"
        )));
    }
    if width < 8 && value >= (1u64 << (width * 8)) {
        return Err(Error::InvalidFormat(format!(
            "v1 B-tree integer value {value:#x} does not fit in {width} bytes"
        )));
    }
    out.extend_from_slice(&value.to_le_bytes()[..width]);
    Ok(())
}

/// Write a variable-width little-endian address. The undefined-address sentinel
/// is written as all-`0xff` bytes regardless of width.
fn write_addr_le(out: &mut Vec<u8>, value: u64, width: usize, context: &str) -> Result<()> {
    if width == 0 || width > 8 {
        return Err(Error::Unsupported(format!(
            "{context} width {width} exceeds u64"
        )));
    }
    if value == UNDEF_ADDR {
        out.extend(std::iter::repeat_n(0xff, width));
        return Ok(());
    }
    if width < 8 && value >= (1u64 << (width * 8)) {
        return Err(Error::InvalidFormat(format!(
            "{context} address {value:#x} does not fit in {width} bytes"
        )));
    }
    out.extend_from_slice(&value.to_le_bytes()[..width]);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::reader::HdfReader;
    use std::io::Cursor;

    #[test]
    fn btree_v1_insert_find_remove_and_split() {
        let mut node = BTreeV1Node::create(BTreeType::Group, 0);
        node.insert(0, 100, 10).unwrap();
        node.insert(10, 200, 20).unwrap();
        assert_eq!(node.find(12).unwrap(), Some(200));
        assert_eq!(node.remove(4).unwrap(), Some(100));
        node.insert(20, 300, 30).unwrap();
        let right = node.split().unwrap();
        assert!(node.valid());
        assert!(right.valid());
    }

    #[test]
    fn btree_v1_serializes_group_node_prefix() {
        let mut node = BTreeV1Node::create(BTreeType::Group, 0);
        node.insert(0, 0x1122, 8).unwrap();
        let image = node.cache_serialize(8, 8).unwrap();
        assert_eq!(&image[..4], b"TREE");
        assert_eq!(image[4], 0);
        assert_eq!(image.len(), node.cache_image_len(8, 8).unwrap());
    }

    #[test]
    fn btree_v1_cache_serialize_checks_configured_widths() {
        let mut node = BTreeV1Node::create(BTreeType::Group, 0);
        node.insert(0, 0x1122, 8).unwrap();
        let image = node.cache_serialize(4, 4).unwrap();
        assert_eq!(&image[8..12], &[0xff; 4]);
        assert_eq!(&image[12..16], &[0xff; 4]);

        let mut too_large_child = node.clone();
        too_large_child.children[0] = u64::from(u32::MAX) + 1;
        assert!(too_large_child.cache_serialize(4, 4).is_err());

        let mut too_large_key = node;
        too_large_key.keys[1] = u64::from(u32::MAX) + 1;
        assert!(too_large_key.cache_serialize(4, 4).is_err());
    }

    #[test]
    fn btree_v1_checked_info_rejects_depth_overflow() {
        let node = BTreeV1Node::create(BTreeType::Group, u8::MAX);
        assert!(node.get_info_helper_checked().is_err());
        assert_eq!(node.get_info_helper().depth, u8::MAX);
    }

    #[test]
    fn btree_v1_read_rejects_raw_nodes_without_chunk_context() {
        let mut image = b"TREE".to_vec();
        image.push(1);
        image.push(0);
        image.extend_from_slice(&0u16.to_le_bytes());
        image.extend_from_slice(&u64::MAX.to_le_bytes());
        image.extend_from_slice(&u64::MAX.to_le_bytes());

        let mut reader = HdfReader::new(Cursor::new(image));
        let err = BTreeV1Node::read_at(&mut reader, 0).unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)));
    }
}
