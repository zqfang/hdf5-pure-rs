use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom, Write};
use std::str;

use crate::engine::writer::{DatasetSpec, FillValueSpec};
use crate::error::{Error, Result};
use crate::format::btree_v2::BTreeV2Header;
use crate::format::checksum::checksum_metadata;
use crate::format::fractal_heap::FractalHeapHeader;
use crate::format::messages::link::{LinkMessage, LinkType};
use crate::format::messages::link_info::LinkInfoMessage;
use crate::format::object_header::{
    self, ObjectHeader, ObjectHeaderMessageRef, HDR_ATTR_CRT_ORDER_TRACKED,
    HDR_ATTR_STORE_PHASE_CHANGE, HDR_CHUNK0_SIZE_MASK, HDR_STORE_TIMES, HDR_V2_KNOWN_FLAGS,
};
use crate::format::superblock::Superblock;
use crate::io::reader::HdfReader;

use super::MutableFile;

const OBJECT_HEADER_CHUNK_DATA_LIMIT: usize = 128 * 1024;
const MAX_DATASPACE_RANK: usize = 32;

#[derive(Debug)]
struct CompactLinkMessageLocation {
    msg_type_offset: u64,
    msg_data_offset: u64,
    oh_start: u64,
    oh_check_len: usize,
    name_offset: usize,
    name_size: usize,
    link_type: LinkType,
    hard_link_addr: Option<u64>,
}

#[derive(Debug)]
struct ObjectRefcountLocation {
    value_offset: u64,
    oh_start: Option<u64>,
    oh_check_len: Option<usize>,
    refcount: u32,
}

#[derive(Debug)]
struct DenseLinkLocation {
    btree_header_addr: u64,
    heap: FractalHeapHeader,
    btree: BTreeV2Header,
    leaf_records: Vec<u8>,
    record_index: usize,
    object_offset: u64,
    raw_data: Vec<u8>,
}

#[derive(Debug)]
enum ParentChainLinkStorage {
    Compact,
    Dense,
}

#[derive(Debug)]
struct ParentChainLink {
    ancestor_addr: u64,
    link_name: String,
    child_addr: u64,
    storage: ParentChainLinkStorage,
}

impl MutableFile {
    /// Create a child group by rewriting compact v2/v3 metadata.
    ///
    /// Root children rewrite the root header and superblock. Nested compact
    /// children rewrite each compact hard-link ancestor up to the root so
    /// moved object-header addresses remain reachable.
    pub fn create_root_group_link(&mut self, name: &str) -> Result<()> {
        self.create_compact_group_link("/", name)
    }

    pub fn create_compact_group_link(&mut self, parent_path: &str, name: &str) -> Result<()> {
        validate_direct_child_link_name(name)?;
        if self.superblock.version < 2 {
            return Err(Error::Unsupported(
                "group creation in existing files requires a v2/v3 superblock".into(),
            ));
        }

        let child_addr = self.append_v2_object_header(&[], 0)?;
        let link = encode_hard_link_message(name, child_addr, self.superblock.sizeof_addr)?;
        self.append_compact_parent_link(parent_path, name, "group creation", link)
    }

    /// Create a soft link in the root group of an existing compact v2/v3 file.
    pub fn create_root_soft_link(&mut self, name: &str, target_path: &str) -> Result<()> {
        self.create_compact_soft_link("/", name, target_path)
    }

    pub fn create_compact_soft_link(
        &mut self,
        parent_path: &str,
        name: &str,
        target_path: &str,
    ) -> Result<()> {
        validate_direct_child_link_name(name)?;
        if self.superblock.version < 2 {
            return Err(Error::Unsupported(
                "soft-link creation in existing files requires a v2/v3 superblock".into(),
            ));
        }

        let link = encode_soft_link_message(name, target_path)?;
        self.append_compact_parent_link(parent_path, name, "soft-link creation", link)
    }

    /// Create an external link in the root group of an existing compact v2/v3 file.
    pub fn create_root_external_link(
        &mut self,
        name: &str,
        filename: &str,
        object_path: &str,
    ) -> Result<()> {
        self.create_compact_external_link("/", name, filename, object_path)
    }

    pub fn create_compact_external_link(
        &mut self,
        parent_path: &str,
        name: &str,
        filename: &str,
        object_path: &str,
    ) -> Result<()> {
        validate_direct_child_link_name(name)?;
        if self.superblock.version < 2 {
            return Err(Error::Unsupported(
                "external-link creation in existing files requires a v2/v3 superblock".into(),
            ));
        }

        let link = encode_external_link_message(name, filename, object_path)?;
        self.append_compact_parent_link(parent_path, name, "external-link creation", link)
    }

    /// Create a hard link in the root group of an existing compact v2/v3 file.
    pub fn create_root_hard_link(&mut self, name: &str, target_addr: u64) -> Result<()> {
        self.create_compact_hard_link("/", name, target_addr)
    }

    pub fn create_compact_hard_link(
        &mut self,
        parent_path: &str,
        name: &str,
        target_addr: u64,
    ) -> Result<()> {
        validate_direct_child_link_name(name)?;
        if self.superblock.version < 2 {
            return Err(Error::Unsupported(
                "hard-link creation in existing files requires a v2/v3 superblock".into(),
            ));
        }

        self.increment_object_refcount_if_present(target_addr)?;
        let link = encode_hard_link_message(name, target_addr, self.superblock.sizeof_addr)?;
        self.append_compact_parent_link(parent_path, name, "hard-link creation", link)
    }

    pub fn create_compact_hard_link_to_target_path(
        &mut self,
        parent_path: &str,
        name: &str,
        target_path: &str,
        target_addr: u64,
    ) -> Result<()> {
        validate_direct_child_link_name(name)?;
        if self.superblock.version < 2 {
            return Err(Error::Unsupported(
                "hard-link creation in existing files requires a v2/v3 superblock".into(),
            ));
        }

        let operation = "hard-link creation";
        let (target_parent_path, target_name) =
            split_normalized_absolute_child_path(target_path, operation)?;

        let mut path_addrs = BTreeMap::new();
        path_addrs.insert(Vec::<String>::new(), self.superblock.root_addr);
        self.record_compact_path_addrs(parent_path, operation, &mut path_addrs)?;
        self.record_compact_path_addrs(target_parent_path, operation, &mut path_addrs)?;

        let dest_components = owned_compact_absolute_path_components(parent_path, operation)?;
        let target_parent_components =
            owned_compact_absolute_path_components(target_parent_path, operation)?;
        let target_components = owned_compact_absolute_path_components(target_path, operation)?;
        let target_is_destination_ancestor = target_components.len() <= dest_components.len()
            && target_components == dest_components[..target_components.len()];
        let dest_addr = *path_addrs.get(&dest_components).ok_or_else(|| {
            Error::InvalidFormat(format!("destination group '{parent_path}' not found"))
        })?;
        let target_parent_addr = *path_addrs.get(&target_parent_components).ok_or_else(|| {
            Error::InvalidFormat(format!(
                "target parent group '{target_parent_path}' not found"
            ))
        })?;

        let (dest_flags, mut dest_messages) =
            self.messages_for_append(dest_addr, name, operation)?;
        if target_is_destination_ancestor
            && self.find_object_refcount_location(target_addr)?.is_none()
        {
            return Err(Error::Unsupported(
                "hard-link creation cannot yet materialize a refcount for an ancestor of the destination group"
                    .into(),
            ));
        }
        let new_target_addr = self.ensure_explicit_object_refcount(target_addr, 2)?;
        dest_messages.push(OwnedObjectHeaderMessage {
            msg_type: object_header::MSG_LINK,
            flags: 0,
            creation_index: None,
            data: encode_hard_link_message(name, new_target_addr, self.superblock.sizeof_addr)?,
        });

        if new_target_addr == target_addr {
            return self.append_compact_parent_link(
                parent_path,
                name,
                operation,
                encode_hard_link_message(name, new_target_addr, self.superblock.sizeof_addr)?,
            );
        }

        let mut modified = BTreeMap::new();
        if target_parent_components == dest_components {
            let (flags, messages) = self.compact_messages_replacing_and_appending_hard_link(
                target_parent_addr,
                target_name,
                new_target_addr,
                name,
                operation,
            )?;
            modified.insert(target_parent_components, (flags, messages));
        } else {
            let (target_flags, target_messages) = self.compact_messages_replacing_hard_link(
                target_parent_addr,
                target_name,
                new_target_addr,
                operation,
            )?;
            modified.insert(target_parent_components, (target_flags, target_messages));
            modified.insert(dest_components, (dest_flags, dest_messages));
        }
        self.append_rebuilt_compact_paths(modified, path_addrs, operation)
    }

    pub fn create_compact_same_group_hard_link(
        &mut self,
        parent_path: &str,
        name: &str,
        target_name: &str,
        target_addr: u64,
    ) -> Result<()> {
        validate_direct_child_link_name(name)?;
        validate_direct_child_link_name(target_name)?;
        if self.superblock.version < 2 {
            return Err(Error::Unsupported(
                "same-group hard-link creation in existing files requires a v2/v3 superblock"
                    .into(),
            ));
        }

        let target_addr = self.ensure_explicit_object_refcount(target_addr, 2)?;
        self.append_compact_same_group_hard_link(parent_path, name, target_name, target_addr)
    }

    /// Create a contiguous simple dataset in the root group of an existing compact v2/v3 file.
    pub fn create_root_contiguous_dataset(
        &mut self,
        spec: &DatasetSpec<'_>,
        fill: Option<FillValueSpec<'_>>,
    ) -> Result<u64> {
        self.create_compact_contiguous_dataset("/", spec, fill)
    }

    pub fn create_compact_contiguous_dataset(
        &mut self,
        parent_path: &str,
        spec: &DatasetSpec<'_>,
        fill: Option<FillValueSpec<'_>>,
    ) -> Result<u64> {
        validate_direct_child_link_name(spec.name)?;
        validate_dataset_data_len(spec)?;
        if spec.max_shape.is_some() {
            return Err(Error::Unsupported(
                "existing-file dataset creation does not support max dimensions yet".into(),
            ));
        }
        if self.superblock.version < 2 {
            return Err(Error::Unsupported(
                "dataset creation in existing files requires a v2/v3 superblock".into(),
            ));
        }

        let data_addr = if spec.data.is_empty() {
            crate::io::reader::UNDEF_ADDR
        } else {
            let addr = self.append_aligned_zeros(spec.data.len(), 8)?;
            self.write_handle.seek(SeekFrom::Start(addr))?;
            self.write_handle.write_all(spec.data)?;
            addr
        };
        let data_size = Self::usize_to_u64(spec.data.len(), "dataset data size")?;

        let mut dtype_bytes = Vec::new();
        spec.dtype.encode_into(&mut dtype_bytes)?;
        let mut ds_bytes = Vec::new();
        encode_dataspace_for_spec_into(&mut ds_bytes, spec)?;
        let mut fill_value_bytes = Vec::new();
        encode_fill_value_message_into(&mut fill_value_bytes, fill)?;
        let mut layout_bytes = Vec::new();
        encode_contiguous_layout_into(
            &mut layout_bytes,
            data_addr,
            data_size,
            self.superblock.sizeof_addr,
            self.superblock.sizeof_size,
        )?;

        let messages = [
            ObjectHeaderMessageRef {
                msg_type: object_header::MSG_DATASPACE,
                flags: 0,
                creation_index: None,
                data: &ds_bytes,
            },
            ObjectHeaderMessageRef {
                msg_type: object_header::MSG_DATATYPE,
                flags: 0,
                creation_index: None,
                data: &dtype_bytes,
            },
            ObjectHeaderMessageRef {
                msg_type: object_header::MSG_FILL_VALUE,
                flags: 0,
                creation_index: None,
                data: &fill_value_bytes,
            },
            ObjectHeaderMessageRef {
                msg_type: object_header::MSG_LAYOUT,
                flags: 0,
                creation_index: None,
                data: &layout_bytes,
            },
        ];
        let dataset_addr = self.append_v2_object_header(&messages, 0)?;
        let link = encode_hard_link_message(spec.name, dataset_addr, self.superblock.sizeof_addr)?;
        self.append_compact_parent_link(parent_path, spec.name, "dataset creation", link)?;
        Ok(dataset_addr)
    }

    fn append_compact_parent_link(
        &mut self,
        parent_path: &str,
        name: &str,
        operation: &str,
        link: Vec<u8>,
    ) -> Result<()> {
        let root_addr = self.superblock.root_addr;
        if parent_path == "/" {
            let (root_flags, root_messages) =
                self.messages_for_append(root_addr, name, operation)?;
            return self.append_rebuilt_root_link_message(root_messages, root_flags, link);
        }

        let parent_chain = self.compact_or_dense_hard_link_parent_chain(parent_path, operation)?;
        let old_parent_addr = parent_chain
            .last()
            .map(|link| link.child_addr)
            .ok_or_else(|| {
                Error::InvalidFormat(format!("parent group '{parent_path}' not found"))
            })?;
        let (parent_flags, mut parent_messages) =
            self.messages_for_append(old_parent_addr, name, operation)?;
        parent_messages.push(OwnedObjectHeaderMessage {
            msg_type: object_header::MSG_LINK,
            flags: 0,
            creation_index: None,
            data: link,
        });

        let mut new_child_addr =
            self.append_owned_v2_object_header(&parent_messages, parent_flags)?;
        let mut root_update = None;
        let mut dense_update = false;
        for link in parent_chain.iter().rev() {
            match link.storage {
                ParentChainLinkStorage::Compact => {
                    let (ancestor_flags, ancestor_messages) = self
                        .compact_messages_replacing_hard_link(
                            link.ancestor_addr,
                            &link.link_name,
                            new_child_addr,
                            operation,
                        )?;
                    if link.ancestor_addr == root_addr {
                        root_update = Some((ancestor_flags, ancestor_messages));
                        break;
                    }
                    new_child_addr =
                        self.append_owned_v2_object_header(&ancestor_messages, ancestor_flags)?;
                }
                ParentChainLinkStorage::Dense => {
                    self.rewrite_dense_hard_link_addr(
                        link.ancestor_addr,
                        &link.link_name,
                        new_child_addr,
                        operation,
                    )?;
                    dense_update = true;
                    break;
                }
            }
        }
        if dense_update {
            let eof_addr = self.write_handle.seek(SeekFrom::End(0))?;
            self.rewrite_v2_v3_superblock(self.superblock.root_addr, eof_addr)?;
            self.write_handle.flush()?;
            self.reopen_reader()?;
            return Ok(());
        }
        let (root_flags, root_messages) = root_update.ok_or_else(|| {
            Error::InvalidFormat(format!(
                "parent group '{parent_path}' is not linked from the root group"
            ))
        })?;
        self.append_rebuilt_root_link_message(root_messages, root_flags, Vec::new())
    }

    fn append_compact_same_group_hard_link(
        &mut self,
        parent_path: &str,
        name: &str,
        target_name: &str,
        target_addr: u64,
    ) -> Result<()> {
        let operation = "same-group hard-link creation";
        let root_addr = self.superblock.root_addr;
        if parent_path == "/" {
            let (root_flags, root_messages) = self
                .compact_messages_replacing_and_appending_hard_link(
                    root_addr,
                    target_name,
                    target_addr,
                    name,
                    operation,
                )?;
            return self.append_rebuilt_root_link_message(root_messages, root_flags, Vec::new());
        }

        let parent_chain = self.compact_or_dense_hard_link_parent_chain(parent_path, operation)?;
        let old_parent_addr = parent_chain
            .last()
            .map(|link| link.child_addr)
            .ok_or_else(|| {
                Error::InvalidFormat(format!("parent group '{parent_path}' not found"))
            })?;
        let (parent_flags, parent_messages) = self
            .compact_messages_replacing_and_appending_hard_link(
                old_parent_addr,
                target_name,
                target_addr,
                name,
                operation,
            )?;

        let mut new_child_addr =
            self.append_owned_v2_object_header(&parent_messages, parent_flags)?;
        let mut root_update = None;
        let mut dense_update = false;
        for link in parent_chain.iter().rev() {
            match link.storage {
                ParentChainLinkStorage::Compact => {
                    let (ancestor_flags, ancestor_messages) = self
                        .compact_messages_replacing_hard_link(
                            link.ancestor_addr,
                            &link.link_name,
                            new_child_addr,
                            operation,
                        )?;
                    if link.ancestor_addr == root_addr {
                        root_update = Some((ancestor_flags, ancestor_messages));
                        break;
                    }
                    new_child_addr =
                        self.append_owned_v2_object_header(&ancestor_messages, ancestor_flags)?;
                }
                ParentChainLinkStorage::Dense => {
                    self.rewrite_dense_hard_link_addr(
                        link.ancestor_addr,
                        &link.link_name,
                        new_child_addr,
                        operation,
                    )?;
                    dense_update = true;
                    break;
                }
            }
        }
        if dense_update {
            let eof_addr = self.write_handle.seek(SeekFrom::End(0))?;
            self.rewrite_v2_v3_superblock(self.superblock.root_addr, eof_addr)?;
            self.write_handle.flush()?;
            self.reopen_reader()?;
            return Ok(());
        }
        let (root_flags, root_messages) = root_update.ok_or_else(|| {
            Error::InvalidFormat(format!(
                "parent group '{parent_path}' is not linked from the root group"
            ))
        })?;
        self.append_rebuilt_root_link_message(root_messages, root_flags, Vec::new())
    }

    fn append_rebuilt_root_link_message(
        &mut self,
        mut root_messages: Vec<OwnedObjectHeaderMessage>,
        root_flags: u8,
        link: Vec<u8>,
    ) -> Result<()> {
        if !link.is_empty() {
            root_messages.push(OwnedObjectHeaderMessage {
                msg_type: object_header::MSG_LINK,
                flags: 0,
                creation_index: None,
                data: link,
            });
        }

        let new_root_addr = self.append_owned_v2_object_header(&root_messages, root_flags)?;
        let eof_addr = self.write_handle.seek(SeekFrom::End(0))?;
        self.rewrite_v2_v3_superblock(new_root_addr, eof_addr)?;
        self.write_handle.flush()?;
        self.reopen_reader()?;
        Ok(())
    }

    /// Delete a compact link from a group.
    ///
    /// This is an in-place hdf5-metno compatibility mutation. It currently
    /// supports only v2 object headers with compact link messages.
    pub fn unlink_group_link(&mut self, group_path: &str, name: &str) -> Result<()> {
        let group = self.group(group_path)?;
        let location = match self.find_compact_link_in_oh(group.addr(), name, None) {
            Ok(location) => location,
            Err(Error::Unsupported(msg))
                if msg.contains("dense or creation-order indexed links") =>
            {
                return self.unlink_dense_link(group.addr(), name);
            }
            Err(err) => return Err(err),
        };
        if location.link_type == LinkType::Hard {
            let target_addr = location.hard_link_addr.ok_or_else(|| {
                Error::InvalidFormat("hard link message is missing target address".into())
            })?;
            if !self.decrement_object_refcount_if_present(target_addr)? {
                let same_parent_links =
                    self.count_compact_hard_links_to_addr(group.addr(), target_addr)?;
                if same_parent_links <= 1 {
                    return Err(Error::Unsupported(
                        "hard-link deletion without explicit object refcount requires another compact hard link to the same target in the parent group"
                            .into(),
                    ));
                }
            }
        }
        self.write_handle
            .seek(SeekFrom::Start(location.msg_type_offset))?;
        self.write_handle
            .write_all(&[object_header::MSG_NIL as u8])?;
        self.rewrite_oh_checksum(location.oh_start, location.oh_check_len)?;
        self.write_handle.flush()?;
        self.reopen_reader()?;
        Ok(())
    }

    fn unlink_dense_link(&mut self, group_addr: u64, name: &str) -> Result<()> {
        let mut location = self.find_dense_link_location(group_addr, name, None)?;
        let link = compact_link_view(&location.raw_data, self.superblock.sizeof_addr)?;
        if link.link_type == LinkType::Hard {
            let target_addr = link.hard_link_addr.ok_or_else(|| {
                Error::InvalidFormat("hard link message is missing target address".into())
            })?;
            if !self.decrement_object_refcount_if_present(target_addr)? {
                let same_parent_links =
                    self.count_dense_hard_links_to_addr(&location, target_addr)?;
                if same_parent_links <= 1 {
                    return Err(Error::Unsupported(
                        "dense hard-link deletion without explicit object refcount requires another dense hard link to the same target in the parent group"
                            .into(),
                    ));
                }
            }
        }

        let record_size = usize::from(location.btree.record_size);
        if record_size == 0 || location.leaf_records.len() % record_size != 0 {
            return Err(Error::InvalidFormat(
                "dense link records have inconsistent sizes".into(),
            ));
        }
        let record_count = location.leaf_records.len() / record_size;
        if record_count <= 1 {
            return Err(Error::Unsupported(
                "dense link deletion cannot collapse the final dense name-index record yet".into(),
            ));
        }
        if location.record_index >= record_count {
            return Err(Error::InvalidFormat(
                "dense link record index is invalid".into(),
            ));
        }

        let object_addr = location
            .heap
            .root_block_addr
            .checked_add(location.object_offset)
            .ok_or_else(|| Error::InvalidFormat("dense link object address overflow".into()))?;
        let tombstone = vec![0u8; location.raw_data.len()];
        self.write_handle.seek(SeekFrom::Start(object_addr))?;
        self.write_handle.write_all(&tombstone)?;

        let record_start = location
            .record_index
            .checked_mul(record_size)
            .ok_or_else(|| Error::InvalidFormat("dense link record offset overflow".into()))?;
        let record_end = record_start
            .checked_add(record_size)
            .ok_or_else(|| Error::InvalidFormat("dense link record offset overflow".into()))?;
        location.leaf_records.drain(record_start..record_end);

        self.rewrite_dense_link_direct_block_checksum(&location.heap, object_addr, &tombstone)?;
        self.rewrite_dense_link_name_index(&location)?;
        self.write_handle.flush()?;
        self.reopen_reader()?;
        Ok(())
    }

    fn count_dense_hard_links_to_addr(
        &self,
        location: &DenseLinkLocation,
        target_addr: u64,
    ) -> Result<usize> {
        let record_size = usize::from(location.btree.record_size);
        if record_size == 0 || location.leaf_records.len() % record_size != 0 {
            return Err(Error::InvalidFormat(
                "dense link records have inconsistent sizes".into(),
            ));
        }

        let heap_id_len = usize::from(location.heap.heap_id_len);
        let mut guard = self.inner.lock();
        let mut count = 0usize;
        for record in location.leaf_records.chunks_exact(record_size) {
            let heap_id = checked_window(record, 4, heap_id_len, "dense link heap ID")?;
            let raw_data = location
                .heap
                .read_managed_object(&mut guard.reader, heap_id)?;
            let link = compact_link_view(&raw_data, guard.superblock.sizeof_addr)?;
            if link.link_type == LinkType::Hard && link.hard_link_addr == Some(target_addr) {
                count += 1;
            }
        }
        Ok(count)
    }

    /// Rename a compact link in a group without changing the encoded name size.
    ///
    /// Moving between groups or changing the encoded message length needs
    /// object-header growth and is rejected explicitly for now.
    pub fn rename_group_link(
        &mut self,
        group_path: &str,
        old_name: &str,
        new_name: &str,
    ) -> Result<()> {
        if new_name.is_empty() {
            return Err(Error::InvalidFormat("link name cannot be empty".into()));
        }
        if old_name == new_name {
            return Ok(());
        }

        let group = self.group(group_path)?;
        let location = match self.find_compact_link_in_oh(group.addr(), old_name, Some(new_name)) {
            Ok(location) => location,
            Err(Error::Unsupported(msg))
                if msg.contains("dense or creation-order indexed links") =>
            {
                return self.rename_dense_link(group_path, group.addr(), old_name, new_name);
            }
            Err(err) => return Err(err),
        };
        if new_name.len() != location.name_size {
            if group_path == "/" {
                return self.rename_root_compact_link_by_rebuild(old_name, new_name);
            }
            return self.rename_nested_link_by_rebuild(group_path, old_name, new_name);
        }

        let name_offset_u64 = Self::usize_to_u64(location.name_offset, "link name offset")?;
        let file_name_offset = location
            .msg_data_offset
            .checked_add(name_offset_u64)
            .ok_or_else(|| Error::InvalidFormat("link name offset overflow".into()))?;
        self.write_handle.seek(SeekFrom::Start(file_name_offset))?;
        self.write_handle.write_all(new_name.as_bytes())?;
        self.rewrite_oh_checksum(location.oh_start, location.oh_check_len)?;
        self.write_handle.flush()?;
        self.reopen_reader()?;
        Ok(())
    }

    pub fn move_group_link(
        &mut self,
        source_group_path: &str,
        old_name: &str,
        dest_group_path: &str,
        new_name: &str,
    ) -> Result<()> {
        validate_direct_child_link_name(old_name)?;
        validate_direct_child_link_name(new_name)?;
        if source_group_path == dest_group_path {
            return self.rename_group_link(source_group_path, old_name, new_name);
        }
        if self.superblock.version < 2 {
            return Err(Error::Unsupported(
                "cross-group compact relink requires a v2/v3 superblock".into(),
            ));
        }

        let operation = "cross-group compact relink";
        let mut path_addrs = BTreeMap::new();
        path_addrs.insert(Vec::<String>::new(), self.superblock.root_addr);
        self.record_compact_path_addrs(source_group_path, operation, &mut path_addrs)?;
        self.record_compact_path_addrs(dest_group_path, operation, &mut path_addrs)?;

        let source_components =
            owned_compact_absolute_path_components(source_group_path, operation)?;
        let dest_components = owned_compact_absolute_path_components(dest_group_path, operation)?;
        let source_addr = *path_addrs.get(&source_components).ok_or_else(|| {
            Error::InvalidFormat(format!("source group '{source_group_path}' not found"))
        })?;
        let dest_addr = *path_addrs.get(&dest_components).ok_or_else(|| {
            Error::InvalidFormat(format!("destination group '{dest_group_path}' not found"))
        })?;

        let (source_flags, source_messages, moved_link) =
            self.messages_removing_link(source_addr, old_name, new_name, operation)?;
        let (dest_flags, mut dest_messages) =
            self.messages_for_append(dest_addr, new_name, operation)?;
        dest_messages.push(OwnedObjectHeaderMessage {
            msg_type: object_header::MSG_LINK,
            flags: 0,
            creation_index: None,
            data: moved_link,
        });

        let mut modified = BTreeMap::new();
        modified.insert(source_components, (source_flags, source_messages));
        modified.insert(dest_components, (dest_flags, dest_messages));
        self.append_rebuilt_compact_paths(modified, path_addrs, operation)
    }

    fn rename_root_compact_link_by_rebuild(
        &mut self,
        old_name: &str,
        new_name: &str,
    ) -> Result<()> {
        validate_direct_child_link_name(old_name)?;
        validate_direct_child_link_name(new_name)?;
        if self.superblock.version < 2 {
            return Err(Error::Unsupported(
                "root compact relink with changed name length requires a v2/v3 superblock".into(),
            ));
        }

        let root_addr = self.superblock.root_addr;
        let mut guard = self.inner.lock();
        let sizeof_addr = guard.superblock.sizeof_addr;
        let oh = ObjectHeader::read_at(&mut guard.reader, root_addr)?;
        if oh.version != 2 {
            return Err(Error::Unsupported(
                "root compact relink with changed name length currently supports only v2 object headers"
                    .into(),
            ));
        }
        if oh.flags & HDR_ATTR_CRT_ORDER_TRACKED != 0 {
            return Err(Error::Unsupported(
                "root compact relink with creation-order tracking is not implemented".into(),
            ));
        }

        let mut messages = Vec::new();
        let mut renamed_link = None;
        for msg in &oh.messages {
            match msg.msg_type {
                object_header::MSG_LINK => {
                    let link = LinkMessage::decode(&msg.data, sizeof_addr)?;
                    if link.name == new_name {
                        return Err(Error::InvalidFormat(format!(
                            "link '{new_name}' already exists"
                        )));
                    }
                    if link.name == old_name {
                        if renamed_link.is_some() {
                            return Err(Error::InvalidFormat(format!(
                                "link '{old_name}' appears more than once"
                            )));
                        }
                        renamed_link =
                            Some(encode_renamed_link_message(&link, new_name, sizeof_addr)?);
                    } else {
                        messages.push(OwnedObjectHeaderMessage::from_raw(msg));
                    }
                }
                object_header::MSG_LINK_INFO => {
                    let link_info = LinkInfoMessage::decode(&msg.data, sizeof_addr)?;
                    if link_info.has_dense_storage() || link_info.corder_btree_addr.is_some() {
                        return Err(Error::Unsupported(
                            "root compact relink for dense or creation-order indexed links is not implemented"
                                .into(),
                        ));
                    }
                    messages.push(OwnedObjectHeaderMessage::from_raw(msg));
                }
                object_header::MSG_SYMBOL_TABLE => {
                    return Err(Error::Unsupported(
                        "root compact relink for v1 symbol-table groups is not implemented".into(),
                    ));
                }
                object_header::MSG_HEADER_CONTINUATION | object_header::MSG_NIL => {}
                _ => messages.push(OwnedObjectHeaderMessage::from_raw(msg)),
            }
        }
        drop(guard);

        let renamed_link = renamed_link
            .ok_or_else(|| Error::InvalidFormat(format!("link '{old_name}' not found")))?;
        self.append_rebuilt_root_link_message(messages, oh.flags, renamed_link)
    }

    fn rename_nested_link_by_rebuild(
        &mut self,
        group_path: &str,
        old_name: &str,
        new_name: &str,
    ) -> Result<()> {
        validate_direct_child_link_name(old_name)?;
        validate_direct_child_link_name(new_name)?;
        if self.superblock.version < 2 {
            return Err(Error::Unsupported(
                "nested compact relink with changed name length requires a v2/v3 superblock".into(),
            ));
        }

        let operation = "nested compact relink";
        let root_addr = self.superblock.root_addr;
        let parent_chain = self.compact_or_dense_hard_link_parent_chain(group_path, operation)?;
        let old_group_addr = parent_chain
            .last()
            .map(|link| link.child_addr)
            .ok_or_else(|| {
                Error::InvalidFormat(format!("group '{group_path}' is not linked from the root"))
            })?;
        let (group_flags, group_messages) =
            self.messages_renaming_link(old_group_addr, old_name, new_name, operation)?;

        let mut new_child_addr =
            self.append_owned_v2_object_header(&group_messages, group_flags)?;
        let mut root_update = None;
        let mut dense_update = false;
        for link in parent_chain.iter().rev() {
            match link.storage {
                ParentChainLinkStorage::Compact => {
                    let (ancestor_flags, ancestor_messages) = self
                        .compact_messages_replacing_hard_link(
                            link.ancestor_addr,
                            &link.link_name,
                            new_child_addr,
                            operation,
                        )?;
                    if link.ancestor_addr == root_addr {
                        root_update = Some((ancestor_flags, ancestor_messages));
                        break;
                    }
                    new_child_addr =
                        self.append_owned_v2_object_header(&ancestor_messages, ancestor_flags)?;
                }
                ParentChainLinkStorage::Dense => {
                    self.rewrite_dense_hard_link_addr(
                        link.ancestor_addr,
                        &link.link_name,
                        new_child_addr,
                        operation,
                    )?;
                    dense_update = true;
                    break;
                }
            }
        }
        if dense_update {
            let eof_addr = self.write_handle.seek(SeekFrom::End(0))?;
            self.rewrite_v2_v3_superblock(self.superblock.root_addr, eof_addr)?;
            self.write_handle.flush()?;
            self.reopen_reader()?;
            return Ok(());
        }
        let (root_flags, root_messages) = root_update.ok_or_else(|| {
            Error::InvalidFormat(format!(
                "group '{group_path}' is not linked from the root group"
            ))
        })?;
        self.append_rebuilt_root_link_message(root_messages, root_flags, Vec::new())
    }

    fn compact_messages_renaming_link(
        &self,
        group_addr: u64,
        old_name: &str,
        new_name: &str,
        operation: &str,
    ) -> Result<(u8, Vec<OwnedObjectHeaderMessage>)> {
        let mut guard = self.inner.lock();
        let sizeof_addr = guard.superblock.sizeof_addr;
        let oh = ObjectHeader::read_at(&mut guard.reader, group_addr)?;
        if oh.version != 2 {
            return Err(Error::Unsupported(format!(
                "{operation} currently supports only v2 object headers"
            )));
        }
        if oh.flags & HDR_ATTR_CRT_ORDER_TRACKED != 0 {
            return Err(Error::Unsupported(format!(
                "{operation} with object-header creation-order tracking is not implemented"
            )));
        }

        let mut renamed = false;
        let mut messages = Vec::new();
        for msg in &oh.messages {
            match msg.msg_type {
                object_header::MSG_LINK => {
                    let link = LinkMessage::decode(&msg.data, sizeof_addr)?;
                    if link.name == new_name {
                        return Err(Error::InvalidFormat(format!(
                            "link '{new_name}' already exists"
                        )));
                    }
                    if link.name == old_name {
                        if renamed {
                            return Err(Error::InvalidFormat(format!(
                                "link '{old_name}' appears more than once"
                            )));
                        }
                        messages.push(OwnedObjectHeaderMessage {
                            msg_type: object_header::MSG_LINK,
                            flags: msg.flags,
                            creation_index: msg.creation_index,
                            data: encode_renamed_link_message(&link, new_name, sizeof_addr)?,
                        });
                        renamed = true;
                    } else {
                        messages.push(OwnedObjectHeaderMessage::from_raw(msg));
                    }
                }
                object_header::MSG_LINK_INFO => {
                    let link_info = LinkInfoMessage::decode(&msg.data, sizeof_addr)?;
                    if link_info.has_dense_storage() || link_info.corder_btree_addr.is_some() {
                        return Err(Error::Unsupported(format!(
                            "{operation} for dense or creation-order indexed links is not implemented"
                        )));
                    }
                    messages.push(OwnedObjectHeaderMessage::from_raw(msg));
                }
                object_header::MSG_SYMBOL_TABLE => {
                    return Err(Error::Unsupported(format!(
                        "{operation} for v1 symbol-table groups is not implemented"
                    )));
                }
                object_header::MSG_HEADER_CONTINUATION | object_header::MSG_NIL => {}
                _ => messages.push(OwnedObjectHeaderMessage::from_raw(msg)),
            }
        }
        if !renamed {
            return Err(Error::InvalidFormat(format!("link '{old_name}' not found")));
        }
        Ok((oh.flags, messages))
    }

    fn messages_renaming_link(
        &self,
        group_addr: u64,
        old_name: &str,
        new_name: &str,
        operation: &str,
    ) -> Result<(u8, Vec<OwnedObjectHeaderMessage>)> {
        match self.compact_messages_renaming_link(group_addr, old_name, new_name, operation) {
            Err(Error::Unsupported(msg))
                if msg.contains("dense or creation-order indexed links") =>
            {
                self.compact_messages_from_dense_links_renaming_link(
                    group_addr, old_name, new_name, operation,
                )
            }
            other => other,
        }
    }

    fn compact_messages_from_dense_links_renaming_link(
        &self,
        group_addr: u64,
        old_name: &str,
        new_name: &str,
        operation: &str,
    ) -> Result<(u8, Vec<OwnedObjectHeaderMessage>)> {
        let mut guard = self.inner.lock();
        let sizeof_addr = guard.superblock.sizeof_addr;
        let oh = ObjectHeader::read_at(&mut guard.reader, group_addr)?;
        if oh.version != 2 {
            return Err(Error::Unsupported(format!(
                "{operation} currently supports only v2 object headers"
            )));
        }
        if oh.flags & HDR_ATTR_CRT_ORDER_TRACKED != 0 {
            return Err(Error::Unsupported(format!(
                "{operation} with object-header creation-order tracking is not implemented"
            )));
        }

        let mut renamed = false;
        let mut messages = Vec::new();
        for msg in &oh.messages {
            match msg.msg_type {
                object_header::MSG_LINK => {
                    let link = LinkMessage::decode(&msg.data, sizeof_addr)?;
                    if link.name == new_name {
                        return Err(Error::InvalidFormat(format!(
                            "link '{new_name}' already exists"
                        )));
                    }
                    if link.name == old_name {
                        if renamed {
                            return Err(Error::InvalidFormat(format!(
                                "link '{old_name}' appears more than once"
                            )));
                        }
                        messages.push(OwnedObjectHeaderMessage {
                            msg_type: object_header::MSG_LINK,
                            flags: msg.flags,
                            creation_index: msg.creation_index,
                            data: encode_renamed_link_message(&link, new_name, sizeof_addr)?,
                        });
                        renamed = true;
                    } else {
                        messages.push(OwnedObjectHeaderMessage::from_raw(msg));
                    }
                }
                object_header::MSG_LINK_INFO => {
                    let link_info = LinkInfoMessage::decode(&msg.data, sizeof_addr)?;
                    if link_info.corder_btree_addr.is_some() {
                        return Err(Error::Unsupported(format!(
                            "{operation} for creation-order indexed links is not implemented"
                        )));
                    }
                    if !link_info.has_dense_storage() {
                        messages.push(OwnedObjectHeaderMessage::from_raw(msg));
                        continue;
                    }

                    let heap =
                        FractalHeapHeader::read_at(&mut guard.reader, link_info.fractal_heap_addr)?;
                    if heap.io_filter_len != 0 || heap.current_root_rows != 0 {
                        return Err(Error::Unsupported(
                            "mutating filtered or indirect dense link heaps is not implemented"
                                .into(),
                        ));
                    }
                    let btree =
                        BTreeV2Header::read_at(&mut guard.reader, link_info.name_btree_addr)?;
                    if btree.depth != 0 {
                        return Err(Error::Unsupported(
                            "mutating non-leaf dense link name indexes is not implemented".into(),
                        ));
                    }
                    let mut leaf_records = Vec::new();
                    Self::read_dense_link_leaf_records_into(
                        &mut guard.reader,
                        &btree,
                        &mut leaf_records,
                    )?;
                    let heap_id_len = usize::from(heap.heap_id_len);
                    let record_size = usize::from(btree.record_size);
                    for record in leaf_records.chunks_exact(record_size) {
                        let heap_id = checked_window(record, 4, heap_id_len, "dense link heap ID")?;
                        let raw_data = heap.read_managed_object(&mut guard.reader, heap_id)?;
                        let link = LinkMessage::decode(&raw_data, sizeof_addr)?;
                        if link.name == new_name {
                            return Err(Error::InvalidFormat(format!(
                                "link '{new_name}' already exists"
                            )));
                        }
                        if link.name == old_name {
                            if renamed {
                                return Err(Error::InvalidFormat(format!(
                                    "link '{old_name}' appears more than once"
                                )));
                            }
                            messages.push(OwnedObjectHeaderMessage {
                                msg_type: object_header::MSG_LINK,
                                flags: 0,
                                creation_index: None,
                                data: encode_renamed_link_message(&link, new_name, sizeof_addr)?,
                            });
                            renamed = true;
                        } else {
                            messages.push(OwnedObjectHeaderMessage {
                                msg_type: object_header::MSG_LINK,
                                flags: 0,
                                creation_index: None,
                                data: raw_data,
                            });
                        }
                    }
                }
                object_header::MSG_SYMBOL_TABLE => {
                    return Err(Error::Unsupported(format!(
                        "{operation} for v1 symbol-table groups is not implemented"
                    )));
                }
                object_header::MSG_HEADER_CONTINUATION | object_header::MSG_NIL => {}
                _ => messages.push(OwnedObjectHeaderMessage::from_raw(msg)),
            }
        }
        if !renamed {
            return Err(Error::InvalidFormat(format!("link '{old_name}' not found")));
        }
        Ok((oh.flags, messages))
    }

    fn compact_messages_removing_link(
        &self,
        group_addr: u64,
        old_name: &str,
        new_name: &str,
        operation: &str,
    ) -> Result<(u8, Vec<OwnedObjectHeaderMessage>, Vec<u8>)> {
        let mut guard = self.inner.lock();
        let sizeof_addr = guard.superblock.sizeof_addr;
        let oh = ObjectHeader::read_at(&mut guard.reader, group_addr)?;
        if oh.version != 2 {
            return Err(Error::Unsupported(format!(
                "{operation} currently supports only v2 object headers"
            )));
        }
        if oh.flags & HDR_ATTR_CRT_ORDER_TRACKED != 0 {
            return Err(Error::Unsupported(format!(
                "{operation} with object-header creation-order tracking is not implemented"
            )));
        }

        let mut removed_link = None;
        let mut messages = Vec::new();
        for msg in &oh.messages {
            match msg.msg_type {
                object_header::MSG_LINK => {
                    let link = LinkMessage::decode(&msg.data, sizeof_addr)?;
                    if link.name == old_name {
                        if removed_link.is_some() {
                            return Err(Error::InvalidFormat(format!(
                                "link '{old_name}' appears more than once"
                            )));
                        }
                        removed_link =
                            Some(encode_renamed_link_message(&link, new_name, sizeof_addr)?);
                    } else {
                        messages.push(OwnedObjectHeaderMessage::from_raw(msg));
                    }
                }
                object_header::MSG_LINK_INFO => {
                    let link_info = LinkInfoMessage::decode(&msg.data, sizeof_addr)?;
                    if link_info.has_dense_storage() || link_info.corder_btree_addr.is_some() {
                        return Err(Error::Unsupported(format!(
                            "{operation} for dense or creation-order indexed links is not implemented"
                        )));
                    }
                    messages.push(OwnedObjectHeaderMessage::from_raw(msg));
                }
                object_header::MSG_SYMBOL_TABLE => {
                    return Err(Error::Unsupported(format!(
                        "{operation} for v1 symbol-table groups is not implemented"
                    )));
                }
                object_header::MSG_HEADER_CONTINUATION | object_header::MSG_NIL => {}
                _ => messages.push(OwnedObjectHeaderMessage::from_raw(msg)),
            }
        }

        let removed_link = removed_link
            .ok_or_else(|| Error::InvalidFormat(format!("link '{old_name}' not found")))?;
        Ok((oh.flags, messages, removed_link))
    }

    fn messages_removing_link(
        &self,
        group_addr: u64,
        old_name: &str,
        new_name: &str,
        operation: &str,
    ) -> Result<(u8, Vec<OwnedObjectHeaderMessage>, Vec<u8>)> {
        match self.compact_messages_removing_link(group_addr, old_name, new_name, operation) {
            Err(Error::Unsupported(msg))
                if msg.contains("dense or creation-order indexed links") =>
            {
                self.compact_messages_from_dense_links_removing_link(
                    group_addr, old_name, new_name, operation,
                )
            }
            other => other,
        }
    }

    fn compact_messages_from_dense_links_removing_link(
        &self,
        group_addr: u64,
        old_name: &str,
        new_name: &str,
        operation: &str,
    ) -> Result<(u8, Vec<OwnedObjectHeaderMessage>, Vec<u8>)> {
        let mut guard = self.inner.lock();
        let sizeof_addr = guard.superblock.sizeof_addr;
        let oh = ObjectHeader::read_at(&mut guard.reader, group_addr)?;
        if oh.version != 2 {
            return Err(Error::Unsupported(format!(
                "{operation} currently supports only v2 object headers"
            )));
        }
        if oh.flags & HDR_ATTR_CRT_ORDER_TRACKED != 0 {
            return Err(Error::Unsupported(format!(
                "{operation} with object-header creation-order tracking is not implemented"
            )));
        }

        let mut removed_link = None;
        let mut messages = Vec::new();
        for msg in &oh.messages {
            match msg.msg_type {
                object_header::MSG_LINK => {
                    let link = LinkMessage::decode(&msg.data, sizeof_addr)?;
                    if link.name == old_name {
                        if removed_link.is_some() {
                            return Err(Error::InvalidFormat(format!(
                                "link '{old_name}' appears more than once"
                            )));
                        }
                        removed_link =
                            Some(encode_renamed_link_message(&link, new_name, sizeof_addr)?);
                    } else {
                        messages.push(OwnedObjectHeaderMessage::from_raw(msg));
                    }
                }
                object_header::MSG_LINK_INFO => {
                    let link_info = LinkInfoMessage::decode(&msg.data, sizeof_addr)?;
                    if link_info.corder_btree_addr.is_some() {
                        return Err(Error::Unsupported(format!(
                            "{operation} for creation-order indexed links is not implemented"
                        )));
                    }
                    if !link_info.has_dense_storage() {
                        messages.push(OwnedObjectHeaderMessage::from_raw(msg));
                        continue;
                    }

                    let heap =
                        FractalHeapHeader::read_at(&mut guard.reader, link_info.fractal_heap_addr)?;
                    if heap.io_filter_len != 0 || heap.current_root_rows != 0 {
                        return Err(Error::Unsupported(
                            "mutating filtered or indirect dense link heaps is not implemented"
                                .into(),
                        ));
                    }
                    let btree =
                        BTreeV2Header::read_at(&mut guard.reader, link_info.name_btree_addr)?;
                    if btree.depth != 0 {
                        return Err(Error::Unsupported(
                            "mutating non-leaf dense link name indexes is not implemented".into(),
                        ));
                    }
                    let mut leaf_records = Vec::new();
                    Self::read_dense_link_leaf_records_into(
                        &mut guard.reader,
                        &btree,
                        &mut leaf_records,
                    )?;
                    let heap_id_len = usize::from(heap.heap_id_len);
                    let record_size = usize::from(btree.record_size);
                    for record in leaf_records.chunks_exact(record_size) {
                        let heap_id = checked_window(record, 4, heap_id_len, "dense link heap ID")?;
                        let raw_data = heap.read_managed_object(&mut guard.reader, heap_id)?;
                        let link = LinkMessage::decode(&raw_data, sizeof_addr)?;
                        if link.name == old_name {
                            if removed_link.is_some() {
                                return Err(Error::InvalidFormat(format!(
                                    "link '{old_name}' appears more than once"
                                )));
                            }
                            removed_link =
                                Some(encode_renamed_link_message(&link, new_name, sizeof_addr)?);
                        } else {
                            messages.push(OwnedObjectHeaderMessage {
                                msg_type: object_header::MSG_LINK,
                                flags: 0,
                                creation_index: None,
                                data: raw_data,
                            });
                        }
                    }
                }
                object_header::MSG_SYMBOL_TABLE => {
                    return Err(Error::Unsupported(format!(
                        "{operation} for v1 symbol-table groups is not implemented"
                    )));
                }
                object_header::MSG_HEADER_CONTINUATION | object_header::MSG_NIL => {}
                _ => messages.push(OwnedObjectHeaderMessage::from_raw(msg)),
            }
        }

        let removed_link = removed_link
            .ok_or_else(|| Error::InvalidFormat(format!("link '{old_name}' not found")))?;
        Ok((oh.flags, messages, removed_link))
    }

    fn record_compact_path_addrs(
        &self,
        parent_path: &str,
        operation: &str,
        path_addrs: &mut BTreeMap<Vec<String>, u64>,
    ) -> Result<()> {
        let mut components = Vec::new();
        let mut ancestor_addr = self.superblock.root_addr;
        for name in owned_compact_absolute_path_components(parent_path, operation)? {
            let child_addr = self.compact_hard_link_addr(ancestor_addr, &name, operation)?;
            components.push(name);
            path_addrs.insert(components.clone(), child_addr);
            ancestor_addr = child_addr;
        }
        Ok(())
    }

    fn append_rebuilt_compact_paths(
        &mut self,
        mut modified: BTreeMap<Vec<String>, (u8, Vec<OwnedObjectHeaderMessage>)>,
        path_addrs: BTreeMap<Vec<String>, u64>,
        operation: &str,
    ) -> Result<()> {
        let max_depth = modified.keys().map(Vec::len).max().unwrap_or(0);
        let mut replacements: BTreeMap<Vec<String>, Vec<(String, u64)>> = BTreeMap::new();

        for depth in (1..=max_depth).rev() {
            let paths: Vec<Vec<String>> = modified
                .keys()
                .filter(|path| path.len() == depth)
                .cloned()
                .collect();
            for path in paths {
                let (flags, messages) = modified.remove(&path).ok_or_else(|| {
                    Error::InvalidFormat("compact path rebuild state is inconsistent".into())
                })?;
                let new_addr = self.append_owned_v2_object_header(&messages, flags)?;
                let link_name = path.last().cloned().ok_or_else(|| {
                    Error::InvalidFormat("compact path rebuild missing link name".into())
                })?;
                let parent_path = path[..path.len() - 1].to_vec();
                replacements
                    .entry(parent_path.clone())
                    .or_default()
                    .push((link_name, new_addr));

                let Some(parent_replacements) = replacements.remove(&parent_path) else {
                    continue;
                };
                let parent_addr = *path_addrs.get(&parent_path).ok_or_else(|| {
                    Error::InvalidFormat(format!(
                        "{operation} parent path is not linked from the root group"
                    ))
                })?;
                let (parent_flags, parent_messages) =
                    if let Some((flags, messages)) = modified.remove(&parent_path) {
                        replace_hard_links_in_messages(
                            messages,
                            flags,
                            &parent_replacements,
                            self.superblock.sizeof_addr,
                            operation,
                        )?
                    } else {
                        self.compact_messages_replacing_hard_links(
                            parent_addr,
                            &parent_replacements,
                            operation,
                        )?
                    };
                modified.insert(parent_path, (parent_flags, parent_messages));
            }
        }

        let (root_flags, root_messages) =
            modified.remove(&Vec::<String>::new()).ok_or_else(|| {
                Error::InvalidFormat(format!(
                    "{operation} did not produce a rebuilt root object header"
                ))
            })?;
        self.append_rebuilt_root_link_message(root_messages, root_flags, Vec::new())
    }

    fn rename_dense_link(
        &mut self,
        group_path: &str,
        group_addr: u64,
        old_name: &str,
        new_name: &str,
    ) -> Result<()> {
        let mut location = self.find_dense_link_location(group_addr, old_name, Some(new_name))?;
        let link = compact_link_view(&location.raw_data, self.superblock.sizeof_addr)?;
        let name_offset = link.name_offset;
        let name_size = link.name_size;
        if new_name.len() != name_size {
            let link = LinkMessage::decode(&location.raw_data, self.superblock.sizeof_addr)?;
            let mut renamed =
                encode_renamed_link_message(&link, new_name, self.superblock.sizeof_addr)?;
            if renamed.len() > location.raw_data.len() {
                if group_path == "/" {
                    let operation = "root dense relink";
                    let (root_flags, root_messages) =
                        self.messages_renaming_link(group_addr, old_name, new_name, operation)?;
                    return self.append_rebuilt_root_link_message(
                        root_messages,
                        root_flags,
                        Vec::new(),
                    );
                }
                return self.rename_nested_link_by_rebuild(group_path, old_name, new_name);
            }
            renamed.resize(location.raw_data.len(), 0);

            let object_addr = location
                .heap
                .root_block_addr
                .checked_add(location.object_offset)
                .ok_or_else(|| Error::InvalidFormat("dense link object address overflow".into()))?;
            self.write_handle.seek(SeekFrom::Start(object_addr))?;
            self.write_handle.write_all(&renamed)?;

            let record_size = usize::from(location.btree.record_size);
            let record_start = location
                .record_index
                .checked_mul(record_size)
                .ok_or_else(|| Error::InvalidFormat("dense link record offset overflow".into()))?;
            let record_end = record_start
                .checked_add(record_size)
                .ok_or_else(|| Error::InvalidFormat("dense link record offset overflow".into()))?;
            let record = location
                .leaf_records
                .get_mut(record_start..record_end)
                .ok_or_else(|| Error::InvalidFormat("dense link record index is invalid".into()))?;
            record[..4].copy_from_slice(&dense_link_name_hash(new_name).to_le_bytes());
            Self::reposition_dense_link_record_by_hash(
                &mut location.leaf_records,
                location.record_index,
                record_size,
            )?;

            self.rewrite_dense_link_direct_block_checksum(&location.heap, object_addr, &renamed)?;
            self.rewrite_dense_link_name_index(&location)?;
            self.write_handle.flush()?;
            self.reopen_reader()?;
            return Ok(());
        }

        let name_offset_u64 = Self::usize_to_u64(name_offset, "dense link name offset")?;
        let file_name_offset = location
            .heap
            .root_block_addr
            .checked_add(location.object_offset)
            .and_then(|offset| offset.checked_add(name_offset_u64))
            .ok_or_else(|| Error::InvalidFormat("dense link name offset overflow".into()))?;
        let encoded_name =
            encode_link_name_in_place(&mut location.raw_data, name_offset, name_size, new_name)?;
        let encoded_name = encoded_name.to_vec();
        self.write_handle.seek(SeekFrom::Start(file_name_offset))?;
        self.write_handle.write_all(&encoded_name)?;

        let record_size = usize::from(location.btree.record_size);
        let record_start = location
            .record_index
            .checked_mul(record_size)
            .ok_or_else(|| Error::InvalidFormat("dense link record offset overflow".into()))?;
        let record_end = record_start
            .checked_add(record_size)
            .ok_or_else(|| Error::InvalidFormat("dense link record offset overflow".into()))?;
        let record = location
            .leaf_records
            .get_mut(record_start..record_end)
            .ok_or_else(|| Error::InvalidFormat("dense link record index is invalid".into()))?;
        record[..4].copy_from_slice(&dense_link_name_hash(new_name).to_le_bytes());
        Self::reposition_dense_link_record_by_hash(
            &mut location.leaf_records,
            location.record_index,
            record_size,
        )?;

        self.rewrite_dense_link_direct_block_checksum(
            &location.heap,
            file_name_offset,
            &encoded_name,
        )?;
        self.rewrite_dense_link_name_index(&location)?;
        self.write_handle.flush()?;
        self.reopen_reader()?;
        Ok(())
    }

    fn rewrite_dense_hard_link_addr(
        &mut self,
        group_addr: u64,
        name: &str,
        new_addr: u64,
        operation: &str,
    ) -> Result<()> {
        let mut location = self.find_dense_link_location(group_addr, name, None)?;
        let link = compact_link_view(&location.raw_data, self.superblock.sizeof_addr)?;
        if link.link_type != LinkType::Hard {
            return Err(Error::Unsupported(format!(
                "{operation} address propagation requires hard-link ancestors"
            )));
        }
        let addr_offset = link.hard_link_addr_offset.ok_or_else(|| {
            Error::InvalidFormat("hard link message is missing target address".into())
        })?;
        patch_le_uint_width(
            &mut location.raw_data,
            addr_offset,
            new_addr,
            usize::from(self.superblock.sizeof_addr),
            "hard link address",
        )?;
        let object_addr = location
            .heap
            .root_block_addr
            .checked_add(location.object_offset)
            .ok_or_else(|| Error::InvalidFormat("dense link object address overflow".into()))?;
        self.write_handle.seek(SeekFrom::Start(object_addr))?;
        self.write_handle.write_all(&location.raw_data)?;
        self.rewrite_dense_link_direct_block_checksum(
            &location.heap,
            object_addr,
            &location.raw_data,
        )
    }

    fn find_compact_link_in_oh(
        &self,
        oh_addr: u64,
        target_name: &str,
        reject_duplicate_name: Option<&str>,
    ) -> Result<CompactLinkMessageLocation> {
        let mut guard = self.inner.lock();
        let sizeof_addr = guard.superblock.sizeof_addr;
        let reader = &mut guard.reader;
        reader.seek(oh_addr)?;

        let mut first_bytes = [0u8; 4];
        reader.read_bytes_into(&mut first_bytes)?;
        if first_bytes != [b'O', b'H', b'D', b'R'] {
            return Err(Error::Unsupported(
                "link mutation currently supports only v2 object headers".into(),
            ));
        }

        let version = reader.read_u8()?;
        if version != 2 {
            return Err(Error::Unsupported(
                "link mutation currently supports only v2 object headers".into(),
            ));
        }

        let flags = reader.read_u8()?;
        if flags & !HDR_V2_KNOWN_FLAGS != 0 {
            return Err(Error::InvalidFormat(format!(
                "object header v2 flags contain reserved bits: {flags:#04x}"
            )));
        }
        if flags & HDR_STORE_TIMES != 0 {
            reader.skip(16)?;
        }
        if flags & HDR_ATTR_STORE_PHASE_CHANGE != 0 {
            reader.skip(4)?;
        }

        let chunk0_size_bytes = 1u8 << (flags & HDR_CHUNK0_SIZE_MASK);
        let chunk0_data_size = reader.read_uint(chunk0_size_bytes)?;
        let chunk0_data_start = reader.position()?;
        let chunk0_data_end = chunk0_data_start
            .checked_add(chunk0_data_size)
            .ok_or_else(|| Error::InvalidFormat("object-header chunk range overflow".into()))?;
        let oh_check_len = usize::try_from(chunk0_data_end - oh_addr)
            .map_err(|_| Error::InvalidFormat("object-header checksum range overflow".into()))?;

        let mut has_dense_links = false;
        let mut found = None;
        let mut msg_buf = Vec::new();
        while reader.position()? < chunk0_data_end {
            let msg_header_pos = reader.position()?;
            if msg_header_pos
                .checked_add(4)
                .is_none_or(|end| end > chunk0_data_end)
            {
                break;
            }

            let msg_type = u16::from(reader.read_u8()?);
            let msg_size = usize::from(reader.read_u16()?);
            let _msg_flags = reader.read_u8()?;
            if flags & HDR_ATTR_CRT_ORDER_TRACKED != 0 {
                reader.skip(2)?;
            }

            let msg_data_offset = reader.position()?;
            let msg_size_u64 = Self::usize_to_u64(msg_size, "object-header message size")?;
            if msg_data_offset
                .checked_add(msg_size_u64)
                .is_none_or(|end| end > chunk0_data_end)
            {
                return Err(Error::InvalidFormat(
                    "object-header message payload exceeds chunk".into(),
                ));
            }

            if msg_type == object_header::MSG_LINK {
                read_message_into(reader, &mut msg_buf, msg_size)?;
                let link = compact_link_view(&msg_buf, sizeof_addr)?;
                if let Some(duplicate_name) = reject_duplicate_name {
                    if link.name == duplicate_name {
                        return Err(Error::InvalidFormat(format!(
                            "link '{duplicate_name}' already exists"
                        )));
                    }
                }
                if link.name == target_name {
                    let location = CompactLinkMessageLocation {
                        msg_type_offset: msg_header_pos,
                        msg_data_offset,
                        oh_start: oh_addr,
                        oh_check_len,
                        name_offset: link.name_offset,
                        name_size: link.name_size,
                        link_type: link.link_type,
                        hard_link_addr: link.hard_link_addr,
                    };
                    if reject_duplicate_name.is_none() {
                        return Ok(location);
                    }
                    found = Some(location);
                }
            } else {
                if msg_type == object_header::MSG_SYMBOL_TABLE {
                    return Err(Error::Unsupported(
                        "mutating v1 symbol-table group links is not implemented".into(),
                    ));
                } else if msg_type == object_header::MSG_LINK_INFO {
                    read_message_into(reader, &mut msg_buf, msg_size)?;
                    let link_info = LinkInfoMessage::decode(&msg_buf, sizeof_addr)?;
                    if link_info.has_dense_storage() || link_info.corder_btree_addr.is_some() {
                        has_dense_links = true;
                    }
                } else {
                    reader.skip(msg_size_u64)?;
                }
            }
        }

        if has_dense_links {
            return Err(Error::Unsupported(
                "mutating dense or creation-order indexed links is not implemented".into(),
            ));
        }
        found.ok_or_else(|| Error::InvalidFormat(format!("link '{target_name}' not found")))
    }

    fn find_dense_link_location(
        &self,
        group_addr: u64,
        target_name: &str,
        reject_duplicate_name: Option<&str>,
    ) -> Result<DenseLinkLocation> {
        let mut guard = self.inner.lock();
        let link_info = Self::read_dense_link_info_message(&mut guard.reader, group_addr)?;
        if !link_info.has_dense_storage() {
            return Err(Error::InvalidFormat(format!(
                "link '{target_name}' not found"
            )));
        }
        if link_info.corder_btree_addr.is_some() {
            return Err(Error::Unsupported(
                "mutating creation-order indexed dense links is not implemented".into(),
            ));
        }

        let heap = FractalHeapHeader::read_at(&mut guard.reader, link_info.fractal_heap_addr)?;
        if heap.io_filter_len != 0 || heap.current_root_rows != 0 {
            return Err(Error::Unsupported(
                "mutating filtered or indirect dense link heaps is not implemented".into(),
            ));
        }

        let btree = BTreeV2Header::read_at(&mut guard.reader, link_info.name_btree_addr)?;
        if btree.depth != 0 {
            return Err(Error::Unsupported(
                "mutating non-leaf dense link name indexes is not implemented".into(),
            ));
        }
        let mut leaf_records = Vec::new();
        Self::read_dense_link_leaf_records_into(&mut guard.reader, &btree, &mut leaf_records)?;
        let heap_id_len = usize::from(heap.heap_id_len);
        let record_size = usize::from(btree.record_size);
        let mut found = None;
        for (idx, record) in leaf_records.chunks_exact(record_size).enumerate() {
            let heap_id = checked_window(record, 4, heap_id_len, "dense link heap ID")?;
            let raw_data = heap.read_managed_object(&mut guard.reader, heap_id)?;
            let link = compact_link_view(&raw_data, guard.superblock.sizeof_addr)?;
            if reject_duplicate_name.is_some_and(|name| link.name == name) {
                return Err(Error::InvalidFormat(format!(
                    "link '{}' already exists",
                    link.name
                )));
            }
            if link.name == target_name {
                let object_offset = managed_heap_object_offset(&heap, heap_id)?;
                found = Some((idx, object_offset, raw_data));
            }
        }

        let (record_index, object_offset, raw_data) =
            found.ok_or_else(|| Error::InvalidFormat(format!("link '{target_name}' not found")))?;
        Ok(DenseLinkLocation {
            btree_header_addr: link_info.name_btree_addr,
            heap,
            btree,
            leaf_records,
            record_index,
            object_offset,
            raw_data,
        })
    }

    fn read_dense_link_info_message<R: Read + Seek>(
        reader: &mut crate::io::reader::HdfReader<R>,
        oh_addr: u64,
    ) -> Result<LinkInfoMessage> {
        reader.seek(oh_addr)?;

        let mut first_bytes = [0u8; 4];
        reader.read_bytes_into(&mut first_bytes)?;
        if first_bytes != [b'O', b'H', b'D', b'R'] {
            return Err(Error::Unsupported(
                "link mutation currently supports only v2 object headers".into(),
            ));
        }

        let version = reader.read_u8()?;
        if version != 2 {
            return Err(Error::Unsupported(
                "link mutation currently supports only v2 object headers".into(),
            ));
        }

        let flags = reader.read_u8()?;
        if flags & !HDR_V2_KNOWN_FLAGS != 0 {
            return Err(Error::InvalidFormat(format!(
                "object header v2 flags contain reserved bits: {flags:#04x}"
            )));
        }
        if flags & HDR_STORE_TIMES != 0 {
            reader.skip(16)?;
        }
        if flags & HDR_ATTR_STORE_PHASE_CHANGE != 0 {
            reader.skip(4)?;
        }

        let chunk0_size_bytes = 1u8 << (flags & HDR_CHUNK0_SIZE_MASK);
        let chunk0_data_size = reader.read_uint(chunk0_size_bytes)?;
        let chunk0_data_start = reader.position()?;
        let chunk0_data_end = chunk0_data_start
            .checked_add(chunk0_data_size)
            .ok_or_else(|| Error::InvalidFormat("object-header chunk range overflow".into()))?;

        while reader.position()? < chunk0_data_end {
            let msg_header_pos = reader.position()?;
            if msg_header_pos
                .checked_add(4)
                .is_none_or(|end| end > chunk0_data_end)
            {
                break;
            }

            let msg_type = u16::from(reader.read_u8()?);
            let msg_size = usize::from(reader.read_u16()?);
            let _msg_flags = reader.read_u8()?;
            if flags & HDR_ATTR_CRT_ORDER_TRACKED != 0 {
                reader.skip(2)?;
            }

            let msg_data_offset = reader.position()?;
            let msg_size_u64 = Self::usize_to_u64(msg_size, "object-header message size")?;
            if msg_data_offset
                .checked_add(msg_size_u64)
                .is_none_or(|end| end > chunk0_data_end)
            {
                return Err(Error::InvalidFormat(
                    "object-header message payload exceeds chunk".into(),
                ));
            }

            if msg_type == object_header::MSG_LINK_INFO {
                let mut data = vec![0u8; msg_size];
                reader.read_bytes_into(&mut data)?;
                return LinkInfoMessage::decode(&data, reader.sizeof_addr());
            }
            reader.skip(msg_size_u64)?;
        }

        Err(Error::InvalidFormat(
            "dense link info message not found".into(),
        ))
    }

    fn read_dense_link_leaf_records_into<R: Read + Seek>(
        reader: &mut crate::io::reader::HdfReader<R>,
        btree: &BTreeV2Header,
        records: &mut Vec<u8>,
    ) -> Result<()> {
        let record_size = usize::from(btree.record_size);
        let record_count = usize::from(btree.root_nrecords);
        let records_len = record_size
            .checked_mul(record_count)
            .ok_or_else(|| Error::InvalidFormat("dense link leaf records overflow".into()))?;
        let check_len = 6usize
            .checked_add(records_len)
            .ok_or_else(|| Error::InvalidFormat("dense link leaf size overflow".into()))?;
        let total_len = check_len
            .checked_add(4)
            .ok_or_else(|| Error::InvalidFormat("dense link leaf size overflow".into()))?;

        records.clear();
        records.resize(total_len, 0);
        reader.seek(btree.root_addr)?;
        reader.read_bytes_into(records)?;
        if records.get(..4) != Some(&b"BTLF"[..]) {
            return Err(Error::InvalidFormat(
                "invalid dense link B-tree leaf magic".into(),
            ));
        }
        if records.get(4).copied() != Some(0) || records.get(5).copied() != Some(btree.tree_type) {
            return Err(Error::InvalidFormat(
                "dense link B-tree leaf header does not match index".into(),
            ));
        }
        let stored = read_u32_le_at(records, check_len, "dense link leaf checksum")?;
        let computed = checksum_metadata(&records[..check_len]);
        if stored != computed {
            return Err(Error::InvalidFormat(format!(
                "dense link leaf checksum mismatch: stored={stored:#010x}, computed={computed:#010x}"
            )));
        }
        records.drain(..6);
        records.truncate(records_len);
        Ok(())
    }

    fn rewrite_dense_link_name_index(&mut self, location: &DenseLinkLocation) -> Result<()> {
        let record_size = usize::from(location.btree.record_size);
        if record_size == 0 || location.leaf_records.len() % record_size != 0 {
            return Err(Error::InvalidFormat(
                "dense link records have inconsistent sizes".into(),
            ));
        }
        let record_count = location.leaf_records.len() / record_size;

        let mut leaf = Vec::with_capacity(
            6usize
                .checked_add(location.leaf_records.len())
                .and_then(|len| len.checked_add(4))
                .ok_or_else(|| Error::InvalidFormat("dense link leaf size overflow".into()))?,
        );
        leaf.extend_from_slice(b"BTLF");
        leaf.push(0);
        leaf.push(location.btree.tree_type);
        leaf.extend_from_slice(&location.leaf_records);
        let checksum = checksum_metadata(&leaf);
        leaf.extend_from_slice(&checksum.to_le_bytes());
        self.write_handle
            .seek(SeekFrom::Start(location.btree.root_addr))?;
        self.write_handle.write_all(&leaf)?;

        let record_count_u16 = Self::usize_to_u16(record_count, "dense link record count")?;
        let record_count_u64 = Self::usize_to_u64(record_count, "dense link record count")?;
        let sa = usize::from(self.superblock.sizeof_addr);
        let ss = usize::from(self.superblock.sizeof_size);
        let header_capacity = 22usize
            .checked_add(sa)
            .and_then(|len| len.checked_add(ss))
            .ok_or_else(|| Error::InvalidFormat("dense link B-tree header size overflow".into()))?;
        let mut header = Vec::with_capacity(header_capacity);
        header.extend_from_slice(b"BTHD");
        header.push(0);
        header.push(location.btree.tree_type);
        header.extend_from_slice(&location.btree.node_size.to_le_bytes());
        header.extend_from_slice(&location.btree.record_size.to_le_bytes());
        header.extend_from_slice(&0u16.to_le_bytes());
        header.push(location.btree.split_pct);
        header.push(location.btree.merge_pct);
        let mut scratch = [0u8; 8];
        header.extend_from_slice(Self::encode_uint_le_into(
            location.btree.root_addr,
            &mut scratch,
            sa,
            "dense link B-tree root address",
        )?);
        header.extend_from_slice(&record_count_u16.to_le_bytes());
        header.extend_from_slice(Self::encode_uint_le_into(
            record_count_u64,
            &mut scratch,
            ss,
            "dense link B-tree total record count",
        )?);
        let checksum = checksum_metadata(&header);
        header.extend_from_slice(&checksum.to_le_bytes());
        self.write_handle
            .seek(SeekFrom::Start(location.btree_header_addr))?;
        self.write_handle.write_all(&header)?;
        Ok(())
    }

    fn reposition_dense_link_record_by_hash(
        records: &mut [u8],
        record_index: usize,
        record_size: usize,
    ) -> Result<()> {
        if record_size == 0 || records.len() % record_size != 0 {
            return Err(Error::InvalidFormat(
                "dense link records have inconsistent sizes".into(),
            ));
        }
        let record_count = records.len() / record_size;
        if record_index >= record_count {
            return Err(Error::InvalidFormat(
                "dense link record index is invalid".into(),
            ));
        }
        let start = record_index
            .checked_mul(record_size)
            .ok_or_else(|| Error::InvalidFormat("dense link record offset overflow".into()))?;
        let end = start
            .checked_add(record_size)
            .ok_or_else(|| Error::InvalidFormat("dense link record offset overflow".into()))?;
        let changed_hash = read_u32_le_at(&records[start..end], 0, "dense link record hash")?;

        let mut insert_index = record_count - 1;
        let mut logical_idx = 0usize;
        for (idx, record) in records.chunks_exact(record_size).enumerate() {
            if idx == record_index {
                continue;
            }
            let hash = read_u32_le_at(record, 0, "dense link record hash")?;
            if changed_hash < hash {
                insert_index = logical_idx;
                break;
            }
            logical_idx += 1;
        }

        if insert_index == record_index {
            return Ok(());
        }
        let insert_pos = insert_index
            .checked_mul(record_size)
            .ok_or_else(|| Error::InvalidFormat("dense link record offset overflow".into()))?;
        if insert_index < record_index {
            records[insert_pos..end].rotate_right(record_size);
        } else {
            let rotate_end = insert_pos
                .checked_add(record_size)
                .ok_or_else(|| Error::InvalidFormat("dense link record offset overflow".into()))?;
            records[start..rotate_end].rotate_left(record_size);
        }
        Ok(())
    }

    fn rewrite_dense_link_direct_block_checksum(
        &mut self,
        heap: &FractalHeapHeader,
        patched_addr: u64,
        patched_data: &[u8],
    ) -> Result<()> {
        if !heap.has_checksum {
            return Ok(());
        }
        let block_size = usize::try_from(heap.start_block_size)
            .map_err(|_| Error::InvalidFormat("dense link direct block too large".into()))?;
        let checksum_pos = direct_block_checksum_pos(heap, self.superblock.sizeof_addr)?;
        let checksum_end = checksum_pos
            .checked_add(4)
            .ok_or_else(|| Error::InvalidFormat("direct block checksum offset overflow".into()))?;
        let mut guard = self.inner.lock();
        guard.reader.seek(heap.root_block_addr)?;
        let mut block = vec![0u8; block_size];
        guard.reader.read_bytes_into(&mut block)?;
        drop(guard);

        let patch_start = patched_addr
            .checked_sub(heap.root_block_addr)
            .ok_or_else(|| Error::InvalidFormat("direct block patch address underflow".into()))?;
        let patch_start = usize::try_from(patch_start)
            .map_err(|_| Error::InvalidFormat("direct block patch offset too large".into()))?;
        let patch_end = patch_start
            .checked_add(patched_data.len())
            .ok_or_else(|| Error::InvalidFormat("direct block patch range overflow".into()))?;
        block
            .get_mut(patch_start..patch_end)
            .ok_or_else(|| Error::InvalidFormat("direct block patch exceeds block".into()))?
            .copy_from_slice(patched_data);
        let checksum_window = block.get_mut(checksum_pos..checksum_end).ok_or_else(|| {
            Error::InvalidFormat("direct block checksum field is truncated".into())
        })?;
        checksum_window.fill(0);
        let checksum = checksum_metadata(&block);
        let checksum_pos_u64 = Self::usize_to_u64(checksum_pos, "direct block checksum position")?;
        self.write_handle.seek(SeekFrom::Start(
            heap.root_block_addr
                .checked_add(checksum_pos_u64)
                .ok_or_else(|| {
                    Error::InvalidFormat("direct block checksum address overflow".into())
                })?,
        ))?;
        self.write_handle.write_all(&checksum.to_le_bytes())?;
        Ok(())
    }

    fn decrement_object_refcount_if_present(&mut self, object_addr: u64) -> Result<bool> {
        let Some(location) = self.find_object_refcount_location(object_addr)? else {
            return Ok(false);
        };
        if location.refcount <= 1 {
            return Err(Error::Unsupported(
                "hard-link deletion would drop explicit object refcount below one".into(),
            ));
        }
        let new_refcount = location.refcount - 1;
        self.write_handle
            .seek(SeekFrom::Start(location.value_offset))?;
        self.write_handle.write_all(&new_refcount.to_le_bytes())?;
        if let (Some(oh_start), Some(oh_check_len)) = (location.oh_start, location.oh_check_len) {
            self.rewrite_oh_checksum(oh_start, oh_check_len)?;
        }
        Ok(true)
    }

    fn increment_object_refcount_if_present(&mut self, object_addr: u64) -> Result<bool> {
        let Some(location) = self.find_object_refcount_location(object_addr)? else {
            return Ok(false);
        };
        let new_refcount = location
            .refcount
            .checked_add(1)
            .ok_or_else(|| Error::InvalidFormat("object refcount overflow".into()))?;
        self.write_handle
            .seek(SeekFrom::Start(location.value_offset))?;
        self.write_handle.write_all(&new_refcount.to_le_bytes())?;
        if let (Some(oh_start), Some(oh_check_len)) = (location.oh_start, location.oh_check_len) {
            self.rewrite_oh_checksum(oh_start, oh_check_len)?;
        }
        Ok(true)
    }

    fn ensure_explicit_object_refcount(
        &mut self,
        object_addr: u64,
        minimum_refcount: u32,
    ) -> Result<u64> {
        if self.increment_object_refcount_if_present(object_addr)? {
            return Ok(object_addr);
        }

        let mut guard = self.inner.lock();
        let oh = ObjectHeader::read_at(&mut guard.reader, object_addr)?;
        if oh.version != 2 {
            return Err(Error::Unsupported(
                "creating persistent hard-link refcounts currently supports only v2 object headers"
                    .into(),
            ));
        }
        if oh.flags & HDR_ATTR_CRT_ORDER_TRACKED != 0 {
            return Err(Error::Unsupported(
                "creating persistent hard-link refcounts with object-header creation-order tracking is not implemented"
                    .into(),
            ));
        }

        let mut messages = Vec::new();
        for msg in &oh.messages {
            match msg.msg_type {
                object_header::MSG_HEADER_CONTINUATION | object_header::MSG_NIL => {}
                _ => messages.push(OwnedObjectHeaderMessage::from_raw(msg)),
            }
        }
        drop(guard);

        messages.push(OwnedObjectHeaderMessage {
            msg_type: object_header::MSG_OBJ_REF_COUNT,
            flags: 0,
            creation_index: None,
            data: minimum_refcount.to_le_bytes().to_vec(),
        });
        self.append_owned_v2_object_header(&messages, oh.flags)
    }

    fn find_object_refcount_location(
        &self,
        object_addr: u64,
    ) -> Result<Option<ObjectRefcountLocation>> {
        let mut guard = self.inner.lock();
        let reader = &mut guard.reader;
        reader.seek(object_addr)?;

        let mut first_bytes = [0u8; 4];
        reader.read_bytes_into(&mut first_bytes)?;
        if first_bytes != [b'O', b'H', b'D', b'R'] {
            if first_bytes[0] != 1 {
                return Err(Error::Unsupported(
                    "hard-link deletion can update only v1/v2 object refcounts".into(),
                ));
            }
            let value_offset = object_addr.checked_add(4).ok_or_else(|| {
                Error::InvalidFormat("object-header refcount offset overflow".into())
            })?;
            reader.seek(value_offset)?;
            let refcount = reader.read_u32()?;
            return Ok(Some(ObjectRefcountLocation {
                value_offset,
                oh_start: None,
                oh_check_len: None,
                refcount,
            }));
        }

        let version = reader.read_u8()?;
        if version != 2 {
            return Err(Error::Unsupported(
                "hard-link deletion can update only v1/v2 object refcounts".into(),
            ));
        }

        let flags = reader.read_u8()?;
        if flags & !HDR_V2_KNOWN_FLAGS != 0 {
            return Err(Error::InvalidFormat(format!(
                "object header v2 flags contain reserved bits: {flags:#04x}"
            )));
        }
        if flags & HDR_STORE_TIMES != 0 {
            reader.skip(16)?;
        }
        if flags & HDR_ATTR_STORE_PHASE_CHANGE != 0 {
            reader.skip(4)?;
        }

        let chunk0_size_bytes = 1u8 << (flags & HDR_CHUNK0_SIZE_MASK);
        let chunk0_data_size = reader.read_uint(chunk0_size_bytes)?;
        let chunk0_data_start = reader.position()?;
        let chunk0_data_end = chunk0_data_start
            .checked_add(chunk0_data_size)
            .ok_or_else(|| Error::InvalidFormat("object-header chunk range overflow".into()))?;
        let oh_check_len = usize::try_from(chunk0_data_end - object_addr)
            .map_err(|_| Error::InvalidFormat("object-header checksum range overflow".into()))?;

        while reader.position()? < chunk0_data_end {
            let msg_header_pos = reader.position()?;
            if msg_header_pos
                .checked_add(4)
                .is_none_or(|end| end > chunk0_data_end)
            {
                break;
            }

            let msg_type = u16::from(reader.read_u8()?);
            let msg_size = usize::from(reader.read_u16()?);
            let _msg_flags = reader.read_u8()?;
            if flags & HDR_ATTR_CRT_ORDER_TRACKED != 0 {
                reader.skip(2)?;
            }

            let msg_data_offset = reader.position()?;
            let msg_size_u64 = Self::usize_to_u64(msg_size, "object-header message size")?;
            if msg_data_offset
                .checked_add(msg_size_u64)
                .is_none_or(|end| end > chunk0_data_end)
            {
                return Err(Error::InvalidFormat(
                    "object-header message payload exceeds chunk".into(),
                ));
            }

            if msg_type == object_header::MSG_OBJ_REF_COUNT {
                if msg_size < 4 {
                    return Err(Error::InvalidFormat(
                        "object refcount message is truncated".into(),
                    ));
                }
                let refcount = reader.read_u32()?;
                return Ok(Some(ObjectRefcountLocation {
                    value_offset: msg_data_offset,
                    oh_start: Some(object_addr),
                    oh_check_len: Some(oh_check_len),
                    refcount,
                }));
            }
            reader.skip(msg_size_u64)?;
        }

        Ok(None)
    }

    fn count_compact_hard_links_to_addr(&self, oh_addr: u64, target_addr: u64) -> Result<usize> {
        let mut guard = self.inner.lock();
        let sizeof_addr = guard.superblock.sizeof_addr;
        let reader = &mut guard.reader;
        reader.seek(oh_addr)?;

        let mut first_bytes = [0u8; 4];
        reader.read_bytes_into(&mut first_bytes)?;
        if first_bytes != [b'O', b'H', b'D', b'R'] {
            return Err(Error::Unsupported(
                "hard-link deletion without explicit refcount supports only v2 compact parent groups"
                    .into(),
            ));
        }

        let version = reader.read_u8()?;
        if version != 2 {
            return Err(Error::Unsupported(
                "hard-link deletion without explicit refcount supports only v2 compact parent groups"
                    .into(),
            ));
        }

        let flags = reader.read_u8()?;
        if flags & !HDR_V2_KNOWN_FLAGS != 0 {
            return Err(Error::InvalidFormat(format!(
                "object header v2 flags contain reserved bits: {flags:#04x}"
            )));
        }
        if flags & HDR_STORE_TIMES != 0 {
            reader.skip(16)?;
        }
        if flags & HDR_ATTR_STORE_PHASE_CHANGE != 0 {
            reader.skip(4)?;
        }

        let chunk0_size_bytes = 1u8 << (flags & HDR_CHUNK0_SIZE_MASK);
        let chunk0_data_size = reader.read_uint(chunk0_size_bytes)?;
        let chunk0_data_start = reader.position()?;
        let chunk0_data_end = chunk0_data_start
            .checked_add(chunk0_data_size)
            .ok_or_else(|| Error::InvalidFormat("object-header chunk range overflow".into()))?;

        let mut count = 0usize;
        let mut msg_buf = Vec::new();
        while reader.position()? < chunk0_data_end {
            let msg_header_pos = reader.position()?;
            if msg_header_pos
                .checked_add(4)
                .is_none_or(|end| end > chunk0_data_end)
            {
                break;
            }

            let msg_type = u16::from(reader.read_u8()?);
            let msg_size = usize::from(reader.read_u16()?);
            let _msg_flags = reader.read_u8()?;
            if flags & HDR_ATTR_CRT_ORDER_TRACKED != 0 {
                reader.skip(2)?;
            }

            let msg_data_offset = reader.position()?;
            let msg_size_u64 = Self::usize_to_u64(msg_size, "object-header message size")?;
            if msg_data_offset
                .checked_add(msg_size_u64)
                .is_none_or(|end| end > chunk0_data_end)
            {
                return Err(Error::InvalidFormat(
                    "object-header message payload exceeds chunk".into(),
                ));
            }

            if msg_type == object_header::MSG_LINK {
                read_message_into(reader, &mut msg_buf, msg_size)?;
                let link = compact_link_view(&msg_buf, sizeof_addr)?;
                if link.link_type == LinkType::Hard && link.hard_link_addr == Some(target_addr) {
                    count += 1;
                }
            } else if msg_type == object_header::MSG_LINK_INFO {
                read_message_into(reader, &mut msg_buf, msg_size)?;
                let link_info = LinkInfoMessage::decode(&msg_buf, sizeof_addr)?;
                if link_info.has_dense_storage() || link_info.corder_btree_addr.is_some() {
                    return Err(Error::Unsupported(
                        "hard-link deletion without explicit refcount does not support dense parent links"
                            .into(),
                    ));
                }
            } else if msg_type == object_header::MSG_SYMBOL_TABLE {
                return Err(Error::Unsupported(
                    "hard-link deletion without explicit refcount does not support v1 symbol-table parent groups"
                        .into(),
                ));
            } else {
                reader.skip(msg_size_u64)?;
            }
        }

        Ok(count)
    }

    fn compact_messages_for_append(
        &self,
        group_addr: u64,
        new_name: &str,
        operation: &str,
    ) -> Result<(u8, Vec<OwnedObjectHeaderMessage>)> {
        let mut guard = self.inner.lock();
        let sizeof_addr = guard.superblock.sizeof_addr;
        let oh = ObjectHeader::read_at(&mut guard.reader, group_addr)?;
        if oh.version != 2 {
            return Err(Error::Unsupported(format!(
                "{operation} currently supports only v2 object headers"
            )));
        }
        if oh.flags & HDR_ATTR_CRT_ORDER_TRACKED != 0 {
            return Err(Error::Unsupported(format!(
                "{operation} with object-header creation-order tracking is not implemented"
            )));
        }

        let mut messages = Vec::new();
        for msg in &oh.messages {
            match msg.msg_type {
                object_header::MSG_LINK => {
                    let link = LinkMessage::decode(&msg.data, sizeof_addr)?;
                    if link.name == new_name {
                        return Err(Error::InvalidFormat(format!(
                            "link '{new_name}' already exists"
                        )));
                    }
                    messages.push(OwnedObjectHeaderMessage::from_raw(msg));
                }
                object_header::MSG_LINK_INFO => {
                    let link_info = LinkInfoMessage::decode(&msg.data, sizeof_addr)?;
                    if link_info.has_dense_storage() || link_info.corder_btree_addr.is_some() {
                        return Err(Error::Unsupported(
                            format!("{operation} for dense or creation-order indexed links is not implemented"),
                        ));
                    }
                    messages.push(OwnedObjectHeaderMessage::from_raw(msg));
                }
                object_header::MSG_SYMBOL_TABLE => {
                    return Err(Error::Unsupported(format!(
                        "{operation} for v1 symbol-table groups is not implemented"
                    )));
                }
                object_header::MSG_HEADER_CONTINUATION | object_header::MSG_NIL => {}
                _ => messages.push(OwnedObjectHeaderMessage::from_raw(msg)),
            }
        }

        Ok((oh.flags, messages))
    }

    fn messages_for_append(
        &self,
        group_addr: u64,
        new_name: &str,
        operation: &str,
    ) -> Result<(u8, Vec<OwnedObjectHeaderMessage>)> {
        match self.compact_messages_for_append(group_addr, new_name, operation) {
            Err(Error::Unsupported(msg))
                if msg.contains("dense or creation-order indexed links") =>
            {
                self.compact_messages_from_dense_links_for_append(group_addr, new_name, operation)
            }
            other => other,
        }
    }

    fn compact_messages_from_dense_links_for_append(
        &self,
        group_addr: u64,
        new_name: &str,
        operation: &str,
    ) -> Result<(u8, Vec<OwnedObjectHeaderMessage>)> {
        let mut guard = self.inner.lock();
        let sizeof_addr = guard.superblock.sizeof_addr;
        let oh = ObjectHeader::read_at(&mut guard.reader, group_addr)?;
        if oh.version != 2 {
            return Err(Error::Unsupported(format!(
                "{operation} currently supports only v2 object headers"
            )));
        }
        if oh.flags & HDR_ATTR_CRT_ORDER_TRACKED != 0 {
            return Err(Error::Unsupported(format!(
                "{operation} with object-header creation-order tracking is not implemented"
            )));
        }

        let mut messages = Vec::new();
        for msg in &oh.messages {
            match msg.msg_type {
                object_header::MSG_LINK => {
                    let link = LinkMessage::decode(&msg.data, sizeof_addr)?;
                    if link.name == new_name {
                        return Err(Error::InvalidFormat(format!(
                            "link '{new_name}' already exists"
                        )));
                    }
                    messages.push(OwnedObjectHeaderMessage::from_raw(msg));
                }
                object_header::MSG_LINK_INFO => {
                    let link_info = LinkInfoMessage::decode(&msg.data, sizeof_addr)?;
                    if link_info.corder_btree_addr.is_some() {
                        return Err(Error::Unsupported(format!(
                            "{operation} for creation-order indexed links is not implemented"
                        )));
                    }
                    if !link_info.has_dense_storage() {
                        messages.push(OwnedObjectHeaderMessage::from_raw(msg));
                        continue;
                    }

                    let heap =
                        FractalHeapHeader::read_at(&mut guard.reader, link_info.fractal_heap_addr)?;
                    if heap.io_filter_len != 0 || heap.current_root_rows != 0 {
                        return Err(Error::Unsupported(
                            "mutating filtered or indirect dense link heaps is not implemented"
                                .into(),
                        ));
                    }
                    let btree =
                        BTreeV2Header::read_at(&mut guard.reader, link_info.name_btree_addr)?;
                    if btree.depth != 0 {
                        return Err(Error::Unsupported(
                            "mutating non-leaf dense link name indexes is not implemented".into(),
                        ));
                    }
                    let mut leaf_records = Vec::new();
                    Self::read_dense_link_leaf_records_into(
                        &mut guard.reader,
                        &btree,
                        &mut leaf_records,
                    )?;
                    let heap_id_len = usize::from(heap.heap_id_len);
                    let record_size = usize::from(btree.record_size);
                    for record in leaf_records.chunks_exact(record_size) {
                        let heap_id = checked_window(record, 4, heap_id_len, "dense link heap ID")?;
                        let raw_data = heap.read_managed_object(&mut guard.reader, heap_id)?;
                        let link = LinkMessage::decode(&raw_data, sizeof_addr)?;
                        if link.name == new_name {
                            return Err(Error::InvalidFormat(format!(
                                "link '{new_name}' already exists"
                            )));
                        }
                        messages.push(OwnedObjectHeaderMessage {
                            msg_type: object_header::MSG_LINK,
                            flags: 0,
                            creation_index: None,
                            data: raw_data,
                        });
                    }
                }
                object_header::MSG_SYMBOL_TABLE => {
                    return Err(Error::Unsupported(format!(
                        "{operation} for v1 symbol-table groups is not implemented"
                    )));
                }
                object_header::MSG_HEADER_CONTINUATION | object_header::MSG_NIL => {}
                _ => messages.push(OwnedObjectHeaderMessage::from_raw(msg)),
            }
        }

        Ok((oh.flags, messages))
    }

    fn compact_or_dense_hard_link_parent_chain(
        &self,
        parent_path: &str,
        operation: &str,
    ) -> Result<Vec<ParentChainLink>> {
        let names = compact_absolute_path_components(parent_path, operation)?;
        let mut ancestor_addr = self.superblock.root_addr;
        let mut chain = Vec::with_capacity(names.len());
        for name in names {
            let (child_addr, storage) =
                self.compact_or_dense_hard_link_addr(ancestor_addr, name, operation)?;
            chain.push(ParentChainLink {
                ancestor_addr,
                link_name: name.to_string(),
                child_addr,
                storage,
            });
            ancestor_addr = child_addr;
        }
        Ok(chain)
    }

    fn compact_or_dense_hard_link_addr(
        &self,
        group_addr: u64,
        name: &str,
        operation: &str,
    ) -> Result<(u64, ParentChainLinkStorage)> {
        match self.compact_hard_link_addr(group_addr, name, operation) {
            Ok(addr) => Ok((addr, ParentChainLinkStorage::Compact)),
            Err(Error::Unsupported(msg))
                if msg.contains("dense or creation-order indexed links") =>
            {
                let location = self.find_dense_link_location(group_addr, name, None)?;
                let link = compact_link_view(&location.raw_data, self.superblock.sizeof_addr)?;
                if link.link_type != LinkType::Hard {
                    return Err(Error::Unsupported(format!(
                        "{operation} under a nested group requires hard-link ancestors"
                    )));
                }
                let addr = link.hard_link_addr.ok_or_else(|| {
                    Error::InvalidFormat(format!("hard link '{name}' is missing object address"))
                })?;
                Ok((addr, ParentChainLinkStorage::Dense))
            }
            Err(err) => Err(err),
        }
    }

    fn compact_hard_link_addr(&self, group_addr: u64, name: &str, operation: &str) -> Result<u64> {
        let mut guard = self.inner.lock();
        let sizeof_addr = guard.superblock.sizeof_addr;
        let oh = ObjectHeader::read_at(&mut guard.reader, group_addr)?;
        if oh.version != 2 {
            return Err(Error::Unsupported(format!(
                "{operation} under a nested group currently supports only v2 compact object headers"
            )));
        }
        if oh.flags & HDR_ATTR_CRT_ORDER_TRACKED != 0 {
            return Err(Error::Unsupported(format!(
                "{operation} under a nested group with creation-order tracking is not implemented"
            )));
        }

        for msg in &oh.messages {
            match msg.msg_type {
                object_header::MSG_LINK => {
                    let link = LinkMessage::decode(&msg.data, sizeof_addr)?;
                    if link.name == name {
                        if link.link_type != LinkType::Hard {
                            return Err(Error::Unsupported(format!(
                                "{operation} under a nested group requires compact hard-link ancestors"
                            )));
                        }
                        return link.hard_link_addr.ok_or_else(|| {
                            Error::InvalidFormat(format!(
                                "hard link '{name}' is missing object address"
                            ))
                        });
                    }
                }
                object_header::MSG_LINK_INFO => {
                    let link_info = LinkInfoMessage::decode(&msg.data, sizeof_addr)?;
                    if link_info.has_dense_storage() || link_info.corder_btree_addr.is_some() {
                        return Err(Error::Unsupported(format!(
                            "{operation} under a nested group with dense or creation-order indexed links is not implemented"
                        )));
                    }
                }
                object_header::MSG_SYMBOL_TABLE => {
                    return Err(Error::Unsupported(format!(
                        "{operation} under a nested group with v1 symbol-table links is not implemented"
                    )));
                }
                _ => {}
            }
        }
        Err(Error::InvalidFormat(format!("link '{name}' not found")))
    }

    fn compact_messages_replacing_hard_link(
        &self,
        group_addr: u64,
        target_name: &str,
        new_addr: u64,
        operation: &str,
    ) -> Result<(u8, Vec<OwnedObjectHeaderMessage>)> {
        let mut guard = self.inner.lock();
        let sizeof_addr = guard.superblock.sizeof_addr;
        let oh = ObjectHeader::read_at(&mut guard.reader, group_addr)?;
        if oh.version != 2 {
            return Err(Error::Unsupported(format!(
                "{operation} address propagation currently supports only v2 compact object headers"
            )));
        }
        if oh.flags & HDR_ATTR_CRT_ORDER_TRACKED != 0 {
            return Err(Error::Unsupported(format!(
                "{operation} address propagation with creation-order tracking is not implemented"
            )));
        }

        let mut replaced = false;
        let mut messages = Vec::new();
        for msg in &oh.messages {
            match msg.msg_type {
                object_header::MSG_LINK => {
                    let link = LinkMessage::decode(&msg.data, sizeof_addr)?;
                    if link.name == target_name {
                        if replaced {
                            return Err(Error::InvalidFormat(format!(
                                "duplicate root link '{target_name}'"
                            )));
                        }
                        if link.link_type != LinkType::Hard {
                            return Err(Error::Unsupported(format!(
                                "{operation} address propagation requires compact hard-link ancestors"
                            )));
                        }
                        messages.push(OwnedObjectHeaderMessage {
                            msg_type: object_header::MSG_LINK,
                            flags: msg.flags,
                            creation_index: msg.creation_index,
                            data: encode_hard_link_message(
                                target_name,
                                new_addr,
                                self.superblock.sizeof_addr,
                            )?,
                        });
                        replaced = true;
                    } else {
                        messages.push(OwnedObjectHeaderMessage::from_raw(msg));
                    }
                }
                object_header::MSG_LINK_INFO => {
                    let link_info = LinkInfoMessage::decode(&msg.data, sizeof_addr)?;
                    if link_info.has_dense_storage() || link_info.corder_btree_addr.is_some() {
                        return Err(Error::Unsupported(format!(
                            "{operation} address propagation for dense or creation-order indexed links is not implemented"
                        )));
                    }
                    messages.push(OwnedObjectHeaderMessage::from_raw(msg));
                }
                object_header::MSG_SYMBOL_TABLE => {
                    return Err(Error::Unsupported(format!(
                        "{operation} address propagation for v1 symbol-table links is not implemented"
                    )));
                }
                object_header::MSG_HEADER_CONTINUATION | object_header::MSG_NIL => {}
                _ => messages.push(OwnedObjectHeaderMessage::from_raw(msg)),
            }
        }
        if !replaced {
            return Err(Error::InvalidFormat(format!(
                "link '{target_name}' not found"
            )));
        }
        Ok((oh.flags, messages))
    }

    fn compact_messages_replacing_hard_links(
        &self,
        group_addr: u64,
        replacements: &[(String, u64)],
        operation: &str,
    ) -> Result<(u8, Vec<OwnedObjectHeaderMessage>)> {
        let mut guard = self.inner.lock();
        let sizeof_addr = guard.superblock.sizeof_addr;
        let oh = ObjectHeader::read_at(&mut guard.reader, group_addr)?;
        if oh.version != 2 {
            return Err(Error::Unsupported(format!(
                "{operation} address propagation currently supports only v2 compact object headers"
            )));
        }
        if oh.flags & HDR_ATTR_CRT_ORDER_TRACKED != 0 {
            return Err(Error::Unsupported(format!(
                "{operation} address propagation with creation-order tracking is not implemented"
            )));
        }

        let messages: Vec<_> = oh
            .messages
            .iter()
            .map(OwnedObjectHeaderMessage::from_raw)
            .collect();
        replace_hard_links_in_messages(messages, oh.flags, replacements, sizeof_addr, operation)
    }

    fn compact_messages_replacing_and_appending_hard_link(
        &self,
        group_addr: u64,
        target_name: &str,
        target_addr: u64,
        new_name: &str,
        operation: &str,
    ) -> Result<(u8, Vec<OwnedObjectHeaderMessage>)> {
        let mut guard = self.inner.lock();
        let sizeof_addr = guard.superblock.sizeof_addr;
        let oh = ObjectHeader::read_at(&mut guard.reader, group_addr)?;
        if oh.version != 2 {
            return Err(Error::Unsupported(format!(
                "{operation} currently supports only v2 compact object headers"
            )));
        }
        if oh.flags & HDR_ATTR_CRT_ORDER_TRACKED != 0 {
            return Err(Error::Unsupported(format!(
                "{operation} with object-header creation-order tracking is not implemented"
            )));
        }

        let mut replaced = false;
        let mut messages = Vec::new();
        for msg in &oh.messages {
            match msg.msg_type {
                object_header::MSG_LINK => {
                    let link = LinkMessage::decode(&msg.data, sizeof_addr)?;
                    if link.name == new_name {
                        return Err(Error::InvalidFormat(format!(
                            "link '{new_name}' already exists"
                        )));
                    }
                    if link.name == target_name {
                        if replaced {
                            return Err(Error::InvalidFormat(format!(
                                "link '{target_name}' appears more than once"
                            )));
                        }
                        if link.link_type != LinkType::Hard {
                            return Err(Error::Unsupported(format!(
                                "{operation} requires a compact hard-link target"
                            )));
                        }
                        messages.push(OwnedObjectHeaderMessage {
                            msg_type: object_header::MSG_LINK,
                            flags: msg.flags,
                            creation_index: msg.creation_index,
                            data: encode_hard_link_message(
                                target_name,
                                target_addr,
                                self.superblock.sizeof_addr,
                            )?,
                        });
                        replaced = true;
                    } else {
                        messages.push(OwnedObjectHeaderMessage::from_raw(msg));
                    }
                }
                object_header::MSG_LINK_INFO => {
                    let link_info = LinkInfoMessage::decode(&msg.data, sizeof_addr)?;
                    if link_info.has_dense_storage() || link_info.corder_btree_addr.is_some() {
                        return Err(Error::Unsupported(format!(
                            "{operation} for dense or creation-order indexed links is not implemented"
                        )));
                    }
                    messages.push(OwnedObjectHeaderMessage::from_raw(msg));
                }
                object_header::MSG_SYMBOL_TABLE => {
                    return Err(Error::Unsupported(format!(
                        "{operation} for v1 symbol-table groups is not implemented"
                    )));
                }
                object_header::MSG_HEADER_CONTINUATION | object_header::MSG_NIL => {}
                _ => messages.push(OwnedObjectHeaderMessage::from_raw(msg)),
            }
        }
        if !replaced {
            return Err(Error::InvalidFormat(format!(
                "link '{target_name}' not found"
            )));
        }
        messages.push(OwnedObjectHeaderMessage {
            msg_type: object_header::MSG_LINK,
            flags: 0,
            creation_index: None,
            data: encode_hard_link_message(new_name, target_addr, self.superblock.sizeof_addr)?,
        });
        Ok((oh.flags, messages))
    }

    fn append_v2_object_header(
        &mut self,
        messages: &[ObjectHeaderMessageRef<'_>],
        flags: u8,
    ) -> Result<u64> {
        let encoded = object_header::encode_v2_with_continuations(
            messages,
            flags,
            &[],
            OBJECT_HEADER_CHUNK_DATA_LIMIT,
            OBJECT_HEADER_CHUNK_DATA_LIMIT,
            self.superblock.sizeof_addr,
            self.superblock.sizeof_size,
        )?;
        if !encoded.continuation_chunks.is_empty() {
            return Err(Error::Unsupported(
                "root group creation cannot append continuation object-header chunks yet".into(),
            ));
        }
        let addr = self.append_aligned_zeros(encoded.prefix.len(), 8)?;
        self.write_handle.seek(SeekFrom::Start(addr))?;
        self.write_handle.write_all(&encoded.prefix)?;
        Ok(addr)
    }

    fn append_owned_v2_object_header(
        &mut self,
        messages: &[OwnedObjectHeaderMessage],
        flags: u8,
    ) -> Result<u64> {
        let refs: Vec<_> = messages
            .iter()
            .map(OwnedObjectHeaderMessage::as_ref)
            .collect();
        self.append_v2_object_header(&refs, flags)
    }

    fn rewrite_v2_v3_superblock(&mut self, root_addr: u64, eof_addr: u64) -> Result<()> {
        let sb = Superblock {
            root_addr,
            root_entry_obj_header_addr: root_addr,
            eof_addr,
            ..self.superblock.clone()
        };
        let mut sb_bytes = Vec::new();
        sb.write_v2(&mut sb_bytes)?;
        self.write_handle.seek(SeekFrom::Start(0))?;
        self.write_handle.write_all(&sb_bytes)?;
        self.superblock = sb;
        Ok(())
    }
}

#[derive(Debug)]
struct CompactLinkView<'a> {
    name: &'a str,
    name_offset: usize,
    name_size: usize,
    link_type: LinkType,
    hard_link_addr: Option<u64>,
    hard_link_addr_offset: Option<usize>,
}

#[derive(Debug)]
struct OwnedObjectHeaderMessage {
    msg_type: u16,
    flags: u8,
    creation_index: Option<u16>,
    data: Vec<u8>,
}

impl OwnedObjectHeaderMessage {
    fn from_raw(message: &object_header::RawMessage) -> Self {
        Self {
            msg_type: message.msg_type,
            flags: message.flags,
            creation_index: message.creation_index,
            data: message.data.clone(),
        }
    }

    fn as_ref(&self) -> ObjectHeaderMessageRef<'_> {
        ObjectHeaderMessageRef {
            msg_type: self.msg_type,
            flags: self.flags,
            creation_index: self.creation_index,
            data: &self.data,
        }
    }
}

fn validate_direct_child_link_name(name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(Error::InvalidFormat("link name cannot be empty".into()));
    }
    if name == "." || name == ".." || name.contains('/') {
        return Err(Error::Unsupported(
            "existing-file link creation currently supports only a direct child name".into(),
        ));
    }
    Ok(())
}

fn compact_absolute_path_components<'a>(path: &'a str, operation: &str) -> Result<Vec<&'a str>> {
    let rest = path.strip_prefix('/').ok_or_else(|| {
        Error::Unsupported(format!(
            "{operation} currently supports only absolute compact parent paths"
        ))
    })?;
    if rest.is_empty() {
        return Err(Error::Unsupported(format!(
            "{operation} currently supports only non-root paths here"
        )));
    }
    let names: Vec<_> = rest.split('/').collect();
    if names
        .iter()
        .any(|name| name.is_empty() || *name == "." || *name == "..")
    {
        return Err(Error::Unsupported(format!(
            "{operation} currently supports only normalized compact parent paths"
        )));
    }
    Ok(names)
}

fn split_normalized_absolute_child_path<'a>(
    path: &'a str,
    operation: &str,
) -> Result<(&'a str, &'a str)> {
    if path == "/" || !path.starts_with('/') {
        return Err(Error::Unsupported(format!(
            "{operation} currently supports only absolute compact target paths"
        )));
    }
    if path.contains("/.") || path.contains("//") {
        return Err(Error::Unsupported(format!(
            "{operation} currently supports only normalized compact target paths"
        )));
    }
    let (parent, name) = path.rsplit_once('/').ok_or_else(|| {
        Error::InvalidFormat(format!(
            "{operation} target path '{path}' is not a child path"
        ))
    })?;
    if name.is_empty() {
        return Err(Error::InvalidFormat(format!(
            "{operation} target path cannot end with '/'"
        )));
    }
    Ok((if parent.is_empty() { "/" } else { parent }, name))
}

fn owned_compact_absolute_path_components(path: &str, operation: &str) -> Result<Vec<String>> {
    let rest = path.strip_prefix('/').ok_or_else(|| {
        Error::Unsupported(format!(
            "{operation} currently supports only absolute compact parent paths"
        ))
    })?;
    if rest.is_empty() {
        return Ok(Vec::new());
    }
    let names: Vec<_> = rest.split('/').collect();
    if names
        .iter()
        .any(|name| name.is_empty() || *name == "." || *name == "..")
    {
        return Err(Error::Unsupported(format!(
            "{operation} currently supports only normalized compact parent paths"
        )));
    }
    Ok(names.into_iter().map(str::to_string).collect())
}

fn replace_hard_links_in_messages(
    messages: Vec<OwnedObjectHeaderMessage>,
    flags: u8,
    replacements: &[(String, u64)],
    sizeof_addr: u8,
    operation: &str,
) -> Result<(u8, Vec<OwnedObjectHeaderMessage>)> {
    let mut replaced = vec![false; replacements.len()];
    let mut out = Vec::with_capacity(messages.len());

    for msg in messages {
        match msg.msg_type {
            object_header::MSG_LINK => {
                let link = LinkMessage::decode(&msg.data, sizeof_addr)?;
                if let Some((idx, (_, new_addr))) = replacements
                    .iter()
                    .enumerate()
                    .find(|(_, (name, _))| link.name == *name)
                {
                    if replaced[idx] {
                        return Err(Error::InvalidFormat(format!(
                            "duplicate root link '{}'",
                            link.name
                        )));
                    }
                    if link.link_type != LinkType::Hard {
                        return Err(Error::Unsupported(format!(
                            "{operation} address propagation requires compact hard-link ancestors"
                        )));
                    }
                    out.push(OwnedObjectHeaderMessage {
                        msg_type: object_header::MSG_LINK,
                        flags: msg.flags,
                        creation_index: msg.creation_index,
                        data: encode_hard_link_message(&link.name, *new_addr, sizeof_addr)?,
                    });
                    replaced[idx] = true;
                } else {
                    out.push(msg);
                }
            }
            object_header::MSG_LINK_INFO => {
                let link_info = LinkInfoMessage::decode(&msg.data, sizeof_addr)?;
                if link_info.has_dense_storage() || link_info.corder_btree_addr.is_some() {
                    return Err(Error::Unsupported(format!(
                        "{operation} address propagation for dense or creation-order indexed links is not implemented"
                    )));
                }
                out.push(msg);
            }
            object_header::MSG_SYMBOL_TABLE => {
                return Err(Error::Unsupported(format!(
                    "{operation} address propagation for v1 symbol-table links is not implemented"
                )));
            }
            object_header::MSG_HEADER_CONTINUATION | object_header::MSG_NIL => {}
            _ => out.push(msg),
        }
    }

    if let Some((idx, (name, _))) = replacements
        .iter()
        .enumerate()
        .find(|(idx, _)| !replaced[*idx])
    {
        return Err(Error::InvalidFormat(format!(
            "link '{name}' not found for replacement {idx}"
        )));
    }
    Ok((flags, out))
}

fn encode_hard_link_message(name: &str, target_addr: u64, sizeof_addr: u8) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    let name_bytes = name.as_bytes();
    let size_flag = link_name_size_flag(name_bytes.len())?;
    buf.push(1);
    buf.push(size_flag | 0x10);
    buf.push(1);
    encode_link_name_len(&mut buf, name_bytes.len(), size_flag)?;
    buf.extend_from_slice(name_bytes);
    append_le_uint_width(
        &mut buf,
        target_addr,
        usize::from(sizeof_addr),
        "hard link address",
    )?;
    Ok(buf)
}

fn encode_soft_link_message(name: &str, target_path: &str) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    let name_bytes = name.as_bytes();
    let target_bytes = target_path.as_bytes();
    let target_len = u16::try_from(target_bytes.len()).map_err(|_| {
        Error::InvalidFormat(format!(
            "soft link target is {} bytes, maximum is {}",
            target_bytes.len(),
            u16::MAX
        ))
    })?;
    let size_flag = link_name_size_flag(name_bytes.len())?;

    buf.push(1);
    buf.push(size_flag | 0x08 | 0x10);
    buf.push(1);
    buf.push(1);
    encode_link_name_len(&mut buf, name_bytes.len(), size_flag)?;
    buf.extend_from_slice(name_bytes);
    buf.extend_from_slice(&target_len.to_le_bytes());
    buf.extend_from_slice(target_bytes);
    Ok(buf)
}

fn encode_external_link_message(name: &str, filename: &str, object_path: &str) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    let name_bytes = name.as_bytes();
    let size_flag = link_name_size_flag(name_bytes.len())?;

    let info_len = 1usize
        .checked_add(filename.len())
        .and_then(|len| len.checked_add(1))
        .and_then(|len| len.checked_add(object_path.len()))
        .and_then(|len| len.checked_add(1))
        .ok_or_else(|| Error::InvalidFormat("external link info length overflow".into()))?;
    let info_len = u16::try_from(info_len).map_err(|_| {
        Error::InvalidFormat(format!(
            "external link info is {info_len} bytes, maximum is {}",
            u16::MAX
        ))
    })?;

    buf.push(1);
    buf.push(size_flag | 0x08 | 0x10);
    buf.push(64);
    buf.push(1);
    encode_link_name_len(&mut buf, name_bytes.len(), size_flag)?;
    buf.extend_from_slice(name_bytes);
    buf.extend_from_slice(&info_len.to_le_bytes());
    buf.push(0);
    buf.extend_from_slice(filename.as_bytes());
    buf.push(0);
    buf.extend_from_slice(object_path.as_bytes());
    buf.push(0);
    Ok(buf)
}

fn encode_renamed_link_message(
    link: &LinkMessage,
    new_name: &str,
    sizeof_addr: u8,
) -> Result<Vec<u8>> {
    match link.link_type {
        LinkType::Hard => {
            let target_addr = link.hard_link_addr.ok_or_else(|| {
                Error::InvalidFormat("hard link message is missing target address".into())
            })?;
            encode_hard_link_message(new_name, target_addr, sizeof_addr)
        }
        LinkType::Soft => {
            let target = link.soft_link_target.as_deref().ok_or_else(|| {
                Error::InvalidFormat("soft link message is missing target path".into())
            })?;
            encode_soft_link_message(new_name, target)
        }
        LinkType::External => {
            let (filename, object_path) = link.external_link.as_ref().ok_or_else(|| {
                Error::InvalidFormat("external link message is missing target".into())
            })?;
            encode_external_link_message(new_name, filename, object_path)
        }
        LinkType::UserDefined(kind) => Err(Error::Unsupported(format!(
            "root compact relink for user-defined link type {kind} is not implemented"
        ))),
    }
}

fn link_name_size_flag(len: usize) -> Result<u8> {
    match len {
        0 => Err(Error::InvalidFormat("link name cannot be empty".into())),
        1..=0xff => Ok(0),
        0x100..=0xffff => Ok(1),
        0x1_0000..=0xffff_ffff => Ok(2),
        _ => Ok(3),
    }
}

fn encode_link_name_len(out: &mut Vec<u8>, len: usize, size_flag: u8) -> Result<()> {
    let width = 1usize << (size_flag & 0x03);
    let len = u64::try_from(len)
        .map_err(|_| Error::InvalidFormat("link name length exceeds u64".into()))?;
    append_le_uint_width(out, len, width, "link name length")
}

fn append_le_uint_width(out: &mut Vec<u8>, value: u64, width: usize, context: &str) -> Result<()> {
    if !(1..=8).contains(&width) {
        return Err(Error::InvalidFormat(format!("{context} width is invalid")));
    }
    if width < 8 {
        let bits = width
            .checked_mul(8)
            .ok_or_else(|| Error::InvalidFormat(format!("{context} width overflow")))?;
        if value >= (1u64 << bits) {
            return Err(Error::InvalidFormat(format!(
                "{context} does not fit in {width} bytes"
            )));
        }
    }
    out.extend_from_slice(&value.to_le_bytes()[..width]);
    Ok(())
}

fn patch_le_uint_width(
    raw: &mut [u8],
    pos: usize,
    value: u64,
    width: usize,
    context: &str,
) -> Result<()> {
    if !(1..=8).contains(&width) {
        return Err(Error::InvalidFormat(format!("{context} width is invalid")));
    }
    if width < 8 {
        let bits = width
            .checked_mul(8)
            .ok_or_else(|| Error::InvalidFormat(format!("{context} width overflow")))?;
        if value >= (1u64 << bits) {
            return Err(Error::InvalidFormat(format!(
                "{context} does not fit in {width} bytes"
            )));
        }
    }
    let end = pos
        .checked_add(width)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))?;
    let window = raw
        .get_mut(pos..end)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} field is truncated")))?;
    window.copy_from_slice(&value.to_le_bytes()[..width]);
    Ok(())
}

fn encode_dataspace_for_spec_into(out: &mut Vec<u8>, spec: &DatasetSpec<'_>) -> Result<()> {
    if spec.shape.len() > MAX_DATASPACE_RANK {
        return Err(Error::InvalidFormat(format!(
            "dataspace rank {} exceeds supported maximum {MAX_DATASPACE_RANK}",
            spec.shape.len(),
        )));
    }
    if let Some(max_shape) = spec.max_shape {
        if max_shape.len() != spec.shape.len() {
            return Err(Error::InvalidFormat(format!(
                "max shape rank {} does not match dataset rank {}",
                max_shape.len(),
                spec.shape.len()
            )));
        }
        for (idx, (&dim, &max_dim)) in spec.shape.iter().zip(max_shape.iter()).enumerate() {
            if max_dim != u64::MAX && dim > max_dim {
                return Err(Error::InvalidFormat(format!(
                    "dataset dimension {idx} size {dim} exceeds max dimension {max_dim}"
                )));
            }
        }
    }

    if spec.shape.is_empty() {
        out.extend_from_slice(&[2, 0, 0, 0]);
    } else {
        let ndims = u8::try_from(spec.shape.len())
            .map_err(|_| Error::InvalidFormat("dataspace rank exceeds u8".into()))?;
        out.push(2);
        out.push(ndims);
        out.push(if spec.max_shape.is_some() { 0x01 } else { 0 });
        out.push(1);
        for &dim in spec.shape {
            out.extend_from_slice(&dim.to_le_bytes());
        }
        if let Some(max_shape) = spec.max_shape {
            for &dim in max_shape {
                out.extend_from_slice(&dim.to_le_bytes());
            }
        }
    }
    Ok(())
}

fn encode_contiguous_layout_into(
    out: &mut Vec<u8>,
    data_addr: u64,
    data_size: u64,
    sizeof_addr: u8,
    sizeof_size: u8,
) -> Result<()> {
    out.push(3);
    out.push(1);
    append_encoded_addr(out, data_addr, sizeof_addr)?;
    append_le_uint_width(
        out,
        data_size,
        usize::from(sizeof_size),
        "dataset data size",
    )
}

fn encode_fill_value_message_into(
    out: &mut Vec<u8>,
    fill: Option<FillValueSpec<'_>>,
) -> Result<()> {
    let Some(fill) = fill else {
        out.extend_from_slice(&[3, 0x09]);
        return Ok(());
    };
    if fill.alloc_time > 3 {
        return Err(Error::InvalidFormat(format!(
            "fill allocation time {} exceeds 2-bit field",
            fill.alloc_time
        )));
    }
    if fill.fill_time > 3 {
        return Err(Error::InvalidFormat(format!(
            "fill write time {} exceeds 2-bit field",
            fill.fill_time
        )));
    }

    let mut flags = fill.alloc_time | (fill.fill_time << 2);
    out.push(3);
    if let Some(value) = fill.value {
        flags |= 0x20;
        out.push(flags);
        out.extend_from_slice(
            &u32::try_from(value.len())
                .map_err(|_| Error::InvalidFormat("fill-value payload length exceeds u32".into()))?
                .to_le_bytes(),
        );
        out.extend_from_slice(value);
    } else {
        flags |= 0x10;
        out.push(flags);
    }
    Ok(())
}

fn append_encoded_addr(out: &mut Vec<u8>, value: u64, sizeof_addr: u8) -> Result<()> {
    let width = usize::from(sizeof_addr);
    if !(1..=8).contains(&width) {
        return Err(Error::InvalidFormat(format!(
            "address field width {sizeof_addr} is invalid"
        )));
    }
    if value == crate::io::reader::UNDEF_ADDR {
        out.extend(std::iter::repeat_n(0xff, width));
        return Ok(());
    }
    append_le_uint_width(out, value, width, "address value")
}

fn validate_dataset_data_len(spec: &DatasetSpec<'_>) -> Result<()> {
    let dtype_size = usize::try_from(spec.dtype.size())
        .map_err(|_| Error::InvalidFormat("dataset datatype size exceeds usize".into()))?;
    if dtype_size == 0 {
        return Err(Error::InvalidFormat(
            "dataset datatype size must be nonzero".into(),
        ));
    }
    let expected_count = shape_element_count(spec.shape)?;
    let expected_bytes = usize::try_from(expected_count)
        .map_err(|_| Error::InvalidFormat("dataset element count exceeds usize".into()))?
        .checked_mul(dtype_size)
        .ok_or_else(|| Error::InvalidFormat("dataset byte size overflow".into()))?;
    if expected_bytes != spec.data.len() {
        return Err(Error::InvalidFormat(format!(
            "dataset byte length {} does not match shape element count {expected_count} * datatype size {dtype_size}",
            spec.data.len()
        )));
    }
    Ok(())
}

fn shape_element_count(shape: &[u64]) -> Result<u64> {
    if shape.is_empty() {
        return Ok(1);
    }
    shape.iter().try_fold(1u64, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| Error::InvalidFormat("dataset shape element count overflow".into()))
    })
}

fn read_message_into<R>(
    reader: &mut HdfReader<R>,
    scratch: &mut Vec<u8>,
    msg_size: usize,
) -> Result<()>
where
    R: Read + Seek,
{
    scratch.clear();
    if scratch.capacity() < msg_size {
        scratch.reserve_exact(msg_size - scratch.capacity());
    }
    scratch.resize(msg_size, 0);
    reader.read_bytes_into(scratch)
}

fn compact_link_view(raw: &[u8], sizeof_addr: u8) -> Result<CompactLinkView<'_>> {
    let mut pos = 0usize;
    let version = read_u8(raw, &mut pos, "link message version")?;
    if version != 1 {
        return Err(Error::InvalidFormat(format!(
            "link message version {version}"
        )));
    }
    let flags = read_u8(raw, &mut pos, "link message flags")?;
    if flags & !0x1f != 0 {
        return Err(Error::InvalidFormat(format!(
            "link message flags {flags:#x} are invalid"
        )));
    }
    let name_len_size = 1usize << (flags & 0x03);
    let link_type = if flags & 0x08 != 0 {
        match read_u8(raw, &mut pos, "link message link type")? {
            0 => LinkType::Hard,
            1 => LinkType::Soft,
            64 => LinkType::External,
            65..=u8::MAX => LinkType::UserDefined(raw[pos - 1]),
            other => return Err(Error::InvalidFormat(format!("invalid link type {other}"))),
        }
    } else {
        LinkType::Hard
    };
    if flags & 0x04 != 0 {
        advance_pos(raw, &mut pos, 8, "link creation order")?;
    }
    let char_encoding = if flags & 0x10 != 0 {
        read_u8(raw, &mut pos, "link message character encoding")?
    } else {
        0
    };
    if char_encoding > 1 {
        return Err(Error::InvalidFormat(format!(
            "invalid link character encoding {char_encoding}"
        )));
    }
    let name_size = usize::try_from(read_le_u64(
        raw,
        &mut pos,
        name_len_size,
        "link name length",
    )?)
    .map_err(|_| Error::InvalidFormat("link name length overflows usize".into()))?;
    if name_size == 0 {
        return Err(Error::InvalidFormat("invalid link name length".into()));
    }
    ensure_available(raw, pos, name_size, "link name")?;
    let name_offset = pos;
    let name = str::from_utf8(&raw[name_offset..name_offset + name_size])
        .map_err(|_| Error::InvalidFormat("link name is not valid UTF-8".into()))?;
    advance_pos(raw, &mut pos, name_size, "link name")?;

    let mut hard_link_addr_offset = None;
    let hard_link_addr = match link_type {
        LinkType::Hard => {
            hard_link_addr_offset = Some(pos);
            Some(read_le_u64(
                raw,
                &mut pos,
                usize::from(sizeof_addr),
                "hard link address",
            )?)
        }
        LinkType::Soft => {
            let target_len =
                usize::try_from(read_le_u64(raw, &mut pos, 2, "soft link target length")?)
                    .map_err(|_| Error::InvalidFormat("soft link target length overflow".into()))?;
            if target_len == 0 {
                return Err(Error::InvalidFormat("invalid soft link length".into()));
            }
            ensure_available(raw, pos, target_len, "soft link target")?;
            str::from_utf8(&raw[pos..pos + target_len])
                .map_err(|_| Error::InvalidFormat("soft link target is not valid UTF-8".into()))?;
            advance_pos(raw, &mut pos, target_len, "soft link target")?;
            None
        }
        LinkType::External => {
            let info_len =
                usize::try_from(read_le_u64(raw, &mut pos, 2, "external link info length")?)
                    .map_err(|_| {
                        Error::InvalidFormat("external link info length overflow".into())
                    })?;
            if info_len < 3 {
                return Err(Error::InvalidFormat(
                    "external link info is too short".into(),
                ));
            }
            validate_external_link_info(&raw[pos..], info_len)?;
            advance_pos(raw, &mut pos, info_len, "external link info")?;
            None
        }
        LinkType::UserDefined(_) => None,
    };

    Ok(CompactLinkView {
        name,
        name_offset,
        name_size,
        link_type,
        hard_link_addr,
        hard_link_addr_offset,
    })
}

fn encode_link_name_in_place<'a>(
    raw: &'a mut [u8],
    name_offset: usize,
    name_size: usize,
    name: &str,
) -> Result<&'a [u8]> {
    let name_field = raw
        .get_mut(
            name_offset
                ..name_offset.checked_add(name_size).ok_or_else(|| {
                    Error::InvalidFormat("link name field offset overflow".into())
                })?,
        )
        .ok_or_else(|| Error::InvalidFormat("link name field exceeds message".into()))?;
    if name.len() != name_size {
        return Err(Error::Unsupported(
            "in-place link rename cannot grow or shrink the encoded name field".into(),
        ));
    }
    name_field.copy_from_slice(name.as_bytes());
    Ok(name_field)
}

fn validate_external_link_info(raw: &[u8], info_len: usize) -> Result<()> {
    ensure_available(raw, 0, info_len, "external link info")?;
    let info = &raw[..info_len];
    if info[0] != 0 {
        return Err(Error::InvalidFormat(format!(
            "external link version {}",
            info[0]
        )));
    }
    let rest = &info[1..];
    let first_nul = rest.iter().position(|&byte| byte == 0).ok_or_else(|| {
        Error::InvalidFormat("external link filename is missing terminator".into())
    })?;
    let filename = &rest[..first_nul];
    let obj_path = &rest[first_nul + 1..];
    if obj_path.last() != Some(&0) {
        return Err(Error::InvalidFormat(
            "external link object path is missing terminator".into(),
        ));
    }
    str::from_utf8(filename)
        .map_err(|_| Error::InvalidFormat("external link filename is not valid UTF-8".into()))?;
    str::from_utf8(&obj_path[..obj_path.len() - 1])
        .map_err(|_| Error::InvalidFormat("external link object path is not valid UTF-8".into()))?;
    Ok(())
}

fn checked_window<'a>(data: &'a [u8], pos: usize, len: usize, context: &str) -> Result<&'a [u8]> {
    let end = pos
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} length overflow")))?;
    data.get(pos..end)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))
}

fn read_u32_le_at(raw: &[u8], pos: usize, context: &str) -> Result<u32> {
    let bytes = checked_window(raw, pos, 4, context)?;
    let bytes: [u8; 4] = bytes
        .try_into()
        .map_err(|_| Error::InvalidFormat(format!("{context} is truncated")))?;
    Ok(u32::from_le_bytes(bytes))
}

fn managed_heap_object_offset(heap: &FractalHeapHeader, heap_id: &[u8]) -> Result<u64> {
    match heap_id.first().copied() {
        Some(id) if ((id >> 4) & 0x03) == 0 => {}
        Some(id) => {
            return Err(Error::Unsupported(format!(
                "dense link heap ID type {} cannot be rewritten in place",
                (id >> 4) & 0x03
            )));
        }
        None => {
            return Err(Error::InvalidFormat(
                "dense link heap ID is truncated".into(),
            ))
        }
    }

    let offset_bytes = fractal_heap_offset_width(heap)?;
    let offset_bytes = checked_window(heap_id, 1, offset_bytes, "dense link heap offset")?;
    let mut offset = 0u64;
    for (idx, byte) in offset_bytes.iter().enumerate() {
        offset |= u64::from(*byte) << (idx * 8);
    }
    Ok(offset)
}

fn direct_block_checksum_pos(heap: &FractalHeapHeader, sizeof_addr: u8) -> Result<usize> {
    let offset_bytes = fractal_heap_offset_width(heap)?;
    5usize
        .checked_add(usize::from(sizeof_addr))
        .and_then(|pos| pos.checked_add(offset_bytes))
        .ok_or_else(|| Error::InvalidFormat("direct block checksum position overflow".into()))
}

fn fractal_heap_offset_width(heap: &FractalHeapHeader) -> Result<usize> {
    let max_heap_size = usize::from(heap.max_heap_size);
    let offset_bytes = max_heap_size
        .checked_add(7)
        .ok_or_else(|| Error::InvalidFormat("dense link heap offset width overflow".into()))?
        / 8;
    if offset_bytes == 0 || offset_bytes > 8 {
        return Err(Error::Unsupported(format!(
            "dense link heap offset width {offset_bytes} is unsupported"
        )));
    }
    Ok(offset_bytes)
}

fn dense_link_name_hash(name: &str) -> u32 {
    crate::format::checksum::checksum_lookup3(name.as_bytes(), 0)
}

fn ensure_available(data: &[u8], pos: usize, len: usize, context: &str) -> Result<()> {
    let end = pos
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} length overflow")))?;
    if end > data.len() {
        return Err(Error::InvalidFormat(format!("{context} is truncated")));
    }
    Ok(())
}

fn advance_pos(data: &[u8], pos: &mut usize, len: usize, context: &str) -> Result<()> {
    ensure_available(data, *pos, len, context)?;
    *pos = pos
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} position overflow")))?;
    Ok(())
}

fn read_u8(data: &[u8], pos: &mut usize, context: &str) -> Result<u8> {
    ensure_available(data, *pos, 1, context)?;
    let value = data[*pos];
    advance_pos(data, pos, 1, context)?;
    Ok(value)
}

fn read_le_u64(data: &[u8], pos: &mut usize, size: usize, context: &str) -> Result<u64> {
    if !(1..=8).contains(&size) {
        return Err(Error::InvalidFormat(format!(
            "{context} has invalid byte width {size}"
        )));
    }
    ensure_available(data, *pos, size, context)?;
    let mut val = 0u64;
    for (idx, byte) in data[*pos..*pos + size].iter().enumerate() {
        val |= u64::from(*byte) << (idx * 8);
    }
    advance_pos(data, pos, size, context)?;
    Ok(val)
}
