use std::fs;
use std::io::BufReader;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::error::{Error, Result};
use crate::format::btree_v2;
use crate::format::fractal_heap::FractalHeapHeader;
use crate::format::messages::link::{LinkMessage, LinkType};
use crate::format::messages::link_info::LinkInfoMessage;
use crate::format::messages::symbol_table::SymbolTableMessage;
use crate::format::object_header::{self, ObjectHeader};
use crate::hl::dataset::Dataset;
use crate::hl::file::{
    collect_v1_group_members, collect_v2_link_members, register_open_object,
    unregister_open_object, File, FileInner, ObjectType, OpenObjectKind,
};

/// An HDF5 group.
pub struct Group {
    inner: Arc<Mutex<FileInner<BufReader<fs::File>>>>,
    name: String,
    addr: u64,
    object_id: u64,
}

/// Basic link metadata, mirroring the read-side subset of `H5Lget_info2`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinkInfo {
    pub link_type: LinkType,
    pub creation_order_valid: bool,
    pub creation_order: u64,
    pub char_encoding: u8,
    pub hard_link_addr: Option<u64>,
}

/// Decoded link value for soft and external links.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinkValue {
    Soft(String),
    External {
        filename: String,
        object_path: String,
    },
}

/// Basic object-header metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectInfo {
    pub addr: u64,
    pub header_version: u8,
    pub refcount: u32,
    pub message_count: usize,
    pub atime: Option<u32>,
    pub mtime: Option<u32>,
    pub ctime: Option<u32>,
    pub btime: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupInfo {
    pub nlinks: usize,
    pub mounted: bool,
}

impl Group {
    pub(crate) fn open(
        inner: Arc<Mutex<FileInner<BufReader<fs::File>>>>,
        name: &str,
        addr: u64,
    ) -> Result<Self> {
        let object_id = register_open_object(&inner, OpenObjectKind::Group);
        Ok(Self {
            inner,
            name: name.to_string(),
            addr,
            object_id,
        })
    }

    /// Get the group name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the object header address.
    pub fn addr(&self) -> u64 {
        self.addr
    }

    /// Return this group handle's high-level object id.
    pub fn object_id(&self) -> u64 {
        self.object_id
    }

    /// List all member names in this group.
    pub fn member_names(&self) -> Result<Vec<String>> {
        let members = self.members()?;
        Ok(members.into_iter().map(|(name, _)| name).collect())
    }

    /// List all links in this group.
    ///
    /// v1 symbol-table groups do not store full v2 link messages, so their
    /// members are returned as synthesized hard-link records.
    pub fn links(&self) -> Result<Vec<LinkMessage>> {
        let mut guard = self.inner.lock();
        let sizeof_addr = guard.superblock.sizeof_addr;
        let oh = ObjectHeader::read_at(&mut guard.reader, self.addr)?;

        for msg in &oh.messages {
            if msg.msg_type == object_header::MSG_SYMBOL_TABLE {
                let stab = SymbolTableMessage::decode(&msg.data, sizeof_addr)?;
                let members = collect_v1_group_members(
                    &mut guard.reader,
                    stab.btree_addr,
                    stab.name_heap_addr,
                )?;
                return Ok(members
                    .into_iter()
                    .map(|(name, addr)| LinkMessage {
                        name,
                        link_type: LinkType::Hard,
                        creation_order: None,
                        char_encoding: 0,
                        hard_link_addr: Some(addr),
                        soft_link_target: None,
                        external_link: None,
                    })
                    .collect());
            }
        }

        let mut links = Vec::new();
        for msg in &oh.messages {
            if msg.msg_type == object_header::MSG_LINK {
                links.push(LinkMessage::decode(&msg.data, sizeof_addr)?);
            }
        }
        if !links.is_empty() {
            return Ok(links);
        }

        for msg in &oh.messages {
            if msg.msg_type == object_header::MSG_LINK_INFO {
                let link_info = LinkInfoMessage::decode(&msg.data, sizeof_addr)?;
                if link_info.has_dense_storage() {
                    return Self::read_dense_link_messages(
                        &mut guard.reader,
                        &link_info,
                        sizeof_addr,
                    );
                }
            }
        }

        Ok(Vec::new())
    }

    /// List all links sorted by tracked creation order.
    pub fn links_by_creation_order(&self) -> Result<Vec<LinkMessage>> {
        let mut links = self.links()?;
        if links.is_empty() {
            return Ok(links);
        }
        if links.iter().any(|link| link.creation_order.is_none()) {
            return Err(Error::Unsupported(format!(
                "group '{}' does not track link creation order",
                self.name
            )));
        }
        links.sort_by_key(|link| link.creation_order.unwrap_or(u64::MAX));
        Ok(links)
    }

    /// List all members as (name, object_header_addr) pairs.
    pub fn members(&self) -> Result<Vec<(String, u64)>> {
        let mut guard = self.inner.lock();
        let sizeof_addr = guard.superblock.sizeof_addr;
        let oh = ObjectHeader::read_at(&mut guard.reader, self.addr)?;

        // Check for v1 symbol table message
        for msg in &oh.messages {
            if msg.msg_type == object_header::MSG_SYMBOL_TABLE {
                let stab = SymbolTableMessage::decode(&msg.data, sizeof_addr)?;
                return collect_v1_group_members(
                    &mut guard.reader,
                    stab.btree_addr,
                    stab.name_heap_addr,
                );
            }
        }

        // V2: collect from link messages
        let members = collect_v2_link_members(&oh.messages, sizeof_addr);
        if !members.is_empty() {
            return Ok(members);
        }

        // V2 dense storage: link info message with fractal heap + v2 B-tree
        for msg in &oh.messages {
            if msg.msg_type == object_header::MSG_LINK_INFO {
                let link_info = LinkInfoMessage::decode(&msg.data, sizeof_addr)?;
                if link_info.has_dense_storage() {
                    return Self::read_dense_links(&mut guard.reader, &link_info, sizeof_addr);
                }
            }
        }

        Ok(Vec::new())
    }

    /// Read dense links from fractal heap + v2 B-tree as full LinkMessage objects.
    fn read_dense_link_messages<R: std::io::Read + std::io::Seek>(
        reader: &mut crate::io::reader::HdfReader<R>,
        link_info: &LinkInfoMessage,
        sizeof_addr: u8,
    ) -> Result<Vec<LinkMessage>> {
        let heap = FractalHeapHeader::read_at(reader, link_info.fractal_heap_addr)?;
        let records = btree_v2::collect_all_records(reader, link_info.name_btree_addr)?;
        let mut links = Vec::new();

        for record in &records {
            let Some(heap_id) = dense_link_heap_id(record, usize::from(heap.heap_id_len))? else {
                continue;
            };
            match heap.read_managed_object(reader, heap_id) {
                Ok(link_data) => match LinkMessage::decode(&link_data, sizeof_addr) {
                    Ok(link) => links.push(link),
                    Err(_) => {}
                },
                Err(_) => {}
            }
        }

        Ok(links)
    }

    /// Read dense links as (name, addr) pairs for member listing.
    fn read_dense_links<R: std::io::Read + std::io::Seek>(
        reader: &mut crate::io::reader::HdfReader<R>,
        link_info: &LinkInfoMessage,
        sizeof_addr: u8,
    ) -> Result<Vec<(String, u64)>> {
        let links = Self::read_dense_link_messages(reader, link_info, sizeof_addr)?;
        Ok(links
            .into_iter()
            .map(|l| {
                let addr = l.hard_link_addr.unwrap_or(0);
                (l.name, addr)
            })
            .collect())
    }

    /// Find a specific link by name, checking both inline messages and dense storage.
    pub(crate) fn find_link_by_name(&self, name: &str) -> Result<LinkMessage> {
        let mut guard = self.inner.lock();
        let sizeof_addr = guard.superblock.sizeof_addr;
        let oh = ObjectHeader::read_at(&mut guard.reader, self.addr)?;

        // Check inline link messages
        for msg in &oh.messages {
            if msg.msg_type == object_header::MSG_LINK {
                if let Ok(link) = LinkMessage::decode(&msg.data, sizeof_addr) {
                    if link.name == name {
                        return Ok(link);
                    }
                }
            }
        }

        // Check dense storage
        for msg in &oh.messages {
            if msg.msg_type == object_header::MSG_LINK_INFO {
                let link_info = LinkInfoMessage::decode(&msg.data, sizeof_addr)?;
                if link_info.has_dense_storage() {
                    let links =
                        Self::read_dense_link_messages(&mut guard.reader, &link_info, sizeof_addr)?;
                    if let Some(link) = links.into_iter().find(|l| l.name == name) {
                        return Ok(link);
                    }
                }
            }
        }

        Err(Error::InvalidFormat(format!("link '{name}' not found")))
    }

    /// Open a sub-group by name.
    pub fn open_group(&self, name: &str) -> Result<Group> {
        File::from_inner(self.inner.clone()).group(&group_child_path(&self.name, name))
    }

    /// Get the number of members in this group.
    pub fn len(&self) -> Result<usize> {
        Ok(self.members()?.len())
    }

    pub fn create_plist(&self) -> crate::hl::plist::object_create::ObjectCreate {
        crate::hl::plist::object_create::ObjectCreate::default()
    }

    pub fn info(&self) -> Result<GroupInfo> {
        Ok(GroupInfo {
            nlinks: self.len()?,
            mounted: false,
        })
    }

    pub fn info_async(&self) -> Result<GroupInfo> {
        self.info()
    }

    pub fn linkval(&self, name: &str) -> Result<Option<String>> {
        let link = self.find_link_by_name(name)?;
        Ok(link.soft_link_target.or_else(|| {
            link.external_link
                .map(|(file, path)| format!("{file}:{path}"))
        }))
    }

    pub fn comment(&self, name: &str) -> Result<Option<String>> {
        if name.is_empty() || name == "." {
            self.object_comment()
        } else {
            self.object_comment_by_name(name)
        }
    }

    pub fn num_objs(&self) -> Result<usize> {
        self.len()
    }

    pub fn objinfo(&self, name: &str) -> Result<ObjectInfo> {
        let addr = self.hard_link_addr_by_name(name)?;
        self.object_info_at(addr)
    }

    pub fn objname_by_idx(&self, index: usize) -> Result<String> {
        self.link_name_by_idx(index)
    }

    pub fn objtype_by_idx(&self, index: usize) -> Result<ObjectType> {
        let name = self.link_name_by_idx(index)?;
        self.member_type(&name)
    }

    /// Check if the group is empty.
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Get the type of a member object.
    pub fn member_type(&self, name: &str) -> Result<ObjectType> {
        File::from_inner(self.inner.clone())
            .object_type_for_path(&group_child_path(&self.name, name))
    }

    /// List attribute names.
    pub fn attr_names(&self) -> Result<Vec<String>> {
        crate::hl::attribute::attr_names(&self.inner, self.addr)
    }

    /// List attributes.
    pub fn attrs(&self) -> Result<Vec<crate::hl::attribute::Attribute>> {
        crate::hl::attribute::collect_attributes(&self.inner, self.addr)
    }

    /// List attributes sorted by tracked creation order.
    pub fn attrs_by_creation_order(&self) -> Result<Vec<crate::hl::attribute::Attribute>> {
        crate::hl::attribute::collect_attributes_by_creation_order(&self.inner, self.addr)
    }

    /// Get an attribute by name.
    pub fn attr(&self, name: &str) -> Result<crate::hl::attribute::Attribute> {
        crate::hl::attribute::get_attr(&self.inner, self.addr, name)
    }

    /// Check whether an attribute exists on this group.
    pub fn attr_exists(&self, name: &str) -> Result<bool> {
        crate::hl::attribute::attr_exists(&self.inner, self.addr, name)
    }

    /// Get the link type of a member by name.
    pub fn link_type(&self, name: &str) -> Result<LinkType> {
        let link = self.find_link_by_name(name)?;
        Ok(link.link_type)
    }

    /// Get link metadata by name.
    pub fn link_info(&self, name: &str) -> Result<LinkInfo> {
        link_info_from_message(&self.find_link_by_name(name)?)
    }

    /// Get legacy v1-style link metadata by name.
    pub fn link_info_v1(&self, name: &str) -> Result<LinkInfo> {
        self.link_info(name)
    }

    /// Get link metadata by zero-based storage-order index.
    pub fn link_info_by_idx(&self, index: usize) -> Result<LinkInfo> {
        let links = self.links()?;
        let link = links
            .get(index)
            .ok_or_else(|| Error::InvalidFormat(format!("link index {index} is out of bounds")))?;
        link_info_from_message(link)
    }

    /// Get legacy v1-style link metadata by zero-based storage-order index.
    pub fn link_info_by_idx_v1(&self, index: usize) -> Result<LinkInfo> {
        self.link_info_by_idx(index)
    }

    /// Get a link name by zero-based storage-order index.
    pub fn link_name_by_idx(&self, index: usize) -> Result<String> {
        self.links()?
            .get(index)
            .map(|link| link.name.clone())
            .ok_or_else(|| Error::InvalidFormat(format!("link index {index} is out of bounds")))
    }

    /// Get a soft or external link value by zero-based storage-order index.
    pub fn link_value_by_idx(&self, index: usize) -> Result<Option<LinkValue>> {
        let links = self.links()?;
        let link = links
            .get(index)
            .ok_or_else(|| Error::InvalidFormat(format!("link index {index} is out of bounds")))?;
        Ok(link_value_from_message(link))
    }

    /// Get the target path of a soft link.
    pub fn soft_link_target(&self, name: &str) -> Result<String> {
        let link = self.find_link_by_name(name)?;
        link.soft_link_target
            .ok_or_else(|| Error::InvalidFormat(format!("'{name}' is not a soft link")))
    }

    /// Get the target (filename, object_path) of an external link.
    pub fn external_link_target(&self, name: &str) -> Result<(String, String)> {
        let link = self.find_link_by_name(name)?;
        link.external_link
            .ok_or_else(|| Error::InvalidFormat(format!("'{name}' is not an external link")))
    }

    /// Get this group's object comment, if present.
    pub fn object_comment(&self) -> Result<Option<String>> {
        self.object_comment_at(self.addr)
    }

    /// Get a child object's comment by link name, if present.
    pub fn object_comment_by_name(&self, name: &str) -> Result<Option<String>> {
        let addr = self.hard_link_addr_by_name(name)?;
        self.object_comment_at(addr)
    }

    /// Get native object-header metadata for this group.
    pub fn native_info(&self) -> Result<ObjectInfo> {
        self.object_info_at(self.addr)
    }

    /// Get legacy v1-style child object metadata by zero-based link index.
    pub fn object_info_by_idx_v1(&self, index: usize) -> Result<ObjectInfo> {
        self.object_info_by_idx(index)
    }

    /// Get v2-style child object metadata by zero-based link index.
    pub fn object_info_by_idx_v2(&self, index: usize) -> Result<ObjectInfo> {
        self.object_info_by_idx(index)
    }

    /// Get v3-style child object metadata by zero-based link index.
    pub fn object_info_by_idx(&self, index: usize) -> Result<ObjectInfo> {
        let links = self.links()?;
        let link = links
            .get(index)
            .ok_or_else(|| Error::InvalidFormat(format!("link index {index} is out of bounds")))?;
        let addr = link.hard_link_addr.ok_or_else(|| {
            Error::InvalidFormat(format!(
                "link '{}' does not reference an object header",
                link.name
            ))
        })?;
        self.object_info_at(addr)
    }

    /// Get native object-header metadata by zero-based link index.
    pub fn native_info_by_idx(&self, index: usize) -> Result<ObjectInfo> {
        self.object_info_by_idx(index)
    }

    /// Check if a named member (link) exists in this group.
    pub fn link_exists(&self, name: &str) -> Result<bool> {
        crate::hl::location::link_exists(self, name)
    }

    /// Open a dataset by name.
    pub fn open_dataset(&self, name: &str) -> Result<Dataset> {
        File::from_inner(self.inner.clone()).dataset(&group_child_path(&self.name, name))
    }

    fn hard_link_addr_by_name(&self, name: &str) -> Result<u64> {
        let link = self.find_link_by_name(name)?;
        link.hard_link_addr.ok_or_else(|| {
            Error::InvalidFormat(format!("link '{name}' does not reference an object header"))
        })
    }

    fn object_info_at(&self, addr: u64) -> Result<ObjectInfo> {
        let mut guard = self.inner.lock();
        let oh = ObjectHeader::read_at(&mut guard.reader, addr)?;
        Ok(object_info_from_header(addr, &oh))
    }

    fn object_comment_at(&self, addr: u64) -> Result<Option<String>> {
        let mut guard = self.inner.lock();
        let oh = ObjectHeader::read_at(&mut guard.reader, addr)?;
        object_comment_from_header(&oh)
    }
}

impl Drop for Group {
    fn drop(&mut self) {
        unregister_open_object(&self.inner, self.object_id);
    }
}

pub(crate) fn link_info_from_message(link: &LinkMessage) -> Result<LinkInfo> {
    Ok(LinkInfo {
        link_type: link.link_type,
        creation_order_valid: link.creation_order.is_some(),
        creation_order: link.creation_order.unwrap_or(0),
        char_encoding: link.char_encoding,
        hard_link_addr: link.hard_link_addr,
    })
}

fn link_value_from_message(link: &LinkMessage) -> Option<LinkValue> {
    match link.link_type {
        LinkType::Soft => link.soft_link_target.clone().map(LinkValue::Soft),
        LinkType::External => {
            link.external_link
                .clone()
                .map(|(filename, object_path)| LinkValue::External {
                    filename,
                    object_path,
                })
        }
        _ => None,
    }
}

fn object_info_from_header(addr: u64, oh: &ObjectHeader) -> ObjectInfo {
    ObjectInfo {
        addr,
        header_version: oh.version,
        refcount: oh.refcount,
        message_count: oh.messages.len(),
        atime: oh.atime,
        mtime: oh.mtime,
        ctime: oh.ctime,
        btime: oh.btime,
    }
}

fn object_comment_from_header(oh: &ObjectHeader) -> Result<Option<String>> {
    oh.messages
        .iter()
        .find(|msg| msg.msg_type == object_header::MSG_OBJ_COMMENT)
        .map(|msg| {
            std::str::from_utf8(&msg.data)
                .map(|text| text.trim_end_matches('\0').to_string())
                .map_err(|_| Error::InvalidFormat("object comment is not UTF-8".into()))
        })
        .transpose()
}

fn dense_link_heap_id(record: &[u8], heap_id_len: usize) -> Result<Option<&[u8]>> {
    let end = 4usize
        .checked_add(heap_id_len)
        .ok_or_else(|| Error::InvalidFormat("dense link heap ID length overflow".into()))?;
    Ok(record.get(4..end))
}

fn group_child_path(parent: &str, child: &str) -> String {
    if child.starts_with('/') {
        child.to_string()
    } else if parent == "/" {
        format!("/{child}")
    } else {
        format!("{parent}/{child}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_link_heap_id_rejects_length_overflow() {
        let err = dense_link_heap_id(&[], usize::MAX).unwrap_err();
        assert!(
            err.to_string()
                .contains("dense link heap ID length overflow"),
            "unexpected error: {err}"
        );
    }
}
