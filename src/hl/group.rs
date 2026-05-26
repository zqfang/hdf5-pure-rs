use std::cmp::Ordering;
use std::fs;
use std::io::BufReader;
use std::marker::PhantomData;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::engine::writer::DatasetSpec;
use crate::error::{Error, Result};
use crate::format::btree_v1::{BTreeType, BTreeV1Node};
use crate::format::btree_v2;
use crate::format::checksum::checksum_lookup3;
use crate::format::fractal_heap::{FractalHeapHeader, FractalHeapManagedObjectCache};
use crate::format::local_heap::LocalHeap;
use crate::format::messages::attribute::AttributeMessage;
use crate::format::messages::attribute_info::AttributeInfoMessage;
use crate::format::messages::datatype::DatatypeMessage;
use crate::format::messages::link::{LinkMessage, LinkType};
use crate::format::messages::link_info::LinkInfoMessage;
use crate::format::messages::symbol_table::SymbolTableMessage;
use crate::format::object_header::{self, ObjectHeader};
use crate::format::superblock::Superblock;
use crate::format::symbol_table::SymbolTableNode;
use crate::hl::attribute::Attribute;
use crate::hl::dataset::Dataset;
use crate::hl::dataset_builder::dtype_for_type;
use crate::hl::datatype::Datatype;
use crate::hl::file::{
    object_type_from_messages, register_open_object, unregister_open_object, File, FileInner,
    FileIntent, ObjectType, OpenObjectKind,
};
use crate::hl::link::{get_val_cb_borrowed, LinkValueRef};
use crate::hl::mutable_file::MutableFile;
use crate::hl::types::{slice_as_bytes, H5Type};
use crate::io::reader::HdfReader;

pub(crate) struct LinkMessageRef<'a> {
    pub name: &'a str,
    pub link_type: LinkType,
    pub creation_order: Option<u64>,
    pub char_encoding: u8,
    pub hard_link_addr: Option<u64>,
    pub soft_link_target: Option<&'a str>,
    pub external_link: Option<(&'a str, &'a str)>,
}

impl<'a> LinkMessageRef<'a> {
    fn from_message(link: &'a LinkMessage) -> Self {
        Self {
            name: &link.name,
            link_type: link.link_type,
            creation_order: link.creation_order,
            char_encoding: link.char_encoding,
            hard_link_addr: link.hard_link_addr,
            soft_link_target: link.soft_link_target.as_deref(),
            external_link: link
                .external_link
                .as_ref()
                .map(|(filename, object_path)| (filename.as_str(), object_path.as_str())),
        }
    }

    fn hard_link(name: &'a str, addr: u64) -> Self {
        Self {
            name,
            link_type: LinkType::Hard,
            creation_order: None,
            char_encoding: 0,
            hard_link_addr: Some(addr),
            soft_link_target: None,
            external_link: None,
        }
    }

    fn to_owned(&self) -> LinkMessage {
        LinkMessage {
            name: self.name.to_string(),
            link_type: self.link_type,
            creation_order: self.creation_order,
            char_encoding: self.char_encoding,
            hard_link_addr: self.hard_link_addr,
            soft_link_target: self.soft_link_target.map(str::to_string),
            external_link: self
                .external_link
                .map(|(filename, object_path)| (filename.to_string(), object_path.to_string())),
        }
    }
}

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

/// Safe placeholder for hdf5-metno dataset-builder compatibility.
pub struct GroupDatasetBuilderStub {
    inner: Arc<Mutex<FileInner<BufReader<fs::File>>>>,
    parent_name: String,
}

impl GroupDatasetBuilderStub {
    /// This function is part of the hdf5-metno compatibility layer and should not be removed.
    pub fn empty<T: H5Type>(self) -> GroupDatasetBuilderEmptyStub<T> {
        GroupDatasetBuilderEmptyStub {
            inner: self.inner,
            parent_name: self.parent_name,
            marker: PhantomData,
        }
    }

    /// This function is part of the hdf5-metno compatibility layer and should not be removed.
    pub fn with_data<'d, T: H5Type>(self, data: &'d [T]) -> GroupDatasetBuilderDataStub<'d, T> {
        GroupDatasetBuilderDataStub {
            inner: self.inner,
            parent_name: self.parent_name,
            data,
            shape: None,
        }
    }
}

/// Safe placeholder for hdf5-metno typed dataset-builder compatibility.
pub struct GroupDatasetBuilderEmptyStub<T: H5Type> {
    inner: Arc<Mutex<FileInner<BufReader<fs::File>>>>,
    parent_name: String,
    marker: PhantomData<T>,
}

impl<T: H5Type> GroupDatasetBuilderEmptyStub<T> {
    /// This function is part of the hdf5-metno compatibility layer and should not be removed.
    pub fn shape<S>(self, extents: S) -> GroupDatasetBuilderEmptyShapeStub<T>
    where
        S: IntoDatasetShape,
    {
        GroupDatasetBuilderEmptyShapeStub {
            inner: self.inner,
            parent_name: self.parent_name,
            shape: extents.into_dataset_shape(),
            marker: PhantomData,
        }
    }

    /// This function is part of the hdf5-metno compatibility layer and should not be removed.
    pub fn create<'n, N: Into<Option<&'n str>>>(self, name: N) -> Result<Dataset> {
        self.shape(()).create(name)
    }
}

/// Safe placeholder for hdf5-metno shaped dataset-builder compatibility.
pub struct GroupDatasetBuilderEmptyShapeStub<T: H5Type> {
    inner: Arc<Mutex<FileInner<BufReader<fs::File>>>>,
    parent_name: String,
    shape: Vec<u64>,
    marker: PhantomData<T>,
}

impl<T: H5Type> GroupDatasetBuilderEmptyShapeStub<T> {
    /// This function is part of the hdf5-metno compatibility layer and should not be removed.
    pub fn create<'n, N: Into<Option<&'n str>>>(&self, name: N) -> Result<Dataset> {
        let name = name.into().unwrap_or("<anonymous>");
        create_root_compat_dataset::<T>(&self.inner, &self.parent_name, name, &self.shape, None)
    }
}

/// Safe placeholder for hdf5-metno data-backed dataset-builder compatibility.
pub struct GroupDatasetBuilderDataStub<'d, T: H5Type> {
    inner: Arc<Mutex<FileInner<BufReader<fs::File>>>>,
    parent_name: String,
    data: &'d [T],
    shape: Option<Vec<u64>>,
}

impl<'d, T: H5Type> GroupDatasetBuilderDataStub<'d, T> {
    /// This function is part of the hdf5-metno compatibility layer and should not be removed.
    pub fn shape<S>(mut self, extents: S) -> Self
    where
        S: IntoDatasetShape,
    {
        self.shape = Some(extents.into_dataset_shape());
        self
    }

    /// This function is part of the hdf5-metno compatibility layer and should not be removed.
    pub fn create<'n, N: Into<Option<&'n str>>>(&self, name: N) -> Result<Dataset> {
        let name = name.into().unwrap_or("<anonymous>");
        let inferred_shape;
        let shape = if let Some(shape) = self.shape.as_deref() {
            shape
        } else {
            inferred_shape = vec![usize_to_u64(self.data.len(), "dataset element count")?];
            &inferred_shape
        };
        create_root_compat_dataset::<T>(
            &self.inner,
            &self.parent_name,
            name,
            shape,
            Some(slice_as_bytes(self.data)),
        )
    }
}

/// Shape conversion for the hdf5-metno compatibility dataset builder.
pub trait IntoDatasetShape {
    fn into_dataset_shape(self) -> Vec<u64>;
}

impl IntoDatasetShape for () {
    fn into_dataset_shape(self) -> Vec<u64> {
        Vec::new()
    }
}

impl IntoDatasetShape for u64 {
    fn into_dataset_shape(self) -> Vec<u64> {
        vec![self]
    }
}

impl IntoDatasetShape for usize {
    fn into_dataset_shape(self) -> Vec<u64> {
        vec![self as u64]
    }
}

impl<const N: usize> IntoDatasetShape for [u64; N] {
    fn into_dataset_shape(self) -> Vec<u64> {
        self.to_vec()
    }
}

impl<const N: usize> IntoDatasetShape for [usize; N] {
    fn into_dataset_shape(self) -> Vec<u64> {
        self.into_iter().map(|dim| dim as u64).collect()
    }
}

impl IntoDatasetShape for &[u64] {
    fn into_dataset_shape(self) -> Vec<u64> {
        self.to_vec()
    }
}

impl IntoDatasetShape for &[usize] {
    fn into_dataset_shape(self) -> Vec<u64> {
        self.iter().map(|&dim| dim as u64).collect()
    }
}

fn create_root_compat_dataset<T: H5Type>(
    inner: &Arc<Mutex<FileInner<BufReader<fs::File>>>>,
    parent_name: &str,
    name: &str,
    shape: &[u64],
    data: Option<&[u8]>,
) -> Result<Dataset> {
    if name == "<anonymous>" {
        return Err(Error::Unsupported(
            "anonymous dataset creation is not implemented".into(),
        ));
    }

    let dtype = dtype_for_type::<T>()?;
    let dtype_size = usize::try_from(dtype.size())
        .map_err(|_| Error::InvalidFormat("dataset datatype size exceeds usize".into()))?;
    if dtype_size == 0 {
        return Err(Error::InvalidFormat(
            "dataset datatype size must be nonzero".into(),
        ));
    }
    let element_count = shape_element_count(shape)?;
    let byte_len = usize::try_from(element_count)
        .map_err(|_| Error::InvalidFormat("dataset element count exceeds usize".into()))?
        .checked_mul(dtype_size)
        .ok_or_else(|| Error::InvalidFormat("dataset byte size overflow".into()))?;

    let zero_data;
    let data = if let Some(data) = data {
        if data.len() != byte_len {
            return Err(Error::InvalidFormat(format!(
                "dataset byte length {} does not match shape element count {element_count} * datatype size {dtype_size}",
                data.len()
            )));
        }
        data
    } else {
        zero_data = vec![0; byte_len];
        &zero_data
    };

    let spec = DatasetSpec {
        name,
        shape,
        max_shape: None,
        dtype,
        data,
    };
    let path = {
        let guard = inner.lock();
        if guard.intent != FileIntent::ReadWrite {
            return Err(Error::Unsupported(format!(
                "hdf5-metno compatibility dataset creation requires a read-write File for group '{parent_name}'"
            )));
        }
        guard.path.clone().ok_or_else(|| {
            Error::Unsupported(
                "hdf5-metno compatibility dataset creation requires a file path".into(),
            )
        })?
    };
    let mut file = MutableFile::open_rw(path)?;
    file.create_compact_contiguous_dataset(parent_name, &spec, None)?;
    refresh_shared_reader(inner, parent_name, name)
}

fn refresh_shared_reader(
    inner: &Arc<Mutex<FileInner<BufReader<fs::File>>>>,
    parent_name: &str,
    dataset_name: &str,
) -> Result<Dataset> {
    let (path, access_plist, dset_no_attrs_hint, open_objects, next_object_id) = {
        let guard = inner.lock();
        (
            guard.path.clone().ok_or_else(|| {
                Error::Unsupported(
                    "hdf5-metno compatibility dataset creation requires a file path".into(),
                )
            })?,
            guard.access_plist.clone(),
            guard.dset_no_attrs_hint,
            guard.open_objects.clone(),
            guard.next_object_id,
        )
    };

    let read_file = fs::File::open(&path)?;
    let mut reader = HdfReader::new(BufReader::new(read_file));
    let superblock = Superblock::read(&mut reader)?;
    let root_addr = superblock.root_addr;
    *inner.lock() = FileInner {
        reader,
        superblock,
        path: Some(path),
        intent: FileIntent::ReadWrite,
        access_plist,
        dset_no_attrs_hint,
        open_objects,
        next_object_id,
    };
    Group::open(inner.clone(), "/", root_addr)?
        .open_dataset(&group_child_path(parent_name, dataset_name))
}

fn usize_to_u64(value: usize, context: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| Error::InvalidFormat(format!("{context} exceeds u64")))
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

    fn current_metadata_addr_locked(
        &self,
        inner: &mut FileInner<BufReader<fs::File>>,
    ) -> Result<u64> {
        if self.name == "/" {
            return Ok(inner.superblock.root_addr);
        }

        let mut addr = inner.superblock.root_addr;
        let parts: Vec<&str> = self
            .name
            .trim_start_matches('/')
            .split('/')
            .filter(|part| !part.is_empty())
            .collect();
        for part in parts {
            let oh = ObjectHeader::read_at(&mut inner.reader, addr)?;
            addr = Self::hard_link_addr_in_group_header(
                &mut inner.reader,
                &oh,
                inner.superblock.sizeof_addr,
                part,
            )?;
            let child_oh = ObjectHeader::read_at(&mut inner.reader, addr)?;
            if object_type_from_messages(&child_oh.messages) != ObjectType::Group {
                return Err(Error::InvalidFormat(format!(
                    "'{}' is not a group",
                    group_child_path("/", part)
                )));
            }
        }
        Ok(addr)
    }

    fn hard_link_addr_in_group_header<R>(
        reader: &mut HdfReader<R>,
        oh: &ObjectHeader,
        sizeof_addr: u8,
        name: &str,
    ) -> Result<u64>
    where
        R: std::io::Read + std::io::Seek,
    {
        for msg in &oh.messages {
            if msg.msg_type == object_header::MSG_SYMBOL_TABLE {
                let stab = SymbolTableMessage::decode(&msg.data, sizeof_addr)?;
                let mut found = None;
                Self::visit_v1_group_members(
                    reader,
                    stab.btree_addr,
                    stab.name_heap_addr,
                    |member_name, addr| {
                        if member_name == name {
                            found = Some(addr);
                        }
                        Ok(())
                    },
                )?;
                if let Some(addr) = found {
                    return Ok(addr);
                }
            }
        }

        for msg in &oh.messages {
            if msg.msg_type == object_header::MSG_LINK {
                let link = LinkMessage::decode(&msg.data, sizeof_addr)?;
                if link.name == name {
                    return link.hard_link_addr.ok_or_else(|| {
                        Error::InvalidFormat(format!(
                            "link '{name}' does not reference an object header"
                        ))
                    });
                }
            }
        }

        for msg in &oh.messages {
            if msg.msg_type == object_header::MSG_LINK_INFO {
                let link_info = LinkInfoMessage::decode(&msg.data, sizeof_addr)?;
                if link_info.has_dense_storage() {
                    if let Some(link) =
                        Self::find_dense_link_by_name(reader, &link_info, sizeof_addr, name)?
                    {
                        return link.hard_link_addr.ok_or_else(|| {
                            Error::InvalidFormat(format!(
                                "link '{name}' does not reference an object header"
                            ))
                        });
                    }
                }
            }
        }

        Err(Error::InvalidFormat(format!("link '{name}' not found")))
    }

    fn current_metadata_addr(&self) -> Result<u64> {
        if self.name == "/" {
            return Ok(self.inner.lock().superblock.root_addr);
        } else {
            File::from_inner(self.inner.clone())
                .group(&self.name)
                .map(|group| group.addr())
        }
    }

    fn mutable_file_for_group(&self) -> Result<MutableFile> {
        let guard = self.inner.lock();
        let intent = guard.intent;
        let path = guard.path.clone();
        drop(guard);

        if intent != FileIntent::ReadWrite {
            return Err(Error::Unsupported(format!(
                "hdf5-metno compatibility group mutation requires a read-write File for group '{}'",
                self.name
            )));
        }
        let path = path.ok_or_else(|| {
            Error::Unsupported(
                "hdf5-metno compatibility group mutation requires a file path".into(),
            )
        })?;
        MutableFile::open_rw(path)
    }

    fn refresh_shared_reader_from_path(&self) -> Result<()> {
        let (path, access_plist, dset_no_attrs_hint, open_objects, next_object_id) = {
            let guard = self.inner.lock();
            (
                guard.path.clone().ok_or_else(|| {
                    Error::Unsupported(
                        "hdf5-metno compatibility group mutation requires a file path".into(),
                    )
                })?,
                guard.access_plist.clone(),
                guard.dset_no_attrs_hint,
                guard.open_objects.clone(),
                guard.next_object_id,
            )
        };

        let read_file = fs::File::open(&path)?;
        let mut reader = HdfReader::new(BufReader::new(read_file));
        let superblock = Superblock::read(&mut reader)?;
        *self.inner.lock() = FileInner {
            reader,
            superblock,
            path: Some(path),
            intent: FileIntent::ReadWrite,
            access_plist,
            dset_no_attrs_hint,
            open_objects,
            next_object_id,
        };
        Ok(())
    }

    /// Return this group handle's high-level object id.
    pub fn object_id(&self) -> u64 {
        self.object_id
    }

    /// List all member names in this group.
    pub fn member_names(&self) -> Result<Vec<String>> {
        let mut names = Vec::new();
        self.member_names_into(&mut names)?;
        Ok(names)
    }

    /// Visit all member names in this group without returning an owned list.
    pub fn visit_member_names<F>(&self, mut visitor: F) -> Result<()>
    where
        F: FnMut(&str) -> Result<()>,
    {
        self.visit_link_refs(|link| visitor(link.name))
    }

    /// Store all member names in this group in caller-provided storage.
    pub fn member_names_into(&self, out: &mut Vec<String>) -> Result<()> {
        let mut names = Vec::new();
        self.visit_member_names(|name| {
            names.push(name.to_string());
            Ok(())
        })?;
        *out = names;
        Ok(())
    }

    /// List all links in this group.
    ///
    /// v1 symbol-table groups do not store full v2 link messages, so their
    /// members are returned as synthesized hard-link records.
    pub fn links(&self) -> Result<Vec<LinkMessage>> {
        let mut links = Vec::new();
        self.visit_links_owned(|link| {
            links.push(link);
            Ok(())
        })?;
        Ok(links)
    }

    /// Visit all links in this group.
    pub fn visit_links<F>(&self, mut visitor: F) -> Result<()>
    where
        F: FnMut(&LinkMessage) -> Result<()>,
    {
        self.visit_links_owned(|link| visitor(&link))
    }

    fn visit_links_owned<F>(&self, mut visitor: F) -> Result<()>
    where
        F: FnMut(LinkMessage) -> Result<()>,
    {
        let mut guard = self.inner.lock();
        let sizeof_addr = guard.superblock.sizeof_addr;
        let group_addr = self.current_metadata_addr_locked(&mut guard)?;
        let oh = ObjectHeader::read_at(&mut guard.reader, group_addr)?;

        for msg in &oh.messages {
            if msg.msg_type == object_header::MSG_SYMBOL_TABLE {
                let stab = SymbolTableMessage::decode(&msg.data, sizeof_addr)?;
                return Self::visit_v1_group_members(
                    &mut guard.reader,
                    stab.btree_addr,
                    stab.name_heap_addr,
                    |name, addr| visitor(LinkMessageRef::hard_link(name, addr).to_owned()),
                );
            }
        }

        let mut visited = false;
        for msg in &oh.messages {
            if msg.msg_type == object_header::MSG_LINK {
                let link = LinkMessage::decode(&msg.data, sizeof_addr)?;
                visited = true;
                visitor(link)?;
            }
        }
        if visited {
            return Ok(());
        }

        for msg in &oh.messages {
            if msg.msg_type == object_header::MSG_LINK_INFO {
                let link_info = LinkInfoMessage::decode(&msg.data, sizeof_addr)?;
                if link_info.has_dense_storage() {
                    return Self::visit_dense_link_messages(
                        &mut guard.reader,
                        &link_info,
                        sizeof_addr,
                        |link| visitor(link),
                    );
                }
            }
        }

        Ok(())
    }

    fn visit_link_refs<F>(&self, mut visitor: F) -> Result<()>
    where
        F: FnMut(LinkMessageRef<'_>) -> Result<()>,
    {
        let mut guard = self.inner.lock();
        let sizeof_addr = guard.superblock.sizeof_addr;
        let group_addr = self.current_metadata_addr_locked(&mut guard)?;
        let oh = ObjectHeader::read_at(&mut guard.reader, group_addr)?;

        for msg in &oh.messages {
            if msg.msg_type == object_header::MSG_SYMBOL_TABLE {
                let stab = SymbolTableMessage::decode(&msg.data, sizeof_addr)?;
                return Self::visit_v1_group_members(
                    &mut guard.reader,
                    stab.btree_addr,
                    stab.name_heap_addr,
                    |name, addr| visitor(LinkMessageRef::hard_link(name, addr)),
                );
            }
        }

        let mut visited = false;
        for msg in &oh.messages {
            if msg.msg_type == object_header::MSG_LINK {
                let link = LinkMessage::decode(&msg.data, sizeof_addr)?;
                visited = true;
                visitor(LinkMessageRef::from_message(&link))?;
            }
        }
        if visited {
            return Ok(());
        }

        for msg in &oh.messages {
            if msg.msg_type == object_header::MSG_LINK_INFO {
                let link_info = LinkInfoMessage::decode(&msg.data, sizeof_addr)?;
                if link_info.has_dense_storage() {
                    return Self::visit_dense_link_refs(
                        &mut guard.reader,
                        &link_info,
                        sizeof_addr,
                        |link| visitor(link),
                    );
                }
            }
        }

        Ok(())
    }

    pub(crate) fn visit_link_refs_for_link_access<F>(&self, visitor: F) -> Result<()>
    where
        F: FnMut(LinkMessageRef<'_>) -> Result<()>,
    {
        self.visit_link_refs(visitor)
    }

    /// List all links sorted by tracked creation order.
    pub fn links_by_creation_order(&self) -> Result<Vec<LinkMessage>> {
        let mut links = Vec::new();
        self.visit_links_owned(|link| {
            links.push(link);
            Ok(())
        })?;
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

    /// Visit all links sorted by tracked creation order.
    pub fn visit_links_by_creation_order<F>(&self, mut visitor: F) -> Result<()>
    where
        F: FnMut(&LinkMessage) -> Result<()>,
    {
        let mut links = Vec::new();
        self.visit_links_owned(|link| {
            links.push(link);
            Ok(())
        })?;
        if links.iter().any(|link| link.creation_order.is_none()) {
            return Err(Error::Unsupported(format!(
                "group '{}' does not track link creation order",
                self.name
            )));
        }
        links.sort_by_key(|link| link.creation_order.unwrap_or(u64::MAX));
        for link in &links {
            visitor(link)?;
        }
        Ok(())
    }

    /// List all members as (name, object_header_addr) pairs.
    pub fn members(&self) -> Result<Vec<(String, u64)>> {
        let mut members = Vec::new();
        self.visit_members(|name, addr| {
            members.push((name.to_string(), addr));
            Ok(())
        })?;
        Ok(members)
    }

    /// Visit all members as `(name, object_header_addr)` pairs.
    pub fn visit_members<F>(&self, mut visitor: F) -> Result<()>
    where
        F: FnMut(&str, u64) -> Result<()>,
    {
        let mut guard = self.inner.lock();
        let sizeof_addr = guard.superblock.sizeof_addr;
        let group_addr = self.current_metadata_addr_locked(&mut guard)?;
        let oh = ObjectHeader::read_at(&mut guard.reader, group_addr)?;

        // Check for v1 symbol table message
        for msg in &oh.messages {
            if msg.msg_type == object_header::MSG_SYMBOL_TABLE {
                let stab = SymbolTableMessage::decode(&msg.data, sizeof_addr)?;
                return Self::visit_v1_group_members(
                    &mut guard.reader,
                    stab.btree_addr,
                    stab.name_heap_addr,
                    |name, addr| visitor(name, addr),
                );
            }
        }

        // V2: collect from link messages
        let mut visited = false;
        for msg in &oh.messages {
            if msg.msg_type == object_header::MSG_LINK {
                if let Ok(link) = LinkMessage::decode(&msg.data, sizeof_addr) {
                    visited = true;
                    visitor(&link.name, link.hard_link_addr.unwrap_or(0))?;
                }
            }
        }
        if visited {
            return Ok(());
        }

        // V2 dense storage: link info message with fractal heap + v2 B-tree
        for msg in &oh.messages {
            if msg.msg_type == object_header::MSG_LINK_INFO {
                let link_info = LinkInfoMessage::decode(&msg.data, sizeof_addr)?;
                if link_info.has_dense_storage() {
                    return Self::visit_dense_link_refs(
                        &mut guard.reader,
                        &link_info,
                        sizeof_addr,
                        |link| visitor(link.name, link.hard_link_addr.unwrap_or(0)),
                    );
                }
            }
        }

        Ok(())
    }

    fn visit_v1_group_members<R, F>(
        reader: &mut crate::io::reader::HdfReader<R>,
        btree_addr: u64,
        heap_addr: u64,
        mut visitor: F,
    ) -> Result<()>
    where
        R: std::io::Read + std::io::Seek,
        F: FnMut(&str, u64) -> Result<()>,
    {
        let heap = LocalHeap::read_at(reader, heap_addr)?;
        let mut visited = Vec::new();
        Self::visit_v1_btree_members(reader, btree_addr, 0, &mut visited, &heap, &mut visitor)
    }

    fn visit_v1_btree_members<R, F>(
        reader: &mut crate::io::reader::HdfReader<R>,
        btree_addr: u64,
        depth: usize,
        visited: &mut Vec<u64>,
        heap: &LocalHeap,
        visitor: &mut F,
    ) -> Result<()>
    where
        R: std::io::Read + std::io::Seek,
        F: FnMut(&str, u64) -> Result<()>,
    {
        const MAX_GROUP_BTREE_RECURSION: usize = 64;

        if depth > MAX_GROUP_BTREE_RECURSION {
            return Err(Error::InvalidFormat(
                "v1 group B-tree recursion depth exceeded".into(),
            ));
        }
        if crate::io::reader::is_undef_addr(btree_addr) {
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

        let node = BTreeV1Node::read_at(reader, btree_addr)?;
        if node.node_type != BTreeType::Group {
            visited.pop();
            return Err(Error::InvalidFormat("expected group B-tree".into()));
        }

        if node.level == 0 {
            for child_addr in &node.children {
                let snod = SymbolTableNode::read_at(reader, *child_addr)?;
                for entry in &snod.entries {
                    let name_offset = usize::try_from(entry.name_offset).map_err(|_| {
                        Error::InvalidFormat(
                            "symbol-table name offset does not fit in usize".into(),
                        )
                    })?;
                    let name = heap.get_str(name_offset)?;
                    if !name.is_empty() {
                        visitor(name, entry.obj_header_addr)?;
                    }
                }
            }
        } else {
            for child_addr in &node.children {
                Self::visit_v1_btree_members(
                    reader,
                    *child_addr,
                    depth.checked_add(1).ok_or_else(|| {
                        Error::InvalidFormat("v1 group B-tree recursion depth overflow".into())
                    })?,
                    visited,
                    heap,
                    visitor,
                )?;
            }
        }

        visited.pop();
        Ok(())
    }

    /// Visit dense links from fractal heap + v2 B-tree as full LinkMessage objects.
    fn visit_dense_link_messages<R, F>(
        reader: &mut crate::io::reader::HdfReader<R>,
        link_info: &LinkInfoMessage,
        sizeof_addr: u8,
        mut visitor: F,
    ) -> Result<()>
    where
        R: std::io::Read + std::io::Seek,
        F: FnMut(LinkMessage) -> Result<()>,
    {
        let heap = FractalHeapHeader::read_at(reader, link_info.fractal_heap_addr)?;
        let heap_id_len = usize::from(heap.heap_id_len);
        let mut heap_ids = Vec::new();
        Self::collect_dense_link_heap_ids(reader, link_info, heap_id_len, &mut heap_ids)?;

        let mut cache = FractalHeapManagedObjectCache::new();
        for heap_id in &heap_ids {
            if let Ok(link_data) = heap.read_managed_object_cached(reader, heap_id, &mut cache) {
                if let Ok(link) = LinkMessage::decode(&link_data, sizeof_addr) {
                    visitor(link)?;
                }
            }
        }

        Ok(())
    }

    fn collect_dense_link_heap_ids<R>(
        reader: &mut crate::io::reader::HdfReader<R>,
        link_info: &LinkInfoMessage,
        heap_id_len: usize,
        heap_ids: &mut Vec<Vec<u8>>,
    ) -> Result<()>
    where
        R: std::io::Read + std::io::Seek,
    {
        heap_ids.clear();
        btree_v2::visit_all_records(reader, link_info.name_btree_addr, |record| {
            let Some(heap_id) = dense_link_heap_id(record, heap_id_len)? else {
                return Ok(());
            };
            heap_ids.push(heap_id.to_vec());
            Ok(())
        })
    }

    /// Visit dense links from fractal heap + v2 B-tree as borrowed link views.
    fn visit_dense_link_refs<R, F>(
        reader: &mut crate::io::reader::HdfReader<R>,
        link_info: &LinkInfoMessage,
        sizeof_addr: u8,
        mut visitor: F,
    ) -> Result<()>
    where
        R: std::io::Read + std::io::Seek,
        F: FnMut(LinkMessageRef<'_>) -> Result<()>,
    {
        let heap = FractalHeapHeader::read_at(reader, link_info.fractal_heap_addr)?;
        let heap_id_len = usize::from(heap.heap_id_len);
        let mut heap_ids = Vec::new();
        Self::collect_dense_link_heap_ids(reader, link_info, heap_id_len, &mut heap_ids)?;

        let mut cache = FractalHeapManagedObjectCache::new();
        for heap_id in &heap_ids {
            if let Ok(link_data) = heap.read_managed_object_cached(reader, heap_id, &mut cache) {
                if let Ok(link) = LinkMessage::decode(&link_data, sizeof_addr) {
                    visitor(LinkMessageRef::from_message(&link))?;
                }
            }
        }

        Ok(())
    }

    fn find_dense_link_by_name<R>(
        reader: &mut crate::io::reader::HdfReader<R>,
        link_info: &LinkInfoMessage,
        sizeof_addr: u8,
        name: &str,
    ) -> Result<Option<LinkMessage>>
    where
        R: std::io::Read + std::io::Seek,
    {
        let heap = FractalHeapHeader::read_at(reader, link_info.fractal_heap_addr)?;
        let target_hash = checksum_lookup3(name.as_bytes(), 0);
        let heap_id_len = usize::from(heap.heap_id_len);
        let mut heap_ids = Vec::new();
        btree_v2::visit_matching_records(
            reader,
            link_info.name_btree_addr,
            |record| match dense_link_name_hash(record) {
                Some(hash) => hash.cmp(&target_hash),
                None => Ordering::Less,
            },
            |record| {
                let Some(heap_id) = dense_link_heap_id(record, heap_id_len)? else {
                    return Ok(());
                };
                heap_ids.push(heap_id.to_vec());
                Ok(())
            },
        )?;

        let mut cache = FractalHeapManagedObjectCache::new();
        for heap_id in &heap_ids {
            if let Ok(link_data) = heap.read_managed_object_cached(reader, heap_id, &mut cache) {
                if let Ok(link) = LinkMessage::decode(&link_data, sizeof_addr) {
                    if link.name == name {
                        return Ok(Some(link));
                    }
                }
            }
        }

        Ok(None)
    }

    /// Find a specific link by name, checking both inline messages and dense storage.
    pub(crate) fn find_link_by_name(&self, name: &str) -> Result<LinkMessage> {
        self.with_link_by_name(name, |link| Ok(link.clone()))
    }

    pub(crate) fn with_link_by_name<R, F>(&self, name: &str, visitor: F) -> Result<R>
    where
        F: FnOnce(&LinkMessage) -> Result<R>,
    {
        let mut visitor = Some(visitor);
        let mut guard = self.inner.lock();
        let sizeof_addr = guard.superblock.sizeof_addr;
        let group_addr = self.current_metadata_addr_locked(&mut guard)?;
        let oh = ObjectHeader::read_at(&mut guard.reader, group_addr)?;

        // Check v1 symbol table messages
        for msg in &oh.messages {
            if msg.msg_type == object_header::MSG_SYMBOL_TABLE {
                let stab = SymbolTableMessage::decode(&msg.data, sizeof_addr)?;
                let mut found = None;
                Self::visit_v1_group_members(
                    &mut guard.reader,
                    stab.btree_addr,
                    stab.name_heap_addr,
                    |member_name, addr| {
                        if member_name == name {
                            found = Some(LinkMessage {
                                name: member_name.to_string(),
                                link_type: LinkType::Hard,
                                creation_order: None,
                                char_encoding: 0,
                                hard_link_addr: Some(addr),
                                soft_link_target: None,
                                external_link: None,
                            });
                        }
                        Ok(())
                    },
                )?;
                if let Some(link) = found {
                    let visitor = visitor.take().expect("link visitor called more than once");
                    return visitor(&link);
                }
            }
        }

        // Check inline link messages
        for msg in &oh.messages {
            if msg.msg_type == object_header::MSG_LINK {
                if let Ok(link) = LinkMessage::decode(&msg.data, sizeof_addr) {
                    if link.name == name {
                        let visitor = visitor.take().expect("link visitor called more than once");
                        return visitor(&link);
                    }
                }
            }
        }

        // Check dense storage
        for msg in &oh.messages {
            if msg.msg_type == object_header::MSG_LINK_INFO {
                let link_info = LinkInfoMessage::decode(&msg.data, sizeof_addr)?;
                if link_info.has_dense_storage() {
                    if let Some(link) = Self::find_dense_link_by_name(
                        &mut guard.reader,
                        &link_info,
                        sizeof_addr,
                        name,
                    )? {
                        let visitor = visitor.take().expect("link visitor called more than once");
                        return visitor(&link);
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

    /// This function is part of the hdf5-metno compatibility layer and should not be removed.
    pub fn create_group(&self, name: &str) -> Result<Group> {
        let mut file = self.mutable_file_for_group()?;
        file.create_compact_group_link(&self.name, name)?;
        self.refresh_shared_reader_from_path()?;
        File::from_inner(self.inner.clone()).group(&group_child_path(&self.name, name))
    }

    /// This function is part of the hdf5-metno compatibility layer and should not be removed.
    pub fn group(&self, name: &str) -> Result<Group> {
        self.open_group(name)
    }

    /// Get the number of members in this group.
    pub fn len(&self) -> Result<usize> {
        let mut len = 0usize;
        self.visit_members(|_, _| {
            len += 1;
            Ok(())
        })?;
        Ok(len)
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
        self.with_linkval(name, |value| Ok(value.map(str::to_string)))
    }

    pub fn with_linkval<R, F>(&self, name: &str, visitor: F) -> Result<R>
    where
        F: FnOnce(Option<&str>) -> Result<R>,
    {
        self.with_link_by_name(name, |link| {
            if let Some(target) = link.soft_link_target.as_deref() {
                return visitor(Some(target));
            }
            let mut external = String::new();
            if let Some((file, path)) = link.external_link.as_ref() {
                external.push_str(file);
                external.push(':');
                external.push_str(path);
                return visitor(Some(&external));
            }
            visitor(None)
        })
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

    /// This function is part of the hdf5-metno compatibility layer and should not be removed.
    pub fn iter_visit_default<F, G>(&self, val: G, mut op: F) -> Result<G>
    where
        F: FnMut(&Self, &str, LinkInfo, &mut G) -> bool,
    {
        let mut val = val;
        let mut stop = false;
        self.visit_links(|link| {
            if stop {
                return Ok(());
            }
            let info = link_info_from_message(link)?;
            if !op(self, &link.name, info, &mut val) {
                stop = true;
            }
            Ok(())
        })?;
        Ok(val)
    }

    /// This function is part of the hdf5-metno compatibility layer and should not be removed.
    pub fn groups(&self) -> Result<Vec<Group>> {
        let mut groups = Vec::new();
        let mut hard_links = Vec::new();
        let mut fallback_names = Vec::new();
        self.visit_links_owned(|link| {
            if let Some(addr) = link.hard_link_addr {
                hard_links.push((link.name, addr));
            } else {
                fallback_names.push(link.name);
            }
            Ok(())
        })?;
        for (name, addr) in hard_links {
            if self.object_type_at(addr)? == ObjectType::Group {
                groups.push(Group::open(
                    self.inner.clone(),
                    &group_child_path(&self.name, &name),
                    addr,
                )?);
            }
        }
        for name in fallback_names {
            if self.member_type(&name)? == ObjectType::Group {
                groups.push(self.open_group(&name)?);
            }
        }
        Ok(groups)
    }

    /// This function is part of the hdf5-metno compatibility layer and should not be removed.
    pub fn datasets(&self) -> Result<Vec<Dataset>> {
        let mut datasets = Vec::new();
        let mut hard_links = Vec::new();
        let mut fallback_names = Vec::new();
        self.visit_links_owned(|link| {
            if let Some(addr) = link.hard_link_addr {
                hard_links.push((link.name, addr));
            } else {
                fallback_names.push(link.name);
            }
            Ok(())
        })?;
        for (name, addr) in hard_links {
            if self.object_type_at(addr)? == ObjectType::Dataset {
                datasets.push(Dataset::new(
                    self.inner.clone(),
                    &group_child_path(&self.name, &name),
                    addr,
                ));
            }
        }
        for name in fallback_names {
            if self.member_type(&name)? == ObjectType::Dataset {
                datasets.push(self.open_dataset(&name)?);
            }
        }
        Ok(datasets)
    }

    /// This function is part of the hdf5-metno compatibility layer and should not be removed.
    pub fn named_datatypes(&self) -> Result<Vec<Datatype>> {
        let mut datatypes = Vec::new();
        let mut hard_links = Vec::new();
        let mut fallback_names = Vec::new();
        self.visit_links_owned(|link| {
            if let Some(addr) = link.hard_link_addr {
                hard_links.push((link.name, addr));
            } else {
                fallback_names.push(link.name);
            }
            Ok(())
        })?;
        for (_name, addr) in hard_links {
            if self.object_type_at(addr)? == ObjectType::NamedDatatype {
                datatypes.push(self.named_datatype_at(addr)?);
            }
        }
        for name in fallback_names {
            if self.member_type(&name)? == ObjectType::NamedDatatype {
                let addr = self.hard_link_addr_by_name(&name)?;
                datatypes.push(self.named_datatype_at(addr)?);
            }
        }
        Ok(datatypes)
    }

    pub fn objinfo(&self, name: &str) -> Result<ObjectInfo> {
        let addr = self.hard_link_addr_by_name(name)?;
        self.object_info_at(addr)
    }

    pub fn objname_by_idx(&self, index: usize) -> Result<String> {
        let mut name = String::new();
        self.link_name_by_idx_into(index, &mut name)?;
        Ok(name)
    }

    pub fn objtype_by_idx(&self, index: usize) -> Result<ObjectType> {
        self.with_link_by_idx(index, |link| {
            if let Some(addr) = link.hard_link_addr {
                self.object_type_at(addr)
            } else {
                self.member_type(&link.name)
            }
        })
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
        let mut names = Vec::new();
        self.attr_names_into(&mut names)?;
        Ok(names)
    }

    /// Visit attribute names in storage order.
    pub fn visit_attr_names<F>(&self, mut f: F) -> Result<()>
    where
        F: FnMut(&str) -> Result<()>,
    {
        visit_attr_names_at(&self.inner, self.current_metadata_addr()?, &mut f)
    }

    /// Store attribute names in storage order in caller-provided storage.
    pub fn attr_names_into(&self, out: &mut Vec<String>) -> Result<()> {
        let mut names = Vec::new();
        self.visit_attr_names(|name| {
            names.push(name.to_string());
            Ok(())
        })?;
        *out = names;
        Ok(())
    }

    /// List attributes.
    pub fn attrs(&self) -> Result<Vec<crate::hl::attribute::Attribute>> {
        let mut attrs = Vec::new();
        self.attrs_into(&mut attrs)?;
        Ok(attrs)
    }

    /// Visit attributes in storage order.
    pub fn visit_attrs<F>(&self, mut f: F) -> Result<()>
    where
        F: FnMut(&crate::hl::attribute::Attribute) -> Result<()>,
    {
        visit_attrs_at(&self.inner, self.current_metadata_addr()?, &mut f)
    }

    /// Store attributes in caller-provided storage.
    pub fn attrs_into(&self, out: &mut Vec<crate::hl::attribute::Attribute>) -> Result<()> {
        let mut attrs = Vec::new();
        crate::hl::attribute::collect_attributes_into(
            &self.inner,
            self.current_metadata_addr()?,
            &mut attrs,
        )?;
        *out = attrs;
        Ok(())
    }

    /// List attributes sorted by tracked creation order.
    pub fn attrs_by_creation_order(&self) -> Result<Vec<crate::hl::attribute::Attribute>> {
        let mut attrs = Vec::new();
        self.attrs_by_creation_order_into(&mut attrs)?;
        Ok(attrs)
    }

    /// Visit attributes sorted by tracked creation order.
    pub fn visit_attrs_by_creation_order<F>(&self, mut f: F) -> Result<()>
    where
        F: FnMut(&crate::hl::attribute::Attribute) -> Result<()>,
    {
        let attrs = crate::hl::attribute::collect_attributes_by_creation_order(
            &self.inner,
            self.current_metadata_addr()?,
        )?;
        for attr in &attrs {
            f(attr)?;
        }
        Ok(())
    }

    /// Store attributes sorted by tracked creation order in caller-provided storage.
    pub fn attrs_by_creation_order_into(
        &self,
        out: &mut Vec<crate::hl::attribute::Attribute>,
    ) -> Result<()> {
        let mut attrs = Vec::new();
        crate::hl::attribute::collect_attributes_by_creation_order_into(
            &self.inner,
            self.current_metadata_addr()?,
            &mut attrs,
        )?;
        *out = attrs;
        Ok(())
    }

    /// Get an attribute by name.
    pub fn attr(&self, name: &str) -> Result<crate::hl::attribute::Attribute> {
        crate::hl::attribute::get_attr(&self.inner, self.current_metadata_addr()?, name)
    }

    /// Check whether an attribute exists on this group.
    pub fn attr_exists(&self, name: &str) -> Result<bool> {
        crate::hl::attribute::attr_exists(&self.inner, self.current_metadata_addr()?, name)
    }

    /// Get the link type of a member by name.
    pub fn link_type(&self, name: &str) -> Result<LinkType> {
        self.with_link_ref_by_name(name, |link| Ok(link.link_type))
    }

    /// Get link metadata by name.
    pub fn link_info(&self, name: &str) -> Result<LinkInfo> {
        self.with_link_ref_by_name(name, link_info_from_ref)
    }

    pub(crate) fn with_link_ref_by_name<R, F>(&self, name: &str, visitor: F) -> Result<R>
    where
        F: FnOnce(LinkMessageRef<'_>) -> Result<R>,
    {
        let mut visitor = Some(visitor);
        let mut guard = self.inner.lock();
        let sizeof_addr = guard.superblock.sizeof_addr;
        let group_addr = self.current_metadata_addr_locked(&mut guard)?;
        let oh = ObjectHeader::read_at(&mut guard.reader, group_addr)?;

        for msg in &oh.messages {
            if msg.msg_type == object_header::MSG_SYMBOL_TABLE {
                let stab = SymbolTableMessage::decode(&msg.data, sizeof_addr)?;
                let mut found = None;
                Self::visit_v1_group_members(
                    &mut guard.reader,
                    stab.btree_addr,
                    stab.name_heap_addr,
                    |member_name, addr| {
                        if member_name == name {
                            found = Some((member_name.to_string(), addr));
                        }
                        Ok(())
                    },
                )?;
                if let Some((member_name, addr)) = found {
                    let visit = visitor
                        .take()
                        .expect("link ref visitor called more than once");
                    return visit(LinkMessageRef::hard_link(&member_name, addr));
                }
            }
        }

        for msg in &oh.messages {
            if msg.msg_type == object_header::MSG_LINK {
                if let Ok(link) = LinkMessage::decode(&msg.data, sizeof_addr) {
                    if link.name == name {
                        let visit = visitor
                            .take()
                            .expect("link ref visitor called more than once");
                        return visit(LinkMessageRef::from_message(&link));
                    }
                }
            }
        }

        for msg in &oh.messages {
            if msg.msg_type == object_header::MSG_LINK_INFO {
                let link_info = LinkInfoMessage::decode(&msg.data, sizeof_addr)?;
                if link_info.has_dense_storage() {
                    if let Some(link) = Self::find_dense_link_by_name(
                        &mut guard.reader,
                        &link_info,
                        sizeof_addr,
                        name,
                    )? {
                        let visit = visitor
                            .take()
                            .expect("link ref visitor called more than once");
                        return visit(LinkMessageRef::from_message(&link));
                    }
                }
            }
        }

        Err(Error::InvalidFormat(format!("link '{name}' not found")))
    }

    fn with_link_by_idx<R, F>(&self, index: usize, visitor: F) -> Result<R>
    where
        F: FnOnce(&LinkMessage) -> Result<R>,
    {
        let mut found = None;
        let mut pos = 0usize;
        self.visit_links_owned(|link| {
            if pos == index {
                found = Some(link);
            }
            pos += 1;
            Ok(())
        })?;
        match found {
            Some(link) => visitor(&link),
            None => Err(Error::InvalidFormat(format!(
                "link index {index} is out of bounds"
            ))),
        }
    }

    /// Get link metadata by zero-based storage-order index.
    pub fn link_info_by_idx(&self, index: usize) -> Result<LinkInfo> {
        let mut info = None;
        let mut pos = 0usize;
        self.visit_link_refs(|link| {
            if pos == index {
                info = Some(link_info_from_ref(link)?);
            }
            pos += 1;
            Ok(())
        })?;
        info.ok_or_else(|| Error::InvalidFormat(format!("link index {index} is out of bounds")))
    }

    /// Get a link name by zero-based storage-order index.
    pub fn link_name_by_idx(&self, index: usize) -> Result<String> {
        let mut name = String::new();
        self.link_name_by_idx_into(index, &mut name)?;
        Ok(name)
    }

    /// Get a link name by zero-based storage-order index into caller-provided storage.
    pub fn link_name_by_idx_into(&self, index: usize, out: &mut String) -> Result<()> {
        let mut found = false;
        let mut pos = 0usize;
        self.visit_link_refs(|link| {
            if pos == index {
                out.clear();
                out.push_str(link.name);
                found = true;
            }
            pos += 1;
            Ok(())
        })?;
        if found {
            Ok(())
        } else {
            Err(Error::InvalidFormat(format!(
                "link index {index} is out of bounds"
            )))
        }
    }

    /// Get a soft or external link value by zero-based storage-order index.
    pub fn link_value_by_idx(&self, index: usize) -> Result<Option<LinkValue>> {
        self.link_value_by_idx_with(index, |value| Ok(value.map(LinkValueRef::to_owned)))
    }

    /// Visit a soft or external link value by zero-based storage-order index.
    pub fn link_value_by_idx_with<R, F>(&self, index: usize, visitor: F) -> Result<R>
    where
        F: FnOnce(Option<LinkValueRef<'_>>) -> Result<R>,
    {
        self.with_link_by_idx(index, |link| visitor(get_val_cb_borrowed(link)))
    }

    /// Get the target path of a soft link.
    pub fn soft_link_target(&self, name: &str) -> Result<String> {
        self.soft_link_target_with(name, |target| Ok(target.to_string()))
    }

    /// Visit the target path of a soft link.
    pub fn soft_link_target_with<R, F>(&self, name: &str, visitor: F) -> Result<R>
    where
        F: FnOnce(&str) -> Result<R>,
    {
        self.with_link_by_name(name, |link| {
            link.soft_link_target
                .as_deref()
                .ok_or_else(|| Error::InvalidFormat(format!("'{name}' is not a soft link")))
                .and_then(visitor)
        })
    }

    /// Get the target (filename, object_path) of an external link.
    pub fn external_link_target(&self, name: &str) -> Result<(String, String)> {
        self.external_link_target_with(name, |filename, object_path| {
            Ok((filename.to_string(), object_path.to_string()))
        })
    }

    /// Visit the target (filename, object_path) of an external link.
    pub fn external_link_target_with<R, F>(&self, name: &str, visitor: F) -> Result<R>
    where
        F: FnOnce(&str, &str) -> Result<R>,
    {
        self.with_link_by_name(name, |link| {
            link.external_link
                .as_ref()
                .ok_or_else(|| Error::InvalidFormat(format!("'{name}' is not an external link")))
                .and_then(|(filename, object_path)| visitor(filename, object_path))
        })
    }

    /// Get this group's object comment, if present.
    pub fn object_comment(&self) -> Result<Option<String>> {
        self.object_comment_at(self.current_metadata_addr()?)
    }

    /// Get a child object's comment by link name, if present.
    pub fn object_comment_by_name(&self, name: &str) -> Result<Option<String>> {
        let addr = self.hard_link_addr_by_name(name)?;
        self.object_comment_at(addr)
    }

    /// Get native object-header metadata for this group.
    pub fn native_info(&self) -> Result<ObjectInfo> {
        self.object_info_at(self.current_metadata_addr()?)
    }

    /// Get v3-style child object metadata by zero-based link index.
    pub fn object_info_by_idx(&self, index: usize) -> Result<ObjectInfo> {
        self.with_link_by_idx(index, |link| {
            let addr = link.hard_link_addr.ok_or_else(|| {
                Error::InvalidFormat(format!(
                    "link '{}' does not reference an object header",
                    link.name
                ))
            })?;
            self.object_info_at(addr)
        })
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

    /// This function is part of the hdf5-metno compatibility layer and should not be removed.
    pub fn link_soft(&self, target: &str, link_name: &str) -> Result<()> {
        let mut file = self.mutable_file_for_group()?;
        file.create_compact_soft_link(&self.name, link_name, target)?;
        self.refresh_shared_reader_from_path()
    }

    /// This function is part of the hdf5-metno compatibility layer and should not be removed.
    pub fn link_hard(&self, target: &str, link_name: &str) -> Result<()> {
        let target_addr = self.object_addr_for_hard_link_target(target)?;
        let mut file = self.mutable_file_for_group()?;
        if relative_same_group_direct_target_name(target).is_some() {
            let target_path = group_child_path(&self.name, target);
            file.create_compact_hard_link_to_target_path(
                &self.name,
                link_name,
                &target_path,
                target_addr,
            )?;
        } else if let Some(target_name) = same_group_direct_target_name(&self.name, target) {
            file.create_compact_same_group_hard_link(
                &self.name,
                link_name,
                target_name,
                target_addr,
            )?;
        } else if normalized_absolute_child_path(target).is_some() {
            file.create_compact_hard_link_to_target_path(
                &self.name,
                link_name,
                target,
                target_addr,
            )?;
        } else {
            file.create_compact_hard_link(&self.name, link_name, target_addr)?;
        }
        self.refresh_shared_reader_from_path()
    }

    /// This function is part of the hdf5-metno compatibility layer and should not be removed.
    pub fn link_external(
        &self,
        target_file_name: &str,
        target: &str,
        link_name: &str,
    ) -> Result<()> {
        let mut file = self.mutable_file_for_group()?;
        file.create_compact_external_link(&self.name, link_name, target_file_name, target)?;
        self.refresh_shared_reader_from_path()
    }

    /// This function is part of the hdf5-metno compatibility layer and should not be removed.
    pub fn relink(&self, name: &str, path: &str) -> Result<()> {
        let mut file = self.mutable_file_for_group()?;
        if path.contains('/') {
            let (dest_group, dest_name) = relink_destination(&self.name, path)?;
            file.move_group_link(&self.name, name, &dest_group, &dest_name)?;
        } else {
            file.rename_group_link(&self.name, name, path)?;
        }
        self.refresh_shared_reader_from_path()
    }

    /// This function is part of the hdf5-metno compatibility layer and should not be removed.
    pub fn unlink(&self, name: &str) -> Result<()> {
        let mut file = self.mutable_file_for_group()?;
        file.unlink_group_link(&self.name, name)?;
        self.refresh_shared_reader_from_path()
    }

    /// This function is part of the hdf5-metno compatibility layer and should not be removed.
    pub fn new_dataset<T: H5Type>(&self) -> GroupDatasetBuilderEmptyStub<T> {
        self.new_dataset_builder().empty::<T>()
    }

    /// This function is part of the hdf5-metno compatibility layer and should not be removed.
    pub fn new_dataset_builder(&self) -> GroupDatasetBuilderStub {
        GroupDatasetBuilderStub {
            inner: self.inner.clone(),
            parent_name: self.name.clone(),
        }
    }

    /// This function is part of the hdf5-metno compatibility layer and should not be removed.
    pub fn dataset(&self, name: &str) -> Result<Dataset> {
        self.open_dataset(name)
    }

    fn hard_link_addr_by_name(&self, name: &str) -> Result<u64> {
        self.with_link_ref_by_name(name, |link| {
            link.hard_link_addr.ok_or_else(|| {
                Error::InvalidFormat(format!("link '{name}' does not reference an object header"))
            })
        })
    }

    fn object_addr_for_hard_link_target(&self, target: &str) -> Result<u64> {
        if target.is_empty() {
            return Err(Error::InvalidFormat(
                "hard link target cannot be empty".into(),
            ));
        }
        if target == "/" {
            return Ok(File::from_inner(self.inner.clone()).root_group()?.addr());
        }

        let file = File::from_inner(self.inner.clone());
        if !target.starts_with('/') {
            let relative_target = group_child_path(&self.name, target);
            if let Ok(dataset) = file.dataset(&relative_target) {
                return Ok(dataset.addr());
            }
            if let Ok(group) = file.group(&relative_target) {
                return Ok(group.addr());
            }
        }
        if let Ok(dataset) = file.dataset(target) {
            return Ok(dataset.addr());
        }
        if let Ok(group) = file.group(target) {
            return Ok(group.addr());
        }
        Err(Error::InvalidFormat(format!(
            "hard link target '{target}' not found"
        )))
    }

    fn object_info_at(&self, addr: u64) -> Result<ObjectInfo> {
        let mut guard = self.inner.lock();
        let oh = ObjectHeader::read_at(&mut guard.reader, addr)?;
        Ok(object_info_from_header(addr, &oh))
    }

    fn object_type_at(&self, addr: u64) -> Result<ObjectType> {
        let mut guard = self.inner.lock();
        let oh = ObjectHeader::read_at(&mut guard.reader, addr)?;
        Ok(object_type_from_messages(&oh.messages))
    }

    fn object_comment_at(&self, addr: u64) -> Result<Option<String>> {
        let mut guard = self.inner.lock();
        let oh = ObjectHeader::read_at(&mut guard.reader, addr)?;
        object_comment_from_header(&oh)
    }

    fn named_datatype_at(&self, addr: u64) -> Result<Datatype> {
        let mut guard = self.inner.lock();
        let oh = ObjectHeader::read_at(&mut guard.reader, addr)?;
        for msg in &oh.messages {
            if msg.msg_type == object_header::MSG_DATATYPE {
                return Ok(Datatype::from_message(DatatypeMessage::decode(&msg.data)?));
            }
        }
        Err(Error::InvalidFormat(format!(
            "object at address {addr} does not contain a named datatype message"
        )))
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

pub(crate) fn link_info_from_ref(link: LinkMessageRef<'_>) -> Result<LinkInfo> {
    Ok(LinkInfo {
        link_type: link.link_type,
        creation_order_valid: link.creation_order.is_some(),
        creation_order: link.creation_order.unwrap_or(0),
        char_encoding: link.char_encoding,
        hard_link_addr: link.hard_link_addr,
    })
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

fn same_group_direct_target_name<'a>(group_name: &str, target: &'a str) -> Option<&'a str> {
    let target = target.strip_prefix('/')?;
    if target.is_empty() || target.contains("/.") || target.contains("//") {
        return None;
    }

    if group_name == "/" {
        return (!target.contains('/')).then_some(target);
    }

    let group = group_name.strip_prefix('/')?;
    let child = target.strip_prefix(group)?.strip_prefix('/')?;
    (!child.is_empty() && !child.contains('/')).then_some(child)
}

fn relative_same_group_direct_target_name(target: &str) -> Option<&str> {
    (!target.is_empty()
        && !target.starts_with('/')
        && !target.contains('/')
        && target != "."
        && target != "..")
        .then_some(target)
}

fn normalized_absolute_child_path(path: &str) -> Option<(&str, &str)> {
    if path == "/" || !path.starts_with('/') || path.contains("/.") || path.contains("//") {
        return None;
    }
    let (parent, name) = path.rsplit_once('/')?;
    if name.is_empty() {
        return None;
    }
    Some((if parent.is_empty() { "/" } else { parent }, name))
}

fn relink_destination(group_name: &str, path: &str) -> Result<(String, String)> {
    if path.is_empty() || path == "/" {
        return Err(Error::InvalidFormat(
            "relink destination cannot be empty".into(),
        ));
    }
    if path.contains("/.") || path.contains("//") {
        return Err(Error::Unsupported(
            "cross-group compact relink currently supports only normalized paths".into(),
        ));
    }

    let absolute = if path.starts_with('/') {
        path.to_string()
    } else if group_name == "/" {
        format!("/{path}")
    } else {
        format!("{group_name}/{path}")
    };
    let (parent, name) = absolute.rsplit_once('/').ok_or_else(|| {
        Error::InvalidFormat(format!("relink destination '{path}' is not a child path"))
    })?;
    if name.is_empty() {
        return Err(Error::InvalidFormat(
            "relink destination cannot be empty".into(),
        ));
    }
    let parent = if parent.is_empty() { "/" } else { parent };
    Ok((parent.to_string(), name.to_string()))
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

pub(crate) fn visit_attr_names_at<F>(
    inner: &Arc<Mutex<FileInner<BufReader<fs::File>>>>,
    addr: u64,
    mut visitor: F,
) -> Result<()>
where
    F: FnMut(&str) -> Result<()>,
{
    visit_attr_messages_at(inner, addr, |attr_msg, _creation_order| {
        visitor(&attr_msg.name)
    })
}

pub(crate) fn visit_attrs_at<F>(
    inner: &Arc<Mutex<FileInner<BufReader<fs::File>>>>,
    addr: u64,
    mut visitor: F,
) -> Result<()>
where
    F: FnMut(&Attribute) -> Result<()>,
{
    visit_attr_messages_at(inner, addr, |attr_msg, creation_order| {
        let attr = Attribute::from_message(attr_msg, creation_order, inner);
        visitor(&attr)
    })
}

fn visit_attr_messages_at<F>(
    inner: &Arc<Mutex<FileInner<BufReader<fs::File>>>>,
    addr: u64,
    mut visitor: F,
) -> Result<()>
where
    F: FnMut(AttributeMessage, Option<u64>) -> Result<()>,
{
    let (oh, sizeof_addr) = {
        let mut guard = inner.lock();
        let oh = ObjectHeader::read_at(&mut guard.reader, addr)?;
        (oh, guard.superblock.sizeof_addr)
    };

    for msg in &oh.messages {
        if msg.msg_type == object_header::MSG_ATTRIBUTE {
            match AttributeMessage::decode(&msg.data) {
                Ok(attr_msg) => visitor(attr_msg, msg.creation_index.map(u64::from))?,
                Err(e) => {
                    eprintln!("Warning: failed to decode attribute: {e}");
                }
            }
        }
    }

    for msg in &oh.messages {
        if msg.msg_type != object_header::MSG_ATTR_INFO {
            continue;
        }

        let attr_info = AttributeInfoMessage::decode(&msg.data, sizeof_addr)?;
        if !attr_info.has_dense_storage() {
            continue;
        }

        let (heap, records) = {
            let mut guard = inner.lock();
            let heap = FractalHeapHeader::read_at(&mut guard.reader, attr_info.fractal_heap_addr)?;
            let mut records = Vec::new();
            btree_v2::collect_all_records_into(
                &mut guard.reader,
                attr_info.name_btree_addr,
                &mut records,
            )?;
            (heap, records)
        };
        let heap_id_len = usize::from(heap.heap_id_len);

        let mut heap_ids = Vec::new();
        let mut creation_orders = Vec::new();
        for record in &records {
            if record.len() < heap_id_len {
                continue;
            }

            heap_ids.push(&record[..heap_id_len]);
            creation_orders.push(dense_attribute_record_creation_order(record, heap_id_len));
        }

        let attr_data = {
            let mut guard = inner.lock();
            heap.read_managed_objects_batched(&mut guard.reader, &heap_ids)
        };

        for (attr_data, creation_order) in attr_data.into_iter().zip(creation_orders) {
            if let Ok(attr_data) = attr_data {
                match AttributeMessage::decode(&attr_data) {
                    Ok(attr_msg) => visitor(attr_msg, creation_order)?,
                    Err(e) => {
                        eprintln!("Warning: failed to decode dense attribute: {e}");
                    }
                }
            }
        }
    }

    Ok(())
}

fn dense_attribute_record_creation_order(record: &[u8], heap_id_len: usize) -> Option<u64> {
    let start = heap_id_len.checked_add(1)?;
    let end = start.checked_add(4)?;
    let bytes = record.get(start..end)?;
    Some(u64::from(u32::from_le_bytes(bytes.try_into().ok()?)))
}

fn dense_link_heap_id(record: &[u8], heap_id_len: usize) -> Result<Option<&[u8]>> {
    let end = 4usize
        .checked_add(heap_id_len)
        .ok_or_else(|| Error::InvalidFormat("dense link heap ID length overflow".into()))?;
    Ok(record.get(4..end))
}

fn dense_link_name_hash(record: &[u8]) -> Option<u32> {
    let bytes = record.get(0..4)?;
    Some(u32::from_le_bytes(bytes.try_into().ok()?))
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
