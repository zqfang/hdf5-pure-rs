use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{BufReader, Read, Seek};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::error::{Error, Result};
use crate::format::btree_v1::BTreeV1Node;
use crate::format::local_heap::LocalHeap;
use crate::format::messages::link::{LinkMessage, LinkType};
use crate::format::object_header::{self, RawMessage};
use crate::format::superblock::Superblock;
use crate::format::symbol_table::SymbolTableNode;
use crate::hl::dataset::Dataset;
use crate::hl::group::Group;
use crate::io::reader::HdfReader;

/// Represents the type of an HDF5 object as determined by its object header messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectType {
    Group,
    Dataset,
    NamedDatatype,
    Unknown,
}

/// File open intent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileIntent {
    ReadOnly,
    ReadWrite,
}

/// File-level metadata summary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileInfo {
    pub superblock: SuperblockInfo,
    pub free_space: FreeSpaceInfo,
    pub shared_messages: SharedMessageInfo,
}

/// Superblock/storage information reported by file metadata queries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SuperblockInfo {
    pub version: u8,
    pub size: u64,
    pub extension_size: u64,
}

/// Free-space-manager information reported by file metadata queries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FreeSpaceInfo {
    pub version: u8,
    pub metadata_size: u64,
    pub total_space: u64,
}

/// Shared-object-header-message information reported by file metadata queries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedMessageInfo {
    pub header_size: u64,
    pub message_info_size: u64,
}

/// Metadata-cache size/status snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetadataCacheSize {
    pub max_size: usize,
    pub min_clean_size: usize,
    pub current_size: usize,
    pub current_num_entries: usize,
}

/// Metadata-cache image status.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetadataCacheImageInfo {
    pub generated: bool,
    pub size: usize,
}

/// Page-buffering status counters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageBufferingStats {
    pub metadata_accesses: u64,
    pub metadata_hits: u64,
    pub raw_data_accesses: u64,
    pub raw_data_hits: u64,
}

/// Internal state of an open HDF5 file.
pub(crate) struct FileInner<R: Read + Seek> {
    pub reader: HdfReader<R>,
    pub superblock: Superblock,
    pub path: Option<PathBuf>,
    pub access_plist: crate::hl::plist::file_access::FileAccess,
    pub dset_no_attrs_hint: bool,
    pub open_objects: HashMap<u64, OpenObjectKind>,
    pub next_object_id: u64,
}

/// An open HDF5 file.
pub struct File {
    inner: Arc<Mutex<FileInner<BufReader<fs::File>>>>,
    superblock: Superblock,
    object_id: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OpenObjectKind {
    File,
    Group,
    Dataset,
    Attribute,
}

impl File {
    const MAX_SOFT_LINK_TRAVERSALS: usize = 40;
    /// Per-component byte cap, matching upstream `H5G_TRAVERSE_PATH_MAX`.
    /// Bounds the length of any single name segment between '/' separators.
    const MAX_PATH_COMPONENT_LEN: usize = 1024;

    /// Open an HDF5 file for reading.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let f = fs::File::open(path.as_ref()).map_err(|e| {
            Error::Io(std::io::Error::new(
                e.kind(),
                format!("failed to open {}: {e}", path.as_ref().display()),
            ))
        })?;

        let buf = BufReader::new(f);
        let mut reader = HdfReader::new(buf);

        let superblock = Superblock::read(&mut reader)?;

        let inner = Arc::new(Mutex::new(FileInner {
            reader,
            superblock: superblock.clone(),
            path: Some(path.as_ref().to_path_buf()),
            access_plist: crate::hl::plist::file_access::FileAccess::default(),
            dset_no_attrs_hint: false,
            open_objects: HashMap::new(),
            next_object_id: 1,
        }));

        let object_id = register_open_object(&inner, OpenObjectKind::File);
        Ok(File {
            inner,
            superblock,
            object_id,
        })
    }

    /// Get the superblock.
    pub fn superblock(&self) -> &Superblock {
        &self.superblock
    }

    /// Return the current on-disk file size in bytes.
    ///
    /// This mirrors the useful read-side subset of HDF5's `H5Fget_filesize`
    /// without exposing the broader file-driver API surface.
    pub fn file_size(&self) -> Result<u64> {
        self.inner.lock().reader.len()
    }

    /// Return the path used to open this file, when the file has an on-disk path.
    ///
    /// This mirrors the useful file-level subset of HDF5's `H5Fget_name`.
    pub fn path(&self) -> Option<PathBuf> {
        self.inner.lock().path.clone()
    }

    /// Return the access properties for this open file.
    pub fn access_plist(&self) -> crate::hl::plist::file_access::FileAccess {
        crate::hl::plist::file_access::FileAccess::from_file(self)
    }

    pub(crate) fn access_plist_snapshot(&self) -> crate::hl::plist::file_access::FileAccess {
        self.inner.lock().access_plist.clone()
    }

    /// Replace the file's stored access-property state.
    pub fn set_access_plist(&self, plist: crate::hl::plist::file_access::FileAccess) {
        self.inner.lock().access_plist = plist;
    }

    /// Return this file's open intent.
    pub fn intent(&self) -> FileIntent {
        FileIntent::ReadOnly
    }

    /// Return the parsed end-of-address marker from the superblock.
    pub fn eoa(&self) -> u64 {
        self.superblock.eof_addr
    }

    /// Return the known free-space size. The reader does not currently parse
    /// free-space manager state, so this reports zero known free bytes.
    pub fn freespace(&self) -> u64 {
        0
    }

    /// Return file metadata information in the v2 `H5F_info_t` layout.
    pub fn info(&self) -> Result<FileInfo> {
        Ok(FileInfo {
            superblock: SuperblockInfo {
                version: self.superblock.version,
                size: u64::try_from(self.superblock.checked_size()?)
                    .map_err(|_| Error::InvalidFormat("superblock size does not fit u64".into()))?,
                extension_size: 0,
            },
            free_space: FreeSpaceInfo {
                version: 0,
                metadata_size: 0,
                total_space: self.freespace(),
            },
            shared_messages: SharedMessageInfo {
                header_size: 0,
                message_info_size: 0,
            },
        })
    }

    /// Return file metadata information in the v1 `H5F_info_t` layout.
    pub fn info_v1(&self) -> Result<FileInfo> {
        self.info()
    }

    /// Return the current file image bytes.
    pub fn file_image(&self) -> Result<Vec<u8>> {
        let mut guard = self.inner.lock();
        let pos = guard.reader.position()?;
        let len = guard.reader.len()?;
        let len = usize::try_from(len)
            .map_err(|_| Error::InvalidFormat("file image length does not fit usize".into()))?;
        guard.reader.seek(0)?;
        let image = guard.reader.read_bytes(len);
        let restore = guard.reader.seek(pos);
        match (image, restore) {
            (Ok(image), Ok(_)) => Ok(image),
            (Err(err), _) => Err(err),
            (_, Err(err)) => Err(err),
        }
    }

    /// Return a stable file-number surrogate for this open file.
    pub fn fileno(&self) -> Result<u64> {
        let path = self
            .path()
            .ok_or_else(|| Error::Unsupported("open file has no filesystem path".into()))?;
        file_number_from_path(&path)
    }

    /// Return this file handle's high-level object id.
    pub fn object_id(&self) -> u64 {
        self.object_id
    }

    /// Return the number of currently live high-level objects for this file.
    pub fn obj_count(&self) -> usize {
        self.inner.lock().open_objects.len()
    }

    /// Return currently live high-level object ids for this file.
    pub fn obj_ids(&self) -> Vec<u64> {
        let mut ids: Vec<u64> = self.inner.lock().open_objects.keys().copied().collect();
        ids.sort_unstable();
        ids
    }

    /// Return the native handle for the direct file driver when available.
    #[cfg(unix)]
    pub fn vfd_handle(&self) -> Option<i64> {
        use std::os::fd::AsRawFd;

        Some(i64::from(
            self.inner.lock().reader.get_ref().get_ref().as_raw_fd(),
        ))
    }

    /// Return the native handle for the direct file driver when available.
    #[cfg(not(unix))]
    pub fn vfd_handle(&self) -> Option<i64> {
        None
    }

    /// Return whether MPI atomicity is enabled for this file.
    ///
    /// This pure-Rust reader does not use MPI or the parallel HDF5 VFD, so
    /// atomicity is always disabled.
    pub fn mpi_atomicity(&self) -> bool {
        false
    }

    /// Return metadata cache configuration for this file.
    pub fn mdc_config(&self) -> crate::hl::plist::file_access::MetadataCacheConfig {
        self.access_plist().mdc_config()
    }

    /// Set metadata cache configuration for this open file handle.
    pub fn set_mdc_config(&self, config: crate::hl::plist::file_access::MetadataCacheConfig) {
        self.inner.lock().access_plist.set_mdc_config(config);
    }

    /// Set library format-version bounds for this open file handle.
    pub fn set_libver_bounds(
        &self,
        low: crate::hl::plist::file_access::LibverBound,
        high: crate::hl::plist::file_access::LibverBound,
    ) {
        self.inner.lock().access_plist.set_libver_bounds(low, high);
    }

    /// Set latest-format bounds for this open file handle.
    pub fn set_latest_format(&self) {
        use crate::hl::plist::file_access::LibverBound;

        self.set_libver_bounds(LibverBound::Latest, LibverBound::Latest);
    }

    /// Return metadata cache hit rate. No libhdf5 metadata cache is present.
    pub fn mdc_hit_rate(&self) -> f64 {
        0.0
    }

    /// Return metadata cache size/status.
    pub fn mdc_size(&self) -> MetadataCacheSize {
        MetadataCacheSize {
            max_size: 0,
            min_clean_size: 0,
            current_size: 0,
            current_num_entries: 0,
        }
    }

    /// Return metadata cache logging status `(enabled, currently_logging)`.
    pub fn mdc_logging_status(&self) -> (bool, bool) {
        (false, false)
    }

    /// Return page-buffering status counters.
    pub fn page_buffering_stats(&self) -> PageBufferingStats {
        PageBufferingStats {
            metadata_accesses: 0,
            metadata_hits: 0,
            raw_data_accesses: 0,
            raw_data_hits: 0,
        }
    }

    /// Return metadata cache image status.
    pub fn mdc_image_info(&self) -> MetadataCacheImageInfo {
        MetadataCacheImageInfo {
            generated: false,
            size: 0,
        }
    }

    /// Return the dataset-no-attributes optimization hint.
    pub fn dset_no_attrs_hint(&self) -> bool {
        self.inner.lock().dset_no_attrs_hint
    }

    /// Set the dataset-no-attributes optimization hint for this open file.
    pub fn set_dset_no_attrs_hint(&self, enabled: bool) {
        self.inner.lock().dset_no_attrs_hint = enabled;
    }

    #[cfg(feature = "tracehash")]
    pub(crate) fn inner_arc(&self) -> Arc<Mutex<FileInner<BufReader<fs::File>>>> {
        self.inner.clone()
    }

    pub(crate) fn from_inner(inner: Arc<Mutex<FileInner<BufReader<fs::File>>>>) -> Self {
        let superblock = inner.lock().superblock.clone();
        let object_id = register_open_object(&inner, OpenObjectKind::File);
        Self {
            inner,
            superblock,
            object_id,
        }
    }

    /// Get the root group.
    pub fn root_group(&self) -> Result<Group> {
        Group::open(self.inner.clone(), "/", self.superblock.root_addr)
    }

    /// List all member names in the root group.
    pub fn member_names(&self) -> Result<Vec<String>> {
        self.root_group()?.member_names()
    }

    /// Open a group by path (starting from root).
    pub fn group(&self, path: &str) -> Result<Group> {
        let resolved = self.resolve_path(path)?;
        if resolved.object_type != ObjectType::Group {
            return Err(Error::InvalidFormat(format!(
                "'{path}' is not a group (type: {:?})",
                resolved.object_type
            )));
        }
        Group::open(resolved.inner, &resolved.path, resolved.addr)
    }

    /// List attribute names on the root group.
    pub fn attr_names(&self) -> Result<Vec<String>> {
        crate::hl::attribute::attr_names(&self.inner, self.superblock.root_addr)
    }

    /// List attributes on the root group.
    pub fn attrs(&self) -> Result<Vec<crate::hl::attribute::Attribute>> {
        crate::hl::attribute::collect_attributes(&self.inner, self.superblock.root_addr)
    }

    /// List attributes on the root group sorted by tracked creation order.
    pub fn attrs_by_creation_order(&self) -> Result<Vec<crate::hl::attribute::Attribute>> {
        crate::hl::attribute::collect_attributes_by_creation_order(
            &self.inner,
            self.superblock.root_addr,
        )
    }

    /// Get an attribute by name on the root group.
    pub fn attr(&self, name: &str) -> Result<crate::hl::attribute::Attribute> {
        crate::hl::attribute::get_attr(&self.inner, self.superblock.root_addr, name)
    }

    /// Check whether an attribute exists on the root group.
    pub fn attr_exists(&self, name: &str) -> Result<bool> {
        crate::hl::attribute::attr_exists(&self.inner, self.superblock.root_addr, name)
    }

    /// Async-compatible alias for attribute-existence checks.
    pub fn attr_exists_async(&self, name: &str) -> Result<bool> {
        self.attr_exists(name)
    }

    /// Check whether an attribute exists on an object addressed by path.
    pub fn attr_exists_by_name(&self, object_path: &str, attr_name: &str) -> Result<bool> {
        let resolved = self.resolve_path(object_path)?;
        crate::hl::attribute::attr_exists(&resolved.inner, resolved.addr, attr_name)
    }

    /// Async-compatible alias for path-based attribute-existence checks.
    pub fn attr_exists_by_name_async(&self, object_path: &str, attr_name: &str) -> Result<bool> {
        self.attr_exists_by_name(object_path, attr_name)
    }

    /// Open a dataset by path from the root group.
    pub fn dataset(&self, path: &str) -> Result<Dataset> {
        let resolved = self.resolve_path(path)?;
        if resolved.object_type != ObjectType::Dataset {
            return Err(Error::InvalidFormat(format!(
                "'{path}' is not a dataset (type: {:?})",
                resolved.object_type
            )));
        }
        Ok(Dataset::new(resolved.inner, &resolved.path, resolved.addr))
    }

    pub(crate) fn object_type_for_path(&self, path: &str) -> Result<ObjectType> {
        Ok(self.resolve_path(path)?.object_type)
    }

    fn resolve_path(&self, path: &str) -> Result<ResolvedObject> {
        let mut path = canonical_path(path);
        let mut traversals = 0usize;
        let mut seen_paths = HashSet::new();

        'resolve: loop {
            if !seen_paths.insert(path.clone()) {
                return Err(Error::InvalidFormat(format!(
                    "soft link cycle detected while resolving '{path}'"
                )));
            }
            if path == "/" {
                return Ok(self.root_resolved_object(path));
            }

            let parts = Self::path_components(&path)?;
            let mut current = self.root_group()?;
            let mut current_path = String::from("/");

            for (idx, part) in parts.iter().enumerate() {
                let is_last = idx + 1 == parts.len();
                let link = self.lookup_group_link(&current, part)?;
                match self.resolve_path_component(
                    &current,
                    &current_path,
                    part,
                    is_last,
                    &parts[idx + 1..],
                    link,
                    &mut traversals,
                )? {
                    PathStep::Resolved(resolved) => return Ok(resolved),
                    PathStep::Descend(next_group, next_path) => {
                        current = next_group;
                        current_path = next_path;
                    }
                    PathStep::Restart(new_path) => {
                        path = new_path;
                        continue 'resolve;
                    }
                }
            }
        }
    }

    fn root_resolved_object(&self, path: String) -> ResolvedObject {
        ResolvedObject {
            inner: self.inner.clone(),
            path,
            addr: self.superblock.root_addr,
            object_type: ObjectType::Group,
        }
    }

    fn path_components(path: &str) -> Result<Vec<String>> {
        let parts: Vec<String> = path
            .trim_start_matches('/')
            .split('/')
            .filter(|part| !part.is_empty())
            .map(str::to_string)
            .collect();
        for part in &parts {
            if part.len() > Self::MAX_PATH_COMPONENT_LEN {
                return Err(Error::InvalidFormat(format!(
                    "path component exceeds {}-byte limit ({} bytes)",
                    Self::MAX_PATH_COMPONENT_LEN,
                    part.len()
                )));
            }
        }
        Ok(parts)
    }

    fn lookup_group_link(&self, current: &Group, part: &str) -> Result<LinkMessage> {
        match current.find_link_by_name(part) {
            Ok(link) => Ok(link),
            Err(link_err) => {
                if let Some((_, addr)) = current
                    .members()?
                    .into_iter()
                    .find(|(member_name, _)| member_name == part)
                {
                    Ok(LinkMessage {
                        name: part.to_string(),
                        link_type: LinkType::Hard,
                        creation_order: None,
                        char_encoding: 0,
                        hard_link_addr: Some(addr),
                        soft_link_target: None,
                        external_link: None,
                    })
                } else {
                    Err(link_err)
                }
            }
        }
    }

    fn resolve_path_component(
        &self,
        _current: &Group,
        current_path: &str,
        part: &str,
        is_last: bool,
        remaining_parts: &[String],
        link: LinkMessage,
        traversals: &mut usize,
    ) -> Result<PathStep> {
        match link.link_type {
            LinkType::Hard => {
                let addr = link.hard_link_addr.ok_or_else(|| {
                    Error::InvalidFormat(format!(
                        "hard link '{}' is missing object address",
                        link.name
                    ))
                })?;
                let next_path = join_absolute_path(current_path, part);
                let object_type = self.object_type_at(addr)?;
                self.resolve_hard_path_step(next_path, addr, object_type, is_last)
            }
            LinkType::Soft => {
                *traversals += 1;
                if *traversals > Self::MAX_SOFT_LINK_TRAVERSALS {
                    return Err(Error::InvalidFormat(
                        "soft link traversal limit exceeded".into(),
                    ));
                }
                let target = link.soft_link_target.ok_or_else(|| {
                    Error::InvalidFormat(format!(
                        "soft link '{}' is missing target path",
                        link.name
                    ))
                })?;
                let remaining = remaining_parts.join("/");
                Ok(PathStep::Restart(resolve_soft_target(
                    current_path,
                    &target,
                    &remaining,
                )))
            }
            LinkType::External => {
                let (filename, object_path) = link.external_link.ok_or_else(|| {
                    Error::InvalidFormat(format!(
                        "external link '{}' is missing target path",
                        link.name
                    ))
                })?;
                let remaining = remaining_parts.join("/");
                let target_path = if remaining.is_empty() {
                    canonical_path(&object_path)
                } else {
                    canonical_path(&join_absolute_path(&object_path, &remaining))
                };
                let file_path = self.resolve_external_file_path(&filename)?;
                let external_file = File::open(file_path)?;
                Ok(PathStep::Resolved(
                    external_file.resolve_path(&target_path)?,
                ))
            }
            LinkType::UserDefined(kind) => Err(Error::Unsupported(format!(
                "user-defined link traversal is not supported for link type {kind}"
            ))),
        }
    }

    fn resolve_hard_path_step(
        &self,
        next_path: String,
        addr: u64,
        object_type: ObjectType,
        is_last: bool,
    ) -> Result<PathStep> {
        if is_last {
            return Ok(PathStep::Resolved(ResolvedObject {
                inner: self.inner.clone(),
                path: next_path,
                addr,
                object_type,
            }));
        }

        if object_type != ObjectType::Group {
            return Err(Error::InvalidFormat(format!(
                "'{next_path}' is not a group (type: {object_type:?})"
            )));
        }

        let next_group = Group::open(self.inner.clone(), &next_path, addr)?;
        Ok(PathStep::Descend(next_group, next_path))
    }

    fn object_type_at(&self, addr: u64) -> Result<ObjectType> {
        let mut guard = self.inner.lock();
        let oh = object_header::ObjectHeader::read_at(&mut guard.reader, addr)?;
        Ok(object_type_from_messages(&oh.messages))
    }

    fn resolve_external_file_path(&self, filename: &str) -> Result<PathBuf> {
        let path = PathBuf::from(filename);
        if path.is_absolute() {
            return Ok(path);
        }
        let base = self
            .inner
            .lock()
            .path
            .as_ref()
            .and_then(|path| path.parent().map(Path::to_path_buf))
            .ok_or_else(|| {
                Error::InvalidFormat("relative external link has no base file path".into())
            })?;
        Ok(base.join(path))
    }
}

impl Drop for File {
    fn drop(&mut self) {
        unregister_open_object(&self.inner, self.object_id);
    }
}

pub(crate) fn register_open_object(
    inner: &Arc<Mutex<FileInner<BufReader<fs::File>>>>,
    kind: OpenObjectKind,
) -> u64 {
    let mut guard = inner.lock();
    let id = guard.next_object_id;
    guard.next_object_id = guard.next_object_id.saturating_add(1);
    if guard.next_object_id == id {
        guard.next_object_id = id.saturating_add(1);
    }
    guard.open_objects.insert(id, kind);
    id
}

pub(crate) fn unregister_open_object(
    inner: &Arc<Mutex<FileInner<BufReader<fs::File>>>>,
    object_id: u64,
) {
    inner.lock().open_objects.remove(&object_id);
}

struct ResolvedObject {
    inner: Arc<Mutex<FileInner<BufReader<fs::File>>>>,
    path: String,
    addr: u64,
    object_type: ObjectType,
}

enum PathStep {
    Resolved(ResolvedObject),
    Descend(Group, String),
    Restart(String),
}

fn canonical_path(path: &str) -> String {
    let mut parts = Vec::new();
    for part in path.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            other => parts.push(other),
        }
    }
    if parts.is_empty() {
        "/".to_string()
    } else {
        format!("/{}", parts.join("/"))
    }
}

fn join_absolute_path(parent: &str, child: &str) -> String {
    if parent == "/" {
        format!("/{child}")
    } else {
        format!("{parent}/{child}")
    }
}

fn resolve_soft_target(parent: &str, target: &str, remaining: &str) -> String {
    let base = if target.starts_with('/') {
        canonical_path(target)
    } else {
        canonical_path(&join_absolute_path(parent, target))
    };
    if remaining.is_empty() {
        base
    } else {
        canonical_path(&join_absolute_path(&base, remaining))
    }
}

/// Determine object type from an object header's messages.
pub(crate) fn object_type_from_messages(messages: &[RawMessage]) -> ObjectType {
    let has_dataspace = messages
        .iter()
        .any(|m| m.msg_type == object_header::MSG_DATASPACE);
    let has_layout = messages
        .iter()
        .any(|m| m.msg_type == object_header::MSG_LAYOUT);
    let has_datatype = messages
        .iter()
        .any(|m| m.msg_type == object_header::MSG_DATATYPE);
    let has_stab = messages
        .iter()
        .any(|m| m.msg_type == object_header::MSG_SYMBOL_TABLE);
    let has_link = messages
        .iter()
        .any(|m| m.msg_type == object_header::MSG_LINK);
    let has_link_info = messages
        .iter()
        .any(|m| m.msg_type == object_header::MSG_LINK_INFO);

    if has_layout || (has_dataspace && has_datatype && !has_stab && !has_link && !has_link_info) {
        ObjectType::Dataset
    } else if has_stab || has_link || has_link_info {
        ObjectType::Group
    } else if has_datatype && !has_dataspace {
        ObjectType::NamedDatatype
    } else if messages.is_empty() {
        // Empty object header -- likely an empty group (v2 format)
        ObjectType::Group
    } else {
        ObjectType::Unknown
    }
}

/// Collect group member names from a v1 symbol table.
pub(crate) fn collect_v1_group_members<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    btree_addr: u64,
    heap_addr: u64,
) -> Result<Vec<(String, u64)>> {
    let heap = LocalHeap::read_at(reader, heap_addr)?;
    let snod_addrs = BTreeV1Node::collect_symbol_table_addrs(reader, btree_addr)?;

    let mut members = Vec::new();

    for snod_addr in snod_addrs {
        let snod = SymbolTableNode::read_at(reader, snod_addr)?;
        for entry in &snod.entries {
            let name_offset = usize::try_from(entry.name_offset).map_err(|_| {
                Error::InvalidFormat("symbol-table name offset does not fit in usize".into())
            })?;
            let name = heap.get_string(name_offset)?;
            if !name.is_empty() {
                members.push((name, entry.obj_header_addr));
            }
        }
    }

    Ok(members)
}

/// Collect group member names from v2 link messages in an object header.
pub(crate) fn collect_v2_link_members(
    messages: &[RawMessage],
    sizeof_addr: u8,
) -> Vec<(String, u64)> {
    let mut members = Vec::new();

    for msg in messages {
        if msg.msg_type == object_header::MSG_LINK {
            if let Ok(link) = LinkMessage::decode(&msg.data, sizeof_addr) {
                // Include all link types; use hard_link_addr or 0 for soft/external
                let addr = link.hard_link_addr.unwrap_or(0);
                members.push((link.name, addr));
            }
        }
    }

    members
}

#[cfg(unix)]
fn file_number_from_path(path: &Path) -> Result<u64> {
    use std::os::unix::fs::MetadataExt;

    let metadata = fs::metadata(path)?;
    Ok((metadata.dev() << 32) ^ metadata.ino())
}

#[cfg(not(unix))]
fn file_number_from_path(path: &Path) -> Result<u64> {
    use std::hash::{Hash, Hasher};

    let metadata = fs::metadata(path)?;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    path.hash(&mut hasher);
    metadata.len().hash(&mut hasher);
    Ok(hasher.finish())
}
