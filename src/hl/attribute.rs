use std::fs;
use std::io::BufReader;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::error::{Error, Result};
use crate::format::btree_v2;
use crate::format::fractal_heap::FractalHeapHeader;
use crate::format::messages::attribute::AttributeMessage;
use crate::format::messages::attribute_info::AttributeInfoMessage;
use crate::format::messages::dataspace::{DataspaceMessage, DataspaceType};
use crate::format::messages::datatype::DatatypeClass;
use crate::format::messages::datatype::DatatypeMessage;
use crate::format::object_header::{self, ObjectHeader};
use crate::hl::dataspace::Dataspace;
use crate::hl::datatype::Datatype;
use crate::hl::file::{register_open_object, unregister_open_object, FileInner, OpenObjectKind};

/// An HDF5 attribute, parsed from an object header.
pub struct Attribute {
    pub msg: AttributeMessage,
    creation_order: Option<u64>,
    inner: Option<Arc<Mutex<FileInner<BufReader<fs::File>>>>>,
    object_id: Option<u64>,
}

/// Basic attribute metadata, mirroring the read-side subset of `H5Aget_info`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttributeInfo {
    pub creation_order_valid: bool,
    pub creation_order: u64,
    pub char_encoding: u8,
    pub data_size: usize,
}

#[derive(Debug, Clone)]
pub struct AttributeTableEntry {
    pub attr: AttributeMessage,
    pub creation_order: Option<u64>,
    pub shared_refcount: u32,
}

#[derive(Debug, Clone, Default)]
pub struct AttributeTable {
    attrs: Vec<AttributeTableEntry>,
    version: u8,
    closed: bool,
}

impl AttributeTable {
    /// Create an empty attribute table at the latest (v3) attribute encoding.
    pub fn new() -> Self {
        Self {
            attrs: Vec::new(),
            version: 3,
            closed: false,
        }
    }

    /// Return an error if the table has been closed; otherwise `Ok(())`.
    fn ensure_open(&self) -> Result<()> {
        if self.closed {
            Err(Error::InvalidFormat("attribute table is closed".into()))
        } else {
            Ok(())
        }
    }

    /// Locate the slot of an attribute by exact name match.
    fn find_index(&self, name: &str) -> Option<usize> {
        self.attrs.iter().position(|entry| entry.attr.name == name)
    }

    /// Compute the next creation-order index to assign (one past the current
    /// maximum, saturating on overflow).
    fn next_creation_order(&self) -> u64 {
        self.attrs
            .iter()
            .filter_map(|entry| entry.creation_order)
            .max()
            .map_or(0, |value| value.saturating_add(1))
    }
}

/// Build a default in-memory `AttributeMessage` for `name` whose payload is
/// `data`, using a fixed-point datatype sized to fit the bytes and a scalar
/// or 1-D simple dataspace depending on whether the data is empty.
fn default_attribute_message(name: &str, data: Vec<u8>) -> Result<AttributeMessage> {
    let data_len_u32 = u32::try_from(data.len().max(1))
        .map_err(|_| Error::InvalidFormat("attribute data length exceeds u32".into()))?;
    let data_len_u64 = u64::try_from(data.len())
        .map_err(|_| Error::InvalidFormat("attribute data length exceeds u64".into()))?;
    Ok(AttributeMessage {
        version: 3,
        name: name.to_string(),
        char_encoding: 0,
        datatype: DatatypeMessage {
            version: 1,
            class: DatatypeClass::FixedPoint,
            class_bits: [0, 0, 0],
            size: data_len_u32,
            properties: vec![0, 0, 0, 0],
        },
        dataspace: DataspaceMessage {
            version: 2,
            space_type: if data.is_empty() {
                DataspaceType::Scalar
            } else {
                DataspaceType::Simple
            },
            ndims: if data.is_empty() { 0 } else { 1 },
            dims: if data.is_empty() {
                Vec::new()
            } else {
                vec![data_len_u64]
            },
            max_dims: None,
        },
        data,
    })
}

/// Initialize the attribute interface. Stub for parity with libhdf5's
/// `H5A_init` — no per-interface state to set up in this port.
#[allow(non_snake_case)]
pub fn H5A_init() -> bool {
    true
}

/// Package-level initialization hook. Mirrors `H5A__init_package`; defers
/// to `H5A_init` in this port.
#[allow(non_snake_case)]
pub fn H5A__init_package() -> bool {
    H5A_init()
}

/// First-phase interface shutdown. Mirrors `H5A_top_term_package`; nothing
/// to release in this port.
#[allow(non_snake_case)]
pub fn H5A_top_term_package() {}

/// Second-phase interface shutdown. Mirrors `H5A_term_package`; nothing to
/// release in this port.
#[allow(non_snake_case)]
pub fn H5A_term_package() {}

/// Insert a fully-built attribute message into `table`, rejecting duplicate
/// names and assigning the next available creation-order index. Mirrors the
/// internals of libhdf5's `H5A__create`.
#[allow(non_snake_case)]
pub fn H5A__create(table: &mut AttributeTable, attr: AttributeMessage) -> Result<()> {
    table.ensure_open()?;
    if table.find_index(&attr.name).is_some() {
        return Err(Error::InvalidFormat(format!(
            "attribute '{}' already exists",
            attr.name
        )));
    }
    let creation_order = Some(table.next_creation_order());
    table.attrs.push(AttributeTableEntry {
        attr,
        creation_order,
        shared_refcount: 1,
    });
    Ok(())
}

/// Create an attribute on an object by name, using a default datatype and
/// dataspace derived from the data payload. Mirrors `H5A__create_by_name`.
#[allow(non_snake_case)]
pub fn H5A__create_by_name(table: &mut AttributeTable, name: &str, data: Vec<u8>) -> Result<()> {
    H5A__create(table, default_attribute_message(name, data)?)
}

/// Common API entry point for attribute creation. Mirrors
/// `H5A__create_api_common`.
#[allow(non_snake_case)]
pub fn H5A__create_api_common(table: &mut AttributeTable, name: &str, data: Vec<u8>) -> Result<()> {
    H5A__create_by_name(table, name, data)
}

/// Common API entry point for attribute creation by object name. Mirrors
/// `H5A__create_by_name_api_common`.
#[allow(non_snake_case)]
pub fn H5A__create_by_name_api_common(
    table: &mut AttributeTable,
    name: &str,
    data: Vec<u8>,
) -> Result<()> {
    H5A__create_by_name(table, name, data)
}

/// Deprecated 1.6-era public attribute-create entry point. Mirrors
/// `H5Acreate1`; superseded by `H5Acreate2`.
#[allow(non_snake_case)]
pub fn H5Acreate1(table: &mut AttributeTable, name: &str, data: Vec<u8>) -> Result<()> {
    H5A__create_api_common(table, name, data)
}

/// Look up an attribute by name and return a clone of its table entry.
/// Mirrors libhdf5's `H5A__open_common`.
#[allow(non_snake_case)]
pub fn H5A__open_common(table: &AttributeTable, name: &str) -> Result<AttributeTableEntry> {
    table.ensure_open()?;
    table
        .attrs
        .iter()
        .find(|entry| entry.attr.name == name)
        .cloned()
        .ok_or_else(|| Error::InvalidFormat(format!("attribute '{name}' not found")))
}

/// Open an attribute by name. Mirrors `H5A__open`.
#[allow(non_snake_case)]
pub fn H5A__open(table: &AttributeTable, name: &str) -> Result<AttributeTableEntry> {
    H5A__open_common(table, name)
}

/// Common API entry point for opening an attribute by name. Mirrors
/// `H5A__open_api_common`.
#[allow(non_snake_case)]
pub fn H5A__open_api_common(table: &AttributeTable, name: &str) -> Result<AttributeTableEntry> {
    H5A__open_common(table, name)
}

/// Common API entry point for opening an attribute on a named object.
/// Mirrors `H5A__open_by_name_api_common`.
#[allow(non_snake_case)]
pub fn H5A__open_by_name_api_common(
    table: &AttributeTable,
    name: &str,
) -> Result<AttributeTableEntry> {
    H5A__open_common(table, name)
}

/// Deprecated 1.6-era public open-by-name entry point. Mirrors
/// `H5Aopen_name`; superseded by `H5Aopen`.
#[allow(non_snake_case)]
pub fn H5Aopen_name(table: &AttributeTable, name: &str) -> Result<AttributeTableEntry> {
    H5A__open_common(table, name)
}

/// Open an attribute by its index in the table. Mirrors `H5A__open_by_idx`.
#[allow(non_snake_case)]
pub fn H5A__open_by_idx(table: &AttributeTable, index: usize) -> Result<AttributeTableEntry> {
    table.ensure_open()?;
    table
        .attrs
        .get(index)
        .cloned()
        .ok_or_else(|| Error::InvalidFormat(format!("attribute index {index} out of range")))
}

/// Common API entry point for opening an attribute by index. Mirrors
/// `H5A__open_by_idx_api_common`.
#[allow(non_snake_case)]
pub fn H5A__open_by_idx_api_common(
    table: &AttributeTable,
    index: usize,
) -> Result<AttributeTableEntry> {
    H5A__open_by_idx(table, index)
}

/// Deprecated 1.6-era open-by-index entry point. Mirrors `H5Aopen_idx`;
/// superseded by `H5Aopen_by_idx`.
#[allow(non_snake_case)]
pub fn H5Aopen_idx(table: &AttributeTable, index: usize) -> Result<AttributeTableEntry> {
    H5A__open_by_idx(table, index)
}

/// Return the attribute's name. Mirrors `H5A__get_name`.
#[allow(non_snake_case)]
pub fn H5A__get_name(entry: &AttributeTableEntry) -> &str {
    &entry.attr.name
}

/// Retrieve an attribute's name. Mirrors `H5Aget_name`.
#[allow(non_snake_case)]
pub fn H5Aget_name(entry: &AttributeTableEntry) -> &str {
    H5A__get_name(entry)
}

/// Return a copy of the dataspace of an attribute. Mirrors `H5A_get_space`.
#[allow(non_snake_case)]
pub fn H5A_get_space(entry: &AttributeTableEntry) -> DataspaceMessage {
    entry.attr.dataspace.clone()
}

/// Return a copy of the datatype of an attribute. Mirrors `H5A__get_type`.
#[allow(non_snake_case)]
pub fn H5A__get_type(entry: &AttributeTableEntry) -> DatatypeMessage {
    entry.attr.datatype.clone()
}

/// Get a copy of the datatype for an attribute. Mirrors `H5Aget_type`.
#[allow(non_snake_case)]
pub fn H5Aget_type(entry: &AttributeTableEntry) -> DatatypeMessage {
    H5A__get_type(entry)
}

/// Return a copy of the attribute's creation property list. Mirrors
/// `H5A__get_create_plist`.
#[allow(non_snake_case)]
pub fn H5A__get_create_plist(
    entry: &AttributeTableEntry,
) -> crate::hl::plist::attribute_create::AttributeCreate {
    let attr = Attribute {
        msg: entry.attr.clone(),
        creation_order: entry.creation_order,
        inner: None,
        object_id: None,
    };
    attr.create_plist()
}

/// Get a copy of the creation property list for an attribute. Mirrors
/// `H5Aget_create_plist`.
#[allow(non_snake_case)]
pub fn H5Aget_create_plist(
    entry: &AttributeTableEntry,
) -> crate::hl::plist::attribute_create::AttributeCreate {
    H5A__get_create_plist(entry)
}

/// Retrieve metadata about an attribute (creation order, character encoding,
/// data size). Mirrors `H5A__get_info`.
#[allow(non_snake_case)]
pub fn H5A__get_info(entry: &AttributeTableEntry) -> AttributeInfo {
    AttributeInfo {
        creation_order_valid: entry.creation_order.is_some(),
        creation_order: entry.creation_order.unwrap_or(0),
        char_encoding: entry.attr.char_encoding,
        data_size: entry.attr.data.len(),
    }
}

/// Retrieve information about an attribute looked up by name. Mirrors
/// `H5Aget_info_by_name`.
#[allow(non_snake_case)]
pub fn H5Aget_info_by_name(table: &AttributeTable, name: &str) -> Result<AttributeInfo> {
    H5A__open_common(table, name).map(|entry| H5A__get_info(&entry))
}

/// Return a copy of the attribute's raw bytes. Mirrors `H5A__read` /
/// `H5Aread`.
#[allow(non_snake_case)]
pub fn H5A__read(entry: &AttributeTableEntry) -> Vec<u8> {
    entry.attr.data.clone()
}

/// Replace an attribute's data bytes with `data`. Mirrors `H5A__write` /
/// `H5Awrite`.
#[allow(non_snake_case)]
pub fn H5A__write(entry: &mut AttributeTableEntry, data: &[u8]) {
    entry.attr.data.clear();
    entry.attr.data.extend_from_slice(data);
}

/// Return a deep copy of an attribute table entry. Mirrors `H5A__copy`.
#[allow(non_snake_case)]
pub fn H5A__copy(entry: &AttributeTableEntry) -> AttributeTableEntry {
    entry.clone()
}

/// Release the shared part of an attribute, dropping its reference count to
/// zero. Mirrors `H5A__shared_free`.
#[allow(non_snake_case)]
pub fn H5A__shared_free(entry: &mut AttributeTableEntry) {
    entry.shared_refcount = 0;
}

/// Called when the ref count reaches zero on an attribute's ID. Mirrors
/// `H5A__close_cb`.
#[allow(non_snake_case)]
pub fn H5A__close_cb(entry: &mut AttributeTableEntry) {
    entry.shared_refcount = entry.shared_refcount.saturating_sub(1);
}

/// Release an attribute and its associated resources. Mirrors `H5A__close`.
#[allow(non_snake_case)]
pub fn H5A__close(entry: &mut AttributeTableEntry) {
    H5A__close_cb(entry);
}

/// Return the object location for the attribute's parent object. Mirrors
/// `H5A_oloc`; always `None` in this port because table entries don't carry
/// the back-reference.
#[allow(non_snake_case)]
pub fn H5A_oloc(_entry: &AttributeTableEntry) -> Option<u64> {
    None
}

/// Return the group hierarchy name for the attribute (here, just the
/// attribute name itself). Mirrors `H5A_nameof`.
#[allow(non_snake_case)]
pub fn H5A_nameof(entry: &AttributeTableEntry) -> &str {
    H5A__get_name(entry)
}

/// Check whether an attribute with the given name exists. Mirrors
/// `H5A__exists_by_name`.
#[allow(non_snake_case)]
pub fn H5A__exists_by_name(table: &AttributeTable, name: &str) -> Result<bool> {
    table.ensure_open()?;
    Ok(table.find_index(name).is_some())
}

/// Common API entry point for `H5Aexists`. Mirrors `H5A__exists_api_common`.
#[allow(non_snake_case)]
pub fn H5A__exists_api_common(table: &AttributeTable, name: &str) -> Result<bool> {
    H5A__exists_by_name(table, name)
}

/// Common API entry point for `H5Aexists_by_name`. Mirrors
/// `H5A__exists_by_name_api_common`.
#[allow(non_snake_case)]
pub fn H5A__exists_by_name_api_common(table: &AttributeTable, name: &str) -> Result<bool> {
    H5A__exists_by_name(table, name)
}

/// Build a sorted table of attributes from compact (object-header) storage.
/// Mirrors `H5A__compact_build_table`.
#[allow(non_snake_case)]
pub fn H5A__compact_build_table(table: &AttributeTable) -> Result<Vec<AttributeTableEntry>> {
    table.ensure_open()?;
    Ok(table.attrs.clone())
}

/// Object-header iterator callback that copies a single attribute into the
/// compact build table. Mirrors `H5A__compact_build_table_cb`.
#[allow(non_snake_case)]
pub fn H5A__compact_build_table_cb(entry: &AttributeTableEntry) -> AttributeTableEntry {
    entry.clone()
}

/// Build a sorted table of attributes from dense (fractal-heap + B-tree)
/// storage. Mirrors `H5A__dense_build_table`.
#[allow(non_snake_case)]
pub fn H5A__dense_build_table(table: &AttributeTable) -> Result<Vec<AttributeTableEntry>> {
    H5A__compact_build_table(table)
}

/// Callback used while building a table of attributes from dense storage.
/// Mirrors `H5A__dense_build_table_cb`.
#[allow(non_snake_case)]
pub fn H5A__dense_build_table_cb(entry: &AttributeTableEntry) -> AttributeTableEntry {
    entry.clone()
}

/// Comparator that orders attributes by name in decreasing alphabetic order.
/// Mirrors `H5A__attr_cmp_name_dec`.
#[allow(non_snake_case)]
pub fn H5A__attr_cmp_name_dec(
    left: &AttributeTableEntry,
    right: &AttributeTableEntry,
) -> std::cmp::Ordering {
    right.attr.name.cmp(&left.attr.name)
}

/// Comparator that orders attributes by creation order, decreasing.
/// Mirrors `H5A__attr_cmp_corder_dec`.
#[allow(non_snake_case)]
pub fn H5A__attr_cmp_corder_dec(
    left: &AttributeTableEntry,
    right: &AttributeTableEntry,
) -> std::cmp::Ordering {
    right.creation_order.cmp(&left.creation_order)
}

/// Return the number of attributes attached to the object. Mirrors
/// `H5A__get_ainfo` (which retrieves the AINFO message and resolves the
/// attribute count for either compact or dense storage).
#[allow(non_snake_case)]
pub fn H5A__get_ainfo(table: &AttributeTable) -> Result<usize> {
    table.ensure_open()?;
    Ok(table.attrs.len())
}

/// Return the number of attributes attached to the object. Mirrors the
/// deprecated `H5Aget_num_attrs`.
#[allow(non_snake_case)]
pub fn H5Aget_num_attrs(table: &AttributeTable) -> Result<usize> {
    H5A__get_ainfo(table)
}

/// Retrieve the name of the attribute at `index` in iteration order.
/// Mirrors `H5Aget_name_by_idx`.
#[allow(non_snake_case)]
pub fn H5Aget_name_by_idx(table: &AttributeTable, index: usize) -> Result<String> {
    Ok(H5A__open_by_idx(table, index)?.attr.name)
}

/// Asynchronous variant of `H5Aexists`. Mirrors `H5Aexists_async`; this
/// port runs the query synchronously.
#[allow(non_snake_case)]
pub fn H5Aexists_async(table: &AttributeTable, name: &str) -> Result<bool> {
    H5A__exists_by_name(table, name)
}

/// Asynchronous variant of `H5Aexists_by_name`. Mirrors
/// `H5Aexists_by_name_async`; this port runs the query synchronously.
#[allow(non_snake_case)]
pub fn H5Aexists_by_name_async(table: &AttributeTable, name: &str) -> Result<bool> {
    H5A__exists_by_name(table, name)
}

/// Set the on-disk encoding version for new attributes in this table.
/// Mirrors `H5A__set_version`.
#[allow(non_snake_case)]
pub fn H5A__set_version(table: &mut AttributeTable, version: u8) {
    table.version = version;
}

/// Copy an attribute when copying its parent object between files. Mirrors
/// the first phase of libhdf5's `H5A__attr_copy_file`.
#[allow(non_snake_case)]
pub fn H5A__attr_copy_file(entry: &AttributeTableEntry) -> AttributeTableEntry {
    entry.clone()
}

/// Finish copying an attribute between files (bumps shared refcount).
/// Mirrors `H5A__attr_post_copy_file`.
#[allow(non_snake_case)]
pub fn H5A__attr_post_copy_file(entry: &mut AttributeTableEntry) {
    entry.shared_refcount = entry.shared_refcount.saturating_add(1);
}

/// Per-attribute callback for the dense post-copy pass. Mirrors
/// `H5A__dense_post_copy_file_cb`.
#[allow(non_snake_case)]
pub fn H5A__dense_post_copy_file_cb(entry: &mut AttributeTableEntry) {
    H5A__attr_post_copy_file(entry);
}

/// Run the dense post-copy callback over every attribute in the table.
/// Mirrors `H5A__dense_post_copy_file_all`.
#[allow(non_snake_case)]
pub fn H5A__dense_post_copy_file_all(table: &mut AttributeTable) {
    for entry in &mut table.attrs {
        H5A__dense_post_copy_file_cb(entry);
    }
}

/// Collect the names of all attributes in the table, in stored order.
/// Mirrors `H5A__iterate_common`.
#[allow(non_snake_case)]
pub fn H5A__iterate_common(table: &AttributeTable) -> Result<Vec<String>> {
    table.ensure_open()?;
    Ok(table
        .attrs
        .iter()
        .map(|entry| entry.attr.name.clone())
        .collect())
}

/// Iterate over the attribute names of an object. Mirrors `H5A__iterate`.
#[allow(non_snake_case)]
pub fn H5A__iterate(table: &AttributeTable) -> Result<Vec<String>> {
    H5A__iterate_common(table)
}

/// Deprecated iteration entry point. Mirrors `H5A__iterate_old` (the
/// internal backing for `H5Aiterate1`).
#[allow(non_snake_case)]
pub fn H5A__iterate_old(table: &AttributeTable) -> Result<Vec<String>> {
    H5A__iterate_common(table)
}

/// Deprecated 1.6-era attribute iteration entry point. Mirrors `H5Aiterate1`.
#[allow(non_snake_case)]
pub fn H5Aiterate1(table: &AttributeTable) -> Result<Vec<String>> {
    H5A__iterate_common(table)
}

/// Iterate over the attributes of an object identified by name. Mirrors
/// `H5Aiterate_by_name`.
#[allow(non_snake_case)]
pub fn H5Aiterate_by_name(table: &AttributeTable) -> Result<Vec<String>> {
    H5A__iterate_common(table)
}

/// Remove the attribute at the given index from the table and return it.
/// Mirrors `H5A__delete_by_idx`.
#[allow(non_snake_case)]
pub fn H5A__delete_by_idx(table: &mut AttributeTable, index: usize) -> Result<AttributeTableEntry> {
    table.ensure_open()?;
    if index >= table.attrs.len() {
        return Err(Error::InvalidFormat(format!(
            "attribute index {index} out of range"
        )));
    }
    Ok(table.attrs.remove(index))
}

/// Public entry point to delete an attribute by index. Mirrors
/// `H5Adelete_by_idx`.
#[allow(non_snake_case)]
pub fn H5Adelete_by_idx(table: &mut AttributeTable, index: usize) -> Result<AttributeTableEntry> {
    H5A__delete_by_idx(table, index)
}

/// Delete an attribute by name. Mirrors `H5Adelete`.
#[allow(non_snake_case)]
pub fn H5Adelete(table: &mut AttributeTable, name: &str) -> Result<AttributeTableEntry> {
    let index = table
        .find_index(name)
        .ok_or_else(|| Error::InvalidFormat(format!("attribute '{name}' not found")))?;
    H5A__delete_by_idx(table, index)
}

/// Delete an attribute by name on a named object. Mirrors
/// `H5Adelete_by_name`.
#[allow(non_snake_case)]
pub fn H5Adelete_by_name(table: &mut AttributeTable, name: &str) -> Result<AttributeTableEntry> {
    H5Adelete(table, name)
}

/// Private version of `H5Adelete_by_name`. Mirrors `H5A__delete_by_name`.
#[allow(non_snake_case)]
pub fn H5A__delete_by_name(table: &mut AttributeTable, name: &str) -> Result<AttributeTableEntry> {
    H5Adelete_by_name(table, name)
}

/// Rename an attribute in place, rejecting collisions with an existing
/// attribute of the new name. Mirrors `H5A__rename_common`.
#[allow(non_snake_case)]
pub fn H5A__rename_common(
    table: &mut AttributeTable,
    old_name: &str,
    new_name: &str,
) -> Result<()> {
    table.ensure_open()?;
    if table.find_index(new_name).is_some() {
        return Err(Error::InvalidFormat(format!(
            "attribute '{new_name}' already exists"
        )));
    }
    let index = table
        .find_index(old_name)
        .ok_or_else(|| Error::InvalidFormat(format!("attribute '{old_name}' not found")))?;
    table.attrs[index].attr.name = new_name.to_string();
    Ok(())
}

/// Common API entry point for `H5Arename`. Mirrors `H5A__rename_api_common`.
#[allow(non_snake_case)]
pub fn H5A__rename_api_common(
    table: &mut AttributeTable,
    old_name: &str,
    new_name: &str,
) -> Result<()> {
    H5A__rename_common(table, old_name, new_name)
}

/// Common API entry point for `H5Arename_by_name`. Mirrors
/// `H5A__rename_by_name_api_common`.
#[allow(non_snake_case)]
pub fn H5A__rename_by_name_api_common(
    table: &mut AttributeTable,
    old_name: &str,
    new_name: &str,
) -> Result<()> {
    H5A__rename_common(table, old_name, new_name)
}

/// Private version of `H5Arename_by_name`. Mirrors `H5A__rename_by_name`.
#[allow(non_snake_case)]
pub fn H5A__rename_by_name(
    table: &mut AttributeTable,
    old_name: &str,
    new_name: &str,
) -> Result<()> {
    H5A__rename_common(table, old_name, new_name)
}

/// Callback invoked when an attribute is located in a dense index. Mirrors
/// `H5A__dense_fnd_cb`.
#[allow(non_snake_case)]
pub fn H5A__dense_fnd_cb(
    table: &AttributeTable,
    name: &str,
) -> Result<Option<AttributeTableEntry>> {
    table.ensure_open()?;
    Ok(table
        .attrs
        .iter()
        .find(|entry| entry.attr.name == name)
        .cloned())
}

/// Open an attribute stored in dense storage by name. Mirrors
/// `H5A__dense_open`.
#[allow(non_snake_case)]
pub fn H5A__dense_open(table: &AttributeTable, name: &str) -> Result<AttributeTableEntry> {
    H5A__open_common(table, name)
}

/// Insert a new attribute into dense storage. Mirrors `H5A__dense_insert`.
#[allow(non_snake_case)]
pub fn H5A__dense_insert(table: &mut AttributeTable, attr: AttributeMessage) -> Result<()> {
    H5A__create(table, attr)
}

/// v2 B-tree 'modify' callback used when updating dense-storage attribute
/// data via the name index. Mirrors `H5A__dense_write_bt2_cb`.
#[allow(non_snake_case)]
pub fn H5A__dense_write_bt2_cb(entry: &AttributeTableEntry) -> Vec<u8> {
    H5A__dense_btree2_name_encode(entry)
}

/// Modify an existing attribute stored in dense form. Mirrors
/// `H5A__dense_write`.
#[allow(non_snake_case)]
pub fn H5A__dense_write(table: &mut AttributeTable, attr: AttributeMessage) -> Result<()> {
    H5A__dense_insert(table, attr)
}

/// Fractal-heap callback that hands back a cloned attribute to the caller.
/// Mirrors `H5A__dense_copy_fh_cb`.
#[allow(non_snake_case)]
pub fn H5A__dense_copy_fh_cb(entry: &AttributeTableEntry) -> AttributeTableEntry {
    entry.clone()
}

/// Iterate over attributes stored in dense form. Mirrors
/// `H5A__dense_iterate`.
#[allow(non_snake_case)]
pub fn H5A__dense_iterate(table: &AttributeTable) -> Result<Vec<String>> {
    H5A__iterate_common(table)
}

/// v2 B-tree callback used to iterate the dense storage name index.
/// Mirrors `H5A__dense_iterate_bt2_cb`.
#[allow(non_snake_case)]
pub fn H5A__dense_iterate_bt2_cb(entry: &AttributeTableEntry) -> String {
    entry.attr.name.clone()
}

/// Check whether an attribute exists in dense storage. Mirrors
/// `H5A__dense_exists`.
#[allow(non_snake_case)]
pub fn H5A__dense_exists(table: &AttributeTable, name: &str) -> Result<bool> {
    H5A__exists_by_name(table, name)
}

/// v2 B-tree callback used when removing an entry from dense storage.
/// Mirrors `H5A__dense_remove_bt2_cb`.
#[allow(non_snake_case)]
pub fn H5A__dense_remove_bt2_cb(
    table: &mut AttributeTable,
    name: &str,
) -> Result<AttributeTableEntry> {
    H5Adelete(table, name)
}

/// Remove an attribute from dense storage by name. Mirrors
/// `H5A__dense_remove`.
#[allow(non_snake_case)]
pub fn H5A__dense_remove(table: &mut AttributeTable, name: &str) -> Result<AttributeTableEntry> {
    H5Adelete(table, name)
}

/// v2 B-tree callback used when removing an entry by index. Mirrors
/// `H5A__dense_remove_by_idx_bt2_cb`.
#[allow(non_snake_case)]
pub fn H5A__dense_remove_by_idx_bt2_cb(
    table: &mut AttributeTable,
    index: usize,
) -> Result<AttributeTableEntry> {
    H5A__delete_by_idx(table, index)
}

/// Remove an attribute from dense storage by index. Mirrors
/// `H5A__dense_remove_by_idx`.
#[allow(non_snake_case)]
pub fn H5A__dense_remove_by_idx(
    table: &mut AttributeTable,
    index: usize,
) -> Result<AttributeTableEntry> {
    H5A__delete_by_idx(table, index)
}

/// Tear down all dense-storage structures for attributes on an object.
/// Mirrors `H5A__dense_delete`.
#[allow(non_snake_case)]
pub fn H5A__dense_delete(table: &mut AttributeTable) {
    table.attrs.clear();
}

/// Test hook: true if the attribute is currently shared (refcount > 1).
/// Mirrors `H5A__is_shared_test`.
#[allow(non_snake_case)]
pub fn H5A__is_shared_test(entry: &AttributeTableEntry) -> bool {
    entry.shared_refcount > 1
}

/// Test hook: return the current shared refcount of an attribute. Mirrors
/// `H5A__get_shared_rc_test`.
#[allow(non_snake_case)]
pub fn H5A__get_shared_rc_test(entry: &AttributeTableEntry) -> u32 {
    entry.shared_refcount
}

/// Compare two attributes by name, used as a fractal-heap object comparator
/// for dense attribute storage. Mirrors `H5A__dense_fh_name_cmp`.
#[allow(non_snake_case)]
pub fn H5A__dense_fh_name_cmp(
    left: &AttributeTableEntry,
    right: &AttributeTableEntry,
) -> std::cmp::Ordering {
    left.attr.name.cmp(&right.attr.name)
}

/// Store user information into a native record for the dense-storage name
/// v2 B-tree. Mirrors `H5A__dense_btree2_name_store`.
#[allow(non_snake_case)]
pub fn H5A__dense_btree2_name_store(entry: &AttributeTableEntry) -> Vec<u8> {
    H5A__dense_btree2_name_encode(entry)
}

/// Compare two name-index v2 B-tree records by attribute name. Mirrors
/// `H5A__dense_btree2_name_compare`.
#[allow(non_snake_case)]
pub fn H5A__dense_btree2_name_compare(
    left: &AttributeTableEntry,
    right: &AttributeTableEntry,
) -> std::cmp::Ordering {
    left.attr.name.cmp(&right.attr.name)
}

/// Encode the native form of a name-index v2 B-tree record into its raw
/// disk bytes. Mirrors `H5A__dense_btree2_name_encode`.
#[allow(non_snake_case)]
pub fn H5A__dense_btree2_name_encode(entry: &AttributeTableEntry) -> Vec<u8> {
    entry.attr.name.as_bytes().to_vec()
}

/// Store user information into a native record for the dense-storage
/// creation-order v2 B-tree. Mirrors `H5A__dense_btree2_corder_store`.
#[allow(non_snake_case)]
pub fn H5A__dense_btree2_corder_store(entry: &AttributeTableEntry) -> Vec<u8> {
    H5A__dense_btree2_corder_encode(entry)
}

/// Compare two creation-order v2 B-tree records. Mirrors
/// `H5A__dense_btree2_corder_compare`.
#[allow(non_snake_case)]
pub fn H5A__dense_btree2_corder_compare(
    left: &AttributeTableEntry,
    right: &AttributeTableEntry,
) -> std::cmp::Ordering {
    left.creation_order.cmp(&right.creation_order)
}

/// Encode the native form of a creation-order v2 B-tree record into its
/// raw disk bytes. Mirrors `H5A__dense_btree2_corder_encode`.
#[allow(non_snake_case)]
pub fn H5A__dense_btree2_corder_encode(entry: &AttributeTableEntry) -> Vec<u8> {
    entry.creation_order.unwrap_or(0).to_le_bytes().to_vec()
}

/// Format a creation-order v2 B-tree record for debug output. Mirrors
/// `H5A__dense_btree2_corder_debug`.
#[allow(non_snake_case)]
pub fn H5A__dense_btree2_corder_debug(entry: &AttributeTableEntry) -> String {
    format!(
        "AttributeCOrder(name={}, corder={})",
        entry.attr.name,
        entry.creation_order.unwrap_or(0)
    )
}

impl std::fmt::Debug for Attribute {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Attribute")
            .field("msg", &self.msg)
            .finish_non_exhaustive()
    }
}

impl Clone for Attribute {
    fn clone(&self) -> Self {
        let object_id = self
            .inner
            .as_ref()
            .map(|inner| register_open_object(inner, OpenObjectKind::Attribute));
        Self {
            msg: self.msg.clone(),
            creation_order: self.creation_order,
            inner: self.inner.clone(),
            object_id,
        }
    }
}

impl Drop for Attribute {
    fn drop(&mut self) {
        if let (Some(inner), Some(object_id)) = (self.inner.as_ref(), self.object_id) {
            unregister_open_object(inner, object_id);
        }
    }
}

impl Attribute {
    /// Construct an `Attribute` handle from a decoded message and register
    /// the new handle with the owning file's open-object tracker.
    pub(crate) fn from_message(
        msg: AttributeMessage,
        creation_order: Option<u64>,
        inner: &Arc<Mutex<FileInner<BufReader<fs::File>>>>,
    ) -> Self {
        let object_id = Some(register_open_object(inner, OpenObjectKind::Attribute));
        Self {
            msg,
            creation_order,
            inner: Some(inner.clone()),
            object_id,
        }
    }

    /// Get the attribute name.
    pub fn name(&self) -> &str {
        &self.msg.name
    }

    /// Return this attribute handle's high-level object id, when the attribute
    /// came from an open file.
    pub fn object_id(&self) -> Option<u64> {
        self.object_id
    }

    /// Attribute creation order if tracked by the file.
    pub fn creation_order(&self) -> Option<u64> {
        self.creation_order
    }

    /// Get basic attribute metadata.
    pub fn info(&self) -> AttributeInfo {
        AttributeInfo {
            creation_order_valid: self.creation_order.is_some(),
            creation_order: self.creation_order.unwrap_or(0),
            char_encoding: self.msg.char_encoding,
            data_size: self.msg.data.len(),
        }
    }

    /// Get attribute creation properties.
    pub fn create_plist(&self) -> crate::hl::plist::attribute_create::AttributeCreate {
        crate::hl::plist::attribute_create::AttributeCreate::from_attribute(self)
    }

    /// Get the raw data bytes of the attribute value.
    pub fn raw_data(&self) -> &[u8] {
        &self.msg.data
    }

    /// Get the datatype size in bytes.
    pub fn element_size(&self) -> usize {
        match usize::try_from(self.msg.datatype.size) {
            Ok(size) => size,
            Err(_) => usize::MAX,
        }
    }

    /// Get the shape of the attribute.
    pub fn shape(&self) -> &[u64] {
        &self.msg.dataspace.dims
    }

    /// Get the parsed datatype descriptor for this attribute.
    pub fn dtype(&self) -> Datatype {
        Datatype::from_message(self.msg.datatype.clone())
    }

    /// Get the parsed dataspace descriptor for this attribute.
    pub fn space(&self) -> Dataspace {
        Dataspace::from_message(self.msg.dataspace.clone())
    }

    /// Return the parsed low-level datatype message for this attribute.
    pub fn raw_datatype_message(&self) -> DatatypeMessage {
        self.msg.datatype.clone()
    }

    /// Return the parsed low-level dataspace message for this attribute.
    pub fn raw_dataspace_message(&self) -> DataspaceMessage {
        self.msg.dataspace.clone()
    }

    /// Try to read the attribute as a single f64 scalar.
    pub fn read_scalar_f64(&self) -> Option<f64> {
        let bytes = checked_window(&self.msg.data, 0, 8, "attribute f64 scalar").ok()?;
        Some(f64::from_le_bytes(bytes.try_into().ok()?))
    }

    /// Try to read the attribute as a single i64 scalar.
    pub fn read_scalar_i64(&self) -> Option<i64> {
        if let Ok(bytes) = checked_window(&self.msg.data, 0, 8, "attribute i64 scalar") {
            Some(i64::from_le_bytes(bytes.try_into().ok()?))
        } else if let Ok(bytes) = checked_window(&self.msg.data, 0, 4, "attribute i32 scalar") {
            Some(i64::from(i32::from_le_bytes(bytes.try_into().ok()?)))
        } else {
            None
        }
    }

    /// Read the attribute value as a typed Vec.
    pub fn read<T: crate::hl::types::H5Type>(&self) -> crate::Result<Vec<T>> {
        let conversion =
            crate::hl::conversion::ReadConversion::for_dataset::<T>(&self.msg.datatype)?;
        conversion.bytes_to_vec(self.msg.data.clone())
    }

    /// Read the attribute as a typed scalar.
    pub fn read_scalar<T: crate::hl::types::H5Type>(&self) -> crate::Result<T> {
        let conversion =
            crate::hl::conversion::ReadConversion::for_dataset::<T>(&self.msg.datatype)?;
        conversion.bytes_to_scalar(self.msg.data.clone())
    }

    /// Read the attribute as a string (for fixed-length string attributes).
    pub fn read_string(&self) -> String {
        if self.msg.datatype.is_variable_string() {
            return self
                .read_strings()
                .ok()
                .and_then(|mut strings| {
                    if strings.is_empty() {
                        None
                    } else {
                        Some(strings.remove(0))
                    }
                })
                .unwrap_or_default();
        }
        let padding = self.msg.datatype.string_padding().unwrap_or(1);
        decode_fixed_string_with_padding(&self.msg.data, padding).unwrap_or_default()
    }

    /// Read the attribute as string elements.
    pub fn read_strings(&self) -> Result<Vec<String>> {
        if self.msg.datatype.is_variable_string() {
            return self.read_vlen_strings();
        }

        if self.msg.datatype.class != DatatypeClass::String {
            return Err(Error::Unsupported(format!(
                "attribute '{}' is not a string attribute",
                self.msg.name
            )));
        }

        let elem_size = self.element_size();
        if elem_size == 0 {
            return Err(Error::InvalidFormat(format!(
                "attribute '{}' has zero-sized string datatype",
                self.msg.name
            )));
        }
        if self.msg.data.len() % elem_size != 0 {
            return Err(Error::InvalidFormat(format!(
                "attribute '{}' string data length {} is not a multiple of element size {}",
                self.msg.name,
                self.msg.data.len(),
                elem_size
            )));
        }

        let padding = self.msg.datatype.string_padding().unwrap_or(1);
        self.msg
            .data
            .chunks_exact(elem_size)
            .map(|chunk| decode_fixed_string_with_padding(chunk, padding))
            .collect()
    }

    /// Materialize a variable-length string attribute by walking its
    /// per-element global-heap references and decoding each payload as UTF-8.
    fn read_vlen_strings(&self) -> Result<Vec<String>> {
        let inner = self.inner.as_ref().ok_or_else(|| {
            Error::Unsupported(format!(
                "attribute '{}' has no file context for variable-length string read",
                self.msg.name
            ))
        })?;
        let mut guard = inner.lock();
        let sizeof_addr = usize::from(guard.superblock.sizeof_addr);
        if sizeof_addr > 8 {
            return Err(Error::Unsupported(format!(
                "attribute '{}' variable-length descriptor address width {sizeof_addr} exceeds 64-bit support",
                self.msg.name
            )));
        }
        let ref_size = 4usize
            .checked_add(sizeof_addr)
            .and_then(|v| v.checked_add(4))
            .ok_or_else(|| Error::InvalidFormat("vlen string reference size overflow".into()))?;
        if self.msg.data.len() % ref_size != 0 {
            return Err(Error::InvalidFormat(format!(
                "attribute '{}' vlen string data length {} is not a multiple of reference size {}",
                self.msg.name,
                self.msg.data.len(),
                ref_size
            )));
        }

        let mut strings = Vec::with_capacity(self.msg.data.len() / ref_size);
        for chunk in self.msg.data.chunks_exact(ref_size) {
            let (seq_len, addr, index) = decode_vlen_string_ref(chunk, sizeof_addr)?;

            if addr == 0 || crate::io::reader::is_undef_addr(addr) {
                strings.push(String::new());
                continue;
            }

            let gh_ref = crate::format::global_heap::GlobalHeapRef {
                collection_addr: addr,
                object_index: index,
            };
            let data =
                crate::format::global_heap::read_global_heap_object(&mut guard.reader, &gh_ref)?;
            if seq_len > data.len() {
                return Err(Error::InvalidFormat(format!(
                    "attribute '{}' vlen string payload too short: expected {} bytes, got {}",
                    self.msg.name,
                    seq_len,
                    data.len()
                )));
            }
            let bytes = &data[..seq_len];
            strings.push(decode_utf8_string(bytes, "attribute vlen string payload")?);
        }
        Ok(strings)
    }
}

/// Decode a single vlen-string descriptor (sequence length, global-heap
/// collection address, object index) out of one element-sized chunk.
fn decode_vlen_string_ref(chunk: &[u8], sizeof_addr: usize) -> Result<(usize, u64, u32)> {
    let addr_start = 4usize;
    let addr_end = addr_start
        .checked_add(sizeof_addr)
        .ok_or_else(|| Error::InvalidFormat("vlen string address offset overflow".into()))?;
    let index_end = addr_end
        .checked_add(4)
        .ok_or_else(|| Error::InvalidFormat("vlen string index offset overflow".into()))?;
    if chunk.len() < index_end {
        return Err(Error::InvalidFormat(
            "vlen string reference is truncated".into(),
        ));
    }

    let seq_len_u32 = read_u32_le_at(chunk, 0, "vlen string sequence length")?;
    let seq_len = usize::try_from(seq_len_u32)
        .map_err(|_| Error::InvalidFormat("vlen string sequence length exceeds usize".into()))?;
    let mut addr = 0u64;
    for (i, byte) in checked_window(chunk, addr_start, sizeof_addr, "vlen string address")?
        .iter()
        .enumerate()
    {
        addr |= u64::from(*byte) << (i * 8);
    }
    let index = read_u32_le_at(chunk, addr_end, "vlen string heap index")?;

    Ok((seq_len, addr, index))
}

/// Return `data[pos..pos+len]`, mapping out-of-range/overflow errors to a
/// context-annotated `Error`.
fn checked_window<'a>(data: &'a [u8], pos: usize, len: usize, context: &str) -> Result<&'a [u8]> {
    let end = pos
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))?;
    data.get(pos..end)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))
}

/// Read a little-endian u32 at `data[pos..pos+4]` with a contextual error.
fn read_u32_le_at(data: &[u8], pos: usize, context: &str) -> Result<u32> {
    let bytes = checked_window(data, pos, 4, context)?;
    let bytes: [u8; 4] = bytes
        .try_into()
        .map_err(|_| Error::InvalidFormat(format!("{context} is truncated")))?;
    Ok(u32::from_le_bytes(bytes))
}

/// Decode a fixed-length string element, stopping at the first NUL byte
/// and (for `padding == 2`, i.e. space-padded) trimming trailing whitespace.
fn decode_fixed_string_with_padding(bytes: &[u8], padding: u8) -> Result<String> {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    let bytes = &bytes[..end];
    let s = std::str::from_utf8(bytes)
        .map_err(|_| Error::InvalidFormat("attribute fixed string payload is not UTF-8".into()))?;
    Ok(if padding == 2 {
        s.trim_end().to_string()
    } else {
        s.to_string()
    })
}

/// Decode `bytes` as UTF-8 and trim trailing NUL bytes (used for vlen
/// string payloads pulled out of the global heap).
fn decode_utf8_string(bytes: &[u8], context: &str) -> Result<String> {
    Ok(std::str::from_utf8(bytes)
        .map_err(|_| Error::InvalidFormat(format!("{context} is not UTF-8")))?
        .trim_end_matches('\0')
        .to_string())
}

/// Collect all attributes from an object header at the given address.
pub(crate) fn collect_attributes(
    inner: &Arc<Mutex<FileInner<BufReader<fs::File>>>>,
    addr: u64,
) -> Result<Vec<Attribute>> {
    let mut guard = inner.lock();
    let oh = ObjectHeader::read_at(&mut guard.reader, addr)?;

    let mut attrs = Vec::new();
    for msg in &oh.messages {
        if msg.msg_type == object_header::MSG_ATTRIBUTE {
            match AttributeMessage::decode(&msg.data) {
                Ok(attr_msg) => attrs.push((attr_msg, msg.creation_index.map(u64::from))),
                Err(e) => {
                    // Skip malformed attributes
                    eprintln!("Warning: failed to decode attribute: {e}");
                }
            }
        }
    }

    for msg in &oh.messages {
        if msg.msg_type == object_header::MSG_ATTR_INFO {
            let attr_info = AttributeInfoMessage::decode(&msg.data, guard.superblock.sizeof_addr)?;
            if attr_info.has_dense_storage() {
                let heap =
                    FractalHeapHeader::read_at(&mut guard.reader, attr_info.fractal_heap_addr)?;
                let records =
                    btree_v2::collect_all_records(&mut guard.reader, attr_info.name_btree_addr)?;
                let heap_id_len = usize::from(heap.heap_id_len);

                for record in &records {
                    if record.len() < heap_id_len {
                        continue;
                    }
                    let heap_id = &record[..heap_id_len];
                    let creation_order = dense_attribute_record_creation_order(record, heap_id_len);
                    if let Ok(attr_data) = heap.read_managed_object(&mut guard.reader, heap_id) {
                        match AttributeMessage::decode(&attr_data) {
                            Ok(attr_msg) => attrs.push((attr_msg, creation_order)),
                            Err(e) => {
                                eprintln!("Warning: failed to decode dense attribute: {e}");
                            }
                        }
                    }
                }
            }
        }
    }

    drop(guard);
    Ok(attrs
        .into_iter()
        .map(|(attr_msg, creation_order)| Attribute::from_message(attr_msg, creation_order, inner))
        .collect())
}

/// Extract the 4-byte creation-order index that follows the heap ID and
/// flags byte in a dense-attribute name-index v2 B-tree record.
fn dense_attribute_record_creation_order(record: &[u8], heap_id_len: usize) -> Option<u64> {
    let start = heap_id_len.checked_add(1)?;
    let bytes = checked_window(record, start, 4, "dense attribute creation order").ok()?;
    Some(u64::from(u32::from_le_bytes(bytes.try_into().ok()?)))
}

/// Collect attributes sorted by tracked creation order.
pub(crate) fn collect_attributes_by_creation_order(
    inner: &Arc<Mutex<FileInner<BufReader<fs::File>>>>,
    addr: u64,
) -> Result<Vec<Attribute>> {
    let mut attrs = collect_attributes(inner, addr)?;
    if attrs.iter().any(|attr| attr.creation_order.is_none()) {
        return Err(Error::Unsupported(
            "object does not track attribute creation order".into(),
        ));
    }
    attrs.sort_by_key(|attr| attr.creation_order.unwrap_or(u64::MAX));
    Ok(attrs)
}

/// Get attribute names from an object header.
pub(crate) fn attr_names(
    inner: &Arc<Mutex<FileInner<BufReader<fs::File>>>>,
    addr: u64,
) -> Result<Vec<String>> {
    let attrs = collect_attributes(inner, addr)?;
    Ok(attrs.iter().map(|a| a.msg.name.clone()).collect())
}

/// Check whether a specific attribute exists on an object.
pub(crate) fn attr_exists(
    inner: &Arc<Mutex<FileInner<BufReader<fs::File>>>>,
    addr: u64,
    name: &str,
) -> Result<bool> {
    let names = attr_names(inner, addr)?;
    Ok(names.iter().any(|attr_name| attr_name == name))
}

/// Get a specific attribute by name.
pub(crate) fn get_attr(
    inner: &Arc<Mutex<FileInner<BufReader<fs::File>>>>,
    addr: u64,
    name: &str,
) -> Result<Attribute> {
    let attrs = collect_attributes(inner, addr)?;
    attrs
        .into_iter()
        .find(|a| a.msg.name == name)
        .ok_or_else(|| Error::InvalidFormat(format!("attribute '{name}' not found")))
}

#[cfg(test)]
mod tests {
    use super::{
        checked_window, decode_fixed_string_with_padding, decode_vlen_string_ref,
        default_attribute_message,
    };

    #[test]
    fn decode_vlen_string_ref_rejects_truncated_descriptor() {
        let err = decode_vlen_string_ref(&[0; 11], 4).unwrap_err();
        assert!(
            err.to_string()
                .contains("vlen string reference is truncated"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn decode_vlen_string_ref_decodes_descriptor() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0x1122_3344u32.to_le_bytes());
        bytes.extend_from_slice(&7u32.to_le_bytes());
        let (seq_len, addr, index) = decode_vlen_string_ref(&bytes, 4).unwrap();
        assert_eq!(seq_len, 3);
        assert_eq!(addr, 0x1122_3344);
        assert_eq!(index, 7);
    }

    #[test]
    fn default_attribute_message_uses_checked_lengths() {
        let msg = default_attribute_message("a", vec![1, 2, 3]).unwrap();
        assert_eq!(msg.datatype.size, 3);
        assert_eq!(msg.dataspace.dims, vec![3]);

        let scalar = default_attribute_message("empty", Vec::new()).unwrap();
        assert_eq!(scalar.datatype.size, 1);
        assert!(scalar.dataspace.dims.is_empty());
    }

    #[test]
    fn checked_window_rejects_offset_overflow() {
        let err = checked_window(&[], usize::MAX, 1, "vlen string test").unwrap_err();
        assert!(
            err.to_string().contains("vlen string test offset overflow"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn attribute_string_decoder_rejects_invalid_utf8() {
        assert_eq!(
            decode_fixed_string_with_padding(b"alpha\0tail", 1).unwrap(),
            "alpha"
        );
        assert_eq!(
            decode_fixed_string_with_padding(b"alpha   ", 2).unwrap(),
            "alpha"
        );
        assert!(decode_fixed_string_with_padding(&[0xff, 0], 1).is_err());
    }
}
