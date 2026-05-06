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
    pub fn new() -> Self {
        Self {
            attrs: Vec::new(),
            version: 3,
            closed: false,
        }
    }

    fn ensure_open(&self) -> Result<()> {
        if self.closed {
            Err(Error::InvalidFormat("attribute table is closed".into()))
        } else {
            Ok(())
        }
    }

    fn find_index(&self, name: &str) -> Option<usize> {
        self.attrs.iter().position(|entry| entry.attr.name == name)
    }

    fn next_creation_order(&self) -> u64 {
        self.attrs
            .iter()
            .filter_map(|entry| entry.creation_order)
            .max()
            .map_or(0, |value| value.saturating_add(1))
    }
}

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

#[allow(non_snake_case)]
pub fn H5A_init() -> bool {
    true
}

#[allow(non_snake_case)]
pub fn H5A__init_package() -> bool {
    H5A_init()
}

#[allow(non_snake_case)]
pub fn H5A_top_term_package() {}

#[allow(non_snake_case)]
pub fn H5A_term_package() {}

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

#[allow(non_snake_case)]
pub fn H5A__create_by_name(table: &mut AttributeTable, name: &str, data: Vec<u8>) -> Result<()> {
    H5A__create(table, default_attribute_message(name, data)?)
}

#[allow(non_snake_case)]
pub fn H5A__create_api_common(table: &mut AttributeTable, name: &str, data: Vec<u8>) -> Result<()> {
    H5A__create_by_name(table, name, data)
}

#[allow(non_snake_case)]
pub fn H5A__create_by_name_api_common(
    table: &mut AttributeTable,
    name: &str,
    data: Vec<u8>,
) -> Result<()> {
    H5A__create_by_name(table, name, data)
}

#[allow(non_snake_case)]
pub fn H5Acreate1(table: &mut AttributeTable, name: &str, data: Vec<u8>) -> Result<()> {
    H5A__create_api_common(table, name, data)
}

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

#[allow(non_snake_case)]
pub fn H5A__open(table: &AttributeTable, name: &str) -> Result<AttributeTableEntry> {
    H5A__open_common(table, name)
}

#[allow(non_snake_case)]
pub fn H5A__open_api_common(table: &AttributeTable, name: &str) -> Result<AttributeTableEntry> {
    H5A__open_common(table, name)
}

#[allow(non_snake_case)]
pub fn H5A__open_by_name_api_common(
    table: &AttributeTable,
    name: &str,
) -> Result<AttributeTableEntry> {
    H5A__open_common(table, name)
}

#[allow(non_snake_case)]
pub fn H5Aopen_name(table: &AttributeTable, name: &str) -> Result<AttributeTableEntry> {
    H5A__open_common(table, name)
}

#[allow(non_snake_case)]
pub fn H5A__open_by_idx(table: &AttributeTable, index: usize) -> Result<AttributeTableEntry> {
    table.ensure_open()?;
    table
        .attrs
        .get(index)
        .cloned()
        .ok_or_else(|| Error::InvalidFormat(format!("attribute index {index} out of range")))
}

#[allow(non_snake_case)]
pub fn H5A__open_by_idx_api_common(
    table: &AttributeTable,
    index: usize,
) -> Result<AttributeTableEntry> {
    H5A__open_by_idx(table, index)
}

#[allow(non_snake_case)]
pub fn H5Aopen_idx(table: &AttributeTable, index: usize) -> Result<AttributeTableEntry> {
    H5A__open_by_idx(table, index)
}

#[allow(non_snake_case)]
pub fn H5A__get_name(entry: &AttributeTableEntry) -> &str {
    &entry.attr.name
}

#[allow(non_snake_case)]
pub fn H5A_get_space(entry: &AttributeTableEntry) -> DataspaceMessage {
    entry.attr.dataspace.clone()
}

#[allow(non_snake_case)]
pub fn H5A__get_type(entry: &AttributeTableEntry) -> DatatypeMessage {
    entry.attr.datatype.clone()
}

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

#[allow(non_snake_case)]
pub fn H5A__get_info(entry: &AttributeTableEntry) -> AttributeInfo {
    AttributeInfo {
        creation_order_valid: entry.creation_order.is_some(),
        creation_order: entry.creation_order.unwrap_or(0),
        char_encoding: entry.attr.char_encoding,
        data_size: entry.attr.data.len(),
    }
}

#[allow(non_snake_case)]
pub fn H5A__copy(entry: &AttributeTableEntry) -> AttributeTableEntry {
    entry.clone()
}

#[allow(non_snake_case)]
pub fn H5A__shared_free(entry: &mut AttributeTableEntry) {
    entry.shared_refcount = 0;
}

#[allow(non_snake_case)]
pub fn H5A__close_cb(entry: &mut AttributeTableEntry) {
    entry.shared_refcount = entry.shared_refcount.saturating_sub(1);
}

#[allow(non_snake_case)]
pub fn H5A__close(entry: &mut AttributeTableEntry) {
    H5A__close_cb(entry);
}

#[allow(non_snake_case)]
pub fn H5A_oloc(_entry: &AttributeTableEntry) -> Option<u64> {
    None
}

#[allow(non_snake_case)]
pub fn H5A_nameof(entry: &AttributeTableEntry) -> &str {
    H5A__get_name(entry)
}

#[allow(non_snake_case)]
pub fn H5A__exists_by_name(table: &AttributeTable, name: &str) -> Result<bool> {
    table.ensure_open()?;
    Ok(table.find_index(name).is_some())
}

#[allow(non_snake_case)]
pub fn H5A__exists_api_common(table: &AttributeTable, name: &str) -> Result<bool> {
    H5A__exists_by_name(table, name)
}

#[allow(non_snake_case)]
pub fn H5A__exists_by_name_api_common(table: &AttributeTable, name: &str) -> Result<bool> {
    H5A__exists_by_name(table, name)
}

#[allow(non_snake_case)]
pub fn H5A__compact_build_table(table: &AttributeTable) -> Result<Vec<AttributeTableEntry>> {
    table.ensure_open()?;
    Ok(table.attrs.clone())
}

#[allow(non_snake_case)]
pub fn H5A__dense_build_table(table: &AttributeTable) -> Result<Vec<AttributeTableEntry>> {
    H5A__compact_build_table(table)
}

#[allow(non_snake_case)]
pub fn H5A__attr_cmp_name_dec(
    left: &AttributeTableEntry,
    right: &AttributeTableEntry,
) -> std::cmp::Ordering {
    right.attr.name.cmp(&left.attr.name)
}

#[allow(non_snake_case)]
pub fn H5A__attr_cmp_corder_dec(
    left: &AttributeTableEntry,
    right: &AttributeTableEntry,
) -> std::cmp::Ordering {
    right.creation_order.cmp(&left.creation_order)
}

#[allow(non_snake_case)]
pub fn H5A__get_ainfo(table: &AttributeTable) -> Result<usize> {
    table.ensure_open()?;
    Ok(table.attrs.len())
}

#[allow(non_snake_case)]
pub fn H5A__set_version(table: &mut AttributeTable, version: u8) {
    table.version = version;
}

#[allow(non_snake_case)]
pub fn H5A__attr_copy_file(entry: &AttributeTableEntry) -> AttributeTableEntry {
    entry.clone()
}

#[allow(non_snake_case)]
pub fn H5A__attr_post_copy_file(entry: &mut AttributeTableEntry) {
    entry.shared_refcount = entry.shared_refcount.saturating_add(1);
}

#[allow(non_snake_case)]
pub fn H5A__dense_post_copy_file_cb(entry: &mut AttributeTableEntry) {
    H5A__attr_post_copy_file(entry);
}

#[allow(non_snake_case)]
pub fn H5A__dense_post_copy_file_all(table: &mut AttributeTable) {
    for entry in &mut table.attrs {
        H5A__dense_post_copy_file_cb(entry);
    }
}

#[allow(non_snake_case)]
pub fn H5A__iterate_common(table: &AttributeTable) -> Result<Vec<String>> {
    table.ensure_open()?;
    Ok(table
        .attrs
        .iter()
        .map(|entry| entry.attr.name.clone())
        .collect())
}

#[allow(non_snake_case)]
pub fn H5A__iterate(table: &AttributeTable) -> Result<Vec<String>> {
    H5A__iterate_common(table)
}

#[allow(non_snake_case)]
pub fn H5A__iterate_old(table: &AttributeTable) -> Result<Vec<String>> {
    H5A__iterate_common(table)
}

#[allow(non_snake_case)]
pub fn H5Aiterate1(table: &AttributeTable) -> Result<Vec<String>> {
    H5A__iterate_common(table)
}

#[allow(non_snake_case)]
pub fn H5Aiterate_by_name(table: &AttributeTable) -> Result<Vec<String>> {
    H5A__iterate_common(table)
}

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

#[allow(non_snake_case)]
pub fn H5Adelete_by_idx(table: &mut AttributeTable, index: usize) -> Result<AttributeTableEntry> {
    H5A__delete_by_idx(table, index)
}

#[allow(non_snake_case)]
pub fn H5Adelete(table: &mut AttributeTable, name: &str) -> Result<AttributeTableEntry> {
    let index = table
        .find_index(name)
        .ok_or_else(|| Error::InvalidFormat(format!("attribute '{name}' not found")))?;
    H5A__delete_by_idx(table, index)
}

#[allow(non_snake_case)]
pub fn H5Adelete_by_name(table: &mut AttributeTable, name: &str) -> Result<AttributeTableEntry> {
    H5Adelete(table, name)
}

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

#[allow(non_snake_case)]
pub fn H5A__rename_api_common(
    table: &mut AttributeTable,
    old_name: &str,
    new_name: &str,
) -> Result<()> {
    H5A__rename_common(table, old_name, new_name)
}

#[allow(non_snake_case)]
pub fn H5A__rename_by_name_api_common(
    table: &mut AttributeTable,
    old_name: &str,
    new_name: &str,
) -> Result<()> {
    H5A__rename_common(table, old_name, new_name)
}

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

#[allow(non_snake_case)]
pub fn H5A__dense_open(table: &AttributeTable, name: &str) -> Result<AttributeTableEntry> {
    H5A__open_common(table, name)
}

#[allow(non_snake_case)]
pub fn H5A__dense_insert(table: &mut AttributeTable, attr: AttributeMessage) -> Result<()> {
    H5A__create(table, attr)
}

#[allow(non_snake_case)]
pub fn H5A__dense_write_bt2_cb(entry: &AttributeTableEntry) -> Vec<u8> {
    H5A__dense_btree2_name_encode(entry)
}

#[allow(non_snake_case)]
pub fn H5A__dense_write(table: &mut AttributeTable, attr: AttributeMessage) -> Result<()> {
    H5A__dense_insert(table, attr)
}

#[allow(non_snake_case)]
pub fn H5A__dense_copy_fh_cb(entry: &AttributeTableEntry) -> AttributeTableEntry {
    entry.clone()
}

#[allow(non_snake_case)]
pub fn H5A__dense_iterate(table: &AttributeTable) -> Result<Vec<String>> {
    H5A__iterate_common(table)
}

#[allow(non_snake_case)]
pub fn H5A__dense_remove_bt2_cb(
    table: &mut AttributeTable,
    name: &str,
) -> Result<AttributeTableEntry> {
    H5Adelete(table, name)
}

#[allow(non_snake_case)]
pub fn H5A__dense_remove(table: &mut AttributeTable, name: &str) -> Result<AttributeTableEntry> {
    H5Adelete(table, name)
}

#[allow(non_snake_case)]
pub fn H5A__dense_remove_by_idx_bt2_cb(
    table: &mut AttributeTable,
    index: usize,
) -> Result<AttributeTableEntry> {
    H5A__delete_by_idx(table, index)
}

#[allow(non_snake_case)]
pub fn H5A__dense_remove_by_idx(
    table: &mut AttributeTable,
    index: usize,
) -> Result<AttributeTableEntry> {
    H5A__delete_by_idx(table, index)
}

#[allow(non_snake_case)]
pub fn H5A__dense_delete(table: &mut AttributeTable) {
    table.attrs.clear();
}

#[allow(non_snake_case)]
pub fn H5A__is_shared_test(entry: &AttributeTableEntry) -> bool {
    entry.shared_refcount > 1
}

#[allow(non_snake_case)]
pub fn H5A__get_shared_rc_test(entry: &AttributeTableEntry) -> u32 {
    entry.shared_refcount
}

#[allow(non_snake_case)]
pub fn H5A__dense_fh_name_cmp(
    left: &AttributeTableEntry,
    right: &AttributeTableEntry,
) -> std::cmp::Ordering {
    left.attr.name.cmp(&right.attr.name)
}

#[allow(non_snake_case)]
pub fn H5A__dense_btree2_name_store(entry: &AttributeTableEntry) -> Vec<u8> {
    H5A__dense_btree2_name_encode(entry)
}

#[allow(non_snake_case)]
pub fn H5A__dense_btree2_name_compare(
    left: &AttributeTableEntry,
    right: &AttributeTableEntry,
) -> std::cmp::Ordering {
    left.attr.name.cmp(&right.attr.name)
}

#[allow(non_snake_case)]
pub fn H5A__dense_btree2_name_encode(entry: &AttributeTableEntry) -> Vec<u8> {
    entry.attr.name.as_bytes().to_vec()
}

#[allow(non_snake_case)]
pub fn H5A__dense_btree2_corder_store(entry: &AttributeTableEntry) -> Vec<u8> {
    H5A__dense_btree2_corder_encode(entry)
}

#[allow(non_snake_case)]
pub fn H5A__dense_btree2_corder_compare(
    left: &AttributeTableEntry,
    right: &AttributeTableEntry,
) -> std::cmp::Ordering {
    left.creation_order.cmp(&right.creation_order)
}

#[allow(non_snake_case)]
pub fn H5A__dense_btree2_corder_encode(entry: &AttributeTableEntry) -> Vec<u8> {
    entry.creation_order.unwrap_or(0).to_le_bytes().to_vec()
}

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

fn checked_window<'a>(data: &'a [u8], pos: usize, len: usize, context: &str) -> Result<&'a [u8]> {
    let end = pos
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))?;
    data.get(pos..end)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))
}

fn read_u32_le_at(data: &[u8], pos: usize, context: &str) -> Result<u32> {
    let bytes = checked_window(data, pos, 4, context)?;
    let bytes: [u8; 4] = bytes
        .try_into()
        .map_err(|_| Error::InvalidFormat(format!("{context} is truncated")))?;
    Ok(u32::from_le_bytes(bytes))
}

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
