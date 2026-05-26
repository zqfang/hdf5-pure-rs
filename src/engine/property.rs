use std::borrow::Cow;
use std::collections::BTreeMap;

use crate::error::{Error, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Property {
    pub name: String,
    pub value: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PropertyClass {
    pub name: String,
    pub parent: Option<String>,
    properties: BTreeMap<String, Property>,
    closed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PropertyList {
    pub class_name: String,
    properties: BTreeMap<String, Property>,
    closed: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HdfsFaplConfig {
    pub namenode_name: String,
    pub namenode_port: u16,
    pub user_name: String,
    pub buffer_size: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Ros3FaplConfig {
    pub endpoint: Option<String>,
    pub region: Option<String>,
    pub token: Option<String>,
}

impl PropertyClass {
    /// Construct an empty property class with the given name and optional parent.
    pub fn new(name: impl Into<String>, parent: Option<String>) -> Self {
        Self {
            name: name.into(),
            parent,
            properties: BTreeMap::new(),
            closed: false,
        }
    }

    /// Return an error if the property class has been closed.
    fn ensure_open(&self) -> Result<()> {
        if self.closed {
            Err(Error::InvalidFormat("property class is closed".into()))
        } else {
            Ok(())
        }
    }
}

impl PropertyList {
    /// Construct a new property list from the registered properties of a class.
    pub fn new(class: &PropertyClass) -> Self {
        Self {
            class_name: class.name.clone(),
            properties: class.properties.clone(),
            closed: false,
        }
    }

    /// Return an error if the property list has been closed.
    fn ensure_open(&self) -> Result<()> {
        if self.closed {
            Err(Error::InvalidFormat("property list is closed".into()))
        } else {
            Ok(())
        }
    }
}

/// Initialize the property-list interface from another layer; must be followed by [`H5P_init_phase2`].
#[allow(non_snake_case)]
pub fn H5P_init_phase1() -> bool {
    true
}

/// Finish initializing the property-list interface from another package.
#[allow(non_snake_case)]
pub fn H5P_init_phase2() -> bool {
    true
}

/// Initialize property-list interface-specific information.
#[allow(non_snake_case)]
pub fn H5P__init_package() -> bool {
    H5P_init_phase1() && H5P_init_phase2()
}

/// Terminate the property-list package, releasing resources.
#[allow(non_snake_case)]
pub fn H5P_term_package() {}

/// Create a new property-list class.
#[allow(non_snake_case)]
pub fn H5Pcreate_class(name: impl Into<String>, parent: Option<String>) -> PropertyClass {
    PropertyClass::new(name, parent)
}

/// Internal routine to create a new property-list class.
#[allow(non_snake_case)]
pub fn H5P__create_class(name: impl Into<String>, parent: Option<String>) -> PropertyClass {
    H5Pcreate_class(name, parent)
}

/// Called when the reference count reaches zero on a property-class ID.
#[allow(non_snake_case)]
pub fn H5P__close_class_cb(class: &mut PropertyClass) {
    class.closed = true;
    class.properties.clear();
}

/// Internal routine to close a property-list class.
#[allow(non_snake_case)]
pub fn H5P__close_class(class: &mut PropertyClass) {
    H5P__close_class_cb(class);
}

/// Increment or decrement list/class dependencies on a property class.
#[allow(non_snake_case)]
pub fn H5P__access_class(class: &PropertyClass) -> Result<&PropertyClass> {
    class.ensure_open()?;
    Ok(class)
}

/// Internal routine to create a new property list of a given class.
#[allow(non_snake_case)]
pub fn H5P__create(class: &PropertyClass) -> Result<PropertyList> {
    H5P__access_class(class)?;
    Ok(PropertyList::new(class))
}

/// Create a new property list and return an identifier for it.
#[allow(non_snake_case)]
pub fn H5P_create_id(class: &PropertyClass) -> Result<PropertyList> {
    H5P__create(class)
}

/// Create a new property list of a given type.
#[allow(non_snake_case)]
pub fn H5P__new_plist_of_type(class: &PropertyClass) -> Result<PropertyList> {
    H5P__create(class)
}

/// Called when the reference count reaches zero on a property-list ID.
#[allow(non_snake_case)]
pub fn H5P__close_list_cb(list: &mut PropertyList) {
    list.closed = true;
    list.properties.clear();
}

/// Close a property list, releasing its resources.
#[allow(non_snake_case)]
pub fn H5P_close(list: &mut PropertyList) {
    H5P__close_list_cb(list);
}

/// Allocate a new property record with a name and value buffer.
#[allow(non_snake_case)]
pub fn H5P__create_prop(name: impl Into<String>, value: impl Into<Vec<u8>>) -> Property {
    Property {
        name: name.into(),
        value: value.into(),
    }
}

/// Add a property record to a property class.
#[allow(non_snake_case)]
pub fn H5P__add_prop(class: &mut PropertyClass, prop: Property) -> Result<()> {
    class.ensure_open()?;
    class.properties.insert(prop.name.clone(), prop);
    Ok(())
}

/// Internal routine that actually registers a new property on a class.
#[allow(non_snake_case)]
pub fn H5P__register_real(
    class: &mut PropertyClass,
    name: impl Into<String>,
    value: impl Into<Vec<u8>>,
) -> Result<()> {
    H5P__add_prop(class, H5P__create_prop(name, value))
}

/// Register a new property on a property class.
#[allow(non_snake_case)]
pub fn H5P__register(
    class: &mut PropertyClass,
    name: impl Into<String>,
    value: impl Into<Vec<u8>>,
) -> Result<()> {
    H5P__register_real(class, name, value)
}

/// Register a property on a property class (v1 deprecated API).
#[deprecated(note = "use H5P__register")]
#[allow(non_snake_case)]
pub fn H5Pregister1(
    class: &mut PropertyClass,
    name: impl Into<String>,
    value: impl Into<Vec<u8>>,
) -> Result<()> {
    H5P__register(class, name, value)
}

/// Insert a property into an existing property list.
#[allow(non_snake_case)]
pub fn H5P_insert(
    list: &mut PropertyList,
    name: impl Into<String>,
    value: impl Into<Vec<u8>>,
) -> Result<()> {
    list.ensure_open()?;
    let prop = H5P__create_prop(name, value);
    list.properties.insert(prop.name.clone(), prop);
    Ok(())
}

/// Insert a property into a property list (v1 deprecated API).
#[deprecated(note = "use H5P_insert")]
#[allow(non_snake_case)]
pub fn H5Pinsert1(
    list: &mut PropertyList,
    name: impl Into<String>,
    value: impl Into<Vec<u8>>,
) -> Result<()> {
    H5P_insert(list, name, value)
}

/// Duplicate a property record.
#[allow(non_snake_case)]
pub fn H5P__dup_prop(prop: &Property) -> Property {
    prop.clone()
}

/// Copy a property from one property class to another.
#[allow(non_snake_case)]
pub fn H5P__copy_prop_pclass(
    src: &PropertyClass,
    dst: &mut PropertyClass,
    name: &str,
) -> Result<()> {
    let prop = H5P__find_prop_pclass(src, name)?.clone();
    H5P__add_prop(dst, prop)
}

/// Duplicate a property class.
#[allow(non_snake_case)]
pub fn H5P__copy_pclass(class: &PropertyClass) -> PropertyClass {
    class.clone()
}

/// Make a deep copy of a property list.
#[allow(non_snake_case)]
pub fn H5P_copy_plist(list: &PropertyList) -> Result<PropertyList> {
    list.ensure_open()?;
    Ok(list.clone())
}

/// Public API to copy a property list.
#[allow(non_snake_case)]
pub fn H5Pcopy(list: &PropertyList) -> Result<PropertyList> {
    H5P_copy_plist(list)
}

/// Look up a property by name in a property list.
#[allow(non_snake_case)]
pub fn H5P__find_prop_plist<'a>(list: &'a PropertyList, name: &str) -> Result<&'a Property> {
    list.ensure_open()?;
    list.properties
        .get(name)
        .ok_or_else(|| Error::InvalidFormat(format!("property '{name}' not found")))
}

/// Look up a property by name in a property class.
#[allow(non_snake_case)]
pub fn H5P__find_prop_pclass<'a>(class: &'a PropertyClass, name: &str) -> Result<&'a Property> {
    class.ensure_open()?;
    class
        .properties
        .get(name)
        .ok_or_else(|| Error::InvalidFormat(format!("property '{name}' not found")))
}

/// Clear a property record in place, releasing its name and value buffers.
#[allow(non_snake_case)]
pub fn H5P__free_prop(prop: &mut Property) {
    prop.name.clear();
    prop.value.clear();
}

/// Property callback that consumes (drops) a property record.
#[allow(non_snake_case)]
pub fn H5P__free_prop_cb(_prop: Property) {}

/// Callback that consumes a deleted property name.
#[allow(non_snake_case)]
pub fn H5P__free_del_name_cb(_name: String) {}

/// Property iteration callback returning the name and value of a property.
#[allow(non_snake_case)]
pub fn H5P__do_prop_cb1(prop: &Property) -> (&str, &[u8]) {
    (&prop.name, &prop.value)
}

/// Read a property value out of a property list as bytes.
#[allow(non_snake_case)]
pub fn H5P__do_prop_ref<'a>(list: &'a PropertyList, name: &str) -> Result<&'a [u8]> {
    Ok(H5P__find_prop_plist(list, name)?.value.as_slice())
}

/// Read a property value out of a property list as owned bytes.
#[deprecated(note = "use H5P__do_prop_ref to borrow the property value without cloning")]
#[allow(non_snake_case)]
pub fn H5P__do_prop(list: &PropertyList, name: &str) -> Result<Vec<u8>> {
    Ok(H5P__do_prop_ref(list, name)?.to_vec())
}

/// Callback that overwrites a property value on a property list.
#[allow(non_snake_case)]
pub fn H5P__poke_plist_cb(list: &mut PropertyList, name: &str, value: Vec<u8>) -> Result<()> {
    H5P_set(list, name, value)
}

/// Callback that overwrites a property value on a property class.
#[allow(non_snake_case)]
pub fn H5P__poke_pclass_cb(class: &mut PropertyClass, name: &str, value: Vec<u8>) -> Result<()> {
    H5P__register(class, name, value)
}

/// Forcibly write a property value into a property list.
#[allow(non_snake_case)]
pub fn H5P_poke(list: &mut PropertyList, name: &str, value: Vec<u8>) -> Result<()> {
    H5P__poke_plist_cb(list, name, value)
}

/// Set the value of an already-registered property on a property list.
#[allow(non_snake_case)]
pub fn H5P_set(list: &mut PropertyList, name: &str, value: Vec<u8>) -> Result<()> {
    list.ensure_open()?;
    let prop = list
        .properties
        .get_mut(name)
        .ok_or_else(|| Error::InvalidFormat(format!("property '{name}' not found")))?;
    prop.value = value;
    Ok(())
}

/// Read the value of a registered property on a property class.
#[allow(non_snake_case)]
pub fn H5P__class_get_ref<'a>(class: &'a PropertyClass, name: &str) -> Result<&'a [u8]> {
    Ok(H5P__find_prop_pclass(class, name)?.value.as_slice())
}

/// Read the value of a registered property on a property class.
#[deprecated(note = "use H5P__class_get_ref to borrow the property value without cloning")]
#[allow(non_snake_case)]
pub fn H5P__class_get(class: &PropertyClass, name: &str) -> Result<Vec<u8>> {
    Ok(H5P__class_get_ref(class, name)?.to_vec())
}

/// Set the value of a registered property on a property class.
#[allow(non_snake_case)]
pub fn H5P__class_set(class: &mut PropertyClass, name: &str, value: Vec<u8>) -> Result<()> {
    class.ensure_open()?;
    let prop = class
        .properties
        .get_mut(name)
        .ok_or_else(|| Error::InvalidFormat(format!("property '{name}' not found")))?;
    prop.value = value;
    Ok(())
}

/// Check whether a property exists on a property list.
#[allow(non_snake_case)]
pub fn H5P_exist_plist(list: &PropertyList, name: &str) -> Result<bool> {
    list.ensure_open()?;
    Ok(list.properties.contains_key(name))
}

/// Check whether a property exists on a property class.
#[allow(non_snake_case)]
pub fn H5P__exist_pclass(class: &PropertyClass, name: &str) -> bool {
    class.properties.contains_key(name)
}

/// Return the size in bytes of a property value on a property list.
#[allow(non_snake_case)]
pub fn H5P__get_size_plist(list: &PropertyList, name: &str) -> Result<usize> {
    Ok(H5P__find_prop_plist(list, name)?.value.len())
}

/// Return the size in bytes of a property value on a property class.
#[allow(non_snake_case)]
pub fn H5P__get_size_pclass(class: &PropertyClass, name: &str) -> Result<usize> {
    Ok(H5P__find_prop_pclass(class, name)?.value.len())
}

/// Return the number of registered properties on a property list.
#[allow(non_snake_case)]
pub fn H5P__get_nprops_plist(list: &PropertyList) -> Result<usize> {
    list.ensure_open()?;
    Ok(list.properties.len())
}

/// Return the number of registered properties on a property class.
#[allow(non_snake_case)]
pub fn H5P_get_nprops_pclass(class: &PropertyClass) -> usize {
    class.properties.len()
}

/// Compare two property records for equality.
#[allow(non_snake_case)]
pub fn H5P__cmp_prop(left: &Property, right: &Property) -> bool {
    left == right
}

/// Compare two property classes for equality.
#[allow(non_snake_case)]
pub fn H5P__cmp_class(left: &PropertyClass, right: &PropertyClass) -> bool {
    left.name == right.name && left.parent == right.parent && left.properties == right.properties
}

/// Callback that compares the properties of two lists for equality.
#[allow(non_snake_case)]
pub fn H5P__cmp_plist_cb(left: &PropertyList, right: &PropertyList) -> bool {
    left.properties == right.properties
}

/// Compare two property lists for equality.
#[allow(non_snake_case)]
pub fn H5P__cmp_plist(left: &PropertyList, right: &PropertyList) -> bool {
    left.class_name == right.class_name && H5P__cmp_plist_cb(left, right)
}

/// Return true if a class matches or descends from the given ancestor name.
#[allow(non_snake_case)]
pub fn H5P_class_isa(class: &PropertyClass, ancestor_name: &str) -> bool {
    class.name == ancestor_name || class.parent.as_deref() == Some(ancestor_name)
}

/// Iteration callback that returns the names of properties on a class.
#[allow(non_snake_case)]
pub fn H5P__iter_pclass_names(class: &PropertyClass) -> impl Iterator<Item = &str> {
    class.properties.keys().map(String::as_str)
}

/// Iteration callback that returns the names of properties on a class.
#[deprecated(note = "use H5P__iter_pclass_names or H5P__visit_pclass_names")]
#[allow(non_snake_case)]
pub fn H5P__iterate_pclass_cb(class: &PropertyClass) -> Vec<String> {
    H5P__iter_pclass_names(class).map(str::to_owned).collect()
}

/// Visit property names on a class without allocating a name list.
#[allow(non_snake_case)]
pub fn H5P__visit_pclass_names<F>(class: &PropertyClass, callback: F)
where
    F: FnMut(&str),
{
    H5P__iter_pclass_names(class).for_each(callback);
}

/// Append property names on a class into a caller-provided buffer.
#[allow(non_snake_case)]
pub fn H5P__iterate_pclass_into(class: &PropertyClass, names: &mut Vec<String>) {
    H5P__visit_pclass_names(class, |name| names.push(name.to_owned()));
}

/// Return the names of properties on a property class.
#[deprecated(
    note = "use H5P__iter_pclass_names, H5P__visit_pclass_names, or H5P__iterate_pclass_into"
)]
#[allow(non_snake_case)]
pub fn H5P__iterate_pclass(class: &PropertyClass) -> Vec<String> {
    H5P__iter_pclass_names(class).map(str::to_owned).collect()
}

/// Callback wrapper around [`H5P_peek`].
#[allow(non_snake_case)]
pub fn H5P__peek_cb_ref<'a>(list: &'a PropertyList, name: &str) -> Result<&'a [u8]> {
    H5P_peek_ref(list, name)
}

/// Callback wrapper around [`H5P_peek`].
#[deprecated(note = "use H5P__peek_cb_ref to borrow the property value without cloning")]
#[allow(non_snake_case)]
pub fn H5P__peek_cb(list: &PropertyList, name: &str) -> Result<Vec<u8>> {
    Ok(H5P__peek_cb_ref(list, name)?.to_vec())
}

/// Read a property value from a property list without removing it.
#[allow(non_snake_case)]
pub fn H5P_peek_ref<'a>(list: &'a PropertyList, name: &str) -> Result<&'a [u8]> {
    Ok(H5P__find_prop_plist(list, name)?.value.as_slice())
}

/// Read a property value from a property list without removing it.
#[deprecated(note = "use H5P_peek_ref to borrow the property value without cloning")]
#[allow(non_snake_case)]
pub fn H5P_peek(list: &PropertyList, name: &str) -> Result<Vec<u8>> {
    Ok(H5P_peek_ref(list, name)?.to_vec())
}

/// Remove a property from a property list by name.
#[allow(non_snake_case)]
pub fn H5P_remove(list: &mut PropertyList, name: &str) -> Result<Property> {
    list.ensure_open()?;
    list.properties
        .remove(name)
        .ok_or_else(|| Error::InvalidFormat(format!("property '{name}' not found")))
}

/// Remove a registered property from a property class.
#[allow(non_snake_case)]
pub fn H5P__unregister(class: &mut PropertyClass, name: &str) -> Result<Property> {
    class.ensure_open()?;
    class
        .properties
        .remove(name)
        .ok_or_else(|| Error::InvalidFormat(format!("property '{name}' not found")))
}

/// Build the slash-delimited class-path string for a property class.
#[allow(non_snake_case)]
pub fn H5P__get_class_path_cow(class: &PropertyClass) -> Cow<'_, str> {
    match &class.parent {
        Some(parent) => Cow::Owned(format!("{parent}/{}", class.name)),
        None => Cow::Borrowed(class.name.as_str()),
    }
}

/// Build the slash-delimited class-path string for a property class.
#[deprecated(note = "use H5P__get_class_path_cow to avoid cloning parentless class names")]
#[allow(non_snake_case)]
pub fn H5P__get_class_path(class: &PropertyClass) -> String {
    H5P__get_class_path_cow(class).into_owned()
}

/// Return true if the path or name of a class matches the given path string.
#[allow(non_snake_case)]
pub fn H5P__open_class_path(class: &PropertyClass, path: &str) -> bool {
    H5P__get_class_path_cow(class) == path || class.name == path
}

/// Test helper that returns [`H5P__get_class_path_cow`].
#[allow(non_snake_case)]
pub fn H5P__get_class_path_test(class: &PropertyClass) -> Cow<'_, str> {
    H5P__get_class_path_cow(class)
}

/// Test helper that returns [`H5P__open_class_path`].
#[allow(non_snake_case)]
pub fn H5P__open_class_path_test(class: &PropertyClass, path: &str) -> bool {
    H5P__open_class_path(class, path)
}

/// Return the parent class name of a property class.
#[allow(non_snake_case)]
pub fn H5P__get_class_parent(class: &PropertyClass) -> Option<&str> {
    class.parent.as_deref()
}

/// Return a copy of the property list (treated as an ID handle here).
#[allow(non_snake_case)]
pub fn H5P_get_plist_id(list: &PropertyList) -> Result<PropertyList> {
    H5P_copy_plist(list)
}

/// Return the class name of a property list.
#[allow(non_snake_case)]
pub fn H5P_get_class_ref(list: &PropertyList) -> &str {
    list.class_name.as_str()
}

/// Return the class name of a property list.
#[deprecated(note = "use H5P_get_class_ref to borrow the class name without cloning")]
#[allow(non_snake_case)]
pub fn H5P_get_class(list: &PropertyList) -> String {
    H5P_get_class_ref(list).to_owned()
}

/// Encode an unsigned integer property value as little-endian bytes.
#[allow(non_snake_case)]
pub fn H5P__encode_unsigned_into(value: u64, out: &mut Vec<u8>) {
    out.extend_from_slice(&value.to_le_bytes());
}

/// Encode an unsigned integer property value as little-endian bytes.
#[deprecated(note = "use H5P__encode_unsigned_into to reuse caller allocation")]
#[allow(non_snake_case)]
pub fn H5P__encode_unsigned(value: u64) -> Vec<u8> {
    value.to_le_bytes().to_vec()
}

/// Encode a `uint64_t` property value as little-endian bytes.
#[allow(non_snake_case)]
pub fn H5P__encode_uint64_t_into(value: u64, out: &mut Vec<u8>) {
    out.extend_from_slice(&value.to_le_bytes());
}

/// Encode a `uint64_t` property value as little-endian bytes.
#[deprecated(note = "use H5P__encode_uint64_t_into to reuse caller allocation")]
#[allow(non_snake_case)]
pub fn H5P__encode_uint64_t(value: u64) -> Vec<u8> {
    value.to_le_bytes().to_vec()
}

/// Encode a `size_t` property value, erroring if it does not fit in `u64`.
#[allow(non_snake_case)]
pub fn H5P__encode_size_t_into(value: usize, out: &mut Vec<u8>) -> Result<()> {
    H5P__encode_uint64_t_into(
        u64::try_from(value)
            .map_err(|_| Error::InvalidFormat("size_t property exceeds u64".into()))?,
        out,
    );
    Ok(())
}

/// Encode a `size_t` property value, erroring if it does not fit in `u64`.
#[deprecated(note = "use H5P__encode_size_t_into to reuse caller allocation")]
#[allow(non_snake_case)]
pub fn H5P__encode_size_t(value: usize) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(8);
    H5P__encode_size_t_into(value, &mut out)?;
    Ok(out)
}

/// Encode an `hsize_t` property value as little-endian bytes.
#[allow(non_snake_case)]
pub fn H5P__encode_hsize_t_into(value: u64, out: &mut Vec<u8>) {
    H5P__encode_uint64_t_into(value, out);
}

/// Encode an `hsize_t` property value as little-endian bytes.
#[deprecated(note = "use H5P__encode_hsize_t_into to reuse caller allocation")]
#[allow(non_snake_case)]
pub fn H5P__encode_hsize_t(value: u64) -> Vec<u8> {
    value.to_le_bytes().to_vec()
}

/// Encode a `double` property value as little-endian bytes.
#[allow(non_snake_case)]
pub fn H5P__encode_double_into(value: f64, out: &mut Vec<u8>) {
    out.extend_from_slice(&value.to_le_bytes());
}

/// Encode a `double` property value as little-endian bytes.
#[deprecated(note = "use H5P__encode_double_into to reuse caller allocation")]
#[allow(non_snake_case)]
pub fn H5P__encode_double(value: f64) -> Vec<u8> {
    value.to_le_bytes().to_vec()
}

/// Encode a `uint8_t` property value into a caller-provided buffer.
#[allow(non_snake_case)]
pub fn H5P__encode_uint8_t_into(value: u8, out: &mut Vec<u8>) {
    out.push(value);
}

/// Encode a `uint8_t` property value as a one-byte buffer.
#[deprecated(note = "use H5P__encode_uint8_t_into or stack array encoders")]
#[allow(non_snake_case)]
pub fn H5P__encode_uint8_t(value: u8) -> Vec<u8> {
    vec![value]
}

/// Encode a boolean property value into a caller-provided buffer.
#[allow(non_snake_case)]
pub fn H5P__encode_bool_into(value: bool, out: &mut Vec<u8>) {
    H5P__encode_uint8_t_into(u8::from(value), out);
}

/// Encode a boolean property value as a one-byte buffer.
#[deprecated(note = "use H5P__encode_bool_into or stack array encoders")]
#[allow(non_snake_case)]
pub fn H5P__encode_bool(value: bool) -> Vec<u8> {
    vec![u8::from(value)]
}

fn encode_uint8_bytes(value: u8) -> [u8; 1] {
    [value]
}

fn encode_bool_bytes(value: bool) -> [u8; 1] {
    [u8::from(value)]
}

fn encode_uint64_bytes(value: u64) -> [u8; 8] {
    value.to_le_bytes()
}

fn encode_double_bytes(value: f64) -> [u8; 8] {
    value.to_le_bytes()
}

/// Serialize a sequence of property chunks with length prefixes.
#[allow(non_snake_case)]
pub fn H5P__encode_slices_into<'a, I>(chunks: I, out: &mut Vec<u8>) -> Result<()>
where
    I: IntoIterator<Item = &'a [u8]>,
{
    for chunk in chunks {
        let additional = 8usize
            .checked_add(chunk.len())
            .ok_or_else(|| Error::InvalidFormat("property aggregate length overflow".into()))?;
        out.try_reserve(additional)
            .map_err(|_| Error::InvalidFormat("property aggregate length overflow".into()))?;
        encode_property_chunk(chunk, out)?;
    }
    Ok(())
}

fn encode_property_chunks_unreserved<'a, I>(chunks: I, out: &mut Vec<u8>) -> Result<()>
where
    I: IntoIterator<Item = &'a [u8]>,
{
    for chunk in chunks {
        encode_property_chunk(chunk, out)?;
    }
    Ok(())
}

fn encode_property_chunk(chunk: &[u8], out: &mut Vec<u8>) -> Result<()> {
    out.extend_from_slice(
        &u64::try_from(chunk.len())
            .map_err(|_| Error::InvalidFormat("property chunk length exceeds u64".into()))?
            .to_le_bytes(),
    );
    out.extend_from_slice(chunk);
    Ok(())
}

/// Serialize a sequence of property chunks with length prefixes.
#[allow(non_snake_case)]
pub fn H5P__encode_into(chunks: &[Vec<u8>], out: &mut Vec<u8>) -> Result<()> {
    let additional = encoded_slices_len(chunks.iter().map(Vec::as_slice))?;
    out.try_reserve(additional)
        .map_err(|_| Error::InvalidFormat("property aggregate length overflow".into()))?;
    encode_property_chunks_unreserved(chunks.iter().map(Vec::as_slice), out)
}

/// Serialize a sequence of property chunks with length prefixes.
#[deprecated(note = "use H5P__encode_into or H5P__encode_slices_into to reuse caller allocation")]
#[allow(non_snake_case)]
pub fn H5P__encode(chunks: &[Vec<u8>]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(encoded_slices_len(chunks.iter().map(Vec::as_slice))?);
    encode_property_chunks_unreserved(chunks.iter().map(Vec::as_slice), &mut out)?;
    Ok(out)
}

fn encoded_slices_len<'a, I>(chunks: I) -> Result<usize>
where
    I: IntoIterator<Item = &'a [u8]>,
{
    chunks.into_iter().try_fold(0usize, |len, chunk| {
        let chunk_len = u64::try_from(chunk.len())
            .map_err(|_| Error::InvalidFormat("property chunk length exceeds u64".into()))?;
        let chunk_len = usize::try_from(chunk_len).map_err(|_| {
            Error::InvalidFormat("property chunk length does not fit in usize".into())
        })?;
        len.checked_add(8)
            .and_then(|value| value.checked_add(chunk_len))
            .ok_or_else(|| Error::InvalidFormat("property aggregate length overflow".into()))
    })
}

/// Iterator over length-prefixed property chunks.
pub struct H5PDecodeChunks<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Iterator for H5PDecodeChunks<'a> {
    type Item = Result<&'a [u8]>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset >= self.bytes.len() {
            return None;
        }

        let len_u64 = match checked_window(self.bytes, self.offset, 8)
            .ok_or_else(|| Error::InvalidFormat("truncated property chunk length".into()))
            .and_then(H5P__decode_uint64_t)
        {
            Ok(len) => len,
            Err(err) => {
                self.offset = self.bytes.len();
                return Some(Err(err));
            }
        };
        if let Err(err) = advance_offset(&mut self.offset, 8, "property aggregate") {
            self.offset = self.bytes.len();
            return Some(Err(err));
        }

        let len = match usize::try_from(len_u64)
            .map_err(|_| Error::InvalidFormat("property chunk length does not fit in usize".into()))
        {
            Ok(len) => len,
            Err(err) => {
                self.offset = self.bytes.len();
                return Some(Err(err));
            }
        };

        let chunk = match checked_window(self.bytes, self.offset, len)
            .ok_or_else(|| Error::InvalidFormat("truncated property chunk payload".into()))
        {
            Ok(chunk) => chunk,
            Err(err) => {
                self.offset = self.bytes.len();
                return Some(Err(err));
            }
        };
        if let Err(err) = advance_offset(&mut self.offset, len, "property aggregate") {
            self.offset = self.bytes.len();
            return Some(Err(err));
        }
        Some(Ok(chunk))
    }
}

/// Borrow decoded length-prefixed property chunks from a serialized buffer.
#[allow(non_snake_case)]
pub fn H5P__decode_chunks(bytes: &[u8]) -> H5PDecodeChunks<'_> {
    H5PDecodeChunks { bytes, offset: 0 }
}

/// Visit decoded length-prefixed property chunks without allocating a chunk list.
#[allow(non_snake_case)]
pub fn H5P__decode_with<F>(bytes: &[u8], mut callback: F) -> Result<()>
where
    F: FnMut(&[u8]),
{
    for chunk in H5P__decode_chunks(bytes) {
        callback(chunk?);
    }
    Ok(())
}

/// Append decoded length-prefixed property chunks into a caller-provided buffer.
#[allow(non_snake_case)]
pub fn H5P__decode_into(bytes: &[u8], chunks: &mut Vec<Vec<u8>>) -> Result<()> {
    chunks
        .try_reserve(decoded_chunk_count(bytes)?)
        .map_err(|_| Error::InvalidFormat("property chunk count overflow".into()))?;
    H5P__decode_with(bytes, |chunk| chunks.push(chunk.to_vec()))
}

fn decoded_chunk_count(bytes: &[u8]) -> Result<usize> {
    let mut count = 0usize;
    for chunk in H5P__decode_chunks(bytes) {
        chunk?;
        count = count
            .checked_add(1)
            .ok_or_else(|| Error::InvalidFormat("property chunk count overflow".into()))?;
    }
    Ok(count)
}

/// Deserialize a sequence of length-prefixed property chunks.
#[deprecated(note = "use H5P__decode_chunks, H5P__decode_with, or H5P__decode_into")]
#[allow(non_snake_case)]
pub fn H5P__decode(bytes: &[u8]) -> Result<Vec<Vec<u8>>> {
    let mut chunks = Vec::new();
    H5P__decode_into(bytes, &mut chunks)?;
    Ok(chunks)
}

/// Decode a `uint8_t` property value from bytes.
#[allow(non_snake_case)]
pub fn H5P__decode_uint8_t(bytes: &[u8]) -> Result<u8> {
    bytes
        .first()
        .copied()
        .ok_or_else(|| Error::InvalidFormat("missing u8 property".into()))
}

/// Decode a boolean property value from bytes.
#[allow(non_snake_case)]
pub fn H5P__decode_bool(bytes: &[u8]) -> Result<bool> {
    Ok(H5P__decode_uint8_t(bytes)? != 0)
}

/// Decode a `uint64_t` property value from little-endian bytes.
#[allow(non_snake_case)]
pub fn H5P__decode_uint64_t(bytes: &[u8]) -> Result<u64> {
    let raw: [u8; 8] = bytes
        .get(..8)
        .ok_or_else(|| Error::InvalidFormat("truncated u64 property".into()))?
        .try_into()
        .map_err(|_| Error::InvalidFormat("truncated u64 property".into()))?;
    Ok(u64::from_le_bytes(raw))
}

/// Decode a `size_t` property value, erroring if it does not fit `usize`.
#[allow(non_snake_case)]
pub fn H5P__decode_size_t(bytes: &[u8]) -> Result<usize> {
    usize::try_from(H5P__decode_uint64_t(bytes)?)
        .map_err(|_| Error::InvalidFormat("size_t property does not fit usize".into()))
}

/// Decode an `hsize_t` property value from little-endian bytes.
#[allow(non_snake_case)]
pub fn H5P__decode_hsize_t(bytes: &[u8]) -> Result<u64> {
    H5P__decode_uint64_t(bytes)
}

/// Decode a `double` property value from little-endian bytes.
#[allow(non_snake_case)]
pub fn H5P__decode_double(bytes: &[u8]) -> Result<f64> {
    let raw: [u8; 8] = bytes
        .get(..8)
        .ok_or_else(|| Error::InvalidFormat("truncated double property".into()))?
        .try_into()
        .map_err(|_| Error::InvalidFormat("truncated double property".into()))?;
    Ok(f64::from_le_bytes(raw))
}

/// Internal helper to write a property value into a list.
fn plist_set(list: &mut PropertyList, name: &str, value: Vec<u8>) -> Result<()> {
    list.ensure_open()?;
    list.properties.insert(
        name.to_string(),
        Property {
            name: name.to_string(),
            value,
        },
    );
    Ok(())
}

/// Internal helper to read a property value from a list.
fn plist_get_ref<'a>(list: &'a PropertyList, name: &str) -> Result<&'a [u8]> {
    list.ensure_open()?;
    Ok(list
        .properties
        .get(name)
        .map(|prop| prop.value.as_slice())
        .unwrap_or_default())
}

/// Borrow a property value from a list, returning an empty slice if absent.
#[allow(non_snake_case)]
pub fn H5P__plist_get_ref<'a>(list: &'a PropertyList, name: &str) -> Result<&'a [u8]> {
    plist_get_ref(list, name)
}

/// Append a property value from a list into a caller-provided buffer.
#[allow(non_snake_case)]
pub fn H5P__plist_get_into(list: &PropertyList, name: &str, out: &mut Vec<u8>) -> Result<()> {
    out.extend_from_slice(plist_get_ref(list, name)?);
    Ok(())
}

macro_rules! owned_property_get_wrapper {
    ($owned:ident, $borrowed:ident) => {
        #[deprecated(note = "use the _ref or _into variant to avoid cloning")]
        #[allow(non_snake_case)]
        pub fn $owned(list: &PropertyList) -> Result<Vec<u8>> {
            Ok($borrowed(list)?.to_vec())
        }
    };
}

/// Internal helper to remove a property from a list.
fn plist_del(list: &mut PropertyList, name: &str) -> Result<()> {
    list.ensure_open()?;
    list.properties.remove(name);
    Ok(())
}

fn plist_clear_fapl_driver_config(list: &mut PropertyList) -> Result<()> {
    plist_del(list, "fapl_hdfs_config")?;
    plist_del(list, "fapl_ros3_config")
}

/// Internal helper that compares two property value buffers.
fn prop_cmp(left: &[u8], right: &[u8]) -> bool {
    left == right
}

/// Return an [`Error::Unsupported`] for a driver name not implemented by the pure-Rust backend.
fn unsupported_driver(name: &str) -> Result<()> {
    Err(Error::Unsupported(format!(
        "{name} driver is not supported by the pure Rust local backend"
    )))
}

/// Encode an optional UTF-8 string as a presence flag, length, and payload.
fn encode_optional_string(out: &mut Vec<u8>, value: Option<&str>, context: &str) -> Result<()> {
    match value {
        Some(value) => {
            let len = u32::try_from(value.len()).map_err(|_| {
                Error::InvalidFormat(format!("{context} string length exceeds u32"))
            })?;
            out.push(1);
            out.extend_from_slice(&len.to_le_bytes());
            out.extend_from_slice(value.as_bytes());
        }
        None => {
            out.push(0);
            out.extend_from_slice(&0u32.to_le_bytes());
        }
    }
    Ok(())
}

fn encoded_optional_string_len(value: Option<&str>, context: &str) -> Result<usize> {
    let len = match value {
        Some(value) => {
            let len = u32::try_from(value.len()).map_err(|_| {
                Error::InvalidFormat(format!("{context} string length exceeds u32"))
            })?;
            usize::try_from(len).map_err(|_| {
                Error::InvalidFormat(format!("{context} string length does not fit in usize"))
            })?
        }
        None => 0,
    };
    5usize
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} encoded string length overflow")))
}

/// Advance a decode cursor by `delta`, surfacing `context` in overflow errors.
fn advance_offset(offset: &mut usize, delta: usize, context: &str) -> Result<()> {
    *offset = offset
        .checked_add(delta)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))?;
    Ok(())
}

/// Borrow an optional UTF-8 string written by [`encode_optional_string`].
fn decode_optional_str<'a>(
    bytes: &'a [u8],
    offset: &mut usize,
    context: &str,
) -> Result<Option<&'a str>> {
    let present = *bytes
        .get(*offset)
        .ok_or_else(|| Error::InvalidFormat(format!("truncated {context} string presence flag")))?;
    advance_offset(offset, 1, context)?;
    let len = read_u32_le_at(bytes, *offset)
        .ok_or_else(|| Error::InvalidFormat(format!("truncated {context} string length")))
        .and_then(|value| {
            usize::try_from(value).map_err(|_| {
                Error::InvalidFormat(format!("{context} string length does not fit in usize"))
            })
        })?;
    advance_offset(offset, 4, context)?;
    let value = checked_window(bytes, *offset, len)
        .ok_or_else(|| Error::InvalidFormat(format!("truncated {context} string value")))?;
    advance_offset(offset, len, context)?;
    if present == 0 {
        if len == 0 {
            Ok(None)
        } else {
            Err(Error::InvalidFormat(format!(
                "{context} absent string has nonzero length"
            )))
        }
    } else if present == 1 {
        Ok(Some(std::str::from_utf8(value).map_err(|_| {
            Error::InvalidFormat(format!("{context} string is not UTF-8"))
        })?))
    } else {
        Err(Error::InvalidFormat(format!(
            "{context} string presence flag is invalid"
        )))
    }
}

/// Encode an HDFS FAPL configuration into a byte buffer.
#[allow(non_snake_case)]
pub fn H5P__encode_hdfs_fapl_config_into(config: &HdfsFaplConfig, out: &mut Vec<u8>) -> Result<()> {
    out.try_reserve(encoded_hdfs_fapl_config_len(config)?)
        .map_err(|_| Error::InvalidFormat("HDFS FAPL config length overflow".into()))?;
    encode_optional_string(out, Some(&config.namenode_name), "HDFS namenode")?;
    out.extend_from_slice(&config.namenode_port.to_le_bytes());
    encode_optional_string(out, Some(&config.user_name), "HDFS user")?;
    out.extend_from_slice(&config.buffer_size.to_le_bytes());
    Ok(())
}

/// Encode an HDFS FAPL configuration into a byte buffer.
#[deprecated(note = "use H5P__encode_hdfs_fapl_config_into to reuse caller allocation")]
#[allow(non_snake_case)]
pub fn H5P__encode_hdfs_fapl_config(config: &HdfsFaplConfig) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(encoded_hdfs_fapl_config_len(config)?);
    H5P__encode_hdfs_fapl_config_into(config, &mut out)?;
    Ok(out)
}

fn encoded_hdfs_fapl_config_len(config: &HdfsFaplConfig) -> Result<usize> {
    let user_len = encoded_optional_string_len(Some(&config.user_name), "HDFS user")?;
    encoded_optional_string_len(Some(&config.namenode_name), "HDFS namenode")?
        .checked_add(2)
        .and_then(|len| len.checked_add(user_len))
        .and_then(|len| len.checked_add(4))
        .ok_or_else(|| Error::InvalidFormat("HDFS FAPL config length overflow".into()))
}

/// Decode an HDFS FAPL configuration from a byte buffer.
#[allow(non_snake_case)]
pub fn H5P__decode_hdfs_fapl_config(bytes: &[u8]) -> Result<HdfsFaplConfig> {
    let mut offset = 0;
    let namenode_name = decode_optional_str(bytes, &mut offset, "HDFS namenode")?
        .ok_or_else(|| Error::InvalidFormat("HDFS namenode string is required".into()))?
        .to_owned();
    let namenode_port = read_u16_le_at(bytes, offset)
        .ok_or_else(|| Error::InvalidFormat("truncated HDFS namenode port".into()))?;
    advance_offset(&mut offset, 2, "HDFS FAPL config")?;
    let user_name = decode_optional_str(bytes, &mut offset, "HDFS user")?
        .ok_or_else(|| Error::InvalidFormat("HDFS user string is required".into()))?
        .to_owned();
    let buffer_size = read_u32_le_at(bytes, offset)
        .ok_or_else(|| Error::InvalidFormat("truncated HDFS buffer size".into()))?;
    advance_offset(&mut offset, 4, "HDFS FAPL config")?;
    if offset != bytes.len() {
        return Err(Error::InvalidFormat(
            "trailing bytes in HDFS FAPL config".into(),
        ));
    }
    Ok(HdfsFaplConfig {
        namenode_name,
        namenode_port,
        user_name,
        buffer_size,
    })
}

/// Encode a ROS3 FAPL configuration into a byte buffer.
#[allow(non_snake_case)]
pub fn H5P__encode_ros3_fapl_config_into(config: &Ros3FaplConfig, out: &mut Vec<u8>) -> Result<()> {
    out.try_reserve(encoded_ros3_fapl_config_len(config)?)
        .map_err(|_| Error::InvalidFormat("ROS3 FAPL config length overflow".into()))?;
    encode_optional_string(out, config.endpoint.as_deref(), "ROS3 endpoint")?;
    encode_optional_string(out, config.region.as_deref(), "ROS3 region")?;
    encode_optional_string(out, config.token.as_deref(), "ROS3 token")?;
    Ok(())
}

/// Encode a ROS3 FAPL configuration into a byte buffer.
#[deprecated(note = "use H5P__encode_ros3_fapl_config_into to reuse caller allocation")]
#[allow(non_snake_case)]
pub fn H5P__encode_ros3_fapl_config(config: &Ros3FaplConfig) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(encoded_ros3_fapl_config_len(config)?);
    H5P__encode_ros3_fapl_config_into(config, &mut out)?;
    Ok(out)
}

fn encoded_ros3_fapl_config_len(config: &Ros3FaplConfig) -> Result<usize> {
    let region_len = encoded_optional_string_len(config.region.as_deref(), "ROS3 region")?;
    let token_len = encoded_optional_string_len(config.token.as_deref(), "ROS3 token")?;
    encoded_optional_string_len(config.endpoint.as_deref(), "ROS3 endpoint")?
        .checked_add(region_len)
        .and_then(|len| len.checked_add(token_len))
        .ok_or_else(|| Error::InvalidFormat("ROS3 FAPL config length overflow".into()))
}

/// Decode a ROS3 FAPL configuration from a byte buffer.
#[allow(non_snake_case)]
pub fn H5P__decode_ros3_fapl_config(bytes: &[u8]) -> Result<Ros3FaplConfig> {
    let mut offset = 0;
    let endpoint = decode_optional_str(bytes, &mut offset, "ROS3 endpoint")?.map(str::to_owned);
    let region = decode_optional_str(bytes, &mut offset, "ROS3 region")?.map(str::to_owned);
    let token = decode_optional_str(bytes, &mut offset, "ROS3 token")?.map(str::to_owned);
    if offset != bytes.len() {
        return Err(Error::InvalidFormat(
            "trailing bytes in ROS3 FAPL config".into(),
        ));
    }
    Ok(Ros3FaplConfig {
        endpoint,
        region,
        token,
    })
}

/// Bounds-checked subslice `data[pos..pos+len]`.
fn checked_window(data: &[u8], pos: usize, len: usize) -> Option<&[u8]> {
    let end = pos.checked_add(len)?;
    data.get(pos..end)
}

/// Read a little-endian `u16` from `data` at `pos`.
fn read_u16_le_at(data: &[u8], pos: usize) -> Option<u16> {
    Some(u16::from_le_bytes(
        checked_window(data, pos, 2)?.try_into().ok()?,
    ))
}

/// Read a little-endian `u32` from `data` at `pos`.
fn read_u32_le_at(data: &[u8], pos: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        checked_window(data, pos, 4)?.try_into().ok()?,
    ))
}

/// Read the file-space strategy property from a file-creation list.
#[allow(non_snake_case)]
pub fn H5P__get_file_space_ref(list: &PropertyList) -> Result<&[u8]> {
    H5P__plist_get_ref(list, "file_space_strategy")
}

/// Read the file-space strategy property into a caller-provided buffer.
#[allow(non_snake_case)]
pub fn H5P__get_file_space_into(list: &PropertyList, out: &mut Vec<u8>) -> Result<()> {
    H5P__plist_get_into(list, "file_space_strategy", out)
}

/// Set the HDFS file-access driver with default configuration.
#[allow(non_snake_case)]
pub fn H5Pset_fapl_hdfs(list: &mut PropertyList) -> Result<()> {
    H5Pset_fapl_hdfs_config(list, HdfsFaplConfig::default())
}

/// Set the HDFS file-access driver with the given configuration.
#[allow(non_snake_case)]
pub fn H5Pset_fapl_hdfs_config(list: &mut PropertyList, config: HdfsFaplConfig) -> Result<()> {
    let mut encoded = Vec::new();
    H5P__encode_hdfs_fapl_config_into(&config, &mut encoded)?;
    plist_clear_fapl_driver_config(list)?;
    plist_set(list, "driver", b"hdfs".to_vec())?;
    plist_set(list, "fapl_hdfs_config", encoded)
}

/// Get the stored HDFS file-access driver configuration, if any.
#[allow(non_snake_case)]
pub fn H5Pget_fapl_hdfs_config(list: &PropertyList) -> Result<Option<HdfsFaplConfig>> {
    let bytes = plist_get_ref(list, "fapl_hdfs_config")?;
    if bytes.is_empty() {
        Ok(None)
    } else {
        H5P__decode_hdfs_fapl_config(bytes).map(Some)
    }
}

/// Set the direct file-access driver on the property list.
#[allow(non_snake_case)]
pub fn H5Pset_fapl_direct(list: &mut PropertyList) -> Result<()> {
    plist_clear_fapl_driver_config(list)?;
    plist_set(list, "driver", b"direct".to_vec())
}

/// The mirror driver is not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5Pset_fapl_mirror(_list: &mut PropertyList) -> Result<()> {
    unsupported_driver("mirror")
}

/// The MPI-IO driver is not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5Pset_fapl_mpio(_list: &mut PropertyList) -> Result<()> {
    unsupported_driver("mpio")
}

/// The MPI-IO data-transfer property is not supported.
#[allow(non_snake_case)]
pub fn H5Pset_dxpl_mpio(_list: &mut PropertyList) -> Result<()> {
    unsupported_driver("mpio dxpl")
}

/// The MPI-IO collective option is not supported.
#[allow(non_snake_case)]
pub fn H5Pset_dxpl_mpio_collective_opt(_list: &mut PropertyList) -> Result<()> {
    unsupported_driver("mpio collective")
}

/// Set the family file-access driver on the property list.
#[allow(non_snake_case)]
pub fn H5Pset_fapl_family(list: &mut PropertyList) -> Result<()> {
    plist_clear_fapl_driver_config(list)?;
    plist_set(list, "driver", b"family".to_vec())
}

/// Register properties for the `dacc` property class.
#[allow(non_snake_case)]
pub fn H5P__dacc_reg_prop(class: &mut PropertyClass) -> Result<()> {
    H5P__register(class, "dacc", Vec::new())
}

/// Register properties for the `lcrt` property class.
#[allow(non_snake_case)]
pub fn H5P__lcrt_reg_prop(class: &mut PropertyClass) -> Result<()> {
    H5P__register(class, "lcrt", Vec::new())
}

/// Register properties for the `ocpy` property class.
#[allow(non_snake_case)]
pub fn H5P__ocpy_reg_prop(class: &mut PropertyClass) -> Result<()> {
    H5P__register(class, "ocpy", Vec::new())
}

/// Register properties for the `fcrt` property class.
#[allow(non_snake_case)]
pub fn H5P__fcrt_reg_prop(class: &mut PropertyClass) -> Result<()> {
    H5P__register(class, "fcrt", Vec::new())
}

/// Register properties for the `mcrt` property class.
#[allow(non_snake_case)]
pub fn H5P__mcrt_reg_prop(class: &mut PropertyClass) -> Result<()> {
    H5P__register(class, "mcrt", Vec::new())
}

/// Register properties for the `dcrt` property class.
#[allow(non_snake_case)]
pub fn H5P__dcrt_reg_prop(class: &mut PropertyClass) -> Result<()> {
    H5P__register(class, "dcrt", Vec::new())
}

/// Set the `dapl_vds_file_pref` property on a property list.
#[allow(non_snake_case)]
pub fn H5P__dapl_vds_file_pref_set(list: &mut PropertyList, value: Vec<u8>) -> Result<()> {
    plist_set(list, "dapl_vds_file_prefix", value)
}

/// Get the `dapl_vds_file_pref` property from a property list.
#[allow(non_snake_case)]
pub fn H5P__dapl_vds_file_pref_get_ref(list: &PropertyList) -> Result<&[u8]> {
    H5P__plist_get_ref(list, "dapl_vds_file_prefix")
}

/// Get the `dapl_vds_file_pref` property into a caller-provided buffer.
#[allow(non_snake_case)]
pub fn H5P__dapl_vds_file_pref_get_into(list: &PropertyList, out: &mut Vec<u8>) -> Result<()> {
    H5P__plist_get_into(list, "dapl_vds_file_prefix", out)
}

/// Encode a `dapl_vds_file_pref` property value into bytes.
#[allow(non_snake_case)]
pub fn H5P__dapl_vds_file_pref_enc(value: &[u8]) -> &[u8] {
    value
}

/// Decode a `dapl_vds_file_pref` property value from bytes.
#[allow(non_snake_case)]
pub fn H5P__dapl_vds_file_pref_dec(value: &[u8]) -> &[u8] {
    value
}

/// Delete the `dapl_vds_file_pref` property from a property list.
#[allow(non_snake_case)]
pub fn H5P__dapl_vds_file_pref_del(list: &mut PropertyList) -> Result<()> {
    plist_del(list, "dapl_vds_file_prefix")
}

/// Copy a `dapl_vds_file_pref` property value.
#[allow(non_snake_case)]
pub fn H5P__dapl_vds_file_pref_copy(value: &[u8]) -> &[u8] {
    value
}

/// Close (release storage for) a `dapl_vds_file_pref` property value.
#[allow(non_snake_case)]
pub fn H5P__dapl_vds_file_pref_close(_value: &[u8]) {}

/// Set the `dapl_efile_pref` property on a property list.
#[allow(non_snake_case)]
pub fn H5P__dapl_efile_pref_set(list: &mut PropertyList, value: Vec<u8>) -> Result<()> {
    plist_set(list, "dapl_external_file_prefix", value)
}

/// Get the `dapl_efile_pref` property from a property list.
#[allow(non_snake_case)]
pub fn H5P__dapl_efile_pref_get_ref(list: &PropertyList) -> Result<&[u8]> {
    H5P__plist_get_ref(list, "dapl_external_file_prefix")
}

/// Get the `dapl_efile_pref` property into a caller-provided buffer.
#[allow(non_snake_case)]
pub fn H5P__dapl_efile_pref_get_into(list: &PropertyList, out: &mut Vec<u8>) -> Result<()> {
    H5P__plist_get_into(list, "dapl_external_file_prefix", out)
}

/// Encode a `dapl_efile_pref` property value into bytes.
#[allow(non_snake_case)]
pub fn H5P__dapl_efile_pref_enc(value: &[u8]) -> &[u8] {
    value
}

/// Decode a `dapl_efile_pref` property value from bytes.
#[allow(non_snake_case)]
pub fn H5P__dapl_efile_pref_dec(value: &[u8]) -> &[u8] {
    value
}

/// Delete the `dapl_efile_pref` property from a property list.
#[allow(non_snake_case)]
pub fn H5P__dapl_efile_pref_del(list: &mut PropertyList) -> Result<()> {
    plist_del(list, "dapl_external_file_prefix")
}

/// Copy a `dapl_efile_pref` property value.
#[allow(non_snake_case)]
pub fn H5P__dapl_efile_pref_copy(value: &[u8]) -> &[u8] {
    value
}

/// Compare two `dapl_efile_pref` property values.
#[allow(non_snake_case)]
pub fn H5P__dapl_efile_pref_cmp(left: &[u8], right: &[u8]) -> bool {
    prop_cmp(left, right)
}

/// Close (release storage for) a `dapl_efile_pref` property value.
#[allow(non_snake_case)]
pub fn H5P__dapl_efile_pref_close(_value: &[u8]) {}

/// Encode a `dacc_vds_view` property value into bytes.
#[allow(non_snake_case)]
pub fn H5P__dacc_vds_view_enc(value: u8) -> [u8; 1] {
    encode_uint8_bytes(value)
}

/// Decode a `dacc_vds_view` property value from bytes.
#[allow(non_snake_case)]
pub fn H5P__dacc_vds_view_dec(value: &[u8]) -> Result<u8> {
    H5P__decode_uint8_t(value)
}

/// Duplicate a committed-datatype merge list buffer.
#[allow(non_snake_case)]
pub fn H5P__copy_merge_comm_dt_list(value: &[u8]) -> &[u8] {
    value
}

/// Set the `ocpy_merge_comm_dt_list` property on a property list.
#[allow(non_snake_case)]
pub fn H5P__ocpy_merge_comm_dt_list_set(list: &mut PropertyList, value: Vec<u8>) -> Result<()> {
    plist_set(list, "ocpy_merge_committed_datatypes", value)
}

/// Get the `ocpy_merge_comm_dt_list` property from a property list.
#[allow(non_snake_case)]
pub fn H5P__ocpy_merge_comm_dt_list_get_ref(list: &PropertyList) -> Result<&[u8]> {
    H5P__plist_get_ref(list, "ocpy_merge_committed_datatypes")
}

/// Get the `ocpy_merge_comm_dt_list` property into a caller-provided buffer.
#[allow(non_snake_case)]
pub fn H5P__ocpy_merge_comm_dt_list_get_into(list: &PropertyList, out: &mut Vec<u8>) -> Result<()> {
    H5P__plist_get_into(list, "ocpy_merge_committed_datatypes", out)
}

/// Encode a `ocpy_merge_comm_dt_list` property value into bytes.
#[allow(non_snake_case)]
pub fn H5P__ocpy_merge_comm_dt_list_enc(value: &[u8]) -> &[u8] {
    value
}

/// Decode a `ocpy_merge_comm_dt_list` property value from bytes.
#[allow(non_snake_case)]
pub fn H5P__ocpy_merge_comm_dt_list_dec(value: &[u8]) -> &[u8] {
    value
}

/// Delete the `ocpy_merge_comm_dt_list` property from a property list.
#[allow(non_snake_case)]
pub fn H5P__ocpy_merge_comm_dt_list_del(list: &mut PropertyList) -> Result<()> {
    plist_del(list, "ocpy_merge_committed_datatypes")
}

/// Copy a `ocpy_merge_comm_dt_list` property value.
#[allow(non_snake_case)]
pub fn H5P__ocpy_merge_comm_dt_list_copy(value: &[u8]) -> &[u8] {
    value
}

/// Compare two `ocpy_merge_comm_dt_list` property values.
#[allow(non_snake_case)]
pub fn H5P__ocpy_merge_comm_dt_list_cmp(left: &[u8], right: &[u8]) -> bool {
    prop_cmp(left, right)
}

/// Close (release storage for) a `ocpy_merge_comm_dt_list` property value.
#[allow(non_snake_case)]
pub fn H5P__ocpy_merge_comm_dt_list_close(_value: &[u8]) {}

/// Decode the B-tree rank stored on a file-creation property list.
#[allow(non_snake_case)]
pub fn H5P__fcrt_btree_rank_dec(value: &[u8]) -> Result<u8> {
    H5P__decode_uint8_t(value)
}

/// Encode the shared-message index type bitmask.
#[allow(non_snake_case)]
pub fn H5P__fcrt_shmsg_index_types_enc(value: u64) -> [u8; 8] {
    encode_uint64_bytes(value)
}

/// Encode the shared-message index minimum-size threshold.
#[allow(non_snake_case)]
pub fn H5P__fcrt_shmsg_index_minsize_enc(value: u64) -> [u8; 8] {
    encode_uint64_bytes(value)
}

/// Set the file-space strategy bytes on a file-creation property list.
#[allow(non_snake_case)]
pub fn H5P__set_file_space_strategy(list: &mut PropertyList, value: Vec<u8>) -> Result<()> {
    plist_set(list, "file_space_strategy", value)
}

/// Decode the file-space strategy property buffer.
#[allow(non_snake_case)]
pub fn H5P__fcrt_fspace_strategy_dec(value: &[u8]) -> &[u8] {
    value
}

/// Encode a `dcrt_layout` property value into bytes.
#[allow(non_snake_case)]
pub fn H5P__dcrt_layout_enc(value: u8) -> [u8; 1] {
    encode_uint8_bytes(value)
}

/// Decode a `dcrt_layout` property value from bytes.
#[allow(non_snake_case)]
pub fn H5P__dcrt_layout_dec(value: &[u8]) -> Result<u8> {
    H5P__decode_uint8_t(value)
}

/// Copy a `dcrt_layout` property value.
#[allow(non_snake_case)]
pub fn H5P__dcrt_layout_copy(value: &[u8]) -> &[u8] {
    value
}

/// Compare two `dcrt_layout` property values.
#[allow(non_snake_case)]
pub fn H5P__dcrt_layout_cmp(left: &[u8], right: &[u8]) -> bool {
    prop_cmp(left, right)
}

/// Close (release storage for) a `dcrt_layout` property value.
#[allow(non_snake_case)]
pub fn H5P__dcrt_layout_close(_value: &[u8]) {}

/// Delete the dataset layout property, releasing its storage.
#[allow(non_snake_case)]
pub fn H5P__dcrt_layout_del(list: &mut PropertyList) -> Result<()> {
    list.ensure_open()?;
    if let Some(mut prop) = list.properties.remove("layout") {
        prop.value.clear();
        prop.name.clear();
    }
    Ok(())
}

/// Set the dataset layout property on a dataset-creation list.
#[allow(non_snake_case)]
pub fn H5P__set_layout(list: &mut PropertyList, layout: u8) -> Result<()> {
    plist_set(list, "layout", H5P__dcrt_layout_enc(layout).to_vec())
}

/// Set the `dcrt_fill_value` property on a property list.
#[allow(non_snake_case)]
pub fn H5P__dcrt_fill_value_set(list: &mut PropertyList, value: Vec<u8>) -> Result<()> {
    plist_set(list, "fill_value", value)
}

/// Get the `dcrt_fill_value` property from a property list.
#[allow(non_snake_case)]
pub fn H5P__dcrt_fill_value_get_ref(list: &PropertyList) -> Result<&[u8]> {
    H5P__plist_get_ref(list, "fill_value")
}

/// Get the `dcrt_fill_value` property into a caller-provided buffer.
#[allow(non_snake_case)]
pub fn H5P__dcrt_fill_value_get_into(list: &PropertyList, out: &mut Vec<u8>) -> Result<()> {
    H5P__plist_get_into(list, "fill_value", out)
}

/// Encode a `dcrt_fill_value` property value into bytes.
#[allow(non_snake_case)]
pub fn H5P__dcrt_fill_value_enc(value: &[u8]) -> &[u8] {
    value
}

/// Decode a `dcrt_fill_value` property value from bytes.
#[allow(non_snake_case)]
pub fn H5P__dcrt_fill_value_dec(value: &[u8]) -> &[u8] {
    value
}

/// Delete the `dcrt_fill_value` property from a property list.
#[allow(non_snake_case)]
pub fn H5P__dcrt_fill_value_del(list: &mut PropertyList) -> Result<()> {
    plist_del(list, "fill_value")
}

/// Compare two encoded fill-value property buffers.
#[allow(non_snake_case)]
pub fn H5P_fill_value_cmp(left: &[u8], right: &[u8]) -> bool {
    prop_cmp(left, right)
}

/// Close (release storage for) a `dcrt_fill_value` property value.
#[allow(non_snake_case)]
pub fn H5P__dcrt_fill_value_close(_value: &[u8]) {}

/// Get the encoded fill value from a property list.
#[allow(non_snake_case)]
pub fn H5P_get_fill_value_ref(list: &PropertyList) -> Result<&[u8]> {
    H5P__dcrt_fill_value_get_ref(list)
}

/// Get the encoded fill value into a caller-provided buffer.
#[allow(non_snake_case)]
pub fn H5P_get_fill_value_into(list: &PropertyList, out: &mut Vec<u8>) -> Result<()> {
    H5P__dcrt_fill_value_get_into(list, out)
}

/// Get the encoded fill value from a property list.
#[deprecated(note = "use H5P_get_fill_value_ref or H5P_get_fill_value_into")]
#[allow(non_snake_case)]
pub fn H5P_get_fill_value(list: &PropertyList) -> Result<Vec<u8>> {
    Ok(H5P_get_fill_value_ref(list)?.to_vec())
}

/// Return true if a non-empty fill value is set.
#[allow(non_snake_case)]
pub fn H5P_is_fill_value_defined(list: &PropertyList) -> Result<bool> {
    Ok(!H5P__dcrt_fill_value_get_ref(list)?.is_empty())
}

/// Return true if a non-empty fill value is set.
#[allow(non_snake_case)]
pub fn H5P_fill_value_defined(list: &PropertyList) -> Result<bool> {
    H5P_is_fill_value_defined(list)
}

/// Public API that returns true if a non-empty fill value is set.
#[allow(non_snake_case)]
pub fn H5Pfill_value_defined(list: &PropertyList) -> Result<bool> {
    H5P_is_fill_value_defined(list)
}

/// Set the `dcrt_ext_file_list` property on a property list.
#[allow(non_snake_case)]
pub fn H5P__dcrt_ext_file_list_set(list: &mut PropertyList, value: Vec<u8>) -> Result<()> {
    plist_set(list, "external_file_list", value)
}

/// Get the `dcrt_ext_file_list` property from a property list.
#[allow(non_snake_case)]
pub fn H5P__dcrt_ext_file_list_get_ref(list: &PropertyList) -> Result<&[u8]> {
    H5P__plist_get_ref(list, "external_file_list")
}

/// Get the `dcrt_ext_file_list` property into a caller-provided buffer.
#[allow(non_snake_case)]
pub fn H5P__dcrt_ext_file_list_get_into(list: &PropertyList, out: &mut Vec<u8>) -> Result<()> {
    H5P__plist_get_into(list, "external_file_list", out)
}

/// Encode a `dcrt_ext_file_list` property value into bytes.
#[allow(non_snake_case)]
pub fn H5P__dcrt_ext_file_list_enc(value: &[u8]) -> &[u8] {
    value
}

/// Decode a `dcrt_ext_file_list` property value from bytes.
#[allow(non_snake_case)]
pub fn H5P__dcrt_ext_file_list_dec(value: &[u8]) -> &[u8] {
    value
}

/// Delete the `dcrt_ext_file_list` property from a property list.
#[allow(non_snake_case)]
pub fn H5P__dcrt_ext_file_list_del(list: &mut PropertyList) -> Result<()> {
    plist_del(list, "external_file_list")
}

/// Copy a `dcrt_ext_file_list` property value.
#[allow(non_snake_case)]
pub fn H5P__dcrt_ext_file_list_copy(value: &[u8]) -> &[u8] {
    value
}

/// Compare two `dcrt_ext_file_list` property values.
#[allow(non_snake_case)]
pub fn H5P__dcrt_ext_file_list_cmp(left: &[u8], right: &[u8]) -> bool {
    prop_cmp(left, right)
}

/// Close (release storage for) a `dcrt_ext_file_list` property value.
#[allow(non_snake_case)]
pub fn H5P__dcrt_ext_file_list_close(_value: &[u8]) {}

/// Register properties for the `facc` property class.
#[allow(non_snake_case)]
pub fn H5P__facc_reg_prop(class: &mut PropertyClass) -> Result<()> {
    H5P__register(class, "facc", Vec::new())
}

/// Set the default `sec2` driver on the FAPL.
#[allow(non_snake_case)]
pub fn H5P__facc_set_def_driver(list: &mut PropertyList) -> Result<()> {
    plist_clear_fapl_driver_config(list)?;
    plist_set(list, "driver", b"sec2".to_vec())
}

/// Return true for a built-in supported VFD name.
#[allow(non_snake_case)]
pub fn H5P__facc_set_def_driver_check_predefined(name: &str) -> bool {
    matches!(name, "sec2" | "stdio" | "core" | "direct" | "family")
}

/// Set the VFD driver on a FAPL by name.
#[allow(non_snake_case)]
pub fn H5P_set_driver(list: &mut PropertyList, name: &str) -> Result<()> {
    if H5P__facc_set_def_driver_check_predefined(name) {
        plist_clear_fapl_driver_config(list)?;
        plist_set(list, "driver", name.as_bytes().to_vec())
    } else {
        unsupported_driver(name)
    }
}

/// Set the VFD driver on a FAPL by name.
#[allow(non_snake_case)]
pub fn H5P_set_driver_by_name(list: &mut PropertyList, name: &str) -> Result<()> {
    H5P_set_driver(list, name)
}

/// Set the VFD driver on a FAPL by numeric value.
#[allow(non_snake_case)]
pub fn H5P_set_driver_by_value(list: &mut PropertyList, value: u8) -> Result<()> {
    let name = match value {
        0 => "sec2",
        1 => "stdio",
        2 => "core",
        3 => "direct",
        4 => "family",
        _ => return unsupported_driver("unknown"),
    };
    H5P_set_driver(list, name)
}

/// Read the driver name from a file-access property list.
#[allow(non_snake_case)]
pub fn H5P__facc_file_driver_get_ref(list: &PropertyList) -> Result<&[u8]> {
    H5P__plist_get_ref(list, "driver")
}

/// Read the driver name into a caller-provided buffer.
#[allow(non_snake_case)]
pub fn H5P__facc_file_driver_get_into(list: &PropertyList, out: &mut Vec<u8>) -> Result<()> {
    H5P__plist_get_into(list, "driver", out)
}

/// Set the multi-driver memory type property.
#[allow(non_snake_case)]
pub fn H5Pset_multi_type(list: &mut PropertyList, value: u8) -> Result<()> {
    plist_set(list, "multi_type", encode_uint8_bytes(value).to_vec())
}

/// Compare two `facc_cache_image_config` property values.
#[allow(non_snake_case)]
pub fn H5P__facc_cache_image_config_cmp(left: &[u8], right: &[u8]) -> bool {
    prop_cmp(left, right)
}

/// Encode a `facc_cache_image_config` property value into bytes.
#[allow(non_snake_case)]
pub fn H5P__facc_cache_image_config_enc(value: &[u8]) -> &[u8] {
    value
}

/// Decode a `facc_cache_image_config` property value from bytes.
#[allow(non_snake_case)]
pub fn H5P__facc_cache_image_config_dec(value: &[u8]) -> &[u8] {
    value
}

/// Set the `facc_file_image_info` property on a property list.
#[allow(non_snake_case)]
pub fn H5P__facc_file_image_info_set(list: &mut PropertyList, value: Vec<u8>) -> Result<()> {
    plist_set(list, "file_image_info", value)
}

/// Get the `facc_file_image_info` property from a property list.
#[allow(non_snake_case)]
pub fn H5P__facc_file_image_info_get_ref(list: &PropertyList) -> Result<&[u8]> {
    H5P__plist_get_ref(list, "file_image_info")
}

/// Get the `facc_file_image_info` property into a caller-provided buffer.
#[allow(non_snake_case)]
pub fn H5P__facc_file_image_info_get_into(list: &PropertyList, out: &mut Vec<u8>) -> Result<()> {
    H5P__plist_get_into(list, "file_image_info", out)
}

/// Delete the `facc_file_image_info` property from a property list.
#[allow(non_snake_case)]
pub fn H5P__facc_file_image_info_del(list: &mut PropertyList) -> Result<()> {
    plist_del(list, "file_image_info")
}

/// Compare two `facc_file_image_info` property values.
#[allow(non_snake_case)]
pub fn H5P__facc_file_image_info_cmp(left: &[u8], right: &[u8]) -> bool {
    prop_cmp(left, right)
}

/// Encode a `facc_cache_config` property value into bytes.
#[allow(non_snake_case)]
pub fn H5P__facc_cache_config_enc(value: &[u8]) -> &[u8] {
    value
}

/// Decode a `facc_cache_config` property value from bytes.
#[allow(non_snake_case)]
pub fn H5P__facc_cache_config_dec(value: &[u8]) -> &[u8] {
    value
}

/// Encode a `facc_fclose_degree` property value into bytes.
#[allow(non_snake_case)]
pub fn H5P__facc_fclose_degree_enc(value: u8) -> [u8; 1] {
    encode_uint8_bytes(value)
}

/// Decode a `facc_fclose_degree` property value from bytes.
#[allow(non_snake_case)]
pub fn H5P__facc_fclose_degree_dec(value: &[u8]) -> Result<u8> {
    H5P__decode_uint8_t(value)
}

/// Encode a `facc_multi_type` property value into bytes.
#[allow(non_snake_case)]
pub fn H5P__facc_multi_type_enc(value: u8) -> [u8; 1] {
    encode_uint8_bytes(value)
}

/// Decode a `facc_multi_type` property value from bytes.
#[allow(non_snake_case)]
pub fn H5P__facc_multi_type_dec(value: &[u8]) -> Result<u8> {
    H5P__decode_uint8_t(value)
}

/// Encode a `facc_libver_type` property value into bytes.
#[allow(non_snake_case)]
pub fn H5P__facc_libver_type_enc(value: u8) -> [u8; 1] {
    encode_uint8_bytes(value)
}

/// Decode a `facc_libver_type` property value from bytes.
#[allow(non_snake_case)]
pub fn H5P__facc_libver_type_dec(value: &[u8]) -> Result<u8> {
    H5P__decode_uint8_t(value)
}

/// Encode a `facc_mdc_log_location` property value into bytes.
#[allow(non_snake_case)]
pub fn H5P__facc_mdc_log_location_enc(value: &[u8]) -> &[u8] {
    value
}

/// Decode a `facc_mdc_log_location` property value from bytes.
#[allow(non_snake_case)]
pub fn H5P__facc_mdc_log_location_dec(value: &[u8]) -> &[u8] {
    value
}

/// Copy a `facc_mdc_log_location` property value.
#[allow(non_snake_case)]
pub fn H5P__facc_mdc_log_location_copy(value: &[u8]) -> &[u8] {
    value
}

/// Compare two `facc_mdc_log_location` property values.
#[allow(non_snake_case)]
pub fn H5P__facc_mdc_log_location_cmp(left: &[u8], right: &[u8]) -> bool {
    prop_cmp(left, right)
}

/// Close (release storage for) a `facc_mdc_log_location` property value.
#[allow(non_snake_case)]
pub fn H5P__facc_mdc_log_location_close(_value: &[u8]) {}

/// The MPI parameters are not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5Pset_mpi_params(_list: &mut PropertyList) -> Result<()> {
    unsupported_driver("mpi params")
}

/// Set the `facc_mpi_comm` property on a property list.
#[allow(non_snake_case)]
pub fn H5P__facc_mpi_comm_set(_list: &mut PropertyList, _value: Vec<u8>) -> Result<()> {
    unsupported_driver("mpi communicator")
}

/// Get the `facc_mpi_comm` property from a property list.
#[allow(non_snake_case)]
pub fn H5P__facc_mpi_comm_get(_list: &PropertyList) -> Result<&[u8]> {
    Err(Error::Unsupported(
        "mpi communicator driver is not supported by the pure Rust local backend".into(),
    ))
}

/// Delete the `facc_mpi_comm` property from a property list.
#[allow(non_snake_case)]
pub fn H5P__facc_mpi_comm_del(_list: &mut PropertyList) -> Result<()> {
    unsupported_driver("mpi communicator")
}

/// Copy a `facc_mpi_comm` property value.
#[allow(non_snake_case)]
pub fn H5P__facc_mpi_comm_copy(value: &[u8]) -> &[u8] {
    value
}

/// Compare two `facc_mpi_comm` property values.
#[allow(non_snake_case)]
pub fn H5P__facc_mpi_comm_cmp(left: &[u8], right: &[u8]) -> bool {
    prop_cmp(left, right)
}

/// Close (release storage for) a `facc_mpi_comm` property value.
#[allow(non_snake_case)]
pub fn H5P__facc_mpi_comm_close(_value: &[u8]) {}

/// Set the `facc_mpi_info` property on a property list.
#[allow(non_snake_case)]
pub fn H5P__facc_mpi_info_set(_list: &mut PropertyList, _value: Vec<u8>) -> Result<()> {
    unsupported_driver("mpi info")
}

/// Get the `facc_mpi_info` property from a property list.
#[allow(non_snake_case)]
pub fn H5P__facc_mpi_info_get(_list: &PropertyList) -> Result<&[u8]> {
    Err(Error::Unsupported(
        "mpi info driver is not supported by the pure Rust local backend".into(),
    ))
}

/// Delete the `facc_mpi_info` property from a property list.
#[allow(non_snake_case)]
pub fn H5P__facc_mpi_info_del(_list: &mut PropertyList) -> Result<()> {
    unsupported_driver("mpi info")
}

/// Copy a `facc_mpi_info` property value.
#[allow(non_snake_case)]
pub fn H5P__facc_mpi_info_copy(value: &[u8]) -> &[u8] {
    value
}

/// Compare two `facc_mpi_info` property values.
#[allow(non_snake_case)]
pub fn H5P__facc_mpi_info_cmp(left: &[u8], right: &[u8]) -> bool {
    prop_cmp(left, right)
}

/// Close (release storage for) a `facc_mpi_info` property value.
#[allow(non_snake_case)]
pub fn H5P__facc_mpi_info_close(_value: &[u8]) {}

/// Encode a `facc_page_buffer_size` property value into bytes.
#[allow(non_snake_case)]
pub fn H5P__facc_page_buffer_size_enc(value: u64) -> [u8; 8] {
    encode_uint64_bytes(value)
}

/// Decode a `facc_page_buffer_size` property value from bytes.
#[allow(non_snake_case)]
pub fn H5P__facc_page_buffer_size_dec(value: &[u8]) -> Result<u64> {
    H5P__decode_uint64_t(value)
}

/// VOL connectors are not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5P_set_vol(_list: &mut PropertyList, _value: Vec<u8>) -> Result<()> {
    Err(Error::Unsupported(
        "VOL connectors are not supported".into(),
    ))
}

/// Clear the VOL connector property from a FAPL.
#[allow(non_snake_case)]
pub fn H5P_reset_vol_class(list: &mut PropertyList) -> Result<()> {
    plist_del(list, "vol")
}

/// Public API to set a VOL connector (not supported).
#[allow(non_snake_case)]
pub fn H5Pset_vol(list: &mut PropertyList, value: Vec<u8>) -> Result<()> {
    H5P_set_vol(list, value)
}

/// Get the `facc_vol` property from a property list.
#[allow(non_snake_case)]
pub fn H5P__facc_vol_get_ref(list: &PropertyList) -> Result<&[u8]> {
    H5P__plist_get_ref(list, "vol")
}

/// Get the `facc_vol` property into a caller-provided buffer.
#[allow(non_snake_case)]
pub fn H5P__facc_vol_get_into(list: &PropertyList, out: &mut Vec<u8>) -> Result<()> {
    H5P__plist_get_into(list, "vol", out)
}

/// The IOC subfiling driver is not supported.
#[allow(non_snake_case)]
pub fn H5Pset_fapl_ioc(_list: &mut PropertyList) -> Result<()> {
    unsupported_driver("ioc")
}

/// The subfiling driver is not supported.
#[allow(non_snake_case)]
pub fn H5Pset_fapl_subfiling(_list: &mut PropertyList) -> Result<()> {
    unsupported_driver("subfiling")
}

/// Register properties for the `dxfr` property class.
#[allow(non_snake_case)]
pub fn H5P__dxfr_reg_prop(class: &mut PropertyClass) -> Result<()> {
    H5P__register(class, "dxfr", Vec::new())
}

/// Encode a `dxfr_bkgr_buf_type` property value into bytes.
#[allow(non_snake_case)]
pub fn H5P__dxfr_bkgr_buf_type_enc(value: u8) -> [u8; 1] {
    encode_uint8_bytes(value)
}

/// Decode a `dxfr_bkgr_buf_type` property value from bytes.
#[allow(non_snake_case)]
pub fn H5P__dxfr_bkgr_buf_type_dec(value: &[u8]) -> Result<u8> {
    H5P__decode_uint8_t(value)
}

/// Encode a `dxfr_btree_split_ratio` property value into bytes.
#[allow(non_snake_case)]
pub fn H5P__dxfr_btree_split_ratio_enc(value: f64) -> [u8; 8] {
    encode_double_bytes(value)
}

/// Decode a `dxfr_btree_split_ratio` property value from bytes.
#[allow(non_snake_case)]
pub fn H5P__dxfr_btree_split_ratio_dec(value: &[u8]) -> Result<f64> {
    H5P__decode_double(value)
}

/// Set the `dxfr_xform` property on a property list.
#[allow(non_snake_case)]
pub fn H5P__dxfr_xform_set(list: &mut PropertyList, value: Vec<u8>) -> Result<()> {
    plist_set(list, "data_transform", value)
}

/// Get the `dxfr_xform` property from a property list.
#[allow(non_snake_case)]
pub fn H5P__dxfr_xform_get_ref(list: &PropertyList) -> Result<&[u8]> {
    H5P__plist_get_ref(list, "data_transform")
}

/// Get the `dxfr_xform` property into a caller-provided buffer.
#[allow(non_snake_case)]
pub fn H5P__dxfr_xform_get_into(list: &PropertyList, out: &mut Vec<u8>) -> Result<()> {
    H5P__plist_get_into(list, "data_transform", out)
}

/// Encode a `dxfr_xform` property value into bytes.
#[allow(non_snake_case)]
pub fn H5P__dxfr_xform_enc(value: &[u8]) -> &[u8] {
    value
}

/// Decode a `dxfr_xform` property value from bytes.
#[allow(non_snake_case)]
pub fn H5P__dxfr_xform_dec(value: &[u8]) -> &[u8] {
    value
}

/// Delete the `dxfr_xform` property from a property list.
#[allow(non_snake_case)]
pub fn H5P__dxfr_xform_del(list: &mut PropertyList) -> Result<()> {
    plist_del(list, "data_transform")
}

/// Copy a `dxfr_xform` property value.
#[allow(non_snake_case)]
pub fn H5P__dxfr_xform_copy(value: &[u8]) -> &[u8] {
    value
}

/// Compare two `dxfr_xform` property values.
#[allow(non_snake_case)]
pub fn H5P__dxfr_xform_cmp(left: &[u8], right: &[u8]) -> bool {
    prop_cmp(left, right)
}

/// Close (release storage for) a `dxfr_xform` property value.
#[allow(non_snake_case)]
pub fn H5P__dxfr_xform_close(_value: &[u8]) {}

/// Set the variable-length memory-manager callbacks on a transfer list.
#[allow(non_snake_case)]
pub fn H5P_set_vlen_mem_manager(list: &mut PropertyList, value: Vec<u8>) -> Result<()> {
    plist_set(list, "vlen_mem_manager", value)
}

/// Encode a `dxfr_io_xfer_mode` property value into bytes.
#[allow(non_snake_case)]
pub fn H5P__dxfr_io_xfer_mode_enc(value: u8) -> [u8; 1] {
    encode_uint8_bytes(value)
}

/// Decode a `dxfr_io_xfer_mode` property value from bytes.
#[allow(non_snake_case)]
pub fn H5P__dxfr_io_xfer_mode_dec(value: &[u8]) -> Result<u8> {
    H5P__decode_uint8_t(value)
}

/// Encode a `dxfr_mpio_collective_opt` property value into bytes.
#[allow(non_snake_case)]
pub fn H5P__dxfr_mpio_collective_opt_enc(value: u8) -> [u8; 1] {
    encode_uint8_bytes(value)
}

/// Decode a `dxfr_mpio_collective_opt` property value from bytes.
#[allow(non_snake_case)]
pub fn H5P__dxfr_mpio_collective_opt_dec(value: &[u8]) -> Result<u8> {
    H5P__decode_uint8_t(value)
}

/// Encode a `dxfr_mpio_chunk_opt_hard` property value into bytes.
#[allow(non_snake_case)]
pub fn H5P__dxfr_mpio_chunk_opt_hard_enc(value: u8) -> [u8; 1] {
    encode_uint8_bytes(value)
}

/// Decode a `dxfr_mpio_chunk_opt_hard` property value from bytes.
#[allow(non_snake_case)]
pub fn H5P__dxfr_mpio_chunk_opt_hard_dec(value: &[u8]) -> Result<u8> {
    H5P__decode_uint8_t(value)
}

/// Encode a `dxfr_edc` property value into bytes.
#[allow(non_snake_case)]
pub fn H5P__dxfr_edc_enc(value: u8) -> [u8; 1] {
    encode_uint8_bytes(value)
}

/// Decode a `dxfr_edc` property value from bytes.
#[allow(non_snake_case)]
pub fn H5P__dxfr_edc_dec(value: &[u8]) -> Result<u8> {
    H5P__decode_uint8_t(value)
}

/// Copy a `dxfr_dset_io_hyp_sel` property value.
#[allow(non_snake_case)]
pub fn H5P__dxfr_dset_io_hyp_sel_copy(value: &[u8]) -> &[u8] {
    value
}

/// Compare two `dxfr_dset_io_hyp_sel` property values.
#[allow(non_snake_case)]
pub fn H5P__dxfr_dset_io_hyp_sel_cmp(left: &[u8], right: &[u8]) -> bool {
    prop_cmp(left, right)
}

/// Close (release storage for) a `dxfr_dset_io_hyp_sel` property value.
#[allow(non_snake_case)]
pub fn H5P__dxfr_dset_io_hyp_sel_close(_value: &[u8]) {}

/// Set a hyperslab selection on a dataset I/O transfer list.
#[allow(non_snake_case)]
pub fn H5Pset_dataset_io_hyperslab_selection(
    list: &mut PropertyList,
    rank: usize,
    op: i8,
    start: &[u64],
    stride: Option<&[u64]>,
    count: &[u64],
    block: Option<&[u64]>,
) -> Result<()> {
    list.ensure_open()?;
    if rank == 0 || rank > 32 {
        return Err(Error::InvalidFormat(format!("invalid rank value: {rank}")));
    }
    if !(0..8).contains(&op) {
        return Err(Error::Unsupported("invalid selection operation".into()));
    }
    if start.len() != rank {
        return Err(Error::InvalidFormat("'start' rank mismatch".into()));
    }
    if count.len() != rank {
        return Err(Error::InvalidFormat("'count' rank mismatch".into()));
    }
    if let Some(stride) = stride {
        if stride.len() != rank {
            return Err(Error::InvalidFormat("'stride' rank mismatch".into()));
        }
        for (index, value) in stride.iter().enumerate() {
            if *value == 0 {
                return Err(Error::InvalidFormat(format!(
                    "invalid value - stride[{index}]==0"
                )));
            }
        }
    }
    if let Some(block) = block {
        if block.len() != rank {
            return Err(Error::InvalidFormat("'block' rank mismatch".into()));
        }
    }

    if let Some(existing) = list.properties.get("dataset_io_hyperslab_selection") {
        if existing.value.len() >= 8 {
            let existing_rank = u32::from_le_bytes([
                existing.value[0],
                existing.value[1],
                existing.value[2],
                existing.value[3],
            ]) as usize;
            if existing_rank != rank && op != 0 {
                return Err(Error::InvalidFormat(
                    "different rank for previous and new selections".into(),
                ));
            }
        }
    }

    let mut encoded = Vec::with_capacity(
        8usize
            .checked_add(rank.checked_mul(32).ok_or_else(|| {
                Error::InvalidFormat("dataset I/O hyperslab selection size overflow".into())
            })?)
            .ok_or_else(|| {
                Error::InvalidFormat("dataset I/O hyperslab selection size overflow".into())
            })?,
    );
    encoded.extend_from_slice(
        &(u32::try_from(rank)
            .map_err(|_| Error::InvalidFormat("rank does not fit in u32".into()))?)
        .to_le_bytes(),
    );
    encoded.extend_from_slice(&(i32::from(op)).to_le_bytes());
    for axis in 0..rank {
        encoded.extend_from_slice(&start[axis].to_le_bytes());
        encoded.extend_from_slice(&stride.map(|values| values[axis]).unwrap_or(1).to_le_bytes());
        encoded.extend_from_slice(&count[axis].to_le_bytes());
        encoded.extend_from_slice(&block.map(|values| values[axis]).unwrap_or(1).to_le_bytes());
    }
    list.properties.insert(
        "dataset_io_hyperslab_selection".to_string(),
        Property {
            name: "dataset_io_hyperslab_selection".to_string(),
            value: encoded,
        },
    );
    Ok(())
}

/// Encode a `dxfr_selection_io_mode` property value into bytes.
#[allow(non_snake_case)]
pub fn H5P__dxfr_selection_io_mode_enc(value: u8) -> [u8; 1] {
    encode_uint8_bytes(value)
}

/// Encode a `dxfr_modify_write_buf` property value into bytes.
#[allow(non_snake_case)]
pub fn H5P__dxfr_modify_write_buf_enc(value: bool) -> [u8; 1] {
    encode_bool_bytes(value)
}

/// Decode a `dxfr_modify_write_buf` property value from bytes.
#[allow(non_snake_case)]
pub fn H5P__dxfr_modify_write_buf_dec(value: &[u8]) -> Result<bool> {
    H5P__decode_bool(value)
}

/// Register properties for the `fmnt` property class.
#[allow(non_snake_case)]
pub fn H5P__fmnt_reg_prop(class: &mut PropertyClass) -> Result<()> {
    H5P__register(class, "fmnt", Vec::new())
}

/// Register properties for the `ocrt` property class.
#[allow(non_snake_case)]
pub fn H5P__ocrt_reg_prop(class: &mut PropertyClass) -> Result<()> {
    H5P__register(class, "ocrt", Vec::new())
}

/// Modify the filter pipeline stored on a property list.
#[allow(non_snake_case)]
pub fn H5P_modify_filter(list: &mut PropertyList, filter: Vec<u8>) -> Result<()> {
    plist_set(list, "filter_pipeline", filter)
}

/// Public API that modifies the filter pipeline on a property list.
#[allow(non_snake_case)]
pub fn H5Pmodify_filter(list: &mut PropertyList, filter: Vec<u8>) -> Result<()> {
    H5P_modify_filter(list, filter)
}

/// Internal helper alias for [`H5P_modify_filter`].
#[allow(non_snake_case)]
pub fn H5P__set_filter(list: &mut PropertyList, filter: Vec<u8>) -> Result<()> {
    H5P_modify_filter(list, filter)
}

/// Return the encoded filter pipeline (ignoring filter id lookup).
#[deprecated(note = "use H5P_get_filter_by_id_ref or H5P_get_filter_by_id_into")]
#[allow(non_snake_case)]
pub fn H5P_get_filter_by_id(list: &PropertyList, _filter_id: u32) -> Result<Vec<u8>> {
    Ok(H5P_get_filter_by_id_ref(list, _filter_id)?.to_vec())
}

/// Return the encoded filter pipeline by borrowed reference.
#[allow(non_snake_case)]
pub fn H5P_get_filter_by_id_ref(list: &PropertyList, _filter_id: u32) -> Result<&[u8]> {
    H5P__plist_get_ref(list, "filter_pipeline")
}

/// Append the encoded filter pipeline into a caller-provided buffer.
#[allow(non_snake_case)]
pub fn H5P_get_filter_by_id_into(
    list: &PropertyList,
    _filter_id: u32,
    out: &mut Vec<u8>,
) -> Result<()> {
    H5P__plist_get_into(list, "filter_pipeline", out)
}

/// Return true if any filter pipeline is set on the property list.
#[allow(non_snake_case)]
pub fn H5P_filter_in_pline(list: &PropertyList, _filter_id: u32) -> Result<bool> {
    Ok(!plist_get_ref(list, "filter_pipeline")?.is_empty())
}

/// Remove all filters from the filter pipeline on a property list.
#[allow(non_snake_case)]
pub fn H5Premove_filter(list: &mut PropertyList, _filter_id: u32) -> Result<()> {
    plist_del(list, "filter_pipeline")
}

/// Read the encoded filter pipeline from a property list.
#[allow(non_snake_case)]
pub fn H5P__get_filter_ref(list: &PropertyList) -> Result<&[u8]> {
    H5P__plist_get_ref(list, "filter_pipeline")
}

/// Read the encoded filter pipeline into a caller-provided buffer.
#[allow(non_snake_case)]
pub fn H5P__get_filter_into(list: &PropertyList, out: &mut Vec<u8>) -> Result<()> {
    H5P__plist_get_into(list, "filter_pipeline", out)
}

/// Set the `ocrt_pipeline` property on a property list.
#[allow(non_snake_case)]
pub fn H5P__ocrt_pipeline_set(list: &mut PropertyList, value: Vec<u8>) -> Result<()> {
    plist_set(list, "object_pipeline", value)
}

/// Get the `ocrt_pipeline` property from a property list.
#[allow(non_snake_case)]
pub fn H5P__ocrt_pipeline_get_ref(list: &PropertyList) -> Result<&[u8]> {
    H5P__plist_get_ref(list, "object_pipeline")
}

/// Get the `ocrt_pipeline` property into a caller-provided buffer.
#[allow(non_snake_case)]
pub fn H5P__ocrt_pipeline_get_into(list: &PropertyList, out: &mut Vec<u8>) -> Result<()> {
    H5P__plist_get_into(list, "object_pipeline", out)
}

/// Encode a `ocrt_pipeline` property value into bytes.
#[allow(non_snake_case)]
pub fn H5P__ocrt_pipeline_enc(value: &[u8]) -> &[u8] {
    value
}

/// Decode a `ocrt_pipeline` property value from bytes.
#[allow(non_snake_case)]
pub fn H5P__ocrt_pipeline_dec(value: &[u8]) -> &[u8] {
    value
}

/// Delete the `ocrt_pipeline` property from a property list.
#[allow(non_snake_case)]
pub fn H5P__ocrt_pipeline_del(list: &mut PropertyList) -> Result<()> {
    plist_del(list, "object_pipeline")
}

/// Copy a `ocrt_pipeline` property value.
#[allow(non_snake_case)]
pub fn H5P__ocrt_pipeline_copy(value: &[u8]) -> &[u8] {
    value
}

/// Compare two `ocrt_pipeline` property values.
#[allow(non_snake_case)]
pub fn H5P__ocrt_pipeline_cmp(left: &[u8], right: &[u8]) -> bool {
    prop_cmp(left, right)
}

/// Close (release storage for) a `ocrt_pipeline` property value.
#[allow(non_snake_case)]
pub fn H5P__ocrt_pipeline_close(_value: &[u8]) {}

/// The splitter VFD is not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5Pset_fapl_splitter(_list: &mut PropertyList) -> Result<()> {
    unsupported_driver("splitter")
}

/// Register properties for the `strcrt` property class.
#[allow(non_snake_case)]
pub fn H5P__strcrt_reg_prop(class: &mut PropertyClass) -> Result<()> {
    H5P__register(class, "strcrt", Vec::new())
}

/// Encode a `strcrt_char_encoding` property value into bytes.
#[allow(non_snake_case)]
pub fn H5P__strcrt_char_encoding_enc(value: u8) -> [u8; 1] {
    encode_uint8_bytes(value)
}

/// Decode a `strcrt_char_encoding` property value from bytes.
#[allow(non_snake_case)]
pub fn H5P__strcrt_char_encoding_dec(value: &[u8]) -> Result<u8> {
    H5P__decode_uint8_t(value)
}

/// The split VFD is not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5Pset_fapl_split(_list: &mut PropertyList) -> Result<()> {
    unsupported_driver("split")
}

/// The multi VFD is not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5Pset_fapl_multi(_list: &mut PropertyList) -> Result<()> {
    unsupported_driver("multi")
}

/// The onion VFD is not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5Pset_fapl_onion(_list: &mut PropertyList) -> Result<()> {
    unsupported_driver("onion")
}

/// The log VFD is not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5Pset_fapl_log(_list: &mut PropertyList) -> Result<()> {
    unsupported_driver("log")
}

/// Set the core file-access driver on the property list.
#[allow(non_snake_case)]
pub fn H5Pset_fapl_core(list: &mut PropertyList) -> Result<()> {
    plist_clear_fapl_driver_config(list)?;
    plist_set(list, "driver", b"core".to_vec())
}

/// Register properties for the `macc` property class.
#[allow(non_snake_case)]
pub fn H5P__macc_reg_prop(class: &mut PropertyClass) -> Result<()> {
    H5P__register(class, "macc", Vec::new())
}

/// Register properties for the `lacc` property class.
#[allow(non_snake_case)]
pub fn H5P__lacc_reg_prop(class: &mut PropertyClass) -> Result<()> {
    H5P__register(class, "lacc", Vec::new())
}

/// Set the `lacc_elink_fapl` property on a property list.
#[allow(non_snake_case)]
pub fn H5P__lacc_elink_fapl_set(list: &mut PropertyList, value: Vec<u8>) -> Result<()> {
    plist_set(list, "external_link_fapl", value)
}

/// Get the `lacc_elink_fapl` property from a property list.
#[allow(non_snake_case)]
pub fn H5P__lacc_elink_fapl_get_ref(list: &PropertyList) -> Result<&[u8]> {
    H5P__plist_get_ref(list, "external_link_fapl")
}

/// Get the `lacc_elink_fapl` property into a caller-provided buffer.
#[allow(non_snake_case)]
pub fn H5P__lacc_elink_fapl_get_into(list: &PropertyList, out: &mut Vec<u8>) -> Result<()> {
    H5P__plist_get_into(list, "external_link_fapl", out)
}

/// Encode a `lacc_elink_fapl` property value into bytes.
#[allow(non_snake_case)]
pub fn H5P__lacc_elink_fapl_enc(value: &[u8]) -> &[u8] {
    value
}

/// Decode a `lacc_elink_fapl` property value from bytes.
#[allow(non_snake_case)]
pub fn H5P__lacc_elink_fapl_dec(value: &[u8]) -> &[u8] {
    value
}

/// Delete the `lacc_elink_fapl` property from a property list.
#[allow(non_snake_case)]
pub fn H5P__lacc_elink_fapl_del(list: &mut PropertyList) -> Result<()> {
    plist_del(list, "external_link_fapl")
}

/// Copy a `lacc_elink_fapl` property value.
#[allow(non_snake_case)]
pub fn H5P__lacc_elink_fapl_copy(value: &[u8]) -> &[u8] {
    value
}

/// Compare two `lacc_elink_fapl` property values.
#[allow(non_snake_case)]
pub fn H5P__lacc_elink_fapl_cmp(left: &[u8], right: &[u8]) -> bool {
    prop_cmp(left, right)
}

/// Close (release storage for) a `lacc_elink_fapl` property value.
#[allow(non_snake_case)]
pub fn H5P__lacc_elink_fapl_close(_value: &[u8]) {}

/// Set the `lacc_elink_pref` property on a property list.
#[allow(non_snake_case)]
pub fn H5P__lacc_elink_pref_set(list: &mut PropertyList, value: Vec<u8>) -> Result<()> {
    plist_set(list, "external_link_prefix", value)
}

/// Get the `lacc_elink_pref` property from a property list.
#[allow(non_snake_case)]
pub fn H5P__lacc_elink_pref_get_ref(list: &PropertyList) -> Result<&[u8]> {
    H5P__plist_get_ref(list, "external_link_prefix")
}

/// Get the `lacc_elink_pref` property into a caller-provided buffer.
#[allow(non_snake_case)]
pub fn H5P__lacc_elink_pref_get_into(list: &PropertyList, out: &mut Vec<u8>) -> Result<()> {
    H5P__plist_get_into(list, "external_link_prefix", out)
}

/// Encode a `lacc_elink_pref` property value into bytes.
#[allow(non_snake_case)]
pub fn H5P__lacc_elink_pref_enc(value: &[u8]) -> &[u8] {
    value
}

/// Decode a `lacc_elink_pref` property value from bytes.
#[allow(non_snake_case)]
pub fn H5P__lacc_elink_pref_dec(value: &[u8]) -> &[u8] {
    value
}

/// Delete the `lacc_elink_pref` property from a property list.
#[allow(non_snake_case)]
pub fn H5P__lacc_elink_pref_del(list: &mut PropertyList) -> Result<()> {
    plist_del(list, "external_link_prefix")
}

/// Copy a `lacc_elink_pref` property value.
#[allow(non_snake_case)]
pub fn H5P__lacc_elink_pref_copy(value: &[u8]) -> &[u8] {
    value
}

/// Compare two `lacc_elink_pref` property values.
#[allow(non_snake_case)]
pub fn H5P__lacc_elink_pref_cmp(left: &[u8], right: &[u8]) -> bool {
    prop_cmp(left, right)
}

/// Close (release storage for) a `lacc_elink_pref` property value.
#[allow(non_snake_case)]
pub fn H5P__lacc_elink_pref_close(_value: &[u8]) {}

/// Set the ROS3 file-access driver with default configuration.
#[allow(non_snake_case)]
pub fn H5Pset_fapl_ros3(list: &mut PropertyList) -> Result<()> {
    H5Pset_fapl_ros3_config(list, Ros3FaplConfig::default())
}

/// Set the ROS3 file-access driver with the given configuration.
#[allow(non_snake_case)]
pub fn H5Pset_fapl_ros3_config(list: &mut PropertyList, config: Ros3FaplConfig) -> Result<()> {
    let mut encoded = Vec::new();
    H5P__encode_ros3_fapl_config_into(&config, &mut encoded)?;
    plist_clear_fapl_driver_config(list)?;
    plist_set(list, "driver", b"ros3".to_vec())?;
    plist_set(list, "fapl_ros3_config", encoded)
}

/// Get the stored ROS3 file-access driver configuration, if any.
#[allow(non_snake_case)]
pub fn H5Pget_fapl_ros3_config(list: &PropertyList) -> Result<Option<Ros3FaplConfig>> {
    let bytes = plist_get_ref(list, "fapl_ros3_config")?;
    if bytes.is_empty() {
        Ok(None)
    } else {
        H5P__decode_ros3_fapl_config(bytes).map(Some)
    }
}

/// Update the security token in the stored ROS3 driver configuration.
#[allow(non_snake_case)]
pub fn H5Pset_fapl_ros3_token(list: &mut PropertyList, token: impl Into<String>) -> Result<()> {
    let mut config = H5Pget_fapl_ros3_config(list)?.unwrap_or_default();
    config.token = Some(token.into());
    H5Pset_fapl_ros3_config(list, config)
}

/// Update the endpoint in the stored ROS3 driver configuration.
#[allow(non_snake_case)]
pub fn H5Pset_fapl_ros3_endpoint(
    list: &mut PropertyList,
    endpoint: impl Into<String>,
) -> Result<()> {
    let mut config = H5Pget_fapl_ros3_config(list)?.unwrap_or_default();
    config.endpoint = Some(endpoint.into());
    H5Pset_fapl_ros3_config(list, config)
}

/// Register properties for the `gcrt` property class.
#[allow(non_snake_case)]
pub fn H5P__gcrt_reg_prop(class: &mut PropertyClass) -> Result<()> {
    H5P__register(class, "gcrt", Vec::new())
}

/// Encode a `gcrt_group_info` property value into bytes.
#[allow(non_snake_case)]
pub fn H5P__gcrt_group_info_enc(value: &[u8]) -> &[u8] {
    value
}

/// Decode a `gcrt_group_info` property value from bytes.
#[allow(non_snake_case)]
pub fn H5P__gcrt_group_info_dec(value: &[u8]) -> &[u8] {
    value
}

/// Encode a `gcrt_link_info` property value into bytes.
#[allow(non_snake_case)]
pub fn H5P__gcrt_link_info_enc(value: &[u8]) -> &[u8] {
    value
}

/// Decode a `gcrt_link_info` property value from bytes.
#[allow(non_snake_case)]
pub fn H5P__gcrt_link_info_dec(value: &[u8]) -> &[u8] {
    value
}

owned_property_get_wrapper!(H5P__get_file_space, H5P__get_file_space_ref);
owned_property_get_wrapper!(H5P__dapl_vds_file_pref_get, H5P__dapl_vds_file_pref_get_ref);
owned_property_get_wrapper!(H5P__dapl_efile_pref_get, H5P__dapl_efile_pref_get_ref);
owned_property_get_wrapper!(
    H5P__ocpy_merge_comm_dt_list_get,
    H5P__ocpy_merge_comm_dt_list_get_ref
);
owned_property_get_wrapper!(H5P__dcrt_fill_value_get, H5P__dcrt_fill_value_get_ref);
owned_property_get_wrapper!(H5P__dcrt_ext_file_list_get, H5P__dcrt_ext_file_list_get_ref);
owned_property_get_wrapper!(H5P__facc_file_driver_get, H5P__facc_file_driver_get_ref);
owned_property_get_wrapper!(
    H5P__facc_file_image_info_get,
    H5P__facc_file_image_info_get_ref
);
owned_property_get_wrapper!(H5P__facc_vol_get, H5P__facc_vol_get_ref);
owned_property_get_wrapper!(H5P__dxfr_xform_get, H5P__dxfr_xform_get_ref);
owned_property_get_wrapper!(H5P__get_filter, H5P__get_filter_ref);
owned_property_get_wrapper!(H5P__ocrt_pipeline_get, H5P__ocrt_pipeline_get_ref);
owned_property_get_wrapper!(H5P__lacc_elink_fapl_get, H5P__lacc_elink_fapl_get_ref);
owned_property_get_wrapper!(H5P__lacc_elink_pref_get, H5P__lacc_elink_pref_get_ref);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn property_class_and_list_track_registered_values() {
        let mut class = H5Pcreate_class("dataset_create", Some("root".into()));
        H5P__register(&mut class, "layout", vec![1]).unwrap();
        H5P__register(&mut class, "fill", vec![0]).unwrap();
        assert_eq!(
            H5P__iter_pclass_names(&class).collect::<Vec<_>>(),
            vec!["fill", "layout"]
        );
        let mut visited = Vec::new();
        H5P__visit_pclass_names(&class, |name| visited.push(name.to_owned()));
        assert_eq!(visited, vec!["fill", "layout"]);
        let mut buffered = Vec::new();
        H5P__iterate_pclass_into(&class, &mut buffered);
        assert_eq!(buffered, vec!["fill".to_string(), "layout".to_string()]);
        let mut appended_names = vec!["stale".to_string()];
        H5P__iterate_pclass_into(&class, &mut appended_names);
        assert_eq!(
            appended_names,
            vec![
                "stale".to_string(),
                "fill".to_string(),
                "layout".to_string()
            ]
        );
        let mut list = H5P__create(&class).unwrap();
        assert_eq!(H5P_peek_ref(&list, "layout").unwrap(), &[1]);
        H5P_set(&mut list, "layout", vec![2]).unwrap();
        assert_eq!(H5P__get_size_plist(&list, "layout").unwrap(), 1);
        let mut copied_layout = b"stale".to_vec();
        H5P__plist_get_into(&list, "layout", &mut copied_layout).unwrap();
        assert_eq!(copied_layout, b"stale\x02");
        H5P__plist_get_into(&list, "missing", &mut copied_layout).unwrap();
        assert_eq!(copied_layout, b"stale\x02");
        let mut closed = list.clone();
        H5P_close(&mut closed);
        assert!(H5P__plist_get_into(&closed, "layout", &mut copied_layout).is_err());
        assert_eq!(copied_layout, b"stale\x02");
        assert!(H5P__open_class_path_test(&class, "root/dataset_create"));
        assert!(H5P_remove(&mut list, "layout").is_ok());
    }

    #[test]
    fn property_delete_and_free_callbacks_clear_owned_storage() {
        let mut class = H5Pcreate_class("dataset_create", None);
        H5P__register(&mut class, "layout", vec![2]).unwrap();
        let mut list = H5P__create(&class).unwrap();
        H5P__dcrt_layout_del(&mut list).unwrap();
        assert!(!H5P_exist_plist(&list, "layout").unwrap());

        let mut prop = H5P__create_prop("owned", vec![1, 2, 3]);
        H5P__free_prop(&mut prop);
        assert!(prop.name.is_empty());
        assert!(prop.value.is_empty());
    }

    #[test]
    fn closed_property_class_rejects_direct_property_helpers() {
        let mut class = H5Pcreate_class("dataset_create", None);
        H5P__register(&mut class, "layout", vec![2]).unwrap();
        H5P__close_class(&mut class);

        let class_get_error = H5P__class_get_ref(&class, "layout").unwrap_err();
        assert!(class_get_error
            .to_string()
            .contains("property class is closed"));
        assert!(H5P__get_size_pclass(&class, "layout").is_err());
        assert!(
            H5P__copy_prop_pclass(&class, &mut H5Pcreate_class("copy", None), "layout").is_err()
        );
        assert!(H5P__class_set(&mut class, "layout", vec![3]).is_err());
        assert!(H5P__unregister(&mut class, "layout").is_err());
        assert_eq!(H5P_get_nprops_pclass(&class), 0);
    }

    #[test]
    fn dataset_io_hyperslab_selection_validates_and_encodes_payload() {
        let class = H5Pcreate_class("dataset_xfer", None);
        let mut list = H5P__create(&class).unwrap();
        H5Pset_dataset_io_hyperslab_selection(
            &mut list,
            2,
            0,
            &[3, 5],
            Some(&[2, 4]),
            &[7, 11],
            None,
        )
        .unwrap();
        let encoded = H5P_peek_ref(&list, "dataset_io_hyperslab_selection").unwrap();
        assert_eq!(&encoded[..8], &[2, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(u64::from_le_bytes(encoded[8..16].try_into().unwrap()), 3);
        assert_eq!(u64::from_le_bytes(encoded[16..24].try_into().unwrap()), 2);
        assert_eq!(u64::from_le_bytes(encoded[24..32].try_into().unwrap()), 7);
        assert_eq!(u64::from_le_bytes(encoded[32..40].try_into().unwrap()), 1);

        assert!(H5Pset_dataset_io_hyperslab_selection(
            &mut list,
            3,
            1,
            &[0, 0, 0],
            None,
            &[1, 1, 1],
            None
        )
        .is_err());
        H5Pset_dataset_io_hyperslab_selection(&mut list, 3, 0, &[0, 0, 0], None, &[1, 1, 1], None)
            .unwrap();
        assert!(H5Pset_dataset_io_hyperslab_selection(
            &mut list,
            1,
            0,
            &[0],
            Some(&[0]),
            &[1],
            None
        )
        .is_err());
    }

    #[test]
    fn property_encoding_roundtrips_scalars() {
        let mut bool_bytes = Vec::new();
        H5P__encode_bool_into(true, &mut bool_bytes);
        assert!(H5P__decode_bool(&bool_bytes).unwrap());
        let mut uint64_bytes = Vec::new();
        H5P__encode_uint64_t_into(42, &mut uint64_bytes);
        assert_eq!(H5P__decode_uint64_t(&uint64_bytes).unwrap(), 42);
        let mut size_bytes = Vec::new();
        H5P__encode_size_t_into(42, &mut size_bytes).unwrap();
        assert_eq!(H5P__decode_size_t(&size_bytes).unwrap(), 42);
        let mut double_bytes = Vec::new();
        H5P__encode_double_into(1.5, &mut double_bytes);
        assert_eq!(H5P__decode_double(&double_bytes).unwrap(), 1.5);
        let mut encoded_chunks = Vec::new();
        encoded_chunks.extend_from_slice(&3u64.to_le_bytes());
        encoded_chunks.extend_from_slice(b"abc");
        encoded_chunks.extend_from_slice(&0u64.to_le_bytes());
        let chunks = [b"abc".as_slice(), &[]];
        let mut encoded = Vec::new();
        H5P__encode_slices_into(chunks, &mut encoded).unwrap();
        assert_eq!(encoded, encoded_chunks);
        assert_eq!(
            H5P__decode_chunks(&encoded_chunks)
                .collect::<Result<Vec<_>>>()
                .unwrap(),
            vec![b"abc".as_slice(), &[]]
        );
        let mut decoded = Vec::new();
        H5P__decode_into(&encoded_chunks, &mut decoded).unwrap();
        assert_eq!(decoded, vec![b"abc".to_vec(), Vec::new()]);
        assert!(H5P__decode_chunks(&encoded_chunks[..10])
            .collect::<Result<Vec<_>>>()
            .is_err());
        assert!(H5P__decode_chunks(&[0; 7])
            .collect::<Result<Vec<_>>>()
            .is_err());
    }

    #[test]
    fn unsupported_remote_vfd_configs_are_stored_not_rejected() {
        let class = H5Pcreate_class("file_access", None);
        let mut list = H5P__create(&class).unwrap();

        let hdfs = HdfsFaplConfig {
            namenode_name: "nn.example.org".into(),
            namenode_port: 8020,
            user_name: "hdf5".into(),
            buffer_size: 65536,
        };
        H5Pset_fapl_hdfs_config(&mut list, hdfs.clone()).unwrap();
        assert_eq!(H5P_peek_ref(&list, "driver").unwrap(), b"hdfs");
        assert_eq!(H5Pget_fapl_hdfs_config(&list).unwrap(), Some(hdfs));

        let ros3 = Ros3FaplConfig {
            endpoint: Some("s3.us-east-1.amazonaws.com".into()),
            region: Some("us-east-1".into()),
            token: None,
        };
        H5Pset_fapl_ros3_config(&mut list, ros3.clone()).unwrap();
        H5Pset_fapl_ros3_token(&mut list, "token").unwrap();
        let stored = H5Pget_fapl_ros3_config(&list).unwrap().unwrap();
        assert_eq!(stored.endpoint, ros3.endpoint);
        assert_eq!(stored.region, ros3.region);
        assert_eq!(stored.token.as_deref(), Some("token"));
    }

    #[test]
    fn fapl_config_decoders_reject_malformed_payloads() {
        let hdfs = HdfsFaplConfig {
            namenode_name: "nn.example.org".into(),
            namenode_port: 8020,
            user_name: "hdf5".into(),
            buffer_size: 65536,
        };
        let mut hdfs_bytes = Vec::new();
        H5P__encode_hdfs_fapl_config_into(&hdfs, &mut hdfs_bytes).unwrap();
        assert_eq!(H5P__decode_hdfs_fapl_config(&hdfs_bytes).unwrap(), hdfs);
        hdfs_bytes.pop();
        assert!(H5P__decode_hdfs_fapl_config(&hdfs_bytes).is_err());

        let mut hdfs_absent_name = Vec::new();
        H5P__encode_hdfs_fapl_config_into(
            &HdfsFaplConfig {
                namenode_name: String::new(),
                namenode_port: 8020,
                user_name: "hdf5".into(),
                buffer_size: 65536,
            },
            &mut hdfs_absent_name,
        )
        .unwrap();
        hdfs_absent_name[0] = 0;
        assert!(H5P__decode_hdfs_fapl_config(&hdfs_absent_name).is_err());

        let ros3 = Ros3FaplConfig {
            endpoint: Some("s3.us-east-1.amazonaws.com".into()),
            region: Some("us-east-1".into()),
            token: None,
        };
        let mut ros3_bytes = Vec::new();
        H5P__encode_ros3_fapl_config_into(&ros3, &mut ros3_bytes).unwrap();
        assert_eq!(H5P__decode_ros3_fapl_config(&ros3_bytes).unwrap(), ros3);
        ros3_bytes.push(0);
        assert!(H5P__decode_ros3_fapl_config(&ros3_bytes).is_err());

        let mut invalid_utf8 = Vec::new();
        encode_optional_string(&mut invalid_utf8, Some("ok"), "test endpoint").unwrap();
        encode_optional_string(&mut invalid_utf8, Some("ok"), "test region").unwrap();
        invalid_utf8.push(1);
        invalid_utf8.extend_from_slice(&1u32.to_le_bytes());
        invalid_utf8.push(0xff);
        assert!(H5P__decode_ros3_fapl_config(&invalid_utf8).is_err());
    }
}
