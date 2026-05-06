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
    pub fn new(name: impl Into<String>, parent: Option<String>) -> Self {
        Self {
            name: name.into(),
            parent,
            properties: BTreeMap::new(),
            closed: false,
        }
    }
}

impl PropertyList {
    pub fn new(class: &PropertyClass) -> Self {
        Self {
            class_name: class.name.clone(),
            properties: class.properties.clone(),
            closed: false,
        }
    }

    fn ensure_open(&self) -> Result<()> {
        if self.closed {
            Err(Error::InvalidFormat("property list is closed".into()))
        } else {
            Ok(())
        }
    }
}

#[allow(non_snake_case)]
pub fn H5P_init_phase1() -> bool {
    true
}

#[allow(non_snake_case)]
pub fn H5P_init_phase2() -> bool {
    true
}

#[allow(non_snake_case)]
pub fn H5P__init_package() -> bool {
    H5P_init_phase1() && H5P_init_phase2()
}

#[allow(non_snake_case)]
pub fn H5P_term_package() {}

#[allow(non_snake_case)]
pub fn H5Pcreate_class(name: impl Into<String>, parent: Option<String>) -> PropertyClass {
    PropertyClass::new(name, parent)
}

#[allow(non_snake_case)]
pub fn H5P__create_class(name: impl Into<String>, parent: Option<String>) -> PropertyClass {
    H5Pcreate_class(name, parent)
}

#[allow(non_snake_case)]
pub fn H5P__close_class_cb(class: &mut PropertyClass) {
    class.closed = true;
    class.properties.clear();
}

#[allow(non_snake_case)]
pub fn H5P__close_class(class: &mut PropertyClass) {
    H5P__close_class_cb(class);
}

#[allow(non_snake_case)]
pub fn H5P__access_class(class: &PropertyClass) -> Result<&PropertyClass> {
    if class.closed {
        Err(Error::InvalidFormat("property class is closed".into()))
    } else {
        Ok(class)
    }
}

#[allow(non_snake_case)]
pub fn H5P__create(class: &PropertyClass) -> Result<PropertyList> {
    H5P__access_class(class)?;
    Ok(PropertyList::new(class))
}

#[allow(non_snake_case)]
pub fn H5P_create_id(class: &PropertyClass) -> Result<PropertyList> {
    H5P__create(class)
}

#[allow(non_snake_case)]
pub fn H5P__new_plist_of_type(class: &PropertyClass) -> Result<PropertyList> {
    H5P__create(class)
}

#[allow(non_snake_case)]
pub fn H5P__close_list_cb(list: &mut PropertyList) {
    list.closed = true;
    list.properties.clear();
}

#[allow(non_snake_case)]
pub fn H5P_close(list: &mut PropertyList) {
    H5P__close_list_cb(list);
}

#[allow(non_snake_case)]
pub fn H5P__create_prop(name: impl Into<String>, value: impl Into<Vec<u8>>) -> Property {
    Property {
        name: name.into(),
        value: value.into(),
    }
}

#[allow(non_snake_case)]
pub fn H5P__add_prop(class: &mut PropertyClass, prop: Property) -> Result<()> {
    if class.closed {
        return Err(Error::InvalidFormat("property class is closed".into()));
    }
    class.properties.insert(prop.name.clone(), prop);
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5P__register_real(
    class: &mut PropertyClass,
    name: impl Into<String>,
    value: impl Into<Vec<u8>>,
) -> Result<()> {
    H5P__add_prop(class, H5P__create_prop(name, value))
}

#[allow(non_snake_case)]
pub fn H5P__register(
    class: &mut PropertyClass,
    name: impl Into<String>,
    value: impl Into<Vec<u8>>,
) -> Result<()> {
    H5P__register_real(class, name, value)
}

#[allow(non_snake_case)]
pub fn H5Pregister1(
    class: &mut PropertyClass,
    name: impl Into<String>,
    value: impl Into<Vec<u8>>,
) -> Result<()> {
    H5P__register(class, name, value)
}

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

#[allow(non_snake_case)]
pub fn H5Pinsert1(
    list: &mut PropertyList,
    name: impl Into<String>,
    value: impl Into<Vec<u8>>,
) -> Result<()> {
    H5P_insert(list, name, value)
}

#[allow(non_snake_case)]
pub fn H5P__dup_prop(prop: &Property) -> Property {
    prop.clone()
}

#[allow(non_snake_case)]
pub fn H5P__copy_prop_pclass(
    src: &PropertyClass,
    dst: &mut PropertyClass,
    name: &str,
) -> Result<()> {
    let prop = H5P__find_prop_pclass(src, name)?.clone();
    H5P__add_prop(dst, prop)
}

#[allow(non_snake_case)]
pub fn H5P__copy_pclass(class: &PropertyClass) -> PropertyClass {
    class.clone()
}

#[allow(non_snake_case)]
pub fn H5P_copy_plist(list: &PropertyList) -> Result<PropertyList> {
    list.ensure_open()?;
    Ok(list.clone())
}

#[allow(non_snake_case)]
pub fn H5Pcopy(list: &PropertyList) -> Result<PropertyList> {
    H5P_copy_plist(list)
}

#[allow(non_snake_case)]
pub fn H5P__find_prop_plist<'a>(list: &'a PropertyList, name: &str) -> Result<&'a Property> {
    list.ensure_open()?;
    list.properties
        .get(name)
        .ok_or_else(|| Error::InvalidFormat(format!("property '{name}' not found")))
}

#[allow(non_snake_case)]
pub fn H5P__find_prop_pclass<'a>(class: &'a PropertyClass, name: &str) -> Result<&'a Property> {
    class
        .properties
        .get(name)
        .ok_or_else(|| Error::InvalidFormat(format!("property '{name}' not found")))
}

#[allow(non_snake_case)]
pub fn H5P__free_prop_cb(_prop: Property) {}

#[allow(non_snake_case)]
pub fn H5P__free_del_name_cb(_name: String) {}

#[allow(non_snake_case)]
pub fn H5P__do_prop_cb1(prop: &Property) -> (&str, &[u8]) {
    (&prop.name, &prop.value)
}

#[allow(non_snake_case)]
pub fn H5P__do_prop(list: &PropertyList, name: &str) -> Result<Vec<u8>> {
    Ok(H5P__find_prop_plist(list, name)?.value.clone())
}

#[allow(non_snake_case)]
pub fn H5P__poke_plist_cb(list: &mut PropertyList, name: &str, value: Vec<u8>) -> Result<()> {
    H5P_set(list, name, value)
}

#[allow(non_snake_case)]
pub fn H5P__poke_pclass_cb(class: &mut PropertyClass, name: &str, value: Vec<u8>) -> Result<()> {
    H5P__register(class, name.to_string(), value)
}

#[allow(non_snake_case)]
pub fn H5P_poke(list: &mut PropertyList, name: &str, value: Vec<u8>) -> Result<()> {
    H5P__poke_plist_cb(list, name, value)
}

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

#[allow(non_snake_case)]
pub fn H5P__class_get(class: &PropertyClass, name: &str) -> Result<Vec<u8>> {
    Ok(H5P__find_prop_pclass(class, name)?.value.clone())
}

#[allow(non_snake_case)]
pub fn H5P__class_set(class: &mut PropertyClass, name: &str, value: Vec<u8>) -> Result<()> {
    let prop = class
        .properties
        .get_mut(name)
        .ok_or_else(|| Error::InvalidFormat(format!("property '{name}' not found")))?;
    prop.value = value;
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5P_exist_plist(list: &PropertyList, name: &str) -> Result<bool> {
    list.ensure_open()?;
    Ok(list.properties.contains_key(name))
}

#[allow(non_snake_case)]
pub fn H5P__exist_pclass(class: &PropertyClass, name: &str) -> bool {
    class.properties.contains_key(name)
}

#[allow(non_snake_case)]
pub fn H5P__get_size_plist(list: &PropertyList, name: &str) -> Result<usize> {
    Ok(H5P__find_prop_plist(list, name)?.value.len())
}

#[allow(non_snake_case)]
pub fn H5P__get_size_pclass(class: &PropertyClass, name: &str) -> Result<usize> {
    Ok(H5P__find_prop_pclass(class, name)?.value.len())
}

#[allow(non_snake_case)]
pub fn H5P__get_nprops_plist(list: &PropertyList) -> Result<usize> {
    list.ensure_open()?;
    Ok(list.properties.len())
}

#[allow(non_snake_case)]
pub fn H5P_get_nprops_pclass(class: &PropertyClass) -> usize {
    class.properties.len()
}

#[allow(non_snake_case)]
pub fn H5P__cmp_prop(left: &Property, right: &Property) -> bool {
    left == right
}

#[allow(non_snake_case)]
pub fn H5P__cmp_class(left: &PropertyClass, right: &PropertyClass) -> bool {
    left.name == right.name && left.parent == right.parent && left.properties == right.properties
}

#[allow(non_snake_case)]
pub fn H5P__cmp_plist_cb(left: &PropertyList, right: &PropertyList) -> bool {
    left.properties == right.properties
}

#[allow(non_snake_case)]
pub fn H5P__cmp_plist(left: &PropertyList, right: &PropertyList) -> bool {
    left.class_name == right.class_name && H5P__cmp_plist_cb(left, right)
}

#[allow(non_snake_case)]
pub fn H5P_class_isa(class: &PropertyClass, ancestor_name: &str) -> bool {
    class.name == ancestor_name || class.parent.as_deref() == Some(ancestor_name)
}

#[allow(non_snake_case)]
pub fn H5P__iterate_pclass_cb(class: &PropertyClass) -> Vec<String> {
    class.properties.keys().cloned().collect()
}

#[allow(non_snake_case)]
pub fn H5P__iterate_pclass(class: &PropertyClass) -> Vec<String> {
    H5P__iterate_pclass_cb(class)
}

#[allow(non_snake_case)]
pub fn H5P__peek_cb(list: &PropertyList, name: &str) -> Result<Vec<u8>> {
    H5P_peek(list, name)
}

#[allow(non_snake_case)]
pub fn H5P_peek(list: &PropertyList, name: &str) -> Result<Vec<u8>> {
    Ok(H5P__find_prop_plist(list, name)?.value.clone())
}

#[allow(non_snake_case)]
pub fn H5P_remove(list: &mut PropertyList, name: &str) -> Result<Property> {
    list.ensure_open()?;
    list.properties
        .remove(name)
        .ok_or_else(|| Error::InvalidFormat(format!("property '{name}' not found")))
}

#[allow(non_snake_case)]
pub fn H5P__unregister(class: &mut PropertyClass, name: &str) -> Result<Property> {
    class
        .properties
        .remove(name)
        .ok_or_else(|| Error::InvalidFormat(format!("property '{name}' not found")))
}

#[allow(non_snake_case)]
pub fn H5P__get_class_path(class: &PropertyClass) -> String {
    match &class.parent {
        Some(parent) => format!("{parent}/{}", class.name),
        None => class.name.clone(),
    }
}

#[allow(non_snake_case)]
pub fn H5P__open_class_path(class: &PropertyClass, path: &str) -> bool {
    H5P__get_class_path(class) == path || class.name == path
}

#[allow(non_snake_case)]
pub fn H5P__get_class_path_test(class: &PropertyClass) -> String {
    H5P__get_class_path(class)
}

#[allow(non_snake_case)]
pub fn H5P__open_class_path_test(class: &PropertyClass, path: &str) -> bool {
    H5P__open_class_path(class, path)
}

#[allow(non_snake_case)]
pub fn H5P__get_class_parent(class: &PropertyClass) -> Option<&str> {
    class.parent.as_deref()
}

#[allow(non_snake_case)]
pub fn H5P_get_plist_id(list: &PropertyList) -> Result<PropertyList> {
    H5P_copy_plist(list)
}

#[allow(non_snake_case)]
pub fn H5P_get_class(list: &PropertyList) -> String {
    list.class_name.clone()
}

#[allow(non_snake_case)]
pub fn H5P__encode_unsigned(value: u64) -> Vec<u8> {
    value.to_le_bytes().to_vec()
}

#[allow(non_snake_case)]
pub fn H5P__encode_uint8_t(value: u8) -> Vec<u8> {
    vec![value]
}

#[allow(non_snake_case)]
pub fn H5P__encode_bool(value: bool) -> Vec<u8> {
    vec![u8::from(value)]
}

#[allow(non_snake_case)]
pub fn H5P__encode_uint64_t(value: u64) -> Vec<u8> {
    value.to_le_bytes().to_vec()
}

#[allow(non_snake_case)]
pub fn H5P__encode_size_t(value: usize) -> Result<Vec<u8>> {
    Ok(u64::try_from(value)
        .map_err(|_| Error::InvalidFormat("size_t property exceeds u64".into()))?
        .to_le_bytes()
        .to_vec())
}

#[allow(non_snake_case)]
pub fn H5P__encode_hsize_t(value: u64) -> Vec<u8> {
    value.to_le_bytes().to_vec()
}

#[allow(non_snake_case)]
pub fn H5P__encode_double(value: f64) -> Vec<u8> {
    value.to_le_bytes().to_vec()
}

#[allow(non_snake_case)]
pub fn H5P__encode(chunks: &[Vec<u8>]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    for chunk in chunks {
        out.extend_from_slice(
            &u64::try_from(chunk.len())
                .map_err(|_| Error::InvalidFormat("property chunk length exceeds u64".into()))?
                .to_le_bytes(),
        );
        out.extend_from_slice(chunk);
    }
    Ok(out)
}

#[allow(non_snake_case)]
pub fn H5P__decode_uint8_t(bytes: &[u8]) -> Result<u8> {
    bytes
        .first()
        .copied()
        .ok_or_else(|| Error::InvalidFormat("missing u8 property".into()))
}

#[allow(non_snake_case)]
pub fn H5P__decode_bool(bytes: &[u8]) -> Result<bool> {
    Ok(H5P__decode_uint8_t(bytes)? != 0)
}

#[allow(non_snake_case)]
pub fn H5P__decode_uint64_t(bytes: &[u8]) -> Result<u64> {
    let raw: [u8; 8] = bytes
        .get(..8)
        .ok_or_else(|| Error::InvalidFormat("truncated u64 property".into()))?
        .try_into()
        .map_err(|_| Error::InvalidFormat("truncated u64 property".into()))?;
    Ok(u64::from_le_bytes(raw))
}

#[allow(non_snake_case)]
pub fn H5P__decode_size_t(bytes: &[u8]) -> Result<usize> {
    usize::try_from(H5P__decode_uint64_t(bytes)?)
        .map_err(|_| Error::InvalidFormat("size_t property does not fit usize".into()))
}

#[allow(non_snake_case)]
pub fn H5P__decode_hsize_t(bytes: &[u8]) -> Result<u64> {
    H5P__decode_uint64_t(bytes)
}

#[allow(non_snake_case)]
pub fn H5P__decode_double(bytes: &[u8]) -> Result<f64> {
    let raw: [u8; 8] = bytes
        .get(..8)
        .ok_or_else(|| Error::InvalidFormat("truncated double property".into()))?
        .try_into()
        .map_err(|_| Error::InvalidFormat("truncated double property".into()))?;
    Ok(f64::from_le_bytes(raw))
}

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

fn plist_get(list: &PropertyList, name: &str) -> Result<Vec<u8>> {
    list.ensure_open()?;
    Ok(list
        .properties
        .get(name)
        .map(|prop| prop.value.clone())
        .unwrap_or_default())
}

fn plist_del(list: &mut PropertyList, name: &str) -> Result<()> {
    list.ensure_open()?;
    list.properties.remove(name);
    Ok(())
}

fn prop_copy(value: &[u8]) -> Vec<u8> {
    value.to_vec()
}

fn prop_cmp(left: &[u8], right: &[u8]) -> bool {
    left == right
}

fn prop_close(_value: Vec<u8>) {}

fn unsupported_driver(name: &str) -> Result<()> {
    Err(Error::Unsupported(format!(
        "{name} driver is not supported by the pure Rust local backend"
    )))
}

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

fn advance_offset(offset: &mut usize, delta: usize, context: &str) -> Result<()> {
    *offset = offset
        .checked_add(delta)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))?;
    Ok(())
}

fn decode_optional_string(
    bytes: &[u8],
    offset: &mut usize,
    context: &str,
) -> Result<Option<String>> {
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
        Ok(Some(
            std::str::from_utf8(value)
                .map_err(|_| Error::InvalidFormat(format!("{context} string is not UTF-8")))?
                .to_string(),
        ))
    } else {
        Err(Error::InvalidFormat(format!(
            "{context} string presence flag is invalid"
        )))
    }
}

#[allow(non_snake_case)]
pub fn H5P__encode_hdfs_fapl_config(config: &HdfsFaplConfig) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    encode_optional_string(&mut out, Some(&config.namenode_name), "HDFS namenode")?;
    out.extend_from_slice(&config.namenode_port.to_le_bytes());
    encode_optional_string(&mut out, Some(&config.user_name), "HDFS user")?;
    out.extend_from_slice(&config.buffer_size.to_le_bytes());
    Ok(out)
}

#[allow(non_snake_case)]
pub fn H5P__decode_hdfs_fapl_config(bytes: &[u8]) -> Result<HdfsFaplConfig> {
    let mut offset = 0;
    let namenode_name = decode_optional_string(bytes, &mut offset, "HDFS namenode")?
        .ok_or_else(|| Error::InvalidFormat("HDFS namenode string is required".into()))?;
    let namenode_port = read_u16_le_at(bytes, offset)
        .ok_or_else(|| Error::InvalidFormat("truncated HDFS namenode port".into()))?;
    advance_offset(&mut offset, 2, "HDFS FAPL config")?;
    let user_name = decode_optional_string(bytes, &mut offset, "HDFS user")?
        .ok_or_else(|| Error::InvalidFormat("HDFS user string is required".into()))?;
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

#[allow(non_snake_case)]
pub fn H5P__encode_ros3_fapl_config(config: &Ros3FaplConfig) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    encode_optional_string(&mut out, config.endpoint.as_deref(), "ROS3 endpoint")?;
    encode_optional_string(&mut out, config.region.as_deref(), "ROS3 region")?;
    encode_optional_string(&mut out, config.token.as_deref(), "ROS3 token")?;
    Ok(out)
}

#[allow(non_snake_case)]
pub fn H5P__decode_ros3_fapl_config(bytes: &[u8]) -> Result<Ros3FaplConfig> {
    let mut offset = 0;
    let endpoint = decode_optional_string(bytes, &mut offset, "ROS3 endpoint")?;
    let region = decode_optional_string(bytes, &mut offset, "ROS3 region")?;
    let token = decode_optional_string(bytes, &mut offset, "ROS3 token")?;
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

fn checked_window(data: &[u8], pos: usize, len: usize) -> Option<&[u8]> {
    let end = pos.checked_add(len)?;
    data.get(pos..end)
}

fn read_u16_le_at(data: &[u8], pos: usize) -> Option<u16> {
    Some(u16::from_le_bytes(
        checked_window(data, pos, 2)?.try_into().ok()?,
    ))
}

fn read_u32_le_at(data: &[u8], pos: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        checked_window(data, pos, 4)?.try_into().ok()?,
    ))
}

#[allow(non_snake_case)]
pub fn H5P__get_file_space(list: &PropertyList) -> Result<Vec<u8>> {
    plist_get(list, "file_space_strategy")
}

#[allow(non_snake_case)]
pub fn H5Pset_fapl_hdfs(list: &mut PropertyList) -> Result<()> {
    H5Pset_fapl_hdfs_config(list, HdfsFaplConfig::default())
}

#[allow(non_snake_case)]
pub fn H5Pset_fapl_hdfs_config(list: &mut PropertyList, config: HdfsFaplConfig) -> Result<()> {
    plist_set(list, "driver", b"hdfs".to_vec())?;
    plist_set(
        list,
        "fapl_hdfs_config",
        H5P__encode_hdfs_fapl_config(&config)?,
    )
}

#[allow(non_snake_case)]
pub fn H5Pget_fapl_hdfs_config(list: &PropertyList) -> Result<Option<HdfsFaplConfig>> {
    let bytes = plist_get(list, "fapl_hdfs_config")?;
    if bytes.is_empty() {
        Ok(None)
    } else {
        H5P__decode_hdfs_fapl_config(&bytes).map(Some)
    }
}

#[allow(non_snake_case)]
pub fn H5Pset_fapl_direct(list: &mut PropertyList) -> Result<()> {
    plist_set(list, "driver", b"direct".to_vec())
}

#[allow(non_snake_case)]
pub fn H5Pset_fapl_mirror(_list: &mut PropertyList) -> Result<()> {
    unsupported_driver("mirror")
}

#[allow(non_snake_case)]
pub fn H5Pset_fapl_mpio(_list: &mut PropertyList) -> Result<()> {
    unsupported_driver("mpio")
}

#[allow(non_snake_case)]
pub fn H5Pset_dxpl_mpio(_list: &mut PropertyList) -> Result<()> {
    unsupported_driver("mpio dxpl")
}

#[allow(non_snake_case)]
pub fn H5Pset_dxpl_mpio_collective_opt(_list: &mut PropertyList) -> Result<()> {
    unsupported_driver("mpio collective")
}

#[allow(non_snake_case)]
pub fn H5Pset_fapl_family(list: &mut PropertyList) -> Result<()> {
    plist_set(list, "driver", b"family".to_vec())
}

#[allow(non_snake_case)]
pub fn H5P__dacc_reg_prop(class: &mut PropertyClass) -> Result<()> {
    H5P__register(class, "dacc", Vec::new())
}

#[allow(non_snake_case)]
pub fn H5P__lcrt_reg_prop(class: &mut PropertyClass) -> Result<()> {
    H5P__register(class, "lcrt", Vec::new())
}

#[allow(non_snake_case)]
pub fn H5P__ocpy_reg_prop(class: &mut PropertyClass) -> Result<()> {
    H5P__register(class, "ocpy", Vec::new())
}

#[allow(non_snake_case)]
pub fn H5P__fcrt_reg_prop(class: &mut PropertyClass) -> Result<()> {
    H5P__register(class, "fcrt", Vec::new())
}

#[allow(non_snake_case)]
pub fn H5P__mcrt_reg_prop(class: &mut PropertyClass) -> Result<()> {
    H5P__register(class, "mcrt", Vec::new())
}

#[allow(non_snake_case)]
pub fn H5P__dcrt_reg_prop(class: &mut PropertyClass) -> Result<()> {
    H5P__register(class, "dcrt", Vec::new())
}

#[allow(non_snake_case)]
pub fn H5P__dapl_vds_file_pref_set(list: &mut PropertyList, value: Vec<u8>) -> Result<()> {
    plist_set(list, "dapl_vds_file_prefix", value)
}

#[allow(non_snake_case)]
pub fn H5P__dapl_vds_file_pref_get(list: &PropertyList) -> Result<Vec<u8>> {
    plist_get(list, "dapl_vds_file_prefix")
}

#[allow(non_snake_case)]
pub fn H5P__dapl_vds_file_pref_enc(value: &[u8]) -> Vec<u8> {
    prop_copy(value)
}

#[allow(non_snake_case)]
pub fn H5P__dapl_vds_file_pref_dec(value: &[u8]) -> Vec<u8> {
    prop_copy(value)
}

#[allow(non_snake_case)]
pub fn H5P__dapl_vds_file_pref_del(list: &mut PropertyList) -> Result<()> {
    plist_del(list, "dapl_vds_file_prefix")
}

#[allow(non_snake_case)]
pub fn H5P__dapl_vds_file_pref_copy(value: &[u8]) -> Vec<u8> {
    prop_copy(value)
}

#[allow(non_snake_case)]
pub fn H5P__dapl_vds_file_pref_close(value: Vec<u8>) {
    prop_close(value)
}

#[allow(non_snake_case)]
pub fn H5P__dapl_efile_pref_set(list: &mut PropertyList, value: Vec<u8>) -> Result<()> {
    plist_set(list, "dapl_external_file_prefix", value)
}

#[allow(non_snake_case)]
pub fn H5P__dapl_efile_pref_get(list: &PropertyList) -> Result<Vec<u8>> {
    plist_get(list, "dapl_external_file_prefix")
}

#[allow(non_snake_case)]
pub fn H5P__dapl_efile_pref_enc(value: &[u8]) -> Vec<u8> {
    prop_copy(value)
}

#[allow(non_snake_case)]
pub fn H5P__dapl_efile_pref_dec(value: &[u8]) -> Vec<u8> {
    prop_copy(value)
}

#[allow(non_snake_case)]
pub fn H5P__dapl_efile_pref_del(list: &mut PropertyList) -> Result<()> {
    plist_del(list, "dapl_external_file_prefix")
}

#[allow(non_snake_case)]
pub fn H5P__dapl_efile_pref_copy(value: &[u8]) -> Vec<u8> {
    prop_copy(value)
}

#[allow(non_snake_case)]
pub fn H5P__dapl_efile_pref_cmp(left: &[u8], right: &[u8]) -> bool {
    prop_cmp(left, right)
}

#[allow(non_snake_case)]
pub fn H5P__dapl_efile_pref_close(value: Vec<u8>) {
    prop_close(value)
}

#[allow(non_snake_case)]
pub fn H5P__dacc_vds_view_enc(value: u8) -> Vec<u8> {
    H5P__encode_uint8_t(value)
}

#[allow(non_snake_case)]
pub fn H5P__dacc_vds_view_dec(value: &[u8]) -> Result<u8> {
    H5P__decode_uint8_t(value)
}

#[allow(non_snake_case)]
pub fn H5P__copy_merge_comm_dt_list(value: &[u8]) -> Vec<u8> {
    prop_copy(value)
}

#[allow(non_snake_case)]
pub fn H5P__ocpy_merge_comm_dt_list_set(list: &mut PropertyList, value: Vec<u8>) -> Result<()> {
    plist_set(list, "ocpy_merge_committed_datatypes", value)
}

#[allow(non_snake_case)]
pub fn H5P__ocpy_merge_comm_dt_list_get(list: &PropertyList) -> Result<Vec<u8>> {
    plist_get(list, "ocpy_merge_committed_datatypes")
}

#[allow(non_snake_case)]
pub fn H5P__ocpy_merge_comm_dt_list_enc(value: &[u8]) -> Vec<u8> {
    prop_copy(value)
}

#[allow(non_snake_case)]
pub fn H5P__ocpy_merge_comm_dt_list_dec(value: &[u8]) -> Vec<u8> {
    prop_copy(value)
}

#[allow(non_snake_case)]
pub fn H5P__ocpy_merge_comm_dt_list_del(list: &mut PropertyList) -> Result<()> {
    plist_del(list, "ocpy_merge_committed_datatypes")
}

#[allow(non_snake_case)]
pub fn H5P__ocpy_merge_comm_dt_list_copy(value: &[u8]) -> Vec<u8> {
    prop_copy(value)
}

#[allow(non_snake_case)]
pub fn H5P__ocpy_merge_comm_dt_list_cmp(left: &[u8], right: &[u8]) -> bool {
    prop_cmp(left, right)
}

#[allow(non_snake_case)]
pub fn H5P__ocpy_merge_comm_dt_list_close(value: Vec<u8>) {
    prop_close(value)
}

#[allow(non_snake_case)]
pub fn H5P__fcrt_btree_rank_dec(value: &[u8]) -> Result<u8> {
    H5P__decode_uint8_t(value)
}

#[allow(non_snake_case)]
pub fn H5P__fcrt_shmsg_index_types_enc(value: u64) -> Vec<u8> {
    H5P__encode_uint64_t(value)
}

#[allow(non_snake_case)]
pub fn H5P__fcrt_shmsg_index_minsize_enc(value: u64) -> Vec<u8> {
    H5P__encode_uint64_t(value)
}

#[allow(non_snake_case)]
pub fn H5P__set_file_space_strategy(list: &mut PropertyList, value: Vec<u8>) -> Result<()> {
    plist_set(list, "file_space_strategy", value)
}

#[allow(non_snake_case)]
pub fn H5P__fcrt_fspace_strategy_dec(value: &[u8]) -> Vec<u8> {
    prop_copy(value)
}

#[allow(non_snake_case)]
pub fn H5P__dcrt_layout_enc(value: u8) -> Vec<u8> {
    H5P__encode_uint8_t(value)
}

#[allow(non_snake_case)]
pub fn H5P__dcrt_layout_dec(value: &[u8]) -> Result<u8> {
    H5P__decode_uint8_t(value)
}

#[allow(non_snake_case)]
pub fn H5P__dcrt_layout_copy(value: &[u8]) -> Vec<u8> {
    prop_copy(value)
}

#[allow(non_snake_case)]
pub fn H5P__dcrt_layout_cmp(left: &[u8], right: &[u8]) -> bool {
    prop_cmp(left, right)
}

#[allow(non_snake_case)]
pub fn H5P__dcrt_layout_close(value: Vec<u8>) {
    prop_close(value)
}

#[allow(non_snake_case)]
pub fn H5P__set_layout(list: &mut PropertyList, layout: u8) -> Result<()> {
    plist_set(list, "layout", H5P__dcrt_layout_enc(layout))
}

#[allow(non_snake_case)]
pub fn H5P__dcrt_fill_value_set(list: &mut PropertyList, value: Vec<u8>) -> Result<()> {
    plist_set(list, "fill_value", value)
}

#[allow(non_snake_case)]
pub fn H5P__dcrt_fill_value_get(list: &PropertyList) -> Result<Vec<u8>> {
    plist_get(list, "fill_value")
}

#[allow(non_snake_case)]
pub fn H5P__dcrt_fill_value_enc(value: &[u8]) -> Vec<u8> {
    prop_copy(value)
}

#[allow(non_snake_case)]
pub fn H5P__dcrt_fill_value_dec(value: &[u8]) -> Vec<u8> {
    prop_copy(value)
}

#[allow(non_snake_case)]
pub fn H5P__dcrt_fill_value_del(list: &mut PropertyList) -> Result<()> {
    plist_del(list, "fill_value")
}

#[allow(non_snake_case)]
pub fn H5P_fill_value_cmp(left: &[u8], right: &[u8]) -> bool {
    prop_cmp(left, right)
}

#[allow(non_snake_case)]
pub fn H5P__dcrt_fill_value_close(value: Vec<u8>) {
    prop_close(value)
}

#[allow(non_snake_case)]
pub fn H5P_get_fill_value(list: &PropertyList) -> Result<Vec<u8>> {
    H5P__dcrt_fill_value_get(list)
}

#[allow(non_snake_case)]
pub fn H5P_is_fill_value_defined(list: &PropertyList) -> Result<bool> {
    Ok(!H5P__dcrt_fill_value_get(list)?.is_empty())
}

#[allow(non_snake_case)]
pub fn H5P_fill_value_defined(list: &PropertyList) -> Result<bool> {
    H5P_is_fill_value_defined(list)
}

#[allow(non_snake_case)]
pub fn H5Pfill_value_defined(list: &PropertyList) -> Result<bool> {
    H5P_is_fill_value_defined(list)
}

#[allow(non_snake_case)]
pub fn H5P__dcrt_ext_file_list_set(list: &mut PropertyList, value: Vec<u8>) -> Result<()> {
    plist_set(list, "external_file_list", value)
}

#[allow(non_snake_case)]
pub fn H5P__dcrt_ext_file_list_get(list: &PropertyList) -> Result<Vec<u8>> {
    plist_get(list, "external_file_list")
}

#[allow(non_snake_case)]
pub fn H5P__dcrt_ext_file_list_enc(value: &[u8]) -> Vec<u8> {
    prop_copy(value)
}

#[allow(non_snake_case)]
pub fn H5P__dcrt_ext_file_list_dec(value: &[u8]) -> Vec<u8> {
    prop_copy(value)
}

#[allow(non_snake_case)]
pub fn H5P__dcrt_ext_file_list_del(list: &mut PropertyList) -> Result<()> {
    plist_del(list, "external_file_list")
}

#[allow(non_snake_case)]
pub fn H5P__dcrt_ext_file_list_copy(value: &[u8]) -> Vec<u8> {
    prop_copy(value)
}

#[allow(non_snake_case)]
pub fn H5P__dcrt_ext_file_list_cmp(left: &[u8], right: &[u8]) -> bool {
    prop_cmp(left, right)
}

#[allow(non_snake_case)]
pub fn H5P__dcrt_ext_file_list_close(value: Vec<u8>) {
    prop_close(value)
}

#[allow(non_snake_case)]
pub fn H5P__facc_reg_prop(class: &mut PropertyClass) -> Result<()> {
    H5P__register(class, "facc", Vec::new())
}

#[allow(non_snake_case)]
pub fn H5P__facc_set_def_driver(list: &mut PropertyList) -> Result<()> {
    plist_set(list, "driver", b"sec2".to_vec())
}

#[allow(non_snake_case)]
pub fn H5P__facc_set_def_driver_check_predefined(name: &str) -> bool {
    matches!(name, "sec2" | "stdio" | "core" | "direct" | "family")
}

#[allow(non_snake_case)]
pub fn H5P_set_driver(list: &mut PropertyList, name: &str) -> Result<()> {
    if H5P__facc_set_def_driver_check_predefined(name) {
        plist_set(list, "driver", name.as_bytes().to_vec())
    } else {
        unsupported_driver(name)
    }
}

#[allow(non_snake_case)]
pub fn H5P_set_driver_by_name(list: &mut PropertyList, name: &str) -> Result<()> {
    H5P_set_driver(list, name)
}

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

#[allow(non_snake_case)]
pub fn H5P__facc_file_driver_get(list: &PropertyList) -> Result<Vec<u8>> {
    plist_get(list, "driver")
}

#[allow(non_snake_case)]
pub fn H5Pset_multi_type(list: &mut PropertyList, value: u8) -> Result<()> {
    plist_set(list, "multi_type", H5P__encode_uint8_t(value))
}

#[allow(non_snake_case)]
pub fn H5P__facc_cache_image_config_cmp(left: &[u8], right: &[u8]) -> bool {
    prop_cmp(left, right)
}

#[allow(non_snake_case)]
pub fn H5P__facc_cache_image_config_enc(value: &[u8]) -> Vec<u8> {
    prop_copy(value)
}

#[allow(non_snake_case)]
pub fn H5P__facc_cache_image_config_dec(value: &[u8]) -> Vec<u8> {
    prop_copy(value)
}

#[allow(non_snake_case)]
pub fn H5P__facc_file_image_info_set(list: &mut PropertyList, value: Vec<u8>) -> Result<()> {
    plist_set(list, "file_image_info", value)
}

#[allow(non_snake_case)]
pub fn H5P__facc_file_image_info_get(list: &PropertyList) -> Result<Vec<u8>> {
    plist_get(list, "file_image_info")
}

#[allow(non_snake_case)]
pub fn H5P__facc_file_image_info_del(list: &mut PropertyList) -> Result<()> {
    plist_del(list, "file_image_info")
}

#[allow(non_snake_case)]
pub fn H5P__facc_file_image_info_cmp(left: &[u8], right: &[u8]) -> bool {
    prop_cmp(left, right)
}

#[allow(non_snake_case)]
pub fn H5P__facc_cache_config_enc(value: &[u8]) -> Vec<u8> {
    prop_copy(value)
}

#[allow(non_snake_case)]
pub fn H5P__facc_cache_config_dec(value: &[u8]) -> Vec<u8> {
    prop_copy(value)
}

#[allow(non_snake_case)]
pub fn H5P__facc_fclose_degree_enc(value: u8) -> Vec<u8> {
    H5P__encode_uint8_t(value)
}

#[allow(non_snake_case)]
pub fn H5P__facc_fclose_degree_dec(value: &[u8]) -> Result<u8> {
    H5P__decode_uint8_t(value)
}

#[allow(non_snake_case)]
pub fn H5P__facc_multi_type_enc(value: u8) -> Vec<u8> {
    H5P__encode_uint8_t(value)
}

#[allow(non_snake_case)]
pub fn H5P__facc_multi_type_dec(value: &[u8]) -> Result<u8> {
    H5P__decode_uint8_t(value)
}

#[allow(non_snake_case)]
pub fn H5P__facc_libver_type_enc(value: u8) -> Vec<u8> {
    H5P__encode_uint8_t(value)
}

#[allow(non_snake_case)]
pub fn H5P__facc_libver_type_dec(value: &[u8]) -> Result<u8> {
    H5P__decode_uint8_t(value)
}

#[allow(non_snake_case)]
pub fn H5P__facc_mdc_log_location_enc(value: &[u8]) -> Vec<u8> {
    prop_copy(value)
}

#[allow(non_snake_case)]
pub fn H5P__facc_mdc_log_location_dec(value: &[u8]) -> Vec<u8> {
    prop_copy(value)
}

#[allow(non_snake_case)]
pub fn H5P__facc_mdc_log_location_copy(value: &[u8]) -> Vec<u8> {
    prop_copy(value)
}

#[allow(non_snake_case)]
pub fn H5P__facc_mdc_log_location_cmp(left: &[u8], right: &[u8]) -> bool {
    prop_cmp(left, right)
}

#[allow(non_snake_case)]
pub fn H5P__facc_mdc_log_location_close(value: Vec<u8>) {
    prop_close(value)
}

#[allow(non_snake_case)]
pub fn H5Pset_mpi_params(_list: &mut PropertyList) -> Result<()> {
    unsupported_driver("mpi params")
}

#[allow(non_snake_case)]
pub fn H5P__facc_mpi_comm_set(_list: &mut PropertyList, _value: Vec<u8>) -> Result<()> {
    unsupported_driver("mpi communicator")
}

#[allow(non_snake_case)]
pub fn H5P__facc_mpi_comm_get(_list: &PropertyList) -> Result<Vec<u8>> {
    Err(Error::Unsupported(
        "mpi communicator driver is not supported by the pure Rust local backend".into(),
    ))
}

#[allow(non_snake_case)]
pub fn H5P__facc_mpi_comm_del(_list: &mut PropertyList) -> Result<()> {
    unsupported_driver("mpi communicator")
}

#[allow(non_snake_case)]
pub fn H5P__facc_mpi_comm_copy(value: &[u8]) -> Vec<u8> {
    prop_copy(value)
}

#[allow(non_snake_case)]
pub fn H5P__facc_mpi_comm_cmp(left: &[u8], right: &[u8]) -> bool {
    prop_cmp(left, right)
}

#[allow(non_snake_case)]
pub fn H5P__facc_mpi_comm_close(value: Vec<u8>) {
    prop_close(value)
}

#[allow(non_snake_case)]
pub fn H5P__facc_mpi_info_set(_list: &mut PropertyList, _value: Vec<u8>) -> Result<()> {
    unsupported_driver("mpi info")
}

#[allow(non_snake_case)]
pub fn H5P__facc_mpi_info_get(_list: &PropertyList) -> Result<Vec<u8>> {
    Err(Error::Unsupported(
        "mpi info driver is not supported by the pure Rust local backend".into(),
    ))
}

#[allow(non_snake_case)]
pub fn H5P__facc_mpi_info_del(_list: &mut PropertyList) -> Result<()> {
    unsupported_driver("mpi info")
}

#[allow(non_snake_case)]
pub fn H5P__facc_mpi_info_copy(value: &[u8]) -> Vec<u8> {
    prop_copy(value)
}

#[allow(non_snake_case)]
pub fn H5P__facc_mpi_info_cmp(left: &[u8], right: &[u8]) -> bool {
    prop_cmp(left, right)
}

#[allow(non_snake_case)]
pub fn H5P__facc_mpi_info_close(value: Vec<u8>) {
    prop_close(value)
}

#[allow(non_snake_case)]
pub fn H5P__facc_page_buffer_size_enc(value: u64) -> Vec<u8> {
    H5P__encode_uint64_t(value)
}

#[allow(non_snake_case)]
pub fn H5P__facc_page_buffer_size_dec(value: &[u8]) -> Result<u64> {
    H5P__decode_uint64_t(value)
}

#[allow(non_snake_case)]
pub fn H5P_set_vol(_list: &mut PropertyList, _value: Vec<u8>) -> Result<()> {
    Err(Error::Unsupported(
        "VOL connectors are not supported".into(),
    ))
}

#[allow(non_snake_case)]
pub fn H5P_reset_vol_class(list: &mut PropertyList) -> Result<()> {
    plist_del(list, "vol")
}

#[allow(non_snake_case)]
pub fn H5Pset_vol(list: &mut PropertyList, value: Vec<u8>) -> Result<()> {
    H5P_set_vol(list, value)
}

#[allow(non_snake_case)]
pub fn H5P__facc_vol_get(list: &PropertyList) -> Result<Vec<u8>> {
    plist_get(list, "vol")
}

#[allow(non_snake_case)]
pub fn H5Pset_fapl_ioc(_list: &mut PropertyList) -> Result<()> {
    unsupported_driver("ioc")
}

#[allow(non_snake_case)]
pub fn H5Pset_fapl_subfiling(_list: &mut PropertyList) -> Result<()> {
    unsupported_driver("subfiling")
}

#[allow(non_snake_case)]
pub fn H5P__dxfr_reg_prop(class: &mut PropertyClass) -> Result<()> {
    H5P__register(class, "dxfr", Vec::new())
}

#[allow(non_snake_case)]
pub fn H5P__dxfr_bkgr_buf_type_enc(value: u8) -> Vec<u8> {
    H5P__encode_uint8_t(value)
}

#[allow(non_snake_case)]
pub fn H5P__dxfr_bkgr_buf_type_dec(value: &[u8]) -> Result<u8> {
    H5P__decode_uint8_t(value)
}

#[allow(non_snake_case)]
pub fn H5P__dxfr_btree_split_ratio_enc(value: f64) -> Vec<u8> {
    H5P__encode_double(value)
}

#[allow(non_snake_case)]
pub fn H5P__dxfr_btree_split_ratio_dec(value: &[u8]) -> Result<f64> {
    H5P__decode_double(value)
}

#[allow(non_snake_case)]
pub fn H5P__dxfr_xform_set(list: &mut PropertyList, value: Vec<u8>) -> Result<()> {
    plist_set(list, "data_transform", value)
}

#[allow(non_snake_case)]
pub fn H5P__dxfr_xform_get(list: &PropertyList) -> Result<Vec<u8>> {
    plist_get(list, "data_transform")
}

#[allow(non_snake_case)]
pub fn H5P__dxfr_xform_enc(value: &[u8]) -> Vec<u8> {
    prop_copy(value)
}

#[allow(non_snake_case)]
pub fn H5P__dxfr_xform_dec(value: &[u8]) -> Vec<u8> {
    prop_copy(value)
}

#[allow(non_snake_case)]
pub fn H5P__dxfr_xform_del(list: &mut PropertyList) -> Result<()> {
    plist_del(list, "data_transform")
}

#[allow(non_snake_case)]
pub fn H5P__dxfr_xform_copy(value: &[u8]) -> Vec<u8> {
    prop_copy(value)
}

#[allow(non_snake_case)]
pub fn H5P__dxfr_xform_cmp(left: &[u8], right: &[u8]) -> bool {
    prop_cmp(left, right)
}

#[allow(non_snake_case)]
pub fn H5P__dxfr_xform_close(value: Vec<u8>) {
    prop_close(value)
}

#[allow(non_snake_case)]
pub fn H5P_set_vlen_mem_manager(list: &mut PropertyList, value: Vec<u8>) -> Result<()> {
    plist_set(list, "vlen_mem_manager", value)
}

#[allow(non_snake_case)]
pub fn H5P__dxfr_io_xfer_mode_enc(value: u8) -> Vec<u8> {
    H5P__encode_uint8_t(value)
}

#[allow(non_snake_case)]
pub fn H5P__dxfr_io_xfer_mode_dec(value: &[u8]) -> Result<u8> {
    H5P__decode_uint8_t(value)
}

#[allow(non_snake_case)]
pub fn H5P__dxfr_mpio_collective_opt_enc(value: u8) -> Vec<u8> {
    H5P__encode_uint8_t(value)
}

#[allow(non_snake_case)]
pub fn H5P__dxfr_mpio_collective_opt_dec(value: &[u8]) -> Result<u8> {
    H5P__decode_uint8_t(value)
}

#[allow(non_snake_case)]
pub fn H5P__dxfr_mpio_chunk_opt_hard_enc(value: u8) -> Vec<u8> {
    H5P__encode_uint8_t(value)
}

#[allow(non_snake_case)]
pub fn H5P__dxfr_mpio_chunk_opt_hard_dec(value: &[u8]) -> Result<u8> {
    H5P__decode_uint8_t(value)
}

#[allow(non_snake_case)]
pub fn H5P__dxfr_edc_enc(value: u8) -> Vec<u8> {
    H5P__encode_uint8_t(value)
}

#[allow(non_snake_case)]
pub fn H5P__dxfr_edc_dec(value: &[u8]) -> Result<u8> {
    H5P__decode_uint8_t(value)
}

#[allow(non_snake_case)]
pub fn H5P__dxfr_dset_io_hyp_sel_copy(value: &[u8]) -> Vec<u8> {
    prop_copy(value)
}

#[allow(non_snake_case)]
pub fn H5P__dxfr_dset_io_hyp_sel_cmp(left: &[u8], right: &[u8]) -> bool {
    prop_cmp(left, right)
}

#[allow(non_snake_case)]
pub fn H5P__dxfr_dset_io_hyp_sel_close(value: Vec<u8>) {
    prop_close(value)
}

#[allow(non_snake_case)]
pub fn H5P__dxfr_selection_io_mode_enc(value: u8) -> Vec<u8> {
    H5P__encode_uint8_t(value)
}

#[allow(non_snake_case)]
pub fn H5P__dxfr_modify_write_buf_enc(value: bool) -> Vec<u8> {
    H5P__encode_bool(value)
}

#[allow(non_snake_case)]
pub fn H5P__dxfr_modify_write_buf_dec(value: &[u8]) -> Result<bool> {
    H5P__decode_bool(value)
}

#[allow(non_snake_case)]
pub fn H5P__fmnt_reg_prop(class: &mut PropertyClass) -> Result<()> {
    H5P__register(class, "fmnt", Vec::new())
}

#[allow(non_snake_case)]
pub fn H5P__ocrt_reg_prop(class: &mut PropertyClass) -> Result<()> {
    H5P__register(class, "ocrt", Vec::new())
}

#[allow(non_snake_case)]
pub fn H5P_modify_filter(list: &mut PropertyList, filter: Vec<u8>) -> Result<()> {
    plist_set(list, "filter_pipeline", filter)
}

#[allow(non_snake_case)]
pub fn H5Pmodify_filter(list: &mut PropertyList, filter: Vec<u8>) -> Result<()> {
    H5P_modify_filter(list, filter)
}

#[allow(non_snake_case)]
pub fn H5P__set_filter(list: &mut PropertyList, filter: Vec<u8>) -> Result<()> {
    H5P_modify_filter(list, filter)
}

#[allow(non_snake_case)]
pub fn H5P_get_filter_by_id(list: &PropertyList, _filter_id: u32) -> Result<Vec<u8>> {
    plist_get(list, "filter_pipeline")
}

#[allow(non_snake_case)]
pub fn H5P_filter_in_pline(list: &PropertyList, _filter_id: u32) -> Result<bool> {
    Ok(!plist_get(list, "filter_pipeline")?.is_empty())
}

#[allow(non_snake_case)]
pub fn H5Premove_filter(list: &mut PropertyList, _filter_id: u32) -> Result<()> {
    plist_del(list, "filter_pipeline")
}

#[allow(non_snake_case)]
pub fn H5P__get_filter(list: &PropertyList) -> Result<Vec<u8>> {
    plist_get(list, "filter_pipeline")
}

#[allow(non_snake_case)]
pub fn H5P__ocrt_pipeline_set(list: &mut PropertyList, value: Vec<u8>) -> Result<()> {
    plist_set(list, "object_pipeline", value)
}

#[allow(non_snake_case)]
pub fn H5P__ocrt_pipeline_get(list: &PropertyList) -> Result<Vec<u8>> {
    plist_get(list, "object_pipeline")
}

#[allow(non_snake_case)]
pub fn H5P__ocrt_pipeline_enc(value: &[u8]) -> Vec<u8> {
    prop_copy(value)
}

#[allow(non_snake_case)]
pub fn H5P__ocrt_pipeline_dec(value: &[u8]) -> Vec<u8> {
    prop_copy(value)
}

#[allow(non_snake_case)]
pub fn H5P__ocrt_pipeline_del(list: &mut PropertyList) -> Result<()> {
    plist_del(list, "object_pipeline")
}

#[allow(non_snake_case)]
pub fn H5P__ocrt_pipeline_copy(value: &[u8]) -> Vec<u8> {
    prop_copy(value)
}

#[allow(non_snake_case)]
pub fn H5P__ocrt_pipeline_cmp(left: &[u8], right: &[u8]) -> bool {
    prop_cmp(left, right)
}

#[allow(non_snake_case)]
pub fn H5P__ocrt_pipeline_close(value: Vec<u8>) {
    prop_close(value)
}

#[allow(non_snake_case)]
pub fn H5Pset_fapl_splitter(_list: &mut PropertyList) -> Result<()> {
    unsupported_driver("splitter")
}

#[allow(non_snake_case)]
pub fn H5P__strcrt_reg_prop(class: &mut PropertyClass) -> Result<()> {
    H5P__register(class, "strcrt", Vec::new())
}

#[allow(non_snake_case)]
pub fn H5P__strcrt_char_encoding_enc(value: u8) -> Vec<u8> {
    H5P__encode_uint8_t(value)
}

#[allow(non_snake_case)]
pub fn H5P__strcrt_char_encoding_dec(value: &[u8]) -> Result<u8> {
    H5P__decode_uint8_t(value)
}

#[allow(non_snake_case)]
pub fn H5Pset_fapl_split(_list: &mut PropertyList) -> Result<()> {
    unsupported_driver("split")
}

#[allow(non_snake_case)]
pub fn H5Pset_fapl_multi(_list: &mut PropertyList) -> Result<()> {
    unsupported_driver("multi")
}

#[allow(non_snake_case)]
pub fn H5Pset_fapl_onion(_list: &mut PropertyList) -> Result<()> {
    unsupported_driver("onion")
}

#[allow(non_snake_case)]
pub fn H5Pset_fapl_log(_list: &mut PropertyList) -> Result<()> {
    unsupported_driver("log")
}

#[allow(non_snake_case)]
pub fn H5Pset_fapl_core(list: &mut PropertyList) -> Result<()> {
    plist_set(list, "driver", b"core".to_vec())
}

#[allow(non_snake_case)]
pub fn H5P__macc_reg_prop(class: &mut PropertyClass) -> Result<()> {
    H5P__register(class, "macc", Vec::new())
}

#[allow(non_snake_case)]
pub fn H5P__lacc_reg_prop(class: &mut PropertyClass) -> Result<()> {
    H5P__register(class, "lacc", Vec::new())
}

#[allow(non_snake_case)]
pub fn H5P__lacc_elink_fapl_set(list: &mut PropertyList, value: Vec<u8>) -> Result<()> {
    plist_set(list, "external_link_fapl", value)
}

#[allow(non_snake_case)]
pub fn H5P__lacc_elink_fapl_get(list: &PropertyList) -> Result<Vec<u8>> {
    plist_get(list, "external_link_fapl")
}

#[allow(non_snake_case)]
pub fn H5P__lacc_elink_fapl_enc(value: &[u8]) -> Vec<u8> {
    prop_copy(value)
}

#[allow(non_snake_case)]
pub fn H5P__lacc_elink_fapl_dec(value: &[u8]) -> Vec<u8> {
    prop_copy(value)
}

#[allow(non_snake_case)]
pub fn H5P__lacc_elink_fapl_del(list: &mut PropertyList) -> Result<()> {
    plist_del(list, "external_link_fapl")
}

#[allow(non_snake_case)]
pub fn H5P__lacc_elink_fapl_copy(value: &[u8]) -> Vec<u8> {
    prop_copy(value)
}

#[allow(non_snake_case)]
pub fn H5P__lacc_elink_fapl_cmp(left: &[u8], right: &[u8]) -> bool {
    prop_cmp(left, right)
}

#[allow(non_snake_case)]
pub fn H5P__lacc_elink_fapl_close(value: Vec<u8>) {
    prop_close(value)
}

#[allow(non_snake_case)]
pub fn H5P__lacc_elink_pref_set(list: &mut PropertyList, value: Vec<u8>) -> Result<()> {
    plist_set(list, "external_link_prefix", value)
}

#[allow(non_snake_case)]
pub fn H5P__lacc_elink_pref_get(list: &PropertyList) -> Result<Vec<u8>> {
    plist_get(list, "external_link_prefix")
}

#[allow(non_snake_case)]
pub fn H5P__lacc_elink_pref_enc(value: &[u8]) -> Vec<u8> {
    prop_copy(value)
}

#[allow(non_snake_case)]
pub fn H5P__lacc_elink_pref_dec(value: &[u8]) -> Vec<u8> {
    prop_copy(value)
}

#[allow(non_snake_case)]
pub fn H5P__lacc_elink_pref_del(list: &mut PropertyList) -> Result<()> {
    plist_del(list, "external_link_prefix")
}

#[allow(non_snake_case)]
pub fn H5P__lacc_elink_pref_copy(value: &[u8]) -> Vec<u8> {
    prop_copy(value)
}

#[allow(non_snake_case)]
pub fn H5P__lacc_elink_pref_cmp(left: &[u8], right: &[u8]) -> bool {
    prop_cmp(left, right)
}

#[allow(non_snake_case)]
pub fn H5P__lacc_elink_pref_close(value: Vec<u8>) {
    prop_close(value)
}

#[allow(non_snake_case)]
pub fn H5Pset_fapl_ros3(list: &mut PropertyList) -> Result<()> {
    H5Pset_fapl_ros3_config(list, Ros3FaplConfig::default())
}

#[allow(non_snake_case)]
pub fn H5Pset_fapl_ros3_config(list: &mut PropertyList, config: Ros3FaplConfig) -> Result<()> {
    plist_set(list, "driver", b"ros3".to_vec())?;
    plist_set(
        list,
        "fapl_ros3_config",
        H5P__encode_ros3_fapl_config(&config)?,
    )
}

#[allow(non_snake_case)]
pub fn H5Pget_fapl_ros3_config(list: &PropertyList) -> Result<Option<Ros3FaplConfig>> {
    let bytes = plist_get(list, "fapl_ros3_config")?;
    if bytes.is_empty() {
        Ok(None)
    } else {
        H5P__decode_ros3_fapl_config(&bytes).map(Some)
    }
}

#[allow(non_snake_case)]
pub fn H5Pset_fapl_ros3_token(list: &mut PropertyList, token: impl Into<String>) -> Result<()> {
    let mut config = H5Pget_fapl_ros3_config(list)?.unwrap_or_default();
    config.token = Some(token.into());
    H5Pset_fapl_ros3_config(list, config)
}

#[allow(non_snake_case)]
pub fn H5Pset_fapl_ros3_endpoint(
    list: &mut PropertyList,
    endpoint: impl Into<String>,
) -> Result<()> {
    let mut config = H5Pget_fapl_ros3_config(list)?.unwrap_or_default();
    config.endpoint = Some(endpoint.into());
    H5Pset_fapl_ros3_config(list, config)
}

#[allow(non_snake_case)]
pub fn H5P__gcrt_reg_prop(class: &mut PropertyClass) -> Result<()> {
    H5P__register(class, "gcrt", Vec::new())
}

#[allow(non_snake_case)]
pub fn H5P__gcrt_group_info_enc(value: &[u8]) -> Vec<u8> {
    prop_copy(value)
}

#[allow(non_snake_case)]
pub fn H5P__gcrt_group_info_dec(value: &[u8]) -> Vec<u8> {
    prop_copy(value)
}

#[allow(non_snake_case)]
pub fn H5P__gcrt_link_info_enc(value: &[u8]) -> Vec<u8> {
    prop_copy(value)
}

#[allow(non_snake_case)]
pub fn H5P__gcrt_link_info_dec(value: &[u8]) -> Vec<u8> {
    prop_copy(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn property_class_and_list_track_registered_values() {
        let mut class = H5Pcreate_class("dataset_create", Some("root".into()));
        H5Pregister1(&mut class, "layout", vec![1]).unwrap();
        let mut list = H5P__create(&class).unwrap();
        assert_eq!(H5P_peek(&list, "layout").unwrap(), vec![1]);
        H5P_set(&mut list, "layout", vec![2]).unwrap();
        assert_eq!(H5P__get_size_plist(&list, "layout").unwrap(), 1);
        assert!(H5P__open_class_path_test(&class, "root/dataset_create"));
        assert!(H5P_remove(&mut list, "layout").is_ok());
    }

    #[test]
    fn property_encoding_roundtrips_scalars() {
        assert_eq!(H5P__decode_bool(&H5P__encode_bool(true)).unwrap(), true);
        assert_eq!(H5P__decode_uint64_t(&H5P__encode_uint64_t(42)).unwrap(), 42);
        assert_eq!(
            H5P__decode_size_t(&H5P__encode_size_t(42).unwrap()).unwrap(),
            42
        );
        assert_eq!(H5P__decode_double(&H5P__encode_double(1.5)).unwrap(), 1.5);
        let mut encoded_chunks = Vec::new();
        encoded_chunks.extend_from_slice(&3u64.to_le_bytes());
        encoded_chunks.extend_from_slice(b"abc");
        encoded_chunks.extend_from_slice(&0u64.to_le_bytes());
        assert_eq!(
            H5P__encode(&[b"abc".to_vec(), Vec::new()]).unwrap(),
            encoded_chunks
        );
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
        assert_eq!(H5P_peek(&list, "driver").unwrap(), b"hdfs".to_vec());
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
        let mut hdfs_bytes = H5P__encode_hdfs_fapl_config(&hdfs).unwrap();
        assert_eq!(H5P__decode_hdfs_fapl_config(&hdfs_bytes).unwrap(), hdfs);
        hdfs_bytes.pop();
        assert!(H5P__decode_hdfs_fapl_config(&hdfs_bytes).is_err());

        let mut hdfs_absent_name = H5P__encode_hdfs_fapl_config(&HdfsFaplConfig {
            namenode_name: String::new(),
            namenode_port: 8020,
            user_name: "hdf5".into(),
            buffer_size: 65536,
        })
        .unwrap();
        hdfs_absent_name[0] = 0;
        assert!(H5P__decode_hdfs_fapl_config(&hdfs_absent_name).is_err());

        let ros3 = Ros3FaplConfig {
            endpoint: Some("s3.us-east-1.amazonaws.com".into()),
            region: Some("us-east-1".into()),
            token: None,
        };
        let mut ros3_bytes = H5P__encode_ros3_fapl_config(&ros3).unwrap();
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
