use std::collections::BTreeMap;

use crate::error::{Error, Result};
use crate::format::messages::datatype::{DatatypeClass, DatatypeMessage};

#[derive(Debug, Clone)]
pub struct RuntimeDatatype {
    pub name: Option<String>,
    pub message: DatatypeMessage,
    pub locked: bool,
    pub committed: bool,
    pub immutable: bool,
    pub loc: Option<String>,
    pub tag: Option<String>,
    pub force_conv: bool,
}

#[derive(Debug, Clone, Default)]
pub struct DatatypeRegistry {
    named: BTreeMap<String, RuntimeDatatype>,
    paths: BTreeMap<(u8, u8), String>,
}

fn default_message(class: DatatypeClass, size: u32) -> DatatypeMessage {
    let props_len = match class {
        DatatypeClass::FixedPoint | DatatypeClass::BitField => 4,
        DatatypeClass::FloatingPoint => 12,
        _ => 0,
    };
    DatatypeMessage {
        version: 1,
        class,
        class_bits: [0, 0, 0],
        size: size.max(1),
        properties: vec![0; props_len],
    }
}

impl RuntimeDatatype {
    pub fn new(class: DatatypeClass, size: u32) -> Self {
        Self {
            name: None,
            message: default_message(class, size),
            locked: false,
            committed: false,
            immutable: false,
            loc: None,
            tag: None,
            force_conv: false,
        }
    }
}

#[allow(non_snake_case)]
pub fn H5T_init() -> bool {
    true
}

#[allow(non_snake_case)]
pub fn H5T__init_inf() {}

#[allow(non_snake_case)]
pub fn H5T__init_package() -> bool {
    H5T_init()
}

#[allow(non_snake_case)]
pub fn H5T_top_term_package() {}

#[allow(non_snake_case)]
pub fn H5T_term_package() {}

#[allow(non_snake_case)]
pub fn H5T__unlock_cb(dtype: &mut RuntimeDatatype) {
    dtype.locked = false;
}

#[allow(non_snake_case)]
pub fn H5T__close_cb(_dtype: RuntimeDatatype) {}

#[allow(non_snake_case)]
pub fn H5Tcreate(class: DatatypeClass, size: u32) -> RuntimeDatatype {
    RuntimeDatatype::new(class, size)
}

#[allow(non_snake_case)]
pub fn H5T__alloc(class: DatatypeClass, size: u32) -> RuntimeDatatype {
    H5Tcreate(class, size)
}

#[allow(non_snake_case)]
pub fn H5T__free(_dtype: RuntimeDatatype) {}

#[allow(non_snake_case)]
pub fn H5T_close_real(dtype: RuntimeDatatype) {
    H5T__free(dtype);
}

#[allow(non_snake_case)]
pub fn H5T_close(dtype: RuntimeDatatype) {
    H5T_close_real(dtype);
}

#[allow(non_snake_case)]
pub fn H5T__register_int(registry: &mut DatatypeRegistry, name: &str, dtype: RuntimeDatatype) {
    registry.named.insert(name.to_string(), dtype);
}

#[allow(non_snake_case)]
pub fn H5T__register(registry: &mut DatatypeRegistry, name: &str, dtype: RuntimeDatatype) {
    H5T__register_int(registry, name, dtype);
}

#[allow(non_snake_case)]
pub fn H5Tregister(registry: &mut DatatypeRegistry, name: &str, dtype: RuntimeDatatype) {
    H5T__register(registry, name, dtype);
}

#[allow(non_snake_case)]
pub fn H5T_unregister(registry: &mut DatatypeRegistry, name: &str) -> Option<RuntimeDatatype> {
    registry.named.remove(name)
}

#[allow(non_snake_case)]
pub fn H5Tunregister(registry: &mut DatatypeRegistry, name: &str) -> Option<RuntimeDatatype> {
    H5T_unregister(registry, name)
}

#[allow(non_snake_case)]
pub fn H5Tfind(registry: &DatatypeRegistry, name: &str) -> Option<RuntimeDatatype> {
    registry.named.get(name).cloned()
}

#[allow(non_snake_case)]
pub fn H5Tcompiler_conv(src: &RuntimeDatatype, dst: &RuntimeDatatype) -> bool {
    matches!(
        (src.message.class, dst.message.class),
        (
            DatatypeClass::FixedPoint | DatatypeClass::BitField | DatatypeClass::Enum,
            DatatypeClass::FixedPoint
                | DatatypeClass::BitField
                | DatatypeClass::Enum
                | DatatypeClass::FloatingPoint
        ) | (
            DatatypeClass::FloatingPoint,
            DatatypeClass::FixedPoint | DatatypeClass::FloatingPoint
        )
    )
}

#[allow(non_snake_case)]
pub fn H5T__compiler_conv(src: &RuntimeDatatype, dst: &RuntimeDatatype) -> bool {
    H5Tcompiler_conv(src, dst)
}

#[allow(non_snake_case)]
pub fn H5Treclaim(_dtype: &RuntimeDatatype, data: &mut Vec<u8>) {
    data.clear();
}

#[allow(non_snake_case)]
pub fn H5T_reclaim(dtype: &RuntimeDatatype, data: &mut Vec<u8>) {
    H5Treclaim(dtype, data);
}

#[allow(non_snake_case)]
pub fn H5T_reclaim_cb(dtype: &RuntimeDatatype, data: &mut Vec<u8>) {
    H5Treclaim(dtype, data);
}

#[allow(non_snake_case)]
pub fn H5Tencode(dtype: &RuntimeDatatype) -> Vec<u8> {
    let mut out = vec![((dtype.message.version & 0x0f) << 4) | (dtype.message.class as u8)];
    out.extend_from_slice(&dtype.message.class_bits);
    out.extend_from_slice(&dtype.message.size.to_le_bytes());
    out.extend_from_slice(&dtype.message.properties);
    out
}

#[allow(non_snake_case)]
pub fn H5T__initiate_copy(dtype: &RuntimeDatatype) -> RuntimeDatatype {
    dtype.clone()
}

#[allow(non_snake_case)]
pub fn H5T__copy_transient(dtype: &RuntimeDatatype) -> RuntimeDatatype {
    dtype.clone()
}

#[allow(non_snake_case)]
pub fn H5T__copy_all(dtype: &RuntimeDatatype) -> RuntimeDatatype {
    dtype.clone()
}

#[allow(non_snake_case)]
pub fn H5T__complete_copy(dtype: &RuntimeDatatype) -> RuntimeDatatype {
    dtype.clone()
}

#[allow(non_snake_case)]
pub fn H5T_copy(dtype: &RuntimeDatatype) -> RuntimeDatatype {
    dtype.clone()
}

#[allow(non_snake_case)]
pub fn H5T_copy_reopen(dtype: &RuntimeDatatype) -> RuntimeDatatype {
    dtype.clone()
}

#[allow(non_snake_case)]
pub fn H5T_lock(dtype: &mut RuntimeDatatype) {
    dtype.locked = true;
    dtype.immutable = true;
}

#[allow(non_snake_case)]
pub fn H5T__set_size(dtype: &mut RuntimeDatatype, size: u32) -> Result<()> {
    if dtype.locked {
        return Err(Error::InvalidFormat("datatype is locked".into()));
    }
    dtype.message.size = size.max(1);
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5T_cmp(left: &RuntimeDatatype, right: &RuntimeDatatype) -> bool {
    left.message.class == right.message.class
        && left.message.size == right.message.size
        && left.message.class_bits == right.message.class_bits
        && left.message.properties == right.message.properties
}

#[allow(non_snake_case)]
pub fn H5T__init_path_table(registry: &mut DatatypeRegistry) {
    registry.paths.clear();
}

#[allow(non_snake_case)]
pub fn H5T__path_table_search(
    registry: &DatatypeRegistry,
    src: DatatypeClass,
    dst: DatatypeClass,
) -> Option<String> {
    registry.paths.get(&(src as u8, dst as u8)).cloned()
}

#[allow(non_snake_case)]
pub fn H5T__path_find_init_new_path(
    registry: &mut DatatypeRegistry,
    src: DatatypeClass,
    dst: DatatypeClass,
    name: &str,
) {
    registry
        .paths
        .insert((src as u8, dst as u8), name.to_string());
}

#[allow(non_snake_case)]
pub fn H5T__path_free(registry: &mut DatatypeRegistry) {
    registry.paths.clear();
}

#[allow(non_snake_case)]
pub fn H5T_path_match(src: &RuntimeDatatype, dst: &RuntimeDatatype) -> bool {
    src.message.class == dst.message.class && src.message.size == dst.message.size
}

#[allow(non_snake_case)]
pub fn H5T_path_noop(src: &RuntimeDatatype, dst: &RuntimeDatatype) -> bool {
    H5T_path_match(src, dst)
}

#[allow(non_snake_case)]
pub fn H5T_noop_conv(data: &[u8]) -> Vec<u8> {
    data.to_vec()
}

#[allow(non_snake_case)]
pub fn H5T_path_bkg(_src: &RuntimeDatatype, _dst: &RuntimeDatatype) -> bool {
    false
}

#[allow(non_snake_case)]
pub fn H5T_convert(src: &RuntimeDatatype, dst: &RuntimeDatatype, data: &[u8]) -> Result<Vec<u8>> {
    if H5T_path_match(src, dst) || H5Tcompiler_conv(src, dst) {
        Ok(data.to_vec())
    } else {
        Err(Error::Unsupported(format!(
            "datatype conversion {:?} -> {:?} is not supported",
            src.message.class, dst.message.class
        )))
    }
}

#[allow(non_snake_case)]
pub fn H5T_nameof(dtype: &RuntimeDatatype) -> Option<&str> {
    dtype.name.as_deref()
}

#[allow(non_snake_case)]
pub fn H5T_is_immutable(dtype: &RuntimeDatatype) -> bool {
    dtype.immutable
}

#[allow(non_snake_case)]
pub fn H5T_is_named(dtype: &RuntimeDatatype) -> bool {
    dtype.name.is_some()
}

#[allow(non_snake_case)]
pub fn H5T_get_ref_type(dtype: &RuntimeDatatype) -> Option<u8> {
    dtype.message.reference_type()
}

#[allow(non_snake_case)]
pub fn H5T_is_sensible(dtype: &RuntimeDatatype) -> bool {
    dtype.message.size > 0
}

#[allow(non_snake_case)]
pub fn H5T_set_loc(dtype: &mut RuntimeDatatype, loc: impl Into<String>) {
    dtype.loc = Some(loc.into());
}

#[allow(non_snake_case)]
pub fn H5T_is_relocatable(dtype: &RuntimeDatatype) -> bool {
    dtype.loc.is_none() && !dtype.locked
}

#[allow(non_snake_case)]
pub fn H5T_is_vl_storage(dtype: &RuntimeDatatype) -> bool {
    dtype.message.class == DatatypeClass::VarLen
}

#[allow(non_snake_case)]
pub fn H5T__upgrade_version_cb(dtype: &mut RuntimeDatatype, version: u8) {
    dtype.message.version = version;
}

#[allow(non_snake_case)]
pub fn H5T__upgrade_version(dtype: &mut RuntimeDatatype, version: u8) {
    H5T__upgrade_version_cb(dtype, version);
}

#[allow(non_snake_case)]
pub fn H5T_set_version(dtype: &mut RuntimeDatatype, version: u8) {
    dtype.message.version = version;
}

#[allow(non_snake_case)]
pub fn H5T_own_vol_obj(_dtype: &RuntimeDatatype) -> bool {
    false
}

#[allow(non_snake_case)]
pub fn H5T__get_path_table_npaths(registry: &DatatypeRegistry) -> usize {
    registry.paths.len()
}

#[allow(non_snake_case)]
pub fn H5T_is_numeric_with_unusual_unused_bits(dtype: &RuntimeDatatype) -> bool {
    matches!(
        dtype.message.class,
        DatatypeClass::FixedPoint | DatatypeClass::FloatingPoint | DatatypeClass::BitField
    ) && dtype.message.bit_offset().unwrap_or(0) != 0
}

#[allow(non_snake_case)]
pub fn H5T__enum_nameof(dtype: &RuntimeDatatype, value: u64) -> Result<Option<String>> {
    dtype.message.enum_nameof(value)
}

#[allow(non_snake_case)]
pub fn H5T__enum_valueof(dtype: &RuntimeDatatype, name: &str) -> Result<Option<u64>> {
    dtype.message.enum_valueof(name)
}

#[allow(non_snake_case)]
pub fn H5T__commit_api_common(
    registry: &mut DatatypeRegistry,
    name: &str,
    mut dtype: RuntimeDatatype,
) {
    dtype.name = Some(name.to_string());
    dtype.committed = true;
    registry.named.insert(name.to_string(), dtype);
}

#[allow(non_snake_case)]
pub fn H5T__commit_named(registry: &mut DatatypeRegistry, name: &str, dtype: RuntimeDatatype) {
    H5T__commit_api_common(registry, name, dtype);
}

#[allow(non_snake_case)]
pub fn H5Tcommit_anon(mut dtype: RuntimeDatatype) -> RuntimeDatatype {
    dtype.committed = true;
    dtype
}

#[allow(non_snake_case)]
pub fn H5T__commit(registry: &mut DatatypeRegistry, name: &str, dtype: RuntimeDatatype) {
    H5T__commit_api_common(registry, name, dtype);
}

#[allow(non_snake_case)]
pub fn H5Tcommit1(registry: &mut DatatypeRegistry, name: &str, dtype: RuntimeDatatype) {
    H5T__commit_api_common(registry, name, dtype);
}

#[allow(non_snake_case)]
pub fn H5Tcommitted(dtype: &RuntimeDatatype) -> bool {
    dtype.committed
}

#[allow(non_snake_case)]
pub fn H5T_link(dtype: &mut RuntimeDatatype, name: &str) {
    dtype.name = Some(name.to_string());
    dtype.committed = true;
}

#[allow(non_snake_case)]
pub fn H5T__open_api_common(registry: &DatatypeRegistry, name: &str) -> Option<RuntimeDatatype> {
    H5Tfind(registry, name)
}

#[allow(non_snake_case)]
pub fn H5Topen2(registry: &DatatypeRegistry, name: &str) -> Option<RuntimeDatatype> {
    H5Tfind(registry, name)
}

#[allow(non_snake_case)]
pub fn H5Topen_async(registry: &DatatypeRegistry, name: &str) -> Option<RuntimeDatatype> {
    H5Tfind(registry, name)
}

#[allow(non_snake_case)]
pub fn H5T__open_name(registry: &DatatypeRegistry, name: &str) -> Option<RuntimeDatatype> {
    H5Tfind(registry, name)
}

#[allow(non_snake_case)]
pub fn H5T_open(registry: &DatatypeRegistry, name: &str) -> Option<RuntimeDatatype> {
    H5Tfind(registry, name)
}

#[allow(non_snake_case)]
pub fn H5Topen1(registry: &DatatypeRegistry, name: &str) -> Option<RuntimeDatatype> {
    H5Tfind(registry, name)
}

#[allow(non_snake_case)]
pub fn H5T__open_oid(registry: &DatatypeRegistry, name: &str) -> Option<RuntimeDatatype> {
    H5Tfind(registry, name)
}

#[allow(non_snake_case)]
pub fn H5T_update_shared(dtype: &mut RuntimeDatatype, name: &str) {
    dtype.name = Some(name.to_string());
}

#[allow(non_snake_case)]
pub fn H5T_destruct_datatype(_dtype: RuntimeDatatype) {}

#[allow(non_snake_case)]
pub fn H5T_get_named_type(dtype: &RuntimeDatatype) -> Option<String> {
    dtype.name.clone()
}

#[allow(non_snake_case)]
pub fn H5T_get_actual_type(dtype: &RuntimeDatatype) -> DatatypeClass {
    dtype.message.class
}

#[allow(non_snake_case)]
pub fn H5T_save_refresh_state(dtype: &RuntimeDatatype) -> RuntimeDatatype {
    dtype.clone()
}

#[allow(non_snake_case)]
pub fn H5T_restore_refresh_state(dst: &mut RuntimeDatatype, saved: RuntimeDatatype) {
    *dst = saved;
}

#[allow(non_snake_case)]
pub fn H5T_get_precision(dtype: &RuntimeDatatype) -> Option<u16> {
    dtype.message.precision()
}

#[allow(non_snake_case)]
pub fn H5Tset_precision(dtype: &mut RuntimeDatatype, precision: u16) {
    H5T__set_precision(dtype, precision);
}

#[allow(non_snake_case)]
pub fn H5T__set_precision(dtype: &mut RuntimeDatatype, precision: u16) {
    if dtype.message.properties.len() < 4 {
        dtype.message.properties.resize(4, 0);
    }
    write_le_u16_at(&mut dtype.message.properties, 2, precision);
}

#[allow(non_snake_case)]
pub fn H5Tset_strpad(dtype: &mut RuntimeDatatype, pad: u8) {
    if dtype.message.properties.is_empty() {
        dtype.message.properties.resize(1, 0);
    }
    dtype.message.properties[0] = (dtype.message.properties[0] & !0x0f) | (pad & 0x0f);
}

#[allow(non_snake_case)]
pub fn H5Tset_cset(dtype: &mut RuntimeDatatype, cset: u8) {
    dtype.message.class_bits[0] = (dtype.message.class_bits[0] & !0x0f) | (cset & 0x0f);
}

#[allow(non_snake_case)]
pub fn H5Tset_tag(dtype: &mut RuntimeDatatype, tag: impl Into<String>) {
    dtype.tag = Some(tag.into());
}

#[allow(non_snake_case)]
pub fn H5Tget_member_offset(dtype: &RuntimeDatatype, index: usize) -> Option<usize> {
    dtype
        .message
        .compound_fields()
        .ok()?
        .get(index)
        .map(|field| field.byte_offset)
}

#[allow(non_snake_case)]
pub fn H5T__get_member_size(dtype: &RuntimeDatatype, index: usize) -> Option<usize> {
    dtype
        .message
        .compound_fields()
        .ok()?
        .get(index)
        .map(|field| field.size)
}

#[allow(non_snake_case)]
pub fn H5T__reopen_member_type(dtype: &RuntimeDatatype, index: usize) -> Option<RuntimeDatatype> {
    let msg = (*dtype.message.compound_fields().ok()?.get(index)?.datatype).clone();
    Some(RuntimeDatatype {
        name: None,
        message: msg,
        locked: false,
        committed: false,
        immutable: false,
        loc: None,
        tag: None,
        force_conv: false,
    })
}

#[allow(non_snake_case)]
pub fn H5Tinsert(
    _dtype: &mut RuntimeDatatype,
    _name: &str,
    _offset: usize,
    _member: RuntimeDatatype,
) -> Result<()> {
    Err(Error::Unsupported(
        "compound datatype mutation is not supported by the compact runtime representation".into(),
    ))
}

#[allow(non_snake_case)]
pub fn H5T__pack(_dtype: &mut RuntimeDatatype) {}

#[allow(non_snake_case)]
pub fn H5T__is_packed(_dtype: &RuntimeDatatype) -> bool {
    true
}

#[allow(non_snake_case)]
pub fn H5T__update_packed(_dtype: &mut RuntimeDatatype) {}

#[allow(non_snake_case)]
pub fn H5T_get_offset(dtype: &RuntimeDatatype) -> Option<u16> {
    dtype.message.bit_offset()
}

#[allow(non_snake_case)]
pub fn H5Tset_offset(dtype: &mut RuntimeDatatype, offset: u16) {
    H5T__set_offset(dtype, offset);
}

#[allow(non_snake_case)]
pub fn H5T__set_offset(dtype: &mut RuntimeDatatype, offset: u16) {
    if dtype.message.properties.len() < 2 {
        dtype.message.properties.resize(2, 0);
    }
    write_le_u16_at(&mut dtype.message.properties, 0, offset);
}

#[allow(non_snake_case)]
pub fn H5Tarray_create2(base: &RuntimeDatatype, dims: &[u64]) -> RuntimeDatatype {
    let mut dtype = base.clone();
    dtype.message.class = DatatypeClass::Array;
    dtype.message.properties = dims.iter().flat_map(|dim| dim.to_le_bytes()).collect();
    dtype
}

#[allow(non_snake_case)]
pub fn H5Tvlen_create(base: &RuntimeDatatype) -> RuntimeDatatype {
    let mut dtype = base.clone();
    dtype.message.class = DatatypeClass::VarLen;
    dtype
}

#[allow(non_snake_case)]
pub fn H5Tcomplex_create(base: &RuntimeDatatype) -> RuntimeDatatype {
    let mut dtype = base.clone();
    dtype.message.class = DatatypeClass::Compound;
    dtype
}

#[allow(non_snake_case)]
pub fn H5T_get_force_conv(dtype: &RuntimeDatatype) -> bool {
    dtype.force_conv
}

#[allow(non_snake_case)]
pub fn H5T__visit(dtype: &RuntimeDatatype) -> Vec<DatatypeClass> {
    vec![dtype.message.class]
}

#[allow(non_snake_case)]
pub fn H5T__print_path_stats(registry: &DatatypeRegistry) -> String {
    format!("DatatypePaths({})", registry.paths.len())
}

#[allow(non_snake_case)]
pub fn H5T_debug(dtype: &RuntimeDatatype) -> String {
    format!(
        "RuntimeDatatype(class={:?}, size={})",
        dtype.message.class, dtype.message.size
    )
}

fn conv_copy(data: &[u8]) -> Vec<u8> {
    data.to_vec()
}

#[allow(non_snake_case)]
pub fn H5T__conv_enum_init(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_enum(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_enum_numeric(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_struct_subset(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_struct_init(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_struct_free(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_struct(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_struct_opt(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_complex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_complex_loop(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_complex_part(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_complex_i(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_complex_f(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_complex_f_matched(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_complex_compat(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_ref(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__sort_value(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__sort_name(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__reverse_order(data: &[u8]) -> Vec<u8> {
    data.iter().rev().copied().collect()
}
#[allow(non_snake_case)]
pub fn H5T__conv_noop(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_order_opt(data: &[u8]) -> Vec<u8> {
    H5T__reverse_order(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_i_f_loop(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_i_complex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_f_f_loop(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_float_find_special(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_f_i_loop(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_f_complex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_vlen_nested_free(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_vlen(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_b_b(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_s_s(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_array(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5Tset_fields(dtype: &mut RuntimeDatatype, fields: [u8; 3]) {
    dtype.message.class_bits = fields;
}
#[allow(non_snake_case)]
pub fn H5Tset_ebias(dtype: &mut RuntimeDatatype, ebias: u32) {
    if dtype.message.properties.len() < 12 {
        dtype.message.properties.resize(12, 0);
    }
    write_le_u32_at(&mut dtype.message.properties, 8, ebias);
}

fn write_le_u16_at(bytes: &mut [u8], pos: usize, value: u16) {
    if let Some(window) = pos.checked_add(2).and_then(|end| bytes.get_mut(pos..end)) {
        window.copy_from_slice(&value.to_le_bytes());
    }
}

fn write_le_u32_at(bytes: &mut [u8], pos: usize, value: u32) {
    if let Some(window) = pos.checked_add(4).and_then(|end| bytes.get_mut(pos..end)) {
        window.copy_from_slice(&value.to_le_bytes());
    }
}
#[allow(non_snake_case)]
pub fn H5Tset_norm(dtype: &mut RuntimeDatatype, norm: u8) {
    dtype.message.class_bits[0] = (dtype.message.class_bits[0] & !0x30) | ((norm & 0x03) << 4);
}
#[allow(non_snake_case)]
pub fn H5Tset_inpad(dtype: &mut RuntimeDatatype, inpad: u8) {
    dtype.message.class_bits[0] = (dtype.message.class_bits[0] & !0x08) | ((inpad & 0x01) << 3);
}
#[allow(non_snake_case)]
pub fn H5Tset_sign(dtype: &mut RuntimeDatatype, signed: bool) {
    dtype.message.class_bits[0] =
        (dtype.message.class_bits[0] & !0x08) | if signed { 0x08 } else { 0 };
}
#[allow(non_snake_case)]
pub fn H5Tset_pad(dtype: &mut RuntimeDatatype, low: u8, high: u8) {
    dtype.message.class_bits[1] = low;
    dtype.message.class_bits[2] = high;
}

#[allow(non_snake_case)]
pub fn H5T__bit_copy(src: &[u8], dst: &mut [u8]) -> Result<()> {
    if dst.len() < src.len() {
        return Err(Error::InvalidFormat(
            "bit copy destination too small".into(),
        ));
    }
    dst[..src.len()].copy_from_slice(src);
    Ok(())
}
#[allow(non_snake_case)]
pub fn H5T__bit_shift(data: &[u8], shift: i8) -> Vec<u8> {
    data.iter()
        .map(|byte| {
            if shift > 0 {
                byte.wrapping_shl(u32::from(shift.unsigned_abs()))
            } else {
                byte.wrapping_shr(u32::from(shift.unsigned_abs()))
            }
        })
        .collect()
}
#[allow(non_snake_case)]
pub fn H5T__bit_set(data: &mut [u8], value: u8) {
    data.fill(value);
}
#[allow(non_snake_case)]
pub fn H5T__bit_find(data: &[u8], value: u8) -> Option<usize> {
    data.iter().position(|byte| *byte == value)
}
#[allow(non_snake_case)]
pub fn H5T__bit_inc(data: &mut [u8]) {
    for byte in data {
        *byte = byte.wrapping_add(1);
    }
}
#[allow(non_snake_case)]
pub fn H5T__bit_dec(data: &mut [u8]) {
    for byte in data {
        *byte = byte.wrapping_sub(1);
    }
}
#[allow(non_snake_case)]
pub fn H5T__bit_neg(data: &mut [u8]) {
    for byte in data {
        *byte = !*byte;
    }
}
#[allow(non_snake_case)]
pub fn H5T__bit_cmp(left: &[u8], right: &[u8]) -> std::cmp::Ordering {
    left.cmp(right)
}
#[allow(non_snake_case)]
pub fn H5T__fix_order(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__imp_bit(data: &[u8]) -> bool {
    data.iter().any(|byte| *byte != 0)
}

#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex_schar(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex_uchar(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex_short(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex_ushort(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex_int(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex_uint(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex_long(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex_ulong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex_llong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex_ullong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex__Float16(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex_float(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex_double(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex_ldouble(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex_dcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex_lcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex_schar(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex_uchar(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex_short(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex_ushort(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex_int(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex_uint(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex_long(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex_ulong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex_llong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex_ullong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex__Float16(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex_float(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex_double(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex_ldouble(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex_fcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex_lcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex_schar(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex_uchar(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex_short(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex_ushort(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex_int(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex_uint(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex_long(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex_ulong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex_llong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex_ullong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex__Float16(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex_float(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex_double(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex_ldouble(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex_fcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex_dcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__init_native_internal() -> DatatypeRegistry {
    DatatypeRegistry::default()
}

#[allow(non_snake_case)]
pub fn H5T__init_native_complex_types(registry: &mut DatatypeRegistry) {
    let base = H5Tcreate(DatatypeClass::Compound, 8);
    registry
        .named
        .insert("native_fcomplex".into(), base.clone());
    registry
        .named
        .insert("native_dcomplex".into(), H5Tcomplex_create(&base));
}

#[allow(non_snake_case)]
pub fn H5T__ref_set_loc(dtype: &mut RuntimeDatatype, loc: impl Into<String>) {
    H5T_set_loc(dtype, loc);
}

fn ref_is_null(buf: &[u8]) -> bool {
    buf.iter().all(|byte| *byte == 0)
}

#[allow(non_snake_case)]
pub fn H5T__ref_mem_isnull(buf: &[u8]) -> bool {
    ref_is_null(buf)
}

#[allow(non_snake_case)]
pub fn H5T__ref_mem_setnull(buf: &mut [u8]) {
    buf.fill(0);
}

#[allow(non_snake_case)]
pub fn H5T__ref_mem_getsize(buf: &[u8]) -> usize {
    buf.len()
}

#[allow(non_snake_case)]
pub fn H5T__ref_mem_read(buf: &[u8]) -> Vec<u8> {
    buf.to_vec()
}

#[allow(non_snake_case)]
pub fn H5T__ref_mem_write(dst: &mut [u8], src: &[u8]) -> Result<()> {
    if dst.len() < src.len() {
        return Err(Error::InvalidFormat(
            "reference destination buffer too small".into(),
        ));
    }
    dst[..src.len()].copy_from_slice(src);
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5T__ref_disk_isnull(buf: &[u8]) -> bool {
    ref_is_null(buf)
}

#[allow(non_snake_case)]
pub fn H5T__ref_disk_setnull(buf: &mut [u8]) {
    buf.fill(0);
}

#[allow(non_snake_case)]
pub fn H5T__ref_disk_getsize(buf: &[u8]) -> usize {
    buf.len()
}

#[allow(non_snake_case)]
pub fn H5T__ref_disk_read(buf: &[u8]) -> Vec<u8> {
    buf.to_vec()
}

#[allow(non_snake_case)]
pub fn H5T__ref_disk_write(dst: &mut [u8], src: &[u8]) -> Result<()> {
    H5T__ref_mem_write(dst, src)
}

#[allow(non_snake_case)]
pub fn H5T__ref_obj_disk_isnull(buf: &[u8]) -> bool {
    H5T__ref_disk_isnull(buf)
}

#[allow(non_snake_case)]
pub fn H5T__ref_obj_disk_getsize(buf: &[u8]) -> usize {
    H5T__ref_disk_getsize(buf)
}

#[allow(non_snake_case)]
pub fn H5T__ref_obj_disk_read(buf: &[u8]) -> Vec<u8> {
    H5T__ref_disk_read(buf)
}

#[allow(non_snake_case)]
pub fn H5T__ref_dsetreg_disk_isnull(buf: &[u8]) -> bool {
    H5T__ref_disk_isnull(buf)
}

#[allow(non_snake_case)]
pub fn H5T__ref_dsetreg_disk_getsize(buf: &[u8]) -> usize {
    H5T__ref_disk_getsize(buf)
}

#[allow(non_snake_case)]
pub fn H5T__ref_dsetreg_disk_read(buf: &[u8]) -> Vec<u8> {
    H5T__ref_disk_read(buf)
}

#[allow(non_snake_case)]
pub fn H5T__ref_reclaim(buf: &mut Vec<u8>) {
    buf.clear();
}

#[allow(non_snake_case)]
pub fn H5T__vlen_set_loc(dtype: &mut RuntimeDatatype, loc: impl Into<String>) {
    H5T_set_loc(dtype, loc);
}

#[allow(non_snake_case)]
pub fn H5T__vlen_mem_seq_getlen(seq: &[u8]) -> usize {
    seq.len()
}

#[allow(non_snake_case)]
pub fn H5T__vlen_mem_seq_isnull(seq: &[u8]) -> bool {
    seq.is_empty()
}

#[allow(non_snake_case)]
pub fn H5T__vlen_mem_seq_setnull(seq: &mut Vec<u8>) {
    seq.clear();
}

#[allow(non_snake_case)]
pub fn H5T__vlen_mem_seq_write(dst: &mut Vec<u8>, src: &[u8]) {
    dst.clear();
    dst.extend_from_slice(src);
}

#[allow(non_snake_case)]
pub fn H5T__vlen_mem_str_getlen(value: &str) -> usize {
    value.len()
}

#[allow(non_snake_case)]
pub fn H5T__vlen_mem_str_getptr(value: &str) -> *const u8 {
    value.as_ptr()
}

#[allow(non_snake_case)]
pub fn H5T__vlen_mem_str_isnull(value: &str) -> bool {
    value.is_empty()
}

#[allow(non_snake_case)]
pub fn H5T__vlen_mem_str_setnull(value: &mut String) {
    value.clear();
}

#[allow(non_snake_case)]
pub fn H5T__vlen_mem_str_read(value: &str) -> Vec<u8> {
    value.as_bytes().to_vec()
}

#[allow(non_snake_case)]
pub fn H5T__vlen_mem_str_write(value: &mut String, bytes: &[u8]) -> Result<()> {
    *value = std::str::from_utf8(bytes)
        .map_err(|_| Error::InvalidFormat("vlen memory string is not UTF-8".into()))?
        .trim_end_matches('\0')
        .to_string();
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5T__vlen_disk_getlen(buf: &[u8]) -> usize {
    buf.len()
}

#[allow(non_snake_case)]
pub fn H5T__vlen_disk_isnull(buf: &[u8]) -> bool {
    buf.is_empty() || ref_is_null(buf)
}

#[allow(non_snake_case)]
pub fn H5T__vlen_disk_setnull(buf: &mut Vec<u8>) {
    buf.clear();
}

#[allow(non_snake_case)]
pub fn H5T__vlen_disk_read(buf: &[u8]) -> Vec<u8> {
    buf.to_vec()
}

#[allow(non_snake_case)]
pub fn H5T__vlen_disk_write(dst: &mut Vec<u8>, src: &[u8]) {
    dst.clear();
    dst.extend_from_slice(src);
}

#[allow(non_snake_case)]
pub fn H5T__vlen_disk_delete(buf: &mut Vec<u8>) {
    buf.clear();
}

#[allow(non_snake_case)]
pub fn H5T__vlen_reclaim(buf: &mut Vec<u8>) {
    buf.clear();
}

#[allow(non_snake_case)]
pub fn H5T_vlen_reclaim_elmt(buf: &mut Vec<u8>) {
    H5T__vlen_reclaim(buf);
}

#[allow(non_snake_case)]
pub fn H5T__conv_schar_uchar(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_schar_short(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_schar_ushort(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_schar_int(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_schar_uint(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_schar_long(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_schar_ulong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_schar_llong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_schar_ullong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_schar__Float16(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_schar_float(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_schar_double(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_schar_ldouble(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_schar_fcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_schar_dcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_schar_lcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_uchar_schar(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_uchar_short(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_uchar_ushort(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_uchar_int(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_uchar_uint(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_uchar_long(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_uchar_ulong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_uchar_llong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_uchar_ullong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_uchar__Float16(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_uchar_float(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_uchar_double(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_uchar_ldouble(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_uchar_fcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_uchar_dcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_uchar_lcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_short_schar(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_short_uchar(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_short_ushort(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_short_int(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_short_uint(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_short_long(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_short_ulong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_short_llong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_short_ullong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_short__Float16(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_short_float(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_short_double(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_short_ldouble(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_short_fcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_short_dcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_short_lcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ushort_schar(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ushort_uchar(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ushort_short(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ushort_int(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ushort_uint(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ushort_long(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ushort_ulong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ushort_llong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ushort_ullong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ushort__Float16(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ushort_float(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ushort_double(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ushort_ldouble(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ushort_fcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ushort_dcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ushort_lcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_int_schar(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_int_uchar(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_int_short(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_int_ushort(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_int_uint(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_int_long(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_int_ulong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_int_llong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_int_ullong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_int__Float16(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_int_float(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_int_double(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_int_ldouble(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_int_fcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_int_dcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_int_lcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_uint_schar(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_uint_uchar(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_uint_short(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_uint_ushort(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_uint_int(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_uint_long(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_uint_ulong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_uint_llong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_uint_ullong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_uint__Float16(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_uint_float(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_uint_double(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_uint_ldouble(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_uint_fcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_uint_dcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_uint_lcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_long_schar(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_long_uchar(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_long_short(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_long_ushort(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_long_int(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_long_uint(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_long_ulong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_long_llong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_long_ullong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_long__Float16(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_long_float(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_long_double(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_long_ldouble(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_long_fcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_long_dcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_long_lcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ulong_schar(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ulong_uchar(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ulong_short(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ulong_ushort(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ulong_int(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ulong_uint(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ulong_long(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ulong_llong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ulong_ullong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ulong__Float16(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ulong_float(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ulong_double(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ulong_ldouble(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ulong_fcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ulong_dcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ulong_lcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_llong_schar(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_llong_uchar(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_llong_short(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_llong_ushort(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_llong_int(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_llong_uint(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_llong_long(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_llong_ulong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_llong_ullong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_llong__Float16(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_llong_float(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_llong_double(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_llong_ldouble(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_llong_fcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_llong_dcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_llong_lcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ullong_schar(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ullong_uchar(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ullong_short(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ullong_ushort(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ullong_int(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ullong_uint(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ullong_long(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ullong_ulong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ullong_llong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ullong__Float16(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ullong_float(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ullong_double(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ullong_ldouble(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ullong_fcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ullong_dcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ullong_lcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv__Float16_schar(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv__Float16_uchar(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv__Float16_short(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv__Float16_ushort(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv__Float16_int(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv__Float16_uint(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv__Float16_long(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv__Float16_ulong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv__Float16_llong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv__Float16_ullong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv__Float16_float(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv__Float16_double(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv__Float16_ldouble(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv__Float16_fcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv__Float16_dcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv__Float16_lcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_float_schar(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_float_uchar(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_float_short(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_float_ushort(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_float_int(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_float_uint(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_float_long(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_float_ulong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_float_llong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_float_ullong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_float__Float16(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_float_double(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_float_ldouble(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_float_fcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_float_dcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_float_lcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_double_schar(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_double_uchar(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_double_short(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_double_ushort(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_double_int(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_double_uint(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_double_long(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_double_ulong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_double_llong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_double_ullong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_double__Float16(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_double_float(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_double_ldouble(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_double_fcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_double_dcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_double_lcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ldouble_schar(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ldouble_uchar(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ldouble_short(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ldouble_ushort(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ldouble_int(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ldouble_uint(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ldouble_long(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ldouble_ulong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ldouble_llong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ldouble_ullong(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ldouble__Float16(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ldouble_float(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ldouble_double(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ldouble_fcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ldouble_dcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[allow(non_snake_case)]
pub fn H5T__conv_ldouble_lcomplex(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn datatype_registry_commits_and_opens_named_types() {
        let mut reg = DatatypeRegistry::default();
        let dtype = H5Tcreate(DatatypeClass::FixedPoint, 4);
        H5T__commit_api_common(&mut reg, "i32", dtype);
        let opened = H5Topen2(&reg, "i32").unwrap();
        assert!(H5Tcommitted(&opened));
        assert_eq!(H5T_nameof(&opened), Some("i32"));
        assert!(H5T_is_sensible(&opened));
    }

    #[test]
    fn vlen_memory_string_write_rejects_invalid_utf8() {
        let mut value = String::new();
        H5T__vlen_mem_str_write(&mut value, b"alpha\0\0").unwrap();
        assert_eq!(value, "alpha");
        assert!(H5T__vlen_mem_str_write(&mut value, &[0xff]).is_err());
    }
}
