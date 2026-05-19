use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fmt;

use crate::error::{Error, Result};
use crate::format::messages::datatype::{
    ArrayDims, ByteOrder, CompoundField, CompoundFieldView, CompoundFields, DatatypeClass,
    DatatypeMessage, EnumMembers, FloatFields,
};
use crate::hl::plist::datatype_create::DatatypeCreate;

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
    pub owned_vol_obj: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct DatatypeRegistry {
    named: BTreeMap<String, RuntimeDatatype>,
    paths: BTreeMap<(u8, u8), String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum H5TBitSearchDirection {
    Lsb,
    Msb,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum H5TDetectedOrder {
    LittleEndian,
    BigEndian,
    Vax,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum H5TConvFloatSpecval {
    Regular,
    PosZero,
    NegZero,
    PosInf,
    NegInf,
    Nan,
}

/// Internal helper `default_message`.
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

fn compound_field_from_view(field: CompoundFieldView<'_>) -> CompoundField {
    CompoundField {
        name: field.name.into_owned(),
        byte_offset: field.byte_offset,
        size: field.size,
        class: field.class,
        byte_order: field.byte_order,
        datatype: Box::new(field.datatype),
    }
}

fn collect_compound_fields(message: &DatatypeMessage) -> Result<Vec<CompoundField>> {
    message
        .compound_fields_iter()?
        .map(|field| field.map(compound_field_from_view))
        .collect()
}

fn enum_has_members(message: &DatatypeMessage) -> Result<bool> {
    Ok(message.enum_members_iter()?.next().transpose()?.is_some())
}

impl RuntimeDatatype {
    /// Construct a runtime datatype of the given class and size.
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
            owned_vol_obj: None,
        }
    }
}

/// Initialize the datatype subsystem.
#[allow(non_snake_case)]
pub fn H5T_init() -> bool {
    true
}

/// Initialize the datatype subsystem.
#[allow(non_snake_case)]
pub fn H5T__init_inf() {}

/// Initialize the datatype package.
#[allow(non_snake_case)]
pub fn H5T__init_package() -> bool {
    H5T_init()
}

/// Terminate the datatype package and release its resources.
#[allow(non_snake_case)]
pub fn H5T_top_term_package() {}

/// Terminate the datatype package and release its resources.
#[allow(non_snake_case)]
pub fn H5T_term_package() {}

/// Unlock a datatype.
#[allow(non_snake_case)]
pub fn H5T__unlock_cb(dtype: &mut RuntimeDatatype) {
    dtype.locked = false;
}

/// Close callback for datatype objects.
#[allow(non_snake_case)]
pub fn H5T__close_cb(_dtype: RuntimeDatatype) {}

/// Create a new datatype.
#[allow(non_snake_case)]
pub fn H5Tcreate(class: DatatypeClass, size: u32) -> RuntimeDatatype {
    RuntimeDatatype::new(class, size)
}

/// Allocate storage for a datatype.
#[allow(non_snake_case)]
pub fn H5T__alloc(class: DatatypeClass, size: u32) -> RuntimeDatatype {
    H5Tcreate(class, size)
}

/// Free a datatype's in-memory resources.
#[allow(non_snake_case)]
pub fn H5T__free(_dtype: RuntimeDatatype) {}

/// Close a datatype.
#[allow(non_snake_case)]
pub fn H5T_close_real(dtype: RuntimeDatatype) {
    H5T__free(dtype);
}

/// Close a datatype.
#[allow(non_snake_case)]
pub fn H5T_close(dtype: RuntimeDatatype) {
    H5T_close_real(dtype);
}

/// Register a datatype.
#[allow(non_snake_case)]
pub fn H5T__register_int(registry: &mut DatatypeRegistry, name: &str, dtype: RuntimeDatatype) {
    registry.named.insert(name.to_string(), dtype);
}

/// Register a datatype.
#[allow(non_snake_case)]
pub fn H5T__register(registry: &mut DatatypeRegistry, name: &str, dtype: RuntimeDatatype) {
    H5T__register_int(registry, name, dtype);
}

/// Register a datatype.
#[allow(non_snake_case)]
pub fn H5Tregister(registry: &mut DatatypeRegistry, name: &str, dtype: RuntimeDatatype) {
    H5T__register(registry, name, dtype);
}

/// Unregister a datatype.
#[allow(non_snake_case)]
pub fn H5T_unregister(registry: &mut DatatypeRegistry, name: &str) -> Option<RuntimeDatatype> {
    registry.named.remove(name)
}

/// Unregister a datatype.
#[allow(non_snake_case)]
pub fn H5Tunregister(registry: &mut DatatypeRegistry, name: &str) -> Option<RuntimeDatatype> {
    H5T_unregister(registry, name)
}

/// Find an entry in a datatype.
#[allow(non_snake_case)]
pub fn H5Tfind_ref<'a>(registry: &'a DatatypeRegistry, name: &str) -> Option<&'a RuntimeDatatype> {
    registry.named.get(name)
}

/// Find an entry in a datatype.
#[deprecated(note = "use H5Tfind_ref() to borrow the registered datatype")]
#[allow(non_snake_case)]
pub fn H5Tfind(registry: &DatatypeRegistry, name: &str) -> Option<RuntimeDatatype> {
    H5Tfind_ref(registry, name).cloned()
}

/// Datatype operation: compiler conv.
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

/// Datatype operation: compiler conv.
#[allow(non_snake_case)]
pub fn H5T__compiler_conv(src: &RuntimeDatatype, dst: &RuntimeDatatype) -> bool {
    H5Tcompiler_conv(src, dst)
}

/// Datatype operation: reclaim.
#[allow(non_snake_case)]
pub fn H5Treclaim(_dtype: &RuntimeDatatype, data: &mut Vec<u8>) {
    data.clear();
}

/// Datatype operation: reclaim.
#[allow(non_snake_case)]
pub fn H5T_reclaim(dtype: &RuntimeDatatype, data: &mut Vec<u8>) {
    H5Treclaim(dtype, data);
}

/// Datatype operation: reclaim cb.
#[allow(non_snake_case)]
pub fn H5T_reclaim_cb(dtype: &RuntimeDatatype, data: &mut Vec<u8>) {
    H5Treclaim(dtype, data);
}

/// Encode a datatype to its on-disk representation.
#[allow(non_snake_case)]
pub fn H5Tencode_into(dtype: &RuntimeDatatype, out: &mut Vec<u8>) -> Result<()> {
    out.clear();
    out.push(((dtype.message.version & 0x0f) << 4) | (dtype.message.class as u8));
    out.extend_from_slice(&dtype.message.class_bits);
    out.extend_from_slice(&dtype.message.size.to_le_bytes());
    out.extend_from_slice(&dtype.message.properties);
    DatatypeMessage::decode(&out)?;
    Ok(())
}

/// Encode a datatype to its on-disk representation.
#[deprecated(note = "use H5Tencode_into() to reuse caller-provided output storage")]
#[allow(non_snake_case)]
pub fn H5Tencode(dtype: &RuntimeDatatype) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(8 + dtype.message.properties.len());
    H5Tencode_into(dtype, &mut out)?;
    Ok(out)
}

/// Decode a datatype from its on-disk representation.
#[allow(non_snake_case)]
pub fn H5Tdecode(bytes: &[u8]) -> Result<RuntimeDatatype> {
    Ok(RuntimeDatatype {
        name: None,
        message: DatatypeMessage::decode(bytes)?,
        locked: false,
        committed: false,
        immutable: false,
        loc: None,
        tag: None,
        force_conv: false,
        owned_vol_obj: None,
    })
}

/// Return a deep copy of a datatype.
#[allow(non_snake_case)]
pub fn H5T__initiate_copy(dtype: &RuntimeDatatype) -> RuntimeDatatype {
    dtype.clone()
}

/// Return a deep copy of a datatype.
#[allow(non_snake_case)]
pub fn H5T__copy_transient(dtype: &RuntimeDatatype) -> RuntimeDatatype {
    dtype.clone()
}

/// Return a deep copy of a datatype.
#[allow(non_snake_case)]
pub fn H5T__copy_all(dtype: &RuntimeDatatype) -> RuntimeDatatype {
    dtype.clone()
}

/// Return a deep copy of a datatype.
#[allow(non_snake_case)]
pub fn H5T__complete_copy(dtype: &RuntimeDatatype) -> RuntimeDatatype {
    dtype.clone()
}

/// Return a deep copy of a datatype.
#[allow(non_snake_case)]
pub fn H5T_copy(dtype: &RuntimeDatatype) -> RuntimeDatatype {
    dtype.clone()
}

/// Return a deep copy of a datatype.
#[allow(non_snake_case)]
pub fn H5T_copy_reopen(dtype: &RuntimeDatatype) -> RuntimeDatatype {
    dtype.clone()
}

/// Lock a datatype against further modification.
#[allow(non_snake_case)]
pub fn H5T_lock(dtype: &mut RuntimeDatatype) {
    dtype.locked = true;
    dtype.immutable = true;
}

/// Set the size of a datatype.
#[allow(non_snake_case)]
pub fn H5Tset_size(dtype: &mut RuntimeDatatype, size: u32) -> Result<()> {
    H5T__set_size(dtype, size)
}

/// Set the size of a datatype.
#[allow(non_snake_case)]
pub fn H5T__set_size(dtype: &mut RuntimeDatatype, size: u32) -> Result<()> {
    if dtype.locked || dtype.immutable {
        return Err(Error::InvalidFormat(
            "datatype size cannot be changed on a locked datatype".into(),
        ));
    }
    if size == 0 {
        return Err(Error::InvalidFormat(
            "datatype size must be positive".into(),
        ));
    }

    if dtype.message.class == DatatypeClass::String && size == u32::MAX {
        dtype.message.class = DatatypeClass::VarLen;
        dtype.message.class_bits[0] = (dtype.message.class_bits[0] & !0x0f) | 1;
        dtype.message.size = size;
        dtype.force_conv = true;
        return Ok(());
    }
    if size == u32::MAX {
        return Err(Error::InvalidFormat(
            "only string datatypes may be variable length".into(),
        ));
    }

    match dtype.message.class {
        DatatypeClass::FixedPoint
        | DatatypeClass::Time
        | DatatypeClass::BitField
        | DatatypeClass::Opaque => {}
        DatatypeClass::FloatingPoint => {
            if dtype.message.properties.len() < 8 {
                dtype.message.properties.resize(8, 0);
            }
        }
        DatatypeClass::Compound => {
            if size < dtype.message.size {
                for field in dtype.message.compound_fields_iter()? {
                    let field = field?;
                    let member_end = u32::try_from(field.byte_offset)
                        .map_err(|_| {
                            Error::InvalidFormat("compound member offset exceeds u32".into())
                        })?
                        .checked_add(field.datatype.size)
                        .ok_or_else(|| {
                            Error::InvalidFormat("compound member end offset overflows".into())
                        })?;
                    if size < member_end {
                        return Err(Error::InvalidFormat(
                            "datatype size shrink would cut off a compound member".into(),
                        ));
                    }
                }
            }
        }
        DatatypeClass::String => {}
        DatatypeClass::Enum => {
            if enum_has_members(&dtype.message)? {
                return Err(Error::InvalidFormat(
                    "enum datatype size cannot change after members are defined".into(),
                ));
            }
            return Err(Error::Unsupported(
                "setting size on encoded enum base datatypes is unsupported".into(),
            ));
        }
        DatatypeClass::VarLen if dtype.message.is_variable_string() => {
            dtype.message.size = size;
            return Ok(());
        }
        DatatypeClass::VarLen | DatatypeClass::Array | DatatypeClass::Reference => {
            return Err(Error::Unsupported(
                "datatype size is not defined for this datatype class".into(),
            ));
        }
    }

    if matches!(
        dtype.message.class,
        DatatypeClass::FixedPoint
            | DatatypeClass::Time
            | DatatypeClass::BitField
            | DatatypeClass::FloatingPoint
    ) {
        if dtype.message.properties.len() < 4 {
            dtype.message.properties.resize(4, 0);
        }
        let mut offset = u32::from(dtype.message.bit_offset().unwrap_or(0));
        let mut precision = u32::from(dtype.message.precision().unwrap_or(0));
        let size_bits = size
            .checked_mul(8)
            .ok_or_else(|| Error::InvalidFormat("datatype size in bits overflows u32".into()))?;
        if precision > size_bits {
            offset = 0;
        } else if offset.checked_add(precision).ok_or_else(|| {
            Error::InvalidFormat("datatype offset plus precision overflows".into())
        })? > size_bits
        {
            offset = size_bits - precision;
        }
        if precision > size_bits {
            precision = size_bits;
        }

        if dtype.message.class == DatatypeClass::FloatingPoint {
            let end = offset
                .checked_add(precision)
                .ok_or_else(|| Error::InvalidFormat("datatype precision end overflows".into()))?;
            let sign = u32::from(dtype.message.class_bits[1]);
            let exponent = u32::from(dtype.message.properties[4])
                .checked_add(u32::from(dtype.message.properties[5]))
                .ok_or_else(|| {
                    Error::InvalidFormat("floating-point exponent field overflows".into())
                })?;
            let mantissa = u32::from(dtype.message.properties[6])
                .checked_add(u32::from(dtype.message.properties[7]))
                .ok_or_else(|| {
                    Error::InvalidFormat("floating-point mantissa field overflows".into())
                })?;
            if sign >= end || exponent > end || mantissa > end {
                return Err(Error::InvalidFormat(
                    "adjust sign, mantissa, and exponent fields before decreasing size".into(),
                ));
            }
        }

        write_le_u16_at(
            &mut dtype.message.properties,
            0,
            u16::try_from(offset)
                .map_err(|_| Error::InvalidFormat("datatype bit offset exceeds u16".into()))?,
        );
        write_le_u16_at(
            &mut dtype.message.properties,
            2,
            u16::try_from(precision)
                .map_err(|_| Error::InvalidFormat("datatype precision exceeds u16".into()))?,
        );
    }

    dtype.message.size = size;
    Ok(())
}

/// Datatype operation: cmp.
#[allow(non_snake_case)]
pub fn H5T_cmp(left: &RuntimeDatatype, right: &RuntimeDatatype) -> bool {
    left.message.class == right.message.class
        && left.message.size == right.message.size
        && left.message.class_bits == right.message.class_bits
        && left.message.properties == right.message.properties
}

/// Initialize the datatype subsystem.
#[allow(non_snake_case)]
pub fn H5T__init_path_table(registry: &mut DatatypeRegistry) {
    registry.paths.clear();
}

/// Datatype operation: path table search.
#[allow(non_snake_case)]
pub fn H5T__path_table_search_ref(
    registry: &DatatypeRegistry,
    src: DatatypeClass,
    dst: DatatypeClass,
) -> Option<&str> {
    registry
        .paths
        .get(&(src as u8, dst as u8))
        .map(String::as_str)
}

/// Datatype operation: path table search.
#[deprecated(note = "use H5T__path_table_search_ref() to borrow the path name")]
#[allow(non_snake_case)]
pub fn H5T__path_table_search(
    registry: &DatatypeRegistry,
    src: DatatypeClass,
    dst: DatatypeClass,
) -> Option<String> {
    H5T__path_table_search_ref(registry, src, dst).map(str::to_string)
}

/// Find an entry in a datatype.
#[allow(non_snake_case)]
pub fn H5T__path_find_real<'a>(
    registry: &'a DatatypeRegistry,
    src: &RuntimeDatatype,
    dst: &RuntimeDatatype,
) -> Option<Cow<'a, str>> {
    if H5T_path_noop(src, dst) {
        Some(Cow::Borrowed("noop"))
    } else {
        H5T__path_table_search_ref(registry, src.message.class, dst.message.class)
            .map(Cow::Borrowed)
            .or_else(|| H5Tcompiler_conv(src, dst).then_some(Cow::Borrowed("compiler")))
    }
}

/// Find an entry in a datatype.
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

/// Free a datatype's in-memory resources.
#[allow(non_snake_case)]
pub fn H5T__path_free(registry: &mut DatatypeRegistry) {
    registry.paths.clear();
}

/// Datatype operation: path match.
#[allow(non_snake_case)]
pub fn H5T_path_match(src: &RuntimeDatatype, dst: &RuntimeDatatype) -> bool {
    if src.force_conv || dst.force_conv {
        return false;
    }
    if src.message.class != dst.message.class {
        return false;
    }
    if src.message.size != dst.message.size {
        return false;
    }
    if src.message.class_bits != dst.message.class_bits {
        return false;
    }
    if src.message.properties != dst.message.properties {
        return false;
    }
    match (&src.owned_vol_obj, &dst.owned_vol_obj) {
        (Some(left), Some(right)) => left == right,
        (None, None) => true,
        _ => false,
    }
}

/// Datatype operation: path noop.
#[allow(non_snake_case)]
pub fn H5T_path_noop(src: &RuntimeDatatype, dst: &RuntimeDatatype) -> bool {
    H5T_path_match(src, dst)
}

/// Datatype operation: noop conv.
#[allow(non_snake_case)]
pub fn H5T_noop_conv(
    registry: &DatatypeRegistry,
    src: &RuntimeDatatype,
    dst: &RuntimeDatatype,
) -> bool {
    if !src.force_conv && !dst.force_conv && H5T_cmp(src, dst) {
        return true;
    }

    registry
        .paths
        .get(&(src.message.class as u8, dst.message.class as u8))
        .is_some_and(|name| name == "noop")
}

/// Datatype operation: path bkg.
#[allow(non_snake_case)]
pub fn H5T_path_bkg(_src: &RuntimeDatatype, _dst: &RuntimeDatatype) -> bool {
    false
}

/// Convert a datatype.
#[allow(non_snake_case)]
pub fn H5T_convert_into(
    src: &RuntimeDatatype,
    dst: &RuntimeDatatype,
    data: &[u8],
    out: &mut Vec<u8>,
) -> Result<()> {
    if H5T_path_match(src, dst) || H5Tcompiler_conv(src, dst) {
        out.clear();
        out.extend_from_slice(data);
        Ok(())
    } else {
        Err(Error::Unsupported(format!(
            "datatype conversion {:?} -> {:?} is not supported",
            src.message.class, dst.message.class
        )))
    }
}

/// Convert a datatype.
#[deprecated(note = "use H5T_convert_into() to reuse caller-provided output storage")]
#[allow(non_snake_case)]
pub fn H5T_convert(src: &RuntimeDatatype, dst: &RuntimeDatatype, data: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(data.len());
    H5T_convert_into(src, dst, data, &mut out)?;
    Ok(out)
}

/// Convert a datatype.
#[deprecated(note = "use H5T_convert_into() to reuse caller-provided output storage")]
#[allow(non_snake_case)]
pub fn H5T_convert_committed_datatype(
    src: &RuntimeDatatype,
    dst: &RuntimeDatatype,
    data: &[u8],
) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(data.len());
    H5T_convert_into(src, dst, data, &mut out)?;
    Ok(out)
}

/// Datatype operation: nameof.
#[allow(non_snake_case)]
pub fn H5T_nameof(dtype: &RuntimeDatatype) -> Result<&str> {
    if dtype.immutable {
        return Err(Error::InvalidFormat("not a named datatype".into()));
    }
    if !dtype.committed {
        return Err(Error::InvalidFormat("not a named datatype".into()));
    }
    dtype
        .name
        .as_deref()
        .ok_or_else(|| Error::InvalidFormat("not a named datatype".into()))
}

/// Return whether a datatype is immutable.
#[allow(non_snake_case)]
pub fn H5T_is_immutable(dtype: &RuntimeDatatype) -> bool {
    dtype.immutable
}

/// Return whether a datatype is a named (committed) type.
#[allow(non_snake_case)]
pub fn H5T_is_named(dtype: &RuntimeDatatype) -> bool {
    dtype.name.is_some()
}

/// Datatype operation: get ref type.
#[allow(non_snake_case)]
pub fn H5T_get_ref_type(dtype: &RuntimeDatatype) -> Option<u8> {
    dtype.message.reference_type()
}

/// Return whether a datatype is sensible (well-formed).
#[allow(non_snake_case)]
pub fn H5T_is_sensible(dtype: &RuntimeDatatype) -> bool {
    match dtype.message.class {
        DatatypeClass::Compound => dtype.message.compound_nmembers().unwrap_or(0) > 0,
        DatatypeClass::Enum => dtype.message.enum_nmembers().unwrap_or(0) > 0,
        DatatypeClass::FixedPoint
        | DatatypeClass::FloatingPoint
        | DatatypeClass::Time
        | DatatypeClass::String
        | DatatypeClass::BitField
        | DatatypeClass::Opaque
        | DatatypeClass::Reference
        | DatatypeClass::VarLen
        | DatatypeClass::Array => true,
    }
}

/// Datatype operation: set loc.
#[allow(non_snake_case)]
pub fn H5T_set_loc(dtype: &mut RuntimeDatatype, loc: impl Into<String>) {
    dtype.loc = Some(loc.into());
}

/// Return whether a datatype requires relocation handling.
#[allow(non_snake_case)]
pub fn H5T_is_relocatable(dtype: &RuntimeDatatype) -> bool {
    dtype.loc.is_none() && !dtype.locked
}

/// Return whether a datatype uses variable-length storage.
#[allow(non_snake_case)]
pub fn H5T_is_vl_storage(dtype: &RuntimeDatatype) -> bool {
    let mut stack = vec![dtype.message.clone()];
    while let Some(message) = stack.pop() {
        match message.class {
            DatatypeClass::VarLen => return true,
            DatatypeClass::Reference => return true,
            DatatypeClass::Compound => {
                if let Ok(fields) = message.compound_fields_iter() {
                    for field in fields {
                        if let Ok(field) = field {
                            stack.push(field.datatype);
                        }
                    }
                }
            }
            DatatypeClass::Array => {
                if let Ok(base) = message.array_base() {
                    stack.push(base);
                }
            }
            DatatypeClass::Enum => {
                if let Ok(base) = message.enum_base() {
                    stack.push(base);
                }
            }
            DatatypeClass::FixedPoint
            | DatatypeClass::FloatingPoint
            | DatatypeClass::Time
            | DatatypeClass::String
            | DatatypeClass::BitField
            | DatatypeClass::Opaque => {}
        }
    }
    false
}

/// Datatype operation: detect vlen ref.
#[allow(non_snake_case)]
pub fn H5T__detect_vlen_ref(dtype: &RuntimeDatatype) -> bool {
    matches!(
        dtype.message.class,
        DatatypeClass::VarLen | DatatypeClass::Reference
    ) || dtype
        .message
        .compound_fields_iter()
        .map(|fields| {
            for field in fields {
                let Ok(field) = field else {
                    return false;
                };
                if matches!(
                    field.datatype.class,
                    DatatypeClass::VarLen | DatatypeClass::Reference
                ) {
                    return true;
                }
            }
            false
        })
        .unwrap_or(false)
}

/// Datatype operation: upgrade version cb.
#[allow(non_snake_case)]
pub fn H5T__upgrade_version_cb(dtype: &mut RuntimeDatatype, version: u8) {
    dtype.message.version = version;
}

/// Datatype operation: upgrade version.
#[allow(non_snake_case)]
pub fn H5T__upgrade_version(dtype: &mut RuntimeDatatype, version: u8) {
    H5T__upgrade_version_cb(dtype, version);
}

/// Datatype operation: set version.
#[allow(non_snake_case)]
pub fn H5T_set_version(
    dtype: &mut RuntimeDatatype,
    low_bound: usize,
    high_bound: usize,
) -> Result<()> {
    let low_version = *H5O_DTYPE_VER_BOUNDS
        .get(low_bound)
        .ok_or_else(|| Error::InvalidFormat("datatype low version bound is invalid".into()))?;
    let high_version = *H5O_DTYPE_VER_BOUNDS
        .get(high_bound)
        .ok_or_else(|| Error::InvalidFormat("datatype high version bound is invalid".into()))?;

    if low_version > dtype.message.version {
        H5T__upgrade_version(dtype, low_version);
    }
    if dtype.message.version > high_version {
        return Err(Error::InvalidFormat(
            "datatype version out of bounds".into(),
        ));
    }
    Ok(())
}

/// Datatype operation: own vol obj.
#[allow(non_snake_case)]
pub fn H5T_own_vol_obj(dtype: &mut RuntimeDatatype, vol_obj: impl Into<String>) -> Result<()> {
    let vol_obj = vol_obj.into();
    if vol_obj.is_empty() {
        return Err(Error::InvalidFormat("VOL object must not be empty".into()));
    }
    dtype.owned_vol_obj = Some(vol_obj);
    Ok(())
}

/// Datatype operation: get path table npaths.
#[allow(non_snake_case)]
pub fn H5T__get_path_table_npaths(registry: &DatatypeRegistry) -> usize {
    registry.paths.len()
}

/// Datatype operation: is numeric with unusual unused bits.
#[allow(non_snake_case)]
pub fn H5T_is_numeric_with_unusual_unused_bits(dtype: &RuntimeDatatype) -> bool {
    matches!(
        dtype.message.class,
        DatatypeClass::FixedPoint | DatatypeClass::FloatingPoint | DatatypeClass::BitField
    ) && dtype.message.bit_offset().unwrap_or(0) != 0
}

/// Datatype operation: enum nameof.
#[allow(non_snake_case)]
pub fn H5T__enum_nameof_ref(dtype: &RuntimeDatatype, value: u64) -> Result<Option<&str>> {
    if dtype.message.class != DatatypeClass::Enum {
        return Err(Error::InvalidFormat("not an enum datatype".into()));
    }

    if !enum_has_members(&dtype.message)? {
        return Err(Error::InvalidFormat("datatype has no members".into()));
    }
    for member in dtype.message.enum_members_iter()? {
        let member = member?;
        if member.value == value {
            return Ok(Some(member.name));
        }
    }
    Ok(None)
}

/// Datatype operation: enum nameof.
#[deprecated(note = "use H5T__enum_nameof_ref() to borrow the enum member name")]
#[allow(non_snake_case)]
pub fn H5T__enum_nameof(dtype: &RuntimeDatatype, value: u64) -> Result<Option<String>> {
    H5T__enum_nameof_ref(dtype, value).map(|name| name.map(str::to_string))
}

/// Datatype operation: enum valueof.
#[allow(non_snake_case)]
pub fn H5T__enum_valueof(dtype: &RuntimeDatatype, name: &str) -> Result<Option<u64>> {
    if dtype.message.class != DatatypeClass::Enum {
        return Err(Error::InvalidFormat("not an enum datatype".into()));
    }
    if name.is_empty() {
        return Err(Error::InvalidFormat("enum member name is empty".into()));
    }

    if !enum_has_members(&dtype.message)? {
        return Err(Error::InvalidFormat("datatype has no members".into()));
    }
    dtype.message.enum_valueof(name)
}

/// Create a new datatype.
#[allow(non_snake_case)]
pub fn H5T__enum_create(base: &RuntimeDatatype) -> Result<RuntimeDatatype> {
    Ok(RuntimeDatatype {
        name: None,
        message: DatatypeMessage::enum_create(base.message.clone())?,
        locked: false,
        committed: false,
        immutable: false,
        loc: None,
        tag: None,
        force_conv: false,
        owned_vol_obj: None,
    })
}

/// Create a new datatype.
#[allow(non_snake_case)]
pub fn H5Tenum_create(base: &RuntimeDatatype) -> Result<RuntimeDatatype> {
    H5T__enum_create(base)
}

/// Insert an entry into a datatype.
#[allow(non_snake_case)]
pub fn H5T__enum_insert(dtype: &mut RuntimeDatatype, name: &str, value: u64) -> Result<()> {
    if dtype.message.class != DatatypeClass::Enum {
        return Err(Error::InvalidFormat("not an enum datatype".into()));
    }
    if name.is_empty() {
        return Err(Error::InvalidFormat(
            "enum datatype member name must not be empty".into(),
        ));
    }
    for member in dtype.message.enum_members_iter()? {
        let member = member?;
        let member_name = member.name;
        let member_value = member.value;
        if member_name == name {
            return Err(Error::InvalidFormat("name redefinition".into()));
        }
        if member_value == value {
            return Err(Error::InvalidFormat("value redefinition".into()));
        }
    }
    dtype.message.enum_insert(name, value)
}

/// Insert an entry into a datatype.
#[allow(non_snake_case)]
pub fn H5Tenum_insert(dtype: &mut RuntimeDatatype, name: &str, value: u64) -> Result<()> {
    H5T__enum_insert(dtype, name, value)
}

/// Datatype operation: enum nameof.
#[deprecated(note = "use H5T__enum_nameof_ref() to borrow the enum member name")]
#[allow(non_snake_case)]
pub fn H5Tenum_nameof(dtype: &RuntimeDatatype, value: u64) -> Result<Option<String>> {
    H5T__enum_nameof_ref(dtype, value).map(|name| name.map(str::to_string))
}

/// Datatype operation: enum valueof.
#[allow(non_snake_case)]
pub fn H5Tenum_valueof(dtype: &RuntimeDatatype, name: &str) -> Result<Option<u64>> {
    H5T__enum_valueof(dtype, name)
}

/// Iterate enum members without allocating member names.
#[allow(non_snake_case)]
pub fn H5Tenum_members_iter(dtype: &RuntimeDatatype) -> Result<EnumMembers<'_>> {
    dtype.message.enum_members_iter()
}

/// Commit a datatype to the file as a named object.
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

/// Commit a datatype to the file as a named object.
#[allow(non_snake_case)]
pub fn H5T__commit_named(registry: &mut DatatypeRegistry, name: &str, dtype: RuntimeDatatype) {
    H5T__commit_api_common(registry, name, dtype);
}

/// Commit a datatype to the file as a named object.
#[allow(non_snake_case)]
pub fn H5Tcommit_anon(mut dtype: RuntimeDatatype) -> RuntimeDatatype {
    dtype.committed = true;
    dtype
}

/// Commit a datatype to the file as a named object.
#[allow(non_snake_case)]
pub fn H5T__commit(registry: &mut DatatypeRegistry, name: &str, dtype: RuntimeDatatype) {
    H5T__commit_api_common(registry, name, dtype);
}

/// Datatype operation: commit1.
#[allow(non_snake_case)]
pub fn H5Tcommit1(registry: &mut DatatypeRegistry, name: &str, dtype: RuntimeDatatype) {
    H5T__commit_api_common(registry, name, dtype);
}

/// Datatype operation: committed.
#[allow(non_snake_case)]
pub fn H5Tcommitted(dtype: &RuntimeDatatype) -> bool {
    dtype.committed
}

/// Link a datatype.
#[allow(non_snake_case)]
pub fn H5T_link(dtype: &mut RuntimeDatatype, name: &str) {
    dtype.name = Some(name.to_string());
    dtype.committed = true;
}

/// Open a datatype.
#[allow(non_snake_case)]
pub fn H5T__open_api_common_ref<'a>(
    registry: &'a DatatypeRegistry,
    name: &str,
) -> Option<&'a RuntimeDatatype> {
    H5Tfind_ref(registry, name)
}

/// Open a datatype.
#[deprecated(note = "use H5T__open_api_common_ref() to borrow the registered datatype")]
#[allow(non_snake_case)]
pub fn H5T__open_api_common(registry: &DatatypeRegistry, name: &str) -> Option<RuntimeDatatype> {
    H5T__open_api_common_ref(registry, name).cloned()
}

/// Datatype operation: open2.
#[deprecated(note = "use H5Topen2_ref() to borrow the registered datatype")]
#[allow(non_snake_case)]
pub fn H5Topen2(registry: &DatatypeRegistry, name: &str) -> Option<RuntimeDatatype> {
    H5Tfind_ref(registry, name).cloned()
}

/// Datatype operation: open2.
#[allow(non_snake_case)]
pub fn H5Topen2_ref<'a>(registry: &'a DatatypeRegistry, name: &str) -> Option<&'a RuntimeDatatype> {
    H5Tfind_ref(registry, name)
}

/// Open a datatype.
#[deprecated(note = "use H5Topen_async_ref() to borrow the registered datatype")]
#[allow(non_snake_case)]
pub fn H5Topen_async(registry: &DatatypeRegistry, name: &str) -> Option<RuntimeDatatype> {
    H5Tfind_ref(registry, name).cloned()
}

/// Open a datatype.
#[allow(non_snake_case)]
pub fn H5Topen_async_ref<'a>(
    registry: &'a DatatypeRegistry,
    name: &str,
) -> Option<&'a RuntimeDatatype> {
    H5Tfind_ref(registry, name)
}

/// Open a datatype by name.
#[deprecated(note = "use H5T__open_name_ref() to borrow the registered datatype")]
#[allow(non_snake_case)]
pub fn H5T__open_name(registry: &DatatypeRegistry, name: &str) -> Option<RuntimeDatatype> {
    H5Tfind_ref(registry, name).cloned()
}

/// Open a datatype by name.
#[allow(non_snake_case)]
pub fn H5T__open_name_ref<'a>(
    registry: &'a DatatypeRegistry,
    name: &str,
) -> Option<&'a RuntimeDatatype> {
    H5Tfind_ref(registry, name)
}

/// Open a datatype.
#[deprecated(note = "use H5T_open_ref() to borrow the registered datatype")]
#[allow(non_snake_case)]
pub fn H5T_open(registry: &DatatypeRegistry, name: &str) -> Option<RuntimeDatatype> {
    H5Tfind_ref(registry, name).cloned()
}

/// Open a datatype.
#[allow(non_snake_case)]
pub fn H5T_open_ref<'a>(registry: &'a DatatypeRegistry, name: &str) -> Option<&'a RuntimeDatatype> {
    H5Tfind_ref(registry, name)
}

/// Datatype operation: open1.
#[deprecated(note = "use H5Topen1_ref() to borrow the registered datatype")]
#[allow(non_snake_case)]
pub fn H5Topen1(registry: &DatatypeRegistry, name: &str) -> Option<RuntimeDatatype> {
    H5Tfind_ref(registry, name).cloned()
}

/// Datatype operation: open1.
#[allow(non_snake_case)]
pub fn H5Topen1_ref<'a>(registry: &'a DatatypeRegistry, name: &str) -> Option<&'a RuntimeDatatype> {
    H5Tfind_ref(registry, name)
}

/// Open a datatype.
#[deprecated(note = "use H5T__open_oid_ref() to borrow the registered datatype")]
#[allow(non_snake_case)]
pub fn H5T__open_oid(registry: &DatatypeRegistry, name: &str) -> Option<RuntimeDatatype> {
    H5Tfind_ref(registry, name).cloned()
}

/// Open a datatype.
#[allow(non_snake_case)]
pub fn H5T__open_oid_ref<'a>(
    registry: &'a DatatypeRegistry,
    name: &str,
) -> Option<&'a RuntimeDatatype> {
    H5Tfind_ref(registry, name)
}

/// Update a datatype.
#[allow(non_snake_case)]
pub fn H5T_update_shared(dtype: &mut RuntimeDatatype, name: &str) {
    dtype.name = Some(name.to_string());
}

/// Datatype operation: destruct datatype.
#[allow(non_snake_case)]
pub fn H5T_destruct_datatype(_dtype: RuntimeDatatype) {}

/// Datatype operation: construct datatype.
#[allow(non_snake_case)]
pub fn H5T_construct_datatype(message: DatatypeMessage) -> RuntimeDatatype {
    RuntimeDatatype {
        name: None,
        message,
        locked: false,
        committed: false,
        immutable: false,
        loc: None,
        tag: None,
        force_conv: false,
        owned_vol_obj: None,
    }
}

/// Datatype operation: already vol managed.
#[allow(non_snake_case)]
pub fn H5T_already_vol_managed(_dtype: &RuntimeDatatype) -> bool {
    false
}

/// Datatype operation: get named type.
#[allow(non_snake_case)]
pub fn H5T_get_named_type_ref(dtype: &RuntimeDatatype) -> Option<&str> {
    dtype.name.as_deref()
}

/// Datatype operation: get named type.
#[deprecated(note = "use H5T_get_named_type_ref() to borrow the datatype name")]
#[allow(non_snake_case)]
pub fn H5T_get_named_type(dtype: &RuntimeDatatype) -> Option<String> {
    H5T_get_named_type_ref(dtype).map(str::to_string)
}

/// Datatype operation: get actual type.
#[allow(non_snake_case)]
pub fn H5T_get_actual_type(dtype: &RuntimeDatatype) -> DatatypeClass {
    dtype.message.class
}

/// Refresh the datatype from storage.
#[allow(non_snake_case)]
pub fn H5T_save_refresh_state(dtype: &RuntimeDatatype) -> RuntimeDatatype {
    dtype.clone()
}

/// Refresh the datatype from storage.
#[allow(non_snake_case)]
pub fn H5T_restore_refresh_state(dst: &mut RuntimeDatatype, saved: RuntimeDatatype) {
    *dst = saved;
}

/// Return the precision of a datatype.
#[allow(non_snake_case)]
pub fn H5T_get_precision(dtype: &RuntimeDatatype) -> Option<u16> {
    dtype.message.precision()
}

/// Return the size of a datatype.
#[allow(non_snake_case)]
pub fn H5Tget_size(dtype: &RuntimeDatatype) -> usize {
    usize::try_from(dtype.message.size).unwrap_or(usize::MAX)
}

/// Return the class of a datatype.
#[allow(non_snake_case)]
pub fn H5Tget_class(dtype: &RuntimeDatatype) -> DatatypeClass {
    dtype.message.class
}

/// Return the byte order of a datatype.
#[allow(non_snake_case)]
pub fn H5Tget_order(dtype: &RuntimeDatatype) -> Option<ByteOrder> {
    match dtype.message.class {
        DatatypeClass::FixedPoint | DatatypeClass::FloatingPoint | DatatypeClass::BitField => {
            if dtype.message.class_bits[0] & 0x01 == 0 {
                Some(ByteOrder::LittleEndian)
            } else {
                Some(ByteOrder::BigEndian)
            }
        }
        DatatypeClass::Enum => dtype
            .message
            .enum_base()
            .ok()
            .and_then(|base| H5Tget_order(&H5T_construct_datatype(base))),
        _ => None,
    }
}

/// Return the sign convention of a datatype.
#[allow(non_snake_case)]
pub fn H5Tget_sign(dtype: &RuntimeDatatype) -> Option<bool> {
    dtype.message.is_signed()
}

/// Return the offset of a datatype.
#[allow(non_snake_case)]
pub fn H5Tget_offset(dtype: &RuntimeDatatype) -> Option<u16> {
    dtype.message.bit_offset()
}

/// Return the precision of a datatype.
#[allow(non_snake_case)]
pub fn H5Tget_precision(dtype: &RuntimeDatatype) -> Option<u16> {
    dtype.message.precision()
}

/// Return the floating-point field layout of a datatype.
#[allow(non_snake_case)]
pub fn H5Tget_fields(dtype: &RuntimeDatatype) -> Option<FloatFields> {
    if dtype.message.class != DatatypeClass::FloatingPoint {
        return None;
    }
    let properties = &dtype.message.properties;
    if properties.len() < 8 {
        return None;
    }
    Some(FloatFields {
        sign_position: dtype.message.class_bits[1],
        exponent_position: properties[4],
        exponent_size: properties[5],
        mantissa_position: properties[6],
        mantissa_size: properties[7],
    })
}

/// Return the floating-point exponent bias of a datatype.
#[allow(non_snake_case)]
pub fn H5Tget_ebias(dtype: &RuntimeDatatype) -> Option<u32> {
    dtype.message.exponent_bias()
}

/// Return the floating-point normalization of a datatype.
#[allow(non_snake_case)]
pub fn H5Tget_norm(dtype: &RuntimeDatatype) -> Option<u8> {
    if dtype.message.class != DatatypeClass::FloatingPoint {
        return None;
    }
    match (dtype.message.class_bits[0] >> 4) & 0x03 {
        0 => Some(2),
        1 => Some(1),
        2 => Some(0),
        _ => None,
    }
}

/// Return the internal padding of a datatype.
#[allow(non_snake_case)]
pub fn H5Tget_inpad(dtype: &RuntimeDatatype) -> Option<u8> {
    if dtype.message.class == DatatypeClass::FloatingPoint {
        Some((dtype.message.class_bits[0] >> 3) & 0x01)
    } else {
        None
    }
}

/// Return the creation property list for a datatype.
#[allow(non_snake_case)]
pub fn H5Tget_create_plist(dtype: &RuntimeDatatype) -> DatatypeCreate {
    DatatypeCreate::from_datatype(&crate::hl::datatype::Datatype::from_message(
        dtype.message.clone(),
    ))
}

/// Return the native equivalent of a datatype.
#[allow(non_snake_case)]
pub fn H5Tget_native_type(dtype: &RuntimeDatatype) -> RuntimeDatatype {
    dtype.clone()
}

/// Return the padding of a datatype.
#[allow(non_snake_case)]
pub fn H5Tget_pad(dtype: &RuntimeDatatype) -> Option<(u8, u8)> {
    match dtype.message.class {
        DatatypeClass::FixedPoint | DatatypeClass::BitField | DatatypeClass::FloatingPoint => {
            let plist = H5Tget_create_plist(dtype);
            Some((plist.low_pad(), plist.high_pad()))
        }
        _ => None,
    }
}

/// Return the index of a compound or enum member by name.
#[allow(non_snake_case)]
pub fn H5Tget_member_index(dtype: &RuntimeDatatype, name: &str) -> Option<usize> {
    if name.is_empty() {
        return None;
    }

    match dtype.message.class {
        DatatypeClass::Compound => {
            for (index, field) in dtype.message.compound_fields_iter().ok()?.enumerate() {
                let field = field.ok()?;
                if field.name == name {
                    return Some(index);
                }
            }
            None
        }
        DatatypeClass::Enum => {
            for (index, member) in dtype.message.enum_members_iter().ok()?.enumerate() {
                let member = member.ok()?;
                if member.name == name {
                    return Some(index);
                }
            }
            None
        }
        _ => None,
    }
}

/// Return the class of a compound member.
#[allow(non_snake_case)]
pub fn H5Tget_member_class(dtype: &RuntimeDatatype, index: usize) -> Option<DatatypeClass> {
    dtype
        .message
        .compound_fields_iter()
        .ok()?
        .nth(index)?
        .ok()
        .map(|field| field.class)
}

/// Return whether a datatype is a variable-length string.
#[allow(non_snake_case)]
pub fn H5Tis_variable_str(dtype: &RuntimeDatatype) -> bool {
    H5T_is_variable_str(dtype).unwrap_or(false)
}

/// Return whether a datatype is a variable-length string.
#[allow(non_snake_case)]
pub fn H5T_is_variable_str(dtype: &RuntimeDatatype) -> Result<bool> {
    match dtype.message.class {
        DatatypeClass::VarLen => Ok(dtype.message.is_variable_string()),
        _ => Ok(false),
    }
}

/// Return the character set of a datatype.
#[allow(non_snake_case)]
pub fn H5Tget_cset(dtype: &RuntimeDatatype) -> Option<u8> {
    match dtype.message.class {
        DatatypeClass::String => Some((dtype.message.class_bits[0] >> 4) & 0x0f),
        DatatypeClass::VarLen if dtype.message.is_variable_string() => {
            Some(dtype.message.class_bits[1] & 0x0f)
        }
        _ => None,
    }
}

/// Return the string-padding of a datatype.
#[allow(non_snake_case)]
pub fn H5Tget_strpad(dtype: &RuntimeDatatype) -> Option<u8> {
    match dtype.message.class {
        DatatypeClass::String => Some(dtype.message.class_bits[0] & 0x0f),
        DatatypeClass::VarLen if dtype.message.is_variable_string() => {
            Some((dtype.message.class_bits[0] >> 4) & 0x0f)
        }
        _ => None,
    }
}

/// Return the tag string of a datatype.
#[allow(non_snake_case)]
pub fn H5Tget_tag_ref(dtype: &RuntimeDatatype) -> Option<&str> {
    if let Some(tag) = &dtype.tag {
        return Some(tag);
    }
    if dtype.message.class == DatatypeClass::Opaque {
        dtype.message.opaque_tag_str()
    } else {
        None
    }
}

/// Return the tag string of a datatype.
#[deprecated(note = "use H5Tget_tag_ref() to borrow the tag")]
#[allow(non_snake_case)]
pub fn H5Tget_tag(dtype: &RuntimeDatatype) -> Option<String> {
    H5Tget_tag_ref(dtype).map(str::to_string)
}

/// Return info about compound members without collecting all fields.
#[allow(non_snake_case)]
pub fn H5Tget_member_info_iter(dtype: &RuntimeDatatype) -> Result<CompoundFields<'_>> {
    dtype.message.compound_fields_iter()
}

/// Return info about a compound or enum member.
#[deprecated(note = "use H5Tget_member_info_iter() to avoid collecting all fields")]
#[allow(non_snake_case)]
pub fn H5Tget_member_info(dtype: &RuntimeDatatype) -> Result<Vec<CompoundField>> {
    collect_compound_fields(&dtype.message)
}

/// Return the value of an enum member.
#[allow(non_snake_case)]
pub fn H5Tget_member_value(dtype: &RuntimeDatatype, index: usize) -> Result<Option<u64>> {
    if dtype.message.class != DatatypeClass::Enum {
        return Err(Error::InvalidFormat("not an enum datatype".into()));
    }
    Ok(dtype
        .message
        .enum_members_iter()?
        .nth(index)
        .transpose()?
        .map(|member| member.value))
}

/// Return the base type of a derived datatype.
#[allow(non_snake_case)]
pub fn H5Tget_super(dtype: &RuntimeDatatype) -> Result<Option<RuntimeDatatype>> {
    let message = match dtype.message.class {
        DatatypeClass::Enum => Some(dtype.message.enum_base()?),
        DatatypeClass::VarLen => dtype.message.vlen_base()?,
        DatatypeClass::Array => Some(dtype.message.array_base()?),
        _ => None,
    };
    Ok(message.map(H5T_construct_datatype))
}

/// Return the dimensions of an array datatype.
#[allow(non_snake_case)]
pub fn H5Tget_array_dims_iter(dtype: &RuntimeDatatype) -> Result<Option<ArrayDims<'_>>> {
    if dtype.message.class == DatatypeClass::Array {
        dtype.message.array_dims_iter().map(Some)
    } else {
        Ok(None)
    }
}

/// Return the dimensions of an array datatype.
#[deprecated(note = "use H5Tget_array_dims_iter() to avoid allocating dimensions")]
#[allow(non_snake_case)]
pub fn H5Tget_array_dims(dtype: &RuntimeDatatype) -> Result<Option<Vec<u64>>> {
    H5Tget_array_dims_iter(dtype)?
        .map(|dims| dims.collect::<Result<Vec<_>>>())
        .transpose()
}

/// Set the precision of a datatype.
#[allow(non_snake_case)]
pub fn H5Tset_precision(dtype: &mut RuntimeDatatype, precision: u16) -> Result<()> {
    H5T__set_precision(dtype, precision)
}

/// Set the precision of a datatype.
#[allow(non_snake_case)]
pub fn H5T__set_precision(dtype: &mut RuntimeDatatype, precision: u16) -> Result<()> {
    if precision == 0 {
        return Err(Error::InvalidFormat(
            "datatype precision must be positive".into(),
        ));
    }
    if dtype.locked || dtype.immutable {
        return Err(Error::InvalidFormat(
            "datatype precision cannot be changed on a locked datatype".into(),
        ));
    }

    match dtype.message.class {
        DatatypeClass::FixedPoint | DatatypeClass::BitField | DatatypeClass::FloatingPoint => {}
        DatatypeClass::Enum => {
            if !enum_has_members(&dtype.message)? {
                return Err(Error::InvalidFormat(
                    "enum datatype with no members has no precision to set".into(),
                ));
            }
            return Err(Error::Unsupported(
                "setting precision on encoded enum base datatypes is unsupported".into(),
            ));
        }
        _ => {
            return Err(Error::Unsupported(
                "datatype precision is not defined for this datatype class".into(),
            ));
        }
    }

    if dtype.message.properties.len() < 4 {
        dtype.message.properties.resize(4, 0);
    }

    let mut offset = u32::from(dtype.message.bit_offset().unwrap_or(0));
    let mut size = dtype.message.size.max(1);
    let precision_u32 = u32::from(precision);
    let size_bits = size
        .checked_mul(8)
        .ok_or_else(|| Error::InvalidFormat("datatype size in bits overflows u32".into()))?;

    if precision_u32 > size_bits {
        offset = 0;
    } else if offset
        .checked_add(precision_u32)
        .ok_or_else(|| Error::InvalidFormat("datatype offset plus precision overflows".into()))?
        > size_bits
    {
        offset = size_bits - precision_u32;
    }
    if precision_u32 > size_bits {
        size = precision_u32.div_ceil(8);
    }

    if dtype.message.class == DatatypeClass::FloatingPoint {
        if dtype.message.properties.len() < 8 {
            dtype.message.properties.resize(8, 0);
        }
        let end = offset
            .checked_add(precision_u32)
            .ok_or_else(|| Error::InvalidFormat("datatype precision end overflows".into()))?;
        let sign = u32::from(dtype.message.class_bits[1]);
        let exponent = u32::from(dtype.message.properties[4])
            .checked_add(u32::from(dtype.message.properties[5]))
            .ok_or_else(|| {
                Error::InvalidFormat("floating-point exponent field overflows".into())
            })?;
        let mantissa = u32::from(dtype.message.properties[6])
            .checked_add(u32::from(dtype.message.properties[7]))
            .ok_or_else(|| {
                Error::InvalidFormat("floating-point mantissa field overflows".into())
            })?;
        if sign >= end || exponent > end || mantissa > end {
            return Err(Error::InvalidFormat(
                "adjust sign, mantissa, and exponent fields before decreasing precision".into(),
            ));
        }
    }

    dtype.message.size = size;
    write_le_u16_at(
        &mut dtype.message.properties,
        0,
        u16::try_from(offset)
            .map_err(|_| Error::InvalidFormat("datatype bit offset exceeds u16".into()))?,
    );
    write_le_u16_at(&mut dtype.message.properties, 2, precision);
    Ok(())
}

/// Set the string-padding of a datatype.
#[allow(non_snake_case)]
pub fn H5Tset_strpad(dtype: &mut RuntimeDatatype, pad: u8) -> Result<()> {
    if dtype.locked || dtype.immutable {
        return Err(Error::InvalidFormat(
            "datatype string padding cannot be changed on a locked datatype".into(),
        ));
    }
    if pad >= 3 {
        return Err(Error::InvalidFormat("illegal string pad type".into()));
    }
    match dtype.message.class {
        DatatypeClass::String => {
            dtype.message.class_bits[0] = (dtype.message.class_bits[0] & !0x0f) | pad;
            Ok(())
        }
        DatatypeClass::VarLen if dtype.message.is_variable_string() => {
            dtype.message.class_bits[0] = (dtype.message.class_bits[0] & !0xf0) | (pad << 4);
            Ok(())
        }
        _ => Err(Error::Unsupported(
            "datatype string padding is not defined for this datatype class".into(),
        )),
    }
}

/// Set the character set of a datatype.
#[allow(non_snake_case)]
pub fn H5Tset_cset(dtype: &mut RuntimeDatatype, cset: u8) -> Result<()> {
    if dtype.locked || dtype.immutable {
        return Err(Error::InvalidFormat(
            "datatype character set cannot be changed on a locked datatype".into(),
        ));
    }
    if cset >= 2 {
        return Err(Error::InvalidFormat("illegal character set type".into()));
    }
    match dtype.message.class {
        DatatypeClass::String => {
            dtype.message.class_bits[0] = (dtype.message.class_bits[0] & !0xf0) | (cset << 4);
            Ok(())
        }
        DatatypeClass::VarLen if dtype.message.is_variable_string() => {
            dtype.message.class_bits[1] = (dtype.message.class_bits[1] & !0x0f) | cset;
            Ok(())
        }
        _ => Err(Error::Unsupported(
            "datatype character set is not defined for this datatype class".into(),
        )),
    }
}

/// Set the tag string of a datatype.
#[allow(non_snake_case)]
pub fn H5Tset_tag(dtype: &mut RuntimeDatatype, tag: impl Into<String>) -> Result<()> {
    if dtype.locked || dtype.immutable {
        return Err(Error::InvalidFormat(
            "opaque datatype tag cannot be changed on a locked datatype".into(),
        ));
    }
    if dtype.message.class != DatatypeClass::Opaque {
        return Err(Error::InvalidFormat("not an opaque data type".into()));
    }
    let tag = tag.into();
    if tag.len() >= 256 {
        return Err(Error::InvalidFormat("opaque datatype tag too long".into()));
    }
    dtype.tag = Some(tag);
    Ok(())
}

/// Return a compound-member offset.
#[allow(non_snake_case)]
pub fn H5Tget_member_offset(dtype: &RuntimeDatatype, index: usize) -> Option<usize> {
    dtype
        .message
        .compound_fields_iter()
        .ok()?
        .nth(index)?
        .ok()
        .map(|field| field.byte_offset)
}

/// Return a compound-member offset.
#[allow(non_snake_case)]
pub fn H5T_get_member_offset(dtype: &RuntimeDatatype, index: usize) -> Option<usize> {
    H5Tget_member_offset(dtype, index)
}

/// Datatype operation: get member size.
#[allow(non_snake_case)]
pub fn H5T__get_member_size(dtype: &RuntimeDatatype, index: usize) -> Option<usize> {
    dtype
        .message
        .compound_fields_iter()
        .ok()?
        .nth(index)?
        .ok()
        .map(|field| field.size)
}

/// Datatype operation: reopen member type.
#[allow(non_snake_case)]
pub fn H5T__reopen_member_type(dtype: &RuntimeDatatype, index: usize) -> Option<RuntimeDatatype> {
    let msg = dtype
        .message
        .compound_fields_iter()
        .ok()?
        .nth(index)?
        .ok()?
        .datatype;
    Some(RuntimeDatatype {
        name: None,
        message: msg,
        locked: false,
        committed: false,
        immutable: false,
        loc: None,
        tag: None,
        force_conv: false,
        owned_vol_obj: None,
    })
}

/// Return a compound-member type.
#[allow(non_snake_case)]
pub fn H5T_get_member_type(dtype: &RuntimeDatatype, index: usize) -> Option<RuntimeDatatype> {
    H5T__reopen_member_type(dtype, index)
}

/// Insert an entry into a datatype.
#[allow(non_snake_case)]
pub fn H5Tinsert(
    dtype: &mut RuntimeDatatype,
    name: &str,
    offset: usize,
    member: RuntimeDatatype,
) -> Result<()> {
    if dtype.message.class != DatatypeClass::Compound {
        return Err(Error::InvalidFormat("not a compound datatype".into()));
    }
    if dtype.locked || dtype.immutable {
        return Err(Error::InvalidFormat("parent type read-only".into()));
    }
    if name.is_empty() || name.as_bytes().contains(&0) {
        return Err(Error::InvalidFormat("no member name".into()));
    }
    if member.message.version > dtype.message.version {
        return Err(Error::Unsupported(
            "upgrading existing compound member encodings is unsupported".into(),
        ));
    }

    let old_nmembers =
        u16::from(dtype.message.class_bits[0]) | (u16::from(dtype.message.class_bits[1]) << 8);
    if old_nmembers == u16::MAX {
        return Err(Error::InvalidFormat(
            "compound datatype member count exceeds u16".into(),
        ));
    }
    if old_nmembers > 0 {
        for field in dtype.message.compound_fields_iter()? {
            let field = field?;
            if field.name == name {
                return Err(Error::InvalidFormat("member name is not unique".into()));
            }
            let field_end = field
                .byte_offset
                .checked_add(field.size)
                .ok_or_else(|| Error::InvalidFormat("compound member end overflows".into()))?;
            let new_end = offset
                .checked_add(usize::try_from(member.message.size).map_err(|_| {
                    Error::InvalidFormat("member datatype size exceeds usize".into())
                })?)
                .ok_or_else(|| Error::InvalidFormat("compound member end overflows".into()))?;
            if (offset <= field.byte_offset && new_end > field.byte_offset)
                || (field.byte_offset <= offset && field_end > offset)
            {
                return Err(Error::InvalidFormat(
                    "member overlaps with another member".into(),
                ));
            }
        }
    }

    let member_size = usize::try_from(member.message.size)
        .map_err(|_| Error::InvalidFormat("member datatype size exceeds usize".into()))?;
    let member_end = offset
        .checked_add(member_size)
        .ok_or_else(|| Error::InvalidFormat("compound member end overflows".into()))?;
    if member_end
        > usize::try_from(dtype.message.size)
            .map_err(|_| Error::InvalidFormat("compound datatype size exceeds usize".into()))?
    {
        return Err(Error::InvalidFormat(
            "member extends past end of compound type".into(),
        ));
    }

    let name_start = dtype.message.properties.len();
    dtype.message.properties.extend_from_slice(name.as_bytes());
    dtype.message.properties.push(0);
    if dtype.message.version < 3 {
        while (dtype.message.properties.len() - name_start) % 8 != 0 {
            dtype.message.properties.push(0);
        }
        dtype.message.properties.extend_from_slice(
            &u32::try_from(offset)
                .map_err(|_| Error::InvalidFormat("compound member offset exceeds u32".into()))?
                .to_le_bytes(),
        );
        dtype.message.properties.extend_from_slice(&[0; 28]);
    } else {
        let mut max_offset = usize::try_from(dtype.message.size)
            .map_err(|_| Error::InvalidFormat("compound datatype size exceeds usize".into()))?
            .saturating_sub(1)
            .max(1);
        let mut offset_size = 1usize;
        while max_offset > 0xff {
            max_offset >>= 8;
            offset_size += 1;
        }
        for shift in 0..offset_size {
            dtype
                .message
                .properties
                .push(((offset >> (8 * shift)) & 0xff) as u8);
        }
    }
    let mut encoded_member = Vec::new();
    H5Tencode_into(&member, &mut encoded_member)?;
    dtype.message.properties.extend_from_slice(&encoded_member);

    let new_nmembers = old_nmembers + 1;
    dtype.message.class_bits[0] = (new_nmembers & 0xff) as u8;
    dtype.message.class_bits[1] = (new_nmembers >> 8) as u8;
    dtype.force_conv |= member.force_conv;
    let mut encoded_dtype = Vec::new();
    H5Tencode_into(dtype, &mut encoded_dtype)?;
    DatatypeMessage::decode(&encoded_dtype)?;
    Ok(())
}

/// Datatype operation: pack.
#[allow(non_snake_case)]
pub fn H5T__pack(_dtype: &mut RuntimeDatatype) {}

/// Datatype operation: is packed.
#[allow(non_snake_case)]
pub fn H5T__is_packed(_dtype: &RuntimeDatatype) -> bool {
    true
}

/// Update a datatype.
#[allow(non_snake_case)]
pub fn H5T__update_packed(_dtype: &mut RuntimeDatatype) {}

/// Return the offset of a datatype.
#[allow(non_snake_case)]
pub fn H5T_get_offset(dtype: &RuntimeDatatype) -> Option<u16> {
    dtype.message.bit_offset()
}

/// Set the offset of a datatype.
#[allow(non_snake_case)]
pub fn H5Tset_offset(dtype: &mut RuntimeDatatype, offset: u16) -> Result<()> {
    H5T__set_offset(dtype, offset)
}

/// Set the offset of a datatype.
#[allow(non_snake_case)]
pub fn H5T__set_offset(dtype: &mut RuntimeDatatype, offset: u16) -> Result<()> {
    if dtype.locked || dtype.immutable {
        return Err(Error::InvalidFormat(
            "datatype offset cannot be changed on a locked datatype".into(),
        ));
    }

    match dtype.message.class {
        DatatypeClass::FixedPoint | DatatypeClass::BitField | DatatypeClass::FloatingPoint => {}
        DatatypeClass::String if offset == 0 => return Ok(()),
        DatatypeClass::String => {
            return Err(Error::InvalidFormat(
                "datatype offset must be zero for strings".into(),
            ));
        }
        DatatypeClass::Enum => {
            if enum_has_members(&dtype.message)? {
                return Err(Error::InvalidFormat(
                    "enum datatype offset cannot change after members are defined".into(),
                ));
            }
            return Err(Error::Unsupported(
                "setting offset on encoded enum base datatypes is unsupported".into(),
            ));
        }
        _ => {
            return Err(Error::Unsupported(
                "datatype offset is not defined for this datatype class".into(),
            ));
        }
    }

    if dtype.message.properties.len() < 4 {
        dtype.message.properties.resize(4, 0);
    }
    let precision = u32::from(dtype.message.precision().unwrap_or(0));
    let end_bit = u32::from(offset)
        .checked_add(precision)
        .ok_or_else(|| Error::InvalidFormat("datatype offset plus precision overflows".into()))?;
    let size_bits = dtype
        .message
        .size
        .max(1)
        .checked_mul(8)
        .ok_or_else(|| Error::InvalidFormat("datatype size in bits overflows u32".into()))?;
    if end_bit > size_bits {
        dtype.message.size = end_bit.div_ceil(8);
    }
    write_le_u16_at(&mut dtype.message.properties, 0, offset);
    Ok(())
}

/// Datatype operation: array create2.
#[allow(non_snake_case)]
pub fn H5Tarray_create2(base: &RuntimeDatatype, dims: &[u64]) -> RuntimeDatatype {
    let mut dtype = base.clone();
    dtype.message.class = DatatypeClass::Array;
    dtype.message.properties = dims.iter().flat_map(|dim| dim.to_le_bytes()).collect();
    dtype
}

/// Create a new datatype.
#[allow(non_snake_case)]
pub fn H5Tvlen_create(base: &RuntimeDatatype) -> RuntimeDatatype {
    let mut dtype = base.clone();
    dtype.message.class = DatatypeClass::VarLen;
    dtype
}

/// Create a new datatype.
#[allow(non_snake_case)]
pub fn H5Tcomplex_create(base: &RuntimeDatatype) -> RuntimeDatatype {
    let mut dtype = base.clone();
    dtype.message.class = DatatypeClass::Compound;
    dtype
}

/// Datatype operation: get force conv.
#[allow(non_snake_case)]
pub fn H5T_get_force_conv(dtype: &RuntimeDatatype) -> bool {
    dtype.force_conv
}

/// Visit the entries of a datatype.
#[allow(non_snake_case)]
pub fn H5T__visit_cb(
    dtype: &RuntimeDatatype,
    mut visitor: impl FnMut(DatatypeClass) -> Result<()>,
) -> Result<()> {
    visitor(dtype.message.class)
}

/// Visit the entries of a datatype.
#[deprecated(note = "use H5T__visit_cb() to avoid allocating a class list")]
#[allow(non_snake_case)]
pub fn H5T__visit(dtype: &RuntimeDatatype) -> Vec<DatatypeClass> {
    let mut classes = Vec::with_capacity(1);
    let _ = H5T__visit_cb(dtype, |class| {
        classes.push(class);
        Ok(())
    });
    classes
}

/// Datatype operation: print path stats.
#[allow(non_snake_case)]
pub fn H5T__print_path_stats_fmt(
    registry: &DatatypeRegistry,
    out: &mut impl fmt::Write,
) -> fmt::Result {
    write!(out, "DatatypePaths({})", registry.paths.len())
}

/// Datatype operation: print path stats.
#[allow(non_snake_case)]
pub fn H5T__print_path_stats_into(registry: &DatatypeRegistry, out: &mut String) {
    let _ = H5T__print_path_stats_fmt(registry, out);
}

/// Datatype operation: print path stats.
#[deprecated(note = "use H5T__print_path_stats_into() to reuse caller-provided String storage")]
#[allow(non_snake_case)]
pub fn H5T__print_path_stats(registry: &DatatypeRegistry) -> String {
    let mut out = String::new();
    H5T__print_path_stats_into(registry, &mut out);
    out
}

/// Return a debug-friendly representation of a datatype.
#[allow(non_snake_case)]
pub fn H5T_debug_fmt(dtype: &RuntimeDatatype, out: &mut impl fmt::Write) -> fmt::Result {
    write!(
        out,
        "RuntimeDatatype(class={:?}, size={})",
        dtype.message.class, dtype.message.size
    )
}

/// Return a debug-friendly representation of a datatype.
#[allow(non_snake_case)]
pub fn H5T_debug_into(dtype: &RuntimeDatatype, out: &mut String) {
    let _ = H5T_debug_fmt(dtype, out);
}

/// Return a debug-friendly representation of a datatype.
#[deprecated(note = "use H5T_debug_into() to reuse caller-provided String storage")]
#[allow(non_snake_case)]
pub fn H5T_debug(dtype: &RuntimeDatatype) -> String {
    let mut out = String::new();
    H5T_debug_into(dtype, &mut out);
    out
}

/// Internal helper `conv_copy`.
fn conv_copy(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    conv_copy_into(data, &mut out);
    out
}

/// Internal helper `conv_copy_into`.
fn conv_copy_into(data: &[u8], out: &mut Vec<u8>) {
    out.clear();
    out.extend_from_slice(data);
}

/// Initialize the datatype subsystem.
#[allow(non_snake_case)]
pub fn H5T__conv_enum_init_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Initialize the datatype subsystem.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_enum_init(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_enum_init_into(data, &mut out);
    out
}
/// Datatype operation: conv enum.
#[allow(non_snake_case)]
pub fn H5T__conv_enum_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv enum.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_enum(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_enum_into(data, &mut out);
    out
}
/// Datatype operation: conv enum numeric.
#[allow(non_snake_case)]
pub fn H5T__conv_enum_numeric_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv enum numeric.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_enum_numeric(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_enum_numeric_into(data, &mut out);
    out
}
/// Datatype operation: conv struct subset.
#[allow(non_snake_case)]
pub fn H5T__conv_struct_subset_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv struct subset.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_struct_subset(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_struct_subset_into(data, &mut out);
    out
}
/// Initialize the datatype subsystem.
#[allow(non_snake_case)]
pub fn H5T__conv_struct_init_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Initialize the datatype subsystem.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_struct_init(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_struct_init_into(data, &mut out);
    out
}
/// Free a datatype's in-memory resources.
#[allow(non_snake_case)]
pub fn H5T__conv_struct_free_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Free a datatype's in-memory resources.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_struct_free(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_struct_free_into(data, &mut out);
    out
}
/// Datatype operation: conv struct.
#[allow(non_snake_case)]
pub fn H5T__conv_struct_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv struct.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_struct(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_struct_into(data, &mut out);
    out
}
/// Datatype operation: conv struct opt.
#[allow(non_snake_case)]
pub fn H5T__conv_struct_opt_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv struct opt.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_struct_opt(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_struct_opt_into(data, &mut out);
    out
}
/// Datatype operation: conv complex.
#[allow(non_snake_case)]
pub fn H5T__conv_complex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv complex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_complex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_complex_into(data, &mut out);
    out
}
/// Datatype operation: conv complex loop.
#[allow(non_snake_case)]
pub fn H5T__conv_complex_loop_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv complex loop.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_complex_loop(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_complex_loop_into(data, &mut out);
    out
}
/// Datatype operation: conv complex part.
#[allow(non_snake_case)]
pub fn H5T__conv_complex_part_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv complex part.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_complex_part(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_complex_part_into(data, &mut out);
    out
}
/// Datatype operation: conv complex i.
#[allow(non_snake_case)]
pub fn H5T__conv_complex_i_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv complex i.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_complex_i(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_complex_i_into(data, &mut out);
    out
}
/// Datatype operation: conv complex f.
#[allow(non_snake_case)]
pub fn H5T__conv_complex_f_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv complex f.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_complex_f(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_complex_f_into(data, &mut out);
    out
}
/// Datatype operation: conv complex f matched.
#[allow(non_snake_case)]
pub fn H5T__conv_complex_f_matched_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv complex f matched.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_complex_f_matched(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_complex_f_matched_into(data, &mut out);
    out
}
/// Datatype operation: conv complex compat.
#[allow(non_snake_case)]
pub fn H5T__conv_complex_compat_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv complex compat.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_complex_compat(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_complex_compat_into(data, &mut out);
    out
}
/// Datatype operation: conv ref.
#[allow(non_snake_case)]
pub fn H5T__conv_ref_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ref.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ref(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ref_into(data, &mut out);
    out
}
/// Datatype operation: sort value.
#[allow(non_snake_case)]
pub fn H5T__sort_value(dtype: &mut RuntimeDatatype, map: Option<&mut [usize]>) -> Result<()> {
    match dtype.message.class {
        DatatypeClass::Compound => {
            let mut members = collect_compound_fields(&dtype.message)?
                .into_iter()
                .enumerate()
                .collect::<Vec<_>>();
            members.sort_by_key(|(_, field)| field.byte_offset);

            let mut sorted = RuntimeDatatype::new(DatatypeClass::Compound, dtype.message.size);
            sorted.message.version = dtype.message.version;
            let locked = dtype.locked;
            let immutable = dtype.immutable;
            for (_, field) in &members {
                H5Tinsert(
                    &mut sorted,
                    &field.name,
                    field.byte_offset,
                    H5T_construct_datatype((*field.datatype).clone()),
                )?;
            }
            if let Some(map) = map {
                let old = map.to_vec();
                for (idx, (old_idx, _)) in members.iter().enumerate().take(map.len()) {
                    if let Some(value) = old.get(*old_idx) {
                        map[idx] = *value;
                    }
                }
            }
            dtype.message = sorted.message;
            dtype.locked = locked;
            dtype.immutable = immutable;
            Ok(())
        }
        DatatypeClass::Enum => {
            let base = dtype.message.enum_base()?;
            let mut members = dtype
                .message
                .enum_members_iter()?
                .map(|member| member.map(|member| (member.name.to_string(), member.value)))
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .enumerate()
                .collect::<Vec<_>>();
            members.sort_by_key(|(_, (_, value))| *value);

            let mut sorted = RuntimeDatatype {
                name: None,
                message: DatatypeMessage::enum_create(base)?,
                locked: false,
                committed: false,
                immutable: false,
                loc: None,
                tag: None,
                force_conv: false,
                owned_vol_obj: None,
            };
            sorted.message.version = dtype.message.version;
            for (_, (name, value)) in &members {
                H5T__enum_insert(&mut sorted, name, *value)?;
            }
            if let Some(map) = map {
                let old = map.to_vec();
                for (idx, (old_idx, _)) in members.iter().enumerate().take(map.len()) {
                    if let Some(value) = old.get(*old_idx) {
                        map[idx] = *value;
                    }
                }
            }
            dtype.message = sorted.message;
            Ok(())
        }
        _ => Err(Error::InvalidFormat(
            "datatype members can only be sorted for compound and enum types".into(),
        )),
    }
}
/// Datatype operation: sort name.
#[allow(non_snake_case)]
pub fn H5T__sort_name(dtype: &mut RuntimeDatatype, map: Option<&mut [usize]>) -> Result<()> {
    match dtype.message.class {
        DatatypeClass::Compound => {
            let mut members = collect_compound_fields(&dtype.message)?
                .into_iter()
                .enumerate()
                .collect::<Vec<_>>();
            members.sort_by(|(_, left), (_, right)| left.name.cmp(&right.name));

            let mut sorted = RuntimeDatatype::new(DatatypeClass::Compound, dtype.message.size);
            sorted.message.version = dtype.message.version;
            let locked = dtype.locked;
            let immutable = dtype.immutable;
            for (_, field) in &members {
                H5Tinsert(
                    &mut sorted,
                    &field.name,
                    field.byte_offset,
                    H5T_construct_datatype((*field.datatype).clone()),
                )?;
            }
            if let Some(map) = map {
                let old = map.to_vec();
                for (idx, (old_idx, _)) in members.iter().enumerate().take(map.len()) {
                    if let Some(value) = old.get(*old_idx) {
                        map[idx] = *value;
                    }
                }
            }
            dtype.message = sorted.message;
            dtype.locked = locked;
            dtype.immutable = immutable;
            Ok(())
        }
        DatatypeClass::Enum => {
            let base = dtype.message.enum_base()?;
            let mut members = dtype
                .message
                .enum_members_iter()?
                .map(|member| member.map(|member| (member.name.to_string(), member.value)))
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .enumerate()
                .collect::<Vec<_>>();
            members.sort_by(|(_, (left, _)), (_, (right, _))| left.cmp(right));

            let mut sorted = RuntimeDatatype {
                name: None,
                message: DatatypeMessage::enum_create(base)?,
                locked: false,
                committed: false,
                immutable: false,
                loc: None,
                tag: None,
                force_conv: false,
                owned_vol_obj: None,
            };
            sorted.message.version = dtype.message.version;
            for (_, (name, value)) in &members {
                H5T__enum_insert(&mut sorted, name, *value)?;
            }
            if let Some(map) = map {
                let old = map.to_vec();
                for (idx, (old_idx, _)) in members.iter().enumerate().take(map.len()) {
                    if let Some(value) = old.get(*old_idx) {
                        map[idx] = *value;
                    }
                }
            }
            dtype.message = sorted.message;
            Ok(())
        }
        _ => Err(Error::InvalidFormat(
            "datatype members can only be sorted for compound and enum types".into(),
        )),
    }
}
/// Datatype operation: reverse order.
#[allow(non_snake_case)]
pub fn H5T__reverse_order_into(
    data: &[u8],
    order: H5TDetectedOrder,
    is_complex: bool,
    out: &mut Vec<u8>,
) -> Result<()> {
    let size = data.len();
    if order == H5TDetectedOrder::Vax && size % 2 != 0 {
        return Err(Error::InvalidFormat(
            "VAX byte order requires an even byte count".into(),
        ));
    }
    if order == H5TDetectedOrder::BigEndian && is_complex && size % 2 != 0 {
        return Err(Error::InvalidFormat(
            "complex byte order requires an even byte count".into(),
        ));
    }

    out.clear();
    out.resize(size, 0);
    match order {
        H5TDetectedOrder::Vax => {
            for i in (0..size).step_by(2) {
                out[i] = data[(size - 2) - i];
                out[i + 1] = data[(size - 1) - i];
            }
        }
        H5TDetectedOrder::BigEndian => {
            if is_complex {
                let part_size = size / 2;
                for i in 0..part_size {
                    out[part_size - (i + 1)] = data[i];
                }
                for i in 0..part_size {
                    out[part_size + part_size - (i + 1)] = data[part_size + i];
                }
            } else {
                for i in 0..size {
                    out[size - (i + 1)] = data[i];
                }
            }
        }
        H5TDetectedOrder::LittleEndian => {
            out.copy_from_slice(data);
        }
    }
    Ok(())
}

/// Datatype operation: reverse order.
#[deprecated(note = "use H5T__reverse_order_into() to reuse caller-provided output storage")]
#[allow(non_snake_case)]
pub fn H5T__reverse_order(
    data: &[u8],
    order: H5TDetectedOrder,
    is_complex: bool,
) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(data.len());
    H5T__reverse_order_into(data, order, is_complex, &mut out)?;
    Ok(out)
}
/// Datatype operation: conv noop.
#[allow(non_snake_case)]
pub fn H5T__conv_noop_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv noop.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_noop(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_noop_into(data, &mut out);
    out
}
/// Datatype operation: conv order opt.
#[allow(non_snake_case)]
pub fn H5T__conv_order_opt_into(
    data: &[u8],
    element_size: usize,
    order: H5TDetectedOrder,
    is_complex: bool,
    out: &mut Vec<u8>,
) -> Result<()> {
    if element_size == 0 || !matches!(element_size, 1 | 2 | 4 | 8 | 16) {
        return Err(Error::Unsupported(
            "byte-order optimized conversion only supports 1, 2, 4, 8, or 16 byte elements".into(),
        ));
    }
    if data.len() % element_size != 0 {
        return Err(Error::InvalidFormat(
            "byte-order conversion buffer is not element aligned".into(),
        ));
    }
    if order == H5TDetectedOrder::Vax && element_size % 2 != 0 {
        return Err(Error::InvalidFormat(
            "VAX byte order requires an even byte count".into(),
        ));
    }
    if order == H5TDetectedOrder::BigEndian && is_complex && element_size % 2 != 0 {
        return Err(Error::InvalidFormat(
            "complex byte order requires an even byte count".into(),
        ));
    }

    out.clear();
    let mut reversed = Vec::with_capacity(element_size);
    for element in data.chunks_exact(element_size) {
        H5T__reverse_order_into(element, order, is_complex, &mut reversed)?;
        out.extend_from_slice(&reversed);
    }
    Ok(())
}

/// Datatype operation: conv order opt.
#[deprecated(note = "use H5T__conv_order_opt_into() to reuse caller-provided output storage")]
#[allow(non_snake_case)]
pub fn H5T__conv_order_opt(
    data: &[u8],
    element_size: usize,
    order: H5TDetectedOrder,
    is_complex: bool,
) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_order_opt_into(data, element_size, order, is_complex, &mut out)?;
    Ok(out)
}
/// Datatype operation: conv i f loop.
#[allow(non_snake_case)]
pub fn H5T__conv_i_f_loop_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv i f loop.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_i_f_loop(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_i_f_loop_into(data, &mut out);
    out
}
/// Datatype operation: conv i complex.
#[allow(non_snake_case)]
pub fn H5T__conv_i_complex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv i complex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_i_complex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_i_complex_into(data, &mut out);
    out
}
/// Datatype operation: conv f f loop.
#[allow(non_snake_case)]
pub fn H5T__conv_f_f_loop_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv f f loop.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_f_f_loop(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_f_f_loop_into(data, &mut out);
    out
}
/// Find an entry in a datatype.
#[allow(non_snake_case)]
pub fn H5T__conv_float_find_special(
    data: &[u8],
    fields: FloatFields,
    norm: u8,
) -> Result<(H5TConvFloatSpecval, u64)> {
    let sign_position = fields.sign_position as usize;
    if sign_position >= data.len().saturating_mul(8) {
        return Err(Error::InvalidFormat(
            "floating-point sign bit is outside buffer".into(),
        ));
    }

    let sign = ((data[sign_position / 8] >> (sign_position % 8)) & 1) as u64;
    let mantissa_has_one = H5T__bit_find(
        data,
        fields.mantissa_position as usize,
        fields.mantissa_size as usize,
        H5TBitSearchDirection::Lsb,
        true,
    )
    .is_some();
    let exponent_has_one = H5T__bit_find(
        data,
        fields.exponent_position as usize,
        fields.exponent_size as usize,
        H5TBitSearchDirection::Lsb,
        true,
    )
    .is_some();
    let exponent_all_ones = H5T__bit_find(
        data,
        fields.exponent_position as usize,
        fields.exponent_size as usize,
        H5TBitSearchDirection::Lsb,
        false,
    )
    .is_none();

    let special = if !mantissa_has_one {
        if !exponent_has_one {
            if sign != 0 {
                H5TConvFloatSpecval::NegZero
            } else {
                H5TConvFloatSpecval::PosZero
            }
        } else if exponent_all_ones {
            if sign != 0 {
                H5TConvFloatSpecval::NegInf
            } else {
                H5TConvFloatSpecval::PosInf
            }
        } else {
            H5TConvFloatSpecval::Regular
        }
    } else if norm == 2
        && exponent_all_ones
        && fields.mantissa_size > 0
        && H5T__bit_find(
            data,
            fields.mantissa_position as usize,
            fields.mantissa_size.saturating_sub(1) as usize,
            H5TBitSearchDirection::Lsb,
            true,
        )
        .is_none()
    {
        if sign != 0 {
            H5TConvFloatSpecval::NegInf
        } else {
            H5TConvFloatSpecval::PosInf
        }
    } else if exponent_all_ones {
        H5TConvFloatSpecval::Nan
    } else {
        H5TConvFloatSpecval::Regular
    };

    Ok((special, sign))
}
/// Datatype operation: conv f i loop.
#[allow(non_snake_case)]
pub fn H5T__conv_f_i_loop_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv f i loop.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_f_i_loop(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_f_i_loop_into(data, &mut out);
    out
}
/// Datatype operation: conv f complex.
#[allow(non_snake_case)]
pub fn H5T__conv_f_complex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv f complex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_f_complex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_f_complex_into(data, &mut out);
    out
}
/// Free a datatype's in-memory resources.
#[allow(non_snake_case)]
pub fn H5T__conv_vlen_nested_free_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Free a datatype's in-memory resources.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_vlen_nested_free(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_vlen_nested_free_into(data, &mut out);
    out
}
/// Datatype operation: conv vlen.
#[allow(non_snake_case)]
pub fn H5T__conv_vlen_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv vlen.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_vlen(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_vlen_into(data, &mut out);
    out
}
/// Read from a datatype.
#[allow(non_snake_case)]
pub fn H5T__vlen_mem_seq_read_ref(data: &[u8]) -> &[u8] {
    data
}

/// Read from a datatype.
#[deprecated(note = "use H5T__vlen_mem_seq_read_ref() to borrow the vlen sequence bytes")]
#[allow(non_snake_case)]
pub fn H5T__vlen_mem_seq_read(data: &[u8]) -> Vec<u8> {
    conv_copy(data)
}
/// Datatype operation: conv b b.
#[allow(non_snake_case)]
pub fn H5T__conv_b_b_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv b b.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_b_b(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_b_b_into(data, &mut out);
    out
}
/// Datatype operation: conv s s.
#[allow(non_snake_case)]
pub fn H5T__conv_s_s_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv s s.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_s_s(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_s_s_into(data, &mut out);
    out
}
/// Datatype operation: conv array.
#[allow(non_snake_case)]
pub fn H5T__conv_array_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv array.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_array(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_array_into(data, &mut out);
    out
}

/// Set the floating-point field layout of a datatype.
#[allow(non_snake_case)]
pub fn H5Tset_fields(
    dtype: &mut RuntimeDatatype,
    sign_position: usize,
    exponent_position: usize,
    exponent_size: usize,
    mantissa_position: usize,
    mantissa_size: usize,
) -> Result<()> {
    if dtype.locked || dtype.immutable {
        return Err(Error::InvalidFormat(
            "floating-point fields cannot be changed on a locked datatype".into(),
        ));
    }
    if dtype.message.class != DatatypeClass::FloatingPoint {
        return Err(Error::InvalidFormat(
            "operation not defined for datatype class".into(),
        ));
    }
    if dtype.message.properties.len() < 8 {
        dtype.message.properties.resize(8, 0);
    }
    let offset = usize::from(dtype.message.bit_offset().unwrap_or(0));
    let precision = usize::from(dtype.message.precision().unwrap_or(0));
    let exponent_end = exponent_position
        .checked_add(exponent_size)
        .ok_or_else(|| Error::InvalidFormat("exponent field overflows".into()))?;
    let mantissa_end = mantissa_position
        .checked_add(mantissa_size)
        .ok_or_else(|| Error::InvalidFormat("mantissa field overflows".into()))?;
    let precision_end = offset
        .checked_add(precision)
        .ok_or_else(|| Error::InvalidFormat("floating-point precision overflows".into()))?;
    if exponent_position < offset || exponent_end > precision_end {
        return Err(Error::InvalidFormat(
            "exponent bit field size/location is invalid".into(),
        ));
    }
    if mantissa_position < offset || mantissa_end > precision_end {
        return Err(Error::InvalidFormat(
            "mantissa bit field size/location is invalid".into(),
        ));
    }
    if sign_position < offset || sign_position >= precision_end {
        return Err(Error::InvalidFormat("sign location is not valid".into()));
    }
    if sign_position >= exponent_position && sign_position < exponent_end {
        return Err(Error::InvalidFormat(
            "sign bit appears within exponent field".into(),
        ));
    }
    if sign_position >= mantissa_position && sign_position < mantissa_end {
        return Err(Error::InvalidFormat(
            "sign bit appears within mantissa field".into(),
        ));
    }
    if (mantissa_position < exponent_position && mantissa_end > exponent_position)
        || (exponent_position < mantissa_position && exponent_end > mantissa_position)
    {
        return Err(Error::InvalidFormat(
            "exponent and mantissa fields overlap".into(),
        ));
    }

    dtype.message.class_bits[1] = u8::try_from(sign_position)
        .map_err(|_| Error::InvalidFormat("sign position exceeds u8".into()))?;
    dtype.message.properties[4] = u8::try_from(exponent_position)
        .map_err(|_| Error::InvalidFormat("exponent position exceeds u8".into()))?;
    dtype.message.properties[5] = u8::try_from(exponent_size)
        .map_err(|_| Error::InvalidFormat("exponent size exceeds u8".into()))?;
    dtype.message.properties[6] = u8::try_from(mantissa_position)
        .map_err(|_| Error::InvalidFormat("mantissa position exceeds u8".into()))?;
    dtype.message.properties[7] = u8::try_from(mantissa_size)
        .map_err(|_| Error::InvalidFormat("mantissa size exceeds u8".into()))?;
    Ok(())
}
/// Set the floating-point exponent bias of a datatype.
#[allow(non_snake_case)]
pub fn H5Tset_ebias(dtype: &mut RuntimeDatatype, ebias: u32) {
    if dtype.message.properties.len() < 12 {
        dtype.message.properties.resize(12, 0);
    }
    write_le_u32_at(&mut dtype.message.properties, 8, ebias);
}

/// Internal helper `write_le_u16_at`.
fn write_le_u16_at(bytes: &mut [u8], pos: usize, value: u16) {
    if let Some(window) = pos.checked_add(2).and_then(|end| bytes.get_mut(pos..end)) {
        window.copy_from_slice(&value.to_le_bytes());
    }
}

/// Internal helper `write_le_u32_at`.
fn write_le_u32_at(bytes: &mut [u8], pos: usize, value: u32) {
    if let Some(window) = pos.checked_add(4).and_then(|end| bytes.get_mut(pos..end)) {
        window.copy_from_slice(&value.to_le_bytes());
    }
}
/// Set the floating-point normalization of a datatype.
#[allow(non_snake_case)]
pub fn H5Tset_norm(dtype: &mut RuntimeDatatype, norm: u8) -> Result<()> {
    if dtype.locked || dtype.immutable {
        return Err(Error::InvalidFormat(
            "floating-point normalization cannot be changed on a locked datatype".into(),
        ));
    }
    if dtype.message.class != DatatypeClass::FloatingPoint {
        return Err(Error::InvalidFormat(
            "operation not defined for datatype class".into(),
        ));
    }
    let encoded = match norm {
        0 => 2,
        1 => 1,
        2 => 0,
        _ => return Err(Error::InvalidFormat("illegal normalization".into())),
    };
    dtype.message.class_bits[0] = (dtype.message.class_bits[0] & !0x30) | (encoded << 4);
    Ok(())
}
/// Set the internal padding of a datatype.
#[allow(non_snake_case)]
pub fn H5Tset_inpad(dtype: &mut RuntimeDatatype, inpad: u8) -> Result<()> {
    if dtype.locked || dtype.immutable {
        return Err(Error::InvalidFormat(
            "floating-point internal padding cannot be changed on a locked datatype".into(),
        ));
    }
    if inpad >= 3 {
        return Err(Error::InvalidFormat("illegal internal pad type".into()));
    }
    if dtype.message.class != DatatypeClass::FloatingPoint {
        return Err(Error::InvalidFormat(
            "operation not defined for datatype class".into(),
        ));
    }
    if inpad == 2 {
        return Err(Error::Unsupported(
            "background internal padding is not supported in the datatype message flags".into(),
        ));
    }
    dtype.message.class_bits[0] = (dtype.message.class_bits[0] & !0x08) | (inpad << 3);
    Ok(())
}
/// Set the sign convention of a datatype.
#[allow(non_snake_case)]
pub fn H5Tset_sign(dtype: &mut RuntimeDatatype, sign: u8) -> Result<()> {
    if dtype.locked || dtype.immutable {
        return Err(Error::InvalidFormat(
            "integer sign cannot be changed on a locked datatype".into(),
        ));
    }
    if sign >= 2 {
        return Err(Error::InvalidFormat("illegal sign type".into()));
    }
    match dtype.message.class {
        DatatypeClass::FixedPoint => {
            dtype.message.class_bits[0] =
                (dtype.message.class_bits[0] & !0x08) | (if sign == 1 { 0x08 } else { 0 });
            Ok(())
        }
        DatatypeClass::Enum => {
            if enum_has_members(&dtype.message)? {
                return Err(Error::InvalidFormat(
                    "operation not allowed after members are defined".into(),
                ));
            }
            Err(Error::Unsupported(
                "setting sign on encoded enum base datatypes is unsupported".into(),
            ))
        }
        _ => Err(Error::InvalidFormat(
            "operation not defined for datatype class".into(),
        )),
    }
}
/// Set the padding of a datatype.
#[allow(non_snake_case)]
pub fn H5Tset_pad(dtype: &mut RuntimeDatatype, low: u8, high: u8) -> Result<()> {
    if dtype.locked || dtype.immutable {
        return Err(Error::InvalidFormat(
            "datatype padding cannot be changed on a locked datatype".into(),
        ));
    }
    if low >= 3 || high >= 3 {
        return Err(Error::InvalidFormat("invalid pad type".into()));
    }
    if low == 2 || high == 2 {
        return Err(Error::Unsupported(
            "background bit padding is not supported in the datatype message flags".into(),
        ));
    }
    match dtype.message.class {
        DatatypeClass::FixedPoint | DatatypeClass::BitField | DatatypeClass::FloatingPoint => {
            dtype.message.class_bits[0] =
                (dtype.message.class_bits[0] & !0x06) | (low << 1) | (high << 2);
            Ok(())
        }
        DatatypeClass::Enum => {
            if enum_has_members(&dtype.message)? {
                return Err(Error::InvalidFormat(
                    "operation not allowed after members are defined".into(),
                ));
            }
            Err(Error::Unsupported(
                "setting padding on encoded enum base datatypes is unsupported".into(),
            ))
        }
        _ => Err(Error::Unsupported(
            "operation not defined for specified data type".into(),
        )),
    }
}

/// Return a deep copy of a datatype.
#[allow(non_snake_case)]
pub fn H5T__bit_copy(
    dst: &mut [u8],
    dst_offset: usize,
    src: &[u8],
    src_offset: usize,
    size: usize,
) -> Result<()> {
    let src_end = src_offset
        .checked_add(size)
        .ok_or_else(|| Error::InvalidFormat("source bit range overflows".into()))?;
    let dst_end = dst_offset
        .checked_add(size)
        .ok_or_else(|| Error::InvalidFormat("destination bit range overflows".into()))?;
    if src_end > src.len().saturating_mul(8) || dst_end > dst.len().saturating_mul(8) {
        return Err(Error::InvalidFormat("bit copy range exceeds buffer".into()));
    }

    for bit in 0..size {
        let source_bit = src_offset + bit;
        let destination_bit = dst_offset + bit;
        let source_value = ((src[source_bit / 8] >> (source_bit % 8)) & 1) != 0;
        let byte = &mut dst[destination_bit / 8];
        let mask = 1u8 << (destination_bit % 8);
        if source_value {
            *byte |= mask;
        } else {
            *byte &= !mask;
        }
    }
    Ok(())
}
/// Datatype operation: bit shift.
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
/// Datatype operation: bit set.
#[allow(non_snake_case)]
pub fn H5T__bit_set(data: &mut [u8], offset: usize, mut size: usize, value: bool) {
    let mut idx = offset / 8;
    let mut bit_offset = offset % 8;

    if size > 0 && bit_offset != 0 {
        let nbits = size.min(8 - bit_offset);
        let mask = (((1u16 << nbits) - 1) as u8) << bit_offset;
        if let Some(byte) = data.get_mut(idx) {
            if value {
                *byte |= mask;
            } else {
                *byte &= !mask;
            }
        }
        idx += 1;
        size -= nbits;
        bit_offset = 0;
    }

    while size >= 8 {
        if let Some(byte) = data.get_mut(idx) {
            *byte = if value { 0xff } else { 0x00 };
        }
        idx += 1;
        size -= 8;
    }

    if size > 0 && bit_offset == 0 {
        let mask = ((1u16 << size) - 1) as u8;
        if let Some(byte) = data.get_mut(idx) {
            if value {
                *byte |= mask;
            } else {
                *byte &= !mask;
            }
        }
    }
}
/// Find an entry in a datatype.
#[allow(non_snake_case)]
pub fn H5T__bit_find(
    data: &[u8],
    offset: usize,
    mut size: usize,
    direction: H5TBitSearchDirection,
    value: bool,
) -> Option<usize> {
    match direction {
        H5TBitSearchDirection::Lsb => {
            let mut idx = offset / 8;
            let mut bit_offset = offset % 8;

            if bit_offset != 0 {
                while bit_offset < 8 && size > 0 {
                    let bit = data.get(idx).map(|byte| ((byte >> bit_offset) & 1) != 0)?;
                    if bit == value {
                        return Some(8 * idx + bit_offset - offset);
                    }
                    bit_offset += 1;
                    size -= 1;
                }
                idx += 1;
            }

            while size >= 8 {
                let byte = *data.get(idx)?;
                if byte != if value { 0x00 } else { 0xff } {
                    for bit in 0..8 {
                        if ((byte >> bit) & 1 != 0) == value {
                            return Some(8 * idx + bit - offset);
                        }
                    }
                }
                idx += 1;
                size -= 8;
            }

            if size > 0 {
                let byte = *data.get(idx)?;
                for bit in 0..size {
                    if ((byte >> bit) & 1 != 0) == value {
                        return Some(8 * idx + bit - offset);
                    }
                }
            }
            None
        }
        H5TBitSearchDirection::Msb => {
            if size == 0 {
                return None;
            }
            let mut idx = (offset + size - 1) / 8;
            let bit_offset = offset % 8;

            if size > 8 - bit_offset && (offset + size) % 8 != 0 {
                let byte = *data.get(idx)?;
                let mut bit = (offset + size) % 8;
                while bit > 0 {
                    bit -= 1;
                    if ((byte >> bit) & 1 != 0) == value {
                        return Some(8 * idx + bit - offset);
                    }
                    size -= 1;
                }
                idx = idx.checked_sub(1)?;
            }

            while size >= 8 {
                let byte = *data.get(idx)?;
                if byte != if value { 0x00 } else { 0xff } {
                    for bit in (0..8).rev() {
                        if ((byte >> bit) & 1 != 0) == value {
                            return Some(8 * idx + bit - offset);
                        }
                    }
                }
                size -= 8;
                if size >= 8 || size > 0 {
                    idx = idx.checked_sub(1)?;
                }
            }

            if size > 0 {
                let byte = *data.get(idx)?;
                for bit in (bit_offset..(bit_offset + size)).rev() {
                    if ((byte >> bit) & 1 != 0) == value {
                        return Some(8 * idx + bit - offset);
                    }
                }
            }
            None
        }
    }
}
/// Datatype operation: bit inc.
#[allow(non_snake_case)]
pub fn H5T__bit_inc(data: &mut [u8], start: usize, size: usize) -> Result<bool> {
    let end = start
        .checked_add(size)
        .ok_or_else(|| Error::InvalidFormat("bit increment range overflows".into()))?;
    if size == 0 || end > data.len().saturating_mul(8) {
        return Err(Error::InvalidFormat(
            "bit increment range exceeds buffer".into(),
        ));
    }

    for bit in start..end {
        let byte = &mut data[bit / 8];
        let mask = 1u8 << (bit % 8);
        if (*byte & mask) == 0 {
            *byte |= mask;
            return Ok(false);
        }
        *byte &= !mask;
    }
    Ok(true)
}
/// Datatype operation: bit dec.
#[allow(non_snake_case)]
pub fn H5T__bit_dec(data: &mut [u8], start: usize, size: usize) -> Result<bool> {
    let end = start
        .checked_add(size)
        .ok_or_else(|| Error::InvalidFormat("bit decrement range overflows".into()))?;
    if size == 0 || end > data.len().saturating_mul(8) {
        return Err(Error::InvalidFormat(
            "bit decrement range exceeds buffer".into(),
        ));
    }

    for bit in start..end {
        let byte = &mut data[bit / 8];
        let mask = 1u8 << (bit % 8);
        if (*byte & mask) != 0 {
            *byte &= !mask;
            return Ok(false);
        }
        *byte |= mask;
    }
    Ok(true)
}
/// Datatype operation: bit neg.
#[allow(non_snake_case)]
pub fn H5T__bit_neg(data: &mut [u8], start: usize, size: usize) -> Result<()> {
    let end = start
        .checked_add(size)
        .ok_or_else(|| Error::InvalidFormat("bit negation range overflows".into()))?;
    if size == 0 || end > data.len().saturating_mul(8) {
        return Err(Error::InvalidFormat(
            "bit negation range exceeds buffer".into(),
        ));
    }

    for bit in start..end {
        data[bit / 8] ^= 1u8 << (bit % 8);
    }
    Ok(())
}
/// Datatype operation: bit cmp.
#[allow(non_snake_case)]
pub fn H5T__bit_cmp(
    nbytes: usize,
    perm: &[usize],
    a: &[u8],
    b: &[u8],
    pad_mask: &[u8],
) -> Result<usize> {
    if nbytes > perm.len() || nbytes > a.len() || nbytes > b.len() || nbytes > pad_mask.len() {
        return Err(Error::InvalidFormat(
            "bit comparison range exceeds buffer".into(),
        ));
    }

    for i in 0..nbytes {
        if perm[i] >= nbytes {
            return Err(Error::InvalidFormat("failure in bit comparison".into()));
        }

        let mut aa = a[perm[i]] & pad_mask[perm[i]];
        let mut bb = b[perm[i]] & pad_mask[perm[i]];
        if aa != bb {
            for j in 0..8 {
                if (aa & 1) != (bb & 1) {
                    return Ok(i * 8 + j);
                }
                aa >>= 1;
                bb >>= 1;
            }
        }
    }

    Err(Error::InvalidFormat(
        "didn't find a value for `first`".into(),
    ))
}
/// Datatype operation: byte cmp.
#[allow(non_snake_case)]
pub fn H5T__byte_cmp(left: &[u8], right: &[u8]) -> std::cmp::Ordering {
    left.cmp(right)
}
/// Datatype operation: fix order.
#[allow(non_snake_case)]
pub fn H5T__fix_order(n: usize, last: usize, perm: &mut [usize]) -> Result<H5TDetectedOrder> {
    if last == 0 || n > perm.len() || last >= perm.len() {
        return Err(Error::InvalidFormat("failed to detect byte order".into()));
    }

    if perm[last] < perm[last - 1] && (last < 2 || perm[last - 1] < perm[last - 2]) {
        for (i, value) in perm.iter_mut().enumerate().take(n) {
            *value = i;
        }
        Ok(H5TDetectedOrder::LittleEndian)
    } else if perm[last] > perm[last - 1] && (last < 2 || perm[last - 1] > perm[last - 2]) {
        for (i, value) in perm.iter_mut().enumerate().take(n) {
            *value = (n - 1) - i;
        }
        Ok(H5TDetectedOrder::BigEndian)
    } else {
        if n % 2 != 0 {
            return Err(Error::InvalidFormat("n is not a power of 2".into()));
        }
        for i in (0..n).step_by(2) {
            perm[i] = (n - 2) - i;
            perm[i + 1] = (n - 1) - i;
        }
        Ok(H5TDetectedOrder::Vax)
    }
}
/// Datatype operation: imp bit.
#[allow(non_snake_case)]
pub fn H5T__imp_bit(data: &[u8]) -> bool {
    data.iter().any(|byte| *byte != 0)
}

/// Datatype operation: conv fcomplex schar.
#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex_schar_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv fcomplex schar.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex_schar(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_fcomplex_schar_into(data, &mut out);
    out
}
/// Datatype operation: conv fcomplex uchar.
#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex_uchar_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv fcomplex uchar.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex_uchar(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_fcomplex_uchar_into(data, &mut out);
    out
}
/// Datatype operation: conv fcomplex short.
#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex_short_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv fcomplex short.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex_short(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_fcomplex_short_into(data, &mut out);
    out
}
/// Datatype operation: conv fcomplex ushort.
#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex_ushort_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv fcomplex ushort.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex_ushort(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_fcomplex_ushort_into(data, &mut out);
    out
}
/// Datatype operation: conv fcomplex int.
#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex_int_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv fcomplex int.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex_int(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_fcomplex_int_into(data, &mut out);
    out
}
/// Datatype operation: conv fcomplex uint.
#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex_uint_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv fcomplex uint.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex_uint(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_fcomplex_uint_into(data, &mut out);
    out
}
/// Datatype operation: conv fcomplex long.
#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex_long_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv fcomplex long.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex_long(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_fcomplex_long_into(data, &mut out);
    out
}
/// Datatype operation: conv fcomplex ulong.
#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex_ulong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv fcomplex ulong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex_ulong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_fcomplex_ulong_into(data, &mut out);
    out
}
/// Datatype operation: conv fcomplex llong.
#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex_llong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv fcomplex llong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex_llong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_fcomplex_llong_into(data, &mut out);
    out
}
/// Datatype operation: conv fcomplex ullong.
#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex_ullong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv fcomplex ullong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex_ullong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_fcomplex_ullong_into(data, &mut out);
    out
}
/// Datatype operation: conv fcomplex  Float16.
#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex__Float16_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv fcomplex  Float16.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex__Float16(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_fcomplex__Float16_into(data, &mut out);
    out
}
/// Datatype operation: conv fcomplex float.
#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex_float_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv fcomplex float.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex_float(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_fcomplex_float_into(data, &mut out);
    out
}
/// Datatype operation: conv fcomplex double.
#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex_double_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv fcomplex double.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex_double(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_fcomplex_double_into(data, &mut out);
    out
}
/// Datatype operation: conv fcomplex ldouble.
#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex_ldouble_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv fcomplex ldouble.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex_ldouble(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_fcomplex_ldouble_into(data, &mut out);
    out
}
/// Datatype operation: conv fcomplex dcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex_dcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv fcomplex dcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex_dcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_fcomplex_dcomplex_into(data, &mut out);
    out
}
/// Datatype operation: conv fcomplex lcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex_lcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv fcomplex lcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_fcomplex_lcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_fcomplex_lcomplex_into(data, &mut out);
    out
}
/// Datatype operation: conv dcomplex schar.
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex_schar_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv dcomplex schar.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex_schar(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_dcomplex_schar_into(data, &mut out);
    out
}
/// Datatype operation: conv dcomplex uchar.
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex_uchar_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv dcomplex uchar.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex_uchar(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_dcomplex_uchar_into(data, &mut out);
    out
}
/// Datatype operation: conv dcomplex short.
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex_short_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv dcomplex short.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex_short(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_dcomplex_short_into(data, &mut out);
    out
}
/// Datatype operation: conv dcomplex ushort.
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex_ushort_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv dcomplex ushort.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex_ushort(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_dcomplex_ushort_into(data, &mut out);
    out
}
/// Datatype operation: conv dcomplex int.
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex_int_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv dcomplex int.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex_int(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_dcomplex_int_into(data, &mut out);
    out
}
/// Datatype operation: conv dcomplex uint.
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex_uint_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv dcomplex uint.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex_uint(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_dcomplex_uint_into(data, &mut out);
    out
}
/// Datatype operation: conv dcomplex long.
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex_long_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv dcomplex long.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex_long(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_dcomplex_long_into(data, &mut out);
    out
}
/// Datatype operation: conv dcomplex ulong.
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex_ulong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv dcomplex ulong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex_ulong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_dcomplex_ulong_into(data, &mut out);
    out
}
/// Datatype operation: conv dcomplex llong.
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex_llong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv dcomplex llong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex_llong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_dcomplex_llong_into(data, &mut out);
    out
}
/// Datatype operation: conv dcomplex ullong.
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex_ullong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv dcomplex ullong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex_ullong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_dcomplex_ullong_into(data, &mut out);
    out
}
/// Datatype operation: conv dcomplex  Float16.
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex__Float16_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv dcomplex  Float16.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex__Float16(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_dcomplex__Float16_into(data, &mut out);
    out
}
/// Datatype operation: conv dcomplex float.
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex_float_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv dcomplex float.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex_float(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_dcomplex_float_into(data, &mut out);
    out
}
/// Datatype operation: conv dcomplex double.
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex_double_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv dcomplex double.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex_double(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_dcomplex_double_into(data, &mut out);
    out
}
/// Datatype operation: conv dcomplex ldouble.
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex_ldouble_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv dcomplex ldouble.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex_ldouble(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_dcomplex_ldouble_into(data, &mut out);
    out
}
/// Datatype operation: conv dcomplex fcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex_fcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv dcomplex fcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex_fcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_dcomplex_fcomplex_into(data, &mut out);
    out
}
/// Datatype operation: conv dcomplex lcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex_lcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv dcomplex lcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_dcomplex_lcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_dcomplex_lcomplex_into(data, &mut out);
    out
}
/// Datatype operation: conv lcomplex schar.
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex_schar_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv lcomplex schar.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex_schar(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_lcomplex_schar_into(data, &mut out);
    out
}
/// Datatype operation: conv lcomplex uchar.
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex_uchar_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv lcomplex uchar.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex_uchar(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_lcomplex_uchar_into(data, &mut out);
    out
}
/// Datatype operation: conv lcomplex short.
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex_short_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv lcomplex short.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex_short(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_lcomplex_short_into(data, &mut out);
    out
}
/// Datatype operation: conv lcomplex ushort.
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex_ushort_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv lcomplex ushort.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex_ushort(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_lcomplex_ushort_into(data, &mut out);
    out
}
/// Datatype operation: conv lcomplex int.
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex_int_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv lcomplex int.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex_int(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_lcomplex_int_into(data, &mut out);
    out
}
/// Datatype operation: conv lcomplex uint.
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex_uint_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv lcomplex uint.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex_uint(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_lcomplex_uint_into(data, &mut out);
    out
}
/// Datatype operation: conv lcomplex long.
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex_long_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv lcomplex long.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex_long(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_lcomplex_long_into(data, &mut out);
    out
}
/// Datatype operation: conv lcomplex ulong.
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex_ulong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv lcomplex ulong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex_ulong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_lcomplex_ulong_into(data, &mut out);
    out
}
/// Datatype operation: conv lcomplex llong.
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex_llong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv lcomplex llong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex_llong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_lcomplex_llong_into(data, &mut out);
    out
}
/// Datatype operation: conv lcomplex ullong.
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex_ullong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv lcomplex ullong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex_ullong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_lcomplex_ullong_into(data, &mut out);
    out
}
/// Datatype operation: conv lcomplex  Float16.
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex__Float16_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv lcomplex  Float16.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex__Float16(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_lcomplex__Float16_into(data, &mut out);
    out
}
/// Datatype operation: conv lcomplex float.
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex_float_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv lcomplex float.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex_float(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_lcomplex_float_into(data, &mut out);
    out
}
/// Datatype operation: conv lcomplex double.
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex_double_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv lcomplex double.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex_double(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_lcomplex_double_into(data, &mut out);
    out
}
/// Datatype operation: conv lcomplex ldouble.
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex_ldouble_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv lcomplex ldouble.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex_ldouble(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_lcomplex_ldouble_into(data, &mut out);
    out
}
/// Datatype operation: conv lcomplex fcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex_fcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv lcomplex fcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex_fcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_lcomplex_fcomplex_into(data, &mut out);
    out
}
/// Datatype operation: conv lcomplex dcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex_dcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv lcomplex dcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_lcomplex_dcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_lcomplex_dcomplex_into(data, &mut out);
    out
}

/// Initialize the datatype subsystem.
#[allow(non_snake_case)]
pub fn H5T__init_native_internal() -> DatatypeRegistry {
    DatatypeRegistry::default()
}

/// Initialize the datatype subsystem.
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

/// Datatype operation: ref set loc.
#[allow(non_snake_case)]
pub fn H5T__ref_set_loc(dtype: &mut RuntimeDatatype, loc: impl Into<String>) {
    H5T_set_loc(dtype, loc);
}

const H5R_ENCODE_HEADER_SIZE: usize = 2;
const H5R_IS_EXTERNAL: u8 = 0x01;
const H5R_OBJECT2: u8 = 2;
const H5R_MAXTYPE: u8 = 5;
const H5O_DTYPE_VER_BOUNDS: [u8; 7] = [1, 3, 3, 4, 4, 5, 5];

/// Datatype operation: ref mem isnull.
#[allow(non_snake_case)]
pub fn H5T__ref_mem_isnull(buf: &[u8]) -> bool {
    buf.iter().all(|byte| *byte == 0)
}

/// Datatype operation: ref mem setnull.
#[allow(non_snake_case)]
pub fn H5T__ref_mem_setnull(buf: &mut [u8]) {
    buf.fill(0);
}

/// Datatype operation: ref mem getsize.
#[allow(non_snake_case)]
pub fn H5T__ref_mem_getsize(buf: &[u8]) -> usize {
    buf.len()
}

/// Read from a datatype.
#[allow(non_snake_case)]
pub fn H5T__ref_mem_read_ref(buf: &[u8]) -> &[u8] {
    buf
}

/// Read from a datatype.
#[deprecated(note = "use H5T__ref_mem_read_ref() to borrow the reference bytes")]
#[allow(non_snake_case)]
pub fn H5T__ref_mem_read(buf: &[u8]) -> Vec<u8> {
    buf.to_vec()
}

/// Write to a datatype.
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

/// Datatype operation: ref disk isnull.
#[allow(non_snake_case)]
pub fn H5T__ref_disk_isnull(buf: &[u8]) -> Result<bool> {
    if buf.is_empty() {
        return Err(Error::InvalidFormat("reference buffer is empty".into()));
    }

    let ref_type = buf[0];
    if ref_type != 0 {
        return Ok(false);
    }

    let blob_offset = H5R_ENCODE_HEADER_SIZE + std::mem::size_of::<u32>();
    if buf.len() < blob_offset {
        return Err(Error::InvalidFormat(
            "reference disk buffer is truncated".into(),
        ));
    }
    Ok(buf[blob_offset..].iter().all(|byte| *byte == 0))
}

/// Datatype operation: ref disk setnull.
#[allow(non_snake_case)]
pub fn H5T__ref_disk_setnull(buf: &mut [u8], background: Option<&mut [u8]>) -> Result<()> {
    let blob_offset = H5R_ENCODE_HEADER_SIZE + std::mem::size_of::<u32>();
    if buf.len() < blob_offset {
        return Err(Error::InvalidFormat(
            "reference destination buffer is truncated".into(),
        ));
    }
    if let Some(bg) = background {
        if bg.len() < blob_offset {
            return Err(Error::InvalidFormat(
                "reference background buffer is truncated".into(),
            ));
        }
        bg[blob_offset..].fill(0);
    }

    buf[..H5R_ENCODE_HEADER_SIZE].fill(0);
    buf[H5R_ENCODE_HEADER_SIZE..blob_offset].fill(0);
    buf[blob_offset..].fill(0);
    Ok(())
}

/// Datatype operation: ref disk getsize.
#[allow(non_snake_case)]
pub fn H5T__ref_disk_getsize(buf: &[u8]) -> Result<(usize, bool)> {
    if buf.len() < H5R_ENCODE_HEADER_SIZE {
        return Err(Error::InvalidFormat(
            "reference disk buffer is truncated".into(),
        ));
    }

    let ref_type = buf[0];
    if ref_type >= H5R_MAXTYPE {
        return Err(Error::InvalidFormat("invalid reference type".into()));
    }
    let flags = buf[1];
    if (flags & H5R_IS_EXTERNAL) == 0 && ref_type == H5R_OBJECT2 {
        return Ok((buf.len(), true));
    }

    let blob_offset = H5R_ENCODE_HEADER_SIZE + std::mem::size_of::<u32>();
    if buf.len() < blob_offset {
        return Err(Error::InvalidFormat(
            "reference disk buffer is truncated".into(),
        ));
    }
    let payload_size = u32::from_le_bytes(
        buf[H5R_ENCODE_HEADER_SIZE..blob_offset]
            .try_into()
            .expect("slice length checked"),
    ) as usize;
    Ok((payload_size + H5R_ENCODE_HEADER_SIZE, false))
}

/// Read from a datatype.
#[allow(non_snake_case)]
pub fn H5T__ref_disk_read_into(buf: &[u8], dst: &mut Vec<u8>) -> Result<()> {
    let blob_offset = H5R_ENCODE_HEADER_SIZE + std::mem::size_of::<u32>();
    if buf.len() < blob_offset {
        return Err(Error::InvalidFormat(
            "reference disk buffer is truncated".into(),
        ));
    }

    let payload_size = u32::from_le_bytes(
        buf[H5R_ENCODE_HEADER_SIZE..blob_offset]
            .try_into()
            .expect("slice length checked"),
    ) as usize;
    let end = blob_offset
        .checked_add(payload_size)
        .ok_or_else(|| Error::InvalidFormat("reference payload size overflows".into()))?;
    if end > buf.len() {
        return Err(Error::InvalidFormat(
            "reference payload is truncated".into(),
        ));
    }

    dst.clear();
    dst.extend_from_slice(&buf[..H5R_ENCODE_HEADER_SIZE]);
    dst.extend_from_slice(&buf[blob_offset..end]);
    Ok(())
}

/// Read from a datatype.
#[deprecated(note = "use H5T__ref_disk_read_into() to reuse caller-provided output storage")]
#[allow(non_snake_case)]
pub fn H5T__ref_disk_read(buf: &[u8]) -> Result<Vec<u8>> {
    let mut dst = Vec::new();
    H5T__ref_disk_read_into(buf, &mut dst)?;
    Ok(dst)
}

/// Write to a datatype.
#[allow(non_snake_case)]
pub fn H5T__ref_disk_write(
    dst: &mut Vec<u8>,
    src: &[u8],
    background: Option<&mut Vec<u8>>,
) -> Result<()> {
    if src.len() < H5R_ENCODE_HEADER_SIZE {
        return Err(Error::InvalidFormat(
            "reference source buffer is truncated".into(),
        ));
    }
    if let Some(bg) = background {
        bg.clear();
    }

    let payload_size = src.len() - H5R_ENCODE_HEADER_SIZE;
    if payload_size > u32::MAX as usize {
        return Err(Error::InvalidFormat(
            "reference payload is too large".into(),
        ));
    }

    dst.clear();
    dst.extend_from_slice(&src[..H5R_ENCODE_HEADER_SIZE]);
    dst.extend_from_slice(&(payload_size as u32).to_le_bytes());
    dst.extend_from_slice(&src[H5R_ENCODE_HEADER_SIZE..]);
    Ok(())
}

/// Datatype operation: ref obj disk isnull.
#[allow(non_snake_case)]
pub fn H5T__ref_obj_disk_isnull(buf: &[u8], address_size: usize) -> Result<bool> {
    if address_size == 0 || buf.len() < address_size {
        return Err(Error::InvalidFormat(
            "object reference disk buffer is truncated".into(),
        ));
    }
    Ok(buf[..address_size].iter().all(|byte| *byte == 0))
}

/// Datatype operation: ref obj disk getsize.
#[allow(non_snake_case)]
pub fn H5T__ref_obj_disk_getsize(buf: &[u8], address_size: usize) -> Result<usize> {
    if address_size == 0 || buf.len() != address_size {
        return Err(Error::InvalidFormat(
            "object reference disk size does not match file address size".into(),
        ));
    }
    Ok(address_size)
}

/// Read from a datatype.
#[allow(non_snake_case)]
pub fn H5T__ref_obj_disk_read_into(
    buf: &[u8],
    address_size: usize,
    dst: &mut Vec<u8>,
) -> Result<()> {
    if address_size == 0 || buf.len() < address_size {
        return Err(Error::InvalidFormat(
            "object reference disk buffer is truncated".into(),
        ));
    }
    dst.clear();
    dst.extend_from_slice(&buf[..address_size]);
    Ok(())
}

/// Read from a datatype.
#[deprecated(note = "use H5T__ref_obj_disk_read_into() to reuse caller-provided output storage")]
#[allow(non_snake_case)]
pub fn H5T__ref_obj_disk_read(buf: &[u8], address_size: usize) -> Result<Vec<u8>> {
    let mut dst = Vec::with_capacity(address_size);
    H5T__ref_obj_disk_read_into(buf, address_size, &mut dst)?;
    Ok(dst)
}

/// Datatype operation: ref dsetreg disk isnull.
#[allow(non_snake_case)]
pub fn H5T__ref_dsetreg_disk_isnull(buf: &[u8], address_size: usize) -> Result<bool> {
    if address_size == 0 || buf.len() < address_size {
        return Err(Error::InvalidFormat(
            "dataset-region reference disk buffer is truncated".into(),
        ));
    }
    Ok(buf[..address_size].iter().all(|byte| *byte == 0))
}

/// Datatype operation: ref dsetreg disk getsize.
#[allow(non_snake_case)]
pub fn H5T__ref_dsetreg_disk_getsize(buf: &[u8], address_size: usize) -> Result<usize> {
    if address_size == 0 || buf.len() < address_size {
        return Err(Error::InvalidFormat(
            "dataset-region reference disk buffer is truncated".into(),
        ));
    }
    Ok(address_size + std::mem::size_of::<usize>())
}

/// Read from a datatype.
#[allow(non_snake_case)]
pub fn H5T__ref_dsetreg_disk_read_into(
    buf: &[u8],
    address_size: usize,
    token: &mut Vec<u8>,
    encoded_space: &mut Vec<u8>,
) -> Result<()> {
    if address_size == 0 || buf.len() < address_size {
        return Err(Error::InvalidFormat(
            "dataset-region reference disk buffer is truncated".into(),
        ));
    }
    token.clear();
    token.extend_from_slice(&buf[..address_size]);
    encoded_space.clear();
    encoded_space.extend_from_slice(&buf[address_size..]);
    Ok(())
}

/// Read from a datatype.
#[deprecated(
    note = "use H5T__ref_dsetreg_disk_read_into() to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__ref_dsetreg_disk_read(buf: &[u8], address_size: usize) -> Result<(Vec<u8>, Vec<u8>)> {
    let mut token = Vec::with_capacity(address_size);
    let mut encoded_space = Vec::new();
    H5T__ref_dsetreg_disk_read_into(buf, address_size, &mut token, &mut encoded_space)?;
    Ok((token, encoded_space))
}

/// Datatype operation: ref reclaim.
#[allow(non_snake_case)]
pub fn H5T__ref_reclaim(buf: &mut Vec<u8>) {
    buf.clear();
}

/// Datatype operation: vlen set loc.
#[allow(non_snake_case)]
pub fn H5T__vlen_set_loc(dtype: &mut RuntimeDatatype, loc: impl Into<String>) {
    H5T_set_loc(dtype, loc);
}

/// Datatype operation: vlen mem seq getlen.
#[allow(non_snake_case)]
pub fn H5T__vlen_mem_seq_getlen(seq: &[u8]) -> usize {
    seq.len()
}

/// Datatype operation: vlen mem seq isnull.
#[allow(non_snake_case)]
pub fn H5T__vlen_mem_seq_isnull(seq: &[u8]) -> bool {
    seq.is_empty()
}

/// Datatype operation: vlen mem seq setnull.
#[allow(non_snake_case)]
pub fn H5T__vlen_mem_seq_setnull(seq: &mut Vec<u8>) {
    seq.clear();
}

/// Write to a datatype.
#[allow(non_snake_case)]
pub fn H5T__vlen_mem_seq_write(dst: &mut Vec<u8>, src: &[u8]) {
    dst.clear();
    dst.extend_from_slice(src);
}

/// Datatype operation: vlen mem str getlen.
#[allow(non_snake_case)]
pub fn H5T__vlen_mem_str_getlen(value: &str) -> usize {
    value.len()
}

/// Datatype operation: vlen mem str getptr.
#[allow(non_snake_case)]
pub fn H5T__vlen_mem_str_getptr(value: &str) -> *const u8 {
    value.as_ptr()
}

/// Datatype operation: vlen mem str isnull.
#[allow(non_snake_case)]
pub fn H5T__vlen_mem_str_isnull(value: &str) -> bool {
    value.is_empty()
}

/// Datatype operation: vlen mem str setnull.
#[allow(non_snake_case)]
pub fn H5T__vlen_mem_str_setnull(value: &mut String) {
    value.clear();
}

/// Read from a datatype.
#[allow(non_snake_case)]
pub fn H5T__vlen_mem_str_read_ref(value: &str) -> &[u8] {
    value.as_bytes()
}

/// Read from a datatype.
#[deprecated(note = "use H5T__vlen_mem_str_read_ref() to borrow the string bytes")]
#[allow(non_snake_case)]
pub fn H5T__vlen_mem_str_read(value: &str) -> Vec<u8> {
    value.as_bytes().to_vec()
}

/// Write to a datatype.
#[allow(non_snake_case)]
pub fn H5T__vlen_mem_str_write(value: &mut String, bytes: &[u8]) -> Result<()> {
    *value = std::str::from_utf8(bytes)
        .map_err(|_| Error::InvalidFormat("vlen memory string is not UTF-8".into()))?
        .trim_end_matches('\0')
        .to_string();
    Ok(())
}

/// Datatype operation: vlen disk getlen.
#[allow(non_snake_case)]
pub fn H5T__vlen_disk_getlen(buf: &[u8]) -> Result<usize> {
    if buf.len() < std::mem::size_of::<u32>() {
        return Err(Error::InvalidFormat("vlen disk buffer is truncated".into()));
    }
    Ok(u32::from_le_bytes(
        buf[..std::mem::size_of::<u32>()]
            .try_into()
            .expect("slice length checked"),
    ) as usize)
}

/// Datatype operation: vlen disk isnull.
#[allow(non_snake_case)]
pub fn H5T__vlen_disk_isnull(buf: &[u8]) -> Result<bool> {
    if buf.len() < std::mem::size_of::<u32>() {
        return Err(Error::InvalidFormat("vlen disk buffer is truncated".into()));
    }
    Ok(buf[std::mem::size_of::<u32>()..]
        .iter()
        .all(|byte| *byte == 0))
}

/// Datatype operation: vlen disk setnull.
#[allow(non_snake_case)]
pub fn H5T__vlen_disk_setnull(buf: &mut Vec<u8>, background: Option<&mut Vec<u8>>) -> Result<()> {
    if let Some(bg) = background {
        H5T__vlen_disk_delete(bg)?;
    }
    buf.clear();
    buf.extend_from_slice(&0u32.to_le_bytes());
    Ok(())
}

/// Read from a datatype.
#[allow(non_snake_case)]
pub fn H5T__vlen_disk_read_ref(buf: &[u8], len: usize) -> Result<&[u8]> {
    let blob_offset = std::mem::size_of::<u32>();
    let end = blob_offset
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat("vlen disk read size overflows".into()))?;
    if end > buf.len() {
        return Err(Error::InvalidFormat("vlen disk blob is truncated".into()));
    }
    Ok(&buf[blob_offset..end])
}

/// Read from a datatype.
#[deprecated(note = "use H5T__vlen_disk_read_ref() to borrow the vlen blob bytes")]
#[allow(non_snake_case)]
pub fn H5T__vlen_disk_read(buf: &[u8], len: usize) -> Result<Vec<u8>> {
    H5T__vlen_disk_read_ref(buf, len).map(<[u8]>::to_vec)
}

/// Write to a datatype.
#[allow(non_snake_case)]
pub fn H5T__vlen_disk_write(
    dst: &mut Vec<u8>,
    src: &[u8],
    background: Option<&mut Vec<u8>>,
    seq_len: usize,
    base_size: usize,
) -> Result<()> {
    if let Some(bg) = background {
        H5T__vlen_disk_delete(bg)?;
    }
    let blob_size = seq_len
        .checked_mul(base_size)
        .ok_or_else(|| Error::InvalidFormat("vlen disk blob size overflows".into()))?;
    if seq_len > u32::MAX as usize {
        return Err(Error::InvalidFormat(
            "vlen sequence length too large".into(),
        ));
    }
    if blob_size > src.len() {
        return Err(Error::InvalidFormat("vlen source blob is truncated".into()));
    }

    dst.clear();
    dst.extend_from_slice(&(seq_len as u32).to_le_bytes());
    dst.extend_from_slice(&src[..blob_size]);
    Ok(())
}

/// Delete a datatype.
#[allow(non_snake_case)]
pub fn H5T__vlen_disk_delete(buf: &mut Vec<u8>) -> Result<()> {
    buf.clear();
    Ok(())
}

/// Datatype operation: vlen reclaim.
#[allow(non_snake_case)]
pub fn H5T__vlen_reclaim(buf: &mut Vec<u8>) {
    buf.clear();
}

/// Datatype operation: vlen reclaim elmt.
#[allow(non_snake_case)]
pub fn H5T_vlen_reclaim_elmt(buf: &mut Vec<u8>) {
    H5T__vlen_reclaim(buf);
}

/// Datatype operation: conv schar uchar.
#[allow(non_snake_case)]
pub fn H5T__conv_schar_uchar_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv schar uchar.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_schar_uchar(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_schar_uchar_into(data, &mut out);
    out
}

/// Datatype operation: conv schar short.
#[allow(non_snake_case)]
pub fn H5T__conv_schar_short_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv schar short.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_schar_short(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_schar_short_into(data, &mut out);
    out
}

/// Datatype operation: conv schar ushort.
#[allow(non_snake_case)]
pub fn H5T__conv_schar_ushort_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv schar ushort.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_schar_ushort(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_schar_ushort_into(data, &mut out);
    out
}

/// Datatype operation: conv schar int.
#[allow(non_snake_case)]
pub fn H5T__conv_schar_int_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv schar int.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_schar_int(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_schar_int_into(data, &mut out);
    out
}

/// Datatype operation: conv schar uint.
#[allow(non_snake_case)]
pub fn H5T__conv_schar_uint_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv schar uint.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_schar_uint(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_schar_uint_into(data, &mut out);
    out
}

/// Datatype operation: conv schar long.
#[allow(non_snake_case)]
pub fn H5T__conv_schar_long_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv schar long.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_schar_long(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_schar_long_into(data, &mut out);
    out
}

/// Datatype operation: conv schar ulong.
#[allow(non_snake_case)]
pub fn H5T__conv_schar_ulong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv schar ulong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_schar_ulong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_schar_ulong_into(data, &mut out);
    out
}

/// Datatype operation: conv schar llong.
#[allow(non_snake_case)]
pub fn H5T__conv_schar_llong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv schar llong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_schar_llong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_schar_llong_into(data, &mut out);
    out
}

/// Datatype operation: conv schar ullong.
#[allow(non_snake_case)]
pub fn H5T__conv_schar_ullong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv schar ullong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_schar_ullong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_schar_ullong_into(data, &mut out);
    out
}

/// Datatype operation: conv schar  Float16.
#[allow(non_snake_case)]
pub fn H5T__conv_schar__Float16_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv schar  Float16.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_schar__Float16(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_schar__Float16_into(data, &mut out);
    out
}

/// Datatype operation: conv schar float.
#[allow(non_snake_case)]
pub fn H5T__conv_schar_float_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv schar float.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_schar_float(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_schar_float_into(data, &mut out);
    out
}

/// Datatype operation: conv schar double.
#[allow(non_snake_case)]
pub fn H5T__conv_schar_double_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv schar double.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_schar_double(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_schar_double_into(data, &mut out);
    out
}

/// Datatype operation: conv schar ldouble.
#[allow(non_snake_case)]
pub fn H5T__conv_schar_ldouble_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv schar ldouble.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_schar_ldouble(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_schar_ldouble_into(data, &mut out);
    out
}

/// Datatype operation: conv schar fcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv_schar_fcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv schar fcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_schar_fcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_schar_fcomplex_into(data, &mut out);
    out
}

/// Datatype operation: conv schar dcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv_schar_dcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv schar dcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_schar_dcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_schar_dcomplex_into(data, &mut out);
    out
}

/// Datatype operation: conv schar lcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv_schar_lcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv schar lcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_schar_lcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_schar_lcomplex_into(data, &mut out);
    out
}

/// Datatype operation: conv uchar schar.
#[allow(non_snake_case)]
pub fn H5T__conv_uchar_schar_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv uchar schar.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_uchar_schar(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_uchar_schar_into(data, &mut out);
    out
}

/// Datatype operation: conv uchar short.
#[allow(non_snake_case)]
pub fn H5T__conv_uchar_short_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv uchar short.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_uchar_short(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_uchar_short_into(data, &mut out);
    out
}

/// Datatype operation: conv uchar ushort.
#[allow(non_snake_case)]
pub fn H5T__conv_uchar_ushort_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv uchar ushort.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_uchar_ushort(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_uchar_ushort_into(data, &mut out);
    out
}

/// Datatype operation: conv uchar int.
#[allow(non_snake_case)]
pub fn H5T__conv_uchar_int_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv uchar int.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_uchar_int(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_uchar_int_into(data, &mut out);
    out
}

/// Datatype operation: conv uchar uint.
#[allow(non_snake_case)]
pub fn H5T__conv_uchar_uint_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv uchar uint.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_uchar_uint(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_uchar_uint_into(data, &mut out);
    out
}

/// Datatype operation: conv uchar long.
#[allow(non_snake_case)]
pub fn H5T__conv_uchar_long_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv uchar long.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_uchar_long(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_uchar_long_into(data, &mut out);
    out
}

/// Datatype operation: conv uchar ulong.
#[allow(non_snake_case)]
pub fn H5T__conv_uchar_ulong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv uchar ulong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_uchar_ulong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_uchar_ulong_into(data, &mut out);
    out
}

/// Datatype operation: conv uchar llong.
#[allow(non_snake_case)]
pub fn H5T__conv_uchar_llong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv uchar llong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_uchar_llong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_uchar_llong_into(data, &mut out);
    out
}

/// Datatype operation: conv uchar ullong.
#[allow(non_snake_case)]
pub fn H5T__conv_uchar_ullong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv uchar ullong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_uchar_ullong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_uchar_ullong_into(data, &mut out);
    out
}

/// Datatype operation: conv uchar  Float16.
#[allow(non_snake_case)]
pub fn H5T__conv_uchar__Float16_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv uchar  Float16.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_uchar__Float16(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_uchar__Float16_into(data, &mut out);
    out
}

/// Datatype operation: conv uchar float.
#[allow(non_snake_case)]
pub fn H5T__conv_uchar_float_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv uchar float.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_uchar_float(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_uchar_float_into(data, &mut out);
    out
}

/// Datatype operation: conv uchar double.
#[allow(non_snake_case)]
pub fn H5T__conv_uchar_double_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv uchar double.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_uchar_double(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_uchar_double_into(data, &mut out);
    out
}

/// Datatype operation: conv uchar ldouble.
#[allow(non_snake_case)]
pub fn H5T__conv_uchar_ldouble_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv uchar ldouble.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_uchar_ldouble(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_uchar_ldouble_into(data, &mut out);
    out
}

/// Datatype operation: conv uchar fcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv_uchar_fcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv uchar fcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_uchar_fcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_uchar_fcomplex_into(data, &mut out);
    out
}

/// Datatype operation: conv uchar dcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv_uchar_dcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv uchar dcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_uchar_dcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_uchar_dcomplex_into(data, &mut out);
    out
}

/// Datatype operation: conv uchar lcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv_uchar_lcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv uchar lcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_uchar_lcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_uchar_lcomplex_into(data, &mut out);
    out
}

/// Datatype operation: conv short schar.
#[allow(non_snake_case)]
pub fn H5T__conv_short_schar_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv short schar.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_short_schar(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_short_schar_into(data, &mut out);
    out
}

/// Datatype operation: conv short uchar.
#[allow(non_snake_case)]
pub fn H5T__conv_short_uchar_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv short uchar.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_short_uchar(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_short_uchar_into(data, &mut out);
    out
}

/// Datatype operation: conv short ushort.
#[allow(non_snake_case)]
pub fn H5T__conv_short_ushort_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv short ushort.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_short_ushort(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_short_ushort_into(data, &mut out);
    out
}

/// Datatype operation: conv short int.
#[allow(non_snake_case)]
pub fn H5T__conv_short_int_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv short int.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_short_int(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_short_int_into(data, &mut out);
    out
}

/// Datatype operation: conv short uint.
#[allow(non_snake_case)]
pub fn H5T__conv_short_uint_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv short uint.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_short_uint(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_short_uint_into(data, &mut out);
    out
}

/// Datatype operation: conv short long.
#[allow(non_snake_case)]
pub fn H5T__conv_short_long_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv short long.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_short_long(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_short_long_into(data, &mut out);
    out
}

/// Datatype operation: conv short ulong.
#[allow(non_snake_case)]
pub fn H5T__conv_short_ulong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv short ulong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_short_ulong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_short_ulong_into(data, &mut out);
    out
}

/// Datatype operation: conv short llong.
#[allow(non_snake_case)]
pub fn H5T__conv_short_llong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv short llong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_short_llong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_short_llong_into(data, &mut out);
    out
}

/// Datatype operation: conv short ullong.
#[allow(non_snake_case)]
pub fn H5T__conv_short_ullong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv short ullong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_short_ullong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_short_ullong_into(data, &mut out);
    out
}

/// Datatype operation: conv short  Float16.
#[allow(non_snake_case)]
pub fn H5T__conv_short__Float16_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv short  Float16.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_short__Float16(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_short__Float16_into(data, &mut out);
    out
}

/// Datatype operation: conv short float.
#[allow(non_snake_case)]
pub fn H5T__conv_short_float_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv short float.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_short_float(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_short_float_into(data, &mut out);
    out
}

/// Datatype operation: conv short double.
#[allow(non_snake_case)]
pub fn H5T__conv_short_double_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv short double.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_short_double(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_short_double_into(data, &mut out);
    out
}

/// Datatype operation: conv short ldouble.
#[allow(non_snake_case)]
pub fn H5T__conv_short_ldouble_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv short ldouble.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_short_ldouble(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_short_ldouble_into(data, &mut out);
    out
}

/// Datatype operation: conv short fcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv_short_fcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv short fcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_short_fcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_short_fcomplex_into(data, &mut out);
    out
}

/// Datatype operation: conv short dcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv_short_dcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv short dcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_short_dcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_short_dcomplex_into(data, &mut out);
    out
}

/// Datatype operation: conv short lcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv_short_lcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv short lcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_short_lcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_short_lcomplex_into(data, &mut out);
    out
}

/// Datatype operation: conv ushort schar.
#[allow(non_snake_case)]
pub fn H5T__conv_ushort_schar_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ushort schar.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ushort_schar(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ushort_schar_into(data, &mut out);
    out
}

/// Datatype operation: conv ushort uchar.
#[allow(non_snake_case)]
pub fn H5T__conv_ushort_uchar_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ushort uchar.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ushort_uchar(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ushort_uchar_into(data, &mut out);
    out
}

/// Datatype operation: conv ushort short.
#[allow(non_snake_case)]
pub fn H5T__conv_ushort_short_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ushort short.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ushort_short(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ushort_short_into(data, &mut out);
    out
}

/// Datatype operation: conv ushort int.
#[allow(non_snake_case)]
pub fn H5T__conv_ushort_int_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ushort int.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ushort_int(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ushort_int_into(data, &mut out);
    out
}

/// Datatype operation: conv ushort uint.
#[allow(non_snake_case)]
pub fn H5T__conv_ushort_uint_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ushort uint.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ushort_uint(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ushort_uint_into(data, &mut out);
    out
}

/// Datatype operation: conv ushort long.
#[allow(non_snake_case)]
pub fn H5T__conv_ushort_long_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ushort long.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ushort_long(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ushort_long_into(data, &mut out);
    out
}

/// Datatype operation: conv ushort ulong.
#[allow(non_snake_case)]
pub fn H5T__conv_ushort_ulong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ushort ulong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ushort_ulong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ushort_ulong_into(data, &mut out);
    out
}

/// Datatype operation: conv ushort llong.
#[allow(non_snake_case)]
pub fn H5T__conv_ushort_llong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ushort llong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ushort_llong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ushort_llong_into(data, &mut out);
    out
}

/// Datatype operation: conv ushort ullong.
#[allow(non_snake_case)]
pub fn H5T__conv_ushort_ullong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ushort ullong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ushort_ullong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ushort_ullong_into(data, &mut out);
    out
}

/// Datatype operation: conv ushort  Float16.
#[allow(non_snake_case)]
pub fn H5T__conv_ushort__Float16_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ushort  Float16.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ushort__Float16(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ushort__Float16_into(data, &mut out);
    out
}

/// Datatype operation: conv ushort float.
#[allow(non_snake_case)]
pub fn H5T__conv_ushort_float_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ushort float.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ushort_float(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ushort_float_into(data, &mut out);
    out
}

/// Datatype operation: conv ushort double.
#[allow(non_snake_case)]
pub fn H5T__conv_ushort_double_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ushort double.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ushort_double(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ushort_double_into(data, &mut out);
    out
}

/// Datatype operation: conv ushort ldouble.
#[allow(non_snake_case)]
pub fn H5T__conv_ushort_ldouble_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ushort ldouble.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ushort_ldouble(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ushort_ldouble_into(data, &mut out);
    out
}

/// Datatype operation: conv ushort fcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv_ushort_fcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ushort fcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ushort_fcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ushort_fcomplex_into(data, &mut out);
    out
}

/// Datatype operation: conv ushort dcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv_ushort_dcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ushort dcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ushort_dcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ushort_dcomplex_into(data, &mut out);
    out
}

/// Datatype operation: conv ushort lcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv_ushort_lcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ushort lcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ushort_lcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ushort_lcomplex_into(data, &mut out);
    out
}

/// Datatype operation: conv int schar.
#[allow(non_snake_case)]
pub fn H5T__conv_int_schar_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv int schar.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_int_schar(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_int_schar_into(data, &mut out);
    out
}

/// Datatype operation: conv int uchar.
#[allow(non_snake_case)]
pub fn H5T__conv_int_uchar_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv int uchar.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_int_uchar(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_int_uchar_into(data, &mut out);
    out
}

/// Datatype operation: conv int short.
#[allow(non_snake_case)]
pub fn H5T__conv_int_short_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv int short.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_int_short(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_int_short_into(data, &mut out);
    out
}

/// Datatype operation: conv int ushort.
#[allow(non_snake_case)]
pub fn H5T__conv_int_ushort_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv int ushort.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_int_ushort(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_int_ushort_into(data, &mut out);
    out
}

/// Datatype operation: conv int uint.
#[allow(non_snake_case)]
pub fn H5T__conv_int_uint_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv int uint.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_int_uint(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_int_uint_into(data, &mut out);
    out
}

/// Datatype operation: conv int long.
#[allow(non_snake_case)]
pub fn H5T__conv_int_long_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv int long.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_int_long(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_int_long_into(data, &mut out);
    out
}

/// Datatype operation: conv int ulong.
#[allow(non_snake_case)]
pub fn H5T__conv_int_ulong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv int ulong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_int_ulong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_int_ulong_into(data, &mut out);
    out
}

/// Datatype operation: conv int llong.
#[allow(non_snake_case)]
pub fn H5T__conv_int_llong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv int llong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_int_llong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_int_llong_into(data, &mut out);
    out
}

/// Datatype operation: conv int ullong.
#[allow(non_snake_case)]
pub fn H5T__conv_int_ullong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv int ullong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_int_ullong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_int_ullong_into(data, &mut out);
    out
}

/// Datatype operation: conv int  Float16.
#[allow(non_snake_case)]
pub fn H5T__conv_int__Float16_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv int  Float16.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_int__Float16(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_int__Float16_into(data, &mut out);
    out
}

/// Datatype operation: conv int float.
#[allow(non_snake_case)]
pub fn H5T__conv_int_float_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv int float.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_int_float(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_int_float_into(data, &mut out);
    out
}

/// Datatype operation: conv int double.
#[allow(non_snake_case)]
pub fn H5T__conv_int_double_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv int double.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_int_double(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_int_double_into(data, &mut out);
    out
}

/// Datatype operation: conv int ldouble.
#[allow(non_snake_case)]
pub fn H5T__conv_int_ldouble_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv int ldouble.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_int_ldouble(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_int_ldouble_into(data, &mut out);
    out
}

/// Datatype operation: conv int fcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv_int_fcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv int fcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_int_fcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_int_fcomplex_into(data, &mut out);
    out
}

/// Datatype operation: conv int dcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv_int_dcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv int dcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_int_dcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_int_dcomplex_into(data, &mut out);
    out
}

/// Datatype operation: conv int lcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv_int_lcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv int lcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_int_lcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_int_lcomplex_into(data, &mut out);
    out
}

/// Datatype operation: conv uint schar.
#[allow(non_snake_case)]
pub fn H5T__conv_uint_schar_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv uint schar.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_uint_schar(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_uint_schar_into(data, &mut out);
    out
}

/// Datatype operation: conv uint uchar.
#[allow(non_snake_case)]
pub fn H5T__conv_uint_uchar_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv uint uchar.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_uint_uchar(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_uint_uchar_into(data, &mut out);
    out
}

/// Datatype operation: conv uint short.
#[allow(non_snake_case)]
pub fn H5T__conv_uint_short_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv uint short.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_uint_short(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_uint_short_into(data, &mut out);
    out
}

/// Datatype operation: conv uint ushort.
#[allow(non_snake_case)]
pub fn H5T__conv_uint_ushort_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv uint ushort.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_uint_ushort(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_uint_ushort_into(data, &mut out);
    out
}

/// Datatype operation: conv uint int.
#[allow(non_snake_case)]
pub fn H5T__conv_uint_int_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv uint int.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_uint_int(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_uint_int_into(data, &mut out);
    out
}

/// Datatype operation: conv uint long.
#[allow(non_snake_case)]
pub fn H5T__conv_uint_long_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv uint long.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_uint_long(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_uint_long_into(data, &mut out);
    out
}

/// Datatype operation: conv uint ulong.
#[allow(non_snake_case)]
pub fn H5T__conv_uint_ulong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv uint ulong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_uint_ulong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_uint_ulong_into(data, &mut out);
    out
}

/// Datatype operation: conv uint llong.
#[allow(non_snake_case)]
pub fn H5T__conv_uint_llong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv uint llong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_uint_llong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_uint_llong_into(data, &mut out);
    out
}

/// Datatype operation: conv uint ullong.
#[allow(non_snake_case)]
pub fn H5T__conv_uint_ullong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv uint ullong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_uint_ullong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_uint_ullong_into(data, &mut out);
    out
}

/// Datatype operation: conv uint  Float16.
#[allow(non_snake_case)]
pub fn H5T__conv_uint__Float16_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv uint  Float16.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_uint__Float16(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_uint__Float16_into(data, &mut out);
    out
}

/// Datatype operation: conv uint float.
#[allow(non_snake_case)]
pub fn H5T__conv_uint_float_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv uint float.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_uint_float(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_uint_float_into(data, &mut out);
    out
}

/// Datatype operation: conv uint double.
#[allow(non_snake_case)]
pub fn H5T__conv_uint_double_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv uint double.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_uint_double(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_uint_double_into(data, &mut out);
    out
}

/// Datatype operation: conv uint ldouble.
#[allow(non_snake_case)]
pub fn H5T__conv_uint_ldouble_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv uint ldouble.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_uint_ldouble(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_uint_ldouble_into(data, &mut out);
    out
}

/// Datatype operation: conv uint fcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv_uint_fcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv uint fcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_uint_fcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_uint_fcomplex_into(data, &mut out);
    out
}

/// Datatype operation: conv uint dcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv_uint_dcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv uint dcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_uint_dcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_uint_dcomplex_into(data, &mut out);
    out
}

/// Datatype operation: conv uint lcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv_uint_lcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv uint lcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_uint_lcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_uint_lcomplex_into(data, &mut out);
    out
}

/// Datatype operation: conv long schar.
#[allow(non_snake_case)]
pub fn H5T__conv_long_schar_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv long schar.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_long_schar(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_long_schar_into(data, &mut out);
    out
}

/// Datatype operation: conv long uchar.
#[allow(non_snake_case)]
pub fn H5T__conv_long_uchar_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv long uchar.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_long_uchar(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_long_uchar_into(data, &mut out);
    out
}

/// Datatype operation: conv long short.
#[allow(non_snake_case)]
pub fn H5T__conv_long_short_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv long short.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_long_short(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_long_short_into(data, &mut out);
    out
}

/// Datatype operation: conv long ushort.
#[allow(non_snake_case)]
pub fn H5T__conv_long_ushort_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv long ushort.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_long_ushort(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_long_ushort_into(data, &mut out);
    out
}

/// Datatype operation: conv long int.
#[allow(non_snake_case)]
pub fn H5T__conv_long_int_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv long int.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_long_int(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_long_int_into(data, &mut out);
    out
}

/// Datatype operation: conv long uint.
#[allow(non_snake_case)]
pub fn H5T__conv_long_uint_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv long uint.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_long_uint(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_long_uint_into(data, &mut out);
    out
}

/// Datatype operation: conv long ulong.
#[allow(non_snake_case)]
pub fn H5T__conv_long_ulong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv long ulong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_long_ulong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_long_ulong_into(data, &mut out);
    out
}

/// Datatype operation: conv long llong.
#[allow(non_snake_case)]
pub fn H5T__conv_long_llong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv long llong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_long_llong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_long_llong_into(data, &mut out);
    out
}

/// Datatype operation: conv long ullong.
#[allow(non_snake_case)]
pub fn H5T__conv_long_ullong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv long ullong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_long_ullong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_long_ullong_into(data, &mut out);
    out
}

/// Datatype operation: conv long  Float16.
#[allow(non_snake_case)]
pub fn H5T__conv_long__Float16_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv long  Float16.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_long__Float16(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_long__Float16_into(data, &mut out);
    out
}

/// Datatype operation: conv long float.
#[allow(non_snake_case)]
pub fn H5T__conv_long_float_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv long float.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_long_float(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_long_float_into(data, &mut out);
    out
}

/// Datatype operation: conv long double.
#[allow(non_snake_case)]
pub fn H5T__conv_long_double_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv long double.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_long_double(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_long_double_into(data, &mut out);
    out
}

/// Datatype operation: conv long ldouble.
#[allow(non_snake_case)]
pub fn H5T__conv_long_ldouble_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv long ldouble.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_long_ldouble(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_long_ldouble_into(data, &mut out);
    out
}

/// Datatype operation: conv long fcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv_long_fcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv long fcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_long_fcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_long_fcomplex_into(data, &mut out);
    out
}

/// Datatype operation: conv long dcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv_long_dcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv long dcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_long_dcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_long_dcomplex_into(data, &mut out);
    out
}

/// Datatype operation: conv long lcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv_long_lcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv long lcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_long_lcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_long_lcomplex_into(data, &mut out);
    out
}

/// Datatype operation: conv ulong schar.
#[allow(non_snake_case)]
pub fn H5T__conv_ulong_schar_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ulong schar.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ulong_schar(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ulong_schar_into(data, &mut out);
    out
}

/// Datatype operation: conv ulong uchar.
#[allow(non_snake_case)]
pub fn H5T__conv_ulong_uchar_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ulong uchar.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ulong_uchar(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ulong_uchar_into(data, &mut out);
    out
}

/// Datatype operation: conv ulong short.
#[allow(non_snake_case)]
pub fn H5T__conv_ulong_short_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ulong short.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ulong_short(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ulong_short_into(data, &mut out);
    out
}

/// Datatype operation: conv ulong ushort.
#[allow(non_snake_case)]
pub fn H5T__conv_ulong_ushort_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ulong ushort.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ulong_ushort(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ulong_ushort_into(data, &mut out);
    out
}

/// Datatype operation: conv ulong int.
#[allow(non_snake_case)]
pub fn H5T__conv_ulong_int_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ulong int.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ulong_int(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ulong_int_into(data, &mut out);
    out
}

/// Datatype operation: conv ulong uint.
#[allow(non_snake_case)]
pub fn H5T__conv_ulong_uint_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ulong uint.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ulong_uint(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ulong_uint_into(data, &mut out);
    out
}

/// Datatype operation: conv ulong long.
#[allow(non_snake_case)]
pub fn H5T__conv_ulong_long_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ulong long.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ulong_long(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ulong_long_into(data, &mut out);
    out
}

/// Datatype operation: conv ulong llong.
#[allow(non_snake_case)]
pub fn H5T__conv_ulong_llong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ulong llong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ulong_llong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ulong_llong_into(data, &mut out);
    out
}

/// Datatype operation: conv ulong ullong.
#[allow(non_snake_case)]
pub fn H5T__conv_ulong_ullong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ulong ullong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ulong_ullong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ulong_ullong_into(data, &mut out);
    out
}

/// Datatype operation: conv ulong  Float16.
#[allow(non_snake_case)]
pub fn H5T__conv_ulong__Float16_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ulong  Float16.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ulong__Float16(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ulong__Float16_into(data, &mut out);
    out
}

/// Datatype operation: conv ulong float.
#[allow(non_snake_case)]
pub fn H5T__conv_ulong_float_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ulong float.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ulong_float(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ulong_float_into(data, &mut out);
    out
}

/// Datatype operation: conv ulong double.
#[allow(non_snake_case)]
pub fn H5T__conv_ulong_double_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ulong double.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ulong_double(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ulong_double_into(data, &mut out);
    out
}

/// Datatype operation: conv ulong ldouble.
#[allow(non_snake_case)]
pub fn H5T__conv_ulong_ldouble_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ulong ldouble.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ulong_ldouble(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ulong_ldouble_into(data, &mut out);
    out
}

/// Datatype operation: conv ulong fcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv_ulong_fcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ulong fcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ulong_fcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ulong_fcomplex_into(data, &mut out);
    out
}

/// Datatype operation: conv ulong dcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv_ulong_dcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ulong dcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ulong_dcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ulong_dcomplex_into(data, &mut out);
    out
}

/// Datatype operation: conv ulong lcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv_ulong_lcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ulong lcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ulong_lcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ulong_lcomplex_into(data, &mut out);
    out
}

/// Datatype operation: conv llong schar.
#[allow(non_snake_case)]
pub fn H5T__conv_llong_schar_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv llong schar.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_llong_schar(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_llong_schar_into(data, &mut out);
    out
}

/// Datatype operation: conv llong uchar.
#[allow(non_snake_case)]
pub fn H5T__conv_llong_uchar_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv llong uchar.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_llong_uchar(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_llong_uchar_into(data, &mut out);
    out
}

/// Datatype operation: conv llong short.
#[allow(non_snake_case)]
pub fn H5T__conv_llong_short_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv llong short.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_llong_short(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_llong_short_into(data, &mut out);
    out
}

/// Datatype operation: conv llong ushort.
#[allow(non_snake_case)]
pub fn H5T__conv_llong_ushort_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv llong ushort.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_llong_ushort(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_llong_ushort_into(data, &mut out);
    out
}

/// Datatype operation: conv llong int.
#[allow(non_snake_case)]
pub fn H5T__conv_llong_int_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv llong int.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_llong_int(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_llong_int_into(data, &mut out);
    out
}

/// Datatype operation: conv llong uint.
#[allow(non_snake_case)]
pub fn H5T__conv_llong_uint_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv llong uint.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_llong_uint(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_llong_uint_into(data, &mut out);
    out
}

/// Datatype operation: conv llong long.
#[allow(non_snake_case)]
pub fn H5T__conv_llong_long_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv llong long.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_llong_long(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_llong_long_into(data, &mut out);
    out
}

/// Datatype operation: conv llong ulong.
#[allow(non_snake_case)]
pub fn H5T__conv_llong_ulong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv llong ulong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_llong_ulong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_llong_ulong_into(data, &mut out);
    out
}

/// Datatype operation: conv llong ullong.
#[allow(non_snake_case)]
pub fn H5T__conv_llong_ullong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv llong ullong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_llong_ullong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_llong_ullong_into(data, &mut out);
    out
}

/// Datatype operation: conv llong  Float16.
#[allow(non_snake_case)]
pub fn H5T__conv_llong__Float16_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv llong  Float16.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_llong__Float16(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_llong__Float16_into(data, &mut out);
    out
}

/// Datatype operation: conv llong float.
#[allow(non_snake_case)]
pub fn H5T__conv_llong_float_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv llong float.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_llong_float(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_llong_float_into(data, &mut out);
    out
}

/// Datatype operation: conv llong double.
#[allow(non_snake_case)]
pub fn H5T__conv_llong_double_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv llong double.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_llong_double(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_llong_double_into(data, &mut out);
    out
}

/// Datatype operation: conv llong ldouble.
#[allow(non_snake_case)]
pub fn H5T__conv_llong_ldouble_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv llong ldouble.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_llong_ldouble(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_llong_ldouble_into(data, &mut out);
    out
}

/// Datatype operation: conv llong fcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv_llong_fcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv llong fcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_llong_fcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_llong_fcomplex_into(data, &mut out);
    out
}

/// Datatype operation: conv llong dcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv_llong_dcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv llong dcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_llong_dcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_llong_dcomplex_into(data, &mut out);
    out
}

/// Datatype operation: conv llong lcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv_llong_lcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv llong lcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_llong_lcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_llong_lcomplex_into(data, &mut out);
    out
}

/// Datatype operation: conv ullong schar.
#[allow(non_snake_case)]
pub fn H5T__conv_ullong_schar_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ullong schar.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ullong_schar(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ullong_schar_into(data, &mut out);
    out
}

/// Datatype operation: conv ullong uchar.
#[allow(non_snake_case)]
pub fn H5T__conv_ullong_uchar_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ullong uchar.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ullong_uchar(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ullong_uchar_into(data, &mut out);
    out
}

/// Datatype operation: conv ullong short.
#[allow(non_snake_case)]
pub fn H5T__conv_ullong_short_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ullong short.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ullong_short(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ullong_short_into(data, &mut out);
    out
}

/// Datatype operation: conv ullong ushort.
#[allow(non_snake_case)]
pub fn H5T__conv_ullong_ushort_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ullong ushort.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ullong_ushort(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ullong_ushort_into(data, &mut out);
    out
}

/// Datatype operation: conv ullong int.
#[allow(non_snake_case)]
pub fn H5T__conv_ullong_int_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ullong int.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ullong_int(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ullong_int_into(data, &mut out);
    out
}

/// Datatype operation: conv ullong uint.
#[allow(non_snake_case)]
pub fn H5T__conv_ullong_uint_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ullong uint.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ullong_uint(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ullong_uint_into(data, &mut out);
    out
}

/// Datatype operation: conv ullong long.
#[allow(non_snake_case)]
pub fn H5T__conv_ullong_long_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ullong long.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ullong_long(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ullong_long_into(data, &mut out);
    out
}

/// Datatype operation: conv ullong ulong.
#[allow(non_snake_case)]
pub fn H5T__conv_ullong_ulong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ullong ulong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ullong_ulong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ullong_ulong_into(data, &mut out);
    out
}

/// Datatype operation: conv ullong llong.
#[allow(non_snake_case)]
pub fn H5T__conv_ullong_llong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ullong llong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ullong_llong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ullong_llong_into(data, &mut out);
    out
}

/// Datatype operation: conv ullong  Float16.
#[allow(non_snake_case)]
pub fn H5T__conv_ullong__Float16_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ullong  Float16.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ullong__Float16(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ullong__Float16_into(data, &mut out);
    out
}

/// Datatype operation: conv ullong float.
#[allow(non_snake_case)]
pub fn H5T__conv_ullong_float_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ullong float.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ullong_float(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ullong_float_into(data, &mut out);
    out
}

/// Datatype operation: conv ullong double.
#[allow(non_snake_case)]
pub fn H5T__conv_ullong_double_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ullong double.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ullong_double(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ullong_double_into(data, &mut out);
    out
}

/// Datatype operation: conv ullong ldouble.
#[allow(non_snake_case)]
pub fn H5T__conv_ullong_ldouble_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ullong ldouble.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ullong_ldouble(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ullong_ldouble_into(data, &mut out);
    out
}

/// Datatype operation: conv ullong fcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv_ullong_fcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ullong fcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ullong_fcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ullong_fcomplex_into(data, &mut out);
    out
}

/// Datatype operation: conv ullong dcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv_ullong_dcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ullong dcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ullong_dcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ullong_dcomplex_into(data, &mut out);
    out
}

/// Datatype operation: conv ullong lcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv_ullong_lcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ullong lcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ullong_lcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ullong_lcomplex_into(data, &mut out);
    out
}

/// Datatype operation: conv  Float16 schar.
#[allow(non_snake_case)]
pub fn H5T__conv__Float16_schar_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv  Float16 schar.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv__Float16_schar(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv__Float16_schar_into(data, &mut out);
    out
}

/// Datatype operation: conv  Float16 uchar.
#[allow(non_snake_case)]
pub fn H5T__conv__Float16_uchar_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv  Float16 uchar.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv__Float16_uchar(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv__Float16_uchar_into(data, &mut out);
    out
}

/// Datatype operation: conv  Float16 short.
#[allow(non_snake_case)]
pub fn H5T__conv__Float16_short_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv  Float16 short.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv__Float16_short(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv__Float16_short_into(data, &mut out);
    out
}

/// Datatype operation: conv  Float16 ushort.
#[allow(non_snake_case)]
pub fn H5T__conv__Float16_ushort_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv  Float16 ushort.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv__Float16_ushort(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv__Float16_ushort_into(data, &mut out);
    out
}

/// Datatype operation: conv  Float16 int.
#[allow(non_snake_case)]
pub fn H5T__conv__Float16_int_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv  Float16 int.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv__Float16_int(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv__Float16_int_into(data, &mut out);
    out
}

/// Datatype operation: conv  Float16 uint.
#[allow(non_snake_case)]
pub fn H5T__conv__Float16_uint_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv  Float16 uint.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv__Float16_uint(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv__Float16_uint_into(data, &mut out);
    out
}

/// Datatype operation: conv  Float16 long.
#[allow(non_snake_case)]
pub fn H5T__conv__Float16_long_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv  Float16 long.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv__Float16_long(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv__Float16_long_into(data, &mut out);
    out
}

/// Datatype operation: conv  Float16 ulong.
#[allow(non_snake_case)]
pub fn H5T__conv__Float16_ulong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv  Float16 ulong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv__Float16_ulong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv__Float16_ulong_into(data, &mut out);
    out
}

/// Datatype operation: conv  Float16 llong.
#[allow(non_snake_case)]
pub fn H5T__conv__Float16_llong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv  Float16 llong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv__Float16_llong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv__Float16_llong_into(data, &mut out);
    out
}

/// Datatype operation: conv  Float16 ullong.
#[allow(non_snake_case)]
pub fn H5T__conv__Float16_ullong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv  Float16 ullong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv__Float16_ullong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv__Float16_ullong_into(data, &mut out);
    out
}

/// Datatype operation: conv  Float16 float.
#[allow(non_snake_case)]
pub fn H5T__conv__Float16_float_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv  Float16 float.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv__Float16_float(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv__Float16_float_into(data, &mut out);
    out
}

/// Datatype operation: conv  Float16 double.
#[allow(non_snake_case)]
pub fn H5T__conv__Float16_double_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv  Float16 double.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv__Float16_double(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv__Float16_double_into(data, &mut out);
    out
}

/// Datatype operation: conv  Float16 ldouble.
#[allow(non_snake_case)]
pub fn H5T__conv__Float16_ldouble_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv  Float16 ldouble.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv__Float16_ldouble(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv__Float16_ldouble_into(data, &mut out);
    out
}

/// Datatype operation: conv  Float16 fcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv__Float16_fcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv  Float16 fcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv__Float16_fcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv__Float16_fcomplex_into(data, &mut out);
    out
}

/// Datatype operation: conv  Float16 dcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv__Float16_dcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv  Float16 dcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv__Float16_dcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv__Float16_dcomplex_into(data, &mut out);
    out
}

/// Datatype operation: conv  Float16 lcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv__Float16_lcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv  Float16 lcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv__Float16_lcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv__Float16_lcomplex_into(data, &mut out);
    out
}

/// Datatype operation: conv float schar.
#[allow(non_snake_case)]
pub fn H5T__conv_float_schar_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv float schar.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_float_schar(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_float_schar_into(data, &mut out);
    out
}

/// Datatype operation: conv float uchar.
#[allow(non_snake_case)]
pub fn H5T__conv_float_uchar_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv float uchar.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_float_uchar(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_float_uchar_into(data, &mut out);
    out
}

/// Datatype operation: conv float short.
#[allow(non_snake_case)]
pub fn H5T__conv_float_short_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv float short.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_float_short(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_float_short_into(data, &mut out);
    out
}

/// Datatype operation: conv float ushort.
#[allow(non_snake_case)]
pub fn H5T__conv_float_ushort_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv float ushort.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_float_ushort(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_float_ushort_into(data, &mut out);
    out
}

/// Datatype operation: conv float int.
#[allow(non_snake_case)]
pub fn H5T__conv_float_int_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv float int.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_float_int(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_float_int_into(data, &mut out);
    out
}

/// Datatype operation: conv float uint.
#[allow(non_snake_case)]
pub fn H5T__conv_float_uint_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv float uint.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_float_uint(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_float_uint_into(data, &mut out);
    out
}

/// Datatype operation: conv float long.
#[allow(non_snake_case)]
pub fn H5T__conv_float_long_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv float long.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_float_long(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_float_long_into(data, &mut out);
    out
}

/// Datatype operation: conv float ulong.
#[allow(non_snake_case)]
pub fn H5T__conv_float_ulong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv float ulong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_float_ulong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_float_ulong_into(data, &mut out);
    out
}

/// Datatype operation: conv float llong.
#[allow(non_snake_case)]
pub fn H5T__conv_float_llong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv float llong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_float_llong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_float_llong_into(data, &mut out);
    out
}

/// Datatype operation: conv float ullong.
#[allow(non_snake_case)]
pub fn H5T__conv_float_ullong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv float ullong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_float_ullong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_float_ullong_into(data, &mut out);
    out
}

/// Datatype operation: conv float  Float16.
#[allow(non_snake_case)]
pub fn H5T__conv_float__Float16_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv float  Float16.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_float__Float16(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_float__Float16_into(data, &mut out);
    out
}

/// Datatype operation: conv float double.
#[allow(non_snake_case)]
pub fn H5T__conv_float_double_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv float double.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_float_double(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_float_double_into(data, &mut out);
    out
}

/// Datatype operation: conv float ldouble.
#[allow(non_snake_case)]
pub fn H5T__conv_float_ldouble_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv float ldouble.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_float_ldouble(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_float_ldouble_into(data, &mut out);
    out
}

/// Datatype operation: conv float fcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv_float_fcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv float fcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_float_fcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_float_fcomplex_into(data, &mut out);
    out
}

/// Datatype operation: conv float dcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv_float_dcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv float dcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_float_dcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_float_dcomplex_into(data, &mut out);
    out
}

/// Datatype operation: conv float lcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv_float_lcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv float lcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_float_lcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_float_lcomplex_into(data, &mut out);
    out
}

/// Datatype operation: conv double schar.
#[allow(non_snake_case)]
pub fn H5T__conv_double_schar_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv double schar.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_double_schar(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_double_schar_into(data, &mut out);
    out
}

/// Datatype operation: conv double uchar.
#[allow(non_snake_case)]
pub fn H5T__conv_double_uchar_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv double uchar.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_double_uchar(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_double_uchar_into(data, &mut out);
    out
}

/// Datatype operation: conv double short.
#[allow(non_snake_case)]
pub fn H5T__conv_double_short_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv double short.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_double_short(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_double_short_into(data, &mut out);
    out
}

/// Datatype operation: conv double ushort.
#[allow(non_snake_case)]
pub fn H5T__conv_double_ushort_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv double ushort.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_double_ushort(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_double_ushort_into(data, &mut out);
    out
}

/// Datatype operation: conv double int.
#[allow(non_snake_case)]
pub fn H5T__conv_double_int_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv double int.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_double_int(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_double_int_into(data, &mut out);
    out
}

/// Datatype operation: conv double uint.
#[allow(non_snake_case)]
pub fn H5T__conv_double_uint_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv double uint.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_double_uint(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_double_uint_into(data, &mut out);
    out
}

/// Datatype operation: conv double long.
#[allow(non_snake_case)]
pub fn H5T__conv_double_long_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv double long.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_double_long(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_double_long_into(data, &mut out);
    out
}

/// Datatype operation: conv double ulong.
#[allow(non_snake_case)]
pub fn H5T__conv_double_ulong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv double ulong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_double_ulong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_double_ulong_into(data, &mut out);
    out
}

/// Datatype operation: conv double llong.
#[allow(non_snake_case)]
pub fn H5T__conv_double_llong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv double llong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_double_llong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_double_llong_into(data, &mut out);
    out
}

/// Datatype operation: conv double ullong.
#[allow(non_snake_case)]
pub fn H5T__conv_double_ullong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv double ullong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_double_ullong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_double_ullong_into(data, &mut out);
    out
}

/// Datatype operation: conv double  Float16.
#[allow(non_snake_case)]
pub fn H5T__conv_double__Float16_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv double  Float16.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_double__Float16(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_double__Float16_into(data, &mut out);
    out
}

/// Datatype operation: conv double float.
#[allow(non_snake_case)]
pub fn H5T__conv_double_float_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv double float.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_double_float(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_double_float_into(data, &mut out);
    out
}

/// Datatype operation: conv double ldouble.
#[allow(non_snake_case)]
pub fn H5T__conv_double_ldouble_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv double ldouble.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_double_ldouble(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_double_ldouble_into(data, &mut out);
    out
}

/// Datatype operation: conv double fcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv_double_fcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv double fcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_double_fcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_double_fcomplex_into(data, &mut out);
    out
}

/// Datatype operation: conv double dcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv_double_dcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv double dcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_double_dcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_double_dcomplex_into(data, &mut out);
    out
}

/// Datatype operation: conv double lcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv_double_lcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv double lcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_double_lcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_double_lcomplex_into(data, &mut out);
    out
}

/// Datatype operation: conv ldouble schar.
#[allow(non_snake_case)]
pub fn H5T__conv_ldouble_schar_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ldouble schar.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ldouble_schar(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ldouble_schar_into(data, &mut out);
    out
}

/// Datatype operation: conv ldouble uchar.
#[allow(non_snake_case)]
pub fn H5T__conv_ldouble_uchar_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ldouble uchar.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ldouble_uchar(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ldouble_uchar_into(data, &mut out);
    out
}

/// Datatype operation: conv ldouble short.
#[allow(non_snake_case)]
pub fn H5T__conv_ldouble_short_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ldouble short.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ldouble_short(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ldouble_short_into(data, &mut out);
    out
}

/// Datatype operation: conv ldouble ushort.
#[allow(non_snake_case)]
pub fn H5T__conv_ldouble_ushort_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ldouble ushort.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ldouble_ushort(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ldouble_ushort_into(data, &mut out);
    out
}

/// Datatype operation: conv ldouble int.
#[allow(non_snake_case)]
pub fn H5T__conv_ldouble_int_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ldouble int.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ldouble_int(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ldouble_int_into(data, &mut out);
    out
}

/// Datatype operation: conv ldouble uint.
#[allow(non_snake_case)]
pub fn H5T__conv_ldouble_uint_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ldouble uint.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ldouble_uint(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ldouble_uint_into(data, &mut out);
    out
}

/// Datatype operation: conv ldouble long.
#[allow(non_snake_case)]
pub fn H5T__conv_ldouble_long_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ldouble long.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ldouble_long(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ldouble_long_into(data, &mut out);
    out
}

/// Datatype operation: conv ldouble ulong.
#[allow(non_snake_case)]
pub fn H5T__conv_ldouble_ulong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ldouble ulong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ldouble_ulong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ldouble_ulong_into(data, &mut out);
    out
}

/// Datatype operation: conv ldouble llong.
#[allow(non_snake_case)]
pub fn H5T__conv_ldouble_llong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ldouble llong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ldouble_llong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ldouble_llong_into(data, &mut out);
    out
}

/// Datatype operation: conv ldouble ullong.
#[allow(non_snake_case)]
pub fn H5T__conv_ldouble_ullong_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ldouble ullong.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ldouble_ullong(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ldouble_ullong_into(data, &mut out);
    out
}

/// Datatype operation: conv ldouble  Float16.
#[allow(non_snake_case)]
pub fn H5T__conv_ldouble__Float16_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ldouble  Float16.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ldouble__Float16(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ldouble__Float16_into(data, &mut out);
    out
}

/// Datatype operation: conv ldouble float.
#[allow(non_snake_case)]
pub fn H5T__conv_ldouble_float_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ldouble float.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ldouble_float(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ldouble_float_into(data, &mut out);
    out
}

/// Datatype operation: conv ldouble double.
#[allow(non_snake_case)]
pub fn H5T__conv_ldouble_double_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ldouble double.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ldouble_double(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ldouble_double_into(data, &mut out);
    out
}

/// Datatype operation: conv ldouble fcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv_ldouble_fcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ldouble fcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ldouble_fcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ldouble_fcomplex_into(data, &mut out);
    out
}

/// Datatype operation: conv ldouble dcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv_ldouble_dcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ldouble dcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ldouble_dcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ldouble_dcomplex_into(data, &mut out);
    out
}

/// Datatype operation: conv ldouble lcomplex.
#[allow(non_snake_case)]
pub fn H5T__conv_ldouble_lcomplex_into(data: &[u8], out: &mut Vec<u8>) {
    conv_copy_into(data, out);
}

/// Datatype operation: conv ldouble lcomplex.
#[deprecated(
    note = "use the corresponding _into() function to reuse caller-provided output storage"
)]
#[allow(non_snake_case)]
pub fn H5T__conv_ldouble_lcomplex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    H5T__conv_ldouble_lcomplex_into(data, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn datatype_registry_commits_and_opens_named_types() {
        let mut reg = DatatypeRegistry::default();
        let dtype = H5Tcreate(DatatypeClass::FixedPoint, 4);
        H5T__commit_api_common(&mut reg, "i32", dtype);
        let opened = H5Topen2_ref(&reg, "i32").unwrap();
        assert!(H5Tcommitted(&opened));
        assert_eq!(H5T_nameof(&opened).unwrap(), "i32");
        assert!(H5T_nameof(&H5Tcreate(DatatypeClass::FixedPoint, 4)).is_err());
        let mut anonymous = H5Tcommit_anon(H5Tcreate(DatatypeClass::FixedPoint, 4));
        assert!(H5T_nameof(&anonymous).is_err());
        anonymous.name = Some("anon".to_string());
        anonymous.immutable = true;
        assert!(H5T_nameof(&anonymous).is_err());
        assert!(H5T_is_sensible(&opened));
    }

    #[test]
    fn datatype_encode_decode_uses_message_validator() {
        let mut dtype = H5Tcreate(DatatypeClass::FixedPoint, 4);
        dtype.message.properties[2..4].copy_from_slice(&32u16.to_le_bytes());
        let mut image = Vec::new();
        H5Tencode_into(&dtype, &mut image).unwrap();
        let decoded = H5Tdecode(&image).unwrap();
        assert_eq!(decoded.message.version, dtype.message.version);
        assert_eq!(decoded.message.class, dtype.message.class);
        assert_eq!(decoded.message.class_bits, dtype.message.class_bits);
        assert_eq!(decoded.message.size, dtype.message.size);
        assert_eq!(decoded.message.properties, dtype.message.properties);

        let mut malformed = dtype.clone();
        malformed.message.properties.truncate(3);
        assert!(H5Tencode_into(&malformed, &mut image).is_err());
        assert!(H5Tdecode(&image[..7]).is_err());
    }

    #[test]
    fn set_precision_adjusts_atomic_offset_and_size() {
        let mut dtype = H5Tcreate(DatatypeClass::FixedPoint, 4);
        H5Tset_offset(&mut dtype, 24).unwrap();

        H5Tset_precision(&mut dtype, 16).unwrap();
        assert_eq!(dtype.message.size, 4);
        assert_eq!(H5Tget_offset(&dtype), Some(16));
        assert_eq!(H5Tget_precision(&dtype), Some(16));

        H5Tset_precision(&mut dtype, 40).unwrap();
        assert_eq!(dtype.message.size, 5);
        assert_eq!(H5Tget_offset(&dtype), Some(0));
        assert_eq!(H5Tget_precision(&dtype), Some(40));
    }

    #[test]
    fn set_precision_rejects_invalid_float_and_read_only_classes() {
        let mut float = H5Tcreate(DatatypeClass::FloatingPoint, 4);
        float.message.class_bits[1] = 31;
        float.message.properties[2..4].copy_from_slice(&32u16.to_le_bytes());
        float.message.properties[4] = 23;
        float.message.properties[5] = 8;
        float.message.properties[6] = 0;
        float.message.properties[7] = 23;
        assert!(H5Tset_precision(&mut float, 16).is_err());

        let mut string = H5Tcreate(DatatypeClass::String, 8);
        assert!(H5Tset_precision(&mut string, 8).is_err());

        let mut locked = H5Tcreate(DatatypeClass::BitField, 1);
        H5T_lock(&mut locked);
        assert!(H5Tset_precision(&mut locked, 8).is_err());
        assert!(H5Tset_precision(&mut H5Tcreate(DatatypeClass::FixedPoint, 1), 0).is_err());
    }

    #[test]
    fn set_offset_grows_atomic_size_and_rejects_read_only_classes() {
        let mut dtype = H5Tcreate(DatatypeClass::BitField, 1);
        H5Tset_precision(&mut dtype, 8).unwrap();
        H5Tset_offset(&mut dtype, 12).unwrap();
        assert_eq!(dtype.message.size, 3);
        assert_eq!(H5Tget_offset(&dtype), Some(12));

        let mut string = H5Tcreate(DatatypeClass::String, 8);
        assert!(H5Tset_offset(&mut string, 1).is_err());
        assert!(H5Tset_offset(&mut string, 0).is_ok());

        let mut locked = H5Tcreate(DatatypeClass::FixedPoint, 1);
        H5T_lock(&mut locked);
        assert!(H5Tset_offset(&mut locked, 1).is_err());
    }

    #[test]
    fn set_size_adjusts_atomic_precision_and_string_kind() {
        let mut dtype = H5Tcreate(DatatypeClass::FixedPoint, 4);
        H5Tset_precision(&mut dtype, 32).unwrap();
        H5Tset_offset(&mut dtype, 8).unwrap();
        H5Tset_size(&mut dtype, 2).unwrap();
        assert_eq!(dtype.message.size, 2);
        assert_eq!(H5Tget_offset(&dtype), Some(0));
        assert_eq!(H5Tget_precision(&dtype), Some(16));

        let mut string = H5Tcreate(DatatypeClass::String, 4);
        H5Tset_size(&mut string, 2).unwrap();
        assert_eq!(string.message.size, 2);
        assert_eq!(H5Tget_precision(&string), None);

        H5Tset_size(&mut string, 8192).unwrap();
        assert_eq!(string.message.class, DatatypeClass::String);
        assert_eq!(string.message.size, 8192);
        assert_eq!(H5Tget_precision(&string), None);
        assert!(!H5Tis_variable_str(&string));
        let mut encoded = Vec::new();
        H5Tencode_into(&string, &mut encoded).unwrap();

        H5Tset_size(&mut string, u32::MAX).unwrap();
        assert_eq!(string.message.class, DatatypeClass::VarLen);
        assert!(H5Tis_variable_str(&string));
        assert!(string.force_conv);
        H5Tset_size(&mut string, 8192).unwrap();
        assert_eq!(string.message.class, DatatypeClass::VarLen);
        assert!(H5Tis_variable_str(&string));
        assert_eq!(string.message.size, 8192);
    }

    #[test]
    fn set_size_rejects_invalid_and_unsupported_datatypes() {
        let mut fixed = H5Tcreate(DatatypeClass::FixedPoint, 4);
        assert!(H5Tset_size(&mut fixed, 0).is_err());

        let mut reference = H5Tcreate(DatatypeClass::Reference, 8);
        assert!(H5Tset_size(&mut reference, 4).is_err());
        assert!(H5Tset_size(&mut reference, u32::MAX).is_err());

        let mut locked = H5Tcreate(DatatypeClass::BitField, 1);
        H5T_lock(&mut locked);
        assert!(H5Tset_size(&mut locked, 2).is_err());
    }

    #[test]
    fn member_index_searches_compound_and_enum_members() {
        let mut base = H5Tcreate(DatatypeClass::FixedPoint, 1);
        H5Tset_precision(&mut base, 8).unwrap();
        let mut enum_type = H5Tenum_create(&base).unwrap();
        H5Tenum_insert(&mut enum_type, "red", 1).unwrap();
        H5Tenum_insert(&mut enum_type, "blue", 2).unwrap();

        assert_eq!(H5Tget_member_index(&enum_type, "red"), Some(0));
        assert_eq!(H5Tget_member_index(&enum_type, "blue"), Some(1));
        assert_eq!(H5Tget_member_index(&enum_type, "green"), None);
        assert_eq!(
            H5Tget_member_index(&H5Tcreate(DatatypeClass::String, 4), "red"),
            None
        );
    }

    #[test]
    fn enum_lookup_and_insert_follow_sorted_search_and_duplicate_rules() {
        let mut base = H5Tcreate(DatatypeClass::FixedPoint, 1);
        H5Tset_precision(&mut base, 8).unwrap();
        let mut enum_type = H5Tenum_create(&base).unwrap();
        assert!(H5T__enum_nameof_ref(&enum_type, 1).is_err());
        assert!(H5T__enum_valueof(&enum_type, "one").is_err());

        H5T__enum_insert(&mut enum_type, "two", 2).unwrap();
        H5T__enum_insert(&mut enum_type, "one", 1).unwrap();
        assert_eq!(H5T__enum_nameof_ref(&enum_type, 1).unwrap(), Some("one"));
        assert_eq!(H5T__enum_nameof_ref(&enum_type, 3).unwrap(), None);
        assert_eq!(H5T__enum_valueof(&enum_type, "two").unwrap(), Some(2));
        assert_eq!(H5T__enum_valueof(&enum_type, "missing").unwrap(), None);
        assert!(H5T__enum_insert(&mut enum_type, "two", 3).is_err());
        assert!(H5T__enum_insert(&mut enum_type, "three", 2).is_err());
        assert!(H5T__enum_valueof(&enum_type, "").is_err());
        assert!(H5T__enum_nameof_ref(&H5Tcreate(DatatypeClass::FixedPoint, 1), 1).is_err());
    }

    #[test]
    fn noop_conversion_checks_force_flags_and_registered_noop_paths() {
        let mut registry = DatatypeRegistry::default();
        let src = H5Tcreate(DatatypeClass::FixedPoint, 4);
        let mut dst = src.clone();
        assert!(H5T_noop_conv(&registry, &src, &dst));
        assert!(H5T_path_match(&src, &dst));

        dst.force_conv = true;
        assert!(!H5T_noop_conv(&registry, &src, &dst));
        assert!(!H5T_path_match(&src, &dst));

        let float = H5Tcreate(DatatypeClass::FloatingPoint, 4);
        assert!(!H5T_noop_conv(&registry, &src, &float));
        H5T__path_find_init_new_path(
            &mut registry,
            DatatypeClass::FixedPoint,
            DatatypeClass::FloatingPoint,
            "noop",
        );
        assert!(H5T_noop_conv(&registry, &src, &float));
    }

    #[test]
    fn owned_vol_object_is_replaced_on_datatype() {
        let mut dtype = H5Tcreate(DatatypeClass::VarLen, 16);
        H5T_own_vol_obj(&mut dtype, "file-a").unwrap();
        assert_eq!(dtype.owned_vol_obj.as_deref(), Some("file-a"));

        H5T_own_vol_obj(&mut dtype, "file-b").unwrap();
        assert_eq!(dtype.owned_vol_obj.as_deref(), Some("file-b"));
        assert!(H5T_own_vol_obj(&mut dtype, "").is_err());
    }

    #[test]
    fn datatype_member_sorting_reorders_enum_and_compound_members() {
        let mut base = H5Tcreate(DatatypeClass::FixedPoint, 1);
        H5Tset_precision(&mut base, 8).unwrap();
        let mut enum_type = H5Tenum_create(&base).unwrap();
        H5T__enum_insert(&mut enum_type, "zeta", 3).unwrap();
        H5T__enum_insert(&mut enum_type, "alpha", 1).unwrap();
        H5T__enum_insert(&mut enum_type, "middle", 2).unwrap();

        let mut map = [10usize, 11, 12];
        H5T__sort_name(&mut enum_type, Some(&mut map)).unwrap();
        assert_eq!(
            enum_type
                .message
                .enum_members_iter()
                .unwrap()
                .next()
                .unwrap()
                .unwrap()
                .name,
            "alpha"
        );
        assert_eq!(map, [11, 12, 10]);

        H5T__sort_value(&mut enum_type, None).unwrap();
        let values = enum_type
            .message
            .enum_members_iter()
            .unwrap()
            .map(|member| member.unwrap().value)
            .collect::<Vec<_>>();
        assert_eq!(values, vec![1, 2, 3]);

        let mut compound = H5Tcreate(DatatypeClass::Compound, 8);
        H5Tinsert(&mut compound, "b", 4, base.clone()).unwrap();
        H5Tinsert(&mut compound, "a", 0, base).unwrap();
        H5T__sort_name(&mut compound, None).unwrap();
        assert_eq!(
            compound
                .message
                .compound_fields_iter()
                .unwrap()
                .next()
                .unwrap()
                .unwrap()
                .name,
            "a"
        );
        H5T__sort_value(&mut compound, None).unwrap();
        assert_eq!(
            compound
                .message
                .compound_fields_iter()
                .unwrap()
                .next()
                .unwrap()
                .unwrap()
                .byte_offset,
            0
        );
    }

    #[test]
    fn variable_string_query_uses_vlen_string_class_bits() {
        let mut vlen_string = H5Tcreate(DatatypeClass::VarLen, 16);
        vlen_string.message.class_bits[0] = 1;
        assert!(H5Tis_variable_str(&vlen_string));
        assert_eq!(H5T_is_variable_str(&vlen_string).unwrap(), true);

        let vlen_sequence = H5Tcreate(DatatypeClass::VarLen, 16);
        assert!(!H5Tis_variable_str(&vlen_sequence));
        assert!(!H5Tis_variable_str(&H5Tcreate(DatatypeClass::String, 4)));
    }

    #[test]
    fn set_version_applies_file_low_and_high_bounds() {
        let mut dtype = H5Tcreate(DatatypeClass::FixedPoint, 4);
        dtype.message.version = 1;

        H5T_set_version(&mut dtype, 3, 6).unwrap();
        assert_eq!(dtype.message.version, 4);

        assert!(H5T_set_version(&mut dtype, 0, 1).is_err());
        assert!(H5T_set_version(&mut dtype, 7, 6).is_err());
        assert!(H5T_set_version(&mut dtype, 0, 7).is_err());
    }

    #[test]
    fn variable_length_storage_detection_recurses_into_members() {
        assert!(H5T_is_vl_storage(&H5Tcreate(DatatypeClass::VarLen, 16)));
        assert!(H5T_is_vl_storage(&H5Tcreate(DatatypeClass::Reference, 8)));
        assert!(!H5T_is_vl_storage(&H5Tcreate(DatatypeClass::FixedPoint, 4)));

        let mut compound = H5Tcreate(DatatypeClass::Compound, 16);
        let mut fixed = H5Tcreate(DatatypeClass::FixedPoint, 4);
        H5Tset_precision(&mut fixed, 32).unwrap();
        let reference = H5Tcreate(DatatypeClass::Reference, 8);
        H5Tinsert(&mut compound, "fixed", 0, fixed).unwrap();
        assert!(!H5T_is_vl_storage(&compound));
        H5Tinsert(&mut compound, "ref", 8, reference).unwrap();
        assert!(H5T_is_vl_storage(&compound));
    }

    #[test]
    fn sensible_datatype_check_rejects_only_empty_compound_and_enum() {
        assert!(!H5T_is_sensible(&H5Tcreate(DatatypeClass::Compound, 8)));
        assert!(H5T_is_sensible(&H5Tcreate(DatatypeClass::FixedPoint, 0)));

        let mut base = H5Tcreate(DatatypeClass::FixedPoint, 1);
        H5Tset_precision(&mut base, 8).unwrap();
        let mut enum_type = H5Tenum_create(&base).unwrap();
        assert!(!H5T_is_sensible(&enum_type));
        H5Tenum_insert(&mut enum_type, "one", 1).unwrap();
        assert!(H5T_is_sensible(&enum_type));

        let mut compound = H5Tcreate(DatatypeClass::Compound, 4);
        H5Tinsert(&mut compound, "x", 0, base).unwrap();
        assert!(H5T_is_sensible(&compound));
    }

    #[test]
    fn cset_and_opaque_tag_setters_validate_class_and_state() {
        let mut string = H5Tcreate(DatatypeClass::String, 4);
        H5Tset_strpad(&mut string, 2).unwrap();
        H5Tset_cset(&mut string, 1).unwrap();
        assert_eq!(H5Tget_cset(&string), Some(1));
        assert_eq!(H5Tget_strpad(&string), Some(2));
        assert!(H5Tset_cset(&mut string, 2).is_err());
        assert!(H5Tset_strpad(&mut string, 3).is_err());

        let mut vlen_string = H5Tcreate(DatatypeClass::VarLen, 16);
        vlen_string.message.class_bits[0] = 1;
        H5Tset_cset(&mut vlen_string, 1).unwrap();
        H5Tset_strpad(&mut vlen_string, 2).unwrap();
        assert_eq!(H5Tget_cset(&vlen_string), Some(1));
        assert_eq!(H5Tget_strpad(&vlen_string), Some(2));
        assert_eq!(vlen_string.message.class_bits[0] & 0x0f, 1);

        let mut fixed = H5Tcreate(DatatypeClass::FixedPoint, 4);
        assert!(H5Tset_cset(&mut fixed, 1).is_err());
        assert!(H5Tset_strpad(&mut fixed, 1).is_err());

        let mut opaque = H5Tcreate(DatatypeClass::Opaque, 3);
        H5Tset_tag(&mut opaque, "rgb").unwrap();
        assert_eq!(H5Tget_tag_ref(&opaque), Some("rgb"));
        assert!(H5Tset_tag(&mut opaque, "x".repeat(256)).is_err());
        assert!(H5Tset_tag(&mut string, "bad").is_err());

        H5T_lock(&mut opaque);
        assert!(H5Tset_tag(&mut opaque, "new").is_err());
    }

    #[test]
    fn numeric_setters_validate_class_state_and_field_layout() {
        let mut int_type = H5Tcreate(DatatypeClass::FixedPoint, 4);
        H5Tset_sign(&mut int_type, 1).unwrap();
        H5Tset_pad(&mut int_type, 1, 0).unwrap();
        assert_eq!(H5Tget_sign(&int_type), Some(true));
        assert_eq!(int_type.message.class_bits[0] & 0x06, 0x02);
        assert!(H5Tset_sign(&mut int_type, 2).is_err());
        assert!(H5Tset_pad(&mut int_type, 2, 0).is_err());

        let mut float = H5Tcreate(DatatypeClass::FloatingPoint, 4);
        H5Tset_precision(&mut float, 32).unwrap();
        H5Tset_fields(&mut float, 31, 23, 8, 0, 23).unwrap();
        H5Tset_norm(&mut float, 0).unwrap();
        H5Tset_inpad(&mut float, 1).unwrap();
        H5Tset_pad(&mut float, 0, 1).unwrap();
        assert_eq!(H5Tget_fields(&float).unwrap().sign_position, 31);
        assert_eq!(H5Tget_fields(&float).unwrap().exponent_position, 23);
        assert_eq!(H5Tget_fields(&float).unwrap().mantissa_size, 23);
        assert_eq!(H5Tget_norm(&float), Some(0));
        assert_eq!(H5Tget_inpad(&float), Some(1));
        assert_eq!(float.message.class_bits[0] & 0x06, 0x04);
        assert!(H5Tset_fields(&mut float, 23, 23, 8, 0, 23).is_err());
        assert!(H5Tset_norm(&mut float, 3).is_err());
        assert!(H5Tset_inpad(&mut float, 2).is_err());

        let mut string = H5Tcreate(DatatypeClass::String, 4);
        assert!(H5Tset_sign(&mut string, 1).is_err());
        assert!(H5Tset_fields(&mut string, 0, 1, 1, 2, 1).is_err());

        H5T_lock(&mut int_type);
        assert!(H5Tset_sign(&mut int_type, 0).is_err());
    }

    #[test]
    fn compound_insert_appends_validated_member_encoding() {
        let mut compound = H5Tcreate(DatatypeClass::Compound, 8);
        let mut member = H5Tcreate(DatatypeClass::FixedPoint, 4);
        H5Tset_precision(&mut member, 32).unwrap();

        H5Tinsert(&mut compound, "a", 0, member.clone()).unwrap();
        H5Tinsert(&mut compound, "b", 4, member.clone()).unwrap();
        let fields = H5Tget_member_info_iter(&compound)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].name, "a");
        assert_eq!(fields[1].byte_offset, 4);
        assert_eq!(H5Tget_member_index(&compound, "b"), Some(1));
        let mut image = Vec::new();
        assert!(H5Tencode_into(&compound, &mut image).is_ok());

        assert!(H5Tinsert(&mut compound, "a", 0, member.clone()).is_err());
        assert!(H5Tinsert(&mut compound, "c", 2, member.clone()).is_err());
        assert!(H5Tinsert(&mut compound, "c", 6, member.clone()).is_err());
        assert!(H5Tinsert(&mut compound, "", 0, member.clone()).is_err());

        let mut locked = H5Tcreate(DatatypeClass::Compound, 4);
        H5T_lock(&mut locked);
        assert!(H5Tinsert(&mut locked, "x", 0, member).is_err());
    }

    #[test]
    fn bit_set_and_find_use_c_bit_region_semantics() {
        let mut bytes = [0u8; 3];
        H5T__bit_set(&mut bytes, 3, 12, true);
        assert_eq!(bytes, [0b1111_1000, 0b0111_1111, 0]);

        H5T__bit_set(&mut bytes, 5, 6, false);
        assert_eq!(bytes, [0b0001_1000, 0b0111_1000, 0]);

        assert_eq!(
            H5T__bit_find(&bytes, 3, 12, H5TBitSearchDirection::Lsb, true),
            Some(0)
        );
        assert_eq!(
            H5T__bit_find(&bytes, 3, 12, H5TBitSearchDirection::Lsb, false),
            Some(2)
        );
        assert_eq!(
            H5T__bit_find(&bytes, 3, 12, H5TBitSearchDirection::Msb, true),
            Some(11)
        );
        assert_eq!(
            H5T__bit_find(&bytes, 3, 12, H5TBitSearchDirection::Msb, false),
            Some(7)
        );
        assert_eq!(
            H5T__bit_find(&bytes, 16, 0, H5TBitSearchDirection::Lsb, true),
            None
        );

        let mut bits = [0b0000_0111u8];
        assert_eq!(H5T__bit_inc(&mut bits, 0, 3).unwrap(), true);
        assert_eq!(bits[0], 0);
        assert_eq!(H5T__bit_dec(&mut bits, 0, 3).unwrap(), true);
        assert_eq!(bits[0] & 0b0000_0111, 0b0000_0111);
        H5T__bit_neg(&mut bits, 1, 3).unwrap();
        assert_eq!(bits[0], 0b0000_1001);
        assert!(H5T__bit_inc(&mut bits, 7, 2).is_err());
    }

    #[test]
    fn bit_copy_uses_little_endian_bit_vector_offsets() {
        let src = [0b1011_0101, 0b0000_0011];
        let mut dst = [0u8; 2];

        H5T__bit_copy(&mut dst, 3, &src, 2, 10).unwrap();
        assert_eq!(dst, [0b0110_1000, 0b0000_0111]);

        H5T__bit_copy(&mut dst, 15, &src, 0, 2).unwrap_err();
        H5T__bit_copy(&mut dst, 0, &src, usize::MAX, 1).unwrap_err();
    }

    #[test]
    fn native_float_bit_cmp_and_fix_order_follow_c_permutation_rules() {
        let perm = [1usize, 0];
        let a = [0b0000_0000, 0b0000_0100];
        let b = [0b0000_0000, 0b0000_0000];
        let pad = [0xffu8, 0xff];
        assert_eq!(H5T__bit_cmp(2, &perm, &a, &b, &pad).unwrap(), 2);

        let pad = [0xffu8, 0x00];
        assert!(H5T__bit_cmp(2, &perm, &a, &b, &pad).is_err());
        assert!(H5T__bit_cmp(2, &[2, 0], &a, &b, &[0xff, 0xff]).is_err());

        let mut little = [3usize, 2, 1, 0];
        assert_eq!(
            H5T__fix_order(4, 3, &mut little).unwrap(),
            H5TDetectedOrder::LittleEndian
        );
        assert_eq!(little, [0, 1, 2, 3]);

        let mut big = [0usize, 1, 2, 3];
        assert_eq!(
            H5T__fix_order(4, 3, &mut big).unwrap(),
            H5TDetectedOrder::BigEndian
        );
        assert_eq!(big, [3, 2, 1, 0]);

        let mut mixed = [1usize, 3, 0, 2];
        assert_eq!(
            H5T__fix_order(4, 3, &mut mixed).unwrap(),
            H5TDetectedOrder::Vax
        );
        assert_eq!(mixed, [2, 3, 0, 1]);

        assert!(H5T__fix_order(3, 2, &mut [1usize, 0, 2]).is_err());
        assert!(H5T__fix_order(2, 0, &mut [1usize, 0]).is_err());
    }

    #[test]
    fn reverse_order_handles_atomic_complex_and_vax_layouts() {
        let mut converted = Vec::new();
        H5T__reverse_order_into(
            &[1, 2, 3, 4],
            H5TDetectedOrder::LittleEndian,
            false,
            &mut converted,
        )
        .unwrap();
        assert_eq!(converted, vec![1, 2, 3, 4]);
        H5T__reverse_order_into(
            &[1, 2, 3, 4],
            H5TDetectedOrder::BigEndian,
            false,
            &mut converted,
        )
        .unwrap();
        assert_eq!(converted, vec![4, 3, 2, 1]);
        H5T__reverse_order_into(
            &[1, 2, 3, 4],
            H5TDetectedOrder::BigEndian,
            true,
            &mut converted,
        )
        .unwrap();
        assert_eq!(converted, vec![2, 1, 4, 3]);
        H5T__reverse_order_into(&[1, 2, 3, 4], H5TDetectedOrder::Vax, false, &mut converted)
            .unwrap();
        assert_eq!(converted, vec![3, 4, 1, 2]);
        assert!(
            H5T__reverse_order_into(&[1, 2, 3], H5TDetectedOrder::Vax, false, &mut converted)
                .is_err()
        );

        H5T__conv_order_opt_into(
            &[1, 2, 3, 4, 5, 6, 7, 8],
            4,
            H5TDetectedOrder::BigEndian,
            false,
            &mut converted,
        )
        .unwrap();
        assert_eq!(converted, vec![4, 3, 2, 1, 8, 7, 6, 5]);
        assert!(H5T__conv_order_opt_into(
            &[1, 2, 3],
            3,
            H5TDetectedOrder::BigEndian,
            false,
            &mut converted,
        )
        .is_err());
    }

    #[test]
    fn float_special_value_detection_uses_c_bit_fields() {
        let f32_fields = FloatFields {
            sign_position: 31,
            exponent_position: 23,
            exponent_size: 8,
            mantissa_position: 0,
            mantissa_size: 23,
        };

        assert_eq!(
            H5T__conv_float_find_special(&0.0f32.to_bits().to_le_bytes(), f32_fields, 1).unwrap(),
            (H5TConvFloatSpecval::PosZero, 0)
        );
        assert_eq!(
            H5T__conv_float_find_special(&(-0.0f32).to_bits().to_le_bytes(), f32_fields, 1)
                .unwrap(),
            (H5TConvFloatSpecval::NegZero, 1)
        );
        assert_eq!(
            H5T__conv_float_find_special(&f32::INFINITY.to_bits().to_le_bytes(), f32_fields, 1)
                .unwrap(),
            (H5TConvFloatSpecval::PosInf, 0)
        );
        assert_eq!(
            H5T__conv_float_find_special(&f32::NEG_INFINITY.to_bits().to_le_bytes(), f32_fields, 1)
                .unwrap(),
            (H5TConvFloatSpecval::NegInf, 1)
        );
        assert_eq!(
            H5T__conv_float_find_special(&f32::NAN.to_bits().to_le_bytes(), f32_fields, 1)
                .unwrap()
                .0,
            H5TConvFloatSpecval::Nan
        );
        assert_eq!(
            H5T__conv_float_find_special(&1.5f32.to_bits().to_le_bytes(), f32_fields, 1)
                .unwrap()
                .0,
            H5TConvFloatSpecval::Regular
        );

        let mut intel_long_double_like = [0u8; 3];
        H5T__bit_set(&mut intel_long_double_like, 10, 1, true);
        H5T__bit_set(&mut intel_long_double_like, 11, 5, true);
        let no_implied_mantissa_fields = FloatFields {
            sign_position: 16,
            exponent_position: 11,
            exponent_size: 5,
            mantissa_position: 0,
            mantissa_size: 11,
        };
        assert_eq!(
            H5T__conv_float_find_special(&intel_long_double_like, no_implied_mantissa_fields, 2)
                .unwrap(),
            (H5TConvFloatSpecval::PosInf, 0)
        );
    }

    #[test]
    fn disk_reference_callbacks_encode_header_size_and_blob_payload() {
        let src = [H5R_OBJECT2, 0, 0xaa, 0xbb, 0xcc];
        let mut disk = Vec::new();
        H5T__ref_disk_write(&mut disk, &src, None).unwrap();
        assert_eq!(&disk[..2], &[H5R_OBJECT2, 0]);
        assert_eq!(
            u32::from_le_bytes(disk[2..6].try_into().unwrap()),
            (src.len() - H5R_ENCODE_HEADER_SIZE) as u32
        );
        let mut decoded_ref = Vec::new();
        H5T__ref_disk_read_into(&disk, &mut decoded_ref).unwrap();
        assert_eq!(decoded_ref, src);
        assert_eq!(H5T__ref_disk_getsize(&disk).unwrap(), (disk.len(), true));
        assert!(!H5T__ref_disk_isnull(&disk).unwrap());

        let mut external = disk.clone();
        external[1] = H5R_IS_EXTERNAL;
        assert_eq!(
            H5T__ref_disk_getsize(&external).unwrap(),
            (src.len(), false)
        );

        let mut null_ref = [0xffu8; 8];
        H5T__ref_disk_setnull(&mut null_ref, None).unwrap();
        assert!(H5T__ref_disk_isnull(&null_ref).unwrap());
        assert!(H5T__ref_disk_read_into(&disk[..5], &mut decoded_ref).is_err());
        assert!(H5T__ref_disk_getsize(&[]).is_err());

        let legacy_object = [0x34u8, 0x12, 0, 0, 0, 0, 0, 0];
        assert!(!H5T__ref_obj_disk_isnull(&legacy_object, 8).unwrap());
        assert_eq!(H5T__ref_obj_disk_getsize(&legacy_object, 8).unwrap(), 8);
        H5T__ref_obj_disk_read_into(&legacy_object, 8, &mut decoded_ref).unwrap();
        assert_eq!(decoded_ref, legacy_object);
        assert!(H5T__ref_obj_disk_isnull(&[0u8; 8], 8).unwrap());

        let legacy_region = [0x34u8, 0x12, 0, 0, 1, 2, 3, 4, 5, 6];
        assert!(!H5T__ref_dsetreg_disk_isnull(&legacy_region, 4).unwrap());
        let mut token = Vec::new();
        let mut space = Vec::new();
        H5T__ref_dsetreg_disk_read_into(&legacy_region, 4, &mut token, &mut space).unwrap();
        assert_eq!(token, vec![0x34, 0x12, 0, 0]);
        assert_eq!(space, vec![1, 2, 3, 4, 5, 6]);
        assert!(H5T__ref_dsetreg_disk_getsize(&legacy_region, 4).unwrap() >= 4);
        assert!(
            H5T__ref_dsetreg_disk_read_into(&legacy_region[..2], 4, &mut token, &mut space)
                .is_err()
        );
    }

    #[test]
    fn disk_vlen_callbacks_use_length_prefix_and_blob_payload() {
        let src = [1u8, 2, 3, 4, 5, 6];
        let mut disk = Vec::new();
        H5T__vlen_disk_write(&mut disk, &src, None, 3, 2).unwrap();
        assert_eq!(H5T__vlen_disk_getlen(&disk).unwrap(), 3);
        assert_eq!(H5T__vlen_disk_read_ref(&disk, 6).unwrap(), src);
        assert!(!H5T__vlen_disk_isnull(&disk).unwrap());

        let mut old = vec![9u8; 4];
        H5T__vlen_disk_write(&mut disk, &src, Some(&mut old), 2, 2).unwrap();
        assert!(old.is_empty());
        assert_eq!(H5T__vlen_disk_getlen(&disk).unwrap(), 2);
        assert_eq!(H5T__vlen_disk_read_ref(&disk, 4).unwrap(), &src[..4]);

        H5T__vlen_disk_setnull(&mut disk, None).unwrap();
        assert_eq!(H5T__vlen_disk_getlen(&disk).unwrap(), 0);
        assert!(H5T__vlen_disk_isnull(&disk).unwrap());
        assert!(H5T__vlen_disk_read_ref(&disk, 1).is_err());
        assert!(H5T__vlen_disk_write(&mut disk, &src, None, usize::MAX, 2).is_err());
    }

    #[test]
    fn vlen_memory_string_write_rejects_invalid_utf8() {
        let mut value = String::new();
        H5T__vlen_mem_str_write(&mut value, b"alpha\0\0").unwrap();
        assert_eq!(value, "alpha");
        assert!(H5T__vlen_mem_str_write(&mut value, &[0xff]).is_err());
    }
}
