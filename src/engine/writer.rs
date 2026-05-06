use std::collections::{HashMap, HashSet};
use std::io::{Seek, SeekFrom, Write};

use crate::engine::allocator::FileAllocator;
use crate::error::{Error, Result};
use crate::format::checksum::checksum_metadata;
use crate::format::object_header::*;
use crate::format::superblock::Superblock;
use crate::io::reader::UNDEF_ADDR;

const MAX_DATASPACE_RANK: usize = 32;

/// A writable HDF5 file under construction.
pub struct HdfFileWriter<W: Write + Seek> {
    writer: W,
    allocator: FileAllocator,
    sizeof_addr: u8,
    sizeof_size: u8,
    /// Map of group path -> object header address.
    groups: HashMap<String, u64>,
    /// Pending links: (parent_path, child_name, child_addr).
    links: Vec<(String, String, u64)>,
    /// Pending hard-link aliases: (parent_path, link_name, target_path).
    hard_links: Vec<(String, String, String)>,
    /// Pending pre-encoded attribute messages for the root group.
    pending_root_attrs: Vec<(u16, Vec<u8>)>,
    /// Pending root attributes added through the typed writer API.
    pending_root_attr_specs: Vec<OwnedAttrSpec>,
    /// Pending group attributes added through the typed writer API.
    pending_group_attr_specs: HashMap<String, Vec<OwnedAttrSpec>>,
    /// Pre-encoded link messages (for soft/external links): (parent_path, child_name, encoded_link_msg).
    special_links: Vec<(String, String, Vec<u8>)>,
}

/// Describes an attribute to attach.
pub struct AttrSpec<'a> {
    pub name: &'a str,
    pub shape: &'a [u64],
    pub dtype: DtypeSpec,
    pub data: &'a [u8],
}

#[derive(Clone)]
struct OwnedAttrSpec {
    name: String,
    shape: Vec<u64>,
    dtype: DtypeSpec,
    data: Vec<u8>,
}

impl OwnedAttrSpec {
    fn as_attr_spec(&self) -> AttrSpec<'_> {
        AttrSpec {
            name: &self.name,
            shape: &self.shape,
            dtype: self.dtype.clone(),
            data: &self.data,
        }
    }
}

/// Describes a dataset to create.
pub struct DatasetSpec<'a> {
    pub name: &'a str,
    pub shape: &'a [u64],
    pub max_shape: Option<&'a [u64]>,
    pub dtype: DtypeSpec,
    pub data: &'a [u8],
}

#[derive(Debug, Clone)]
struct ChunkBTreeEntry {
    coords: Vec<u64>,
    chunk_size: u32,
    filter_mask: u32,
    child_addr: u64,
}

/// Describes the dataset fill-value message to write.
#[derive(Debug, Clone, Copy)]
pub struct FillValueSpec<'a> {
    pub alloc_time: u8,
    pub fill_time: u8,
    pub value: Option<&'a [u8]>,
}

impl<'a> FillValueSpec<'a> {
    pub fn undefined(alloc_time: u8, fill_time: u8) -> Self {
        Self {
            alloc_time,
            fill_time,
            value: None,
        }
    }

    pub fn with_value(alloc_time: u8, fill_time: u8, value: &'a [u8]) -> Self {
        Self {
            alloc_time,
            fill_time,
            value: Some(value),
        }
    }
}

/// Describes a compound datatype field.
#[derive(Debug, Clone)]
pub struct CompoundFieldSpec {
    pub name: String,
    pub offset: u32,
    pub dtype: DtypeSpec,
}

/// Describes a datatype.
#[derive(Debug, Clone)]
pub enum DtypeSpec {
    F64,
    F32,
    I128,
    I64,
    I32,
    I16,
    I8,
    U128,
    U64,
    U32,
    U16,
    U8,
    FixedAsciiString {
        len: u32,
        padding: u8,
    },
    FixedUtf8String {
        len: u32,
        padding: u8,
    },
    VarLenUtf8String,
    Compound {
        size: u32,
        fields: Vec<CompoundFieldSpec>,
    },
    Enum {
        base: Box<DtypeSpec>,
        members: Vec<(String, u64)>,
    },
    Opaque {
        size: u32,
        tag: String,
    },
    Array {
        dims: Vec<u32>,
        base: Box<DtypeSpec>,
    },
}

impl DtypeSpec {
    pub fn size(&self) -> u32 {
        match self {
            DtypeSpec::I128 | DtypeSpec::U128 => 16,
            DtypeSpec::F64 | DtypeSpec::I64 | DtypeSpec::U64 => 8,
            DtypeSpec::F32 | DtypeSpec::I32 | DtypeSpec::U32 => 4,
            DtypeSpec::I16 | DtypeSpec::U16 => 2,
            DtypeSpec::I8 | DtypeSpec::U8 => 1,
            DtypeSpec::FixedAsciiString { len, .. } | DtypeSpec::FixedUtf8String { len, .. } => {
                *len
            }
            DtypeSpec::VarLenUtf8String => 16,
            DtypeSpec::Compound { size, .. } => *size,
            DtypeSpec::Enum { base, .. } => base.size(),
            DtypeSpec::Opaque { size, .. } => *size,
            DtypeSpec::Array { .. } => self.checked_size().unwrap_or(u32::MAX),
        }
    }

    /// Encode as HDF5 datatype message bytes.
    pub fn encode(&self) -> Result<Vec<u8>> {
        self.encode_with_padding(true)
    }

    fn encode_embedded(&self) -> Result<Vec<u8>> {
        self.encode_with_padding(false)
    }

    fn encode_with_padding(&self, pad_top_level: bool) -> Result<Vec<u8>> {
        let mut buf = match self {
            DtypeSpec::F32 | DtypeSpec::F64 => self.encode_floating_point(),
            DtypeSpec::FixedAsciiString { len, padding } => {
                Self::encode_fixed_string(*len, *padding, false)
            }
            DtypeSpec::FixedUtf8String { len, padding } => {
                Self::encode_fixed_string(*len, *padding, true)
            }
            DtypeSpec::VarLenUtf8String => Self::encode_vlen_utf8_string(),
            DtypeSpec::Compound { size, fields } => Self::encode_compound(*size, fields)?,
            DtypeSpec::Enum { base, members } => Self::encode_enum(base, members)?,
            DtypeSpec::Opaque { size, tag } => Self::encode_opaque(*size, tag)?,
            DtypeSpec::Array { dims, base } => {
                Self::encode_array(self.checked_size()?, dims, base)?
            }
            _ => self.encode_fixed_point(),
        };

        if pad_top_level
            && matches!(
                self,
                DtypeSpec::Compound { .. }
                    | DtypeSpec::Enum { .. }
                    | DtypeSpec::Opaque { .. }
                    | DtypeSpec::Array { .. }
                    | DtypeSpec::VarLenUtf8String
            )
        {
            while buf.len() % 8 != 0 {
                buf.push(0);
            }
        }

        Ok(buf)
    }

    fn checked_size(&self) -> Result<u32> {
        match self {
            DtypeSpec::Array { dims, base } => {
                if dims.is_empty() {
                    return Err(Error::InvalidFormat(
                        "array datatype rank must be positive".into(),
                    ));
                }
                dims.iter().try_fold(base.checked_size()?, |acc, dim| {
                    if *dim == 0 {
                        return Err(Error::InvalidFormat(
                            "array datatype dimension must be positive".into(),
                        ));
                    }
                    acc.checked_mul(*dim).ok_or_else(|| {
                        Error::InvalidFormat("array datatype byte size overflow".into())
                    })
                })
            }
            DtypeSpec::FixedAsciiString { len, .. } | DtypeSpec::FixedUtf8String { len, .. } => {
                Ok(*len)
            }
            DtypeSpec::Compound { size, .. } | DtypeSpec::Opaque { size, .. } => Ok(*size),
            DtypeSpec::Enum { base, .. } => base.checked_size(),
            _ => Ok(self.size()),
        }
    }

    fn encode_floating_point(&self) -> Vec<u8> {
        let size = self.size();
        let mut buf = Vec::new();
        let class_and_version = 0x11u8;
        buf.push(class_and_version);
        if size == 4 {
            buf.extend_from_slice(&[0x20, 31, 0x00]);
            buf.extend_from_slice(&size.to_le_bytes());
            buf.extend_from_slice(&0u16.to_le_bytes());
            buf.extend_from_slice(&32u16.to_le_bytes());
            buf.push(23);
            buf.push(8);
            buf.push(0);
            buf.push(23);
            buf.extend_from_slice(&127u32.to_le_bytes());
        } else {
            buf.extend_from_slice(&[0x20, 63, 0x00]);
            buf.extend_from_slice(&size.to_le_bytes());
            buf.extend_from_slice(&0u16.to_le_bytes());
            buf.extend_from_slice(&64u16.to_le_bytes());
            buf.push(52);
            buf.push(11);
            buf.push(0);
            buf.push(52);
            buf.extend_from_slice(&1023u32.to_le_bytes());
        }
        buf
    }

    fn encode_fixed_string(len: u32, padding: u8, utf8: bool) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.push(0x13);
        buf.push((padding & 0x0f) | if utf8 { 0x10 } else { 0x00 });
        buf.extend_from_slice(&[0x00, 0x00]);
        buf.extend_from_slice(&len.to_le_bytes());
        buf
    }

    fn encode_vlen_utf8_string() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.push(0x19);
        buf.extend_from_slice(&[0x01, 0x01, 0x00]);
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&[
            0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00,
            0x00, 0x00,
        ]);
        buf
    }

    fn encode_compound(size: u32, fields: &[CompoundFieldSpec]) -> Result<Vec<u8>> {
        let mut buf = Self::encode_compound_header(size, fields.len())?;
        for field in fields {
            Self::encode_compound_field(&mut buf, field)?;
        }
        Ok(buf)
    }

    fn encode_compound_header(size: u32, field_count: usize) -> Result<Vec<u8>> {
        let field_count = u16::try_from(field_count).map_err(|_| {
            Error::InvalidFormat("compound datatype member count exceeds u16".into())
        })?;
        let mut buf = Vec::new();
        buf.push(0x16);
        buf.extend_from_slice(&field_count.to_le_bytes());
        buf.push(0);
        buf.extend_from_slice(&size.to_le_bytes());
        Ok(buf)
    }

    fn encode_compound_field(buf: &mut Vec<u8>, field: &CompoundFieldSpec) -> Result<()> {
        validate_dtype_name(&field.name, "compound datatype member name")?;
        let name_start = buf.len();
        buf.extend_from_slice(field.name.as_bytes());
        buf.push(0);
        let padded_name_len = (buf.len() - name_start + 7) & !7;
        while buf.len() < name_start + padded_name_len {
            buf.push(0);
        }
        buf.extend_from_slice(&field.offset.to_le_bytes());
        buf.extend_from_slice(&[0; 28]);
        buf.extend_from_slice(&field.dtype.encode_embedded()?);
        Ok(())
    }

    fn encode_enum(base: &DtypeSpec, members: &[(String, u64)]) -> Result<Vec<u8>> {
        let base_bytes = base.encode_embedded()?;
        let base_size = base.checked_size()?;
        let mut buf = Self::encode_enum_header(base_size, members.len(), &base_bytes)?;
        Self::encode_enum_names(&mut buf, members)?;
        let base_size = usize::try_from(base_size)
            .map_err(|_| Error::InvalidFormat("enum datatype base size exceeds usize".into()))?;
        Self::encode_enum_values(&mut buf, base_size, members);
        Ok(buf)
    }

    fn encode_enum_header(
        base_size: u32,
        member_count: usize,
        base_bytes: &[u8],
    ) -> Result<Vec<u8>> {
        let member_count = u16::try_from(member_count)
            .map_err(|_| Error::InvalidFormat("enum datatype member count exceeds u16".into()))?;
        let mut buf = Vec::new();
        buf.push(0x18);
        buf.extend_from_slice(&member_count.to_le_bytes());
        buf.push(0);
        buf.extend_from_slice(&base_size.to_le_bytes());
        buf.extend_from_slice(base_bytes);
        Ok(buf)
    }

    fn encode_enum_names(buf: &mut Vec<u8>, members: &[(String, u64)]) -> Result<()> {
        for (name, _) in members {
            validate_dtype_name(name, "enum datatype member name")?;
            Self::encode_padded_name(buf, name);
        }
        Ok(())
    }

    fn encode_enum_values(buf: &mut Vec<u8>, value_size: usize, members: &[(String, u64)]) {
        for (_, value) in members {
            let encoded = value.to_le_bytes();
            buf.extend_from_slice(&encoded[..value_size.min(encoded.len())]);
            if value_size > encoded.len() {
                buf.resize(buf.len() + (value_size - encoded.len()), 0);
            }
        }
    }

    fn encode_opaque(size: u32, tag: &str) -> Result<Vec<u8>> {
        validate_dtype_name(tag, "opaque datatype tag")?;
        let mut buf = Self::encode_opaque_header(size, tag)?;
        buf.extend_from_slice(tag.as_bytes());
        buf.push(0);
        while buf.len() % 8 != 0 {
            buf.push(0);
        }
        Ok(buf)
    }

    fn encode_opaque_header(size: u32, tag: &str) -> Result<Vec<u8>> {
        let mut buf = Vec::new();
        buf.push(0x15);
        let tag_with_null = tag
            .len()
            .checked_add(1)
            .ok_or_else(|| Error::InvalidFormat("opaque datatype tag length overflow".into()))?;
        let padded_tag_len = tag_with_null
            .checked_add(7)
            .ok_or_else(|| Error::InvalidFormat("opaque datatype tag length overflow".into()))?
            & !7;
        let padded_tag_len = u8::try_from(padded_tag_len).map_err(|_| {
            Error::InvalidFormat("opaque datatype tag padded length exceeds u8".into())
        })?;
        buf.extend_from_slice(&[padded_tag_len, 0x00, 0x00]);
        buf.extend_from_slice(&size.to_le_bytes());
        Ok(buf)
    }

    fn encode_array(size: u32, dims: &[u32], base: &DtypeSpec) -> Result<Vec<u8>> {
        let mut buf = Self::encode_array_header(size, dims)?;
        buf.extend_from_slice(&base.encode_embedded()?);
        Ok(buf)
    }

    fn encode_array_header(size: u32, dims: &[u32]) -> Result<Vec<u8>> {
        if dims.is_empty() {
            return Err(Error::InvalidFormat(
                "array datatype rank must be positive".into(),
            ));
        }
        let rank = u8::try_from(dims.len())
            .map_err(|_| Error::InvalidFormat("array datatype rank exceeds u8".into()))?;
        if dims.contains(&0) {
            return Err(Error::InvalidFormat(
                "array datatype dimension must be positive".into(),
            ));
        }
        let mut buf = Vec::new();
        buf.push(0x4a);
        buf.extend_from_slice(&[0x00, 0x00, 0x00]);
        buf.extend_from_slice(&size.to_le_bytes());
        buf.push(rank);
        for dim in dims {
            buf.extend_from_slice(&dim.to_le_bytes());
        }
        Ok(buf)
    }

    fn encode_fixed_point(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        let size = self.size();
        let is_signed = matches!(
            self,
            DtypeSpec::I8 | DtypeSpec::I16 | DtypeSpec::I32 | DtypeSpec::I64 | DtypeSpec::I128
        );
        let bf0 = if is_signed { 0x08u8 } else { 0x00u8 };
        buf.push(0x10u8);
        buf.extend_from_slice(&[bf0, 0x00, 0x00]);
        buf.extend_from_slice(&size.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        let bit_precision = u16::try_from(size).unwrap_or(u16::MAX).saturating_mul(8);
        buf.extend_from_slice(&bit_precision.to_le_bytes());
        buf
    }

    fn encode_padded_name(buf: &mut Vec<u8>, name: &str) {
        let name_start = buf.len();
        buf.extend_from_slice(name.as_bytes());
        buf.push(0);
        let padded_name_len = (buf.len() - name_start + 7) & !7;
        while buf.len() < name_start + padded_name_len {
            buf.push(0);
        }
    }
}

fn validate_dtype_name(name: &str, context: &str) -> Result<()> {
    if name.as_bytes().contains(&0) {
        return Err(Error::InvalidFormat(format!("{context} contains NUL byte")));
    }
    Ok(())
}

fn link_name_size_flag(name_len: usize) -> Result<u8> {
    if u8::try_from(name_len).is_ok() {
        Ok(0)
    } else if u16::try_from(name_len).is_ok() {
        Ok(1)
    } else if u32::try_from(name_len).is_ok() {
        Ok(2)
    } else {
        Err(Error::InvalidFormat(format!(
            "link name is {name_len} bytes, maximum is {}",
            u32::MAX
        )))
    }
}

fn encode_link_name_len(out: &mut Vec<u8>, name_len: usize, size_flag: u8) -> Result<()> {
    match size_flag {
        0 => out
            .push(u8::try_from(name_len).map_err(|_| {
                Error::InvalidFormat("link name length exceeds u8 encoding".into())
            })?),
        1 => out.extend_from_slice(
            &u16::try_from(name_len)
                .map_err(|_| Error::InvalidFormat("link name length exceeds u16 encoding".into()))?
                .to_le_bytes(),
        ),
        2 => out.extend_from_slice(
            &u32::try_from(name_len)
                .map_err(|_| Error::InvalidFormat("link name length exceeds u32 encoding".into()))?
                .to_le_bytes(),
        ),
        _ => {
            return Err(Error::InvalidFormat(
                "link name length size flag is invalid".into(),
            ))
        }
    }
    Ok(())
}

fn append_encoded_addr(out: &mut Vec<u8>, value: u64, sizeof_addr: u8) -> Result<()> {
    if sizeof_addr == 0 || sizeof_addr > 8 {
        return Err(Error::InvalidFormat(format!(
            "address field width {sizeof_addr} is invalid"
        )));
    }
    out.extend_from_slice(&value.to_le_bytes()[..usize::from(sizeof_addr)]);
    Ok(())
}

fn append_encoded_size(out: &mut Vec<u8>, value: u64, sizeof_size: u8) -> Result<()> {
    if sizeof_size == 0 || sizeof_size > 8 {
        return Err(Error::InvalidFormat(format!(
            "size field width {sizeof_size} is invalid"
        )));
    }
    out.extend_from_slice(&value.to_le_bytes()[..usize::from(sizeof_size)]);
    Ok(())
}

/// Encode a dataspace message.
fn encode_dataspace(shape: &[u64]) -> Result<Vec<u8>> {
    if shape.len() > MAX_DATASPACE_RANK {
        return Err(Error::InvalidFormat(format!(
            "dataspace rank {} exceeds supported maximum {MAX_DATASPACE_RANK}",
            shape.len()
        )));
    }
    encode_dataspace_impl(shape, None)
}

fn encode_dataspace_for_spec(spec: &DatasetSpec<'_>) -> Result<Vec<u8>> {
    if spec.shape.len() > MAX_DATASPACE_RANK {
        return Err(Error::InvalidFormat(format!(
            "dataspace rank {} exceeds supported maximum {MAX_DATASPACE_RANK}",
            spec.shape.len()
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
    encode_dataspace_impl(spec.shape, spec.max_shape)
}

fn encode_dataspace_impl(shape: &[u64], max_shape: Option<&[u64]>) -> Result<Vec<u8>> {
    let mut buf = Vec::new();

    if shape.is_empty() {
        // Scalar
        buf.push(2); // version 2
        buf.push(0); // ndims
        buf.push(0); // flags
        buf.push(0); // type = scalar
    } else {
        let ndims = u8::try_from(shape.len())
            .map_err(|_| Error::InvalidFormat("dataspace rank exceeds u8".into()))?;
        buf.push(2); // version 2
        buf.push(ndims); // ndims
        buf.push(if max_shape.is_some() { 0x01 } else { 0 }); // flags
        buf.push(1); // type = simple
        for &d in shape {
            buf.extend_from_slice(&d.to_le_bytes());
        }
        if let Some(max_shape) = max_shape {
            for &d in max_shape {
                buf.extend_from_slice(&d.to_le_bytes());
            }
        }
    }

    Ok(buf)
}

/// Encode a dense hard link message. Libhdf5 stores ASCII dense-link
/// names without the optional character-set field when possible.
fn encode_dense_link_message(name: &str, target_addr: u64, sizeof_addr: u8) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    let name_bytes = name.as_bytes();
    buf.push(1);
    let size_flag = link_name_size_flag(name_bytes.len())?;
    buf.push(size_flag);
    encode_link_name_len(&mut buf, name_bytes.len(), size_flag)?;
    buf.extend_from_slice(name_bytes);
    append_encoded_addr(&mut buf, target_addr, sizeof_addr)?;
    Ok(buf)
}

/// Encode a link message (v1, hard link).
fn encode_link_message(name: &str, target_addr: u64, sizeof_addr: u8) -> Result<Vec<u8>> {
    let mut buf = Vec::new();

    let name_bytes = name.as_bytes();
    let size_flag = link_name_size_flag(name_bytes.len())?;

    // Version
    buf.push(1);

    // Flags: size_flag | has_char_encoding(0x10)
    let flags = size_flag | 0x10;
    buf.push(flags);

    // Character encoding: UTF-8 = 1
    buf.push(1);

    // Name length
    encode_link_name_len(&mut buf, name_bytes.len(), size_flag)?;

    // Name
    buf.extend_from_slice(name_bytes);

    // Hard link target address
    append_encoded_addr(&mut buf, target_addr, sizeof_addr)?;

    Ok(buf)
}

/// Encode a soft link message (v1).
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

    buf.push(1); // version
    buf.push(size_flag | 0x08 | 0x10); // flags: size_flag + has_link_type + has_char_encoding

    buf.push(1); // link type = soft
    buf.push(1); // char encoding = UTF-8

    encode_link_name_len(&mut buf, name_bytes.len(), size_flag)?;
    buf.extend_from_slice(name_bytes);

    // Soft link value: target_length(2) + target_path
    buf.extend_from_slice(&target_len.to_le_bytes());
    buf.extend_from_slice(target_bytes);

    Ok(buf)
}

/// Encode an external link message (v1).
fn encode_external_link_message(name: &str, filename: &str, obj_path: &str) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    let name_bytes = name.as_bytes();
    let size_flag = link_name_size_flag(name_bytes.len())?;

    buf.push(1); // version
    buf.push(size_flag | 0x08 | 0x10); // flags: size_flag + has_link_type + has_char_encoding

    buf.push(64); // link type = external
    buf.push(1); // char encoding = UTF-8

    encode_link_name_len(&mut buf, name_bytes.len(), size_flag)?;
    buf.extend_from_slice(name_bytes);

    // External link value: info_length(2) + version(1) + filename(null-term) + obj_path(null-term)
    // Version 0: no flags byte
    let info_len = 1usize
        .checked_add(filename.len())
        .and_then(|len| len.checked_add(1))
        .and_then(|len| len.checked_add(obj_path.len()))
        .and_then(|len| len.checked_add(1))
        .ok_or_else(|| Error::InvalidFormat("external link info length overflow".into()))?;
    let info_len = u16::try_from(info_len).map_err(|_| {
        Error::InvalidFormat(format!(
            "external link info is {info_len} bytes, maximum is {}",
            u16::MAX
        ))
    })?;
    buf.extend_from_slice(&info_len.to_le_bytes());
    buf.push(0); // ext version = 0 (no flags byte)
    buf.extend_from_slice(filename.as_bytes());
    buf.push(0); // null terminator
    buf.extend_from_slice(obj_path.as_bytes());
    buf.push(0); // null terminator

    Ok(buf)
}

/// Encode a data layout message (v3, contiguous).
fn encode_contiguous_layout(
    data_addr: u64,
    data_size: u64,
    sizeof_addr: u8,
    sizeof_size: u8,
) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    buf.push(3); // version 3
    buf.push(1); // layout class = contiguous

    append_encoded_addr(&mut buf, data_addr, sizeof_addr)?;

    append_encoded_size(&mut buf, data_size, sizeof_size)?;

    Ok(buf)
}

/// Encode a data layout message (v3, chunked).
fn encode_chunked_layout_v3(
    btree_addr: u64,
    chunk_dims: &[u64],
    element_size: u32,
    sizeof_addr: u8,
) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    buf.push(3); // version 3
    buf.push(2); // layout class = chunked

    // ndims = chunk_dims.len() + 1 (extra dim for element size)
    let ndims = chunk_dims
        .len()
        .checked_add(1)
        .ok_or_else(|| Error::InvalidFormat("chunked layout rank overflow".into()))?;
    let ndims = u8::try_from(ndims)
        .map_err(|_| Error::InvalidFormat("chunked layout rank exceeds u8".into()))?;
    buf.push(ndims);

    // B-tree address
    append_encoded_addr(&mut buf, btree_addr, sizeof_addr)?;

    // Chunk dimensions (each 4 bytes) + element size as last dim
    for &d in chunk_dims {
        let dim = u32::try_from(d)
            .map_err(|_| Error::InvalidFormat("chunk dimension exceeds u32".into()))?;
        if dim == 0 {
            return Err(Error::InvalidFormat(
                "chunk dimension must be positive".into(),
            ));
        }
        buf.extend_from_slice(&dim.to_le_bytes());
    }
    buf.extend_from_slice(&element_size.to_le_bytes());

    Ok(buf)
}

/// Encode a filter pipeline message.
fn encode_filter_pipeline(
    compression_level: Option<u32>,
    shuffle: bool,
    fletcher32: bool,
) -> Result<Vec<u8>> {
    let mut filters = Vec::new();

    if shuffle {
        filters.push((2u16, Vec::<u32>::new())); // SHUFFLE, no params
    }
    if let Some(level) = compression_level {
        filters.push((1u16, vec![level])); // DEFLATE, 1 param = level
    }
    if fletcher32 {
        filters.push((3u16, Vec::<u32>::new())); // FLETCHER32, no params
    }

    if filters.is_empty() {
        return Ok(Vec::new());
    }

    let mut buf = Vec::new();
    buf.push(2); // version 2
    buf.push(
        u8::try_from(filters.len())
            .map_err(|_| Error::InvalidFormat("filter count exceeds u8".into()))?,
    ); // number of filters

    for (id, params) in &filters {
        buf.extend_from_slice(&id.to_le_bytes()); // filter ID
                                                  // v2: skip name_length for known filter IDs (< 256)
        buf.extend_from_slice(&0u16.to_le_bytes()); // flags
        buf.extend_from_slice(
            &u16::try_from(params.len())
                .map_err(|_| {
                    Error::InvalidFormat("filter client-data value count exceeds u16".into())
                })?
                .to_le_bytes(),
        ); // number of client data values
        for &p in params {
            buf.extend_from_slice(&p.to_le_bytes());
        }
    }

    Ok(buf)
}

/// Encode an attribute message (v3).
fn encode_attribute_message(
    name: &str,
    dtype: &DtypeSpec,
    shape: &[u64],
    data: &[u8],
) -> Result<Vec<u8>> {
    let mut buf = Vec::new();

    validate_attr_payload(name, dtype, shape, data)?;
    let dtype_bytes = dtype.encode()?;
    let ds_bytes = encode_dataspace(shape)?;
    let name_bytes = name.as_bytes();
    let name_with_null = name_bytes.len() + 1; // include null terminator
    let name_with_null = u16::try_from(name_with_null).map_err(|_| {
        Error::InvalidFormat(format!(
            "attribute name encodes to {} bytes, maximum is {}",
            name_bytes.len() + 1,
            u16::MAX
        ))
    })?;
    let dtype_len = u16::try_from(dtype_bytes.len()).map_err(|_| {
        Error::InvalidFormat(format!(
            "attribute datatype message is {} bytes, maximum is {}",
            dtype_bytes.len(),
            u16::MAX
        ))
    })?;
    let ds_len = u16::try_from(ds_bytes.len()).map_err(|_| {
        Error::InvalidFormat(format!(
            "attribute dataspace message is {} bytes, maximum is {}",
            ds_bytes.len(),
            u16::MAX
        ))
    })?;

    buf.push(3); // version 3
    buf.push(0); // flags
    buf.extend_from_slice(&name_with_null.to_le_bytes());
    buf.extend_from_slice(&dtype_len.to_le_bytes());
    buf.extend_from_slice(&ds_len.to_le_bytes());
    buf.push(0); // character encoding: ASCII

    buf.extend_from_slice(name_bytes);
    buf.push(0); // null terminator

    buf.extend_from_slice(&dtype_bytes);
    buf.extend_from_slice(&ds_bytes);
    buf.extend_from_slice(data);

    Ok(buf)
}

fn encode_link_info_message(
    heap_addr: u64,
    name_btree_addr: u64,
    sizeof_addr: u8,
) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    buf.push(0);
    buf.push(0);
    append_encoded_addr(&mut buf, heap_addr, sizeof_addr)?;
    append_encoded_addr(&mut buf, name_btree_addr, sizeof_addr)?;
    Ok(buf)
}

fn encode_attr_info_message(
    heap_addr: u64,
    name_btree_addr: u64,
    sizeof_addr: u8,
) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    buf.push(0);
    buf.push(0);
    append_encoded_addr(&mut buf, heap_addr, sizeof_addr)?;
    append_encoded_addr(&mut buf, name_btree_addr, sizeof_addr)?;
    Ok(buf)
}

fn dense_name_hash(name: &str) -> u32 {
    crate::format::checksum::checksum_lookup3(name.as_bytes(), 0)
}

/// Build a v2 object header from a list of messages.
fn build_v2_object_header(messages: &[(u16, &[u8])], flags: u8) -> Result<Vec<u8>> {
    // Calculate chunk 0 data size
    let mut chunk_data_size: usize = 0;
    for (msg_type, data) in messages {
        if u8::try_from(*msg_type).is_err() {
            return Err(Error::InvalidFormat(format!(
                "object-header message type {msg_type:#06x} exceeds v2 compact encoding"
            )));
        }
        if u16::try_from(data.len()).is_err() {
            return Err(Error::InvalidFormat(format!(
                "object-header message {msg_type:#06x} is {} bytes, maximum is {}",
                data.len(),
                u16::MAX
            )));
        }
        // Message header: type(1) + size(2) + flags(1) = 4
        chunk_data_size = chunk_data_size
            .checked_add(4)
            .and_then(|size| size.checked_add(data.len()))
            .ok_or_else(|| Error::InvalidFormat("object-header chunk size overflow".into()))?;
    }
    if u32::try_from(chunk_data_size).is_err() {
        return Err(Error::InvalidFormat(format!(
            "object-header chunk is {chunk_data_size} bytes, maximum is {}",
            u32::MAX
        )));
    }

    // Determine chunk0 size encoding
    let (chunk0_flag, chunk0_bytes) = if chunk_data_size < 256 {
        (0u8, 1usize)
    } else if chunk_data_size < 65536 {
        (1u8, 2usize)
    } else {
        (2u8, 4usize)
    };

    let oh_flags = flags | chunk0_flag;

    let mut buf = Vec::new();

    // Magic
    buf.extend_from_slice(b"OHDR");
    // Version
    buf.push(2);
    // Flags
    buf.push(oh_flags);

    // Optional timestamps (if HDR_STORE_TIMES)
    if oh_flags & HDR_STORE_TIMES != 0 {
        let now = 0u32; // placeholder
        buf.extend_from_slice(&now.to_le_bytes()); // atime
        buf.extend_from_slice(&now.to_le_bytes()); // mtime
        buf.extend_from_slice(&now.to_le_bytes()); // ctime
        buf.extend_from_slice(&now.to_le_bytes()); // btime
    }

    // Chunk 0 data size
    match chunk0_bytes {
        1 => buf.push(
            u8::try_from(chunk_data_size)
                .map_err(|_| Error::InvalidFormat("object-header chunk size exceeds u8".into()))?,
        ),
        2 => buf.extend_from_slice(
            &u16::try_from(chunk_data_size)
                .map_err(|_| Error::InvalidFormat("object-header chunk size exceeds u16".into()))?
                .to_le_bytes(),
        ),
        4 => buf.extend_from_slice(
            &u32::try_from(chunk_data_size)
                .map_err(|_| Error::InvalidFormat("object-header chunk size exceeds u32".into()))?
                .to_le_bytes(),
        ),
        _ => unreachable!(),
    }

    // Messages
    for (msg_type, data) in messages {
        buf.push(
            u8::try_from(*msg_type).map_err(|_| {
                Error::InvalidFormat("object-header message type exceeds u8".into())
            })?,
        ); // type (1 byte in v2)
        buf.extend_from_slice(
            &u16::try_from(data.len())
                .map_err(|_| Error::InvalidFormat("object-header message size exceeds u16".into()))?
                .to_le_bytes(),
        ); // size
        let msg_flags = if *msg_type == MSG_GROUP_INFO { 0x01 } else { 0 };
        buf.push(msg_flags);
        buf.extend_from_slice(data);
    }

    // Checksum over everything so far
    let checksum = checksum_metadata(&buf);
    buf.extend_from_slice(&checksum.to_le_bytes());

    Ok(buf)
}

fn encode_fill_value_message(fill: Option<FillValueSpec<'_>>) -> Result<Vec<u8>> {
    let Some(fill) = fill else {
        return Ok(vec![3u8, 0x09]);
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
    let mut buf = vec![3u8];
    if let Some(value) = fill.value {
        flags |= 0x20;
        buf.push(flags);
        buf.extend_from_slice(
            &u32::try_from(value.len())
                .map_err(|_| Error::InvalidFormat("fill-value payload length exceeds u32".into()))?
                .to_le_bytes(),
        );
        buf.extend_from_slice(value);
    } else {
        flags |= 0x10;
        buf.push(flags);
    }
    Ok(buf)
}

fn encode_global_heap_collection(objects: &[Vec<u8>], sizeof_size: u8) -> Result<Vec<u8>> {
    if sizeof_size != 8 {
        return Err(Error::Unsupported(
            "global heap writer currently requires 8-byte size fields".into(),
        ));
    }
    if u16::try_from(objects.len()).is_err() {
        return Err(Error::InvalidFormat(
            "too many global heap objects for vlen string dataset".into(),
        ));
    }

    let mut buf = Vec::new();
    buf.extend_from_slice(b"GCOL");
    buf.push(1);
    buf.extend_from_slice(&[0; 3]);
    buf.extend_from_slice(&0u64.to_le_bytes());

    for (idx, object) in objects.iter().enumerate() {
        let object_size = u64::try_from(object.len()).map_err(|_| {
            Error::InvalidFormat("global heap object length does not fit in u64".into())
        })?;
        let padded_size = object_size
            .checked_add(7)
            .map(|size| size & !7)
            .ok_or_else(|| Error::InvalidFormat("global heap object size overflow".into()))?;
        let padded_len = usize::try_from(padded_size).map_err(|_| {
            Error::InvalidFormat("global heap padded object length does not fit in usize".into())
        })?;

        let object_index = u16::try_from(idx + 1)
            .map_err(|_| Error::InvalidFormat("global heap object index exceeds u16".into()))?;
        buf.extend_from_slice(&object_index.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&[0; 4]);
        buf.extend_from_slice(&object_size.to_le_bytes());
        buf.extend_from_slice(object);
        buf.resize(buf.len() + (padded_len - object.len()), 0);
    }

    let min_collection_size = 4096usize;
    let target_size = buf
        .len()
        .max(min_collection_size)
        .checked_add(4095)
        .map(|size| size & !4095)
        .ok_or_else(|| Error::InvalidFormat("global heap collection size overflow".into()))?;
    let free_size = target_size
        .checked_sub(buf.len())
        .ok_or_else(|| Error::InvalidFormat("global heap free object size overflow".into()))?;
    buf.extend_from_slice(&0u16.to_le_bytes());
    buf.extend_from_slice(&0u16.to_le_bytes());
    buf.extend_from_slice(&[0; 4]);
    let free_size = u64::try_from(free_size)
        .map_err(|_| Error::InvalidFormat("global heap free object size exceeds u64".into()))?;
    buf.extend_from_slice(&free_size.to_le_bytes());
    buf.resize(target_size, 0);

    let collection_size = u64::try_from(buf.len()).map_err(|_| {
        Error::InvalidFormat("global heap collection length does not fit in u64".into())
    })?;
    checked_window_mut(&mut buf, 8, 8, "global heap collection size")?
        .copy_from_slice(&collection_size.to_le_bytes());
    Ok(buf)
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

fn ceil_div_nonzero_u64(value: u64, divisor: u64, context: &str) -> Result<u64> {
    if divisor == 0 {
        return Err(Error::InvalidFormat(format!("{context} divisor is zero")));
    }
    if value == 0 {
        return Ok(0);
    }
    value
        .checked_sub(1)
        .and_then(|v| v.checked_div(divisor))
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| Error::InvalidFormat(format!("{context} overflow")))
}

fn usize_from_u64_writer(value: u64, context: &str) -> Result<usize> {
    usize::try_from(value)
        .map_err(|_| Error::InvalidFormat(format!("{context} does not fit in usize")))
}

fn u64_from_usize_writer(value: usize, context: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| Error::InvalidFormat(format!("{context} exceeds u64")))
}

fn checked_next_power_of_two(value: usize, context: &str) -> Result<usize> {
    value
        .checked_next_power_of_two()
        .ok_or_else(|| Error::InvalidFormat(format!("{context} overflow")))
}

fn checked_window_mut<'a>(
    data: &'a mut [u8],
    pos: usize,
    len: usize,
    context: &str,
) -> Result<&'a mut [u8]> {
    let end = pos
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))?;
    data.get_mut(pos..end)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))
}

#[cfg(test)]
fn checked_window<'a>(data: &'a [u8], pos: usize, len: usize, context: &str) -> Result<&'a [u8]> {
    let end = pos
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))?;
    data.get(pos..end)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))
}

#[cfg(test)]
fn read_u64_le_at(data: &[u8], pos: usize, context: &str) -> Result<u64> {
    let bytes = checked_window(data, pos, 8, context)?;
    Ok(u64::from_le_bytes(bytes.try_into().map_err(|_| {
        Error::InvalidFormat(format!("{context} is truncated"))
    })?))
}

fn dense_record_hash(record: &[u8]) -> Result<u32> {
    let bytes = record
        .get(..4)
        .ok_or_else(|| Error::InvalidFormat("dense B-tree record hash is truncated".into()))?;
    let bytes: [u8; 4] = bytes
        .try_into()
        .map_err(|_| Error::InvalidFormat("dense B-tree record hash is truncated".into()))?;
    Ok(u32::from_le_bytes(bytes))
}

fn validate_chunked_dataset_spec(spec: &DatasetSpec<'_>, chunk_dims: &[u64]) -> Result<usize> {
    let ndims = spec.shape.len();
    if ndims == 0 {
        return Err(Error::InvalidFormat(
            "chunked scalar datasets are not supported".into(),
        ));
    }
    if chunk_dims.len() != ndims {
        return Err(Error::InvalidFormat(format!(
            "chunk dimension rank {} does not match dataset rank {}",
            chunk_dims.len(),
            ndims
        )));
    }
    if u8::try_from(chunk_dims.len() + 1).is_err() {
        return Err(Error::InvalidFormat(format!(
            "chunked layout rank {} exceeds encoded maximum {}",
            chunk_dims.len() + 1,
            u8::MAX
        )));
    }
    for (idx, &dim) in chunk_dims.iter().enumerate() {
        if dim == 0 {
            return Err(Error::InvalidFormat(format!(
                "chunk dimension {idx} is zero"
            )));
        }
        if u32::try_from(dim).is_err() {
            return Err(Error::InvalidFormat(format!(
                "chunk dimension {idx} exceeds 32-bit layout field"
            )));
        }
        if usize::try_from(dim).is_err() {
            return Err(Error::InvalidFormat(format!(
                "chunk dimension {idx} does not fit in usize"
            )));
        }
    }
    for (idx, &dim) in spec.shape.iter().enumerate() {
        if usize::try_from(dim).is_err() {
            return Err(Error::InvalidFormat(format!(
                "dataset dimension {idx} does not fit in usize"
            )));
        }
    }
    let chunk_elements_u64 = shape_element_count(chunk_dims)?;
    let chunk_elements = usize::try_from(chunk_elements_u64)
        .map_err(|_| Error::InvalidFormat("chunk element count exceeds usize".into()))?;
    let element_size = usize::try_from(spec.dtype.size())
        .map_err(|_| Error::InvalidFormat("dataset element size exceeds usize".into()))?;
    if u32::try_from(element_size).is_err() {
        return Err(Error::InvalidFormat(
            "dataset element size exceeds 32-bit chunk-layout field".into(),
        ));
    }
    let chunk_raw_bytes = chunk_elements
        .checked_mul(element_size)
        .ok_or_else(|| Error::InvalidFormat("chunk byte size overflow".into()))?;
    Ok(chunk_raw_bytes)
}

fn validate_deflate_level(compression_level: Option<u32>) -> Result<()> {
    if let Some(level) = compression_level {
        if level > 9 {
            return Err(Error::InvalidFormat(format!(
                "deflate compression level {level} exceeds maximum 9"
            )));
        }
    }
    Ok(())
}

fn validate_dataset_data_len(spec: &DatasetSpec<'_>) -> Result<()> {
    validate_dtype_spec(&spec.dtype)?;
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

fn validate_child_name(name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(Error::InvalidFormat("link name must not be empty".into()));
    }
    if name.contains('/') {
        return Err(Error::InvalidFormat(format!(
            "link name '{name}' must not contain '/'"
        )));
    }
    Ok(())
}

fn validate_attr_payload(name: &str, dtype: &DtypeSpec, shape: &[u64], data: &[u8]) -> Result<()> {
    validate_dtype_spec(dtype)?;
    if name.as_bytes().len() == usize::MAX {
        return Err(Error::InvalidFormat(
            "attribute name length overflow".into(),
        ));
    }
    let dtype_size = usize::try_from(dtype.size())
        .map_err(|_| Error::InvalidFormat("attribute datatype size exceeds usize".into()))?;
    if dtype_size == 0 {
        return Err(Error::InvalidFormat(
            "attribute datatype size must be nonzero".into(),
        ));
    }
    let expected_count = shape_element_count(shape)?;
    let expected_bytes = usize::try_from(expected_count)
        .map_err(|_| Error::InvalidFormat("attribute element count exceeds usize".into()))?
        .checked_mul(dtype_size)
        .ok_or_else(|| Error::InvalidFormat("attribute byte size overflow".into()))?;
    if expected_bytes != data.len() {
        return Err(Error::InvalidFormat(format!(
            "attribute byte length {} does not match shape element count {expected_count} * datatype size {dtype_size}",
            data.len()
        )));
    }
    Ok(())
}

fn validate_unique_attr_names(attrs: &[AttrSpec<'_>]) -> Result<()> {
    let mut names = HashSet::with_capacity(attrs.len());
    for attr in attrs {
        if !names.insert(attr.name) {
            return Err(Error::InvalidFormat(format!(
                "attribute '{}' already exists",
                attr.name
            )));
        }
    }
    Ok(())
}

fn validate_dtype_spec(dtype: &DtypeSpec) -> Result<()> {
    match dtype {
        DtypeSpec::Compound { size, fields } => {
            if u16::try_from(fields.len()).is_err() {
                return Err(Error::InvalidFormat(format!(
                    "compound datatype has {} fields, maximum is {}",
                    fields.len(),
                    u16::MAX
                )));
            }
            for field in fields {
                let field_end = field
                    .offset
                    .checked_add(field.dtype.size())
                    .ok_or_else(|| Error::InvalidFormat("compound field offset overflow".into()))?;
                if field_end > *size {
                    return Err(Error::InvalidFormat(format!(
                        "compound field '{}' ends at byte {field_end}, beyond compound size {size}",
                        field.name
                    )));
                }
                validate_dtype_spec(&field.dtype)?;
            }
        }
        DtypeSpec::Enum { base, members } => {
            if u16::try_from(members.len()).is_err() {
                return Err(Error::InvalidFormat(format!(
                    "enum datatype has {} members, maximum is {}",
                    members.len(),
                    u16::MAX
                )));
            }
            validate_dtype_spec(base)?;
            if !matches!(
                base.as_ref(),
                DtypeSpec::I8
                    | DtypeSpec::I16
                    | DtypeSpec::I32
                    | DtypeSpec::I64
                    | DtypeSpec::U8
                    | DtypeSpec::U16
                    | DtypeSpec::U32
                    | DtypeSpec::U64
            ) {
                return Err(Error::Unsupported(
                    "enum writer supports only integer base datatypes up to 8 bytes".into(),
                ));
            }
        }
        DtypeSpec::Opaque { tag, .. } => {
            let padded_tag_len = tag
                .len()
                .checked_add(1)
                .and_then(|len| len.checked_add(7))
                .map(|len| len & !7)
                .ok_or_else(|| Error::InvalidFormat("opaque tag length overflow".into()))?;
            if u8::try_from(padded_tag_len).is_err() {
                return Err(Error::InvalidFormat(format!(
                    "opaque tag encodes to {padded_tag_len} bytes, maximum is {}",
                    u8::MAX
                )));
            }
        }
        DtypeSpec::Array { dims, base } => {
            if dims.is_empty() {
                return Err(Error::InvalidFormat(
                    "array datatype must have at least one dimension".into(),
                ));
            }
            if u8::try_from(dims.len()).is_err() {
                return Err(Error::InvalidFormat(format!(
                    "array datatype rank {} exceeds maximum {}",
                    dims.len(),
                    u8::MAX
                )));
            }
            for (idx, &dim) in dims.iter().enumerate() {
                if dim == 0 {
                    return Err(Error::InvalidFormat(format!(
                        "array datatype dimension {idx} is zero"
                    )));
                }
            }
            let _size = dtype.checked_size()?;
            validate_dtype_spec(base)?;
        }
        _ => {}
    }
    Ok(())
}

impl<W: Write + Seek> HdfFileWriter<W> {
    /// Create a new HDF5 file writer.
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            allocator: FileAllocator::new(0),
            sizeof_addr: 8,
            sizeof_size: 8,
            groups: HashMap::new(),
            links: Vec::new(),
            hard_links: Vec::new(),
            pending_root_attrs: Vec::new(),
            pending_root_attr_specs: Vec::new(),
            pending_group_attr_specs: HashMap::new(),
            special_links: Vec::new(),
        }
    }

    /// Write the initial file structure: superblock placeholder.
    /// Call finalize() when done to write the superblock with correct EOF.
    pub fn begin(&mut self) -> Result<()> {
        // Reserve space for superblock (v2 with 8-byte addresses: 48 bytes)
        let sb_size = 4u64
            .checked_mul(u64::from(self.sizeof_addr))
            .and_then(|value| value.checked_add(16))
            .ok_or_else(|| Error::InvalidFormat("superblock placeholder size overflow".into()))?;
        self.allocator = FileAllocator::new(sb_size);

        // Write placeholder bytes for superblock
        let zeros = vec![
            0u8;
            usize::try_from(sb_size).map_err(|_| {
                Error::InvalidFormat("superblock placeholder size exceeds usize".into())
            })?
        ];
        self.write_at(0, &zeros)?;

        Ok(())
    }

    fn ensure_child_name_available(&self, parent: &str, name: &str) -> Result<()> {
        validate_child_name(name)?;
        if self.child_name_exists(parent, name) {
            return Err(Error::InvalidFormat(format!(
                "link '{name}' already exists in group '{parent}'"
            )));
        }
        Ok(())
    }

    fn child_name_exists(&self, parent: &str, name: &str) -> bool {
        let path = child_path(parent, name);
        self.groups.contains_key(&path)
            || self
                .links
                .iter()
                .any(|(link_parent, link_name, _)| link_parent == parent && link_name == name)
            || self
                .hard_links
                .iter()
                .any(|(link_parent, link_name, _)| link_parent == parent && link_name == name)
            || self
                .special_links
                .iter()
                .any(|(link_parent, link_name, _)| link_parent == parent && link_name == name)
    }

    /// Write an empty group object header (will be rewritten with links in finalize).
    fn write_group_object_header(&mut self, extra_messages: &[(u16, &[u8])]) -> Result<u64> {
        let messages: Vec<(u16, &[u8])> = extra_messages.to_vec();
        let oh_bytes = build_v2_object_header(&messages, 0)?;
        let oh_addr = self.allocator.allocate(
            u64_from_usize_writer(oh_bytes.len(), "object header size")?,
            8,
        );
        self.write_at(oh_addr, &oh_bytes)?;
        Ok(oh_addr)
    }

    /// Create the root group.
    pub fn create_root_group(&mut self) -> Result<u64> {
        let addr = self.write_group_object_header(&[])?;
        self.groups.insert("/".to_string(), addr);
        Ok(addr)
    }

    /// Create a sub-group.
    pub fn create_group(&mut self, parent: &str, name: &str) -> Result<u64> {
        self.ensure_child_name_available(parent, name)?;
        let addr = self.write_group_object_header(&[])?;
        let full_path = if parent == "/" {
            format!("/{name}")
        } else {
            format!("{parent}/{name}")
        };
        self.groups.insert(full_path.clone(), addr);
        self.links
            .push((parent.to_string(), name.to_string(), addr));
        Ok(addr)
    }

    /// Create a soft link in a group.
    pub fn create_soft_link(&mut self, parent: &str, name: &str, target_path: &str) -> Result<()> {
        self.ensure_child_name_available(parent, name)?;
        let msg = encode_soft_link_message(name, target_path)?;
        self.special_links
            .push((parent.to_string(), name.to_string(), msg));
        Ok(())
    }

    /// Create an external link in a group.
    pub fn create_external_link(
        &mut self,
        parent: &str,
        name: &str,
        filename: &str,
        obj_path: &str,
    ) -> Result<()> {
        self.ensure_child_name_available(parent, name)?;
        let msg = encode_external_link_message(name, filename, obj_path)?;
        self.special_links
            .push((parent.to_string(), name.to_string(), msg));
        Ok(())
    }

    /// Create a hard-link alias in a group.
    pub fn create_hard_link(&mut self, parent: &str, name: &str, target_path: &str) -> Result<()> {
        self.ensure_child_name_available(parent, name)?;
        self.object_addr_for_path(target_path).ok_or_else(|| {
            Error::InvalidFormat(format!("hard link target '{target_path}' not found"))
        })?;
        self.hard_links.push((
            parent.to_string(),
            name.to_string(),
            normalize_object_path(target_path),
        ));
        Ok(())
    }

    /// Create a dataset with compact storage (data embedded in the object header).
    /// Best for small datasets (< ~64KB).
    pub fn create_compact_dataset(&mut self, parent: &str, spec: &DatasetSpec) -> Result<u64> {
        self.create_compact_dataset_with_fill(parent, spec, None)
    }

    /// Create a compact dataset with attributes.
    pub fn create_compact_dataset_with_attrs(
        &mut self,
        parent: &str,
        spec: &DatasetSpec,
        attrs: &[AttrSpec],
    ) -> Result<u64> {
        self.create_compact_dataset_with_attrs_and_fill(parent, spec, attrs, None)
    }

    pub fn create_compact_dataset_with_attrs_and_fill(
        &mut self,
        parent: &str,
        spec: &DatasetSpec,
        attrs: &[AttrSpec],
        fill: Option<FillValueSpec<'_>>,
    ) -> Result<u64> {
        self.ensure_child_name_available(parent, spec.name)?;
        validate_unique_attr_names(attrs)?;
        validate_dataset_data_len(spec)?;
        let dtype_bytes = spec.dtype.encode()?;
        let ds_bytes = encode_dataspace_for_spec(spec)?;
        let compact_data_len = u16::try_from(spec.data.len()).map_err(|_| {
            Error::InvalidFormat(format!(
                "compact dataset payload is {} bytes, maximum is {}",
                spec.data.len(),
                u16::MAX
            ))
        })?;

        let mut layout_bytes = Vec::new();
        layout_bytes.push(3);
        layout_bytes.push(0);
        layout_bytes.extend_from_slice(&compact_data_len.to_le_bytes());
        layout_bytes.extend_from_slice(spec.data);

        let fill_value_bytes = encode_fill_value_message(fill)?;
        let mut messages: Vec<(u16, Vec<u8>)> = vec![
            (MSG_DATASPACE, ds_bytes),
            (MSG_DATATYPE, dtype_bytes),
            (MSG_FILL_VALUE, fill_value_bytes),
            (MSG_LAYOUT, layout_bytes),
        ];

        if attrs.len() > 8 {
            let (heap_addr, btree_addr) = self.write_dense_attribute_storage(attrs)?;
            messages.push((
                MSG_ATTR_INFO,
                encode_attr_info_message(heap_addr, btree_addr, self.sizeof_addr)?,
            ));
        } else {
            for attr in attrs {
                messages.push((
                    MSG_ATTRIBUTE,
                    encode_attribute_message(attr.name, &attr.dtype, attr.shape, attr.data)?,
                ));
            }
        }

        let msg_refs: Vec<(u16, &[u8])> =
            messages.iter().map(|(t, d)| (*t, d.as_slice())).collect();
        let oh_bytes = build_v2_object_header(&msg_refs, 0)?;
        let oh_addr = self.allocator.allocate(
            u64_from_usize_writer(oh_bytes.len(), "object header size")?,
            8,
        );
        self.write_at(oh_addr, &oh_bytes)?;

        self.links
            .push((parent.to_string(), spec.name.to_string(), oh_addr));

        Ok(oh_addr)
    }

    pub fn create_compact_dataset_with_fill(
        &mut self,
        parent: &str,
        spec: &DatasetSpec,
        fill: Option<FillValueSpec<'_>>,
    ) -> Result<u64> {
        self.ensure_child_name_available(parent, spec.name)?;
        validate_dataset_data_len(spec)?;
        let dtype_bytes = spec.dtype.encode()?;
        let ds_bytes = encode_dataspace_for_spec(spec)?;
        let compact_data_len = u16::try_from(spec.data.len()).map_err(|_| {
            Error::InvalidFormat(format!(
                "compact dataset payload is {} bytes, maximum is {}",
                spec.data.len(),
                u16::MAX
            ))
        })?;

        // Compact layout: version 3, class 0, size(2) + data
        let mut layout_bytes = Vec::new();
        layout_bytes.push(3); // version 3
        layout_bytes.push(0); // class = compact
        layout_bytes.extend_from_slice(&compact_data_len.to_le_bytes());
        layout_bytes.extend_from_slice(spec.data);

        let fill_value_bytes = encode_fill_value_message(fill)?;

        let messages: Vec<(u16, &[u8])> = vec![
            (MSG_DATASPACE, &ds_bytes),
            (MSG_DATATYPE, &dtype_bytes),
            (MSG_FILL_VALUE, &fill_value_bytes),
            (MSG_LAYOUT, &layout_bytes),
        ];

        let oh_bytes = build_v2_object_header(&messages, 0)?;
        let oh_addr = self.allocator.allocate(
            u64_from_usize_writer(oh_bytes.len(), "object header size")?,
            8,
        );
        self.write_at(oh_addr, &oh_bytes)?;

        self.links
            .push((parent.to_string(), spec.name.to_string(), oh_addr));

        Ok(oh_addr)
    }

    /// Create a dataset with contiguous storage.
    pub fn create_dataset(&mut self, parent: &str, spec: &DatasetSpec) -> Result<u64> {
        self.create_dataset_with_fill(parent, spec, None)
    }

    pub fn create_dataset_with_fill(
        &mut self,
        parent: &str,
        spec: &DatasetSpec,
        fill: Option<FillValueSpec<'_>>,
    ) -> Result<u64> {
        self.ensure_child_name_available(parent, spec.name)?;
        validate_dataset_data_len(spec)?;
        let dtype_bytes = spec.dtype.encode()?;
        let ds_bytes = encode_dataspace_for_spec(spec)?;

        // Allocate space for the data
        let data_size = u64_from_usize_writer(spec.data.len(), "dataset data size")?;
        let data_addr = if data_size > 0 {
            let addr = self.allocator.allocate(data_size, 8);
            self.write_at(addr, spec.data)?;
            addr
        } else {
            UNDEF_ADDR
        };

        let layout_bytes =
            encode_contiguous_layout(data_addr, data_size, self.sizeof_addr, self.sizeof_size)?;

        let fill_value_bytes = encode_fill_value_message(fill)?;

        let messages: Vec<(u16, &[u8])> = vec![
            (MSG_DATASPACE, &ds_bytes),
            (MSG_DATATYPE, &dtype_bytes),
            (MSG_FILL_VALUE, &fill_value_bytes),
            (MSG_LAYOUT, &layout_bytes),
        ];

        let oh_bytes = build_v2_object_header(&messages, 0)?;
        let oh_addr = self.allocator.allocate(
            u64_from_usize_writer(oh_bytes.len(), "object header size")?,
            8,
        );
        self.write_at(oh_addr, &oh_bytes)?;

        self.links
            .push((parent.to_string(), spec.name.to_string(), oh_addr));

        Ok(oh_addr)
    }

    /// Create a contiguous variable-length UTF-8 string dataset backed by a global heap.
    pub fn create_vlen_utf8_string_dataset(
        &mut self,
        parent: &str,
        name: &str,
        shape: &[u64],
        strings: &[&str],
    ) -> Result<u64> {
        self.create_vlen_utf8_string_dataset_with_attrs(parent, name, shape, strings, None, &[])
    }

    pub fn create_vlen_utf8_string_dataset_with_attrs(
        &mut self,
        parent: &str,
        name: &str,
        shape: &[u64],
        strings: &[&str],
        max_shape: Option<&[u64]>,
        attrs: &[AttrSpec],
    ) -> Result<u64> {
        self.ensure_child_name_available(parent, name)?;
        validate_unique_attr_names(attrs)?;

        let expected_count = shape_element_count(shape)?;
        let string_count = u64_from_usize_writer(strings.len(), "vlen string count")?;
        if expected_count != string_count {
            return Err(Error::InvalidFormat(format!(
                "vlen string data length {} does not match dataset shape element count {expected_count}",
                strings.len()
            )));
        }

        let mut heap_objects = Vec::new();
        let mut heap_indices = vec![0u32; strings.len()];
        for (pos, value) in strings.iter().enumerate().rev() {
            heap_objects.push(value.as_bytes().to_vec());
            heap_indices[pos] = u32::try_from(heap_objects.len())
                .map_err(|_| Error::InvalidFormat("global heap object index exceeds u32".into()))?;
        }

        let heap_bytes = if strings.is_empty() {
            None
        } else {
            Some(encode_global_heap_collection(
                &heap_objects,
                self.sizeof_size,
            )?)
        };

        let vlen_descriptor_size =
            usize::try_from(DtypeSpec::VarLenUtf8String.size()).map_err(|_| {
                Error::InvalidFormat("vlen string descriptor size exceeds usize".into())
            })?;
        let vlen_payload_size =
            strings
                .len()
                .checked_mul(vlen_descriptor_size)
                .ok_or_else(|| {
                    Error::InvalidFormat("vlen string descriptor payload size overflow".into())
                })?;

        let data_addr = if vlen_payload_size == 0 {
            UNDEF_ADDR
        } else {
            self.allocator.allocate(
                u64::try_from(vlen_payload_size).map_err(|_| {
                    Error::InvalidFormat("vlen string descriptor payload too large".into())
                })?,
                8,
            )
        };
        let heap_addr = if let Some(heap_bytes) = &heap_bytes {
            self.allocator.allocate(
                u64::try_from(heap_bytes.len())
                    .map_err(|_| Error::InvalidFormat("global heap collection too large".into()))?,
                8,
            )
        } else {
            UNDEF_ADDR
        };

        let mut data = Vec::with_capacity(vlen_payload_size);
        for (value, index) in strings.iter().zip(heap_indices) {
            let len = u32::try_from(value.len())
                .map_err(|_| Error::InvalidFormat("vlen string length exceeds u32".into()))?;
            data.extend_from_slice(&len.to_le_bytes());
            data.extend_from_slice(&heap_addr.to_le_bytes());
            data.extend_from_slice(&index.to_le_bytes());
        }

        if data_addr != UNDEF_ADDR {
            self.write_at(data_addr, &data)?;
        }
        if let Some(heap_bytes) = &heap_bytes {
            self.write_at(heap_addr, heap_bytes)?;
        }

        let dtype_bytes = DtypeSpec::VarLenUtf8String.encode()?;
        let spec = DatasetSpec {
            name,
            shape,
            max_shape,
            dtype: DtypeSpec::VarLenUtf8String,
            data: &data,
        };
        let ds_bytes = encode_dataspace_for_spec(&spec)?;
        let layout_bytes = encode_contiguous_layout(
            data_addr,
            u64::try_from(vlen_payload_size).map_err(|_| {
                Error::InvalidFormat("vlen string descriptor payload too large".into())
            })?,
            self.sizeof_addr,
            self.sizeof_size,
        )?;
        let fill_value_bytes = encode_fill_value_message(None)?;

        let mut messages: Vec<(u16, Vec<u8>)> = vec![
            (MSG_DATASPACE, ds_bytes),
            (MSG_DATATYPE, dtype_bytes),
            (MSG_FILL_VALUE, fill_value_bytes),
            (MSG_LAYOUT, layout_bytes),
        ];

        if attrs.len() > 8 {
            let (heap_addr, btree_addr) = self.write_dense_attribute_storage(attrs)?;
            messages.push((
                MSG_ATTR_INFO,
                encode_attr_info_message(heap_addr, btree_addr, self.sizeof_addr)?,
            ));
        } else {
            for attr in attrs {
                let attr_bytes =
                    encode_attribute_message(attr.name, &attr.dtype, attr.shape, attr.data)?;
                messages.push((MSG_ATTRIBUTE, attr_bytes));
            }
        }

        let msg_refs: Vec<(u16, &[u8])> =
            messages.iter().map(|(t, d)| (*t, d.as_slice())).collect();
        let oh_bytes = build_v2_object_header(&msg_refs, 0)?;
        let oh_addr = self.allocator.allocate(
            u64_from_usize_writer(oh_bytes.len(), "object header size")?,
            8,
        );
        self.write_at(oh_addr, &oh_bytes)?;

        self.links
            .push((parent.to_string(), name.to_string(), oh_addr));

        Ok(oh_addr)
    }

    /// Create a dataset with attributes.
    pub fn create_dataset_with_attrs(
        &mut self,
        parent: &str,
        spec: &DatasetSpec,
        attrs: &[AttrSpec],
    ) -> Result<u64> {
        self.create_dataset_with_attrs_and_fill(parent, spec, attrs, None)
    }

    pub fn create_dataset_with_attrs_and_fill(
        &mut self,
        parent: &str,
        spec: &DatasetSpec,
        attrs: &[AttrSpec],
        fill: Option<FillValueSpec<'_>>,
    ) -> Result<u64> {
        self.ensure_child_name_available(parent, spec.name)?;
        validate_unique_attr_names(attrs)?;
        validate_dataset_data_len(spec)?;
        let dtype_bytes = spec.dtype.encode()?;
        let ds_bytes = encode_dataspace_for_spec(spec)?;

        let data_size = u64_from_usize_writer(spec.data.len(), "dataset data size")?;
        let data_addr = if data_size > 0 {
            let addr = self.allocator.allocate(data_size, 8);
            self.write_at(addr, spec.data)?;
            addr
        } else {
            UNDEF_ADDR
        };

        let layout_bytes =
            encode_contiguous_layout(data_addr, data_size, self.sizeof_addr, self.sizeof_size)?;
        let fill_value_bytes = encode_fill_value_message(fill)?;

        let mut messages: Vec<(u16, Vec<u8>)> = vec![
            (MSG_DATASPACE, ds_bytes),
            (MSG_DATATYPE, dtype_bytes),
            (MSG_FILL_VALUE, fill_value_bytes),
            (MSG_LAYOUT, layout_bytes),
        ];

        if attrs.len() > 8 {
            let (heap_addr, btree_addr) = self.write_dense_attribute_storage(attrs)?;
            messages.push((
                MSG_ATTR_INFO,
                encode_attr_info_message(heap_addr, btree_addr, self.sizeof_addr)?,
            ));
        } else {
            for attr in attrs {
                let attr_bytes =
                    encode_attribute_message(attr.name, &attr.dtype, attr.shape, attr.data)?;
                messages.push((MSG_ATTRIBUTE, attr_bytes));
            }
        }

        let msg_refs: Vec<(u16, &[u8])> =
            messages.iter().map(|(t, d)| (*t, d.as_slice())).collect();
        let oh_bytes = build_v2_object_header(&msg_refs, 0)?;
        let oh_addr = self.allocator.allocate(
            u64_from_usize_writer(oh_bytes.len(), "object header size")?,
            8,
        );
        self.write_at(oh_addr, &oh_bytes)?;

        self.links
            .push((parent.to_string(), spec.name.to_string(), oh_addr));

        Ok(oh_addr)
    }

    /// Add attributes to the root group (call before finalize).
    pub fn set_root_attrs(&mut self, attrs: Vec<(u16, Vec<u8>)>) {
        // Store as pending attribute messages for the root group
        // These will be included when finalize rewrites the root group
        for (msg_type, data) in attrs {
            self.pending_root_attrs.push((msg_type, data));
        }
    }

    /// Create a root group attribute from spec.
    pub fn add_root_attr(&mut self, attr: &AttrSpec) -> Result<()> {
        if self
            .pending_root_attr_specs
            .iter()
            .any(|existing| existing.name == attr.name)
        {
            return Err(Error::InvalidFormat(format!(
                "attribute '{}' already exists",
                attr.name
            )));
        }
        self.pending_root_attr_specs.push(OwnedAttrSpec {
            name: attr.name.to_string(),
            shape: attr.shape.to_vec(),
            dtype: attr.dtype.clone(),
            data: attr.data.to_vec(),
        });
        Ok(())
    }

    /// Create a group attribute from spec.
    pub fn add_group_attr(&mut self, group_path: &str, attr: &AttrSpec) -> Result<()> {
        let attrs = self
            .pending_group_attr_specs
            .entry(group_path.to_string())
            .or_default();
        if attrs.iter().any(|existing| existing.name == attr.name) {
            return Err(Error::InvalidFormat(format!(
                "attribute '{}' already exists",
                attr.name
            )));
        }
        attrs.push(OwnedAttrSpec {
            name: attr.name.to_string(),
            shape: attr.shape.to_vec(),
            dtype: attr.dtype.clone(),
            data: attr.data.to_vec(),
        });
        Ok(())
    }

    /// Create a chunked dataset with optional compression.
    pub fn create_chunked_dataset(
        &mut self,
        parent: &str,
        spec: &DatasetSpec,
        chunk_dims: &[u64],
        compression_level: Option<u32>,
        shuffle: bool,
    ) -> Result<u64> {
        self.create_chunked_dataset_with_filters(
            parent,
            spec,
            chunk_dims,
            compression_level,
            shuffle,
            false,
        )
    }

    /// Create a chunked dataset with optional deflate/shuffle/Fletcher32 filters.
    pub fn create_chunked_dataset_with_filters(
        &mut self,
        parent: &str,
        spec: &DatasetSpec,
        chunk_dims: &[u64],
        compression_level: Option<u32>,
        shuffle: bool,
        fletcher32: bool,
    ) -> Result<u64> {
        self.create_chunked_dataset_with_filters_and_fill(
            parent,
            spec,
            chunk_dims,
            compression_level,
            shuffle,
            fletcher32,
            None,
        )
    }

    pub fn create_chunked_dataset_with_fill(
        &mut self,
        parent: &str,
        spec: &DatasetSpec,
        chunk_dims: &[u64],
        compression_level: Option<u32>,
        shuffle: bool,
        fill: Option<FillValueSpec<'_>>,
    ) -> Result<u64> {
        self.create_chunked_dataset_with_filters_and_fill(
            parent,
            spec,
            chunk_dims,
            compression_level,
            shuffle,
            false,
            fill,
        )
    }

    pub fn create_chunked_dataset_with_filters_and_fill(
        &mut self,
        parent: &str,
        spec: &DatasetSpec,
        chunk_dims: &[u64],
        compression_level: Option<u32>,
        shuffle: bool,
        fletcher32: bool,
        fill: Option<FillValueSpec<'_>>,
    ) -> Result<u64> {
        self.create_chunked_dataset_with_attrs_and_fill(
            parent,
            spec,
            chunk_dims,
            compression_level,
            shuffle,
            fletcher32,
            fill,
            &[],
        )
    }

    pub fn create_chunked_dataset_with_attrs_and_fill(
        &mut self,
        parent: &str,
        spec: &DatasetSpec,
        chunk_dims: &[u64],
        compression_level: Option<u32>,
        shuffle: bool,
        fletcher32: bool,
        fill: Option<FillValueSpec<'_>>,
        attrs: &[AttrSpec],
    ) -> Result<u64> {
        self.ensure_child_name_available(parent, spec.name)?;
        validate_unique_attr_names(attrs)?;
        validate_dataset_data_len(spec)?;
        let dtype_bytes = spec.dtype.encode()?;
        let ds_bytes = encode_dataspace_for_spec(spec)?;
        let element_size = usize::try_from(spec.dtype.size())
            .map_err(|_| Error::InvalidFormat("dataset element size exceeds usize".into()))?;
        let ndims = spec.shape.len();
        validate_deflate_level(compression_level)?;
        let chunk_raw_bytes = validate_chunked_dataset_spec(spec, chunk_dims)?;

        // Split data into chunks, apply filters, write each chunk

        // Calculate number of chunks per dimension
        let mut n_chunks_per_dim = Vec::with_capacity(ndims);
        for i in 0..ndims {
            let chunks = ceil_div_nonzero_u64(spec.shape[i], chunk_dims[i], "chunk count")?;
            n_chunks_per_dim.push(
                usize::try_from(chunks).map_err(|_| {
                    Error::InvalidFormat("chunk count does not fit in usize".into())
                })?,
            );
        }
        let total_chunks = n_chunks_per_dim.iter().try_fold(1usize, |acc, &count| {
            acc.checked_mul(count)
                .ok_or_else(|| Error::InvalidFormat("total chunk count overflow".into()))
        })?;

        // Write each chunk and collect (coords, addr, compressed_size)
        let mut chunk_entries: Vec<(Vec<u64>, u64, u32)> = Vec::new();

        for chunk_idx in 0..total_chunks {
            // Calculate chunk coordinates
            let mut coords = vec![0u64; ndims];
            let mut rem = chunk_idx;
            for d in (0..ndims).rev() {
                let chunk_coord = u64::try_from(rem % n_chunks_per_dim[d]).map_err(|_| {
                    Error::InvalidFormat("chunk coordinate index exceeds u64".into())
                })?;
                coords[d] = chunk_coord.checked_mul(chunk_dims[d]).ok_or_else(|| {
                    Error::InvalidFormat("chunk coordinate offset overflow".into())
                })?;
                rem /= n_chunks_per_dim[d];
            }

            // Extract chunk data from the source array
            let mut chunk_buf = vec![0u8; chunk_raw_bytes];
            self.extract_chunk(
                spec.data,
                spec.shape,
                &coords,
                chunk_dims,
                element_size,
                &mut chunk_buf,
            )?;

            // Apply filters in forward order
            let mut filtered = chunk_buf;
            if shuffle {
                filtered = crate::filters::shuffle::shuffle(&filtered, element_size)?;
            }
            if let Some(level) = compression_level {
                filtered = crate::filters::deflate::compress(&filtered, level)?;
            }
            if fletcher32 {
                filtered = crate::filters::fletcher32::append_checksum(&filtered)?;
            }

            let compressed_size = u32::try_from(filtered.len())
                .map_err(|_| Error::InvalidFormat("compressed chunk size exceeds u32".into()))?;
            let addr = self.allocator.allocate(
                u64::try_from(filtered.len()).map_err(|_| {
                    Error::InvalidFormat("compressed chunk size exceeds u64".into())
                })?,
                1,
            );
            self.write_at(addr, &filtered)?;

            chunk_entries.push((coords, addr, compressed_size));
        }

        // Write v1 B-tree for chunk index
        let element_size = u32::try_from(element_size)
            .map_err(|_| Error::InvalidFormat("chunk element size exceeds u32".into()))?;
        let btree_addr = self.write_chunk_btree_v1(&chunk_entries, ndims, element_size)?;

        // Encode layout message (v3 chunked)
        let layout_bytes =
            encode_chunked_layout_v3(btree_addr, chunk_dims, element_size, self.sizeof_addr)?;

        // Encode filter pipeline message
        let pipeline_bytes = encode_filter_pipeline(compression_level, shuffle, fletcher32)?;

        let fill_value_bytes = encode_fill_value_message(fill)?;

        let mut messages: Vec<(u16, Vec<u8>)> = vec![
            (MSG_DATASPACE, ds_bytes),
            (MSG_DATATYPE, dtype_bytes),
            (MSG_FILL_VALUE, fill_value_bytes),
            (MSG_LAYOUT, layout_bytes),
        ];
        if !pipeline_bytes.is_empty() {
            messages.push((MSG_FILTER_PIPELINE, pipeline_bytes));
        }

        if attrs.len() > 8 {
            let (heap_addr, btree_addr) = self.write_dense_attribute_storage(attrs)?;
            messages.push((
                MSG_ATTR_INFO,
                encode_attr_info_message(heap_addr, btree_addr, self.sizeof_addr)?,
            ));
        } else {
            for attr in attrs {
                messages.push((
                    MSG_ATTRIBUTE,
                    encode_attribute_message(attr.name, &attr.dtype, attr.shape, attr.data)?,
                ));
            }
        }

        let msg_refs: Vec<(u16, &[u8])> =
            messages.iter().map(|(t, d)| (*t, d.as_slice())).collect();
        let oh_bytes = build_v2_object_header(&msg_refs, 0)?;
        let oh_addr = self.allocator.allocate(
            u64_from_usize_writer(oh_bytes.len(), "object header size")?,
            8,
        );
        self.write_at(oh_addr, &oh_bytes)?;

        self.links
            .push((parent.to_string(), spec.name.to_string(), oh_addr));

        Ok(oh_addr)
    }

    /// Extract chunk data from a flat array.
    fn extract_chunk(
        &self,
        data: &[u8],
        shape: &[u64],
        chunk_start: &[u64],
        chunk_dims: &[u64],
        element_size: usize,
        out: &mut [u8],
    ) -> Result<()> {
        let ndims = shape.len();
        if chunk_start.len() != ndims || chunk_dims.len() != ndims {
            return Err(Error::InvalidFormat(
                "chunk rank does not match dataset rank".into(),
            ));
        }
        if element_size == 0 {
            return Err(Error::InvalidFormat("chunk element size is zero".into()));
        }
        if ndims == 1 {
            // Fast path for 1D
            let start = usize_from_u64_writer(chunk_start[0], "chunk start")?;
            let chunk_len = usize_from_u64_writer(chunk_dims[0], "chunk dimension")?;
            let data_len = usize_from_u64_writer(shape[0], "dataset dimension")?;
            let remaining = data_len.checked_sub(start).ok_or_else(|| {
                Error::InvalidFormat("chunk start exceeds dataset dimension".into())
            })?;
            let copy_len = chunk_len.min(remaining);
            let copy_bytes = copy_len
                .checked_mul(element_size)
                .ok_or_else(|| Error::InvalidFormat("chunk copy byte count overflow".into()))?;
            let src_start = start
                .checked_mul(element_size)
                .ok_or_else(|| Error::InvalidFormat("chunk source offset overflow".into()))?;
            let src_end = src_start
                .checked_add(copy_bytes)
                .ok_or_else(|| Error::InvalidFormat("chunk source offset overflow".into()))?;
            let src = data
                .get(src_start..src_end)
                .ok_or_else(|| Error::InvalidFormat("chunk source range exceeds data".into()))?;
            let dst = out.get_mut(..copy_bytes).ok_or_else(|| {
                Error::InvalidFormat("chunk destination range exceeds output".into())
            })?;
            dst.copy_from_slice(src);
        } else {
            // General N-D: iterate over elements
            let chunk_dim_usizes = chunk_dims
                .iter()
                .map(|&dim| usize_from_u64_writer(dim, "chunk dimension"))
                .collect::<Result<Vec<_>>>()?;
            let shape_usizes = shape
                .iter()
                .map(|&dim| usize_from_u64_writer(dim, "dataset dimension"))
                .collect::<Result<Vec<_>>>()?;
            let chunk_start_usizes = chunk_start
                .iter()
                .map(|&dim| usize_from_u64_writer(dim, "chunk start"))
                .collect::<Result<Vec<_>>>()?;
            let chunk_elements = chunk_dim_usizes.iter().try_fold(1usize, |acc, &dim| {
                acc.checked_mul(dim)
                    .ok_or_else(|| Error::InvalidFormat("chunk element count overflow".into()))
            })?;
            let mut idx = vec![0usize; ndims];

            for elem in 0..chunk_elements {
                // Convert linear index within chunk to N-D
                let mut rem = elem;
                for d in (0..ndims).rev() {
                    idx[d] = rem % chunk_dim_usizes[d];
                    rem /= chunk_dim_usizes[d];
                }

                // Global position
                let mut in_bounds = true;
                let mut src_linear = 0usize;
                let mut stride = 1usize;
                for d in (0..ndims).rev() {
                    let global = chunk_start_usizes[d].checked_add(idx[d]).ok_or_else(|| {
                        Error::InvalidFormat("chunk global coordinate overflow".into())
                    })?;
                    if global >= shape_usizes[d] {
                        in_bounds = false;
                        break;
                    }
                    let contribution = global.checked_mul(stride).ok_or_else(|| {
                        Error::InvalidFormat("chunk source linear offset overflow".into())
                    })?;
                    src_linear = src_linear.checked_add(contribution).ok_or_else(|| {
                        Error::InvalidFormat("chunk source linear offset overflow".into())
                    })?;
                    stride = stride.checked_mul(shape_usizes[d]).ok_or_else(|| {
                        Error::InvalidFormat("chunk source stride overflow".into())
                    })?;
                }

                if in_bounds {
                    let src_offset = src_linear.checked_mul(element_size).ok_or_else(|| {
                        Error::InvalidFormat("chunk source byte offset overflow".into())
                    })?;
                    let dst_offset = elem.checked_mul(element_size).ok_or_else(|| {
                        Error::InvalidFormat("chunk destination byte offset overflow".into())
                    })?;
                    let src_end = src_offset.checked_add(element_size).ok_or_else(|| {
                        Error::InvalidFormat("chunk source byte offset overflow".into())
                    })?;
                    let dst_end = dst_offset.checked_add(element_size).ok_or_else(|| {
                        Error::InvalidFormat("chunk destination byte offset overflow".into())
                    })?;
                    let src = data.get(src_offset..src_end).ok_or_else(|| {
                        Error::InvalidFormat("chunk source range exceeds data".into())
                    })?;
                    let dst = out.get_mut(dst_offset..dst_end).ok_or_else(|| {
                        Error::InvalidFormat("chunk destination range exceeds output".into())
                    })?;
                    dst.copy_from_slice(src);
                }
            }
        }
        Ok(())
    }

    /// Write a v1 B-tree leaf node for chunk index.
    fn write_chunk_btree_v1(
        &mut self,
        chunks: &[(Vec<u64>, u64, u32)],
        ndims: usize,
        element_size: u32,
    ) -> Result<u64> {
        let entries: Vec<ChunkBTreeEntry> = chunks
            .iter()
            .map(|(coords, addr, compressed_size)| ChunkBTreeEntry {
                coords: coords.clone(),
                chunk_size: *compressed_size,
                filter_mask: 0,
                child_addr: *addr,
            })
            .collect();
        self.write_chunk_btree_entries_v1(&entries, ndims, element_size)
    }

    fn write_chunk_btree_entries_v1(
        &mut self,
        entries: &[ChunkBTreeEntry],
        ndims: usize,
        element_size: u32,
    ) -> Result<u64> {
        let node_size = Self::chunk_btree_node_size(ndims, usize::from(self.sizeof_addr))?;
        let root_addr = self.allocator.allocate(
            u64_from_usize_writer(node_size, "chunk B-tree node size")?,
            8,
        );
        let final_coords = entries[entries.len() - 1].coords.clone();

        if entries.len() <= 64 {
            let leaf =
                self.encode_chunk_btree_node_v1(0, entries, &final_coords, ndims, element_size)?;
            self.write_at(root_addr, &leaf)?;
            return Ok(root_addr);
        }

        let leaf_count = entries.len().div_ceil(64);
        if leaf_count > 64 {
            return Err(Error::Unsupported(
                "writer chunk B-tree currently supports at most a two-level root".into(),
            ));
        }

        let mut root_entries = Vec::with_capacity(leaf_count);
        for leaf_entries in entries.chunks(64) {
            let leaf_addr = self.allocator.allocate(
                u64_from_usize_writer(node_size, "chunk B-tree node size")?,
                8,
            );
            let leaf_final_coords = leaf_entries[leaf_entries.len() - 1].coords.clone();
            let leaf = self.encode_chunk_btree_node_v1(
                0,
                leaf_entries,
                &leaf_final_coords,
                ndims,
                element_size,
            )?;
            self.write_at(leaf_addr, &leaf)?;
            root_entries.push(ChunkBTreeEntry {
                coords: leaf_entries[0].coords.clone(),
                chunk_size: leaf_entries[0].chunk_size,
                filter_mask: leaf_entries[0].filter_mask,
                child_addr: leaf_addr,
            });
        }

        let root =
            self.encode_chunk_btree_node_v1(1, &root_entries, &final_coords, ndims, element_size)?;
        self.write_at(root_addr, &root)?;
        Ok(root_addr)
    }

    fn chunk_btree_node_size(ndims: usize, sizeof_addr: usize) -> Result<usize> {
        let key_size = ndims
            .checked_add(1)
            .and_then(|value| value.checked_mul(8))
            .and_then(|value| value.checked_add(8))
            .ok_or_else(|| Error::InvalidFormat("chunk B-tree key size overflow".into()))?;
        let max_entries = 64usize;
        let header_size = sizeof_addr
            .checked_mul(2)
            .and_then(|value| value.checked_add(8))
            .ok_or_else(|| Error::InvalidFormat("chunk B-tree node size overflow".into()))?;
        let key_bytes = max_entries
            .checked_add(1)
            .and_then(|value| value.checked_mul(key_size))
            .ok_or_else(|| Error::InvalidFormat("chunk B-tree node size overflow".into()))?;
        let child_bytes = max_entries
            .checked_mul(sizeof_addr)
            .ok_or_else(|| Error::InvalidFormat("chunk B-tree node size overflow".into()))?;
        header_size
            .checked_add(key_bytes)
            .and_then(|value| value.checked_add(child_bytes))
            .ok_or_else(|| Error::InvalidFormat("chunk B-tree node size overflow".into()))
    }

    fn encode_chunk_btree_node_v1(
        &self,
        level: u8,
        entries: &[ChunkBTreeEntry],
        final_coords: &[u64],
        ndims: usize,
        element_size: u32,
    ) -> Result<Vec<u8>> {
        if entries.is_empty() {
            return Err(Error::InvalidFormat(
                "cannot write empty chunk B-tree node".into(),
            ));
        }
        if entries.len() > 64 {
            return Err(Error::InvalidFormat(
                "chunk B-tree node entry count exceeds v1 node capacity".into(),
            ));
        }

        let sa = usize::from(self.sizeof_addr);
        let node_size = Self::chunk_btree_node_size(ndims, sa)?;
        let mut buf = Vec::with_capacity(node_size);
        let undef = UNDEF_ADDR.to_le_bytes();

        buf.extend_from_slice(b"TREE");
        buf.push(1);
        buf.push(level);
        buf.extend_from_slice(
            &u16::try_from(entries.len())
                .map_err(|_| Error::InvalidFormat("chunk B-tree entry count exceeds u16".into()))?
                .to_le_bytes(),
        );
        buf.extend_from_slice(&undef[..sa]);
        buf.extend_from_slice(&undef[..sa]);

        for entry in entries {
            buf.extend_from_slice(&entry.chunk_size.to_le_bytes());
            buf.extend_from_slice(&entry.filter_mask.to_le_bytes());
            for &coord in &entry.coords {
                buf.extend_from_slice(&coord.to_le_bytes());
            }
            buf.extend_from_slice(&0u64.to_le_bytes());
            buf.extend_from_slice(&entry.child_addr.to_le_bytes()[..sa]);
        }

        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        for &coord in final_coords {
            buf.extend_from_slice(&coord.to_le_bytes());
        }
        buf.extend_from_slice(&u64::from(element_size).to_le_bytes());
        buf.resize(node_size, 0);
        Ok(buf)
    }

    fn write_dense_link_storage(&mut self, links: &[(String, u64)]) -> Result<(u64, u64)> {
        let mut payloads = Vec::with_capacity(links.len());
        let mut managed_size = 0u64;
        for (name, addr) in links {
            let link_bytes = encode_dense_link_message(name, *addr, self.sizeof_addr)?;
            managed_size = managed_size
                .checked_add(u64_from_usize_writer(
                    link_bytes.len(),
                    "dense link payload size",
                )?)
                .ok_or_else(|| Error::InvalidFormat("dense link heap size overflow".into()))?;
            payloads.push((name.as_str(), link_bytes));
        }

        let (heap_addr, heap_ids) = self.write_managed_fractal_heap(&payloads, 7)?;
        let mut records = Vec::with_capacity(payloads.len());
        for ((name, _), heap_id) in payloads.iter().zip(heap_ids) {
            let mut record = Vec::with_capacity(4 + heap_id.len());
            record.extend_from_slice(&dense_name_hash(name).to_le_bytes());
            record.extend_from_slice(&heap_id);
            records.push(record);
        }
        records.sort_by_key(|record| dense_record_hash(record).unwrap_or(u32::MAX));

        let btree_addr = self.write_dense_name_btree(5, &records)?;
        Ok((heap_addr, btree_addr))
    }

    fn write_dense_attribute_storage(&mut self, attrs: &[AttrSpec<'_>]) -> Result<(u64, u64)> {
        validate_unique_attr_names(attrs)?;
        let mut payloads = Vec::with_capacity(attrs.len());
        for attr in attrs {
            let attr_bytes =
                encode_attribute_message(attr.name, &attr.dtype, attr.shape, attr.data)?;
            payloads.push((attr.name, attr_bytes));
        }

        let (heap_addr, heap_ids) = self.write_managed_fractal_heap(&payloads, 8)?;
        let mut records = Vec::with_capacity(payloads.len());
        for (creation_order, ((name, _), heap_id)) in payloads.iter().zip(heap_ids).enumerate() {
            let mut record = Vec::with_capacity(heap_id.len() + 9);
            record.extend_from_slice(&heap_id);
            record.push(0);
            record.extend_from_slice(
                &u32::try_from(creation_order)
                    .map_err(|_| {
                        Error::InvalidFormat("dense attribute creation order exceeds u32".into())
                    })?
                    .to_le_bytes(),
            );
            record.extend_from_slice(&dense_name_hash(name).to_le_bytes());
            records.push(record);
        }
        records.sort_by_key(|record| {
            record
                .len()
                .checked_sub(4)
                .and_then(|hash_pos| dense_record_hash(&record[hash_pos..]).ok())
                .unwrap_or(u32::MAX)
        });

        let btree_addr = self.write_dense_name_btree(8, &records)?;
        Ok((heap_addr, btree_addr))
    }

    fn write_managed_fractal_heap(
        &mut self,
        payloads: &[(&str, Vec<u8>)],
        heap_id_len: u16,
    ) -> Result<(u64, Vec<Vec<u8>>)> {
        let offset_bytes = 4usize;
        let offset_bytes_u16 = u16::try_from(offset_bytes)
            .map_err(|_| Error::InvalidFormat("managed heap ID offset width exceeds u16".into()))?;
        let length_bytes = usize::from(
            heap_id_len
                .checked_sub(1 + offset_bytes_u16)
                .ok_or_else(|| {
                    Error::InvalidFormat("managed heap ID length is too short".into())
                })?,
        );
        if length_bytes == 0 || length_bytes > 8 {
            return Err(Error::Unsupported(format!(
                "managed heap ID length {heap_id_len} leaves unsupported length byte count {length_bytes}"
            )));
        }
        let payload_bytes = payloads.iter().try_fold(0usize, |acc, (_, payload)| {
            acc.checked_add(payload.len())
                .ok_or_else(|| Error::InvalidFormat("managed heap payload size overflow".into()))
        })?;
        let needed_block_size = 25usize
            .checked_add(payload_bytes)
            .ok_or_else(|| Error::InvalidFormat("managed heap block size overflow".into()))?;
        let block_size =
            checked_next_power_of_two(needed_block_size, "managed heap block size")?.max(512);
        let heap_header_len = self.minimal_fractal_heap_header_len()?;
        let heap_addr = self.allocator.allocate(
            u64_from_usize_writer(heap_header_len, "fractal heap header length")?,
            8,
        );
        let direct_addr = self.allocator.allocate(
            u64_from_usize_writer(block_size, "fractal heap direct block size")?,
            8,
        );

        let mut direct = Vec::with_capacity(block_size);
        direct.extend_from_slice(b"FHDB");
        direct.push(0);
        direct.extend_from_slice(&heap_addr.to_le_bytes()[..usize::from(self.sizeof_addr)]);
        direct.extend_from_slice(&0u32.to_le_bytes());
        direct.extend_from_slice(&0u32.to_le_bytes());

        let mut heap_ids = Vec::with_capacity(payloads.len());
        for (_, payload) in payloads {
            let offset = u32::try_from(direct.len()).map_err(|_| {
                Error::InvalidFormat("managed heap object offset exceeds u32".into())
            })?;
            direct.extend_from_slice(payload);
            let len = u64_from_usize_writer(payload.len(), "dense heap payload length")?;
            let max_len = if length_bytes == 8 {
                u64::MAX
            } else {
                (1u64 << (length_bytes * 8)) - 1
            };
            if len > max_len {
                return Err(Error::Unsupported(format!(
                    "dense heap payload length {len} exceeds {length_bytes}-byte managed heap ID length"
                )));
            }
            let mut heap_id = Vec::with_capacity(usize::from(heap_id_len));
            heap_id.push(0);
            heap_id.extend_from_slice(&offset.to_le_bytes()[..offset_bytes]);
            heap_id.extend_from_slice(&len.to_le_bytes()[..length_bytes]);
            heap_ids.push(heap_id);
        }
        direct.resize(block_size, 0);
        let checksum = checksum_metadata(&direct);
        checked_window_mut(&mut direct, 17, 4, "fractal heap direct block checksum")?
            .copy_from_slice(&checksum.to_le_bytes());
        self.write_at(direct_addr, &direct)?;

        let used_managed_space = u64_from_usize_writer(
            direct.iter().rposition(|byte| *byte != 0).unwrap_or(20),
            "managed heap used space",
        )?
        .checked_add(1)
        .ok_or_else(|| Error::InvalidFormat("managed heap used space overflow".into()))?;
        let heap = self.encode_minimal_fractal_heap(
            heap_id_len,
            payloads.len(),
            used_managed_space,
            u64_from_usize_writer(block_size, "fractal heap direct block size")?,
            direct_addr,
        )?;
        debug_assert_eq!(heap.len(), heap_header_len);
        self.write_at(heap_addr, &heap)?;
        Ok((heap_addr, heap_ids))
    }

    fn minimal_fractal_heap_header_len(&self) -> Result<usize> {
        let sa = usize::from(self.sizeof_addr);
        let ss = usize::from(self.sizeof_size);
        let fixed = 4usize + 1 + 2 + 2 + 1 + 4 + 2 + 2 + 2 + 2 + 4;
        let size_fields = ss
            .checked_mul(12)
            .ok_or_else(|| Error::InvalidFormat("fractal heap header size overflow".into()))?;
        let addr_fields = sa
            .checked_mul(3)
            .ok_or_else(|| Error::InvalidFormat("fractal heap header size overflow".into()))?;
        fixed
            .checked_add(size_fields)
            .and_then(|value| value.checked_add(addr_fields))
            .ok_or_else(|| Error::InvalidFormat("fractal heap header size overflow".into()))
    }

    fn encode_minimal_fractal_heap(
        &self,
        heap_id_len: u16,
        managed_nobjs: usize,
        _managed_size: u64,
        managed_alloc_size: u64,
        root_block_addr: u64,
    ) -> Result<Vec<u8>> {
        let mut buf = Vec::new();
        let undef = UNDEF_ADDR.to_le_bytes();
        let sa = usize::from(self.sizeof_addr);
        let ss = usize::from(self.sizeof_size);
        let free_space = 0u64;
        let managed_nobjs = u64_from_usize_writer(managed_nobjs, "managed heap object count")?;

        buf.extend_from_slice(b"FRHP");
        buf.push(0);
        buf.extend_from_slice(&heap_id_len.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.push(0x02);
        buf.extend_from_slice(&4096u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()[..ss]);
        buf.extend_from_slice(&undef[..sa]);
        buf.extend_from_slice(&free_space.to_le_bytes()[..ss]);
        buf.extend_from_slice(&undef[..sa]);
        buf.extend_from_slice(&managed_alloc_size.to_le_bytes()[..ss]);
        buf.extend_from_slice(&managed_alloc_size.to_le_bytes()[..ss]);
        buf.extend_from_slice(&0u64.to_le_bytes()[..ss]);
        buf.extend_from_slice(&managed_nobjs.to_le_bytes()[..ss]);
        buf.extend_from_slice(&0u64.to_le_bytes()[..ss]);
        buf.extend_from_slice(&0u64.to_le_bytes()[..ss]);
        buf.extend_from_slice(&0u64.to_le_bytes()[..ss]);
        buf.extend_from_slice(&0u64.to_le_bytes()[..ss]);
        buf.extend_from_slice(&4u16.to_le_bytes());
        buf.extend_from_slice(&managed_alloc_size.to_le_bytes()[..ss]);
        buf.extend_from_slice(&65536u64.to_le_bytes()[..ss]);
        buf.extend_from_slice(&32u16.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&root_block_addr.to_le_bytes()[..sa]);
        buf.extend_from_slice(&0u16.to_le_bytes());
        let checksum = checksum_metadata(&buf);
        buf.extend_from_slice(&checksum.to_le_bytes());
        Ok(buf)
    }

    fn write_dense_name_btree(&mut self, tree_type: u8, records: &[Vec<u8>]) -> Result<u64> {
        let record_size = records
            .first()
            .map(|record| record.len())
            .ok_or_else(|| Error::InvalidFormat("cannot write empty dense name B-tree".into()))?;
        if records.iter().any(|record| record.len() != record_size) {
            return Err(Error::InvalidFormat(
                "dense name B-tree records have inconsistent sizes".into(),
            ));
        }
        if u16::try_from(records.len()).is_err() {
            return Err(Error::Unsupported(
                "dense name B-tree writer supports at most 65535 records".into(),
            ));
        }

        let node_size = 512usize.max(10 + records.len() * record_size);
        let mut leaf = Vec::with_capacity(6 + records.len() * record_size + 4);
        leaf.extend_from_slice(b"BTLF");
        leaf.push(0);
        leaf.push(tree_type);
        for record in records {
            leaf.extend_from_slice(record);
        }
        let leaf_checksum = checksum_metadata(&leaf);
        leaf.extend_from_slice(&leaf_checksum.to_le_bytes());
        let root_addr = self.allocator.allocate(
            u64_from_usize_writer(leaf.len(), "dense B-tree leaf size")?,
            8,
        );
        self.write_at(root_addr, &leaf)?;

        let mut header = Vec::new();
        header.extend_from_slice(b"BTHD");
        header.push(0);
        header.push(tree_type);
        header.extend_from_slice(
            &u32::try_from(node_size)
                .map_err(|_| Error::InvalidFormat("dense B-tree node size exceeds u32".into()))?
                .to_le_bytes(),
        );
        header.extend_from_slice(
            &u16::try_from(record_size)
                .map_err(|_| Error::InvalidFormat("dense B-tree record size exceeds u16".into()))?
                .to_le_bytes(),
        );
        header.extend_from_slice(&0u16.to_le_bytes());
        header.push(100);
        header.push(40);
        header.extend_from_slice(&root_addr.to_le_bytes()[..usize::from(self.sizeof_addr)]);
        header.extend_from_slice(
            &u16::try_from(records.len())
                .map_err(|_| Error::InvalidFormat("dense B-tree record count exceeds u16".into()))?
                .to_le_bytes(),
        );
        let record_count = u64_from_usize_writer(records.len(), "dense B-tree record count")?;
        header.extend_from_slice(&record_count.to_le_bytes()[..usize::from(self.sizeof_size)]);
        let checksum = checksum_metadata(&header);
        header.extend_from_slice(&checksum.to_le_bytes());
        let header_addr = self.allocator.allocate(
            u64_from_usize_writer(header.len(), "dense B-tree header size")?,
            8,
        );
        self.write_at(header_addr, &header)?;
        Ok(header_addr)
    }

    /// Finalize the file: update root group with links and write superblock.
    pub fn finalize(&mut self) -> Result<()> {
        // Sort groups by depth (deepest first) so child groups are written
        // before their parents, and parent links point to correct addresses.
        let mut group_paths: Vec<String> = self.groups.keys().cloned().collect();
        group_paths.sort_by(|a, b| {
            let depth_a = a.split('/').filter(|s| !s.is_empty()).count();
            let depth_b = b.split('/').filter(|s| !s.is_empty()).count();
            depth_b.cmp(&depth_a) // deepest first
        });

        for path in group_paths {
            // Collect links for this group, using CURRENT addresses
            let group_links: Vec<(String, u64)> = self
                .links
                .iter()
                .filter(|(parent, _, _)| *parent == path)
                .map(|(_, name, addr)| {
                    // If the target is a group, use its updated address
                    let target_path = if path == "/" {
                        format!("/{name}")
                    } else {
                        format!("{path}/{name}")
                    };
                    let current_addr = self.groups.get(&target_path).copied().unwrap_or(*addr);
                    (name.clone(), current_addr)
                })
                .collect();

            let has_special_links = self
                .special_links
                .iter()
                .any(|(parent, _, _)| *parent == path);
            let has_hard_links = self.hard_links.iter().any(|(parent, _, _)| *parent == path);
            let has_group_attrs = self
                .pending_group_attr_specs
                .get(&path)
                .is_some_and(|attrs| !attrs.is_empty());

            if group_links.is_empty()
                && !has_special_links
                && !has_hard_links
                && !has_group_attrs
                && path != "/"
            {
                continue;
            }

            // Build messages: link info + link messages. Groups above the
            // compact threshold use dense link storage, backed by a v2 B-tree
            // name index and heap IDs that point directly at link payloads.
            let mut messages: Vec<(u16, Vec<u8>)> = Vec::new();

            if group_links.len() > 8 {
                let (heap_addr, btree_addr) = self.write_dense_link_storage(&group_links)?;
                messages.push((MSG_GROUP_INFO, vec![0, 0]));
                messages.push((
                    MSG_LINK_INFO,
                    encode_link_info_message(heap_addr, btree_addr, self.sizeof_addr)?,
                ));
            } else {
                messages.push((MSG_GROUP_INFO, vec![0, 0]));
                let link_info = encode_link_info_message(UNDEF_ADDR, UNDEF_ADDR, self.sizeof_addr)?;
                messages.push((MSG_LINK_INFO, link_info));

                for (name, addr) in &group_links {
                    let link_bytes = encode_link_message(name, *addr, self.sizeof_addr)?;
                    messages.push((MSG_LINK, link_bytes));
                }
            }

            // Add special links (soft/external) for this group
            for (parent, _, link_data) in &self.special_links {
                if *parent == path {
                    messages.push((MSG_LINK, link_data.clone()));
                }
            }

            // Add explicit hard-link aliases after resolving final group
            // addresses. Dataset addresses are stable; group object headers
            // may have been rewritten earlier in this finalize pass.
            for (parent, name, target_path) in &self.hard_links {
                if *parent == path {
                    let target_addr = self.object_addr_for_path(target_path).ok_or_else(|| {
                        Error::InvalidFormat(format!("hard link target '{target_path}' not found"))
                    })?;
                    messages.push((
                        MSG_LINK,
                        encode_link_message(name, target_addr, self.sizeof_addr)?,
                    ));
                }
            }

            // Add pending root attributes. Attributes added through the typed
            // API can spill to dense storage; pre-encoded messages remain compact.
            if path == "/" {
                for (msg_type, attr_data) in &self.pending_root_attrs {
                    messages.push((*msg_type, attr_data.clone()));
                }

                if self.pending_root_attr_specs.len() > 8 {
                    let owned_attrs = self.pending_root_attr_specs.clone();
                    let attr_specs: Vec<AttrSpec<'_>> = owned_attrs
                        .iter()
                        .map(OwnedAttrSpec::as_attr_spec)
                        .collect();
                    let (heap_addr, btree_addr) =
                        self.write_dense_attribute_storage(&attr_specs)?;
                    messages.push((
                        MSG_ATTR_INFO,
                        encode_attr_info_message(heap_addr, btree_addr, self.sizeof_addr)?,
                    ));
                } else {
                    for attr in &self.pending_root_attr_specs {
                        let attr_bytes = encode_attribute_message(
                            &attr.name,
                            &attr.dtype,
                            &attr.shape,
                            &attr.data,
                        )?;
                        messages.push((MSG_ATTRIBUTE, attr_bytes));
                    }
                }
            }

            if let Some(group_attrs) = self.pending_group_attr_specs.get(&path).cloned() {
                if group_attrs.len() > 8 {
                    let attr_specs: Vec<AttrSpec<'_>> = group_attrs
                        .iter()
                        .map(OwnedAttrSpec::as_attr_spec)
                        .collect();
                    let (heap_addr, btree_addr) =
                        self.write_dense_attribute_storage(&attr_specs)?;
                    messages.push((
                        MSG_ATTR_INFO,
                        encode_attr_info_message(heap_addr, btree_addr, self.sizeof_addr)?,
                    ));
                } else {
                    for attr in &group_attrs {
                        let attr_bytes = encode_attribute_message(
                            &attr.name,
                            &attr.dtype,
                            &attr.shape,
                            &attr.data,
                        )?;
                        messages.push((MSG_ATTRIBUTE, attr_bytes));
                    }
                }
            }

            let msg_refs: Vec<(u16, &[u8])> =
                messages.iter().map(|(t, d)| (*t, d.as_slice())).collect();

            let oh_bytes = build_v2_object_header(&msg_refs, 0)?;
            let oh_addr = self.allocator.allocate(
                u64_from_usize_writer(oh_bytes.len(), "object header size")?,
                8,
            );
            self.write_at(oh_addr, &oh_bytes)?;

            // Update the group address
            self.groups.insert(path.clone(), oh_addr);
        }

        // Write superblock
        let root_addr = *self
            .groups
            .get("/")
            .ok_or_else(|| Error::Other("no root group".into()))?;
        let eof = self.allocator.eof();

        let sb = Superblock {
            version: 2,
            sizeof_addr: self.sizeof_addr,
            sizeof_size: self.sizeof_size,
            status_flags: 0,
            base_addr: 0,
            ext_addr: UNDEF_ADDR,
            eof_addr: eof,
            root_addr,
            ..Default::default()
        };

        let mut sb_bytes = Vec::new();
        sb.write_v2(&mut sb_bytes)?;
        self.write_at(0, &sb_bytes)?;

        self.writer.flush()?;

        Ok(())
    }

    fn write_at(&mut self, offset: u64, data: &[u8]) -> Result<()> {
        self.writer.seek(SeekFrom::Start(offset))?;
        self.writer.write_all(data)?;
        Ok(())
    }

    fn object_addr_for_path(&self, path: &str) -> Option<u64> {
        let path = normalize_object_path(path);
        if let Some(addr) = self.groups.get(&path) {
            return Some(*addr);
        }
        self.links
            .iter()
            .find(|(parent, name, _)| child_path(parent, name) == path)
            .map(|(_, _, addr)| *addr)
    }
}

fn normalize_object_path(path: &str) -> String {
    if path == "/" {
        return "/".to_string();
    }
    let trimmed = path.trim_matches('/');
    if trimmed.is_empty() {
        "/".to_string()
    } else {
        format!("/{trimmed}")
    }
}

fn child_path(parent: &str, name: &str) -> String {
    if parent == "/" {
        format!("/{name}")
    } else {
        format!("{parent}/{name}")
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ceil_div_nonzero_u64, checked_next_power_of_two, dense_record_hash,
        encode_global_heap_collection, read_u64_le_at, CompoundFieldSpec, DtypeSpec, FillValueSpec,
        HdfFileWriter,
    };
    use std::io::Cursor;

    #[test]
    fn ceil_div_nonzero_u64_rejects_zero_divisor() {
        let err = ceil_div_nonzero_u64(10, 0, "test ceil-div").unwrap_err();
        assert!(
            err.to_string().contains("test ceil-div divisor is zero"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn ceil_div_nonzero_u64_rounds_up_without_overflow() {
        assert_eq!(ceil_div_nonzero_u64(0, 4, "test ceil-div").unwrap(), 0);
        assert_eq!(ceil_div_nonzero_u64(1, 4, "test ceil-div").unwrap(), 1);
        assert_eq!(ceil_div_nonzero_u64(5, 4, "test ceil-div").unwrap(), 2);
        assert_eq!(
            ceil_div_nonzero_u64(u64::MAX, u64::MAX, "test ceil-div").unwrap(),
            1
        );
    }

    #[test]
    fn usize_from_u64_writer_accepts_normal_values() {
        assert_eq!(super::usize_from_u64_writer(42, "test value").unwrap(), 42);
    }

    #[test]
    fn checked_writer_metadata_sizing_rejects_overflow() {
        assert!(HdfFileWriter::<Cursor<Vec<u8>>>::chunk_btree_node_size(usize::MAX, 8).is_err());
        assert!(checked_next_power_of_two(usize::MAX, "test power").is_err());
    }

    #[test]
    fn dense_record_hash_rejects_truncated_record() {
        let err = dense_record_hash(&[1, 2, 3]).unwrap_err();
        assert!(
            err.to_string()
                .contains("dense B-tree record hash is truncated"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn global_heap_free_object_size_includes_header() {
        let heap =
            encode_global_heap_collection(&[b"alpha".to_vec(), vec![1, 2, 3], Vec::new()], 8)
                .unwrap();
        let collection_size = read_u64_le_at(&heap, 8, "test global heap collection size")
            .expect("collection size field should decode");
        let free_size = read_u64_le_at(&heap, 88, "test global heap free size")
            .expect("free size field should decode");

        assert_eq!(collection_size, 4096);
        assert_eq!(free_size, 4096 - 80);
    }

    #[test]
    fn dtype_encoder_rejects_narrowing_and_invalid_metadata() {
        assert!(DtypeSpec::Array {
            dims: vec![u32::MAX, u32::MAX],
            base: Box::new(DtypeSpec::U64),
        }
        .encode()
        .is_err());
        assert!(DtypeSpec::Array {
            dims: vec![0],
            base: Box::new(DtypeSpec::U8),
        }
        .encode()
        .is_err());
        assert!(DtypeSpec::Array {
            dims: vec![1; 256],
            base: Box::new(DtypeSpec::U8),
        }
        .encode()
        .is_err());
        assert!(DtypeSpec::Opaque {
            size: 1,
            tag: "x".repeat(255),
        }
        .encode()
        .is_err());
        assert!(DtypeSpec::Opaque {
            size: 1,
            tag: "bad\0tag".into(),
        }
        .encode()
        .is_err());
        assert!(DtypeSpec::Compound {
            size: 1,
            fields: vec![CompoundFieldSpec {
                name: "bad\0field".into(),
                offset: 0,
                dtype: DtypeSpec::U8,
            }],
        }
        .encode()
        .is_err());
        assert!(DtypeSpec::Enum {
            base: Box::new(DtypeSpec::U8),
            members: vec![("bad\0member".into(), 0)],
        }
        .encode()
        .is_err());
    }

    #[test]
    fn writer_metadata_encoders_reject_narrowing() {
        assert!(super::append_encoded_addr(&mut Vec::new(), 0, 0).is_err());
        assert!(super::append_encoded_addr(&mut Vec::new(), 0, 9).is_err());
        assert!(super::encode_link_name_len(&mut Vec::new(), 256, 0).is_err());
        assert!(super::encode_link_name_len(&mut Vec::new(), 65_536, 1).is_err());
        if usize::BITS > u32::BITS {
            let too_large = usize::try_from(u64::from(u32::MAX) + 1)
                .expect("test value should fit on this target");
            assert!(super::encode_link_name_len(&mut Vec::new(), too_large, 2).is_err());
        }
        assert!(super::encode_contiguous_layout(0, 0, 8, 0).is_err());
        assert!(super::encode_contiguous_layout(0, 0, 8, 9).is_err());
        assert!(super::encode_chunked_layout_v3(0, &[u64::from(u32::MAX) + 1], 1, 8).is_err());
        assert!(super::encode_chunked_layout_v3(0, &[0], 1, 8).is_err());
        assert!(super::encode_fill_value_message(Some(FillValueSpec {
            alloc_time: 4,
            fill_time: 0,
            value: None,
        }))
        .is_err());
    }
}
