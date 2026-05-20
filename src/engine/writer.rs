use std::collections::{HashMap, HashSet};
use std::io::{Seek, SeekFrom, Write};

use crate::engine::allocator::FileAllocator;
use crate::error::{Error, Result};
use crate::format::checksum::checksum_metadata;
use crate::format::messages::datatype::DatatypeMessage;
use crate::format::object_header::*;
use crate::format::superblock::Superblock;
use crate::io::reader::UNDEF_ADDR;

const MAX_DATASPACE_RANK: usize = 32;
const OBJECT_HEADER_CHUNK_DATA_LIMIT: usize = 128 * 1024;
const FIXED_ARRAY_CHUNK_PAGE_BITS: u8 = 12;
const EXTENSIBLE_ARRAY_INDEX_BLOCK_ELEMENTS_LIMIT: usize = u8::MAX as usize;

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

struct EncodedLinkRecord {
    name: String,
    compact_message: Vec<u8>,
    dense_message: Vec<u8>,
}

impl OwnedAttrSpec {
    /// Borrow this owned attribute spec as a non-owning [`AttrSpec`].
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

/// One already materialized chunk to write into a chunked dataset.
pub struct ChunkWriteSpec<'a> {
    pub coords: &'a [u64],
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
    /// Fill value with no payload (corresponds to `H5P_DEFAULT`).
    pub fn undefined(alloc_time: u8, fill_time: u8) -> Self {
        Self {
            alloc_time,
            fill_time,
            value: None,
        }
    }

    /// Fill value initialized to the provided bytes (mirrors `H5P_set_fill_value`).
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
    /// Encoded byte size of one element of this datatype (mirrors `H5O__dtype_size`).
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
        let mut buf = Vec::new();
        self.encode_into(&mut buf)?;
        Ok(buf)
    }

    /// Append this datatype as HDF5 datatype message bytes.
    pub fn encode_into(&self, out: &mut Vec<u8>) -> Result<()> {
        self.encode_with_padding_into(out, true)
    }

    /// Append without top-level alignment padding for use inside other messages.
    fn encode_embedded_into(&self, out: &mut Vec<u8>) -> Result<()> {
        self.encode_with_padding_into(out, false)
    }

    /// Append optionally adding top-level 8-byte alignment padding.
    fn encode_with_padding_into(&self, out: &mut Vec<u8>, pad_top_level: bool) -> Result<()> {
        let start = out.len();
        let result = (|| {
            match self {
                DtypeSpec::F32 | DtypeSpec::F64 => self.encode_floating_point_into(out),
                DtypeSpec::FixedAsciiString { len, padding } => {
                    Self::encode_fixed_string_into(out, *len, *padding, false)?
                }
                DtypeSpec::FixedUtf8String { len, padding } => {
                    Self::encode_fixed_string_into(out, *len, *padding, true)?
                }
                DtypeSpec::VarLenUtf8String => Self::encode_vlen_utf8_string_into(out),
                DtypeSpec::Compound { size, fields } => {
                    Self::encode_compound_into(out, *size, fields)?
                }
                DtypeSpec::Enum { base, members } => Self::encode_enum_into(out, base, members)?,
                DtypeSpec::Opaque { size, tag } => Self::encode_opaque_into(out, *size, tag)?,
                DtypeSpec::Array { dims, base } => {
                    Self::encode_array_into(out, self.checked_size()?, dims, base)?
                }
                _ => self.encode_fixed_point_into(out),
            }

            DatatypeMessage::decode(&out[start..])?;

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
                while out.len() % 8 != 0 {
                    out.push(0);
                }
            }

            Ok(())
        })();
        if result.is_err() {
            out.truncate(start);
        }
        result
    }

    /// Compute the byte size with overflow checking.
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

    /// Append an IEEE float datatype message.
    fn encode_floating_point_into(&self, buf: &mut Vec<u8>) {
        let size = self.size();
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
    }

    /// Append a fixed-length string datatype message.
    fn encode_fixed_string_into(
        buf: &mut Vec<u8>,
        len: u32,
        padding: u8,
        utf8: bool,
    ) -> Result<()> {
        if padding > 2 {
            return Err(Error::InvalidFormat(format!(
                "fixed-length string padding {padding} is invalid"
            )));
        }
        buf.push(0x13);
        buf.push(padding | if utf8 { 0x10 } else { 0x00 });
        buf.extend_from_slice(&[0x00, 0x00]);
        buf.extend_from_slice(&len.to_le_bytes());
        Ok(())
    }

    /// Append a variable-length UTF-8 string datatype message.
    fn encode_vlen_utf8_string_into(buf: &mut Vec<u8>) {
        buf.push(0x19);
        buf.extend_from_slice(&[0x01, 0x01, 0x00]);
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&[
            0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00,
            0x00, 0x00,
        ]);
    }

    /// Append a compound datatype message.
    fn encode_compound_into(
        buf: &mut Vec<u8>,
        size: u32,
        fields: &[CompoundFieldSpec],
    ) -> Result<()> {
        Self::encode_compound_header_into(buf, size, fields.len())?;
        for field in fields {
            Self::encode_compound_field(buf, field)?;
        }
        Ok(())
    }

    /// Append the leading bytes of a compound datatype message.
    fn encode_compound_header_into(buf: &mut Vec<u8>, size: u32, field_count: usize) -> Result<()> {
        let field_count = u16::try_from(field_count).map_err(|_| {
            Error::InvalidFormat("compound datatype member count exceeds u16".into())
        })?;
        buf.push(0x16);
        buf.extend_from_slice(&field_count.to_le_bytes());
        buf.push(0);
        buf.extend_from_slice(&size.to_le_bytes());
        Ok(())
    }

    /// Append one compound field record to `buf`.
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
        field.dtype.encode_embedded_into(buf)?;
        Ok(())
    }

    /// Append an enumeration datatype message.
    fn encode_enum_into(
        buf: &mut Vec<u8>,
        base: &DtypeSpec,
        members: &[(String, u64)],
    ) -> Result<()> {
        let mut base_bytes = Vec::new();
        base.encode_embedded_into(&mut base_bytes)?;
        let base_size = base.checked_size()?;
        Self::encode_enum_header_into(buf, base_size, members.len(), &base_bytes)?;
        Self::encode_enum_names(buf, members)?;
        let base_size = usize::try_from(base_size)
            .map_err(|_| Error::InvalidFormat("enum datatype base size exceeds usize".into()))?;
        Self::encode_enum_values(buf, base_size, members)?;
        Ok(())
    }

    /// Append the leading bytes of an enum datatype message.
    fn encode_enum_header_into(
        buf: &mut Vec<u8>,
        base_size: u32,
        member_count: usize,
        base_bytes: &[u8],
    ) -> Result<()> {
        let member_count = u16::try_from(member_count)
            .map_err(|_| Error::InvalidFormat("enum datatype member count exceeds u16".into()))?;
        buf.push(0x18);
        buf.extend_from_slice(&member_count.to_le_bytes());
        buf.push(0);
        buf.extend_from_slice(&base_size.to_le_bytes());
        buf.extend_from_slice(base_bytes);
        Ok(())
    }

    /// Append the padded member names of an enum datatype.
    fn encode_enum_names(buf: &mut Vec<u8>, members: &[(String, u64)]) -> Result<()> {
        for (name, _) in members {
            validate_dtype_name(name, "enum datatype member name")?;
            Self::encode_padded_name(buf, name);
        }
        Ok(())
    }

    /// Append the binary values of an enum datatype.
    fn encode_enum_values(
        buf: &mut Vec<u8>,
        value_size: usize,
        members: &[(String, u64)],
    ) -> Result<()> {
        for (_, value) in members {
            let encoded = value.to_le_bytes();
            buf.extend_from_slice(&encoded[..value_size.min(encoded.len())]);
            if value_size > encoded.len() {
                let padded_len = buf
                    .len()
                    .checked_add(value_size - encoded.len())
                    .ok_or_else(|| {
                        Error::InvalidFormat("enum datatype value padding overflow".into())
                    })?;
                buf.resize(padded_len, 0);
            }
        }
        Ok(())
    }

    /// Append an opaque datatype message.
    fn encode_opaque_into(buf: &mut Vec<u8>, size: u32, tag: &str) -> Result<()> {
        validate_dtype_name(tag, "opaque datatype tag")?;
        Self::encode_opaque_header_into(buf, size, tag)?;
        buf.extend_from_slice(tag.as_bytes());
        buf.push(0);
        while buf.len() % 8 != 0 {
            buf.push(0);
        }
        Ok(())
    }

    /// Append the leading bytes of an opaque datatype message.
    fn encode_opaque_header_into(buf: &mut Vec<u8>, size: u32, tag: &str) -> Result<()> {
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
        Ok(())
    }

    /// Append an array datatype message.
    fn encode_array_into(
        buf: &mut Vec<u8>,
        size: u32,
        dims: &[u32],
        base: &DtypeSpec,
    ) -> Result<()> {
        Self::encode_array_header_into(buf, size, dims)?;
        base.encode_embedded_into(buf)?;
        Ok(())
    }

    /// Append the leading bytes of an array datatype message.
    fn encode_array_header_into(buf: &mut Vec<u8>, size: u32, dims: &[u32]) -> Result<()> {
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
        buf.push(0x4a);
        buf.extend_from_slice(&[0x00, 0x00, 0x00]);
        buf.extend_from_slice(&size.to_le_bytes());
        buf.push(rank);
        for dim in dims {
            buf.extend_from_slice(&dim.to_le_bytes());
        }
        Ok(())
    }

    /// Append a fixed-point (integer) datatype message.
    fn encode_fixed_point_into(&self, buf: &mut Vec<u8>) {
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
    }

    /// Append a null-terminated name padded to 8-byte alignment.
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

/// Reject NUL bytes embedded inside a datatype name string.
fn validate_dtype_name(name: &str, context: &str) -> Result<()> {
    if name.as_bytes().contains(&0) {
        return Err(Error::InvalidFormat(format!("{context} contains NUL byte")));
    }
    Ok(())
}

/// Pick the link-name size-flag value for the given name length.
fn link_name_size_flag(name_len: usize) -> Result<u8> {
    if u8::try_from(name_len).is_ok() {
        Ok(0)
    } else if u16::try_from(name_len).is_ok() {
        Ok(1)
    } else if u32::try_from(name_len).is_ok() {
        Ok(2)
    } else if u64::try_from(name_len).is_ok() {
        Ok(3)
    } else {
        Err(Error::InvalidFormat(format!(
            "link name is {name_len} bytes, maximum is {}",
            u64::MAX
        )))
    }
}

/// Encode the link-name length field using the right width.
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
        3 => out.extend_from_slice(
            &u64::try_from(name_len)
                .map_err(|_| Error::InvalidFormat("link name length exceeds u64 encoding".into()))?
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

/// Append a file address using the given offset width.
fn append_encoded_addr(out: &mut Vec<u8>, value: u64, sizeof_addr: u8) -> Result<()> {
    if sizeof_addr == 0 || sizeof_addr > 8 {
        return Err(Error::InvalidFormat(format!(
            "address field width {sizeof_addr} is invalid"
        )));
    }
    let max = encoded_width_max(sizeof_addr);
    let encoded = if value == UNDEF_ADDR {
        max
    } else {
        if value > max {
            return Err(Error::InvalidFormat(format!(
                "address value {value:#x} does not fit in {sizeof_addr} bytes"
            )));
        }
        value
    };
    out.extend_from_slice(&encoded.to_le_bytes()[..usize::from(sizeof_addr)]);
    Ok(())
}

/// Append a length value using the given length width.
fn append_encoded_size(out: &mut Vec<u8>, value: u64, sizeof_size: u8) -> Result<()> {
    if sizeof_size == 0 || sizeof_size > 8 {
        return Err(Error::InvalidFormat(format!(
            "size field width {sizeof_size} is invalid"
        )));
    }
    let max = encoded_width_max(sizeof_size);
    if value > max {
        return Err(Error::InvalidFormat(format!(
            "size value {value:#x} does not fit in {sizeof_size} bytes"
        )));
    }
    out.extend_from_slice(&value.to_le_bytes()[..usize::from(sizeof_size)]);
    Ok(())
}

/// Append an unsigned integer using an arbitrary 1..=8 byte little-endian width.
fn append_encoded_uint(out: &mut Vec<u8>, value: u64, width: usize, context: &str) -> Result<()> {
    if width == 0 || width > 8 {
        return Err(Error::InvalidFormat(format!(
            "{context} field width {width} is invalid"
        )));
    }
    let max = encoded_width_max(u8::try_from(width).unwrap_or(8));
    if value > max {
        return Err(Error::InvalidFormat(format!(
            "{context} value {value:#x} does not fit in {width} bytes"
        )));
    }
    out.extend_from_slice(&value.to_le_bytes()[..width]);
    Ok(())
}

/// Maximum representable value for a given encoded width in bytes.
fn encoded_width_max(width: u8) -> u64 {
    if width == 8 {
        u64::MAX
    } else {
        (1u64 << (u32::from(width) * 8)) - 1
    }
}

fn filtered_chunk_size_len_v4(unfiltered_chunk_bytes: usize) -> usize {
    let bits = if unfiltered_chunk_bytes == 0 {
        0
    } else {
        usize::try_from(usize::BITS - unfiltered_chunk_bytes.leading_zeros()).unwrap_or(usize::MAX)
    };
    (1 + ((bits + 8) / 8)).min(8)
}

fn extensible_array_max_elements_bits(elements: usize) -> Result<u8> {
    if elements == 0 {
        return Err(Error::InvalidFormat(
            "extensible-array element count must be positive".into(),
        ));
    }
    u8::try_from(usize::BITS - elements.leading_zeros())
        .map_err(|_| Error::InvalidFormat("extensible-array element count bits exceed u8".into()))
}

/// Append a dataspace message.
fn encode_dataspace_into(out: &mut Vec<u8>, shape: &[u64]) -> Result<()> {
    if shape.len() > MAX_DATASPACE_RANK {
        return Err(Error::InvalidFormat(format!(
            "dataspace rank {} exceeds supported maximum {MAX_DATASPACE_RANK}",
            shape.len()
        )));
    }
    encode_dataspace_impl_into(out, shape, None)
}

/// Append the dataspace for a dataset spec, including max-shape.
fn encode_dataspace_for_spec_into(out: &mut Vec<u8>, spec: &DatasetSpec<'_>) -> Result<()> {
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
    encode_dataspace_impl_into(out, spec.shape, spec.max_shape)
}

/// Append a dataspace from a shape and optional max-shape.
fn encode_dataspace_impl_into(
    buf: &mut Vec<u8>,
    shape: &[u64],
    max_shape: Option<&[u64]>,
) -> Result<()> {
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

    Ok(())
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

/// Append a data layout message (v3, contiguous).
fn encode_contiguous_layout_into(
    buf: &mut Vec<u8>,
    data_addr: u64,
    data_size: u64,
    sizeof_addr: u8,
    sizeof_size: u8,
) -> Result<()> {
    buf.push(3); // version 3
    buf.push(1); // layout class = contiguous

    append_encoded_addr(buf, data_addr, sizeof_addr)?;

    append_encoded_size(buf, data_size, sizeof_size)?;

    Ok(())
}

/// Append a data layout message (v3, chunked).
fn encode_chunked_layout_v3_into(
    buf: &mut Vec<u8>,
    btree_addr: u64,
    chunk_dims: &[u64],
    element_size: u32,
    sizeof_addr: u8,
) -> Result<()> {
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
    append_encoded_addr(buf, btree_addr, sizeof_addr)?;

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

    Ok(())
}

/// Append a data layout message (v4, single chunk).
fn encode_single_chunk_layout_v4_into(
    buf: &mut Vec<u8>,
    chunk_addr: u64,
    chunk_dims: &[u64],
    element_size: u32,
    filtered_size: Option<u64>,
    filter_mask: u32,
    sizeof_addr: u8,
    sizeof_size: u8,
) -> Result<()> {
    buf.push(4); // version 4
    buf.push(2); // layout class = chunked

    let flags = if filtered_size.is_some() { 0x02 } else { 0 };
    buf.push(flags);

    let encoded_ndims = chunk_dims
        .len()
        .checked_add(1)
        .ok_or_else(|| Error::InvalidFormat("chunked layout rank overflow".into()))?;
    let ndims = u8::try_from(encoded_ndims)
        .map_err(|_| Error::InvalidFormat("chunked layout rank exceeds u8".into()))?;
    if ndims == 0 {
        return Err(Error::InvalidFormat(
            "chunked layout rank must be positive".into(),
        ));
    }
    buf.push(ndims);

    let max_dim = chunk_dims
        .iter()
        .copied()
        .chain(std::iter::once(u64::from(element_size)))
        .max()
        .unwrap_or(0);
    let enc_bytes_per_dim = (1usize..=8)
        .find(|width| u128::from(max_dim) < (1u128 << (width * 8)))
        .unwrap_or(8);
    buf.push(u8::try_from(enc_bytes_per_dim).unwrap_or(8));
    for &dim in chunk_dims {
        if dim == 0 {
            return Err(Error::InvalidFormat(
                "chunk dimension must be positive".into(),
            ));
        }
        buf.extend_from_slice(&dim.to_le_bytes()[..enc_bytes_per_dim]);
    }
    if element_size == 0 {
        return Err(Error::InvalidFormat(
            "chunk element size must be positive".into(),
        ));
    }
    buf.extend_from_slice(&u64::from(element_size).to_le_bytes()[..enc_bytes_per_dim]);

    buf.push(1); // chunk index type = single chunk
    if let Some(size) = filtered_size {
        append_encoded_size(buf, size, sizeof_size)?;
        buf.extend_from_slice(&filter_mask.to_le_bytes());
    }
    append_encoded_addr(buf, chunk_addr, sizeof_addr)?;

    Ok(())
}

/// Append a data layout message (v4, fixed-array chunk index).
fn encode_fixed_array_chunk_layout_v4_into(
    buf: &mut Vec<u8>,
    fixed_array_addr: u64,
    chunk_dims: &[u64],
    element_size: u32,
    page_bits: u8,
    sizeof_addr: u8,
) -> Result<()> {
    if page_bits == 0 {
        return Err(Error::InvalidFormat(
            "fixed-array chunk page bits must be positive".into(),
        ));
    }

    buf.push(4); // version 4
    buf.push(2); // layout class = chunked
    buf.push(0); // flags

    let encoded_ndims = chunk_dims
        .len()
        .checked_add(1)
        .ok_or_else(|| Error::InvalidFormat("chunked layout rank overflow".into()))?;
    let ndims = u8::try_from(encoded_ndims)
        .map_err(|_| Error::InvalidFormat("chunked layout rank exceeds u8".into()))?;
    if ndims == 0 {
        return Err(Error::InvalidFormat(
            "chunked layout rank must be positive".into(),
        ));
    }
    buf.push(ndims);

    let max_dim = chunk_dims
        .iter()
        .copied()
        .chain(std::iter::once(u64::from(element_size)))
        .max()
        .unwrap_or(0);
    let enc_bytes_per_dim = (1usize..=8)
        .find(|width| u128::from(max_dim) < (1u128 << (width * 8)))
        .unwrap_or(8);
    buf.push(u8::try_from(enc_bytes_per_dim).unwrap_or(8));
    for &dim in chunk_dims {
        if dim == 0 {
            return Err(Error::InvalidFormat(
                "chunk dimension must be positive".into(),
            ));
        }
        buf.extend_from_slice(&dim.to_le_bytes()[..enc_bytes_per_dim]);
    }
    if element_size == 0 {
        return Err(Error::InvalidFormat(
            "chunk element size must be positive".into(),
        ));
    }
    buf.extend_from_slice(&u64::from(element_size).to_le_bytes()[..enc_bytes_per_dim]);

    buf.push(3); // chunk index type = fixed array
    buf.push(page_bits);
    append_encoded_addr(buf, fixed_array_addr, sizeof_addr)?;

    Ok(())
}

/// Append a data layout message (v4, extensible-array chunk index).
#[allow(clippy::too_many_arguments)]
fn encode_extensible_array_chunk_layout_v4_into(
    buf: &mut Vec<u8>,
    extensible_array_addr: u64,
    chunk_dims: &[u64],
    element_size: u32,
    max_elements_bits: u8,
    index_block_elements: u8,
    super_block_min_data_ptrs: u8,
    data_block_min_elements: u8,
    max_data_block_page_elements_bits: u8,
    sizeof_addr: u8,
) -> Result<()> {
    for (value, context) in [
        (max_elements_bits, "extensible-array max elements bits"),
        (
            index_block_elements,
            "extensible-array index block elements",
        ),
        (
            super_block_min_data_ptrs,
            "extensible-array super block min data pointers",
        ),
        (
            data_block_min_elements,
            "extensible-array data block min elements",
        ),
        (
            max_data_block_page_elements_bits,
            "extensible-array max data block page elements bits",
        ),
    ] {
        if value == 0 {
            return Err(Error::InvalidFormat(format!("{context} must be positive")));
        }
    }

    buf.push(4); // version 4
    buf.push(2); // layout class = chunked
    buf.push(0); // flags

    let encoded_ndims = chunk_dims
        .len()
        .checked_add(1)
        .ok_or_else(|| Error::InvalidFormat("chunked layout rank overflow".into()))?;
    let ndims = u8::try_from(encoded_ndims)
        .map_err(|_| Error::InvalidFormat("chunked layout rank exceeds u8".into()))?;
    if ndims == 0 {
        return Err(Error::InvalidFormat(
            "chunked layout rank must be positive".into(),
        ));
    }
    buf.push(ndims);

    let max_dim = chunk_dims
        .iter()
        .copied()
        .chain(std::iter::once(u64::from(element_size)))
        .max()
        .unwrap_or(0);
    let enc_bytes_per_dim = (1usize..=8)
        .find(|width| u128::from(max_dim) < (1u128 << (width * 8)))
        .unwrap_or(8);
    buf.push(u8::try_from(enc_bytes_per_dim).unwrap_or(8));
    for &dim in chunk_dims {
        if dim == 0 {
            return Err(Error::InvalidFormat(
                "chunk dimension must be positive".into(),
            ));
        }
        buf.extend_from_slice(&dim.to_le_bytes()[..enc_bytes_per_dim]);
    }
    if element_size == 0 {
        return Err(Error::InvalidFormat(
            "chunk element size must be positive".into(),
        ));
    }
    buf.extend_from_slice(&u64::from(element_size).to_le_bytes()[..enc_bytes_per_dim]);

    buf.push(4); // chunk index type = extensible array
    buf.push(max_elements_bits);
    buf.push(index_block_elements);
    buf.push(super_block_min_data_ptrs);
    buf.push(data_block_min_elements);
    buf.push(max_data_block_page_elements_bits);
    append_encoded_addr(buf, extensible_array_addr, sizeof_addr)?;

    Ok(())
}

/// Append a filter pipeline message.
fn encode_filter_pipeline_into(
    buf: &mut Vec<u8>,
    compression_level: Option<u32>,
    shuffle: bool,
    fletcher32: bool,
) -> Result<()> {
    let filter_count =
        u8::from(shuffle) + u8::from(compression_level.is_some()) + u8::from(fletcher32);
    if filter_count == 0 {
        return Ok(());
    }

    buf.push(2); // version 2
    buf.push(filter_count); // number of filters

    if shuffle {
        encode_filter_pipeline_entry_into(buf, 2, &[])?;
    }
    if let Some(level) = compression_level {
        encode_filter_pipeline_entry_into(buf, 1, &[level])?;
    }
    if fletcher32 {
        encode_filter_pipeline_entry_into(buf, 3, &[])?;
    }

    Ok(())
}

/// Append one known-filter pipeline entry.
fn encode_filter_pipeline_entry_into(buf: &mut Vec<u8>, id: u16, params: &[u32]) -> Result<()> {
    buf.extend_from_slice(&id.to_le_bytes()); // filter ID
                                              // v2: skip name_length for known filter IDs (< 256)
    buf.extend_from_slice(&0u16.to_le_bytes()); // flags
    buf.extend_from_slice(
        &u16::try_from(params.len())
            .map_err(|_| Error::InvalidFormat("filter client-data value count exceeds u16".into()))?
            .to_le_bytes(),
    ); // number of client data values
    for &p in params {
        buf.extend_from_slice(&p.to_le_bytes());
    }
    Ok(())
}

/// Append an attribute message (v3).
fn encode_attribute_message_into(
    buf: &mut Vec<u8>,
    name: &str,
    dtype: &DtypeSpec,
    shape: &[u64],
    data: &[u8],
) -> Result<()> {
    validate_attr_payload(name, dtype, shape, data)?;
    let mut dtype_bytes = Vec::new();
    dtype.encode_into(&mut dtype_bytes)?;
    let mut ds_bytes = Vec::new();
    encode_dataspace_into(&mut ds_bytes, shape)?;
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

    Ok(())
}

/// Append a link-info message (mirrors `H5O__linfo_encode`).
fn encode_link_info_message_into(
    buf: &mut Vec<u8>,
    heap_addr: u64,
    name_btree_addr: u64,
    sizeof_addr: u8,
) -> Result<()> {
    buf.push(0);
    buf.push(0);
    append_encoded_addr(buf, heap_addr, sizeof_addr)?;
    append_encoded_addr(buf, name_btree_addr, sizeof_addr)?;
    Ok(())
}

/// Append an attribute-info message (mirrors `H5O__ainfo_encode`).
fn encode_attr_info_message_into(
    buf: &mut Vec<u8>,
    heap_addr: u64,
    name_btree_addr: u64,
    sizeof_addr: u8,
) -> Result<()> {
    buf.push(0);
    buf.push(0);
    append_encoded_addr(buf, heap_addr, sizeof_addr)?;
    append_encoded_addr(buf, name_btree_addr, sizeof_addr)?;
    Ok(())
}

fn attrs_need_dense_storage(attrs: &[AttrSpec<'_>]) -> Result<bool> {
    if attrs.len() > 8 {
        return Ok(true);
    }

    for attr in attrs {
        let mut attr_bytes = Vec::new();
        encode_attribute_message_into(
            &mut attr_bytes,
            attr.name,
            &attr.dtype,
            attr.shape,
            attr.data,
        )?;
        if u16::try_from(attr_bytes.len()).is_err() {
            return Ok(true);
        }
    }

    Ok(false)
}

/// Compute the Jenkins lookup3 hash of a link name (mirrors `H5_checksum_lookup3`).
fn dense_name_hash(name: &str) -> u32 {
    crate::format::checksum::checksum_lookup3(name.as_bytes(), 0)
}

/// Append a new-style fill-value message (mirrors `H5O__fill_new_encode`).
fn encode_fill_value_message_into(
    buf: &mut Vec<u8>,
    fill: Option<FillValueSpec<'_>>,
) -> Result<()> {
    let Some(fill) = fill else {
        buf.extend_from_slice(&[3u8, 0x09]);
        return Ok(());
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
    buf.push(3u8);
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
    Ok(())
}

/// Encode a global heap collection (mirrors `H5HG__cache_heap_serialize`).
const MAX_GLOBAL_HEAP_OBJECTS_PER_COLLECTION: usize = u16::MAX as usize;

fn encode_global_heap_collection<T: AsRef<[u8]>>(
    objects: &[T],
    sizeof_size: u8,
) -> Result<Vec<u8>> {
    if sizeof_size == 0 || sizeof_size > 8 {
        return Err(Error::InvalidFormat(format!(
            "global heap size field width {sizeof_size} is invalid"
        )));
    }
    if objects.len() > MAX_GLOBAL_HEAP_OBJECTS_PER_COLLECTION {
        return Err(Error::InvalidFormat(
            "too many global heap objects for vlen string dataset".into(),
        ));
    }

    let size_len = usize::from(sizeof_size);
    let heap_header_len = 8usize
        .checked_add(size_len)
        .ok_or_else(|| Error::InvalidFormat("global heap header size overflow".into()))?;
    let object_header_len = heap_header_len;

    let mut buf = Vec::new();
    buf.extend_from_slice(b"GCOL");
    buf.push(1);
    buf.extend_from_slice(&[0; 3]);
    append_encoded_size(&mut buf, 0, sizeof_size)?;

    for (idx, object) in objects.iter().enumerate() {
        let object = object.as_ref();
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
        append_encoded_size(&mut buf, object_size, sizeof_size)?;
        buf.extend_from_slice(object);
        let padded_end = buf
            .len()
            .checked_add(padded_len - object.len())
            .ok_or_else(|| Error::InvalidFormat("global heap object padding overflow".into()))?;
        buf.resize(padded_end, 0);
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
    append_encoded_size(&mut buf, free_size, sizeof_size)?;
    debug_assert_eq!(
        buf.len(),
        object_header_len + (target_size - free_size as usize)
    );
    buf.resize(target_size, 0);

    let collection_size = u64::try_from(buf.len()).map_err(|_| {
        Error::InvalidFormat("global heap collection length does not fit in u64".into())
    })?;
    if collection_size > encoded_width_max(sizeof_size) {
        return Err(Error::InvalidFormat(format!(
            "global heap collection size {collection_size:#x} does not fit in {sizeof_size} bytes"
        )));
    }
    let encoded_collection_size = collection_size.to_le_bytes();
    checked_window_mut(&mut buf, 8, size_len, "global heap collection size")?
        .copy_from_slice(&encoded_collection_size[..size_len]);
    debug_assert!(buf.len() >= heap_header_len);
    Ok(buf)
}

/// Product of all shape dimensions (mirrors `H5S_get_simple_extent_npoints`).
fn shape_element_count(shape: &[u64]) -> Result<u64> {
    if shape.is_empty() {
        return Ok(1);
    }
    shape.iter().try_fold(1u64, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| Error::InvalidFormat("dataset shape element count overflow".into()))
    })
}

/// Ceiling division by a non-zero divisor, erroring on overflow.
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

/// Convert a `u64` to `usize`, surfacing `context` on overflow.
fn usize_from_u64_writer(value: u64, context: &str) -> Result<usize> {
    usize::try_from(value)
        .map_err(|_| Error::InvalidFormat(format!("{context} does not fit in usize")))
}

/// Convert a `usize` to `u64`, surfacing `context` on overflow.
fn u64_from_usize_writer(value: usize, context: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| Error::InvalidFormat(format!("{context} exceeds u64")))
}

/// Round `value` up to the next power of two, erroring on overflow.
fn checked_next_power_of_two(value: usize, context: &str) -> Result<usize> {
    value
        .checked_next_power_of_two()
        .ok_or_else(|| Error::InvalidFormat(format!("{context} overflow")))
}

/// Sum a slice of `usize` values, erroring on overflow.
fn checked_usize_sum_writer(parts: &[usize], context: &str) -> Result<usize> {
    parts.iter().try_fold(0usize, |acc, &part| {
        acc.checked_add(part)
            .ok_or_else(|| Error::InvalidFormat(format!("{context} overflow")))
    })
}

/// Bounds-checked mutable subslice with `context` for error messages.
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

/// Bounds-checked subslice with `context` for error messages.
#[cfg(test)]
fn checked_window<'a>(data: &'a [u8], pos: usize, len: usize, context: &str) -> Result<&'a [u8]> {
    let end = pos
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))?;
    data.get(pos..end)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))
}

/// Read a little-endian `u64` from `data` at `pos`.
#[cfg(test)]
fn read_u64_le_at(data: &[u8], pos: usize, context: &str) -> Result<u64> {
    let bytes = checked_window(data, pos, 8, context)?;
    Ok(u64::from_le_bytes(bytes.try_into().map_err(|_| {
        Error::InvalidFormat(format!("{context} is truncated"))
    })?))
}

/// Read a little-endian encoded size field from `data` at `pos`.
#[cfg(test)]
fn read_encoded_size_le_at(data: &[u8], pos: usize, sizeof_size: u8, context: &str) -> Result<u64> {
    if sizeof_size == 0 || sizeof_size > 8 {
        return Err(Error::InvalidFormat(format!(
            "{context} width {sizeof_size} is invalid"
        )));
    }
    let bytes = checked_window(data, pos, usize::from(sizeof_size), context)?;
    let mut encoded = [0; 8];
    encoded[..bytes.len()].copy_from_slice(bytes);
    Ok(u64::from_le_bytes(encoded))
}

/// Hash a dense-link record using its name field.
fn dense_record_hash(record: &[u8]) -> Result<u32> {
    let bytes = record
        .get(..4)
        .ok_or_else(|| Error::InvalidFormat("dense B-tree record hash is truncated".into()))?;
    let bytes: [u8; 4] = bytes
        .try_into()
        .map_err(|_| Error::InvalidFormat("dense B-tree record hash is truncated".into()))?;
    Ok(u32::from_le_bytes(bytes))
}

fn dense_btree_leaf_capacity(node_size: usize, record_size: usize) -> Result<usize> {
    if record_size == 0 || node_size <= 10 {
        return Err(Error::InvalidFormat(
            "invalid dense B-tree node sizing".into(),
        ));
    }
    let capacity = (node_size - 10) / record_size;
    if capacity == 0 {
        return Err(Error::InvalidFormat(
            "dense B-tree leaf cannot hold any records".into(),
        ));
    }
    Ok(capacity)
}

fn dense_btree_internal_capacity(
    node_size: usize,
    record_size: usize,
    leaf_max_records: usize,
    child_all_nrec_size: usize,
    sizeof_addr: u8,
) -> Result<usize> {
    if record_size == 0 || leaf_max_records == 0 {
        return Err(Error::InvalidFormat(
            "invalid dense B-tree node sizing".into(),
        ));
    }
    let max_nrec_size = dense_btree_bytes_needed(u64_from_usize_writer(
        leaf_max_records,
        "dense B-tree leaf capacity",
    )?);
    let pointer_size = checked_usize_sum_writer(
        &[usize::from(sizeof_addr), max_nrec_size, child_all_nrec_size],
        "dense B-tree pointer size",
    )?;
    let prefix_and_pointer =
        checked_usize_sum_writer(&[10, pointer_size], "dense B-tree internal node prefix")?;
    if node_size <= prefix_and_pointer {
        return Err(Error::InvalidFormat(
            "dense B-tree internal node cannot hold records".into(),
        ));
    }
    let record_slot = record_size
        .checked_add(pointer_size)
        .ok_or_else(|| Error::InvalidFormat("dense B-tree internal record slot overflow".into()))?;
    let capacity = (node_size - prefix_and_pointer) / record_slot;
    if capacity == 0 {
        return Err(Error::InvalidFormat(
            "dense B-tree internal node cannot hold records".into(),
        ));
    }
    Ok(capacity)
}

#[derive(Debug, Clone, Copy)]
struct DenseBtreeLevelInfo {
    max_nrecords: usize,
    cumulative_max_records: u64,
    cumulative_max_record_size: usize,
}

#[derive(Debug, Clone, Copy)]
struct DenseBtreeChild {
    addr: u64,
    node_nrecords: usize,
    total_records: u64,
}

fn dense_btree_level_info(
    node_size: usize,
    record_size: usize,
    record_count: usize,
    sizeof_addr: u8,
) -> Result<Vec<DenseBtreeLevelInfo>> {
    let leaf_max_records = dense_btree_leaf_capacity(node_size, record_size)?;
    let leaf_max_records_u64 =
        u64_from_usize_writer(leaf_max_records, "dense B-tree leaf capacity")?;
    let target_count = u64_from_usize_writer(record_count, "dense B-tree record count")?;
    let mut levels = vec![DenseBtreeLevelInfo {
        max_nrecords: leaf_max_records,
        cumulative_max_records: leaf_max_records_u64,
        cumulative_max_record_size: 0,
    }];

    while levels
        .last()
        .map(|level| level.cumulative_max_records < target_count)
        .unwrap_or(false)
    {
        let child_level = *levels
            .last()
            .ok_or_else(|| Error::InvalidFormat("dense B-tree level info is empty".into()))?;
        let child_all_nrec_size = if levels.len() > 1 {
            child_level.cumulative_max_record_size
        } else {
            0
        };
        let max_nrecords = dense_btree_internal_capacity(
            node_size,
            record_size,
            leaf_max_records,
            child_all_nrec_size,
            sizeof_addr,
        )?;
        let max_nrecords_u64 =
            u64_from_usize_writer(max_nrecords, "dense B-tree internal capacity")?;
        let cumulative_max_records = max_nrecords_u64
            .checked_add(1)
            .and_then(|children| children.checked_mul(child_level.cumulative_max_records))
            .and_then(|child_records| child_records.checked_add(max_nrecords_u64))
            .ok_or_else(|| Error::InvalidFormat("dense B-tree capacity overflow".into()))?;
        levels.push(DenseBtreeLevelInfo {
            max_nrecords,
            cumulative_max_records,
            cumulative_max_record_size: dense_btree_bytes_needed(cumulative_max_records),
        });
    }

    Ok(levels)
}

fn dense_btree_bytes_needed(mut value: u64) -> usize {
    let mut bytes = 1usize;
    while value > 0xff {
        value >>= 8;
        bytes += 1;
    }
    bytes
}

fn append_dense_btree_var_uint(out: &mut Vec<u8>, value: u64, size: usize) -> Result<()> {
    if size == 0 || size > 8 {
        return Err(Error::InvalidFormat(format!(
            "invalid dense B-tree variable integer size {size}"
        )));
    }
    let max = if size == 8 {
        u64::MAX
    } else {
        (1u64 << (size * 8)) - 1
    };
    if value > max {
        return Err(Error::InvalidFormat(
            "dense B-tree variable integer value exceeds encoded width".into(),
        ));
    }
    out.extend_from_slice(&value.to_le_bytes()[..size]);
    Ok(())
}

fn dense_heap_id_len_for_payloads(
    payloads: &[Vec<u8>],
    offset_bytes: usize,
    min_heap_id_len: u16,
) -> Result<u16> {
    let max_payload_len = payloads.iter().map(Vec::len).max().unwrap_or(0);
    let length_bytes = dense_btree_bytes_needed(u64_from_usize_writer(
        max_payload_len,
        "dense heap payload length",
    )?);
    let heap_id_len =
        checked_usize_sum_writer(&[1, offset_bytes, length_bytes], "managed heap ID length")?;
    let heap_id_len = heap_id_len.max(usize::from(min_heap_id_len));
    u16::try_from(heap_id_len)
        .map_err(|_| Error::InvalidFormat("managed heap ID length exceeds u16".into()))
}

/// Verify a chunked dataset spec for consistency before writing.
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

/// Reject invalid deflate compression levels.
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

fn validate_chunk_write_coords(shape: &[u64], chunk_dims: &[u64], coords: &[u64]) -> Result<()> {
    if coords.len() != shape.len() || chunk_dims.len() != shape.len() {
        return Err(Error::InvalidFormat(
            "chunk coordinate rank does not match dataset rank".into(),
        ));
    }
    for (idx, ((&coord, &chunk_dim), &dim)) in coords.iter().zip(chunk_dims).zip(shape).enumerate()
    {
        if chunk_dim == 0 {
            return Err(Error::InvalidFormat(format!(
                "chunk dimension {idx} is zero"
            )));
        }
        if coord >= dim {
            return Err(Error::InvalidFormat(format!(
                "chunk coordinate {idx}={coord} exceeds dataset dimension {dim}"
            )));
        }
        if coord % chunk_dim != 0 {
            return Err(Error::InvalidFormat(format!(
                "chunk coordinate {idx}={coord} is not aligned to chunk dimension {chunk_dim}"
            )));
        }
    }
    Ok(())
}

fn encode_chunk_payload(
    data: &[u8],
    element_size: usize,
    compression_level: Option<u32>,
    shuffle: bool,
    fletcher32: bool,
) -> Result<Vec<u8>> {
    let mut filtered = data.to_vec();
    if shuffle {
        let mut shuffled = vec![0u8; filtered.len()];
        crate::filters::shuffle::shuffle_into(&filtered, element_size, &mut shuffled)?;
        filtered = shuffled;
    }
    if let Some(level) = compression_level {
        let mut compressed = Vec::new();
        crate::filters::deflate::compress_into(&filtered, level, &mut compressed)?;
        filtered = compressed;
    }
    if fletcher32 {
        crate::filters::fletcher32::append_checksum_in_place(&mut filtered)?;
    }
    Ok(filtered)
}

/// Verify the data buffer matches the dataset shape times element size.
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

/// Reject empty or NUL-containing child names.
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

/// Verify an attribute payload size matches the shape and datatype.
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

/// Verify a list of attributes has no duplicate names.
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

/// Recursively validate a datatype spec.
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

    /// Return an error if `name` already exists under `parent`.
    fn ensure_child_name_available(&self, parent: &str, name: &str) -> Result<()> {
        validate_child_name(name)?;
        if self.child_name_exists(parent, name) {
            return Err(Error::InvalidFormat(format!(
                "link '{name}' already exists in group '{parent}'"
            )));
        }
        Ok(())
    }

    /// True if `parent` already has a link, hard-link, or special-link named `name`.
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

    /// Encode, allocate, and write a v2 object header plus any continuation chunks.
    fn write_v2_object_header(&mut self, messages: &[(u16, &[u8])], flags: u8) -> Result<u64> {
        let message_refs: Vec<ObjectHeaderMessageRef<'_>> = messages
            .iter()
            .map(|(msg_type, data)| ObjectHeaderMessageRef {
                msg_type: *msg_type,
                flags: if *msg_type == MSG_GROUP_INFO { 0x01 } else { 0 },
                creation_index: None,
                data,
            })
            .collect();

        let mut continuation_addrs = Vec::new();
        let encoded = loop {
            match encode_v2_with_continuations(
                &message_refs,
                flags,
                &continuation_addrs,
                OBJECT_HEADER_CHUNK_DATA_LIMIT,
                OBJECT_HEADER_CHUNK_DATA_LIMIT,
                self.sizeof_addr,
                self.sizeof_size,
            ) {
                Ok(encoded) => break encoded,
                Err(err)
                    if err.to_string().contains("continuation addresses")
                        && continuation_addrs.len() < messages.len() =>
                {
                    continuation_addrs.push(0);
                }
                Err(err) => return Err(err),
            }
        };

        let oh_addr = self.allocator.allocate(
            u64_from_usize_writer(encoded.prefix.len(), "object header size")?,
            8,
        );
        let real_continuation_addrs = encoded
            .continuation_chunks
            .iter()
            .map(|(_, image)| {
                Ok(self.allocator.allocate(
                    u64_from_usize_writer(image.len(), "object header continuation chunk size")?,
                    8,
                ))
            })
            .collect::<Result<Vec<_>>>()?;

        let encoded = encode_v2_with_continuations(
            &message_refs,
            flags,
            &real_continuation_addrs,
            OBJECT_HEADER_CHUNK_DATA_LIMIT,
            OBJECT_HEADER_CHUNK_DATA_LIMIT,
            self.sizeof_addr,
            self.sizeof_size,
        )?;
        self.write_at(oh_addr, &encoded.prefix)?;
        for (addr, image) in encoded.continuation_chunks {
            self.write_at(addr, &image)?;
        }

        Ok(oh_addr)
    }

    /// Write an empty group object header (will be rewritten with links in finalize).
    fn write_group_object_header(&mut self, extra_messages: &[(u16, &[u8])]) -> Result<u64> {
        let messages: Vec<(u16, &[u8])> = extra_messages.to_vec();
        self.write_v2_object_header(&messages, 0)
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

    /// Like [`create_compact_dataset_with_attrs`] including a fill value.
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
        let mut dtype_bytes = Vec::new();
        spec.dtype.encode_into(&mut dtype_bytes)?;
        let mut ds_bytes = Vec::new();
        encode_dataspace_for_spec_into(&mut ds_bytes, spec)?;
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

        let mut fill_value_bytes = Vec::new();
        encode_fill_value_message_into(&mut fill_value_bytes, fill)?;
        let mut messages: Vec<(u16, Vec<u8>)> = vec![
            (MSG_DATASPACE, ds_bytes),
            (MSG_DATATYPE, dtype_bytes),
            (MSG_FILL_VALUE, fill_value_bytes),
            (MSG_LAYOUT, layout_bytes),
        ];

        if attrs_need_dense_storage(attrs)? {
            let (heap_addr, btree_addr) = self.write_dense_attribute_storage(attrs)?;
            messages.push((MSG_ATTR_INFO, {
                let mut attr_info = Vec::new();
                encode_attr_info_message_into(
                    &mut attr_info,
                    heap_addr,
                    btree_addr,
                    self.sizeof_addr,
                )?;
                attr_info
            }));
        } else {
            for attr in attrs {
                messages.push((MSG_ATTRIBUTE, {
                    let mut attr_msg = Vec::new();
                    encode_attribute_message_into(
                        &mut attr_msg,
                        attr.name,
                        &attr.dtype,
                        attr.shape,
                        attr.data,
                    )?;
                    attr_msg
                }));
            }
        }

        let msg_refs: Vec<(u16, &[u8])> =
            messages.iter().map(|(t, d)| (*t, d.as_slice())).collect();
        let oh_addr = self.write_v2_object_header(&msg_refs, 0)?;

        self.links
            .push((parent.to_string(), spec.name.to_string(), oh_addr));

        Ok(oh_addr)
    }

    /// Like [`create_compact_dataset`] including a fill value.
    pub fn create_compact_dataset_with_fill(
        &mut self,
        parent: &str,
        spec: &DatasetSpec,
        fill: Option<FillValueSpec<'_>>,
    ) -> Result<u64> {
        self.ensure_child_name_available(parent, spec.name)?;
        validate_dataset_data_len(spec)?;
        let mut dtype_bytes = Vec::new();
        spec.dtype.encode_into(&mut dtype_bytes)?;
        let mut ds_bytes = Vec::new();
        encode_dataspace_for_spec_into(&mut ds_bytes, spec)?;
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

        let mut fill_value_bytes = Vec::new();
        encode_fill_value_message_into(&mut fill_value_bytes, fill)?;

        let messages: Vec<(u16, &[u8])> = vec![
            (MSG_DATASPACE, &ds_bytes),
            (MSG_DATATYPE, &dtype_bytes),
            (MSG_FILL_VALUE, &fill_value_bytes),
            (MSG_LAYOUT, &layout_bytes),
        ];

        let oh_addr = self.write_v2_object_header(&messages, 0)?;

        self.links
            .push((parent.to_string(), spec.name.to_string(), oh_addr));

        Ok(oh_addr)
    }

    /// Create a dataset with contiguous storage.
    pub fn create_dataset(&mut self, parent: &str, spec: &DatasetSpec) -> Result<u64> {
        self.create_dataset_with_fill(parent, spec, None)
    }

    /// Create a contiguous dataset with an explicit fill value (mirrors `H5D__create_named`).
    pub fn create_dataset_with_fill(
        &mut self,
        parent: &str,
        spec: &DatasetSpec,
        fill: Option<FillValueSpec<'_>>,
    ) -> Result<u64> {
        self.ensure_child_name_available(parent, spec.name)?;
        validate_dataset_data_len(spec)?;
        let mut dtype_bytes = Vec::new();
        spec.dtype.encode_into(&mut dtype_bytes)?;
        let mut ds_bytes = Vec::new();
        encode_dataspace_for_spec_into(&mut ds_bytes, spec)?;

        // Allocate space for the data
        let data_size = u64_from_usize_writer(spec.data.len(), "dataset data size")?;
        let data_addr = if data_size > 0 {
            let addr = self.allocator.allocate(data_size, 8);
            self.write_at(addr, spec.data)?;
            addr
        } else {
            UNDEF_ADDR
        };

        let mut layout_bytes = Vec::new();
        encode_contiguous_layout_into(
            &mut layout_bytes,
            data_addr,
            data_size,
            self.sizeof_addr,
            self.sizeof_size,
        )?;

        let mut fill_value_bytes = Vec::new();
        encode_fill_value_message_into(&mut fill_value_bytes, fill)?;

        let messages: Vec<(u16, &[u8])> = vec![
            (MSG_DATASPACE, &ds_bytes),
            (MSG_DATATYPE, &dtype_bytes),
            (MSG_FILL_VALUE, &fill_value_bytes),
            (MSG_LAYOUT, &layout_bytes),
        ];

        let oh_addr = self.write_v2_object_header(&messages, 0)?;

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
        self.create_vlen_utf8_string_dataset_with_attrs(
            parent,
            name,
            shape,
            strings,
            None,
            None,
            &[],
        )
    }

    /// Variable-length UTF-8 string dataset with attached attributes.
    pub fn create_vlen_utf8_string_dataset_with_attrs(
        &mut self,
        parent: &str,
        name: &str,
        shape: &[u64],
        strings: &[&str],
        max_shape: Option<&[u64]>,
        fill: Option<FillValueSpec<'_>>,
        attrs: &[AttrSpec],
    ) -> Result<u64> {
        self.create_vlen_utf8_string_dataset_with_attrs_and_vlen_fill(
            parent, name, shape, strings, max_shape, fill, None, attrs,
        )
    }

    /// Variable-length UTF-8 string dataset with attached attributes and optional string fill.
    pub(crate) fn create_vlen_utf8_string_dataset_with_attrs_and_vlen_fill(
        &mut self,
        parent: &str,
        name: &str,
        shape: &[u64],
        strings: &[&str],
        max_shape: Option<&[u64]>,
        fill: Option<FillValueSpec<'_>>,
        vlen_fill: Option<&str>,
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

        let data = self.prepare_vlen_utf8_string_data(strings)?;
        let vlen_payload_size = data.len();
        let data_addr = if vlen_payload_size == 0 {
            UNDEF_ADDR
        } else {
            let addr = self.allocator.allocate(
                u64::try_from(vlen_payload_size).map_err(|_| {
                    Error::InvalidFormat("vlen string descriptor payload too large".into())
                })?,
                8,
            );
            self.write_at(addr, &data)?;
            addr
        };

        let mut dtype_bytes = Vec::new();
        DtypeSpec::VarLenUtf8String.encode_into(&mut dtype_bytes)?;
        let spec = DatasetSpec {
            name,
            shape,
            max_shape,
            dtype: DtypeSpec::VarLenUtf8String,
            data: &data,
        };
        let mut ds_bytes = Vec::new();
        encode_dataspace_for_spec_into(&mut ds_bytes, &spec)?;
        let mut layout_bytes = Vec::new();
        encode_contiguous_layout_into(
            &mut layout_bytes,
            data_addr,
            u64::try_from(vlen_payload_size).map_err(|_| {
                Error::InvalidFormat("vlen string descriptor payload too large".into())
            })?,
            self.sizeof_addr,
            self.sizeof_size,
        )?;
        let vlen_fill_data;
        let effective_fill = if let Some(value) = vlen_fill {
            if fill.as_ref().and_then(|fill| fill.value).is_some() {
                return Err(Error::InvalidFormat(
                    "vlen UTF-8 fill value conflicts with raw fill-value bytes".into(),
                ));
            }
            let alloc_time = fill.map(|fill| fill.alloc_time).unwrap_or(1);
            let fill_time = fill.map(|fill| fill.fill_time).unwrap_or(2);
            vlen_fill_data = self.prepare_vlen_utf8_string_data(&[value])?;
            Some(FillValueSpec::with_value(
                alloc_time,
                fill_time,
                &vlen_fill_data,
            ))
        } else {
            fill
        };
        let mut fill_value_bytes = Vec::new();
        encode_fill_value_message_into(&mut fill_value_bytes, effective_fill)?;

        let mut messages: Vec<(u16, Vec<u8>)> = vec![
            (MSG_DATASPACE, ds_bytes),
            (MSG_DATATYPE, dtype_bytes),
            (MSG_FILL_VALUE, fill_value_bytes),
            (MSG_LAYOUT, layout_bytes),
        ];

        if attrs_need_dense_storage(attrs)? {
            let (heap_addr, btree_addr) = self.write_dense_attribute_storage(attrs)?;
            messages.push((MSG_ATTR_INFO, {
                let mut attr_info = Vec::new();
                encode_attr_info_message_into(
                    &mut attr_info,
                    heap_addr,
                    btree_addr,
                    self.sizeof_addr,
                )?;
                attr_info
            }));
        } else {
            for attr in attrs {
                let mut attr_bytes = Vec::new();
                encode_attribute_message_into(
                    &mut attr_bytes,
                    attr.name,
                    &attr.dtype,
                    attr.shape,
                    attr.data,
                )?;
                messages.push((MSG_ATTRIBUTE, attr_bytes));
            }
        }

        let msg_refs: Vec<(u16, &[u8])> =
            messages.iter().map(|(t, d)| (*t, d.as_slice())).collect();
        let oh_addr = self.write_v2_object_header(&msg_refs, 0)?;

        self.links
            .push((parent.to_string(), name.to_string(), oh_addr));

        Ok(oh_addr)
    }

    /// Chunked variable-length UTF-8 string dataset backed by a global heap.
    pub fn create_chunked_vlen_utf8_string_dataset_with_attrs(
        &mut self,
        parent: &str,
        name: &str,
        shape: &[u64],
        strings: &[&str],
        max_shape: Option<&[u64]>,
        chunk_dims: &[u64],
        compression_level: Option<u32>,
        shuffle: bool,
        fletcher32: bool,
        fill: Option<FillValueSpec<'_>>,
        attrs: &[AttrSpec],
    ) -> Result<u64> {
        self.create_chunked_vlen_utf8_string_dataset_with_attrs_and_vlen_fill(
            parent,
            name,
            shape,
            strings,
            max_shape,
            chunk_dims,
            compression_level,
            shuffle,
            fletcher32,
            fill,
            None,
            attrs,
        )
    }

    /// Chunked variable-length UTF-8 string dataset with optional string fill.
    pub(crate) fn create_chunked_vlen_utf8_string_dataset_with_attrs_and_vlen_fill(
        &mut self,
        parent: &str,
        name: &str,
        shape: &[u64],
        strings: &[&str],
        max_shape: Option<&[u64]>,
        chunk_dims: &[u64],
        compression_level: Option<u32>,
        shuffle: bool,
        fletcher32: bool,
        fill: Option<FillValueSpec<'_>>,
        vlen_fill: Option<&str>,
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

        let data = self.prepare_vlen_utf8_string_data(strings)?;
        let vlen_fill_data;
        let effective_fill = if let Some(value) = vlen_fill {
            if fill.as_ref().and_then(|fill| fill.value).is_some() {
                return Err(Error::InvalidFormat(
                    "vlen UTF-8 fill value conflicts with raw fill-value bytes".into(),
                ));
            }
            let alloc_time = fill.map(|fill| fill.alloc_time).unwrap_or(1);
            let fill_time = fill.map(|fill| fill.fill_time).unwrap_or(2);
            vlen_fill_data = self.prepare_vlen_utf8_string_data(&[value])?;
            Some(FillValueSpec::with_value(
                alloc_time,
                fill_time,
                &vlen_fill_data,
            ))
        } else {
            fill
        };
        let spec = DatasetSpec {
            name,
            shape,
            max_shape,
            dtype: DtypeSpec::VarLenUtf8String,
            data: &data,
        };
        self.create_chunked_dataset_with_attrs_and_fill(
            parent,
            &spec,
            chunk_dims,
            compression_level,
            shuffle,
            fletcher32,
            effective_fill,
            attrs,
        )
    }

    fn prepare_vlen_utf8_string_data(&mut self, strings: &[&str]) -> Result<Vec<u8>> {
        let mut heap_collections: Vec<Vec<&[u8]>> = Vec::new();
        let mut heap_refs = Vec::with_capacity(strings.len());
        for value in strings {
            if heap_collections
                .last()
                .map(|objects| objects.len() == MAX_GLOBAL_HEAP_OBJECTS_PER_COLLECTION)
                .unwrap_or(true)
            {
                heap_collections.push(Vec::new());
            }
            let collection_index = heap_collections
                .len()
                .checked_sub(1)
                .ok_or_else(|| Error::InvalidFormat("missing global heap collection".into()))?;
            let objects = heap_collections
                .last_mut()
                .expect("global heap collection was just created when missing");
            objects.push(value.as_bytes());
            let object_index = u32::try_from(objects.len())
                .map_err(|_| Error::InvalidFormat("global heap object index exceeds u32".into()))?;
            heap_refs.push((collection_index, object_index));
        }

        let mut heap_bytes = Vec::with_capacity(heap_collections.len());
        for objects in &heap_collections {
            heap_bytes.push(encode_global_heap_collection(objects, self.sizeof_size)?);
        }

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

        let mut heap_addrs = Vec::with_capacity(heap_bytes.len());
        for heap in &heap_bytes {
            heap_addrs.push(
                self.allocator.allocate(
                    u64::try_from(heap.len()).map_err(|_| {
                        Error::InvalidFormat("global heap collection too large".into())
                    })?,
                    8,
                ),
            );
        }

        let mut data = Vec::with_capacity(vlen_payload_size);
        for (value, (collection_index, object_index)) in strings.iter().zip(heap_refs) {
            let len = u32::try_from(value.len())
                .map_err(|_| Error::InvalidFormat("vlen string length exceeds u32".into()))?;
            let heap_addr = heap_addrs.get(collection_index).copied().ok_or_else(|| {
                Error::InvalidFormat("missing global heap address for vlen string".into())
            })?;
            data.extend_from_slice(&len.to_le_bytes());
            data.extend_from_slice(&heap_addr.to_le_bytes());
            data.extend_from_slice(&object_index.to_le_bytes());
        }

        for (heap_addr, heap) in heap_addrs.iter().copied().zip(&heap_bytes) {
            self.write_at(heap_addr, heap)?;
        }
        Ok(data)
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

    /// Like [`create_dataset_with_attrs`] including a fill value.
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
        let mut dtype_bytes = Vec::new();
        spec.dtype.encode_into(&mut dtype_bytes)?;
        let mut ds_bytes = Vec::new();
        encode_dataspace_for_spec_into(&mut ds_bytes, spec)?;

        let data_size = u64_from_usize_writer(spec.data.len(), "dataset data size")?;
        let data_addr = if data_size > 0 {
            let addr = self.allocator.allocate(data_size, 8);
            self.write_at(addr, spec.data)?;
            addr
        } else {
            UNDEF_ADDR
        };

        let mut layout_bytes = Vec::new();
        encode_contiguous_layout_into(
            &mut layout_bytes,
            data_addr,
            data_size,
            self.sizeof_addr,
            self.sizeof_size,
        )?;
        let mut fill_value_bytes = Vec::new();
        encode_fill_value_message_into(&mut fill_value_bytes, fill)?;

        let mut messages: Vec<(u16, Vec<u8>)> = vec![
            (MSG_DATASPACE, ds_bytes),
            (MSG_DATATYPE, dtype_bytes),
            (MSG_FILL_VALUE, fill_value_bytes),
            (MSG_LAYOUT, layout_bytes),
        ];

        if attrs_need_dense_storage(attrs)? {
            let (heap_addr, btree_addr) = self.write_dense_attribute_storage(attrs)?;
            messages.push((MSG_ATTR_INFO, {
                let mut attr_info = Vec::new();
                encode_attr_info_message_into(
                    &mut attr_info,
                    heap_addr,
                    btree_addr,
                    self.sizeof_addr,
                )?;
                attr_info
            }));
        } else {
            for attr in attrs {
                let mut attr_bytes = Vec::new();
                encode_attribute_message_into(
                    &mut attr_bytes,
                    attr.name,
                    &attr.dtype,
                    attr.shape,
                    attr.data,
                )?;
                messages.push((MSG_ATTRIBUTE, attr_bytes));
            }
        }

        let msg_refs: Vec<(u16, &[u8])> =
            messages.iter().map(|(t, d)| (*t, d.as_slice())).collect();
        let oh_addr = self.write_v2_object_header(&msg_refs, 0)?;

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

    /// Chunked dataset with explicit fill value.
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

    /// Chunked dataset combining filters and fill value.
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

    /// Chunked dataset combining filters, attributes, and fill value.
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
        let mut dtype_bytes = Vec::new();
        spec.dtype.encode_into(&mut dtype_bytes)?;
        let mut ds_bytes = Vec::new();
        encode_dataspace_for_spec_into(&mut ds_bytes, spec)?;
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

        // Write each chunk and collect v1 B-tree entries.
        let mut chunk_entries: Vec<ChunkBTreeEntry> = Vec::with_capacity(total_chunks);
        let has_filters = compression_level.is_some() || shuffle || fletcher32;

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

            let (addr, compressed_size) = if !has_filters && ndims == 1 {
                let start = usize_from_u64_writer(coords[0], "chunk start")?;
                let chunk_len = usize_from_u64_writer(chunk_dims[0], "chunk dimension")?;
                let data_len = usize_from_u64_writer(spec.shape[0], "dataset dimension")?;
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
                let src = spec.data.get(src_start..src_end).ok_or_else(|| {
                    Error::InvalidFormat("chunk source range exceeds data".into())
                })?;
                if copy_bytes == chunk_raw_bytes {
                    let addr = self.allocator.allocate(
                        u64::try_from(src.len())
                            .map_err(|_| Error::InvalidFormat("chunk size exceeds u64".into()))?,
                        1,
                    );
                    self.write_at(addr, src)?;
                    let compressed_size = u32::try_from(src.len())
                        .map_err(|_| Error::InvalidFormat("chunk size exceeds u32".into()))?;
                    (addr, compressed_size)
                } else {
                    let mut chunk_buf = vec![0u8; chunk_raw_bytes];
                    let dst = chunk_buf.get_mut(..copy_bytes).ok_or_else(|| {
                        Error::InvalidFormat("chunk destination range exceeds output".into())
                    })?;
                    dst.copy_from_slice(src);
                    let addr = self.allocator.allocate(
                        u64::try_from(chunk_buf.len())
                            .map_err(|_| Error::InvalidFormat("chunk size exceeds u64".into()))?,
                        1,
                    );
                    self.write_at(addr, &chunk_buf)?;
                    let compressed_size = u32::try_from(chunk_buf.len())
                        .map_err(|_| Error::InvalidFormat("chunk size exceeds u32".into()))?;
                    (addr, compressed_size)
                }
            } else {
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

                let filtered = encode_chunk_payload(
                    &chunk_buf,
                    element_size,
                    compression_level,
                    shuffle,
                    fletcher32,
                )?;

                let compressed_size = u32::try_from(filtered.len()).map_err(|_| {
                    Error::InvalidFormat("compressed chunk size exceeds u32".into())
                })?;
                let addr = self.allocator.allocate(
                    u64::try_from(filtered.len()).map_err(|_| {
                        Error::InvalidFormat("compressed chunk size exceeds u64".into())
                    })?,
                    1,
                );
                self.write_at(addr, &filtered)?;
                (addr, compressed_size)
            };

            chunk_entries.push(ChunkBTreeEntry {
                coords,
                chunk_size: compressed_size,
                filter_mask: 0,
                child_addr: addr,
            });
        }

        let element_size = u32::try_from(element_size)
            .map_err(|_| Error::InvalidFormat("chunk element size exceeds u32".into()))?;
        let use_single_chunk_index = total_chunks == 1 && spec.max_shape.is_none();
        let use_fixed_array_index = !use_single_chunk_index && spec.max_shape.is_none();
        let use_extensible_array_index =
            spec.max_shape.is_some() && total_chunks <= EXTENSIBLE_ARRAY_INDEX_BLOCK_ELEMENTS_LIMIT;

        let mut layout_bytes = Vec::new();
        if use_single_chunk_index {
            let entry = chunk_entries.first().ok_or_else(|| {
                Error::InvalidFormat("single-chunk dataset is missing chunk entry".into())
            })?;
            encode_single_chunk_layout_v4_into(
                &mut layout_bytes,
                entry.child_addr,
                chunk_dims,
                element_size,
                has_filters.then_some(u64::from(entry.chunk_size)),
                entry.filter_mask,
                self.sizeof_addr,
                self.sizeof_size,
            )?;
        } else if use_fixed_array_index {
            let fixed_array_addr = self.write_fixed_array_chunk_index(
                &chunk_entries,
                FIXED_ARRAY_CHUNK_PAGE_BITS,
                has_filters,
                chunk_raw_bytes,
            )?;
            encode_fixed_array_chunk_layout_v4_into(
                &mut layout_bytes,
                fixed_array_addr,
                chunk_dims,
                element_size,
                FIXED_ARRAY_CHUNK_PAGE_BITS,
                self.sizeof_addr,
            )?;
        } else if use_extensible_array_index {
            let index_block_elements = u8::try_from(total_chunks).map_err(|_| {
                Error::InvalidFormat("extensible-array inline chunk count exceeds u8".into())
            })?;
            let max_elements_bits = extensible_array_max_elements_bits(total_chunks)?;
            let extensible_array_addr = self.write_inline_extensible_array_chunk_index(
                &chunk_entries,
                index_block_elements,
                has_filters,
                chunk_raw_bytes,
                max_elements_bits,
            )?;
            encode_extensible_array_chunk_layout_v4_into(
                &mut layout_bytes,
                extensible_array_addr,
                chunk_dims,
                element_size,
                max_elements_bits,
                index_block_elements,
                1,
                1,
                1,
                self.sizeof_addr,
            )?;
        } else {
            return Err(Error::Unsupported(format!(
                "max-shape chunked writer currently supports at most {EXTENSIBLE_ARRAY_INDEX_BLOCK_ELEMENTS_LIMIT} chunks with inline extensible-array indexing; larger chunk grids need full extensible-array or v2 B-tree index creation"
            )));
        }

        // Encode filter pipeline message
        let mut pipeline_bytes = Vec::new();
        encode_filter_pipeline_into(&mut pipeline_bytes, compression_level, shuffle, fletcher32)?;

        let mut fill_value_bytes = Vec::new();
        encode_fill_value_message_into(&mut fill_value_bytes, fill)?;

        let mut messages: Vec<(u16, Vec<u8>)> = vec![
            (MSG_DATASPACE, ds_bytes),
            (MSG_DATATYPE, dtype_bytes),
            (MSG_FILL_VALUE, fill_value_bytes),
            (MSG_LAYOUT, layout_bytes),
        ];
        if !pipeline_bytes.is_empty() {
            messages.push((MSG_FILTER_PIPELINE, pipeline_bytes));
        }

        if attrs_need_dense_storage(attrs)? {
            let (heap_addr, btree_addr) = self.write_dense_attribute_storage(attrs)?;
            messages.push((MSG_ATTR_INFO, {
                let mut attr_info = Vec::new();
                encode_attr_info_message_into(
                    &mut attr_info,
                    heap_addr,
                    btree_addr,
                    self.sizeof_addr,
                )?;
                attr_info
            }));
        } else {
            for attr in attrs {
                messages.push((MSG_ATTRIBUTE, {
                    let mut attr_msg = Vec::new();
                    encode_attribute_message_into(
                        &mut attr_msg,
                        attr.name,
                        &attr.dtype,
                        attr.shape,
                        attr.data,
                    )?;
                    attr_msg
                }));
            }
        }

        let msg_refs: Vec<(u16, &[u8])> =
            messages.iter().map(|(t, d)| (*t, d.as_slice())).collect();
        let oh_addr = self.write_v2_object_header(&msg_refs, 0)?;

        self.links
            .push((parent.to_string(), spec.name.to_string(), oh_addr));

        Ok(oh_addr)
    }

    /// Chunked dataset from explicitly supplied full chunks.
    ///
    /// Only the supplied chunks are allocated and indexed. Missing chunks read
    /// as the fill value, matching sparse chunked HDF5 storage.
    pub fn create_chunked_dataset_from_chunks_with_attrs_and_fill(
        &mut self,
        parent: &str,
        spec: &DatasetSpec,
        chunk_dims: &[u64],
        chunks: &[ChunkWriteSpec<'_>],
        compression_level: Option<u32>,
        shuffle: bool,
        fletcher32: bool,
        fill: Option<FillValueSpec<'_>>,
        attrs: &[AttrSpec],
    ) -> Result<u64> {
        if chunks.is_empty() {
            return self.create_sparse_chunked_dataset_with_attrs_and_fill(
                parent,
                spec,
                chunk_dims,
                compression_level,
                shuffle,
                fletcher32,
                fill,
                attrs,
            );
        }

        self.ensure_child_name_available(parent, spec.name)?;
        validate_unique_attr_names(attrs)?;
        let mut dtype_bytes = Vec::new();
        spec.dtype.encode_into(&mut dtype_bytes)?;
        let mut ds_bytes = Vec::new();
        encode_dataspace_for_spec_into(&mut ds_bytes, spec)?;
        let element_size = usize::try_from(spec.dtype.size())
            .map_err(|_| Error::InvalidFormat("dataset element size exceeds usize".into()))?;
        let ndims = spec.shape.len();
        validate_deflate_level(compression_level)?;
        let chunk_raw_bytes = validate_chunked_dataset_spec(spec, chunk_dims)?;

        let mut seen = HashSet::with_capacity(chunks.len());
        let mut chunk_entries = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            validate_chunk_write_coords(spec.shape, chunk_dims, chunk.coords)?;
            if chunk.data.len() != chunk_raw_bytes {
                return Err(Error::InvalidFormat(format!(
                    "chunk at {:?} has {} bytes, expected {chunk_raw_bytes}",
                    chunk.coords,
                    chunk.data.len()
                )));
            }
            if !seen.insert(chunk.coords.to_vec()) {
                return Err(Error::InvalidFormat(format!(
                    "duplicate chunk coordinates {:?}",
                    chunk.coords
                )));
            }

            let filtered = encode_chunk_payload(
                chunk.data,
                element_size,
                compression_level,
                shuffle,
                fletcher32,
            )?;
            let compressed_size = u32::try_from(filtered.len())
                .map_err(|_| Error::InvalidFormat("compressed chunk size exceeds u32".into()))?;
            let addr = self.allocator.allocate(
                u64::try_from(filtered.len()).map_err(|_| {
                    Error::InvalidFormat("compressed chunk size exceeds u64".into())
                })?,
                1,
            );
            self.write_at(addr, &filtered)?;
            chunk_entries.push(ChunkBTreeEntry {
                coords: chunk.coords.to_vec(),
                chunk_size: compressed_size,
                filter_mask: 0,
                child_addr: addr,
            });
        }

        chunk_entries.sort_by(|left, right| left.coords.cmp(&right.coords));

        let element_size_u32 = u32::try_from(element_size)
            .map_err(|_| Error::InvalidFormat("chunk element size exceeds u32".into()))?;
        let btree_addr =
            self.write_chunk_btree_entries_v1(&chunk_entries, ndims, element_size_u32)?;

        let mut layout_bytes = Vec::new();
        encode_chunked_layout_v3_into(
            &mut layout_bytes,
            btree_addr,
            chunk_dims,
            element_size_u32,
            self.sizeof_addr,
        )?;

        let mut pipeline_bytes = Vec::new();
        encode_filter_pipeline_into(&mut pipeline_bytes, compression_level, shuffle, fletcher32)?;

        let mut fill_value_bytes = Vec::new();
        encode_fill_value_message_into(&mut fill_value_bytes, fill)?;

        let mut messages: Vec<(u16, Vec<u8>)> = vec![
            (MSG_DATASPACE, ds_bytes),
            (MSG_DATATYPE, dtype_bytes),
            (MSG_FILL_VALUE, fill_value_bytes),
            (MSG_LAYOUT, layout_bytes),
        ];
        if !pipeline_bytes.is_empty() {
            messages.push((MSG_FILTER_PIPELINE, pipeline_bytes));
        }

        if attrs_need_dense_storage(attrs)? {
            let (heap_addr, btree_addr) = self.write_dense_attribute_storage(attrs)?;
            messages.push((MSG_ATTR_INFO, {
                let mut attr_info = Vec::new();
                encode_attr_info_message_into(
                    &mut attr_info,
                    heap_addr,
                    btree_addr,
                    self.sizeof_addr,
                )?;
                attr_info
            }));
        } else {
            for attr in attrs {
                messages.push((MSG_ATTRIBUTE, {
                    let mut attr_msg = Vec::new();
                    encode_attribute_message_into(
                        &mut attr_msg,
                        attr.name,
                        &attr.dtype,
                        attr.shape,
                        attr.data,
                    )?;
                    attr_msg
                }));
            }
        }

        let msg_refs: Vec<(u16, &[u8])> =
            messages.iter().map(|(t, d)| (*t, d.as_slice())).collect();
        let oh_addr = self.write_v2_object_header(&msg_refs, 0)?;

        self.links
            .push((parent.to_string(), spec.name.to_string(), oh_addr));

        Ok(oh_addr)
    }

    /// Create a chunked dataset with no allocated chunks.
    ///
    /// The chunk index address is left undefined so readers materialize the
    /// dataset from its fill value. This mirrors the sparse chunked layout
    /// libhdf5 writes before any chunks are allocated.
    pub fn create_sparse_chunked_dataset_with_attrs_and_fill(
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
        let mut dtype_bytes = Vec::new();
        spec.dtype.encode_into(&mut dtype_bytes)?;
        let mut ds_bytes = Vec::new();
        encode_dataspace_for_spec_into(&mut ds_bytes, spec)?;
        let element_size = usize::try_from(spec.dtype.size())
            .map_err(|_| Error::InvalidFormat("dataset element size exceeds usize".into()))?;
        validate_deflate_level(compression_level)?;
        validate_chunked_dataset_spec(spec, chunk_dims)?;

        let element_size_u32 = u32::try_from(element_size)
            .map_err(|_| Error::InvalidFormat("chunk element size exceeds u32".into()))?;
        let mut layout_bytes = Vec::new();
        encode_chunked_layout_v3_into(
            &mut layout_bytes,
            UNDEF_ADDR,
            chunk_dims,
            element_size_u32,
            self.sizeof_addr,
        )?;

        let mut pipeline_bytes = Vec::new();
        encode_filter_pipeline_into(&mut pipeline_bytes, compression_level, shuffle, fletcher32)?;

        let mut fill_value_bytes = Vec::new();
        encode_fill_value_message_into(&mut fill_value_bytes, fill)?;

        let mut messages: Vec<(u16, Vec<u8>)> = vec![
            (MSG_DATASPACE, ds_bytes),
            (MSG_DATATYPE, dtype_bytes),
            (MSG_FILL_VALUE, fill_value_bytes),
            (MSG_LAYOUT, layout_bytes),
        ];
        if !pipeline_bytes.is_empty() {
            messages.push((MSG_FILTER_PIPELINE, pipeline_bytes));
        }

        if attrs_need_dense_storage(attrs)? {
            let (heap_addr, btree_addr) = self.write_dense_attribute_storage(attrs)?;
            messages.push((MSG_ATTR_INFO, {
                let mut attr_info = Vec::new();
                encode_attr_info_message_into(
                    &mut attr_info,
                    heap_addr,
                    btree_addr,
                    self.sizeof_addr,
                )?;
                attr_info
            }));
        } else {
            for attr in attrs {
                messages.push((MSG_ATTRIBUTE, {
                    let mut attr_msg = Vec::new();
                    encode_attribute_message_into(
                        &mut attr_msg,
                        attr.name,
                        &attr.dtype,
                        attr.shape,
                        attr.data,
                    )?;
                    attr_msg
                }));
            }
        }

        let msg_refs: Vec<(u16, &[u8])> =
            messages.iter().map(|(t, d)| (*t, d.as_slice())).collect();
        let oh_addr = self.write_v2_object_header(&msg_refs, 0)?;

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

    /// Write the leaf entries of a v1 chunk B-tree.
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
        let final_coords = entries[entries.len() - 1].coords.as_slice();

        if entries.len() <= 64 {
            let leaf =
                self.encode_chunk_btree_node_v1(0, entries, final_coords, ndims, element_size)?;
            self.write_at(root_addr, &leaf)?;
            return Ok(root_addr);
        }

        let (root_level, root_entries) =
            self.write_chunk_btree_v1_child_level(entries, 0, ndims, element_size, node_size)?;
        let root = self.encode_chunk_btree_node_v1(
            root_level,
            &root_entries,
            final_coords,
            ndims,
            element_size,
        )?;
        self.write_at(root_addr, &root)?;
        Ok(root_addr)
    }

    fn write_chunk_btree_v1_child_level(
        &mut self,
        entries: &[ChunkBTreeEntry],
        child_level: u8,
        ndims: usize,
        element_size: u32,
        node_size: usize,
    ) -> Result<(u8, Vec<ChunkBTreeEntry>)> {
        let parent_level = child_level
            .checked_add(1)
            .ok_or_else(|| Error::InvalidFormat("chunk B-tree level overflow".into()))?;
        let child_count = entries.len().div_ceil(64);
        let mut parent_entries = Vec::with_capacity(child_count);

        for child_entries in entries.chunks(64) {
            let child_addr = self.allocator.allocate(
                u64_from_usize_writer(node_size, "chunk B-tree node size")?,
                8,
            );
            let child_final_coords = child_entries[child_entries.len() - 1].coords.as_slice();
            let child = self.encode_chunk_btree_node_v1(
                child_level,
                child_entries,
                child_final_coords,
                ndims,
                element_size,
            )?;
            self.write_at(child_addr, &child)?;
            parent_entries.push(ChunkBTreeEntry {
                coords: child_entries[0].coords.clone(),
                chunk_size: child_entries[0].chunk_size,
                filter_mask: child_entries[0].filter_mask,
                child_addr,
            });
        }

        if parent_entries.len() <= 64 {
            Ok((parent_level, parent_entries))
        } else {
            self.write_chunk_btree_v1_child_level(
                &parent_entries,
                parent_level,
                ndims,
                element_size,
                node_size,
            )
        }
    }

    /// Compute the encoded size of one v1 chunk B-tree node.
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

    /// Encode a v1 chunk B-tree node payload.
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

        buf.extend_from_slice(b"TREE");
        buf.push(1);
        buf.push(level);
        buf.extend_from_slice(
            &u16::try_from(entries.len())
                .map_err(|_| Error::InvalidFormat("chunk B-tree entry count exceeds u16".into()))?
                .to_le_bytes(),
        );
        append_encoded_addr(&mut buf, UNDEF_ADDR, self.sizeof_addr)?;
        append_encoded_addr(&mut buf, UNDEF_ADDR, self.sizeof_addr)?;

        for entry in entries {
            buf.extend_from_slice(&entry.chunk_size.to_le_bytes());
            buf.extend_from_slice(&entry.filter_mask.to_le_bytes());
            for &coord in &entry.coords {
                buf.extend_from_slice(&coord.to_le_bytes());
            }
            buf.extend_from_slice(&0u64.to_le_bytes());
            append_encoded_addr(&mut buf, entry.child_addr, self.sizeof_addr)?;
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

    /// Write a v4 fixed-array chunk index for a fully materialized chunk grid.
    fn write_fixed_array_chunk_index(
        &mut self,
        entries: &[ChunkBTreeEntry],
        page_bits: u8,
        filtered: bool,
        unfiltered_chunk_bytes: usize,
    ) -> Result<u64> {
        let sa = usize::from(self.sizeof_addr);
        let ss = usize::from(self.sizeof_size);
        if page_bits == 0 {
            return Err(Error::InvalidFormat(
                "fixed-array chunk page bits must be positive".into(),
            ));
        }
        let page_elements = 1usize.checked_shl(u32::from(page_bits)).ok_or_else(|| {
            Error::InvalidFormat("fixed-array page element count overflow".into())
        })?;
        if u8::try_from(sa).is_err() {
            return Err(Error::InvalidFormat(
                "fixed-array address width exceeds raw element size field".into(),
            ));
        }
        let chunk_size_len = if filtered {
            filtered_chunk_size_len_v4(unfiltered_chunk_bytes)
        } else {
            0
        };
        let raw_element_size = if filtered {
            checked_usize_sum_writer(
                &[sa, chunk_size_len, 4],
                "fixed-array chunk raw element size",
            )?
        } else {
            sa
        };
        let class_id = if filtered { 1 } else { 0 };
        let raw_element_size_u8 = u8::try_from(raw_element_size)
            .map_err(|_| Error::InvalidFormat("fixed-array raw element size exceeds u8".into()))?;

        let header_len =
            checked_usize_sum_writer(&[4, 1, 1, 1, 1, ss, sa, 4], "fixed-array chunk header size")?;
        let prefix_payload_len =
            checked_usize_sum_writer(&[4, 1, 1, sa], "fixed-array chunk data block prefix size")?;
        let paginated = entries.len() > page_elements;
        let data_block_len = if paginated {
            let page_count = entries.len().div_ceil(page_elements);
            let page_init_len = page_count.div_ceil(8);
            let page_payload_len =
                page_elements.checked_mul(raw_element_size).ok_or_else(|| {
                    Error::InvalidFormat("fixed-array chunk page payload size overflow".into())
                })?;
            let page_len =
                checked_usize_sum_writer(&[page_payload_len, 4], "fixed-array chunk page size")?;
            let pages_len = page_count.checked_mul(page_len).ok_or_else(|| {
                Error::InvalidFormat("fixed-array chunk page block size overflow".into())
            })?;
            checked_usize_sum_writer(
                &[prefix_payload_len, page_init_len, 4, pages_len],
                "fixed-array chunk data block size",
            )?
        } else {
            let payload_len = entries.len().checked_mul(raw_element_size).ok_or_else(|| {
                Error::InvalidFormat("fixed-array chunk payload size overflow".into())
            })?;
            checked_usize_sum_writer(
                &[prefix_payload_len, payload_len, 4],
                "fixed-array chunk data block size",
            )?
        };

        let header_addr = self.allocator.allocate(
            u64_from_usize_writer(header_len, "fixed-array chunk header size")?,
            8,
        );
        let data_block_addr = self.allocator.allocate(
            u64_from_usize_writer(data_block_len, "fixed-array chunk data block size")?,
            8,
        );

        let mut data_block = Vec::with_capacity(data_block_len);
        data_block.extend_from_slice(b"FADB");
        data_block.push(0); // version
        data_block.push(class_id);
        append_encoded_addr(&mut data_block, header_addr, self.sizeof_addr)?;
        if paginated {
            let page_count = entries.len().div_ceil(page_elements);
            let page_init_len = page_count.div_ceil(8);
            let page_init_start = data_block.len();
            data_block.resize(page_init_start + page_init_len, 0);
            for page_index in 0..page_count {
                let byte = data_block
                    .get_mut(page_init_start + page_index / 8)
                    .ok_or_else(|| {
                        Error::InvalidFormat("fixed-array chunk page bitmap overflow".into())
                    })?;
                *byte |= 0x80 >> (page_index % 8);
            }
            let prefix_checksum = checksum_metadata(&data_block);
            data_block.extend_from_slice(&prefix_checksum.to_le_bytes());

            let page_payload_len =
                page_elements.checked_mul(raw_element_size).ok_or_else(|| {
                    Error::InvalidFormat("fixed-array chunk page payload size overflow".into())
                })?;
            for page_entries in entries.chunks(page_elements) {
                let page_start = data_block.len();
                for entry in page_entries {
                    append_encoded_addr(&mut data_block, entry.child_addr, self.sizeof_addr)?;
                    if filtered {
                        append_encoded_uint(
                            &mut data_block,
                            u64::from(entry.chunk_size),
                            chunk_size_len,
                            "fixed-array filtered chunk size",
                        )?;
                        data_block.extend_from_slice(&entry.filter_mask.to_le_bytes());
                    }
                }
                data_block.resize(page_start + page_payload_len, 0);
                let page_checksum = checksum_metadata(&data_block[page_start..]);
                data_block.extend_from_slice(&page_checksum.to_le_bytes());
            }
        } else {
            for entry in entries {
                append_encoded_addr(&mut data_block, entry.child_addr, self.sizeof_addr)?;
                if filtered {
                    append_encoded_uint(
                        &mut data_block,
                        u64::from(entry.chunk_size),
                        chunk_size_len,
                        "fixed-array filtered chunk size",
                    )?;
                    data_block.extend_from_slice(&entry.filter_mask.to_le_bytes());
                }
            }
            let data_checksum = checksum_metadata(&data_block);
            data_block.extend_from_slice(&data_checksum.to_le_bytes());
        }
        self.write_at(data_block_addr, &data_block)?;

        let mut header = Vec::with_capacity(header_len);
        header.extend_from_slice(b"FAHD");
        header.push(0); // version
        header.push(class_id);
        header.push(raw_element_size_u8);
        header.push(page_bits);
        append_encoded_size(
            &mut header,
            u64_from_usize_writer(entries.len(), "fixed-array chunk element count")?,
            self.sizeof_size,
        )?;
        append_encoded_addr(&mut header, data_block_addr, self.sizeof_addr)?;
        let header_checksum = checksum_metadata(&header);
        header.extend_from_slice(&header_checksum.to_le_bytes());
        self.write_at(header_addr, &header)?;

        Ok(header_addr)
    }

    /// Write a v4 extensible-array chunk index whose chunks fit in the index block.
    fn write_inline_extensible_array_chunk_index(
        &mut self,
        entries: &[ChunkBTreeEntry],
        index_block_elements: u8,
        filtered: bool,
        unfiltered_chunk_bytes: usize,
        max_elements_bits: u8,
    ) -> Result<u64> {
        let sa = usize::from(self.sizeof_addr);
        let ss = usize::from(self.sizeof_size);
        if entries.is_empty() {
            return Err(Error::InvalidFormat(
                "cannot write empty extensible-array chunk index".into(),
            ));
        }
        if usize::from(index_block_elements) != entries.len() {
            return Err(Error::InvalidFormat(
                "inline extensible-array chunk count mismatch".into(),
            ));
        }
        let chunk_size_len = if filtered {
            filtered_chunk_size_len_v4(unfiltered_chunk_bytes)
        } else {
            0
        };
        let raw_element_size = if filtered {
            checked_usize_sum_writer(
                &[sa, chunk_size_len, 4],
                "extensible-array chunk raw element size",
            )?
        } else {
            sa
        };
        let raw_element_size_u8 = u8::try_from(raw_element_size).map_err(|_| {
            Error::InvalidFormat("extensible-array raw element size exceeds u8".into())
        })?;
        let class_id = if filtered { 1 } else { 0 };
        let super_block_addr_count =
            usize::from(max_elements_bits)
                .checked_add(1)
                .ok_or_else(|| {
                    Error::InvalidFormat("extensible-array super block count overflow".into())
                })?;

        let header_len =
            checked_usize_sum_writer(&[4, 8, ss * 6, sa, 4], "extensible-array chunk header size")?;
        let inline_bytes = entries.len().checked_mul(raw_element_size).ok_or_else(|| {
            Error::InvalidFormat("extensible-array inline element bytes overflow".into())
        })?;
        let super_block_addr_bytes = super_block_addr_count.checked_mul(sa).ok_or_else(|| {
            Error::InvalidFormat("extensible-array super block address bytes overflow".into())
        })?;
        let index_block_len = checked_usize_sum_writer(
            &[4, 1, 1, sa, inline_bytes, super_block_addr_bytes, 4],
            "extensible-array chunk index block size",
        )?;

        let header_addr = self.allocator.allocate(
            u64_from_usize_writer(header_len, "extensible-array chunk header size")?,
            8,
        );
        let index_block_addr = self.allocator.allocate(
            u64_from_usize_writer(index_block_len, "extensible-array chunk index block size")?,
            8,
        );

        let mut index = Vec::with_capacity(index_block_len);
        index.extend_from_slice(b"EAIB");
        index.push(0); // version
        index.push(class_id);
        append_encoded_addr(&mut index, header_addr, self.sizeof_addr)?;
        for entry in entries {
            append_encoded_addr(&mut index, entry.child_addr, self.sizeof_addr)?;
            if filtered {
                append_encoded_uint(
                    &mut index,
                    u64::from(entry.chunk_size),
                    chunk_size_len,
                    "extensible-array filtered chunk size",
                )?;
                index.extend_from_slice(&entry.filter_mask.to_le_bytes());
            }
        }
        for _ in 0..super_block_addr_count {
            append_encoded_addr(&mut index, UNDEF_ADDR, self.sizeof_addr)?;
        }
        let index_checksum = checksum_metadata(&index);
        index.extend_from_slice(&index_checksum.to_le_bytes());
        self.write_at(index_block_addr, &index)?;

        let mut header = Vec::with_capacity(header_len);
        header.extend_from_slice(b"EAHD");
        header.push(0); // version
        header.push(class_id);
        header.push(raw_element_size_u8);
        header.push(max_elements_bits);
        header.push(index_block_elements);
        header.push(1); // data block min elements
        header.push(1); // super block min data pointers
        header.push(1); // max data block page elements bits
        append_encoded_size(&mut header, 0, self.sizeof_size)?; // super block count
        append_encoded_size(&mut header, 0, self.sizeof_size)?; // super block size
        append_encoded_size(&mut header, 0, self.sizeof_size)?; // data block count
        append_encoded_size(&mut header, 0, self.sizeof_size)?; // data block size
        append_encoded_size(
            &mut header,
            u64_from_usize_writer(entries.len(), "extensible-array max index set")?,
            self.sizeof_size,
        )?;
        append_encoded_size(
            &mut header,
            u64_from_usize_writer(entries.len(), "extensible-array realized elements")?,
            self.sizeof_size,
        )?;
        append_encoded_addr(&mut header, index_block_addr, self.sizeof_addr)?;
        let header_checksum = checksum_metadata(&header);
        header.extend_from_slice(&header_checksum.to_le_bytes());
        self.write_at(header_addr, &header)?;

        Ok(header_addr)
    }

    /// Write the fractal-heap and B-tree storage for dense links (mirrors `H5G__dense_create`).
    fn write_dense_link_storage(&mut self, links: &[(String, Vec<u8>)]) -> Result<(u64, u64)> {
        let payloads: Vec<Vec<u8>> = links
            .iter()
            .map(|(_, link_bytes)| link_bytes.clone())
            .collect();
        let (heap_addr, heap_ids) = self.write_managed_fractal_heap(&payloads, 7)?;
        let mut records = Vec::with_capacity(payloads.len());
        for ((name, _), heap_id) in links.iter().zip(heap_ids) {
            let record_len =
                checked_usize_sum_writer(&[4, heap_id.len()], "dense link record size")?;
            let mut record = Vec::with_capacity(record_len);
            record.extend_from_slice(&dense_name_hash(name).to_le_bytes());
            record.extend_from_slice(&heap_id);
            records.push(record);
        }
        records.sort_by_key(|record| dense_record_hash(record).unwrap_or(u32::MAX));

        let btree_addr = self.write_dense_name_btree(5, &records)?;
        Ok((heap_addr, btree_addr))
    }

    /// Write the fractal-heap and B-tree storage for dense attributes (mirrors `H5A__dense_create`).
    fn write_dense_attribute_storage(&mut self, attrs: &[AttrSpec<'_>]) -> Result<(u64, u64)> {
        validate_unique_attr_names(attrs)?;
        let mut payloads = Vec::with_capacity(attrs.len());
        for attr in attrs {
            let mut attr_bytes = Vec::new();
            encode_attribute_message_into(
                &mut attr_bytes,
                attr.name,
                &attr.dtype,
                attr.shape,
                attr.data,
            )?;
            payloads.push(attr_bytes);
        }

        let (heap_addr, heap_ids) = self.write_managed_fractal_heap(&payloads, 8)?;
        let mut records = Vec::with_capacity(payloads.len());
        for (creation_order, (attr, heap_id)) in attrs.iter().zip(heap_ids).enumerate() {
            let record_len =
                checked_usize_sum_writer(&[heap_id.len(), 9], "dense attribute record size")?;
            let mut record = Vec::with_capacity(record_len);
            record.extend_from_slice(&heap_id);
            record.push(0);
            record.extend_from_slice(
                &u32::try_from(creation_order)
                    .map_err(|_| {
                        Error::InvalidFormat("dense attribute creation order exceeds u32".into())
                    })?
                    .to_le_bytes(),
            );
            record.extend_from_slice(&dense_name_hash(attr.name).to_le_bytes());
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

    /// Write a minimal managed fractal heap containing the given records.
    fn write_managed_fractal_heap(
        &mut self,
        payloads: &[Vec<u8>],
        min_heap_id_len: u16,
    ) -> Result<(u64, Vec<Vec<u8>>)> {
        let offset_bytes = 4usize;
        let heap_id_len = dense_heap_id_len_for_payloads(payloads, offset_bytes, min_heap_id_len)?;
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
        let payload_bytes = payloads.iter().try_fold(0usize, |acc, payload| {
            acc.checked_add(payload.len())
                .ok_or_else(|| Error::InvalidFormat("managed heap payload size overflow".into()))
        })?;
        let max_payload_len = payloads.iter().map(Vec::len).max().unwrap_or(0);
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
        append_encoded_addr(&mut direct, heap_addr, self.sizeof_addr)?;
        direct.extend_from_slice(&0u32.to_le_bytes());
        direct.extend_from_slice(&0u32.to_le_bytes());

        let mut heap_ids = Vec::with_capacity(payloads.len());
        for payload in payloads {
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

        let heap = self.encode_minimal_fractal_heap(
            heap_id_len,
            payloads.len(),
            u64_from_usize_writer(max_payload_len, "managed heap max payload length")?,
            u64_from_usize_writer(block_size, "fractal heap direct block size")?,
            direct_addr,
        )?;
        debug_assert_eq!(heap.len(), heap_header_len);
        self.write_at(heap_addr, &heap)?;
        Ok((heap_addr, heap_ids))
    }

    /// Encoded size of the minimal fractal-heap header used by this writer.
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

    /// Encode the minimal fractal-heap header for dense storage.
    fn encode_minimal_fractal_heap(
        &self,
        heap_id_len: u16,
        managed_nobjs: usize,
        max_managed_obj_size: u64,
        managed_alloc_size: u64,
        root_block_addr: u64,
    ) -> Result<Vec<u8>> {
        let mut buf = Vec::new();
        let free_space = 0u64;
        let managed_nobjs = u64_from_usize_writer(managed_nobjs, "managed heap object count")?;
        let max_managed_obj_size = u32::try_from(max_managed_obj_size.max(4096)).map_err(|_| {
            Error::Unsupported("managed heap payload exceeds u32 max object size".into())
        })?;

        buf.extend_from_slice(b"FRHP");
        buf.push(0);
        buf.extend_from_slice(&heap_id_len.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.push(0x02);
        buf.extend_from_slice(&max_managed_obj_size.to_le_bytes());
        append_encoded_size(&mut buf, 0, self.sizeof_size)?;
        append_encoded_addr(&mut buf, UNDEF_ADDR, self.sizeof_addr)?;
        append_encoded_size(&mut buf, free_space, self.sizeof_size)?;
        append_encoded_addr(&mut buf, UNDEF_ADDR, self.sizeof_addr)?;
        append_encoded_size(&mut buf, managed_alloc_size, self.sizeof_size)?;
        append_encoded_size(&mut buf, managed_alloc_size, self.sizeof_size)?;
        append_encoded_size(&mut buf, 0, self.sizeof_size)?;
        append_encoded_size(&mut buf, managed_nobjs, self.sizeof_size)?;
        append_encoded_size(&mut buf, 0, self.sizeof_size)?;
        append_encoded_size(&mut buf, 0, self.sizeof_size)?;
        append_encoded_size(&mut buf, 0, self.sizeof_size)?;
        append_encoded_size(&mut buf, 0, self.sizeof_size)?;
        buf.extend_from_slice(&4u16.to_le_bytes());
        append_encoded_size(&mut buf, managed_alloc_size, self.sizeof_size)?;
        append_encoded_size(&mut buf, managed_alloc_size.max(65536), self.sizeof_size)?;
        buf.extend_from_slice(&32u16.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        append_encoded_addr(&mut buf, root_block_addr, self.sizeof_addr)?;
        buf.extend_from_slice(&0u16.to_le_bytes());
        let checksum = checksum_metadata(&buf);
        buf.extend_from_slice(&checksum.to_le_bytes());
        Ok(buf)
    }

    /// Write a v2 B-tree indexed by name hash, for dense links or attributes.
    fn write_dense_name_btree(&mut self, tree_type: u8, records: &[Vec<u8>]) -> Result<u64> {
        const DENSE_BTREE_NODE_SIZE: usize = 512;

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

        let level_info = dense_btree_level_info(
            DENSE_BTREE_NODE_SIZE,
            record_size,
            records.len(),
            self.sizeof_addr,
        )?;
        let leaf_max_records = level_info[0].max_nrecords;
        let (root_addr, root_nrecords, depth, node_size) = if tree_type != 5 {
            let record_bytes = records
                .len()
                .checked_mul(record_size)
                .ok_or_else(|| Error::InvalidFormat("dense B-tree record bytes overflow".into()))?;
            let node_payload_size = 10usize
                .checked_add(record_bytes)
                .ok_or_else(|| Error::InvalidFormat("dense B-tree node size overflow".into()))?;
            let node_size = DENSE_BTREE_NODE_SIZE.max(node_payload_size);
            let root_addr = self.write_dense_btree_leaf_node(tree_type, records, node_size)?;
            (root_addr, records.len(), 0u16, node_size)
        } else if records.len() <= leaf_max_records {
            (
                self.write_dense_btree_leaf_node(tree_type, records, DENSE_BTREE_NODE_SIZE)?,
                records.len(),
                0u16,
                DENSE_BTREE_NODE_SIZE,
            )
        } else {
            let depth = u16::try_from(level_info.len() - 1)
                .map_err(|_| Error::Unsupported("dense name B-tree depth exceeds u16".into()))?;
            let root = self.write_dense_btree_subtree(
                tree_type,
                records,
                record_size,
                DENSE_BTREE_NODE_SIZE,
                &level_info,
                usize::from(depth),
            )?;
            (root.addr, root.node_nrecords, depth, DENSE_BTREE_NODE_SIZE)
        };

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
        header.extend_from_slice(&depth.to_le_bytes());
        header.push(100);
        header.push(40);
        append_encoded_addr(&mut header, root_addr, self.sizeof_addr)?;
        header.extend_from_slice(
            &u16::try_from(root_nrecords)
                .map_err(|_| Error::InvalidFormat("dense B-tree record count exceeds u16".into()))?
                .to_le_bytes(),
        );
        let record_count = u64_from_usize_writer(records.len(), "dense B-tree record count")?;
        append_encoded_size(&mut header, record_count, self.sizeof_size)?;
        let checksum = checksum_metadata(&header);
        header.extend_from_slice(&checksum.to_le_bytes());
        let header_addr = self.allocator.allocate(
            u64_from_usize_writer(header.len(), "dense B-tree header size")?,
            8,
        );
        self.write_at(header_addr, &header)?;
        Ok(header_addr)
    }

    fn write_dense_btree_subtree(
        &mut self,
        tree_type: u8,
        records: &[Vec<u8>],
        record_size: usize,
        node_size: usize,
        level_info: &[DenseBtreeLevelInfo],
        depth: usize,
    ) -> Result<DenseBtreeChild> {
        if depth == 0 {
            let max_records = level_info
                .first()
                .ok_or_else(|| Error::InvalidFormat("dense B-tree level info is empty".into()))?
                .max_nrecords;
            if records.len() > max_records {
                return Err(Error::InvalidFormat(
                    "dense B-tree leaf receives too many records".into(),
                ));
            }
            let addr = self.write_dense_btree_leaf_node(tree_type, records, node_size)?;
            return Ok(DenseBtreeChild {
                addr,
                node_nrecords: records.len(),
                total_records: u64_from_usize_writer(records.len(), "dense B-tree leaf records")?,
            });
        }

        let level = level_info.get(depth).ok_or_else(|| {
            Error::InvalidFormat("dense B-tree subtree depth is out of range".into())
        })?;
        let child_level = level_info.get(depth - 1).ok_or_else(|| {
            Error::InvalidFormat("dense B-tree child depth is out of range".into())
        })?;
        let total_records = u64_from_usize_writer(records.len(), "dense B-tree subtree records")?;
        if total_records > level.cumulative_max_records {
            return Err(Error::InvalidFormat(
                "dense B-tree subtree receives too many records".into(),
            ));
        }

        let mut child_addrs = Vec::new();
        let mut child_nrecords = Vec::new();
        let mut child_total_records = Vec::new();
        let mut root_records: Vec<&[u8]> = Vec::new();
        let mut start = 0usize;
        while start < records.len() {
            let remaining = records.len() - start;
            let child_len =
                remaining.min(usize::try_from(child_level.cumulative_max_records).map_err(
                    |_| Error::InvalidFormat("dense B-tree child capacity exceeds usize".into()),
                )?);
            let child_end = start
                .checked_add(child_len)
                .ok_or_else(|| Error::InvalidFormat("dense B-tree child range overflow".into()))?;
            let child = self.write_dense_btree_subtree(
                tree_type,
                &records[start..child_end],
                record_size,
                node_size,
                level_info,
                depth - 1,
            )?;
            child_addrs.push(child.addr);
            child_nrecords.push(child.node_nrecords);
            child_total_records.push(child.total_records);
            start = child_end;
            if start < records.len() {
                root_records.push(records[start].as_slice());
                start = start.checked_add(1).ok_or_else(|| {
                    Error::InvalidFormat("dense B-tree separator range overflow".into())
                })?;
            }
        }

        let root_nrecords = root_records.len();
        if root_nrecords > level.max_nrecords {
            return Err(Error::Unsupported(
                "dense name B-tree node has too many separator records".into(),
            ));
        }
        let addr = self.write_dense_btree_internal_node(
            tree_type,
            record_size,
            level_info[0].max_nrecords,
            child_level.cumulative_max_record_size,
            node_size,
            &root_records,
            &child_addrs,
            &child_nrecords,
            &child_total_records,
        )?;
        Ok(DenseBtreeChild {
            addr,
            node_nrecords: root_nrecords,
            total_records,
        })
    }

    fn write_dense_btree_leaf_node(
        &mut self,
        tree_type: u8,
        records: &[Vec<u8>],
        node_size: usize,
    ) -> Result<u64> {
        let record_bytes = records.iter().try_fold(0usize, |acc, record| {
            acc.checked_add(record.len())
                .ok_or_else(|| Error::InvalidFormat("dense B-tree record bytes overflow".into()))
        })?;
        let leaf_len =
            checked_usize_sum_writer(&[6, record_bytes, 4], "dense B-tree leaf image length")?;
        if leaf_len > node_size {
            return Err(Error::InvalidFormat(
                "dense B-tree leaf image exceeds node size".into(),
            ));
        }
        let mut leaf = Vec::with_capacity(node_size);
        leaf.extend_from_slice(b"BTLF");
        leaf.push(0);
        leaf.push(tree_type);
        for record in records {
            leaf.extend_from_slice(record);
        }
        let leaf_checksum = checksum_metadata(&leaf);
        leaf.extend_from_slice(&leaf_checksum.to_le_bytes());
        let leaf_addr = self.allocator.allocate(
            u64_from_usize_writer(leaf.len(), "dense B-tree leaf size")?,
            8,
        );
        self.write_at(leaf_addr, &leaf)?;
        Ok(leaf_addr)
    }

    fn write_dense_btree_internal_node(
        &mut self,
        tree_type: u8,
        record_size: usize,
        leaf_max_records: usize,
        child_all_nrec_size: usize,
        node_size: usize,
        records: &[&[u8]],
        child_addrs: &[u64],
        child_nrecords: &[usize],
        child_total_records: &[u64],
    ) -> Result<u64> {
        if child_addrs.len() != records.len().saturating_add(1)
            || child_nrecords.len() != child_addrs.len()
            || child_total_records.len() != child_addrs.len()
        {
            return Err(Error::InvalidFormat(
                "dense B-tree internal child count is inconsistent".into(),
            ));
        }
        if records.iter().any(|record| record.len() != record_size) {
            return Err(Error::InvalidFormat(
                "dense B-tree internal records have inconsistent sizes".into(),
            ));
        }

        let max_nrec_size = dense_btree_bytes_needed(u64_from_usize_writer(
            leaf_max_records,
            "dense B-tree leaf capacity",
        )?);
        let pointer_size = checked_usize_sum_writer(
            &[
                usize::from(self.sizeof_addr),
                max_nrec_size,
                child_all_nrec_size,
            ],
            "dense B-tree pointer size",
        )?;
        let record_bytes = records.len().checked_mul(record_size).ok_or_else(|| {
            Error::InvalidFormat("dense B-tree internal record bytes overflow".into())
        })?;
        let child_bytes = child_addrs.len().checked_mul(pointer_size).ok_or_else(|| {
            Error::InvalidFormat("dense B-tree internal child bytes overflow".into())
        })?;
        let node_capacity = checked_usize_sum_writer(
            &[6, record_bytes, child_bytes, 4],
            "dense B-tree internal image length",
        )?;
        if node_capacity > node_size {
            return Err(Error::InvalidFormat(
                "dense B-tree internal image exceeds node size".into(),
            ));
        }

        let mut node = Vec::with_capacity(node_capacity);
        node.extend_from_slice(b"BTIN");
        node.push(0);
        node.push(tree_type);
        for record in records {
            node.extend_from_slice(record);
        }
        for ((&child_addr, &child_nrecords), &child_total_records) in child_addrs
            .iter()
            .zip(child_nrecords)
            .zip(child_total_records)
        {
            append_encoded_addr(&mut node, child_addr, self.sizeof_addr)?;
            append_dense_btree_var_uint(
                &mut node,
                u64_from_usize_writer(child_nrecords, "dense B-tree child record count")?,
                max_nrec_size,
            )?;
            if child_all_nrec_size > 0 {
                append_dense_btree_var_uint(&mut node, child_total_records, child_all_nrec_size)?;
            }
        }
        let checksum = checksum_metadata(&node);
        node.extend_from_slice(&checksum.to_le_bytes());
        let node_addr = self.allocator.allocate(
            u64_from_usize_writer(node.len(), "dense B-tree internal size")?,
            8,
        );
        self.write_at(node_addr, &node)?;
        Ok(node_addr)
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

            let has_group_attrs = self
                .pending_group_attr_specs
                .get(&path)
                .is_some_and(|attrs| !attrs.is_empty());

            let mut link_records = Vec::new();
            for (name, addr) in &group_links {
                link_records.push(EncodedLinkRecord {
                    name: name.clone(),
                    compact_message: encode_link_message(name, *addr, self.sizeof_addr)?,
                    dense_message: encode_dense_link_message(name, *addr, self.sizeof_addr)?,
                });
            }

            for (parent, name, link_data) in &self.special_links {
                if *parent == path {
                    link_records.push(EncodedLinkRecord {
                        name: name.clone(),
                        compact_message: link_data.clone(),
                        dense_message: link_data.clone(),
                    });
                }
            }

            // Resolve explicit hard-link aliases after child groups have been
            // finalized, because group object headers may have moved.
            for (parent, name, target_path) in &self.hard_links {
                if *parent == path {
                    let target_addr = self.object_addr_for_path(target_path).ok_or_else(|| {
                        Error::InvalidFormat(format!("hard link target '{target_path}' not found"))
                    })?;
                    link_records.push(EncodedLinkRecord {
                        name: name.clone(),
                        compact_message: encode_link_message(name, target_addr, self.sizeof_addr)?,
                        dense_message: encode_dense_link_message(
                            name,
                            target_addr,
                            self.sizeof_addr,
                        )?,
                    });
                }
            }

            if link_records.is_empty() && !has_group_attrs && path != "/" {
                continue;
            }

            // Build messages: link info + link messages. Groups above the
            // compact threshold use dense link storage, backed by a v2 B-tree
            // name index and heap IDs that point directly at link payloads.
            let mut messages: Vec<(u16, Vec<u8>)> = Vec::new();

            if link_records.len() > 8 {
                let dense_links: Vec<(String, Vec<u8>)> = link_records
                    .iter()
                    .map(|link| (link.name.clone(), link.dense_message.clone()))
                    .collect();
                let (heap_addr, btree_addr) = self.write_dense_link_storage(&dense_links)?;
                messages.push((MSG_GROUP_INFO, vec![0, 0]));
                let mut link_info = Vec::new();
                encode_link_info_message_into(
                    &mut link_info,
                    heap_addr,
                    btree_addr,
                    self.sizeof_addr,
                )?;
                messages.push((MSG_LINK_INFO, link_info));
            } else {
                messages.push((MSG_GROUP_INFO, vec![0, 0]));
                let mut link_info = Vec::new();
                encode_link_info_message_into(
                    &mut link_info,
                    UNDEF_ADDR,
                    UNDEF_ADDR,
                    self.sizeof_addr,
                )?;
                messages.push((MSG_LINK_INFO, link_info));

                for link in &link_records {
                    messages.push((MSG_LINK, link.compact_message.clone()));
                }
            }

            // Add pending root attributes. Attributes added through the typed
            // API can spill to dense storage; pre-encoded messages remain compact.
            if path == "/" {
                for (msg_type, attr_data) in &self.pending_root_attrs {
                    messages.push((*msg_type, attr_data.clone()));
                }

                let root_attr_specs: Vec<AttrSpec<'_>> = self
                    .pending_root_attr_specs
                    .iter()
                    .map(OwnedAttrSpec::as_attr_spec)
                    .collect();
                if attrs_need_dense_storage(&root_attr_specs)? {
                    let root_attrs = std::mem::take(&mut self.pending_root_attr_specs);
                    let result: Result<()> = (|| {
                        let attr_specs: Vec<AttrSpec<'_>> =
                            root_attrs.iter().map(OwnedAttrSpec::as_attr_spec).collect();
                        let (heap_addr, btree_addr) =
                            self.write_dense_attribute_storage(&attr_specs)?;
                        messages.push((MSG_ATTR_INFO, {
                            let mut attr_info = Vec::new();
                            encode_attr_info_message_into(
                                &mut attr_info,
                                heap_addr,
                                btree_addr,
                                self.sizeof_addr,
                            )?;
                            attr_info
                        }));
                        Ok(())
                    })();
                    self.pending_root_attr_specs = root_attrs;
                    result?;
                } else {
                    for attr in &self.pending_root_attr_specs {
                        let mut attr_bytes = Vec::new();
                        encode_attribute_message_into(
                            &mut attr_bytes,
                            &attr.name,
                            &attr.dtype,
                            &attr.shape,
                            &attr.data,
                        )?;
                        messages.push((MSG_ATTRIBUTE, attr_bytes));
                    }
                }
            }

            if let Some(group_attrs) = self.pending_group_attr_specs.remove(&path) {
                let result: Result<()> = (|| {
                    let attr_specs: Vec<AttrSpec<'_>> = group_attrs
                        .iter()
                        .map(OwnedAttrSpec::as_attr_spec)
                        .collect();
                    if attrs_need_dense_storage(&attr_specs)? {
                        let (heap_addr, btree_addr) =
                            self.write_dense_attribute_storage(&attr_specs)?;
                        messages.push((MSG_ATTR_INFO, {
                            let mut attr_info = Vec::new();
                            encode_attr_info_message_into(
                                &mut attr_info,
                                heap_addr,
                                btree_addr,
                                self.sizeof_addr,
                            )?;
                            attr_info
                        }));
                    } else {
                        for attr in &group_attrs {
                            let mut attr_bytes = Vec::new();
                            encode_attribute_message_into(
                                &mut attr_bytes,
                                &attr.name,
                                &attr.dtype,
                                &attr.shape,
                                &attr.data,
                            )?;
                            messages.push((MSG_ATTRIBUTE, attr_bytes));
                        }
                    }
                    Ok(())
                })();
                self.pending_group_attr_specs
                    .insert(path.clone(), group_attrs);
                result?;
            }

            let msg_refs: Vec<(u16, &[u8])> =
                messages.iter().map(|(t, d)| (*t, d.as_slice())).collect();

            let oh_addr = self.write_v2_object_header(&msg_refs, 0)?;

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

    /// Write `data` at the given file offset (mirrors `H5FD_write`).
    fn write_at(&mut self, offset: u64, data: &[u8]) -> Result<()> {
        self.writer.seek(SeekFrom::Start(offset))?;
        self.writer.write_all(data)?;
        Ok(())
    }

    /// Look up the object header address for a normalized group path.
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

/// Strip duplicate slashes and trailing slashes from an HDF5 object path.
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

/// Compose a child object path from a parent path and child name.
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
        ceil_div_nonzero_u64, checked_next_power_of_two, checked_usize_sum_writer,
        dense_record_hash, encode_global_heap_collection, read_encoded_size_le_at, read_u64_le_at,
        CompoundFieldSpec, DtypeSpec, FillValueSpec, HdfFileWriter,
    };
    use crate::format::btree_v2::{collect_all_records_into, BTreeV2Header};
    use crate::io::reader::HdfReader;
    use crate::io::reader::UNDEF_ADDR;
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
        assert!(checked_usize_sum_writer(&[usize::MAX, 1], "test sum").is_err());
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
    fn dense_name_btree_writer_round_trips_depth2_tree() {
        let mut records = Vec::new();
        for idx in 0..4096u32 {
            let mut record = Vec::new();
            record.extend_from_slice(&idx.to_le_bytes());
            record.push((idx & 0xff) as u8);
            records.push(record);
        }

        let mut writer = HdfFileWriter::new(Cursor::new(Vec::new()));
        let header_addr = writer.write_dense_name_btree(5, &records).unwrap();
        let bytes = writer.writer.get_ref().clone();
        let mut reader = HdfReader::new(Cursor::new(bytes));
        let header = BTreeV2Header::read_at(&mut reader, header_addr).unwrap();
        assert_eq!(header.depth, 2);
        assert_eq!(header.total_records, 4096);

        let mut decoded = Vec::new();
        collect_all_records_into(&mut reader, header_addr, &mut decoded).unwrap();
        assert_eq!(decoded, records);
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
    fn global_heap_collection_uses_configured_size_width() {
        let heap =
            encode_global_heap_collection(&[b"alpha".to_vec(), vec![1, 2, 3], Vec::new()], 2)
                .unwrap();
        let collection_size =
            read_encoded_size_le_at(&heap, 8, 2, "test global heap collection size")
                .expect("collection size field should decode");
        let first_object_size =
            read_encoded_size_le_at(&heap, 18, 2, "test global heap first object size")
                .expect("object size field should decode");
        let free_size = read_encoded_size_le_at(&heap, 64, 2, "test global heap free size")
            .expect("free size field should decode");

        assert_eq!(collection_size, 4096);
        assert_eq!(first_object_size, 5);
        assert_eq!(free_size, 4096 - 56);
        assert_eq!(&heap[20..25], b"alpha");
    }

    #[test]
    fn global_heap_collection_rejects_too_narrow_size_width() {
        let err = encode_global_heap_collection(&[vec![0; 256]], 1).unwrap_err();
        assert!(
            err.to_string().contains("does not fit in 1 bytes"),
            "unexpected error: {err}"
        );
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
        assert!(DtypeSpec::FixedAsciiString { len: 0, padding: 1 }
            .encode()
            .is_err());
        assert!(DtypeSpec::FixedUtf8String { len: 8, padding: 3 }
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
        assert!(super::append_encoded_addr(&mut Vec::new(), u64::from(u32::MAX) + 1, 4).is_err());
        let mut undefined_addr = Vec::new();
        super::append_encoded_addr(&mut undefined_addr, UNDEF_ADDR, 4).unwrap();
        assert_eq!(undefined_addr, vec![0xff; 4]);
        assert!(super::append_encoded_size(&mut Vec::new(), u64::from(u32::MAX) + 1, 4).is_err());
        assert!(super::encode_link_name_len(&mut Vec::new(), 256, 0).is_err());
        assert!(super::encode_link_name_len(&mut Vec::new(), 65_536, 1).is_err());
        if usize::BITS > u32::BITS {
            let eight_byte_len = usize::try_from(u64::from(u32::MAX) + 1)
                .expect("test value should fit on this target");
            assert_eq!(super::link_name_size_flag(eight_byte_len).unwrap(), 3);
            assert!(super::encode_link_name_len(&mut Vec::new(), eight_byte_len, 2).is_err());
            let mut encoded = Vec::new();
            super::encode_link_name_len(&mut encoded, eight_byte_len, 3).unwrap();
            assert_eq!(encoded, (u64::from(u32::MAX) + 1).to_le_bytes());
        }
        assert!(super::encode_link_name_len(&mut Vec::new(), 1, 4).is_err());
        assert!(super::encode_contiguous_layout_into(&mut Vec::new(), 0, 0, 8, 0).is_err());
        assert!(super::encode_contiguous_layout_into(&mut Vec::new(), 0, 0, 8, 9).is_err());
        assert!(super::encode_chunked_layout_v3_into(
            &mut Vec::new(),
            0,
            &[u64::from(u32::MAX) + 1],
            1,
            8
        )
        .is_err());
        assert!(super::encode_chunked_layout_v3_into(&mut Vec::new(), 0, &[0], 1, 8).is_err());
        assert!(super::encode_fill_value_message_into(
            &mut Vec::new(),
            Some(FillValueSpec {
                alloc_time: 4,
                fill_time: 0,
                value: None,
            })
        )
        .is_err());
    }

    #[test]
    fn v4_chunk_layout_encoders_include_trailing_element_size_dimension() {
        let mut single = Vec::new();
        super::encode_single_chunk_layout_v4_into(
            &mut single,
            0x1122_3344_5566_7788,
            &[25],
            4,
            Some(64),
            0,
            8,
            8,
        )
        .unwrap();
        let decoded =
            crate::format::messages::data_layout::DataLayoutMessage::decode(&single, 8, 8).unwrap();
        assert_eq!(decoded.chunk_encoded_dims, Some(vec![25, 4]));
        assert_eq!(decoded.chunk_dims, Some(vec![25, 4]));
        assert_eq!(decoded.single_chunk_filtered_size, Some(64));

        let mut fixed = Vec::new();
        super::encode_fixed_array_chunk_layout_v4_into(
            &mut fixed,
            0x8877_6655_4433_2211,
            &[25],
            4,
            6,
            8,
        )
        .unwrap();
        let decoded =
            crate::format::messages::data_layout::DataLayoutMessage::decode(&fixed, 8, 8).unwrap();
        assert_eq!(decoded.chunk_encoded_dims, Some(vec![25, 4]));
        assert_eq!(decoded.chunk_dims, Some(vec![25, 4]));
        assert_eq!(
            decoded.chunk_index_type,
            Some(crate::format::messages::data_layout::ChunkIndexType::FixedArray)
        );

        let mut extensible = Vec::new();
        super::encode_extensible_array_chunk_layout_v4_into(
            &mut extensible,
            0x0102_0304_0506_0708,
            &[10],
            4,
            4,
            10,
            1,
            1,
            1,
            8,
        )
        .unwrap();
        let decoded =
            crate::format::messages::data_layout::DataLayoutMessage::decode(&extensible, 8, 8)
                .unwrap();
        assert_eq!(decoded.chunk_encoded_dims, Some(vec![10, 4]));
        assert_eq!(
            decoded.chunk_index_type,
            Some(crate::format::messages::data_layout::ChunkIndexType::ExtensibleArray)
        );
    }

    #[test]
    fn link_name_length_encoder_uses_hdf5_size_flags() {
        fn encoded_len(name_len: usize) -> Vec<u8> {
            let size_flag = super::link_name_size_flag(name_len).unwrap();
            let mut encoded = Vec::new();
            super::encode_link_name_len(&mut encoded, name_len, size_flag).unwrap();
            encoded
        }

        assert_eq!(super::link_name_size_flag(255).unwrap(), 0);
        assert_eq!(encoded_len(255), vec![0xff]);

        assert_eq!(super::link_name_size_flag(256).unwrap(), 1);
        assert_eq!(encoded_len(256), 256u16.to_le_bytes());
        assert_eq!(super::link_name_size_flag(65_535).unwrap(), 1);

        assert_eq!(super::link_name_size_flag(65_536).unwrap(), 2);
        assert_eq!(encoded_len(65_536), 65_536u32.to_le_bytes());
        assert_eq!(super::link_name_size_flag(u32::MAX as usize).unwrap(), 2);

        if usize::BITS > u32::BITS {
            let eight_byte_len = usize::try_from(u64::from(u32::MAX) + 1)
                .expect("test value should fit on this target");
            assert_eq!(super::link_name_size_flag(eight_byte_len).unwrap(), 3);
            assert_eq!(encoded_len(eight_byte_len), eight_byte_len.to_le_bytes());
        }
    }
}
