//! Fixed array header — mirrors libhdf5's `H5FAhdr.c` + the header-half
//! of `H5FAcache.c` (`H5FA__cache_hdr_deserialize`).

#![allow(dead_code)]

use std::{
    fmt,
    io::{Read, Seek},
};

use crate::error::{Error, Result};
use crate::format::checksum::checksum_metadata;
use crate::io::reader::{HdfReader, UNDEF_ADDR};

const MAX_FIXED_ARRAY_ELEMENTS: usize = 1_000_000;

#[derive(Debug, Clone)]
pub(super) struct FixedArrayHeader {
    pub(super) class_id: u8,
    pub(super) raw_element_size: usize,
    pub(super) max_page_elements_bits: u8,
    pub(super) elements: u64,
    pub(super) data_block_addr: u64,
}

/// Format a fixed array header for debug printing.
pub(super) fn write_hdr_debug(header: &FixedArrayHeader, out: &mut impl fmt::Write) -> fmt::Result {
    write!(
        out,
        "FixedArrayHeader(class_id={}, raw_element_size={}, max_page_elements_bits={}, elements={}, data_block_addr={:#x})",
        header.class_id,
        header.raw_element_size,
        header.max_page_elements_bits,
        header.elements,
        header.data_block_addr
    )
}

/// Initial number of bytes the metadata cache must read to determine the
/// full fixed array header size (magic + version byte).
pub(super) fn cache_hdr_get_initial_load_size() -> usize {
    4 + 1
}

/// Compute the on-disk size of a fixed array header for the given widths.
pub(super) fn cache_hdr_image_len(addr_size: usize, length_size: usize) -> Result<usize> {
    4usize
        .checked_add(1)
        .and_then(|value| value.checked_add(1))
        .and_then(|value| value.checked_add(1))
        .and_then(|value| value.checked_add(1))
        .and_then(|value| value.checked_add(length_size))
        .and_then(|value| value.checked_add(addr_size))
        .and_then(|value| value.checked_add(4))
        .ok_or_else(|| Error::InvalidFormat("fixed array header image length overflow".into()))
}

/// Serialize a dirty fixed array header to its on-disk image.
pub(super) fn cache_hdr_serialize_into(
    header: &FixedArrayHeader,
    addr_size: usize,
    length_size: usize,
    out: &mut Vec<u8>,
) -> Result<()> {
    out.clear();
    out.reserve(cache_hdr_image_len(addr_size, length_size)?);
    out.extend_from_slice(b"FAHD");
    out.push(0);
    out.push(header.class_id);
    out.push(u8::try_from(header.raw_element_size).map_err(|_| {
        Error::InvalidFormat("fixed array raw element size does not fit in u8".into())
    })?);
    out.push(header.max_page_elements_bits);
    encode_var(out, header.elements, length_size)?;
    encode_addr(out, header.data_block_addr, addr_size)?;
    let checksum = checksum_metadata(&out);
    out.extend_from_slice(&checksum.to_le_bytes());
    Ok(())
}

/// Handle metadata-cache action notifications for the header.
pub(super) fn cache_hdr_notify(_header: &FixedArrayHeader) {}

/// Destroy/release an in-core representation of a fixed array header.
pub(super) fn cache_hdr_free_icr(_header: FixedArrayHeader) {}

/// Allocate a shared fixed array header with an undefined data-block address.
pub(super) fn hdr_alloc(
    class_id: u8,
    raw_element_size: usize,
    max_page_elements_bits: u8,
    elements: u64,
) -> FixedArrayHeader {
    FixedArrayHeader {
        class_id,
        raw_element_size,
        max_page_elements_bits,
        elements,
        data_block_addr: crate::io::reader::UNDEF_ADDR,
    }
}

/// Reset the dynamic fields on a fixed array header.
pub(super) fn hdr_init(header: &mut FixedArrayHeader) {
    header.elements = 0;
    header.data_block_addr = crate::io::reader::UNDEF_ADDR;
}

/// Create a new fixed array header (returns the provided header value).
pub(super) fn hdr_create(header: FixedArrayHeader) -> FixedArrayHeader {
    header
}

/// Increment the component reference count on the shared header (saturates).
pub(super) fn hdr_incr(ref_count: &mut usize) {
    let _ = hdr_incr_checked(ref_count);
}

/// Checked variant of `hdr_incr` that returns an error on overflow.
pub(super) fn hdr_incr_checked(ref_count: &mut usize) -> Result<()> {
    *ref_count = ref_count
        .checked_add(1)
        .ok_or_else(|| Error::InvalidFormat("fixed array header reference overflow".into()))?;
    Ok(())
}

/// Decrement the component reference count on the shared header.
pub(super) fn hdr_decr(ref_count: &mut usize) -> Result<()> {
    if *ref_count == 0 {
        return Err(Error::InvalidFormat(
            "fixed array header reference underflow".into(),
        ));
    }
    *ref_count -= 1;
    Ok(())
}

/// Increment the file reference count on the shared header.
pub(super) fn hdr_fuse_incr(ref_count: &mut usize) {
    hdr_incr(ref_count);
}

/// Checked variant of `hdr_fuse_incr`.
pub(super) fn hdr_fuse_incr_checked(ref_count: &mut usize) -> Result<()> {
    hdr_incr_checked(ref_count)
}

/// Decrement the file reference count on the shared header.
pub(super) fn hdr_fuse_decr(ref_count: &mut usize) -> Result<()> {
    hdr_decr(ref_count)
}

/// Mark the fixed array as modified so it will be flushed.
pub(super) fn hdr_modified(_header: &mut FixedArrayHeader) {}

/// Protect the fixed array header (loads it from disk).
pub(super) fn hdr_protect<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    addr: u64,
) -> Result<FixedArrayHeader> {
    read_header(reader, addr)
}

/// Release a protected fixed array header.
pub(super) fn hdr_unprotect(_header: FixedArrayHeader) {}

/// Mark the fixed array deleted by clearing the data block address and element count.
pub(super) fn hdr_delete(header: &mut FixedArrayHeader) {
    header.elements = 0;
    header.data_block_addr = crate::io::reader::UNDEF_ADDR;
}

/// Destroy the fixed array header in memory.
pub(super) fn hdr_dest(_header: FixedArrayHeader) {}

/// Load a fixed array header from disk: validates magic, version, fixed fields, and checksum.
pub(super) fn read_header<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    addr: u64,
) -> Result<FixedArrayHeader> {
    reader.seek(addr)?;
    let mut magic = [0u8; 4];
    reader.read_bytes_into(&mut magic)?;
    if magic != *b"FAHD" {
        return Err(Error::InvalidFormat(
            "invalid fixed array header magic".into(),
        ));
    }

    let version = reader.read_u8()?;
    if version != 0 {
        return Err(Error::Unsupported(format!(
            "fixed array header version {version}"
        )));
    }

    let class_id = reader.read_u8()?;
    let raw_element_size = usize::from(reader.read_u8()?);
    if raw_element_size == 0 {
        return Err(Error::InvalidFormat(
            "fixed array element size must be nonzero".into(),
        ));
    }
    let max_page_elements_bits = reader.read_u8()?;
    let elements = reader.read_length()?;
    let element_count = super::usize_from_u64(elements, "fixed array element count")?;
    if element_count > MAX_FIXED_ARRAY_ELEMENTS {
        return Err(Error::InvalidFormat(format!(
            "fixed array element count {element_count} exceeds supported maximum {MAX_FIXED_ARRAY_ELEMENTS}"
        )));
    }
    let data_block_addr = reader.read_addr()?;
    verify_checksum(reader, addr, "fixed array header")?;

    Ok(FixedArrayHeader {
        class_id,
        raw_element_size,
        max_page_elements_bits,
        elements,
        data_block_addr,
    })
}

/// Encode an unsigned integer as `size` little-endian bytes, validating range.
fn encode_var(out: &mut Vec<u8>, value: u64, size: usize) -> Result<()> {
    if size == 0 || size > 8 {
        return Err(Error::InvalidFormat(
            "fixed array encoded integer size is invalid".into(),
        ));
    }
    if size < 8 && value >= (1u64 << (size * 8)) {
        return Err(Error::InvalidFormat(format!(
            "fixed array encoded integer value {value:#x} does not fit in {size} bytes"
        )));
    }
    let bytes = value.to_le_bytes();
    out.extend_from_slice(&bytes[..size]);
    Ok(())
}

/// Encode an address as `size` little-endian bytes, writing the all-`0xff`
/// sentinel for undefined addresses.
fn encode_addr(out: &mut Vec<u8>, value: u64, size: usize) -> Result<()> {
    if size == 0 || size > 8 {
        return Err(Error::InvalidFormat(
            "fixed array encoded address size is invalid".into(),
        ));
    }
    if value == UNDEF_ADDR {
        out.extend(std::iter::repeat_n(0xff, size));
        return Ok(());
    }
    encode_var(out, value, size)
}

/// Verify the trailing metadata checksum of a fixed array image span.
pub(super) fn verify_checksum<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    start: u64,
    context: &str,
) -> Result<()> {
    let checksum_pos = reader.position()?;
    let stored_checksum = reader.read_u32()?;
    let check_len = usize::try_from(checksum_pos - start)
        .map_err(|_| Error::InvalidFormat(format!("{context} checksum span is too large")))?;
    reader.seek(start)?;
    let mut check_data = vec![0; check_len];
    reader.read_bytes_into(&mut check_data)?;
    let computed = checksum_metadata(&check_data);
    if stored_checksum != computed {
        return Err(Error::InvalidFormat(format!(
            "{context} checksum mismatch: stored={stored_checksum:#010x}, computed={computed:#010x}"
        )));
    }
    reader.seek(checksum_pos.checked_add(4).ok_or_else(|| {
        Error::InvalidFormat(format!("{context} checksum end offset overflow"))
    })?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use crate::io::HdfReader;

    use super::{
        cache_hdr_serialize_into, hdr_fuse_incr_checked, hdr_incr_checked, read_header,
        FixedArrayHeader,
    };

    #[test]
    fn fixed_array_header_rejects_zero_element_size() {
        let mut bytes = b"FAHD".to_vec();
        bytes.push(0);
        bytes.push(1);
        bytes.push(0);
        bytes.push(4);
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&crate::io::reader::UNDEF_ADDR.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());

        let mut reader = HdfReader::new(Cursor::new(bytes));
        let err = read_header(&mut reader, 0).expect_err("zero element size should fail");
        assert!(
            err.to_string().contains("element size"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn fixed_array_header_cache_serialize_checks_configured_widths() {
        let header = FixedArrayHeader {
            class_id: 1,
            raw_element_size: 4,
            max_page_elements_bits: 0,
            elements: 7,
            data_block_addr: crate::io::reader::UNDEF_ADDR,
        };
        let mut image = Vec::new();
        cache_hdr_serialize_into(&header, 4, 4, &mut image).unwrap();
        assert_eq!(&image[12..16], &[0xff; 4]);

        let too_large_elements = FixedArrayHeader {
            elements: u64::from(u32::MAX) + 1,
            ..header.clone()
        };
        assert!(cache_hdr_serialize_into(&too_large_elements, 4, 4, &mut Vec::new()).is_err());

        let too_large_addr = FixedArrayHeader {
            data_block_addr: u64::from(u32::MAX) + 1,
            ..header
        };
        assert!(cache_hdr_serialize_into(&too_large_addr, 4, 4, &mut Vec::new()).is_err());
    }

    #[test]
    fn fixed_array_header_refcount_checked_rejects_overflow() {
        let mut ref_count = usize::MAX;
        assert!(hdr_incr_checked(&mut ref_count).is_err());
        assert!(hdr_fuse_incr_checked(&mut ref_count).is_err());
    }
}
