//! Fixed array header — mirrors libhdf5's `H5FAhdr.c` + the header-half
//! of `H5FAcache.c` (`H5FA__cache_hdr_deserialize`).

#![allow(dead_code)]

use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::format::checksum::checksum_metadata;
use crate::io::reader::HdfReader;

const MAX_FIXED_ARRAY_ELEMENTS: usize = 1_000_000;

#[derive(Debug, Clone)]
pub(super) struct FixedArrayHeader {
    pub(super) class_id: u8,
    pub(super) raw_element_size: usize,
    pub(super) max_page_elements_bits: u8,
    pub(super) elements: u64,
    pub(super) data_block_addr: u64,
}

pub(super) fn hdr_debug(header: &FixedArrayHeader) -> String {
    format!(
        "FixedArrayHeader(class_id={}, raw_element_size={}, max_page_elements_bits={}, elements={}, data_block_addr={:#x})",
        header.class_id,
        header.raw_element_size,
        header.max_page_elements_bits,
        header.elements,
        header.data_block_addr
    )
}

pub(super) fn cache_hdr_get_initial_load_size() -> usize {
    4 + 1
}

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

pub(super) fn cache_hdr_serialize(
    header: &FixedArrayHeader,
    addr_size: usize,
    length_size: usize,
) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(cache_hdr_image_len(addr_size, length_size)?);
    out.extend_from_slice(b"FAHD");
    out.push(0);
    out.push(header.class_id);
    out.push(u8::try_from(header.raw_element_size).map_err(|_| {
        Error::InvalidFormat("fixed array raw element size does not fit in u8".into())
    })?);
    out.push(header.max_page_elements_bits);
    encode_var(&mut out, header.elements, length_size)?;
    encode_var(&mut out, header.data_block_addr, addr_size)?;
    let checksum = checksum_metadata(&out);
    out.extend_from_slice(&checksum.to_le_bytes());
    Ok(out)
}

pub(super) fn cache_hdr_notify(_header: &FixedArrayHeader) {}

pub(super) fn cache_hdr_free_icr(_header: FixedArrayHeader) {}

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

pub(super) fn hdr_init(header: &mut FixedArrayHeader) {
    header.elements = 0;
    header.data_block_addr = crate::io::reader::UNDEF_ADDR;
}

pub(super) fn hdr_create(header: FixedArrayHeader) -> FixedArrayHeader {
    header
}

pub(super) fn hdr_incr(ref_count: &mut usize) {
    *ref_count = ref_count.saturating_add(1);
}

pub(super) fn hdr_decr(ref_count: &mut usize) -> Result<()> {
    if *ref_count == 0 {
        return Err(Error::InvalidFormat(
            "fixed array header reference underflow".into(),
        ));
    }
    *ref_count -= 1;
    Ok(())
}

pub(super) fn hdr_fuse_incr(ref_count: &mut usize) {
    hdr_incr(ref_count);
}

pub(super) fn hdr_fuse_decr(ref_count: &mut usize) -> Result<()> {
    hdr_decr(ref_count)
}

pub(super) fn hdr_modified(_header: &mut FixedArrayHeader) {}

pub(super) fn hdr_protect<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    addr: u64,
) -> Result<FixedArrayHeader> {
    read_header(reader, addr)
}

pub(super) fn hdr_unprotect(_header: FixedArrayHeader) {}

pub(super) fn hdr_delete(header: &mut FixedArrayHeader) {
    header.elements = 0;
    header.data_block_addr = crate::io::reader::UNDEF_ADDR;
}

pub(super) fn hdr_dest(_header: FixedArrayHeader) {}

pub(super) fn read_header<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    addr: u64,
) -> Result<FixedArrayHeader> {
    reader.seek(addr)?;
    let magic = reader.read_bytes(4)?;
    if magic != b"FAHD" {
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

fn encode_var(out: &mut Vec<u8>, value: u64, size: usize) -> Result<()> {
    if size > 8 {
        return Err(Error::InvalidFormat(
            "fixed array encoded integer size exceeds u64".into(),
        ));
    }
    let bytes = value.to_le_bytes();
    out.extend_from_slice(&bytes[..size]);
    Ok(())
}

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
    let check_data = reader.read_bytes(check_len)?;
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

    use super::read_header;

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
}
