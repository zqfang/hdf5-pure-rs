//! Extensible array header — mirrors libhdf5's `H5EAhdr.c` plus the
//! header-half of `H5EAcache.c` (`H5EA__cache_hdr_deserialize`).

#![allow(dead_code)]

use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::format::checksum::checksum_metadata;
use crate::io::reader::HdfReader;

const MAX_EXTENSIBLE_ARRAY_ELEMENTS: usize = 1_000_000;

#[derive(Debug, Clone)]
pub(crate) struct ParsedExtensibleArrayHeader {
    pub(crate) class_id: u8,
    pub(crate) raw_element_size: usize,
    pub(crate) index_block_elements: u8,
    pub(crate) data_block_min_elements: usize,
    pub(crate) super_block_count: u64,
    pub(crate) super_block_size: u64,
    pub(crate) data_block_count: u64,
    pub(crate) data_block_size: u64,
    pub(crate) max_index_set: u64,
    pub(crate) realized_elements: u64,
    pub(crate) index_block_addr: u64,
    pub(crate) array_offset_size: u8,
    pub(crate) data_block_page_elements: usize,
    pub(crate) index_block_super_blocks: usize,
    pub(crate) index_block_data_block_addrs: usize,
    pub(crate) index_block_super_block_addrs: usize,
    pub(crate) derived_super_block_count: usize,
    pub(crate) super_block_info: Vec<SuperBlockInfo>,
    pub(crate) checksum_pos: u64,
    pub(crate) super_block_count_pos: u64,
    pub(crate) super_block_size_pos: u64,
    pub(crate) data_block_count_pos: u64,
    pub(crate) data_block_size_pos: u64,
    pub(crate) max_index_set_pos: u64,
    pub(crate) realized_elements_pos: u64,
}

#[derive(Debug, Clone)]
pub(super) struct ExtensibleArrayHeader {
    pub(super) class_id: u8,
    pub(super) raw_element_size: usize,
    pub(super) index_block_elements: u8,
    pub(super) max_index_set: u64,
    pub(super) index_block_addr: u64,
    pub(super) array_offset_size: u8,
    pub(super) data_block_page_elements: usize,
    pub(super) index_block_super_blocks: usize,
    pub(super) index_block_data_block_addrs: usize,
    pub(super) index_block_super_block_addrs: usize,
    pub(super) super_block_info: Vec<SuperBlockInfo>,
}

#[derive(Debug, Clone)]
pub(crate) struct SuperBlockInfo {
    pub(super) data_blocks: usize,
    pub(super) data_block_elements: usize,
    pub(super) start_data_block: u64,
}

pub(super) fn read_header<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    addr: u64,
) -> Result<ExtensibleArrayHeader> {
    let parsed = read_header_core(reader, addr)?;

    Ok(ExtensibleArrayHeader {
        class_id: parsed.class_id,
        raw_element_size: parsed.raw_element_size,
        index_block_elements: parsed.index_block_elements,
        max_index_set: parsed.max_index_set,
        index_block_addr: parsed.index_block_addr,
        array_offset_size: parsed.array_offset_size,
        data_block_page_elements: parsed.data_block_page_elements,
        index_block_super_blocks: parsed.index_block_super_blocks,
        index_block_data_block_addrs: parsed.index_block_data_block_addrs,
        index_block_super_block_addrs: parsed.index_block_super_block_addrs,
        super_block_info: parsed.super_block_info,
    })
}

pub(super) fn hdr_debug(header: &ExtensibleArrayHeader) -> String {
    format!(
        "ExtensibleArrayHeader(class_id={}, raw_element_size={}, index_block_elements={}, max_index_set={}, index_block_addr={:#x})",
        header.class_id,
        header.raw_element_size,
        header.index_block_elements,
        header.max_index_set,
        header.index_block_addr
    )
}

pub(super) fn cache_hdr_get_initial_load_size() -> usize {
    4 + 1
}

pub(super) fn cache_hdr_image_len(addr_size: usize, length_size: usize) -> Result<usize> {
    let fixed = 4usize
        .checked_add(8)
        .and_then(|value| value.checked_add(6usize.checked_mul(length_size)?))
        .and_then(|value| value.checked_add(addr_size))
        .and_then(|value| value.checked_add(4))
        .ok_or_else(|| {
            Error::InvalidFormat("extensible array header image length overflow".into())
        })?;
    Ok(fixed)
}

pub(super) fn cache_hdr_serialize(
    header: &ParsedExtensibleArrayHeader,
    addr_size: usize,
    length_size: usize,
) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(cache_hdr_image_len(addr_size, length_size)?);
    out.extend_from_slice(b"EAHD");
    out.push(0);
    out.push(header.class_id);
    out.push(u8::try_from(header.raw_element_size).map_err(|_| {
        Error::InvalidFormat("extensible array raw element size does not fit in u8".into())
    })?);
    let max_bits = 64 - header.max_index_set.max(1).leading_zeros() as u8;
    out.push(max_bits);
    out.push(header.index_block_elements);
    out.push(u8::try_from(header.data_block_min_elements).map_err(|_| {
        Error::InvalidFormat("extensible array minimum data block elements do not fit in u8".into())
    })?);
    out.push(1);
    out.push(
        header
            .data_block_page_elements
            .checked_ilog2()
            .unwrap_or(0)
            .try_into()
            .map_err(|_| Error::InvalidFormat("extensible array page bits overflow".into()))?,
    );
    encode_var(&mut out, header.super_block_count, length_size)?;
    encode_var(&mut out, header.super_block_size, length_size)?;
    encode_var(&mut out, header.data_block_count, length_size)?;
    encode_var(&mut out, header.data_block_size, length_size)?;
    encode_var(&mut out, header.max_index_set, length_size)?;
    encode_var(&mut out, header.realized_elements, length_size)?;
    encode_var(&mut out, header.index_block_addr, addr_size)?;
    let checksum = checksum_metadata(&out);
    out.extend_from_slice(&checksum.to_le_bytes());
    Ok(out)
}

pub(super) fn cache_hdr_notify(_header: &ExtensibleArrayHeader) {}

pub(super) fn cache_hdr_free_icr(_header: ExtensibleArrayHeader) {}

pub(super) fn hdr_alloc(parsed: ParsedExtensibleArrayHeader) -> ParsedExtensibleArrayHeader {
    parsed
}

pub(super) fn hdr_init(header: &mut ParsedExtensibleArrayHeader) {
    header.realized_elements = 0;
    header.max_index_set = 0;
}

pub(super) fn hdr_alloc_elmts(count: usize) -> Vec<crate::format::fixed_array::FixedArrayElement> {
    vec![
        crate::format::fixed_array::FixedArrayElement {
            addr: crate::io::reader::UNDEF_ADDR,
            nbytes: None,
            filter_mask: 0,
        };
        count
    ]
}

pub(super) fn hdr_free_elmts(elements: &mut Vec<crate::format::fixed_array::FixedArrayElement>) {
    elements.clear();
}

pub(super) fn hdr_create(parsed: ParsedExtensibleArrayHeader) -> ParsedExtensibleArrayHeader {
    parsed
}

pub(super) fn hdr_incr(ref_count: &mut usize) {
    *ref_count = ref_count.saturating_add(1);
}

pub(super) fn hdr_decr(ref_count: &mut usize) -> Result<()> {
    if *ref_count == 0 {
        return Err(Error::InvalidFormat(
            "extensible array header reference underflow".into(),
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

pub(super) fn hdr_modified(_header: &mut ParsedExtensibleArrayHeader) {}

pub(super) fn hdr_protect<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    addr: u64,
) -> Result<ParsedExtensibleArrayHeader> {
    read_header_core(reader, addr)
}

pub(super) fn hdr_unprotect(_header: ParsedExtensibleArrayHeader) {}

pub(super) fn hdr_delete(header: &mut ParsedExtensibleArrayHeader) {
    header.index_block_addr = crate::io::reader::UNDEF_ADDR;
    header.max_index_set = 0;
    header.realized_elements = 0;
}

pub(super) fn hdr_dest(_header: ParsedExtensibleArrayHeader) {}

pub(crate) fn read_header_core<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    addr: u64,
) -> Result<ParsedExtensibleArrayHeader> {
    reader.seek(addr)?;
    let magic = reader.read_bytes(4)?;
    if magic != b"EAHD" {
        return Err(Error::InvalidFormat(
            "invalid extensible array header magic".into(),
        ));
    }

    let version = reader.read_u8()?;
    if version != 0 {
        return Err(Error::Unsupported(format!(
            "extensible array header version {version}"
        )));
    }

    let class_id = reader.read_u8()?;
    let raw_element_size = usize::from(reader.read_u8()?);
    if raw_element_size == 0 {
        return Err(Error::InvalidFormat(
            "extensible array element size must be nonzero".into(),
        ));
    }
    let max_elements_bits = reader.read_u8()?;
    let index_block_elements = reader.read_u8()?;
    let data_block_min_elements = reader.read_u8()?;
    let super_block_min_data_ptrs = reader.read_u8()?;
    let max_data_block_page_elements_bits = reader.read_u8()?;

    let super_block_count_pos = reader.position()?;
    let stored_super_block_count = reader.read_length()?;
    let super_block_size_pos = reader.position()?;
    let super_block_size = reader.read_length()?;
    let data_block_count_pos = reader.position()?;
    let data_block_count = reader.read_length()?;
    let data_block_size_pos = reader.position()?;
    let data_block_size = reader.read_length()?;
    let max_index_set_pos = reader.position()?;
    let max_index_set = reader.read_length()?;
    let max_index_count = super::usize_from_u64(max_index_set, "extensible array max index")?;
    if max_index_count > MAX_EXTENSIBLE_ARRAY_ELEMENTS {
        return Err(Error::InvalidFormat(format!(
            "extensible array max index {max_index_count} exceeds supported maximum {MAX_EXTENSIBLE_ARRAY_ELEMENTS}"
        )));
    }
    let realized_elements_pos = reader.position()?;
    let realized_elements = reader.read_length()?;
    let index_block_addr = reader.read_addr()?;
    let checksum_pos = reader.position()?;
    verify_checksum(reader, addr, "extensible array header")?;

    if index_block_elements == 0
        || data_block_min_elements == 0
        || !data_block_min_elements.is_power_of_two()
        || !super_block_min_data_ptrs.is_power_of_two()
    {
        return Err(Error::InvalidFormat(
            "invalid extensible array block parameters".into(),
        ));
    }

    let array_offset_size = max_elements_bits.div_ceil(8);
    let data_block_page_elements = 1usize
        .checked_shl(u32::from(max_data_block_page_elements_bits))
        .ok_or_else(|| {
            Error::InvalidFormat("extensible array page element count overflow".into())
        })?;
    let derived_super_block_count = usize::from(max_elements_bits)
        .checked_sub(super::log2_power2(u64::from(data_block_min_elements))?)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| Error::InvalidFormat("invalid extensible array block parameters".into()))?;
    let index_block_super_blocks = super::log2_power2(u64::from(super_block_min_data_ptrs))?
        .checked_mul(2)
        .ok_or_else(|| {
            Error::InvalidFormat("extensible array index block sizing overflow".into())
        })?;
    let index_block_data_block_addrs = usize::from(super_block_min_data_ptrs)
        .checked_sub(1)
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| {
            Error::InvalidFormat("extensible array index block sizing overflow".into())
        })?;
    let index_block_super_block_addrs = derived_super_block_count
        .checked_sub(index_block_super_blocks)
        .ok_or_else(|| {
            Error::InvalidFormat("invalid extensible array super block layout".into())
        })?;
    let super_block_info = build_super_block_info(
        derived_super_block_count,
        usize::from(data_block_min_elements),
    )?;

    Ok(ParsedExtensibleArrayHeader {
        class_id,
        raw_element_size,
        index_block_elements,
        data_block_min_elements: usize::from(data_block_min_elements),
        super_block_count: stored_super_block_count,
        super_block_size,
        data_block_count,
        data_block_size,
        max_index_set,
        realized_elements,
        index_block_addr,
        array_offset_size,
        data_block_page_elements,
        index_block_super_blocks,
        index_block_data_block_addrs,
        index_block_super_block_addrs,
        derived_super_block_count,
        super_block_info,
        checksum_pos,
        super_block_count_pos,
        super_block_size_pos,
        data_block_count_pos,
        data_block_size_pos,
        max_index_set_pos,
        realized_elements_pos,
    })
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

fn build_super_block_info(
    count: usize,
    min_data_block_elements: usize,
) -> Result<Vec<SuperBlockInfo>> {
    let mut infos = Vec::with_capacity(count);
    let mut start_index = 0u64;
    let mut start_data_block = 0u64;
    for index in 0..count {
        let data_blocks = 1usize
            .checked_shl(u32::try_from(index / 2).map_err(|_| {
                Error::InvalidFormat("extensible array data block shift overflow".into())
            })?)
            .ok_or_else(|| {
                Error::InvalidFormat("extensible array data block count overflow".into())
            })?;
        let data_block_elements = min_data_block_elements
            .checked_mul(
                1usize
                    .checked_shl(u32::try_from(index.div_ceil(2)).map_err(|_| {
                        Error::InvalidFormat(
                            "extensible array data block element shift overflow".into(),
                        )
                    })?)
                    .ok_or_else(|| {
                        Error::InvalidFormat(
                            "extensible array data block element count overflow".into(),
                        )
                    })?,
            )
            .ok_or_else(|| {
                Error::InvalidFormat("extensible array data block size overflow".into())
            })?;
        infos.push(SuperBlockInfo {
            data_blocks,
            data_block_elements,
            start_data_block,
        });
        let index_span = super::u64_from_usize(data_blocks, "extensible array data block count")?
            .checked_mul(super::u64_from_usize(
                data_block_elements,
                "extensible array data block elements",
            )?)
            .ok_or_else(|| Error::InvalidFormat("extensible array start index overflow".into()))?;
        start_index = start_index
            .checked_add(index_span)
            .ok_or_else(|| Error::InvalidFormat("extensible array start index overflow".into()))?;
        start_data_block = start_data_block
            .checked_add(super::u64_from_usize(
                data_blocks,
                "extensible array data block count",
            )?)
            .ok_or_else(|| {
                Error::InvalidFormat("extensible array data block index overflow".into())
            })?;
    }
    Ok(infos)
}

fn encode_var(out: &mut Vec<u8>, value: u64, size: usize) -> Result<()> {
    if size > 8 {
        return Err(Error::InvalidFormat(
            "extensible array encoded integer size exceeds u64".into(),
        ));
    }
    let bytes = value.to_le_bytes();
    out.extend_from_slice(&bytes[..size]);
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use crate::io::HdfReader;

    use super::{build_super_block_info, read_header_core};

    #[test]
    fn build_super_block_info_rejects_start_index_product_overflow() {
        let err = build_super_block_info(3, usize::MAX / 4 + 1).unwrap_err();
        assert!(err.to_string().contains("start index overflow"));
    }

    #[test]
    fn extensible_array_header_rejects_zero_element_size() {
        let mut bytes = b"EAHD".to_vec();
        bytes.push(0);
        bytes.push(1);
        bytes.push(0);
        bytes.push(4);
        bytes.push(1);
        bytes.push(1);
        bytes.push(1);
        bytes.push(1);
        bytes.extend_from_slice(&0u64.to_le_bytes()); // super block count
        bytes.extend_from_slice(&0u64.to_le_bytes()); // super block size
        bytes.extend_from_slice(&0u64.to_le_bytes()); // data block count
        bytes.extend_from_slice(&0u64.to_le_bytes()); // data block size
        bytes.extend_from_slice(&1u64.to_le_bytes()); // max index set
        bytes.extend_from_slice(&0u64.to_le_bytes()); // realized elements
        bytes.extend_from_slice(&crate::io::reader::UNDEF_ADDR.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());

        let mut reader = HdfReader::new(Cursor::new(bytes));
        let err = read_header_core(&mut reader, 0).expect_err("zero element size should fail");
        assert!(
            err.to_string().contains("element size"),
            "unexpected error: {err}"
        );
    }
}
