//! Extensible array super-block — mirrors libhdf5's `H5EAsblock.c` plus
//! the super-block half of `H5EAcache.c`
//! (`H5EA__cache_sblock_deserialize`).

#![allow(dead_code)]

use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::format::checksum::checksum_metadata;
use crate::io::reader::{is_undef_addr, HdfReader};

use super::dblock::append_data_block_elements;
use super::fixed_array::FixedArrayElement;
use super::hdr::{ExtensibleArrayHeader, SuperBlockInfo};

/// Decoded extensible-array super-block contents — page-init bitmap and
/// the data-block address table. Mirrors `H5EA__cache_sblock_deserialize`:
/// magic+version+class+owner+offset are validated, all variable-length
/// arrays are pulled, but no descent into the listed data blocks happens.
pub(super) struct ExtArraySuperBlock {
    /// Concatenated page-init bytes; one slice of `page_init_size` bytes
    /// per data block, or empty when the data blocks aren't paginated.
    pub(super) page_init: Vec<u8>,
    /// Bytes of page-init data per data block (0 if unpaginated).
    pub(super) page_init_size: usize,
    /// Addresses of the data blocks owned by this super-block.
    pub(super) data_block_addrs: Vec<u64>,
}

pub(super) fn sblock_debug(block: &ExtArraySuperBlock) -> String {
    format!(
        "ExtArraySuperBlock(page_init_len={}, page_init_size={}, data_block_addrs={})",
        block.page_init.len(),
        block.page_init_size,
        block.data_block_addrs.len()
    )
}

pub(super) fn cache_sblock_verify_chksum(data: &[u8]) -> Result<()> {
    verify_trailing_checksum(data, "extensible array super block")
}

pub(super) fn cache_sblock_image_len(
    header: &ExtensibleArrayHeader,
    info: &SuperBlockInfo,
    addr_size: usize,
) -> Result<usize> {
    let page_init_size = super::data_block_pages(header, info.data_block_elements).div_ceil(8);
    let page_init_len = super::checked_usize_mul(
        info.data_blocks,
        page_init_size,
        "extensible array super block image length",
    )?;
    let addr_len = super::checked_usize_mul(
        info.data_blocks,
        addr_size,
        "extensible array super block image length",
    )?;
    4usize
        .checked_add(1)
        .and_then(|value| value.checked_add(1))
        .and_then(|value| value.checked_add(addr_size))
        .and_then(|value| value.checked_add(usize::from(header.array_offset_size)))
        .and_then(|value| value.checked_add(page_init_len))
        .and_then(|value| value.checked_add(addr_len))
        .and_then(|value| value.checked_add(4))
        .ok_or_else(|| {
            Error::InvalidFormat("extensible array super block image length overflow".into())
        })
}

pub(super) fn cache_sblock_serialize(prefix_and_payload: &[u8]) -> Vec<u8> {
    let mut out = prefix_and_payload.to_vec();
    let checksum = crate::format::checksum::checksum_metadata(&out);
    out.extend_from_slice(&checksum.to_le_bytes());
    out
}

pub(super) fn cache_sblock_notify(_block: &ExtArraySuperBlock) {}

pub(super) fn cache_sblock_free_icr(_block: ExtArraySuperBlock) {}

pub(super) fn sblock_alloc(page_init_size: usize, data_blocks: usize) -> ExtArraySuperBlock {
    ExtArraySuperBlock {
        page_init: vec![0; page_init_size.saturating_mul(data_blocks)],
        page_init_size,
        data_block_addrs: vec![crate::io::reader::UNDEF_ADDR; data_blocks],
    }
}

pub(super) fn sblock_unprotect(_block: ExtArraySuperBlock) {}

pub(super) fn sblock_delete(block: &mut ExtArraySuperBlock) {
    block.page_init.clear();
    block.data_block_addrs.clear();
}

pub(super) fn sblock_dest(_block: ExtArraySuperBlock) {}

/// Pure deserializer for an extensible-array super-block.
pub(super) fn decode_super_block<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header_addr: u64,
    header: &ExtensibleArrayHeader,
    super_block_addr: u64,
    info: &SuperBlockInfo,
) -> Result<ExtArraySuperBlock> {
    reader.seek(super_block_addr)?;
    let magic = reader.read_bytes(4)?;
    if magic != b"EASB" {
        return Err(Error::InvalidFormat(
            "invalid extensible array super block magic".into(),
        ));
    }

    let version = reader.read_u8()?;
    if version != 0 {
        return Err(Error::Unsupported(format!(
            "extensible array super block version {version}"
        )));
    }

    let class_id = reader.read_u8()?;
    if class_id != header.class_id {
        return Err(Error::InvalidFormat(
            "extensible array super block class does not match header".into(),
        ));
    }

    let owner = reader.read_addr()?;
    if owner != header_addr {
        return Err(Error::InvalidFormat(
            "extensible array super block owner address does not match header".into(),
        ));
    }

    let _block_offset = reader.read_uint(header.array_offset_size)?;
    let data_block_pages = super::data_block_pages(header, info.data_block_elements);
    let page_init_size = if data_block_pages > 0 {
        data_block_pages.div_ceil(8)
    } else {
        0
    };
    let page_init = if page_init_size > 0 {
        let page_init_len = super::checked_usize_mul(
            info.data_blocks,
            page_init_size,
            "extensible array super block page-init size",
        )?;
        reader.read_bytes(page_init_len)?
    } else {
        Vec::new()
    };

    let mut data_block_addrs = Vec::with_capacity(info.data_blocks);
    for _ in 0..info.data_blocks {
        data_block_addrs.push(reader.read_addr()?);
    }

    verify_reader_checksum(reader, super_block_addr, "extensible array super block")?;
    Ok(ExtArraySuperBlock {
        page_init,
        page_init_size,
        data_block_addrs,
    })
}

fn verify_trailing_checksum(data: &[u8], context: &str) -> Result<()> {
    if data.len() < 4 {
        return Err(Error::InvalidFormat(format!("{context} image too short")));
    }
    let split = data.len() - 4;
    let stored = u32::from_le_bytes(
        data[split..]
            .try_into()
            .map_err(|_| Error::InvalidFormat(format!("{context} checksum is truncated")))?,
    );
    let computed = crate::format::checksum::checksum_metadata(&data[..split]);
    if stored != computed {
        return Err(Error::InvalidFormat(format!(
            "{context} checksum mismatch: stored={stored:#010x}, computed={computed:#010x}"
        )));
    }
    Ok(())
}

fn verify_reader_checksum<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    start: u64,
    context: &str,
) -> Result<()> {
    let checksum_pos = reader.position()?;
    let stored = reader.read_u32()?;
    let check_len = usize::try_from(
        checksum_pos
            .checked_sub(start)
            .ok_or_else(|| Error::InvalidFormat(format!("{context} checksum span underflow")))?,
    )
    .map_err(|_| Error::InvalidFormat(format!("{context} checksum span is too large")))?;
    reader.seek(start)?;
    let bytes = reader.read_bytes(check_len)?;
    let computed = checksum_metadata(&bytes);
    if stored != computed {
        return Err(Error::InvalidFormat(format!(
            "{context} checksum mismatch: stored={stored:#010x}, computed={computed:#010x}"
        )));
    }
    reader.seek(checksum_pos.checked_add(4).ok_or_else(|| {
        Error::InvalidFormat(format!("{context} checksum end offset overflow"))
    })?)?;
    Ok(())
}

/// Walk a decoded super-block: descend into each owned data block and
/// stream elements into the shared output vector.
#[allow(clippy::too_many_arguments)]
pub(super) fn read_super_block<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header_addr: u64,
    header: &ExtensibleArrayHeader,
    filtered: bool,
    chunk_size_len: usize,
    super_block_addr: u64,
    info: &SuperBlockInfo,
    elements: &mut Vec<FixedArrayElement>,
) -> Result<()> {
    if is_undef_addr(super_block_addr) {
        let fill_count = super::checked_usize_mul(
            info.data_blocks,
            info.data_block_elements,
            "extensible array super block fill count",
        )?;
        super::append_fill_elements(header, fill_count, elements)?;
        return Ok(());
    }

    let sblock = decode_super_block(reader, header_addr, header, super_block_addr, info)?;

    for (data_block_index, &data_block_addr) in sblock.data_block_addrs.iter().enumerate() {
        let elements_len =
            super::u64_from_usize(elements.len(), "extensible array decoded element count")?;
        if elements_len >= header.max_index_set {
            break;
        }
        let remaining = super::usize_from_u64(
            header.max_index_set - elements_len,
            "extensible array remaining element count",
        )?;
        let count = info.data_block_elements.min(remaining);
        let page_init_for_block = if sblock.page_init_size > 0 {
            let start = super::checked_usize_mul(
                data_block_index,
                sblock.page_init_size,
                "extensible array super block page-init offset",
            )?;
            let end = super::checked_usize_add(
                start,
                sblock.page_init_size,
                "extensible array super block page-init offset",
            )?;
            Some(sblock.page_init.get(start..end).ok_or_else(|| {
                Error::InvalidFormat("extensible array page-init slice out of bounds".into())
            })?)
        } else {
            None
        };
        append_data_block_elements(
            reader,
            header_addr,
            header,
            filtered,
            chunk_size_len,
            data_block_addr,
            info.data_block_elements,
            page_init_for_block,
            count,
            elements,
        )?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use crate::io::HdfReader;

    use super::*;

    fn header() -> ExtensibleArrayHeader {
        ExtensibleArrayHeader {
            class_id: 1,
            raw_element_size: 8,
            index_block_elements: 0,
            max_index_set: 0,
            index_block_addr: 0,
            array_offset_size: 1,
            data_block_page_elements: 4,
            index_block_super_blocks: 0,
            index_block_data_block_addrs: 0,
            index_block_super_block_addrs: 0,
            super_block_info: Vec::new(),
        }
    }

    #[test]
    fn extensible_array_super_block_rejects_bad_checksum() {
        let mut bytes = b"EASB".to_vec();
        bytes.push(0);
        bytes.push(1);
        bytes.extend_from_slice(&100u64.to_le_bytes());
        bytes.push(0); // block offset, array_offset_size = 1
        bytes.extend_from_slice(&55u64.to_le_bytes());
        let checksum = checksum_metadata(&bytes);
        bytes.extend_from_slice(&checksum.to_le_bytes());
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;

        let info = SuperBlockInfo {
            data_blocks: 1,
            data_block_elements: 1,
            start_data_block: 0,
        };
        let mut reader = HdfReader::new(Cursor::new(bytes));
        let err = match decode_super_block(&mut reader, 100, &header(), 0, &info) {
            Ok(_) => panic!("bad extensible array super block checksum should fail"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("checksum"));
    }
}
