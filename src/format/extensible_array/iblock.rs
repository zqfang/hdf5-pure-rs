//! Extensible array index block — mirrors libhdf5's `H5EAiblock.c` plus
//! the iblock half of `H5EAcache.c` (`H5EA__cache_iblock_deserialize`).
//! Includes the spillover descent (the post-iblock walk into super-blocks
//! and data-blocks) that `H5EA_iterate` performs after `H5EA__iblock_protect`.

#![allow(dead_code)]

use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::format::checksum::checksum_metadata;
use crate::io::reader::HdfReader;

use super::dblock::{append_data_block_elements, read_element};
use super::fixed_array::FixedArrayElement;
use super::hdr::ExtensibleArrayHeader;
use super::sblock::read_super_block;

/// Decoded extensible-array index block: the inline elements plus the
/// data-block and super-block address tables. Mirrors
/// `H5EA__cache_iblock_deserialize` in libhdf5: parse the prefix and the
/// variable-length tables, but don't dereference the listed addresses.
pub(super) struct ExtArrayIndexBlock {
    /// Inline elements stored directly in the index block.
    pub(super) elements: Vec<FixedArrayElement>,
    /// Addresses of the data blocks owned directly by the index block.
    pub(super) data_block_addrs: Vec<u64>,
    /// Addresses of the super-blocks owned by the index block.
    pub(super) super_block_addrs: Vec<u64>,
}

pub(super) fn iblock_debug(block: &ExtArrayIndexBlock) -> String {
    format!(
        "ExtArrayIndexBlock(elements={}, data_block_addrs={}, super_block_addrs={})",
        block.elements.len(),
        block.data_block_addrs.len(),
        block.super_block_addrs.len()
    )
}

pub(super) fn cache_iblock_get_initial_load_size() -> usize {
    4 + 1
}

pub(super) fn cache_iblock_verify_chksum(data: &[u8]) -> Result<()> {
    verify_trailing_checksum(data, "extensible array index block")
}

pub(super) fn cache_iblock_image_len(
    header: &ExtensibleArrayHeader,
    addr_size: usize,
) -> Result<usize> {
    let element_bytes = super::checked_usize_mul(
        usize::from(header.index_block_elements),
        header.raw_element_size,
        "extensible array index block image length",
    )?;
    let data_addr_bytes = super::checked_usize_mul(
        header.index_block_data_block_addrs,
        addr_size,
        "extensible array index block image length",
    )?;
    let super_addr_bytes = super::checked_usize_mul(
        header.index_block_super_block_addrs,
        addr_size,
        "extensible array index block image length",
    )?;
    4usize
        .checked_add(1)
        .and_then(|value| value.checked_add(1))
        .and_then(|value| value.checked_add(addr_size))
        .and_then(|value| value.checked_add(element_bytes))
        .and_then(|value| value.checked_add(data_addr_bytes))
        .and_then(|value| value.checked_add(super_addr_bytes))
        .and_then(|value| value.checked_add(4))
        .ok_or_else(|| {
            Error::InvalidFormat("extensible array index block image length overflow".into())
        })
}

pub(super) fn cache_iblock_serialize(prefix_and_payload: &[u8]) -> Result<Vec<u8>> {
    let image_len = prefix_and_payload.len().checked_add(4).ok_or_else(|| {
        Error::InvalidFormat("extensible array index block image length overflow".into())
    })?;
    let mut out = Vec::with_capacity(image_len);
    out.extend_from_slice(prefix_and_payload);
    let checksum = crate::format::checksum::checksum_metadata(&out);
    out.extend_from_slice(&checksum.to_le_bytes());
    Ok(out)
}

pub(super) fn cache_iblock_notify(_block: &ExtArrayIndexBlock) {}

pub(super) fn cache_iblock_free_icr(_block: ExtArrayIndexBlock) {}

pub(super) fn iblock_alloc() -> ExtArrayIndexBlock {
    ExtArrayIndexBlock {
        elements: Vec::new(),
        data_block_addrs: Vec::new(),
        super_block_addrs: Vec::new(),
    }
}

pub(super) fn iblock_create(elements: Vec<FixedArrayElement>) -> ExtArrayIndexBlock {
    ExtArrayIndexBlock {
        elements,
        data_block_addrs: Vec::new(),
        super_block_addrs: Vec::new(),
    }
}

pub(super) fn iblock_unprotect(_block: ExtArrayIndexBlock) {}

pub(super) fn iblock_delete(block: &mut ExtArrayIndexBlock) {
    block.elements.clear();
    block.data_block_addrs.clear();
    block.super_block_addrs.clear();
}

pub(super) fn iblock_dest(_block: ExtArrayIndexBlock) {}

/// Pure deserializer for the extensible-array index block.
pub(super) fn decode_index_block<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header_addr: u64,
    header: &ExtensibleArrayHeader,
    filtered: bool,
    chunk_size_len: usize,
) -> Result<ExtArrayIndexBlock> {
    reader.seek(header.index_block_addr)?;
    let magic = reader.read_bytes(4)?;
    if magic != b"EAIB" {
        return Err(Error::InvalidFormat(
            "invalid extensible array index block magic".into(),
        ));
    }

    let version = reader.read_u8()?;
    if version != 0 {
        return Err(Error::Unsupported(format!(
            "extensible array index block version {version}"
        )));
    }

    let class_id = reader.read_u8()?;
    if class_id != header.class_id {
        return Err(Error::InvalidFormat(
            "extensible array index block class does not match header".into(),
        ));
    }

    let owner = reader.read_addr()?;
    if owner != header_addr {
        return Err(Error::InvalidFormat(
            "extensible array index block owner address does not match header".into(),
        ));
    }

    let expected_element_size = if filtered {
        super::checked_usize_add(
            super::checked_usize_add(
                usize::from(reader.sizeof_addr()),
                chunk_size_len,
                "extensible array raw element size",
            )?,
            4,
            "extensible array raw element size",
        )?
    } else {
        usize::from(reader.sizeof_addr())
    };
    if header.raw_element_size != expected_element_size {
        return Err(Error::InvalidFormat(format!(
            "extensible array raw element size {} does not match expected {}",
            header.raw_element_size, expected_element_size
        )));
    }

    let max_index_count =
        super::usize_from_u64(header.max_index_set, "extensible array max index")?;
    let mut elements = Vec::with_capacity(max_index_count);
    for idx in 0..header.index_block_elements {
        let element = read_element(reader, filtered, chunk_size_len)?;
        if u64::from(idx) < header.max_index_set {
            elements.push(element);
        }
    }

    let mut data_block_addrs = Vec::with_capacity(header.index_block_data_block_addrs);
    for _ in 0..header.index_block_data_block_addrs {
        data_block_addrs.push(reader.read_addr()?);
    }

    let mut super_block_addrs = Vec::with_capacity(header.index_block_super_block_addrs);
    for _ in 0..header.index_block_super_block_addrs {
        super_block_addrs.push(reader.read_addr()?);
    }

    verify_reader_checksum(
        reader,
        header.index_block_addr,
        "extensible array index block",
    )?;
    Ok(ExtArrayIndexBlock {
        elements,
        data_block_addrs,
        super_block_addrs,
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

/// Drive the decoded index block to materialize the full element vector
/// — descends into spillover data/super blocks. Composition mirrors
/// the C-side `H5EA_iterate` after `H5EA__iblock_protect` returns.
pub(super) fn read_index_block<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header_addr: u64,
    header: &ExtensibleArrayHeader,
    filtered: bool,
    chunk_size_len: usize,
) -> Result<Vec<FixedArrayElement>> {
    let iblock = decode_index_block(reader, header_addr, header, filtered, chunk_size_len)?;
    let ExtArrayIndexBlock {
        mut elements,
        data_block_addrs,
        super_block_addrs,
    } = iblock;
    read_spillover_blocks(
        reader,
        header_addr,
        header,
        filtered,
        chunk_size_len,
        &data_block_addrs,
        &super_block_addrs,
        &mut elements,
    )?;
    Ok(elements)
}

#[allow(clippy::too_many_arguments)]
fn read_spillover_blocks<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header_addr: u64,
    header: &ExtensibleArrayHeader,
    filtered: bool,
    chunk_size_len: usize,
    data_block_addrs: &[u64],
    super_block_addrs: &[u64],
    elements: &mut Vec<FixedArrayElement>,
) -> Result<()> {
    for (super_block_index, info) in header.super_block_info.iter().enumerate() {
        let elements_len =
            super::u64_from_usize(elements.len(), "extensible array decoded element count")?;
        if elements_len >= header.max_index_set {
            break;
        }

        if super_block_index < header.index_block_super_blocks {
            for local_data_block in 0..info.data_blocks {
                let elements_len = super::u64_from_usize(
                    elements.len(),
                    "extensible array decoded element count",
                )?;
                if elements_len >= header.max_index_set {
                    break;
                }

                let data_block_index = super::usize_from_u64(
                    info.start_data_block,
                    "extensible array data block start index",
                )
                .and_then(|start| {
                    super::checked_usize_add(
                        start,
                        local_data_block,
                        "extensible array data block address index",
                    )
                })?;
                let Some(&data_block_addr) = data_block_addrs.get(data_block_index) else {
                    return Err(Error::InvalidFormat(
                        "extensible array data block address index out of bounds".into(),
                    ));
                };
                let remaining = super::usize_from_u64(
                    header.max_index_set - elements_len,
                    "extensible array remaining element count",
                )?;
                let count = info.data_block_elements.min(remaining);
                append_data_block_elements(
                    reader,
                    header_addr,
                    header,
                    filtered,
                    chunk_size_len,
                    data_block_addr,
                    info.data_block_elements,
                    None,
                    count,
                    elements,
                )?;
            }
        } else {
            let super_block_addr_index = super_block_index - header.index_block_super_blocks;
            let Some(&super_block_addr) = super_block_addrs.get(super_block_addr_index) else {
                return Err(Error::InvalidFormat(
                    "extensible array super block address index out of bounds".into(),
                ));
            };
            read_super_block(
                reader,
                header_addr,
                header,
                filtered,
                chunk_size_len,
                super_block_addr,
                info,
                elements,
            )?;
        }
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
            index_block_elements: 1,
            max_index_set: 1,
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
    fn extensible_array_index_block_rejects_bad_checksum() {
        let mut bytes = b"EAIB".to_vec();
        bytes.push(0);
        bytes.push(1);
        bytes.extend_from_slice(&100u64.to_le_bytes());
        bytes.extend_from_slice(&55u64.to_le_bytes());
        let checksum = checksum_metadata(&bytes);
        bytes.extend_from_slice(&checksum.to_le_bytes());
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;

        let mut reader = HdfReader::new(Cursor::new(bytes));
        let err = match decode_index_block(&mut reader, 100, &header(), false, 0) {
            Ok(_) => panic!("bad extensible array index block checksum should fail"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("checksum"));
    }
}
