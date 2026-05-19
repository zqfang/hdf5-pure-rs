//! Fixed array data block — mirrors libhdf5's `H5FAdblock.c` plus the
//! data-block half of `H5FAcache.c` (`H5FA__cache_dblock_deserialize`).
//! Page-init + per-page iteration (libhdf5's `H5FAdblkpage.c`) is folded
//! in here because the Rust port doesn't model pages as a separate cache
//! entry.

#![allow(dead_code)]

use std::{
    fmt,
    io::{Read, Seek},
};

use crate::error::{Error, Result};
use crate::format::checksum::checksum_metadata;
use crate::io::reader::HdfReader;

use super::hdr::FixedArrayHeader;
use super::{
    bit_is_set, checked_u64_add, checked_usize_add, checked_usize_mul, u64_from_usize,
    FixedArrayElement,
};

/// Decoded fixed-array data-block prefix — magic, version, class id,
/// owner-address validation, page-init bitmap (if paginated), and the
/// computed page-layout numbers. Mirrors `H5FA__cache_dblock_deserialize`
/// in libhdf5: pure parsing of the fixed-size header, no element walk.
pub(super) struct FixedArrayDataBlockPrefix {
    /// Whether the elements are split into pages (vs a single contiguous run).
    pub(super) paginated: bool,
    /// Number of pages (only meaningful when `paginated`).
    pub(super) pages: usize,
    /// Page-initialized bitmap (one bit per page).
    pub(super) page_init: Vec<u8>,
    /// Total size of the on-disk header (used to compute page addresses).
    pub(super) prefix_size: usize,
    /// On-disk size of one element record (filtered vs unfiltered).
    pub(super) raw_element_size: usize,
    /// Per-page element count (`1 << max_page_elements_bits`).
    pub(super) page_elements: usize,
    /// Total element count.
    pub(super) element_count: usize,
}

/// Format a data-block prefix for debug printing.
pub(super) fn write_dblock_debug(
    prefix: &FixedArrayDataBlockPrefix,
    out: &mut impl fmt::Write,
) -> fmt::Result {
    write!(
        out,
        "FixedArrayDataBlockPrefix(paginated={}, pages={}, prefix_size={}, raw_element_size={}, page_elements={}, element_count={})",
        prefix.paginated,
        prefix.pages,
        prefix.prefix_size,
        prefix.raw_element_size,
        prefix.page_elements,
        prefix.element_count
    )
}

/// Compute the on-disk size of a fixed array data block.
pub(super) fn cache_dblock_image_len(prefix: &FixedArrayDataBlockPrefix) -> Result<usize> {
    if prefix.paginated {
        Ok(prefix.prefix_size)
    } else {
        checked_usize_mul(
            prefix.element_count,
            prefix.raw_element_size,
            "fixed array data block image length",
        )
        .and_then(|value| checked_usize_add(value, 4, "fixed array data block image length"))
    }
}

/// Serialize a dirty data block to its on-disk image (payload + checksum).
pub(super) fn cache_dblock_serialize_into(
    prefix_and_payload: &[u8],
    out: &mut Vec<u8>,
) -> Result<()> {
    let image_len = prefix_and_payload.len().checked_add(4).ok_or_else(|| {
        Error::InvalidFormat("fixed array data block image length overflow".into())
    })?;
    out.clear();
    out.reserve(image_len);
    out.extend_from_slice(prefix_and_payload);
    let checksum = crate::format::checksum::checksum_metadata(&out);
    out.extend_from_slice(&checksum.to_le_bytes());
    Ok(())
}

/// Handle metadata-cache action notifications for the data block.
pub(super) fn cache_dblock_notify(_prefix: &FixedArrayDataBlockPrefix) {}

/// Destroy/release an in-core representation of a data block.
pub(super) fn cache_dblock_free_icr(_prefix: FixedArrayDataBlockPrefix) {}

/// Report the file-space size of a data block to the metadata cache.
pub(super) fn cache_dblock_fsf_size(prefix: &FixedArrayDataBlockPrefix) -> usize {
    prefix.prefix_size
}

/// Initial number of bytes the metadata cache must read for a data block page.
pub(super) fn cache_dblk_page_get_initial_load_size() -> usize {
    4
}

/// Verify the trailing checksum of a data-block-page image.
pub(super) fn cache_dblk_page_verify_chksum(data: &[u8]) -> Result<()> {
    verify_trailing_checksum(data, "fixed array data block page")
}

/// Verify the page's trailing checksum and return its element payload.
pub(super) fn cache_dblk_page_deserialize(payload: &[u8]) -> Result<&[u8]> {
    if payload.len() < 4 {
        return Err(Error::InvalidFormat(
            "fixed array data block page is truncated".into(),
        ));
    }
    cache_dblk_page_verify_chksum(payload)?;
    Ok(&payload[..payload.len() - 4])
}

/// Compute the on-disk size of a data-block page.
pub(super) fn cache_dblk_page_image_len(payload_len: usize) -> Result<usize> {
    payload_len.checked_add(4).ok_or_else(|| {
        Error::InvalidFormat("fixed array data block page image length overflow".into())
    })
}

/// Serialize a data-block page to its on-disk image (payload + checksum).
pub(super) fn cache_dblk_page_serialize_into(payload: &[u8], out: &mut Vec<u8>) -> Result<()> {
    let image_len = cache_dblk_page_image_len(payload.len())?;
    out.clear();
    out.reserve(image_len);
    out.extend_from_slice(payload);
    let checksum = crate::format::checksum::checksum_metadata(&out);
    out.extend_from_slice(&checksum.to_le_bytes());
    Ok(())
}

/// Handle metadata-cache action notifications for a data-block page.
pub(super) fn cache_dblk_page_notify(_page_index: usize) {}

/// Destroy/release an in-core representation of a data-block page.
pub(super) fn cache_dblk_page_free_icr(_payload: Vec<u8>) {}

/// Allocate a zero-filled fixed-array data-block page buffer of `size` bytes.
pub(super) fn dblk_page_alloc(size: usize) -> Vec<u8> {
    vec![0; size]
}

/// Protect a data-block page in the metadata cache (no-op borrow).
pub(super) fn dblk_page_protect(payload: &[u8]) -> &[u8] {
    payload
}

/// Unprotect a data-block page.
pub(super) fn dblk_page_unprotect(_payload: &[u8]) {}

/// Destroy a fixed-array data-block page in memory.
pub(super) fn dblk_page_dest(_payload: Vec<u8>) {}

/// Allocate a fixed-array data-block prefix descriptor with the given sizing fields.
pub(super) fn dblock_alloc(
    paginated: bool,
    pages: usize,
    raw_element_size: usize,
    page_elements: usize,
    element_count: usize,
) -> FixedArrayDataBlockPrefix {
    FixedArrayDataBlockPrefix {
        paginated,
        pages,
        page_init: if paginated {
            vec![0; pages.div_ceil(8)]
        } else {
            Vec::new()
        },
        prefix_size: 0,
        raw_element_size,
        page_elements,
        element_count,
    }
}

/// Create a new fixed-array data block (returns the provided prefix value).
pub(super) fn dblock_create(prefix: FixedArrayDataBlockPrefix) -> FixedArrayDataBlockPrefix {
    prefix
}

/// Unprotect a previously protected fixed-array data block.
pub(super) fn dblock_unprotect(_prefix: FixedArrayDataBlockPrefix) {}

/// Delete a data block by clearing its page-init bitmap and element count.
pub(super) fn dblock_delete(prefix: &mut FixedArrayDataBlockPrefix) {
    prefix.page_init.clear();
    prefix.element_count = 0;
}

/// Destroy a fixed-array data block in memory.
pub(super) fn dblock_dest(_prefix: FixedArrayDataBlockPrefix) {}

/// Deserialize a fixed-array data block prefix from disk.
/// Validates magic, version, class, owner address, raw element size, and
/// (for paginated blocks) reads the page-init bitmap.
pub(super) fn decode_data_block_prefix<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header_addr: u64,
    header: &FixedArrayHeader,
    filtered: bool,
    chunk_size_len: usize,
) -> Result<FixedArrayDataBlockPrefix> {
    reader.seek(header.data_block_addr)?;
    let mut magic = [0u8; 4];
    reader.read_bytes_into(&mut magic)?;
    if magic != *b"FADB" {
        return Err(Error::InvalidFormat(
            "invalid fixed array data block magic".into(),
        ));
    }

    let version = reader.read_u8()?;
    if version != 0 {
        return Err(Error::Unsupported(format!(
            "fixed array data block version {version}"
        )));
    }

    let class_id = reader.read_u8()?;
    if class_id != header.class_id {
        return Err(Error::InvalidFormat(
            "fixed array data block class does not match header".into(),
        ));
    }

    let owner = reader.read_addr()?;
    if owner != header_addr {
        return Err(Error::InvalidFormat(
            "fixed array data block owner address does not match header".into(),
        ));
    }

    let page_elements = 1usize
        .checked_shl(u32::from(header.max_page_elements_bits))
        .ok_or_else(|| Error::InvalidFormat("fixed array page size overflow".into()))?;
    let expected_element_size = if filtered {
        checked_usize_add(
            checked_usize_add(
                usize::from(reader.sizeof_addr()),
                chunk_size_len,
                "fixed array raw element size",
            )?,
            4,
            "fixed array raw element size",
        )?
    } else {
        usize::from(reader.sizeof_addr())
    };
    if header.raw_element_size != expected_element_size {
        return Err(Error::InvalidFormat(format!(
            "fixed array raw element size {} does not match expected {}",
            header.raw_element_size, expected_element_size
        )));
    }

    let element_count = super::usize_from_u64(header.elements, "fixed array element count")?;
    let paginated = element_count > page_elements;
    if paginated {
        let pages = element_count.div_ceil(page_elements);
        let page_init_size = pages.div_ceil(8);
        let mut page_init = vec![0; page_init_size];
        reader.read_bytes_into(&mut page_init)?;
        verify_reader_checksum(
            reader,
            header.data_block_addr,
            "fixed array data block prefix",
        )?;
        let prefix_size = checked_usize_add(
            4 + 1 + 1,
            usize::from(reader.sizeof_addr()),
            "fixed array data block prefix size",
        )
        .and_then(|value| {
            checked_usize_add(value, page_init_size, "fixed array data block prefix size")
        })
        .and_then(|value| checked_usize_add(value, 4, "fixed array data block prefix size"))?;
        Ok(FixedArrayDataBlockPrefix {
            paginated: true,
            pages,
            page_init,
            prefix_size,
            raw_element_size: header.raw_element_size,
            page_elements,
            element_count,
        })
    } else {
        Ok(FixedArrayDataBlockPrefix {
            paginated: false,
            pages: 0,
            page_init: Vec::new(),
            prefix_size: 0,
            raw_element_size: header.raw_element_size,
            page_elements,
            element_count,
        })
    }
}

/// Verify the trailing 4-byte metadata checksum of an in-memory image.
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

/// Walk a decoded data-block prefix and materialize its element vector,
/// honoring page-init bitmaps for paginated blocks.
pub(super) fn collect_data_block_elements_into<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header: &FixedArrayHeader,
    prefix: &FixedArrayDataBlockPrefix,
    filtered: bool,
    chunk_size_len: usize,
    elements: &mut Vec<FixedArrayElement>,
) -> Result<()> {
    elements.clear();
    elements.reserve(prefix.element_count);
    if prefix.paginated {
        let page_payload = checked_usize_mul(
            prefix.page_elements,
            prefix.raw_element_size,
            "fixed array data block page size",
        )?;
        let page_size = checked_usize_add(page_payload, 4, "fixed array data block page size")?;
        for page_index in 0..prefix.pages {
            let page_start = checked_usize_mul(
                page_index,
                prefix.page_elements,
                "fixed array page start index",
            )?;
            let remaining = prefix
                .element_count
                .checked_sub(page_start)
                .ok_or_else(|| {
                    Error::InvalidFormat("fixed array page start exceeds element count".into())
                })?;
            let page_count = prefix.page_elements.min(remaining);
            if bit_is_set(&prefix.page_init, page_index) {
                let page_offset =
                    checked_usize_mul(page_index, page_size, "fixed array data block page offset")?;
                let offset = checked_usize_add(
                    prefix.prefix_size,
                    page_offset,
                    "fixed array data block page offset",
                )?;
                let page_addr = checked_u64_add(
                    header.data_block_addr,
                    u64_from_usize(offset, "fixed array data block page offset")?,
                    "fixed array data block page address",
                )?;
                reader.seek(page_addr)?;
                for _ in 0..page_count {
                    elements.push(read_element(reader, filtered, chunk_size_len)?);
                }
                verify_reader_checksum(reader, page_addr, "fixed array data block page")?;
            } else {
                super::append_fill_elements(page_count, elements);
            }
        }
    } else {
        for _ in 0..header.elements {
            elements.push(read_element(reader, filtered, chunk_size_len)?);
        }
        verify_reader_checksum(reader, header.data_block_addr, "fixed array data block")?;
    }
    Ok(())
}

/// Walk a decoded data-block prefix and stream each element to `visitor`.
pub(super) fn visit_data_block_elements<R, F>(
    reader: &mut HdfReader<R>,
    header: &FixedArrayHeader,
    prefix: &FixedArrayDataBlockPrefix,
    filtered: bool,
    chunk_size_len: usize,
    mut visitor: F,
) -> Result<()>
where
    R: Read + Seek,
    F: FnMut(FixedArrayElement) -> Result<()>,
{
    if prefix.paginated {
        let page_payload = checked_usize_mul(
            prefix.page_elements,
            prefix.raw_element_size,
            "fixed array data block page size",
        )?;
        let page_size = checked_usize_add(page_payload, 4, "fixed array data block page size")?;
        for page_index in 0..prefix.pages {
            let page_start = checked_usize_mul(
                page_index,
                prefix.page_elements,
                "fixed array page start index",
            )?;
            let remaining = prefix
                .element_count
                .checked_sub(page_start)
                .ok_or_else(|| {
                    Error::InvalidFormat("fixed array page start exceeds element count".into())
                })?;
            let page_count = prefix.page_elements.min(remaining);
            if bit_is_set(&prefix.page_init, page_index) {
                let page_offset =
                    checked_usize_mul(page_index, page_size, "fixed array data block page offset")?;
                let offset = checked_usize_add(
                    prefix.prefix_size,
                    page_offset,
                    "fixed array data block page offset",
                )?;
                let page_addr = checked_u64_add(
                    header.data_block_addr,
                    u64_from_usize(offset, "fixed array data block page offset")?,
                    "fixed array data block page address",
                )?;
                reader.seek(page_addr)?;
                for _ in 0..page_count {
                    visitor(read_element(reader, filtered, chunk_size_len)?)?;
                }
                verify_reader_checksum(reader, page_addr, "fixed array data block page")?;
            } else {
                for _ in 0..page_count {
                    visitor(FixedArrayElement {
                        addr: crate::io::reader::UNDEF_ADDR,
                        nbytes: None,
                        filter_mask: 0,
                    })?;
                }
            }
        }
    } else {
        for _ in 0..header.elements {
            visitor(read_element(reader, filtered, chunk_size_len)?)?;
        }
        verify_reader_checksum(reader, header.data_block_addr, "fixed array data block")?;
    }
    Ok(())
}

/// Read and validate the trailing checksum of the span starting at `start`.
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
    let mut bytes = vec![0; check_len];
    reader.read_bytes_into(&mut bytes)?;
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

/// Protect (load) a fixed-array data block by composing
/// `decode_data_block_prefix` and `collect_data_block_elements`. Mirrors
/// `H5FA__dblock_protect` followed by the iterate path in `H5FA_iterate`.
pub(super) fn read_data_block_into<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header_addr: u64,
    header: &FixedArrayHeader,
    filtered: bool,
    chunk_size_len: usize,
    elements: &mut Vec<FixedArrayElement>,
) -> Result<()> {
    let prefix = decode_data_block_prefix(reader, header_addr, header, filtered, chunk_size_len)?;
    collect_data_block_elements_into(reader, header, &prefix, filtered, chunk_size_len, elements)
}

/// Protect and stream a fixed-array data block to `visitor`.
pub(super) fn visit_data_block<R, F>(
    reader: &mut HdfReader<R>,
    header_addr: u64,
    header: &FixedArrayHeader,
    filtered: bool,
    chunk_size_len: usize,
    visitor: F,
) -> Result<()>
where
    R: Read + Seek,
    F: FnMut(FixedArrayElement) -> Result<()>,
{
    let prefix = decode_data_block_prefix(reader, header_addr, header, filtered, chunk_size_len)?;
    visit_data_block_elements(reader, header, &prefix, filtered, chunk_size_len, visitor)
}

/// Read a single chunk element (address, optional filtered size and mask) from the reader.
pub(super) fn read_element<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    filtered: bool,
    chunk_size_len: usize,
) -> Result<FixedArrayElement> {
    let addr = reader.read_addr()?;
    if filtered {
        let nbytes = reader.read_uint(chunk_size_len as u8)?;
        let filter_mask = reader.read_u32()?;
        Ok(FixedArrayElement {
            addr,
            nbytes: Some(nbytes),
            filter_mask,
        })
    } else {
        Ok(FixedArrayElement {
            addr,
            nbytes: None,
            filter_mask: 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use crate::io::HdfReader;

    use super::*;

    fn header(elements: u64, max_page_elements_bits: u8) -> FixedArrayHeader {
        FixedArrayHeader {
            class_id: 1,
            raw_element_size: 8,
            max_page_elements_bits,
            elements,
            data_block_addr: 0,
        }
    }

    fn append_checksum(bytes: &mut Vec<u8>) {
        let checksum = checksum_metadata(bytes);
        bytes.extend_from_slice(&checksum.to_le_bytes());
    }

    fn fixed_array_prefix(owner: u64) -> Vec<u8> {
        let mut bytes = b"FADB".to_vec();
        bytes.push(0);
        bytes.push(1);
        bytes.extend_from_slice(&owner.to_le_bytes());
        bytes
    }

    #[test]
    fn fixed_array_unpaginated_rejects_bad_checksum() {
        let mut bytes = fixed_array_prefix(100);
        bytes.extend_from_slice(&55u64.to_le_bytes());
        append_checksum(&mut bytes);
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;

        let mut reader = HdfReader::new(Cursor::new(bytes));
        let mut elements = Vec::new();
        let err = read_data_block_into(&mut reader, 100, &header(1, 4), false, 0, &mut elements)
            .expect_err("bad fixed array data block checksum should fail");
        assert!(err.to_string().contains("checksum"));
    }

    #[test]
    fn fixed_array_paginated_rejects_bad_prefix_and_page_checksums() {
        let mut bytes = fixed_array_prefix(100);
        bytes.push(0x80); // page 0 initialized, page 1 fill-valued.
        append_checksum(&mut bytes);
        let page_start = bytes.len();
        bytes.extend_from_slice(&55u64.to_le_bytes());
        append_checksum(&mut bytes);

        let mut bad_prefix = bytes.clone();
        bad_prefix[14] ^= 0x02;
        let mut reader = HdfReader::new(Cursor::new(bad_prefix));
        let mut elements = Vec::new();
        let err = read_data_block_into(&mut reader, 100, &header(2, 0), false, 0, &mut elements)
            .expect_err("bad fixed array prefix checksum should fail");
        assert!(err.to_string().contains("checksum"));

        let mut bad_page = bytes;
        let last = bad_page.len() - 1;
        bad_page[last] ^= 0xff;
        let mut reader = HdfReader::new(Cursor::new(bad_page));
        let mut elements = Vec::new();
        let err = read_data_block_into(&mut reader, 100, &header(2, 0), false, 0, &mut elements)
            .expect_err("bad fixed array page checksum should fail");
        assert!(err.to_string().contains("checksum"));
        assert_eq!(page_start, 19);
    }

    #[test]
    fn fixed_array_page_cache_serializes_and_validates_checksum() {
        let payload = 55u64.to_le_bytes();
        let mut image = Vec::new();
        cache_dblk_page_serialize_into(&payload, &mut image).unwrap();
        assert_eq!(
            cache_dblk_page_image_len(payload.len()).unwrap(),
            image.len()
        );
        assert_eq!(cache_dblk_page_deserialize(&image).unwrap(), payload);

        let mut bad = image;
        let last = bad.len() - 1;
        bad[last] ^= 0xff;
        assert!(cache_dblk_page_deserialize(&bad).is_err());
        assert!(cache_dblk_page_deserialize(&bad[..3]).is_err());
    }
}
