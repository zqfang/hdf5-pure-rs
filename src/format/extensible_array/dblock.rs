//! Extensible array data block — mirrors libhdf5's `H5EAdblock.c` plus
//! the data-block half of `H5EAcache.c`. The page handling that lives
//! in libhdf5's `H5EAdblkpage.c` is folded in here because the Rust port
//! doesn't model pages as a separate cache entry.

#![allow(dead_code)]

use std::{
    fmt,
    io::{Cursor, Read, Seek},
};

use crate::error::{Error, Result};
use crate::format::checksum::checksum_metadata;
use crate::io::reader::{is_undef_addr, HdfReader};

use super::fixed_array::FixedArrayElement;
use super::hdr::ExtensibleArrayHeader;

/// Decoded extensible-array data-block prefix: page count + raw geometry
/// numbers, all derived from the on-disk magic+version+class+owner block
/// header. Mirrors `H5EA__cache_dblock_deserialize` — pure parse, no I/O
/// over the element pages themselves.
pub(super) struct ExtArrayDataBlockPrefix {
    /// Number of pages this data block is split into (0 = unpaginated).
    pub(super) pages: usize,
    /// Total prefix size on disk (used to compute per-page offsets).
    pub(super) prefix_size: usize,
}

/// Format a data-block prefix for debug printing.
pub(super) fn write_dblock_debug(
    prefix: &ExtArrayDataBlockPrefix,
    out: &mut impl fmt::Write,
) -> fmt::Result {
    write!(
        out,
        "ExtArrayDataBlockPrefix(pages={}, prefix_size={})",
        prefix.pages, prefix.prefix_size
    )
}

/// Verify the trailing checksum of a data-block image.
pub(super) fn cache_dblock_verify_chksum(data: &[u8]) -> Result<()> {
    verify_trailing_checksum(data, "extensible array data block")
}

/// Compute the on-disk size of an extensible array data block.
pub(super) fn cache_dblock_image_len(
    prefix: &ExtArrayDataBlockPrefix,
    payload_len: usize,
) -> Result<usize> {
    prefix
        .prefix_size
        .checked_add(payload_len)
        .and_then(|value| value.checked_add(4))
        .ok_or_else(|| {
            Error::InvalidFormat("extensible array data block image length overflow".into())
        })
}

/// Serialize a dirty data block to its on-disk image (prefix + payload + checksum).
pub(super) fn cache_dblock_serialize_into(
    prefix: &[u8],
    payload: &[u8],
    out: &mut Vec<u8>,
) -> Result<()> {
    let image_len = prefix
        .len()
        .checked_add(payload.len())
        .and_then(|value| value.checked_add(4))
        .ok_or_else(|| {
            Error::InvalidFormat("extensible array data block image length overflow".into())
        })?;
    let mut image = Vec::new();
    image.try_reserve_exact(image_len).map_err(|_| {
        Error::InvalidFormat("extensible array data block image allocation failed".into())
    })?;
    image.extend_from_slice(prefix);
    image.extend_from_slice(payload);
    let checksum = crate::format::checksum::checksum_metadata(&image);
    image.extend_from_slice(&checksum.to_le_bytes());
    *out = image;
    Ok(())
}

/// Handle metadata-cache action notifications for a data block.
pub(super) fn cache_dblock_notify(_prefix: &ExtArrayDataBlockPrefix) {}

/// Destroy/release an in-core representation of a data block.
pub(super) fn cache_dblock_free_icr(_prefix: ExtArrayDataBlockPrefix) {}

/// Report the file-space size of a data block to the metadata cache.
pub(super) fn cache_dblock_fsf_size(prefix: &ExtArrayDataBlockPrefix) -> usize {
    prefix.prefix_size
}

/// Initial number of bytes the metadata cache must read for a data block page.
pub(super) fn cache_dblk_page_get_initial_load_size() -> usize {
    4
}

/// Verify the trailing checksum of a data-block-page image.
pub(super) fn cache_dblk_page_verify_chksum(data: &[u8]) -> Result<()> {
    verify_trailing_checksum(data, "extensible array data block page")
}

/// Compute the on-disk size of a data-block page.
pub(super) fn cache_dblk_page_image_len(payload_len: usize) -> Result<usize> {
    payload_len.checked_add(4).ok_or_else(|| {
        Error::InvalidFormat("extensible array data block page image length overflow".into())
    })
}

/// Verify the trailing checksum and return the page's element payload.
pub(super) fn cache_dblk_page_deserialize(payload: &[u8]) -> Result<&[u8]> {
    if payload.len() < 4 {
        return Err(Error::InvalidFormat(
            "extensible array data block page is truncated".into(),
        ));
    }
    cache_dblk_page_verify_chksum(payload)?;
    Ok(&payload[..payload.len() - 4])
}

/// Serialize a data-block page to its on-disk image (payload + checksum).
pub(super) fn cache_dblk_page_serialize_into(payload: &[u8], out: &mut Vec<u8>) -> Result<()> {
    let image_len = cache_dblk_page_image_len(payload.len())?;
    let mut image = Vec::new();
    image.try_reserve_exact(image_len).map_err(|_| {
        Error::InvalidFormat("extensible array data block page image allocation failed".into())
    })?;
    image.extend_from_slice(payload);
    let checksum = crate::format::checksum::checksum_metadata(&image);
    image.extend_from_slice(&checksum.to_le_bytes());
    *out = image;
    Ok(())
}

/// Handle metadata-cache action notifications for a data-block page.
pub(super) fn cache_dblk_page_notify(_page_index: usize) {}

/// Destroy/release an in-core representation of a data-block page.
pub(super) fn cache_dblk_page_free_icr(_payload: Vec<u8>) {}

/// Allocate a zero-filled data-block page buffer of the given size.
pub(super) fn dblk_page_alloc(size: usize) -> Vec<u8> {
    vec![0; size]
}

/// Create a new data-block page (returns the provided payload buffer).
pub(super) fn dblk_page_create(payload: Vec<u8>) -> Vec<u8> {
    payload
}

/// Protect a data-block page in the metadata cache (no-op borrow).
pub(super) fn dblk_page_protect(payload: &[u8]) -> &[u8] {
    payload
}

/// Unprotect a data-block page.
pub(super) fn dblk_page_unprotect(_payload: &[u8]) {}

/// Destroy a data-block page in memory.
pub(super) fn dblk_page_dest(_payload: Vec<u8>) {}

/// Allocate an extensible-array data-block prefix descriptor.
pub(super) fn dblock_alloc(pages: usize, prefix_size: usize) -> ExtArrayDataBlockPrefix {
    ExtArrayDataBlockPrefix { pages, prefix_size }
}

/// Compute the index of the super block that owns a data block of the given size.
pub(super) fn dblock_sblk_idx(
    header: &ExtensibleArrayHeader,
    data_block_elements: usize,
) -> Option<usize> {
    header
        .super_block_info
        .iter()
        .position(|info| info.data_block_elements == data_block_elements)
}

/// Protect an extensible array data block in the metadata cache (deserializes the prefix).
pub(super) fn dblock_protect<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header_addr: u64,
    header: &ExtensibleArrayHeader,
    data_block_addr: u64,
    data_block_elements: usize,
) -> Result<ExtArrayDataBlockPrefix> {
    decode_data_block_prefix(
        reader,
        header_addr,
        header,
        data_block_addr,
        data_block_elements,
    )
}

/// Unprotect a previously protected extensible array data block.
pub(super) fn dblock_unprotect(_prefix: ExtArrayDataBlockPrefix) {}

/// Delete a data block by zeroing the prefix sizing fields.
pub(super) fn dblock_delete(prefix: &mut ExtArrayDataBlockPrefix) {
    prefix.pages = 0;
    prefix.prefix_size = 0;
}

/// Destroy a data block in memory.
pub(super) fn dblock_dest(_prefix: ExtArrayDataBlockPrefix) {}

/// Pure prefix decode for an extensible-array data block.
pub(super) fn decode_data_block_prefix<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header_addr: u64,
    header: &ExtensibleArrayHeader,
    data_block_addr: u64,
    data_block_elements: usize,
) -> Result<ExtArrayDataBlockPrefix> {
    let pages = super::data_block_pages(header, data_block_elements);
    let prefix_payload_size =
        extensible_array_prefix_payload_size(usize::from(reader.sizeof_addr()), header)?;
    let prefix_image_size = if pages > 0 {
        super::checked_usize_add(
            prefix_payload_size,
            4,
            "extensible array data block prefix size",
        )?
    } else {
        prefix_payload_size
    };
    let mut image = vec![0; prefix_image_size];
    reader.seek(data_block_addr)?;
    reader.read_bytes_into(&mut image)?;
    let mut image_reader = image_reader(&image, reader);

    let mut magic = [0u8; 4];
    image_reader.read_bytes_into(&mut magic)?;
    if magic != *b"EADB" {
        return Err(Error::InvalidFormat(
            "invalid extensible array data block magic".into(),
        ));
    }

    let version = image_reader.read_u8()?;
    if version != 0 {
        return Err(Error::Unsupported(format!(
            "extensible array data block version {version}"
        )));
    }

    let class_id = image_reader.read_u8()?;
    if class_id != header.class_id {
        return Err(Error::InvalidFormat(
            "extensible array data block class does not match header".into(),
        ));
    }

    let owner = image_reader.read_addr()?;
    if owner != header_addr {
        return Err(Error::InvalidFormat(
            "extensible array data block owner address does not match header".into(),
        ));
    }

    let _block_offset = image_reader.read_uint(header.array_offset_size)?;
    if pages > 0 {
        verify_trailing_checksum(&image, "extensible array data block prefix")?;
    }
    Ok(ExtArrayDataBlockPrefix {
        pages,
        prefix_size: prefix_image_size,
    })
}

fn image_reader<'a, R: Read + Seek>(
    image: &'a [u8],
    source: &HdfReader<R>,
) -> HdfReader<Cursor<&'a [u8]>> {
    let mut reader = HdfReader::new(Cursor::new(image));
    reader.set_sizeof_addr(source.sizeof_addr());
    reader.set_sizeof_size(source.sizeof_size());
    reader
}

fn extensible_array_prefix_payload_size(
    sizeof_addr: usize,
    header: &ExtensibleArrayHeader,
) -> Result<usize> {
    super::checked_usize_add(
        4 + 1 + 1,
        sizeof_addr,
        "extensible array data block prefix size",
    )
    .and_then(|value| {
        super::checked_usize_add(
            value,
            usize::from(header.array_offset_size),
            "extensible array data block prefix size",
        )
    })
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

/// Drive a decoded data-block prefix to push `count` elements onto the
/// shared output vector. C-side analogue: the iteration done inside
/// `H5EA_iterate` after a page or block has been protected.
#[allow(clippy::too_many_arguments)]
pub(super) fn append_data_block_elements<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header_addr: u64,
    header: &ExtensibleArrayHeader,
    filtered: bool,
    chunk_size_len: usize,
    data_block_addr: u64,
    data_block_elements: usize,
    page_init: Option<&[u8]>,
    count: usize,
    elements: &mut Vec<FixedArrayElement>,
) -> Result<()> {
    let mut checksum_scratch = Vec::new();
    append_data_block_elements_with_scratch(
        reader,
        header_addr,
        header,
        filtered,
        chunk_size_len,
        data_block_addr,
        data_block_elements,
        page_init,
        count,
        elements,
        &mut checksum_scratch,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn append_data_block_elements_with_scratch<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header_addr: u64,
    header: &ExtensibleArrayHeader,
    filtered: bool,
    chunk_size_len: usize,
    data_block_addr: u64,
    data_block_elements: usize,
    page_init: Option<&[u8]>,
    count: usize,
    elements: &mut Vec<FixedArrayElement>,
    checksum_scratch: &mut Vec<u8>,
) -> Result<()> {
    if count == 0 {
        return Ok(());
    }
    if is_undef_addr(data_block_addr) {
        super::append_fill_elements(header, count, elements)?;
        return Ok(());
    }

    let prefix = decode_data_block_prefix(
        reader,
        header_addr,
        header,
        data_block_addr,
        data_block_elements,
    )?;
    if count > data_block_elements {
        return Err(Error::InvalidFormat(
            "extensible array data block read count exceeds data block elements".into(),
        ));
    }
    if prefix.pages == 0 {
        let prefix_payload_size =
            extensible_array_prefix_payload_size(usize::from(reader.sizeof_addr()), header)?;
        let payload_size = super::checked_usize_mul(
            data_block_elements,
            header.raw_element_size,
            "extensible array data block payload size",
        )?;
        let image_size = super::checked_usize_add(
            super::checked_usize_add(
                prefix_payload_size,
                payload_size,
                "extensible array data block image size",
            )?,
            4,
            "extensible array data block image size",
        )?;
        checksum_scratch.clear();
        checksum_scratch.resize(image_size, 0);
        reader.seek(data_block_addr)?;
        reader.read_bytes_into(checksum_scratch)?;
        verify_trailing_checksum(checksum_scratch, "extensible array data block")?;
        let payload = &checksum_scratch[prefix_payload_size..prefix_payload_size + payload_size];
        let mut image_reader = image_reader(payload, reader);
        for _ in 0..count {
            elements.push(read_element(&mut image_reader, filtered, chunk_size_len)?);
        }
    } else {
        let page_payload = super::checked_usize_mul(
            header.data_block_page_elements,
            header.raw_element_size,
            "extensible array data block page size",
        )?;
        let page_size =
            super::checked_usize_add(page_payload, 4, "extensible array data block page size")?;
        let mut remaining = count;
        for page_index in 0..prefix.pages {
            if remaining == 0 {
                break;
            }
            let page_elements = header.data_block_page_elements.min(remaining);
            let page_offset = super::checked_usize_mul(
                page_index,
                page_size,
                "extensible array data block page offset",
            )?;
            let page_addr = super::checked_u64_add(
                data_block_addr,
                super::u64_from_usize(
                    prefix.prefix_size,
                    "extensible array data block prefix size",
                )?,
                "extensible array data block page address",
            )
            .and_then(|value| {
                super::checked_u64_add(
                    value,
                    super::u64_from_usize(page_offset, "extensible array data block page offset")?,
                    "extensible array data block page address",
                )
            })?;
            let page_initialized = page_init
                .map(|bits| super::bit_is_set(bits, page_index))
                .unwrap_or(true);
            if page_initialized {
                checksum_scratch.clear();
                checksum_scratch.resize(page_size, 0);
                reader.seek(page_addr)?;
                reader.read_bytes_into(checksum_scratch)?;
                cache_dblk_page_verify_chksum(checksum_scratch)?;
                let mut page_reader = image_reader(&checksum_scratch[..page_payload], reader);
                for _ in 0..page_elements {
                    elements.push(read_element(&mut page_reader, filtered, chunk_size_len)?);
                }
            } else {
                super::append_fill_elements(header, page_elements, elements)?;
            }
            remaining -= page_elements;
        }
    }

    Ok(())
}

/// Read and validate the trailing checksum of the span starting at `start`.
fn verify_reader_checksum<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    start: u64,
    context: &str,
) -> Result<()> {
    let mut scratch = Vec::new();
    verify_reader_checksum_with_scratch(reader, start, context, &mut scratch)
}

fn verify_reader_checksum_with_scratch<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    start: u64,
    context: &str,
    scratch: &mut Vec<u8>,
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
    scratch.clear();
    scratch.resize(check_len, 0);
    reader.read_bytes_into(scratch)?;
    let computed = checksum_metadata(scratch);
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

    fn header(data_block_page_elements: usize) -> ExtensibleArrayHeader {
        ExtensibleArrayHeader {
            class_id: 1,
            raw_element_size: 8,
            index_block_elements: 0,
            max_index_set: 0,
            index_block_addr: 0,
            array_offset_size: 1,
            data_block_page_elements,
            index_block_super_blocks: 0,
            index_block_data_block_addrs: 0,
            index_block_super_block_addrs: 0,
            super_block_info: Vec::new(),
        }
    }

    fn append_checksum(bytes: &mut Vec<u8>) {
        let checksum = checksum_metadata(bytes);
        bytes.extend_from_slice(&checksum.to_le_bytes());
    }

    fn extensible_array_prefix(owner: u64) -> Vec<u8> {
        let mut bytes = b"EADB".to_vec();
        bytes.push(0);
        bytes.push(1);
        bytes.extend_from_slice(&owner.to_le_bytes());
        bytes.push(0); // block offset, array_offset_size = 1
        bytes
    }

    #[test]
    fn extensible_array_unpaginated_rejects_bad_checksum() {
        let mut bytes = extensible_array_prefix(100);
        bytes.extend_from_slice(&55u64.to_le_bytes());
        append_checksum(&mut bytes);
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;

        let mut elements = Vec::new();
        let mut reader = HdfReader::new(Cursor::new(bytes));
        let err = append_data_block_elements(
            &mut reader,
            100,
            &header(4),
            false,
            0,
            0,
            1,
            None,
            1,
            &mut elements,
        )
        .expect_err("bad extensible array data block checksum should fail");
        assert!(err.to_string().contains("checksum"));
    }

    #[test]
    fn extensible_array_paginated_rejects_bad_prefix_and_page_checksums() {
        let mut bytes = extensible_array_prefix(100);
        append_checksum(&mut bytes);
        bytes.extend_from_slice(&55u64.to_le_bytes());
        append_checksum(&mut bytes);

        let mut bad_prefix = bytes.clone();
        bad_prefix[14] ^= 0xff;
        let mut elements = Vec::new();
        let mut reader = HdfReader::new(Cursor::new(bad_prefix));
        let err = append_data_block_elements(
            &mut reader,
            100,
            &header(1),
            false,
            0,
            0,
            2,
            Some(&[0x80]),
            1,
            &mut elements,
        )
        .expect_err("bad extensible array data block prefix checksum should fail");
        assert!(err.to_string().contains("checksum"));

        let mut bad_page = bytes;
        let last = bad_page.len() - 1;
        bad_page[last] ^= 0xff;
        let mut elements = Vec::new();
        let mut reader = HdfReader::new(Cursor::new(bad_page));
        let err = append_data_block_elements(
            &mut reader,
            100,
            &header(1),
            false,
            0,
            0,
            2,
            Some(&[0x80]),
            1,
            &mut elements,
        )
        .expect_err("bad extensible array data block page checksum should fail");
        assert!(err.to_string().contains("checksum"));
    }

    #[test]
    fn extensible_array_page_cache_serializes_and_validates_checksum() {
        let payload = 55u64.to_le_bytes();
        let mut image = b"stale".to_vec();
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

    #[test]
    fn extensible_array_paginated_rejects_read_count_past_block_elements() {
        let mut bytes = extensible_array_prefix(100);
        append_checksum(&mut bytes);

        let mut elements = vec![FixedArrayElement {
            addr: 7,
            nbytes: None,
            filter_mask: 0,
        }];
        let mut reader = HdfReader::new(Cursor::new(bytes));
        let err = append_data_block_elements(
            &mut reader,
            100,
            &header(1),
            false,
            0,
            0,
            2,
            Some(&[0x80]),
            3,
            &mut elements,
        )
        .expect_err("paginated data block over-read should fail");
        assert!(err
            .to_string()
            .contains("read count exceeds data block elements"));
        assert_eq!(elements.len(), 1);
        assert_eq!(elements[0].addr, 7);
    }
}
