//! Fixed array — top-level public API. Mirrors libhdf5's `H5FA.c` (the
//! file-spanning entry points). Per-component code lives in sibling
//! modules: `hdr` (header decode + checksum), `dblock` (data-block
//! decode + element iteration; absorbs `dblkpage` because the Rust port
//! doesn't model pages as a separate cache entry).

mod dblock;
mod hdr;

use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::io::reader::{is_undef_addr, HdfReader, UNDEF_ADDR};

use hdr::read_header;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedArrayElement {
    pub addr: u64,
    pub nbytes: Option<u64>,
    pub filter_mask: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FixedArrayStats {
    pub elements: usize,
    pub allocated_elements: usize,
    pub flush_dependencies: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedArrayElementLocation {
    pub element_addr: u64,
    pub checksum_start: u64,
    pub checksum_len: usize,
    pub checksum_addr: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedArray {
    elements: Vec<FixedArrayElement>,
    flush_dependencies: usize,
    deleted: bool,
}

impl FixedArray {
    pub fn new() -> Self {
        Self {
            elements: Vec::new(),
            flush_dependencies: 0,
            deleted: false,
        }
    }

    pub fn create(size: usize) -> Self {
        let mut elements = Vec::with_capacity(size);
        append_fill_elements(size, &mut elements);
        Self {
            elements,
            flush_dependencies: 0,
            deleted: false,
        }
    }

    pub fn open(elements: Vec<FixedArrayElement>) -> Self {
        Self {
            elements,
            flush_dependencies: 0,
            deleted: false,
        }
    }

    pub fn get_nelmts(&self) -> usize {
        self.elements.len()
    }

    pub fn get_addr(&self, index: usize) -> Result<u64> {
        Ok(self
            .elements
            .get(index)
            .ok_or_else(|| {
                Error::InvalidFormat(format!("fixed array index {index} out of bounds"))
            })?
            .addr)
    }

    pub fn set(&mut self, index: usize, element: FixedArrayElement) -> Result<()> {
        if self.deleted {
            return Err(Error::InvalidFormat(
                "cannot set element in deleted fixed array".into(),
            ));
        }
        let slot = self.elements.get_mut(index).ok_or_else(|| {
            Error::InvalidFormat(format!("fixed array index {index} out of bounds"))
        })?;
        *slot = element;
        Ok(())
    }

    pub fn close(self) {}

    pub fn delete(&mut self) {
        self.elements.clear();
        self.deleted = true;
    }

    pub fn depend(&mut self) {
        self.flush_dependencies = self.flush_dependencies.saturating_add(1);
    }

    pub fn patch_file(&mut self, index: usize, addr: u64) -> Result<()> {
        let slot = self.elements.get_mut(index).ok_or_else(|| {
            Error::InvalidFormat(format!("fixed array index {index} out of bounds"))
        })?;
        slot.addr = addr;
        Ok(())
    }

    pub fn get_stats(&self) -> FixedArrayStats {
        FixedArrayStats {
            elements: self.elements.len(),
            allocated_elements: self.elements.capacity(),
            flush_dependencies: self.flush_dependencies,
        }
    }

    pub fn create_flush_depend(&mut self) {
        self.depend();
    }

    pub fn destroy_flush_depend(&mut self) -> Result<()> {
        if self.flush_dependencies == 0 {
            return Err(Error::InvalidFormat(
                "fixed array flush dependency underflow".into(),
            ));
        }
        self.flush_dependencies -= 1;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedArrayCreateParams {
    pub raw_element_size: usize,
    pub max_page_elements_bits: u8,
    pub elements: u64,
}

pub fn test_crt_context() -> FixedArrayCreateParams {
    FixedArrayCreateParams {
        raw_element_size: 8,
        max_page_elements_bits: 8,
        elements: 0,
    }
}

pub fn test_dst_context(_params: FixedArrayCreateParams) {}

pub fn test_fill(count: usize) -> Vec<FixedArrayElement> {
    let mut elements = Vec::with_capacity(count);
    append_fill_elements(count, &mut elements);
    elements
}

pub fn test_encode(params: &FixedArrayCreateParams) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    out.extend_from_slice(
        &u64_from_usize(params.raw_element_size, "fixed array raw element size")?.to_le_bytes(),
    );
    out.push(params.max_page_elements_bits);
    out.extend_from_slice(&params.elements.to_le_bytes());
    Ok(out)
}

pub fn test_decode(data: &[u8]) -> Result<FixedArrayCreateParams> {
    if data.len() < 17 {
        return Err(Error::InvalidFormat(
            "fixed array test params are truncated".into(),
        ));
    }
    let raw_element_size = usize_from_u64(
        read_u64_le_at(data, 0, "fixed array raw element size")?,
        "fixed array raw element size",
    )?;
    let max_page_elements_bits = data[8];
    let elements = read_u64_le_at(data, 9, "fixed array element count")?;
    Ok(FixedArrayCreateParams {
        raw_element_size,
        max_page_elements_bits,
        elements,
    })
}

fn checked_window<'a>(data: &'a [u8], pos: usize, len: usize, context: &str) -> Result<&'a [u8]> {
    let end = pos
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))?;
    data.get(pos..end)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))
}

fn read_u64_le_at(data: &[u8], pos: usize, context: &str) -> Result<u64> {
    let bytes = checked_window(data, pos, 8, context)?;
    Ok(u64::from_le_bytes(bytes.try_into().map_err(|_| {
        Error::InvalidFormat(format!("{context} is truncated"))
    })?))
}

pub fn test_debug(params: &FixedArrayCreateParams) -> String {
    format!(
        "FixedArrayCreateParams(raw_element_size={}, max_page_elements_bits={}, elements={})",
        params.raw_element_size, params.max_page_elements_bits, params.elements
    )
}

pub fn test_crt_dbg_context() -> FixedArrayCreateParams {
    test_crt_context()
}

pub fn get_cparam_test(params: &FixedArrayCreateParams) -> FixedArrayCreateParams {
    params.clone()
}

pub fn cmp_cparam_test(lhs: &FixedArrayCreateParams, rhs: &FixedArrayCreateParams) -> bool {
    lhs == rhs
}

pub fn read_fixed_array_chunks<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    addr: u64,
    filtered: bool,
    chunk_size_len: usize,
) -> Result<Vec<FixedArrayElement>> {
    let header = read_header(reader, addr)?;
    let expected_class = if filtered { 1 } else { 0 };
    if header.class_id != expected_class {
        return Err(Error::InvalidFormat(format!(
            "fixed array class {} does not match filtered={filtered}",
            header.class_id
        )));
    }

    if is_undef_addr(header.data_block_addr) {
        return Ok(Vec::new());
    }

    dblock::read_data_block(reader, addr, &header, filtered, chunk_size_len)
}

/// Locate the file offset of an existing fixed-array element.
///
/// The returned offset points at the element address field. For filtered chunk
/// arrays, the filtered-size and filter-mask fields follow immediately.
pub fn locate_fixed_array_element<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    addr: u64,
    filtered: bool,
    chunk_size_len: usize,
    element_index: usize,
) -> Result<u64> {
    Ok(locate_fixed_array_element_with_checksum(
        reader,
        addr,
        filtered,
        chunk_size_len,
        element_index,
    )?
    .element_addr)
}

pub fn locate_fixed_array_element_with_checksum<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    addr: u64,
    filtered: bool,
    chunk_size_len: usize,
    element_index: usize,
) -> Result<FixedArrayElementLocation> {
    let header = read_header(reader, addr)?;
    let expected_class = if filtered { 1 } else { 0 };
    if header.class_id != expected_class {
        return Err(Error::InvalidFormat(format!(
            "fixed array class {} does not match filtered={filtered}",
            header.class_id
        )));
    }
    let element_count = usize_from_u64(header.elements, "fixed array element count")?;
    if element_index >= element_count {
        return Err(Error::InvalidFormat(format!(
            "fixed array element index {element_index} out of bounds for {} elements",
            header.elements
        )));
    }
    if is_undef_addr(header.data_block_addr) {
        return Err(Error::Unsupported(
            "cannot update fixed-array chunk entry without a data block".into(),
        ));
    }

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

    let page_elements = 1usize
        .checked_shl(u32::from(header.max_page_elements_bits))
        .ok_or_else(|| Error::InvalidFormat("fixed array page size overflow".into()))?;
    let data_prefix_size = checked_usize_add(
        4 + 1 + 1,
        usize::from(reader.sizeof_addr()),
        "fixed array data block prefix size",
    )?;

    if element_count > page_elements {
        reader.seek(checked_u64_add(
            header.data_block_addr,
            u64_from_usize(data_prefix_size, "fixed array data block prefix size")?,
            "fixed array data block page-init address",
        )?)?;
        let pages = element_count.div_ceil(page_elements);
        let page_init_size = pages.div_ceil(8);
        let page_init = reader.read_bytes(page_init_size)?;
        let page_index = element_index / page_elements;
        if !bit_is_set(&page_init, page_index) {
            return Err(Error::Unsupported(
                "cannot update uninitialized fixed-array chunk page".into(),
            ));
        }
        let prefix_size = checked_usize_add(
            data_prefix_size,
            page_init_size,
            "fixed array data block prefix size",
        )
        .and_then(|value| checked_usize_add(value, 4, "fixed array data block prefix size"))?;
        let page_payload = checked_usize_mul(
            page_elements,
            header.raw_element_size,
            "fixed array data block page size",
        )?;
        let page_size = checked_usize_add(page_payload, 4, "fixed array data block page size")?;
        let within_page = element_index % page_elements;
        let page_offset =
            checked_usize_mul(page_index, page_size, "fixed array data block page offset")?;
        let element_offset = checked_usize_mul(
            within_page,
            header.raw_element_size,
            "fixed array data block element offset",
        )?;
        let page_addr_offset = checked_usize_add(
            prefix_size,
            page_offset,
            "fixed array data block page offset",
        )?;
        let offset = checked_usize_add(
            page_addr_offset,
            element_offset,
            "fixed array data block element offset",
        )?;
        let element_addr = checked_u64_add(
            header.data_block_addr,
            u64_from_usize(offset, "fixed array data block element offset")?,
            "fixed array data block element address",
        )?;
        let page_addr = checked_u64_add(
            header.data_block_addr,
            u64_from_usize(page_addr_offset, "fixed array data block page offset")?,
            "fixed array data block page address",
        )?;
        let checksum_addr = checked_u64_add(
            page_addr,
            u64_from_usize(page_payload, "fixed array data block page checksum offset")?,
            "fixed array data block page checksum address",
        )?;
        Ok(FixedArrayElementLocation {
            element_addr,
            checksum_start: page_addr,
            checksum_len: page_payload,
            checksum_addr,
        })
    } else {
        let element_offset = checked_usize_mul(
            element_index,
            header.raw_element_size,
            "fixed array data block element offset",
        )?;
        let offset = checked_usize_add(
            data_prefix_size,
            element_offset,
            "fixed array data block element offset",
        )?;
        let element_addr = checked_u64_add(
            header.data_block_addr,
            u64_from_usize(offset, "fixed array data block element offset")?,
            "fixed array data block element address",
        )?;
        let payload_len = checked_usize_mul(
            element_count,
            header.raw_element_size,
            "fixed array data block payload size",
        )?;
        let checksum_offset = checked_usize_add(
            data_prefix_size,
            payload_len,
            "fixed array data block checksum offset",
        )?;
        let checksum_addr = checked_u64_add(
            header.data_block_addr,
            u64_from_usize(checksum_offset, "fixed array data block checksum offset")?,
            "fixed array data block checksum address",
        )?;
        Ok(FixedArrayElementLocation {
            element_addr,
            checksum_start: header.data_block_addr,
            checksum_len: checksum_offset,
            checksum_addr,
        })
    }
}

// ---------------------------------------------------------------------------
// Internal helpers — shared across `hdr` and `dblock`. Mirrors libhdf5's
// `H5FAint.c` (the package-internal helper file).
// ---------------------------------------------------------------------------

pub(super) fn append_fill_elements(count: usize, elements: &mut Vec<FixedArrayElement>) {
    for _ in 0..count {
        elements.push(FixedArrayElement {
            addr: UNDEF_ADDR,
            nbytes: None,
            filter_mask: 0,
        });
    }
}

pub(super) fn usize_from_u64(value: u64, context: &str) -> Result<usize> {
    usize::try_from(value)
        .map_err(|_| Error::InvalidFormat(format!("{context} does not fit in usize")))
}

pub(super) fn u64_from_usize(value: usize, context: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| Error::InvalidFormat(format!("{context} does not fit in u64")))
}

pub(super) fn checked_usize_add(lhs: usize, rhs: usize, context: &str) -> Result<usize> {
    lhs.checked_add(rhs)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} overflow")))
}

pub(super) fn checked_usize_mul(lhs: usize, rhs: usize, context: &str) -> Result<usize> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} overflow")))
}

pub(super) fn checked_u64_add(lhs: u64, rhs: u64, context: &str) -> Result<u64> {
    lhs.checked_add(rhs)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} overflow")))
}

pub(super) fn bit_is_set(bytes: &[u8], bit: usize) -> bool {
    bytes
        .get(bit / 8)
        .map(|byte| (byte & (0x80 >> (bit % 8))) != 0)
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::{checked_u64_add, checked_usize_mul};

    #[test]
    fn fixed_array_checked_helpers_reject_overflow() {
        assert!(checked_usize_mul(usize::MAX, 2, "fixed array size").is_err());
        assert!(checked_u64_add(u64::MAX, 1, "fixed array address").is_err());
    }
}
