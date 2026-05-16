//! Extensible array — top-level public API. Mirrors libhdf5's `H5EA.c`
//! (the file-spanning entry points). Per-component code lives in sibling
//! modules: `hdr` (header decode + checksum + super-block-info build),
//! `iblock` (index block decode + spillover descent), `sblock` (super
//! block decode + walk), `dblock` (data block decode + element walk;
//! absorbs `dblkpage` because the Rust port doesn't model pages as a
//! separate cache entry).

mod dblock;
pub(crate) mod hdr;
mod iblock;
mod sblock;

use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::io::reader::{is_undef_addr, HdfReader, UNDEF_ADDR};

use super::fixed_array;
use super::fixed_array::FixedArrayElement;
use hdr::{read_header, ExtensibleArrayHeader};
use iblock::read_index_block;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExtensibleArrayStats {
    pub elements: usize,
    pub allocated_elements: usize,
    pub flush_dependencies: usize,
}

#[derive(Debug, Clone)]
pub struct ExtensibleArray {
    elements: Vec<FixedArrayElement>,
    flush_dependencies: usize,
    deleted: bool,
}

impl ExtensibleArray {
    /// Allocate and initialize an empty extensible array wrapper.
    pub fn new() -> Self {
        Self {
            elements: Vec::new(),
            flush_dependencies: 0,
            deleted: false,
        }
    }

    /// Create a new empty extensible array with the requested element capacity.
    pub fn create(capacity: usize) -> Self {
        Self {
            elements: Vec::with_capacity(capacity),
            flush_dependencies: 0,
            deleted: false,
        }
    }

    /// Open an existing extensible array around already-loaded elements.
    pub fn open(elements: Vec<FixedArrayElement>) -> Self {
        Self {
            elements,
            flush_dependencies: 0,
            deleted: false,
        }
    }

    /// Query the current number of elements in the array.
    pub fn get_nelmts(&self) -> usize {
        self.elements.len()
    }

    /// Retrieve a reference to the element at `index`.
    pub fn lookup_elmt(&self, index: usize) -> Result<&FixedArrayElement> {
        self.elements.get(index).ok_or_else(|| {
            Error::InvalidFormat(format!("extensible array index {index} out of bounds"))
        })
    }

    /// Get a copy of the element at `index`.
    pub fn get(&self, index: usize) -> Result<FixedArrayElement> {
        Ok(self.lookup_elmt(index)?.clone())
    }

    /// Set an element of the extensible array, growing it by one if `index`
    /// equals the current length. Sparse sets (gaps) are rejected.
    pub fn set(&mut self, index: usize, element: FixedArrayElement) -> Result<()> {
        if self.deleted {
            return Err(Error::InvalidFormat(
                "cannot set element in deleted extensible array".into(),
            ));
        }
        if index > self.elements.len() {
            return Err(Error::InvalidFormat(format!(
                "extensible array sparse set at index {index}"
            )));
        }
        if index == self.elements.len() {
            self.elements.push(element);
        } else {
            self.elements[index] = element;
        }
        Ok(())
    }

    /// Make a child flush dependency between this array and another data structure.
    /// Saturates on overflow.
    pub fn depend(&mut self) {
        let _ = self.depend_checked();
    }

    /// Checked variant of `depend` that returns an error on overflow.
    pub fn depend_checked(&mut self) -> Result<()> {
        self.flush_dependencies = self.flush_dependencies.checked_add(1).ok_or_else(|| {
            Error::InvalidFormat("extensible array flush dependency overflow".into())
        })?;
        Ok(())
    }

    /// Close the extensible array, consuming the handle.
    pub fn close(self) {}

    /// Delete the extensible array, clearing its elements and marking it deleted.
    pub fn delete(&mut self) {
        self.elements.clear();
        self.deleted = true;
    }

    /// Patch the on-disk address recorded for the element at `index`.
    pub fn patch_file(&mut self, index: usize, addr: u64) -> Result<()> {
        let element = self.elements.get_mut(index).ok_or_else(|| {
            Error::InvalidFormat(format!("extensible array index {index} out of bounds"))
        })?;
        element.addr = addr;
        Ok(())
    }

    /// Query the metadata statistics of the array.
    pub fn get_stats(&self) -> ExtensibleArrayStats {
        ExtensibleArrayStats {
            elements: self.elements.len(),
            allocated_elements: self.elements.capacity(),
            flush_dependencies: self.flush_dependencies,
        }
    }

    /// Create a flush dependency between two data-structure components.
    pub fn create_flush_depend(&mut self) {
        self.depend();
    }

    /// Checked variant of `create_flush_depend`.
    pub fn create_flush_depend_checked(&mut self) -> Result<()> {
        self.depend_checked()
    }

    /// Destroy a flush dependency between two data-structure components.
    pub fn destroy_flush_depend(&mut self) -> Result<()> {
        if self.flush_dependencies == 0 {
            return Err(Error::InvalidFormat(
                "extensible array flush dependency underflow".into(),
            ));
        }
        self.flush_dependencies -= 1;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtensibleArrayCreateParams {
    pub raw_element_size: usize,
    pub index_block_elements: u8,
    pub data_block_min_elements: usize,
    pub max_index_set: u64,
}

/// Destroy a client callback context (no-op for the test driver).
pub fn test_dst_context() {}

/// Encode the test creation parameters into a stable little-endian byte stream.
pub fn test_encode(params: &ExtensibleArrayCreateParams) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    out.extend_from_slice(
        &u64_from_usize(params.raw_element_size, "extensible array raw element size")?
            .to_le_bytes(),
    );
    out.push(params.index_block_elements);
    out.extend_from_slice(
        &u64_from_usize(
            params.data_block_min_elements,
            "extensible array data block minimum elements",
        )?
        .to_le_bytes(),
    );
    out.extend_from_slice(&params.max_index_set.to_le_bytes());
    Ok(out)
}

/// Display the test creation parameters for debugging.
pub fn test_debug(params: &ExtensibleArrayCreateParams) -> String {
    format!(
        "ExtensibleArrayCreateParams(raw_element_size={}, index_block_elements={}, data_block_min_elements={}, max_index_set={})",
        params.raw_element_size,
        params.index_block_elements,
        params.data_block_min_elements,
        params.max_index_set
    )
}

/// Create a debugging callback context with default extensible array parameters.
pub fn test_crt_dbg_context() -> ExtensibleArrayCreateParams {
    ExtensibleArrayCreateParams {
        raw_element_size: 8,
        index_block_elements: 4,
        data_block_min_elements: 4,
        max_index_set: 0,
    }
}

/// Destroy a debugging callback context.
pub fn test_dst_dbg_context(_params: ExtensibleArrayCreateParams) {}

/// Retrieve the parameters used to create the extensible array.
pub fn get_cparam_test(params: &ExtensibleArrayCreateParams) -> ExtensibleArrayCreateParams {
    params.clone()
}

/// Compare two sets of extensible-array creation parameters for equality.
pub fn cmp_cparam_test(
    lhs: &ExtensibleArrayCreateParams,
    rhs: &ExtensibleArrayCreateParams,
) -> bool {
    lhs == rhs
}

/// Iterate over the elements of an extensible array, returning the decoded chunk records.
pub fn read_extensible_array_chunks<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    addr: u64,
    filtered: bool,
    chunk_size_len: usize,
) -> Result<Vec<FixedArrayElement>> {
    let header = read_header(reader, addr)?;
    let expected_class = if filtered { 1 } else { 0 };
    if header.class_id != expected_class {
        return Err(Error::InvalidFormat(format!(
            "extensible array class {} does not match filtered={filtered}",
            header.class_id
        )));
    }

    if is_undef_addr(header.index_block_addr) {
        return Ok(Vec::new());
    }
    read_index_block(reader, addr, &header, filtered, chunk_size_len)
}

// ---------------------------------------------------------------------------
// Internal helpers shared across hdr/iblock/sblock/dblock. Mirrors libhdf5's
// `H5EAint.c` (the package-internal helper file).
// `unreachable_pub` is allowed because submodules need `pub(super)`
// visibility on these, even though the helpers operate on
// extensible-array-private types.
// ---------------------------------------------------------------------------

/// Append up to `count` fill elements (undefined-address sentinels) to the
/// running element list, capped at the header's `max_index_set`.
#[allow(private_interfaces)]
pub(super) fn append_fill_elements(
    header: &ExtensibleArrayHeader,
    count: usize,
    elements: &mut Vec<FixedArrayElement>,
) -> Result<()> {
    let max_index_set = usize_from_u64(header.max_index_set, "extensible array max index")?;
    let remaining = max_index_set.checked_sub(elements.len()).ok_or_else(|| {
        Error::InvalidFormat("extensible array element count exceeds max index set".into())
    })?;
    for _ in 0..count.min(remaining) {
        elements.push(FixedArrayElement {
            addr: UNDEF_ADDR,
            nbytes: None,
            filter_mask: 0,
        });
    }
    Ok(())
}

/// Compute the number of pages a data block of `data_block_elements` is split into,
/// or `0` if the block is unpaginated.
#[allow(private_interfaces)]
pub(super) fn data_block_pages(
    header: &ExtensibleArrayHeader,
    data_block_elements: usize,
) -> usize {
    if data_block_elements > header.data_block_page_elements {
        data_block_elements / header.data_block_page_elements
    } else {
        0
    }
}

/// Test whether the given bit (MSB-first within a byte) is set in the bitmap.
pub(super) fn bit_is_set(bytes: &[u8], bit: usize) -> bool {
    bytes
        .get(bit / 8)
        .map(|byte| (byte & (0x80 >> (bit % 8))) != 0)
        .unwrap_or(false)
}

/// Return `log2(value)`, requiring `value` to be a positive power of two.
pub(super) fn log2_power2(value: u64) -> Result<usize> {
    if value == 0 || !value.is_power_of_two() {
        return Err(Error::InvalidFormat(format!(
            "extensible array value {value} is not a power of two"
        )));
    }
    usize::try_from(value.trailing_zeros())
        .map_err(|_| Error::InvalidFormat("extensible array log2 value is too large".into()))
}

/// Convert `u64` to `usize` with a contextual error on overflow.
pub(super) fn usize_from_u64(value: u64, context: &str) -> Result<usize> {
    usize::try_from(value)
        .map_err(|_| Error::InvalidFormat(format!("{context} does not fit in usize")))
}

/// Add two `usize` values, returning an error on overflow.
pub(super) fn checked_usize_add(lhs: usize, rhs: usize, context: &str) -> Result<usize> {
    lhs.checked_add(rhs)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} overflow")))
}

/// Multiply two `usize` values, returning an error on overflow.
pub(super) fn checked_usize_mul(lhs: usize, rhs: usize, context: &str) -> Result<usize> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} overflow")))
}

/// Add two `u64` values, returning an error on overflow.
pub(super) fn checked_u64_add(lhs: u64, rhs: u64, context: &str) -> Result<u64> {
    lhs.checked_add(rhs)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} overflow")))
}

/// Convert `usize` to `u64` with a contextual error on overflow.
pub(super) fn u64_from_usize(value: usize, context: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| Error::InvalidFormat(format!("{context} does not fit in u64")))
}

#[cfg(test)]
mod tests {
    use super::{
        append_fill_elements, checked_u64_add, checked_usize_mul, u64_from_usize, ExtensibleArray,
        ExtensibleArrayHeader,
    };

    #[test]
    fn checked_helpers_reject_overflow() {
        assert!(checked_usize_mul(usize::MAX, 2, "ea size").is_err());
        assert!(checked_u64_add(u64::MAX, 1, "ea addr").is_err());

        let mut array = ExtensibleArray::create(0);
        array.flush_dependencies = usize::MAX;
        assert!(array.depend_checked().is_err());
        assert!(array.create_flush_depend_checked().is_err());
    }

    #[test]
    fn usize_to_u64_helper_accepts_normal_values() {
        assert_eq!(u64_from_usize(42, "ea value").unwrap(), 42);
    }

    #[test]
    fn append_fill_elements_rejects_overfilled_array() {
        let header = ExtensibleArrayHeader {
            class_id: 1,
            raw_element_size: 8,
            index_block_elements: 1,
            data_block_page_elements: 0,
            max_index_set: 1,
            index_block_addr: 0,
            array_offset_size: 1,
            index_block_super_blocks: 0,
            index_block_data_block_addrs: 0,
            index_block_super_block_addrs: 0,
            super_block_info: Vec::new(),
        };
        let mut elements = vec![
            super::fixed_array::FixedArrayElement {
                addr: 0,
                nbytes: None,
                filter_mask: 0,
            },
            super::fixed_array::FixedArrayElement {
                addr: 0,
                nbytes: None,
                filter_mask: 0,
            },
        ];

        let err = append_fill_elements(&header, 1, &mut elements).unwrap_err();
        assert!(
            err.to_string()
                .contains("extensible array element count exceeds max index set"),
            "unexpected error: {err}"
        );
    }
}
