use std::{
    fmt,
    io::{Read, Seek},
};

use crate::error::{Error, Result};
use crate::io::reader::{HdfReader, UNDEF_ADDR};

/// Local heap magic: "HEAP"
const HEAP_MAGIC: [u8; 4] = [b'H', b'E', b'A', b'P'];
const MAX_LOCAL_HEAP_BYTES: usize = 4 * 1024 * 1024 * 1024;

/// Parsed local-heap prefix (the on-disk header that precedes the data
/// segment). Mirrors the work `H5HL__cache_prefix_deserialize` does in
/// libhdf5: header bytes in, struct out, no data-segment I/O.
#[derive(Debug, Clone, Copy)]
pub struct LocalHeapPrefix {
    pub data_size: u64,
    pub free_list_offset: u64,
    pub data_addr: u64,
}

/// A local heap stores variable-length strings (link names) for v1 groups.
#[derive(Debug, Clone)]
pub struct LocalHeap {
    /// The raw data content of the heap.
    pub data: Vec<u8>,
}

impl LocalHeap {
    /// Create a new, empty in-memory local heap.
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    /// Create a new local heap with an initial data-area size hint.
    /// Mirrors libhdf5's `H5HL_create`.
    pub fn create(size_hint: usize) -> Self {
        Self {
            data: Vec::with_capacity(size_hint),
        }
    }

    /// Pin a heap for use. Mirrors libhdf5's `H5HL_protect`
    /// (`H5AC_protect` wrapper); the Rust port owns the value outright.
    pub fn protect(heap: Self) -> Self {
        heap
    }

    /// Release the protection acquired by `protect`. Mirrors
    /// `H5HL_unprotect`; nothing to do for the owned Rust value.
    pub fn unprotect(_heap: Self) {}

    /// Trim trailing free space (zero bytes) from the heap's data area.
    /// Mirrors `H5HL__remove_free`.
    pub fn remove_free(&mut self) {
        while self.data.last().copied() == Some(0) {
            self.data.pop();
        }
    }

    /// Mark the heap as dirty. Mirrors `H5HL__dirty`; no-op since the
    /// Rust port doesn't use the metadata cache's dirty tracking.
    pub fn dirty(&mut self) {}

    /// Insert a new item into the heap, returning the offset of the new
    /// item within the heap data. Mirrors libhdf5's `H5HL_insert`.
    pub fn insert(&mut self, value: &[u8]) -> Result<usize> {
        let offset = self.data.len();
        self.data.extend_from_slice(value);
        if !value.ends_with(&[0]) {
            self.data.push(0);
        }
        Ok(offset)
    }

    /// Remove a (null-terminated) string from the heap by zero-filling
    /// its bytes. Mirrors libhdf5's `H5HL_remove`.
    pub fn remove(&mut self, offset: usize) -> Result<()> {
        let end = self.string_end(offset)?;
        for byte in &mut self.data[offset..=end] {
            *byte = 0;
        }
        Ok(())
    }

    /// Delete the heap's contents. Mirrors libhdf5's `H5HL_delete`,
    /// which frees the on-disk heap; for the in-memory port we just
    /// clear the data buffer.
    pub fn delete(&mut self) {
        self.data.clear();
    }

    /// Current size of the heap's data block. Mirrors
    /// `H5HL_heap_get_size`.
    pub fn heap_get_size(&self) -> usize {
        self.data.len()
    }

    /// Retrieve the current size of the heap. Mirrors `H5HL_get_size`.
    pub fn get_size(&self) -> usize {
        self.heap_get_size()
    }

    /// Compute the size in bytes of this `LocalHeap`. Mirrors
    /// `H5HL_heapsize`.
    pub fn heapsize(&self) -> usize {
        self.heap_get_size()
    }

    /// Create a new local heap prefix object. Mirrors `H5HL__prfx_new`.
    pub fn prfx_new(data_size: u64, free_list_offset: u64, data_addr: u64) -> LocalHeapPrefix {
        LocalHeapPrefix {
            data_size,
            free_list_offset,
            data_addr,
        }
    }

    /// Destroy a prefix object. Mirrors `H5HL__prfx_dest`; the Rust
    /// `Copy` value drops on its own.
    pub fn prfx_dest(_prefix: LocalHeapPrefix) {}

    /// Decode a local heap's header at `addr`. Mirrors
    /// `H5HL__hdr_deserialize`.
    pub fn hdr_deserialize<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        addr: u64,
    ) -> Result<LocalHeapPrefix> {
        Self::decode_prefix(reader, addr)
    }

    /// Deserialize the free list for a heap data block. Mirrors
    /// `H5HL__fl_deserialize`: parses pairs of (offset, size) records.
    pub fn fl_entries(data: &[u8]) -> Result<impl Iterator<Item = (u64, u64)> + '_> {
        if data.len() % 16 != 0 {
            return Err(Error::InvalidFormat(
                "local heap free-list image has a partial trailing record".into(),
            ));
        }
        Ok(data.chunks_exact(16).map(|record| {
            let offset = read_le_u64_from_record(record, 0)
                .expect("validated local heap free-list record offset");
            let size = read_le_u64_from_record(record, 8)
                .expect("validated local heap free-list record size");
            (offset, size)
        }))
    }

    /// Deserialize the free list for a heap data block into `out`.
    ///
    /// The output buffer is cleared before entries are written.
    pub fn fl_deserialize_into(data: &[u8], out: &mut Vec<(u64, u64)>) -> Result<()> {
        let entries = Self::fl_entries(data)?;
        out.clear();
        out.extend(entries);
        Ok(())
    }

    /// Deserialize the free list for a heap data block into an owned vector.
    pub fn fl_deserialize(data: &[u8]) -> Result<Vec<(u64, u64)>> {
        let entries = Self::fl_entries(data)?;
        let mut out = Vec::with_capacity(data.len() / 16);
        out.extend(entries);
        Ok(out)
    }

    /// Serialize the free list for a heap data block into `out`.
    ///
    /// The output buffer is cleared before the image is written.
    pub fn fl_serialize_into(entries: &[(u64, u64)], out: &mut Vec<u8>) -> Result<()> {
        let capacity = entries.len().checked_mul(16).ok_or_else(|| {
            Error::InvalidFormat("local heap free-list image length overflow".into())
        })?;
        out.clear();
        out.reserve_exact(capacity);
        for &(offset, size) in entries {
            out.extend_from_slice(&offset.to_le_bytes());
            out.extend_from_slice(&size.to_le_bytes());
        }
        Ok(())
    }

    /// Serialize the free list for a heap data block to an owned image.
    pub fn fl_serialize(entries: &[(u64, u64)]) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        Self::fl_serialize_into(entries, &mut out)?;
        Ok(out)
    }

    /// Initial buffer size the metadata cache should speculatively read
    /// for a local heap prefix. Mirrors
    /// `H5HL__cache_prefix_get_initial_load_size`.
    pub fn cache_prefix_get_initial_load_size() -> usize {
        8
    }

    /// Final buffer size for a local heap prefix once the header has been
    /// inspected. Mirrors `H5HL__cache_prefix_get_final_load_size`.
    pub fn cache_prefix_get_final_load_size(addr_size: usize, length_size: usize) -> Result<usize> {
        Self::cache_prefix_image_len(addr_size, length_size)
    }

    /// On-disk image length of a local heap prefix given the file's
    /// address and length field widths. Mirrors
    /// `H5HL__cache_prefix_image_len`.
    pub fn cache_prefix_image_len(addr_size: usize, length_size: usize) -> Result<usize> {
        4usize
            .checked_add(4)
            .and_then(|value| value.checked_add(length_size))
            .and_then(|value| value.checked_add(length_size))
            .and_then(|value| value.checked_add(addr_size))
            .ok_or_else(|| Error::InvalidFormat("local heap prefix image length overflow".into()))
    }

    /// Serialize a local heap prefix to `out`.
    ///
    /// The output buffer is cleared before the image is written.
    pub fn cache_prefix_serialize_into(
        prefix: &LocalHeapPrefix,
        addr_size: usize,
        length_size: usize,
        out: &mut Vec<u8>,
    ) -> Result<()> {
        let image_len = Self::cache_prefix_image_len(addr_size, length_size)?;
        out.clear();
        out.reserve_exact(image_len);
        out.extend_from_slice(&HEAP_MAGIC);
        out.push(0);
        out.extend_from_slice(&[0, 0, 0]);
        encode_var(out, prefix.data_size, length_size)?;
        encode_optional_length(out, prefix.free_list_offset, length_size)?;
        encode_addr(out, prefix.data_addr, addr_size)?;
        Ok(())
    }

    /// Serialize a local heap prefix to its on-disk image.
    pub fn cache_prefix_serialize(
        prefix: &LocalHeapPrefix,
        addr_size: usize,
        length_size: usize,
    ) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        Self::cache_prefix_serialize_into(prefix, addr_size, length_size, &mut out)?;
        Ok(out)
    }

    /// Free the in-core representation of a local heap prefix. Mirrors
    /// `H5HL__cache_prefix_free_icr` (no-op in the Rust port).
    pub fn cache_prefix_free_icr(_prefix: LocalHeapPrefix) {}

    /// Buffer size to read for the local heap data block, derived from
    /// the prefix. Mirrors `H5HL__cache_datablock_get_initial_load_size`.
    pub fn cache_datablock_get_initial_load_size(prefix: &LocalHeapPrefix) -> Result<usize> {
        heap_len(prefix.data_size, "local heap data size")
    }

    /// On-disk size of the local heap's data block. Mirrors
    /// `H5HL__cache_datablock_image_len`.
    pub fn cache_datablock_image_len(&self) -> usize {
        self.data.len()
    }

    /// Deserialize a local heap data block from its on-disk image.
    /// Mirrors libhdf5's `H5HL__cache_datablock_deserialize`.
    pub fn cache_datablock_deserialize(prefix: &LocalHeapPrefix, image: &[u8]) -> Result<Self> {
        let mut data = Vec::new();
        Self::cache_datablock_deserialize_into(prefix, image, &mut data)?;
        Ok(Self { data })
    }

    /// Deserialize a local heap data block into `out`.
    ///
    /// The output buffer is cleared only after the borrowed image length has
    /// been validated against the prefix.
    pub fn cache_datablock_deserialize_into(
        prefix: &LocalHeapPrefix,
        image: &[u8],
        out: &mut Vec<u8>,
    ) -> Result<()> {
        let expected_len = heap_len(prefix.data_size, "local heap data block size")?;
        if image.len() != expected_len {
            return Err(Error::InvalidFormat(format!(
                "local heap data block image length {} does not match prefix size {expected_len}",
                image.len()
            )));
        }
        out.clear();
        out.reserve_exact(image.len());
        out.extend_from_slice(image);
        Ok(())
    }

    /// Borrow the local heap's data block as its on-disk image.
    pub fn cache_datablock_image(&self) -> Result<&[u8]> {
        let len = u64::try_from(self.data.len()).map_err(|_| {
            Error::InvalidFormat("local heap data block length does not fit in u64".into())
        })?;
        heap_len(len, "local heap data block size")?;
        Ok(&self.data)
    }

    /// Serialize the local heap's data block into `out`.
    ///
    /// The output buffer is cleared before the image is written.
    pub fn cache_datablock_serialize_into(&self, out: &mut Vec<u8>) -> Result<()> {
        let image = self.cache_datablock_image()?;
        out.clear();
        out.reserve_exact(image.len());
        out.extend_from_slice(image);
        Ok(())
    }

    /// Serialize the local heap's data block to its on-disk image.
    pub fn cache_datablock_serialize(&self) -> Result<Vec<u8>> {
        Ok(self.cache_datablock_image()?.to_vec())
    }

    /// Flush-dependency lifecycle hook. Mirrors
    /// `H5HL__cache_datablock_notify` (no-op without a metadata cache).
    pub fn cache_datablock_notify(&self) {}

    /// Free the in-core representation of the data block. Mirrors
    /// `H5HL__cache_datablock_free_icr`.
    pub fn cache_datablock_free_icr(self) {}

    /// Increment the heap's reference count. Mirrors `H5HL__inc_rc`.
    pub fn inc_rc(&mut self) {}

    /// Decrement the heap's reference count. Mirrors `H5HL__dec_rc`.
    pub fn dec_rc(&mut self) -> Result<()> {
        Ok(())
    }

    /// Destroy the in-memory heap. Mirrors `H5HL__dest`.
    pub fn dest(self) {}

    /// Create a new local heap data block of the given byte size.
    /// Mirrors `H5HL__dblk_new`.
    pub fn dblk_new(size: usize) -> Self {
        Self {
            data: vec![0; size],
        }
    }

    /// Destroy a local heap data block. Mirrors `H5HL__dblk_dest`.
    pub fn dblk_dest(self) {}

    /// Reallocate the heap's data block to a new size, zero-filling any
    /// growth. Mirrors `H5HL__dblk_realloc`.
    pub fn dblk_realloc(&mut self, new_size: usize) {
        self.data.resize(new_size, 0);
    }

    /// Render debugging information about the heap into `out`.
    pub fn write_debug(&self, out: &mut impl fmt::Write) -> fmt::Result {
        write!(out, "LocalHeap(size={})", self.data.len())
    }

    /// Render debugging information about the heap.
    pub fn debug(&self) -> String {
        let mut out = String::new();
        self.write_debug(&mut out)
            .expect("writing LocalHeap debug output to String cannot fail");
        out
    }

    /// Read a local heap from the given address.
    ///
    /// Composition mirrors the C side: `decode_prefix` parses the on-disk
    /// header (`H5HL__cache_prefix_deserialize`), then we follow the
    /// `data_addr` pointer to pull the actual heap bytes.
    pub fn read_at<R: Read + Seek>(reader: &mut HdfReader<R>, addr: u64) -> Result<Self> {
        let prefix = Self::decode_prefix(reader, addr)?;
        Self::load_data_segment(reader, &prefix)
    }

    /// Pure header decode: read & validate the local-heap prefix at `addr`,
    /// returning its parsed fields. No I/O against the data segment.
    pub fn decode_prefix<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        addr: u64,
    ) -> Result<LocalHeapPrefix> {
        reader.seek(addr)?;

        // Magic
        let mut magic = [0u8; 4];
        reader.read_bytes_into(&mut magic)?;
        if magic != HEAP_MAGIC {
            return Err(Error::InvalidFormat("invalid local heap magic".into()));
        }

        // Version
        let version = reader.read_u8()?;
        if version != 0 {
            return Err(Error::Unsupported(format!("local heap version {version}")));
        }

        reader.skip(3)?;

        // Data segment size
        let data_size = reader.read_length()?;

        // Free list head offset
        let free_list_offset = reader.read_length()?;
        let free_list_null = undefined_length(reader.sizeof_size())?;
        if free_list_offset != free_list_null && free_list_offset >= data_size {
            return Err(Error::InvalidFormat(
                "local heap free-list offset is out of bounds".into(),
            ));
        }

        // Data segment address
        let data_addr = reader.read_addr()?;
        let undef_addr = undefined_address(reader.sizeof_addr())?;
        if data_size > 0 && data_addr == undef_addr {
            return Err(Error::InvalidFormat(
                "local heap data address is undefined".into(),
            ));
        }

        Ok(LocalHeapPrefix {
            data_size,
            free_list_offset,
            data_addr,
        })
    }

    /// Follow a decoded prefix to load the heap's data segment.
    pub fn load_data_segment<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        prefix: &LocalHeapPrefix,
    ) -> Result<Self> {
        if prefix.data_size == 0 {
            return Ok(Self { data: Vec::new() });
        }
        let data_len = heap_len(prefix.data_size, "local heap data size")?;
        let data_end = prefix
            .data_addr
            .checked_add(prefix.data_size)
            .ok_or_else(|| {
                Error::InvalidFormat("local heap data segment address overflow".into())
            })?;
        let file_len = reader.len()?;
        if data_end > file_len {
            return Err(Error::InvalidFormat(
                "local heap data segment extends past end of file".into(),
            ));
        }
        reader.seek(prefix.data_addr)?;
        let mut data = vec![0; data_len];
        reader.read_bytes_into(&mut data)?;
        Ok(Self { data })
    }

    /// Borrow a null-terminated UTF-8 string at the given offset.
    pub fn get_str(&self, offset: usize) -> Result<&str> {
        std::str::from_utf8(self.get_bytes(offset)?)
            .map_err(|_| Error::InvalidFormat("local heap string is not UTF-8".into()))
    }

    /// Borrow the raw bytes of a null-terminated string at the given offset.
    pub fn get_bytes(&self, offset: usize) -> Result<&[u8]> {
        if offset >= self.data.len() {
            return Err(Error::InvalidFormat(
                "local heap string offset is out of bounds".into(),
            ));
        }

        let end = self.data[offset..]
            .iter()
            .position(|&b| b == 0)
            .map(|p| offset + p)
            .ok_or_else(|| {
                Error::InvalidFormat("local heap string is not null-terminated".into())
            })?;

        Ok(&self.data[offset..end])
    }

    /// Copy a null-terminated UTF-8 string at the given offset into `out`.
    ///
    /// The borrowed string range and UTF-8 are validated before `out` is
    /// cleared, so errors leave the caller's buffer untouched.
    pub fn get_string_into(&self, offset: usize, out: &mut String) -> Result<()> {
        let value = self.get_str(offset)?;
        out.clear();
        out.push_str(value);
        Ok(())
    }

    /// Get a null-terminated UTF-8 string at the given offset as an owned value.
    pub fn get_string(&self, offset: usize) -> Result<String> {
        let mut out = String::new();
        self.get_string_into(offset, &mut out)?;
        Ok(out)
    }

    /// Locate the terminating null byte of the string starting at `offset`,
    /// returning its byte index.
    fn string_end(&self, offset: usize) -> Result<usize> {
        if offset >= self.data.len() {
            return Err(Error::InvalidFormat(
                "local heap string offset is out of bounds".into(),
            ));
        }
        self.data[offset..]
            .iter()
            .position(|&b| b == 0)
            .map(|p| offset + p)
            .ok_or_else(|| Error::InvalidFormat("local heap string is not null-terminated".into()))
    }
}

/// Convert a heap-encoded length into a `usize`, rejecting overflow or
/// values that exceed the supported maximum.
fn heap_len(value: u64, context: &str) -> Result<usize> {
    let len = usize::try_from(value)
        .map_err(|_| Error::InvalidFormat(format!("{context} does not fit in usize")))?;
    if len > MAX_LOCAL_HEAP_BYTES {
        return Err(Error::InvalidFormat(format!(
            "{context} {len} exceeds supported maximum {MAX_LOCAL_HEAP_BYTES}"
        )));
    }
    Ok(len)
}

/// Encode an unsigned integer into `size` little-endian bytes, rejecting
/// invalid widths and values that don't fit.
fn encode_var(out: &mut Vec<u8>, value: u64, size: usize) -> Result<()> {
    if size == 0 || size > 8 {
        return Err(Error::InvalidFormat(
            "local heap encoded integer size is invalid".into(),
        ));
    }
    if size < 8 && value >= (1u64 << (size * 8)) {
        return Err(Error::InvalidFormat(format!(
            "local heap encoded integer value {value:#x} does not fit in {size} bytes"
        )));
    }
    let bytes = value.to_le_bytes();
    out.extend_from_slice(&bytes[..size]);
    Ok(())
}

/// Encode a file address (or `UNDEF_ADDR` as all-0xff) into `size`
/// little-endian bytes.
fn encode_addr(out: &mut Vec<u8>, value: u64, size: usize) -> Result<()> {
    if size == 0 || size > 8 {
        return Err(Error::InvalidFormat(
            "local heap encoded address size is invalid".into(),
        ));
    }
    if value == UNDEF_ADDR {
        out.extend(std::iter::repeat_n(0xff, size));
        return Ok(());
    }
    encode_var(out, value, size)
}

/// Encode an optional length field, writing the all-ones sentinel for
/// `UNDEF_ADDR` (the free-list "no head" marker).
fn encode_optional_length(out: &mut Vec<u8>, value: u64, size: usize) -> Result<()> {
    if size == 0 || size > 8 {
        return Err(Error::InvalidFormat(
            "local heap encoded length size is invalid".into(),
        ));
    }
    if value == UNDEF_ADDR {
        out.extend(std::iter::repeat_n(0xff, size));
        return Ok(());
    }
    encode_var(out, value, size)
}

/// Sentinel "undefined length" value for a length field of the given byte
/// width (all bits set, capped at u64::MAX for width 8).
fn undefined_length(width: u8) -> Result<u64> {
    if width == 0 || width > 8 {
        return Err(Error::Unsupported(format!(
            "local heap length width {width} exceeds u64"
        )));
    }
    Ok(if width == 8 {
        u64::MAX
    } else {
        let bits = u32::from(width)
            .checked_mul(8)
            .ok_or_else(|| Error::InvalidFormat("local heap length width overflow".into()))?;
        (1u64 << bits) - 1
    })
}

/// Sentinel "undefined address" value for an address field of the given
/// byte width.
fn undefined_address(width: u8) -> Result<u64> {
    if width == 0 || width > 8 {
        return Err(Error::Unsupported(format!(
            "local heap address width {width} exceeds u64"
        )));
    }
    Ok(if width == 8 {
        u64::MAX
    } else {
        let bits = u32::from(width)
            .checked_mul(8)
            .ok_or_else(|| Error::InvalidFormat("local heap address width overflow".into()))?;
        (1u64 << bits) - 1
    })
}

/// Read an 8-byte little-endian `u64` from `record` at `offset`, returning
/// `None` if the slice is too short.
fn read_le_u64_from_record(record: &[u8], offset: usize) -> Option<u64> {
    let end = offset.checked_add(8)?;
    let bytes: [u8; 8] = record.get(offset..end)?.try_into().ok()?;
    Some(u64::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_get_and_remove_string() {
        let mut heap = LocalHeap::new();
        let offset = heap.insert(b"name").unwrap();
        assert_eq!(heap.get_str(offset).unwrap(), "name");
        assert_eq!(heap.get_bytes(offset).unwrap(), b"name");
        let mut owned = String::from("previous");
        heap.get_string_into(offset, &mut owned).unwrap();
        assert_eq!(owned, "name");
        heap.remove(offset).unwrap();
        assert_eq!(heap.data[offset], 0);
    }

    #[test]
    fn prefix_serializes_fixed_header_fields() {
        let prefix = LocalHeap::prfx_new(16, 0, 128);
        let mut encoded = vec![99];
        LocalHeap::cache_prefix_serialize_into(&prefix, 8, 8, &mut encoded).unwrap();
        assert_eq!(&encoded[..4], b"HEAP");
        assert_eq!(encoded[4], 0);
    }

    #[test]
    fn local_heap_prefix_serialize_checks_configured_widths() {
        let prefix = LocalHeap::prfx_new(16, u64::MAX, UNDEF_ADDR);
        let mut encoded = Vec::new();
        LocalHeap::cache_prefix_serialize_into(&prefix, 4, 4, &mut encoded).unwrap();
        assert_eq!(&encoded[12..16], &[0xff; 4]);
        assert_eq!(&encoded[16..20], &[0xff; 4]);

        let too_large_size = LocalHeap::prfx_new(u64::from(u32::MAX) + 1, 0, 128);
        assert!(
            LocalHeap::cache_prefix_serialize_into(&too_large_size, 4, 4, &mut encoded).is_err()
        );

        let too_large_addr = LocalHeap::prfx_new(16, 0, u64::from(u32::MAX) + 1);
        assert!(
            LocalHeap::cache_prefix_serialize_into(&too_large_addr, 4, 4, &mut encoded).is_err()
        );
    }

    #[test]
    fn local_heap_prefix_rejects_out_of_bounds_free_list_offset() {
        let mut image = b"HEAP".to_vec();
        image.push(0);
        image.extend_from_slice(&[0; 3]);
        image.extend_from_slice(&16u64.to_le_bytes());
        image.extend_from_slice(&16u64.to_le_bytes());
        image.extend_from_slice(&128u64.to_le_bytes());

        let mut reader = HdfReader::new(std::io::Cursor::new(image));
        let err = LocalHeap::decode_prefix(&mut reader, 0)
            .expect_err("free-list offset equal to heap size should fail");
        assert!(
            err.to_string().contains("free-list offset"),
            "unexpected error: {err}"
        );

        let prefix = LocalHeap::prfx_new(0, u64::MAX, 128);
        let mut image = Vec::new();
        LocalHeap::cache_prefix_serialize_into(&prefix, 8, 8, &mut image).unwrap();
        let mut reader = HdfReader::new(std::io::Cursor::new(image));
        LocalHeap::decode_prefix(&mut reader, 0)
            .expect("undefined free-list offset should be accepted");
    }

    #[test]
    fn local_heap_prefix_rejects_nonempty_undefined_data_address() {
        let prefix = LocalHeap::prfx_new(16, u64::MAX, u64::MAX);
        let mut image = Vec::new();
        LocalHeap::cache_prefix_serialize_into(&prefix, 8, 8, &mut image).unwrap();
        let mut reader = HdfReader::new(std::io::Cursor::new(image));
        let err = LocalHeap::decode_prefix(&mut reader, 0)
            .expect_err("nonempty heap with undefined data address should fail");
        assert!(
            err.to_string().contains("data address"),
            "unexpected error: {err}"
        );

        let prefix = LocalHeap::prfx_new(0, u64::MAX, u64::MAX);
        let mut image = Vec::new();
        LocalHeap::cache_prefix_serialize_into(&prefix, 8, 8, &mut image).unwrap();
        let mut reader = HdfReader::new(std::io::Cursor::new(image));
        LocalHeap::decode_prefix(&mut reader, 0)
            .expect("empty heap with undefined data address should be accepted");
    }

    #[test]
    fn empty_local_heap_does_not_seek_to_undefined_data_address() {
        let prefix = LocalHeap::prfx_new(0, u64::MAX, u64::MAX);
        let mut reader = HdfReader::new(std::io::Cursor::new(Vec::<u8>::new()));
        let heap = LocalHeap::load_data_segment(&mut reader, &prefix)
            .expect("empty heap should not read undefined data address");
        assert!(heap.data.is_empty());
    }

    #[test]
    fn local_heap_data_segment_bounds_checked_before_allocation() {
        let prefix = LocalHeap::prfx_new(1024, u64::MAX, 4096);
        let mut reader = HdfReader::new(std::io::Cursor::new(Vec::<u8>::new()));
        let err = LocalHeap::load_data_segment(&mut reader, &prefix)
            .expect_err("out-of-file heap data segment should fail before allocation");
        assert!(
            err.to_string().contains("extends past end of file"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn local_heap_datablock_cache_roundtrips_and_checks_prefix_size() {
        let prefix = LocalHeap::prfx_new(4, u64::MAX, 128);
        let heap = LocalHeap::cache_datablock_deserialize(&prefix, b"name").unwrap();
        assert_eq!(heap.cache_datablock_image_len(), 4);
        assert_eq!(heap.cache_datablock_image().unwrap(), b"name");
        let mut data = vec![99, 99];
        LocalHeap::cache_datablock_deserialize_into(&prefix, b"name", &mut data).unwrap();
        assert_eq!(data, b"name");
        let mut encoded = vec![99];
        heap.cache_datablock_serialize_into(&mut encoded).unwrap();
        assert_eq!(encoded, b"name");

        let err = LocalHeap::cache_datablock_deserialize(&prefix, b"nam").unwrap_err();
        assert!(err.to_string().contains("does not match prefix size"));
        assert_eq!(data, b"name");
    }

    #[test]
    fn free_list_deserialize_rejects_partial_trailing_record_without_panicking() {
        let mut image = Vec::new();
        image.extend_from_slice(&3u64.to_le_bytes());
        image.extend_from_slice(&5u64.to_le_bytes());
        image.extend_from_slice(&7u64.to_le_bytes());

        let err = match LocalHeap::fl_entries(&image) {
            Ok(_) => panic!("partial free-list record should fail"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("partial trailing record"));

        let mut complete = Vec::new();
        complete.extend_from_slice(&3u64.to_le_bytes());
        complete.extend_from_slice(&5u64.to_le_bytes());
        let entries: Vec<_> = LocalHeap::fl_entries(&complete).unwrap().collect();
        assert_eq!(entries, vec![(3, 5)]);

        let mut entries = vec![(99, 100)];
        LocalHeap::fl_deserialize_into(&complete, &mut entries).unwrap();
        assert_eq!(entries, vec![(3, 5)]);

        let mut encoded = vec![99];
        LocalHeap::fl_serialize_into(&[(3, 5)], &mut encoded).unwrap();
        assert_eq!(encoded, complete);
    }
}
