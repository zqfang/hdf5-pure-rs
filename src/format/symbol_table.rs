use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::io::reader::{HdfReader, UNDEF_ADDR};

/// Symbol table node magic: "SNOD"
const SNOD_MAGIC: [u8; 4] = [b'S', b'N', b'O', b'D'];

/// An entry in a v1 symbol table node.
#[derive(Debug, Clone)]
pub struct SymbolTableEntry {
    /// Offset of the link name in the local heap.
    pub name_offset: u64,
    /// Address of the object header.
    pub obj_header_addr: u64,
    /// Cache type (0=none, 1=group with stab, 2=symbolic link).
    pub cache_type: u32,
    /// Cached B-tree address (if cache_type == 1).
    pub cached_btree_addr: Option<u64>,
    /// Cached name heap address (if cache_type == 1).
    pub cached_name_heap_addr: Option<u64>,
    /// Cached link offset (if cache_type == 2).
    pub cached_link_offset: Option<u32>,
}

/// A v1 symbol table node (SNOD).
#[derive(Debug, Clone)]
pub struct SymbolTableNode {
    pub version: u8,
    pub entries: Vec<SymbolTableEntry>,
}

impl SymbolTableNode {
    /// Read a symbol table node at the given address.
    pub fn read_at<R: Read + Seek>(reader: &mut HdfReader<R>, addr: u64) -> Result<Self> {
        reader.seek(addr)?;

        let mut magic = [0u8; 4];
        reader.read_bytes_into(&mut magic)?;
        if magic != SNOD_MAGIC {
            return Err(Error::InvalidFormat(
                "invalid symbol table node magic".into(),
            ));
        }

        let version = reader.read_u8()?;
        if version != 1 {
            return Err(Error::Unsupported(format!(
                "symbol table node version {version}"
            )));
        }

        reader.skip(1)?;

        let num_symbols = usize::from(reader.read_u16()?);

        let mut entries = Vec::with_capacity(num_symbols);
        for _ in 0..num_symbols {
            let entry = Self::read_entry(reader)?;
            entries.push(entry);
        }

        Ok(Self { version, entries })
    }

    /// Initial number of bytes the metadata cache should load to inspect the
    /// SNOD prefix (magic + version + reserved + entry count).
    pub fn cache_node_get_initial_load_size() -> usize {
        8
    }

    /// Compute the on-disk image size of this symbol-table node for the
    /// file's configured address/size widths.
    pub fn cache_node_image_len(&self, sizeof_addr: u8, sizeof_size: u8) -> Result<usize> {
        validate_width(sizeof_addr, "symbol table address width")?;
        validate_width(sizeof_size, "symbol table size width")?;
        let entry_len = usize::from(sizeof_size)
            .checked_add(usize::from(sizeof_addr))
            .and_then(|value| value.checked_add(4))
            .and_then(|value| value.checked_add(4))
            .and_then(|value| value.checked_add(16))
            .ok_or_else(|| Error::InvalidFormat("symbol table entry length overflow".into()))?;
        8usize
            .checked_add(self.entries.len().checked_mul(entry_len).ok_or_else(|| {
                Error::InvalidFormat("symbol table node image length overflow".into())
            })?)
            .ok_or_else(|| Error::InvalidFormat("symbol table node image length overflow".into()))
    }

    /// Encode this symbol-table node into its on-disk SNOD image, using the
    /// supplied address and length widths.
    pub fn cache_node_serialize_into(
        &self,
        sizeof_addr: u8,
        sizeof_size: u8,
        out: &mut Vec<u8>,
    ) -> Result<()> {
        validate_width(sizeof_addr, "symbol table address width")?;
        validate_width(sizeof_size, "symbol table size width")?;
        if self.version != 1 {
            return Err(Error::InvalidFormat(format!(
                "symbol table node version {} is unsupported",
                self.version
            )));
        }
        if u16::try_from(self.entries.len()).is_err() {
            return Err(Error::InvalidFormat(
                "symbol table entry count exceeds u16".into(),
            ));
        }
        out.clear();
        out.reserve_exact(self.cache_node_image_len(sizeof_addr, sizeof_size)?);
        out.extend_from_slice(&SNOD_MAGIC);
        out.push(1);
        out.push(0);
        out.extend_from_slice(&(self.entries.len() as u16).to_le_bytes());
        for entry in &self.entries {
            Self::write_entry(out, entry, sizeof_addr, sizeof_size)?;
        }
        Ok(())
    }

    /// Encode this symbol-table node into an owned on-disk SNOD image.
    pub fn cache_node_serialize(&self, sizeof_addr: u8, sizeof_size: u8) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        self.cache_node_serialize_into(sizeof_addr, sizeof_size, &mut out)?;
        Ok(out)
    }

    /// Release a serialized SNOD image (no-op in this port, present for
    /// parity with libhdf5's metadata-cache callback API).
    pub fn cache_node_free_icr(_image: Vec<u8>) {}

    /// Decode a single symbol-table entry pointed to by the reader. Mirrors
    /// libhdf5's `H5G_ent_decode`: name offset, object header address, cache
    /// type, then the 16-byte scratch-pad whose layout depends on cache type.
    fn read_entry<R: Read + Seek>(reader: &mut HdfReader<R>) -> Result<SymbolTableEntry> {
        let name_offset = reader.read_length()?;
        let obj_header_addr = reader.read_addr()?;
        let cache_type = reader.read_u32()?;

        reader.skip(4)?;

        // Scratch-pad space: 16 bytes
        let (cached_btree_addr, cached_name_heap_addr, cached_link_offset) = match cache_type {
            0 => {
                // No cached info
                reader.skip(16)?;
                (None, None, None)
            }
            1 => {
                // Group: B-tree addr + name heap addr
                let btree = reader.read_addr()?;
                let heap = reader.read_addr()?;
                let used = u64::from(reader.sizeof_addr())
                    .checked_mul(2)
                    .ok_or_else(|| {
                        Error::InvalidFormat("symbol table scratch-pad size overflow".into())
                    })?;
                if used > 16 {
                    return Err(Error::InvalidFormat(
                        "symbol table group scratch-pad exceeds fixed size".into(),
                    ));
                }
                if used < 16 {
                    reader.skip(16 - used)?;
                }
                (Some(btree), Some(heap), None)
            }
            2 => {
                // Symbolic link: offset into local heap
                let offset = reader.read_u32()?;
                reader.skip(12)?; // remaining scratch-pad
                (None, None, Some(offset))
            }
            other => {
                return Err(Error::InvalidFormat(format!(
                    "invalid symbol table entry cache type {other}"
                )));
            }
        };

        Ok(SymbolTableEntry {
            name_offset,
            obj_header_addr,
            cache_type,
            cached_btree_addr,
            cached_name_heap_addr,
            cached_link_offset,
        })
    }

    /// Encode a single symbol-table entry into `out`, padding the scratch-pad
    /// area to its fixed 16-byte width and rejecting any mismatch between
    /// `cache_type` and the populated optional fields.
    fn write_entry(
        out: &mut Vec<u8>,
        entry: &SymbolTableEntry,
        sizeof_addr: u8,
        sizeof_size: u8,
    ) -> Result<()> {
        encode_uint_width(
            out,
            entry.name_offset,
            sizeof_size,
            "symbol table entry name offset",
        )?;
        encode_addr_width(
            out,
            entry.obj_header_addr,
            sizeof_addr,
            "symbol table entry object header address",
        )?;
        out.extend_from_slice(&entry.cache_type.to_le_bytes());
        out.extend_from_slice(&[0; 4]);
        match entry.cache_type {
            0 => {
                if entry.cached_btree_addr.is_some()
                    || entry.cached_name_heap_addr.is_some()
                    || entry.cached_link_offset.is_some()
                {
                    return Err(Error::InvalidFormat(
                        "symbol table no-cache entry has cached fields".into(),
                    ));
                }
                out.extend_from_slice(&[0; 16]);
            }
            1 => {
                let btree = entry.cached_btree_addr.ok_or_else(|| {
                    Error::InvalidFormat(
                        "symbol table group cache is missing B-tree address".into(),
                    )
                })?;
                let heap = entry.cached_name_heap_addr.ok_or_else(|| {
                    Error::InvalidFormat("symbol table group cache is missing heap address".into())
                })?;
                if entry.cached_link_offset.is_some() {
                    return Err(Error::InvalidFormat(
                        "symbol table group cache has symbolic-link offset".into(),
                    ));
                }
                let before = out.len();
                encode_addr_width(
                    out,
                    btree,
                    sizeof_addr,
                    "symbol table cached B-tree address",
                )?;
                encode_addr_width(out, heap, sizeof_addr, "symbol table cached heap address")?;
                let used = out.len().checked_sub(before).ok_or_else(|| {
                    Error::InvalidFormat("symbol table scratch-pad length underflow".into())
                })?;
                if used > 16 {
                    return Err(Error::InvalidFormat(
                        "symbol table group scratch-pad exceeds fixed size".into(),
                    ));
                }
                out.extend(std::iter::repeat_n(0, 16 - used));
            }
            2 => {
                if entry.cached_btree_addr.is_some() || entry.cached_name_heap_addr.is_some() {
                    return Err(Error::InvalidFormat(
                        "symbol table symbolic-link cache has group addresses".into(),
                    ));
                }
                let offset = entry.cached_link_offset.ok_or_else(|| {
                    Error::InvalidFormat(
                        "symbol table symbolic-link cache is missing link offset".into(),
                    )
                })?;
                out.extend_from_slice(&offset.to_le_bytes());
                out.extend_from_slice(&[0; 12]);
            }
            other => {
                return Err(Error::InvalidFormat(format!(
                    "invalid symbol table entry cache type {other}"
                )));
            }
        }
        Ok(())
    }
}

/// Reject zero or oversized address/length widths (must be 1..=8 bytes).
fn validate_width(width: u8, context: &str) -> Result<()> {
    if width == 0 || width > 8 {
        return Err(Error::InvalidFormat(format!("{context} is invalid")));
    }
    Ok(())
}

/// Append `value` as a little-endian unsigned integer of `width` bytes,
/// erroring if the value does not fit in the requested width.
fn encode_uint_width(out: &mut Vec<u8>, value: u64, width: u8, context: &str) -> Result<()> {
    validate_width(width, context)?;
    let width = usize::from(width);
    if width < 8 && value >= (1u64 << (width * 8)) {
        return Err(Error::InvalidFormat(format!(
            "{context} value {value:#x} does not fit in {width} bytes"
        )));
    }
    out.extend_from_slice(&value.to_le_bytes()[..width]);
    Ok(())
}

/// Append a file address as a width-sized little-endian integer, emitting
/// all-`0xff` bytes when the address is the undefined-address sentinel.
fn encode_addr_width(out: &mut Vec<u8>, value: u64, width: u8, context: &str) -> Result<()> {
    validate_width(width, context)?;
    if value == UNDEF_ADDR {
        out.extend(std::iter::repeat_n(0xff, usize::from(width)));
        return Ok(());
    }
    encode_uint_width(out, value, width, context)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn symbol_table_cache_node_roundtrips_with_configured_widths() {
        let node = SymbolTableNode {
            version: 1,
            entries: vec![
                SymbolTableEntry {
                    name_offset: 4,
                    obj_header_addr: 0x1234,
                    cache_type: 0,
                    cached_btree_addr: None,
                    cached_name_heap_addr: None,
                    cached_link_offset: None,
                },
                SymbolTableEntry {
                    name_offset: 8,
                    obj_header_addr: 0x5678,
                    cache_type: 1,
                    cached_btree_addr: Some(0x1000),
                    cached_name_heap_addr: Some(0x2000),
                    cached_link_offset: None,
                },
                SymbolTableEntry {
                    name_offset: 12,
                    obj_header_addr: 0x9abc,
                    cache_type: 2,
                    cached_btree_addr: None,
                    cached_name_heap_addr: None,
                    cached_link_offset: Some(44),
                },
            ],
        };

        let mut image = Vec::new();
        node.cache_node_serialize_into(4, 4, &mut image).unwrap();
        assert_eq!(image.len(), node.cache_node_image_len(4, 4).unwrap());
        let mut reader = HdfReader::new(Cursor::new(image));
        reader.set_sizeof_addr(4);
        reader.set_sizeof_size(4);
        let decoded = SymbolTableNode::read_at(&mut reader, 0).unwrap();
        assert_eq!(decoded.version, 1);
        assert_eq!(decoded.entries.len(), 3);
        assert_eq!(decoded.entries[0].obj_header_addr, 0x1234);
        assert_eq!(decoded.entries[1].cached_btree_addr, Some(0x1000));
        assert_eq!(decoded.entries[1].cached_name_heap_addr, Some(0x2000));
        assert_eq!(decoded.entries[2].cached_link_offset, Some(44));
    }

    #[test]
    fn symbol_table_cache_node_serialize_rejects_invalid_images() {
        let node = SymbolTableNode {
            version: 1,
            entries: vec![SymbolTableEntry {
                name_offset: u64::from(u32::MAX) + 1,
                obj_header_addr: 0x1234,
                cache_type: 0,
                cached_btree_addr: None,
                cached_name_heap_addr: None,
                cached_link_offset: None,
            }],
        };
        let mut image = Vec::new();
        assert!(node.cache_node_serialize_into(4, 4, &mut image).is_err());

        let bad_cache_fields = SymbolTableNode {
            version: 1,
            entries: vec![SymbolTableEntry {
                name_offset: 0,
                obj_header_addr: 0x1234,
                cache_type: 0,
                cached_btree_addr: Some(1),
                cached_name_heap_addr: None,
                cached_link_offset: None,
            }],
        };
        assert!(bad_cache_fields
            .cache_node_serialize_into(4, 4, &mut image)
            .is_err());

        let invalid_version = SymbolTableNode {
            version: 2,
            entries: Vec::new(),
        };
        assert!(invalid_version
            .cache_node_serialize_into(4, 4, &mut image)
            .is_err());
    }
}
