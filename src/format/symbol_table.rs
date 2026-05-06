use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::io::reader::HdfReader;

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

        let magic = reader.read_bytes(4)?;
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
}
