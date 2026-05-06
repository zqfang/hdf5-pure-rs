use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::format::checksum::checksum_metadata;
use crate::io::reader::{HdfReader, UNDEF_ADDR};

/// HDF5 file signature: `\211HDF\r\n\032\n`
pub const HDF5_SIGNATURE: [u8; 8] = [0x89, 0x48, 0x44, 0x46, 0x0D, 0x0A, 0x1A, 0x0A];

/// Hardcoded version numbers used in superblock v0/v1.
const HDF5_FREESPACE_VERSION: u8 = 0;
const HDF5_OBJECTDIR_VERSION: u8 = 0;
const HDF5_SHAREDHEADER_VERSION: u8 = 0;

/// Parsed HDF5 superblock.
#[derive(Debug, Clone)]
pub struct Superblock {
    /// Superblock version (0, 1, 2, or 3).
    pub version: u8,
    /// Size of file addresses in bytes.
    pub sizeof_addr: u8,
    /// Size of file lengths in bytes.
    pub sizeof_size: u8,
    /// File status flags.
    pub status_flags: u8,
    /// Base address of the file (usually 0).
    pub base_addr: u64,
    /// Address of the superblock extension object header (UNDEF if none).
    pub ext_addr: u64,
    /// End-of-file address.
    pub eof_addr: u64,
    /// Root group object header address (v2+) or root symbol table entry header (v0/v1).
    pub root_addr: u64,
    /// Driver information block address (v0/v1 only, UNDEF for v2+).
    pub driver_addr: u64,

    // V0/V1 specific fields
    /// Symbol table leaf node 1/2 rank (K value).
    pub sym_leaf_k: u16,
    /// B-tree symbol table internal node 1/2 rank (K value).
    pub snode_btree_k: u16,
    /// B-tree chunk internal node 1/2 rank (K value, v1 only).
    pub chunk_btree_k: u16,

    // V0/V1: root group symbol table entry
    /// Root group symbol table entry: link name offset in local heap.
    pub root_entry_name_offset: u64,
    /// Root group symbol table entry: object header address.
    pub root_entry_obj_header_addr: u64,
    /// Root group symbol table entry: cache type.
    pub root_entry_cache_type: u32,
    /// Root group symbol table entry: B-tree address (if cache_type == 1).
    pub root_entry_btree_addr: u64,
    /// Root group symbol table entry: name heap address (if cache_type == 1).
    pub root_entry_name_heap_addr: u64,
}

impl Default for Superblock {
    fn default() -> Self {
        Self {
            version: 2,
            sizeof_addr: 8,
            sizeof_size: 8,
            status_flags: 0,
            base_addr: 0,
            ext_addr: UNDEF_ADDR,
            eof_addr: 0,
            root_addr: UNDEF_ADDR,
            driver_addr: UNDEF_ADDR,
            sym_leaf_k: 4,
            snode_btree_k: 16,
            chunk_btree_k: 32,
            root_entry_name_offset: 0,
            root_entry_obj_header_addr: UNDEF_ADDR,
            root_entry_cache_type: 0,
            root_entry_btree_addr: UNDEF_ADDR,
            root_entry_name_heap_addr: UNDEF_ADDR,
        }
    }
}

impl Superblock {
    /// Read and parse the superblock from the beginning of an HDF5 file.
    pub fn read<R: Read + Seek>(reader: &mut HdfReader<R>) -> Result<Self> {
        // Seek to beginning
        reader.seek(0)?;

        // Read and verify signature
        let sig = reader.read_bytes(8)?;
        if sig != HDF5_SIGNATURE {
            return Err(Error::InvalidFormat("invalid HDF5 file signature".into()));
        }

        // Read superblock version
        let version = reader.read_u8()?;
        if version > 3 {
            return Err(Error::Unsupported(format!(
                "superblock version {version} not supported"
            )));
        }

        if version < 2 {
            Self::read_v0_v1(reader, version)
        } else {
            Self::read_v2_v3(reader, version)
        }
    }

    /// Read superblock version 0 or 1.
    fn read_v0_v1<R: Read + Seek>(reader: &mut HdfReader<R>, version: u8) -> Result<Self> {
        // Freespace version (must be 0)
        let freespace_ver = reader.read_u8()?;
        if freespace_ver != HDF5_FREESPACE_VERSION {
            return Err(Error::InvalidFormat(format!(
                "bad free space version: {freespace_ver}"
            )));
        }

        // Root group version (must be 0)
        let rootgrp_ver = reader.read_u8()?;
        if rootgrp_ver != HDF5_OBJECTDIR_VERSION {
            return Err(Error::InvalidFormat(format!(
                "bad object directory version: {rootgrp_ver}"
            )));
        }

        let reserved = reader.read_u8()?;
        if reserved != 0 {
            return Err(Error::InvalidFormat(
                "nonzero superblock reserved byte".into(),
            ));
        }

        // Shared header version (must be 0)
        let shared_ver = reader.read_u8()?;
        if shared_ver != HDF5_SHAREDHEADER_VERSION {
            return Err(Error::InvalidFormat(format!(
                "bad shared header version: {shared_ver}"
            )));
        }

        // Size of offsets and lengths
        let sizeof_addr = reader.read_u8()?;
        let sizeof_size = reader.read_u8()?;
        Self::validate_sizes(sizeof_addr, sizeof_size)?;

        // Update the reader with the correct sizes
        reader.set_sizeof_addr(sizeof_addr);
        reader.set_sizeof_size(sizeof_size);

        let reserved = reader.read_u8()?;
        if reserved != 0 {
            return Err(Error::InvalidFormat(
                "nonzero superblock reserved byte".into(),
            ));
        }

        // B-tree K values
        let sym_leaf_k = reader.read_u16()?;
        let snode_btree_k = reader.read_u16()?;

        // File consistency flags (4 bytes in v0/v1)
        let status_flags = reader.read_u32()? as u8;

        // Indexed storage B-tree internal K (v1 only)
        let chunk_btree_k = if version > 0 {
            let k = reader.read_u16()?;
            if version == 1 {
                let reserved = reader.read_u16()?;
                if reserved != 0 {
                    return Err(Error::InvalidFormat(
                        "nonzero superblock v1 reserved bytes".into(),
                    ));
                }
            }
            k
        } else {
            32 // HDF5_BTREE_CHUNK_IK_DEF
        };

        // Addresses
        let base_addr = reader.read_addr()?;
        let ext_addr = reader.read_addr()?; // "free space info" / unused in v0/v1
        let eof_addr = reader.read_addr()?;
        let driver_addr = reader.read_addr()?;

        // Root group symbol table entry
        // H5G_SIZEOF_ENTRY = sizeof_addr + sizeof_size + 4 + 16
        let root_entry_name_offset = reader.read_length()?;
        let root_entry_obj_header_addr = reader.read_addr()?;
        let root_entry_cache_type = reader.read_u32()?;
        let root_entry_reserved = reader.read_u32()?;
        if root_entry_reserved != 0 {
            return Err(Error::InvalidFormat(
                "nonzero root symbol table entry reserved bytes".into(),
            ));
        }

        // Scratch-pad space (16 bytes) -- for cache_type == 1, contains btree + heap addr
        let (root_entry_btree_addr, root_entry_name_heap_addr) = match root_entry_cache_type {
            0 | 2 => {
                reader.skip(16)?;
                (UNDEF_ADDR, UNDEF_ADDR)
            }
            1 => {
                let btree = reader.read_addr()?;
                let heap = reader.read_addr()?;
                // Skip remaining scratch-pad bytes
                let used = u64::from(sizeof_addr).checked_mul(2).ok_or_else(|| {
                    Error::InvalidFormat("root symbol table scratch-pad size overflow".into())
                })?;
                if used > 16 {
                    return Err(Error::InvalidFormat(
                        "root symbol table scratch-pad exceeds fixed size".into(),
                    ));
                }
                if used < 16 {
                    reader.skip(16 - used)?;
                }
                (btree, heap)
            }
            other => {
                return Err(Error::InvalidFormat(format!(
                    "invalid root symbol table cache type {other}"
                )));
            }
        };

        Ok(Superblock {
            version,
            sizeof_addr,
            sizeof_size,
            status_flags,
            base_addr,
            ext_addr,
            eof_addr,
            root_addr: root_entry_obj_header_addr,
            driver_addr,
            sym_leaf_k,
            snode_btree_k,
            chunk_btree_k,
            root_entry_name_offset,
            root_entry_obj_header_addr,
            root_entry_cache_type,
            root_entry_btree_addr,
            root_entry_name_heap_addr,
        })
    }

    /// Read superblock version 2 or 3.
    fn read_v2_v3<R: Read + Seek>(reader: &mut HdfReader<R>, version: u8) -> Result<Self> {
        // For v2+, we need to capture raw bytes for checksum verification.
        // The superblock starts at offset 0 (signature already read at 0..8, version at 8).
        // After version byte: sizeof_addr, sizeof_size, status_flags, then 4 addresses, then checksum.

        let sizeof_addr = reader.read_u8()?;
        let sizeof_size = reader.read_u8()?;
        Self::validate_sizes(sizeof_addr, sizeof_size)?;

        reader.set_sizeof_addr(sizeof_addr);
        reader.set_sizeof_size(sizeof_size);

        let status_flags = reader.read_u8()?;

        let base_addr = reader.read_addr()?;
        let ext_addr = reader.read_addr()?;
        let eof_addr = reader.read_addr()?;
        let root_addr = reader.read_addr()?;

        // Read checksum
        let stored_checksum = reader.read_checksum()?;

        // Verify checksum: compute over the entire superblock except the checksum itself.
        // Superblock v2: signature(8) + version(1) + sizeof_addr(1) + sizeof_size(1) +
        //                status_flags(1) + 4*sizeof_addr addresses = 12 + 4*sizeof_addr bytes
        let sb_size = Self::v2_checksum_span(sizeof_addr)?;
        reader.seek(0)?;
        let sb_data = reader.read_bytes(sb_size)?;
        let computed_checksum = checksum_metadata(&sb_data);

        if stored_checksum != computed_checksum {
            return Err(Error::InvalidFormat(format!(
                "superblock checksum mismatch: stored={stored_checksum:#010x}, computed={computed_checksum:#010x}"
            )));
        }

        // Seek past the checksum to leave reader in correct position
        let checksum_end = u64::try_from(sb_size)
            .ok()
            .and_then(|value| value.checked_add(4))
            .ok_or_else(|| Error::InvalidFormat("superblock checksum offset overflow".into()))?;
        reader.seek(checksum_end)?;

        Ok(Superblock {
            version,
            sizeof_addr,
            sizeof_size,
            status_flags,
            base_addr,
            ext_addr,
            eof_addr,
            root_addr,
            driver_addr: UNDEF_ADDR,
            sym_leaf_k: 4,
            snode_btree_k: 16,
            chunk_btree_k: 32,
            root_entry_name_offset: 0,
            root_entry_obj_header_addr: root_addr,
            root_entry_cache_type: 0,
            root_entry_btree_addr: UNDEF_ADDR,
            root_entry_name_heap_addr: UNDEF_ADDR,
        })
    }

    /// Write a v2 superblock to a writer.
    pub fn write_v2(&self, buf: &mut Vec<u8>) -> Result<()> {
        Self::validate_sizes(self.sizeof_addr, self.sizeof_size)?;
        // Signature
        buf.extend_from_slice(&HDF5_SIGNATURE);
        // Version
        buf.push(self.version);
        // sizeof_addr, sizeof_size
        buf.push(self.sizeof_addr);
        buf.push(self.sizeof_size);
        // Status flags
        buf.push(self.status_flags);

        // Addresses (little-endian, sizeof_addr bytes each)
        write_addr(
            buf,
            self.base_addr,
            self.sizeof_addr,
            "superblock base address",
        )?;
        write_addr(
            buf,
            self.ext_addr,
            self.sizeof_addr,
            "superblock extension address",
        )?;
        write_addr(
            buf,
            self.eof_addr,
            self.sizeof_addr,
            "superblock EOF address",
        )?;
        write_addr(
            buf,
            self.root_addr,
            self.sizeof_addr,
            "superblock root address",
        )?;

        // Checksum (over everything before this point)
        let checksum = checksum_metadata(buf);
        buf.extend_from_slice(&checksum.to_le_bytes());
        Ok(())
    }

    /// Compute the total size of the superblock in bytes.
    ///
    /// This infallible helper saturates on arithmetic overflow. Use
    /// [`Superblock::checked_size`] when malformed metadata should surface as
    /// an error.
    pub fn size(&self) -> usize {
        self.checked_size().unwrap_or(usize::MAX)
    }

    /// Compute the total size of the superblock in bytes with checked arithmetic.
    pub fn checked_size(&self) -> Result<usize> {
        if self.version < 2 {
            // Fixed: sig(8) + version(1) = 9
            // Variable v0: 16 + 4*sizeof_addr + H5G_SIZEOF_ENTRY
            // H5G_SIZEOF_ENTRY = sizeof_size + sizeof_addr + 4 + 4 + 16
            let entry_size = usize::from(self.sizeof_size)
                .checked_add(usize::from(self.sizeof_addr))
                .and_then(|value| value.checked_add(24))
                .ok_or_else(|| Error::InvalidFormat("superblock size overflow".into()))?;
            let common = 16; // freespace(1) + rootgrp(1) + reserved(1) + shared(1) + sizes(2) + reserved(1) + btree_k(4) + flags(4) + 1 extra reserved
            let addrs = 4usize
                .checked_mul(usize::from(self.sizeof_addr))
                .ok_or_else(|| Error::InvalidFormat("superblock size overflow".into()))?;
            let v1_extra = if self.version > 0 {
                2 + if self.version == 1 { 2 } else { 0 }
            } else {
                0
            };
            9usize
                .checked_add(common)
                .and_then(|value| value.checked_add(v1_extra))
                .and_then(|value| value.checked_add(addrs))
                .and_then(|value| value.checked_add(entry_size))
                .ok_or_else(|| Error::InvalidFormat("superblock size overflow".into()))
        } else {
            // Fixed: sig(8) + version(1) + sizeof_addr(1) + sizeof_size(1) + flags(1) +
            //        4*sizeof_addr + checksum(4)
            Self::v2_checksum_span(self.sizeof_addr)?
                .checked_add(4)
                .ok_or_else(|| Error::InvalidFormat("superblock size overflow".into()))
        }
    }

    fn v2_checksum_span(sizeof_addr: u8) -> Result<usize> {
        4usize
            .checked_mul(usize::from(sizeof_addr))
            .and_then(|addr_bytes| 12usize.checked_add(addr_bytes))
            .ok_or_else(|| Error::InvalidFormat("superblock checksum span overflow".into()))
    }

    fn validate_sizes(sizeof_addr: u8, sizeof_size: u8) -> Result<()> {
        let valid = [2, 4, 8, 16, 32];
        if !valid.contains(&sizeof_addr) {
            return Err(Error::InvalidFormat(format!(
                "invalid sizeof_addr: {sizeof_addr}"
            )));
        }
        if !valid.contains(&sizeof_size) {
            return Err(Error::InvalidFormat(format!(
                "invalid sizeof_size: {sizeof_size}"
            )));
        }
        if sizeof_addr > 8 {
            return Err(Error::Unsupported(format!(
                "sizeof_addr {sizeof_addr} exceeds current 64-bit address support"
            )));
        }
        if sizeof_size > 8 {
            return Err(Error::Unsupported(format!(
                "sizeof_size {sizeof_size} exceeds current 64-bit length support"
            )));
        }
        Ok(())
    }
}

/// Write an address as little-endian bytes of the given width.
fn write_addr(buf: &mut Vec<u8>, addr: u64, size: u8, context: &str) -> Result<()> {
    let size = usize::from(size);
    if !(1..=8).contains(&size) {
        return Err(Error::InvalidFormat(format!("{context} width is invalid")));
    }
    if addr == UNDEF_ADDR {
        buf.extend(std::iter::repeat_n(0xff, size));
        return Ok(());
    }
    if size < 8 {
        let bits = size
            .checked_mul(8)
            .ok_or_else(|| Error::InvalidFormat(format!("{context} width overflow")))?;
        if addr >= (1u64 << bits) {
            return Err(Error::InvalidFormat(format!(
                "{context} does not fit in {size} bytes"
            )));
        }
    }
    buf.extend_from_slice(&addr.to_le_bytes()[..size]);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_v2_superblock_roundtrip() {
        let sb = Superblock {
            version: 2,
            sizeof_addr: 8,
            sizeof_size: 8,
            status_flags: 0,
            base_addr: 0,
            ext_addr: UNDEF_ADDR,
            eof_addr: 0x1000,
            root_addr: 0x60,
            ..Default::default()
        };

        // Write
        let mut buf = Vec::new();
        sb.write_v2(&mut buf).unwrap();

        // Expected size: 8 + 1 + 1 + 1 + 1 + 4*8 + 4 = 48
        assert_eq!(buf.len(), 48);

        // Read back
        let mut reader = HdfReader::new(Cursor::new(buf));
        let sb2 = Superblock::read(&mut reader).unwrap();

        assert_eq!(sb2.version, 2);
        assert_eq!(sb2.sizeof_addr, 8);
        assert_eq!(sb2.sizeof_size, 8);
        assert_eq!(sb2.status_flags, 0);
        assert_eq!(sb2.base_addr, 0);
        assert_eq!(sb2.ext_addr, UNDEF_ADDR);
        assert_eq!(sb2.eof_addr, 0x1000);
        assert_eq!(sb2.root_addr, 0x60);
    }

    #[test]
    fn rejects_wide_address_and_length_sizes() {
        assert!(matches!(
            Superblock::validate_sizes(16, 8),
            Err(Error::Unsupported(_))
        ));
        assert!(matches!(
            Superblock::validate_sizes(8, 16),
            Err(Error::Unsupported(_))
        ));
    }

    #[test]
    fn write_v2_rejects_unrepresentable_address_width() {
        let sb = Superblock {
            version: 2,
            sizeof_addr: 2,
            sizeof_size: 8,
            eof_addr: 0x1_0000,
            ..Default::default()
        };
        let mut buf = Vec::new();
        let err = sb.write_v2(&mut buf).unwrap_err();
        assert!(
            err.to_string()
                .contains("superblock EOF address does not fit in 2 bytes"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_invalid_signature() {
        let data = vec![0u8; 64];
        let mut reader = HdfReader::new(Cursor::new(data));
        assert!(Superblock::read(&mut reader).is_err());
    }

    #[test]
    fn test_superblock_size_v2() {
        let sb = Superblock::default();
        assert_eq!(sb.size(), 48); // 12 + 4*8 + 4
        assert_eq!(sb.checked_size().unwrap(), 48);
    }

    #[test]
    fn v2_checksum_span_uses_checked_arithmetic() {
        assert_eq!(Superblock::v2_checksum_span(8).unwrap(), 44);
    }
}
