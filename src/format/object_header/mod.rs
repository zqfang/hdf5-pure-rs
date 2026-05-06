//! Object header — top-level public API. Mirrors libhdf5's `H5O.c` /
//! `H5Opkg.h` (the file-spanning entry points and shared types). The
//! per-component code lives in sibling modules:
//!   - `cache` → `H5Ocache.c` (`H5O__cache_deserialize` for v1 and v2)
//!   - `chunk` → `H5Ochunk.c` (continuation chunks + range bookkeeping)
//!   - `msg`   → `H5Omessage.c` + `H5Oint.c` (per-message decode loop +
//!              shared-message validation)

mod cache;
mod chunk;
mod msg;

use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::io::reader::HdfReader;

/// Magic number for v2 object headers: "OHDR"
pub(super) const OHDR_MAGIC: [u8; 4] = [b'O', b'H', b'D', b'R'];

/// Magic number for v2 continuation chunks: "OCHK"
pub(super) const OCHK_MAGIC: [u8; 4] = [b'O', b'C', b'H', b'K'];

// Object header flags (v2)
pub const HDR_CHUNK0_SIZE_MASK: u8 = 0x03;
pub const HDR_ATTR_CRT_ORDER_TRACKED: u8 = 0x04;
pub const HDR_ATTR_CRT_ORDER_INDEXED: u8 = 0x08;
pub const HDR_ATTR_STORE_PHASE_CHANGE: u8 = 0x10;
pub const HDR_STORE_TIMES: u8 = 0x20;
pub const HDR_V2_KNOWN_FLAGS: u8 = HDR_CHUNK0_SIZE_MASK
    | HDR_ATTR_CRT_ORDER_TRACKED
    | HDR_ATTR_CRT_ORDER_INDEXED
    | HDR_ATTR_STORE_PHASE_CHANGE
    | HDR_STORE_TIMES;

pub(super) const MSG_FLAG_SHARED: u8 = 0x02;
pub(super) const SHARED_MESSAGE_TABLE_VERSION: u8 = 0;
pub(super) const SHARED_MESSAGE_MAX_INDEXES: u8 = 8;
pub(super) const SHARED_REFERENCE_VERSION_1: u8 = 1;
pub(super) const SHARED_REFERENCE_VERSION_2: u8 = 2;
pub(super) const SHARED_REFERENCE_VERSION_3: u8 = 3;
pub(super) const SHARED_TYPE_SOHM: u8 = 1;
pub(super) const SHARED_TYPE_COMMITTED: u8 = 2;
pub(super) const SHARED_HEAP_ID_LEN: usize = 8;

// Message type IDs
pub const MSG_NIL: u16 = 0x0000;
pub const MSG_DATASPACE: u16 = 0x0001;
pub const MSG_LINK_INFO: u16 = 0x0002;
pub const MSG_DATATYPE: u16 = 0x0003;
pub const MSG_FILL_VALUE_OLD: u16 = 0x0004;
pub const MSG_FILL_VALUE: u16 = 0x0005;
pub const MSG_LINK: u16 = 0x0006;
pub const MSG_EXTERNAL_FILE_LIST: u16 = 0x0007;
pub const MSG_LAYOUT: u16 = 0x0008;
pub const MSG_BOGUS: u16 = 0x0009;
pub const MSG_GROUP_INFO: u16 = 0x000A;
pub const MSG_FILTER_PIPELINE: u16 = 0x000B;
pub const MSG_ATTRIBUTE: u16 = 0x000C;
pub const MSG_OBJ_COMMENT: u16 = 0x000D;
pub const MSG_OBJ_MOD_TIME_OLD: u16 = 0x000E;
pub const MSG_SHARED_MSG_TABLE: u16 = 0x000F;
pub const MSG_HEADER_CONTINUATION: u16 = 0x0010;
pub const MSG_SYMBOL_TABLE: u16 = 0x0011;
pub const MSG_OBJ_MOD_TIME: u16 = 0x0012;
pub const MSG_BTREE_K: u16 = 0x0013;
pub const MSG_DRIVER_INFO: u16 = 0x0014;
pub const MSG_ATTR_INFO: u16 = 0x0015;
pub const MSG_OBJ_REF_COUNT: u16 = 0x0016;
pub const MSG_FILE_SPACE_INFO: u16 = 0x0017;

/// A raw message from an object header.
#[derive(Debug, Clone)]
pub struct RawMessage {
    /// Message type ID.
    pub msg_type: u16,
    /// Message flags.
    pub flags: u8,
    /// Creation order index (v2 only, if tracked).
    pub creation_index: Option<u16>,
    /// Object header chunk this message was read from.
    pub chunk_index: u16,
    /// Raw message data bytes.
    pub data: Vec<u8>,
}

/// Parsed object header.
#[derive(Debug, Clone)]
pub struct ObjectHeader {
    /// Header version (1 or 2).
    pub version: u8,
    /// Header flags (v2 only).
    pub flags: u8,
    /// Reference count.
    pub refcount: u32,
    /// Access time (v2, if HDR_STORE_TIMES).
    pub atime: Option<u32>,
    /// Modification time (v2, if HDR_STORE_TIMES).
    pub mtime: Option<u32>,
    /// Change time (v2, if HDR_STORE_TIMES).
    pub ctime: Option<u32>,
    /// Birth time (v2, if HDR_STORE_TIMES).
    pub btime: Option<u32>,
    /// Max compact attributes (v2, if HDR_ATTR_STORE_PHASE_CHANGE).
    pub max_compact_attrs: Option<u16>,
    /// Min dense attributes (v2, if HDR_ATTR_STORE_PHASE_CHANGE).
    pub min_dense_attrs: Option<u16>,
    /// All messages parsed from this header (including continuation chunks).
    pub messages: Vec<RawMessage>,
}

impl ObjectHeader {
    /// Read an object header at the given file address. Mirrors the
    /// dispatcher in libhdf5's `H5O_protect`: peek at the first 4 bytes
    /// to decide whether this is a v2 (OHDR magic) or v1 (version byte)
    /// header, then hand off to the version-specific decoder in `cache.rs`.
    pub fn read_at<R: Read + Seek>(reader: &mut HdfReader<R>, addr: u64) -> Result<Self> {
        reader.seek(addr)?;

        let first_bytes = reader.read_bytes(4)?;

        let result = if first_bytes == OHDR_MAGIC {
            Self::read_v2(reader, addr)
        } else {
            // Seek back and re-read as v1. The first byte is the version.
            reader.seek(addr)?;
            Self::read_v1(reader)
        };

        #[cfg(feature = "tracehash")]
        if let Ok(header) = &result {
            let traced_messages: Vec<_> = header
                .messages
                .iter()
                .filter(|message| message.chunk_index == 0)
                .collect();
            let mut th = tracehash::th_call!("hdf5.object_header.read");
            th.input_u64(addr);
            let Ok(message_count) = u64::try_from(traced_messages.len()) else {
                return result;
            };
            th.output_u64(u64::from(header.version));
            th.output_u64(u64::from(header.flags));
            th.output_u64(u64::from(header.refcount));
            th.output_u64(message_count);
            for message in traced_messages {
                let Ok(data_len) = u64::try_from(message.data.len()) else {
                    return result;
                };
                th.output_u64(u64::from(message.msg_type));
                th.output_u64(data_len);
            }
            th.finish();
        }

        result
    }
}

// ---------------------------------------------------------------------------
// Internal numeric helpers shared across submodules.
// ---------------------------------------------------------------------------

pub(super) fn read_le_uint(data: &[u8]) -> Result<u64> {
    if data.len() > 8 {
        return Err(Error::InvalidFormat(
            "integer payload is wider than u64".into(),
        ));
    }
    Ok(data.iter().enumerate().fold(0u64, |value, (idx, byte)| {
        value | (u64::from(*byte) << (idx * 8))
    }))
}

pub(super) fn is_undefined_addr(addr: u64, sizeof_addr: u8) -> Result<bool> {
    let bits = u32::from(sizeof_addr)
        .checked_mul(8)
        .ok_or_else(|| Error::InvalidFormat("address size overflow".into()))?;
    let undef = if bits == 64 {
        u64::MAX
    } else if bits < 64 {
        (1u64 << bits) - 1
    } else {
        return Err(Error::InvalidFormat(
            "address payload is wider than u64".into(),
        ));
    };
    Ok(addr == undef)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::BufReader;

    #[test]
    fn test_parse_v0_root_object_header() {
        let f = File::open("tests/data/simple_v0.h5").unwrap();
        let mut reader = HdfReader::new(BufReader::new(f));
        let sb = crate::format::superblock::Superblock::read(&mut reader).unwrap();

        let oh = ObjectHeader::read_at(&mut reader, sb.root_addr).unwrap();
        println!(
            "v0 root OH: version={}, refcount={}, messages:",
            oh.version, oh.refcount
        );
        for msg in &oh.messages {
            println!(
                "  type={:#06x}, flags={:#04x}, len={}",
                msg.msg_type,
                msg.flags,
                msg.data.len()
            );
        }

        assert_eq!(oh.version, 1);
        // Root group should have a symbol table message
        assert!(oh.messages.iter().any(|m| m.msg_type == MSG_SYMBOL_TABLE));
    }

    #[test]
    fn test_parse_v3_root_object_header() {
        let f = File::open("tests/data/simple_v2.h5").unwrap();
        let mut reader = HdfReader::new(BufReader::new(f));
        let sb = crate::format::superblock::Superblock::read(&mut reader).unwrap();

        let oh = ObjectHeader::read_at(&mut reader, sb.root_addr).unwrap();
        println!(
            "v3 root OH: version={}, flags={:#04x}, messages:",
            oh.version, oh.flags
        );
        for msg in &oh.messages {
            println!(
                "  type={:#06x}, flags={:#04x}, len={}",
                msg.msg_type,
                msg.flags,
                msg.data.len()
            );
        }

        assert_eq!(oh.version, 2);
        // V2 root group should have link messages or link info
        let has_links = oh
            .messages
            .iter()
            .any(|m| m.msg_type == MSG_LINK || m.msg_type == MSG_LINK_INFO);
        assert!(
            has_links,
            "v2 root group should have link or link info messages"
        );
    }
}
