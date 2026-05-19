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
use crate::format::checksum::checksum_metadata;
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
pub const MSG_MDCI: u16 = 0x0018;

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

/// Message input for v2 object-header encoding.
#[derive(Debug, Clone, Copy)]
pub struct ObjectHeaderMessageRef<'a> {
    /// Message type ID. V2 compact object headers store this in one byte.
    pub msg_type: u16,
    /// Message flags.
    pub flags: u8,
    /// Creation order index, present only when the object header tracks it.
    pub creation_index: Option<u16>,
    /// Raw message payload bytes.
    pub data: &'a [u8],
}

/// Encoded v2 object header and any continuation chunks it references.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedV2ObjectHeader {
    /// The object-header prefix plus chunk 0 and checksum.
    pub prefix: Vec<u8>,
    /// Continuation chunks as `(address, image)` pairs. Each image includes
    /// the OCHK magic and trailing checksum.
    pub continuation_chunks: Vec<(u64, Vec<u8>)>,
}

impl ObjectHeader {
    /// Read an object header at the given file address. Mirrors the
    /// dispatcher in libhdf5's `H5O_protect`: peek at the first 4 bytes
    /// to decide whether this is a v2 (OHDR magic) or v1 (version byte)
    /// header, then hand off to the version-specific decoder in `cache.rs`.
    pub fn read_at<R: Read + Seek>(reader: &mut HdfReader<R>, addr: u64) -> Result<Self> {
        reader.seek(addr)?;

        let mut first_bytes = [0u8; 4];
        reader.read_bytes_into(&mut first_bytes)?;

        let result = if first_bytes == OHDR_MAGIC {
            Self::read_v2(reader, addr)
        } else {
            // Seek back and re-read as v1. The first byte is the version.
            reader.seek(addr)?;
            Self::read_v1(reader)
        };

        #[cfg(feature = "tracehash")]
        if let Ok(header) = &result {
            let mut th = tracehash::th_call!("hdf5.object_header.read");
            th.input_u64(addr);
            let message_count = header
                .messages
                .iter()
                .filter(|message| message.chunk_index == 0)
                .count();
            let Ok(message_count) = u64::try_from(message_count) else {
                return result;
            };
            th.output_u64(u64::from(header.version));
            th.output_u64(u64::from(header.flags));
            th.output_u64(u64::from(header.refcount));
            th.output_u64(message_count);
            for message in header
                .messages
                .iter()
                .filter(|message| message.chunk_index == 0)
            {
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

/// Encode a v2 object header, splitting messages across continuation chunks
/// when they do not fit in chunk 0. This mirrors the HDF5 object-header chunk
/// model: chunk 0 stores regular messages plus a HEADER_CONTINUATION message;
/// each continuation chunk starts with OCHK, stores more messages, and carries
/// its own checksum.
#[allow(clippy::too_many_arguments)]
pub fn encode_v2_with_continuations(
    messages: &[ObjectHeaderMessageRef<'_>],
    flags: u8,
    continuation_addrs: &[u64],
    chunk0_data_limit: usize,
    continuation_data_limit: usize,
    sizeof_addr: u8,
    sizeof_size: u8,
) -> Result<EncodedV2ObjectHeader> {
    if flags & !HDR_V2_KNOWN_FLAGS != 0 {
        return Err(Error::InvalidFormat(format!(
            "object header v2 flags contain reserved bits: {flags:#04x}"
        )));
    }
    let has_crt_order = flags & HDR_ATTR_CRT_ORDER_TRACKED != 0;
    let mut encoded_messages = Vec::with_capacity(messages.len());
    for message in messages {
        encoded_messages.push(encode_v2_message(message, has_crt_order)?);
    }

    let continuation_payload_len = checked_usize_add(
        usize::from(sizeof_addr),
        usize::from(sizeof_size),
        "object header continuation payload",
    )?;
    let continuation_message_len = checked_usize_add(
        v2_message_header_len(has_crt_order),
        continuation_payload_len,
        "object header continuation message",
    )?;

    let chunk_payloads = pack_v2_message_chunks(
        &encoded_messages,
        chunk0_data_limit,
        continuation_data_limit,
        continuation_message_len,
    )?;
    let continuation_count = chunk_payloads.len().saturating_sub(1);
    if continuation_addrs.len() != continuation_count {
        return Err(Error::InvalidFormat(format!(
            "object header encoder needs {continuation_count} continuation addresses, got {}",
            continuation_addrs.len()
        )));
    }

    let mut chunk0_data = chunk_payloads[0].clone();
    if continuation_count > 0 {
        append_v2_continuation_message(
            &mut chunk0_data,
            continuation_addrs[0],
            continuation_chunk_file_len(final_continuation_data_len(
                &chunk_payloads,
                1,
                continuation_message_len,
            )?)?,
            has_crt_order,
            sizeof_addr,
            sizeof_size,
        )?;
    }

    let chunk0_size_len = chunk0_size_len(chunk0_data.len())?;
    let chunk0_flag = match chunk0_size_len {
        1 => 0,
        2 => 1,
        4 => 2,
        8 => 3,
        _ => unreachable!(),
    };
    let oh_flags = flags | chunk0_flag;
    let mut prefix = Vec::new();
    prefix.extend_from_slice(&OHDR_MAGIC);
    prefix.push(2);
    prefix.push(oh_flags);
    write_le_uint_width(&mut prefix, chunk0_data.len() as u64, chunk0_size_len)?;
    prefix.extend_from_slice(&chunk0_data);
    let checksum = checksum_metadata(&prefix);
    prefix.extend_from_slice(&checksum.to_le_bytes());

    let mut continuation_chunks = Vec::with_capacity(continuation_count);
    for idx in 0..continuation_count {
        let mut chunk_data = chunk_payloads[idx + 1].clone();
        if idx + 1 < continuation_count {
            append_v2_continuation_message(
                &mut chunk_data,
                continuation_addrs[idx + 1],
                continuation_chunk_file_len(final_continuation_data_len(
                    &chunk_payloads,
                    idx + 2,
                    continuation_message_len,
                )?)?,
                has_crt_order,
                sizeof_addr,
                sizeof_size,
            )?;
        }
        let mut image = Vec::new();
        image.extend_from_slice(&OCHK_MAGIC);
        image.extend_from_slice(&chunk_data);
        let checksum = checksum_metadata(&image);
        image.extend_from_slice(&checksum.to_le_bytes());
        continuation_chunks.push((continuation_addrs[idx], image));
    }

    Ok(EncodedV2ObjectHeader {
        prefix,
        continuation_chunks,
    })
}

fn encode_v2_message(message: &ObjectHeaderMessageRef<'_>, has_crt_order: bool) -> Result<Vec<u8>> {
    if u8::try_from(message.msg_type).is_err() {
        return Err(Error::InvalidFormat(format!(
            "object-header message type {:#06x} exceeds v2 compact encoding",
            message.msg_type
        )));
    }
    if u16::try_from(message.data.len()).is_err() {
        return Err(Error::InvalidFormat(format!(
            "object-header message {:#06x} exceeds v2 message size",
            message.msg_type
        )));
    }
    let mut out = Vec::with_capacity(
        v2_message_header_len(has_crt_order)
            .checked_add(message.data.len())
            .ok_or_else(|| Error::InvalidFormat("object header message size overflow".into()))?,
    );
    out.push(message.msg_type as u8);
    out.extend_from_slice(&(message.data.len() as u16).to_le_bytes());
    out.push(message.flags);
    if has_crt_order {
        let creation_index = message.creation_index.ok_or_else(|| {
            Error::InvalidFormat("object header message is missing creation order".into())
        })?;
        out.extend_from_slice(&creation_index.to_le_bytes());
    } else if message.creation_index.is_some() {
        return Err(Error::InvalidFormat(
            "object header message has creation order but header does not track it".into(),
        ));
    }
    out.extend_from_slice(message.data);
    Ok(out)
}

fn pack_v2_message_chunks(
    encoded_messages: &[Vec<u8>],
    chunk0_data_limit: usize,
    continuation_data_limit: usize,
    continuation_message_len: usize,
) -> Result<Vec<Vec<u8>>> {
    let mut chunks = Vec::new();
    let mut index = 0usize;
    loop {
        let limit = if chunks.is_empty() {
            chunk0_data_limit
        } else {
            continuation_data_limit
        };
        let remaining_size =
            encoded_messages[index..]
                .iter()
                .try_fold(0usize, |sum, message| {
                    sum.checked_add(message.len()).ok_or_else(|| {
                        Error::InvalidFormat("object header chunk size overflow".into())
                    })
                })?;
        if remaining_size <= limit {
            let mut chunk = Vec::with_capacity(remaining_size);
            for message in &encoded_messages[index..] {
                chunk.extend_from_slice(message);
            }
            chunks.push(chunk);
            break;
        }

        if limit < continuation_message_len {
            return Err(Error::InvalidFormat(
                "object header chunk cannot fit continuation message".into(),
            ));
        }
        let payload_limit = limit - continuation_message_len;
        let mut chunk = Vec::new();
        while index < encoded_messages.len()
            && chunk.len().saturating_add(encoded_messages[index].len()) <= payload_limit
        {
            chunk.extend_from_slice(&encoded_messages[index]);
            index += 1;
        }
        if chunk.is_empty() && chunks.is_empty() {
            chunks.push(chunk);
            continue;
        }
        if chunk.is_empty() {
            return Err(Error::InvalidFormat(
                "object header message cannot fit before a required continuation message".into(),
            ));
        }
        chunks.push(chunk);
    }
    Ok(chunks)
}

fn append_v2_continuation_message(
    out: &mut Vec<u8>,
    addr: u64,
    size: u64,
    has_crt_order: bool,
    sizeof_addr: u8,
    sizeof_size: u8,
) -> Result<()> {
    let mut payload = Vec::new();
    write_le_uint_width(&mut payload, addr, usize::from(sizeof_addr))?;
    write_le_uint_width(&mut payload, size, usize::from(sizeof_size))?;
    let message = ObjectHeaderMessageRef {
        msg_type: MSG_HEADER_CONTINUATION,
        flags: 0,
        creation_index: has_crt_order.then_some(0),
        data: &payload,
    };
    out.extend_from_slice(&encode_v2_message(&message, has_crt_order)?);
    Ok(())
}

fn continuation_chunk_file_len(data_len: usize) -> Result<u64> {
    let total = checked_usize_add(4, data_len, "object header continuation chunk")?;
    let total = checked_usize_add(total, 4, "object header continuation checksum")?;
    u64::try_from(total)
        .map_err(|_| Error::InvalidFormat("object header continuation chunk too large".into()))
}

fn final_continuation_data_len(
    chunk_payloads: &[Vec<u8>],
    chunk_index: usize,
    continuation_message_len: usize,
) -> Result<usize> {
    let mut len = chunk_payloads
        .get(chunk_index)
        .ok_or_else(|| Error::InvalidFormat("object header continuation chunk missing".into()))?
        .len();
    if chunk_index + 1 < chunk_payloads.len() {
        len = checked_usize_add(
            len,
            continuation_message_len,
            "object header continuation data",
        )?;
    }
    Ok(len)
}

fn chunk0_size_len(chunk_data_size: usize) -> Result<usize> {
    if chunk_data_size <= u8::MAX as usize {
        Ok(1)
    } else if chunk_data_size <= u16::MAX as usize {
        Ok(2)
    } else if chunk_data_size <= u32::MAX as usize {
        Ok(4)
    } else {
        Ok(8)
    }
}

fn v2_message_header_len(has_crt_order: bool) -> usize {
    if has_crt_order {
        6
    } else {
        4
    }
}

fn checked_usize_add(lhs: usize, rhs: usize, context: &str) -> Result<usize> {
    lhs.checked_add(rhs)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} size overflow")))
}

fn write_le_uint_width(out: &mut Vec<u8>, value: u64, width: usize) -> Result<()> {
    if width == 0 || width > 8 {
        return Err(Error::InvalidFormat(format!(
            "unsupported integer width: {width}"
        )));
    }
    if width < 8 && value >= (1u64 << (width * 8)) {
        return Err(Error::InvalidFormat(
            "integer value does not fit configured width".into(),
        ));
    }
    out.extend_from_slice(&value.to_le_bytes()[..width]);
    Ok(())
}

// ---------------------------------------------------------------------------
// Internal numeric helpers shared across submodules.
// ---------------------------------------------------------------------------

/// Decode a little-endian unsigned integer of up to 8 bytes from `data`.
/// Mirrors libhdf5's `H5F_DECODE_LENGTH` family of macros: the on-disk length
/// fields use a configurable width and are always stored LSB-first.
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

/// True when `addr` equals the "undefined address" sentinel for the file's
/// configured address width. Equivalent to libhdf5's `H5F_addr_defined` /
/// `HADDR_UNDEF` check, generalized to arbitrary `sizeof_addr` values.
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
    use std::io::Cursor;

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

    #[test]
    fn encode_v2_with_continuations_roundtrips_nested_chunks() {
        let payloads = [
            b"a", b"b", b"c", b"d", b"e", b"f", b"g", b"h", b"i", b"j", b"k", b"l",
        ];
        let messages = payloads.map(|data| ObjectHeaderMessageRef {
            msg_type: MSG_OBJ_COMMENT,
            flags: 0,
            creation_index: None,
            data,
        });
        let continuation_addrs = [64, 128, 192, 256, 320, 384, 448];
        let encoded =
            encode_v2_with_continuations(&messages, 0, &continuation_addrs, 25, 25, 8, 8).unwrap();

        assert_eq!(encoded.continuation_chunks.len(), continuation_addrs.len());
        assert!(encoded
            .prefix
            .iter()
            .any(|byte| *byte == MSG_HEADER_CONTINUATION as u8));
        assert!(encoded.continuation_chunks[0]
            .1
            .iter()
            .any(|byte| *byte == MSG_HEADER_CONTINUATION as u8));

        let mut file = vec![0u8; 512];
        file[..encoded.prefix.len()].copy_from_slice(&encoded.prefix);
        for (addr, image) in &encoded.continuation_chunks {
            let start = usize::try_from(*addr).unwrap();
            file[start..start + image.len()].copy_from_slice(image);
        }

        let mut reader = HdfReader::new(Cursor::new(file));
        let header = ObjectHeader::read_at(&mut reader, 0).unwrap();
        assert_eq!(header.version, 2);
        assert_eq!(header.messages.len(), messages.len());
        assert_eq!(
            header
                .messages
                .iter()
                .map(|message| message.data.as_slice())
                .collect::<Vec<_>>(),
            payloads
                .iter()
                .map(|payload| &payload[..])
                .collect::<Vec<_>>()
        );
        assert!(header
            .messages
            .iter()
            .any(|message| message.chunk_index > 1));
    }

    #[test]
    fn encode_v2_with_continuations_requires_exact_addresses() {
        let messages = [
            b"a", b"b", b"c", b"d", b"e", b"f", b"g", b"h", b"i", b"j", b"k", b"l",
        ]
        .map(|data| ObjectHeaderMessageRef {
            msg_type: MSG_OBJ_COMMENT,
            flags: 0,
            creation_index: None,
            data,
        });

        let err = encode_v2_with_continuations(&messages, 0, &[], 25, 25, 8, 8).unwrap_err();
        assert!(err.to_string().contains("continuation addresses"));
    }
}
