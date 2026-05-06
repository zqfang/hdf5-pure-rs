//! Object header message decoding — mirrors libhdf5's `H5Omessage.c` plus
//! the per-message decode helpers in `H5Oint.c`. Iterates the on-disk
//! message stream within one chunk, dispatches NIL / HEADER_CONTINUATION
//! to the chunk layer, and forwards real messages back to the caller.

use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::format::messages::attribute::AttributeMessage;
use crate::format::messages::attribute_info::AttributeInfoMessage;
use crate::format::messages::data_layout::DataLayoutMessage;
use crate::format::messages::dataspace::DataspaceMessage;
use crate::format::messages::datatype::DatatypeMessage;
use crate::format::messages::fill_value::FillValueMessage;
use crate::format::messages::filter_pipeline::FilterPipelineMessage;
use crate::format::messages::link::LinkMessage;
use crate::format::messages::link_info::LinkInfoMessage;
use crate::format::messages::symbol_table::SymbolTableMessage;
use crate::io::reader::HdfReader;

use super::chunk::reserve_continuation_range;
use super::{
    is_undefined_addr, read_le_uint, RawMessage, MSG_ATTRIBUTE, MSG_ATTR_INFO, MSG_BTREE_K,
    MSG_DATASPACE, MSG_DATATYPE, MSG_EXTERNAL_FILE_LIST, MSG_FILL_VALUE, MSG_FILL_VALUE_OLD,
    MSG_FILTER_PIPELINE, MSG_FLAG_SHARED, MSG_GROUP_INFO, MSG_HEADER_CONTINUATION, MSG_LAYOUT,
    MSG_LINK, MSG_LINK_INFO, MSG_NIL, MSG_OBJ_REF_COUNT, MSG_SHARED_MSG_TABLE, MSG_SYMBOL_TABLE,
    SHARED_HEAP_ID_LEN, SHARED_MESSAGE_MAX_INDEXES, SHARED_MESSAGE_TABLE_VERSION,
    SHARED_REFERENCE_VERSION_1, SHARED_REFERENCE_VERSION_2, SHARED_REFERENCE_VERSION_3,
    SHARED_TYPE_COMMITTED, SHARED_TYPE_SOHM,
};

const MSG_FLAG_DONTSHARE: u8 = 0x04;
const MSG_FLAG_FAIL_IF_UNKNOWN_AND_OPEN_FOR_WRITE: u8 = 0x08;
const MSG_FLAG_MARK_IF_UNKNOWN: u8 = 0x10;
const MSG_FLAG_WAS_UNKNOWN: u8 = 0x20;
const MSG_KNOWN_FLAGS: u8 = 0xff;

#[allow(clippy::too_many_arguments)]
pub(super) fn read_v1_messages<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    chunk_end: u64,
    _num_messages: u16,
    messages: &mut Vec<RawMessage>,
    continuations: &mut Vec<(u64, u64)>,
    chunk_ranges: &mut Vec<(u64, u64)>,
    chunk_index: u16,
) -> Result<()> {
    while reader.position()? < chunk_end {
        let pos = reader.position()?;
        let header_end = checked_u64_add(pos, 8, "object header v1 message header")?;
        if header_end > chunk_end {
            let remaining = usize_from_u64(chunk_end - pos, "object header v1 message padding")?;
            let padding = reader.read_bytes(remaining)?;
            if padding.iter().all(|&b| b == 0) {
                break;
            }
            return Err(Error::InvalidFormat(
                "object header v1 message header is truncated".into(),
            ));
        }

        let msg_type = reader.read_u16()?;
        let msg_size = u64::from(reader.read_u16()?);
        let msg_flags = reader.read_u8()?;
        validate_message_flags(msg_flags)?;
        reader.skip(3)?;

        // Aligned message size (v1 messages are 8-byte aligned)
        let aligned_size = msg_size
            .checked_add(7)
            .map(|n| n & !7)
            .ok_or_else(|| Error::InvalidFormat("object header message size overflow".into()))?;
        let data_start = checked_u64_add(pos, 8, "object header v1 message data start")?;
        let data_end = data_start
            .checked_add(aligned_size)
            .ok_or_else(|| Error::InvalidFormat("object header message range overflow".into()))?;
        if data_end > chunk_end {
            return Err(Error::InvalidFormat(
                "object header v1 message payload exceeds chunk".into(),
            ));
        }

        if msg_type == MSG_NIL {
            reader.skip(aligned_size)?;
            continue;
        }

        if msg_type == MSG_HEADER_CONTINUATION {
            // Continuation message: contains offset + length
            let used = u64::from(reader.sizeof_addr())
                .checked_add(u64::from(reader.sizeof_size()))
                .ok_or_else(|| {
                    Error::InvalidFormat("object header continuation width overflow".into())
                })?;
            if msg_size < used {
                return Err(Error::InvalidFormat(
                    "object header continuation message is truncated".into(),
                ));
            }
            let cont_offset = reader.read_addr()?;
            let cont_length = reader.read_length()?;
            reserve_continuation_range(reader, cont_offset, cont_length, 8, chunk_ranges)?;
            let remaining = aligned_size - used;
            if remaining > 0 {
                reader.skip(remaining)?;
            }
            continuations.push((cont_offset, cont_length));
            continue;
        }

        let data = reader.read_bytes(usize_from_u64(msg_size, "object header message size")?)?;
        // Skip padding to alignment
        let padding = aligned_size - msg_size;
        if padding > 0 {
            reader.skip(padding)?;
        }
        validate_message_payload(
            msg_type,
            msg_flags,
            &data,
            reader.sizeof_addr(),
            reader.sizeof_size(),
        )?;

        messages.push(RawMessage {
            msg_type,
            flags: msg_flags,
            creation_index: None,
            chunk_index,
            data,
        });
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn read_v2_messages<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    chunk_data_end: u64,
    has_crt_order: bool,
    messages: &mut Vec<RawMessage>,
    continuations: &mut Vec<(u64, u64)>,
    chunk_ranges: &mut Vec<(u64, u64)>,
    chunk_index: u16,
) -> Result<()> {
    while reader.position()? < chunk_data_end {
        let pos = reader.position()?;
        // Minimum message header size: 4 bytes (type:1, size:2, flags:1)
        let header_end = checked_u64_add(pos, 4, "object header v2 message header")?;
        if header_end > chunk_data_end {
            let remaining =
                usize_from_u64(chunk_data_end - pos, "object header v2 message padding")?;
            let padding = reader.read_bytes(remaining)?;
            if padding.iter().all(|&b| b == 0) {
                break;
            }
            return Err(Error::InvalidFormat(
                "object header v2 message header is truncated".into(),
            ));
        }

        let msg_type = u16::from(reader.read_u8()?);
        let msg_size = u64::from(reader.read_u16()?);
        let msg_flags = reader.read_u8()?;
        validate_message_flags(msg_flags)?;

        let creation_index = if has_crt_order {
            let creation_order_end = checked_u64_add(
                reader.position()?,
                2,
                "object header v2 message creation order",
            )?;
            if creation_order_end > chunk_data_end {
                return Err(Error::InvalidFormat(
                    "object header v2 message creation order is truncated".into(),
                ));
            }
            Some(reader.read_u16()?)
        } else {
            None
        };
        let data_start = reader.position()?;
        let data_end = data_start.checked_add(msg_size).ok_or_else(|| {
            Error::InvalidFormat("object header v2 message range overflow".into())
        })?;
        if data_end > chunk_data_end {
            return Err(Error::InvalidFormat(
                "object header v2 message payload exceeds chunk".into(),
            ));
        }

        if msg_type == MSG_NIL {
            reader.skip(msg_size)?;
            continue;
        }

        if msg_type == MSG_HEADER_CONTINUATION {
            let used = u64::from(reader.sizeof_addr())
                .checked_add(u64::from(reader.sizeof_size()))
                .ok_or_else(|| {
                    Error::InvalidFormat("object header continuation width overflow".into())
                })?;
            if msg_size < used {
                return Err(Error::InvalidFormat(
                    "object header continuation message is truncated".into(),
                ));
            }
            let cont_offset = reader.read_addr()?;
            let cont_length = reader.read_length()?;
            reserve_continuation_range(reader, cont_offset, cont_length, 8, chunk_ranges)?;
            if msg_size > used {
                reader.skip(msg_size - used)?;
            }
            continuations.push((cont_offset, cont_length));
            continue;
        }

        let data = reader.read_bytes(usize_from_u64(msg_size, "object header message size")?)?;
        validate_message_payload(
            msg_type,
            msg_flags,
            &data,
            reader.sizeof_addr(),
            reader.sizeof_size(),
        )?;

        messages.push(RawMessage {
            msg_type,
            flags: msg_flags,
            creation_index,
            chunk_index,
            data,
        });
    }

    Ok(())
}

fn validate_message_payload(
    msg_type: u16,
    msg_flags: u8,
    data: &[u8],
    sizeof_addr: u8,
    sizeof_size: u8,
) -> Result<()> {
    if msg_type == MSG_SHARED_MSG_TABLE {
        validate_shared_message_table(data, sizeof_addr)?;
    }
    if msg_type == MSG_GROUP_INFO {
        validate_group_info_message(data)?;
    }
    if msg_type == MSG_BTREE_K {
        validate_btree_k_message(data)?;
    }
    if msg_flags & MSG_FLAG_SHARED == 0 {
        if msg_type == MSG_DATASPACE {
            DataspaceMessage::decode(data)?;
        }
        if msg_type == MSG_LINK_INFO {
            LinkInfoMessage::decode(data, sizeof_addr)?;
        }
        if msg_type == MSG_DATATYPE {
            DatatypeMessage::decode(data)?;
        }
        if msg_type == MSG_FILL_VALUE {
            FillValueMessage::decode(data)?;
        }
        if msg_type == MSG_FILL_VALUE_OLD {
            FillValueMessage::decode_old(data)?;
        }
        if msg_type == MSG_FILTER_PIPELINE {
            FilterPipelineMessage::decode(data)?;
        }
        if msg_type == MSG_ATTRIBUTE {
            AttributeMessage::decode(data)?;
        }
        if msg_type == MSG_LINK {
            LinkMessage::decode(data, sizeof_addr)?;
        }
        if msg_type == MSG_LAYOUT {
            DataLayoutMessage::decode(data, sizeof_addr, sizeof_size)?;
        }
        if msg_type == MSG_EXTERNAL_FILE_LIST {
            validate_external_file_list(data, sizeof_addr, sizeof_size)?;
        }
        if msg_type == MSG_SYMBOL_TABLE {
            SymbolTableMessage::decode(data, sizeof_addr)?;
        }
        if msg_type == MSG_ATTR_INFO {
            AttributeInfoMessage::decode(data, sizeof_addr)?;
        }
    }
    if msg_type == MSG_OBJ_REF_COUNT {
        validate_refcount_message(data)?;
    }
    if msg_flags & MSG_FLAG_SHARED != 0 {
        validate_shared_message_reference(data, sizeof_addr, sizeof_size)?;
    }
    Ok(())
}

fn validate_group_info_message(data: &[u8]) -> Result<()> {
    if data.len() < 2 {
        return Err(Error::InvalidFormat(
            "group info message is truncated".into(),
        ));
    }
    let version = data[0];
    if version != 0 {
        return Err(Error::InvalidFormat(format!(
            "group info message version {version}"
        )));
    }
    let flags = data[1];
    if flags & !0x03 != 0 {
        return Err(Error::InvalidFormat(format!(
            "group info message flags {flags:#x} are invalid"
        )));
    }
    let needed = 2usize
        .checked_add(if flags & 0x01 != 0 { 4 } else { 0 })
        .and_then(|value| value.checked_add(if flags & 0x02 != 0 { 4 } else { 0 }))
        .ok_or_else(|| Error::InvalidFormat("group info message size overflow".into()))?;
    if data.len() < needed {
        return Err(Error::InvalidFormat(
            "group info message is truncated".into(),
        ));
    }
    Ok(())
}

fn validate_btree_k_message(data: &[u8]) -> Result<()> {
    if data.len() < 7 {
        return Err(Error::InvalidFormat(
            "B-tree K values message is truncated".into(),
        ));
    }
    let version = data[0];
    if version != 0 {
        return Err(Error::InvalidFormat(format!(
            "B-tree K values message version {version}"
        )));
    }
    Ok(())
}

fn validate_refcount_message(data: &[u8]) -> Result<()> {
    if data.len() < 4 {
        return Err(Error::InvalidFormat(
            "object refcount message is truncated".into(),
        ));
    }
    Ok(())
}

fn validate_external_file_list(data: &[u8], sizeof_addr: u8, sizeof_size: u8) -> Result<()> {
    let mut pos = 0usize;
    let version = read_u8_at(data, &mut pos, "external file list version")?;
    if version != 1 {
        return Err(Error::Unsupported(format!(
            "external file list version {version}"
        )));
    }
    pos = checked_usize_add(pos, 3, "external file list reserved bytes")?;
    let allocated_slots = read_le_uint_at(data, &mut pos, 2, "external file list allocated slots")?;
    if allocated_slots == 0 {
        return Err(Error::InvalidFormat(
            "external file list has no allocated slots".into(),
        ));
    }
    let used_slots = read_le_uint_at(data, &mut pos, 2, "external file list used slots")?;
    if used_slots > allocated_slots {
        return Err(Error::InvalidFormat(
            "external file list uses more slots than allocated".into(),
        ));
    }
    let heap_addr = read_le_uint_at(
        data,
        &mut pos,
        usize::from(sizeof_addr),
        "external file list heap address",
    )?;
    if is_undefined_addr(heap_addr, sizeof_addr)? {
        return Err(Error::InvalidFormat(
            "external file list heap address is undefined".into(),
        ));
    }
    for _ in 0..used_slots {
        read_le_uint_at(
            data,
            &mut pos,
            usize::from(sizeof_size),
            "external file list name offset",
        )?;
        read_le_uint_at(
            data,
            &mut pos,
            usize::from(sizeof_size),
            "external file list file offset",
        )?;
        read_le_uint_at(
            data,
            &mut pos,
            usize::from(sizeof_size),
            "external file list size",
        )?;
    }
    Ok(())
}

fn validate_message_flags(msg_flags: u8) -> Result<()> {
    if msg_flags & !MSG_KNOWN_FLAGS != 0 {
        return Err(Error::InvalidFormat(format!(
            "object header message flags contain reserved bits: {msg_flags:#04x}"
        )));
    }
    if msg_flags & MSG_FLAG_SHARED != 0 && msg_flags & MSG_FLAG_DONTSHARE != 0 {
        return Err(Error::InvalidFormat(
            "object header message flags contain shared and do-not-share".into(),
        ));
    }
    if msg_flags & MSG_FLAG_WAS_UNKNOWN != 0
        && msg_flags & MSG_FLAG_FAIL_IF_UNKNOWN_AND_OPEN_FOR_WRITE != 0
    {
        return Err(Error::InvalidFormat(
            "object header message flags contain was-unknown and fail-if-unknown-on-write".into(),
        ));
    }
    if msg_flags & MSG_FLAG_WAS_UNKNOWN != 0 && msg_flags & MSG_FLAG_MARK_IF_UNKNOWN == 0 {
        return Err(Error::InvalidFormat(
            "object header message flags contain was-unknown without mark-if-unknown".into(),
        ));
    }
    Ok(())
}

fn validate_shared_message_table(data: &[u8], sizeof_addr: u8) -> Result<()> {
    let expected_len = 1usize
        .checked_add(usize::from(sizeof_addr))
        .and_then(|len| len.checked_add(1))
        .ok_or_else(|| Error::InvalidFormat("shared message table size overflow".into()))?;
    if data.len() < expected_len {
        return Err(Error::InvalidFormat(
            "shared message table payload is truncated".into(),
        ));
    }
    let version = data[0];
    if version != SHARED_MESSAGE_TABLE_VERSION {
        return Err(Error::InvalidFormat(format!(
            "unsupported shared message table version: {version}"
        )));
    }
    let addr_end = checked_usize_add(1, usize::from(sizeof_addr), "shared message table address")?;
    let table_addr = read_le_uint(&data[1..addr_end])?;
    if is_undefined_addr(table_addr, sizeof_addr)? {
        return Err(Error::InvalidFormat(
            "shared message table address is undefined".into(),
        ));
    }
    let nindexes = data[addr_end];
    if nindexes == 0 || nindexes > SHARED_MESSAGE_MAX_INDEXES {
        return Err(Error::InvalidFormat(
            "shared message table index count is invalid".into(),
        ));
    }
    Ok(())
}

fn validate_shared_message_reference(data: &[u8], sizeof_addr: u8, sizeof_size: u8) -> Result<()> {
    if data.len() < 2 {
        return Err(Error::InvalidFormat(
            "shared object-header message reference is truncated".into(),
        ));
    }

    let version = data[0];
    match version {
        SHARED_REFERENCE_VERSION_1 => {
            let expected_len = 2usize
                .checked_add(6)
                .and_then(|len| len.checked_add(usize::from(sizeof_size)))
                .and_then(|len| len.checked_add(usize::from(sizeof_addr)))
                .ok_or_else(|| {
                    Error::InvalidFormat("shared message reference size overflow".into())
                })?;
            if data.len() < expected_len {
                return Err(Error::InvalidFormat(
                    "shared object-header message v1 reference is truncated".into(),
                ));
            }
            let addr_start = checked_usize_add(
                checked_usize_add(2, 6, "shared message v1 reference prefix")?,
                usize::from(sizeof_size),
                "shared message v1 reference address",
            )?;
            let addr_end = checked_usize_add(
                addr_start,
                usize::from(sizeof_addr),
                "shared message v1 address",
            )?;
            let addr = read_le_uint(&data[addr_start..addr_end])?;
            if is_undefined_addr(addr, sizeof_addr)? {
                return Err(Error::InvalidFormat(
                    "shared object-header message address is undefined".into(),
                ));
            }
        }
        SHARED_REFERENCE_VERSION_2 => {
            let expected_len = 2usize
                .checked_add(usize::from(sizeof_addr))
                .ok_or_else(|| {
                    Error::InvalidFormat("shared message reference size overflow".into())
                })?;
            if data.len() < expected_len {
                return Err(Error::InvalidFormat(
                    "shared object-header message v2 reference is truncated".into(),
                ));
            }
            let addr_end =
                checked_usize_add(2, usize::from(sizeof_addr), "shared message v2 address")?;
            let addr = read_le_uint(&data[2..addr_end])?;
            if is_undefined_addr(addr, sizeof_addr)? {
                return Err(Error::InvalidFormat(
                    "shared object-header message address is undefined".into(),
                ));
            }
        }
        SHARED_REFERENCE_VERSION_3 => match data[1] {
            SHARED_TYPE_SOHM => {
                let expected_len =
                    checked_usize_add(2, SHARED_HEAP_ID_LEN, "shared SOHM reference")?;
                if data.len() < expected_len {
                    return Err(Error::InvalidFormat(
                        "shared object-header message SOHM reference is truncated".into(),
                    ));
                }
            }
            SHARED_TYPE_COMMITTED => {
                let expected_len =
                    2usize
                        .checked_add(usize::from(sizeof_addr))
                        .ok_or_else(|| {
                            Error::InvalidFormat("shared message reference size overflow".into())
                        })?;
                if data.len() < expected_len {
                    return Err(Error::InvalidFormat(
                        "shared object-header message committed reference is truncated".into(),
                    ));
                }
                let addr_end =
                    checked_usize_add(2, usize::from(sizeof_addr), "shared message v3 address")?;
                let addr = read_le_uint(&data[2..addr_end])?;
                if is_undefined_addr(addr, sizeof_addr)? {
                    return Err(Error::InvalidFormat(
                        "shared object-header message address is undefined".into(),
                    ));
                }
            }
            _ => {
                return Err(Error::InvalidFormat(
                    "shared object-header message type is invalid".into(),
                ));
            }
        },
        _ => {
            return Err(Error::InvalidFormat(
                "shared object-header message version is invalid".into(),
            ));
        }
    }

    Ok(())
}

fn checked_u64_add(lhs: u64, rhs: u64, context: &str) -> Result<u64> {
    lhs.checked_add(rhs)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))
}

fn checked_usize_add(lhs: usize, rhs: usize, context: &str) -> Result<usize> {
    lhs.checked_add(rhs)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} size overflow")))
}

fn usize_from_u64(value: u64, context: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| Error::InvalidFormat(format!("{context} exceeds usize")))
}

fn read_u8_at(data: &[u8], pos: &mut usize, context: &str) -> Result<u8> {
    let value = data
        .get(*pos)
        .copied()
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))?;
    *pos = checked_usize_add(*pos, 1, context)?;
    Ok(value)
}

fn read_le_uint_at(data: &[u8], pos: &mut usize, width: usize, context: &str) -> Result<u64> {
    if !(1..=8).contains(&width) {
        return Err(Error::InvalidFormat(format!(
            "{context} width {width} is invalid"
        )));
    }
    let end = checked_usize_add(*pos, width, context)?;
    let value = read_le_uint(
        data.get(*pos..end)
            .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))?,
    )?;
    *pos = end;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use crate::io::reader::HdfReader;

    use super::*;
    use super::{checked_u64_add, checked_usize_add};

    #[test]
    fn object_header_checked_u64_add_rejects_overflow() {
        let err = checked_u64_add(u64::MAX, 1, "message header").unwrap_err();
        assert!(err.to_string().contains("overflow"));
    }

    #[test]
    fn object_header_checked_usize_add_rejects_overflow() {
        let err = checked_usize_add(usize::MAX, 1, "message size").unwrap_err();
        assert!(err.to_string().contains("overflow"));
    }

    #[test]
    fn object_header_messages_reject_bad_flag_combinations() {
        let mut messages = Vec::new();
        let mut continuations = Vec::new();
        let mut ranges = vec![(0, 8)];
        let mut v1 = Vec::new();
        v1.extend_from_slice(&1u16.to_le_bytes());
        v1.extend_from_slice(&0u16.to_le_bytes());
        v1.push(MSG_FLAG_SHARED | MSG_FLAG_DONTSHARE);
        v1.extend_from_slice(&[0; 3]);
        let mut reader = HdfReader::new(Cursor::new(v1));
        let err = read_v1_messages(
            &mut reader,
            8,
            1,
            &mut messages,
            &mut continuations,
            &mut ranges,
            0,
        )
        .expect_err("v1 message bad flag combination should fail");
        assert!(err.to_string().contains("shared and do-not-share"));

        let mut messages = Vec::new();
        let mut continuations = Vec::new();
        let mut ranges = vec![(0, 4)];
        let mut v2 = Vec::new();
        v2.push(1);
        v2.extend_from_slice(&0u16.to_le_bytes());
        v2.push(MSG_FLAG_WAS_UNKNOWN);
        let mut reader = HdfReader::new(Cursor::new(v2));
        let err = read_v2_messages(
            &mut reader,
            4,
            false,
            &mut messages,
            &mut continuations,
            &mut ranges,
            0,
        )
        .expect_err("v2 message bad flag combination should fail");
        assert!(err
            .to_string()
            .contains("was-unknown without mark-if-unknown"));
    }
}
