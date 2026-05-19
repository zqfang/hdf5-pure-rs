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
    MSG_DATASPACE, MSG_DATATYPE, MSG_DRIVER_INFO, MSG_EXTERNAL_FILE_LIST, MSG_FILE_SPACE_INFO,
    MSG_FILL_VALUE, MSG_FILL_VALUE_OLD, MSG_FILTER_PIPELINE, MSG_FLAG_SHARED, MSG_GROUP_INFO,
    MSG_HEADER_CONTINUATION, MSG_LAYOUT, MSG_LINK, MSG_LINK_INFO, MSG_MDCI, MSG_NIL,
    MSG_OBJ_REF_COUNT, MSG_SHARED_MSG_TABLE, MSG_SYMBOL_TABLE, SHARED_HEAP_ID_LEN,
    SHARED_MESSAGE_MAX_INDEXES, SHARED_MESSAGE_TABLE_VERSION, SHARED_REFERENCE_VERSION_1,
    SHARED_REFERENCE_VERSION_2, SHARED_REFERENCE_VERSION_3, SHARED_TYPE_COMMITTED,
    SHARED_TYPE_SOHM,
};

const MSG_FLAG_DONTSHARE: u8 = 0x04;
const MSG_FLAG_FAIL_IF_UNKNOWN_AND_OPEN_FOR_WRITE: u8 = 0x08;
const MSG_FLAG_MARK_IF_UNKNOWN: u8 = 0x10;
const MSG_FLAG_WAS_UNKNOWN: u8 = 0x20;
const MSG_KNOWN_FLAGS: u8 = 0xff;

/// Decode the message stream of a v1 object header chunk. Walks the chunk
/// between `[reader.position(), chunk_end)`, dispatches NIL padding and
/// HEADER_CONTINUATION messages to the chunk layer, validates payloads, and
/// appends real messages to `messages`. Mirrors the per-message decode loop
/// in libhdf5's `H5Omessage.c` / `H5Oint.c` for v1 headers.
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
            if read_zero_padding(reader, remaining)? {
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

        let data = read_message_data(
            reader,
            msg_type,
            msg_flags,
            msg_size,
            reader.sizeof_addr(),
            reader.sizeof_size(),
        )?;
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

/// Decode the message stream of a v2 object header chunk. Same role as
/// `read_v1_messages` but uses the more compact v2 message header layout
/// (1-byte type, 2-byte size, 1-byte flags, optional 2-byte creation order
/// when the header tracks one).
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
            if read_zero_padding(reader, remaining)? {
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

        let data = read_message_data(
            reader,
            msg_type,
            msg_flags,
            msg_size,
            reader.sizeof_addr(),
            reader.sizeof_size(),
        )?;
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

fn read_zero_padding<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    mut remaining: usize,
) -> Result<bool> {
    let mut scratch = [0u8; 256];
    while remaining > 0 {
        let take = remaining.min(scratch.len());
        reader.read_bytes_into(&mut scratch[..take])?;
        if scratch[..take].iter().any(|&byte| byte != 0) {
            return Ok(false);
        }
        remaining -= take;
    }
    Ok(true)
}

fn read_message_data<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    msg_type: u16,
    msg_flags: u8,
    msg_size: u64,
    sizeof_addr: u8,
    sizeof_size: u8,
) -> Result<Vec<u8>> {
    if msg_flags & MSG_FLAG_SHARED == 0 {
        let mut data = vec![0u8; usize_from_u64(msg_size, "object header message size")?];
        reader.read_bytes_into(&mut data)?;
        return Ok(data);
    }

    read_shared_message_data_after_reference_check(
        reader,
        msg_type,
        msg_size,
        sizeof_addr,
        sizeof_size,
    )
}

fn read_shared_message_data_after_reference_check<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    msg_type: u16,
    msg_size: u64,
    sizeof_addr: u8,
    sizeof_size: u8,
) -> Result<Vec<u8>> {
    if msg_type == MSG_SHARED_MSG_TABLE
        || msg_type == MSG_GROUP_INFO
        || msg_type == MSG_BTREE_K
        || msg_type == MSG_OBJ_REF_COUNT
    {
        let mut data = vec![0u8; usize_from_u64(msg_size, "object header message size")?];
        reader.read_bytes_into(&mut data)?;
        return Ok(data);
    }

    let msg_size_usize = usize_from_u64(msg_size, "object header message size")?;
    let prefix_len = shared_message_reference_validation_len(msg_size_usize, reader)?;
    let mut data = vec![0u8; msg_size_usize];
    reader.read_bytes_into(&mut data[..prefix_len])?;
    validate_shared_message_reference(&data[..prefix_len], sizeof_addr, sizeof_size)?;

    reader.read_bytes_into(&mut data[prefix_len..])?;
    Ok(data)
}

fn shared_message_reference_validation_len<R: Read + Seek>(
    msg_size: usize,
    reader: &mut HdfReader<R>,
) -> Result<usize> {
    let start = reader.position()?;
    let mut version_and_type = [0u8; 2];
    let header_len = msg_size.min(version_and_type.len());
    reader.read_bytes_into(&mut version_and_type[..header_len])?;
    reader.seek(start)?;
    if header_len < version_and_type.len() {
        return Ok(header_len);
    }

    let len = match version_and_type[0] {
        SHARED_REFERENCE_VERSION_1 => 2usize
            .checked_add(6)
            .and_then(|len| len.checked_add(usize::from(reader.sizeof_size())))
            .and_then(|len| len.checked_add(usize::from(reader.sizeof_addr())))
            .ok_or_else(|| Error::InvalidFormat("shared message reference size overflow".into()))?,
        SHARED_REFERENCE_VERSION_2 => checked_usize_add(
            2,
            usize::from(reader.sizeof_addr()),
            "shared message v2 address",
        )?,
        SHARED_REFERENCE_VERSION_3 => match version_and_type[1] {
            SHARED_TYPE_SOHM => checked_usize_add(2, SHARED_HEAP_ID_LEN, "shared SOHM reference")?,
            SHARED_TYPE_COMMITTED => checked_usize_add(
                2,
                usize::from(reader.sizeof_addr()),
                "shared message v3 address",
            )?,
            _ => version_and_type.len(),
        },
        _ => version_and_type.len(),
    };
    Ok(len.min(msg_size))
}

/// Sanity-check a decoded message payload. For shared messages we only verify
/// the shared reference; for unshared messages we delegate to each message
/// type's own `decode`/`validate` routine.
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
        if msg_type == MSG_DRIVER_INFO {
            validate_driver_info_message(data)?;
        }
        if msg_type == MSG_MDCI {
            validate_metadata_cache_image_message(data, sizeof_addr, sizeof_size)?;
        }
        if msg_type == MSG_FILE_SPACE_INFO {
            validate_file_space_info_message(data, sizeof_addr, sizeof_size)?;
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

/// Validate a Group Info message payload (version 0 with up to two optional
/// 4-byte phase-change fields gated by the flags byte).
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

/// Validate a B-tree 'K' values message (version 0, at least 7 bytes for the
/// indexed-storage and group-leaf split values).
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

/// Validate an Object Reference Count message (4-byte refcount value).
fn validate_refcount_message(data: &[u8]) -> Result<()> {
    if data.len() < 4 {
        return Err(Error::InvalidFormat(
            "object refcount message is truncated".into(),
        ));
    }
    Ok(())
}

/// Validate an External File List message: version 1 header, slot table
/// with name/file offsets, and a defined local-heap address holding the
/// external file names.
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

/// Validate a Driver Info message (version 0, 8-byte driver name, followed
/// by a length-prefixed opaque driver-specific payload).
fn validate_driver_info_message(data: &[u8]) -> Result<()> {
    let mut pos = 0usize;
    let version = read_u8_at(data, &mut pos, "driver info version")?;
    if version != 0 {
        return Err(Error::InvalidFormat(format!(
            "driver info message version {version}"
        )));
    }
    pos = checked_usize_add(pos, 8, "driver info name")?;
    if data.len() < pos {
        return Err(Error::InvalidFormat("driver info name is truncated".into()));
    }
    let len = read_le_uint_at(data, &mut pos, 2, "driver info length")?;
    if len == 0 {
        return Err(Error::InvalidFormat(
            "driver info message length is zero".into(),
        ));
    }
    let len = usize::try_from(len)
        .map_err(|_| Error::InvalidFormat("driver info length exceeds usize".into()))?;
    let end = checked_usize_add(pos, len, "driver info payload")?;
    if data.len() < end {
        return Err(Error::InvalidFormat(
            "driver info payload is truncated".into(),
        ));
    }
    Ok(())
}

/// Validate a Metadata Cache Image (MDCI) message: version 0 plus an
/// address/size pair that must not overflow the address space.
fn validate_metadata_cache_image_message(
    data: &[u8],
    sizeof_addr: u8,
    sizeof_size: u8,
) -> Result<()> {
    let mut pos = 0usize;
    let version = read_u8_at(data, &mut pos, "metadata cache image version")?;
    if version != 0 {
        return Err(Error::InvalidFormat(format!(
            "metadata cache image message version {version}"
        )));
    }
    let addr = read_le_uint_at(
        data,
        &mut pos,
        usize::from(sizeof_addr),
        "metadata cache image address",
    )?;
    let size = read_le_uint_at(
        data,
        &mut pos,
        usize::from(sizeof_size),
        "metadata cache image size",
    )?;
    if !is_undefined_addr(addr, sizeof_addr)? && size != 0 {
        let undef = undefined_addr_value(sizeof_addr)?;
        if addr >= undef.saturating_sub(size) {
            return Err(Error::InvalidFormat(
                "metadata cache image address plus size overflows".into(),
            ));
        }
    }
    Ok(())
}

/// Validate a File Space Info message (version 1): strategy, persistence
/// flag, threshold/page-size fields, and optionally 12 free-space-manager
/// addresses when free-space tracking is persisted.
fn validate_file_space_info_message(data: &[u8], sizeof_addr: u8, sizeof_size: u8) -> Result<()> {
    let mut pos = 0usize;
    let version = read_u8_at(data, &mut pos, "file-space info version")?;
    if version != 1 {
        return Err(Error::InvalidFormat(format!(
            "file-space info message version {version}"
        )));
    }
    read_u8_at(data, &mut pos, "file-space info strategy")?;
    let persist = read_u8_at(data, &mut pos, "file-space info persist")? != 0;
    read_le_uint_at(
        data,
        &mut pos,
        usize::from(sizeof_size),
        "file-space info threshold",
    )?;
    let page_size = read_le_uint_at(
        data,
        &mut pos,
        usize::from(sizeof_size),
        "file-space info page size",
    )?;
    if page_size == 0 || page_size > 1024 * 1024 * 1024 {
        return Err(Error::InvalidFormat(
            "file-space info page size is invalid".into(),
        ));
    }
    read_le_uint_at(
        data,
        &mut pos,
        2,
        "file-space info page-end metadata threshold",
    )?;
    read_le_uint_at(
        data,
        &mut pos,
        usize::from(sizeof_addr),
        "file-space info pre-free-space EOA",
    )?;
    if persist {
        for _ in 0..12 {
            read_le_uint_at(
                data,
                &mut pos,
                usize::from(sizeof_addr),
                "file-space info free-space-manager address",
            )?;
        }
    }
    Ok(())
}

/// Reject message-header flag bytes that set reserved bits or contain
/// mutually exclusive flag combinations (shared & do-not-share, was-unknown
/// & fail-if-unknown-on-write, was-unknown without mark-if-unknown).
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

/// Validate a Shared Message Table message: supported version, defined
/// table address, and an in-range index count.
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

/// Validate a shared-message reference embedded in a message payload.
/// Handles the three SOHM reference versions (v1 with heap/addr pair, v2
/// addr-only, v3 SOHM heap ID or committed-datatype address).
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

/// Add two `u64` values, mapping overflow to a context-annotated error.
fn checked_u64_add(lhs: u64, rhs: u64, context: &str) -> Result<u64> {
    lhs.checked_add(rhs)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))
}

/// Add two `usize` values, mapping overflow to a context-annotated error.
fn checked_usize_add(lhs: usize, rhs: usize, context: &str) -> Result<usize> {
    lhs.checked_add(rhs)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} size overflow")))
}

/// Narrow a `u64` to `usize`, mapping overflow to a context-annotated error.
fn usize_from_u64(value: u64, context: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| Error::InvalidFormat(format!("{context} exceeds usize")))
}

/// Read a single byte from `data[*pos]` and advance the cursor.
fn read_u8_at(data: &[u8], pos: &mut usize, context: &str) -> Result<u8> {
    let value = data
        .get(*pos)
        .copied()
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))?;
    *pos = checked_usize_add(*pos, 1, context)?;
    Ok(value)
}

/// Read a little-endian unsigned integer of `width` bytes (1..=8) from
/// `data[*pos]` and advance the cursor.
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

/// Return the "undefined address" sentinel value for the given address
/// width (all bits set within the on-disk address byte count).
fn undefined_addr_value(sizeof_addr: u8) -> Result<u64> {
    let width = usize::from(sizeof_addr);
    if !(1..=8).contains(&width) {
        return Err(Error::InvalidFormat(format!(
            "address size {width} is invalid"
        )));
    }
    Ok(if width == 8 {
        u64::MAX
    } else {
        (1u64 << (width * 8)) - 1
    })
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
