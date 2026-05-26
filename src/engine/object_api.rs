use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::error::{Error, Result};
use crate::format::checksum::checksum_metadata;
use crate::format::messages::attribute::AttributeMessage;
use crate::format::messages::attribute_info::AttributeInfoMessage;
use crate::format::messages::data_layout::{ChunkIndexType, DataLayoutMessage, LayoutClass};
use crate::format::messages::dataspace::DataspaceMessage;
use crate::format::messages::datatype::{DatatypeMessage, DATATYPE_MESSAGE_VERSION_LATEST};
use crate::format::messages::fill_value::FillValueMessage;
use crate::format::messages::filter_pipeline::{FilterDesc, FilterPipelineMessage};
use crate::format::messages::link::LinkMessage;
use crate::format::messages::link_info::LinkInfoMessage;
use crate::format::object_header::{
    HDR_ATTR_CRT_ORDER_TRACKED, HDR_ATTR_STORE_PHASE_CHANGE, HDR_CHUNK0_SIZE_MASK, HDR_STORE_TIMES,
    HDR_V2_KNOWN_FLAGS,
};

const OBJECT_HEADER_V2_MAGIC: &[u8; 4] = b"OHDR";
const OBJECT_HEADER_V2_CHUNK_MAGIC: &[u8; 4] = b"OCHK";

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ObjectMessage {
    pub msg_type: u16,
    pub flags: u8,
    pub creation_index: u16,
    pub data: Vec<u8>,
    pub shared: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ObjectHeaderState {
    pub addr: u64,
    pub messages: Vec<ObjectMessage>,
    pub refcount: u32,
    pub comment: Option<String>,
    pub flush_disabled: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectHeaderPrefixImage {
    pub version: u8,
    pub raw: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectHeaderChunkImage {
    pub is_v2_continuation: bool,
    pub raw: Vec<u8>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SharedMessageTable {
    pub refs: BTreeMap<u64, usize>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SharedMessageTableInfo {
    pub version: u8,
    pub table_addr: u64,
    pub nindexes: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SharedMessageReference {
    V1 {
        message_type: u8,
        index: u64,
        addr: u64,
    },
    V2 {
        message_type: u8,
        addr: u64,
    },
    V3Sohm {
        heap_id: [u8; 8],
    },
    V3Committed {
        addr: u64,
    },
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExternalFileListMessage {
    pub version: u8,
    pub allocated_slots: u16,
    pub heap_addr: u64,
    pub entries: Vec<ExternalFileListEntry>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExternalFileListEntry {
    pub name_offset: u64,
    pub file_offset: u64,
    pub size: u64,
}

#[derive(Debug, Clone)]
pub struct AttributeObjectMessage {
    pub message: AttributeMessage,
    pub raw_size: usize,
}

#[derive(Debug, Clone)]
pub struct LinkObjectMessage {
    pub message: LinkMessage,
    pub raw_size: usize,
}

#[derive(Debug, Clone)]
pub struct LinkInfoObjectMessage {
    pub message: LinkInfoMessage,
    pub raw_size: usize,
}

#[derive(Debug, Clone)]
pub struct AttributeInfoObjectMessage {
    pub message: AttributeInfoMessage,
    pub raw_size: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FsInfoMessage {
    pub version: u8,
    pub free_space_strategy: u8,
    pub persist: bool,
    pub threshold: u64,
    pub page_size: u64,
    pub pgend_meta_thres: u16,
    pub eoa_pre_fsm_fsalloc: u64,
    pub fs_addr: Vec<u64>,
    pub sizeof_addr: u8,
    pub sizeof_size: u8,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GroupInfoMessage {
    pub version: u8,
    pub max_compact: Option<u16>,
    pub min_dense: Option<u16>,
    pub estimated_entries: Option<u16>,
    pub estimated_name_len: Option<u16>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BTreeKMessage {
    pub version: u8,
    pub indexed_storage_internal_k: u16,
    pub group_internal_k: u16,
    pub group_leaf_k: u16,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SymbolTableMessage {
    pub btree_addr: u64,
    pub heap_addr: u64,
}

#[derive(Debug, Clone)]
pub struct LayoutObjectMessage {
    pub message: DataLayoutMessage,
    pub raw: Vec<u8>,
    pub sizeof_addr: u8,
    pub sizeof_size: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataspaceObjectMessage {
    pub message: DataspaceMessage,
    pub raw: Vec<u8>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MetadataCacheImageMessage {
    pub version: u8,
    pub addr: u64,
    pub size: u64,
    pub sizeof_addr: u8,
    pub sizeof_size: u8,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DriverInfoMessage {
    pub version: u8,
    pub name: [u8; 8],
    pub data: Vec<u8>,
}

/// Link an object.
#[allow(non_snake_case)]
pub fn H5O__shared_link_adj(table: &mut SharedMessageTable, addr: u64, delta: isize) {
    if addr == u64::MAX || delta == 0 {
        return;
    }
    let entry = table.refs.entry(addr).or_default();
    if delta.is_negative() {
        let decrease = delta.unsigned_abs();
        if decrease >= *entry {
            table.refs.remove(&addr);
        } else {
            *entry -= decrease;
        }
    } else if let Ok(increase) = usize::try_from(delta) {
        *entry = entry.saturating_add(increase);
    }
}

/// Read from an object.
#[allow(non_snake_case)]
pub fn H5O__shared_read(table: &SharedMessageTable, addr: u64) -> Option<usize> {
    if addr == u64::MAX {
        return None;
    }
    if table.refs.is_empty() {
        return None;
    }
    let mut count = None;
    for (&stored_addr, &stored_count) in &table.refs {
        if stored_addr == addr {
            count = Some(stored_count);
            break;
        }
    }
    let count = count?;
    if count == 0 {
        None
    } else {
        Some(count)
    }
}

/// Decode an object from its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__shared_decode(bytes: &[u8], sizeof_addr: u8) -> Result<SharedMessageReference> {
    if sizeof_addr == 0 || sizeof_addr > 8 {
        return Err(Error::InvalidFormat(
            "shared message address width is invalid".into(),
        ));
    }
    if bytes.len() < 2 {
        return Err(Error::InvalidFormat(
            "shared object-header message reference is truncated".into(),
        ));
    }

    match bytes[0] {
        1 => {
            let index_start = 8usize;
            let addr_start = checked_add(index_start, 8, "shared message v1 reference address")?;
            let addr_end = checked_add(
                addr_start,
                usize::from(sizeof_addr),
                "shared message v1 address",
            )?;
            if bytes.len() < addr_end {
                return Err(Error::InvalidFormat(
                    "shared object-header message v1 reference is truncated".into(),
                ));
            }
            let index = read_le_uint_width(bytes, index_start, 8, "shared message v1 index")?;
            let addr = read_le_uint_width(
                bytes,
                addr_start,
                usize::from(sizeof_addr),
                "shared message v1 address",
            )?;
            if is_undefined_addr_width(addr, sizeof_addr)? {
                return Err(Error::InvalidFormat(
                    "shared object-header message address is undefined".into(),
                ));
            }
            Ok(SharedMessageReference::V1 {
                message_type: bytes[1],
                index,
                addr,
            })
        }
        2 => {
            let addr_end = checked_add(2, usize::from(sizeof_addr), "shared message v2 address")?;
            if bytes.len() < addr_end {
                return Err(Error::InvalidFormat(
                    "shared object-header message v2 reference is truncated".into(),
                ));
            }
            let addr = read_le_uint_width(
                bytes,
                2,
                usize::from(sizeof_addr),
                "shared message v2 address",
            )?;
            if is_undefined_addr_width(addr, sizeof_addr)? {
                return Err(Error::InvalidFormat(
                    "shared object-header message address is undefined".into(),
                ));
            }
            Ok(SharedMessageReference::V2 {
                message_type: bytes[1],
                addr,
            })
        }
        3 => match bytes[1] {
            1 => {
                let end = checked_add(2, 8, "shared SOHM reference")?;
                if bytes.len() < end {
                    return Err(Error::InvalidFormat(
                        "shared object-header message SOHM reference is truncated".into(),
                    ));
                }
                let heap_id: [u8; 8] = bytes[2..end].try_into().map_err(|_| {
                    Error::InvalidFormat(
                        "shared object-header message SOHM reference is truncated".into(),
                    )
                })?;
                Ok(SharedMessageReference::V3Sohm { heap_id })
            }
            2 => {
                let addr_end =
                    checked_add(2, usize::from(sizeof_addr), "shared message v3 address")?;
                if bytes.len() < addr_end {
                    return Err(Error::InvalidFormat(
                        "shared object-header message committed reference is truncated".into(),
                    ));
                }
                let addr = read_le_uint_width(
                    bytes,
                    2,
                    usize::from(sizeof_addr),
                    "shared message v3 address",
                )?;
                if is_undefined_addr_width(addr, sizeof_addr)? {
                    return Err(Error::InvalidFormat(
                        "shared object-header message address is undefined".into(),
                    ));
                }
                Ok(SharedMessageReference::V3Committed { addr })
            }
            _ => Err(Error::InvalidFormat(
                "shared object-header message type is invalid".into(),
            )),
        },
        _ => Err(Error::InvalidFormat(
            "shared object-header message version is invalid".into(),
        )),
    }
}

/// Link an object.
#[allow(non_snake_case)]
pub fn H5O__shared_link_adj_checked(
    table: &mut SharedMessageTable,
    addr: u64,
    delta: isize,
) -> Result<()> {
    let entry = table.refs.entry(addr).or_default();
    if delta.is_negative() {
        *entry = entry
            .checked_sub(delta.unsigned_abs())
            .ok_or_else(|| Error::InvalidFormat("shared object link refcount underflow".into()))?;
    } else {
        let delta = usize::try_from(delta).map_err(|_| {
            Error::InvalidFormat("shared object link refcount delta overflow".into())
        })?;
        *entry = entry
            .checked_add(delta)
            .ok_or_else(|| Error::InvalidFormat("shared object link refcount overflow".into()))?;
    }
    if *entry == 0 {
        table.refs.remove(&addr);
    }
    Ok(())
}

/// Object operation: set shared.
#[allow(non_snake_case)]
pub fn H5O_set_shared(message: &mut ObjectMessage, shared: bool) {
    message.shared = shared;
}

/// Delete an object.
#[allow(non_snake_case)]
pub fn H5O__shared_delete(table: &mut SharedMessageTable, addr: u64) {
    table.refs.remove(&addr);
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__shared_copy_file(table: &SharedMessageTable) -> SharedMessageTable {
    let mut copied = SharedMessageTable::default();
    for (&addr, &count) in &table.refs {
        if addr != u64::MAX && count != 0 {
            copied.refs.insert(addr, count);
        }
    }
    copied
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__shared_post_copy_file(table: &SharedMessageTable) -> SharedMessageTable {
    let mut copied = SharedMessageTable::default();
    for (&addr, &count) in &table.refs {
        if addr == u64::MAX || count == 0 {
            continue;
        }
        copied.refs.insert(addr, count);
    }
    copied
}

/// Return a debug-friendly representation of an object.
#[allow(non_snake_case)]
pub fn H5O__shared_debug_fmt(table: &SharedMessageTable, out: &mut dyn fmt::Write) -> fmt::Result {
    let total_refs = table.refs.values().copied().sum::<usize>();
    let max_refcount = table.refs.values().copied().max().unwrap_or(0);
    write!(
        out,
        "shared_messages={}, total_refs={}, max_refcount={}",
        table.refs.len(),
        total_refs,
        max_refcount
    )
}

/// Object operation: group isa.
#[allow(non_snake_case)]
pub fn H5O__group_isa(header: &ObjectHeaderState) -> bool {
    header
        .messages
        .iter()
        .any(|msg| matches!(msg.msg_type, 0x0011 | 0x000A | 0x000B))
}

/// Object operation: group get oloc.
#[allow(non_snake_case)]
pub fn H5O__group_get_oloc(header: &ObjectHeaderState) -> u64 {
    header.addr
}

/// Object operation: group bh info.
#[allow(non_snake_case)]
pub fn H5O__group_bh_info(header: &ObjectHeaderState) -> usize {
    let mut has_link_info = false;
    let mut old_style_storage = 0usize;
    let mut total = 0usize;
    for message in &header.messages {
        match message.msg_type {
            0x0002 => {
                has_link_info = true;
                total = total.saturating_add(message.data.len());
            }
            0x0011 => old_style_storage = old_style_storage.saturating_add(message.data.len()),
            _ => {}
        }
    }
    if has_link_info {
        total
    } else {
        old_style_storage
    }
}

/// Object operation: msg append oh.
#[allow(non_snake_case)]
pub fn H5O_msg_append_oh(header: &mut ObjectHeaderState, message: ObjectMessage) {
    let mut appended = message;
    if appended.creation_index == 0 {
        appended.creation_index = u16::try_from(header.messages.len()).unwrap_or(u16::MAX);
    }
    header.messages.push(appended);
    header.refcount = header.refcount.max(1);
}

/// Object operation: msg append real.
#[allow(non_snake_case)]
pub fn H5O__msg_append_real(header: &mut ObjectHeaderState, message: ObjectMessage) {
    let mut appended = message;
    if appended.creation_index == 0 {
        appended.creation_index = u16::try_from(header.messages.len()).unwrap_or(u16::MAX);
    }
    header.messages.push(appended);
    header.flush_disabled = false;
}

/// Write to an object.
#[allow(non_snake_case)]
pub fn H5O__msg_write_real(
    header: &mut ObjectHeaderState,
    index: usize,
    data: Vec<u8>,
) -> Result<()> {
    let msg = header
        .messages
        .get_mut(index)
        .ok_or_else(|| Error::InvalidFormat("object message index out of range".into()))?;
    msg.data = data;
    Ok(())
}

/// Write to an object.
#[allow(non_snake_case)]
pub fn H5O_msg_write_oh(header: &mut ObjectHeaderState, index: usize, data: Vec<u8>) -> Result<()> {
    let msg = header
        .messages
        .get_mut(index)
        .ok_or_else(|| Error::InvalidFormat("object message index out of range".into()))?;
    msg.data.clear();
    msg.data.extend_from_slice(&data);
    msg.flags |= 0x01;
    Ok(())
}

/// Read from an object.
#[allow(non_snake_case)]
pub fn H5O_msg_read_oh(header: &ObjectHeaderState, index: usize) -> Result<ObjectMessage> {
    header
        .messages
        .get(index)
        .cloned()
        .ok_or_else(|| Error::InvalidFormat("object message index out of range".into()))
}

/// Reset an object to its default state.
#[allow(non_snake_case)]
pub fn H5O_msg_reset(message: &mut ObjectMessage) {
    H5O__msg_reset_real(message);
}

/// Reset an object to its default state.
#[allow(non_snake_case)]
pub fn H5O__msg_reset_real(message: &mut ObjectMessage) {
    message.data.clear();
    message.flags = 0;
    message.shared = false;
}

/// Free an object's in-memory resources.
#[allow(non_snake_case)]
pub fn H5O_msg_free(mut message: ObjectMessage) {
    message.data.clear();
    message.flags = 0;
    message.creation_index = 0;
    message.shared = false;
    drop(message);
}

/// Free an object's in-memory resources.
#[allow(non_snake_case)]
pub fn H5O__msg_free_mesg(message: &mut ObjectMessage) {
    message.data.clear();
}

/// Free an object's in-memory resources.
#[allow(non_snake_case)]
pub fn H5O_msg_free_real(mut message: ObjectMessage) {
    H5O__msg_free_mesg(&mut message);
    message.flags = 0;
    message.creation_index = 0;
    message.shared = false;
    drop(message);
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O_msg_copy(message: &ObjectMessage) -> ObjectMessage {
    ObjectMessage {
        msg_type: message.msg_type,
        flags: message.flags,
        creation_index: message.creation_index,
        data: message.data.to_vec(),
        shared: message.shared,
    }
}

/// Object operation: msg exists.
#[allow(non_snake_case)]
pub fn H5O_msg_exists(header: &ObjectHeaderState, msg_type: u16) -> bool {
    for message in &header.messages {
        if message.msg_type == msg_type && !message.data.is_empty() {
            return true;
        }
    }
    false
}

/// Object operation: msg exists oh.
#[allow(non_snake_case)]
pub fn H5O_msg_exists_oh(header: &ObjectHeaderState, msg_type: u16) -> bool {
    for message in &header.messages {
        if message.msg_type == msg_type {
            return true;
        }
    }
    false
}

/// Remove an entry from an object.
#[allow(non_snake_case)]
pub fn H5O_msg_remove(header: &mut ObjectHeaderState, msg_type: u16) -> Option<ObjectMessage> {
    let mut found = None;
    for (idx, message) in header.messages.iter().enumerate() {
        if message.msg_type == msg_type {
            found = Some(idx);
            break;
        }
    }
    let removed = header.messages.remove(found?);
    if header.messages.is_empty() {
        header.flush_disabled = false;
    }
    Some(removed)
}

/// Remove an entry from an object.
#[allow(non_snake_case)]
pub fn H5O_msg_remove_op(message: &ObjectMessage, msg_type: u16) -> bool {
    if message.msg_type != msg_type {
        return false;
    }
    if message.shared {
        return false;
    }
    if message.msg_type == 0 || message.msg_type == 0xffff {
        return false;
    }
    true
}

/// Remove an entry from an object.
#[allow(non_snake_case)]
pub fn H5O__msg_remove_cb(message: &ObjectMessage, msg_type: u16) -> bool {
    if message.shared {
        return false;
    }
    message.msg_type == msg_type
}

/// Remove an entry from an object.
#[allow(non_snake_case)]
pub fn H5O__msg_remove_real(
    header: &mut ObjectHeaderState,
    msg_type: u16,
) -> Option<ObjectMessage> {
    let pos = header
        .messages
        .iter()
        .position(|message| message.msg_type == msg_type)?;
    Some(header.messages.remove(pos))
}

/// Iterate over the entries of an object.
#[allow(non_snake_case)]
pub fn H5O_msg_iterate(header: &ObjectHeaderState) -> impl Iterator<Item = &ObjectMessage> {
    header
        .messages
        .iter()
        .filter(|message| message.msg_type != 0 && !message.data.is_empty())
}

/// Iterate over the entries of an object.
#[allow(non_snake_case)]
pub fn H5O__msg_iterate_real(header: &ObjectHeaderState) -> impl Iterator<Item = &ObjectMessage> {
    let mut sequence = 0usize;
    header
        .messages
        .iter()
        .enumerate()
        .filter_map(move |(_idx, message)| {
            if matches!(message.msg_type, 0 | 0xffff | 0x001a) {
                return None;
            }
            sequence = sequence.saturating_add(1);
            Some(message)
        })
}

/// Object operation: msg raw size.
#[allow(non_snake_case)]
pub fn H5O_msg_raw_size(message: &ObjectMessage) -> usize {
    if message.data.is_empty() {
        return 0;
    }
    message.data.len()
}

/// Object operation: msg size f.
#[allow(non_snake_case)]
pub fn H5O_msg_size_f(message: &ObjectMessage) -> usize {
    let header_size = 5usize;
    match header_size.checked_add(message.data.len()) {
        Some(size) => size,
        None => usize::MAX,
    }
}

/// Object operation: msg size oh.
#[allow(non_snake_case)]
pub fn H5O_msg_size_oh(header: &ObjectHeaderState) -> Result<usize> {
    let mut size = 0usize;
    for message in &header.messages {
        size = size
            .checked_add(H5O_msg_raw_size(message))
            .ok_or_else(|| Error::InvalidFormat("object header message size overflow".into()))?;
    }
    Ok(size)
}

/// Object operation: msg can share.
#[allow(non_snake_case)]
pub fn H5O_msg_can_share(message: &ObjectMessage) -> bool {
    if message.data.is_empty() || message.shared {
        return false;
    }
    !matches!(message.msg_type, 0 | 0xffff)
}

/// Object operation: msg can share in ohdr.
#[allow(non_snake_case)]
pub fn H5O_msg_can_share_in_ohdr(message: &ObjectMessage) -> bool {
    H5O_msg_can_share(message)
}

/// Object operation: msg is shared.
#[allow(non_snake_case)]
pub fn H5O_msg_is_shared(message: &ObjectMessage) -> bool {
    if !message.shared {
        return false;
    }
    if message.data.is_empty() {
        return false;
    }
    message.flags & 0x02 != 0 || message.shared
}

/// Object operation: msg set share.
#[allow(non_snake_case)]
pub fn H5O_msg_set_share(message: &mut ObjectMessage) {
    if message.data.is_empty() {
        return;
    }
    message.shared = true;
    message.flags |= 0x02;
}

/// Reset an object to its default state.
#[allow(non_snake_case)]
pub fn H5O_msg_reset_share(message: &mut ObjectMessage) {
    message.shared = false;
}

/// Object operation: msg get crt index.
#[allow(non_snake_case)]
pub fn H5O_msg_get_crt_index(message: &ObjectMessage) -> u16 {
    if message.creation_index == 0 {
        0
    } else {
        message.creation_index
    }
}

/// Encode an object to its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O_msg_encode_into(message: &ObjectMessage, out: &mut Vec<u8>) -> Result<()> {
    let len = 5usize
        .checked_add(message.data.len())
        .ok_or_else(|| Error::InvalidFormat("object message image length overflow".into()))?;
    out.reserve(len);
    out.extend_from_slice(&message.msg_type.to_le_bytes());
    out.push(message.flags);
    out.extend_from_slice(&message.creation_index.to_le_bytes());
    out.extend_from_slice(&message.data);
    Ok(())
}

/// Encode an object to its on-disk representation.
#[allow(non_snake_case)]
#[deprecated(note = "use H5O_msg_encode_into to append into a caller-owned buffer")]
pub fn H5O_msg_encode(message: &ObjectMessage) -> Result<Vec<u8>> {
    let len = 5usize
        .checked_add(message.data.len())
        .ok_or_else(|| Error::InvalidFormat("object message image length overflow".into()))?;
    let mut out = Vec::with_capacity(len);
    H5O_msg_encode_into(message, &mut out)?;
    Ok(out)
}

/// Decode an object from its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O_msg_decode(bytes: &[u8]) -> Result<ObjectMessage> {
    if bytes.len() < 5 {
        return Err(Error::InvalidFormat(
            "object message image is too short".into(),
        ));
    }
    let msg_type = u16::from_le_bytes(
        bytes
            .get(0..2)
            .and_then(|raw| raw.try_into().ok())
            .ok_or_else(|| Error::InvalidFormat("object message type is truncated".into()))?,
    );
    let creation_index = u16::from_le_bytes(
        bytes
            .get(3..5)
            .and_then(|raw| raw.try_into().ok())
            .ok_or_else(|| {
                Error::InvalidFormat("object message creation index is truncated".into())
            })?,
    );
    Ok(ObjectMessage {
        msg_type,
        flags: bytes[2],
        creation_index,
        data: bytes[5..].to_vec(),
        shared: false,
    })
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__msg_copy_file(message: &ObjectMessage) -> ObjectMessage {
    ObjectMessage {
        msg_type: message.msg_type,
        flags: message.flags,
        creation_index: message.creation_index,
        data: message.data.clone(),
        shared: false,
    }
}

/// Allocate storage for an object.
#[allow(non_snake_case)]
pub fn H5O__msg_alloc(msg_type: u16, data: Vec<u8>) -> ObjectMessage {
    let mut flags = if data.is_empty() { 0 } else { 0x01 };
    if msg_type == 0 || msg_type == 0x001a || msg_type == 0xffff {
        flags &= !0x03;
    }
    ObjectMessage {
        msg_type,
        flags,
        data,
        ..ObjectMessage::default()
    }
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__copy_mesg(message: &ObjectMessage) -> ObjectMessage {
    ObjectMessage {
        msg_type: message.msg_type,
        flags: H5O_msg_get_flags(message),
        creation_index: message.creation_index,
        data: message.data.clone(),
        shared: message.shared,
    }
}

/// Delete an object.
#[allow(non_snake_case)]
pub fn H5O_msg_delete(message: &mut ObjectMessage) {
    message.data.clear();
    message.flags &= !0x03;
    message.shared = false;
}

/// Delete an object.
#[allow(non_snake_case)]
pub fn H5O__delete_mesg(message: &mut ObjectMessage) {
    message.data.clear();
    message.flags &= !0x03;
    message.shared = false;
}

/// Flush the object to storage.
#[allow(non_snake_case)]
pub fn H5O_msg_flush(message: &ObjectMessage) {
    let flags = H5O_msg_get_flags(message);
    let raw_size = H5O_msg_raw_size(message);
    let image_size = 5usize.saturating_add(raw_size);
    let mut prefix = Vec::with_capacity(5);
    prefix.extend_from_slice(&message.msg_type.to_le_bytes());
    prefix.push(flags);
    prefix.extend_from_slice(&message.creation_index.to_le_bytes());
    debug_assert_eq!(prefix.len(), image_size.saturating_sub(raw_size));
    if raw_size == 0 {
        debug_assert_eq!(flags & 0x01, 0);
    }
    if message.shared {
        debug_assert_ne!(flags & 0x02, 0);
    }
}

/// Flush the object to storage.
#[allow(non_snake_case)]
pub fn H5O__flush_msgs(header: &mut ObjectHeaderState) {
    for message in &mut header.messages {
        if message.data.is_empty() {
            message.flags &= !0x01;
        } else {
            message.flags |= 0x01;
        }
        if message.shared {
            message.flags |= 0x02;
        } else {
            message.flags &= !0x02;
        }
    }
    H5O__condense_header(header);
    header.flush_disabled = false;
}

/// Object operation: msg get flags.
#[allow(non_snake_case)]
pub fn H5O_msg_get_flags(message: &ObjectMessage) -> u8 {
    let mut flags = message.flags;
    if message.shared {
        flags |= 0x02;
    }
    if message.data.is_empty() {
        flags &= !0x01;
    }
    flags
}

/// Object operation: cache verify chksum.
#[allow(non_snake_case)]
pub fn H5O__cache_verify_chksum(image: &[u8], checksum: u32) -> bool {
    image
        .iter()
        .fold(0u32, |acc, byte| acc.wrapping_add(u32::from(*byte)))
        == checksum
}

/// Serialize an object to bytes.
#[allow(non_snake_case)]
pub fn H5O__cache_serialize_into(header: &ObjectHeaderState, out: &mut Vec<u8>) -> Result<()> {
    let mut len = 0usize;
    for message in &header.messages {
        len = len
            .checked_add(5)
            .and_then(|value| value.checked_add(message.data.len()))
            .ok_or_else(|| {
                Error::InvalidFormat("object header cache image length overflow".into())
            })?;
    }
    out.reserve(len);
    for message in &header.messages {
        H5O_msg_encode_into(message, out)?;
    }
    Ok(())
}

/// Object operation: cache get final load size.
#[allow(non_snake_case)]
pub fn H5O__cache_get_final_load_size(image: &[u8]) -> usize {
    if image.starts_with(OBJECT_HEADER_V2_MAGIC) {
        if image.len() < 7 {
            return image.len();
        }
        let flags = image[5];
        let mut pos = 6usize;
        if flags & HDR_STORE_TIMES != 0 {
            pos = match pos.checked_add(16) {
                Some(value) => value,
                None => return image.len(),
            };
        }
        if flags & HDR_ATTR_STORE_PHASE_CHANGE != 0 {
            pos = match pos.checked_add(4) {
                Some(value) => value,
                None => return image.len(),
            };
        }
        let size_len = 1usize << (flags & HDR_CHUNK0_SIZE_MASK);
        let chunk_size =
            match read_le_uint_width(image, pos, size_len, "object header v2 chunk size") {
                Ok(value) => value,
                Err(_) => return image.len(),
            };
        let chunk_size = match usize::try_from(chunk_size) {
            Ok(value) => value,
            Err(_) => return image.len(),
        };
        pos.checked_add(size_len)
            .and_then(|value| value.checked_add(chunk_size))
            .and_then(|value| value.checked_add(4))
            .unwrap_or(image.len())
    } else if image.first() == Some(&1) && image.len() >= 12 {
        let chunk_size = match read_le_u32_at(image, 8, "object header v1 chunk size") {
            Ok(value) => value as usize,
            Err(_) => return image.len(),
        };
        16usize
            .checked_add(chunk_size)
            .unwrap_or_else(|| image.len())
    } else {
        image.len()
    }
}

/// Deserialize an object from bytes.
#[allow(non_snake_case)]
pub fn H5O__cache_deserialize(image: &[u8]) -> Result<ObjectHeaderPrefixImage> {
    H5O__prefix_deserialize(image)
}

/// Object operation: cache image len.
#[allow(non_snake_case)]
pub fn H5O__cache_image_len(header: &ObjectHeaderState) -> Result<usize> {
    let mut len = 0usize;
    for message in &header.messages {
        len = len
            .checked_add(5)
            .and_then(|value| value.checked_add(message.data.len()))
            .ok_or_else(|| {
                Error::InvalidFormat("object header cache image length overflow".into())
            })?;
    }
    Ok(len)
}

/// Object operation: cache notify.
#[allow(non_snake_case)]
pub fn H5O__cache_notify(header: &ObjectHeaderState) {
    let mut image_len = 0usize;
    let mut dirty_like = false;
    for message in &header.messages {
        image_len = image_len
            .saturating_add(5)
            .saturating_add(message.data.len());
        dirty_like |= message.flags != H5O_msg_get_flags(message);
    }
    let protected_like = header.refcount > 0 && header.addr != u64::MAX;
    std::hint::black_box((image_len, dirty_like, protected_like));
}

/// Free an object's in-memory resources.
#[allow(non_snake_case)]
pub fn H5O__cache_free_icr(mut header: ObjectHeaderState) {
    for message in &mut header.messages {
        message.data.clear();
        message.flags = 0;
        message.creation_index = 0;
        message.shared = false;
    }
    header.messages.clear();
    header.comment = None;
    header.refcount = 0;
    header.flush_disabled = false;
    drop(header);
}

/// Object operation: cache chk get initial load size.
#[allow(non_snake_case)]
pub fn H5O__cache_chk_get_initial_load_size(size: usize) -> usize {
    size.min(512)
}

/// Object operation: cache chk verify chksum.
#[allow(non_snake_case)]
pub fn H5O__cache_chk_verify_chksum(image: &[u8], checksum: u32) -> bool {
    if image.starts_with(OBJECT_HEADER_V2_CHUNK_MAGIC) {
        if image.len() < 8 {
            return false;
        }
        let checksum_pos = image.len() - 4;
        let stored = match read_le_u32_at(image, checksum_pos, "object header chunk checksum") {
            Ok(value) => value,
            Err(_) => return false,
        };
        stored == checksum && checksum_metadata(&image[..checksum_pos]) == checksum
    } else {
        image
            .iter()
            .fold(0u32, |acc, byte| acc.wrapping_add(u32::from(*byte)))
            == checksum
    }
}

/// Deserialize an object from bytes.
#[allow(non_snake_case)]
pub fn H5O__cache_chk_deserialize(image: &[u8]) -> Result<ObjectHeaderChunkImage> {
    validate_object_header_chunk_image(image)?;
    Ok(ObjectHeaderChunkImage {
        is_v2_continuation: image.starts_with(OBJECT_HEADER_V2_CHUNK_MAGIC),
        raw: image.to_vec(),
    })
}

/// Object operation: cache chk image len.
#[allow(non_snake_case)]
pub fn H5O__cache_chk_image_len(image: &ObjectHeaderChunkImage) -> usize {
    image.raw.len()
}

/// Serialize an object to bytes.
#[allow(non_snake_case)]
pub fn H5O__cache_chk_serialize_into(
    image: &ObjectHeaderChunkImage,
    out: &mut Vec<u8>,
) -> Result<()> {
    validate_object_header_chunk_image(&image.raw)?;
    let start = out.len();
    out.extend_from_slice(&image.raw);
    if image.is_v2_continuation && image.raw.starts_with(OBJECT_HEADER_V2_CHUNK_MAGIC) {
        let image_len = out.len() - start;
        if image_len < 8 {
            return Err(Error::InvalidFormat(
                "object header v2 continuation chunk image is truncated".into(),
            ));
        }
        let checksum_pos = out.len() - 4;
        let checksum = checksum_metadata(&out[start..checksum_pos]);
        out[checksum_pos..].copy_from_slice(&checksum.to_le_bytes());
    }
    Ok(())
}

/// Serialize an object to bytes.
#[allow(non_snake_case)]
#[deprecated(note = "use H5O__cache_chk_serialize_into to append into a caller-owned buffer")]
pub fn H5O__cache_chk_serialize(image: &ObjectHeaderChunkImage) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(H5O__cache_chk_image_len(image));
    H5O__cache_chk_serialize_into(image, &mut out)?;
    Ok(out)
}

/// Object operation: cache chk notify.
#[allow(non_snake_case)]
pub fn H5O__cache_chk_notify(image: &ObjectHeaderChunkImage) {
    let mut checksum = 0u32;
    let mut has_magic = false;
    if image.is_v2_continuation && image.raw.starts_with(OBJECT_HEADER_V2_CHUNK_MAGIC) {
        has_magic = true;
        if image.raw.len() >= 4 {
            checksum = checksum_metadata(&image.raw[..image.raw.len() - 4]);
        }
    } else {
        for byte in &image.raw {
            checksum = checksum.wrapping_add(u32::from(*byte));
        }
    }
    std::hint::black_box((checksum, has_magic, image.raw.len()));
}

/// Free an object's in-memory resources.
#[allow(non_snake_case)]
pub fn H5O__cache_chk_free_icr(mut image: ObjectHeaderChunkImage) {
    image.raw.clear();
    image.is_v2_continuation = false;
    drop(image);
}

/// Object operation: add cont msg.
#[allow(non_snake_case)]
pub fn H5O__add_cont_msg(header: &mut ObjectHeaderState, addr: u64, size: u64) {
    let mut data = Vec::with_capacity(16);
    data.extend_from_slice(&addr.to_le_bytes());
    data.extend_from_slice(&size.to_le_bytes());
    H5O_msg_append_oh(header, H5O__msg_alloc(0x0010, data));
}

/// Internal helper `validate_object_header_chunk_image`.
fn validate_object_header_chunk_image(image: &[u8]) -> Result<()> {
    if !image.starts_with(OBJECT_HEADER_V2_CHUNK_MAGIC) {
        return Ok(());
    }
    if image.len() < 8 {
        return Err(Error::InvalidFormat(
            "object header v2 continuation chunk image is truncated".into(),
        ));
    }
    let checksum_pos = image.len() - 4;
    let stored_checksum = read_le_u32_at(
        image,
        checksum_pos,
        "object header v2 continuation chunk checksum",
    )?;
    let computed_checksum = checksum_metadata(&image[..checksum_pos]);
    if stored_checksum != computed_checksum {
        return Err(Error::InvalidFormat(format!(
            "object header continuation chunk checksum mismatch: stored={stored_checksum:#010x}, computed={computed_checksum:#010x}"
        )));
    }
    Ok(())
}

/// Deserialize an object from bytes.
#[allow(non_snake_case)]
pub fn H5O__prefix_deserialize(image: &[u8]) -> Result<ObjectHeaderPrefixImage> {
    let version = if image.starts_with(OBJECT_HEADER_V2_MAGIC) {
        if image.len() < 7 {
            return Err(Error::InvalidFormat(
                "object header v2 prefix image is truncated".into(),
            ));
        }
        let version = image[4];
        if version != 2 {
            return Err(Error::InvalidFormat(format!(
                "object header v2 prefix version {version} is invalid"
            )));
        }
        let flags = image[5];
        if flags & !HDR_V2_KNOWN_FLAGS != 0 {
            return Err(Error::InvalidFormat(format!(
                "object header v2 flags contain reserved bits: {flags:#04x}"
            )));
        }

        let mut pos = 6usize;
        if flags & HDR_STORE_TIMES != 0 {
            pos = checked_add(pos, 16, "object header v2 stored times")?;
        }
        if flags & HDR_ATTR_STORE_PHASE_CHANGE != 0 {
            let max_compact = read_le_u16_at(image, pos, "object header max compact attributes")?;
            let min_dense = read_le_u16_at(image, pos + 2, "object header min dense attributes")?;
            if max_compact < min_dense {
                return Err(Error::InvalidFormat(
                    "object header attribute phase change max compact is less than min dense"
                        .into(),
                ));
            }
            pos = checked_add(pos, 4, "object header attribute phase change")?;
        }

        let size_len = 1usize << (flags & HDR_CHUNK0_SIZE_MASK);
        let chunk0_data_size =
            read_le_uint_width(image, pos, size_len, "object header v2 chunk size")?;
        pos = checked_add(pos, size_len, "object header v2 chunk size")?;
        let chunk0_data_size = usize::try_from(chunk0_data_size)
            .map_err(|_| Error::InvalidFormat("object header v2 chunk size is too large".into()))?;
        let min_msg_header = if flags & HDR_ATTR_CRT_ORDER_TRACKED != 0 {
            6
        } else {
            4
        };
        if chunk0_data_size > 0 && chunk0_data_size < min_msg_header {
            return Err(Error::InvalidFormat("bad object header chunk size".into()));
        }
        let checksum_pos = checked_add(pos, chunk0_data_size, "object header v2 chunk data")?;
        let image_len = checked_add(checksum_pos, 4, "object header v2 checksum")?;
        if image.len() < image_len {
            return Err(Error::InvalidFormat(
                "object header v2 prefix image is truncated".into(),
            ));
        }
        let stored_checksum = read_le_u32_at(image, checksum_pos, "object header v2 checksum")?;
        let computed_checksum = checksum_metadata(&image[..checksum_pos]);
        if stored_checksum != computed_checksum {
            return Err(Error::InvalidFormat(format!(
                "object header checksum mismatch: stored={stored_checksum:#010x}, computed={computed_checksum:#010x}"
            )));
        }
        version
    } else {
        if image.len() < 16 {
            return Err(Error::InvalidFormat(
                "object header v1 prefix image is truncated".into(),
            ));
        }
        let version = image[0];
        if version != 1 {
            return Err(Error::InvalidFormat(format!(
                "object header prefix version {version} is invalid"
            )));
        }
        let nmesgs = read_le_u16_at(image, 2, "object header v1 message count")?;
        let chunk_size = usize::try_from(read_le_u32_at(image, 8, "object header v1 chunk size")?)
            .map_err(|_| {
                Error::InvalidFormat("object header v1 chunk size exceeds usize".into())
            })?;
        if (nmesgs > 0 && chunk_size < 8) || (nmesgs == 0 && chunk_size > 0) {
            return Err(Error::InvalidFormat("bad object header chunk size".into()));
        }
        let expected = 16usize
            .checked_add(chunk_size)
            .ok_or_else(|| Error::InvalidFormat("object header v1 image size overflow".into()))?;
        if image.len() < expected {
            return Err(Error::InvalidFormat(
                "object header v1 prefix image is truncated".into(),
            ));
        }
        version
    };
    Ok(ObjectHeaderPrefixImage {
        version,
        raw: image.to_vec(),
    })
}

/// Deserialize an object from bytes.
#[allow(non_snake_case)]
pub fn H5O__chunk_deserialize(image: &[u8]) -> Result<ObjectHeaderChunkImage> {
    let is_v2_continuation = image.starts_with(OBJECT_HEADER_V2_CHUNK_MAGIC);
    if is_v2_continuation {
        if image.len() < 8 {
            return Err(Error::InvalidFormat(
                "object header v2 continuation chunk image is truncated".into(),
            ));
        }
        let checksum_pos = image.len() - 4;
        let stored_checksum = read_le_u32_at(
            image,
            checksum_pos,
            "object header v2 continuation chunk checksum",
        )?;
        let computed_checksum = checksum_metadata(&image[..checksum_pos]);
        if stored_checksum != computed_checksum {
            return Err(Error::InvalidFormat(format!(
                "object header continuation chunk checksum mismatch: stored={stored_checksum:#010x}, computed={computed_checksum:#010x}"
            )));
        }

        let mut parsed = false;
        let mut first_error: Option<Error> = None;
        for has_creation_index in [false, true] {
            let msg_header_size = if has_creation_index { 6usize } else { 4usize };
            let mut pos = 4usize;
            let mut null_messages = 0usize;
            let mut attempt = Ok(());
            while pos < checksum_pos {
                let remaining = checksum_pos - pos;
                if remaining < msg_header_size {
                    if null_messages != 0 {
                        attempt = Err(Error::InvalidFormat(
                            "gap in object header chunk with null messages".into(),
                        ));
                    }
                    pos = checksum_pos;
                    break;
                }
                let id = image[pos];
                pos = checked_add(pos, 1, "object header v2 message id")?;
                let mesg_size =
                    usize::from(read_le_u16_at(image, pos, "object header v2 message size")?);
                pos = checked_add(pos, 2, "object header v2 message size")?;
                let flags = image[pos];
                pos = checked_add(pos, 1, "object header v2 message flags")?;
                if (flags & 0x02 != 0) && (flags & 0x04 != 0) {
                    attempt = Err(Error::InvalidFormat(
                        "bad object header message sharing flag combination".into(),
                    ));
                    break;
                }
                if (flags & 0x20 != 0) && (flags & 0x08 != 0) {
                    attempt = Err(Error::InvalidFormat(
                        "bad object header unknown-message flag combination".into(),
                    ));
                    break;
                }
                if (flags & 0x20 != 0) && (flags & 0x10 == 0) {
                    attempt = Err(Error::InvalidFormat(
                        "object header unknown-message flag is missing mark-if-unknown".into(),
                    ));
                    break;
                }
                if has_creation_index {
                    let _crt_idx =
                        read_le_u16_at(image, pos, "object header v2 message creation index")?;
                    pos = checked_add(pos, 2, "object header v2 message creation index")?;
                }
                let data_end = match pos.checked_add(mesg_size) {
                    Some(value) if value <= checksum_pos => value,
                    _ => {
                        attempt = Err(Error::InvalidFormat(
                            "object header message size exceeds buffer end".into(),
                        ));
                        break;
                    }
                };
                if id == 0 {
                    null_messages = null_messages.saturating_add(1);
                }
                pos = data_end;
            }
            if attempt.is_ok() && pos == checksum_pos {
                parsed = true;
                break;
            }
            if first_error.is_none() {
                first_error = attempt.err();
            }
        }
        if !parsed {
            return Err(first_error.unwrap_or_else(|| {
                Error::InvalidFormat("object header chunk image size mismatch".into())
            }));
        }
    }
    Ok(ObjectHeaderChunkImage {
        is_v2_continuation,
        raw: image.to_vec(),
    })
}

/// Encode an object to its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__bogus_encode_ref(data: &[u8]) -> &[u8] {
    data
}

/// Encode an object to its on-disk representation.
#[allow(non_snake_case)]
#[deprecated(note = "use H5O__bogus_encode_ref to borrow the existing bytes")]
pub fn H5O__bogus_encode(data: &[u8]) -> Vec<u8> {
    H5O__bogus_encode_ref(data).to_vec()
}

/// Decode an object from its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__bogus_decode_ref(bytes: &[u8]) -> &[u8] {
    bytes
}

/// Decode an object from its on-disk representation.
#[allow(non_snake_case)]
#[deprecated(note = "use H5O__bogus_decode_ref to borrow the message bytes")]
pub fn H5O__bogus_decode(bytes: &[u8]) -> Vec<u8> {
    H5O__bogus_decode_ref(bytes).to_vec()
}

/// Object operation: bogus size.
#[allow(non_snake_case)]
pub fn H5O__bogus_size(data: &[u8]) -> usize {
    data.len()
}

/// Return a debug-friendly representation of an object.
#[allow(non_snake_case)]
pub fn H5O__bogus_debug_fmt(data: &[u8], out: &mut dyn fmt::Write) -> fmt::Result {
    let mut value = 0u32;
    for (idx, byte) in data.iter().take(4).enumerate() {
        value |= u32::from(*byte) << (idx * 8);
    }
    write!(out, "bogus(bytes={}, value={value})", data.len())
}

/// Decode an object from its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__layout_decode(bytes: &[u8]) -> Result<LayoutObjectMessage> {
    if bytes.is_empty() {
        return Err(Error::InvalidFormat(
            "data layout message is truncated".into(),
        ));
    }
    let sizeof_addr = 8u8;
    let sizeof_size = 8u8;
    let addr_width = usize::from(sizeof_addr);
    let size_width = usize::from(sizeof_size);
    let version = bytes[0];
    if !(1..=4).contains(&version) {
        return Err(Error::InvalidFormat(format!(
            "bad version number for layout message: {version}"
        )));
    }
    if version < 3 {
        if bytes.len() < 8 {
            return Err(Error::InvalidFormat(
                "data layout v1/v2 header is truncated".into(),
            ));
        }
        let ndims = usize::from(bytes[1]);
        if ndims == 0 || ndims > 32 {
            return Err(Error::InvalidFormat(
                "dimensionality is out of range".into(),
            ));
        }
        let layout_class = bytes[2];
        if layout_class > 2 {
            return Err(Error::InvalidFormat(
                "bad layout type for layout message".into(),
            ));
        }
        let mut pos = 8usize;
        if layout_class != 0 {
            pos = checked_add(pos, addr_width, "data layout v1/v2 address")?;
        }
        for dimno in 0..ndims {
            let dim = read_le_u32_at(bytes, pos, "data layout v1/v2 dimensions")?;
            if layout_class == 2 && dimno + 1 < ndims && dim == 0 {
                return Err(Error::InvalidFormat(
                    "chunk dimension must be positive".into(),
                ));
            }
            pos = checked_add(pos, 4, "data layout v1/v2 dimensions")?;
        }
        if layout_class == 2 && ndims < 2 {
            return Err(Error::InvalidFormat(
                "bad dimensions for chunked storage".into(),
            ));
        }
        if layout_class == 0 {
            let compact_size = usize::try_from(read_le_u32_at(
                bytes,
                pos,
                "data layout v1/v2 compact size",
            )?)
            .map_err(|_| Error::InvalidFormat("compact layout data size exceeds usize".into()))?;
            pos = checked_add(pos, 4, "data layout v1/v2 compact size")?;
            let _compact_end = checked_add(pos, compact_size, "data layout v1/v2 compact data")?;
            if bytes.len() < _compact_end {
                return Err(Error::InvalidFormat(
                    "data layout v1/v2 compact data is truncated".into(),
                ));
            }
        }
    } else {
        if bytes.len() < 2 {
            return Err(Error::InvalidFormat(
                "data layout v3/v4 header is truncated".into(),
            ));
        }
        let layout_class = bytes[1];
        if layout_class > 3 || (layout_class == 3 && version < 4) {
            return Err(Error::InvalidFormat(
                "bad layout type for layout message".into(),
            ));
        }
        let mut pos = 2usize;
        match layout_class {
            0 => {
                let compact_size = usize::from(read_le_u16_at(
                    bytes,
                    pos,
                    "data layout v3/v4 compact size",
                )?);
                pos = checked_add(pos, 2, "data layout v3/v4 compact size")?;
                let compact_end = checked_add(pos, compact_size, "data layout v3/v4 compact data")?;
                if bytes.len() < compact_end {
                    return Err(Error::InvalidFormat(
                        "data layout v3/v4 compact data is truncated".into(),
                    ));
                }
            }
            1 => {
                pos = checked_add(pos, addr_width, "data layout v3/v4 contiguous address")?;
                let end = checked_add(pos, size_width, "data layout v3/v4 contiguous size")?;
                if bytes.len() < end {
                    return Err(Error::InvalidFormat(
                        "data layout v3/v4 contiguous storage is truncated".into(),
                    ));
                }
            }
            2 if version == 3 => {
                let ndims = usize::from(*bytes.get(pos).ok_or_else(|| {
                    Error::InvalidFormat("data layout v3 chunk rank is truncated".into())
                })?);
                if !(2..=32).contains(&ndims) {
                    return Err(Error::InvalidFormat(
                        "bad dimensions for chunked storage".into(),
                    ));
                }
                pos = checked_add(pos, 1, "data layout v3 chunk rank")?;
                pos = checked_add(pos, addr_width, "data layout v3 chunk index address")?;
                for dimno in 0..ndims {
                    let dim = read_le_u32_at(bytes, pos, "data layout v3 chunk dimensions")?;
                    if dim == 0 {
                        return Err(Error::InvalidFormat(
                            "chunk dimension must be positive".into(),
                        ));
                    }
                    pos = checked_add(pos, 4, "data layout v3 chunk dimensions")?;
                    if dimno + 1 == ndims {
                        break;
                    }
                }
            }
            2 => {
                let flags = *bytes.get(pos).ok_or_else(|| {
                    Error::InvalidFormat("data layout v4 chunk flags are truncated".into())
                })?;
                if flags != 0 && flags != 0x02 {
                    return Err(Error::InvalidFormat(
                        "data layout v4 chunk flags are invalid".into(),
                    ));
                }
                pos = checked_add(pos, 1, "data layout v4 chunk flags")?;
                let ndims = usize::from(*bytes.get(pos).ok_or_else(|| {
                    Error::InvalidFormat("data layout v4 chunk rank is truncated".into())
                })?);
                if ndims == 0 || ndims > 32 {
                    return Err(Error::InvalidFormat(
                        "bad dimensions for chunked storage".into(),
                    ));
                }
                pos = checked_add(pos, 1, "data layout v4 chunk rank")?;
                let enc_bytes_per_dim = usize::from(*bytes.get(pos).ok_or_else(|| {
                    Error::InvalidFormat(
                        "data layout v4 encoded dimension size is truncated".into(),
                    )
                })?);
                if !(1..=8).contains(&enc_bytes_per_dim) {
                    return Err(Error::InvalidFormat(
                        "data layout v4 encoded dimension size is invalid".into(),
                    ));
                }
                pos = checked_add(pos, 1, "data layout v4 encoded dimension size")?;
                for _ in 0..ndims {
                    let dim = read_le_uint_width(
                        bytes,
                        pos,
                        enc_bytes_per_dim,
                        "data layout v4 chunk dimensions",
                    )?;
                    if dim == 0 {
                        return Err(Error::InvalidFormat(
                            "chunk dimension must be positive".into(),
                        ));
                    }
                    pos = checked_add(pos, enc_bytes_per_dim, "data layout v4 chunk dimensions")?;
                }
            }
            3 => {
                pos = checked_add(pos, addr_width, "data layout v4 virtual heap address")?;
                let end = checked_add(pos, 4, "data layout v4 virtual heap index")?;
                if bytes.len() < end {
                    return Err(Error::InvalidFormat(
                        "data layout v4 virtual storage is truncated".into(),
                    ));
                }
            }
            _ => unreachable!(),
        }
    }
    let message = DataLayoutMessage::decode(bytes, sizeof_addr, sizeof_size)?;
    if message.version != version {
        return Err(Error::InvalidFormat(
            "data layout message version does not match payload".into(),
        ));
    }
    Ok(LayoutObjectMessage {
        message,
        raw: bytes.to_vec(),
        sizeof_addr,
        sizeof_size,
    })
}

/// Decode an object from its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__layout_decode_with_sizes(
    bytes: &[u8],
    sizeof_addr: u8,
    sizeof_size: u8,
) -> Result<LayoutObjectMessage> {
    Ok(LayoutObjectMessage {
        message: DataLayoutMessage::decode(bytes, sizeof_addr, sizeof_size)?,
        raw: bytes.to_vec(),
        sizeof_addr,
        sizeof_size,
    })
}

/// Encode an object to its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__layout_encode(layout: &LayoutObjectMessage) -> Result<Vec<u8>> {
    if let Some(version) = layout.raw.first().copied() {
        if version != layout.message.version {
            return Err(Error::InvalidFormat(
                "data layout message version does not match raw payload".into(),
            ));
        }
        if DataLayoutMessage::decode(&layout.raw, layout.sizeof_addr, layout.sizeof_size).is_ok() {
            return Ok(layout.raw.clone());
        }
    }

    let encoded_version = layout.message.version.max(3);
    let mut out = Vec::new();
    out.push(encoded_version);
    match layout.message.layout_class {
        LayoutClass::Compact => {
            out.push(0);
            let data = layout.message.compact_data.as_deref().unwrap_or(&[]);
            let len = u16::try_from(data.len()).map_err(|_| {
                Error::InvalidFormat("compact layout data exceeds u16 length".into())
            })?;
            out.extend_from_slice(&len.to_le_bytes());
            out.extend_from_slice(data);
        }
        LayoutClass::Contiguous => {
            out.push(1);
            let addr = layout
                .message
                .contiguous_addr
                .or(layout.message.data_addr)
                .ok_or_else(|| {
                    Error::InvalidFormat("contiguous layout address is missing".into())
                })?;
            let size = layout
                .message
                .contiguous_size
                .ok_or_else(|| Error::InvalidFormat("contiguous layout size is missing".into()))?;
            let addr_width = usize::from(layout.sizeof_addr);
            let size_width = usize::from(layout.sizeof_size);
            if !(1..=8).contains(&addr_width) || !(1..=8).contains(&size_width) {
                return Err(Error::InvalidFormat(
                    "data layout address or size width is invalid".into(),
                ));
            }
            out.extend_from_slice(&addr.to_le_bytes()[..addr_width]);
            out.extend_from_slice(&size.to_le_bytes()[..size_width]);
        }
        LayoutClass::Chunked if encoded_version < 4 => {
            out.push(2);
            let dims = layout.message.chunk_dims.as_deref().ok_or_else(|| {
                Error::InvalidFormat("chunked layout dimensions are missing".into())
            })?;
            let elem_size = layout.message.chunk_element_size.ok_or_else(|| {
                Error::InvalidFormat("chunked layout element size is missing".into())
            })?;
            let ndims = u8::try_from(dims.len() + 1)
                .map_err(|_| Error::InvalidFormat("chunked layout rank exceeds u8".into()))?;
            if ndims < 2 {
                return Err(Error::InvalidFormat(
                    "chunked layout rank must include at least one dimension and element size"
                        .into(),
                ));
            }
            let addr = layout
                .message
                .chunk_index_addr
                .or(layout.message.data_addr)
                .ok_or_else(|| {
                    Error::InvalidFormat("chunked layout index address is missing".into())
                })?;
            let addr_width = usize::from(layout.sizeof_addr);
            if !(1..=8).contains(&addr_width) {
                return Err(Error::InvalidFormat(
                    "data layout address width is invalid".into(),
                ));
            }
            out.push(ndims);
            out.extend_from_slice(&addr.to_le_bytes()[..addr_width]);
            for &dim in dims {
                let dim = u32::try_from(dim).map_err(|_| {
                    Error::InvalidFormat("chunked layout dimension exceeds u32".into())
                })?;
                out.extend_from_slice(&dim.to_le_bytes());
            }
            out.extend_from_slice(&elem_size.to_le_bytes());
        }
        LayoutClass::Chunked => {
            out.push(2);
            let dims = layout
                .message
                .chunk_encoded_dims
                .as_deref()
                .or(layout.message.chunk_dims.as_deref())
                .ok_or_else(|| {
                    Error::InvalidFormat("chunked layout dimensions are missing".into())
                })?;
            let flags = layout.message.chunk_flags.unwrap_or(0);
            let max_dim = dims.iter().copied().max().unwrap_or(0);
            let enc_bytes_per_dim = (1usize..=8)
                .find(|width| u128::from(max_dim) < (1u128 << (width * 8)))
                .unwrap_or(8);
            let ndims = u8::try_from(dims.len())
                .map_err(|_| Error::InvalidFormat("chunked layout rank exceeds u8".into()))?;
            if ndims == 0 {
                return Err(Error::InvalidFormat(
                    "chunked layout rank must be positive".into(),
                ));
            }
            out.push(flags);
            out.push(ndims);
            out.push(u8::try_from(enc_bytes_per_dim).unwrap_or(8));
            for &dim in dims {
                out.extend_from_slice(&dim.to_le_bytes()[..enc_bytes_per_dim]);
            }
            let idx_type = layout.message.chunk_index_type.ok_or_else(|| {
                Error::InvalidFormat("chunked layout index type is missing".into())
            })?;
            out.push(match idx_type {
                ChunkIndexType::BTreeV1 => {
                    return Err(Error::InvalidFormat(
                        "v1 B-tree index type is invalid in v4 layout message".into(),
                    ))
                }
                ChunkIndexType::SingleChunk => 1,
                ChunkIndexType::Implicit => 2,
                ChunkIndexType::FixedArray => 3,
                ChunkIndexType::ExtensibleArray => 4,
                ChunkIndexType::BTreeV2 => 5,
            });
            let size_width = usize::from(layout.sizeof_size);
            if idx_type == ChunkIndexType::SingleChunk && flags & 0x02 != 0 {
                let filtered_size = layout.message.single_chunk_filtered_size.ok_or_else(|| {
                    Error::InvalidFormat("single chunk filtered size is missing".into())
                })?;
                let filter_mask = layout.message.single_chunk_filter_mask.ok_or_else(|| {
                    Error::InvalidFormat("single chunk filter mask is missing".into())
                })?;
                if !(1..=8).contains(&size_width) {
                    return Err(Error::InvalidFormat(
                        "data layout size width is invalid".into(),
                    ));
                }
                out.extend_from_slice(&filtered_size.to_le_bytes()[..size_width]);
                out.extend_from_slice(&filter_mask.to_le_bytes());
            }
            let addr = layout
                .message
                .chunk_index_addr
                .or(layout.message.data_addr)
                .ok_or_else(|| {
                    Error::InvalidFormat("chunked layout index address is missing".into())
                })?;
            let addr_width = usize::from(layout.sizeof_addr);
            if !(1..=8).contains(&addr_width) {
                return Err(Error::InvalidFormat(
                    "data layout address width is invalid".into(),
                ));
            }
            out.extend_from_slice(&addr.to_le_bytes()[..addr_width]);
        }
        LayoutClass::Virtual => {
            out.push(3);
            let addr = layout.message.virtual_heap_addr.ok_or_else(|| {
                Error::InvalidFormat("virtual layout heap address is missing".into())
            })?;
            let index = layout.message.virtual_heap_index.ok_or_else(|| {
                Error::InvalidFormat("virtual layout heap index is missing".into())
            })?;
            let addr_width = usize::from(layout.sizeof_addr);
            if !(1..=8).contains(&addr_width) {
                return Err(Error::InvalidFormat(
                    "data layout address width is invalid".into(),
                ));
            }
            out.extend_from_slice(&addr.to_le_bytes()[..addr_width]);
            out.extend_from_slice(&index.to_le_bytes());
        }
    }
    Ok(out)
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__layout_copy(layout: &LayoutObjectMessage) -> LayoutObjectMessage {
    let message = DataLayoutMessage {
        version: layout.message.version,
        layout_class: layout.message.layout_class,
        compact_data: layout
            .message
            .compact_data
            .as_ref()
            .map(|data| data.to_vec()),
        contiguous_addr: layout.message.contiguous_addr,
        contiguous_size: layout.message.contiguous_size,
        chunk_dims: layout.message.chunk_dims.as_ref().map(|dims| dims.to_vec()),
        chunk_index_addr: layout.message.chunk_index_addr,
        chunk_index_type: layout.message.chunk_index_type,
        chunk_element_size: layout.message.chunk_element_size,
        chunk_flags: layout.message.chunk_flags,
        chunk_encoded_dims: layout
            .message
            .chunk_encoded_dims
            .as_ref()
            .map(|dims| dims.to_vec()),
        single_chunk_filtered_size: layout.message.single_chunk_filtered_size,
        single_chunk_filter_mask: layout.message.single_chunk_filter_mask,
        data_addr: layout.message.data_addr,
        virtual_heap_addr: layout.message.virtual_heap_addr,
        virtual_heap_index: layout.message.virtual_heap_index,
    };
    let mut raw = Vec::with_capacity(layout.raw.len());
    raw.extend_from_slice(&layout.raw);
    LayoutObjectMessage {
        message,
        raw,
        sizeof_addr: layout.sizeof_addr,
        sizeof_size: layout.sizeof_size,
    }
}

/// Object operation: layout size.
#[allow(non_snake_case)]
pub fn H5O__layout_size(layout: &LayoutObjectMessage) -> usize {
    layout.raw.len()
}

/// Reset an object to its default state.
#[allow(non_snake_case)]
pub fn H5O__layout_reset(layout: &mut LayoutObjectMessage) {
    layout.raw.clear();
}

/// Free an object's in-memory resources.
#[allow(non_snake_case)]
pub fn H5O__layout_free(mut layout: LayoutObjectMessage) {
    if let Some(compact) = layout.message.compact_data.as_mut() {
        compact.clear();
    }
    if let Some(dims) = layout.message.chunk_dims.as_mut() {
        dims.clear();
    }
    if let Some(dims) = layout.message.chunk_encoded_dims.as_mut() {
        dims.clear();
    }
    layout.raw.clear();
    drop(layout);
}

/// Delete an object.
#[allow(non_snake_case)]
pub fn H5O__layout_delete(layout: &mut LayoutObjectMessage) {
    layout.message =
        DataLayoutMessage::decode(&[1, 0, 0, 0], layout.sizeof_addr, layout.sizeof_size)
            .unwrap_or_else(|_| layout.message.clone());
    layout.raw.clear();
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__layout_pre_copy_file(layout: &LayoutObjectMessage) -> LayoutObjectMessage {
    layout.clone()
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__layout_copy_file(layout: &LayoutObjectMessage) -> LayoutObjectMessage {
    let mut message = layout.message.clone();
    match message.layout_class {
        LayoutClass::Compact => {
            if message.compact_data.is_none() {
                message.compact_data = Some(Vec::new());
            }
            message.contiguous_addr = None;
            message.contiguous_size = None;
            message.chunk_dims = None;
            message.chunk_index_addr = None;
            message.chunk_index_type = None;
            message.chunk_element_size = None;
            message.chunk_flags = None;
            message.chunk_encoded_dims = None;
            message.single_chunk_filtered_size = None;
            message.single_chunk_filter_mask = None;
            message.data_addr = None;
            message.virtual_heap_addr = None;
            message.virtual_heap_index = None;
        }
        LayoutClass::Contiguous => {
            if message.contiguous_addr.is_none() {
                message.contiguous_addr = message.data_addr;
            }
            if message.data_addr.is_none() {
                message.data_addr = message.contiguous_addr;
            }
            message.compact_data = None;
            message.chunk_dims = None;
            message.chunk_index_addr = None;
            message.chunk_index_type = None;
            message.chunk_element_size = None;
            message.chunk_flags = None;
            message.chunk_encoded_dims = None;
            message.single_chunk_filtered_size = None;
            message.single_chunk_filter_mask = None;
            message.virtual_heap_addr = None;
            message.virtual_heap_index = None;
        }
        LayoutClass::Chunked => {
            if message.chunk_encoded_dims.is_none() {
                message.chunk_encoded_dims = message.chunk_dims.clone();
            }
            if message.chunk_index_addr.is_none() {
                message.chunk_index_addr = message.data_addr;
            }
            if message.data_addr.is_none() {
                message.data_addr = message.chunk_index_addr;
            }
            message.compact_data = None;
            message.contiguous_addr = None;
            message.contiguous_size = None;
            message.virtual_heap_addr = None;
            message.virtual_heap_index = None;
        }
        LayoutClass::Virtual => {
            message.compact_data = None;
            message.contiguous_addr = None;
            message.contiguous_size = None;
            message.chunk_dims = None;
            message.chunk_index_addr = None;
            message.chunk_index_type = None;
            message.chunk_element_size = None;
            message.chunk_flags = None;
            message.chunk_encoded_dims = None;
            message.single_chunk_filtered_size = None;
            message.single_chunk_filter_mask = None;
            message.data_addr = None;
        }
    }
    let temp = LayoutObjectMessage {
        message: message.clone(),
        raw: Vec::new(),
        sizeof_addr: layout.sizeof_addr,
        sizeof_size: layout.sizeof_size,
    };
    let raw = H5O__layout_encode(&temp).unwrap_or_else(|_| layout.raw.clone());
    LayoutObjectMessage {
        message,
        raw,
        sizeof_addr: layout.sizeof_addr,
        sizeof_size: layout.sizeof_size,
    }
}

/// Return a debug-friendly representation of an object.
#[allow(non_snake_case)]
pub fn H5O__layout_debug_fmt(
    layout: &LayoutObjectMessage,
    out: &mut dyn fmt::Write,
) -> fmt::Result {
    write!(
        out,
        "layout(version={}, class={:?}, bytes={}, storage=",
        layout.message.version,
        layout.message.layout_class,
        layout.raw.len()
    )?;
    match layout.message.layout_class {
        crate::format::messages::data_layout::LayoutClass::Compact => write!(
            out,
            "{}",
            layout
                .message
                .compact_data
                .as_ref()
                .map(|data| data.len())
                .unwrap_or(0)
        )?,
        crate::format::messages::data_layout::LayoutClass::Contiguous => write!(
            out,
            "addr={:?}, size={:?}",
            layout.message.contiguous_addr, layout.message.contiguous_size
        )?,
        crate::format::messages::data_layout::LayoutClass::Chunked => write!(
            out,
            "dims={:?}, index={:?}",
            layout.message.chunk_dims, layout.message.chunk_index_type
        )?,
        crate::format::messages::data_layout::LayoutClass::Virtual => write!(
            out,
            "heap={:?}, index={:?}",
            layout.message.virtual_heap_addr, layout.message.virtual_heap_index
        )?,
    }
    out.write_char(')')
}

/// Return a debug-friendly representation of an object.
#[allow(non_snake_case)]
#[deprecated(note = "use H5O__layout_debug_fmt to write into a caller-provided formatter")]
pub fn H5O__layout_debug(layout: &LayoutObjectMessage) -> String {
    let mut text = String::new();
    H5O__layout_debug_fmt(layout, &mut text).expect("writing to String cannot fail");
    text
}

/// Decode an object from its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__refcount_decode(bytes: &[u8]) -> Result<u32> {
    if bytes.len() < 4 {
        return Err(Error::InvalidFormat(
            "object refcount message is truncated".into(),
        ));
    }
    let refcount = read_le_u32_at(bytes, 0, "object refcount message")?;
    if refcount == 0 {
        return Err(Error::InvalidFormat(
            "object refcount message stores zero links".into(),
        ));
    }
    Ok(refcount)
}

/// Encode an object to its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__refcount_encode_into(refcount: u32, out: &mut Vec<u8>) {
    out.extend_from_slice(&refcount.to_le_bytes());
}

/// Encode an object to its on-disk representation.
#[allow(non_snake_case)]
#[deprecated(note = "use H5O__refcount_encode_into to append into a caller-owned buffer")]
pub fn H5O__refcount_encode(refcount: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(4);
    H5O__refcount_encode_into(refcount, &mut out);
    out
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__refcount_copy(refcount: u32) -> u32 {
    let mut dest = 0u32;
    if refcount != 0 {
        dest = refcount;
    }
    dest
}

/// Object operation: refcount size.
#[allow(non_snake_case)]
pub fn H5O__refcount_size(_refcount: u32) -> usize {
    4
}

/// Free an object's in-memory resources.
#[allow(non_snake_case)]
pub fn H5O__refcount_free(mut refcount: u32) {
    let original = refcount;
    refcount = 0;
    std::hint::black_box((original, refcount));
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__refcount_pre_copy_file(refcount: u32) -> u32 {
    refcount
}

/// Return a debug-friendly representation of an object.
#[allow(non_snake_case)]
pub fn H5O__refcount_debug_fmt(refcount: u32, out: &mut dyn fmt::Write) -> fmt::Result {
    let state = if refcount == 0 { "invalid" } else { "defined" };
    write!(out, "refcount(value={refcount}, state={state})")
}

/// Return a debug-friendly representation of an object.
#[allow(non_snake_case)]
#[deprecated(note = "use H5O__refcount_debug_fmt to write into a caller-provided formatter")]
pub fn H5O__refcount_debug(refcount: u32) -> String {
    let mut text = String::new();
    H5O__refcount_debug_fmt(refcount, &mut text).expect("writing to String cannot fail");
    text
}

/// Decode an object from its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__fsinfo_decode(bytes: &[u8]) -> Result<FsInfoMessage> {
    let sizeof_addr = 8;
    let sizeof_size = 8;
    let mut pos = 0usize;
    let version = read_u8_cursor(bytes, &mut pos, "file-space info version")?;
    if version != 1 {
        return Err(Error::InvalidFormat(format!(
            "file-space info message version {version}"
        )));
    }
    let free_space_strategy = read_u8_cursor(bytes, &mut pos, "file-space info strategy")?;
    let persist_byte = read_u8_cursor(bytes, &mut pos, "file-space info persist")?;
    if persist_byte > 1 {
        return Err(Error::InvalidFormat(
            "file-space info persist flag is invalid".into(),
        ));
    }
    let persist = persist_byte != 0;
    let threshold = read_le_uint_cursor(bytes, &mut pos, sizeof_size, "file-space info threshold")?;
    let page_size = read_le_uint_cursor(bytes, &mut pos, sizeof_size, "file-space info page size")?;
    if page_size == 0 || page_size > 1024 * 1024 * 1024 {
        return Err(Error::InvalidFormat(
            "file-space info page size is invalid".into(),
        ));
    }
    let pgend_meta_thres = u16::try_from(read_le_uint_cursor(
        bytes,
        &mut pos,
        2,
        "file-space info page-end metadata threshold",
    )?)
    .map_err(|_| Error::InvalidFormat("file-space info page-end threshold exceeds u16".into()))?;
    let eoa_pre_fsm_fsalloc = read_le_uint_cursor(
        bytes,
        &mut pos,
        sizeof_addr,
        "file-space info pre-free-space EOA",
    )?;
    let mut fs_addr = Vec::new();
    if persist {
        fs_addr.reserve(12);
        for _ in 0..12 {
            fs_addr.push(read_le_uint_cursor(
                bytes,
                &mut pos,
                sizeof_addr,
                "file-space info free-space-manager address",
            )?);
        }
    }
    Ok(FsInfoMessage {
        version,
        free_space_strategy,
        persist,
        threshold,
        page_size,
        pgend_meta_thres,
        eoa_pre_fsm_fsalloc,
        fs_addr,
        sizeof_addr: sizeof_addr as u8,
        sizeof_size: sizeof_size as u8,
    })
}

/// Decode an object from its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__fsinfo_decode_with_sizes(
    bytes: &[u8],
    sizeof_addr: u8,
    sizeof_size: u8,
) -> Result<FsInfoMessage> {
    let mut pos = 0usize;
    let version = read_u8_cursor(bytes, &mut pos, "file-space info version")?;
    if version != 1 {
        return Err(Error::InvalidFormat(format!(
            "file-space info message version {version}"
        )));
    }
    let free_space_strategy = read_u8_cursor(bytes, &mut pos, "file-space info strategy")?;
    let persist = read_u8_cursor(bytes, &mut pos, "file-space info persist")? != 0;
    let threshold = read_le_uint_cursor(
        bytes,
        &mut pos,
        usize::from(sizeof_size),
        "file-space info threshold",
    )?;
    let page_size = read_le_uint_cursor(
        bytes,
        &mut pos,
        usize::from(sizeof_size),
        "file-space info page size",
    )?;
    if page_size == 0 || page_size > 1024 * 1024 * 1024 {
        return Err(Error::InvalidFormat(
            "file-space info page size is invalid".into(),
        ));
    }
    let pgend_meta_thres = u16::try_from(read_le_uint_cursor(
        bytes,
        &mut pos,
        2,
        "file-space info page-end metadata threshold",
    )?)
    .map_err(|_| Error::InvalidFormat("file-space info page-end threshold exceeds u16".into()))?;
    let eoa_pre_fsm_fsalloc = read_le_uint_cursor(
        bytes,
        &mut pos,
        usize::from(sizeof_addr),
        "file-space info pre-free-space EOA",
    )?;
    let mut fs_addr = Vec::new();
    if persist {
        fs_addr.reserve(12);
        for _ in 0..12 {
            fs_addr.push(read_le_uint_cursor(
                bytes,
                &mut pos,
                usize::from(sizeof_addr),
                "file-space info free-space-manager address",
            )?);
        }
    }
    Ok(FsInfoMessage {
        version,
        free_space_strategy,
        persist,
        threshold,
        page_size,
        pgend_meta_thres,
        eoa_pre_fsm_fsalloc,
        fs_addr,
        sizeof_addr,
        sizeof_size,
    })
}

/// Encode an object to its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__fsinfo_encode_into(info: &FsInfoMessage, out: &mut Vec<u8>) -> Result<()> {
    if !H5O_fsinfo_check_version(info) {
        return Err(Error::InvalidFormat(format!(
            "file-space info message version {}",
            info.version
        )));
    }
    if info.page_size == 0 || info.page_size > 1024 * 1024 * 1024 {
        return Err(Error::InvalidFormat(
            "file-space info page size is invalid".into(),
        ));
    }
    let image_len = H5O__fsinfo_image_len(info)?;
    let mut image = Vec::with_capacity(image_len);
    image.push(info.version);
    image.push(info.free_space_strategy);
    image.push(u8::from(info.persist));
    encode_le_uint_width(
        &mut image,
        info.threshold,
        usize::from(info.sizeof_size),
        "file-space info threshold",
    )?;
    encode_le_uint_width(
        &mut image,
        info.page_size,
        usize::from(info.sizeof_size),
        "file-space info page size",
    )?;
    image.extend_from_slice(&info.pgend_meta_thres.to_le_bytes());
    encode_le_uint_width(
        &mut image,
        info.eoa_pre_fsm_fsalloc,
        usize::from(info.sizeof_addr),
        "file-space info pre-free-space EOA",
    )?;
    if info.persist {
        if info.fs_addr.len() != 12 {
            return Err(Error::InvalidFormat(
                "file-space info persistent address count is invalid".into(),
            ));
        }
        for &addr in &info.fs_addr {
            encode_le_uint_width(
                &mut image,
                addr,
                usize::from(info.sizeof_addr),
                "file-space info free-space-manager address",
            )?;
        }
    }
    out.extend_from_slice(&image);
    Ok(())
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__fsinfo_copy(info: &FsInfoMessage) -> FsInfoMessage {
    FsInfoMessage {
        version: info.version,
        free_space_strategy: info.free_space_strategy,
        persist: info.persist,
        threshold: info.threshold,
        page_size: info.page_size,
        pgend_meta_thres: info.pgend_meta_thres,
        eoa_pre_fsm_fsalloc: info.eoa_pre_fsm_fsalloc,
        fs_addr: info.fs_addr.clone(),
        sizeof_addr: info.sizeof_addr,
        sizeof_size: info.sizeof_size,
    }
}

/// Object operation: fsinfo size.
#[allow(non_snake_case)]
pub fn H5O__fsinfo_image_len(info: &FsInfoMessage) -> Result<usize> {
    if !H5O_fsinfo_check_version(info) {
        return Err(Error::InvalidFormat(format!(
            "file-space info message version {}",
            info.version
        )));
    }
    if info.persist && info.fs_addr.len() != 12 {
        return Err(Error::InvalidFormat(
            "file-space info persistent address count is invalid".into(),
        ));
    }
    if !(1..=8).contains(&info.sizeof_addr) || !(1..=8).contains(&info.sizeof_size) {
        return Err(Error::InvalidFormat(
            "file-space info address or size width is invalid".into(),
        ));
    }
    let base = 3usize
        .checked_add(usize::from(info.sizeof_size))
        .and_then(|value| value.checked_add(usize::from(info.sizeof_size)))
        .and_then(|value| value.checked_add(2))
        .and_then(|value| value.checked_add(usize::from(info.sizeof_addr)))
        .ok_or_else(|| Error::InvalidFormat("file-space info image length overflow".into()))?;
    if info.persist {
        base.checked_add(12usize.saturating_mul(usize::from(info.sizeof_addr)))
            .ok_or_else(|| Error::InvalidFormat("file-space info image length overflow".into()))
    } else {
        Ok(base)
    }
}

/// Free an object's in-memory resources.
#[allow(non_snake_case)]
pub fn H5O__fsinfo_free(mut info: FsInfoMessage) {
    info.fs_addr.clear();
    info.version = 0;
    info.free_space_strategy = 0;
    info.persist = false;
    info.threshold = 0;
    info.page_size = 0;
    info.pgend_meta_thres = 0;
    info.eoa_pre_fsm_fsalloc = 0;
    info.sizeof_addr = 0;
    info.sizeof_size = 0;
    drop(info);
}

/// Return a debug-friendly representation of an object.
#[allow(non_snake_case)]
pub fn H5O__fsinfo_debug_fmt(info: &FsInfoMessage, out: &mut dyn fmt::Write) -> fmt::Result {
    let first_persist_addr = info.fs_addr.iter().find(|&&addr| addr != u64::MAX).copied();
    write!(
        out,
        "fsinfo(version={}, strategy={}, persist={}, threshold={}, page_size={}, page_end_meta={}, eoa_pre_fsm={}, fs_addr_count={}, first_fs_addr=",
        info.version,
        info.free_space_strategy,
        info.persist,
        info.threshold,
        info.page_size,
        info.pgend_meta_thres,
        info.eoa_pre_fsm_fsalloc,
        info.fs_addr.len()
    )?;
    if let Some(addr) = first_persist_addr {
        write!(out, "{addr:#x}")?;
    } else {
        out.write_str("none")?;
    }
    out.write_char(')')
}

/// Object operation: fsinfo set version.
#[allow(non_snake_case)]
pub fn H5O_fsinfo_set_version(info: &mut FsInfoMessage, version: u8) {
    info.version = version;
    if version != 1 {
        info.persist = false;
        info.fs_addr.clear();
    } else if info.persist && info.fs_addr.is_empty() {
        info.fs_addr.resize(12, u64::MAX);
    }
}

/// Object operation: fsinfo check version.
#[allow(non_snake_case)]
pub fn H5O_fsinfo_check_version(info: &FsInfoMessage) -> bool {
    info.version == 1
}

/// Decode an object from its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__stab_decode(bytes: &[u8]) -> Result<SymbolTableMessage> {
    let width = 8usize;
    if bytes.len() < width * 2 {
        return Err(Error::InvalidFormat(
            "symbol table message is truncated".into(),
        ));
    }
    let btree_addr = read_le_uint_width(bytes, 0, width, "symbol table B-tree address")?;
    let heap_addr = read_le_uint_width(bytes, width, width, "symbol table heap address")?;
    if btree_addr == u64::MAX {
        return Err(Error::InvalidFormat(
            "symbol table B-tree address is undefined".into(),
        ));
    }
    if heap_addr == u64::MAX {
        return Err(Error::InvalidFormat(
            "symbol table heap address is undefined".into(),
        ));
    }
    Ok(SymbolTableMessage {
        btree_addr,
        heap_addr,
    })
}

/// Decode an object from its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__stab_decode_with_size(bytes: &[u8], sizeof_addr: u8) -> Result<SymbolTableMessage> {
    let width = usize::from(sizeof_addr);
    let expected = width
        .checked_mul(2)
        .ok_or_else(|| Error::InvalidFormat("symbol table message width overflow".into()))?;
    if bytes.len() < expected {
        return Err(Error::InvalidFormat(
            "symbol table message is truncated".into(),
        ));
    }
    let btree_addr = read_le_uint_width(bytes, 0, width, "symbol table B-tree address")?;
    let heap_addr = read_le_uint_width(bytes, width, width, "symbol table heap address")?;
    if is_undefined_addr_width(btree_addr, sizeof_addr)? {
        return Err(Error::InvalidFormat(
            "symbol table B-tree address is undefined".into(),
        ));
    }
    if is_undefined_addr_width(heap_addr, sizeof_addr)? {
        return Err(Error::InvalidFormat(
            "symbol table heap address is undefined".into(),
        ));
    }
    Ok(SymbolTableMessage {
        btree_addr,
        heap_addr,
    })
}

/// Encode an object to its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__stab_encode(stab: &SymbolTableMessage) -> Result<Vec<u8>> {
    if stab.btree_addr == u64::MAX {
        return Err(Error::InvalidFormat(
            "symbol table B-tree address is undefined".into(),
        ));
    }
    if stab.heap_addr == u64::MAX {
        return Err(Error::InvalidFormat(
            "symbol table heap address is undefined".into(),
        ));
    }
    let mut out = Vec::with_capacity(16);
    out.extend_from_slice(&stab.btree_addr.to_le_bytes());
    out.extend_from_slice(&stab.heap_addr.to_le_bytes());
    Ok(out)
}

/// Encode an object to its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__stab_encode_with_size(stab: &SymbolTableMessage, sizeof_addr: u8) -> Result<Vec<u8>> {
    if is_undefined_addr_width(stab.btree_addr, sizeof_addr)? {
        return Err(Error::InvalidFormat(
            "symbol table B-tree address is undefined".into(),
        ));
    }
    if is_undefined_addr_width(stab.heap_addr, sizeof_addr)? {
        return Err(Error::InvalidFormat(
            "symbol table heap address is undefined".into(),
        ));
    }
    let width = usize::from(sizeof_addr);
    let len = width
        .checked_mul(2)
        .ok_or_else(|| Error::InvalidFormat("symbol table message width overflow".into()))?;
    let mut out = Vec::with_capacity(len);
    encode_le_uint_width(
        &mut out,
        stab.btree_addr,
        width,
        "symbol table B-tree address",
    )?;
    encode_le_uint_width(&mut out, stab.heap_addr, width, "symbol table heap address")?;
    Ok(out)
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__stab_copy(stab: &SymbolTableMessage) -> SymbolTableMessage {
    SymbolTableMessage {
        btree_addr: stab.btree_addr,
        heap_addr: stab.heap_addr,
    }
}

/// Object operation: stab size.
#[allow(non_snake_case)]
pub fn H5O__stab_size(stab: &SymbolTableMessage) -> Result<usize> {
    H5O__stab_size_with_size(stab, 8)
}

/// Object operation: stab size with size.
#[allow(non_snake_case)]
pub fn H5O__stab_size_with_size(stab: &SymbolTableMessage, sizeof_addr: u8) -> Result<usize> {
    Ok(H5O__stab_encode_with_size(stab, sizeof_addr)?.len())
}

/// Free an object's in-memory resources.
#[allow(non_snake_case)]
pub fn H5O__stab_free(mut stab: SymbolTableMessage) {
    stab.btree_addr = u64::MAX;
    stab.heap_addr = u64::MAX;
    drop(stab);
}

/// Delete an object.
#[allow(non_snake_case)]
pub fn H5O__stab_delete(stab: &mut SymbolTableMessage) {
    *stab = SymbolTableMessage::default();
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__stab_copy_file(stab: &SymbolTableMessage) -> SymbolTableMessage {
    SymbolTableMessage {
        btree_addr: stab.btree_addr,
        heap_addr: stab.heap_addr,
    }
}

/// Return a debug-friendly representation of an object.
#[allow(non_snake_case)]
pub fn H5O__stab_debug_fmt(stab: &SymbolTableMessage, out: &mut dyn fmt::Write) -> fmt::Result {
    let btree_state = if stab.btree_addr == u64::MAX {
        "undefined"
    } else {
        "defined"
    };
    let heap_state = if stab.heap_addr == u64::MAX {
        "undefined"
    } else {
        "defined"
    };
    write!(
        out,
        "stab(btree={:#x}:{}, heap={:#x}:{})",
        stab.btree_addr, btree_state, stab.heap_addr, heap_state
    )
}

/// Return a debug-friendly representation of an object.
#[allow(non_snake_case)]
#[deprecated(note = "use H5O__stab_debug_fmt to write into a caller-provided formatter")]
pub fn H5O__stab_debug(stab: &SymbolTableMessage) -> String {
    let mut text = String::new();
    H5O__stab_debug_fmt(stab, &mut text).expect("writing to String cannot fail");
    text
}

/// Decode an object from its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__sdspace_decode(bytes: &[u8]) -> Result<DataspaceObjectMessage> {
    if bytes.is_empty() {
        return Err(Error::InvalidFormat(
            "dataspace message is truncated".into(),
        ));
    }
    let version = bytes[0];
    if !(1..=2).contains(&version) {
        return Err(Error::InvalidFormat(format!(
            "dataspace message version {version}"
        )));
    }
    let message = DataspaceMessage::decode(bytes)?;
    if message.version != bytes[0] {
        return Err(Error::InvalidFormat(
            "dataspace message version does not match payload".into(),
        ));
    }
    if usize::from(message.ndims) != message.dims.len() {
        return Err(Error::InvalidFormat(
            "dataspace message rank does not match dimension count".into(),
        ));
    }
    if let Some(max_dims) = &message.max_dims {
        if max_dims.len() != message.dims.len() {
            return Err(Error::InvalidFormat(
                "dataspace message max rank does not match dimension count".into(),
            ));
        }
        for (&dim, &max_dim) in message.dims.iter().zip(max_dims) {
            if max_dim != u64::MAX && dim > max_dim {
                return Err(Error::InvalidFormat(
                    "dataspace current dimension exceeds max dimension".into(),
                ));
            }
        }
    }
    Ok(DataspaceObjectMessage {
        message,
        raw: bytes.to_vec(),
    })
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__sdspace_copy(space: &DataspaceObjectMessage) -> DataspaceObjectMessage {
    DataspaceObjectMessage {
        message: space.message.clone(),
        raw: space.raw.clone(),
    }
}

/// Object operation: sdspace size.
#[allow(non_snake_case)]
pub fn H5O__sdspace_size(space: &DataspaceObjectMessage) -> usize {
    space.raw.len()
}

/// Reset an object to its default state.
#[allow(non_snake_case)]
pub fn H5O__sdspace_reset(space: &mut DataspaceObjectMessage) {
    space.message.dims.clear();
    space.message.max_dims = None;
    space.raw.clear();
}

/// Free an object's in-memory resources.
#[allow(non_snake_case)]
pub fn H5O__sdspace_free(mut space: DataspaceObjectMessage) {
    space.message.dims.clear();
    space.message.max_dims = None;
    space.raw.clear();
    drop(space);
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__sdspace_pre_copy_file(space: &DataspaceObjectMessage) -> DataspaceObjectMessage {
    let mut copied = H5O__sdspace_copy(space);
    if copied.raw.is_empty() {
        copied.raw = space.raw.clone();
    }
    copied
}

/// Return a debug-friendly representation of an object.
#[allow(non_snake_case)]
pub fn H5O__sdspace_debug_fmt(
    space: &DataspaceObjectMessage,
    out: &mut dyn fmt::Write,
) -> fmt::Result {
    write!(
        out,
        "sdspace(version={}, type={:?}, ndims={}, dims={:?}, max_dims=",
        space.message.version, space.message.space_type, space.message.ndims, space.message.dims
    )?;
    match &space.message.max_dims {
        Some(dims) => write!(out, "{dims:?}")?,
        None => out.write_str("none")?,
    }
    write!(out, ", raw={} bytes)", space.raw.len())
}

/// Return a debug-friendly representation of an object.
#[allow(non_snake_case)]
#[deprecated(note = "use H5O__sdspace_debug_fmt to write into a caller-provided formatter")]
pub fn H5O__sdspace_debug(space: &DataspaceObjectMessage) -> String {
    let mut text = String::new();
    H5O__sdspace_debug_fmt(space, &mut text).expect("writing to String cannot fail");
    text
}

/// Link an object.
#[allow(non_snake_case)]
pub fn H5Olink(header: &mut ObjectHeaderState, delta: i32) {
    if delta.is_negative() {
        let decrease = delta.unsigned_abs();
        header.refcount = header.refcount.saturating_sub(decrease);
    } else if let Ok(increase) = u32::try_from(delta) {
        header.refcount = header.refcount.saturating_add(increase);
    }
    if header.refcount == 0 {
        for message in &mut header.messages {
            if message.shared {
                message.flags |= 0x02;
            } else if message.msg_type != 0 {
                message.flags &= !0x02;
            }
        }
        header.flush_disabled = false;
    } else {
        for message in &mut header.messages {
            if message.data.is_empty() {
                message.flags &= !0x01;
            } else {
                message.flags |= 0x01;
            }
        }
        header.flush_disabled = false;
    }
}

/// Link an object.
#[allow(non_snake_case)]
pub fn H5Olink_checked(header: &mut ObjectHeaderState, delta: i32) -> Result<()> {
    if delta.is_negative() {
        header.refcount = header
            .refcount
            .checked_sub(delta.unsigned_abs())
            .ok_or_else(|| Error::InvalidFormat("object header refcount underflow".into()))?;
    } else {
        let delta = u32::try_from(delta)
            .map_err(|_| Error::InvalidFormat("object header refcount delta overflow".into()))?;
        header.refcount = header
            .refcount
            .checked_add(delta)
            .ok_or_else(|| Error::InvalidFormat("object header refcount overflow".into()))?;
    }
    Ok(())
}

/// Object operation: incr refcount.
#[allow(non_snake_case)]
pub fn H5Oincr_refcount(header: &mut ObjectHeaderState) {
    header.refcount = header.refcount.saturating_add(1);
    if header.refcount > 0 {
        header.flush_disabled = false;
    }
}

/// Object operation: decr refcount.
#[allow(non_snake_case)]
pub fn H5Odecr_refcount(header: &mut ObjectHeaderState) {
    header.refcount = header.refcount.saturating_sub(1);
    if header.refcount == 0 {
        header.flush_disabled = false;
    }
}

/// Object operation: exists by name.
#[allow(non_snake_case)]
pub fn H5Oexists_by_name(objects: &BTreeMap<String, ObjectHeaderState>, name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    objects
        .get(name)
        .map(|header| header.refcount > 0 || !header.messages.is_empty())
        .unwrap_or(false)
}

/// Attach a comment to an object.
#[allow(non_snake_case)]
pub fn H5Oset_comment(header: &mut ObjectHeaderState, comment: impl Into<String>) {
    let comment = comment.into();
    if comment.is_empty() {
        header.comment = None;
    } else {
        header.comment = Some(comment);
    }
    header.flush_disabled = false;
}

/// Attach a comment to an object.
#[allow(non_snake_case)]
pub fn H5Oset_comment_by_name(
    objects: &mut BTreeMap<String, ObjectHeaderState>,
    name: &str,
    comment: impl Into<String>,
) -> Result<()> {
    let header = objects
        .get_mut(name)
        .ok_or_else(|| Error::InvalidFormat(format!("object '{name}' not found")))?;
    H5Oset_comment(header, comment);
    Ok(())
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ObjectLocation {
    pub file_name: Option<String>,
    pub addr: u64,
    pub held: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ObjectInfo {
    pub addr: u64,
    pub refcount: u32,
    pub msg_count: usize,
    pub has_checksum: bool,
}

/// Object operation: visit3.
#[allow(non_snake_case)]
pub fn H5Ovisit3_refs(objects: &BTreeMap<String, ObjectHeaderState>) -> Vec<&str> {
    let mut ordered = Vec::with_capacity(objects.len());
    let mut seen_addrs = BTreeSet::new();
    for (name, header) in objects {
        if header.refcount == 0 && header.messages.is_empty() {
            continue;
        }
        if name.is_empty() {
            continue;
        }
        if header.addr == u64::MAX && header.messages.is_empty() {
            continue;
        }
        if header.addr != u64::MAX && !seen_addrs.insert(header.addr) {
            continue;
        }
        let depth = name.split('/').filter(|part| !part.is_empty()).count();
        let has_attrs = header
            .messages
            .iter()
            .any(|message| message.msg_type == 0x000c);
        let has_links = header
            .messages
            .iter()
            .any(|message| matches!(message.msg_type, 0x0006 | 0x000a | 0x000b | 0x0011));
        ordered.push((
            depth,
            name.as_str(),
            header.addr,
            header.refcount,
            has_links,
            has_attrs,
        ));
    }
    ordered.sort_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| left.2.cmp(&right.2))
            .then_with(|| right.3.cmp(&left.3))
            .then_with(|| right.4.cmp(&left.4))
            .then_with(|| right.5.cmp(&left.5))
    });
    let mut names = Vec::with_capacity(ordered.len());
    let mut seen_names = BTreeSet::new();
    for (_, name, _, _, _, _) in ordered {
        if seen_names.insert(name) {
            names.push(name);
        }
    }
    names
}

/// Object operation: visit3.
#[allow(non_snake_case)]
#[deprecated(note = "use H5Ovisit3_refs to avoid allocating cloned object names")]
pub fn H5Ovisit3(objects: &BTreeMap<String, ObjectHeaderState>) -> Vec<String> {
    H5Ovisit3_refs(objects)
        .into_iter()
        .map(str::to_owned)
        .collect()
}

/// Object operation: are mdc flushes disabled.
#[allow(non_snake_case)]
pub fn H5O__are_mdc_flushes_disabled(header: &ObjectHeaderState) -> bool {
    header.flush_disabled
}

/// Object operation: are mdc flushes disabled.
#[allow(non_snake_case)]
pub fn H5Oare_mdc_flushes_disabled(header: &ObjectHeaderState) -> bool {
    if header.refcount == 0 {
        return false;
    }
    if header.addr == u64::MAX && header.messages.is_empty() {
        return false;
    }
    header.flush_disabled
}

/// Object operation: token cmp.
#[allow(non_snake_case)]
pub fn H5Otoken_cmp(left: u64, right: u64) -> std::cmp::Ordering {
    match (left == u64::MAX, right == u64::MAX) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => left.cmp(&right),
    }
}

/// Object operation: token to str.
#[allow(non_snake_case)]
pub fn H5Otoken_fmt(token: u64, out: &mut dyn fmt::Write) -> fmt::Result {
    if token == u64::MAX {
        out.write_str("UNDEF")
    } else {
        write!(out, "{token:#016x}")
    }
}

/// Object operation: token to str.
#[allow(non_snake_case)]
#[deprecated(note = "use H5Otoken_fmt to write into a caller-provided formatter")]
pub fn H5Otoken_to_str(token: u64) -> String {
    let mut text = String::new();
    H5Otoken_fmt(token, &mut text).expect("writing to String cannot fail");
    text
}

/// Object operation: token from str.
#[allow(non_snake_case)]
pub fn H5Otoken_from_str(token: &str) -> Result<u64> {
    let trimmed = token.strip_prefix("0x").unwrap_or(token);
    u64::from_str_radix(trimmed, 16)
        .map_err(|_| Error::InvalidFormat("invalid object token".into()))
}

/// Object operation: print time field.
#[allow(non_snake_case)]
pub fn H5O__print_time_field_fmt(timestamp: u64, out: &mut dyn fmt::Write) -> fmt::Result {
    write!(out, "{timestamp}")
}

/// Object operation: print time field.
#[allow(non_snake_case)]
#[deprecated(note = "use H5O__print_time_field_fmt to write into a caller-provided formatter")]
pub fn H5O__print_time_field(timestamp: u64) -> String {
    let mut text = String::new();
    H5O__print_time_field_fmt(timestamp, &mut text).expect("writing to String cannot fail");
    text
}

/// Object operation: assert.
#[allow(non_snake_case)]
pub fn H5O__assert(header: &ObjectHeaderState) -> Result<()> {
    if header.addr == u64::MAX && (!header.messages.is_empty() || header.refcount != 0) {
        return Err(Error::InvalidFormat(
            "undefined object header address has live state".into(),
        ));
    }
    if header.refcount == 0 && !header.messages.is_empty() {
        return Err(Error::InvalidFormat(
            "object header with messages has zero refcount".into(),
        ));
    }

    let mut free_space = 0usize;
    let mut mesg_space = 0usize;
    let mut meta_space = 0usize;
    let mut continuation_messages = 0usize;
    let mut previous_crt_idx = None;
    for (idx, message) in header.messages.iter().enumerate() {
        if message.msg_type == 0 {
            if message.shared {
                return Err(Error::InvalidFormat(
                    "null object header message is marked shared".into(),
                ));
            }
            if message.flags != 0 {
                return Err(Error::InvalidFormat(
                    "null object header message has nonzero flags".into(),
                ));
            }
            free_space = free_space
                .checked_add(message.data.len())
                .ok_or_else(|| Error::InvalidFormat("object header free-space overflow".into()))?;
        } else {
            if message.data.is_empty() {
                return Err(Error::InvalidFormat(
                    "non-null object header message has empty payload".into(),
                ));
            }
            if message.shared && message.flags & 0x02 == 0 {
                return Err(Error::InvalidFormat(
                    "shared object header message is missing shared flag".into(),
                ));
            }
            if !message.shared && message.flags & 0x02 != 0 {
                return Err(Error::InvalidFormat(
                    "unshared object header message carries shared flag".into(),
                ));
            }
            if message.flags & 0x20 != 0 && message.flags & 0x10 == 0 {
                return Err(Error::InvalidFormat(
                    "was-unknown object header message lacks mark-if-unknown".into(),
                ));
            }
            meta_space = meta_space
                .checked_add(5)
                .ok_or_else(|| Error::InvalidFormat("object header metadata overflow".into()))?;
            mesg_space = mesg_space.checked_add(message.data.len()).ok_or_else(|| {
                Error::InvalidFormat("object header message-space overflow".into())
            })?;
            if message.msg_type == 0x0010 {
                continuation_messages = continuation_messages.checked_add(1).ok_or_else(|| {
                    Error::InvalidFormat("object header continuation count overflow".into())
                })?;
                if message.data.len() != 16 {
                    return Err(Error::InvalidFormat(
                        "continuation object header message has invalid payload size".into(),
                    ));
                }
            }
        }
        if let Some(prev) = previous_crt_idx {
            if message.creation_index < prev && message.msg_type != 0 {
                return Err(Error::InvalidFormat(
                    "object header messages are not in creation-index order".into(),
                ));
            }
        }
        previous_crt_idx = Some(message.creation_index);
        if usize::from(message.creation_index) > header.messages.len().saturating_add(idx) {
            return Err(Error::InvalidFormat(
                "object header message creation index is out of range".into(),
            ));
        }
    }
    let total_space = free_space
        .checked_add(meta_space)
        .and_then(|value| value.checked_add(mesg_space))
        .ok_or_else(|| Error::InvalidFormat("object header total-space overflow".into()))?;
    if total_space == 0 && !header.messages.is_empty() {
        return Err(Error::InvalidFormat(
            "object header messages account for no space".into(),
        ));
    }
    if continuation_messages > header.messages.len().saturating_sub(1) {
        return Err(Error::InvalidFormat(
            "object header has more continuation messages than chunks".into(),
        ));
    }
    Ok(())
}

/// Return a debug-friendly representation of an object.
#[allow(non_snake_case)]
pub fn H5O_debug_id(addr: u64) -> String {
    if addr == u64::MAX {
        "object@UNDEF".to_string()
    } else {
        format!("object@{addr:#016x}")
    }
}

/// Return a debug-friendly representation of an object.
#[allow(non_snake_case)]
pub fn H5O__debug_real_fmt(header: &ObjectHeaderState, out: &mut dyn fmt::Write) -> fmt::Result {
    let mut total_message_bytes = 0usize;
    let mut shared_messages = 0usize;
    let mut null_messages = 0usize;
    let mut max_message_size = 0usize;
    let mut type_counts = BTreeMap::new();
    for message in &header.messages {
        total_message_bytes = total_message_bytes.saturating_add(message.data.len());
        max_message_size = max_message_size.max(message.data.len());
        if message.shared {
            shared_messages = shared_messages.saturating_add(1);
        }
        if message.msg_type == 0 {
            null_messages = null_messages.saturating_add(1);
        }
        let entry = type_counts.entry(message.msg_type).or_insert(0usize);
        *entry = entry.saturating_add(1);
    }
    write!(
        out,
        "object(addr={:#x}, refcount={}, messages={}, message_bytes={}, shared_messages={}, null_messages={}, max_message_size={}, type_counts={:?}, comment={})",
        header.addr,
        header.refcount,
        header.messages.len(),
        total_message_bytes,
        shared_messages,
        null_messages,
        max_message_size,
        type_counts,
        header.comment.as_deref().unwrap_or("")
    )
}

/// Return a debug-friendly representation of an object.
#[allow(non_snake_case)]
pub fn H5O_debug_fmt(header: &ObjectHeaderState, out: &mut dyn fmt::Write) -> fmt::Result {
    H5O__debug_real_fmt(header, out)?;
    if header.flush_disabled {
        out.write_str(", flush_disabled=true")?;
    }
    Ok(())
}

/// Return a debug-friendly representation of an object.
#[allow(non_snake_case)]
#[deprecated(note = "use H5O_debug_fmt to write into a caller-provided formatter")]
pub fn H5O_debug(header: &ObjectHeaderState) -> String {
    let mut text = String::new();
    H5O_debug_fmt(header, &mut text).expect("writing to String cannot fail");
    text
}

/// Decode an object from its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__mdci_decode(bytes: &[u8]) -> Result<MetadataCacheImageMessage> {
    if bytes.len() < 17 {
        return Err(Error::InvalidFormat(
            "metadata cache image message is truncated".into(),
        ));
    }
    let version = bytes[0];
    if version != 0 {
        return Err(Error::InvalidFormat(format!(
            "metadata cache image message version {version}"
        )));
    }
    let addr = read_le_uint_width(bytes, 1, 8, "metadata cache image address")?;
    let size = read_le_uint_width(bytes, 9, 8, "metadata cache image size")?;
    if addr != u64::MAX && size != 0 && addr >= u64::MAX.saturating_sub(size) {
        return Err(Error::InvalidFormat(
            "metadata cache image address plus size overflows".into(),
        ));
    }
    Ok(MetadataCacheImageMessage {
        version,
        addr,
        size,
        sizeof_addr: 8,
        sizeof_size: 8,
    })
}

/// Decode an object from its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__mdci_decode_with_sizes(
    bytes: &[u8],
    sizeof_addr: u8,
    sizeof_size: u8,
) -> Result<MetadataCacheImageMessage> {
    let mut pos = 0usize;
    let version = read_u8_cursor(bytes, &mut pos, "metadata cache image version")?;
    if version != 0 {
        return Err(Error::InvalidFormat(format!(
            "metadata cache image message version {version}"
        )));
    }
    let addr = read_le_uint_cursor(
        bytes,
        &mut pos,
        usize::from(sizeof_addr),
        "metadata cache image address",
    )?;
    let size = read_le_uint_cursor(
        bytes,
        &mut pos,
        usize::from(sizeof_size),
        "metadata cache image size",
    )?;
    if !is_undefined_addr_width(addr, sizeof_addr)? && size != 0 {
        let undef = undefined_addr_value(sizeof_addr)?;
        if addr >= undef.saturating_sub(size) {
            return Err(Error::InvalidFormat(
                "metadata cache image address plus size overflows".into(),
            ));
        }
    }
    Ok(MetadataCacheImageMessage {
        version,
        addr,
        size,
        sizeof_addr,
        sizeof_size,
    })
}

/// Encode an object to its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__mdci_encode(message: &MetadataCacheImageMessage) -> Result<Vec<u8>> {
    if message.version != 0 {
        return Err(Error::InvalidFormat(format!(
            "metadata cache image message version {}",
            message.version
        )));
    }
    let mut out = Vec::with_capacity(H5O__mdci_size(message));
    out.push(message.version);
    encode_le_uint_width(
        &mut out,
        message.addr,
        usize::from(message.sizeof_addr),
        "metadata cache image address",
    )?;
    encode_le_uint_width(
        &mut out,
        message.size,
        usize::from(message.sizeof_size),
        "metadata cache image size",
    )?;
    Ok(out)
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__mdci_copy(message: &MetadataCacheImageMessage) -> MetadataCacheImageMessage {
    MetadataCacheImageMessage {
        version: message.version,
        addr: message.addr,
        size: message.size,
        sizeof_addr: message.sizeof_addr,
        sizeof_size: message.sizeof_size,
    }
}

/// Object operation: mdci size.
#[allow(non_snake_case)]
pub fn H5O__mdci_size(message: &MetadataCacheImageMessage) -> usize {
    1 + usize::from(message.sizeof_addr) + usize::from(message.sizeof_size)
}

/// Free an object's in-memory resources.
#[allow(non_snake_case)]
pub fn H5O__mdci_free(mut message: MetadataCacheImageMessage) {
    message.addr = u64::MAX;
    message.size = 0;
    message.sizeof_addr = 0;
    message.sizeof_size = 0;
    drop(message);
}

/// Delete an object.
#[allow(non_snake_case)]
pub fn H5O__mdci_delete(message: &mut MetadataCacheImageMessage) {
    match undefined_addr_value(message.sizeof_addr) {
        Ok(undefined) => {
            message.addr = undefined;
            message.size = 0;
        }
        Err(_) => {
            message.addr = u64::MAX;
            message.size = 0;
        }
    }
}

/// Delete an object.
#[allow(non_snake_case)]
pub fn H5O__mdci_delete_checked(message: &mut MetadataCacheImageMessage) -> Result<()> {
    message.addr = undefined_addr_value(message.sizeof_addr)?;
    message.size = 0;
    Ok(())
}

/// Return a debug-friendly representation of an object.
#[allow(non_snake_case)]
pub fn H5O__mdci_debug_fmt(
    message: &MetadataCacheImageMessage,
    out: &mut dyn fmt::Write,
) -> fmt::Result {
    let status = match is_undefined_addr_width(message.addr, message.sizeof_addr) {
        Ok(true) => "undefined",
        Ok(false) => "defined",
        Err(_) => "invalid-width",
    };
    write!(
        out,
        "mdci(version={}, addr={:#x}, size={}, addr_size={}, size_size={}, status={})",
        message.version,
        message.addr,
        message.size,
        message.sizeof_addr,
        message.sizeof_size,
        status
    )
}

/// Return a debug-friendly representation of an object.
#[allow(non_snake_case)]
#[deprecated(note = "use H5O__mdci_debug_fmt to write into a caller-provided formatter")]
pub fn H5O__mdci_debug(message: &MetadataCacheImageMessage) -> String {
    let mut text = String::new();
    H5O__mdci_debug_fmt(message, &mut text).expect("writing to String cannot fail");
    text
}

/// Decode an object from its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__attr_decode(bytes: &[u8]) -> Result<AttributeObjectMessage> {
    if bytes.len() < 6 {
        return Err(Error::InvalidFormat("attribute message too short".into()));
    }
    let version = bytes[0];
    let mut pos = match version {
        1 => {
            if bytes.len() < 8 {
                return Err(Error::InvalidFormat(
                    "attribute v1 header is truncated".into(),
                ));
            }
            8usize
        }
        2 => {
            if bytes.len() < 8 {
                return Err(Error::InvalidFormat(
                    "attribute v2 header is truncated".into(),
                ));
            }
            let flags = bytes[1];
            if flags & !0x03 != 0 {
                return Err(Error::InvalidFormat(format!(
                    "attribute message flags {flags:#x} are invalid"
                )));
            }
            8usize
        }
        3 => {
            if bytes.len() < 9 {
                return Err(Error::InvalidFormat(
                    "attribute v3 header is truncated".into(),
                ));
            }
            let flags = bytes[1];
            if flags & !0x03 != 0 {
                return Err(Error::InvalidFormat(format!(
                    "attribute message flags {flags:#x} are invalid"
                )));
            }
            if bytes[8] > 1 {
                return Err(Error::InvalidFormat(format!(
                    "invalid attribute character encoding {}",
                    bytes[8]
                )));
            }
            9usize
        }
        _ => {
            return Err(Error::InvalidFormat(format!(
                "attribute message version {version}"
            )));
        }
    };
    let name_size = usize::from(read_le_u16_at(bytes, 2, "attribute name size")?);
    let dtype_size = usize::from(read_le_u16_at(bytes, 4, "attribute datatype size")?);
    let dspace_size = usize::from(read_le_u16_at(bytes, 6, "attribute dataspace size")?);
    if name_size <= 1 {
        return Err(Error::InvalidFormat(
            "attribute message name length is invalid".into(),
        ));
    }

    let name_end = checked_add(pos, name_size, "attribute name")?;
    let name_bytes = bytes
        .get(pos..name_end)
        .ok_or_else(|| Error::InvalidFormat("attribute name is truncated".into()))?;
    let name_text_len = name_size
        .checked_sub(1)
        .ok_or_else(|| Error::InvalidFormat("attribute name length underflow".into()))?;
    if name_bytes.get(name_text_len).copied() != Some(0) || name_bytes[..name_text_len].contains(&0)
    {
        return Err(Error::InvalidFormat(
            "attribute name has different length than stored length".into(),
        ));
    }
    std::str::from_utf8(&name_bytes[..name_text_len])
        .map_err(|_| Error::InvalidFormat("attribute name is not UTF-8".into()))?;
    pos = if version == 1 {
        checked_add(
            pos,
            align8_len_checked(name_size, "attribute v1 name")?,
            "attribute v1 padded name",
        )?
    } else {
        name_end
    };

    let dtype_end = checked_add(pos, dtype_size, "attribute datatype")?;
    let dtype_bytes = bytes
        .get(pos..dtype_end)
        .ok_or_else(|| Error::InvalidFormat("attribute datatype is truncated".into()))?;
    DatatypeMessage::decode(dtype_bytes)?;
    pos = if version == 1 {
        checked_add(
            pos,
            align8_len_checked(dtype_size, "attribute v1 datatype")?,
            "attribute v1 padded datatype",
        )?
    } else {
        dtype_end
    };

    let dspace_end = checked_add(pos, dspace_size, "attribute dataspace")?;
    let dspace_bytes = bytes
        .get(pos..dspace_end)
        .ok_or_else(|| Error::InvalidFormat("attribute dataspace is truncated".into()))?;
    DataspaceMessage::decode(dspace_bytes)?;
    if version == 1 {
        let padded = align8_len_checked(dspace_size, "attribute v1 dataspace")?;
        let padded_end = checked_add(pos, padded, "attribute v1 padded dataspace")?;
        if bytes.get(pos..padded_end).is_none() {
            return Err(Error::InvalidFormat(
                "attribute v1 padded dataspace is truncated".into(),
            ));
        }
    }

    Ok(AttributeObjectMessage {
        message: AttributeMessage::decode(bytes)?,
        raw_size: bytes.len(),
    })
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__attr_copy(message: &AttributeObjectMessage) -> AttributeObjectMessage {
    AttributeObjectMessage {
        message: AttributeMessage {
            version: message.message.version,
            name: message.message.name.clone(),
            char_encoding: message.message.char_encoding,
            datatype: message.message.datatype.clone(),
            dataspace: message.message.dataspace.clone(),
            data: message.message.data.clone(),
        },
        raw_size: message.raw_size,
    }
}

/// Object operation: attr size.
#[allow(non_snake_case)]
pub fn H5O__attr_size(message: &AttributeObjectMessage) -> usize {
    if message.raw_size != 0 {
        message.raw_size
    } else {
        message.message.data_size().unwrap_or(usize::MAX)
    }
}

/// Free an object's in-memory resources.
#[allow(non_snake_case)]
pub fn H5O__attr_free(mut message: AttributeObjectMessage) {
    message.message.name.clear();
    message.message.data.clear();
    message.raw_size = 0;
    drop(message);
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__attr_pre_copy_file(message: &AttributeObjectMessage) -> AttributeObjectMessage {
    AttributeObjectMessage {
        message: AttributeMessage {
            version: message.message.version,
            name: message.message.name.to_string(),
            char_encoding: message.message.char_encoding,
            datatype: message.message.datatype.clone(),
            dataspace: message.message.dataspace.clone(),
            data: message.message.data.to_vec(),
        },
        raw_size: message.raw_size,
    }
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__attr_copy_file(message: &AttributeObjectMessage) -> AttributeObjectMessage {
    let mut copied = AttributeObjectMessage {
        message: AttributeMessage {
            version: message.message.version,
            name: message.message.name.to_string(),
            char_encoding: message.message.char_encoding,
            datatype: message.message.datatype.clone(),
            dataspace: message.message.dataspace.clone(),
            data: message.message.data.to_vec(),
        },
        raw_size: message.raw_size,
    };
    if copied.message.name.is_empty() {
        copied.raw_size = 0;
    }
    copied
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__attr_post_copy_file(message: &AttributeObjectMessage) -> AttributeObjectMessage {
    H5O__attr_copy(message)
}

/// Return a debug-friendly representation of an object.
#[allow(non_snake_case)]
pub fn H5O__attr_debug_fmt(
    message: &AttributeObjectMessage,
    out: &mut dyn fmt::Write,
) -> fmt::Result {
    write!(
        out,
        "attr(version={}, name={}, encoding={}, dtype_size={}, rank={}, data={} bytes, raw={} bytes)",
        message.message.version,
        message.message.name,
        message.message.char_encoding,
        message.message.datatype.size,
        message.message.dataspace.ndims,
        message.message.data.len(),
        message.raw_size
    )
}

/// Return a debug-friendly representation of an object.
#[allow(non_snake_case)]
#[deprecated(note = "use H5O__attr_debug_fmt to write into a caller-provided formatter")]
pub fn H5O__attr_debug(message: &AttributeObjectMessage) -> String {
    let mut text = String::new();
    H5O__attr_debug_fmt(message, &mut text).expect("writing to String cannot fail");
    text
}

/// Object operation: chunk add.
#[allow(non_snake_case)]
pub fn H5O__chunk_add(header: &mut ObjectHeaderState, message: ObjectMessage) {
    let mut message = message;
    if message.creation_index == 0 && !header.messages.is_empty() {
        message.creation_index = header
            .messages
            .iter()
            .map(|msg| msg.creation_index)
            .max()
            .unwrap_or(0)
            .saturating_add(1);
    }
    header.flush_disabled = false;
    H5O_msg_append_oh(header, message);
}

/// Object operation: chunk protect.
#[allow(non_snake_case)]
pub fn H5O__chunk_protect(header: &mut ObjectHeaderState) -> &mut ObjectHeaderState {
    header
}

/// Object operation: chunk unprotect.
#[allow(non_snake_case)]
pub fn H5O__chunk_unprotect(header: &mut ObjectHeaderState) {
    for message in &mut header.messages {
        if message.data.is_empty() {
            message.flags &= !0x01;
        } else {
            message.flags |= 0x01;
        }
        if message.shared {
            message.flags |= 0x02;
        } else {
            message.flags &= !0x02;
        }
    }
    for (idx, message) in header.messages.iter_mut().enumerate() {
        message.creation_index = u16::try_from(idx).unwrap_or(u16::MAX);
    }
    if header.refcount == 0 {
        header.flush_disabled = false;
    }
}

/// Object operation: chunk resize.
#[allow(non_snake_case)]
pub fn H5O__chunk_resize(message: &mut ObjectMessage, new_size: usize) {
    if new_size == 0 {
        message.data.clear();
        message.msg_type = 0;
        return;
    }
    message.data.resize(new_size, 0);
}

/// Object operation: chunk dest.
#[allow(non_snake_case)]
pub fn H5O__chunk_dest(message: ObjectMessage) {
    drop(message);
}

/// Update an object.
#[allow(non_snake_case)]
pub fn H5O__chunk_update_idx(header: &mut ObjectHeaderState) {
    for (idx, msg) in header.messages.iter_mut().enumerate() {
        msg.creation_index = u16::try_from(idx).unwrap_or(u16::MAX);
    }
    header.flush_disabled = false;
}

/// Object operation: add gap.
#[allow(non_snake_case)]
pub fn H5O__add_gap(header: &mut ObjectHeaderState, size: usize) {
    if size == 0 {
        return;
    }
    if let Some(message) = header
        .messages
        .last_mut()
        .filter(|message| message.msg_type == 0)
    {
        let old_len = message.data.len();
        message.data.resize(old_len.saturating_add(size), 0);
        message.flags = 0;
        message.shared = false;
    } else {
        let mut message = ObjectMessage {
            msg_type: 0,
            flags: 0,
            creation_index: u16::try_from(header.messages.len()).unwrap_or(u16::MAX),
            data: vec![0; size],
            shared: false,
        };
        if message.data.is_empty() {
            message.creation_index = 0;
        }
        header.messages.push(message);
        header.refcount = header.refcount.max(1);
    }
    header.flush_disabled = false;
}

/// Object operation: eliminate gap.
#[allow(non_snake_case)]
pub fn H5O__eliminate_gap(header: &mut ObjectHeaderState) {
    let mut compacted = Vec::with_capacity(header.messages.len());
    for message in header.messages.drain(..) {
        if message.msg_type == 0 && message.data.iter().all(|byte| *byte == 0) {
            if let Some(last) = compacted
                .last_mut()
                .filter(|last: &&mut ObjectMessage| last.msg_type == 0)
            {
                last.data.extend_from_slice(&message.data);
            } else if !message.data.is_empty() {
                compacted.push(message);
            }
        } else {
            compacted.push(message);
        }
    }
    header.messages = compacted;
    H5O__chunk_update_idx(header);
}

/// Allocate storage for an object.
#[allow(non_snake_case)]
pub fn H5O__alloc_null(size: usize) -> ObjectMessage {
    let mut data = Vec::with_capacity(size);
    data.resize(size, 0);
    ObjectMessage {
        msg_type: 0,
        flags: 0,
        creation_index: 0,
        data,
        shared: false,
    }
}

/// Allocate storage for an object.
#[allow(non_snake_case)]
pub fn H5O__alloc_msgs(header: &mut ObjectHeaderState, count: usize) {
    if count > header.messages.len() {
        header.messages.reserve(count - header.messages.len());
    }
}

/// Allocate storage for an object.
#[allow(non_snake_case)]
pub fn H5O__alloc_extend_chunk(header: &mut ObjectHeaderState, size: usize) {
    if size == 0 {
        return;
    }
    let aligned_size = size
        .checked_add(7)
        .map(|value| value & !7)
        .unwrap_or(usize::MAX);
    let message_header = 5usize;
    let mut trailing_null = None;
    if let Some((idx, message)) = header
        .messages
        .iter()
        .enumerate()
        .rev()
        .find(|(_, message)| message.msg_type == 0)
    {
        if header.messages[idx + 1..]
            .iter()
            .all(|message| message.msg_type == 0)
        {
            trailing_null = Some((idx, message.data.len()));
        }
    }
    if let Some((idx, current_len)) = trailing_null {
        let delta = aligned_size.saturating_sub(current_len);
        let message = &mut header.messages[idx];
        let new_len = current_len.saturating_add(delta.max(aligned_size));
        message.data.resize(new_len, 0);
        message.flags = 0;
        message.shared = false;
        H5O__merge_null(header);
        H5O__chunk_update_idx(header);
        return;
    }

    let mut best = None;
    let mut best_len = 0usize;
    for (idx, message) in header.messages.iter().enumerate() {
        if message.msg_type == 0 && message.data.len() >= best_len {
            best = Some(idx);
            best_len = message.data.len();
        }
    }
    if let Some(pos) = best {
        let message = &mut header.messages[pos];
        let new_len = message
            .data
            .len()
            .saturating_add(aligned_size)
            .saturating_add(message_header);
        message.data.resize(new_len, 0);
        message.flags = 0;
        message.shared = false;
    } else {
        H5O__add_gap(header, aligned_size.saturating_add(message_header));
    }
    if header.refcount == 0 && !header.messages.is_empty() {
        header.refcount = 1;
    }
    H5O__merge_null(header);
    H5O__chunk_update_idx(header);
}

/// Allocate storage for an object.
#[allow(non_snake_case)]
pub fn H5O__alloc_new_chunk(size: usize) -> Vec<u8> {
    let mut chunk = Vec::with_capacity(size);
    chunk.resize(size, 0);
    chunk
}

/// Allocate storage for an object.
#[allow(non_snake_case)]
pub fn H5O__alloc_find_best_null(header: &ObjectHeaderState) -> Option<usize> {
    header
        .messages
        .iter()
        .enumerate()
        .filter(|(_, msg)| msg.msg_type == 0)
        .max_by_key(|(_, msg)| msg.data.len())
        .map(|(idx, _)| idx)
}

/// Allocate storage for an object.
#[allow(non_snake_case)]
pub fn H5O__alloc_find_best_nonnull(header: &ObjectHeaderState, msg_type: u16) -> Option<usize> {
    let mut best_idx = None;
    let mut best_key = (usize::MAX, u16::MAX, u16::MAX);
    for (idx, msg) in header.messages.iter().enumerate() {
        if msg.msg_type == 0 || msg.msg_type == 0x0010 || msg.data.is_empty() {
            continue;
        }
        if msg_type != 0 && msg.msg_type != msg_type {
            continue;
        }
        let attr_penalty = if msg.msg_type == 0x000c { 1 } else { 0 };
        let key = (msg.data.len(), attr_penalty, msg.creation_index);
        if key < best_key {
            best_idx = Some(idx);
            best_key = key;
        }
    }
    best_idx
}

/// Allocate storage for an object.
#[allow(non_snake_case)]
pub fn H5O__alloc_chunk(size: usize) -> ObjectMessage {
    let aligned_size = size
        .checked_add(7)
        .map(|value| value & !7)
        .unwrap_or(usize::MAX);
    let message_header = 5usize;
    let chunk_magic = 4usize;
    let checksum_size = 4usize;
    let min_payload = 32usize;
    let payload_size = aligned_size.max(min_payload);
    let total_size = payload_size
        .checked_add(message_header)
        .and_then(|value| value.checked_add(chunk_magic))
        .and_then(|value| value.checked_add(checksum_size))
        .unwrap_or(usize::MAX);
    let mut message = H5O__alloc_null(total_size);
    if message.data.len() >= chunk_magic {
        message.data[..chunk_magic].copy_from_slice(OBJECT_HEADER_V2_CHUNK_MAGIC);
    }
    if message.data.len() >= checksum_size {
        let checksum_pos = message.data.len() - checksum_size;
        let checksum = checksum_metadata(&message.data[..checksum_pos]);
        message.data[checksum_pos..].copy_from_slice(&checksum.to_le_bytes());
    }
    message.creation_index = 0;
    message.flags = 0;
    message.shared = false;
    if message.data.len() > chunk_magic + message_header + checksum_size {
        let start = chunk_magic + message_header;
        let end = message.data.len() - checksum_size;
        for byte in &mut message.data[start..end] {
            *byte = 0;
        }
    }
    message
}

/// Allocate storage for an object.
#[allow(non_snake_case)]
pub fn H5O__alloc(header: &mut ObjectHeaderState, message: ObjectMessage) {
    let needed = message.data.len();
    if needed > 0 {
        if let Some(pos) = header
            .messages
            .iter()
            .enumerate()
            .filter(|(_, msg)| msg.msg_type == 0 && msg.data.len() >= needed)
            .min_by_key(|(_, msg)| msg.data.len())
            .map(|(idx, _)| idx)
        {
            let remainder = header.messages[pos].data.len().saturating_sub(needed);
            header.messages[pos] = ObjectMessage {
                msg_type: message.msg_type,
                flags: H5O_msg_get_flags(&message),
                creation_index: u16::try_from(pos).unwrap_or(u16::MAX),
                data: message.data,
                shared: message.shared,
            };
            if remainder > 0 {
                header.messages.insert(pos + 1, H5O__alloc_null(remainder));
            }
            H5O__chunk_update_idx(header);
            return;
        }
    }
    let mut appended = message;
    if appended.creation_index == 0 {
        appended.creation_index = u16::try_from(header.messages.len()).unwrap_or(u16::MAX);
    }
    header.messages.push(appended);
    header.refcount = header.refcount.max(1);
    header.flush_disabled = false;
}

/// Object operation: release mesg.
#[allow(non_snake_case)]
pub fn H5O__release_mesg(message: &mut ObjectMessage) {
    message.msg_type = 0;
    for byte in &mut message.data {
        *byte = 0;
    }
    message.flags = 0;
    message.shared = false;
    message.creation_index = 0;
}

/// Object operation: move cont.
#[allow(non_snake_case)]
pub fn H5O__move_cont(header: &mut ObjectHeaderState, from: usize, to: usize) -> Result<()> {
    if from >= header.messages.len() || to > header.messages.len() {
        return Err(Error::InvalidFormat(
            "object message move index out of range".into(),
        ));
    }
    if from == to {
        return Ok(());
    }
    let msg = header.messages.remove(from);
    let insert_at = if from < to { to.saturating_sub(1) } else { to };
    header
        .messages
        .insert(insert_at.min(header.messages.len()), msg);
    H5O__chunk_update_idx(header);
    Ok(())
}

/// Object operation: move msgs forward.
#[allow(non_snake_case)]
pub fn H5O__move_msgs_forward(header: &mut ObjectHeaderState) {
    let mut did_packing = false;
    loop {
        let mut null_idx = None;
        for (idx, message) in header.messages.iter().enumerate() {
            if message.msg_type == 0 && !message.data.is_empty() {
                null_idx = Some(idx);
                break;
            }
        }
        let Some(null_idx) = null_idx else {
            break;
        };
        let null_size = header.messages[null_idx].data.len();
        let mut msg_idx = None;
        for idx in null_idx + 1..header.messages.len() {
            let message = &header.messages[idx];
            if message.msg_type != 0 && !message.data.is_empty() && message.data.len() <= null_size
            {
                msg_idx = Some(idx);
                break;
            }
        }
        let Some(msg_idx) = msg_idx else {
            break;
        };

        let mut moved_msg = header.messages.remove(msg_idx);
        let moved_size = moved_msg.data.len();
        if moved_msg.shared {
            moved_msg.flags |= 0x02;
        } else {
            moved_msg.flags &= !0x02;
        }
        moved_msg.flags |= 0x01;
        if moved_size == null_size {
            let old_null = std::mem::replace(&mut header.messages[null_idx], moved_msg);
            let insert_at = msg_idx.min(header.messages.len());
            header.messages.insert(insert_at, old_null);
        } else {
            let remainder = null_size.saturating_sub(moved_size);
            header.messages[null_idx].data.resize(moved_size, 0);
            let old_null = std::mem::replace(&mut header.messages[null_idx], moved_msg);
            if remainder >= 5 {
                let mut split_null = H5O__alloc_null(remainder);
                split_null.flags = 0;
                split_null.shared = false;
                header.messages.insert(null_idx + 1, split_null);
                let adjusted_msg_idx = if msg_idx >= null_idx + 1 {
                    msg_idx + 1
                } else {
                    msg_idx
                };
                let insert_at = adjusted_msg_idx.min(header.messages.len());
                header.messages.insert(insert_at, old_null);
            } else {
                let insert_at = msg_idx.min(header.messages.len());
                let mut gap_null = old_null;
                gap_null
                    .data
                    .resize(moved_size.saturating_add(remainder), 0);
                header.messages.insert(insert_at, gap_null);
            }
        }
        did_packing = true;
        H5O__merge_null(header);
        H5O__remove_empty_chunks(header);
    }

    let mut tail_null = 0usize;
    let mut packed = Vec::with_capacity(header.messages.len());
    for message in header.messages.drain(..) {
        if message.msg_type == 0 {
            tail_null = tail_null.saturating_add(message.data.len());
        } else if !message.data.is_empty() {
            packed.push(message);
        }
    }
    if tail_null > 0 {
        packed.push(H5O__alloc_null(tail_null));
    }
    header.messages = packed;
    if did_packing {
        header.flush_disabled = false;
    }
    H5O__chunk_update_idx(header);
}

/// Object operation: merge null.
#[allow(non_snake_case)]
pub fn H5O__merge_null(header: &mut ObjectHeaderState) {
    let mut merged = Vec::with_capacity(header.messages.len());
    for message in header.messages.drain(..) {
        if message.msg_type == 0 {
            if message.data.is_empty() {
                continue;
            }
            if let Some(last) = merged
                .last_mut()
                .filter(|last: &&mut ObjectMessage| last.msg_type == 0)
            {
                last.data.extend_from_slice(&message.data);
            } else {
                merged.push(message);
            }
        } else {
            merged.push(message);
        }
    }
    header.messages = merged;
    H5O__chunk_update_idx(header);
}

/// Remove an entry from an object.
#[allow(non_snake_case)]
pub fn H5O__remove_empty_chunks(header: &mut ObjectHeaderState) {
    let mut retained = Vec::with_capacity(header.messages.len());
    let mut removed = false;
    for mut message in header.messages.drain(..) {
        if message.msg_type == 0 {
            if message.data.is_empty() {
                removed = true;
                continue;
            }
            message.flags = 0;
            message.shared = false;
            retained.push(message);
        } else if !message.data.is_empty() {
            retained.push(message);
        } else {
            removed = true;
        }
    }
    header.messages = retained;
    if removed {
        header.flush_disabled = false;
    }
    H5O__chunk_update_idx(header);
}

/// Object operation: condense header.
#[allow(non_snake_case)]
pub fn H5O__condense_header(header: &mut ObjectHeaderState) {
    loop {
        let before = (
            header.messages.len(),
            header
                .messages
                .iter()
                .map(|message| (message.msg_type, message.data.len()))
                .collect::<Vec<_>>(),
        );
        H5O__move_msgs_forward(header);
        H5O__merge_null(header);
        H5O__remove_empty_chunks(header);
        let after = (
            header.messages.len(),
            header
                .messages
                .iter()
                .map(|message| (message.msg_type, message.data.len()))
                .collect::<Vec<_>>(),
        );
        if before == after {
            break;
        }
    }
    header.messages.shrink_to_fit();
}

/// Allocate storage for an object.
#[allow(non_snake_case)]
pub fn H5O__alloc_shrink_chunk(header: &mut ObjectHeaderState) {
    let before = header.messages.len();
    H5O__move_msgs_forward(header);
    H5O__merge_null(header);
    header.messages.retain(|message| {
        if message.msg_type == 0 {
            !message.data.is_empty()
        } else {
            !message.data.is_empty()
        }
    });
    if header.messages.len() != before {
        H5O__chunk_update_idx(header);
    }
    if header
        .messages
        .last()
        .is_some_and(|message| message.msg_type == 0)
    {
        if header.messages.iter().any(|message| message.msg_type != 0) {
            header.flush_disabled = false;
        }
    }
    header.messages.shrink_to_fit();
    header.flush_disabled = false;
}

/// Decode an object from its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__mtime_new_decode(bytes: &[u8]) -> Result<u64> {
    if bytes.len() < 8 {
        return Err(Error::InvalidFormat(
            "new modification time message is truncated".into(),
        ));
    }
    let version = bytes[0];
    if version != 1 {
        return Err(Error::InvalidFormat(format!(
            "new modification time message version {version}"
        )));
    }
    Ok(u64::from(read_le_u32_at(
        bytes,
        4,
        "new modification time message timestamp",
    )?))
}

/// Decode an object from its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__mtime_decode(bytes: &[u8]) -> Result<u64> {
    if bytes.len() < 8 {
        return Err(Error::InvalidFormat(
            "modification time message is truncated".into(),
        ));
    }
    read_le_u64_at(bytes, 0, "modification time message")
}

/// Encode an object to its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__mtime_new_encode(timestamp: u64) -> Result<Vec<u8>> {
    let timestamp = u32::try_from(timestamp).map_err(|_| {
        Error::InvalidFormat("new modification time message timestamp exceeds u32".into())
    })?;
    let mut out = Vec::with_capacity(8);
    out.push(1);
    out.extend_from_slice(&[0; 3]);
    out.extend_from_slice(&timestamp.to_le_bytes());
    Ok(out)
}

/// Encode an object to its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__mtime_encode(timestamp: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(8);
    out.extend_from_slice(&timestamp.to_le_bytes());
    out
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__mtime_copy(timestamp: u64) -> u64 {
    let copied = timestamp;
    copied
}

/// Object operation: mtime new size.
#[allow(non_snake_case)]
pub fn H5O__mtime_new_size(_timestamp: u64) -> usize {
    8
}

/// Object operation: mtime size.
#[allow(non_snake_case)]
pub fn H5O__mtime_size(_timestamp: u64) -> usize {
    8
}

/// Free an object's in-memory resources.
#[allow(non_snake_case)]
pub fn H5O__mtime_free(mut timestamp: u64) {
    let original = timestamp;
    timestamp = 0;
    std::hint::black_box((original, timestamp));
}

/// Return a debug-friendly representation of an object.
#[allow(non_snake_case)]
pub fn H5O__mtime_debug(timestamp: u64) -> String {
    let state = if timestamp == 0 { "unset" } else { "set" };
    let days = timestamp / 86_400;
    let seconds_of_day = timestamp % 86_400;
    format!(
        "mtime(timestamp={timestamp}, state={state}, days={days}, seconds_of_day={seconds_of_day})"
    )
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__copy_header_real(header: &ObjectHeaderState) -> ObjectHeaderState {
    let mut messages = Vec::with_capacity(header.messages.len());
    let mut trailing_null: Option<ObjectMessage> = None;
    let mut saw_nonnull = false;
    for message in &header.messages {
        if matches!(message.msg_type, 0xffff | 0x001a) {
            continue;
        }
        if message.msg_type == 0 && message.data.is_empty() {
            continue;
        }
        let mut data = Vec::with_capacity(message.data.len());
        data.extend_from_slice(&message.data);
        let mut copied = ObjectMessage {
            msg_type: message.msg_type,
            flags: message.flags,
            creation_index: message.creation_index,
            data,
            shared: message.shared,
        };
        if copied.data.is_empty() {
            copied.flags &= !0x01;
        } else {
            copied.flags |= 0x01;
        }
        if copied.shared {
            copied.flags |= 0x02;
        } else {
            copied.flags &= !0x02;
        }
        if copied.msg_type == 0 {
            copied.flags = 0;
            copied.shared = false;
            for byte in &mut copied.data {
                *byte = 0;
            }
            if let Some(null) = trailing_null.as_mut() {
                null.data.extend_from_slice(&copied.data);
            } else {
                trailing_null = Some(copied);
            }
            continue;
        }
        if let Some(null) = trailing_null.take() {
            messages.push(null);
        }
        saw_nonnull = true;
        messages.push(copied);
    }
    if let Some(null) = trailing_null {
        if saw_nonnull {
            messages.push(null);
        }
    }
    let mut copied = ObjectHeaderState {
        addr: header.addr,
        messages,
        refcount: if header.addr == u64::MAX {
            0
        } else {
            header.refcount.max(u32::from(saw_nonnull))
        },
        comment: header
            .comment
            .as_ref()
            .map(|comment| comment.trim_end_matches('\0').to_string())
            .filter(|comment| !comment.is_empty()),
        flush_disabled: false,
    };
    H5O__chunk_update_idx(&mut copied);
    if copied.messages.is_empty() && copied.comment.is_none() {
        copied.refcount = 0;
    }
    if H5O__assert(&copied).is_err() {
        copied.flush_disabled = true;
    }
    copied
}

/// Free an object's in-memory resources.
#[allow(non_snake_case)]
pub fn H5O__copy_free_addrmap_cb(_addr: u64) {
    let mut addr = _addr;
    if addr == u64::MAX {
        addr = 0;
    }
    std::hint::black_box(addr);
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__copy_header(header: &ObjectHeaderState) -> ObjectHeaderState {
    let mut messages = Vec::with_capacity(header.messages.len());
    for message in &header.messages {
        messages.push(ObjectMessage {
            msg_type: message.msg_type,
            flags: H5O_msg_get_flags(message),
            creation_index: message.creation_index,
            data: message.data.to_vec(),
            shared: message.shared,
        });
    }
    let mut copied = ObjectHeaderState {
        addr: header.addr,
        messages,
        refcount: header.refcount,
        comment: header.comment.as_ref().map(|comment| comment.to_string()),
        flush_disabled: false,
    };
    H5O__chunk_update_idx(&mut copied);
    copied
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__copy(header: &ObjectHeaderState) -> ObjectHeaderState {
    let mut copied = H5O__copy_header(header);
    copied.refcount = copied.refcount.max(1);
    if copied.messages.is_empty() && copied.addr == u64::MAX {
        copied.refcount = 0;
        copied.flush_disabled = true;
    }
    copied
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O_copy_header_map(header: &ObjectHeaderState) -> BTreeMap<u16, usize> {
    let mut counts = BTreeMap::new();
    for message in &header.messages {
        if message.msg_type == 0x001a {
            continue;
        }
        let entry = counts.entry(message.msg_type).or_insert(0usize);
        *entry = entry.saturating_add(1);
    }
    counts
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__copy_obj(header: &ObjectHeaderState) -> ObjectHeaderState {
    let mut copied = ObjectHeaderState {
        addr: header.addr,
        messages: Vec::with_capacity(header.messages.len()),
        refcount: header.refcount.max(1),
        comment: header.comment.as_ref().map(|comment| comment.to_string()),
        flush_disabled: false,
    };
    for message in &header.messages {
        copied.messages.push(H5O__copy_mesg(message));
    }
    if header.addr == u64::MAX && copied.messages.is_empty() {
        copied.refcount = 0;
        copied.flush_disabled = true;
    }
    H5O__chunk_update_idx(&mut copied);
    copied
}

/// Free an object's in-memory resources.
#[allow(non_snake_case)]
pub fn H5O__copy_free_comm_dt_cb(_addr: u64) {
    let mut addr = _addr;
    if addr == u64::MAX {
        addr = 0;
    }
    std::hint::black_box(addr);
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__copy_comm_dt_cmp(left: u64, right: u64) -> std::cmp::Ordering {
    left.cmp(&right)
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__copy_search_comm_dt_attr_cb(message: &ObjectMessage) -> bool {
    if message.msg_type != 0x000c || message.data.is_empty() {
        return false;
    }
    let Some(name_end) = message.data.iter().position(|byte| *byte == 0) else {
        return false;
    };
    let payload = match name_end.checked_add(1) {
        Some(start) => &message.data[start..],
        None => return false,
    };
    for window in payload.windows(2) {
        if window == [0x03, 0x00] {
            return true;
        }
    }
    false
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__copy_search_comm_dt_check(header: &ObjectHeaderState) -> bool {
    if header.addr == u64::MAX {
        return false;
    }
    let mut saw_named_type = false;
    let mut saw_shared_reference = false;
    let mut saw_attribute_type = false;
    for message in &header.messages {
        if message.msg_type == 0x0003 {
            if message.data.is_empty() {
                continue;
            }
            if DatatypeMessage::decode(&message.data).is_ok() {
                saw_named_type = true;
            }
            if message.shared || message.flags & 0x02 != 0 {
                saw_shared_reference = true;
            }
            if saw_named_type || saw_shared_reference {
                return true;
            }
        }
        if H5O__copy_search_comm_dt_attr_cb(message) {
            saw_attribute_type = true;
        }
        if message.msg_type == 0x000c && !message.data.is_empty() {
            let payload_start = message
                .data
                .iter()
                .position(|byte| *byte == 0)
                .and_then(|idx| idx.checked_add(1))
                .unwrap_or(0);
            let payload = &message.data[payload_start..];
            for offset in 0..payload.len() {
                if DatatypeMessage::decode(&payload[offset..]).is_ok() {
                    saw_attribute_type = true;
                    break;
                }
                if payload.len().saturating_sub(offset) >= 2
                    && payload[offset] == 0x03
                    && payload[offset + 1] == 0x00
                {
                    saw_attribute_type = true;
                    break;
                }
            }
        }
    }
    saw_named_type || saw_shared_reference || saw_attribute_type
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__copy_search_comm_dt_cb(header: &ObjectHeaderState) -> Option<u64> {
    if H5O__copy_search_comm_dt_check(header) {
        Some(header.addr)
    } else {
        None
    }
}

/// Insert an entry into an object.
#[allow(non_snake_case)]
pub fn H5O__copy_insert_comm_dt(
    committed: &mut BTreeMap<u64, u64>,
    src_addr: u64,
    dst_addr: u64,
) -> Result<()> {
    if src_addr == u64::MAX || dst_addr == u64::MAX {
        return Err(Error::InvalidFormat(
            "committed datatype copy address is undefined".into(),
        ));
    }
    if src_addr == 0 || dst_addr == 0 {
        return Err(Error::InvalidFormat(
            "committed datatype copy address is zero".into(),
        ));
    }
    match committed.get(&src_addr).copied() {
        Some(existing) if existing != dst_addr => Err(Error::InvalidFormat(
            "committed datatype copy map contains conflicting destination".into(),
        )),
        Some(_) => Ok(()),
        None => {
            if committed.values().any(|addr| *addr == dst_addr) {
                return Err(Error::InvalidFormat(
                    "committed datatype copy map reuses destination address".into(),
                ));
            }
            committed.insert(src_addr, dst_addr);
            Ok(())
        }
    }
}

/// Flush the object to storage.
#[allow(non_snake_case)]
pub fn H5O_flush(header: &mut ObjectHeaderState) {
    for message in &mut header.messages {
        if message.data.is_empty() {
            message.flags &= !0x01;
        } else {
            message.flags |= 0x01;
        }
        if message.shared {
            message.flags |= 0x02;
        } else {
            message.flags &= !0x02;
        }
    }
    H5O__condense_header(header);
    header.flush_disabled = false;
}

/// Flush the object to storage.
#[allow(non_snake_case)]
pub fn H5O_flush_common(header: &mut ObjectHeaderState) {
    if header.addr == u64::MAX && header.messages.is_empty() {
        header.flush_disabled = false;
        return;
    }
    H5O_flush(header);
    if header.refcount == 0 && !header.messages.is_empty() {
        header.refcount = 1;
    }
}

/// Object operation: protect.
#[allow(non_snake_case)]
pub fn H5O_protect(header: &ObjectHeaderState) -> ObjectHeaderState {
    header.clone()
}

/// Object operation: oh tag.
#[allow(non_snake_case)]
pub fn H5O__oh_tag(header: &ObjectHeaderState) -> u64 {
    if header.addr == u64::MAX {
        0
    } else {
        header.addr
    }
}

/// Refresh the object from storage.
#[allow(non_snake_case)]
pub fn H5O_refresh_metadata(header: &mut ObjectHeaderState) {
    H5O__chunk_update_idx(header);
    H5O__remove_empty_chunks(header);
    header.flush_disabled = false;
}

/// Refresh the object from storage.
#[allow(non_snake_case)]
pub fn H5O__refresh_metadata_close(header: &mut ObjectHeaderState) {
    for message in &mut header.messages {
        if message.data.is_empty() {
            message.flags &= !0x01;
        }
        if !message.shared {
            message.flags &= !0x02;
        }
    }
    if header.refcount == 0 {
        header.flush_disabled = false;
    }
    H5O__chunk_update_idx(header);
}

/// Refresh the object from storage.
#[allow(non_snake_case)]
pub fn H5O_refresh_metadata_reopen(header: &ObjectHeaderState) -> ObjectHeaderState {
    let mut reopened = ObjectHeaderState {
        addr: header.addr,
        messages: Vec::with_capacity(header.messages.len()),
        refcount: header.refcount,
        comment: header.comment.as_ref().map(|comment| comment.to_string()),
        flush_disabled: false,
    };
    for message in &header.messages {
        if message.msg_type == 0x001a {
            continue;
        }
        reopened.messages.push(ObjectMessage {
            msg_type: message.msg_type,
            flags: H5O_msg_get_flags(message),
            creation_index: message.creation_index,
            data: message.data.to_vec(),
            shared: message.shared,
        });
    }
    H5O__chunk_update_idx(&mut reopened);
    H5O__remove_empty_chunks(&mut reopened);
    reopened
}

/// Decode an object from its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__shmesg_decode(bytes: &[u8]) -> Result<SharedMessageTableInfo> {
    if bytes.len() < 10 {
        return Err(Error::InvalidFormat(
            "shared message table payload is truncated".into(),
        ));
    }
    let version = bytes[0];
    if version != 0 {
        return Err(Error::InvalidFormat(format!(
            "unsupported shared message table version: {version}"
        )));
    }
    let table_addr = read_le_uint_width(bytes, 1, 8, "shared message table address")?;
    if table_addr == u64::MAX {
        return Err(Error::InvalidFormat(
            "shared message table address is undefined".into(),
        ));
    }
    let nindexes = bytes[9];
    if nindexes == 0 || nindexes > 8 {
        return Err(Error::InvalidFormat(
            "shared message table index count is invalid".into(),
        ));
    }
    Ok(SharedMessageTableInfo {
        version,
        table_addr,
        nindexes,
    })
}

/// Decode an object from its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__shmesg_decode_with_addr_size(
    bytes: &[u8],
    sizeof_addr: u8,
) -> Result<SharedMessageTableInfo> {
    if bytes.len() < 2 {
        return Err(Error::InvalidFormat(
            "shared message table payload is truncated".into(),
        ));
    }
    let addr_start = 1usize;
    let addr_end = checked_add(
        addr_start,
        usize::from(sizeof_addr),
        "shared message table address",
    )?;
    let nindexes_offset = addr_end;
    if bytes.len() <= nindexes_offset {
        return Err(Error::InvalidFormat(
            "shared message table payload is truncated".into(),
        ));
    }
    let version = bytes[0];
    if version != 0 {
        return Err(Error::InvalidFormat(format!(
            "unsupported shared message table version: {version}"
        )));
    }
    let table_addr = read_le_uint_width(
        bytes,
        addr_start,
        usize::from(sizeof_addr),
        "shared message table address",
    )?;
    if is_undefined_addr_width(table_addr, sizeof_addr)? {
        return Err(Error::InvalidFormat(
            "shared message table address is undefined".into(),
        ));
    }
    let nindexes = bytes[nindexes_offset];
    if nindexes == 0 || nindexes > 8 {
        return Err(Error::InvalidFormat(
            "shared message table index count is invalid".into(),
        ));
    }
    Ok(SharedMessageTableInfo {
        version,
        table_addr,
        nindexes,
    })
}

/// Encode an object to its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__shmesg_encode(table: &SharedMessageTableInfo) -> Result<Vec<u8>> {
    if table.version != 0 {
        return Err(Error::InvalidFormat(format!(
            "unsupported shared message table version: {}",
            table.version
        )));
    }
    if table.table_addr == u64::MAX {
        return Err(Error::InvalidFormat(
            "shared message table address is undefined".into(),
        ));
    }
    if table.nindexes == 0 || table.nindexes > 8 {
        return Err(Error::InvalidFormat(
            "shared message table index count is invalid".into(),
        ));
    }
    let mut out = Vec::with_capacity(10);
    out.push(table.version);
    out.extend_from_slice(&table.table_addr.to_le_bytes());
    out.push(table.nindexes);
    Ok(out)
}

/// Encode an object to its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__shmesg_encode_with_addr_size(
    table: &SharedMessageTableInfo,
    sizeof_addr: u8,
) -> Result<Vec<u8>> {
    if table.version != 0 {
        return Err(Error::InvalidFormat(format!(
            "unsupported shared message table version: {}",
            table.version
        )));
    }
    if is_undefined_addr_width(table.table_addr, sizeof_addr)? {
        return Err(Error::InvalidFormat(
            "shared message table address is undefined".into(),
        ));
    }
    if table.nindexes == 0 || table.nindexes > 8 {
        return Err(Error::InvalidFormat(
            "shared message table index count is invalid".into(),
        ));
    }
    let addr_width = usize::from(sizeof_addr);
    let len = 2usize
        .checked_add(addr_width)
        .ok_or_else(|| Error::InvalidFormat("shared message table size overflow".into()))?;
    let mut out = Vec::with_capacity(len);
    out.push(table.version);
    encode_le_uint_width(
        &mut out,
        table.table_addr,
        addr_width,
        "shared message table address",
    )?;
    out.push(table.nindexes);
    Ok(out)
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__shmesg_copy(table: &SharedMessageTableInfo) -> SharedMessageTableInfo {
    SharedMessageTableInfo {
        version: table.version,
        table_addr: table.table_addr,
        nindexes: table.nindexes,
    }
}

/// Object operation: shmesg size.
#[allow(non_snake_case)]
pub fn H5O__shmesg_size(_table: &SharedMessageTableInfo) -> usize {
    10
}

/// Object operation: shmesg size with addr size.
#[allow(non_snake_case)]
pub fn H5O__shmesg_size_with_addr_size(
    table: &SharedMessageTableInfo,
    sizeof_addr: u8,
) -> Result<usize> {
    Ok(H5O__shmesg_encode_with_addr_size(table, sizeof_addr)?.len())
}

/// Return a debug-friendly representation of an object.
#[allow(non_snake_case)]
pub fn H5O__shmesg_debug(table: &SharedMessageTableInfo) -> String {
    format!(
        "shmesg(version={}, table_addr={:#x}, indexes={})",
        table.version, table.table_addr, table.nindexes
    )
}

/// Decode an object from its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__pline_decode(bytes: &[u8]) -> Result<FilterPipelineMessage> {
    let mut pos = 0usize;
    let version = read_u8_cursor(bytes, &mut pos, "filter pipeline version")?;
    let nfilters = usize::from(read_u8_cursor(
        bytes,
        &mut pos,
        "filter pipeline filter count",
    )?);
    if nfilters > 32 {
        return Err(Error::InvalidFormat(format!(
            "filter pipeline has too many filters: {nfilters}"
        )));
    }

    let mut filters = Vec::with_capacity(nfilters);
    match version {
        1 => {
            let reserved_end = checked_add(pos, 6, "filter pipeline v1 reserved bytes")?;
            if bytes.get(pos..reserved_end).is_none() {
                return Err(Error::InvalidFormat(
                    "filter pipeline v1 header is truncated".into(),
                ));
            }
            pos = reserved_end;

            for _ in 0..nfilters {
                let id =
                    read_le_uint_cursor(bytes, &mut pos, 2, "filter pipeline v1 filter id")? as u16;
                let name_len =
                    read_le_uint_cursor(bytes, &mut pos, 2, "filter pipeline v1 name length")?
                        as usize;
                if name_len % 8 != 0 {
                    return Err(Error::InvalidFormat(format!(
                        "filter pipeline v1 name length {name_len} is not a multiple of eight"
                    )));
                }
                let flags =
                    read_le_uint_cursor(bytes, &mut pos, 2, "filter pipeline v1 flags")? as u16;
                let cd_nelmts = read_le_uint_cursor(
                    bytes,
                    &mut pos,
                    2,
                    "filter pipeline v1 client data count",
                )? as usize;
                if cd_nelmts > 1024 {
                    return Err(Error::InvalidFormat(format!(
                        "filter pipeline v1 client data count {cd_nelmts} exceeds supported maximum 1024"
                    )));
                }

                let name = if name_len == 0 {
                    None
                } else {
                    let name_end = checked_add(pos, name_len, "filter pipeline v1 name")?;
                    let name_bytes = bytes.get(pos..name_end).ok_or_else(|| {
                        Error::InvalidFormat("filter pipeline v1 name is truncated".into())
                    })?;
                    let null_pos =
                        name_bytes
                            .iter()
                            .position(|&byte| byte == 0)
                            .ok_or_else(|| {
                                Error::InvalidFormat(
                                    "filter pipeline v1 name is not null-terminated".into(),
                                )
                            })?;
                    let text = std::str::from_utf8(&name_bytes[..null_pos])
                        .map_err(|_| {
                            Error::InvalidFormat("filter pipeline v1 name text is not UTF-8".into())
                        })?
                        .to_string();
                    let padded = align8_len_checked(name_len, "filter pipeline v1 name")?;
                    let padded_end = checked_add(pos, padded, "filter pipeline v1 padded name")?;
                    if bytes.get(pos..padded_end).is_none() {
                        return Err(Error::InvalidFormat(
                            "filter pipeline v1 padded name is truncated".into(),
                        ));
                    }
                    pos = padded_end;
                    Some(text)
                };

                let mut client_data = Vec::with_capacity(cd_nelmts);
                for _ in 0..cd_nelmts {
                    client_data.push(read_le_uint_cursor(
                        bytes,
                        &mut pos,
                        4,
                        "filter pipeline v1 client data",
                    )? as u32);
                }
                if cd_nelmts % 2 != 0 {
                    let padding_end =
                        checked_add(pos, 4, "filter pipeline v1 client data padding")?;
                    if bytes.get(pos..padding_end).is_none() {
                        return Err(Error::InvalidFormat(
                            "filter pipeline v1 client data padding is truncated".into(),
                        ));
                    }
                    pos = padding_end;
                }

                filters.push(FilterDesc {
                    id,
                    name,
                    flags,
                    client_data,
                });
            }
        }
        2 => {
            for _ in 0..nfilters {
                let id =
                    read_le_uint_cursor(bytes, &mut pos, 2, "filter pipeline v2 filter id")? as u16;
                let name = if id >= 256 {
                    let name_len =
                        read_le_uint_cursor(bytes, &mut pos, 2, "filter pipeline v2 name length")?
                            as usize;
                    if name_len == 0 {
                        None
                    } else {
                        let name_end = checked_add(pos, name_len, "filter pipeline v2 name")?;
                        let name_bytes = bytes.get(pos..name_end).ok_or_else(|| {
                            Error::InvalidFormat("filter pipeline v2 name is truncated".into())
                        })?;
                        let null_pos =
                            name_bytes
                                .iter()
                                .position(|&byte| byte == 0)
                                .ok_or_else(|| {
                                    Error::InvalidFormat(
                                        "filter pipeline v2 name is not null-terminated".into(),
                                    )
                                })?;
                        let text = std::str::from_utf8(&name_bytes[..null_pos])
                            .map_err(|_| {
                                Error::InvalidFormat(
                                    "filter pipeline v2 name text is not UTF-8".into(),
                                )
                            })?
                            .to_string();
                        pos = name_end;
                        Some(text)
                    }
                } else {
                    None
                };
                let flags =
                    read_le_uint_cursor(bytes, &mut pos, 2, "filter pipeline v2 flags")? as u16;
                let cd_nelmts = read_le_uint_cursor(
                    bytes,
                    &mut pos,
                    2,
                    "filter pipeline v2 client data count",
                )? as usize;
                if cd_nelmts > 1024 {
                    return Err(Error::InvalidFormat(format!(
                        "filter pipeline v2 client data count {cd_nelmts} exceeds supported maximum 1024"
                    )));
                }
                let mut client_data = Vec::with_capacity(cd_nelmts);
                for _ in 0..cd_nelmts {
                    client_data.push(read_le_uint_cursor(
                        bytes,
                        &mut pos,
                        4,
                        "filter pipeline v2 client data",
                    )? as u32);
                }

                filters.push(FilterDesc {
                    id,
                    name,
                    flags,
                    client_data,
                });
            }
        }
        _ => {
            return Err(Error::InvalidFormat(format!(
                "filter pipeline version {version}"
            )));
        }
    }

    Ok(FilterPipelineMessage { version, filters })
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__pline_copy(message: &FilterPipelineMessage) -> FilterPipelineMessage {
    FilterPipelineMessage {
        version: message.version,
        filters: message
            .filters
            .iter()
            .map(|filter| FilterDesc {
                id: filter.id,
                name: filter.name.clone(),
                flags: filter.flags,
                client_data: filter.client_data.clone(),
            })
            .collect(),
    }
}

/// Object operation: pline size.
#[allow(non_snake_case)]
pub fn H5O__pline_size(message: &FilterPipelineMessage) -> usize {
    match H5O__pline_size_checked(message) {
        Ok(size) if size > 0 => size,
        Ok(_) => 0,
        Err(_) => usize::MAX,
    }
}

/// Object operation: pline size checked.
#[allow(non_snake_case)]
pub fn H5O__pline_size_checked(message: &FilterPipelineMessage) -> Result<usize> {
    match message.version {
        1 => {
            let filters_size = message.filters.iter().try_fold(0usize, |acc, filter| {
                let name_len = filter
                    .name
                    .as_ref()
                    .map(|name| {
                        checked_add(name.len(), 1, "filter pipeline v1 name length")
                            .and_then(|len| align8_len_checked(len, "filter pipeline v1 name"))
                    })
                    .transpose()?
                    .unwrap_or(0);
                let client_data_len = filter.client_data.len().checked_mul(4).ok_or_else(|| {
                    Error::InvalidFormat(
                        "filter pipeline v1 client data byte length overflow".into(),
                    )
                })?;
                let client_data_padding = if filter.client_data.len() % 2 != 0 {
                    4
                } else {
                    0
                };
                let filter_size = checked_usize_sum(
                    &[8, name_len, client_data_len, client_data_padding],
                    "filter pipeline v1 filter size",
                )?;
                acc.checked_add(filter_size)
                    .ok_or_else(|| Error::InvalidFormat("filter pipeline v1 size overflow".into()))
            })?;
            checked_add(8, filters_size, "filter pipeline v1 size")
        }
        2 => {
            let filters_size = message.filters.iter().try_fold(0usize, |acc, filter| {
                let name_len = if filter.id >= 256 {
                    let name_len = filter
                        .name
                        .as_ref()
                        .map(|name| checked_add(name.len(), 1, "filter pipeline v2 name length"))
                        .transpose()?
                        .unwrap_or(0);
                    checked_add(2, name_len, "filter pipeline v2 encoded name length")?
                } else {
                    0
                };
                let client_data_len = filter.client_data.len().checked_mul(4).ok_or_else(|| {
                    Error::InvalidFormat(
                        "filter pipeline v2 client data byte length overflow".into(),
                    )
                })?;
                let filter_size = checked_usize_sum(
                    &[2, name_len, 2, 2, client_data_len],
                    "filter pipeline v2 filter size",
                )?;
                acc.checked_add(filter_size)
                    .ok_or_else(|| Error::InvalidFormat("filter pipeline v2 size overflow".into()))
            })?;
            checked_add(2, filters_size, "filter pipeline v2 size")
        }
        _ => Err(Error::InvalidFormat(format!(
            "filter pipeline version {}",
            message.version
        ))),
    }
}

/// Reset an object to its default state.
#[allow(non_snake_case)]
pub fn H5O__pline_reset(message: &mut FilterPipelineMessage) {
    message.version = match message.version {
        1 | 2 => message.version,
        _ => 2,
    };
    message.filters.clear();
}

/// Free an object's in-memory resources.
#[allow(non_snake_case)]
pub fn H5O__pline_free(mut message: FilterPipelineMessage) {
    for filter in &mut message.filters {
        if let Some(name) = filter.name.as_mut() {
            name.clear();
        }
        filter.client_data.clear();
    }
    message.filters.clear();
    message.version = 0;
    drop(message);
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__pline_pre_copy_file(message: &FilterPipelineMessage) -> FilterPipelineMessage {
    let mut filters = Vec::with_capacity(message.filters.len());
    for filter in &message.filters {
        filters.push(FilterDesc {
            id: filter.id,
            name: filter.name.as_ref().map(|name| name.to_string()),
            flags: filter.flags,
            client_data: filter.client_data.to_vec(),
        });
    }
    FilterPipelineMessage {
        version: message.version,
        filters,
    }
}

/// Return a debug-friendly representation of an object.
#[allow(non_snake_case)]
pub fn H5O__pline_debug(message: &FilterPipelineMessage) -> String {
    let filter_ids = message
        .filters
        .iter()
        .map(|filter| filter.id.to_string())
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "pline(version={}, filters={}, ids=[{}])",
        message.version,
        message.filters.len(),
        filter_ids
    )
}

/// Object operation: pline set version.
#[allow(non_snake_case)]
pub fn H5O_pline_set_version(bytes: &mut Vec<u8>, version: u8) {
    if bytes.is_empty() {
        bytes.push(version);
    } else {
        bytes[0] = version;
    }
}

/// Decode an object from its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__drvinfo_decode(bytes: &[u8]) -> Result<DriverInfoMessage> {
    let mut pos = 0usize;
    let version = read_u8_cursor(bytes, &mut pos, "driver info version")?;
    if version != 0 {
        return Err(Error::InvalidFormat(format!(
            "driver info message version {version}"
        )));
    }
    let name_end = checked_add(pos, 8, "driver info name")?;
    let name_bytes = bytes
        .get(pos..name_end)
        .ok_or_else(|| Error::InvalidFormat("driver info name is truncated".into()))?;
    let mut name = [0u8; 8];
    name.copy_from_slice(name_bytes);
    pos = name_end;
    let len = usize::try_from(read_le_uint_cursor(
        bytes,
        &mut pos,
        2,
        "driver info length",
    )?)
    .map_err(|_| Error::InvalidFormat("driver info length exceeds usize".into()))?;
    if len == 0 {
        return Err(Error::InvalidFormat(
            "driver info message length is zero".into(),
        ));
    }
    let data_end = checked_add(pos, len, "driver info payload")?;
    let data = bytes
        .get(pos..data_end)
        .ok_or_else(|| Error::InvalidFormat("driver info payload is truncated".into()))?
        .to_vec();
    Ok(DriverInfoMessage {
        version,
        name,
        data,
    })
}

/// Encode an object to its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__drvinfo_encode(message: &DriverInfoMessage) -> Result<Vec<u8>> {
    if message.version != 0 {
        return Err(Error::InvalidFormat(format!(
            "driver info message version {}",
            message.version
        )));
    }
    let len = u16::try_from(message.data.len())
        .map_err(|_| Error::InvalidFormat("driver info payload exceeds u16".into()))?;
    if len == 0 {
        return Err(Error::InvalidFormat(
            "driver info message length is zero".into(),
        ));
    }
    let mut out = Vec::with_capacity(H5O__drvinfo_size(message)?);
    out.push(message.version);
    out.extend_from_slice(&message.name);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&message.data);
    Ok(out)
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__drvinfo_copy(message: &DriverInfoMessage) -> DriverInfoMessage {
    let mut name = [0u8; 8];
    name.copy_from_slice(&message.name);
    let mut data = Vec::with_capacity(message.data.len());
    data.extend_from_slice(&message.data);
    DriverInfoMessage {
        version: message.version,
        name,
        data,
    }
}

/// Object operation: drvinfo size.
#[allow(non_snake_case)]
pub fn H5O__drvinfo_size(message: &DriverInfoMessage) -> Result<usize> {
    message
        .data
        .len()
        .checked_add(11)
        .ok_or_else(|| Error::InvalidFormat("driver info image length overflow".into()))
}

/// Reset an object to its default state.
#[allow(non_snake_case)]
pub fn H5O__drvinfo_reset(message: &mut DriverInfoMessage) {
    message.data.clear();
}

/// Return a debug-friendly representation of an object.
#[allow(non_snake_case)]
pub fn H5O__drvinfo_debug(message: &DriverInfoMessage) -> String {
    let nul = message
        .name
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(message.name.len());
    let name = String::from_utf8_lossy(&message.name[..nul]);
    format!("drvinfo(name={}, size={})", name, message.data.len())
}

/// Initialize the object subsystem.
#[allow(non_snake_case)]
pub fn H5O_init() -> bool {
    H5O__init_package()
}

/// Initialize the object package.
#[allow(non_snake_case)]
pub fn H5O__init_package() -> bool {
    true
}

/// Object operation: set version.
#[allow(non_snake_case)]
pub fn H5O__set_version(layout: &mut LayoutObjectMessage, version: u8) {
    layout.message.version = version;
    if let Some(raw_version) = layout.raw.first_mut() {
        *raw_version = version;
    }
}

/// Create a new object.
#[allow(non_snake_case)]
pub fn H5O_create_ohdr(addr: u64) -> ObjectHeaderState {
    let mut header = ObjectHeaderState {
        addr,
        refcount: 1,
        ..ObjectHeaderState::default()
    };
    if addr == u64::MAX {
        header.refcount = 0;
        header.flush_disabled = true;
    }
    header
}

/// Object operation: apply ohdr.
#[allow(non_snake_case)]
pub fn H5O_apply_ohdr(header: &mut ObjectHeaderState, f: impl FnOnce(&mut ObjectHeaderState)) {
    let before_count = header.messages.len();
    let before_refcount = header.refcount;
    let before_addr = header.addr;
    let before_comment = header.comment.clone();
    f(header);
    let mut changed = header.messages.len() != before_count
        || header.refcount != before_refcount
        || header.addr != before_addr
        || header.comment != before_comment;
    for message in &mut header.messages {
        if message.data.is_empty() {
            message.flags &= !0x01;
        } else {
            message.flags |= 0x01;
        }
        if message.shared {
            message.flags |= 0x02;
        } else {
            message.flags &= !0x02;
        }
        if message.msg_type == 0 {
            message.flags = 0;
            message.shared = false;
            for byte in &mut message.data {
                *byte = 0;
            }
        }
        if message.flags & 0x20 != 0 && message.flags & 0x10 == 0 {
            message.flags |= 0x10;
            changed = true;
        }
    }
    if header.refcount == 0 && !header.messages.is_empty() {
        header.refcount = 1;
        changed = true;
    }
    if header.addr == u64::MAX && !header.messages.is_empty() {
        header.addr = before_addr;
        if header.addr == u64::MAX {
            header.refcount = 0;
            header.messages.clear();
            header.comment = None;
        }
        changed = true;
    }
    if let Some(comment) = header.comment.as_mut() {
        while comment.ends_with('\0') {
            comment.pop();
            changed = true;
        }
        if comment.is_empty() {
            header.comment = None;
            changed = true;
        }
    }
    H5O__condense_header(header);
    H5O__chunk_update_idx(header);
    if changed || before_refcount != header.refcount {
        header.flush_disabled = false;
    }
    let _ = H5O__assert(header);
    header.flush_disabled = false;
}

/// Open an object.
#[allow(non_snake_case)]
pub fn H5O_open(header: &ObjectHeaderState) -> ObjectHeaderState {
    let mut opened = H5O__copy_header_real(header);
    opened.refcount = opened.refcount.saturating_add(1);
    opened
}

/// Open an object by name.
#[allow(non_snake_case)]
pub fn H5O_open_name(
    objects: &BTreeMap<String, ObjectHeaderState>,
    name: &str,
) -> Option<ObjectHeaderState> {
    objects.get(name).cloned()
}

/// Open the object at the given index.
#[allow(non_snake_case)]
pub fn H5O__open_by_idx(
    objects: &BTreeMap<String, ObjectHeaderState>,
    idx: usize,
) -> Option<ObjectHeaderState> {
    objects.values().nth(idx).cloned()
}

/// Open an object by its on-disk address.
#[allow(non_snake_case)]
pub fn H5O__open_by_addr(
    objects: &BTreeMap<String, ObjectHeaderState>,
    addr: u64,
) -> Option<ObjectHeaderState> {
    objects.values().find(|header| header.addr == addr).cloned()
}

/// Open an object.
#[allow(non_snake_case)]
pub fn H5O_open_by_loc(
    location: &ObjectLocation,
    objects: &BTreeMap<String, ObjectHeaderState>,
) -> Option<ObjectHeaderState> {
    H5O__open_by_addr(objects, location.addr)
}

/// Close an object.
#[allow(non_snake_case)]
pub fn H5O_close(mut header: ObjectHeaderState) {
    H5O__flush_msgs(&mut header);
    H5O__delete_oh(&mut header);
}

/// Link an object.
#[allow(non_snake_case)]
pub fn H5O__link_oh(header: &mut ObjectHeaderState, delta: i32) {
    if delta.is_negative() {
        header.refcount = header.refcount.saturating_sub(delta.unsigned_abs());
    } else if let Ok(increase) = u32::try_from(delta) {
        header.refcount = header.refcount.saturating_add(increase);
    }
    if header.refcount == 0 {
        header.messages.retain(|message| message.shared);
        header.flush_disabled = false;
    } else {
        header.flush_disabled = false;
    }
}

/// Link an object.
#[allow(non_snake_case)]
pub fn H5O_link(header: &mut ObjectHeaderState, delta: i32) {
    if delta.is_negative() {
        header.refcount = header.refcount.saturating_sub(delta.unsigned_abs());
    } else if let Ok(increase) = u32::try_from(delta) {
        header.refcount = header.refcount.saturating_add(increase);
    }
    if header.refcount == 0 {
        header.flush_disabled = false;
    }
}

/// Object operation: pin.
#[allow(non_snake_case)]
pub fn H5O_pin(location: &mut ObjectLocation) {
    if location.addr == u64::MAX {
        location.held = false;
        return;
    }
    if location.file_name.as_deref() == Some("") {
        location.file_name = None;
    }
    location.held = true;
}

/// Object operation: unpin.
#[allow(non_snake_case)]
pub fn H5O_unpin(location: &mut ObjectLocation) {
    location.held = false;
}

/// Object operation: unprotect.
#[allow(non_snake_case)]
pub fn H5O_unprotect(header: &mut ObjectHeaderState) {
    for message in &mut header.messages {
        if message.data.is_empty() {
            message.flags &= !0x01;
        } else {
            message.flags |= 0x01;
        }
        if message.shared {
            message.flags |= 0x02;
        }
    }
    for (idx, msg) in header.messages.iter_mut().enumerate() {
        msg.creation_index = u16::try_from(idx).unwrap_or(u16::MAX);
    }
    if header.refcount == 0 {
        header.flush_disabled = false;
    }
}

/// Object operation: touch oh.
#[allow(non_snake_case)]
pub fn H5O_touch_oh(header: &mut ObjectHeaderState) {
    header.flush_disabled = false;
    for message in &mut header.messages {
        if message.data.is_empty() {
            message.flags &= !0x01;
        } else {
            message.flags |= 0x01;
        }
    }
    if header.refcount == 0 && !header.messages.is_empty() {
        header.refcount = 1;
    }
}

/// Object operation: touch.
#[allow(non_snake_case)]
pub fn H5O_touch(header: &mut ObjectHeaderState) {
    header.flush_disabled = false;
    for message in &mut header.messages {
        if message.data.is_empty() {
            message.flags &= !0x01;
        } else {
            message.flags |= 0x01;
        }
    }
    if header.refcount == 0 && !header.messages.is_empty() {
        header.refcount = 1;
    }
}

/// Object operation: bogus oh.
#[allow(non_snake_case)]
pub fn H5O_bogus_oh(header: &ObjectHeaderState) -> bool {
    if header.addr == u64::MAX {
        return true;
    }
    if header.refcount == 0 && header.messages.is_empty() {
        return true;
    }
    header
        .messages
        .iter()
        .any(|message| message.msg_type == 0x0009 || message.msg_type == 0x001b)
}

/// Delete an object.
#[allow(non_snake_case)]
pub fn H5O_delete(header: &mut ObjectHeaderState) {
    for message in &mut header.messages {
        message.data.clear();
        message.flags = 0;
        message.shared = false;
    }
    header.messages.clear();
    header.refcount = 0;
    header.comment = None;
    header.flush_disabled = false;
}

/// Delete an object.
#[allow(non_snake_case)]
pub fn H5O__delete_oh(header: &mut ObjectHeaderState) {
    header.messages.clear();
    header.refcount = 0;
    header.comment = None;
    header.flush_disabled = false;
}

/// Object operation: obj type.
#[allow(non_snake_case)]
pub fn H5O_obj_type(header: &ObjectHeaderState) -> &'static str {
    H5O__obj_type_real(header)
}

/// Object operation: obj type real.
#[allow(non_snake_case)]
pub fn H5O__obj_type_real(header: &ObjectHeaderState) -> &'static str {
    if H5O__group_isa(header) {
        "group"
    } else if header.messages.iter().any(|msg| msg.msg_type == 0x0001) {
        "dataset"
    } else {
        "unknown"
    }
}

/// Object operation: obj class.
#[allow(non_snake_case)]
pub fn H5O__obj_class(header: &ObjectHeaderState) -> &'static str {
    H5O_obj_type(header)
}

/// Object operation: get loc.
#[allow(non_snake_case)]
pub fn H5O_get_loc(header: &ObjectHeaderState) -> ObjectLocation {
    ObjectLocation {
        addr: header.addr,
        ..ObjectLocation::default()
    }
}

/// Reset an object to its default state.
#[allow(non_snake_case)]
pub fn H5O_loc_reset(location: &mut ObjectLocation) {
    *location = ObjectLocation::default();
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O_loc_copy(location: &ObjectLocation) -> ObjectLocation {
    location.clone()
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O_loc_copy_shallow(location: &ObjectLocation) -> ObjectLocation {
    location.clone()
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O_loc_copy_deep(location: &ObjectLocation) -> ObjectLocation {
    location.clone()
}

/// Object operation: loc hold file.
#[allow(non_snake_case)]
pub fn H5O_loc_hold_file(location: &mut ObjectLocation) {
    location.held = true;
}

/// Free an object's in-memory resources.
#[allow(non_snake_case)]
pub fn H5O_loc_free(mut location: ObjectLocation) {
    location.file_name = None;
    location.addr = 0;
    location.held = false;
}

/// Object operation: get hdr info.
#[allow(non_snake_case)]
pub fn H5O_get_hdr_info(header: &ObjectHeaderState) -> ObjectInfo {
    let mut msg_count = 0usize;
    let mut has_checksum = false;
    for message in &header.messages {
        if message.msg_type != 0 || !message.data.is_empty() {
            msg_count = msg_count.saturating_add(1);
        }
        if !message.data.is_empty() || message.flags != 0 || message.shared {
            has_checksum = true;
        }
    }
    ObjectInfo {
        addr: header.addr,
        refcount: header.refcount,
        msg_count,
        has_checksum,
    }
}

/// Object operation: get hdr info real.
#[allow(non_snake_case)]
pub fn H5O__get_hdr_info_real(header: &ObjectHeaderState) -> ObjectInfo {
    let msg_count = header
        .messages
        .iter()
        .filter(|message| message.msg_type != 0 || !message.data.is_empty())
        .count();
    ObjectInfo {
        addr: header.addr,
        refcount: header.refcount,
        msg_count,
        has_checksum: H5O_has_chksum(header),
    }
}

/// Return info about an object.
#[allow(non_snake_case)]
pub fn H5O_get_info(header: &ObjectHeaderState) -> ObjectInfo {
    let mut msg_count = 0usize;
    let mut has_checksum = false;
    for message in &header.messages {
        if message.msg_type != 0 || !message.data.is_empty() {
            msg_count = msg_count.saturating_add(1);
        }
        if !message.data.is_empty() || message.flags != 0 || message.shared {
            has_checksum = true;
        }
    }
    if header.addr == u64::MAX || (header.refcount == 0 && header.messages.is_empty()) {
        has_checksum = false;
    }
    ObjectInfo {
        addr: header.addr,
        refcount: header.refcount,
        msg_count,
        has_checksum,
    }
}

/// Return info about an object.
#[allow(non_snake_case)]
pub fn H5Oget_info(header: &ObjectHeaderState) -> ObjectInfo {
    H5O_get_info(header)
}

/// Return info about an object.
#[allow(non_snake_case)]
pub fn H5Oget_info_by_name(
    objects: &BTreeMap<String, ObjectHeaderState>,
    name: &str,
) -> Option<ObjectInfo> {
    objects.get(name).map(H5O_get_info)
}

/// Return info about an object.
#[allow(non_snake_case)]
pub fn H5Oget_info_by_idx1(
    objects: &BTreeMap<String, ObjectHeaderState>,
    idx: usize,
) -> Option<ObjectInfo> {
    let mut current = 0usize;
    for header in objects.values() {
        if current == idx {
            return Some(H5O_get_info(header));
        }
        current = current.checked_add(1)?;
    }
    None
}

/// Return info about an object.
#[allow(non_snake_case)]
pub fn H5Oget_info_by_idx2(
    objects: &BTreeMap<String, ObjectHeaderState>,
    idx: usize,
) -> Option<ObjectInfo> {
    let mut current = 0usize;
    for header in objects.values() {
        if current == idx {
            return Some(ObjectInfo {
                addr: header.addr,
                refcount: header.refcount,
                msg_count: header.messages.len(),
                has_checksum: H5O_has_chksum(header),
            });
        }
        current = current.checked_add(1)?;
    }
    None
}

/// Return info about an object.
#[allow(non_snake_case)]
pub fn H5Oget_info_by_idx3(
    objects: &BTreeMap<String, ObjectHeaderState>,
    idx: usize,
) -> Option<ObjectInfo> {
    let header = objects.values().nth(idx)?;
    Some(ObjectInfo {
        addr: header.addr,
        refcount: header.refcount,
        msg_count: header.messages.len(),
        has_checksum: H5O_has_chksum(header),
    })
}

/// Return native-specific info about an object.
#[allow(non_snake_case)]
pub fn H5O_get_native_info(header: &ObjectHeaderState) -> ObjectInfo {
    let mut info = ObjectInfo {
        addr: header.addr,
        refcount: header.refcount,
        msg_count: 0,
        has_checksum: false,
    };
    for message in &header.messages {
        if message.msg_type == 0 && message.data.is_empty() {
            continue;
        }
        info.msg_count = info.msg_count.saturating_add(1);
        if message.creation_index != 0 || message.flags != 0 || !message.data.is_empty() {
            info.has_checksum = true;
        }
    }
    if H5O_bogus_oh(header) {
        info.has_checksum = false;
    }
    info
}

/// Return native-specific info about an object.
#[allow(non_snake_case)]
pub fn H5Oget_native_info(header: &ObjectHeaderState) -> ObjectInfo {
    let mut info = ObjectInfo::default();
    info.addr = header.addr;
    info.refcount = header.refcount;
    let mut msg_count = 0usize;
    let mut has_checksum = false;
    for message in &header.messages {
        msg_count = msg_count.saturating_add(1);
        if !message.data.is_empty() || message.flags != 0 || message.creation_index != 0 {
            has_checksum = true;
        }
    }
    info.msg_count = msg_count;
    info.has_checksum = has_checksum;
    info
}

/// Return native-specific info about an object.
#[allow(non_snake_case)]
pub fn H5Oget_native_info_by_idx(
    objects: &BTreeMap<String, ObjectHeaderState>,
    idx: usize,
) -> Option<ObjectInfo> {
    let mut current = 0usize;
    for header in objects.values() {
        if current == idx {
            return Some(H5Oget_native_info(header));
        }
        current = current.checked_add(1)?;
    }
    None
}

/// Return the comment associated with an object.
#[allow(non_snake_case)]
pub fn H5Oget_comment_ref(header: &ObjectHeaderState) -> Option<&str> {
    let comment = header.comment.as_ref()?;
    if comment.is_empty() {
        None
    } else {
        Some(comment.as_str())
    }
}

/// Return the comment associated with an object.
#[allow(non_snake_case)]
#[deprecated(note = "use H5Oget_comment_ref to borrow the stored comment")]
pub fn H5Oget_comment(header: &ObjectHeaderState) -> Option<String> {
    H5Oget_comment_ref(header).map(str::to_owned)
}

/// Return the comment associated with an object.
#[allow(non_snake_case)]
pub fn H5Oget_comment_by_name_ref<'a>(
    objects: &'a BTreeMap<String, ObjectHeaderState>,
    name: &str,
) -> Option<&'a str> {
    objects.get(name).and_then(H5Oget_comment_ref)
}

/// Return the comment associated with an object.
#[allow(non_snake_case)]
#[deprecated(note = "use H5Oget_comment_by_name_ref to borrow the stored comment")]
pub fn H5Oget_comment_by_name(
    objects: &BTreeMap<String, ObjectHeaderState>,
    name: &str,
) -> Option<String> {
    H5Oget_comment_by_name_ref(objects, name).map(str::to_owned)
}

/// Return the creation property list for an object.
#[allow(non_snake_case)]
pub fn H5O_get_create_plist(header: &ObjectHeaderState) -> BTreeMap<String, String> {
    let mut plist = BTreeMap::new();
    plist.insert("addr".to_string(), header.addr.to_string());
    plist.insert("refcount".to_string(), header.refcount.to_string());
    plist.insert("messages".to_string(), header.messages.len().to_string());
    plist.insert(
        "has_comment".to_string(),
        header
            .comment
            .as_ref()
            .map(|s| !s.is_empty())
            .unwrap_or(false)
            .to_string(),
    );
    plist
}

/// Object operation: get nlinks.
#[allow(non_snake_case)]
pub fn H5O_get_nlinks(header: &ObjectHeaderState) -> u32 {
    if header.addr == u64::MAX && header.messages.is_empty() {
        0
    } else {
        header.refcount
    }
}

/// Create a new object.
#[allow(non_snake_case)]
pub fn H5O_obj_create(addr: u64) -> ObjectHeaderState {
    let mut header = H5O_create_ohdr(addr);
    header.comment = Some(String::new());
    header
}

/// Object operation: get oh addr.
#[allow(non_snake_case)]
pub fn H5O_get_oh_addr(header: &ObjectHeaderState) -> u64 {
    header.addr
}

/// Object operation: get oh flags.
#[allow(non_snake_case)]
pub fn H5O_get_oh_flags(header: &ObjectHeaderState) -> u8 {
    header.messages.iter().fold(0, |acc, msg| acc | msg.flags)
}

/// Object operation: get oh mtime.
#[allow(non_snake_case)]
pub fn H5O_get_oh_mtime(header: &ObjectHeaderState) -> u64 {
    header
        .messages
        .iter()
        .find(|message| message.msg_type == 0x0012 || message.msg_type == 0x000e)
        .and_then(|message| {
            if message.msg_type == 0x0012 {
                H5O__mtime_new_decode(&message.data).ok()
            } else {
                H5O__mtime_decode(&message.data).ok()
            }
        })
        .unwrap_or(0)
}

/// Object operation: get oh version.
#[allow(non_snake_case)]
pub fn H5O_get_oh_version(_header: &ObjectHeaderState) -> u8 {
    2
}

/// Object operation: get rc and type.
#[allow(non_snake_case)]
pub fn H5O_get_rc_and_type(header: &ObjectHeaderState) -> (u32, &'static str) {
    let refcount = if header.addr == u64::MAX && header.messages.is_empty() {
        0
    } else {
        header.refcount
    };
    let obj_type = if H5O__group_isa(header) {
        "group"
    } else if header.messages.iter().any(|msg| msg.msg_type == 0x0001) {
        "dataset"
    } else {
        "unknown"
    };
    (refcount, obj_type)
}

/// Visit the entries of an object.
#[allow(non_snake_case)]
pub fn H5O__visit_cb_ref<'a>(name: &'a str, header: &ObjectHeaderState) -> Option<&'a str> {
    if header.refcount == 0 && header.messages.is_empty() {
        None
    } else {
        Some(name)
    }
}

/// Visit the entries of an object.
#[allow(non_snake_case)]
pub fn H5O__visit_refs(objects: &BTreeMap<String, ObjectHeaderState>) -> Vec<&str> {
    let mut names = Vec::with_capacity(objects.len());
    for (name, header) in objects {
        if header.refcount == 0 && header.messages.is_empty() {
            continue;
        }
        if name.is_empty() {
            continue;
        }
        if header.addr == u64::MAX && header.messages.is_empty() {
            continue;
        }
        names.push(name.as_str());
    }
    names.sort_unstable();
    names.dedup();
    names
}

/// Visit the entries of an object.
#[allow(non_snake_case)]
pub fn H5O__visit_into(objects: BTreeMap<String, ObjectHeaderState>) -> Vec<String> {
    let mut names = Vec::with_capacity(objects.len());
    for (name, header) in objects {
        if header.refcount == 0 && header.messages.is_empty() {
            continue;
        }
        if name.is_empty() {
            continue;
        }
        if header.addr == u64::MAX && header.messages.is_empty() {
            continue;
        }
        names.push(name);
    }
    names.sort();
    names.dedup();
    names
}

/// Object operation: inc rc.
#[allow(non_snake_case)]
pub fn H5O__inc_rc(header: &mut ObjectHeaderState) {
    H5Oincr_refcount(header);
}

/// Object operation: dec rc.
#[allow(non_snake_case)]
pub fn H5O__dec_rc(header: &mut ObjectHeaderState) {
    H5Odecr_refcount(header);
}

/// Object operation: get proxy.
#[allow(non_snake_case)]
pub fn H5O_get_proxy(header: &ObjectHeaderState) -> u64 {
    header.addr
}

/// Free an object's in-memory resources.
#[allow(non_snake_case)]
pub fn H5O__free(mut header: ObjectHeaderState) {
    for message in &mut header.messages {
        message.data.clear();
        message.flags = 0;
        message.creation_index = 0;
        message.shared = false;
    }
    header.messages.clear();
    header.comment = None;
    header.refcount = 0;
    header.flush_disabled = false;
    drop(header);
}

/// Reset an object to its default state.
#[allow(non_snake_case)]
pub fn H5O__reset_info2(info: &mut ObjectInfo) {
    *info = ObjectInfo::default();
}

/// Object operation: has chksum.
#[allow(non_snake_case)]
pub fn H5O_has_chksum(header: &ObjectHeaderState) -> bool {
    !header.messages.is_empty()
}

/// Object operation: get version bound.
#[allow(non_snake_case)]
pub fn H5O_get_version_bound(_header: &ObjectHeaderState) -> (u8, u8) {
    (0, 4)
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__copy_obj_by_ref(header: &ObjectHeaderState) -> ObjectHeaderState {
    let mut copied = H5O__copy_obj(header);
    copied.comment = header
        .comment
        .as_ref()
        .map(|comment| comment.trim_end_matches('\0').to_string());
    copied
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__copy_expand_ref_object1(token: u64) -> u64 {
    let src = token.to_le_bytes();
    if src.iter().all(|byte| *byte == 0) {
        return 0;
    }
    if src.iter().all(|byte| *byte == 0xff) {
        return 0;
    }
    let mut decoded = 0u64;
    for (idx, byte) in src.iter().enumerate() {
        decoded |= u64::from(*byte) << (idx * 8);
    }
    if decoded == 0 || decoded == u64::MAX {
        return 0;
    }
    let mut dst = [0u8; 8];
    for (idx, byte) in decoded.to_le_bytes().iter().enumerate() {
        dst[idx] = *byte;
    }
    u64::from_le_bytes(dst)
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__copy_expand_ref_region1(region: &[u8]) -> Vec<u8> {
    let mut copied = Vec::with_capacity(region.len());
    for chunk in region.chunks(8) {
        if chunk.len() == 8 {
            let token = u64::from_le_bytes(chunk.try_into().unwrap_or([0; 8]));
            copied.extend_from_slice(&H5O__copy_expand_ref_object1(token).to_le_bytes());
        } else {
            copied.extend_from_slice(chunk);
        }
    }
    copied
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__copy_expand_ref_object2(token: u64) -> u64 {
    let src = token.to_le_bytes();
    if src.iter().all(|byte| *byte == 0) {
        return 0;
    }
    if src.iter().all(|byte| *byte == 0xff) {
        return 0;
    }
    let mut decoded = 0u64;
    for (idx, byte) in src.iter().enumerate() {
        decoded |= u64::from(*byte) << (idx * 8);
    }
    if decoded == 0 || decoded == u64::MAX {
        return 0;
    }
    let mut dst = [0u8; 8];
    for (idx, byte) in decoded.to_le_bytes().iter().enumerate() {
        dst[idx] = *byte;
    }
    u64::from_le_bytes(dst)
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O_copy_expand_ref(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len());
    let mut offset = 0usize;
    for chunk in bytes.chunks(8) {
        if chunk.len() == 8 {
            let token = u64::from_le_bytes(chunk.try_into().unwrap_or([0; 8]));
            let copied = if offset == 0 {
                H5O__copy_expand_ref_object1(token)
            } else {
                H5O__copy_expand_ref_object2(token)
            };
            out.extend_from_slice(&copied.to_le_bytes());
        } else {
            out.extend_from_slice(chunk);
        }
        offset = offset.saturating_add(chunk.len());
    }
    out
}

/// Decode an object from its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__cont_decode(bytes: &[u8]) -> Result<(u64, u64)> {
    if bytes.len() < 16 {
        return Err(Error::InvalidFormat(
            "object-header continuation message is truncated".into(),
        ));
    }
    let addr = u64::from_le_bytes(bytes[0..8].try_into().map_err(|_| {
        Error::InvalidFormat("object-header continuation address is truncated".into())
    })?);
    let size = u64::from_le_bytes(bytes[8..16].try_into().map_err(|_| {
        Error::InvalidFormat("object-header continuation size is truncated".into())
    })?);
    if addr == u64::MAX {
        return Err(Error::InvalidFormat(
            "object-header continuation address is undefined".into(),
        ));
    }
    if size == 0 {
        return Err(Error::InvalidFormat(
            "object-header continuation size is zero".into(),
        ));
    }
    Ok((addr, size))
}

/// Decode an object from its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__cont_decode_with_sizes(
    bytes: &[u8],
    sizeof_addr: u8,
    sizeof_size: u8,
) -> Result<(u64, u64)> {
    let addr_width = usize::from(sizeof_addr);
    let size_width = usize::from(sizeof_size);
    let expected = addr_width
        .checked_add(size_width)
        .ok_or_else(|| Error::InvalidFormat("object-header continuation width overflow".into()))?;
    if bytes.len() < expected {
        return Err(Error::InvalidFormat(
            "object-header continuation message is truncated".into(),
        ));
    }
    let addr = read_le_uint_width(bytes, 0, addr_width, "object-header continuation address")?;
    let size = read_le_uint_width(
        bytes,
        addr_width,
        size_width,
        "object-header continuation size",
    )?;
    if is_undefined_addr_width(addr, sizeof_addr)? {
        return Err(Error::InvalidFormat(
            "object-header continuation address is undefined".into(),
        ));
    }
    if size == 0 {
        return Err(Error::InvalidFormat(
            "object-header continuation size is zero".into(),
        ));
    }
    Ok((addr, size))
}

/// Encode an object to its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__cont_encode(addr: u64, size: u64) -> Result<Vec<u8>> {
    if addr == u64::MAX {
        return Err(Error::InvalidFormat(
            "object-header continuation address is undefined".into(),
        ));
    }
    if size == 0 {
        return Err(Error::InvalidFormat(
            "object-header continuation size is zero".into(),
        ));
    }
    let mut out = Vec::with_capacity(16);
    out.extend_from_slice(&addr.to_le_bytes());
    out.extend_from_slice(&size.to_le_bytes());
    Ok(out)
}

/// Encode an object to its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__cont_encode_with_sizes(
    addr: u64,
    size: u64,
    sizeof_addr: u8,
    sizeof_size: u8,
) -> Result<Vec<u8>> {
    if is_undefined_addr_width(addr, sizeof_addr)? {
        return Err(Error::InvalidFormat(
            "object-header continuation address is undefined".into(),
        ));
    }
    if size == 0 {
        return Err(Error::InvalidFormat(
            "object-header continuation size is zero".into(),
        ));
    }
    let addr_width = usize::from(sizeof_addr);
    let size_width = usize::from(sizeof_size);
    let len = addr_width
        .checked_add(size_width)
        .ok_or_else(|| Error::InvalidFormat("object-header continuation width overflow".into()))?;
    let mut out = Vec::with_capacity(len);
    encode_le_uint_width(
        &mut out,
        addr,
        addr_width,
        "object-header continuation address",
    )?;
    encode_le_uint_width(
        &mut out,
        size,
        size_width,
        "object-header continuation size",
    )?;
    Ok(out)
}

/// Object operation: cont size.
#[allow(non_snake_case)]
pub fn H5O__cont_size(addr: u64, size: u64) -> Result<usize> {
    H5O__cont_size_with_sizes(addr, size, 8, 8)
}

/// Object operation: cont size with sizes.
#[allow(non_snake_case)]
pub fn H5O__cont_size_with_sizes(
    addr: u64,
    size: u64,
    sizeof_addr: u8,
    sizeof_size: u8,
) -> Result<usize> {
    Ok(H5O__cont_encode_with_sizes(addr, size, sizeof_addr, sizeof_size)?.len())
}

/// Internal helper `read_le_u64_at`.
fn read_le_u64_at(data: &[u8], offset: usize, context: &str) -> Result<u64> {
    let end = offset
        .checked_add(8)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))?;
    let bytes: [u8; 8] = data
        .get(offset..end)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))?
        .try_into()
        .map_err(|_| Error::InvalidFormat(format!("{context} is truncated")))?;
    Ok(u64::from_le_bytes(bytes))
}

/// Internal helper `read_le_u32_at`.
fn read_le_u32_at(data: &[u8], offset: usize, context: &str) -> Result<u32> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))?;
    let bytes: [u8; 4] = data
        .get(offset..end)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))?
        .try_into()
        .map_err(|_| Error::InvalidFormat(format!("{context} is truncated")))?;
    Ok(u32::from_le_bytes(bytes))
}

/// Internal helper `read_le_u16_at`.
fn read_le_u16_at(data: &[u8], offset: usize, context: &str) -> Result<u16> {
    let end = offset
        .checked_add(2)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))?;
    let bytes: [u8; 2] = data
        .get(offset..end)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))?
        .try_into()
        .map_err(|_| Error::InvalidFormat(format!("{context} is truncated")))?;
    Ok(u16::from_le_bytes(bytes))
}

/// Internal helper `checked_add`.
fn checked_add(offset: usize, len: usize, context: &str) -> Result<usize> {
    offset
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))
}

/// Internal helper `checked_usize_sum`.
fn checked_usize_sum(parts: &[usize], context: &str) -> Result<usize> {
    parts.iter().try_fold(0usize, |acc, &part| {
        acc.checked_add(part)
            .ok_or_else(|| Error::InvalidFormat(format!("{context} overflow")))
    })
}

/// Internal helper `read_le_uint_width`.
fn read_le_uint_width(data: &[u8], offset: usize, width: usize, context: &str) -> Result<u64> {
    if !(1..=8).contains(&width) {
        return Err(Error::InvalidFormat(format!(
            "{context} width {width} is invalid"
        )));
    }
    let end = checked_add(offset, width, context)?;
    let bytes = data
        .get(offset..end)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))?;
    let mut value = 0u64;
    for (idx, byte) in bytes.iter().enumerate() {
        value |= u64::from(*byte) << (idx * 8);
    }
    Ok(value)
}

/// Internal helper `read_u8_cursor`.
fn read_u8_cursor(data: &[u8], pos: &mut usize, context: &str) -> Result<u8> {
    let value = data
        .get(*pos)
        .copied()
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))?;
    *pos = checked_add(*pos, 1, context)?;
    Ok(value)
}

/// Internal helper `read_le_uint_cursor`.
fn read_le_uint_cursor(data: &[u8], pos: &mut usize, width: usize, context: &str) -> Result<u64> {
    let value = read_le_uint_width(data, *pos, width, context)?;
    *pos = checked_add(*pos, width, context)?;
    Ok(value)
}

/// Internal helper `encode_le_uint_width`.
fn encode_le_uint_width(out: &mut Vec<u8>, value: u64, width: usize, context: &str) -> Result<()> {
    if !(1..=8).contains(&width) {
        return Err(Error::InvalidFormat(format!(
            "{context} width {width} is invalid"
        )));
    }
    if width < 8 && value >= (1u64 << (width * 8)) {
        return Err(Error::InvalidFormat(format!(
            "{context} does not fit in {width} bytes"
        )));
    }
    out.extend((0..width).map(|idx| ((value >> (idx * 8)) & 0xff) as u8));
    Ok(())
}

/// Internal helper `undefined_addr_value`.
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

/// Internal helper `is_undefined_addr_width`.
fn is_undefined_addr_width(addr: u64, sizeof_addr: u8) -> Result<bool> {
    Ok(addr == undefined_addr_value(sizeof_addr)?)
}

/// Internal helper `align8_len_checked`.
fn align8_len_checked(len: usize, context: &str) -> Result<usize> {
    len.checked_add(7)
        .map(|value| value & !7)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} aligned length overflow")))
}

/// Free an object's in-memory resources.
#[allow(non_snake_case)]
pub fn H5O__cont_free(mut cont: (u64, u64)) {
    cont.0 = 0;
    cont.1 = 0;
    let _ = cont;
}

/// Delete an object.
#[allow(non_snake_case)]
pub fn H5O__cont_delete(cont: &mut (u64, u64)) {
    *cont = (0, 0);
}

/// Return a debug-friendly representation of an object.
#[allow(non_snake_case)]
pub fn H5O__cont_debug(cont: (u64, u64)) -> String {
    let addr_state = if cont.0 == u64::MAX {
        "undefined"
    } else {
        "defined"
    };
    format!(
        "cont(addr={}, size={}, addr_state={addr_state})",
        cont.0, cont.1
    )
}

/// Decode an object from its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__ginfo_decode(bytes: &[u8]) -> Result<GroupInfoMessage> {
    if bytes.len() < 2 {
        return Err(Error::InvalidFormat(
            "group info message is truncated".into(),
        ));
    }
    let version = bytes[0];
    if version != 0 {
        return Err(Error::InvalidFormat(format!(
            "group info message version {version}"
        )));
    }
    let flags = bytes[1];
    if flags & !0x03 != 0 {
        return Err(Error::InvalidFormat(format!(
            "group info message flags {flags:#x} are invalid"
        )));
    }

    let mut pos = 2usize;
    let (max_compact, min_dense) = if flags & 0x01 != 0 {
        let max_compact = read_le_u16_at(bytes, pos, "group info max compact")?;
        pos += 2;
        let min_dense = read_le_u16_at(bytes, pos, "group info min dense")?;
        pos += 2;
        (Some(max_compact), Some(min_dense))
    } else {
        (None, None)
    };
    let (estimated_entries, estimated_name_len) = if flags & 0x02 != 0 {
        let estimated_entries = read_le_u16_at(bytes, pos, "group info estimated entries")?;
        pos += 2;
        let estimated_name_len = read_le_u16_at(bytes, pos, "group info estimated name length")?;
        (Some(estimated_entries), Some(estimated_name_len))
    } else {
        (None, None)
    };

    Ok(GroupInfoMessage {
        version,
        max_compact,
        min_dense,
        estimated_entries,
        estimated_name_len,
    })
}

/// Encode an object to its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__ginfo_encode(info: &GroupInfoMessage) -> Result<Vec<u8>> {
    if info.version != 0 {
        return Err(Error::InvalidFormat(format!(
            "group info message version {}",
            info.version
        )));
    }
    if info.max_compact.is_some() != info.min_dense.is_some() {
        return Err(Error::InvalidFormat(
            "group info compact/dense limits must be encoded as a pair".into(),
        ));
    }
    if info.estimated_entries.is_some() != info.estimated_name_len.is_some() {
        return Err(Error::InvalidFormat(
            "group info estimated entry count/name length must be encoded as a pair".into(),
        ));
    }

    let mut flags = 0u8;
    if info.max_compact.is_some() || info.min_dense.is_some() {
        flags |= 0x01;
    }
    if info.estimated_entries.is_some() || info.estimated_name_len.is_some() {
        flags |= 0x02;
    }
    let mut out = vec![info.version, flags];
    if flags & 0x01 != 0 {
        let max_compact = info
            .max_compact
            .expect("validated paired group info limits");
        let min_dense = info.min_dense.expect("validated paired group info limits");
        out.extend_from_slice(&max_compact.to_le_bytes());
        out.extend_from_slice(&min_dense.to_le_bytes());
    }
    if flags & 0x02 != 0 {
        let estimated_entries = info
            .estimated_entries
            .expect("validated paired group info estimates");
        let estimated_name_len = info
            .estimated_name_len
            .expect("validated paired group info estimates");
        out.extend_from_slice(&estimated_entries.to_le_bytes());
        out.extend_from_slice(&estimated_name_len.to_le_bytes());
    }
    Ok(out)
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__ginfo_copy(info: &GroupInfoMessage) -> GroupInfoMessage {
    GroupInfoMessage {
        version: info.version,
        max_compact: info.max_compact,
        min_dense: info.min_dense,
        estimated_entries: info.estimated_entries,
        estimated_name_len: info.estimated_name_len,
    }
}

/// Object operation: ginfo size.
#[allow(non_snake_case)]
pub fn H5O__ginfo_size(info: &GroupInfoMessage) -> Result<usize> {
    H5O__ginfo_encode(info).map(|bytes| bytes.len())
}

/// Free an object's in-memory resources.
#[allow(non_snake_case)]
pub fn H5O__ginfo_free(mut info: GroupInfoMessage) {
    info.version = 0;
    info.max_compact = None;
    info.min_dense = None;
    info.estimated_entries = None;
    info.estimated_name_len = None;
    drop(info);
}

/// Return a debug-friendly representation of an object.
#[allow(non_snake_case)]
pub fn H5O__ginfo_debug(info: &GroupInfoMessage) -> String {
    format!(
        "ginfo(version={}, max_compact={:?}, min_dense={:?})",
        info.version, info.max_compact, info.min_dense
    )
}

/// Create a new object.
#[allow(non_snake_case)]
pub fn H5O__attr_create(header: &mut ObjectHeaderState, name: &str, value: &[u8]) {
    if name.is_empty() {
        return;
    }
    let needle = name.as_bytes();
    for message in &header.messages {
        if message.msg_type != 0x000c {
            continue;
        }
        if let Some(nul) = message.data.iter().position(|byte| *byte == 0) {
            if message.data.get(..nul) == Some(needle) {
                return;
            }
        }
    }
    let mut data = Vec::with_capacity(name.len().saturating_add(1).saturating_add(value.len()));
    data.extend_from_slice(needle);
    data.push(0);
    data.extend_from_slice(value);
    let mut message = ObjectMessage {
        msg_type: 0x000c,
        flags: 0x01,
        creation_index: 0,
        data,
        shared: false,
    };
    message.creation_index = u16::try_from(H5O__attr_count_real(header)).unwrap_or(u16::MAX);
    header.messages.push(message);
    header.refcount = header.refcount.max(1);
    header.flush_disabled = false;
}

/// Open an object.
#[allow(non_snake_case)]
pub fn H5O__attr_open_by_name_ref<'a>(
    header: &'a ObjectHeaderState,
    name: &str,
) -> Option<&'a ObjectMessage> {
    if name.is_empty() {
        return None;
    }
    let needle = name.as_bytes();
    header.messages.iter().find_map(|msg| {
        if msg.msg_type != 0x000c {
            return None;
        }
        let nul = msg.data.iter().position(|byte| *byte == 0)?;
        if msg.data.get(..nul) == Some(needle) {
            Some(msg)
        } else {
            None
        }
    })
}

/// Open the object at the given index.
#[allow(non_snake_case)]
pub fn H5O__attr_open_by_idx_cb_ref(message: &ObjectMessage) -> &ObjectMessage {
    message
}

/// Open the object at the given index.
#[allow(non_snake_case)]
pub fn H5O__attr_open_by_idx_ref(
    header: &ObjectHeaderState,
    index: usize,
) -> Option<&ObjectMessage> {
    header
        .messages
        .iter()
        .filter(|msg| msg.msg_type == 0x000c)
        .nth(index)
}

/// Find an entry in an object.
#[allow(non_snake_case)]
pub fn H5O__attr_find_opened_attr(header: &ObjectHeaderState, name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    let needle = name.as_bytes();
    header.messages.iter().any(|msg| {
        if msg.msg_type != 0x000c {
            return false;
        }
        match msg.data.iter().position(|byte| *byte == 0) {
            Some(nul) => msg.data.get(..nul) == Some(needle),
            None => false,
        }
    })
}

/// Update an object.
#[allow(non_snake_case)]
pub fn H5O__attr_update_shared(message: &mut ObjectMessage, shared: bool) {
    if message.msg_type != 0x000c {
        return;
    }
    if message.data.is_empty() || !message.data.iter().any(|byte| *byte == 0) {
        message.shared = false;
        message.flags &= !0x02;
        return;
    }
    message.shared = shared;
    if shared {
        message.flags |= 0x02;
    } else {
        message.flags &= !0x02;
    }
}

/// Write to an object.
#[allow(non_snake_case)]
pub fn H5O__attr_write_cb(message: &mut ObjectMessage, data: &[u8]) {
    if message.msg_type != 0x000c {
        return;
    }
    if let Some(value_start) = message
        .data
        .iter()
        .position(|byte| *byte == 0)
        .and_then(|pos| pos.checked_add(1))
    {
        message.data.truncate(value_start);
        message.data.extend_from_slice(data);
    }
}

/// Write to an object.
#[allow(non_snake_case)]
pub fn H5O__attr_write(header: &mut ObjectHeaderState, name: &str, value: &[u8]) {
    let needle = name.as_bytes();
    if let Some(pos) = header.messages.iter().position(|msg| {
        if msg.msg_type != 0x000c {
            return false;
        }
        match msg.data.iter().position(|byte| *byte == 0) {
            Some(nul) => msg.data.get(..nul) == Some(needle),
            None => false,
        }
    }) {
        H5O__attr_write_cb(&mut header.messages[pos], value);
    } else {
        H5O__attr_create(header, name, value);
    }
}

/// Object operation: attr rename.
#[allow(non_snake_case)]
pub fn H5O__attr_rename(header: &mut ObjectHeaderState, old_name: &str, new_name: &str) -> bool {
    if old_name.is_empty() || new_name.is_empty() {
        return false;
    }
    let old = old_name.as_bytes();
    for msg in &mut header.messages {
        if msg.msg_type != 0x000c {
            continue;
        }
        let Some(nul) = msg.data.iter().position(|byte| *byte == 0) else {
            continue;
        };
        if msg.data.get(..nul) != Some(old) {
            continue;
        }
        let value_start = match nul.checked_add(1) {
            Some(value_start) => value_start,
            None => return false,
        };
        let value = msg.data[value_start..].to_vec();
        msg.data.clear();
        msg.data.extend_from_slice(new_name.as_bytes());
        msg.data.push(0);
        msg.data.extend_from_slice(&value);
        return true;
    }
    false
}

/// Object operation: attr rename mod cb.
#[allow(non_snake_case)]
pub fn H5O__attr_rename_mod_cb(
    header: &mut ObjectHeaderState,
    old_name: &str,
    new_name: &str,
) -> Result<bool> {
    if old_name.is_empty() || new_name.is_empty() {
        return Ok(false);
    }
    let old = old_name.as_bytes();
    for message in &mut header.messages {
        if message.msg_type != 0x000c {
            continue;
        }
        let Some(nul) = message.data.iter().position(|byte| *byte == 0) else {
            if message.data.starts_with(old) {
                return Err(Error::InvalidFormat(
                    "attribute message name is not NUL-terminated".into(),
                ));
            }
            continue;
        };
        if message.data.get(..nul) != Some(old) {
            continue;
        }
        let value_start = nul
            .checked_add(1)
            .ok_or_else(|| Error::InvalidFormat("attribute message name offset overflow".into()))?;
        let value = message.data[value_start..].to_vec();
        message.data.clear();
        message.data.extend_from_slice(new_name.as_bytes());
        message.data.push(0);
        message.data.extend_from_slice(&value);
        message.flags |= 0x01;
        return Ok(true);
    }
    Ok(false)
}

/// Object operation: attr rename checked.
#[allow(non_snake_case)]
pub fn H5O__attr_rename_checked(
    header: &mut ObjectHeaderState,
    old_name: &str,
    new_name: &str,
) -> Result<bool> {
    if old_name.is_empty() || new_name.is_empty() {
        return Ok(false);
    }
    let old = old_name.as_bytes();
    if let Some(msg) = header.messages.iter_mut().find(|msg| {
        if msg.msg_type != 0x000c {
            return false;
        }
        match msg.data.iter().position(|byte| *byte == 0) {
            Some(nul) => msg.data.get(..nul) == Some(old),
            None => msg.data.starts_with(old),
        }
    }) {
        let value_start = msg
            .data
            .iter()
            .position(|byte| *byte == 0)
            .ok_or_else(|| {
                Error::InvalidFormat("attribute message name is not NUL-terminated".into())
            })?
            .checked_add(1)
            .ok_or_else(|| Error::InvalidFormat("attribute message name offset overflow".into()))?;
        let value = msg.data[value_start..].to_vec();
        msg.data = new_name.as_bytes().to_vec();
        msg.data.push(0);
        msg.data.extend_from_slice(&value);
        Ok(true)
    } else {
        Ok(false)
    }
}

/// Iterate over the entries of an object.
#[allow(non_snake_case)]
pub fn H5O_attr_iterate_real_refs(header: &ObjectHeaderState) -> Vec<&ObjectMessage> {
    let mut attrs: Vec<&ObjectMessage> = header
        .messages
        .iter()
        .filter(|msg| msg.msg_type == 0x000c && msg.data.iter().any(|byte| *byte == 0))
        .map(H5O__attr_open_by_idx_cb_ref)
        .collect();
    attrs.sort_by_key(|msg| msg.creation_index);
    attrs
}

/// Iterate over the entries of an object.
#[allow(non_snake_case)]
pub fn H5O__attr_iterate_refs(header: &ObjectHeaderState) -> Vec<&ObjectMessage> {
    let mut attrs = Vec::new();
    for message in &header.messages {
        if message.msg_type == 0x000c && message.data.iter().any(|byte| *byte == 0) {
            attrs.push(H5O__attr_open_by_idx_cb_ref(message));
        }
    }
    attrs.sort_by_key(|message| message.creation_index);
    attrs
}

/// Remove an entry from an object.
#[allow(non_snake_case)]
pub fn H5O__attr_remove_update(header: &mut ObjectHeaderState) {
    let mut creation_index = 0u16;
    for message in &mut header.messages {
        if message.msg_type != 0x000c {
            continue;
        }
        if message.data.is_empty() || !message.data.iter().any(|byte| *byte == 0) {
            message.flags &= !0x01;
            continue;
        }
        message.creation_index = creation_index;
        message.flags |= 0x01;
        creation_index = creation_index.saturating_add(1);
    }
    header.flush_disabled = false;
}

/// Remove an entry from an object.
#[allow(non_snake_case)]
pub fn H5O__attr_remove(header: &mut ObjectHeaderState, name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    let needle = name.as_bytes();
    if let Some(pos) = header.messages.iter().position(|msg| {
        if msg.msg_type != 0x000c {
            return false;
        }
        match msg.data.iter().position(|byte| *byte == 0) {
            Some(nul) => msg.data.get(..nul) == Some(needle),
            None => false,
        }
    }) {
        header.messages.remove(pos);
        H5O__attr_remove_update(header);
        true
    } else {
        false
    }
}

/// Remove an entry from an object.
#[allow(non_snake_case)]
pub fn H5O__attr_remove_by_idx(
    header: &mut ObjectHeaderState,
    index: usize,
) -> Option<ObjectMessage> {
    let pos = header
        .messages
        .iter()
        .enumerate()
        .filter(|(_, msg)| msg.msg_type == 0x000c)
        .nth(index)
        .map(|(pos, _)| pos)?;
    Some(header.messages.remove(pos))
}

/// Object operation: attr count real.
#[allow(non_snake_case)]
pub fn H5O__attr_count_real(header: &ObjectHeaderState) -> usize {
    header
        .messages
        .iter()
        .filter(|msg| msg.msg_type == 0x000c && msg.data.iter().any(|byte| *byte == 0))
        .count()
}

/// Object operation: attr exists.
#[allow(non_snake_case)]
pub fn H5O__attr_exists(header: &ObjectHeaderState, name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    let needle = name.as_bytes();
    for message in &header.messages {
        if message.msg_type != 0x000c {
            continue;
        }
        let Some(nul) = message.data.iter().position(|byte| *byte == 0) else {
            continue;
        };
        if message.data.get(..nul) == Some(needle) {
            return true;
        }
    }
    false
}

/// Object operation: attr bh info.
#[allow(non_snake_case)]
pub fn H5O__attr_bh_info(header: &ObjectHeaderState) -> usize {
    header
        .messages
        .iter()
        .filter(|msg| msg.msg_type == 0x000c)
        .map(|msg| msg.data.len())
        .sum()
}

/// Decode an object from its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__fill_new_decode(bytes: &[u8]) -> Result<FillValueMessage> {
    if bytes.is_empty() {
        return Err(Error::InvalidFormat("empty fill value message".into()));
    }

    let version = bytes[0];
    match version {
        1 | 2 => {
            if bytes.len() < 4 {
                return Err(Error::InvalidFormat("fill value v2 too short".into()));
            }
            let alloc_time = bytes[1];
            if alloc_time > 3 {
                return Err(Error::InvalidFormat(format!(
                    "fill value v2 allocation time {} is invalid",
                    alloc_time
                )));
            }
            let fill_time = bytes[2];
            if fill_time > 2 {
                return Err(Error::InvalidFormat(format!(
                    "fill value v2 write time {} is invalid",
                    fill_time
                )));
            }
            let defined_state = bytes[3];
            if defined_state > 2 {
                return Err(Error::InvalidFormat(format!(
                    "fill value v2 defined state {} is invalid",
                    defined_state
                )));
            }
            let defined = defined_state != 0;
            let value = if defined {
                if bytes.len() < 8 {
                    return Err(Error::InvalidFormat(
                        "fill value v2 missing value size".into(),
                    ));
                }
                let size = u32::from_le_bytes(bytes[4..8].try_into().map_err(|_| {
                    Error::InvalidFormat("fill value v2 value size is truncated".into())
                })?);
                let size = usize::try_from(size).map_err(|_| {
                    Error::InvalidFormat("fill value v2 value size does not fit in usize".into())
                })?;
                let end = 8usize.checked_add(size).ok_or_else(|| {
                    Error::InvalidFormat("fill value v2 value offset overflow".into())
                })?;
                if end > bytes.len() {
                    return Err(Error::InvalidFormat(
                        "fill value v2 value is truncated".into(),
                    ));
                }
                if size == 0 {
                    None
                } else {
                    Some(bytes[8..end].to_vec())
                }
            } else {
                None
            };
            Ok(FillValueMessage {
                version,
                alloc_time,
                fill_time,
                defined,
                value,
            })
        }
        3 => {
            if bytes.len() < 2 {
                return Err(Error::InvalidFormat("fill value v3 too short".into()));
            }
            let flags = bytes[1];
            if flags & !0x3f != 0 {
                return Err(Error::InvalidFormat(format!(
                    "fill value v3 flags {flags:#x} are invalid"
                )));
            }
            let alloc_time = flags & 0x03;
            let fill_time = (flags >> 2) & 0x03;
            if alloc_time > 3 {
                return Err(Error::InvalidFormat(format!(
                    "fill value v3 allocation time {} is invalid",
                    alloc_time
                )));
            }
            if fill_time > 2 {
                return Err(Error::InvalidFormat(format!(
                    "fill value v3 write time {} is invalid",
                    fill_time
                )));
            }
            let undefined = flags & 0x10 != 0;
            let have_value = flags & 0x20 != 0;
            if undefined && have_value {
                return Err(Error::InvalidFormat(
                    "fill value v3 has both undefined and value-present flags".into(),
                ));
            }
            let value = if have_value {
                if bytes.len() < 6 {
                    return Err(Error::InvalidFormat(
                        "fill value v3 missing value size".into(),
                    ));
                }
                let size = u32::from_le_bytes(bytes[2..6].try_into().map_err(|_| {
                    Error::InvalidFormat("fill value v3 value size is truncated".into())
                })?);
                let size = usize::try_from(size).map_err(|_| {
                    Error::InvalidFormat("fill value v3 value size does not fit in usize".into())
                })?;
                let end = 6usize.checked_add(size).ok_or_else(|| {
                    Error::InvalidFormat("fill value v3 value offset overflow".into())
                })?;
                if end > bytes.len() {
                    return Err(Error::InvalidFormat(
                        "fill value v3 value is truncated".into(),
                    ));
                }
                if size == 0 {
                    None
                } else {
                    Some(bytes[6..end].to_vec())
                }
            } else {
                None
            };
            Ok(FillValueMessage {
                version: 3,
                alloc_time,
                fill_time,
                defined: !undefined,
                value,
            })
        }
        _ => Err(Error::InvalidFormat(format!(
            "fill value message version {version}"
        ))),
    }
}

/// Decode an object from its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__fill_old_decode(bytes: &[u8]) -> Result<FillValueMessage> {
    FillValueMessage::decode_old(bytes)
}

/// Encode an object to its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__fill_old_encode(message: &FillValueMessage) -> Result<Vec<u8>> {
    let value = message.value.as_deref().unwrap_or(&[]);
    let value_len = u32::try_from(value.len())
        .map_err(|_| Error::InvalidFormat("old fill value payload length exceeds u32".into()))?;
    let capacity = 4usize
        .checked_add(value.len())
        .ok_or_else(|| Error::InvalidFormat("old fill value image length overflow".into()))?;
    let mut out = Vec::with_capacity(capacity);
    out.extend_from_slice(&value_len.to_le_bytes());
    out.extend_from_slice(value);
    Ok(out)
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__fill_copy(message: &FillValueMessage) -> FillValueMessage {
    let value = message.value.as_ref().map(|value| {
        let mut copied = Vec::with_capacity(value.len());
        copied.extend_from_slice(value);
        copied
    });
    let defined = message.defined && value.is_some();
    FillValueMessage {
        version: message.version,
        alloc_time: message.alloc_time,
        fill_time: message.fill_time,
        defined,
        value,
    }
}

/// Object operation: fill new size.
#[allow(non_snake_case)]
pub fn H5O__fill_new_size(message: &FillValueMessage) -> usize {
    H5O__fill_new_size_checked(message).unwrap_or(usize::MAX)
}

/// Object operation: fill new size checked.
#[allow(non_snake_case)]
pub fn H5O__fill_new_size_checked(message: &FillValueMessage) -> Result<usize> {
    match message.version {
        1 | 2 => {
            let payload = if message.defined {
                checked_add(
                    4,
                    message.value.as_ref().map_or(0, Vec::len),
                    "fill value payload length",
                )?
            } else {
                0
            };
            checked_add(4, payload, "fill value message size")
        }
        3 => {
            let payload = match message.value.as_ref() {
                Some(value) => checked_add(4, value.len(), "fill value v3 payload length")?,
                None => 0,
            };
            checked_add(2, payload, "fill value v3 message size")
        }
        _ => Err(Error::InvalidFormat(format!(
            "fill value message version {}",
            message.version
        ))),
    }
}

/// Object operation: fill old size.
#[allow(non_snake_case)]
pub fn H5O__fill_old_size(message: &FillValueMessage) -> Result<usize> {
    4usize
        .checked_add(message.value.as_ref().map_or(0, Vec::len))
        .ok_or_else(|| Error::InvalidFormat("old fill value image length overflow".into()))
}

/// Reset an object to its default state.
#[allow(non_snake_case)]
pub fn H5O_fill_reset_dyn(message: &mut FillValueMessage) {
    message.version = match message.version {
        0 | 1 | 2 | 3 => message.version,
        _ => 2,
    };
    message.alloc_time = 2;
    message.fill_time = 2;
    message.value = None;
    message.defined = false;
}

/// Reset an object to its default state.
#[allow(non_snake_case)]
pub fn H5O__fill_reset(message: &mut FillValueMessage) {
    message.version = match message.version {
        0 | 1 | 2 | 3 => message.version,
        _ => 2,
    };
    message.alloc_time = 2;
    message.fill_time = 2;
    message.value = None;
    message.defined = false;
}

/// Free an object's in-memory resources.
#[allow(non_snake_case)]
pub fn H5O__fill_free(mut message: FillValueMessage) {
    message.value.take();
    message.defined = false;
    message.alloc_time = 2;
    message.fill_time = 2;
    drop(message);
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__fill_pre_copy_file(message: &FillValueMessage) -> FillValueMessage {
    H5O__fill_copy(message)
}

/// Object operation: fill set version.
#[allow(non_snake_case)]
pub fn H5O_fill_set_version(bytes: &mut Vec<u8>, version: u8) {
    if bytes.is_empty() {
        bytes.push(version);
    } else {
        bytes[0] = version;
    }
}

/// Reset an object to its default state.
#[allow(non_snake_case)]
pub fn H5O__reset_info1(info: &mut ObjectInfo) {
    *info = ObjectInfo::default();
}

/// Object operation: iterate1 adapter.
#[allow(non_snake_case)]
pub fn H5O__iterate1_adapter(header: &ObjectHeaderState) -> Vec<ObjectMessage> {
    let mut messages = Vec::with_capacity(header.messages.len());
    for message in &header.messages {
        messages.push(ObjectMessage {
            msg_type: message.msg_type,
            flags: message.flags,
            creation_index: message.creation_index,
            data: message.data.to_vec(),
            shared: message.shared,
        });
    }
    messages
}

/// Return info about an object.
#[allow(non_snake_case)]
pub fn H5O__get_info_old(header: &ObjectHeaderState) -> ObjectInfo {
    let mut msg_count = 0usize;
    let mut has_checksum = false;
    for message in &header.messages {
        if message.msg_type != 0 || !message.data.is_empty() {
            msg_count = msg_count.saturating_add(1);
        }
        if !message.data.is_empty() || message.flags != 0 || message.shared {
            has_checksum = true;
        }
    }
    let bogus = header.addr == u64::MAX || (header.refcount == 0 && header.messages.is_empty());
    ObjectInfo {
        addr: header.addr,
        refcount: if bogus { 0 } else { header.refcount },
        msg_count,
        has_checksum: if bogus { false } else { has_checksum },
    }
}

/// Open an object by its on-disk address.
#[allow(non_snake_case)]
pub fn H5Oopen_by_addr(
    objects: &BTreeMap<String, ObjectHeaderState>,
    addr: u64,
) -> Option<ObjectHeaderState> {
    if addr == u64::MAX {
        return None;
    }
    for header in objects.values() {
        if header.addr != addr {
            continue;
        }
        if header.refcount == 0 && header.messages.is_empty() {
            return None;
        }
        let mut opened = H5O__copy_header_real(header);
        opened.refcount = opened.refcount.saturating_add(1);
        return Some(opened);
    }
    None
}

/// Object operation: visit1.
#[allow(non_snake_case)]
pub fn H5Ovisit1_refs(objects: &BTreeMap<String, ObjectHeaderState>) -> Vec<&str> {
    let mut ordered = Vec::with_capacity(objects.len());
    let mut seen_addrs = BTreeSet::new();
    for (name, header) in objects {
        if header.refcount == 0 && header.messages.is_empty() {
            continue;
        }
        if name.is_empty() || header.addr == u64::MAX {
            continue;
        }
        if !seen_addrs.insert(header.addr) {
            continue;
        }
        let depth = name.split('/').filter(|part| !part.is_empty()).count();
        let mut old_info_score = 0usize;
        let mut has_comment = false;
        let mut has_mtime = false;
        let mut has_datatype = false;
        let mut has_attr = false;
        let mut data_bytes = 0usize;
        for message in &header.messages {
            if message.msg_type == 0 || message.data.is_empty() {
                continue;
            }
            data_bytes = data_bytes.saturating_add(message.data.len());
            match message.msg_type {
                0x0001 => {
                    has_comment = true;
                    old_info_score = old_info_score.saturating_add(4);
                }
                0x0002 | 0x0012 => {
                    has_mtime = true;
                    old_info_score = old_info_score.saturating_add(3);
                }
                0x0003 => {
                    has_datatype = true;
                    old_info_score = old_info_score.saturating_add(2);
                }
                0x000c => {
                    has_attr = true;
                    old_info_score = old_info_score.saturating_add(1);
                }
                _ => {}
            }
        }
        let legacy_flags = (usize::from(has_comment) << 3)
            | (usize::from(has_mtime) << 2)
            | (usize::from(has_datatype) << 1)
            | usize::from(has_attr);
        ordered.push((
            depth,
            name.as_str(),
            header.addr,
            old_info_score,
            legacy_flags,
            data_bytes,
        ));
    }
    ordered.sort_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| left.2.cmp(&right.2))
            .then_with(|| right.3.cmp(&left.3))
            .then_with(|| right.4.cmp(&left.4))
            .then_with(|| right.5.cmp(&left.5))
    });
    let mut names = Vec::with_capacity(ordered.len());
    let mut seen_names = BTreeSet::new();
    for (_, name, _, _, _, _) in ordered {
        if seen_names.insert(name) {
            names.push(name);
        }
    }
    names
}

/// Visit the entries of an object.
#[allow(non_snake_case)]
#[deprecated(note = "use H5Ovisit1_refs to avoid allocating cloned object names")]
pub fn H5Ovisit1(objects: &BTreeMap<String, ObjectHeaderState>) -> Vec<String> {
    H5Ovisit1_refs(objects)
        .into_iter()
        .map(str::to_owned)
        .collect()
}

/// Visit the entries of an object.
#[allow(non_snake_case)]
pub fn H5Ovisit_by_name2_refs<'a>(
    objects: &'a BTreeMap<String, ObjectHeaderState>,
    prefix: &str,
) -> Vec<&'a str> {
    objects
        .keys()
        .filter(|name| name.starts_with(prefix))
        .map(String::as_str)
        .collect()
}

/// Visit the entries of an object.
#[allow(non_snake_case)]
#[deprecated(note = "use H5Ovisit_by_name2_refs to avoid allocating cloned object names")]
pub fn H5Ovisit_by_name2(
    objects: &BTreeMap<String, ObjectHeaderState>,
    prefix: &str,
) -> Vec<String> {
    H5Ovisit_by_name2_refs(objects, prefix)
        .into_iter()
        .map(str::to_owned)
        .collect()
}

/// Decode an object from its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__btreek_decode(bytes: &[u8]) -> Result<BTreeKMessage> {
    if bytes.len() < 7 {
        return Err(Error::InvalidFormat(
            "B-tree K values message is truncated".into(),
        ));
    }
    let version = bytes[0];
    if version != 0 {
        return Err(Error::InvalidFormat(format!(
            "B-tree K values message version {version}"
        )));
    }
    Ok(BTreeKMessage {
        version,
        indexed_storage_internal_k: read_le_u16_at(
            bytes,
            1,
            "B-tree K indexed-storage internal node K",
        )?,
        group_internal_k: read_le_u16_at(bytes, 3, "B-tree K group internal node K")?,
        group_leaf_k: read_le_u16_at(bytes, 5, "B-tree K group leaf node K")?,
    })
}

/// Encode an object to its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__btreek_encode(message: &BTreeKMessage) -> Result<Vec<u8>> {
    if message.version != 0 {
        return Err(Error::InvalidFormat(format!(
            "B-tree K values message version {}",
            message.version
        )));
    }
    let mut out = Vec::with_capacity(7);
    out.push(message.version);
    out.extend_from_slice(&message.indexed_storage_internal_k.to_le_bytes());
    out.extend_from_slice(&message.group_internal_k.to_le_bytes());
    out.extend_from_slice(&message.group_leaf_k.to_le_bytes());
    Ok(out)
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__btreek_copy(message: &BTreeKMessage) -> BTreeKMessage {
    BTreeKMessage {
        version: message.version,
        indexed_storage_internal_k: message.indexed_storage_internal_k,
        group_internal_k: message.group_internal_k,
        group_leaf_k: message.group_leaf_k,
    }
}

/// Object operation: btreek size.
#[allow(non_snake_case)]
pub fn H5O__btreek_size(_message: &BTreeKMessage) -> usize {
    7
}

/// Return a debug-friendly representation of an object.
#[allow(non_snake_case)]
pub fn H5O__btreek_debug(message: &BTreeKMessage) -> String {
    format!(
        "btreek(indexed={}, group_internal={}, group_leaf={})",
        message.indexed_storage_internal_k, message.group_internal_k, message.group_leaf_k
    )
}

/// Free an object's in-memory resources.
#[allow(non_snake_case)]
pub fn H5O__unknown_free(mut bytes: Vec<u8>) {
    bytes.clear();
    drop(bytes);
}

/// Decode an object from its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__link_decode(bytes: &[u8]) -> Result<LinkObjectMessage> {
    H5O__link_decode_with_addr_size(bytes, 8)
}

/// Decode an object from its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__link_decode_with_addr_size(bytes: &[u8], sizeof_addr: u8) -> Result<LinkObjectMessage> {
    Ok(LinkObjectMessage {
        message: LinkMessage::decode(bytes, sizeof_addr)?,
        raw_size: bytes.len(),
    })
}

/// Link an object.
#[allow(non_snake_case)]
pub fn H5O__link_size(message: &LinkObjectMessage) -> usize {
    if message.raw_size != 0 {
        return message.raw_size;
    }
    let name_len = message.message.name.len();
    let value_len = match message.message.link_type {
        crate::format::messages::link::LinkType::Hard => 8,
        crate::format::messages::link::LinkType::Soft => message
            .message
            .soft_link_target
            .as_ref()
            .map(|target| target.len())
            .unwrap_or(0),
        crate::format::messages::link::LinkType::External => message
            .message
            .external_link
            .as_ref()
            .map(|(file, path)| file.len().saturating_add(path.len()).saturating_add(2))
            .unwrap_or(0),
        crate::format::messages::link::LinkType::UserDefined(_) => 0,
    };
    2usize
        .saturating_add(name_len)
        .saturating_add(value_len)
        .saturating_add(usize::from(message.message.creation_order.is_some()) * 8)
        .saturating_add(usize::from(message.message.char_encoding != 0))
}

/// Reset an object to its default state.
#[allow(non_snake_case)]
pub fn H5O__link_reset(message: &mut LinkObjectMessage) {
    message.message.name.clear();
    message.message.creation_order = None;
    message.message.char_encoding = 0;
    message.message.hard_link_addr = None;
    message.message.soft_link_target = None;
    message.message.external_link = None;
    message.raw_size = 0;
}

/// Free an object's in-memory resources.
#[allow(non_snake_case)]
pub fn H5O__link_free(mut message: LinkObjectMessage) {
    message.message.name.clear();
    message.message.creation_order = None;
    message.message.hard_link_addr = None;
    message.message.soft_link_target = None;
    message.message.external_link = None;
    message.raw_size = 0;
    drop(message);
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__link_copy_file(message: &LinkObjectMessage) -> LinkObjectMessage {
    LinkObjectMessage {
        message: LinkMessage {
            name: message.message.name.clone(),
            link_type: message.message.link_type,
            creation_order: message.message.creation_order,
            char_encoding: message.message.char_encoding,
            hard_link_addr: message
                .message
                .hard_link_addr
                .filter(|addr| *addr != u64::MAX),
            soft_link_target: message.message.soft_link_target.clone(),
            external_link: message.message.external_link.clone(),
        },
        raw_size: message.raw_size,
    }
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__link_post_copy_file(message: &LinkObjectMessage) -> LinkObjectMessage {
    let mut copied = H5O__link_copy_file(message);
    if copied.message.name.is_empty() {
        copied.raw_size = 0;
    }
    copied
}

/// Return a debug-friendly representation of an object.
#[allow(non_snake_case)]
pub fn H5O__link_debug(message: &LinkObjectMessage) -> String {
    format!(
        "link(name={}, type={:?}, encoding={}, corder={:?}, hard={:?}, soft={:?}, external={:?}, raw_size={})",
        message.message.name,
        message.message.link_type,
        message.message.char_encoding,
        message.message.creation_order,
        message.message.hard_link_addr,
        message.message.soft_link_target,
        message.message.external_link,
        message.raw_size
    )
}

/// Decode an object from its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__linfo_decode(bytes: &[u8]) -> Result<LinkInfoObjectMessage> {
    H5O__linfo_decode_with_addr_size(bytes, 8)
}

/// Decode an object from its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__linfo_decode_with_addr_size(
    bytes: &[u8],
    sizeof_addr: u8,
) -> Result<LinkInfoObjectMessage> {
    Ok(LinkInfoObjectMessage {
        message: LinkInfoMessage::decode(bytes, sizeof_addr)?,
        raw_size: bytes.len(),
    })
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__linfo_copy(message: &LinkInfoObjectMessage) -> LinkInfoObjectMessage {
    LinkInfoObjectMessage {
        message: LinkInfoMessage {
            version: message.message.version,
            flags: message.message.flags,
            max_creation_index: message.message.max_creation_index,
            fractal_heap_addr: message.message.fractal_heap_addr,
            name_btree_addr: message.message.name_btree_addr,
            corder_btree_addr: message.message.corder_btree_addr,
        },
        raw_size: message.raw_size,
    }
}

/// Object operation: linfo size.
#[allow(non_snake_case)]
pub fn H5O__linfo_size(message: &LinkInfoObjectMessage) -> usize {
    message.raw_size
}

/// Free an object's in-memory resources.
#[allow(non_snake_case)]
pub fn H5O__linfo_free(mut message: LinkInfoObjectMessage) {
    message.message.max_creation_index = None;
    message.message.fractal_heap_addr = u64::MAX;
    message.message.name_btree_addr = u64::MAX;
    message.message.corder_btree_addr = None;
    message.raw_size = 0;
    drop(message);
}

/// Delete an object.
#[allow(non_snake_case)]
pub fn H5O__linfo_delete(message: &mut LinkInfoObjectMessage) {
    message.message.max_creation_index = None;
    message.message.fractal_heap_addr = 0;
    message.message.name_btree_addr = 0;
    message.message.corder_btree_addr = None;
    message.raw_size = 0;
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__linfo_copy_file(message: &LinkInfoObjectMessage) -> LinkInfoObjectMessage {
    let mut copied = H5O__linfo_copy(message);
    if copied.message.fractal_heap_addr == u64::MAX {
        copied.message.name_btree_addr = u64::MAX;
        copied.message.corder_btree_addr = None;
    }
    copied
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__linfo_post_copy_file_cb(message: &LinkInfoObjectMessage) -> LinkInfoObjectMessage {
    let mut copied = LinkInfoObjectMessage {
        message: LinkInfoMessage {
            version: message.message.version,
            flags: message.message.flags,
            max_creation_index: message.message.max_creation_index,
            fractal_heap_addr: message.message.fractal_heap_addr,
            name_btree_addr: message.message.name_btree_addr,
            corder_btree_addr: message.message.corder_btree_addr,
        },
        raw_size: message.raw_size,
    };
    if copied.message.fractal_heap_addr == u64::MAX {
        copied.message.name_btree_addr = u64::MAX;
        copied.message.corder_btree_addr = None;
    }
    copied
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__linfo_post_copy_file(message: &LinkInfoObjectMessage) -> LinkInfoObjectMessage {
    let mut copied = H5O__linfo_copy_file(message);
    if copied.message.corder_btree_addr.is_none() {
        copied.message.flags &= !0x02;
    }
    if copied.message.max_creation_index.is_none() {
        copied.message.flags &= !0x01;
    }
    copied
}

/// Return a debug-friendly representation of an object.
#[allow(non_snake_case)]
pub fn H5O__linfo_debug(message: &LinkInfoObjectMessage) -> String {
    format!(
        "linfo(version={}, flags={:#x}, max_corder={:?}, heap={:#x}, name_btree={:#x}, corder_btree={:?}, raw_size={})",
        message.message.version,
        message.message.flags,
        message.message.max_creation_index,
        message.message.fractal_heap_addr,
        message.message.name_btree_addr,
        message.message.corder_btree_addr,
        message.raw_size
    )
}

/// Decode an object from its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__efl_decode(bytes: &[u8]) -> Result<ExternalFileListMessage> {
    let mut pos = 0usize;
    let version = read_u8_cursor(bytes, &mut pos, "external file list version")?;
    if version != 1 {
        return Err(Error::Unsupported(format!(
            "external file list version {version}"
        )));
    }
    if bytes.len() < 8 {
        return Err(Error::InvalidFormat(
            "external file list header is truncated".into(),
        ));
    }
    pos = checked_add(pos, 3, "external file list reserved bytes")?;
    let allocated_slots = u16::try_from(read_le_uint_cursor(
        bytes,
        &mut pos,
        2,
        "external file list allocated slots",
    )?)
    .map_err(|_| Error::InvalidFormat("external file list allocated slots exceeds u16".into()))?;
    if allocated_slots == 0 {
        return Err(Error::InvalidFormat(
            "external file list has no allocated slots".into(),
        ));
    }
    let used_slots = usize::try_from(read_le_uint_cursor(
        bytes,
        &mut pos,
        2,
        "external file list used slots",
    )?)
    .map_err(|_| Error::InvalidFormat("external file list used slots exceeds usize".into()))?;
    if used_slots > usize::from(allocated_slots) {
        return Err(Error::InvalidFormat(
            "external file list uses more slots than allocated".into(),
        ));
    }
    let heap_addr = read_le_uint_cursor(bytes, &mut pos, 8, "external file list heap address")?;
    if heap_addr == u64::MAX {
        return Err(Error::InvalidFormat(
            "external file list heap address is undefined".into(),
        ));
    }
    let mut entries = Vec::with_capacity(used_slots);
    for _ in 0..used_slots {
        let name_offset =
            read_le_uint_cursor(bytes, &mut pos, 8, "external file list name offset")?;
        let file_offset =
            read_le_uint_cursor(bytes, &mut pos, 8, "external file list file offset")?;
        let size = read_le_uint_cursor(bytes, &mut pos, 8, "external file list size")?;
        entries.push(ExternalFileListEntry {
            name_offset,
            file_offset,
            size,
        });
    }
    Ok(ExternalFileListMessage {
        version,
        allocated_slots,
        heap_addr,
        entries,
    })
}

/// Decode an object from its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__efl_decode_with_sizes(
    bytes: &[u8],
    sizeof_addr: u8,
    sizeof_size: u8,
) -> Result<ExternalFileListMessage> {
    let mut pos = 0usize;
    let version = read_u8_cursor(bytes, &mut pos, "external file list version")?;
    if version != 1 {
        return Err(Error::Unsupported(format!(
            "external file list version {version}"
        )));
    }
    pos = checked_add(pos, 3, "external file list reserved bytes")?;
    let allocated_slots = u16::try_from(read_le_uint_cursor(
        bytes,
        &mut pos,
        2,
        "external file list allocated slots",
    )?)
    .map_err(|_| Error::InvalidFormat("external file list allocated slots exceeds u16".into()))?;
    if allocated_slots == 0 {
        return Err(Error::InvalidFormat(
            "external file list has no allocated slots".into(),
        ));
    }
    let used_slots = usize::try_from(read_le_uint_cursor(
        bytes,
        &mut pos,
        2,
        "external file list used slots",
    )?)
    .map_err(|_| Error::InvalidFormat("external file list used slots exceeds usize".into()))?;
    if used_slots > usize::from(allocated_slots) {
        return Err(Error::InvalidFormat(
            "external file list uses more slots than allocated".into(),
        ));
    }
    let heap_addr = read_le_uint_cursor(
        bytes,
        &mut pos,
        usize::from(sizeof_addr),
        "external file list heap address",
    )?;
    if is_undefined_addr_width(heap_addr, sizeof_addr)? {
        return Err(Error::InvalidFormat(
            "external file list heap address is undefined".into(),
        ));
    }
    let mut entries = Vec::with_capacity(used_slots);
    for _ in 0..used_slots {
        entries.push(ExternalFileListEntry {
            name_offset: read_le_uint_cursor(
                bytes,
                &mut pos,
                usize::from(sizeof_size),
                "external file list name offset",
            )?,
            file_offset: read_le_uint_cursor(
                bytes,
                &mut pos,
                usize::from(sizeof_size),
                "external file list file offset",
            )?,
            size: read_le_uint_cursor(
                bytes,
                &mut pos,
                usize::from(sizeof_size),
                "external file list size",
            )?,
        });
    }
    Ok(ExternalFileListMessage {
        version,
        allocated_slots,
        heap_addr,
        entries,
    })
}

/// Encode an object to its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__efl_encode(message: &ExternalFileListMessage) -> Result<Vec<u8>> {
    if message.version != 1 {
        return Err(Error::Unsupported(format!(
            "external file list version {}",
            message.version
        )));
    }
    if message.allocated_slots == 0 {
        return Err(Error::InvalidFormat(
            "external file list has no allocated slots".into(),
        ));
    }
    let used_slots = u16::try_from(message.entries.len()).map_err(|_| {
        Error::InvalidFormat("external file list used slot count exceeds u16".into())
    })?;
    if used_slots > message.allocated_slots {
        return Err(Error::InvalidFormat(
            "external file list uses more slots than allocated".into(),
        ));
    }
    if message.heap_addr == u64::MAX {
        return Err(Error::InvalidFormat(
            "external file list heap address is undefined".into(),
        ));
    }
    let mut out = Vec::with_capacity(H5O__efl_size(message)?);
    out.push(message.version);
    out.extend_from_slice(&[0, 0, 0]);
    out.extend_from_slice(&message.allocated_slots.to_le_bytes());
    out.extend_from_slice(&used_slots.to_le_bytes());
    out.extend_from_slice(&message.heap_addr.to_le_bytes());
    for entry in &message.entries {
        out.extend_from_slice(&entry.name_offset.to_le_bytes());
        out.extend_from_slice(&entry.file_offset.to_le_bytes());
        out.extend_from_slice(&entry.size.to_le_bytes());
    }
    Ok(out)
}

/// Encode an object to its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__efl_encode_with_sizes(
    message: &ExternalFileListMessage,
    sizeof_addr: u8,
    sizeof_size: u8,
) -> Result<Vec<u8>> {
    if message.version != 1 {
        return Err(Error::Unsupported(format!(
            "external file list version {}",
            message.version
        )));
    }
    if message.allocated_slots == 0 {
        return Err(Error::InvalidFormat(
            "external file list has no allocated slots".into(),
        ));
    }
    let used_slots = u16::try_from(message.entries.len()).map_err(|_| {
        Error::InvalidFormat("external file list used slot count exceeds u16".into())
    })?;
    if used_slots > message.allocated_slots {
        return Err(Error::InvalidFormat(
            "external file list uses more slots than allocated".into(),
        ));
    }
    if is_undefined_addr_width(message.heap_addr, sizeof_addr)? {
        return Err(Error::InvalidFormat(
            "external file list heap address is undefined".into(),
        ));
    }
    let addr_width = usize::from(sizeof_addr);
    let size_width = usize::from(sizeof_size);
    let mut out = Vec::with_capacity(H5O__efl_size_with_sizes(message, sizeof_addr, sizeof_size)?);
    out.push(message.version);
    out.extend_from_slice(&[0; 3]);
    out.extend_from_slice(&message.allocated_slots.to_le_bytes());
    out.extend_from_slice(&used_slots.to_le_bytes());
    encode_le_uint_width(
        &mut out,
        message.heap_addr,
        addr_width,
        "external file list heap address",
    )?;
    for entry in &message.entries {
        encode_le_uint_width(
            &mut out,
            entry.name_offset,
            size_width,
            "external file list name offset",
        )?;
        encode_le_uint_width(
            &mut out,
            entry.file_offset,
            size_width,
            "external file list file offset",
        )?;
        encode_le_uint_width(&mut out, entry.size, size_width, "external file list size")?;
    }
    Ok(out)
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__efl_copy(message: &ExternalFileListMessage) -> ExternalFileListMessage {
    ExternalFileListMessage {
        version: message.version,
        allocated_slots: message.allocated_slots,
        heap_addr: message.heap_addr,
        entries: message
            .entries
            .iter()
            .map(|entry| ExternalFileListEntry {
                name_offset: entry.name_offset,
                file_offset: entry.file_offset,
                size: entry.size,
            })
            .collect(),
    }
}

/// Object operation: efl size.
#[allow(non_snake_case)]
pub fn H5O__efl_size(message: &ExternalFileListMessage) -> Result<usize> {
    H5O__efl_size_with_sizes(message, 8, 8)
}

/// Object operation: efl size with sizes.
#[allow(non_snake_case)]
pub fn H5O__efl_size_with_sizes(
    message: &ExternalFileListMessage,
    sizeof_addr: u8,
    sizeof_size: u8,
) -> Result<usize> {
    let addr_width = usize::from(sizeof_addr);
    let size_width = usize::from(sizeof_size);
    if !(1..=8).contains(&addr_width) {
        return Err(Error::InvalidFormat(format!(
            "external file list address width {addr_width} is invalid"
        )));
    }
    if !(1..=8).contains(&size_width) {
        return Err(Error::InvalidFormat(format!(
            "external file list size width {size_width} is invalid"
        )));
    }
    let entry_width = size_width
        .checked_mul(3)
        .ok_or_else(|| Error::InvalidFormat("external file list entry width overflow".into()))?;
    message
        .entries
        .len()
        .checked_mul(entry_width)
        .and_then(|payload| payload.checked_add(8))
        .and_then(|payload| payload.checked_add(addr_width))
        .ok_or_else(|| Error::InvalidFormat("external file list image length overflow".into()))
}

/// Reset an object to its default state.
#[allow(non_snake_case)]
pub fn H5O__efl_reset(message: &mut ExternalFileListMessage) {
    message.entries.clear();
    message.allocated_slots = 0;
    message.heap_addr = u64::MAX;
}

/// Object operation: efl total size.
#[allow(non_snake_case)]
pub fn H5O_efl_total_size(message: &ExternalFileListMessage) -> u64 {
    let mut total = 0u64;
    for entry in &message.entries {
        total = total.saturating_add(entry.size);
    }
    total
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__efl_copy_file(message: &ExternalFileListMessage) -> ExternalFileListMessage {
    let mut entries = Vec::with_capacity(message.entries.len());
    for entry in &message.entries {
        if entry.size == 0 {
            continue;
        }
        entries.push(ExternalFileListEntry {
            name_offset: entry.name_offset,
            file_offset: entry.file_offset,
            size: entry.size,
        });
    }
    let entry_slots = u16::try_from(entries.len()).unwrap_or(u16::MAX);
    ExternalFileListMessage {
        version: message.version,
        allocated_slots: entry_slots.max(message.allocated_slots),
        heap_addr: message.heap_addr,
        entries,
    }
}

/// Return a debug-friendly representation of an object.
#[allow(non_snake_case)]
pub fn H5O__efl_debug(message: &ExternalFileListMessage) -> String {
    format!(
        "efl(version={}, allocated={}, heap_addr={:#x}, entries={}, total_size={})",
        message.version,
        message.allocated_slots,
        message.heap_addr,
        message.entries.len(),
        H5O_efl_total_size(message)
    )
}

/// Decode an object from its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__ainfo_decode(bytes: &[u8]) -> Result<AttributeInfoObjectMessage> {
    H5O__ainfo_decode_with_addr_size(bytes, 8)
}

/// Decode an object from its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__ainfo_decode_with_addr_size(
    bytes: &[u8],
    sizeof_addr: u8,
) -> Result<AttributeInfoObjectMessage> {
    Ok(AttributeInfoObjectMessage {
        message: AttributeInfoMessage::decode(bytes, sizeof_addr)?,
        raw_size: bytes.len(),
    })
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__ainfo_copy(message: &AttributeInfoObjectMessage) -> AttributeInfoObjectMessage {
    AttributeInfoObjectMessage {
        message: AttributeInfoMessage {
            version: message.message.version,
            flags: message.message.flags,
            max_creation_index: message.message.max_creation_index,
            fractal_heap_addr: message.message.fractal_heap_addr,
            name_btree_addr: message.message.name_btree_addr,
            corder_btree_addr: message.message.corder_btree_addr,
        },
        raw_size: message.raw_size,
    }
}

/// Object operation: ainfo size.
#[allow(non_snake_case)]
pub fn H5O__ainfo_size(message: &AttributeInfoObjectMessage) -> usize {
    message.raw_size
}

/// Free an object's in-memory resources.
#[allow(non_snake_case)]
pub fn H5O__ainfo_free(mut message: AttributeInfoObjectMessage) {
    message.message.max_creation_index = None;
    message.message.fractal_heap_addr = u64::MAX;
    message.message.name_btree_addr = u64::MAX;
    message.message.corder_btree_addr = None;
    message.raw_size = 0;
    drop(message);
}

/// Delete an object.
#[allow(non_snake_case)]
pub fn H5O__ainfo_delete(message: &mut AttributeInfoObjectMessage) {
    message.message.max_creation_index = None;
    message.message.fractal_heap_addr = 0;
    message.message.name_btree_addr = 0;
    message.message.corder_btree_addr = None;
    message.raw_size = 0;
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__ainfo_pre_copy_file(
    message: &AttributeInfoObjectMessage,
) -> AttributeInfoObjectMessage {
    H5O__ainfo_copy(message)
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__ainfo_copy_file(message: &AttributeInfoObjectMessage) -> AttributeInfoObjectMessage {
    let mut copied = H5O__ainfo_copy(message);
    if copied.message.fractal_heap_addr == u64::MAX {
        copied.message.name_btree_addr = u64::MAX;
        copied.message.corder_btree_addr = None;
    }
    copied
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__ainfo_post_copy_file(
    message: &AttributeInfoObjectMessage,
) -> AttributeInfoObjectMessage {
    let mut copied = H5O__ainfo_copy_file(message);
    if copied.message.corder_btree_addr.is_none() {
        copied.message.flags &= !0x02;
    }
    if copied.message.max_creation_index.is_none() {
        copied.message.flags &= !0x01;
    }
    copied
}

/// Return a debug-friendly representation of an object.
#[allow(non_snake_case)]
pub fn H5O__ainfo_debug(message: &AttributeInfoObjectMessage) -> String {
    format!(
        "ainfo(version={}, flags={:#x}, max_corder={:?}, heap={:#x}, name_btree={:#x}, corder_btree={:?}, raw_size={})",
        message.message.version,
        message.message.flags,
        message.message.max_creation_index,
        message.message.fractal_heap_addr,
        message.message.name_btree_addr,
        message.message.corder_btree_addr,
        message.raw_size
    )
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__dset_get_copy_file_udata(header: &ObjectHeaderState) -> ObjectHeaderState {
    H5O__copy_header(header)
}

/// Free an object's in-memory resources.
#[allow(non_snake_case)]
pub fn H5O__dset_free_copy_file_udata(mut header: ObjectHeaderState) {
    H5O__delete_oh(&mut header);
    header.addr = u64::MAX;
    drop(header);
}

/// Object operation: dset isa.
#[allow(non_snake_case)]
pub fn H5O__dset_isa(header: &ObjectHeaderState) -> bool {
    header
        .messages
        .iter()
        .any(|msg| msg.msg_type == 0x0001 || msg.msg_type == 0x0003)
}

/// Open an object.
#[allow(non_snake_case)]
pub fn H5O__dset_open(header: &ObjectHeaderState) -> ObjectHeaderState {
    let mut opened = ObjectHeaderState {
        addr: header.addr,
        messages: Vec::with_capacity(header.messages.len()),
        refcount: header.refcount.saturating_add(1),
        comment: header.comment.as_ref().map(|comment| comment.to_string()),
        flush_disabled: false,
    };
    for message in &header.messages {
        opened.messages.push(H5O__copy_mesg(message));
    }
    H5O__chunk_update_idx(&mut opened);
    opened
}

/// Create a new object.
#[allow(non_snake_case)]
pub fn H5O__dset_create(addr: u64) -> ObjectHeaderState {
    let mut header = H5O_create_ohdr(addr);
    if addr != u64::MAX {
        header.messages.push(ObjectMessage {
            msg_type: 0x0001,
            flags: 0,
            creation_index: 0,
            data: Vec::new(),
            shared: false,
        });
    }
    H5O__chunk_update_idx(&mut header);
    header
}

/// Object operation: dset get oloc.
#[allow(non_snake_case)]
pub fn H5O__dset_get_oloc(header: &ObjectHeaderState) -> u64 {
    header.addr
}

/// Object operation: dset bh info.
#[allow(non_snake_case)]
pub fn H5O__dset_bh_info(header: &ObjectHeaderState) -> usize {
    let mut total = 0usize;
    for message in &header.messages {
        if matches!(
            message.msg_type,
            0x0001 | 0x0003 | 0x0004 | 0x0005 | 0x0008 | 0x000b
        ) {
            total = total.saturating_add(message.data.len());
        }
    }
    total
}

/// Flush the object to storage.
#[allow(non_snake_case)]
pub fn H5O__dset_flush(header: &mut ObjectHeaderState) {
    for message in &mut header.messages {
        if message.data.is_empty() {
            message.flags &= !0x01;
        } else {
            message.flags |= 0x01;
        }
    }
    header.flush_disabled = false;
}

/// Object operation: dtype isa.
#[allow(non_snake_case)]
pub fn H5O__dtype_isa(header: &ObjectHeaderState) -> bool {
    header.messages.iter().any(|msg| msg.msg_type == 0x0003)
}

/// Open an object.
#[allow(non_snake_case)]
pub fn H5O__dtype_open(header: &ObjectHeaderState) -> ObjectHeaderState {
    let mut opened = H5O__copy_header(header);
    opened.refcount = opened.refcount.saturating_add(1);
    opened
}

/// Create a new object.
#[allow(non_snake_case)]
pub fn H5O__dtype_create(addr: u64) -> ObjectHeaderState {
    let mut header = H5O_create_ohdr(addr);
    if addr != u64::MAX {
        header.messages.push(ObjectMessage {
            msg_type: 0x0003,
            flags: 0,
            creation_index: 0,
            data: Vec::new(),
            shared: false,
        });
    }
    H5O__chunk_update_idx(&mut header);
    header
}

/// Object operation: dtype get oloc.
#[allow(non_snake_case)]
pub fn H5O__dtype_get_oloc(header: &ObjectHeaderState) -> u64 {
    if header
        .messages
        .iter()
        .any(|message| message.msg_type == 0x0003)
    {
        header.addr
    } else {
        u64::MAX
    }
}

/// Object operation: is attr dense test.
#[allow(non_snake_case)]
pub fn H5O__is_attr_dense_test(header: &ObjectHeaderState) -> bool {
    let mut count = 0usize;
    for message in &header.messages {
        if message.msg_type == 0x000c && message.data.iter().any(|byte| *byte == 0) {
            count = count.saturating_add(1);
        }
    }
    count > 8
}

/// Object operation: is attr empty test.
#[allow(non_snake_case)]
pub fn H5O__is_attr_empty_test(header: &ObjectHeaderState) -> bool {
    for message in &header.messages {
        if message.msg_type != 0x000c {
            continue;
        }
        if message.data.is_empty() {
            continue;
        }
        let Some(nul) = message.data.iter().position(|byte| *byte == 0) else {
            continue;
        };
        if nul == 0 {
            continue;
        }
        if message.data.len() > nul.saturating_add(1) || message.flags & 0x01 != 0 {
            return false;
        }
    }
    true
}

/// Object operation: num attrs test.
#[allow(non_snake_case)]
pub fn H5O__num_attrs_test(header: &ObjectHeaderState) -> usize {
    let mut count = 0usize;
    for message in &header.messages {
        if message.msg_type != 0x000c {
            continue;
        }
        if message.data.is_empty() {
            continue;
        }
        if message.data.iter().any(|byte| *byte == 0) {
            count = count.saturating_add(1)
        }
    }
    count
}

/// Object operation: attr dense info test.
#[allow(non_snake_case)]
pub fn H5O__attr_dense_info_test(header: &ObjectHeaderState) -> usize {
    let mut total = 0usize;
    for message in &header.messages {
        if message.msg_type != 0x000c || message.data.is_empty() {
            continue;
        }
        if let Some(nul) = message.data.iter().position(|byte| *byte == 0) {
            let name_len = nul;
            let value_len = message.data.len().saturating_sub(nul.saturating_add(1));
            total = total.saturating_add(name_len).saturating_add(value_len);
        }
    }
    total
}

/// Object operation: check msg marked test.
#[allow(non_snake_case)]
pub fn H5O__check_msg_marked_test(message: &ObjectMessage) -> bool {
    let present = !message.data.is_empty() && message.flags & 0x01 != 0;
    let shared = message.shared && message.flags & 0x02 != 0;
    present || shared || message.creation_index != 0
}

/// Object operation: expunge chunks test.
#[allow(non_snake_case)]
pub fn H5O__expunge_chunks_test(header: &mut ObjectHeaderState) {
    header.messages.clear();
    header.comment = None;
    header.flush_disabled = false;
    header.refcount = 0;
}

/// Object operation: get rc test.
#[allow(non_snake_case)]
pub fn H5O__get_rc_test(header: &ObjectHeaderState) -> u32 {
    if header.messages.is_empty() && header.addr == u64::MAX {
        0
    } else {
        header.refcount
    }
}

/// Object operation: msg get chunkno test.
#[allow(non_snake_case)]
pub fn H5O__msg_get_chunkno_test(message: &ObjectMessage) -> usize {
    if message.msg_type == 0 || message.msg_type == 0x001a {
        return 0;
    }
    usize::from(message.creation_index)
}

/// Object operation: msg move to new chunk test.
#[allow(non_snake_case)]
pub fn H5O__msg_move_to_new_chunk_test(header: &mut ObjectHeaderState, idx: usize) -> Result<()> {
    if idx >= header.messages.len() {
        return Err(Error::InvalidFormat(
            "object message move index out of range".into(),
        ));
    }
    let mut message = header.messages.remove(idx);
    message.creation_index = header
        .messages
        .iter()
        .map(|message| message.creation_index)
        .max()
        .unwrap_or(0)
        .saturating_add(1);
    header.messages.push(message);
    H5O__chunk_update_idx(header);
    Ok(())
}

/// Object operation: SHARED DECODE.
#[allow(non_snake_case)]
pub fn H5O_SHARED_DECODE(bytes: &[u8]) -> Result<SharedMessageReference> {
    if bytes.len() < 2 {
        return Err(Error::InvalidFormat(
            "shared object-header message reference is truncated".into(),
        ));
    }
    match bytes[0] {
        1 => {
            if bytes.len() < 24 {
                return Err(Error::InvalidFormat(
                    "shared object-header message v1 reference is truncated".into(),
                ));
            }
            let index = u64::from_le_bytes(bytes[8..16].try_into().map_err(|_| {
                Error::InvalidFormat("shared message v1 index is truncated".into())
            })?);
            let addr = u64::from_le_bytes(bytes[16..24].try_into().map_err(|_| {
                Error::InvalidFormat("shared message v1 address is truncated".into())
            })?);
            if addr == u64::MAX {
                return Err(Error::InvalidFormat(
                    "shared object-header message address is undefined".into(),
                ));
            }
            Ok(SharedMessageReference::V1 {
                message_type: bytes[1],
                index,
                addr,
            })
        }
        2 => {
            if bytes.len() < 10 {
                return Err(Error::InvalidFormat(
                    "shared object-header message v2 reference is truncated".into(),
                ));
            }
            let addr = u64::from_le_bytes(bytes[2..10].try_into().map_err(|_| {
                Error::InvalidFormat("shared message v2 address is truncated".into())
            })?);
            if addr == u64::MAX {
                return Err(Error::InvalidFormat(
                    "shared object-header message address is undefined".into(),
                ));
            }
            Ok(SharedMessageReference::V2 {
                message_type: bytes[1],
                addr,
            })
        }
        3 => match bytes[1] {
            1 => {
                if bytes.len() < 10 {
                    return Err(Error::InvalidFormat(
                        "shared object-header message SOHM reference is truncated".into(),
                    ));
                }
                let heap_id = bytes[2..10].try_into().map_err(|_| {
                    Error::InvalidFormat(
                        "shared object-header message SOHM reference is truncated".into(),
                    )
                })?;
                Ok(SharedMessageReference::V3Sohm { heap_id })
            }
            2 => {
                if bytes.len() < 10 {
                    return Err(Error::InvalidFormat(
                        "shared object-header message committed reference is truncated".into(),
                    ));
                }
                let addr = u64::from_le_bytes(bytes[2..10].try_into().map_err(|_| {
                    Error::InvalidFormat("shared message v3 address is truncated".into())
                })?);
                if addr == u64::MAX {
                    return Err(Error::InvalidFormat(
                        "shared object-header message address is undefined".into(),
                    ));
                }
                Ok(SharedMessageReference::V3Committed { addr })
            }
            _ => Err(Error::InvalidFormat(
                "shared object-header message type is invalid".into(),
            )),
        },
        _ => Err(Error::InvalidFormat(
            "shared object-header message version is invalid".into(),
        )),
    }
}

/// Object operation: SHARED DECODE WITH CONTEXT.
#[allow(non_snake_case)]
pub fn H5O_SHARED_DECODE_WITH_CONTEXT(
    bytes: &[u8],
    sizeof_addr: u8,
    sizeof_size: u8,
) -> Result<SharedMessageReference> {
    if bytes.len() < 2 {
        return Err(Error::InvalidFormat(
            "shared object-header message reference is truncated".into(),
        ));
    }
    match bytes[0] {
        1 => {
            let index_start = 8usize;
            let addr_start = checked_add(
                index_start,
                usize::from(sizeof_size),
                "shared message v1 reference address",
            )?;
            let addr_end = checked_add(
                addr_start,
                usize::from(sizeof_addr),
                "shared message v1 address",
            )?;
            if bytes.len() < addr_end {
                return Err(Error::InvalidFormat(
                    "shared object-header message v1 reference is truncated".into(),
                ));
            }
            let index = read_le_uint_width(
                bytes,
                index_start,
                usize::from(sizeof_size),
                "shared message v1 index",
            )?;
            let addr = read_le_uint_width(
                bytes,
                addr_start,
                usize::from(sizeof_addr),
                "shared message v1 address",
            )?;
            if is_undefined_addr_width(addr, sizeof_addr)? {
                return Err(Error::InvalidFormat(
                    "shared object-header message address is undefined".into(),
                ));
            }
            Ok(SharedMessageReference::V1 {
                message_type: bytes[1],
                index,
                addr,
            })
        }
        2 => {
            let addr_end = checked_add(2, usize::from(sizeof_addr), "shared message v2 address")?;
            if bytes.len() < addr_end {
                return Err(Error::InvalidFormat(
                    "shared object-header message v2 reference is truncated".into(),
                ));
            }
            let addr = read_le_uint_width(
                bytes,
                2,
                usize::from(sizeof_addr),
                "shared message v2 address",
            )?;
            if is_undefined_addr_width(addr, sizeof_addr)? {
                return Err(Error::InvalidFormat(
                    "shared object-header message address is undefined".into(),
                ));
            }
            Ok(SharedMessageReference::V2 {
                message_type: bytes[1],
                addr,
            })
        }
        3 => match bytes[1] {
            1 => {
                let end = checked_add(2, 8, "shared SOHM reference")?;
                if bytes.len() < end {
                    return Err(Error::InvalidFormat(
                        "shared object-header message SOHM reference is truncated".into(),
                    ));
                }
                let heap_id = bytes[2..end].try_into().map_err(|_| {
                    Error::InvalidFormat(
                        "shared object-header message SOHM reference is truncated".into(),
                    )
                })?;
                Ok(SharedMessageReference::V3Sohm { heap_id })
            }
            2 => {
                let addr_end =
                    checked_add(2, usize::from(sizeof_addr), "shared message v3 address")?;
                if bytes.len() < addr_end {
                    return Err(Error::InvalidFormat(
                        "shared object-header message committed reference is truncated".into(),
                    ));
                }
                let addr = read_le_uint_width(
                    bytes,
                    2,
                    usize::from(sizeof_addr),
                    "shared message v3 address",
                )?;
                if is_undefined_addr_width(addr, sizeof_addr)? {
                    return Err(Error::InvalidFormat(
                        "shared object-header message address is undefined".into(),
                    ));
                }
                Ok(SharedMessageReference::V3Committed { addr })
            }
            _ => Err(Error::InvalidFormat(
                "shared object-header message type is invalid".into(),
            )),
        },
        _ => Err(Error::InvalidFormat(
            "shared object-header message version is invalid".into(),
        )),
    }
}

/// Object operation: SHARED ENCODE.
#[allow(non_snake_case)]
pub fn H5O_SHARED_ENCODE(reference: &SharedMessageReference) -> Result<Vec<u8>> {
    match reference {
        SharedMessageReference::V1 {
            message_type,
            index,
            addr,
        } => {
            if *addr == u64::MAX {
                return Err(Error::InvalidFormat(
                    "shared object-header message address is undefined".into(),
                ));
            }
            let mut out = Vec::with_capacity(24);
            out.push(1);
            out.push(*message_type);
            out.extend_from_slice(&[0; 6]);
            out.extend_from_slice(&index.to_le_bytes());
            out.extend_from_slice(&addr.to_le_bytes());
            Ok(out)
        }
        SharedMessageReference::V2 { message_type, addr } => {
            if *addr == u64::MAX {
                return Err(Error::InvalidFormat(
                    "shared object-header message address is undefined".into(),
                ));
            }
            let mut out = Vec::with_capacity(10);
            out.push(2);
            out.push(*message_type);
            out.extend_from_slice(&addr.to_le_bytes());
            Ok(out)
        }
        SharedMessageReference::V3Sohm { heap_id } => {
            let mut out = Vec::with_capacity(10);
            out.push(3);
            out.push(1);
            out.extend_from_slice(heap_id);
            Ok(out)
        }
        SharedMessageReference::V3Committed { addr } => {
            if *addr == u64::MAX {
                return Err(Error::InvalidFormat(
                    "shared object-header message address is undefined".into(),
                ));
            }
            let mut out = Vec::with_capacity(10);
            out.push(3);
            out.push(2);
            out.extend_from_slice(&addr.to_le_bytes());
            Ok(out)
        }
    }
}

/// Object operation: SHARED ENCODE WITH CONTEXT.
#[allow(non_snake_case)]
pub fn H5O_SHARED_ENCODE_WITH_CONTEXT(
    reference: &SharedMessageReference,
    sizeof_addr: u8,
    sizeof_size: u8,
) -> Result<Vec<u8>> {
    match reference {
        SharedMessageReference::V1 {
            message_type,
            index,
            addr,
        } => {
            if is_undefined_addr_width(*addr, sizeof_addr)? {
                return Err(Error::InvalidFormat(
                    "shared object-header message address is undefined".into(),
                ));
            }
            let size_width = usize::from(sizeof_size);
            let addr_width = usize::from(sizeof_addr);
            let len = 8usize
                .checked_add(size_width)
                .and_then(|value| value.checked_add(addr_width))
                .ok_or_else(|| {
                    Error::InvalidFormat("shared message v1 reference length overflow".into())
                })?;
            let mut out = Vec::with_capacity(len);
            out.push(1);
            out.push(*message_type);
            out.extend_from_slice(&[0; 6]);
            encode_le_uint_width(&mut out, *index, size_width, "shared message v1 index")?;
            encode_le_uint_width(&mut out, *addr, addr_width, "shared message v1 address")?;
            Ok(out)
        }
        SharedMessageReference::V2 { message_type, addr } => {
            if is_undefined_addr_width(*addr, sizeof_addr)? {
                return Err(Error::InvalidFormat(
                    "shared object-header message address is undefined".into(),
                ));
            }
            let addr_width = usize::from(sizeof_addr);
            let len = 2usize
                .checked_add(addr_width)
                .ok_or_else(|| Error::InvalidFormat("shared message v2 length overflow".into()))?;
            let mut out = Vec::with_capacity(len);
            out.push(2);
            out.push(*message_type);
            encode_le_uint_width(&mut out, *addr, addr_width, "shared message v2 address")?;
            Ok(out)
        }
        SharedMessageReference::V3Sohm { heap_id } => {
            let mut out = Vec::with_capacity(10);
            out.push(3);
            out.push(1);
            out.extend_from_slice(heap_id);
            Ok(out)
        }
        SharedMessageReference::V3Committed { addr } => {
            if is_undefined_addr_width(*addr, sizeof_addr)? {
                return Err(Error::InvalidFormat(
                    "shared object-header message address is undefined".into(),
                ));
            }
            let addr_width = usize::from(sizeof_addr);
            let len = 2usize
                .checked_add(addr_width)
                .ok_or_else(|| Error::InvalidFormat("shared message v3 length overflow".into()))?;
            let mut out = Vec::with_capacity(len);
            out.push(3);
            out.push(2);
            encode_le_uint_width(&mut out, *addr, addr_width, "shared message v3 address")?;
            Ok(out)
        }
    }
}

/// Object operation: SHARED SIZE.
#[allow(non_snake_case)]
pub fn H5O_SHARED_SIZE(reference: &SharedMessageReference) -> Result<usize> {
    match reference {
        SharedMessageReference::V1 { addr, .. } => {
            if *addr == u64::MAX {
                return Err(Error::InvalidFormat(
                    "shared object-header message address is undefined".into(),
                ));
            }
            Ok(24)
        }
        SharedMessageReference::V2 { addr, .. } | SharedMessageReference::V3Committed { addr } => {
            if *addr == u64::MAX {
                return Err(Error::InvalidFormat(
                    "shared object-header message address is undefined".into(),
                ));
            }
            Ok(10)
        }
        SharedMessageReference::V3Sohm { .. } => Ok(10),
    }
}

/// Object operation: SHARED SIZE WITH CONTEXT.
#[allow(non_snake_case)]
pub fn H5O_SHARED_SIZE_WITH_CONTEXT(
    reference: &SharedMessageReference,
    sizeof_addr: u8,
    sizeof_size: u8,
) -> Result<usize> {
    Ok(H5O_SHARED_ENCODE_WITH_CONTEXT(reference, sizeof_addr, sizeof_size)?.len())
}

/// Object operation: SHARED DELETE.
#[allow(non_snake_case)]
pub fn H5O_SHARED_DELETE(reference: &mut Option<SharedMessageReference>) {
    if let Some(old) = reference.as_mut() {
        match old {
            SharedMessageReference::V1 { index, addr, .. } => {
                *index = 0;
                *addr = u64::MAX;
            }
            SharedMessageReference::V2 { addr, .. }
            | SharedMessageReference::V3Committed { addr } => {
                *addr = u64::MAX;
            }
            SharedMessageReference::V3Sohm { heap_id } => {
                *heap_id = [0; 8];
            }
        }
    }
    *reference = None;
}

/// Object operation: SHARED LINK.
#[allow(non_snake_case)]
pub fn H5O_SHARED_LINK(message: &mut ObjectMessage, shared: bool) {
    message.shared = shared;
    if shared {
        message.flags |= 0x02;
    } else {
        message.flags &= !0x02;
    }
}

/// Object operation: SHARED COPY FILE.
#[allow(non_snake_case)]
pub fn H5O_SHARED_COPY_FILE(reference: &SharedMessageReference) -> SharedMessageReference {
    match reference {
        SharedMessageReference::V1 {
            message_type,
            index,
            addr,
        } => SharedMessageReference::V1 {
            message_type: *message_type,
            index: *index,
            addr: *addr,
        },
        SharedMessageReference::V2 { message_type, addr } => SharedMessageReference::V2 {
            message_type: *message_type,
            addr: *addr,
        },
        SharedMessageReference::V3Sohm { heap_id } => {
            SharedMessageReference::V3Sohm { heap_id: *heap_id }
        }
        SharedMessageReference::V3Committed { addr } => {
            SharedMessageReference::V3Committed { addr: *addr }
        }
    }
}

/// Object operation: SHARED POST COPY FILE.
#[allow(non_snake_case)]
pub fn H5O_SHARED_POST_COPY_FILE(reference: &SharedMessageReference) -> SharedMessageReference {
    match reference {
        SharedMessageReference::V1 {
            message_type,
            index,
            addr,
        } => SharedMessageReference::V1 {
            message_type: *message_type,
            index: *index,
            addr: *addr,
        },
        SharedMessageReference::V2 { message_type, addr } => SharedMessageReference::V2 {
            message_type: *message_type,
            addr: *addr,
        },
        SharedMessageReference::V3Sohm { heap_id } => {
            SharedMessageReference::V3Sohm { heap_id: *heap_id }
        }
        SharedMessageReference::V3Committed { addr } => {
            SharedMessageReference::V3Committed { addr: *addr }
        }
    }
}

/// Object operation: SHARED DEBUG.
#[allow(non_snake_case)]
pub fn H5O_SHARED_DEBUG(reference: &SharedMessageReference) -> String {
    match reference {
        SharedMessageReference::V1 {
            message_type,
            index,
            addr,
        } => format!("shared(v1,type={message_type},index={index},addr={addr:#x})"),
        SharedMessageReference::V2 { message_type, addr } => {
            format!("shared(v2,type={message_type},addr={addr:#x})")
        }
        SharedMessageReference::V3Sohm { heap_id } => {
            format!("shared(v3,sohm,heap_id={} bytes)", heap_id.len())
        }
        SharedMessageReference::V3Committed { addr } => {
            format!("shared(v3,committed,addr={addr:#x})")
        }
    }
}

/// Decode an object from its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__dtype_decode_helper(bytes: &[u8]) -> Result<DatatypeMessage> {
    if bytes.len() < 8 {
        return Err(Error::InvalidFormat("datatype message too short".into()));
    }
    let flags = read_le_u32_at(bytes, 0, "datatype message flags")?;
    let version = ((flags >> 4) & 0x0f) as u8;
    if version == 0 || version > DATATYPE_MESSAGE_VERSION_LATEST {
        return Err(Error::InvalidFormat(format!(
            "bad version number for datatype message: {version}"
        )));
    }
    let class_value = (flags & 0x0f) as u8;
    let class = crate::format::messages::datatype::DatatypeClass::from_u8(class_value)?;
    let class_bits = [bytes[1], bytes[2], bytes[3]];
    let size = read_le_u32_at(bytes, 4, "datatype size")?;
    if size == 0 {
        return Err(Error::InvalidFormat("invalid datatype size".into()));
    }
    let size_bits = u64::from(size)
        .checked_mul(8)
        .ok_or_else(|| Error::InvalidFormat("datatype bit size overflow".into()))?;
    let properties = &bytes[8..];
    match class {
        crate::format::messages::datatype::DatatypeClass::FixedPoint
        | crate::format::messages::datatype::DatatypeClass::BitField => {
            if properties.len() < 4 {
                return Err(Error::InvalidFormat(
                    "datatype message truncated fixed-size properties".into(),
                ));
            }
            let bit_offset = u64::from(read_le_u16_at(properties, 0, "datatype bit offset")?);
            let precision = u64::from(read_le_u16_at(properties, 2, "datatype precision")?);
            if precision == 0 {
                return Err(Error::InvalidFormat("precision is zero".into()));
            }
            if bit_offset >= size_bits {
                return Err(Error::InvalidFormat("integer offset out of bounds".into()));
            }
            if bit_offset
                .checked_add(precision)
                .and_then(|value| value.checked_sub(1))
                .is_none_or(|last_bit| last_bit >= size_bits)
            {
                return Err(Error::InvalidFormat(
                    "integer offset+precision out of bounds".into(),
                ));
            }
        }
        crate::format::messages::datatype::DatatypeClass::FloatingPoint => {
            if properties.len() < 12 {
                return Err(Error::InvalidFormat(
                    "datatype message truncated fixed-size properties".into(),
                ));
            }
            if version >= 3 && (class_bits[0] & 0x40 != 0) && (class_bits[0] & 0x01 == 0) {
                return Err(Error::Unsupported(
                    "bad byte order for datatype message".into(),
                ));
            }
            let norm = (class_bits[0] >> 4) & 0x03;
            if norm == 3 {
                return Err(Error::Unsupported(
                    "unknown floating-point normalization".into(),
                ));
            }
            let sign = u64::from(class_bits[1]);
            if sign >= size_bits {
                return Err(Error::InvalidFormat(
                    "sign bit position out of bounds".into(),
                ));
            }
            let bit_offset = u64::from(read_le_u16_at(properties, 0, "float bit offset")?);
            let precision = u64::from(read_le_u16_at(properties, 2, "float precision")?);
            if precision == 0 {
                return Err(Error::InvalidFormat("precision is zero".into()));
            }
            if bit_offset >= size_bits
                || bit_offset
                    .checked_add(precision)
                    .and_then(|value| value.checked_sub(1))
                    .is_none_or(|last_bit| last_bit >= size_bits)
            {
                return Err(Error::InvalidFormat(
                    "floating-point precision range out of bounds".into(),
                ));
            }
            let exp_pos = u64::from(properties[4]);
            let exp_size = u64::from(properties[5]);
            let mant_pos = u64::from(properties[6]);
            let mant_size = u64::from(properties[7]);
            if exp_size == 0 {
                return Err(Error::InvalidFormat("exponent size can't be zero".into()));
            }
            if mant_size == 0 {
                return Err(Error::InvalidFormat("mantissa size can't be zero".into()));
            }
            for (name, pos, width) in [
                ("exponent", exp_pos, exp_size),
                ("mantissa", mant_pos, mant_size),
            ] {
                if pos >= size_bits
                    || pos
                        .checked_add(width)
                        .and_then(|value| value.checked_sub(1))
                        .is_none_or(|last_bit| last_bit >= size_bits)
                {
                    return Err(Error::InvalidFormat(format!("{name} range out of bounds")));
                }
            }
        }
        crate::format::messages::datatype::DatatypeClass::Time => {
            if class_bits[0] & !0x01 != 0 || class_bits[1] != 0 || class_bits[2] != 0 {
                return Err(Error::Unsupported(
                    "time datatype has unsupported class flags".into(),
                ));
            }
            if properties.len() < 2 {
                return Err(Error::InvalidFormat(
                    "time datatype precision is truncated".into(),
                ));
            }
            let precision = u64::from(read_le_u16_at(properties, 0, "time datatype precision")?);
            if precision == 0 || precision > size_bits {
                return Err(Error::InvalidFormat(
                    "time datatype precision out of bounds".into(),
                ));
            }
        }
        crate::format::messages::datatype::DatatypeClass::String => {}
        crate::format::messages::datatype::DatatypeClass::Opaque => {
            let tag_len = usize::from(class_bits[0]);
            let padded = tag_len
                .checked_add(7)
                .map(|value| value & !7)
                .ok_or_else(|| {
                    Error::InvalidFormat("opaque datatype tag length overflow".into())
                })?;
            if properties.len() < padded {
                return Err(Error::InvalidFormat(
                    "opaque datatype tag is truncated".into(),
                ));
            }
        }
        crate::format::messages::datatype::DatatypeClass::Reference => {
            if class_bits[0] > 1 {
                return Err(Error::InvalidFormat(
                    "reference datatype has invalid reference type".into(),
                ));
            }
        }
        crate::format::messages::datatype::DatatypeClass::VarLen => {
            if class_bits[0] & 0x0f > 1 {
                return Err(Error::InvalidFormat(
                    "variable-length datatype has invalid class type".into(),
                ));
            }
        }
        crate::format::messages::datatype::DatatypeClass::Array if version < 2 => {
            return Err(Error::InvalidFormat(
                "array datatype cannot use datatype message version 1".into(),
            ));
        }
        _ => {}
    }
    DatatypeMessage::decode(bytes)
}

/// Decode an object from its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__dtype_decode(bytes: &[u8]) -> Result<DatatypeMessage> {
    H5O__dtype_decode_helper(bytes)
}

/// Encode an object to its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__dtype_encode_helper(message: &DatatypeMessage) -> Result<Vec<u8>> {
    if message.version == 0 || message.version > DATATYPE_MESSAGE_VERSION_LATEST {
        return Err(Error::InvalidFormat(
            "datatype message version is invalid".into(),
        ));
    }
    if message.size == 0 {
        return Err(Error::InvalidFormat("datatype size is zero".into()));
    }
    let size_bits = u64::from(message.size)
        .checked_mul(8)
        .ok_or_else(|| Error::InvalidFormat("datatype bit size overflow".into()))?;
    match message.class {
        crate::format::messages::datatype::DatatypeClass::FixedPoint
        | crate::format::messages::datatype::DatatypeClass::BitField => {
            if message.properties.len() < 4 {
                return Err(Error::InvalidFormat(
                    "datatype message truncated fixed-size properties".into(),
                ));
            }
            if message.class_bits[0] & !0x0f != 0 {
                return Err(Error::Unsupported(
                    "integer datatype has unsupported class flags".into(),
                ));
            }
            let bit_offset = u64::from(read_le_u16_at(
                &message.properties,
                0,
                "datatype bit offset",
            )?);
            let precision = u64::from(read_le_u16_at(
                &message.properties,
                2,
                "datatype precision",
            )?);
            if precision == 0 {
                return Err(Error::InvalidFormat("precision is zero".into()));
            }
            if bit_offset >= size_bits
                || bit_offset
                    .checked_add(precision)
                    .and_then(|value| value.checked_sub(1))
                    .is_none_or(|last_bit| last_bit >= size_bits)
            {
                return Err(Error::InvalidFormat(
                    "integer offset+precision out of bounds".into(),
                ));
            }
        }
        crate::format::messages::datatype::DatatypeClass::FloatingPoint => {
            if message.properties.len() < 12 {
                return Err(Error::InvalidFormat(
                    "datatype message truncated fixed-size properties".into(),
                ));
            }
            if message.class_bits[0] & !0x7f != 0 {
                return Err(Error::Unsupported(
                    "floating-point datatype has unsupported class flags".into(),
                ));
            }
            if message.version < 3 && message.class_bits[0] & 0x40 != 0 {
                return Err(Error::Unsupported(
                    "VAX byte order requires datatype message version 3".into(),
                ));
            }
            let norm = (message.class_bits[0] >> 4) & 0x03;
            if norm == 3 {
                return Err(Error::Unsupported(
                    "normalization scheme is not supported in file format yet".into(),
                ));
            }
            let sign = u64::from(message.class_bits[1]);
            if sign >= size_bits {
                return Err(Error::InvalidFormat(
                    "sign bit position out of bounds".into(),
                ));
            }
            let bit_offset = u64::from(read_le_u16_at(&message.properties, 0, "float bit offset")?);
            let precision = u64::from(read_le_u16_at(&message.properties, 2, "float precision")?);
            if precision == 0 {
                return Err(Error::InvalidFormat("precision is zero".into()));
            }
            if bit_offset >= size_bits
                || bit_offset
                    .checked_add(precision)
                    .and_then(|value| value.checked_sub(1))
                    .is_none_or(|last_bit| last_bit >= size_bits)
            {
                return Err(Error::InvalidFormat(
                    "floating-point precision range out of bounds".into(),
                ));
            }
            for (name, pos, width) in [
                (
                    "exponent",
                    u64::from(message.properties[4]),
                    u64::from(message.properties[5]),
                ),
                (
                    "mantissa",
                    u64::from(message.properties[6]),
                    u64::from(message.properties[7]),
                ),
            ] {
                if width == 0 {
                    return Err(Error::InvalidFormat(format!("{name} size can't be zero")));
                }
                if pos >= size_bits
                    || pos
                        .checked_add(width)
                        .and_then(|value| value.checked_sub(1))
                        .is_none_or(|last_bit| last_bit >= size_bits)
                {
                    return Err(Error::InvalidFormat(format!("{name} range out of bounds")));
                }
            }
        }
        crate::format::messages::datatype::DatatypeClass::Time => {
            if message.class_bits[0] & !0x01 != 0
                || message.class_bits[1] != 0
                || message.class_bits[2] != 0
            {
                return Err(Error::Unsupported(
                    "time datatype has unsupported class flags".into(),
                ));
            }
            if message.properties.len() < 2 {
                return Err(Error::InvalidFormat(
                    "time datatype precision is truncated".into(),
                ));
            }
            let precision = u64::from(read_le_u16_at(
                &message.properties,
                0,
                "time datatype precision",
            )?);
            if precision == 0 || precision > size_bits {
                return Err(Error::InvalidFormat(
                    "time datatype precision out of bounds".into(),
                ));
            }
        }
        crate::format::messages::datatype::DatatypeClass::String => {
            if message.class_bits[1] != 0 || message.class_bits[2] != 0 {
                return Err(Error::Unsupported(
                    "string datatype has unsupported class flags".into(),
                ));
            }
        }
        crate::format::messages::datatype::DatatypeClass::Opaque => {
            let aligned = usize::from(message.class_bits[0]);
            if aligned & 7 != 0 {
                return Err(Error::InvalidFormat(
                    "opaque datatype tag length is not aligned".into(),
                ));
            }
            if message.properties.len() < aligned {
                return Err(Error::InvalidFormat(
                    "opaque datatype tag is truncated".into(),
                ));
            }
        }
        crate::format::messages::datatype::DatatypeClass::Compound
        | crate::format::messages::datatype::DatatypeClass::Enum => {
            if message.class_bits[2] != 0 {
                return Err(Error::Unsupported(
                    "datatype member count uses unsupported class flags".into(),
                ));
            }
        }
        crate::format::messages::datatype::DatatypeClass::Reference => {
            if message.class_bits[0] & 0x0f > 1 {
                return Err(Error::InvalidFormat(
                    "reference datatype has invalid reference type".into(),
                ));
            }
        }
        crate::format::messages::datatype::DatatypeClass::VarLen => {
            if message.class_bits[0] & 0x0f > 1 {
                return Err(Error::InvalidFormat(
                    "variable-length datatype has invalid class type".into(),
                ));
            }
        }
        crate::format::messages::datatype::DatatypeClass::Array if message.version < 2 => {
            return Err(Error::InvalidFormat(
                "array datatype cannot use datatype message version 1".into(),
            ));
        }
        _ => {}
    }
    let total_size = 8usize
        .checked_add(message.properties.len())
        .ok_or_else(|| Error::InvalidFormat("datatype message image length overflow".into()))?;
    let flags = (u32::from(message.version) << 4)
        | u32::from(message.class as u8)
        | (u32::from(message.class_bits[0]) << 8)
        | (u32::from(message.class_bits[1]) << 16)
        | (u32::from(message.class_bits[2]) << 24);
    let mut out = Vec::with_capacity(total_size);
    out.extend_from_slice(&flags.to_le_bytes());
    out.extend_from_slice(&message.size.to_le_bytes());
    out.extend_from_slice(&message.properties);
    let decoded = H5O__dtype_decode_helper(&out)?;
    if decoded.version != message.version
        || decoded.class != message.class
        || decoded.size != message.size
    {
        return Err(Error::InvalidFormat(
            "datatype encode/decode roundtrip changed header fields".into(),
        ));
    }
    Ok(out)
}

/// Encode an object to its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__dtype_encode(message: &DatatypeMessage) -> Result<Vec<u8>> {
    let class_and_version = (message.version << 4) | (message.class as u8);
    let mut out = Vec::with_capacity(H5O__dtype_size(message)?);
    out.push(class_and_version);
    out.extend_from_slice(&message.class_bits);
    out.extend_from_slice(&message.size.to_le_bytes());
    out.extend_from_slice(&message.properties);
    Ok(out)
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__dtype_copy(message: &DatatypeMessage) -> DatatypeMessage {
    DatatypeMessage {
        version: message.version,
        class: message.class,
        class_bits: message.class_bits,
        size: message.size,
        properties: message.properties.to_vec(),
    }
}

/// Reset an object to its default state.
#[allow(non_snake_case)]
pub fn H5O__dtype_reset(message: &mut DatatypeMessage) {
    message.properties.clear();
}

/// Object operation: dtype can share.
#[allow(non_snake_case)]
pub fn H5O__dtype_can_share(message: &DatatypeMessage) -> bool {
    message.size > 0
        && !matches!(
            message.class,
            crate::format::messages::datatype::DatatypeClass::VarLen
        )
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__dtype_pre_copy_file(message: &DatatypeMessage) -> DatatypeMessage {
    DatatypeMessage {
        version: message.version,
        class: message.class,
        class_bits: message.class_bits,
        size: message.size,
        properties: message.properties.to_vec(),
    }
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
pub fn H5O__dtype_copy_file(message: &DatatypeMessage) -> DatatypeMessage {
    let mut copied = DatatypeMessage {
        version: message.version,
        class: message.class,
        class_bits: message.class_bits,
        size: message.size,
        properties: message.properties.to_vec(),
    };
    if copied.size == 0 {
        copied.properties.clear();
        copied.class_bits = [0; 3];
    }
    if copied.properties.is_empty() {
        copied.class_bits[0] &= !0x80;
    }
    copied
}

/// Return a debug-friendly representation of an object.
#[allow(non_snake_case)]
pub fn H5O__dtype_debug(message: &DatatypeMessage) -> String {
    let order = if message.class == crate::format::messages::datatype::DatatypeClass::String {
        "none"
    } else if message.class_bits[0] & 0x40 != 0 {
        "vax"
    } else if message.class_bits[0] & 0x01 != 0 {
        "be"
    } else {
        "le"
    };
    let mut fields = Vec::new();
    fields.push(format!("version={}", message.version));
    fields.push(format!("class={:?}", message.class));
    fields.push(format!("size={}", message.size));
    fields.push(format!("order={order}"));
    fields.push(format!("class_bits={:02x?}", message.class_bits));
    fields.push(format!("properties={}", message.properties.len()));
    match message.class {
        crate::format::messages::datatype::DatatypeClass::FixedPoint
        | crate::format::messages::datatype::DatatypeClass::BitField
            if message.properties.len() >= 4 =>
        {
            let bit_offset = u16::from_le_bytes([message.properties[0], message.properties[1]]);
            let precision = u16::from_le_bytes([message.properties[2], message.properties[3]]);
            let low_pad = if message.class_bits[0] & 0x02 != 0 {
                "one"
            } else {
                "zero"
            };
            let high_pad = if message.class_bits[0] & 0x04 != 0 {
                "one"
            } else {
                "zero"
            };
            fields.push(format!("bit_offset={bit_offset}"));
            fields.push(format!("precision={precision}"));
            fields.push(format!("low_pad={low_pad}"));
            fields.push(format!("high_pad={high_pad}"));
            fields.push(format!(
                "signed={}",
                if message.class_bits[0] & 0x08 != 0 {
                    1
                } else {
                    0
                }
            ));
        }
        crate::format::messages::datatype::DatatypeClass::FloatingPoint
            if message.properties.len() >= 12 =>
        {
            let bit_offset = u16::from_le_bytes([message.properties[0], message.properties[1]]);
            let precision = u16::from_le_bytes([message.properties[2], message.properties[3]]);
            let exp_bias = u32::from_le_bytes([
                message.properties[8],
                message.properties[9],
                message.properties[10],
                message.properties[11],
            ]);
            let low_pad = if message.class_bits[0] & 0x02 != 0 {
                "one"
            } else {
                "zero"
            };
            let high_pad = if message.class_bits[0] & 0x04 != 0 {
                "one"
            } else {
                "zero"
            };
            let internal_pad = if message.class_bits[0] & 0x08 != 0 {
                "one"
            } else {
                "zero"
            };
            let norm_code = (message.class_bits[0] >> 4) & 0x03;
            let normalization = match norm_code {
                0 => "none".to_string(),
                1 => "msb set".to_string(),
                2 => "implied".to_string(),
                _ => format!("H5T_NORM_{norm_code}"),
            };
            fields.push(format!("bit_offset={bit_offset}"));
            fields.push(format!("sign={}", message.class_bits[1]));
            fields.push(format!("precision={precision}"));
            fields.push(format!("low_pad={low_pad}"));
            fields.push(format!("high_pad={high_pad}"));
            fields.push(format!("internal_pad={internal_pad}"));
            fields.push(format!("normalization={normalization}"));
            fields.push(format!("exp_pos={}", message.properties[4]));
            fields.push(format!("exp_size={}", message.properties[5]));
            fields.push(format!("mant_pos={}", message.properties[6]));
            fields.push(format!("mant_size={}", message.properties[7]));
            fields.push(format!("exp_bias={exp_bias}"));
        }
        crate::format::messages::datatype::DatatypeClass::String => {
            let padding_code = message.class_bits[0] & 0x0f;
            let padding = match padding_code {
                0 => "nullterm".to_string(),
                1 => "nullpad".to_string(),
                2 => "spacepad".to_string(),
                3..=15 => format!("H5T_STR_RESERVED_{padding_code}"),
                _ => format!("unknown string padding: {padding_code}"),
            };
            let charset_code = (message.class_bits[0] >> 4) & 0x0f;
            let charset = match charset_code {
                0 => "ASCII".to_string(),
                1 => "UTF-8".to_string(),
                2..=15 => format!("H5T_CSET_RESERVED_{charset_code}"),
                _ => format!("unknown character set: {charset_code}"),
            };
            fields.push(format!("padding={padding}"));
            fields.push(format!("charset={charset}"));
        }
        crate::format::messages::datatype::DatatypeClass::Opaque => {
            fields.push(format!("tag_len={}", message.class_bits[0]));
            let tag_len = usize::from(message.class_bits[0]);
            if tag_len > 0 && message.properties.len() >= tag_len {
                let nul = message.properties[..tag_len]
                    .iter()
                    .position(|byte| *byte == 0)
                    .unwrap_or(tag_len);
                let tag = String::from_utf8_lossy(&message.properties[..nul]);
                fields.push(format!("tag=\"{tag}\""));
            }
        }
        crate::format::messages::datatype::DatatypeClass::Compound
        | crate::format::messages::datatype::DatatypeClass::Enum => {
            let members =
                u16::from(message.class_bits[0]) | (u16::from(message.class_bits[1]) << 8);
            fields.push(format!("members={members}"));
            fields.push(format!("raw_body={} bytes", message.properties.len()));
        }
        crate::format::messages::datatype::DatatypeClass::Array => {
            fields.push(format!("rank={}", message.class_bits[0]));
            let rank = usize::from(message.class_bits[0]);
            if !message.properties.is_empty() {
                let mut p = 0usize;
                let mut dims = Vec::new();
                if p < message.properties.len() {
                    let ndims = usize::from(message.properties[p]);
                    p += 1;
                    if message.version < 3 {
                        p = p.saturating_add(3);
                    }
                    for _ in 0..ndims.min(rank) {
                        if p.checked_add(4)
                            .is_some_and(|end| end <= message.properties.len())
                        {
                            dims.push(u32::from_le_bytes([
                                message.properties[p],
                                message.properties[p + 1],
                                message.properties[p + 2],
                                message.properties[p + 3],
                            ]));
                            p += 4;
                        }
                    }
                }
                if !dims.is_empty() {
                    fields.push(format!("dims={dims:?}"));
                }
            }
        }
        crate::format::messages::datatype::DatatypeClass::Reference => {
            let ref_type = message.class_bits[0] & 0x0f;
            let ref_version = (message.class_bits[0] >> 4) & 0x0f;
            fields.push(format!("ref_type={ref_type}"));
            fields.push(format!("ref_version={ref_version}"));
        }
        crate::format::messages::datatype::DatatypeClass::VarLen => {
            let vlen_type_code = message.class_bits[0] & 0x0f;
            let vlen_type = match vlen_type_code {
                0 => "sequence".to_string(),
                1 => "string".to_string(),
                _ => format!("H5T_VLEN_{vlen_type_code}"),
            };
            fields.push(format!("vlen_type={vlen_type}"));
            if vlen_type_code == 1 {
                let padding_code = (message.class_bits[0] >> 4) & 0x0f;
                let charset_code = message.class_bits[1] & 0x0f;
                fields.push(format!("padding={padding_code}"));
                fields.push(format!("charset={charset_code}"));
            }
            fields.push(format!("base={} bytes", message.properties.len()));
        }
        crate::format::messages::datatype::DatatypeClass::Time if message.properties.len() >= 2 => {
            let precision = u16::from_le_bytes([message.properties[0], message.properties[1]]);
            fields.push(format!("precision={precision}"));
        }
        crate::format::messages::datatype::DatatypeClass::Time => {}
        _ => {
            fields.push(format!("raw_body={} bytes", message.properties.len()));
        }
    }
    format!("dtype({})", fields.join(", "))
}

/// Object operation: dtype size.
#[allow(non_snake_case)]
pub fn H5O__dtype_size(message: &DatatypeMessage) -> Result<usize> {
    8usize
        .checked_add(message.properties.len())
        .ok_or_else(|| Error::InvalidFormat("datatype message image length overflow".into()))
}

/// Decode an object from its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__name_decode_ref(bytes: &[u8]) -> Result<&str> {
    let nul = bytes
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(bytes.len());
    std::str::from_utf8(&bytes[..nul])
        .map_err(|_| Error::InvalidFormat("object name is not UTF-8".into()))
}

/// Decode an object from its on-disk representation.
#[allow(non_snake_case)]
#[deprecated(note = "use H5O__name_decode_ref to borrow from the input image")]
pub fn H5O__name_decode(bytes: &[u8]) -> Result<String> {
    H5O__name_decode_ref(bytes).map(str::to_owned)
}

/// Encode an object to its on-disk representation.
#[allow(non_snake_case)]
pub fn H5O__name_encode_into(name: &str, out: &mut Vec<u8>) -> Result<()> {
    let len = name
        .len()
        .checked_add(1)
        .ok_or_else(|| Error::InvalidFormat("object name image length overflow".into()))?;
    out.reserve(len);
    out.extend_from_slice(name.as_bytes());
    out.push(0);
    Ok(())
}

/// Encode an object to its on-disk representation.
#[allow(non_snake_case)]
#[deprecated(note = "use H5O__name_encode_into to append into a caller-owned buffer")]
pub fn H5O__name_encode(name: &str) -> Result<Vec<u8>> {
    let len = name
        .len()
        .checked_add(1)
        .ok_or_else(|| Error::InvalidFormat("object name image length overflow".into()))?;
    let mut out = Vec::with_capacity(len);
    H5O__name_encode_into(name, &mut out)?;
    Ok(out)
}

/// Return a deep copy of an object.
#[allow(non_snake_case)]
#[deprecated(note = "use str::to_owned where an owned copy is actually required")]
pub fn H5O__name_copy(name: &str) -> String {
    name.to_owned()
}

/// Object operation: name size.
#[allow(non_snake_case)]
pub fn H5O__name_size(name: &str) -> usize {
    name.len() + 1
}

/// Reset an object to its default state.
#[allow(non_snake_case)]
pub fn H5O__name_reset(name: &mut String) {
    name.clear();
}

/// Return a debug-friendly representation of an object.
#[allow(non_snake_case)]
pub fn H5O__name_debug_fmt(name: &str, out: &mut dyn fmt::Write) -> fmt::Result {
    let state = if name.is_empty() { "empty" } else { "set" };
    let nul = name.as_bytes().iter().position(|byte| *byte == 0);
    write!(
        out,
        "name(len={}, state={}, nul={:?}, value={name})",
        name.len(),
        state,
        nul
    )
}

/// Return a debug-friendly representation of an object.
#[allow(non_snake_case)]
#[deprecated(note = "use H5O__name_debug_fmt to write into a caller-provided formatter")]
pub fn H5O__name_debug(name: &str) -> String {
    let mut text = String::new();
    H5O__name_debug_fmt(name, &mut text).expect("writing to String cannot fail");
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn object_messages_roundtrip_and_remove() {
        let msg = H5O__msg_alloc(42, b"abc".to_vec());
        let mut encoded = Vec::new();
        H5O_msg_encode_into(&msg, &mut encoded).unwrap();
        let decoded = H5O_msg_decode(&encoded).unwrap();
        assert_eq!(decoded.msg_type, 42);
        assert_eq!(decoded.data, b"abc");

        let mut header = ObjectHeaderState::default();
        H5O_msg_append_oh(&mut header, decoded);
        assert!(H5O_msg_exists(&header, 42));
        assert_eq!(H5O_msg_remove(&mut header, 42).unwrap().data, b"abc");
    }

    #[test]
    fn object_cache_serialize_checks_message_image_sizes() {
        let first = ObjectMessage {
            msg_type: 42,
            flags: 0x80,
            creation_index: 7,
            data: b"abc".to_vec(),
            shared: false,
        };
        let second = H5O__msg_alloc(43, b"z".to_vec());
        let header = ObjectHeaderState {
            messages: vec![first.clone(), second.clone()],
            ..ObjectHeaderState::default()
        };

        assert_eq!(H5O_msg_size_oh(&header).unwrap(), 4);
        let mut first_image = Vec::new();
        H5O_msg_encode_into(&first, &mut first_image).unwrap();
        assert_eq!(first_image.len(), 8);
        let mut image = Vec::new();
        H5O__cache_serialize_into(&header, &mut image).unwrap();
        let mut second_image = Vec::new();
        H5O_msg_encode_into(&second, &mut second_image).unwrap();
        assert_eq!(image.len(), first_image.len() + second_image.len());
        assert_eq!(&image[..first_image.len()], &first_image);
        assert_eq!(H5O__cache_image_len(&header).unwrap(), image.len());
    }

    #[test]
    fn caller_owned_object_encoders_append_to_existing_output() {
        let msg = ObjectMessage {
            msg_type: 42,
            flags: 0x80,
            creation_index: 7,
            data: b"abc".to_vec(),
            shared: false,
        };
        let mut msg_image = b"stale".to_vec();
        H5O_msg_encode_into(&msg, &mut msg_image).unwrap();
        assert_eq!(&msg_image[..5], b"stale");
        assert_eq!(&msg_image[5..], &[42, 0, 0x80, 7, 0, b'a', b'b', b'c']);

        let header = ObjectHeaderState {
            messages: vec![msg.clone()],
            ..ObjectHeaderState::default()
        };
        let mut header_image = b"stale".to_vec();
        H5O__cache_serialize_into(&header, &mut header_image).unwrap();
        assert_eq!(&header_image[..5], b"stale");
        assert_eq!(&header_image[5..], &msg_image[5..]);

        let raw_chunk = ObjectHeaderChunkImage {
            is_v2_continuation: false,
            raw: vec![1, 2, 3],
        };
        let mut chunk_image = b"stale".to_vec();
        H5O__cache_chk_serialize_into(&raw_chunk, &mut chunk_image).unwrap();
        assert_eq!(chunk_image, b"stale\x01\x02\x03".to_vec());

        let fsinfo = FsInfoMessage {
            version: 1,
            free_space_strategy: 2,
            persist: false,
            threshold: 8,
            page_size: 4096,
            pgend_meta_thres: 0,
            eoa_pre_fsm_fsalloc: u64::MAX,
            fs_addr: Vec::new(),
            sizeof_addr: 8,
            sizeof_size: 8,
        };
        let mut fsinfo_image = b"stale".to_vec();
        H5O__fsinfo_encode_into(&fsinfo, &mut fsinfo_image).unwrap();
        assert_eq!(&fsinfo_image[..5], b"stale");
        assert_eq!(
            fsinfo_image.len(),
            5 + H5O__fsinfo_image_len(&fsinfo).unwrap()
        );

        let mut name_image = b"stale".to_vec();
        H5O__name_encode_into("alpha", &mut name_image).unwrap();
        assert_eq!(name_image, b"stalealpha\0".to_vec());
    }

    #[test]
    fn caller_owned_object_encoders_preserve_output_on_errors() {
        let stale = b"stale".to_vec();

        let mut bad_chunk = b"OCHK".to_vec();
        bad_chunk.extend_from_slice(&[0, 0, 0, 0]);
        bad_chunk.extend_from_slice(&0u32.to_le_bytes());
        let mut chunk_image = stale.clone();
        assert!(H5O__cache_chk_serialize_into(
            &ObjectHeaderChunkImage {
                is_v2_continuation: true,
                raw: bad_chunk,
            },
            &mut chunk_image,
        )
        .is_err());
        assert_eq!(chunk_image, stale);

        let mut bad_fsinfo = FsInfoMessage {
            version: 1,
            free_space_strategy: 2,
            persist: true,
            threshold: 8,
            page_size: 4096,
            pgend_meta_thres: 0,
            eoa_pre_fsm_fsalloc: u64::MAX,
            fs_addr: vec![u64::MAX],
            sizeof_addr: 8,
            sizeof_size: 8,
        };
        let mut fsinfo_image = stale.clone();
        assert!(H5O__fsinfo_encode_into(&bad_fsinfo, &mut fsinfo_image).is_err());
        assert_eq!(fsinfo_image, stale);

        bad_fsinfo.persist = false;
        bad_fsinfo.fs_addr.clear();
        bad_fsinfo.threshold = 256;
        bad_fsinfo.page_size = 1;
        bad_fsinfo.sizeof_size = 1;
        let mut fsinfo_image = stale.clone();
        assert!(H5O__fsinfo_encode_into(&bad_fsinfo, &mut fsinfo_image).is_err());
        assert_eq!(fsinfo_image, stale);
    }

    #[test]
    fn object_message_decode_rejects_truncated_header() {
        let err = H5O_msg_decode(&[1, 0, 0, 0]).unwrap_err();
        assert!(matches!(err, Error::InvalidFormat(_)));
    }

    #[test]
    fn object_layout_decode_rejects_malformed_payload() {
        let err = H5O__layout_decode(&[]).unwrap_err();
        assert!(matches!(err, Error::InvalidFormat(_)));
        let err = H5O__layout_decode(&[9]).unwrap_err();
        assert!(matches!(err, Error::InvalidFormat(_)));
        let err = H5O__layout_decode(&[4, 1, 2, 3]).unwrap_err();
        assert!(matches!(err, Error::InvalidFormat(_)));
    }

    #[test]
    fn object_layout_decode_parses_and_preserves_raw_payload() {
        let raw = [4, 0, 3, 0, b'a', b'b', b'c'];
        let decoded = H5O__layout_decode(&raw).unwrap();
        assert_eq!(decoded.message.version, 4);
        assert_eq!(decoded.message.compact_data.as_deref(), Some(&b"abc"[..]));
        assert_eq!(decoded.raw, raw);
        assert_eq!(H5O__layout_encode(&decoded).unwrap(), raw);
    }

    #[test]
    fn object_dataspace_decode_parses_and_preserves_raw_payload() {
        let raw = [2, 1, 0, 1, 7, 0, 0, 0, 0, 0, 0, 0];
        let decoded = H5O__sdspace_decode(&raw).unwrap();
        assert_eq!(decoded.message.version, 2);
        assert_eq!(decoded.message.ndims, 1);
        assert_eq!(decoded.message.dims, vec![7]);
        assert_eq!(decoded.raw, raw);

        let err = H5O__sdspace_decode(&[7, 0, 0, 0]).unwrap_err();
        assert!(matches!(err, Error::InvalidFormat(_)));
    }

    #[test]
    fn object_prefix_deserialize_validates_header_images() {
        let mut v1 = vec![0; 16];
        v1[0] = 1;
        let decoded_v1 = H5O__prefix_deserialize(&v1).unwrap();
        assert_eq!(decoded_v1.version, 1);
        assert_eq!(decoded_v1.raw, v1);
        assert!(H5O__prefix_deserialize(&[3, 0, 0, 0]).is_err());

        let mut v2 = b"OHDR".to_vec();
        v2.push(2);
        v2.push(0);
        v2.push(0);
        let checksum = checksum_metadata(&v2);
        v2.extend_from_slice(&checksum.to_le_bytes());
        let decoded_v2 = H5O__prefix_deserialize(&v2).unwrap();
        assert_eq!(decoded_v2.version, 2);
        assert_eq!(decoded_v2.raw, v2);

        let mut bad_flags = b"OHDR".to_vec();
        bad_flags.extend_from_slice(&[2, 0x40, 0]);
        bad_flags.extend_from_slice(&checksum_metadata(&bad_flags).to_le_bytes());
        assert!(H5O__prefix_deserialize(&bad_flags).is_err());

        let mut bad_checksum = b"OHDR".to_vec();
        bad_checksum.extend_from_slice(&[2, 0, 0, 0, 0, 0, 0]);
        assert!(H5O__prefix_deserialize(&bad_checksum).is_err());
    }

    #[test]
    fn object_chunk_deserialize_validates_v2_checksum() {
        let v1_raw = vec![1, 2, 3];
        let v1_chunk = H5O__chunk_deserialize(&v1_raw).unwrap();
        assert!(!v1_chunk.is_v2_continuation);
        assert_eq!(v1_chunk.raw, v1_raw);
        let v1_cached = H5O__cache_chk_deserialize(&v1_raw).unwrap();
        assert_eq!(H5O__cache_chk_image_len(&v1_cached), v1_raw.len());
        let mut v1_cached_image = Vec::new();
        H5O__cache_chk_serialize_into(&v1_cached, &mut v1_cached_image).unwrap();
        assert_eq!(v1_cached_image, v1_raw);

        let mut v2 = b"OCHK".to_vec();
        v2.extend_from_slice(&[0, 0, 0, 0]);
        let checksum = checksum_metadata(&v2);
        v2.extend_from_slice(&checksum.to_le_bytes());
        let v2_chunk = H5O__chunk_deserialize(&v2).unwrap();
        assert!(v2_chunk.is_v2_continuation);
        assert_eq!(v2_chunk.raw, v2);
        let v2_cached = H5O__cache_chk_deserialize(&v2).unwrap();
        assert_eq!(H5O__cache_chk_image_len(&v2_cached), v2.len());
        let mut v2_cached_image = Vec::new();
        H5O__cache_chk_serialize_into(&v2_cached, &mut v2_cached_image).unwrap();
        assert_eq!(v2_cached_image, v2);

        let mut bad = b"OCHK".to_vec();
        bad.extend_from_slice(&[0, 0, 0, 0]);
        bad.extend_from_slice(&0u32.to_le_bytes());
        assert!(H5O__chunk_deserialize(&bad).is_err());
        assert!(H5O__cache_chk_deserialize(&bad).is_err());
        let mut bad_image = Vec::new();
        assert!(H5O__cache_chk_serialize_into(
            &ObjectHeaderChunkImage {
                is_v2_continuation: true,
                raw: bad,
            },
            &mut bad_image,
        )
        .is_err());
    }

    #[test]
    fn object_aux_decoders_reject_truncated_payloads() {
        assert!(matches!(
            H5O__fsinfo_decode(&[0; 10]).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        assert!(matches!(
            H5O__sdspace_decode(&[0; 7]).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        assert!(matches!(
            H5O__mtime_decode(&[0; 7]).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        assert!(matches!(
            H5O__mtime_new_decode(&[0; 7]).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        assert!(matches!(
            H5O__refcount_decode(&[0; 3]).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        assert!(matches!(
            H5O__ginfo_decode(&[0]).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        assert!(matches!(
            H5O__btreek_decode(&[0; 6]).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        assert!(matches!(
            H5O__cont_decode(&[0; 15]).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        assert!(matches!(
            H5O__fill_new_decode(&[]).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        assert!(matches!(
            H5O__fill_old_decode(&[2, 0, 0, 0, 0xaa]).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        assert!(matches!(
            H5O__pline_decode(&[2, 1]).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        assert!(matches!(
            H5O__shmesg_decode(&[0, 64, 0, 0]).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        assert!(matches!(
            H5O_SHARED_DECODE(&[3, 1, 0, 1]).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        assert!(matches!(
            H5O__efl_decode(&[1, 0, 0, 0, 0, 0, 0, 0]).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        assert!(matches!(
            H5O__attr_decode(&[4, 0, 0, 0, 0, 0]).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        assert!(matches!(
            H5O__link_decode(&[1, 0, 1, b'x']).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        assert!(matches!(
            H5O__linfo_decode(&[0, 0]).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        assert!(matches!(
            H5O__ainfo_decode(&[0, 0]).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        assert!(matches!(
            H5O__mdci_decode(&[1, 0, 0, 0]).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        assert!(matches!(
            H5O__mdci_decode_with_sizes(&[0, 1, 0, 0], 2, 2).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        assert!(matches!(
            H5O__drvinfo_decode(&[0, b's', b'e', b'c']).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        assert!(matches!(
            H5O__drvinfo_decode(&[0, b's', b'e', b'c', b'2', 0, 0, 0, 0, 0, 0]).unwrap_err(),
            Error::InvalidFormat(_)
        ));
    }

    #[test]
    fn object_aux_decoders_accept_complete_payloads() {
        let mut fsinfo_bytes = vec![1, 2, 0];
        fsinfo_bytes.extend_from_slice(&8u64.to_le_bytes());
        fsinfo_bytes.extend_from_slice(&4096u64.to_le_bytes());
        fsinfo_bytes.extend_from_slice(&0u16.to_le_bytes());
        fsinfo_bytes.extend_from_slice(&u64::MAX.to_le_bytes());
        fsinfo_bytes.push(0xaa);
        let fsinfo = H5O__fsinfo_decode(&fsinfo_bytes).unwrap();
        assert_eq!(fsinfo.version, 1);
        assert_eq!(fsinfo.free_space_strategy, 2);
        assert!(!fsinfo.persist);
        assert_eq!(fsinfo.threshold, 8);
        assert_eq!(fsinfo.page_size, 4096);
        assert_eq!(fsinfo.eoa_pre_fsm_fsalloc, u64::MAX);
        assert_eq!(H5O__fsinfo_image_len(&fsinfo).unwrap(), 29);
        let mut fsinfo_image = Vec::new();
        H5O__fsinfo_encode_into(&fsinfo, &mut fsinfo_image).unwrap();
        assert_eq!(fsinfo_image, fsinfo_bytes[..29].to_vec());

        let sdspace = H5O__sdspace_decode(&[2, 1, 0, 1, 16, 0, 0, 0, 0, 0, 0, 0]).unwrap();
        assert_eq!(sdspace.message.dims, vec![16]);
        let mdci = H5O__mdci_decode_with_sizes(&[0, 0x34, 0x12, 0x09, 0], 2, 2).unwrap();
        assert_eq!(mdci.addr, 0x1234);
        assert_eq!(mdci.size, 9);
        assert_eq!(H5O__mdci_size(&mdci), 5);
        assert_eq!(
            H5O__mdci_encode(&mdci).unwrap(),
            vec![0, 0x34, 0x12, 0x09, 0]
        );
        let drvinfo = H5O__drvinfo_decode(&[
            0, b's', b'e', b'c', b'2', 0, 0, 0, 0, 3, 0, b'a', b'b', b'c', 0xaa,
        ])
        .unwrap();
        assert_eq!(&drvinfo.name[..4], b"sec2");
        assert_eq!(drvinfo.data, b"abc");
        assert_eq!(H5O__drvinfo_size(&drvinfo).unwrap(), 14);
        assert_eq!(
            H5O__drvinfo_encode(&drvinfo).unwrap(),
            vec![0, b's', b'e', b'c', b'2', 0, 0, 0, 0, 3, 0, b'a', b'b', b'c']
        );
        assert_eq!(H5O__mtime_decode(&9u64.to_le_bytes()).unwrap(), 9);
        assert_eq!(
            H5O__mtime_new_decode(&[1, 0xaa, 0xbb, 0xcc, 9, 0, 0, 0]).unwrap(),
            9
        );
        assert_eq!(
            H5O__mtime_new_encode(9).unwrap(),
            vec![1, 0, 0, 0, 9, 0, 0, 0]
        );
        assert!(H5O__mtime_new_encode(u64::from(u32::MAX) + 1).is_err());
        assert_eq!(H5O__refcount_decode(&[7, 0, 0, 0, 0xaa]).unwrap(), 7);
        let mut refcount_image = Vec::new();
        H5O__refcount_encode_into(7, &mut refcount_image);
        assert_eq!(refcount_image, vec![7, 0, 0, 0]);
        let mut header = ObjectHeaderState {
            refcount: 1,
            ..ObjectHeaderState::default()
        };
        H5Olink_checked(&mut header, 1).unwrap();
        assert_eq!(header.refcount, 2);
        assert!(H5Olink_checked(&mut header, -3).is_err());

        let mut shared = SharedMessageTable::default();
        H5O__shared_link_adj_checked(&mut shared, 10, 2).unwrap();
        assert_eq!(shared.refs.get(&10), Some(&2));
        assert!(H5O__shared_link_adj_checked(&mut shared, 10, -3).is_err());

        let mut deleted_mdci = mdci.clone();
        H5O__mdci_delete_checked(&mut deleted_mdci).unwrap();
        assert_eq!(deleted_mdci.addr, 0xffff);
        assert_eq!(deleted_mdci.size, 0);
        deleted_mdci.sizeof_addr = 9;
        assert!(H5O__mdci_delete_checked(&mut deleted_mdci).is_err());
        let ginfo = H5O__ginfo_decode(&[0, 0x03, 8, 0, 6, 0, 5, 0, 12, 0, 0xaa]).unwrap();
        assert_eq!(ginfo.max_compact, Some(8));
        assert_eq!(ginfo.min_dense, Some(6));
        assert_eq!(ginfo.estimated_entries, Some(5));
        assert_eq!(ginfo.estimated_name_len, Some(12));
        assert_eq!(
            H5O__ginfo_encode(&ginfo).unwrap(),
            vec![0, 0x03, 8, 0, 6, 0, 5, 0, 12, 0]
        );
        assert!(H5O__ginfo_encode(&GroupInfoMessage {
            max_compact: Some(8),
            ..GroupInfoMessage::default()
        })
        .is_err());
        let btreek = H5O__btreek_decode(&[0, 32, 0, 16, 0, 4, 0, 0xaa]).unwrap();
        assert_eq!(btreek.indexed_storage_internal_k, 32);
        assert_eq!(btreek.group_internal_k, 16);
        assert_eq!(btreek.group_leaf_k, 4);
        assert_eq!(
            H5O__btreek_encode(&btreek).unwrap(),
            vec![0, 32, 0, 16, 0, 4, 0]
        );
        assert_eq!(H5O__name_decode_ref(b"alpha\0ignored").unwrap(), "alpha");
        assert_eq!(H5O__name_decode_ref(b"alpha").unwrap(), "alpha");
        assert!(H5O__name_decode_ref(&[0xff, 0]).is_err());
        let mut name_image = Vec::new();
        H5O__name_encode_into("alpha", &mut name_image).unwrap();
        assert_eq!(name_image, b"alpha\0".to_vec());
        assert_eq!(H5O__name_size("alpha"), 6);

        let mut link_bytes = vec![1, 0, 1, b'x'];
        link_bytes.extend_from_slice(&64u64.to_le_bytes());
        link_bytes.push(0xaa);
        let link = H5O__link_decode(&link_bytes).unwrap();
        assert_eq!(link.message.name, "x");
        assert_eq!(link.message.hard_link_addr, Some(64));
        assert_eq!(H5O__link_size(&link), link_bytes.len());
        assert!(H5O__link_debug(&link).contains("name=x"));

        let mut linfo_bytes = vec![0, 0];
        linfo_bytes.extend_from_slice(&16u64.to_le_bytes());
        linfo_bytes.extend_from_slice(&32u64.to_le_bytes());
        linfo_bytes.push(0xaa);
        let linfo = H5O__linfo_decode(&linfo_bytes).unwrap();
        assert_eq!(linfo.message.fractal_heap_addr, 16);
        assert_eq!(linfo.message.name_btree_addr, 32);
        assert_eq!(H5O__linfo_size(&linfo), linfo_bytes.len());
        assert!(H5O__linfo_debug(&linfo).contains("heap=0x10"));

        let mut ainfo_bytes = vec![0, 0];
        ainfo_bytes.extend_from_slice(&48u64.to_le_bytes());
        ainfo_bytes.extend_from_slice(&64u64.to_le_bytes());
        ainfo_bytes.push(0xaa);
        let ainfo = H5O__ainfo_decode(&ainfo_bytes).unwrap();
        assert_eq!(ainfo.message.fractal_heap_addr, 48);
        assert_eq!(ainfo.message.name_btree_addr, 64);
        assert_eq!(H5O__ainfo_size(&ainfo), ainfo_bytes.len());
        assert!(H5O__ainfo_debug(&ainfo).contains("name_btree=0x40"));

        let fill = H5O__fill_new_decode(&[2, 0, 0, 1, 2, 0, 0, 0, 0xaa, 0xbb]).unwrap();
        assert_eq!(fill.version, 2);
        assert!(fill.defined);
        assert_eq!(fill.value.as_deref(), Some(&[0xaa, 0xbb][..]));
        assert_eq!(H5O__fill_new_size(&fill), 10);
        assert_eq!(H5O__fill_new_size_checked(&fill).unwrap(), 10);
        assert!(H5O__fill_new_size_checked(&FillValueMessage {
            version: 99,
            ..fill.clone()
        })
        .is_err());

        let fill_old = H5O__fill_old_decode(&[2, 0, 0, 0, 0xcc, 0xdd, 0xaa]).unwrap();
        assert_eq!(fill_old.version, 0);
        assert!(fill_old.defined);
        assert_eq!(fill_old.value.as_deref(), Some(&[0xcc, 0xdd][..]));
        assert_eq!(H5O__fill_old_size(&fill_old).unwrap(), 6);
        assert_eq!(
            H5O__fill_old_encode(&fill_old).unwrap(),
            vec![2, 0, 0, 0, 0xcc, 0xdd]
        );
        let fill_v3 = H5O__fill_new_decode(&[3, 0x29, 3, 0, 0, 0, 1, 2, 3]).unwrap();
        assert_eq!(fill_v3.version, 3);
        assert_eq!(fill_v3.alloc_time, 1);
        assert_eq!(fill_v3.fill_time, 2);
        assert!(fill_v3.defined);
        assert_eq!(fill_v3.value.as_deref(), Some(&[1, 2, 3][..]));
        assert!(H5O__fill_new_decode(&[3, 0x30, 0, 0, 0, 0]).is_err());
        let fill_copy = H5O__fill_copy(&fill_v3);
        assert_eq!(fill_copy, fill_v3);
        let mut fill_reset = fill_copy;
        H5O__fill_reset(&mut fill_reset);
        assert_eq!(fill_reset.version, 3);
        assert_eq!(fill_reset.alloc_time, 2);
        assert_eq!(fill_reset.fill_time, 2);
        assert!(!fill_reset.defined);
        assert!(fill_reset.value.is_none());

        let pline = H5O__pline_decode(&[2, 1, 1, 0, 0, 0, 1, 0, 6, 0, 0, 0]).unwrap();
        assert_eq!(pline.version, 2);
        assert_eq!(pline.filters.len(), 1);
        assert_eq!(pline.filters[0].id, 1);
        assert_eq!(pline.filters[0].client_data, vec![6]);
        assert_eq!(H5O__pline_size(&pline), 12);
        assert_eq!(H5O__pline_size_checked(&pline).unwrap(), 12);
        assert_eq!(
            H5O__pline_debug(&pline),
            "pline(version=2, filters=1, ids=[1])"
        );

        assert!(H5O__pline_size_checked(&FilterPipelineMessage {
            version: 7,
            filters: Vec::new(),
        })
        .is_err());
        assert!(align8_len_checked(usize::MAX, "test filter name").is_err());

        let dtype = H5O__dtype_decode_helper(&[0x10, 0, 0, 0, 4, 0, 0, 0, 0, 0, 32, 0]).unwrap();
        assert_eq!(dtype.version, 1);
        assert_eq!(dtype.size, 4);
        assert_eq!(H5O__dtype_size(&dtype).unwrap(), 12);
        assert_eq!(
            H5O__dtype_encode(&dtype).unwrap(),
            vec![0x10, 0, 0, 0, 4, 0, 0, 0, 0, 0, 32, 0]
        );
        assert!(H5O__dtype_can_share(&dtype));
        assert!(H5O__dtype_debug(&dtype).contains("FixedPoint"));

        let shmesg = H5O__shmesg_decode(&[0, 64, 0, 0, 0, 0, 0, 0, 0, 1, 0xaa]).unwrap();
        assert_eq!(shmesg.version, 0);
        assert_eq!(shmesg.table_addr, 64);
        assert_eq!(shmesg.nindexes, 1);
        assert_eq!(H5O__shmesg_size(&shmesg), 10);
        assert_eq!(
            H5O__shmesg_encode(&shmesg).unwrap(),
            vec![0, 64, 0, 0, 0, 0, 0, 0, 0, 1]
        );
        let shmesg_4 = H5O__shmesg_decode_with_addr_size(&[0, 64, 0, 0, 0, 1, 0xaa], 4).unwrap();
        assert_eq!(shmesg_4.table_addr, 64);
        assert_eq!(H5O__shmesg_size_with_addr_size(&shmesg_4, 4).unwrap(), 6);
        assert_eq!(
            H5O__shmesg_encode_with_addr_size(&shmesg_4, 4).unwrap(),
            vec![0, 64, 0, 0, 0, 1]
        );
        assert!(H5O__shmesg_encode_with_addr_size(
            &SharedMessageTableInfo {
                table_addr: u64::from(u32::MAX),
                ..shmesg_4.clone()
            },
            4
        )
        .is_err());
        assert!(H5O__shmesg_encode_with_addr_size(
            &SharedMessageTableInfo {
                table_addr: u64::from(u32::MAX) + 1,
                ..shmesg_4.clone()
            },
            4
        )
        .is_err());
        assert!(H5O__shmesg_encode(&SharedMessageTableInfo {
            nindexes: 0,
            ..shmesg.clone()
        })
        .is_err());

        let shared =
            H5O_SHARED_DECODE(&[3, 1, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0xaa])
                .unwrap();
        assert_eq!(
            shared,
            SharedMessageReference::V3Sohm { heap_id: [0x5a; 8] }
        );
        assert_eq!(H5O_SHARED_SIZE(&shared).unwrap(), 10);
        assert_eq!(
            H5O_SHARED_ENCODE(&shared).unwrap(),
            vec![3, 1, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a]
        );
        assert!(H5O_SHARED_DEBUG(&shared).contains("sohm"));

        let shared_v1 = SharedMessageReference::V1 {
            message_type: 3,
            index: 0x1234,
            addr: 0x5678,
        };
        let shared_v1_image = H5O_SHARED_ENCODE_WITH_CONTEXT(&shared_v1, 4, 4).unwrap();
        assert_eq!(shared_v1_image.len(), 16);
        assert_eq!(
            H5O_SHARED_DECODE_WITH_CONTEXT(&shared_v1_image, 4, 4).unwrap(),
            shared_v1
        );
        assert!(H5O_SHARED_ENCODE_WITH_CONTEXT(
            &SharedMessageReference::V2 {
                message_type: 3,
                addr: u64::MAX
            },
            8,
            8
        )
        .is_err());

        let mut efl_bytes = vec![1, 1, 0, 0];
        efl_bytes.extend_from_slice(&1u16.to_le_bytes());
        efl_bytes.extend_from_slice(&1u16.to_le_bytes());
        efl_bytes.extend_from_slice(&16u64.to_le_bytes());
        efl_bytes.extend_from_slice(&4u64.to_le_bytes());
        efl_bytes.extend_from_slice(&8u64.to_le_bytes());
        efl_bytes.extend_from_slice(&12u64.to_le_bytes());
        efl_bytes.push(0xaa);
        let efl = H5O__efl_decode(&efl_bytes).unwrap();
        assert_eq!(efl.version, 1);
        assert_eq!(efl.heap_addr, 16);
        assert_eq!(efl.entries.len(), 1);
        assert_eq!(efl.entries[0].name_offset, 4);
        assert_eq!(efl.entries[0].file_offset, 8);
        assert_eq!(efl.entries[0].size, 12);
        assert_eq!(H5O_efl_total_size(&efl), 12);
        assert_eq!(H5O__efl_size(&efl).unwrap(), 40);
        let mut expected_efl = vec![1, 0, 0, 0];
        expected_efl.extend_from_slice(&1u16.to_le_bytes());
        expected_efl.extend_from_slice(&1u16.to_le_bytes());
        expected_efl.extend_from_slice(&16u64.to_le_bytes());
        expected_efl.extend_from_slice(&4u64.to_le_bytes());
        expected_efl.extend_from_slice(&8u64.to_le_bytes());
        expected_efl.extend_from_slice(&12u64.to_le_bytes());
        assert_eq!(H5O__efl_encode(&efl).unwrap(), expected_efl);
        let efl_4byte = H5O__efl_encode_with_sizes(&efl, 4, 4).unwrap();
        assert_eq!(H5O__efl_size_with_sizes(&efl, 4, 4).unwrap(), 24);
        assert_eq!(efl_4byte.len(), 24);
        assert_eq!(H5O__efl_decode_with_sizes(&efl_4byte, 4, 4).unwrap(), efl);
        assert!(H5O__efl_encode_with_sizes(
            &ExternalFileListMessage {
                heap_addr: u64::MAX,
                ..efl.clone()
            },
            8,
            8
        )
        .is_err());
        assert!(H5O__efl_encode(&ExternalFileListMessage {
            allocated_slots: 0,
            ..efl.clone()
        })
        .is_err());
        assert!(H5O__efl_debug(&efl).contains("entries=1"));

        let attr_dtype = vec![0x10, 0, 0, 0, 4, 0, 0, 0, 0, 0, 32, 0];
        let attr_dataspace = vec![2, 0, 0, 0];
        let mut attr_bytes = vec![3, 0];
        attr_bytes.extend_from_slice(&2u16.to_le_bytes());
        let attr_dtype_len = u16::try_from(attr_dtype.len()).unwrap();
        let attr_dataspace_len = u16::try_from(attr_dataspace.len()).unwrap();
        attr_bytes.extend_from_slice(&attr_dtype_len.to_le_bytes());
        attr_bytes.extend_from_slice(&attr_dataspace_len.to_le_bytes());
        attr_bytes.push(0);
        attr_bytes.extend_from_slice(b"x\0");
        attr_bytes.extend_from_slice(&attr_dtype);
        attr_bytes.extend_from_slice(&attr_dataspace);
        attr_bytes.extend_from_slice(&[1, 2, 3, 4]);
        let attr = H5O__attr_decode(&attr_bytes).unwrap();
        assert_eq!(attr.message.name, "x");
        assert_eq!(attr.message.data, vec![1, 2, 3, 4]);
        assert_eq!(H5O__attr_size(&attr), attr_bytes.len());
        let mut attr_debug = String::new();
        H5O__attr_debug_fmt(&attr, &mut attr_debug).unwrap();
        assert!(attr_debug.contains("name=x"));

        let mut cont = Vec::new();
        cont.extend_from_slice(&24u64.to_le_bytes());
        cont.extend_from_slice(&32u64.to_le_bytes());
        assert_eq!(H5O__cont_decode(&cont).unwrap(), (24, 32));
    }

    #[test]
    fn symbol_table_and_continuation_messages_use_configured_widths() {
        let stab = SymbolTableMessage {
            btree_addr: 0x1234,
            heap_addr: 0x5678,
        };
        let stab_image = H5O__stab_encode_with_size(&stab, 4).unwrap();
        assert_eq!(stab_image.len(), 8);
        assert_eq!(H5O__stab_size_with_size(&stab, 4).unwrap(), 8);
        assert_eq!(H5O__stab_decode_with_size(&stab_image, 4).unwrap(), stab);
        assert!(H5O__stab_decode_with_size(&stab_image[..7], 4).is_err());
        assert!(H5O__stab_encode_with_size(
            &SymbolTableMessage {
                btree_addr: u64::MAX,
                heap_addr: 0x5678,
            },
            8,
        )
        .is_err());

        let cont_image = H5O__cont_encode_with_sizes(0x1234, 0x5678, 4, 4).unwrap();
        assert_eq!(cont_image.len(), 8);
        assert_eq!(
            H5O__cont_decode_with_sizes(&cont_image, 4, 4).unwrap(),
            (0x1234, 0x5678)
        );
        assert_eq!(H5O__cont_size_with_sizes(0x1234, 0x5678, 4, 4).unwrap(), 8);
        assert!(H5O__cont_decode_with_sizes(&cont_image[..7], 4, 4).is_err());
        assert!(H5O__cont_encode_with_sizes(0x1234, 0, 4, 4).is_err());
    }

    #[test]
    fn attr_rename_rejects_missing_name_terminator() {
        let mut header = ObjectHeaderState {
            messages: vec![ObjectMessage {
                msg_type: 0x000c,
                data: b"old-name-without-terminator".to_vec(),
                ..ObjectMessage::default()
            }],
            ..ObjectHeaderState::default()
        };

        let err = H5O__attr_rename_checked(&mut header, "old", "new").unwrap_err();
        assert!(matches!(err, Error::InvalidFormat(_)));
        assert!(!H5O__attr_rename(&mut header, "old", "new"));
    }

    #[test]
    fn attr_rename_preserves_value_payload() {
        let mut header = ObjectHeaderState::default();
        H5O__attr_create(&mut header, "old", b"value");

        assert!(H5O__attr_rename_checked(&mut header, "old", "new").unwrap());
        assert_eq!(
            H5O__attr_open_by_name_ref(&header, "new").unwrap().data,
            b"new\0value"
        );
        assert!(H5O__attr_open_by_name_ref(&header, "old").is_none());
    }

    #[test]
    fn attr_lookup_write_and_remove_use_exact_names() {
        let mut header = ObjectHeaderState::default();
        H5O__attr_create(&mut header, "a", b"one");
        H5O__attr_create(&mut header, "ab", b"two");

        H5O__attr_write(&mut header, "a", b"uno");
        assert_eq!(
            H5O__attr_open_by_name_ref(&header, "a").unwrap().data,
            b"a\0uno"
        );
        assert_eq!(
            H5O__attr_open_by_name_ref(&header, "ab").unwrap().data,
            b"ab\0two"
        );

        assert!(H5O__attr_remove(&mut header, "a"));
        assert!(H5O__attr_open_by_name_ref(&header, "a").is_none());
        assert!(H5O__attr_open_by_name_ref(&header, "ab").is_some());
        assert_eq!(H5O__attr_count_real(&header), 1);
    }

    #[test]
    fn borrowed_object_apis_return_existing_storage() {
        let mut header = ObjectHeaderState::default();
        H5Oset_comment(&mut header, "comment");
        H5O__attr_create(&mut header, "b", b"two");
        H5O__attr_create(&mut header, "a", b"one");

        let comment = H5Oget_comment_ref(&header).unwrap();
        assert_eq!(comment, "comment");
        assert_eq!(comment.as_ptr(), header.comment.as_ref().unwrap().as_ptr());

        let attr = H5O__attr_open_by_name_ref(&header, "a").unwrap();
        assert_eq!(attr.data.as_slice(), b"a\0one");
        assert_eq!(attr.data.as_ptr(), header.messages[1].data.as_ptr());

        let attrs = H5O__attr_iterate_refs(&header);
        assert_eq!(
            attrs
                .iter()
                .map(|message| message.creation_index)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );

        let mut objects = BTreeMap::new();
        objects.insert("root".to_string(), header);
        let visited = H5O__visit_refs(&objects);
        assert_eq!(visited, vec!["root"]);
        assert_eq!(visited[0].as_ptr(), objects.keys().next().unwrap().as_ptr());
        assert_eq!(
            H5Oget_comment_by_name_ref(&objects, "root"),
            Some("comment")
        );
    }

    #[test]
    fn object_allocation_merges_null_gaps_and_reindexes() {
        let mut header = ObjectHeaderState::default();
        H5O__add_gap(&mut header, 2);
        H5O__add_gap(&mut header, 3);
        H5O__chunk_add(
            &mut header,
            ObjectMessage {
                msg_type: 7,
                data: vec![1],
                ..ObjectMessage::default()
            },
        );
        H5O__add_gap(&mut header, 4);
        H5O__condense_header(&mut header);

        assert_eq!(header.messages.len(), 2);
        assert_eq!(header.messages[0].msg_type, 7);
        assert_eq!(header.messages[1].msg_type, 0);
        assert_eq!(header.messages[1].data.len(), 9);
        assert_eq!(
            header
                .messages
                .iter()
                .map(|message| message.creation_index)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
    }

    #[test]
    fn object_copy_and_flush_normalize_messages() {
        let mut header = ObjectHeaderState {
            addr: 42,
            refcount: 0,
            flush_disabled: true,
            messages: vec![
                ObjectMessage {
                    msg_type: 0x0003,
                    flags: 0,
                    creation_index: 9,
                    data: vec![1, 2, 3],
                    shared: true,
                },
                ObjectMessage {
                    msg_type: 8,
                    flags: 0x03,
                    creation_index: 3,
                    data: Vec::new(),
                    shared: false,
                },
            ],
            ..ObjectHeaderState::default()
        };

        H5O_flush(&mut header);
        assert!(!header.flush_disabled);
        assert_eq!(header.messages.len(), 1);
        assert_eq!(header.messages[0].flags & 0x03, 0x03);

        let copied = H5O__copy(&header);
        assert_eq!(copied.refcount, 1);
        assert_eq!(copied.messages[0].data, vec![1, 2, 3]);
        assert!(H5O__copy_search_comm_dt_check(&copied));
        assert_eq!(H5O__copy_search_comm_dt_cb(&copied), Some(42));
    }

    #[test]
    fn object_lifecycle_updates_info_and_touch_state() {
        let mut header = H5O_create_ohdr(77);
        assert_eq!(header.refcount, 1);

        H5O_apply_ohdr(&mut header, |header| {
            header.refcount = 0;
            header.messages.push(ObjectMessage {
                msg_type: 0x0012,
                data: H5O__mtime_new_encode(123).unwrap(),
                ..ObjectMessage::default()
            });
            header.flush_disabled = true;
        });
        assert_eq!(header.refcount, 1);
        assert!(!header.flush_disabled);
        assert_eq!(H5O_get_oh_mtime(&header), 123);

        let info = H5O_get_info(&header);
        assert_eq!(info.addr, 77);
        assert_eq!(info.msg_count, 1);
        assert!(info.has_checksum);

        let plist = H5O_get_create_plist(&header);
        assert_eq!(plist.get("addr").map(String::as_str), Some("77"));
        assert_eq!(plist.get("messages").map(String::as_str), Some("1"));

        H5O_touch(&mut header);
        assert_eq!(header.messages[0].flags & 0x01, 0x01);
        H5O_delete(&mut header);
        assert!(H5O_bogus_oh(&header));
        assert_eq!(H5O_get_info(&header).msg_count, 0);
    }

    #[test]
    fn object_link_info_and_efl_copy_file_normalize_deleted_storage() {
        let link = LinkObjectMessage {
            message: LinkMessage {
                name: "child".to_string(),
                link_type: crate::format::messages::link::LinkType::Hard,
                creation_order: Some(3),
                char_encoding: 0,
                hard_link_addr: Some(u64::MAX),
                soft_link_target: None,
                external_link: None,
            },
            raw_size: 0,
        };
        let copied_link = H5O__link_copy_file(&link);
        assert_eq!(copied_link.message.hard_link_addr, None);
        assert!(H5O__link_size(&copied_link) >= "child".len());

        let linfo = LinkInfoObjectMessage {
            message: LinkInfoMessage {
                version: 0,
                flags: 0x03,
                max_creation_index: None,
                fractal_heap_addr: u64::MAX,
                name_btree_addr: 12,
                corder_btree_addr: Some(13),
            },
            raw_size: 30,
        };
        let copied_linfo = H5O__linfo_post_copy_file(&linfo);
        assert_eq!(copied_linfo.message.name_btree_addr, u64::MAX);
        assert_eq!(copied_linfo.message.corder_btree_addr, None);
        assert_eq!(copied_linfo.message.flags & 0x03, 0);

        let mut efl = ExternalFileListMessage {
            version: 1,
            allocated_slots: 2,
            heap_addr: 20,
            entries: vec![
                ExternalFileListEntry {
                    name_offset: 1,
                    file_offset: 0,
                    size: 0,
                },
                ExternalFileListEntry {
                    name_offset: 2,
                    file_offset: 4,
                    size: 8,
                },
            ],
        };
        let copied_efl = H5O__efl_copy_file(&efl);
        assert_eq!(copied_efl.entries.len(), 1);
        assert_eq!(H5O_efl_total_size(&copied_efl), 8);
        H5O__efl_reset(&mut efl);
        assert_eq!(efl.allocated_slots, 0);
        assert_eq!(efl.heap_addr, u64::MAX);
    }
}
