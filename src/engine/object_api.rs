use std::collections::BTreeMap;

use crate::error::{Error, Result};
use crate::format::checksum::checksum_metadata;
use crate::format::messages::attribute::AttributeMessage;
use crate::format::messages::attribute_info::AttributeInfoMessage;
use crate::format::messages::datatype::DatatypeMessage;
use crate::format::messages::fill_value::FillValueMessage;
use crate::format::messages::filter_pipeline::FilterPipelineMessage;
use crate::format::messages::link::LinkMessage;
use crate::format::messages::link_info::LinkInfoMessage;
use crate::format::object_header::{
    HDR_ATTR_STORE_PHASE_CHANGE, HDR_CHUNK0_SIZE_MASK, HDR_STORE_TIMES, HDR_V2_KNOWN_FLAGS,
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
    pub page_size: Option<u64>,
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

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LayoutMessage {
    pub version: u8,
    pub raw: Vec<u8>,
}

#[allow(non_snake_case)]
pub fn H5O__shared_link_adj(table: &mut SharedMessageTable, addr: u64, delta: isize) {
    let entry = table.refs.entry(addr).or_default();
    if delta.is_negative() {
        *entry = entry.saturating_sub(delta.unsigned_abs());
    } else {
        let delta = usize::try_from(delta).unwrap_or(usize::MAX);
        *entry = entry.saturating_add(delta);
    }
    if *entry == 0 {
        table.refs.remove(&addr);
    }
}

#[allow(non_snake_case)]
pub fn H5O_set_shared(message: &mut ObjectMessage, shared: bool) {
    message.shared = shared;
}

#[allow(non_snake_case)]
pub fn H5O__shared_delete(table: &mut SharedMessageTable, addr: u64) {
    table.refs.remove(&addr);
}

#[allow(non_snake_case)]
pub fn H5O__shared_copy_file(table: &SharedMessageTable) -> SharedMessageTable {
    table.clone()
}

#[allow(non_snake_case)]
pub fn H5O__shared_debug(table: &SharedMessageTable) -> String {
    format!("shared_messages={}", table.refs.len())
}

#[allow(non_snake_case)]
pub fn H5O__group_isa(header: &ObjectHeaderState) -> bool {
    header
        .messages
        .iter()
        .any(|msg| matches!(msg.msg_type, 0x0011 | 0x000A | 0x000B))
}

#[allow(non_snake_case)]
pub fn H5O__group_get_oloc(header: &ObjectHeaderState) -> u64 {
    header.addr
}

#[allow(non_snake_case)]
pub fn H5O__group_bh_info(header: &ObjectHeaderState) -> usize {
    header.messages.len()
}

#[allow(non_snake_case)]
pub fn H5O_msg_append_oh(header: &mut ObjectHeaderState, message: ObjectMessage) {
    H5O__msg_append_real(header, message);
}

#[allow(non_snake_case)]
pub fn H5O__msg_append_real(header: &mut ObjectHeaderState, message: ObjectMessage) {
    header.messages.push(message);
}

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

#[allow(non_snake_case)]
pub fn H5O_msg_reset(message: &mut ObjectMessage) {
    H5O__msg_reset_real(message);
}

#[allow(non_snake_case)]
pub fn H5O__msg_reset_real(message: &mut ObjectMessage) {
    message.data.clear();
    message.flags = 0;
    message.shared = false;
}

#[allow(non_snake_case)]
pub fn H5O_msg_free(_message: ObjectMessage) {}

#[allow(non_snake_case)]
pub fn H5O__msg_free_mesg(message: &mut ObjectMessage) {
    message.data.clear();
}

#[allow(non_snake_case)]
pub fn H5O_msg_free_real(_message: ObjectMessage) {}

#[allow(non_snake_case)]
pub fn H5O_msg_copy(message: &ObjectMessage) -> ObjectMessage {
    message.clone()
}

#[allow(non_snake_case)]
pub fn H5O_msg_exists(header: &ObjectHeaderState, msg_type: u16) -> bool {
    header.messages.iter().any(|msg| msg.msg_type == msg_type)
}

#[allow(non_snake_case)]
pub fn H5O_msg_exists_oh(header: &ObjectHeaderState, msg_type: u16) -> bool {
    H5O_msg_exists(header, msg_type)
}

#[allow(non_snake_case)]
pub fn H5O_msg_remove(header: &mut ObjectHeaderState, msg_type: u16) -> Option<ObjectMessage> {
    H5O__msg_remove_real(header, msg_type)
}

#[allow(non_snake_case)]
pub fn H5O_msg_remove_op(message: &ObjectMessage, msg_type: u16) -> bool {
    message.msg_type == msg_type
}

#[allow(non_snake_case)]
pub fn H5O__msg_remove_cb(message: &ObjectMessage, msg_type: u16) -> bool {
    H5O_msg_remove_op(message, msg_type)
}

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

#[allow(non_snake_case)]
pub fn H5O_msg_iterate(header: &ObjectHeaderState) -> impl Iterator<Item = &ObjectMessage> {
    H5O__msg_iterate_real(header)
}

#[allow(non_snake_case)]
pub fn H5O__msg_iterate_real(header: &ObjectHeaderState) -> impl Iterator<Item = &ObjectMessage> {
    header.messages.iter()
}

#[allow(non_snake_case)]
pub fn H5O_msg_raw_size(message: &ObjectMessage) -> usize {
    message.data.len()
}

#[allow(non_snake_case)]
pub fn H5O_msg_size_f(message: &ObjectMessage) -> usize {
    H5O_msg_raw_size(message)
}

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

#[allow(non_snake_case)]
pub fn H5O_msg_can_share(message: &ObjectMessage) -> bool {
    !message.data.is_empty()
}

#[allow(non_snake_case)]
pub fn H5O_msg_can_share_in_ohdr(message: &ObjectMessage) -> bool {
    H5O_msg_can_share(message)
}

#[allow(non_snake_case)]
pub fn H5O_msg_is_shared(message: &ObjectMessage) -> bool {
    message.shared
}

#[allow(non_snake_case)]
pub fn H5O_msg_set_share(message: &mut ObjectMessage) {
    message.shared = true;
}

#[allow(non_snake_case)]
pub fn H5O_msg_reset_share(message: &mut ObjectMessage) {
    message.shared = false;
}

#[allow(non_snake_case)]
pub fn H5O_msg_get_crt_index(message: &ObjectMessage) -> u16 {
    message.creation_index
}

#[allow(non_snake_case)]
pub fn H5O_msg_encode(message: &ObjectMessage) -> Result<Vec<u8>> {
    let len = 5usize
        .checked_add(message.data.len())
        .ok_or_else(|| Error::InvalidFormat("object message image length overflow".into()))?;
    let mut out = Vec::with_capacity(len);
    out.extend_from_slice(&message.msg_type.to_le_bytes());
    out.push(message.flags);
    out.extend_from_slice(&message.creation_index.to_le_bytes());
    out.extend_from_slice(&message.data);
    Ok(out)
}

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

#[allow(non_snake_case)]
pub fn H5O__msg_copy_file(message: &ObjectMessage) -> ObjectMessage {
    message.clone()
}

#[allow(non_snake_case)]
pub fn H5O__msg_alloc(msg_type: u16, data: Vec<u8>) -> ObjectMessage {
    ObjectMessage {
        msg_type,
        data,
        ..ObjectMessage::default()
    }
}

#[allow(non_snake_case)]
pub fn H5O__copy_mesg(message: &ObjectMessage) -> ObjectMessage {
    message.clone()
}

#[allow(non_snake_case)]
pub fn H5O_msg_delete(message: &mut ObjectMessage) {
    message.data.clear();
}

#[allow(non_snake_case)]
pub fn H5O_msg_flush(_message: &ObjectMessage) {}

#[allow(non_snake_case)]
pub fn H5O__flush_msgs(_header: &mut ObjectHeaderState) {}

#[allow(non_snake_case)]
pub fn H5O_msg_get_flags(message: &ObjectMessage) -> u8 {
    message.flags
}

#[allow(non_snake_case)]
pub fn H5O__cache_verify_chksum(image: &[u8], checksum: u32) -> bool {
    image
        .iter()
        .fold(0u32, |acc, byte| acc.wrapping_add(u32::from(*byte)))
        == checksum
}

#[allow(non_snake_case)]
pub fn H5O__cache_serialize(header: &ObjectHeaderState) -> Result<Vec<u8>> {
    let mut len = 0usize;
    for message in &header.messages {
        len = len
            .checked_add(5)
            .and_then(|value| value.checked_add(message.data.len()))
            .ok_or_else(|| {
                Error::InvalidFormat("object header cache image length overflow".into())
            })?;
    }
    let mut out = Vec::with_capacity(len);
    for message in &header.messages {
        out.extend_from_slice(&H5O_msg_encode(message)?);
    }
    Ok(out)
}

#[allow(non_snake_case)]
pub fn H5O__cache_notify(_header: &ObjectHeaderState) {}

#[allow(non_snake_case)]
pub fn H5O__cache_free_icr(_header: ObjectHeaderState) {}

#[allow(non_snake_case)]
pub fn H5O__cache_chk_get_initial_load_size(size: usize) -> usize {
    size.min(512)
}

#[allow(non_snake_case)]
pub fn H5O__cache_chk_verify_chksum(image: &[u8], checksum: u32) -> bool {
    H5O__cache_verify_chksum(image, checksum)
}

#[allow(non_snake_case)]
pub fn H5O__cache_chk_deserialize(image: &[u8]) -> Result<Vec<u8>> {
    validate_object_header_chunk_image(image)?;
    Ok(image.to_vec())
}

#[allow(non_snake_case)]
pub fn H5O__cache_chk_image_len(image: &[u8]) -> usize {
    image.len()
}

#[allow(non_snake_case)]
pub fn H5O__cache_chk_serialize(image: &[u8]) -> Result<Vec<u8>> {
    validate_object_header_chunk_image(image)?;
    Ok(image.to_vec())
}

#[allow(non_snake_case)]
pub fn H5O__cache_chk_notify(_image: &[u8]) {}

#[allow(non_snake_case)]
pub fn H5O__cache_chk_free_icr(_image: Vec<u8>) {}

#[allow(non_snake_case)]
pub fn H5O__add_cont_msg(header: &mut ObjectHeaderState, addr: u64, size: u64) {
    let mut data = Vec::with_capacity(16);
    data.extend_from_slice(&addr.to_le_bytes());
    data.extend_from_slice(&size.to_le_bytes());
    H5O_msg_append_oh(header, H5O__msg_alloc(0x0010, data));
}

fn validate_object_header_prefix_image(image: &[u8]) -> Result<()> {
    if image.starts_with(OBJECT_HEADER_V2_MAGIC) {
        validate_object_header_v2_prefix_image(image)
    } else {
        validate_object_header_v1_prefix_image(image)
    }
}

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

fn validate_object_header_v1_prefix_image(image: &[u8]) -> Result<()> {
    if image.len() < 16 {
        return Err(Error::InvalidFormat(
            "object header v1 prefix image is truncated".into(),
        ));
    }
    if image[0] != 1 {
        return Err(Error::InvalidFormat(format!(
            "object header prefix version {} is invalid",
            image[0]
        )));
    }
    let chunk_size = usize::try_from(read_le_u32_at(image, 8, "object header v1 chunk size")?)
        .map_err(|_| Error::InvalidFormat("object header v1 chunk size exceeds usize".into()))?;
    let expected = 16usize
        .checked_add(chunk_size)
        .ok_or_else(|| Error::InvalidFormat("object header v1 image size overflow".into()))?;
    if image.len() < expected {
        return Err(Error::InvalidFormat(
            "object header v1 prefix image is truncated".into(),
        ));
    }
    Ok(())
}

fn validate_object_header_v2_prefix_image(image: &[u8]) -> Result<()> {
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
                "object header attribute phase change max compact is less than min dense".into(),
            ));
        }
        pos = checked_add(pos, 4, "object header attribute phase change")?;
    }

    let size_len = 1usize << (flags & HDR_CHUNK0_SIZE_MASK);
    let chunk0_data_size = read_le_uint_width(image, pos, size_len, "object header v2 chunk size")?;
    pos = checked_add(pos, size_len, "object header v2 chunk size")?;
    let chunk0_data_size = usize::try_from(chunk0_data_size)
        .map_err(|_| Error::InvalidFormat("object header v2 chunk size is too large".into()))?;
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
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5O__prefix_deserialize(image: &[u8]) -> Result<Vec<u8>> {
    validate_object_header_prefix_image(image)?;
    Ok(image.to_vec())
}

#[allow(non_snake_case)]
pub fn H5O__chunk_deserialize(image: &[u8]) -> Result<Vec<u8>> {
    validate_object_header_chunk_image(image)?;
    Ok(image.to_vec())
}

#[allow(non_snake_case)]
pub fn H5O__bogus_encode(data: &[u8]) -> Vec<u8> {
    data.to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__bogus_size(data: &[u8]) -> usize {
    data.len()
}

#[allow(non_snake_case)]
pub fn H5O__bogus_debug(data: &[u8]) -> String {
    format!("bogus({} bytes)", data.len())
}

#[allow(non_snake_case)]
pub fn H5O__layout_decode(bytes: &[u8]) -> Result<LayoutMessage> {
    let version = bytes.first().copied().ok_or_else(|| {
        Error::InvalidFormat("data layout message is missing version byte".into())
    })?;
    if !(1..=4).contains(&version) {
        return Err(Error::InvalidFormat(format!(
            "data layout message version {version}"
        )));
    }
    Ok(LayoutMessage {
        version,
        raw: bytes.to_vec(),
    })
}

#[allow(non_snake_case)]
pub fn H5O__layout_encode(layout: &LayoutMessage) -> Result<Vec<u8>> {
    let version = layout.raw.first().copied().ok_or_else(|| {
        Error::InvalidFormat("data layout message is missing version byte".into())
    })?;
    if version != layout.version {
        return Err(Error::InvalidFormat(
            "data layout message version does not match raw payload".into(),
        ));
    }
    if !(1..=4).contains(&version) {
        return Err(Error::InvalidFormat(format!(
            "data layout message version {version}"
        )));
    }
    Ok(layout.raw.clone())
}

#[allow(non_snake_case)]
pub fn H5O__layout_copy(layout: &LayoutMessage) -> LayoutMessage {
    layout.clone()
}

#[allow(non_snake_case)]
pub fn H5O__layout_size(layout: &LayoutMessage) -> usize {
    layout.raw.len()
}

#[allow(non_snake_case)]
pub fn H5O__layout_reset(layout: &mut LayoutMessage) {
    layout.raw.clear();
}

#[allow(non_snake_case)]
pub fn H5O__layout_free(_layout: LayoutMessage) {}

#[allow(non_snake_case)]
pub fn H5O__layout_delete(layout: &mut LayoutMessage) {
    layout.raw.clear();
}

#[allow(non_snake_case)]
pub fn H5O__layout_pre_copy_file(layout: &LayoutMessage) -> LayoutMessage {
    layout.clone()
}

#[allow(non_snake_case)]
pub fn H5O__layout_copy_file(layout: &LayoutMessage) -> LayoutMessage {
    layout.clone()
}

#[allow(non_snake_case)]
pub fn H5O__layout_debug(layout: &LayoutMessage) -> String {
    format!(
        "layout(version={}, bytes={})",
        layout.version,
        layout.raw.len()
    )
}

#[allow(non_snake_case)]
pub fn H5O__refcount_decode(bytes: &[u8]) -> Result<u32> {
    read_le_u32_at(bytes, 0, "object refcount message")
}

#[allow(non_snake_case)]
pub fn H5O__refcount_encode(refcount: u32) -> Vec<u8> {
    refcount.to_le_bytes().to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__refcount_copy(refcount: u32) -> u32 {
    refcount
}

#[allow(non_snake_case)]
pub fn H5O__refcount_size(_refcount: u32) -> usize {
    4
}

#[allow(non_snake_case)]
pub fn H5O__refcount_free(_refcount: u32) {}

#[allow(non_snake_case)]
pub fn H5O__refcount_pre_copy_file(refcount: u32) -> u32 {
    refcount
}

#[allow(non_snake_case)]
pub fn H5O__refcount_debug(refcount: u32) -> String {
    format!("refcount={refcount}")
}

#[allow(non_snake_case)]
pub fn H5O__fsinfo_decode(bytes: &[u8]) -> Result<FsInfoMessage> {
    if bytes.len() < 11 {
        return Err(Error::InvalidFormat(
            "file-space info message is truncated".into(),
        ));
    }
    Ok(FsInfoMessage {
        version: bytes[0],
        free_space_strategy: bytes[1],
        persist: bytes[2] != 0,
        threshold: read_le_u64_at(bytes, 3, "file-space info threshold")?,
        page_size: if bytes.len() >= 19 {
            Some(read_le_u64_at(bytes, 11, "file-space info page size")?)
        } else {
            None
        },
    })
}

#[allow(non_snake_case)]
pub fn H5O__fsinfo_encode(info: &FsInfoMessage) -> Result<Vec<u8>> {
    if !H5O_fsinfo_check_version(info) {
        return Err(Error::InvalidFormat(format!(
            "file-space info message version {}",
            info.version
        )));
    }
    let mut out = vec![
        info.version,
        info.free_space_strategy,
        u8::from(info.persist),
    ];
    out.extend_from_slice(&info.threshold.to_le_bytes());
    if let Some(page_size) = info.page_size {
        out.extend_from_slice(&page_size.to_le_bytes());
    }
    Ok(out)
}

#[allow(non_snake_case)]
pub fn H5O__fsinfo_copy(info: &FsInfoMessage) -> FsInfoMessage {
    info.clone()
}

#[allow(non_snake_case)]
pub fn H5O__fsinfo_size(_info: &FsInfoMessage) -> Result<usize> {
    Ok(H5O__fsinfo_encode(_info)?.len())
}

#[allow(non_snake_case)]
pub fn H5O__fsinfo_free(_info: FsInfoMessage) {}

#[allow(non_snake_case)]
pub fn H5O__fsinfo_debug(info: &FsInfoMessage) -> String {
    format!(
        "fsinfo(version={}, threshold={}, page_size={:?})",
        info.version, info.threshold, info.page_size
    )
}

#[allow(non_snake_case)]
pub fn H5O_fsinfo_set_version(info: &mut FsInfoMessage, version: u8) {
    info.version = version;
}

#[allow(non_snake_case)]
pub fn H5O_fsinfo_check_version(info: &FsInfoMessage) -> bool {
    info.version <= 1
}

#[allow(non_snake_case)]
pub fn H5O__stab_encode(stab: &SymbolTableMessage) -> Vec<u8> {
    let mut out = Vec::with_capacity(16);
    out.extend_from_slice(&stab.btree_addr.to_le_bytes());
    out.extend_from_slice(&stab.heap_addr.to_le_bytes());
    out
}

#[allow(non_snake_case)]
pub fn H5O__stab_copy(stab: &SymbolTableMessage) -> SymbolTableMessage {
    stab.clone()
}

#[allow(non_snake_case)]
pub fn H5O__stab_size(_stab: &SymbolTableMessage) -> usize {
    16
}

#[allow(non_snake_case)]
pub fn H5O__stab_free(_stab: SymbolTableMessage) {}

#[allow(non_snake_case)]
pub fn H5O__stab_delete(stab: &mut SymbolTableMessage) {
    *stab = SymbolTableMessage::default();
}

#[allow(non_snake_case)]
pub fn H5O__stab_copy_file(stab: &SymbolTableMessage) -> SymbolTableMessage {
    stab.clone()
}

#[allow(non_snake_case)]
pub fn H5O__stab_debug(stab: &SymbolTableMessage) -> String {
    format!("stab(btree={}, heap={})", stab.btree_addr, stab.heap_addr)
}

#[allow(non_snake_case)]
pub fn H5O__sdspace_decode(bytes: &[u8]) -> Result<Vec<u64>> {
    if bytes.len() % 8 != 0 {
        return Err(Error::InvalidFormat(
            "dataspace extent dimension payload is truncated".into(),
        ));
    }
    bytes
        .chunks_exact(8)
        .map(|raw| {
            let bytes: [u8; 8] = raw.try_into().map_err(|_| {
                Error::InvalidFormat("dataspace extent dimension is truncated".into())
            })?;
            Ok(u64::from_le_bytes(bytes))
        })
        .collect()
}

#[allow(non_snake_case)]
pub fn H5O__sdspace_copy(space: &[u64]) -> Vec<u64> {
    space.to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__sdspace_reset(space: &mut Vec<u64>) {
    space.clear();
}

#[allow(non_snake_case)]
pub fn H5O__sdspace_free(_space: Vec<u64>) {}

#[allow(non_snake_case)]
pub fn H5O__sdspace_pre_copy_file(space: &[u64]) -> Vec<u64> {
    space.to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__sdspace_debug(space: &[u64]) -> String {
    format!("sdspace{:?}", space)
}

#[allow(non_snake_case)]
pub fn H5Olink(header: &mut ObjectHeaderState, delta: i32) {
    if delta.is_negative() {
        header.refcount = header.refcount.saturating_sub(delta.unsigned_abs());
    } else {
        let delta = u32::try_from(delta).unwrap_or(u32::MAX);
        header.refcount = header.refcount.saturating_add(delta);
    }
}

#[allow(non_snake_case)]
pub fn H5Oincr_refcount(header: &mut ObjectHeaderState) {
    H5Olink(header, 1);
}

#[allow(non_snake_case)]
pub fn H5Odecr_refcount(header: &mut ObjectHeaderState) {
    H5Olink(header, -1);
}

#[allow(non_snake_case)]
pub fn H5Oexists_by_name(objects: &BTreeMap<String, ObjectHeaderState>, name: &str) -> bool {
    objects.contains_key(name)
}

#[allow(non_snake_case)]
pub fn H5Oset_comment(header: &mut ObjectHeaderState, comment: impl Into<String>) {
    header.comment = Some(comment.into());
}

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

fn bytes_decode(bytes: &[u8]) -> Vec<u8> {
    bytes.to_vec()
}

fn bytes_encode(bytes: &[u8]) -> Vec<u8> {
    bytes.to_vec()
}

fn bytes_size(bytes: &[u8]) -> usize {
    bytes.len()
}

fn bytes_debug(label: &str, bytes: &[u8]) -> String {
    format!("{label}({} bytes)", bytes.len())
}

#[allow(non_snake_case)]
pub fn H5Ovisit3(objects: &BTreeMap<String, ObjectHeaderState>) -> Vec<String> {
    objects.keys().cloned().collect()
}

#[allow(non_snake_case)]
pub fn H5O__are_mdc_flushes_disabled(header: &ObjectHeaderState) -> bool {
    header.flush_disabled
}

#[allow(non_snake_case)]
pub fn H5Oare_mdc_flushes_disabled(header: &ObjectHeaderState) -> bool {
    H5O__are_mdc_flushes_disabled(header)
}

#[allow(non_snake_case)]
pub fn H5Otoken_cmp(left: u64, right: u64) -> std::cmp::Ordering {
    left.cmp(&right)
}

#[allow(non_snake_case)]
pub fn H5Otoken_to_str(token: u64) -> String {
    format!("{token:#x}")
}

#[allow(non_snake_case)]
pub fn H5Otoken_from_str(token: &str) -> Result<u64> {
    let trimmed = token.strip_prefix("0x").unwrap_or(token);
    u64::from_str_radix(trimmed, 16)
        .map_err(|_| Error::InvalidFormat("invalid object token".into()))
}

#[allow(non_snake_case)]
pub fn H5O__print_time_field(timestamp: u64) -> String {
    timestamp.to_string()
}

#[allow(non_snake_case)]
pub fn H5O__assert(condition: bool) -> Result<()> {
    condition
        .then_some(())
        .ok_or_else(|| Error::InvalidFormat("object assertion failed".into()))
}

#[allow(non_snake_case)]
pub fn H5O_debug_id(addr: u64) -> String {
    format!("object@{addr:#x}")
}

#[allow(non_snake_case)]
pub fn H5O__debug_real(header: &ObjectHeaderState) -> String {
    format!(
        "object(addr={:#x}, messages={})",
        header.addr,
        header.messages.len()
    )
}

#[allow(non_snake_case)]
pub fn H5O_debug(header: &ObjectHeaderState) -> String {
    H5O__debug_real(header)
}

#[allow(non_snake_case)]
pub fn H5O__mdci_encode(bytes: &[u8]) -> Vec<u8> {
    bytes_encode(bytes)
}

#[allow(non_snake_case)]
pub fn H5O__mdci_copy(bytes: &[u8]) -> Vec<u8> {
    bytes.to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__mdci_size(bytes: &[u8]) -> usize {
    bytes_size(bytes)
}

#[allow(non_snake_case)]
pub fn H5O__mdci_free(_bytes: Vec<u8>) {}

#[allow(non_snake_case)]
pub fn H5O__mdci_delete(bytes: &mut Vec<u8>) {
    bytes.clear();
}

#[allow(non_snake_case)]
pub fn H5O__mdci_debug(bytes: &[u8]) -> String {
    bytes_debug("mdci", bytes)
}

#[allow(non_snake_case)]
pub fn H5O__attr_decode(bytes: &[u8]) -> Result<AttributeObjectMessage> {
    Ok(AttributeObjectMessage {
        message: AttributeMessage::decode(bytes)?,
        raw_size: bytes.len(),
    })
}

#[allow(non_snake_case)]
pub fn H5O__attr_copy(message: &AttributeObjectMessage) -> AttributeObjectMessage {
    message.clone()
}

#[allow(non_snake_case)]
pub fn H5O__attr_size(message: &AttributeObjectMessage) -> usize {
    message.raw_size
}

#[allow(non_snake_case)]
pub fn H5O__attr_free(_message: AttributeObjectMessage) {}

#[allow(non_snake_case)]
pub fn H5O__attr_pre_copy_file(message: &AttributeObjectMessage) -> AttributeObjectMessage {
    message.clone()
}

#[allow(non_snake_case)]
pub fn H5O__attr_copy_file(message: &AttributeObjectMessage) -> AttributeObjectMessage {
    message.clone()
}

#[allow(non_snake_case)]
pub fn H5O__attr_post_copy_file(message: &AttributeObjectMessage) -> AttributeObjectMessage {
    message.clone()
}

#[allow(non_snake_case)]
pub fn H5O__attr_debug(message: &AttributeObjectMessage) -> String {
    format!(
        "attr(version={}, name={}, data={} bytes)",
        message.message.version,
        message.message.name,
        message.message.data.len()
    )
}

#[allow(non_snake_case)]
pub fn H5O__chunk_add(header: &mut ObjectHeaderState, message: ObjectMessage) {
    H5O_msg_append_oh(header, message);
}

#[allow(non_snake_case)]
pub fn H5O__chunk_unprotect(_header: &mut ObjectHeaderState) {}

#[allow(non_snake_case)]
pub fn H5O__chunk_update_idx(header: &mut ObjectHeaderState) {
    for (idx, msg) in header.messages.iter_mut().enumerate() {
        msg.creation_index = u16::try_from(idx).unwrap_or(u16::MAX);
    }
}

#[allow(non_snake_case)]
pub fn H5O__add_gap(header: &mut ObjectHeaderState, size: usize) {
    H5O_msg_append_oh(header, H5O__msg_alloc(0, vec![0; size]));
}

#[allow(non_snake_case)]
pub fn H5O__eliminate_gap(header: &mut ObjectHeaderState) {
    header
        .messages
        .retain(|msg| msg.msg_type != 0 || msg.data.iter().any(|b| *b != 0));
}

#[allow(non_snake_case)]
pub fn H5O__alloc_null(size: usize) -> ObjectMessage {
    H5O__msg_alloc(0, vec![0; size])
}

#[allow(non_snake_case)]
pub fn H5O__alloc_msgs(header: &mut ObjectHeaderState, count: usize) {
    header.messages.reserve(count);
}

#[allow(non_snake_case)]
pub fn H5O__alloc_extend_chunk(header: &mut ObjectHeaderState, size: usize) {
    H5O__add_gap(header, size);
}

#[allow(non_snake_case)]
pub fn H5O__alloc_new_chunk(size: usize) -> Vec<u8> {
    vec![0; size]
}

#[allow(non_snake_case)]
pub fn H5O__alloc_find_best_null(header: &ObjectHeaderState) -> Option<usize> {
    header.messages.iter().position(|msg| msg.msg_type == 0)
}

#[allow(non_snake_case)]
pub fn H5O__alloc(header: &mut ObjectHeaderState, message: ObjectMessage) {
    H5O_msg_append_oh(header, message);
}

#[allow(non_snake_case)]
pub fn H5O__release_mesg(_message: &mut ObjectMessage) {}

#[allow(non_snake_case)]
pub fn H5O__move_cont(header: &mut ObjectHeaderState, from: usize, to: usize) -> Result<()> {
    if from >= header.messages.len() || to > header.messages.len() {
        return Err(Error::InvalidFormat(
            "object message move index out of range".into(),
        ));
    }
    let msg = header.messages.remove(from);
    header.messages.insert(to.min(header.messages.len()), msg);
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5O__move_msgs_forward(header: &mut ObjectHeaderState) {
    header.messages.sort_by_key(|msg| msg.creation_index);
}

#[allow(non_snake_case)]
pub fn H5O__merge_null(header: &mut ObjectHeaderState) {
    let total: usize = header
        .messages
        .iter()
        .filter(|msg| msg.msg_type == 0)
        .map(|msg| msg.data.len())
        .sum();
    header.messages.retain(|msg| msg.msg_type != 0);
    if total > 0 {
        header.messages.push(H5O__alloc_null(total));
    }
}

#[allow(non_snake_case)]
pub fn H5O__remove_empty_chunks(header: &mut ObjectHeaderState) {
    header
        .messages
        .retain(|msg| !msg.data.is_empty() || msg.msg_type == 0);
}

#[allow(non_snake_case)]
pub fn H5O__condense_header(header: &mut ObjectHeaderState) {
    H5O__merge_null(header);
    H5O__remove_empty_chunks(header);
}

#[allow(non_snake_case)]
pub fn H5O__alloc_shrink_chunk(header: &mut ObjectHeaderState) {
    H5O__condense_header(header);
}

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

#[allow(non_snake_case)]
pub fn H5O__mtime_decode(bytes: &[u8]) -> Result<u64> {
    read_le_u64_at(bytes, 0, "modification time message")
}

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

#[allow(non_snake_case)]
pub fn H5O__mtime_encode(timestamp: u64) -> Vec<u8> {
    timestamp.to_le_bytes().to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__mtime_copy(timestamp: u64) -> u64 {
    timestamp
}

#[allow(non_snake_case)]
pub fn H5O__mtime_new_size(_timestamp: u64) -> usize {
    8
}

#[allow(non_snake_case)]
pub fn H5O__mtime_size(_timestamp: u64) -> usize {
    8
}

#[allow(non_snake_case)]
pub fn H5O__mtime_free(_timestamp: u64) {}

#[allow(non_snake_case)]
pub fn H5O__mtime_debug(timestamp: u64) -> String {
    format!("mtime={timestamp}")
}

#[allow(non_snake_case)]
pub fn H5O__copy_header_real(header: &ObjectHeaderState) -> ObjectHeaderState {
    header.clone()
}

#[allow(non_snake_case)]
pub fn H5O__copy_free_addrmap_cb(_addr: u64) {}

#[allow(non_snake_case)]
pub fn H5O__copy_header(header: &ObjectHeaderState) -> ObjectHeaderState {
    header.clone()
}

#[allow(non_snake_case)]
pub fn H5O__copy_obj(header: &ObjectHeaderState) -> ObjectHeaderState {
    header.clone()
}

#[allow(non_snake_case)]
pub fn H5O__copy_free_comm_dt_cb(_addr: u64) {}

#[allow(non_snake_case)]
pub fn H5O__copy_comm_dt_cmp(left: u64, right: u64) -> std::cmp::Ordering {
    left.cmp(&right)
}

#[allow(non_snake_case)]
pub fn H5O__copy_search_comm_dt_attr_cb(_message: &ObjectMessage) -> bool {
    false
}

#[allow(non_snake_case)]
pub fn H5O__copy_search_comm_dt_check(_header: &ObjectHeaderState) -> bool {
    false
}

#[allow(non_snake_case)]
pub fn H5O__copy_search_comm_dt_cb(_header: &ObjectHeaderState) -> Option<u64> {
    None
}

#[allow(non_snake_case)]
pub fn H5O__copy_insert_comm_dt(_addr: u64) {}

#[allow(non_snake_case)]
pub fn H5O_flush(_header: &mut ObjectHeaderState) {}

#[allow(non_snake_case)]
pub fn H5O_flush_common(header: &mut ObjectHeaderState) {
    H5O_flush(header);
}

#[allow(non_snake_case)]
pub fn H5O__oh_tag(header: &ObjectHeaderState) -> u64 {
    header.addr
}

#[allow(non_snake_case)]
pub fn H5O_refresh_metadata(_header: &mut ObjectHeaderState) {}

#[allow(non_snake_case)]
pub fn H5O__refresh_metadata_close(_header: &mut ObjectHeaderState) {}

#[allow(non_snake_case)]
pub fn H5O_refresh_metadata_reopen(header: &ObjectHeaderState) -> ObjectHeaderState {
    header.clone()
}

#[allow(non_snake_case)]
pub fn H5O__shmesg_decode(bytes: &[u8]) -> Result<SharedMessageTableInfo> {
    H5O__shmesg_decode_with_addr_size(bytes, 8)
}

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

#[allow(non_snake_case)]
pub fn H5O__shmesg_copy(table: &SharedMessageTableInfo) -> SharedMessageTableInfo {
    table.clone()
}

#[allow(non_snake_case)]
pub fn H5O__shmesg_size(_table: &SharedMessageTableInfo) -> usize {
    10
}

#[allow(non_snake_case)]
pub fn H5O__shmesg_debug(table: &SharedMessageTableInfo) -> String {
    format!(
        "shmesg(version={}, table_addr={:#x}, indexes={})",
        table.version, table.table_addr, table.nindexes
    )
}

#[allow(non_snake_case)]
pub fn H5O__pline_decode(bytes: &[u8]) -> Result<FilterPipelineMessage> {
    FilterPipelineMessage::decode(bytes)
}

#[allow(non_snake_case)]
pub fn H5O__pline_copy(message: &FilterPipelineMessage) -> FilterPipelineMessage {
    message.clone()
}

#[allow(non_snake_case)]
pub fn H5O__pline_size(message: &FilterPipelineMessage) -> usize {
    match message.version {
        1 => {
            let filters_size: usize = message
                .filters
                .iter()
                .map(|filter| {
                    let name_len = filter
                        .name
                        .as_ref()
                        .map(|name| align8_len(name.len() + 1))
                        .unwrap_or(0);
                    let client_data_len = filter.client_data.len() * 4;
                    let client_data_padding = if filter.client_data.len() % 2 != 0 {
                        4
                    } else {
                        0
                    };
                    8 + name_len + client_data_len + client_data_padding
                })
                .sum();
            8 + filters_size
        }
        2 => {
            let filters_size: usize = message
                .filters
                .iter()
                .map(|filter| {
                    let name_len = if filter.id >= 256 {
                        2 + filter.name.as_ref().map(|name| name.len() + 1).unwrap_or(0)
                    } else {
                        0
                    };
                    2 + name_len + 2 + 2 + filter.client_data.len() * 4
                })
                .sum();
            2 + filters_size
        }
        _ => 0,
    }
}

#[allow(non_snake_case)]
pub fn H5O__pline_reset(message: &mut FilterPipelineMessage) {
    message.filters.clear();
}

#[allow(non_snake_case)]
pub fn H5O__pline_free(_message: FilterPipelineMessage) {}

#[allow(non_snake_case)]
pub fn H5O__pline_pre_copy_file(message: &FilterPipelineMessage) -> FilterPipelineMessage {
    message.clone()
}

#[allow(non_snake_case)]
pub fn H5O__pline_debug(message: &FilterPipelineMessage) -> String {
    format!(
        "pline(version={}, filters={})",
        message.version,
        message.filters.len()
    )
}

#[allow(non_snake_case)]
pub fn H5O_pline_set_version(bytes: &mut Vec<u8>, version: u8) {
    if bytes.is_empty() {
        bytes.push(version);
    } else {
        bytes[0] = version;
    }
}

#[allow(non_snake_case)]
pub fn H5O__drvinfo_decode(bytes: &[u8]) -> Vec<u8> {
    bytes_decode(bytes)
}

#[allow(non_snake_case)]
pub fn H5O__drvinfo_encode(bytes: &[u8]) -> Vec<u8> {
    bytes_encode(bytes)
}

#[allow(non_snake_case)]
pub fn H5O__drvinfo_copy(bytes: &[u8]) -> Vec<u8> {
    bytes.to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__drvinfo_size(bytes: &[u8]) -> usize {
    bytes.len()
}

#[allow(non_snake_case)]
pub fn H5O__drvinfo_reset(bytes: &mut Vec<u8>) {
    bytes.clear();
}

#[allow(non_snake_case)]
pub fn H5O__drvinfo_debug(bytes: &[u8]) -> String {
    bytes_debug("drvinfo", bytes)
}

#[allow(non_snake_case)]
pub fn H5O_init() -> bool {
    H5O__init_package()
}

#[allow(non_snake_case)]
pub fn H5O__init_package() -> bool {
    true
}

#[allow(non_snake_case)]
pub fn H5O__set_version(layout: &mut LayoutMessage, version: u8) {
    layout.version = version;
}

#[allow(non_snake_case)]
pub fn H5O_create_ohdr(addr: u64) -> ObjectHeaderState {
    ObjectHeaderState {
        addr,
        refcount: 1,
        ..ObjectHeaderState::default()
    }
}

#[allow(non_snake_case)]
pub fn H5O_apply_ohdr(header: &mut ObjectHeaderState, f: impl FnOnce(&mut ObjectHeaderState)) {
    f(header);
}

#[allow(non_snake_case)]
pub fn H5O_open(header: &ObjectHeaderState) -> ObjectHeaderState {
    header.clone()
}

#[allow(non_snake_case)]
pub fn H5O_open_name(
    objects: &BTreeMap<String, ObjectHeaderState>,
    name: &str,
) -> Option<ObjectHeaderState> {
    objects.get(name).cloned()
}

#[allow(non_snake_case)]
pub fn H5O__open_by_idx(
    objects: &BTreeMap<String, ObjectHeaderState>,
    idx: usize,
) -> Option<ObjectHeaderState> {
    objects.values().nth(idx).cloned()
}

#[allow(non_snake_case)]
pub fn H5O__open_by_addr(
    objects: &BTreeMap<String, ObjectHeaderState>,
    addr: u64,
) -> Option<ObjectHeaderState> {
    objects.values().find(|header| header.addr == addr).cloned()
}

#[allow(non_snake_case)]
pub fn H5O_open_by_loc(
    location: &ObjectLocation,
    objects: &BTreeMap<String, ObjectHeaderState>,
) -> Option<ObjectHeaderState> {
    H5O__open_by_addr(objects, location.addr)
}

#[allow(non_snake_case)]
pub fn H5O_close(_header: ObjectHeaderState) {}

#[allow(non_snake_case)]
pub fn H5O__link_oh(header: &mut ObjectHeaderState, delta: i32) {
    H5Olink(header, delta);
}

#[allow(non_snake_case)]
pub fn H5O_link(header: &mut ObjectHeaderState, delta: i32) {
    H5Olink(header, delta);
}

#[allow(non_snake_case)]
pub fn H5O_pin(location: &mut ObjectLocation) {
    location.held = true;
}

#[allow(non_snake_case)]
pub fn H5O_unpin(location: &mut ObjectLocation) {
    location.held = false;
}

#[allow(non_snake_case)]
pub fn H5O_unprotect(_header: &mut ObjectHeaderState) {}

#[allow(non_snake_case)]
pub fn H5O_touch_oh(_header: &mut ObjectHeaderState) {}

#[allow(non_snake_case)]
pub fn H5O_touch(header: &mut ObjectHeaderState) {
    H5O_touch_oh(header);
}

#[allow(non_snake_case)]
pub fn H5O_bogus_oh(header: &ObjectHeaderState) -> bool {
    header.addr == u64::MAX
}

#[allow(non_snake_case)]
pub fn H5O_delete(header: &mut ObjectHeaderState) {
    H5O__delete_oh(header);
}

#[allow(non_snake_case)]
pub fn H5O__delete_oh(header: &mut ObjectHeaderState) {
    header.messages.clear();
    header.refcount = 0;
}

#[allow(non_snake_case)]
pub fn H5O_obj_type(header: &ObjectHeaderState) -> &'static str {
    H5O__obj_type_real(header)
}

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

#[allow(non_snake_case)]
pub fn H5O__obj_class(header: &ObjectHeaderState) -> &'static str {
    H5O_obj_type(header)
}

#[allow(non_snake_case)]
pub fn H5O_get_loc(header: &ObjectHeaderState) -> ObjectLocation {
    ObjectLocation {
        addr: header.addr,
        ..ObjectLocation::default()
    }
}

#[allow(non_snake_case)]
pub fn H5O_loc_reset(location: &mut ObjectLocation) {
    *location = ObjectLocation::default();
}

#[allow(non_snake_case)]
pub fn H5O_loc_copy(location: &ObjectLocation) -> ObjectLocation {
    location.clone()
}

#[allow(non_snake_case)]
pub fn H5O_loc_copy_shallow(location: &ObjectLocation) -> ObjectLocation {
    location.clone()
}

#[allow(non_snake_case)]
pub fn H5O_loc_copy_deep(location: &ObjectLocation) -> ObjectLocation {
    location.clone()
}

#[allow(non_snake_case)]
pub fn H5O_loc_hold_file(location: &mut ObjectLocation) {
    location.held = true;
}

#[allow(non_snake_case)]
pub fn H5O_loc_free(_location: ObjectLocation) {}

#[allow(non_snake_case)]
pub fn H5O_get_hdr_info(header: &ObjectHeaderState) -> ObjectInfo {
    H5O__get_hdr_info_real(header)
}

#[allow(non_snake_case)]
pub fn H5O__get_hdr_info_real(header: &ObjectHeaderState) -> ObjectInfo {
    ObjectInfo {
        addr: header.addr,
        refcount: header.refcount,
        msg_count: header.messages.len(),
        has_checksum: H5O_has_chksum(header),
    }
}

#[allow(non_snake_case)]
pub fn H5O_get_info(header: &ObjectHeaderState) -> ObjectInfo {
    H5O_get_hdr_info(header)
}

#[allow(non_snake_case)]
pub fn H5O_get_native_info(header: &ObjectHeaderState) -> ObjectInfo {
    H5O_get_hdr_info(header)
}

#[allow(non_snake_case)]
pub fn H5O_get_create_plist(_header: &ObjectHeaderState) -> BTreeMap<String, String> {
    BTreeMap::new()
}

#[allow(non_snake_case)]
pub fn H5O_get_nlinks(header: &ObjectHeaderState) -> u32 {
    header.refcount
}

#[allow(non_snake_case)]
pub fn H5O_obj_create(addr: u64) -> ObjectHeaderState {
    H5O_create_ohdr(addr)
}

#[allow(non_snake_case)]
pub fn H5O_get_oh_addr(header: &ObjectHeaderState) -> u64 {
    header.addr
}

#[allow(non_snake_case)]
pub fn H5O_get_oh_flags(header: &ObjectHeaderState) -> u8 {
    header.messages.iter().fold(0, |acc, msg| acc | msg.flags)
}

#[allow(non_snake_case)]
pub fn H5O_get_oh_mtime(_header: &ObjectHeaderState) -> u64 {
    0
}

#[allow(non_snake_case)]
pub fn H5O_get_oh_version(_header: &ObjectHeaderState) -> u8 {
    2
}

#[allow(non_snake_case)]
pub fn H5O_get_rc_and_type(header: &ObjectHeaderState) -> (u32, &'static str) {
    (header.refcount, H5O_obj_type(header))
}

#[allow(non_snake_case)]
pub fn H5O__visit_cb(name: &str, _header: &ObjectHeaderState) -> String {
    name.to_string()
}

#[allow(non_snake_case)]
pub fn H5O__visit(objects: &BTreeMap<String, ObjectHeaderState>) -> Vec<String> {
    H5Ovisit3(objects)
}

#[allow(non_snake_case)]
pub fn H5O__inc_rc(header: &mut ObjectHeaderState) {
    H5Oincr_refcount(header);
}

#[allow(non_snake_case)]
pub fn H5O__dec_rc(header: &mut ObjectHeaderState) {
    H5Odecr_refcount(header);
}

#[allow(non_snake_case)]
pub fn H5O_get_proxy(header: &ObjectHeaderState) -> u64 {
    header.addr
}

#[allow(non_snake_case)]
pub fn H5O__free(_header: ObjectHeaderState) {}

#[allow(non_snake_case)]
pub fn H5O__reset_info2(info: &mut ObjectInfo) {
    *info = ObjectInfo::default();
}

#[allow(non_snake_case)]
pub fn H5O_has_chksum(header: &ObjectHeaderState) -> bool {
    !header.messages.is_empty()
}

#[allow(non_snake_case)]
pub fn H5O_get_version_bound(_header: &ObjectHeaderState) -> (u8, u8) {
    (0, 4)
}

#[allow(non_snake_case)]
pub fn H5O__copy_obj_by_ref(header: &ObjectHeaderState) -> ObjectHeaderState {
    header.clone()
}

#[allow(non_snake_case)]
pub fn H5O__copy_expand_ref_object1(token: u64) -> u64 {
    token
}

#[allow(non_snake_case)]
pub fn H5O__copy_expand_ref_region1(region: &[u8]) -> Vec<u8> {
    region.to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__copy_expand_ref_object2(token: u64) -> u64 {
    token
}

#[allow(non_snake_case)]
pub fn H5O_copy_expand_ref(bytes: &[u8]) -> Vec<u8> {
    bytes.to_vec()
}

#[allow(non_snake_case)]
pub fn H5O__cont_decode(bytes: &[u8]) -> Result<(u64, u64)> {
    let addr = read_le_u64_at(bytes, 0, "object-header continuation address")?;
    let size = read_le_u64_at(bytes, 8, "object-header continuation size")?;
    Ok((addr, size))
}

#[allow(non_snake_case)]
pub fn H5O__cont_encode(addr: u64, size: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(16);
    out.extend_from_slice(&addr.to_le_bytes());
    out.extend_from_slice(&size.to_le_bytes());
    out
}

#[allow(non_snake_case)]
pub fn H5O__cont_size(_addr: u64, _size: u64) -> usize {
    16
}

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

fn checked_add(offset: usize, len: usize, context: &str) -> Result<usize> {
    offset
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))
}

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

fn read_u8_cursor(data: &[u8], pos: &mut usize, context: &str) -> Result<u8> {
    let value = data
        .get(*pos)
        .copied()
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))?;
    *pos = checked_add(*pos, 1, context)?;
    Ok(value)
}

fn read_le_uint_cursor(data: &[u8], pos: &mut usize, width: usize, context: &str) -> Result<u64> {
    let value = read_le_uint_width(data, *pos, width, context)?;
    *pos = checked_add(*pos, width, context)?;
    Ok(value)
}

fn is_undefined_addr_width(addr: u64, sizeof_addr: u8) -> Result<bool> {
    let width = usize::from(sizeof_addr);
    if !(1..=8).contains(&width) {
        return Err(Error::InvalidFormat(format!(
            "address size {width} is invalid"
        )));
    }
    let undef = if width == 8 {
        u64::MAX
    } else {
        (1u64 << (width * 8)) - 1
    };
    Ok(addr == undef)
}

fn align8_len(len: usize) -> usize {
    len.saturating_add(7) & !7
}

#[allow(non_snake_case)]
pub fn H5O__cont_free(_cont: (u64, u64)) {}

#[allow(non_snake_case)]
pub fn H5O__cont_delete(cont: &mut (u64, u64)) {
    *cont = (0, 0);
}

#[allow(non_snake_case)]
pub fn H5O__cont_debug(cont: (u64, u64)) -> String {
    format!("cont(addr={}, size={})", cont.0, cont.1)
}

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

#[allow(non_snake_case)]
pub fn H5O__ginfo_copy(info: &GroupInfoMessage) -> GroupInfoMessage {
    info.clone()
}

#[allow(non_snake_case)]
pub fn H5O__ginfo_size(info: &GroupInfoMessage) -> Result<usize> {
    H5O__ginfo_encode(info).map(|bytes| bytes.len())
}

#[allow(non_snake_case)]
pub fn H5O__ginfo_free(_info: GroupInfoMessage) {}

#[allow(non_snake_case)]
pub fn H5O__ginfo_debug(info: &GroupInfoMessage) -> String {
    format!(
        "ginfo(version={}, max_compact={:?}, min_dense={:?})",
        info.version, info.max_compact, info.min_dense
    )
}

#[allow(non_snake_case)]
pub fn H5O__attr_create(header: &mut ObjectHeaderState, name: &str, value: &[u8]) {
    let mut data = name.as_bytes().to_vec();
    data.push(0);
    data.extend_from_slice(value);
    H5O_msg_append_oh(header, H5O__msg_alloc(0x000c, data));
}

#[allow(non_snake_case)]
pub fn H5O__attr_open_by_name(header: &ObjectHeaderState, name: &str) -> Option<ObjectMessage> {
    let prefix = name.as_bytes();
    header
        .messages
        .iter()
        .find(|msg| msg.msg_type == 0x000c && msg.data.starts_with(prefix))
        .cloned()
}

#[allow(non_snake_case)]
pub fn H5O__attr_open_by_idx_cb(message: &ObjectMessage) -> ObjectMessage {
    message.clone()
}

#[allow(non_snake_case)]
pub fn H5O__attr_open_by_idx(header: &ObjectHeaderState, index: usize) -> Option<ObjectMessage> {
    header
        .messages
        .iter()
        .filter(|msg| msg.msg_type == 0x000c)
        .nth(index)
        .cloned()
}

#[allow(non_snake_case)]
pub fn H5O__attr_find_opened_attr(header: &ObjectHeaderState, name: &str) -> bool {
    H5O__attr_open_by_name(header, name).is_some()
}

#[allow(non_snake_case)]
pub fn H5O__attr_update_shared(message: &mut ObjectMessage, shared: bool) {
    message.shared = shared;
}

#[allow(non_snake_case)]
pub fn H5O__attr_write_cb(message: &mut ObjectMessage, data: &[u8]) {
    message.data.clear();
    message.data.extend_from_slice(data);
}

#[allow(non_snake_case)]
pub fn H5O__attr_write(header: &mut ObjectHeaderState, name: &str, value: &[u8]) {
    if let Some(pos) = header
        .messages
        .iter()
        .position(|msg| msg.msg_type == 0x000c && msg.data.starts_with(name.as_bytes()))
    {
        H5O__attr_write_cb(&mut header.messages[pos], value);
    } else {
        H5O__attr_create(header, name, value);
    }
}

#[allow(non_snake_case)]
pub fn H5O__attr_rename(header: &mut ObjectHeaderState, old_name: &str, new_name: &str) -> bool {
    H5O__attr_rename_checked(header, old_name, new_name).unwrap_or(false)
}

#[allow(non_snake_case)]
pub fn H5O__attr_rename_checked(
    header: &mut ObjectHeaderState,
    old_name: &str,
    new_name: &str,
) -> Result<bool> {
    if let Some(msg) = header
        .messages
        .iter_mut()
        .find(|msg| msg.msg_type == 0x000c && msg.data.starts_with(old_name.as_bytes()))
    {
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

#[allow(non_snake_case)]
pub fn H5O_attr_iterate_real(header: &ObjectHeaderState) -> Vec<ObjectMessage> {
    header
        .messages
        .iter()
        .filter(|msg| msg.msg_type == 0x000c)
        .cloned()
        .collect()
}

#[allow(non_snake_case)]
pub fn H5O__attr_iterate(header: &ObjectHeaderState) -> Vec<ObjectMessage> {
    H5O_attr_iterate_real(header)
}

#[allow(non_snake_case)]
pub fn H5O__attr_remove_update(_header: &mut ObjectHeaderState) {}

#[allow(non_snake_case)]
pub fn H5O__attr_remove(header: &mut ObjectHeaderState, name: &str) -> bool {
    if let Some(pos) = header
        .messages
        .iter()
        .position(|msg| msg.msg_type == 0x000c && msg.data.starts_with(name.as_bytes()))
    {
        header.messages.remove(pos);
        true
    } else {
        false
    }
}

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

#[allow(non_snake_case)]
pub fn H5O__attr_count_real(header: &ObjectHeaderState) -> usize {
    H5O_attr_iterate_real(header).len()
}

#[allow(non_snake_case)]
pub fn H5O__attr_exists(header: &ObjectHeaderState, name: &str) -> bool {
    H5O__attr_find_opened_attr(header, name)
}

#[allow(non_snake_case)]
pub fn H5O__attr_bh_info(header: &ObjectHeaderState) -> usize {
    H5O__attr_count_real(header)
}

#[allow(non_snake_case)]
pub fn H5O__fill_new_decode(bytes: &[u8]) -> Result<FillValueMessage> {
    FillValueMessage::decode(bytes)
}

#[allow(non_snake_case)]
pub fn H5O__fill_old_decode(bytes: &[u8]) -> Result<FillValueMessage> {
    FillValueMessage::decode_old(bytes)
}

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

#[allow(non_snake_case)]
pub fn H5O__fill_copy(message: &FillValueMessage) -> FillValueMessage {
    message.clone()
}

#[allow(non_snake_case)]
pub fn H5O__fill_new_size(message: &FillValueMessage) -> usize {
    match message.version {
        1 | 2 => {
            4 + if message.defined {
                4 + message.value.as_ref().map_or(0, Vec::len)
            } else {
                0
            }
        }
        3 => 2 + message.value.as_ref().map_or(0, |value| 4 + value.len()),
        _ => 0,
    }
}

#[allow(non_snake_case)]
pub fn H5O__fill_old_size(message: &FillValueMessage) -> Result<usize> {
    4usize
        .checked_add(message.value.as_ref().map_or(0, Vec::len))
        .ok_or_else(|| Error::InvalidFormat("old fill value image length overflow".into()))
}

#[allow(non_snake_case)]
pub fn H5O_fill_reset_dyn(message: &mut FillValueMessage) {
    message.value = None;
    message.defined = false;
}

#[allow(non_snake_case)]
pub fn H5O__fill_reset(message: &mut FillValueMessage) {
    H5O_fill_reset_dyn(message);
}

#[allow(non_snake_case)]
pub fn H5O__fill_free(_message: FillValueMessage) {}

#[allow(non_snake_case)]
pub fn H5O__fill_pre_copy_file(message: &FillValueMessage) -> FillValueMessage {
    message.clone()
}

#[allow(non_snake_case)]
pub fn H5O_fill_set_version(bytes: &mut Vec<u8>, version: u8) {
    if bytes.is_empty() {
        bytes.push(version);
    } else {
        bytes[0] = version;
    }
}

#[allow(non_snake_case)]
pub fn H5O__reset_info1(info: &mut ObjectInfo) {
    *info = ObjectInfo::default();
}

#[allow(non_snake_case)]
pub fn H5O__iterate1_adapter(header: &ObjectHeaderState) -> Vec<ObjectMessage> {
    header.messages.clone()
}

#[allow(non_snake_case)]
pub fn H5O__get_info_old(header: &ObjectHeaderState) -> ObjectInfo {
    H5O_get_info(header)
}

#[allow(non_snake_case)]
pub fn H5Oopen_by_addr(
    objects: &BTreeMap<String, ObjectHeaderState>,
    addr: u64,
) -> Option<ObjectHeaderState> {
    H5O__open_by_addr(objects, addr)
}

#[allow(non_snake_case)]
pub fn H5Ovisit1(objects: &BTreeMap<String, ObjectHeaderState>) -> Vec<String> {
    H5Ovisit3(objects)
}

#[allow(non_snake_case)]
pub fn H5Ovisit_by_name2(
    objects: &BTreeMap<String, ObjectHeaderState>,
    prefix: &str,
) -> Vec<String> {
    objects
        .keys()
        .filter(|name| name.starts_with(prefix))
        .cloned()
        .collect()
}

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

#[allow(non_snake_case)]
pub fn H5O__btreek_copy(message: &BTreeKMessage) -> BTreeKMessage {
    message.clone()
}

#[allow(non_snake_case)]
pub fn H5O__btreek_size(_message: &BTreeKMessage) -> usize {
    7
}

#[allow(non_snake_case)]
pub fn H5O__btreek_debug(message: &BTreeKMessage) -> String {
    format!(
        "btreek(indexed={}, group_internal={}, group_leaf={})",
        message.indexed_storage_internal_k, message.group_internal_k, message.group_leaf_k
    )
}

#[allow(non_snake_case)]
pub fn H5O__unknown_free(_bytes: Vec<u8>) {}

#[allow(non_snake_case)]
pub fn H5O__link_decode(bytes: &[u8]) -> Result<LinkObjectMessage> {
    H5O__link_decode_with_addr_size(bytes, 8)
}

#[allow(non_snake_case)]
pub fn H5O__link_decode_with_addr_size(bytes: &[u8], sizeof_addr: u8) -> Result<LinkObjectMessage> {
    Ok(LinkObjectMessage {
        message: LinkMessage::decode(bytes, sizeof_addr)?,
        raw_size: bytes.len(),
    })
}

#[allow(non_snake_case)]
pub fn H5O__link_size(message: &LinkObjectMessage) -> usize {
    message.raw_size
}

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

#[allow(non_snake_case)]
pub fn H5O__link_free(_message: LinkObjectMessage) {}

#[allow(non_snake_case)]
pub fn H5O__link_copy_file(message: &LinkObjectMessage) -> LinkObjectMessage {
    message.clone()
}

#[allow(non_snake_case)]
pub fn H5O__link_post_copy_file(message: &LinkObjectMessage) -> LinkObjectMessage {
    message.clone()
}

#[allow(non_snake_case)]
pub fn H5O__link_debug(message: &LinkObjectMessage) -> String {
    format!(
        "link(name={}, type={:?}, raw_size={})",
        message.message.name, message.message.link_type, message.raw_size
    )
}

#[allow(non_snake_case)]
pub fn H5O__linfo_decode(bytes: &[u8]) -> Result<LinkInfoObjectMessage> {
    H5O__linfo_decode_with_addr_size(bytes, 8)
}

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

#[allow(non_snake_case)]
pub fn H5O__linfo_copy(message: &LinkInfoObjectMessage) -> LinkInfoObjectMessage {
    message.clone()
}

#[allow(non_snake_case)]
pub fn H5O__linfo_size(message: &LinkInfoObjectMessage) -> usize {
    message.raw_size
}

#[allow(non_snake_case)]
pub fn H5O__linfo_free(_message: LinkInfoObjectMessage) {}

#[allow(non_snake_case)]
pub fn H5O__linfo_delete(message: &mut LinkInfoObjectMessage) {
    message.message.max_creation_index = None;
    message.message.fractal_heap_addr = 0;
    message.message.name_btree_addr = 0;
    message.message.corder_btree_addr = None;
    message.raw_size = 0;
}

#[allow(non_snake_case)]
pub fn H5O__linfo_copy_file(message: &LinkInfoObjectMessage) -> LinkInfoObjectMessage {
    message.clone()
}

#[allow(non_snake_case)]
pub fn H5O__linfo_post_copy_file_cb(message: &LinkInfoObjectMessage) -> LinkInfoObjectMessage {
    message.clone()
}

#[allow(non_snake_case)]
pub fn H5O__linfo_post_copy_file(message: &LinkInfoObjectMessage) -> LinkInfoObjectMessage {
    message.clone()
}

#[allow(non_snake_case)]
pub fn H5O__linfo_debug(message: &LinkInfoObjectMessage) -> String {
    format!(
        "linfo(version={}, flags={:#x}, heap={:#x}, name_btree={:#x})",
        message.message.version,
        message.message.flags,
        message.message.fractal_heap_addr,
        message.message.name_btree_addr
    )
}

#[allow(non_snake_case)]
pub fn H5O__efl_decode(bytes: &[u8]) -> Result<ExternalFileListMessage> {
    H5O__efl_decode_with_sizes(bytes, 8, 8)
}

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
    out.extend_from_slice(&[0; 3]);
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

#[allow(non_snake_case)]
pub fn H5O__efl_copy(message: &ExternalFileListMessage) -> ExternalFileListMessage {
    message.clone()
}

#[allow(non_snake_case)]
pub fn H5O__efl_size(message: &ExternalFileListMessage) -> Result<usize> {
    message
        .entries
        .len()
        .checked_mul(24)
        .and_then(|payload| payload.checked_add(16))
        .ok_or_else(|| Error::InvalidFormat("external file list image length overflow".into()))
}

#[allow(non_snake_case)]
pub fn H5O__efl_reset(message: &mut ExternalFileListMessage) {
    message.entries.clear();
}

#[allow(non_snake_case)]
pub fn H5O_efl_total_size(message: &ExternalFileListMessage) -> u64 {
    message.entries.iter().map(|entry| entry.size).sum()
}

#[allow(non_snake_case)]
pub fn H5O__efl_copy_file(message: &ExternalFileListMessage) -> ExternalFileListMessage {
    message.clone()
}

#[allow(non_snake_case)]
pub fn H5O__efl_debug(message: &ExternalFileListMessage) -> String {
    format!(
        "efl(version={}, heap_addr={:#x}, entries={})",
        message.version,
        message.heap_addr,
        message.entries.len()
    )
}

#[allow(non_snake_case)]
pub fn H5O__ainfo_decode(bytes: &[u8]) -> Result<AttributeInfoObjectMessage> {
    H5O__ainfo_decode_with_addr_size(bytes, 8)
}

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

#[allow(non_snake_case)]
pub fn H5O__ainfo_copy(message: &AttributeInfoObjectMessage) -> AttributeInfoObjectMessage {
    message.clone()
}

#[allow(non_snake_case)]
pub fn H5O__ainfo_size(message: &AttributeInfoObjectMessage) -> usize {
    message.raw_size
}

#[allow(non_snake_case)]
pub fn H5O__ainfo_free(_message: AttributeInfoObjectMessage) {}

#[allow(non_snake_case)]
pub fn H5O__ainfo_delete(message: &mut AttributeInfoObjectMessage) {
    message.message.max_creation_index = None;
    message.message.fractal_heap_addr = 0;
    message.message.name_btree_addr = 0;
    message.message.corder_btree_addr = None;
    message.raw_size = 0;
}

#[allow(non_snake_case)]
pub fn H5O__ainfo_pre_copy_file(
    message: &AttributeInfoObjectMessage,
) -> AttributeInfoObjectMessage {
    message.clone()
}

#[allow(non_snake_case)]
pub fn H5O__ainfo_copy_file(message: &AttributeInfoObjectMessage) -> AttributeInfoObjectMessage {
    message.clone()
}

#[allow(non_snake_case)]
pub fn H5O__ainfo_post_copy_file(
    message: &AttributeInfoObjectMessage,
) -> AttributeInfoObjectMessage {
    message.clone()
}

#[allow(non_snake_case)]
pub fn H5O__ainfo_debug(message: &AttributeInfoObjectMessage) -> String {
    format!(
        "ainfo(version={}, flags={:#x}, heap={:#x}, name_btree={:#x})",
        message.message.version,
        message.message.flags,
        message.message.fractal_heap_addr,
        message.message.name_btree_addr
    )
}

#[allow(non_snake_case)]
pub fn H5O__dset_get_copy_file_udata(header: &ObjectHeaderState) -> ObjectHeaderState {
    header.clone()
}

#[allow(non_snake_case)]
pub fn H5O__dset_free_copy_file_udata(_header: ObjectHeaderState) {}

#[allow(non_snake_case)]
pub fn H5O__dset_isa(header: &ObjectHeaderState) -> bool {
    header
        .messages
        .iter()
        .any(|msg| msg.msg_type == 0x0001 || msg.msg_type == 0x0003)
}

#[allow(non_snake_case)]
pub fn H5O__dset_open(header: &ObjectHeaderState) -> ObjectHeaderState {
    header.clone()
}

#[allow(non_snake_case)]
pub fn H5O__dset_create(addr: u64) -> ObjectHeaderState {
    H5O_create_ohdr(addr)
}

#[allow(non_snake_case)]
pub fn H5O__dset_get_oloc(header: &ObjectHeaderState) -> u64 {
    header.addr
}

#[allow(non_snake_case)]
pub fn H5O__dset_bh_info(header: &ObjectHeaderState) -> usize {
    header.messages.len()
}

#[allow(non_snake_case)]
pub fn H5O__dset_flush(_header: &mut ObjectHeaderState) {}

#[allow(non_snake_case)]
pub fn H5O__dtype_isa(header: &ObjectHeaderState) -> bool {
    header.messages.iter().any(|msg| msg.msg_type == 0x0003)
}

#[allow(non_snake_case)]
pub fn H5O__dtype_open(header: &ObjectHeaderState) -> ObjectHeaderState {
    header.clone()
}

#[allow(non_snake_case)]
pub fn H5O__dtype_create(addr: u64) -> ObjectHeaderState {
    H5O_create_ohdr(addr)
}

#[allow(non_snake_case)]
pub fn H5O__dtype_get_oloc(header: &ObjectHeaderState) -> u64 {
    header.addr
}

#[allow(non_snake_case)]
pub fn H5O__is_attr_dense_test(header: &ObjectHeaderState) -> bool {
    H5O__attr_count_real(header) > 8
}

#[allow(non_snake_case)]
pub fn H5O__is_attr_empty_test(header: &ObjectHeaderState) -> bool {
    H5O__attr_count_real(header) == 0
}

#[allow(non_snake_case)]
pub fn H5O__num_attrs_test(header: &ObjectHeaderState) -> usize {
    H5O__attr_count_real(header)
}

#[allow(non_snake_case)]
pub fn H5O__attr_dense_info_test(header: &ObjectHeaderState) -> usize {
    H5O__attr_count_real(header)
}

#[allow(non_snake_case)]
pub fn H5O__check_msg_marked_test(message: &ObjectMessage) -> bool {
    message.flags != 0
}

#[allow(non_snake_case)]
pub fn H5O__expunge_chunks_test(header: &mut ObjectHeaderState) {
    header.messages.clear();
}

#[allow(non_snake_case)]
pub fn H5O__get_rc_test(header: &ObjectHeaderState) -> u32 {
    header.refcount
}

#[allow(non_snake_case)]
pub fn H5O__msg_get_chunkno_test(_message: &ObjectMessage) -> usize {
    0
}

#[allow(non_snake_case)]
pub fn H5O__msg_move_to_new_chunk_test(header: &mut ObjectHeaderState, idx: usize) -> Result<()> {
    H5O__move_cont(header, idx, header.messages.len())
}

#[allow(non_snake_case)]
pub fn H5O_SHARED_DECODE(bytes: &[u8]) -> Result<SharedMessageReference> {
    H5O_SHARED_DECODE_WITH_CONTEXT(bytes, 8, 8)
}

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

#[allow(non_snake_case)]
pub fn H5O_SHARED_ENCODE(reference: &SharedMessageReference) -> Vec<u8> {
    match reference {
        SharedMessageReference::V1 {
            message_type,
            index,
            addr,
        } => {
            let mut out = Vec::with_capacity(24);
            out.push(1);
            out.push(*message_type);
            out.extend_from_slice(&[0; 6]);
            out.extend_from_slice(&index.to_le_bytes());
            out.extend_from_slice(&addr.to_le_bytes());
            out
        }
        SharedMessageReference::V2 { message_type, addr } => {
            let mut out = Vec::with_capacity(10);
            out.push(2);
            out.push(*message_type);
            out.extend_from_slice(&addr.to_le_bytes());
            out
        }
        SharedMessageReference::V3Sohm { heap_id } => {
            let mut out = Vec::with_capacity(10);
            out.push(3);
            out.push(1);
            out.extend_from_slice(heap_id);
            out
        }
        SharedMessageReference::V3Committed { addr } => {
            let mut out = Vec::with_capacity(10);
            out.push(3);
            out.push(2);
            out.extend_from_slice(&addr.to_le_bytes());
            out
        }
    }
}

#[allow(non_snake_case)]
pub fn H5O_SHARED_SIZE(reference: &SharedMessageReference) -> usize {
    H5O_SHARED_ENCODE(reference).len()
}

#[allow(non_snake_case)]
pub fn H5O_SHARED_DELETE(reference: &mut Option<SharedMessageReference>) {
    *reference = None;
}

#[allow(non_snake_case)]
pub fn H5O_SHARED_LINK(message: &mut ObjectMessage, shared: bool) {
    message.shared = shared;
}

#[allow(non_snake_case)]
pub fn H5O_SHARED_COPY_FILE(reference: &SharedMessageReference) -> SharedMessageReference {
    reference.clone()
}

#[allow(non_snake_case)]
pub fn H5O_SHARED_POST_COPY_FILE(reference: &SharedMessageReference) -> SharedMessageReference {
    reference.clone()
}

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

#[allow(non_snake_case)]
pub fn H5O__dtype_decode_helper(bytes: &[u8]) -> Result<DatatypeMessage> {
    DatatypeMessage::decode(bytes)
}

#[allow(non_snake_case)]
pub fn H5O__dtype_encode_helper(message: &DatatypeMessage) -> Result<Vec<u8>> {
    H5O__dtype_encode(message)
}

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

#[allow(non_snake_case)]
pub fn H5O__dtype_copy(message: &DatatypeMessage) -> DatatypeMessage {
    message.clone()
}

#[allow(non_snake_case)]
pub fn H5O__dtype_reset(message: &mut DatatypeMessage) {
    message.properties.clear();
}

#[allow(non_snake_case)]
pub fn H5O__dtype_can_share(message: &DatatypeMessage) -> bool {
    message.size > 0
}

#[allow(non_snake_case)]
pub fn H5O__dtype_pre_copy_file(message: &DatatypeMessage) -> DatatypeMessage {
    message.clone()
}

#[allow(non_snake_case)]
pub fn H5O__dtype_copy_file(message: &DatatypeMessage) -> DatatypeMessage {
    message.clone()
}

#[allow(non_snake_case)]
pub fn H5O__dtype_debug(message: &DatatypeMessage) -> String {
    format!(
        "dtype(version={}, class={:?}, size={}, properties={})",
        message.version,
        message.class,
        message.size,
        message.properties.len()
    )
}

#[allow(non_snake_case)]
pub fn H5O__dtype_size(message: &DatatypeMessage) -> Result<usize> {
    8usize
        .checked_add(message.properties.len())
        .ok_or_else(|| Error::InvalidFormat("datatype message image length overflow".into()))
}

#[allow(non_snake_case)]
pub fn H5O__name_decode(bytes: &[u8]) -> Result<String> {
    let nul = bytes
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(bytes.len());
    std::str::from_utf8(&bytes[..nul])
        .map(str::to_string)
        .map_err(|_| Error::InvalidFormat("object name is not UTF-8".into()))
}

#[allow(non_snake_case)]
pub fn H5O__name_encode(name: &str) -> Result<Vec<u8>> {
    let len = name
        .len()
        .checked_add(1)
        .ok_or_else(|| Error::InvalidFormat("object name image length overflow".into()))?;
    let mut out = Vec::with_capacity(len);
    out.extend_from_slice(name.as_bytes());
    out.push(0);
    Ok(out)
}

#[allow(non_snake_case)]
pub fn H5O__name_copy(name: &str) -> String {
    name.to_string()
}

#[allow(non_snake_case)]
pub fn H5O__name_size(name: &str) -> usize {
    name.len() + 1
}

#[allow(non_snake_case)]
pub fn H5O__name_reset(name: &mut String) {
    name.clear();
}

#[allow(non_snake_case)]
pub fn H5O__name_debug(name: &str) -> String {
    format!("name={name}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn object_messages_roundtrip_and_remove() {
        let msg = H5O__msg_alloc(42, b"abc".to_vec());
        let decoded = H5O_msg_decode(&H5O_msg_encode(&msg).unwrap()).unwrap();
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
        let first_image = H5O_msg_encode(&first).unwrap();
        assert_eq!(first_image.len(), 8);
        let image = H5O__cache_serialize(&header).unwrap();
        assert_eq!(
            image.len(),
            first_image.len() + H5O_msg_encode(&second).unwrap().len()
        );
        assert_eq!(&image[..first_image.len()], &first_image);
    }

    #[test]
    fn object_message_decode_rejects_truncated_header() {
        let err = H5O_msg_decode(&[1, 0, 0, 0]).unwrap_err();
        assert!(matches!(err, Error::InvalidFormat(_)));
    }

    #[test]
    fn object_layout_decode_rejects_missing_version() {
        let err = H5O__layout_decode(&[]).unwrap_err();
        assert!(matches!(err, Error::InvalidFormat(_)));
        let err = H5O__layout_decode(&[9]).unwrap_err();
        assert!(matches!(err, Error::InvalidFormat(_)));
    }

    #[test]
    fn object_layout_decode_preserves_raw_payload() {
        let decoded = H5O__layout_decode(&[4, 1, 2, 3]).unwrap();
        assert_eq!(decoded.version, 4);
        assert_eq!(decoded.raw, vec![4, 1, 2, 3]);
    }

    #[test]
    fn object_prefix_deserialize_validates_header_images() {
        let mut v1 = vec![0; 16];
        v1[0] = 1;
        assert_eq!(H5O__prefix_deserialize(&v1).unwrap(), v1);
        assert!(H5O__prefix_deserialize(&[3, 0, 0, 0]).is_err());

        let mut v2 = b"OHDR".to_vec();
        v2.push(2);
        v2.push(0);
        v2.push(0);
        let checksum = checksum_metadata(&v2);
        v2.extend_from_slice(&checksum.to_le_bytes());
        assert_eq!(H5O__prefix_deserialize(&v2).unwrap(), v2);

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
        assert_eq!(H5O__chunk_deserialize(&v1_raw).unwrap(), v1_raw);
        assert_eq!(H5O__cache_chk_deserialize(&v1_raw).unwrap(), v1_raw);
        assert_eq!(H5O__cache_chk_serialize(&v1_raw).unwrap(), v1_raw);

        let mut v2 = b"OCHKpayload".to_vec();
        let checksum = checksum_metadata(&v2);
        v2.extend_from_slice(&checksum.to_le_bytes());
        assert_eq!(H5O__chunk_deserialize(&v2).unwrap(), v2);
        assert_eq!(H5O__cache_chk_deserialize(&v2).unwrap(), v2);
        assert_eq!(H5O__cache_chk_serialize(&v2).unwrap(), v2);

        let mut bad = b"OCHKpayload".to_vec();
        bad.extend_from_slice(&0u32.to_le_bytes());
        assert!(H5O__chunk_deserialize(&bad).is_err());
        assert!(H5O__cache_chk_deserialize(&bad).is_err());
        assert!(H5O__cache_chk_serialize(&bad).is_err());
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
    }

    #[test]
    fn object_aux_decoders_accept_complete_payloads() {
        let fsinfo = H5O__fsinfo_decode(&[1, 2, 1, 8, 0, 0, 0, 0, 0, 0, 0]).unwrap();
        assert_eq!(fsinfo.version, 1);
        assert_eq!(fsinfo.free_space_strategy, 2);
        assert!(fsinfo.persist);
        assert_eq!(fsinfo.threshold, 8);
        assert_eq!(fsinfo.page_size, None);
        assert_eq!(H5O__fsinfo_size(&fsinfo).unwrap(), 11);
        let fsinfo_page = H5O__fsinfo_decode(&[
            1, 1, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0xaa,
        ])
        .unwrap();
        assert_eq!(fsinfo_page.threshold, 8);
        assert_eq!(fsinfo_page.page_size, Some(4096));
        assert_eq!(H5O__fsinfo_size(&fsinfo_page).unwrap(), 19);
        assert_eq!(
            H5O__fsinfo_encode(&fsinfo_page).unwrap(),
            vec![1, 1, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0]
        );

        assert_eq!(H5O__sdspace_decode(&16u64.to_le_bytes()).unwrap(), vec![16]);
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
        assert_eq!(H5O__refcount_encode(7), vec![7, 0, 0, 0]);
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
        assert_eq!(H5O__name_decode(b"alpha\0ignored").unwrap(), "alpha");
        assert_eq!(H5O__name_decode(b"alpha").unwrap(), "alpha");
        assert!(H5O__name_decode(&[0xff, 0]).is_err());
        assert_eq!(H5O__name_encode("alpha").unwrap(), b"alpha\0".to_vec());
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

        let fill_old = H5O__fill_old_decode(&[2, 0, 0, 0, 0xcc, 0xdd, 0xaa]).unwrap();
        assert_eq!(fill_old.version, 0);
        assert!(fill_old.defined);
        assert_eq!(fill_old.value.as_deref(), Some(&[0xcc, 0xdd][..]));
        assert_eq!(H5O__fill_old_size(&fill_old).unwrap(), 6);
        assert_eq!(
            H5O__fill_old_encode(&fill_old).unwrap(),
            vec![2, 0, 0, 0, 0xcc, 0xdd]
        );

        let pline = H5O__pline_decode(&[2, 1, 1, 0, 0, 0, 1, 0, 6, 0, 0, 0]).unwrap();
        assert_eq!(pline.version, 2);
        assert_eq!(pline.filters.len(), 1);
        assert_eq!(pline.filters[0].id, 1);
        assert_eq!(pline.filters[0].client_data, vec![6]);
        assert_eq!(H5O__pline_size(&pline), 12);
        assert_eq!(H5O__pline_debug(&pline), "pline(version=2, filters=1)");

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
        assert_eq!(H5O_SHARED_SIZE(&shared), 10);
        assert_eq!(
            H5O_SHARED_ENCODE(&shared),
            vec![3, 1, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a]
        );
        assert!(H5O_SHARED_DEBUG(&shared).contains("sohm"));

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
        assert!(H5O__attr_debug(&attr).contains("name=x"));

        let mut cont = Vec::new();
        cont.extend_from_slice(&24u64.to_le_bytes());
        cont.extend_from_slice(&32u64.to_le_bytes());
        assert_eq!(H5O__cont_decode(&cont).unwrap(), (24, 32));
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
            H5O__attr_open_by_name(&header, "new").unwrap().data,
            b"new\0value"
        );
        assert!(H5O__attr_open_by_name(&header, "old").is_none());
    }
}
