use std::io::{Read, Seek, SeekFrom, Write};
use std::str;

use crate::error::{Error, Result};
use crate::format::messages::link::LinkType;
use crate::format::messages::link_info::LinkInfoMessage;
use crate::format::object_header::{
    self, HDR_ATTR_CRT_ORDER_TRACKED, HDR_ATTR_STORE_PHASE_CHANGE, HDR_CHUNK0_SIZE_MASK,
    HDR_STORE_TIMES, HDR_V2_KNOWN_FLAGS,
};
use crate::io::reader::HdfReader;

use super::MutableFile;

#[derive(Debug)]
struct CompactLinkMessageLocation {
    msg_type_offset: u64,
    msg_data_offset: u64,
    oh_start: u64,
    oh_check_len: usize,
    name_offset: usize,
    name_size: usize,
    link_type: LinkType,
    hard_link_addr: Option<u64>,
}

#[derive(Debug)]
struct ObjectRefcountLocation {
    value_offset: u64,
    oh_start: Option<u64>,
    oh_check_len: Option<usize>,
    refcount: u32,
}

impl MutableFile {
    /// Delete a compact link from a group.
    ///
    /// This is an in-place hdf5-metno compatibility mutation. It currently
    /// supports only v2 object headers with compact link messages.
    pub fn unlink_group_link(&mut self, group_path: &str, name: &str) -> Result<()> {
        let group = self.group(group_path)?;
        let location = self.find_compact_link_in_oh(group.addr(), name, None)?;
        if location.link_type == LinkType::Hard {
            let target_addr = location.hard_link_addr.ok_or_else(|| {
                Error::InvalidFormat("hard link message is missing target address".into())
            })?;
            if !self.decrement_object_refcount_if_present(target_addr)? {
                let same_parent_links =
                    self.count_compact_hard_links_to_addr(group.addr(), target_addr)?;
                if same_parent_links <= 1 {
                    return Err(Error::Unsupported(
                        "hard-link deletion without explicit object refcount requires another compact hard link to the same target in the parent group"
                            .into(),
                    ));
                }
            }
        }
        self.write_handle
            .seek(SeekFrom::Start(location.msg_type_offset))?;
        self.write_handle
            .write_all(&[object_header::MSG_NIL as u8])?;
        self.rewrite_oh_checksum(location.oh_start, location.oh_check_len)?;
        self.write_handle.flush()?;
        self.reopen_reader()?;
        Ok(())
    }

    /// Rename a compact link in a group without changing the encoded name size.
    ///
    /// Moving between groups or changing the encoded message length needs
    /// object-header growth and is rejected explicitly for now.
    pub fn rename_group_link(
        &mut self,
        group_path: &str,
        old_name: &str,
        new_name: &str,
    ) -> Result<()> {
        if new_name.is_empty() {
            return Err(Error::InvalidFormat("link name cannot be empty".into()));
        }
        if old_name == new_name {
            return Ok(());
        }

        let group = self.group(group_path)?;
        let location = self.find_compact_link_in_oh(group.addr(), old_name, Some(new_name))?;
        if new_name.len() != location.name_size {
            return Err(Error::Unsupported(
                "in-place compact link rename cannot grow or shrink the encoded name field".into(),
            ));
        }

        let name_offset_u64 = Self::usize_to_u64(location.name_offset, "link name offset")?;
        let file_name_offset = location
            .msg_data_offset
            .checked_add(name_offset_u64)
            .ok_or_else(|| Error::InvalidFormat("link name offset overflow".into()))?;
        self.write_handle.seek(SeekFrom::Start(file_name_offset))?;
        self.write_handle.write_all(new_name.as_bytes())?;
        self.rewrite_oh_checksum(location.oh_start, location.oh_check_len)?;
        self.write_handle.flush()?;
        self.reopen_reader()?;
        Ok(())
    }

    fn find_compact_link_in_oh(
        &self,
        oh_addr: u64,
        target_name: &str,
        reject_duplicate_name: Option<&str>,
    ) -> Result<CompactLinkMessageLocation> {
        let mut guard = self.inner.lock();
        let sizeof_addr = guard.superblock.sizeof_addr;
        let reader = &mut guard.reader;
        reader.seek(oh_addr)?;

        let mut first_bytes = [0u8; 4];
        reader.read_bytes_into(&mut first_bytes)?;
        if first_bytes != [b'O', b'H', b'D', b'R'] {
            return Err(Error::Unsupported(
                "link mutation currently supports only v2 object headers".into(),
            ));
        }

        let version = reader.read_u8()?;
        if version != 2 {
            return Err(Error::Unsupported(
                "link mutation currently supports only v2 object headers".into(),
            ));
        }

        let flags = reader.read_u8()?;
        if flags & !HDR_V2_KNOWN_FLAGS != 0 {
            return Err(Error::InvalidFormat(format!(
                "object header v2 flags contain reserved bits: {flags:#04x}"
            )));
        }
        if flags & HDR_STORE_TIMES != 0 {
            reader.skip(16)?;
        }
        if flags & HDR_ATTR_STORE_PHASE_CHANGE != 0 {
            reader.skip(4)?;
        }

        let chunk0_size_bytes = 1u8 << (flags & HDR_CHUNK0_SIZE_MASK);
        let chunk0_data_size = reader.read_uint(chunk0_size_bytes)?;
        let chunk0_data_start = reader.position()?;
        let chunk0_data_end = chunk0_data_start
            .checked_add(chunk0_data_size)
            .ok_or_else(|| Error::InvalidFormat("object-header chunk range overflow".into()))?;
        let oh_check_len = usize::try_from(chunk0_data_end - oh_addr)
            .map_err(|_| Error::InvalidFormat("object-header checksum range overflow".into()))?;

        let mut has_dense_links = false;
        let mut found = None;
        let mut msg_buf = Vec::new();
        while reader.position()? < chunk0_data_end {
            let msg_header_pos = reader.position()?;
            if msg_header_pos
                .checked_add(4)
                .is_none_or(|end| end > chunk0_data_end)
            {
                break;
            }

            let msg_type = u16::from(reader.read_u8()?);
            let msg_size = usize::from(reader.read_u16()?);
            let _msg_flags = reader.read_u8()?;
            if flags & HDR_ATTR_CRT_ORDER_TRACKED != 0 {
                reader.skip(2)?;
            }

            let msg_data_offset = reader.position()?;
            let msg_size_u64 = Self::usize_to_u64(msg_size, "object-header message size")?;
            if msg_data_offset
                .checked_add(msg_size_u64)
                .is_none_or(|end| end > chunk0_data_end)
            {
                return Err(Error::InvalidFormat(
                    "object-header message payload exceeds chunk".into(),
                ));
            }

            if msg_type == object_header::MSG_LINK {
                read_message_into(reader, &mut msg_buf, msg_size)?;
                let link = compact_link_view(&msg_buf, sizeof_addr)?;
                if let Some(duplicate_name) = reject_duplicate_name {
                    if link.name == duplicate_name {
                        return Err(Error::InvalidFormat(format!(
                            "link '{duplicate_name}' already exists"
                        )));
                    }
                }
                if link.name == target_name {
                    let location = CompactLinkMessageLocation {
                        msg_type_offset: msg_header_pos,
                        msg_data_offset,
                        oh_start: oh_addr,
                        oh_check_len,
                        name_offset: link.name_offset,
                        name_size: link.name_size,
                        link_type: link.link_type,
                        hard_link_addr: link.hard_link_addr,
                    };
                    if reject_duplicate_name.is_none() {
                        return Ok(location);
                    }
                    found = Some(location);
                }
            } else {
                if msg_type == object_header::MSG_SYMBOL_TABLE {
                    return Err(Error::Unsupported(
                        "mutating v1 symbol-table group links is not implemented".into(),
                    ));
                } else if msg_type == object_header::MSG_LINK_INFO {
                    read_message_into(reader, &mut msg_buf, msg_size)?;
                    let link_info = LinkInfoMessage::decode(&msg_buf, sizeof_addr)?;
                    if link_info.has_dense_storage() || link_info.corder_btree_addr.is_some() {
                        has_dense_links = true;
                    }
                } else {
                    reader.skip(msg_size_u64)?;
                }
            }
        }

        if has_dense_links {
            return Err(Error::Unsupported(
                "mutating dense or creation-order indexed links is not implemented".into(),
            ));
        }
        found.ok_or_else(|| Error::InvalidFormat(format!("link '{target_name}' not found")))
    }

    fn decrement_object_refcount_if_present(&mut self, object_addr: u64) -> Result<bool> {
        let Some(location) = self.find_object_refcount_location(object_addr)? else {
            return Ok(false);
        };
        if location.refcount <= 1 {
            return Err(Error::Unsupported(
                "hard-link deletion would drop explicit object refcount below one".into(),
            ));
        }
        let new_refcount = location.refcount - 1;
        self.write_handle
            .seek(SeekFrom::Start(location.value_offset))?;
        self.write_handle.write_all(&new_refcount.to_le_bytes())?;
        if let (Some(oh_start), Some(oh_check_len)) = (location.oh_start, location.oh_check_len) {
            self.rewrite_oh_checksum(oh_start, oh_check_len)?;
        }
        Ok(true)
    }

    fn find_object_refcount_location(
        &self,
        object_addr: u64,
    ) -> Result<Option<ObjectRefcountLocation>> {
        let mut guard = self.inner.lock();
        let reader = &mut guard.reader;
        reader.seek(object_addr)?;

        let mut first_bytes = [0u8; 4];
        reader.read_bytes_into(&mut first_bytes)?;
        if first_bytes != [b'O', b'H', b'D', b'R'] {
            if first_bytes[0] != 1 {
                return Err(Error::Unsupported(
                    "hard-link deletion can update only v1/v2 object refcounts".into(),
                ));
            }
            let value_offset = object_addr.checked_add(4).ok_or_else(|| {
                Error::InvalidFormat("object-header refcount offset overflow".into())
            })?;
            reader.seek(value_offset)?;
            let refcount = reader.read_u32()?;
            return Ok(Some(ObjectRefcountLocation {
                value_offset,
                oh_start: None,
                oh_check_len: None,
                refcount,
            }));
        }

        let version = reader.read_u8()?;
        if version != 2 {
            return Err(Error::Unsupported(
                "hard-link deletion can update only v1/v2 object refcounts".into(),
            ));
        }

        let flags = reader.read_u8()?;
        if flags & !HDR_V2_KNOWN_FLAGS != 0 {
            return Err(Error::InvalidFormat(format!(
                "object header v2 flags contain reserved bits: {flags:#04x}"
            )));
        }
        if flags & HDR_STORE_TIMES != 0 {
            reader.skip(16)?;
        }
        if flags & HDR_ATTR_STORE_PHASE_CHANGE != 0 {
            reader.skip(4)?;
        }

        let chunk0_size_bytes = 1u8 << (flags & HDR_CHUNK0_SIZE_MASK);
        let chunk0_data_size = reader.read_uint(chunk0_size_bytes)?;
        let chunk0_data_start = reader.position()?;
        let chunk0_data_end = chunk0_data_start
            .checked_add(chunk0_data_size)
            .ok_or_else(|| Error::InvalidFormat("object-header chunk range overflow".into()))?;
        let oh_check_len = usize::try_from(chunk0_data_end - object_addr)
            .map_err(|_| Error::InvalidFormat("object-header checksum range overflow".into()))?;

        while reader.position()? < chunk0_data_end {
            let msg_header_pos = reader.position()?;
            if msg_header_pos
                .checked_add(4)
                .is_none_or(|end| end > chunk0_data_end)
            {
                break;
            }

            let msg_type = u16::from(reader.read_u8()?);
            let msg_size = usize::from(reader.read_u16()?);
            let _msg_flags = reader.read_u8()?;
            if flags & HDR_ATTR_CRT_ORDER_TRACKED != 0 {
                reader.skip(2)?;
            }

            let msg_data_offset = reader.position()?;
            let msg_size_u64 = Self::usize_to_u64(msg_size, "object-header message size")?;
            if msg_data_offset
                .checked_add(msg_size_u64)
                .is_none_or(|end| end > chunk0_data_end)
            {
                return Err(Error::InvalidFormat(
                    "object-header message payload exceeds chunk".into(),
                ));
            }

            if msg_type == object_header::MSG_OBJ_REF_COUNT {
                if msg_size < 4 {
                    return Err(Error::InvalidFormat(
                        "object refcount message is truncated".into(),
                    ));
                }
                let refcount = reader.read_u32()?;
                return Ok(Some(ObjectRefcountLocation {
                    value_offset: msg_data_offset,
                    oh_start: Some(object_addr),
                    oh_check_len: Some(oh_check_len),
                    refcount,
                }));
            }
            reader.skip(msg_size_u64)?;
        }

        Ok(None)
    }

    fn count_compact_hard_links_to_addr(&self, oh_addr: u64, target_addr: u64) -> Result<usize> {
        let mut guard = self.inner.lock();
        let sizeof_addr = guard.superblock.sizeof_addr;
        let reader = &mut guard.reader;
        reader.seek(oh_addr)?;

        let mut first_bytes = [0u8; 4];
        reader.read_bytes_into(&mut first_bytes)?;
        if first_bytes != [b'O', b'H', b'D', b'R'] {
            return Err(Error::Unsupported(
                "hard-link deletion without explicit refcount supports only v2 compact parent groups"
                    .into(),
            ));
        }

        let version = reader.read_u8()?;
        if version != 2 {
            return Err(Error::Unsupported(
                "hard-link deletion without explicit refcount supports only v2 compact parent groups"
                    .into(),
            ));
        }

        let flags = reader.read_u8()?;
        if flags & !HDR_V2_KNOWN_FLAGS != 0 {
            return Err(Error::InvalidFormat(format!(
                "object header v2 flags contain reserved bits: {flags:#04x}"
            )));
        }
        if flags & HDR_STORE_TIMES != 0 {
            reader.skip(16)?;
        }
        if flags & HDR_ATTR_STORE_PHASE_CHANGE != 0 {
            reader.skip(4)?;
        }

        let chunk0_size_bytes = 1u8 << (flags & HDR_CHUNK0_SIZE_MASK);
        let chunk0_data_size = reader.read_uint(chunk0_size_bytes)?;
        let chunk0_data_start = reader.position()?;
        let chunk0_data_end = chunk0_data_start
            .checked_add(chunk0_data_size)
            .ok_or_else(|| Error::InvalidFormat("object-header chunk range overflow".into()))?;

        let mut count = 0usize;
        let mut msg_buf = Vec::new();
        while reader.position()? < chunk0_data_end {
            let msg_header_pos = reader.position()?;
            if msg_header_pos
                .checked_add(4)
                .is_none_or(|end| end > chunk0_data_end)
            {
                break;
            }

            let msg_type = u16::from(reader.read_u8()?);
            let msg_size = usize::from(reader.read_u16()?);
            let _msg_flags = reader.read_u8()?;
            if flags & HDR_ATTR_CRT_ORDER_TRACKED != 0 {
                reader.skip(2)?;
            }

            let msg_data_offset = reader.position()?;
            let msg_size_u64 = Self::usize_to_u64(msg_size, "object-header message size")?;
            if msg_data_offset
                .checked_add(msg_size_u64)
                .is_none_or(|end| end > chunk0_data_end)
            {
                return Err(Error::InvalidFormat(
                    "object-header message payload exceeds chunk".into(),
                ));
            }

            if msg_type == object_header::MSG_LINK {
                read_message_into(reader, &mut msg_buf, msg_size)?;
                let link = compact_link_view(&msg_buf, sizeof_addr)?;
                if link.link_type == LinkType::Hard && link.hard_link_addr == Some(target_addr) {
                    count += 1;
                }
            } else if msg_type == object_header::MSG_LINK_INFO {
                read_message_into(reader, &mut msg_buf, msg_size)?;
                let link_info = LinkInfoMessage::decode(&msg_buf, sizeof_addr)?;
                if link_info.has_dense_storage() || link_info.corder_btree_addr.is_some() {
                    return Err(Error::Unsupported(
                        "hard-link deletion without explicit refcount does not support dense parent links"
                            .into(),
                    ));
                }
            } else if msg_type == object_header::MSG_SYMBOL_TABLE {
                return Err(Error::Unsupported(
                    "hard-link deletion without explicit refcount does not support v1 symbol-table parent groups"
                        .into(),
                ));
            } else {
                reader.skip(msg_size_u64)?;
            }
        }

        Ok(count)
    }
}

#[derive(Debug)]
struct CompactLinkView<'a> {
    name: &'a str,
    name_offset: usize,
    name_size: usize,
    link_type: LinkType,
    hard_link_addr: Option<u64>,
}

fn read_message_into<R>(
    reader: &mut HdfReader<R>,
    scratch: &mut Vec<u8>,
    msg_size: usize,
) -> Result<()>
where
    R: Read + Seek,
{
    scratch.clear();
    if scratch.capacity() < msg_size {
        scratch.reserve_exact(msg_size - scratch.capacity());
    }
    scratch.resize(msg_size, 0);
    reader.read_bytes_into(scratch)
}

fn compact_link_view(raw: &[u8], sizeof_addr: u8) -> Result<CompactLinkView<'_>> {
    let mut pos = 0usize;
    let version = read_u8(raw, &mut pos, "link message version")?;
    if version != 1 {
        return Err(Error::InvalidFormat(format!(
            "link message version {version}"
        )));
    }
    let flags = read_u8(raw, &mut pos, "link message flags")?;
    if flags & !0x1f != 0 {
        return Err(Error::InvalidFormat(format!(
            "link message flags {flags:#x} are invalid"
        )));
    }
    let name_len_size = 1usize << (flags & 0x03);
    let link_type = if flags & 0x08 != 0 {
        match read_u8(raw, &mut pos, "link message link type")? {
            0 => LinkType::Hard,
            1 => LinkType::Soft,
            64 => LinkType::External,
            65..=u8::MAX => LinkType::UserDefined(raw[pos - 1]),
            other => return Err(Error::InvalidFormat(format!("invalid link type {other}"))),
        }
    } else {
        LinkType::Hard
    };
    if flags & 0x04 != 0 {
        advance_pos(raw, &mut pos, 8, "link creation order")?;
    }
    let char_encoding = if flags & 0x10 != 0 {
        read_u8(raw, &mut pos, "link message character encoding")?
    } else {
        0
    };
    if char_encoding > 1 {
        return Err(Error::InvalidFormat(format!(
            "invalid link character encoding {char_encoding}"
        )));
    }
    let name_size = usize::try_from(read_le_u64(
        raw,
        &mut pos,
        name_len_size,
        "link name length",
    )?)
    .map_err(|_| Error::InvalidFormat("link name length overflows usize".into()))?;
    if name_size == 0 {
        return Err(Error::InvalidFormat("invalid link name length".into()));
    }
    ensure_available(raw, pos, name_size, "link name")?;
    let name_offset = pos;
    let name = str::from_utf8(&raw[name_offset..name_offset + name_size])
        .map_err(|_| Error::InvalidFormat("link name is not valid UTF-8".into()))?;
    advance_pos(raw, &mut pos, name_size, "link name")?;

    let hard_link_addr = match link_type {
        LinkType::Hard => Some(read_le_u64(
            raw,
            &mut pos,
            usize::from(sizeof_addr),
            "hard link address",
        )?),
        LinkType::Soft => {
            let target_len =
                usize::try_from(read_le_u64(raw, &mut pos, 2, "soft link target length")?)
                    .map_err(|_| Error::InvalidFormat("soft link target length overflow".into()))?;
            if target_len == 0 {
                return Err(Error::InvalidFormat("invalid soft link length".into()));
            }
            ensure_available(raw, pos, target_len, "soft link target")?;
            str::from_utf8(&raw[pos..pos + target_len])
                .map_err(|_| Error::InvalidFormat("soft link target is not valid UTF-8".into()))?;
            advance_pos(raw, &mut pos, target_len, "soft link target")?;
            None
        }
        LinkType::External => {
            let info_len =
                usize::try_from(read_le_u64(raw, &mut pos, 2, "external link info length")?)
                    .map_err(|_| {
                        Error::InvalidFormat("external link info length overflow".into())
                    })?;
            if info_len < 3 {
                return Err(Error::InvalidFormat(
                    "external link info is too short".into(),
                ));
            }
            validate_external_link_info(&raw[pos..], info_len)?;
            advance_pos(raw, &mut pos, info_len, "external link info")?;
            None
        }
        LinkType::UserDefined(_) => None,
    };

    Ok(CompactLinkView {
        name,
        name_offset,
        name_size,
        link_type,
        hard_link_addr,
    })
}

fn validate_external_link_info(raw: &[u8], info_len: usize) -> Result<()> {
    ensure_available(raw, 0, info_len, "external link info")?;
    let info = &raw[..info_len];
    if info[0] != 0 {
        return Err(Error::InvalidFormat(format!(
            "external link version {}",
            info[0]
        )));
    }
    let rest = &info[1..];
    let first_nul = rest.iter().position(|&byte| byte == 0).ok_or_else(|| {
        Error::InvalidFormat("external link filename is missing terminator".into())
    })?;
    let filename = &rest[..first_nul];
    let obj_path = &rest[first_nul + 1..];
    if obj_path.last() != Some(&0) {
        return Err(Error::InvalidFormat(
            "external link object path is missing terminator".into(),
        ));
    }
    str::from_utf8(filename)
        .map_err(|_| Error::InvalidFormat("external link filename is not valid UTF-8".into()))?;
    str::from_utf8(&obj_path[..obj_path.len() - 1])
        .map_err(|_| Error::InvalidFormat("external link object path is not valid UTF-8".into()))?;
    Ok(())
}

fn ensure_available(data: &[u8], pos: usize, len: usize, context: &str) -> Result<()> {
    let end = pos
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} length overflow")))?;
    if end > data.len() {
        return Err(Error::InvalidFormat(format!("{context} is truncated")));
    }
    Ok(())
}

fn advance_pos(data: &[u8], pos: &mut usize, len: usize, context: &str) -> Result<()> {
    ensure_available(data, *pos, len, context)?;
    *pos = pos
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} position overflow")))?;
    Ok(())
}

fn read_u8(data: &[u8], pos: &mut usize, context: &str) -> Result<u8> {
    ensure_available(data, *pos, 1, context)?;
    let value = data[*pos];
    advance_pos(data, pos, 1, context)?;
    Ok(value)
}

fn read_le_u64(data: &[u8], pos: &mut usize, size: usize, context: &str) -> Result<u64> {
    if !(1..=8).contains(&size) {
        return Err(Error::InvalidFormat(format!(
            "{context} has invalid byte width {size}"
        )));
    }
    ensure_available(data, *pos, size, context)?;
    let mut val = 0u64;
    for (idx, byte) in data[*pos..*pos + size].iter().enumerate() {
        val |= u64::from(*byte) << (idx * 8);
    }
    advance_pos(data, pos, size, context)?;
    Ok(val)
}
