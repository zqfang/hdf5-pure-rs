use crate::error::{Error, Result};

const EXTERNAL_LINK_VERSION: u8 = 0;
const EXTERNAL_LINK_FLAGS_ALL: u8 = 0;
const LINK_ALL_FLAGS: u8 = 0x1f;

/// Link type values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkType {
    Hard,
    Soft,
    External,
    UserDefined(u8),
}

/// A parsed Link message (type 0x0006).
#[derive(Debug, Clone)]
pub struct LinkMessage {
    /// Link name.
    pub name: String,
    /// Link type.
    pub link_type: LinkType,
    /// Creation order (if tracked).
    pub creation_order: Option<u64>,
    /// Character encoding (0=ASCII, 1=UTF-8).
    pub char_encoding: u8,
    /// For hard links: object header address.
    pub hard_link_addr: Option<u64>,
    /// For soft links: target path.
    pub soft_link_target: Option<String>,
    /// For external links: (filename, obj_path).
    pub external_link: Option<(String, String)>,
}

impl LinkMessage {
    /// Decode a link message from raw bytes.
    pub fn decode(data: &[u8], sizeof_addr: u8) -> Result<Self> {
        Self::decode_impl(data, sizeof_addr)
    }

    fn decode_impl(data: &[u8], sizeof_addr: u8) -> Result<Self> {
        let mut pos = 0;

        // Version
        let version = read_u8(data, &mut pos, "link message version")?;
        if version != 1 {
            return Err(Error::InvalidFormat(format!(
                "link message version {version}"
            )));
        }

        // Flags
        let flags = read_u8(data, &mut pos, "link message flags")?;
        if flags & !LINK_ALL_FLAGS != 0 {
            return Err(Error::InvalidFormat(format!(
                "link message flags {flags:#x} are invalid"
            )));
        }

        let size_of_len_of_link_name = flags & 0x03; // 2 bits
        let has_creation_order = flags & 0x04 != 0;
        let has_link_type = flags & 0x08 != 0;
        let has_char_encoding = flags & 0x10 != 0;

        // Link type (optional)
        let link_type = if has_link_type {
            let t = read_u8(data, &mut pos, "link message link type")?;
            match t {
                0 => LinkType::Hard,
                1 => LinkType::Soft,
                64 => LinkType::External,
                65..=u8::MAX => LinkType::UserDefined(t),
                other => {
                    return Err(Error::InvalidFormat(format!("invalid link type {other}")));
                }
            }
        } else {
            LinkType::Hard // default
        };

        // Creation order (optional)
        let creation_order = if has_creation_order {
            let val = read_le_u64(data, &mut pos, 8, "link message creation order")?;
            Some(val)
        } else {
            None
        };

        // Character encoding (optional)
        let char_encoding = if has_char_encoding {
            read_u8(data, &mut pos, "link message character encoding")?
        } else {
            0 // ASCII
        };
        if char_encoding > 1 {
            return Err(Error::InvalidFormat(format!(
                "invalid link character encoding {char_encoding}"
            )));
        }

        // Length of link name
        let name_len_size = 1 << size_of_len_of_link_name; // 1, 2, 4, or 8
        let name_len = read_le_u64(data, &mut pos, name_len_size, "link message name length")?
            .try_into()
            .map_err(|_| Error::InvalidFormat("link name length overflows usize".into()))?;
        if name_len == 0 {
            return Err(Error::InvalidFormat("invalid link name length".into()));
        }

        // Link name
        let name = decode_utf8_text(
            checked_window(data, pos, name_len, "link name")?,
            "link name",
        )?
        .to_owned();
        advance_pos(&mut pos, name_len, "link name")?;

        // Link value based on type
        let mut hard_link_addr = None;
        let mut soft_link_target = None;
        let mut external_link = None;

        match link_type {
            LinkType::Hard => {
                hard_link_addr = Some(read_le_u64(
                    data,
                    &mut pos,
                    usize::from(sizeof_addr),
                    "hard link address",
                )?);
            }
            LinkType::Soft => {
                let target_len =
                    usize::try_from(read_le_u64(data, &mut pos, 2, "soft link target length")?)
                        .map_err(|_| {
                            Error::InvalidFormat("soft link target length overflow".into())
                        })?;
                if target_len == 0 {
                    return Err(Error::InvalidFormat("invalid soft link length".into()));
                }
                soft_link_target = Some(
                    decode_utf8_text(
                        checked_window(data, pos, target_len, "soft link target")?,
                        "soft link target",
                    )?
                    .to_owned(),
                );
                advance_pos(&mut pos, target_len, "soft link target")?;
            }
            LinkType::External => {
                let info_len =
                    usize::try_from(read_le_u64(data, &mut pos, 2, "external link info length")?)
                        .map_err(|_| {
                        Error::InvalidFormat("external link info length overflow".into())
                    })?;
                if info_len < 3 {
                    return Err(Error::InvalidFormat(
                        "external link info is too short".into(),
                    ));
                }
                let (filename, obj_path) = unpack_external_link_value(checked_window(
                    data,
                    pos,
                    info_len,
                    "external link info",
                )?)?;
                trace_external_link_resolve(filename, obj_path);

                external_link = Some((filename.to_owned(), obj_path.to_owned()));
                advance_pos(&mut pos, info_len, "external link info")?;
            }
            LinkType::UserDefined(_) => {
                // Skip user-defined link data
            }
        }
        Ok(LinkMessage {
            name,
            link_type,
            creation_order,
            char_encoding,
            hard_link_addr,
            soft_link_target,
            external_link,
        })
    }
}

#[cfg(feature = "tracehash")]
fn trace_external_link_resolve(filename: &str, obj_path: &str) {
    let mut th = tracehash::th_call!("hdf5.external_link.resolve");
    th.input_bytes(filename.as_bytes());
    th.input_bytes(obj_path.as_bytes());
    th.output_value(&(true));
    th.output_value(filename.as_bytes());
    th.output_value(obj_path.as_bytes());
    th.finish();
}

#[cfg(not(feature = "tracehash"))]
fn trace_external_link_resolve(_filename: &str, _obj_path: &str) {}

fn ensure_available(data: &[u8], pos: usize, len: usize, context: &str) -> Result<()> {
    let end = pos
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} length overflow")))?;
    if end > data.len() {
        return Err(Error::InvalidFormat(format!("{context} is truncated")));
    }
    Ok(())
}

fn read_u8(data: &[u8], pos: &mut usize, context: &str) -> Result<u8> {
    ensure_available(data, *pos, 1, context)?;
    let value = data[*pos];
    advance_pos(pos, 1, context)?;
    Ok(value)
}

fn read_le_u64(data: &[u8], pos: &mut usize, size: usize, context: &str) -> Result<u64> {
    if !(1..=8).contains(&size) {
        return Err(Error::InvalidFormat(format!(
            "{context} has invalid byte width {size}"
        )));
    }
    let bytes = checked_window(data, *pos, size, context)?;
    let mut val = 0u64;
    for (i, byte) in bytes.iter().enumerate() {
        val |= u64::from(*byte) << (i * 8);
    }
    advance_pos(pos, size, context)?;
    Ok(val)
}

fn checked_window<'a>(data: &'a [u8], pos: usize, len: usize, context: &str) -> Result<&'a [u8]> {
    let end = checked_end(pos, len, context)?;
    data.get(pos..end)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))
}

fn checked_end(pos: usize, len: usize, context: &str) -> Result<usize> {
    pos.checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))
}

fn advance_pos(pos: &mut usize, len: usize, context: &str) -> Result<()> {
    *pos = checked_end(*pos, len, context)?;
    Ok(())
}

fn decode_utf8_text<'a>(bytes: &'a [u8], context: &str) -> Result<&'a str> {
    std::str::from_utf8(bytes).map_err(|_| Error::InvalidFormat(format!("{context} is not UTF-8")))
}

fn unpack_external_link_value(data: &[u8]) -> Result<(&str, &str)> {
    if data.is_empty() {
        return Err(Error::InvalidFormat(
            "not a valid external link buffer".into(),
        ));
    }

    let header = data[0];
    let ext_version = (header >> 4) & 0x0f;
    let ext_flags = header & 0x0f;
    if ext_version > EXTERNAL_LINK_VERSION {
        return Err(Error::InvalidFormat(format!(
            "external link version {ext_version} exceeds supported version {EXTERNAL_LINK_VERSION}"
        )));
    }
    if ext_flags & !EXTERNAL_LINK_FLAGS_ALL != 0 {
        return Err(Error::InvalidFormat(format!(
            "external link flags {ext_flags:#x} are invalid"
        )));
    }
    if data.len() <= 2 {
        return Err(Error::InvalidFormat(
            "not a valid external link buffer".into(),
        ));
    }
    if data[data.len() - 1] != 0 {
        return Err(Error::InvalidFormat(
            "external link buffer is not NULL-terminated".into(),
        ));
    }

    let payload = &data[1..];
    let filename_end = payload.iter().position(|&b| b == 0).ok_or_else(|| {
        Error::InvalidFormat("external link buffer does not contain an object path".into())
    })?;
    let obj_start = filename_end
        .checked_add(1)
        .ok_or_else(|| Error::InvalidFormat("external link filename offset overflow".into()))?;
    if obj_start >= payload.len() {
        return Err(Error::InvalidFormat(
            "external link buffer does not contain an object path".into(),
        ));
    }
    let obj_payload = payload.get(obj_start..).ok_or_else(|| {
        Error::InvalidFormat("external link buffer does not contain an object path".into())
    })?;
    let obj_end = obj_payload.iter().position(|&b| b == 0).ok_or_else(|| {
        Error::InvalidFormat("external link buffer does not contain an object path".into())
    })?;
    if obj_end == 0 {
        return Err(Error::InvalidFormat(
            "external link buffer does not contain an object path".into(),
        ));
    }

    let filename = decode_utf8_text(&payload[..filename_end], "external link filename")?;
    let obj_path = decode_utf8_text(&obj_payload[..obj_end], "external link object path")?;
    Ok((filename, obj_path))
}
