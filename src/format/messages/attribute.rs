use crate::error::{Error, Result};
use crate::format::messages::dataspace::{DataspaceMessage, DataspaceType};
use crate::format::messages::datatype::DatatypeMessage;

const ATTRIBUTE_FLAGS_ALL: u8 = 0x03;

/// Parsed Attribute message (type 0x000C).
#[derive(Debug, Clone)]
pub struct AttributeMessage {
    pub version: u8,
    pub name: String,
    /// Character encoding for the attribute name: 0=ASCII, 1=UTF-8.
    pub char_encoding: u8,
    pub datatype: DatatypeMessage,
    pub dataspace: DataspaceMessage,
    /// Raw attribute data bytes.
    pub data: Vec<u8>,
}

impl AttributeMessage {
    pub fn decode(raw: &[u8]) -> Result<Self> {
        Self::decode_impl(raw)
    }

    fn decode_impl(raw: &[u8]) -> Result<Self> {
        if raw.len() < 6 {
            return Err(Error::InvalidFormat("attribute message too short".into()));
        }

        let version = raw[0];
        match version {
            1 => Self::decode_v1(raw),
            2 => Self::decode_v2(raw),
            3 => Self::decode_v3(raw),
            _ => Err(Error::InvalidFormat(format!(
                "attribute message version {version}"
            ))),
        }
    }

    fn decode_v1(raw: &[u8]) -> Result<Self> {
        // v1: version(1) + reserved(1) + name_size(2) + datatype_size(2) + dataspace_size(2)
        ensure_available(raw, 0, 8, "attribute v1 header")?;
        let name_size = usize::from(read_u16_le_at(raw, 2, "attribute v1 name size")?);
        let dt_size = usize::from(read_u16_le_at(raw, 4, "attribute v1 datatype size")?);
        let ds_size = usize::from(read_u16_le_at(raw, 6, "attribute v1 dataspace size")?);
        let mut pos = 8;

        // Name (null-padded to 8-byte boundary)
        let name = decode_attribute_name(raw, pos, name_size, "attribute v1 name")?;
        let name_padded = align8(name_size, "attribute v1 name")?;
        ensure_available(raw, pos, name_padded, "attribute v1 padded name")?;
        advance_pos(&mut pos, name_padded, "attribute v1 padded name")?;

        // Datatype (padded to 8-byte boundary)
        let datatype = decode_datatype_message(raw, pos, dt_size, "attribute v1 datatype")?;
        let dt_padded = align8(dt_size, "attribute v1 datatype")?;
        ensure_available(raw, pos, dt_padded, "attribute v1 padded datatype")?;
        advance_pos(&mut pos, dt_padded, "attribute v1 padded datatype")?;

        // Dataspace (padded to 8-byte boundary)
        let dataspace = decode_dataspace_message(raw, pos, ds_size, "attribute v1 dataspace")?;
        let ds_padded = align8(ds_size, "attribute v1 dataspace")?;
        ensure_available(raw, pos, ds_padded, "attribute v1 padded dataspace")?;
        advance_pos(&mut pos, ds_padded, "attribute v1 padded dataspace")?;

        // Data
        let data = checked_attribute_data(raw, pos, &datatype, &dataspace, "attribute v1 data")?;

        Ok(Self {
            version: 1,
            name,
            char_encoding: 0,
            datatype,
            dataspace,
            data: data.to_vec(),
        })
    }

    fn decode_v2(raw: &[u8]) -> Result<Self> {
        // v2: version(1) + flags(1) + name_size(2) + datatype_size(2) + dataspace_size(2)
        ensure_available(raw, 0, 8, "attribute v2 header")?;
        validate_attribute_flags(raw[1])?;
        let name_size = usize::from(read_u16_le_at(raw, 2, "attribute v2 name size")?);
        let dt_size = usize::from(read_u16_le_at(raw, 4, "attribute v2 datatype size")?);
        let ds_size = usize::from(read_u16_le_at(raw, 6, "attribute v2 dataspace size")?);
        let mut pos = 8;

        // Name (NOT padded in v2)
        let name = decode_attribute_name(raw, pos, name_size, "attribute v2 name")?;
        advance_pos(&mut pos, name_size, "attribute v2 name")?;

        // Datatype (NOT padded in v2)
        let datatype = decode_datatype_message(raw, pos, dt_size, "attribute v2 datatype")?;
        advance_pos(&mut pos, dt_size, "attribute v2 datatype")?;

        // Dataspace (NOT padded in v2)
        let dataspace = decode_dataspace_message(raw, pos, ds_size, "attribute v2 dataspace")?;
        advance_pos(&mut pos, ds_size, "attribute v2 dataspace")?;

        let data = checked_attribute_data(raw, pos, &datatype, &dataspace, "attribute v2 data")?;

        Ok(Self {
            version: 2,
            name,
            char_encoding: 0,
            datatype,
            dataspace,
            data: data.to_vec(),
        })
    }

    fn decode_v3(raw: &[u8]) -> Result<Self> {
        // v3: version(1) + flags(1) + name_size(2) + datatype_size(2) + dataspace_size(2) + encoding(1)
        ensure_available(raw, 0, 9, "attribute v3 header")?;
        validate_attribute_flags(raw[1])?;
        let name_size = usize::from(read_u16_le_at(raw, 2, "attribute v3 name size")?);
        let dt_size = usize::from(read_u16_le_at(raw, 4, "attribute v3 datatype size")?);
        let ds_size = usize::from(read_u16_le_at(raw, 6, "attribute v3 dataspace size")?);
        let encoding = raw[8]; // character encoding: 0=ASCII, 1=UTF-8
        if encoding > 1 {
            return Err(Error::InvalidFormat(format!(
                "invalid attribute character encoding {encoding}"
            )));
        }
        let mut pos = 9;

        let name = decode_attribute_name(raw, pos, name_size, "attribute v3 name")?;
        advance_pos(&mut pos, name_size, "attribute v3 name")?;

        let datatype = decode_datatype_message(raw, pos, dt_size, "attribute v3 datatype")?;
        advance_pos(&mut pos, dt_size, "attribute v3 datatype")?;

        let dataspace = decode_dataspace_message(raw, pos, ds_size, "attribute v3 dataspace")?;
        advance_pos(&mut pos, ds_size, "attribute v3 dataspace")?;

        let data = checked_attribute_data(raw, pos, &datatype, &dataspace, "attribute v3 data")?;

        Ok(Self {
            version: 3,
            name,
            char_encoding: encoding,
            datatype,
            dataspace,
            data: data.to_vec(),
        })
    }

    /// Get total number of elements.
    pub fn num_elements(&self) -> Result<u64> {
        match self.dataspace.space_type {
            DataspaceType::Null => Ok(0),
            DataspaceType::Scalar => Ok(1),
            DataspaceType::Simple => self.dataspace.dims.iter().try_fold(1u64, |acc, &dim| {
                acc.checked_mul(dim)
                    .ok_or_else(|| Error::InvalidFormat("attribute element count overflow".into()))
            }),
        }
    }

    /// Get total data size in bytes.
    pub fn data_size(&self) -> Result<usize> {
        let elements = usize::try_from(self.num_elements()?)
            .map_err(|_| Error::InvalidFormat("attribute element count overflow".into()))?;
        let datatype_size = usize::try_from(self.datatype.size)
            .map_err(|_| Error::InvalidFormat("attribute datatype size overflow".into()))?;
        elements
            .checked_mul(datatype_size)
            .ok_or_else(|| Error::InvalidFormat("attribute data size overflow".into()))
    }
}

fn validate_attribute_data_length(
    datatype: &DatatypeMessage,
    dataspace: &DataspaceMessage,
    actual_len: usize,
) -> Result<()> {
    let elements = match dataspace.space_type {
        DataspaceType::Null => 0usize,
        DataspaceType::Scalar => 1usize,
        DataspaceType::Simple => dataspace.dims.iter().try_fold(1usize, |acc, &dim| {
            let dim = usize::try_from(dim)
                .map_err(|_| Error::InvalidFormat("attribute element count overflow".into()))?;
            acc.checked_mul(dim)
                .ok_or_else(|| Error::InvalidFormat("attribute element count overflow".into()))
        })?,
    };
    let datatype_size = usize::try_from(datatype.size)
        .map_err(|_| Error::InvalidFormat("attribute datatype size overflow".into()))?;
    let expected_len = elements
        .checked_mul(datatype_size)
        .ok_or_else(|| Error::InvalidFormat("attribute data size overflow".into()))?;
    if actual_len < expected_len {
        return Err(Error::InvalidFormat(format!(
            "attribute data is truncated: expected at least {expected_len} bytes, got {actual_len}"
        )));
    }
    Ok(())
}

fn checked_attribute_data<'a>(
    raw: &'a [u8],
    pos: usize,
    datatype: &DatatypeMessage,
    dataspace: &DataspaceMessage,
    context: &str,
) -> Result<&'a [u8]> {
    let data = raw
        .get(pos..)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))?;
    validate_attribute_data_length(datatype, dataspace, data.len())?;
    Ok(data)
}

fn validate_attribute_flags(flags: u8) -> Result<()> {
    if flags & !ATTRIBUTE_FLAGS_ALL != 0 {
        return Err(Error::InvalidFormat(format!(
            "attribute message flags {flags:#x} are invalid"
        )));
    }
    Ok(())
}

fn decode_datatype_message(
    raw: &[u8],
    pos: usize,
    len: usize,
    context: &str,
) -> Result<DatatypeMessage> {
    DatatypeMessage::decode(checked_window(raw, pos, len, context)?)
}

fn decode_dataspace_message(
    raw: &[u8],
    pos: usize,
    len: usize,
    context: &str,
) -> Result<DataspaceMessage> {
    DataspaceMessage::decode(checked_window(raw, pos, len, context)?)
}

fn decode_attribute_name(
    raw: &[u8],
    pos: usize,
    name_size: usize,
    context: &str,
) -> Result<String> {
    if name_size <= 1 {
        return Err(Error::InvalidFormat(
            "attribute message name length is invalid".into(),
        ));
    }
    let name_bytes = checked_window(raw, pos, name_size, context)?;
    let name_text = &name_bytes[..name_size - 1];
    if name_text.contains(&0) {
        return Err(Error::InvalidFormat(
            "attribute name has different length than stored length".into(),
        ));
    }
    if name_bytes[name_size - 1] != 0 {
        return Err(Error::InvalidFormat(
            "attribute name has different length than stored length".into(),
        ));
    }
    std::str::from_utf8(name_text)
        .map(str::to_string)
        .map_err(|_| Error::InvalidFormat("attribute name is not UTF-8".into()))
}

fn ensure_available(raw: &[u8], pos: usize, len: usize, context: &str) -> Result<()> {
    let end = pos
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} length overflow")))?;
    if end > raw.len() {
        return Err(Error::InvalidFormat(format!("{context} is truncated")));
    }
    Ok(())
}

fn checked_window<'a>(raw: &'a [u8], pos: usize, len: usize, context: &str) -> Result<&'a [u8]> {
    let end = checked_end(pos, len, context)?;
    raw.get(pos..end)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))
}

fn read_u16_le_at(raw: &[u8], pos: usize, context: &str) -> Result<u16> {
    let bytes = checked_window(raw, pos, 2, context)?;
    let bytes: [u8; 2] = bytes
        .try_into()
        .map_err(|_| Error::InvalidFormat(format!("{context} is truncated")))?;
    Ok(u16::from_le_bytes(bytes))
}

fn advance_pos(pos: &mut usize, len: usize, context: &str) -> Result<()> {
    *pos = checked_end(*pos, len, context)?;
    Ok(())
}

fn checked_end(pos: usize, len: usize, context: &str) -> Result<usize> {
    pos.checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))
}

fn align8(len: usize, context: &str) -> Result<usize> {
    len.checked_add(7)
        .map(|value| value & !7)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} padded size overflow")))
}

#[cfg(test)]
mod tests {
    use super::{advance_pos, align8, read_u16_le_at};

    #[test]
    fn attribute_padding_rejects_overflow() {
        let err = align8(usize::MAX, "attribute").unwrap_err();
        assert!(err.to_string().contains("overflow"));
    }

    #[test]
    fn attribute_cursor_advance_rejects_overflow() {
        let mut pos = usize::MAX;
        let err = advance_pos(&mut pos, 1, "attribute").unwrap_err();
        assert!(err.to_string().contains("overflow"));
    }

    #[test]
    fn attribute_u16_reader_rejects_offset_overflow() {
        let err = read_u16_le_at(&[], usize::MAX, "attribute test field").unwrap_err();
        assert!(err.to_string().contains("overflow"));
    }
}
