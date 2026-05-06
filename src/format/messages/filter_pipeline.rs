use crate::error::{Error, Result};

const MAX_FILTERS: usize = 32;
const MAX_FILTER_CLIENT_VALUES: usize = 1024;

/// Filter IDs.
pub const FILTER_DEFLATE: u16 = 1;
pub const FILTER_SHUFFLE: u16 = 2;
pub const FILTER_FLETCHER32: u16 = 3;
pub const FILTER_SZIP: u16 = 4;
pub const FILTER_NBIT: u16 = 5;
pub const FILTER_SCALEOFFSET: u16 = 6;

/// A single filter in the pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilterDesc {
    pub id: u16,
    pub name: Option<String>,
    pub flags: u16,
    pub client_data: Vec<u32>,
}

/// Parsed Filter Pipeline message (type 0x000B).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilterPipelineMessage {
    pub version: u8,
    pub filters: Vec<FilterDesc>,
}

impl FilterPipelineMessage {
    pub fn decode(data: &[u8]) -> Result<Self> {
        let result = Self::decode_impl(data);

        #[cfg(feature = "tracehash")]
        if let Ok(message) = &result {
            let mut th = tracehash::th_call!("hdf5.filter_pipeline.decode");
            th.input_bytes(data);
            th.output_value(&(true));
            th.output_u64(u64::from(message.version));
            th.output_u64(u64::try_from(message.filters.len()).unwrap_or(u64::MAX));
            th.finish();
        }

        result
    }

    fn decode_impl(data: &[u8]) -> Result<Self> {
        if data.len() < 2 {
            return Err(Error::InvalidFormat(
                "filter pipeline message too short".into(),
            ));
        }

        let version = data[0];
        let nfilters = usize::from(data[1]);
        if nfilters > MAX_FILTERS {
            return Err(Error::InvalidFormat(format!(
                "filter pipeline has too many filters: {nfilters}"
            )));
        }

        let result = match version {
            1 => Self::decode_v1(data, nfilters),
            2 => Self::decode_v2(data, nfilters),
            _ => Err(Error::InvalidFormat(format!(
                "filter pipeline version {version}"
            ))),
        };

        result
    }

    fn decode_v1(data: &[u8], nfilters: usize) -> Result<Self> {
        // v1: version(1) + nfilters(1) + reserved(6)
        ensure_available(data, 0, 8, "filter pipeline v1 header")?;
        let mut pos = 8;
        let mut filters = Vec::with_capacity(nfilters);

        for _ in 0..nfilters {
            let id = read_u16_le(data, &mut pos, "filter pipeline v1 filter id")?;
            let name_len = usize::from(read_u16_le(
                data,
                &mut pos,
                "filter pipeline v1 name length",
            )?);
            // The v1 spec requires the name length (including null terminator
            // and 8-byte padding) to itself be a multiple of eight; matches
            // upstream `H5O__pline_decode`.
            if name_len % 8 != 0 {
                return Err(Error::InvalidFormat(format!(
                    "filter pipeline v1 name length {name_len} is not a multiple of eight"
                )));
            }
            let flags = read_u16_le(data, &mut pos, "filter pipeline v1 flags")?;
            let cd_nelmts = usize::from(read_u16_le(
                data,
                &mut pos,
                "filter pipeline v1 client data count",
            )?);
            if cd_nelmts > MAX_FILTER_CLIENT_VALUES {
                return Err(Error::InvalidFormat(format!(
                    "filter pipeline v1 client data count {cd_nelmts} exceeds supported maximum {MAX_FILTER_CLIENT_VALUES}"
                )));
            }

            // Name (null-terminated, padded to 8-byte boundary)
            let name = if name_len > 0 {
                ensure_available(data, pos, name_len, "filter pipeline v1 name")?;
                let name_bytes = checked_window(data, pos, name_len, "filter pipeline v1 name")?;
                let null_pos = name_bytes.iter().position(|&b| b == 0).ok_or_else(|| {
                    Error::InvalidFormat("filter pipeline v1 name is not null-terminated".into())
                })?;
                let n = decode_utf8_name(
                    checked_window(name_bytes, 0, null_pos, "filter pipeline v1 name text")?,
                    "filter pipeline v1 name text",
                )?;
                // Pad to 8-byte boundary
                let padded = align8(name_len, "filter pipeline v1 name")?;
                ensure_available(data, pos, padded, "filter pipeline v1 padded name")?;
                advance_pos(&mut pos, padded, "filter pipeline v1 padded name")?;
                Some(n)
            } else {
                None
            };

            // Client data values
            let mut client_data = Vec::with_capacity(cd_nelmts);
            for _ in 0..cd_nelmts {
                let val = read_u32_le(data, &mut pos, "filter pipeline v1 client data")?;
                client_data.push(val);
            }

            // Pad cd_nelmts to even number in v1
            if cd_nelmts % 2 != 0 {
                ensure_available(data, pos, 4, "filter pipeline v1 client data padding")?;
                advance_pos(&mut pos, 4, "filter pipeline v1 client data padding")?;
            }

            filters.push(FilterDesc {
                id,
                name,
                flags,
                client_data,
            });
        }
        Ok(Self {
            version: 1,
            filters,
        })
    }

    fn decode_v2(data: &[u8], nfilters: usize) -> Result<Self> {
        // v2: version(1) + nfilters(1), no reserved bytes
        let mut pos = 2;
        let mut filters = Vec::with_capacity(nfilters);

        for _ in 0..nfilters {
            let id = read_u16_le(data, &mut pos, "filter pipeline v2 filter id")?;

            // v2: name_length and name are OMITTED for known filter IDs (< 256)
            let name = if id >= 256 {
                let name_len = usize::from(read_u16_le(
                    data,
                    &mut pos,
                    "filter pipeline v2 name length",
                )?);
                if name_len > 0 {
                    ensure_available(data, pos, name_len, "filter pipeline v2 name")?;
                    let name_bytes =
                        checked_window(data, pos, name_len, "filter pipeline v2 name")?;
                    let null_pos = name_bytes.iter().position(|&b| b == 0).ok_or_else(|| {
                        Error::InvalidFormat(
                            "filter pipeline v2 name is not null-terminated".into(),
                        )
                    })?;
                    let n = decode_utf8_name(
                        checked_window(name_bytes, 0, null_pos, "filter pipeline v2 name text")?,
                        "filter pipeline v2 name text",
                    )?;
                    advance_pos(&mut pos, name_len, "filter pipeline v2 name")?;
                    Some(n)
                } else {
                    None
                }
            } else {
                None
            };

            let flags = read_u16_le(data, &mut pos, "filter pipeline v2 flags")?;
            let cd_nelmts = usize::from(read_u16_le(
                data,
                &mut pos,
                "filter pipeline v2 client data count",
            )?);
            if cd_nelmts > MAX_FILTER_CLIENT_VALUES {
                return Err(Error::InvalidFormat(format!(
                    "filter pipeline v2 client data count {cd_nelmts} exceeds supported maximum {MAX_FILTER_CLIENT_VALUES}"
                )));
            }

            let mut client_data = Vec::with_capacity(cd_nelmts);
            for _ in 0..cd_nelmts {
                let val = read_u32_le(data, &mut pos, "filter pipeline v2 client data")?;
                client_data.push(val);
            }

            filters.push(FilterDesc {
                id,
                name,
                flags,
                client_data,
            });
        }
        Ok(Self {
            version: 2,
            filters,
        })
    }
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

fn read_u16_le(data: &[u8], pos: &mut usize, context: &str) -> Result<u16> {
    ensure_available(data, *pos, 2, context)?;
    let end = checked_add_pos(*pos, 2, context)?;
    let bytes = data
        .get(*pos..end)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))?;
    let value = u16::from_le_bytes(
        bytes
            .try_into()
            .map_err(|_| Error::InvalidFormat(format!("{context} is truncated")))?,
    );
    advance_pos(pos, 2, context)?;
    Ok(value)
}

fn checked_window<'a>(data: &'a [u8], pos: usize, len: usize, context: &str) -> Result<&'a [u8]> {
    let end = checked_add_pos(pos, len, context)?;
    data.get(pos..end)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))
}

fn read_u32_le(data: &[u8], pos: &mut usize, context: &str) -> Result<u32> {
    ensure_available(data, *pos, 4, context)?;
    let end = checked_add_pos(*pos, 4, context)?;
    let bytes = data
        .get(*pos..end)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))?;
    let value = u32::from_le_bytes(
        bytes
            .try_into()
            .map_err(|_| Error::InvalidFormat(format!("{context} is truncated")))?,
    );
    advance_pos(pos, 4, context)?;
    Ok(value)
}

fn checked_add_pos(pos: usize, len: usize, context: &str) -> Result<usize> {
    pos.checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))
}

fn advance_pos(pos: &mut usize, len: usize, context: &str) -> Result<()> {
    *pos = checked_add_pos(*pos, len, context)?;
    Ok(())
}

fn decode_utf8_name(bytes: &[u8], context: &str) -> Result<String> {
    std::str::from_utf8(bytes)
        .map(str::to_string)
        .map_err(|_| Error::InvalidFormat(format!("{context} is not UTF-8")))
}

fn align8(len: usize, context: &str) -> Result<usize> {
    len.checked_add(7)
        .map(|value| value & !7)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} padded size overflow")))
}

#[cfg(test)]
mod tests {
    use super::{advance_pos, align8, checked_window};

    #[test]
    fn filter_pipeline_padding_rejects_overflow() {
        let err = align8(usize::MAX, "filter name").unwrap_err();
        assert!(err.to_string().contains("overflow"));
    }

    #[test]
    fn filter_pipeline_cursor_advance_rejects_overflow() {
        let mut pos = usize::MAX;
        let err = advance_pos(&mut pos, 1, "filter cursor").unwrap_err();
        assert!(err.to_string().contains("overflow"));
    }

    #[test]
    fn filter_pipeline_checked_window_rejects_overflow() {
        let err = checked_window(&[], usize::MAX, 1, "filter window").unwrap_err();
        assert!(err.to_string().contains("overflow"));
    }
}
