use crate::error::{Error, Result};

pub const FILL_TIME_NEVER: u8 = 1;
const FILL_V3_FLAGS_ALL: u8 = 0x3f;
const FILL_ALLOC_TIME_MAX: u8 = 3;
const FILL_WRITE_TIME_MAX: u8 = 2;
const FILL_DEFINED_MAX: u8 = 2;

/// Parsed Fill Value message (types 0x0004 old and 0x0005 new).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FillValueMessage {
    pub version: u8,
    /// Allocation-time policy encoded by the fill-value message.
    pub alloc_time: u8,
    /// Fill-write-time policy encoded by the fill-value message.
    pub fill_time: u8,
    /// Whether a fill value is defined.
    pub defined: bool,
    /// The raw fill value bytes (if defined).
    pub value: Option<Vec<u8>>,
}

impl FillValueMessage {
    /// Decode fill value message (type 0x0005, version 2 or 3).
    pub fn decode(data: &[u8]) -> Result<Self> {
        Self::decode_impl(data)
    }

    fn decode_impl(data: &[u8]) -> Result<Self> {
        if data.is_empty() {
            return Err(Error::InvalidFormat("empty fill value message".into()));
        }

        let version = data[0];
        match version {
            1 | 2 => Self::decode_v2(data),
            3 => Self::decode_v3(data),
            _ => Err(Error::InvalidFormat(format!(
                "fill value message version {version}"
            ))),
        }
    }

    fn decode_v2(data: &[u8]) -> Result<Self> {
        // v2: version(1) + space_alloc_time(1) + fill_write_time(1) + fill_defined(1) + [size(4) + value]
        if data.len() < 4 {
            return Err(Error::InvalidFormat("fill value v2 too short".into()));
        }

        let alloc_time = data[1];
        let fill_time = data[2];
        validate_alloc_time(alloc_time, "fill value v2 allocation time")?;
        validate_fill_time(fill_time, "fill value v2 write time")?;
        let defined_state = data[3];
        if defined_state > FILL_DEFINED_MAX {
            return Err(Error::InvalidFormat(format!(
                "fill value v2 defined state {} is invalid",
                defined_state
            )));
        }
        let defined = defined_state != 0;
        let value_payload = if defined {
            if data.len() < 8 {
                return Err(Error::InvalidFormat(
                    "fill value v2 missing value size".into(),
                ));
            }
            let size = read_u32_len_at(data, 4, "fill value v2 value size")?;
            if size > 0 {
                Some(checked_window(data, 8, size, "fill value v2 value")?)
            } else {
                None
            }
        } else {
            None
        };
        let value = value_payload.map(<[u8]>::to_vec);

        let message = Self {
            version: data[0],
            alloc_time,
            fill_time,
            defined,
            value,
        };
        trace_fill_value(
            data,
            message.version,
            alloc_time,
            fill_time,
            defined,
            &message.value,
        );
        Ok(message)
    }

    fn decode_v3(data: &[u8]) -> Result<Self> {
        // v3: version(1) + flags(1) + [size(4) + value]
        if data.len() < 2 {
            return Err(Error::InvalidFormat("fill value v3 too short".into()));
        }

        let flags = data[1];
        if flags & !FILL_V3_FLAGS_ALL != 0 {
            return Err(Error::InvalidFormat(format!(
                "fill value v3 flags {flags:#x} are invalid"
            )));
        }
        let alloc_time = flags & 0x03;
        let fill_time = (flags >> 2) & 0x03;
        validate_alloc_time(alloc_time, "fill value v3 allocation time")?;
        validate_fill_time(fill_time, "fill value v3 write time")?;
        let undefined = flags & 0x10 != 0;
        let have_value = flags & 0x20 != 0;
        if undefined && have_value {
            return Err(Error::InvalidFormat(
                "fill value v3 has both undefined and value-present flags".into(),
            ));
        }
        let defined = !undefined;

        let value_payload = if have_value {
            if data.len() < 6 {
                return Err(Error::InvalidFormat(
                    "fill value v3 missing value size".into(),
                ));
            }
            let size = read_u32_len_at(data, 2, "fill value v3 value size")?;
            if size > 0 {
                Some(checked_window(data, 6, size, "fill value v3 value")?)
            } else {
                None
            }
        } else {
            None
        };
        let value = value_payload.map(<[u8]>::to_vec);

        let message = Self {
            version: 3,
            alloc_time,
            fill_time,
            defined,
            value,
        };
        trace_fill_value(
            data,
            message.version,
            alloc_time,
            fill_time,
            defined,
            &message.value,
        );
        Ok(message)
    }

    /// Decode old-style fill value message (type 0x0004).
    pub fn decode_old(data: &[u8]) -> Result<Self> {
        Self::decode_old_with_datatype_size(data, None)
    }

    /// Decode old-style fill value message (type 0x0004), optionally
    /// validating that the payload width matches the dataset datatype size.
    pub(crate) fn decode_old_with_datatype_size(
        data: &[u8],
        datatype_size: Option<usize>,
    ) -> Result<Self> {
        if data.len() < 4 {
            return Err(Error::InvalidFormat("old fill value too short".into()));
        }

        let size = read_u32_len_at(data, 0, "old fill value size")?;
        let value_payload = if size > 0 {
            Some(checked_window(data, 4, size, "old fill value")?)
        } else {
            None
        };

        if let Some(datatype_size) = datatype_size {
            if size > 0 && size != datatype_size {
                return Err(Error::InvalidFormat(format!(
                    "old fill value size {size} does not match datatype size {datatype_size}"
                )));
            }
        }

        let value = value_payload.map(<[u8]>::to_vec);

        let message = Self {
            version: 0,
            alloc_time: 2,
            fill_time: 2,
            defined: size > 0,
            value,
        };
        trace_fill_value(data, 0, 2, 2, message.defined, &message.value);
        Ok(message)
    }
}

fn validate_alloc_time(value: u8, context: &str) -> Result<()> {
    if value > FILL_ALLOC_TIME_MAX {
        return Err(Error::InvalidFormat(format!(
            "{context} {value} is invalid"
        )));
    }
    Ok(())
}

fn validate_fill_time(value: u8, context: &str) -> Result<()> {
    if value > FILL_WRITE_TIME_MAX {
        return Err(Error::InvalidFormat(format!(
            "{context} {value} is invalid"
        )));
    }
    Ok(())
}

fn checked_end(pos: usize, len: usize, context: &str) -> Result<usize> {
    pos.checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))
}

fn checked_window<'a>(data: &'a [u8], pos: usize, len: usize, context: &str) -> Result<&'a [u8]> {
    let end = checked_end(pos, len, context)?;
    data.get(pos..end)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))
}

fn read_u32_le_at(data: &[u8], pos: usize, context: &str) -> Result<u32> {
    let bytes = checked_window(data, pos, 4, context)?;
    let bytes: [u8; 4] = bytes
        .try_into()
        .map_err(|_| Error::InvalidFormat(format!("{context} is truncated")))?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u32_len_at(data: &[u8], pos: usize, context: &'static str) -> Result<usize> {
    usize::try_from(read_u32_le_at(data, pos, context)?)
        .map_err(|_| Error::InvalidFormat(format!("{context} does not fit in usize")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_window_rejects_offset_overflow() {
        let err = checked_window(&[], usize::MAX, 1, "fill value test window").unwrap_err();
        assert!(
            err.to_string()
                .contains("fill value test window offset overflow"),
            "unexpected error: {err}"
        );
    }
}

#[cfg(feature = "tracehash")]
fn trace_fill_value(
    data: &[u8],
    version: u8,
    alloc_time: u8,
    fill_time: u8,
    defined: bool,
    value: &Option<Vec<u8>>,
) {
    let mut th = tracehash::th_call!("hdf5.fill_value.decode");
    th.input_bytes(data);
    th.output_value(&(true));
    th.output_u64(u64::from(version));
    th.output_u64(u64::from(alloc_time));
    th.output_u64(u64::from(fill_time));
    th.output_value(&(defined));
    if let Some(value) = value {
        th.output_value(&(true));
        th.output_u64(u64::try_from(value.len()).unwrap_or(u64::MAX));
        th.output_value(value);
    } else {
        th.output_value(&(false));
        th.output_u64(0);
    }
    th.finish();
}

#[cfg(not(feature = "tracehash"))]
fn trace_fill_value(
    _data: &[u8],
    _version: u8,
    _alloc_time: u8,
    _fill_time: u8,
    _defined: bool,
    _value: &Option<Vec<u8>>,
) {
}
