use crate::error::{Error, Result};

const MAX_DATASPACE_RANK: usize = 32;

/// Dataspace type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataspaceType {
    Scalar,
    Simple,
    Null,
}

/// Parsed Dataspace message (type 0x0001).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataspaceMessage {
    pub version: u8,
    pub space_type: DataspaceType,
    pub ndims: u8,
    /// Current dimension sizes.
    pub dims: Vec<u64>,
    /// Maximum dimension sizes (None means same as current).
    pub max_dims: Option<Vec<u64>>,
}

impl DataspaceMessage {
    pub fn decode(data: &[u8]) -> Result<Self> {
        Self::decode_impl(data)
    }

    fn decode_impl(data: &[u8]) -> Result<Self> {
        if data.len() < 4 {
            return Err(Error::InvalidFormat("dataspace message too short".into()));
        }

        let version = data[0];
        let ndims = data[1];
        let rank = usize::from(ndims);
        if rank > MAX_DATASPACE_RANK {
            return Err(Error::InvalidFormat(format!(
                "dataspace rank {} exceeds supported maximum {MAX_DATASPACE_RANK}",
                ndims
            )));
        }

        match version {
            1 => Self::decode_v1(data, ndims),
            2 => Self::decode_v2(data, ndims),
            _ => Err(Error::InvalidFormat(format!(
                "dataspace message version {version}"
            ))),
        }
    }

    fn decode_v1(data: &[u8], ndims: u8) -> Result<Self> {
        ensure_available(data, 0, 8, "dataspace v1 header")?;
        let flags = data[2];
        if flags & !0x01 != 0 {
            return Err(Error::Unsupported(format!(
                "dataspace v1 flags {flags:#x} are not supported"
            )));
        }
        // v1: reserved bytes [3..8], then dimensions. Upstream
        // `H5O__sdspace_decode` checks availability and skips them.
        ensure_available(data, 3, 5, "dataspace v1 reserved bytes")?;
        let has_max = flags & 0x01 != 0;

        let mut pos = 8; // skip 5 reserved bytes after version(1)+ndims(1)+flags(1)

        let rank = usize::from(ndims);
        let dims = read_dims(data, &mut pos, rank, "dataspace v1 dimensions")?;
        let max_dims = if has_max {
            let max_dims = read_dims(data, &mut pos, rank, "dataspace v1 max dimensions")?;
            validate_dims_not_greater_than_max(&dims, &max_dims)?;
            Some(max_dims)
        } else {
            None
        };

        // Permutation indices (deprecated, skip if present)
        // flags & 0x02 != 0 means permutation present

        let space_type = if ndims == 0 {
            DataspaceType::Scalar
        } else {
            DataspaceType::Simple
        };

        let message = Self {
            version: 1,
            space_type,
            ndims,
            dims,
            max_dims,
        };
        trace_dataspace_extent(data, flags, &message);
        Ok(message)
    }

    fn decode_v2(data: &[u8], ndims: u8) -> Result<Self> {
        let flags = data[2];
        if flags & !0x01 != 0 {
            return Err(Error::InvalidFormat(format!(
                "dataspace v2 flags {flags:#x} are invalid"
            )));
        }
        let space_type_val = data[3];

        let space_type = match space_type_val {
            0 => DataspaceType::Scalar,
            1 => DataspaceType::Simple,
            2 => DataspaceType::Null,
            _ => {
                return Err(Error::InvalidFormat(format!(
                    "unknown dataspace type {space_type_val}"
                )))
            }
        };

        // Scalar and Null dataspaces have no dimensions; a non-zero rank is
        // a corrupted message (matches `H5O__sdspace_decode`'s "invalid rank
        // for scalar or NULL dataspace" check).
        if matches!(space_type, DataspaceType::Scalar | DataspaceType::Null) && ndims != 0 {
            return Err(Error::InvalidFormat(format!(
                "dataspace type {space_type:?} has rank {ndims}, expected 0"
            )));
        }
        if space_type == DataspaceType::Simple && ndims == 0 {
            return Err(Error::InvalidFormat(
                "simple dataspace must have nonzero rank".into(),
            ));
        }

        let has_max = flags & 0x01 != 0;
        let mut pos = 4;

        let rank = usize::from(ndims);
        let dims = read_dims(data, &mut pos, rank, "dataspace v2 dimensions")?;
        let max_dims = if has_max {
            let max_dims = read_dims(data, &mut pos, rank, "dataspace v2 max dimensions")?;
            validate_dims_not_greater_than_max(&dims, &max_dims)?;
            Some(max_dims)
        } else {
            None
        };

        let message = Self {
            version: 2,
            space_type,
            ndims,
            dims,
            max_dims,
        };
        trace_dataspace_extent(data, flags, &message);
        Ok(message)
    }
}

#[cfg(feature = "tracehash")]
fn trace_dataspace_extent(data: &[u8], flags: u8, message: &DataspaceMessage) {
    let mut th = tracehash::th_call!("hdf5.dataspace.extent");
    th.input_bytes(data);
    th.output_value(&(true));
    th.output_u64(u64::from(message.version));
    th.output_u64(u64::from(message.ndims));
    th.output_u64(u64::from(flags));
    th.output_u64(match message.space_type {
        DataspaceType::Scalar => 0,
        DataspaceType::Simple => 1,
        DataspaceType::Null => 2,
    });
    th.output_u64(u64::try_from(message.dims.len()).unwrap_or(u64::MAX));
    for &dim in &message.dims {
        th.output_u64(dim);
    }
    th.output_value(&(message.max_dims.is_some()));
    if let Some(max_dims) = &message.max_dims {
        th.output_u64(u64::try_from(max_dims.len()).unwrap_or(u64::MAX));
        for &dim in max_dims {
            th.output_u64(dim);
        }
    }
    th.finish();
}

#[cfg(not(feature = "tracehash"))]
fn trace_dataspace_extent(_data: &[u8], _flags: u8, _message: &DataspaceMessage) {}

fn read_dims(data: &[u8], pos: &mut usize, count: usize, context: &str) -> Result<Vec<u64>> {
    let mut dims = Vec::with_capacity(count);
    for _ in 0..count {
        let val = read_le_u64(data, pos, 8, context)?;
        dims.push(val);
    }
    Ok(dims)
}

fn validate_dims_not_greater_than_max(dims: &[u64], max_dims: &[u64]) -> Result<()> {
    for (idx, (&dim, &max_dim)) in dims.iter().zip(max_dims).enumerate() {
        if dim > max_dim {
            return Err(Error::InvalidFormat(format!(
                "dataspace dimension {idx} size {dim} exceeds maximum {max_dim}"
            )));
        }
    }
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
    let end = pos
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} length overflow")))?;
    data.get(pos..end)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))
}

fn advance_pos(pos: &mut usize, len: usize, context: &str) -> Result<()> {
    *pos = pos
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))?;
    Ok(())
}
