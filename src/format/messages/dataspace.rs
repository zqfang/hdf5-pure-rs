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

    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        self.encode_into(&mut out)?;
        Ok(out)
    }

    pub fn encode_into(&self, out: &mut Vec<u8>) -> Result<()> {
        self.validate_for_encode()?;

        let rank = usize::from(self.ndims);
        let has_max = self.max_dims.is_some();
        let header_len = match self.version {
            1 => 8usize,
            2 => 4usize,
            _ => {
                return Err(Error::InvalidFormat(format!(
                    "dataspace message version {}",
                    self.version
                )))
            }
        };
        let dim_vectors = 1usize + usize::from(has_max);
        let dim_bytes = rank
            .checked_mul(dim_vectors)
            .and_then(|count| count.checked_mul(8))
            .ok_or_else(|| Error::InvalidFormat("dataspace message size overflow".into()))?;
        let capacity = header_len
            .checked_add(dim_bytes)
            .ok_or_else(|| Error::InvalidFormat("dataspace message size overflow".into()))?;

        out.clear();
        if out.capacity() < capacity {
            out.reserve_exact(capacity - out.capacity());
        }

        match self.version {
            1 => {
                out.push(1);
                out.push(self.ndims);
                out.push(if has_max { 0x01 } else { 0x00 });
                out.extend_from_slice(&[0; 5]);
            }
            2 => {
                out.push(2);
                out.push(self.ndims);
                out.push(if has_max { 0x01 } else { 0x00 });
                out.push(self.space_type.encoded_v2_type());
            }
            _ => unreachable!("version was validated above"),
        }

        write_dims(out, &self.dims);
        if let Some(max_dims) = &self.max_dims {
            write_dims(out, max_dims);
        }

        Ok(())
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

    fn validate_for_encode(&self) -> Result<()> {
        if !matches!(self.version, 1 | 2) {
            return Err(Error::InvalidFormat(format!(
                "dataspace message version {}",
                self.version
            )));
        }

        let rank = usize::from(self.ndims);
        if rank > MAX_DATASPACE_RANK {
            return Err(Error::InvalidFormat(format!(
                "dataspace rank {} exceeds supported maximum {MAX_DATASPACE_RANK}",
                self.ndims
            )));
        }
        if rank != self.dims.len() {
            return Err(Error::InvalidFormat(format!(
                "dataspace rank {} does not match {} current dimensions",
                self.ndims,
                self.dims.len()
            )));
        }

        if let Some(max_dims) = &self.max_dims {
            if max_dims.len() != self.dims.len() {
                return Err(Error::InvalidFormat(format!(
                    "dataspace max rank {} does not match rank {}",
                    max_dims.len(),
                    self.dims.len()
                )));
            }
            validate_dims_not_greater_than_max(&self.dims, max_dims)?;
        }

        match self.version {
            1 => match self.space_type {
                DataspaceType::Scalar if self.ndims == 0 => Ok(()),
                DataspaceType::Simple if self.ndims != 0 => Ok(()),
                DataspaceType::Null => Err(Error::Unsupported(
                    "dataspace v1 cannot encode null dataspaces".into(),
                )),
                DataspaceType::Scalar | DataspaceType::Simple => Err(Error::InvalidFormat(
                    "dataspace v1 extent type does not match rank".into(),
                )),
            },
            2 => {
                if matches!(self.space_type, DataspaceType::Scalar | DataspaceType::Null)
                    && self.ndims != 0
                {
                    return Err(Error::InvalidFormat(format!(
                        "dataspace type {:?} has rank {}, expected 0",
                        self.space_type, self.ndims
                    )));
                }
                if self.space_type == DataspaceType::Simple && self.ndims == 0 {
                    return Err(Error::InvalidFormat(
                        "simple dataspace must have nonzero rank".into(),
                    ));
                }
                Ok(())
            }
            _ => unreachable!("version was validated above"),
        }
    }
}

impl DataspaceType {
    fn encoded_v2_type(self) -> u8 {
        match self {
            DataspaceType::Scalar => 0,
            DataspaceType::Simple => 1,
            DataspaceType::Null => 2,
        }
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
    let bytes_len = count
        .checked_mul(8)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} length overflow")))?;
    let bytes = checked_window(data, *pos, bytes_len, context)?;
    let mut dims = Vec::with_capacity(count);
    for chunk in bytes.chunks_exact(8) {
        let mut val = 0u64;
        for (i, byte) in chunk.iter().enumerate() {
            val |= u64::from(*byte) << (i * 8);
        }
        dims.push(val);
    }
    advance_pos(pos, bytes_len, context)?;
    Ok(dims)
}

fn write_dims(out: &mut Vec<u8>, dims: &[u64]) {
    for &dim in dims {
        out.extend_from_slice(&dim.to_le_bytes());
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_v2_simple_with_max_dims_matches_decode_layout() {
        let message = DataspaceMessage {
            version: 2,
            space_type: DataspaceType::Simple,
            ndims: 2,
            dims: vec![3, 5],
            max_dims: Some(vec![10, u64::MAX]),
        };

        let encoded = message.encode().unwrap();
        let mut expected = vec![2, 2, 1, 1];
        expected.extend_from_slice(&3u64.to_le_bytes());
        expected.extend_from_slice(&5u64.to_le_bytes());
        expected.extend_from_slice(&10u64.to_le_bytes());
        expected.extend_from_slice(&u64::MAX.to_le_bytes());
        assert_eq!(encoded, expected);
        assert_eq!(DataspaceMessage::decode(&encoded).unwrap(), message);
    }

    #[test]
    fn encode_v2_scalar_and_null_have_no_dimensions() {
        let scalar = DataspaceMessage {
            version: 2,
            space_type: DataspaceType::Scalar,
            ndims: 0,
            dims: Vec::new(),
            max_dims: None,
        };
        assert_eq!(scalar.encode().unwrap(), vec![2, 0, 0, 0]);

        let null = DataspaceMessage {
            version: 2,
            space_type: DataspaceType::Null,
            ndims: 0,
            dims: Vec::new(),
            max_dims: None,
        };
        assert_eq!(null.encode().unwrap(), vec![2, 0, 0, 2]);
    }

    #[test]
    fn encode_v1_simple_uses_reserved_header_bytes() {
        let message = DataspaceMessage {
            version: 1,
            space_type: DataspaceType::Simple,
            ndims: 1,
            dims: vec![7],
            max_dims: Some(vec![9]),
        };

        let encoded = message.encode().unwrap();
        let mut expected = vec![1, 1, 1, 0, 0, 0, 0, 0];
        expected.extend_from_slice(&7u64.to_le_bytes());
        expected.extend_from_slice(&9u64.to_le_bytes());
        assert_eq!(encoded, expected);
        assert_eq!(DataspaceMessage::decode(&encoded).unwrap(), message);
    }

    #[test]
    fn encode_rejects_invalid_dataspaces() {
        let simple_rank_zero = DataspaceMessage {
            version: 2,
            space_type: DataspaceType::Simple,
            ndims: 0,
            dims: Vec::new(),
            max_dims: None,
        };
        assert!(simple_rank_zero.encode().is_err());

        let null_v1 = DataspaceMessage {
            version: 1,
            space_type: DataspaceType::Null,
            ndims: 0,
            dims: Vec::new(),
            max_dims: None,
        };
        assert!(null_v1.encode().is_err());

        let dim_exceeds_max = DataspaceMessage {
            version: 2,
            space_type: DataspaceType::Simple,
            ndims: 1,
            dims: vec![11],
            max_dims: Some(vec![10]),
        };
        assert!(dim_exceeds_max.encode().is_err());
    }
}
