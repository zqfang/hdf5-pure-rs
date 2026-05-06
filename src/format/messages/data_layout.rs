use crate::error::{Error, Result};

const MAX_LAYOUT_RANK: usize = 32;

/// Storage layout type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutClass {
    Compact,    // 0
    Contiguous, // 1
    Chunked,    // 2
    Virtual,    // 3
}

/// Chunk index type (v4+ layout).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkIndexType {
    BTreeV1,         // 0 - legacy
    SingleChunk,     // 1
    Implicit,        // 2
    FixedArray,      // 3
    ExtensibleArray, // 4
    BTreeV2,         // 5
}

struct DecodedV4ChunkPrelude {
    flags: u8,
    dims: Vec<u64>,
}

/// Parsed Data Layout message (type 0x0008).
#[derive(Debug, Clone)]
pub struct DataLayoutMessage {
    pub version: u8,
    pub layout_class: LayoutClass,
    /// For compact: raw data bytes.
    pub compact_data: Option<Vec<u8>>,
    /// For contiguous: data address.
    pub contiguous_addr: Option<u64>,
    /// For contiguous: data size.
    pub contiguous_size: Option<u64>,
    /// For chunked: chunk dimensions.
    pub chunk_dims: Option<Vec<u64>>,
    /// For chunked (v3): address of the chunk index (B-tree).
    pub chunk_index_addr: Option<u64>,
    /// For chunked (v4+): chunk index type.
    pub chunk_index_type: Option<ChunkIndexType>,
    /// For chunked: data element size stored in chunk dims (v1/v2).
    pub chunk_element_size: Option<u32>,
    /// For chunked (v4): flags.
    pub chunk_flags: Option<u8>,
    /// For chunked (v4): encoded chunk dimensions.
    pub chunk_encoded_dims: Option<Vec<u64>>,
    /// For single chunk (v4): filtered chunk size.
    pub single_chunk_filtered_size: Option<u64>,
    /// For single chunk (v4): filter mask.
    pub single_chunk_filter_mask: Option<u32>,
    /// For single chunk or contiguous with address.
    pub data_addr: Option<u64>,
    /// For virtual datasets: address of the global heap storing virtual mapping.
    pub virtual_heap_addr: Option<u64>,
    /// For virtual datasets: index into the global heap.
    pub virtual_heap_index: Option<u32>,
}

impl DataLayoutMessage {
    pub fn decode(data: &[u8], sizeof_addr: u8, sizeof_size: u8) -> Result<Self> {
        let result = Self::decode_impl(data, sizeof_addr, sizeof_size);

        #[cfg(feature = "tracehash")]
        if let Ok(message) = &result {
            let mut th = tracehash::th_call!("hdf5.data_layout.decode");
            th.input_bytes(data);
            th.output_value(&(true));
            th.output_u64(u64::from(message.version));
            th.output_u64(layout_class_trace_value(message.layout_class));
            th.output_u64(
                message
                    .chunk_index_type
                    .map(chunk_index_trace_value)
                    .unwrap_or(0),
            );
            th.finish();
        }

        result
    }

    fn decode_impl(data: &[u8], sizeof_addr: u8, sizeof_size: u8) -> Result<Self> {
        if data.is_empty() {
            return Err(Error::InvalidFormat("empty data layout message".into()));
        }

        let version = data[0];
        let sa = usize::from(sizeof_addr);
        let ss = usize::from(sizeof_size);
        match version {
            1 | 2 => Self::decode_v1_v2(data, version, sa),
            3 | 4 => Self::decode_v3_v4(data, version, sa, ss),
            _ => Err(Error::InvalidFormat(format!(
                "data layout message version {version}"
            ))),
        }
    }

    fn decode_v1_v2(data: &[u8], version: u8, sizeof_addr: usize) -> Result<Self> {
        ensure_available(data, 0, 8, "data layout v1/v2 header")?;
        let ndims = usize::from(data[1]);
        if ndims == 0 {
            return Err(Error::InvalidFormat(
                "data layout v1/v2 rank must be positive".into(),
            ));
        }
        if ndims > MAX_LAYOUT_RANK {
            return Err(Error::InvalidFormat(format!(
                "data layout rank {ndims} exceeds supported maximum {MAX_LAYOUT_RANK}"
            )));
        }
        // `H5O__layout_decode` only skips these reserved bytes after the
        // header availability check.
        ensure_available(data, 3, 5, "data layout v1/v2 reserved bytes")?;

        let layout_class = decode_layout_class(data[2], false)?;
        let mut pos = 8;
        let data_addr = if layout_class != LayoutClass::Compact {
            Some(read_le_u64(
                data,
                &mut pos,
                sizeof_addr,
                "data layout v1/v2 address",
            )?)
        } else {
            None
        };

        let mut dims = Vec::with_capacity(ndims);
        for _ in 0..ndims {
            dims.push(u64::from(read_u32_le(
                data,
                &mut pos,
                "data layout v1/v2 dimensions",
            )?));
        }

        let mut result = Self::empty(version, layout_class);
        result.data_addr = data_addr;

        match layout_class {
            LayoutClass::Compact => {
                let compact_size = read_u32_len(data, &mut pos, "data layout v1/v2 compact size")?;
                ensure_available(data, pos, compact_size, "data layout v1/v2 compact data")?;
                let compact_end = checked_end(pos, compact_size, "data layout v1/v2 compact data")?;
                result.compact_data = Some(data[pos..compact_end].to_vec());
            }
            LayoutClass::Contiguous => {
                result.contiguous_addr = data_addr;
            }
            LayoutClass::Chunked => {
                if dims.len() < 2 {
                    return Err(Error::InvalidFormat(
                        "data layout v1/v2 chunk rank must be at least 2".into(),
                    ));
                }
                result.chunk_index_addr = data_addr;
                if let Some(&last) = dims.last() {
                    if last == 0 {
                        return Err(Error::InvalidFormat(
                            "data layout v1/v2 chunk element size must be positive".into(),
                        ));
                    }
                    let chunk_dims = &dims[..dims.len() - 1];
                    validate_chunk_dims_positive(chunk_dims, "data layout v1/v2")?;
                    result.chunk_element_size = Some(u32::try_from(last).map_err(|_| {
                        Error::InvalidFormat(
                            "data layout v1/v2 chunk element size exceeds u32".into(),
                        )
                    })?);
                    result.chunk_dims = Some(chunk_dims.to_vec());
                }
            }
            LayoutClass::Virtual => unreachable!(),
        }

        Ok(result)
    }

    fn decode_v3_v4(
        data: &[u8],
        version: u8,
        sizeof_addr: usize,
        sizeof_size: usize,
    ) -> Result<Self> {
        ensure_available(data, 0, 2, "data layout v3/v4 header")?;
        let layout_class = decode_layout_class(data[1], true)?;
        let mut pos = 2;
        let mut result = Self::empty(version, layout_class);

        match layout_class {
            LayoutClass::Compact => {
                Self::decode_v3_v4_compact_layout(data, &mut pos, version, &mut result)?
            }
            LayoutClass::Contiguous => Self::decode_v3_v4_contiguous_layout(
                data,
                &mut pos,
                version,
                sizeof_addr,
                sizeof_size,
                &mut result,
            )?,
            LayoutClass::Chunked => Self::decode_v3_v4_chunked_layout(
                data,
                &mut pos,
                version,
                sizeof_addr,
                sizeof_size,
                &mut result,
            )?,
            LayoutClass::Virtual => Self::decode_v3_v4_virtual_layout(
                data,
                &mut pos,
                version,
                sizeof_addr,
                &mut result,
            )?,
        }

        Ok(result)
    }

    fn decode_v3_v4_compact_layout(
        data: &[u8],
        pos: &mut usize,
        version: u8,
        result: &mut Self,
    ) -> Result<()> {
        let context = if version == 3 {
            "data layout v3 compact size"
        } else {
            "data layout v4 compact size"
        };
        let size = usize::from(read_u16_le(data, pos, context)?);
        let context = if version == 3 {
            "data layout v3 compact data"
        } else {
            "data layout v4 compact data"
        };
        ensure_available(data, *pos, size, context)?;
        let end = checked_end(*pos, size, context)?;
        result.compact_data = Some(data[*pos..end].to_vec());
        advance_pos(pos, size, context)?;
        Ok(())
    }

    fn decode_v3_v4_contiguous_layout(
        data: &[u8],
        pos: &mut usize,
        version: u8,
        sizeof_addr: usize,
        sizeof_size: usize,
        result: &mut Self,
    ) -> Result<()> {
        let addr = read_le_u64(
            data,
            pos,
            sizeof_addr,
            if version == 3 {
                "data layout v3 contiguous address"
            } else {
                "data layout v4 contiguous address"
            },
        )?;
        let size = read_le_u64(
            data,
            pos,
            sizeof_size,
            if version == 3 {
                "data layout v3 contiguous size"
            } else {
                "data layout v4 contiguous size"
            },
        )?;
        result.contiguous_addr = Some(addr);
        result.contiguous_size = Some(size);
        result.data_addr = Some(addr);
        Ok(())
    }

    fn decode_v3_v4_chunked_layout(
        data: &[u8],
        pos: &mut usize,
        version: u8,
        sizeof_addr: usize,
        sizeof_size: usize,
        result: &mut Self,
    ) -> Result<()> {
        if version == 3 {
            Self::decode_v3_chunked_layout(data, pos, sizeof_addr, result)
        } else {
            Self::decode_v4_chunked_layout(data, pos, sizeof_addr, sizeof_size, result)
        }
    }

    fn decode_v3_v4_virtual_layout(
        data: &[u8],
        pos: &mut usize,
        version: u8,
        sizeof_addr: usize,
        result: &mut Self,
    ) -> Result<()> {
        if version < 4 {
            return Err(Error::InvalidFormat(
                "data layout virtual layout requires version 4".into(),
            ));
        }
        let addr = read_le_u64(
            data,
            pos,
            sizeof_addr,
            if version == 3 {
                "data layout v3 virtual heap address"
            } else {
                "data layout v4 virtual heap address"
            },
        )?;
        let index = read_u32_le(
            data,
            pos,
            if version == 3 {
                "data layout v3 virtual heap index"
            } else {
                "data layout v4 virtual heap index"
            },
        )?;
        result.virtual_heap_addr = Some(addr);
        result.virtual_heap_index = Some(index);
        Ok(())
    }

    fn decode_v3_chunked_layout(
        data: &[u8],
        pos: &mut usize,
        sizeof_addr: usize,
        result: &mut Self,
    ) -> Result<()> {
        let ndims = usize::from(read_u8(data, pos, "data layout v3 chunk rank")?);
        if ndims < 2 {
            return Err(Error::InvalidFormat(
                "data layout v3 chunk rank must be at least 2".into(),
            ));
        }
        if ndims > MAX_LAYOUT_RANK {
            return Err(Error::InvalidFormat(format!(
                "data layout v3 chunk rank {ndims} exceeds supported maximum {MAX_LAYOUT_RANK}"
            )));
        }
        let addr = read_le_u64(data, pos, sizeof_addr, "data layout v3 chunk index address")?;

        let mut dims = Vec::with_capacity(ndims);
        for _ in 0..ndims {
            dims.push(u64::from(read_u32_le(
                data,
                pos,
                "data layout v3 chunk dimensions",
            )?));
        }
        if let Some(&last) = dims.last() {
            if last == 0 {
                return Err(Error::InvalidFormat(
                    "data layout v3 chunk element size must be positive".into(),
                ));
            }
            let chunk_dims = &dims[..dims.len() - 1];
            validate_chunk_dims_positive(chunk_dims, "data layout v3")?;
            result.chunk_element_size = Some(u32::try_from(last).map_err(|_| {
                Error::InvalidFormat("data layout v3 chunk element size exceeds u32".into())
            })?);
            result.chunk_dims = Some(chunk_dims.to_vec());
        }
        result.chunk_index_addr = Some(addr);
        result.data_addr = Some(addr);
        Ok(())
    }

    fn decode_v4_chunked_layout(
        data: &[u8],
        pos: &mut usize,
        sizeof_addr: usize,
        sizeof_size: usize,
        result: &mut Self,
    ) -> Result<()> {
        let prelude = Self::decode_v4_chunked_prelude(data, pos)?;
        let flags = prelude.flags;
        let dims = prelude.dims;
        validate_chunk_dims_positive(&dims, "data layout v4")?;
        result.chunk_encoded_dims = Some(dims.clone());
        result.chunk_dims = Some(dims);
        result.chunk_flags = Some(flags);

        let idx_type = Self::decode_v4_chunk_index_type(data, pos)?;
        result.chunk_index_type = Some(idx_type);

        match idx_type {
            ChunkIndexType::SingleChunk => Self::decode_v4_single_chunk_layout(
                data,
                pos,
                sizeof_addr,
                sizeof_size,
                flags,
                result,
            )?,
            ChunkIndexType::Implicit => {
                Self::decode_v4_implicit_chunk_layout(data, pos, sizeof_addr, result)?
            }
            ChunkIndexType::FixedArray => {
                Self::decode_v4_fixed_array_chunk_layout(data, pos, sizeof_addr, result)?
            }
            ChunkIndexType::ExtensibleArray => {
                Self::decode_v4_extensible_array_chunk_layout(data, pos, sizeof_addr, result)?
            }
            ChunkIndexType::BTreeV2 => {
                Self::decode_v4_btree2_chunk_layout(data, pos, sizeof_addr, result)?
            }
            ChunkIndexType::BTreeV1 => {
                return Err(Error::InvalidFormat(
                    "data layout v4 must not use B-tree v1 chunk indexing".into(),
                ));
            }
        }

        Ok(())
    }

    fn decode_v4_chunked_prelude(data: &[u8], pos: &mut usize) -> Result<DecodedV4ChunkPrelude> {
        let flags = read_u8(data, pos, "data layout v4 chunk flags")?;
        if flags != 0 && flags != 0x02 {
            return Err(Error::InvalidFormat(format!(
                "data layout v4 chunk flags {flags:#x} are invalid"
            )));
        }
        let ndims = usize::from(read_u8(data, pos, "data layout v4 chunk rank")?);
        if ndims == 0 {
            return Err(Error::InvalidFormat(
                "data layout v4 chunk rank must be positive".into(),
            ));
        }
        if ndims > MAX_LAYOUT_RANK {
            return Err(Error::InvalidFormat(format!(
                "data layout v4 chunk rank {ndims} exceeds supported maximum {MAX_LAYOUT_RANK}"
            )));
        }
        let enc_bytes_per_dim =
            usize::from(read_u8(data, pos, "data layout v4 encoded dimension size")?);
        if enc_bytes_per_dim == 0 || enc_bytes_per_dim > 8 {
            return Err(Error::InvalidFormat(format!(
                "data layout v4 encoded dimension size {enc_bytes_per_dim} is invalid"
            )));
        }

        let mut dims = Vec::with_capacity(ndims);
        for _ in 0..ndims {
            dims.push(read_le_u64(
                data,
                pos,
                enc_bytes_per_dim,
                "data layout v4 chunk dimensions",
            )?);
        }
        Ok(DecodedV4ChunkPrelude { flags, dims })
    }

    fn decode_v4_chunk_index_type(data: &[u8], pos: &mut usize) -> Result<ChunkIndexType> {
        match read_u8(data, pos, "data layout v4 chunk index type")? {
            0 => Ok(ChunkIndexType::BTreeV1),
            1 => Ok(ChunkIndexType::SingleChunk),
            2 => Ok(ChunkIndexType::Implicit),
            3 => Ok(ChunkIndexType::FixedArray),
            4 => Ok(ChunkIndexType::ExtensibleArray),
            5 => Ok(ChunkIndexType::BTreeV2),
            idx_type_val => Err(Error::InvalidFormat(format!(
                "invalid chunk index type {idx_type_val}"
            ))),
        }
    }

    fn decode_v4_single_chunk_layout(
        data: &[u8],
        pos: &mut usize,
        sizeof_addr: usize,
        sizeof_size: usize,
        flags: u8,
        result: &mut Self,
    ) -> Result<()> {
        if flags & 0x02 != 0 {
            let filtered_size = read_le_u64(
                data,
                pos,
                sizeof_size,
                "data layout v4 single chunk filtered size",
            )?;
            let filter_mask = read_u32_le(data, pos, "data layout v4 single chunk mask")?;
            result.single_chunk_filtered_size = Some(filtered_size);
            result.single_chunk_filter_mask = Some(filter_mask);
        }
        let addr = read_le_u64(
            data,
            pos,
            sizeof_addr,
            "data layout v4 single chunk address",
        )?;
        result.chunk_index_addr = Some(addr);
        result.data_addr = Some(addr);
        Ok(())
    }

    fn decode_v4_implicit_chunk_layout(
        data: &[u8],
        pos: &mut usize,
        sizeof_addr: usize,
        result: &mut Self,
    ) -> Result<()> {
        let addr = read_le_u64(data, pos, sizeof_addr, "data layout v4 implicit address")?;
        result.chunk_index_addr = Some(addr);
        result.data_addr = Some(addr);
        Ok(())
    }

    fn decode_v4_fixed_array_chunk_layout(
        data: &[u8],
        pos: &mut usize,
        sizeof_addr: usize,
        result: &mut Self,
    ) -> Result<()> {
        let page_bits = read_u8(data, pos, "data layout v4 fixed array page bits")?;
        if page_bits == 0 {
            return Err(Error::InvalidFormat(
                "data layout v4 fixed array page bits must be positive".into(),
            ));
        }
        let addr = read_le_u64(data, pos, sizeof_addr, "data layout v4 fixed array address")?;
        result.chunk_index_addr = Some(addr);
        result.data_addr = Some(addr);
        Ok(())
    }

    fn decode_v4_extensible_array_chunk_layout(
        data: &[u8],
        pos: &mut usize,
        sizeof_addr: usize,
        result: &mut Self,
    ) -> Result<()> {
        for context in [
            "data layout v4 extensible array max elements bits",
            "data layout v4 extensible array index block elements",
            "data layout v4 extensible array super block min data pointers",
            "data layout v4 extensible array data block min elements",
            "data layout v4 extensible array max data block page elements bits",
        ] {
            if read_u8(data, pos, context)? == 0 {
                return Err(Error::InvalidFormat(format!("{context} must be positive")));
            }
        }
        let addr = read_le_u64(
            data,
            pos,
            sizeof_addr,
            "data layout v4 extensible array address",
        )?;
        result.chunk_index_addr = Some(addr);
        result.data_addr = Some(addr);
        Ok(())
    }

    fn decode_v4_btree2_chunk_layout(
        data: &[u8],
        pos: &mut usize,
        sizeof_addr: usize,
        result: &mut Self,
    ) -> Result<()> {
        let node_size = read_u32_le(data, pos, "data layout v4 btree2 node size")?;
        if node_size == 0 {
            return Err(Error::InvalidFormat(
                "data layout v4 btree2 node size must be positive".into(),
            ));
        }
        let split_percent = read_u8(data, pos, "data layout v4 btree2 split percent")?;
        let merge_percent = read_u8(data, pos, "data layout v4 btree2 merge percent")?;
        if split_percent == 0 || split_percent > 100 {
            return Err(Error::InvalidFormat(format!(
                "data layout v4 btree2 split percent {split_percent} must be in 1..=100"
            )));
        }
        if merge_percent == 0 || merge_percent > 100 {
            return Err(Error::InvalidFormat(format!(
                "data layout v4 btree2 merge percent {merge_percent} must be in 1..=100"
            )));
        }
        let addr = read_le_u64(data, pos, sizeof_addr, "data layout v4 btree2 address")?;
        result.chunk_index_addr = Some(addr);
        result.data_addr = Some(addr);
        Ok(())
    }

    fn empty(version: u8, layout_class: LayoutClass) -> Self {
        Self {
            version,
            layout_class,
            compact_data: None,
            contiguous_addr: None,
            contiguous_size: None,
            chunk_dims: None,
            chunk_index_addr: None,
            chunk_index_type: None,
            chunk_element_size: None,
            chunk_flags: None,
            chunk_encoded_dims: None,
            single_chunk_filtered_size: None,
            single_chunk_filter_mask: None,
            data_addr: None,
            virtual_heap_addr: None,
            virtual_heap_index: None,
        }
    }
}

fn decode_layout_class(layout_class_val: u8, allow_virtual: bool) -> Result<LayoutClass> {
    match layout_class_val {
        0 => Ok(LayoutClass::Compact),
        1 => Ok(LayoutClass::Contiguous),
        2 => Ok(LayoutClass::Chunked),
        3 if allow_virtual => Ok(LayoutClass::Virtual),
        _ => Err(Error::InvalidFormat(format!(
            "unknown layout class {layout_class_val}"
        ))),
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

fn read_u8(data: &[u8], pos: &mut usize, context: &str) -> Result<u8> {
    ensure_available(data, *pos, 1, context)?;
    let value = data[*pos];
    advance_pos(pos, 1, context)?;
    Ok(value)
}

fn read_u16_le(data: &[u8], pos: &mut usize, context: &str) -> Result<u16> {
    ensure_available(data, *pos, 2, context)?;
    let end = checked_end(*pos, 2, context)?;
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

fn read_u32_le(data: &[u8], pos: &mut usize, context: &str) -> Result<u32> {
    ensure_available(data, *pos, 4, context)?;
    let end = checked_end(*pos, 4, context)?;
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

fn read_u32_len(data: &[u8], pos: &mut usize, context: &'static str) -> Result<usize> {
    usize::try_from(read_u32_le(data, pos, context)?)
        .map_err(|_| Error::InvalidFormat(format!("{context} does not fit in usize")))
}

fn read_le_u64(data: &[u8], pos: &mut usize, size: usize, context: &str) -> Result<u64> {
    if !(1..=8).contains(&size) {
        return Err(Error::InvalidFormat(format!(
            "{context} has invalid byte width {size}"
        )));
    }
    ensure_available(data, *pos, size, context)?;
    let end = checked_end(*pos, size, context)?;
    let mut val = 0u64;
    for (i, byte) in data[*pos..end].iter().enumerate() {
        val |= u64::from(*byte) << (i * 8);
    }
    advance_pos(pos, size, context)?;
    Ok(val)
}

#[cfg(feature = "tracehash")]
fn layout_class_trace_value(layout_class: LayoutClass) -> u64 {
    match layout_class {
        LayoutClass::Compact => 0,
        LayoutClass::Contiguous => 1,
        LayoutClass::Chunked => 2,
        LayoutClass::Virtual => 3,
    }
}

#[cfg(feature = "tracehash")]
fn chunk_index_trace_value(chunk_index_type: ChunkIndexType) -> u64 {
    match chunk_index_type {
        ChunkIndexType::BTreeV1 => 0,
        ChunkIndexType::SingleChunk => 1,
        ChunkIndexType::Implicit => 2,
        ChunkIndexType::FixedArray => 3,
        ChunkIndexType::ExtensibleArray => 4,
        ChunkIndexType::BTreeV2 => 5,
    }
}

fn checked_end(pos: usize, len: usize, context: &str) -> Result<usize> {
    pos.checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))
}

fn advance_pos(pos: &mut usize, len: usize, context: &str) -> Result<()> {
    *pos = checked_end(*pos, len, context)?;
    Ok(())
}

/// Reject any chunk dimension equal to zero — matches upstream
/// `H5O__layout_decode`'s "chunk dimension must be positive" check. A
/// zero-sized chunk yields no data and is a corrupted layout message.
fn validate_chunk_dims_positive(dims: &[u64], context: &str) -> Result<()> {
    for (i, &d) in dims.iter().enumerate() {
        if d == 0 {
            return Err(Error::InvalidFormat(format!(
                "{context} chunk dimension {i} must be positive (got 0)"
            )));
        }
    }
    Ok(())
}
