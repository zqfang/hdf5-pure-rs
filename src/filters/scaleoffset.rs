use crate::error::{Error, Result};

const PARM_SCALETYPE: usize = 0;
const PARM_NELMTS: usize = 2;
const PARM_CLASS: usize = 3;
const PARM_SIZE: usize = 4;
const PARM_SIGN: usize = 5;
const PARM_ORDER: usize = 6;
const PARM_FILAVAIL: usize = 7;
const PARM_FILVAL: usize = 8;

const CLS_INTEGER: u32 = 0;
const CLS_FLOAT: u32 = 1;
const SIGN_UNSIGNED: u32 = 0;
const SIGN_TWOS: u32 = 1;
const ORDER_LE: u32 = 0;
const ORDER_BE: u32 = 1;
const HEADER_LEN: usize = 21;

#[derive(Debug, Clone, Copy)]
struct Parms {
    size: usize,
    minbits: usize,
    order: u32,
}

/// Decompress HDF5 ScaleOffset-filtered chunks using the datatype-aware
/// parameters stored in the filter pipeline.
pub fn decompress(data: &[u8], client_data: &[u32]) -> Result<Vec<u8>> {
    can_apply_scaleoffset(client_data)?;

    if client_data.len() <= PARM_ORDER {
        return Err(Error::InvalidFormat(
            "scaleoffset filter missing datatype parameters".into(),
        ));
    }

    let scale_type = client_data[PARM_SCALETYPE];
    let nelmts = scaleoffset_usize(client_data[PARM_NELMTS], "scaleoffset element count")?;
    let class = client_data[PARM_CLASS];
    let size = scaleoffset_usize(client_data[PARM_SIZE], "scaleoffset datatype size")?;
    let sign = client_data[PARM_SIGN];
    let order = client_data[PARM_ORDER];

    if size == 0 {
        return Err(Error::InvalidFormat(
            "scaleoffset datatype size is zero".into(),
        ));
    }
    if size > 16 {
        return Err(Error::Unsupported(format!(
            "scaleoffset datatype size {size} exceeds 16-byte arithmetic support"
        )));
    }
    if order != ORDER_LE && order != ORDER_BE {
        return Err(Error::InvalidFormat(format!(
            "invalid scaleoffset byte order {order}"
        )));
    }
    validate_scaleoffset_datatype(class, sign, size, scale_type)?;
    if data.len() < HEADER_LEN {
        return Err(Error::InvalidFormat("scaleoffset data too short".into()));
    }

    let minbits = scaleoffset_usize(
        read_u32_le_at(data, 0, "scaleoffset minimum bit count")?,
        "scaleoffset minimum bit count",
    )?;
    validate_scaleoffset_parameters(class, sign, size, minbits)?;
    let minval_size = usize::from(data[4]);
    if minval_size > 16 {
        return Err(Error::InvalidFormat(
            "invalid scaleoffset minimum value header".into(),
        ));
    }
    let minval = read_le_u128(checked_window(
        data,
        5,
        minval_size,
        "scaleoffset minimum value header",
    )?);

    let out_len = nelmts
        .checked_mul(size)
        .ok_or_else(|| Error::InvalidFormat("scaleoffset output size overflow".into()))?;
    let mut out = vec![0u8; out_len];

    if minbits == size * 8 {
        let raw_end = HEADER_LEN.checked_add(out_len).ok_or_else(|| {
            Error::InvalidFormat("scaleoffset full-precision data too short".into())
        })?;
        let raw = data.get(HEADER_LEN..raw_end).ok_or_else(|| {
            Error::InvalidFormat("scaleoffset full-precision data too short".into())
        })?;
        out.copy_from_slice(raw);
    } else if minbits != 0 {
        let parms = Parms {
            size,
            minbits,
            order,
        };
        let mut stream = BitStream::new(&data[HEADER_LEN..]);
        for idx in 0..nelmts {
            let data_offset = idx
                .checked_mul(size)
                .ok_or_else(|| Error::InvalidFormat("scaleoffset output offset overflow".into()))?;
            decompress_atomic(&mut out, data_offset, &mut stream, parms)?;
        }
    }

    match class {
        CLS_INTEGER => {
            let fill = if client_data.get(PARM_FILAVAIL).copied().unwrap_or(0) != 0 {
                Some(read_fill_value(client_data, size, order))
            } else {
                None
            };
            postprocess_integer(&mut out, size, sign, order, minbits, minval, fill)?
        }
        CLS_FLOAT if scale_type == 0 => {
            postprocess_float(&mut out, size, order, minbits, minval, client_data)?
        }
        CLS_FLOAT => {
            return Err(Error::Unsupported(format!(
                "scaleoffset float scale type {scale_type}"
            )));
        }
        other => {
            return Err(Error::Unsupported(format!(
                "scaleoffset datatype class {other}"
            )));
        }
    }

    Ok(out)
}

pub fn can_apply_scaleoffset(client_data: &[u32]) -> Result<()> {
    if client_data.len() <= PARM_ORDER {
        return Err(Error::InvalidFormat(
            "scaleoffset filter missing datatype parameters".into(),
        ));
    }
    let scale_type = client_data[PARM_SCALETYPE];
    let class = client_data[PARM_CLASS];
    let size = scaleoffset_usize(client_data[PARM_SIZE], "scaleoffset datatype size")?;
    let sign = client_data[PARM_SIGN];
    let order = client_data[PARM_ORDER];
    if size == 0 {
        return Err(Error::InvalidFormat(
            "scaleoffset datatype size is zero".into(),
        ));
    }
    if size > 16 {
        return Err(Error::Unsupported(format!(
            "scaleoffset datatype size {size} exceeds 16-byte arithmetic support"
        )));
    }
    if order != ORDER_LE && order != ORDER_BE {
        return Err(Error::InvalidFormat(format!(
            "invalid scaleoffset byte order {order}"
        )));
    }
    validate_scaleoffset_datatype(class, sign, size, scale_type)
}

pub fn scaleoffset_convert(
    data: &mut [u8],
    element_size: usize,
    from_order: u32,
    to_order: u32,
) -> Result<()> {
    if from_order != ORDER_LE && from_order != ORDER_BE {
        return Err(Error::InvalidFormat(format!(
            "invalid scaleoffset byte order {from_order}"
        )));
    }
    if to_order != ORDER_LE && to_order != ORDER_BE {
        return Err(Error::InvalidFormat(format!(
            "invalid scaleoffset byte order {to_order}"
        )));
    }
    if element_size == 0 {
        return Err(Error::InvalidFormat(
            "scaleoffset datatype size is zero".into(),
        ));
    }
    if from_order != to_order {
        for chunk in data.chunks_exact_mut(element_size) {
            chunk.reverse();
        }
    }
    Ok(())
}

pub fn scaleoffset_log2(value: u64) -> Option<u32> {
    (value != 0).then(|| u64::BITS - 1 - value.leading_zeros())
}

pub fn filter_scaleoffset(data: &[u8], client_data: &[u32], reverse: bool) -> Result<Vec<u8>> {
    if reverse {
        decompress(data, client_data)
    } else {
        scaleoffset_compress(data, client_data)
    }
}

pub fn scaleoffset_compress(data: &[u8], client_data: &[u32]) -> Result<Vec<u8>> {
    can_apply_scaleoffset(client_data)?;
    let scale_type = client_data[PARM_SCALETYPE];
    let nelmts = scaleoffset_usize(client_data[PARM_NELMTS], "scaleoffset element count")?;
    let class = client_data[PARM_CLASS];
    let size = scaleoffset_usize(client_data[PARM_SIZE], "scaleoffset datatype size")?;
    let sign = client_data[PARM_SIGN];
    let order = client_data[PARM_ORDER];
    let expected = nelmts
        .checked_mul(size)
        .ok_or_else(|| Error::InvalidFormat("scaleoffset input size overflow".into()))?;
    if data.len() < expected {
        return Err(Error::InvalidFormat(
            "scaleoffset input data too short".into(),
        ));
    }

    let (packed, minbits, minval) = match class {
        CLS_INTEGER => scaleoffset_precompress_i(&data[..expected], size, sign, order)?,
        CLS_FLOAT if scale_type == 0 => {
            scaleoffset_precompress_fd(&data[..expected], size, order, client_data)?
        }
        CLS_FLOAT => {
            return Err(Error::Unsupported(format!(
                "scaleoffset float scale type {scale_type}"
            )));
        }
        other => {
            return Err(Error::Unsupported(format!(
                "scaleoffset datatype class {other}"
            )));
        }
    };

    let mut out = vec![0u8; HEADER_LEN];
    let minbits_u32 = u32::try_from(minbits).map_err(|_| {
        Error::InvalidFormat("scaleoffset minimum bit count does not fit in u32".into())
    })?;
    out[..4].copy_from_slice(&minbits_u32.to_le_bytes());
    out[4] = size as u8;
    write_uint(
        checked_window_mut(&mut out, 5, size, "scaleoffset minimum value header")?,
        ORDER_LE,
        minval,
    );

    if minbits == size * 8 {
        out.extend_from_slice(&data[..expected]);
    } else if minbits != 0 {
        let mut writer = BitWriter::new();
        let parms = Parms {
            size,
            minbits,
            order,
        };
        for idx in 0..nelmts {
            let data_offset = idx
                .checked_mul(size)
                .ok_or_else(|| Error::InvalidFormat("scaleoffset packed offset overflow".into()))?;
            scaleoffset_compress_one_atomic(&packed, data_offset, &mut writer, parms)?;
        }
        out.extend_from_slice(&writer.finish());
    }
    Ok(out)
}

pub fn scaleoffset_precompress_i(
    data: &[u8],
    size: usize,
    sign: u32,
    order: u32,
) -> Result<(Vec<u8>, usize, u128)> {
    validate_scaleoffset_parameters(CLS_INTEGER, sign, size, 0)?;
    if size == 0 || data.len() % size != 0 {
        return Err(Error::InvalidFormat(
            "scaleoffset integer input size mismatch".into(),
        ));
    }
    if data.is_empty() {
        return Ok((Vec::new(), 0, 0));
    }
    let values: Vec<u128> = data
        .chunks_exact(size)
        .map(|chunk| read_uint(chunk, order))
        .collect();
    let minval = *values.iter().min().unwrap_or(&0);
    let max_delta = values
        .iter()
        .map(|value| value.wrapping_sub(minval))
        .max()
        .unwrap_or(0);
    let minbits = if max_delta == 0 {
        0
    } else {
        u64::try_from(max_delta)
            .ok()
            .and_then(scaleoffset_log2)
            .and_then(|bits| usize::try_from(bits).ok())
            .and_then(|bits| bits.checked_add(1))
            .unwrap_or(size * 8)
            .min(size * 8)
    };
    let mut packed = vec![0u8; data.len()];
    for (idx, value) in values.iter().enumerate() {
        let offset = idx
            .checked_mul(size)
            .ok_or_else(|| Error::InvalidFormat("scaleoffset packed offset overflow".into()))?;
        write_uint(
            checked_window_mut(
                &mut packed,
                offset,
                size,
                "scaleoffset packed integer value",
            )?,
            order,
            value.wrapping_sub(minval),
        );
    }
    Ok((packed, minbits, minval))
}

pub fn scaleoffset_precompress_fd(
    data: &[u8],
    size: usize,
    order: u32,
    client_data: &[u32],
) -> Result<(Vec<u8>, usize, u128)> {
    if !matches!(size, 4 | 8) || data.len() % size != 0 {
        return Err(Error::InvalidFormat(
            "scaleoffset floating-point input size mismatch".into(),
        ));
    }
    if data.is_empty() {
        return Ok((Vec::new(), 0, 0));
    }
    let scale = client_data
        .get(1)
        .copied()
        .ok_or_else(|| Error::InvalidFormat("scaleoffset missing scale factor".into()))?
        .to_ne_bytes();
    let scale = i32::from_ne_bytes(scale);
    let multiplier = 10f64.powi(scale);
    let values: Vec<f64> = data
        .chunks_exact(size)
        .map(|chunk| {
            if size == 4 {
                let mut raw = [0u8; 4];
                raw.copy_from_slice(chunk);
                if order == ORDER_LE {
                    f32::from_le_bytes(raw) as f64
                } else {
                    f32::from_be_bytes(raw) as f64
                }
            } else {
                let mut raw = [0u8; 8];
                raw.copy_from_slice(chunk);
                if order == ORDER_LE {
                    f64::from_le_bytes(raw)
                } else {
                    f64::from_be_bytes(raw)
                }
            }
        })
        .collect();
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max_delta = values
        .iter()
        .map(|value| ((*value - min) * multiplier).round().max(0.0) as u128)
        .max()
        .unwrap_or(0);
    let minbits = if max_delta == 0 {
        0
    } else {
        usize::try_from(u128::BITS - max_delta.leading_zeros())
            .map_err(|_| Error::InvalidFormat("scaleoffset bit count overflow".into()))?
    }
    .min(size * 8);
    let mut packed = vec![0u8; data.len()];
    for (idx, value) in values.iter().enumerate() {
        let delta = ((*value - min) * multiplier).round().max(0.0) as u128;
        let offset = idx
            .checked_mul(size)
            .ok_or_else(|| Error::InvalidFormat("scaleoffset packed offset overflow".into()))?;
        write_uint(
            checked_window_mut(&mut packed, offset, size, "scaleoffset packed float value")?,
            order,
            delta,
        );
    }
    let minval = if size == 4 {
        (min as f32).to_bits() as u128
    } else {
        min.to_bits() as u128
    };
    Ok((packed, minbits, minval))
}

fn validate_scaleoffset_datatype(
    class: u32,
    sign: u32,
    size: usize,
    scale_type: u32,
) -> Result<()> {
    match class {
        CLS_INTEGER => {
            if sign != SIGN_UNSIGNED && sign != SIGN_TWOS {
                return Err(Error::InvalidFormat(format!(
                    "invalid scaleoffset integer sign {sign}"
                )));
            }
        }
        CLS_FLOAT => {
            if !matches!(size, 4 | 8) {
                return Err(Error::Unsupported(format!(
                    "scaleoffset floating-point size {size}"
                )));
            }
            if scale_type != 0 {
                return Err(Error::Unsupported(format!(
                    "scaleoffset float scale type {scale_type}"
                )));
            }
        }
        other => {
            return Err(Error::Unsupported(format!(
                "scaleoffset datatype class {other}"
            )));
        }
    }
    Ok(())
}

fn validate_scaleoffset_parameters(
    class: u32,
    sign: u32,
    size: usize,
    minbits: usize,
) -> Result<()> {
    if minbits > size * 8 {
        return Err(Error::InvalidFormat(
            "invalid scaleoffset minimum bit count".into(),
        ));
    }
    if class == CLS_INTEGER && sign != SIGN_UNSIGNED && sign != SIGN_TWOS {
        return Err(Error::InvalidFormat(format!(
            "invalid scaleoffset integer sign {sign}"
        )));
    }
    Ok(())
}

fn decompress_atomic(
    out: &mut [u8],
    data_offset: usize,
    stream: &mut BitStream<'_>,
    parms: Parms,
) -> Result<()> {
    let dtype_bits = parms.size * 8;
    if parms.minbits == 0 || parms.minbits > dtype_bits {
        return Err(Error::InvalidFormat(
            "invalid scaleoffset minimum bit count".into(),
        ));
    }

    if parms.order == ORDER_LE {
        let begin = parms.size - 1 - (dtype_bits - parms.minbits) / 8;
        for k in (0..=begin).rev() {
            decompress_byte(out, data_offset, k, begin, stream, parms, dtype_bits)?;
        }
    } else {
        let begin = (dtype_bits - parms.minbits) / 8;
        for k in begin..parms.size {
            decompress_byte(out, data_offset, k, begin, stream, parms, dtype_bits)?;
        }
    }
    Ok(())
}

fn scaleoffset_compress_one_atomic(
    input: &[u8],
    data_offset: usize,
    writer: &mut BitWriter,
    parms: Parms,
) -> Result<()> {
    let dtype_bits = parms.size * 8;
    if parms.minbits == 0 || parms.minbits > dtype_bits {
        return Err(Error::InvalidFormat(
            "invalid scaleoffset minimum bit count".into(),
        ));
    }
    if parms.order == ORDER_LE {
        let begin = parms.size - 1 - (dtype_bits - parms.minbits) / 8;
        for k in (0..=begin).rev() {
            scaleoffset_compress_one_byte(input, data_offset, k, begin, writer, parms, dtype_bits)?;
        }
    } else {
        let begin = (dtype_bits - parms.minbits) / 8;
        for k in begin..parms.size {
            scaleoffset_compress_one_byte(input, data_offset, k, begin, writer, parms, dtype_bits)?;
        }
    }
    Ok(())
}

fn decompress_byte(
    out: &mut [u8],
    data_offset: usize,
    k: usize,
    begin: usize,
    stream: &mut BitStream<'_>,
    parms: Parms,
    dtype_bits: usize,
) -> Result<()> {
    let bits_to_copy = if k == begin {
        8 - (dtype_bits - parms.minbits) % 8
    } else {
        8
    };
    let bits = stream.read_bits(bits_to_copy)? as u8;
    let out_idx = data_offset
        .checked_add(k)
        .ok_or_else(|| Error::InvalidFormat("scaleoffset output offset overflow".into()))?;
    if out_idx >= out.len() {
        return Err(Error::InvalidFormat(
            "scaleoffset output offset out of range".into(),
        ));
    }
    out[out_idx] = bits;
    Ok(())
}

fn scaleoffset_compress_one_byte(
    input: &[u8],
    data_offset: usize,
    k: usize,
    begin: usize,
    writer: &mut BitWriter,
    parms: Parms,
    dtype_bits: usize,
) -> Result<()> {
    let bits_to_copy = if k == begin {
        8 - (dtype_bits - parms.minbits) % 8
    } else {
        8
    };
    let idx = data_offset
        .checked_add(k)
        .ok_or_else(|| Error::InvalidFormat("scaleoffset input offset overflow".into()))?;
    let byte = *input
        .get(idx)
        .ok_or_else(|| Error::InvalidFormat("scaleoffset input offset out of range".into()))?;
    let mask = if bits_to_copy == 8 {
        0xff
    } else {
        ((1u16 << bits_to_copy) - 1) as u8
    };
    writer.write_bits(u16::from(byte & mask), bits_to_copy)
}

fn postprocess_integer(
    out: &mut [u8],
    size: usize,
    sign: u32,
    order: u32,
    minbits: usize,
    minval: u128,
    fill: Option<u128>,
) -> Result<()> {
    let fill_marker = if minbits > 0 && minbits < 128 {
        Some((1u128 << minbits) - 1)
    } else if minbits == 128 {
        Some(u128::MAX)
    } else {
        None
    };

    for chunk in out.chunks_exact_mut(size) {
        let value = read_uint(chunk, order);
        let value = if let (Some(fill), Some(marker)) = (fill, fill_marker) {
            if value == marker {
                fill
            } else {
                minval.wrapping_add(value)
            }
        } else if minbits == 0 {
            minval
        } else {
            minval.wrapping_add(value)
        };
        write_uint(chunk, order, value);

        debug_assert!(sign == SIGN_UNSIGNED || sign == SIGN_TWOS);
    }
    Ok(())
}

fn postprocess_float(
    out: &mut [u8],
    size: usize,
    order: u32,
    minbits: usize,
    minval: u128,
    client_data: &[u32],
) -> Result<()> {
    let scale = client_data
        .get(1)
        .copied()
        .ok_or_else(|| Error::InvalidFormat("scaleoffset missing scale factor".into()))?
        .to_ne_bytes();
    let scale = i32::from_ne_bytes(scale);
    let divisor = 10f64.powi(scale);
    let marker = if minbits > 0 && minbits < 128 {
        Some((1u128 << minbits) - 1)
    } else {
        None
    };
    let fill = if client_data.get(PARM_FILAVAIL).copied().unwrap_or(0) != 0 {
        Some(read_fill_value(client_data, size, order))
    } else {
        None
    };

    match size {
        4 => {
            let min = f64::from(f32::from_le_bytes(low_u32(minval).to_le_bytes()));
            let fill = fill.map(|v| f32::from_le_bytes(low_u32(v).to_le_bytes()));
            for chunk in out.chunks_exact_mut(size) {
                let packed = read_uint(chunk, order);
                let value = if let (Some(marker), Some(fill)) = (marker, fill) {
                    if packed == marker {
                        fill
                    } else {
                        (signed_low_i64(packed) as f64 / divisor + min) as f32
                    }
                } else {
                    (signed_low_i64(packed) as f64 / divisor + min) as f32
                };
                write_float32(chunk, order, value);
            }
        }
        8 => {
            let min = f64::from_le_bytes(low_u64(minval).to_le_bytes());
            let fill = fill.map(|v| f64::from_le_bytes(low_u64(v).to_le_bytes()));
            for chunk in out.chunks_exact_mut(size) {
                let packed = read_uint(chunk, order);
                let value = if let (Some(marker), Some(fill)) = (marker, fill) {
                    if packed == marker {
                        fill
                    } else {
                        signed_low_i64(packed) as f64 / divisor + min
                    }
                } else {
                    signed_low_i64(packed) as f64 / divisor + min
                };
                write_float64(chunk, order, value);
            }
        }
        _ => {
            return Err(Error::Unsupported(format!(
                "scaleoffset floating-point size {size}"
            )));
        }
    }

    Ok(())
}

fn read_uint(bytes: &[u8], order: u32) -> u128 {
    let mut value = 0u128;
    if order == ORDER_LE {
        for (idx, byte) in bytes.iter().take(16).enumerate() {
            value |= u128::from(*byte) << (idx * 8);
        }
    } else {
        for byte in bytes.iter().take(16) {
            value = (value << 8) | u128::from(*byte);
        }
    }
    value
}

fn low_u32(value: u128) -> u32 {
    let bytes = value.to_le_bytes();
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn low_u64(value: u128) -> u64 {
    let bytes = value.to_le_bytes();
    u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ])
}

fn signed_low_i64(value: u128) -> i64 {
    i64::from_le_bytes(low_u64(value).to_le_bytes())
}

fn write_uint(bytes: &mut [u8], order: u32, value: u128) {
    if order == ORDER_LE {
        for (idx, byte) in bytes.iter_mut().take(16).enumerate() {
            *byte = (value >> (idx * 8)) as u8;
        }
    } else {
        let n = bytes.len().min(16);
        for (idx, byte) in bytes.iter_mut().take(n).enumerate() {
            *byte = (value >> ((n - idx - 1) * 8)) as u8;
        }
    }
}

fn read_le_u128(bytes: &[u8]) -> u128 {
    let mut value = 0u128;
    for (idx, byte) in bytes.iter().take(16).enumerate() {
        value |= u128::from(*byte) << (idx * 8);
    }
    value
}

fn read_u32_le_at(data: &[u8], offset: usize, context: &str) -> Result<u32> {
    let bytes = checked_window(data, offset, 4, context)?;
    Ok(u32::from_le_bytes(bytes.try_into().map_err(|_| {
        Error::InvalidFormat(format!("{context} is truncated"))
    })?))
}

fn scaleoffset_usize(value: u32, context: &'static str) -> Result<usize> {
    usize::try_from(value)
        .map_err(|_| Error::InvalidFormat(format!("{context} does not fit in usize")))
}

fn checked_window<'a>(
    data: &'a [u8],
    offset: usize,
    len: usize,
    context: &str,
) -> Result<&'a [u8]> {
    let end = offset
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))?;
    data.get(offset..end)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))
}

fn checked_window_mut<'a>(
    data: &'a mut [u8],
    offset: usize,
    len: usize,
    context: &str,
) -> Result<&'a mut [u8]> {
    let end = offset
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))?;
    data.get_mut(offset..end)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))
}

fn read_fill_value(client_data: &[u32], size: usize, order: u32) -> u128 {
    let mut raw = vec![0u8; size];
    let mut pos = 0usize;
    for value in client_data.iter().skip(PARM_FILVAL) {
        let bytes = value.to_le_bytes();
        for byte in bytes {
            if pos < raw.len() {
                raw[pos] = byte;
                pos += 1;
            }
        }
    }
    read_uint(&raw, order)
}

fn write_float32(bytes: &mut [u8], order: u32, value: f32) {
    let raw = if order == ORDER_LE {
        value.to_le_bytes()
    } else {
        value.to_be_bytes()
    };
    bytes[..4].copy_from_slice(&raw);
}

fn write_float64(bytes: &mut [u8], order: u32, value: f64) {
    let raw = if order == ORDER_LE {
        value.to_le_bytes()
    } else {
        value.to_be_bytes()
    };
    bytes[..8].copy_from_slice(&raw);
}

struct BitStream<'a> {
    data: &'a [u8],
    byte: usize,
    bits_left: usize,
}

impl<'a> BitStream<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte: 0,
            bits_left: 8,
        }
    }

    fn read_bits(&mut self, mut nbits: usize) -> Result<u16> {
        if nbits > 16 {
            return Err(Error::InvalidFormat("scaleoffset bit run too long".into()));
        }

        let mut value = 0u16;
        while nbits > 0 {
            let byte = *self
                .data
                .get(self.byte)
                .ok_or_else(|| Error::InvalidFormat("scaleoffset data too short".into()))?;
            let take = self.bits_left.min(nbits);
            let shift = self.bits_left - take;
            let mask = if take == 8 {
                0xff
            } else {
                ((1u16 << take) - 1) as u8
            };
            value = (value << take) | u16::from((byte >> shift) & mask);
            self.bits_left -= take;
            nbits -= take;
            if self.bits_left == 0 {
                self.byte += 1;
                self.bits_left = 8;
            }
        }
        Ok(value)
    }
}

struct BitWriter {
    data: Vec<u8>,
    current: u8,
    bits_used: usize,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            data: Vec::new(),
            current: 0,
            bits_used: 0,
        }
    }

    fn next_byte(&mut self) {
        self.data.push(self.current);
        self.current = 0;
        self.bits_used = 0;
    }

    fn write_bits(&mut self, value: u16, mut nbits: usize) -> Result<()> {
        if nbits > 16 {
            return Err(Error::InvalidFormat("scaleoffset bit run too long".into()));
        }
        while nbits > 0 {
            let free = 8 - self.bits_used;
            let take = free.min(nbits);
            let shift = nbits - take;
            let mask = if take == 16 {
                u16::MAX
            } else {
                (1u16 << take) - 1
            };
            let bits = u8::try_from((value >> shift) & mask)
                .map_err(|_| Error::InvalidFormat("scaleoffset bit run exceeds byte".into()))?;
            self.current |= bits << (free - take);
            self.bits_used += take;
            nbits -= take;
            if self.bits_used == 8 {
                self.next_byte();
            }
        }
        Ok(())
    }

    fn finish(mut self) -> Vec<u8> {
        if self.bits_used != 0 {
            self.next_byte();
        }
        self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(minbits: u32, minval_size: u8) -> Vec<u8> {
        let mut data = vec![0u8; HEADER_LEN];
        data[..4].copy_from_slice(&minbits.to_le_bytes());
        data[4] = minval_size;
        data
    }

    #[test]
    fn rejects_missing_client_data() {
        let err = decompress(&[], &[0, 0, 1]).unwrap_err();
        assert!(
            err.to_string()
                .contains("scaleoffset filter missing datatype parameters"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_invalid_float_scale_type() {
        let params = vec![1, 2, 1, CLS_FLOAT, 4, SIGN_UNSIGNED, ORDER_LE];
        let err = decompress(&header(0, 0), &params).unwrap_err();
        assert!(
            err.to_string().contains("scaleoffset float scale type 1"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_full_precision_output_size_mismatch() {
        let params = vec![2, 0, 2, CLS_INTEGER, 4, SIGN_UNSIGNED, ORDER_LE];
        let mut data = header(32, 0);
        data.extend_from_slice(&1u32.to_le_bytes());
        let err = decompress(&data, &params).unwrap_err();
        assert!(
            err.to_string()
                .contains("scaleoffset full-precision data too short"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_invalid_integer_sign_even_for_empty_chunks() {
        let params = vec![2, 0, 0, CLS_INTEGER, 4, 99, ORDER_LE];
        let err = decompress(&header(0, 0), &params).unwrap_err();
        assert!(
            err.to_string()
                .contains("invalid scaleoffset integer sign 99"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_unsupported_datatype_class_before_chunk_header() {
        let params = vec![2, 0, 0, 99, 4, SIGN_UNSIGNED, ORDER_LE];
        let err = decompress(&[], &params).unwrap_err();
        assert!(
            err.to_string().contains("scaleoffset datatype class 99"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_unsupported_float_size_before_chunk_header() {
        let params = vec![0, 0, 0, CLS_FLOAT, 2, SIGN_UNSIGNED, ORDER_LE];
        let err = decompress(&[], &params).unwrap_err();
        assert!(
            err.to_string()
                .contains("scaleoffset floating-point size 2"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_unsupported_float_scale_type_before_chunk_header() {
        let params = vec![1, 0, 0, CLS_FLOAT, 4, SIGN_UNSIGNED, ORDER_LE];
        let err = decompress(&[], &params).unwrap_err();
        assert!(
            err.to_string().contains("scaleoffset float scale type 1"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_minbits_larger_than_datatype_even_for_empty_chunks() {
        let params = vec![2, 0, 0, CLS_INTEGER, 4, SIGN_UNSIGNED, ORDER_LE];
        let err = decompress(&header(33, 0), &params).unwrap_err();
        assert!(
            err.to_string()
                .contains("invalid scaleoffset minimum bit count"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_datatype_sizes_beyond_internal_integer_width() {
        let params = vec![2, 0, 0, CLS_INTEGER, 17, SIGN_UNSIGNED, ORDER_LE];
        let err = decompress(&header(0, 0), &params).unwrap_err();
        assert!(
            err.to_string()
                .contains("scaleoffset datatype size 17 exceeds 16-byte arithmetic support"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn decompress_byte_rejects_output_offset_overflow() {
        let parms = Parms {
            size: 1,
            minbits: 8,
            order: ORDER_LE,
        };
        let mut out = [0u8; 1];
        let mut stream = BitStream::new(&[0xff]);
        let err = decompress_byte(&mut out, usize::MAX, 1, 1, &mut stream, parms, 8).unwrap_err();
        assert!(
            err.to_string()
                .contains("scaleoffset output offset overflow"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn decompress_byte_rejects_output_offset_out_of_range() {
        let parms = Parms {
            size: 1,
            minbits: 8,
            order: ORDER_LE,
        };
        let mut out = [0u8; 1];
        let mut stream = BitStream::new(&[0xff]);
        let err = decompress_byte(&mut out, 1, 0, 0, &mut stream, parms, 8).unwrap_err();
        assert!(
            err.to_string()
                .contains("scaleoffset output offset out of range"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn checked_window_rejects_offset_overflow() {
        let err = checked_window(&[], usize::MAX, 1, "scaleoffset test window").unwrap_err();
        assert!(
            err.to_string()
                .contains("scaleoffset test window offset overflow"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn checked_window_mut_rejects_offset_overflow() {
        let mut data = [];
        let err = checked_window_mut(&mut data, usize::MAX, 1, "scaleoffset test mutable window")
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("scaleoffset test mutable window offset overflow"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn scaleoffset_integer_compress_roundtrips() {
        let params = vec![2, 0, 4, CLS_INTEGER, 2, SIGN_UNSIGNED, ORDER_LE];
        let input = [10u16, 11, 12, 16]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();
        let compressed = scaleoffset_compress(&input, &params).unwrap();
        let decoded = decompress(&compressed, &params).unwrap();
        assert_eq!(decoded, input);
    }
}
