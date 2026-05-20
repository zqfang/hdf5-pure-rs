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
const SCALE_FLOAT_DSCALE: u32 = 0;
const SCALE_INT: u32 = 2;

#[derive(Debug, Clone, Copy)]
struct Parms {
    size: usize,
    minbits: usize,
    order: u32,
}

/// Decompress HDF5 ScaleOffset-filtered chunks, appending decoded bytes to
/// `out`.
pub fn decompress_into(data: &[u8], client_data: &[u32], out: &mut Vec<u8>) -> Result<()> {
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
    let start = out.len();
    out.resize(
        start
            .checked_add(out_len)
            .ok_or_else(|| Error::InvalidFormat("scaleoffset output size overflow".into()))?,
        0,
    );
    let out = &mut out[start..];

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
            decompress_atomic(out, data_offset, &mut stream, parms)?;
        }
    }

    match class {
        CLS_INTEGER => {
            let fill = if client_data.get(PARM_FILAVAIL).copied().unwrap_or(0) != 0 {
                Some(read_fill_value(client_data, size, order))
            } else {
                None
            };
            postprocess_integer(out, size, sign, order, minbits, minval, fill)?
        }
        CLS_FLOAT if scale_type == 0 => {
            postprocess_float(out, size, order, minbits, minval, client_data)?
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

    Ok(())
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
    validate_scaleoffset_datatype(class, sign, size, scale_type)?;
    validate_scaleoffset_fill_value(client_data, size)?;
    Ok(())
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

/// HDF5 ScaleOffset filter entry point, appending the result to `out`.
pub fn filter_scaleoffset_into(
    data: &[u8],
    client_data: &[u32],
    reverse: bool,
    out: &mut Vec<u8>,
) -> Result<()> {
    if reverse {
        decompress_into(data, client_data, out)
    } else {
        scaleoffset_compress_into(data, client_data, out)
    }
}

/// Compress ScaleOffset-filtered data, appending the encoded bytes to `out`.
pub fn scaleoffset_compress_into(
    data: &[u8],
    client_data: &[u32],
    out: &mut Vec<u8>,
) -> Result<()> {
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

    let mut packed = Vec::with_capacity(expected);
    let (minbits, minval) = match class {
        CLS_INTEGER => scaleoffset_precompress_i_with_minbits_into(
            &data[..expected],
            size,
            sign,
            order,
            Some(client_data[1]),
            if client_data.get(PARM_FILAVAIL).copied().unwrap_or(0) != 0 {
                Some(read_fill_value(client_data, size, order))
            } else {
                None
            },
            &mut packed,
        )?,
        CLS_FLOAT if scale_type == 0 => scaleoffset_precompress_fd_into(
            &data[..expected],
            size,
            order,
            client_data,
            &mut packed,
        )?,
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

    let header_start = out.len();
    out.resize(
        header_start
            .checked_add(HEADER_LEN)
            .ok_or_else(|| Error::InvalidFormat("scaleoffset output size overflow".into()))?,
        0,
    );
    let header = &mut out[header_start..header_start + HEADER_LEN];
    let minbits_u32 = u32::try_from(minbits).map_err(|_| {
        Error::InvalidFormat("scaleoffset minimum bit count does not fit in u32".into())
    })?;
    header[..4].copy_from_slice(&minbits_u32.to_le_bytes());
    header[4] = size as u8;
    write_uint(
        checked_window_mut(header, 5, size, "scaleoffset minimum value header")?,
        ORDER_LE,
        minval,
    );

    if minbits == size * 8 {
        out.extend_from_slice(&packed);
    } else if minbits != 0 {
        let mut writer = BitWriter::new(out);
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
        writer.finish();
    }
    Ok(())
}

pub fn scaleoffset_precompress_i_into(
    data: &[u8],
    size: usize,
    sign: u32,
    order: u32,
    packed: &mut Vec<u8>,
) -> Result<(usize, u128)> {
    scaleoffset_precompress_i_with_minbits_into(data, size, sign, order, None, None, packed)
}

fn scaleoffset_precompress_i_with_minbits_into(
    data: &[u8],
    size: usize,
    sign: u32,
    order: u32,
    requested_minbits: Option<u32>,
    fill: Option<u128>,
    packed: &mut Vec<u8>,
) -> Result<(usize, u128)> {
    validate_scaleoffset_parameters(CLS_INTEGER, sign, size, 0)?;
    if size == 0 || data.len() % size != 0 {
        return Err(Error::InvalidFormat(
            "scaleoffset integer input size mismatch".into(),
        ));
    }
    if data.is_empty() {
        return Ok((0, 0));
    }
    let requested_minbits = requested_minbits
        .filter(|bits| *bits != 0)
        .map(|bits| scaleoffset_usize(bits, "scaleoffset integer minimum bit count"))
        .transpose()?
        .map(|bits| bits.min(size * 8));
    let (computed_minbits, minval) = if sign == SIGN_TWOS {
        let fill = fill.map(|value| read_int_from_bits(value, size));
        let minval = data
            .chunks_exact(size)
            .map(|chunk| read_int(chunk, order))
            .filter(|value| Some(*value) != fill)
            .min()
            .unwrap_or(0);
        let max_delta = data
            .chunks_exact(size)
            .map(|chunk| read_int(chunk, order))
            .filter(|value| Some(*value) != fill)
            .map(|value| signed_delta(value, minval))
            .max()
            .unwrap_or(0);
        (
            scaleoffset_integer_minbits(max_delta, fill.is_some(), size)?,
            signed_to_uint_bits(minval, size),
        )
    } else {
        let minval = data
            .chunks_exact(size)
            .map(|chunk| read_uint(chunk, order))
            .filter(|value| Some(*value) != fill)
            .min()
            .unwrap_or(0);
        let max_delta = data
            .chunks_exact(size)
            .map(|chunk| read_uint(chunk, order))
            .filter(|value| Some(*value) != fill)
            .map(|value| value.wrapping_sub(minval))
            .max()
            .unwrap_or(0);
        (
            scaleoffset_integer_minbits(max_delta, fill.is_some(), size)?,
            minval,
        )
    };
    let minbits = requested_minbits.unwrap_or(computed_minbits);
    let fill_marker = fill
        .filter(|_| minbits != size * 8)
        .map(|_| fill_marker(minbits));
    let start = packed.len();
    packed.resize(
        start
            .checked_add(data.len())
            .ok_or_else(|| Error::InvalidFormat("scaleoffset packed size overflow".into()))?,
        0,
    );
    for (idx, chunk) in data.chunks_exact(size).enumerate() {
        let offset =
            start
                .checked_add(idx.checked_mul(size).ok_or_else(|| {
                    Error::InvalidFormat("scaleoffset packed offset overflow".into())
                })?)
                .ok_or_else(|| Error::InvalidFormat("scaleoffset packed offset overflow".into()))?;
        let raw = read_uint(chunk, order);
        let value = if Some(raw) == fill {
            fill_marker.unwrap_or(raw)
        } else if sign == SIGN_TWOS {
            signed_delta(
                read_int_from_bits(raw, size),
                read_int_from_bits(minval, size),
            )
        } else {
            raw.wrapping_sub(minval)
        };
        write_uint(
            checked_window_mut(packed, offset, size, "scaleoffset packed integer value")?,
            order,
            value,
        );
    }
    Ok((minbits, minval))
}

pub fn scaleoffset_precompress_fd_into(
    data: &[u8],
    size: usize,
    order: u32,
    client_data: &[u32],
    packed: &mut Vec<u8>,
) -> Result<(usize, u128)> {
    if !matches!(size, 4 | 8) || data.len() % size != 0 {
        return Err(Error::InvalidFormat(
            "scaleoffset floating-point input size mismatch".into(),
        ));
    }
    if data.is_empty() {
        return Ok((0, 0));
    }
    let scale = client_data
        .get(1)
        .copied()
        .ok_or_else(|| Error::InvalidFormat("scaleoffset missing scale factor".into()))?
        .to_ne_bytes();
    let scale = i32::from_ne_bytes(scale);
    let multiplier = 10f64.powi(scale);
    let fill = if client_data.get(PARM_FILAVAIL).copied().unwrap_or(0) != 0 {
        Some(read_fill_float(client_data, size, order))
    } else {
        None
    };
    let fill_epsilon = 10f64.powi(-scale);
    let is_fill = |value: f64| {
        fill.is_some_and(|fill| {
            if value.is_nan() || fill.is_nan() {
                value.to_bits() == fill.to_bits()
            } else {
                (value - fill).abs() < fill_epsilon
            }
        })
    };
    let mut values = data
        .chunks_exact(size)
        .map(|chunk| read_float(chunk, size, order))
        .filter(|value| !is_fill(*value));
    let first = values.next().unwrap_or(0.0);
    let (min, max) = values.fold((first, first), |(min, max), value| {
        (min.min(value), max.max(value))
    });
    let max_delta = data
        .chunks_exact(size)
        .map(|chunk| read_float(chunk, size, order))
        .filter(|value| !is_fill(*value))
        .map(|value| ((value - min) * multiplier).round().max(0.0) as u128)
        .max()
        .unwrap_or_else(|| ((max - min) * multiplier).round().max(0.0) as u128);
    let minbits = if fill.is_some() {
        scaleoffset_integer_minbits(max_delta, true, size)?
    } else if max_delta == 0 {
        0
    } else {
        usize::try_from(u128::BITS - max_delta.leading_zeros())
            .map_err(|_| Error::InvalidFormat("scaleoffset bit count overflow".into()))?
    }
    .min(size * 8);
    let fill_marker = fill
        .filter(|_| minbits != size * 8)
        .map(|_| fill_marker(minbits));
    let start = packed.len();
    packed.resize(
        start
            .checked_add(data.len())
            .ok_or_else(|| Error::InvalidFormat("scaleoffset packed size overflow".into()))?,
        0,
    );
    for (idx, chunk) in data.chunks_exact(size).enumerate() {
        let value = read_float(chunk, size, order);
        let delta = if is_fill(value) {
            fill_marker.unwrap_or_else(|| read_uint(chunk, order))
        } else {
            ((value - min) * multiplier).round().max(0.0) as u128
        };
        let offset =
            start
                .checked_add(idx.checked_mul(size).ok_or_else(|| {
                    Error::InvalidFormat("scaleoffset packed offset overflow".into())
                })?)
                .ok_or_else(|| Error::InvalidFormat("scaleoffset packed offset overflow".into()))?;
        write_uint(
            checked_window_mut(packed, offset, size, "scaleoffset packed float value")?,
            order,
            delta,
        );
    }
    let minval = if size == 4 {
        (min as f32).to_bits() as u128
    } else {
        min.to_bits() as u128
    };
    Ok((minbits, minval))
}

fn read_float(chunk: &[u8], size: usize, order: u32) -> f64 {
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
}

fn read_fill_float(client_data: &[u32], size: usize, order: u32) -> f64 {
    let fill = read_fill_value(client_data, size, order);
    if size == 4 {
        f32::from_le_bytes(low_u32(fill).to_le_bytes()) as f64
    } else {
        f64::from_le_bytes(low_u64(fill).to_le_bytes())
    }
}

fn validate_scaleoffset_datatype(
    class: u32,
    sign: u32,
    size: usize,
    scale_type: u32,
) -> Result<()> {
    match class {
        CLS_INTEGER => {
            if scale_type != SCALE_INT {
                return Err(Error::InvalidFormat(format!(
                    "invalid scaleoffset integer scale type {scale_type}"
                )));
            }
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
            if scale_type != SCALE_FLOAT_DSCALE {
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

fn validate_scaleoffset_fill_value(client_data: &[u32], size: usize) -> Result<()> {
    if client_data.get(PARM_FILAVAIL).copied().unwrap_or(0) == 0 {
        return Ok(());
    }
    let fill_words = size
        .checked_add(3)
        .ok_or_else(|| Error::InvalidFormat("scaleoffset fill value size overflow".into()))?
        / 4;
    let required = PARM_FILVAL
        .checked_add(fill_words)
        .ok_or_else(|| Error::InvalidFormat("scaleoffset fill value size overflow".into()))?;
    if client_data.len() < required {
        return Err(Error::InvalidFormat(
            "scaleoffset fill value parameters are truncated".into(),
        ));
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
    writer: &mut BitWriter<'_>,
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
    writer: &mut BitWriter<'_>,
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

fn read_int(bytes: &[u8], order: u32) -> i128 {
    read_int_from_bits(read_uint(bytes, order), bytes.len())
}

fn read_int_from_bits(value: u128, size: usize) -> i128 {
    let bits = size.saturating_mul(8).min(128);
    if bits == 0 {
        return 0;
    }
    let shift = 128 - bits;
    ((value << shift) as i128) >> shift
}

fn signed_to_uint_bits(value: i128, size: usize) -> u128 {
    let bits = size.saturating_mul(8).min(128);
    if bits == 128 {
        value as u128
    } else {
        (value as u128) & ((1u128 << bits) - 1)
    }
}

fn signed_abs(value: i128) -> u128 {
    if value == i128::MIN {
        1u128 << 127
    } else {
        value.unsigned_abs()
    }
}

fn signed_delta(value: i128, minval: i128) -> u128 {
    debug_assert!(value >= minval);
    if minval >= 0 {
        (value - minval) as u128
    } else if value < 0 {
        signed_abs(minval) - signed_abs(value)
    } else {
        signed_abs(minval) + value as u128
    }
}

fn minbits_for_delta(delta: u128) -> Result<usize> {
    if delta == 0 {
        Ok(0)
    } else {
        usize::try_from(u128::BITS - delta.leading_zeros())
            .map_err(|_| Error::InvalidFormat("scaleoffset bit count overflow".into()))
    }
}

fn scaleoffset_integer_minbits(delta: u128, has_fill: bool, size: usize) -> Result<usize> {
    let delta = if has_fill {
        delta.saturating_add(1)
    } else {
        delta
    };
    Ok(minbits_for_delta(delta)?.min(size * 8))
}

fn fill_marker(minbits: usize) -> u128 {
    if minbits >= 128 {
        u128::MAX
    } else {
        (1u128 << minbits) - 1
    }
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
    let mut raw = [0u8; 16];
    let mut pos = 0usize;
    for value in client_data.iter().skip(PARM_FILVAL) {
        let bytes = value.to_le_bytes();
        for byte in bytes {
            if pos < size {
                raw[pos] = byte;
                pos += 1;
            }
        }
    }
    read_uint(&raw[..size], order)
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

struct BitWriter<'a> {
    out: &'a mut Vec<u8>,
    current: u8,
    bits_used: usize,
}

impl<'a> BitWriter<'a> {
    fn new(out: &'a mut Vec<u8>) -> Self {
        Self {
            out,
            current: 0,
            bits_used: 0,
        }
    }

    fn next_byte(&mut self) {
        self.out.push(self.current);
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

    fn finish(mut self) {
        if self.bits_used != 0 {
            self.next_byte();
        }
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

    fn decompress_err(data: &[u8], params: &[u32]) -> Error {
        let mut out = Vec::new();
        decompress_into(data, params, &mut out).unwrap_err()
    }

    fn integer_params(nelmts: u32, size: u32) -> Vec<u32> {
        vec![2, 0, nelmts, CLS_INTEGER, size, SIGN_UNSIGNED, ORDER_LE]
    }

    #[test]
    fn rejects_missing_client_data() {
        let err = decompress_err(&[], &[0, 0, 1]);
        assert!(
            err.to_string()
                .contains("scaleoffset filter missing datatype parameters"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_invalid_float_scale_type() {
        let params = vec![1, 2, 1, CLS_FLOAT, 4, SIGN_UNSIGNED, ORDER_LE];
        let err = decompress_err(&header(0, 0), &params);
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
        let err = decompress_err(&data, &params);
        assert!(
            err.to_string()
                .contains("scaleoffset full-precision data too short"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_invalid_integer_sign_even_for_empty_chunks() {
        let params = vec![2, 0, 0, CLS_INTEGER, 4, 99, ORDER_LE];
        let err = decompress_err(&header(0, 0), &params);
        assert!(
            err.to_string()
                .contains("invalid scaleoffset integer sign 99"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_invalid_integer_scale_type_even_for_empty_chunks() {
        let params = vec![0, 0, 0, CLS_INTEGER, 4, SIGN_UNSIGNED, ORDER_LE];
        let err = decompress_err(&header(0, 0), &params);
        assert!(
            err.to_string()
                .contains("invalid scaleoffset integer scale type 0"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_unsupported_datatype_class_before_chunk_header() {
        let params = vec![2, 0, 0, 99, 4, SIGN_UNSIGNED, ORDER_LE];
        let err = decompress_err(&[], &params);
        assert!(
            err.to_string().contains("scaleoffset datatype class 99"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_unsupported_float_size_before_chunk_header() {
        let params = vec![0, 0, 0, CLS_FLOAT, 2, SIGN_UNSIGNED, ORDER_LE];
        let err = decompress_err(&[], &params);
        assert!(
            err.to_string()
                .contains("scaleoffset floating-point size 2"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_unsupported_float_scale_type_before_chunk_header() {
        let params = vec![1, 0, 0, CLS_FLOAT, 4, SIGN_UNSIGNED, ORDER_LE];
        let err = decompress_err(&[], &params);
        assert!(
            err.to_string().contains("scaleoffset float scale type 1"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_truncated_fill_value_parameters_before_chunk_header() {
        let params = vec![2, 0, 0, CLS_INTEGER, 1, SIGN_UNSIGNED, ORDER_LE, 1];
        let err = decompress_err(&[], &params);
        assert!(
            err.to_string()
                .contains("scaleoffset fill value parameters are truncated"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_partial_multibyte_fill_value_parameters_before_chunk_header() {
        let params = vec![2, 0, 0, CLS_INTEGER, 8, SIGN_UNSIGNED, ORDER_LE, 1, 0];
        let err = decompress_err(&[], &params);
        assert!(
            err.to_string()
                .contains("scaleoffset fill value parameters are truncated"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_minbits_larger_than_datatype_even_for_empty_chunks() {
        let params = integer_params(0, 4);
        let err = decompress_err(&header(33, 0), &params);
        assert!(
            err.to_string()
                .contains("invalid scaleoffset minimum bit count"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_datatype_sizes_beyond_internal_integer_width() {
        let params = vec![2, 0, 0, CLS_INTEGER, 17, SIGN_UNSIGNED, ORDER_LE];
        let err = decompress_err(&header(0, 0), &params);
        assert!(
            err.to_string()
                .contains("scaleoffset datatype size 17 exceeds 16-byte arithmetic support"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_truncated_chunk_header() {
        let params = integer_params(1, 2);
        let err = decompress_err(&header(8, 0)[..HEADER_LEN - 1], &params);
        assert!(
            err.to_string().contains("scaleoffset data too short"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_invalid_minimum_value_header_size() {
        let params = integer_params(1, 2);
        let err = decompress_err(&header(0, 17), &params);
        assert!(
            err.to_string()
                .contains("invalid scaleoffset minimum value header"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_truncated_packed_payload() {
        let params = integer_params(2, 2);
        let mut data = header(9, 0);
        data.push(0xff);
        let err = decompress_err(&data, &params);
        assert!(
            err.to_string().contains("scaleoffset data too short"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_truncated_full_precision_payload_generated_in_memory() {
        let params = integer_params(2, 2);
        let mut data = header(16, 0);
        data.extend_from_slice(&1u16.to_le_bytes());
        let err = decompress_err(&data, &params);
        assert!(
            err.to_string()
                .contains("scaleoffset full-precision data too short"),
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
        let mut compressed = Vec::new();
        scaleoffset_compress_into(&input, &params, &mut compressed).unwrap();
        let mut decoded = Vec::new();
        decompress_into(&compressed, &params, &mut decoded).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn scaleoffset_integer_full_precision_payload_stores_deltas() {
        let params = vec![2, 0, 2, CLS_INTEGER, 1, SIGN_UNSIGNED, ORDER_LE];

        let mut encoded = header(8, 1);
        encoded[5] = 100;
        encoded.extend_from_slice(&[0, 155]);
        let mut decoded = Vec::new();
        decompress_into(&encoded, &params, &mut decoded).unwrap();
        assert_eq!(decoded, [100, 255]);

        let mut compressed = Vec::new();
        scaleoffset_compress_into(&[100, 255], &params, &mut compressed).unwrap();
        assert_eq!(&compressed[HEADER_LEN..], &[0, 155]);

        let mut roundtripped = Vec::new();
        decompress_into(&compressed, &params, &mut roundtripped).unwrap();
        assert_eq!(roundtripped, [100, 255]);
    }

    #[test]
    fn scaleoffset_integer_compress_respects_fixed_minbits() {
        let params = vec![2, 2, 4, CLS_INTEGER, 2, SIGN_UNSIGNED, ORDER_LE];
        let input = [10u16, 11, 12, 16]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();

        let mut compressed = Vec::new();
        scaleoffset_compress_into(&input, &params, &mut compressed).unwrap();

        assert_eq!(&compressed[..4], &2u32.to_le_bytes());
        assert_eq!(&compressed[5..7], &10u16.to_le_bytes());

        let mut decoded = Vec::new();
        decompress_into(&compressed, &params, &mut decoded).unwrap();
        let expected = [10u16, 11, 12, 12]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();
        assert_eq!(decoded, expected);
    }

    #[test]
    fn scaleoffset_integer_compress_caps_oversized_fixed_minbits() {
        let params = vec![2, 100, 2, CLS_INTEGER, 2, SIGN_UNSIGNED, ORDER_LE];
        let input = [1u16, 257]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();

        let mut compressed = Vec::new();
        scaleoffset_compress_into(&input, &params, &mut compressed).unwrap();

        assert_eq!(&compressed[..4], &16u32.to_le_bytes());

        let mut decoded = Vec::new();
        decompress_into(&compressed, &params, &mut decoded).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn scaleoffset_integer_compress_reserves_fill_marker() {
        let params = vec![2, 0, 4, CLS_INTEGER, 1, SIGN_UNSIGNED, ORDER_LE, 1, 250];
        let input = [10, 250, 12, 13];

        let mut compressed = Vec::new();
        scaleoffset_compress_into(&input, &params, &mut compressed).unwrap();

        assert_eq!(&compressed[..4], &3u32.to_le_bytes());
        assert_eq!(compressed[5], 10);
        assert_eq!(&compressed[HEADER_LEN..], &[0x1d, 0x30]);

        let mut decoded = Vec::new();
        decompress_into(&compressed, &params, &mut decoded).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn scaleoffset_signed_integer_compress_reserves_fill_marker() {
        let params = vec![2, 0, 3, CLS_INTEGER, 1, SIGN_TWOS, ORDER_LE, 1, 0xf7];
        let input = [0xfe, 0xf7, 0x01];

        let mut compressed = Vec::new();
        scaleoffset_compress_into(&input, &params, &mut compressed).unwrap();

        assert_eq!(&compressed[..4], &3u32.to_le_bytes());
        assert_eq!(compressed[5], 0xfe);

        let mut decoded = Vec::new();
        decompress_into(&compressed, &params, &mut decoded).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn scaleoffset_signed_integer_uses_signed_minimum() {
        let params = vec![2, 0, 3, CLS_INTEGER, 1, SIGN_TWOS, ORDER_LE];
        let input = [0xff, 0x00, 0x01];

        let mut compressed = Vec::new();
        scaleoffset_compress_into(&input, &params, &mut compressed).unwrap();

        assert_eq!(&compressed[..4], &2u32.to_le_bytes());
        assert_eq!(compressed[4], 1);
        assert_eq!(compressed[5], 0xff);
        assert_eq!(compressed.len(), HEADER_LEN + 1);

        let mut decoded = Vec::new();
        decompress_into(&compressed, &params, &mut decoded).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn scaleoffset_signed_integer_big_endian_roundtrips_crossing_zero() {
        let params = vec![2, 0, 3, CLS_INTEGER, 2, SIGN_TWOS, ORDER_BE];
        let input = [0xff, 0xfe, 0xff, 0xff, 0x00, 0x00];

        let mut compressed = Vec::new();
        scaleoffset_compress_into(&input, &params, &mut compressed).unwrap();

        assert_eq!(&compressed[..4], &2u32.to_le_bytes());
        assert_eq!(compressed[4], 2);
        assert_eq!(&compressed[5..7], &[0xfe, 0xff]);

        let mut decoded = Vec::new();
        decompress_into(&compressed, &params, &mut decoded).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn scaleoffset_float_compress_reserves_fill_marker() {
        let fill = -999.0f32;
        let mut params = vec![0, 1, 4, CLS_FLOAT, 4, SIGN_UNSIGNED, ORDER_LE, 1];
        params.push(fill.to_bits());
        let input = [1.0f32, fill, 1.2, 1.3]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();

        let mut compressed = Vec::new();
        scaleoffset_compress_into(&input, &params, &mut compressed).unwrap();

        assert_eq!(&compressed[..4], &3u32.to_le_bytes());
        assert_eq!(&compressed[5..9], &1.0f32.to_le_bytes());

        let mut decoded = Vec::new();
        decompress_into(&compressed, &params, &mut decoded).unwrap();
        let values = decoded
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(values, [1.0, fill, 1.2, 1.3]);
    }

    #[test]
    fn scaleoffset_integer_all_fill_values_roundtrip() {
        let fill = 0xffffu32;
        let params = vec![2, 0, 3, CLS_INTEGER, 2, SIGN_UNSIGNED, ORDER_LE, 1, fill];
        let input = [0xffffu16, 0xffff, 0xffff]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();

        let mut compressed = Vec::new();
        scaleoffset_compress_into(&input, &params, &mut compressed).unwrap();
        assert_eq!(&compressed[..4], &1u32.to_le_bytes());
        assert_eq!(&compressed[5..7], &0u16.to_le_bytes());

        let mut decoded = Vec::new();
        decompress_into(&compressed, &params, &mut decoded).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn scaleoffset_float_all_fill_values_roundtrip() {
        let fill = -999.0f64;
        let low = fill.to_bits() as u32;
        let high = (fill.to_bits() >> 32) as u32;
        let params = vec![0, 2, 2, CLS_FLOAT, 8, SIGN_UNSIGNED, ORDER_LE, 1, low, high];
        let input = [fill, fill]
            .into_iter()
            .flat_map(f64::to_le_bytes)
            .collect::<Vec<_>>();

        let mut compressed = Vec::new();
        scaleoffset_compress_into(&input, &params, &mut compressed).unwrap();
        assert_eq!(&compressed[..4], &1u32.to_le_bytes());
        assert_eq!(&compressed[5..13], &0f64.to_le_bytes());

        let mut decoded = Vec::new();
        decompress_into(&compressed, &params, &mut decoded).unwrap();
        assert_eq!(decoded, input);
    }
}
