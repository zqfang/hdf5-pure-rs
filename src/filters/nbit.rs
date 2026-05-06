use crate::error::{Error, Result};

const NBIT_ATOMIC: u32 = 1;
const NBIT_ARRAY: u32 = 2;
const NBIT_COMPOUND: u32 = 3;
const NBIT_NOOPTYPE: u32 = 4;
const NBIT_ORDER_LE: u32 = 0;
const NBIT_ORDER_BE: u32 = 1;

#[derive(Debug, Clone, Copy)]
struct AtomicParms {
    size: usize,
    order: u32,
    precision: usize,
    offset: usize,
}

/// Decompress HDF5 NBit-filtered data using the datatype-aware filter
/// parameters stored by HDF5's set-local callback.
pub fn decompress(data: &[u8], client_data: &[u32]) -> Result<Vec<u8>> {
    can_apply_nbit(client_data)?;

    if client_data.len() < 5 {
        return Err(Error::InvalidFormat(
            "nbit filter missing datatype parameters".into(),
        ));
    }

    let nparams = nbit_usize(client_data[0], "nbit parameter count")?;
    if nparams != client_data.len() {
        return Err(Error::InvalidFormat(format!(
            "nbit parameter count mismatch: header says {nparams}, got {}",
            client_data.len()
        )));
    }

    if client_data[1] != 0 {
        return Ok(data.to_vec());
    }

    let nelmts = nbit_usize(client_data[2], "nbit element count")?;
    let dtype_size = nbit_usize(client_data[4], "nbit datatype size")?;
    if dtype_size == 0 {
        return Err(Error::InvalidFormat("nbit datatype size is zero".into()));
    }
    let mut out =
        vec![
            0u8;
            nelmts
                .checked_mul(dtype_size)
                .ok_or_else(|| Error::InvalidFormat("nbit output size overflow".into()))?
        ];
    let mut stream = BitStream::new(data);

    match client_data[3] {
        NBIT_ATOMIC => {
            let parms = AtomicParms {
                size: dtype_size,
                order: *client_data
                    .get(5)
                    .ok_or_else(|| Error::InvalidFormat("nbit missing byte order".into()))?,
                precision: nbit_usize(
                    client_data
                        .get(6)
                        .copied()
                        .ok_or_else(|| Error::InvalidFormat("nbit missing precision".into()))?,
                    "nbit precision",
                )?,
                offset: nbit_usize(
                    client_data
                        .get(7)
                        .copied()
                        .ok_or_else(|| Error::InvalidFormat("nbit missing bit offset".into()))?,
                    "nbit bit offset",
                )?,
            };
            validate_atomic(parms)?;
            for idx in 0..nelmts {
                let offset = nbit_nested_offset(0, idx, parms.size)?;
                decompress_atomic(&mut out, offset, &mut stream, parms)?;
            }
        }
        NBIT_ARRAY => {
            for idx in 0..nelmts {
                let mut pidx = 4usize;
                let offset = nbit_nested_offset(0, idx, dtype_size)?;
                decompress_array(&mut out, offset, &mut stream, client_data, &mut pidx)?;
            }
        }
        NBIT_COMPOUND => {
            for idx in 0..nelmts {
                let mut pidx = 4usize;
                let offset = nbit_nested_offset(0, idx, dtype_size)?;
                decompress_compound(&mut out, offset, &mut stream, client_data, &mut pidx)?;
            }
        }
        other => {
            return Err(Error::Unsupported(format!(
                "nbit datatype class parameter {other}"
            )));
        }
    }

    Ok(out)
}

pub fn nbit_decompress(data: &[u8], client_data: &[u32]) -> Result<Vec<u8>> {
    decompress(data, client_data)
}

pub fn nbit_compress(data: &[u8], client_data: &[u32]) -> Result<Vec<u8>> {
    can_apply_nbit(client_data)?;
    if client_data[1] != 0 {
        return Ok(data.to_vec());
    }

    let nelmts = nbit_usize(client_data[2], "nbit element count")?;
    let dtype_size = nbit_usize(client_data[4], "nbit datatype size")?;
    let expected = nelmts
        .checked_mul(dtype_size)
        .ok_or_else(|| Error::InvalidFormat("nbit input size overflow".into()))?;
    if data.len() < expected {
        return Err(Error::InvalidFormat("nbit input data too short".into()));
    }

    let mut writer = BitWriter::new();
    match client_data[3] {
        NBIT_ATOMIC => {
            let parms = AtomicParms {
                size: dtype_size,
                order: *client_data
                    .get(5)
                    .ok_or_else(|| Error::InvalidFormat("nbit missing byte order".into()))?,
                precision: nbit_usize(
                    client_data
                        .get(6)
                        .copied()
                        .ok_or_else(|| Error::InvalidFormat("nbit missing precision".into()))?,
                    "nbit precision",
                )?,
                offset: nbit_usize(
                    client_data
                        .get(7)
                        .copied()
                        .ok_or_else(|| Error::InvalidFormat("nbit missing bit offset".into()))?,
                    "nbit bit offset",
                )?,
            };
            validate_atomic(parms)?;
            for idx in 0..nelmts {
                let offset = nbit_nested_offset(0, idx, parms.size)?;
                compress_atomic(data, offset, &mut writer, parms)?;
            }
        }
        NBIT_ARRAY => {
            for idx in 0..nelmts {
                let mut pidx = 4usize;
                let offset = nbit_nested_offset(0, idx, dtype_size)?;
                compress_array(data, offset, &mut writer, client_data, &mut pidx)?;
            }
        }
        NBIT_COMPOUND => {
            for idx in 0..nelmts {
                let mut pidx = 4usize;
                let offset = nbit_nested_offset(0, idx, dtype_size)?;
                compress_compound(data, offset, &mut writer, client_data, &mut pidx)?;
            }
        }
        other => {
            return Err(Error::Unsupported(format!(
                "nbit datatype class parameter {other}"
            )));
        }
    }
    Ok(writer.finish())
}

pub fn can_apply_nbit(client_data: &[u32]) -> Result<()> {
    if client_data.len() < 5 {
        return Err(Error::InvalidFormat(
            "nbit filter missing datatype parameters".into(),
        ));
    }
    let nparams = nbit_usize(client_data[0], "nbit parameter count")?;
    if nparams != client_data.len() {
        return Err(Error::InvalidFormat(format!(
            "nbit parameter count mismatch: header says {nparams}, got {}",
            client_data.len()
        )));
    }
    if client_data[1] != 0 {
        return Ok(());
    }
    if client_data[4] == 0 {
        return Err(Error::InvalidFormat("nbit datatype size is zero".into()));
    }
    let mut pidx = 4usize;
    validate_nbit_type(client_data, &mut pidx, client_data[3])?;
    Ok(())
}

pub fn set_parms_compound(client_data: &[u32]) -> Result<usize> {
    if client_data.len() < 6 || client_data[3] != NBIT_COMPOUND {
        return Err(Error::InvalidFormat(
            "nbit compound parameters are missing".into(),
        ));
    }
    let mut pidx = 4usize;
    validate_nbit_compound(client_data, &mut pidx)
}

pub fn set_local_nbit(client_data: &[u32]) -> Result<()> {
    can_apply_nbit(client_data)
}

fn nbit_get_parms_atomic(parms: &[u32], pidx: &mut usize, size: usize) -> Result<AtomicParms> {
    let parsed = AtomicParms {
        size,
        order: take(parms, pidx)?,
        precision: take_usize(parms, pidx, "nbit precision")?,
        offset: take_usize(parms, pidx, "nbit bit offset")?,
    };
    validate_atomic(parsed)?;
    Ok(parsed)
}

fn compress_array(
    input: &[u8],
    data_offset: usize,
    writer: &mut BitWriter,
    parms: &[u32],
    pidx: &mut usize,
) -> Result<()> {
    let total_size = take_usize(parms, pidx, "nbit array total size")?;
    let base_class = take(parms, pidx)?;

    match base_class {
        NBIT_ATOMIC => {
            let p = AtomicParms {
                size: take_usize(parms, pidx, "nbit atomic size")?,
                order: take(parms, pidx)?,
                precision: take_usize(parms, pidx, "nbit precision")?,
                offset: take_usize(parms, pidx, "nbit bit offset")?,
            };
            validate_atomic(p)?;
            if total_size % p.size != 0 {
                return Err(Error::InvalidFormat(
                    "nbit array element size is not a multiple of base size".into(),
                ));
            }
            for idx in 0..(total_size / p.size) {
                let offset = nbit_nested_offset(data_offset, idx, p.size)?;
                compress_atomic(input, offset, writer, p)?;
            }
        }
        NBIT_ARRAY | NBIT_COMPOUND => {
            let base_size = nbit_usize(
                parms
                    .get(*pidx)
                    .copied()
                    .ok_or_else(|| Error::InvalidFormat("nbit missing nested size".into()))?,
                "nbit nested size",
            )?;
            if base_size == 0 || total_size % base_size != 0 {
                return Err(Error::InvalidFormat(
                    "nbit array element size is not a multiple of nested size".into(),
                ));
            }
            let begin = *pidx;
            for idx in 0..(total_size / base_size) {
                *pidx = begin;
                let offset = nbit_nested_offset(data_offset, idx, base_size)?;
                if base_class == NBIT_ARRAY {
                    compress_array(input, offset, writer, parms, pidx)?;
                } else {
                    compress_compound(input, offset, writer, parms, pidx)?;
                }
            }
        }
        NBIT_NOOPTYPE => {
            let _size = take(parms, pidx)?;
            compress_one_nooptype(input, data_offset, total_size, writer)?;
        }
        other => {
            return Err(Error::InvalidFormat(format!(
                "invalid nbit array base class {other}"
            )));
        }
    }
    Ok(())
}

fn compress_compound(
    input: &[u8],
    data_offset: usize,
    writer: &mut BitWriter,
    parms: &[u32],
    pidx: &mut usize,
) -> Result<()> {
    let size = take_usize(parms, pidx, "nbit compound size")?;
    let nmembers = take_usize(parms, pidx, "nbit compound member count")?;
    for _ in 0..nmembers {
        let member_offset = take_usize(parms, pidx, "nbit compound member offset")?;
        let member_class = take(parms, pidx)?;
        let member_size = nbit_usize(
            parms
                .get(*pidx)
                .copied()
                .ok_or_else(|| Error::InvalidFormat("nbit missing compound member size".into()))?,
            "nbit compound member size",
        )?;
        if member_offset
            .checked_add(member_size)
            .ok_or_else(|| Error::InvalidFormat("nbit compound member bounds overflow".into()))?
            > size
        {
            return Err(Error::InvalidFormat(
                "nbit compound member exceeds compound bounds".into(),
            ));
        }
        let offset = data_offset
            .checked_add(member_offset)
            .ok_or_else(|| Error::InvalidFormat("nbit compound member offset overflow".into()))?;
        match member_class {
            NBIT_ATOMIC => {
                let p = AtomicParms {
                    size: take_usize(parms, pidx, "nbit atomic size")?,
                    order: take(parms, pidx)?,
                    precision: take_usize(parms, pidx, "nbit precision")?,
                    offset: take_usize(parms, pidx, "nbit bit offset")?,
                };
                validate_atomic(p)?;
                compress_atomic(input, offset, writer, p)?;
            }
            NBIT_ARRAY => compress_array(input, offset, writer, parms, pidx)?,
            NBIT_COMPOUND => compress_compound(input, offset, writer, parms, pidx)?,
            NBIT_NOOPTYPE => {
                let _size = take(parms, pidx)?;
                compress_one_nooptype(input, offset, member_size, writer)?;
            }
            other => {
                return Err(Error::InvalidFormat(format!(
                    "invalid nbit compound member class {other}"
                )));
            }
        }
    }
    Ok(())
}

fn compress_one_nooptype(
    input: &[u8],
    offset: usize,
    size: usize,
    writer: &mut BitWriter,
) -> Result<()> {
    let end = offset
        .checked_add(size)
        .ok_or_else(|| Error::InvalidFormat("nbit input offset overflow".into()))?;
    let window = input
        .get(offset..end)
        .ok_or_else(|| Error::InvalidFormat("nbit input offset out of range".into()))?;
    for &byte in window {
        writer.write_bits(u16::from(byte), 8)?;
    }
    Ok(())
}

fn decompress_array(
    out: &mut [u8],
    data_offset: usize,
    stream: &mut BitStream<'_>,
    parms: &[u32],
    pidx: &mut usize,
) -> Result<()> {
    let total_size = take_usize(parms, pidx, "nbit array total size")?;
    let base_class = take(parms, pidx)?;

    match base_class {
        NBIT_ATOMIC => {
            let p = AtomicParms {
                size: take_usize(parms, pidx, "nbit atomic size")?,
                order: take(parms, pidx)?,
                precision: take_usize(parms, pidx, "nbit precision")?,
                offset: take_usize(parms, pidx, "nbit bit offset")?,
            };
            validate_atomic(p)?;
            if total_size % p.size != 0 {
                return Err(Error::InvalidFormat(
                    "nbit array element size is not a multiple of base size".into(),
                ));
            }
            let count = total_size / p.size;
            for idx in 0..count {
                let offset = nbit_nested_offset(data_offset, idx, p.size)?;
                decompress_atomic(out, offset, stream, p)?;
            }
        }
        NBIT_ARRAY | NBIT_COMPOUND => {
            let base_size = nbit_usize(
                parms
                    .get(*pidx)
                    .copied()
                    .ok_or_else(|| Error::InvalidFormat("nbit missing nested size".into()))?,
                "nbit nested size",
            )?;
            if base_size == 0 {
                return Err(Error::InvalidFormat(
                    "nbit nested datatype size is zero".into(),
                ));
            }
            if total_size % base_size != 0 {
                return Err(Error::InvalidFormat(
                    "nbit array element size is not a multiple of nested size".into(),
                ));
            }
            let count = total_size / base_size;
            let begin = *pidx;
            for idx in 0..count {
                *pidx = begin;
                let offset = nbit_nested_offset(data_offset, idx, base_size)?;
                if base_class == NBIT_ARRAY {
                    decompress_array(out, offset, stream, parms, pidx)?;
                } else {
                    decompress_compound(out, offset, stream, parms, pidx)?;
                }
            }
        }
        NBIT_NOOPTYPE => {
            let _size = take(parms, pidx)?;
            stream.copy_bytes(out, data_offset, total_size)?;
        }
        other => {
            return Err(Error::InvalidFormat(format!(
                "invalid nbit array base class {other}"
            )));
        }
    }

    Ok(())
}

fn decompress_compound(
    out: &mut [u8],
    data_offset: usize,
    stream: &mut BitStream<'_>,
    parms: &[u32],
    pidx: &mut usize,
) -> Result<()> {
    let size = take_usize(parms, pidx, "nbit compound size")?;
    let nmembers = take_usize(parms, pidx, "nbit compound member count")?;
    let mut used_size = 0usize;

    for _ in 0..nmembers {
        let member_offset = take_usize(parms, pidx, "nbit compound member offset")?;
        let member_class = take(parms, pidx)?;
        let member_size = nbit_usize(
            parms
                .get(*pidx)
                .copied()
                .ok_or_else(|| Error::InvalidFormat("nbit missing compound member size".into()))?,
            "nbit compound member size",
        )?;

        used_size = used_size
            .checked_add(member_size)
            .ok_or_else(|| Error::InvalidFormat("nbit compound member size overflow".into()))?;
        let member_end = member_offset
            .checked_add(member_size)
            .ok_or_else(|| Error::InvalidFormat("nbit compound member bounds overflow".into()))?;
        if used_size > size || member_end > size {
            return Err(Error::InvalidFormat(
                "nbit compound member exceeds compound bounds".into(),
            ));
        }

        match member_class {
            NBIT_ATOMIC => {
                let p = AtomicParms {
                    size: take_usize(parms, pidx, "nbit atomic size")?,
                    order: take(parms, pidx)?,
                    precision: take_usize(parms, pidx, "nbit precision")?,
                    offset: take_usize(parms, pidx, "nbit bit offset")?,
                };
                validate_atomic(p)?;
                let offset = data_offset.checked_add(member_offset).ok_or_else(|| {
                    Error::InvalidFormat("nbit compound member offset overflow".into())
                })?;
                decompress_atomic(out, offset, stream, p)?;
            }
            NBIT_ARRAY => {
                let offset = data_offset.checked_add(member_offset).ok_or_else(|| {
                    Error::InvalidFormat("nbit compound member offset overflow".into())
                })?;
                decompress_array(out, offset, stream, parms, pidx)?;
            }
            NBIT_COMPOUND => {
                let offset = data_offset.checked_add(member_offset).ok_or_else(|| {
                    Error::InvalidFormat("nbit compound member offset overflow".into())
                })?;
                decompress_compound(out, offset, stream, parms, pidx)?;
            }
            NBIT_NOOPTYPE => {
                let _size = take(parms, pidx)?;
                let offset = data_offset.checked_add(member_offset).ok_or_else(|| {
                    Error::InvalidFormat("nbit compound member offset overflow".into())
                })?;
                stream.copy_bytes(out, offset, member_size)?;
            }
            other => {
                return Err(Error::InvalidFormat(format!(
                    "invalid nbit compound member class {other}"
                )));
            }
        }
    }

    Ok(())
}

fn nbit_nested_offset(base: usize, idx: usize, element_size: usize) -> Result<usize> {
    let rel = idx
        .checked_mul(element_size)
        .ok_or_else(|| Error::InvalidFormat("nbit nested output offset overflow".into()))?;
    base.checked_add(rel)
        .ok_or_else(|| Error::InvalidFormat("nbit nested output offset overflow".into()))
}

fn decompress_atomic(
    out: &mut [u8],
    data_offset: usize,
    stream: &mut BitStream<'_>,
    parms: AtomicParms,
) -> Result<()> {
    let dtype_bits = parms.size * 8;
    if parms.order == NBIT_ORDER_LE {
        let begin = if (parms.precision + parms.offset) % 8 != 0 {
            (parms.precision + parms.offset) / 8
        } else {
            (parms.precision + parms.offset) / 8 - 1
        };
        let end = parms.offset / 8;
        for k in (end..=begin).rev() {
            decompress_atomic_byte(out, data_offset, k, begin, end, stream, parms, dtype_bits)?;
        }
    } else if parms.order == NBIT_ORDER_BE {
        let begin = (dtype_bits - parms.precision - parms.offset) / 8;
        let end = if parms.offset % 8 != 0 {
            (dtype_bits - parms.offset) / 8
        } else {
            (dtype_bits - parms.offset) / 8 - 1
        };
        for k in begin..=end {
            decompress_atomic_byte(out, data_offset, k, begin, end, stream, parms, dtype_bits)?;
        }
    } else {
        return Err(Error::InvalidFormat(format!(
            "invalid nbit byte order {}",
            parms.order
        )));
    }

    Ok(())
}

fn compress_atomic(
    input: &[u8],
    data_offset: usize,
    writer: &mut BitWriter,
    parms: AtomicParms,
) -> Result<()> {
    let dtype_bits = parms.size * 8;
    if parms.order == NBIT_ORDER_LE {
        let begin = if (parms.precision + parms.offset) % 8 != 0 {
            (parms.precision + parms.offset) / 8
        } else {
            (parms.precision + parms.offset) / 8 - 1
        };
        let end = parms.offset / 8;
        for k in (end..=begin).rev() {
            compress_atomic_byte(input, data_offset, k, begin, end, writer, parms, dtype_bits)?;
        }
    } else if parms.order == NBIT_ORDER_BE {
        let begin = (dtype_bits - parms.precision - parms.offset) / 8;
        let end = if parms.offset % 8 != 0 {
            (dtype_bits - parms.offset) / 8
        } else {
            (dtype_bits - parms.offset) / 8 - 1
        };
        for k in begin..=end {
            compress_atomic_byte(input, data_offset, k, begin, end, writer, parms, dtype_bits)?;
        }
    } else {
        return Err(Error::InvalidFormat(format!(
            "invalid nbit byte order {}",
            parms.order
        )));
    }
    Ok(())
}

fn decompress_atomic_byte(
    out: &mut [u8],
    data_offset: usize,
    k: usize,
    begin: usize,
    end: usize,
    stream: &mut BitStream<'_>,
    parms: AtomicParms,
    dtype_bits: usize,
) -> Result<()> {
    let (dat_offset, dat_len) = if begin != end {
        if k == begin {
            (0, 8 - (dtype_bits - parms.precision - parms.offset) % 8)
        } else if k == end {
            let len = 8 - parms.offset % 8;
            (8 - len, len)
        } else {
            (0, 8)
        }
    } else {
        (parms.offset % 8, parms.precision)
    };

    let bits = stream.read_bits(dat_len)? as u8;
    let out_idx = data_offset
        .checked_add(k)
        .ok_or_else(|| Error::InvalidFormat("nbit output offset overflow".into()))?;
    if out_idx >= out.len() {
        return Err(Error::InvalidFormat(
            "nbit output offset out of range".into(),
        ));
    }
    out[out_idx] |= bits << dat_offset;
    Ok(())
}

fn compress_atomic_byte(
    input: &[u8],
    data_offset: usize,
    k: usize,
    begin: usize,
    end: usize,
    writer: &mut BitWriter,
    parms: AtomicParms,
    dtype_bits: usize,
) -> Result<()> {
    let (dat_offset, dat_len) = if begin != end {
        if k == begin {
            (0, 8 - (dtype_bits - parms.precision - parms.offset) % 8)
        } else if k == end {
            let len = 8 - parms.offset % 8;
            (8 - len, len)
        } else {
            (0, 8)
        }
    } else {
        (parms.offset % 8, parms.precision)
    };
    let idx = data_offset
        .checked_add(k)
        .ok_or_else(|| Error::InvalidFormat("nbit input offset overflow".into()))?;
    let byte = *input
        .get(idx)
        .ok_or_else(|| Error::InvalidFormat("nbit input offset out of range".into()))?;
    let mask = if dat_len == 8 {
        0xff
    } else {
        ((1u16 << dat_len) - 1) as u8
    };
    writer.write_bits(u16::from((byte >> dat_offset) & mask), dat_len)
}

fn validate_atomic(parms: AtomicParms) -> Result<()> {
    let dtype_bits = parms
        .size
        .checked_mul(8)
        .ok_or_else(|| Error::InvalidFormat("invalid nbit datatype precision/offset".into()))?;
    let precision_end = parms
        .precision
        .checked_add(parms.offset)
        .ok_or_else(|| Error::InvalidFormat("invalid nbit datatype precision/offset".into()))?;
    if parms.size == 0
        || parms.precision == 0
        || parms.precision > dtype_bits
        || precision_end > dtype_bits
    {
        return Err(Error::InvalidFormat(
            "invalid nbit datatype precision/offset".into(),
        ));
    }
    if parms.order != NBIT_ORDER_LE && parms.order != NBIT_ORDER_BE {
        return Err(Error::InvalidFormat(format!(
            "invalid nbit byte order {}",
            parms.order
        )));
    }
    Ok(())
}

fn validate_nbit_type(parms: &[u32], pidx: &mut usize, class: u32) -> Result<usize> {
    match class {
        NBIT_ATOMIC => {
            let size = take_usize(parms, pidx, "nbit atomic size")?;
            let p = nbit_get_parms_atomic(parms, pidx, size)?;
            Ok(p.size)
        }
        NBIT_ARRAY => {
            let total_size = take_usize(parms, pidx, "nbit array total size")?;
            let base_class = take(parms, pidx)?;
            if matches!(
                base_class,
                NBIT_ATOMIC | NBIT_ARRAY | NBIT_COMPOUND | NBIT_NOOPTYPE
            ) && parms.get(*pidx).copied() == Some(0)
            {
                return Err(Error::InvalidFormat(
                    "nbit nested datatype size is zero".into(),
                ));
            }
            let begin = *pidx;
            let base_size = validate_nbit_type(parms, pidx, base_class)?;
            if base_size == 0 || total_size % base_size != 0 {
                return Err(Error::InvalidFormat(
                    "nbit array element size is not a multiple of base size".into(),
                ));
            }
            *pidx = (*pidx).max(begin);
            Ok(total_size)
        }
        NBIT_COMPOUND => validate_nbit_compound(parms, pidx),
        NBIT_NOOPTYPE => take_usize(parms, pidx, "nbit noop size"),
        other => Err(Error::InvalidFormat(format!(
            "invalid nbit datatype class {other}"
        ))),
    }
}

fn validate_nbit_compound(parms: &[u32], pidx: &mut usize) -> Result<usize> {
    let size = take_usize(parms, pidx, "nbit compound size")?;
    let nmembers = take_usize(parms, pidx, "nbit compound member count")?;
    for _ in 0..nmembers {
        let member_offset = take_usize(parms, pidx, "nbit compound member offset")?;
        let member_class = take(parms, pidx)?;
        let member_size = validate_nbit_type(parms, pidx, member_class)?;
        let member_end = member_offset
            .checked_add(member_size)
            .ok_or_else(|| Error::InvalidFormat("nbit compound member bounds overflow".into()))?;
        if member_end > size {
            return Err(Error::InvalidFormat(
                "nbit compound member exceeds compound bounds".into(),
            ));
        }
    }
    Ok(size)
}

fn take(parms: &[u32], pidx: &mut usize) -> Result<u32> {
    let value = *parms
        .get(*pidx)
        .ok_or_else(|| Error::InvalidFormat("truncated nbit parameters".into()))?;
    *pidx += 1;
    Ok(value)
}

fn take_usize(parms: &[u32], pidx: &mut usize, context: &'static str) -> Result<usize> {
    nbit_usize(take(parms, pidx)?, context)
}

fn nbit_usize(value: u32, context: &'static str) -> Result<usize> {
    usize::try_from(value)
        .map_err(|_| Error::InvalidFormat(format!("{context} does not fit in usize")))
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
            return Err(Error::InvalidFormat("nbit bit run too long".into()));
        }

        let mut value = 0u16;
        while nbits > 0 {
            let byte = *self
                .data
                .get(self.byte)
                .ok_or_else(|| Error::InvalidFormat("nbit data too short".into()))?;
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

    fn copy_bytes(&mut self, out: &mut [u8], offset: usize, size: usize) -> Result<()> {
        let end = offset
            .checked_add(size)
            .ok_or_else(|| Error::InvalidFormat("nbit output offset overflow".into()))?;
        let window = out
            .get_mut(offset..end)
            .ok_or_else(|| Error::InvalidFormat("nbit output offset out of range".into()))?;
        for byte in window {
            *byte = self.read_bits(8)? as u8;
        }
        Ok(())
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
            return Err(Error::InvalidFormat("nbit bit run too long".into()));
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
                .map_err(|_| Error::InvalidFormat("nbit bit run exceeds byte".into()))?;
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

    fn atomic_params(precision: u32, offset: u32) -> Vec<u32> {
        vec![8, 0, 1, NBIT_ATOMIC, 2, NBIT_ORDER_LE, precision, offset]
    }

    #[test]
    fn rejects_zero_precision() {
        let err = decompress(&[], &atomic_params(0, 0)).unwrap_err();
        assert!(
            err.to_string()
                .contains("invalid nbit datatype precision/offset"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_precision_larger_than_datatype() {
        let err = decompress(&[], &atomic_params(17, 0)).unwrap_err();
        assert!(
            err.to_string()
                .contains("invalid nbit datatype precision/offset"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_precision_plus_offset_larger_than_datatype() {
        let err = decompress(&[], &atomic_params(12, 8)).unwrap_err();
        assert!(
            err.to_string()
                .contains("invalid nbit datatype precision/offset"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_invalid_atomic_byte_order_even_for_empty_chunks() {
        let mut params = atomic_params(8, 0);
        params[5] = 99;
        params[2] = 0;
        let err = decompress(&[], &params).unwrap_err();
        assert!(
            err.to_string().contains("invalid nbit byte order 99"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_zero_top_level_datatype_size() {
        let params = vec![8, 0, 0, NBIT_ATOMIC, 0, NBIT_ORDER_LE, 8, 0];
        let err = decompress(&[], &params).unwrap_err();
        assert!(
            err.to_string().contains("nbit datatype size is zero"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_zero_nested_array_base_size() {
        let params = vec![7, 0, 1, NBIT_ARRAY, 4, NBIT_ARRAY, 0];
        let err = decompress(&[], &params).unwrap_err();
        assert!(
            err.to_string()
                .contains("nbit nested datatype size is zero"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_array_size_not_multiple_of_atomic_base_size() {
        let params = vec![10, 0, 1, NBIT_ARRAY, 3, NBIT_ATOMIC, 2, NBIT_ORDER_LE, 8, 0];
        let err = decompress(&[], &params).unwrap_err();
        assert!(
            err.to_string()
                .contains("nbit array element size is not a multiple of base size"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn copy_bytes_rejects_output_offset_overflow() {
        let mut stream = BitStream::new(&[]);
        let mut out = [];
        let err = stream.copy_bytes(&mut out, usize::MAX, 1).unwrap_err();
        assert!(
            err.to_string().contains("nbit output offset overflow"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn copy_bytes_rejects_output_offset_out_of_range() {
        let mut stream = BitStream::new(&[]);
        let mut out = [0u8; 1];
        let err = stream.copy_bytes(&mut out, 1, 1).unwrap_err();
        assert!(
            err.to_string().contains("nbit output offset out of range"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn nested_offset_rejects_relative_overflow() {
        let err = nbit_nested_offset(0, usize::MAX, 2).unwrap_err();
        assert!(
            err.to_string()
                .contains("nbit nested output offset overflow"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn nested_offset_rejects_base_overflow() {
        let err = nbit_nested_offset(usize::MAX, 1, 1).unwrap_err();
        assert!(
            err.to_string()
                .contains("nbit nested output offset overflow"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn nbit_atomic_compress_roundtrips() {
        let params = atomic_params(12, 2);
        let input = [0b0011_1100, 0b0000_0011];
        let compressed = nbit_compress(&input, &params).unwrap();
        let decoded = decompress(&compressed, &params).unwrap();
        assert_eq!(decoded, input);
    }
}
