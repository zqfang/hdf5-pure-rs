use crate::error::{Error, Result};

pub(super) fn read_le_uint(bytes: &[u8], size: usize) -> Result<u64> {
    if size == 0 || size > 8 || bytes.len() < size {
        return Err(Error::InvalidFormat(format!(
            "invalid little-endian integer size {size}"
        )));
    }

    let mut value = 0u64;
    for (idx, byte) in bytes[..size].iter().enumerate() {
        value |= u64::from(*byte) << (idx * 8);
    }
    Ok(value)
}

pub(super) fn usize_from_u64(value: u64, context: &str) -> Result<usize> {
    usize::try_from(value)
        .map_err(|_| Error::InvalidFormat(format!("{context} does not fit in usize")))
}

pub(super) fn u64_from_usize(value: usize, context: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| Error::InvalidFormat(format!("{context} does not fit in u64")))
}

pub(super) fn read_u8_at(bytes: &[u8], pos: &mut usize) -> Result<u8> {
    let value = *bytes
        .get(*pos)
        .ok_or_else(|| Error::InvalidFormat("truncated byte field".into()))?;
    *pos += 1;
    Ok(value)
}

pub(super) fn read_le_u32_at(bytes: &[u8], pos: &mut usize) -> Result<u32> {
    let end = pos
        .checked_add(4)
        .ok_or_else(|| Error::InvalidFormat("truncated u32 field".into()))?;
    if end > bytes.len() {
        return Err(Error::InvalidFormat("truncated u32 field".into()));
    }
    let value = read_le_u32(bytes, *pos)?;
    *pos = end;
    Ok(value)
}

pub(super) fn read_le_u32(bytes: &[u8], pos: usize) -> Result<u32> {
    let end = pos
        .checked_add(4)
        .ok_or_else(|| Error::InvalidFormat("truncated u32 field".into()))?;
    let window = bytes
        .get(pos..end)
        .ok_or_else(|| Error::InvalidFormat("truncated u32 field".into()))?;
    Ok(u32::from_le_bytes([
        window[0], window[1], window[2], window[3],
    ]))
}

pub(super) fn read_le_uint_at(bytes: &[u8], pos: &mut usize, size: usize) -> Result<u64> {
    let end = pos
        .checked_add(size)
        .ok_or_else(|| Error::InvalidFormat("truncated integer field".into()))?;
    if end > bytes.len() {
        return Err(Error::InvalidFormat("truncated integer field".into()));
    }
    let value = read_le_uint(&bytes[*pos..end], size)?;
    *pos = end;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_le_u32_rejects_position_overflow() {
        let err = read_le_u32(&[1, 2, 3, 4], usize::MAX - 1).unwrap_err();
        assert!(
            err.to_string().contains("truncated u32 field"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn read_le_u32_at_rejects_position_overflow() {
        let mut pos = usize::MAX - 1;
        let err = read_le_u32_at(&[1, 2, 3, 4], &mut pos).unwrap_err();
        assert!(
            err.to_string().contains("truncated u32 field"),
            "unexpected error: {err}"
        );
        assert_eq!(pos, usize::MAX - 1);
    }

    #[test]
    fn read_le_uint_at_rejects_position_overflow() {
        let mut pos = usize::MAX - 1;
        let err = read_le_uint_at(&[1, 2, 3, 4], &mut pos, 4).unwrap_err();
        assert!(
            err.to_string().contains("truncated integer field"),
            "unexpected error: {err}"
        );
        assert_eq!(pos, usize::MAX - 1);
    }
}
