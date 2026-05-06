use crate::error::{Error, Result};

/// Verify Fletcher32 checksum and strip it from the data.
/// The last 4 bytes of the data are the checksum (stored little-endian).
pub fn verify_and_strip(data: &[u8]) -> Result<Vec<u8>> {
    if data.len() < 4 {
        return Err(Error::InvalidFormat(
            "data too short for fletcher32 checksum".into(),
        ));
    }

    let checksum_offset = data
        .len()
        .checked_sub(4)
        .ok_or_else(|| Error::InvalidFormat("data too short for fletcher32 checksum".into()))?;
    let payload = checked_window(data, 0, checksum_offset, "fletcher32 payload")?;
    // Stored checksum is little-endian (UINT32ENCODE in HDF5 C library)
    let stored = read_u32_le_at(data, checksum_offset, "fletcher32 stored checksum")?;

    let computed = fletcher32(payload);

    // HDF5 also checks a byte-swapped version for compatibility with pre-1.6.3
    let reversed = fletcher32_reversed(computed);

    if stored != computed && stored != reversed {
        return Err(Error::InvalidFormat(format!(
            "fletcher32 checksum mismatch: stored={stored:#010x}, computed={computed:#010x}"
        )));
    }

    Ok(payload.to_vec())
}

/// Append an HDF5 Fletcher32 checksum to a filter payload.
pub fn append_checksum(data: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(
        data.len()
            .checked_add(4)
            .ok_or_else(|| Error::InvalidFormat("fletcher32 output size overflow".into()))?,
    );
    out.extend_from_slice(data);
    out.extend_from_slice(&fletcher32(data).to_le_bytes());
    Ok(out)
}

/// Compute Fletcher32 checksum matching the HDF5 C library implementation.
/// Data is processed as big-endian 16-bit words.
fn fletcher32(data: &[u8]) -> u32 {
    let mut sum1: u32 = 0;
    let mut sum2: u32 = 0;

    let even_len = data.len() & !1;
    for batch in data[..even_len].chunks(720) {
        // Process in batches of 360 words to avoid overflow.
        for word in batch.chunks_exact(2) {
            // Big-endian 16-bit word (matching HDF5 C library)
            let val = (u32::from(word[0]) << 8) | u32::from(word[1]);
            sum1 += val;
            sum2 += sum1;
        }

        // Ones-complement reduction
        sum1 = (sum1 & 0xffff) + (sum1 >> 16);
        sum2 = (sum2 & 0xffff) + (sum2 >> 16);
    }

    // Handle odd byte
    if data.len() % 2 != 0 {
        sum1 += u32::from(data[even_len]) << 8;
        sum2 += sum1;
        sum1 = (sum1 & 0xffff) + (sum1 >> 16);
        sum2 = (sum2 & 0xffff) + (sum2 >> 16);
    }

    // Final reduction
    sum1 = (sum1 & 0xffff) + (sum1 >> 16);
    sum2 = (sum2 & 0xffff) + (sum2 >> 16);

    (sum2 << 16) | sum1
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

fn read_u32_le_at(data: &[u8], offset: usize, context: &str) -> Result<u32> {
    let bytes = checked_window(data, offset, 4, context)?;
    Ok(u32::from_le_bytes(bytes.try_into().map_err(|_| {
        Error::InvalidFormat(format!("{context} is truncated"))
    })?))
}

/// Compute the reversed (byte-swapped) checksum for pre-1.6.3 compatibility.
fn fletcher32_reversed(checksum: u32) -> u32 {
    let bytes = checksum.to_ne_bytes();
    u32::from_ne_bytes([bytes[1], bytes[0], bytes[3], bytes[2]])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_window_rejects_offset_overflow() {
        let err = checked_window(&[], usize::MAX, 1, "fletcher32 test window").unwrap_err();
        assert!(
            err.to_string()
                .contains("fletcher32 test window offset overflow"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn append_checksum_roundtrips_through_verify() {
        let payload = b"fletcher32 write payload";
        let encoded = append_checksum(payload).unwrap();
        assert_eq!(verify_and_strip(&encoded).unwrap(), payload);
    }
}
