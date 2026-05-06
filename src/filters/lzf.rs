use crate::error::{Error, Result};

/// Decompress LZF-compressed data.
///
/// LZF is a very fast, low-ratio compression algorithm.
/// Format: sequence of literal runs and back-references.
///
/// Each chunk starts with a control byte:
/// - If high 3 bits == 0: literal run of (control + 1) bytes follows
/// - Otherwise: back-reference: length = high 3 bits + 2 (or read next byte for long match),
///   offset = ((control & 0x1f) << 8) | next_byte + 1
pub fn decompress(data: &[u8], expected_size: usize) -> Result<Vec<u8>> {
    let mut output = Vec::with_capacity(expected_size);
    let mut ip = 0; // input position

    while ip < data.len() {
        let ctrl = usize::from(data[ip]);
        ip += 1;

        if ctrl < 32 {
            // Literal run: copy (ctrl + 1) bytes
            let count = ctrl + 1;
            let literal_end = ip.checked_add(count).ok_or_else(|| {
                Error::InvalidFormat("lzf: literal run input offset overflow".into())
            })?;
            if literal_end > data.len() {
                return Err(Error::InvalidFormat(
                    "lzf: literal run exceeds input".into(),
                ));
            }
            if output
                .len()
                .checked_add(count)
                .is_none_or(|len| len > expected_size)
            {
                return Err(Error::InvalidFormat(format!(
                    "lzf: literal run exceeds expected output size {expected_size}"
                )));
            }
            output.extend_from_slice(&data[ip..literal_end]);
            ip = literal_end;
        } else {
            // Back-reference
            let mut length = ctrl >> 5;
            let mut offset = (ctrl & 0x1f) << 8;

            if length == 7 {
                // Long match: read additional length byte
                if ip >= data.len() {
                    return Err(Error::InvalidFormat(
                        "lzf: unexpected end in long match".into(),
                    ));
                }
                length += usize::from(data[ip]);
                ip += 1;
            }
            length += 2; // minimum match length is 2

            if ip >= data.len() {
                return Err(Error::InvalidFormat(
                    "lzf: unexpected end reading offset".into(),
                ));
            }
            offset += usize::from(data[ip]) + 1;
            ip += 1;

            if offset > output.len() {
                return Err(Error::InvalidFormat(format!(
                    "lzf: back-reference offset {} exceeds output size {}",
                    offset,
                    output.len()
                )));
            }
            if output
                .len()
                .checked_add(length)
                .is_none_or(|len| len > expected_size)
            {
                return Err(Error::InvalidFormat(format!(
                    "lzf: back-reference exceeds expected output size {expected_size}"
                )));
            }

            let start = output.len() - offset;
            for i in 0..length {
                let src = start.checked_add(i).ok_or_else(|| {
                    Error::InvalidFormat("lzf: back-reference source offset overflow".into())
                })?;
                let byte = *output.get(src).ok_or_else(|| {
                    Error::InvalidFormat("lzf: back-reference source out of range".into())
                })?;
                output.push(byte);
            }
        }
    }

    if output.len() != expected_size {
        return Err(Error::InvalidFormat(format!(
            "lzf: output length mismatch: expected {expected_size}, got {}",
            output.len()
        )));
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lzf_literal_only() {
        // Control byte 0x04 = literal run of 5 bytes
        let compressed = vec![0x04, b'H', b'e', b'l', b'l', b'o'];
        let result = decompress(&compressed, 5).unwrap();
        assert_eq!(result, b"Hello");
    }

    #[test]
    fn test_lzf_with_backref() {
        // "abcabc" = literal "abc" + backref to position 0 length 3
        // literal: ctrl=2 (3-1), then 'a','b','c'
        // backref: length=3 (1 in high bits = 3-2=1, shifted = 0x20), offset=3
        //   ctrl = (1 << 5) | 0 = 0x20, next_byte = 2 (offset=0*256+2+1=3)
        let compressed = vec![0x02, b'a', b'b', b'c', 0x20, 0x02];
        let result = decompress(&compressed, 6).unwrap();
        assert_eq!(result, b"abcabc");
    }

    #[test]
    fn test_lzf_rejects_output_size_mismatch() {
        let compressed = vec![0x04, b'H', b'e', b'l', b'l', b'o'];
        let err = decompress(&compressed, 4).unwrap_err();
        assert!(
            err.to_string()
                .contains("lzf: literal run exceeds expected output size 4"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_lzf_rejects_backref_past_expected_output_size() {
        let compressed = vec![0x00, b'a', 0x20, 0x00];
        let err = decompress(&compressed, 2).unwrap_err();
        assert!(
            err.to_string()
                .contains("lzf: back-reference exceeds expected output size 2"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_lzf_rejects_literal_run_past_input() {
        let compressed = vec![0x04, b'H', b'e'];
        let err = decompress(&compressed, 5).unwrap_err();
        assert!(
            err.to_string().contains("lzf: literal run exceeds input"),
            "unexpected error: {err}"
        );
    }
}
