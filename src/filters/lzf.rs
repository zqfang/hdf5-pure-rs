use crate::error::{Error, Result};

/// Decompress LZF-compressed data into an exactly-sized output buffer.
///
/// LZF is a very fast, low-ratio compression algorithm.
/// Format: sequence of literal runs and back-references.
///
/// Each chunk starts with a control byte:
/// - If high 3 bits == 0: literal run of (control + 1) bytes follows
/// - Otherwise: back-reference: length = high 3 bits + 2 (or read next byte for long match),
///   offset = ((control & 0x1f) << 8) | next_byte + 1
pub fn decompress_into(data: &[u8], output: &mut [u8]) -> Result<()> {
    let mut scratch = vec![0; output.len()];
    decompress_checked(data, &mut scratch)?;
    output.copy_from_slice(&scratch);
    Ok(())
}

fn decompress_checked(data: &[u8], output: &mut [u8]) -> Result<()> {
    let expected_size = output.len();
    let mut ip = 0; // input position
    let mut op: usize = 0; // output position

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
            if op.checked_add(count).is_none_or(|len| len > expected_size) {
                return Err(Error::InvalidFormat(format!(
                    "lzf: literal run exceeds expected output size {expected_size}"
                )));
            }
            output[op..op + count].copy_from_slice(&data[ip..literal_end]);
            op += count;
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

            if offset > op {
                return Err(Error::InvalidFormat(format!(
                    "lzf: back-reference offset {} exceeds output size {}",
                    offset, op
                )));
            }
            if op.checked_add(length).is_none_or(|len| len > expected_size) {
                return Err(Error::InvalidFormat(format!(
                    "lzf: back-reference exceeds expected output size {expected_size}"
                )));
            }

            let start = op - offset;
            for i in 0..length {
                let src = start.checked_add(i).ok_or_else(|| {
                    Error::InvalidFormat("lzf: back-reference source offset overflow".into())
                })?;
                let byte = *output.get(src).ok_or_else(|| {
                    Error::InvalidFormat("lzf: back-reference source out of range".into())
                })?;
                output[op] = byte;
                op += 1;
            }
        }
    }

    if op != expected_size {
        return Err(Error::InvalidFormat(format!(
            "lzf: output length mismatch: expected {expected_size}, got {op}"
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lzf_literal_only() {
        // Control byte 0x04 = literal run of 5 bytes
        let compressed = vec![0x04, b'H', b'e', b'l', b'l', b'o'];
        let mut result = vec![0; 5];
        decompress_into(&compressed, &mut result).unwrap();
        assert_eq!(result, b"Hello");
    }

    #[test]
    fn test_lzf_with_backref() {
        // "abcabc" = literal "abc" + backref to position 0 length 3
        // literal: ctrl=2 (3-1), then 'a','b','c'
        // backref: length=3 (1 in high bits = 3-2=1, shifted = 0x20), offset=3
        //   ctrl = (1 << 5) | 0 = 0x20, next_byte = 2 (offset=0*256+2+1=3)
        let compressed = vec![0x02, b'a', b'b', b'c', 0x20, 0x02];
        let mut result = vec![0; 6];
        decompress_into(&compressed, &mut result).unwrap();
        assert_eq!(result, b"abcabc");
    }

    #[test]
    fn test_lzf_rejects_output_size_mismatch() {
        let compressed = vec![0x04, b'H', b'e', b'l', b'l', b'o'];
        let mut result = vec![0; 4];
        let err = decompress_into(&compressed, &mut result).unwrap_err();
        assert!(
            err.to_string()
                .contains("lzf: literal run exceeds expected output size 4"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_lzf_rejects_backref_past_expected_output_size() {
        let compressed = vec![0x00, b'a', 0x20, 0x00];
        let mut result = vec![0; 2];
        let err = decompress_into(&compressed, &mut result).unwrap_err();
        assert!(
            err.to_string()
                .contains("lzf: back-reference exceeds expected output size 2"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_lzf_rejects_literal_run_past_input() {
        let compressed = vec![0x04, b'H', b'e'];
        let mut result = vec![0; 5];
        let err = decompress_into(&compressed, &mut result).unwrap_err();
        assert!(
            err.to_string().contains("lzf: literal run exceeds input"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_lzf_preserves_output_on_late_backref_error() {
        let compressed = vec![0x02, b'a', b'b', b'c', 0x20, 0x04];
        let mut result = b"stale!".to_vec();
        let err = decompress_into(&compressed, &mut result).unwrap_err();
        assert!(
            err.to_string()
                .contains("lzf: back-reference offset 5 exceeds output size 3"),
            "unexpected error: {err}"
        );
        assert_eq!(result, b"stale!");
    }

    #[test]
    fn test_lzf_preserves_output_on_literal_and_final_length_errors() {
        let mut result = b"stale".to_vec();
        let err = decompress_into(&[0x04, b'H', b'e'], &mut result).unwrap_err();
        assert!(
            err.to_string().contains("lzf: literal run exceeds input"),
            "unexpected error: {err}"
        );
        assert_eq!(result, b"stale");

        let err = decompress_into(&[0x00, b'a'], &mut result).unwrap_err();
        assert!(
            err.to_string()
                .contains("lzf: output length mismatch: expected 5, got 1"),
            "unexpected error: {err}"
        );
        assert_eq!(result, b"stale");
    }

    #[test]
    fn test_lzf_preserves_output_on_truncated_backref_headers() {
        let mut result = b"stale".to_vec();

        let err = decompress_into(&[0xe0], &mut result).unwrap_err();
        assert!(
            err.to_string()
                .contains("lzf: unexpected end in long match"),
            "unexpected error: {err}"
        );
        assert_eq!(result, b"stale");

        let err = decompress_into(&[0x20], &mut result).unwrap_err();
        assert!(
            err.to_string()
                .contains("lzf: unexpected end reading offset"),
            "unexpected error: {err}"
        );
        assert_eq!(result, b"stale");
    }
}
