use std::io::Read;

use flate2::read::ZlibDecoder;
use flate2::{Decompress, FlushDecompress, Status};

use crate::error::{Error, Result};

/// Decompress deflate (zlib) compressed data and append it to `out`.
pub fn decompress_into(data: &[u8], out: &mut Vec<u8>) -> Result<()> {
    decompress_with_hint_into(data, None, out)
}

/// Decompress deflate (zlib) compressed data and append it to `out`, reserving
/// an optional decoded-size hint before reading.
pub fn decompress_with_hint_into(
    data: &[u8],
    expected_len: Option<usize>,
    out: &mut Vec<u8>,
) -> Result<()> {
    let mut decoded = Vec::new();
    if let Some(expected_len) = expected_len {
        decoded.reserve(expected_len);
    }
    let mut decoder = ZlibDecoder::new(data);
    decoder
        .read_to_end(&mut decoded)
        .map_err(|e| Error::InvalidFormat(format!("deflate decompression failed: {e}")))?;
    if decoder.total_in() != data.len() as u64 {
        return Err(Error::InvalidFormat(
            "deflate decompression left trailing input bytes".into(),
        ));
    }
    out.extend_from_slice(&decoded);
    Ok(())
}

/// Decompress deflate (zlib) compressed data into the provided output buffer
/// and require the decoded size to match exactly.
pub fn decompress_exact_into(data: &[u8], out: &mut [u8]) -> Result<()> {
    let mut decoded = vec![0; out.len()];
    let mut decoder = Decompress::new(true);
    let status = decoder
        .decompress(data, &mut decoded, FlushDecompress::Finish)
        .map_err(|e| Error::InvalidFormat(format!("deflate decompression failed: {e}")))?;

    if status != Status::StreamEnd {
        if decoder.total_out() == out.len() as u64 {
            return Err(Error::InvalidFormat(
                "deflate decompression produced more bytes than expected".into(),
            ));
        }
        return Err(Error::InvalidFormat(
            "deflate decompression ended before zlib stream end".into(),
        ));
    }
    if decoder.total_out() != out.len() as u64 {
        return Err(Error::InvalidFormat(format!(
            "deflate decompression output length mismatch: expected {}, got {}",
            out.len(),
            decoder.total_out()
        )));
    }
    if decoder.total_in() != data.len() as u64 {
        return Err(Error::InvalidFormat(
            "deflate decompression left trailing input bytes".into(),
        ));
    }
    out.copy_from_slice(&decoded);
    Ok(())
}

/// Compress data with deflate at the given level (0-9), appending to `out`.
pub fn compress_into(data: &[u8], level: u32, out: &mut Vec<u8>) -> Result<()> {
    use flate2::write::ZlibEncoder;
    use flate2::Compression;
    use std::io::Write;

    let mut encoder = ZlibEncoder::new(out, Compression::new(level));
    encoder
        .write_all(data)
        .map_err(|e| Error::InvalidFormat(format!("deflate compression failed: {e}")))?;
    encoder
        .finish()
        .map_err(|e| Error::InvalidFormat(format!("deflate compression finish failed: {e}")))?;
    Ok(())
}

/// HDF5 deflate filter entry point: reverse decodes, forward encodes, appending
/// the result to `out`.
pub fn filter_deflate_into(
    data: &[u8],
    level: u32,
    reverse: bool,
    out: &mut Vec<u8>,
) -> Result<()> {
    if reverse {
        decompress_into(data, out)
    } else {
        compress_into(data, level, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deflate_vec_decode_preserves_output_on_trailing_input() {
        let mut encoded = Vec::new();
        compress_into(b"payload", 6, &mut encoded).unwrap();
        encoded.extend_from_slice(b"trailing");

        let mut out = b"stale".to_vec();
        let err = decompress_into(&encoded, &mut out).unwrap_err();
        assert!(
            err.to_string()
                .contains("deflate decompression left trailing input bytes"),
            "unexpected error: {err}"
        );
        assert_eq!(out, b"stale");
    }

    #[test]
    fn deflate_exact_decode_preserves_output_on_size_error() {
        let mut encoded = Vec::new();
        compress_into(b"payload", 6, &mut encoded).unwrap();

        let mut out = *b"stale!";
        let err = decompress_exact_into(&encoded, &mut out).unwrap_err();
        assert!(
            err.to_string()
                .contains("deflate decompression produced more bytes than expected"),
            "unexpected error: {err}"
        );
        assert_eq!(&out, b"stale!");
    }
}
