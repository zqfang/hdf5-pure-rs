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
    if let Some(expected_len) = expected_len {
        out.reserve(expected_len);
    }
    let mut decoder = ZlibDecoder::new(data);
    decoder
        .read_to_end(out)
        .map_err(|e| Error::InvalidFormat(format!("deflate decompression failed: {e}")))?;
    Ok(())
}

/// Decompress deflate (zlib) compressed data into the provided output buffer
/// and require the decoded size to match exactly.
pub fn decompress_exact_into(data: &[u8], out: &mut [u8]) -> Result<()> {
    let mut decoder = Decompress::new(true);
    let status = decoder
        .decompress(data, out, FlushDecompress::Finish)
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
