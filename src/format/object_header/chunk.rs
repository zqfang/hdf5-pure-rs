//! Object header continuation chunks — mirrors libhdf5's `H5Ochunk.c`.
//! `read_v1_continuation` / `read_v2_continuation` follow continuation
//! pointers and load successor chunks; `reserve_continuation_range`
//! validates that the continuation range is in-bounds and non-overlapping
//! with already-tracked chunks.

use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::format::checksum::checksum_metadata;
use crate::io::reader::HdfReader;

use super::msg::{read_v1_messages, read_v2_messages};
use super::{RawMessage, OCHK_MAGIC};

pub(super) fn read_v1_continuation<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    addr: u64,
    length: u64,
    messages: &mut Vec<RawMessage>,
    chunk_ranges: &mut Vec<(u64, u64)>,
    chunk_index: u16,
) -> Result<()> {
    reader.seek(addr)?;

    // V1 continuation chunks are just raw messages, no header.
    let chunk_end = addr
        .checked_add(length)
        .ok_or_else(|| Error::InvalidFormat("object header continuation range overflow".into()))?;
    let mut continuations = Vec::new();
    read_v1_messages(
        reader,
        chunk_end,
        0,
        messages,
        &mut continuations,
        chunk_ranges,
        chunk_index,
    )?;

    for (cont_addr, cont_len) in continuations {
        read_v1_continuation(
            reader,
            cont_addr,
            cont_len,
            messages,
            chunk_ranges,
            next_chunk_index(chunk_index)?,
        )?;
    }

    Ok(())
}

pub(super) fn read_v2_continuation<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    addr: u64,
    length: u64,
    has_crt_order: bool,
    messages: &mut Vec<RawMessage>,
    chunk_ranges: &mut Vec<(u64, u64)>,
    chunk_index: u16,
) -> Result<()> {
    reader.seek(addr)?;

    // V2 continuation chunks start with "OCHK" magic
    let magic = reader.read_bytes(4)?;
    if magic != OCHK_MAGIC {
        return Err(Error::InvalidFormat(
            "invalid continuation chunk magic".into(),
        ));
    }

    // Data runs from after magic to before checksum
    let _data_start = reader.position()?;
    let data_end = addr
        .checked_add(length)
        .and_then(|end| end.checked_sub(4))
        .ok_or_else(|| Error::InvalidFormat("object header continuation range overflow".into()))?; // minus checksum

    let mut continuations = Vec::new();
    read_v2_messages(
        reader,
        data_end,
        has_crt_order,
        messages,
        &mut continuations,
        chunk_ranges,
        chunk_index,
    )?;

    // Verify checksum
    reader.seek(data_end)?;
    let stored_checksum = reader.read_u32()?;
    let check_len = usize::try_from(data_end - addr).map_err(|_| {
        Error::InvalidFormat("continuation chunk checksum span exceeds usize".into())
    })?;
    reader.seek(addr)?;
    let check_data = reader.read_bytes(check_len)?;
    let computed = checksum_metadata(&check_data);

    if stored_checksum != computed {
        return Err(Error::InvalidFormat(
            "continuation chunk checksum mismatch".into(),
        ));
    }

    // Process nested continuations
    for (cont_addr, cont_len) in continuations {
        read_v2_continuation(
            reader,
            cont_addr,
            cont_len,
            has_crt_order,
            messages,
            chunk_ranges,
            next_chunk_index(chunk_index)?,
        )?;
    }

    Ok(())
}

pub(super) fn reserve_continuation_range<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    addr: u64,
    length: u64,
    min_length: u64,
    chunk_ranges: &mut Vec<(u64, u64)>,
) -> Result<()> {
    if length < min_length {
        return Err(Error::InvalidFormat(
            "object header continuation chunk is too small".into(),
        ));
    }
    let end = addr
        .checked_add(length)
        .ok_or_else(|| Error::InvalidFormat("object header continuation range overflow".into()))?;
    let file_len = reader.len()?;
    if end > file_len {
        return Err(Error::InvalidFormat(
            "object header continuation range exceeds file size".into(),
        ));
    }
    if chunk_ranges
        .iter()
        .any(|&(range_start, range_end)| addr < range_end && range_start < end)
    {
        return Err(Error::InvalidFormat(
            "object header continuation range overlaps another metadata chunk".into(),
        ));
    }
    chunk_ranges.push((addr, end));
    Ok(())
}

fn next_chunk_index(chunk_index: u16) -> Result<u16> {
    chunk_index
        .checked_add(1)
        .ok_or_else(|| Error::InvalidFormat("object header continuation depth overflow".into()))
}

#[cfg(test)]
mod tests {
    use super::next_chunk_index;

    #[test]
    fn continuation_chunk_index_rejects_overflow() {
        let err = next_chunk_index(u16::MAX).unwrap_err();
        assert!(err.to_string().contains("overflow"));
    }
}
