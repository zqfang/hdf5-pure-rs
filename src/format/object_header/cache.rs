//! Object header cache deserializers — mirrors libhdf5's `H5Ocache.c`
//! `H5O__cache_deserialize`. Two entry points, one per header version,
//! that pull the prefix into memory and hand off to `msg.rs` for the
//! per-message decode loop and to `chunk.rs` for continuation chunks.

use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::format::checksum::checksum_metadata;
use crate::io::reader::HdfReader;

use super::chunk::read_v1_continuation;
use super::chunk::read_v2_continuation;
use super::msg::{read_v1_messages, read_v2_messages};
use super::{
    ObjectHeader, HDR_ATTR_CRT_ORDER_TRACKED, HDR_ATTR_STORE_PHASE_CHANGE, HDR_CHUNK0_SIZE_MASK,
    HDR_STORE_TIMES, HDR_V2_KNOWN_FLAGS, MSG_OBJ_REF_COUNT, OHDR_MAGIC,
};

impl ObjectHeader {
    /// Read a v1 object header. Mirrors `H5O__cache_deserialize` for
    /// version 1 prefixes (which are identified by version byte == 1
    /// rather than the v2 "OHDR" magic).
    pub(super) fn read_v1<R: Read + Seek>(reader: &mut HdfReader<R>) -> Result<Self> {
        let header_start = reader.position()?;
        let version = reader.read_u8()?;
        if version != 1 {
            return Err(Error::InvalidFormat(format!(
                "expected object header v1, got {version}"
            )));
        }

        reader.skip(1)?;

        let num_messages = reader.read_u16()?;
        let refcount = reader.read_u32()?;
        let chunk_data_size = u64::from(reader.read_u32()?);

        // Reserved/padding to 8-byte boundary (v1 header is 12 bytes after version,
        // total prefix = 1+1+2+4+4 = 12, need to align to 8: 12 is already aligned to 4,
        // but the v1 header macro says H5O_ALIGN_OLD(12) = align to 8 = 16, so 4 padding bytes)
        reader.skip(4)?;

        // Now read messages from chunk data
        let chunk_start = reader.position()?;
        let chunk_end = chunk_start
            .checked_add(chunk_data_size)
            .ok_or_else(|| Error::InvalidFormat("object header v1 chunk size overflow".into()))?;

        let mut messages = Vec::new();
        let mut continuations = Vec::new();
        let mut chunk_ranges = vec![(header_start, chunk_end)];

        read_v1_messages(
            reader,
            chunk_end,
            num_messages,
            &mut messages,
            &mut continuations,
            &mut chunk_ranges,
            0,
        )?;

        // Process continuation chunks
        for (cont_addr, cont_len) in continuations {
            read_v1_continuation(
                reader,
                cont_addr,
                cont_len,
                &mut messages,
                &mut chunk_ranges,
                1,
            )?;
        }

        // The v1 spec says the stored message count is an upper bound on the
        // actual count (including NIL and HEADER_CONTINUATION); we strip both
        // before pushing, so the kept count must be ≤ declared. A `messages`
        // count exceeding `num_messages` indicates a corrupted header.
        let num_messages = usize::from(num_messages);
        if messages.len() > num_messages {
            return Err(Error::InvalidFormat(format!(
                "object header v1 declared {num_messages} messages but decoded {} non-NIL/non-continuation messages",
                messages.len()
            )));
        }

        Ok(ObjectHeader {
            version: 1,
            flags: 0,
            refcount,
            atime: None,
            mtime: None,
            ctime: None,
            btime: None,
            max_compact_attrs: None,
            min_dense_attrs: None,
            messages,
        })
    }

    /// Read a v2 object header. Assumes the "OHDR" magic has already been
    /// read by the dispatcher in `mod.rs::read_at`.
    pub(super) fn read_v2<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        header_addr: u64,
    ) -> Result<Self> {
        let _ = OHDR_MAGIC; // referenced for documentation; actually consumed by caller
        let version = reader.read_u8()?;
        if version != 2 {
            return Err(Error::InvalidFormat(format!(
                "expected object header v2, got {version}"
            )));
        }

        let flags = reader.read_u8()?;
        if flags & !HDR_V2_KNOWN_FLAGS != 0 {
            return Err(Error::InvalidFormat(format!(
                "object header v2 flags contain reserved bits: {flags:#04x}"
            )));
        }
        // Optional timestamps
        let (atime, mtime, ctime, btime) = if flags & HDR_STORE_TIMES != 0 {
            (
                Some(reader.read_u32()?),
                Some(reader.read_u32()?),
                Some(reader.read_u32()?),
                Some(reader.read_u32()?),
            )
        } else {
            (None, None, None, None)
        };

        // Optional attribute phase change
        let (max_compact_attrs, min_dense_attrs) = if flags & HDR_ATTR_STORE_PHASE_CHANGE != 0 {
            let max_compact = reader.read_u16()?;
            let min_dense = reader.read_u16()?;
            if max_compact < min_dense {
                return Err(Error::InvalidFormat(
                    "object header attribute phase change max compact is less than min dense"
                        .into(),
                ));
            }
            (Some(max_compact), Some(min_dense))
        } else {
            (None, None)
        };

        // Chunk 0 data size (1, 2, 4, or 8 bytes based on flags)
        let chunk0_size_bytes = 1u8 << (flags & HDR_CHUNK0_SIZE_MASK);
        let chunk0_data_size = reader.read_uint(chunk0_size_bytes)?;

        // Now we know where chunk 0 data starts and where its checksum is
        let chunk0_data_start = reader.position()?;
        let chunk0_data_end = chunk0_data_start
            .checked_add(chunk0_data_size)
            .ok_or_else(|| Error::InvalidFormat("object header v2 chunk size overflow".into()))?;

        // Verify checksum: it covers from "OHDR" magic to just before the checksum
        let checksum_pos = chunk0_data_end;
        // Read the stored checksum
        reader.seek(checksum_pos)?;
        let stored_checksum = reader.read_u32()?;

        // Compute checksum over header_addr .. checksum_pos
        let check_len =
            usize::try_from(checksum_pos.checked_sub(header_addr).ok_or_else(|| {
                Error::InvalidFormat("object header checksum span underflow".into())
            })?)
            .map_err(|_| {
                Error::InvalidFormat("object header checksum span exceeds usize".into())
            })?;
        reader.seek(header_addr)?;
        let check_data = reader.read_bytes(check_len)?;
        let computed = checksum_metadata(&check_data);

        if stored_checksum != computed {
            return Err(Error::InvalidFormat(format!(
                "object header checksum mismatch: stored={stored_checksum:#010x}, computed={computed:#010x}"
            )));
        }

        // Now parse messages from chunk 0 data
        reader.seek(chunk0_data_start)?;

        let has_crt_order = flags & HDR_ATTR_CRT_ORDER_TRACKED != 0;
        let mut messages = Vec::new();
        let mut continuations = Vec::new();
        let chunk0_range_end = chunk0_data_end
            .checked_add(4)
            .ok_or_else(|| Error::InvalidFormat("object header v2 chunk range overflow".into()))?;
        let mut chunk_ranges = vec![(header_addr, chunk0_range_end)];

        read_v2_messages(
            reader,
            chunk0_data_end,
            has_crt_order,
            &mut messages,
            &mut continuations,
            &mut chunk_ranges,
            0,
        )?;

        // Process continuation chunks
        for (cont_addr, cont_len) in continuations {
            read_v2_continuation(
                reader,
                cont_addr,
                cont_len,
                has_crt_order,
                &mut messages,
                &mut chunk_ranges,
                1,
            )?;
        }

        let refcount = messages
            .iter()
            .find(|msg| msg.msg_type == MSG_OBJ_REF_COUNT)
            .and_then(|msg| msg.data.get(0..4))
            .and_then(|raw| raw.try_into().ok())
            .map(u32::from_le_bytes)
            .unwrap_or(1);

        Ok(ObjectHeader {
            version: 2,
            flags,
            refcount,
            atime,
            mtime,
            ctime,
            btime,
            max_compact_attrs,
            min_dense_attrs,
            messages,
        })
    }
}
