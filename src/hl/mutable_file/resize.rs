use std::io::{Seek, SeekFrom, Write};

use crate::error::{Error, Result};
use crate::format::messages::data_layout::LayoutClass;
use crate::format::object_header::{
    self, HDR_ATTR_STORE_PHASE_CHANGE, HDR_CHUNK0_SIZE_MASK, HDR_STORE_TIMES, HDR_V2_KNOWN_FLAGS,
};

use super::MutableFile;

impl MutableFile {
    /// Resize a chunked dataset to new dimensions.
    ///
    /// The new dimensions must not exceed the dataset's maximum dimensions.
    /// Only the dataspace message is rewritten; no data is moved or deleted.
    /// New regions are filled with the default fill value (zeros).
    pub fn resize_dataset(&mut self, path: &str, new_dims: &[u64]) -> Result<()> {
        let ds = self.dataset(path)?;
        let info = ds.info()?;

        if info.layout.layout_class != LayoutClass::Chunked {
            return Err(Error::InvalidFormat(
                "can only resize chunked datasets".into(),
            ));
        }

        if new_dims.len() != info.dataspace.dims.len() {
            return Err(Error::InvalidFormat(format!(
                "dimension count mismatch: dataset has {} dims, new shape has {}",
                info.dataspace.dims.len(),
                new_dims.len()
            )));
        }

        if let Some(ref max_dims) = info.dataspace.max_dims {
            for (i, (&new_d, &max_d)) in new_dims.iter().zip(max_dims.iter()).enumerate() {
                if max_d != u64::MAX && new_d > max_d {
                    return Err(Error::InvalidFormat(format!(
                        "dimension {i}: new size {new_d} exceeds max {max_d}"
                    )));
                }
            }
        }

        let ds_addr = ds.addr();
        let (msg_data_offset, msg_data_len, oh_start, oh_check_len) =
            self.find_message_in_oh(ds_addr, object_header::MSG_DATASPACE)?;

        let mut new_ds_bytes = Vec::with_capacity(msg_data_len);
        Self::encode_dataspace_v2_into(
            new_dims,
            info.dataspace.max_dims.as_deref(),
            &mut new_ds_bytes,
        )?;

        if new_ds_bytes.len() != msg_data_len {
            return Err(Error::InvalidFormat(format!(
                "dataspace message size changed ({} -> {}); in-place resize not possible",
                msg_data_len,
                new_ds_bytes.len()
            )));
        }

        self.write_handle.seek(SeekFrom::Start(msg_data_offset))?;
        self.write_handle.write_all(&new_ds_bytes)?;
        self.rewrite_oh_checksum(oh_start, oh_check_len)?;
        self.write_handle.flush()?;
        self.reopen_reader()?;

        Ok(())
    }

    /// Find a message of the given type in an object header.
    /// Returns (message_data_file_offset, message_data_len, oh_start, oh_checksum_data_len).
    fn find_message_in_oh(
        &self,
        oh_addr: u64,
        target_msg_type: u16,
    ) -> Result<(u64, usize, u64, usize)> {
        let mut guard = self.inner.lock();
        let reader = &mut guard.reader;
        reader.seek(oh_addr)?;

        let mut first_bytes = [0u8; 4];
        reader.read_bytes_into(&mut first_bytes)?;
        if first_bytes != [b'O', b'H', b'D', b'R'] {
            return Err(Error::InvalidFormat(
                "expected v2 object header for resize".into(),
            ));
        }

        let version = reader.read_u8()?;
        if version != 2 {
            return Err(Error::Unsupported(
                "resize only supported for v2 object headers".into(),
            ));
        }

        let flags = reader.read_u8()?;
        if flags & !HDR_V2_KNOWN_FLAGS != 0 {
            return Err(Error::InvalidFormat(format!(
                "object header v2 flags contain reserved bits: {flags:#04x}"
            )));
        }
        if flags & HDR_STORE_TIMES != 0 {
            reader.skip(16)?;
        }
        if flags & HDR_ATTR_STORE_PHASE_CHANGE != 0 {
            reader.skip(4)?;
        }

        let chunk0_size_bytes = 1u8 << (flags & HDR_CHUNK0_SIZE_MASK);
        let chunk0_data_size = reader.read_uint(chunk0_size_bytes)?;
        let chunk0_data_start = reader.position()?;
        let chunk0_data_end = chunk0_data_start
            .checked_add(chunk0_data_size)
            .ok_or_else(|| Error::InvalidFormat("object-header chunk range overflow".into()))?;
        let oh_check_len = usize::try_from(chunk0_data_end - oh_addr)
            .map_err(|_| Error::InvalidFormat("object-header checksum range overflow".into()))?;

        while reader.position()? < chunk0_data_end {
            let msg_header_pos = reader.position()?;
            if msg_header_pos
                .checked_add(4)
                .is_none_or(|end| end > chunk0_data_end)
            {
                break;
            }

            let msg_type = u16::from(reader.read_u8()?);
            let msg_size = usize::from(reader.read_u16()?);
            let _msg_flags = reader.read_u8()?;

            if flags & object_header::HDR_ATTR_CRT_ORDER_TRACKED != 0 {
                reader.skip(2)?;
            }

            let msg_data_offset = reader.position()?;
            let msg_size_u64 = Self::usize_to_u64(msg_size, "object-header message size")?;
            if msg_data_offset
                .checked_add(msg_size_u64)
                .is_none_or(|end| end > chunk0_data_end)
            {
                return Err(Error::InvalidFormat(
                    "object-header message payload exceeds chunk".into(),
                ));
            }

            if msg_type == target_msg_type {
                return Ok((msg_data_offset, msg_size, oh_addr, oh_check_len));
            }

            reader.skip(msg_size_u64)?;
        }

        Err(Error::InvalidFormat(format!(
            "message type {target_msg_type:#06x} not found in object header"
        )))
    }

    /// Encode a v2 dataspace message with the given dims and optional max_dims.
    fn encode_dataspace_v2_into(
        dims: &[u64],
        max_dims: Option<&[u64]>,
        buf: &mut Vec<u8>,
    ) -> Result<()> {
        if let Some(max) = max_dims {
            if max.len() != dims.len() {
                return Err(Error::InvalidFormat(format!(
                    "dataspace max rank {} does not match rank {}",
                    max.len(),
                    dims.len()
                )));
            }
        }
        let has_max = max_dims.is_some();
        let rank = Self::usize_to_u8(dims.len(), "dataspace rank")?;
        let payload_dims = dims
            .len()
            .checked_mul(if has_max { 2 } else { 1 })
            .ok_or_else(|| Error::InvalidFormat("dataspace message size overflow".into()))?;
        let capacity =
            4usize
                .checked_add(payload_dims.checked_mul(8).ok_or_else(|| {
                    Error::InvalidFormat("dataspace message size overflow".into())
                })?)
                .ok_or_else(|| Error::InvalidFormat("dataspace message size overflow".into()))?;

        buf.clear();
        if buf.capacity() < capacity {
            buf.reserve_exact(capacity - buf.capacity());
        }
        buf.push(2);
        buf.push(rank);
        buf.push(if has_max { 0x01 } else { 0x00 });
        buf.push(1);

        for &d in dims {
            buf.extend_from_slice(&d.to_le_bytes());
        }

        if let Some(max) = max_dims {
            for &d in max {
                buf.extend_from_slice(&d.to_le_bytes());
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_dataspace_encoder_rejects_rank_narrowing_and_mismatch() {
        let dims = vec![1u64; usize::from(u8::MAX) + 1];
        let mut buf = Vec::new();
        let err = MutableFile::encode_dataspace_v2_into(&dims, None, &mut buf).unwrap_err();
        assert!(
            err.to_string().contains("dataspace rank"),
            "unexpected error: {err}"
        );

        let err =
            MutableFile::encode_dataspace_v2_into(&[1, 2], Some(&[10]), &mut buf).unwrap_err();
        assert!(
            err.to_string().contains("dataspace max rank"),
            "unexpected error: {err}"
        );
    }
}
