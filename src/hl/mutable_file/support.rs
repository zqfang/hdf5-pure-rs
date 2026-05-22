use std::collections::HashMap;
use std::fs;
use std::io::{BufReader, Read, Seek, SeekFrom, Write};

use crate::error::{Error, Result};
use crate::format::checksum::checksum_metadata;
use crate::format::superblock::Superblock;
use crate::hl::file::{FileInner, FileIntent};
use crate::io::reader::HdfReader;

use super::MutableFile;

impl MutableFile {
    pub(super) fn physical_addr(&self, logical_addr: u64, context: &str) -> Result<u64> {
        self.superblock
            .base_addr
            .checked_add(logical_addr)
            .ok_or_else(|| Error::InvalidFormat(format!("{context} address overflow")))
    }

    pub(super) fn logical_addr_from_physical(
        &self,
        physical_addr: u64,
        context: &str,
    ) -> Result<u64> {
        physical_addr
            .checked_sub(self.superblock.base_addr)
            .ok_or_else(|| Error::InvalidFormat(format!("{context} is before HDF5 base address")))
    }

    /// Recompute and rewrite the v2 OH checksum.
    pub(super) fn rewrite_oh_checksum(&mut self, oh_start: u64, check_len: usize) -> Result<()> {
        let mut guard = self.inner.lock();
        guard.reader.seek(oh_start)?;
        let mut oh_data = vec![0u8; check_len];
        guard.reader.read_bytes_into(&mut oh_data)?;
        drop(guard);

        let checksum = checksum_metadata(&oh_data);

        let check_len_u64 = Self::usize_to_u64(check_len, "object-header checksum length")?;
        let checksum_pos = oh_start
            .checked_add(check_len_u64)
            .ok_or_else(|| Error::InvalidFormat("object-header checksum offset overflow".into()))?;
        let checksum_pos = self.physical_addr(checksum_pos, "object-header checksum")?;
        self.write_handle.seek(SeekFrom::Start(checksum_pos))?;
        self.write_handle.write_all(&checksum.to_le_bytes())?;

        Ok(())
    }

    /// Reopen the read handle to pick up file changes.
    pub(super) fn reopen_reader(&mut self) -> Result<()> {
        let read_file = fs::File::open(&self.path)?;
        let mut reader = HdfReader::new(BufReader::new(read_file));
        let superblock = Superblock::read(&mut reader)?;
        self.superblock = superblock.clone();
        *self.inner.lock() = FileInner {
            reader,
            superblock,
            path: Some(self.path.clone()),
            intent: FileIntent::ReadWrite,
            access_plist: crate::hl::plist::file_access::FileAccess::default(),
            dset_no_attrs_hint: false,
            open_objects: HashMap::new(),
            next_object_id: 1,
        };
        Ok(())
    }

    pub(super) fn linear_chunk_index(
        chunk_coords: &[u64],
        data_dims: &[u64],
        chunk_dims: &[u64],
    ) -> Result<usize> {
        if chunk_coords.len() != data_dims.len() || chunk_dims.len() != data_dims.len() {
            return Err(Error::InvalidFormat(
                "chunk coordinate rank does not match dataset rank".into(),
            ));
        }

        let mut index = 0usize;
        for ((&coord, &dim), &chunk) in chunk_coords.iter().zip(data_dims).zip(chunk_dims) {
            if chunk == 0 {
                return Err(Error::InvalidFormat("zero chunk dimension".into()));
            }
            let chunks_in_dim = dim
                .checked_add(chunk - 1)
                .ok_or_else(|| Error::InvalidFormat("chunk count overflow".into()))?
                / chunk;
            let scaled = coord / chunk;
            if scaled >= chunks_in_dim {
                return Err(Error::Unsupported(
                    "fixed-array chunk index updates can replace existing chunks only".into(),
                ));
            }
            let count = usize::try_from(chunks_in_dim)
                .map_err(|_| Error::InvalidFormat("chunks per dimension overflow".into()))?;
            let scaled = usize::try_from(scaled)
                .map_err(|_| Error::InvalidFormat("chunk coordinate overflow".into()))?;
            index = index
                .checked_mul(count)
                .and_then(|value| value.checked_add(scaled))
                .ok_or_else(|| Error::InvalidFormat("chunk index overflow".into()))?;
        }
        Ok(index)
    }

    pub(super) fn filtered_chunk_size_len(
        layout_version: u8,
        unfiltered_chunk_bytes: usize,
        sizeof_size: u8,
    ) -> usize {
        if layout_version > 4 {
            return usize::from(sizeof_size);
        }
        let bits = if unfiltered_chunk_bytes == 0 {
            0
        } else {
            usize::try_from(usize::BITS - unfiltered_chunk_bytes.leading_zeros())
                .unwrap_or(usize::MAX)
        };
        ((bits + 15) / 8).clamp(2, 8)
    }

    pub(super) fn write_uint_le(&mut self, value: u64, size: usize) -> Result<()> {
        let mut bytes = [0u8; 8];
        let bytes = Self::encode_uint_le_into(value, &mut bytes, size, "mutable metadata integer")?;
        self.write_handle.write_all(bytes)?;
        Ok(())
    }

    pub(super) fn encode_uint_le_into<'a>(
        value: u64,
        out: &'a mut [u8],
        size: usize,
        context: &str,
    ) -> Result<&'a [u8]> {
        Self::validate_uint_le(value, size, context)?;
        if out.len() < size {
            return Err(Error::InvalidFormat(format!(
                "{context} output buffer is too small"
            )));
        }
        out[..size].copy_from_slice(&value.to_le_bytes()[..size]);
        Ok(&out[..size])
    }

    fn validate_uint_le(value: u64, size: usize, context: &str) -> Result<()> {
        if !(1..=8).contains(&size) {
            return Err(Error::InvalidFormat(format!(
                "{context} integer width is invalid"
            )));
        }
        if size < 8 {
            let bits = size
                .checked_mul(8)
                .ok_or_else(|| Error::InvalidFormat(format!("{context} integer width overflow")))?;
            if value >= (1u64 << bits) {
                return Err(Error::InvalidFormat(format!(
                    "{context} value does not fit in {size} bytes"
                )));
            }
        }
        Ok(())
    }

    pub(super) fn undefined_addr_bytes_into<'a>(
        out: &'a mut [u8],
        size: usize,
        context: &str,
    ) -> Result<&'a [u8]> {
        Self::validate_addr_size(size, context)?;
        if out.len() < size {
            return Err(Error::InvalidFormat(format!(
                "{context} output buffer is too small"
            )));
        }
        out[..size].fill(0xff);
        Ok(&out[..size])
    }

    fn validate_addr_size(size: usize, context: &str) -> Result<()> {
        if !(1..=8).contains(&size) {
            return Err(Error::InvalidFormat(format!(
                "{context} address width is invalid"
            )));
        }
        Ok(())
    }

    pub(super) fn usize_to_u16(value: usize, context: &str) -> Result<u16> {
        u16::try_from(value).map_err(|_| Error::InvalidFormat(format!("{context} exceeds u16")))
    }

    pub(super) fn usize_to_u32(value: usize, context: &str) -> Result<u32> {
        u32::try_from(value).map_err(|_| Error::InvalidFormat(format!("{context} exceeds u32")))
    }

    pub(super) fn usize_to_u64(value: usize, context: &str) -> Result<u64> {
        u64::try_from(value).map_err(|_| Error::InvalidFormat(format!("{context} exceeds u64")))
    }

    pub(super) fn u64_to_usize(value: u64, context: &str) -> Result<usize> {
        usize::try_from(value).map_err(|_| Error::InvalidFormat(format!("{context} exceeds usize")))
    }

    pub(super) fn u64_to_u32(value: u64, context: &str) -> Result<u32> {
        u32::try_from(value).map_err(|_| Error::InvalidFormat(format!("{context} exceeds u32")))
    }

    pub(super) fn read_fresh_bytes_into(&self, offset: u64, out: &mut [u8]) -> Result<()> {
        let mut file = fs::File::open(&self.path)?;
        file.seek(SeekFrom::Start(offset))?;
        file.read_exact(out)?;
        Ok(())
    }

    fn write_zeros(&mut self, mut size: usize) -> Result<()> {
        const ZERO_BLOCK: [u8; 8192] = [0; 8192];
        while size > 0 {
            let chunk = size.min(ZERO_BLOCK.len());
            self.write_handle.write_all(&ZERO_BLOCK[..chunk])?;
            size -= chunk;
        }
        Ok(())
    }

    pub(super) fn append_aligned_zeros(&mut self, size: usize, align: u64) -> Result<u64> {
        if align == 0 {
            return Err(Error::InvalidFormat("alignment cannot be zero".into()));
        }
        let mut pos = self.write_handle.seek(SeekFrom::End(0))?;
        let padding = (align - (pos % align)) % align;
        if padding != 0 {
            let padding = usize::try_from(padding)
                .map_err(|_| Error::InvalidFormat("alignment padding overflow".into()))?;
            self.write_zeros(padding)?;
            let padding_u64 = Self::usize_to_u64(padding, "alignment padding")?;
            pos = pos
                .checked_add(padding_u64)
                .ok_or_else(|| Error::InvalidFormat("aligned append offset overflow".into()))?;
        }
        self.write_zeros(size)?;
        Ok(pos)
    }

    pub(super) fn rewrite_superblock_eof(&mut self, eof_addr: u64) -> Result<()> {
        if self.superblock.version >= 2 {
            let sb = crate::format::superblock::Superblock {
                eof_addr,
                ..self.superblock.clone()
            };
            let mut sb_bytes = Vec::new();
            sb.write_v2(&mut sb_bytes)?;
            self.write_handle
                .seek(SeekFrom::Start(self.superblock.base_addr))?;
            self.write_handle.write_all(&sb_bytes)?;
            self.superblock = sb;
            return Ok(());
        }

        let mut addr_start = 8usize
            .checked_add(1)
            .and_then(|value| value.checked_add(4))
            .and_then(|value| value.checked_add(2))
            .and_then(|value| value.checked_add(1))
            .and_then(|value| value.checked_add(4))
            .and_then(|value| value.checked_add(4))
            .ok_or_else(|| Error::InvalidFormat("superblock EOF field offset overflow".into()))?;
        if self.superblock.version > 0 {
            addr_start = addr_start
                .checked_add(2)
                .and_then(|value| {
                    if self.superblock.version == 1 {
                        value.checked_add(2)
                    } else {
                        Some(value)
                    }
                })
                .ok_or_else(|| {
                    Error::InvalidFormat("superblock EOF field offset overflow".into())
                })?;
        }
        let eof_offset = addr_start
            .checked_add(
                2usize
                    .checked_mul(usize::from(self.superblock.sizeof_addr))
                    .ok_or_else(|| {
                        Error::InvalidFormat("superblock EOF field offset overflow".into())
                    })?,
            )
            .ok_or_else(|| Error::InvalidFormat("superblock EOF field offset overflow".into()))?;
        let mut bytes = [0u8; 8];
        let bytes = Self::encode_uint_le_into(
            eof_addr,
            &mut bytes,
            usize::from(self.superblock.sizeof_addr),
            "superblock EOF address",
        )?;
        let eof_pos = self
            .superblock
            .base_addr
            .checked_add(Self::usize_to_u64(
                eof_offset,
                "superblock EOF field offset",
            )?)
            .ok_or_else(|| Error::InvalidFormat("superblock EOF field position overflow".into()))?;
        self.write_handle.seek(SeekFrom::Start(eof_pos))?;
        self.write_handle.write_all(bytes)?;
        self.superblock.eof_addr = eof_addr;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_chunk_index_rejects_zero_chunk_dimension() {
        let err = MutableFile::linear_chunk_index(&[0], &[10], &[0]).unwrap_err();
        assert!(
            err.to_string().contains("zero chunk dimension"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn linear_chunk_index_rejects_chunk_count_overflow() {
        let err = MutableFile::linear_chunk_index(&[0], &[u64::MAX], &[2]).unwrap_err();
        assert!(
            err.to_string().contains("chunk count overflow"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn checksum_offset_addition_rejects_overflow() {
        let max_usize_as_u64 =
            u64::try_from(usize::MAX).expect("usize::MAX should fit in u64 on supported targets");
        let err = u64::MAX
            .checked_add(max_usize_as_u64)
            .ok_or_else(|| Error::InvalidFormat("object-header checksum offset overflow".into()))
            .unwrap_err();
        assert!(err.to_string().contains("checksum offset overflow"));
    }

    #[test]
    fn append_aligned_zeros_rejects_zero_alignment() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("zero_alignment.h5");
        {
            let mut file = crate::WritableFile::create(&path).unwrap();
            file.flush().unwrap();
        }

        let mut file = MutableFile::open_rw(&path).unwrap();
        let err = file.append_aligned_zeros(1, 0).unwrap_err();
        assert!(
            err.to_string().contains("alignment cannot be zero"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn mutable_file_integer_helpers_reject_narrowing() {
        assert!(MutableFile::usize_to_u16(usize::from(u16::MAX) + 1, "record count").is_err());
        if let Ok(value) = usize::try_from(u64::from(u32::MAX) + 1) {
            assert!(MutableFile::usize_to_u32(value, "element size").is_err());
        }
        assert!(MutableFile::u64_to_u32(u64::from(u32::MAX) + 1, "chunk size").is_err());
        let mut out = [0u8; 8];
        assert!(MutableFile::encode_uint_le_into(256, &mut out, 1, "test integer").is_err());
        assert!(MutableFile::encode_uint_le_into(0, &mut out, 9, "test integer").is_err());
        assert_eq!(
            MutableFile::encode_uint_le_into(0x1234, &mut out, 2, "test integer").unwrap(),
            &[0x34, 0x12]
        );
        assert!(MutableFile::encode_uint_le_into(1, &mut out[..0], 1, "test integer").is_err());
        assert_eq!(
            MutableFile::undefined_addr_bytes_into(&mut out, 2, "test address").unwrap(),
            &[0xff, 0xff]
        );
        assert!(MutableFile::undefined_addr_bytes_into(&mut out[..1], 2, "test address").is_err());
    }
}
