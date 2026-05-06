use std::collections::HashMap;
use std::fs;
use std::io::{BufReader, Read, Seek, SeekFrom, Write};

use crate::error::{Error, Result};
use crate::format::checksum::checksum_metadata;
use crate::format::superblock::Superblock;
use crate::hl::file::FileInner;
use crate::io::reader::HdfReader;

use super::MutableFile;

impl MutableFile {
    /// Recompute and rewrite the v2 OH checksum.
    pub(super) fn rewrite_oh_checksum(&mut self, oh_start: u64, check_len: usize) -> Result<()> {
        let mut guard = self.inner.lock();
        guard.reader.seek(oh_start)?;
        let oh_data = guard.reader.read_bytes(check_len)?;
        drop(guard);

        let checksum = checksum_metadata(&oh_data);

        let check_len_u64 = Self::usize_to_u64(check_len, "object-header checksum length")?;
        let checksum_pos = oh_start
            .checked_add(check_len_u64)
            .ok_or_else(|| Error::InvalidFormat("object-header checksum offset overflow".into()))?;
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

        let chunks_per_dim: Vec<u64> = data_dims
            .iter()
            .zip(chunk_dims)
            .map(|(&dim, &chunk)| {
                if chunk == 0 {
                    return Err(Error::InvalidFormat("zero chunk dimension".into()));
                }
                dim.checked_add(chunk - 1)
                    .ok_or_else(|| Error::InvalidFormat("chunk count overflow".into()))
                    .map(|extent| extent / chunk)
            })
            .collect::<Result<_>>()?;
        let mut index = 0usize;
        for dim in 0..data_dims.len() {
            let scaled = chunk_coords[dim] / chunk_dims[dim];
            if scaled >= chunks_per_dim[dim] {
                return Err(Error::Unsupported(
                    "fixed-array chunk index updates can replace existing chunks only".into(),
                ));
            }
            let count = usize::try_from(chunks_per_dim[dim])
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
        (1 + ((bits + 8) / 8)).min(8)
    }

    pub(super) fn write_uint_le(&mut self, value: u64, size: usize) -> Result<()> {
        let bytes = Self::encode_uint_le(value, size, "mutable metadata integer")?;
        self.write_handle.write_all(&bytes)?;
        Ok(())
    }

    pub(super) fn encode_uint_le(value: u64, size: usize, context: &str) -> Result<Vec<u8>> {
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
        Ok(value.to_le_bytes()[..size].to_vec())
    }

    pub(super) fn undefined_addr_bytes(size: usize, context: &str) -> Result<Vec<u8>> {
        if !(1..=8).contains(&size) {
            return Err(Error::InvalidFormat(format!(
                "{context} address width is invalid"
            )));
        }
        Ok(vec![0xff; size])
    }

    pub(super) fn usize_to_u8(value: usize, context: &str) -> Result<u8> {
        u8::try_from(value).map_err(|_| Error::InvalidFormat(format!("{context} exceeds u8")))
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

    pub(super) fn read_fresh_bytes(&self, offset: u64, len: usize) -> Result<Vec<u8>> {
        let mut file = fs::File::open(&self.path)?;
        file.seek(SeekFrom::Start(offset))?;
        let mut bytes = vec![0u8; len];
        file.read_exact(&mut bytes)?;
        Ok(bytes)
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
            self.write_handle.write_all(&vec![0u8; padding])?;
            let padding_u64 = Self::usize_to_u64(padding, "alignment padding")?;
            pos = pos
                .checked_add(padding_u64)
                .ok_or_else(|| Error::InvalidFormat("aligned append offset overflow".into()))?;
        }
        self.write_handle.write_all(&vec![0u8; size])?;
        Ok(pos)
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
        assert!(MutableFile::usize_to_u8(usize::from(u8::MAX) + 1, "rank").is_err());
        assert!(MutableFile::usize_to_u16(usize::from(u16::MAX) + 1, "record count").is_err());
        if let Ok(value) = usize::try_from(u64::from(u32::MAX) + 1) {
            assert!(MutableFile::usize_to_u32(value, "element size").is_err());
        }
        assert!(MutableFile::u64_to_u32(u64::from(u32::MAX) + 1, "chunk size").is_err());
        assert!(MutableFile::encode_uint_le(256, 1, "test integer").is_err());
        assert!(MutableFile::encode_uint_le(0, 9, "test integer").is_err());
        assert_eq!(
            MutableFile::undefined_addr_bytes(2, "test address").unwrap(),
            vec![0xff, 0xff]
        );
    }
}
