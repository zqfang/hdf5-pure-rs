use std::io::{Read, Seek, SeekFrom, Write};

use crate::error::{Error, Result};
use crate::format::checksum::checksum_metadata;

use super::MutableFile;

impl MutableFile {
    pub(super) fn rewrite_fixed_array_chunk(
        &mut self,
        index_addr: u64,
        info: &crate::hl::dataset::DatasetInfo,
        chunk_coords: &[u64],
        chunk_dims: &[u64],
        chunk_size: u64,
        chunk_addr: u64,
        unfiltered_chunk_bytes: usize,
    ) -> Result<()> {
        let element_index =
            Self::linear_chunk_index(chunk_coords, &info.dataspace.dims, chunk_dims)?;
        let filtered = info
            .filter_pipeline
            .as_ref()
            .map(|pipeline| !pipeline.filters.is_empty())
            .unwrap_or(false);
        let chunk_size_len = if filtered {
            Self::filtered_chunk_size_len(
                info.layout.version,
                unfiltered_chunk_bytes,
                self.superblock.sizeof_size,
            )
        } else {
            0
        };

        let mut guard = self.inner.lock();
        let element_location =
            crate::format::fixed_array::locate_fixed_array_element_with_checksum(
                &mut guard.reader,
                index_addr,
                filtered,
                chunk_size_len,
                element_index,
            )?;
        drop(guard);

        let sa = usize::from(self.superblock.sizeof_addr);
        self.write_handle
            .seek(SeekFrom::Start(element_location.element_addr))?;
        let chunk_addr = Self::encode_uint_le(chunk_addr, sa, "fixed array chunk address")?;
        self.write_handle.write_all(&chunk_addr)?;
        if filtered {
            self.write_uint_le(chunk_size, chunk_size_len)?;
            self.write_handle.write_all(&0u32.to_le_bytes())?;
        }
        self.rewrite_fixed_array_element_checksum(
            element_location.checksum_start,
            element_location.checksum_len,
            element_location.checksum_addr,
        )?;
        Ok(())
    }

    fn rewrite_fixed_array_element_checksum(
        &mut self,
        checksum_start: u64,
        checksum_len: usize,
        checksum_addr: u64,
    ) -> Result<()> {
        self.write_handle.flush()?;
        let mut reader = std::fs::File::open(&self.path)?;
        reader.seek(SeekFrom::Start(checksum_start))?;
        let mut bytes = vec![0; checksum_len];
        reader.read_exact(&mut bytes)?;
        let checksum = checksum_metadata(&bytes);
        let checksum_end = checksum_addr
            .checked_add(4)
            .ok_or_else(|| Error::InvalidFormat("fixed array checksum address overflow".into()))?;
        let file_len = reader.metadata()?.len();
        if checksum_end > file_len {
            return Err(Error::InvalidFormat(
                "fixed array checksum address exceeds file size".into(),
            ));
        }
        self.write_handle.seek(SeekFrom::Start(checksum_addr))?;
        self.write_handle.write_all(&checksum.to_le_bytes())?;
        Ok(())
    }
}
