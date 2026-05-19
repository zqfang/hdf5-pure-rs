//! Fractal heap direct blocks — mirrors libhdf5's `H5HFdblock.c` plus
//! the dblock half of `H5HFcache.c` (`H5HF__cache_dblock_deserialize`).
//! In the Rust port, direct-block reads are pull-style (no separate
//! cache layer), so this file just holds `read_from_direct_block`.

use std::{
    collections::hash_map::Entry,
    io::{Read, Seek},
    ops::Range,
};

use crate::error::{Error, Result};
use crate::io::reader::HdfReader;

use super::{
    heap_object_len, verify_direct_block_checksum, DirectBlockCacheKey, FractalHeapHeader,
    FractalHeapManagedObjectCache,
};

impl FractalHeapHeader {
    pub(super) fn read_from_direct_block<R: Read + Seek>(
        &self,
        reader: &mut HdfReader<R>,
        block_addr: u64,
        block_size: u64,
        filtered_size: Option<u64>,
        filter_mask: u32,
        offset: u64,
        length: u64,
    ) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        self.read_from_direct_block_into(
            reader,
            block_addr,
            block_size,
            filtered_size,
            filter_mask,
            offset,
            length,
            &mut out,
        )?;
        Ok(out)
    }

    pub(super) fn read_from_direct_block_into<R: Read + Seek>(
        &self,
        reader: &mut HdfReader<R>,
        block_addr: u64,
        block_size: u64,
        filtered_size: Option<u64>,
        filter_mask: u32,
        offset: u64,
        length: u64,
        out: &mut Vec<u8>,
    ) -> Result<()> {
        let range = direct_block_object_range(
            block_size,
            offset,
            length,
            if filtered_size.is_some() {
                "fractal heap object exceeds filtered direct block"
            } else {
                "fractal heap object exceeds direct block"
            },
        )?;

        if let Some(filtered_size) = filtered_size {
            reader.seek(block_addr)?;
            let mut filtered =
                vec![0; heap_object_len(filtered_size, "filtered fractal heap block size",)?];
            reader.read_bytes_into(&mut filtered)?;
            let pipeline = self.filter_pipeline.as_ref().ok_or_else(|| {
                Error::InvalidFormat("filtered fractal heap missing filter pipeline".into())
            })?;
            let object_len = range.len();
            out.clear();
            crate::filters::apply_pipeline_reverse_with_mask_into(
                &filtered,
                pipeline,
                1,
                filter_mask,
                out,
            )?;
            if range.start == 0 && range.end == out.len() {
                self.trace_managed_object(
                    block_addr,
                    block_size,
                    offset,
                    length,
                    filter_mask,
                    true,
                );
                return Ok(());
            }
            let end = range.end;
            if end > out.len() {
                return Err(Error::InvalidFormat(
                    "fractal heap object exceeds filtered direct block".into(),
                ));
            }
            out.copy_within(range, 0);
            out.truncate(object_len);
            self.trace_managed_object(block_addr, block_size, offset, length, filter_mask, true);
            return Ok(());
        }

        if self.has_checksum && self.heap_addr == 0 {
            verify_direct_block_checksum(reader, block_addr, self.max_heap_size, block_size)?;
        }
        let addr = block_addr
            .checked_add(offset)
            .ok_or_else(|| Error::InvalidFormat("fractal heap object address overflow".into()))?;
        reader.seek(addr)?;
        out.clear();
        out.resize(range.len(), 0);
        reader.read_bytes_into(out)?;
        self.trace_managed_object(block_addr, block_size, offset, length, 0, false);
        Ok(())
    }

    pub(super) fn read_from_direct_block_cached<R: Read + Seek>(
        &self,
        reader: &mut HdfReader<R>,
        block_addr: u64,
        block_size: u64,
        filtered_size: Option<u64>,
        filter_mask: u32,
        offset: u64,
        length: u64,
        cache: &mut FractalHeapManagedObjectCache,
    ) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        self.read_from_direct_block_cached_into(
            reader,
            block_addr,
            block_size,
            filtered_size,
            filter_mask,
            offset,
            length,
            cache,
            &mut out,
        )?;
        Ok(out)
    }

    pub(super) fn read_from_direct_block_cached_into<R: Read + Seek>(
        &self,
        reader: &mut HdfReader<R>,
        block_addr: u64,
        block_size: u64,
        filtered_size: Option<u64>,
        filter_mask: u32,
        offset: u64,
        length: u64,
        cache: &mut FractalHeapManagedObjectCache,
        out: &mut Vec<u8>,
    ) -> Result<()> {
        let slice = self.read_from_direct_block_cached_slice(
            reader,
            block_addr,
            block_size,
            filtered_size,
            filter_mask,
            offset,
            length,
            cache,
        )?;
        out.clear();
        out.extend_from_slice(slice);
        Ok(())
    }

    pub(super) fn read_from_direct_block_cached_slice<'cache, R: Read + Seek>(
        &self,
        reader: &mut HdfReader<R>,
        block_addr: u64,
        block_size: u64,
        filtered_size: Option<u64>,
        filter_mask: u32,
        offset: u64,
        length: u64,
        cache: &'cache mut FractalHeapManagedObjectCache,
    ) -> Result<&'cache [u8]> {
        let range = direct_block_object_range(
            block_size,
            offset,
            length,
            if filtered_size.is_some() {
                "fractal heap object exceeds filtered direct block"
            } else {
                "fractal heap object exceeds direct block"
            },
        )?;
        let key = DirectBlockCacheKey {
            block_addr,
            block_size,
            filtered_size,
            filter_mask,
        };

        if let Some(filtered_size) = filtered_size {
            let data = match cache.direct_blocks.entry(key) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => {
                    reader.seek(block_addr)?;
                    let mut filtered =
                        vec![
                            0;
                            heap_object_len(filtered_size, "filtered fractal heap block size",)?
                        ];
                    reader.read_bytes_into(&mut filtered)?;
                    let pipeline = self.filter_pipeline.as_ref().ok_or_else(|| {
                        Error::InvalidFormat("filtered fractal heap missing filter pipeline".into())
                    })?;
                    let mut data = Vec::new();
                    crate::filters::apply_pipeline_reverse_with_mask_into(
                        &filtered,
                        pipeline,
                        1,
                        filter_mask,
                        &mut data,
                    )?;
                    entry.insert(data)
                }
            };
            let slice = data.get(range).ok_or_else(|| {
                Error::InvalidFormat("fractal heap object exceeds filtered direct block".into())
            })?;
            self.trace_managed_object(block_addr, block_size, offset, length, filter_mask, true);
            return Ok(slice);
        }

        if self.has_checksum && self.heap_addr == 0 {
            verify_direct_block_checksum(reader, block_addr, self.max_heap_size, block_size)?;
        }
        let block = match cache.direct_blocks.entry(key) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                reader.seek(block_addr)?;
                let mut data =
                    vec![0; heap_object_len(block_size, "fractal heap direct block size")?];
                reader.read_bytes_into(&mut data)?;
                entry.insert(data)
            }
        };
        let data = &block[range];
        self.trace_managed_object(block_addr, block_size, offset, length, 0, false);
        Ok(data)
    }
}

fn direct_block_object_range(
    block_size: u64,
    offset: u64,
    length: u64,
    exceeds_context: &str,
) -> Result<Range<usize>> {
    let start = heap_object_len(offset, "fractal heap object offset")?;
    let len = heap_object_len(length, "fractal heap object length")?;
    let block_len = heap_object_len(block_size, "fractal heap direct block size")?;
    let end = start
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat("fractal heap object range overflow".into()))?;
    if end > block_len {
        return Err(Error::InvalidFormat(exceeds_context.into()));
    }
    Ok(start..end)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use crate::io::reader::HdfReader;

    use super::*;

    fn test_header() -> FractalHeapHeader {
        FractalHeapHeader {
            heap_addr: 0,
            heap_id_len: 8,
            io_filter_len: 0,
            flags: 0,
            max_managed_obj_size: 64,
            table_width: 1,
            start_block_size: 8,
            max_direct_block_size: 64,
            max_heap_size: 8,
            start_root_rows: 0,
            root_block_addr: 0,
            current_root_rows: 0,
            num_managed_objects: 0,
            has_checksum: false,
            sizeof_addr: 8,
            sizeof_size: 8,
            huge_btree_addr: 0,
            root_direct_filtered_size: None,
            root_direct_filter_mask: 0,
            filter_pipeline: None,
        }
    }

    #[test]
    fn direct_block_into_reuses_output_buffer() {
        let header = test_header();
        let mut reader = HdfReader::new(Cursor::new(b"0123456789abcdef".to_vec()));
        let mut out = Vec::with_capacity(32);

        header
            .read_from_direct_block_into(&mut reader, 0, 16, None, 0, 4, 6, &mut out)
            .unwrap();

        assert_eq!(out, b"456789");
        assert!(out.capacity() >= 32);
    }

    #[test]
    fn filtered_direct_block_validates_window_before_pipeline_lookup() {
        let header = FractalHeapHeader {
            io_filter_len: 1,
            ..test_header()
        };
        let mut reader = HdfReader::new(Cursor::new(Vec::<u8>::new()));
        let err = header
            .read_from_direct_block(&mut reader, 0, 8, Some(4), 0, 7, 2)
            .expect_err("invalid filtered object window should fail before filter decode");

        assert!(err
            .to_string()
            .contains("fractal heap object exceeds filtered direct block"));
    }

    #[test]
    fn cached_direct_block_slice_borrows_cached_payload() {
        let header = test_header();
        let mut reader = HdfReader::new(Cursor::new(b"abcdefghijklmnop".to_vec()));
        let mut cache = FractalHeapManagedObjectCache::new();

        let slice = header
            .read_from_direct_block_cached_slice(&mut reader, 0, 16, None, 0, 2, 5, &mut cache)
            .unwrap();

        assert_eq!(slice, b"cdefg");
    }
}
