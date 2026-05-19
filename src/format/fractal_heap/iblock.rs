//! Fractal heap indirect blocks — mirrors libhdf5's `H5HFiblock.c` plus
//! the iblock half of `H5HFcache.c` (`H5HF__cache_iblock_deserialize`).
//! Pure decoders for both unfiltered and filtered indirect blocks; the
//! traversal logic that consumes these decoded blocks lives in `man.rs`.

use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::io::reader::HdfReader;

use super::{verify_metadata_checksum, FractalHeapHeader, FHIB_MAGIC};

/// Decoded fractal-heap indirect block: row count + flat list of child
/// pointers in (row, column) order. Output of `decode_indirect_block`,
/// consumed by `lookup_in_indirect_block`.
#[derive(Debug, Clone)]
pub(super) struct IndirectBlock {
    pub(super) nrows: usize,
    pub(super) child_addrs: Vec<u64>,
}

/// Decoded *filtered* fractal-heap indirect block. Each direct-row entry
/// carries the `(addr, filtered_size, filter_mask)` triple read off disk;
/// indirect-row entries only carry the address (the other two fields are
/// 0 for those rows). Output of `decode_filtered_indirect_block`.
#[derive(Debug, Clone)]
pub(super) struct FilteredIndirectEntry {
    pub(super) addr: u64,
    pub(super) filtered_size: u64,
    pub(super) filter_mask: u32,
}

#[derive(Debug, Clone)]
pub(super) struct FilteredIndirectBlock {
    pub(super) nrows: usize,
    pub(super) block_offset_bytes: usize,
    pub(super) direct_rows: usize,
    pub(super) entries: Vec<FilteredIndirectEntry>,
}

impl FractalHeapHeader {
    /// Pure deserializer for a fractal-heap indirect block: validates the
    /// FHIB magic, reads the prefix, verifies the metadata checksum, and
    /// returns the table of child entries. Mirrors libhdf5's
    /// `H5HF__cache_iblock_deserialize` — no traversal of the listed
    /// addresses.
    pub(super) fn decode_indirect_block<R: Read + Seek>(
        &self,
        reader: &mut HdfReader<R>,
        block_addr: u64,
        nrows: usize,
    ) -> Result<IndirectBlock> {
        reader.seek(block_addr)?;

        let mut magic = [0; 4];
        reader.read_bytes_into(&mut magic)?;
        if magic != FHIB_MAGIC {
            return Err(Error::InvalidFormat(
                "invalid fractal heap indirect block magic".into(),
            ));
        }

        let version = reader.read_u8()?;
        if version != 0 {
            return Err(Error::InvalidFormat(format!(
                "fractal heap indirect block version {version}"
            )));
        }
        let heap_header_addr = reader.read_addr()?;
        if heap_header_addr != self.heap_addr {
            return Err(Error::InvalidFormat(
                "fractal heap indirect block owner address does not match header".into(),
            ));
        }

        let block_offset_bytes = heap_offset_bytes(self.max_heap_size)?;
        let width = usize::from(self.table_width);
        let total_entries = nrows
            .checked_mul(width)
            .ok_or_else(|| Error::InvalidFormat("fractal heap entry count overflow".into()))?;
        if self.has_checksum {
            let checksum_span = 4usize
                .checked_add(1)
                .and_then(|n| n.checked_add(usize::from(self.sizeof_addr)))
                .and_then(|n| n.checked_add(block_offset_bytes))
                .and_then(|n| {
                    n.checked_add(total_entries.checked_mul(usize::from(self.sizeof_addr))?)
                })
                .ok_or_else(|| {
                    Error::InvalidFormat(
                        "fractal heap indirect block checksum span overflow".into(),
                    )
                })?;
            verify_metadata_checksum(
                reader,
                block_addr,
                usize_to_u64(checksum_span, "fractal heap indirect block checksum span")?,
                "fractal heap indirect block",
            )?;
        }

        reader.skip(usize_to_u64(
            block_offset_bytes,
            "fractal heap indirect block offset width",
        )?)?;
        let mut child_addrs = Vec::with_capacity(total_entries);
        for _ in 0..total_entries {
            child_addrs.push(reader.read_addr()?);
        }
        Ok(IndirectBlock { nrows, child_addrs })
    }

    /// Pure deserializer for a *filtered* fractal-heap indirect block.
    /// Each direct-row entry carries an extra (filtered_size, filter_mask)
    /// pair after the address; rows past `max_direct_rows` only carry
    /// addresses (and we still need to consume them to keep the reader
    /// aligned). Mirrors the filtered branch of
    /// `H5HF__cache_iblock_deserialize`.
    pub(super) fn decode_filtered_indirect_block<R: Read + Seek>(
        &self,
        reader: &mut HdfReader<R>,
        block_addr: u64,
    ) -> Result<FilteredIndirectBlock> {
        reader.seek(block_addr)?;
        let mut magic = [0; 4];
        reader.read_bytes_into(&mut magic)?;
        if magic != FHIB_MAGIC {
            return Err(Error::InvalidFormat(
                "invalid fractal heap indirect block magic".into(),
            ));
        }
        let version = reader.read_u8()?;
        if version != 0 {
            return Err(Error::InvalidFormat(format!(
                "fractal heap indirect block version {version}"
            )));
        }
        let heap_header_addr = reader.read_addr()?;
        if heap_header_addr != self.heap_addr {
            return Err(Error::InvalidFormat(
                "fractal heap indirect block owner address does not match header".into(),
            ));
        }
        let block_offset_bytes = heap_offset_bytes(self.max_heap_size)?;
        reader.skip(usize_to_u64(
            block_offset_bytes,
            "fractal heap indirect block offset width",
        )?)?;

        let nrows = usize::from(self.current_root_rows);
        let width = usize::from(self.table_width);
        if self.has_checksum {
            let mut entry_bytes = 0usize;
            for row in 0..nrows {
                let block_size = self.checked_row_block_size(row)?;
                let direct = block_size <= self.max_direct_block_size;
                let per_entry = usize::from(self.sizeof_addr)
                    + if direct {
                        usize::from(self.sizeof_size) + 4
                    } else {
                        0
                    };
                entry_bytes = entry_bytes
                    .checked_add(width.checked_mul(per_entry).ok_or_else(|| {
                        Error::InvalidFormat(
                            "fractal heap filtered indirect block checksum span overflow".into(),
                        )
                    })?)
                    .ok_or_else(|| {
                        Error::InvalidFormat(
                            "fractal heap filtered indirect block checksum span overflow".into(),
                        )
                    })?;
            }
            let checksum_span = 4usize
                .checked_add(1)
                .and_then(|n| n.checked_add(usize::from(self.sizeof_addr)))
                .and_then(|n| n.checked_add(block_offset_bytes))
                .and_then(|n| n.checked_add(entry_bytes))
                .ok_or_else(|| {
                    Error::InvalidFormat(
                        "fractal heap filtered indirect block checksum span overflow".into(),
                    )
                })?;
            verify_metadata_checksum(
                reader,
                block_addr,
                usize_to_u64(
                    checksum_span,
                    "fractal heap filtered indirect block checksum span",
                )?,
                "fractal heap filtered indirect block",
            )?;
        }

        let max_direct_rows = self.max_direct_rows_checked()?;
        let direct_rows = nrows.min(max_direct_rows);
        let mut entries = Vec::with_capacity(direct_rows.checked_mul(width).ok_or_else(|| {
            Error::InvalidFormat("fractal heap filtered direct entry count overflow".into())
        })?);
        for row in 0..nrows {
            let block_size = self.checked_row_block_size(row)?;
            let direct = block_size <= self.max_direct_block_size;
            for _ in 0..width {
                let addr = reader.read_addr()?;
                if direct {
                    let filtered_size = reader.read_length()?;
                    let filter_mask = reader.read_u32()?;
                    entries.push(FilteredIndirectEntry {
                        addr,
                        filtered_size,
                        filter_mask,
                    });
                } else {
                    debug_assert!(row >= max_direct_rows);
                }
            }
        }
        Ok(FilteredIndirectBlock {
            nrows,
            block_offset_bytes,
            direct_rows,
            entries,
        })
    }
}

fn heap_offset_bytes(max_heap_size: u16) -> Result<usize> {
    usize::from(max_heap_size)
        .checked_add(7)
        .map(|value| value / 8)
        .ok_or_else(|| Error::InvalidFormat("fractal heap offset width overflow".into()))
}

fn usize_to_u64(value: usize, context: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| Error::InvalidFormat(format!("{context} exceeds u64")))
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
            max_direct_block_size: 8,
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

    fn invalid_indirect_block() -> Vec<u8> {
        let mut block = b"FHIB".to_vec();
        block.push(1); // invalid version
        block.extend_from_slice(&0u64.to_le_bytes()); // heap header address
        block.push(0); // block offset, max_heap_size=8 -> 1 byte
        block
    }

    fn filtered_indirect_block_with_bad_checksum() -> Vec<u8> {
        let mut block = b"FHIB".to_vec();
        block.push(0);
        block.extend_from_slice(&0u64.to_le_bytes()); // heap header address
        block.push(0); // block offset
        block.extend_from_slice(&64u64.to_le_bytes()); // child direct block address
        block.extend_from_slice(&8u64.to_le_bytes()); // filtered size
        block.extend_from_slice(&0u32.to_le_bytes()); // filter mask
        block.extend_from_slice(&0xdead_beefu32.to_le_bytes()); // bad checksum
        block
    }

    fn indirect_block_with_owner(owner: u64) -> Vec<u8> {
        let mut block = b"FHIB".to_vec();
        block.push(0);
        block.extend_from_slice(&owner.to_le_bytes());
        block.push(0); // block offset, max_heap_size=8 -> 1 byte
        block
    }

    #[test]
    fn indirect_block_rejects_unsupported_version() {
        let header = test_header();
        let mut reader = HdfReader::new(Cursor::new(invalid_indirect_block()));
        let err = match header.decode_indirect_block(&mut reader, 0, 0) {
            Ok(_) => panic!("invalid indirect block version should fail"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("version"));
    }

    #[test]
    fn indirect_blocks_reject_owner_mismatch() {
        let header = test_header();
        let mut reader = HdfReader::new(Cursor::new(indirect_block_with_owner(1)));
        let err = match header.decode_indirect_block(&mut reader, 0, 0) {
            Ok(_) => panic!("indirect block owner mismatch should fail"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("owner address"));

        let header = FractalHeapHeader {
            io_filter_len: 1,
            ..test_header()
        };
        let mut reader = HdfReader::new(Cursor::new(indirect_block_with_owner(1)));
        let err = match header.decode_filtered_indirect_block(&mut reader, 0) {
            Ok(_) => panic!("filtered indirect block owner mismatch should fail"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("owner address"));
    }

    #[test]
    fn filtered_indirect_block_rejects_unsupported_version() {
        let header = FractalHeapHeader {
            io_filter_len: 1,
            ..test_header()
        };
        let mut reader = HdfReader::new(Cursor::new(invalid_indirect_block()));
        let err = match header.decode_filtered_indirect_block(&mut reader, 0) {
            Ok(_) => panic!("invalid filtered indirect block version should fail"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("version"));
    }

    #[test]
    fn filtered_indirect_block_rejects_bad_checksum() {
        let header = FractalHeapHeader {
            io_filter_len: 1,
            current_root_rows: 1,
            has_checksum: true,
            ..test_header()
        };
        let mut reader = HdfReader::new(Cursor::new(filtered_indirect_block_with_bad_checksum()));
        let err = match header.decode_filtered_indirect_block(&mut reader, 0) {
            Ok(_) => panic!("bad filtered indirect block checksum should fail"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("checksum"));
    }
}
