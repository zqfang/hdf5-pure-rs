//! Fractal heap managed-object access — mirrors libhdf5's `H5HFman.c`.
//! Decode the heap-ID for managed objects, descend through the doubling
//! table (direct or indirect / filtered or unfiltered), and return the
//! object bytes. Composes with the iblock/dblock decoders.

use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::io::reader::HdfReader;

use super::iblock::{FilteredIndirectBlock, IndirectBlock};
use super::FractalHeapHeader;

impl FractalHeapHeader {
    /// Read a managed (type 0) object.
    pub(super) fn read_managed<R: Read + Seek>(
        &self,
        reader: &mut HdfReader<R>,
        heap_id: &[u8],
    ) -> Result<Vec<u8>> {
        // Managed object heap ID:
        // byte 0: version(2 bits) + type(2 bits) + reserved(4 bits)
        // then: offset (ceil(max_heap_size/8) bytes) + length (remaining bytes)

        let offset_bytes = managed_heap_offset_bytes(self.max_heap_size)?;
        let offset_window = managed_heap_id_window(heap_id, 1, offset_bytes, "offset")?;
        let offset = read_le_u64_prefix(offset_window, offset_bytes, "heap ID offset")?;

        let len_start = 1usize
            .checked_add(offset_bytes)
            .ok_or_else(|| Error::InvalidFormat("heap ID length offset overflow".into()))?;
        let length_window = heap_id
            .get(len_start..)
            .ok_or_else(|| Error::InvalidFormat("heap ID too short for length".into()))?;
        let length =
            read_le_u64_prefix(length_window, length_window.len().min(8), "heap ID length")?;

        // Bound checks matching libhdf5's `H5HF__man_op_real`:
        //  - offset must be < 2^max_heap_size (the heap's address space)
        //  - object size must fit in the managed object size limit
        if self.max_heap_size < 64 && offset >= (1u64 << self.max_heap_size) {
            return Err(Error::InvalidFormat(format!(
                "fractal heap object offset {offset} exceeds 2^{} address space",
                self.max_heap_size
            )));
        }
        if length > u64::from(self.max_managed_obj_size) {
            return Err(Error::InvalidFormat(format!(
                "fractal heap object size {length} exceeds max managed object size {}",
                self.max_managed_obj_size
            )));
        }

        if self.current_root_rows == 0 {
            // Root is a direct block -- offset is relative to block start
            self.read_from_direct_block(
                reader,
                self.root_block_addr,
                self.start_block_size,
                self.root_direct_filtered_size,
                self.root_direct_filter_mask,
                offset,
                length,
            )
        } else {
            // Root is an indirect block -- need to find which direct block contains the offset
            self.read_from_indirect_block(reader, self.root_block_addr, offset, length)
        }
    }

    pub(super) fn read_from_indirect_block<R: Read + Seek>(
        &self,
        reader: &mut HdfReader<R>,
        block_addr: u64,
        offset: u64,
        length: u64,
    ) -> Result<Vec<u8>> {
        if self.io_filter_len > 0 {
            return self.read_from_filtered_indirect_block(reader, block_addr, offset, length);
        }

        self.read_from_indirect_block_rows(
            reader,
            block_addr,
            usize::from(self.current_root_rows),
            0,
            offset,
            length,
        )
    }

    /// Walk a decoded indirect block to locate the heap object covering
    /// `offset`. Mirrors libhdf5's `H5HF__man_op_real` traversal: walks
    /// the row table, descending into nested indirect blocks once we leave
    /// the direct-row range.
    pub(super) fn lookup_in_indirect_block<R: Read + Seek>(
        &self,
        reader: &mut HdfReader<R>,
        iblock: &IndirectBlock,
        block_start: u64,
        offset: u64,
        length: u64,
    ) -> Result<Vec<u8>> {
        let width = usize::from(self.table_width);
        let max_direct_rows = self.max_direct_rows();
        let mut current_heap_offset = block_start;
        let mut entry_index = 0usize;

        for row in 0..iblock.nrows {
            if row < max_direct_rows {
                let block_span = self.checked_row_block_size(row)?;
                for _ in 0..width {
                    let child_addr = iblock.child_addrs[entry_index];
                    entry_index += 1;

                    if crate::io::reader::is_undef_addr(child_addr) {
                        current_heap_offset =
                            checked_add_heap_offset(current_heap_offset, block_span)?;
                        continue;
                    }

                    let block_end = checked_add_heap_offset(current_heap_offset, block_span)?;
                    if offset >= current_heap_offset && offset < block_end {
                        let local_offset = offset - current_heap_offset;
                        return self.read_from_direct_block(
                            reader,
                            child_addr,
                            block_span,
                            None,
                            0,
                            local_offset,
                            length,
                        );
                    }

                    current_heap_offset = block_end;
                }
            } else {
                let child_rows = self.child_indirect_rows(row)?;
                let child_span = self.indirect_data_span(reader, child_rows)?;
                for _ in 0..width {
                    let child_addr = iblock.child_addrs[entry_index];
                    entry_index += 1;
                    let child_end = checked_add_heap_offset(current_heap_offset, child_span)?;
                    if offset >= current_heap_offset && offset < child_end {
                        if crate::io::reader::is_undef_addr(child_addr) {
                            break;
                        }
                        return self.read_from_indirect_block_rows(
                            reader,
                            child_addr,
                            child_rows,
                            current_heap_offset,
                            offset,
                            length,
                        );
                    }
                    current_heap_offset = child_end;
                }
            }
        }

        Err(Error::InvalidFormat(format!(
            "fractal heap offset {offset} not found in indirect block"
        )))
    }

    /// Drive `decode_indirect_block` + `lookup_in_indirect_block` — the
    /// C-side composition is `H5HF__man_iblock_protect` (which loads &
    /// deserializes the iblock) followed by the lookup loop in
    /// `H5HF__man_op_real`.
    pub(super) fn read_from_indirect_block_rows<R: Read + Seek>(
        &self,
        reader: &mut HdfReader<R>,
        block_addr: u64,
        nrows: usize,
        block_start: u64,
        offset: u64,
        length: u64,
    ) -> Result<Vec<u8>> {
        let iblock = self.decode_indirect_block(reader, block_addr, nrows)?;
        self.lookup_in_indirect_block(reader, &iblock, block_start, offset, length)
    }

    /// Walk a decoded filtered indirect block to locate the heap object
    /// covering `offset`. Mirrors the filtered traversal in
    /// `H5HF__man_op_real`.
    pub(super) fn lookup_in_filtered_indirect_block<R: Read + Seek>(
        &self,
        reader: &mut HdfReader<R>,
        iblock: &FilteredIndirectBlock,
        offset: u64,
        length: u64,
    ) -> Result<Vec<u8>> {
        let width = usize::from(self.table_width);
        let dblock_header_size = checked_add_heap_offset(
            checked_add_heap_offset(5, u64::from(self.sizeof_addr))?,
            checked_add_heap_offset(
                u64::try_from(iblock.block_offset_bytes).map_err(|_| {
                    Error::InvalidFormat("filtered indirect block offset width overflow".into())
                })?,
                if self.has_checksum { 4 } else { 0 },
            )?,
        )?;
        let mut current_heap_offset = 0u64;
        let mut entry_index = 0usize;

        for row in 0..iblock.nrows {
            let block_size = self.checked_row_block_size(row)?;

            if block_size > self.max_direct_block_size {
                entry_index = entry_index.checked_add(width).ok_or_else(|| {
                    Error::InvalidFormat("fractal heap filtered entry index overflow".into())
                })?; // indirect-row entries carry no payload to consume here
                continue;
            }

            let data_capacity = block_size.checked_sub(dblock_header_size).ok_or_else(|| {
                Error::InvalidFormat("fractal heap direct block header exceeds block size".into())
            })?;
            for _ in 0..width {
                let entry = &iblock.entries[entry_index];
                entry_index += 1;
                if crate::io::reader::is_undef_addr(entry.addr) {
                    current_heap_offset =
                        checked_add_heap_offset(current_heap_offset, data_capacity)?;
                    continue;
                }
                let block_end = checked_add_heap_offset(current_heap_offset, data_capacity)?;
                if offset >= current_heap_offset && offset < block_end {
                    return self.read_from_direct_block(
                        reader,
                        entry.addr,
                        block_size,
                        Some(entry.filtered_size),
                        entry.filter_mask,
                        offset - current_heap_offset,
                        length,
                    );
                }
                current_heap_offset = block_end;
            }
        }

        Err(Error::InvalidFormat(format!(
            "filtered fractal heap offset {offset} not found in indirect block"
        )))
    }

    /// Drive `decode_filtered_indirect_block` + `lookup_in_filtered_…` —
    /// C-side composition is `H5HF__man_iblock_protect` + the filtered
    /// branch of `H5HF__man_op_real`.
    pub(super) fn read_from_filtered_indirect_block<R: Read + Seek>(
        &self,
        reader: &mut HdfReader<R>,
        block_addr: u64,
        offset: u64,
        length: u64,
    ) -> Result<Vec<u8>> {
        let iblock = self.decode_filtered_indirect_block(reader, block_addr)?;
        self.lookup_in_filtered_indirect_block(reader, &iblock, offset, length)
    }
}

fn managed_heap_offset_bytes(max_heap_size: u16) -> Result<usize> {
    let bytes = (usize::from(max_heap_size) + 7) / 8;
    if bytes > 8 {
        return Err(Error::Unsupported(format!(
            "managed heap ID offset uses {bytes} bytes"
        )));
    }
    Ok(bytes)
}

fn managed_heap_id_window<'a>(
    heap_id: &'a [u8],
    offset: usize,
    len: usize,
    field: &str,
) -> Result<&'a [u8]> {
    let end = offset
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("heap ID {field} offset overflow")))?;
    heap_id
        .get(offset..end)
        .ok_or_else(|| Error::InvalidFormat(format!("heap ID too short for {field}")))
}

fn read_le_u64_prefix(bytes: &[u8], len: usize, context: &str) -> Result<u64> {
    if len > 8 || len > bytes.len() {
        return Err(Error::InvalidFormat(format!(
            "{context} byte count is invalid"
        )));
    }
    let mut value = 0u64;
    for (i, byte) in bytes.iter().take(len).enumerate() {
        value |= u64::from(*byte) << (i * 8);
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn managed_heap_offset_bytes_rejects_wider_than_u64() {
        let err = managed_heap_offset_bytes(72).unwrap_err();
        assert!(
            err.to_string()
                .contains("managed heap ID offset uses 9 bytes"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn managed_heap_id_window_rejects_offset_overflow() {
        let err = managed_heap_id_window(&[], usize::MAX, 1, "offset").unwrap_err();
        assert!(
            err.to_string().contains("heap ID offset offset overflow"),
            "unexpected error: {err}"
        );
    }
}

fn checked_add_heap_offset(lhs: u64, rhs: u64) -> Result<u64> {
    lhs.checked_add(rhs)
        .ok_or_else(|| Error::InvalidFormat("fractal heap offset span overflow".into()))
}
