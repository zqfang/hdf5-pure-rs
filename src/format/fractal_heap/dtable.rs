//! Fractal heap doubling-table geometry — mirrors libhdf5's
//! `H5HFdtable.c`. Pure-arithmetic helpers that compute block sizes,
//! row counts, and span totals from the heap header parameters.

use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::io::reader::HdfReader;

use super::{log2_floor, log2_power2, FractalHeapHeader};

impl FractalHeapHeader {
    pub(super) fn max_direct_rows(&self) -> usize {
        let start_bits = log2_power2(self.start_block_size);
        let max_direct_bits = log2_power2(self.max_direct_block_size);
        usize::try_from(max_direct_bits - start_bits + 2).unwrap_or(usize::MAX)
    }

    pub(super) fn indirect_data_span<R: Read + Seek>(
        &self,
        reader: &HdfReader<R>,
        nrows: usize,
    ) -> Result<u64> {
        let width = u64::from(self.table_width);
        let max_direct_rows = self.max_direct_rows();
        let mut span = 0u64;

        for row in 0..nrows {
            if row < max_direct_rows {
                let row_span = self
                    .checked_row_block_size(row)?
                    .checked_mul(width)
                    .ok_or_else(|| Error::InvalidFormat("fractal heap row span overflow".into()))?;
                span = span
                    .checked_add(row_span)
                    .ok_or_else(|| Error::InvalidFormat("fractal heap span overflow".into()))?;
            } else {
                let child_rows = self.child_indirect_rows(row)?;
                let child_span = self
                    .indirect_data_span(reader, child_rows)?
                    .checked_mul(width)
                    .ok_or_else(|| {
                        Error::InvalidFormat("fractal heap child span overflow".into())
                    })?;
                span = span
                    .checked_add(child_span)
                    .ok_or_else(|| Error::InvalidFormat("fractal heap span overflow".into()))?;
            }
        }

        Ok(span)
    }

    pub(super) fn checked_row_block_size(&self, row: usize) -> Result<u64> {
        if row == 0 {
            return Ok(self.start_block_size);
        }
        let shift = u32::try_from(row - 1)
            .ok()
            .and_then(|shift| 1u64.checked_shl(shift))
            .ok_or_else(|| Error::InvalidFormat("fractal heap row block shift overflow".into()))?;
        self.start_block_size
            .checked_mul(shift)
            .ok_or_else(|| Error::InvalidFormat("fractal heap row block size overflow".into()))
    }

    pub(super) fn child_indirect_rows(&self, row: usize) -> Result<usize> {
        let first_row_bits =
            log2_power2(self.start_block_size) + log2_power2(u64::from(self.table_width));
        let block_bits = log2_floor(self.checked_row_block_size(row)?);
        if block_bits < first_row_bits {
            return Err(Error::InvalidFormat(
                "fractal heap indirect row geometry underflows".into(),
            ));
        }
        usize::try_from(block_bits - first_row_bits + 1)
            .map_err(|_| Error::InvalidFormat("fractal heap child row count overflow".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::FractalHeapHeader;

    fn test_header() -> FractalHeapHeader {
        FractalHeapHeader {
            heap_addr: 0,
            heap_id_len: 0,
            io_filter_len: 0,
            flags: 0,
            max_managed_obj_size: 1024,
            table_width: 4,
            start_block_size: 8,
            max_direct_block_size: 1024,
            max_heap_size: 64,
            start_root_rows: 0,
            root_block_addr: 0,
            current_root_rows: 0,
            num_managed_objects: 0,
            has_checksum: false,
            sizeof_addr: 8,
            sizeof_size: 8,
            huge_btree_addr: u64::MAX,
            root_direct_filtered_size: None,
            root_direct_filter_mask: 0,
            filter_pipeline: None,
        }
    }

    #[test]
    fn row_block_size_rejects_shift_overflow() {
        let header = test_header();
        let err = header.checked_row_block_size(65).unwrap_err();
        assert!(err.to_string().contains("overflow"));
    }

    #[test]
    fn row_block_size_rejects_multiply_overflow() {
        let mut header = test_header();
        header.start_block_size = 8;
        let err = header.checked_row_block_size(64).unwrap_err();
        assert!(err.to_string().contains("overflow"));
    }
}
