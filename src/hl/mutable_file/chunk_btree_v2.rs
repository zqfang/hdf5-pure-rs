use std::io::{Seek, SeekFrom, Write};

use crate::error::{Error, Result};
use crate::format::btree_v2::BTreeV2Header;
use crate::format::checksum::checksum_metadata;

use super::MutableFile;

impl MutableFile {
    pub(super) fn rewrite_btree_v2_chunk(
        &mut self,
        index_addr: u64,
        info: &crate::hl::dataset::DatasetInfo,
        chunk_coords: &[u64],
        chunk_dims: &[u64],
        chunk_size: u64,
        chunk_addr: u64,
        unfiltered_chunk_bytes: usize,
    ) -> Result<()> {
        let scaled_coords = Self::scaled_chunk_coords(chunk_coords, chunk_dims)?;
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
        let sa = usize::from(self.superblock.sizeof_addr);
        let expected_record_size =
            Self::btree_v2_chunk_record_size(sa, filtered, chunk_size_len, scaled_coords.len())?;

        let new_record = Self::encode_btree_v2_chunk_record(
            chunk_addr,
            chunk_size,
            &scaled_coords,
            filtered,
            chunk_size_len,
            sa,
        )?;

        let mut guard = self.inner.lock();
        let reader = &mut guard.reader;
        let header = BTreeV2Header::read_at(reader, index_addr)?;
        if usize::from(header.record_size) != expected_record_size {
            return Err(Error::InvalidFormat(format!(
                "v2 B-tree chunk record size {} does not match expected {expected_record_size}",
                header.record_size
            )));
        }
        if header.tree_type != 10 && header.tree_type != 11 {
            return Err(Error::Unsupported(format!(
                "write_chunk cannot update v2 B-tree type {} chunk indexes",
                header.tree_type
            )));
        }

        let raw_records = crate::format::btree_v2::collect_all_records(reader, index_addr)?;
        let mut sortable_records = Vec::with_capacity(raw_records.len() + 1);
        let mut replacing = false;
        for record in raw_records {
            let existing_scaled = Self::decode_btree_v2_scaled_coords(
                &record,
                filtered,
                chunk_size_len,
                sa,
                scaled_coords.len(),
            )?;
            if existing_scaled == scaled_coords {
                sortable_records.push((existing_scaled, new_record.clone()));
                replacing = true;
            } else {
                sortable_records.push((existing_scaled, record));
            }
        }
        if !replacing {
            sortable_records.push((scaled_coords, new_record));
        }
        sortable_records.sort_by(|a, b| a.0.cmp(&b.0));
        let records: Vec<Vec<u8>> = sortable_records
            .into_iter()
            .map(|(_, record)| record)
            .collect();
        drop(guard);

        self.rebuild_btree_v2_chunk_tree(index_addr, &header, &records)
    }

    fn rebuild_btree_v2_chunk_tree(
        &mut self,
        header_addr: u64,
        header: &BTreeV2Header,
        records: &[Vec<u8>],
    ) -> Result<()> {
        let leaf_capacity = Self::btree_v2_leaf_capacity(header)?;
        if records.len() <= leaf_capacity {
            let root_addr = self.append_btree_v2_leaf(header, records)?;
            let root_nrecords = Self::usize_to_u16(records.len(), "v2 B-tree root record count")?;
            self.rewrite_btree_v2_header_root(
                header_addr,
                header,
                0,
                root_addr,
                root_nrecords,
                Self::usize_to_u64(records.len(), "v2 B-tree total record count")?,
            )?;
            return Ok(());
        }

        let internal_capacity = self.btree_v2_depth1_internal_capacity(header)?;
        let mut leaf_count = 2usize;
        while records.len()
            > leaf_count
                .checked_mul(leaf_capacity.checked_add(1).ok_or_else(|| {
                    Error::InvalidFormat("v2 B-tree leaf capacity overflow".into())
                })?)
                .and_then(|value| value.checked_sub(1))
                .ok_or_else(|| Error::InvalidFormat("v2 B-tree leaf count overflow".into()))?
        {
            leaf_count = leaf_count
                .checked_add(1)
                .ok_or_else(|| Error::InvalidFormat("v2 B-tree leaf count overflow".into()))?;
        }
        if leaf_count - 1 > internal_capacity {
            return Err(Error::Unsupported(
                "write_chunk cannot rebuild v2 B-tree chunk indexes beyond a depth-1 root yet"
                    .into(),
            ));
        }

        let leaf_record_total = records.len() - (leaf_count - 1);
        let mut record_pos = 0usize;
        let mut remaining_leaf_records = leaf_record_total;
        let mut children = Vec::with_capacity(leaf_count);
        let mut separators = Vec::with_capacity(leaf_count - 1);

        for leaf_index in 0..leaf_count {
            let remaining_leaves = leaf_count - leaf_index;
            let take = remaining_leaf_records.div_ceil(remaining_leaves);
            if take == 0 || take > leaf_capacity {
                return Err(Error::InvalidFormat(
                    "invalid v2 B-tree chunk leaf distribution".into(),
                ));
            }
            let leaf_end = record_pos.checked_add(take).ok_or_else(|| {
                Error::InvalidFormat("v2 B-tree rebuild record offset overflow".into())
            })?;
            let leaf_records = records.get(record_pos..leaf_end).ok_or_else(|| {
                Error::InvalidFormat("invalid v2 B-tree chunk leaf distribution".into())
            })?;
            let leaf_addr = self.append_btree_v2_leaf(header, leaf_records)?;
            children.push((
                leaf_addr,
                Self::usize_to_u16(take, "v2 B-tree child record count")?,
            ));
            record_pos = leaf_end;
            remaining_leaf_records = remaining_leaf_records.checked_sub(take).ok_or_else(|| {
                Error::InvalidFormat("invalid v2 B-tree chunk leaf distribution".into())
            })?;
            if leaf_index + 1 < leaf_count {
                let separator = records.get(record_pos).ok_or_else(|| {
                    Error::InvalidFormat("invalid v2 B-tree chunk leaf distribution".into())
                })?;
                separators.push(separator.clone());
                record_pos = record_pos.checked_add(1).ok_or_else(|| {
                    Error::InvalidFormat("v2 B-tree rebuild record offset overflow".into())
                })?;
            }
        }
        if record_pos != records.len() {
            return Err(Error::InvalidFormat(
                "v2 B-tree chunk rebuild did not consume all records".into(),
            ));
        }

        let root_addr = self.append_btree_v2_depth1_internal(header, &separators, &children)?;
        self.rewrite_btree_v2_header_root(
            header_addr,
            header,
            1,
            root_addr,
            Self::usize_to_u16(separators.len(), "v2 B-tree root record count")?,
            Self::usize_to_u64(records.len(), "v2 B-tree total record count")?,
        )
    }

    /// Pure encoder for a v2 B-tree leaf node (BTLF magic + records +
    /// checksum). Mirrors the serialize half of libhdf5's
    /// `H5B2__cache_leaf_serialize`.
    fn encode_btree_v2_leaf(header: &BTreeV2Header, records: &[Vec<u8>]) -> Result<Vec<u8>> {
        if records.len() > usize::from(u16::MAX) {
            return Err(Error::Unsupported(
                "v2 B-tree leaf record count exceeds u16".into(),
            ));
        }
        let record_size = usize::from(header.record_size);
        let records_bytes = records
            .len()
            .checked_mul(record_size)
            .ok_or_else(|| Error::InvalidFormat("v2 B-tree leaf size overflow".into()))?;
        let leaf_capacity = 6usize
            .checked_add(records_bytes)
            .and_then(|value| value.checked_add(4))
            .ok_or_else(|| Error::InvalidFormat("v2 B-tree leaf size overflow".into()))?;
        let mut leaf = Vec::with_capacity(leaf_capacity);
        leaf.extend_from_slice(b"BTLF");
        leaf.push(0);
        leaf.push(header.tree_type);
        for record in records {
            if record.len() != record_size {
                return Err(Error::InvalidFormat(
                    "v2 B-tree leaf record has wrong size".into(),
                ));
            }
            leaf.extend_from_slice(record);
        }
        let checksum = checksum_metadata(&leaf);
        leaf.extend_from_slice(&checksum.to_le_bytes());
        Ok(leaf)
    }

    /// Allocate + encode + write a v2 B-tree leaf node.
    fn append_btree_v2_leaf(&mut self, header: &BTreeV2Header, records: &[Vec<u8>]) -> Result<u64> {
        let leaf = Self::encode_btree_v2_leaf(header, records)?;
        let addr = self.append_aligned_zeros(leaf.len(), 8)?;
        self.write_handle.seek(SeekFrom::Start(addr))?;
        self.write_handle.write_all(&leaf)?;
        Ok(addr)
    }

    /// Pure encoder for a v2 B-tree depth-1 internal node (BTIN magic +
    /// separators + child pointers + checksum). Mirrors the serialize
    /// half of libhdf5's `H5B2__cache_int_serialize`.
    fn encode_btree_v2_depth1_internal(
        &self,
        header: &BTreeV2Header,
        separators: &[Vec<u8>],
        children: &[(u64, u16)],
    ) -> Result<Vec<u8>> {
        if children.len() != separators.len() + 1 {
            return Err(Error::InvalidFormat(
                "v2 B-tree internal child/record count mismatch".into(),
            ));
        }
        if separators.len() > usize::from(u16::MAX) {
            return Err(Error::Unsupported(
                "v2 B-tree internal record count exceeds u16".into(),
            ));
        }
        let leaf_capacity = Self::btree_v2_leaf_capacity(header)?;
        let child_nrecords_size = Self::bytes_needed(Self::usize_to_u64(
            leaf_capacity,
            "v2 B-tree leaf record capacity",
        )?);
        let sa = usize::from(self.superblock.sizeof_addr);
        let record_size = usize::from(header.record_size);

        let mut node = Vec::new();
        node.extend_from_slice(b"BTIN");
        node.push(0);
        node.push(header.tree_type);
        for record in separators {
            if record.len() != record_size {
                return Err(Error::InvalidFormat(
                    "v2 B-tree internal separator has wrong size".into(),
                ));
            }
            node.extend_from_slice(record);
        }
        for &(child_addr, child_nrecords) in children {
            node.extend_from_slice(&Self::encode_uint_le(
                child_addr,
                sa,
                "v2 B-tree child address",
            )?);
            node.extend_from_slice(&Self::encode_uint_le(
                u64::from(child_nrecords),
                child_nrecords_size,
                "v2 B-tree child record count",
            )?);
        }
        let checksum = checksum_metadata(&node);
        node.extend_from_slice(&checksum.to_le_bytes());
        Ok(node)
    }

    /// Allocate + encode + write a v2 B-tree depth-1 internal node.
    fn append_btree_v2_depth1_internal(
        &mut self,
        header: &BTreeV2Header,
        separators: &[Vec<u8>],
        children: &[(u64, u16)],
    ) -> Result<u64> {
        let node = self.encode_btree_v2_depth1_internal(header, separators, children)?;
        let addr = self.append_aligned_zeros(node.len(), 8)?;
        self.write_handle.seek(SeekFrom::Start(addr))?;
        self.write_handle.write_all(&node)?;
        Ok(addr)
    }

    fn btree_v2_depth1_internal_capacity(&self, header: &BTreeV2Header) -> Result<usize> {
        let node_size = usize::try_from(header.node_size)
            .map_err(|_| Error::InvalidFormat("v2 B-tree node size is too large".into()))?;
        let record_size = usize::from(header.record_size);
        let leaf_capacity = Self::btree_v2_leaf_capacity(header)?;
        let max_nrec_size = Self::bytes_needed(Self::usize_to_u64(
            leaf_capacity,
            "v2 B-tree leaf record capacity",
        )?);
        let pointer_size = usize::from(self.superblock.sizeof_addr)
            .checked_add(max_nrec_size)
            .ok_or_else(|| Error::InvalidFormat("v2 B-tree pointer size overflow".into()))?;
        let metadata_and_pointer = 10usize
            .checked_add(pointer_size)
            .ok_or_else(|| Error::InvalidFormat("v2 B-tree pointer size overflow".into()))?;
        if node_size <= metadata_and_pointer || record_size == 0 {
            return Err(Error::InvalidFormat(
                "v2 B-tree internal node cannot hold records".into(),
            ));
        }
        let record_and_pointer = record_size
            .checked_add(pointer_size)
            .ok_or_else(|| Error::InvalidFormat("v2 B-tree pointer size overflow".into()))?;
        let capacity = (node_size - metadata_and_pointer) / record_and_pointer;
        if capacity == 0 {
            return Err(Error::InvalidFormat(
                "v2 B-tree internal node cannot hold records".into(),
            ));
        }
        Ok(capacity)
    }

    fn bytes_needed(mut value: u64) -> usize {
        let mut bytes = 1usize;
        while value > 0xff {
            value >>= 8;
            bytes += 1;
        }
        bytes
    }

    fn rewrite_btree_v2_header_root(
        &mut self,
        header_addr: u64,
        _header: &BTreeV2Header,
        new_depth: u16,
        new_root_addr: u64,
        new_root_nrecords: u16,
        new_total_records: u64,
    ) -> Result<()> {
        let sa = usize::from(self.superblock.sizeof_addr);
        let ss = usize::from(self.superblock.sizeof_size);
        let depth_pos = header_addr
            .checked_add(u64::from(4u8 + 1 + 1 + 4 + 2))
            .ok_or_else(|| Error::InvalidFormat("v2 B-tree header offset overflow".into()))?;
        let root_addr_pos = header_addr
            .checked_add(u64::from(4u8 + 1 + 1 + 4 + 2 + 2 + 1 + 1))
            .ok_or_else(|| Error::InvalidFormat("v2 B-tree header offset overflow".into()))?;
        let root_nrecords_pos = root_addr_pos
            .checked_add(Self::usize_to_u64(sa, "v2 B-tree address width")?)
            .ok_or_else(|| Error::InvalidFormat("v2 B-tree header offset overflow".into()))?;
        let total_records_pos = root_nrecords_pos
            .checked_add(2)
            .ok_or_else(|| Error::InvalidFormat("v2 B-tree header offset overflow".into()))?;
        let checksum_pos = total_records_pos
            .checked_add(Self::usize_to_u64(ss, "v2 B-tree length width")?)
            .ok_or_else(|| Error::InvalidFormat("v2 B-tree header offset overflow".into()))?;

        self.write_handle.seek(SeekFrom::Start(depth_pos))?;
        self.write_handle.write_all(&new_depth.to_le_bytes())?;
        self.write_handle.seek(SeekFrom::Start(root_addr_pos))?;
        self.write_handle.write_all(&Self::encode_uint_le(
            new_root_addr,
            sa,
            "v2 B-tree root address",
        )?)?;
        self.write_handle.seek(SeekFrom::Start(root_nrecords_pos))?;
        self.write_handle
            .write_all(&new_root_nrecords.to_le_bytes())?;
        self.write_handle.seek(SeekFrom::Start(total_records_pos))?;
        self.write_uint_le(new_total_records, ss)?;

        let check_len = usize::try_from(checksum_pos - header_addr).map_err(|_| {
            Error::InvalidFormat("v2 B-tree header checksum span is too large".into())
        })?;
        let mut guard = self.inner.lock();
        guard.reader.seek(header_addr)?;
        let mut header_bytes = guard.reader.read_bytes(check_len)?;
        drop(guard);
        let depth_offset = Self::checked_header_relative_offset(depth_pos, header_addr)?;
        let root_addr_offset = Self::checked_header_relative_offset(root_addr_pos, header_addr)?;
        let root_nrecords_offset =
            Self::checked_header_relative_offset(root_nrecords_pos, header_addr)?;
        let total_offset = Self::checked_header_relative_offset(total_records_pos, header_addr)?;
        Self::header_window_mut(&mut header_bytes, depth_offset, 2)?
            .copy_from_slice(&new_depth.to_le_bytes());
        Self::header_window_mut(&mut header_bytes, root_addr_offset, sa)?.copy_from_slice(
            &Self::encode_uint_le(new_root_addr, sa, "v2 B-tree root address")?,
        );
        Self::header_window_mut(&mut header_bytes, root_nrecords_offset, 2)?
            .copy_from_slice(&new_root_nrecords.to_le_bytes());
        Self::header_window_mut(&mut header_bytes, total_offset, ss)?.copy_from_slice(
            &Self::encode_uint_le(new_total_records, ss, "v2 B-tree total record count")?,
        );
        let checksum = checksum_metadata(&header_bytes);
        self.write_handle.seek(SeekFrom::Start(checksum_pos))?;
        self.write_handle.write_all(&checksum.to_le_bytes())?;
        Ok(())
    }

    fn btree_v2_leaf_capacity(header: &BTreeV2Header) -> Result<usize> {
        let node_size = usize::try_from(header.node_size)
            .map_err(|_| Error::InvalidFormat("v2 B-tree node size is too large".into()))?;
        let record_size = usize::from(header.record_size);
        if node_size <= 10 || record_size == 0 {
            return Err(Error::InvalidFormat("invalid v2 B-tree node sizing".into()));
        }
        let capacity = (node_size - 10) / record_size;
        if capacity == 0 {
            return Err(Error::InvalidFormat(
                "v2 B-tree leaf cannot hold any records".into(),
            ));
        }
        Ok(capacity)
    }

    fn btree_v2_chunk_record_size(
        sizeof_addr: usize,
        filtered: bool,
        chunk_size_len: usize,
        ndims: usize,
    ) -> Result<usize> {
        let filter_bytes = if filtered {
            chunk_size_len
                .checked_add(4)
                .ok_or_else(|| Error::InvalidFormat("v2 B-tree record size overflow".into()))?
        } else {
            0
        };
        let coord_bytes = ndims
            .checked_mul(8)
            .ok_or_else(|| Error::InvalidFormat("v2 B-tree record size overflow".into()))?;
        sizeof_addr
            .checked_add(filter_bytes)
            .and_then(|value| value.checked_add(coord_bytes))
            .ok_or_else(|| Error::InvalidFormat("v2 B-tree record size overflow".into()))
    }

    fn encode_btree_v2_chunk_record(
        addr: u64,
        chunk_size: u64,
        scaled_coords: &[u64],
        filtered: bool,
        chunk_size_len: usize,
        sizeof_addr: usize,
    ) -> Result<Vec<u8>> {
        let mut record = Vec::new();
        record.extend_from_slice(&Self::encode_uint_le(
            addr,
            sizeof_addr,
            "v2 B-tree chunk address",
        )?);
        if filtered {
            record.extend_from_slice(&Self::encode_uint_le(
                chunk_size,
                chunk_size_len,
                "v2 B-tree chunk size",
            )?);
            record.extend_from_slice(&0u32.to_le_bytes());
        }
        for &coord in scaled_coords {
            record.extend_from_slice(&coord.to_le_bytes());
        }
        Ok(record)
    }

    fn decode_btree_v2_scaled_coords(
        record: &[u8],
        filtered: bool,
        chunk_size_len: usize,
        sizeof_addr: usize,
        ndims: usize,
    ) -> Result<Vec<u64>> {
        let mut pos = sizeof_addr;
        if record.len() < pos {
            return Err(Error::InvalidFormat(
                "truncated v2 B-tree chunk address".into(),
            ));
        }
        if filtered {
            pos = pos
                .checked_add(chunk_size_len)
                .and_then(|value| value.checked_add(4))
                .ok_or_else(|| Error::InvalidFormat("v2 B-tree record offset overflow".into()))?;
        }
        let coords_end = pos
            .checked_add(
                ndims.checked_mul(8).ok_or_else(|| {
                    Error::InvalidFormat("v2 B-tree record offset overflow".into())
                })?,
            )
            .ok_or_else(|| Error::InvalidFormat("v2 B-tree record offset overflow".into()))?;
        if record.len() < coords_end {
            return Err(Error::InvalidFormat(
                "truncated v2 B-tree scaled chunk coordinates".into(),
            ));
        }

        let mut coords = Vec::with_capacity(ndims);
        for _ in 0..ndims {
            let next = pos
                .checked_add(8)
                .ok_or_else(|| Error::InvalidFormat("v2 B-tree record offset overflow".into()))?;
            coords.push(Self::read_le_uint(&record[pos..next])?);
            pos = next;
        }
        Ok(coords)
    }

    fn scaled_chunk_coords(chunk_coords: &[u64], chunk_dims: &[u64]) -> Result<Vec<u64>> {
        if chunk_coords.len() != chunk_dims.len() {
            return Err(Error::InvalidFormat(
                "chunk coordinate rank does not match chunk dimensions".into(),
            ));
        }
        chunk_coords
            .iter()
            .zip(chunk_dims)
            .map(|(&coord, &dim)| {
                if dim == 0 {
                    return Err(Error::InvalidFormat("chunk dimension is zero".into()));
                }
                if coord % dim != 0 {
                    return Err(Error::InvalidFormat(
                        "chunk coordinate is not aligned to chunk dimension".into(),
                    ));
                }
                Ok(coord / dim)
            })
            .collect()
    }

    fn read_le_uint(bytes: &[u8]) -> Result<u64> {
        if bytes.len() > 8 {
            return Err(Error::InvalidFormat(
                "little-endian integer is wider than u64".into(),
            ));
        }
        let mut value = 0u64;
        for (idx, byte) in bytes.iter().enumerate() {
            value |= u64::from(*byte) << (idx * 8);
        }
        Ok(value)
    }

    fn checked_header_relative_offset(pos: u64, base: u64) -> Result<usize> {
        let delta = pos.checked_sub(base).ok_or_else(|| {
            Error::InvalidFormat("v2 B-tree header field precedes header base".into())
        })?;
        usize::try_from(delta)
            .map_err(|_| Error::InvalidFormat("v2 B-tree header field offset is too large".into()))
    }

    fn header_window_mut(buf: &mut [u8], offset: usize, len: usize) -> Result<&mut [u8]> {
        let end = offset
            .checked_add(len)
            .ok_or_else(|| Error::InvalidFormat("v2 B-tree header field offset overflow".into()))?;
        buf.get_mut(offset..end).ok_or_else(|| {
            Error::InvalidFormat("v2 B-tree header field exceeds checksum span".into())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::MutableFile;
    use crate::format::btree_v2::BTreeV2Header;

    fn test_header(record_size: u16) -> BTreeV2Header {
        BTreeV2Header {
            tree_type: 10,
            node_size: 512,
            record_size,
            depth: 0,
            split_pct: 100,
            merge_pct: 40,
            root_addr: 0,
            root_nrecords: 0,
            total_records: 0,
        }
    }

    #[test]
    fn btree_v2_chunk_record_size_rejects_coord_overflow() {
        let result = MutableFile::btree_v2_chunk_record_size(8, false, 0, usize::MAX / 8 + 1);
        assert!(result.is_err());
    }

    #[test]
    fn decode_btree_v2_scaled_coords_rejects_coordinate_span_overflow() {
        let result =
            MutableFile::decode_btree_v2_scaled_coords(&[0; 8], false, 0, 0, usize::MAX / 8 + 1);
        assert!(result.is_err());
    }

    #[test]
    fn decode_btree_v2_scaled_coords_rejects_truncated_coordinates() {
        let result = MutableFile::decode_btree_v2_scaled_coords(&[0; 7], false, 0, 0, 1);
        assert!(result.is_err());
    }

    #[test]
    fn encode_btree_v2_leaf_rejects_wrong_record_size() {
        let header = test_header(8);
        let result = MutableFile::encode_btree_v2_leaf(&header, &[vec![0; 7]]);
        assert!(result.is_err());
    }

    #[test]
    fn btree_v2_header_window_rejects_bad_offsets() {
        let err = MutableFile::checked_header_relative_offset(9, 10).unwrap_err();
        assert!(
            err.to_string()
                .contains("v2 B-tree header field precedes header base"),
            "unexpected error: {err}"
        );

        let mut buf = vec![0u8; 4];
        let err = MutableFile::header_window_mut(&mut buf, 3, 2).unwrap_err();
        assert!(
            err.to_string()
                .contains("v2 B-tree header field exceeds checksum span"),
            "unexpected error: {err}"
        );
    }
}
