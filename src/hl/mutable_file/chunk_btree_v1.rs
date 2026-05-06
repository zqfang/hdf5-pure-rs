use std::io::{Seek, SeekFrom, Write};

use crate::error::{Error, Result};
use crate::io::reader::HdfReader;

use super::MutableFile;

#[derive(Debug, Clone)]
struct ChunkBTreeEntry {
    coords: Vec<u64>,
    chunk_size: u32,
    filter_mask: u32,
    child_addr: u64,
}

const CHUNK_BTREE_MAX_LEAF_ENTRIES: usize = 64;

impl MutableFile {
    pub(super) fn rewrite_leaf_chunk_btree(
        &mut self,
        btree_addr: u64,
        chunk_coords: &[u64],
        chunk_size: u32,
        chunk_addr: u64,
        element_size: u32,
    ) -> Result<()> {
        let ndims = chunk_coords.len();
        let sa = usize::from(self.superblock.sizeof_addr);

        let mut guard = self.inner.lock();
        guard.reader.seek(btree_addr)?;
        if guard.reader.read_bytes(4)? != [b'T', b'R', b'E', b'E'] {
            return Err(Error::InvalidFormat("invalid chunk B-tree magic".into()));
        }
        let node_type = guard.reader.read_u8()?;
        let level = guard.reader.read_u8()?;
        let entries_used = usize::from(guard.reader.read_u16()?);
        if node_type != 1 {
            return Err(Error::InvalidFormat(format!(
                "expected raw-data chunk B-tree, got type {node_type}"
            )));
        }
        if level != 0 {
            drop(guard);
            let mut entries = self.collect_chunk_btree_entries(btree_addr, ndims)?;
            if let Some(entry) = entries
                .iter_mut()
                .find(|entry| entry.coords.as_slice() == chunk_coords)
            {
                entry.chunk_size = chunk_size;
                entry.filter_mask = 0;
                entry.child_addr = chunk_addr;
            } else {
                entries.push(ChunkBTreeEntry {
                    coords: chunk_coords.to_vec(),
                    chunk_size,
                    filter_mask: 0,
                    child_addr: chunk_addr,
                });
            }
            entries.sort_by(|a, b| a.coords.cmp(&b.coords));
            self.rebuild_chunk_btree_from_entries(btree_addr, &entries, element_size, sa)?;
            return Ok(());
        }
        let entries = Self::read_chunk_btree_leaf_entries(
            &mut guard.reader,
            btree_addr,
            ndims,
            entries_used,
            sa,
        )?;
        drop(guard);

        if let Some((entry_pos, _)) =
            Self::find_chunk_btree_entry_position(&entries, btree_addr, ndims, sa, chunk_coords)?
        {
            self.write_btree_entry(entry_pos, chunk_coords, chunk_size, chunk_addr, sa)?;
            return Ok(());
        }

        if entries_used >= CHUNK_BTREE_MAX_LEAF_ENTRIES {
            let mut entries = entries;
            entries.push(ChunkBTreeEntry {
                coords: chunk_coords.to_vec(),
                chunk_size,
                filter_mask: 0,
                child_addr: chunk_addr,
            });
            entries.sort_by(|a, b| a.coords.cmp(&b.coords));
            self.rebuild_chunk_btree_from_entries(btree_addr, &entries, element_size, sa)?;
            return Ok(());
        }

        self.append_chunk_btree_leaf_entry(
            btree_addr,
            ndims,
            entries_used,
            chunk_coords,
            chunk_size,
            chunk_addr,
            element_size,
            sa,
        )?;

        Ok(())
    }

    fn chunk_btree_checked_usize_add(lhs: usize, rhs: usize, context: &str) -> Result<usize> {
        lhs.checked_add(rhs)
            .ok_or_else(|| Error::InvalidFormat(format!("{context} overflow")))
    }

    fn chunk_btree_checked_usize_mul(lhs: usize, rhs: usize, context: &str) -> Result<usize> {
        lhs.checked_mul(rhs)
            .ok_or_else(|| Error::InvalidFormat(format!("{context} overflow")))
    }

    fn chunk_btree_checked_u64_add(lhs: u64, rhs: u64, context: &str) -> Result<u64> {
        lhs.checked_add(rhs)
            .ok_or_else(|| Error::InvalidFormat(format!("{context} overflow")))
    }

    fn chunk_btree_u64_from_usize(value: usize, context: &str) -> Result<u64> {
        u64::try_from(value)
            .map_err(|_| Error::InvalidFormat(format!("{context} does not fit in u64")))
    }

    fn chunk_btree_key_size(ndims: usize) -> Result<usize> {
        let coord_words = Self::chunk_btree_checked_usize_add(ndims, 1, "chunk B-tree key size")?;
        let coord_bytes =
            Self::chunk_btree_checked_usize_mul(coord_words, 8, "chunk B-tree key size")?;
        Self::chunk_btree_checked_usize_add(4 + 4, coord_bytes, "chunk B-tree key size")
    }

    fn chunk_btree_header_size(sizeof_addr: usize) -> Result<usize> {
        let sibling_bytes =
            Self::chunk_btree_checked_usize_mul(sizeof_addr, 2, "chunk B-tree node header size")?;
        Self::chunk_btree_checked_usize_add(
            4 + 1 + 1 + 2,
            sibling_bytes,
            "chunk B-tree node header size",
        )
    }

    fn chunk_btree_leaf_layout(
        node_addr: u64,
        ndims: usize,
        sizeof_addr: usize,
    ) -> Result<(u64, usize)> {
        let key_size = Self::chunk_btree_key_size(ndims)?;
        let entry_size =
            Self::chunk_btree_checked_usize_add(key_size, sizeof_addr, "chunk B-tree entry size")?;
        let header_size = Self::chunk_btree_header_size(sizeof_addr)?;
        let entries_start = Self::chunk_btree_checked_u64_add(
            node_addr,
            Self::chunk_btree_u64_from_usize(header_size, "chunk B-tree node header size")?,
            "chunk B-tree entries start",
        )?;
        Ok((entries_start, entry_size))
    }

    fn chunk_btree_entry_pos(
        entries_start: u64,
        entry_idx: usize,
        entry_size: usize,
    ) -> Result<u64> {
        let entry_offset = Self::chunk_btree_checked_usize_mul(
            entry_idx,
            entry_size,
            "chunk B-tree entry offset",
        )?;
        Self::chunk_btree_checked_u64_add(
            entries_start,
            Self::chunk_btree_u64_from_usize(entry_offset, "chunk B-tree entry offset")?,
            "chunk B-tree entry position",
        )
    }

    fn read_chunk_btree_leaf_entries<R: std::io::Read + Seek>(
        reader: &mut HdfReader<R>,
        node_addr: u64,
        ndims: usize,
        entries_used: usize,
        sizeof_addr: usize,
    ) -> Result<Vec<ChunkBTreeEntry>> {
        let (entries_start, entry_size) =
            Self::chunk_btree_leaf_layout(node_addr, ndims, sizeof_addr)?;
        let mut entries = Vec::with_capacity(Self::chunk_btree_checked_usize_add(
            entries_used,
            1,
            "chunk B-tree entries",
        )?);
        for entry_idx in 0..entries_used {
            let key_pos = Self::chunk_btree_entry_pos(entries_start, entry_idx, entry_size)?;
            reader.seek(key_pos)?;
            let chunk_size = reader.read_u32()?;
            let filter_mask = reader.read_u32()?;
            let mut coords = Vec::with_capacity(ndims);
            for _ in 0..ndims {
                coords.push(reader.read_u64()?);
            }
            let _extra = reader.read_u64()?;
            let child_addr = reader.read_addr()?;
            entries.push(ChunkBTreeEntry {
                coords,
                chunk_size,
                filter_mask,
                child_addr,
            });
        }
        Ok(entries)
    }

    fn find_chunk_btree_entry_position(
        entries: &[ChunkBTreeEntry],
        node_addr: u64,
        ndims: usize,
        sizeof_addr: usize,
        chunk_coords: &[u64],
    ) -> Result<Option<(u64, usize)>> {
        let (entries_start, entry_size) =
            Self::chunk_btree_leaf_layout(node_addr, ndims, sizeof_addr)?;
        let Some(entry_idx) = entries
            .iter()
            .position(|entry| entry.coords.as_slice() == chunk_coords)
        else {
            return Ok(None);
        };
        Ok(Some((
            Self::chunk_btree_entry_pos(entries_start, entry_idx, entry_size)?,
            entry_idx,
        )))
    }

    fn append_chunk_btree_leaf_entry(
        &mut self,
        node_addr: u64,
        ndims: usize,
        entries_used: usize,
        chunk_coords: &[u64],
        chunk_size: u32,
        chunk_addr: u64,
        element_size: u32,
        sizeof_addr: usize,
    ) -> Result<()> {
        let (entries_start, entry_size) =
            Self::chunk_btree_leaf_layout(node_addr, ndims, sizeof_addr)?;
        self.write_handle
            .seek(SeekFrom::Start(Self::chunk_btree_checked_u64_add(
                node_addr,
                6,
                "chunk B-tree entry-count field",
            )?))?;
        self.write_handle.write_all(
            &(u16::try_from(Self::chunk_btree_checked_usize_add(
                entries_used,
                1,
                "chunk B-tree entry count",
            )?)
            .map_err(|_| Error::InvalidFormat("chunk B-tree entry count exceeds u16".into()))?)
            .to_le_bytes(),
        )?;

        let new_entry_pos = Self::chunk_btree_entry_pos(entries_start, entries_used, entry_size)?;
        self.write_btree_entry(
            new_entry_pos,
            chunk_coords,
            chunk_size,
            chunk_addr,
            sizeof_addr,
        )?;

        let final_key_pos = Self::chunk_btree_entry_pos(
            entries_start,
            Self::chunk_btree_checked_usize_add(entries_used, 1, "chunk B-tree final key index")?,
            entry_size,
        )?;
        self.write_btree_final_key(final_key_pos, chunk_coords, element_size)?;
        Ok(())
    }

    fn collect_chunk_btree_entries(
        &mut self,
        node_addr: u64,
        ndims: usize,
    ) -> Result<Vec<ChunkBTreeEntry>> {
        let mut guard = self.inner.lock();
        Self::collect_chunk_btree_entries_with_reader(&mut guard.reader, node_addr, ndims, None)
    }

    fn collect_chunk_btree_entries_with_reader<R: std::io::Read + Seek>(
        reader: &mut HdfReader<R>,
        node_addr: u64,
        ndims: usize,
        expected_level: Option<u8>,
    ) -> Result<Vec<ChunkBTreeEntry>> {
        reader.seek(node_addr)?;
        if reader.read_bytes(4)? != [b'T', b'R', b'E', b'E'] {
            return Err(Error::InvalidFormat("invalid chunk B-tree magic".into()));
        }
        let node_type = reader.read_u8()?;
        if node_type != 1 {
            return Err(Error::InvalidFormat(format!(
                "expected raw-data chunk B-tree, got type {node_type}"
            )));
        }
        let level = reader.read_u8()?;
        if let Some(expected_level) = expected_level {
            if level != expected_level {
                return Err(Error::InvalidFormat(format!(
                    "chunk B-tree child level {level} does not match expected {expected_level}"
                )));
            }
        }
        let entries_used = usize::from(reader.read_u16()?);
        if entries_used > CHUNK_BTREE_MAX_LEAF_ENTRIES {
            return Err(Error::InvalidFormat(format!(
                "chunk B-tree node entry count {entries_used} exceeds v1 node capacity"
            )));
        }
        let _left_sibling = reader.read_addr()?;
        let _right_sibling = reader.read_addr()?;

        if level == 0 {
            let mut entries = Vec::with_capacity(entries_used);
            for _ in 0..entries_used {
                let chunk_size = reader.read_u32()?;
                let filter_mask = reader.read_u32()?;
                let mut coords = Vec::with_capacity(ndims);
                for _ in 0..ndims {
                    coords.push(reader.read_u64()?);
                }
                let _extra = reader.read_u64()?;
                let child_addr = reader.read_addr()?;
                entries.push(ChunkBTreeEntry {
                    coords,
                    chunk_size,
                    filter_mask,
                    child_addr,
                });
            }
            Ok(entries)
        } else {
            if level > 1 {
                return Err(Error::Unsupported(
                    "write_chunk cannot collect v1 chunk B-trees deeper than two levels".into(),
                ));
            }
            let mut child_addrs = Vec::with_capacity(entries_used);
            for _ in 0..entries_used {
                let _chunk_size = reader.read_u32()?;
                let _filter_mask = reader.read_u32()?;
                let key_words = Self::chunk_btree_checked_usize_add(
                    ndims,
                    1,
                    "chunk B-tree child key coordinate count",
                )?;
                for _ in 0..key_words {
                    let _ = reader.read_u64()?;
                }
                child_addrs.push(reader.read_addr()?);
            }

            let mut entries = Vec::new();
            for child_addr in child_addrs {
                if crate::io::reader::is_undef_addr(child_addr) {
                    return Err(Error::InvalidFormat(
                        "chunk B-tree child address is undefined".into(),
                    ));
                }
                let child_entries = Self::collect_chunk_btree_entries_with_reader(
                    reader,
                    child_addr,
                    ndims,
                    Some(level - 1),
                )?;
                let new_len = Self::chunk_btree_checked_usize_add(
                    entries.len(),
                    child_entries.len(),
                    "chunk B-tree collected entry count",
                )?;
                if new_len > CHUNK_BTREE_MAX_LEAF_ENTRIES * CHUNK_BTREE_MAX_LEAF_ENTRIES {
                    return Err(Error::Unsupported(
                        "write_chunk cannot collect v1 chunk B-trees beyond a two-level root"
                            .into(),
                    ));
                }
                entries.reserve(child_entries.len());
                entries.extend(child_entries);
            }
            Ok(entries)
        }
    }

    fn rebuild_chunk_btree_from_entries(
        &mut self,
        root_addr: u64,
        entries: &[ChunkBTreeEntry],
        element_size: u32,
        sizeof_addr: usize,
    ) -> Result<()> {
        let ndims = entries
            .first()
            .map(|entry| entry.coords.len())
            .ok_or_else(|| Error::InvalidFormat("cannot rebuild empty chunk B-tree".into()))?;
        let node_size = Self::chunk_btree_node_size(ndims, sizeof_addr)?;

        if entries.len() <= 64 {
            self.write_chunk_btree_node(root_addr, 0, entries, element_size, sizeof_addr)?;
            return Ok(());
        }

        let leaf_count = entries.len().div_ceil(64);
        if leaf_count > 64 {
            return Err(Error::Unsupported(
                "write_chunk cannot grow v1 chunk B-tree beyond a two-level root".into(),
            ));
        }

        let mut root_entries = Vec::with_capacity(leaf_count);
        for leaf_entries in entries.chunks(64) {
            let leaf_addr = self.append_aligned_zeros(node_size, 8)?;
            self.write_chunk_btree_node(leaf_addr, 0, leaf_entries, element_size, sizeof_addr)?;
            root_entries.push(ChunkBTreeEntry {
                coords: leaf_entries[0].coords.clone(),
                chunk_size: leaf_entries[0].chunk_size,
                filter_mask: leaf_entries[0].filter_mask,
                child_addr: leaf_addr,
            });
        }
        self.write_chunk_btree_node(root_addr, 1, &root_entries, element_size, sizeof_addr)?;

        Ok(())
    }

    fn chunk_btree_node_size(ndims: usize, sizeof_addr: usize) -> Result<usize> {
        let key_size = Self::chunk_btree_key_size(ndims)?;
        let max_entries = 64usize;
        let header_size = Self::chunk_btree_header_size(sizeof_addr)?;
        let key_bytes = Self::chunk_btree_checked_usize_mul(
            Self::chunk_btree_checked_usize_add(max_entries, 1, "chunk B-tree node size")?,
            key_size,
            "chunk B-tree node size",
        )?;
        let addr_bytes = Self::chunk_btree_checked_usize_mul(
            max_entries,
            sizeof_addr,
            "chunk B-tree node size",
        )?;
        Self::chunk_btree_checked_usize_add(header_size, key_bytes, "chunk B-tree node size")
            .and_then(|value| {
                Self::chunk_btree_checked_usize_add(value, addr_bytes, "chunk B-tree node size")
            })
    }

    /// Pure encoder for a v1 chunk-index B-tree node (TREE magic, header,
    /// up to 64 (key, child) pairs, trailing key, zero-padded to
    /// `chunk_btree_node_size`). Mirrors libhdf5's `H5B__cache_serialize`
    /// for chunk-index B-trees.
    fn encode_chunk_btree_node(
        &self,
        level: u8,
        entries: &[ChunkBTreeEntry],
        element_size: u32,
        sizeof_addr: usize,
    ) -> Result<Vec<u8>> {
        if entries.len() > 64 {
            return Err(Error::InvalidFormat(
                "chunk B-tree node entry count exceeds v1 node capacity".into(),
            ));
        }
        let ndims = entries
            .first()
            .map(|entry| entry.coords.len())
            .ok_or_else(|| Error::InvalidFormat("cannot write empty chunk B-tree node".into()))?;
        let node_size = Self::chunk_btree_node_size(ndims, sizeof_addr)?;
        let mut buf = Vec::with_capacity(node_size);
        buf.extend_from_slice(b"TREE");
        buf.push(1);
        buf.push(level);
        buf.extend_from_slice(
            &Self::usize_to_u16(entries.len(), "chunk B-tree entry count")?.to_le_bytes(),
        );
        let undef = Self::undefined_addr_bytes(sizeof_addr, "chunk B-tree sibling address")?;
        buf.extend_from_slice(&undef);
        buf.extend_from_slice(&undef);

        for entry in entries {
            buf.extend_from_slice(&entry.chunk_size.to_le_bytes());
            buf.extend_from_slice(&entry.filter_mask.to_le_bytes());
            for &coord in &entry.coords {
                buf.extend_from_slice(&coord.to_le_bytes());
            }
            buf.extend_from_slice(&0u64.to_le_bytes());
            buf.extend_from_slice(&Self::encode_uint_le(
                entry.child_addr,
                sizeof_addr,
                "chunk B-tree child address",
            )?);
        }

        let final_coords = &entries[entries.len() - 1].coords;
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        for &coord in final_coords {
            buf.extend_from_slice(&coord.to_le_bytes());
        }
        buf.extend_from_slice(&u64::from(element_size).to_le_bytes());
        buf.resize(node_size, 0);
        Ok(buf)
    }

    /// Encode + write a v1 chunk-index B-tree node.
    fn write_chunk_btree_node(
        &mut self,
        pos: u64,
        level: u8,
        entries: &[ChunkBTreeEntry],
        element_size: u32,
        sizeof_addr: usize,
    ) -> Result<()> {
        let buf = self.encode_chunk_btree_node(level, entries, element_size, sizeof_addr)?;
        self.write_handle.seek(SeekFrom::Start(pos))?;
        self.write_handle.write_all(&buf)?;
        Ok(())
    }

    fn write_btree_entry(
        &mut self,
        pos: u64,
        chunk_coords: &[u64],
        chunk_size: u32,
        chunk_addr: u64,
        sizeof_addr: usize,
    ) -> Result<()> {
        self.write_handle.seek(SeekFrom::Start(pos))?;
        self.write_handle.write_all(&chunk_size.to_le_bytes())?;
        self.write_handle.write_all(&0u32.to_le_bytes())?;
        for &coord in chunk_coords {
            self.write_handle.write_all(&coord.to_le_bytes())?;
        }
        self.write_handle.write_all(&0u64.to_le_bytes())?;
        self.write_handle.write_all(&Self::encode_uint_le(
            chunk_addr,
            sizeof_addr,
            "chunk B-tree child address",
        )?)?;

        Ok(())
    }

    fn write_btree_final_key(
        &mut self,
        pos: u64,
        chunk_coords: &[u64],
        element_size: u32,
    ) -> Result<()> {
        self.write_handle.seek(SeekFrom::Start(pos))?;
        self.write_handle.write_all(&0u32.to_le_bytes())?;
        self.write_handle.write_all(&0u32.to_le_bytes())?;
        for &coord in chunk_coords {
            self.write_handle.write_all(&coord.to_le_bytes())?;
        }
        self.write_handle
            .write_all(&u64::from(element_size).to_le_bytes())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use crate::io::HdfReader;

    use super::MutableFile;

    #[test]
    fn chunk_btree_key_size_rejects_dimension_overflow() {
        let err = MutableFile::chunk_btree_key_size(usize::MAX).unwrap_err();
        assert!(err.to_string().contains("chunk B-tree key size"));
    }

    #[test]
    fn chunk_btree_leaf_layout_rejects_address_overflow() {
        let err = MutableFile::chunk_btree_leaf_layout(u64::MAX, 1, 8).unwrap_err();
        assert!(err.to_string().contains("entries start"));
    }

    #[test]
    fn chunk_btree_entry_pos_rejects_offset_overflow() {
        let err = MutableFile::chunk_btree_entry_pos(0, usize::MAX, 2).unwrap_err();
        assert!(err.to_string().contains("entry offset"));
    }

    #[test]
    fn chunk_btree_child_key_count_rejects_dimension_overflow() {
        let err = MutableFile::chunk_btree_checked_usize_add(
            usize::MAX,
            1,
            "chunk B-tree child key coordinate count",
        )
        .unwrap_err();
        assert!(err.to_string().contains("child key coordinate count"));
    }

    #[test]
    fn chunk_btree_collect_rejects_child_level_mismatch() {
        fn leaf_node(level: u8, addr: u64) -> Vec<u8> {
            let mut node = Vec::new();
            node.extend_from_slice(b"TREE");
            node.push(1);
            node.push(level);
            node.extend_from_slice(&1u16.to_le_bytes());
            node.extend_from_slice(&u64::MAX.to_le_bytes());
            node.extend_from_slice(&u64::MAX.to_le_bytes());
            node.extend_from_slice(&16u32.to_le_bytes());
            node.extend_from_slice(&0u32.to_le_bytes());
            node.extend_from_slice(&0u64.to_le_bytes());
            node.extend_from_slice(&0u64.to_le_bytes());
            node.extend_from_slice(&addr.to_le_bytes());
            node.extend_from_slice(&0u32.to_le_bytes());
            node.extend_from_slice(&0u32.to_le_bytes());
            node.extend_from_slice(&0u64.to_le_bytes());
            node.extend_from_slice(&0u64.to_le_bytes());
            node
        }

        let mut root = Vec::new();
        root.extend_from_slice(b"TREE");
        root.push(1);
        root.push(1);
        root.extend_from_slice(&1u16.to_le_bytes());
        root.extend_from_slice(&u64::MAX.to_le_bytes());
        root.extend_from_slice(&u64::MAX.to_le_bytes());
        root.extend_from_slice(&0u32.to_le_bytes());
        root.extend_from_slice(&0u32.to_le_bytes());
        root.extend_from_slice(&0u64.to_le_bytes());
        root.extend_from_slice(&0u64.to_le_bytes());
        root.extend_from_slice(&128u64.to_le_bytes());
        root.extend_from_slice(&0u32.to_le_bytes());
        root.extend_from_slice(&0u32.to_le_bytes());
        root.extend_from_slice(&0u64.to_le_bytes());
        root.extend_from_slice(&0u64.to_le_bytes());

        let mut file = root;
        file.resize(128, 0);
        file.extend_from_slice(&leaf_node(1, 256)); // should be level 0 under root level 1.

        let mut reader = HdfReader::new(Cursor::new(file));
        reader.set_sizeof_addr(8);
        let err = MutableFile::collect_chunk_btree_entries_with_reader(&mut reader, 0, 1, None)
            .expect_err("child level mismatch should fail");
        assert!(err.to_string().contains("child level"));
    }
}
