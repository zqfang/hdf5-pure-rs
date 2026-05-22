use std::cmp::Ordering;
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
        chunk_dims: &[u64],
        chunk_size: u32,
        chunk_addr: u64,
        element_size: u32,
    ) -> Result<()> {
        let ndims = chunk_coords.len();
        let sa = usize::from(self.superblock.sizeof_addr);

        let mut guard = self.inner.lock();
        guard.reader.seek(btree_addr)?;
        let mut magic = [0u8; 4];
        guard.reader.read_bytes_into(&mut magic)?;
        if magic != [b'T', b'R', b'E', b'E'] {
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
            let existing_entry_pos = Self::locate_chunk_btree_entry_position(
                &mut guard.reader,
                btree_addr,
                ndims,
                sa,
                chunk_coords,
                Some(level),
            )?;
            drop(guard);
            if let Some(entry_pos) = existing_entry_pos {
                self.write_btree_entry(entry_pos, chunk_coords, chunk_size, chunk_addr, sa)?;
                return Ok(());
            }

            let mut entries = self.collect_chunk_btree_entries(btree_addr, ndims)?;
            match Self::find_chunk_btree_entry_index(&entries, chunk_coords) {
                Ok(index) => {
                    let entry = &mut entries[index];
                    entry.chunk_size = chunk_size;
                    entry.filter_mask = 0;
                    entry.child_addr = chunk_addr;
                }
                Err(index) => entries.insert(
                    index,
                    ChunkBTreeEntry {
                        coords: chunk_coords.to_vec(),
                        chunk_size,
                        filter_mask: 0,
                        child_addr: chunk_addr,
                    },
                ),
            }
            self.rebuild_chunk_btree_from_entries(
                btree_addr,
                entries,
                chunk_dims,
                element_size,
                sa,
            )?;
            return Ok(());
        }
        let existing_entry_pos = Self::locate_chunk_btree_leaf_entry_position(
            &mut guard.reader,
            btree_addr,
            ndims,
            entries_used,
            sa,
            chunk_coords,
        )?;
        if let Some(entry_pos) = existing_entry_pos {
            drop(guard);
            self.write_btree_entry(entry_pos, chunk_coords, chunk_size, chunk_addr, sa)?;
            return Ok(());
        }
        let mut entries = Self::read_chunk_btree_leaf_entries(
            &mut guard.reader,
            btree_addr,
            ndims,
            entries_used,
            sa,
        )?;
        let entry_lookup = Self::find_chunk_btree_entry_index(&entries, chunk_coords);
        drop(guard);

        match entry_lookup {
            Ok(entry_idx) => {
                let (entries_start, entry_size) =
                    Self::chunk_btree_leaf_layout(btree_addr, ndims, sa)?;
                let entry_pos = Self::chunk_btree_entry_pos(entries_start, entry_idx, entry_size)?;
                self.write_btree_entry(entry_pos, chunk_coords, chunk_size, chunk_addr, sa)?;
                return Ok(());
            }
            Err(insert_idx) if entries_used < CHUNK_BTREE_MAX_LEAF_ENTRIES => {
                entries.insert(
                    insert_idx,
                    ChunkBTreeEntry {
                        coords: chunk_coords.to_vec(),
                        chunk_size,
                        filter_mask: 0,
                        child_addr: chunk_addr,
                    },
                );
                self.write_chunk_btree_node(btree_addr, 0, &entries, chunk_dims, element_size, sa)?;
                return Ok(());
            }
            Err(insert_idx) => {
                entries.insert(
                    insert_idx,
                    ChunkBTreeEntry {
                        coords: chunk_coords.to_vec(),
                        chunk_size,
                        filter_mask: 0,
                        child_addr: chunk_addr,
                    },
                );
                self.rebuild_chunk_btree_from_entries(
                    btree_addr,
                    entries,
                    chunk_dims,
                    element_size,
                    sa,
                )?;
                return Ok(());
            }
        }
    }

    fn find_chunk_btree_entry_index(
        entries: &[ChunkBTreeEntry],
        chunk_coords: &[u64],
    ) -> std::result::Result<usize, usize> {
        entries.binary_search_by(|entry| entry.coords.as_slice().cmp(chunk_coords))
    }

    fn locate_chunk_btree_entry_position<R: std::io::Read + Seek>(
        reader: &mut HdfReader<R>,
        node_addr: u64,
        ndims: usize,
        sizeof_addr: usize,
        chunk_coords: &[u64],
        expected_level: Option<u8>,
    ) -> Result<Option<u64>> {
        reader.seek(node_addr)?;
        let mut magic = [0u8; 4];
        reader.read_bytes_into(&mut magic)?;
        if magic != [b'T', b'R', b'E', b'E'] {
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
            return Self::locate_chunk_btree_leaf_entry_position(
                reader,
                node_addr,
                ndims,
                entries_used,
                sizeof_addr,
                chunk_coords,
            );
        }

        let mut child_addr = None;
        for _ in 0..entries_used {
            let _chunk_size = reader.read_u32()?;
            let _filter_mask = reader.read_u32()?;
            let key_order = Self::read_chunk_btree_coords_cmp(reader, ndims, chunk_coords)?;
            let _extra = reader.read_u64()?;
            let entry_child_addr = reader.read_addr()?;
            if key_order != Ordering::Greater {
                child_addr = Some(entry_child_addr);
            }
        }

        let Some(child_addr) = child_addr else {
            return Ok(None);
        };
        if crate::io::reader::is_undef_addr(child_addr) {
            return Err(Error::InvalidFormat(
                "chunk B-tree child address is undefined".into(),
            ));
        }
        Self::locate_chunk_btree_entry_position(
            reader,
            child_addr,
            ndims,
            sizeof_addr,
            chunk_coords,
            Some(level - 1),
        )
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

    fn read_chunk_btree_coords_cmp<R: std::io::Read + Seek>(
        reader: &mut HdfReader<R>,
        ndims: usize,
        chunk_coords: &[u64],
    ) -> Result<Ordering> {
        let mut ordering = Ordering::Equal;
        for dim_idx in 0..ndims {
            let coord = reader.read_u64()?;
            if ordering == Ordering::Equal {
                ordering = match chunk_coords.get(dim_idx) {
                    Some(target) => coord.cmp(target),
                    None => Ordering::Greater,
                };
            }
        }
        if ordering == Ordering::Equal && chunk_coords.len() > ndims {
            Ok(Ordering::Less)
        } else {
            Ok(ordering)
        }
    }

    fn locate_chunk_btree_leaf_entry_position<R: std::io::Read + Seek>(
        reader: &mut HdfReader<R>,
        node_addr: u64,
        ndims: usize,
        entries_used: usize,
        sizeof_addr: usize,
        chunk_coords: &[u64],
    ) -> Result<Option<u64>> {
        let (entries_start, entry_size) =
            Self::chunk_btree_leaf_layout(node_addr, ndims, sizeof_addr)?;
        for entry_idx in 0..entries_used {
            let key_pos = Self::chunk_btree_entry_pos(entries_start, entry_idx, entry_size)?;
            reader.seek(key_pos)?;
            let _chunk_size = reader.read_u32()?;
            let _filter_mask = reader.read_u32()?;
            let matches =
                Self::read_chunk_btree_coords_cmp(reader, ndims, chunk_coords)? == Ordering::Equal;
            let _extra = reader.read_u64()?;
            let _child_addr = reader.read_addr()?;
            if matches {
                return Ok(Some(key_pos));
            }
        }
        Ok(None)
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
        let mut magic = [0u8; 4];
        reader.read_bytes_into(&mut magic)?;
        if magic != [b'T', b'R', b'E', b'E'] {
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
            let mut entries = Vec::new();
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
                let child_addr = reader.read_addr()?;
                if crate::io::reader::is_undef_addr(child_addr) {
                    return Err(Error::InvalidFormat(
                        "chunk B-tree child address is undefined".into(),
                    ));
                }
                let next_entry_pos = reader.position()?;
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
                reader.seek(next_entry_pos)?;
            }
            Ok(entries)
        }
    }

    fn rebuild_chunk_btree_from_entries(
        &mut self,
        root_addr: u64,
        mut entries: Vec<ChunkBTreeEntry>,
        chunk_dims: &[u64],
        element_size: u32,
        sizeof_addr: usize,
    ) -> Result<()> {
        let ndims = entries
            .first()
            .map(|entry| entry.coords.len())
            .ok_or_else(|| Error::InvalidFormat("cannot rebuild empty chunk B-tree".into()))?;
        let node_size = Self::chunk_btree_node_size(ndims, sizeof_addr)?;

        if entries.len() <= 64 {
            let mut scratch = Vec::new();
            let final_key_coords = Self::chunk_btree_upper_bound_coords(
                &entries,
                chunk_dims,
                "chunk B-tree node key",
            )?;
            self.write_chunk_btree_node_with_buf(
                root_addr,
                0,
                &entries,
                &final_key_coords,
                element_size,
                sizeof_addr,
                &mut scratch,
            )?;
            return Ok(());
        }

        let leaf_count = entries.len().div_ceil(64);
        if leaf_count > 64 {
            return Err(Error::Unsupported(
                "write_chunk cannot grow v1 chunk B-tree beyond a two-level root".into(),
            ));
        }

        let base_leaf_entries = entries.len() / leaf_count;
        let extra_leaf_entries = entries.len() % leaf_count;
        let root_final_key =
            Self::chunk_btree_upper_bound_coords(&entries, chunk_dims, "chunk B-tree root key")?;
        let mut root_entries = Vec::with_capacity(leaf_count);
        let mut scratch = Vec::new();
        let mut start = 0;
        for leaf_idx in 0..leaf_count {
            let count = base_leaf_entries + usize::from(leaf_idx < extra_leaf_entries);
            let end = Self::chunk_btree_checked_usize_add(
                start,
                count,
                "chunk B-tree rebuilt leaf range",
            )?;
            let leaf_entries = entries.get_mut(start..end).ok_or_else(|| {
                Error::InvalidFormat("chunk B-tree rebuilt leaf range is out of bounds".into())
            })?;
            let leaf_final_key = Self::chunk_btree_upper_bound_coords(
                leaf_entries,
                chunk_dims,
                "chunk B-tree leaf key",
            )?;
            let leaf_physical_addr = self.append_aligned_zeros(node_size, 8)?;
            let leaf_addr =
                self.logical_addr_from_physical(leaf_physical_addr, "chunk B-tree leaf address")?;
            self.write_chunk_btree_node_with_buf(
                leaf_addr,
                0,
                leaf_entries,
                &leaf_final_key,
                element_size,
                sizeof_addr,
                &mut scratch,
            )?;
            let coords = std::mem::take(&mut leaf_entries[0].coords);
            root_entries.push(ChunkBTreeEntry {
                coords,
                chunk_size: leaf_entries[0].chunk_size,
                filter_mask: leaf_entries[0].filter_mask,
                child_addr: leaf_addr,
            });
            start = end;
        }
        self.write_chunk_btree_node_with_buf(
            root_addr,
            1,
            &root_entries,
            &root_final_key,
            element_size,
            sizeof_addr,
            &mut scratch,
        )?;

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
    fn encode_chunk_btree_node_into(
        &self,
        level: u8,
        entries: &[ChunkBTreeEntry],
        final_key_coords: &[u64],
        element_size: u32,
        sizeof_addr: usize,
        buf: &mut Vec<u8>,
    ) -> Result<()> {
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
        buf.clear();
        buf.reserve(node_size);
        buf.extend_from_slice(b"TREE");
        buf.push(1);
        buf.push(level);
        buf.extend_from_slice(
            &Self::usize_to_u16(entries.len(), "chunk B-tree entry count")?.to_le_bytes(),
        );
        let mut addr_buf = [0u8; 8];
        let undef = Self::undefined_addr_bytes_into(
            &mut addr_buf,
            sizeof_addr,
            "chunk B-tree sibling address",
        )?;
        buf.extend_from_slice(&undef);
        buf.extend_from_slice(&undef);

        for entry in entries {
            buf.extend_from_slice(&entry.chunk_size.to_le_bytes());
            buf.extend_from_slice(&entry.filter_mask.to_le_bytes());
            for &coord in &entry.coords {
                buf.extend_from_slice(&coord.to_le_bytes());
            }
            buf.extend_from_slice(&0u64.to_le_bytes());
            buf.extend_from_slice(Self::encode_uint_le_into(
                entry.child_addr,
                &mut addr_buf,
                sizeof_addr,
                "chunk B-tree child address",
            )?);
        }

        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        if final_key_coords.len() != ndims {
            return Err(Error::InvalidFormat(
                "chunk B-tree final key rank does not match entries".into(),
            ));
        }
        for &coord in final_key_coords {
            buf.extend_from_slice(&coord.to_le_bytes());
        }
        buf.extend_from_slice(&u64::from(element_size).to_le_bytes());
        buf.resize(node_size, 0);
        Ok(())
    }

    fn encode_chunk_btree_node(
        &self,
        level: u8,
        entries: &[ChunkBTreeEntry],
        final_key_coords: &[u64],
        element_size: u32,
        sizeof_addr: usize,
    ) -> Result<Vec<u8>> {
        let mut buf = Vec::new();
        self.encode_chunk_btree_node_into(
            level,
            entries,
            final_key_coords,
            element_size,
            sizeof_addr,
            &mut buf,
        )?;
        Ok(buf)
    }

    /// Encode + write a v1 chunk-index B-tree node.
    fn write_chunk_btree_node(
        &mut self,
        pos: u64,
        level: u8,
        entries: &[ChunkBTreeEntry],
        chunk_dims: &[u64],
        element_size: u32,
        sizeof_addr: usize,
    ) -> Result<()> {
        let final_key_coords =
            Self::chunk_btree_upper_bound_coords(entries, chunk_dims, "chunk B-tree node key")?;
        let buf = self.encode_chunk_btree_node(
            level,
            entries,
            &final_key_coords,
            element_size,
            sizeof_addr,
        )?;
        let pos = self.physical_addr(pos, "chunk B-tree node")?;
        self.write_handle.seek(SeekFrom::Start(pos))?;
        self.write_handle.write_all(&buf)?;
        Ok(())
    }

    fn write_chunk_btree_node_with_buf(
        &mut self,
        pos: u64,
        level: u8,
        entries: &[ChunkBTreeEntry],
        final_key_coords: &[u64],
        element_size: u32,
        sizeof_addr: usize,
        buf: &mut Vec<u8>,
    ) -> Result<()> {
        self.encode_chunk_btree_node_into(
            level,
            entries,
            final_key_coords,
            element_size,
            sizeof_addr,
            buf,
        )?;
        let pos = self.physical_addr(pos, "chunk B-tree node")?;
        self.write_handle.seek(SeekFrom::Start(pos))?;
        self.write_handle.write_all(buf)?;
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
        let pos = self.physical_addr(pos, "chunk B-tree entry")?;
        self.write_handle.seek(SeekFrom::Start(pos))?;
        self.write_handle.write_all(&chunk_size.to_le_bytes())?;
        self.write_handle.write_all(&0u32.to_le_bytes())?;
        for &coord in chunk_coords {
            self.write_handle.write_all(&coord.to_le_bytes())?;
        }
        self.write_handle.write_all(&0u64.to_le_bytes())?;
        let mut addr_buf = [0u8; 8];
        let addr_bytes = Self::encode_uint_le_into(
            chunk_addr,
            &mut addr_buf,
            sizeof_addr,
            "chunk B-tree child address",
        )?;
        self.write_handle.write_all(addr_bytes)?;

        Ok(())
    }

    fn chunk_btree_upper_bound_coords(
        entries: &[ChunkBTreeEntry],
        chunk_dims: &[u64],
        context: &str,
    ) -> Result<Vec<u64>> {
        let last = entries
            .last()
            .ok_or_else(|| Error::InvalidFormat(format!("{context} cannot be empty")))?;
        if last.coords.len() != chunk_dims.len() {
            return Err(Error::InvalidFormat(format!(
                "{context} rank {} does not match chunk rank {}",
                last.coords.len(),
                chunk_dims.len()
            )));
        }
        let mut upper = last.coords.clone();
        if let Some((coord, dim)) = upper.last_mut().zip(chunk_dims.last()) {
            *coord = coord
                .checked_add(*dim)
                .ok_or_else(|| Error::InvalidFormat(format!("{context} coordinate overflow")))?;
        }
        Ok(upper)
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
