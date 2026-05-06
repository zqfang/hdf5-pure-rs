use std::io::{Seek, SeekFrom, Write};

use crate::error::{Error, Result};
use crate::format::btree_v2::{self, BTreeV2Header};
use crate::format::checksum::checksum_metadata;
use crate::format::fractal_heap::FractalHeapHeader;
use crate::format::messages::attribute::AttributeMessage;
use crate::format::messages::attribute_info::AttributeInfoMessage;
use crate::format::object_header::{
    self, ObjectHeader, HDR_ATTR_STORE_PHASE_CHANGE, HDR_CHUNK0_SIZE_MASK, HDR_STORE_TIMES,
    HDR_V2_KNOWN_FLAGS,
};

use super::MutableFile;

#[derive(Debug, Clone)]
struct CompactAttributeMessageLocation {
    msg_type_offset: u64,
    msg_data_offset: u64,
    oh_start: u64,
    oh_check_len: usize,
    raw_data: Vec<u8>,
}

#[derive(Debug, Clone)]
struct DenseAttributeLocation {
    attr_info_addr: u64,
    heap: FractalHeapHeader,
    btree: BTreeV2Header,
    records: Vec<Vec<u8>>,
    record_index: usize,
    object_offset: u64,
    raw_data: Vec<u8>,
}

impl MutableFile {
    /// Delete a compact root attribute by name.
    ///
    /// This is an in-place object-header edit for compact attributes. Dense
    /// attribute storage requires fractal-heap/B-tree mutation and is rejected
    /// explicitly for now.
    pub fn delete_root_attr(&mut self, name: &str) -> Result<()> {
        self.delete_attr_at(self.superblock.root_addr, name)
    }

    /// Delete a compact group attribute by group path and attribute name.
    pub fn delete_group_attr(&mut self, group_path: &str, name: &str) -> Result<()> {
        let group = self.group(group_path)?;
        self.delete_attr_at(group.addr(), name)
    }

    /// Delete a compact dataset attribute by dataset path and attribute name.
    pub fn delete_dataset_attr(&mut self, dataset_path: &str, name: &str) -> Result<()> {
        let dataset = self.dataset(dataset_path)?;
        self.delete_attr_at(dataset.addr(), name)
    }

    /// Rename a compact root attribute in-place.
    ///
    /// The new name must fit in the existing compact attribute name field so
    /// the compact object-header message size remains unchanged.
    pub fn rename_root_attr(&mut self, old_name: &str, new_name: &str) -> Result<()> {
        self.rename_attr_at(self.superblock.root_addr, old_name, new_name)
    }

    /// Rename a compact group attribute in-place.
    pub fn rename_group_attr(
        &mut self,
        group_path: &str,
        old_name: &str,
        new_name: &str,
    ) -> Result<()> {
        let group = self.group(group_path)?;
        self.rename_attr_at(group.addr(), old_name, new_name)
    }

    /// Rename a compact dataset attribute in-place.
    pub fn rename_dataset_attr(
        &mut self,
        dataset_path: &str,
        old_name: &str,
        new_name: &str,
    ) -> Result<()> {
        let dataset = self.dataset(dataset_path)?;
        self.rename_attr_at(dataset.addr(), old_name, new_name)
    }

    fn delete_attr_at(&mut self, object_addr: u64, name: &str) -> Result<()> {
        match self.find_compact_attribute_message_in_oh(object_addr, name) {
            Ok(location) => {
                self.write_handle
                    .seek(SeekFrom::Start(location.msg_type_offset))?;
                self.write_handle
                    .write_all(&[object_header::MSG_NIL as u8])?;
                self.rewrite_oh_checksum(location.oh_start, location.oh_check_len)?;
            }
            Err(Error::Unsupported(message)) if message.contains("dense attributes") => {
                let location = self.find_dense_attribute_location(object_addr, name, None)?;
                self.delete_dense_attribute(location)?;
            }
            Err(err) => return Err(err),
        }
        self.write_handle.flush()?;
        self.reopen_reader()?;
        Ok(())
    }

    fn rename_attr_at(&mut self, object_addr: u64, old_name: &str, new_name: &str) -> Result<()> {
        if new_name.is_empty() {
            return Err(Error::InvalidFormat(
                "attribute name cannot be empty".into(),
            ));
        }
        if old_name == new_name {
            return Ok(());
        }
        match self.find_compact_attribute_rename_location(object_addr, old_name, new_name) {
            Ok(location) => {
                let (name_offset, name_size) =
                    Self::compact_attribute_name_field(&location.raw_data)?;
                if new_name.len() + 1 != name_size {
                    return Err(Error::Unsupported(
                        "in-place compact attribute rename cannot grow or shrink the encoded name field"
                            .into(),
                    ));
                }
                let name_offset_u64 = Self::usize_to_u64(name_offset, "attribute name offset")?;
                let file_name_offset = location
                    .msg_data_offset
                    .checked_add(name_offset_u64)
                    .ok_or_else(|| Error::InvalidFormat("attribute name offset overflow".into()))?;

                let mut encoded_name = vec![0u8; name_size];
                encoded_name[..new_name.len()].copy_from_slice(new_name.as_bytes());
                self.write_handle.seek(SeekFrom::Start(file_name_offset))?;
                self.write_handle.write_all(&encoded_name)?;
                self.rewrite_oh_checksum(location.oh_start, location.oh_check_len)?;
            }
            Err(Error::Unsupported(message)) if message.contains("dense attributes") => {
                let location =
                    self.find_dense_attribute_location(object_addr, old_name, Some(new_name))?;
                self.rename_dense_attribute(location, new_name)?;
            }
            Err(err) => return Err(err),
        }
        self.write_handle.flush()?;
        self.reopen_reader()?;
        Ok(())
    }

    fn compact_attribute_name_field(raw: &[u8]) -> Result<(usize, usize)> {
        if raw.len() < 4 {
            return Err(Error::InvalidFormat("attribute message too short".into()));
        }
        let version = raw
            .first()
            .copied()
            .ok_or_else(|| Error::InvalidFormat("attribute message too short".into()))?;
        let name_size = usize::from(read_u16_le_at(raw, 2, "attribute name size")?);
        if name_size == 0 {
            return Err(Error::InvalidFormat(
                "attribute message name length is zero".into(),
            ));
        }
        let offset: usize = match version {
            1 | 2 => 8,
            3 => 9,
            other => Err(Error::InvalidFormat(format!(
                "attribute message version {other}"
            )))?,
        };
        let end = offset
            .checked_add(name_size)
            .ok_or_else(|| Error::InvalidFormat("attribute name field overflow".into()))?;
        if end > raw.len() {
            return Err(Error::InvalidFormat(
                "attribute name field exceeds message".into(),
            ));
        }
        Ok((offset, name_size))
    }

    fn find_compact_attribute_message_in_oh(
        &self,
        oh_addr: u64,
        name: &str,
    ) -> Result<CompactAttributeMessageLocation> {
        self.find_compact_attribute_in_oh(oh_addr, name, None)
    }

    fn find_compact_attribute_rename_location(
        &self,
        oh_addr: u64,
        old_name: &str,
        new_name: &str,
    ) -> Result<CompactAttributeMessageLocation> {
        self.find_compact_attribute_in_oh(oh_addr, old_name, Some(new_name))
    }

    fn find_compact_attribute_in_oh(
        &self,
        oh_addr: u64,
        target_name: &str,
        reject_duplicate_name: Option<&str>,
    ) -> Result<CompactAttributeMessageLocation> {
        let mut guard = self.inner.lock();
        let reader = &mut guard.reader;
        reader.seek(oh_addr)?;

        let first_bytes = reader.read_bytes(4)?;
        if first_bytes != [b'O', b'H', b'D', b'R'] {
            return Err(Error::Unsupported(
                "attribute mutation currently supports only v2 object headers".into(),
            ));
        }

        let version = reader.read_u8()?;
        if version != 2 {
            return Err(Error::Unsupported(
                "attribute mutation currently supports only v2 object headers".into(),
            ));
        }

        let flags = reader.read_u8()?;
        if flags & !HDR_V2_KNOWN_FLAGS != 0 {
            return Err(Error::InvalidFormat(format!(
                "object header v2 flags contain reserved bits: {flags:#04x}"
            )));
        }
        if flags & HDR_STORE_TIMES != 0 {
            reader.skip(16)?;
        }
        if flags & HDR_ATTR_STORE_PHASE_CHANGE != 0 {
            reader.skip(4)?;
        }

        let chunk0_size_bytes = 1u8 << (flags & HDR_CHUNK0_SIZE_MASK);
        let chunk0_data_size = reader.read_uint(chunk0_size_bytes)?;
        let chunk0_data_start = reader.position()?;
        let chunk0_data_end = chunk0_data_start
            .checked_add(chunk0_data_size)
            .ok_or_else(|| Error::InvalidFormat("object-header chunk range overflow".into()))?;
        let oh_check_len = usize::try_from(chunk0_data_end - oh_addr)
            .map_err(|_| Error::InvalidFormat("object-header checksum range overflow".into()))?;

        let mut has_dense_attrs = false;
        let mut found = None;
        while reader.position()? < chunk0_data_end {
            let msg_header_pos = reader.position()?;
            if msg_header_pos
                .checked_add(4)
                .is_none_or(|end| end > chunk0_data_end)
            {
                break;
            }

            let msg_type = u16::from(reader.read_u8()?);
            let msg_size = usize::from(reader.read_u16()?);
            let _msg_flags = reader.read_u8()?;
            if flags & object_header::HDR_ATTR_CRT_ORDER_TRACKED != 0 {
                reader.skip(2)?;
            }

            let msg_data_offset = reader.position()?;
            let msg_size_u64 = Self::usize_to_u64(msg_size, "object-header message size")?;
            if msg_data_offset
                .checked_add(msg_size_u64)
                .is_none_or(|end| end > chunk0_data_end)
            {
                return Err(Error::InvalidFormat(
                    "object-header message payload exceeds chunk".into(),
                ));
            }

            if msg_type == object_header::MSG_ATTRIBUTE {
                let data = reader.read_bytes(msg_size)?;
                let attr = AttributeMessage::decode(&data)?;
                if reject_duplicate_name.is_some_and(|name| attr.name == name) {
                    return Err(Error::InvalidFormat(format!(
                        "attribute '{}' already exists",
                        attr.name
                    )));
                }
                if attr.name == target_name {
                    let location = CompactAttributeMessageLocation {
                        msg_type_offset: msg_header_pos,
                        msg_data_offset,
                        oh_start: oh_addr,
                        oh_check_len,
                        raw_data: data,
                    };
                    if reject_duplicate_name.is_none() {
                        return Ok(location);
                    }
                    found = Some(location);
                }
            } else {
                if msg_type == object_header::MSG_ATTR_INFO {
                    has_dense_attrs = true;
                }
                reader.skip(msg_size_u64)?;
            }
        }

        if let Some(location) = found {
            return Ok(location);
        }

        if has_dense_attrs {
            Err(Error::Unsupported(
                "mutating dense attributes is not implemented".into(),
            ))
        } else {
            Err(Error::InvalidFormat(format!(
                "attribute '{target_name}' not found"
            )))
        }
    }

    fn find_dense_attribute_location(
        &self,
        object_addr: u64,
        target_name: &str,
        reject_duplicate_name: Option<&str>,
    ) -> Result<DenseAttributeLocation> {
        let mut guard = self.inner.lock();
        let oh = ObjectHeader::read_at(&mut guard.reader, object_addr)?;
        let attr_info_raw = oh
            .messages
            .iter()
            .find(|msg| msg.msg_type == object_header::MSG_ATTR_INFO)
            .ok_or_else(|| Error::InvalidFormat(format!("attribute '{target_name}' not found")))?;
        let attr_info =
            AttributeInfoMessage::decode(&attr_info_raw.data, guard.superblock.sizeof_addr)?;
        if !attr_info.has_dense_storage() {
            return Err(Error::InvalidFormat(format!(
                "attribute '{target_name}' not found"
            )));
        }
        if attr_info.corder_btree_addr.is_some() {
            return Err(Error::Unsupported(
                "mutating creation-order indexed dense attributes is not implemented".into(),
            ));
        }

        let heap = FractalHeapHeader::read_at(&mut guard.reader, attr_info.fractal_heap_addr)?;
        if heap.io_filter_len != 0 || heap.current_root_rows != 0 {
            return Err(Error::Unsupported(
                "mutating filtered or indirect dense attribute heaps is not implemented".into(),
            ));
        }

        let btree = BTreeV2Header::read_at(&mut guard.reader, attr_info.name_btree_addr)?;
        if btree.tree_type != 8 || btree.depth != 0 {
            return Err(Error::Unsupported(
                "mutating non-leaf dense attribute name indexes is not implemented".into(),
            ));
        }
        let records = btree_v2::collect_all_records(&mut guard.reader, attr_info.name_btree_addr)?;
        let heap_id_len = usize::from(heap.heap_id_len);
        let mut found = None;
        for (idx, record) in records.iter().enumerate() {
            let heap_id = checked_window(record, 0, heap_id_len, "dense attribute heap ID")?;
            let raw_data = heap.read_managed_object(&mut guard.reader, heap_id)?;
            let attr = AttributeMessage::decode(&raw_data)?;
            if reject_duplicate_name.is_some_and(|name| attr.name == name) {
                return Err(Error::InvalidFormat(format!(
                    "attribute '{}' already exists",
                    attr.name
                )));
            }
            if attr.name == target_name {
                let object_offset = managed_heap_object_offset(&heap, heap_id)?;
                found = Some(DenseAttributeLocation {
                    attr_info_addr: attr_info.name_btree_addr,
                    heap: heap.clone(),
                    btree: btree.clone(),
                    records: records.clone(),
                    record_index: idx,
                    object_offset,
                    raw_data,
                });
            }
        }

        found.ok_or_else(|| Error::InvalidFormat(format!("attribute '{target_name}' not found")))
    }

    fn delete_dense_attribute(&mut self, mut location: DenseAttributeLocation) -> Result<()> {
        location.records.remove(location.record_index);
        self.rewrite_dense_attribute_name_index(&location)
    }

    fn rename_dense_attribute(
        &mut self,
        mut location: DenseAttributeLocation,
        new_name: &str,
    ) -> Result<()> {
        let (name_offset, name_size) = Self::compact_attribute_name_field(&location.raw_data)?;
        if new_name.len() + 1 != name_size {
            return Err(Error::Unsupported(
                "in-place dense attribute rename cannot grow or shrink the encoded name field"
                    .into(),
            ));
        }

        let name_offset_u64 = Self::usize_to_u64(name_offset, "dense attribute name offset")?;
        let file_name_offset = location
            .heap
            .root_block_addr
            .checked_add(location.object_offset)
            .and_then(|offset| offset.checked_add(name_offset_u64))
            .ok_or_else(|| Error::InvalidFormat("dense attribute name offset overflow".into()))?;
        let mut encoded_name = vec![0u8; name_size];
        encoded_name[..new_name.len()].copy_from_slice(new_name.as_bytes());
        self.write_handle.seek(SeekFrom::Start(file_name_offset))?;
        self.write_handle.write_all(&encoded_name)?;

        let record = location
            .records
            .get_mut(location.record_index)
            .ok_or_else(|| {
                Error::InvalidFormat("dense attribute record index is invalid".into())
            })?;
        let hash_pos = dense_attribute_record_hash_pos(&location.heap, record)?;
        let hash_end = hash_pos
            .checked_add(4)
            .ok_or_else(|| Error::InvalidFormat("dense attribute hash offset overflow".into()))?;
        record[hash_pos..hash_end].copy_from_slice(&dense_name_hash(new_name).to_le_bytes());
        location.records.sort_by_key(|record| {
            dense_attribute_record_hash_pos(&location.heap, record)
                .ok()
                .and_then(|pos| read_u32_le_at(record, pos, "dense attribute record hash").ok())
                .unwrap_or(u32::MAX)
        });

        self.rewrite_dense_attribute_direct_block_checksum(
            &location.heap,
            file_name_offset,
            &encoded_name,
        )?;
        self.rewrite_dense_attribute_name_index(&location)
    }

    fn rewrite_dense_attribute_name_index(
        &mut self,
        location: &DenseAttributeLocation,
    ) -> Result<()> {
        let record_count_u16 =
            Self::usize_to_u16(location.records.len(), "dense attribute record count")?;
        let record_count_u64 =
            Self::usize_to_u64(location.records.len(), "dense attribute record count")?;
        let record_size = usize::from(location.btree.record_size);
        if location
            .records
            .iter()
            .any(|record| record.len() != record_size)
        {
            return Err(Error::InvalidFormat(
                "dense attribute records have inconsistent sizes".into(),
            ));
        }

        let mut leaf = Vec::with_capacity(
            6usize
                .checked_add(
                    record_size
                        .checked_mul(location.records.len())
                        .ok_or_else(|| {
                            Error::InvalidFormat(
                                "dense attribute leaf records size overflow".into(),
                            )
                        })?,
                )
                .and_then(|len| len.checked_add(4))
                .ok_or_else(|| Error::InvalidFormat("dense attribute leaf size overflow".into()))?,
        );
        leaf.extend_from_slice(b"BTLF");
        leaf.push(0);
        leaf.push(location.btree.tree_type);
        for record in &location.records {
            leaf.extend_from_slice(record);
        }
        let checksum = checksum_metadata(&leaf);
        leaf.extend_from_slice(&checksum.to_le_bytes());
        self.write_handle
            .seek(SeekFrom::Start(location.btree.root_addr))?;
        self.write_handle.write_all(&leaf)?;

        let sa = usize::from(self.superblock.sizeof_addr);
        let ss = usize::from(self.superblock.sizeof_size);
        let mut header = Vec::new();
        header.extend_from_slice(b"BTHD");
        header.push(0);
        header.push(location.btree.tree_type);
        header.extend_from_slice(&location.btree.node_size.to_le_bytes());
        header.extend_from_slice(&location.btree.record_size.to_le_bytes());
        header.extend_from_slice(&0u16.to_le_bytes());
        header.push(location.btree.split_pct);
        header.push(location.btree.merge_pct);
        header.extend_from_slice(&Self::encode_uint_le(
            location.btree.root_addr,
            sa,
            "dense attribute B-tree root address",
        )?);
        header.extend_from_slice(&record_count_u16.to_le_bytes());
        header.extend_from_slice(&Self::encode_uint_le(
            record_count_u64,
            ss,
            "dense attribute B-tree total record count",
        )?);
        let checksum = checksum_metadata(&header);
        header.extend_from_slice(&checksum.to_le_bytes());
        self.write_handle
            .seek(SeekFrom::Start(location.attr_info_addr))?;
        self.write_handle.write_all(&header)?;
        Ok(())
    }

    fn rewrite_dense_attribute_direct_block_checksum(
        &mut self,
        heap: &FractalHeapHeader,
        patched_addr: u64,
        patched_data: &[u8],
    ) -> Result<()> {
        if !heap.has_checksum {
            return Ok(());
        }
        let block_size = usize::try_from(heap.start_block_size)
            .map_err(|_| Error::InvalidFormat("dense attribute direct block too large".into()))?;
        let checksum_pos = direct_block_checksum_pos(heap, self.superblock.sizeof_addr)?;
        let checksum_end = checksum_pos
            .checked_add(4)
            .ok_or_else(|| Error::InvalidFormat("direct block checksum offset overflow".into()))?;
        let mut guard = self.inner.lock();
        guard.reader.seek(heap.root_block_addr)?;
        let mut block = guard.reader.read_bytes(block_size)?;
        drop(guard);
        let patch_start = patched_addr
            .checked_sub(heap.root_block_addr)
            .ok_or_else(|| Error::InvalidFormat("direct block patch address underflow".into()))?;
        let patch_start = usize::try_from(patch_start)
            .map_err(|_| Error::InvalidFormat("direct block patch offset too large".into()))?;
        let patch_end = patch_start
            .checked_add(patched_data.len())
            .ok_or_else(|| Error::InvalidFormat("direct block patch range overflow".into()))?;
        block
            .get_mut(patch_start..patch_end)
            .ok_or_else(|| Error::InvalidFormat("direct block patch exceeds block".into()))?
            .copy_from_slice(patched_data);
        let checksum_window = block.get_mut(checksum_pos..checksum_end).ok_or_else(|| {
            Error::InvalidFormat("direct block checksum field is truncated".into())
        })?;
        checksum_window.fill(0);
        let checksum = checksum_metadata(&block);
        let checksum_pos_u64 = Self::usize_to_u64(checksum_pos, "direct block checksum position")?;
        self.write_handle.seek(SeekFrom::Start(
            heap.root_block_addr
                .checked_add(checksum_pos_u64)
                .ok_or_else(|| {
                    Error::InvalidFormat("direct block checksum address overflow".into())
                })?,
        ))?;
        self.write_handle.write_all(&checksum.to_le_bytes())?;
        Ok(())
    }
}

fn checked_window<'a>(raw: &'a [u8], pos: usize, len: usize, context: &str) -> Result<&'a [u8]> {
    let end = pos
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))?;
    raw.get(pos..end)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))
}

fn read_u16_le_at(raw: &[u8], pos: usize, context: &str) -> Result<u16> {
    let bytes = checked_window(raw, pos, 2, context)?;
    let bytes: [u8; 2] = bytes
        .try_into()
        .map_err(|_| Error::InvalidFormat(format!("{context} is truncated")))?;
    Ok(u16::from_le_bytes(bytes))
}

fn read_u32_le_at(raw: &[u8], pos: usize, context: &str) -> Result<u32> {
    let bytes = checked_window(raw, pos, 4, context)?;
    let bytes: [u8; 4] = bytes
        .try_into()
        .map_err(|_| Error::InvalidFormat(format!("{context} is truncated")))?;
    Ok(u32::from_le_bytes(bytes))
}

fn managed_heap_object_offset(heap: &FractalHeapHeader, heap_id: &[u8]) -> Result<u64> {
    let offset_bytes = fractal_heap_offset_width(heap)?;
    let offset_bytes = checked_window(heap_id, 1, offset_bytes, "dense attribute heap offset")?;
    let mut offset = 0u64;
    for (idx, byte) in offset_bytes.iter().enumerate() {
        offset |= u64::from(*byte) << (idx * 8);
    }
    Ok(offset)
}

fn direct_block_checksum_pos(heap: &FractalHeapHeader, sizeof_addr: u8) -> Result<usize> {
    let offset_bytes = fractal_heap_offset_width(heap)?;
    5usize
        .checked_add(usize::from(sizeof_addr))
        .and_then(|pos| pos.checked_add(offset_bytes))
        .ok_or_else(|| Error::InvalidFormat("direct block checksum position overflow".into()))
}

fn fractal_heap_offset_width(heap: &FractalHeapHeader) -> Result<usize> {
    let max_heap_size = usize::from(heap.max_heap_size);
    let offset_bytes = max_heap_size
        .checked_add(7)
        .ok_or_else(|| Error::InvalidFormat("dense attribute heap offset width overflow".into()))?
        / 8;
    if offset_bytes == 0 || offset_bytes > 8 {
        return Err(Error::Unsupported(format!(
            "dense attribute heap offset width {offset_bytes} is unsupported"
        )));
    }
    Ok(offset_bytes)
}

fn dense_attribute_record_hash_pos(heap: &FractalHeapHeader, record: &[u8]) -> Result<usize> {
    let heap_id_len = usize::from(heap.heap_id_len);
    let hash_pos = heap_id_len
        .checked_add(1)
        .and_then(|pos| pos.checked_add(4))
        .ok_or_else(|| Error::InvalidFormat("dense attribute hash position overflow".into()))?;
    checked_window(record, hash_pos, 4, "dense attribute record hash")?;
    Ok(hash_pos)
}

fn dense_name_hash(name: &str) -> u32 {
    crate::format::checksum::checksum_lookup3(name.as_bytes(), 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_window_rejects_offset_overflow() {
        let err = checked_window(&[], usize::MAX, 1, "attribute mutation test").unwrap_err();
        assert!(
            err.to_string()
                .contains("attribute mutation test offset overflow"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn fractal_heap_offset_width_rejects_invalid_widths() {
        let heap = FractalHeapHeader {
            heap_addr: 0,
            heap_id_len: 0,
            io_filter_len: 0,
            flags: 0,
            max_managed_obj_size: 0,
            table_width: 0,
            start_block_size: 0,
            max_direct_block_size: 0,
            max_heap_size: 0,
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
        };
        assert!(fractal_heap_offset_width(&heap).is_err());

        let mut heap = heap;
        heap.max_heap_size = 65;
        assert!(fractal_heap_offset_width(&heap).is_err());

        heap.max_heap_size = 64;
        assert_eq!(fractal_heap_offset_width(&heap).unwrap(), 8);
    }
}
