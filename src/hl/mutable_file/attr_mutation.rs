use std::io::{Seek, SeekFrom, Write};

use crate::error::{Error, Result};
use crate::format::btree_v2::BTreeV2Header;
use crate::format::checksum::checksum_metadata;
use crate::format::fractal_heap::FractalHeapHeader;
use crate::format::messages::attribute::AttributeMessage;
use crate::format::messages::attribute_info::AttributeInfoMessage;
use crate::format::object_header::{
    self, HDR_ATTR_STORE_PHASE_CHANGE, HDR_CHUNK0_SIZE_MASK, HDR_STORE_TIMES, HDR_V2_KNOWN_FLAGS,
};

use super::MutableFile;

#[derive(Debug)]
struct CompactAttributeMessageLocation {
    msg_type_offset: u64,
    msg_data_offset: u64,
    oh_start: u64,
    oh_check_len: usize,
    raw_data: Vec<u8>,
}

#[derive(Debug)]
struct DenseAttributeLocation {
    attr_info_addr: u64,
    heap: FractalHeapHeader,
    btree: BTreeV2Header,
    leaf_records: Vec<u8>,
    record_index: usize,
    direct_block_addr: u64,
    direct_block_size: u64,
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
            Ok(mut location) => {
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

                let encoded_name = encode_attribute_name_in_place(
                    &mut location.raw_data,
                    name_offset,
                    name_size,
                    new_name,
                )?;
                self.write_handle.seek(SeekFrom::Start(file_name_offset))?;
                self.write_handle.write_all(encoded_name)?;
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

        let mut first_bytes = [0u8; 4];
        reader.read_bytes_into(&mut first_bytes)?;
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
                let mut data = vec![0u8; msg_size];
                reader.read_bytes_into(&mut data)?;
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
        let attr_info =
            Self::read_dense_attribute_info_message(&mut guard.reader, object_addr, target_name)?;
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
        if heap.io_filter_len != 0 {
            return Err(Error::Unsupported(
                "mutating filtered dense attribute heaps is not implemented".into(),
            ));
        }

        let btree = BTreeV2Header::read_at(&mut guard.reader, attr_info.name_btree_addr)?;
        if btree.tree_type != 8 || btree.depth != 0 {
            return Err(Error::Unsupported(
                "mutating non-leaf dense attribute name indexes is not implemented".into(),
            ));
        }
        let mut leaf_records = Vec::new();
        Self::read_dense_attribute_leaf_records_into(&mut guard.reader, &btree, &mut leaf_records)?;
        let heap_id_len = usize::from(heap.heap_id_len);
        let mut found = None;
        let record_size = usize::from(btree.record_size);
        for (idx, record) in leaf_records.chunks_exact(record_size).enumerate() {
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
                let object_location = heap.managed_object_location(&mut guard.reader, heap_id)?;
                found = Some((idx, object_location, raw_data));
            }
        }

        let (record_index, object_location, raw_data) = found
            .ok_or_else(|| Error::InvalidFormat(format!("attribute '{target_name}' not found")))?;
        Ok(DenseAttributeLocation {
            attr_info_addr: attr_info.name_btree_addr,
            heap,
            btree,
            leaf_records,
            record_index,
            direct_block_addr: object_location.block_addr,
            direct_block_size: object_location.block_size,
            object_offset: object_location.object_offset,
            raw_data,
        })
    }

    fn delete_dense_attribute(&mut self, mut location: DenseAttributeLocation) -> Result<()> {
        let record_size = usize::from(location.btree.record_size);
        let start = location
            .record_index
            .checked_mul(record_size)
            .ok_or_else(|| Error::InvalidFormat("dense attribute record offset overflow".into()))?;
        let end = start
            .checked_add(record_size)
            .ok_or_else(|| Error::InvalidFormat("dense attribute record offset overflow".into()))?;
        if end > location.leaf_records.len() {
            return Err(Error::InvalidFormat(
                "dense attribute record index is invalid".into(),
            ));
        }
        location.leaf_records.copy_within(end.., start);
        location
            .leaf_records
            .truncate(location.leaf_records.len() - record_size);
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
            .direct_block_addr
            .checked_add(location.object_offset)
            .and_then(|offset| offset.checked_add(name_offset_u64))
            .ok_or_else(|| Error::InvalidFormat("dense attribute name offset overflow".into()))?;
        self.write_handle.seek(SeekFrom::Start(file_name_offset))?;
        {
            let encoded_name = encode_attribute_name_in_place(
                &mut location.raw_data,
                name_offset,
                name_size,
                new_name,
            )?;
            self.write_handle.write_all(encoded_name)?;
        }

        let record_size = usize::from(location.btree.record_size);
        let record_start = location
            .record_index
            .checked_mul(record_size)
            .ok_or_else(|| Error::InvalidFormat("dense attribute record offset overflow".into()))?;
        let record_end = record_start
            .checked_add(record_size)
            .ok_or_else(|| Error::InvalidFormat("dense attribute record offset overflow".into()))?;
        let record = location
            .leaf_records
            .get_mut(record_start..record_end)
            .ok_or_else(|| {
                Error::InvalidFormat("dense attribute record index is invalid".into())
            })?;
        let hash_pos = dense_attribute_record_hash_pos(&location.heap, record)?;
        let hash_end = hash_pos
            .checked_add(4)
            .ok_or_else(|| Error::InvalidFormat("dense attribute hash offset overflow".into()))?;
        record[hash_pos..hash_end].copy_from_slice(&dense_name_hash(new_name).to_le_bytes());
        Self::reposition_dense_attribute_record_by_hash(
            &location.heap,
            &mut location.leaf_records,
            location.record_index,
            record_size,
        )?;

        let encoded_name = checked_window(
            &location.raw_data,
            name_offset,
            name_size,
            "dense attribute encoded name",
        )?;
        self.rewrite_dense_attribute_direct_block_checksum(
            &location.heap,
            location.direct_block_addr,
            location.direct_block_size,
            file_name_offset,
            encoded_name,
        )?;
        self.rewrite_dense_attribute_name_index(&location)
    }

    fn rewrite_dense_attribute_name_index(
        &mut self,
        location: &DenseAttributeLocation,
    ) -> Result<()> {
        let record_size = usize::from(location.btree.record_size);
        if record_size == 0 || location.leaf_records.len() % record_size != 0 {
            return Err(Error::InvalidFormat(
                "dense attribute records have inconsistent sizes".into(),
            ));
        }
        let record_count = location.leaf_records.len() / record_size;
        let record_count_u16 = Self::usize_to_u16(record_count, "dense attribute record count")?;
        let record_count_u64 = Self::usize_to_u64(record_count, "dense attribute record count")?;

        let mut leaf = Vec::with_capacity(
            6usize
                .checked_add(location.leaf_records.len())
                .and_then(|len| len.checked_add(4))
                .ok_or_else(|| Error::InvalidFormat("dense attribute leaf size overflow".into()))?,
        );
        leaf.extend_from_slice(b"BTLF");
        leaf.push(0);
        leaf.push(location.btree.tree_type);
        leaf.extend_from_slice(&location.leaf_records);
        let checksum = checksum_metadata(&leaf);
        leaf.extend_from_slice(&checksum.to_le_bytes());
        self.write_handle
            .seek(SeekFrom::Start(location.btree.root_addr))?;
        self.write_handle.write_all(&leaf)?;

        let sa = usize::from(self.superblock.sizeof_addr);
        let ss = usize::from(self.superblock.sizeof_size);
        let header_capacity = 22usize
            .checked_add(sa)
            .and_then(|len| len.checked_add(ss))
            .ok_or_else(|| {
                Error::InvalidFormat("dense attribute B-tree header size overflow".into())
            })?;
        let mut header = Vec::with_capacity(header_capacity);
        header.extend_from_slice(b"BTHD");
        header.push(0);
        header.push(location.btree.tree_type);
        header.extend_from_slice(&location.btree.node_size.to_le_bytes());
        header.extend_from_slice(&location.btree.record_size.to_le_bytes());
        header.extend_from_slice(&0u16.to_le_bytes());
        header.push(location.btree.split_pct);
        header.push(location.btree.merge_pct);
        let mut scratch = [0u8; 8];
        header.extend_from_slice(Self::encode_uint_le_into(
            location.btree.root_addr,
            &mut scratch,
            sa,
            "dense attribute B-tree root address",
        )?);
        header.extend_from_slice(&record_count_u16.to_le_bytes());
        header.extend_from_slice(Self::encode_uint_le_into(
            record_count_u64,
            &mut scratch,
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

    fn read_dense_attribute_info_message<R: std::io::Read + Seek>(
        reader: &mut crate::io::reader::HdfReader<R>,
        oh_addr: u64,
        target_name: &str,
    ) -> Result<AttributeInfoMessage> {
        reader.seek(oh_addr)?;

        let mut first_bytes = [0u8; 4];
        reader.read_bytes_into(&mut first_bytes)?;
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

            if msg_type == object_header::MSG_ATTR_INFO {
                let mut data = vec![0u8; msg_size];
                reader.read_bytes_into(&mut data)?;
                return AttributeInfoMessage::decode(&data, reader.sizeof_addr());
            }
            reader.skip(msg_size_u64)?;
        }

        Err(Error::InvalidFormat(format!(
            "attribute '{target_name}' not found"
        )))
    }

    fn read_dense_attribute_leaf_records_into<R: std::io::Read + Seek>(
        reader: &mut crate::io::reader::HdfReader<R>,
        btree: &BTreeV2Header,
        records: &mut Vec<u8>,
    ) -> Result<()> {
        let record_size = usize::from(btree.record_size);
        let record_count = usize::from(btree.root_nrecords);
        let records_len = record_size
            .checked_mul(record_count)
            .ok_or_else(|| Error::InvalidFormat("dense attribute leaf records overflow".into()))?;
        let check_len = 6usize
            .checked_add(records_len)
            .ok_or_else(|| Error::InvalidFormat("dense attribute leaf size overflow".into()))?;
        let total_len = check_len
            .checked_add(4)
            .ok_or_else(|| Error::InvalidFormat("dense attribute leaf size overflow".into()))?;

        records.clear();
        records.resize(total_len, 0);
        reader.seek(btree.root_addr)?;
        reader.read_bytes_into(records)?;
        if records.get(..4) != Some(&b"BTLF"[..]) {
            return Err(Error::InvalidFormat(
                "invalid dense attribute B-tree leaf magic".into(),
            ));
        }
        if records.get(4).copied() != Some(0) || records.get(5).copied() != Some(btree.tree_type) {
            return Err(Error::InvalidFormat(
                "dense attribute B-tree leaf header does not match index".into(),
            ));
        }
        let stored = read_u32_le_at(records, check_len, "dense attribute leaf checksum")?;
        let computed = checksum_metadata(&records[..check_len]);
        if stored != computed {
            return Err(Error::InvalidFormat(format!(
                "dense attribute leaf checksum mismatch: stored={stored:#010x}, computed={computed:#010x}"
            )));
        }
        records.drain(..6);
        records.truncate(records_len);
        Ok(())
    }

    fn reposition_dense_attribute_record_by_hash(
        heap: &FractalHeapHeader,
        records: &mut Vec<u8>,
        record_index: usize,
        record_size: usize,
    ) -> Result<()> {
        if record_size == 0 || records.len() % record_size != 0 {
            return Err(Error::InvalidFormat(
                "dense attribute records have inconsistent sizes".into(),
            ));
        }
        let record_count = records.len() / record_size;
        if record_index >= record_count {
            return Err(Error::InvalidFormat(
                "dense attribute record index is invalid".into(),
            ));
        }
        let start = record_index
            .checked_mul(record_size)
            .ok_or_else(|| Error::InvalidFormat("dense attribute record offset overflow".into()))?;
        let end = start
            .checked_add(record_size)
            .ok_or_else(|| Error::InvalidFormat("dense attribute record offset overflow".into()))?;
        let record = records.get(start..end).ok_or_else(|| {
            Error::InvalidFormat("dense attribute record index is invalid".into())
        })?;
        let hash_pos = dense_attribute_record_hash_pos(heap, record)?;
        let changed_hash = read_u32_le_at(record, hash_pos, "dense attribute record hash")?;

        let mut insert_index = record_count - 1;
        let mut logical_idx = 0usize;
        for (idx, record) in records.chunks_exact(record_size).enumerate() {
            if idx == record_index {
                continue;
            }
            let hash_pos = dense_attribute_record_hash_pos(heap, record)?;
            let hash = read_u32_le_at(record, hash_pos, "dense attribute record hash")?;
            if changed_hash < hash {
                insert_index = logical_idx;
                break;
            }
            logical_idx += 1;
        }

        if insert_index == record_index {
            return Ok(());
        }
        let insert_pos = insert_index
            .checked_mul(record_size)
            .ok_or_else(|| Error::InvalidFormat("dense attribute record offset overflow".into()))?;
        if insert_index < record_index {
            records[insert_pos..end].rotate_right(record_size);
        } else {
            let rotate_end = insert_pos.checked_add(record_size).ok_or_else(|| {
                Error::InvalidFormat("dense attribute record offset overflow".into())
            })?;
            records[start..rotate_end].rotate_left(record_size);
        }
        Ok(())
    }

    fn rewrite_dense_attribute_direct_block_checksum(
        &mut self,
        heap: &FractalHeapHeader,
        block_addr: u64,
        block_size: u64,
        patched_addr: u64,
        patched_data: &[u8],
    ) -> Result<()> {
        if !heap.has_checksum {
            return Ok(());
        }
        let block_size = usize::try_from(block_size)
            .map_err(|_| Error::InvalidFormat("dense attribute direct block too large".into()))?;
        let checksum_pos = direct_block_checksum_pos(heap, self.superblock.sizeof_addr)?;
        let checksum_end = checksum_pos
            .checked_add(4)
            .ok_or_else(|| Error::InvalidFormat("direct block checksum offset overflow".into()))?;
        let mut guard = self.inner.lock();
        guard.reader.seek(block_addr)?;
        let mut block = vec![0u8; block_size];
        guard.reader.read_bytes_into(&mut block)?;
        drop(guard);
        let patch_start = patched_addr
            .checked_sub(block_addr)
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
            block_addr.checked_add(checksum_pos_u64).ok_or_else(|| {
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

fn encode_attribute_name_in_place<'a>(
    raw: &'a mut [u8],
    name_offset: usize,
    name_size: usize,
    name: &str,
) -> Result<&'a [u8]> {
    let name_field = raw
        .get_mut(
            name_offset..name_offset.checked_add(name_size).ok_or_else(|| {
                Error::InvalidFormat("attribute name field offset overflow".into())
            })?,
        )
        .ok_or_else(|| Error::InvalidFormat("attribute name field exceeds message".into()))?;
    name_field.fill(0);
    name_field[..name.len()].copy_from_slice(name.as_bytes());
    Ok(name_field)
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
