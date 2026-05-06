use crate::error::{Error, Result};
use crate::format::local_heap::LocalHeap;
use crate::format::messages::data_layout::{ChunkIndexType, DataLayoutMessage, LayoutClass};
use crate::format::messages::dataspace::DataspaceMessage;
use crate::format::messages::datatype::DatatypeMessage;
use crate::format::messages::fill_value::FillValueMessage;
use crate::format::messages::filter_pipeline::FilterPipelineMessage;
use crate::format::object_header::{self, ObjectHeader, RawMessage};

use super::{
    read_le_uint_at, read_u8_at, u64_from_usize, usize_from_u64, Dataset, DatasetAccess, VdsView,
};

fn is_undefined_external_addr(addr: u64, sizeof_addr: usize) -> Result<bool> {
    let bits = sizeof_addr
        .checked_mul(8)
        .ok_or_else(|| Error::InvalidFormat("external file list address size overflow".into()))?;
    let undef = if bits == 64 {
        u64::MAX
    } else if bits < 64 {
        (1u64 << bits) - 1
    } else {
        return Err(Error::InvalidFormat(
            "external file list address is wider than u64".into(),
        ));
    };
    Ok(addr == undef)
}

/// Metadata about a dataset parsed from its object header.
#[derive(Debug)]
pub struct DatasetInfo {
    pub dataspace: DataspaceMessage,
    pub datatype: DatatypeMessage,
    pub layout: DataLayoutMessage,
    pub filter_pipeline: Option<FilterPipelineMessage>,
    pub fill_value: Option<FillValueMessage>,
    pub external_file_list: Option<ExternalFileList>,
}

#[derive(Debug, Clone)]
pub struct ExternalFileList {
    pub heap_addr: u64,
    pub entries: Vec<ExternalFileEntry>,
}

#[derive(Debug, Clone)]
pub struct ExternalFileEntry {
    pub name_offset: u64,
    pub file_offset: u64,
    pub size: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetSpaceStatus {
    NotAllocated,
    PartAllocated,
    Allocated,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkInfo {
    pub offset: Vec<u64>,
    pub filter_mask: u32,
    pub addr: u64,
    pub size: u64,
}

impl Dataset {
    /// List attribute names.
    pub fn attr_names(&self) -> Result<Vec<String>> {
        crate::hl::attribute::attr_names(&self.inner, self.addr)
    }

    /// List attributes.
    pub fn attrs(&self) -> Result<Vec<crate::hl::attribute::Attribute>> {
        crate::hl::attribute::collect_attributes(&self.inner, self.addr)
    }

    /// List attributes sorted by tracked creation order.
    pub fn attrs_by_creation_order(&self) -> Result<Vec<crate::hl::attribute::Attribute>> {
        crate::hl::attribute::collect_attributes_by_creation_order(&self.inner, self.addr)
    }

    /// Get an attribute by name.
    pub fn attr(&self, name: &str) -> Result<crate::hl::attribute::Attribute> {
        crate::hl::attribute::get_attr(&self.inner, self.addr, name)
    }

    /// Check whether an attribute exists on this dataset.
    pub fn attr_exists(&self, name: &str) -> Result<bool> {
        crate::hl::attribute::attr_exists(&self.inner, self.addr, name)
    }

    /// Parse the dataset's metadata from its object header.
    pub fn info(&self) -> Result<DatasetInfo> {
        let mut guard = self.inner.lock();
        let sizeof_addr = guard.superblock.sizeof_addr;
        let sizeof_size = guard.superblock.sizeof_size;
        let oh = ObjectHeader::read_at(&mut guard.reader, self.addr)?;

        Self::parse_info(&oh.messages, sizeof_addr, sizeof_size)
    }

    pub(crate) fn parse_info(
        messages: &[RawMessage],
        sizeof_addr: u8,
        sizeof_size: u8,
    ) -> Result<DatasetInfo> {
        let mut dataspace = None;
        let mut datatype = None;
        let mut layout = None;
        let mut filter_pipeline = None;
        let mut fill_value = None;
        let mut old_fill_value_raw = None;
        let mut external_file_list = None;

        for msg in messages {
            match msg.msg_type {
                object_header::MSG_DATASPACE => {
                    dataspace = Some(DataspaceMessage::decode(&msg.data)?);
                }
                object_header::MSG_DATATYPE => {
                    datatype = Some(DatatypeMessage::decode(&msg.data)?);
                }
                object_header::MSG_LAYOUT => {
                    layout = Some(DataLayoutMessage::decode(
                        &msg.data,
                        sizeof_addr,
                        sizeof_size,
                    )?);
                }
                object_header::MSG_FILTER_PIPELINE => {
                    filter_pipeline = Some(FilterPipelineMessage::decode(&msg.data)?);
                }
                object_header::MSG_FILL_VALUE => {
                    fill_value = Some(FillValueMessage::decode(&msg.data)?);
                }
                object_header::MSG_FILL_VALUE_OLD => {
                    old_fill_value_raw = Some(msg.data.as_slice());
                }
                object_header::MSG_EXTERNAL_FILE_LIST => {
                    external_file_list = Some(Self::decode_external_file_list(
                        &msg.data,
                        usize::from(sizeof_addr),
                        usize::from(sizeof_size),
                    )?);
                }
                _ => {}
            }
        }

        if fill_value.is_none() {
            if let (Some(raw), Some(datatype)) = (old_fill_value_raw, datatype.as_ref()) {
                fill_value = Some(FillValueMessage::decode_old_with_datatype_size(
                    raw,
                    Some(usize_from_u64(u64::from(datatype.size), "datatype size")?),
                )?);
            }
        }

        Ok(DatasetInfo {
            dataspace: dataspace
                .ok_or_else(|| Error::InvalidFormat("dataset missing dataspace message".into()))?,
            datatype: datatype
                .ok_or_else(|| Error::InvalidFormat("dataset missing datatype message".into()))?,
            layout: layout
                .ok_or_else(|| Error::InvalidFormat("dataset missing layout message".into()))?,
            filter_pipeline,
            fill_value,
            external_file_list,
        })
    }

    pub(super) fn decode_external_file_list(
        data: &[u8],
        sizeof_addr: usize,
        sizeof_size: usize,
    ) -> Result<ExternalFileList> {
        let mut pos = 0usize;
        let version = read_u8_at(data, &mut pos)?;
        if version != 1 {
            return Err(Error::Unsupported(format!(
                "external file list version {version}"
            )));
        }
        let reserved_end = pos
            .checked_add(3)
            .ok_or_else(|| Error::InvalidFormat("external file list offset overflow".into()))?;
        pos = reserved_end;
        let allocated_slots = read_le_uint_at(data, &mut pos, 2)?;
        if allocated_slots == 0 {
            return Err(Error::InvalidFormat(
                "external file list has no allocated slots".into(),
            ));
        }
        let used_slots = usize_from_u64(
            read_le_uint_at(data, &mut pos, 2)?,
            "external file list slot count",
        )?;
        if u64_from_usize(used_slots, "external file list slot count")? > allocated_slots {
            return Err(Error::InvalidFormat(
                "external file list uses more slots than allocated".into(),
            ));
        }
        let heap_addr = read_le_uint_at(data, &mut pos, sizeof_addr)?;
        if is_undefined_external_addr(heap_addr, sizeof_addr)? {
            return Err(Error::InvalidFormat(
                "external file list heap address is undefined".into(),
            ));
        }
        let mut entries = Vec::with_capacity(used_slots);
        for _ in 0..used_slots {
            entries.push(ExternalFileEntry {
                name_offset: read_le_uint_at(data, &mut pos, sizeof_size)?,
                file_offset: read_le_uint_at(data, &mut pos, sizeof_size)?,
                size: read_le_uint_at(data, &mut pos, sizeof_size)?,
            });
        }
        Ok(ExternalFileList { heap_addr, entries })
    }

    /// Get the shape of the dataset.
    pub fn shape(&self) -> Result<Vec<u64>> {
        self.shape_with_vds_view(VdsView::LastAvailable)
    }

    /// Get the shape of the dataset, overriding the VDS view policy.
    pub fn shape_with_vds_view(&self, view: VdsView) -> Result<Vec<u64>> {
        self.shape_with_access(&DatasetAccess::new().with_virtual_view(view))
    }

    /// Get the shape of the dataset, overriding dataset access properties.
    pub fn shape_with_access(&self, access: &DatasetAccess) -> Result<Vec<u64>> {
        let info = self.info()?;
        if info.layout.layout_class == LayoutClass::Virtual {
            return self.virtual_shape_with_info(&info, access);
        }
        Ok(info.dataspace.dims)
    }

    /// Get the total number of elements.
    pub fn size(&self) -> Result<u64> {
        self.size_with_vds_view(VdsView::LastAvailable)
    }

    /// Get the total number of elements, overriding the VDS view policy.
    pub fn size_with_vds_view(&self, view: VdsView) -> Result<u64> {
        self.size_with_access(&DatasetAccess::new().with_virtual_view(view))
    }

    /// Get the total number of elements, overriding dataset access properties.
    pub fn size_with_access(&self, access: &DatasetAccess) -> Result<u64> {
        let info = self.info()?;
        if info.layout.layout_class == LayoutClass::Virtual {
            let shape = self.virtual_shape_with_info(&info, access)?;
            return Self::dataspace_element_count(info.dataspace.space_type, &shape);
        }
        Self::dataspace_element_count(info.dataspace.space_type, &info.dataspace.dims)
    }

    /// Get the element size in bytes.
    pub fn element_size(&self) -> Result<usize> {
        let info = self.info()?;
        usize_from_u64(u64::from(info.datatype.size), "datatype size")
    }

    /// Get the datatype.
    pub fn dtype(&self) -> Result<crate::hl::datatype::Datatype> {
        let info = self.info()?;
        Ok(crate::hl::datatype::Datatype::from_message(info.datatype))
    }

    /// Return the parsed low-level datatype message.
    pub fn raw_datatype_message(&self) -> Result<DatatypeMessage> {
        Ok(self.info()?.datatype)
    }

    /// Get the dataspace.
    pub fn space(&self) -> Result<crate::hl::dataspace::Dataspace> {
        let info = self.info()?;
        Ok(crate::hl::dataspace::Dataspace::from_message(
            info.dataspace,
        ))
    }

    /// Return the parsed low-level dataspace message.
    pub fn raw_dataspace_message(&self) -> Result<DataspaceMessage> {
        Ok(self.info()?.dataspace)
    }

    /// Whether the dataset uses chunked storage.
    pub fn is_chunked(&self) -> Result<bool> {
        let info = self.info()?;
        Ok(info.layout.layout_class == LayoutClass::Chunked)
    }

    /// Whether this is a virtual dataset.
    pub fn is_virtual(&self) -> Result<bool> {
        let info = self.info()?;
        Ok(info.layout.layout_class == LayoutClass::Virtual)
    }

    /// Whether the dataset is resizable (has unlimited dimensions).
    pub fn is_resizable(&self) -> Result<bool> {
        Ok(self.space()?.is_resizable())
    }

    /// Get the storage layout type.
    pub fn layout(&self) -> Result<LayoutClass> {
        let info = self.info()?;
        Ok(info.layout.layout_class)
    }

    /// Get the contiguous raw-data file offset, if this dataset has one.
    ///
    /// This is the useful read-side subset of HDF5's `H5Dget_offset`.
    /// Compact, chunked, virtual, and not-yet-allocated contiguous datasets
    /// return `None`.
    pub fn offset(&self) -> Result<Option<u64>> {
        let info = self.info()?;
        if info.layout.layout_class != LayoutClass::Contiguous {
            return Ok(None);
        }
        Ok(info
            .layout
            .contiguous_addr
            .filter(|&addr| !crate::io::reader::is_undef_addr(addr)))
    }

    /// Get the chunk dimensions (None if not chunked).
    pub fn chunk(&self) -> Result<Option<Vec<u64>>> {
        let info = self.info()?;
        Ok(info.layout.chunk_dims.clone())
    }

    /// Get the filter pipeline (empty if no filters).
    pub fn filters(&self) -> Result<Vec<crate::format::messages::filter_pipeline::FilterDesc>> {
        let info = self.info()?;
        Ok(info.filter_pipeline.map(|p| p.filters).unwrap_or_default())
    }

    /// Get the dataset creation properties.
    pub fn create_plist(&self) -> Result<crate::hl::plist::dataset_create::DatasetCreate> {
        crate::hl::plist::dataset_create::DatasetCreate::from_dataset(self)
    }

    /// Get dataset access properties for default high-level reads.
    pub fn access_plist(&self) -> DatasetAccess {
        DatasetAccess::new()
    }

    /// Return the allocation status of dataset raw storage.
    pub fn space_status(&self) -> Result<DatasetSpaceStatus> {
        let info = self.info()?;
        match info.layout.layout_class {
            LayoutClass::Compact => Ok(DatasetSpaceStatus::Allocated),
            LayoutClass::Contiguous => {
                if info
                    .layout
                    .contiguous_addr
                    .is_some_and(|addr| !crate::io::reader::is_undef_addr(addr))
                    || info.external_file_list.is_some()
                {
                    Ok(DatasetSpaceStatus::Allocated)
                } else {
                    Ok(DatasetSpaceStatus::NotAllocated)
                }
            }
            LayoutClass::Chunked => {
                let count = self.num_chunks()?;
                if count == 0 {
                    Ok(DatasetSpaceStatus::NotAllocated)
                } else {
                    let expected = self.logical_chunk_count(&info)?;
                    if count >= expected {
                        Ok(DatasetSpaceStatus::Allocated)
                    } else {
                        Ok(DatasetSpaceStatus::PartAllocated)
                    }
                }
            }
            LayoutClass::Virtual => Ok(DatasetSpaceStatus::Allocated),
        }
    }

    /// Return the number of allocated chunks.
    pub fn num_chunks(&self) -> Result<usize> {
        Ok(self.chunk_infos()?.len())
    }

    /// Return allocated chunk metadata by storage-order index.
    pub fn chunk_info(&self, index: usize) -> Result<ChunkInfo> {
        self.chunk_infos()?
            .get(index)
            .cloned()
            .ok_or_else(|| Error::InvalidFormat(format!("chunk index {index} is out of bounds")))
    }

    fn chunk_infos(&self) -> Result<Vec<ChunkInfo>> {
        let info = self.info()?;
        if info.layout.layout_class != LayoutClass::Chunked {
            return Ok(Vec::new());
        }
        let idx_addr = info
            .layout
            .chunk_index_addr
            .ok_or_else(|| Error::InvalidFormat("chunked dataset missing index address".into()))?;
        if crate::io::reader::is_undef_addr(idx_addr) {
            return Ok(Vec::new());
        }

        let data_dims = &info.dataspace.dims;
        let raw_chunk_dims = info
            .layout
            .chunk_dims
            .as_ref()
            .ok_or_else(|| Error::InvalidFormat("chunked dataset missing chunk dims".into()))?;
        let chunk_dims = Self::chunk_data_dims(data_dims, raw_chunk_dims)?;
        let chunk_bytes = Self::chunk_byte_len(
            raw_chunk_dims,
            chunk_dims,
            usize_from_u64(u64::from(info.datatype.size), "datatype size")?,
        )?;
        let filtered = info
            .filter_pipeline
            .as_ref()
            .map(|pipeline| !pipeline.filters.is_empty())
            .unwrap_or(false);
        let mut guard = self.inner.lock();
        let sizeof_addr = usize::from(guard.reader.sizeof_addr());
        let sizeof_size = usize::from(guard.reader.sizeof_size());
        let infos = match info
            .layout
            .chunk_index_type
            .or_else(|| (info.layout.version <= 3).then_some(ChunkIndexType::BTreeV1))
        {
            Some(ChunkIndexType::SingleChunk) => {
                let size = info
                    .layout
                    .single_chunk_filtered_size
                    .unwrap_or(u64_from_usize(chunk_bytes, "single-chunk size")?);
                vec![ChunkInfo {
                    offset: vec![0; data_dims.len()],
                    filter_mask: info.layout.single_chunk_filter_mask.unwrap_or(0),
                    addr: idx_addr,
                    size,
                }]
            }
            Some(ChunkIndexType::BTreeV1) => {
                let records =
                    Self::collect_btree_v1_chunks(&mut guard.reader, idx_addr, data_dims.len())?;
                records
                    .into_iter()
                    .filter(|record| !crate::io::reader::is_undef_addr(record.chunk_addr))
                    .map(|record| ChunkInfo {
                        offset: record.coords,
                        filter_mask: record.filter_mask,
                        addr: record.chunk_addr,
                        size: record.chunk_size,
                    })
                    .collect()
            }
            Some(ChunkIndexType::Implicit) => {
                self.implicit_chunk_infos(&info, idx_addr, chunk_dims, chunk_bytes)?
            }
            Some(ChunkIndexType::FixedArray) => {
                let chunk_size_len = if filtered {
                    Self::filtered_chunk_size_len(&info, chunk_bytes, sizeof_size)?
                } else {
                    0
                };
                let elements = crate::format::fixed_array::read_fixed_array_chunks(
                    &mut guard.reader,
                    idx_addr,
                    filtered,
                    chunk_size_len,
                )?;
                self.linear_index_chunk_infos(elements, &info, chunk_dims, chunk_bytes)?
            }
            Some(ChunkIndexType::ExtensibleArray) => {
                let chunk_size_len = if filtered {
                    Self::filtered_chunk_size_len(&info, chunk_bytes, sizeof_size)?
                } else {
                    0
                };
                let elements = crate::format::extensible_array::read_extensible_array_chunks(
                    &mut guard.reader,
                    idx_addr,
                    filtered,
                    chunk_size_len,
                )?;
                self.linear_index_chunk_infos(elements, &info, chunk_dims, chunk_bytes)?
            }
            Some(ChunkIndexType::BTreeV2) => {
                let chunk_size_len = if filtered {
                    Self::filtered_chunk_size_len(&info, chunk_bytes, sizeof_size)?
                } else {
                    0
                };
                let records =
                    crate::format::btree_v2::collect_all_records(&mut guard.reader, idx_addr)?;
                records
                    .into_iter()
                    .map(|record| {
                        let (addr, size, filter_mask, scaled) = Self::decode_btree_v2_chunk_record(
                            &record,
                            filtered,
                            chunk_size_len,
                            sizeof_addr,
                            data_dims.len(),
                            chunk_bytes,
                        )?;
                        let offset = scaled
                            .iter()
                            .zip(chunk_dims)
                            .map(|(&coord, &chunk)| {
                                coord.checked_mul(chunk).ok_or_else(|| {
                                    Error::InvalidFormat("chunk coordinate overflow".into())
                                })
                            })
                            .collect::<Result<Vec<_>>>()?;
                        Ok(ChunkInfo {
                            offset,
                            filter_mask,
                            addr,
                            size,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?
                    .into_iter()
                    .filter(|chunk| !crate::io::reader::is_undef_addr(chunk.addr))
                    .collect()
            }
            None => Vec::new(),
        };
        Ok(infos)
    }

    fn logical_chunk_count(&self, info: &DatasetInfo) -> Result<usize> {
        let Some(raw_chunk_dims) = info.layout.chunk_dims.as_ref() else {
            return Ok(0);
        };
        let chunk_dims = Self::chunk_data_dims(&info.dataspace.dims, raw_chunk_dims)?;
        Self::chunks_per_dim(&info.dataspace.dims, chunk_dims)?
            .into_iter()
            .try_fold(1usize, |acc, value| acc.checked_mul(value))
            .ok_or_else(|| Error::InvalidFormat("chunk count overflow".into()))
    }

    fn implicit_chunk_infos(
        &self,
        info: &DatasetInfo,
        idx_addr: u64,
        chunk_dims: &[u64],
        chunk_bytes: usize,
    ) -> Result<Vec<ChunkInfo>> {
        let chunks_per_dim = Self::chunks_per_dim(&info.dataspace.dims, chunk_dims)?;
        let total_chunks = chunks_per_dim
            .iter()
            .try_fold(1usize, |acc, &value| acc.checked_mul(value))
            .ok_or_else(|| Error::InvalidFormat("chunk count overflow".into()))?;
        let mut infos = Vec::with_capacity(total_chunks);
        for chunk_index in 0..total_chunks {
            let scaled = Self::implicit_chunk_coords(chunk_index, chunk_dims, &chunks_per_dim)?;
            let offset = scaled
                .iter()
                .zip(chunk_dims)
                .map(|(&coord, &chunk)| {
                    coord
                        .checked_mul(chunk)
                        .ok_or_else(|| Error::InvalidFormat("chunk coordinate overflow".into()))
                })
                .collect::<Result<Vec<_>>>()?;
            let addr = idx_addr
                .checked_add(
                    u64_from_usize(chunk_index, "implicit chunk index")?
                        .checked_mul(u64_from_usize(chunk_bytes, "implicit chunk byte size")?)
                        .ok_or_else(|| Error::InvalidFormat("chunk address overflow".into()))?,
                )
                .ok_or_else(|| Error::InvalidFormat("chunk address overflow".into()))?;
            infos.push(ChunkInfo {
                offset,
                filter_mask: 0,
                addr,
                size: u64_from_usize(chunk_bytes, "implicit chunk size")?,
            });
        }
        Ok(infos)
    }

    fn linear_index_chunk_infos(
        &self,
        elements: Vec<crate::format::fixed_array::FixedArrayElement>,
        info: &DatasetInfo,
        chunk_dims: &[u64],
        chunk_bytes: usize,
    ) -> Result<Vec<ChunkInfo>> {
        let chunks_per_dim = Self::chunks_per_dim(&info.dataspace.dims, chunk_dims)?;
        let mut chunks = Vec::new();
        for (chunk_index, element) in elements.into_iter().enumerate() {
            if crate::io::reader::is_undef_addr(element.addr) {
                continue;
            }
            let scaled = Self::implicit_chunk_coords(chunk_index, chunk_dims, &chunks_per_dim)?;
            let offset = scaled
                .iter()
                .zip(chunk_dims)
                .map(|(&coord, &chunk)| {
                    coord
                        .checked_mul(chunk)
                        .ok_or_else(|| Error::InvalidFormat("chunk coordinate overflow".into()))
                })
                .collect::<Result<Vec<_>>>()?;
            chunks.push(ChunkInfo {
                offset,
                filter_mask: element.filter_mask,
                addr: element.addr,
                size: element
                    .nbytes
                    .unwrap_or(u64_from_usize(chunk_bytes, "linear chunk size")?),
            });
        }
        Ok(chunks)
    }

    pub(crate) fn external_storage_entries_with_info(
        &self,
        info: &DatasetInfo,
    ) -> Result<Vec<crate::hl::plist::dataset_create::ExternalStorageInfo>> {
        let Some(external) = info.external_file_list.as_ref() else {
            return Ok(Vec::new());
        };
        let mut guard = self.inner.lock();
        let heap = LocalHeap::read_at(&mut guard.reader, external.heap_addr)?;
        external
            .entries
            .iter()
            .map(|entry| {
                let name_offset = usize_from_u64(entry.name_offset, "external file name offset")?;
                Ok(crate::hl::plist::dataset_create::ExternalStorageInfo {
                    name: heap.get_string(name_offset)?,
                    file_offset: entry.file_offset,
                    size: entry.size,
                })
            })
            .collect()
    }
}
