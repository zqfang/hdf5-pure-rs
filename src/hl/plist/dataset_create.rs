use crate::format::messages::data_layout::LayoutClass;
use crate::format::messages::filter_pipeline::FilterDesc;

/// Dataset creation properties (read from an existing dataset).
#[derive(Debug, Clone)]
pub struct DatasetCreate {
    pub layout: LayoutClass,
    pub chunk_dims: Option<Vec<u64>>,
    pub chunk_opts: Option<u8>,
    pub filters: Vec<FilterInfo>,
    pub external_files: Vec<ExternalStorageInfo>,
    pub virtual_mappings: Vec<VirtualMappingInfo>,
    pub virtual_spatial_tree: bool,
    pub fill_alloc_time: Option<u8>,
    pub fill_time: Option<u8>,
    pub fill_value_defined: bool,
    pub fill_value: Option<Vec<u8>>,
}

/// Simplified filter description.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilterInfo {
    pub id: u16,
    pub name: String,
    pub flags: u16,
    pub params: Vec<u32>,
}

/// Borrowed filter description view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FilterInfoRef<'a> {
    pub id: u16,
    pub name: &'a str,
    pub flags: u16,
    pub params: &'a [u32],
}

/// External raw-data storage entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalStorageInfo {
    pub name: String,
    pub file_offset: u64,
    pub size: u64,
}

/// Borrowed external raw-data storage entry view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExternalStorageInfoRef<'a> {
    pub name: &'a str,
    pub file_offset: u64,
    pub size: u64,
}

/// Virtual-dataset mapping entry stored in a dataset creation property list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VirtualMappingInfo {
    pub file_name: String,
    pub dataset_name: String,
    pub source_select: VirtualSelectionInfo,
    pub virtual_select: VirtualSelectionInfo,
}

/// Borrowed virtual-dataset mapping entry view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VirtualMappingInfoRef<'a> {
    pub file_name: &'a str,
    pub dataset_name: &'a str,
    pub source_select: VirtualSelectionInfoRef<'a>,
    pub virtual_select: VirtualSelectionInfoRef<'a>,
}

/// Serialized virtual-dataset source or destination selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VirtualSelectionInfo {
    All,
    Points(Vec<Vec<u64>>),
    Regular {
        start: Vec<u64>,
        stride: Vec<u64>,
        count: Vec<u64>,
        block: Vec<u64>,
    },
    Irregular(Vec<IrregularHyperslabBlockInfo>),
}

/// Borrowed serialized virtual-dataset source or destination selection view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VirtualSelectionInfoRef<'a> {
    All,
    Points(&'a [Vec<u64>]),
    Regular {
        start: &'a [u64],
        stride: &'a [u64],
        count: &'a [u64],
        block: &'a [u64],
    },
    Irregular(&'a [IrregularHyperslabBlockInfo]),
}

/// Irregular hyperslab block in a virtual-dataset mapping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrregularHyperslabBlockInfo {
    pub start: Vec<u64>,
    pub block: Vec<u64>,
}

impl FilterInfo {
    pub fn from_desc(desc: &FilterDesc) -> Self {
        let name = match desc.id {
            1 => "deflate".to_string(),
            2 => "shuffle".to_string(),
            3 => "fletcher32".to_string(),
            4 => "szip".to_string(),
            5 => "nbit".to_string(),
            6 => "scaleoffset".to_string(),
            _ => desc
                .name
                .clone()
                .unwrap_or_else(|| format!("filter_{}", desc.id)),
        };
        Self {
            id: desc.id,
            name,
            flags: desc.flags,
            params: desc.client_data.clone(),
        }
    }

    pub fn from_desc_owned(desc: FilterDesc) -> Self {
        let name = match desc.id {
            1 => "deflate".to_string(),
            2 => "shuffle".to_string(),
            3 => "fletcher32".to_string(),
            4 => "szip".to_string(),
            5 => "nbit".to_string(),
            6 => "scaleoffset".to_string(),
            _ => desc.name.unwrap_or_else(|| format!("filter_{}", desc.id)),
        };
        Self {
            id: desc.id,
            name,
            flags: desc.flags,
            params: desc.client_data,
        }
    }

    pub fn as_view(&self) -> FilterInfoRef<'_> {
        FilterInfoRef {
            id: self.id,
            name: self.name.as_str(),
            flags: self.flags,
            params: self.params.as_slice(),
        }
    }
}

impl ExternalStorageInfo {
    pub fn as_view(&self) -> ExternalStorageInfoRef<'_> {
        ExternalStorageInfoRef {
            name: self.name.as_str(),
            file_offset: self.file_offset,
            size: self.size,
        }
    }
}

impl VirtualMappingInfo {
    pub fn as_view(&self) -> VirtualMappingInfoRef<'_> {
        VirtualMappingInfoRef {
            file_name: self.file_name.as_str(),
            dataset_name: self.dataset_name.as_str(),
            source_select: self.source_select.as_view(),
            virtual_select: self.virtual_select.as_view(),
        }
    }
}

impl VirtualSelectionInfo {
    pub fn as_view(&self) -> VirtualSelectionInfoRef<'_> {
        match self {
            Self::All => VirtualSelectionInfoRef::All,
            Self::Points(points) => VirtualSelectionInfoRef::Points(points.as_slice()),
            Self::Regular {
                start,
                stride,
                count,
                block,
            } => VirtualSelectionInfoRef::Regular {
                start: start.as_slice(),
                stride: stride.as_slice(),
                count: count.as_slice(),
                block: block.as_slice(),
            },
            Self::Irregular(blocks) => VirtualSelectionInfoRef::Irregular(blocks.as_slice()),
        }
    }
}

impl DatasetCreate {
    /// Extract dataset creation properties from a Dataset.
    pub fn from_dataset(ds: &crate::hl::dataset::Dataset) -> crate::Result<Self> {
        let info = ds.info()?;
        let mut out = Self {
            layout: info.layout.layout_class,
            chunk_dims: None,
            chunk_opts: None,
            filters: Vec::new(),
            external_files: Vec::new(),
            virtual_mappings: Vec::new(),
            virtual_spatial_tree: false,
            fill_alloc_time: None,
            fill_time: None,
            fill_value_defined: false,
            fill_value: None,
        };
        Self::from_dataset_info_into(ds, info, &mut out)?;
        Ok(out)
    }

    /// Extract dataset creation properties into existing storage.
    ///
    /// This keeps the public owned representation while allowing callers that
    /// repeatedly inspect datasets to reuse filter, external-file, and VDS
    /// mapping allocations between calls.
    pub fn from_dataset_into(
        ds: &crate::hl::dataset::Dataset,
        out: &mut Self,
    ) -> crate::Result<()> {
        let info = ds.info()?;
        Self::from_dataset_info_into(ds, info, out)
    }

    fn from_dataset_info_into(
        ds: &crate::hl::dataset::Dataset,
        info: crate::hl::dataset::DatasetInfo,
        out: &mut Self,
    ) -> crate::Result<()> {
        ds.external_storage_entries_with_info_into(&info, &mut out.external_files)?;
        ds.virtual_mapping_infos_with_info_into(&info, &mut out.virtual_mappings)?;

        out.filters.clear();
        if let Some(pipeline) = info.filter_pipeline {
            out.filters.reserve(pipeline.filters.len());
            out.filters.extend(
                pipeline
                    .filters
                    .into_iter()
                    .map(FilterInfo::from_desc_owned),
            );
        }

        out.layout = info.layout.layout_class;
        out.chunk_dims = info.layout.chunk_dims;
        out.chunk_opts = info.layout.chunk_flags;
        out.virtual_spatial_tree = false;
        out.fill_alloc_time = info.fill_value.as_ref().map(|fill| fill.alloc_time);
        out.fill_time = info.fill_value.as_ref().map(|fill| fill.fill_time);
        out.fill_value_defined = info
            .fill_value
            .as_ref()
            .map(|fill| fill.defined)
            .unwrap_or(false);
        out.fill_value = info.fill_value.and_then(|fill| fill.value);
        Ok(())
    }

    /// Whether the dataset is chunked.
    pub fn is_chunked(&self) -> bool {
        self.layout == LayoutClass::Chunked
    }

    /// Whether the dataset has any compression filters.
    pub fn is_compressed(&self) -> bool {
        self.filters.iter().any(|f| f.id == 1 || f.id == 4) // deflate or szip
    }

    /// Whether the dataset uses shuffle filter.
    pub fn has_shuffle(&self) -> bool {
        self.filters.iter().any(|f| f.id == 2)
    }

    /// Get the deflate compression level (if deflate is used).
    pub fn deflate_level(&self) -> Option<u32> {
        self.filters
            .iter()
            .find(|f| f.id == 1)
            .and_then(|f| f.params.first().copied())
    }

    /// Return chunk option flags for v4 chunked datasets.
    pub fn chunk_opts(&self) -> Option<u8> {
        self.chunk_opts
    }

    /// Borrow chunk dimensions when this is a chunked dataset.
    pub fn chunk_dims(&self) -> Option<&[u64]> {
        self.chunk_dims.as_deref()
    }

    /// Set v4 chunk option flags.
    pub fn set_chunk_opts(&mut self, opts: Option<u8>) {
        self.chunk_opts = opts;
    }

    /// Return the number of external raw-storage files.
    pub fn external_count(&self) -> usize {
        self.external_files.len()
    }

    /// Return one external raw-storage entry by index.
    pub fn external(&self, index: usize) -> Option<&ExternalStorageInfo> {
        self.external_files.get(index)
    }

    /// Return one borrowed external raw-storage entry view by index.
    pub fn external_entry(&self, index: usize) -> Option<ExternalStorageInfoRef<'_>> {
        self.external(index).map(ExternalStorageInfo::as_view)
    }

    /// Return borrowed external raw-storage entry views.
    pub fn external_entries(&self) -> impl Iterator<Item = ExternalStorageInfoRef<'_>> {
        self.external_files.iter().map(ExternalStorageInfo::as_view)
    }

    /// Append an external raw-storage file entry.
    pub fn set_external<S: Into<String>>(&mut self, name: S, file_offset: u64, size: u64) {
        self.external_files.push(ExternalStorageInfo {
            name: name.into(),
            file_offset,
            size,
        });
    }

    /// Return one filter by pipeline index.
    pub fn filter(&self, index: usize) -> Option<&FilterInfo> {
        self.filters.get(index)
    }

    /// Return one borrowed filter view by pipeline index.
    pub fn filter_entry(&self, index: usize) -> Option<FilterInfoRef<'_>> {
        self.filter(index).map(FilterInfo::as_view)
    }

    /// Return borrowed filter views in pipeline order.
    pub fn filter_entries(&self) -> impl Iterator<Item = FilterInfoRef<'_>> {
        self.filters.iter().map(FilterInfo::as_view)
    }

    /// Return one filter by filter id.
    pub fn filter_by_id(&self, id: u16) -> Option<&FilterInfo> {
        self.filters.iter().find(|filter| filter.id == id)
    }

    /// Return one borrowed filter view by filter id.
    pub fn filter_entry_by_id(&self, id: u16) -> Option<FilterInfoRef<'_>> {
        self.filter_by_id(id).map(FilterInfo::as_view)
    }

    /// Append or replace one filter in the dataset creation pipeline.
    pub fn set_filter<S: Into<String>>(&mut self, id: u16, name: S, flags: u16, params: Vec<u32>) {
        if let Some(filter) = self.filters.iter_mut().find(|filter| filter.id == id) {
            *filter = FilterInfo {
                id,
                name: name.into(),
                flags,
                params,
            };
        } else {
            self.filters.push(FilterInfo {
                id,
                name: name.into(),
                flags,
                params,
            });
        }
    }

    /// Enable the shuffle filter.
    pub fn set_shuffle(&mut self) {
        self.set_filter(2, "shuffle", 0, Vec::new());
    }

    /// Enable the NBit filter.
    pub fn set_nbit(&mut self) {
        self.set_filter(5, "nbit", 0, Vec::new());
    }

    /// Enable the ScaleOffset filter with client parameters.
    pub fn set_scaleoffset(&mut self, params: Vec<u32>) {
        self.set_filter(6, "scaleoffset", 0, params);
    }

    /// Enable the Fletcher32 checksum filter.
    pub fn set_fletcher32(&mut self) {
        self.set_filter(3, "fletcher32", 0, Vec::new());
    }

    /// Enable the SZip filter with client parameters.
    pub fn set_szip(&mut self, params: Vec<u32>) {
        self.set_filter(4, "szip", 0, params);
    }

    /// Return the number of filters in the dataset creation pipeline.
    pub fn filter_count(&self) -> usize {
        self.filters.len()
    }

    /// Return the number of virtual-dataset mappings.
    pub fn virtual_count(&self) -> usize {
        self.virtual_mappings.len()
    }

    /// Append one virtual-dataset mapping.
    pub fn set_virtual(&mut self, mapping: VirtualMappingInfo) {
        self.virtual_mappings.push(mapping);
    }

    /// Return borrowed virtual-dataset mapping views.
    pub fn virtual_mapping_entries(&self) -> impl Iterator<Item = VirtualMappingInfoRef<'_>> {
        self.virtual_mappings
            .iter()
            .map(VirtualMappingInfo::as_view)
    }

    /// Return the source file name for one virtual-dataset mapping.
    pub fn virtual_filename(&self, index: usize) -> Option<&str> {
        self.virtual_mappings
            .get(index)
            .map(|mapping| mapping.file_name.as_str())
    }

    /// Return the source dataset name for one virtual-dataset mapping.
    pub fn virtual_dsetname(&self, index: usize) -> Option<&str> {
        self.virtual_mappings
            .get(index)
            .map(|mapping| mapping.dataset_name.as_str())
    }

    /// Return the source selection for one virtual-dataset mapping.
    pub fn virtual_srcspace(&self, index: usize) -> Option<&VirtualSelectionInfo> {
        self.virtual_mappings
            .get(index)
            .map(|mapping| &mapping.source_select)
    }

    /// Return the borrowed source selection view for one virtual-dataset mapping.
    pub fn virtual_srcspace_view(&self, index: usize) -> Option<VirtualSelectionInfoRef<'_>> {
        self.virtual_srcspace(index)
            .map(VirtualSelectionInfo::as_view)
    }

    /// Return the virtual-dataset destination selection for one mapping.
    pub fn virtual_vspace(&self, index: usize) -> Option<&VirtualSelectionInfo> {
        self.virtual_mappings
            .get(index)
            .map(|mapping| &mapping.virtual_select)
    }

    /// Return the borrowed destination selection view for one virtual-dataset mapping.
    pub fn virtual_vspace_view(&self, index: usize) -> Option<VirtualSelectionInfoRef<'_>> {
        self.virtual_vspace(index)
            .map(VirtualSelectionInfo::as_view)
    }

    /// Whether this property list requests an HDF5 spatial tree for VDS lookups.
    ///
    /// The pure Rust reader materializes mapping coordinates directly and does
    /// not persist an HDF5 spatial-tree build flag, so existing files report the
    /// default disabled state.
    pub fn virtual_spatial_tree(&self) -> bool {
        self.virtual_spatial_tree
    }

    /// Set whether a VDS spatial tree should be requested.
    pub fn set_virtual_spatial_tree(&mut self, enabled: bool) {
        self.virtual_spatial_tree = enabled;
    }

    /// Set fill allocation time.
    pub fn set_alloc_time(&mut self, alloc_time: u8) {
        self.fill_alloc_time = Some(alloc_time);
    }

    /// Set raw fill value bytes.
    pub fn set_fill_value(&mut self, value: Option<Vec<u8>>) {
        self.fill_value_defined = value.is_some();
        self.fill_value = value;
    }

    /// Borrow raw fill value bytes.
    pub fn fill_value(&self) -> Option<&[u8]> {
        self.fill_value.as_deref()
    }
}
