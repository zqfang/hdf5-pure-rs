use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::engine::dataset_api::{
    H5D__virtual_build_source_name_into, H5D_virtual_parse_source_name, VirtualParsedName,
};
use crate::error::{Error, Result};
use crate::format::messages::data_layout::LayoutClass;
use crate::hl::plist::dataset_create::{
    IrregularHyperslabBlockInfo, VirtualMappingInfo, VirtualSelectionInfo,
};

use super::{usize_from_u64, Dataset, DatasetAccess, DatasetInfo, VdsMissingSourcePolicy, VdsView};

#[derive(Debug, Clone)]
pub(super) struct VirtualMapping {
    pub(super) file_name: String,
    pub(super) dataset_name: String,
    pub(super) source_select: VirtualSelection,
    pub(super) virtual_select: VirtualSelection,
}

struct VirtualSourceCacheEntry {
    dataset: Dataset,
    info: DatasetInfo,
    raw: Option<Vec<u8>>,
}

struct VirtualSourceCache {
    indexes: HashMap<PathBuf, Vec<VirtualSourceCacheDatasetIndex>>,
    entries: Vec<VirtualSourceCacheEntry>,
}

struct VirtualSourceCacheDatasetIndex {
    dataset_name: String,
    index: usize,
}

impl VirtualSourceCache {
    fn new() -> Self {
        Self {
            indexes: HashMap::new(),
            entries: Vec::new(),
        }
    }

    fn source_info(
        &mut self,
        file_path: Option<&Path>,
        mapping: &VirtualMapping,
        access: &DatasetAccess,
    ) -> Result<&DatasetInfo> {
        let index = self.source_index(file_path, mapping, access)?;
        Ok(&self.entries[index].info)
    }

    fn source_raw(
        &mut self,
        file_path: Option<&Path>,
        mapping: &VirtualMapping,
        dest_info: &DatasetInfo,
        access: &DatasetAccess,
    ) -> Result<(&DatasetInfo, &[u8])> {
        let index = self.source_index(file_path, mapping, access)?;
        let entry = &mut self.entries[index];
        if entry.raw.is_none() {
            let (_, _, source_bytes) = Dataset::raw_read_size(&entry.info)?;
            let mut source_raw = vec![0; source_bytes];
            entry
                .dataset
                .read_raw_into_with_access(access, &mut source_raw)?;
            entry.raw = Some(crate::hl::conversion::convert_between_datatypes(
                &source_raw,
                &entry.info.datatype,
                &dest_info.datatype,
            )?);
        }
        let raw = entry
            .raw
            .as_deref()
            .expect("virtual source raw data must be cached after read");
        Ok((&entry.info, raw))
    }

    fn source_index(
        &mut self,
        file_path: Option<&Path>,
        mapping: &VirtualMapping,
        access: &DatasetAccess,
    ) -> Result<usize> {
        let source_path =
            Dataset::resolve_virtual_source_path(file_path, &mapping.file_name, access)?;
        if let Some(indexes) = self.indexes.get(&source_path) {
            if let Some(index) = indexes
                .iter()
                .find(|entry| entry.dataset_name == mapping.dataset_name.as_str())
                .map(|entry| entry.index)
            {
                return Ok(index);
            }
        }

        let source = crate::hl::file::File::open(&source_path)?;
        let dataset = source.dataset(&mapping.dataset_name)?;
        let info = dataset.info()?;
        let index = self.entries.len();
        self.entries.push(VirtualSourceCacheEntry {
            dataset,
            info,
            raw: None,
        });
        self.indexes
            .entry(source_path)
            .or_default()
            .push(VirtualSourceCacheDatasetIndex {
                dataset_name: mapping.dataset_name.clone(),
                index,
            });
        Ok(index)
    }
}

enum VirtualSourceElementIndexes<'a> {
    All { len: usize },
    Borrowed(&'a [Vec<u64>]),
    Owned(Vec<usize>),
}

impl VirtualSourceElementIndexes<'_> {
    fn len(&self) -> usize {
        match self {
            Self::All { len } => *len,
            Self::Borrowed(points) => points.len(),
            Self::Owned(indexes) => indexes.len(),
        }
    }

    fn linear_index(&self, index: usize, strides: &[usize]) -> Result<Option<usize>> {
        match self {
            Self::All { len } => Ok((index < *len).then_some(index)),
            Self::Borrowed(points) => points
                .get(index)
                .map(|point| Dataset::linear_index(point, strides))
                .transpose(),
            Self::Owned(indexes) => Ok(indexes.get(index).copied()),
        }
    }
}

#[derive(Debug, Clone)]
pub(super) enum VirtualSelection {
    All,
    Points(Vec<Vec<u64>>),
    Regular(RegularHyperslab),
    Irregular(Vec<IrregularHyperslabBlock>),
}

#[derive(Debug, Clone)]
pub(super) struct RegularHyperslab {
    pub(super) start: Vec<u64>,
    pub(super) stride: Vec<u64>,
    pub(super) count: Vec<u64>,
    pub(super) block: Vec<u64>,
}

#[derive(Debug, Clone)]
pub(super) struct IrregularHyperslabBlock {
    pub(super) start: Vec<u64>,
    pub(super) block: Vec<u64>,
}

impl Dataset {
    pub(crate) fn virtual_mapping_infos_with_info_into(
        &self,
        info: &DatasetInfo,
        out: &mut Vec<VirtualMappingInfo>,
    ) -> Result<()> {
        out.clear();
        if info.layout.layout_class != LayoutClass::Virtual {
            return Ok(());
        }
        let mut guard = self.inner.lock();
        let heap_addr = info.layout.virtual_heap_addr.ok_or_else(|| {
            Error::InvalidFormat("virtual dataset missing global heap address".into())
        })?;
        let heap_index = info.layout.virtual_heap_index.ok_or_else(|| {
            Error::InvalidFormat("virtual dataset missing global heap index".into())
        })?;
        let mut heap_data = Vec::new();
        crate::format::global_heap::read_global_heap_object_into(
            &mut guard.reader,
            &crate::format::global_heap::GlobalHeapRef {
                collection_addr: heap_addr,
                object_index: heap_index,
            },
            &mut heap_data,
        )?;
        let sizeof_size = usize::from(guard.reader.sizeof_size());
        drop(guard);

        let mappings = Self::decode_virtual_mappings(&heap_data, sizeof_size)?;
        out.extend(
            mappings
                .into_iter()
                .map(Self::virtual_mapping_info_from_mapping),
        );
        Ok(())
    }

    fn virtual_mapping_info_from_mapping(mapping: VirtualMapping) -> VirtualMappingInfo {
        VirtualMappingInfo {
            file_name: mapping.file_name,
            dataset_name: mapping.dataset_name,
            source_select: Self::virtual_selection_info_from_selection(mapping.source_select),
            virtual_select: Self::virtual_selection_info_from_selection(mapping.virtual_select),
        }
    }

    fn virtual_selection_info_from_selection(selection: VirtualSelection) -> VirtualSelectionInfo {
        match selection {
            VirtualSelection::All => VirtualSelectionInfo::All,
            VirtualSelection::Points(points) => VirtualSelectionInfo::Points(points),
            VirtualSelection::Regular(regular) => VirtualSelectionInfo::Regular {
                start: regular.start,
                stride: regular.stride,
                count: regular.count,
                block: regular.block,
            },
            VirtualSelection::Irregular(blocks) => VirtualSelectionInfo::Irregular(
                blocks
                    .into_iter()
                    .map(|block| IrregularHyperslabBlockInfo {
                        start: block.start,
                        block: block.block,
                    })
                    .collect(),
            ),
        }
    }

    pub(super) fn virtual_shape_with_info(
        &self,
        info: &DatasetInfo,
        access: &DatasetAccess,
    ) -> Result<Vec<u64>> {
        let mut guard = self.inner.lock();
        let heap_addr = info.layout.virtual_heap_addr.ok_or_else(|| {
            Error::InvalidFormat("virtual dataset missing global heap address".into())
        })?;
        let heap_index = info.layout.virtual_heap_index.ok_or_else(|| {
            Error::InvalidFormat("virtual dataset missing global heap index".into())
        })?;
        let path = guard.path.clone();
        let mut heap_data = Vec::new();
        crate::format::global_heap::read_global_heap_object_into(
            &mut guard.reader,
            &crate::format::global_heap::GlobalHeapRef {
                collection_addr: heap_addr,
                object_index: heap_index,
            },
            &mut heap_data,
        )?;
        let sizeof_size = usize::from(guard.reader.sizeof_size());
        drop(guard);

        let mappings = Self::decode_virtual_mappings(&heap_data, sizeof_size)?;
        let mappings = Self::expand_virtual_printf_mappings(&mappings, path.as_deref(), access)?;
        let mut source_cache = VirtualSourceCache::new();
        Self::virtual_output_dims_with_cache(
            &mappings,
            path.as_deref(),
            info,
            access,
            &mut source_cache,
        )
    }

    pub(super) fn read_virtual_dataset(
        heap_data: &[u8],
        sizeof_size: usize,
        file_path: Option<&Path>,
        info: &DatasetInfo,
        access: &DatasetAccess,
    ) -> Result<Vec<u8>> {
        let mappings = Self::decode_virtual_mappings(heap_data, sizeof_size)?;
        let mappings = Self::expand_virtual_printf_mappings(&mappings, file_path, access)?;
        let element_size = usize_from_u64(u64::from(info.datatype.size), "datatype size")?;

        let mut source_cache = VirtualSourceCache::new();
        let output_dims = Self::virtual_output_dims_with_cache(
            &mappings,
            file_path,
            info,
            access,
            &mut source_cache,
        )?;
        let total_elements = usize_from_u64(
            Self::dataspace_element_count(info.dataspace.space_type, &output_dims)?,
            "virtual dataset element count",
        )?;
        let mut output = Self::filled_data(total_elements, element_size, info)?;
        Self::populate_virtual_dataset_output(
            mappings,
            file_path,
            info,
            access,
            &output_dims,
            element_size,
            &mut output,
            &mut source_cache,
        )?;
        Ok(output)
    }

    pub(super) fn read_virtual_dataset_into(
        heap_data: &[u8],
        sizeof_size: usize,
        file_path: Option<&Path>,
        info: &DatasetInfo,
        access: &DatasetAccess,
        output: &mut [u8],
    ) -> Result<()> {
        let mappings = Self::decode_virtual_mappings(heap_data, sizeof_size)?;
        let mappings = Self::expand_virtual_printf_mappings(&mappings, file_path, access)?;
        let element_size = usize_from_u64(u64::from(info.datatype.size), "datatype size")?;

        let mut source_cache = VirtualSourceCache::new();
        let output_dims = Self::virtual_output_dims_with_cache(
            &mappings,
            file_path,
            info,
            access,
            &mut source_cache,
        )?;
        let total_elements = usize_from_u64(
            Self::dataspace_element_count(info.dataspace.space_type, &output_dims)?,
            "virtual dataset element count",
        )?;
        let expected = total_elements
            .checked_mul(element_size)
            .ok_or_else(|| Error::InvalidFormat("virtual dataset byte size overflow".into()))?;
        if output.len() != expected {
            return Err(Error::InvalidFormat(format!(
                "raw output buffer has {} bytes, expected {expected}",
                output.len()
            )));
        }
        let mut staged = vec![0u8; output.len()];
        Self::filled_data_into(total_elements, element_size, info, &mut staged)?;
        Self::populate_virtual_dataset_output(
            mappings,
            file_path,
            info,
            access,
            &output_dims,
            element_size,
            &mut staged,
            &mut source_cache,
        )?;
        output.copy_from_slice(&staged);
        Ok(())
    }

    fn expand_virtual_printf_mappings(
        mappings: &[VirtualMapping],
        file_path: Option<&Path>,
        access: &DatasetAccess,
    ) -> Result<Vec<VirtualMapping>> {
        let mut expanded = Vec::with_capacity(mappings.len());
        for mapping in mappings {
            let parsed_file = H5D_virtual_parse_source_name(&mapping.file_name)?;
            let parsed_dataset = H5D_virtual_parse_source_name(&mapping.dataset_name)?;
            let has_block_substitution = parsed_file
                .as_ref()
                .map(|parsed| parsed.substitutions > 0)
                .unwrap_or(false)
                || parsed_dataset
                    .as_ref()
                    .map(|parsed| parsed.substitutions > 0)
                    .unwrap_or(false);

            if !has_block_substitution {
                expanded.push(Self::virtual_mapping_with_unescaped_names(
                    mapping,
                    parsed_file.as_ref(),
                    parsed_dataset.as_ref(),
                )?);
                continue;
            }

            let Some(parsed_file) = parsed_file.as_ref() else {
                expanded.push(mapping.clone());
                continue;
            };
            let block_numbers = Self::available_virtual_printf_file_blocks(
                file_path,
                access,
                mapping,
                parsed_file,
            )?;
            if block_numbers.is_empty() {
                expanded.push(mapping.clone());
                continue;
            }
            for block_number in block_numbers {
                expanded.push(Self::virtual_printf_mapping_for_block(
                    mapping,
                    parsed_file,
                    parsed_dataset.as_ref(),
                    block_number,
                )?);
            }
        }
        Ok(expanded)
    }

    fn virtual_mapping_with_unescaped_names(
        mapping: &VirtualMapping,
        parsed_file: Option<&VirtualParsedName>,
        parsed_dataset: Option<&VirtualParsedName>,
    ) -> Result<VirtualMapping> {
        if parsed_file.is_none() && parsed_dataset.is_none() {
            return Ok(mapping.clone());
        }
        let mut out = mapping.clone();
        if let Some(parsed) = parsed_file {
            H5D__virtual_build_source_name_into(
                &mapping.file_name,
                Some(parsed),
                0,
                &mut out.file_name,
            )?;
        }
        if let Some(parsed) = parsed_dataset {
            H5D__virtual_build_source_name_into(
                &mapping.dataset_name,
                Some(parsed),
                0,
                &mut out.dataset_name,
            )?;
        }
        Ok(out)
    }

    fn virtual_printf_mapping_for_block(
        mapping: &VirtualMapping,
        parsed_file: &VirtualParsedName,
        parsed_dataset: Option<&VirtualParsedName>,
        block_number: u64,
    ) -> Result<VirtualMapping> {
        let mut out = mapping.clone();
        H5D__virtual_build_source_name_into(
            &mapping.file_name,
            Some(parsed_file),
            block_number,
            &mut out.file_name,
        )?;
        if let Some(parsed) = parsed_dataset {
            H5D__virtual_build_source_name_into(
                &mapping.dataset_name,
                Some(parsed),
                block_number,
                &mut out.dataset_name,
            )?;
        }
        out.virtual_select =
            Self::virtual_printf_block_selection(&mapping.virtual_select, block_number)?;
        Ok(out)
    }

    fn virtual_printf_block_selection(
        selection: &VirtualSelection,
        block_number: u64,
    ) -> Result<VirtualSelection> {
        let VirtualSelection::Regular(regular) = selection else {
            return Ok(selection.clone());
        };
        let unlimited_dims = regular
            .count
            .iter()
            .enumerate()
            .filter_map(|(dim, &count)| (count == u64::MAX).then_some(dim))
            .collect::<Vec<_>>();
        if unlimited_dims.len() != 1 {
            return Ok(selection.clone());
        }
        let dim = unlimited_dims[0];
        let mut adjusted = regular.clone();
        let stride = adjusted.stride[dim].max(1);
        let offset = block_number
            .checked_mul(stride)
            .ok_or_else(|| Error::InvalidFormat("VDS printf block coordinate overflow".into()))?;
        adjusted.start[dim] = adjusted.start[dim]
            .checked_add(offset)
            .ok_or_else(|| Error::InvalidFormat("VDS printf block coordinate overflow".into()))?;
        adjusted.count[dim] = 1;
        Ok(VirtualSelection::Regular(adjusted))
    }

    fn available_virtual_printf_file_blocks(
        file_path: Option<&Path>,
        access: &DatasetAccess,
        mapping: &VirtualMapping,
        parsed_file: &VirtualParsedName,
    ) -> Result<Vec<u64>> {
        let mut blocks = Vec::new();
        let source_path = Path::new(&mapping.file_name);
        let source_file_name = source_path.file_name().ok_or_else(|| {
            Error::InvalidFormat("virtual dataset source has no file name".into())
        })?;
        let source_file_name = source_file_name.to_str().ok_or_else(|| {
            Error::InvalidFormat("virtual dataset source file name is not UTF-8".into())
        })?;
        let source_parent = source_path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty());

        for dir in Self::virtual_printf_search_dirs(file_path, access, source_path)? {
            let dir = source_parent.map(|parent| dir.join(parent)).unwrap_or(dir);
            let entries = match std::fs::read_dir(&dir) {
                Ok(entries) => entries,
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => continue,
                Err(err) => return Err(Error::Io(err)),
            };
            for entry in entries {
                let entry = entry.map_err(Error::Io)?;
                let name = entry.file_name();
                let Some(name) = name.to_str() else {
                    continue;
                };
                if let Some(block) =
                    parsed_virtual_block_number(source_file_name, parsed_file, name)?
                {
                    blocks.push(block);
                }
            }
        }

        blocks.sort_unstable();
        blocks.dedup();
        match access.virtual_view() {
            VdsView::LastAvailable => {}
            VdsView::FirstMissing => {
                let first_missing = blocks
                    .iter()
                    .copied()
                    .enumerate()
                    .find_map(|(idx, block)| (block != idx as u64).then_some(idx as u64))
                    .unwrap_or(blocks.len() as u64);
                blocks.retain(|&block| block < first_missing);
            }
        }
        Ok(blocks)
    }

    fn virtual_printf_search_dirs(
        file_path: Option<&Path>,
        access: &DatasetAccess,
        source_path: &Path,
    ) -> Result<Vec<PathBuf>> {
        let mut dirs = Vec::new();
        if source_path.is_absolute() {
            if let Some(parent) = source_path.parent() {
                dirs.push(parent.to_path_buf());
            }
            return Ok(dirs);
        }
        if let Some(prefix) = access.virtual_prefix() {
            if !prefix.as_os_str().is_empty() && prefix != Path::new(".") {
                dirs.push(expand_virtual_origin_prefix(file_path, prefix)?);
            }
        }
        if let Ok(prefixes) = std::env::var("HDF5_VDS_PREFIX") {
            for prefix in prefixes.split(':') {
                if prefix.is_empty() || prefix == "." {
                    continue;
                }
                dirs.push(expand_virtual_origin_prefix_str(file_path, prefix)?);
            }
        }
        if let Some(base) = file_path.and_then(Path::parent) {
            dirs.push(base.to_path_buf());
        }
        dirs.sort_unstable();
        dirs.dedup();
        Ok(dirs)
    }

    fn populate_virtual_dataset_output(
        mappings: Vec<VirtualMapping>,
        file_path: Option<&Path>,
        info: &DatasetInfo,
        access: &DatasetAccess,
        output_dims: &[u64],
        element_size: usize,
        output: &mut [u8],
        source_cache: &mut VirtualSourceCache,
    ) -> Result<()> {
        let virtual_strides = Self::row_major_strides(&output_dims)?;

        for mapping in mappings {
            let (source_info, source_raw) =
                match source_cache.source_raw(file_path, &mapping, info, access) {
                    Ok(source) => source,
                    Err(err) if Self::should_fill_missing_virtual_source(&err, access) => {
                        continue;
                    }
                    Err(err) => return Err(err),
                };
            Self::copy_virtual_mapping(
                &mapping,
                source_raw,
                &source_info.dataspace.dims,
                &virtual_strides,
                output_dims,
                element_size,
                output,
            )?;
        }
        Ok(())
    }

    fn should_fill_missing_virtual_source(err: &Error, access: &DatasetAccess) -> bool {
        access.virtual_missing_source_policy() == VdsMissingSourcePolicy::Fill
            && (matches!(err, Error::Io(io_err) if io_err.kind() == std::io::ErrorKind::NotFound)
                || matches!(err, Error::InvalidFormat(message) if is_missing_virtual_source_message(message)))
    }

    fn copy_virtual_mapping(
        mapping: &VirtualMapping,
        source_raw: &[u8],
        source_dims: &[u64],
        virtual_strides: &[usize],
        output_dims: &[u64],
        element_size: usize,
        output: &mut [u8],
    ) -> Result<()> {
        let source_strides = Self::row_major_strides(source_dims)?;
        let source_indexes = Self::virtual_source_element_indexes(
            &mapping.source_select,
            source_dims,
            &source_strides,
        )?;
        let mut copied_points = 0usize;
        Self::visit_virtual_selection_points(&mapping.virtual_select, output_dims, |dst| {
            let src_index = source_indexes
                .linear_index(copied_points, &source_strides)?
                .ok_or_else(|| {
                    Error::InvalidFormat(
                        "virtual dataset source and destination selections differ in size".into(),
                    )
                })?;
            let dst_index = Self::linear_index(dst, virtual_strides)?;
            copied_points += 1;
            Self::copy_virtual_element(source_raw, src_index, dst_index, element_size, output)
        })?;
        if copied_points != source_indexes.len() {
            return Err(Error::InvalidFormat(
                "virtual dataset source and destination selections differ in size".into(),
            ));
        }
        Ok(())
    }

    fn virtual_source_element_indexes<'a>(
        selection: &'a VirtualSelection,
        dims: &[u64],
        strides: &[usize],
    ) -> Result<VirtualSourceElementIndexes<'a>> {
        match selection {
            VirtualSelection::All => Ok(VirtualSourceElementIndexes::All {
                len: usize_from_u64(
                    dims.iter().try_fold(1u64, |total, dim| {
                        total.checked_mul(*dim).ok_or_else(|| {
                            Error::InvalidFormat("virtual source element count overflow".into())
                        })
                    })?,
                    "virtual source element count",
                )?,
            }),
            VirtualSelection::Points(points) => {
                Self::validate_virtual_point_coords(points, dims)?;
                Ok(VirtualSourceElementIndexes::Borrowed(points))
            }
            _ => Ok(VirtualSourceElementIndexes::Owned(
                Self::materialize_virtual_selection_linear_indexes(selection, dims, strides)?,
            )),
        }
    }

    fn materialize_virtual_selection_linear_indexes(
        selection: &VirtualSelection,
        dims: &[u64],
        strides: &[usize],
    ) -> Result<Vec<usize>> {
        let mut indexes = Vec::new();
        Self::visit_virtual_selection_points(selection, dims, |point| {
            indexes.push(Self::linear_index(point, strides)?);
            Ok(())
        })?;
        Ok(indexes)
    }

    fn copy_virtual_element(
        source_raw: &[u8],
        src_index: usize,
        dst_index: usize,
        element_size: usize,
        output: &mut [u8],
    ) -> Result<()> {
        let src_start = src_index
            .checked_mul(element_size)
            .ok_or_else(|| Error::InvalidFormat("virtual source byte offset overflow".into()))?;
        let dst_start = dst_index.checked_mul(element_size).ok_or_else(|| {
            Error::InvalidFormat("virtual destination byte offset overflow".into())
        })?;
        let Some(src) = checked_byte_window(
            source_raw,
            src_start,
            element_size,
            "virtual source byte range",
        )?
        else {
            return Ok(());
        };
        let Some(dst) = checked_byte_window_mut(
            output,
            dst_start,
            element_size,
            "virtual destination byte range",
        )?
        else {
            return Ok(());
        };
        dst.copy_from_slice(src);
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn virtual_output_dims(
        mappings: &[VirtualMapping],
        file_path: Option<&Path>,
        info: &DatasetInfo,
        access: &DatasetAccess,
    ) -> Result<Vec<u64>> {
        let mut source_cache = VirtualSourceCache::new();
        Self::virtual_output_dims_with_cache(mappings, file_path, info, access, &mut source_cache)
    }

    fn virtual_output_dims_with_cache(
        mappings: &[VirtualMapping],
        file_path: Option<&Path>,
        info: &DatasetInfo,
        access: &DatasetAccess,
        source_cache: &mut VirtualSourceCache,
    ) -> Result<Vec<u64>> {
        let mut output_dims = info.dataspace.dims.clone();
        let has_unlimited_dims = output_dims
            .iter()
            .enumerate()
            .any(|(dim, _)| Self::is_unlimited_vds_dim(info, dim));
        if output_dims.iter().all(|&dim| dim != 0) && !has_unlimited_dims {
            return Ok(output_dims);
        }
        for dim in 0..output_dims.len() {
            if Self::is_unlimited_vds_dim(info, dim) {
                output_dims[dim] = 0;
            }
        }
        let mut unlimited_extents: Vec<Option<(u64, u64)>> = vec![None; output_dims.len()];
        for mapping in mappings {
            let source_info = match source_cache.source_info(file_path, mapping, access) {
                Ok(source_info) => source_info,
                Err(err) if Self::should_fill_missing_virtual_source(&err, access) => {
                    for dim in 0..output_dims.len() {
                        if output_dims[dim] != 0 {
                            continue;
                        }
                        if let Some(extent) = Self::virtual_mapping_declared_output_extent(
                            &mapping.virtual_select,
                            dim,
                        )? {
                            Self::update_virtual_output_extent(
                                info,
                                &mut output_dims,
                                &mut unlimited_extents,
                                dim,
                                extent,
                            );
                        }
                    }
                    continue;
                }
                Err(err) => return Err(err),
            };
            for dim in 0..output_dims.len() {
                if output_dims[dim] != 0 {
                    continue;
                }
                let extent = Self::virtual_mapping_output_extent(
                    &mapping.source_select,
                    &mapping.virtual_select,
                    &source_info.dataspace.dims,
                    dim,
                )?;
                Self::update_virtual_output_extent(
                    info,
                    &mut output_dims,
                    &mut unlimited_extents,
                    dim,
                    extent,
                );
            }
        }
        for (dim, extents) in unlimited_extents.into_iter().enumerate() {
            if output_dims[dim] != 0 {
                continue;
            }
            if let Some((min_extent, max_extent)) = extents {
                output_dims[dim] = match access.virtual_view() {
                    VdsView::LastAvailable => max_extent,
                    VdsView::FirstMissing => min_extent,
                };
            }
        }
        Ok(output_dims)
    }

    fn update_virtual_output_extent(
        info: &DatasetInfo,
        output_dims: &mut [u64],
        unlimited_extents: &mut [Option<(u64, u64)>],
        dim: usize,
        extent: u64,
    ) {
        if Self::is_unlimited_vds_dim(info, dim) {
            let entry = &mut unlimited_extents[dim];
            match entry {
                Some((min_extent, max_extent)) => {
                    *min_extent = (*min_extent).min(extent);
                    *max_extent = (*max_extent).max(extent);
                }
                None => *entry = Some((extent, extent)),
            }
        } else {
            output_dims[dim] = output_dims[dim].max(extent);
        }
    }

    fn is_unlimited_vds_dim(info: &DatasetInfo, dim: usize) -> bool {
        info.dataspace
            .max_dims
            .as_ref()
            .and_then(|max_dims| max_dims.get(dim))
            .copied()
            == Some(u64::MAX)
    }

    pub(super) fn virtual_mapping_output_extent(
        source_select: &VirtualSelection,
        virtual_select: &VirtualSelection,
        source_dims: &[u64],
        dim: usize,
    ) -> Result<u64> {
        match virtual_select {
            VirtualSelection::All => Ok(source_dims[dim]),
            VirtualSelection::Points(points) => {
                let mut max_extent = 0u64;
                for point in points {
                    let extent = point[dim].checked_add(1).ok_or_else(|| {
                        Error::InvalidFormat("virtual point-selection extent overflow".into())
                    })?;
                    max_extent = max_extent.max(extent);
                }
                Ok(max_extent)
            }
            VirtualSelection::Regular(_) => Self::virtual_selection_start(virtual_select, dim)
                .checked_add(Self::virtual_selection_span(
                    source_select,
                    source_dims,
                    dim,
                )?)
                .ok_or_else(|| Error::InvalidFormat("virtual selection extent overflow".into())),
            VirtualSelection::Irregular(blocks) => {
                Ok(blocks.iter().try_fold(0u64, |max_extent, block| {
                    block.start[dim]
                        .checked_add(block.block[dim])
                        .map(|extent| max_extent.max(extent))
                        .ok_or_else(|| {
                            Error::InvalidFormat(
                                "virtual irregular-selection extent overflow".into(),
                            )
                        })
                })?)
            }
        }
    }

    pub(super) fn virtual_mapping_declared_output_extent(
        virtual_select: &VirtualSelection,
        dim: usize,
    ) -> Result<Option<u64>> {
        match virtual_select {
            VirtualSelection::All => Ok(None),
            VirtualSelection::Points(points) => {
                let mut max_extent = 0u64;
                for point in points {
                    let extent = point[dim].checked_add(1).ok_or_else(|| {
                        Error::InvalidFormat(
                            "virtual declared point-selection extent overflow".into(),
                        )
                    })?;
                    max_extent = max_extent.max(extent);
                }
                Ok(Some(max_extent))
            }
            VirtualSelection::Regular(selection) => {
                if selection.count[dim] == u64::MAX || selection.block[dim] == u64::MAX {
                    return Ok(None);
                }
                if selection.count[dim] == 0 {
                    return Ok(Some(selection.start[dim]));
                }
                let stride = selection.stride[dim].max(1);
                let selected_span = selection.count[dim]
                    .checked_sub(1)
                    .and_then(|count| count.checked_mul(stride))
                    .and_then(|offset| offset.checked_add(selection.block[dim]))
                    .ok_or_else(|| {
                        Error::InvalidFormat("virtual declared selection extent overflow".into())
                    })?;
                Ok(Some(
                    selection.start[dim]
                        .checked_add(selected_span)
                        .ok_or_else(|| {
                            Error::InvalidFormat(
                                "virtual declared selection extent overflow".into(),
                            )
                        })?,
                ))
            }
            VirtualSelection::Irregular(blocks) => {
                let mut max_extent = 0u64;
                for block in blocks {
                    let extent =
                        block.start[dim]
                            .checked_add(block.block[dim])
                            .ok_or_else(|| {
                                Error::InvalidFormat(
                                    "virtual declared irregular-selection extent overflow".into(),
                                )
                            })?;
                    max_extent = max_extent.max(extent);
                }
                Ok(Some(max_extent))
            }
        }
    }

    #[cfg(test)]
    pub(super) fn materialize_virtual_selection_points(
        selection: &VirtualSelection,
        dims: &[u64],
    ) -> Result<Vec<Vec<u64>>> {
        match selection {
            VirtualSelection::All => {
                let all = RegularHyperslab {
                    start: vec![0; dims.len()],
                    stride: vec![1; dims.len()],
                    count: vec![1; dims.len()],
                    block: dims.to_vec(),
                };
                Self::materialize_regular_hyperslab_points(&all, dims)
            }
            VirtualSelection::Points(points) => {
                Self::validate_virtual_point_coords(points, dims)?;
                Ok(points.clone())
            }
            VirtualSelection::Regular(selection) => {
                Self::materialize_regular_hyperslab_points(selection, dims)
            }
            VirtualSelection::Irregular(blocks) => {
                Self::materialize_irregular_hyperslab_points(blocks, dims)
            }
        }
    }

    fn visit_virtual_selection_points<F>(
        selection: &VirtualSelection,
        dims: &[u64],
        mut visit: F,
    ) -> Result<()>
    where
        F: FnMut(&[u64]) -> Result<()>,
    {
        match selection {
            VirtualSelection::All => {
                let mut current = vec![0u64; dims.len()];
                Self::visit_all_selection_points(dims, 0, &mut current, &mut visit)
            }
            VirtualSelection::Points(points) => {
                Self::validate_virtual_point_coords(points, dims)?;
                for point in points {
                    visit(point)?;
                }
                Ok(())
            }
            VirtualSelection::Regular(selection) => {
                let mut current = vec![0u64; dims.len()];
                Self::visit_regular_hyperslab_points(selection, dims, 0, &mut current, &mut visit)
            }
            VirtualSelection::Irregular(blocks) => {
                let mut current = vec![0u64; dims.len()];
                for block in blocks {
                    if block.start.len() != dims.len() || block.block.len() != dims.len() {
                        return Err(Error::InvalidFormat(
                            "virtual hyperslab rank does not match dataspace".into(),
                        ));
                    }
                    Self::visit_irregular_block_points(block, dims, 0, &mut current, &mut visit)?;
                }
                Ok(())
            }
        }
    }

    #[cfg(test)]
    fn materialize_regular_hyperslab_points(
        selection: &RegularHyperslab,
        dims: &[u64],
    ) -> Result<Vec<Vec<u64>>> {
        if selection.start.len() != dims.len() {
            return Err(Error::InvalidFormat(
                "virtual hyperslab rank does not match dataspace".into(),
            ));
        }
        let mut points = Vec::new();
        let mut current = vec![0u64; dims.len()];
        Self::push_hyperslab_points(selection, dims, 0, &mut current, &mut points)?;
        Ok(points)
    }

    fn visit_all_selection_points<F>(
        dims: &[u64],
        dim: usize,
        current: &mut [u64],
        visit: &mut F,
    ) -> Result<()>
    where
        F: FnMut(&[u64]) -> Result<()>,
    {
        if dim == dims.len() {
            return visit(current);
        }
        for coord in 0..dims[dim] {
            current[dim] = coord;
            Self::visit_all_selection_points(dims, dim + 1, current, visit)?;
        }
        Ok(())
    }

    pub(super) fn virtual_selection_start(selection: &VirtualSelection, dim: usize) -> u64 {
        match selection {
            VirtualSelection::All => 0,
            VirtualSelection::Points(points) => {
                points.iter().map(|point| point[dim]).min().unwrap_or(0)
            }
            VirtualSelection::Regular(selection) => selection.start[dim],
            VirtualSelection::Irregular(blocks) => blocks
                .iter()
                .map(|block| block.start[dim])
                .min()
                .unwrap_or(0),
        }
    }

    pub(super) fn virtual_selection_span(
        selection: &VirtualSelection,
        dims: &[u64],
        dim: usize,
    ) -> Result<u64> {
        match selection {
            VirtualSelection::All => Ok(dims[dim]),
            VirtualSelection::Points(points) => {
                if points.is_empty() {
                    return Ok(0);
                }
                let start = Self::virtual_selection_start(selection, dim);
                let end = points.iter().map(|point| point[dim]).max().ok_or_else(|| {
                    Error::InvalidFormat("virtual point-selection is empty".into())
                })?;
                end.checked_add(1)
                    .and_then(|extent| extent.checked_sub(start))
                    .ok_or_else(|| {
                        Error::InvalidFormat("virtual point-selection span overflow".into())
                    })
            }
            VirtualSelection::Regular(selection) => {
                Self::regular_hyperslab_selected_span(selection, dims, dim)
            }
            VirtualSelection::Irregular(blocks) => {
                if blocks.is_empty() {
                    return Ok(0);
                }
                let start = Self::virtual_selection_start(selection, dim);
                let end = blocks.iter().try_fold(0u64, |max_extent, block| {
                    block.start[dim]
                        .checked_add(block.block[dim])
                        .map(|extent| max_extent.max(extent))
                        .ok_or_else(|| {
                            Error::InvalidFormat("virtual irregular-selection span overflow".into())
                        })
                })?;
                end.checked_sub(start).ok_or_else(|| {
                    Error::InvalidFormat("virtual irregular-selection span overflow".into())
                })
            }
        }
    }

    fn regular_hyperslab_selected_span(
        selection: &RegularHyperslab,
        dims: &[u64],
        dim: usize,
    ) -> Result<u64> {
        let start = selection.start[dim];
        let stride = selection.stride[dim].max(1);
        let count = if selection.count[dim] == u64::MAX {
            let remaining = Self::virtual_hyperslab_remaining(dims[dim], start)?;
            Self::ceil_div_u64(remaining, stride, "virtual regular hyperslab span")?
        } else {
            selection.count[dim]
        };
        let block = if selection.block[dim] == u64::MAX {
            Self::virtual_hyperslab_remaining(dims[dim], start)?
        } else {
            selection.block[dim]
        };
        if count == 0 {
            Ok(0)
        } else {
            count
                .checked_sub(1)
                .and_then(|value| value.checked_mul(stride))
                .and_then(|value| value.checked_add(block))
                .ok_or_else(|| {
                    Error::InvalidFormat("virtual regular hyperslab span overflow".into())
                })
        }
    }

    #[cfg(test)]
    fn push_hyperslab_points(
        selection: &RegularHyperslab,
        dims: &[u64],
        dim: usize,
        current: &mut [u64],
        points: &mut Vec<Vec<u64>>,
    ) -> Result<()> {
        if dim == dims.len() {
            points.push(current.to_vec());
            return Ok(());
        }
        let start = selection.start[dim];
        let stride = selection.stride[dim].max(1);
        let count = if selection.count[dim] == u64::MAX {
            let remaining = Self::virtual_hyperslab_remaining(dims[dim], start)?;
            Self::ceil_div_u64(remaining, stride, "virtual hyperslab point count")?
        } else {
            selection.count[dim]
        };
        let block = if selection.block[dim] == u64::MAX {
            Self::virtual_hyperslab_remaining(dims[dim], start)?
        } else {
            selection.block[dim]
        };

        for count_idx in 0..count {
            let base = count_idx
                .checked_mul(stride)
                .and_then(|offset| start.checked_add(offset))
                .ok_or_else(|| {
                    Error::InvalidFormat("virtual hyperslab coordinate overflow".into())
                })?;
            for block_idx in 0..block {
                let coord = base.checked_add(block_idx).ok_or_else(|| {
                    Error::InvalidFormat("virtual hyperslab coordinate overflow".into())
                })?;
                if coord < dims[dim] {
                    current[dim] = coord;
                    Self::push_hyperslab_points(selection, dims, dim + 1, current, points)?;
                }
            }
        }
        Ok(())
    }

    fn visit_regular_hyperslab_points<F>(
        selection: &RegularHyperslab,
        dims: &[u64],
        dim: usize,
        current: &mut [u64],
        visit: &mut F,
    ) -> Result<()>
    where
        F: FnMut(&[u64]) -> Result<()>,
    {
        if selection.start.len() != dims.len() {
            return Err(Error::InvalidFormat(
                "virtual hyperslab rank does not match dataspace".into(),
            ));
        }
        if dim == dims.len() {
            return visit(current);
        }
        let start = selection.start[dim];
        let stride = selection.stride[dim].max(1);
        let count = if selection.count[dim] == u64::MAX {
            let remaining = Self::virtual_hyperslab_remaining(dims[dim], start)?;
            Self::ceil_div_u64(remaining, stride, "virtual hyperslab point count")?
        } else {
            selection.count[dim]
        };
        let block = if selection.block[dim] == u64::MAX {
            Self::virtual_hyperslab_remaining(dims[dim], start)?
        } else {
            selection.block[dim]
        };

        for count_idx in 0..count {
            let base = count_idx
                .checked_mul(stride)
                .and_then(|offset| start.checked_add(offset))
                .ok_or_else(|| {
                    Error::InvalidFormat("virtual hyperslab coordinate overflow".into())
                })?;
            for block_idx in 0..block {
                let coord = base.checked_add(block_idx).ok_or_else(|| {
                    Error::InvalidFormat("virtual hyperslab coordinate overflow".into())
                })?;
                if coord < dims[dim] {
                    current[dim] = coord;
                    Self::visit_regular_hyperslab_points(selection, dims, dim + 1, current, visit)?;
                }
            }
        }
        Ok(())
    }

    fn virtual_hyperslab_remaining(dim_extent: u64, start: u64) -> Result<u64> {
        dim_extent.checked_sub(start).ok_or_else(|| {
            Error::InvalidFormat("virtual regular hyperslab start exceeds dataspace extent".into())
        })
    }

    fn ceil_div_u64(value: u64, divisor: u64, context: &str) -> Result<u64> {
        if divisor == 0 {
            return Err(Error::InvalidFormat(format!("{context} divisor is zero")));
        }
        if value == 0 {
            return Ok(0);
        }
        value
            .checked_sub(1)
            .and_then(|v| v.checked_div(divisor))
            .and_then(|v| v.checked_add(1))
            .ok_or_else(|| Error::InvalidFormat(format!("{context} overflow")))
    }

    #[cfg(test)]
    fn materialize_irregular_hyperslab_points(
        blocks: &[IrregularHyperslabBlock],
        dims: &[u64],
    ) -> Result<Vec<Vec<u64>>> {
        let mut points = Vec::new();
        for block in blocks {
            if block.start.len() != dims.len() || block.block.len() != dims.len() {
                return Err(Error::InvalidFormat(
                    "virtual hyperslab rank does not match dataspace".into(),
                ));
            }
            let mut current = vec![0u64; dims.len()];
            Self::push_irregular_block_points(block, dims, 0, &mut current, &mut points)?;
        }
        Ok(points)
    }

    #[cfg(test)]
    fn push_irregular_block_points(
        block: &IrregularHyperslabBlock,
        dims: &[u64],
        dim: usize,
        current: &mut [u64],
        points: &mut Vec<Vec<u64>>,
    ) -> Result<()> {
        if dim == dims.len() {
            points.push(current.to_vec());
            return Ok(());
        }
        for offset in 0..block.block[dim] {
            let coord = block.start[dim].checked_add(offset).ok_or_else(|| {
                Error::InvalidFormat("virtual irregular hyperslab coordinate overflow".into())
            })?;
            if coord < dims[dim] {
                current[dim] = coord;
                Self::push_irregular_block_points(block, dims, dim + 1, current, points)?;
            }
        }
        Ok(())
    }

    fn visit_irregular_block_points<F>(
        block: &IrregularHyperslabBlock,
        dims: &[u64],
        dim: usize,
        current: &mut [u64],
        visit: &mut F,
    ) -> Result<()>
    where
        F: FnMut(&[u64]) -> Result<()>,
    {
        if dim == dims.len() {
            return visit(current);
        }
        for offset in 0..block.block[dim] {
            let coord = block.start[dim].checked_add(offset).ok_or_else(|| {
                Error::InvalidFormat("virtual irregular hyperslab coordinate overflow".into())
            })?;
            if coord < dims[dim] {
                current[dim] = coord;
                Self::visit_irregular_block_points(block, dims, dim + 1, current, visit)?;
            }
        }
        Ok(())
    }

    fn validate_virtual_point_coords(points: &[Vec<u64>], dims: &[u64]) -> Result<()> {
        for point in points {
            if point.len() != dims.len() {
                return Err(Error::InvalidFormat(
                    "virtual point-selection rank does not match dataspace".into(),
                ));
            }
            for (&coord, &dim_extent) in point.iter().zip(dims) {
                if coord >= dim_extent {
                    return Err(Error::InvalidFormat(
                        "virtual point-selection coordinate exceeds dataspace extent".into(),
                    ));
                }
            }
        }
        Ok(())
    }
}

fn checked_byte_window<'a>(
    data: &'a [u8],
    offset: usize,
    len: usize,
    context: &str,
) -> Result<Option<&'a [u8]>> {
    let end = offset
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} overflow")))?;
    Ok(data.get(offset..end))
}

fn checked_byte_window_mut<'a>(
    data: &'a mut [u8],
    offset: usize,
    len: usize,
    context: &str,
) -> Result<Option<&'a mut [u8]>> {
    let end = offset
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} overflow")))?;
    Ok(data.get_mut(offset..end))
}

fn is_missing_virtual_source_message(message: &str) -> bool {
    (message.starts_with("link '") && message.ends_with("' not found"))
        || message.starts_with("hard link target '")
        || message.contains("missing source dataset")
}

fn parsed_virtual_block_number(
    source_name: &str,
    parsed: &VirtualParsedName,
    candidate: &str,
) -> Result<Option<u64>> {
    if parsed.substitutions == 0 {
        let mut unescaped = String::new();
        H5D__virtual_build_source_name_into(source_name, Some(parsed), 0, &mut unescaped)?;
        return Ok((candidate == unescaped).then_some(0));
    }
    if parsed.segments.len() != parsed.substitutions + 1 {
        return Err(Error::InvalidFormat(
            "VDS parsed source-name segment count does not match substitutions".into(),
        ));
    }

    let mut tail = candidate;
    let mut block_number = None;
    for idx in 0..parsed.substitutions {
        let segment = parsed.segments[idx].as_str();
        let Some(rest) = tail.strip_prefix(segment) else {
            return Ok(None);
        };
        let next_segment = parsed.segments[idx + 1].as_str();
        let next_pos = if next_segment.is_empty() {
            rest.len()
        } else {
            let Some(next_pos) = rest.find(next_segment) else {
                return Ok(None);
            };
            next_pos
        };
        let digits = &rest[..next_pos];
        if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
            return Ok(None);
        }
        let parsed_block = digits.parse::<u64>().map_err(|err| {
            Error::InvalidFormat(format!("VDS source-name block number overflow: {err}"))
        })?;
        if block_number
            .replace(parsed_block)
            .is_some_and(|block| block != parsed_block)
        {
            return Ok(None);
        }
        tail = &rest[next_pos..];
    }

    let Some(last_segment) = parsed.segments.last() else {
        return Ok(None);
    };
    if tail == last_segment {
        Ok(block_number)
    } else {
        Ok(None)
    }
}

fn expand_virtual_origin_prefix(file_path: Option<&Path>, prefix: &Path) -> Result<PathBuf> {
    if let Some(prefix_str) = prefix.to_str() {
        if prefix_str.starts_with("${ORIGIN}") {
            return expand_virtual_origin_prefix_str(file_path, prefix_str);
        }
    }
    Ok(prefix.to_path_buf())
}

fn expand_virtual_origin_prefix_str(file_path: Option<&Path>, prefix: &str) -> Result<PathBuf> {
    const ORIGIN: &str = "${ORIGIN}";

    if let Some(rest) = prefix.strip_prefix(ORIGIN) {
        let origin_dir = file_path
            .and_then(Path::parent)
            .map(Path::to_path_buf)
            .ok_or_else(|| {
                Error::Unsupported("VDS ${ORIGIN} prefix has no base file path".into())
            })?;
        let trimmed = rest.strip_prefix(['/', '\\']).unwrap_or(rest);
        if trimmed.is_empty() {
            return Ok(origin_dir);
        }
        return Ok(origin_dir.join(trimmed));
    }

    Ok(PathBuf::from(prefix))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_byte_window_rejects_offset_overflow() {
        let err = checked_byte_window(&[], usize::MAX, 1, "virtual test range").unwrap_err();
        assert!(
            err.to_string().contains("virtual test range overflow"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn checked_byte_window_mut_rejects_offset_overflow() {
        let mut data = [];
        let err =
            checked_byte_window_mut(&mut data, usize::MAX, 1, "virtual test range").unwrap_err();
        assert!(
            err.to_string().contains("virtual test range overflow"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn missing_virtual_source_message_matches_path_resolution_only() {
        assert!(is_missing_virtual_source_message("link 'source' not found"));
        assert!(is_missing_virtual_source_message(
            "hard link target '/source' not found"
        ));
        assert!(!is_missing_virtual_source_message(
            "global heap object 9 not found in collection at 0x0"
        ));
        assert!(!is_missing_virtual_source_message(
            "fractal heap offset 42 not found in indirect block"
        ));
    }

    #[test]
    fn parsed_virtual_block_number_matches_trailing_substitution() {
        let parsed = H5D_virtual_parse_source_name("tile-%b")
            .unwrap()
            .expect("source name should parse");
        assert_eq!(
            parsed_virtual_block_number("tile-%b", &parsed, "tile-37").unwrap(),
            Some(37)
        );
        assert_eq!(
            parsed_virtual_block_number("tile-%b", &parsed, "tile-").unwrap(),
            None
        );
    }
}
