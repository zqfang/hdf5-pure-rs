use std::path::Path;

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

pub(super) struct VirtualSourceData {
    pub(super) info: DatasetInfo,
    pub(super) raw: Vec<u8>,
}

pub(super) struct VirtualPointMap {
    pub(super) source_points: Vec<Vec<u64>>,
    pub(super) virtual_points: Vec<Vec<u64>>,
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
    pub(crate) fn virtual_mapping_infos_with_info(
        &self,
        info: &DatasetInfo,
    ) -> Result<Vec<VirtualMappingInfo>> {
        if info.layout.layout_class != LayoutClass::Virtual {
            return Ok(Vec::new());
        }
        let mut guard = self.inner.lock();
        let heap_addr = info.layout.virtual_heap_addr.ok_or_else(|| {
            Error::InvalidFormat("virtual dataset missing global heap address".into())
        })?;
        let heap_index = info.layout.virtual_heap_index.ok_or_else(|| {
            Error::InvalidFormat("virtual dataset missing global heap index".into())
        })?;
        let heap_data = crate::format::global_heap::read_global_heap_object(
            &mut guard.reader,
            &crate::format::global_heap::GlobalHeapRef {
                collection_addr: heap_addr,
                object_index: heap_index,
            },
        )?;
        let sizeof_size = usize::from(guard.reader.sizeof_size());
        drop(guard);

        let mappings = Self::decode_virtual_mappings(&heap_data, sizeof_size)?;
        Ok(mappings
            .into_iter()
            .map(Self::virtual_mapping_info_from_mapping)
            .collect())
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
        let heap_data = crate::format::global_heap::read_global_heap_object(
            &mut guard.reader,
            &crate::format::global_heap::GlobalHeapRef {
                collection_addr: heap_addr,
                object_index: heap_index,
            },
        )?;
        let sizeof_size = usize::from(guard.reader.sizeof_size());
        drop(guard);

        let mappings = Self::decode_virtual_mappings(&heap_data, sizeof_size)?;
        Self::virtual_output_dims(&mappings, path.as_deref(), info, access)
    }

    pub(super) fn read_virtual_dataset(
        heap_data: &[u8],
        sizeof_size: usize,
        file_path: Option<&Path>,
        info: &DatasetInfo,
        access: &DatasetAccess,
    ) -> Result<Vec<u8>> {
        let mappings = Self::decode_virtual_mappings(heap_data, sizeof_size)?;
        let element_size = usize_from_u64(u64::from(info.datatype.size), "datatype size")?;

        let output_dims = Self::virtual_output_dims(&mappings, file_path, info, access)?;
        let total_elements = usize_from_u64(
            Self::dataspace_element_count(info.dataspace.space_type, &output_dims)?,
            "virtual dataset element count",
        )?;
        let mut output = Self::filled_data(total_elements, element_size, info)?;
        let virtual_strides = Self::row_major_strides(&output_dims)?;

        for mapping in mappings {
            let source = match Self::open_virtual_source_dataset(file_path, &mapping, info, access)
            {
                Ok(source) => source,
                Err(err) if Self::should_fill_missing_virtual_source(&err, access) => {
                    continue;
                }
                Err(err) => return Err(err),
            };
            let point_map = Self::materialize_virtual_point_map(
                &mapping,
                &source.info.dataspace.dims,
                &output_dims,
            )?;
            Self::copy_virtual_mapping(
                &source.raw,
                &source.info.dataspace.dims,
                &virtual_strides,
                &point_map,
                element_size,
                &mut output,
            )?;
        }

        Ok(output)
    }

    fn should_fill_missing_virtual_source(err: &Error, access: &DatasetAccess) -> bool {
        access.virtual_missing_source_policy() == VdsMissingSourcePolicy::Fill
            && matches!(err, Error::Io(io_err) if io_err.kind() == std::io::ErrorKind::NotFound)
    }

    fn open_virtual_source_dataset(
        file_path: Option<&Path>,
        mapping: &VirtualMapping,
        dest_info: &DatasetInfo,
        access: &DatasetAccess,
    ) -> Result<VirtualSourceData> {
        let source_file = Self::resolve_virtual_source_path(file_path, &mapping.file_name, access)?;
        let source = crate::hl::file::File::open(&source_file)?;
        let source_ds = source.dataset(&mapping.dataset_name)?;
        let source_info = source_ds.info()?;
        let source_raw = source_ds.read_raw_with_access(access)?;
        let source_raw = crate::hl::conversion::convert_between_datatypes(
            &source_raw,
            &source_info.datatype,
            &dest_info.datatype,
        )?;
        Ok(VirtualSourceData {
            info: source_info,
            raw: source_raw,
        })
    }

    fn materialize_virtual_point_map(
        mapping: &VirtualMapping,
        source_dims: &[u64],
        output_dims: &[u64],
    ) -> Result<VirtualPointMap> {
        let source_points =
            Self::materialize_virtual_selection_points(&mapping.source_select, source_dims)?;
        let virtual_points =
            Self::materialize_virtual_selection_points(&mapping.virtual_select, output_dims)?;
        if source_points.len() != virtual_points.len() {
            return Err(Error::InvalidFormat(
                "virtual dataset source and destination selections differ in size".into(),
            ));
        }
        Ok(VirtualPointMap {
            source_points,
            virtual_points,
        })
    }

    fn copy_virtual_mapping(
        source_raw: &[u8],
        source_dims: &[u64],
        virtual_strides: &[usize],
        point_map: &VirtualPointMap,
        element_size: usize,
        output: &mut [u8],
    ) -> Result<()> {
        let source_strides = Self::row_major_strides(source_dims)?;
        for (src, dst) in point_map
            .source_points
            .iter()
            .zip(point_map.virtual_points.iter())
        {
            let src_index = Self::linear_index(src, &source_strides)?;
            let dst_index = Self::linear_index(dst, virtual_strides)?;
            let src_start = src_index.checked_mul(element_size).ok_or_else(|| {
                Error::InvalidFormat("virtual source byte offset overflow".into())
            })?;
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
                continue;
            };
            let Some(dst) = checked_byte_window_mut(
                output,
                dst_start,
                element_size,
                "virtual destination byte range",
            )?
            else {
                continue;
            };
            dst.copy_from_slice(src);
        }
        Ok(())
    }

    pub(super) fn virtual_output_dims(
        mappings: &[VirtualMapping],
        file_path: Option<&Path>,
        info: &DatasetInfo,
        access: &DatasetAccess,
    ) -> Result<Vec<u64>> {
        let mut output_dims = info.dataspace.dims.clone();
        if output_dims.iter().all(|&dim| dim != 0) {
            return Ok(output_dims);
        }
        let mut unlimited_extents: Vec<Option<(u64, u64)>> = vec![None; output_dims.len()];
        for mapping in mappings {
            let source_file =
                Self::resolve_virtual_source_path(file_path, &mapping.file_name, access)?;
            let source = match crate::hl::file::File::open(&source_file) {
                Ok(source) => source,
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
            let source_info = source.dataset(&mapping.dataset_name)?.info()?;
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
}
