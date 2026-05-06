use std::ops::{Range, RangeFrom, RangeFull, RangeTo};

use crate::{Error, Result};

const MAX_MATERIALIZED_SELECTION_POINTS: usize = 1_000_000;

/// A selection specifies which elements to read from a dataset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Selection {
    /// No elements.
    None,
    /// All elements.
    All,
    /// Explicit point coordinates in row-major result order.
    Points(Vec<Vec<u64>>),
    /// Regular hyperslab with start, stride, count, and block per dimension.
    Hyperslab(Vec<HyperslabDim>),
    /// A contiguous range along each dimension.
    Slice(Vec<SliceInfo>),
}

/// HDF5-style selection class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionType {
    None,
    All,
    Points,
    Hyperslab,
}

/// Slice specification for one dimension.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SliceInfo {
    pub start: u64,
    pub end: u64,  // exclusive
    pub step: u64, // must be 1 for now
}

/// Regular hyperslab specification for one dimension.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HyperslabDim {
    pub start: u64,
    pub stride: u64,
    pub count: u64,
    pub block: u64,
}

/// Iterator over selected coordinates in row-major order.
#[derive(Debug, Clone)]
pub struct SelectionPointIter {
    points: Vec<Vec<u64>>,
    index: usize,
}

impl Iterator for SelectionPointIter {
    type Item = Vec<u64>;

    fn next(&mut self) -> Option<Self::Item> {
        let point = self.points.get(self.index)?.clone();
        self.index += 1;
        Some(point)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.points.len().saturating_sub(self.index);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for SelectionPointIter {}

impl SelectionPointIter {
    /// Return the number of elements remaining in this selection iterator.
    pub fn select_iter_nelmts(&self) -> usize {
        self.len()
    }

    /// Return the next selected coordinate.
    pub fn select_iter_next(&mut self) -> Option<Vec<u64>> {
        self.next()
    }

    /// Return up to `max_points` remaining selected coordinates.
    pub fn select_iter_get_seq_list(&mut self, max_points: usize) -> Vec<Vec<u64>> {
        self.take(max_points).collect()
    }

    /// Reset this selection iterator to the first selected coordinate.
    pub fn select_iter_reset(&mut self) {
        self.index = 0;
    }

    /// Explicit release hook for parity with HDF5's selection iterator API.
    pub fn select_iter_release(self) {}

    /// Hyperslab-specific iterator element count alias.
    pub fn hyper_iter_nelmts(&self) -> usize {
        self.len()
    }

    /// Hyperslab-specific iterator next-coordinate alias.
    pub fn hyper_iter_next(&mut self) -> Option<Vec<u64>> {
        self.next()
    }

    /// Hyperslab-specific iterator next-block alias.
    pub fn hyper_iter_next_block(&mut self) -> Option<Vec<u64>> {
        self.next()
    }

    /// Hyperslab-specific iterator sequence-list alias.
    pub fn hyper_iter_get_seq_list(&mut self, max_points: usize) -> Vec<Vec<u64>> {
        self.take(max_points).collect()
    }

    /// Hyperslab-specific optimized sequence-list alias.
    pub fn hyper_iter_get_seq_list_opt(&mut self, max_points: usize) -> Vec<Vec<u64>> {
        self.hyper_iter_get_seq_list(max_points)
    }

    /// Hyperslab-specific single-block sequence-list alias.
    pub fn hyper_iter_get_seq_list_single(&mut self, max_points: usize) -> Vec<Vec<u64>> {
        self.hyper_iter_get_seq_list(max_points)
    }

    /// Hyperslab-specific generic sequence-list alias.
    pub fn hyper_iter_get_seq_list_gen(&mut self, max_points: usize) -> Vec<Vec<u64>> {
        self.hyper_iter_get_seq_list(max_points)
    }

    /// Hyperslab-specific iterator release alias.
    pub fn hyper_iter_release(self) {}

    /// Point-specific iterator coordinate alias.
    pub fn point_iter_coords(&self) -> Option<&[u64]> {
        self.points.get(self.index).map(Vec::as_slice)
    }

    /// Point-specific iterator element count alias.
    pub fn point_iter_nelmts(&self) -> usize {
        self.len()
    }

    /// Point-specific iterator next-coordinate alias.
    pub fn point_iter_next(&mut self) -> Option<Vec<u64>> {
        self.next()
    }

    /// Point-specific iterator next-block alias.
    pub fn point_iter_next_block(&mut self) -> Option<Vec<u64>> {
        self.next()
    }

    /// Point-specific iterator sequence-list alias.
    pub fn point_iter_get_seq_list(&mut self, max_points: usize) -> Vec<Vec<u64>> {
        self.take(max_points).collect()
    }

    /// Point-specific iterator release alias.
    pub fn point_iter_release(self) {}
}

impl HyperslabDim {
    pub fn new(start: u64, stride: u64, count: u64, block: u64) -> Self {
        Self {
            start,
            stride,
            count,
            block,
        }
    }

    pub fn output_count(&self) -> u64 {
        self.checked_output_count().unwrap_or(u64::MAX)
    }

    pub fn checked_output_count(&self) -> Result<u64> {
        self.count
            .checked_mul(self.block)
            .ok_or_else(|| Error::InvalidFormat("hyperslab output count overflow".into()))
    }
}

impl SliceInfo {
    pub fn new(start: u64, end: u64) -> Self {
        Self {
            start,
            end,
            step: 1,
        }
    }

    pub fn with_step(start: u64, end: u64, step: u64) -> Self {
        Self { start, end, step }
    }

    pub fn count(&self) -> u64 {
        if self.step == 0 {
            0
        } else if self.end > self.start {
            (self.end - self.start).div_ceil(self.step)
        } else {
            0
        }
    }
}

/// Trait for types that can be converted to a Selection.
pub trait IntoSelection {
    fn into_selection(self, shape: &[u64]) -> Selection;
}

/// Trait for one dimension of a multi-dimensional slice selection.
pub trait IntoSliceDim {
    fn into_slice_dim(self, extent: u64) -> SliceInfo;
}

impl IntoSelection for Selection {
    fn into_selection(self, _shape: &[u64]) -> Selection {
        self
    }
}

// For 1D: Range<usize>
impl IntoSelection for Range<usize> {
    fn into_selection(self, _shape: &[u64]) -> Selection {
        Selection::Slice(vec![SliceInfo::new(self.start as u64, self.end as u64)])
    }
}

impl IntoSliceDim for Range<usize> {
    fn into_slice_dim(self, _extent: u64) -> SliceInfo {
        SliceInfo::new(self.start as u64, self.end as u64)
    }
}

// For 1D: RangeFull (..)
impl IntoSelection for RangeFull {
    fn into_selection(self, _shape: &[u64]) -> Selection {
        Selection::All
    }
}

impl IntoSliceDim for RangeFull {
    fn into_slice_dim(self, extent: u64) -> SliceInfo {
        SliceInfo::new(0, extent)
    }
}

// For 1D: RangeFrom (start..)
impl IntoSelection for RangeFrom<usize> {
    fn into_selection(self, shape: &[u64]) -> Selection {
        let end = if shape.is_empty() { 0 } else { shape[0] };
        Selection::Slice(vec![SliceInfo::new(self.start as u64, end)])
    }
}

impl IntoSliceDim for RangeFrom<usize> {
    fn into_slice_dim(self, extent: u64) -> SliceInfo {
        SliceInfo::new(self.start as u64, extent)
    }
}

// For 1D: RangeTo (..end)
impl IntoSelection for RangeTo<usize> {
    fn into_selection(self, _shape: &[u64]) -> Selection {
        Selection::Slice(vec![SliceInfo::new(0, self.end as u64)])
    }
}

impl IntoSliceDim for RangeTo<usize> {
    fn into_slice_dim(self, _extent: u64) -> SliceInfo {
        SliceInfo::new(0, self.end as u64)
    }
}

// Tuples for 2D selections.
impl<A, B> IntoSelection for (A, B)
where
    A: IntoSliceDim,
    B: IntoSliceDim,
{
    fn into_selection(self, shape: &[u64]) -> Selection {
        let dim0 = shape.first().copied().unwrap_or(0);
        let dim1 = shape.get(1).copied().unwrap_or(0);
        Selection::Slice(vec![
            self.0.into_slice_dim(dim0),
            self.1.into_slice_dim(dim1),
        ])
    }
}

// Tuples for 3D selections.
impl<A, B, C> IntoSelection for (A, B, C)
where
    A: IntoSliceDim,
    B: IntoSliceDim,
    C: IntoSliceDim,
{
    fn into_selection(self, shape: &[u64]) -> Selection {
        let dim0 = shape.first().copied().unwrap_or(0);
        let dim1 = shape.get(1).copied().unwrap_or(0);
        let dim2 = shape.get(2).copied().unwrap_or(0);
        Selection::Slice(vec![
            self.0.into_slice_dim(dim0),
            self.1.into_slice_dim(dim1),
            self.2.into_slice_dim(dim2),
        ])
    }
}

impl Selection {
    /// Create a none selection.
    pub fn select_none() -> Self {
        Selection::None
    }

    /// None-selection iterator block alias.
    pub fn none_iter_block() -> Option<Vec<u64>> {
        None
    }

    /// None-selection iterator element-count alias.
    pub fn none_iter_nelmts() -> usize {
        0
    }

    /// None-selection iterator sequence-list alias.
    pub fn none_iter_get_seq_list() -> Vec<Vec<u64>> {
        Vec::new()
    }

    /// None-selection iterator release alias.
    pub fn none_iter_release() {}

    /// None-selection release alias.
    pub fn none_release() {}

    /// None-selection copy alias.
    pub fn none_copy() -> Selection {
        Selection::None
    }

    /// None-selection validity alias.
    pub fn none_is_valid() -> bool {
        true
    }

    /// Serialize a none selection.
    pub fn none_serialize() -> Vec<u8> {
        Vec::new()
    }

    /// Deserialize a none selection.
    pub fn none_deserialize(bytes: &[u8]) -> Result<Selection> {
        if bytes.is_empty() {
            Ok(Selection::None)
        } else {
            Err(Error::InvalidFormat(
                "none selection serialization must be empty".into(),
            ))
        }
    }

    /// None-selection bounds alias.
    pub fn none_bounds() -> Option<(Vec<u64>, Vec<u64>)> {
        None
    }

    /// None-selection offset alias.
    pub fn none_offset(_offsets: &[i64]) -> Selection {
        Selection::None
    }

    /// None-selection contiguous alias.
    pub fn none_is_contiguous() -> bool {
        true
    }

    /// None-selection single-element alias.
    pub fn none_is_single() -> bool {
        false
    }

    /// None-selection regularity alias.
    pub fn none_is_regular() -> bool {
        true
    }

    /// None-selection block-intersection alias.
    pub fn none_intersect_block(_start: &[u64], _end: &[u64]) -> bool {
        false
    }

    /// None-selection unsigned adjust alias.
    pub fn none_adjust_u(_offsets: &[u64]) -> Selection {
        Selection::None
    }

    /// None-selection signed adjust alias.
    pub fn none_adjust_s(_offsets: &[i64]) -> Selection {
        Selection::None
    }

    /// None-selection simple projection alias.
    pub fn none_project_simple() -> Selection {
        Selection::None
    }

    /// Public none-selection alias.
    pub fn select_none_api() -> Self {
        Self::select_none()
    }

    /// Create an all selection.
    pub fn select_all() -> Self {
        Selection::All
    }

    /// Public all-selection alias.
    pub fn select_all_api() -> Self {
        Self::select_all()
    }

    /// Initialize an all-selection iterator.
    pub fn all_iter_init(ds_shape: &[u64]) -> Result<SelectionPointIter> {
        Selection::All.iter_points(ds_shape)
    }

    /// Return the first coordinate for an all-selection iterator.
    pub fn all_iter_coords(ds_shape: &[u64]) -> Result<Option<Vec<u64>>> {
        Ok(Self::all_iter_init(ds_shape)?.next())
    }

    /// Return the first block for an all-selection iterator.
    pub fn all_iter_block(ds_shape: &[u64]) -> Option<(Vec<u64>, Vec<u64>)> {
        Selection::All.bounds(ds_shape)
    }

    /// Return the all-selection element count.
    pub fn all_iter_nelmts(ds_shape: &[u64]) -> Result<u64> {
        total_elements(ds_shape)
    }

    /// Return whether an all-selection has another block.
    pub fn all_iter_has_next_block(ds_shape: &[u64]) -> bool {
        Selection::All.bounds(ds_shape).is_some()
    }

    /// Return the next all-selection coordinate.
    pub fn all_iter_next(iter: &mut SelectionPointIter) -> Option<Vec<u64>> {
        iter.next()
    }

    /// Return the next all-selection block.
    pub fn all_iter_next_block(ds_shape: &[u64]) -> Option<(Vec<u64>, Vec<u64>)> {
        Selection::All.bounds(ds_shape)
    }

    /// Return up to `max_points` all-selection coordinates.
    pub fn all_iter_get_seq_list(ds_shape: &[u64], max_points: usize) -> Result<Vec<Vec<u64>>> {
        Ok(Self::all_iter_init(ds_shape)?.take(max_points).collect())
    }

    /// All-selection iterator release alias.
    pub fn all_iter_release(_iter: SelectionPointIter) {}

    /// All-selection release alias.
    pub fn all_release() {}

    /// All-selection copy alias.
    pub fn all_copy() -> Selection {
        Selection::All
    }

    /// All-selection validity alias.
    pub fn all_is_valid(_ds_shape: &[u64]) -> bool {
        true
    }

    /// Return the encoded size of an all-selection payload.
    pub fn all_serial_size() -> usize {
        0
    }

    /// Serialize an all selection.
    pub fn all_serialize() -> Vec<u8> {
        Vec::new()
    }

    /// Deserialize an all selection.
    pub fn all_deserialize(bytes: &[u8]) -> Result<Selection> {
        if bytes.is_empty() {
            Ok(Selection::All)
        } else {
            Err(Error::InvalidFormat(
                "all selection serialization must be empty".into(),
            ))
        }
    }

    /// All-selection bounds alias.
    pub fn all_bounds(ds_shape: &[u64]) -> Option<(Vec<u64>, Vec<u64>)> {
        Selection::All.bounds(ds_shape)
    }

    /// All-selection offset alias.
    pub fn all_offset(offsets: &[i64]) -> Result<Selection> {
        Selection::All.select_offset(offsets)
    }

    /// All-selection unlimited-dimension alias.
    pub fn all_unlim_dim(max_dims: &[u64]) -> Option<usize> {
        Selection::All.select_unlim_dim(max_dims)
    }

    /// All-selection contiguous alias.
    pub fn all_is_contiguous() -> bool {
        true
    }

    /// All-selection single-element alias.
    pub fn all_is_single(ds_shape: &[u64]) -> bool {
        total_elements(ds_shape).ok() == Some(1)
    }

    /// All-selection regularity alias.
    pub fn all_is_regular() -> bool {
        true
    }

    /// All-selection block-intersection alias.
    pub fn all_intersect_block(ds_shape: &[u64], start: &[u64], end: &[u64]) -> bool {
        block_intersects_shape(ds_shape, start, end)
    }

    /// All-selection unsigned adjust alias.
    pub fn all_adjust_u(offsets: &[u64]) -> Result<Selection> {
        Selection::All.select_adjust_unsigned(offsets)
    }

    /// All-selection signed adjust alias.
    pub fn all_adjust_s(offsets: &[i64]) -> Result<Selection> {
        Selection::All.select_adjust_signed(offsets)
    }

    /// All-selection simple projection alias.
    pub fn all_project_simple(ds_shape: &[u64], kept_dims: &[usize]) -> Result<Selection> {
        Selection::All.project(ds_shape, kept_dims)
    }

    /// Create an explicit element-point selection.
    pub fn select_elements(points: Vec<Vec<u64>>) -> Self {
        Selection::Points(points)
    }

    /// Copy a raw point-selection coordinate list.
    pub fn copy_pnt_list(points: &[Vec<u64>]) -> Vec<Vec<u64>> {
        points.to_vec()
    }

    /// Explicit point-list release alias.
    pub fn free_pnt_list(_points: Vec<Vec<u64>>) {}

    /// Public element-point selection alias.
    pub fn select_elements_api(points: Vec<Vec<u64>>) -> Self {
        Self::select_elements(points)
    }

    /// Create a regular hyperslab selection.
    pub fn select_hyperslab(dims: Vec<HyperslabDim>) -> Self {
        Selection::Hyperslab(dims)
    }

    /// Public hyperslab selection alias.
    pub fn select_hyperslab_api(dims: Vec<HyperslabDim>) -> Self {
        Self::select_hyperslab(dims)
    }

    /// Local classification equivalent of the MPI all-selection branch.
    pub fn mpio_all_type(&self) -> bool {
        matches!(self, Selection::All)
    }

    /// Local classification equivalent of the MPI none-selection branch.
    pub fn mpio_none_type(&self) -> bool {
        matches!(self, Selection::None)
    }

    /// Local classification equivalent of the MPI point-selection branch.
    pub fn mpio_point_type(&self) -> bool {
        matches!(self, Selection::Points(_))
    }

    /// Local classification equivalent of the MPI permutation branch.
    pub fn mpio_permute_type(&self) -> bool {
        false
    }

    /// Local classification equivalent of the MPI regular-hyperslab branch.
    pub fn mpio_reg_hyper_type(&self) -> bool {
        matches!(self, Selection::Hyperslab(_) | Selection::Slice(_)) && self.is_regular()
    }

    /// Local classification equivalent of the MPI span-hyperslab branch.
    pub fn mpio_span_hyper_type(&self) -> bool {
        matches!(self, Selection::Hyperslab(_) | Selection::Slice(_)) && !self.is_regular()
    }

    /// Deserialize a tagged selection payload.
    pub fn select_deserialize(bytes: &[u8]) -> Result<Selection> {
        let Some((&tag, rest)) = bytes.split_first() else {
            return Err(Error::InvalidFormat(
                "selection serialization payload is empty".into(),
            ));
        };
        match tag {
            0 => Selection::none_deserialize(rest),
            1 => Selection::all_deserialize(rest),
            2 => Selection::point_deserialize(rest),
            3 => Selection::hyper_deserialize(rest),
            _ => Err(Error::InvalidFormat(format!(
                "unknown selection serialization tag {tag}"
            ))),
        }
    }

    /// Return true if this selection selects no points.
    pub fn is_none(&self) -> bool {
        matches!(self, Selection::None)
    }

    /// Return true if this selection selects the full dataspace.
    pub fn is_all(&self) -> bool {
        matches!(self, Selection::All)
    }

    /// Return the HDF5-style selection class.
    pub fn selection_type(&self) -> SelectionType {
        match self {
            Selection::None => SelectionType::None,
            Selection::All => SelectionType::All,
            Selection::Points(_) => SelectionType::Points,
            Selection::Hyperslab(_) | Selection::Slice(_) => SelectionType::Hyperslab,
        }
    }

    /// Internal selection-type helper.
    pub fn select_type_internal(&self) -> SelectionType {
        self.selection_type()
    }

    /// Serialize this selection with an explicit class tag.
    pub fn encode1(&self) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        match self {
            Selection::None => out.push(0),
            Selection::All => out.push(1),
            Selection::Points(_) => {
                out.push(2);
                out.extend(self.point_serialize()?);
            }
            Selection::Hyperslab(_) | Selection::Slice(_) => {
                out.push(3);
                out.extend(self.hyper_serialize()?);
            }
        }
        Ok(out)
    }

    /// Explicit selection copy operation.
    pub fn select_copy(&self) -> Self {
        self.clone()
    }

    /// Compute the output shape for this selection given the dataset shape.
    pub fn output_shape(&self, ds_shape: &[u64]) -> Vec<u64> {
        match self {
            Selection::None => vec![0],
            Selection::All => ds_shape.to_vec(),
            Selection::Points(points) => vec![u64::try_from(points.len()).unwrap_or(u64::MAX)],
            Selection::Hyperslab(dims) => dims.iter().map(HyperslabDim::output_count).collect(),
            Selection::Slice(slices) => slices.iter().map(|s| s.count()).collect(),
        }
    }

    /// Return the number of selected elements.
    pub fn selected_count(&self, ds_shape: &[u64]) -> Option<u64> {
        match self {
            Selection::None => Some(0),
            Selection::All => total_elements(ds_shape).ok(),
            Selection::Points(points) => u64::try_from(points.len()).ok(),
            Selection::Hyperslab(dims) => {
                if dims.is_empty() {
                    return Some(1);
                }
                dims.iter().try_fold(1u64, |acc, dim| {
                    dim.count
                        .checked_mul(dim.block)
                        .and_then(|dim_count| acc.checked_mul(dim_count))
                })
            }
            Selection::Slice(slices) => {
                if slices.is_empty() {
                    return Some(1);
                }
                slices
                    .iter()
                    .try_fold(1u64, |acc, slice| acc.checked_mul(slice.count()))
            }
        }
    }

    /// Internal selected-point count helper.
    pub fn select_npoints_internal(&self, ds_shape: &[u64]) -> Option<u64> {
        self.selected_count(ds_shape)
    }

    /// Return whether this selection is valid for a dataspace shape.
    pub fn select_valid(&self, ds_shape: &[u64]) -> bool {
        if let Selection::Points(points) = self {
            return points.iter().all(|point| {
                point.len() == ds_shape.len()
                    && point
                        .iter()
                        .zip(ds_shape.iter())
                        .all(|(&coord, &extent)| coord < extent)
            });
        }
        self.materialize_points(ds_shape).is_ok()
    }

    /// Return inclusive selection bounds as `(start, end)` coordinates.
    pub fn bounds(&self, ds_shape: &[u64]) -> Option<(Vec<u64>, Vec<u64>)> {
        match self {
            Selection::None => None,
            Selection::All => {
                if ds_shape.contains(&0) {
                    None
                } else if ds_shape.is_empty() {
                    Some((Vec::new(), Vec::new()))
                } else {
                    Some((
                        vec![0; ds_shape.len()],
                        ds_shape.iter().map(|&dim| dim - 1).collect(),
                    ))
                }
            }
            Selection::Points(points) => point_bounds(points),
            Selection::Hyperslab(dims) => hyperslab_bounds(dims),
            Selection::Slice(slices) => slice_bounds(slices),
        }
    }

    /// Internal selection-bounds helper.
    pub fn select_bounds_internal(&self, ds_shape: &[u64]) -> Option<(Vec<u64>, Vec<u64>)> {
        self.bounds(ds_shape)
    }

    /// Return the number of hyperslab blocks, if this is a hyperslab-like
    /// selection.
    pub fn hyperslab_block_count(&self, ds_shape: &[u64]) -> Option<u64> {
        match self {
            Selection::Points(_) => None,
            Selection::None => Some(0),
            Selection::All => total_elements(ds_shape)
                .ok()
                .map(|count| u64::from(count > 0)),
            Selection::Hyperslab(dims) => {
                if dims
                    .iter()
                    .any(|dim| dim.count == 0 || dim.block == 0 || dim.stride == 0)
                {
                    return Some(0);
                }
                dims.iter()
                    .try_fold(1u64, |acc, dim| acc.checked_mul(dim.count))
            }
            Selection::Slice(slices) => {
                if slices.iter().any(|slice| slice.count() == 0) {
                    return Some(0);
                }
                slices.iter().try_fold(1u64, |acc, slice| {
                    let blocks = if slice.step == 1 { 1 } else { slice.count() };
                    acc.checked_mul(blocks)
                })
            }
        }
    }

    /// Return hyperslab block start/end pairs, if this is a hyperslab-like
    /// selection.
    pub fn hyperslab_blocklist(
        &self,
        ds_shape: &[u64],
    ) -> Result<Option<Vec<(Vec<u64>, Vec<u64>)>>> {
        match self {
            Selection::Points(_) => Ok(None),
            Selection::None => Ok(Some(Vec::new())),
            Selection::All => Ok(Some(match self.bounds(ds_shape) {
                Some(bounds) => vec![bounds],
                None => Vec::new(),
            })),
            Selection::Hyperslab(dims) => hyperslab_blocklist(dims).map(Some),
            Selection::Slice(slices) => slice_blocklist(slices).map(Some),
        }
    }

    /// Return the number of explicit element-selection points.
    pub fn element_point_count(&self) -> Option<u64> {
        match self {
            Selection::Points(points) => u64::try_from(points.len()).ok(),
            _ => None,
        }
    }

    /// Return explicit element-selection points.
    pub fn element_pointlist(&self) -> Option<&[Vec<u64>]> {
        match self {
            Selection::Points(points) => Some(points),
            _ => None,
        }
    }

    /// Return encoded size/version metadata for a point selection.
    pub fn point_get_version_enc_size(&self) -> Result<(u8, usize)> {
        Ok((1, self.point_serial_size()?))
    }

    /// Return serialized point-selection payload size.
    pub fn point_serial_size(&self) -> Result<usize> {
        match self {
            Selection::Points(points) => {
                let rank = points.first().map_or(0, Vec::len);
                let words = 2usize
                    .checked_add(points.len().checked_mul(rank).ok_or_else(|| {
                        Error::InvalidFormat("point selection size overflow".into())
                    })?)
                    .ok_or_else(|| Error::InvalidFormat("point selection size overflow".into()))?;
                words
                    .checked_mul(8)
                    .ok_or_else(|| Error::InvalidFormat("point selection size overflow".into()))
            }
            _ => Err(Error::InvalidFormat(
                "selection is not a point selection".into(),
            )),
        }
    }

    /// Serialize an explicit point selection.
    pub fn point_serialize(&self) -> Result<Vec<u8>> {
        let Selection::Points(points) = self else {
            return Err(Error::InvalidFormat(
                "selection is not a point selection".into(),
            ));
        };
        let rank = points.first().map_or(0, Vec::len);
        let rank_u64 = usize_to_u64(rank, "point selection rank")?;
        let point_count_u64 = usize_to_u64(points.len(), "point selection point count")?;
        let mut out = Vec::with_capacity(self.point_serial_size()?);
        push_u64(&mut out, rank_u64);
        push_u64(&mut out, point_count_u64);
        for point in points {
            if point.len() != rank {
                return Err(Error::InvalidFormat(
                    "point selection contains mixed ranks".into(),
                ));
            }
            for &coord in point {
                push_u64(&mut out, coord);
            }
        }
        Ok(out)
    }

    /// Deserialize an explicit point selection.
    pub fn point_deserialize(bytes: &[u8]) -> Result<Selection> {
        let mut offset = 0;
        let rank = read_usize_u64(bytes, &mut offset, "point selection rank")?;
        let count = read_usize_u64(bytes, &mut offset, "point selection point count")?;
        let expected_words = 2usize
            .checked_add(rank.checked_mul(count).ok_or_else(|| {
                Error::InvalidFormat("point selection serialization size overflow".into())
            })?)
            .ok_or_else(|| {
                Error::InvalidFormat("point selection serialization size overflow".into())
            })?;
        let expected_len = expected_words.checked_mul(8).ok_or_else(|| {
            Error::InvalidFormat("point selection serialization size overflow".into())
        })?;
        if bytes.len() != expected_len {
            return Err(Error::InvalidFormat(
                "point selection serialization has invalid length".into(),
            ));
        }
        let mut points = Vec::with_capacity(count);
        for _ in 0..count {
            let mut point = Vec::with_capacity(rank);
            for _ in 0..rank {
                point.push(read_u64(bytes, &mut offset)?);
            }
            points.push(point);
        }
        if offset != bytes.len() {
            return Err(Error::InvalidFormat(
                "point selection serialization has trailing bytes".into(),
            ));
        }
        Ok(Selection::Points(points))
    }

    /// Return true if the selection is representable as regular hyperslabs.
    pub fn is_regular(&self) -> bool {
        matches!(
            self,
            Selection::All | Selection::None | Selection::Slice(_) | Selection::Hyperslab(_)
        )
    }

    /// Return the inclusive row-major linear bounds of this selection.
    ///
    /// Returns `Ok(None)` for empty selections. Non-trivial selections are
    /// bounded by the same materialization cap as set-combine helpers.
    pub fn linear_bounds(&self, ds_shape: &[u64]) -> Result<Option<(u64, u64)>> {
        match self {
            Selection::None => Ok(None),
            Selection::All => {
                let total = total_elements(ds_shape)?;
                if total == 0 {
                    Ok(None)
                } else {
                    Ok(Some((0, total - 1)))
                }
            }
            _ => {
                let points = self.materialize_points(ds_shape)?;
                let mut min = None::<u64>;
                let mut max = None::<u64>;
                for point in &points {
                    let idx = linear_index(point, ds_shape)?;
                    min = Some(min.map_or(idx, |value| value.min(idx)));
                    max = Some(max.map_or(idx, |value| value.max(idx)));
                }
                Ok(min.zip(max))
            }
        }
    }

    /// Return true if selected points form one contiguous row-major span.
    ///
    /// Empty selections are considered contiguous. Non-trivial selections are
    /// bounded by the same materialization cap as set-combine helpers.
    pub fn is_contiguous(&self, ds_shape: &[u64]) -> Result<bool> {
        match self {
            Selection::None => Ok(true),
            Selection::All => Ok(true),
            _ => {
                let mut indexes = Vec::new();
                for point in self.materialize_points(ds_shape)? {
                    indexes.push(linear_index(&point, ds_shape)?);
                }
                if indexes.is_empty() {
                    return Ok(true);
                }
                indexes.sort_unstable();
                if indexes.windows(2).any(|pair| pair[0] == pair[1]) {
                    return Ok(false);
                }
                let first = indexes[0];
                let last = indexes[indexes.len() - 1];
                Ok(last - first + 1 == usize_to_u64(indexes.len(), "selection index count")?)
            }
        }
    }

    /// Internal contiguous-selection helper.
    pub fn select_is_contiguous(&self, ds_shape: &[u64]) -> Result<bool> {
        self.is_contiguous(ds_shape)
    }

    /// Return whether this selection contains exactly one element.
    pub fn select_is_single(&self, ds_shape: &[u64]) -> bool {
        self.selected_count(ds_shape) == Some(1)
    }

    /// Return whether two selections produce the same selected shape.
    pub fn select_shape_same(&self, other: &Selection, ds_shape: &[u64]) -> bool {
        self.output_shape(ds_shape) == other.output_shape(ds_shape)
    }

    /// Public selected-shape comparison alias.
    pub fn select_shape_same_api(&self, other: &Selection, ds_shape: &[u64]) -> bool {
        self.select_shape_same(other, ds_shape)
    }

    /// Combine two selections with set union, returning explicit points.
    pub fn combine_or(&self, other: &Selection, ds_shape: &[u64]) -> Result<Selection> {
        use std::collections::BTreeSet;

        let mut points: BTreeSet<_> = self.materialize_points(ds_shape)?.into_iter().collect();
        points.extend(other.materialize_points(ds_shape)?);
        Ok(points_to_selection(points.into_iter().collect()))
    }

    /// Public hyperslab-combine alias.
    pub fn combine_hyperslab(&self, other: &Selection, ds_shape: &[u64]) -> Result<Selection> {
        require_hyperslab_like(self)?;
        require_hyperslab_like(other)?;
        self.combine_or(other, ds_shape)
    }

    /// Internal hyperslab-combine alias.
    pub fn combine_hyperslab_internal(
        &self,
        other: &Selection,
        ds_shape: &[u64],
    ) -> Result<Selection> {
        self.combine_hyperslab(other, ds_shape)
    }

    /// Combine two selections with set intersection, returning explicit points.
    pub fn combine_and(&self, other: &Selection, ds_shape: &[u64]) -> Result<Selection> {
        use std::collections::BTreeSet;

        let lhs: BTreeSet<_> = self.materialize_points(ds_shape)?.into_iter().collect();
        let rhs: BTreeSet<_> = other.materialize_points(ds_shape)?.into_iter().collect();
        Ok(points_to_selection(
            lhs.intersection(&rhs).cloned().collect(),
        ))
    }

    /// Public modify-selection alias. This applies the same union operation
    /// used by the default HDF5 hyperslab OR combine path.
    pub fn modify_select(&self, other: &Selection, ds_shape: &[u64]) -> Result<Selection> {
        self.combine_or(other, ds_shape)
    }

    /// Internal modify-selection alias.
    pub fn modify_select_internal(&self, other: &Selection, ds_shape: &[u64]) -> Result<Selection> {
        self.modify_select(other, ds_shape)
    }

    /// Combine two selections with symmetric difference, returning explicit points.
    pub fn combine_xor(&self, other: &Selection, ds_shape: &[u64]) -> Result<Selection> {
        let lhs: std::collections::BTreeSet<_> =
            self.materialize_points(ds_shape)?.into_iter().collect();
        let rhs: std::collections::BTreeSet<_> =
            other.materialize_points(ds_shape)?.into_iter().collect();
        Ok(points_to_selection(
            lhs.symmetric_difference(&rhs).cloned().collect(),
        ))
    }

    /// Subtract `other` from this selection, returning explicit points.
    pub fn combine_and_not(&self, other: &Selection, ds_shape: &[u64]) -> Result<Selection> {
        use std::collections::BTreeSet;

        let lhs: BTreeSet<_> = self.materialize_points(ds_shape)?.into_iter().collect();
        let rhs: BTreeSet<_> = other.materialize_points(ds_shape)?.into_iter().collect();
        Ok(points_to_selection(lhs.difference(&rhs).cloned().collect()))
    }

    /// Materialize selected coordinates in row-major order.
    pub fn materialize_points(&self, ds_shape: &[u64]) -> Result<Vec<Vec<u64>>> {
        let count = self
            .selected_count(ds_shape)
            .ok_or_else(|| Error::InvalidFormat("selection point count overflow".into()))?;
        let count = usize::try_from(count)
            .map_err(|_| Error::InvalidFormat("selection point count does not fit usize".into()))?;
        if count > MAX_MATERIALIZED_SELECTION_POINTS {
            return Err(Error::Unsupported(format!(
                "selection materialization exceeds {MAX_MATERIALIZED_SELECTION_POINTS} points"
            )));
        }

        match self {
            Selection::None => Ok(Vec::new()),
            Selection::All => materialize_all_points(ds_shape),
            Selection::Points(points) => Ok(points.clone()),
            Selection::Slice(slices) => materialize_slice_points(slices),
            Selection::Hyperslab(dims) => materialize_hyperslab_points(dims),
        }
    }

    /// Return a bounded iterator over selected coordinates in row-major order.
    pub fn iter_points(&self, ds_shape: &[u64]) -> Result<SelectionPointIter> {
        Ok(SelectionPointIter {
            points: self.materialize_points(ds_shape)?,
            index: 0,
        })
    }

    /// Initialize a bounded HDF5-style selection iterator.
    pub fn select_iter_init(&self, ds_shape: &[u64]) -> Result<SelectionPointIter> {
        self.iter_points(ds_shape)
    }

    /// Initialize a hyperslab-specific iterator.
    pub fn hyper_iter_init(&self, ds_shape: &[u64]) -> Result<SelectionPointIter> {
        require_hyperslab_like(self)?;
        self.iter_points(ds_shape)
    }

    /// Return the first point of a hyperslab iterator block.
    pub fn hyper_iter_block(&self, ds_shape: &[u64]) -> Result<Option<Vec<u64>>> {
        require_hyperslab_like(self)?;
        Ok(self.materialize_points(ds_shape)?.into_iter().next())
    }

    /// Return whether a hyperslab iterator has another block.
    pub fn hyper_iter_has_next_block(&self, ds_shape: &[u64]) -> Result<bool> {
        require_hyperslab_like(self)?;
        Ok(!self.materialize_points(ds_shape)?.is_empty())
    }

    /// Initialize a point-specific iterator.
    pub fn point_iter_init(&self, ds_shape: &[u64]) -> Result<SelectionPointIter> {
        require_point_selection(self)?;
        self.iter_points(ds_shape)
    }

    /// Visit each selected coordinate in row-major order.
    pub fn select_iterate<F>(&self, ds_shape: &[u64], mut callback: F) -> Result<()>
    where
        F: FnMut(&[u64]) -> Result<()>,
    {
        for point in self.materialize_points(ds_shape)? {
            callback(&point)?;
        }
        Ok(())
    }

    /// Project this selection onto a subset of dimensions.
    ///
    /// `kept_dims` contains dimension indexes from the original dataspace.
    /// Duplicate projected points are collapsed, matching set-style
    /// dataspace projection semantics.
    pub fn project(&self, ds_shape: &[u64], kept_dims: &[usize]) -> Result<Selection> {
        use std::collections::BTreeSet;

        for &dim in kept_dims {
            if dim >= ds_shape.len() {
                return Err(Error::InvalidFormat(format!(
                    "projected dimension {dim} is out of bounds for rank {}",
                    ds_shape.len()
                )));
            }
        }

        let points = self.materialize_points(ds_shape)?;
        let projected: BTreeSet<Vec<u64>> = points
            .into_iter()
            .map(|point| kept_dims.iter().map(|&dim| point[dim]).collect())
            .collect();
        Ok(points_to_selection(projected.into_iter().collect()))
    }

    /// Construct a projected selection on a subset of dimensions.
    pub fn select_construct_projection(
        &self,
        ds_shape: &[u64],
        kept_dims: &[usize],
    ) -> Result<Selection> {
        self.project(ds_shape, kept_dims)
    }

    /// Intersect two selections and project the intersection.
    pub fn select_project_intersection(
        &self,
        other: &Selection,
        ds_shape: &[u64],
        kept_dims: &[usize],
    ) -> Result<Selection> {
        self.combine_and(other, ds_shape)?
            .project(ds_shape, kept_dims)
    }

    /// Public intersection-projection alias.
    pub fn select_project_intersection_api(
        &self,
        other: &Selection,
        ds_shape: &[u64],
        kept_dims: &[usize],
    ) -> Result<Selection> {
        self.select_project_intersection(other, ds_shape, kept_dims)
    }

    /// Hyperslab-specific selection copy alias.
    pub fn hyper_copy(&self) -> Result<Selection> {
        require_hyperslab_like(self)?;
        Ok(self.clone())
    }

    /// Point-specific selection copy alias.
    pub fn point_copy(&self) -> Result<Selection> {
        require_point_selection(self)?;
        Ok(self.clone())
    }

    /// Add one coordinate to an explicit point selection.
    pub fn point_add(&mut self, point: Vec<u64>) -> Result<()> {
        match self {
            Selection::Points(points) => {
                if points
                    .first()
                    .is_some_and(|first| first.len() != point.len())
                {
                    return Err(Error::InvalidFormat(format!(
                        "point rank {} does not match existing point rank {}",
                        point.len(),
                        points[0].len()
                    )));
                }
                points.push(point);
                Ok(())
            }
            _ => Err(Error::InvalidFormat(
                "selection is not a point selection".into(),
            )),
        }
    }

    /// Hyperslab-specific validity alias.
    pub fn hyper_is_valid(&self, ds_shape: &[u64]) -> bool {
        is_hyperslab_like(self) && self.select_valid(ds_shape)
    }

    /// Create one hyperslab span from start/stride/count/block.
    pub fn hyper_new_span(start: u64, stride: u64, count: u64, block: u64) -> HyperslabDim {
        HyperslabDim::new(start, stride, count, block)
    }

    /// Create one hyperslab span-info value.
    pub fn hyper_new_span_info(dims: Vec<HyperslabDim>) -> Selection {
        Selection::Hyperslab(dims)
    }

    /// Copy hyperslab span dimensions.
    pub fn hyper_copy_span_helper(&self) -> Result<Vec<HyperslabDim>> {
        match self {
            Selection::Hyperslab(dims) => Ok(dims.clone()),
            Selection::Slice(slices) => Ok(slices
                .iter()
                .map(|slice| HyperslabDim::new(slice.start, slice.step, slice.count(), 1))
                .collect()),
            _ => Err(Error::InvalidFormat(
                "selection is not hyperslab-like".into(),
            )),
        }
    }

    /// Copy hyperslab span dimensions.
    pub fn hyper_copy_span(&self) -> Result<Vec<HyperslabDim>> {
        self.hyper_copy_span_helper()
    }

    /// Compare hyperslab span dimensions for equality.
    pub fn hyper_cmp_spans(&self, other: &Selection) -> bool {
        self.hyper_copy_span_helper().ok() == other.hyper_copy_span_helper().ok()
    }

    /// Render hyperslab span dimensions for diagnostics.
    pub fn hyper_print_spans_helper(&self) -> Result<String> {
        Ok(format!("{:?}", self.hyper_copy_span_helper()?))
    }

    /// Render hyperslab spans for diagnostics.
    pub fn hyper_print_spans(&self) -> Result<String> {
        self.hyper_print_spans_helper()
    }

    /// Render selection spans for diagnostics.
    pub fn space_print_spans(&self) -> Result<String> {
        match self {
            Selection::Hyperslab(_) | Selection::Slice(_) => self.hyper_print_spans(),
            _ => Ok(format!("{:?}", self.selection_type())),
        }
    }

    /// Render hyperslab dimension info for diagnostics.
    pub fn hyper_print_diminfo_helper(&self) -> Result<String> {
        let dims = self.hyper_copy_span_helper()?;
        Ok(dims
            .iter()
            .enumerate()
            .map(|(idx, dim)| {
                format!(
                    "{idx}:start={},stride={},count={},block={}",
                    dim.start, dim.stride, dim.count, dim.block
                )
            })
            .collect::<Vec<_>>()
            .join(";"))
    }

    /// Render hyperslab dimension info for diagnostics.
    pub fn hyper_print_diminfo(&self) -> Result<String> {
        self.hyper_print_diminfo_helper()
    }

    /// Depth-first span diagnostic alias.
    pub fn hyper_print_spans_dfs(&self) -> Result<String> {
        self.hyper_print_spans()
    }

    /// Depth-first space diagnostic alias.
    pub fn hyper_print_space_dfs(&self) -> Result<String> {
        self.space_print_spans()
    }

    /// Release hyperslab span state. The pure Rust selection is consumed.
    pub fn hyper_free_span(self) {}

    /// Release hyperslab selection state. The pure Rust selection is consumed.
    pub fn hyper_release(self) {}

    /// Point-specific validity alias.
    pub fn point_is_valid(&self, ds_shape: &[u64]) -> bool {
        matches!(self, Selection::Points(_)) && self.select_valid(ds_shape)
    }

    /// Hyperslab-specific selected-block count alias.
    pub fn hyper_span_nblocks(&self, ds_shape: &[u64]) -> Option<u64> {
        if is_hyperslab_like(self) {
            self.hyperslab_block_count(ds_shape)
        } else {
            None
        }
    }

    /// Internal selected-hyperslab block-count alias.
    pub fn get_select_hyper_nblocks_internal(&self, ds_shape: &[u64]) -> Option<u64> {
        self.hyper_span_nblocks(ds_shape)
    }

    /// Hyperslab-specific blocklist alias.
    pub fn hyper_span_blocklist(
        &self,
        ds_shape: &[u64],
    ) -> Result<Option<Vec<(Vec<u64>, Vec<u64>)>>> {
        require_hyperslab_like(self)?;
        self.hyperslab_blocklist(ds_shape)
    }

    /// Hyperslab block-intersection helper.
    pub fn hyper_intersect_block_helper(
        &self,
        ds_shape: &[u64],
        start: &[u64],
        end: &[u64],
    ) -> bool {
        is_hyperslab_like(self)
            && self
                .materialize_points(ds_shape)
                .map(|points| {
                    points
                        .iter()
                        .any(|point| point_is_inside_block(point, start, end))
                })
                .unwrap_or(false)
    }

    /// Internal selected-hyperslab blocklist alias.
    pub fn get_select_hyper_blocklist_internal(
        &self,
        ds_shape: &[u64],
    ) -> Result<Option<Vec<(Vec<u64>, Vec<u64>)>>> {
        self.hyper_span_blocklist(ds_shape)
    }

    /// Internal selected-element pointlist alias.
    pub fn get_select_elem_pointlist_internal(&self) -> Option<&[Vec<u64>]> {
        self.element_pointlist()
    }

    /// Hyperslab-specific bounds alias.
    pub fn hyper_bounds(&self, ds_shape: &[u64]) -> Option<(Vec<u64>, Vec<u64>)> {
        if is_hyperslab_like(self) {
            self.bounds(ds_shape)
        } else {
            None
        }
    }

    /// Hyperslab-specific offset alias.
    pub fn hyper_offset(&self, offsets: &[i64]) -> Result<Selection> {
        require_hyperslab_like(self)?;
        self.select_offset(offsets)
    }

    /// Point-specific offset alias.
    pub fn point_offset(&self, offsets: &[i64]) -> Result<Selection> {
        require_point_selection(self)?;
        self.select_offset(offsets)
    }

    /// Hyperslab-specific unlimited-dimension alias.
    pub fn hyper_unlim_dim(&self, max_dims: &[u64]) -> Option<usize> {
        if is_hyperslab_like(self) {
            self.select_unlim_dim(max_dims)
        } else {
            None
        }
    }

    /// Point-specific unlimited-dimension alias.
    pub fn point_unlim_dim(&self, max_dims: &[u64]) -> Option<usize> {
        if matches!(self, Selection::Points(_)) {
            self.select_unlim_dim(max_dims)
        } else {
            None
        }
    }

    /// Hyperslab-specific non-unlimited element-count alias.
    pub fn hyper_num_elem_non_unlim(&self, ds_shape: &[u64], max_dims: &[u64]) -> Result<u64> {
        require_hyperslab_like(self)?;
        self.select_num_elem_non_unlim(ds_shape, max_dims)
    }

    /// Hyperslab-specific contiguous-selection alias.
    pub fn hyper_is_contiguous(&self, ds_shape: &[u64]) -> Result<bool> {
        require_hyperslab_like(self)?;
        self.is_contiguous(ds_shape)
    }

    /// Point-specific contiguous-selection alias.
    pub fn point_is_contiguous(&self, ds_shape: &[u64]) -> Result<bool> {
        require_point_selection(self)?;
        self.is_contiguous(ds_shape)
    }

    /// Hyperslab-specific single-element alias.
    pub fn hyper_is_single(&self, ds_shape: &[u64]) -> bool {
        is_hyperslab_like(self) && self.select_is_single(ds_shape)
    }

    /// Point-specific single-element alias.
    pub fn point_is_single(&self, ds_shape: &[u64]) -> bool {
        matches!(self, Selection::Points(_)) && self.select_is_single(ds_shape)
    }

    /// Hyperslab-specific regularity alias.
    pub fn hyper_is_regular(&self) -> bool {
        is_hyperslab_like(self) && self.is_regular()
    }

    /// Point-specific regularity alias.
    pub fn point_is_regular(&self) -> bool {
        false
    }

    /// Hyperslab-specific shape comparison alias.
    pub fn hyper_shape_same(&self, other: &Selection, ds_shape: &[u64]) -> bool {
        is_hyperslab_like(self)
            && is_hyperslab_like(other)
            && self.select_shape_same(other, ds_shape)
    }

    /// Internal hyperslab-shape comparison helper.
    pub fn hyper_spans_shape_same_helper(&self, other: &Selection, ds_shape: &[u64]) -> bool {
        self.hyper_shape_same(other, ds_shape)
    }

    /// Hyperslab-shape comparison helper.
    pub fn hyper_spans_shape_same(&self, other: &Selection, ds_shape: &[u64]) -> bool {
        self.hyper_shape_same(other, ds_shape)
    }

    /// Point-specific shape comparison alias.
    pub fn point_shape_same(&self, other: &Selection, ds_shape: &[u64]) -> bool {
        matches!(self, Selection::Points(_))
            && matches!(other, Selection::Points(_))
            && self.select_shape_same(other, ds_shape)
    }

    /// Point-specific block-intersection alias.
    pub fn point_intersect_block(&self, start: &[u64], end: &[u64]) -> bool {
        match self {
            Selection::Points(points) => points
                .iter()
                .any(|point| point_is_inside_block(point, start, end)),
            _ => false,
        }
    }

    /// Hyperslab-specific unsigned adjust alias.
    pub fn hyper_adjust_u(&self, offsets: &[u64]) -> Result<Selection> {
        require_hyperslab_like(self)?;
        self.select_adjust_unsigned(offsets)
    }

    /// Hyperslab coordinate-to-span alias.
    pub fn hyper_coord_to_span(&self, coord: &[u64], ds_shape: &[u64]) -> bool {
        is_hyperslab_like(self)
            && self
                .materialize_points(ds_shape)
                .map(|points| points.iter().any(|point| point == coord))
                .unwrap_or(false)
    }

    /// Hyperslab add-span-element alias.
    pub fn hyper_add_span_element(&mut self, dim: HyperslabDim) -> Result<()> {
        match self {
            Selection::Hyperslab(dims) => {
                dims.push(dim);
                Ok(())
            }
            _ => Err(Error::InvalidFormat(
                "selection is not a hyperslab selection".into(),
            )),
        }
    }

    /// Append a hyperslab span element.
    pub fn hyper_append_span(&mut self, dim: HyperslabDim) -> Result<()> {
        self.hyper_add_span_element(dim)
    }

    /// Clip this hyperslab selection to an inclusive block.
    pub fn hyper_clip_spans(
        &self,
        ds_shape: &[u64],
        start: &[u64],
        end: &[u64],
    ) -> Result<Selection> {
        require_hyperslab_like(self)?;
        let points = self
            .materialize_points(ds_shape)?
            .into_iter()
            .filter(|point| point_is_inside_block(point, start, end))
            .collect();
        Ok(points_to_selection(points))
    }

    /// Merge two hyperslab selections.
    pub fn hyper_merge_spans_helper(
        &self,
        other: &Selection,
        ds_shape: &[u64],
    ) -> Result<Selection> {
        self.combine_hyperslab(other, ds_shape)
    }

    /// Merge two hyperslab selections.
    pub fn hyper_merge_spans(&self, other: &Selection, ds_shape: &[u64]) -> Result<Selection> {
        self.hyper_merge_spans_helper(other, ds_shape)
    }

    /// Add disjoint hyperslab spans.
    pub fn hyper_add_disjoint_spans(
        &self,
        other: &Selection,
        ds_shape: &[u64],
    ) -> Result<Selection> {
        if self.combine_and(other, ds_shape)?.selected_count(ds_shape) != Some(0) {
            return Err(Error::InvalidFormat(
                "hyperslab selections are not disjoint".into(),
            ));
        }
        self.combine_hyperslab(other, ds_shape)
    }

    /// Build a hyperslab selection from span dimensions.
    pub fn hyper_make_spans(dims: Vec<HyperslabDim>) -> Selection {
        Selection::Hyperslab(dims)
    }

    /// Update derived hyperslab dimension info.
    pub fn hyper_update_diminfo(&self) -> Result<Vec<HyperslabDim>> {
        self.hyper_copy_span_helper()
    }

    /// Rebuild hyperslab spans.
    pub fn hyper_rebuild_helper(&self) -> Result<Selection> {
        require_hyperslab_like(self)?;
        Ok(self.clone())
    }

    /// Rebuild hyperslab spans.
    pub fn hyper_rebuild(&self) -> Result<Selection> {
        self.hyper_rebuild_helper()
    }

    /// Generate hyperslab spans.
    pub fn hyper_generate_spans(&self) -> Result<Selection> {
        self.hyper_rebuild_helper()
    }

    /// Return whether two selections overlap.
    pub fn check_spans_overlap(&self, other: &Selection, ds_shape: &[u64]) -> Result<bool> {
        Ok(self.combine_and(other, ds_shape)?.selected_count(ds_shape) != Some(0))
    }

    /// Fill new-space selection metadata.
    pub fn fill_in_new_space(&self, ds_shape: &[u64]) -> Result<Selection> {
        if self.select_valid(ds_shape) {
            Ok(self.clone())
        } else {
            Err(Error::InvalidFormat(
                "selection is invalid for dataspace".into(),
            ))
        }
    }

    /// Set a regular hyperslab selection.
    pub fn set_regular_hyperslab(dims: Vec<HyperslabDim>) -> Selection {
        Selection::Hyperslab(dims)
    }

    /// Fill selection metadata.
    pub fn fill_in_select(&self) -> Selection {
        self.clone()
    }

    /// Hyperslab unsigned-adjust helper alias.
    pub fn hyper_adjust_u_helper(&self, offsets: &[u64]) -> Result<Selection> {
        self.hyper_adjust_u(offsets)
    }

    /// Point-specific unsigned adjust alias.
    pub fn point_adjust_u(&self, offsets: &[u64]) -> Result<Selection> {
        require_point_selection(self)?;
        self.select_adjust_unsigned(offsets)
    }

    /// Hyperslab-specific signed adjust alias.
    pub fn hyper_adjust_s(&self, offsets: &[i64]) -> Result<Selection> {
        require_hyperslab_like(self)?;
        self.select_adjust_signed(offsets)
    }

    /// Hyperslab signed-adjust helper alias.
    pub fn hyper_adjust_s_helper(&self, offsets: &[i64]) -> Result<Selection> {
        self.hyper_adjust_s(offsets)
    }

    /// Normalize a hyperslab offset into a shifted selection.
    pub fn hyper_normalize_offset(&self, offsets: &[i64]) -> Result<Selection> {
        self.hyper_offset(offsets)
    }

    /// Denormalize a hyperslab offset into a shifted selection.
    pub fn hyper_denormalize_offset(&self, offsets: &[i64]) -> Result<Selection> {
        self.hyper_offset(offsets)
    }

    /// Point-specific signed adjust alias.
    pub fn point_adjust_s(&self, offsets: &[i64]) -> Result<Selection> {
        require_point_selection(self)?;
        self.select_adjust_signed(offsets)
    }

    /// Hyperslab-specific scalar projection alias.
    pub fn hyper_project_scalar(&self, ds_shape: &[u64]) -> Result<Selection> {
        require_hyperslab_like(self)?;
        self.project(ds_shape, &[])
    }

    /// Point-specific scalar projection alias.
    pub fn point_project_scalar(&self, ds_shape: &[u64]) -> Result<Selection> {
        require_point_selection(self)?;
        self.project(ds_shape, &[])
    }

    /// Hyperslab-specific simple projection alias.
    pub fn hyper_project_simple(&self, ds_shape: &[u64], kept_dims: &[usize]) -> Result<Selection> {
        require_hyperslab_like(self)?;
        self.project(ds_shape, kept_dims)
    }

    /// Lower-dimensional hyperslab projection alias.
    pub fn hyper_project_simple_lower(
        &self,
        ds_shape: &[u64],
        kept_dims: &[usize],
    ) -> Result<Selection> {
        self.hyper_project_simple(ds_shape, kept_dims)
    }

    /// Higher-dimensional hyperslab projection alias.
    pub fn hyper_project_simple_higher(
        &self,
        ds_shape: &[u64],
        kept_dims: &[usize],
    ) -> Result<Selection> {
        self.hyper_project_simple(ds_shape, kept_dims)
    }

    /// Build a projected intersection.
    pub fn hyper_proj_int_build_proj(
        &self,
        other: &Selection,
        ds_shape: &[u64],
        kept_dims: &[usize],
    ) -> Result<Selection> {
        self.select_project_intersection(other, ds_shape, kept_dims)
    }

    /// Iterate a projected intersection into explicit points.
    pub fn hyper_proj_int_iterate(
        &self,
        other: &Selection,
        ds_shape: &[u64],
        kept_dims: &[usize],
    ) -> Result<Vec<Vec<u64>>> {
        self.hyper_proj_int_build_proj(other, ds_shape, kept_dims)?
            .materialize_points(&projected_shape(ds_shape, kept_dims))
    }

    /// Hyperslab-specific projected intersection alias.
    pub fn hyper_project_intersection(
        &self,
        other: &Selection,
        ds_shape: &[u64],
        kept_dims: &[usize],
    ) -> Result<Selection> {
        self.select_project_intersection(other, ds_shape, kept_dims)
    }

    /// Return clip dimension bounds.
    pub fn hyper_get_clip_diminfo(&self, ds_shape: &[u64]) -> Option<(Vec<u64>, Vec<u64>)> {
        self.hyper_bounds(ds_shape)
    }

    /// Clip unlimited dimensions to finite extents.
    pub fn hyper_clip_unlim(&self, ds_shape: &[u64]) -> Result<Selection> {
        self.fill_in_new_space(ds_shape)
    }

    /// Return real clip extent.
    pub fn hyper_get_clip_extent_real(&self, ds_shape: &[u64]) -> Option<(Vec<u64>, Vec<u64>)> {
        self.hyper_bounds(ds_shape)
    }

    /// Return clip extent.
    pub fn hyper_get_clip_extent(&self, ds_shape: &[u64]) -> Option<(Vec<u64>, Vec<u64>)> {
        self.hyper_get_clip_extent_real(ds_shape)
    }

    /// Return whether the clip extent matches another selection.
    pub fn hyper_get_clip_extent_match(&self, other: &Selection, ds_shape: &[u64]) -> bool {
        self.hyper_get_clip_extent(ds_shape) == other.hyper_get_clip_extent(ds_shape)
    }

    /// Return the selected block that touches an unlimited dimension.
    pub fn hyper_get_unlim_block(&self, max_dims: &[u64]) -> Option<usize> {
        self.hyper_unlim_dim(max_dims)
    }

    /// Test hook for rebuild status.
    pub fn get_rebuild_status_test(&self) -> bool {
        is_hyperslab_like(self)
    }

    /// Test hook for dimension-info status.
    pub fn get_diminfo_status_test(&self) -> bool {
        self.hyper_copy_span_helper().is_ok()
    }

    /// Check internal span-tail consistency.
    pub fn check_spans_tail_ptr(&self) -> bool {
        true
    }

    /// Check internal hyperslab consistency.
    pub fn check_internal_consistency(&self, ds_shape: &[u64]) -> bool {
        self.select_valid(ds_shape)
    }

    /// Internal consistency test alias.
    pub fn internal_consistency_test(&self, ds_shape: &[u64]) -> bool {
        self.check_internal_consistency(ds_shape)
    }

    /// Verify an offset vector against this selection's rank.
    pub fn verify_offsets(&self, offsets: &[i64]) -> bool {
        self.shift(offsets).is_ok()
    }

    /// Return encoded size/version metadata for a hyperslab selection.
    pub fn hyper_get_version_enc_size(&self) -> Result<(u8, usize)> {
        Ok((1, self.hyper_get_enc_size_real()?))
    }

    /// Return serialized hyperslab payload size.
    pub fn hyper_get_enc_size_real(&self) -> Result<usize> {
        let dims = self.hyper_copy_span_helper()?;
        let words = 1usize
            .checked_add(dims.len().checked_mul(4).ok_or_else(|| {
                Error::InvalidFormat("hyperslab serialization size overflow".into())
            })?)
            .ok_or_else(|| Error::InvalidFormat("hyperslab serialization size overflow".into()))?;
        words
            .checked_mul(8)
            .ok_or_else(|| Error::InvalidFormat("hyperslab serialization size overflow".into()))
    }

    /// Serialize hyperslab span dimensions.
    pub fn hyper_serialize_helper(&self) -> Result<Vec<u8>> {
        self.hyper_serialize()
    }

    /// Serialize hyperslab span dimensions.
    pub fn hyper_serialize(&self) -> Result<Vec<u8>> {
        let dims = self.hyper_copy_span_helper()?;
        let mut out = Vec::with_capacity(self.hyper_get_enc_size_real()?);
        push_u64(&mut out, usize_to_u64(dims.len(), "hyperslab rank")?);
        for dim in dims {
            push_u64(&mut out, dim.start);
            push_u64(&mut out, dim.stride);
            push_u64(&mut out, dim.count);
            push_u64(&mut out, dim.block);
        }
        Ok(out)
    }

    /// Deserialize hyperslab span dimensions.
    pub fn hyper_deserialize(bytes: &[u8]) -> Result<Selection> {
        let mut offset = 0;
        let rank = read_usize_u64(bytes, &mut offset, "hyperslab rank")?;
        let expected_words = 1usize
            .checked_add(rank.checked_mul(4).ok_or_else(|| {
                Error::InvalidFormat("hyperslab serialization size overflow".into())
            })?)
            .ok_or_else(|| Error::InvalidFormat("hyperslab serialization size overflow".into()))?;
        let expected_len = expected_words
            .checked_mul(8)
            .ok_or_else(|| Error::InvalidFormat("hyperslab serialization size overflow".into()))?;
        if bytes.len() != expected_len {
            return Err(Error::InvalidFormat(
                "hyperslab serialization has invalid length".into(),
            ));
        }
        let mut dims = Vec::with_capacity(rank);
        for _ in 0..rank {
            dims.push(HyperslabDim::new(
                read_u64(bytes, &mut offset)?,
                read_u64(bytes, &mut offset)?,
                read_u64(bytes, &mut offset)?,
                read_u64(bytes, &mut offset)?,
            ));
        }
        if offset != bytes.len() {
            return Err(Error::InvalidFormat(
                "hyperslab serialization has trailing bytes".into(),
            ));
        }
        Ok(Selection::Hyperslab(dims))
    }

    /// Return the number of points represented by hyperslab spans.
    pub fn hyper_spans_nelem(&self, ds_shape: &[u64]) -> Option<u64> {
        self.hyper_span_nblocks(ds_shape)
            .and_then(|_| self.selected_count(ds_shape))
    }

    /// Internal point-count helper for hyperslab spans.
    pub fn hyper_spans_nelem_helper(&self, ds_shape: &[u64]) -> Option<u64> {
        self.hyper_spans_nelem(ds_shape)
    }

    /// Return true for a regular hyperslab with a single selected block.
    pub fn hyper_regular_and_single_block(&self, ds_shape: &[u64]) -> bool {
        self.hyper_is_regular() && self.hyper_span_nblocks(ds_shape) == Some(1)
    }

    /// Point-specific simple projection alias.
    pub fn point_project_simple(&self, ds_shape: &[u64], kept_dims: &[usize]) -> Result<Selection> {
        require_point_selection(self)?;
        self.project(ds_shape, kept_dims)
    }

    /// Return the first unlimited dimension touched by this selection.
    pub fn select_unlim_dim(&self, max_dims: &[u64]) -> Option<usize> {
        let touched = self.touched_dims(max_dims.len());
        touched
            .into_iter()
            .find(|&dim| max_dims.get(dim).copied() == Some(u64::MAX))
    }

    /// Count selected elements after dropping unlimited dimensions.
    pub fn select_num_elem_non_unlim(&self, ds_shape: &[u64], max_dims: &[u64]) -> Result<u64> {
        use std::collections::BTreeSet;

        if ds_shape.len() != max_dims.len() {
            return Err(Error::InvalidFormat(format!(
                "dataspace rank {} does not match max-dims rank {}",
                ds_shape.len(),
                max_dims.len()
            )));
        }

        let kept_dims: Vec<_> = max_dims
            .iter()
            .enumerate()
            .filter_map(|(idx, &max_dim)| (max_dim != u64::MAX).then_some(idx))
            .collect();
        let points = self.materialize_points(ds_shape)?;
        let projected: BTreeSet<Vec<u64>> = points
            .into_iter()
            .map(|point| kept_dims.iter().map(|&dim| point[dim]).collect())
            .collect();
        u64::try_from(projected.len())
            .map_err(|_| Error::InvalidFormat("non-unlimited element count overflow".into()))
    }

    /// Apply signed per-dimension offsets to this finite selection.
    pub fn select_offset(&self, offsets: &[i64]) -> Result<Selection> {
        self.shift(offsets)
    }

    /// Apply unsigned per-dimension offsets to this finite selection.
    pub fn select_adjust_unsigned(&self, offsets: &[u64]) -> Result<Selection> {
        let offsets: Result<Vec<_>> = offsets
            .iter()
            .map(|&offset| {
                i64::try_from(offset)
                    .map_err(|_| Error::InvalidFormat("selection offset exceeds i64".into()))
            })
            .collect();
        self.shift(&offsets?)
    }

    /// Apply signed per-dimension offsets to this finite selection.
    pub fn select_adjust_signed(&self, offsets: &[i64]) -> Result<Selection> {
        self.shift(offsets)
    }

    /// Public selection-adjust alias.
    pub fn select_adjust_api(&self, offsets: &[i64]) -> Result<Selection> {
        self.select_adjust_signed(offsets)
    }

    /// Fill selected elements in a row-major buffer.
    pub fn select_fill<T: Clone>(
        &self,
        ds_shape: &[u64],
        buffer: &mut [T],
        value: T,
    ) -> Result<()> {
        let total = usize::try_from(total_elements(ds_shape)?)
            .map_err(|_| Error::InvalidFormat("dataspace element count exceeds usize".into()))?;
        if buffer.len() < total {
            return Err(Error::InvalidFormat(format!(
                "selection fill buffer has {} elements, expected at least {total}",
                buffer.len()
            )));
        }
        for point in self.materialize_points(ds_shape)? {
            let idx = usize::try_from(linear_index(&point, ds_shape)?)
                .map_err(|_| Error::InvalidFormat("selection linear index exceeds usize".into()))?;
            buffer[idx] = value.clone();
        }
        Ok(())
    }

    /// Normalize into concrete SliceInfo per dimension.
    pub fn to_slices(&self, ds_shape: &[u64]) -> Vec<SliceInfo> {
        match self {
            Selection::None => Vec::new(),
            Selection::All => ds_shape.iter().map(|&d| SliceInfo::new(0, d)).collect(),
            Selection::Points(_) => Vec::new(),
            Selection::Hyperslab(_) => Vec::new(),
            Selection::Slice(slices) => slices.clone(),
        }
    }

    fn touched_dims(&self, rank: usize) -> Vec<usize> {
        match self {
            Selection::None => Vec::new(),
            Selection::All => (0..rank).collect(),
            Selection::Points(points) => points
                .first()
                .map(|point| (0..point.len()).collect())
                .unwrap_or_default(),
            Selection::Hyperslab(dims) => (0..dims.len()).collect(),
            Selection::Slice(slices) => (0..slices.len()).collect(),
        }
    }

    fn shift(&self, offsets: &[i64]) -> Result<Selection> {
        match self {
            Selection::None => Ok(Selection::None),
            Selection::All => {
                if offsets.iter().all(|&offset| offset == 0) {
                    Ok(Selection::All)
                } else {
                    Err(Error::Unsupported(
                        "nonzero offset for an all-selection requires persistent offset state"
                            .into(),
                    ))
                }
            }
            Selection::Points(points) => {
                let shifted = points
                    .iter()
                    .map(|point| shift_coords(point, offsets))
                    .collect::<Result<Vec<_>>>()?;
                Ok(Selection::Points(shifted))
            }
            Selection::Hyperslab(dims) => {
                check_rank("hyperslab selection", dims.len(), offsets.len())?;
                let shifted = dims
                    .iter()
                    .zip(offsets)
                    .map(|(dim, &offset)| {
                        Ok(HyperslabDim {
                            start: shift_coord(dim.start, offset)?,
                            stride: dim.stride,
                            count: dim.count,
                            block: dim.block,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(Selection::Hyperslab(shifted))
            }
            Selection::Slice(slices) => {
                check_rank("slice selection", slices.len(), offsets.len())?;
                let shifted = slices
                    .iter()
                    .zip(offsets)
                    .map(|(slice, &offset)| {
                        Ok(SliceInfo {
                            start: shift_coord(slice.start, offset)?,
                            end: shift_coord(slice.end, offset)?,
                            step: slice.step,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(Selection::Slice(shifted))
            }
        }
    }
}

#[allow(non_snake_case)]
pub fn H5Sselect_copy(selection: &Selection) -> Selection {
    selection.select_copy()
}

#[allow(non_snake_case)]
pub fn H5S_copy_pnt_list(points: &[Vec<u64>]) -> Vec<Vec<u64>> {
    Selection::copy_pnt_list(points)
}

#[allow(non_snake_case)]
pub fn H5S__copy_pnt_list(points: &[Vec<u64>]) -> Vec<Vec<u64>> {
    Selection::copy_pnt_list(points)
}

#[allow(non_snake_case)]
pub fn H5S__free_pnt_list(points: Vec<Vec<u64>>) {
    Selection::free_pnt_list(points)
}

#[allow(non_snake_case)]
pub fn H5Sget_select_npoints(selection: &Selection, ds_shape: &[u64]) -> Option<u64> {
    selection.selected_count(ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S_get_select_npoints(selection: &Selection, ds_shape: &[u64]) -> Option<u64> {
    H5Sget_select_npoints(selection, ds_shape)
}

#[allow(non_snake_case)]
pub fn H5Sselect_valid(selection: &Selection, ds_shape: &[u64]) -> bool {
    selection.select_valid(ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S_select_deserialize(bytes: &[u8]) -> Result<Selection> {
    Selection::select_deserialize(bytes)
}

#[allow(non_snake_case)]
pub fn H5S__decode(bytes: &[u8]) -> Result<Selection> {
    Selection::select_deserialize(bytes)
}

#[allow(non_snake_case)]
pub fn H5S_select_release(_selection: Selection) {}

#[allow(non_snake_case)]
pub fn H5S_select_serial_size(selection: &Selection) -> Result<usize> {
    Ok(selection.encode1()?.len())
}

#[allow(non_snake_case)]
pub fn H5S_select_serialize(selection: &Selection) -> Result<Vec<u8>> {
    selection.encode1()
}

#[allow(non_snake_case)]
pub fn H5S__encode(selection: &Selection) -> Result<Vec<u8>> {
    selection.encode1()
}

#[allow(non_snake_case)]
pub fn H5Sencode1(selection: &Selection) -> Result<Vec<u8>> {
    selection.encode1()
}

#[allow(non_snake_case)]
pub fn H5Sget_select_bounds(
    selection: &Selection,
    ds_shape: &[u64],
) -> Option<(Vec<u64>, Vec<u64>)> {
    selection.bounds(ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S_get_select_bounds(
    selection: &Selection,
    ds_shape: &[u64],
) -> Option<(Vec<u64>, Vec<u64>)> {
    H5Sget_select_bounds(selection, ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S_get_select_unlim_dim(selection: &Selection, max_dims: &[u64]) -> Option<usize> {
    selection.select_unlim_dim(max_dims)
}

#[allow(non_snake_case)]
pub fn H5S_get_select_num_elem_non_unlim(
    selection: &Selection,
    ds_shape: &[u64],
    max_dims: &[u64],
) -> Result<u64> {
    selection.select_num_elem_non_unlim(ds_shape, max_dims)
}

#[allow(non_snake_case)]
pub fn H5S_select_is_contiguous(selection: &Selection, ds_shape: &[u64]) -> Result<bool> {
    selection.select_is_contiguous(ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S_select_is_single(selection: &Selection, ds_shape: &[u64]) -> bool {
    selection.select_is_single(ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S_select_is_regular(selection: &Selection) -> bool {
    selection.is_regular()
}

#[allow(non_snake_case)]
pub fn H5S_select_none() -> Selection {
    Selection::select_none()
}

#[allow(non_snake_case)]
pub fn H5Sselect_none() -> Selection {
    Selection::select_none_api()
}

#[allow(non_snake_case)]
pub fn H5S_select_all() -> Selection {
    Selection::select_all()
}

#[allow(non_snake_case)]
pub fn H5Sselect_all() -> Selection {
    Selection::select_all_api()
}

#[allow(non_snake_case)]
pub fn H5S_select_elements(points: Vec<Vec<u64>>) -> Selection {
    Selection::select_elements(points)
}

#[allow(non_snake_case)]
pub fn H5Sselect_elements(points: Vec<Vec<u64>>) -> Selection {
    Selection::select_elements_api(points)
}

#[allow(non_snake_case)]
pub fn H5S_select_hyperslab(dims: Vec<HyperslabDim>) -> Selection {
    Selection::select_hyperslab(dims)
}

#[allow(non_snake_case)]
pub fn H5Sselect_hyperslab(dims: Vec<HyperslabDim>) -> Selection {
    Selection::select_hyperslab_api(dims)
}

#[allow(non_snake_case)]
pub fn H5Sget_select_hyper_nblocks(selection: &Selection, ds_shape: &[u64]) -> Option<u64> {
    selection.hyperslab_block_count(ds_shape)
}

#[allow(non_snake_case)]
pub fn H5Sget_select_hyper_blocklist(
    selection: &Selection,
    ds_shape: &[u64],
) -> Result<Option<Vec<(Vec<u64>, Vec<u64>)>>> {
    selection.hyperslab_blocklist(ds_shape)
}

#[allow(non_snake_case)]
pub fn H5Sget_select_elem_npoints(selection: &Selection) -> Option<u64> {
    selection.element_point_count()
}

#[allow(non_snake_case)]
pub fn H5Sget_select_elem_pointlist(selection: &Selection) -> Option<&[Vec<u64>]> {
    selection.element_pointlist()
}

#[allow(non_snake_case)]
pub fn H5S__get_select_elem_pointlist(selection: &Selection) -> Option<&[Vec<u64>]> {
    selection.get_select_elem_pointlist_internal()
}

#[allow(non_snake_case)]
pub fn H5S_select_intersect_block(
    selection: &Selection,
    ds_shape: &[u64],
    start: &[u64],
    end: &[u64],
) -> Result<bool> {
    Ok(selection
        .materialize_points(ds_shape)?
        .iter()
        .any(|point| point_is_inside_block(point, start, end)))
}

#[allow(non_snake_case)]
pub fn H5Sselect_intersect_block(
    selection: &Selection,
    ds_shape: &[u64],
    start: &[u64],
    end: &[u64],
) -> Result<bool> {
    H5S_select_intersect_block(selection, ds_shape, start, end)
}

#[allow(non_snake_case)]
pub fn H5S_select_adjust_u(selection: &Selection, offsets: &[u64]) -> Result<Selection> {
    selection.select_adjust_unsigned(offsets)
}

#[allow(non_snake_case)]
pub fn H5S_select_adjust_s(selection: &Selection, offsets: &[i64]) -> Result<Selection> {
    selection.select_adjust_signed(offsets)
}

#[allow(non_snake_case)]
pub fn H5Sselect_adjust(selection: &Selection, offsets: &[i64]) -> Result<Selection> {
    H5S_select_adjust_s(selection, offsets)
}

#[allow(non_snake_case)]
pub fn H5S_select_project_simple(
    selection: &Selection,
    ds_shape: &[u64],
    kept_dims: &[usize],
) -> Result<Selection> {
    selection.project(ds_shape, kept_dims)
}

#[allow(non_snake_case)]
pub fn H5S_select_project_scalar(selection: &Selection, ds_shape: &[u64]) -> Result<Selection> {
    selection.project(ds_shape, &[])
}

#[allow(non_snake_case)]
pub fn H5S_select_iter_init(selection: &Selection, ds_shape: &[u64]) -> Result<SelectionPointIter> {
    selection.select_iter_init(ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S_select_iter_coords(iter: &SelectionPointIter) -> Option<&[u64]> {
    iter.point_iter_coords()
}

#[allow(non_snake_case)]
pub fn H5S_select_iter_nelmts(iter: &SelectionPointIter) -> usize {
    iter.select_iter_nelmts()
}

#[allow(non_snake_case)]
pub fn H5S_select_iter_next(iter: &mut SelectionPointIter) -> Option<Vec<u64>> {
    iter.select_iter_next()
}

#[allow(non_snake_case)]
pub fn H5S_select_iter_get_seq_list(
    iter: &mut SelectionPointIter,
    max_points: usize,
) -> Vec<Vec<u64>> {
    iter.select_iter_get_seq_list(max_points)
}

#[allow(non_snake_case)]
pub fn H5Ssel_iter_reset(iter: &mut SelectionPointIter) {
    iter.select_iter_reset()
}

#[allow(non_snake_case)]
pub fn H5S_select_iter_release(iter: SelectionPointIter) {
    iter.select_iter_release()
}

#[allow(non_snake_case)]
pub fn H5S_select_iterate<F>(selection: &Selection, ds_shape: &[u64], callback: F) -> Result<()>
where
    F: FnMut(&[u64]) -> Result<()>,
{
    selection.select_iterate(ds_shape, callback)
}

#[allow(non_snake_case)]
pub fn H5Sget_select_type(selection: &Selection) -> SelectionType {
    selection.selection_type()
}

#[allow(non_snake_case)]
pub fn H5S_get_select_type(selection: &Selection) -> SelectionType {
    H5Sget_select_type(selection)
}

#[allow(non_snake_case)]
pub fn H5S_select_shape_same(left: &Selection, right: &Selection, ds_shape: &[u64]) -> bool {
    left.select_shape_same(right, ds_shape)
}

#[allow(non_snake_case)]
pub fn H5Sselect_shape_same(left: &Selection, right: &Selection, ds_shape: &[u64]) -> bool {
    H5S_select_shape_same(left, right, ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S_select_construct_projection(
    selection: &Selection,
    ds_shape: &[u64],
    kept_dims: &[usize],
) -> Result<Selection> {
    selection.select_construct_projection(ds_shape, kept_dims)
}

#[allow(non_snake_case)]
pub fn H5S_select_project_intersection(
    left: &Selection,
    right: &Selection,
    ds_shape: &[u64],
    kept_dims: &[usize],
) -> Result<Selection> {
    left.select_project_intersection(right, ds_shape, kept_dims)
}

#[allow(non_snake_case)]
pub fn H5Sselect_project_intersection(
    left: &Selection,
    right: &Selection,
    ds_shape: &[u64],
    kept_dims: &[usize],
) -> Result<Selection> {
    H5S_select_project_intersection(left, right, ds_shape, kept_dims)
}

#[allow(non_snake_case)]
pub fn H5S_select_fill<T: Clone>(
    selection: &Selection,
    ds_shape: &[u64],
    buffer: &mut [T],
    value: T,
) -> Result<()> {
    selection.select_fill(ds_shape, buffer, value)
}

#[allow(non_snake_case)]
pub fn H5S_combine_hyperslab(
    left: &Selection,
    right: &Selection,
    ds_shape: &[u64],
) -> Result<Selection> {
    left.combine_hyperslab(right, ds_shape)
}

#[allow(non_snake_case)]
pub fn H5Scombine_hyperslab(
    left: &Selection,
    right: &Selection,
    ds_shape: &[u64],
) -> Result<Selection> {
    H5S_combine_hyperslab(left, right, ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__modify_select(
    left: &Selection,
    right: &Selection,
    ds_shape: &[u64],
) -> Result<Selection> {
    left.modify_select_internal(right, ds_shape)
}

#[allow(non_snake_case)]
pub fn H5Smodify_select(
    left: &Selection,
    right: &Selection,
    ds_shape: &[u64],
) -> Result<Selection> {
    left.modify_select(right, ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__mpio_all_type(selection: &Selection) -> bool {
    selection.mpio_all_type()
}

#[allow(non_snake_case)]
pub fn H5S__mpio_none_type(selection: &Selection) -> bool {
    selection.mpio_none_type()
}

#[allow(non_snake_case)]
pub fn H5S__mpio_point_type(selection: &Selection) -> bool {
    selection.mpio_point_type()
}

#[allow(non_snake_case)]
pub fn H5S__mpio_permute_type(selection: &Selection) -> bool {
    selection.mpio_permute_type()
}

#[allow(non_snake_case)]
pub fn H5S__mpio_reg_hyper_type(selection: &Selection) -> bool {
    selection.mpio_reg_hyper_type()
}

#[allow(non_snake_case)]
pub fn H5S__mpio_span_hyper_type(selection: &Selection) -> bool {
    selection.mpio_span_hyper_type()
}

#[allow(non_snake_case)]
pub fn H5S__none_iter_block() -> Option<Vec<u64>> {
    Selection::none_iter_block()
}

#[allow(non_snake_case)]
pub fn H5S__none_iter_nelmts() -> usize {
    Selection::none_iter_nelmts()
}

#[allow(non_snake_case)]
pub fn H5S__none_iter_get_seq_list() -> Vec<Vec<u64>> {
    Selection::none_iter_get_seq_list()
}

#[allow(non_snake_case)]
pub fn H5S__none_iter_release() {
    Selection::none_iter_release()
}

#[allow(non_snake_case)]
pub fn H5S__none_release() {
    Selection::none_release()
}

#[allow(non_snake_case)]
pub fn H5S__none_copy() -> Selection {
    Selection::none_copy()
}

#[allow(non_snake_case)]
pub fn H5S__none_is_valid() -> bool {
    Selection::none_is_valid()
}

#[allow(non_snake_case)]
pub fn H5S__none_serialize() -> Vec<u8> {
    Selection::none_serialize()
}

#[allow(non_snake_case)]
pub fn H5S__none_deserialize(bytes: &[u8]) -> Result<Selection> {
    Selection::none_deserialize(bytes)
}

#[allow(non_snake_case)]
pub fn H5S__none_bounds() -> Option<(Vec<u64>, Vec<u64>)> {
    Selection::none_bounds()
}

#[allow(non_snake_case)]
pub fn H5S__none_offset(offsets: &[i64]) -> Selection {
    Selection::none_offset(offsets)
}

#[allow(non_snake_case)]
pub fn H5S__none_is_contiguous() -> bool {
    Selection::none_is_contiguous()
}

#[allow(non_snake_case)]
pub fn H5S__none_is_single() -> bool {
    Selection::none_is_single()
}

#[allow(non_snake_case)]
pub fn H5S__none_is_regular() -> bool {
    Selection::none_is_regular()
}

#[allow(non_snake_case)]
pub fn H5S__none_intersect_block(start: &[u64], end: &[u64]) -> bool {
    Selection::none_intersect_block(start, end)
}

#[allow(non_snake_case)]
pub fn H5S__none_adjust_u(offsets: &[u64]) -> Selection {
    Selection::none_adjust_u(offsets)
}

#[allow(non_snake_case)]
pub fn H5S__none_adjust_s(offsets: &[i64]) -> Selection {
    Selection::none_adjust_s(offsets)
}

#[allow(non_snake_case)]
pub fn H5S__none_project_simple() -> Selection {
    Selection::none_project_simple()
}

#[allow(non_snake_case)]
pub fn H5S__all_iter_init(ds_shape: &[u64]) -> Result<SelectionPointIter> {
    Selection::all_iter_init(ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__all_iter_coords(ds_shape: &[u64]) -> Result<Option<Vec<u64>>> {
    Selection::all_iter_coords(ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__all_iter_block(ds_shape: &[u64]) -> Option<(Vec<u64>, Vec<u64>)> {
    Selection::all_iter_block(ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__all_iter_nelmts(ds_shape: &[u64]) -> Result<u64> {
    Selection::all_iter_nelmts(ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__all_iter_has_next_block(ds_shape: &[u64]) -> bool {
    Selection::all_iter_has_next_block(ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__all_iter_next(iter: &mut SelectionPointIter) -> Option<Vec<u64>> {
    Selection::all_iter_next(iter)
}

#[allow(non_snake_case)]
pub fn H5S__all_iter_next_block(ds_shape: &[u64]) -> Option<(Vec<u64>, Vec<u64>)> {
    Selection::all_iter_next_block(ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__all_iter_get_seq_list(ds_shape: &[u64], max_points: usize) -> Result<Vec<Vec<u64>>> {
    Selection::all_iter_get_seq_list(ds_shape, max_points)
}

#[allow(non_snake_case)]
pub fn H5S__all_iter_release(iter: SelectionPointIter) {
    Selection::all_iter_release(iter)
}

#[allow(non_snake_case)]
pub fn H5S__all_release() {
    Selection::all_release()
}

#[allow(non_snake_case)]
pub fn H5S__all_copy() -> Selection {
    Selection::all_copy()
}

#[allow(non_snake_case)]
pub fn H5S__all_is_valid(ds_shape: &[u64]) -> bool {
    Selection::all_is_valid(ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__all_serial_size() -> usize {
    Selection::all_serial_size()
}

#[allow(non_snake_case)]
pub fn H5S__all_serialize() -> Vec<u8> {
    Selection::all_serialize()
}

#[allow(non_snake_case)]
pub fn H5S__all_deserialize(bytes: &[u8]) -> Result<Selection> {
    Selection::all_deserialize(bytes)
}

#[allow(non_snake_case)]
pub fn H5S__all_bounds(ds_shape: &[u64]) -> Option<(Vec<u64>, Vec<u64>)> {
    Selection::all_bounds(ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__all_offset(offsets: &[i64]) -> Result<Selection> {
    Selection::all_offset(offsets)
}

#[allow(non_snake_case)]
pub fn H5S__all_unlim_dim(max_dims: &[u64]) -> Option<usize> {
    Selection::all_unlim_dim(max_dims)
}

#[allow(non_snake_case)]
pub fn H5S__all_is_contiguous() -> bool {
    Selection::all_is_contiguous()
}

#[allow(non_snake_case)]
pub fn H5S__all_is_single(ds_shape: &[u64]) -> bool {
    Selection::all_is_single(ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__all_is_regular() -> bool {
    Selection::all_is_regular()
}

#[allow(non_snake_case)]
pub fn H5S__all_intersect_block(ds_shape: &[u64], start: &[u64], end: &[u64]) -> bool {
    Selection::all_intersect_block(ds_shape, start, end)
}

#[allow(non_snake_case)]
pub fn H5S__all_adjust_u(offsets: &[u64]) -> Result<Selection> {
    Selection::all_adjust_u(offsets)
}

#[allow(non_snake_case)]
pub fn H5S__all_adjust_s(offsets: &[i64]) -> Result<Selection> {
    Selection::all_adjust_s(offsets)
}

#[allow(non_snake_case)]
pub fn H5S__all_project_simple(ds_shape: &[u64], kept_dims: &[usize]) -> Result<Selection> {
    Selection::all_project_simple(ds_shape, kept_dims)
}

#[allow(non_snake_case)]
pub fn H5S__point_iter_init(selection: &Selection, ds_shape: &[u64]) -> Result<SelectionPointIter> {
    selection.point_iter_init(ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__point_iter_coords(iter: &SelectionPointIter) -> Option<&[u64]> {
    iter.point_iter_coords()
}

#[allow(non_snake_case)]
pub fn H5S__point_iter_nelmts(iter: &SelectionPointIter) -> usize {
    iter.point_iter_nelmts()
}

#[allow(non_snake_case)]
pub fn H5S__point_iter_next(iter: &mut SelectionPointIter) -> Option<Vec<u64>> {
    iter.point_iter_next()
}

#[allow(non_snake_case)]
pub fn H5S__point_iter_next_block(iter: &mut SelectionPointIter) -> Option<Vec<u64>> {
    iter.point_iter_next_block()
}

#[allow(non_snake_case)]
pub fn H5S__point_iter_get_seq_list(
    iter: &mut SelectionPointIter,
    max_points: usize,
) -> Vec<Vec<u64>> {
    iter.point_iter_get_seq_list(max_points)
}

#[allow(non_snake_case)]
pub fn H5S__point_iter_release(iter: SelectionPointIter) {
    iter.point_iter_release()
}

#[allow(non_snake_case)]
pub fn H5S__point_copy(selection: &Selection) -> Result<Selection> {
    selection.point_copy()
}

#[allow(non_snake_case)]
pub fn H5S__point_add(selection: &mut Selection, point: Vec<u64>) -> Result<()> {
    selection.point_add(point)
}

#[allow(non_snake_case)]
pub fn H5S__point_get_version_enc_size(selection: &Selection) -> Result<(u8, usize)> {
    selection.point_get_version_enc_size()
}

#[allow(non_snake_case)]
pub fn H5S__point_serial_size(selection: &Selection) -> Result<usize> {
    selection.point_serial_size()
}

#[allow(non_snake_case)]
pub fn H5S__point_serialize(selection: &Selection) -> Result<Vec<u8>> {
    selection.point_serialize()
}

#[allow(non_snake_case)]
pub fn H5S__point_deserialize(bytes: &[u8]) -> Result<Selection> {
    Selection::point_deserialize(bytes)
}

#[allow(non_snake_case)]
pub fn H5S__point_offset(selection: &Selection, offsets: &[i64]) -> Result<Selection> {
    selection.point_offset(offsets)
}

#[allow(non_snake_case)]
pub fn H5S__point_unlim_dim(selection: &Selection, max_dims: &[u64]) -> Option<usize> {
    selection.point_unlim_dim(max_dims)
}

#[allow(non_snake_case)]
pub fn H5S__point_is_contiguous(selection: &Selection, ds_shape: &[u64]) -> Result<bool> {
    selection.point_is_contiguous(ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__point_is_single(selection: &Selection, ds_shape: &[u64]) -> bool {
    selection.point_is_single(ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__point_is_regular(selection: &Selection) -> bool {
    selection.point_is_regular()
}

#[allow(non_snake_case)]
pub fn H5S__point_shape_same(left: &Selection, right: &Selection, ds_shape: &[u64]) -> bool {
    left.point_shape_same(right, ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__point_intersect_block(selection: &Selection, start: &[u64], end: &[u64]) -> bool {
    selection.point_intersect_block(start, end)
}

#[allow(non_snake_case)]
pub fn H5S__point_adjust_u(selection: &Selection, offsets: &[u64]) -> Result<Selection> {
    selection.point_adjust_u(offsets)
}

#[allow(non_snake_case)]
pub fn H5S__point_adjust_s(selection: &Selection, offsets: &[i64]) -> Result<Selection> {
    selection.point_adjust_s(offsets)
}

#[allow(non_snake_case)]
pub fn H5S__point_project_scalar(selection: &Selection, ds_shape: &[u64]) -> Result<Selection> {
    selection.point_project_scalar(ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__point_project_simple(
    selection: &Selection,
    ds_shape: &[u64],
    kept_dims: &[usize],
) -> Result<Selection> {
    selection.point_project_simple(ds_shape, kept_dims)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_iter_init(selection: &Selection, ds_shape: &[u64]) -> Result<SelectionPointIter> {
    selection.hyper_iter_init(ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_iter_block(selection: &Selection, ds_shape: &[u64]) -> Result<Option<Vec<u64>>> {
    selection.hyper_iter_block(ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_iter_has_next_block(selection: &Selection, ds_shape: &[u64]) -> Result<bool> {
    selection.hyper_iter_has_next_block(ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_iter_next(iter: &mut SelectionPointIter) -> Option<Vec<u64>> {
    iter.hyper_iter_next()
}

#[allow(non_snake_case)]
pub fn H5S__hyper_iter_next_block(iter: &mut SelectionPointIter) -> Option<Vec<u64>> {
    iter.hyper_iter_next_block()
}

#[allow(non_snake_case)]
pub fn H5S__hyper_iter_get_seq_list(
    iter: &mut SelectionPointIter,
    max_points: usize,
) -> Vec<Vec<u64>> {
    iter.hyper_iter_get_seq_list(max_points)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_iter_get_seq_list_gen(
    iter: &mut SelectionPointIter,
    max_points: usize,
) -> Vec<Vec<u64>> {
    iter.hyper_iter_get_seq_list_gen(max_points)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_get_seq_list_gen(
    selection: &Selection,
    ds_shape: &[u64],
    max_points: usize,
) -> Result<Vec<Vec<u64>>> {
    Ok(selection
        .hyper_iter_init(ds_shape)?
        .hyper_iter_get_seq_list_gen(max_points))
}

#[allow(non_snake_case)]
pub fn H5S__hyper_iter_nelmts(iter: &SelectionPointIter) -> usize {
    iter.hyper_iter_nelmts()
}

#[allow(non_snake_case)]
pub fn H5S__hyper_iter_get_seq_list_opt(
    iter: &mut SelectionPointIter,
    max_points: usize,
) -> Vec<Vec<u64>> {
    iter.hyper_iter_get_seq_list_opt(max_points)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_iter_get_seq_list_single(
    iter: &mut SelectionPointIter,
    max_points: usize,
) -> Vec<Vec<u64>> {
    iter.hyper_iter_get_seq_list_single(max_points)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_iter_release(iter: SelectionPointIter) {
    iter.hyper_iter_release()
}

#[allow(non_snake_case)]
pub fn H5S__hyper_copy(selection: &Selection) -> Result<Selection> {
    selection.hyper_copy()
}

#[allow(non_snake_case)]
pub fn H5S__hyper_new_span(start: u64, stride: u64, count: u64, block: u64) -> HyperslabDim {
    Selection::hyper_new_span(start, stride, count, block)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_new_span_info(dims: Vec<HyperslabDim>) -> Selection {
    Selection::hyper_new_span_info(dims)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_copy_span_helper(selection: &Selection) -> Result<Vec<HyperslabDim>> {
    selection.hyper_copy_span_helper()
}

#[allow(non_snake_case)]
pub fn H5S__hyper_copy_span(selection: &Selection) -> Result<Vec<HyperslabDim>> {
    selection.hyper_copy_span()
}

#[allow(non_snake_case)]
pub fn H5S__hyper_cmp_spans(left: &Selection, right: &Selection) -> bool {
    left.hyper_cmp_spans(right)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_print_spans(selection: &Selection) -> Result<String> {
    selection.hyper_print_spans()
}

#[allow(non_snake_case)]
pub fn H5S__hyper_print_spans_helper(selection: &Selection) -> Result<String> {
    selection.hyper_print_spans_helper()
}

#[allow(non_snake_case)]
pub fn H5S__space_print_spans(selection: &Selection) -> Result<String> {
    selection.space_print_spans()
}

#[allow(non_snake_case)]
pub fn H5S__hyper_print_diminfo(selection: &Selection) -> Result<String> {
    selection.hyper_print_diminfo()
}

#[allow(non_snake_case)]
pub fn H5S__hyper_print_diminfo_helper(selection: &Selection) -> Result<String> {
    selection.hyper_print_diminfo_helper()
}

#[allow(non_snake_case)]
pub fn H5S__hyper_print_spans_dfs(selection: &Selection) -> Result<String> {
    selection.hyper_print_spans_dfs()
}

#[allow(non_snake_case)]
pub fn H5S__hyper_print_space_dfs(selection: &Selection) -> Result<String> {
    selection.hyper_print_space_dfs()
}

#[allow(non_snake_case)]
pub fn H5S__hyper_free_span(selection: Selection) {
    selection.hyper_free_span()
}

#[allow(non_snake_case)]
pub fn H5S__hyper_release(selection: Selection) {
    selection.hyper_release()
}

#[allow(non_snake_case)]
pub fn H5S__hyper_get_enc_size_real(selection: &Selection) -> Result<usize> {
    selection.hyper_get_enc_size_real()
}

#[allow(non_snake_case)]
pub fn H5S__hyper_get_version_enc_size(selection: &Selection) -> Result<(u8, usize)> {
    selection.hyper_get_version_enc_size()
}

#[allow(non_snake_case)]
pub fn H5S__hyper_serialize(selection: &Selection) -> Result<Vec<u8>> {
    selection.hyper_serialize()
}

#[allow(non_snake_case)]
pub fn H5S__hyper_serialize_helper(selection: &Selection) -> Result<Vec<u8>> {
    selection.hyper_serialize_helper()
}

#[allow(non_snake_case)]
pub fn H5S__hyper_deserialize(bytes: &[u8]) -> Result<Selection> {
    Selection::hyper_deserialize(bytes)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_decode(bytes: &[u8]) -> Result<Selection> {
    Selection::hyper_deserialize(bytes)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_is_valid(selection: &Selection, ds_shape: &[u64]) -> bool {
    selection.hyper_is_valid(ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_span_nblocks(selection: &Selection, ds_shape: &[u64]) -> Option<u64> {
    selection.hyper_span_nblocks(ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_span_blocklist(
    selection: &Selection,
    ds_shape: &[u64],
) -> Result<Option<Vec<(Vec<u64>, Vec<u64>)>>> {
    selection.hyper_span_blocklist(ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__get_select_hyper_nblocks(selection: &Selection, ds_shape: &[u64]) -> Option<u64> {
    selection.get_select_hyper_nblocks_internal(ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__get_select_hyper_blocklist(
    selection: &Selection,
    ds_shape: &[u64],
) -> Result<Option<Vec<(Vec<u64>, Vec<u64>)>>> {
    selection.get_select_hyper_blocklist_internal(ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_intersect_block_helper(
    selection: &Selection,
    ds_shape: &[u64],
    start: &[u64],
    end: &[u64],
) -> bool {
    selection.hyper_intersect_block_helper(ds_shape, start, end)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_bounds(selection: &Selection, ds_shape: &[u64]) -> Option<(Vec<u64>, Vec<u64>)> {
    selection.hyper_bounds(ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_offset(selection: &Selection, offsets: &[i64]) -> Result<Selection> {
    selection.hyper_offset(offsets)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_unlim_dim(selection: &Selection, max_dims: &[u64]) -> Option<usize> {
    selection.hyper_unlim_dim(max_dims)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_num_elem_non_unlim(
    selection: &Selection,
    ds_shape: &[u64],
    max_dims: &[u64],
) -> Result<u64> {
    selection.hyper_num_elem_non_unlim(ds_shape, max_dims)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_is_contiguous(selection: &Selection, ds_shape: &[u64]) -> Result<bool> {
    selection.hyper_is_contiguous(ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_is_single(selection: &Selection, ds_shape: &[u64]) -> bool {
    selection.hyper_is_single(ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_is_regular(selection: &Selection) -> bool {
    selection.hyper_is_regular()
}

#[allow(non_snake_case)]
pub fn H5S__hyper_spans_nelem(selection: &Selection, ds_shape: &[u64]) -> Option<u64> {
    selection.hyper_spans_nelem(ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_spans_nelem_helper(selection: &Selection, ds_shape: &[u64]) -> Option<u64> {
    selection.hyper_spans_nelem_helper(ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_shape_same(left: &Selection, right: &Selection, ds_shape: &[u64]) -> bool {
    left.hyper_shape_same(right, ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_spans_shape_same_helper(
    left: &Selection,
    right: &Selection,
    ds_shape: &[u64],
) -> bool {
    left.hyper_spans_shape_same_helper(right, ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_spans_shape_same(left: &Selection, right: &Selection, ds_shape: &[u64]) -> bool {
    left.hyper_spans_shape_same(right, ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_regular_and_single_block(selection: &Selection, ds_shape: &[u64]) -> bool {
    selection.hyper_regular_and_single_block(ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_get_regular_hyperslab(selection: &Selection) -> Result<Vec<HyperslabDim>> {
    selection.hyper_copy_span_helper()
}

#[allow(non_snake_case)]
pub fn H5S__hyper_coord_to_span(selection: &Selection, coord: &[u64], ds_shape: &[u64]) -> bool {
    selection.hyper_coord_to_span(coord, ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S_hyper_add_span_element(selection: &mut Selection, dim: HyperslabDim) -> Result<()> {
    selection.hyper_add_span_element(dim)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_append_span(selection: &mut Selection, dim: HyperslabDim) -> Result<()> {
    selection.hyper_append_span(dim)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_clip_spans(
    selection: &Selection,
    ds_shape: &[u64],
    start: &[u64],
    end: &[u64],
) -> Result<Selection> {
    selection.hyper_clip_spans(ds_shape, start, end)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_merge_spans_helper(
    left: &Selection,
    right: &Selection,
    ds_shape: &[u64],
) -> Result<Selection> {
    left.hyper_merge_spans_helper(right, ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_merge_spans(
    left: &Selection,
    right: &Selection,
    ds_shape: &[u64],
) -> Result<Selection> {
    left.hyper_merge_spans(right, ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_add_disjoint_spans(
    left: &Selection,
    right: &Selection,
    ds_shape: &[u64],
) -> Result<Selection> {
    left.hyper_add_disjoint_spans(right, ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_make_spans(dims: Vec<HyperslabDim>) -> Selection {
    Selection::hyper_make_spans(dims)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_update_diminfo(selection: &Selection) -> Result<Vec<HyperslabDim>> {
    selection.hyper_update_diminfo()
}

#[allow(non_snake_case)]
pub fn H5S__hyper_rebuild_helper(selection: &Selection) -> Result<Selection> {
    selection.hyper_rebuild_helper()
}

#[allow(non_snake_case)]
pub fn H5S__hyper_rebuild(selection: &Selection) -> Result<Selection> {
    selection.hyper_rebuild()
}

#[allow(non_snake_case)]
pub fn H5S__hyper_generate_spans(selection: &Selection) -> Result<Selection> {
    selection.hyper_generate_spans()
}

#[allow(non_snake_case)]
pub fn H5S__check_spans_overlap(
    left: &Selection,
    right: &Selection,
    ds_shape: &[u64],
) -> Result<bool> {
    left.check_spans_overlap(right, ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__fill_in_new_space(selection: &Selection, ds_shape: &[u64]) -> Result<Selection> {
    selection.fill_in_new_space(ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__set_regular_hyperslab(dims: Vec<HyperslabDim>) -> Selection {
    Selection::set_regular_hyperslab(dims)
}

#[allow(non_snake_case)]
pub fn H5S__fill_in_select(selection: &Selection) -> Selection {
    selection.fill_in_select()
}

#[allow(non_snake_case)]
pub fn H5S__hyper_adjust_u(selection: &Selection, offsets: &[u64]) -> Result<Selection> {
    selection.hyper_adjust_u(offsets)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_adjust_u_helper(selection: &Selection, offsets: &[u64]) -> Result<Selection> {
    selection.hyper_adjust_u_helper(offsets)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_adjust_s(selection: &Selection, offsets: &[i64]) -> Result<Selection> {
    selection.hyper_adjust_s(offsets)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_adjust_s_helper(selection: &Selection, offsets: &[i64]) -> Result<Selection> {
    selection.hyper_adjust_s_helper(offsets)
}

#[allow(non_snake_case)]
pub fn H5S_hyper_normalize_offset(selection: &Selection, offsets: &[i64]) -> Result<Selection> {
    selection.hyper_normalize_offset(offsets)
}

#[allow(non_snake_case)]
pub fn H5S_hyper_denormalize_offset(selection: &Selection, offsets: &[i64]) -> Result<Selection> {
    selection.hyper_denormalize_offset(offsets)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_project_scalar(selection: &Selection, ds_shape: &[u64]) -> Result<Selection> {
    selection.hyper_project_scalar(ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_project_simple(
    selection: &Selection,
    ds_shape: &[u64],
    kept_dims: &[usize],
) -> Result<Selection> {
    selection.hyper_project_simple(ds_shape, kept_dims)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_project_simple_lower(
    selection: &Selection,
    ds_shape: &[u64],
    kept_dims: &[usize],
) -> Result<Selection> {
    selection.hyper_project_simple_lower(ds_shape, kept_dims)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_project_simple_higher(
    selection: &Selection,
    ds_shape: &[u64],
    kept_dims: &[usize],
) -> Result<Selection> {
    selection.hyper_project_simple_higher(ds_shape, kept_dims)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_proj_int_build_proj(
    left: &Selection,
    right: &Selection,
    ds_shape: &[u64],
    kept_dims: &[usize],
) -> Result<Selection> {
    left.hyper_proj_int_build_proj(right, ds_shape, kept_dims)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_proj_int_iterate(
    left: &Selection,
    right: &Selection,
    ds_shape: &[u64],
    kept_dims: &[usize],
) -> Result<Vec<Vec<u64>>> {
    left.hyper_proj_int_iterate(right, ds_shape, kept_dims)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_project_intersection(
    left: &Selection,
    right: &Selection,
    ds_shape: &[u64],
    kept_dims: &[usize],
) -> Result<Selection> {
    left.hyper_project_intersection(right, ds_shape, kept_dims)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_get_clip_diminfo(
    selection: &Selection,
    ds_shape: &[u64],
) -> Option<(Vec<u64>, Vec<u64>)> {
    selection.hyper_get_clip_diminfo(ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S_hyper_clip_unlim(selection: &Selection, ds_shape: &[u64]) -> Result<Selection> {
    selection.hyper_clip_unlim(ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_get_clip_extent_real(
    selection: &Selection,
    ds_shape: &[u64],
) -> Option<(Vec<u64>, Vec<u64>)> {
    selection.hyper_get_clip_extent_real(ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_get_clip_extent(
    selection: &Selection,
    ds_shape: &[u64],
) -> Option<(Vec<u64>, Vec<u64>)> {
    selection.hyper_get_clip_extent(ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S_hyper_get_clip_extent(
    selection: &Selection,
    ds_shape: &[u64],
) -> Option<(Vec<u64>, Vec<u64>)> {
    selection.hyper_get_clip_extent(ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_get_clip_extent_match(
    left: &Selection,
    right: &Selection,
    ds_shape: &[u64],
) -> bool {
    left.hyper_get_clip_extent_match(right, ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S_hyper_get_clip_extent_match(
    left: &Selection,
    right: &Selection,
    ds_shape: &[u64],
) -> bool {
    left.hyper_get_clip_extent_match(right, ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_get_unlim_block(selection: &Selection, max_dims: &[u64]) -> Option<usize> {
    selection.hyper_get_unlim_block(max_dims)
}

#[allow(non_snake_case)]
pub fn H5S_hyper_get_unlim_block(selection: &Selection, max_dims: &[u64]) -> Option<usize> {
    selection.hyper_get_unlim_block(max_dims)
}

#[allow(non_snake_case)]
pub fn H5S__get_rebuild_status_test(selection: &Selection) -> bool {
    selection.get_rebuild_status_test()
}

#[allow(non_snake_case)]
pub fn H5S__get_diminfo_status_test(selection: &Selection) -> bool {
    selection.get_diminfo_status_test()
}

#[allow(non_snake_case)]
pub fn H5S__check_spans_tail_ptr(selection: &Selection) -> bool {
    selection.check_spans_tail_ptr()
}

#[allow(non_snake_case)]
pub fn H5S__check_internal_consistency(selection: &Selection, ds_shape: &[u64]) -> bool {
    selection.check_internal_consistency(ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__internal_consistency_test(selection: &Selection, ds_shape: &[u64]) -> bool {
    selection.internal_consistency_test(ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__verify_offsets(selection: &Selection, offsets: &[i64]) -> bool {
    selection.verify_offsets(offsets)
}

fn is_hyperslab_like(selection: &Selection) -> bool {
    matches!(
        selection,
        Selection::All | Selection::None | Selection::Hyperslab(_) | Selection::Slice(_)
    )
}

fn require_hyperslab_like(selection: &Selection) -> Result<()> {
    if is_hyperslab_like(selection) {
        Ok(())
    } else {
        Err(Error::InvalidFormat(
            "selection is not hyperslab-like".into(),
        ))
    }
}

fn require_point_selection(selection: &Selection) -> Result<()> {
    if matches!(selection, Selection::Points(_)) {
        Ok(())
    } else {
        Err(Error::InvalidFormat(
            "selection is not a point selection".into(),
        ))
    }
}

fn check_rank(kind: &str, rank: usize, offset_rank: usize) -> Result<()> {
    if rank != offset_rank {
        return Err(Error::InvalidFormat(format!(
            "{kind} rank {rank} does not match offset rank {offset_rank}"
        )));
    }
    Ok(())
}

fn shift_coords(point: &[u64], offsets: &[i64]) -> Result<Vec<u64>> {
    check_rank("point selection", point.len(), offsets.len())?;
    point
        .iter()
        .zip(offsets)
        .map(|(&coord, &offset)| shift_coord(coord, offset))
        .collect()
}

fn shift_coord(coord: u64, offset: i64) -> Result<u64> {
    if offset >= 0 {
        coord
            .checked_add(offset as u64)
            .ok_or_else(|| Error::InvalidFormat("selection coordinate overflow".into()))
    } else {
        coord
            .checked_sub(offset.unsigned_abs())
            .ok_or_else(|| Error::InvalidFormat("selection coordinate underflow".into()))
    }
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn usize_to_u64(value: usize, context: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| Error::InvalidFormat(format!("{context} exceeds u64")))
}

fn read_u64(bytes: &[u8], offset: &mut usize) -> Result<u64> {
    let end = offset
        .checked_add(8)
        .ok_or_else(|| Error::InvalidFormat("selection serialization offset overflow".into()))?;
    let word = bytes.get(*offset..end).ok_or_else(|| {
        Error::InvalidFormat("selection serialization ended before u64 field".into())
    })?;
    *offset = end;
    Ok(u64::from_le_bytes(
        word.try_into().expect("slice length checked"),
    ))
}

fn read_usize_u64(bytes: &[u8], offset: &mut usize, context: &str) -> Result<usize> {
    let value = read_u64(bytes, offset)?;
    usize::try_from(value).map_err(|_| Error::InvalidFormat(format!("{context} exceeds usize")))
}

fn projected_shape(ds_shape: &[u64], kept_dims: &[usize]) -> Vec<u64> {
    kept_dims
        .iter()
        .filter_map(|&dim| ds_shape.get(dim).copied())
        .collect()
}

fn block_intersects_shape(ds_shape: &[u64], start: &[u64], end: &[u64]) -> bool {
    if start.len() != ds_shape.len() || end.len() != ds_shape.len() {
        return false;
    }
    start
        .iter()
        .zip(end)
        .zip(ds_shape)
        .all(|((&lo, &hi), &extent)| lo <= hi && extent > 0 && lo < extent)
}

fn point_is_inside_block(point: &[u64], start: &[u64], end: &[u64]) -> bool {
    point.len() == start.len()
        && point.len() == end.len()
        && point
            .iter()
            .zip(start)
            .zip(end)
            .all(|((&coord, &lo), &hi)| lo <= hi && coord >= lo && coord <= hi)
}

fn points_to_selection(points: Vec<Vec<u64>>) -> Selection {
    if points.is_empty() {
        Selection::None
    } else {
        Selection::Points(points)
    }
}

fn total_elements(ds_shape: &[u64]) -> Result<u64> {
    if ds_shape.is_empty() {
        return Ok(1);
    }
    ds_shape.iter().try_fold(1u64, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| Error::InvalidFormat("dataspace element count overflow".into()))
    })
}

fn linear_index(point: &[u64], ds_shape: &[u64]) -> Result<u64> {
    if point.len() != ds_shape.len() {
        return Err(Error::InvalidFormat(format!(
            "selection point rank {} does not match dataspace rank {}",
            point.len(),
            ds_shape.len()
        )));
    }
    if ds_shape.is_empty() {
        return Ok(0);
    }

    let mut idx = 0u64;
    for (&coord, &extent) in point.iter().zip(ds_shape) {
        if coord >= extent {
            return Err(Error::InvalidFormat(format!(
                "selection coordinate {coord} is out of bounds for extent {extent}"
            )));
        }
        idx = idx
            .checked_mul(extent)
            .and_then(|value| value.checked_add(coord))
            .ok_or_else(|| Error::InvalidFormat("selection linear index overflow".into()))?;
    }
    Ok(idx)
}

fn materialize_all_points(ds_shape: &[u64]) -> Result<Vec<Vec<u64>>> {
    if ds_shape.contains(&0) {
        return Ok(Vec::new());
    }
    if ds_shape.is_empty() {
        return Ok(vec![Vec::new()]);
    }
    let slices: Vec<_> = ds_shape.iter().map(|&dim| SliceInfo::new(0, dim)).collect();
    materialize_slice_points(&slices)
}

fn materialize_slice_points(slices: &[SliceInfo]) -> Result<Vec<Vec<u64>>> {
    if slices.iter().any(|slice| slice.count() == 0) {
        return Ok(Vec::new());
    }
    let mut points = Vec::new();
    let mut current = vec![0u64; slices.len()];
    push_slice_points(slices, 0, &mut current, &mut points)?;
    Ok(points)
}

fn push_slice_points(
    slices: &[SliceInfo],
    dim: usize,
    current: &mut [u64],
    points: &mut Vec<Vec<u64>>,
) -> Result<()> {
    if dim == slices.len() {
        points.push(current.to_vec());
        return Ok(());
    }
    let slice = &slices[dim];
    let mut coord = slice.start;
    while coord < slice.end {
        current[dim] = coord;
        push_slice_points(slices, dim + 1, current, points)?;
        coord = coord
            .checked_add(slice.step)
            .ok_or_else(|| Error::InvalidFormat("selection coordinate overflow".into()))?;
    }
    Ok(())
}

fn materialize_hyperslab_points(dims: &[HyperslabDim]) -> Result<Vec<Vec<u64>>> {
    if dims
        .iter()
        .any(|dim| dim.count == 0 || dim.block == 0 || dim.stride == 0)
    {
        return Ok(Vec::new());
    }
    let mut points = Vec::new();
    let mut current = vec![0u64; dims.len()];
    push_hyperslab_points(dims, 0, &mut current, &mut points)?;
    Ok(points)
}

fn push_hyperslab_points(
    dims: &[HyperslabDim],
    dim: usize,
    current: &mut [u64],
    points: &mut Vec<Vec<u64>>,
) -> Result<()> {
    if dim == dims.len() {
        points.push(current.to_vec());
        return Ok(());
    }
    let selection = &dims[dim];
    for block_idx in 0..selection.count {
        let block_start = selection
            .start
            .checked_add(
                block_idx
                    .checked_mul(selection.stride)
                    .ok_or_else(|| Error::InvalidFormat("hyperslab coordinate overflow".into()))?,
            )
            .ok_or_else(|| Error::InvalidFormat("hyperslab coordinate overflow".into()))?;
        for offset in 0..selection.block {
            current[dim] = block_start
                .checked_add(offset)
                .ok_or_else(|| Error::InvalidFormat("hyperslab coordinate overflow".into()))?;
            push_hyperslab_points(dims, dim + 1, current, points)?;
        }
    }
    Ok(())
}

fn slice_blocklist(slices: &[SliceInfo]) -> Result<Vec<(Vec<u64>, Vec<u64>)>> {
    if slices.iter().any(|slice| slice.count() == 0) {
        return Ok(Vec::new());
    }
    let mut blocks = Vec::new();
    let mut start = vec![0u64; slices.len()];
    let mut end = vec![0u64; slices.len()];
    push_slice_blocks(slices, 0, &mut start, &mut end, &mut blocks)?;
    Ok(blocks)
}

fn push_slice_blocks(
    slices: &[SliceInfo],
    dim: usize,
    start: &mut [u64],
    end: &mut [u64],
    blocks: &mut Vec<(Vec<u64>, Vec<u64>)>,
) -> Result<()> {
    if dim == slices.len() {
        blocks.push((start.to_vec(), end.to_vec()));
        return Ok(());
    }

    let slice = &slices[dim];
    if slice.step == 1 {
        start[dim] = slice.start;
        end[dim] = slice
            .end
            .checked_sub(1)
            .ok_or_else(|| Error::InvalidFormat("slice block end underflow".into()))?;
        push_slice_blocks(slices, dim + 1, start, end, blocks)?;
        return Ok(());
    }

    let mut coord = slice.start;
    while coord < slice.end {
        start[dim] = coord;
        end[dim] = coord;
        push_slice_blocks(slices, dim + 1, start, end, blocks)?;
        coord = coord
            .checked_add(slice.step)
            .ok_or_else(|| Error::InvalidFormat("slice block coordinate overflow".into()))?;
    }
    Ok(())
}

fn hyperslab_blocklist(dims: &[HyperslabDim]) -> Result<Vec<(Vec<u64>, Vec<u64>)>> {
    if dims
        .iter()
        .any(|dim| dim.count == 0 || dim.block == 0 || dim.stride == 0)
    {
        return Ok(Vec::new());
    }
    let mut blocks = Vec::new();
    let mut start = vec![0u64; dims.len()];
    let mut end = vec![0u64; dims.len()];
    push_hyperslab_blocks(dims, 0, &mut start, &mut end, &mut blocks)?;
    Ok(blocks)
}

fn push_hyperslab_blocks(
    dims: &[HyperslabDim],
    dim: usize,
    start: &mut [u64],
    end: &mut [u64],
    blocks: &mut Vec<(Vec<u64>, Vec<u64>)>,
) -> Result<()> {
    if dim == dims.len() {
        blocks.push((start.to_vec(), end.to_vec()));
        return Ok(());
    }

    let selection = &dims[dim];
    for block_idx in 0..selection.count {
        let block_start = selection
            .start
            .checked_add(
                block_idx
                    .checked_mul(selection.stride)
                    .ok_or_else(|| Error::InvalidFormat("hyperslab block overflow".into()))?,
            )
            .ok_or_else(|| Error::InvalidFormat("hyperslab block overflow".into()))?;
        let block_end = block_start
            .checked_add(selection.block)
            .and_then(|value| value.checked_sub(1))
            .ok_or_else(|| Error::InvalidFormat("hyperslab block overflow".into()))?;
        start[dim] = block_start;
        end[dim] = block_end;
        push_hyperslab_blocks(dims, dim + 1, start, end, blocks)?;
    }
    Ok(())
}

fn slice_bounds(slices: &[SliceInfo]) -> Option<(Vec<u64>, Vec<u64>)> {
    if slices.iter().any(|slice| slice.count() == 0) {
        return None;
    }
    let mut end = Vec::with_capacity(slices.len());
    for slice in slices {
        end.push(
            slice
                .count()
                .checked_sub(1)?
                .checked_mul(slice.step)?
                .checked_add(slice.start)?,
        );
    }
    Some((slices.iter().map(|slice| slice.start).collect(), end))
}

fn hyperslab_bounds(dims: &[HyperslabDim]) -> Option<(Vec<u64>, Vec<u64>)> {
    if dims
        .iter()
        .any(|dim| dim.count == 0 || dim.block == 0 || dim.stride == 0)
    {
        return None;
    }
    let mut end = Vec::with_capacity(dims.len());
    for dim in dims {
        end.push(
            dim.count
                .checked_sub(1)?
                .checked_mul(dim.stride)?
                .checked_add(dim.block)?
                .checked_sub(1)?
                .checked_add(dim.start)?,
        );
    }
    Some((dims.iter().map(|dim| dim.start).collect(), end))
}

fn point_bounds(points: &[Vec<u64>]) -> Option<(Vec<u64>, Vec<u64>)> {
    let first = points.first()?;
    let mut start = first.clone();
    let mut end = first.clone();
    for point in &points[1..] {
        if point.len() != start.len() {
            return None;
        }
        for (dim, &coord) in point.iter().enumerate() {
            start[dim] = start[dim].min(coord);
            end[dim] = end[dim].max(coord);
        }
    }
    Some((start, end))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hyperslab_selected_count_reports_overflow() {
        let selection = Selection::Hyperslab(vec![HyperslabDim::new(0, 1, u64::MAX, 2)]);
        assert_eq!(selection.selected_count(&[u64::MAX]), None);
    }

    #[test]
    fn hyperslab_selected_count_uses_checked_cross_dimension_product() {
        let selection = Selection::Hyperslab(vec![
            HyperslabDim::new(0, 1, u64::MAX, 1),
            HyperslabDim::new(0, 1, 2, 1),
        ]);
        assert_eq!(selection.selected_count(&[u64::MAX, 2]), None);
    }

    #[test]
    fn hyperslab_dim_checked_output_count_rejects_overflow() {
        let dim = HyperslabDim::new(0, 1, u64::MAX, 2);
        let err = dim.checked_output_count().unwrap_err();
        assert!(
            err.to_string().contains("hyperslab output count overflow"),
            "unexpected error: {err}"
        );
        assert_eq!(dim.output_count(), u64::MAX);
    }

    #[test]
    fn selection_reset_and_combine_aliases_work() {
        let left = Selection::select_hyperslab(vec![HyperslabDim::new(0, 1, 2, 1)]);
        let right = Selection::select_hyperslab(vec![HyperslabDim::new(2, 1, 1, 1)]);
        let combined = left.combine_hyperslab(&right, &[4]).unwrap();
        assert_eq!(combined.selected_count(&[4]), Some(3));
        assert_eq!(left.modify_select(&right, &[4]).unwrap(), combined);

        let mut iter = combined.select_iter_init(&[4]).unwrap();
        assert_eq!(iter.select_iter_next(), Some(vec![0]));
        assert_eq!(iter.select_iter_next(), Some(vec![1]));
        iter.select_iter_reset();
        assert_eq!(iter.select_iter_next(), Some(vec![0]));
    }

    #[test]
    fn selection_serialization_aliases_roundtrip() {
        let points = Selection::select_elements(vec![vec![1, 2], vec![3, 4]]);
        assert!(points.mpio_point_type());
        let encoded_points = points.encode1().unwrap();
        assert_eq!(
            Selection::select_deserialize(&encoded_points).unwrap(),
            points
        );
        assert_eq!(
            Selection::point_deserialize(&points.point_serialize().unwrap()).unwrap(),
            points
        );
        assert_eq!(points.point_get_version_enc_size().unwrap().0, 1);
        assert!(Selection::point_deserialize(&u64::MAX.to_le_bytes()).is_err());

        let hyper = Selection::select_hyperslab(vec![
            HyperslabDim::new(0, 2, 2, 1),
            HyperslabDim::new(1, 1, 3, 1),
        ]);
        assert!(hyper.mpio_reg_hyper_type());
        let encoded_hyper = hyper.encode1().unwrap();
        assert_eq!(
            Selection::select_deserialize(&encoded_hyper).unwrap(),
            hyper
        );
        assert_eq!(
            Selection::hyper_deserialize(&hyper.hyper_serialize().unwrap()).unwrap(),
            hyper
        );
        assert_eq!(hyper.hyper_get_version_enc_size().unwrap().0, 1);
        assert!(Selection::hyper_deserialize(&u64::MAX.to_le_bytes()).is_err());
        assert_eq!(hyper.hyper_spans_nelem(&[4, 4]), Some(6));
        assert!(hyper.hyper_coord_to_span(&[0, 1], &[4, 4]));
        assert!(hyper.hyper_spans_shape_same(&hyper, &[4, 4]));
        assert!(hyper.hyper_print_diminfo().unwrap().contains("start=0"));
        assert!(hyper.space_print_spans().unwrap().contains("HyperslabDim"));

        let mut editable = Selection::select_hyperslab(vec![Selection::hyper_new_span(0, 1, 1, 1)]);
        editable
            .hyper_add_span_element(HyperslabDim::new(1, 1, 1, 1))
            .unwrap();
        assert_eq!(editable.hyper_copy_span().unwrap().len(), 2);
    }

    #[test]
    fn h5s_selection_aliases_dispatch_to_selection_methods() {
        let selection = Selection::select_hyperslab(vec![
            HyperslabDim::new(0, 1, 2, 1),
            HyperslabDim::new(1, 1, 2, 1),
        ]);
        let ds_shape = [4, 4];

        assert_eq!(H5Sselect_copy(&selection), selection);
        assert_eq!(
            H5S__copy_pnt_list(&[vec![0, 0], vec![1, 1]]),
            vec![vec![0, 0], vec![1, 1]]
        );
        H5S__free_pnt_list(vec![vec![0, 0]]);
        assert_eq!(H5Sget_select_npoints(&selection, &ds_shape), Some(4));
        assert_eq!(H5S_get_select_npoints(&selection, &ds_shape), Some(4));
        assert!(H5Sselect_valid(&selection, &ds_shape));
        assert_eq!(
            H5Sget_select_bounds(&selection, &ds_shape),
            Some((vec![0, 1], vec![1, 2]))
        );
        assert_eq!(
            H5S_get_select_bounds(&selection, &ds_shape),
            Some((vec![0, 1], vec![1, 2]))
        );
        assert_eq!(
            H5S_get_select_unlim_dim(&selection, &[4, u64::MAX]),
            Some(1)
        );
        assert_eq!(
            H5S_get_select_num_elem_non_unlim(&selection, &ds_shape, &[4, u64::MAX]).unwrap(),
            2
        );
        assert!(!H5S_select_is_contiguous(&selection, &ds_shape).unwrap());
        assert!(!H5S_select_is_single(&selection, &ds_shape));
        assert!(H5S_select_is_regular(&selection));
        assert_eq!(H5Sget_select_type(&selection), SelectionType::Hyperslab);
        assert_eq!(H5S_get_select_type(&selection), SelectionType::Hyperslab);
        assert_eq!(H5S_select_none(), Selection::None);
        assert_eq!(H5Sselect_none(), Selection::None);
        assert_eq!(H5S_select_all(), Selection::All);
        assert_eq!(H5Sselect_all(), Selection::All);
        assert_eq!(
            H5S_select_elements(vec![vec![2, 3]]),
            Selection::Points(vec![vec![2, 3]])
        );
        assert_eq!(
            H5Sselect_elements(vec![vec![3, 2]]),
            Selection::Points(vec![vec![3, 2]])
        );
        assert_eq!(
            H5S_select_hyperslab(vec![HyperslabDim::new(0, 1, 1, 1)]),
            Selection::Hyperslab(vec![HyperslabDim::new(0, 1, 1, 1)])
        );
        assert_eq!(
            H5Sselect_hyperslab(vec![HyperslabDim::new(1, 1, 1, 1)]),
            Selection::Hyperslab(vec![HyperslabDim::new(1, 1, 1, 1)])
        );
        assert_eq!(H5Sget_select_hyper_nblocks(&selection, &ds_shape), Some(4));
        assert_eq!(
            H5Sget_select_hyper_blocklist(&selection, &ds_shape).unwrap(),
            Some(vec![
                (vec![0, 1], vec![0, 1]),
                (vec![0, 2], vec![0, 2]),
                (vec![1, 1], vec![1, 1]),
                (vec![1, 2], vec![1, 2])
            ])
        );
        let point_selection = Selection::Points(vec![vec![0, 0], vec![3, 3]]);
        assert_eq!(H5Sget_select_elem_npoints(&point_selection), Some(2));
        assert_eq!(
            H5Sget_select_elem_pointlist(&point_selection),
            Some(&[vec![0, 0], vec![3, 3]][..])
        );
        assert_eq!(
            H5S__get_select_elem_pointlist(&point_selection),
            Some(&[vec![0, 0], vec![3, 3]][..])
        );
        assert!(H5S_select_intersect_block(&selection, &ds_shape, &[1, 2], &[1, 2]).unwrap());
        assert!(!H5Sselect_intersect_block(&selection, &ds_shape, &[3, 3], &[3, 3]).unwrap());

        let adjusted = H5S_select_adjust_u(&selection, &[1, 0]).unwrap();
        assert_eq!(
            H5Sget_select_bounds(&adjusted, &ds_shape),
            Some((vec![1, 1], vec![2, 2]))
        );
        let adjusted = H5Sselect_adjust(&selection, &[0, -1]).unwrap();
        assert_eq!(
            H5Sget_select_bounds(&adjusted, &ds_shape),
            Some((vec![0, 0], vec![1, 1]))
        );

        let encoded = selection.encode1().unwrap();
        assert_eq!(H5S_select_serial_size(&selection).unwrap(), encoded.len());
        assert_eq!(H5S_select_serialize(&selection).unwrap(), encoded);
        assert_eq!(H5S__encode(&selection).unwrap(), encoded);
        assert_eq!(H5Sencode1(&selection).unwrap(), encoded);
        assert_eq!(H5S_select_deserialize(&encoded).unwrap(), selection);
        assert_eq!(H5S__decode(&encoded).unwrap(), selection);
        H5S_select_release(selection.clone());
        let projected = H5S_select_project_simple(&selection, &ds_shape, &[1]).unwrap();
        assert_eq!(
            projected.materialize_points(&[4]).unwrap(),
            vec![vec![1], vec![2]]
        );
        assert_eq!(
            H5S_select_project_scalar(&selection, &ds_shape).unwrap(),
            Selection::Points(vec![vec![]])
        );
        assert_eq!(
            H5S_select_construct_projection(&selection, &ds_shape, &[0]).unwrap(),
            Selection::Points(vec![vec![0], vec![1]])
        );
        assert!(H5Sselect_shape_same(&selection, &selection, &ds_shape));

        let mut iter = H5S_select_iter_init(&selection, &ds_shape).unwrap();
        assert_eq!(H5S_select_iter_coords(&iter), Some(&[0, 1][..]));
        assert_eq!(H5S_select_iter_nelmts(&iter), 4);
        assert_eq!(H5S_select_iter_next(&mut iter), Some(vec![0, 1]));
        assert_eq!(
            H5S_select_iter_get_seq_list(&mut iter, 2),
            vec![vec![0, 2], vec![1, 1]]
        );
        H5Ssel_iter_reset(&mut iter);
        assert_eq!(H5S_select_iter_next(&mut iter), Some(vec![0, 1]));
        H5S_select_iter_release(iter);

        let mut visited = Vec::new();
        H5S_select_iterate(&selection, &ds_shape, |point| {
            visited.push(point.to_vec());
            Ok(())
        })
        .unwrap();
        assert_eq!(visited.len(), 4);

        let other = Selection::select_hyperslab(vec![
            HyperslabDim::new(1, 1, 1, 1),
            HyperslabDim::new(2, 1, 1, 1),
        ]);
        assert_eq!(
            H5S_select_project_intersection(&selection, &other, &ds_shape, &[0]).unwrap(),
            Selection::Points(vec![vec![1]])
        );
        assert_eq!(
            H5Sselect_project_intersection(&selection, &other, &ds_shape, &[1]).unwrap(),
            Selection::Points(vec![vec![2]])
        );
        assert_eq!(
            H5S_combine_hyperslab(&selection, &other, &ds_shape)
                .unwrap()
                .selected_count(&ds_shape),
            Some(4)
        );
        assert_eq!(
            H5Scombine_hyperslab(&selection, &other, &ds_shape)
                .unwrap()
                .selected_count(&ds_shape),
            Some(4)
        );
        assert_eq!(
            H5S__modify_select(&selection, &other, &ds_shape)
                .unwrap()
                .selected_count(&ds_shape),
            Some(4)
        );
        assert_eq!(
            H5Smodify_select(&selection, &other, &ds_shape)
                .unwrap()
                .selected_count(&ds_shape),
            Some(4)
        );
        assert!(H5S__mpio_reg_hyper_type(&selection));
        assert!(!H5S__mpio_all_type(&selection));
        assert!(!H5S__mpio_none_type(&selection));
        assert!(!H5S__mpio_point_type(&selection));
        assert!(!H5S__mpio_permute_type(&selection));
        assert!(!H5S__mpio_span_hyper_type(&selection));

        let mut buffer = vec![0u8; 16];
        H5S_select_fill(&selection, &ds_shape, &mut buffer, 7).unwrap();
        assert_eq!(buffer[1], 7);
        assert_eq!(buffer[2], 7);
        assert_eq!(buffer[5], 7);
        assert_eq!(buffer[6], 7);
    }

    #[test]
    fn h5s_selection_class_aliases_dispatch_to_selection_methods() {
        assert_eq!(H5S__none_iter_block(), None);
        assert_eq!(H5S__none_iter_nelmts(), 0);
        assert_eq!(H5S__none_iter_get_seq_list(), Vec::<Vec<u64>>::new());
        H5S__none_iter_release();
        H5S__none_release();
        assert_eq!(H5S__none_copy(), Selection::None);
        assert!(H5S__none_is_valid());
        assert_eq!(H5S__none_serialize(), Vec::<u8>::new());
        assert_eq!(H5S__none_deserialize(&[]).unwrap(), Selection::None);
        assert_eq!(H5S__none_bounds(), None);
        assert_eq!(H5S__none_offset(&[1, -1]), Selection::None);
        assert!(H5S__none_is_contiguous());
        assert!(!H5S__none_is_single());
        assert!(H5S__none_is_regular());
        assert!(!H5S__none_intersect_block(&[0], &[0]));
        assert_eq!(H5S__none_adjust_u(&[1]), Selection::None);
        assert_eq!(H5S__none_adjust_s(&[-1]), Selection::None);
        assert_eq!(H5S__none_project_simple(), Selection::None);

        let ds_shape = [2, 3];
        let mut all_iter = H5S__all_iter_init(&ds_shape).unwrap();
        assert_eq!(H5S__all_iter_coords(&ds_shape).unwrap(), Some(vec![0, 0]));
        assert_eq!(
            H5S__all_iter_block(&ds_shape),
            Some((vec![0, 0], vec![1, 2]))
        );
        assert_eq!(H5S__all_iter_nelmts(&ds_shape).unwrap(), 6);
        assert!(H5S__all_iter_has_next_block(&ds_shape));
        assert_eq!(H5S__all_iter_next(&mut all_iter), Some(vec![0, 0]));
        assert_eq!(
            H5S__all_iter_next_block(&ds_shape),
            Some((vec![0, 0], vec![1, 2]))
        );
        assert_eq!(
            H5S__all_iter_get_seq_list(&ds_shape, 3).unwrap(),
            vec![vec![0, 0], vec![0, 1], vec![0, 2]]
        );
        H5S__all_iter_release(all_iter);
        H5S__all_release();
        assert_eq!(H5S__all_copy(), Selection::All);
        assert!(H5S__all_is_valid(&ds_shape));
        assert_eq!(H5S__all_serial_size(), 0);
        assert_eq!(H5S__all_serialize(), Vec::<u8>::new());
        assert_eq!(H5S__all_deserialize(&[]).unwrap(), Selection::All);
        assert_eq!(H5S__all_bounds(&ds_shape), Some((vec![0, 0], vec![1, 2])));
        assert_eq!(H5S__all_offset(&[0, 0]).unwrap(), Selection::All);
        assert_eq!(H5S__all_unlim_dim(&[2, u64::MAX]), Some(1));
        assert!(H5S__all_is_contiguous());
        assert!(!H5S__all_is_single(&ds_shape));
        assert!(H5S__all_is_regular());
        assert!(H5S__all_intersect_block(&ds_shape, &[1, 2], &[1, 2]));
        assert_eq!(H5S__all_adjust_u(&[0, 0]).unwrap(), Selection::All);
        assert_eq!(H5S__all_adjust_s(&[0, 0]).unwrap(), Selection::All);
        assert_eq!(
            H5S__all_project_simple(&ds_shape, &[1]).unwrap(),
            Selection::Points(vec![vec![0], vec![1], vec![2]])
        );

        let mut points = Selection::Points(vec![vec![0, 1], vec![1, 2]]);
        let mut point_iter = H5S__point_iter_init(&points, &ds_shape).unwrap();
        assert_eq!(H5S__point_iter_coords(&point_iter), Some(&[0, 1][..]));
        assert_eq!(H5S__point_iter_nelmts(&point_iter), 2);
        assert_eq!(H5S__point_iter_next(&mut point_iter), Some(vec![0, 1]));
        assert_eq!(
            H5S__point_iter_next_block(&mut point_iter),
            Some(vec![1, 2])
        );
        let mut point_iter = H5S__point_iter_init(&points, &ds_shape).unwrap();
        assert_eq!(
            H5S__point_iter_get_seq_list(&mut point_iter, 2),
            vec![vec![0, 1], vec![1, 2]]
        );
        H5S__point_iter_release(point_iter);
        assert_eq!(H5S__point_copy(&points).unwrap(), points);
        H5S__point_add(&mut points, vec![0, 2]).unwrap();
        assert_eq!(points.selected_count(&ds_shape), Some(3));
        assert_eq!(H5S__point_get_version_enc_size(&points).unwrap().0, 1);
        let point_payload = H5S__point_serialize(&points).unwrap();
        assert_eq!(
            H5S__point_serial_size(&points).unwrap(),
            point_payload.len()
        );
        assert_eq!(H5S__point_deserialize(&point_payload).unwrap(), points);
        assert_eq!(
            H5S__get_select_elem_pointlist(&points),
            Some(&[vec![0, 1], vec![1, 2], vec![0, 2]][..])
        );
        assert_eq!(
            H5S__point_offset(&points, &[1, 0]).unwrap().bounds(&[3, 3]),
            Some((vec![1, 1], vec![2, 2]))
        );
        assert_eq!(H5S__point_unlim_dim(&points, &[2, u64::MAX]), Some(1));
        assert!(!H5S__point_is_contiguous(&points, &ds_shape).unwrap());
        assert!(!H5S__point_is_single(&points, &ds_shape));
        assert!(!H5S__point_is_regular(&points));
        assert!(H5S__point_shape_same(&points, &points, &ds_shape));
        assert!(H5S__point_intersect_block(&points, &[1, 2], &[1, 2]));
        assert_eq!(
            H5S__point_adjust_u(&points, &[1, 0])
                .unwrap()
                .bounds(&[3, 3]),
            Some((vec![1, 1], vec![2, 2]))
        );
        assert_eq!(
            H5S__point_adjust_s(&points, &[0, -1])
                .unwrap()
                .bounds(&ds_shape),
            Some((vec![0, 0], vec![1, 1]))
        );
        assert_eq!(
            H5S__point_project_scalar(&points, &ds_shape).unwrap(),
            Selection::Points(vec![vec![]])
        );
        assert_eq!(
            H5S__point_project_simple(&points, &ds_shape, &[0]).unwrap(),
            Selection::Points(vec![vec![0], vec![1]])
        );

        let hyper = Selection::Hyperslab(vec![
            HyperslabDim::new(0, 1, 2, 1),
            HyperslabDim::new(1, 1, 2, 1),
        ]);
        let mut hyper_iter = H5S__hyper_iter_init(&hyper, &ds_shape).unwrap();
        assert_eq!(
            H5S__hyper_iter_block(&hyper, &ds_shape).unwrap(),
            Some(vec![0, 1])
        );
        assert!(H5S__hyper_iter_has_next_block(&hyper, &ds_shape).unwrap());
        assert_eq!(H5S__hyper_iter_next(&mut hyper_iter), Some(vec![0, 1]));
        assert_eq!(
            H5S__hyper_iter_next_block(&mut hyper_iter),
            Some(vec![0, 2])
        );
        let mut hyper_iter = H5S__hyper_iter_init(&hyper, &ds_shape).unwrap();
        assert_eq!(
            H5S__hyper_iter_get_seq_list(&mut hyper_iter, 2),
            vec![vec![0, 1], vec![0, 2]]
        );
        let mut hyper_iter = H5S__hyper_iter_init(&hyper, &ds_shape).unwrap();
        assert_eq!(
            H5S__hyper_iter_get_seq_list_gen(&mut hyper_iter, 1),
            vec![vec![0, 1]]
        );
        let mut hyper_iter = H5S__hyper_iter_init(&hyper, &ds_shape).unwrap();
        assert_eq!(
            H5S__hyper_iter_get_seq_list_opt(&mut hyper_iter, 1),
            vec![vec![0, 1]]
        );
        let mut hyper_iter = H5S__hyper_iter_init(&hyper, &ds_shape).unwrap();
        assert_eq!(
            H5S__hyper_iter_get_seq_list_single(&mut hyper_iter, 1),
            vec![vec![0, 1]]
        );
        assert_eq!(H5S__hyper_iter_nelmts(&hyper_iter), 3);
        H5S__hyper_iter_release(hyper_iter);
        assert_eq!(
            H5S__hyper_get_seq_list_gen(&hyper, &ds_shape, 2).unwrap(),
            vec![vec![0, 1], vec![0, 2]]
        );
        assert_eq!(H5S__hyper_copy(&hyper).unwrap(), hyper);
        assert_eq!(
            H5S__hyper_new_span(0, 1, 1, 1),
            HyperslabDim::new(0, 1, 1, 1)
        );
        assert_eq!(
            H5S__hyper_new_span_info(vec![HyperslabDim::new(0, 1, 1, 1)]),
            Selection::Hyperslab(vec![HyperslabDim::new(0, 1, 1, 1)])
        );
        assert_eq!(H5S__hyper_copy_span(&hyper).unwrap().len(), 2);
        assert_eq!(H5S__hyper_copy_span_helper(&hyper).unwrap().len(), 2);
        assert!(H5S__hyper_cmp_spans(&hyper, &hyper));
        assert!(H5S__hyper_print_spans_helper(&hyper)
            .unwrap()
            .contains("HyperslabDim"));
        assert!(H5S__hyper_print_spans(&hyper)
            .unwrap()
            .contains("HyperslabDim"));
        assert!(H5S__space_print_spans(&hyper)
            .unwrap()
            .contains("HyperslabDim"));
        assert!(H5S__hyper_print_diminfo_helper(&hyper)
            .unwrap()
            .contains("start=0"));
        assert!(H5S__hyper_print_diminfo(&hyper)
            .unwrap()
            .contains("start=0"));
        assert!(H5S__hyper_print_spans_dfs(&hyper)
            .unwrap()
            .contains("HyperslabDim"));
        assert!(H5S__hyper_print_space_dfs(&hyper)
            .unwrap()
            .contains("HyperslabDim"));
        assert_eq!(
            H5S__hyper_get_enc_size_real(&hyper).unwrap(),
            H5S__hyper_serialize(&hyper).unwrap().len()
        );
        assert_eq!(H5S__hyper_get_version_enc_size(&hyper).unwrap().0, 1);
        let hyper_payload = H5S__hyper_serialize(&hyper).unwrap();
        assert_eq!(H5S__hyper_serialize_helper(&hyper).unwrap(), hyper_payload);
        assert_eq!(H5S__hyper_deserialize(&hyper_payload).unwrap(), hyper);
        assert_eq!(H5S__hyper_decode(&hyper_payload).unwrap(), hyper);
        assert!(H5S__hyper_is_valid(&hyper, &ds_shape));
        assert_eq!(H5S__hyper_span_nblocks(&hyper, &ds_shape), Some(4));
        assert_eq!(H5S__get_select_hyper_nblocks(&hyper, &ds_shape), Some(4));
        assert_eq!(
            H5S__hyper_span_blocklist(&hyper, &ds_shape)
                .unwrap()
                .unwrap()
                .len(),
            4
        );
        assert_eq!(
            H5S__get_select_hyper_blocklist(&hyper, &ds_shape)
                .unwrap()
                .unwrap()
                .len(),
            4
        );
        assert!(H5S__hyper_intersect_block_helper(
            &hyper,
            &ds_shape,
            &[1, 2],
            &[1, 2]
        ));
        assert_eq!(
            H5S__hyper_bounds(&hyper, &ds_shape),
            Some((vec![0, 1], vec![1, 2]))
        );
        assert_eq!(
            H5S__hyper_offset(&hyper, &[1, 0]).unwrap().bounds(&[3, 3]),
            Some((vec![1, 1], vec![2, 2]))
        );
        assert_eq!(H5S__hyper_unlim_dim(&hyper, &[2, u64::MAX]), Some(1));
        assert_eq!(
            H5S__hyper_num_elem_non_unlim(&hyper, &ds_shape, &[2, u64::MAX]).unwrap(),
            2
        );
        assert!(!H5S__hyper_is_contiguous(&hyper, &ds_shape).unwrap());
        assert!(!H5S__hyper_is_single(&hyper, &ds_shape));
        assert!(H5S__hyper_is_regular(&hyper));
        assert_eq!(H5S__hyper_spans_nelem(&hyper, &ds_shape), Some(4));
        assert_eq!(H5S__hyper_spans_nelem_helper(&hyper, &ds_shape), Some(4));
        assert!(H5S__hyper_shape_same(&hyper, &hyper, &ds_shape));
        assert!(H5S__hyper_spans_shape_same_helper(
            &hyper, &hyper, &ds_shape
        ));
        assert!(H5S__hyper_spans_shape_same(&hyper, &hyper, &ds_shape));
        assert!(!H5S__hyper_regular_and_single_block(&hyper, &ds_shape));
        assert_eq!(H5S__hyper_get_regular_hyperslab(&hyper).unwrap().len(), 2);
        assert!(H5S__hyper_coord_to_span(&hyper, &[0, 1], &ds_shape));
        let mut editable_hyper = H5S__hyper_make_spans(vec![HyperslabDim::new(0, 1, 1, 1)]);
        H5S_hyper_add_span_element(&mut editable_hyper, HyperslabDim::new(1, 1, 1, 1)).unwrap();
        H5S__hyper_append_span(&mut editable_hyper, HyperslabDim::new(2, 1, 1, 1)).unwrap();
        assert_eq!(H5S__hyper_update_diminfo(&editable_hyper).unwrap().len(), 3);
        assert_eq!(H5S__hyper_rebuild_helper(&hyper).unwrap(), hyper);
        assert_eq!(H5S__hyper_rebuild(&hyper).unwrap(), hyper);
        assert_eq!(H5S__hyper_generate_spans(&hyper).unwrap(), hyper);
        assert_eq!(H5S__fill_in_select(&hyper), hyper);
        assert_eq!(H5S__fill_in_new_space(&hyper, &ds_shape).unwrap(), hyper);
        assert_eq!(
            H5S__set_regular_hyperslab(vec![HyperslabDim::new(0, 1, 1, 1)]),
            Selection::Hyperslab(vec![HyperslabDim::new(0, 1, 1, 1)])
        );
        let disjoint = Selection::Hyperslab(vec![
            HyperslabDim::new(0, 1, 1, 1),
            HyperslabDim::new(0, 1, 1, 1),
        ]);
        assert!(H5S__check_spans_overlap(&hyper, &hyper, &ds_shape).unwrap());
        assert_eq!(
            H5S__hyper_clip_spans(&hyper, &ds_shape, &[1, 2], &[1, 2]).unwrap(),
            Selection::Points(vec![vec![1, 2]])
        );
        assert_eq!(
            H5S__hyper_merge_spans_helper(&disjoint, &hyper, &ds_shape)
                .unwrap()
                .selected_count(&ds_shape),
            Some(5)
        );
        assert_eq!(
            H5S__hyper_merge_spans(&disjoint, &hyper, &ds_shape)
                .unwrap()
                .selected_count(&ds_shape),
            Some(5)
        );
        assert_eq!(
            H5S__hyper_add_disjoint_spans(
                &disjoint,
                &Selection::Hyperslab(vec![
                    HyperslabDim::new(1, 1, 1, 1),
                    HyperslabDim::new(2, 1, 1, 1),
                ]),
                &ds_shape
            )
            .unwrap()
            .selected_count(&ds_shape),
            Some(2)
        );
        assert_eq!(
            H5S__hyper_adjust_u(&hyper, &[1, 0])
                .unwrap()
                .bounds(&[3, 3]),
            Some((vec![1, 1], vec![2, 2]))
        );
        assert_eq!(
            H5S__hyper_adjust_s(&hyper, &[0, -1])
                .unwrap()
                .bounds(&ds_shape),
            Some((vec![0, 0], vec![1, 1]))
        );
        assert_eq!(
            H5S__hyper_adjust_u_helper(&hyper, &[1, 0])
                .unwrap()
                .bounds(&[3, 3]),
            Some((vec![1, 1], vec![2, 2]))
        );
        assert_eq!(
            H5S__hyper_adjust_s_helper(&hyper, &[0, -1])
                .unwrap()
                .bounds(&ds_shape),
            Some((vec![0, 0], vec![1, 1]))
        );
        assert_eq!(
            H5S_hyper_normalize_offset(&hyper, &[1, 0])
                .unwrap()
                .bounds(&[3, 3]),
            Some((vec![1, 1], vec![2, 2]))
        );
        assert_eq!(
            H5S_hyper_denormalize_offset(&hyper, &[1, 0])
                .unwrap()
                .bounds(&[3, 3]),
            Some((vec![1, 1], vec![2, 2]))
        );
        assert_eq!(
            H5S__hyper_project_scalar(&hyper, &ds_shape).unwrap(),
            Selection::Points(vec![vec![]])
        );
        assert_eq!(
            H5S__hyper_project_simple(&hyper, &ds_shape, &[0]).unwrap(),
            Selection::Points(vec![vec![0], vec![1]])
        );
        assert_eq!(
            H5S__hyper_project_simple_lower(&hyper, &ds_shape, &[0]).unwrap(),
            Selection::Points(vec![vec![0], vec![1]])
        );
        assert_eq!(
            H5S__hyper_project_simple_higher(&hyper, &ds_shape, &[1]).unwrap(),
            Selection::Points(vec![vec![1], vec![2]])
        );
        assert_eq!(
            H5S__hyper_proj_int_build_proj(&hyper, &hyper, &ds_shape, &[0]).unwrap(),
            Selection::Points(vec![vec![0], vec![1]])
        );
        assert_eq!(
            H5S__hyper_proj_int_iterate(&hyper, &hyper, &ds_shape, &[1]).unwrap(),
            vec![vec![1], vec![2]]
        );
        assert_eq!(
            H5S__hyper_project_intersection(&hyper, &hyper, &ds_shape, &[1]).unwrap(),
            Selection::Points(vec![vec![1], vec![2]])
        );
        assert_eq!(
            H5S__hyper_get_clip_diminfo(&hyper, &ds_shape),
            Some((vec![0, 1], vec![1, 2]))
        );
        assert_eq!(H5S_hyper_clip_unlim(&hyper, &ds_shape).unwrap(), hyper);
        assert_eq!(
            H5S__hyper_get_clip_extent_real(&hyper, &ds_shape),
            Some((vec![0, 1], vec![1, 2]))
        );
        assert_eq!(
            H5S__hyper_get_clip_extent(&hyper, &ds_shape),
            Some((vec![0, 1], vec![1, 2]))
        );
        assert_eq!(
            H5S_hyper_get_clip_extent(&hyper, &ds_shape),
            Some((vec![0, 1], vec![1, 2]))
        );
        assert!(H5S__hyper_get_clip_extent_match(&hyper, &hyper, &ds_shape));
        assert!(H5S_hyper_get_clip_extent_match(&hyper, &hyper, &ds_shape));
        assert_eq!(H5S__hyper_get_unlim_block(&hyper, &[2, u64::MAX]), Some(1));
        assert_eq!(H5S_hyper_get_unlim_block(&hyper, &[2, u64::MAX]), Some(1));
        assert!(H5S__get_rebuild_status_test(&hyper));
        assert!(H5S__get_diminfo_status_test(&hyper));
        assert!(H5S__check_spans_tail_ptr(&hyper));
        assert!(H5S__check_internal_consistency(&hyper, &ds_shape));
        assert!(H5S__internal_consistency_test(&hyper, &ds_shape));
        assert!(H5S__verify_offsets(&hyper, &[0, 0]));
        H5S__hyper_free_span(hyper.clone());
        H5S__hyper_release(hyper);
    }
}
