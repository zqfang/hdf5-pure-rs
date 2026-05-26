use std::{
    fmt,
    ops::{Range, RangeFrom, RangeFull, RangeTo},
};

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

/// hdf5-metno compatibility raw-selection alias backed by this crate's selection type.
pub type RawSelection = Selection;

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
    kind: SelectionPointIterKind,
    index: usize,
    len: usize,
    current: Vec<u64>,
    yielded: Vec<u64>,
}

#[derive(Debug, Clone)]
enum SelectionPointIterKind {
    Points(Vec<Vec<u64>>),
    All {
        ds_shape: Vec<u64>,
    },
    Slice {
        slices: Vec<SliceInfo>,
        out_shape: Vec<u64>,
    },
    Hyperslab {
        dims: Vec<HyperslabDim>,
        out_shape: Vec<u64>,
    },
}

impl Iterator for SelectionPointIter {
    type Item = Vec<u64>;

    fn next(&mut self) -> Option<Self::Item> {
        self.select_iter_next_ref().map(<[u64]>::to_vec)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len.saturating_sub(self.index);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for SelectionPointIter {}

impl SelectionPointIter {
    fn new(selection: &Selection, ds_shape: &[u64]) -> Result<Self> {
        let len = selection.selected_point_len(ds_shape)?;
        let kind = match selection {
            Selection::None => SelectionPointIterKind::Points(Vec::new()),
            Selection::Points(points) => SelectionPointIterKind::Points(points.clone()),
            Selection::All => SelectionPointIterKind::All {
                ds_shape: ds_shape.to_vec(),
            },
            Selection::Slice(slices) => SelectionPointIterKind::Slice {
                slices: slices.clone(),
                out_shape: slices.iter().map(SliceInfo::count).collect(),
            },
            Selection::Hyperslab(dims) => SelectionPointIterKind::Hyperslab {
                dims: dims.clone(),
                out_shape: dims.iter().map(HyperslabDim::output_count).collect(),
            },
        };
        let mut iter = Self {
            kind,
            index: 0,
            len,
            current: Vec::new(),
            yielded: Vec::new(),
        };
        iter.refresh_current()?;
        Ok(iter)
    }

    fn refresh_current(&mut self) -> Result<()> {
        if self.index >= self.len {
            self.current.clear();
            return Ok(());
        }
        Self::coordinate_at(&self.kind, self.index, &mut self.current)
    }

    fn coordinate_at(
        kind: &SelectionPointIterKind,
        index: usize,
        out: &mut Vec<u64>,
    ) -> Result<()> {
        match kind {
            SelectionPointIterKind::Points(points) => {
                out.clear();
                if let Some(point) = points.get(index) {
                    out.extend_from_slice(point);
                }
                Ok(())
            }
            SelectionPointIterKind::All { ds_shape } => {
                row_major_coord_from_index(index, ds_shape, out)
            }
            SelectionPointIterKind::Slice { slices, out_shape } => {
                row_major_coord_from_index(index, out_shape, out)?;
                for (coord, slice) in out.iter_mut().zip(slices) {
                    *coord = slice
                        .start
                        .checked_add(coord.checked_mul(slice.step).ok_or_else(|| {
                            Error::InvalidFormat("selection coordinate overflow".into())
                        })?)
                        .ok_or_else(|| {
                            Error::InvalidFormat("selection coordinate overflow".into())
                        })?;
                }
                Ok(())
            }
            SelectionPointIterKind::Hyperslab { dims, out_shape } => {
                row_major_coord_from_index(index, out_shape, out)?;
                for (coord, dim) in out.iter_mut().zip(dims) {
                    let selected_block = *coord / dim.block;
                    let selected_offset = *coord % dim.block;
                    *coord = dim
                        .start
                        .checked_add(selected_block.checked_mul(dim.stride).ok_or_else(|| {
                            Error::InvalidFormat("hyperslab coordinate overflow".into())
                        })?)
                        .and_then(|coord| coord.checked_add(selected_offset))
                        .ok_or_else(|| {
                            Error::InvalidFormat("hyperslab coordinate overflow".into())
                        })?;
                }
                Ok(())
            }
        }
    }

    /// Return the current selected coordinate without advancing the iterator.
    pub fn select_iter_current(&self) -> Option<&[u64]> {
        match &self.kind {
            SelectionPointIterKind::Points(points) => points.get(self.index).map(Vec::as_slice),
            _ if self.index < self.len => Some(self.current.as_slice()),
            _ => None,
        }
    }

    /// Return the number of elements remaining in this selection iterator.
    pub fn select_iter_nelmts(&self) -> usize {
        self.len()
    }

    /// Return the next selected coordinate by borrowing iterator storage.
    pub fn select_iter_next_ref(&mut self) -> Option<&[u64]> {
        if self.index >= self.len {
            return None;
        }
        if Self::coordinate_at(&self.kind, self.index, &mut self.yielded).is_err() {
            return None;
        }
        self.index += 1;
        if self.refresh_current().is_err() {
            return None;
        }
        Some(self.yielded.as_slice())
    }

    /// Copy the next selected coordinate into `out`.
    ///
    /// Returns `Ok(false)` when the iterator is exhausted.
    pub fn select_iter_next_into(&mut self, out: &mut [u64]) -> Result<bool> {
        let Some(point) = self.select_iter_next_ref() else {
            return Ok(false);
        };
        if out.len() < point.len() {
            return Err(Error::InvalidFormat(
                "selection coordinate buffer is too small".into(),
            ));
        }
        out[..point.len()].copy_from_slice(point);
        Ok(true)
    }

    /// Copy up to `max_points` remaining selected coordinates into flat storage.
    ///
    /// Coordinates are written contiguously as `point0_dim0, point0_dim1, ...`.
    /// The return value is the number of complete points copied.
    pub fn select_iter_get_seq_list_into(
        &mut self,
        max_points: usize,
        out: &mut [u64],
    ) -> Result<usize> {
        let Some(rank) = self.select_iter_current().map(<[u64]>::len) else {
            return Ok(0);
        };
        let capacity = if rank == 0 {
            max_points
        } else {
            out.len() / rank
        };
        let count = max_points
            .min(capacity)
            .min(self.len.saturating_sub(self.index));
        for dst_idx in 0..count {
            let start = dst_idx
                .checked_mul(rank)
                .ok_or_else(|| Error::InvalidFormat("selection buffer offset overflow".into()))?;
            if !self.select_iter_next_into(&mut out[start..start + rank])? {
                return Ok(dst_idx);
            }
        }
        Ok(count)
    }

    /// Reset this selection iterator to the first selected coordinate.
    pub fn select_iter_reset(&mut self) {
        self.index = 0;
        let _ = self.refresh_current();
    }

    /// Explicit release hook for parity with HDF5's selection iterator API.
    pub fn select_iter_release(self) {}

    /// Hyperslab-specific iterator element count alias.
    pub fn hyper_iter_nelmts(&self) -> usize {
        self.len()
    }

    /// Hyperslab-specific borrowed next-coordinate alias.
    pub fn hyper_iter_next_ref(&mut self) -> Option<&[u64]> {
        self.select_iter_next_ref()
    }

    /// Hyperslab-specific copy-into next-coordinate alias.
    pub fn hyper_iter_next_into(&mut self, out: &mut [u64]) -> Result<bool> {
        self.select_iter_next_into(out)
    }

    /// Hyperslab-specific borrowed next-block alias.
    pub fn hyper_iter_next_block_ref(&mut self) -> Option<&[u64]> {
        self.select_iter_next_ref()
    }

    /// Hyperslab-specific copy-into next-block alias.
    pub fn hyper_iter_next_block_into(&mut self, out: &mut [u64]) -> Result<bool> {
        self.select_iter_next_into(out)
    }

    /// Hyperslab-specific copy-into sequence-list alias.
    pub fn hyper_iter_get_seq_list_into(
        &mut self,
        max_points: usize,
        out: &mut [u64],
    ) -> Result<usize> {
        self.select_iter_get_seq_list_into(max_points, out)
    }

    /// Hyperslab-specific iterator release alias.
    pub fn hyper_iter_release(self) {}

    /// Point-specific iterator coordinate alias.
    pub fn point_iter_coords(&self) -> Option<&[u64]> {
        self.select_iter_current()
    }

    /// Point-specific iterator element count alias.
    pub fn point_iter_nelmts(&self) -> usize {
        self.len()
    }

    /// Point-specific borrowed next-coordinate alias.
    pub fn point_iter_next_ref(&mut self) -> Option<&[u64]> {
        self.select_iter_next_ref()
    }

    /// Point-specific copy-into next-coordinate alias.
    pub fn point_iter_next_into(&mut self, out: &mut [u64]) -> Result<bool> {
        self.select_iter_next_into(out)
    }

    /// Point-specific borrowed next-block alias.
    pub fn point_iter_next_block_ref(&mut self) -> Option<&[u64]> {
        self.select_iter_next_ref()
    }

    /// Point-specific copy-into next-block alias.
    pub fn point_iter_next_block_into(&mut self, out: &mut [u64]) -> Result<bool> {
        self.select_iter_next_into(out)
    }

    /// Point-specific copy-into sequence-list alias.
    pub fn point_iter_get_seq_list_into(
        &mut self,
        max_points: usize,
        out: &mut [u64],
    ) -> Result<usize> {
        self.select_iter_get_seq_list_into(max_points, out)
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
        SelectionPointIter::new(&Selection::All, ds_shape)
    }

    /// Return the first block for an all-selection iterator.
    pub fn all_iter_block(ds_shape: &[u64]) -> Option<(Vec<u64>, Vec<u64>)> {
        Selection::All.bounds_owned(ds_shape)
    }

    /// Return the all-selection element count.
    pub fn all_iter_nelmts(ds_shape: &[u64]) -> Result<u64> {
        total_elements(ds_shape)
    }

    /// Return whether an all-selection has another block.
    pub fn all_iter_has_next_block(ds_shape: &[u64]) -> bool {
        ds_shape.is_empty() || !ds_shape.contains(&0)
    }

    /// Return the next all-selection block.
    pub fn all_iter_next_block(ds_shape: &[u64]) -> Option<(Vec<u64>, Vec<u64>)> {
        Selection::All.bounds_owned(ds_shape)
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
        Selection::All.bounds_owned(ds_shape)
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

    /// hdf5-metno compatibility layer: construct a selection from this crate's raw-selection alias; do not remove.
    pub fn from_raw(selection: RawSelection) -> Result<Self> {
        Ok(selection)
    }

    /// hdf5-metno compatibility layer: return required input rank when known; do not remove.
    pub fn in_ndim(&self) -> Option<usize> {
        match self {
            Selection::All | Selection::None => None,
            Selection::Points(points) => points.first().map(Vec::len),
            Selection::Hyperslab(dims) => Some(dims.len()),
            Selection::Slice(slices) => Some(slices.len()),
        }
    }

    /// hdf5-metno compatibility layer: return output rank when known; do not remove.
    pub fn out_ndim(&self) -> Option<usize> {
        match self {
            Selection::All | Selection::None => None,
            Selection::Points(_) => Some(1),
            Selection::Hyperslab(dims) => Some(usize::from(!dims.is_empty())),
            Selection::Slice(slices) => Some(slices.len()),
        }
    }

    /// hdf5-metno compatibility layer: classify explicit point selections; do not remove.
    pub fn is_points(&self) -> bool {
        matches!(self, Selection::Points(_))
    }

    /// hdf5-metno compatibility layer: classify hyperslab-like selections; do not remove.
    pub fn is_hyperslab(&self) -> bool {
        matches!(self, Selection::Hyperslab(_) | Selection::Slice(_))
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

    /// Serialize this selection with an explicit class tag into caller storage.
    pub fn encode1_into(&self, out: &mut Vec<u8>) -> Result<()> {
        let mut encoded = Vec::new();
        match self {
            Selection::None => encoded.push(0),
            Selection::All => encoded.push(1),
            Selection::Points(_) => {
                encoded.push(2);
                self.point_serialize_into(&mut encoded)?;
            }
            Selection::Hyperslab(_) | Selection::Slice(_) => {
                encoded.push(3);
                self.hyper_serialize_into(&mut encoded)?;
            }
        }
        out.clear();
        out.extend_from_slice(&encoded);
        Ok(())
    }

    /// Explicit selection copy operation.
    pub fn select_copy(&self) -> Self {
        self.clone()
    }

    /// Compute the output shape for this selection into caller storage.
    pub fn output_shape_into(&self, ds_shape: &[u64], out: &mut Vec<u64>) {
        out.clear();
        match self {
            Selection::None => out.push(0),
            Selection::All => out.extend_from_slice(ds_shape),
            Selection::Points(points) => out.push(u64::try_from(points.len()).unwrap_or(u64::MAX)),
            Selection::Hyperslab(dims) => out.extend(dims.iter().map(HyperslabDim::output_count)),
            Selection::Slice(slices) => out.extend(slices.iter().map(|s| s.count())),
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
        self.visit_points(ds_shape, |point| {
            linear_index(point, ds_shape)?;
            Ok(())
        })
        .is_ok()
    }

    /// Copy inclusive selection bounds into caller storage.
    pub fn bounds_into(&self, ds_shape: &[u64], start: &mut Vec<u64>, end: &mut Vec<u64>) -> bool {
        start.clear();
        end.clear();
        match self {
            Selection::None => false,
            Selection::All => {
                if ds_shape.contains(&0) {
                    false
                } else if ds_shape.is_empty() {
                    true
                } else {
                    start.resize(ds_shape.len(), 0);
                    end.extend(ds_shape.iter().map(|&dim| dim - 1));
                    true
                }
            }
            Selection::Points(points) => {
                if !point_bounds_into(points, start, end) {
                    return false;
                }
                true
            }
            Selection::Hyperslab(dims) => {
                if !hyperslab_bounds_into(dims, start, end) {
                    return false;
                }
                true
            }
            Selection::Slice(slices) => {
                if !slice_bounds_into(slices, start, end) {
                    return false;
                }
                true
            }
        }
    }

    fn bounds_owned(&self, ds_shape: &[u64]) -> Option<(Vec<u64>, Vec<u64>)> {
        let mut start = Vec::new();
        let mut end = Vec::new();
        if self.bounds_into(ds_shape, &mut start, &mut end) {
            Some((start, end))
        } else {
            None
        }
    }

    /// Internal selection-bounds helper.
    pub fn select_bounds_internal(&self, ds_shape: &[u64]) -> Option<(Vec<u64>, Vec<u64>)> {
        self.bounds_owned(ds_shape)
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

    /// Visit hyperslab block start/end pairs without materializing a block list.
    ///
    /// Returns `Ok(false)` when called for a point selection. `start` and `end`
    /// are inclusive coordinates and are reused between callback calls.
    pub fn visit_hyperslab_blocks<F>(&self, ds_shape: &[u64], mut callback: F) -> Result<bool>
    where
        F: FnMut(&[u64], &[u64]) -> Result<()>,
    {
        match self {
            Selection::Points(_) => Ok(false),
            Selection::None => Ok(true),
            Selection::All => {
                let mut start = Vec::new();
                let mut end = Vec::new();
                if self.bounds_into(ds_shape, &mut start, &mut end) {
                    callback(&start, &end)?;
                }
                Ok(true)
            }
            Selection::Hyperslab(dims) => {
                visit_hyperslab_blocks(dims, &mut callback)?;
                Ok(true)
            }
            Selection::Slice(slices) => {
                visit_slice_blocks(slices, &mut callback)?;
                Ok(true)
            }
        }
    }

    /// Copy hyperslab block start/end pairs into caller-provided flat buffers.
    ///
    /// Returns `Ok(None)` when called for a point selection. Otherwise returns
    /// the number of complete blocks copied. Coordinates are written
    /// contiguously per block.
    pub fn hyperslab_blocklist_into(
        &self,
        ds_shape: &[u64],
        starts: &mut [u64],
        ends: &mut [u64],
    ) -> Result<Option<usize>> {
        let Some(rank) = self.hyperslab_block_rank(ds_shape) else {
            return Ok(None);
        };
        let capacity = if rank == 0 {
            usize::MAX
        } else {
            starts.len().min(ends.len()) / rank
        };
        let mut copied = 0usize;
        self.visit_hyperslab_blocks(ds_shape, |start, end| {
            if copied >= capacity {
                return Ok(());
            }
            let offset = copied
                .checked_mul(rank)
                .ok_or_else(|| Error::InvalidFormat("hyperslab block buffer overflow".into()))?;
            starts[offset..offset + rank].copy_from_slice(start);
            ends[offset..offset + rank].copy_from_slice(end);
            copied += 1;
            Ok(())
        })?;
        Ok(Some(copied))
    }

    fn hyperslab_block_rank(&self, ds_shape: &[u64]) -> Option<usize> {
        match self {
            Selection::Points(_) => None,
            Selection::None => Some(0),
            Selection::All => Some(ds_shape.len()),
            Selection::Hyperslab(dims) => Some(dims.len()),
            Selection::Slice(slices) => Some(slices.len()),
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

    /// Iterate over explicit element-selection points as borrowed slices.
    pub fn element_points(&self) -> Option<impl ExactSizeIterator<Item = &[u64]>> {
        match self {
            Selection::Points(points) => Some(points.iter().map(Vec::as_slice)),
            _ => None,
        }
    }

    /// Copy explicit element-selection coordinates into flat caller storage.
    ///
    /// Returns `Ok(None)` when called for a non-point selection. Otherwise
    /// returns the number of complete points copied.
    pub fn element_pointlist_into(&self, out: &mut [u64]) -> Result<Option<usize>> {
        let Selection::Points(points) = self else {
            return Ok(None);
        };
        let rank = points.first().map_or(0, Vec::len);
        let capacity = if rank == 0 {
            points.len()
        } else {
            out.len() / rank
        };
        let copied = capacity.min(points.len());
        for (idx, point) in points.iter().take(copied).enumerate() {
            if point.len() != rank {
                return Err(Error::InvalidFormat(
                    "point selection contains mixed ranks".into(),
                ));
            }
            let offset = idx
                .checked_mul(rank)
                .ok_or_else(|| Error::InvalidFormat("point buffer offset overflow".into()))?;
            out[offset..offset + rank].copy_from_slice(point);
        }
        Ok(Some(copied))
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
    pub fn point_serialize_into(&self, out: &mut Vec<u8>) -> Result<()> {
        let Selection::Points(points) = self else {
            return Err(Error::InvalidFormat(
                "selection is not a point selection".into(),
            ));
        };
        let rank = points.first().map_or(0, Vec::len);
        let rank_u64 = usize_to_u64(rank, "point selection rank")?;
        let point_count_u64 = usize_to_u64(points.len(), "point selection point count")?;
        let mut encoded = Vec::with_capacity(self.point_serial_size()?);
        push_u64(&mut encoded, rank_u64);
        push_u64(&mut encoded, point_count_u64);
        for point in points {
            if point.len() != rank {
                return Err(Error::InvalidFormat(
                    "point selection contains mixed ranks".into(),
                ));
            }
            for &coord in point {
                push_u64(&mut encoded, coord);
            }
        }
        out.extend_from_slice(&encoded);
        Ok(())
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
                self.bounded_materialized_point_count(ds_shape)?;
                let mut min = None::<u64>;
                let mut max = None::<u64>;
                self.visit_points(ds_shape, |point| {
                    let idx = linear_index(point, ds_shape)?;
                    min = Some(min.map_or(idx, |value| value.min(idx)));
                    max = Some(max.map_or(idx, |value| value.max(idx)));
                    Ok(())
                })?;
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
                let count = self.bounded_materialized_point_count(ds_shape)?;
                let mut indexes = Vec::with_capacity(count);
                self.visit_points(ds_shape, |point| {
                    indexes.push(linear_index(point, ds_shape)?);
                    Ok(())
                })?;
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
        let mut left = Vec::new();
        let mut right = Vec::new();
        self.output_shape_into(ds_shape, &mut left);
        other.output_shape_into(ds_shape, &mut right);
        left == right
    }

    /// Public selected-shape comparison alias.
    pub fn select_shape_same_api(&self, other: &Selection, ds_shape: &[u64]) -> bool {
        self.select_shape_same(other, ds_shape)
    }

    /// Combine two selections with set union, returning explicit points.
    pub fn combine_or(&self, other: &Selection, ds_shape: &[u64]) -> Result<Selection> {
        use std::collections::BTreeSet;

        let mut points = BTreeSet::new();
        self.bounded_materialized_point_count(ds_shape)?;
        other.bounded_materialized_point_count(ds_shape)?;
        self.visit_points(ds_shape, |point| {
            points.insert(point.to_vec());
            Ok(())
        })?;
        other.visit_points(ds_shape, |point| {
            points.insert(point.to_vec());
            Ok(())
        })?;
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
        let lhs = self.collect_point_set(ds_shape)?;
        other.bounded_materialized_point_count(ds_shape)?;
        let mut points = std::collections::BTreeSet::new();
        other.visit_points(ds_shape, |point| {
            if lhs.contains(point) {
                points.insert(point.to_vec());
            }
            Ok(())
        })?;
        Ok(points_to_selection(points.into_iter().collect()))
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
        let mut lhs = self.collect_point_set(ds_shape)?;
        for point in other.collect_point_set(ds_shape)? {
            if !lhs.remove(point.as_slice()) {
                lhs.insert(point);
            }
        }
        Ok(points_to_selection(lhs.into_iter().collect()))
    }

    /// Subtract `other` from this selection, returning explicit points.
    pub fn combine_and_not(&self, other: &Selection, ds_shape: &[u64]) -> Result<Selection> {
        let rhs = other.collect_point_set(ds_shape)?;
        self.bounded_materialized_point_count(ds_shape)?;
        let mut points = std::collections::BTreeSet::new();
        self.visit_points(ds_shape, |point| {
            if !rhs.contains(point) {
                points.insert(point.to_vec());
            }
            Ok(())
        })?;
        Ok(points_to_selection(points.into_iter().collect()))
    }

    fn intersects(&self, other: &Selection, ds_shape: &[u64]) -> Result<bool> {
        let lhs = self.collect_point_set(ds_shape)?;
        other.bounded_materialized_point_count(ds_shape)?;
        let mut found = false;
        other.visit_points(ds_shape, |point| {
            found |= lhs.contains(point);
            Ok(())
        })?;
        Ok(found)
    }

    fn bounded_materialized_point_count(&self, ds_shape: &[u64]) -> Result<usize> {
        let count = self.selected_point_len(ds_shape)?;
        if count > MAX_MATERIALIZED_SELECTION_POINTS {
            return Err(Error::Unsupported(format!(
                "selection materialization exceeds {MAX_MATERIALIZED_SELECTION_POINTS} points"
            )));
        }
        Ok(count)
    }

    fn selected_point_len(&self, ds_shape: &[u64]) -> Result<usize> {
        let count = self
            .selected_count(ds_shape)
            .ok_or_else(|| Error::InvalidFormat("selection point count overflow".into()))?;
        usize::try_from(count)
            .map_err(|_| Error::InvalidFormat("selection point count does not fit usize".into()))
    }

    fn collect_point_set(&self, ds_shape: &[u64]) -> Result<std::collections::BTreeSet<Vec<u64>>> {
        self.bounded_materialized_point_count(ds_shape)?;
        let mut points = std::collections::BTreeSet::new();
        self.visit_points(ds_shape, |point| {
            points.insert(point.to_vec());
            Ok(())
        })?;
        Ok(points)
    }

    /// Initialize a bounded HDF5-style selection iterator.
    pub fn select_iter_init(&self, ds_shape: &[u64]) -> Result<SelectionPointIter> {
        SelectionPointIter::new(self, ds_shape)
    }

    /// Initialize a hyperslab-specific iterator.
    pub fn hyper_iter_init(&self, ds_shape: &[u64]) -> Result<SelectionPointIter> {
        require_hyperslab_like(self)?;
        SelectionPointIter::new(self, ds_shape)
    }

    /// Return whether a hyperslab iterator has another block.
    pub fn hyper_iter_has_next_block(&self, ds_shape: &[u64]) -> Result<bool> {
        require_hyperslab_like(self)?;
        self.selected_count(ds_shape)
            .map(|count| count != 0)
            .ok_or_else(|| Error::InvalidFormat("selection point count overflow".into()))
    }

    /// Initialize a point-specific iterator.
    pub fn point_iter_init(&self, ds_shape: &[u64]) -> Result<SelectionPointIter> {
        require_point_selection(self)?;
        SelectionPointIter::new(self, ds_shape)
    }

    /// Visit each selected coordinate in row-major order.
    pub fn visit_points<F>(&self, ds_shape: &[u64], mut callback: F) -> Result<()>
    where
        F: FnMut(&[u64]) -> Result<()>,
    {
        match self {
            Selection::None => Ok(()),
            Selection::All => visit_all_points(ds_shape, &mut callback),
            Selection::Points(points) => {
                for point in points {
                    callback(point)?;
                }
                Ok(())
            }
            Selection::Slice(slices) => visit_slice_points(slices, &mut callback),
            Selection::Hyperslab(dims) => visit_hyperslab_points(dims, &mut callback),
        }
    }

    /// Copy selected coordinates into flat caller-provided storage.
    ///
    /// Coordinates are written contiguously per point. Returns the number of
    /// complete points copied.
    pub fn copy_points_into(&self, ds_shape: &[u64], out: &mut [u64]) -> Result<usize> {
        let rank = match self {
            Selection::None => return Ok(0),
            Selection::All => ds_shape.len(),
            Selection::Points(points) => points.first().map_or(ds_shape.len(), Vec::len),
            Selection::Slice(slices) => slices.len(),
            Selection::Hyperslab(dims) => dims.len(),
        };
        let capacity = if rank == 0 {
            usize::MAX
        } else {
            out.len() / rank
        };
        let mut copied = 0usize;
        self.visit_points(ds_shape, |point| {
            if point.len() != rank {
                return Err(Error::InvalidFormat(
                    "selection contains mixed coordinate ranks".into(),
                ));
            }
            if copied >= capacity {
                return Ok(());
            }
            let offset = copied
                .checked_mul(rank)
                .ok_or_else(|| Error::InvalidFormat("selection buffer offset overflow".into()))?;
            out[offset..offset + rank].copy_from_slice(point);
            copied += 1;
            Ok(())
        })?;
        Ok(copied)
    }

    /// Visit each selected coordinate in row-major order.
    pub fn select_iterate<F>(&self, ds_shape: &[u64], mut callback: F) -> Result<()>
    where
        F: FnMut(&[u64]) -> Result<()>,
    {
        self.visit_points(ds_shape, |point| callback(point))
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

        let point_count = self.bounded_materialized_point_count(ds_shape)?;
        if kept_dims.is_empty() {
            return Ok(if point_count == 0 {
                Selection::None
            } else {
                Selection::Points(vec![Vec::new()])
            });
        }

        let mut projected: BTreeSet<Vec<u64>> = BTreeSet::new();
        self.visit_points(ds_shape, |point| {
            projected.insert(kept_dims.iter().map(|&dim| point[dim]).collect());
            Ok(())
        })?;
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
        use std::collections::BTreeSet;

        for &dim in kept_dims {
            if dim >= ds_shape.len() {
                return Err(Error::InvalidFormat(format!(
                    "projected dimension {dim} is out of bounds for rank {}",
                    ds_shape.len()
                )));
            }
        }

        self.bounded_materialized_point_count(ds_shape)?;
        other.bounded_materialized_point_count(ds_shape)?;

        let rhs = other.collect_point_set(ds_shape)?;
        if rhs.is_empty() {
            return Ok(Selection::None);
        }

        if kept_dims.is_empty() {
            let mut intersects = false;
            self.visit_points(ds_shape, |point| {
                if rhs.contains(point) {
                    intersects = true;
                }
                Ok(())
            })?;
            return Ok(if intersects {
                Selection::Points(vec![Vec::new()])
            } else {
                Selection::None
            });
        }

        let mut projected: BTreeSet<Vec<u64>> = BTreeSet::new();
        self.visit_points(ds_shape, |point| {
            if rhs.contains(point) {
                projected.insert(kept_dims.iter().map(|&dim| point[dim]).collect());
            }
            Ok(())
        })?;
        Ok(points_to_selection(projected.into_iter().collect()))
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

    /// Copy hyperslab span dimensions into caller storage.
    pub fn hyper_copy_span_into(&self, out: &mut Vec<HyperslabDim>) -> Result<()> {
        out.clear();
        match self {
            Selection::Hyperslab(dims) => out.extend_from_slice(dims),
            Selection::Slice(slices) => out.extend(
                slices
                    .iter()
                    .map(|slice| HyperslabDim::new(slice.start, slice.step, slice.count(), 1)),
            ),
            _ => {
                return Err(Error::InvalidFormat(
                    "selection is not hyperslab-like".into(),
                ));
            }
        }
        Ok(())
    }

    /// Compare hyperslab span dimensions for equality.
    pub fn hyper_cmp_spans(&self, other: &Selection) -> bool {
        let mut left = Vec::new();
        let mut right = Vec::new();
        self.hyper_copy_span_into(&mut left).ok() == other.hyper_copy_span_into(&mut right).ok()
            && left == right
    }

    /// Render hyperslab span dimensions for diagnostics into a formatter.
    pub fn hyper_print_spans_fmt(&self, out: &mut impl fmt::Write) -> Result<()> {
        let mut dims = Vec::new();
        self.hyper_copy_span_into(&mut dims)?;
        write!(out, "{dims:?}")
            .map_err(|_| Error::InvalidFormat("failed to format hyperslab spans".into()))
    }

    /// Render selection spans for diagnostics into a formatter.
    pub fn space_print_spans_fmt(&self, out: &mut impl fmt::Write) -> Result<()> {
        match self {
            Selection::Hyperslab(_) | Selection::Slice(_) => self.hyper_print_spans_fmt(out),
            _ => write!(out, "{:?}", self.selection_type())
                .map_err(|_| Error::InvalidFormat("failed to format selection spans".into())),
        }
    }

    /// Render hyperslab dimension info for diagnostics into a formatter.
    pub fn hyper_print_diminfo_fmt(&self, out: &mut impl fmt::Write) -> Result<()> {
        let mut dims = Vec::new();
        self.hyper_copy_span_into(&mut dims)?;
        for (idx, dim) in dims.iter().enumerate() {
            if idx != 0 {
                out.write_char(';')
                    .map_err(|_| Error::InvalidFormat("failed to format hyperslab dims".into()))?;
            }
            write!(
                out,
                "{idx}:start={},stride={},count={},block={}",
                dim.start, dim.stride, dim.count, dim.block
            )
            .map_err(|_| Error::InvalidFormat("failed to format hyperslab dims".into()))?;
        }
        Ok(())
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

    /// Hyperslab-specific copy-into blocklist alias.
    pub fn hyper_span_blocklist_into(
        &self,
        ds_shape: &[u64],
        starts: &mut [u64],
        ends: &mut [u64],
    ) -> Result<Option<usize>> {
        require_hyperslab_like(self)?;
        self.hyperslab_blocklist_into(ds_shape, starts, ends)
    }

    /// Hyperslab block-intersection helper.
    pub fn hyper_intersect_block_helper(
        &self,
        ds_shape: &[u64],
        start: &[u64],
        end: &[u64],
    ) -> bool {
        if !is_hyperslab_like(self) {
            return false;
        }
        let mut intersects = false;
        self.visit_points(ds_shape, |point| {
            intersects |= point_is_inside_block(point, start, end);
            Ok(())
        })
        .is_ok()
            && intersects
    }

    /// Internal selected-hyperslab copy-into blocklist alias.
    pub fn get_select_hyper_blocklist_into_internal(
        &self,
        ds_shape: &[u64],
        starts: &mut [u64],
        ends: &mut [u64],
    ) -> Result<Option<usize>> {
        self.hyper_span_blocklist_into(ds_shape, starts, ends)
    }

    /// Internal selected-element pointlist alias.
    pub fn get_select_elem_pointlist_internal(&self) -> Option<&[Vec<u64>]> {
        self.element_pointlist()
    }

    /// Hyperslab-specific bounds alias.
    pub fn hyper_bounds(&self, ds_shape: &[u64]) -> Option<(Vec<u64>, Vec<u64>)> {
        if is_hyperslab_like(self) {
            self.bounds_owned(ds_shape)
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
        if !is_hyperslab_like(self) {
            return false;
        }
        let mut found = false;
        self.visit_points(ds_shape, |point| {
            found |= point == coord;
            Ok(())
        })
        .is_ok()
            && found
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
        self.bounded_materialized_point_count(ds_shape)?;
        let mut points = Vec::new();
        self.visit_points(ds_shape, |point| {
            if point_is_inside_block(point, start, end) {
                points.push(point.to_vec());
            }
            Ok(())
        })?;
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
        if self.intersects(other, ds_shape)? {
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
        self.intersects(other, ds_shape)
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
    pub fn hyper_proj_int_visit<F>(
        &self,
        other: &Selection,
        ds_shape: &[u64],
        kept_dims: &[usize],
        callback: F,
    ) -> Result<()>
    where
        F: FnMut(&[u64]) -> Result<()>,
    {
        self.hyper_proj_int_build_proj(other, ds_shape, kept_dims)?
            .visit_points(&projected_shape(ds_shape, kept_dims), callback)
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
        let mut dims = Vec::new();
        self.hyper_copy_span_into(&mut dims).is_ok()
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
        let mut dims = Vec::new();
        self.hyper_copy_span_into(&mut dims)?;
        let words = 1usize
            .checked_add(dims.len().checked_mul(4).ok_or_else(|| {
                Error::InvalidFormat("hyperslab serialization size overflow".into())
            })?)
            .ok_or_else(|| Error::InvalidFormat("hyperslab serialization size overflow".into()))?;
        words
            .checked_mul(8)
            .ok_or_else(|| Error::InvalidFormat("hyperslab serialization size overflow".into()))
    }

    /// Serialize hyperslab span dimensions into caller-provided storage.
    pub fn hyper_serialize_into(&self, out: &mut Vec<u8>) -> Result<()> {
        let mut dims = Vec::new();
        self.hyper_copy_span_into(&mut dims)?;
        push_u64(out, usize_to_u64(dims.len(), "hyperslab rank")?);
        for dim in dims {
            push_u64(out, dim.start);
            push_u64(out, dim.stride);
            push_u64(out, dim.count);
            push_u64(out, dim.block);
        }
        Ok(())
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
        let touched = match self {
            Selection::None => 0,
            Selection::All => max_dims.len(),
            Selection::Points(points) => points.first().map_or(0, Vec::len),
            Selection::Hyperslab(dims) => dims.len(),
            Selection::Slice(slices) => slices.len(),
        };
        (0..touched).find(|&dim| max_dims.get(dim).copied() == Some(u64::MAX))
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
        self.bounded_materialized_point_count(ds_shape)?;
        let mut projected: BTreeSet<Vec<u64>> = BTreeSet::new();
        self.visit_points(ds_shape, |point| {
            projected.insert(kept_dims.iter().map(|&dim| point[dim]).collect());
            Ok(())
        })?;
        u64::try_from(projected.len())
            .map_err(|_| Error::InvalidFormat("non-unlimited element count overflow".into()))
    }

    /// Apply signed per-dimension offsets to this finite selection.
    pub fn select_offset(&self, offsets: &[i64]) -> Result<Selection> {
        self.shift(offsets)
    }

    /// Apply unsigned per-dimension offsets to this finite selection.
    pub fn select_adjust_unsigned(&self, offsets: &[u64]) -> Result<Selection> {
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
                    .map(|point| shift_coords_unsigned(point, offsets))
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
                            start: shift_coord_unsigned(dim.start, offset)?,
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
                            start: shift_coord_unsigned(slice.start, offset)?,
                            end: shift_coord_unsigned(slice.end, offset)?,
                            step: slice.step,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(Selection::Slice(shifted))
            }
        }
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
        self.visit_points(ds_shape, |point| {
            let idx = usize::try_from(linear_index(point, ds_shape)?)
                .map_err(|_| Error::InvalidFormat("selection linear index exceeds usize".into()))?;
            buffer[idx] = value.clone();
            Ok(())
        })
    }

    /// Normalize into concrete SliceInfo per dimension.
    pub fn to_slices_into(&self, ds_shape: &[u64], out: &mut Vec<SliceInfo>) {
        out.clear();
        match self {
            Selection::All => out.extend(ds_shape.iter().map(|&d| SliceInfo::new(0, d))),
            Selection::Slice(slices) => out.extend_from_slice(slices),
            Selection::None | Selection::Points(_) | Selection::Hyperslab(_) => {}
        }
    }

    /// Normalize into concrete SliceInfo per dimension.
    pub fn to_slices(&self, ds_shape: &[u64]) -> Vec<SliceInfo> {
        let mut out = Vec::new();
        self.to_slices_into(ds_shape, &mut out);
        out
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
    Ok(1 + match selection {
        Selection::None | Selection::All => 0,
        Selection::Points(_) => selection.point_serial_size()?,
        Selection::Hyperslab(_) | Selection::Slice(_) => selection.hyper_get_enc_size_real()?,
    })
}

#[allow(non_snake_case)]
pub fn H5S_select_serialize_into(selection: &Selection, out: &mut Vec<u8>) -> Result<()> {
    selection.encode1_into(out)
}

#[allow(non_snake_case)]
pub fn H5S__encode_into(selection: &Selection, out: &mut Vec<u8>) -> Result<()> {
    selection.encode1_into(out)
}

#[allow(non_snake_case)]
pub fn H5Sencode1_into(selection: &Selection, out: &mut Vec<u8>) -> Result<()> {
    selection.encode1_into(out)
}

#[allow(non_snake_case)]
pub fn H5Sget_select_bounds_into(
    selection: &Selection,
    ds_shape: &[u64],
    start: &mut Vec<u64>,
    end: &mut Vec<u64>,
) -> bool {
    selection.bounds_into(ds_shape, start, end)
}

#[allow(non_snake_case)]
pub fn H5S_get_select_bounds_into(
    selection: &Selection,
    ds_shape: &[u64],
    start: &mut Vec<u64>,
    end: &mut Vec<u64>,
) -> bool {
    H5Sget_select_bounds_into(selection, ds_shape, start, end)
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
pub fn H5Sget_select_hyper_blocklist_into(
    selection: &Selection,
    ds_shape: &[u64],
    starts: &mut [u64],
    ends: &mut [u64],
) -> Result<Option<usize>> {
    selection.hyperslab_blocklist_into(ds_shape, starts, ends)
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
pub fn H5Sget_select_elem_pointlist_into(
    selection: &Selection,
    out: &mut [u64],
) -> Result<Option<usize>> {
    selection.element_pointlist_into(out)
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
    let mut intersects = false;
    selection.visit_points(ds_shape, |point| {
        intersects |= point_is_inside_block(point, start, end);
        Ok(())
    })?;
    Ok(intersects)
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
pub fn H5S_select_iter_next_ref(iter: &mut SelectionPointIter) -> Option<&[u64]> {
    iter.select_iter_next_ref()
}

#[allow(non_snake_case)]
pub fn H5S_select_iter_next_into(iter: &mut SelectionPointIter, out: &mut [u64]) -> Result<bool> {
    iter.select_iter_next_into(out)
}

#[allow(non_snake_case)]
pub fn H5S_select_iter_get_seq_list_into(
    iter: &mut SelectionPointIter,
    max_points: usize,
    out: &mut [u64],
) -> Result<usize> {
    iter.select_iter_get_seq_list_into(max_points, out)
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
pub fn H5S_select_copy_points_into(
    selection: &Selection,
    ds_shape: &[u64],
    out: &mut [u64],
) -> Result<usize> {
    selection.copy_points_into(ds_shape, out)
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
    Vec::new()
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
pub fn H5S__all_iter_coords_into(ds_shape: &[u64], out: &mut [u64]) -> Result<bool> {
    let mut iter = Selection::all_iter_init(ds_shape)?;
    iter.select_iter_next_into(out)
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
pub fn H5S__all_iter_next_ref(iter: &mut SelectionPointIter) -> Option<&[u64]> {
    iter.select_iter_next_ref()
}

#[allow(non_snake_case)]
pub fn H5S__all_iter_next_into(iter: &mut SelectionPointIter, out: &mut [u64]) -> Result<bool> {
    iter.select_iter_next_into(out)
}

#[allow(non_snake_case)]
pub fn H5S__all_iter_next_block(ds_shape: &[u64]) -> Option<(Vec<u64>, Vec<u64>)> {
    Selection::all_iter_next_block(ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__all_iter_get_seq_list_into(
    ds_shape: &[u64],
    max_points: usize,
    out: &mut [u64],
) -> Result<usize> {
    let mut iter = Selection::all_iter_init(ds_shape)?;
    iter.select_iter_get_seq_list_into(max_points, out)
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
pub fn H5S__point_iter_next_ref(iter: &mut SelectionPointIter) -> Option<&[u64]> {
    iter.point_iter_next_ref()
}

#[allow(non_snake_case)]
pub fn H5S__point_iter_next_into(iter: &mut SelectionPointIter, out: &mut [u64]) -> Result<bool> {
    iter.point_iter_next_into(out)
}

#[allow(non_snake_case)]
pub fn H5S__point_iter_next_block_ref(iter: &mut SelectionPointIter) -> Option<&[u64]> {
    iter.point_iter_next_block_ref()
}

#[allow(non_snake_case)]
pub fn H5S__point_iter_next_block_into(
    iter: &mut SelectionPointIter,
    out: &mut [u64],
) -> Result<bool> {
    iter.point_iter_next_block_into(out)
}

#[allow(non_snake_case)]
pub fn H5S__point_iter_get_seq_list_into(
    iter: &mut SelectionPointIter,
    max_points: usize,
    out: &mut [u64],
) -> Result<usize> {
    iter.point_iter_get_seq_list_into(max_points, out)
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
pub fn H5S__point_serialize_into(selection: &Selection, out: &mut Vec<u8>) -> Result<()> {
    selection.point_serialize_into(out)
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
pub fn H5S__hyper_iter_block_into(
    selection: &Selection,
    ds_shape: &[u64],
    out: &mut [u64],
) -> Result<bool> {
    let mut iter = selection.hyper_iter_init(ds_shape)?;
    iter.hyper_iter_next_into(out)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_iter_has_next_block(selection: &Selection, ds_shape: &[u64]) -> Result<bool> {
    selection.hyper_iter_has_next_block(ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_iter_next_ref(iter: &mut SelectionPointIter) -> Option<&[u64]> {
    iter.hyper_iter_next_ref()
}

#[allow(non_snake_case)]
pub fn H5S__hyper_iter_next_into(iter: &mut SelectionPointIter, out: &mut [u64]) -> Result<bool> {
    iter.hyper_iter_next_into(out)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_iter_next_block_ref(iter: &mut SelectionPointIter) -> Option<&[u64]> {
    iter.hyper_iter_next_block_ref()
}

#[allow(non_snake_case)]
pub fn H5S__hyper_iter_next_block_into(
    iter: &mut SelectionPointIter,
    out: &mut [u64],
) -> Result<bool> {
    iter.hyper_iter_next_block_into(out)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_iter_get_seq_list_into(
    iter: &mut SelectionPointIter,
    max_points: usize,
    out: &mut [u64],
) -> Result<usize> {
    iter.hyper_iter_get_seq_list_into(max_points, out)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_iter_nelmts(iter: &SelectionPointIter) -> usize {
    iter.hyper_iter_nelmts()
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
pub fn H5S__hyper_copy_span_into(selection: &Selection, out: &mut Vec<HyperslabDim>) -> Result<()> {
    selection.hyper_copy_span_into(out)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_cmp_spans(left: &Selection, right: &Selection) -> bool {
    left.hyper_cmp_spans(right)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_print_spans_fmt(selection: &Selection, out: &mut impl fmt::Write) -> Result<()> {
    selection.hyper_print_spans_fmt(out)
}

#[allow(non_snake_case)]
pub fn H5S__space_print_spans_fmt(selection: &Selection, out: &mut impl fmt::Write) -> Result<()> {
    selection.space_print_spans_fmt(out)
}

#[allow(non_snake_case)]
pub fn H5S__hyper_print_diminfo_fmt(
    selection: &Selection,
    out: &mut impl fmt::Write,
) -> Result<()> {
    selection.hyper_print_diminfo_fmt(out)
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
pub fn H5S__hyper_serialize_into(selection: &Selection, out: &mut Vec<u8>) -> Result<()> {
    selection.hyper_serialize_into(out)
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
pub fn H5S__hyper_span_blocklist_into(
    selection: &Selection,
    ds_shape: &[u64],
    starts: &mut [u64],
    ends: &mut [u64],
) -> Result<Option<usize>> {
    selection.hyper_span_blocklist_into(ds_shape, starts, ends)
}

#[allow(non_snake_case)]
pub fn H5S__get_select_hyper_nblocks(selection: &Selection, ds_shape: &[u64]) -> Option<u64> {
    selection.get_select_hyper_nblocks_internal(ds_shape)
}

#[allow(non_snake_case)]
pub fn H5S__get_select_hyper_blocklist_into(
    selection: &Selection,
    ds_shape: &[u64],
    starts: &mut [u64],
    ends: &mut [u64],
) -> Result<Option<usize>> {
    selection.get_select_hyper_blocklist_into_internal(ds_shape, starts, ends)
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
pub fn H5S__hyper_get_regular_hyperslab_into(
    selection: &Selection,
    out: &mut Vec<HyperslabDim>,
) -> Result<()> {
    selection.hyper_copy_span_into(out)
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
pub fn H5S__hyper_update_diminfo_into(
    selection: &Selection,
    out: &mut Vec<HyperslabDim>,
) -> Result<()> {
    selection.hyper_copy_span_into(out)
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
pub fn H5S__hyper_proj_int_visit<F>(
    left: &Selection,
    right: &Selection,
    ds_shape: &[u64],
    kept_dims: &[usize],
    callback: F,
) -> Result<()>
where
    F: FnMut(&[u64]) -> Result<()>,
{
    left.hyper_proj_int_visit(right, ds_shape, kept_dims, callback)
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

fn shift_coords_unsigned(point: &[u64], offsets: &[u64]) -> Result<Vec<u64>> {
    check_rank("point selection", point.len(), offsets.len())?;
    point
        .iter()
        .zip(offsets)
        .map(|(&coord, &offset)| shift_coord_unsigned(coord, offset))
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

fn shift_coord_unsigned(coord: u64, offset: u64) -> Result<u64> {
    coord
        .checked_add(offset)
        .ok_or_else(|| Error::InvalidFormat("selection coordinate overflow".into()))
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

fn row_major_coord_from_index(index: usize, shape: &[u64], out: &mut Vec<u64>) -> Result<()> {
    out.clear();
    out.resize(shape.len(), 0);
    if shape.is_empty() {
        return Ok(());
    }

    let mut remainder = usize_to_u64(index, "selection iterator index")?;
    for dim in (0..shape.len()).rev() {
        let extent = shape[dim];
        if extent == 0 {
            return Err(Error::InvalidFormat(
                "selection iterator shape contains zero extent".into(),
            ));
        }
        out[dim] = remainder % extent;
        remainder /= extent;
    }
    if remainder == 0 {
        Ok(())
    } else {
        Err(Error::InvalidFormat(
            "selection iterator index exceeds shape".into(),
        ))
    }
}

fn visit_all_points<F>(ds_shape: &[u64], callback: &mut F) -> Result<()>
where
    F: FnMut(&[u64]) -> Result<()>,
{
    if ds_shape.contains(&0) {
        return Ok(());
    }
    if ds_shape.is_empty() {
        return callback(&[]);
    }
    let mut current = vec![0u64; ds_shape.len()];
    visit_all_points_recursive(ds_shape, 0, &mut current, callback)
}

fn visit_all_points_recursive<F>(
    ds_shape: &[u64],
    dim: usize,
    current: &mut [u64],
    callback: &mut F,
) -> Result<()>
where
    F: FnMut(&[u64]) -> Result<()>,
{
    if dim == ds_shape.len() {
        return callback(current);
    }
    for coord in 0..ds_shape[dim] {
        current[dim] = coord;
        visit_all_points_recursive(ds_shape, dim + 1, current, callback)?;
    }
    Ok(())
}

fn visit_slice_points<F>(slices: &[SliceInfo], callback: &mut F) -> Result<()>
where
    F: FnMut(&[u64]) -> Result<()>,
{
    if slices.iter().any(|slice| slice.count() == 0) {
        return Ok(());
    }
    let mut current = vec![0u64; slices.len()];
    visit_slice_points_recursive(slices, 0, &mut current, callback)
}

fn visit_slice_points_recursive<F>(
    slices: &[SliceInfo],
    dim: usize,
    current: &mut [u64],
    callback: &mut F,
) -> Result<()>
where
    F: FnMut(&[u64]) -> Result<()>,
{
    if dim == slices.len() {
        return callback(current);
    }
    let slice = &slices[dim];
    let mut coord = slice.start;
    while coord < slice.end {
        current[dim] = coord;
        visit_slice_points_recursive(slices, dim + 1, current, callback)?;
        coord = coord
            .checked_add(slice.step)
            .ok_or_else(|| Error::InvalidFormat("selection coordinate overflow".into()))?;
    }
    Ok(())
}

fn visit_hyperslab_points<F>(dims: &[HyperslabDim], callback: &mut F) -> Result<()>
where
    F: FnMut(&[u64]) -> Result<()>,
{
    if dims
        .iter()
        .any(|dim| dim.count == 0 || dim.block == 0 || dim.stride == 0)
    {
        return Ok(());
    }
    let mut current = vec![0u64; dims.len()];
    visit_hyperslab_points_recursive(dims, 0, &mut current, callback)
}

fn visit_hyperslab_points_recursive<F>(
    dims: &[HyperslabDim],
    dim: usize,
    current: &mut [u64],
    callback: &mut F,
) -> Result<()>
where
    F: FnMut(&[u64]) -> Result<()>,
{
    if dim == dims.len() {
        return callback(current);
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
            visit_hyperslab_points_recursive(dims, dim + 1, current, callback)?;
        }
    }
    Ok(())
}

fn visit_slice_blocks<F>(slices: &[SliceInfo], callback: &mut F) -> Result<()>
where
    F: FnMut(&[u64], &[u64]) -> Result<()>,
{
    if slices.iter().any(|slice| slice.count() == 0) {
        return Ok(());
    }
    let mut start = vec![0u64; slices.len()];
    let mut end = vec![0u64; slices.len()];
    visit_slice_blocks_recursive(slices, 0, &mut start, &mut end, callback)
}

fn visit_slice_blocks_recursive<F>(
    slices: &[SliceInfo],
    dim: usize,
    start: &mut [u64],
    end: &mut [u64],
    callback: &mut F,
) -> Result<()>
where
    F: FnMut(&[u64], &[u64]) -> Result<()>,
{
    if dim == slices.len() {
        return callback(start, end);
    }

    let slice = &slices[dim];
    if slice.step == 1 {
        start[dim] = slice.start;
        end[dim] = slice
            .end
            .checked_sub(1)
            .ok_or_else(|| Error::InvalidFormat("slice block end underflow".into()))?;
        return visit_slice_blocks_recursive(slices, dim + 1, start, end, callback);
    }

    let mut coord = slice.start;
    while coord < slice.end {
        start[dim] = coord;
        end[dim] = coord;
        visit_slice_blocks_recursive(slices, dim + 1, start, end, callback)?;
        coord = coord
            .checked_add(slice.step)
            .ok_or_else(|| Error::InvalidFormat("slice block coordinate overflow".into()))?;
    }
    Ok(())
}

fn visit_hyperslab_blocks<F>(dims: &[HyperslabDim], callback: &mut F) -> Result<()>
where
    F: FnMut(&[u64], &[u64]) -> Result<()>,
{
    if dims
        .iter()
        .any(|dim| dim.count == 0 || dim.block == 0 || dim.stride == 0)
    {
        return Ok(());
    }
    let mut start = vec![0u64; dims.len()];
    let mut end = vec![0u64; dims.len()];
    visit_hyperslab_blocks_recursive(dims, 0, &mut start, &mut end, callback)
}

fn visit_hyperslab_blocks_recursive<F>(
    dims: &[HyperslabDim],
    dim: usize,
    start: &mut [u64],
    end: &mut [u64],
    callback: &mut F,
) -> Result<()>
where
    F: FnMut(&[u64], &[u64]) -> Result<()>,
{
    if dim == dims.len() {
        return callback(start, end);
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
        visit_hyperslab_blocks_recursive(dims, dim + 1, start, end, callback)?;
    }
    Ok(())
}

fn slice_bounds_into(slices: &[SliceInfo], start: &mut Vec<u64>, end: &mut Vec<u64>) -> bool {
    if slices.iter().any(|slice| slice.count() == 0) {
        return false;
    }
    start.reserve(slices.len());
    end.reserve(slices.len());
    for slice in slices {
        let Some(last) = slice
            .count()
            .checked_sub(1)
            .and_then(|value| value.checked_mul(slice.step))
            .and_then(|value| value.checked_add(slice.start))
        else {
            start.clear();
            end.clear();
            return false;
        };
        start.push(slice.start);
        end.push(last);
    }
    true
}

fn hyperslab_bounds_into(dims: &[HyperslabDim], start: &mut Vec<u64>, end: &mut Vec<u64>) -> bool {
    if dims
        .iter()
        .any(|dim| dim.count == 0 || dim.block == 0 || dim.stride == 0)
    {
        return false;
    }
    start.reserve(dims.len());
    end.reserve(dims.len());
    for dim in dims {
        let Some(last) = dim
            .count
            .checked_sub(1)
            .and_then(|value| value.checked_mul(dim.stride))
            .and_then(|value| value.checked_add(dim.block))
            .and_then(|value| value.checked_sub(1))
            .and_then(|value| value.checked_add(dim.start))
        else {
            start.clear();
            end.clear();
            return false;
        };
        start.push(dim.start);
        end.push(last);
    }
    true
}

fn point_bounds_into(points: &[Vec<u64>], start: &mut Vec<u64>, end: &mut Vec<u64>) -> bool {
    let Some(first) = points.first() else {
        return false;
    };
    start.extend_from_slice(first);
    end.extend_from_slice(first);
    for point in &points[1..] {
        if point.len() != start.len() {
            start.clear();
            end.clear();
            return false;
        }
        for (dim, &coord) in point.iter().enumerate() {
            start[dim] = start[dim].min(coord);
            end[dim] = end[dim].max(coord);
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn selection_bounds(selection: &Selection, ds_shape: &[u64]) -> Option<(Vec<u64>, Vec<u64>)> {
        let mut start = Vec::new();
        let mut end = Vec::new();
        selection
            .bounds_into(ds_shape, &mut start, &mut end)
            .then_some((start, end))
    }

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
        assert_eq!(iter.select_iter_next_ref(), Some(&[0][..]));
        assert_eq!(iter.select_iter_next_ref(), Some(&[1][..]));
        iter.select_iter_reset();
        assert_eq!(iter.select_iter_next_ref(), Some(&[0][..]));
    }

    #[test]
    fn selection_serialization_aliases_roundtrip() {
        let points = Selection::select_elements(vec![vec![1, 2], vec![3, 4]]);
        assert!(points.mpio_point_type());
        let mut encoded_points = Vec::new();
        points.encode1_into(&mut encoded_points).unwrap();
        assert_eq!(
            Selection::select_deserialize(&encoded_points).unwrap(),
            points
        );
        let mut point_payload = Vec::new();
        points.point_serialize_into(&mut point_payload).unwrap();
        assert_eq!(
            Selection::point_deserialize(&point_payload).unwrap(),
            points
        );
        assert_eq!(points.point_get_version_enc_size().unwrap().0, 1);
        assert!(Selection::point_deserialize(&u64::MAX.to_le_bytes()).is_err());

        let hyper = Selection::select_hyperslab(vec![
            HyperslabDim::new(0, 2, 2, 1),
            HyperslabDim::new(1, 1, 3, 1),
        ]);
        assert!(hyper.mpio_reg_hyper_type());
        let mut encoded_hyper = Vec::new();
        hyper.encode1_into(&mut encoded_hyper).unwrap();
        assert_eq!(
            Selection::select_deserialize(&encoded_hyper).unwrap(),
            hyper
        );
        let mut hyper_payload = Vec::new();
        hyper.hyper_serialize_into(&mut hyper_payload).unwrap();
        assert_eq!(Selection::hyper_deserialize(&hyper_payload).unwrap(), hyper);
        assert_eq!(hyper.hyper_get_version_enc_size().unwrap().0, 1);
        assert!(Selection::hyper_deserialize(&u64::MAX.to_le_bytes()).is_err());
        assert_eq!(hyper.hyper_spans_nelem(&[4, 4]), Some(6));
        assert!(hyper.hyper_coord_to_span(&[0, 1], &[4, 4]));
        assert!(hyper.hyper_spans_shape_same(&hyper, &[4, 4]));
        let mut diminfo = String::new();
        hyper.hyper_print_diminfo_fmt(&mut diminfo).unwrap();
        assert!(diminfo.contains("start=0"));
        let mut spans = String::new();
        hyper.space_print_spans_fmt(&mut spans).unwrap();
        assert!(spans.contains("HyperslabDim"));

        let mut editable = Selection::select_hyperslab(vec![Selection::hyper_new_span(0, 1, 1, 1)]);
        editable
            .hyper_add_span_element(HyperslabDim::new(1, 1, 1, 1))
            .unwrap();
        let mut copied = Vec::new();
        editable.hyper_copy_span_into(&mut copied).unwrap();
        assert_eq!(copied.len(), 2);
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
        let mut bound_start = Vec::new();
        let mut bound_end = Vec::new();
        assert!(H5Sget_select_bounds_into(
            &selection,
            &ds_shape,
            &mut bound_start,
            &mut bound_end
        ));
        assert_eq!(bound_start, vec![0, 1]);
        assert_eq!(bound_end, vec![1, 2]);
        bound_start.clear();
        bound_end.clear();
        assert!(H5S_get_select_bounds_into(
            &selection,
            &ds_shape,
            &mut bound_start,
            &mut bound_end
        ));
        assert_eq!((bound_start, bound_end), (vec![0, 1], vec![1, 2]));
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
        let mut starts = vec![0; 8];
        let mut ends = vec![0; 8];
        assert_eq!(
            H5Sget_select_hyper_blocklist_into(&selection, &ds_shape, &mut starts, &mut ends)
                .unwrap(),
            Some(4)
        );
        assert_eq!(starts, vec![0, 1, 0, 2, 1, 1, 1, 2]);
        assert_eq!(ends, vec![0, 1, 0, 2, 1, 1, 1, 2]);
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
            selection_bounds(&adjusted, &ds_shape),
            Some((vec![1, 1], vec![2, 2]))
        );
        let adjusted = H5Sselect_adjust(&selection, &[0, -1]).unwrap();
        assert_eq!(
            selection_bounds(&adjusted, &ds_shape),
            Some((vec![0, 0], vec![1, 1]))
        );

        let mut encoded = Vec::new();
        selection.encode1_into(&mut encoded).unwrap();
        assert_eq!(H5S_select_serial_size(&selection).unwrap(), encoded.len());
        let mut encoded_via_h5s = Vec::new();
        H5S_select_serialize_into(&selection, &mut encoded_via_h5s).unwrap();
        assert_eq!(encoded_via_h5s, encoded);
        encoded_via_h5s.clear();
        H5S__encode_into(&selection, &mut encoded_via_h5s).unwrap();
        assert_eq!(encoded_via_h5s, encoded);
        encoded_via_h5s.clear();
        H5Sencode1_into(&selection, &mut encoded_via_h5s).unwrap();
        assert_eq!(encoded_via_h5s, encoded);
        assert_eq!(H5S_select_deserialize(&encoded).unwrap(), selection);
        assert_eq!(H5S__decode(&encoded).unwrap(), selection);
        H5S_select_release(selection.clone());
        let projected = H5S_select_project_simple(&selection, &ds_shape, &[1]).unwrap();
        let mut projected_points = Vec::new();
        projected
            .visit_points(&[4], |point| {
                projected_points.push(point.to_vec());
                Ok(())
            })
            .unwrap();
        assert_eq!(projected_points, vec![vec![1], vec![2]]);
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
        assert_eq!(H5S_select_iter_next_ref(&mut iter), Some(&[0, 1][..]));
        let mut seq = vec![0; 4];
        assert_eq!(
            H5S_select_iter_get_seq_list_into(&mut iter, 2, &mut seq).unwrap(),
            2
        );
        assert_eq!(seq, vec![0, 2, 1, 1]);
        H5Ssel_iter_reset(&mut iter);
        assert_eq!(H5S_select_iter_next_ref(&mut iter), Some(&[0, 1][..]));
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
        let mut all_coords = vec![u64::MAX; 2];
        assert!(H5S__all_iter_coords_into(&ds_shape, &mut all_coords).unwrap());
        assert_eq!(all_coords, vec![0, 0]);
        assert_eq!(
            H5S__all_iter_block(&ds_shape),
            Some((vec![0, 0], vec![1, 2]))
        );
        assert_eq!(H5S__all_iter_nelmts(&ds_shape).unwrap(), 6);
        assert!(H5S__all_iter_has_next_block(&ds_shape));
        assert_eq!(H5S__all_iter_next_ref(&mut all_iter), Some(&[0, 0][..]));
        assert_eq!(
            H5S__all_iter_next_block(&ds_shape),
            Some((vec![0, 0], vec![1, 2]))
        );
        let mut all_seq = vec![0; 6];
        assert_eq!(
            H5S__all_iter_get_seq_list_into(&ds_shape, 3, &mut all_seq).unwrap(),
            3
        );
        assert_eq!(all_seq, vec![0, 0, 0, 1, 0, 2]);
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
        assert_eq!(H5S__point_iter_next_ref(&mut point_iter), Some(&[0, 1][..]));
        assert_eq!(
            H5S__point_iter_next_block_ref(&mut point_iter),
            Some(&[1, 2][..])
        );
        let mut point_iter = H5S__point_iter_init(&points, &ds_shape).unwrap();
        let mut point_seq = vec![0; 4];
        assert_eq!(
            H5S__point_iter_get_seq_list_into(&mut point_iter, 2, &mut point_seq).unwrap(),
            2
        );
        assert_eq!(point_seq, vec![0, 1, 1, 2]);
        H5S__point_iter_release(point_iter);
        assert_eq!(H5S__point_copy(&points).unwrap(), points);
        H5S__point_add(&mut points, vec![0, 2]).unwrap();
        assert_eq!(points.selected_count(&ds_shape), Some(3));
        assert_eq!(H5S__point_get_version_enc_size(&points).unwrap().0, 1);
        let mut point_payload = Vec::new();
        H5S__point_serialize_into(&points, &mut point_payload).unwrap();
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
            selection_bounds(&H5S__point_offset(&points, &[1, 0]).unwrap(), &[3, 3]),
            Some((vec![1, 1], vec![2, 2]))
        );
        assert_eq!(H5S__point_unlim_dim(&points, &[2, u64::MAX]), Some(1));
        assert!(!H5S__point_is_contiguous(&points, &ds_shape).unwrap());
        assert!(!H5S__point_is_single(&points, &ds_shape));
        assert!(!H5S__point_is_regular(&points));
        assert!(H5S__point_shape_same(&points, &points, &ds_shape));
        assert!(H5S__point_intersect_block(&points, &[1, 2], &[1, 2]));
        assert_eq!(
            selection_bounds(&H5S__point_adjust_u(&points, &[1, 0]).unwrap(), &[3, 3]),
            Some((vec![1, 1], vec![2, 2]))
        );
        assert_eq!(
            selection_bounds(&H5S__point_adjust_s(&points, &[0, -1]).unwrap(), &ds_shape),
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
        let mut hyper_block = vec![u64::MAX; 2];
        assert!(H5S__hyper_iter_block_into(&hyper, &ds_shape, &mut hyper_block).unwrap());
        assert_eq!(hyper_block, vec![0, 1]);
        assert!(H5S__hyper_iter_has_next_block(&hyper, &ds_shape).unwrap());
        assert_eq!(H5S__hyper_iter_next_ref(&mut hyper_iter), Some(&[0, 1][..]));
        assert_eq!(
            H5S__hyper_iter_next_block_ref(&mut hyper_iter),
            Some(&[0, 2][..])
        );
        let mut hyper_iter = H5S__hyper_iter_init(&hyper, &ds_shape).unwrap();
        let mut hyper_seq = vec![0; 4];
        assert_eq!(
            H5S__hyper_iter_get_seq_list_into(&mut hyper_iter, 2, &mut hyper_seq).unwrap(),
            2
        );
        assert_eq!(hyper_seq, vec![0, 1, 0, 2]);
        let mut hyper_iter = H5S__hyper_iter_init(&hyper, &ds_shape).unwrap();
        let mut hyper_seq = vec![0; 2];
        assert_eq!(
            H5S__hyper_iter_get_seq_list_into(&mut hyper_iter, 1, &mut hyper_seq).unwrap(),
            1
        );
        assert_eq!(hyper_seq, vec![0, 1]);
        let mut hyper_iter = H5S__hyper_iter_init(&hyper, &ds_shape).unwrap();
        let mut hyper_seq = vec![0; 2];
        assert_eq!(
            H5S__hyper_iter_get_seq_list_into(&mut hyper_iter, 1, &mut hyper_seq).unwrap(),
            1
        );
        assert_eq!(hyper_seq, vec![0, 1]);
        let mut hyper_iter = H5S__hyper_iter_init(&hyper, &ds_shape).unwrap();
        let mut hyper_seq = vec![0; 2];
        assert_eq!(
            H5S__hyper_iter_get_seq_list_into(&mut hyper_iter, 1, &mut hyper_seq).unwrap(),
            1
        );
        assert_eq!(hyper_seq, vec![0, 1]);
        assert_eq!(H5S__hyper_iter_nelmts(&hyper_iter), 3);
        H5S__hyper_iter_release(hyper_iter);
        let mut hyper_iter = H5S__hyper_iter_init(&hyper, &ds_shape).unwrap();
        let mut hyper_seq = vec![0; 4];
        assert_eq!(
            H5S__hyper_iter_get_seq_list_into(&mut hyper_iter, 2, &mut hyper_seq).unwrap(),
            2
        );
        assert_eq!(hyper_seq, vec![0, 1, 0, 2]);
        assert_eq!(H5S__hyper_copy(&hyper).unwrap(), hyper);
        assert_eq!(
            H5S__hyper_new_span(0, 1, 1, 1),
            HyperslabDim::new(0, 1, 1, 1)
        );
        assert_eq!(
            H5S__hyper_new_span_info(vec![HyperslabDim::new(0, 1, 1, 1)]),
            Selection::Hyperslab(vec![HyperslabDim::new(0, 1, 1, 1)])
        );
        let mut copied_spans = Vec::new();
        H5S__hyper_copy_span_into(&hyper, &mut copied_spans).unwrap();
        assert_eq!(copied_spans.len(), 2);
        assert!(H5S__hyper_cmp_spans(&hyper, &hyper));
        let mut rendered = String::new();
        H5S__hyper_print_spans_fmt(&hyper, &mut rendered).unwrap();
        assert!(rendered.contains("HyperslabDim"));
        rendered.clear();
        H5S__space_print_spans_fmt(&hyper, &mut rendered).unwrap();
        assert!(rendered.contains("HyperslabDim"));
        rendered.clear();
        H5S__hyper_print_diminfo_fmt(&hyper, &mut rendered).unwrap();
        assert!(rendered.contains("start=0"));
        let mut hyper_payload = Vec::new();
        H5S__hyper_serialize_into(&hyper, &mut hyper_payload).unwrap();
        assert_eq!(
            H5S__hyper_get_enc_size_real(&hyper).unwrap(),
            hyper_payload.len()
        );
        assert_eq!(H5S__hyper_get_version_enc_size(&hyper).unwrap().0, 1);
        assert_eq!(H5S__hyper_deserialize(&hyper_payload).unwrap(), hyper);
        assert_eq!(H5S__hyper_decode(&hyper_payload).unwrap(), hyper);
        assert!(H5S__hyper_is_valid(&hyper, &ds_shape));
        assert_eq!(H5S__hyper_span_nblocks(&hyper, &ds_shape), Some(4));
        assert_eq!(H5S__get_select_hyper_nblocks(&hyper, &ds_shape), Some(4));
        let mut starts = vec![0; 8];
        let mut ends = vec![0; 8];
        assert_eq!(
            H5S__hyper_span_blocklist_into(&hyper, &ds_shape, &mut starts, &mut ends).unwrap(),
            Some(4)
        );
        assert_eq!(
            H5S__get_select_hyper_blocklist_into(&hyper, &ds_shape, &mut starts, &mut ends)
                .unwrap(),
            Some(4)
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
            selection_bounds(&H5S__hyper_offset(&hyper, &[1, 0]).unwrap(), &[3, 3]),
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
        let mut regular_hyperslab = Vec::new();
        H5S__hyper_get_regular_hyperslab_into(&hyper, &mut regular_hyperslab).unwrap();
        assert_eq!(regular_hyperslab.len(), 2);
        assert!(H5S__hyper_coord_to_span(&hyper, &[0, 1], &ds_shape));
        let mut editable_hyper = H5S__hyper_make_spans(vec![HyperslabDim::new(0, 1, 1, 1)]);
        H5S_hyper_add_span_element(&mut editable_hyper, HyperslabDim::new(1, 1, 1, 1)).unwrap();
        H5S__hyper_append_span(&mut editable_hyper, HyperslabDim::new(2, 1, 1, 1)).unwrap();
        let mut updated_diminfo = Vec::new();
        H5S__hyper_update_diminfo_into(&editable_hyper, &mut updated_diminfo).unwrap();
        assert_eq!(updated_diminfo.len(), 3);
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
            selection_bounds(&H5S__hyper_adjust_u(&hyper, &[1, 0]).unwrap(), &[3, 3]),
            Some((vec![1, 1], vec![2, 2]))
        );
        assert_eq!(
            selection_bounds(&H5S__hyper_adjust_s(&hyper, &[0, -1]).unwrap(), &ds_shape),
            Some((vec![0, 0], vec![1, 1]))
        );
        assert_eq!(
            selection_bounds(
                &H5S__hyper_adjust_u_helper(&hyper, &[1, 0]).unwrap(),
                &[3, 3]
            ),
            Some((vec![1, 1], vec![2, 2]))
        );
        assert_eq!(
            selection_bounds(
                &H5S__hyper_adjust_s_helper(&hyper, &[0, -1]).unwrap(),
                &ds_shape
            ),
            Some((vec![0, 0], vec![1, 1]))
        );
        assert_eq!(
            selection_bounds(
                &H5S_hyper_normalize_offset(&hyper, &[1, 0]).unwrap(),
                &[3, 3]
            ),
            Some((vec![1, 1], vec![2, 2]))
        );
        assert_eq!(
            selection_bounds(
                &H5S_hyper_denormalize_offset(&hyper, &[1, 0]).unwrap(),
                &[3, 3]
            ),
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
        let mut projected_intersection = Vec::new();
        H5S__hyper_proj_int_visit(&hyper, &hyper, &ds_shape, &[1], |point| {
            projected_intersection.push(point.to_vec());
            Ok(())
        })
        .unwrap();
        assert_eq!(projected_intersection, vec![vec![1], vec![2]]);
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

    #[test]
    fn allocation_aware_selection_apis_borrow_copy_and_visit() {
        let ds_shape = [2, 3];
        let hyper = Selection::Hyperslab(vec![
            HyperslabDim::new(0, 1, 2, 1),
            HyperslabDim::new(1, 1, 2, 1),
        ]);

        let mut iter = H5S_select_iter_init(&hyper, &ds_shape).unwrap();
        assert_eq!(iter.select_iter_current(), Some(&[0, 1][..]));
        assert_eq!(H5S_select_iter_next_ref(&mut iter), Some(&[0, 1][..]));

        let mut coord = [0; 2];
        assert!(H5S_select_iter_next_into(&mut iter, &mut coord).unwrap());
        assert_eq!(coord, [0, 2]);

        let mut coords = [0; 4];
        assert_eq!(
            H5S_select_iter_get_seq_list_into(&mut iter, 8, &mut coords).unwrap(),
            2
        );
        assert_eq!(coords, [1, 1, 1, 2]);
        assert!(!H5S_select_iter_next_into(&mut iter, &mut coord).unwrap());

        let mut copied_points = [0; 8];
        assert_eq!(
            H5S_select_copy_points_into(&hyper, &ds_shape, &mut copied_points).unwrap(),
            4
        );
        assert_eq!(copied_points, [0, 1, 0, 2, 1, 1, 1, 2]);

        let mut visited = Vec::new();
        hyper
            .visit_points(&ds_shape, |point| {
                visited.push(point.to_vec());
                Ok(())
            })
            .unwrap();
        assert_eq!(
            visited,
            vec![vec![0, 1], vec![0, 2], vec![1, 1], vec![1, 2]]
        );

        let mut starts = [0; 8];
        let mut ends = [0; 8];
        assert_eq!(
            H5Sget_select_hyper_blocklist_into(&hyper, &ds_shape, &mut starts, &mut ends).unwrap(),
            Some(4)
        );
        assert_eq!(starts, [0, 1, 0, 2, 1, 1, 1, 2]);
        assert_eq!(ends, [0, 1, 0, 2, 1, 1, 1, 2]);

        let mut visited_blocks = 0;
        assert!(hyper
            .visit_hyperslab_blocks(&ds_shape, |start, end| {
                assert_eq!(start, end);
                visited_blocks += 1;
                Ok(())
            })
            .unwrap());
        assert_eq!(visited_blocks, 4);

        let points = Selection::Points(vec![vec![3, 4], vec![5, 6]]);
        let borrowed: Vec<_> = points.element_points().unwrap().collect();
        assert_eq!(borrowed, vec![&[3, 4][..], &[5, 6][..]]);
        let mut flat_points = [0; 4];
        assert_eq!(
            H5Sget_select_elem_pointlist_into(&points, &mut flat_points).unwrap(),
            Some(2)
        );
        assert_eq!(flat_points, [3, 4, 5, 6]);
    }

    #[test]
    fn regular_selection_iterators_do_not_use_materialization_cap() {
        let ds_shape = [MAX_MATERIALIZED_SELECTION_POINTS as u64 + 1];
        let mut iter = Selection::All.select_iter_init(&ds_shape).unwrap();
        assert_eq!(iter.len(), MAX_MATERIALIZED_SELECTION_POINTS + 1);
        assert_eq!(iter.select_iter_next_ref(), Some(&[0][..]));
    }
}
