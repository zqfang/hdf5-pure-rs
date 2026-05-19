use crate::error::{Error, Result};
use crate::format::messages::data_layout::LayoutClass;

use super::{u64_from_usize, usize_from_u64, Dataset, DatasetInfo};

impl Dataset {
    /// Read a subset of the dataset using a selection.
    ///
    /// Example: `ds.read_slice::<f64>(10..20)` reads elements 10-19 from a 1D dataset.
    pub fn read_slice<T: crate::hl::types::H5Type, S: crate::hl::selection::IntoSelection>(
        &self,
        sel: S,
    ) -> Result<Vec<T>> {
        let mut shape = Vec::new();
        self.shape_into(&mut shape)?;
        let selection = sel.into_selection(&shape);
        self.read_slice_alloc::<T>(&shape, &selection)
    }

    fn read_slice_alloc<T: crate::hl::types::H5Type>(
        &self,
        shape: &[u64],
        selection: &crate::hl::selection::Selection,
    ) -> Result<Vec<T>> {
        if matches!(selection, crate::hl::selection::Selection::None) {
            return Ok(Vec::new());
        }
        if let crate::hl::selection::Selection::Points(points) = &selection {
            Self::validate_selection_points(shape, points)?;
            let all_data = self.read_dataset_values::<T>()?;
            return Self::extract_point_selection(&all_data, shape, points);
        }
        if let crate::hl::selection::Selection::Hyperslab(dims) = &selection {
            Self::validate_hyperslab_selection(shape, dims)?;
            let mut out_shape = Vec::new();
            selection.output_shape_into(shape, &mut out_shape);
            let total_out = Self::selection_output_elements(&out_shape)?;
            if total_out == 0 {
                return Ok(Vec::new());
            }
            let total_out_usize = usize_from_u64(total_out, "hyperslab selection element count")?;
            let all_data = self.read_dataset_values::<T>()?;
            return Self::extract_hyperslab_selection(
                &all_data,
                shape,
                dims,
                &out_shape,
                total_out,
                total_out_usize,
            );
        }

        let slices = selection.to_slices(shape);
        Self::validate_selection_slices(shape, &slices)?;
        let mut out_shape = Vec::new();
        selection.output_shape_into(shape, &mut out_shape);
        let total_out = Self::selection_output_elements(&out_shape)?;
        let total_out_usize = usize_from_u64(total_out, "selection element count")?;
        if total_out == 0 {
            return Ok(Vec::new());
        }

        if let Some(values) = self.try_read_slice_direct_1d::<T>(&shape, &slices)? {
            return Ok(values);
        }

        let all_data = self.read_dataset_values::<T>()?;
        if shape.is_empty() {
            return Ok(all_data);
        }
        if Self::selection_slices_cover_shape(shape, &slices) {
            return Ok(all_data);
        }

        if shape.len() == 1 && slices.len() == 1 {
            return Self::extract_1d_selection(&all_data, &slices[0]);
        }

        Self::extract_nd_selection(
            &all_data,
            shape,
            &slices,
            &out_shape,
            total_out,
            total_out_usize,
        )
    }

    /// Read a subset of the dataset into caller-provided storage.
    ///
    /// The output buffer length must exactly match the number of selected
    /// elements.
    pub fn read_slice_into<T: crate::hl::types::H5Type, S: crate::hl::selection::IntoSelection>(
        &self,
        sel: S,
        out: &mut [T],
    ) -> Result<()> {
        let mut shape = Vec::new();
        self.shape_into(&mut shape)?;
        let selection = sel.into_selection(&shape);
        self.read_selection_into(&shape, &selection, out)
    }

    /// Read an already-built selection into caller-provided storage.
    ///
    /// This is the allocation-aware form for callers that already have the
    /// dataset shape and selection object, such as repeated reads over reused
    /// selection metadata.
    pub fn read_selection_into<T: crate::hl::types::H5Type>(
        &self,
        shape: &[u64],
        selection: &crate::hl::selection::Selection,
        out: &mut [T],
    ) -> Result<()> {
        let expected_len = Self::selection_output_len(shape, selection)?;
        if out.len() != expected_len {
            return Err(Error::InvalidFormat(format!(
                "slice output buffer has {} elements, expected {expected_len}",
                out.len()
            )));
        }
        if expected_len == 0 {
            return Ok(());
        }

        if !matches!(
            selection,
            crate::hl::selection::Selection::Points(_)
                | crate::hl::selection::Selection::Hyperslab(_)
        ) {
            let slices = selection.to_slices(shape);
            let elem_size = T::type_size();
            if elem_size == 0 {
                return Err(Error::Other("zero-size type".into()));
            }
            if self.try_read_slice_direct_1d_into(shape, &slices, out)? {
                return Ok(());
            }
        }

        self.read_slice_into_impl(shape, selection, out)
    }

    fn read_slice_into_impl<T: crate::hl::types::H5Type>(
        &self,
        shape: &[u64],
        selection: &crate::hl::selection::Selection,
        out: &mut [T],
    ) -> Result<()> {
        if matches!(selection, crate::hl::selection::Selection::None) {
            return Ok(());
        }

        let all_data = self.read_dataset_values::<T>()?;
        if let crate::hl::selection::Selection::Points(points) = selection {
            Self::extract_point_selection_into(&all_data, shape, points, out)?;
            return Ok(());
        }
        if let crate::hl::selection::Selection::Hyperslab(dims) = selection {
            let mut out_shape = Vec::new();
            selection.output_shape_into(shape, &mut out_shape);
            let total_out = Self::selection_output_elements(&out_shape)?;
            Self::extract_hyperslab_selection_into(
                &all_data, shape, dims, &out_shape, total_out, out,
            )?;
            return Ok(());
        }

        let slices = selection.to_slices(shape);
        if shape.is_empty() {
            out.copy_from_slice(&all_data);
            return Ok(());
        }
        if shape.len() == 1 && slices.len() == 1 {
            Self::extract_1d_selection_into(&all_data, &slices[0], out)?;
            return Ok(());
        }

        let mut out_shape = Vec::new();
        selection.output_shape_into(shape, &mut out_shape);
        let total_out = Self::selection_output_elements(&out_shape)?;
        Self::extract_nd_selection_into(&all_data, shape, &slices, &out_shape, total_out, out)
    }

    fn read_dataset_values<T: crate::hl::types::H5Type>(&self) -> Result<Vec<T>> {
        let info = self.info()?;
        let conversion = crate::hl::conversion::ReadConversion::for_dataset::<T>(&info.datatype)?;
        let element_size = usize_from_u64(u64::from(info.datatype.size), "datatype size")?;
        if element_size == 0 {
            return Err(Error::InvalidFormat("zero-sized datatype".into()));
        }
        let access = crate::hl::dataset::DatasetAccess::new();
        let total_elements = if info.layout.layout_class == LayoutClass::Virtual {
            let shape = self.virtual_shape_with_info(&info, &access)?;
            Self::dataspace_element_count(info.dataspace.space_type, &shape)?
        } else {
            Self::dataspace_element_count(info.dataspace.space_type, &info.dataspace.dims)?
        };
        let total_elements_usize = usize_from_u64(total_elements, "dimension product")?;
        let total_bytes = total_elements_usize
            .checked_mul(element_size)
            .ok_or_else(|| Error::InvalidFormat("total data size overflow".into()))?;
        if conversion.is_same_size_bytes() {
            let mut values = Vec::<T>::with_capacity(total_elements_usize);
            let raw_out = unsafe {
                std::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut u8, total_bytes)
            };
            self.read_raw_into_with_info(&info, &access, raw_out)?;
            conversion.convert_bytes_in_place(raw_out);
            unsafe {
                values.set_len(total_elements_usize);
            }
            return Ok(values);
        }
        let mut raw = vec![0u8; total_bytes];
        self.read_raw_into_with_info(&info, &access, &mut raw)?;
        conversion.bytes_to_vec(raw)
    }

    fn selection_output_len(
        shape: &[u64],
        selection: &crate::hl::selection::Selection,
    ) -> Result<usize> {
        match selection {
            crate::hl::selection::Selection::None => Ok(0),
            crate::hl::selection::Selection::Points(points) => {
                Self::validate_selection_points(shape, points)?;
                Ok(points.len())
            }
            crate::hl::selection::Selection::Hyperslab(dims) => {
                Self::validate_hyperslab_selection(shape, dims)?;
                let mut out_shape = Vec::new();
                selection.output_shape_into(shape, &mut out_shape);
                let total_out = Self::selection_output_elements(&out_shape)?;
                usize_from_u64(total_out, "hyperslab selection element count")
            }
            _ => {
                let slices = selection.to_slices(shape);
                Self::validate_selection_slices(shape, &slices)?;
                let mut out_shape = Vec::new();
                selection.output_shape_into(shape, &mut out_shape);
                let total_out = Self::selection_output_elements(&out_shape)?;
                usize_from_u64(total_out, "selection element count")
            }
        }
    }

    fn selection_output_elements(out_shape: &[u64]) -> Result<u64> {
        if out_shape.is_empty() {
            return Ok(1);
        }
        out_shape.iter().try_fold(1u64, |acc, &dim| {
            acc.checked_mul(dim)
                .ok_or_else(|| Error::InvalidFormat("selection element count overflow".into()))
        })
    }

    fn selection_slices_cover_shape(
        shape: &[u64],
        slices: &[crate::hl::selection::SliceInfo],
    ) -> bool {
        shape.len() == slices.len()
            && shape
                .iter()
                .zip(slices)
                .all(|(&extent, slice)| slice.start == 0 && slice.end == extent && slice.step == 1)
    }

    fn validate_selection_slices(
        shape: &[u64],
        slices: &[crate::hl::selection::SliceInfo],
    ) -> Result<()> {
        if slices.len() != shape.len() {
            return Err(Error::InvalidFormat(format!(
                "selection rank {} does not match dataset rank {}",
                slices.len(),
                shape.len()
            )));
        }
        for (dim, (slice, &extent)) in slices.iter().zip(shape).enumerate() {
            if slice.step == 0 {
                return Err(Error::InvalidFormat(format!(
                    "selection dimension {dim} has zero step"
                )));
            }
            if slice.start > extent || slice.end > extent {
                return Err(Error::InvalidFormat(format!(
                    "selection dimension {dim} range {}..{} exceeds extent {extent}",
                    slice.start, slice.end
                )));
            }
        }
        Ok(())
    }

    fn validate_hyperslab_selection(
        shape: &[u64],
        dims: &[crate::hl::selection::HyperslabDim],
    ) -> Result<()> {
        if dims.len() != shape.len() {
            return Err(Error::InvalidFormat(format!(
                "hyperslab rank {} does not match dataset rank {}",
                dims.len(),
                shape.len()
            )));
        }
        for (dim, (selection, &extent)) in dims.iter().zip(shape).enumerate() {
            if selection.stride == 0 {
                return Err(Error::InvalidFormat(format!(
                    "hyperslab dimension {dim} has zero stride"
                )));
            }
            if selection.block == 0 && selection.count != 0 {
                return Err(Error::InvalidFormat(format!(
                    "hyperslab dimension {dim} has zero block"
                )));
            }
            if selection.count == 0 || selection.block == 0 {
                continue;
            }
            let span_start = selection
                .count
                .checked_sub(1)
                .and_then(|count_minus_one| count_minus_one.checked_mul(selection.stride))
                .and_then(|offset| selection.start.checked_add(offset))
                .ok_or_else(|| Error::InvalidFormat("hyperslab extent overflow".into()))?;
            let span_end = span_start
                .checked_add(selection.block)
                .ok_or_else(|| Error::InvalidFormat("hyperslab extent overflow".into()))?;
            if span_end > extent {
                return Err(Error::InvalidFormat(format!(
                    "hyperslab dimension {dim} exceeds extent {extent}"
                )));
            }
        }
        Ok(())
    }

    fn validate_selection_points(shape: &[u64], points: &[Vec<u64>]) -> Result<()> {
        for point in points {
            if point.len() != shape.len() {
                return Err(Error::InvalidFormat(format!(
                    "point selection rank {} does not match dataset rank {}",
                    point.len(),
                    shape.len()
                )));
            }
            for (dim, (&coord, &extent)) in point.iter().zip(shape).enumerate() {
                if coord >= extent {
                    return Err(Error::InvalidFormat(format!(
                        "point selection coordinate {coord} in dimension {dim} exceeds extent {extent}"
                    )));
                }
            }
        }
        Ok(())
    }

    fn extract_point_selection<T: crate::hl::types::H5Type>(
        all_data: &[T],
        shape: &[u64],
        points: &[Vec<u64>],
    ) -> Result<Vec<T>> {
        let strides = Self::row_major_strides(shape)?;
        let mut result = Vec::with_capacity(points.len());
        for point in points {
            let index = Self::linear_index(point, &strides)?;
            if index < all_data.len() {
                result.push(all_data[index]);
            }
        }
        Ok(result)
    }

    fn extract_point_selection_into<T: crate::hl::types::H5Type>(
        all_data: &[T],
        shape: &[u64],
        points: &[Vec<u64>],
        out: &mut [T],
    ) -> Result<()> {
        let strides = Self::row_major_strides(shape)?;
        let mut written = 0usize;
        for point in points {
            let index = Self::linear_index(point, &strides)?;
            if index < all_data.len() {
                out[written] = all_data[index];
                written += 1;
            }
        }
        if written == out.len() {
            Ok(())
        } else {
            Err(Error::InvalidFormat(format!(
                "point selection produced {written} elements, expected {}",
                out.len()
            )))
        }
    }

    fn extract_hyperslab_selection<T: crate::hl::types::H5Type>(
        all_data: &[T],
        shape: &[u64],
        dims: &[crate::hl::selection::HyperslabDim],
        out_shape: &[u64],
        total_out: u64,
        total_out_usize: usize,
    ) -> Result<Vec<T>> {
        let strides = Self::row_major_strides(shape)?;
        let ndims = shape.len();
        let mut result = Vec::with_capacity(total_out_usize);
        let mut out_idx = vec![0u64; ndims];
        for _ in 0..total_out {
            let mut in_linear = 0usize;
            for dim in 0..ndims {
                let selected_block = out_idx[dim] / dims[dim].block;
                let selected_offset = out_idx[dim] % dims[dim].block;
                let in_d =
                    dims[dim]
                        .start
                        .checked_add(selected_block.checked_mul(dims[dim].stride).ok_or_else(
                            || Error::InvalidFormat("hyperslab coordinate overflow".into()),
                        )?)
                        .and_then(|coord| coord.checked_add(selected_offset))
                        .ok_or_else(|| {
                            Error::InvalidFormat("hyperslab coordinate overflow".into())
                        })?;
                let term = usize_from_u64(in_d, "hyperslab input index")?
                    .checked_mul(strides[dim])
                    .ok_or_else(|| {
                        Error::InvalidFormat("hyperslab linear index overflow".into())
                    })?;
                in_linear = in_linear.checked_add(term).ok_or_else(|| {
                    Error::InvalidFormat("hyperslab linear index overflow".into())
                })?;
            }
            if in_linear < all_data.len() {
                result.push(all_data[in_linear]);
            }
            for dim in (0..ndims).rev() {
                out_idx[dim] = out_idx[dim].checked_add(1).ok_or_else(|| {
                    Error::InvalidFormat("hyperslab output index overflow".into())
                })?;
                if out_idx[dim] < out_shape[dim] {
                    break;
                }
                out_idx[dim] = 0;
            }
        }
        Ok(result)
    }

    fn extract_hyperslab_selection_into<T: crate::hl::types::H5Type>(
        all_data: &[T],
        shape: &[u64],
        dims: &[crate::hl::selection::HyperslabDim],
        out_shape: &[u64],
        total_out: u64,
        out: &mut [T],
    ) -> Result<()> {
        let strides = Self::row_major_strides(shape)?;
        let ndims = shape.len();
        let mut out_idx = vec![0u64; ndims];
        let mut written = 0usize;
        for _ in 0..total_out {
            let mut in_linear = 0usize;
            for dim in 0..ndims {
                let selected_block = out_idx[dim] / dims[dim].block;
                let selected_offset = out_idx[dim] % dims[dim].block;
                let in_d =
                    dims[dim]
                        .start
                        .checked_add(selected_block.checked_mul(dims[dim].stride).ok_or_else(
                            || Error::InvalidFormat("hyperslab coordinate overflow".into()),
                        )?)
                        .and_then(|coord| coord.checked_add(selected_offset))
                        .ok_or_else(|| {
                            Error::InvalidFormat("hyperslab coordinate overflow".into())
                        })?;
                let term = usize_from_u64(in_d, "hyperslab input index")?
                    .checked_mul(strides[dim])
                    .ok_or_else(|| {
                        Error::InvalidFormat("hyperslab linear index overflow".into())
                    })?;
                in_linear = in_linear.checked_add(term).ok_or_else(|| {
                    Error::InvalidFormat("hyperslab linear index overflow".into())
                })?;
            }
            if in_linear < all_data.len() {
                out[written] = all_data[in_linear];
                written += 1;
            }
            for dim in (0..ndims).rev() {
                out_idx[dim] = out_idx[dim].checked_add(1).ok_or_else(|| {
                    Error::InvalidFormat("hyperslab output index overflow".into())
                })?;
                if out_idx[dim] < out_shape[dim] {
                    break;
                }
                out_idx[dim] = 0;
            }
        }
        if written == out.len() {
            Ok(())
        } else {
            Err(Error::InvalidFormat(format!(
                "hyperslab selection produced {written} elements, expected {}",
                out.len()
            )))
        }
    }

    fn try_read_slice_direct_1d<T: crate::hl::types::H5Type>(
        &self,
        shape: &[u64],
        slices: &[crate::hl::selection::SliceInfo],
    ) -> Result<Option<Vec<T>>> {
        if !(shape.len() == 1 && slices.len() == 1 && slices[0].step == 1) {
            return Ok(None);
        }

        let info = self.info()?;
        let conversion = crate::hl::conversion::ReadConversion::for_dataset::<T>(&info.datatype)?;
        if !conversion.is_same_size_bytes() {
            return Ok(None);
        }
        let elem_size = usize_from_u64(u64::from(info.datatype.size), "datatype size")?;
        if elem_size == 0 {
            return Err(Error::Other("zero-size type".into()));
        }
        let nbytes = usize_from_u64(slices[0].count(), "selection count")?
            .checked_mul(elem_size)
            .ok_or_else(|| Error::InvalidFormat("selection byte count overflow".into()))?;
        let count = usize_from_u64(slices[0].count(), "selection count")?;
        let mut values = Vec::<T>::with_capacity(count);
        if nbytes > 0 {
            let raw_out =
                unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut u8, nbytes) };
            match info.layout.layout_class {
                LayoutClass::Contiguous => {
                    let Some(addr) = info.layout.contiguous_addr else {
                        return Ok(None);
                    };
                    if crate::io::reader::is_undef_addr(addr) {
                        return Ok(None);
                    }

                    let start_byte = usize_from_u64(slices[0].start, "selection start")?
                        .checked_mul(elem_size)
                        .ok_or_else(|| {
                            Error::InvalidFormat("selection byte offset overflow".into())
                        })?;
                    let read_addr = addr
                        .checked_add(u64_from_usize(start_byte, "selection byte offset")?)
                        .ok_or_else(|| {
                            Error::InvalidFormat("selection read address overflow".into())
                        })?;

                    let mut guard = self.inner.lock();
                    guard.reader.seek(read_addr)?;
                    guard.reader.read_exact(raw_out)?;
                    conversion.convert_bytes_in_place(raw_out);
                }
                LayoutClass::Chunked => {
                    if !self
                        .try_read_chunked_slice_1d_into(&info, &slices[0], elem_size, raw_out)?
                    {
                        return Ok(None);
                    }
                    conversion.convert_bytes_in_place(raw_out);
                }
                _ => return Ok(None),
            }
        }
        unsafe {
            values.set_len(count);
        }
        Ok(Some(values))
    }

    fn try_read_slice_direct_1d_into<T: crate::hl::types::H5Type>(
        &self,
        shape: &[u64],
        slices: &[crate::hl::selection::SliceInfo],
        out: &mut [T],
    ) -> Result<bool> {
        if !(shape.len() == 1 && slices.len() == 1 && slices[0].step == 1) {
            return Ok(false);
        }

        let info = self.info()?;
        let conversion = crate::hl::conversion::ReadConversion::for_dataset::<T>(&info.datatype)?;
        if !conversion.is_same_size_bytes() {
            return Ok(false);
        }
        let elem_size = usize_from_u64(u64::from(info.datatype.size), "datatype size")?;
        if elem_size == 0 {
            return Err(Error::Other("zero-size type".into()));
        }
        let raw_out = crate::hl::types::slice_as_bytes_mut(out);
        match info.layout.layout_class {
            LayoutClass::Contiguous => {
                let Some(addr) = info.layout.contiguous_addr else {
                    return Ok(false);
                };
                if crate::io::reader::is_undef_addr(addr) {
                    return Ok(false);
                }

                let start_byte = usize_from_u64(slices[0].start, "selection start")?
                    .checked_mul(elem_size)
                    .ok_or_else(|| Error::InvalidFormat("selection byte offset overflow".into()))?;
                let read_addr = addr
                    .checked_add(u64_from_usize(start_byte, "selection byte offset")?)
                    .ok_or_else(|| {
                        Error::InvalidFormat("selection read address overflow".into())
                    })?;

                let mut guard = self.inner.lock();
                guard.reader.seek(read_addr)?;
                guard.reader.read_exact(raw_out)?;
                conversion.convert_bytes_in_place(raw_out);
                Ok(true)
            }
            LayoutClass::Chunked => {
                if self.try_read_chunked_slice_1d_into(&info, &slices[0], elem_size, raw_out)? {
                    conversion.convert_bytes_in_place(raw_out);
                    return Ok(true);
                }
                Ok(false)
            }
            _ => Ok(false),
        }
    }

    fn try_read_chunked_slice_1d_into(
        &self,
        info: &DatasetInfo,
        slice: &crate::hl::selection::SliceInfo,
        elem_size: usize,
        raw_out: &mut [u8],
    ) -> Result<bool> {
        let filtered = info
            .filter_pipeline
            .as_ref()
            .map(|pipeline| !pipeline.filters.is_empty())
            .unwrap_or(false);
        if filtered {
            return Ok(false);
        }

        let raw_chunk_dims = info
            .layout
            .chunk_dims
            .as_ref()
            .ok_or_else(|| Error::InvalidFormat("chunked dataset missing chunk dims".into()))?;
        let chunk_dims = Self::chunk_data_dims(&info.dataspace.dims, raw_chunk_dims)?;
        if info.dataspace.dims.len() != 1 || chunk_dims.len() != 1 {
            return Ok(false);
        }

        Self::filled_data_into(raw_out.len() / elem_size, elem_size, info, raw_out)?;

        let selection_start = usize_from_u64(slice.start, "selection start")?;
        let selection_end = usize_from_u64(slice.end, "selection end")?;
        let chunk_len = usize_from_u64(chunk_dims[0], "chunk dimension")?;
        let data_len = usize_from_u64(info.dataspace.dims[0], "dataset dimension")?;

        let mut reads = Vec::new();
        let mut can_direct = true;
        self.visit_chunk_infos(|offset, filter_mask, addr, size| {
            if offset.len() != 1 || crate::io::reader::is_undef_addr(addr) {
                return Ok(());
            }

            let chunk_start = usize_from_u64(offset[0], "chunk coordinate")?;
            let chunk_end = chunk_start
                .checked_add(chunk_len)
                .ok_or_else(|| Error::InvalidFormat("chunk coordinate overflow".into()))?
                .min(data_len);
            let copy_start = selection_start.max(chunk_start);
            let copy_end = selection_end.min(chunk_end);
            if copy_start >= copy_end {
                return Ok(());
            }
            if filter_mask != 0 {
                can_direct = false;
                return Ok(());
            }

            let src_offset = copy_start
                .checked_sub(chunk_start)
                .and_then(|elements| elements.checked_mul(elem_size))
                .ok_or_else(|| Error::InvalidFormat("chunk slice offset overflow".into()))?;
            let copy_bytes = copy_end
                .checked_sub(copy_start)
                .and_then(|elements| elements.checked_mul(elem_size))
                .ok_or_else(|| Error::InvalidFormat("chunk slice byte count overflow".into()))?;
            let read_size = usize_from_u64(size, "chunk size")?;
            let src_end = src_offset
                .checked_add(copy_bytes)
                .ok_or_else(|| Error::InvalidFormat("chunk slice range overflow".into()))?;
            if src_end > read_size {
                can_direct = false;
                return Ok(());
            }

            let dst_offset = copy_start
                .checked_sub(selection_start)
                .and_then(|elements| elements.checked_mul(elem_size))
                .ok_or_else(|| Error::InvalidFormat("selection output offset overflow".into()))?;
            let read_addr = addr
                .checked_add(u64_from_usize(src_offset, "chunk slice offset")?)
                .ok_or_else(|| Error::InvalidFormat("chunk slice read address overflow".into()))?;
            reads.push((read_addr, dst_offset, copy_bytes));
            Ok(())
        })?;
        if !can_direct {
            return Ok(false);
        }

        let mut guard = self.inner.lock();
        for (addr, dst_offset, copy_bytes) in reads {
            let dst_end = dst_offset
                .checked_add(copy_bytes)
                .ok_or_else(|| Error::InvalidFormat("selection output range overflow".into()))?;
            guard.reader.seek(addr)?;
            guard.reader.read_exact(&mut raw_out[dst_offset..dst_end])?;
        }
        Ok(true)
    }

    fn extract_1d_selection<T: crate::hl::types::H5Type>(
        all_data: &[T],
        slice: &crate::hl::selection::SliceInfo,
    ) -> Result<Vec<T>> {
        let start = usize_from_u64(slice.start, "selection start")?;
        let end = usize_from_u64(slice.end, "selection end")?;
        if start > all_data.len() {
            return Ok(Vec::new());
        }
        let end = end.min(all_data.len());
        if slice.step == 1 {
            return Ok(all_data[start..end].to_vec());
        }
        let step = usize_from_u64(slice.step, "selection step")?;
        Ok(all_data[start..end].iter().step_by(step).copied().collect())
    }

    fn extract_1d_selection_into<T: crate::hl::types::H5Type>(
        all_data: &[T],
        slice: &crate::hl::selection::SliceInfo,
        out: &mut [T],
    ) -> Result<()> {
        let start = usize_from_u64(slice.start, "selection start")?;
        let end = usize_from_u64(slice.end, "selection end")?.min(all_data.len());
        if start > all_data.len() {
            if out.is_empty() {
                return Ok(());
            }
            return Err(Error::InvalidFormat(format!(
                "1D selection produced 0 elements, expected {}",
                out.len()
            )));
        }
        let step = usize_from_u64(slice.step, "selection step")?;
        let mut written = 0usize;
        for value in all_data[start..end].iter().step_by(step) {
            out[written] = *value;
            written += 1;
        }
        if written == out.len() {
            Ok(())
        } else {
            Err(Error::InvalidFormat(format!(
                "1D selection produced {written} elements, expected {}",
                out.len()
            )))
        }
    }

    fn extract_nd_selection<T: crate::hl::types::H5Type>(
        all_data: &[T],
        shape: &[u64],
        slices: &[crate::hl::selection::SliceInfo],
        out_shape: &[u64],
        total_out: u64,
        total_out_usize: usize,
    ) -> Result<Vec<T>> {
        let mut result = Vec::with_capacity(total_out_usize);
        let ndims = shape.len();

        let mut in_strides = vec![1usize; ndims];
        for d in (0..ndims - 1).rev() {
            in_strides[d] = in_strides[d + 1]
                .checked_mul(usize_from_u64(shape[d + 1], "selection shape")?)
                .ok_or_else(|| Error::InvalidFormat("selection stride overflow".into()))?;
        }

        let mut out_idx = vec![0u64; ndims];
        for _ in 0..total_out {
            let mut in_linear = 0usize;
            for d in 0..ndims {
                let in_d = out_idx[d]
                    .checked_mul(slices[d].step)
                    .and_then(|offset| slices[d].start.checked_add(offset))
                    .ok_or_else(|| Error::InvalidFormat("selection coordinate overflow".into()))?;
                let term = usize_from_u64(in_d, "selection input index")?
                    .checked_mul(in_strides[d])
                    .ok_or_else(|| {
                        Error::InvalidFormat("selection linear index overflow".into())
                    })?;
                in_linear = in_linear.checked_add(term).ok_or_else(|| {
                    Error::InvalidFormat("selection linear index overflow".into())
                })?;
            }

            if in_linear < all_data.len() {
                result.push(all_data[in_linear]);
            }

            for d in (0..ndims).rev() {
                out_idx[d] = out_idx[d].checked_add(1).ok_or_else(|| {
                    Error::InvalidFormat("selection output index overflow".into())
                })?;
                if out_idx[d] < out_shape[d] {
                    break;
                }
                out_idx[d] = 0;
            }
        }

        Ok(result)
    }

    fn extract_nd_selection_into<T: crate::hl::types::H5Type>(
        all_data: &[T],
        shape: &[u64],
        slices: &[crate::hl::selection::SliceInfo],
        out_shape: &[u64],
        total_out: u64,
        out: &mut [T],
    ) -> Result<()> {
        let ndims = shape.len();

        let mut in_strides = vec![1usize; ndims];
        for d in (0..ndims - 1).rev() {
            in_strides[d] = in_strides[d + 1]
                .checked_mul(usize_from_u64(shape[d + 1], "selection shape")?)
                .ok_or_else(|| Error::InvalidFormat("selection stride overflow".into()))?;
        }

        let mut out_idx = vec![0u64; ndims];
        let mut written = 0usize;
        for _ in 0..total_out {
            let mut in_linear = 0usize;
            for d in 0..ndims {
                let in_d = out_idx[d]
                    .checked_mul(slices[d].step)
                    .and_then(|offset| slices[d].start.checked_add(offset))
                    .ok_or_else(|| Error::InvalidFormat("selection coordinate overflow".into()))?;
                let term = usize_from_u64(in_d, "selection input index")?
                    .checked_mul(in_strides[d])
                    .ok_or_else(|| {
                        Error::InvalidFormat("selection linear index overflow".into())
                    })?;
                in_linear = in_linear.checked_add(term).ok_or_else(|| {
                    Error::InvalidFormat("selection linear index overflow".into())
                })?;
            }

            if in_linear < all_data.len() {
                out[written] = all_data[in_linear];
                written += 1;
            }

            for d in (0..ndims).rev() {
                out_idx[d] = out_idx[d].checked_add(1).ok_or_else(|| {
                    Error::InvalidFormat("selection output index overflow".into())
                })?;
                if out_idx[d] < out_shape[d] {
                    break;
                }
                out_idx[d] = 0;
            }
        }

        if written == out.len() {
            Ok(())
        } else {
            Err(Error::InvalidFormat(format!(
                "selection produced {written} elements, expected {}",
                out.len()
            )))
        }
    }

    pub(super) fn row_major_strides(dims: &[u64]) -> Result<Vec<usize>> {
        let mut strides = vec![1usize; dims.len()];
        for dim in (0..dims.len().saturating_sub(1)).rev() {
            strides[dim] = strides[dim + 1]
                .checked_mul(usize_from_u64(dims[dim + 1], "dataspace dimension")?)
                .ok_or_else(|| Error::InvalidFormat("dataspace stride overflow".into()))?;
        }
        Ok(strides)
    }

    pub(super) fn linear_index(coords: &[u64], strides: &[usize]) -> Result<usize> {
        coords
            .iter()
            .zip(strides)
            .try_fold(0usize, |acc, (&coord, &stride)| {
                acc.checked_add(
                    usize_from_u64(coord, "dataspace coordinate")
                        .ok()?
                        .checked_mul(stride)?,
                )
            })
            .ok_or_else(|| Error::InvalidFormat("linear index overflow".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hl::selection::SliceInfo;

    #[test]
    fn nd_slice_extraction_rejects_coordinate_overflow() {
        let slices = [SliceInfo {
            start: u64::MAX - 1,
            end: u64::MAX,
            step: 2,
        }];
        let err = Dataset::extract_nd_selection::<u8>(&[0], &[u64::MAX], &slices, &[2], 2, 2)
            .unwrap_err();
        assert!(
            err.to_string().contains("selection coordinate overflow"),
            "unexpected error: {err}"
        );
    }
}
