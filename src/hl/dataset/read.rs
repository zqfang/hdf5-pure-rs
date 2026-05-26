use crate::error::{Error, Result};
use crate::format::messages::data_layout::LayoutClass;
use crate::format::messages::dataspace::DataspaceType;
use crate::format::messages::fill_value::FILL_TIME_NEVER;
use crate::format::object_header::ObjectHeader;

use super::{u64_from_usize, usize_from_u64, Dataset, DatasetAccess, DatasetInfo, VdsView};

impl Dataset {
    /// Read all raw data bytes from the dataset.
    pub fn read_raw(&self) -> Result<Vec<u8>> {
        self.read_raw_vec_with_access(
            &DatasetAccess::new().with_virtual_view(VdsView::LastAvailable),
        )
    }

    /// Read all raw bytes, overriding the VDS view policy.
    pub fn read_raw_with_vds_view(&self, view: VdsView) -> Result<Vec<u8>> {
        self.read_raw_vec_with_access(&DatasetAccess::new().with_virtual_view(view))
    }

    /// Read all raw bytes, overriding dataset access properties.
    pub fn read_raw_with_access(&self, access: &DatasetAccess) -> Result<Vec<u8>> {
        self.read_raw_vec_with_access(access)
    }

    fn read_raw_vec_with_access(&self, access: &DatasetAccess) -> Result<Vec<u8>> {
        let info = {
            let mut guard = self.inner.lock();
            let sizeof_addr = guard.superblock.sizeof_addr;
            let sizeof_size = guard.superblock.sizeof_size;
            let oh = ObjectHeader::read_at(&mut guard.reader, self.addr)?;
            Self::parse_info(&oh.messages, sizeof_addr, sizeof_size)?
        };

        self.read_raw_with_info(info, access)
    }

    /// Read all raw bytes into caller-provided storage.
    ///
    /// This mirrors HDF5's caller-buffer `H5Dread` shape for raw bytes. The
    /// existing `read_raw` convenience method remains available when ownership
    /// of a new `Vec` is desired.
    pub fn read_raw_into(&self, out: &mut [u8]) -> Result<()> {
        self.read_raw_into_with_vds_view(VdsView::LastAvailable, out)
    }

    /// Read all raw bytes into caller-provided storage, overriding the VDS view.
    pub fn read_raw_into_with_vds_view(&self, view: VdsView, out: &mut [u8]) -> Result<()> {
        self.read_raw_into_with_access(&DatasetAccess::new().with_virtual_view(view), out)
    }

    /// Read all raw bytes into caller-provided storage, overriding access properties.
    pub fn read_raw_into_with_access(&self, access: &DatasetAccess, out: &mut [u8]) -> Result<()> {
        let info = {
            let mut guard = self.inner.lock();
            let sizeof_addr = guard.superblock.sizeof_addr;
            let sizeof_size = guard.superblock.sizeof_size;
            let oh = ObjectHeader::read_at(&mut guard.reader, self.addr)?;
            Self::parse_info(&oh.messages, sizeof_addr, sizeof_size)?
        };

        self.read_raw_into_with_info(&info, access, out)
    }

    fn read_raw_with_info(&self, info: DatasetInfo, access: &DatasetAccess) -> Result<Vec<u8>> {
        if info.layout.layout_class == LayoutClass::Virtual {
            return self.read_virtual_raw_with_info(info, access);
        }

        let (_, _, total_bytes) = Self::raw_read_size(&info)?;

        // Sanity limit: refuse to allocate more than 4GB in a single read
        const MAX_READ_BYTES: usize = 4 * 1024 * 1024 * 1024;
        if total_bytes > MAX_READ_BYTES {
            return Err(Error::InvalidFormat(format!(
                "dataset too large for single read: {total_bytes} bytes (max {MAX_READ_BYTES})"
            )));
        }

        let mut out = vec![0; total_bytes];
        self.read_raw_into_with_info(&info, access, &mut out)?;
        Ok(out)
    }

    pub(crate) fn read_raw_into_with_info(
        &self,
        info: &DatasetInfo,
        access: &DatasetAccess,
        out: &mut [u8],
    ) -> Result<()> {
        if info.layout.layout_class == LayoutClass::Virtual {
            return self.read_virtual_raw_into_with_info(info, access, out);
        }

        let (element_size, total_elements_usize, total_bytes) = Self::raw_read_size(&info)?;
        if out.len() != total_bytes {
            return Err(Error::InvalidFormat(format!(
                "raw output buffer has {} bytes, expected {total_bytes}",
                out.len()
            )));
        }

        let mut guard = self.inner.lock();
        match info.layout.layout_class {
            LayoutClass::Compact => {
                let data =
                    info.layout.compact_data.as_ref().ok_or_else(|| {
                        Error::InvalidFormat("compact dataset missing data".into())
                    })?;
                if data.len() < total_bytes {
                    return Err(Error::InvalidFormat(format!(
                        "compact dataset data size {} is smaller than expected {total_bytes}",
                        data.len()
                    )));
                }
                out.copy_from_slice(&data[..total_bytes]);
                Ok(())
            }
            LayoutClass::Contiguous => {
                let addr = info.layout.contiguous_addr.ok_or_else(|| {
                    Error::InvalidFormat("contiguous dataset missing address".into())
                })?;
                let size = usize_from_u64(
                    info.layout
                        .contiguous_size
                        .unwrap_or(u64_from_usize(total_bytes, "contiguous dataset size")?),
                    "contiguous dataset size",
                )?;

                if crate::io::reader::is_undef_addr(addr) {
                    if info.external_file_list.is_some() {
                        let path = guard.path.clone();
                        Self::read_external_raw_data_into(
                            &mut guard.reader,
                            path.as_deref(),
                            access,
                            &info,
                            out,
                        )?;
                        return Ok(());
                    }
                    return Self::filled_data_into(total_elements_usize, element_size, &info, out);
                }

                if size < total_bytes {
                    return Err(Error::InvalidFormat(format!(
                        "contiguous dataset data size {size} is smaller than expected {total_bytes}"
                    )));
                }
                guard.reader.seek(addr)?;
                guard.reader.read_exact(out)
            }
            LayoutClass::Chunked => {
                Self::read_chunked_into(&mut guard.reader, &info, total_bytes, out)
            }
            LayoutClass::Virtual => unreachable!("virtual datasets are handled before raw sizing"),
        }
    }

    fn read_virtual_raw_with_info(
        &self,
        info: DatasetInfo,
        access: &DatasetAccess,
    ) -> Result<Vec<u8>> {
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
        Self::read_virtual_dataset(&heap_data, sizeof_size, path.as_deref(), &info, access)
    }

    fn read_virtual_raw_into_with_info(
        &self,
        info: &DatasetInfo,
        access: &DatasetAccess,
        out: &mut [u8],
    ) -> Result<()> {
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
        Self::read_virtual_dataset_into(&heap_data, sizeof_size, path.as_deref(), info, access, out)
    }

    pub(super) fn raw_read_size(info: &DatasetInfo) -> Result<(usize, usize, usize)> {
        let element_size = usize_from_u64(u64::from(info.datatype.size), "datatype size")?;
        if element_size == 0 {
            return Err(Error::InvalidFormat("zero-sized datatype".into()));
        }
        let total_elements =
            Self::dataspace_element_count(info.dataspace.space_type, &info.dataspace.dims)?;
        let total_elements_usize = usize_from_u64(total_elements, "dimension product")?;
        let total_bytes = total_elements_usize
            .checked_mul(element_size)
            .ok_or_else(|| Error::InvalidFormat("total data size overflow".into()))?;
        Ok((element_size, total_elements_usize, total_bytes))
    }

    pub(super) fn filled_data(
        total_elements: usize,
        element_size: usize,
        info: &DatasetInfo,
    ) -> Result<Vec<u8>> {
        let total_bytes = total_elements
            .checked_mul(element_size)
            .ok_or_else(|| Error::InvalidFormat("fill buffer size overflow".into()))?;
        let Some(fill) = &info.fill_value else {
            return Ok(vec![0u8; total_bytes]);
        };
        if fill.fill_time == FILL_TIME_NEVER {
            return Ok(vec![0u8; total_bytes]);
        }
        let Some(value) = fill.value.as_deref() else {
            return Ok(vec![0u8; total_bytes]);
        };
        if value.len() != element_size {
            return Err(Error::Unsupported(format!(
                "fill value size {} does not match element size {}",
                value.len(),
                element_size
            )));
        }

        let mut out = vec![0u8; total_bytes];
        Self::filled_data_into(total_elements, element_size, info, &mut out)?;
        Ok(out)
    }

    pub(super) fn filled_data_into(
        total_elements: usize,
        element_size: usize,
        info: &DatasetInfo,
        out: &mut [u8],
    ) -> Result<()> {
        let total_bytes = total_elements
            .checked_mul(element_size)
            .ok_or_else(|| Error::InvalidFormat("fill buffer size overflow".into()))?;
        if out.len() != total_bytes {
            return Err(Error::InvalidFormat(format!(
                "fill output buffer has {} bytes, expected {total_bytes}",
                out.len()
            )));
        }
        let Some(fill) = &info.fill_value else {
            out.fill(0);
            return Ok(());
        };
        if fill.fill_time == FILL_TIME_NEVER {
            out.fill(0);
            return Ok(());
        }
        let Some(value) = fill.value.as_deref() else {
            out.fill(0);
            return Ok(());
        };
        if value.len() != element_size {
            return Err(Error::Unsupported(format!(
                "fill value size {} does not match element size {}",
                value.len(),
                element_size
            )));
        }

        for chunk in out.chunks_exact_mut(element_size) {
            chunk.copy_from_slice(value);
        }
        Ok(())
    }

    /// Read all data as a typed Vec.
    ///
    /// This uses the crate's supported conversion table, not the full libhdf5
    /// conversion matrix. Unsupported or lossy HDF5 datatype conversions return
    /// an error rather than attempting C-library parity.
    pub fn read<T: crate::hl::types::H5Type>(&self) -> Result<Vec<T>> {
        self.read_vec_with_access(&DatasetAccess::new().with_virtual_view(VdsView::LastAvailable))
    }

    /// Read all data as a typed Vec, overriding the VDS view policy.
    pub fn read_with_vds_view<T: crate::hl::types::H5Type>(&self, view: VdsView) -> Result<Vec<T>> {
        self.read_vec_with_access(&DatasetAccess::new().with_virtual_view(view))
    }

    /// Read all data as a typed Vec, overriding dataset access properties.
    pub fn read_with_access<T: crate::hl::types::H5Type>(
        &self,
        access: &DatasetAccess,
    ) -> Result<Vec<T>> {
        self.read_vec_with_access(access)
    }

    fn read_vec_with_access<T: crate::hl::types::H5Type>(
        &self,
        access: &DatasetAccess,
    ) -> Result<Vec<T>> {
        let info = self.info()?;
        let conversion = crate::hl::conversion::ReadConversion::for_dataset::<T>(&info.datatype)?;
        if conversion.is_same_size_bytes() {
            let (total_elements, total_bytes) = self.typed_read_size_with_info(&info, access)?;
            let mut values = Vec::<T>::with_capacity(total_elements);
            let raw_out = unsafe {
                std::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut u8, total_bytes)
            };
            self.read_raw_into_with_info(&info, access, raw_out)?;
            conversion.convert_bytes_in_place(raw_out);
            unsafe {
                values.set_len(total_elements);
            }
            return Ok(values);
        }

        let raw = if info.layout.layout_class == LayoutClass::Virtual {
            self.read_virtual_raw_with_info(info, access)?
        } else {
            let (_, _, total_bytes) = Self::raw_read_size(&info)?;
            let mut raw = vec![0; total_bytes];
            self.read_raw_into_with_info(&info, access, &mut raw)?;
            raw
        };
        conversion.bytes_to_vec(raw)
    }

    fn typed_read_size_with_info(
        &self,
        info: &DatasetInfo,
        access: &DatasetAccess,
    ) -> Result<(usize, usize)> {
        if info.layout.layout_class == LayoutClass::Virtual {
            let element_size = usize_from_u64(u64::from(info.datatype.size), "datatype size")?;
            if element_size == 0 {
                return Err(Error::InvalidFormat("zero-sized datatype".into()));
            }
            let shape = self.virtual_shape_with_info(info, access)?;
            let total_elements = usize_from_u64(
                Self::dataspace_element_count(info.dataspace.space_type, &shape)?,
                "dimension product",
            )?;
            let total_bytes = total_elements
                .checked_mul(element_size)
                .ok_or_else(|| Error::InvalidFormat("total data size overflow".into()))?;
            Ok((total_elements, total_bytes))
        } else {
            let (_, total_elements, total_bytes) = Self::raw_read_size(info)?;
            Ok((total_elements, total_bytes))
        }
    }

    /// Read all data into caller-provided typed storage.
    pub fn read_into<T: crate::hl::types::H5Type>(&self, out: &mut [T]) -> Result<()> {
        self.read_into_with_vds_view(VdsView::LastAvailable, out)
    }

    /// Read all data into caller-provided typed storage, overriding the VDS view.
    pub fn read_into_with_vds_view<T: crate::hl::types::H5Type>(
        &self,
        view: VdsView,
        out: &mut [T],
    ) -> Result<()> {
        self.read_into_with_access(&DatasetAccess::new().with_virtual_view(view), out)
    }

    /// Read all data into caller-provided typed storage, overriding access properties.
    pub fn read_into_with_access<T: crate::hl::types::H5Type>(
        &self,
        access: &DatasetAccess,
        out: &mut [T],
    ) -> Result<()> {
        let info = self.info()?;
        let conversion = crate::hl::conversion::ReadConversion::for_dataset::<T>(&info.datatype)?;
        let total_elements = if info.layout.layout_class == LayoutClass::Virtual {
            let shape = self.virtual_shape_with_info(&info, access)?;
            usize_from_u64(
                Self::dataspace_element_count(info.dataspace.space_type, &shape)?,
                "dimension product",
            )?
        } else {
            let (_, total_elements, _) = Self::raw_read_size(&info)?;
            total_elements
        };
        if out.len() != total_elements {
            return Err(Error::InvalidFormat(format!(
                "typed output buffer has {} elements, expected {total_elements}",
                out.len()
            )));
        }

        if conversion.is_same_size_bytes() {
            let raw_out = crate::hl::types::slice_as_bytes_mut(out);
            self.read_raw_into_with_info(&info, access, raw_out)?;
            conversion.convert_bytes_in_place(raw_out);
            return Ok(());
        }

        let (_, _, total_bytes) = if info.layout.layout_class == LayoutClass::Virtual {
            let element_size = usize_from_u64(u64::from(info.datatype.size), "datatype size")?;
            let total_bytes = total_elements
                .checked_mul(element_size)
                .ok_or_else(|| Error::InvalidFormat("total data size overflow".into()))?;
            (element_size, total_elements, total_bytes)
        } else {
            Self::raw_read_size(&info)?
        };
        let mut raw = vec![0; total_bytes];
        self.read_raw_into_with_info(&info, access, &mut raw)?;
        conversion.bytes_into_slice(&raw, out)
    }

    /// Read a scalar value.
    pub fn read_scalar<T: crate::hl::types::H5Type>(&self) -> Result<T> {
        self.read_scalar_with_vds_view(VdsView::LastAvailable)
    }

    /// Read a scalar value, overriding the VDS view policy.
    pub fn read_scalar_with_vds_view<T: crate::hl::types::H5Type>(
        &self,
        view: VdsView,
    ) -> Result<T> {
        self.read_scalar_with_access(&DatasetAccess::new().with_virtual_view(view))
    }

    /// Read a scalar value, overriding dataset access properties.
    pub fn read_scalar_with_access<T: crate::hl::types::H5Type>(
        &self,
        access: &DatasetAccess,
    ) -> Result<T> {
        let info = self.info()?;
        let conversion = crate::hl::conversion::ReadConversion::for_dataset::<T>(&info.datatype)?;
        let total_elements = if info.layout.layout_class == LayoutClass::Virtual {
            let shape = self.virtual_shape_with_info(&info, access)?;
            usize_from_u64(
                Self::dataspace_element_count(info.dataspace.space_type, &shape)?,
                "dimension product",
            )?
        } else {
            let (_, total_elements, _) = Self::raw_read_size(&info)?;
            total_elements
        };
        if total_elements != 1 {
            return Err(Error::InvalidFormat(format!(
                "scalar read expected 1 element, found {total_elements}"
            )));
        }

        if conversion.is_same_size_bytes() {
            let mut value = std::mem::MaybeUninit::<T>::uninit();
            let raw_out = unsafe {
                std::slice::from_raw_parts_mut(value.as_mut_ptr() as *mut u8, T::type_size())
            };
            self.read_raw_into_with_info(&info, access, raw_out)?;
            conversion.convert_bytes_in_place(raw_out);
            return Ok(unsafe { value.assume_init() });
        }

        let raw = if info.layout.layout_class == LayoutClass::Virtual {
            self.read_virtual_raw_with_info(info, access)?
        } else {
            let (_, _, total_bytes) = Self::raw_read_size(&info)?;
            let mut raw = vec![0; total_bytes];
            self.read_raw_into_with_info(&info, access, &mut raw)?;
            raw
        };
        conversion.bytes_to_scalar_from_slice(&raw)
    }

    /// Read a scalar value into caller-provided storage.
    pub fn read_scalar_into<T: crate::hl::types::H5Type>(&self, out: &mut T) -> Result<()> {
        self.read_scalar_into_with_vds_view(VdsView::LastAvailable, out)
    }

    /// Read a scalar value into caller-provided storage, overriding the VDS view policy.
    pub fn read_scalar_into_with_vds_view<T: crate::hl::types::H5Type>(
        &self,
        view: VdsView,
        out: &mut T,
    ) -> Result<()> {
        self.read_scalar_into_with_access(&DatasetAccess::new().with_virtual_view(view), out)
    }

    /// Read a scalar value into caller-provided storage, overriding dataset access properties.
    pub fn read_scalar_into_with_access<T: crate::hl::types::H5Type>(
        &self,
        access: &DatasetAccess,
        out: &mut T,
    ) -> Result<()> {
        let info = self.info()?;
        let conversion = crate::hl::conversion::ReadConversion::for_dataset::<T>(&info.datatype)?;
        let total_elements = if info.layout.layout_class == LayoutClass::Virtual {
            let shape = self.virtual_shape_with_info(&info, access)?;
            usize_from_u64(
                Self::dataspace_element_count(info.dataspace.space_type, &shape)?,
                "dimension product",
            )?
        } else {
            let (_, total_elements, _) = Self::raw_read_size(&info)?;
            total_elements
        };
        if total_elements != 1 {
            return Err(Error::InvalidFormat(format!(
                "scalar read expected 1 element, found {total_elements}"
            )));
        }

        if conversion.is_same_size_bytes() {
            let output = std::slice::from_mut(out);
            let raw_out = crate::hl::types::slice_as_bytes_mut(output);
            self.read_raw_into_with_info(&info, access, raw_out)?;
            conversion.convert_bytes_in_place(raw_out);
            return Ok(());
        }

        let element_size = usize_from_u64(u64::from(info.datatype.size), "datatype size")?;
        let mut raw = vec![0u8; element_size];
        self.read_raw_into_with_info(&info, access, &mut raw)?;
        conversion.bytes_into_slice(&raw, std::slice::from_mut(out))
    }

    pub(super) fn dataspace_element_count(space_type: DataspaceType, dims: &[u64]) -> Result<u64> {
        if space_type == DataspaceType::Null {
            return Ok(0);
        }
        if dims.is_empty() {
            return Ok(1);
        }
        dims.iter().try_fold(1u64, |acc, &dim| {
            acc.checked_mul(dim)
                .ok_or_else(|| Error::InvalidFormat("dimension product overflow".into()))
        })
    }

    /// Read data as a 1D ndarray.
    pub fn read_1d<T: crate::hl::types::H5Type>(&self) -> Result<ndarray::Array1<T>> {
        let vec = self.read_vec_with_access::<T>(
            &DatasetAccess::new().with_virtual_view(VdsView::LastAvailable),
        )?;
        Ok(ndarray::Array1::from_vec(vec))
    }

    /// Read data as a 2D ndarray (row-major).
    pub fn read_2d<T: crate::hl::types::H5Type>(&self) -> Result<ndarray::Array2<T>> {
        let mut shape = Vec::new();
        self.shape_into(&mut shape)?;
        if shape.len() != 2 {
            return Err(Error::InvalidFormat(format!(
                "expected 2D dataset, got {}D",
                shape.len()
            )));
        }
        let vec = self.read_vec_with_access::<T>(
            &DatasetAccess::new().with_virtual_view(VdsView::LastAvailable),
        )?;
        let rows = usize_from_u64(shape[0], "ndarray row count")?;
        let cols = usize_from_u64(shape[1], "ndarray column count")?;
        ndarray::Array2::from_shape_vec((rows, cols), vec)
            .map_err(|e| Error::Other(format!("ndarray shape error: {e}")))
    }

    /// Read data as an N-dimensional ndarray (row-major).
    pub fn read_dyn<T: crate::hl::types::H5Type>(&self) -> Result<ndarray::ArrayD<T>> {
        let mut shape = Vec::new();
        self.shape_into(&mut shape)?;
        let dims: Vec<usize> = shape
            .iter()
            .map(|&dim| usize_from_u64(dim, "ndarray dimension"))
            .collect::<Result<Vec<_>>>()?;
        let vec = self.read_vec_with_access::<T>(
            &DatasetAccess::new().with_virtual_view(VdsView::LastAvailable),
        )?;
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&dims), vec)
            .map_err(|e| Error::Other(format!("ndarray shape error: {e}")))
    }
}
