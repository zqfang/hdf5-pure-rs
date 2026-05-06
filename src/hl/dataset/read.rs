use crate::error::{Error, Result};
use crate::format::messages::data_layout::LayoutClass;
use crate::format::messages::dataspace::DataspaceType;
use crate::format::messages::fill_value::FILL_TIME_NEVER;
use crate::format::object_header::ObjectHeader;

use super::{u64_from_usize, usize_from_u64, Dataset, DatasetAccess, DatasetInfo, VdsView};

impl Dataset {
    /// Read all raw data bytes from the dataset.
    pub fn read_raw(&self) -> Result<Vec<u8>> {
        self.read_raw_with_vds_view(VdsView::LastAvailable)
    }

    /// Read all raw bytes, overriding the VDS view policy.
    pub fn read_raw_with_vds_view(&self, view: VdsView) -> Result<Vec<u8>> {
        self.read_raw_with_access(&DatasetAccess::new().with_virtual_view(view))
    }

    /// Read all raw bytes, overriding dataset access properties.
    pub fn read_raw_with_access(&self, access: &DatasetAccess) -> Result<Vec<u8>> {
        let info = {
            let mut guard = self.inner.lock();
            let sizeof_addr = guard.superblock.sizeof_addr;
            let sizeof_size = guard.superblock.sizeof_size;
            let oh = ObjectHeader::read_at(&mut guard.reader, self.addr)?;
            Self::parse_info(&oh.messages, sizeof_addr, sizeof_size)?
        };

        self.read_raw_with_info(info, access)
    }

    pub(crate) fn read_raw_with_info(
        &self,
        info: DatasetInfo,
        access: &DatasetAccess,
    ) -> Result<Vec<u8>> {
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

        // Sanity limit: refuse to allocate more than 4GB in a single read
        const MAX_READ_BYTES: usize = 4 * 1024 * 1024 * 1024;
        if total_bytes > MAX_READ_BYTES {
            return Err(Error::InvalidFormat(format!(
                "dataset too large for single read: {total_bytes} bytes (max {MAX_READ_BYTES})"
            )));
        }

        let mut guard = self.inner.lock();
        match info.layout.layout_class {
            LayoutClass::Compact => {
                let data = info
                    .layout
                    .compact_data
                    .ok_or_else(|| Error::InvalidFormat("compact dataset missing data".into()))?;
                if data.len() < total_bytes {
                    return Err(Error::InvalidFormat(format!(
                        "compact dataset data size {} is smaller than expected {total_bytes}",
                        data.len()
                    )));
                }
                Ok(data[..total_bytes].to_vec())
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
                        return Self::read_external_raw_data(
                            &mut guard.reader,
                            path.as_deref(),
                            &info,
                            total_bytes,
                        );
                    }
                    return Self::filled_data(total_elements_usize, element_size, &info);
                }

                guard.reader.seek(addr)?;
                if size < total_bytes {
                    return Err(Error::InvalidFormat(format!(
                        "contiguous dataset data size {size} is smaller than expected {total_bytes}"
                    )));
                }
                let data = guard.reader.read_bytes(total_bytes)?;
                Ok(data)
            }
            LayoutClass::Chunked => Self::read_chunked(&mut guard.reader, &info, total_bytes),
            LayoutClass::Virtual => {
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
                Self::read_virtual_dataset(&heap_data, sizeof_size, path.as_deref(), &info, access)
            }
        }
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
        for chunk in out.chunks_exact_mut(element_size) {
            chunk.copy_from_slice(value);
        }
        Ok(out)
    }

    /// Read all data as a typed Vec.
    ///
    /// This uses the crate's supported conversion table, not the full libhdf5
    /// conversion matrix. Unsupported or lossy HDF5 datatype conversions return
    /// an error rather than attempting C-library parity.
    pub fn read<T: crate::hl::types::H5Type>(&self) -> Result<Vec<T>> {
        self.read_with_vds_view(VdsView::LastAvailable)
    }

    /// Read all data as a typed Vec, overriding the VDS view policy.
    pub fn read_with_vds_view<T: crate::hl::types::H5Type>(&self, view: VdsView) -> Result<Vec<T>> {
        self.read_with_access(&DatasetAccess::new().with_virtual_view(view))
    }

    /// Read all data as a typed Vec, overriding dataset access properties.
    pub fn read_with_access<T: crate::hl::types::H5Type>(
        &self,
        access: &DatasetAccess,
    ) -> Result<Vec<T>> {
        let info = self.info()?;
        let conversion = crate::hl::conversion::ReadConversion::for_dataset::<T>(&info.datatype)?;
        let raw = self.read_raw_with_access(access)?;
        conversion.bytes_to_vec(raw)
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
        let raw = self.read_raw_with_access(access)?;
        conversion.bytes_to_scalar(raw)
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
        let vec = self.read::<T>()?;
        Ok(ndarray::Array1::from_vec(vec))
    }

    /// Read data as a 2D ndarray (row-major).
    pub fn read_2d<T: crate::hl::types::H5Type>(&self) -> Result<ndarray::Array2<T>> {
        let shape = self.shape()?;
        if shape.len() != 2 {
            return Err(Error::InvalidFormat(format!(
                "expected 2D dataset, got {}D",
                shape.len()
            )));
        }
        let vec = self.read::<T>()?;
        let rows = usize_from_u64(shape[0], "ndarray row count")?;
        let cols = usize_from_u64(shape[1], "ndarray column count")?;
        ndarray::Array2::from_shape_vec((rows, cols), vec)
            .map_err(|e| Error::Other(format!("ndarray shape error: {e}")))
    }

    /// Read data as an N-dimensional ndarray (row-major).
    pub fn read_dyn<T: crate::hl::types::H5Type>(&self) -> Result<ndarray::ArrayD<T>> {
        let shape = self.shape()?;
        let dims: Vec<usize> = shape
            .iter()
            .map(|&dim| usize_from_u64(dim, "ndarray dimension"))
            .collect::<Result<Vec<_>>>()?;
        let vec = self.read::<T>()?;
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&dims), vec)
            .map_err(|e| Error::Other(format!("ndarray shape error: {e}")))
    }
}
