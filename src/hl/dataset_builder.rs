use std::{borrow::Cow, fs};

use crate::engine::writer::{
    ChunkWriteSpec, CompoundFieldSpec, DatasetSpec, DtypeSpec, FillValueSpec, HdfFileWriter,
};
use crate::error::{Error, Result};
use crate::hl::types::{slice_as_bytes, H5Type, TypeClass};

/// Builder for creating datasets with a fluent API.
pub struct DatasetBuilder<'a> {
    writer: &'a mut HdfFileWriter<fs::File>,
    parent: String,
    name: String,
    shape: Option<Vec<u64>>,
    max_shape: Option<Vec<u64>>,
    resizable: bool,
    chunk_dims: Option<Vec<u64>>,
    deflate_level: Option<u32>,
    shuffle: bool,
    fletcher32: bool,
    compact: bool,
    fill_value: Option<Vec<u8>>,
    vlen_utf8_fill_value: Option<String>,
    alloc_time: u8,
    fill_time: u8,
    attrs: Vec<OwnedBuilderAttr>,
}

struct OwnedBuilderAttr {
    name: String,
    shape: Vec<u64>,
    dtype: DtypeSpec,
    data: Vec<u8>,
}

impl OwnedBuilderAttr {
    fn as_attr_spec(&self) -> crate::engine::writer::AttrSpec<'_> {
        crate::engine::writer::AttrSpec {
            name: &self.name,
            shape: &self.shape,
            dtype: self.dtype.clone(),
            data: &self.data,
        }
    }
}

fn collect_attr_specs(attrs: &[OwnedBuilderAttr]) -> Vec<crate::engine::writer::AttrSpec<'_>> {
    let mut out = Vec::with_capacity(attrs.len());
    out.extend(attrs.iter().map(OwnedBuilderAttr::as_attr_spec));
    out
}

fn with_attr_specs<R, F>(attrs: &[OwnedBuilderAttr], f: F) -> Result<R>
where
    F: FnOnce(&[crate::engine::writer::AttrSpec<'_>]) -> Result<R>,
{
    if attrs.is_empty() {
        f(&[])
    } else {
        let specs = collect_attr_specs(attrs);
        f(&specs)
    }
}

impl<'a> DatasetBuilder<'a> {
    pub(crate) fn new(writer: &'a mut HdfFileWriter<fs::File>, parent: &str, name: &str) -> Self {
        Self {
            writer,
            parent: parent.to_string(),
            name: name.to_string(),
            shape: None,
            max_shape: None,
            resizable: false,
            chunk_dims: None,
            deflate_level: None,
            shuffle: false,
            fletcher32: false,
            compact: false,
            fill_value: None,
            vlen_utf8_fill_value: None,
            alloc_time: 1,
            fill_time: 2,
            attrs: Vec::new(),
        }
    }

    /// Set the dataset shape.
    pub fn shape(mut self, dims: &[u64]) -> Self {
        self.shape = Some(dims.to_vec());
        self
    }

    /// Set chunk dimensions (enables chunked storage).
    pub fn chunk(mut self, dims: &[u64]) -> Self {
        self.chunk_dims = Some(dims.to_vec());
        self
    }

    /// Enable deflate (gzip) compression at the given level (0-9).
    pub fn deflate(mut self, level: u32) -> Self {
        self.deflate_level = Some(level);
        self
    }

    /// Enable byte shuffle filter (should be used with compression).
    pub fn shuffle(mut self) -> Self {
        self.shuffle = true;
        self
    }

    /// Enable Fletcher32 chunk checksums.
    pub fn fletcher32(mut self) -> Self {
        self.fletcher32 = true;
        self
    }

    /// Make the dataset resizable (unlimited max dimensions).
    /// Requires chunked storage.
    pub fn resizable(mut self) -> Self {
        self.resizable = true;
        self
    }

    /// Set explicit maximum dimensions.
    pub fn max_shape(mut self, dims: &[u64]) -> Self {
        self.max_shape = Some(dims.to_vec());
        self.resizable = false;
        self
    }

    /// Use compact storage (data embedded in object header, for small datasets).
    pub fn compact(mut self) -> Self {
        self.compact = true;
        self
    }

    /// Set raw HDF5 allocation-time and fill-write-time properties.
    pub fn fill_properties(mut self, alloc_time: u8, fill_time: u8) -> Self {
        self.alloc_time = alloc_time;
        self.fill_time = fill_time;
        self
    }

    /// Set a scalar fill value for missing or newly allocated dataset storage.
    pub fn fill_value<T: H5Type>(mut self, value: T) -> Self {
        self.fill_value = Some(slice_as_bytes(std::slice::from_ref(&value)).to_vec());
        self
    }

    /// Set a scalar variable-length UTF-8 string fill value.
    pub fn vlen_utf8_fill_value(mut self, value: &str) -> Self {
        self.vlen_utf8_fill_value = Some(value.to_string());
        self
    }

    /// Add a scalar attribute to the dataset being created.
    pub fn attr<T: H5Type>(mut self, name: &str, value: T) -> Result<Self> {
        let dtype = dtype_for_type::<T>()?;
        self.push_attr(OwnedBuilderAttr {
            name: name.to_string(),
            shape: Vec::new(),
            dtype,
            data: slice_as_bytes(std::slice::from_ref(&value)).to_vec(),
        })?;
        Ok(self)
    }

    /// Add a one-dimensional array attribute to the dataset being created.
    pub fn attr_array<T: H5Type>(mut self, name: &str, values: &[T]) -> Result<Self> {
        let dtype = dtype_for_type::<T>()?;
        let byte_len = values
            .len()
            .checked_mul(T::type_size())
            .ok_or_else(|| Error::InvalidFormat("attribute byte size overflow".into()))?;
        let data = slice_as_bytes(values);
        debug_assert_eq!(data.len(), byte_len);
        self.push_attr(OwnedBuilderAttr {
            name: name.to_string(),
            shape: vec![usize_to_u64(values.len(), "attribute element count")?],
            dtype,
            data: data.to_vec(),
        })?;
        Ok(self)
    }

    /// Add a fixed-length ASCII string attribute to the dataset being created.
    pub fn fixed_ascii_attr(mut self, name: &str, value: &str, len: usize) -> Result<Self> {
        let (dtype, data) = fixed_string_attr(&[value], len, false)?;
        self.push_attr(OwnedBuilderAttr {
            name: name.to_string(),
            shape: Vec::new(),
            dtype,
            data,
        })?;
        Ok(self)
    }

    /// Add a fixed-length UTF-8 string attribute to the dataset being created.
    pub fn fixed_utf8_attr(mut self, name: &str, value: &str, len: usize) -> Result<Self> {
        let (dtype, data) = fixed_string_attr(&[value], len, true)?;
        self.push_attr(OwnedBuilderAttr {
            name: name.to_string(),
            shape: Vec::new(),
            dtype,
            data,
        })?;
        Ok(self)
    }

    /// Add a one-dimensional fixed-length ASCII string array attribute.
    pub fn fixed_ascii_attr_array(
        mut self,
        name: &str,
        values: &[&str],
        len: usize,
    ) -> Result<Self> {
        let (dtype, data) = fixed_string_attr(values, len, false)?;
        self.push_attr(OwnedBuilderAttr {
            name: name.to_string(),
            shape: vec![usize_to_u64(values.len(), "attribute element count")?],
            dtype,
            data,
        })?;
        Ok(self)
    }

    /// Add a one-dimensional fixed-length UTF-8 string array attribute.
    pub fn fixed_utf8_attr_array(
        mut self,
        name: &str,
        values: &[&str],
        len: usize,
    ) -> Result<Self> {
        let (dtype, data) = fixed_string_attr(values, len, true)?;
        self.push_attr(OwnedBuilderAttr {
            name: name.to_string(),
            shape: vec![usize_to_u64(values.len(), "attribute element count")?],
            dtype,
            data,
        })?;
        Ok(self)
    }

    /// Write data and create the dataset. Infers shape from data length if not set.
    pub fn write<T: H5Type>(self, data: &[T]) -> Result<()> {
        let dtype = dtype_for_type::<T>()?;
        let fill = Self::fill_spec(
            self.fill_value.as_deref(),
            dtype.size() as usize,
            self.alloc_time,
            self.fill_time,
        )?;
        let shape = match self.shape.as_deref() {
            Some(shape) => Cow::Borrowed(shape),
            None => Cow::Owned(vec![usize_to_u64(data.len(), "dataset element count")?]),
        };
        Self::validate_element_count(shape.as_ref(), data.len())?;
        let max_shape =
            Self::effective_max_shape(self.resizable, self.max_shape.as_deref(), shape.as_ref())?;

        let data_bytes = slice_as_bytes(data);

        let spec = DatasetSpec {
            name: &self.name,
            shape: shape.as_ref(),
            max_shape: max_shape.as_ref().map(|shape| shape.as_ref()),
            dtype,
            data: data_bytes,
        };
        if self.compact {
            self.validate_compact_options()?;
            if self.attrs.is_empty() {
                self.writer
                    .create_compact_dataset_with_fill(&self.parent, &spec, fill)?;
            } else {
                let attrs = collect_attr_specs(&self.attrs);
                self.writer.create_compact_dataset_with_attrs_and_fill(
                    &self.parent,
                    &spec,
                    &attrs,
                    fill,
                )?;
            }
        } else if self.chunk_dims.is_some()
            || self.deflate_level.is_some()
            || self.shuffle
            || self.fletcher32
        {
            let chunk_dims = self.chunk_dims.as_deref().unwrap_or_else(|| shape.as_ref());
            with_attr_specs(&self.attrs, |attrs| {
                self.writer.create_chunked_dataset_with_attrs_and_fill(
                    &self.parent,
                    &spec,
                    chunk_dims,
                    self.deflate_level,
                    self.shuffle,
                    self.fletcher32,
                    fill,
                    attrs,
                )
            })?;
        } else {
            if self.attrs.is_empty() {
                self.writer
                    .create_dataset_with_fill(&self.parent, &spec, fill)?;
            } else {
                let attrs = collect_attr_specs(&self.attrs);
                self.writer.create_dataset_with_attrs_and_fill(
                    &self.parent,
                    &spec,
                    &attrs,
                    fill,
                )?;
            }
        }

        Ok(())
    }

    /// Create a fill-only chunked dataset without allocating chunk payloads.
    ///
    /// The dataset shape must be set explicitly. Missing chunks read back as
    /// the configured fill value, or zero bytes when no fill value was set.
    pub fn write_fill<T: H5Type>(self) -> Result<()> {
        if self.compact {
            return Err(Error::Unsupported(
                "fill-only dataset writer does not support compact storage".into(),
            ));
        }
        if self.chunk_dims.is_none()
            && self.deflate_level.is_none()
            && !self.shuffle
            && !self.fletcher32
        {
            return Err(Error::Unsupported(
                "fill-only dataset writer currently supports chunked storage only".into(),
            ));
        }
        let dtype = dtype_for_type::<T>()?;
        let fill = Self::fill_spec(
            self.fill_value.as_deref(),
            dtype.size() as usize,
            self.alloc_time,
            self.fill_time,
        )?;
        let shape = self.shape.as_deref().ok_or_else(|| {
            Error::InvalidFormat("fill-only dataset writer requires an explicit shape".into())
        })?;
        let max_shape =
            Self::effective_max_shape(self.resizable, self.max_shape.as_deref(), shape)?;
        let chunk_dims = self.chunk_dims.as_deref().unwrap_or(shape);
        let spec = DatasetSpec {
            name: &self.name,
            shape,
            max_shape: max_shape.as_ref().map(|shape| shape.as_ref()),
            dtype,
            data: &[],
        };
        with_attr_specs(&self.attrs, |attrs| {
            self.writer
                .create_sparse_chunked_dataset_with_attrs_and_fill(
                    &self.parent,
                    &spec,
                    chunk_dims,
                    self.deflate_level,
                    self.shuffle,
                    self.fletcher32,
                    fill,
                    attrs,
                )
        })?;
        Ok(())
    }

    /// Create a sparse chunked dataset from caller-supplied full chunks.
    ///
    /// Only the listed chunks are written. Unlisted chunks read as the
    /// configured fill value, or zero bytes when no fill value was set.
    pub fn write_chunks<'b, T, I, C>(self, chunks: I) -> Result<()>
    where
        T: H5Type + 'b,
        I: IntoIterator<Item = (C, &'b [T])>,
        C: AsRef<[u64]>,
    {
        if self.compact {
            return Err(Error::Unsupported(
                "chunk-list dataset writer does not support compact storage".into(),
            ));
        }
        let dtype = dtype_for_type::<T>()?;
        let fill = Self::fill_spec(
            self.fill_value.as_deref(),
            dtype.size() as usize,
            self.alloc_time,
            self.fill_time,
        )?;
        let shape = self.shape.as_deref().ok_or_else(|| {
            Error::InvalidFormat("chunk-list dataset writer requires an explicit shape".into())
        })?;
        let max_shape =
            Self::effective_max_shape(self.resizable, self.max_shape.as_deref(), shape)?;
        let chunk_dims = self.chunk_dims.as_deref().ok_or_else(|| {
            Error::InvalidFormat(
                "chunk-list dataset writer requires explicit chunk dimensions".into(),
            )
        })?;

        let owned_chunks: Vec<(Vec<u64>, &'b [T])> = chunks
            .into_iter()
            .map(|(coords, data)| (coords.as_ref().to_vec(), data))
            .collect();
        let byte_chunks: Vec<ChunkWriteSpec<'_>> = owned_chunks
            .iter()
            .map(|(coords, data)| ChunkWriteSpec {
                coords,
                data: slice_as_bytes(*data),
            })
            .collect();

        let spec = DatasetSpec {
            name: &self.name,
            shape,
            max_shape: max_shape.as_ref().map(|shape| shape.as_ref()),
            dtype,
            data: &[],
        };
        with_attr_specs(&self.attrs, |attrs| {
            self.writer
                .create_chunked_dataset_from_chunks_with_attrs_and_fill(
                    &self.parent,
                    &spec,
                    chunk_dims,
                    &byte_chunks,
                    self.deflate_level,
                    self.shuffle,
                    self.fletcher32,
                    fill,
                    attrs,
                )
        })?;
        Ok(())
    }

    /// Write raw element bytes with an explicit HDF5 datatype.
    ///
    /// This exposes writer datatypes that cannot be inferred from a Rust
    /// primitive type, such as enum, opaque, array, and nested compound
    /// datatypes. If no shape was set, a one-dimensional shape is inferred
    /// from `data.len() / dtype.size()`.
    pub fn write_raw_with_dtype(self, dtype: DtypeSpec, data: &[u8]) -> Result<()> {
        let dtype_size = dtype.size() as usize;
        if dtype_size == 0 {
            return Err(Error::InvalidFormat(
                "dataset datatype size must be nonzero".into(),
            ));
        }
        if data.len() % dtype_size != 0 {
            return Err(Error::InvalidFormat(format!(
                "raw dataset byte length {} is not a multiple of datatype size {dtype_size}",
                data.len()
            )));
        }

        let fill = Self::fill_spec(
            self.fill_value.as_deref(),
            dtype_size,
            self.alloc_time,
            self.fill_time,
        )?;
        let shape = match self.shape.as_deref() {
            Some(shape) => Cow::Borrowed(shape),
            None => vec![usize_to_u64(
                data.len() / dtype_size,
                "dataset element count",
            )?]
            .into(),
        };
        let max_shape =
            Self::effective_max_shape(self.resizable, self.max_shape.as_deref(), shape.as_ref())?;
        let expected_count = shape_element_count(shape.as_ref())?;
        let expected_bytes = usize::try_from(expected_count)
            .map_err(|_| Error::InvalidFormat("dataset element count exceeds usize".into()))?
            .checked_mul(dtype_size)
            .ok_or_else(|| Error::InvalidFormat("dataset byte size overflow".into()))?;
        if expected_bytes != data.len() {
            return Err(Error::InvalidFormat(format!(
                "raw dataset byte length {} does not match shape element count {expected_count} * datatype size {dtype_size}",
                data.len()
            )));
        }

        let spec = DatasetSpec {
            name: &self.name,
            shape: shape.as_ref(),
            max_shape: max_shape.as_ref().map(|shape| shape.as_ref()),
            dtype,
            data,
        };
        if self.compact {
            self.validate_compact_options()?;
            if self.attrs.is_empty() {
                self.writer
                    .create_compact_dataset_with_fill(&self.parent, &spec, fill)?;
            } else {
                let attrs = collect_attr_specs(&self.attrs);
                self.writer.create_compact_dataset_with_attrs_and_fill(
                    &self.parent,
                    &spec,
                    &attrs,
                    fill,
                )?;
            }
        } else if self.chunk_dims.is_some()
            || self.deflate_level.is_some()
            || self.shuffle
            || self.fletcher32
        {
            let chunk_dims = self.chunk_dims.as_deref().unwrap_or_else(|| shape.as_ref());
            with_attr_specs(&self.attrs, |attrs| {
                self.writer.create_chunked_dataset_with_attrs_and_fill(
                    &self.parent,
                    &spec,
                    chunk_dims,
                    self.deflate_level,
                    self.shuffle,
                    self.fletcher32,
                    fill,
                    attrs,
                )
            })?;
        } else if self.attrs.is_empty() {
            self.writer
                .create_dataset_with_fill(&self.parent, &spec, fill)?;
        } else {
            let attrs = collect_attr_specs(&self.attrs);
            self.writer
                .create_dataset_with_attrs_and_fill(&self.parent, &spec, &attrs, fill)?;
        }

        Ok(())
    }

    /// Write a scalar value.
    pub fn write_scalar<T: H5Type>(self, value: T) -> Result<()> {
        if self.chunk_dims.is_some()
            || self.deflate_level.is_some()
            || self.shuffle
            || self.fletcher32
        {
            return Err(Error::Unsupported(
                "scalar dataset writer does not support chunked storage or filters".into(),
            ));
        }
        if self.resizable || self.max_shape.is_some() {
            return Err(Error::InvalidFormat(
                "scalar dataset writer does not support max dimensions".into(),
            ));
        }
        let dtype = dtype_for_type::<T>()?;
        let fill = Self::fill_spec(
            self.fill_value.as_deref(),
            dtype.size() as usize,
            self.alloc_time,
            self.fill_time,
        )?;
        let data_bytes = slice_as_bytes(std::slice::from_ref(&value));

        let spec = DatasetSpec {
            name: &self.name,
            shape: &[],
            max_shape: None,
            dtype,
            data: data_bytes,
        };

        if self.compact {
            self.validate_compact_options()?;
            if self.attrs.is_empty() {
                self.writer
                    .create_compact_dataset_with_fill(&self.parent, &spec, fill)?;
            } else {
                let attrs = collect_attr_specs(&self.attrs);
                self.writer.create_compact_dataset_with_attrs_and_fill(
                    &self.parent,
                    &spec,
                    &attrs,
                    fill,
                )?;
            }
        } else if self.attrs.is_empty() {
            self.writer
                .create_dataset_with_fill(&self.parent, &spec, fill)?;
        } else {
            let attrs = collect_attr_specs(&self.attrs);
            self.writer
                .create_dataset_with_attrs_and_fill(&self.parent, &spec, &attrs, fill)?;
        }
        Ok(())
    }

    /// Write fixed-length ASCII strings. Strings longer than `len` bytes are rejected.
    pub fn write_fixed_ascii_strings(self, data: &[&str], len: usize) -> Result<()> {
        self.write_fixed_strings(data, len, false)
    }

    /// Write fixed-length UTF-8 strings. Strings longer than `len` bytes are rejected.
    pub fn write_fixed_utf8_strings(self, data: &[&str], len: usize) -> Result<()> {
        self.write_fixed_strings(data, len, true)
    }

    /// Write variable-length UTF-8 strings using HDF5 global heap storage.
    pub fn write_vlen_utf8_strings(self, data: &[&str]) -> Result<()> {
        if self.compact {
            return Err(Error::Unsupported(
                "variable-length string writer does not support compact storage".into(),
            ));
        }
        let fill = Self::fill_spec(
            self.fill_value.as_deref(),
            DtypeSpec::VarLenUtf8String.size() as usize,
            self.alloc_time,
            self.fill_time,
        )?;
        if self.vlen_utf8_fill_value.is_some()
            && fill.as_ref().and_then(|fill| fill.value).is_some()
        {
            return Err(Error::InvalidFormat(
                "vlen UTF-8 fill value conflicts with raw fill-value bytes".into(),
            ));
        }
        let shape = match self.shape.as_deref() {
            Some(shape) => Cow::Borrowed(shape),
            None => Cow::Owned(vec![usize_to_u64(data.len(), "dataset element count")?]),
        };
        let max_shape =
            Self::effective_max_shape(self.resizable, self.max_shape.as_deref(), shape.as_ref())?;
        let vlen_fill = self.vlen_utf8_fill_value.as_deref();
        with_attr_specs(&self.attrs, |attrs| {
            if self.chunk_dims.is_some()
                || self.deflate_level.is_some()
                || self.shuffle
                || self.fletcher32
            {
                let chunk_dims = self.chunk_dims.as_deref().unwrap_or_else(|| shape.as_ref());
                self.writer
                    .create_chunked_vlen_utf8_string_dataset_with_attrs_and_vlen_fill(
                        &self.parent,
                        &self.name,
                        shape.as_ref(),
                        data,
                        max_shape.as_ref().map(|shape| shape.as_ref()),
                        chunk_dims,
                        self.deflate_level,
                        self.shuffle,
                        self.fletcher32,
                        fill,
                        vlen_fill,
                        attrs,
                    )
            } else {
                self.writer
                    .create_vlen_utf8_string_dataset_with_attrs_and_vlen_fill(
                        &self.parent,
                        &self.name,
                        shape.as_ref(),
                        data,
                        max_shape.as_ref().map(|shape| shape.as_ref()),
                        fill,
                        vlen_fill,
                        attrs,
                    )
            }
        })?;
        Ok(())
    }

    fn write_fixed_strings(self, data: &[&str], len: usize, utf8: bool) -> Result<()> {
        let len_u32 = usize_to_u32(len, "fixed string length")?;
        let dtype = if utf8 {
            DtypeSpec::FixedUtf8String {
                len: len_u32,
                padding: 1,
            }
        } else {
            DtypeSpec::FixedAsciiString {
                len: len_u32,
                padding: 1,
            }
        };
        let fill = Self::fill_spec(
            self.fill_value.as_deref(),
            dtype.size() as usize,
            self.alloc_time,
            self.fill_time,
        )?;
        let shape = match self.shape.as_deref() {
            Some(shape) => Cow::Borrowed(shape),
            None => Cow::Owned(vec![usize_to_u64(data.len(), "dataset element count")?]),
        };
        let max_shape =
            Self::effective_max_shape(self.resizable, self.max_shape.as_deref(), shape.as_ref())?;
        let expected_count = shape_element_count(shape.as_ref())?;
        let actual_count = usize_to_u64(data.len(), "dataset element count")?;
        if expected_count != actual_count {
            return Err(Error::InvalidFormat(format!(
                "fixed string data length {} does not match dataset shape element count {expected_count}",
                data.len()
            )));
        }

        let data_len = data.len().checked_mul(len).ok_or_else(|| {
            Error::InvalidFormat("fixed string dataset payload size overflow".into())
        })?;
        for value in data {
            let bytes = value.as_bytes();
            if bytes.len() > len {
                return Err(Error::InvalidFormat(format!(
                    "fixed string value has {} bytes, maximum is {len}",
                    bytes.len()
                )));
            }
        }
        let mut data_bytes = vec![0; data_len];
        for (slot, value) in data_bytes.chunks_exact_mut(len).zip(data.iter()) {
            let bytes = value.as_bytes();
            slot[..bytes.len()].copy_from_slice(bytes);
        }

        let spec = DatasetSpec {
            name: &self.name,
            shape: shape.as_ref(),
            max_shape: max_shape.as_ref().map(|shape| shape.as_ref()),
            dtype,
            data: &data_bytes,
        };
        if self.compact {
            self.validate_compact_options()?;
            if self.attrs.is_empty() {
                self.writer
                    .create_compact_dataset_with_fill(&self.parent, &spec, fill)?;
            } else {
                let attrs = collect_attr_specs(&self.attrs);
                self.writer.create_compact_dataset_with_attrs_and_fill(
                    &self.parent,
                    &spec,
                    &attrs,
                    fill,
                )?;
            }
        } else if self.chunk_dims.is_some()
            || self.deflate_level.is_some()
            || self.shuffle
            || self.fletcher32
        {
            let chunk_dims = self.chunk_dims.as_deref().unwrap_or_else(|| shape.as_ref());
            with_attr_specs(&self.attrs, |attrs| {
                self.writer.create_chunked_dataset_with_attrs_and_fill(
                    &self.parent,
                    &spec,
                    chunk_dims,
                    self.deflate_level,
                    self.shuffle,
                    self.fletcher32,
                    fill,
                    attrs,
                )
            })?;
        } else if self.attrs.is_empty() {
            self.writer
                .create_dataset_with_fill(&self.parent, &spec, fill)?;
        } else {
            let attrs = collect_attr_specs(&self.attrs);
            self.writer
                .create_dataset_with_attrs_and_fill(&self.parent, &spec, &attrs, fill)?;
        }
        Ok(())
    }

    fn push_attr(&mut self, attr: OwnedBuilderAttr) -> Result<()> {
        if self.attrs.iter().any(|existing| existing.name == attr.name) {
            return Err(Error::InvalidFormat(format!(
                "attribute '{}' already exists",
                attr.name
            )));
        }
        self.attrs.push(attr);
        Ok(())
    }

    fn fill_spec(
        value: Option<&[u8]>,
        dtype_size: usize,
        alloc_time: u8,
        fill_time: u8,
    ) -> Result<Option<FillValueSpec<'_>>> {
        if let Some(value) = value {
            if value.len() != dtype_size {
                return Err(Error::InvalidFormat(format!(
                    "fill value has {} bytes, expected {} for dataset datatype",
                    value.len(),
                    dtype_size
                )));
            }
            Ok(Some(FillValueSpec::with_value(
                alloc_time, fill_time, value,
            )))
        } else if alloc_time != 1 || fill_time != 2 {
            Ok(Some(FillValueSpec::undefined(alloc_time, fill_time)))
        } else {
            Ok(None)
        }
    }

    fn effective_max_shape<'b>(
        resizable: bool,
        max_shape: Option<&'b [u64]>,
        shape: &[u64],
    ) -> Result<Option<Cow<'b, [u64]>>> {
        let max_shape = if resizable {
            Some(Cow::Owned(vec![u64::MAX; shape.len()]))
        } else {
            max_shape.map(Cow::Borrowed)
        };
        let Some(max_shape) = max_shape else {
            return Ok(None);
        };
        if max_shape.len() != shape.len() {
            return Err(Error::InvalidFormat(format!(
                "max shape rank {} does not match dataset rank {}",
                max_shape.len(),
                shape.len()
            )));
        }
        for (idx, (&dim, &max_dim)) in shape.iter().zip(max_shape.iter()).enumerate() {
            if max_dim != u64::MAX && dim > max_dim {
                return Err(Error::InvalidFormat(format!(
                    "dataset dimension {idx} size {dim} exceeds max dimension {max_dim}"
                )));
            }
        }
        Ok(Some(max_shape))
    }

    fn validate_compact_options(&self) -> Result<()> {
        if self.chunk_dims.is_some()
            || self.deflate_level.is_some()
            || self.shuffle
            || self.fletcher32
        {
            return Err(Error::Unsupported(
                "compact dataset writer does not support chunked storage or filters".into(),
            ));
        }
        if self.resizable || self.max_shape.is_some() {
            return Err(Error::InvalidFormat(
                "compact dataset writer does not support max dimensions".into(),
            ));
        }
        Ok(())
    }

    fn validate_element_count(shape: &[u64], data_len: usize) -> Result<()> {
        let expected_count = shape_element_count(shape)?;
        let actual_count = u64::try_from(data_len)
            .map_err(|_| Error::InvalidFormat("dataset element count exceeds u64".into()))?;
        if expected_count != actual_count {
            return Err(Error::InvalidFormat(format!(
                "dataset data length {actual_count} does not match shape element count {expected_count}"
            )));
        }
        Ok(())
    }
}

/// Map a Rust type to DtypeSpec.
pub(crate) fn dtype_for_type<T: H5Type>() -> Result<DtypeSpec> {
    let size = T::type_size();
    // Use TypeId to determine the exact type
    use std::any::TypeId;
    let id = TypeId::of::<T>();

    if id == TypeId::of::<f64>() {
        Ok(DtypeSpec::F64)
    } else if id == TypeId::of::<f32>() {
        Ok(DtypeSpec::F32)
    } else if id == TypeId::of::<i128>() {
        Ok(DtypeSpec::I128)
    } else if id == TypeId::of::<i64>() {
        Ok(DtypeSpec::I64)
    } else if id == TypeId::of::<i32>() {
        Ok(DtypeSpec::I32)
    } else if id == TypeId::of::<i16>() {
        Ok(DtypeSpec::I16)
    } else if id == TypeId::of::<i8>() {
        Ok(DtypeSpec::I8)
    } else if id == TypeId::of::<u128>() {
        Ok(DtypeSpec::U128)
    } else if id == TypeId::of::<u64>() {
        Ok(DtypeSpec::U64)
    } else if id == TypeId::of::<u32>() {
        Ok(DtypeSpec::U32)
    } else if id == TypeId::of::<u16>() {
        Ok(DtypeSpec::U16)
    } else if id == TypeId::of::<u8>() {
        Ok(DtypeSpec::U8)
    } else {
        let mut fields = Vec::new();
        if T::compound_fields_into(&mut fields).is_none() {
            return Err(Error::Unsupported(format!(
                "unsupported type with size {size}"
            )));
        }
        let mut out = Vec::with_capacity(fields.len());
        for field in fields {
            let dtype = match field.type_class {
                TypeClass::Integer { signed: true } => match field.size {
                    1 => DtypeSpec::I8,
                    2 => DtypeSpec::I16,
                    4 => DtypeSpec::I32,
                    8 => DtypeSpec::I64,
                    16 => DtypeSpec::I128,
                    other => {
                        return Err(Error::Unsupported(format!(
                            "unsupported signed compound field size {other}"
                        )))
                    }
                },
                TypeClass::Integer { signed: false } => match field.size {
                    1 => DtypeSpec::U8,
                    2 => DtypeSpec::U16,
                    4 => DtypeSpec::U32,
                    8 => DtypeSpec::U64,
                    16 => DtypeSpec::U128,
                    other => {
                        return Err(Error::Unsupported(format!(
                            "unsupported unsigned compound field size {other}"
                        )))
                    }
                },
                TypeClass::Float => match field.size {
                    4 => DtypeSpec::F32,
                    8 => DtypeSpec::F64,
                    other => {
                        return Err(Error::Unsupported(format!(
                            "unsupported floating compound field size {other}"
                        )))
                    }
                },
                TypeClass::Compound => {
                    return Err(Error::Unsupported(
                        "nested compound writer type descriptors are not supported".into(),
                    ))
                }
            };
            out.push(CompoundFieldSpec {
                name: field.name,
                offset: usize_to_u32(field.offset, "compound field offset")?,
                dtype,
            });
        }
        Ok(DtypeSpec::Compound {
            size: usize_to_u32(T::type_size(), "compound type size")?,
            fields: out,
        })
    }
}

fn fixed_string_attr(values: &[&str], len: usize, utf8: bool) -> Result<(DtypeSpec, Vec<u8>)> {
    let len_u32 = usize_to_u32(len, "fixed string length")?;
    let dtype = if utf8 {
        DtypeSpec::FixedUtf8String {
            len: len_u32,
            padding: 1,
        }
    } else {
        DtypeSpec::FixedAsciiString {
            len: len_u32,
            padding: 1,
        }
    };
    let data_len = values.len().checked_mul(len).ok_or_else(|| {
        Error::InvalidFormat("fixed string attribute payload size overflow".into())
    })?;
    for value in values {
        let bytes = value.as_bytes();
        if bytes.len() > len {
            return Err(Error::InvalidFormat(format!(
                "fixed string attribute has {} bytes, maximum is {len}",
                bytes.len()
            )));
        }
    }
    let mut data = vec![0; data_len];
    for (slot, value) in data.chunks_exact_mut(len).zip(values.iter()) {
        let bytes = value.as_bytes();
        slot[..bytes.len()].copy_from_slice(bytes);
    }
    Ok((dtype, data))
}

fn usize_to_u64(value: usize, context: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| Error::InvalidFormat(format!("{context} exceeds u64")))
}

fn usize_to_u32(value: usize, context: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::InvalidFormat(format!("{context} exceeds u32")))
}

fn shape_element_count(shape: &[u64]) -> Result<u64> {
    if shape.is_empty() {
        return Ok(1);
    }
    shape.iter().try_fold(1u64, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| Error::InvalidFormat("dataset shape element count overflow".into()))
    })
}

#[cfg(test)]
mod tests {
    use super::shape_element_count;

    #[test]
    fn shape_element_count_rejects_overflow() {
        let err = shape_element_count(&[u64::MAX, 2]).unwrap_err();
        assert!(
            err.to_string()
                .contains("dataset shape element count overflow"),
            "unexpected error: {err}"
        );
    }
}
