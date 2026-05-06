use std::fs;
use std::path::{Path, PathBuf};

use crate::engine::writer::{AttrSpec, HdfFileWriter};
use crate::error::{Error, Result};
use crate::hl::dataset_builder::DatasetBuilder;
use crate::hl::types::H5Type;

/// A writable HDF5 file under construction.
///
/// Use the builder methods to create groups, datasets, and attributes,
/// then call `flush()` or `close()` to finalize and write to disk.
pub struct WritableFile {
    writer: HdfFileWriter<fs::File>,
    path: PathBuf,
    #[allow(dead_code)]
    current_group: String,
}

impl WritableFile {
    /// Create a new HDF5 file (truncating if it exists).
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let f = fs::File::create(&path)?;
        let mut writer = HdfFileWriter::new(f);
        writer.begin()?;
        writer.create_root_group()?;

        Ok(Self {
            writer,
            path,
            current_group: "/".to_string(),
        })
    }

    /// Create a new HDF5 file, failing if it already exists.
    pub fn create_excl<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if path.exists() {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                format!("file already exists: {}", path.display()),
            )));
        }
        Self::create(path)
    }

    /// Create a subgroup in the root group.
    pub fn create_group(&mut self, name: &str) -> Result<WritableGroup<'_>> {
        self.writer.create_group("/", name)?;
        let full_path = format!("/{name}");
        Ok(WritableGroup {
            writer: &mut self.writer,
            path: full_path,
        })
    }

    /// Get a builder for creating a dataset in the root group.
    pub fn new_dataset_builder(&mut self, name: &str) -> DatasetBuilder<'_> {
        DatasetBuilder::new(&mut self.writer, "/", name)
    }

    /// Add an attribute to the root group.
    pub fn add_attr<T: H5Type>(&mut self, name: &str, value: T) -> Result<()> {
        let dtype = crate::hl::dataset_builder::dtype_for_type::<T>()?;
        let byte_ptr = &value as *const T as *const u8;
        let data = unsafe { std::slice::from_raw_parts(byte_ptr, T::type_size()) };
        self.writer.add_root_attr(&AttrSpec {
            name,
            shape: &[],
            dtype,
            data,
        })
    }

    /// Add a one-dimensional array attribute to the root group.
    pub fn add_attr_array<T: H5Type>(&mut self, name: &str, values: &[T]) -> Result<()> {
        let dtype = crate::hl::dataset_builder::dtype_for_type::<T>()?;
        let byte_len = values
            .len()
            .checked_mul(T::type_size())
            .ok_or_else(|| Error::InvalidFormat("attribute byte size overflow".into()))?;
        let data = unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, byte_len) };
        let shape = [usize_to_u64(values.len(), "attribute element count")?];
        self.writer.add_root_attr(&AttrSpec {
            name,
            shape: &shape,
            dtype,
            data,
        })
    }

    /// Add a fixed-length ASCII string attribute to the root group.
    pub fn add_fixed_ascii_attr(&mut self, name: &str, value: &str, len: usize) -> Result<()> {
        let (dtype, data) = fixed_string_attr(&[value], len, false)?;
        self.writer.add_root_attr(&AttrSpec {
            name,
            shape: &[],
            dtype,
            data: &data,
        })
    }

    /// Add a fixed-length UTF-8 string attribute to the root group.
    pub fn add_fixed_utf8_attr(&mut self, name: &str, value: &str, len: usize) -> Result<()> {
        let (dtype, data) = fixed_string_attr(&[value], len, true)?;
        self.writer.add_root_attr(&AttrSpec {
            name,
            shape: &[],
            dtype,
            data: &data,
        })
    }

    /// Add a one-dimensional fixed-length ASCII string array attribute to the root group.
    pub fn add_fixed_ascii_attr_array(
        &mut self,
        name: &str,
        values: &[&str],
        len: usize,
    ) -> Result<()> {
        let (dtype, data) = fixed_string_attr(values, len, false)?;
        let shape = [usize_to_u64(values.len(), "attribute element count")?];
        self.writer.add_root_attr(&AttrSpec {
            name,
            shape: &shape,
            dtype,
            data: &data,
        })
    }

    /// Add a one-dimensional fixed-length UTF-8 string array attribute to the root group.
    pub fn add_fixed_utf8_attr_array(
        &mut self,
        name: &str,
        values: &[&str],
        len: usize,
    ) -> Result<()> {
        let (dtype, data) = fixed_string_attr(values, len, true)?;
        let shape = [usize_to_u64(values.len(), "attribute element count")?];
        self.writer.add_root_attr(&AttrSpec {
            name,
            shape: &shape,
            dtype,
            data: &data,
        })
    }

    /// Create a soft link in the root group.
    pub fn link_soft(&mut self, name: &str, target_path: &str) -> Result<()> {
        self.writer.create_soft_link("/", name, target_path)
    }

    /// Create a hard-link alias in the root group.
    pub fn link_hard(&mut self, name: &str, target_path: &str) -> Result<()> {
        self.writer.create_hard_link("/", name, target_path)
    }

    /// Create an external link in the root group.
    pub fn link_external(&mut self, name: &str, filename: &str, obj_path: &str) -> Result<()> {
        self.writer
            .create_external_link("/", name, filename, obj_path)
    }

    /// Finalize and close the file. Returns a read-only File handle.
    pub fn close(mut self) -> Result<crate::hl::file::File> {
        self.writer.finalize()?;
        crate::hl::file::File::open(&self.path)
    }

    /// Finalize the file (writes superblock and all metadata).
    pub fn flush(&mut self) -> Result<()> {
        self.writer.finalize()
    }
}

/// A writable group within a WritableFile.
pub struct WritableGroup<'a> {
    writer: &'a mut HdfFileWriter<fs::File>,
    path: String,
}

impl<'a> WritableGroup<'a> {
    /// Create a subgroup.
    pub fn create_group(&mut self, name: &str) -> Result<WritableGroup<'_>> {
        self.writer.create_group(&self.path, name)?;
        let full_path = format!("{}/{name}", self.path);
        Ok(WritableGroup {
            writer: self.writer,
            path: full_path,
        })
    }

    /// Get a builder for creating a dataset in this group.
    pub fn new_dataset_builder(&mut self, name: &str) -> DatasetBuilder<'_> {
        DatasetBuilder::new(self.writer, &self.path, name)
    }

    /// Add an attribute to this group.
    pub fn add_attr<T: H5Type>(&mut self, name: &str, value: T) -> Result<()> {
        let dtype = crate::hl::dataset_builder::dtype_for_type::<T>()?;
        let byte_ptr = &value as *const T as *const u8;
        let data = unsafe { std::slice::from_raw_parts(byte_ptr, T::type_size()) };
        self.writer.add_group_attr(
            &self.path,
            &AttrSpec {
                name,
                shape: &[],
                dtype,
                data,
            },
        )
    }

    /// Add a one-dimensional array attribute to this group.
    pub fn add_attr_array<T: H5Type>(&mut self, name: &str, values: &[T]) -> Result<()> {
        let dtype = crate::hl::dataset_builder::dtype_for_type::<T>()?;
        let byte_len = values
            .len()
            .checked_mul(T::type_size())
            .ok_or_else(|| Error::InvalidFormat("attribute byte size overflow".into()))?;
        let data = unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, byte_len) };
        let shape = [usize_to_u64(values.len(), "attribute element count")?];
        self.writer.add_group_attr(
            &self.path,
            &AttrSpec {
                name,
                shape: &shape,
                dtype,
                data,
            },
        )
    }

    /// Add a fixed-length ASCII string attribute to this group.
    pub fn add_fixed_ascii_attr(&mut self, name: &str, value: &str, len: usize) -> Result<()> {
        let (dtype, data) = fixed_string_attr(&[value], len, false)?;
        self.writer.add_group_attr(
            &self.path,
            &AttrSpec {
                name,
                shape: &[],
                dtype,
                data: &data,
            },
        )
    }

    /// Add a fixed-length UTF-8 string attribute to this group.
    pub fn add_fixed_utf8_attr(&mut self, name: &str, value: &str, len: usize) -> Result<()> {
        let (dtype, data) = fixed_string_attr(&[value], len, true)?;
        self.writer.add_group_attr(
            &self.path,
            &AttrSpec {
                name,
                shape: &[],
                dtype,
                data: &data,
            },
        )
    }

    /// Add a one-dimensional fixed-length ASCII string array attribute to this group.
    pub fn add_fixed_ascii_attr_array(
        &mut self,
        name: &str,
        values: &[&str],
        len: usize,
    ) -> Result<()> {
        let (dtype, data) = fixed_string_attr(values, len, false)?;
        let shape = [usize_to_u64(values.len(), "attribute element count")?];
        self.writer.add_group_attr(
            &self.path,
            &AttrSpec {
                name,
                shape: &shape,
                dtype,
                data: &data,
            },
        )
    }

    /// Add a one-dimensional fixed-length UTF-8 string array attribute to this group.
    pub fn add_fixed_utf8_attr_array(
        &mut self,
        name: &str,
        values: &[&str],
        len: usize,
    ) -> Result<()> {
        let (dtype, data) = fixed_string_attr(values, len, true)?;
        let shape = [usize_to_u64(values.len(), "attribute element count")?];
        self.writer.add_group_attr(
            &self.path,
            &AttrSpec {
                name,
                shape: &shape,
                dtype,
                data: &data,
            },
        )
    }

    /// Create a soft link in this group.
    pub fn link_soft(&mut self, name: &str, target_path: &str) -> Result<()> {
        self.writer.create_soft_link(&self.path, name, target_path)
    }

    /// Create a hard-link alias in this group.
    pub fn link_hard(&mut self, name: &str, target_path: &str) -> Result<()> {
        self.writer.create_hard_link(&self.path, name, target_path)
    }

    /// Create an external link in this group.
    pub fn link_external(&mut self, name: &str, filename: &str, obj_path: &str) -> Result<()> {
        self.writer
            .create_external_link(&self.path, name, filename, obj_path)
    }
}

fn fixed_string_attr(
    values: &[&str],
    len: usize,
    utf8: bool,
) -> Result<(crate::engine::writer::DtypeSpec, Vec<u8>)> {
    let len_u32 = usize_to_u32(len, "fixed string length")?;
    let dtype = if utf8 {
        crate::engine::writer::DtypeSpec::FixedUtf8String {
            len: len_u32,
            padding: 1,
        }
    } else {
        crate::engine::writer::DtypeSpec::FixedAsciiString {
            len: len_u32,
            padding: 1,
        }
    };
    let capacity = values.len().checked_mul(len).ok_or_else(|| {
        Error::InvalidFormat("fixed string attribute payload size overflow".into())
    })?;
    let mut data = Vec::with_capacity(capacity);
    for value in values {
        let bytes = value.as_bytes();
        if bytes.len() > len {
            return Err(Error::InvalidFormat(format!(
                "fixed string attribute has {} bytes, maximum is {len}",
                bytes.len()
            )));
        }
        data.extend_from_slice(bytes);
        data.resize(data.len() + (len - bytes.len()), 0);
    }
    Ok((dtype, data))
}

fn usize_to_u64(value: usize, context: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| Error::InvalidFormat(format!("{context} exceeds u64")))
}

fn usize_to_u32(value: usize, context: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::InvalidFormat(format!("{context} exceeds u32")))
}
