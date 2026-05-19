use std::collections::HashMap;
use std::fs;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::error::Result;
use crate::format::superblock::Superblock;
use crate::hl::dataset::Dataset;
use crate::hl::file::{FileInner, FileIntent};
use crate::hl::group::Group;
use crate::io::reader::HdfReader;

#[path = "mutable_file/attr_mutation.rs"]
mod attr_mutation;
#[path = "mutable_file/chunk_btree_v1.rs"]
mod chunk_btree_v1;
#[path = "mutable_file/chunk_btree_v2.rs"]
mod chunk_btree_v2;
#[path = "mutable_file/chunk_extensible_array.rs"]
mod chunk_extensible_array;
#[path = "mutable_file/chunk_fixed_array.rs"]
mod chunk_fixed_array;
#[path = "mutable_file/group_mutation.rs"]
mod group_mutation;
#[path = "mutable_file/resize.rs"]
mod resize;
#[path = "mutable_file/support.rs"]
mod support;
#[path = "mutable_file/write_chunk.rs"]
mod write_chunk;

/// A mutable HDF5 file opened for read-write access.
///
/// Supports resizing chunked datasets and writing new chunks.
pub struct MutableFile {
    /// Read path (for parsing)
    inner: Arc<Mutex<FileInner<BufReader<fs::File>>>>,
    /// Write path (for modifying)
    write_handle: fs::File,
    superblock: Superblock,
    path: PathBuf,
}

impl MutableFile {
    /// Open an existing HDF5 file for read-write access.
    pub fn open_rw<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        // Open for reading
        let read_file = fs::File::open(&path)?;
        let mut reader = HdfReader::new(BufReader::new(read_file));
        let superblock = Superblock::read(&mut reader)?;

        let inner = Arc::new(Mutex::new(FileInner {
            reader,
            superblock: superblock.clone(),
            path: Some(path.clone()),
            intent: FileIntent::ReadWrite,
            access_plist: crate::hl::plist::file_access::FileAccess::default(),
            dset_no_attrs_hint: false,
            open_objects: HashMap::new(),
            next_object_id: 1,
        }));

        // Open separately for writing
        let write_handle = fs::OpenOptions::new().write(true).open(&path)?;

        Ok(Self {
            inner,
            write_handle,
            superblock,
            path,
        })
    }

    /// Get the root group (read-only access).
    pub fn root_group(&self) -> Result<Group> {
        Group::open(self.inner.clone(), "/", self.superblock.root_addr)
    }

    /// Visit member names in the root group.
    pub fn visit_member_names<F>(&self, visitor: F) -> Result<()>
    where
        F: FnMut(&str) -> Result<()>,
    {
        self.root_group()?.visit_member_names(visitor)
    }

    /// Open a dataset by path.
    pub fn dataset(&self, path: &str) -> Result<Dataset> {
        let path_str = path.trim_start_matches('/');
        if let Some(last_slash) = path_str.rfind('/') {
            let group_path = &path_str[..last_slash];
            let ds_name = &path_str[last_slash + 1..];
            let root = self.root_group()?;
            let mut current = root;
            for part in group_path.split('/').filter(|s| !s.is_empty()) {
                current = current.open_group(part)?;
            }
            current.open_dataset(ds_name)
        } else {
            self.root_group()?.open_dataset(path_str)
        }
    }

    /// Open a group by path.
    pub fn group(&self, path: &str) -> Result<Group> {
        let path_str = path.trim_matches('/');
        if path_str.is_empty() {
            return self.root_group();
        }
        let mut current = self.root_group()?;
        for part in path_str.split('/').filter(|s| !s.is_empty()) {
            current = current.open_group(part)?;
        }
        Ok(current)
    }
}
