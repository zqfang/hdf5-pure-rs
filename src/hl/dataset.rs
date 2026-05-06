use std::fs;
use std::io::BufReader;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::hl::file::{register_open_object, unregister_open_object, FileInner, OpenObjectKind};

#[path = "dataset/access.rs"]
mod access;
#[path = "dataset/chunk_btree_v1.rs"]
mod chunk_btree_v1;
#[path = "dataset/chunk_btree_v2.rs"]
mod chunk_btree_v2;
#[path = "dataset/chunk_copy.rs"]
mod chunk_copy;
#[path = "dataset/chunk_linear_index.rs"]
mod chunk_linear_index;
#[path = "dataset/chunk_read.rs"]
mod chunk_read;
#[path = "dataset/info.rs"]
mod info;
#[path = "dataset/read.rs"]
mod read;
#[path = "dataset/selection.rs"]
mod selection;
#[path = "dataset/storage.rs"]
mod storage;
#[path = "dataset/support.rs"]
mod support;
#[path = "dataset/value_read.rs"]
mod value_read;
#[path = "dataset/virtual_dataset.rs"]
mod virtual_dataset;
#[path = "dataset/virtual_decode.rs"]
mod virtual_decode;
#[path = "dataset/virtual_source.rs"]
mod virtual_source;

#[cfg(test)]
#[path = "dataset/tests.rs"]
mod tests;

pub use access::{DatasetAccess, VdsMissingSourcePolicy, VdsView};
pub use info::{ChunkInfo, DatasetInfo, DatasetSpaceStatus, ExternalFileEntry, ExternalFileList};

use support::{read_le_u32_at, read_le_uint_at, read_u8_at, u64_from_usize, usize_from_u64};

/// An HDF5 dataset.
pub struct Dataset {
    inner: Arc<Mutex<FileInner<BufReader<fs::File>>>>,
    name: String,
    addr: u64,
    object_id: u64,
}

impl Dataset {
    pub(crate) fn new(
        inner: Arc<Mutex<FileInner<BufReader<fs::File>>>>,
        name: &str,
        addr: u64,
    ) -> Self {
        let object_id = register_open_object(&inner, OpenObjectKind::Dataset);
        Self {
            inner,
            name: name.to_string(),
            addr,
            object_id,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the object header address.
    pub fn addr(&self) -> u64 {
        self.addr
    }

    /// Return this dataset handle's high-level object id.
    pub fn object_id(&self) -> u64 {
        self.object_id
    }
}

impl Drop for Dataset {
    fn drop(&mut self) {
        unregister_open_object(&self.inner, self.object_id);
    }
}
