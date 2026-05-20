use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{BufReader, Read, Seek};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::error::{Error, Result};
use crate::format::messages::link::LinkType;
use crate::format::object_header::{self, RawMessage};
use crate::format::superblock::Superblock;
use crate::hl::dataset::Dataset;
use crate::hl::group::{visit_attr_names_at, visit_attrs_at, Group};
use crate::io::reader::HdfReader;

/// Represents the type of an HDF5 object as determined by its object header messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectType {
    Group,
    Dataset,
    NamedDatatype,
    Unknown,
}

/// File open intent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileIntent {
    ReadOnly,
    ReadWrite,
}

/// File opening mode.
///
/// Part of the hdf5-metno compatibility layer and should not be removed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpenMode {
    /// Open a file as read-only, file must exist.
    Read,
    /// Open a file as read-only in SWMR mode, file must exist.
    ReadSWMR,
    /// Open a file as read/write, file must exist.
    ReadWrite,
    /// Create a file, truncate if exists.
    Create,
    /// Create a file, fail if exists.
    CreateExcl,
    /// Open a file as read/write if exists, create otherwise.
    Append,
}

/// File-level metadata summary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileInfo {
    pub superblock: SuperblockInfo,
    pub free_space: FreeSpaceInfo,
    pub shared_messages: SharedMessageInfo,
}

/// Superblock/storage information reported by file metadata queries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SuperblockInfo {
    pub version: u8,
    pub size: u64,
    pub extension_size: u64,
}

/// Free-space-manager information reported by file metadata queries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FreeSpaceInfo {
    pub version: u8,
    pub metadata_size: u64,
    pub total_space: u64,
}

/// Shared-object-header-message information reported by file metadata queries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedMessageInfo {
    pub header_size: u64,
    pub message_info_size: u64,
}

/// Metadata-cache size/status snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetadataCacheSize {
    pub max_size: usize,
    pub min_clean_size: usize,
    pub current_size: usize,
    pub current_num_entries: usize,
}

/// Metadata-cache image status.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetadataCacheImageInfo {
    pub generated: bool,
    pub size: usize,
}

/// Page-buffering status counters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageBufferingStats {
    pub metadata_accesses: u64,
    pub metadata_hits: u64,
    pub raw_data_accesses: u64,
    pub raw_data_hits: u64,
}

/// Internal state of an open HDF5 file.
pub(crate) struct FileInner<R: Read + Seek> {
    pub reader: HdfReader<R>,
    pub superblock: Superblock,
    pub path: Option<PathBuf>,
    pub intent: FileIntent,
    pub access_plist: crate::hl::plist::file_access::FileAccess,
    pub dset_no_attrs_hint: bool,
    pub open_objects: HashMap<u64, OpenObjectKind>,
    pub next_object_id: u64,
}

/// An open HDF5 file.
pub struct File {
    inner: Arc<Mutex<FileInner<BufReader<fs::File>>>>,
    superblock: Superblock,
    path: Option<Arc<PathBuf>>,
    object_id: u64,
    intent: FileIntent,
}

/// File-creation builder placeholder.
///
/// Part of the hdf5-metno compatibility layer and should not be removed.
#[derive(Default, Clone, Debug)]
pub struct FileCreateBuilder;

/// File builder allowing limited compatibility with hdf5-metno open helpers.
///
/// Part of the hdf5-metno compatibility layer and should not be removed.
#[derive(Clone, Debug)]
pub struct FileBuilder {
    fapl: crate::hl::plist::file_access::FileAccess,
    fcpl: FileCreateBuilder,
}

impl Default for FileBuilder {
    /// Creates the default file builder placeholder state.
    ///
    /// Part of the hdf5-metno compatibility layer and should not be removed.
    fn default() -> Self {
        Self {
            fapl: crate::hl::plist::file_access::FileAccess::default(),
            fcpl: FileCreateBuilder,
        }
    }
}

impl FileBuilder {
    /// Creates a new file builder with default property lists.
    ///
    /// Part of the hdf5-metno compatibility layer and should not be removed.
    pub fn new() -> Self {
        Self::default()
    }

    /// Opens a file as read-only, file must exist.
    ///
    /// Part of the hdf5-metno compatibility layer and should not be removed.
    pub fn open<P: AsRef<Path>>(&self, filename: P) -> Result<File> {
        self.open_as(filename, OpenMode::Read)
    }

    /// Opens a file as read/write, file must exist.
    ///
    /// Part of the hdf5-metno compatibility layer and should not be removed.
    pub fn open_rw<P: AsRef<Path>>(&self, filename: P) -> Result<File> {
        self.open_as(filename, OpenMode::ReadWrite)
    }

    /// Creates a file, truncates if exists.
    ///
    /// Part of the hdf5-metno compatibility layer and should not be removed.
    pub fn create<P: AsRef<Path>>(&self, filename: P) -> Result<File> {
        self.open_as(filename, OpenMode::Create)
    }

    /// Creates a file, fails if exists.
    ///
    /// Part of the hdf5-metno compatibility layer and should not be removed.
    pub fn create_excl<P: AsRef<Path>>(&self, filename: P) -> Result<File> {
        self.open_as(filename, OpenMode::CreateExcl)
    }

    /// Opens a file as read/write if exists, creates otherwise.
    ///
    /// Part of the hdf5-metno compatibility layer and should not be removed.
    pub fn append<P: AsRef<Path>>(&self, filename: P) -> Result<File> {
        self.open_as(filename, OpenMode::Append)
    }

    /// Opens a file in a given mode.
    ///
    /// Part of the hdf5-metno compatibility layer and should not be removed.
    pub fn open_as<P: AsRef<Path>>(&self, filename: P, mode: OpenMode) -> Result<File> {
        self.fapl.ensure_runtime_supported_driver()?;
        let file = match mode {
            OpenMode::Read => File::open(filename)?,
            OpenMode::ReadSWMR => Err(Error::Unsupported(
                "hdf5-metno compatibility: SWMR File open is not supported".into(),
            ))?,
            OpenMode::ReadWrite => File::open_rw(filename)?,
            OpenMode::Create => File::create(filename)?,
            OpenMode::CreateExcl => File::create_excl(filename)?,
            OpenMode::Append => File::append(filename)?,
        };
        file.set_access_plist(self.fapl.clone());
        Ok(file)
    }

    /// Sets current file access property list to a given one.
    ///
    /// Part of the hdf5-metno compatibility layer and should not be removed.
    pub fn set_access_plist(
        &mut self,
        fapl: &crate::hl::plist::file_access::FileAccess,
    ) -> Result<&mut Self> {
        self.fapl = fapl.clone();
        Ok(self)
    }

    /// A short alias for `set_access_plist()`.
    ///
    /// Part of the hdf5-metno compatibility layer and should not be removed.
    pub fn set_fapl(
        &mut self,
        fapl: &crate::hl::plist::file_access::FileAccess,
    ) -> Result<&mut Self> {
        self.set_access_plist(fapl)
    }

    /// Returns the builder object for the file access property list.
    ///
    /// Part of the hdf5-metno compatibility layer and should not be removed.
    pub fn access_plist(&mut self) -> &mut crate::hl::plist::file_access::FileAccess {
        &mut self.fapl
    }

    /// A short alias for `access_plist()`.
    ///
    /// Part of the hdf5-metno compatibility layer and should not be removed.
    pub fn fapl(&mut self) -> &mut crate::hl::plist::file_access::FileAccess {
        self.access_plist()
    }

    /// Allows accessing the builder object for the file access property list.
    ///
    /// Part of the hdf5-metno compatibility layer and should not be removed.
    pub fn with_access_plist<F>(&mut self, func: F) -> &mut Self
    where
        F: Fn(
            &mut crate::hl::plist::file_access::FileAccess,
        ) -> &mut crate::hl::plist::file_access::FileAccess,
    {
        func(&mut self.fapl);
        self
    }

    /// A short alias for `with_access_plist()`.
    ///
    /// Part of the hdf5-metno compatibility layer and should not be removed.
    pub fn with_fapl<F>(&mut self, func: F) -> &mut Self
    where
        F: Fn(
            &mut crate::hl::plist::file_access::FileAccess,
        ) -> &mut crate::hl::plist::file_access::FileAccess,
    {
        self.with_access_plist(func)
    }

    /// Sets current file creation property list to a given one.
    ///
    /// Part of the hdf5-metno compatibility layer and should not be removed.
    pub fn set_create_plist(
        &mut self,
        _fcpl: &crate::hl::plist::file_create::FileCreate,
    ) -> Result<&mut Self> {
        Ok(self)
    }

    /// A short alias for `set_create_plist()`.
    ///
    /// Part of the hdf5-metno compatibility layer and should not be removed.
    pub fn set_fcpl(
        &mut self,
        fcpl: &crate::hl::plist::file_create::FileCreate,
    ) -> Result<&mut Self> {
        self.set_create_plist(fcpl)
    }

    /// Returns the placeholder builder object for the file creation property list.
    ///
    /// Part of the hdf5-metno compatibility layer and should not be removed.
    pub fn create_plist(&mut self) -> &mut FileCreateBuilder {
        &mut self.fcpl
    }

    /// A short alias for `create_plist()`.
    ///
    /// Part of the hdf5-metno compatibility layer and should not be removed.
    pub fn fcpl(&mut self) -> &mut FileCreateBuilder {
        self.create_plist()
    }

    /// Allows accessing the placeholder builder object for the file creation property list.
    ///
    /// Part of the hdf5-metno compatibility layer and should not be removed.
    pub fn with_create_plist<F>(&mut self, func: F) -> &mut Self
    where
        F: Fn(&mut FileCreateBuilder) -> &mut FileCreateBuilder,
    {
        func(&mut self.fcpl);
        self
    }

    /// A short alias for `with_create_plist()`.
    ///
    /// Part of the hdf5-metno compatibility layer and should not be removed.
    pub fn with_fcpl<F>(&mut self, func: F) -> &mut Self
    where
        F: Fn(&mut FileCreateBuilder) -> &mut FileCreateBuilder,
    {
        self.with_create_plist(func)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OpenObjectKind {
    File,
    Group,
    Dataset,
    Attribute,
}

impl File {
    const MAX_SOFT_LINK_TRAVERSALS: usize = 40;
    /// Per-component byte cap, matching upstream `H5G_TRAVERSE_PATH_MAX`.
    /// Bounds the length of any single name segment between '/' separators.
    const MAX_PATH_COMPONENT_LEN: usize = 1024;

    /// Open an HDF5 file for reading.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_with_intent(path, FileIntent::ReadOnly)
    }

    fn open_with_intent<P: AsRef<Path>>(path: P, intent: FileIntent) -> Result<Self> {
        let path_ref = path.as_ref();
        let f = match intent {
            FileIntent::ReadOnly => fs::File::open(path_ref),
            FileIntent::ReadWrite => fs::OpenOptions::new().read(true).write(true).open(path_ref),
        }
        .map_err(|e| {
            Error::Io(std::io::Error::new(
                e.kind(),
                format!("failed to open {}: {e}", path_ref.display()),
            ))
        })?;

        let buf = BufReader::new(f);
        let mut reader = HdfReader::new(buf);

        let superblock = Superblock::read(&mut reader)?;

        let inner = Arc::new(Mutex::new(FileInner {
            reader,
            superblock: superblock.clone(),
            path: Some(path_ref.to_path_buf()),
            intent,
            access_plist: crate::hl::plist::file_access::FileAccess::default(),
            dset_no_attrs_hint: false,
            open_objects: HashMap::new(),
            next_object_id: 1,
        }));

        let object_id = register_open_object(&inner, OpenObjectKind::File);
        Ok(File {
            inner,
            superblock,
            path: Some(Arc::new(path_ref.to_path_buf())),
            object_id,
            intent,
        })
    }

    /// Opens a file as read/write, file must exist.
    ///
    /// Part of the hdf5-metno compatibility layer and should not be removed.
    pub fn open_rw<P: AsRef<Path>>(filename: P) -> Result<Self> {
        Self::open_with_intent(filename, FileIntent::ReadWrite)
    }

    /// Creates a file, truncates if exists.
    ///
    /// Part of the hdf5-metno compatibility layer and should not be removed.
    pub fn create<P: AsRef<Path>>(filename: P) -> Result<Self> {
        let path = filename.as_ref().to_path_buf();
        let writable = crate::hl::writable_file::WritableFile::create(&path)?;
        let _ = writable.close()?;
        Self::open_with_intent(&path, FileIntent::ReadWrite)
    }

    /// Creates a file, fails if exists.
    ///
    /// Part of the hdf5-metno compatibility layer and should not be removed.
    pub fn create_excl<P: AsRef<Path>>(filename: P) -> Result<Self> {
        let path = filename.as_ref().to_path_buf();
        let writable = crate::hl::writable_file::WritableFile::create_excl(&path)?;
        let _ = writable.close()?;
        Self::open_with_intent(&path, FileIntent::ReadWrite)
    }

    /// Opens a file as read/write if exists, creates otherwise.
    ///
    /// Part of the hdf5-metno compatibility layer and should not be removed.
    pub fn append<P: AsRef<Path>>(filename: P) -> Result<Self> {
        let path = filename.as_ref().to_path_buf();
        if path.exists() {
            Self::open_rw(&path)
        } else {
            Self::create(&path)
        }
    }

    /// Opens a file in a given mode.
    ///
    /// Part of the hdf5-metno compatibility layer and should not be removed.
    pub fn open_as<P: AsRef<Path>>(filename: P, mode: OpenMode) -> Result<Self> {
        FileBuilder::new().open_as(filename, mode)
    }

    /// Opens a file with custom file-access and file-creation options.
    ///
    /// Part of the hdf5-metno compatibility layer and should not be removed.
    pub fn with_options() -> FileBuilder {
        FileBuilder::new()
    }

    /// Get the superblock.
    pub fn superblock(&self) -> &Superblock {
        &self.superblock
    }

    /// Return the current on-disk file size in bytes.
    ///
    /// This mirrors the useful read-side subset of HDF5's `H5Fget_filesize`
    /// without exposing the broader file-driver API surface.
    pub fn file_size(&self) -> Result<u64> {
        self.inner.lock().reader.len()
    }

    /// Returns the file size in bytes, or 0 if it cannot be read.
    ///
    /// Part of the hdf5-metno compatibility layer and should not be removed.
    pub fn size(&self) -> u64 {
        self.file_size().unwrap_or(0)
    }

    /// Returns the free space in the file in bytes.
    ///
    /// Part of the hdf5-metno compatibility layer and should not be removed.
    pub fn free_space(&self) -> u64 {
        self.freespace()
    }

    /// Returns true if the file was opened in a read-only mode.
    ///
    /// Part of the hdf5-metno compatibility layer and should not be removed.
    pub fn is_read_only(&self) -> bool {
        self.intent() == FileIntent::ReadOnly
    }

    /// Returns the userblock size in bytes.
    ///
    /// Part of the hdf5-metno compatibility layer and should not be removed.
    pub fn userblock(&self) -> u64 {
        self.superblock.base_addr
    }

    /// Flushes the file to the storage medium.
    ///
    /// Part of the hdf5-metno compatibility layer and should not be removed.
    pub fn flush(&self) -> Result<()> {
        Ok(())
    }

    /// Closes the file handle.
    ///
    /// Part of the hdf5-metno compatibility layer and should not be removed.
    pub fn close(self) -> Result<()> {
        Ok(())
    }

    /// Mark this file as ready for opening as SWMR.
    ///
    /// Part of the hdf5-metno compatibility layer and should not be removed.
    pub fn start_swmr(&self) -> Result<()> {
        Err(Error::Unsupported(
            "hdf5-metno compatibility: SWMR write mode is not supported".into(),
        ))
    }

    /// Return the path used to open this file, when the file has an on-disk path.
    ///
    /// This mirrors the useful file-level subset of HDF5's `H5Fget_name`.
    pub fn path(&self) -> Option<PathBuf> {
        self.path.as_deref().cloned()
    }

    /// Borrow the path used to open this file for the duration of `visitor`.
    pub fn with_path<T, F>(&self, visitor: F) -> T
    where
        F: FnOnce(Option<&Path>) -> T,
    {
        visitor(self.path.as_deref().map(PathBuf::as_path))
    }

    /// Return the access properties for this open file.
    pub fn access_plist(&self) -> crate::hl::plist::file_access::FileAccess {
        crate::hl::plist::file_access::FileAccess::from_file(self)
    }

    /// A short alias for `access_plist()`.
    ///
    /// Part of the hdf5-metno compatibility layer and should not be removed.
    pub fn fapl(&self) -> crate::hl::plist::file_access::FileAccess {
        self.access_plist()
    }

    /// Returns a copy of the file creation property list.
    ///
    /// Part of the hdf5-metno compatibility layer and should not be removed.
    pub fn create_plist(&self) -> crate::hl::plist::file_create::FileCreate {
        crate::hl::plist::file_create::FileCreate::from_file(self)
    }

    /// A short alias for `create_plist()`.
    ///
    /// Part of the hdf5-metno compatibility layer and should not be removed.
    pub fn fcpl(&self) -> crate::hl::plist::file_create::FileCreate {
        self.create_plist()
    }

    pub(crate) fn access_plist_snapshot(&self) -> crate::hl::plist::file_access::FileAccess {
        self.inner.lock().access_plist.clone()
    }

    /// Replace the file's stored access-property state.
    pub fn set_access_plist(&self, plist: crate::hl::plist::file_access::FileAccess) {
        self.inner.lock().access_plist = plist;
    }

    /// Return this file's open intent.
    pub fn intent(&self) -> FileIntent {
        self.intent
    }

    /// Return the parsed end-of-address marker from the superblock.
    pub fn eoa(&self) -> u64 {
        self.superblock.eof_addr
    }

    /// Return the known free-space size. The reader does not currently parse
    /// free-space manager state, so this reports zero known free bytes.
    pub fn freespace(&self) -> u64 {
        0
    }

    /// Return file metadata information in the v2 `H5F_info_t` layout.
    pub fn info(&self) -> Result<FileInfo> {
        Ok(FileInfo {
            superblock: SuperblockInfo {
                version: self.superblock.version,
                size: u64::try_from(self.superblock.checked_size()?)
                    .map_err(|_| Error::InvalidFormat("superblock size does not fit u64".into()))?,
                extension_size: 0,
            },
            free_space: FreeSpaceInfo {
                version: 0,
                metadata_size: 0,
                total_space: self.freespace(),
            },
            shared_messages: SharedMessageInfo {
                header_size: 0,
                message_info_size: 0,
            },
        })
    }

    /// Return file metadata information in the v1 `H5F_info_t` layout.
    pub fn info_v1(&self) -> Result<FileInfo> {
        self.info()
    }

    /// Return the current file image bytes.
    pub fn file_image(&self) -> Result<Vec<u8>> {
        let mut guard = self.inner.lock();
        let pos = guard.reader.position_physical()?;
        let len = guard.reader.len_physical()?;
        let len = usize::try_from(len)
            .map_err(|_| Error::InvalidFormat("file image length does not fit usize".into()))?;
        let mut image = vec![0u8; len];
        Self::read_file_image_into(&mut guard.reader, pos, &mut image)?;
        Ok(image)
    }

    /// Read the current file image into caller-provided storage.
    pub fn file_image_into(&self, out: &mut [u8]) -> Result<()> {
        let mut guard = self.inner.lock();
        let pos = guard.reader.position_physical()?;
        let len = guard.reader.len_physical()?;
        let len = usize::try_from(len)
            .map_err(|_| Error::InvalidFormat("file image length does not fit usize".into()))?;
        if out.len() != len {
            return Err(Error::InvalidFormat(format!(
                "file image output length mismatch: expected {len}, got {}",
                out.len()
            )));
        }
        Self::read_file_image_into(&mut guard.reader, pos, out)
    }

    fn read_file_image_into<R: std::io::Read + std::io::Seek>(
        reader: &mut crate::io::reader::HdfReader<R>,
        pos: u64,
        out: &mut [u8],
    ) -> Result<()> {
        reader.seek_physical(0)?;
        let read = reader.read_bytes_into(out);
        let restore = reader.seek_physical(pos);
        match (read, restore) {
            (Ok(_), Ok(_)) => Ok(()),
            (Err(err), _) => Err(err),
            (_, Err(err)) => Err(err),
        }
    }

    /// Return a stable file-number surrogate for this open file.
    pub fn fileno(&self) -> Result<u64> {
        self.with_path(|path| {
            let path =
                path.ok_or_else(|| Error::Unsupported("open file has no filesystem path".into()))?;
            file_number_from_path(path)
        })
    }

    /// Return this file handle's high-level object id.
    pub fn object_id(&self) -> u64 {
        self.object_id
    }

    /// Return the number of currently live high-level objects for this file.
    pub fn obj_count(&self) -> usize {
        self.inner.lock().open_objects.len()
    }

    /// Return currently live high-level object ids for this file.
    pub fn obj_ids(&self) -> Vec<u64> {
        let mut ids = Vec::new();
        self.obj_ids_into(&mut ids);
        ids
    }

    /// Visit currently live high-level object ids for this file in sorted order.
    pub fn visit_obj_ids<F>(&self, mut visitor: F)
    where
        F: FnMut(u64),
    {
        let mut ids = Vec::new();
        self.obj_ids_into(&mut ids);
        for id in ids {
            visitor(id);
        }
    }

    /// Store currently live high-level object ids in caller-provided storage.
    pub fn obj_ids_into(&self, out: &mut Vec<u64>) {
        out.clear();
        out.extend(self.inner.lock().open_objects.keys().copied());
        out.sort_unstable();
    }

    /// Return the native handle for the direct file driver when available.
    #[cfg(unix)]
    pub fn vfd_handle(&self) -> Option<i64> {
        use std::os::fd::AsRawFd;

        Some(i64::from(
            self.inner.lock().reader.get_ref().get_ref().as_raw_fd(),
        ))
    }

    /// Return the native handle for the direct file driver when available.
    #[cfg(not(unix))]
    pub fn vfd_handle(&self) -> Option<i64> {
        None
    }

    /// Return whether MPI atomicity is enabled for this file.
    ///
    /// This pure-Rust reader does not use MPI or the parallel HDF5 VFD, so
    /// atomicity is always disabled.
    pub fn mpi_atomicity(&self) -> bool {
        false
    }

    /// Return metadata cache configuration for this file.
    pub fn mdc_config(&self) -> crate::hl::plist::file_access::MetadataCacheConfig {
        self.access_plist().mdc_config()
    }

    /// Set metadata cache configuration for this open file handle.
    pub fn set_mdc_config(&self, config: crate::hl::plist::file_access::MetadataCacheConfig) {
        self.inner.lock().access_plist.set_mdc_config(config);
    }

    /// Set library format-version bounds for this open file handle.
    pub fn set_libver_bounds(
        &self,
        low: crate::hl::plist::file_access::LibverBound,
        high: crate::hl::plist::file_access::LibverBound,
    ) {
        self.inner.lock().access_plist.set_libver_bounds(low, high);
    }

    /// Set latest-format bounds for this open file handle.
    pub fn set_latest_format(&self) {
        use crate::hl::plist::file_access::LibverBound;

        self.set_libver_bounds(LibverBound::Latest, LibverBound::Latest);
    }

    /// Return metadata cache hit rate. No libhdf5 metadata cache is present.
    pub fn mdc_hit_rate(&self) -> f64 {
        0.0
    }

    /// Return metadata cache size/status.
    pub fn mdc_size(&self) -> MetadataCacheSize {
        MetadataCacheSize {
            max_size: 0,
            min_clean_size: 0,
            current_size: 0,
            current_num_entries: 0,
        }
    }

    /// Return metadata cache logging status `(enabled, currently_logging)`.
    pub fn mdc_logging_status(&self) -> (bool, bool) {
        (false, false)
    }

    /// Return page-buffering status counters.
    pub fn page_buffering_stats(&self) -> PageBufferingStats {
        PageBufferingStats {
            metadata_accesses: 0,
            metadata_hits: 0,
            raw_data_accesses: 0,
            raw_data_hits: 0,
        }
    }

    /// Return metadata cache image status.
    pub fn mdc_image_info(&self) -> MetadataCacheImageInfo {
        MetadataCacheImageInfo {
            generated: false,
            size: 0,
        }
    }

    /// Return the dataset-no-attributes optimization hint.
    pub fn dset_no_attrs_hint(&self) -> bool {
        self.inner.lock().dset_no_attrs_hint
    }

    /// Set the dataset-no-attributes optimization hint for this open file.
    pub fn set_dset_no_attrs_hint(&self, enabled: bool) {
        self.inner.lock().dset_no_attrs_hint = enabled;
    }

    pub(crate) fn from_inner(inner: Arc<Mutex<FileInner<BufReader<fs::File>>>>) -> Self {
        let guard = inner.lock();
        let superblock = guard.superblock.clone();
        let intent = guard.intent;
        let path = guard.path.clone().map(Arc::new);
        drop(guard);
        let object_id = register_open_object(&inner, OpenObjectKind::File);
        Self {
            inner,
            superblock,
            path,
            object_id,
            intent,
        }
    }

    /// Get the root group.
    pub fn root_group(&self) -> Result<Group> {
        Group::open(self.inner.clone(), "/", self.root_addr())
    }

    fn root_addr(&self) -> u64 {
        self.inner.lock().superblock.root_addr
    }

    /// List all member names in the root group.
    pub fn member_names(&self) -> Result<Vec<String>> {
        let mut names = Vec::new();
        self.member_names_into(&mut names)?;
        Ok(names)
    }

    /// Visit all root-group member names without returning an owned list.
    pub fn visit_member_names<F>(&self, visitor: F) -> Result<()>
    where
        F: FnMut(&str) -> Result<()>,
    {
        self.root_group()?.visit_member_names(visitor)
    }

    /// Visit all root-group members as `(name, object_header_addr)` pairs.
    pub fn visit_members<F>(&self, visitor: F) -> Result<()>
    where
        F: FnMut(&str, u64) -> Result<()>,
    {
        self.root_group()?.visit_members(visitor)
    }

    /// Store root-group member names in caller-provided storage.
    pub fn member_names_into(&self, out: &mut Vec<String>) -> Result<()> {
        out.clear();
        self.visit_member_names(|name| {
            out.push(name.to_string());
            Ok(())
        })
    }

    /// Open a group by path (starting from root).
    pub fn group(&self, path: &str) -> Result<Group> {
        let resolved = self.resolve_path(path)?;
        if resolved.object_type != ObjectType::Group {
            return Err(Error::InvalidFormat(format!(
                "'{path}' is not a group (type: {:?})",
                resolved.object_type
            )));
        }
        Group::open(resolved.inner, &resolved.path, resolved.addr)
    }

    /// List attribute names on the root group.
    pub fn attr_names(&self) -> Result<Vec<String>> {
        let mut names = Vec::new();
        self.attr_names_into(&mut names)?;
        Ok(names)
    }

    /// Visit attribute names on the root group in storage order.
    pub fn visit_attr_names<F>(&self, mut f: F) -> Result<()>
    where
        F: FnMut(&str) -> Result<()>,
    {
        visit_attr_names_at(&self.inner, self.root_addr(), &mut f)
    }

    /// Append root-group attribute names into caller-provided storage.
    pub fn attr_names_into(&self, out: &mut Vec<String>) -> Result<()> {
        out.clear();
        self.visit_attr_names(|name| {
            out.push(name.to_string());
            Ok(())
        })
    }

    /// List attributes on the root group.
    pub fn attrs(&self) -> Result<Vec<crate::hl::attribute::Attribute>> {
        let mut attrs = Vec::new();
        self.attrs_into(&mut attrs)?;
        Ok(attrs)
    }

    /// Visit attributes on the root group in storage order.
    pub fn visit_attrs<F>(&self, mut f: F) -> Result<()>
    where
        F: FnMut(&crate::hl::attribute::Attribute) -> Result<()>,
    {
        visit_attrs_at(&self.inner, self.root_addr(), &mut f)
    }

    /// Store root-group attributes in caller-provided storage.
    pub fn attrs_into(&self, out: &mut Vec<crate::hl::attribute::Attribute>) -> Result<()> {
        out.clear();
        out.extend(crate::hl::attribute::collect_attributes(
            &self.inner,
            self.root_addr(),
        )?);
        Ok(())
    }

    /// List attributes on the root group sorted by tracked creation order.
    pub fn attrs_by_creation_order(&self) -> Result<Vec<crate::hl::attribute::Attribute>> {
        let mut attrs = Vec::new();
        self.attrs_by_creation_order_into(&mut attrs)?;
        Ok(attrs)
    }

    /// Visit root-group attributes sorted by tracked creation order.
    pub fn visit_attrs_by_creation_order<F>(&self, mut f: F) -> Result<()>
    where
        F: FnMut(&crate::hl::attribute::Attribute) -> Result<()>,
    {
        let mut attrs = Vec::new();
        self.attrs_by_creation_order_into(&mut attrs)?;
        for attr in attrs.iter() {
            f(attr)?;
        }
        Ok(())
    }

    /// Store root-group attributes sorted by tracked creation order in caller-provided storage.
    pub fn attrs_by_creation_order_into(
        &self,
        out: &mut Vec<crate::hl::attribute::Attribute>,
    ) -> Result<()> {
        out.clear();
        crate::hl::attribute::collect_attributes_by_creation_order_into(
            &self.inner,
            self.root_addr(),
            out,
        )?;
        Ok(())
    }

    /// Get an attribute by name on the root group.
    pub fn attr(&self, name: &str) -> Result<crate::hl::attribute::Attribute> {
        crate::hl::attribute::get_attr(&self.inner, self.root_addr(), name)
    }

    /// Check whether an attribute exists on the root group.
    pub fn attr_exists(&self, name: &str) -> Result<bool> {
        crate::hl::attribute::attr_exists(&self.inner, self.root_addr(), name)
    }

    /// Async-compatible alias for attribute-existence checks.
    pub fn attr_exists_async(&self, name: &str) -> Result<bool> {
        self.attr_exists(name)
    }

    /// Check whether an attribute exists on an object addressed by path.
    pub fn attr_exists_by_name(&self, object_path: &str, attr_name: &str) -> Result<bool> {
        let resolved = self.resolve_path(object_path)?;
        crate::hl::attribute::attr_exists(&resolved.inner, resolved.addr, attr_name)
    }

    /// Async-compatible alias for path-based attribute-existence checks.
    pub fn attr_exists_by_name_async(&self, object_path: &str, attr_name: &str) -> Result<bool> {
        self.attr_exists_by_name(object_path, attr_name)
    }

    /// Open a dataset by path from the root group.
    pub fn dataset(&self, path: &str) -> Result<Dataset> {
        let resolved = self.resolve_path(path)?;
        if resolved.object_type != ObjectType::Dataset {
            return Err(Error::InvalidFormat(format!(
                "'{path}' is not a dataset (type: {:?})",
                resolved.object_type
            )));
        }
        Ok(Dataset::new(resolved.inner, &resolved.path, resolved.addr))
    }

    pub(crate) fn object_type_for_path(&self, path: &str) -> Result<ObjectType> {
        Ok(self.resolve_path(path)?.object_type)
    }

    fn resolve_path(&self, path: &str) -> Result<ResolvedObject> {
        let mut path = canonical_path(path);
        let mut traversals = 0usize;
        let mut seen_paths = HashSet::new();

        'resolve: loop {
            if seen_paths.contains(&path) {
                return Err(Error::InvalidFormat(format!(
                    "soft link cycle detected while resolving '{path}'"
                )));
            }
            seen_paths.insert(path.clone());
            if path == "/" {
                return Ok(self.root_resolved_object(path));
            }

            let parts = Self::path_components(&path)?;
            let mut current = self.root_group()?;
            let mut current_path = String::from("/");

            for (idx, part) in parts.iter().enumerate() {
                let is_last = idx + 1 == parts.len();
                let link = self.lookup_group_link(&current, part)?;
                match self.resolve_path_component(
                    &current,
                    &current_path,
                    part,
                    is_last,
                    &parts[idx + 1..],
                    link,
                    &mut traversals,
                )? {
                    PathStep::Resolved(resolved) => return Ok(resolved),
                    PathStep::Descend(next_group, next_path) => {
                        current = next_group;
                        current_path = next_path;
                    }
                    PathStep::Restart(new_path) => {
                        path = new_path;
                        continue 'resolve;
                    }
                }
            }
        }
    }

    fn root_resolved_object(&self, path: String) -> ResolvedObject {
        ResolvedObject {
            inner: self.inner.clone(),
            path,
            addr: self.root_addr(),
            object_type: ObjectType::Group,
        }
    }

    fn path_components(path: &str) -> Result<Vec<&str>> {
        let parts: Vec<&str> = path
            .trim_start_matches('/')
            .split('/')
            .filter(|part| !part.is_empty())
            .collect();
        for part in &parts {
            if part.len() > Self::MAX_PATH_COMPONENT_LEN {
                return Err(Error::InvalidFormat(format!(
                    "path component exceeds {}-byte limit ({} bytes)",
                    Self::MAX_PATH_COMPONENT_LEN,
                    part.len()
                )));
            }
        }
        Ok(parts)
    }

    fn lookup_group_link(&self, current: &Group, part: &str) -> Result<ResolvedLink> {
        current.with_link_ref_by_name(part, ResolvedLink::from_ref)
    }

    fn resolve_path_component(
        &self,
        _current: &Group,
        current_path: &str,
        part: &str,
        is_last: bool,
        remaining_parts: &[&str],
        link: ResolvedLink,
        traversals: &mut usize,
    ) -> Result<PathStep> {
        match link {
            ResolvedLink::Hard { addr } => {
                let next_path = join_absolute_path(current_path, part);
                let object_type = self.object_type_at(addr)?;
                self.resolve_hard_path_step(next_path, addr, object_type, is_last)
            }
            ResolvedLink::Soft { target } => {
                *traversals += 1;
                if *traversals > Self::MAX_SOFT_LINK_TRAVERSALS {
                    return Err(Error::InvalidFormat(
                        "soft link traversal limit exceeded".into(),
                    ));
                }
                Ok(PathStep::Restart(resolve_soft_target(
                    current_path,
                    &target,
                    remaining_parts,
                )))
            }
            ResolvedLink::External {
                filename,
                object_path,
            } => {
                let target_path = append_path_parts(canonical_path(&object_path), remaining_parts);
                let file_path = self.resolve_external_file_path(&filename)?;
                let external_file = File::open(file_path)?;
                Ok(PathStep::Resolved(
                    external_file.resolve_path(&target_path)?,
                ))
            }
            ResolvedLink::UserDefined(kind) => Err(Error::Unsupported(format!(
                "user-defined link traversal is not supported for link type {kind}"
            ))),
        }
    }

    fn resolve_hard_path_step(
        &self,
        next_path: String,
        addr: u64,
        object_type: ObjectType,
        is_last: bool,
    ) -> Result<PathStep> {
        if is_last {
            return Ok(PathStep::Resolved(ResolvedObject {
                inner: self.inner.clone(),
                path: next_path,
                addr,
                object_type,
            }));
        }

        if object_type != ObjectType::Group {
            return Err(Error::InvalidFormat(format!(
                "'{next_path}' is not a group (type: {object_type:?})"
            )));
        }

        let next_group = Group::open(self.inner.clone(), &next_path, addr)?;
        Ok(PathStep::Descend(next_group, next_path))
    }

    fn object_type_at(&self, addr: u64) -> Result<ObjectType> {
        let mut guard = self.inner.lock();
        let oh = object_header::ObjectHeader::read_at(&mut guard.reader, addr)?;
        Ok(object_type_from_messages(&oh.messages))
    }

    fn resolve_external_file_path(&self, filename: &str) -> Result<PathBuf> {
        let path = PathBuf::from(filename);
        if path.is_absolute() {
            return Ok(path);
        }
        let base = self
            .inner
            .lock()
            .path
            .as_ref()
            .and_then(|path| path.parent().map(Path::to_path_buf))
            .ok_or_else(|| {
                Error::InvalidFormat("relative external link has no base file path".into())
            })?;
        Ok(base.join(path))
    }
}

impl Drop for File {
    fn drop(&mut self) {
        unregister_open_object(&self.inner, self.object_id);
    }
}

pub(crate) fn register_open_object(
    inner: &Arc<Mutex<FileInner<BufReader<fs::File>>>>,
    kind: OpenObjectKind,
) -> u64 {
    let mut guard = inner.lock();
    let id = guard.next_object_id;
    guard.next_object_id = guard.next_object_id.saturating_add(1);
    if guard.next_object_id == id {
        guard.next_object_id = id.saturating_add(1);
    }
    guard.open_objects.insert(id, kind);
    id
}

pub(crate) fn unregister_open_object(
    inner: &Arc<Mutex<FileInner<BufReader<fs::File>>>>,
    object_id: u64,
) {
    inner.lock().open_objects.remove(&object_id);
}

struct ResolvedObject {
    inner: Arc<Mutex<FileInner<BufReader<fs::File>>>>,
    path: String,
    addr: u64,
    object_type: ObjectType,
}

enum ResolvedLink {
    Hard {
        addr: u64,
    },
    Soft {
        target: String,
    },
    External {
        filename: String,
        object_path: String,
    },
    UserDefined(u8),
}

impl ResolvedLink {
    fn from_ref(link: crate::hl::group::LinkMessageRef<'_>) -> Result<Self> {
        match link.link_type {
            LinkType::Hard => {
                let addr = link.hard_link_addr.ok_or_else(|| {
                    Error::InvalidFormat(format!(
                        "hard link '{}' is missing object address",
                        link.name
                    ))
                })?;
                Ok(Self::Hard { addr })
            }
            LinkType::Soft => {
                let target = link.soft_link_target.ok_or_else(|| {
                    Error::InvalidFormat(format!(
                        "soft link '{}' is missing target path",
                        link.name
                    ))
                })?;
                Ok(Self::Soft {
                    target: target.to_string(),
                })
            }
            LinkType::External => {
                let (filename, object_path) = link.external_link.ok_or_else(|| {
                    Error::InvalidFormat(format!(
                        "external link '{}' is missing target path",
                        link.name
                    ))
                })?;
                Ok(Self::External {
                    filename: filename.to_string(),
                    object_path: object_path.to_string(),
                })
            }
            LinkType::UserDefined(kind) => Ok(Self::UserDefined(kind)),
        }
    }
}

enum PathStep {
    Resolved(ResolvedObject),
    Descend(Group, String),
    Restart(String),
}

fn canonical_path(path: &str) -> String {
    let mut parts = Vec::new();
    for part in path.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            other => parts.push(other),
        }
    }
    if parts.is_empty() {
        "/".to_string()
    } else {
        format!("/{}", parts.join("/"))
    }
}

fn join_absolute_path(parent: &str, child: &str) -> String {
    if parent == "/" {
        format!("/{child}")
    } else {
        format!("{parent}/{child}")
    }
}

fn append_path_parts(mut base: String, parts: &[&str]) -> String {
    if parts.is_empty() {
        return base;
    }
    base.reserve(parts.iter().map(|part| part.len() + 1).sum());
    for part in parts {
        if base != "/" {
            base.push('/');
        }
        base.push_str(part);
    }
    base
}

fn resolve_soft_target(parent: &str, target: &str, remaining: &[&str]) -> String {
    let base = if target.starts_with('/') {
        canonical_path(target)
    } else {
        canonical_path(&join_absolute_path(parent, target))
    };
    append_path_parts(base, remaining)
}

/// Determine object type from an object header's messages.
pub(crate) fn object_type_from_messages(messages: &[RawMessage]) -> ObjectType {
    let has_dataspace = messages
        .iter()
        .any(|m| m.msg_type == object_header::MSG_DATASPACE);
    let has_layout = messages
        .iter()
        .any(|m| m.msg_type == object_header::MSG_LAYOUT);
    let has_datatype = messages
        .iter()
        .any(|m| m.msg_type == object_header::MSG_DATATYPE);
    let has_stab = messages
        .iter()
        .any(|m| m.msg_type == object_header::MSG_SYMBOL_TABLE);
    let has_link = messages
        .iter()
        .any(|m| m.msg_type == object_header::MSG_LINK);
    let has_link_info = messages
        .iter()
        .any(|m| m.msg_type == object_header::MSG_LINK_INFO);

    if has_layout || (has_dataspace && has_datatype && !has_stab && !has_link && !has_link_info) {
        ObjectType::Dataset
    } else if has_stab || has_link || has_link_info {
        ObjectType::Group
    } else if has_datatype && !has_dataspace {
        ObjectType::NamedDatatype
    } else if messages.is_empty() {
        // Empty object header -- likely an empty group (v2 format)
        ObjectType::Group
    } else {
        ObjectType::Unknown
    }
}

#[cfg(unix)]
fn file_number_from_path(path: &Path) -> Result<u64> {
    use std::os::unix::fs::MetadataExt;

    let metadata = fs::metadata(path)?;
    Ok((metadata.dev() << 32) ^ metadata.ino())
}

#[cfg(not(unix))]
fn file_number_from_path(path: &Path) -> Result<u64> {
    use std::hash::{Hash, Hasher};

    let metadata = fs::metadata(path)?;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    path.hash(&mut hasher);
    metadata.len().hash(&mut hasher);
    Ok(hasher.finish())
}
