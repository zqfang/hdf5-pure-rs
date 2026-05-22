use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileDriverKind {
    Sec2,
    Stdio,
    Core,
    Direct,
}

/// Return the canonical integer id assigned to a [`FileDriverKind`].
fn file_driver_kind_id(kind: FileDriverKind) -> u64 {
    match kind {
        FileDriverKind::Sec2 => 0,
        FileDriverKind::Stdio => 1,
        FileDriverKind::Core => 2,
        FileDriverKind::Direct => 3,
    }
}

#[derive(Debug)]
pub struct LocalFileDriver {
    kind: FileDriverKind,
    path: Option<PathBuf>,
    file: Option<File>,
    core_image: Vec<u8>,
    eoa: u64,
    locked: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreFileConfig {
    pub increment: usize,
    pub backing_store: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectFileConfig {
    pub memory_alignment: u64,
    pub block_size: u64,
    pub copy_buffer_size: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VfdRegistry {
    by_name: HashMap<String, u64>,
    by_value: HashMap<u64, String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VfdMemType {
    Default,
    Super,
    BTree,
    RawData,
    GlobalHeap,
    LocalHeap,
    ObjectHeader,
    Draw,
    Garbage,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VfdIoRequest {
    pub addr: u64,
    pub bytes: Vec<u8>,
}

impl Default for DirectFileConfig {
    fn default() -> Self {
        Self {
            memory_alignment: 4096,
            block_size: 4096,
            copy_buffer_size: 1024 * 1024,
        }
    }
}

impl Default for CoreFileConfig {
    fn default() -> Self {
        Self {
            increment: 64 * 1024,
            backing_store: false,
        }
    }
}

impl LocalFileDriver {
    /// `sec2` VFD: register.
    pub fn sec2_register() -> FileDriverKind {
        FileDriverKind::Sec2
    }

    /// `sec2` VFD: unregister.
    pub fn sec2_unregister() {}

    /// `sec2` VFD: open.
    pub fn sec2_open(path: impl AsRef<Path>, read_write: bool) -> Result<Self> {
        Self::open_file(FileDriverKind::Sec2, path, read_write)
    }

    /// `sec2` VFD: close.
    pub fn sec2_close(self) {}

    /// `sec2` VFD: compare two driver instances.
    pub fn sec2_cmp(&self, other: &Self) -> Ordering {
        self.driver_cmp(other)
    }

    /// `sec2` VFD: query feature flags.
    pub fn sec2_query(&self) -> u64 {
        self.driver_query()
    }

    /// `sec2` VFD: get the end-of-allocation address.
    pub fn sec2_get_eoa(&self) -> u64 {
        self.eoa
    }

    /// `sec2` VFD: set the end-of-allocation address.
    pub fn sec2_set_eoa(&mut self, eoa: u64) {
        self.eoa = eoa;
    }

    /// `sec2` VFD: get the end-of-file address.
    pub fn sec2_get_eof(&mut self) -> Result<u64> {
        self.driver_get_eof()
    }

    /// `sec2` VFD: get the underlying file handle.
    pub fn sec2_get_handle(&self) -> Option<&File> {
        self.file.as_ref()
    }

    /// `sec2` VFD: read bytes from the file.
    pub fn sec2_read(&mut self, addr: u64, buf: &mut [u8]) -> Result<()> {
        self.driver_read(addr, buf)
    }

    /// `sec2` VFD: write bytes to the file.
    pub fn sec2_write(&mut self, addr: u64, data: &[u8]) -> Result<()> {
        self.driver_write(addr, data)
    }

    /// `sec2` VFD: truncate the file to the current EOA.
    pub fn sec2_truncate(&mut self) -> Result<()> {
        self.driver_truncate()
    }

    /// `sec2` VFD: acquire an advisory file lock.
    pub fn sec2_lock(&mut self) {
        self.locked = true;
    }

    /// `sec2` VFD: release an advisory file lock.
    pub fn sec2_unlock(&mut self) {
        self.locked = false;
    }

    /// `sec2` VFD: delete the file.
    pub fn sec2_delete(path: impl AsRef<Path>) -> Result<()> {
        delete_existing(path)
    }

    /// `sec2` VFD: invoke a driver-specific control op.
    pub fn sec2_ctl(&mut self, eoa: Option<u64>) {
        if let Some(eoa) = eoa {
            self.eoa = eoa;
        }
    }

    /// `stdio` VFD: register.
    pub fn stdio_register() -> FileDriverKind {
        FileDriverKind::Stdio
    }

    /// `stdio` VFD: unregister.
    pub fn stdio_unregister() {}

    /// `stdio` VFD: initialize.
    pub fn stdio_init() -> FileDriverKind {
        FileDriverKind::Stdio
    }

    /// `stdio` VFD: open.
    pub fn stdio_open(path: impl AsRef<Path>, read_write: bool) -> Result<Self> {
        Self::open_file(FileDriverKind::Stdio, path, read_write)
    }

    /// `stdio` VFD: close.
    pub fn stdio_close(self) {}

    /// `stdio` VFD: compare two driver instances.
    pub fn stdio_cmp(&self, other: &Self) -> Ordering {
        self.driver_cmp(other)
    }

    /// `stdio` VFD: query feature flags.
    pub fn stdio_query(&self) -> u64 {
        self.driver_query()
    }

    /// `stdio` VFD: allocate space in the file.
    pub fn stdio_alloc(&mut self, size: u64) -> Result<u64> {
        self.driver_alloc(size)
    }

    /// `stdio` VFD: get the end-of-allocation address.
    pub fn stdio_get_eoa(&self) -> u64 {
        self.eoa
    }

    /// `stdio` VFD: set the end-of-allocation address.
    pub fn stdio_set_eoa(&mut self, eoa: u64) {
        self.eoa = eoa;
    }

    /// `stdio` VFD: get the end-of-file address.
    pub fn stdio_get_eof(&mut self) -> Result<u64> {
        self.driver_get_eof()
    }

    /// `stdio` VFD: get the underlying file handle.
    pub fn stdio_get_handle(&self) -> Option<&File> {
        self.file.as_ref()
    }

    /// `stdio` VFD: read bytes from the file.
    pub fn stdio_read(&mut self, addr: u64, buf: &mut [u8]) -> Result<()> {
        self.driver_read(addr, buf)
    }

    /// `stdio` VFD: write bytes to the file.
    pub fn stdio_write(&mut self, addr: u64, data: &[u8]) -> Result<()> {
        self.driver_write(addr, data)
    }

    /// `stdio` VFD: flush buffered writes to disk.
    pub fn stdio_flush(&mut self) -> Result<()> {
        self.driver_flush()
    }

    /// `stdio` VFD: truncate the file to the current EOA.
    pub fn stdio_truncate(&mut self) -> Result<()> {
        self.driver_truncate()
    }

    /// `stdio` VFD: acquire an advisory file lock.
    pub fn stdio_lock(&mut self) {
        self.locked = true;
    }

    /// `stdio` VFD: release an advisory file lock.
    pub fn stdio_unlock(&mut self) {
        self.locked = false;
    }

    /// `stdio` VFD: delete the file.
    pub fn stdio_delete(path: impl AsRef<Path>) -> Result<()> {
        delete_existing(path)
    }

    /// `direct` VFD: register.
    pub fn direct_register() -> FileDriverKind {
        FileDriverKind::Direct
    }

    /// `direct` VFD: unregister.
    pub fn direct_unregister() {}

    /// `direct` VFD: populate the default driver-specific configuration.
    pub fn direct_populate_config() -> DirectFileConfig {
        DirectFileConfig::default()
    }

    /// `direct` VFD: get the driver-specific FAPL configuration.
    pub fn direct_fapl_get(&self) -> Option<DirectFileConfig> {
        (self.kind == FileDriverKind::Direct).then(DirectFileConfig::default)
    }

    /// `direct` VFD: copy the driver-specific FAPL configuration.
    pub fn direct_fapl_copy(config: &DirectFileConfig) -> DirectFileConfig {
        config.clone()
    }

    /// `direct` VFD: open.
    pub fn direct_open(path: impl AsRef<Path>, read_write: bool) -> Result<Self> {
        Self::open_file(FileDriverKind::Direct, path, read_write)
    }

    /// `direct` VFD: check that an I/O request meets alignment requirements.
    pub fn direct_check_alignment_reqs(addr: u64, size: usize, config: &DirectFileConfig) -> bool {
        let Ok(size) = u64::try_from(size) else {
            return false;
        };
        config.memory_alignment != 0
            && config.block_size != 0
            && addr % config.memory_alignment == 0
            && size % config.block_size == 0
    }

    /// `direct` VFD: close.
    pub fn direct_close(self) {}

    /// `direct` VFD: compare two driver instances.
    pub fn direct_cmp(&self, other: &Self) -> Ordering {
        self.driver_cmp(other)
    }

    /// `direct` VFD: query feature flags.
    pub fn direct_query(&self) -> u64 {
        self.driver_query()
    }

    /// `direct` VFD: set the end-of-allocation address.
    pub fn direct_set_eoa(&mut self, eoa: u64) {
        self.eoa = eoa;
    }

    /// `direct` VFD: get the end-of-file address.
    pub fn direct_get_eof(&mut self) -> Result<u64> {
        self.driver_get_eof()
    }

    /// `direct` VFD: get the underlying file handle.
    pub fn direct_get_handle(&self) -> Option<&File> {
        self.file.as_ref()
    }

    /// `direct` VFD: read bytes from the file.
    pub fn direct_read(&mut self, addr: u64, buf: &mut [u8]) -> Result<()> {
        self.driver_read(addr, buf)
    }

    /// `direct` VFD: write bytes to the file.
    pub fn direct_write(&mut self, addr: u64, data: &[u8]) -> Result<()> {
        self.driver_write(addr, data)
    }

    /// `direct` VFD: truncate the file to the current EOA.
    pub fn direct_truncate(&mut self) -> Result<()> {
        self.driver_truncate()
    }

    /// `direct` VFD: acquire an advisory file lock.
    pub fn direct_lock(&mut self) {
        self.locked = true;
    }

    /// `direct` VFD: release an advisory file lock.
    pub fn direct_unlock(&mut self) {
        self.locked = false;
    }

    /// `direct` VFD: delete the file.
    pub fn direct_delete(path: impl AsRef<Path>) -> Result<()> {
        delete_existing(path)
    }

    /// `core` VFD: return the default driver-specific configuration.
    pub fn core_get_default_config() -> CoreFileConfig {
        CoreFileConfig::default()
    }

    /// `core` VFD: register.
    pub fn core_register() -> FileDriverKind {
        FileDriverKind::Core
    }

    /// `core` VFD: unregister.
    pub fn core_unregister() {}

    /// `core` VFD: get the driver-specific FAPL configuration.
    pub fn core_fapl_get(&self) -> Option<CoreFileConfig> {
        (self.kind == FileDriverKind::Core).then(CoreFileConfig::default)
    }

    /// `core` VFD: compare two driver instances.
    pub fn core_cmp(&self, other: &Self) -> Ordering {
        self.driver_cmp(other)
    }

    /// `core` VFD: query feature flags.
    pub fn core_query(&self) -> u64 {
        self.driver_query()
    }

    /// `core` VFD: get the end-of-allocation address.
    pub fn core_get_eoa(&self) -> u64 {
        self.eoa
    }

    /// `core` VFD: set the end-of-allocation address.
    pub fn core_set_eoa(&mut self, eoa: u64) {
        self.eoa = eoa;
        if let Ok(eoa) = usize::try_from(eoa) {
            if self.core_image.len() < eoa {
                self.core_image.resize(eoa, 0);
            }
        }
    }

    /// `core` VFD: get the end-of-file address.
    pub fn core_get_eof(&self) -> u64 {
        self.core_get_eof_checked().unwrap_or(u64::MAX)
    }

    /// `core` VFD: get the end-of-file address with overflow checking.
    pub fn core_get_eof_checked(&self) -> Result<u64> {
        u64::try_from(self.core_image.len())
            .map_err(|_| Error::InvalidFormat("core VFD EOF exceeds u64".into()))
    }

    /// `core` VFD: get the underlying file handle.
    pub fn core_get_handle(&self) -> Option<&[u8]> {
        (self.kind == FileDriverKind::Core).then_some(self.core_image.as_slice())
    }

    /// `core` VFD: read bytes from the file.
    pub fn core_read(&mut self, addr: u64, buf: &mut [u8]) -> Result<()> {
        self.driver_read(addr, buf)
    }

    /// `core` VFD: write bytes to the file.
    pub fn core_write(&mut self, addr: u64, data: &[u8]) -> Result<()> {
        self.driver_write(addr, data)
    }

    /// `core` VFD: flush buffered writes to disk.
    pub fn core_flush(&mut self) -> Result<()> {
        self.driver_flush()
    }

    /// `core` VFD: truncate the file to the current EOA.
    pub fn core_truncate(&mut self) -> Result<()> {
        self.driver_truncate()
    }

    /// `core` VFD: acquire an advisory file lock.
    pub fn core_lock(&mut self) {
        self.locked = true;
    }

    /// `core` VFD: release an advisory file lock.
    pub fn core_unlock(&mut self) {
        self.locked = false;
    }

    /// `core` VFD: delete the file.
    pub fn core_delete(&mut self) {
        self.core_image.clear();
        self.eoa = 0;
    }

    /// `core` VFD: mark a byte range dirty for later writeback.
    pub fn core_add_dirty_region(&mut self, addr: u64, size: u64) -> Result<()> {
        let end = addr
            .checked_add(size)
            .ok_or_else(|| Error::InvalidFormat("core VFD dirty region overflow".into()))?;
        if end > self.eoa {
            self.core_set_eoa(end);
        }
        Ok(())
    }

    /// `core` VFD: clear the in-memory dirty region list.
    pub fn core_destroy_dirty_list(&mut self) {}

    /// `core` VFD: write a buffer to the backing store.
    pub fn core_write_to_bstore(&mut self, addr: u64, data: &[u8]) -> Result<()> {
        self.core_write(addr, data)
    }

    /// VFD: allocate space in the file.
    pub fn alloc(&mut self, size: u64) -> Result<u64> {
        self.driver_alloc(size)
    }

    /// VFD: free a previously allocated region.
    pub fn free(&mut self, _addr: u64, _size: u64) {}

    /// VFD: try to extend the file by `extra` bytes.
    pub fn try_extend(&mut self, addr: u64, old_size: u64, extra: u64) -> Result<bool> {
        let end = addr
            .checked_add(old_size)
            .and_then(|v| v.checked_add(extra))
            .ok_or_else(|| Error::InvalidFormat("VFD extension overflow".into()))?;
        if end > self.eoa {
            self.eoa = end;
        }
        Ok(true)
    }

    /// Internal helper: open the on-disk file for the given driver kind.
    fn open_file(kind: FileDriverKind, path: impl AsRef<Path>, read_write: bool) -> Result<Self> {
        let mut options = OpenOptions::new();
        options.read(true);
        if read_write {
            options.write(true).create(true);
        }
        let file = options.open(path.as_ref())?;
        let eoa = file.metadata()?.len();
        Ok(Self {
            kind,
            path: Some(path.as_ref().to_path_buf()),
            file: Some(file),
            core_image: Vec::new(),
            eoa,
            locked: false,
        })
    }

    /// Dispatch [`cmp`] to the active driver implementation.
    fn driver_cmp(&self, other: &Self) -> Ordering {
        self.kind
            .cmp(&other.kind)
            .then_with(|| self.path.cmp(&other.path))
    }

    /// Dispatch [`query`] to the active driver implementation.
    fn driver_query(&self) -> u64 {
        match self.kind {
            FileDriverKind::Sec2 | FileDriverKind::Stdio | FileDriverKind::Direct => 0x01,
            FileDriverKind::Core => 0x03,
        }
    }

    /// Dispatch [`alloc`] to the active driver implementation.
    fn driver_alloc(&mut self, size: u64) -> Result<u64> {
        let addr = self.eoa;
        self.eoa = self
            .eoa
            .checked_add(size)
            .ok_or_else(|| Error::InvalidFormat("VFD allocation overflow".into()))?;
        if self.kind == FileDriverKind::Core {
            self.core_set_eoa(self.eoa);
        }
        Ok(addr)
    }

    /// Dispatch [`get_eof`] to the active driver implementation.
    fn driver_get_eof(&mut self) -> Result<u64> {
        if self.kind == FileDriverKind::Core {
            return Ok(self.core_get_eof());
        }
        let file = self
            .file
            .as_ref()
            .ok_or_else(|| Error::InvalidFormat("VFD file handle is closed".into()))?;
        Ok(file.metadata()?.len())
    }

    /// Dispatch [`read`] to the active driver implementation.
    fn driver_read(&mut self, addr: u64, buf: &mut [u8]) -> Result<()> {
        if self.kind == FileDriverKind::Core {
            let start = usize::try_from(addr)
                .map_err(|_| Error::InvalidFormat("core VFD read address too large".into()))?;
            let end = start
                .checked_add(buf.len())
                .ok_or_else(|| Error::InvalidFormat("core VFD read span overflow".into()))?;
            if end > self.core_image.len() {
                return Err(Error::InvalidFormat("core VFD read past EOF".into()));
            }
            buf.copy_from_slice(&self.core_image[start..end]);
            return Ok(());
        }
        let file = self
            .file
            .as_mut()
            .ok_or_else(|| Error::InvalidFormat("VFD file handle is closed".into()))?;
        file.seek(SeekFrom::Start(addr))?;
        file.read_exact(buf)?;
        Ok(())
    }

    /// Dispatch [`write`] to the active driver implementation.
    fn driver_write(&mut self, addr: u64, data: &[u8]) -> Result<()> {
        let data_len = u64::try_from(data.len())
            .map_err(|_| Error::InvalidFormat("VFD write length exceeds u64".into()))?;
        let end = addr
            .checked_add(data_len)
            .ok_or_else(|| Error::InvalidFormat("VFD write span overflow".into()))?;
        if self.kind == FileDriverKind::Core {
            let start = usize::try_from(addr)
                .map_err(|_| Error::InvalidFormat("core VFD write address too large".into()))?;
            let end_usize = usize::try_from(end)
                .map_err(|_| Error::InvalidFormat("core VFD write end too large".into()))?;
            if self.core_image.len() < end_usize {
                self.core_image.resize(end_usize, 0);
            }
            self.core_image[start..end_usize].copy_from_slice(data);
        } else {
            let file = self
                .file
                .as_mut()
                .ok_or_else(|| Error::InvalidFormat("VFD file handle is closed".into()))?;
            file.seek(SeekFrom::Start(addr))?;
            file.write_all(data)?;
        }
        if end > self.eoa {
            self.eoa = end;
        }
        Ok(())
    }

    /// Dispatch [`flush`] to the active driver implementation.
    fn driver_flush(&mut self) -> Result<()> {
        if let Some(file) = self.file.as_mut() {
            file.flush()?;
        }
        Ok(())
    }

    /// Dispatch [`truncate`] to the active driver implementation.
    fn driver_truncate(&mut self) -> Result<()> {
        if self.kind == FileDriverKind::Core {
            let eoa = usize::try_from(self.eoa)
                .map_err(|_| Error::InvalidFormat("core VFD EOA too large".into()))?;
            self.core_image.truncate(eoa);
            return Ok(());
        }
        let file = self
            .file
            .as_mut()
            .ok_or_else(|| Error::InvalidFormat("VFD file handle is closed".into()))?;
        file.set_len(self.eoa)?;
        Ok(())
    }
}

impl Ord for FileDriverKind {
    /// Compare two driver instances by kind then path/state.
    fn cmp(&self, other: &Self) -> Ordering {
        (*self as u8).cmp(&(*other as u8))
    }
}

impl PartialOrd for FileDriverKind {
    /// Total-order partial comparison delegating to [`cmp`].
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Delete the file at `path` if it exists, ignoring not-found errors.
fn delete_existing(path: impl AsRef<Path>) -> Result<()> {
    match fs::remove_file(path.as_ref()) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err.into()),
    }
}

/// Render a [`VfdMemType`] as its short name.
#[allow(non_snake_case)]
pub fn H5FD__mem_t_to_str(mem_type: VfdMemType) -> &'static str {
    match mem_type {
        VfdMemType::Default => "H5FD_MEM_DEFAULT",
        VfdMemType::Super => "H5FD_MEM_SUPER",
        VfdMemType::BTree => "H5FD_MEM_BTREE",
        VfdMemType::RawData => "H5FD_MEM_DRAW",
        VfdMemType::GlobalHeap => "H5FD_MEM_GHEAP",
        VfdMemType::LocalHeap => "H5FD_MEM_LHEAP",
        VfdMemType::ObjectHeader => "H5FD_MEM_OHDR",
        VfdMemType::Draw => "H5FD_MEM_DRAW",
        VfdMemType::Garbage => "H5FD_MEM_NTYPES",
    }
}

/// Initialize an empty VFD registry.
#[allow(non_snake_case)]
pub fn H5FD_init() -> VfdRegistry {
    H5FD__init_package()
}

/// Initialize the VFD package and pre-register the built-in drivers.
#[allow(non_snake_case)]
pub fn H5FD__init_package() -> VfdRegistry {
    let mut registry = VfdRegistry::default();
    H5FD_register_driver_by_name(
        &mut registry,
        "sec2",
        file_driver_kind_id(FileDriverKind::Sec2),
    );
    H5FD_register_driver_by_name(
        &mut registry,
        "stdio",
        file_driver_kind_id(FileDriverKind::Stdio),
    );
    H5FD_register_driver_by_name(
        &mut registry,
        "core",
        file_driver_kind_id(FileDriverKind::Core),
    );
    H5FD_register_driver_by_name(
        &mut registry,
        "direct",
        file_driver_kind_id(FileDriverKind::Direct),
    );
    registry
}

/// Tear down the VFD registry, dropping all driver registrations.
#[allow(non_snake_case)]
pub fn H5FD_term_package(registry: &mut VfdRegistry) {
    registry.by_name.clear();
    registry.by_value.clear();
}

/// Free a registered driver class by name.
#[allow(non_snake_case)]
pub fn H5FD__free_cls(registry: &mut VfdRegistry, name: &str) -> Option<u64> {
    H5FDunregister(registry, name)
}

/// Public API: register a VFD driver by name and value.
#[allow(non_snake_case)]
pub fn H5FDregister(registry: &mut VfdRegistry, name: &str, value: u64) -> u64 {
    H5FD_register_driver_by_name(registry, name, value)
}

/// Register a VFD driver by name and value.
#[allow(non_snake_case)]
pub fn H5FD_register(registry: &mut VfdRegistry, name: &str, value: u64) -> u64 {
    H5FDregister(registry, name, value)
}

/// Public API: unregister a VFD driver by name.
#[allow(non_snake_case)]
pub fn H5FDunregister(registry: &mut VfdRegistry, name: &str) -> Option<u64> {
    let value = registry.by_name.remove(name)?;
    registry.by_value.remove(&value);
    Some(value)
}

/// Register a VFD driver by name.
#[allow(non_snake_case)]
pub fn H5FD_register_driver_by_name(registry: &mut VfdRegistry, name: &str, value: u64) -> u64 {
    registry.by_name.insert(name.to_string(), value);
    registry.by_value.insert(value, name.to_string());
    value
}

/// Register a VFD driver by numeric value, providing its name.
#[allow(non_snake_case)]
pub fn H5FD_register_driver_by_value(registry: &mut VfdRegistry, value: u64, name: &str) -> u64 {
    H5FD_register_driver_by_name(registry, name, value)
}

/// Return whether a driver is currently registered under the given name.
#[allow(non_snake_case)]
pub fn H5FD_is_driver_registered_by_name(registry: &VfdRegistry, name: &str) -> bool {
    registry.by_name.contains_key(name)
}

/// Return the value/id of a driver registered under the given name.
#[allow(non_snake_case)]
pub fn H5FD_get_driver_id_by_name(registry: &VfdRegistry, name: &str) -> Option<u64> {
    registry.by_name.get(name).copied()
}

/// Look up the name of a driver registered with the given value.
#[allow(non_snake_case)]
pub fn H5FD_get_driver_id_by_value(registry: &VfdRegistry, value: u64) -> Option<&str> {
    registry.by_value.get(&value).map(String::as_str)
}

/// Look up the class name of a driver registered with the given value.
#[allow(non_snake_case)]
pub fn H5FD_get_class(registry: &VfdRegistry, value: u64) -> Option<&str> {
    H5FD_get_driver_id_by_value(registry, value)
}

/// Driver-lookup callback used during package initialization.
#[allow(non_snake_case)]
pub fn H5FD__get_driver_cb(registry: &VfdRegistry, name: &str) -> Option<u64> {
    H5FD_get_driver_id_by_name(registry, name)
}

/// Superblock extension size required for a given driver kind.
#[allow(non_snake_case)]
pub fn H5FD_sb_size(_driver: FileDriverKind) -> usize {
    0
}

/// Return the kind of driver currently in use by the FAPL.
#[allow(non_snake_case)]
pub fn H5FD_fapl_get(driver: &LocalFileDriver) -> FileDriverKind {
    driver.kind
}

/// Public API: open a file using a given driver kind.
#[allow(non_snake_case)]
pub fn H5FDopen(
    path: impl AsRef<Path>,
    kind: FileDriverKind,
    read_write: bool,
) -> Result<LocalFileDriver> {
    H5FD_open(path, kind, read_write)
}

/// Open a file using a given driver kind.
#[allow(non_snake_case)]
pub fn H5FD_open(
    path: impl AsRef<Path>,
    kind: FileDriverKind,
    read_write: bool,
) -> Result<LocalFileDriver> {
    match kind {
        FileDriverKind::Sec2 => LocalFileDriver::sec2_open(path, read_write),
        FileDriverKind::Stdio => LocalFileDriver::stdio_open(path, read_write),
        FileDriverKind::Direct => LocalFileDriver::direct_open(path, read_write),
        FileDriverKind::Core => Err(Error::Unsupported(
            "core VFD open from path is represented by in-memory driver state".into(),
        )),
    }
}

/// Public API: close a driver instance.
#[allow(non_snake_case)]
pub fn H5FDclose(driver: LocalFileDriver) {
    H5FD_close(driver);
}

/// Close a driver instance.
#[allow(non_snake_case)]
pub fn H5FD_close(_driver: LocalFileDriver) {}

/// Public API: compare two driver instances.
#[allow(non_snake_case)]
pub fn H5FDcmp(left: &LocalFileDriver, right: &LocalFileDriver) -> Ordering {
    H5FD_cmp(left, right)
}

/// Compare two driver instances.
#[allow(non_snake_case)]
pub fn H5FD_cmp(left: &LocalFileDriver, right: &LocalFileDriver) -> Ordering {
    left.driver_cmp(right)
}

/// Public API: query the feature flags reported by a driver.
#[allow(non_snake_case)]
pub fn H5FDquery(driver: &LocalFileDriver) -> u64 {
    H5FD_get_feature_flags(driver)
}

/// Public API: allocate `size` bytes in the file.
#[allow(non_snake_case)]
pub fn H5FDalloc(driver: &mut LocalFileDriver, size: u64) -> Result<u64> {
    H5FD_alloc(driver, size)
}

/// Internal VFD API: allocate `size` bytes in the file.
#[allow(non_snake_case)]
pub fn H5FD_alloc(driver: &mut LocalFileDriver, size: u64) -> Result<u64> {
    driver.alloc(size)
}

/// Public API: free a region previously allocated in the file.
#[allow(non_snake_case)]
pub fn H5FDfree(driver: &mut LocalFileDriver, addr: u64, size: u64) {
    H5FD_free(driver, addr, size);
}

/// Internal VFD API: free a region previously allocated in the file.
#[allow(non_snake_case)]
pub fn H5FD_free(driver: &mut LocalFileDriver, addr: u64, size: u64) {
    driver.free(addr, size);
}

/// Internal VFD API: attempt to extend an existing allocation.
#[allow(non_snake_case)]
pub fn H5FD_try_extend(
    driver: &mut LocalFileDriver,
    addr: u64,
    old_size: u64,
    extra: u64,
) -> Result<bool> {
    driver.try_extend(addr, old_size, extra)
}

/// Public API: return the current EOA.
#[allow(non_snake_case)]
pub fn H5FDget_eoa(driver: &LocalFileDriver) -> u64 {
    driver.eoa
}

/// Public API: set the current EOA.
#[allow(non_snake_case)]
pub fn H5FDset_eoa(driver: &mut LocalFileDriver, eoa: u64) {
    driver.eoa = eoa;
}

/// Public API: return the current EOF.
#[allow(non_snake_case)]
pub fn H5FDget_eof(driver: &mut LocalFileDriver) -> Result<u64> {
    driver.driver_get_eof()
}

/// Return the maximum addressable file offset for the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD_get_maxaddr() -> u64 {
    u64::MAX
}

/// Return the feature flags reported by a driver.
#[allow(non_snake_case)]
pub fn H5FD_get_feature_flags(driver: &LocalFileDriver) -> u64 {
    driver.driver_query()
}

/// Public API: query feature flags for a registered driver id before opening a file.
#[allow(non_snake_case)]
pub fn H5FDdriver_query(registry: &VfdRegistry, driver_id: u64) -> Result<u64> {
    if !registry.by_value.contains_key(&driver_id) {
        return Err(Error::InvalidFormat(format!(
            "VFD driver id {driver_id} is not registered"
        )));
    }
    match driver_id {
        id if id == file_driver_kind_id(FileDriverKind::Sec2) => Ok(0x01),
        id if id == file_driver_kind_id(FileDriverKind::Stdio) => Ok(0x01),
        id if id == file_driver_kind_id(FileDriverKind::Direct) => Ok(0x01),
        id if id == file_driver_kind_id(FileDriverKind::Core) => Ok(0x03),
        _ => Err(unsupported_vfd_driver("custom registered")),
    }
}

/// Setting feature flags is a no-op in the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD_set_feature_flags(_driver: &mut LocalFileDriver, _flags: u64) {}

/// Public API: read bytes from the file.
#[allow(non_snake_case)]
#[deprecated(note = "use driver-specific read methods")]
pub fn H5FDread(driver: &mut LocalFileDriver, addr: u64, buf: &mut [u8]) -> Result<()> {
    driver.driver_read(addr, buf)
}

/// Public API: write bytes to the file.
#[allow(non_snake_case)]
#[deprecated(note = "use driver-specific write methods")]
pub fn H5FDwrite(driver: &mut LocalFileDriver, addr: u64, data: &[u8]) -> Result<()> {
    driver.driver_write(addr, data)
}

/// Read a list of (addr, buffer) requests sequentially.
#[allow(non_snake_case)]
pub fn H5FDread_vector(driver: &mut LocalFileDriver, requests: &mut [VfdIoRequest]) -> Result<()> {
    H5FD_read_vector_from_selection(driver, requests)
}

/// Write a list of (addr, bytes) requests sequentially.
#[allow(non_snake_case)]
pub fn H5FDwrite_vector(driver: &mut LocalFileDriver, requests: &[VfdIoRequest]) -> Result<()> {
    H5FD_write_vector_from_selection(driver, requests)
}

/// Read a list of selection-based requests sequentially.
#[allow(non_snake_case)]
pub fn H5FDread_selection(
    driver: &mut LocalFileDriver,
    requests: &mut [VfdIoRequest],
) -> Result<()> {
    H5FD_read_from_selection(driver, requests)
}

/// Write a list of selection-based requests sequentially.
#[allow(non_snake_case)]
pub fn H5FDwrite_selection(driver: &mut LocalFileDriver, requests: &[VfdIoRequest]) -> Result<()> {
    H5FD_write_selection(driver, requests)
}

/// Public API: flush buffered writes to disk.
#[allow(non_snake_case)]
pub fn H5FDflush(driver: &mut LocalFileDriver) -> Result<()> {
    driver.driver_flush()
}

/// Public API: truncate the file to the current end-of-allocation address.
#[allow(non_snake_case)]
pub fn H5FDtruncate(driver: &mut LocalFileDriver) -> Result<()> {
    driver.driver_truncate()
}

/// Public API: acquire a VFD lock.
#[allow(non_snake_case)]
pub fn H5FDlock(driver: &mut LocalFileDriver, _read_write: bool) -> Result<()> {
    driver.locked = true;
    Ok(())
}

/// Public API: release a VFD lock.
#[allow(non_snake_case)]
pub fn H5FDunlock(driver: &mut LocalFileDriver) -> Result<()> {
    driver.locked = false;
    Ok(())
}

/// Return the file path of the underlying file, if any.
#[allow(non_snake_case)]
pub fn H5FD_get_fileno(driver: &LocalFileDriver) -> Option<&Path> {
    driver.path.as_deref()
}

/// Public API: return the underlying file handle, if any.
#[allow(non_snake_case)]
pub fn H5FDget_vfd_handle(driver: &LocalFileDriver) -> Option<&File> {
    driver.file.as_ref()
}

/// Toggling paged aggregation is a no-op in the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD_set_paged_aggr(_driver: &mut LocalFileDriver, _enabled: bool) {}

/// Locate the HDF5 superblock signature within an image buffer.
#[allow(non_snake_case)]
pub fn H5FD_locate_signature(image: &[u8]) -> Option<u64> {
    const SIG: &[u8; 8] = b"\x89HDF\r\n\x1a\n";
    let mut offset = 0usize;
    while offset.checked_add(SIG.len())? <= image.len() {
        if image.get(offset..offset.checked_add(SIG.len())?) == Some(SIG.as_slice()) {
            return u64::try_from(offset).ok();
        }
        offset = if offset == 0 {
            512
        } else {
            offset.checked_mul(2)?
        };
    }
    None
}

/// Translate selection requests into a flat vector form.
#[allow(non_snake_case)]
pub fn H5FD__read_selection_translate_view(requests: &[VfdIoRequest]) -> &[VfdIoRequest] {
    requests
}

/// Translate selection requests into a caller-owned vector.
#[allow(non_snake_case)]
pub fn H5FD__read_selection_translate_into(requests: &[VfdIoRequest], out: &mut Vec<VfdIoRequest>) {
    out.extend_from_slice(requests);
}

/// Translate selection requests into a flat vector form.
#[allow(non_snake_case)]
#[deprecated(
    note = "use H5FD__read_selection_translate_view or H5FD__read_selection_translate_into"
)]
pub fn H5FD__read_selection_translate(requests: &[VfdIoRequest]) -> Vec<VfdIoRequest> {
    let mut out = Vec::with_capacity(requests.len());
    H5FD__read_selection_translate_into(requests, &mut out);
    out
}

/// Translate selection write requests into a flat vector form.
#[allow(non_snake_case)]
pub fn H5FD__write_selection_translate_view(requests: &[VfdIoRequest]) -> &[VfdIoRequest] {
    requests
}

/// Translate selection write requests into a caller-owned vector.
#[allow(non_snake_case)]
pub fn H5FD__write_selection_translate_into(
    requests: &[VfdIoRequest],
    out: &mut Vec<VfdIoRequest>,
) {
    out.extend_from_slice(requests);
}

/// Translate selection write requests into a flat vector form.
#[allow(non_snake_case)]
#[deprecated(
    note = "use H5FD__write_selection_translate_view or H5FD__write_selection_translate_into"
)]
pub fn H5FD__write_selection_translate(requests: &[VfdIoRequest]) -> Vec<VfdIoRequest> {
    let mut out = Vec::with_capacity(requests.len());
    H5FD__write_selection_translate_into(requests, &mut out);
    out
}

/// Write a list of selection requests via the active driver.
#[allow(non_snake_case)]
pub fn H5FD_write_selection(driver: &mut LocalFileDriver, requests: &[VfdIoRequest]) -> Result<()> {
    for request in requests {
        driver.driver_write(request.addr, &request.bytes)?;
    }
    Ok(())
}

/// Write a selection request identified by id.
#[allow(non_snake_case)]
pub fn H5FD_write_selection_id(
    driver: &mut LocalFileDriver,
    _selection_id: u64,
    requests: &[VfdIoRequest],
) -> Result<()> {
    H5FD_write_selection(driver, requests)
}

/// Build a vector read request set from a selection.
#[allow(non_snake_case)]
pub fn H5FD_read_vector_from_selection(
    driver: &mut LocalFileDriver,
    requests: &mut [VfdIoRequest],
) -> Result<()> {
    for request in requests {
        driver.driver_read(request.addr, &mut request.bytes)?;
    }
    Ok(())
}

/// Build a vector write request set from a selection.
#[allow(non_snake_case)]
pub fn H5FD_write_vector_from_selection(
    driver: &mut LocalFileDriver,
    requests: &[VfdIoRequest],
) -> Result<()> {
    H5FD_write_selection(driver, requests)
}

/// Read directly from a selection request set.
#[allow(non_snake_case)]
pub fn H5FD_read_from_selection(
    driver: &mut LocalFileDriver,
    requests: &mut [VfdIoRequest],
) -> Result<()> {
    H5FD_read_vector_from_selection(driver, requests)
}

/// Write directly from a selection request set.
#[allow(non_snake_case)]
pub fn H5FD_write_from_selection(
    driver: &mut LocalFileDriver,
    requests: &[VfdIoRequest],
) -> Result<()> {
    H5FD_write_selection(driver, requests)
}

/// Public API wrapper for [`H5FD_read_vector_from_selection`].
#[allow(non_snake_case)]
pub fn H5FDread_vector_from_selection(
    driver: &mut LocalFileDriver,
    requests: &mut [VfdIoRequest],
) -> Result<()> {
    H5FD_read_vector_from_selection(driver, requests)
}

/// Public API wrapper for [`H5FD_write_vector_from_selection`].
#[allow(non_snake_case)]
pub fn H5FDwrite_vector_from_selection(
    driver: &mut LocalFileDriver,
    requests: &[VfdIoRequest],
) -> Result<()> {
    H5FD_write_vector_from_selection(driver, requests)
}

/// Public API wrapper for [`H5FD_read_from_selection`].
#[allow(non_snake_case)]
pub fn H5FDread_from_selection(
    driver: &mut LocalFileDriver,
    requests: &mut [VfdIoRequest],
) -> Result<()> {
    H5FD_read_from_selection(driver, requests)
}

/// Public API wrapper for [`H5FD_write_from_selection`].
#[allow(non_snake_case)]
pub fn H5FDwrite_from_selection(
    driver: &mut LocalFileDriver,
    requests: &[VfdIoRequest],
) -> Result<()> {
    H5FD_write_from_selection(driver, requests)
}

/// Comparator that sorts I/O requests by file address.
#[allow(non_snake_case)]
pub fn H5FD__srt_tmp_cmp(left: &VfdIoRequest, right: &VfdIoRequest) -> Ordering {
    left.addr
        .cmp(&right.addr)
        .then_with(|| left.bytes.len().cmp(&right.bytes.len()))
}

/// Internal helper that sorts a request slice by address.
#[allow(non_snake_case)]
pub fn H5FD__sort_io_req_real(requests: &mut [VfdIoRequest]) {
    requests.sort_by(H5FD__srt_tmp_cmp);
}

/// Sort a vector I/O request set by address.
#[allow(non_snake_case)]
pub fn H5FD_sort_vector_io_req(requests: &mut [VfdIoRequest]) {
    H5FD__sort_io_req_real(requests);
}

/// Sort a selection I/O request set by address.
#[allow(non_snake_case)]
pub fn H5FD_sort_selection_io_req(requests: &mut [VfdIoRequest]) {
    H5FD__sort_io_req_real(requests);
}

/// MPI: getting the rank is not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD_mpi_get_rank() -> Result<u32> {
    Err(Error::Unsupported(
        "MPI VFD is intentionally unsupported; use rayon for local parallelism".into(),
    ))
}

/// MPI: getting the communicator is not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD_mpi_get_comm() -> Result<()> {
    Err(Error::Unsupported(
        "MPI communicator is unavailable in pure-Rust non-MPI mode".into(),
    ))
}

/// MPI: getting the info object is not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD_mpi_get_info() -> Result<()> {
    Err(Error::Unsupported(
        "MPI info is unavailable in pure-Rust non-MPI mode".into(),
    ))
}

/// Convert an `MPI_Offset` to an HDF5 address with sign checks.
#[allow(non_snake_case)]
pub fn H5FD_mpi_MPIOff_to_haddr(offset: i64) -> Option<u64> {
    u64::try_from(offset).ok()
}

/// Convert an HDF5 address to an `MPI_Offset`, rejecting overflow.
#[allow(non_snake_case)]
pub fn H5FD_mpi_haddr_to_MPIOff(addr: u64) -> Option<i64> {
    i64::try_from(addr).ok()
}

/// In the pure-Rust backend file sync is never required.
#[allow(non_snake_case)]
pub fn H5FD_mpi_get_file_sync_required() -> bool {
    false
}

/// MPI: waiting for the left neighbor is not supported.
#[allow(non_snake_case)]
pub fn H5FD_mpio_wait_for_left_neighbor() -> Result<()> {
    Err(Error::Unsupported(
        "MPI synchronization is intentionally unsupported".into(),
    ))
}

/// MPI: signalling the right neighbor is not supported.
#[allow(non_snake_case)]
pub fn H5FD_mpio_signal_right_neighbor() -> Result<()> {
    Err(Error::Unsupported(
        "MPI synchronization is intentionally unsupported".into(),
    ))
}

/// `H5FD__mpio_register`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__mpio_register() -> Result<()> {
    Err(unsupported_vfd_driver("MPIO"))
}

/// `H5FD__mpio_unregister`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__mpio_unregister() {}

/// `H5FD__mpio_init`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__mpio_init() -> Result<()> {
    Err(unsupported_vfd_driver("MPIO"))
}

/// `H5FD__mpio_term`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__mpio_term() {}

/// MPI: setting atomicity is not supported.
#[allow(non_snake_case)]
pub fn H5FD_set_mpio_atomicity(_atomicity: bool) -> Result<()> {
    Err(unsupported_vfd_driver("MPIO atomicity"))
}

/// MPI: getting atomicity is not supported.
#[allow(non_snake_case)]
pub fn H5FD_get_mpio_atomicity() -> Result<bool> {
    Err(unsupported_vfd_driver("MPIO atomicity"))
}

/// `H5FD__mpio_open`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__mpio_open(_path: &str) -> Result<()> {
    Err(unsupported_vfd_driver("MPIO"))
}

/// `H5FD__mpio_close`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__mpio_close() {}

/// `H5FD__mpio_query`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__mpio_query() -> u64 {
    0
}

/// `H5FD__mpio_get_eoa`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__mpio_get_eoa() -> Result<u64> {
    Err(unsupported_vfd_driver("MPIO"))
}

/// `H5FD__mpio_set_eoa`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__mpio_set_eoa(_eoa: u64) -> Result<()> {
    Err(unsupported_vfd_driver("MPIO"))
}

/// `H5FD__mpio_get_eof`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__mpio_get_eof() -> Result<u64> {
    Err(unsupported_vfd_driver("MPIO"))
}

/// `H5FD__mpio_get_handle`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__mpio_get_handle() -> Result<()> {
    Err(unsupported_vfd_driver("MPIO"))
}

/// `H5FD__mpio_read`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__mpio_read(_addr: u64, _buf: &mut [u8]) -> Result<()> {
    Err(unsupported_vfd_driver("MPIO"))
}

/// `H5FD__mpio_write`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__mpio_write(_addr: u64, _data: &[u8]) -> Result<()> {
    Err(unsupported_vfd_driver("MPIO"))
}

/// `H5FD__mpio_vector_build_types`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__mpio_vector_build_types_view(requests: &[VfdIoRequest]) -> &[VfdIoRequest] {
    requests
}

/// `H5FD__mpio_vector_build_types`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__mpio_vector_build_types_into(requests: &[VfdIoRequest], out: &mut Vec<VfdIoRequest>) {
    out.extend_from_slice(requests);
}

/// `H5FD__mpio_vector_build_types`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
#[deprecated(note = "use H5FD__mpio_vector_build_types_view or H5FD__mpio_vector_build_types_into")]
pub fn H5FD__mpio_vector_build_types(requests: &[VfdIoRequest]) -> Vec<VfdIoRequest> {
    let mut out = Vec::with_capacity(requests.len());
    H5FD__mpio_vector_build_types_into(requests, &mut out);
    out
}

/// VFD: selection build types.
#[allow(non_snake_case)]
pub fn H5FD__selection_build_types_view(requests: &[VfdIoRequest]) -> &[VfdIoRequest] {
    requests
}

/// VFD: selection build types.
#[allow(non_snake_case)]
pub fn H5FD__selection_build_types_into(requests: &[VfdIoRequest], out: &mut Vec<VfdIoRequest>) {
    out.extend_from_slice(requests);
}

/// VFD: selection build types.
#[allow(non_snake_case)]
#[deprecated(note = "use H5FD__selection_build_types_view or H5FD__selection_build_types_into")]
pub fn H5FD__selection_build_types(requests: &[VfdIoRequest]) -> Vec<VfdIoRequest> {
    let mut out = Vec::with_capacity(requests.len());
    H5FD__selection_build_types_into(requests, &mut out);
    out
}

/// `H5FD__mpio_read_vector`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__mpio_read_vector(_requests: &mut [VfdIoRequest]) -> Result<()> {
    Err(unsupported_vfd_driver("MPIO"))
}

/// `H5FD__mpio_write_vector`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__mpio_write_vector(_requests: &[VfdIoRequest]) -> Result<()> {
    Err(unsupported_vfd_driver("MPIO"))
}

/// `H5FD__mpio_read_selection`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__mpio_read_selection(_requests: &mut [VfdIoRequest]) -> Result<()> {
    Err(unsupported_vfd_driver("MPIO"))
}

/// `H5FD__mpio_write_selection`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__mpio_write_selection(_requests: &[VfdIoRequest]) -> Result<()> {
    Err(unsupported_vfd_driver("MPIO"))
}

/// `H5FD__mpio_flush`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__mpio_flush() -> Result<()> {
    Err(unsupported_vfd_driver("MPIO"))
}

/// `H5FD__mpio_truncate`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__mpio_truncate() -> Result<()> {
    Err(unsupported_vfd_driver("MPIO"))
}

/// `H5FD__mpio_delete`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__mpio_delete(_path: &str) -> Result<()> {
    Err(unsupported_vfd_driver("MPIO"))
}

/// `H5FD__mpio_ctl`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__mpio_ctl(_opcode: u64) -> Result<()> {
    Err(unsupported_vfd_driver("MPIO"))
}

/// Build an [`Error::Unsupported`] for an unimplemented VFD driver.
fn unsupported_vfd_driver(driver: &str) -> Error {
    Error::Unsupported(format!(
        "{driver} VFD is not implemented in pure-Rust local-only mode"
    ))
}

/// Read a little-endian `u32` from `data` at `offset` with bounds checks.
fn read_le_u32_at(data: &[u8], offset: usize, context: &str) -> Result<u32> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))?;
    let bytes: [u8; 4] = data
        .get(offset..end)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))?
        .try_into()
        .map_err(|_| Error::InvalidFormat(format!("{context} is truncated")))?;
    Ok(u32::from_le_bytes(bytes))
}

/// Read a little-endian `u32` length field as a `usize`.
fn read_le_u32_len_at(data: &[u8], offset: usize, context: &'static str) -> Result<usize> {
    usize::try_from(read_le_u32_at(data, offset, context)?)
        .map_err(|_| Error::InvalidFormat(format!("{context} does not fit in usize")))
}

/// Read a little-endian `u64` from `data` at `offset` with bounds checks.
fn read_le_u64_at(data: &[u8], offset: usize, context: &str) -> Result<u64> {
    let end = offset
        .checked_add(8)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))?;
    let bytes: [u8; 8] = data
        .get(offset..end)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))?
        .try_into()
        .map_err(|_| Error::InvalidFormat(format!("{context} is truncated")))?;
    Ok(u64::from_le_bytes(bytes))
}

fn validate_config_string(value: &str) -> bool {
    !value.is_empty() && !value.as_bytes().contains(&0)
}

fn validate_optional_config_string(value: Option<&str>) -> bool {
    value.is_none_or(validate_config_string)
}

/// `H5FD__hdfs_register`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__hdfs_register() -> Result<()> {
    Err(unsupported_vfd_driver("HDFS"))
}

/// `H5FD__hdfs_unregister`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__hdfs_unregister() {}

/// `H5FD__hdfs_init`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__hdfs_init() -> Result<()> {
    Err(unsupported_vfd_driver("HDFS"))
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HdfsConfig {
    pub namenode_name: String,
    pub namenode_port: u16,
    pub user_name: String,
    pub buffer_size: u32,
}

const HDFS_FAPL_VERSION: u32 = 1;
const HDFS_STRING_SPACE: usize = 128;
const HDFS_STRING_FIELD_SIZE: usize = HDFS_STRING_SPACE + 1;
const HDFS_CONFIG_IMAGE_SIZE: usize =
    4 + HDFS_STRING_FIELD_SIZE + 4 + HDFS_STRING_FIELD_SIZE + HDFS_STRING_FIELD_SIZE + 4;

/// `H5FD__hdfs_validate_config`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__hdfs_validate_config(config: &HdfsConfig) -> bool {
    validate_config_string(&config.namenode_name)
        && validate_config_string(&config.user_name)
        && hdfs_fixed_string_is_valid(&config.namenode_name)
        && hdfs_fixed_string_is_valid(&config.user_name)
}

/// `H5FD__hdfs_sb_size`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__hdfs_sb_size(_config: &HdfsConfig) -> usize {
    HDFS_CONFIG_IMAGE_SIZE
}

/// `H5FD__hdfs_sb_encode`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__hdfs_sb_encode_into(config: &HdfsConfig, out: &mut Vec<u8>) -> Result<()> {
    if !H5FD__hdfs_validate_config(config) {
        return Err(Error::InvalidFormat("invalid HDFS VFD config".into()));
    }
    out.reserve(H5FD__hdfs_sb_size(config));
    out.extend_from_slice(&HDFS_FAPL_VERSION.to_le_bytes());
    hdfs_encode_fixed_string(&config.namenode_name, out)?;
    out.extend_from_slice(&u32::from(config.namenode_port).to_le_bytes());
    hdfs_encode_fixed_string(&config.user_name, out)?;
    hdfs_encode_fixed_string("", out)?;
    out.extend_from_slice(&config.buffer_size.to_le_bytes());
    Ok(())
}

/// `H5FD__hdfs_sb_encode`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
#[deprecated(note = "use H5FD__hdfs_sb_encode_into")]
pub fn H5FD__hdfs_sb_encode(config: &HdfsConfig) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(H5FD__hdfs_sb_size(config));
    H5FD__hdfs_sb_encode_into(config, &mut out)?;
    Ok(out)
}

/// `H5FD__hdfs_sb_decode`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__hdfs_sb_decode(bytes: &[u8]) -> Result<HdfsConfig> {
    if bytes.len() != HDFS_CONFIG_IMAGE_SIZE {
        return Err(Error::InvalidFormat(
            "HDFS VFD config image has invalid length".into(),
        ));
    }
    let version = read_le_u32_at(bytes, 0, "HDFS VFD config version")?;
    if version != HDFS_FAPL_VERSION {
        return Err(Error::InvalidFormat(
            "unknown HDFS VFD config version".into(),
        ));
    }

    let namenode_start = 4usize;
    let port_offset = namenode_start + HDFS_STRING_FIELD_SIZE;
    let user_start = port_offset + 4;
    let kerberos_start = user_start + HDFS_STRING_FIELD_SIZE;
    let buffer_offset = kerberos_start + HDFS_STRING_FIELD_SIZE;

    let port = read_le_u32_at(bytes, port_offset, "HDFS VFD namenode port")?;
    let namenode_port = u16::try_from(port)
        .map_err(|_| Error::InvalidFormat("HDFS VFD namenode port exceeds u16".into()))?;
    let config = HdfsConfig {
        namenode_name: hdfs_decode_fixed_string(bytes, namenode_start, "HDFS VFD namenode name")?,
        namenode_port,
        user_name: hdfs_decode_fixed_string(bytes, user_start, "HDFS VFD user name")?,
        buffer_size: read_le_u32_at(bytes, buffer_offset, "HDFS VFD stream buffer size")?,
    };
    let kerberos =
        hdfs_decode_fixed_string(bytes, kerberos_start, "HDFS VFD Kerberos ticket cache")?;
    if !kerberos.is_empty() {
        return Err(Error::InvalidFormat(
            "HDFS VFD Kerberos ticket cache is unsupported".into(),
        ));
    }
    if !H5FD__hdfs_validate_config(&config) {
        return Err(Error::InvalidFormat("invalid HDFS VFD config".into()));
    }
    Ok(config)
}

fn hdfs_fixed_string_is_valid(value: &str) -> bool {
    value.len() <= HDFS_STRING_SPACE && !value.as_bytes().contains(&0)
}

fn hdfs_encode_fixed_string(value: &str, out: &mut Vec<u8>) -> Result<()> {
    if !hdfs_fixed_string_is_valid(value) {
        return Err(Error::InvalidFormat("invalid HDFS VFD string field".into()));
    }
    out.extend_from_slice(value.as_bytes());
    out.resize(out.len() + HDFS_STRING_FIELD_SIZE - value.len(), 0);
    Ok(())
}

fn hdfs_decode_fixed_string(bytes: &[u8], offset: usize, context: &str) -> Result<String> {
    let end = offset
        .checked_add(HDFS_STRING_FIELD_SIZE)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))?;
    let field = bytes
        .get(offset..end)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))?;
    let nul_pos = field
        .iter()
        .position(|&byte| byte == 0)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is not NUL terminated")))?;
    if field[nul_pos..].iter().any(|&byte| byte != 0) {
        return Err(Error::InvalidFormat(format!(
            "{context} has nonzero bytes after NUL terminator"
        )));
    }
    std::str::from_utf8(&field[..nul_pos])
        .map(str::to_string)
        .map_err(|_| Error::InvalidFormat(format!("{context} is not UTF-8")))
}

/// `H5FD__hdfs_handle_open`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__hdfs_handle_open(_path: &str) -> Result<()> {
    Err(unsupported_vfd_driver("HDFS"))
}

/// `H5FD__hdfs_handle_close`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__hdfs_handle_close() {}

/// `H5FD__hdfs_fapl_get`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__hdfs_fapl_get(config: &HdfsConfig) -> HdfsConfig {
    config.clone()
}

/// `H5FD__hdfs_fapl_copy`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__hdfs_fapl_copy(config: &HdfsConfig) -> HdfsConfig {
    config.clone()
}

/// `H5FD__hdfs_fapl_free`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__hdfs_fapl_free(_config: HdfsConfig) {}

/// `H5FD__hdfs_open`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__hdfs_open(_path: &str, _config: &HdfsConfig) -> Result<()> {
    Err(unsupported_vfd_driver("HDFS"))
}

/// `H5FD__hdfs_close`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__hdfs_close() {}

/// `H5FD__hdfs_cmp`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__hdfs_cmp() -> Ordering {
    Ordering::Equal
}

/// `H5FD__hdfs_query`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__hdfs_query() -> u64 {
    0
}

/// `H5FD__hdfs_get_eoa`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__hdfs_get_eoa() -> Result<u64> {
    Err(unsupported_vfd_driver("HDFS"))
}

/// `H5FD__hdfs_set_eoa`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__hdfs_set_eoa(_eoa: u64) -> Result<()> {
    Err(unsupported_vfd_driver("HDFS"))
}

/// `H5FD__hdfs_get_eof`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__hdfs_get_eof() -> Result<u64> {
    Err(unsupported_vfd_driver("HDFS"))
}

/// `H5FD__hdfs_get_handle`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__hdfs_get_handle() -> Result<()> {
    Err(unsupported_vfd_driver("HDFS"))
}

/// `H5FD__hdfs_read`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__hdfs_read(_addr: u64, _buf: &mut [u8]) -> Result<()> {
    Err(unsupported_vfd_driver("HDFS"))
}

/// `H5FD__hdfs_write`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__hdfs_write(_addr: u64, _data: &[u8]) -> Result<()> {
    Err(unsupported_vfd_driver("HDFS"))
}

/// `H5FD__hdfs_truncate`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__hdfs_truncate() -> Result<()> {
    Err(unsupported_vfd_driver("HDFS"))
}

/// `H5FD__hdfs_lock`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__hdfs_lock() -> Result<()> {
    Err(unsupported_vfd_driver("HDFS"))
}

/// `H5FD__hdfs_unlock`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__hdfs_unlock() -> Result<()> {
    Err(unsupported_vfd_driver("HDFS"))
}

/// `H5FD__hdfs_delete`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__hdfs_delete(_path: &str) -> Result<()> {
    Err(unsupported_vfd_driver("HDFS"))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S3ParsedUrl {
    pub scheme: String,
    pub bucket: String,
    pub key: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S3ParsedUrlRef<'a> {
    pub scheme: &'a str,
    pub bucket: &'a str,
    pub key: &'a str,
}

/// VFD: s3comms init.
#[allow(non_snake_case)]
pub fn H5FD__s3comms_init() -> Result<()> {
    Err(unsupported_vfd_driver("S3/ROS3"))
}

/// VFD: s3comms term func.
#[allow(non_snake_case)]
pub fn H5FD__s3comms_term_func() {}

/// VFD: s3comms term.
#[allow(non_snake_case)]
pub fn H5FD__s3comms_term() {}

/// VFD: s3comms s3r req finish cb.
#[allow(non_snake_case)]
pub fn H5FD__s3comms_s3r_req_finish_cb() -> Result<()> {
    Err(unsupported_vfd_driver("S3/ROS3"))
}

/// VFD: s3comms s3r req finish pred.
#[allow(non_snake_case)]
pub fn H5FD__s3comms_s3r_req_finish_pred(done: bool) -> bool {
    done
}

/// VFD: s3comms cred provider get creds cb.
#[allow(non_snake_case)]
pub fn H5FD__s3comms_cred_provider_get_creds_cb() -> Result<()> {
    Err(unsupported_vfd_driver("S3/ROS3 credentials"))
}

/// VFD: s3comms cred provider pred.
#[allow(non_snake_case)]
pub fn H5FD__s3comms_cred_provider_pred(has_credentials: bool) -> bool {
    has_credentials
}

/// VFD: s3comms s3r open.
#[allow(non_snake_case)]
pub fn H5FD__s3comms_s3r_open(_url: &str) -> Result<()> {
    Err(unsupported_vfd_driver("S3/ROS3"))
}

/// VFD: s3comms s3r close.
#[allow(non_snake_case)]
pub fn H5FD__s3comms_s3r_close() {}

/// VFD: s3comms s3r get filesize.
#[allow(non_snake_case)]
pub fn H5FD__s3comms_s3r_get_filesize(_url: &str) -> Result<u64> {
    Err(unsupported_vfd_driver("S3/ROS3"))
}

/// VFD: s3comms s3r getsize.
#[allow(non_snake_case)]
pub fn H5FD__s3comms_s3r_getsize(_url: &str) -> Result<u64> {
    Err(unsupported_vfd_driver("S3/ROS3"))
}

/// VFD: s3comms s3r getsize headers cb.
#[allow(non_snake_case)]
pub fn H5FD__s3comms_s3r_getsize_headers_cb(_headers: &[(&str, &str)]) -> Option<u64> {
    None
}

/// VFD: s3comms parse url.
#[allow(non_snake_case)]
pub fn H5FD__s3comms_parse_url_ref(url: &str) -> Result<S3ParsedUrlRef<'_>> {
    let (scheme, rest) = url
        .split_once("://")
        .ok_or_else(|| Error::InvalidFormat("S3 URL missing scheme".into()))?;
    let (bucket, key) = rest
        .split_once('/')
        .ok_or_else(|| Error::InvalidFormat("S3 URL missing object key".into()))?;
    if scheme.is_empty() || bucket.is_empty() || key.is_empty() {
        return Err(Error::InvalidFormat("S3 URL has an empty component".into()));
    }
    Ok(S3ParsedUrlRef {
        scheme,
        bucket,
        key,
    })
}

/// VFD: s3comms parse url.
#[allow(non_snake_case)]
#[deprecated(note = "use H5FD__s3comms_parse_url_ref to borrow URL components")]
pub fn H5FD__s3comms_parse_url(url: &str) -> Result<S3ParsedUrl> {
    let parsed = H5FD__s3comms_parse_url_ref(url)?;
    Ok(S3ParsedUrl {
        scheme: parsed.scheme.to_string(),
        bucket: parsed.bucket.to_string(),
        key: parsed.key.to_string(),
    })
}

/// VFD: s3comms free purl.
#[allow(non_snake_case)]
pub fn H5FD__s3comms_free_purl(_url: S3ParsedUrl) {}

/// VFD: s3comms get aws region.
#[allow(non_snake_case)]
pub fn H5FD__s3comms_get_aws_region_str(endpoint: &str) -> Option<&str> {
    endpoint
        .split('.')
        .find(|part| part.starts_with("us-") || part.starts_with("eu-") || part.starts_with("ap-"))
}

/// VFD: s3comms get aws region.
#[allow(non_snake_case)]
#[deprecated(note = "use H5FD__s3comms_get_aws_region_str")]
pub fn H5FD__s3comms_get_aws_region(endpoint: &str) -> Option<String> {
    H5FD__s3comms_get_aws_region_str(endpoint).map(str::to_string)
}

/// VFD: s3comms get credentials provider.
#[allow(non_snake_case)]
pub fn H5FD__s3comms_get_credentials_provider() -> Result<()> {
    Err(unsupported_vfd_driver("S3/ROS3 credentials"))
}

/// VFD: s3comms format user agent header.
#[allow(non_snake_case)]
pub fn H5FD__s3comms_format_user_agent_header_to_writer(
    writer: &mut impl fmt::Write,
    product: &str,
    version: &str,
) -> fmt::Result {
    write!(writer, "{product}/{version}")
}

/// VFD: s3comms format user agent header.
#[allow(non_snake_case)]
pub fn H5FD__s3comms_format_user_agent_header_into(product: &str, version: &str, out: &mut String) {
    H5FD__s3comms_format_user_agent_header_to_writer(out, product, version)
        .expect("writing to String should not fail");
}

/// VFD: s3comms format user agent header.
#[allow(non_snake_case)]
#[deprecated(
    note = "use H5FD__s3comms_format_user_agent_header_to_writer or H5FD__s3comms_format_user_agent_header_into"
)]
pub fn H5FD__s3comms_format_user_agent_header(product: &str, version: &str) -> String {
    let mut out = String::with_capacity(product.len() + 1 + version.len());
    H5FD__s3comms_format_user_agent_header_into(product, version, &mut out);
    out
}

/// VFD: s3comms httpcode to str.
#[allow(non_snake_case)]
pub fn H5FD__s3comms_httpcode_to_str(code: u16) -> &'static str {
    match code {
        200 => "OK",
        206 => "Partial Content",
        301 => "Moved Permanently",
        403 => "Forbidden",
        404 => "Not Found",
        416 => "Range Not Satisfiable",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "Unknown",
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MirrorXmit {
    Close,
    Lock,
    Open { path: String },
    Reply { status: i32 },
    SetEoa { eoa: u64 },
    Write { addr: u64, data: Vec<u8> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MirrorXmitRef<'a> {
    Close,
    Lock,
    Open { path: &'a str },
    Reply { status: i32 },
    SetEoa { eoa: u64 },
    Write { addr: u64, data: &'a [u8] },
}

/// `H5FD__mirror_register`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__mirror_register() -> Result<()> {
    Err(unsupported_vfd_driver("mirror"))
}

/// `H5FD__mirror_unregister`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__mirror_unregister() {}

/// `H5FD__mirror_xmit_encode_uint8`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__mirror_xmit_encode_uint8_into(value: u8, out: &mut Vec<u8>) {
    out.push(value);
}

/// `H5FD__mirror_xmit_encode_uint8`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
#[deprecated(note = "use H5FD__mirror_xmit_encode_uint8_into")]
pub fn H5FD__mirror_xmit_encode_uint8(value: u8) -> Vec<u8> {
    let mut out = Vec::with_capacity(1);
    H5FD__mirror_xmit_encode_uint8_into(value, &mut out);
    out
}

/// `H5FD__mirror_xmit_decode_uint64`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__mirror_xmit_decode_uint64(bytes: &[u8]) -> Result<u64> {
    if bytes.len() != 8 {
        return Err(Error::InvalidFormat(
            "mirror transmit uint64 payload has invalid length".into(),
        ));
    }
    read_le_u64_at(bytes, 0, "mirror transmit uint64")
}

/// `mirror` VFD: xmit decode lock.
#[allow(non_snake_case)]
pub fn H5FD_mirror_xmit_decode_lock(bytes: &[u8]) -> Result<MirrorXmit> {
    if !bytes.is_empty() {
        return Err(Error::InvalidFormat(
            "mirror transmit lock payload has invalid length".into(),
        ));
    }
    Ok(MirrorXmit::Lock)
}

/// `mirror` VFD: xmit decode open.
#[allow(non_snake_case)]
pub fn H5FD_mirror_xmit_decode_open_ref(bytes: &[u8]) -> Result<MirrorXmitRef<'_>> {
    let path = std::str::from_utf8(bytes)
        .map_err(|_| Error::InvalidFormat("mirror transmit open path is not UTF-8".into()))?;
    Ok(MirrorXmitRef::Open { path })
}

/// `mirror` VFD: xmit decode open.
#[allow(non_snake_case)]
#[deprecated(note = "use H5FD_mirror_xmit_decode_open_ref to borrow the path")]
pub fn H5FD_mirror_xmit_decode_open(bytes: &[u8]) -> Result<MirrorXmit> {
    match H5FD_mirror_xmit_decode_open_ref(bytes)? {
        MirrorXmitRef::Open { path } => Ok(MirrorXmit::Open {
            path: path.to_string(),
        }),
        _ => unreachable!("open decoder always returns an open message"),
    }
}

/// `mirror` VFD: xmit decode reply.
#[allow(non_snake_case)]
pub fn H5FD_mirror_xmit_decode_reply(bytes: &[u8]) -> Result<MirrorXmit> {
    if bytes.len() != 4 {
        return Err(Error::InvalidFormat(
            "mirror transmit reply payload has invalid length".into(),
        ));
    }
    let status = i32::from_le_bytes(
        bytes
            .try_into()
            .map_err(|_| Error::InvalidFormat("mirror transmit reply is truncated".into()))?,
    );
    Ok(MirrorXmit::Reply { status })
}

/// `mirror` VFD: xmit decode set eoa.
#[allow(non_snake_case)]
pub fn H5FD_mirror_xmit_decode_set_eoa(bytes: &[u8]) -> Result<MirrorXmit> {
    H5FD__mirror_xmit_decode_uint64(bytes).map(|eoa| MirrorXmit::SetEoa { eoa })
}

/// `mirror` VFD: xmit decode write.
#[allow(non_snake_case)]
pub fn H5FD_mirror_xmit_decode_write_ref(bytes: &[u8]) -> Result<MirrorXmitRef<'_>> {
    if bytes.len() < 8 {
        return Err(Error::InvalidFormat(
            "mirror transmit write payload is truncated".into(),
        ));
    }
    let addr = read_le_u64_at(bytes, 0, "mirror transmit write address")?;
    Ok(MirrorXmitRef::Write {
        addr,
        data: &bytes[8..],
    })
}

/// `mirror` VFD: xmit decode write.
#[allow(non_snake_case)]
#[deprecated(note = "use H5FD_mirror_xmit_decode_write_ref to borrow the payload")]
pub fn H5FD_mirror_xmit_decode_write(bytes: &[u8]) -> Result<MirrorXmit> {
    match H5FD_mirror_xmit_decode_write_ref(bytes)? {
        MirrorXmitRef::Write { addr, data } => Ok(MirrorXmit::Write {
            addr,
            data: data.to_vec(),
        }),
        _ => unreachable!("write decoder always returns a write message"),
    }
}

/// `mirror` VFD: xmit encode open.
#[allow(non_snake_case)]
pub fn H5FD_mirror_xmit_encode_open_into(path: &str, out: &mut Vec<u8>) -> Result<()> {
    out.extend_from_slice(path.as_bytes());
    Ok(())
}

/// `mirror` VFD: xmit encode open.
#[allow(non_snake_case)]
#[deprecated(note = "use H5FD_mirror_xmit_encode_open_into")]
pub fn H5FD_mirror_xmit_encode_open(path: &str) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(path.len());
    H5FD_mirror_xmit_encode_open_into(path, &mut out)?;
    Ok(out)
}

/// `mirror` VFD: xmit encode reply.
#[allow(non_snake_case)]
pub fn H5FD_mirror_xmit_encode_reply_into(status: i32, out: &mut Vec<u8>) -> Result<()> {
    out.extend_from_slice(&status.to_le_bytes());
    Ok(())
}

/// `mirror` VFD: xmit encode reply.
#[allow(non_snake_case)]
#[deprecated(note = "use H5FD_mirror_xmit_encode_reply_into")]
pub fn H5FD_mirror_xmit_encode_reply(status: i32) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(4);
    H5FD_mirror_xmit_encode_reply_into(status, &mut out)?;
    Ok(out)
}

/// `mirror` VFD: xmit encode set eoa.
#[allow(non_snake_case)]
pub fn H5FD_mirror_xmit_encode_set_eoa_into(eoa: u64, out: &mut Vec<u8>) -> Result<()> {
    out.extend_from_slice(&eoa.to_le_bytes());
    Ok(())
}

/// `mirror` VFD: xmit encode set eoa.
#[allow(non_snake_case)]
#[deprecated(note = "use H5FD_mirror_xmit_encode_set_eoa_into")]
pub fn H5FD_mirror_xmit_encode_set_eoa(eoa: u64) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(8);
    H5FD_mirror_xmit_encode_set_eoa_into(eoa, &mut out)?;
    Ok(out)
}

/// `mirror` VFD: xmit encode write.
#[allow(non_snake_case)]
pub fn H5FD_mirror_xmit_encode_write_into(addr: u64, data: &[u8], out: &mut Vec<u8>) -> Result<()> {
    let image_len = 8usize.checked_add(data.len()).ok_or_else(|| {
        Error::InvalidFormat("mirror transmit write image length overflow".into())
    })?;
    out.reserve(image_len);
    out.extend_from_slice(&addr.to_le_bytes());
    out.extend_from_slice(data);
    Ok(())
}

/// `mirror` VFD: xmit encode write.
#[allow(non_snake_case)]
#[deprecated(note = "use H5FD_mirror_xmit_encode_write_into")]
pub fn H5FD_mirror_xmit_encode_write(addr: u64, data: &[u8]) -> Result<Vec<u8>> {
    let image_len = 8usize.checked_add(data.len()).ok_or_else(|| {
        Error::InvalidFormat("mirror transmit write image length overflow".into())
    })?;
    let mut out = Vec::with_capacity(image_len);
    H5FD_mirror_xmit_encode_write_into(addr, data, &mut out)?;
    Ok(out)
}

/// `mirror` VFD: xmit is close.
#[allow(non_snake_case)]
pub fn H5FD_mirror_xmit_is_close(message: &MirrorXmit) -> bool {
    matches!(message, MirrorXmit::Close)
}

/// `mirror` VFD: xmit is lock.
#[allow(non_snake_case)]
pub fn H5FD_mirror_xmit_is_lock(message: &MirrorXmit) -> bool {
    matches!(message, MirrorXmit::Lock)
}

/// `mirror` VFD: xmit is set eoa.
#[allow(non_snake_case)]
pub fn H5FD_mirror_xmit_is_set_eoa(message: &MirrorXmit) -> bool {
    matches!(message, MirrorXmit::SetEoa { .. })
}

/// `mirror` VFD: xmit is reply.
#[allow(non_snake_case)]
pub fn H5FD_mirror_xmit_is_reply(message: &MirrorXmit) -> bool {
    matches!(message, MirrorXmit::Reply { .. })
}

/// `mirror` VFD: xmit is write.
#[allow(non_snake_case)]
pub fn H5FD_mirror_xmit_is_write(message: &MirrorXmit) -> bool {
    matches!(message, MirrorXmit::Write { .. })
}

/// `mirror` VFD: xmit is xmit.
#[allow(non_snake_case)]
pub fn H5FD_mirror_xmit_is_xmit(_message: &MirrorXmit) -> bool {
    true
}

/// `H5FD__mirror_verify_reply`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__mirror_verify_reply(message: &MirrorXmit) -> bool {
    matches!(message, MirrorXmit::Reply { status: 0 })
}

/// `H5FD__mirror_fapl_get`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__mirror_fapl_get() -> Result<()> {
    Err(unsupported_vfd_driver("mirror"))
}

/// `H5FD__mirror_fapl_copy`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__mirror_fapl_copy() -> Result<()> {
    Err(unsupported_vfd_driver("mirror"))
}

/// `H5FD__mirror_fapl_free`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__mirror_fapl_free() {}

/// `H5FD__mirror_open`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__mirror_open(_path: &str) -> Result<()> {
    Err(unsupported_vfd_driver("mirror"))
}

/// `H5FD__mirror_close`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__mirror_close() {}

/// `H5FD__mirror_query`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__mirror_query() -> u64 {
    0
}

/// `H5FD__mirror_get_eoa`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__mirror_get_eoa() -> Result<u64> {
    Err(unsupported_vfd_driver("mirror"))
}

/// `H5FD__mirror_set_eoa`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__mirror_set_eoa(_eoa: u64) -> Result<()> {
    Err(unsupported_vfd_driver("mirror"))
}

/// `H5FD__mirror_get_eof`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__mirror_get_eof() -> Result<u64> {
    Err(unsupported_vfd_driver("mirror"))
}

/// `H5FD__mirror_read`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__mirror_read(_addr: u64, _buf: &mut [u8]) -> Result<()> {
    Err(unsupported_vfd_driver("mirror"))
}

/// `H5FD__mirror_write`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__mirror_write(_addr: u64, _data: &[u8]) -> Result<()> {
    Err(unsupported_vfd_driver("mirror"))
}

/// `H5FD__mirror_truncate`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__mirror_truncate() -> Result<()> {
    Err(unsupported_vfd_driver("mirror"))
}

/// `H5FD__mirror_lock`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__mirror_lock() -> Result<()> {
    Err(unsupported_vfd_driver("mirror"))
}

/// `H5FD__mirror_unlock`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__mirror_unlock() -> Result<()> {
    Err(unsupported_vfd_driver("mirror"))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FamilyFileConfig {
    pub member_size: u64,
    pub printf_filename: String,
}

impl Default for FamilyFileConfig {
    fn default() -> Self {
        Self {
            member_size: 100 * 1024 * 1024,
            printf_filename: "%05d.h5".into(),
        }
    }
}

/// `H5FD__family_get_default_config`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__family_get_default_config() -> FamilyFileConfig {
    FamilyFileConfig::default()
}

/// `H5FD__family_get_default_printf_filename`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__family_get_default_printf_filename() -> &'static str {
    "%05d.h5"
}

/// `H5FD__family_register`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__family_register() -> Result<()> {
    Err(unsupported_vfd_driver("family"))
}

/// `H5FD__family_unregister`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__family_unregister() {}

/// `H5FD__family_fapl_get`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__family_fapl_get(config: &FamilyFileConfig) -> FamilyFileConfig {
    config.clone()
}

/// `H5FD__family_fapl_copy`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__family_fapl_copy(config: &FamilyFileConfig) -> FamilyFileConfig {
    config.clone()
}

/// `H5FD__family_fapl_free`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__family_fapl_free(_config: FamilyFileConfig) {}

/// `H5FD__family_validate_config`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__family_validate_config(config: &FamilyFileConfig) -> bool {
    !config.printf_filename.is_empty()
}

/// `H5FD__family_sb_size`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__family_sb_size(_config: &FamilyFileConfig) -> Result<usize> {
    Ok(8)
}

/// `H5FD__family_sb_encode`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__family_sb_encode_into(config: &FamilyFileConfig, out: &mut Vec<u8>) -> Result<()> {
    if !H5FD__family_validate_config(config) {
        return Err(Error::InvalidFormat("invalid family VFD config".into()));
    }
    out.reserve(H5FD__family_sb_size(config)?);
    out.extend_from_slice(&config.member_size.to_le_bytes());
    Ok(())
}

/// `H5FD__family_sb_encode`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
#[deprecated(note = "use H5FD__family_sb_encode_into")]
pub fn H5FD__family_sb_encode(config: &FamilyFileConfig) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(H5FD__family_sb_size(config)?);
    H5FD__family_sb_encode_into(config, &mut out)?;
    Ok(out)
}

/// `H5FD__family_sb_decode`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__family_sb_decode(bytes: &[u8]) -> Result<FamilyFileConfig> {
    if bytes.len() != H5FD__family_sb_size(&FamilyFileConfig::default())? {
        return Err(Error::InvalidFormat(
            "family VFD config image has invalid length".into(),
        ));
    }
    let member_size = read_le_u64_at(bytes, 0, "family VFD member size")?;
    let config = FamilyFileConfig {
        member_size,
        printf_filename: FamilyFileConfig::default().printf_filename,
    };
    if !H5FD__family_validate_config(&config) {
        return Err(Error::InvalidFormat("invalid family VFD config".into()));
    }
    Ok(config)
}

/// `H5FD__family_open`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__family_open(_pattern: &str, _config: &FamilyFileConfig) -> Result<()> {
    Err(unsupported_vfd_driver("family"))
}

/// `H5FD__family_close`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__family_close() {}

/// `H5FD__family_cmp`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__family_cmp(left: &FamilyFileConfig, right: &FamilyFileConfig) -> Ordering {
    left.member_size
        .cmp(&right.member_size)
        .then_with(|| left.printf_filename.cmp(&right.printf_filename))
}

/// `H5FD__family_query`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__family_query() -> u64 {
    0
}

/// `H5FD__family_get_eoa`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__family_get_eoa() -> Result<u64> {
    Err(unsupported_vfd_driver("family"))
}

/// `H5FD__family_set_eoa`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__family_set_eoa(_eoa: u64) -> Result<()> {
    Err(unsupported_vfd_driver("family"))
}

/// `H5FD__family_read`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__family_read(_addr: u64, _buf: &mut [u8]) -> Result<()> {
    Err(unsupported_vfd_driver("family"))
}

/// `H5FD__family_flush`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__family_flush() -> Result<()> {
    Err(unsupported_vfd_driver("family"))
}

/// `H5FD__family_truncate`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__family_truncate() -> Result<()> {
    Err(unsupported_vfd_driver("family"))
}

/// `H5FD__family_lock`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__family_lock() -> Result<()> {
    Err(unsupported_vfd_driver("family"))
}

/// `H5FD__family_unlock`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__family_unlock() -> Result<()> {
    Err(unsupported_vfd_driver("family"))
}

/// `H5FD__family_delete`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__family_delete(_pattern: &str) -> Result<()> {
    Err(unsupported_vfd_driver("family"))
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MultiFileConfig {
    pub memb_map: HashMap<VfdMemType, FileDriverKind>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MultiFileState {
    pub memb_addr: HashMap<VfdMemType, u64>,
    pub memb_next: HashMap<VfdMemType, u64>,
}

/// `multi` VFD: populate the default driver-specific configuration.
#[allow(non_snake_case)]
pub fn H5FD_multi_populate_config() -> MultiFileConfig {
    let mut config = MultiFileConfig::default();
    config
        .memb_map
        .insert(VfdMemType::Default, FileDriverKind::Sec2);
    config
}

/// VFD: vfd mem type code.
fn vfd_mem_type_code(mem_type: VfdMemType) -> u8 {
    match mem_type {
        VfdMemType::Default => 0,
        VfdMemType::Super => 1,
        VfdMemType::BTree => 2,
        VfdMemType::RawData => 3,
        VfdMemType::GlobalHeap => 4,
        VfdMemType::LocalHeap => 5,
        VfdMemType::ObjectHeader => 6,
        VfdMemType::Draw => 7,
        VfdMemType::Garbage => 8,
    }
}

/// VFD: vfd mem type from code.
fn vfd_mem_type_from_code(code: u8) -> Option<VfdMemType> {
    Some(match code {
        0 => VfdMemType::Default,
        1 => VfdMemType::Super,
        2 => VfdMemType::BTree,
        3 => VfdMemType::RawData,
        4 => VfdMemType::GlobalHeap,
        5 => VfdMemType::LocalHeap,
        6 => VfdMemType::ObjectHeader,
        7 => VfdMemType::Draw,
        8 => VfdMemType::Garbage,
        _ => return None,
    })
}

const VFD_MEM_TYPES_IN_CODE_ORDER: [VfdMemType; 9] = [
    VfdMemType::Default,
    VfdMemType::Super,
    VfdMemType::BTree,
    VfdMemType::RawData,
    VfdMemType::GlobalHeap,
    VfdMemType::LocalHeap,
    VfdMemType::ObjectHeader,
    VfdMemType::Draw,
    VfdMemType::Garbage,
];

/// VFD: file driver kind code.
fn file_driver_kind_code(kind: FileDriverKind) -> u8 {
    match kind {
        FileDriverKind::Sec2 => 0,
        FileDriverKind::Stdio => 1,
        FileDriverKind::Core => 2,
        FileDriverKind::Direct => 3,
    }
}

/// VFD: file driver kind from code.
fn file_driver_kind_from_code(code: u8) -> Option<FileDriverKind> {
    Some(match code {
        0 => FileDriverKind::Sec2,
        1 => FileDriverKind::Stdio,
        2 => FileDriverKind::Core,
        3 => FileDriverKind::Direct,
        _ => return None,
    })
}

/// `multi` VFD: validate config.
#[allow(non_snake_case)]
pub fn H5FD_multi_validate_config(config: &MultiFileConfig) -> bool {
    !config.memb_map.is_empty()
}

/// `multi` VFD: return the superblock extension size for a driver.
#[allow(non_snake_case)]
pub fn H5FD_multi_sb_size(config: &MultiFileConfig) -> Result<usize> {
    config
        .memb_map
        .len()
        .checked_mul(2)
        .and_then(|payload| payload.checked_add(4))
        .ok_or_else(|| Error::InvalidFormat("multi VFD member map length overflow".into()))
}

/// `multi` VFD: sb encode.
#[allow(non_snake_case)]
pub fn H5FD_multi_sb_encode_into(config: &MultiFileConfig, out: &mut Vec<u8>) -> Result<()> {
    if !H5FD_multi_validate_config(config) {
        return Err(Error::InvalidFormat("invalid multi VFD config".into()));
    }
    let count = u32::try_from(config.memb_map.len())
        .map_err(|_| Error::InvalidFormat("multi VFD member count exceeds u32".into()))?;
    out.reserve(H5FD_multi_sb_size(config)?);
    out.extend_from_slice(&count.to_le_bytes());
    for mem_type in VFD_MEM_TYPES_IN_CODE_ORDER {
        if let Some(driver) = config.memb_map.get(&mem_type) {
            out.push(vfd_mem_type_code(mem_type));
            out.push(file_driver_kind_code(*driver));
        }
    }
    Ok(())
}

/// `multi` VFD: sb encode.
#[allow(non_snake_case)]
#[deprecated(note = "use H5FD_multi_sb_encode_into")]
pub fn H5FD_multi_sb_encode(config: &MultiFileConfig) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(H5FD_multi_sb_size(config)?);
    H5FD_multi_sb_encode_into(config, &mut out)?;
    Ok(out)
}

/// `multi` VFD: sb decode.
#[allow(non_snake_case)]
pub fn H5FD_multi_sb_decode(bytes: &[u8]) -> Result<MultiFileConfig> {
    let count = read_le_u32_len_at(bytes, 0, "multi VFD member count")?;
    let payload_len = count
        .checked_mul(2)
        .and_then(|len| 4usize.checked_add(len))
        .ok_or_else(|| Error::InvalidFormat("multi VFD member map length overflow".into()))?;
    if bytes.len() != payload_len {
        return Err(Error::InvalidFormat(
            "multi VFD member map has invalid length".into(),
        ));
    }
    let mut memb_map = HashMap::new();
    for entry in bytes[4..].chunks_exact(2) {
        let mem_type = vfd_mem_type_from_code(entry[0])
            .ok_or_else(|| Error::InvalidFormat("invalid multi VFD memory type".into()))?;
        let driver = file_driver_kind_from_code(entry[1])
            .ok_or_else(|| Error::InvalidFormat("invalid multi VFD driver kind".into()))?;
        if memb_map.insert(mem_type, driver).is_some() {
            return Err(Error::InvalidFormat(
                "multi VFD member map contains duplicate memory type".into(),
            ));
        }
    }
    let config = MultiFileConfig { memb_map };
    if !H5FD_multi_validate_config(&config) {
        return Err(Error::InvalidFormat("invalid multi VFD config".into()));
    }
    Ok(config)
}

/// `multi` VFD: get the driver-specific FAPL configuration.
#[allow(non_snake_case)]
pub fn H5FD_multi_fapl_get(config: &MultiFileConfig) -> MultiFileConfig {
    config.clone()
}

/// `multi` VFD: copy the driver-specific FAPL configuration.
#[allow(non_snake_case)]
pub fn H5FD_multi_fapl_copy(config: &MultiFileConfig) -> MultiFileConfig {
    config.clone()
}

/// `multi` VFD: free the driver-specific FAPL configuration.
#[allow(non_snake_case)]
pub fn H5FD_multi_fapl_free(_config: MultiFileConfig) {}

/// `multi` VFD: open.
#[allow(non_snake_case)]
pub fn H5FD_multi_open(_path: &str, _config: &MultiFileConfig) -> Result<()> {
    Err(unsupported_vfd_driver("multi"))
}

/// `multi` VFD: close.
#[allow(non_snake_case)]
pub fn H5FD_multi_close() {}

/// `multi` VFD: compare two driver instances.
#[allow(non_snake_case)]
pub fn H5FD_multi_cmp(left: &MultiFileConfig, right: &MultiFileConfig) -> Ordering {
    left.memb_map.len().cmp(&right.memb_map.len())
}

/// `multi` VFD: query feature flags.
#[allow(non_snake_case)]
pub fn H5FD_multi_query() -> u64 {
    0
}

/// `multi` VFD: get type map.
#[allow(non_snake_case)]
pub fn H5FD_multi_get_type_map(config: &MultiFileConfig) -> &HashMap<VfdMemType, FileDriverKind> {
    &config.memb_map
}

/// VFD: compute next.
pub fn compute_next(file: &mut MultiFileState) {
    file.memb_next.clear();
    for (mt1, addr1) in &file.memb_addr {
        let next = file
            .memb_addr
            .iter()
            .filter_map(|(_, addr2)| (*addr2 > *addr1).then_some(*addr2))
            .min()
            .unwrap_or(u64::MAX);
        file.memb_next.insert(*mt1, next);
    }
}

/// `multi` VFD: get the end-of-allocation address.
#[allow(non_snake_case)]
pub fn H5FD_multi_get_eoa() -> Result<u64> {
    Err(unsupported_vfd_driver("multi"))
}

/// `multi` VFD: set the end-of-allocation address.
#[allow(non_snake_case)]
pub fn H5FD_multi_set_eoa(_eoa: u64) -> Result<()> {
    Err(unsupported_vfd_driver("multi"))
}

/// `multi` VFD: get the end-of-file address.
#[allow(non_snake_case)]
pub fn H5FD_multi_get_eof() -> Result<u64> {
    Err(unsupported_vfd_driver("multi"))
}

/// `multi` VFD: get the underlying file handle.
#[allow(non_snake_case)]
pub fn H5FD_multi_get_handle() -> Result<()> {
    Err(unsupported_vfd_driver("multi"))
}

/// `multi` VFD: allocate space in the file.
#[allow(non_snake_case)]
pub fn H5FD_multi_alloc(_size: u64) -> Result<u64> {
    Err(unsupported_vfd_driver("multi"))
}

/// `multi` VFD: free a previously allocated region.
#[allow(non_snake_case)]
pub fn H5FD_multi_free(_addr: u64, _size: u64) -> Result<()> {
    Err(unsupported_vfd_driver("multi"))
}

/// `multi` VFD: read bytes from the file.
#[allow(non_snake_case)]
pub fn H5FD_multi_read(_addr: u64, _buf: &mut [u8]) -> Result<()> {
    Err(unsupported_vfd_driver("multi"))
}

/// `multi` VFD: write bytes to the file.
#[allow(non_snake_case)]
pub fn H5FD_multi_write(_addr: u64, _data: &[u8]) -> Result<()> {
    Err(unsupported_vfd_driver("multi"))
}

/// `multi` VFD: flush buffered writes to disk.
#[allow(non_snake_case)]
pub fn H5FD_multi_flush() -> Result<()> {
    Err(unsupported_vfd_driver("multi"))
}

/// `multi` VFD: truncate the file to the current EOA.
#[allow(non_snake_case)]
pub fn H5FD_multi_truncate() -> Result<()> {
    Err(unsupported_vfd_driver("multi"))
}

/// `multi` VFD: acquire an advisory file lock.
#[allow(non_snake_case)]
pub fn H5FD_multi_lock() -> Result<()> {
    Err(unsupported_vfd_driver("multi"))
}

/// `multi` VFD: release an advisory file lock.
#[allow(non_snake_case)]
pub fn H5FD_multi_unlock() -> Result<()> {
    Err(unsupported_vfd_driver("multi"))
}

/// `multi` VFD: delete the file.
#[allow(non_snake_case)]
pub fn H5FD_multi_delete(_path: &str) -> Result<()> {
    Err(unsupported_vfd_driver("multi"))
}

/// `multi` VFD: invoke a driver-specific control op.
#[allow(non_snake_case)]
pub fn H5FD_multi_ctl(_opcode: u64) -> Result<()> {
    Err(unsupported_vfd_driver("multi"))
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SplitterFileConfig {
    pub write_only_path: Option<PathBuf>,
    pub ignore_wo_errors: bool,
}

/// `H5FD__splitter_populate_config`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__splitter_populate_config(write_only_path: Option<PathBuf>) -> SplitterFileConfig {
    SplitterFileConfig {
        write_only_path,
        ignore_wo_errors: false,
    }
}

/// `H5FD__splitter_get_default_wo_path`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__splitter_get_default_wo_path() -> &'static str {
    "%s.splitter"
}

/// VFD: split populate config.
#[allow(non_snake_case)]
pub fn H5FD_split_populate_config(write_only_path: Option<PathBuf>) -> SplitterFileConfig {
    H5FD__splitter_populate_config(write_only_path)
}

/// `H5FD__splitter_register`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__splitter_register() -> Result<()> {
    Err(unsupported_vfd_driver("splitter"))
}

/// `H5FD__splitter_unregister`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__splitter_unregister() {}

/// `H5FD__splitter_fapl_get`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__splitter_fapl_get(config: &SplitterFileConfig) -> SplitterFileConfig {
    config.clone()
}

/// `H5FD__splitter_fapl_copy`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__splitter_fapl_copy(config: &SplitterFileConfig) -> SplitterFileConfig {
    config.clone()
}

/// `H5FD__splitter_fapl_free`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__splitter_fapl_free(_config: SplitterFileConfig) {}

/// `H5FD__splitter_validate_config`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__splitter_validate_config(config: &SplitterFileConfig) -> bool {
    config
        .write_only_path
        .as_ref()
        .is_none_or(|path| !path.as_os_str().is_empty())
}

/// `H5FD__splitter_open`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__splitter_open(_path: &str, _config: &SplitterFileConfig) -> Result<()> {
    Err(unsupported_vfd_driver("splitter"))
}

/// `H5FD__splitter_close`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__splitter_close() {}

/// `H5FD__splitter_flush`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__splitter_flush() -> Result<()> {
    Err(unsupported_vfd_driver("splitter"))
}

/// `H5FD__splitter_read`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__splitter_read(_addr: u64, _buf: &mut [u8]) -> Result<()> {
    Err(unsupported_vfd_driver("splitter"))
}

/// `H5FD__splitter_write`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__splitter_write(_addr: u64, _data: &[u8]) -> Result<()> {
    Err(unsupported_vfd_driver("splitter"))
}

/// `H5FD__splitter_get_eoa`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__splitter_get_eoa() -> Result<u64> {
    Err(unsupported_vfd_driver("splitter"))
}

/// `H5FD__splitter_set_eoa`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__splitter_set_eoa(_eoa: u64) -> Result<()> {
    Err(unsupported_vfd_driver("splitter"))
}

/// `H5FD__splitter_get_eof`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__splitter_get_eof() -> Result<u64> {
    Err(unsupported_vfd_driver("splitter"))
}

/// `H5FD__splitter_truncate`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__splitter_truncate() -> Result<()> {
    Err(unsupported_vfd_driver("splitter"))
}

/// `H5FD__splitter_sb_size`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__splitter_sb_size(config: &SplitterFileConfig) -> Result<usize> {
    5usize
        .checked_add(
            config
                .write_only_path
                .as_ref()
                .map(|path| path.as_os_str().as_encoded_bytes().len())
                .unwrap_or(0),
        )
        .ok_or_else(|| Error::InvalidFormat("splitter VFD config image length overflow".into()))
}

/// `H5FD__splitter_sb_encode`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__splitter_sb_encode_into(config: &SplitterFileConfig, out: &mut Vec<u8>) -> Result<()> {
    if !H5FD__splitter_validate_config(config) {
        return Err(Error::InvalidFormat("invalid splitter VFD config".into()));
    }
    let path = config
        .write_only_path
        .as_ref()
        .map(|path| path.as_os_str().as_encoded_bytes())
        .unwrap_or(&[]);
    let path_len = u32::try_from(path.len())
        .map_err(|_| Error::InvalidFormat("splitter VFD path length exceeds u32".into()))?;
    out.reserve(H5FD__splitter_sb_size(config)?);
    out.push(u8::from(config.ignore_wo_errors));
    out.extend_from_slice(&path_len.to_le_bytes());
    out.extend_from_slice(path);
    Ok(())
}

/// `H5FD__splitter_sb_encode`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
#[deprecated(note = "use H5FD__splitter_sb_encode_into")]
pub fn H5FD__splitter_sb_encode(config: &SplitterFileConfig) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(H5FD__splitter_sb_size(config)?);
    H5FD__splitter_sb_encode_into(config, &mut out)?;
    Ok(out)
}

/// `H5FD__splitter_sb_decode`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__splitter_sb_decode(bytes: &[u8]) -> Result<SplitterFileConfig> {
    let ignore_wo_errors = *bytes
        .first()
        .ok_or_else(|| Error::InvalidFormat("splitter VFD flags are truncated".into()))?
        != 0;
    if bytes[0] > 1 {
        return Err(Error::InvalidFormat(
            "splitter VFD ignore-errors flag is invalid".into(),
        ));
    }
    let path_len = read_le_u32_len_at(bytes, 1, "splitter VFD path length")?;
    let path_end = 5usize
        .checked_add(path_len)
        .ok_or_else(|| Error::InvalidFormat("splitter VFD path length overflow".into()))?;
    let path_bytes = bytes
        .get(5..path_end)
        .ok_or_else(|| Error::InvalidFormat("splitter VFD path is truncated".into()))?;
    if bytes.len() != path_end {
        return Err(Error::InvalidFormat(
            "splitter VFD config has trailing bytes".into(),
        ));
    }
    let write_only_path = if path_bytes.is_empty() {
        None
    } else {
        Some(PathBuf::from(std::str::from_utf8(path_bytes).map_err(
            |_| Error::InvalidFormat("splitter VFD path is not UTF-8".into()),
        )?))
    };
    let config = SplitterFileConfig {
        write_only_path,
        ignore_wo_errors,
    };
    if !H5FD__splitter_validate_config(&config) {
        return Err(Error::InvalidFormat("invalid splitter VFD config".into()));
    }
    Ok(config)
}

/// `H5FD__splitter_cmp`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__splitter_cmp(left: &SplitterFileConfig, right: &SplitterFileConfig) -> Ordering {
    left.write_only_path
        .cmp(&right.write_only_path)
        .then_with(|| left.ignore_wo_errors.cmp(&right.ignore_wo_errors))
}

/// `H5FD__splitter_get_handle`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__splitter_get_handle() -> Result<()> {
    Err(unsupported_vfd_driver("splitter"))
}

/// `H5FD__splitter_lock`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__splitter_lock() -> Result<()> {
    Err(unsupported_vfd_driver("splitter"))
}

/// `H5FD__splitter_unlock`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__splitter_unlock() -> Result<()> {
    Err(unsupported_vfd_driver("splitter"))
}

/// `H5FD__splitter_ctl`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__splitter_ctl(_opcode: u64) -> Result<()> {
    Err(unsupported_vfd_driver("splitter"))
}

/// `H5FD__splitter_query`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__splitter_query() -> u64 {
    0
}

/// `H5FD__splitter_alloc`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__splitter_alloc(_size: u64) -> Result<u64> {
    Err(unsupported_vfd_driver("splitter"))
}

/// `H5FD__splitter_get_type_map`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__splitter_get_type_map() -> Result<()> {
    Err(unsupported_vfd_driver("splitter"))
}

/// `H5FD__splitter_free`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__splitter_free(_addr: u64, _size: u64) -> Result<()> {
    Err(unsupported_vfd_driver("splitter"))
}

/// `H5FD__splitter_delete`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__splitter_delete(_path: &str) -> Result<()> {
    Err(unsupported_vfd_driver("splitter"))
}

/// `H5FD__splitter_log_error`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__splitter_log_error_str(message: &str) -> &str {
    message
}

/// `H5FD__splitter_log_error`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__splitter_log_error_to_writer(
    writer: &mut impl fmt::Write,
    message: &str,
) -> fmt::Result {
    fmt::Write::write_str(writer, message)
}

/// `H5FD__splitter_log_error`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
#[deprecated(note = "use H5FD__splitter_log_error_str or H5FD__splitter_log_error_to_writer")]
pub fn H5FD__splitter_log_error(message: &str) -> String {
    H5FD__splitter_log_error_str(message).to_string()
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LogFileConfig {
    pub log_path: Option<PathBuf>,
    pub flags: u64,
    pub buffer_size: usize,
}

/// `H5FD__log_register`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__log_register() -> Result<()> {
    Err(unsupported_vfd_driver("log"))
}

/// `H5FD__log_unregister`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__log_unregister() {}

/// `H5FD__log_fapl_get`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__log_fapl_get(config: &LogFileConfig) -> LogFileConfig {
    config.clone()
}

/// `H5FD__log_fapl_copy`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__log_fapl_copy(config: &LogFileConfig) -> LogFileConfig {
    config.clone()
}

/// `H5FD__log_fapl_free`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__log_fapl_free(_config: LogFileConfig) {}

/// `H5FD__log_validate_config`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__log_validate_config(config: &LogFileConfig) -> bool {
    config
        .log_path
        .as_ref()
        .is_none_or(|path| !path.as_os_str().is_empty())
}

/// `H5FD__log_sb_size`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__log_sb_size(config: &LogFileConfig) -> Result<usize> {
    20usize
        .checked_add(
            config
                .log_path
                .as_ref()
                .map(|path| path.as_os_str().as_encoded_bytes().len())
                .unwrap_or(0),
        )
        .ok_or_else(|| Error::InvalidFormat("log VFD config image length overflow".into()))
}

/// `H5FD__log_sb_encode`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__log_sb_encode_into(config: &LogFileConfig, out: &mut Vec<u8>) -> Result<()> {
    if !H5FD__log_validate_config(config) {
        return Err(Error::InvalidFormat("invalid log VFD config".into()));
    }
    let path = config
        .log_path
        .as_ref()
        .map(|path| path.as_os_str().as_encoded_bytes())
        .unwrap_or(&[]);
    let buffer_size = u64::try_from(config.buffer_size)
        .map_err(|_| Error::InvalidFormat("log VFD buffer size exceeds u64".into()))?;
    let path_len = u32::try_from(path.len())
        .map_err(|_| Error::InvalidFormat("log VFD path length exceeds u32".into()))?;
    out.reserve(H5FD__log_sb_size(config)?);
    out.extend_from_slice(&config.flags.to_le_bytes());
    out.extend_from_slice(&buffer_size.to_le_bytes());
    out.extend_from_slice(&path_len.to_le_bytes());
    out.extend_from_slice(path);
    Ok(())
}

/// `H5FD__log_sb_encode`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
#[deprecated(note = "use H5FD__log_sb_encode_into")]
pub fn H5FD__log_sb_encode(config: &LogFileConfig) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(H5FD__log_sb_size(config)?);
    H5FD__log_sb_encode_into(config, &mut out)?;
    Ok(out)
}

/// `H5FD__log_sb_decode`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__log_sb_decode(bytes: &[u8]) -> Result<LogFileConfig> {
    let flags = read_le_u64_at(bytes, 0, "log VFD flags")?;
    let buffer_size = usize::try_from(read_le_u64_at(bytes, 8, "log VFD buffer size")?)
        .map_err(|_| Error::InvalidFormat("log VFD buffer size does not fit usize".into()))?;
    let path_len = read_le_u32_len_at(bytes, 16, "log VFD path length")?;
    let path_end = 20usize
        .checked_add(path_len)
        .ok_or_else(|| Error::InvalidFormat("log VFD path length overflow".into()))?;
    let path_bytes = bytes
        .get(20..path_end)
        .ok_or_else(|| Error::InvalidFormat("log VFD path is truncated".into()))?;
    if bytes.len() != path_end {
        return Err(Error::InvalidFormat(
            "log VFD config has trailing bytes".into(),
        ));
    }
    let log_path = if path_bytes.is_empty() {
        None
    } else {
        Some(PathBuf::from(std::str::from_utf8(path_bytes).map_err(
            |_| Error::InvalidFormat("log VFD path is not UTF-8".into()),
        )?))
    };
    let config = LogFileConfig {
        log_path,
        flags,
        buffer_size,
    };
    if !H5FD__log_validate_config(&config) {
        return Err(Error::InvalidFormat("invalid log VFD config".into()));
    }
    Ok(config)
}

/// `H5FD__log_open`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__log_open(_path: &str, _config: &LogFileConfig) -> Result<()> {
    Err(unsupported_vfd_driver("log"))
}

/// `H5FD__log_close`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__log_close() {}

/// `H5FD__log_cmp`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__log_cmp(left: &LogFileConfig, right: &LogFileConfig) -> Ordering {
    left.log_path
        .cmp(&right.log_path)
        .then_with(|| left.flags.cmp(&right.flags))
        .then_with(|| left.buffer_size.cmp(&right.buffer_size))
}

/// `H5FD__log_query`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__log_query() -> u64 {
    0
}

/// `H5FD__log_alloc`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__log_alloc(_size: u64) -> Result<u64> {
    Err(unsupported_vfd_driver("log"))
}

/// `H5FD__log_free`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__log_free(_addr: u64, _size: u64) -> Result<()> {
    Err(unsupported_vfd_driver("log"))
}

/// `H5FD__log_get_eoa`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__log_get_eoa() -> Result<u64> {
    Err(unsupported_vfd_driver("log"))
}

/// `H5FD__log_set_eoa`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__log_set_eoa(_eoa: u64) -> Result<()> {
    Err(unsupported_vfd_driver("log"))
}

/// `H5FD__log_get_eof`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__log_get_eof() -> Result<u64> {
    Err(unsupported_vfd_driver("log"))
}

/// `H5FD__log_get_handle`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__log_get_handle() -> Result<()> {
    Err(unsupported_vfd_driver("log"))
}

/// `H5FD__log_read`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__log_read(_addr: u64, _buf: &mut [u8]) -> Result<()> {
    Err(unsupported_vfd_driver("log"))
}

/// `H5FD__log_write`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__log_write(_addr: u64, _data: &[u8]) -> Result<()> {
    Err(unsupported_vfd_driver("log"))
}

/// `H5FD__log_truncate`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__log_truncate() -> Result<()> {
    Err(unsupported_vfd_driver("log"))
}

/// `H5FD__log_lock`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__log_lock() -> Result<()> {
    Err(unsupported_vfd_driver("log"))
}

/// `H5FD__log_unlock`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__log_unlock() -> Result<()> {
    Err(unsupported_vfd_driver("log"))
}

/// `H5FD__log_delete`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__log_delete(_path: &str) -> Result<()> {
    Err(unsupported_vfd_driver("log"))
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Ros3Config {
    pub endpoint: Option<String>,
    pub region: Option<String>,
    pub token: Option<String>,
}

/// `H5FD__ros3_register`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ros3_register() -> Result<()> {
    Err(unsupported_vfd_driver("ROS3"))
}

/// `H5FD__ros3_unregister`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ros3_unregister() {}

/// `H5FD__ros3_init`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ros3_init() -> Result<()> {
    Err(unsupported_vfd_driver("ROS3"))
}

/// `H5FD__ros3_term`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ros3_term() {}

/// `H5FD__ros3_validate_config`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ros3_validate_config(config: &Ros3Config) -> bool {
    validate_optional_config_string(config.endpoint.as_deref())
        && validate_optional_config_string(config.region.as_deref())
        && validate_optional_config_string(config.token.as_deref())
}

/// `H5FD__ros3_fapl_get`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ros3_fapl_get(config: &Ros3Config) -> Ros3Config {
    config.clone()
}

/// `H5FD__ros3_fapl_copy`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ros3_fapl_copy(config: &Ros3Config) -> Ros3Config {
    config.clone()
}

/// `H5FD__ros3_fapl_free`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ros3_fapl_free(_config: Ros3Config) {}

/// `H5FD__ros3_sb_size`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ros3_sb_size(config: &Ros3Config) -> Result<usize> {
    12usize
        .checked_add(config.endpoint.as_ref().map(String::len).unwrap_or(0))
        .and_then(|len| len.checked_add(config.region.as_ref().map(String::len).unwrap_or(0)))
        .and_then(|len| len.checked_add(config.token.as_ref().map(String::len).unwrap_or(0)))
        .ok_or_else(|| Error::InvalidFormat("ROS3 VFD config image length overflow".into()))
}

/// `H5FD__ros3_sb_encode`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ros3_sb_encode_into(config: &Ros3Config, out: &mut Vec<u8>) -> Result<()> {
    if !H5FD__ros3_validate_config(config) {
        return Err(Error::InvalidFormat("invalid ROS3 VFD config".into()));
    }
    let endpoint = config.endpoint.as_deref().unwrap_or("").as_bytes();
    let region = config.region.as_deref().unwrap_or("").as_bytes();
    let token = config.token.as_deref().unwrap_or("").as_bytes();
    let endpoint_len = u32::try_from(endpoint.len())
        .map_err(|_| Error::InvalidFormat("ROS3 VFD endpoint length exceeds u32".into()))?;
    let region_len = u32::try_from(region.len())
        .map_err(|_| Error::InvalidFormat("ROS3 VFD region length exceeds u32".into()))?;
    let token_len = u32::try_from(token.len())
        .map_err(|_| Error::InvalidFormat("ROS3 VFD token length exceeds u32".into()))?;
    out.reserve(H5FD__ros3_sb_size(config)?);
    out.extend_from_slice(&endpoint_len.to_le_bytes());
    out.extend_from_slice(&region_len.to_le_bytes());
    out.extend_from_slice(&token_len.to_le_bytes());
    out.extend_from_slice(endpoint);
    out.extend_from_slice(region);
    out.extend_from_slice(token);
    Ok(())
}

/// `H5FD__ros3_sb_encode`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
#[deprecated(note = "use H5FD__ros3_sb_encode_into")]
pub fn H5FD__ros3_sb_encode(config: &Ros3Config) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(H5FD__ros3_sb_size(config)?);
    H5FD__ros3_sb_encode_into(config, &mut out)?;
    Ok(out)
}

/// `H5FD__ros3_sb_decode`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ros3_sb_decode(bytes: &[u8]) -> Result<Ros3Config> {
    let endpoint_len = read_le_u32_len_at(bytes, 0, "ROS3 VFD endpoint length")?;
    let region_len = read_le_u32_len_at(bytes, 4, "ROS3 VFD region length")?;
    let token_len = read_le_u32_len_at(bytes, 8, "ROS3 VFD token length")?;
    let endpoint_start = 12usize;
    let region_start = endpoint_start
        .checked_add(endpoint_len)
        .ok_or_else(|| Error::InvalidFormat("ROS3 VFD endpoint length overflow".into()))?;
    let token_start = region_start
        .checked_add(region_len)
        .ok_or_else(|| Error::InvalidFormat("ROS3 VFD region length overflow".into()))?;
    let end = token_start
        .checked_add(token_len)
        .ok_or_else(|| Error::InvalidFormat("ROS3 VFD token length overflow".into()))?;
    if bytes.len() != end {
        return Err(Error::InvalidFormat(
            "ROS3 VFD config has invalid length".into(),
        ));
    }
    let decode_string = |range: std::ops::Range<usize>, context: &str| -> Result<Option<String>> {
        let raw = bytes
            .get(range)
            .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))?;
        if raw.is_empty() {
            Ok(None)
        } else {
            Ok(Some(
                std::str::from_utf8(raw)
                    .map_err(|_| Error::InvalidFormat(format!("{context} is not UTF-8")))?
                    .to_string(),
            ))
        }
    };
    let config = Ros3Config {
        endpoint: decode_string(endpoint_start..region_start, "ROS3 VFD endpoint")?,
        region: decode_string(region_start..token_start, "ROS3 VFD region")?,
        token: decode_string(token_start..end, "ROS3 VFD token")?,
    };
    if !H5FD__ros3_validate_config(&config) {
        return Err(Error::InvalidFormat("invalid ROS3 VFD config".into()));
    }
    Ok(config)
}

/// `H5FD__ros3_str_token_close`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ros3_str_token_close(token: &mut Option<String>) {
    *token = None;
}

/// `H5FD__ros3_str_token_delete`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ros3_str_token_delete(token: &mut Option<String>) {
    *token = None;
}

/// `H5FD__ros3_str_endpoint_close`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ros3_str_endpoint_close(endpoint: &mut Option<String>) {
    *endpoint = None;
}

/// `H5FD__ros3_str_endpoint_delete`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ros3_str_endpoint_delete(endpoint: &mut Option<String>) {
    *endpoint = None;
}

/// `H5FD__ros3_query`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ros3_query() -> u64 {
    0
}

/// `H5FD__ros3_get_eoa`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ros3_get_eoa() -> Result<u64> {
    Err(unsupported_vfd_driver("ROS3"))
}

/// `H5FD__ros3_set_eoa`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ros3_set_eoa(_eoa: u64) -> Result<()> {
    Err(unsupported_vfd_driver("ROS3"))
}

/// `H5FD__ros3_get_eof`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ros3_get_eof() -> Result<u64> {
    Err(unsupported_vfd_driver("ROS3"))
}

/// `H5FD__ros3_get_handle`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ros3_get_handle() -> Result<()> {
    Err(unsupported_vfd_driver("ROS3"))
}

/// `H5FD__ros3_read`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ros3_read(_addr: u64, _buf: &mut [u8]) -> Result<()> {
    Err(unsupported_vfd_driver("ROS3"))
}

/// `H5FD__ros3_write`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ros3_write(_addr: u64, _data: &[u8]) -> Result<()> {
    Err(unsupported_vfd_driver("ROS3"))
}

/// `H5FD__ros3_truncate`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ros3_truncate() -> Result<()> {
    Err(unsupported_vfd_driver("ROS3"))
}

/// `H5FD__ros3_reset_stats`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ros3_reset_stats() {}

/// `H5FD__ros3_print_stats`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ros3_print_stats_str() -> &'static str {
    "ros3 statistics unavailable: ROS3 VFD unsupported"
}

/// `H5FD__ros3_print_stats`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
#[deprecated(note = "use H5FD__ros3_print_stats_str to borrow the static status text")]
pub fn H5FD__ros3_print_stats() -> String {
    H5FD__ros3_print_stats_str().to_string()
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OnionRevisionRecord {
    pub revision: u64,
    pub address: u64,
    pub size: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OnionRevisionIndex {
    pub records: Vec<OnionRevisionRecord>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OnionHeader {
    pub version: u8,
    pub flags: u8,
    pub revision_count: u64,
}

/// `H5FD__onion_ingest_revision_record`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__onion_ingest_revision_record(
    index: &mut OnionRevisionIndex,
    record: OnionRevisionRecord,
) {
    H5FD__onion_revision_index_insert(index, record);
}

/// `H5FD__onion_archival_index_is_valid`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__onion_archival_index_is_valid(index: &OnionRevisionIndex) -> bool {
    index
        .records
        .windows(2)
        .all(|pair| pair[0].revision <= pair[1].revision)
}

/// `H5FD__onion_revision_index_destroy`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__onion_revision_index_destroy(index: &mut OnionRevisionIndex) {
    index.records.clear();
}

/// `H5FD__onion_revision_index_init`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__onion_revision_index_init() -> OnionRevisionIndex {
    OnionRevisionIndex::default()
}

/// `H5FD__onion_revision_index_resize`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__onion_revision_index_resize(index: &mut OnionRevisionIndex, capacity: usize) {
    if index.records.capacity() < capacity {
        index.records.reserve(capacity - index.records.capacity());
    }
}

/// `H5FD__onion_revision_index_insert`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__onion_revision_index_insert(
    index: &mut OnionRevisionIndex,
    record: OnionRevisionRecord,
) {
    index.records.push(record);
    index
        .records
        .sort_by(H5FD__onion_archival_index_list_sort_cmp);
}

/// `H5FD__onion_revision_record_decode`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__onion_revision_record_decode(bytes: &[u8]) -> Result<OnionRevisionRecord> {
    if bytes.len() != 24 {
        return Err(Error::InvalidFormat(
            "onion revision record image has invalid length".into(),
        ));
    }
    let revision = read_le_u64_at(bytes, 0, "onion revision record revision")?;
    let address = read_le_u64_at(bytes, 8, "onion revision record address")?;
    let size = read_le_u64_at(bytes, 16, "onion revision record size")?;
    Ok(OnionRevisionRecord {
        revision,
        address,
        size,
    })
}

/// `H5FD__onion_revision_record_encode`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__onion_revision_record_encode_into(
    record: &OnionRevisionRecord,
    out: &mut Vec<u8>,
) -> Result<()> {
    out.reserve(24);
    out.extend_from_slice(&record.revision.to_le_bytes());
    out.extend_from_slice(&record.address.to_le_bytes());
    out.extend_from_slice(&record.size.to_le_bytes());
    Ok(())
}

/// `H5FD__onion_revision_record_encode`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
#[deprecated(note = "use H5FD__onion_revision_record_encode_into")]
pub fn H5FD__onion_revision_record_encode(record: &OnionRevisionRecord) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(24);
    H5FD__onion_revision_record_encode_into(record, &mut out)?;
    Ok(out)
}

/// `H5FD__onion_archival_index_list_sort_cmp`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__onion_archival_index_list_sort_cmp(
    left: &OnionRevisionRecord,
    right: &OnionRevisionRecord,
) -> Ordering {
    left.revision
        .cmp(&right.revision)
        .then_with(|| left.address.cmp(&right.address))
        .then_with(|| left.size.cmp(&right.size))
}

/// `H5FD__onion_merge_revision_index_into_archival_index`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__onion_merge_revision_index_into_archival_index(
    archival: &mut OnionRevisionIndex,
    mut revision: OnionRevisionIndex,
) {
    archival.records.append(&mut revision.records);
    archival
        .records
        .sort_by(H5FD__onion_archival_index_list_sort_cmp);
}

/// `H5FD__onion_ingest_header`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__onion_ingest_header(bytes: &[u8]) -> Result<OnionHeader> {
    H5FD__onion_sb_decode(bytes)
}

/// `H5FD__onion_write_header`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__onion_write_header_into(header: &OnionHeader, out: &mut Vec<u8>) -> Result<()> {
    H5FD__onion_header_encode_into(header, out)
}

/// `H5FD__onion_write_header`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
#[deprecated(note = "use H5FD__onion_write_header_into")]
pub fn H5FD__onion_write_header(header: &OnionHeader) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(H5FD__onion_sb_size(header));
    H5FD__onion_write_header_into(header, &mut out)?;
    Ok(out)
}

/// `H5FD__onion_header_encode`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__onion_header_encode_into(header: &OnionHeader, out: &mut Vec<u8>) -> Result<()> {
    H5FD__onion_sb_encode_into(header, out)
}

/// `H5FD__onion_header_encode`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
#[deprecated(note = "use H5FD__onion_header_encode_into")]
pub fn H5FD__onion_header_encode(header: &OnionHeader) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(H5FD__onion_sb_size(header));
    H5FD__onion_header_encode_into(header, &mut out)?;
    Ok(out)
}

/// `H5FD__onion_register`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__onion_register() -> Result<()> {
    Err(unsupported_vfd_driver("onion"))
}

/// `H5FD__onion_unregister`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__onion_unregister() {}

/// `H5FD__onion_sb_size`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__onion_sb_size(_header: &OnionHeader) -> usize {
    10
}

/// `H5FD__onion_sb_encode`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__onion_sb_encode_into(header: &OnionHeader, out: &mut Vec<u8>) -> Result<()> {
    out.reserve(H5FD__onion_sb_size(header));
    out.push(header.version);
    out.push(header.flags);
    out.extend_from_slice(&header.revision_count.to_le_bytes());
    Ok(())
}

/// `H5FD__onion_sb_encode`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
#[deprecated(note = "use H5FD__onion_sb_encode_into")]
pub fn H5FD__onion_sb_encode(header: &OnionHeader) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(H5FD__onion_sb_size(header));
    H5FD__onion_sb_encode_into(header, &mut out)?;
    Ok(out)
}

/// `H5FD__onion_sb_decode`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__onion_sb_decode(bytes: &[u8]) -> Result<OnionHeader> {
    if bytes.len() != H5FD__onion_sb_size(&OnionHeader::default()) {
        return Err(Error::InvalidFormat(
            "onion VFD header image has invalid length".into(),
        ));
    }
    Ok(OnionHeader {
        version: *bytes
            .first()
            .ok_or_else(|| Error::InvalidFormat("onion VFD version is truncated".into()))?,
        flags: *bytes
            .get(1)
            .ok_or_else(|| Error::InvalidFormat("onion VFD flags are truncated".into()))?,
        revision_count: read_le_u64_at(bytes, 2, "onion VFD revision count")?,
    })
}

/// `H5FD__onion_commit_new_revision_record`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__onion_commit_new_revision_record(
    index: &mut OnionRevisionIndex,
    address: u64,
    size: u64,
) -> OnionRevisionRecord {
    let revision = index.records.last().map_or(0, |record| record.revision + 1);
    let record = OnionRevisionRecord {
        revision,
        address,
        size,
    };
    H5FD__onion_revision_index_insert(index, record.clone());
    record
}

/// `H5FD__onion_close`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__onion_close() {}

/// `H5FD__onion_get_eoa`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__onion_get_eoa() -> Result<u64> {
    Err(unsupported_vfd_driver("onion"))
}

/// `H5FD__onion_get_eof`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__onion_get_eof() -> Result<u64> {
    Err(unsupported_vfd_driver("onion"))
}

/// `H5FD__onion_get_legit_fapl_id`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__onion_get_legit_fapl_id() -> Result<()> {
    Err(unsupported_vfd_driver("onion"))
}

/// `H5FD__onion_create_truncate_onion`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__onion_create_truncate_onion(_path: &str) -> Result<()> {
    Err(unsupported_vfd_driver("onion"))
}

/// `H5FD__onion_remove_unused_symbols`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__onion_remove_unused_symbols(symbols: &mut Vec<String>) {
    symbols.retain(|symbol| !symbol.is_empty());
}

/// `H5FD__onion_parse_config_str`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__onion_parse_config_str(config: &str) -> HashMap<String, String> {
    config
        .split(',')
        .filter_map(|entry| entry.split_once('='))
        .map(|(key, value)| (key.trim().to_string(), value.trim().to_string()))
        .collect()
}

/// `H5FD__onion_open`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__onion_open(_path: &str) -> Result<()> {
    Err(unsupported_vfd_driver("onion"))
}

/// `H5FD__onion_open_rw`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__onion_open_rw(_path: &str) -> Result<()> {
    Err(unsupported_vfd_driver("onion"))
}

/// `H5FD__onion_read`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__onion_read(_addr: u64, _buf: &mut [u8]) -> Result<()> {
    Err(unsupported_vfd_driver("onion"))
}

/// `H5FD__onion_set_eoa`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__onion_set_eoa(_eoa: u64) -> Result<()> {
    Err(unsupported_vfd_driver("onion"))
}

/// `H5FD__onion_write`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__onion_write(_addr: u64, _data: &[u8]) -> Result<()> {
    Err(unsupported_vfd_driver("onion"))
}

/// `H5FD__onion_ctl`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__onion_ctl(_opcode: u64) -> Result<()> {
    Err(unsupported_vfd_driver("onion"))
}

/// VFD: H5FDonion get revision count.
#[allow(non_snake_case)]
pub fn H5FDonion_get_revision_count(header: &OnionHeader) -> u64 {
    header.revision_count
}

/// VFD: get onion revision count.
#[allow(non_snake_case)]
pub fn H5FD__get_onion_revision_count(header: &OnionHeader) -> u64 {
    H5FDonion_get_revision_count(header)
}

/// `H5FD__onion_write_final_history`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__onion_write_final_history_into(
    index: &OnionRevisionIndex,
    out: &mut Vec<u8>,
) -> Result<()> {
    H5FD__onion_history_encode_into(index, out)
}

/// `H5FD__onion_write_final_history`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
#[deprecated(note = "use H5FD__onion_write_final_history_into")]
pub fn H5FD__onion_write_final_history(index: &OnionRevisionIndex) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(H5FD__onion_history_size(index)?);
    H5FD__onion_write_final_history_into(index, &mut out)?;
    Ok(out)
}

/// `H5FD__onion_ingest_history`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__onion_ingest_history(bytes: &[u8]) -> Result<OnionRevisionIndex> {
    if bytes.len() % 24 != 0 {
        return Err(Error::InvalidFormat(
            "onion revision history has a partial trailing record".into(),
        ));
    }
    let mut index = OnionRevisionIndex::default();
    for chunk in bytes.chunks_exact(24) {
        let record = H5FD__onion_revision_record_decode(chunk)?;
        H5FD__onion_revision_index_insert(&mut index, record);
    }
    Ok(index)
}

/// `H5FD__onion_write_history`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__onion_write_history_into(index: &OnionRevisionIndex, out: &mut Vec<u8>) -> Result<()> {
    H5FD__onion_history_encode_into(index, out)
}

/// `H5FD__onion_write_history`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
#[deprecated(note = "use H5FD__onion_write_history_into")]
pub fn H5FD__onion_write_history(index: &OnionRevisionIndex) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(H5FD__onion_history_size(index)?);
    H5FD__onion_write_history_into(index, &mut out)?;
    Ok(out)
}

/// `H5FD__onion_history_encode`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__onion_history_size(index: &OnionRevisionIndex) -> Result<usize> {
    index
        .records
        .len()
        .checked_mul(24)
        .ok_or_else(|| Error::InvalidFormat("onion revision history length overflow".into()))
}

/// `H5FD__onion_history_encode`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__onion_history_encode_into(
    index: &OnionRevisionIndex,
    out: &mut Vec<u8>,
) -> Result<()> {
    out.reserve(H5FD__onion_history_size(index)?);
    for record in &index.records {
        H5FD__onion_revision_record_encode_into(record, out)?;
    }
    Ok(())
}

/// `H5FD__onion_history_encode`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
#[deprecated(note = "use H5FD__onion_history_encode_into")]
pub fn H5FD__onion_history_encode(index: &OnionRevisionIndex) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(H5FD__onion_history_size(index)?);
    H5FD__onion_history_encode_into(index, &mut out)?;
    Ok(out)
}

/// VFD: supports swmr test.
#[allow(non_snake_case)]
pub fn H5FD__supports_swmr_test(driver: FileDriverKind) -> bool {
    matches!(driver, FileDriverKind::Sec2 | FileDriverKind::Stdio)
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SubfilingConfig {
    pub ioc_count: u32,
    pub stripe_size: u64,
    pub stripe_count: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IocConfig {
    pub thread_pool_size: usize,
    pub queue_depth: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SubfilingObject {
    pub id: u64,
    pub path: PathBuf,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IocQueueEntry {
    pub request: VfdIoRequest,
    pub complete: bool,
}

/// `H5FD__multi_register`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__multi_register() -> Result<()> {
    Err(unsupported_vfd_driver("multi"))
}

/// `H5FD__multi_unregister`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__multi_unregister() {}

/// `H5FD__ioc_calculate_target_ioc`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ioc_calculate_target_ioc(addr: u64, config: &SubfilingConfig) -> u32 {
    let count = config.ioc_count.max(1);
    let stripe = config.stripe_size.max(1);
    u32::try_from((addr / stripe) % u64::from(count)).unwrap_or(0)
}

/// `H5FD__ioc_write_independent_async`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ioc_write_independent_async(_request: &VfdIoRequest) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling IOC async"))
}

/// `H5FD__ioc_read_independent_async`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ioc_read_independent_async(_request: &mut VfdIoRequest) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling IOC async"))
}

/// `H5FD__ioc_async_completion`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ioc_async_completion(entry: &mut IocQueueEntry) {
    entry.complete = true;
}

/// `H5FD__subfiling_new_object_id`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_new_object_id(previous: u64) -> u64 {
    H5FD__subfiling_new_object_id_checked(previous).unwrap_or(u64::MAX)
}

/// `H5FD__subfiling_new_object_id_checked`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_new_object_id_checked(previous: u64) -> Result<u64> {
    previous
        .checked_add(1)
        .ok_or_else(|| Error::InvalidFormat("subfiling object id overflow".into()))
}

/// `H5FD__subfiling_get_object`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_get_object(
    objects: &[SubfilingObject],
    id: u64,
) -> Option<&SubfilingObject> {
    objects.iter().find(|object| object.id == id)
}

/// `H5FD__subfiling_free_object`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_free_object(_object: SubfilingObject) {}

/// `H5FD__subfiling_free_context`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_free_context(_config: SubfilingConfig) {}

/// `H5FD__subfiling_free_topology`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_free_topology(_objects: Vec<SubfilingObject>) {}

/// `H5FD__subfiling_open_stub_file`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_open_stub_file(_path: &str) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling"))
}

/// `H5FD__subfiling_open_subfiles`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_open_subfiles(_path: &str, _config: &SubfilingConfig) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling"))
}

/// `H5FD__subfiling_setup_context`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_setup_context(config: &SubfilingConfig) -> SubfilingConfig {
    config.clone()
}

/// `H5FD__subfiling_init_app_topology`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_init_app_topology_into(config: &SubfilingConfig, out: &mut Vec<u32>) {
    out.extend(0..config.ioc_count);
}

/// `H5FD__subfiling_init_app_topology`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
#[deprecated(note = "use H5FD__subfiling_init_app_topology_into")]
pub fn H5FD__subfiling_init_app_topology(config: &SubfilingConfig) -> Vec<u32> {
    let mut out = Vec::with_capacity(config.ioc_count as usize);
    H5FD__subfiling_init_app_topology_into(config, &mut out);
    out
}

/// `H5FD__subfiling_get_ioc_selection_criteria_from_env`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_get_ioc_selection_criteria_from_env() -> Option<String> {
    std::env::var("HDF5_SUBFILING_IOC_SELECTION").ok()
}

/// `H5FD__subfiling_find_cached_topology_info`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_find_cached_topology_info_view() -> Option<&'static [u32]> {
    None
}

/// `H5FD__subfiling_find_cached_topology_info`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
#[deprecated(note = "use H5FD__subfiling_find_cached_topology_info_view")]
pub fn H5FD__subfiling_find_cached_topology_info() -> Option<Vec<u32>> {
    H5FD__subfiling_find_cached_topology_info_view().map(<[u32]>::to_vec)
}

/// `H5FD__subfiling_init_app_layout`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_init_app_layout(config: &SubfilingConfig) -> SubfilingConfig {
    config.clone()
}

/// `H5FD__subfiling_gather_topology_info`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_gather_topology_info_into(config: &SubfilingConfig, out: &mut Vec<u32>) {
    H5FD__subfiling_init_app_topology_into(config, out);
}

/// `H5FD__subfiling_gather_topology_info`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
#[deprecated(note = "use H5FD__subfiling_gather_topology_info_into")]
pub fn H5FD__subfiling_gather_topology_info(config: &SubfilingConfig) -> Vec<u32> {
    let mut out = Vec::with_capacity(config.ioc_count as usize);
    H5FD__subfiling_gather_topology_info_into(config, &mut out);
    out
}

/// `H5FD__subfiling_compare_layout_nodelocal`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_compare_layout_nodelocal(
    left: &SubfilingConfig,
    right: &SubfilingConfig,
) -> Ordering {
    left.ioc_count
        .cmp(&right.ioc_count)
        .then_with(|| left.stripe_size.cmp(&right.stripe_size))
        .then_with(|| left.stripe_count.cmp(&right.stripe_count))
}

/// `H5FD__subfiling_identify_ioc_ranks`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_identify_ioc_ranks_into(config: &SubfilingConfig, out: &mut Vec<u32>) {
    H5FD__subfiling_init_app_topology_into(config, out);
}

/// `H5FD__subfiling_identify_ioc_ranks`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
#[deprecated(note = "use H5FD__subfiling_identify_ioc_ranks_into")]
pub fn H5FD__subfiling_identify_ioc_ranks(config: &SubfilingConfig) -> Vec<u32> {
    let mut out = Vec::with_capacity(config.ioc_count as usize);
    H5FD__subfiling_identify_ioc_ranks_into(config, &mut out);
    out
}

/// `H5FD__subfiling_init_context`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_init_context() -> SubfilingConfig {
    H5FD__subfiling_get_default_config()
}

/// `H5FD__subfiling_record_fid_map_entry`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_record_fid_map_entry(_fid: u64, _object: &SubfilingObject) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling"))
}

/// `H5FD__subfiling_get_default_ioc_config`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_get_default_ioc_config() -> IocConfig {
    IocConfig {
        thread_pool_size: 1,
        queue_depth: 64,
    }
}

/// `H5FD__subfiling_ioc_open_files`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_ioc_open_files(_config: &SubfilingConfig) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling IOC"))
}

/// `H5FD__subfiling_create_config_file`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_create_config_file(_path: &str, _config: &SubfilingConfig) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling config file"))
}

/// `H5FD__subfiling_open_config_file`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_open_config_file(_path: &str) -> Result<SubfilingConfig> {
    Err(unsupported_vfd_driver("subfiling config file"))
}

/// `H5FD__subfiling_get_config_from_file`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_get_config_from_file(_path: &str) -> Result<SubfilingConfig> {
    Err(unsupported_vfd_driver("subfiling config file"))
}

/// `H5FD__subfiling_resolve_pathname`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_resolve_pathname(path: &str) -> PathBuf {
    PathBuf::from(path)
}

/// `H5FD__subfiling_close_subfiles`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_close_subfiles() {}

/// `H5FD__subfiling_set_config_prop`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_set_config_prop(config: &mut SubfilingConfig, stripe_size: u64) {
    config.stripe_size = stripe_size;
}

/// `H5FD__subfiling_log`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_log_to_writer(writer: &mut impl fmt::Write, message: &str) -> fmt::Result {
    writeln!(writer, "{message}")
}

/// `H5FD__subfiling_log`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
#[deprecated(note = "use H5FD__subfiling_log_to_writer")]
pub fn H5FD__subfiling_log(message: &str) -> String {
    let mut out = String::new();
    H5FD__subfiling_log_to_writer(&mut out, message).expect("writing to String should not fail");
    out
}

/// `H5FD__subfiling_log_nonewline`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_log_nonewline_str(message: &str) -> &str {
    message
}

/// `H5FD__subfiling_log_nonewline`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_log_nonewline_to_writer(
    writer: &mut impl fmt::Write,
    message: &str,
) -> fmt::Result {
    fmt::Write::write_str(writer, message)
}

/// `H5FD__subfiling_log_nonewline`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
#[deprecated(
    note = "use H5FD__subfiling_log_nonewline_str or H5FD__subfiling_log_nonewline_to_writer"
)]
pub fn H5FD__subfiling_log_nonewline(message: &str) -> String {
    H5FD__subfiling_log_nonewline_str(message).to_string()
}

/// `H5FD__ioc_register`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ioc_register() -> Result<()> {
    Err(unsupported_vfd_driver("subfiling IOC"))
}

/// `H5FD__ioc_unregister`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ioc_unregister() {}

/// `H5FD__ioc_init`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ioc_init() -> Result<()> {
    Err(unsupported_vfd_driver("subfiling IOC"))
}

/// `H5FD__ioc_term`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ioc_term() {}

/// `H5FD__ioc_validate_config`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ioc_validate_config(config: &IocConfig) -> bool {
    config.thread_pool_size > 0 && config.queue_depth > 0
}

/// `H5FD__ioc_sb_size`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ioc_sb_size(_config: &IocConfig) -> usize {
    16
}

/// `H5FD__ioc_sb_encode`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ioc_sb_encode_into(config: &IocConfig, out: &mut Vec<u8>) -> Result<()> {
    if !H5FD__ioc_validate_config(config) {
        return Err(Error::InvalidFormat("invalid subfiling IOC config".into()));
    }
    let thread_pool_size = u64::try_from(config.thread_pool_size)
        .map_err(|_| Error::InvalidFormat("subfiling IOC thread pool size exceeds u64".into()))?;
    let queue_depth = u64::try_from(config.queue_depth)
        .map_err(|_| Error::InvalidFormat("subfiling IOC queue depth exceeds u64".into()))?;
    out.reserve(H5FD__ioc_sb_size(config));
    out.extend_from_slice(&thread_pool_size.to_le_bytes());
    out.extend_from_slice(&queue_depth.to_le_bytes());
    Ok(())
}

/// `H5FD__ioc_sb_encode`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
#[deprecated(note = "use H5FD__ioc_sb_encode_into")]
pub fn H5FD__ioc_sb_encode(config: &IocConfig) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(H5FD__ioc_sb_size(config));
    H5FD__ioc_sb_encode_into(config, &mut out)?;
    Ok(out)
}

/// `H5FD__ioc_sb_decode`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ioc_sb_decode(bytes: &[u8]) -> Result<IocConfig> {
    if bytes.len() != H5FD__ioc_sb_size(&IocConfig::default()) {
        return Err(Error::InvalidFormat(
            "subfiling IOC config image has invalid length".into(),
        ));
    }
    let thread_pool_size =
        usize::try_from(read_le_u64_at(bytes, 0, "subfiling IOC thread pool size")?).map_err(
            |_| Error::InvalidFormat("subfiling IOC thread pool size does not fit usize".into()),
        )?;
    let queue_depth = usize::try_from(read_le_u64_at(bytes, 8, "subfiling IOC queue depth")?)
        .map_err(|_| Error::InvalidFormat("subfiling IOC queue depth does not fit usize".into()))?;
    let config = IocConfig {
        thread_pool_size,
        queue_depth,
    };
    if !H5FD__ioc_validate_config(&config) {
        return Err(Error::InvalidFormat("invalid subfiling IOC config".into()));
    }
    Ok(config)
}

/// `H5FD__ioc_fapl_get`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ioc_fapl_get(config: &IocConfig) -> IocConfig {
    config.clone()
}

/// `H5FD__ioc_fapl_copy`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ioc_fapl_copy(config: &IocConfig) -> IocConfig {
    config.clone()
}

/// `H5FD__ioc_fapl_free`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ioc_fapl_free(_config: IocConfig) {}

/// `H5FD__ioc_open`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ioc_open(_path: &str, _config: &IocConfig) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling IOC"))
}

/// `H5FD__ioc_close_int`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ioc_close_int() {}

/// `H5FD__ioc_close`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ioc_close() {}

/// `H5FD__ioc_query`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ioc_query() -> u64 {
    0
}

/// `H5FD__ioc_get_eoa`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ioc_get_eoa() -> Result<u64> {
    Err(unsupported_vfd_driver("subfiling IOC"))
}

/// `H5FD__ioc_set_eoa`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ioc_set_eoa(_eoa: u64) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling IOC"))
}

/// `H5FD__ioc_get_eof`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ioc_get_eof() -> Result<u64> {
    Err(unsupported_vfd_driver("subfiling IOC"))
}

/// `H5FD__ioc_read`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ioc_read(_addr: u64, _buf: &mut [u8]) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling IOC"))
}

/// `H5FD__ioc_write`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ioc_write(_addr: u64, _data: &[u8]) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling IOC"))
}

/// `H5FD__ioc_write_vector`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ioc_write_vector(_requests: &[VfdIoRequest]) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling IOC"))
}

/// `H5FD__ioc_truncate`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ioc_truncate() -> Result<()> {
    Err(unsupported_vfd_driver("subfiling IOC"))
}

/// `H5FD__ioc_delete`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ioc_delete(_path: &str) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling IOC"))
}

/// `H5FD__ioc_write_vector_internal`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ioc_write_vector_internal(_requests: &[VfdIoRequest]) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling IOC"))
}

/// `H5FD__ioc_read_vector_internal`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ioc_read_vector_internal(_requests: &mut [VfdIoRequest]) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling IOC"))
}

/// `H5FD__subfiling__truncate_sub_files`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling__truncate_sub_files(_config: &SubfilingConfig) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling"))
}

/// `H5FD__subfiling__get_real_eof`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling__get_real_eof() -> Result<u64> {
    Err(unsupported_vfd_driver("subfiling"))
}

/// `H5FD__ioc_init_threads`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ioc_init_threads() -> Result<()> {
    Err(unsupported_vfd_driver("subfiling IOC threads"))
}

/// `H5FD__ioc_finalize_threads`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ioc_finalize_threads() {}

/// `H5FD__ioc_thread_main`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ioc_thread_main() -> Result<()> {
    Err(unsupported_vfd_driver("subfiling IOC threads"))
}

/// VFD: translate opcode.
pub fn translate_opcode(op: u64) -> &'static str {
    match op {
        0 => "READ_OP",
        1 => "WRITE_OP",
        2 => "OPEN_OP",
        3 => "CLOSE_OP",
        4 => "TRUNC_OP",
        5 => "GET_EOF_OP",
        6 => "FINI_OP",
        7 => "LOGGING_OP",
        _ => "unknown",
    }
}

/// `H5FD__ioc_handle_work_request`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ioc_handle_work_request(_entry: &mut IocQueueEntry) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling IOC"))
}

/// `H5FD__ioc_send_ack_to_client`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ioc_send_ack_to_client(entry: &mut IocQueueEntry) {
    entry.complete = true;
}

/// `H5FD__ioc_send_nack_to_client`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ioc_send_nack_to_client(entry: &mut IocQueueEntry) {
    entry.complete = false;
}

/// `H5FD__ioc_file_queue_write_indep`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ioc_file_queue_write_indep(request: VfdIoRequest) -> IocQueueEntry {
    IocQueueEntry {
        request,
        complete: false,
    }
}

/// `H5FD__ioc_file_queue_read_indep`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ioc_file_queue_read_indep(request: VfdIoRequest) -> IocQueueEntry {
    IocQueueEntry {
        request,
        complete: false,
    }
}

/// `H5FD__ioc_file_truncate`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ioc_file_truncate() -> Result<()> {
    Err(unsupported_vfd_driver("subfiling IOC"))
}

/// `H5FD__ioc_file_report_eof`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ioc_file_report_eof() -> Result<u64> {
    Err(unsupported_vfd_driver("subfiling IOC"))
}

/// `H5FD__ioc_io_queue_alloc_entry`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ioc_io_queue_alloc_entry(request: VfdIoRequest) -> IocQueueEntry {
    IocQueueEntry {
        request,
        complete: false,
    }
}

/// `H5FD__ioc_io_queue_add_entry`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ioc_io_queue_add_entry(queue: &mut Vec<IocQueueEntry>, entry: IocQueueEntry) {
    queue.push(entry);
}

/// `H5FD__ioc_io_queue_dispatch_eligible_entries`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ioc_io_queue_dispatch_eligible_entries(queue: &mut [IocQueueEntry]) {
    for entry in queue {
        entry.complete = true;
    }
}

/// `H5FD__ioc_io_queue_complete_entry`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__ioc_io_queue_complete_entry(entry: &mut IocQueueEntry) {
    entry.complete = true;
}

/// `H5FD__subfiling_register`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_register() -> Result<()> {
    Err(unsupported_vfd_driver("subfiling"))
}

/// `H5FD__subfiling_unregister`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_unregister() {}

/// `H5FD__subfiling_init`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_init() -> Result<()> {
    Err(unsupported_vfd_driver("subfiling"))
}

/// `H5FD__subfiling_term`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_term() {}

/// VFD: H5FDsubfiling get file mapping.
#[allow(non_snake_case)]
pub fn H5FDsubfiling_get_file_mapping_into(config: &SubfilingConfig, out: &mut Vec<u32>) {
    H5FD__subfiling_identify_ioc_ranks_into(config, out);
}

/// VFD: H5FDsubfiling get file mapping.
#[allow(non_snake_case)]
#[deprecated(note = "use H5FDsubfiling_get_file_mapping_into")]
pub fn H5FDsubfiling_get_file_mapping(config: &SubfilingConfig) -> Vec<u32> {
    let mut out = Vec::with_capacity(config.ioc_count as usize);
    H5FDsubfiling_get_file_mapping_into(config, &mut out);
    out
}

/// `H5FD__subfiling_get_default_config`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_get_default_config() -> SubfilingConfig {
    SubfilingConfig {
        ioc_count: 1,
        stripe_size: 64 * 1024 * 1024,
        stripe_count: 1,
    }
}

/// `H5FD__subfiling_sb_size`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_sb_size(_config: &SubfilingConfig) -> usize {
    16
}

/// `H5FD__subfiling_sb_encode`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_sb_encode_into(config: &SubfilingConfig, out: &mut Vec<u8>) -> Result<()> {
    if config.stripe_size == 0 || config.ioc_count == 0 || config.stripe_count == 0 {
        return Err(Error::InvalidFormat("invalid subfiling VFD config".into()));
    }
    out.reserve(H5FD__subfiling_sb_size(config));
    out.extend_from_slice(&config.stripe_size.to_le_bytes());
    out.extend_from_slice(&config.ioc_count.to_le_bytes());
    out.extend_from_slice(&config.stripe_count.to_le_bytes());
    Ok(())
}

/// `H5FD__subfiling_sb_encode`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
#[deprecated(note = "use H5FD__subfiling_sb_encode_into")]
pub fn H5FD__subfiling_sb_encode(config: &SubfilingConfig) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(H5FD__subfiling_sb_size(config));
    H5FD__subfiling_sb_encode_into(config, &mut out)?;
    Ok(out)
}

/// `H5FD__subfiling_sb_decode`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_sb_decode(bytes: &[u8]) -> Result<SubfilingConfig> {
    if bytes.len() != H5FD__subfiling_sb_size(&SubfilingConfig::default()) {
        return Err(Error::InvalidFormat(
            "subfiling VFD config image has invalid length".into(),
        ));
    }
    let config = SubfilingConfig {
        stripe_size: read_le_u64_at(bytes, 0, "subfiling VFD stripe size")?,
        ioc_count: read_le_u32_at(bytes, 8, "subfiling VFD IOC count")?,
        stripe_count: read_le_u32_at(bytes, 12, "subfiling VFD stripe count")?,
    };
    if config.stripe_size == 0 || config.ioc_count == 0 || config.stripe_count == 0 {
        return Err(Error::InvalidFormat("invalid subfiling VFD config".into()));
    }
    Ok(config)
}

/// `H5FD__subfiling_fapl_get`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_fapl_get(config: &SubfilingConfig) -> SubfilingConfig {
    config.clone()
}

/// VFD: copy plist.
#[allow(non_snake_case)]
pub fn H5FD__copy_plist<T: Clone>(value: &T) -> T {
    value.clone()
}

/// `H5FD__subfiling_fapl_copy`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_fapl_copy(config: &SubfilingConfig) -> SubfilingConfig {
    config.clone()
}

/// `H5FD__subfiling_fapl_free`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_fapl_free(_config: SubfilingConfig) {}

/// `H5FD__subfiling_open`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_open(_path: &str, _config: &SubfilingConfig) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling"))
}

/// `H5FD__subfiling_close_int`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_close_int() {}

/// `H5FD__subfiling_close`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_close() {}

/// `H5FD__subfiling_cmp`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_cmp(left: &SubfilingConfig, right: &SubfilingConfig) -> Ordering {
    H5FD__subfiling_compare_layout_nodelocal(left, right)
}

/// `H5FD__subfiling_query`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_query() -> u64 {
    0
}

/// `H5FD__subfiling_get_eoa`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_get_eoa() -> Result<u64> {
    Err(unsupported_vfd_driver("subfiling"))
}

/// `H5FD__subfiling_set_eoa`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_set_eoa(_eoa: u64) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling"))
}

/// `H5FD__subfiling_get_eof`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_get_eof() -> Result<u64> {
    Err(unsupported_vfd_driver("subfiling"))
}

/// `H5FD__subfiling_get_handle`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_get_handle() -> Result<()> {
    Err(unsupported_vfd_driver("subfiling"))
}

/// `H5FD__subfiling_write_vector`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_write_vector(_requests: &[VfdIoRequest]) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling"))
}

/// `H5FD__subfiling_truncate`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_truncate() -> Result<()> {
    Err(unsupported_vfd_driver("subfiling"))
}

/// `H5FD__subfiling_delete`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_delete(_path: &str) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling"))
}

/// `H5FD__subfiling_ctl`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_ctl(_opcode: u64) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling"))
}

/// `H5FD__subfiling_io_helper`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_io_helper(_requests: &[VfdIoRequest]) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling"))
}

/// `H5FD__subfiling_mirror_writes_to_stub`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_mirror_writes_to_stub(_enabled: bool) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling"))
}

/// `H5FD__subfiling_generate_io_vectors`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_generate_io_vectors_view(requests: &[VfdIoRequest]) -> &[VfdIoRequest] {
    requests
}

/// `H5FD__subfiling_generate_io_vectors`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_generate_io_vectors_into(
    requests: &[VfdIoRequest],
    out: &mut Vec<VfdIoRequest>,
) {
    out.extend_from_slice(requests);
}

/// `H5FD__subfiling_generate_io_vectors`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
#[deprecated(
    note = "use H5FD__subfiling_generate_io_vectors_view or H5FD__subfiling_generate_io_vectors_into"
)]
pub fn H5FD__subfiling_generate_io_vectors(requests: &[VfdIoRequest]) -> Vec<VfdIoRequest> {
    let mut out = Vec::with_capacity(requests.len());
    H5FD__subfiling_generate_io_vectors_into(requests, &mut out);
    out
}

/// `H5FD__subfiling_get_iovec_sizes`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_iovec_sizes_iter(
    requests: &[VfdIoRequest],
) -> impl Iterator<Item = usize> + '_ {
    requests.iter().map(|request| request.bytes.len())
}

/// `H5FD__subfiling_get_iovec_sizes`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_get_iovec_sizes_into(requests: &[VfdIoRequest], out: &mut Vec<usize>) {
    out.extend(H5FD__subfiling_iovec_sizes_iter(requests));
}

/// `H5FD__subfiling_get_iovec_sizes`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
#[deprecated(note = "use H5FD__subfiling_get_iovec_sizes_into")]
pub fn H5FD__subfiling_get_iovec_sizes(requests: &[VfdIoRequest]) -> Vec<usize> {
    let mut out = Vec::with_capacity(requests.len());
    H5FD__subfiling_get_iovec_sizes_into(requests, &mut out);
    out
}

/// `H5FD__subfiling_translate_io_req_to_iovec`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_translate_io_req_to_iovec_ref(request: &VfdIoRequest) -> &VfdIoRequest {
    request
}

/// `H5FD__subfiling_translate_io_req_to_iovec`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
#[deprecated(note = "use H5FD__subfiling_translate_io_req_to_iovec_ref")]
pub fn H5FD__subfiling_translate_io_req_to_iovec(request: &VfdIoRequest) -> VfdIoRequest {
    H5FD__subfiling_translate_io_req_to_iovec_ref(request).clone()
}

/// `H5FD__subfiling_iovec_fill_first`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_iovec_fill_first_ref(requests: &[VfdIoRequest]) -> Option<&VfdIoRequest> {
    requests.first()
}

/// `H5FD__subfiling_iovec_fill_first`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
#[deprecated(note = "use H5FD__subfiling_iovec_fill_first_ref")]
pub fn H5FD__subfiling_iovec_fill_first(requests: &[VfdIoRequest]) -> Option<VfdIoRequest> {
    H5FD__subfiling_iovec_fill_first_ref(requests).cloned()
}

/// `H5FD__subfiling_iovec_fill_last`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_iovec_fill_last_ref(requests: &[VfdIoRequest]) -> Option<&VfdIoRequest> {
    requests.last()
}

/// `H5FD__subfiling_iovec_fill_last`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
#[deprecated(note = "use H5FD__subfiling_iovec_fill_last_ref")]
pub fn H5FD__subfiling_iovec_fill_last(requests: &[VfdIoRequest]) -> Option<VfdIoRequest> {
    H5FD__subfiling_iovec_fill_last_ref(requests).cloned()
}

/// `H5FD__subfiling_iovec_fill_first_last`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_iovec_fill_first_last_into(
    requests: &[VfdIoRequest],
    out: &mut Vec<VfdIoRequest>,
) {
    if let Some(first) = H5FD__subfiling_iovec_fill_first_ref(requests) {
        out.push(first.clone());
    }
    if requests.len() > 1 {
        if let Some(last) = H5FD__subfiling_iovec_fill_last_ref(requests) {
            out.push(last.clone());
        }
    }
}

/// `H5FD__subfiling_iovec_fill_first_last`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
#[deprecated(note = "use H5FD__subfiling_iovec_fill_first_last_into")]
pub fn H5FD__subfiling_iovec_fill_first_last(requests: &[VfdIoRequest]) -> Vec<VfdIoRequest> {
    let mut out = Vec::with_capacity(requests.len().min(2));
    H5FD__subfiling_iovec_fill_first_last_into(requests, &mut out);
    out
}

/// `H5FD__subfiling_iovec_fill_uniform`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_iovec_fill_uniform_view(requests: &[VfdIoRequest]) -> &[VfdIoRequest] {
    requests
}

/// `H5FD__subfiling_iovec_fill_uniform`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_iovec_fill_uniform_into(
    requests: &[VfdIoRequest],
    out: &mut Vec<VfdIoRequest>,
) {
    out.extend_from_slice(requests);
}

/// `H5FD__subfiling_iovec_fill_uniform`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
#[deprecated(
    note = "use H5FD__subfiling_iovec_fill_uniform_view or H5FD__subfiling_iovec_fill_uniform_into"
)]
pub fn H5FD__subfiling_iovec_fill_uniform(requests: &[VfdIoRequest]) -> Vec<VfdIoRequest> {
    let mut out = Vec::with_capacity(requests.len());
    H5FD__subfiling_iovec_fill_uniform_into(requests, &mut out);
    out
}

/// `H5FD__subfiling_cast_to_void`: distributed/cloud driver, not supported by the pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5FD__subfiling_cast_to_void<T>(value: T) -> T {
    value
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{
        Error, FamilyFileConfig, FileDriverKind, HdfsConfig, IocConfig, LocalFileDriver,
        LogFileConfig, MirrorXmit, MirrorXmitRef, MultiFileConfig, OnionHeader, OnionRevisionIndex,
        OnionRevisionRecord, Ros3Config, SplitterFileConfig, SubfilingConfig, VfdMemType,
    };

    #[test]
    fn core_vfd_reads_writes_and_allocates() {
        let mut driver = LocalFileDriver {
            kind: super::FileDriverKind::Core,
            path: None,
            file: None,
            core_image: Vec::new(),
            eoa: 0,
            locked: false,
        };
        let addr = driver.alloc(4).unwrap();
        assert_eq!(addr, 0);
        driver.core_write(addr, b"abcd").unwrap();
        let mut out = [0; 4];
        driver.core_read(0, &mut out).unwrap();
        assert_eq!(&out, b"abcd");
        assert_eq!(driver.core_get_eof_checked().unwrap(), 4);
    }

    #[test]
    fn public_h5fd_runtime_wrappers_and_driver_query_are_explicit() {
        let mut registry = super::H5FD_init();
        assert_eq!(
            super::H5FDdriver_query(&registry, super::file_driver_kind_id(FileDriverKind::Sec2))
                .unwrap(),
            0x01
        );
        assert_eq!(
            super::H5FDdriver_query(&registry, super::file_driver_kind_id(FileDriverKind::Core))
                .unwrap(),
            0x03
        );
        assert!(matches!(
            super::H5FDdriver_query(&registry, 99).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        super::H5FDregister(&mut registry, "custom", 99);
        assert!(matches!(
            super::H5FDdriver_query(&registry, 99).unwrap_err(),
            Error::Unsupported(_)
        ));

        let mut driver = LocalFileDriver {
            kind: super::FileDriverKind::Core,
            path: None,
            file: None,
            core_image: Vec::new(),
            eoa: 0,
            locked: false,
        };
        assert_eq!(super::H5FD_alloc(&mut driver, 2).unwrap(), 0);
        assert_eq!(super::H5FDget_eoa(&driver), 2);
        assert!(super::H5FD_try_extend(&mut driver, 0, 2, 4).unwrap());
        assert_eq!(super::H5FDget_eoa(&driver), 6);
        super::H5FD_free(&mut driver, 0, 6);
        assert_eq!(super::H5FDget_eoa(&driver), 6);

        driver.core_write(0, b"abcdef").unwrap();
        super::H5FDset_eoa(&mut driver, 3);
        super::H5FDtruncate(&mut driver).unwrap();
        assert_eq!(driver.core_get_handle().unwrap(), b"abc");

        super::H5FDlock(&mut driver, true).unwrap();
        assert!(driver.locked);
        super::H5FDunlock(&mut driver).unwrap();
        assert!(!driver.locked);
    }

    #[test]
    fn family_multi_splitter_and_log_configs_round_trip() {
        let family = FamilyFileConfig {
            member_size: 4096,
            printf_filename: "member-%03d.h5".into(),
        };
        let mut family_bytes = Vec::new();
        super::H5FD__family_sb_encode_into(&family, &mut family_bytes).unwrap();
        assert_eq!(
            super::H5FD__family_sb_size(&family).unwrap(),
            family_bytes.len()
        );
        assert_eq!(
            super::H5FD__family_sb_decode(&family_bytes).unwrap(),
            FamilyFileConfig {
                member_size: family.member_size,
                printf_filename: FamilyFileConfig::default().printf_filename,
            }
        );
        assert_eq!(family_bytes, 4096u64.to_le_bytes());
        let family_default_size = FamilyFileConfig {
            member_size: 0,
            printf_filename: "member-%03d.h5".into(),
        };
        let mut family_default_size_bytes = Vec::new();
        super::H5FD__family_sb_encode_into(&family_default_size, &mut family_default_size_bytes)
            .unwrap();
        assert_eq!(family_default_size_bytes, 0u64.to_le_bytes());
        let mut invalid_family_bytes = Vec::new();
        assert!(super::H5FD__family_sb_encode_into(
            &FamilyFileConfig {
                member_size: 4096,
                printf_filename: String::new(),
            },
            &mut invalid_family_bytes
        )
        .is_err());

        let mut multi = MultiFileConfig::default();
        multi
            .memb_map
            .insert(VfdMemType::Default, FileDriverKind::Sec2);
        multi
            .memb_map
            .insert(VfdMemType::RawData, FileDriverKind::Core);
        let mut multi_bytes = Vec::new();
        super::H5FD_multi_sb_encode_into(&multi, &mut multi_bytes).unwrap();
        assert_eq!(
            super::H5FD_multi_sb_size(&multi).unwrap(),
            multi_bytes.len()
        );
        assert_eq!(super::H5FD_multi_sb_decode(&multi_bytes).unwrap(), multi);
        let mut invalid_multi_bytes = Vec::new();
        assert!(super::H5FD_multi_sb_encode_into(
            &MultiFileConfig::default(),
            &mut invalid_multi_bytes
        )
        .is_err());

        let splitter = SplitterFileConfig {
            write_only_path: Some(PathBuf::from("mirror.h5")),
            ignore_wo_errors: true,
        };
        let mut splitter_bytes = Vec::new();
        super::H5FD__splitter_sb_encode_into(&splitter, &mut splitter_bytes).unwrap();
        assert_eq!(
            super::H5FD__splitter_sb_size(&splitter).unwrap(),
            splitter_bytes.len()
        );
        assert_eq!(
            super::H5FD__splitter_sb_decode(&splitter_bytes).unwrap(),
            splitter
        );
        let mut invalid_splitter_bytes = Vec::new();
        assert!(super::H5FD__splitter_sb_encode_into(
            &SplitterFileConfig {
                write_only_path: Some(PathBuf::from("")),
                ignore_wo_errors: true,
            },
            &mut invalid_splitter_bytes
        )
        .is_err());

        let log = LogFileConfig {
            log_path: Some(PathBuf::from("driver.log")),
            flags: 0x55,
            buffer_size: 8192,
        };
        let mut log_bytes = Vec::new();
        super::H5FD__log_sb_encode_into(&log, &mut log_bytes).unwrap();
        assert_eq!(super::H5FD__log_sb_size(&log).unwrap(), log_bytes.len());
        assert_eq!(super::H5FD__log_sb_decode(&log_bytes).unwrap(), log);
        let mut invalid_log_bytes = Vec::new();
        assert!(super::H5FD__log_sb_encode_into(
            &LogFileConfig {
                log_path: Some(PathBuf::from("")),
                flags: 0,
                buffer_size: 0,
            },
            &mut invalid_log_bytes
        )
        .is_err());

        let ros3 = Ros3Config {
            endpoint: Some("s3.us-east-1.amazonaws.com".into()),
            region: Some("us-east-1".into()),
            token: Some("session-token".into()),
        };
        let mut ros3_bytes = Vec::new();
        super::H5FD__ros3_sb_encode_into(&ros3, &mut ros3_bytes).unwrap();
        assert_eq!(super::H5FD__ros3_sb_size(&ros3).unwrap(), ros3_bytes.len());
        assert_eq!(super::H5FD__ros3_sb_decode(&ros3_bytes).unwrap(), ros3);
        let mut invalid_ros3_bytes = Vec::new();
        assert!(super::H5FD__ros3_sb_encode_into(
            &Ros3Config {
                endpoint: Some(String::new()),
                region: None,
                token: None,
            },
            &mut invalid_ros3_bytes,
        )
        .is_err());

        assert_eq!(
            super::H5FD_mirror_xmit_decode_lock(&[]).unwrap(),
            MirrorXmit::Lock
        );
        let mut uint8_bytes = Vec::new();
        super::H5FD__mirror_xmit_encode_uint8_into(0xab, &mut uint8_bytes);
        assert_eq!(uint8_bytes, vec![0xab]);
        let mut open_bytes = Vec::new();
        super::H5FD_mirror_xmit_encode_open_into("mirror.h5", &mut open_bytes).unwrap();
        assert_eq!(
            super::H5FD_mirror_xmit_decode_open_ref(&open_bytes).unwrap(),
            MirrorXmitRef::Open { path: "mirror.h5" }
        );
        let mut reply_bytes = Vec::new();
        super::H5FD_mirror_xmit_encode_reply_into(0, &mut reply_bytes).unwrap();
        assert_eq!(
            super::H5FD_mirror_xmit_decode_reply(&reply_bytes).unwrap(),
            MirrorXmit::Reply { status: 0 }
        );
        let mut set_eoa_bytes = Vec::new();
        super::H5FD_mirror_xmit_encode_set_eoa_into(4096, &mut set_eoa_bytes).unwrap();
        assert_eq!(
            super::H5FD_mirror_xmit_decode_set_eoa(&set_eoa_bytes).unwrap(),
            MirrorXmit::SetEoa { eoa: 4096 }
        );
        let mut write_bytes = Vec::new();
        super::H5FD_mirror_xmit_encode_write_into(8192, b"abc", &mut write_bytes).unwrap();
        assert_eq!(
            super::H5FD_mirror_xmit_decode_write_ref(&write_bytes).unwrap(),
            MirrorXmitRef::Write {
                addr: 8192,
                data: b"abc"
            }
        );

        let onion = OnionHeader {
            version: 1,
            flags: 0x2,
            revision_count: 3,
        };
        let mut onion_bytes = Vec::new();
        super::H5FD__onion_sb_encode_into(&onion, &mut onion_bytes).unwrap();
        assert_eq!(super::H5FD__onion_sb_size(&onion), onion_bytes.len());
        assert_eq!(super::H5FD__onion_sb_decode(&onion_bytes).unwrap(), onion);
        assert_eq!(
            {
                let mut bytes = Vec::new();
                super::H5FD__onion_write_header_into(&onion, &mut bytes).unwrap();
                bytes
            },
            onion_bytes
        );
        assert_eq!(
            {
                let mut bytes = Vec::new();
                super::H5FD__onion_header_encode_into(&onion, &mut bytes).unwrap();
                bytes
            },
            onion_bytes
        );

        let ioc = IocConfig {
            thread_pool_size: 4,
            queue_depth: 16,
        };
        let mut ioc_bytes = Vec::new();
        super::H5FD__ioc_sb_encode_into(&ioc, &mut ioc_bytes).unwrap();
        assert_eq!(super::H5FD__ioc_sb_size(&ioc), ioc_bytes.len());
        assert_eq!(super::H5FD__ioc_sb_decode(&ioc_bytes).unwrap(), ioc);
        let mut invalid_ioc_bytes = Vec::new();
        assert!(super::H5FD__ioc_sb_encode_into(
            &IocConfig {
                thread_pool_size: 0,
                queue_depth: 16,
            },
            &mut invalid_ioc_bytes
        )
        .is_err());

        let subfiling = SubfilingConfig {
            ioc_count: 2,
            stripe_size: 1024,
            stripe_count: 8,
        };
        let mut subfiling_bytes = Vec::new();
        super::H5FD__subfiling_sb_encode_into(&subfiling, &mut subfiling_bytes).unwrap();
        assert_eq!(
            super::H5FD__subfiling_sb_size(&subfiling),
            subfiling_bytes.len()
        );
        assert_eq!(
            super::H5FD__subfiling_sb_decode(&subfiling_bytes).unwrap(),
            subfiling
        );
        let mut invalid_subfiling_bytes = Vec::new();
        assert!(super::H5FD__subfiling_sb_encode_into(
            &SubfilingConfig {
                ioc_count: 0,
                stripe_size: 1024,
                stripe_count: 8,
            },
            &mut invalid_subfiling_bytes
        )
        .is_err());
        assert_eq!(
            super::H5FD__subfiling_new_object_id_checked(41).unwrap(),
            42
        );
        assert!(super::H5FD__subfiling_new_object_id_checked(u64::MAX).is_err());
        assert_eq!(super::H5FD__subfiling_new_object_id(u64::MAX), u64::MAX);

        let history = OnionRevisionIndex {
            records: vec![
                OnionRevisionRecord {
                    revision: 1,
                    address: 64,
                    size: 8,
                },
                OnionRevisionRecord {
                    revision: 2,
                    address: 128,
                    size: 16,
                },
            ],
        };
        let mut history_bytes = Vec::new();
        super::H5FD__onion_history_encode_into(&history, &mut history_bytes).unwrap();
        assert_eq!(
            super::H5FD__onion_ingest_history(&history_bytes).unwrap(),
            history
        );
        assert_eq!(history_bytes.len(), 48);
        let mut record_bytes = Vec::new();
        super::H5FD__onion_revision_record_encode_into(&history.records[0], &mut record_bytes)
            .unwrap();
        assert_eq!(record_bytes.len(), 24);
        assert_eq!(
            super::H5FD__onion_revision_record_decode(&record_bytes).unwrap(),
            history.records[0]
        );
    }

    #[test]
    fn vfd_config_decoders_reject_truncated_or_invalid_payloads() {
        assert!(matches!(
            super::H5FD__family_sb_decode(&[0; 4]).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        assert!(matches!(
            super::H5FD__family_sb_decode(&[0; 9]).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        assert!(matches!(
            super::H5FD_multi_sb_decode(&[1, 0, 0, 0, 99, 0]).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        assert!(matches!(
            super::H5FD_multi_sb_decode(&[2, 0, 0, 0, 0, 0, 0, 1]).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        assert!(matches!(
            super::H5FD__splitter_sb_decode(&[0, 4, 0, 0, 0, b'a']).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        assert!(matches!(
            super::H5FD__splitter_sb_decode(&[2, 0, 0, 0, 0]).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        assert!(matches!(
            super::H5FD__log_sb_decode(&[0; 7]).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        assert!(matches!(
            super::H5FD_mirror_xmit_decode_lock(&[0]).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        assert!(matches!(
            super::H5FD_mirror_xmit_decode_open_ref(&[0xff]).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        assert!(matches!(
            super::H5FD_mirror_xmit_decode_reply(&[0; 3]).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        assert!(matches!(
            super::H5FD_mirror_xmit_decode_set_eoa(&[0; 7]).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        assert!(matches!(
            super::H5FD_mirror_xmit_decode_write_ref(&[0; 7]).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        assert!(matches!(
            super::H5FD__onion_sb_decode(&[0; 9]).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        assert!(matches!(
            super::H5FD__onion_revision_record_decode(&[0; 23]).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        assert!(matches!(
            super::H5FD__onion_ingest_history(&[0; 25]).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        assert!(matches!(
            super::H5FD__ioc_sb_decode(&[0; 15]).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        assert!(matches!(
            super::H5FD__ioc_sb_decode(&[0; 16]).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        assert!(matches!(
            super::H5FD__subfiling_sb_decode(&[0; 15]).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        assert!(matches!(
            super::H5FD__subfiling_sb_decode(&[0; 16]).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        assert!(matches!(
            super::H5FD__hdfs_sb_decode(&[0; 16]).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        let mut hdfs_bad_version = vec![0; super::H5FD__hdfs_sb_size(&HdfsConfig::default())];
        hdfs_bad_version[0..4].copy_from_slice(&2u32.to_le_bytes());
        assert!(matches!(
            super::H5FD__hdfs_sb_decode(&hdfs_bad_version).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        let mut hdfs_bad_port = vec![0; super::H5FD__hdfs_sb_size(&HdfsConfig::default())];
        hdfs_bad_port[0..4].copy_from_slice(&1u32.to_le_bytes());
        hdfs_bad_port[4..7].copy_from_slice(b"nn\0");
        hdfs_bad_port[133..137].copy_from_slice(&65536u32.to_le_bytes());
        hdfs_bad_port[137..142].copy_from_slice(b"user\0");
        assert!(matches!(
            super::H5FD__hdfs_sb_decode(&hdfs_bad_port).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        let mut hdfs_bad_string = vec![0; super::H5FD__hdfs_sb_size(&HdfsConfig::default())];
        hdfs_bad_string[0..4].copy_from_slice(&1u32.to_le_bytes());
        hdfs_bad_string[4..].fill(b'a');
        assert!(matches!(
            super::H5FD__hdfs_sb_decode(&hdfs_bad_string).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        let mut hdfs_bad_trailing = vec![0; super::H5FD__hdfs_sb_size(&HdfsConfig::default())];
        hdfs_bad_trailing[0..4].copy_from_slice(&1u32.to_le_bytes());
        hdfs_bad_trailing[4..7].copy_from_slice(b"nn\0");
        hdfs_bad_trailing[8] = b'x';
        hdfs_bad_trailing[137..142].copy_from_slice(b"user\0");
        assert!(matches!(
            super::H5FD__hdfs_sb_decode(&hdfs_bad_trailing).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        assert!(matches!(
            super::H5FD__ros3_sb_decode(&[1, 0, 0, 0, 0, 0, 0, 0]).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        let mut ros3_truncated = Vec::new();
        ros3_truncated.extend_from_slice(&4u32.to_le_bytes());
        ros3_truncated.extend_from_slice(&0u32.to_le_bytes());
        ros3_truncated.extend_from_slice(&0u32.to_le_bytes());
        ros3_truncated.extend_from_slice(b"s3");
        assert!(matches!(
            super::H5FD__ros3_sb_decode(&ros3_truncated).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        let mut ros3_nul = Vec::new();
        ros3_nul.extend_from_slice(&4u32.to_le_bytes());
        ros3_nul.extend_from_slice(&0u32.to_le_bytes());
        ros3_nul.extend_from_slice(&0u32.to_le_bytes());
        ros3_nul.extend_from_slice(b"s3\0x");
        assert!(matches!(
            super::H5FD__ros3_sb_decode(&ros3_nul).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        assert!(matches!(
            super::H5FD__s3comms_parse_url_ref("s3://bucket/").unwrap_err(),
            Error::InvalidFormat(_)
        ));
    }

    #[test]
    fn hdfs_vfd_config_helpers_round_trip_copy_and_runtime_ops_are_unsupported() {
        let config = HdfsConfig {
            namenode_name: "nn.example.org".into(),
            namenode_port: 8020,
            user_name: "hdf5".into(),
            buffer_size: 65536,
        };
        assert!(super::H5FD__hdfs_validate_config(&config));
        let mut bytes = Vec::new();
        super::H5FD__hdfs_sb_encode_into(&config, &mut bytes).unwrap();
        assert_eq!(bytes.len(), super::H5FD__hdfs_sb_size(&config));
        assert_eq!(super::H5FD__hdfs_sb_decode(&bytes).unwrap(), config);
        assert_eq!(super::H5FD__hdfs_fapl_get(&config), config);
        assert_eq!(super::H5FD__hdfs_fapl_copy(&config), config);
        super::H5FD__hdfs_fapl_free(config.clone());

        let invalid = HdfsConfig {
            namenode_name: String::new(),
            ..config.clone()
        };
        assert!(!super::H5FD__hdfs_validate_config(&invalid));
        assert!(super::H5FD__hdfs_sb_encode_into(&invalid, &mut Vec::new()).is_err());

        let mut buf = [0; 4];
        assert!(matches!(
            super::H5FD__hdfs_open("hdfs://nn.example.org/data.h5", &config).unwrap_err(),
            Error::Unsupported(_)
        ));
        assert!(matches!(
            super::H5FD__hdfs_read(0, &mut buf).unwrap_err(),
            Error::Unsupported(_)
        ));
        assert!(matches!(
            super::H5FD__hdfs_write(0, &buf).unwrap_err(),
            Error::Unsupported(_)
        ));
        assert!(matches!(
            super::H5FD__hdfs_delete("hdfs://nn.example.org/data.h5").unwrap_err(),
            Error::Unsupported(_)
        ));
        assert!(matches!(
            super::H5FD__hdfs_lock().unwrap_err(),
            Error::Unsupported(_)
        ));
        assert!(matches!(
            super::H5FD__hdfs_unlock().unwrap_err(),
            Error::Unsupported(_)
        ));
    }
}
