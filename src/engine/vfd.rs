use std::cmp::Ordering;
use std::collections::HashMap;
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
    pub fn sec2_register() -> FileDriverKind {
        FileDriverKind::Sec2
    }

    pub fn sec2_unregister() {}

    pub fn sec2_open(path: impl AsRef<Path>, read_write: bool) -> Result<Self> {
        Self::open_file(FileDriverKind::Sec2, path, read_write)
    }

    pub fn sec2_close(self) {}

    pub fn sec2_cmp(&self, other: &Self) -> Ordering {
        self.driver_cmp(other)
    }

    pub fn sec2_query(&self) -> u64 {
        self.driver_query()
    }

    pub fn sec2_get_eoa(&self) -> u64 {
        self.eoa
    }

    pub fn sec2_set_eoa(&mut self, eoa: u64) {
        self.eoa = eoa;
    }

    pub fn sec2_get_eof(&mut self) -> Result<u64> {
        self.driver_get_eof()
    }

    pub fn sec2_get_handle(&self) -> Option<&File> {
        self.file.as_ref()
    }

    pub fn sec2_read(&mut self, addr: u64, buf: &mut [u8]) -> Result<()> {
        self.driver_read(addr, buf)
    }

    pub fn sec2_write(&mut self, addr: u64, data: &[u8]) -> Result<()> {
        self.driver_write(addr, data)
    }

    pub fn sec2_truncate(&mut self) -> Result<()> {
        self.driver_truncate()
    }

    pub fn sec2_lock(&mut self) {
        self.locked = true;
    }

    pub fn sec2_unlock(&mut self) {
        self.locked = false;
    }

    pub fn sec2_delete(path: impl AsRef<Path>) -> Result<()> {
        delete_existing(path)
    }

    pub fn sec2_ctl(&mut self, eoa: Option<u64>) {
        if let Some(eoa) = eoa {
            self.eoa = eoa;
        }
    }

    pub fn stdio_register() -> FileDriverKind {
        FileDriverKind::Stdio
    }

    pub fn stdio_unregister() {}

    pub fn stdio_init() -> FileDriverKind {
        FileDriverKind::Stdio
    }

    pub fn stdio_open(path: impl AsRef<Path>, read_write: bool) -> Result<Self> {
        Self::open_file(FileDriverKind::Stdio, path, read_write)
    }

    pub fn stdio_close(self) {}

    pub fn stdio_cmp(&self, other: &Self) -> Ordering {
        self.driver_cmp(other)
    }

    pub fn stdio_query(&self) -> u64 {
        self.driver_query()
    }

    pub fn stdio_alloc(&mut self, size: u64) -> Result<u64> {
        self.driver_alloc(size)
    }

    pub fn stdio_get_eoa(&self) -> u64 {
        self.eoa
    }

    pub fn stdio_set_eoa(&mut self, eoa: u64) {
        self.eoa = eoa;
    }

    pub fn stdio_get_eof(&mut self) -> Result<u64> {
        self.driver_get_eof()
    }

    pub fn stdio_get_handle(&self) -> Option<&File> {
        self.file.as_ref()
    }

    pub fn stdio_read(&mut self, addr: u64, buf: &mut [u8]) -> Result<()> {
        self.driver_read(addr, buf)
    }

    pub fn stdio_write(&mut self, addr: u64, data: &[u8]) -> Result<()> {
        self.driver_write(addr, data)
    }

    pub fn stdio_flush(&mut self) -> Result<()> {
        self.driver_flush()
    }

    pub fn stdio_truncate(&mut self) -> Result<()> {
        self.driver_truncate()
    }

    pub fn stdio_lock(&mut self) {
        self.locked = true;
    }

    pub fn stdio_unlock(&mut self) {
        self.locked = false;
    }

    pub fn stdio_delete(path: impl AsRef<Path>) -> Result<()> {
        delete_existing(path)
    }

    pub fn direct_register() -> FileDriverKind {
        FileDriverKind::Direct
    }

    pub fn direct_unregister() {}

    pub fn direct_populate_config() -> DirectFileConfig {
        DirectFileConfig::default()
    }

    pub fn direct_fapl_get(&self) -> Option<DirectFileConfig> {
        (self.kind == FileDriverKind::Direct).then(DirectFileConfig::default)
    }

    pub fn direct_fapl_copy(config: &DirectFileConfig) -> DirectFileConfig {
        config.clone()
    }

    pub fn direct_open(path: impl AsRef<Path>, read_write: bool) -> Result<Self> {
        Self::open_file(FileDriverKind::Direct, path, read_write)
    }

    pub fn direct_check_alignment_reqs(addr: u64, size: usize, config: &DirectFileConfig) -> bool {
        let Ok(size) = u64::try_from(size) else {
            return false;
        };
        config.memory_alignment != 0
            && config.block_size != 0
            && addr % config.memory_alignment == 0
            && size % config.block_size == 0
    }

    pub fn direct_close(self) {}

    pub fn direct_cmp(&self, other: &Self) -> Ordering {
        self.driver_cmp(other)
    }

    pub fn direct_query(&self) -> u64 {
        self.driver_query()
    }

    pub fn direct_set_eoa(&mut self, eoa: u64) {
        self.eoa = eoa;
    }

    pub fn direct_get_eof(&mut self) -> Result<u64> {
        self.driver_get_eof()
    }

    pub fn direct_get_handle(&self) -> Option<&File> {
        self.file.as_ref()
    }

    pub fn direct_read(&mut self, addr: u64, buf: &mut [u8]) -> Result<()> {
        self.driver_read(addr, buf)
    }

    pub fn direct_write(&mut self, addr: u64, data: &[u8]) -> Result<()> {
        self.driver_write(addr, data)
    }

    pub fn direct_truncate(&mut self) -> Result<()> {
        self.driver_truncate()
    }

    pub fn direct_lock(&mut self) {
        self.locked = true;
    }

    pub fn direct_unlock(&mut self) {
        self.locked = false;
    }

    pub fn direct_delete(path: impl AsRef<Path>) -> Result<()> {
        delete_existing(path)
    }

    pub fn core_get_default_config() -> CoreFileConfig {
        CoreFileConfig::default()
    }

    pub fn core_register() -> FileDriverKind {
        FileDriverKind::Core
    }

    pub fn core_unregister() {}

    pub fn core_fapl_get(&self) -> Option<CoreFileConfig> {
        (self.kind == FileDriverKind::Core).then(CoreFileConfig::default)
    }

    pub fn core_cmp(&self, other: &Self) -> Ordering {
        self.driver_cmp(other)
    }

    pub fn core_query(&self) -> u64 {
        self.driver_query()
    }

    pub fn core_get_eoa(&self) -> u64 {
        self.eoa
    }

    pub fn core_set_eoa(&mut self, eoa: u64) {
        self.eoa = eoa;
        if let Ok(eoa) = usize::try_from(eoa) {
            if self.core_image.len() < eoa {
                self.core_image.resize(eoa, 0);
            }
        }
    }

    pub fn core_get_eof(&self) -> u64 {
        u64::try_from(self.core_image.len()).unwrap_or(u64::MAX)
    }

    pub fn core_get_handle(&self) -> Option<&[u8]> {
        (self.kind == FileDriverKind::Core).then_some(self.core_image.as_slice())
    }

    pub fn core_read(&mut self, addr: u64, buf: &mut [u8]) -> Result<()> {
        self.driver_read(addr, buf)
    }

    pub fn core_write(&mut self, addr: u64, data: &[u8]) -> Result<()> {
        self.driver_write(addr, data)
    }

    pub fn core_flush(&mut self) -> Result<()> {
        self.driver_flush()
    }

    pub fn core_truncate(&mut self) -> Result<()> {
        self.driver_truncate()
    }

    pub fn core_lock(&mut self) {
        self.locked = true;
    }

    pub fn core_unlock(&mut self) {
        self.locked = false;
    }

    pub fn core_delete(&mut self) {
        self.core_image.clear();
        self.eoa = 0;
    }

    pub fn core_add_dirty_region(&mut self, addr: u64, size: u64) -> Result<()> {
        let end = addr
            .checked_add(size)
            .ok_or_else(|| Error::InvalidFormat("core VFD dirty region overflow".into()))?;
        if end > self.eoa {
            self.core_set_eoa(end);
        }
        Ok(())
    }

    pub fn core_destroy_dirty_list(&mut self) {}

    pub fn core_write_to_bstore(&mut self, addr: u64, data: &[u8]) -> Result<()> {
        self.core_write(addr, data)
    }

    pub fn alloc(&mut self, size: u64) -> Result<u64> {
        self.driver_alloc(size)
    }

    pub fn free(&mut self, _addr: u64, _size: u64) {}

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

    fn driver_cmp(&self, other: &Self) -> Ordering {
        self.kind
            .cmp(&other.kind)
            .then_with(|| self.path.cmp(&other.path))
    }

    fn driver_query(&self) -> u64 {
        match self.kind {
            FileDriverKind::Sec2 | FileDriverKind::Stdio | FileDriverKind::Direct => 0x01,
            FileDriverKind::Core => 0x03,
        }
    }

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

    fn driver_flush(&mut self) -> Result<()> {
        if let Some(file) = self.file.as_mut() {
            file.flush()?;
        }
        Ok(())
    }

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
    fn cmp(&self, other: &Self) -> Ordering {
        (*self as u8).cmp(&(*other as u8))
    }
}

impl PartialOrd for FileDriverKind {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn delete_existing(path: impl AsRef<Path>) -> Result<()> {
    match fs::remove_file(path.as_ref()) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err.into()),
    }
}

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

#[allow(non_snake_case)]
pub fn H5FD_init() -> VfdRegistry {
    H5FD__init_package()
}

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

#[allow(non_snake_case)]
pub fn H5FD_term_package(registry: &mut VfdRegistry) {
    registry.by_name.clear();
    registry.by_value.clear();
}

#[allow(non_snake_case)]
pub fn H5FD__free_cls(registry: &mut VfdRegistry, name: &str) -> Option<u64> {
    H5FDunregister(registry, name)
}

#[allow(non_snake_case)]
pub fn H5FDregister(registry: &mut VfdRegistry, name: &str, value: u64) -> u64 {
    H5FD_register_driver_by_name(registry, name, value)
}

#[allow(non_snake_case)]
pub fn H5FD_register(registry: &mut VfdRegistry, name: &str, value: u64) -> u64 {
    H5FDregister(registry, name, value)
}

#[allow(non_snake_case)]
pub fn H5FDunregister(registry: &mut VfdRegistry, name: &str) -> Option<u64> {
    let value = registry.by_name.remove(name)?;
    registry.by_value.remove(&value);
    Some(value)
}

#[allow(non_snake_case)]
pub fn H5FD_register_driver_by_name(registry: &mut VfdRegistry, name: &str, value: u64) -> u64 {
    registry.by_name.insert(name.to_string(), value);
    registry.by_value.insert(value, name.to_string());
    value
}

#[allow(non_snake_case)]
pub fn H5FD_register_driver_by_value(registry: &mut VfdRegistry, value: u64, name: &str) -> u64 {
    H5FD_register_driver_by_name(registry, name, value)
}

#[allow(non_snake_case)]
pub fn H5FD_is_driver_registered_by_name(registry: &VfdRegistry, name: &str) -> bool {
    registry.by_name.contains_key(name)
}

#[allow(non_snake_case)]
pub fn H5FD_get_driver_id_by_name(registry: &VfdRegistry, name: &str) -> Option<u64> {
    registry.by_name.get(name).copied()
}

#[allow(non_snake_case)]
pub fn H5FD_get_driver_id_by_value(registry: &VfdRegistry, value: u64) -> Option<&str> {
    registry.by_value.get(&value).map(String::as_str)
}

#[allow(non_snake_case)]
pub fn H5FD_get_class(registry: &VfdRegistry, value: u64) -> Option<&str> {
    H5FD_get_driver_id_by_value(registry, value)
}

#[allow(non_snake_case)]
pub fn H5FD__get_driver_cb(registry: &VfdRegistry, name: &str) -> Option<u64> {
    H5FD_get_driver_id_by_name(registry, name)
}

#[allow(non_snake_case)]
pub fn H5FD_sb_size(_driver: FileDriverKind) -> usize {
    0
}

#[allow(non_snake_case)]
pub fn H5FD_fapl_get(driver: &LocalFileDriver) -> FileDriverKind {
    driver.kind
}

#[allow(non_snake_case)]
pub fn H5FDopen(
    path: impl AsRef<Path>,
    kind: FileDriverKind,
    read_write: bool,
) -> Result<LocalFileDriver> {
    H5FD_open(path, kind, read_write)
}

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

#[allow(non_snake_case)]
pub fn H5FDclose(driver: LocalFileDriver) {
    H5FD_close(driver);
}

#[allow(non_snake_case)]
pub fn H5FD_close(_driver: LocalFileDriver) {}

#[allow(non_snake_case)]
pub fn H5FDcmp(left: &LocalFileDriver, right: &LocalFileDriver) -> Ordering {
    H5FD_cmp(left, right)
}

#[allow(non_snake_case)]
pub fn H5FD_cmp(left: &LocalFileDriver, right: &LocalFileDriver) -> Ordering {
    left.driver_cmp(right)
}

#[allow(non_snake_case)]
pub fn H5FDquery(driver: &LocalFileDriver) -> u64 {
    H5FD_get_feature_flags(driver)
}

#[allow(non_snake_case)]
pub fn H5FDalloc(driver: &mut LocalFileDriver, size: u64) -> Result<u64> {
    driver.alloc(size)
}

#[allow(non_snake_case)]
pub fn H5FDfree(driver: &mut LocalFileDriver, addr: u64, size: u64) {
    driver.free(addr, size);
}

#[allow(non_snake_case)]
pub fn H5FDget_eoa(driver: &LocalFileDriver) -> u64 {
    driver.eoa
}

#[allow(non_snake_case)]
pub fn H5FDset_eoa(driver: &mut LocalFileDriver, eoa: u64) {
    driver.eoa = eoa;
}

#[allow(non_snake_case)]
pub fn H5FDget_eof(driver: &mut LocalFileDriver) -> Result<u64> {
    driver.driver_get_eof()
}

#[allow(non_snake_case)]
pub fn H5FD_get_maxaddr() -> u64 {
    u64::MAX
}

#[allow(non_snake_case)]
pub fn H5FD_get_feature_flags(driver: &LocalFileDriver) -> u64 {
    driver.driver_query()
}

#[allow(non_snake_case)]
pub fn H5FD_set_feature_flags(_driver: &mut LocalFileDriver, _flags: u64) {}

#[allow(non_snake_case)]
pub fn H5FDread(driver: &mut LocalFileDriver, addr: u64, buf: &mut [u8]) -> Result<()> {
    driver.driver_read(addr, buf)
}

#[allow(non_snake_case)]
pub fn H5FDwrite(driver: &mut LocalFileDriver, addr: u64, data: &[u8]) -> Result<()> {
    driver.driver_write(addr, data)
}

#[allow(non_snake_case)]
pub fn H5FDread_vector(driver: &mut LocalFileDriver, requests: &mut [VfdIoRequest]) -> Result<()> {
    H5FD_read_vector_from_selection(driver, requests)
}

#[allow(non_snake_case)]
pub fn H5FDwrite_vector(driver: &mut LocalFileDriver, requests: &[VfdIoRequest]) -> Result<()> {
    H5FD_write_vector_from_selection(driver, requests)
}

#[allow(non_snake_case)]
pub fn H5FDread_selection(
    driver: &mut LocalFileDriver,
    requests: &mut [VfdIoRequest],
) -> Result<()> {
    H5FD_read_from_selection(driver, requests)
}

#[allow(non_snake_case)]
pub fn H5FDwrite_selection(driver: &mut LocalFileDriver, requests: &[VfdIoRequest]) -> Result<()> {
    H5FD_write_selection(driver, requests)
}

#[allow(non_snake_case)]
pub fn H5FDflush(driver: &mut LocalFileDriver) -> Result<()> {
    driver.driver_flush()
}

#[allow(non_snake_case)]
pub fn H5FD_get_fileno(driver: &LocalFileDriver) -> Option<&Path> {
    driver.path.as_deref()
}

#[allow(non_snake_case)]
pub fn H5FDget_vfd_handle(driver: &LocalFileDriver) -> Option<&File> {
    driver.file.as_ref()
}

#[allow(non_snake_case)]
pub fn H5FD_set_paged_aggr(_driver: &mut LocalFileDriver, _enabled: bool) {}

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

#[allow(non_snake_case)]
pub fn H5FD__read_selection_translate(requests: &[VfdIoRequest]) -> Vec<VfdIoRequest> {
    requests.to_vec()
}

#[allow(non_snake_case)]
pub fn H5FD__write_selection_translate(requests: &[VfdIoRequest]) -> Vec<VfdIoRequest> {
    requests.to_vec()
}

#[allow(non_snake_case)]
pub fn H5FD_write_selection(driver: &mut LocalFileDriver, requests: &[VfdIoRequest]) -> Result<()> {
    for request in requests {
        H5FDwrite(driver, request.addr, &request.bytes)?;
    }
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5FD_write_selection_id(
    driver: &mut LocalFileDriver,
    _selection_id: u64,
    requests: &[VfdIoRequest],
) -> Result<()> {
    H5FD_write_selection(driver, requests)
}

#[allow(non_snake_case)]
pub fn H5FD_read_vector_from_selection(
    driver: &mut LocalFileDriver,
    requests: &mut [VfdIoRequest],
) -> Result<()> {
    for request in requests {
        H5FDread(driver, request.addr, &mut request.bytes)?;
    }
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5FD_write_vector_from_selection(
    driver: &mut LocalFileDriver,
    requests: &[VfdIoRequest],
) -> Result<()> {
    H5FD_write_selection(driver, requests)
}

#[allow(non_snake_case)]
pub fn H5FD_read_from_selection(
    driver: &mut LocalFileDriver,
    requests: &mut [VfdIoRequest],
) -> Result<()> {
    H5FD_read_vector_from_selection(driver, requests)
}

#[allow(non_snake_case)]
pub fn H5FD_write_from_selection(
    driver: &mut LocalFileDriver,
    requests: &[VfdIoRequest],
) -> Result<()> {
    H5FD_write_selection(driver, requests)
}

#[allow(non_snake_case)]
pub fn H5FDread_vector_from_selection(
    driver: &mut LocalFileDriver,
    requests: &mut [VfdIoRequest],
) -> Result<()> {
    H5FD_read_vector_from_selection(driver, requests)
}

#[allow(non_snake_case)]
pub fn H5FDwrite_vector_from_selection(
    driver: &mut LocalFileDriver,
    requests: &[VfdIoRequest],
) -> Result<()> {
    H5FD_write_vector_from_selection(driver, requests)
}

#[allow(non_snake_case)]
pub fn H5FDread_from_selection(
    driver: &mut LocalFileDriver,
    requests: &mut [VfdIoRequest],
) -> Result<()> {
    H5FD_read_from_selection(driver, requests)
}

#[allow(non_snake_case)]
pub fn H5FDwrite_from_selection(
    driver: &mut LocalFileDriver,
    requests: &[VfdIoRequest],
) -> Result<()> {
    H5FD_write_from_selection(driver, requests)
}

#[allow(non_snake_case)]
pub fn H5FD__srt_tmp_cmp(left: &VfdIoRequest, right: &VfdIoRequest) -> Ordering {
    left.addr
        .cmp(&right.addr)
        .then_with(|| left.bytes.len().cmp(&right.bytes.len()))
}

#[allow(non_snake_case)]
pub fn H5FD__sort_io_req_real(requests: &mut [VfdIoRequest]) {
    requests.sort_by(H5FD__srt_tmp_cmp);
}

#[allow(non_snake_case)]
pub fn H5FD_sort_vector_io_req(requests: &mut [VfdIoRequest]) {
    H5FD__sort_io_req_real(requests);
}

#[allow(non_snake_case)]
pub fn H5FD_sort_selection_io_req(requests: &mut [VfdIoRequest]) {
    H5FD__sort_io_req_real(requests);
}

#[allow(non_snake_case)]
pub fn H5FD_mpi_get_rank() -> Result<u32> {
    Err(Error::Unsupported(
        "MPI VFD is intentionally unsupported; use rayon for local parallelism".into(),
    ))
}

#[allow(non_snake_case)]
pub fn H5FD_mpi_get_comm() -> Result<()> {
    Err(Error::Unsupported(
        "MPI communicator is unavailable in pure-Rust non-MPI mode".into(),
    ))
}

#[allow(non_snake_case)]
pub fn H5FD_mpi_get_info() -> Result<()> {
    Err(Error::Unsupported(
        "MPI info is unavailable in pure-Rust non-MPI mode".into(),
    ))
}

#[allow(non_snake_case)]
pub fn H5FD_mpi_MPIOff_to_haddr(offset: i64) -> Option<u64> {
    u64::try_from(offset).ok()
}

#[allow(non_snake_case)]
pub fn H5FD_mpi_haddr_to_MPIOff(addr: u64) -> Option<i64> {
    i64::try_from(addr).ok()
}

#[allow(non_snake_case)]
pub fn H5FD_mpi_get_file_sync_required() -> bool {
    false
}

#[allow(non_snake_case)]
pub fn H5FD_mpio_wait_for_left_neighbor() -> Result<()> {
    Err(Error::Unsupported(
        "MPI synchronization is intentionally unsupported".into(),
    ))
}

#[allow(non_snake_case)]
pub fn H5FD_mpio_signal_right_neighbor() -> Result<()> {
    Err(Error::Unsupported(
        "MPI synchronization is intentionally unsupported".into(),
    ))
}

#[allow(non_snake_case)]
pub fn H5FD__mpio_register() -> Result<()> {
    Err(unsupported_vfd_driver("MPIO"))
}

#[allow(non_snake_case)]
pub fn H5FD__mpio_unregister() {}

#[allow(non_snake_case)]
pub fn H5FD__mpio_init() -> Result<()> {
    Err(unsupported_vfd_driver("MPIO"))
}

#[allow(non_snake_case)]
pub fn H5FD__mpio_term() {}

#[allow(non_snake_case)]
pub fn H5FD_set_mpio_atomicity(_atomicity: bool) -> Result<()> {
    Err(unsupported_vfd_driver("MPIO atomicity"))
}

#[allow(non_snake_case)]
pub fn H5FD_get_mpio_atomicity() -> Result<bool> {
    Err(unsupported_vfd_driver("MPIO atomicity"))
}

#[allow(non_snake_case)]
pub fn H5FD__mpio_open(_path: &str) -> Result<()> {
    Err(unsupported_vfd_driver("MPIO"))
}

#[allow(non_snake_case)]
pub fn H5FD__mpio_close() {}

#[allow(non_snake_case)]
pub fn H5FD__mpio_query() -> u64 {
    0
}

#[allow(non_snake_case)]
pub fn H5FD__mpio_get_eoa() -> Result<u64> {
    Err(unsupported_vfd_driver("MPIO"))
}

#[allow(non_snake_case)]
pub fn H5FD__mpio_set_eoa(_eoa: u64) -> Result<()> {
    Err(unsupported_vfd_driver("MPIO"))
}

#[allow(non_snake_case)]
pub fn H5FD__mpio_get_eof() -> Result<u64> {
    Err(unsupported_vfd_driver("MPIO"))
}

#[allow(non_snake_case)]
pub fn H5FD__mpio_get_handle() -> Result<()> {
    Err(unsupported_vfd_driver("MPIO"))
}

#[allow(non_snake_case)]
pub fn H5FD__mpio_read(_addr: u64, _buf: &mut [u8]) -> Result<()> {
    Err(unsupported_vfd_driver("MPIO"))
}

#[allow(non_snake_case)]
pub fn H5FD__mpio_write(_addr: u64, _data: &[u8]) -> Result<()> {
    Err(unsupported_vfd_driver("MPIO"))
}

#[allow(non_snake_case)]
pub fn H5FD__mpio_vector_build_types(requests: &[VfdIoRequest]) -> Vec<VfdIoRequest> {
    requests.to_vec()
}

#[allow(non_snake_case)]
pub fn H5FD__selection_build_types(requests: &[VfdIoRequest]) -> Vec<VfdIoRequest> {
    requests.to_vec()
}

#[allow(non_snake_case)]
pub fn H5FD__mpio_read_vector(_requests: &mut [VfdIoRequest]) -> Result<()> {
    Err(unsupported_vfd_driver("MPIO"))
}

#[allow(non_snake_case)]
pub fn H5FD__mpio_write_vector(_requests: &[VfdIoRequest]) -> Result<()> {
    Err(unsupported_vfd_driver("MPIO"))
}

#[allow(non_snake_case)]
pub fn H5FD__mpio_read_selection(_requests: &mut [VfdIoRequest]) -> Result<()> {
    Err(unsupported_vfd_driver("MPIO"))
}

#[allow(non_snake_case)]
pub fn H5FD__mpio_write_selection(_requests: &[VfdIoRequest]) -> Result<()> {
    Err(unsupported_vfd_driver("MPIO"))
}

#[allow(non_snake_case)]
pub fn H5FD__mpio_flush() -> Result<()> {
    Err(unsupported_vfd_driver("MPIO"))
}

#[allow(non_snake_case)]
pub fn H5FD__mpio_truncate() -> Result<()> {
    Err(unsupported_vfd_driver("MPIO"))
}

#[allow(non_snake_case)]
pub fn H5FD__mpio_delete(_path: &str) -> Result<()> {
    Err(unsupported_vfd_driver("MPIO"))
}

#[allow(non_snake_case)]
pub fn H5FD__mpio_ctl(_opcode: u64) -> Result<()> {
    Err(unsupported_vfd_driver("MPIO"))
}

fn unsupported_vfd_driver(driver: &str) -> Error {
    Error::Unsupported(format!(
        "{driver} VFD is not implemented in pure-Rust local-only mode"
    ))
}

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

fn read_le_u32_len_at(data: &[u8], offset: usize, context: &'static str) -> Result<usize> {
    usize::try_from(read_le_u32_at(data, offset, context)?)
        .map_err(|_| Error::InvalidFormat(format!("{context} does not fit in usize")))
}

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

#[allow(non_snake_case)]
pub fn H5FD__hdfs_register() -> Result<()> {
    Err(unsupported_vfd_driver("HDFS"))
}

#[allow(non_snake_case)]
pub fn H5FD__hdfs_unregister() {}

#[allow(non_snake_case)]
pub fn H5FD__hdfs_init() -> Result<()> {
    Err(unsupported_vfd_driver("HDFS"))
}

#[allow(non_snake_case)]
pub fn H5FD__hdfs_handle_open(_path: &str) -> Result<()> {
    Err(unsupported_vfd_driver("HDFS"))
}

#[allow(non_snake_case)]
pub fn H5FD__hdfs_handle_close() {}

#[allow(non_snake_case)]
pub fn H5FD__hdfs_fapl_get() -> Result<()> {
    Err(unsupported_vfd_driver("HDFS"))
}

#[allow(non_snake_case)]
pub fn H5FD__hdfs_fapl_copy() -> Result<()> {
    Err(unsupported_vfd_driver("HDFS"))
}

#[allow(non_snake_case)]
pub fn H5FD__hdfs_fapl_free() {}

#[allow(non_snake_case)]
pub fn H5FD__hdfs_open(_path: &str) -> Result<()> {
    Err(unsupported_vfd_driver("HDFS"))
}

#[allow(non_snake_case)]
pub fn H5FD__hdfs_close() {}

#[allow(non_snake_case)]
pub fn H5FD__hdfs_cmp() -> Ordering {
    Ordering::Equal
}

#[allow(non_snake_case)]
pub fn H5FD__hdfs_query() -> u64 {
    0
}

#[allow(non_snake_case)]
pub fn H5FD__hdfs_get_eoa() -> Result<u64> {
    Err(unsupported_vfd_driver("HDFS"))
}

#[allow(non_snake_case)]
pub fn H5FD__hdfs_set_eoa(_eoa: u64) -> Result<()> {
    Err(unsupported_vfd_driver("HDFS"))
}

#[allow(non_snake_case)]
pub fn H5FD__hdfs_get_eof() -> Result<u64> {
    Err(unsupported_vfd_driver("HDFS"))
}

#[allow(non_snake_case)]
pub fn H5FD__hdfs_get_handle() -> Result<()> {
    Err(unsupported_vfd_driver("HDFS"))
}

#[allow(non_snake_case)]
pub fn H5FD__hdfs_truncate() -> Result<()> {
    Err(unsupported_vfd_driver("HDFS"))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S3ParsedUrl {
    pub scheme: String,
    pub bucket: String,
    pub key: String,
}

#[allow(non_snake_case)]
pub fn H5FD__s3comms_init() -> Result<()> {
    Err(unsupported_vfd_driver("S3/ROS3"))
}

#[allow(non_snake_case)]
pub fn H5FD__s3comms_term_func() {}

#[allow(non_snake_case)]
pub fn H5FD__s3comms_term() {}

#[allow(non_snake_case)]
pub fn H5FD__s3comms_s3r_req_finish_cb() -> Result<()> {
    Err(unsupported_vfd_driver("S3/ROS3"))
}

#[allow(non_snake_case)]
pub fn H5FD__s3comms_s3r_req_finish_pred(done: bool) -> bool {
    done
}

#[allow(non_snake_case)]
pub fn H5FD__s3comms_cred_provider_get_creds_cb() -> Result<()> {
    Err(unsupported_vfd_driver("S3/ROS3 credentials"))
}

#[allow(non_snake_case)]
pub fn H5FD__s3comms_cred_provider_pred(has_credentials: bool) -> bool {
    has_credentials
}

#[allow(non_snake_case)]
pub fn H5FD__s3comms_s3r_open(_url: &str) -> Result<()> {
    Err(unsupported_vfd_driver("S3/ROS3"))
}

#[allow(non_snake_case)]
pub fn H5FD__s3comms_s3r_close() {}

#[allow(non_snake_case)]
pub fn H5FD__s3comms_s3r_get_filesize(_url: &str) -> Result<u64> {
    Err(unsupported_vfd_driver("S3/ROS3"))
}

#[allow(non_snake_case)]
pub fn H5FD__s3comms_s3r_getsize(_url: &str) -> Result<u64> {
    Err(unsupported_vfd_driver("S3/ROS3"))
}

#[allow(non_snake_case)]
pub fn H5FD__s3comms_s3r_getsize_headers_cb(_headers: &[(&str, &str)]) -> Option<u64> {
    None
}

#[allow(non_snake_case)]
pub fn H5FD__s3comms_parse_url(url: &str) -> Result<S3ParsedUrl> {
    let (scheme, rest) = url
        .split_once("://")
        .ok_or_else(|| Error::InvalidFormat("S3 URL missing scheme".into()))?;
    let (bucket, key) = rest
        .split_once('/')
        .ok_or_else(|| Error::InvalidFormat("S3 URL missing object key".into()))?;
    Ok(S3ParsedUrl {
        scheme: scheme.to_string(),
        bucket: bucket.to_string(),
        key: key.to_string(),
    })
}

#[allow(non_snake_case)]
pub fn H5FD__s3comms_free_purl(_url: S3ParsedUrl) {}

#[allow(non_snake_case)]
pub fn H5FD__s3comms_get_aws_region(endpoint: &str) -> Option<String> {
    endpoint
        .split('.')
        .find(|part| part.starts_with("us-") || part.starts_with("eu-") || part.starts_with("ap-"))
        .map(str::to_string)
}

#[allow(non_snake_case)]
pub fn H5FD__s3comms_get_credentials_provider() -> Result<()> {
    Err(unsupported_vfd_driver("S3/ROS3 credentials"))
}

#[allow(non_snake_case)]
pub fn H5FD__s3comms_format_user_agent_header(product: &str, version: &str) -> String {
    format!("{product}/{version}")
}

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

#[allow(non_snake_case)]
pub fn H5FD__mirror_register() -> Result<()> {
    Err(unsupported_vfd_driver("mirror"))
}

#[allow(non_snake_case)]
pub fn H5FD__mirror_unregister() {}

#[allow(non_snake_case)]
pub fn H5FD__mirror_xmit_decode_uint64(bytes: &[u8]) -> Result<u64> {
    if bytes.len() != 8 {
        return Err(Error::InvalidFormat(
            "mirror transmit uint64 payload has invalid length".into(),
        ));
    }
    read_le_u64_at(bytes, 0, "mirror transmit uint64")
}

#[allow(non_snake_case)]
pub fn H5FD_mirror_xmit_decode_lock(bytes: &[u8]) -> Result<MirrorXmit> {
    if !bytes.is_empty() {
        return Err(Error::InvalidFormat(
            "mirror transmit lock payload has invalid length".into(),
        ));
    }
    Ok(MirrorXmit::Lock)
}

#[allow(non_snake_case)]
pub fn H5FD_mirror_xmit_decode_open(bytes: &[u8]) -> Result<MirrorXmit> {
    let path = std::str::from_utf8(bytes)
        .map_err(|_| Error::InvalidFormat("mirror transmit open path is not UTF-8".into()))?;
    Ok(MirrorXmit::Open {
        path: path.to_string(),
    })
}

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

#[allow(non_snake_case)]
pub fn H5FD_mirror_xmit_decode_set_eoa(bytes: &[u8]) -> Result<MirrorXmit> {
    H5FD__mirror_xmit_decode_uint64(bytes).map(|eoa| MirrorXmit::SetEoa { eoa })
}

#[allow(non_snake_case)]
pub fn H5FD_mirror_xmit_decode_write(bytes: &[u8]) -> Result<MirrorXmit> {
    if bytes.len() < 8 {
        return Err(Error::InvalidFormat(
            "mirror transmit write payload is truncated".into(),
        ));
    }
    let addr = read_le_u64_at(bytes, 0, "mirror transmit write address")?;
    Ok(MirrorXmit::Write {
        addr,
        data: bytes[8..].to_vec(),
    })
}

#[allow(non_snake_case)]
pub fn H5FD_mirror_xmit_encode_open(path: &str) -> Vec<u8> {
    path.as_bytes().to_vec()
}

#[allow(non_snake_case)]
pub fn H5FD_mirror_xmit_encode_reply(status: i32) -> Vec<u8> {
    status.to_le_bytes().to_vec()
}

#[allow(non_snake_case)]
pub fn H5FD_mirror_xmit_encode_set_eoa(eoa: u64) -> Vec<u8> {
    eoa.to_le_bytes().to_vec()
}

#[allow(non_snake_case)]
pub fn H5FD_mirror_xmit_encode_write(addr: u64, data: &[u8]) -> Vec<u8> {
    let mut out = addr.to_le_bytes().to_vec();
    out.extend_from_slice(data);
    out
}

#[allow(non_snake_case)]
pub fn H5FD_mirror_xmit_is_close(message: &MirrorXmit) -> bool {
    matches!(message, MirrorXmit::Close)
}

#[allow(non_snake_case)]
pub fn H5FD_mirror_xmit_is_lock(message: &MirrorXmit) -> bool {
    matches!(message, MirrorXmit::Lock)
}

#[allow(non_snake_case)]
pub fn H5FD_mirror_xmit_is_set_eoa(message: &MirrorXmit) -> bool {
    matches!(message, MirrorXmit::SetEoa { .. })
}

#[allow(non_snake_case)]
pub fn H5FD_mirror_xmit_is_reply(message: &MirrorXmit) -> bool {
    matches!(message, MirrorXmit::Reply { .. })
}

#[allow(non_snake_case)]
pub fn H5FD_mirror_xmit_is_write(message: &MirrorXmit) -> bool {
    matches!(message, MirrorXmit::Write { .. })
}

#[allow(non_snake_case)]
pub fn H5FD_mirror_xmit_is_xmit(_message: &MirrorXmit) -> bool {
    true
}

#[allow(non_snake_case)]
pub fn H5FD__mirror_verify_reply(message: &MirrorXmit) -> bool {
    matches!(message, MirrorXmit::Reply { status: 0 })
}

#[allow(non_snake_case)]
pub fn H5FD__mirror_fapl_get() -> Result<()> {
    Err(unsupported_vfd_driver("mirror"))
}

#[allow(non_snake_case)]
pub fn H5FD__mirror_fapl_copy() -> Result<()> {
    Err(unsupported_vfd_driver("mirror"))
}

#[allow(non_snake_case)]
pub fn H5FD__mirror_fapl_free() {}

#[allow(non_snake_case)]
pub fn H5FD__mirror_open(_path: &str) -> Result<()> {
    Err(unsupported_vfd_driver("mirror"))
}

#[allow(non_snake_case)]
pub fn H5FD__mirror_close() {}

#[allow(non_snake_case)]
pub fn H5FD__mirror_query() -> u64 {
    0
}

#[allow(non_snake_case)]
pub fn H5FD__mirror_get_eoa() -> Result<u64> {
    Err(unsupported_vfd_driver("mirror"))
}

#[allow(non_snake_case)]
pub fn H5FD__mirror_set_eoa(_eoa: u64) -> Result<()> {
    Err(unsupported_vfd_driver("mirror"))
}

#[allow(non_snake_case)]
pub fn H5FD__mirror_get_eof() -> Result<u64> {
    Err(unsupported_vfd_driver("mirror"))
}

#[allow(non_snake_case)]
pub fn H5FD__mirror_read(_addr: u64, _buf: &mut [u8]) -> Result<()> {
    Err(unsupported_vfd_driver("mirror"))
}

#[allow(non_snake_case)]
pub fn H5FD__mirror_write(_addr: u64, _data: &[u8]) -> Result<()> {
    Err(unsupported_vfd_driver("mirror"))
}

#[allow(non_snake_case)]
pub fn H5FD__mirror_truncate() -> Result<()> {
    Err(unsupported_vfd_driver("mirror"))
}

#[allow(non_snake_case)]
pub fn H5FD__mirror_lock() -> Result<()> {
    Err(unsupported_vfd_driver("mirror"))
}

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
            member_size: 2 * 1024 * 1024 * 1024,
            printf_filename: "%05d.h5".into(),
        }
    }
}

#[allow(non_snake_case)]
pub fn H5FD__family_get_default_config() -> FamilyFileConfig {
    FamilyFileConfig::default()
}

#[allow(non_snake_case)]
pub fn H5FD__family_get_default_printf_filename() -> &'static str {
    "%05d.h5"
}

#[allow(non_snake_case)]
pub fn H5FD__family_register() -> Result<()> {
    Err(unsupported_vfd_driver("family"))
}

#[allow(non_snake_case)]
pub fn H5FD__family_unregister() {}

#[allow(non_snake_case)]
pub fn H5FD__family_fapl_get(config: &FamilyFileConfig) -> FamilyFileConfig {
    config.clone()
}

#[allow(non_snake_case)]
pub fn H5FD__family_fapl_copy(config: &FamilyFileConfig) -> FamilyFileConfig {
    config.clone()
}

#[allow(non_snake_case)]
pub fn H5FD__family_fapl_free(_config: FamilyFileConfig) {}

#[allow(non_snake_case)]
pub fn H5FD__family_validate_config(config: &FamilyFileConfig) -> bool {
    config.member_size != 0 && !config.printf_filename.is_empty()
}

#[allow(non_snake_case)]
pub fn H5FD__family_sb_size(config: &FamilyFileConfig) -> Result<usize> {
    12usize
        .checked_add(config.printf_filename.len())
        .ok_or_else(|| Error::InvalidFormat("family VFD config image length overflow".into()))
}

#[allow(non_snake_case)]
pub fn H5FD__family_sb_encode(config: &FamilyFileConfig) -> Result<Vec<u8>> {
    if !H5FD__family_validate_config(config) {
        return Err(Error::InvalidFormat("invalid family VFD config".into()));
    }
    let filename = config.printf_filename.as_bytes();
    let filename_len = u32::try_from(filename.len())
        .map_err(|_| Error::InvalidFormat("family VFD filename length exceeds u32".into()))?;
    let mut out = Vec::with_capacity(H5FD__family_sb_size(config)?);
    out.extend_from_slice(&config.member_size.to_le_bytes());
    out.extend_from_slice(&filename_len.to_le_bytes());
    out.extend_from_slice(filename);
    Ok(out)
}

#[allow(non_snake_case)]
pub fn H5FD__family_sb_decode(bytes: &[u8]) -> Result<FamilyFileConfig> {
    let member_size = read_le_u64_at(bytes, 0, "family VFD member size")?;
    let filename_len = read_le_u32_len_at(bytes, 8, "family VFD filename length")?;
    let filename_start = 12usize;
    let filename_end = filename_start
        .checked_add(filename_len)
        .ok_or_else(|| Error::InvalidFormat("family VFD filename length overflow".into()))?;
    let filename_bytes = bytes
        .get(filename_start..filename_end)
        .ok_or_else(|| Error::InvalidFormat("family VFD filename is truncated".into()))?;
    if bytes.len() != filename_end {
        return Err(Error::InvalidFormat(
            "family VFD config has trailing bytes".into(),
        ));
    }
    let filename = std::str::from_utf8(filename_bytes)
        .map_err(|_| Error::InvalidFormat("family VFD filename is not UTF-8".into()))?
        .to_string();
    let config = FamilyFileConfig {
        member_size,
        printf_filename: filename,
    };
    if !H5FD__family_validate_config(&config) {
        return Err(Error::InvalidFormat("invalid family VFD config".into()));
    }
    Ok(config)
}

#[allow(non_snake_case)]
pub fn H5FD__family_open(_pattern: &str, _config: &FamilyFileConfig) -> Result<()> {
    Err(unsupported_vfd_driver("family"))
}

#[allow(non_snake_case)]
pub fn H5FD__family_close() {}

#[allow(non_snake_case)]
pub fn H5FD__family_cmp(left: &FamilyFileConfig, right: &FamilyFileConfig) -> Ordering {
    left.member_size
        .cmp(&right.member_size)
        .then_with(|| left.printf_filename.cmp(&right.printf_filename))
}

#[allow(non_snake_case)]
pub fn H5FD__family_query() -> u64 {
    0
}

#[allow(non_snake_case)]
pub fn H5FD__family_get_eoa() -> Result<u64> {
    Err(unsupported_vfd_driver("family"))
}

#[allow(non_snake_case)]
pub fn H5FD__family_set_eoa(_eoa: u64) -> Result<()> {
    Err(unsupported_vfd_driver("family"))
}

#[allow(non_snake_case)]
pub fn H5FD__family_read(_addr: u64, _buf: &mut [u8]) -> Result<()> {
    Err(unsupported_vfd_driver("family"))
}

#[allow(non_snake_case)]
pub fn H5FD__family_flush() -> Result<()> {
    Err(unsupported_vfd_driver("family"))
}

#[allow(non_snake_case)]
pub fn H5FD__family_truncate() -> Result<()> {
    Err(unsupported_vfd_driver("family"))
}

#[allow(non_snake_case)]
pub fn H5FD__family_lock() -> Result<()> {
    Err(unsupported_vfd_driver("family"))
}

#[allow(non_snake_case)]
pub fn H5FD__family_unlock() -> Result<()> {
    Err(unsupported_vfd_driver("family"))
}

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

#[allow(non_snake_case)]
pub fn H5FD_multi_populate_config() -> MultiFileConfig {
    let mut config = MultiFileConfig::default();
    config
        .memb_map
        .insert(VfdMemType::Default, FileDriverKind::Sec2);
    config
}

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

fn file_driver_kind_code(kind: FileDriverKind) -> u8 {
    match kind {
        FileDriverKind::Sec2 => 0,
        FileDriverKind::Stdio => 1,
        FileDriverKind::Core => 2,
        FileDriverKind::Direct => 3,
    }
}

fn file_driver_kind_from_code(code: u8) -> Option<FileDriverKind> {
    Some(match code {
        0 => FileDriverKind::Sec2,
        1 => FileDriverKind::Stdio,
        2 => FileDriverKind::Core,
        3 => FileDriverKind::Direct,
        _ => return None,
    })
}

#[allow(non_snake_case)]
pub fn H5FD_multi_validate_config(config: &MultiFileConfig) -> bool {
    !config.memb_map.is_empty()
}

#[allow(non_snake_case)]
pub fn H5FD_multi_sb_size(config: &MultiFileConfig) -> Result<usize> {
    config
        .memb_map
        .len()
        .checked_mul(2)
        .and_then(|payload| payload.checked_add(4))
        .ok_or_else(|| Error::InvalidFormat("multi VFD member map length overflow".into()))
}

#[allow(non_snake_case)]
pub fn H5FD_multi_sb_encode(config: &MultiFileConfig) -> Result<Vec<u8>> {
    if !H5FD_multi_validate_config(config) {
        return Err(Error::InvalidFormat("invalid multi VFD config".into()));
    }
    let count = u32::try_from(config.memb_map.len())
        .map_err(|_| Error::InvalidFormat("multi VFD member count exceeds u32".into()))?;
    let mut out = Vec::with_capacity(H5FD_multi_sb_size(config)?);
    out.extend_from_slice(&count.to_le_bytes());
    let mut entries: Vec<_> = config.memb_map.iter().collect();
    entries.sort_by_key(|(mem_type, _)| vfd_mem_type_code(**mem_type));
    for (mem_type, driver) in entries {
        out.push(vfd_mem_type_code(*mem_type));
        out.push(file_driver_kind_code(*driver));
    }
    Ok(out)
}

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
        memb_map.insert(mem_type, driver);
    }
    let config = MultiFileConfig { memb_map };
    if !H5FD_multi_validate_config(&config) {
        return Err(Error::InvalidFormat("invalid multi VFD config".into()));
    }
    Ok(config)
}

#[allow(non_snake_case)]
pub fn H5FD_multi_fapl_get(config: &MultiFileConfig) -> MultiFileConfig {
    config.clone()
}

#[allow(non_snake_case)]
pub fn H5FD_multi_fapl_copy(config: &MultiFileConfig) -> MultiFileConfig {
    config.clone()
}

#[allow(non_snake_case)]
pub fn H5FD_multi_fapl_free(_config: MultiFileConfig) {}

#[allow(non_snake_case)]
pub fn H5FD_multi_open(_path: &str, _config: &MultiFileConfig) -> Result<()> {
    Err(unsupported_vfd_driver("multi"))
}

#[allow(non_snake_case)]
pub fn H5FD_multi_close() {}

#[allow(non_snake_case)]
pub fn H5FD_multi_cmp(left: &MultiFileConfig, right: &MultiFileConfig) -> Ordering {
    left.memb_map.len().cmp(&right.memb_map.len())
}

#[allow(non_snake_case)]
pub fn H5FD_multi_query() -> u64 {
    0
}

#[allow(non_snake_case)]
pub fn H5FD_multi_get_type_map(config: &MultiFileConfig) -> &HashMap<VfdMemType, FileDriverKind> {
    &config.memb_map
}

pub fn compute_next(file: &mut MultiFileState) {
    file.memb_next.clear();
    let members: Vec<_> = file
        .memb_addr
        .iter()
        .map(|(mt, addr)| (*mt, *addr))
        .collect();
    for (mt1, addr1) in &members {
        let next = members
            .iter()
            .filter_map(|(_, addr2)| (*addr2 > *addr1).then_some(*addr2))
            .min()
            .unwrap_or(u64::MAX);
        file.memb_next.insert(*mt1, next);
    }
}

#[allow(non_snake_case)]
pub fn H5FD_multi_get_eoa() -> Result<u64> {
    Err(unsupported_vfd_driver("multi"))
}

#[allow(non_snake_case)]
pub fn H5FD_multi_set_eoa(_eoa: u64) -> Result<()> {
    Err(unsupported_vfd_driver("multi"))
}

#[allow(non_snake_case)]
pub fn H5FD_multi_get_eof() -> Result<u64> {
    Err(unsupported_vfd_driver("multi"))
}

#[allow(non_snake_case)]
pub fn H5FD_multi_get_handle() -> Result<()> {
    Err(unsupported_vfd_driver("multi"))
}

#[allow(non_snake_case)]
pub fn H5FD_multi_alloc(_size: u64) -> Result<u64> {
    Err(unsupported_vfd_driver("multi"))
}

#[allow(non_snake_case)]
pub fn H5FD_multi_free(_addr: u64, _size: u64) -> Result<()> {
    Err(unsupported_vfd_driver("multi"))
}

#[allow(non_snake_case)]
pub fn H5FD_multi_read(_addr: u64, _buf: &mut [u8]) -> Result<()> {
    Err(unsupported_vfd_driver("multi"))
}

#[allow(non_snake_case)]
pub fn H5FD_multi_write(_addr: u64, _data: &[u8]) -> Result<()> {
    Err(unsupported_vfd_driver("multi"))
}

#[allow(non_snake_case)]
pub fn H5FD_multi_flush() -> Result<()> {
    Err(unsupported_vfd_driver("multi"))
}

#[allow(non_snake_case)]
pub fn H5FD_multi_truncate() -> Result<()> {
    Err(unsupported_vfd_driver("multi"))
}

#[allow(non_snake_case)]
pub fn H5FD_multi_lock() -> Result<()> {
    Err(unsupported_vfd_driver("multi"))
}

#[allow(non_snake_case)]
pub fn H5FD_multi_unlock() -> Result<()> {
    Err(unsupported_vfd_driver("multi"))
}

#[allow(non_snake_case)]
pub fn H5FD_multi_delete(_path: &str) -> Result<()> {
    Err(unsupported_vfd_driver("multi"))
}

#[allow(non_snake_case)]
pub fn H5FD_multi_ctl(_opcode: u64) -> Result<()> {
    Err(unsupported_vfd_driver("multi"))
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SplitterFileConfig {
    pub write_only_path: Option<PathBuf>,
    pub ignore_wo_errors: bool,
}

#[allow(non_snake_case)]
pub fn H5FD__splitter_populate_config(write_only_path: Option<PathBuf>) -> SplitterFileConfig {
    SplitterFileConfig {
        write_only_path,
        ignore_wo_errors: false,
    }
}

#[allow(non_snake_case)]
pub fn H5FD__splitter_get_default_wo_path() -> &'static str {
    "%s.splitter"
}

#[allow(non_snake_case)]
pub fn H5FD_split_populate_config(write_only_path: Option<PathBuf>) -> SplitterFileConfig {
    H5FD__splitter_populate_config(write_only_path)
}

#[allow(non_snake_case)]
pub fn H5FD__splitter_register() -> Result<()> {
    Err(unsupported_vfd_driver("splitter"))
}

#[allow(non_snake_case)]
pub fn H5FD__splitter_unregister() {}

#[allow(non_snake_case)]
pub fn H5FD__splitter_fapl_get(config: &SplitterFileConfig) -> SplitterFileConfig {
    config.clone()
}

#[allow(non_snake_case)]
pub fn H5FD__splitter_fapl_copy(config: &SplitterFileConfig) -> SplitterFileConfig {
    config.clone()
}

#[allow(non_snake_case)]
pub fn H5FD__splitter_fapl_free(_config: SplitterFileConfig) {}

#[allow(non_snake_case)]
pub fn H5FD__splitter_validate_config(config: &SplitterFileConfig) -> bool {
    config
        .write_only_path
        .as_ref()
        .is_none_or(|path| !path.as_os_str().is_empty())
}

#[allow(non_snake_case)]
pub fn H5FD__splitter_open(_path: &str, _config: &SplitterFileConfig) -> Result<()> {
    Err(unsupported_vfd_driver("splitter"))
}

#[allow(non_snake_case)]
pub fn H5FD__splitter_close() {}

#[allow(non_snake_case)]
pub fn H5FD__splitter_flush() -> Result<()> {
    Err(unsupported_vfd_driver("splitter"))
}

#[allow(non_snake_case)]
pub fn H5FD__splitter_read(_addr: u64, _buf: &mut [u8]) -> Result<()> {
    Err(unsupported_vfd_driver("splitter"))
}

#[allow(non_snake_case)]
pub fn H5FD__splitter_write(_addr: u64, _data: &[u8]) -> Result<()> {
    Err(unsupported_vfd_driver("splitter"))
}

#[allow(non_snake_case)]
pub fn H5FD__splitter_get_eoa() -> Result<u64> {
    Err(unsupported_vfd_driver("splitter"))
}

#[allow(non_snake_case)]
pub fn H5FD__splitter_set_eoa(_eoa: u64) -> Result<()> {
    Err(unsupported_vfd_driver("splitter"))
}

#[allow(non_snake_case)]
pub fn H5FD__splitter_get_eof() -> Result<u64> {
    Err(unsupported_vfd_driver("splitter"))
}

#[allow(non_snake_case)]
pub fn H5FD__splitter_truncate() -> Result<()> {
    Err(unsupported_vfd_driver("splitter"))
}

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

#[allow(non_snake_case)]
pub fn H5FD__splitter_sb_encode(config: &SplitterFileConfig) -> Result<Vec<u8>> {
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
    let mut out = Vec::with_capacity(H5FD__splitter_sb_size(config)?);
    out.push(u8::from(config.ignore_wo_errors));
    out.extend_from_slice(&path_len.to_le_bytes());
    out.extend_from_slice(path);
    Ok(out)
}

#[allow(non_snake_case)]
pub fn H5FD__splitter_sb_decode(bytes: &[u8]) -> Result<SplitterFileConfig> {
    let ignore_wo_errors = *bytes
        .first()
        .ok_or_else(|| Error::InvalidFormat("splitter VFD flags are truncated".into()))?
        != 0;
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

#[allow(non_snake_case)]
pub fn H5FD__splitter_cmp(left: &SplitterFileConfig, right: &SplitterFileConfig) -> Ordering {
    left.write_only_path
        .cmp(&right.write_only_path)
        .then_with(|| left.ignore_wo_errors.cmp(&right.ignore_wo_errors))
}

#[allow(non_snake_case)]
pub fn H5FD__splitter_get_handle() -> Result<()> {
    Err(unsupported_vfd_driver("splitter"))
}

#[allow(non_snake_case)]
pub fn H5FD__splitter_lock() -> Result<()> {
    Err(unsupported_vfd_driver("splitter"))
}

#[allow(non_snake_case)]
pub fn H5FD__splitter_unlock() -> Result<()> {
    Err(unsupported_vfd_driver("splitter"))
}

#[allow(non_snake_case)]
pub fn H5FD__splitter_ctl(_opcode: u64) -> Result<()> {
    Err(unsupported_vfd_driver("splitter"))
}

#[allow(non_snake_case)]
pub fn H5FD__splitter_query() -> u64 {
    0
}

#[allow(non_snake_case)]
pub fn H5FD__splitter_alloc(_size: u64) -> Result<u64> {
    Err(unsupported_vfd_driver("splitter"))
}

#[allow(non_snake_case)]
pub fn H5FD__splitter_get_type_map() -> Result<()> {
    Err(unsupported_vfd_driver("splitter"))
}

#[allow(non_snake_case)]
pub fn H5FD__splitter_free(_addr: u64, _size: u64) -> Result<()> {
    Err(unsupported_vfd_driver("splitter"))
}

#[allow(non_snake_case)]
pub fn H5FD__splitter_delete(_path: &str) -> Result<()> {
    Err(unsupported_vfd_driver("splitter"))
}

#[allow(non_snake_case)]
pub fn H5FD__splitter_log_error(message: &str) -> String {
    message.to_string()
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LogFileConfig {
    pub log_path: Option<PathBuf>,
    pub flags: u64,
    pub buffer_size: usize,
}

#[allow(non_snake_case)]
pub fn H5FD__log_register() -> Result<()> {
    Err(unsupported_vfd_driver("log"))
}

#[allow(non_snake_case)]
pub fn H5FD__log_unregister() {}

#[allow(non_snake_case)]
pub fn H5FD__log_fapl_get(config: &LogFileConfig) -> LogFileConfig {
    config.clone()
}

#[allow(non_snake_case)]
pub fn H5FD__log_fapl_copy(config: &LogFileConfig) -> LogFileConfig {
    config.clone()
}

#[allow(non_snake_case)]
pub fn H5FD__log_fapl_free(_config: LogFileConfig) {}

#[allow(non_snake_case)]
pub fn H5FD__log_validate_config(config: &LogFileConfig) -> bool {
    config
        .log_path
        .as_ref()
        .is_none_or(|path| !path.as_os_str().is_empty())
}

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

#[allow(non_snake_case)]
pub fn H5FD__log_sb_encode(config: &LogFileConfig) -> Result<Vec<u8>> {
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
    let mut out = Vec::with_capacity(H5FD__log_sb_size(config)?);
    out.extend_from_slice(&config.flags.to_le_bytes());
    out.extend_from_slice(&buffer_size.to_le_bytes());
    out.extend_from_slice(&path_len.to_le_bytes());
    out.extend_from_slice(path);
    Ok(out)
}

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

#[allow(non_snake_case)]
pub fn H5FD__log_open(_path: &str, _config: &LogFileConfig) -> Result<()> {
    Err(unsupported_vfd_driver("log"))
}

#[allow(non_snake_case)]
pub fn H5FD__log_close() {}

#[allow(non_snake_case)]
pub fn H5FD__log_cmp(left: &LogFileConfig, right: &LogFileConfig) -> Ordering {
    left.log_path
        .cmp(&right.log_path)
        .then_with(|| left.flags.cmp(&right.flags))
        .then_with(|| left.buffer_size.cmp(&right.buffer_size))
}

#[allow(non_snake_case)]
pub fn H5FD__log_query() -> u64 {
    0
}

#[allow(non_snake_case)]
pub fn H5FD__log_alloc(_size: u64) -> Result<u64> {
    Err(unsupported_vfd_driver("log"))
}

#[allow(non_snake_case)]
pub fn H5FD__log_free(_addr: u64, _size: u64) -> Result<()> {
    Err(unsupported_vfd_driver("log"))
}

#[allow(non_snake_case)]
pub fn H5FD__log_get_eoa() -> Result<u64> {
    Err(unsupported_vfd_driver("log"))
}

#[allow(non_snake_case)]
pub fn H5FD__log_set_eoa(_eoa: u64) -> Result<()> {
    Err(unsupported_vfd_driver("log"))
}

#[allow(non_snake_case)]
pub fn H5FD__log_get_eof() -> Result<u64> {
    Err(unsupported_vfd_driver("log"))
}

#[allow(non_snake_case)]
pub fn H5FD__log_get_handle() -> Result<()> {
    Err(unsupported_vfd_driver("log"))
}

#[allow(non_snake_case)]
pub fn H5FD__log_read(_addr: u64, _buf: &mut [u8]) -> Result<()> {
    Err(unsupported_vfd_driver("log"))
}

#[allow(non_snake_case)]
pub fn H5FD__log_write(_addr: u64, _data: &[u8]) -> Result<()> {
    Err(unsupported_vfd_driver("log"))
}

#[allow(non_snake_case)]
pub fn H5FD__log_truncate() -> Result<()> {
    Err(unsupported_vfd_driver("log"))
}

#[allow(non_snake_case)]
pub fn H5FD__log_lock() -> Result<()> {
    Err(unsupported_vfd_driver("log"))
}

#[allow(non_snake_case)]
pub fn H5FD__log_unlock() -> Result<()> {
    Err(unsupported_vfd_driver("log"))
}

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

#[allow(non_snake_case)]
pub fn H5FD__ros3_register() -> Result<()> {
    Err(unsupported_vfd_driver("ROS3"))
}

#[allow(non_snake_case)]
pub fn H5FD__ros3_unregister() {}

#[allow(non_snake_case)]
pub fn H5FD__ros3_init() -> Result<()> {
    Err(unsupported_vfd_driver("ROS3"))
}

#[allow(non_snake_case)]
pub fn H5FD__ros3_term() {}

#[allow(non_snake_case)]
pub fn H5FD__ros3_validate_config(config: &Ros3Config) -> bool {
    config.endpoint.as_deref().is_none_or(|s| !s.is_empty())
        && config.region.as_deref().is_none_or(|s| !s.is_empty())
}

#[allow(non_snake_case)]
pub fn H5FD__ros3_fapl_get(config: &Ros3Config) -> Ros3Config {
    config.clone()
}

#[allow(non_snake_case)]
pub fn H5FD__ros3_fapl_copy(config: &Ros3Config) -> Ros3Config {
    config.clone()
}

#[allow(non_snake_case)]
pub fn H5FD__ros3_fapl_free(_config: Ros3Config) {}

#[allow(non_snake_case)]
pub fn H5FD__ros3_str_token_close(token: &mut Option<String>) {
    *token = None;
}

#[allow(non_snake_case)]
pub fn H5FD__ros3_str_token_delete(token: &mut Option<String>) {
    *token = None;
}

#[allow(non_snake_case)]
pub fn H5FD__ros3_str_endpoint_close(endpoint: &mut Option<String>) {
    *endpoint = None;
}

#[allow(non_snake_case)]
pub fn H5FD__ros3_str_endpoint_delete(endpoint: &mut Option<String>) {
    *endpoint = None;
}

#[allow(non_snake_case)]
pub fn H5FD__ros3_query() -> u64 {
    0
}

#[allow(non_snake_case)]
pub fn H5FD__ros3_get_eoa() -> Result<u64> {
    Err(unsupported_vfd_driver("ROS3"))
}

#[allow(non_snake_case)]
pub fn H5FD__ros3_set_eoa(_eoa: u64) -> Result<()> {
    Err(unsupported_vfd_driver("ROS3"))
}

#[allow(non_snake_case)]
pub fn H5FD__ros3_get_eof() -> Result<u64> {
    Err(unsupported_vfd_driver("ROS3"))
}

#[allow(non_snake_case)]
pub fn H5FD__ros3_get_handle() -> Result<()> {
    Err(unsupported_vfd_driver("ROS3"))
}

#[allow(non_snake_case)]
pub fn H5FD__ros3_read(_addr: u64, _buf: &mut [u8]) -> Result<()> {
    Err(unsupported_vfd_driver("ROS3"))
}

#[allow(non_snake_case)]
pub fn H5FD__ros3_write(_addr: u64, _data: &[u8]) -> Result<()> {
    Err(unsupported_vfd_driver("ROS3"))
}

#[allow(non_snake_case)]
pub fn H5FD__ros3_truncate() -> Result<()> {
    Err(unsupported_vfd_driver("ROS3"))
}

#[allow(non_snake_case)]
pub fn H5FD__ros3_reset_stats() {}

#[allow(non_snake_case)]
pub fn H5FD__ros3_print_stats() -> String {
    "ros3 statistics unavailable: ROS3 VFD unsupported".into()
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

#[allow(non_snake_case)]
pub fn H5FD__onion_ingest_revision_record(
    index: &mut OnionRevisionIndex,
    record: OnionRevisionRecord,
) {
    H5FD__onion_revision_index_insert(index, record);
}

#[allow(non_snake_case)]
pub fn H5FD__onion_archival_index_is_valid(index: &OnionRevisionIndex) -> bool {
    index
        .records
        .windows(2)
        .all(|pair| pair[0].revision <= pair[1].revision)
}

#[allow(non_snake_case)]
pub fn H5FD__onion_revision_index_destroy(index: &mut OnionRevisionIndex) {
    index.records.clear();
}

#[allow(non_snake_case)]
pub fn H5FD__onion_revision_index_init() -> OnionRevisionIndex {
    OnionRevisionIndex::default()
}

#[allow(non_snake_case)]
pub fn H5FD__onion_revision_index_resize(index: &mut OnionRevisionIndex, capacity: usize) {
    if index.records.capacity() < capacity {
        index.records.reserve(capacity - index.records.capacity());
    }
}

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

#[allow(non_snake_case)]
pub fn H5FD__onion_revision_record_encode(record: &OnionRevisionRecord) -> Vec<u8> {
    let mut out = Vec::with_capacity(24);
    out.extend_from_slice(&record.revision.to_le_bytes());
    out.extend_from_slice(&record.address.to_le_bytes());
    out.extend_from_slice(&record.size.to_le_bytes());
    out
}

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

#[allow(non_snake_case)]
pub fn H5FD__onion_ingest_header(bytes: &[u8]) -> Result<OnionHeader> {
    H5FD__onion_sb_decode(bytes)
}

#[allow(non_snake_case)]
pub fn H5FD__onion_write_header(header: &OnionHeader) -> Vec<u8> {
    H5FD__onion_header_encode(header)
}

#[allow(non_snake_case)]
pub fn H5FD__onion_header_encode(header: &OnionHeader) -> Vec<u8> {
    H5FD__onion_sb_encode(header)
}

#[allow(non_snake_case)]
pub fn H5FD__onion_register() -> Result<()> {
    Err(unsupported_vfd_driver("onion"))
}

#[allow(non_snake_case)]
pub fn H5FD__onion_unregister() {}

#[allow(non_snake_case)]
pub fn H5FD__onion_sb_size(_header: &OnionHeader) -> usize {
    10
}

#[allow(non_snake_case)]
pub fn H5FD__onion_sb_encode(header: &OnionHeader) -> Vec<u8> {
    let mut out = Vec::with_capacity(10);
    out.push(header.version);
    out.push(header.flags);
    out.extend_from_slice(&header.revision_count.to_le_bytes());
    out
}

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

#[allow(non_snake_case)]
pub fn H5FD__onion_close() {}

#[allow(non_snake_case)]
pub fn H5FD__onion_get_eoa() -> Result<u64> {
    Err(unsupported_vfd_driver("onion"))
}

#[allow(non_snake_case)]
pub fn H5FD__onion_get_eof() -> Result<u64> {
    Err(unsupported_vfd_driver("onion"))
}

#[allow(non_snake_case)]
pub fn H5FD__onion_get_legit_fapl_id() -> Result<()> {
    Err(unsupported_vfd_driver("onion"))
}

#[allow(non_snake_case)]
pub fn H5FD__onion_create_truncate_onion(_path: &str) -> Result<()> {
    Err(unsupported_vfd_driver("onion"))
}

#[allow(non_snake_case)]
pub fn H5FD__onion_remove_unused_symbols(symbols: &mut Vec<String>) {
    symbols.retain(|symbol| !symbol.is_empty());
}

#[allow(non_snake_case)]
pub fn H5FD__onion_parse_config_str(config: &str) -> HashMap<String, String> {
    config
        .split(',')
        .filter_map(|entry| entry.split_once('='))
        .map(|(key, value)| (key.trim().to_string(), value.trim().to_string()))
        .collect()
}

#[allow(non_snake_case)]
pub fn H5FD__onion_open(_path: &str) -> Result<()> {
    Err(unsupported_vfd_driver("onion"))
}

#[allow(non_snake_case)]
pub fn H5FD__onion_open_rw(_path: &str) -> Result<()> {
    Err(unsupported_vfd_driver("onion"))
}

#[allow(non_snake_case)]
pub fn H5FD__onion_read(_addr: u64, _buf: &mut [u8]) -> Result<()> {
    Err(unsupported_vfd_driver("onion"))
}

#[allow(non_snake_case)]
pub fn H5FD__onion_set_eoa(_eoa: u64) -> Result<()> {
    Err(unsupported_vfd_driver("onion"))
}

#[allow(non_snake_case)]
pub fn H5FD__onion_write(_addr: u64, _data: &[u8]) -> Result<()> {
    Err(unsupported_vfd_driver("onion"))
}

#[allow(non_snake_case)]
pub fn H5FD__onion_ctl(_opcode: u64) -> Result<()> {
    Err(unsupported_vfd_driver("onion"))
}

#[allow(non_snake_case)]
pub fn H5FDonion_get_revision_count(header: &OnionHeader) -> u64 {
    header.revision_count
}

#[allow(non_snake_case)]
pub fn H5FD__get_onion_revision_count(header: &OnionHeader) -> u64 {
    H5FDonion_get_revision_count(header)
}

#[allow(non_snake_case)]
pub fn H5FD__onion_write_final_history(index: &OnionRevisionIndex) -> Result<Vec<u8>> {
    H5FD__onion_history_encode(index)
}

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

#[allow(non_snake_case)]
pub fn H5FD__onion_write_history(index: &OnionRevisionIndex) -> Result<Vec<u8>> {
    H5FD__onion_history_encode(index)
}

#[allow(non_snake_case)]
pub fn H5FD__onion_history_encode(index: &OnionRevisionIndex) -> Result<Vec<u8>> {
    let len = index
        .records
        .len()
        .checked_mul(24)
        .ok_or_else(|| Error::InvalidFormat("onion revision history length overflow".into()))?;
    let mut out = Vec::with_capacity(len);
    for record in &index.records {
        out.extend_from_slice(&H5FD__onion_revision_record_encode(record));
    }
    Ok(out)
}

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

#[allow(non_snake_case)]
pub fn H5FD__multi_register() -> Result<()> {
    Err(unsupported_vfd_driver("multi"))
}

#[allow(non_snake_case)]
pub fn H5FD__multi_unregister() {}

#[allow(non_snake_case)]
pub fn H5FD__ioc_calculate_target_ioc(addr: u64, config: &SubfilingConfig) -> u32 {
    let count = config.ioc_count.max(1);
    let stripe = config.stripe_size.max(1);
    u32::try_from((addr / stripe) % u64::from(count)).unwrap_or(0)
}

#[allow(non_snake_case)]
pub fn H5FD__ioc_write_independent_async(_request: &VfdIoRequest) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling IOC async"))
}

#[allow(non_snake_case)]
pub fn H5FD__ioc_read_independent_async(_request: &mut VfdIoRequest) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling IOC async"))
}

#[allow(non_snake_case)]
pub fn H5FD__ioc_async_completion(entry: &mut IocQueueEntry) {
    entry.complete = true;
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_new_object_id(previous: u64) -> u64 {
    previous.saturating_add(1)
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_get_object(
    objects: &[SubfilingObject],
    id: u64,
) -> Option<&SubfilingObject> {
    objects.iter().find(|object| object.id == id)
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_free_object(_object: SubfilingObject) {}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_free_context(_config: SubfilingConfig) {}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_free_topology(_objects: Vec<SubfilingObject>) {}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_open_stub_file(_path: &str) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling"))
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_open_subfiles(_path: &str, _config: &SubfilingConfig) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling"))
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_setup_context(config: &SubfilingConfig) -> SubfilingConfig {
    config.clone()
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_init_app_topology(config: &SubfilingConfig) -> Vec<u32> {
    (0..config.ioc_count).collect()
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_get_ioc_selection_criteria_from_env() -> Option<String> {
    std::env::var("HDF5_SUBFILING_IOC_SELECTION").ok()
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_find_cached_topology_info() -> Option<Vec<u32>> {
    None
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_init_app_layout(config: &SubfilingConfig) -> SubfilingConfig {
    config.clone()
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_gather_topology_info(config: &SubfilingConfig) -> Vec<u32> {
    H5FD__subfiling_init_app_topology(config)
}

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

#[allow(non_snake_case)]
pub fn H5FD__subfiling_identify_ioc_ranks(config: &SubfilingConfig) -> Vec<u32> {
    H5FD__subfiling_init_app_topology(config)
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_init_context() -> SubfilingConfig {
    H5FD__subfiling_get_default_config()
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_record_fid_map_entry(_fid: u64, _object: &SubfilingObject) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling"))
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_get_default_ioc_config() -> IocConfig {
    IocConfig {
        thread_pool_size: 1,
        queue_depth: 64,
    }
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_ioc_open_files(_config: &SubfilingConfig) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling IOC"))
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_create_config_file(_path: &str, _config: &SubfilingConfig) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling config file"))
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_open_config_file(_path: &str) -> Result<SubfilingConfig> {
    Err(unsupported_vfd_driver("subfiling config file"))
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_get_config_from_file(_path: &str) -> Result<SubfilingConfig> {
    Err(unsupported_vfd_driver("subfiling config file"))
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_resolve_pathname(path: &str) -> PathBuf {
    PathBuf::from(path)
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_close_subfiles() {}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_set_config_prop(config: &mut SubfilingConfig, stripe_size: u64) {
    config.stripe_size = stripe_size;
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_log(message: &str) -> String {
    format!("{message}\n")
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_log_nonewline(message: &str) -> String {
    message.to_string()
}

#[allow(non_snake_case)]
pub fn H5FD__ioc_register() -> Result<()> {
    Err(unsupported_vfd_driver("subfiling IOC"))
}

#[allow(non_snake_case)]
pub fn H5FD__ioc_unregister() {}

#[allow(non_snake_case)]
pub fn H5FD__ioc_init() -> Result<()> {
    Err(unsupported_vfd_driver("subfiling IOC"))
}

#[allow(non_snake_case)]
pub fn H5FD__ioc_term() {}

#[allow(non_snake_case)]
pub fn H5FD__ioc_validate_config(config: &IocConfig) -> bool {
    config.thread_pool_size > 0 && config.queue_depth > 0
}

#[allow(non_snake_case)]
pub fn H5FD__ioc_sb_size(_config: &IocConfig) -> usize {
    16
}

#[allow(non_snake_case)]
pub fn H5FD__ioc_sb_encode(config: &IocConfig) -> Result<Vec<u8>> {
    if !H5FD__ioc_validate_config(config) {
        return Err(Error::InvalidFormat("invalid subfiling IOC config".into()));
    }
    let thread_pool_size = u64::try_from(config.thread_pool_size)
        .map_err(|_| Error::InvalidFormat("subfiling IOC thread pool size exceeds u64".into()))?;
    let queue_depth = u64::try_from(config.queue_depth)
        .map_err(|_| Error::InvalidFormat("subfiling IOC queue depth exceeds u64".into()))?;
    let mut out = Vec::with_capacity(16);
    out.extend_from_slice(&thread_pool_size.to_le_bytes());
    out.extend_from_slice(&queue_depth.to_le_bytes());
    Ok(out)
}

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

#[allow(non_snake_case)]
pub fn H5FD__ioc_fapl_get(config: &IocConfig) -> IocConfig {
    config.clone()
}

#[allow(non_snake_case)]
pub fn H5FD__ioc_fapl_copy(config: &IocConfig) -> IocConfig {
    config.clone()
}

#[allow(non_snake_case)]
pub fn H5FD__ioc_fapl_free(_config: IocConfig) {}

#[allow(non_snake_case)]
pub fn H5FD__ioc_open(_path: &str, _config: &IocConfig) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling IOC"))
}

#[allow(non_snake_case)]
pub fn H5FD__ioc_close_int() {}

#[allow(non_snake_case)]
pub fn H5FD__ioc_close() {}

#[allow(non_snake_case)]
pub fn H5FD__ioc_query() -> u64 {
    0
}

#[allow(non_snake_case)]
pub fn H5FD__ioc_get_eoa() -> Result<u64> {
    Err(unsupported_vfd_driver("subfiling IOC"))
}

#[allow(non_snake_case)]
pub fn H5FD__ioc_set_eoa(_eoa: u64) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling IOC"))
}

#[allow(non_snake_case)]
pub fn H5FD__ioc_get_eof() -> Result<u64> {
    Err(unsupported_vfd_driver("subfiling IOC"))
}

#[allow(non_snake_case)]
pub fn H5FD__ioc_read(_addr: u64, _buf: &mut [u8]) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling IOC"))
}

#[allow(non_snake_case)]
pub fn H5FD__ioc_write(_addr: u64, _data: &[u8]) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling IOC"))
}

#[allow(non_snake_case)]
pub fn H5FD__ioc_write_vector(_requests: &[VfdIoRequest]) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling IOC"))
}

#[allow(non_snake_case)]
pub fn H5FD__ioc_truncate() -> Result<()> {
    Err(unsupported_vfd_driver("subfiling IOC"))
}

#[allow(non_snake_case)]
pub fn H5FD__ioc_delete(_path: &str) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling IOC"))
}

#[allow(non_snake_case)]
pub fn H5FD__ioc_write_vector_internal(_requests: &[VfdIoRequest]) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling IOC"))
}

#[allow(non_snake_case)]
pub fn H5FD__ioc_read_vector_internal(_requests: &mut [VfdIoRequest]) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling IOC"))
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling__truncate_sub_files(_config: &SubfilingConfig) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling"))
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling__get_real_eof() -> Result<u64> {
    Err(unsupported_vfd_driver("subfiling"))
}

#[allow(non_snake_case)]
pub fn H5FD__ioc_init_threads() -> Result<()> {
    Err(unsupported_vfd_driver("subfiling IOC threads"))
}

#[allow(non_snake_case)]
pub fn H5FD__ioc_finalize_threads() {}

#[allow(non_snake_case)]
pub fn H5FD__ioc_thread_main() -> Result<()> {
    Err(unsupported_vfd_driver("subfiling IOC threads"))
}

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

#[allow(non_snake_case)]
pub fn H5FD__ioc_handle_work_request(_entry: &mut IocQueueEntry) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling IOC"))
}

#[allow(non_snake_case)]
pub fn H5FD__ioc_send_ack_to_client(entry: &mut IocQueueEntry) {
    entry.complete = true;
}

#[allow(non_snake_case)]
pub fn H5FD__ioc_send_nack_to_client(entry: &mut IocQueueEntry) {
    entry.complete = false;
}

#[allow(non_snake_case)]
pub fn H5FD__ioc_file_queue_write_indep(request: VfdIoRequest) -> IocQueueEntry {
    IocQueueEntry {
        request,
        complete: false,
    }
}

#[allow(non_snake_case)]
pub fn H5FD__ioc_file_queue_read_indep(request: VfdIoRequest) -> IocQueueEntry {
    IocQueueEntry {
        request,
        complete: false,
    }
}

#[allow(non_snake_case)]
pub fn H5FD__ioc_file_truncate() -> Result<()> {
    Err(unsupported_vfd_driver("subfiling IOC"))
}

#[allow(non_snake_case)]
pub fn H5FD__ioc_file_report_eof() -> Result<u64> {
    Err(unsupported_vfd_driver("subfiling IOC"))
}

#[allow(non_snake_case)]
pub fn H5FD__ioc_io_queue_alloc_entry(request: VfdIoRequest) -> IocQueueEntry {
    IocQueueEntry {
        request,
        complete: false,
    }
}

#[allow(non_snake_case)]
pub fn H5FD__ioc_io_queue_add_entry(queue: &mut Vec<IocQueueEntry>, entry: IocQueueEntry) {
    queue.push(entry);
}

#[allow(non_snake_case)]
pub fn H5FD__ioc_io_queue_dispatch_eligible_entries(queue: &mut [IocQueueEntry]) {
    for entry in queue {
        entry.complete = true;
    }
}

#[allow(non_snake_case)]
pub fn H5FD__ioc_io_queue_complete_entry(entry: &mut IocQueueEntry) {
    entry.complete = true;
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_register() -> Result<()> {
    Err(unsupported_vfd_driver("subfiling"))
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_unregister() {}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_init() -> Result<()> {
    Err(unsupported_vfd_driver("subfiling"))
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_term() {}

#[allow(non_snake_case)]
pub fn H5FDsubfiling_get_file_mapping(config: &SubfilingConfig) -> Vec<u32> {
    H5FD__subfiling_identify_ioc_ranks(config)
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_get_default_config() -> SubfilingConfig {
    SubfilingConfig {
        ioc_count: 1,
        stripe_size: 64 * 1024 * 1024,
        stripe_count: 1,
    }
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_sb_size(_config: &SubfilingConfig) -> usize {
    16
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_sb_encode(config: &SubfilingConfig) -> Result<Vec<u8>> {
    if config.stripe_size == 0 || config.ioc_count == 0 || config.stripe_count == 0 {
        return Err(Error::InvalidFormat("invalid subfiling VFD config".into()));
    }
    let mut out = Vec::with_capacity(16);
    out.extend_from_slice(&config.stripe_size.to_le_bytes());
    out.extend_from_slice(&config.ioc_count.to_le_bytes());
    out.extend_from_slice(&config.stripe_count.to_le_bytes());
    Ok(out)
}

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

#[allow(non_snake_case)]
pub fn H5FD__subfiling_fapl_get(config: &SubfilingConfig) -> SubfilingConfig {
    config.clone()
}

#[allow(non_snake_case)]
pub fn H5FD__copy_plist<T: Clone>(value: &T) -> T {
    value.clone()
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_fapl_copy(config: &SubfilingConfig) -> SubfilingConfig {
    config.clone()
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_fapl_free(_config: SubfilingConfig) {}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_open(_path: &str, _config: &SubfilingConfig) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling"))
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_close_int() {}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_close() {}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_cmp(left: &SubfilingConfig, right: &SubfilingConfig) -> Ordering {
    H5FD__subfiling_compare_layout_nodelocal(left, right)
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_query() -> u64 {
    0
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_get_eoa() -> Result<u64> {
    Err(unsupported_vfd_driver("subfiling"))
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_set_eoa(_eoa: u64) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling"))
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_get_eof() -> Result<u64> {
    Err(unsupported_vfd_driver("subfiling"))
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_get_handle() -> Result<()> {
    Err(unsupported_vfd_driver("subfiling"))
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_write_vector(_requests: &[VfdIoRequest]) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling"))
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_truncate() -> Result<()> {
    Err(unsupported_vfd_driver("subfiling"))
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_delete(_path: &str) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling"))
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_ctl(_opcode: u64) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling"))
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_io_helper(_requests: &[VfdIoRequest]) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling"))
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_mirror_writes_to_stub(_enabled: bool) -> Result<()> {
    Err(unsupported_vfd_driver("subfiling"))
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_generate_io_vectors(requests: &[VfdIoRequest]) -> Vec<VfdIoRequest> {
    requests.to_vec()
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_get_iovec_sizes(requests: &[VfdIoRequest]) -> Vec<usize> {
    requests.iter().map(|request| request.bytes.len()).collect()
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_translate_io_req_to_iovec(request: &VfdIoRequest) -> VfdIoRequest {
    request.clone()
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_iovec_fill_first(requests: &[VfdIoRequest]) -> Option<VfdIoRequest> {
    requests.first().cloned()
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_iovec_fill_last(requests: &[VfdIoRequest]) -> Option<VfdIoRequest> {
    requests.last().cloned()
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_iovec_fill_first_last(requests: &[VfdIoRequest]) -> Vec<VfdIoRequest> {
    let mut out = Vec::new();
    if let Some(first) = H5FD__subfiling_iovec_fill_first(requests) {
        out.push(first);
    }
    if requests.len() > 1 {
        if let Some(last) = H5FD__subfiling_iovec_fill_last(requests) {
            out.push(last);
        }
    }
    out
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_iovec_fill_uniform(requests: &[VfdIoRequest]) -> Vec<VfdIoRequest> {
    requests.to_vec()
}

#[allow(non_snake_case)]
pub fn H5FD__subfiling_cast_to_void<T>(value: T) -> T {
    value
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{
        Error, FamilyFileConfig, FileDriverKind, IocConfig, LocalFileDriver, LogFileConfig,
        MirrorXmit, MultiFileConfig, OnionHeader, OnionRevisionIndex, OnionRevisionRecord,
        SplitterFileConfig, SubfilingConfig, VfdMemType,
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
    }

    #[test]
    fn family_multi_splitter_and_log_configs_round_trip() {
        let family = FamilyFileConfig {
            member_size: 4096,
            printf_filename: "member-%03d.h5".into(),
        };
        let family_bytes = super::H5FD__family_sb_encode(&family).unwrap();
        assert_eq!(
            super::H5FD__family_sb_size(&family).unwrap(),
            family_bytes.len()
        );
        assert_eq!(
            super::H5FD__family_sb_decode(&family_bytes).unwrap(),
            family
        );
        assert!(super::H5FD__family_sb_encode(&FamilyFileConfig {
            member_size: 0,
            printf_filename: "member-%03d.h5".into(),
        })
        .is_err());

        let mut multi = MultiFileConfig::default();
        multi
            .memb_map
            .insert(VfdMemType::Default, FileDriverKind::Sec2);
        multi
            .memb_map
            .insert(VfdMemType::RawData, FileDriverKind::Core);
        let multi_bytes = super::H5FD_multi_sb_encode(&multi).unwrap();
        assert_eq!(
            super::H5FD_multi_sb_size(&multi).unwrap(),
            multi_bytes.len()
        );
        assert_eq!(super::H5FD_multi_sb_decode(&multi_bytes).unwrap(), multi);
        assert!(super::H5FD_multi_sb_encode(&MultiFileConfig::default()).is_err());

        let splitter = SplitterFileConfig {
            write_only_path: Some(PathBuf::from("mirror.h5")),
            ignore_wo_errors: true,
        };
        let splitter_bytes = super::H5FD__splitter_sb_encode(&splitter).unwrap();
        assert_eq!(
            super::H5FD__splitter_sb_size(&splitter).unwrap(),
            splitter_bytes.len()
        );
        assert_eq!(
            super::H5FD__splitter_sb_decode(&splitter_bytes).unwrap(),
            splitter
        );
        assert!(super::H5FD__splitter_sb_encode(&SplitterFileConfig {
            write_only_path: Some(PathBuf::from("")),
            ignore_wo_errors: true,
        })
        .is_err());

        let log = LogFileConfig {
            log_path: Some(PathBuf::from("driver.log")),
            flags: 0x55,
            buffer_size: 8192,
        };
        let log_bytes = super::H5FD__log_sb_encode(&log).unwrap();
        assert_eq!(super::H5FD__log_sb_size(&log).unwrap(), log_bytes.len());
        assert_eq!(super::H5FD__log_sb_decode(&log_bytes).unwrap(), log);
        assert!(super::H5FD__log_sb_encode(&LogFileConfig {
            log_path: Some(PathBuf::from("")),
            flags: 0,
            buffer_size: 0,
        })
        .is_err());

        assert_eq!(
            super::H5FD_mirror_xmit_decode_lock(&[]).unwrap(),
            MirrorXmit::Lock
        );
        assert_eq!(
            super::H5FD_mirror_xmit_decode_open(&super::H5FD_mirror_xmit_encode_open("mirror.h5",))
                .unwrap(),
            MirrorXmit::Open {
                path: "mirror.h5".into()
            }
        );
        assert_eq!(
            super::H5FD_mirror_xmit_decode_reply(&super::H5FD_mirror_xmit_encode_reply(0)).unwrap(),
            MirrorXmit::Reply { status: 0 }
        );
        assert_eq!(
            super::H5FD_mirror_xmit_decode_set_eoa(&super::H5FD_mirror_xmit_encode_set_eoa(4096))
                .unwrap(),
            MirrorXmit::SetEoa { eoa: 4096 }
        );
        assert_eq!(
            super::H5FD_mirror_xmit_decode_write(&super::H5FD_mirror_xmit_encode_write(
                8192, b"abc",
            ))
            .unwrap(),
            MirrorXmit::Write {
                addr: 8192,
                data: b"abc".to_vec()
            }
        );

        let onion = OnionHeader {
            version: 1,
            flags: 0x2,
            revision_count: 3,
        };
        let onion_bytes = super::H5FD__onion_sb_encode(&onion);
        assert_eq!(super::H5FD__onion_sb_size(&onion), onion_bytes.len());
        assert_eq!(super::H5FD__onion_sb_decode(&onion_bytes).unwrap(), onion);

        let ioc = IocConfig {
            thread_pool_size: 4,
            queue_depth: 16,
        };
        let ioc_bytes = super::H5FD__ioc_sb_encode(&ioc).unwrap();
        assert_eq!(super::H5FD__ioc_sb_size(&ioc), ioc_bytes.len());
        assert_eq!(super::H5FD__ioc_sb_decode(&ioc_bytes).unwrap(), ioc);
        assert!(super::H5FD__ioc_sb_encode(&IocConfig {
            thread_pool_size: 0,
            queue_depth: 16,
        })
        .is_err());

        let subfiling = SubfilingConfig {
            ioc_count: 2,
            stripe_size: 1024,
            stripe_count: 8,
        };
        let subfiling_bytes = super::H5FD__subfiling_sb_encode(&subfiling).unwrap();
        assert_eq!(
            super::H5FD__subfiling_sb_size(&subfiling),
            subfiling_bytes.len()
        );
        assert_eq!(
            super::H5FD__subfiling_sb_decode(&subfiling_bytes).unwrap(),
            subfiling
        );
        assert!(super::H5FD__subfiling_sb_encode(&SubfilingConfig {
            ioc_count: 0,
            stripe_size: 1024,
            stripe_count: 8,
        })
        .is_err());

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
        let history_bytes = super::H5FD__onion_history_encode(&history).unwrap();
        assert_eq!(
            super::H5FD__onion_ingest_history(&history_bytes).unwrap(),
            history
        );
    }

    #[test]
    fn vfd_config_decoders_reject_truncated_or_invalid_payloads() {
        assert!(matches!(
            super::H5FD__family_sb_decode(&[0; 4]).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        assert!(matches!(
            super::H5FD_multi_sb_decode(&[1, 0, 0, 0, 99, 0]).unwrap_err(),
            Error::InvalidFormat(_)
        ));
        assert!(matches!(
            super::H5FD__splitter_sb_decode(&[0, 4, 0, 0, 0, b'a']).unwrap_err(),
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
            super::H5FD_mirror_xmit_decode_open(&[0xff]).unwrap_err(),
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
            super::H5FD_mirror_xmit_decode_write(&[0; 7]).unwrap_err(),
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
    }
}
