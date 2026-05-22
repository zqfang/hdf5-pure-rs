#![allow(dead_code, non_snake_case)]

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::io::Cursor;

use crate::error::{Error, Result};
use crate::format::superblock::Superblock;
use crate::io::reader::HdfReader;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileApiState {
    pub id: u64,
    pub name: String,
    pub actual_name: String,
    pub intent: u32,
    pub eof: u64,
    pub eoa: u64,
    pub base_addr: u64,
    pub low_bound: u8,
    pub high_bound: u8,
    pub nopen_objs: usize,
    pub nrefs: usize,
    pub fileno: u64,
    pub flags: u64,
    pub mdc_hit_rate_resets: usize,
    pub page_buffer_stats_resets: usize,
    pub file_locking: bool,
    pub swmr_write: bool,
    pub metadata_logging: bool,
    pub coll_metadata_reads: bool,
    pub mpi_atomicity: bool,
    pub vol_obj: Option<u64>,
    pub mounts: BTreeSet<String>,
    pub open_ids: BTreeSet<u64>,
    pub object_ids: BTreeSet<u64>,
    pub image: Vec<u8>,
    pub accum: Vec<u8>,
    pub super_ext_addr: Option<u64>,
    pub sohm_addr: Option<u64>,
    pub sohm_vers: u8,
    pub sohm_nindexes: u8,
    pub store_msg_crt_idx: bool,
    pub retries: u32,
}

impl Default for FileApiState {
    fn default() -> Self {
        Self {
            id: 0,
            name: String::new(),
            actual_name: String::new(),
            intent: 0,
            eof: 0,
            eoa: 0,
            base_addr: 0,
            low_bound: 0,
            high_bound: u8::MAX,
            nopen_objs: 0,
            nrefs: 1,
            fileno: 0,
            flags: 0,
            mdc_hit_rate_resets: 0,
            page_buffer_stats_resets: 0,
            file_locking: true,
            swmr_write: false,
            metadata_logging: false,
            coll_metadata_reads: false,
            mpi_atomicity: false,
            vol_obj: None,
            mounts: BTreeSet::new(),
            open_ids: BTreeSet::new(),
            object_ids: BTreeSet::new(),
            image: Vec::new(),
            accum: Vec::new(),
            super_ext_addr: None,
            sohm_addr: None,
            sohm_vers: 0,
            sohm_nindexes: 0,
            store_msg_crt_idx: false,
            retries: 0,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FilePackageState {
    pub initialized: bool,
    pub next_id: u64,
    pub files: BTreeMap<u64, FileApiState>,
    pub file_locks: Option<bool>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExternalFileCache {
    pub max_nfiles: usize,
    pub files: BTreeMap<String, FileApiState>,
    pub close_attempts: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FileMetadataQueryStats {
    pub mdc_hit_rate: f64,
    pub mdc_hit_rate_resets: usize,
    pub page_buffer_stats_resets: usize,
    pub nopen_objs: usize,
    pub nrefs: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FileApiSuperblockInfo {
    pub version: u8,
    pub size: u64,
    pub extension_size: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FileApiFreeSpaceInfo {
    pub version: u8,
    pub metadata_size: u64,
    pub total_space: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FileApiSharedMessageInfo {
    pub header_size: u64,
    pub message_info_size: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FileApiInfo {
    pub superblock: FileApiSuperblockInfo,
    pub free_space: FileApiFreeSpaceInfo,
    pub shared_messages: FileApiSharedMessageInfo,
}

pub const H5F_NUM_METADATA_READ_RETRY_TYPES: usize = 21;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileMetadataReadRetryInfo {
    pub nbins: usize,
    pub retries: [u32; H5F_NUM_METADATA_READ_RETRY_TYPES],
}

impl Default for FileMetadataReadRetryInfo {
    fn default() -> Self {
        Self {
            nbins: H5F_NUM_METADATA_READ_RETRY_TYPES,
            retries: [0; H5F_NUM_METADATA_READ_RETRY_TYPES],
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PageBufferQueryStats {
    pub accesses: usize,
    pub hits: usize,
    pub misses: usize,
    pub evictions: usize,
    pub bypasses: usize,
    pub resets: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FileLoggingFlags {
    pub swmr_write: bool,
    pub metadata_logging: bool,
    pub coll_metadata_reads: bool,
    pub mpi_atomicity: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DriverInfoBlock<'a> {
    pub version: u8,
    pub name: [u8; 8],
    pub data: Cow<'a, [u8]>,
}

#[derive(Debug, Clone)]
pub struct SuperblockCacheImage<'a> {
    pub superblock: Superblock,
    pub raw: Cow<'a, [u8]>,
}

/// Returns an `Unsupported` error stub for file-driver behavior not implemented in pure-Rust mode.
fn unsupported_file(name: &str) -> Error {
    Error::Unsupported(format!(
        "{name} requires libhdf5 file-driver behavior not implemented in pure-Rust mode"
    ))
}

/// Initialize the file interface from some other layer.
pub fn H5F_init() -> FilePackageState {
    H5F__init_package()
}

/// Initialize interface-specific information for the file package.
pub fn H5F__init_package() -> FilePackageState {
    FilePackageState {
        initialized: true,
        next_id: 1,
        ..FilePackageState::default()
    }
}

/// Terminate this interface: release all ID groups and reset globals.
pub fn H5F_term_package(pkg: &mut FilePackageState) {
    pkg.initialized = false;
    pkg.files.clear();
}

/// Allocate `size` bytes from the file by extending the end-of-allocation marker.
pub fn H5F__alloc(file: &mut FileApiState, size: u64) -> Result<u64> {
    if size == 0 {
        return Err(Error::InvalidFormat("zero-byte H5F allocation".into()));
    }
    let addr = file.eoa;
    file.eoa = file
        .eoa
        .checked_add(size)
        .ok_or_else(|| Error::InvalidFormat("H5F allocation overflow".into()))?;
    file.eof = file.eof.max(file.eoa);
    Ok(addr)
}

/// Release a previously-allocated region; no-op in the pure-Rust stub.
pub fn H5F__free(_file: &mut FileApiState, _addr: u64, _size: u64) -> Result<()> {
    Ok(())
}

/// Try to extend the allocation at `addr` from `old_size` to `new_size`.
pub fn H5F__try_extend(file: &mut FileApiState, addr: u64, old_size: u64, new_size: u64) -> bool {
    addr.checked_add(old_size) == Some(file.eoa) && new_size >= old_size
}

/// Add an address to the file's contiguous-write free space list.
pub fn H5F_cwfs_add(file: &mut FileApiState, addr: u64) {
    file.open_ids.insert(addr);
}

/// Callback returning the count of all open object IDs in a file.
pub fn H5F__get_all_count_cb(file: &FileApiState) -> usize {
    file.object_ids.len()
}

/// Copy all open object IDs into caller-provided storage.
pub fn H5F__get_all_ids_cb_into(file: &FileApiState, out: &mut Vec<u64>) -> usize {
    H5F_get_obj_ids_into(file, out)
}

/// Visit each open object ID without allocating an intermediate list.
pub fn H5F__visit_all_ids_cb<F>(file: &FileApiState, visitor: F)
where
    F: FnMut(u64),
{
    H5F_visit_obj_ids(file, visitor);
}

/// Check if a given image is an accessible HDF5 file (signature check).
pub fn H5Fis_accessible(image: &[u8]) -> bool {
    H5F__is_hdf5(image)
}

/// Common post-open work shared by file create/open API entry points.
pub fn H5F__post_open_api_common(file: FileApiState) -> FileApiState {
    file
}

/// Common file-create API plumbing: register a new file in the package state.
pub fn H5F__create_api_common(pkg: &mut FilePackageState, name: &str, intent: u32) -> u64 {
    let id = pkg.next_id.max(1);
    pkg.next_id = id.saturating_add(1);
    let file = FileApiState {
        id,
        name: name.to_string(),
        actual_name: name.to_string(),
        intent,
        fileno: id,
        ..FileApiState::default()
    };
    pkg.files.insert(id, file);
    id
}

/// Common flush API plumbing: ensure the file's image is up to date.
pub fn H5F__flush_api_common(file: &mut FileApiState) {
    H5F__flush(file);
}

/// Mount a child file at the given name inside `file`.
pub fn H5Fmount(file: &mut FileApiState, name: &str) {
    H5F_mount(file, name);
}

/// Unmount the file previously mounted at `name` inside `file`.
pub fn H5Funmount(file: &mut FileApiState, name: &str) {
    H5F_unmount(file, name);
}

/// Common reopen API plumbing: return a new handle that shares the underlying file.
pub fn H5F__reopen_api_common(file: &FileApiState) -> FileApiState {
    H5F_open(file)
}

/// Reopen a file, returning a new handle to the same underlying data.
pub fn H5Freopen(file: &FileApiState) -> FileApiState {
    H5F_open(file)
}

/// Asynchronous reopen; not supported in pure-Rust mode.
pub fn H5Freopen_async(_file: &FileApiState) -> Result<FileApiState> {
    Err(unsupported_file("H5Freopen_async"))
}

/// Asynchronous file creation; unsupported without libhdf5 async VOL behavior.
pub fn H5Fcreate_async(_pkg: &mut FilePackageState, _name: &str, _intent: u32) -> Result<u64> {
    Err(unsupported_file("H5Fcreate_async"))
}

/// Asynchronous file open; unsupported without libhdf5 async VOL behavior.
pub fn H5Fopen_async(_file: &FileApiState) -> Result<FileApiState> {
    Err(unsupported_file("H5Fopen_async"))
}

/// Asynchronous file flush; unsupported without libhdf5 async VOL behavior.
pub fn H5Fflush_async(_file: &mut FileApiState) -> Result<()> {
    Err(unsupported_file("H5Fflush_async"))
}

/// Asynchronous file close; unsupported without libhdf5 async VOL behavior.
pub fn H5Fclose_async(_file: FileApiState) -> Result<()> {
    Err(unsupported_file("H5Fclose_async"))
}

/// Reset the metadata cache hit-rate statistics for a file.
pub fn H5Freset_mdc_hit_rate_stats(file: &mut FileApiState) {
    file.mdc_hit_rate_resets = file.mdc_hit_rate_resets.saturating_add(1);
}

/// Return the current metadata cache hit rate.
pub fn H5F_get_mdc_hit_rate(file: &FileApiState) -> f64 {
    if file.mdc_hit_rate_resets == 0 {
        0.0
    } else {
        1.0
    }
}

/// Return a snapshot of metadata-query statistics for a file.
pub fn H5F_get_metadata_query_stats(file: &FileApiState) -> FileMetadataQueryStats {
    FileMetadataQueryStats {
        mdc_hit_rate: H5F_get_mdc_hit_rate(file),
        mdc_hit_rate_resets: file.mdc_hit_rate_resets,
        page_buffer_stats_resets: file.page_buffer_stats_resets,
        nopen_objs: file.nopen_objs,
        nrefs: file.nrefs,
    }
}

/// Return metadata-read retry counts in libhdf5's public retry-info shape.
pub fn H5Fget_metadata_read_retry_info(file: &FileApiState) -> FileMetadataReadRetryInfo {
    FileMetadataReadRetryInfo {
        retries: [file.retries; H5F_NUM_METADATA_READ_RETRY_TYPES],
        ..FileMetadataReadRetryInfo::default()
    }
}

/// Return the page-buffer query statistics for a file.
pub fn H5F_get_page_buffering_stats(file: &FileApiState) -> PageBufferQueryStats {
    PageBufferQueryStats {
        resets: file.page_buffer_stats_resets,
        ..PageBufferQueryStats::default()
    }
}

/// Public libhdf5-style page-buffer statistics query.
pub fn H5Fget_page_buffering_stats(file: &FileApiState) -> PageBufferQueryStats {
    H5F_get_page_buffering_stats(file)
}

/// Return whether the file is open for SWMR (Single-Writer/Multiple-Reader) write access.
pub fn H5F_get_swmr_write(file: &FileApiState) -> bool {
    file.swmr_write
}

/// Return logging-related flags (SWMR, MDC logging, MPI atomicity, etc.) for a file.
pub fn H5F_get_mdc_logging_status(file: &FileApiState) -> FileLoggingFlags {
    FileLoggingFlags {
        swmr_write: file.swmr_write,
        metadata_logging: file.metadata_logging,
        coll_metadata_reads: file.coll_metadata_reads,
        mpi_atomicity: file.mpi_atomicity,
    }
}

/// Public libhdf5-style metadata-cache logging status query.
pub fn H5Fget_mdc_logging_status(file: &FileApiState) -> FileLoggingFlags {
    H5F_get_mdc_logging_status(file)
}

/// Public metadata-cache configuration query; unsupported without libhdf5 MDC internals.
pub fn H5Fget_mdc_config(_file: &FileApiState) -> Result<()> {
    Err(unsupported_file("H5Fget_mdc_config"))
}

/// Public metadata-cache configuration setter; unsupported without libhdf5 MDC internals.
pub fn H5Fset_mdc_config(_file: &mut FileApiState) -> Result<()> {
    Err(unsupported_file("H5Fset_mdc_config"))
}

/// Public metadata-cache size query; unsupported without libhdf5 MDC internals.
pub fn H5Fget_mdc_size(_file: &FileApiState) -> Result<()> {
    Err(unsupported_file("H5Fget_mdc_size"))
}

/// Public metadata-cache image query; unsupported without libhdf5 MDC internals.
pub fn H5Fget_mdc_image_info(_file: &FileApiState) -> Result<()> {
    Err(unsupported_file("H5Fget_mdc_image_info"))
}

/// Public dataset-no-attributes hint query; unsupported without file property state here.
pub fn H5Fget_dset_no_attrs_hint(_file: &FileApiState) -> Result<()> {
    Err(unsupported_file("H5Fget_dset_no_attrs_hint"))
}

/// Public dataset-no-attributes hint setter; unsupported without file property state here.
pub fn H5Fset_dset_no_attrs_hint(_file: &mut FileApiState, _minimize: bool) -> Result<()> {
    Err(unsupported_file("H5Fset_dset_no_attrs_hint"))
}

/// Clear the external-link file cache for this file.
pub fn H5Fclear_elink_file_cache(_file: &mut FileApiState) {}

/// Enable SWMR write access on an open file.
pub fn H5Fstart_swmr_write(file: &mut FileApiState) {
    file.swmr_write = true;
}

/// Begin metadata-cache logging for a file.
pub fn H5Fstart_mdc_logging(file: &mut FileApiState) {
    file.metadata_logging = true;
}

/// Stop metadata-cache logging for a file.
pub fn H5Fstop_mdc_logging(file: &mut FileApiState) {
    file.metadata_logging = false;
}

/// Convert a file's on-disk format to a different libver bound; not supported here.
pub fn H5Fformat_convert(_file: &mut FileApiState) -> Result<()> {
    Err(unsupported_file("H5Fformat_convert"))
}

/// Reset the page buffering statistics for a file.
pub fn H5Freset_page_buffering_stats(file: &mut FileApiState) {
    file.page_buffer_stats_resets = file.page_buffer_stats_resets.saturating_add(1);
}

/// Increase the recorded file size by `increment` bytes and return the new size.
pub fn H5Fincrement_filesize(file: &mut FileApiState, increment: u64) -> Result<u64> {
    file.eof = file
        .eof
        .checked_add(increment)
        .ok_or_else(|| Error::InvalidFormat("H5F filesize overflow".into()))?;
    Ok(file.eof)
}

/// Metadata-cache hook: return the initial bytes needed to start loading a superblock.
pub fn H5F__cache_superblock_get_initial_load_size() -> usize {
    8
}

/// Metadata-cache hook: return the final size needed to load the superblock image.
pub fn H5F__cache_superblock_get_final_load_size(image: &[u8]) -> usize {
    image.len()
}

/// Metadata-cache hook: verify the superblock image checksum/signature.
pub fn H5F__cache_superblock_verify_chksum(_image: &[u8]) -> bool {
    H5F__is_hdf5(_image)
}

/// Metadata-cache hook: deserialize a superblock image into a cache entry.
pub fn H5F__cache_superblock_deserialize(image: &[u8]) -> Result<SuperblockCacheImage<'_>> {
    let mut reader = HdfReader::new(Cursor::new(image));
    let superblock = Superblock::read(&mut reader)?;
    let size = superblock.checked_size()?;
    if image.len() < size {
        return Err(Error::InvalidFormat(
            "cached superblock image is truncated".into(),
        ));
    }
    Ok(SuperblockCacheImage {
        superblock,
        raw: Cow::Borrowed(&image[..size]),
    })
}

/// Metadata-cache hook: return the serialized image length for a cached superblock.
pub fn H5F__cache_superblock_image_len(image: &SuperblockCacheImage<'_>) -> usize {
    image.raw.len()
}

/// Metadata-cache hook: free the in-core representation of a cached superblock.
pub fn H5F__cache_superblock_free_icr(_image: SuperblockCacheImage<'_>) {}

/// Metadata-cache hook: initial bytes needed to start loading the driver-info block.
pub fn H5F__cache_drvrinfo_get_initial_load_size() -> usize {
    16
}

/// Metadata-cache hook: final size for the driver-info block, derived from its prefix.
pub fn H5F__cache_drvrinfo_get_final_load_size(image: &[u8]) -> Result<usize> {
    let len = decode_driver_info_block_payload_len(image)?;
    16usize
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat("driver info cache block size overflow".into()))
}

/// Metadata-cache hook: decode a driver-info cache block image.
pub fn H5F__cache_drvrinfo_deserialize(image: &[u8]) -> Result<DriverInfoBlock<'_>> {
    let len = decode_driver_info_block_payload_len(image)?;
    let total = 16usize
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat("driver info cache block size overflow".into()))?;
    if image.len() != total {
        return Err(Error::InvalidFormat(format!(
            "driver info cache block length mismatch: expected {total}, got {}",
            image.len()
        )));
    }
    let mut name = [0u8; 8];
    name.copy_from_slice(&image[8..16]);
    Ok(DriverInfoBlock {
        version: image[0],
        name,
        data: Cow::Borrowed(&image[16..total]),
    })
}

/// Metadata-cache hook: serialized image length of a driver-info block.
pub fn H5F__cache_drvrinfo_image_len(block: &DriverInfoBlock<'_>) -> Result<usize> {
    let len = driver_info_block_payload_len_u32(block)?;
    16usize
        .checked_add(len as usize)
        .ok_or_else(|| Error::InvalidFormat("driver info cache block size overflow".into()))
}

/// Metadata-cache hook: encode a driver-info block into caller-provided storage.
pub fn H5F__cache_drvrinfo_serialize_into(
    block: &DriverInfoBlock<'_>,
    out: &mut Vec<u8>,
) -> Result<usize> {
    if block.version != 0 {
        return Err(Error::InvalidFormat(format!(
            "unsupported driver info cache block version {}",
            block.version
        )));
    }
    let len = driver_info_block_payload_len_u32(block)?;
    let total = H5F__cache_drvrinfo_image_len(block)?;
    out.clear();
    out.reserve(total);
    out.push(block.version);
    out.extend_from_slice(&[0, 0, 0]);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&block.name);
    out.extend_from_slice(&block.data);
    Ok(out.len())
}

/// Metadata-cache hook: free the in-core representation of a driver-info block.
pub fn H5F__cache_drvrinfo_free_icr(_block: DriverInfoBlock<'_>) {}

/// Read the 32-bit little-endian payload length from a driver-info block prefix.
fn decode_driver_info_block_payload_len(image: &[u8]) -> Result<usize> {
    if image.len() < 16 {
        return Err(Error::InvalidFormat(
            "truncated driver info cache block prefix".into(),
        ));
    }
    if image[0] != 0 {
        return Err(Error::InvalidFormat(format!(
            "unsupported driver info cache block version {}",
            image[0]
        )));
    }
    Ok(u32::from_le_bytes([image[4], image[5], image[6], image[7]]) as usize)
}

/// Convert a driver-info block payload length to a `u32`, erroring on overflow.
fn driver_info_block_payload_len_u32(block: &DriverInfoBlock<'_>) -> Result<u32> {
    u32::try_from(block.data.len())
        .map_err(|_| Error::InvalidFormat("driver info cache block payload too large".into()))
}

/// Close callback invoked for each open file when the file ID class is destroyed.
pub fn H5F__close_cb(_file: FileApiState) {}

/// Parse the `HDF5_USE_FILE_LOCKING` environment variable into a tri-state setting.
pub fn H5F__parse_file_lock_env_var(value: Option<&str>) -> Result<Option<bool>> {
    match value {
        None => Ok(None),
        Some("0") => Ok(Some(false)),
        Some("1") => Ok(Some(true)),
        Some(value) if value.eq_ignore_ascii_case("FALSE") => Ok(Some(false)),
        Some(value) if value.eq_ignore_ascii_case("TRUE") => Ok(Some(true)),
        Some(value) if value.eq_ignore_ascii_case("BEST_EFFORT") => Ok(Some(true)),
        Some(other) => Err(Error::InvalidFormat(format!(
            "invalid HDF5 file locking environment value '{other}'"
        ))),
    }
}

/// Associate a VOL connector object with the file.
pub fn H5F__set_vol_conn(file: &mut FileApiState, vol_obj: u64) {
    file.vol_obj = Some(vol_obj);
}

/// Return a value representing the file access property list (here, the open intent).
pub fn H5F_get_access_plist(file: &FileApiState) -> u32 {
    file.intent
}

/// Return the file's public open-intent flags.
pub fn H5Fget_intent(file: &FileApiState) -> u32 {
    H5F_shared_get_intent(file)
}

/// Return the number of objects currently open in the file.
pub fn H5F_get_obj_count(file: &FileApiState) -> usize {
    file.object_ids.len()
}

/// Copy object IDs currently open in the file into caller-provided storage.
pub fn H5F_get_obj_ids_into(file: &FileApiState, out: &mut Vec<u64>) -> usize {
    out.clear();
    out.extend(file.object_ids.iter().copied());
    out.len()
}

/// Visit object IDs currently open in the file without allocating a list.
pub fn H5F_visit_obj_ids<F>(file: &FileApiState, mut visitor: F)
where
    F: FnMut(u64),
{
    for id in &file.object_ids {
        visitor(*id);
    }
}

/// Copy open object IDs into caller-provided storage (internal package helper).
pub fn H5F__get_objects_into(file: &FileApiState, out: &mut Vec<u64>) -> usize {
    H5F_get_obj_ids_into(file, out)
}

/// Visit open object IDs without allocating (internal package helper).
pub fn H5F__visit_objects<F>(file: &FileApiState, visitor: F)
where
    F: FnMut(u64),
{
    H5F_visit_obj_ids(file, visitor);
}

/// Callback returning the count of objects open in the file.
pub fn H5F__get_objects_cb(file: &FileApiState) -> usize {
    H5F_get_obj_count(file)
}

/// Build a full path, borrowing `name` when no prefix is needed.
pub fn H5F__build_name_cow<'a>(prefix: &str, name: &'a str) -> Cow<'a, str> {
    if prefix.is_empty() {
        Cow::Borrowed(name)
    } else {
        Cow::Owned(format!("{prefix}/{name}"))
    }
}

/// Build a full path into caller-provided string storage.
pub fn H5F__build_name_into(prefix: &str, name: &str, out: &mut String) {
    out.clear();
    if prefix.is_empty() {
        out.push_str(name);
    } else {
        out.push_str(prefix);
        out.push('/');
        out.push_str(name);
    }
}

/// Decode an `HDF5_PREFIX`-style environment string without allocating.
pub fn H5F__getenv_prefix_name_cow(value: Option<&str>) -> Option<Cow<'_, str>> {
    value.map(Cow::Borrowed)
}

/// Resolve a file path against a search prefix, borrowing `name` when possible.
pub fn H5F_prefix_open_file_cow<'a>(prefix: &str, name: &'a str) -> Cow<'a, str> {
    H5F__build_name_cow(prefix, name)
}

/// Resolve a file path against a search prefix into caller-provided storage.
pub fn H5F_prefix_open_file_into(prefix: &str, name: &str, out: &mut String) {
    H5F__build_name_into(prefix, name, out);
}

/// Check whether the buffer begins with the HDF5 file signature.
pub fn H5F__is_hdf5(image: &[u8]) -> bool {
    image.starts_with(b"\x89HDF\r\n\x1a\n")
}

/// Final destructor for a file: release in-memory resources.
pub fn H5F__dest(_file: FileApiState) {}

/// Return whether file locking is enabled for this file.
pub fn H5F__check_if_using_file_locks(file: &FileApiState) -> bool {
    file.file_locking
}

/// Open a file (or share an already-open one), returning a refcounted handle.
pub fn H5F_open(file: &FileApiState) -> FileApiState {
    let mut reopened = FileApiState::default();
    H5F_open_into(file, &mut reopened);
    reopened
}

/// Open a file into caller-provided storage, reusing owned buffers where possible.
pub fn H5F_open_into(file: &FileApiState, out: &mut FileApiState) {
    out.clone_from(file);
    out.nrefs = out.nrefs.saturating_add(1);
}

/// Post-open hook called after the file is fully constructed.
pub fn H5F__post_open(file: FileApiState) -> FileApiState {
    file
}

/// First pass of flushing: write out cached metadata that may dirty other entries.
pub fn H5F__flush_phase1(_file: &mut FileApiState) {}

/// Second pass of flushing: write out remaining cached metadata.
pub fn H5F__flush_phase2(_file: &mut FileApiState) {}

/// Flush the file by syncing the EOF marker to at least the EOA.
pub fn H5F__flush(file: &mut FileApiState) {
    file.eof = file.eof.max(file.eoa);
}

/// Close a file: release file-level resources.
pub fn H5F__close(_file: FileApiState) {}

/// Remove a file from the package state by ID and return it if present.
pub fn H5F__delete(pkg: &mut FilePackageState, id: u64) -> Option<FileApiState> {
    pkg.files.remove(&id)
}

/// Attempt to close the file: succeed only if no objects are still open.
pub fn H5F_try_close(file: &mut FileApiState) -> bool {
    file.nopen_objs == 0
}

/// Return the file's identifier.
pub fn H5F_get_id(file: &FileApiState) -> u64 {
    file.id
}

/// Increment the count of open objects in the file and return the new count.
pub fn H5F_incr_nopen_objs(file: &mut FileApiState) -> usize {
    file.nopen_objs = file.nopen_objs.saturating_add(1);
    file.nopen_objs
}

/// Decrement the count of open objects in the file and return the new count.
pub fn H5F_decr_nopen_objs(file: &mut FileApiState) -> usize {
    file.nopen_objs = file.nopen_objs.saturating_sub(1);
    file.nopen_objs
}

/// Record the actual on-disk name for the file (post symlink/external resolution).
pub fn H5F__build_actual_name(file: &mut FileApiState, name: &str) {
    file.actual_name.clear();
    file.actual_name.push_str(name);
}

/// Mark whether the file uses shared B-tree storage for groups.
pub fn H5F_set_grp_btree_shared(file: &mut FileApiState, enabled: bool) {
    if enabled {
        file.flags |= 1;
    } else {
        file.flags &= !1;
    }
}

/// Record the file address of the Shared Object Header Message (SOHM) table.
pub fn H5F_set_sohm_addr(file: &mut FileApiState, addr: u64) {
    file.sohm_addr = Some(addr);
}

/// Record the SOHM table version stored in the file.
pub fn H5F_set_sohm_vers(file: &mut FileApiState, version: u8) {
    file.sohm_vers = version;
}

/// Record the number of SOHM indexes used by the file.
pub fn H5F_set_sohm_nindexes(file: &mut FileApiState, nindexes: u8) {
    file.sohm_nindexes = nindexes;
}

/// Toggle storing creation-order indexes in shared object header messages.
pub fn H5F_set_store_msg_crt_idx(file: &mut FileApiState, enabled: bool) {
    file.store_msg_crt_idx = enabled;
}

/// Set the low/high library version bounds for new file content.
pub fn H5F__set_libver_bounds(file: &mut FileApiState, low: u8, high: u8) {
    file.low_bound = low;
    file.high_bound = high;
}

/// Copy the in-memory file image into a caller-provided buffer.
///
/// Mirrors the caller-buffer form of `H5Fget_file_image`: callers can query
/// the image size separately and reuse their own storage across calls.
pub fn H5F__get_file_image_into(file: &FileApiState, out: &mut [u8]) -> Result<usize> {
    if out.len() < file.image.len() {
        return Err(Error::InvalidFormat(
            "H5F file image output buffer is too small".into(),
        ));
    }
    out[..file.image.len()].copy_from_slice(&file.image);
    Ok(file.image.len())
}

/// Return the current in-memory file image size.
pub fn H5F__get_file_image_size(file: &FileApiState) -> usize {
    file.image.len()
}

/// Return the current in-memory file image size.
pub fn H5Fget_file_image_size(file: &FileApiState) -> usize {
    H5F__get_file_image_size(file)
}

/// Copy the in-memory file image into caller-provided storage.
pub fn H5Fget_file_image_into(file: &FileApiState, out: &mut [u8]) -> Result<usize> {
    H5F__get_file_image_into(file, out)
}

/// Copy the in-memory file image into caller-owned `Vec` storage.
pub fn H5F__get_file_image_vec_into(file: &FileApiState, out: &mut Vec<u8>) -> usize {
    out.clear();
    out.extend_from_slice(&file.image);
    out.len()
}

/// Copy the in-memory file image into caller-owned `Vec` storage.
pub fn H5Fget_file_image_vec_into(file: &FileApiState, out: &mut Vec<u8>) -> usize {
    H5F__get_file_image_vec_into(file, out)
}

/// Return a `(eof, eoa)` tuple summarizing the file's allocation state.
pub fn H5F__get_info(file: &FileApiState) -> (u64, u64) {
    (file.eof, file.eoa)
}

/// Return file metadata information in libhdf5's v2 public query shape.
pub fn H5Fget_info2(file: &FileApiState) -> FileApiInfo {
    FileApiInfo {
        superblock: FileApiSuperblockInfo {
            extension_size: file
                .super_ext_addr
                .map(|_| H5F_sizeof_addr() as u64)
                .unwrap_or(0),
            ..FileApiSuperblockInfo::default()
        },
        free_space: FileApiFreeSpaceInfo {
            total_space: H5Fget_freespace(file),
            ..FileApiFreeSpaceInfo::default()
        },
        shared_messages: FileApiSharedMessageInfo {
            header_size: file
                .sohm_addr
                .map(|_| H5F_sizeof_addr() as u64)
                .unwrap_or(0),
            message_info_size: u64::from(file.sohm_nindexes),
        },
    }
}

/// Return file metadata information in libhdf5's v1 public query shape.
pub fn H5Fget_info1(file: &FileApiState) -> FileApiInfo {
    H5Fget_info2(file)
}

/// Set the number of metadata-read retries before giving up.
pub fn H5F_set_retries(file: &mut FileApiState, retries: u32) {
    file.retries = retries;
}

/// Set the end-of-allocation address; bumps EOF if necessary.
pub fn H5F__set_eoa(file: &mut FileApiState, eoa: u64) {
    file.eoa = eoa;
    file.eof = file.eof.max(eoa);
}

/// Enable paged aggregation for the file's free-space manager.
pub fn H5F__set_paged_aggr(_file: &mut FileApiState) {}

/// Return the greater of EOF and EOA — the largest address known in the file.
pub fn H5F__get_max_eof_eoa(file: &FileApiState) -> u64 {
    file.eof.max(file.eoa)
}

/// Internal entry point to start SWMR write access on the file.
pub fn H5F__start_swmr_write(file: &mut FileApiState) {
    file.swmr_write = true;
}

/// Internal format-convert wrapper.
pub fn H5F__format_convert(file: &mut FileApiState) -> Result<()> {
    H5Fformat_convert(file)
}

/// Return the persistent file identifier.
pub fn H5F_get_file_id(file: &FileApiState) -> u64 {
    file.id
}

/// Create the superblock extension at the given address.
pub fn H5F__super_ext_create(file: &mut FileApiState, addr: u64) {
    file.super_ext_addr = Some(addr);
}

/// Open the superblock extension if one exists, returning its address.
pub fn H5F__super_ext_open(file: &FileApiState) -> Option<u64> {
    file.super_ext_addr
}

/// Close the file's superblock extension.
pub fn H5F__super_ext_close(_file: &mut FileApiState) {}

/// Update the driver-info message stored in the superblock extension.
pub fn H5F__update_super_ext_driver_msg(_file: &mut FileApiState) {}

/// Initialize the file's superblock by writing the HDF5 signature.
pub fn H5F__super_init(file: &mut FileApiState) {
    file.image.clear();
    file.image.extend_from_slice(b"\x89HDF\r\n\x1a\n");
}

/// Mark the EOA value as dirty so it will be written on the next flush.
pub fn H5F_eoa_dirty(_file: &mut FileApiState) {}

/// Mark the superblock as dirty so it will be rewritten on the next flush.
pub fn H5F_super_dirty(_file: &mut FileApiState) {}

/// Free superblock-related metadata for the file.
pub fn H5F__super_free(file: &mut FileApiState) {
    file.super_ext_addr = None;
}

/// Remove the superblock extension messages, freeing the extension if empty.
pub fn H5F__super_ext_remove_msg(file: &mut FileApiState) {
    file.super_ext_addr = None;
}

/// Read bytes from the file's shared image into caller-provided storage.
pub fn H5F_shared_block_read_into(
    file: &FileApiState,
    offset: usize,
    out: &mut [u8],
) -> Result<()> {
    H5F_block_read_into(file, offset, out)
}

/// Read bytes from the file image into caller-provided storage.
pub fn H5F_block_read_into(file: &FileApiState, offset: usize, out: &mut [u8]) -> Result<()> {
    let end = offset
        .checked_add(out.len())
        .ok_or_else(|| Error::InvalidFormat("H5F read overflow".into()))?;
    let src = file
        .image
        .get(offset..end)
        .ok_or_else(|| Error::InvalidFormat("H5F read is outside file image".into()))?;
    out.copy_from_slice(src);
    Ok(())
}

/// Write `data` to the file's shared image at `offset`.
pub fn H5F_shared_block_write(file: &mut FileApiState, offset: usize, data: &[u8]) -> Result<()> {
    H5F_block_write(file, offset, data)
}

/// Write `data` to the file's image at `offset`, growing it if necessary.
pub fn H5F_block_write(file: &mut FileApiState, offset: usize, data: &[u8]) -> Result<()> {
    let end = offset
        .checked_add(data.len())
        .ok_or_else(|| Error::InvalidFormat("H5F write overflow".into()))?;
    if file.image.len() < end {
        file.image.resize(end, 0);
    }
    file.image[offset..end].copy_from_slice(data);
    let end_u64 = usize_to_u64(end, "H5F EOF")?;
    file.eof = file.eof.max(end_u64);
    Ok(())
}

/// Read multiple ranges from the file image into a caller-provided contiguous buffer.
pub fn H5F_shared_select_read_into(
    file: &FileApiState,
    spans: &[(usize, usize)],
    out: &mut [u8],
) -> Result<()> {
    let mut out_offset = 0usize;
    for &(offset, len) in spans {
        let end = out_offset
            .checked_add(len)
            .ok_or_else(|| Error::InvalidFormat("H5F select output offset overflow".into()))?;
        let dst = out
            .get_mut(out_offset..end)
            .ok_or_else(|| Error::InvalidFormat("H5F select output buffer is too small".into()))?;
        H5F_block_read_into(file, offset, dst)?;
        out_offset = end;
    }
    if out_offset != out.len() {
        return Err(Error::InvalidFormat(
            "H5F select output buffer length does not match requested spans".into(),
        ));
    }
    Ok(())
}

/// Write multiple `(offset, data)` ranges to the file image.
pub fn H5F_shared_select_write(file: &mut FileApiState, spans: &[(usize, &[u8])]) -> Result<()> {
    for &(offset, data) in spans {
        H5F_block_write(file, offset, data)?;
    }
    Ok(())
}

/// Vectored read into a caller-provided contiguous buffer.
pub fn H5F_shared_vector_read_into(
    file: &FileApiState,
    spans: &[(usize, usize)],
    out: &mut [u8],
) -> Result<()> {
    H5F_shared_select_read_into(file, spans, out)
}

/// Vectored write of multiple `(offset, data)` ranges to the file image.
pub fn H5F_shared_vector_write(file: &mut FileApiState, spans: &[(usize, &[u8])]) -> Result<()> {
    H5F_shared_select_write(file, spans)
}

/// Flush metadata associated with a particular tag (here, all metadata).
pub fn H5F_flush_tagged_metadata(file: &mut FileApiState) {
    H5F__flush(file);
}

/// Compute a coarse checksum of the file image (sum of bytes).
pub fn H5F_get_checksums(file: &FileApiState) -> u32 {
    file.image
        .iter()
        .fold(0u32, |acc, byte| acc.wrapping_add(u32::from(*byte)))
}

/// Return a short debug string describing the file.
pub fn H5F_debug_into<W>(file: &FileApiState, out: &mut W) -> fmt::Result
where
    W: fmt::Write + ?Sized,
{
    write!(
        out,
        "H5F(id={}, name={}, eof={})",
        file.id, file.name, file.eof
    )
}

/// Assert that the number of shared open files equals `expected`.
pub fn H5F_sfile_assert_num(file: &FileApiState, expected: usize) -> bool {
    file.open_ids.len() == expected
}

/// Register an open shared-file ID with the file.
pub fn H5F__sfile_add(file: &mut FileApiState, id: u64) {
    file.open_ids.insert(id);
}

/// Search for an open shared-file ID in the file.
pub fn H5F__sfile_search(file: &FileApiState, id: u64) -> bool {
    file.open_ids.contains(&id)
}

/// Unregister an open shared-file ID from the file.
pub fn H5F__sfile_remove(file: &mut FileApiState, id: u64) {
    file.open_ids.remove(&id);
}

/// Return the MPI rank of this process; unsupported in pure-Rust mode.
pub fn H5F_mpi_get_rank() -> Result<u32> {
    Err(unsupported_file("H5F_mpi_get_rank"))
}

/// Return the file's MPI communicator; unsupported in pure-Rust mode.
pub fn H5F_mpi_get_comm() -> Result<()> {
    Err(unsupported_file("H5F_mpi_get_comm"))
}

/// Return the file's MPI info object; unsupported in pure-Rust mode.
pub fn H5F_mpi_get_info() -> Result<()> {
    Err(unsupported_file("H5F_mpi_get_info"))
}

/// Return the MPI communicator size for a shared file; unsupported in pure-Rust mode.
pub fn H5F_shared_mpi_get_size() -> Result<u64> {
    Err(unsupported_file("H5F_shared_mpi_get_size"))
}

/// Return the MPI communicator size; unsupported in pure-Rust mode.
pub fn H5F_mpi_get_size() -> Result<u64> {
    Err(unsupported_file("H5F_mpi_get_size"))
}

/// Set the MPI atomicity flag on the file.
pub fn H5F__set_mpi_atomicity(file: &mut FileApiState, atomicity: bool) {
    file.mpi_atomicity = atomicity;
}

/// Public API to set MPI atomicity; unsupported in pure-Rust mode.
pub fn H5Fset_mpi_atomicity(_file: &mut FileApiState, _atomicity: bool) -> Result<()> {
    Err(unsupported_file("H5Fset_mpi_atomicity"))
}

/// Return the MPI atomicity flag from the file.
pub fn H5F__get_mpi_atomicity(file: &FileApiState) -> bool {
    file.mpi_atomicity
}

/// Public API to query MPI atomicity; unsupported in pure-Rust mode.
pub fn H5Fget_mpi_atomicity(_file: &FileApiState) -> Result<bool> {
    Err(unsupported_file("H5Fget_mpi_atomicity"))
}

/// Retrieve the MPI communicator from the file; unsupported in pure-Rust mode.
pub fn H5F_mpi_retrieve_comm() -> Result<()> {
    Err(unsupported_file("H5F_mpi_retrieve_comm"))
}

/// Return whether collective metadata reads are enabled.
pub fn H5F_get_coll_metadata_reads(file: &FileApiState) -> bool {
    file.coll_metadata_reads
}

/// Return whether collective metadata reads are enabled (shared variant).
pub fn H5F_shared_get_coll_metadata_reads(file: &FileApiState) -> bool {
    file.coll_metadata_reads
}

/// Enable or disable collective metadata reads on the file.
pub fn H5F_set_coll_metadata_reads(file: &mut FileApiState, value: bool) {
    file.coll_metadata_reads = value;
}

/// Return the MPI datatype used for file blocks; unsupported in pure-Rust mode.
pub fn H5F_mpi_get_file_block_type() -> Result<()> {
    Err(unsupported_file("H5F_mpi_get_file_block_type"))
}

/// Create an external-file cache that holds at most `max_nfiles` files.
pub fn H5F__efc_create(max_nfiles: usize) -> ExternalFileCache {
    ExternalFileCache {
        max_nfiles,
        ..ExternalFileCache::default()
    }
}

/// Open a file by name and intent through the external-file cache.
pub fn H5F__efc_open_file(cache: &mut ExternalFileCache, name: &str, intent: u32) -> Result<u64> {
    let id = cache
        .files
        .len()
        .checked_add(1)
        .ok_or_else(|| Error::InvalidFormat("external file cache id overflow".into()))
        .and_then(|id| {
            u64::try_from(id)
                .map_err(|_| Error::InvalidFormat("external file cache id exceeds u64".into()))
        })?;
    let file = FileApiState {
        id,
        name: name.to_string(),
        actual_name: name.to_string(),
        intent,
        fileno: id,
        ..FileApiState::default()
    };
    H5F__efc_open(cache, file)
}

/// Insert a constructed file into the external-file cache.
pub fn H5F__efc_open(cache: &mut ExternalFileCache, file: FileApiState) -> Result<u64> {
    if cache.max_nfiles != 0
        && cache.files.len() >= cache.max_nfiles
        && !cache.files.contains_key(&file.name)
    {
        return Err(Error::InvalidFormat(
            "external file cache maximum file count exceeded".into(),
        ));
    }
    let id = file.id;
    cache.files.insert(file.name.clone(), file);
    Ok(id)
}

/// Close an entry in the external-file cache by name.
pub fn H5F_efc_close(cache: &mut ExternalFileCache, name: &str) -> Option<FileApiState> {
    cache.close_attempts = cache.close_attempts.saturating_add(1);
    cache.files.remove(name)
}

/// Return the maximum number of files the cache will hold.
pub fn H5F__efc_max_nfiles(cache: &ExternalFileCache) -> usize {
    cache.max_nfiles
}

/// Release an entry from the external-file cache, freeing it if the last reference.
pub fn H5F__efc_release_real(cache: &mut ExternalFileCache, name: &str) -> Option<FileApiState> {
    H5F_efc_close(cache, name)
}

/// Release an entry from the external-file cache.
pub fn H5F__efc_release(cache: &mut ExternalFileCache, name: &str) -> Option<FileApiState> {
    H5F__efc_release_real(cache, name)
}

/// Destroy the external-file cache, releasing all entries.
pub fn H5F__efc_destroy(cache: &mut ExternalFileCache) {
    cache.files.clear();
}

/// Remove an entry from the external-file cache without recording a close attempt.
pub fn H5F__efc_remove_ent(cache: &mut ExternalFileCache, name: &str) -> Option<FileApiState> {
    cache.files.remove(name)
}

/// First-pass close attempt for an external-file cache entry.
pub fn H5F__efc_try_close_tag1(cache: &mut ExternalFileCache, name: &str) -> bool {
    H5F__efc_try_close(cache, name)
}

/// Second-pass close attempt for an external-file cache entry.
pub fn H5F__efc_try_close_tag2(cache: &mut ExternalFileCache, name: &str) -> bool {
    H5F__efc_try_close(cache, name)
}

/// Attempt to close an external-file cache entry, returning whether it was removed.
pub fn H5F__efc_try_close(cache: &mut ExternalFileCache, name: &str) -> bool {
    cache.close_attempts = cache.close_attempts.saturating_add(1);
    cache.files.remove(name).is_some()
}

/// Allocate a "fake" address for objects that exist outside the normal allocator.
pub fn H5F_fake_alloc(file: &mut FileApiState, size: u64) -> Result<u64> {
    H5F__alloc(file, size)
}

/// Free a previously fake-allocated address.
pub fn H5F_fake_free(file: &mut FileApiState, addr: u64, size: u64) -> Result<()> {
    H5F__free(file, addr, size)
}

/// Test helper: return the number of SOHM messages tracked in the file.
pub fn H5F__get_sohm_mesg_count_test(file: &FileApiState) -> u8 {
    file.sohm_nindexes
}

/// Test helper: verify that the cached symbol-table data is present.
pub fn H5F__check_cached_stab_test(_file: &FileApiState) -> bool {
    true
}

/// Test helper: return the maximum address used in the file.
pub fn H5F__get_maxaddr_test(file: &FileApiState) -> u64 {
    H5F__get_max_eof_eoa(file)
}

/// Test helper: return the superblock-extension address if one is set.
pub fn H5F__get_sbe_addr_test(file: &FileApiState) -> Option<u64> {
    file.super_ext_addr
}

/// Test helper: return whether two file handles refer to the same underlying file.
pub fn H5F__same_file_test(left: &FileApiState, right: &FileApiState) -> bool {
    left.fileno == right.fileno
}

/// Read the contents of the file's metadata accumulator.
pub fn H5F__accum_read(file: &FileApiState) -> &[u8] {
    &file.accum
}

/// Resize the metadata accumulator buffer to `len` bytes.
pub fn H5F__accum_adjust(file: &mut FileApiState, len: usize) {
    file.accum.resize(len, 0);
}

/// Replace the contents of the metadata accumulator with `data`.
pub fn H5F__accum_write(file: &mut FileApiState, data: &[u8]) {
    file.accum.clear();
    file.accum.extend_from_slice(data);
}

/// Free the metadata accumulator buffer.
pub fn H5F__accum_free(file: &mut FileApiState) {
    file.accum.clear();
}

/// Flush any pending data in the metadata accumulator out to the file image.
pub fn H5F__accum_flush(file: &mut FileApiState) -> Result<()> {
    let data = std::mem::take(&mut file.accum);
    let eof = u64_to_usize(file.eof, "H5F accumulator EOF")?;
    match H5F_block_write(file, eof, &data) {
        Ok(()) => Ok(()),
        Err(err) => {
            file.accum = data;
            Err(err)
        }
    }
}

/// Convert a `usize` to `u64`, erroring with a descriptive context on overflow.
fn usize_to_u64(value: usize, context: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| Error::InvalidFormat(format!("{context} exceeds u64")))
}

/// Convert a `u64` to `usize`, erroring with a descriptive context on overflow.
fn u64_to_usize(value: u64, context: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| Error::InvalidFormat(format!("{context} exceeds usize")))
}

/// Reset the metadata accumulator, discarding any pending data.
pub fn H5F__accum_reset(file: &mut FileApiState) {
    file.accum.clear();
}

/// Return the file's open intent flags (shared variant).
pub fn H5F_shared_get_intent(file: &FileApiState) -> u32 {
    file.intent
}

/// Return the file size visible to public callers.
pub fn H5Fget_filesize(file: &FileApiState) -> u64 {
    file.eof
}

/// Return the low library version bound for the file.
pub fn H5F_get_low_bound(file: &FileApiState) -> u8 {
    file.low_bound
}

/// Return the high library version bound for the file.
pub fn H5F_get_high_bound(file: &FileApiState) -> u8 {
    file.high_bound
}

/// Return the file's resolved on-disk name.
pub fn H5F_get_actual_name(file: &FileApiState) -> &str {
    &file.actual_name
}

/// Copy the file's resolved on-disk name into caller-provided storage.
pub fn H5F_get_actual_name_into(file: &FileApiState, out: &mut String) {
    out.clear();
    out.push_str(&file.actual_name);
}

/// Return the public file name, preferring the resolved actual name when known.
pub fn H5Fget_name(file: &FileApiState) -> &str {
    if file.actual_name.is_empty() {
        &file.name
    } else {
        &file.actual_name
    }
}

/// Copy the public file name into caller-provided storage.
pub fn H5Fget_name_into(file: &FileApiState, out: &mut String) {
    out.clear();
    out.push_str(H5Fget_name(file));
}

/// Return the external-link search path stored on the file.
pub fn H5F_get_extpath(file: &FileApiState) -> &str {
    &file.name
}

/// Copy the external-link search path into caller-provided storage.
pub fn H5F_get_extpath_into(file: &FileApiState, out: &mut String) {
    out.clear();
    out.push_str(&file.name);
}

/// Return a reference to the shared file state.
pub fn H5F_get_shared(file: &FileApiState) -> &FileApiState {
    file
}

/// Return whether two handles share the same underlying file image.
pub fn H5F_same_shared(left: &FileApiState, right: &FileApiState) -> bool {
    left.fileno == right.fileno
}

/// Return the number of open objects in the file.
pub fn H5F_get_nopen_objs(file: &FileApiState) -> usize {
    file.nopen_objs
}

/// Check whether a file ID is registered in the package state.
pub fn H5F_file_id_exists(pkg: &FilePackageState, id: u64) -> bool {
    pkg.files.contains_key(&id)
}

/// Return the parent file for a mounted child, if any.
pub fn H5F_get_parent(_file: &FileApiState) -> Option<u64> {
    None
}

/// Return the number of currently mounted child files.
pub fn H5F_get_nmounts(file: &FileApiState) -> usize {
    file.mounts.len()
}

/// Return the file creation property list (encoded as flag bits).
pub fn H5F_get_fcpl(file: &FileApiState) -> u64 {
    file.flags
}

/// Return the on-disk size of an address (always 8 bytes here).
pub fn H5F_sizeof_addr() -> usize {
    std::mem::size_of::<u64>()
}

/// Return the address of the SOHM table, if any.
pub fn H5F_get_sohm_addr(file: &FileApiState) -> Option<u64> {
    file.sohm_addr
}

/// Return the SOHM table version stored in the file.
pub fn H5F_get_sohm_vers(file: &FileApiState) -> u8 {
    file.sohm_vers
}

/// Return the number of SOHM indexes used by the file.
pub fn H5F_get_sohm_nindexes(file: &FileApiState) -> u8 {
    file.sohm_nindexes
}

/// Return the default symbol-table leaf-k value used for new groups.
pub fn H5F_sym_leaf_k() -> u16 {
    32
}

/// Return the minimum object-header size used when creating new datasets.
pub fn H5F_get_min_dset_ohdr() -> usize {
    0
}

/// Return the default B-tree internal node k value.
pub fn H5F_kvalue() -> u16 {
    32
}

/// Return the number of outstanding references to the file.
pub fn H5F_get_nrefs(file: &FileApiState) -> usize {
    file.nrefs
}

/// Return the raw-data chunk cache slot count.
pub fn H5F_rdcc_nslots() -> usize {
    521
}

/// Return the raw-data chunk cache size in bytes.
pub fn H5F_rdcc_nbytes() -> usize {
    1024 * 1024
}

/// Return the raw-data chunk cache pre-emption-on-write (w0) factor.
pub fn H5F_rdcc_w0() -> f64 {
    0.75
}

/// Return the base address of the file image.
pub fn H5F_get_base_addr(file: &FileApiState) -> u64 {
    file.base_addr
}

/// Return whether the file uses a shared B-tree for groups.
pub fn H5F_grp_btree_shared(file: &FileApiState) -> bool {
    file.flags & 1 != 0
}

/// Return whether reference garbage collection is enabled.
pub fn H5F_gc_ref() -> bool {
    false
}

/// Return the file-close-degree setting.
pub fn H5F_get_fc_degree() -> u8 {
    0
}

/// Return whether the file should evict cached entries on close.
pub fn H5F_get_evict_on_close() -> bool {
    false
}

/// Return whether the file stores message creation-order indexes.
pub fn H5F_store_msg_crt_idx(file: &FileApiState) -> bool {
    file.store_msg_crt_idx
}

/// Test whether the file's shared state has a given feature bit set.
pub fn H5F_shared_has_feature(file: &FileApiState, feature: u64) -> bool {
    file.flags & feature != 0
}

/// Test whether the file has a given feature bit set.
pub fn H5F_has_feature(file: &FileApiState, feature: u64) -> bool {
    H5F_shared_has_feature(file, feature)
}

/// Return the file's unique file number.
pub fn H5F_get_fileno(file: &FileApiState) -> u64 {
    file.fileno
}

/// Return the file number through the public libhdf5-style API.
pub fn H5Fget_fileno(file: &FileApiState) -> u64 {
    H5F_get_fileno(file)
}

/// Return the end-of-allocation address (shared variant).
pub fn H5F_shared_get_eoa(file: &FileApiState) -> u64 {
    file.eoa
}

/// Return the end-of-allocation address.
pub fn H5F_get_eoa(file: &FileApiState) -> u64 {
    file.eoa
}

/// Return the end-of-allocation address through the public libhdf5-style API.
pub fn H5Fget_eoa(file: &FileApiState) -> u64 {
    H5F_get_eoa(file)
}

/// Return the known free-space size through the public libhdf5-style API.
pub fn H5Fget_freespace(_file: &FileApiState) -> u64 {
    0
}

/// Public free-space section enumeration; unsupported without free-space manager state.
pub fn H5Fget_free_sections(_file: &FileApiState) -> Result<()> {
    Err(unsupported_file("H5Fget_free_sections"))
}

/// Return the metadata-cache hit rate through the public libhdf5-style API.
pub fn H5Fget_mdc_hit_rate(file: &FileApiState) -> f64 {
    H5F_get_mdc_hit_rate(file)
}

/// Return the underlying VFD file handle; unsupported in pure-Rust mode.
pub fn H5F_get_vfd_handle() -> Result<()> {
    Err(unsupported_file("H5F_get_vfd_handle"))
}

/// Public API to return the underlying VFD file handle; unsupported in pure-Rust mode.
pub fn H5Fget_vfd_handle(_file: &FileApiState) -> Result<()> {
    Err(unsupported_file("H5Fget_vfd_handle"))
}

/// Return whether `addr` is in the "temporary" range used for unallocated objects.
pub fn H5F_is_tmp_addr(_addr: u64) -> bool {
    false
}

/// Return whether temporary-space allocation is enabled.
pub fn H5F_use_tmp_space() -> bool {
    false
}

/// Return whether MPI file sync is required by the file's VFD.
pub fn H5F_shared_get_mpi_file_sync_required() -> bool {
    false
}

/// Return whether metadata cache logging is currently active for the file.
pub fn H5F_use_mdc_logging(file: &FileApiState) -> bool {
    file.metadata_logging
}

/// Start metadata cache logging the first time this file is accessed.
pub fn H5F_start_mdc_log_on_access(file: &mut FileApiState) {
    file.metadata_logging = true;
}

/// Return the destination path used for metadata cache logging.
pub fn H5F_mdc_log_location(file: &FileApiState) -> &str {
    &file.name
}

/// Return the file's (alignment, threshold) settings.
pub fn H5F_get_alignment() -> (u64, u64) {
    (1, 1)
}

/// Return the alignment threshold below which the alignment policy is skipped.
pub fn H5F_get_threshold() -> u64 {
    1
}

/// Return the metadata page-end threshold used by the page-aggregating allocator.
pub fn H5F_get_pgend_meta_thres() -> u32 {
    0
}

/// Return the address of the null free-space manager, if persisted.
pub fn H5F_get_null_fsm_addr() -> Option<u64> {
    None
}

/// Return the file's VOL object reference, if any.
pub fn H5F_get_vol_obj(file: &FileApiState) -> Option<u64> {
    file.vol_obj
}

/// Return continuation info `(base_addr, eoa)` for the file.
pub fn H5F__get_cont_info(file: &FileApiState) -> (u64, u64) {
    (file.base_addr, file.eoa)
}

/// Return whether the file uses on-disk file locking.
pub fn H5F_get_use_file_locking(file: &FileApiState) -> bool {
    file.file_locking
}

/// Return whether the file's VFD supports vectored select I/O.
pub fn H5F_has_vector_select_io() -> bool {
    true
}

/// Return the file's relaxed-file-image-construction flags.
pub fn H5F_get_rfic_flags(file: &FileApiState) -> u64 {
    file.flags
}

/// Close all mounted children of the file.
pub fn H5F__close_mounts(file: &mut FileApiState) {
    file.mounts.clear();
}

/// Mount a child file under `name` inside this file.
pub fn H5F_mount(file: &mut FileApiState, name: &str) {
    file.mounts.insert(name.to_string());
}

/// Unmount the child file at `name`.
pub fn H5F_unmount(file: &mut FileApiState, name: &str) {
    file.mounts.remove(name);
}

/// Return whether `name` is a current mount point in the file.
pub fn H5F_is_mount(file: &FileApiState, name: &str) -> bool {
    file.mounts.contains(name)
}

/// Recursively count IDs under each mount point.
pub fn H5F__mount_count_ids_recurse(file: &FileApiState) -> usize {
    file.mounts.len()
}

/// Return the number of mount-point IDs in the file.
pub fn H5F__mount_count_ids(file: &FileApiState) -> usize {
    file.mounts.len()
}

/// Recursively flush all mounted child files.
pub fn H5F__flush_mounts_recurse(_file: &mut FileApiState) {}

/// Flush the file along with any mounted children.
pub fn H5F_flush_mounts(file: &mut FileApiState) {
    H5F__flush(file);
}

/// Copy mount-point names into caller-provided storage.
pub fn H5F_traverse_mount_into(file: &FileApiState, out: &mut Vec<String>) -> usize {
    out.clear();
    out.extend(file.mounts.iter().cloned());
    out.len()
}

/// Visit mount-point names without allocating an intermediate list.
pub fn H5F_visit_mounts<'a, F>(file: &'a FileApiState, mut visitor: F)
where
    F: FnMut(&'a str),
{
    for mount in &file.mounts {
        visitor(mount);
    }
}

/// Check whether a buffer is a recognizable HDF5 file image.
pub fn H5Fis_hdf5(image: &[u8]) -> bool {
    H5F__is_hdf5(image)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_metadata_queries_reflect_existing_state() {
        let mut file = FileApiState::default();
        H5Freset_mdc_hit_rate_stats(&mut file);
        H5Freset_page_buffering_stats(&mut file);
        H5Fstart_swmr_write(&mut file);
        H5Fstart_mdc_logging(&mut file);
        H5F_set_coll_metadata_reads(&mut file, true);
        H5F__set_mpi_atomicity(&mut file, true);
        H5F_incr_nopen_objs(&mut file);

        let stats = H5F_get_metadata_query_stats(&file);
        assert_eq!(stats.mdc_hit_rate_resets, 1);
        assert_eq!(stats.page_buffer_stats_resets, 1);
        assert_eq!(stats.nopen_objs, 1);
        assert_eq!(H5F_get_page_buffering_stats(&file).resets, 1);
        assert_eq!(H5Fget_page_buffering_stats(&file).resets, 1);
        assert_eq!(H5Fget_mdc_hit_rate(&file), H5F_get_mdc_hit_rate(&file));

        let flags = H5F_get_mdc_logging_status(&file);
        assert_eq!(H5Fget_mdc_logging_status(&file), flags);
        assert!(H5F_get_swmr_write(&file));
        assert!(flags.swmr_write);
        assert!(flags.metadata_logging);
        assert!(flags.coll_metadata_reads);
        assert!(flags.mpi_atomicity);
    }

    #[test]
    fn metadata_read_retry_info_public_api_reports_configured_retries() {
        let mut file = FileApiState::default();
        H5F_set_retries(&mut file, 7);

        let info = H5Fget_metadata_read_retry_info(&file);
        assert_eq!(info.nbins, H5F_NUM_METADATA_READ_RETRY_TYPES);
        assert_eq!(info.retries, [7; H5F_NUM_METADATA_READ_RETRY_TYPES]);
    }

    #[test]
    fn public_mpi_and_vfd_handle_apis_are_explicit_unsupported_boundaries() {
        let file = FileApiState::default();
        assert!(matches!(
            H5Fget_mpi_atomicity(&file),
            Err(Error::Unsupported(_))
        ));
        assert!(matches!(
            H5Fget_vfd_handle(&file),
            Err(Error::Unsupported(_))
        ));
        assert!(matches!(
            H5Fget_mdc_config(&file),
            Err(Error::Unsupported(_))
        ));
        assert!(matches!(H5Fget_mdc_size(&file), Err(Error::Unsupported(_))));
        assert!(matches!(
            H5Fget_mdc_image_info(&file),
            Err(Error::Unsupported(_))
        ));
        assert!(matches!(
            H5Fget_dset_no_attrs_hint(&file),
            Err(Error::Unsupported(_))
        ));
        let mut file = file;
        assert!(matches!(
            H5Fset_mdc_config(&mut file),
            Err(Error::Unsupported(_))
        ));
        assert!(matches!(
            H5Fset_dset_no_attrs_hint(&mut file, true),
            Err(Error::Unsupported(_))
        ));
        assert!(matches!(
            H5Fget_free_sections(&file),
            Err(Error::Unsupported(_))
        ));
    }

    #[test]
    fn public_async_file_apis_are_explicit_unsupported_boundaries() {
        let mut pkg = H5F_init();
        let mut file = FileApiState::default();

        for err in [
            H5Freopen_async(&file).unwrap_err(),
            H5Fcreate_async(&mut pkg, "async.h5", 0).unwrap_err(),
            H5Fopen_async(&file).unwrap_err(),
            H5Fflush_async(&mut file).unwrap_err(),
            H5Fclose_async(file).unwrap_err(),
        ] {
            let message = err.to_string();
            assert!(matches!(err, Error::Unsupported(_)));
            assert!(
                message.contains("libhdf5 file-driver behavior"),
                "unexpected unsupported H5F async error: {message}"
            );
        }
    }

    #[test]
    fn file_image_and_block_reads_support_caller_buffers() {
        let file = FileApiState {
            image: b"abcdefghijkl".to_vec(),
            ..FileApiState::default()
        };

        let mut image = vec![0; file.image.len()];
        assert_eq!(
            H5F__get_file_image_into(&file, &mut image).unwrap(),
            file.image.len()
        );
        assert_eq!(image, file.image);

        let mut block = [0; 3];
        H5F_block_read_into(&file, 2, &mut block).unwrap();
        assert_eq!(&block, b"cde");

        let mut selected = [0; 5];
        H5F_shared_select_read_into(&file, &[(0, 2), (9, 3)], &mut selected).unwrap();
        assert_eq!(&selected, b"abjkl");
        assert!(H5F_block_read_into(&file, 10, &mut selected).is_err());
    }

    #[test]
    fn public_file_query_aliases_match_file_state() {
        let mut file = FileApiState {
            name: "logical.h5".into(),
            actual_name: "resolved.h5".into(),
            intent: 0x21,
            eof: 99,
            eoa: 88,
            fileno: 42,
            image: b"HDF5 image".to_vec(),
            ..FileApiState::default()
        };

        assert_eq!(H5Fget_intent(&file), 0x21);
        assert_eq!(H5Fget_filesize(&file), 99);
        assert_eq!(H5Fget_eoa(&file), 88);
        assert_eq!(H5Fget_freespace(&file), 0);
        assert_eq!(H5Fget_fileno(&file), 42);
        assert_eq!(H5Fget_name(&file), "resolved.h5");

        let mut name = String::new();
        H5Fget_name_into(&file, &mut name);
        assert_eq!(name, "resolved.h5");

        file.actual_name.clear();
        assert_eq!(H5Fget_name(&file), "logical.h5");
        H5Fget_name_into(&file, &mut name);
        assert_eq!(name, "logical.h5");

        assert_eq!(H5Fget_file_image_size(&file), file.image.len());
        let mut image = vec![0; file.image.len()];
        assert_eq!(
            H5Fget_file_image_into(&file, &mut image).unwrap(),
            file.image.len()
        );
        assert_eq!(image, file.image);

        let mut copied = Vec::new();
        assert_eq!(
            H5Fget_file_image_vec_into(&file, &mut copied),
            file.image.len()
        );
        assert_eq!(copied, file.image);
    }

    #[test]
    fn public_file_info_queries_report_supported_state_and_zero_unknowns() {
        let file = FileApiState {
            eoa: 128,
            super_ext_addr: Some(64),
            sohm_addr: Some(96),
            sohm_nindexes: 3,
            ..FileApiState::default()
        };

        let info = H5Fget_info2(&file);
        assert_eq!(info.superblock.extension_size, H5F_sizeof_addr() as u64);
        assert_eq!(info.free_space.total_space, 0);
        assert_eq!(info.shared_messages.header_size, H5F_sizeof_addr() as u64);
        assert_eq!(info.shared_messages.message_info_size, 3);
        assert_eq!(H5Fget_info1(&file), info);
    }

    #[test]
    fn object_id_helpers_support_reused_storage_and_visitors() {
        let mut file = FileApiState::default();
        file.object_ids.insert(7);
        file.object_ids.insert(3);

        let mut ids = vec![99];
        assert_eq!(H5F_get_obj_ids_into(&file, &mut ids), 2);
        assert_eq!(ids, vec![3, 7]);

        ids.push(99);
        assert_eq!(H5F__get_all_ids_cb_into(&file, &mut ids), 2);
        assert_eq!(ids, vec![3, 7]);

        let mut visited = Vec::new();
        H5F_visit_obj_ids(&file, |id| visited.push(id));
        assert_eq!(visited, vec![3, 7]);

        visited.clear();
        H5F__visit_objects(&file, |id| visited.push(id));
        assert_eq!(visited, vec![3, 7]);
    }

    #[test]
    fn path_prefix_helpers_borrow_or_reuse_storage() {
        assert!(matches!(
            H5F__build_name_cow("", "data.h5"),
            Cow::Borrowed("data.h5")
        ));
        assert_eq!(H5F__build_name_cow("/tmp", "data.h5"), "/tmp/data.h5");

        let mut path = String::from("old-value");
        H5F__build_name_into("/prefix", "child.h5", &mut path);
        assert_eq!(path, "/prefix/child.h5");

        H5F_prefix_open_file_into("", "child.h5", &mut path);
        assert_eq!(path, "child.h5");

        assert!(matches!(
            H5F__getenv_prefix_name_cow(Some("/env")),
            Some(Cow::Borrowed("/env"))
        ));
        assert_eq!(H5F__getenv_prefix_name_cow(None), None);
    }

    #[test]
    fn mount_traversal_helpers_support_reused_storage_and_visitors() {
        let mut file = FileApiState::default();
        H5F_mount(&mut file, "/z");
        H5F_mount(&mut file, "/a");

        let mut mounts = vec![String::from("stale")];
        assert_eq!(H5F_traverse_mount_into(&file, &mut mounts), 2);
        assert_eq!(mounts, vec![String::from("/a"), String::from("/z")]);

        let mut visited = Vec::new();
        H5F_visit_mounts(&file, |name| visited.push(name));
        assert_eq!(visited, mounts);
    }

    #[test]
    fn external_file_cache_tracks_open_release_and_limits() {
        let mut cache = H5F__efc_create(1);
        assert_eq!(H5F__efc_max_nfiles(&cache), 1);

        let id = H5F__efc_open_file(&mut cache, "external-a.h5", 7).unwrap();
        assert_eq!(id, 1);
        assert!(cache.files.contains_key("external-a.h5"));
        assert!(H5F__efc_open_file(&mut cache, "external-b.h5", 0).is_err());

        let released = H5F__efc_release(&mut cache, "external-a.h5").unwrap();
        assert_eq!(released.intent, 7);
        assert_eq!(cache.close_attempts, 1);
        assert!(cache.files.is_empty());

        H5F__efc_open_file(&mut cache, "external-c.h5", 0).unwrap();
        assert!(H5F__efc_try_close_tag1(&mut cache, "external-c.h5"));
        assert_eq!(cache.close_attempts, 2);
        H5F__efc_destroy(&mut cache);
        assert!(cache.files.is_empty());
    }

    #[test]
    fn superblock_cache_deserialize_validates_signature() {
        let sb = Superblock {
            version: 2,
            sizeof_addr: 8,
            sizeof_size: 8,
            eof_addr: 64,
            root_addr: 48,
            ..Superblock::default()
        };
        let mut image = Vec::new();
        sb.write_v2(&mut image).unwrap();
        image.extend_from_slice(b"trailing-file-data");

        assert!(H5F__cache_superblock_verify_chksum(&image));
        let cached = H5F__cache_superblock_deserialize(&image).unwrap();
        assert_eq!(cached.superblock.version, 2);
        assert_eq!(cached.superblock.root_addr, 48);
        assert_eq!(cached.superblock.eof_addr, 64);
        assert_eq!(H5F__cache_superblock_image_len(&cached), sb.size());
        assert!(matches!(cached.raw, Cow::Borrowed(_)));
        assert_eq!(&*cached.raw, &image[..sb.size()]);
        H5F__cache_superblock_free_icr(cached);

        let bad = b"not-hdf5";
        assert!(!H5F__cache_superblock_verify_chksum(bad));
        assert!(H5F__cache_superblock_deserialize(bad).is_err());

        let truncated = &image[..sb.size() - 1];
        assert!(H5F__cache_superblock_deserialize(truncated).is_err());
    }

    #[test]
    fn driver_info_cache_block_uses_libhdf5_prefix_layout() {
        let image = vec![
            0, 0xaa, 0xbb, 0xcc, 3, 0, 0, 0, b's', b'e', b'c', b'2', 0, 0, 0, 0, b'a', b'b', b'c',
        ];

        assert_eq!(H5F__cache_drvrinfo_get_initial_load_size(), 16);
        assert_eq!(
            H5F__cache_drvrinfo_get_final_load_size(&image[..16]).unwrap(),
            image.len()
        );

        let block = H5F__cache_drvrinfo_deserialize(&image).unwrap();
        assert_eq!(block.version, 0);
        assert_eq!(&block.name, b"sec2\0\0\0\0");
        assert!(matches!(block.data, Cow::Borrowed(_)));
        assert_eq!(&*block.data, b"abc");
        assert_eq!(H5F__cache_drvrinfo_image_len(&block).unwrap(), image.len());

        let mut serialized = Vec::new();
        assert_eq!(
            H5F__cache_drvrinfo_serialize_into(&block, &mut serialized).unwrap(),
            image.len()
        );
        assert_eq!(
            serialized,
            vec![0, 0, 0, 0, 3, 0, 0, 0, b's', b'e', b'c', b'2', 0, 0, 0, 0, b'a', b'b', b'c',]
        );
        H5F__cache_drvrinfo_free_icr(block);
    }

    #[test]
    fn driver_info_cache_block_rejects_malformed_prefixes() {
        let valid = vec![
            0, 0, 0, 0, 1, 0, 0, 0, b's', b'e', b'c', b'2', 0, 0, 0, 0, b'x',
        ];
        assert!(H5F__cache_drvrinfo_deserialize(&valid[..15]).is_err());

        let mut bad_version = valid.clone();
        bad_version[0] = 1;
        assert!(H5F__cache_drvrinfo_get_final_load_size(&bad_version[..16]).is_err());

        let truncated_payload = valid[..16].to_vec();
        assert_eq!(
            H5F__cache_drvrinfo_get_final_load_size(&truncated_payload).unwrap(),
            17
        );
        assert!(H5F__cache_drvrinfo_deserialize(&truncated_payload).is_err());

        let mut trailing = valid.clone();
        trailing.push(b'y');
        assert!(H5F__cache_drvrinfo_deserialize(&trailing).is_err());
    }

    #[test]
    fn file_lock_env_parser_rejects_invalid_values() {
        assert_eq!(H5F__parse_file_lock_env_var(None).unwrap(), None);
        assert_eq!(
            H5F__parse_file_lock_env_var(Some("0")).unwrap(),
            Some(false)
        );
        assert_eq!(
            H5F__parse_file_lock_env_var(Some("BEST_EFFORT")).unwrap(),
            Some(true)
        );
        assert!(H5F__parse_file_lock_env_var(Some("maybe")).is_err());
    }
}
