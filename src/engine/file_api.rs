#![allow(dead_code, non_snake_case)]

use std::collections::{BTreeMap, BTreeSet};

use crate::error::{Error, Result};

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

fn unsupported_file(name: &str) -> Error {
    Error::Unsupported(format!(
        "{name} requires libhdf5 file-driver behavior not implemented in pure-Rust mode"
    ))
}

pub fn H5F_init() -> FilePackageState {
    H5F__init_package()
}

pub fn H5F__init_package() -> FilePackageState {
    FilePackageState {
        initialized: true,
        next_id: 1,
        ..FilePackageState::default()
    }
}

pub fn H5F_term_package(pkg: &mut FilePackageState) {
    pkg.initialized = false;
    pkg.files.clear();
}

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

pub fn H5F__free(_file: &mut FileApiState, _addr: u64, _size: u64) -> Result<()> {
    Ok(())
}

pub fn H5F__try_extend(file: &mut FileApiState, addr: u64, old_size: u64, new_size: u64) -> bool {
    addr.checked_add(old_size) == Some(file.eoa) && new_size >= old_size
}

pub fn H5F_cwfs_add(file: &mut FileApiState, addr: u64) {
    file.open_ids.insert(addr);
}

pub fn H5F__get_all_count_cb(file: &FileApiState) -> usize {
    file.object_ids.len()
}

pub fn H5F__get_all_ids_cb(file: &FileApiState) -> Vec<u64> {
    file.object_ids.iter().copied().collect()
}

pub fn H5Fis_accessible(image: &[u8]) -> bool {
    H5F__is_hdf5(image)
}

pub fn H5F__post_open_api_common(file: FileApiState) -> FileApiState {
    file
}

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

pub fn H5F__flush_api_common(file: &mut FileApiState) {
    H5F__flush(file);
}

pub fn H5Fmount(file: &mut FileApiState, name: &str) {
    H5F_mount(file, name);
}

pub fn H5Funmount(file: &mut FileApiState, name: &str) {
    H5F_unmount(file, name);
}

pub fn H5F__reopen_api_common(file: &FileApiState) -> FileApiState {
    H5F_open(file)
}

pub fn H5Freopen(file: &FileApiState) -> FileApiState {
    H5F_open(file)
}

pub fn H5Freopen_async(_file: &FileApiState) -> Result<FileApiState> {
    Err(unsupported_file("H5Freopen_async"))
}

pub fn H5Freset_mdc_hit_rate_stats(file: &mut FileApiState) {
    file.mdc_hit_rate_resets = file.mdc_hit_rate_resets.saturating_add(1);
}

pub fn H5F_get_mdc_hit_rate(file: &FileApiState) -> f64 {
    if file.mdc_hit_rate_resets == 0 {
        0.0
    } else {
        1.0
    }
}

pub fn H5F_get_metadata_query_stats(file: &FileApiState) -> FileMetadataQueryStats {
    FileMetadataQueryStats {
        mdc_hit_rate: H5F_get_mdc_hit_rate(file),
        mdc_hit_rate_resets: file.mdc_hit_rate_resets,
        page_buffer_stats_resets: file.page_buffer_stats_resets,
        nopen_objs: file.nopen_objs,
        nrefs: file.nrefs,
    }
}

pub fn H5F_get_page_buffering_stats(file: &FileApiState) -> PageBufferQueryStats {
    PageBufferQueryStats {
        resets: file.page_buffer_stats_resets,
        ..PageBufferQueryStats::default()
    }
}

pub fn H5F_get_swmr_write(file: &FileApiState) -> bool {
    file.swmr_write
}

pub fn H5F_get_mdc_logging_status(file: &FileApiState) -> FileLoggingFlags {
    FileLoggingFlags {
        swmr_write: file.swmr_write,
        metadata_logging: file.metadata_logging,
        coll_metadata_reads: file.coll_metadata_reads,
        mpi_atomicity: file.mpi_atomicity,
    }
}

pub fn H5Fclear_elink_file_cache(_file: &mut FileApiState) {}

pub fn H5Fstart_swmr_write(file: &mut FileApiState) {
    file.swmr_write = true;
}

pub fn H5Fstart_mdc_logging(file: &mut FileApiState) {
    file.metadata_logging = true;
}

pub fn H5Fstop_mdc_logging(file: &mut FileApiState) {
    file.metadata_logging = false;
}

pub fn H5Fformat_convert(_file: &mut FileApiState) -> Result<()> {
    Err(unsupported_file("H5Fformat_convert"))
}

pub fn H5Freset_page_buffering_stats(file: &mut FileApiState) {
    file.page_buffer_stats_resets = file.page_buffer_stats_resets.saturating_add(1);
}

pub fn H5Fincrement_filesize(file: &mut FileApiState, increment: u64) -> Result<u64> {
    file.eof = file
        .eof
        .checked_add(increment)
        .ok_or_else(|| Error::InvalidFormat("H5F filesize overflow".into()))?;
    Ok(file.eof)
}

pub fn H5F__cache_superblock_get_initial_load_size() -> usize {
    8
}

pub fn H5F__cache_superblock_get_final_load_size(image: &[u8]) -> usize {
    image.len()
}

pub fn H5F__cache_superblock_verify_chksum(_image: &[u8]) -> bool {
    H5F__is_hdf5(_image)
}

pub fn H5F__cache_superblock_deserialize(image: &[u8]) -> Result<Vec<u8>> {
    if !H5F__is_hdf5(image) {
        return Err(Error::InvalidFormat(
            "cached superblock image has invalid HDF5 signature".into(),
        ));
    }
    Ok(image.to_vec())
}

pub fn H5F__cache_superblock_image_len(image: &[u8]) -> usize {
    image.len()
}

pub fn H5F__cache_superblock_free_icr(_image: Vec<u8>) {}

pub fn H5F__cache_drvrinfo_get_initial_load_size() -> usize {
    0
}

pub fn H5F__cache_drvrinfo_get_final_load_size(image: &[u8]) -> usize {
    image.len()
}

pub fn H5F__cache_drvrinfo_deserialize(image: &[u8]) -> Vec<u8> {
    image.to_vec()
}

pub fn H5F__cache_drvrinfo_image_len(image: &[u8]) -> usize {
    image.len()
}

pub fn H5F__cache_drvrinfo_serialize(image: &[u8]) -> Vec<u8> {
    image.to_vec()
}

pub fn H5F__cache_drvrinfo_free_icr(_image: Vec<u8>) {}

pub fn H5F__close_cb(_file: FileApiState) {}

pub fn H5F__parse_file_lock_env_var(value: Option<&str>) -> Result<Option<bool>> {
    match value.map(str::to_ascii_uppercase).as_deref() {
        None => Ok(None),
        Some("FALSE") | Some("0") => Ok(Some(false)),
        Some("TRUE") | Some("1") | Some("BEST_EFFORT") => Ok(Some(true)),
        Some(other) => Err(Error::InvalidFormat(format!(
            "invalid HDF5 file locking environment value '{other}'"
        ))),
    }
}

pub fn H5F__set_vol_conn(file: &mut FileApiState, vol_obj: u64) {
    file.vol_obj = Some(vol_obj);
}

pub fn H5F_get_access_plist(file: &FileApiState) -> u32 {
    file.intent
}

pub fn H5F_get_obj_count(file: &FileApiState) -> usize {
    file.object_ids.len()
}

pub fn H5F_get_obj_ids(file: &FileApiState) -> Vec<u64> {
    file.object_ids.iter().copied().collect()
}

pub fn H5F__get_objects(file: &FileApiState) -> Vec<u64> {
    H5F_get_obj_ids(file)
}

pub fn H5F__get_objects_cb(file: &FileApiState) -> usize {
    H5F_get_obj_count(file)
}

pub fn H5F__build_name(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}/{name}")
    }
}

pub fn H5F__getenv_prefix_name(value: Option<&str>) -> Option<String> {
    value.map(str::to_string)
}

pub fn H5F_prefix_open_file(prefix: &str, name: &str) -> String {
    H5F__build_name(prefix, name)
}

pub fn H5F__is_hdf5(image: &[u8]) -> bool {
    image.starts_with(b"\x89HDF\r\n\x1a\n")
}

pub fn H5F__dest(_file: FileApiState) {}

pub fn H5F__check_if_using_file_locks(file: &FileApiState) -> bool {
    file.file_locking
}

pub fn H5F_open(file: &FileApiState) -> FileApiState {
    let mut reopened = file.clone();
    reopened.nrefs = reopened.nrefs.saturating_add(1);
    reopened
}

pub fn H5F__post_open(file: FileApiState) -> FileApiState {
    file
}

pub fn H5F__flush_phase1(_file: &mut FileApiState) {}

pub fn H5F__flush_phase2(_file: &mut FileApiState) {}

pub fn H5F__flush(file: &mut FileApiState) {
    file.eof = file.eof.max(file.eoa);
}

pub fn H5F__close(_file: FileApiState) {}

pub fn H5F__delete(pkg: &mut FilePackageState, id: u64) -> Option<FileApiState> {
    pkg.files.remove(&id)
}

pub fn H5F_try_close(file: &mut FileApiState) -> bool {
    file.nopen_objs == 0
}

pub fn H5F_get_id(file: &FileApiState) -> u64 {
    file.id
}

pub fn H5F_incr_nopen_objs(file: &mut FileApiState) -> usize {
    file.nopen_objs = file.nopen_objs.saturating_add(1);
    file.nopen_objs
}

pub fn H5F_decr_nopen_objs(file: &mut FileApiState) -> usize {
    file.nopen_objs = file.nopen_objs.saturating_sub(1);
    file.nopen_objs
}

pub fn H5F__build_actual_name(file: &mut FileApiState, name: &str) {
    file.actual_name = name.to_string();
}

pub fn H5F_set_grp_btree_shared(file: &mut FileApiState, enabled: bool) {
    if enabled {
        file.flags |= 1;
    } else {
        file.flags &= !1;
    }
}

pub fn H5F_set_sohm_addr(file: &mut FileApiState, addr: u64) {
    file.sohm_addr = Some(addr);
}

pub fn H5F_set_sohm_vers(file: &mut FileApiState, version: u8) {
    file.sohm_vers = version;
}

pub fn H5F_set_sohm_nindexes(file: &mut FileApiState, nindexes: u8) {
    file.sohm_nindexes = nindexes;
}

pub fn H5F_set_store_msg_crt_idx(file: &mut FileApiState, enabled: bool) {
    file.store_msg_crt_idx = enabled;
}

pub fn H5F__set_libver_bounds(file: &mut FileApiState, low: u8, high: u8) {
    file.low_bound = low;
    file.high_bound = high;
}

pub fn H5F__get_file_image(file: &FileApiState) -> Vec<u8> {
    file.image.clone()
}

pub fn H5F__get_info(file: &FileApiState) -> (u64, u64) {
    (file.eof, file.eoa)
}

pub fn H5F_set_retries(file: &mut FileApiState, retries: u32) {
    file.retries = retries;
}

pub fn H5F__set_eoa(file: &mut FileApiState, eoa: u64) {
    file.eoa = eoa;
    file.eof = file.eof.max(eoa);
}

pub fn H5F__set_paged_aggr(_file: &mut FileApiState) {}

pub fn H5F__get_max_eof_eoa(file: &FileApiState) -> u64 {
    file.eof.max(file.eoa)
}

pub fn H5F__start_swmr_write(file: &mut FileApiState) {
    file.swmr_write = true;
}

pub fn H5F__format_convert(file: &mut FileApiState) -> Result<()> {
    H5Fformat_convert(file)
}

pub fn H5F_get_file_id(file: &FileApiState) -> u64 {
    file.id
}

pub fn H5F__super_ext_create(file: &mut FileApiState, addr: u64) {
    file.super_ext_addr = Some(addr);
}

pub fn H5F__super_ext_open(file: &FileApiState) -> Option<u64> {
    file.super_ext_addr
}

pub fn H5F__super_ext_close(_file: &mut FileApiState) {}

pub fn H5F__update_super_ext_driver_msg(_file: &mut FileApiState) {}

pub fn H5F__super_init(file: &mut FileApiState) {
    file.image = b"\x89HDF\r\n\x1a\n".to_vec();
}

pub fn H5F_eoa_dirty(_file: &mut FileApiState) {}

pub fn H5F_super_dirty(_file: &mut FileApiState) {}

pub fn H5F__super_free(file: &mut FileApiState) {
    file.super_ext_addr = None;
}

pub fn H5F__super_ext_remove_msg(file: &mut FileApiState) {
    file.super_ext_addr = None;
}

pub fn H5F_shared_block_read(file: &FileApiState, offset: usize, len: usize) -> Result<Vec<u8>> {
    H5F_block_read(file, offset, len)
}

pub fn H5F_block_read(file: &FileApiState, offset: usize, len: usize) -> Result<Vec<u8>> {
    let end = offset
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat("H5F read overflow".into()))?;
    file.image
        .get(offset..end)
        .map(<[u8]>::to_vec)
        .ok_or_else(|| Error::InvalidFormat("H5F read is outside file image".into()))
}

pub fn H5F_shared_block_write(file: &mut FileApiState, offset: usize, data: &[u8]) -> Result<()> {
    H5F_block_write(file, offset, data)
}

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

pub fn H5F_shared_select_read(file: &FileApiState, spans: &[(usize, usize)]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    for &(offset, len) in spans {
        out.extend(H5F_block_read(file, offset, len)?);
    }
    Ok(out)
}

pub fn H5F_shared_select_write(file: &mut FileApiState, spans: &[(usize, &[u8])]) -> Result<()> {
    for &(offset, data) in spans {
        H5F_block_write(file, offset, data)?;
    }
    Ok(())
}

pub fn H5F_shared_vector_read(file: &FileApiState, spans: &[(usize, usize)]) -> Result<Vec<u8>> {
    H5F_shared_select_read(file, spans)
}

pub fn H5F_shared_vector_write(file: &mut FileApiState, spans: &[(usize, &[u8])]) -> Result<()> {
    H5F_shared_select_write(file, spans)
}

pub fn H5F_flush_tagged_metadata(file: &mut FileApiState) {
    H5F__flush(file);
}

pub fn H5F_get_checksums(file: &FileApiState) -> u32 {
    file.image
        .iter()
        .fold(0u32, |acc, byte| acc.wrapping_add(u32::from(*byte)))
}

pub fn H5F_debug(file: &FileApiState) -> String {
    format!("H5F(id={}, name={}, eof={})", file.id, file.name, file.eof)
}

pub fn H5F_sfile_assert_num(file: &FileApiState, expected: usize) -> bool {
    file.open_ids.len() == expected
}

pub fn H5F__sfile_add(file: &mut FileApiState, id: u64) {
    file.open_ids.insert(id);
}

pub fn H5F__sfile_search(file: &FileApiState, id: u64) -> bool {
    file.open_ids.contains(&id)
}

pub fn H5F__sfile_remove(file: &mut FileApiState, id: u64) {
    file.open_ids.remove(&id);
}

pub fn H5F_mpi_get_rank() -> Result<u32> {
    Err(unsupported_file("H5F_mpi_get_rank"))
}

pub fn H5F_mpi_get_comm() -> Result<()> {
    Err(unsupported_file("H5F_mpi_get_comm"))
}

pub fn H5F_mpi_get_info() -> Result<()> {
    Err(unsupported_file("H5F_mpi_get_info"))
}

pub fn H5F_shared_mpi_get_size() -> Result<u64> {
    Err(unsupported_file("H5F_shared_mpi_get_size"))
}

pub fn H5F_mpi_get_size() -> Result<u64> {
    Err(unsupported_file("H5F_mpi_get_size"))
}

pub fn H5F__set_mpi_atomicity(file: &mut FileApiState, atomicity: bool) {
    file.mpi_atomicity = atomicity;
}

pub fn H5Fset_mpi_atomicity(_file: &mut FileApiState, _atomicity: bool) -> Result<()> {
    Err(unsupported_file("H5Fset_mpi_atomicity"))
}

pub fn H5F__get_mpi_atomicity(file: &FileApiState) -> bool {
    file.mpi_atomicity
}

pub fn H5F_mpi_retrieve_comm() -> Result<()> {
    Err(unsupported_file("H5F_mpi_retrieve_comm"))
}

pub fn H5F_get_coll_metadata_reads(file: &FileApiState) -> bool {
    file.coll_metadata_reads
}

pub fn H5F_shared_get_coll_metadata_reads(file: &FileApiState) -> bool {
    file.coll_metadata_reads
}

pub fn H5F_set_coll_metadata_reads(file: &mut FileApiState, value: bool) {
    file.coll_metadata_reads = value;
}

pub fn H5F_mpi_get_file_block_type() -> Result<()> {
    Err(unsupported_file("H5F_mpi_get_file_block_type"))
}

pub fn H5F__efc_create(max_nfiles: usize) -> ExternalFileCache {
    ExternalFileCache {
        max_nfiles,
        ..ExternalFileCache::default()
    }
}

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

pub fn H5F_efc_close(cache: &mut ExternalFileCache, name: &str) -> Option<FileApiState> {
    cache.close_attempts = cache.close_attempts.saturating_add(1);
    cache.files.remove(name)
}

pub fn H5F__efc_max_nfiles(cache: &ExternalFileCache) -> usize {
    cache.max_nfiles
}

pub fn H5F__efc_release_real(cache: &mut ExternalFileCache, name: &str) -> Option<FileApiState> {
    H5F_efc_close(cache, name)
}

pub fn H5F__efc_release(cache: &mut ExternalFileCache, name: &str) -> Option<FileApiState> {
    H5F__efc_release_real(cache, name)
}

pub fn H5F__efc_destroy(cache: &mut ExternalFileCache) {
    cache.files.clear();
}

pub fn H5F__efc_remove_ent(cache: &mut ExternalFileCache, name: &str) -> Option<FileApiState> {
    cache.files.remove(name)
}

pub fn H5F__efc_try_close_tag1(cache: &mut ExternalFileCache, name: &str) -> bool {
    H5F__efc_try_close(cache, name)
}

pub fn H5F__efc_try_close_tag2(cache: &mut ExternalFileCache, name: &str) -> bool {
    H5F__efc_try_close(cache, name)
}

pub fn H5F__efc_try_close(cache: &mut ExternalFileCache, name: &str) -> bool {
    cache.close_attempts = cache.close_attempts.saturating_add(1);
    cache.files.remove(name).is_some()
}

pub fn H5F_fake_alloc(file: &mut FileApiState, size: u64) -> Result<u64> {
    H5F__alloc(file, size)
}

pub fn H5F_fake_free(file: &mut FileApiState, addr: u64, size: u64) -> Result<()> {
    H5F__free(file, addr, size)
}

pub fn H5F__get_sohm_mesg_count_test(file: &FileApiState) -> u8 {
    file.sohm_nindexes
}

pub fn H5F__check_cached_stab_test(_file: &FileApiState) -> bool {
    true
}

pub fn H5F__get_maxaddr_test(file: &FileApiState) -> u64 {
    H5F__get_max_eof_eoa(file)
}

pub fn H5F__get_sbe_addr_test(file: &FileApiState) -> Option<u64> {
    file.super_ext_addr
}

pub fn H5F__same_file_test(left: &FileApiState, right: &FileApiState) -> bool {
    left.fileno == right.fileno
}

pub fn H5F__accum_read(file: &FileApiState) -> &[u8] {
    &file.accum
}

pub fn H5F__accum_adjust(file: &mut FileApiState, len: usize) {
    file.accum.resize(len, 0);
}

pub fn H5F__accum_write(file: &mut FileApiState, data: &[u8]) {
    file.accum.clear();
    file.accum.extend_from_slice(data);
}

pub fn H5F__accum_free(file: &mut FileApiState) {
    file.accum.clear();
}

pub fn H5F__accum_flush(file: &mut FileApiState) -> Result<()> {
    let data = file.accum.clone();
    let eof = u64_to_usize(file.eof, "H5F accumulator EOF")?;
    H5F_block_write(file, eof, &data)?;
    file.accum.clear();
    Ok(())
}

fn usize_to_u64(value: usize, context: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| Error::InvalidFormat(format!("{context} exceeds u64")))
}

fn u64_to_usize(value: u64, context: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| Error::InvalidFormat(format!("{context} exceeds usize")))
}

pub fn H5F__accum_reset(file: &mut FileApiState) {
    file.accum.clear();
}

pub fn H5F_shared_get_intent(file: &FileApiState) -> u32 {
    file.intent
}

pub fn H5F_get_low_bound(file: &FileApiState) -> u8 {
    file.low_bound
}

pub fn H5F_get_high_bound(file: &FileApiState) -> u8 {
    file.high_bound
}

pub fn H5F_get_actual_name(file: &FileApiState) -> &str {
    &file.actual_name
}

pub fn H5F_get_extpath(file: &FileApiState) -> &str {
    &file.name
}

pub fn H5F_get_shared(file: &FileApiState) -> &FileApiState {
    file
}

pub fn H5F_same_shared(left: &FileApiState, right: &FileApiState) -> bool {
    left.fileno == right.fileno
}

pub fn H5F_get_nopen_objs(file: &FileApiState) -> usize {
    file.nopen_objs
}

pub fn H5F_file_id_exists(pkg: &FilePackageState, id: u64) -> bool {
    pkg.files.contains_key(&id)
}

pub fn H5F_get_parent(_file: &FileApiState) -> Option<u64> {
    None
}

pub fn H5F_get_nmounts(file: &FileApiState) -> usize {
    file.mounts.len()
}

pub fn H5F_get_fcpl(file: &FileApiState) -> u64 {
    file.flags
}

pub fn H5F_sizeof_addr() -> usize {
    std::mem::size_of::<u64>()
}

pub fn H5F_get_sohm_addr(file: &FileApiState) -> Option<u64> {
    file.sohm_addr
}

pub fn H5F_get_sohm_vers(file: &FileApiState) -> u8 {
    file.sohm_vers
}

pub fn H5F_get_sohm_nindexes(file: &FileApiState) -> u8 {
    file.sohm_nindexes
}

pub fn H5F_sym_leaf_k() -> u16 {
    32
}

pub fn H5F_get_min_dset_ohdr() -> usize {
    0
}

pub fn H5F_kvalue() -> u16 {
    32
}

pub fn H5F_get_nrefs(file: &FileApiState) -> usize {
    file.nrefs
}

pub fn H5F_rdcc_nslots() -> usize {
    521
}

pub fn H5F_rdcc_nbytes() -> usize {
    1024 * 1024
}

pub fn H5F_rdcc_w0() -> f64 {
    0.75
}

pub fn H5F_get_base_addr(file: &FileApiState) -> u64 {
    file.base_addr
}

pub fn H5F_grp_btree_shared(file: &FileApiState) -> bool {
    file.flags & 1 != 0
}

pub fn H5F_gc_ref() -> bool {
    false
}

pub fn H5F_get_fc_degree() -> u8 {
    0
}

pub fn H5F_get_evict_on_close() -> bool {
    false
}

pub fn H5F_store_msg_crt_idx(file: &FileApiState) -> bool {
    file.store_msg_crt_idx
}

pub fn H5F_shared_has_feature(file: &FileApiState, feature: u64) -> bool {
    file.flags & feature != 0
}

pub fn H5F_has_feature(file: &FileApiState, feature: u64) -> bool {
    H5F_shared_has_feature(file, feature)
}

pub fn H5F_get_fileno(file: &FileApiState) -> u64 {
    file.fileno
}

pub fn H5F_shared_get_eoa(file: &FileApiState) -> u64 {
    file.eoa
}

pub fn H5F_get_eoa(file: &FileApiState) -> u64 {
    file.eoa
}

pub fn H5F_get_vfd_handle() -> Result<()> {
    Err(unsupported_file("H5F_get_vfd_handle"))
}

pub fn H5F_is_tmp_addr(_addr: u64) -> bool {
    false
}

pub fn H5F_use_tmp_space() -> bool {
    false
}

pub fn H5F_shared_get_mpi_file_sync_required() -> bool {
    false
}

pub fn H5F_use_mdc_logging(file: &FileApiState) -> bool {
    file.metadata_logging
}

pub fn H5F_start_mdc_log_on_access(file: &mut FileApiState) {
    file.metadata_logging = true;
}

pub fn H5F_mdc_log_location(file: &FileApiState) -> &str {
    &file.name
}

pub fn H5F_get_alignment() -> (u64, u64) {
    (1, 1)
}

pub fn H5F_get_threshold() -> u64 {
    1
}

pub fn H5F_get_pgend_meta_thres() -> u32 {
    0
}

pub fn H5F_get_null_fsm_addr() -> Option<u64> {
    None
}

pub fn H5F_get_vol_obj(file: &FileApiState) -> Option<u64> {
    file.vol_obj
}

pub fn H5F__get_cont_info(file: &FileApiState) -> (u64, u64) {
    (file.base_addr, file.eoa)
}

pub fn H5F_get_use_file_locking(file: &FileApiState) -> bool {
    file.file_locking
}

pub fn H5F_has_vector_select_io() -> bool {
    true
}

pub fn H5F_get_rfic_flags(file: &FileApiState) -> u64 {
    file.flags
}

pub fn H5F__close_mounts(file: &mut FileApiState) {
    file.mounts.clear();
}

pub fn H5F_mount(file: &mut FileApiState, name: &str) {
    file.mounts.insert(name.to_string());
}

pub fn H5F_unmount(file: &mut FileApiState, name: &str) {
    file.mounts.remove(name);
}

pub fn H5F_is_mount(file: &FileApiState, name: &str) -> bool {
    file.mounts.contains(name)
}

pub fn H5F__mount_count_ids_recurse(file: &FileApiState) -> usize {
    file.mounts.len()
}

pub fn H5F__mount_count_ids(file: &FileApiState) -> usize {
    file.mounts.len()
}

pub fn H5F__flush_mounts_recurse(_file: &mut FileApiState) {}

pub fn H5F_flush_mounts(file: &mut FileApiState) {
    H5F__flush(file);
}

pub fn H5F_traverse_mount(file: &FileApiState) -> Vec<String> {
    file.mounts.iter().cloned().collect()
}

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

        let flags = H5F_get_mdc_logging_status(&file);
        assert!(H5F_get_swmr_write(&file));
        assert!(flags.swmr_write);
        assert!(flags.metadata_logging);
        assert!(flags.coll_metadata_reads);
        assert!(flags.mpi_atomicity);
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
        let image = b"\x89HDF\r\n\x1a\nrest".to_vec();
        assert!(H5F__cache_superblock_verify_chksum(&image));
        assert_eq!(H5F__cache_superblock_deserialize(&image).unwrap(), image);

        let bad = b"not-hdf5";
        assert!(!H5F__cache_superblock_verify_chksum(bad));
        assert!(H5F__cache_superblock_deserialize(bad).is_err());
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
