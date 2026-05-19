/// File access properties used by this pure-Rust reader.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileAccess {
    driver: String,
    userblock: u64,
    alignment: (u64, u64),
    cache: (usize, usize, usize, u64),
    gc_references: bool,
    fclose_degree: FileCloseDegree,
    meta_block_size: u64,
    sieve_buf_size: usize,
    small_data_block_size: u64,
    libver_bounds: (LibverBound, LibverBound),
    evict_on_close: bool,
    file_locking: (bool, bool),
    mdc_config: MetadataCacheConfig,
    mdc_image_config: MetadataCacheImageConfig,
    mdc_log_options: MetadataCacheLogOptions,
    all_coll_metadata_ops: bool,
    coll_metadata_write: bool,
    page_buffer_size: (usize, u32, u32),
    core_write_tracking: bool,
    map_iterate_hints: Option<String>,
    object_flush_cb: bool,
}

/// File close degree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileCloseDegree {
    Weak,
    Semi,
    Strong,
}

/// HDF5 library format-version bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LibverBound {
    Earliest,
    V18,
    V110,
    V112,
    V114,
    Latest,
}

/// Metadata cache configuration used by this reader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetadataCacheConfig {
    pub enabled: bool,
    pub max_size: usize,
    pub min_clean_size: usize,
}

/// Metadata-cache image settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetadataCacheImageConfig {
    pub enabled: bool,
    pub generation_enabled: bool,
}

/// Metadata-cache logging settings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetadataCacheLogOptions {
    pub enabled: bool,
    pub location: Option<String>,
    pub start_on_access: bool,
}

impl Default for FileAccess {
    fn default() -> Self {
        Self {
            driver: "sec2".to_string(),
            userblock: 0,
            alignment: (1, 1),
            cache: (0, 521, 1024 * 1024, 0.75f64.to_bits()),
            gc_references: false,
            fclose_degree: FileCloseDegree::Weak,
            meta_block_size: 2048,
            sieve_buf_size: 64 * 1024,
            small_data_block_size: 2048,
            libver_bounds: (LibverBound::Earliest, LibverBound::Latest),
            evict_on_close: false,
            file_locking: (true, false),
            mdc_config: MetadataCacheConfig {
                enabled: false,
                max_size: 0,
                min_clean_size: 0,
            },
            mdc_image_config: MetadataCacheImageConfig {
                enabled: false,
                generation_enabled: false,
            },
            mdc_log_options: MetadataCacheLogOptions {
                enabled: false,
                location: None,
                start_on_access: false,
            },
            all_coll_metadata_ops: false,
            coll_metadata_write: false,
            page_buffer_size: (0, 0, 0),
            core_write_tracking: false,
            map_iterate_hints: None,
            object_flush_cb: false,
        }
    }
}

impl FileAccess {
    pub(crate) fn from_file(file: &crate::hl::file::File) -> Self {
        file.access_plist_snapshot()
    }

    /// File-driver name. This crate reads regular files directly.
    pub fn driver(&self) -> &str {
        &self.driver
    }

    /// Set file-driver name as stored property-list state.
    pub fn set_driver<S: Into<String>>(&mut self, driver: S) {
        self.driver = driver.into();
    }

    /// Set file-driver name by registered driver name.
    pub fn set_driver_by_name<S: Into<String>>(&mut self, driver: S) {
        self.set_driver(driver);
    }

    /// Set file-driver name by registered driver value.
    pub fn set_driver_by_value(&mut self, driver_value: u64) {
        self.driver = format!("driver_{driver_value}");
    }

    /// Select the sec2-style local file driver.
    pub fn set_fapl_sec2(&mut self) {
        self.driver = "sec2".to_string();
    }

    /// Select the Windows-style local file driver.
    pub fn set_fapl_windows(&mut self) {
        self.driver = "windows".to_string();
    }

    /// Select the stdio-style local file driver.
    pub fn set_fapl_stdio(&mut self) {
        self.driver = "stdio".to_string();
    }

    /// Driver-specific info. The direct file driver has no extra info here.
    pub fn driver_info(&self) -> Option<()> {
        None
    }

    /// Userblock size in bytes. Current parser supports files whose HDF5
    /// signature starts at byte 0.
    pub fn userblock(&self) -> u64 {
        self.userblock
    }

    /// Set userblock size in bytes.
    pub fn set_userblock(&mut self, userblock: u64) {
        self.userblock = userblock;
    }

    /// Metadata/raw-data alignment as `(threshold, alignment)`.
    pub fn alignment(&self) -> (u64, u64) {
        self.alignment
    }

    /// Set metadata/raw-data alignment as `(threshold, alignment)`.
    pub fn set_alignment(&mut self, threshold: u64, alignment: u64) {
        self.alignment = (threshold, alignment);
    }

    /// Metadata cache settings `(metadata_cache_elements, raw_chunk_cache_elements,
    /// raw_chunk_cache_bytes, raw_chunk_cache_preemption)`.
    pub fn cache(&self) -> (usize, usize, usize, f64) {
        (
            self.cache.0,
            self.cache.1,
            self.cache.2,
            f64::from_bits(self.cache.3),
        )
    }

    /// Set metadata/raw chunk cache settings.
    pub fn set_cache(
        &mut self,
        metadata_elements: usize,
        raw_chunk_elements: usize,
        raw_chunk_bytes: usize,
        raw_chunk_preemption: f64,
    ) {
        self.cache = (
            metadata_elements,
            raw_chunk_elements,
            raw_chunk_bytes,
            raw_chunk_preemption.to_bits(),
        );
    }

    /// Whether garbage collection of references is enabled.
    pub fn gc_references(&self) -> bool {
        self.gc_references
    }

    /// Set whether garbage collection of references is enabled.
    pub fn set_gc_references(&mut self, enabled: bool) {
        self.gc_references = enabled;
    }

    /// File close degree used by the direct reader.
    pub fn fclose_degree(&self) -> FileCloseDegree {
        self.fclose_degree
    }

    /// Set file close degree.
    pub fn set_fclose_degree(&mut self, degree: FileCloseDegree) {
        self.fclose_degree = degree;
    }

    /// Metadata block aggregation size in bytes.
    pub fn meta_block_size(&self) -> u64 {
        self.meta_block_size
    }

    /// Set metadata block aggregation size in bytes.
    pub fn set_meta_block_size(&mut self, size: u64) {
        self.meta_block_size = size;
    }

    /// Data sieve buffer size in bytes.
    pub fn sieve_buf_size(&self) -> usize {
        self.sieve_buf_size
    }

    /// Set data sieve buffer size in bytes.
    pub fn set_sieve_buf_size(&mut self, size: usize) {
        self.sieve_buf_size = size;
    }

    /// Small raw-data block aggregation size in bytes.
    pub fn small_data_block_size(&self) -> u64 {
        self.small_data_block_size
    }

    /// Set small raw-data block aggregation size in bytes.
    pub fn set_small_data_block_size(&mut self, size: u64) {
        self.small_data_block_size = size;
    }

    /// Library format-version bounds used when opening files.
    pub fn libver_bounds(&self) -> (LibverBound, LibverBound) {
        self.libver_bounds
    }

    /// Set library format-version bounds.
    pub fn set_libver_bounds(&mut self, low: LibverBound, high: LibverBound) {
        self.libver_bounds = (low, high);
    }

    /// Whether object metadata should be evicted when objects are closed.
    pub fn evict_on_close(&self) -> bool {
        self.evict_on_close
    }

    /// Set object-metadata eviction on close.
    pub fn set_evict_on_close(&mut self, enabled: bool) {
        self.evict_on_close = enabled;
    }

    /// File-locking policy `(enabled, ignore_when_disabled)`.
    pub fn file_locking(&self) -> (bool, bool) {
        self.file_locking
    }

    /// Set file-locking policy `(enabled, ignore_when_disabled)`.
    pub fn set_file_locking(&mut self, enabled: bool, ignore_when_disabled: bool) {
        self.file_locking = (enabled, ignore_when_disabled);
    }

    /// Metadata cache configuration. This reader does not maintain libhdf5's
    /// adaptive metadata cache.
    pub fn mdc_config(&self) -> MetadataCacheConfig {
        self.mdc_config
    }

    /// Borrow metadata cache configuration without materializing an owned copy.
    pub fn mdc_config_ref(&self) -> &MetadataCacheConfig {
        &self.mdc_config
    }

    /// Set metadata cache configuration.
    pub fn set_mdc_config(&mut self, config: MetadataCacheConfig) {
        self.mdc_config = config;
    }

    /// Metadata cache image configuration.
    pub fn mdc_image_config(&self) -> MetadataCacheImageConfig {
        self.mdc_image_config
    }

    /// Borrow metadata cache image configuration without materializing an owned copy.
    pub fn mdc_image_config_ref(&self) -> &MetadataCacheImageConfig {
        &self.mdc_image_config
    }

    /// Set metadata cache image configuration.
    pub fn set_mdc_image_config(&mut self, config: MetadataCacheImageConfig) {
        self.mdc_image_config = config;
    }

    /// Metadata cache logging options.
    pub fn mdc_log_options(&self) -> MetadataCacheLogOptions {
        self.mdc_log_options.clone()
    }

    /// Borrow metadata cache logging options without cloning the log location.
    pub fn mdc_log_options_ref(&self) -> &MetadataCacheLogOptions {
        &self.mdc_log_options
    }

    /// Set metadata cache logging options.
    pub fn set_mdc_log_options(&mut self, options: MetadataCacheLogOptions) {
        self.mdc_log_options = options;
    }

    /// Whether all metadata reads are collective. MPI collective metadata is
    /// not used by this reader.
    pub fn all_coll_metadata_ops(&self) -> bool {
        self.all_coll_metadata_ops
    }

    /// Set collective-metadata-read flag as property-list state.
    pub fn set_all_coll_metadata_ops(&mut self, enabled: bool) {
        self.all_coll_metadata_ops = enabled;
    }

    /// Whether metadata writes are collective. This read-only access property
    /// list never performs collective metadata writes.
    pub fn coll_metadata_write(&self) -> bool {
        self.coll_metadata_write
    }

    /// Set collective-metadata-write flag as property-list state.
    pub fn set_coll_metadata_write(&mut self, enabled: bool) {
        self.coll_metadata_write = enabled;
    }

    /// Page buffer settings `(size, minimum_metadata_percent,
    /// minimum_raw_data_percent)`.
    pub fn page_buffer_size(&self) -> (usize, u32, u32) {
        self.page_buffer_size
    }

    /// Set page buffer settings.
    pub fn set_page_buffer_size(
        &mut self,
        size: usize,
        min_metadata_percent: u32,
        min_raw_data_percent: u32,
    ) {
        self.page_buffer_size = (size, min_metadata_percent, min_raw_data_percent);
    }

    /// HDFS VFD configuration. Not active for the direct reader.
    pub fn fapl_hdfs(&self) -> Option<()> {
        None
    }

    /// Direct VFD configuration. Not active for the direct reader.
    pub fn fapl_direct(&self) -> Option<()> {
        None
    }

    /// Mirror VFD configuration. Not active for the direct reader.
    pub fn fapl_mirror(&self) -> Option<()> {
        None
    }

    /// MPI-IO VFD configuration. MPI is intentionally not used.
    pub fn fapl_mpio(&self) -> Option<()> {
        None
    }

    /// Dataset transfer MPI-IO configuration. MPI is intentionally not used.
    pub fn dxpl_mpio(&self) -> Option<()> {
        None
    }

    /// Family VFD configuration. Not active for the direct reader.
    pub fn fapl_family(&self) -> Option<()> {
        None
    }

    /// Family VFD member offset. Not active for the direct reader.
    pub fn family_offset(&self) -> Option<u64> {
        None
    }

    /// Multi VFD memory type. Not active for the direct reader.
    pub fn multi_type(&self) -> Option<u32> {
        None
    }

    /// IOC VFD configuration. Not active for the direct reader.
    pub fn fapl_ioc(&self) -> Option<()> {
        None
    }

    /// Subfiling VFD configuration. Not active for the direct reader.
    pub fn fapl_subfiling(&self) -> Option<()> {
        None
    }

    /// Splitter VFD configuration. Not active for the direct reader.
    pub fn fapl_splitter(&self) -> Option<()> {
        None
    }

    /// Legacy multi VFD configuration. Not active for the direct reader.
    pub fn fapl_multi(&self) -> Option<()> {
        None
    }

    /// Onion VFD configuration. Not active for the direct reader.
    pub fn fapl_onion(&self) -> Option<()> {
        None
    }

    /// Core VFD write-tracking flag. Not active for the direct reader.
    pub fn core_write_tracking(&self) -> bool {
        self.core_write_tracking
    }

    /// Set core-driver write-tracking flag as property-list state.
    pub fn set_core_write_tracking(&mut self, enabled: bool) {
        self.core_write_tracking = enabled;
    }

    /// Core VFD configuration. Not active for the direct reader.
    pub fn fapl_core(&self) -> Option<()> {
        None
    }

    /// ROS3 VFD configuration. Not active for the direct reader.
    pub fn fapl_ros3(&self) -> Option<()> {
        None
    }

    /// ROS3 endpoint string. Not active for the direct reader.
    pub fn fapl_ros3_endpoint(&self) -> Option<&str> {
        None
    }

    /// Object flush callback. This reader does not install one.
    pub fn object_flush_cb(&self) -> Option<()> {
        self.object_flush_cb.then_some(())
    }

    /// Set object flush callback presence.
    pub fn set_object_flush_cb(&mut self, installed: bool) {
        self.object_flush_cb = installed;
    }

    /// MPI communicator/info parameters. MPI is intentionally not used.
    pub fn mpi_params(&self) -> Option<()> {
        None
    }

    /// VOL connector id. VOL/plugin infrastructure is intentionally not used.
    pub fn vol_id(&self) -> Option<()> {
        None
    }

    /// VOL connector info. VOL/plugin infrastructure is intentionally not used.
    pub fn vol_info(&self) -> Option<()> {
        None
    }

    /// VOL connector capability flags. VOL/plugin infrastructure is
    /// intentionally not used.
    pub fn vol_cap_flags(&self) -> u64 {
        0
    }

    /// Relaxed file-integrity-check flags.
    pub fn relax_file_integrity_checks(&self) -> u64 {
        0
    }

    /// Map API iteration hints. The Map API is intentionally not implemented.
    pub fn map_iterate_hints(&self) -> Option<()> {
        self.map_iterate_hints.as_ref().map(|_| ())
    }

    /// Set Map API iteration hints as opaque property-list state.
    pub fn set_map_iterate_hints<S: Into<String>>(&mut self, hints: Option<S>) {
        self.map_iterate_hints = hints.map(Into::into);
    }
}
