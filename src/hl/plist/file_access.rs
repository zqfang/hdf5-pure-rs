use crate::engine::property::{H5P__decode_hdfs_fapl_config, H5P__decode_ros3_fapl_config};
use crate::engine::vfd::{
    CoreFileConfig, DirectFileConfig, FamilyFileConfig, H5FD__family_sb_decode,
    H5FD__hdfs_validate_config, H5FD__log_sb_decode, H5FD__onion_sb_decode,
    H5FD__ros3_validate_config, H5FD__splitter_sb_decode, H5FD__subfiling_sb_decode,
    H5FD_multi_sb_decode, HdfsConfig, IocConfig, LogFileConfig, MultiFileConfig, OnionHeader,
    Ros3Config, SplitterFileConfig, SubfilingConfig,
};

const ROS3_DEFAULT_PAGE_BUFFER_SIZE: usize = 64 * 1024 * 1024;

/// File access properties used by this pure-Rust reader.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileAccess {
    driver: String,
    hdfs_config: Option<HdfsConfig>,
    direct_config: Option<DirectFileConfig>,
    family_config: Option<FamilyFileConfig>,
    family_offset: Option<u64>,
    multi_config: Option<MultiFileConfig>,
    multi_type: Option<u32>,
    ioc_config: Option<IocConfig>,
    subfiling_config: Option<SubfilingConfig>,
    splitter_config: Option<SplitterFileConfig>,
    log_config: Option<LogFileConfig>,
    onion_config: Option<OnionHeader>,
    core_config: Option<CoreFileConfig>,
    ros3_config: Option<Ros3Config>,
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
            hdfs_config: None,
            direct_config: None,
            family_config: None,
            family_offset: None,
            multi_config: None,
            multi_type: None,
            ioc_config: None,
            subfiling_config: None,
            splitter_config: None,
            log_config: None,
            onion_config: None,
            core_config: None,
            ros3_config: None,
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
        let mut plist = file.access_plist_snapshot();
        plist.userblock = file.userblock();
        plist
    }

    /// File-driver name. This crate reads regular files directly.
    pub fn driver(&self) -> &str {
        &self.driver
    }

    /// Return an explicit error when a stored FAPL driver cannot be honored by
    /// this pure-Rust local-file backend.
    pub fn ensure_runtime_supported_driver(&self) -> crate::Result<()> {
        match self.driver.as_str() {
            "sec2" | "stdio" | "windows" => Ok(()),
            driver => Err(crate::Error::Unsupported(format!(
                "file access driver '{driver}' is not supported by the pure Rust local backend"
            ))),
        }
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

    /// Select the direct VFD and retain its driver-specific config as
    /// property-list state. Runtime I/O remains unsupported.
    pub fn set_fapl_direct(&mut self, config: DirectFileConfig) {
        self.driver = "direct".to_string();
        self.direct_config = Some(config);
    }

    /// Select the core VFD and retain its driver-specific config as
    /// property-list state. Runtime I/O remains unsupported.
    pub fn set_fapl_core(&mut self, config: CoreFileConfig) {
        self.driver = "core".to_string();
        self.core_config = Some(config);
    }

    /// Select the family VFD and retain its driver-specific config as
    /// property-list state. Runtime I/O remains unsupported.
    pub fn set_fapl_family(&mut self, config: FamilyFileConfig) {
        self.driver = "family".to_string();
        self.family_config = Some(config);
    }

    /// Parse and retain the family VFD superblock-driver-info image as
    /// property-list state. Runtime I/O remains unsupported.
    pub fn set_fapl_family_from_config_image(&mut self, bytes: &[u8]) -> crate::Result<()> {
        let config = H5FD__family_sb_decode(bytes)?;
        self.set_fapl_family(config);
        Ok(())
    }

    /// Set the family VFD member offset as property-list state.
    pub fn set_family_offset(&mut self, offset: Option<u64>) {
        self.family_offset = offset;
    }

    /// Select the multi VFD and retain its driver-specific config as
    /// property-list state. Runtime I/O remains unsupported.
    pub fn set_fapl_multi(&mut self, config: MultiFileConfig) {
        self.driver = "multi".to_string();
        self.multi_config = Some(config);
    }

    /// Parse and retain the multi VFD superblock-driver-info image as
    /// property-list state. Runtime I/O remains unsupported.
    pub fn set_fapl_multi_from_config_image(&mut self, bytes: &[u8]) -> crate::Result<()> {
        let config = H5FD_multi_sb_decode(bytes)?;
        self.set_fapl_multi(config);
        Ok(())
    }

    /// Set the multi VFD memory type as property-list state.
    pub fn set_multi_type(&mut self, memory_type: Option<u32>) {
        self.multi_type = memory_type;
    }

    /// Retain IOC VFD config as property-list state. Runtime I/O remains
    /// unsupported.
    pub fn set_fapl_ioc(&mut self, config: IocConfig) {
        self.driver = "ioc".to_string();
        self.ioc_config = Some(config);
    }

    /// Select the subfiling VFD and retain its driver-specific config as
    /// property-list state. Runtime I/O remains unsupported.
    pub fn set_fapl_subfiling(&mut self, config: SubfilingConfig) {
        self.driver = "subfiling".to_string();
        self.subfiling_config = Some(config);
    }

    /// Parse and retain the subfiling VFD superblock-driver-info image as
    /// property-list state. Runtime I/O remains unsupported.
    pub fn set_fapl_subfiling_from_config_image(&mut self, bytes: &[u8]) -> crate::Result<()> {
        let config = H5FD__subfiling_sb_decode(bytes)?;
        self.set_fapl_subfiling(config);
        Ok(())
    }

    /// Select the splitter VFD and retain its driver-specific config as
    /// property-list state. Runtime I/O remains unsupported.
    pub fn set_fapl_splitter(&mut self, config: SplitterFileConfig) {
        self.driver = "splitter".to_string();
        self.splitter_config = Some(config);
    }

    /// Parse and retain the splitter VFD config image as property-list state.
    /// Runtime I/O remains unsupported.
    pub fn set_fapl_splitter_from_config_image(&mut self, bytes: &[u8]) -> crate::Result<()> {
        let config = H5FD__splitter_sb_decode(bytes)?;
        self.set_fapl_splitter(config);
        Ok(())
    }

    /// Select the log VFD and retain its driver-specific config as
    /// property-list state. Runtime I/O remains unsupported.
    pub fn set_fapl_log(&mut self, config: LogFileConfig) {
        self.driver = "log".to_string();
        self.log_config = Some(config);
    }

    /// Parse and retain the log VFD config image as property-list state.
    /// Runtime I/O remains unsupported.
    pub fn set_fapl_log_from_config_image(&mut self, bytes: &[u8]) -> crate::Result<()> {
        let config = H5FD__log_sb_decode(bytes)?;
        self.set_fapl_log(config);
        Ok(())
    }

    /// Select the onion VFD and retain its driver-specific config as
    /// property-list state. Runtime I/O remains unsupported.
    pub fn set_fapl_onion(&mut self, config: OnionHeader) {
        self.driver = "onion".to_string();
        self.onion_config = Some(config);
    }

    /// Parse and retain the onion VFD superblock-driver-info image as
    /// property-list state. Runtime I/O remains unsupported.
    pub fn set_fapl_onion_from_config_image(&mut self, bytes: &[u8]) -> crate::Result<()> {
        let config = H5FD__onion_sb_decode(bytes)?;
        self.set_fapl_onion(config);
        Ok(())
    }

    /// Select the HDFS VFD and retain its driver-specific config as
    /// property-list state. Runtime I/O remains unsupported.
    pub fn set_fapl_hdfs(&mut self, config: HdfsConfig) {
        self.driver = "hdfs".to_string();
        self.hdfs_config = Some(config);
    }

    /// Parse and retain the HDFS FAPL config buffer as property-list state.
    /// Runtime I/O remains unsupported.
    pub fn set_fapl_hdfs_from_fapl_config_image(&mut self, bytes: &[u8]) -> crate::Result<()> {
        let decoded = H5P__decode_hdfs_fapl_config(bytes)?;
        let config = HdfsConfig {
            namenode_name: decoded.namenode_name,
            namenode_port: decoded.namenode_port,
            user_name: decoded.user_name,
            buffer_size: decoded.buffer_size,
        };
        if !H5FD__hdfs_validate_config(&config) {
            return Err(crate::Error::InvalidFormat(
                "invalid HDFS VFD config".into(),
            ));
        }
        self.set_fapl_hdfs(config);
        Ok(())
    }

    /// Select the ROS3 VFD and retain its driver-specific config as
    /// property-list state. Runtime I/O remains unsupported.
    pub fn set_fapl_ros3(&mut self, config: Ros3Config) {
        self.driver = "ros3".to_string();
        self.ros3_config = Some(config);
        if self.page_buffer_size.0 == 0 {
            self.page_buffer_size.0 = ROS3_DEFAULT_PAGE_BUFFER_SIZE;
        }
    }

    /// Parse and retain the ROS3 FAPL config buffer as property-list state.
    /// Runtime I/O remains unsupported.
    pub fn set_fapl_ros3_from_fapl_config_image(&mut self, bytes: &[u8]) -> crate::Result<()> {
        let decoded = H5P__decode_ros3_fapl_config(bytes)?;
        let config = Ros3Config {
            endpoint: decoded.endpoint,
            region: decoded.region,
            token: decoded.token,
        };
        if !H5FD__ros3_validate_config(&config) {
            return Err(crate::Error::InvalidFormat(
                "invalid ROS3 VFD config".into(),
            ));
        }
        self.set_fapl_ros3(config);
        Ok(())
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
    pub fn fapl_hdfs(&self) -> Option<&HdfsConfig> {
        self.hdfs_config.as_ref()
    }

    /// Direct VFD configuration. Not active for the direct reader.
    pub fn fapl_direct(&self) -> Option<&DirectFileConfig> {
        self.direct_config.as_ref()
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
    pub fn fapl_family(&self) -> Option<&FamilyFileConfig> {
        self.family_config.as_ref()
    }

    /// Family VFD member offset. Not active for the direct reader.
    pub fn family_offset(&self) -> Option<u64> {
        self.family_offset
    }

    /// Multi VFD memory type. Not active for the direct reader.
    pub fn multi_type(&self) -> Option<u32> {
        self.multi_type
    }

    /// IOC VFD configuration. Not active for the direct reader.
    pub fn fapl_ioc(&self) -> Option<&IocConfig> {
        self.ioc_config.as_ref()
    }

    /// Subfiling VFD configuration. Not active for the direct reader.
    pub fn fapl_subfiling(&self) -> Option<&SubfilingConfig> {
        self.subfiling_config.as_ref()
    }

    /// Splitter VFD configuration. Not active for the direct reader.
    pub fn fapl_splitter(&self) -> Option<&SplitterFileConfig> {
        self.splitter_config.as_ref()
    }

    /// Log VFD configuration. Not active for the direct reader.
    pub fn fapl_log(&self) -> Option<&LogFileConfig> {
        self.log_config.as_ref()
    }

    /// Legacy multi VFD configuration. Not active for the direct reader.
    pub fn fapl_multi(&self) -> Option<&MultiFileConfig> {
        self.multi_config.as_ref()
    }

    /// Onion VFD configuration. Not active for the direct reader.
    pub fn fapl_onion(&self) -> Option<&OnionHeader> {
        self.onion_config.as_ref()
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
    pub fn fapl_core(&self) -> Option<&CoreFileConfig> {
        self.core_config.as_ref()
    }

    /// ROS3 VFD configuration. Not active for the direct reader.
    pub fn fapl_ros3(&self) -> Option<&Ros3Config> {
        self.ros3_config.as_ref()
    }

    /// ROS3 endpoint string. Not active for the direct reader.
    pub fn fapl_ros3_endpoint(&self) -> Option<&str> {
        self.ros3_config
            .as_ref()
            .and_then(|config| config.endpoint.as_deref())
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
