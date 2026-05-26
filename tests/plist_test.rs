use hdf5_pure_rust::{
    DataTransfer, DatasetAccess, DatasetSpaceStatus, File, FileBuilder, FileCloseDegree,
    FileSpaceStrategy, LibverBound, LinkAccess, MetadataCacheConfig, MetadataCacheImageConfig,
    MetadataCacheLogOptions, ObjectCopy, ObjectCreate, VdsView,
};

const V0_BASE_ADDR_OFFSET: usize = 24;

fn userblock_v0_fixture() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("plist_userblock_v0.h5");
    let mut original = std::fs::read("tests/data/simple_v0.h5").unwrap();
    original[V0_BASE_ADDR_OFFSET..V0_BASE_ADDR_OFFSET + 8].copy_from_slice(&512u64.to_le_bytes());

    let mut with_userblock = vec![0u8; 512];
    with_userblock[..b"plist test userblock\0".len()].copy_from_slice(b"plist test userblock\0");
    with_userblock.extend_from_slice(&original);
    std::fs::write(&path, with_userblock).unwrap();
    (dir, path)
}

#[test]
fn test_dataset_create_plist_contiguous() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("float64_1d").unwrap();
    let plist = ds.create_plist().unwrap();

    assert!(!plist.is_chunked());
    assert!(!plist.is_compressed());
    assert!(!plist.has_shuffle());
    assert_eq!(plist.deflate_level(), None);
}

#[test]
fn test_dataset_create_plist_chunked_compressed() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("chunked").unwrap();
    let plist = ds.create_plist().unwrap();

    assert!(plist.is_chunked());
    assert!(plist.is_compressed());
    assert_eq!(plist.deflate_level(), Some(1));
    assert_eq!(plist.filter_count(), 1);
    assert_eq!(plist.filter(0).map(|filter| filter.id), Some(1));
    assert_eq!(
        plist.filter_by_id(1).map(|filter| filter.params.as_slice()),
        Some(&[1][..])
    );
    assert!(plist.chunk_dims.is_some());
    assert_eq!(plist.chunk_dims.as_ref().unwrap(), &[10]);
    assert_eq!(ds.space_status().unwrap(), DatasetSpaceStatus::Allocated);
    assert_eq!(ds.num_chunks().unwrap(), 10);
    let mut chunks = Vec::new();
    ds.chunk_infos_into(&mut chunks).unwrap();
    let first_chunk = chunks.first().unwrap();
    assert_eq!(first_chunk.offset, vec![0]);
    assert!(first_chunk.addr > 0);
    assert!(first_chunk.size > 0);
    assert_eq!(chunks.len(), 10);
}

#[test]
fn test_dataset_create_plist_fill_value() {
    let f = File::open("tests/data/hdf5_ref/sparse_chunked_fill_value.h5").unwrap();
    let ds = f.dataset("sparse_chunked_fill").unwrap();
    let plist = ds.create_plist().unwrap();

    assert!(plist.fill_value_defined);
    assert_eq!(plist.fill_value, Some((-7i32).to_le_bytes().to_vec()));
}

#[test]
fn test_dataset_create_plist_setters() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("float64_1d").unwrap();
    let mut plist = ds.create_plist().unwrap();

    plist.set_chunk_opts(Some(3));
    assert_eq!(plist.chunk_opts(), Some(3));
    plist.set_external("raw.bin", 12, 34);
    assert_eq!(plist.external_count(), 1);
    assert_eq!(plist.external(0).unwrap().name, "raw.bin");
    plist.set_filter(32000, "custom", 1, vec![7, 8]);
    assert_eq!(plist.filter_by_id(32000).unwrap().params, vec![7, 8]);
    plist.set_shuffle();
    plist.set_nbit();
    plist.set_scaleoffset(vec![0, 2]);
    plist.set_fletcher32();
    plist.set_szip(vec![8, 16]);
    assert!(plist.has_shuffle());
    assert!(plist.filter_by_id(3).is_some());
    assert!(plist.filter_by_id(4).is_some());
    assert!(plist.filter_by_id(5).is_some());
    assert!(plist.filter_by_id(6).is_some());
    plist.set_virtual_spatial_tree(true);
    assert!(plist.virtual_spatial_tree());
    plist.set_alloc_time(2);
    assert_eq!(plist.fill_alloc_time, Some(2));
    plist.set_fill_value(Some(vec![1, 2, 3]));
    assert!(plist.fill_value_defined);
    assert_eq!(plist.fill_value, Some(vec![1, 2, 3]));
}

#[test]
fn test_file_create_plist() {
    use hdf5_pure_rust::hl::plist::file_create::FileCreate;
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let mut plist = FileCreate::from_file(&f);
    assert_eq!(plist.superblock_version, 0);
    assert_eq!(plist.sizeof_addr, 8);
    assert_eq!(plist.sizeof_size, 8);
    assert_eq!(plist.sym_leaf_k, 4);
    assert_eq!(plist.btree_k, 16);
    assert_eq!(plist.sizes(), (8, 8));
    assert_eq!(plist.sym_k(), (16, 4));
    assert_eq!(plist.istore_k(), 32);
    assert_eq!(plist.userblock(), 0);
    assert_eq!(plist.file_space(), (FileSpaceStrategy::Aggregate, false, 1));
    assert_eq!(plist.file_space_page_size(), 4096);
    assert_eq!(plist.shared_mesg_nindexes(), 0);
    assert_eq!(plist.shared_mesg_index(0), None);
    assert_eq!(plist.shared_mesg_phase_change(), (50, 40));

    assert!(plist.set_superblock_version(2));
    assert_eq!(plist.superblock_version, 2);
    assert!(!plist.set_superblock_version(1));
    assert_eq!(plist.superblock_version, 2);
    assert!(!plist.set_superblock_version(4));
    assert_eq!(plist.superblock_version, 2);
    assert!(plist.set_superblock_version(3));
    assert_eq!(plist.superblock_version, 3);
    assert!(plist.set_sizes(4, 8));
    assert_eq!(plist.sizes(), (4, 8));
    assert!(!plist.set_sizes(1, 8));
    assert_eq!(plist.sizes(), (4, 8));
    assert!(!plist.set_sizes(4, 16));
    assert_eq!(plist.sizes(), (4, 8));
    plist.set_sym_k(32, 8);
    assert_eq!(plist.sym_k(), (32, 8));
    plist.set_istore_k(64);
    assert_eq!(plist.istore_k(), 64);
    assert!(plist.set_userblock(512));
    assert_eq!(plist.userblock(), 512);
    assert!(!plist.set_userblock(256));
    assert_eq!(plist.userblock(), 512);
    assert!(!plist.set_userblock(768));
    assert_eq!(plist.userblock(), 512);
    assert!(plist.set_userblock(0));
    assert_eq!(plist.userblock(), 0);
    plist.set_file_space(FileSpaceStrategy::Page, true, 16);
    assert_eq!(plist.file_space(), (FileSpaceStrategy::Page, true, 16));
    assert!(plist.set_file_space_page_size(8192));
    assert_eq!(plist.file_space_page_size(), 8192);
    assert!(!plist.set_file_space_page_size(0));
    assert_eq!(plist.file_space_page_size(), 8192);
    assert!(!plist.set_file_space_page_size(1024 * 1024 * 1024 + 1));
    assert_eq!(plist.file_space_page_size(), 8192);
    plist.set_shared_mesg_nindexes(1);
    assert_eq!(plist.shared_mesg_nindexes(), 1);
    plist.set_shared_mesg_index(
        0,
        hdf5_pure_rust::SharedMessageIndex {
            message_type_flags: 2,
            minimum_message_size: 32,
        },
    );
    assert_eq!(plist.shared_mesg_index(0).unwrap().minimum_message_size, 32);
    plist.set_shared_mesg_phase_change(60, 30);
    assert_eq!(plist.shared_mesg_phase_change(), (60, 30));
}

#[test]
fn test_file_builder_create_plist_preserves_property_state() {
    use hdf5_pure_rust::hl::plist::file_create::{FileCreate, SharedMessageIndex};

    let mut fcpl = FileCreate::new();
    assert!(fcpl.set_sizes(4, 8));
    fcpl.set_sym_k(32, 8);
    fcpl.set_istore_k(64);
    assert!(fcpl.set_userblock(1024));
    fcpl.set_file_space(FileSpaceStrategy::Page, true, 16);
    assert!(fcpl.set_file_space_page_size(8192));
    fcpl.set_shared_mesg_nindexes(1);
    assert!(fcpl.set_shared_mesg_index(
        0,
        SharedMessageIndex {
            message_type_flags: 2,
            minimum_message_size: 32,
        },
    ));

    let mut builder = FileBuilder::new();
    builder.set_create_plist(&fcpl).unwrap();
    assert_eq!(builder.fcpl().superblock_version, 3);
    assert_eq!(builder.fcpl().sizes(), (4, 8));
    assert_eq!(builder.fcpl().sym_k(), (32, 8));
    assert_eq!(builder.fcpl().istore_k(), 64);
    assert_eq!(builder.fcpl().userblock(), 1024);
    assert_eq!(
        builder.fcpl().file_space(),
        (FileSpaceStrategy::Page, true, 16)
    );
    assert_eq!(builder.fcpl().file_space_page_size(), 8192);
    assert_eq!(
        builder
            .fcpl()
            .shared_mesg_index(0)
            .unwrap()
            .minimum_message_size,
        32
    );

    builder.with_create_plist(|plist| {
        plist.set_sym_k(48, 12);
        plist.set_shared_mesg_phase_change(70, 20);
        plist
    });
    assert_eq!(builder.create_plist().sym_k(), (48, 12));
    assert_eq!(builder.create_plist().shared_mesg_phase_change(), (70, 20));
}

#[test]
fn test_file_builder_applies_fcpl_userblock_on_create() {
    use hdf5_pure_rust::hl::plist::file_create::FileCreate;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("builder_userblock.h5");

    let mut fcpl = FileCreate::new();
    assert!(fcpl.set_userblock(1024));
    let mut builder = FileBuilder::new();
    builder.set_create_plist(&fcpl).unwrap();
    let file = builder.create(&path).unwrap();

    assert_eq!(file.userblock(), 1024);
    assert_eq!(file.create_plist().userblock(), 1024);
    assert_eq!(std::fs::read(&path).unwrap()[..1024], vec![0u8; 1024]);

    let reopened = File::open(&path).unwrap();
    assert_eq!(reopened.userblock(), 1024);
    assert!(reopened
        .root_group()
        .unwrap()
        .member_names()
        .unwrap()
        .is_empty());
}

#[test]
fn test_file_builder_applies_fcpl_size_widths_on_create() {
    use hdf5_pure_rust::hl::plist::file_create::FileCreate;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("builder_size_widths.h5");

    let mut fcpl = FileCreate::new();
    assert!(fcpl.set_sizes(4, 4));
    let mut builder = FileBuilder::new();
    builder.set_create_plist(&fcpl).unwrap();
    let file = builder.create(&path).unwrap();

    assert_eq!(file.create_plist().sizes(), (4, 4));

    let reopened = File::open(&path).unwrap();
    assert_eq!(reopened.superblock().sizeof_addr, 4);
    assert_eq!(reopened.superblock().sizeof_size, 4);
    assert!(reopened
        .root_group()
        .unwrap()
        .member_names()
        .unwrap()
        .is_empty());
}

#[test]
fn test_file_builder_applies_fcpl_superblock_version_on_create() {
    use hdf5_pure_rust::hl::plist::file_create::FileCreate;

    let dir = tempfile::tempdir().unwrap();
    let path_v2 = dir.path().join("builder_superblock_v2.h5");
    let path_v3 = dir.path().join("builder_superblock_v3.h5");

    let mut fcpl = FileCreate::new();
    assert!(fcpl.set_superblock_version(2));
    let mut builder = FileBuilder::new();
    builder.set_create_plist(&fcpl).unwrap();
    let file = builder.create(&path_v2).unwrap();
    assert_eq!(file.superblock().version, 2);
    assert_eq!(File::open(&path_v2).unwrap().superblock().version, 2);

    assert!(fcpl.set_superblock_version(3));
    builder.set_create_plist(&fcpl).unwrap();
    let file = builder.create(&path_v3).unwrap();
    assert_eq!(file.superblock().version, 3);
    assert_eq!(File::open(&path_v3).unwrap().superblock().version, 3);
}

#[test]
fn test_file_builder_applies_fcpl_file_space_info_on_create() {
    use hdf5_pure_rust::hl::plist::file_create::FileCreate;
    use hdf5_pure_rust::io::reader::UNDEF_ADDR;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("builder_file_space_info.h5");

    let mut fcpl = FileCreate::new();
    fcpl.set_file_space(FileSpaceStrategy::Page, true, 64);
    assert!(fcpl.set_file_space_page_size(8192));
    let mut builder = FileBuilder::new();
    builder.set_create_plist(&fcpl).unwrap();
    let file = builder.create(&path).unwrap();

    assert_ne!(file.superblock().ext_addr, UNDEF_ADDR);
    assert_eq!(
        file.create_plist().file_space(),
        (FileSpaceStrategy::Page, true, 64)
    );
    assert_eq!(file.create_plist().file_space_page_size(), 8192);

    let reopened = File::open(&path).unwrap();
    assert_ne!(reopened.superblock().ext_addr, UNDEF_ADDR);
    assert_eq!(
        reopened.create_plist().file_space(),
        (FileSpaceStrategy::Page, true, 64)
    );
    assert_eq!(reopened.create_plist().file_space_page_size(), 8192);
}

#[test]
fn test_file_builder_applies_fcpl_shared_message_info_on_create() {
    use hdf5_pure_rust::hl::plist::file_create::{FileCreate, SharedMessageIndex};
    use hdf5_pure_rust::io::reader::UNDEF_ADDR;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("builder_shared_message_info.h5");

    let mut fcpl = FileCreate::new();
    fcpl.set_shared_mesg_nindexes(2);
    assert!(fcpl.set_shared_mesg_index(
        0,
        SharedMessageIndex {
            message_type_flags: 1 << 3,
            minimum_message_size: 32,
        },
    ));
    assert!(fcpl.set_shared_mesg_index(
        1,
        SharedMessageIndex {
            message_type_flags: 1 << 12,
            minimum_message_size: 64,
        },
    ));
    fcpl.set_shared_mesg_phase_change(60, 30);

    let mut builder = FileBuilder::new();
    builder.set_create_plist(&fcpl).unwrap();
    let file = builder.create(&path).unwrap();

    assert_ne!(file.superblock().ext_addr, UNDEF_ADDR);
    let plist = file.create_plist();
    assert_eq!(plist.shared_mesg_nindexes(), 2);
    assert_eq!(
        plist.shared_mesg_index(0),
        Some(SharedMessageIndex {
            message_type_flags: 1 << 3,
            minimum_message_size: 32,
        })
    );
    assert_eq!(
        plist.shared_mesg_index(1),
        Some(SharedMessageIndex {
            message_type_flags: 1 << 12,
            minimum_message_size: 64,
        })
    );
    assert_eq!(plist.shared_mesg_phase_change(), (60, 30));

    let image = file.file_image().unwrap();
    assert!(image.windows(4).any(|window| window == b"SMTB"));

    let reopened = File::open(&path).unwrap();
    let reopened_plist = reopened.create_plist();
    assert_eq!(reopened_plist.shared_mesg_nindexes(), 2);
    assert_eq!(
        reopened_plist.shared_mesg_index(0).unwrap(),
        SharedMessageIndex {
            message_type_flags: 1 << 3,
            minimum_message_size: 32,
        }
    );
    assert_eq!(
        reopened_plist.shared_mesg_index(1).unwrap(),
        SharedMessageIndex {
            message_type_flags: 1 << 12,
            minimum_message_size: 64,
        }
    );
    assert_eq!(reopened_plist.shared_mesg_phase_change(), (60, 30));
}

#[test]
fn test_file_access_plist_reports_opened_file_userblock() {
    let (_dir, path) = userblock_v0_fixture();
    let f = File::open(&path).unwrap();

    assert_eq!(f.userblock(), 512);
    assert_eq!(f.create_plist().userblock(), 512);
    assert_eq!(f.fcpl().userblock(), 512);
    assert_eq!(f.access_plist().userblock(), 512);
    assert_eq!(f.fapl().userblock(), 512);
}

#[test]
fn test_file_access_driver_switch_clears_stale_driver_config() {
    use hdf5_pure_rust::engine::vfd::{
        DirectFileConfig, FamilyFileConfig, H5FD_multi_populate_config, Ros3Config,
    };
    use hdf5_pure_rust::hl::plist::file_access::FileAccess;

    let mut fapl = FileAccess::default();

    fapl.set_fapl_family(FamilyFileConfig {
        member_size: 8192,
        printf_filename: "family-%05d.h5".into(),
    });
    fapl.set_family_offset(Some(128));
    fapl.set_core_write_tracking(true);
    fapl.set_map_iterate_hints(Some("opaque-map-hints"));
    assert_eq!(fapl.driver(), "family");
    assert_eq!(fapl.fapl_family().unwrap().member_size, 8192);
    assert_eq!(fapl.family_offset(), Some(128));
    assert!(fapl.core_write_tracking());
    assert_eq!(fapl.map_iterate_hints_ref(), Some("opaque-map-hints"));

    fapl.set_fapl_ros3(Ros3Config {
        endpoint: Some("https://s3.example.test".into()),
        region: Some("eu-north-1".into()),
        token: None,
    });
    assert_eq!(fapl.driver(), "ros3");
    assert!(fapl.fapl_family().is_none());
    assert_eq!(fapl.family_offset(), None);
    assert!(!fapl.core_write_tracking());
    assert_eq!(fapl.map_iterate_hints_ref(), Some("opaque-map-hints"));
    assert_eq!(fapl.fapl_ros3_endpoint(), Some("https://s3.example.test"));
    assert_eq!(fapl.page_buffer_size().0, 64 * 1024 * 1024);

    fapl.set_fapl_direct(DirectFileConfig::default());
    assert_eq!(fapl.driver(), "direct");
    assert!(fapl.fapl_ros3().is_none());
    assert!(fapl.fapl_direct().is_some());
    assert_eq!(fapl.page_buffer_size(), (0, 0, 0));
    assert_eq!(fapl.map_iterate_hints_ref(), Some("opaque-map-hints"));

    fapl.set_fapl_multi(H5FD_multi_populate_config());
    fapl.set_multi_type(Some(7));
    assert!(fapl.fapl_multi().is_some());
    assert_eq!(fapl.multi_type(), Some(7));

    fapl.set_fapl_sec2();
    assert_eq!(fapl.driver(), "sec2");
    assert!(fapl.fapl_multi().is_none());
    assert_eq!(fapl.multi_type(), None);

    fapl.set_fapl_family(FamilyFileConfig::default());
    fapl.set_driver_by_value(42);
    assert_eq!(fapl.driver(), "driver_42");
    assert!(fapl.fapl_family().is_none());
}

#[test]
fn test_file_access_ros3_default_page_buffer_does_not_clobber_explicit_size() {
    use hdf5_pure_rust::engine::vfd::Ros3Config;
    use hdf5_pure_rust::hl::plist::file_access::FileAccess;

    let mut fapl = FileAccess::default();
    assert!(fapl.set_page_buffer_size(1024, 20, 30));
    fapl.set_fapl_ros3(Ros3Config {
        endpoint: None,
        region: None,
        token: None,
    });
    assert_eq!(fapl.driver(), "ros3");
    assert_eq!(fapl.page_buffer_size(), (1024, 20, 30));

    fapl.set_fapl_sec2();
    assert_eq!(fapl.driver(), "sec2");
    assert_eq!(fapl.page_buffer_size(), (1024, 20, 30));
}

#[test]
fn test_file_access_userblock_validation_preserves_previous_value() {
    use hdf5_pure_rust::hl::plist::file_access::FileAccess;

    let mut fapl = FileAccess::default();
    assert_eq!(fapl.userblock(), 0);
    assert!(fapl.set_userblock(512));
    assert_eq!(fapl.userblock(), 512);
    assert!(!fapl.set_userblock(256));
    assert_eq!(fapl.userblock(), 512);
    assert!(!fapl.set_userblock(768));
    assert_eq!(fapl.userblock(), 512);
    assert!(fapl.set_userblock(0));
    assert_eq!(fapl.userblock(), 0);
}

#[test]
fn test_file_access_backend_effect_validation_preserves_previous_value() {
    use hdf5_pure_rust::hl::plist::file_access::FileAccess;

    let mut fapl = FileAccess::default();
    assert_eq!(fapl.alignment(), (1, 1));
    assert!(fapl.set_alignment(512, 4096));
    assert_eq!(fapl.alignment(), (512, 4096));
    assert!(!fapl.set_alignment(1024, 0));
    assert_eq!(fapl.alignment(), (512, 4096));

    assert_eq!(fapl.page_buffer_size(), (0, 0, 0));
    assert!(fapl.set_page_buffer_size(4096, 40, 50));
    assert_eq!(fapl.page_buffer_size(), (4096, 40, 50));
    assert!(!fapl.set_page_buffer_size(4096, 75, 50));
    assert_eq!(fapl.page_buffer_size(), (4096, 40, 50));
    assert!(!fapl.set_page_buffer_size(0, 10, 0));
    assert_eq!(fapl.page_buffer_size(), (4096, 40, 50));
    assert!(fapl.set_page_buffer_size(0, 0, 0));
    assert_eq!(fapl.page_buffer_size(), (0, 0, 0));
}

#[test]
fn test_file_builder_enforces_runtime_supported_fapl_and_preserves_state() {
    use hdf5_pure_rust::engine::vfd::FamilyFileConfig;
    use hdf5_pure_rust::hl::plist::file_access::FileAccess;

    let dir = tempfile::tempdir().unwrap();
    let unsupported_path = dir.path().join("family-driver-should-not-exist.h5");

    let mut unsupported = FileAccess::default();
    unsupported.set_fapl_family(FamilyFileConfig {
        member_size: 4096,
        printf_filename: "family-%05d.h5".into(),
    });
    let mut builder = FileBuilder::new();
    builder.set_access_plist(&unsupported).unwrap();

    let err = match builder.create(&unsupported_path) {
        Ok(_) => panic!("family driver unexpectedly created a file"),
        Err(err) => err,
    };
    assert!(err
        .to_string()
        .contains("file access driver 'family' is not supported"));
    assert!(!unsupported_path.exists());

    let supported_path = dir.path().join("stdio-preserves-fapl-state.h5");
    let mut supported = FileAccess::default();
    supported.set_fapl_stdio();
    assert!(supported.set_alignment(512, 4096));
    supported.set_cache(3, 7, 65_536, 0.5);
    supported.set_gc_references(true);
    supported.set_libver_bounds(LibverBound::V110, LibverBound::V114);
    supported.set_file_locking(false, true);
    assert!(supported.set_page_buffer_size(2048, 25, 75));
    supported.set_map_iterate_hints(Some("builder-open-state"));
    supported.set_object_flush_cb(true);

    let mut builder = FileBuilder::new();
    builder.set_access_plist(&supported).unwrap();
    let file = builder.create(&supported_path).unwrap();
    let opened = file.access_plist();

    assert_eq!(opened.driver(), "stdio");
    assert_eq!(opened.alignment(), (512, 4096));
    assert_eq!(opened.cache(), (3, 7, 65_536, 0.5));
    assert!(opened.gc_references());
    assert_eq!(
        opened.libver_bounds(),
        (LibverBound::V110, LibverBound::V114)
    );
    assert_eq!(opened.file_locking(), (false, true));
    assert_eq!(opened.page_buffer_size(), (2048, 25, 75));
    assert_eq!(opened.map_iterate_hints_ref(), Some("builder-open-state"));
    assert_eq!(opened.object_flush_cb(), Some(()));

    let reopened_rw = builder.open_rw(&supported_path).unwrap();
    let reopened_rw_fapl = reopened_rw.access_plist();
    assert_eq!(reopened_rw_fapl.driver(), "stdio");
    assert_eq!(reopened_rw_fapl.alignment(), (512, 4096));
    assert_eq!(reopened_rw_fapl.cache(), (3, 7, 65_536, 0.5));
    assert!(reopened_rw_fapl.gc_references());
    assert_eq!(reopened_rw_fapl.file_locking(), (false, true));
    assert_eq!(reopened_rw_fapl.page_buffer_size(), (2048, 25, 75));
    assert_eq!(
        reopened_rw_fapl.map_iterate_hints_ref(),
        Some("builder-open-state")
    );
    assert_eq!(reopened_rw_fapl.object_flush_cb(), Some(()));

    let appended_existing = builder.append(&supported_path).unwrap();
    let appended_fapl = appended_existing.access_plist();
    assert_eq!(appended_fapl.driver(), "stdio");
    assert_eq!(appended_fapl.file_locking(), (false, true));
    assert_eq!(
        appended_fapl.map_iterate_hints_ref(),
        Some("builder-open-state")
    );
    assert_eq!(appended_fapl.object_flush_cb(), Some(()));
}

#[test]
fn test_file_builder_rejects_unsupported_fapl_before_open_or_append() {
    use hdf5_pure_rust::engine::vfd::FamilyFileConfig;
    use hdf5_pure_rust::hl::plist::file_access::FileAccess;

    let dir = tempfile::tempdir().unwrap();
    let existing_path = dir.path().join("existing.h5");
    File::create(&existing_path).unwrap();
    let missing_path = dir.path().join("append-should-not-create.h5");

    let mut unsupported = FileAccess::default();
    unsupported.set_fapl_family(FamilyFileConfig {
        member_size: 4096,
        printf_filename: "family-%05d.h5".into(),
    });
    let mut builder = FileBuilder::new();
    builder.set_access_plist(&unsupported).unwrap();

    let open_err = match builder.open(&existing_path) {
        Ok(_) => panic!("family driver unexpectedly opened an existing file"),
        Err(err) => err,
    };
    assert!(open_err
        .to_string()
        .contains("file access driver 'family' is not supported"));

    let append_err = match builder.append(&missing_path) {
        Ok(_) => panic!("family driver unexpectedly appended a missing file"),
        Err(err) => err,
    };
    assert!(append_err
        .to_string()
        .contains("file access driver 'family' is not supported"));
    assert!(!missing_path.exists());
}

#[test]
fn test_property_list_fapl_driver_switch_clears_stale_config_properties() {
    use hdf5_pure_rust::engine::property::{
        H5P__create, H5P_peek_ref, H5P_set_driver_by_name, H5Pcreate_class,
        H5Pget_fapl_hdfs_config, H5Pget_fapl_ros3_config, H5Pset_fapl_hdfs_config,
        H5Pset_fapl_ros3_config, HdfsFaplConfig, Ros3FaplConfig,
    };

    let class = H5Pcreate_class("file_access", None);
    let mut list = H5P__create(&class).unwrap();

    let hdfs = HdfsFaplConfig {
        namenode_name: "nn.example.org".into(),
        namenode_port: 8020,
        user_name: "reader".into(),
        buffer_size: 4096,
    };
    H5Pset_fapl_hdfs_config(&mut list, hdfs.clone()).unwrap();
    assert_eq!(H5P_peek_ref(&list, "driver").unwrap(), b"hdfs");
    assert_eq!(H5Pget_fapl_hdfs_config(&list).unwrap(), Some(hdfs));
    assert_eq!(H5Pget_fapl_ros3_config(&list).unwrap(), None);

    let ros3 = Ros3FaplConfig {
        endpoint: Some("https://s3.example.test".into()),
        region: Some("eu-north-1".into()),
        token: Some("token".into()),
    };
    H5Pset_fapl_ros3_config(&mut list, ros3.clone()).unwrap();
    assert_eq!(H5P_peek_ref(&list, "driver").unwrap(), b"ros3");
    assert_eq!(H5Pget_fapl_hdfs_config(&list).unwrap(), None);
    assert_eq!(H5Pget_fapl_ros3_config(&list).unwrap(), Some(ros3));

    H5P_set_driver_by_name(&mut list, "stdio").unwrap();
    assert_eq!(H5P_peek_ref(&list, "driver").unwrap(), b"stdio");
    assert_eq!(H5Pget_fapl_hdfs_config(&list).unwrap(), None);
    assert_eq!(H5Pget_fapl_ros3_config(&list).unwrap(), None);
}

#[test]
fn test_misc_property_list_defaults() {
    let mut dxpl = DataTransfer::new();
    assert_eq!(dxpl.buffer(), (0, None, None));
    assert!(!dxpl.preserve());
    dxpl.set_buffer(4096);
    dxpl.set_preserve(true);
    assert_eq!(dxpl.buffer(), (4096, None, None));
    assert!(dxpl.preserve());
    assert_eq!(dxpl.type_conv_cb(), None);
    assert_eq!(dxpl.vlen_mem_manager(), (None, None));
    assert_eq!(dxpl.mpio_actual_chunk_opt_mode(), None);
    assert_eq!(dxpl.mpio_actual_io_mode(), None);
    assert_eq!(dxpl.mpio_no_collective_cause(), (0, 0));
    dxpl.set_data_transform(Some("x + 1"));
    dxpl.set_edc_check(false);
    dxpl.set_filter_callback(true);
    dxpl.set_type_conv_cb(true);
    dxpl.set_vlen_mem_manager(true);
    dxpl.set_btree_ratios(0.2, 0.4, 0.8);
    dxpl.set_hyper_vector_size(32);
    dxpl.set_modify_write_buf(true);
    assert_eq!(dxpl.data_transform(), Some("x + 1"));
    assert!(!dxpl.edc_check());
    assert_eq!(dxpl.filter_callback(), Some(()));
    assert_eq!(dxpl.type_conv_cb(), Some(()));
    assert_eq!(dxpl.vlen_mem_manager(), (Some(()), Some(())));
    assert_eq!(dxpl.btree_ratios(), (0.2, 0.4, 0.8));
    assert_eq!(dxpl.hyper_vector_size(), 32);
    assert!(dxpl.modify_write_buf());

    let mut ocpy = ObjectCopy::new();
    assert_eq!(ocpy.copy_object(), 0);
    ocpy.set_copy_object(0x3);
    assert_eq!(ocpy.copy_object(), 0x3);
    assert_eq!(ocpy.mcdt_search_cb(), None);
    ocpy.set_mcdt_search_cb(true);
    assert_eq!(ocpy.mcdt_search_cb(), Some(()));

    let mut ocrt = ObjectCreate::new();
    assert!(ocrt.obj_track_times());
    ocrt.set_obj_track_times(false);
    assert!(!ocrt.obj_track_times());
    ocrt.set_attr_phase_change(12, 4);
    ocrt.set_attr_creation_order(3);
    ocrt.set_local_heap_size_hint(4096);
    ocrt.set_link_phase_change(10, 3);
    ocrt.set_est_link_info(12, 24);
    assert_eq!(ocrt.attr_phase_change(), (12, 4));
    assert_eq!(ocrt.attr_creation_order(), 3);
    assert_eq!(ocrt.local_heap_size_hint(), 4096);
    assert_eq!(ocrt.link_phase_change(), (10, 3));
    assert_eq!(ocrt.est_link_info(), (12, 24));

    let mut lapl = LinkAccess::new();
    assert_eq!(lapl.nlinks(), 40);
    lapl.set_nlinks(8);
    lapl.set_elink_prefix(Some("/tmp/hdf5-links"));
    lapl.set_elink_acc_flags(7);
    lapl.set_elink_cb(true);
    lapl.set_elink_file_cache_size(5);
    assert_eq!(lapl.nlinks(), 8);
    assert_eq!(lapl.elink_prefix(), Some("/tmp/hdf5-links"));
    assert_eq!(lapl.elink_acc_flags(), 7);
    assert_eq!(lapl.elink_cb(), Some(()));
    assert_eq!(lapl.elink_file_cache_size(), 5);

    let mut dapl = DatasetAccess::new();
    dapl.set_virtual_view(VdsView::FirstMissing);
    dapl.set_virtual_prefix(Some("/tmp/vds"));
    dapl.set_append_flush(true);
    dapl.set_efile_prefix(Some("/tmp/external"));
    assert_eq!(dapl.virtual_view(), VdsView::FirstMissing);
    assert_eq!(dapl.virtual_prefix().unwrap().to_string_lossy(), "/tmp/vds");
    assert!(dapl.append_flush());
    assert_eq!(
        dapl.efile_prefix().unwrap().to_string_lossy(),
        "/tmp/external"
    );

    let mut fapl = hdf5_pure_rust::hl::plist::file_access::FileAccess::default();
    fapl.set_driver("custom");
    assert_eq!(fapl.driver(), "custom");
    fapl.set_driver_by_name("named");
    assert_eq!(fapl.driver(), "named");
    fapl.set_driver_by_value(42);
    assert_eq!(fapl.driver(), "driver_42");
    fapl.set_fapl_sec2();
    assert_eq!(fapl.driver(), "sec2");
    fapl.set_fapl_windows();
    assert_eq!(fapl.driver(), "windows");
    fapl.set_fapl_stdio();
    assert_eq!(fapl.driver(), "stdio");
    fapl.set_userblock(512);
    assert!(fapl.set_alignment(128, 4096));
    fapl.set_cache(1, 2, 3, 0.25);
    fapl.set_gc_references(true);
    fapl.set_fclose_degree(FileCloseDegree::Strong);
    fapl.set_meta_block_size(77);
    fapl.set_sieve_buf_size(88);
    fapl.set_small_data_block_size(99);
    fapl.set_libver_bounds(LibverBound::V110, LibverBound::V114);
    fapl.set_evict_on_close(true);
    fapl.set_file_locking(false, true);
    fapl.set_mdc_config(MetadataCacheConfig {
        enabled: true,
        max_size: 10,
        min_clean_size: 2,
    });
    fapl.set_mdc_image_config(MetadataCacheImageConfig {
        enabled: true,
        generation_enabled: true,
    });
    fapl.set_mdc_log_options(MetadataCacheLogOptions {
        enabled: true,
        location: Some("mdc.log".into()),
        start_on_access: true,
    });
    fapl.set_all_coll_metadata_ops(true);
    fapl.set_coll_metadata_write(true);
    assert!(fapl.set_page_buffer_size(1024, 20, 30));
    fapl.set_core_write_tracking(true);
    fapl.set_map_iterate_hints(Some("hint"));
    fapl.set_object_flush_cb(true);
    assert_eq!(fapl.userblock(), 512);
    assert_eq!(fapl.alignment(), (128, 4096));
    assert_eq!(fapl.cache(), (1, 2, 3, 0.25));
    assert!(fapl.gc_references());
    assert_eq!(fapl.fclose_degree(), FileCloseDegree::Strong);
    assert_eq!(fapl.meta_block_size(), 77);
    assert_eq!(fapl.sieve_buf_size(), 88);
    assert_eq!(fapl.small_data_block_size(), 99);
    assert_eq!(fapl.libver_bounds(), (LibverBound::V110, LibverBound::V114));
    assert!(fapl.evict_on_close());
    assert_eq!(fapl.file_locking(), (false, true));
    assert!(fapl.mdc_config().enabled);
    assert!(fapl.mdc_image_config().generation_enabled);
    assert_eq!(fapl.mdc_log_options().location.as_deref(), Some("mdc.log"));
    assert!(fapl.all_coll_metadata_ops());
    assert!(fapl.coll_metadata_write());
    assert_eq!(fapl.page_buffer_size(), (1024, 20, 30));
    assert!(fapl.core_write_tracking());
    assert_eq!(fapl.map_iterate_hints(), Some(()));
    assert_eq!(fapl.map_iterate_hints_ref(), Some("hint"));
    assert_eq!(fapl.object_flush_cb(), Some(()));
}

#[test]
fn test_file_access_parses_unsupported_vfd_config_images() {
    use hdf5_pure_rust::engine::property::{
        H5P__encode_hdfs_fapl_config_into, H5P__encode_ros3_fapl_config_into, HdfsFaplConfig,
        Ros3FaplConfig,
    };
    use hdf5_pure_rust::engine::vfd::{
        FamilyFileConfig, H5FD__family_sb_encode_into, H5FD__log_sb_encode_into,
        H5FD__onion_sb_encode_into, H5FD__splitter_sb_encode_into, H5FD__subfiling_sb_encode_into,
        H5FD_multi_populate_config, H5FD_multi_sb_encode_into, HdfsConfig, LogFileConfig,
        OnionHeader, Ros3Config, SplitterFileConfig, SubfilingConfig,
    };
    use hdf5_pure_rust::hl::plist::file_access::FileAccess;
    use std::path::PathBuf;

    let mut access = FileAccess::default();
    let mut bytes = Vec::new();

    let family = FamilyFileConfig {
        member_size: 4096,
        printf_filename: "ignored-by-superblock-image.h5".into(),
    };
    H5FD__family_sb_encode_into(&family, &mut bytes).unwrap();
    access.set_fapl_family_from_config_image(&bytes).unwrap();
    assert_eq!(access.driver(), "family");
    assert_eq!(access.fapl_family().unwrap().member_size, 4096);
    assert!(access.set_fapl_family_from_config_image(&[0; 4]).is_err());
    assert_eq!(access.driver(), "family");
    assert_eq!(access.fapl_family().unwrap().member_size, 4096);

    let multi = H5FD_multi_populate_config();
    bytes.clear();
    H5FD_multi_sb_encode_into(&multi, &mut bytes).unwrap();
    access.set_fapl_multi_from_config_image(&bytes).unwrap();
    assert_eq!(access.fapl_multi(), Some(&multi));
    assert!(access
        .set_fapl_multi_from_config_image(&[1, 0, 0, 0, 99, 0])
        .is_err());
    assert_eq!(access.driver(), "multi");
    assert_eq!(access.fapl_multi(), Some(&multi));

    let splitter = SplitterFileConfig {
        write_only_path: Some(PathBuf::from("mirror.h5")),
        ignore_wo_errors: true,
    };
    bytes.clear();
    H5FD__splitter_sb_encode_into(&splitter, &mut bytes).unwrap();
    access.set_fapl_splitter_from_config_image(&bytes).unwrap();
    assert_eq!(access.fapl_splitter(), Some(&splitter));
    assert!(access.set_fapl_splitter_from_config_image(&[1]).is_err());
    assert_eq!(access.driver(), "splitter");
    assert_eq!(access.fapl_splitter(), Some(&splitter));

    let log = LogFileConfig {
        log_path: Some(PathBuf::from("driver.log")),
        flags: 3,
        buffer_size: 8192,
    };
    bytes.clear();
    H5FD__log_sb_encode_into(&log, &mut bytes).unwrap();
    access.set_fapl_log_from_config_image(&bytes).unwrap();
    assert_eq!(access.fapl_log(), Some(&log));
    assert!(access.set_fapl_log_from_config_image(&[1]).is_err());
    assert_eq!(access.driver(), "log");
    assert_eq!(access.fapl_log(), Some(&log));

    let onion = OnionHeader {
        version: 1,
        flags: 2,
        revision_count: 3,
    };
    bytes.clear();
    H5FD__onion_sb_encode_into(&onion, &mut bytes).unwrap();
    access.set_fapl_onion_from_config_image(&bytes).unwrap();
    assert_eq!(access.fapl_onion(), Some(&onion));
    assert!(access.set_fapl_onion_from_config_image(&[1]).is_err());
    assert_eq!(access.driver(), "onion");
    assert_eq!(access.fapl_onion(), Some(&onion));

    let subfiling = SubfilingConfig {
        ioc_count: 2,
        stripe_size: 1024,
        stripe_count: 8,
    };
    bytes.clear();
    H5FD__subfiling_sb_encode_into(&subfiling, &mut bytes).unwrap();
    access.set_fapl_subfiling_from_config_image(&bytes).unwrap();
    assert_eq!(access.fapl_subfiling(), Some(&subfiling));
    assert!(access.set_fapl_subfiling_from_config_image(&[1]).is_err());
    assert_eq!(access.driver(), "subfiling");
    assert_eq!(access.fapl_subfiling(), Some(&subfiling));

    let hdfs = HdfsFaplConfig {
        namenode_name: "nn.example.org".into(),
        namenode_port: 8020,
        user_name: "reader".into(),
        buffer_size: 4096,
    };
    bytes.clear();
    H5P__encode_hdfs_fapl_config_into(&hdfs, &mut bytes).unwrap();
    access.set_fapl_hdfs_from_fapl_config_image(&bytes).unwrap();
    let expected_hdfs = HdfsConfig {
        namenode_name: "nn.example.org".into(),
        namenode_port: 8020,
        user_name: "reader".into(),
        buffer_size: 4096,
    };
    assert_eq!(access.fapl_hdfs(), Some(&expected_hdfs));
    assert!(access.set_fapl_hdfs_from_fapl_config_image(&[1]).is_err());
    assert_eq!(access.driver(), "hdfs");
    assert_eq!(access.fapl_hdfs(), Some(&expected_hdfs));

    let ros3 = Ros3FaplConfig {
        endpoint: Some("https://s3.us-east-1.amazonaws.com".into()),
        region: Some("us-east-1".into()),
        token: Some("session-token".into()),
    };
    bytes.clear();
    H5P__encode_ros3_fapl_config_into(&ros3, &mut bytes).unwrap();
    access.set_fapl_ros3_from_fapl_config_image(&bytes).unwrap();
    let expected_ros3 = Ros3Config {
        endpoint: Some("https://s3.us-east-1.amazonaws.com".into()),
        region: Some("us-east-1".into()),
        token: Some("session-token".into()),
    };
    assert_eq!(access.fapl_ros3(), Some(&expected_ros3));
    assert_eq!(access.page_buffer_size().0, 64 * 1024 * 1024);
    assert!(access.set_fapl_ros3_from_fapl_config_image(&[1]).is_err());
    assert_eq!(access.driver(), "ros3");
    assert_eq!(access.fapl_ros3(), Some(&expected_ros3));
    assert_eq!(access.page_buffer_size().0, 64 * 1024 * 1024);
    assert!(matches!(
        access.ensure_runtime_supported_driver(),
        Err(hdf5_pure_rust::Error::Unsupported(_))
    ));
}

#[test]
fn test_dataset_metadata_queries() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();

    let ds = f.dataset("chunked").unwrap();
    assert!(ds.is_chunked().unwrap());
    assert_eq!(ds.offset().unwrap(), None);
    let mut chunk = Vec::new();
    assert!(ds.chunk_into(&mut chunk).unwrap());
    assert_eq!(chunk, vec![10]);

    let dtype = ds.dtype().unwrap();
    assert!(dtype.is_float());
    assert_eq!(dtype.size(), 4);

    let space = ds.space().unwrap();
    assert!(space.is_simple());
    assert_eq!(space.ndim(), 1);
    assert_eq!(space.shape(), &[100]);
    assert!(!space.is_resizable());

    let contiguous = f.dataset("float64_1d").unwrap();
    assert!(contiguous.offset().unwrap().is_some());
}

#[test]
fn test_dataset_metadata_into_helpers_replace_stale_output() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let chunked = f.dataset("chunked").unwrap();
    let contiguous = f.dataset("float64_1d").unwrap();

    let mut shape = vec![999, 888];
    contiguous.shape_into(&mut shape).unwrap();
    assert_eq!(shape, vec![5]);

    let mut chunk = vec![123, 456];
    assert!(chunked.chunk_into(&mut chunk).unwrap());
    assert_eq!(chunk, vec![10]);
    assert!(!contiguous.chunk_into(&mut chunk).unwrap());
    assert!(chunk.is_empty());

    let mut filters = chunked.filters().unwrap();
    assert_eq!(filters.len(), 1);
    contiguous.filters_into(&mut filters).unwrap();
    assert!(filters.is_empty());

    let mut chunks = Vec::new();
    chunked.chunk_infos_into(&mut chunks).unwrap();
    assert_eq!(chunks.len(), 10);
    contiguous.chunk_infos_into(&mut chunks).unwrap();
    assert!(chunks.is_empty());
}
