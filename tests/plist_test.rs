use hdf5_pure_rust::{
    DataTransfer, DatasetAccess, DatasetSpaceStatus, File, FileCloseDegree, FileSpaceStrategy,
    LibverBound, LinkAccess, MetadataCacheConfig, MetadataCacheImageConfig,
    MetadataCacheLogOptions, ObjectCopy, ObjectCreate, VdsView,
};

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
    assert_eq!(plist.file_space(), (FileSpaceStrategy::Aggregate, false, 1));
    assert_eq!(plist.file_space_page_size(), 4096);
    assert_eq!(plist.shared_mesg_nindexes(), 0);
    assert_eq!(plist.shared_mesg_index(0), None);
    assert_eq!(plist.shared_mesg_phase_change(), (50, 40));

    plist.set_sizes(4, 8);
    assert_eq!(plist.sizes(), (4, 8));
    plist.set_sym_k(32, 8);
    assert_eq!(plist.sym_k(), (32, 8));
    plist.set_istore_k(64);
    assert_eq!(plist.istore_k(), 64);
    plist.set_file_space(FileSpaceStrategy::Page, true, 16);
    assert_eq!(plist.file_space(), (FileSpaceStrategy::Page, true, 16));
    plist.set_file_space_page_size(8192);
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
    fapl.set_alignment(128, 4096);
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
    fapl.set_page_buffer_size(1024, 20, 30);
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
    assert_eq!(fapl.object_flush_cb(), Some(()));
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
