use hdf5_pure_rust::{File, FileCloseDegree, FileIntent, LibverBound, WritableFile};

fn file_has_member(file: &File, expected: &str) -> hdf5_pure_rust::Result<bool> {
    let mut found = false;
    file.visit_member_names(|name| {
        found |= name == expected;
        Ok(())
    })?;
    Ok(found)
}

fn group_member_count(group: &hdf5_pure_rust::Group) -> hdf5_pure_rust::Result<usize> {
    let mut count = 0;
    group.visit_member_names(|_| {
        count += 1;
        Ok(())
    })?;
    Ok(count)
}

#[test]
fn test_file_size_matches_filesystem_metadata() {
    let path = "tests/data/simple_v0.h5";
    let f = File::open(path).expect("failed to open v0 file");
    let expected = std::fs::metadata(path).unwrap().len();

    assert_eq!(f.file_size().unwrap(), expected);
}

#[test]
fn test_file_path_returns_open_path() {
    let path = std::path::PathBuf::from("tests/data/simple_v0.h5");
    let f = File::open(&path).expect("failed to open v0 file");

    f.with_path(|opened| assert_eq!(opened.unwrap(), path.as_path()));
}

#[test]
fn test_file_compat_create_append_and_open_rw_modes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("compat_modes.h5");

    let created = File::create(&path).expect("create should write an empty HDF5 file");
    assert_eq!(created.intent(), FileIntent::ReadWrite);
    assert!(!created.is_read_only());
    assert!(created.file_size().unwrap() > 0);

    let opened_rw = File::open_rw(&path).expect("open_rw should open an existing file");
    assert_eq!(opened_rw.intent(), FileIntent::ReadWrite);
    assert!(!opened_rw.is_read_only());

    let appended = File::append(&path).expect("append should open an existing file read/write");
    assert_eq!(appended.intent(), FileIntent::ReadWrite);
    assert!(appended.file_size().unwrap() > 0);

    let created_from_append = File::append(dir.path().join("created_by_append.h5"))
        .expect("append should create a missing file");
    assert_eq!(created_from_append.intent(), FileIntent::ReadWrite);
}

#[test]
fn test_file_compat_create_excl_fails_if_present() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("compat_create_excl.h5");

    File::create_excl(&path).expect("create_excl should create missing file");
    let err = match File::create_excl(&path) {
        Ok(_) => panic!("create_excl should reject existing file"),
        Err(err) => err,
    };
    assert!(matches!(err, hdf5_pure_rust::Error::Io(_)));
}

#[test]
fn test_file_builder_compat_create_modes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("builder_create.h5");

    let file = File::with_options()
        .create(&path)
        .expect("builder create should create a file");
    assert_eq!(file.intent(), FileIntent::ReadWrite);

    let reopened = File::with_options()
        .append(&path)
        .expect("builder append should reopen existing file");
    assert_eq!(reopened.intent(), FileIntent::ReadWrite);
}

#[test]
fn test_group_compat_unlink_compact_soft_link() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("unlink_soft.h5");
    {
        let mut writable = WritableFile::create(&path).unwrap();
        writable.link_soft("soft_link", "/missing").unwrap();
        writable.close().unwrap();
    }

    let file = File::open_rw(&path).unwrap();
    let root = file.root_group().unwrap();
    assert!(root.link_exists("soft_link").unwrap());
    root.unlink("soft_link").unwrap();

    let reopened = File::open(&path).unwrap();
    assert!(!reopened
        .root_group()
        .unwrap()
        .link_exists("soft_link")
        .unwrap());
}

#[test]
fn test_group_compat_unlink_compact_hard_link_alias() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("unlink_hard.h5");
    {
        let mut writable = WritableFile::create(&path).unwrap();
        writable
            .new_dataset_builder("real_data")
            .write::<i32>(&[1, 2, 3])
            .unwrap();
        writable.link_hard("alias_data", "/real_data").unwrap();
        writable.close().unwrap();
    }

    let file = File::open_rw(&path).unwrap();
    let root = file.root_group().unwrap();
    assert!(root.link_exists("real_data").unwrap());
    assert!(root.link_exists("alias_data").unwrap());
    root.unlink("alias_data").unwrap();

    let reopened = File::open(&path).unwrap();
    let root = reopened.root_group().unwrap();
    assert!(root.link_exists("real_data").unwrap());
    assert!(!root.link_exists("alias_data").unwrap());
    let mut values = vec![0; 3];
    reopened
        .dataset("real_data")
        .unwrap()
        .read_into(&mut values)
        .unwrap();
    assert_eq!(values, [1, 2, 3]);
}

#[test]
fn test_group_compat_relink_compact_soft_link_same_size() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rename_soft.h5");
    {
        let mut writable = WritableFile::create(&path).unwrap();
        writable.link_soft("old_name", "/missing").unwrap();
        writable.close().unwrap();
    }

    let file = File::open_rw(&path).unwrap();
    let root = file.root_group().unwrap();
    root.relink("old_name", "new_name").unwrap();

    let reopened = File::open(&path).unwrap();
    let root = reopened.root_group().unwrap();
    assert!(!root.link_exists("old_name").unwrap());
    assert!(root.link_exists("new_name").unwrap());
}

#[test]
fn test_group_compat_unlink_requires_read_write_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("readonly_unlink.h5");
    {
        let mut writable = WritableFile::create(&path).unwrap();
        writable.link_soft("soft_link", "/missing").unwrap();
        writable.close().unwrap();
    }

    let file = File::open(&path).unwrap();
    let root = file.root_group().unwrap();
    let err = root.unlink("soft_link").unwrap_err();
    assert!(matches!(err, hdf5_pure_rust::Error::Unsupported(_)));
}

#[test]
fn test_file_metadata_and_access_queries() {
    let path = "tests/data/simple_v0.h5";
    let f = File::open(path).expect("failed to open v0 file");
    let mut image = vec![0u8; f.file_size().unwrap() as usize];
    f.file_image_into(&mut image).unwrap();

    assert_eq!(f.intent(), FileIntent::ReadOnly);
    assert_eq!(f.eoa(), f.superblock().eof_addr);
    assert_eq!(f.freespace(), 0);
    let info = f.info().unwrap();
    assert_eq!(info, f.info_v1().unwrap());
    assert_eq!(info.superblock.version, f.superblock().version);
    assert_eq!(
        info.superblock.size,
        f.superblock().checked_size().unwrap() as u64
    );
    assert_eq!(info.free_space.total_space, 0);
    let access = f.access_plist();
    assert_eq!(f.mdc_config(), access.mdc_config());
    assert_eq!(f.mdc_hit_rate(), 0.0);
    assert_eq!(f.mdc_size().current_size, 0);
    assert_eq!(f.mdc_logging_status(), (false, false));
    assert_eq!(f.page_buffering_stats().raw_data_accesses, 0);
    assert_eq!(f.mdc_image_info().size, 0);
    assert!(!f.dset_no_attrs_hint());
    assert!(!f.mpi_atomicity());
    assert_eq!(image.len() as u64, f.file_size().unwrap());
    assert_eq!(&image[..8], b"\x89HDF\r\n\x1a\n");
    assert!(f.fileno().unwrap() > 0);
    #[cfg(unix)]
    assert!(f.vfd_handle().unwrap() >= 0);

    assert_eq!(access.driver(), "sec2");
    assert_eq!(access.driver_info(), None);
    assert_eq!(access.userblock(), 0);
    assert_eq!(access.alignment(), (1, 1));
    assert_eq!(access.cache(), (0, 521, 1024 * 1024, 0.75));
    assert!(!access.gc_references());
    assert_eq!(access.fclose_degree(), FileCloseDegree::Weak);
    assert_eq!(access.meta_block_size(), 2048);
    assert_eq!(access.sieve_buf_size(), 64 * 1024);
    assert_eq!(access.small_data_block_size(), 2048);
    assert_eq!(
        access.libver_bounds(),
        (LibverBound::Earliest, LibverBound::Latest)
    );
    assert!(!access.evict_on_close());
    assert_eq!(access.file_locking(), (true, false));
    assert_eq!(access.mdc_config().max_size, 0);
    assert!(!access.mdc_image_config().enabled);
    assert!(!access.mdc_log_options().enabled);
    assert!(!access.all_coll_metadata_ops());
    assert!(!access.coll_metadata_write());
    assert_eq!(access.page_buffer_size(), (0, 0, 0));
    assert_eq!(access.fapl_hdfs(), None);
    assert_eq!(access.fapl_direct(), None);
    assert_eq!(access.fapl_mirror(), None);
    assert_eq!(access.fapl_mpio(), None);
    assert_eq!(access.dxpl_mpio(), None);
    assert_eq!(access.fapl_family(), None);
    assert_eq!(access.family_offset(), None);
    assert_eq!(access.multi_type(), None);
    assert_eq!(access.fapl_ioc(), None);
    assert_eq!(access.fapl_subfiling(), None);
    assert_eq!(access.fapl_splitter(), None);
    assert_eq!(access.fapl_multi(), None);
    assert_eq!(access.fapl_onion(), None);
    assert!(!access.core_write_tracking());
    assert_eq!(access.fapl_core(), None);
    assert_eq!(access.fapl_ros3(), None);
    assert_eq!(access.fapl_ros3_endpoint(), None);
    assert_eq!(access.object_flush_cb(), None);
    assert_eq!(access.mpi_params(), None);
    assert_eq!(access.vol_id(), None);
    assert_eq!(access.vol_info(), None);
    assert_eq!(access.vol_cap_flags(), 0);
    assert_eq!(access.relax_file_integrity_checks(), 0);
    assert_eq!(access.map_iterate_hints(), None);
}

#[test]
fn test_file_open_object_registry_queries() {
    let f = File::open("tests/data/simple_v0.h5").expect("failed to open v0 file");
    assert_eq!(f.obj_count(), 1);
    let mut ids = Vec::new();
    f.obj_ids_into(&mut ids);
    assert_eq!(ids, vec![f.object_id()]);

    {
        let group = f.group("group1").unwrap();
        let dataset = f.dataset("data").unwrap();
        f.obj_ids_into(&mut ids);
        assert_eq!(f.obj_count(), 3);
        assert!(ids.contains(&f.object_id()));
        assert!(ids.contains(&group.object_id()));
        assert!(ids.contains(&dataset.object_id()));
    }

    assert_eq!(f.obj_count(), 1);
    f.obj_ids_into(&mut ids);
    assert_eq!(ids, vec![f.object_id()]);
}

#[test]
fn test_list_root_members_v0() {
    let f = File::open("tests/data/simple_v0.h5").expect("failed to open v0 file");

    assert!(file_has_member(&f, "data").expect("failed to list members"));
    assert!(file_has_member(&f, "group1").expect("failed to list members"));
}

#[test]
fn test_list_root_members_v3() {
    let f = File::open("tests/data/simple_v2.h5").expect("failed to open v3 file");

    assert!(file_has_member(&f, "data").expect("failed to list members"));
    assert!(file_has_member(&f, "group1").expect("failed to list members"));
}

#[test]
fn test_open_subgroup_v0() {
    let f = File::open("tests/data/simple_v0.h5").expect("failed to open v0 file");
    let g = f.group("group1").expect("failed to open group1");
    assert_eq!(g.name(), "/group1");

    assert_eq!(
        group_member_count(&g).expect("failed to list group1 members"),
        0
    );
}

#[test]
fn test_open_subgroup_v3() {
    let f = File::open("tests/data/simple_v2.h5").expect("failed to open v3 file");
    let g = f.group("group1").expect("failed to open group1");
    assert_eq!(g.name(), "/group1");

    assert_eq!(
        group_member_count(&g).expect("failed to list group1 members"),
        0
    );
}

#[test]
fn test_member_types_v0() {
    let f = File::open("tests/data/simple_v0.h5").expect("failed to open v0 file");
    let root = f.root_group().expect("failed to get root");

    let data_type = root
        .member_type("data")
        .expect("failed to get type of data");
    let group_type = root
        .member_type("group1")
        .expect("failed to get type of group1");

    println!("v0: data={data_type:?}, group1={group_type:?}");
    assert_eq!(data_type, hdf5_pure_rust::hl::file::ObjectType::Dataset);
    assert_eq!(group_type, hdf5_pure_rust::hl::file::ObjectType::Group);
}

#[test]
fn test_member_types_v3() {
    let f = File::open("tests/data/simple_v2.h5").expect("failed to open v3 file");
    let root = f.root_group().expect("failed to get root");

    let data_type = root
        .member_type("data")
        .expect("failed to get type of data");
    let group_type = root
        .member_type("group1")
        .expect("failed to get type of group1");

    println!("v3: data={data_type:?}, group1={group_type:?}");
    assert_eq!(data_type, hdf5_pure_rust::hl::file::ObjectType::Dataset);
    assert_eq!(group_type, hdf5_pure_rust::hl::file::ObjectType::Group);
}

#[test]
fn test_group_len() {
    let f = File::open("tests/data/simple_v0.h5").unwrap();
    let root = f.root_group().unwrap();
    assert_eq!(root.len().unwrap(), 2); // "data" and "group1"
    assert!(!root.is_empty().unwrap());

    let g1 = f.group("group1").unwrap();
    assert_eq!(g1.len().unwrap(), 0);
    assert!(g1.is_empty().unwrap());
}

#[test]
fn test_path_component_length_cap_rejects_oversized_segment() {
    // A single path component longer than 1024 bytes must be rejected
    // before traversal starts. The shape of the rest of the path doesn't
    // matter; we just need to confirm the cap fires with the documented
    // error rather than returning a generic "not found".
    let f = File::open("tests/data/simple_v0.h5").unwrap();
    let huge = "a".repeat(1025);
    let msg = match f.group(&huge) {
        Ok(_) => panic!("oversized component must not resolve"),
        Err(e) => format!("{e}"),
    };
    assert!(
        msg.contains("path component exceeds 1024-byte limit"),
        "expected length-cap error, got: {msg}"
    );
}

#[test]
fn test_path_component_length_cap_accepts_at_limit() {
    // Exactly 1024 bytes must NOT trigger the cap (it's a strict >, not >=).
    // The lookup will of course fail with a "not found" error — we just
    // assert the failure mode is *not* the length-cap one.
    let f = File::open("tests/data/simple_v0.h5").unwrap();
    let at_limit = "a".repeat(1024);
    let msg = match f.group(&at_limit) {
        Ok(_) => panic!("a 1024-byte component should not resolve in this fixture"),
        Err(e) => format!("{e}"),
    };
    assert!(
        !msg.contains("path component exceeds"),
        "1024-byte component should pass the cap, but got: {msg}"
    );
}
