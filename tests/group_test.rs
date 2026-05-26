use hdf5_pure_rust::engine::vfd::{
    CoreFileConfig, FamilyFileConfig, H5FD_multi_populate_config, HdfsConfig, LogFileConfig,
    OnionHeader, Ros3Config, SplitterFileConfig, SubfilingConfig,
};
use hdf5_pure_rust::format::object_header::ObjectHeader;
use hdf5_pure_rust::hl::plist::file_access::FileAccess;
use hdf5_pure_rust::io::reader::HdfReader;
use hdf5_pure_rust::{File, FileCloseDegree, FileIntent, LibverBound, OpenMode, WritableFile};
use std::io::BufReader;

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

fn object_refcount_at(path: &std::path::Path, file: &File, addr: u64) -> u32 {
    let mut reader = HdfReader::new(BufReader::new(std::fs::File::open(path).unwrap()));
    reader.set_sizeof_addr(file.superblock().sizeof_addr);
    reader.set_sizeof_size(file.superblock().sizeof_size);
    ObjectHeader::read_at(&mut reader, addr).unwrap().refcount
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
fn test_file_compat_swmr_modes_are_explicitly_unsupported() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("swmr_boundary.h5");
    File::create(&path).expect("create should write an empty HDF5 file");

    let read_swmr = match File::open_as(&path, OpenMode::ReadSWMR) {
        Ok(_) => panic!("ReadSWMR should remain an explicit unsupported boundary"),
        Err(err) => err,
    };
    assert!(matches!(read_swmr, hdf5_pure_rust::Error::Unsupported(_)));

    let file = File::open_rw(&path).unwrap();
    let start_swmr = file
        .start_swmr()
        .expect_err("start_swmr should remain an explicit unsupported boundary");
    assert!(matches!(start_swmr, hdf5_pure_rust::Error::Unsupported(_)));
}

#[test]
fn test_file_builder_rejects_unsupported_runtime_fapl_drivers() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("unsupported_driver.h5");
    File::create(&path).expect("create should write an empty HDF5 file");

    for driver in [
        "core",
        "direct",
        "family",
        "multi",
        "splitter",
        "log",
        "onion",
        "subfiling",
        "hdfs",
        "ros3",
    ] {
        let mut builder = File::with_options();
        builder.access_plist().set_driver(driver);
        let err = match builder.open(&path) {
            Ok(_) => panic!("unsupported runtime driver should fail before opening"),
            Err(err) => err,
        };
        assert!(matches!(err, hdf5_pure_rust::Error::Unsupported(_)));
    }
}

#[test]
fn test_file_access_retains_unsupported_vfd_configs_without_runtime_support() {
    let family = FamilyFileConfig {
        member_size: 4096,
        printf_filename: "member-%03d.h5".to_string(),
    };
    let mut access = FileAccess::default();
    access.set_fapl_family(family.clone());
    access.set_family_offset(Some(8192));
    assert_eq!(access.driver(), "family");
    assert_eq!(access.fapl_family(), Some(&family));
    assert_eq!(access.family_offset(), Some(8192));
    assert!(matches!(
        access.ensure_runtime_supported_driver(),
        Err(hdf5_pure_rust::Error::Unsupported(_))
    ));

    let multi = H5FD_multi_populate_config();
    access.set_fapl_multi(multi.clone());
    access.set_multi_type(Some(3));
    assert_eq!(access.driver(), "multi");
    assert_eq!(access.fapl_multi(), Some(&multi));
    assert_eq!(access.multi_type(), Some(3));

    let splitter = SplitterFileConfig {
        write_only_path: Some(std::path::PathBuf::from("mirror.h5")),
        ignore_wo_errors: true,
    };
    access.set_fapl_splitter(splitter.clone());
    assert_eq!(access.driver(), "splitter");
    assert_eq!(access.fapl_splitter(), Some(&splitter));

    let log = LogFileConfig {
        log_path: Some(std::path::PathBuf::from("driver.log")),
        flags: 7,
        buffer_size: 1024,
    };
    access.set_fapl_log(log.clone());
    assert_eq!(access.driver(), "log");
    assert_eq!(access.fapl_log(), Some(&log));

    let onion = OnionHeader {
        version: 1,
        flags: 2,
        revision_count: 3,
    };
    access.set_fapl_onion(onion.clone());
    assert_eq!(access.driver(), "onion");
    assert_eq!(access.fapl_onion(), Some(&onion));

    let subfiling = SubfilingConfig {
        stripe_size: 1024,
        ioc_count: 2,
        stripe_count: 4,
    };
    access.set_fapl_subfiling(subfiling.clone());
    assert_eq!(access.driver(), "subfiling");
    assert_eq!(access.fapl_subfiling(), Some(&subfiling));

    let hdfs = HdfsConfig {
        namenode_name: "namenode.example.org".to_string(),
        namenode_port: 8020,
        user_name: "reader".to_string(),
        buffer_size: 4096,
    };
    access.set_fapl_hdfs(hdfs.clone());
    assert_eq!(access.driver(), "hdfs");
    assert_eq!(access.fapl_hdfs(), Some(&hdfs));

    let ros3 = Ros3Config {
        endpoint: Some("s3.us-east-1.amazonaws.com".to_string()),
        region: Some("us-east-1".to_string()),
        token: Some("session-token".to_string()),
    };
    access.set_fapl_ros3(ros3.clone());
    assert_eq!(access.driver(), "ros3");
    assert_eq!(access.fapl_ros3(), Some(&ros3));
    assert_eq!(
        access.fapl_ros3_endpoint(),
        Some("s3.us-east-1.amazonaws.com")
    );

    access.set_fapl_core(CoreFileConfig::default());
    assert_eq!(access.driver(), "core");
    assert_eq!(access.fapl_core(), Some(&CoreFileConfig::default()));
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
fn test_group_compat_unlink_compact_same_group_hard_link_with_persistent_refcount() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("unlink_same_group_hard_refcount.h5");
    {
        let mut writable = WritableFile::create(&path).unwrap();
        writable
            .new_dataset_builder("real_data")
            .write::<i32>(&[13, 21])
            .unwrap();
        writable.close().unwrap();
    }

    let file = File::open_rw(&path).unwrap();
    let root = file.root_group().unwrap();
    root.link_hard("/real_data", "alias_data").unwrap();
    let root = file.root_group().unwrap();
    let real_addr = root.link_info("real_data").unwrap().hard_link_addr.unwrap();
    let alias_addr = root
        .link_info("alias_data")
        .unwrap()
        .hard_link_addr
        .unwrap();
    assert_eq!(real_addr, alias_addr);
    assert_eq!(object_refcount_at(&path, &file, real_addr), 2);

    root.unlink("alias_data").unwrap();
    let root = file.root_group().unwrap();
    let real_addr = root.link_info("real_data").unwrap().hard_link_addr.unwrap();
    assert_eq!(object_refcount_at(&path, &file, real_addr), 1);

    let reopened = File::open(&path).unwrap();
    let root = reopened.root_group().unwrap();
    assert!(root.link_exists("real_data").unwrap());
    assert!(!root.link_exists("alias_data").unwrap());
    let real_addr = root.link_info("real_data").unwrap().hard_link_addr.unwrap();
    assert_eq!(object_refcount_at(&path, &reopened, real_addr), 1);
    let mut values = vec![0; 2];
    reopened
        .dataset("real_data")
        .unwrap()
        .read_into(&mut values)
        .unwrap();
    assert_eq!(values, [13, 21]);
}

#[test]
fn test_group_compat_unlink_nested_same_group_hard_link_with_persistent_refcount() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("unlink_nested_same_group_hard_refcount.h5");
    {
        let mut writable = WritableFile::create(&path).unwrap();
        let mut parent = writable.create_group("parent").unwrap();
        parent
            .new_dataset_builder("real_data")
            .write::<i32>(&[34, 55])
            .unwrap();
        writable.close().unwrap();
    }

    let file = File::open_rw(&path).unwrap();
    let parent = file.group("parent").unwrap();
    parent.link_hard("/parent/real_data", "alias_data").unwrap();
    let parent = file.group("parent").unwrap();
    let real_addr = parent
        .link_info("real_data")
        .unwrap()
        .hard_link_addr
        .unwrap();
    let alias_addr = parent
        .link_info("alias_data")
        .unwrap()
        .hard_link_addr
        .unwrap();
    assert_eq!(real_addr, alias_addr);
    assert_eq!(object_refcount_at(&path, &file, real_addr), 2);

    parent.unlink("alias_data").unwrap();

    let reopened = File::open(&path).unwrap();
    let parent = reopened.group("parent").unwrap();
    assert!(parent.link_exists("real_data").unwrap());
    assert!(!parent.link_exists("alias_data").unwrap());
    let real_addr = parent
        .link_info("real_data")
        .unwrap()
        .hard_link_addr
        .unwrap();
    assert_eq!(object_refcount_at(&path, &reopened, real_addr), 1);
    let mut values = vec![0; 2];
    reopened
        .dataset("parent/real_data")
        .unwrap()
        .read_into(&mut values)
        .unwrap();
    assert_eq!(values, [34, 55]);
}

#[test]
fn test_group_compat_cross_group_hard_link_materializes_refcount() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir
        .path()
        .join("cross_group_hard_link_materializes_refcount.h5");
    {
        let mut writable = WritableFile::create(&path).unwrap();
        writable
            .new_dataset_builder("real_data")
            .write::<i32>(&[89, 144])
            .unwrap();
        writable.create_group("parent").unwrap();
        writable.close().unwrap();
    }

    let file = File::open_rw(&path).unwrap();
    let parent = file.group("parent").unwrap();
    parent.link_hard("/real_data", "alias_data").unwrap();

    let root = file.root_group().unwrap();
    let parent = file.group("parent").unwrap();
    let real_addr = root.link_info("real_data").unwrap().hard_link_addr.unwrap();
    let alias_addr = parent
        .link_info("alias_data")
        .unwrap()
        .hard_link_addr
        .unwrap();
    assert_eq!(real_addr, alias_addr);
    assert_eq!(object_refcount_at(&path, &file, real_addr), 2);

    parent.unlink("alias_data").unwrap();
    let root = file.root_group().unwrap();
    let real_addr = root.link_info("real_data").unwrap().hard_link_addr.unwrap();
    assert_eq!(object_refcount_at(&path, &file, real_addr), 1);

    let reopened = File::open(&path).unwrap();
    let root = reopened.root_group().unwrap();
    let parent = reopened.group("parent").unwrap();
    assert!(root.link_exists("real_data").unwrap());
    assert!(!parent.link_exists("alias_data").unwrap());
    let real_addr = root.link_info("real_data").unwrap().hard_link_addr.unwrap();
    assert_eq!(object_refcount_at(&path, &reopened, real_addr), 1);
    let mut values = vec![0; 2];
    reopened
        .dataset("real_data")
        .unwrap()
        .read_into(&mut values)
        .unwrap();
    assert_eq!(values, [89, 144]);
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
fn test_group_compat_relink_dense_link_name_growth_reopens() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("relink_dense_name_growth.h5");
    {
        let mut writable = WritableFile::create(&path).unwrap();
        let mut parent = writable.create_group("parent").unwrap();
        for idx in 0..9 {
            parent
                .new_dataset_builder(&format!("data_{idx:02}"))
                .write::<i32>(&[idx])
                .unwrap();
        }
        writable.close().unwrap();
    }

    let file = File::open_rw(&path).unwrap();
    let parent = file.group("parent").unwrap();
    parent.relink("data_08", "longer_data_name_08").unwrap();

    let parent = file.group("parent").unwrap();
    assert!(parent.link_exists("data_00").unwrap());
    assert!(!parent.link_exists("data_08").unwrap());
    assert!(parent.link_exists("longer_data_name_08").unwrap());
    let mut values = vec![0; 1];
    file.dataset("parent/longer_data_name_08")
        .unwrap()
        .read_into(&mut values)
        .unwrap();
    assert_eq!(values, [8]);

    let reopened = File::open(&path).unwrap();
    let parent = reopened.group("parent").unwrap();
    assert!(parent.link_exists("data_00").unwrap());
    assert!(!parent.link_exists("data_08").unwrap());
    assert!(parent.link_exists("longer_data_name_08").unwrap());
    let mut values = vec![0; 1];
    reopened
        .dataset("parent/longer_data_name_08")
        .unwrap()
        .read_into(&mut values)
        .unwrap();
    assert_eq!(values, [8]);
}

#[test]
fn test_group_compat_relink_cross_group_nested_compact_hard_link_reopens() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("relink_cross_group_nested.h5");
    {
        let mut writable = WritableFile::create(&path).unwrap();
        let mut left = writable.create_group("left").unwrap();
        let mut source = left.create_group("source").unwrap();
        source
            .new_dataset_builder("old")
            .write::<i32>(&[144, 233])
            .unwrap();
        let mut right = writable.create_group("right").unwrap();
        right.create_group("dest").unwrap();
        writable.close().unwrap();
    }

    let file = File::open_rw(&path).unwrap();
    let source = file.group("left/source").unwrap();
    source
        .relink("old", "/right/dest/longer_dataset_name")
        .unwrap();

    let source = file.group("left/source").unwrap();
    let dest = file.group("right/dest").unwrap();
    assert!(!source.link_exists("old").unwrap());
    assert!(dest.link_exists("longer_dataset_name").unwrap());
    let mut values = vec![0; 2];
    file.dataset("right/dest/longer_dataset_name")
        .unwrap()
        .read_into(&mut values)
        .unwrap();
    assert_eq!(values, [144, 233]);

    let reopened = File::open(&path).unwrap();
    assert!(!reopened
        .group("left/source")
        .unwrap()
        .link_exists("old")
        .unwrap());
    assert!(reopened
        .group("right/dest")
        .unwrap()
        .link_exists("longer_dataset_name")
        .unwrap());
    let mut values = vec![0; 2];
    reopened
        .dataset("right/dest/longer_dataset_name")
        .unwrap()
        .read_into(&mut values)
        .unwrap();
    assert_eq!(values, [144, 233]);
}

#[test]
fn test_group_compat_relink_cross_group_into_dense_destination_reopens() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("relink_cross_group_into_dense_dest.h5");
    {
        let mut writable = WritableFile::create(&path).unwrap();
        let mut source = writable.create_group("source").unwrap();
        source
            .new_dataset_builder("old")
            .write::<i32>(&[1597, 2584])
            .unwrap();
        let mut dest = writable.create_group("dest").unwrap();
        for idx in 0..9 {
            dest.create_group(&format!("existing_{idx:02}")).unwrap();
        }
        writable.close().unwrap();
    }

    let file = File::open_rw(&path).unwrap();
    let source = file.group("source").unwrap();
    source.relink("old", "/dest/moved").unwrap();

    let source = file.group("source").unwrap();
    let dest = file.group("dest").unwrap();
    assert!(!source.link_exists("old").unwrap());
    assert!(dest.link_exists("existing_00").unwrap());
    assert!(dest.link_exists("existing_08").unwrap());
    assert!(dest.link_exists("moved").unwrap());
    let mut values = vec![0; 2];
    file.dataset("dest/moved")
        .unwrap()
        .read_into(&mut values)
        .unwrap();
    assert_eq!(values, [1597, 2584]);

    let reopened = File::open(&path).unwrap();
    assert!(!reopened
        .group("source")
        .unwrap()
        .link_exists("old")
        .unwrap());
    let dest = reopened.group("dest").unwrap();
    assert!(dest.link_exists("existing_00").unwrap());
    assert!(dest.link_exists("existing_08").unwrap());
    assert!(dest.link_exists("moved").unwrap());
    let mut values = vec![0; 2];
    reopened
        .dataset("dest/moved")
        .unwrap()
        .read_into(&mut values)
        .unwrap();
    assert_eq!(values, [1597, 2584]);
}

#[test]
fn test_group_compat_relink_cross_group_dense_source_into_dense_destination_reopens() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("relink_dense_source_into_dense_dest.h5");
    {
        let mut writable = WritableFile::create(&path).unwrap();
        let mut source = writable.create_group("source").unwrap();
        for idx in 0..9 {
            source
                .new_dataset_builder(&format!("data_{idx:02}"))
                .write::<i32>(&[idx])
                .unwrap();
        }
        let mut dest = writable.create_group("dest").unwrap();
        for idx in 0..9 {
            dest.create_group(&format!("existing_{idx:02}")).unwrap();
        }
        writable.close().unwrap();
    }

    let file = File::open_rw(&path).unwrap();
    let source = file.group("source").unwrap();
    source.relink("data_08", "/dest/moved").unwrap();

    let source = file.group("source").unwrap();
    let dest = file.group("dest").unwrap();
    assert!(!source.link_exists("data_08").unwrap());
    assert!(source.link_exists("data_00").unwrap());
    assert!(dest.link_exists("existing_00").unwrap());
    assert!(dest.link_exists("existing_08").unwrap());
    assert!(dest.link_exists("moved").unwrap());
    let mut values = vec![0; 1];
    file.dataset("dest/moved")
        .unwrap()
        .read_into(&mut values)
        .unwrap();
    assert_eq!(values, [8]);

    let reopened = File::open(&path).unwrap();
    let source = reopened.group("source").unwrap();
    let dest = reopened.group("dest").unwrap();
    assert!(!source.link_exists("data_08").unwrap());
    assert!(source.link_exists("data_00").unwrap());
    assert!(dest.link_exists("existing_00").unwrap());
    assert!(dest.link_exists("existing_08").unwrap());
    assert!(dest.link_exists("moved").unwrap());
    let mut values = vec![0; 1];
    reopened
        .dataset("dest/moved")
        .unwrap()
        .read_into(&mut values)
        .unwrap();
    assert_eq!(values, [8]);
}

#[test]
fn test_group_compat_create_inside_creation_order_indexed_dense_group_is_unsupported() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("creation_order_dense_mutation_boundary.h5");
    std::fs::copy("tests/data/hdf5_ref/dense_group_cases.h5", &path).unwrap();

    let file = File::open_rw(&path).unwrap();
    let tracked = file.group("creation_order_tracked").unwrap();
    let mut creation_orders = Vec::new();
    tracked
        .visit_links_by_creation_order(|link| {
            creation_orders.push(link.creation_order.unwrap());
            Ok(())
        })
        .unwrap();
    assert_eq!(creation_orders, (0..64).collect::<Vec<_>>());

    let err = match tracked.create_group("new_child") {
        Ok(_) => panic!("creation-order indexed dense mutation should stay unsupported"),
        Err(err) => err,
    };
    assert!(matches!(err, hdf5_pure_rust::Error::Unsupported(_)));
    assert!(
        err.to_string()
            .contains("object-header creation-order tracking"),
        "unexpected error: {err}"
    );
    assert!(!tracked.link_exists("new_child").unwrap());
    assert!(tracked.link_exists("tracked_00").unwrap());
    assert!(tracked.link_exists("tracked_63").unwrap());

    let reopened = File::open(&path).unwrap();
    let tracked = reopened.group("creation_order_tracked").unwrap();
    assert!(!tracked.link_exists("new_child").unwrap());
    assert_eq!(group_member_count(&tracked).unwrap(), 64);
}

#[test]
fn test_group_compat_create_dataset_and_link_inside_creation_order_indexed_dense_group_are_unsupported(
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir
        .path()
        .join("creation_order_dense_dataset_link_boundary.h5");
    std::fs::copy("tests/data/hdf5_ref/dense_group_cases.h5", &path).unwrap();

    let file = File::open_rw(&path).unwrap();
    let tracked = file.group("creation_order_tracked").unwrap();
    let initial_count = group_member_count(&tracked).unwrap();

    let dataset_err = match tracked
        .new_dataset_builder()
        .with_data::<i32>(&[1, 2, 3])
        .create(Some("new_values"))
    {
        Ok(_) => panic!("creation-order indexed dense dataset creation should stay unsupported"),
        Err(err) => err,
    };
    assert!(matches!(dataset_err, hdf5_pure_rust::Error::Unsupported(_)));
    assert!(
        dataset_err.to_string().contains("creation-order tracking"),
        "unexpected error: {dataset_err}"
    );

    let link_err = tracked
        .link_soft("/missing", "new_soft_link")
        .expect_err("creation-order indexed dense link creation should stay unsupported");
    assert!(matches!(link_err, hdf5_pure_rust::Error::Unsupported(_)));
    assert!(
        link_err.to_string().contains("creation-order tracking"),
        "unexpected error: {link_err}"
    );

    assert!(!tracked.link_exists("new_values").unwrap());
    assert!(!tracked.link_exists("new_soft_link").unwrap());
    assert_eq!(group_member_count(&tracked).unwrap(), initial_count);

    let reopened = File::open(&path).unwrap();
    let tracked = reopened.group("creation_order_tracked").unwrap();
    assert!(!tracked.link_exists("new_values").unwrap());
    assert!(!tracked.link_exists("new_soft_link").unwrap());
    assert_eq!(group_member_count(&tracked).unwrap(), initial_count);
    let mut creation_orders = Vec::new();
    tracked
        .visit_links_by_creation_order(|link| {
            creation_orders.push(link.creation_order.unwrap());
            Ok(())
        })
        .unwrap();
    assert_eq!(creation_orders, (0..64).collect::<Vec<_>>());
}

#[test]
fn test_group_compat_create_external_and_hard_links_inside_creation_order_indexed_dense_group_are_unsupported(
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir
        .path()
        .join("creation_order_dense_external_hard_link_boundary.h5");
    std::fs::copy("tests/data/hdf5_ref/dense_group_cases.h5", &path).unwrap();

    let file = File::open_rw(&path).unwrap();
    let root = file.root_group().unwrap();
    let tracked = file.group("creation_order_tracked").unwrap();
    let initial_count = group_member_count(&tracked).unwrap();
    let target_addr = root
        .link_info("name_index_deep")
        .unwrap()
        .hard_link_addr
        .unwrap();
    let initial_refcount = object_refcount_at(&path, &file, target_addr);

    let external_err = tracked
        .link_external("external.h5", "/external_target", "new_external_link")
        .expect_err("creation-order indexed dense external-link creation should stay unsupported");
    assert!(matches!(
        external_err,
        hdf5_pure_rust::Error::Unsupported(_)
    ));
    assert!(
        external_err.to_string().contains("creation-order tracking"),
        "unexpected error: {external_err}"
    );

    let hard_err = tracked
        .link_hard("/name_index_deep", "new_hard_link")
        .expect_err("creation-order indexed dense hard-link creation should stay unsupported");
    assert!(matches!(hard_err, hdf5_pure_rust::Error::Unsupported(_)));
    assert!(
        hard_err.to_string().contains("creation-order tracking"),
        "unexpected error: {hard_err}"
    );

    assert!(!tracked.link_exists("new_external_link").unwrap());
    assert!(!tracked.link_exists("new_hard_link").unwrap());
    assert_eq!(group_member_count(&tracked).unwrap(), initial_count);
    assert_eq!(
        object_refcount_at(&path, &file, target_addr),
        initial_refcount
    );

    let reopened = File::open(&path).unwrap();
    let root = reopened.root_group().unwrap();
    let tracked = reopened.group("creation_order_tracked").unwrap();
    assert!(root.link_exists("name_index_deep").unwrap());
    assert!(!tracked.link_exists("new_external_link").unwrap());
    assert!(!tracked.link_exists("new_hard_link").unwrap());
    assert_eq!(group_member_count(&tracked).unwrap(), initial_count);
    assert_eq!(
        object_refcount_at(&path, &reopened, target_addr),
        initial_refcount
    );
}

#[test]
fn test_group_compat_unlink_inside_creation_order_indexed_dense_group_is_unsupported() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("creation_order_dense_unlink_boundary.h5");
    std::fs::copy("tests/data/hdf5_ref/dense_group_cases.h5", &path).unwrap();

    let file = File::open_rw(&path).unwrap();
    let tracked = file.group("creation_order_tracked").unwrap();
    assert!(tracked.link_exists("tracked_08").unwrap());

    let err = match tracked.unlink("tracked_08") {
        Ok(_) => panic!("creation-order indexed dense unlink should stay unsupported"),
        Err(err) => err,
    };
    assert!(matches!(err, hdf5_pure_rust::Error::Unsupported(_)));
    assert!(
        err.to_string()
            .contains("creation-order indexed dense links"),
        "unexpected error: {err}"
    );
    assert!(tracked.link_exists("tracked_08").unwrap());

    let reopened = File::open(&path).unwrap();
    let tracked = reopened.group("creation_order_tracked").unwrap();
    assert!(tracked.link_exists("tracked_08").unwrap());
    assert_eq!(group_member_count(&tracked).unwrap(), 64);
}

#[test]
fn test_group_compat_relink_same_size_inside_creation_order_indexed_dense_group() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("creation_order_dense_relink_same_size.h5");
    let out = std::process::Command::new("python3")
        .arg("-c")
        .arg(
            r#"import sys
try:
    import h5py
except ModuleNotFoundError as exc:
    raise exc
with h5py.File(sys.argv[1], 'w') as f:
    g = f.create_group('creation_order_tracked', track_order=True)
    for i in range(10):
        g.create_group(f'tracked_{i:02d}')
"#,
        )
        .arg(&path)
        .output()
        .unwrap();
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if stderr.contains("No module named 'h5py'") {
            return;
        }
        panic!("h5py fixture creation failed: {stderr}");
    }

    let file = File::open_rw(&path).unwrap();
    let tracked = file.group("creation_order_tracked").unwrap();
    assert!(tracked.link_exists("tracked_08").unwrap());
    assert!(!tracked.link_exists("renamed_08").unwrap());

    tracked.relink("tracked_08", "renamed_08").unwrap();
    assert!(!tracked.link_exists("tracked_08").unwrap());
    assert!(tracked.link_exists("renamed_08").unwrap());

    let reopened = File::open(&path).unwrap();
    let tracked = reopened.group("creation_order_tracked").unwrap();
    assert!(!tracked.link_exists("tracked_08").unwrap());
    assert!(tracked.link_exists("renamed_08").unwrap());
    assert_eq!(group_member_count(&tracked).unwrap(), 10);

    let mut creation_order_names = Vec::new();
    tracked
        .visit_links_by_creation_order(|link| {
            creation_order_names.push(link.name.clone());
            Ok(())
        })
        .unwrap();
    assert_eq!(creation_order_names.len(), 10);
    assert_eq!(creation_order_names[8], "renamed_08");
}

#[test]
fn test_group_compat_relink_into_creation_order_indexed_dense_group_preserves_source() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir
        .path()
        .join("creation_order_dense_relink_destination_boundary.h5");
    std::fs::copy("tests/data/hdf5_ref/dense_group_cases.h5", &path).unwrap();

    let file = File::open_rw(&path).unwrap();
    let root = file.root_group().unwrap();
    let tracked = file.group("creation_order_tracked").unwrap();
    let root_count = group_member_count(&root).unwrap();
    let tracked_count = group_member_count(&tracked).unwrap();
    assert!(root.link_exists("name_index_deep").unwrap());
    assert!(!tracked.link_exists("moved_name_index_deep").unwrap());

    let err = match root.relink(
        "name_index_deep",
        "/creation_order_tracked/moved_name_index_deep",
    ) {
        Ok(_) => panic!("creation-order indexed dense destination should stay unsupported"),
        Err(err) => err,
    };
    assert!(matches!(err, hdf5_pure_rust::Error::Unsupported(_)));
    assert!(
        err.to_string().contains("creation-order tracking"),
        "unexpected error: {err}"
    );
    assert!(root.link_exists("name_index_deep").unwrap());
    assert!(!tracked.link_exists("moved_name_index_deep").unwrap());
    assert_eq!(group_member_count(&root).unwrap(), root_count);
    assert_eq!(group_member_count(&tracked).unwrap(), tracked_count);

    let reopened = File::open(&path).unwrap();
    let root = reopened.root_group().unwrap();
    let tracked = reopened.group("creation_order_tracked").unwrap();
    assert!(root.link_exists("name_index_deep").unwrap());
    assert!(!tracked.link_exists("moved_name_index_deep").unwrap());
    assert_eq!(group_member_count(&root).unwrap(), root_count);
    assert_eq!(group_member_count(&tracked).unwrap(), tracked_count);
}

#[test]
fn test_group_compat_relink_from_creation_order_indexed_dense_group_preserves_source() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir
        .path()
        .join("creation_order_dense_relink_source_boundary.h5");
    std::fs::copy("tests/data/hdf5_ref/dense_group_cases.h5", &path).unwrap();

    let file = File::open_rw(&path).unwrap();
    let root = file.root_group().unwrap();
    let tracked = file.group("creation_order_tracked").unwrap();
    let root_count = group_member_count(&root).unwrap();
    let tracked_count = group_member_count(&tracked).unwrap();
    assert!(tracked.link_exists("tracked_08").unwrap());
    assert!(!root.link_exists("moved_tracked_08").unwrap());

    let err = match tracked.relink("tracked_08", "/moved_tracked_08") {
        Ok(_) => panic!("creation-order indexed dense source relink should stay unsupported"),
        Err(err) => err,
    };
    assert!(matches!(err, hdf5_pure_rust::Error::Unsupported(_)));
    assert!(
        err.to_string().contains("creation-order tracking"),
        "unexpected error: {err}"
    );
    assert!(tracked.link_exists("tracked_08").unwrap());
    assert!(!root.link_exists("moved_tracked_08").unwrap());
    assert_eq!(group_member_count(&root).unwrap(), root_count);
    assert_eq!(group_member_count(&tracked).unwrap(), tracked_count);

    let reopened = File::open(&path).unwrap();
    let root = reopened.root_group().unwrap();
    let tracked = reopened.group("creation_order_tracked").unwrap();
    assert!(tracked.link_exists("tracked_08").unwrap());
    assert!(!root.link_exists("moved_tracked_08").unwrap());
    assert_eq!(group_member_count(&root).unwrap(), root_count);
    assert_eq!(group_member_count(&tracked).unwrap(), tracked_count);
}

#[test]
fn test_group_compat_create_inside_v1_symbol_table_group_is_unsupported() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir
        .path()
        .join("v1_symbol_table_group_mutation_boundary.h5");
    std::fs::copy("tests/data/hdf5_ref/group_old.h5", &path).unwrap();

    let file = File::open_rw(&path).unwrap();
    let old = file.group("old").unwrap();
    assert_eq!(group_member_count(&old).unwrap(), 0);

    let err = match old.create_group("new_child") {
        Ok(_) => panic!("old-format group mutation should stay unsupported"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("v2/v3 superblock"),
        "unexpected error: {err}"
    );
    assert!(!old.link_exists("new_child").unwrap());

    let reopened = File::open(&path).unwrap();
    let old = reopened.group("old").unwrap();
    assert_eq!(group_member_count(&old).unwrap(), 0);
    assert!(!old.link_exists("new_child").unwrap());
}

#[test]
fn test_group_compat_create_dataset_and_link_inside_v1_symbol_table_group_are_unsupported() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v1_symbol_table_dataset_link_boundary.h5");
    std::fs::copy("tests/data/hdf5_ref/group_old.h5", &path).unwrap();

    let file = File::open_rw(&path).unwrap();
    let old = file.group("old").unwrap();
    assert_eq!(group_member_count(&old).unwrap(), 0);

    let dataset_err = match old
        .new_dataset_builder()
        .with_data::<i32>(&[1, 2, 3])
        .create(Some("new_values"))
    {
        Ok(_) => panic!("old-format dataset creation should stay unsupported"),
        Err(err) => err,
    };
    assert!(
        dataset_err.to_string().contains("v2/v3 superblock"),
        "unexpected error: {dataset_err}"
    );

    let link_err = old
        .link_soft("/missing", "new_soft_link")
        .expect_err("old-format soft-link creation should stay unsupported");
    assert!(
        link_err.to_string().contains("v2/v3 superblock"),
        "unexpected error: {link_err}"
    );

    assert!(!old.link_exists("new_values").unwrap());
    assert!(!old.link_exists("new_soft_link").unwrap());
    assert_eq!(group_member_count(&old).unwrap(), 0);

    let reopened = File::open(&path).unwrap();
    let old = reopened.group("old").unwrap();
    assert!(!old.link_exists("new_values").unwrap());
    assert!(!old.link_exists("new_soft_link").unwrap());
    assert_eq!(group_member_count(&old).unwrap(), 0);
}

#[test]
fn test_group_compat_create_external_and_hard_links_inside_v1_symbol_table_group_are_unsupported() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir
        .path()
        .join("v1_symbol_table_external_hard_link_boundary.h5");
    std::fs::copy("tests/data/hdf5_ref/group_old.h5", &path).unwrap();

    let file = File::open_rw(&path).unwrap();
    let root = file.root_group().unwrap();
    let old = file.group("old").unwrap();
    let root_count = group_member_count(&root).unwrap();
    assert_eq!(group_member_count(&old).unwrap(), 0);
    assert!(root.link_exists("old").unwrap());

    let external_err = old
        .link_external("external.h5", "/external_target", "new_external_link")
        .expect_err("old-format external-link creation should stay unsupported");
    assert!(
        external_err.to_string().contains("v2/v3 superblock"),
        "unexpected error: {external_err}"
    );

    let hard_err = old
        .link_hard("/old", "new_hard_link")
        .expect_err("old-format hard-link creation should stay unsupported");
    assert!(
        hard_err.to_string().contains("v2/v3 superblock"),
        "unexpected error: {hard_err}"
    );

    assert!(!old.link_exists("new_external_link").unwrap());
    assert!(!old.link_exists("new_hard_link").unwrap());
    assert_eq!(group_member_count(&old).unwrap(), 0);
    assert_eq!(group_member_count(&root).unwrap(), root_count);

    let reopened = File::open(&path).unwrap();
    let root = reopened.root_group().unwrap();
    let old = reopened.group("old").unwrap();
    assert!(root.link_exists("old").unwrap());
    assert!(!old.link_exists("new_external_link").unwrap());
    assert!(!old.link_exists("new_hard_link").unwrap());
    assert_eq!(group_member_count(&old).unwrap(), 0);
    assert_eq!(group_member_count(&root).unwrap(), root_count);
}

#[test]
fn test_group_compat_relink_and_unlink_v1_symbol_table_root_are_unsupported() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v1_symbol_table_root_mutation_boundary.h5");
    std::fs::copy("tests/data/hdf5_ref/group_old.h5", &path).unwrap();

    let file = File::open_rw(&path).unwrap();
    let root = file.root_group().unwrap();
    let root_count = group_member_count(&root).unwrap();
    assert!(root.link_exists("old").unwrap());
    assert!(!root.link_exists("renamed_old").unwrap());

    let relink_err = match root.relink("old", "renamed_old") {
        Ok(_) => panic!("old-format root relink should stay unsupported"),
        Err(err) => err,
    };
    assert!(matches!(relink_err, hdf5_pure_rust::Error::Unsupported(_)));
    assert!(
        relink_err.to_string().contains("v2 object headers"),
        "unexpected error: {relink_err}"
    );
    assert!(root.link_exists("old").unwrap());
    assert!(!root.link_exists("renamed_old").unwrap());
    assert_eq!(group_member_count(&root).unwrap(), root_count);

    let unlink_err = match root.unlink("old") {
        Ok(_) => panic!("old-format root unlink should stay unsupported"),
        Err(err) => err,
    };
    assert!(matches!(unlink_err, hdf5_pure_rust::Error::Unsupported(_)));
    assert!(
        unlink_err.to_string().contains("v2 object headers"),
        "unexpected error: {unlink_err}"
    );
    assert!(root.link_exists("old").unwrap());
    assert_eq!(group_member_count(&root).unwrap(), root_count);

    let reopened = File::open(&path).unwrap();
    let root = reopened.root_group().unwrap();
    assert!(root.link_exists("old").unwrap());
    assert!(!root.link_exists("renamed_old").unwrap());
    assert_eq!(group_member_count(&root).unwrap(), root_count);
}

#[test]
fn test_group_compat_relink_cross_group_hard_alias_preserves_refcount() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("relink_cross_group_refcount.h5");
    {
        let mut writable = WritableFile::create(&path).unwrap();
        let mut left = writable.create_group("left").unwrap();
        let mut source = left.create_group("source").unwrap();
        source
            .new_dataset_builder("real_data")
            .write::<i32>(&[377, 610])
            .unwrap();
        let mut right = writable.create_group("right").unwrap();
        right.create_group("dest").unwrap();
        writable.close().unwrap();
    }

    let file = File::open_rw(&path).unwrap();
    let source = file.group("left/source").unwrap();
    source.link_hard("/left/source/real_data", "alias").unwrap();
    let source = file.group("left/source").unwrap();
    let real_addr = source
        .link_info("real_data")
        .unwrap()
        .hard_link_addr
        .unwrap();
    assert_eq!(object_refcount_at(&path, &file, real_addr), 2);

    source.relink("alias", "/right/dest/moved_alias").unwrap();
    let source = file.group("left/source").unwrap();
    let dest = file.group("right/dest").unwrap();
    assert!(source.link_exists("real_data").unwrap());
    assert!(!source.link_exists("alias").unwrap());
    assert!(dest.link_exists("moved_alias").unwrap());
    let moved_addr = dest
        .link_info("moved_alias")
        .unwrap()
        .hard_link_addr
        .unwrap();
    assert_eq!(real_addr, moved_addr);
    assert_eq!(object_refcount_at(&path, &file, real_addr), 2);

    let reopened = File::open(&path).unwrap();
    let source = reopened.group("left/source").unwrap();
    let dest = reopened.group("right/dest").unwrap();
    assert!(source.link_exists("real_data").unwrap());
    assert!(!source.link_exists("alias").unwrap());
    assert!(dest.link_exists("moved_alias").unwrap());
    let moved_addr = dest
        .link_info("moved_alias")
        .unwrap()
        .hard_link_addr
        .unwrap();
    assert_eq!(real_addr, moved_addr);
    assert_eq!(object_refcount_at(&path, &reopened, real_addr), 2);
}

#[test]
fn test_group_compat_create_root_child_open_rw() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("create_root_child.h5");
    {
        let mut writable = WritableFile::create(&path).unwrap();
        writable.create_group("existing").unwrap();
        writable.close().unwrap();
    }

    let file = File::open_rw(&path).unwrap();
    let root = file.root_group().unwrap();
    let created = root.create_group("created").unwrap();
    assert_eq!(created.name(), "/created");
    assert!(created.is_empty().unwrap());
    assert!(file.root_group().unwrap().link_exists("created").unwrap());
    assert!(file.group("created").unwrap().is_empty().unwrap());

    let reopened = File::open(&path).unwrap();
    let root = reopened.root_group().unwrap();
    assert!(root.link_exists("existing").unwrap());
    assert!(root.link_exists("created").unwrap());
    assert!(reopened.group("created").unwrap().is_empty().unwrap());
}

#[test]
fn test_group_compat_stale_handles_use_rewritten_metadata_addresses() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("stale_group_handles_after_rewrite.h5");
    {
        let mut writable = WritableFile::create(&path).unwrap();
        writable.add_attr("root_attr", 7i32).unwrap();
        let mut parent = writable.create_group("parent").unwrap();
        parent.add_attr("parent_attr", 11i32).unwrap();
        writable.close().unwrap();
    }

    let file = File::open_rw(&path).unwrap();
    let root = file.root_group().unwrap();
    let parent = file.group("parent").unwrap();

    root.create_group("created_with_a_long_name_to_force_root_rewrite")
        .unwrap();
    parent.create_group("child").unwrap();

    let fresh_root = file.root_group().unwrap();
    let fresh_parent = file.group("parent").unwrap();
    assert_eq!(
        root.native_info().unwrap().addr,
        fresh_root.native_info().unwrap().addr
    );
    assert_eq!(
        parent.native_info().unwrap().addr,
        fresh_parent.native_info().unwrap().addr
    );
    assert!(parent.link_exists("child").unwrap());
    assert!(parent.open_group("child").unwrap().is_empty().unwrap());
    assert!(root.attr_exists("root_attr").unwrap());
    assert!(parent.attr_exists("parent_attr").unwrap());
}

#[test]
fn test_group_compat_create_nested_child_open_rw() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("create_nested_child.h5");
    {
        let mut writable = WritableFile::create(&path).unwrap();
        writable.create_group("parent").unwrap();
        writable.close().unwrap();
    }

    let file = File::open_rw(&path).unwrap();
    let parent = file.group("parent").unwrap();
    let child = parent.create_group("child").unwrap();
    assert_eq!(child.name(), "/parent/child");
    assert!(file.group("parent").unwrap().link_exists("child").unwrap());
    assert!(file.group("parent/child").unwrap().is_empty().unwrap());

    let reopened = File::open(&path).unwrap();
    assert!(reopened
        .group("parent")
        .unwrap()
        .link_exists("child")
        .unwrap());
    assert!(reopened.group("parent/child").unwrap().is_empty().unwrap());
}

#[test]
fn test_group_compat_create_root_dataset_open_rw_with_data() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("create_root_dataset_with_data.h5");
    {
        let writable = WritableFile::create(&path).unwrap();
        writable.close().unwrap();
    }

    let file = File::open_rw(&path).unwrap();
    let root = file.root_group().unwrap();
    root.new_dataset_builder()
        .with_data::<i32>(&[11, 12, 13])
        .create(Some("values"))
        .unwrap();

    assert!(file.root_group().unwrap().link_exists("values").unwrap());
    let mut values = vec![0; 3];
    file.dataset("values")
        .unwrap()
        .read_into(&mut values)
        .unwrap();
    assert_eq!(values, [11, 12, 13]);

    let reopened = File::open(&path).unwrap();
    let mut reopened_values = vec![0; 3];
    reopened
        .dataset("values")
        .unwrap()
        .read_into(&mut reopened_values)
        .unwrap();
    assert_eq!(reopened_values, [11, 12, 13]);
}

#[test]
fn test_group_compat_create_root_dataset_open_rw_empty_shape() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("create_root_dataset_empty_shape.h5");
    {
        let writable = WritableFile::create(&path).unwrap();
        writable.close().unwrap();
    }

    let file = File::open_rw(&path).unwrap();
    let root = file.root_group().unwrap();
    root.new_dataset::<u16>()
        .shape([4usize])
        .create(Some("zeros"))
        .unwrap();

    let mut values = vec![99; 4];
    File::open(&path)
        .unwrap()
        .dataset("zeros")
        .unwrap()
        .read_into(&mut values)
        .unwrap();
    assert_eq!(values, [0, 0, 0, 0]);
}

#[test]
fn test_group_compat_create_nested_dataset_open_rw() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("create_nested_dataset.h5");
    {
        let mut writable = WritableFile::create(&path).unwrap();
        writable.create_group("parent").unwrap();
        writable.close().unwrap();
    }

    let file = File::open_rw(&path).unwrap();
    let parent = file.group("parent").unwrap();
    parent
        .new_dataset_builder()
        .with_data::<i32>(&[1, 2, 3])
        .create(Some("child"))
        .unwrap();

    let mut values = vec![0; 3];
    file.dataset("parent/child")
        .unwrap()
        .read_into(&mut values)
        .unwrap();
    assert_eq!(values, [1, 2, 3]);

    let mut reopened_values = vec![0; 3];
    File::open(&path)
        .unwrap()
        .dataset("parent/child")
        .unwrap()
        .read_into(&mut reopened_values)
        .unwrap();
    assert_eq!(reopened_values, [1, 2, 3]);
}

#[test]
fn test_group_compat_create_nested_links_open_rw() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("create_nested_links.h5");
    {
        let mut writable = WritableFile::create(&path).unwrap();
        writable.create_group("parent").unwrap();
        writable
            .new_dataset_builder("target")
            .write::<i32>(&[5, 8])
            .unwrap();
        writable.close().unwrap();
    }

    let file = File::open_rw(&path).unwrap();
    let parent = file.group("parent").unwrap();
    parent.link_soft("/missing", "soft_child").unwrap();
    parent.link_hard("/target", "hard_child").unwrap();
    parent
        .link_external("external.h5", "/external_target", "external_child")
        .unwrap();

    let reopened = File::open(&path).unwrap();
    let parent = reopened.group("parent").unwrap();
    assert!(parent.link_exists("soft_child").unwrap());
    assert!(parent.link_exists("hard_child").unwrap());
    assert!(parent.link_exists("external_child").unwrap());
    assert_eq!(parent.soft_link_target("soft_child").unwrap(), "/missing");
    assert_eq!(
        parent.external_link_target("external_child").unwrap(),
        ("external.h5".to_string(), "/external_target".to_string())
    );
    let mut values = vec![0; 2];
    reopened
        .dataset("parent/hard_child")
        .unwrap()
        .read_into(&mut values)
        .unwrap();
    assert_eq!(values, [5, 8]);
}

#[test]
fn test_group_compat_create_deeper_nested_child_open_rw() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("create_deeper_nested_child.h5");
    {
        let mut writable = WritableFile::create(&path).unwrap();
        let mut parent = writable.create_group("parent").unwrap();
        parent.create_group("child").unwrap();
        writable.close().unwrap();
    }

    let file = File::open_rw(&path).unwrap();
    let child = file.group("parent/child").unwrap();
    let grandchild = child.create_group("grandchild").unwrap();
    assert_eq!(grandchild.name(), "/parent/child/grandchild");
    assert!(file
        .group("parent/child")
        .unwrap()
        .link_exists("grandchild")
        .unwrap());
    assert!(file
        .group("parent/child/grandchild")
        .unwrap()
        .is_empty()
        .unwrap());

    let reopened = File::open(&path).unwrap();
    assert!(reopened
        .group("parent/child")
        .unwrap()
        .link_exists("grandchild")
        .unwrap());
    assert!(reopened
        .group("parent/child/grandchild")
        .unwrap()
        .is_empty()
        .unwrap());
}

#[test]
fn test_group_compat_create_child_under_dense_root_parent_open_rw() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("create_child_under_dense_root_parent.h5");
    {
        let mut writable = WritableFile::create(&path).unwrap();
        for idx in 0..9 {
            writable.create_group(&format!("parent_{idx:02}")).unwrap();
        }
        writable.close().unwrap();
    }

    let file = File::open_rw(&path).unwrap();
    let parent = file.group("parent_05").unwrap();
    let child = parent.create_group("child").unwrap();
    assert_eq!(child.name(), "/parent_05/child");
    assert!(file
        .group("parent_05")
        .unwrap()
        .link_exists("child")
        .unwrap());
    assert!(file.group("parent_05/child").unwrap().is_empty().unwrap());

    let reopened = File::open(&path).unwrap();
    assert!(reopened
        .group("parent_05")
        .unwrap()
        .link_exists("child")
        .unwrap());
    assert!(reopened
        .group("parent_05/child")
        .unwrap()
        .is_empty()
        .unwrap());
}

#[test]
fn test_group_compat_create_child_inside_dense_group_open_rw() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("create_child_inside_dense_group.h5");
    {
        let mut writable = WritableFile::create(&path).unwrap();
        let mut parent = writable.create_group("parent").unwrap();
        for idx in 0..9 {
            parent.create_group(&format!("existing_{idx:02}")).unwrap();
        }
        writable.close().unwrap();
    }

    let file = File::open_rw(&path).unwrap();
    let parent = file.group("parent").unwrap();
    parent.create_group("new_child").unwrap();

    let parent = file.group("parent").unwrap();
    assert!(parent.link_exists("existing_00").unwrap());
    assert!(parent.link_exists("existing_08").unwrap());
    assert!(parent.link_exists("new_child").unwrap());
    assert!(file.group("parent/new_child").unwrap().is_empty().unwrap());

    let reopened = File::open(&path).unwrap();
    let parent = reopened.group("parent").unwrap();
    assert!(parent.link_exists("existing_00").unwrap());
    assert!(parent.link_exists("existing_08").unwrap());
    assert!(parent.link_exists("new_child").unwrap());
    assert!(reopened
        .group("parent/new_child")
        .unwrap()
        .is_empty()
        .unwrap());
}

#[test]
fn test_group_compat_create_dataset_inside_dense_group_open_rw() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("create_dataset_inside_dense_group.h5");
    {
        let mut writable = WritableFile::create(&path).unwrap();
        let mut parent = writable.create_group("parent").unwrap();
        for idx in 0..9 {
            parent.create_group(&format!("existing_{idx:02}")).unwrap();
        }
        writable.close().unwrap();
    }

    let file = File::open_rw(&path).unwrap();
    let parent = file.group("parent").unwrap();
    parent
        .new_dataset_builder()
        .with_data::<i32>(&[233, 377])
        .create(Some("values"))
        .unwrap();

    let reopened = File::open(&path).unwrap();
    let parent = reopened.group("parent").unwrap();
    assert!(parent.link_exists("existing_00").unwrap());
    assert!(parent.link_exists("existing_08").unwrap());
    assert!(parent.link_exists("values").unwrap());
    let mut values = vec![0; 2];
    reopened
        .dataset("parent/values")
        .unwrap()
        .read_into(&mut values)
        .unwrap();
    assert_eq!(values, [233, 377]);
}

#[test]
fn test_group_compat_create_soft_and_external_links_inside_dense_group_open_rw() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("create_links_inside_dense_group.h5");
    {
        let mut writable = WritableFile::create(&path).unwrap();
        let mut parent = writable.create_group("parent").unwrap();
        for idx in 0..9 {
            parent.create_group(&format!("existing_{idx:02}")).unwrap();
        }
        writable.close().unwrap();
    }

    let file = File::open_rw(&path).unwrap();
    let parent = file.group("parent").unwrap();
    parent.link_soft("/missing", "soft_child").unwrap();
    parent
        .link_external("external.h5", "/external_target", "external_child")
        .unwrap();

    let reopened = File::open(&path).unwrap();
    let parent = reopened.group("parent").unwrap();
    assert!(parent.link_exists("existing_00").unwrap());
    assert!(parent.link_exists("existing_08").unwrap());
    assert_eq!(parent.soft_link_target("soft_child").unwrap(), "/missing");
    assert_eq!(
        parent.external_link_target("external_child").unwrap(),
        ("external.h5".to_string(), "/external_target".to_string())
    );
}

#[test]
fn test_group_compat_create_hard_link_inside_dense_group_open_rw() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("create_hard_link_inside_dense_group.h5");
    {
        let mut writable = WritableFile::create(&path).unwrap();
        writable
            .new_dataset_builder("target")
            .write::<i32>(&[610, 987])
            .unwrap();
        let mut parent = writable.create_group("parent").unwrap();
        for idx in 0..9 {
            parent.create_group(&format!("existing_{idx:02}")).unwrap();
        }
        writable.close().unwrap();
    }

    let file = File::open_rw(&path).unwrap();
    let parent = file.group("parent").unwrap();
    parent.link_hard("/target", "alias").unwrap();

    let reopened = File::open(&path).unwrap();
    let parent = reopened.group("parent").unwrap();
    assert!(parent.link_exists("existing_00").unwrap());
    assert!(parent.link_exists("existing_08").unwrap());
    assert!(parent.link_exists("alias").unwrap());
    let mut values = vec![0; 2];
    reopened
        .dataset("parent/alias")
        .unwrap()
        .read_into(&mut values)
        .unwrap();
    assert_eq!(values, [610, 987]);
}

#[test]
fn test_group_compat_create_dataset_under_dense_root_parent_open_rw() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("create_dataset_under_dense_root_parent.h5");
    {
        let mut writable = WritableFile::create(&path).unwrap();
        for idx in 0..9 {
            writable.create_group(&format!("parent_{idx:02}")).unwrap();
        }
        writable.close().unwrap();
    }

    let file = File::open_rw(&path).unwrap();
    let parent = file.group("parent_05").unwrap();
    parent
        .new_dataset_builder()
        .with_data::<i32>(&[55, 89, 144])
        .create(Some("values"))
        .unwrap();

    let mut values = vec![0; 3];
    file.dataset("parent_05/values")
        .unwrap()
        .read_into(&mut values)
        .unwrap();
    assert_eq!(values, [55, 89, 144]);

    let reopened = File::open(&path).unwrap();
    assert!(reopened
        .group("parent_05")
        .unwrap()
        .link_exists("values")
        .unwrap());
    let mut reopened_values = vec![0; 3];
    reopened
        .dataset("parent_05/values")
        .unwrap()
        .read_into(&mut reopened_values)
        .unwrap();
    assert_eq!(reopened_values, [55, 89, 144]);
}

#[test]
fn test_group_compat_create_soft_and_external_links_under_dense_root_parent_open_rw() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("create_links_under_dense_root_parent.h5");
    {
        let mut writable = WritableFile::create(&path).unwrap();
        for idx in 0..9 {
            writable.create_group(&format!("parent_{idx:02}")).unwrap();
        }
        writable.close().unwrap();
    }

    let file = File::open_rw(&path).unwrap();
    let parent = file.group("parent_05").unwrap();
    parent.link_soft("/missing", "soft_child").unwrap();
    parent
        .link_external("external.h5", "/external_target", "external_child")
        .unwrap();

    let reopened = File::open(&path).unwrap();
    let parent = reopened.group("parent_05").unwrap();
    assert_eq!(parent.soft_link_target("soft_child").unwrap(), "/missing");
    assert_eq!(
        parent.external_link_target("external_child").unwrap(),
        ("external.h5".to_string(), "/external_target".to_string())
    );
}

#[test]
fn test_group_compat_create_same_group_hard_link_under_dense_root_parent_open_rw() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir
        .path()
        .join("create_same_group_hard_link_under_dense_root_parent.h5");
    {
        let mut writable = WritableFile::create(&path).unwrap();
        for idx in 0..9 {
            let mut parent = writable.create_group(&format!("parent_{idx:02}")).unwrap();
            if idx == 5 {
                parent
                    .new_dataset_builder("real_data")
                    .write::<i32>(&[21, 34])
                    .unwrap();
            }
        }
        writable.close().unwrap();
    }

    let file = File::open_rw(&path).unwrap();
    let parent = file.group("parent_05").unwrap();
    parent
        .link_hard("/parent_05/real_data", "alias_data")
        .unwrap();

    let parent = file.group("parent_05").unwrap();
    assert!(parent.link_exists("real_data").unwrap());
    assert!(parent.link_exists("alias_data").unwrap());
    let real_addr = parent
        .link_info("real_data")
        .unwrap()
        .hard_link_addr
        .unwrap();
    let alias_addr = parent
        .link_info("alias_data")
        .unwrap()
        .hard_link_addr
        .unwrap();
    assert_eq!(real_addr, alias_addr);
    assert_eq!(object_refcount_at(&path, &file, real_addr), 2);

    let reopened = File::open(&path).unwrap();
    let parent = reopened.group("parent_05").unwrap();
    assert!(parent.link_exists("alias_data").unwrap());
    let mut values = vec![0; 2];
    reopened
        .dataset("parent_05/alias_data")
        .unwrap()
        .read_into(&mut values)
        .unwrap();
    assert_eq!(values, [21, 34]);
}

#[test]
fn test_group_compat_create_same_group_hard_link_inside_dense_parent_open_rw() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir
        .path()
        .join("create_same_group_hard_link_inside_dense_parent.h5");
    {
        let mut writable = WritableFile::create(&path).unwrap();
        let mut parent = writable.create_group("parent").unwrap();
        for idx in 0..9 {
            parent
                .new_dataset_builder(&format!("child_{idx:02}"))
                .write::<i32>(&[idx])
                .unwrap();
        }
        parent
            .new_dataset_builder("real_data")
            .write::<i32>(&[55, 89])
            .unwrap();
        writable.close().unwrap();
    }

    let file = File::open_rw(&path).unwrap();
    let parent = file.group("parent").unwrap();
    let initial_count = group_member_count(&parent).unwrap();
    let old_real_addr = parent
        .link_info("real_data")
        .unwrap()
        .hard_link_addr
        .unwrap();
    assert_eq!(object_refcount_at(&path, &file, old_real_addr), 1);

    parent.link_hard("real_data", "alias_data").unwrap();

    let parent = file.group("parent").unwrap();
    assert!(parent.link_exists("real_data").unwrap());
    assert!(parent.link_exists("alias_data").unwrap());
    let real_addr = parent
        .link_info("real_data")
        .unwrap()
        .hard_link_addr
        .unwrap();
    let alias_addr = parent
        .link_info("alias_data")
        .unwrap()
        .hard_link_addr
        .unwrap();
    assert_eq!(real_addr, alias_addr);
    assert_eq!(object_refcount_at(&path, &file, real_addr), 2);
    assert_eq!(group_member_count(&parent).unwrap(), initial_count + 1);

    let reopened = File::open(&path).unwrap();
    let parent = reopened.group("parent").unwrap();
    assert!(parent.link_exists("real_data").unwrap());
    assert!(parent.link_exists("alias_data").unwrap());
    let real_addr = parent
        .link_info("real_data")
        .unwrap()
        .hard_link_addr
        .unwrap();
    let alias_addr = parent
        .link_info("alias_data")
        .unwrap()
        .hard_link_addr
        .unwrap();
    assert_eq!(real_addr, alias_addr);
    assert_eq!(object_refcount_at(&path, &reopened, real_addr), 2);
    assert_eq!(group_member_count(&parent).unwrap(), initial_count + 1);
    let mut values = vec![0; 2];
    reopened
        .dataset("parent/alias_data")
        .unwrap()
        .read_into(&mut values)
        .unwrap();
    assert_eq!(values, [55, 89]);
}

#[test]
fn test_group_compat_open_rw_relative_hard_link_target_in_same_group() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("open_rw_relative_hard_link_target.h5");
    {
        let mut writable = WritableFile::create(&path).unwrap();
        let mut parent = writable.create_group("parent").unwrap();
        parent
            .new_dataset_builder("real_data")
            .write::<i32>(&[13, 21])
            .unwrap();
        writable.close().unwrap();
    }

    let file = File::open_rw(&path).unwrap();
    let parent = file.group("parent").unwrap();
    parent.link_hard("real_data", "alias_data").unwrap();

    let parent = file.group("parent").unwrap();
    let real_addr = parent
        .link_info("real_data")
        .unwrap()
        .hard_link_addr
        .unwrap();
    let alias_addr = parent
        .link_info("alias_data")
        .unwrap()
        .hard_link_addr
        .unwrap();
    assert_eq!(real_addr, alias_addr);
    assert_eq!(object_refcount_at(&path, &file, real_addr), 2);

    let mut values = vec![0; 2];
    File::open(&path)
        .unwrap()
        .dataset("parent/alias_data")
        .unwrap()
        .read_into(&mut values)
        .unwrap();
    assert_eq!(values, [13, 21]);
}

#[test]
fn test_group_compat_open_rw_relative_hard_link_target_group_in_same_group() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir
        .path()
        .join("open_rw_relative_hard_link_group_target.h5");
    {
        let mut writable = WritableFile::create(&path).unwrap();
        let mut parent = writable.create_group("parent").unwrap();
        parent.create_group("child").unwrap();
        writable.close().unwrap();
    }

    let file = File::open_rw(&path).unwrap();
    let parent = file.group("parent").unwrap();
    parent.link_hard("child", "child_alias").unwrap();

    let parent = file.group("parent").unwrap();
    let child_addr = parent.link_info("child").unwrap().hard_link_addr.unwrap();
    let alias_addr = parent
        .link_info("child_alias")
        .unwrap()
        .hard_link_addr
        .unwrap();
    assert_eq!(child_addr, alias_addr);
    assert_eq!(object_refcount_at(&path, &file, child_addr), 2);
    assert_eq!(
        parent.member_type("child_alias").unwrap(),
        hdf5_pure_rust::hl::file::ObjectType::Group
    );
    assert!(File::open(&path)
        .unwrap()
        .group("parent/child_alias")
        .unwrap()
        .is_empty()
        .unwrap());
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
fn test_group_compat_mutation_entry_points_require_read_write_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("readonly_group_mutation_entry_points.h5");
    {
        let mut writable = WritableFile::create(&path).unwrap();
        writable.create_group("existing").unwrap();
        writable
            .new_dataset_builder("data")
            .write::<i32>(&[1, 2, 3])
            .unwrap();
        writable.close().unwrap();
    }

    let file = File::open(&path).unwrap();
    let root = file.root_group().unwrap();

    let create_group_err = match root.create_group("created") {
        Ok(_) => panic!("read-only create_group should fail"),
        Err(err) => err,
    };
    let create_dataset_err = match root
        .new_dataset_builder()
        .with_data::<i32>(&[4, 5, 6])
        .create(Some("values"))
    {
        Ok(_) => panic!("read-only dataset creation should fail"),
        Err(err) => err,
    };

    for err in [
        create_group_err,
        create_dataset_err,
        root.link_soft("/data", "soft_alias").unwrap_err(),
        root.link_external("other.h5", "/data", "external_alias")
            .unwrap_err(),
        root.link_hard("/data", "hard_alias").unwrap_err(),
        root.relink("existing", "renamed").unwrap_err(),
    ] {
        assert!(
            matches!(err, hdf5_pure_rust::Error::Unsupported(_)),
            "read-only group mutation should be explicitly unsupported: {err}"
        );
    }

    let root = File::open(&path).unwrap().root_group().unwrap();
    assert!(root.link_exists("existing").unwrap());
    assert!(root.link_exists("data").unwrap());
    assert!(!root.link_exists("created").unwrap());
    assert!(!root.link_exists("values").unwrap());
    assert!(!root.link_exists("soft_alias").unwrap());
    assert!(!root.link_exists("external_alias").unwrap());
    assert!(!root.link_exists("hard_alias").unwrap());
    assert!(!root.link_exists("renamed").unwrap());
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
fn test_file_and_group_list_into_helpers_replace_stale_output_on_success() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("list_into_success_replaces_stale.h5");
    {
        let mut writable = WritableFile::create(&path).unwrap();
        writable.add_attr("root_attr", 7i32).unwrap();
        let mut group = writable.create_group("parent").unwrap();
        group.add_attr("group_attr", 11i32).unwrap();
        group.create_group("child").unwrap();
        writable.close().unwrap();
    }

    let file = File::open(&path).unwrap();
    let mut file_names = vec!["stale".to_string()];
    file.member_names_into(&mut file_names).unwrap();
    assert_eq!(file_names, vec!["parent"]);

    let mut file_attr_names = vec!["stale".to_string()];
    file.attr_names_into(&mut file_attr_names).unwrap();
    assert_eq!(file_attr_names, vec!["root_attr"]);

    let mut file_attrs = Vec::new();
    file.attrs_into(&mut file_attrs).unwrap();
    assert_eq!(file_attrs.len(), 1);
    assert_eq!(file_attrs[0].name(), "root_attr");

    let group = file.group("parent").unwrap();
    let mut group_names = vec!["stale".to_string()];
    group.member_names_into(&mut group_names).unwrap();
    assert_eq!(group_names, vec!["child"]);

    let mut group_attr_names = vec!["stale".to_string()];
    group.attr_names_into(&mut group_attr_names).unwrap();
    assert_eq!(group_attr_names, vec!["group_attr"]);

    let mut group_attrs = Vec::new();
    group.attrs_into(&mut group_attrs).unwrap();
    assert_eq!(group_attrs.len(), 1);
    assert_eq!(group_attrs[0].name(), "group_attr");

    let mut ids = vec![u64::MAX];
    file.obj_ids_into(&mut ids);
    assert!(!ids.contains(&u64::MAX));
    assert!(ids.contains(&file.object_id()));
    assert!(ids.contains(&group.object_id()));
}

#[test]
fn test_file_and_group_list_into_helpers_preserve_stale_output_on_errors() {
    let file = File::open("tests/data/attrs.h5").unwrap();
    let stale_attr = file.attr("int_attr").unwrap();
    let mut attrs = vec![stale_attr.clone()];
    let err = file.attrs_by_creation_order_into(&mut attrs).unwrap_err();
    assert!(matches!(err, hdf5_pure_rust::Error::Unsupported(_)));
    assert_eq!(attrs.len(), 1);
    assert_eq!(attrs[0].name(), stale_attr.name());

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("list_into_error_preserves_stale.h5");
    {
        let mut writable = WritableFile::create(&path).unwrap();
        writable.add_attr("root_attr", 7i32).unwrap();
        writable.create_group("parent").unwrap();
        writable.close().unwrap();
    }

    let file = File::open_rw(&path).unwrap();
    let root = file.root_group().unwrap();
    let parent = file.group("parent").unwrap();
    root.relink("parent", "renamed_parent").unwrap();

    let mut names = vec!["stale".to_string()];
    assert!(parent.member_names_into(&mut names).is_err());
    assert_eq!(names, vec!["stale"]);

    let mut attr_names = vec!["stale".to_string()];
    assert!(parent.attr_names_into(&mut attr_names).is_err());
    assert_eq!(attr_names, vec!["stale"]);

    let stale_root_attr = file.attr("root_attr").unwrap();
    let mut parent_attrs = vec![stale_root_attr.clone()];
    assert!(parent.attrs_into(&mut parent_attrs).is_err());
    assert_eq!(parent_attrs.len(), 1);
    assert_eq!(parent_attrs[0].name(), stale_root_attr.name());
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
