use hdf5_pure_rust::format::messages::link::{LinkMessage, LinkType};
use hdf5_pure_rust::{Error, File, LinkAccess, WritableFile};

#[test]
fn test_link_access_defaults() {
    let access = LinkAccess::new();
    assert_eq!(access.nlinks(), 40);
    assert_eq!(access.elink_prefix(), None);
    assert_eq!(access.elink_fapl().driver(), "sec2");
    assert_eq!(access.elink_acc_flags(), 0);
    assert_eq!(access.elink_cb(), None);
    assert_eq!(access.elink_file_cache_size(), 0);
}

#[test]
fn test_write_and_read_soft_link() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("soft_link_test.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("real_data")
            .write::<f64>(&[1.0, 2.0, 3.0])
            .unwrap();
        wf.link_soft("alias", "/real_data").unwrap();
        wf.flush().unwrap();
    }

    {
        let f = File::open(&path).unwrap();
        let names = f.member_names().unwrap();
        assert!(names.contains(&"real_data".to_string()));
        assert!(names.contains(&"alias".to_string()));

        let root = f.root_group().unwrap();
        let links = root.links().unwrap();
        assert!(links
            .iter()
            .any(|link| link.name == "alias" && link.link_type == LinkType::Soft));

        let lt = root.link_type("alias").unwrap();
        assert_eq!(lt, LinkType::Soft);

        let target = root.soft_link_target("alias").unwrap();
        assert_eq!(target, "/real_data");
    }
}

#[test]
fn test_write_and_read_hard_links() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("hard_link_test.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("real_data")
            .write::<i32>(&[7, 8, 9])
            .unwrap();
        let mut group = wf.create_group("aliases").unwrap();
        group.link_hard("nested_data", "/real_data").unwrap();
        wf.link_hard("alias_data", "/real_data").unwrap();
        wf.link_hard("alias_group", "/aliases").unwrap();
        assert!(wf.link_hard("missing", "/does_not_exist").is_err());
        wf.flush().unwrap();
    }

    let f = File::open(&path).unwrap();
    assert_eq!(
        f.dataset("alias_data").unwrap().read::<i32>().unwrap(),
        vec![7, 8, 9]
    );
    assert_eq!(
        f.dataset("aliases/nested_data")
            .unwrap()
            .read::<i32>()
            .unwrap(),
        vec![7, 8, 9]
    );
    assert_eq!(f.group("alias_group").unwrap().name(), "/alias_group");

    let root = f.root_group().unwrap();
    let links = root.links().unwrap();
    assert!(links
        .iter()
        .any(|link| link.name == "alias_data" && link.link_type == LinkType::Hard));
    assert!(links
        .iter()
        .any(|link| link.name == "alias_group" && link.link_type == LinkType::Hard));
}

#[test]
fn test_soft_link_resolution_and_cycle_limit() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("soft_link_resolution.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("real_data")
            .write::<i32>(&[10, 20, 30])
            .unwrap();
        wf.create_group("real_group").unwrap();
        wf.link_soft("alias_data", "/real_data").unwrap();
        wf.link_soft("alias_group", "/real_group").unwrap();
        wf.link_soft("cycle_a", "/cycle_b").unwrap();
        wf.link_soft("cycle_b", "/cycle_a").unwrap();
        wf.flush().unwrap();
    }

    let f = File::open(&path).unwrap();
    let alias_values: Vec<i32> = f.dataset("alias_data").unwrap().read().unwrap();
    assert_eq!(alias_values, vec![10, 20, 30]);
    assert_eq!(f.group("alias_group").unwrap().name(), "/real_group");

    let err = match f.dataset("cycle_a") {
        Ok(_) => panic!("soft-link cycle should hit traversal limit"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::InvalidFormat(_)));
    assert!(err.to_string().contains("soft link cycle"));
}

#[test]
fn test_soft_link_resolution_normalizes_relative_targets() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("soft_link_relative_resolution.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        let mut real = wf.create_group("real").unwrap();
        real.new_dataset_builder("data")
            .write::<i32>(&[11, 22])
            .unwrap();
        let mut aliases = wf.create_group("aliases").unwrap();
        aliases
            .link_soft("relative_data", "../real/./data")
            .unwrap();
        aliases.link_soft("relative_group", "../real").unwrap();
        wf.link_soft("through_alias", "/aliases/relative_data")
            .unwrap();
        wf.flush().unwrap();
    }

    let f = File::open(&path).unwrap();
    assert_eq!(
        f.dataset("aliases/relative_data")
            .unwrap()
            .read::<i32>()
            .unwrap(),
        vec![11, 22]
    );
    assert_eq!(
        f.dataset("through_alias").unwrap().read::<i32>().unwrap(),
        vec![11, 22]
    );
    let aliases = f.group("aliases").unwrap();
    assert_eq!(
        aliases
            .open_dataset("relative_data")
            .unwrap()
            .read::<i32>()
            .unwrap(),
        vec![11, 22]
    );
    assert_eq!(
        aliases.member_type("relative_group").unwrap(),
        hdf5_pure_rust::hl::file::ObjectType::Group
    );
    assert_eq!(
        aliases.open_group("relative_group").unwrap().name(),
        "/real"
    );
}

#[test]
fn test_link_exists() {
    let f = File::open("tests/data/simple_v0.h5").unwrap();
    let root = f.root_group().unwrap();
    assert!(root.link_exists("data").unwrap());
    assert!(root.link_exists("group1").unwrap());
    assert!(!root.link_exists("nonexistent").unwrap());
}

#[test]
fn test_link_exists_sees_soft_and_external_links() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("link_exists_non_hard.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("real_data")
            .write::<i32>(&[1, 2, 3])
            .unwrap();
        wf.link_soft("soft_alias", "/real_data").unwrap();
        wf.link_external("external_alias", "missing.h5", "/data")
            .unwrap();
        wf.flush().unwrap();
    }

    let f = File::open(&path).unwrap();
    let root = f.root_group().unwrap();
    assert!(root.link_exists("soft_alias").unwrap());
    assert!(root.link_exists("external_alias").unwrap());
    assert!(!root.link_exists("missing_alias").unwrap());
}

#[test]
fn test_write_external_link() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ext_link_test.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.link_external("remote", "other_file.h5", "/some/dataset")
            .unwrap();
        wf.flush().unwrap();
    }

    {
        let f = File::open(&path).unwrap();
        let names = f.member_names().unwrap();
        assert!(names.contains(&"remote".to_string()));

        let root = f.root_group().unwrap();
        let links = root.links().unwrap();
        assert!(links
            .iter()
            .any(|link| link.name == "remote" && link.link_type == LinkType::External));

        let lt = root.link_type("remote").unwrap();
        assert_eq!(lt, LinkType::External);

        let (filename, obj_path) = root.external_link_target("remote").unwrap();
        assert_eq!(filename, "other_file.h5");
        assert_eq!(obj_path, "/some/dataset");
    }
}

#[test]
fn test_write_links_reject_oversized_link_values() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("oversized_link_values.h5");

    let mut wf = WritableFile::create(&path).unwrap();
    let long_target = format!("/{}", "x".repeat(u16::MAX as usize + 1));
    let err = wf
        .link_soft("too_long", &long_target)
        .expect_err("soft link target should fit u16 length field");
    assert!(err.to_string().contains("soft link target"));

    let long_filename = "x".repeat(u16::MAX as usize + 1);
    let err = wf
        .link_external("too_long_external", &long_filename, "/data")
        .expect_err("external link info should fit u16 length field");
    assert!(err.to_string().contains("external link info"));
}

#[test]
fn test_external_link_traversal_missing_relative_absolute_and_same_directory() {
    let dir = tempfile::tempdir().unwrap();
    let source_path = dir.path().join("source.h5");
    let target_path = dir.path().join("target.h5");
    let nested_dir = dir.path().join("nested");
    std::fs::create_dir(&nested_dir).unwrap();
    let nested_target_path = nested_dir.join("nested_target.h5");

    {
        let mut target = WritableFile::create(&target_path).unwrap();
        target
            .new_dataset_builder("data")
            .write::<i32>(&[1, 2, 3])
            .unwrap();
        target.create_group("group").unwrap();
        target.flush().unwrap();

        let mut nested = WritableFile::create(&nested_target_path).unwrap();
        nested
            .new_dataset_builder("data")
            .write::<i32>(&[4, 5, 6])
            .unwrap();
        nested.flush().unwrap();

        let mut source = WritableFile::create(&source_path).unwrap();
        source
            .link_external("same_dir", "target.h5", "/data")
            .unwrap();
        source
            .link_external("relative", "nested/nested_target.h5", "/data")
            .unwrap();
        source
            .link_external("absolute", target_path.to_str().unwrap(), "/data")
            .unwrap();
        source
            .link_external("remote_group", "target.h5", "/group")
            .unwrap();
        source
            .link_external("missing", "missing.h5", "/data")
            .unwrap();
        source.flush().unwrap();
    }

    let f = File::open(&source_path).unwrap();
    assert_eq!(
        f.dataset("same_dir").unwrap().read::<i32>().unwrap(),
        vec![1, 2, 3]
    );
    assert_eq!(
        f.dataset("relative").unwrap().read::<i32>().unwrap(),
        vec![4, 5, 6]
    );
    assert_eq!(
        f.dataset("absolute").unwrap().read::<i32>().unwrap(),
        vec![1, 2, 3]
    );
    assert_eq!(f.group("remote_group").unwrap().name(), "/group");
    assert!(matches!(f.dataset("missing"), Err(Error::Io(_))));
}

#[test]
fn test_utf8_link_names_and_non_ascii_external_filename() {
    let f = File::open("tests/data/hdf5_ref/link_edge_cases.h5").unwrap();
    let root = f.root_group().unwrap();
    let names = root.member_names().unwrap();

    assert!(names.contains(&"猫_group".to_string()));
    assert!(names.contains(&"å_link".to_string()));
    assert!(names.contains(&"external_å".to_string()));
    assert_eq!(
        root.member_type("å_link").unwrap(),
        hdf5_pure_rust::hl::file::ObjectType::Dataset
    );

    let (filename, object_path) = root.external_link_target("external_å").unwrap();
    assert_eq!(filename, "målfil.h5");
    assert_eq!(object_path, "/dåta");
}

#[test]
fn test_link_decoder_rejects_invalid_character_encoding() {
    let mut raw = vec![1, 0x10, 2, 1, b'x'];
    raw.extend_from_slice(&0u64.to_le_bytes());
    let err = LinkMessage::decode(&raw, 8).expect_err("invalid link cset should fail");
    assert!(matches!(err, Error::InvalidFormat(_)));
}
