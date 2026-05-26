use hdf5_pure_rust::format::messages::link::{LinkMessage, LinkType};
use hdf5_pure_rust::{Dataset, Error, File, Group, LinkAccess, Result, WritableFile};

fn file_has_member(file: &File, expected: &str) -> Result<bool> {
    let mut found = false;
    file.visit_member_names(|name| {
        found |= name == expected;
        Ok(())
    })?;
    Ok(found)
}

fn group_has_member(group: &Group, expected: &str) -> Result<bool> {
    let mut found = false;
    group.visit_member_names(|name| {
        found |= name == expected;
        Ok(())
    })?;
    Ok(found)
}

fn group_has_link(group: &Group, expected: &str, expected_type: LinkType) -> Result<bool> {
    let mut found = false;
    group.visit_links(|link| {
        found |= link.name == expected && link.link_type == expected_type;
        Ok(())
    })?;
    Ok(found)
}

fn assert_i32_dataset_values(dataset: &Dataset, expected: &[i32]) -> Result<()> {
    let mut shape = Vec::new();
    dataset.shape_into(&mut shape)?;
    let len = shape.iter().map(|dim| *dim as usize).product();
    let mut values = vec![0; len];
    dataset.read_into(&mut values)?;
    assert_eq!(values.as_slice(), expected);
    Ok(())
}

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
        assert!(file_has_member(&f, "real_data").unwrap());
        assert!(file_has_member(&f, "alias").unwrap());

        let root = f.root_group().unwrap();
        assert!(group_has_link(&root, "alias", LinkType::Soft).unwrap());

        let lt = root.link_type("alias").unwrap();
        assert_eq!(lt, LinkType::Soft);

        root.soft_link_target_with("alias", |target| {
            assert_eq!(target, "/real_data");
            Ok(())
        })
        .unwrap();
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
    assert_i32_dataset_values(&f.dataset("alias_data").unwrap(), &[7, 8, 9]).unwrap();
    assert_i32_dataset_values(&f.dataset("aliases/nested_data").unwrap(), &[7, 8, 9]).unwrap();
    assert_eq!(f.group("alias_group").unwrap().name(), "/alias_group");

    let root = f.root_group().unwrap();
    assert!(group_has_link(&root, "alias_data", LinkType::Hard).unwrap());
    assert!(group_has_link(&root, "alias_group", LinkType::Hard).unwrap());
}

#[test]
fn test_link_name_by_idx_into_preserves_output_on_missing_index() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("link_name_by_idx_into.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("a").write::<i32>(&[1]).unwrap();
        wf.new_dataset_builder("b").write::<i32>(&[2]).unwrap();
        wf.flush().unwrap();
    }

    let f = File::open(&path).unwrap();
    let root = f.root_group().unwrap();
    let mut name = String::from("stale");

    hdf5_pure_rust::hl::link::get_name_by_idx_into(&root, 0, &mut name).unwrap();
    assert!(name == "a" || name == "b");

    name = String::from("stale");
    let err = hdf5_pure_rust::hl::link::get_name_by_idx_into(&root, 2, &mut name)
        .expect_err("out-of-bounds link index should fail");
    assert!(err.to_string().contains("out of bounds"));
    assert_eq!(name, "stale");
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
    assert_i32_dataset_values(&f.dataset("alias_data").unwrap(), &[10, 20, 30]).unwrap();
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
        aliases
            .link_soft("relative_group_dotted", "./../real/.")
            .unwrap();
        wf.link_soft("through_alias", "/aliases/relative_data")
            .unwrap();
        wf.flush().unwrap();
    }

    let f = File::open(&path).unwrap();
    assert_i32_dataset_values(&f.dataset("aliases/relative_data").unwrap(), &[11, 22]).unwrap();
    assert_i32_dataset_values(&f.dataset("through_alias").unwrap(), &[11, 22]).unwrap();
    let aliases = f.group("aliases").unwrap();
    assert_i32_dataset_values(&aliases.open_dataset("relative_data").unwrap(), &[11, 22]).unwrap();
    assert_eq!(
        aliases.member_type("relative_group").unwrap(),
        hdf5_pure_rust::hl::file::ObjectType::Group
    );
    assert_eq!(
        aliases.open_group("relative_group").unwrap().name(),
        "/real"
    );
    assert_i32_dataset_values(
        &f.dataset("aliases/relative_group_dotted/data").unwrap(),
        &[11, 22],
    )
    .unwrap();
}

#[test]
fn test_soft_link_resolution_normalizes_deep_relative_targets() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("soft_link_deep_relative_resolution.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        let mut real = wf.create_group("real").unwrap();
        real.new_dataset_builder("data")
            .write::<i32>(&[77, 88])
            .unwrap();
        let mut aliases = wf.create_group("aliases").unwrap();
        let mut nested = aliases.create_group("nested").unwrap();
        nested
            .link_soft("deep_relative_data", "../.././real//data")
            .unwrap();
        nested
            .link_soft(
                "absolute_with_dotdots",
                "/aliases/nested/../nested/../../real/data",
            )
            .unwrap();
        wf.flush().unwrap();
    }

    let f = File::open(&path).unwrap();
    assert_i32_dataset_values(
        &f.dataset("aliases/nested/deep_relative_data").unwrap(),
        &[77, 88],
    )
    .unwrap();
    assert_i32_dataset_values(
        &f.dataset("aliases/nested/absolute_with_dotdots").unwrap(),
        &[77, 88],
    )
    .unwrap();
}

#[test]
fn test_soft_link_resolution_clamps_parent_walks_at_root() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("soft_link_root_clamp_resolution.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        let mut real = wf.create_group("real").unwrap();
        real.new_dataset_builder("data")
            .write::<i32>(&[101, 202])
            .unwrap();
        let mut aliases = wf.create_group("aliases").unwrap();
        aliases
            .link_soft("above_root_group", "../../../real")
            .unwrap();
        wf.flush().unwrap();
    }

    let f = File::open(&path).unwrap();
    assert_i32_dataset_values(
        &f.dataset("aliases/above_root_group/data").unwrap(),
        &[101, 202],
    )
    .unwrap();
}

#[test]
fn test_soft_link_resolution_normalizes_group_targets_with_remaining_path() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir
        .path()
        .join("soft_link_group_target_remaining_path_resolution.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        let mut real = wf.create_group("real").unwrap();
        let mut deep = real.create_group("deep").unwrap();
        deep.new_dataset_builder("data")
            .write::<i32>(&[303, 404])
            .unwrap();
        let mut aliases = wf.create_group("aliases").unwrap();
        let mut nested = aliases.create_group("nested").unwrap();
        nested
            .link_soft("group_alias", "../.././real//deep/..")
            .unwrap();
        nested
            .link_soft("clamped_group", "../../../../real/./")
            .unwrap();
        wf.link_soft("through_group_alias", "/aliases/nested/group_alias/deep")
            .unwrap();
        wf.flush().unwrap();
    }

    let f = File::open(&path).unwrap();
    assert_i32_dataset_values(
        &f.dataset("aliases/nested/group_alias/deep/./data").unwrap(),
        &[303, 404],
    )
    .unwrap();
    assert_i32_dataset_values(
        &f.dataset("aliases/nested/clamped_group/deep/data").unwrap(),
        &[303, 404],
    )
    .unwrap();
    assert_i32_dataset_values(&f.dataset("through_group_alias/data").unwrap(), &[303, 404])
        .unwrap();
}

#[test]
fn test_soft_link_resolution_normalizes_chained_relative_parent_targets() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir
        .path()
        .join("soft_link_chained_relative_parent_targets.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        let mut org = wf.create_group("org").unwrap();
        let mut team_a = org.create_group("team_a").unwrap();
        let mut datasets = team_a.create_group("datasets").unwrap();
        let mut year = datasets.create_group("2026").unwrap();
        let mut run = year.create_group("run").unwrap();
        run.new_dataset_builder("data")
            .write::<i32>(&[505, 606])
            .unwrap();

        team_a
            .link_soft("current", "datasets/./2026/../2026")
            .unwrap();
        let mut team_b = org.create_group("team_b").unwrap();
        let mut views = team_b.create_group("views").unwrap();
        views
            .link_soft("current_run", "../../team_a/./current/run/..")
            .unwrap();
        wf.flush().unwrap();
    }

    let f = File::open(&path).unwrap();
    assert_i32_dataset_values(
        &f.dataset("org/team_b/views/current_run/run/data").unwrap(),
        &[505, 606],
    )
    .unwrap();

    let views = f.group("org/team_b/views").unwrap();
    assert_i32_dataset_values(
        &views.open_dataset("current_run/run/data").unwrap(),
        &[505, 606],
    )
    .unwrap();
}

#[test]
fn test_soft_link_resolution_normalizes_remaining_parent_after_current_group_target() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir
        .path()
        .join("soft_link_remaining_parent_after_current_group_target.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        let mut aliases = wf.create_group("aliases").unwrap();
        let mut nested = aliases.create_group("nested").unwrap();
        nested.link_soft("here", ".").unwrap();
        let mut sibling = aliases.create_group("sibling").unwrap();
        sibling
            .new_dataset_builder("data")
            .write::<i32>(&[707, 808])
            .unwrap();
        wf.flush().unwrap();
    }

    let f = File::open(&path).unwrap();
    assert_i32_dataset_values(
        &f.dataset("aliases/nested/here/../sibling/data").unwrap(),
        &[707, 808],
    )
    .unwrap();

    let nested = f.group("aliases/nested").unwrap();
    assert_i32_dataset_values(
        &nested.open_dataset("here/../sibling/data").unwrap(),
        &[707, 808],
    )
    .unwrap();
}

#[test]
fn test_soft_link_resolution_normalizes_remaining_parent_after_absolute_group_target() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir
        .path()
        .join("soft_link_remaining_parent_after_absolute_group_target.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        let mut root_target = wf.create_group("root_target").unwrap();
        let mut leaf = root_target.create_group("leaf").unwrap();
        leaf.new_dataset_builder("data")
            .write::<i32>(&[909, 1001])
            .unwrap();
        let mut aliases = wf.create_group("aliases").unwrap();
        aliases
            .link_soft("absolute_leaf", "/root_target/leaf")
            .unwrap();
        wf.flush().unwrap();
    }

    let f = File::open(&path).unwrap();
    assert_i32_dataset_values(
        &f.dataset("aliases/absolute_leaf/../leaf/data").unwrap(),
        &[909, 1001],
    )
    .unwrap();

    let aliases = f.group("aliases").unwrap();
    assert_i32_dataset_values(
        &aliases.open_dataset("absolute_leaf/../leaf/data").unwrap(),
        &[909, 1001],
    )
    .unwrap();
}

#[test]
fn test_soft_link_resolution_normalizes_remaining_path_after_root_target() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir
        .path()
        .join("soft_link_remaining_path_after_root_target.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("root_data")
            .write::<i32>(&[111, 222])
            .unwrap();
        let mut aliases = wf.create_group("aliases").unwrap();
        aliases.link_soft("root_alias", "/").unwrap();
        wf.flush().unwrap();
    }

    let f = File::open(&path).unwrap();
    assert_i32_dataset_values(
        &f.dataset("aliases/root_alias/./aliases/../root_data")
            .unwrap(),
        &[111, 222],
    )
    .unwrap();

    let aliases = f.group("aliases").unwrap();
    assert_i32_dataset_values(
        &aliases
            .open_dataset("root_alias/./aliases/../root_data")
            .unwrap(),
        &[111, 222],
    )
    .unwrap();
}

#[test]
fn test_soft_link_cycle_detected_after_relative_target_normalization() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("soft_link_normalized_cycle.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        let mut aliases = wf.create_group("aliases").unwrap();
        aliases
            .link_soft("self_cycle", "./../aliases/./self_cycle")
            .unwrap();
        wf.flush().unwrap();
    }

    let f = File::open(&path).unwrap();
    let root = f.root_group().unwrap();
    assert!(group_has_link(
        &root.open_group("aliases").unwrap(),
        "self_cycle",
        LinkType::Soft
    )
    .unwrap());

    let err = match f.dataset("aliases/self_cycle") {
        Ok(_) => panic!("normalized relative soft link should resolve back to itself"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::InvalidFormat(_)));
    assert!(err.to_string().contains("soft link cycle"));
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
fn test_dense_links_include_soft_external_and_hard_aliases() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dense_alias_links.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        for idx in 0..9 {
            wf.new_dataset_builder(&format!("data_{idx:02}"))
                .write::<i32>(&[idx])
                .unwrap();
        }
        wf.link_soft("soft_alias", "/data_00").unwrap();
        wf.link_external("external_alias", "missing.h5", "/remote")
            .unwrap();
        wf.link_hard("hard_alias", "/data_01").unwrap();
        wf.flush().unwrap();
    }

    let f = File::open(&path).unwrap();
    assert!(file_has_member(&f, "data_08").unwrap());
    assert!(file_has_member(&f, "soft_alias").unwrap());
    assert!(file_has_member(&f, "external_alias").unwrap());
    assert!(file_has_member(&f, "hard_alias").unwrap());
    assert_i32_dataset_values(&f.dataset("hard_alias").unwrap(), &[1]).unwrap();

    let root = f.root_group().unwrap();
    assert!(group_has_link(&root, "data_08", LinkType::Hard).unwrap());
    assert!(group_has_link(&root, "soft_alias", LinkType::Soft).unwrap());
    assert!(group_has_link(&root, "external_alias", LinkType::External).unwrap());
    assert!(group_has_link(&root, "hard_alias", LinkType::Hard).unwrap());

    root.soft_link_target_with("soft_alias", |target| {
        assert_eq!(target, "/data_00");
        Ok(())
    })
    .unwrap();
    root.external_link_target_with("external_alias", |filename, obj_path| {
        assert_eq!(filename, "missing.h5");
        assert_eq!(obj_path, "/remote");
        Ok(())
    })
    .unwrap();

    let out = std::process::Command::new("h5dump")
        .arg("-H")
        .arg(&path)
        .output();
    if let Ok(out) = out {
        assert!(
            out.status.success(),
            "h5dump -H failed on dense mixed-link writer fixture: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(stdout.contains("soft_alias"));
        assert!(stdout.contains("external_alias"));
        assert!(stdout.contains("hard_alias"));
    }

    let output = std::process::Command::new("python3")
        .arg("-c")
        .arg(
            "import sys, importlib.util\n\
             spec = importlib.util.find_spec('h5py')\n\
             (print('SKIP h5py unavailable'), sys.exit(0)) if spec is None else None\n\
             import h5py\n\
             f = h5py.File(sys.argv[1], 'r')\n\
             assert int(f['hard_alias'][0]) == 1\n\
             assert int(f['soft_alias'][0]) == 0\n\
             soft = f.get('soft_alias', getlink=True)\n\
             external = f.get('external_alias', getlink=True)\n\
             assert isinstance(soft, h5py.SoftLink)\n\
             assert soft.path == '/data_00'\n\
             assert isinstance(external, h5py.ExternalLink)\n\
             assert external.filename == 'missing.h5'\n\
             assert external.path == '/remote'\n\
             f.close()\n\
             print('OK')",
        )
        .arg(&path)
        .output();
    if let Ok(out) = output {
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            out.status.success() && (stdout.contains("OK") || stdout.contains("SKIP")),
            "h5py failed on dense mixed-link writer fixture: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

#[test]
fn test_group_compat_unlink_dense_root_soft_and_external_links() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("open_rw_root_dense_unlink_non_hard.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        for idx in 0..9 {
            wf.new_dataset_builder(&format!("data_{idx:02}"))
                .write::<i32>(&[idx])
                .unwrap();
        }
        wf.link_soft("soft_alias", "/data_00").unwrap();
        wf.link_external("external_alias", "missing.h5", "/remote")
            .unwrap();
        wf.flush().unwrap();
    }

    let file = File::open_rw(&path).unwrap();
    let root = file.root_group().unwrap();
    root.unlink("soft_alias").unwrap();
    root.unlink("external_alias").unwrap();

    assert!(!root.link_exists("soft_alias").unwrap());
    assert!(!root.link_exists("external_alias").unwrap());
    assert!(root.link_exists("data_08").unwrap());
    assert_i32_dataset_values(&file.dataset("data_08").unwrap(), &[8]).unwrap();

    let reopened = File::open(&path).unwrap();
    assert!(!file_has_member(&reopened, "soft_alias").unwrap());
    assert!(!file_has_member(&reopened, "external_alias").unwrap());
    assert!(file_has_member(&reopened, "data_08").unwrap());
    assert_i32_dataset_values(&reopened.dataset("data_08").unwrap(), &[8]).unwrap();
}

#[test]
fn test_group_compat_unlink_nested_dense_soft_and_external_links() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("open_rw_nested_dense_unlink_non_hard.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        let mut parent = wf.create_group("parent").unwrap();
        for idx in 0..9 {
            parent
                .new_dataset_builder(&format!("data_{idx:02}"))
                .write::<i32>(&[idx])
                .unwrap();
        }
        parent.link_soft("soft_alias", "/parent/data_00").unwrap();
        parent
            .link_external("external_alias", "missing.h5", "/remote")
            .unwrap();
        wf.flush().unwrap();
    }

    let file = File::open_rw(&path).unwrap();
    let parent = file.group("parent").unwrap();
    parent.unlink("soft_alias").unwrap();
    parent.unlink("external_alias").unwrap();

    assert!(!parent.link_exists("soft_alias").unwrap());
    assert!(!parent.link_exists("external_alias").unwrap());
    assert!(parent.link_exists("data_08").unwrap());
    assert_i32_dataset_values(&file.dataset("parent/data_08").unwrap(), &[8]).unwrap();

    let reopened = File::open(&path).unwrap();
    let parent = reopened.group("parent").unwrap();
    assert!(!parent.link_exists("soft_alias").unwrap());
    assert!(!parent.link_exists("external_alias").unwrap());
    assert!(parent.link_exists("data_08").unwrap());
    assert_i32_dataset_values(&reopened.dataset("parent/data_08").unwrap(), &[8]).unwrap();
}

#[test]
fn test_group_compat_unlink_dense_root_hard_link_is_unsupported() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir
        .path()
        .join("open_rw_root_dense_unlink_hard_rejected.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        for idx in 0..9 {
            wf.new_dataset_builder(&format!("data_{idx:02}"))
                .write::<i32>(&[idx])
                .unwrap();
        }
        wf.link_soft("soft_alias", "/data_00").unwrap();
        wf.flush().unwrap();
    }

    let file = File::open_rw(&path).unwrap();
    let root = file.root_group().unwrap();
    let err = root.unlink("data_08").unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)));
    assert!(root.link_exists("data_08").unwrap());
    assert!(root.link_exists("soft_alias").unwrap());
    assert_i32_dataset_values(&file.dataset("data_08").unwrap(), &[8]).unwrap();

    let reopened = File::open(&path).unwrap();
    assert!(file_has_member(&reopened, "data_08").unwrap());
    let reopened_root = reopened.root_group().unwrap();
    assert!(reopened_root.link_exists("soft_alias").unwrap());
    assert_i32_dataset_values(&reopened.dataset("data_08").unwrap(), &[8]).unwrap();
}

#[test]
fn test_group_compat_unlink_dense_root_hard_alias_keeps_target() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("open_rw_root_dense_unlink_hard_alias.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        for idx in 0..9 {
            wf.new_dataset_builder(&format!("data_{idx:02}"))
                .write::<i32>(&[idx])
                .unwrap();
        }
        wf.link_hard("hard_alias", "/data_01").unwrap();
        wf.flush().unwrap();
    }

    let file = File::open_rw(&path).unwrap();
    let root = file.root_group().unwrap();
    root.unlink("hard_alias").unwrap();

    assert!(root.link_exists("data_01").unwrap());
    assert!(!root.link_exists("hard_alias").unwrap());
    assert_i32_dataset_values(&file.dataset("data_01").unwrap(), &[1]).unwrap();

    let reopened = File::open(&path).unwrap();
    assert!(file_has_member(&reopened, "data_01").unwrap());
    assert!(!file_has_member(&reopened, "hard_alias").unwrap());
    assert_i32_dataset_values(&reopened.dataset("data_01").unwrap(), &[1]).unwrap();
}

#[test]
fn test_dense_links_split_name_btree_across_leaves() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dense_multileaf_links.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        for idx in 0..70 {
            wf.create_group(&format!("group_{idx:02}")).unwrap();
        }
        wf.flush().unwrap();
    }

    let bytes = std::fs::read(&path).unwrap();
    assert!(bytes.windows(4).any(|window| window == b"BTIN"));

    let f = File::open(&path).unwrap();
    let root = f.root_group().unwrap();
    assert!(group_has_member(&root, "group_00").unwrap());
    assert!(group_has_member(&root, "group_45").unwrap());
    assert!(group_has_member(&root, "group_69").unwrap());
    assert_eq!(root.open_group("group_69").unwrap().name(), "/group_69");
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
        assert!(file_has_member(&f, "remote").unwrap());

        let root = f.root_group().unwrap();
        assert!(group_has_link(&root, "remote", LinkType::External).unwrap());

        let lt = root.link_type("remote").unwrap();
        assert_eq!(lt, LinkType::External);

        root.external_link_target_with("remote", |filename, obj_path| {
            assert_eq!(filename, "other_file.h5");
            assert_eq!(obj_path, "/some/dataset");
            Ok(())
        })
        .unwrap();
    }
}

#[test]
fn test_group_compat_create_root_soft_link_open_rw() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("open_rw_root_soft_link.h5");
    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("real_data")
            .write::<i32>(&[3, 2, 1])
            .unwrap();
        wf.flush().unwrap();
    }

    let file = File::open_rw(&path).unwrap();
    let root = file.root_group().unwrap();
    root.link_soft("/real_data", "soft_alias").unwrap();

    let reopened = File::open(&path).unwrap();
    assert_i32_dataset_values(&reopened.dataset("soft_alias").unwrap(), &[3, 2, 1]).unwrap();
    let root = reopened.root_group().unwrap();
    assert!(group_has_link(&root, "soft_alias", LinkType::Soft).unwrap());
    root.soft_link_target_with("soft_alias", |target| {
        assert_eq!(target, "/real_data");
        Ok(())
    })
    .unwrap();
}

#[test]
fn test_group_compat_create_root_external_link_open_rw() {
    let dir = tempfile::tempdir().unwrap();
    let source_path = dir.path().join("open_rw_root_external_link.h5");
    let target_path = dir.path().join("target.h5");
    {
        let mut target = WritableFile::create(&target_path).unwrap();
        target
            .new_dataset_builder("data")
            .write::<i32>(&[8, 9])
            .unwrap();
        target.flush().unwrap();

        let mut source = WritableFile::create(&source_path).unwrap();
        source.create_group("existing").unwrap();
        source.flush().unwrap();
    }

    let file = File::open_rw(&source_path).unwrap();
    let root = file.root_group().unwrap();
    root.link_external("target.h5", "/data", "external_alias")
        .unwrap();

    let reopened = File::open(&source_path).unwrap();
    assert_i32_dataset_values(&reopened.dataset("external_alias").unwrap(), &[8, 9]).unwrap();
    let root = reopened.root_group().unwrap();
    assert!(group_has_link(&root, "existing", LinkType::Hard).unwrap());
    assert!(group_has_link(&root, "external_alias", LinkType::External).unwrap());
    root.external_link_target_with("external_alias", |filename, object_path| {
        assert_eq!(filename, "target.h5");
        assert_eq!(object_path, "/data");
        Ok(())
    })
    .unwrap();
}

#[test]
fn test_group_compat_create_root_hard_link_open_rw() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("open_rw_root_hard_link.h5");
    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("real_data")
            .write::<i32>(&[4, 5, 6])
            .unwrap();
        wf.flush().unwrap();
    }

    let file = File::open_rw(&path).unwrap();
    let root = file.root_group().unwrap();
    root.link_hard("/real_data", "hard_alias").unwrap();

    assert_i32_dataset_values(&file.dataset("hard_alias").unwrap(), &[4, 5, 6]).unwrap();
    assert!(group_has_link(&root, "hard_alias", LinkType::Hard).unwrap());

    let reopened = File::open(&path).unwrap();
    assert_i32_dataset_values(&reopened.dataset("hard_alias").unwrap(), &[4, 5, 6]).unwrap();
    let root = reopened.root_group().unwrap();
    assert!(group_has_link(&root, "real_data", LinkType::Hard).unwrap());
    assert!(group_has_link(&root, "hard_alias", LinkType::Hard).unwrap());
}

#[test]
fn test_group_compat_relink_same_size_refreshes_open_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("open_rw_root_relink_same_size.h5");
    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("old")
            .write::<i32>(&[12, 13])
            .unwrap();
        wf.flush().unwrap();
    }

    let file = File::open_rw(&path).unwrap();
    let root = file.root_group().unwrap();
    root.relink("old", "new").unwrap();

    assert!(!root.link_exists("old").unwrap());
    assert!(root.link_exists("new").unwrap());
    assert_i32_dataset_values(&file.dataset("new").unwrap(), &[12, 13]).unwrap();

    let reopened = File::open(&path).unwrap();
    assert!(!file_has_member(&reopened, "old").unwrap());
    assert_i32_dataset_values(&reopened.dataset("new").unwrap(), &[12, 13]).unwrap();
}

#[test]
fn test_group_compat_relink_changed_size_rebuilds_root_header() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("open_rw_root_relink_changed_size.h5");
    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("old")
            .write::<i32>(&[21, 34])
            .unwrap();
        wf.flush().unwrap();
    }

    let file = File::open_rw(&path).unwrap();
    let root = file.root_group().unwrap();
    root.relink("old", "longer_dataset_name").unwrap();

    assert!(!root.link_exists("old").unwrap());
    assert!(root.link_exists("longer_dataset_name").unwrap());
    assert_i32_dataset_values(&file.dataset("longer_dataset_name").unwrap(), &[21, 34]).unwrap();

    let reopened = File::open(&path).unwrap();
    assert!(!file_has_member(&reopened, "old").unwrap());
    assert_i32_dataset_values(&reopened.dataset("longer_dataset_name").unwrap(), &[21, 34])
        .unwrap();
}

#[test]
fn test_group_compat_relink_changed_size_rebuilds_nested_parent_chain() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("open_rw_nested_relink_changed_size.h5");
    {
        let mut wf = WritableFile::create(&path).unwrap();
        let mut parent = wf.create_group("parent").unwrap();
        let mut child = parent.create_group("child").unwrap();
        child
            .new_dataset_builder("old")
            .write::<i32>(&[55, 89])
            .unwrap();
        wf.flush().unwrap();
    }

    let file = File::open_rw(&path).unwrap();
    let child = file.group("parent/child").unwrap();
    child.relink("old", "longer_dataset_name").unwrap();

    let child = file.group("parent/child").unwrap();
    assert!(!child.link_exists("old").unwrap());
    assert!(child.link_exists("longer_dataset_name").unwrap());
    assert_i32_dataset_values(
        &file.dataset("parent/child/longer_dataset_name").unwrap(),
        &[55, 89],
    )
    .unwrap();

    let reopened = File::open(&path).unwrap();
    let child = reopened.group("parent/child").unwrap();
    assert!(!child.link_exists("old").unwrap());
    assert!(child.link_exists("longer_dataset_name").unwrap());
    assert_i32_dataset_values(
        &reopened
            .dataset("parent/child/longer_dataset_name")
            .unwrap(),
        &[55, 89],
    )
    .unwrap();
}

#[test]
fn test_group_compat_relink_same_size_dense_root_link() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("open_rw_root_dense_relink_same_size.h5");
    {
        let mut wf = WritableFile::create(&path).unwrap();
        for idx in 0..9 {
            wf.new_dataset_builder(&format!("data_{idx:02}"))
                .write::<i32>(&[idx])
                .unwrap();
        }
        wf.flush().unwrap();
    }

    let file = File::open_rw(&path).unwrap();
    let root = file.root_group().unwrap();
    root.relink("data_08", "item_08").unwrap();

    assert!(!root.link_exists("data_08").unwrap());
    assert!(root.link_exists("item_08").unwrap());
    assert_i32_dataset_values(&file.dataset("item_08").unwrap(), &[8]).unwrap();

    let reopened = File::open(&path).unwrap();
    assert!(!file_has_member(&reopened, "data_08").unwrap());
    assert!(file_has_member(&reopened, "item_08").unwrap());
    assert_i32_dataset_values(&reopened.dataset("item_08").unwrap(), &[8]).unwrap();
}

#[test]
fn test_group_compat_relink_changed_size_dense_root_link_shrinks() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir
        .path()
        .join("open_rw_root_dense_relink_changed_size_shrink.h5");
    {
        let mut wf = WritableFile::create(&path).unwrap();
        for idx in 0..9 {
            wf.new_dataset_builder(&format!("data_{idx:02}"))
                .write::<i32>(&[idx])
                .unwrap();
        }
        wf.flush().unwrap();
    }

    let file = File::open_rw(&path).unwrap();
    let root = file.root_group().unwrap();
    root.relink("data_08", "d8").unwrap();

    assert!(!root.link_exists("data_08").unwrap());
    assert!(root.link_exists("d8").unwrap());
    assert_i32_dataset_values(&file.dataset("d8").unwrap(), &[8]).unwrap();

    let reopened = File::open(&path).unwrap();
    assert!(!file_has_member(&reopened, "data_08").unwrap());
    assert!(file_has_member(&reopened, "d8").unwrap());
    assert_i32_dataset_values(&reopened.dataset("d8").unwrap(), &[8]).unwrap();
}

#[test]
fn test_group_compat_relink_changed_size_dense_root_link_grows() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir
        .path()
        .join("open_rw_root_dense_relink_changed_size_growth.h5");
    {
        let mut wf = WritableFile::create(&path).unwrap();
        for idx in 0..9 {
            wf.new_dataset_builder(&format!("data_{idx:02}"))
                .write::<i32>(&[idx])
                .unwrap();
        }
        wf.flush().unwrap();
    }

    let file = File::open_rw(&path).unwrap();
    let root = file.root_group().unwrap();
    root.relink("data_08", "longer_data_name_08").unwrap();

    assert!(!root.link_exists("data_08").unwrap());
    assert!(root.link_exists("longer_data_name_08").unwrap());
    assert_i32_dataset_values(&file.dataset("longer_data_name_08").unwrap(), &[8]).unwrap();

    let reopened = File::open(&path).unwrap();
    assert!(!file_has_member(&reopened, "data_08").unwrap());
    assert!(file_has_member(&reopened, "longer_data_name_08").unwrap());
    assert_i32_dataset_values(&reopened.dataset("longer_data_name_08").unwrap(), &[8]).unwrap();
}

#[test]
fn test_group_compat_relink_same_size_indirect_dense_root_link_heap() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("open_rw_root_indirect_dense_relink.h5");
    let suffix = "x".repeat(2_000);
    let old_name = format!("link_008_{suffix}");
    let new_name = format!("renm_008_{suffix}");
    let out = std::process::Command::new("python3")
        .arg("-c")
        .arg(
            r#"import sys
try:
    import h5py
except ModuleNotFoundError as exc:
    raise exc
suffix = "x" * 2000
with h5py.File(sys.argv[1], "w", libver="latest") as f:
    for i in range(40):
        f.create_group(f"link_{i:03d}_{suffix}")
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
    let root = file.root_group().unwrap();
    root.relink(&old_name, &new_name).unwrap();
    assert!(!root.link_exists(&old_name).unwrap());
    assert!(root.link_exists(&new_name).unwrap());

    let reopened = File::open(&path).unwrap();
    assert!(!file_has_member(&reopened, &old_name).unwrap());
    assert!(file_has_member(&reopened, &new_name).unwrap());
}

#[test]
fn test_group_compat_create_links_under_direct_nested_existing_file_group() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("open_rw_nested_links.h5");
    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("real_data")
            .write::<i32>(&[31, 32])
            .unwrap();
        wf.create_group("parent").unwrap();
        wf.flush().unwrap();
    }

    let file = File::open_rw(&path).unwrap();
    let parent = file.group("parent").unwrap();
    parent.link_soft("/missing", "soft_alias").unwrap();
    parent.link_hard("/real_data", "hard_alias").unwrap();
    parent
        .link_external("target.h5", "/data", "external_alias")
        .unwrap();

    let parent = file.group("parent").unwrap();
    assert!(parent.link_exists("soft_alias").unwrap());
    assert!(parent.link_exists("hard_alias").unwrap());
    assert!(parent.link_exists("external_alias").unwrap());
    assert_i32_dataset_values(&file.dataset("parent/hard_alias").unwrap(), &[31, 32]).unwrap();

    let reopened = File::open(&path).unwrap();
    let parent = reopened.group("parent").unwrap();
    assert!(group_has_link(&parent, "soft_alias", LinkType::Soft).unwrap());
    assert!(group_has_link(&parent, "hard_alias", LinkType::Hard).unwrap());
    assert!(group_has_link(&parent, "external_alias", LinkType::External).unwrap());
    assert_i32_dataset_values(&reopened.dataset("parent/hard_alias").unwrap(), &[31, 32]).unwrap();
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
    assert_i32_dataset_values(&f.dataset("same_dir").unwrap(), &[1, 2, 3]).unwrap();
    assert_i32_dataset_values(&f.dataset("relative").unwrap(), &[4, 5, 6]).unwrap();
    assert_i32_dataset_values(&f.dataset("absolute").unwrap(), &[1, 2, 3]).unwrap();
    assert_eq!(f.group("remote_group").unwrap().name(), "/group");
    assert!(matches!(f.dataset("missing"), Err(Error::Io(_))));
}

#[test]
fn test_utf8_link_names_and_non_ascii_external_filename() {
    let f = File::open("tests/data/hdf5_ref/link_edge_cases.h5").unwrap();
    let root = f.root_group().unwrap();

    assert!(group_has_member(&root, "猫_group").unwrap());
    assert!(group_has_member(&root, "å_link").unwrap());
    assert!(group_has_member(&root, "external_å").unwrap());
    assert_eq!(
        root.member_type("å_link").unwrap(),
        hdf5_pure_rust::hl::file::ObjectType::Dataset
    );

    root.external_link_target_with("external_å", |filename, object_path| {
        assert_eq!(filename, "målfil.h5");
        assert_eq!(object_path, "/dåta");
        Ok(())
    })
    .unwrap();
}

#[test]
fn test_link_decoder_rejects_invalid_character_encoding() {
    let mut raw = vec![1, 0x10, 2, 1, b'x'];
    raw.extend_from_slice(&0u64.to_le_bytes());
    let err = LinkMessage::decode(&raw, 8).expect_err("invalid link cset should fail");
    assert!(matches!(err, Error::InvalidFormat(_)));
}
