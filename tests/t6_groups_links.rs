//! Phase T6: Group and link tests.

use hdf5_pure_rust::format::messages::link::LinkType;
use hdf5_pure_rust::hl::link::LinkValueRef;
use hdf5_pure_rust::{Dataset, File, Group, Result};

const FILE: &str = "tests/data/hdf5_ref/groups_and_links.h5";
fn open() -> File {
    File::open(FILE).unwrap()
}

fn file_member_names(file: &File) -> Result<Vec<String>> {
    let mut names = Vec::new();
    file.visit_member_names(|name| {
        names.push(name.to_string());
        Ok(())
    })?;
    Ok(names)
}

fn group_member_names(group: &Group) -> Result<Vec<String>> {
    let mut names = Vec::new();
    group.visit_member_names(|name| {
        names.push(name.to_string());
        Ok(())
    })?;
    Ok(names)
}

fn read_vec<T: hdf5_pure_rust::H5Type + Default + Clone>(ds: &Dataset) -> Result<Vec<T>> {
    let mut values = vec![T::default(); ds.size()? as usize];
    ds.read_into(&mut values)?;
    Ok(values)
}

// T6a: Nested groups

#[test]
fn t6a_navigate_nested() {
    let f = open();
    let g = f.group("a").unwrap();
    let names = group_member_names(&g).unwrap();
    assert!(names.contains(&"b".to_string()));
    assert!(names.contains(&"e".to_string()));

    let gb = f.group("a/b").unwrap();
    let names = group_member_names(&gb).unwrap();
    assert!(names.contains(&"c".to_string()));
    assert!(names.contains(&"d".to_string()));
}

#[test]
fn t6a_deep_dataset() {
    let ds = open().dataset("a/b/c/data").unwrap();
    let vals: Vec<i32> = read_vec(&ds).unwrap();
    assert_eq!(vals, vec![1, 2, 3]);
}

// T6b: Links

#[test]
fn t6b_hard_link_alias() {
    let f = open();
    // alias_data is a hard link to same object as /a/b/c/data
    let names = file_member_names(&f).unwrap();
    assert!(names.contains(&"alias_data".to_string()));
    // Should be readable as a dataset
    let ds = f.dataset("alias_data").unwrap();
    let vals: Vec<i32> = read_vec(&ds).unwrap();
    assert_eq!(vals, vec![1, 2, 3]);
}

#[test]
fn t6b_soft_link() {
    let f = open();
    let names = file_member_names(&f).unwrap();
    assert!(names.contains(&"soft_link".to_string()));

    let root = f.root_group().unwrap();
    root.soft_link_target_with("soft_link", |target| {
        assert_eq!(target, "/a/b/c/data");
        Ok(())
    })
    .unwrap();
}

#[test]
fn t6b_external_link() {
    let f = open();
    let root = f.root_group().unwrap();
    root.external_link_target_with("ext_link", |filename, obj_path| {
        assert_eq!(filename, "other_file.h5");
        assert_eq!(obj_path, "/some/path");
        Ok(())
    })
    .unwrap();
}

#[test]
fn t6b_link_exists() {
    let root = open().root_group().unwrap();
    assert!(root.link_exists("a").unwrap());
    assert!(root.link_exists("alias_data").unwrap());
    assert!(root.link_exists("soft_link").unwrap());
    assert!(root.link_exists("ext_link").unwrap());
    assert!(!root.link_exists("nonexistent").unwrap());
}

#[test]
fn t6b_link_info_name_and_value_by_index() {
    let root = open().root_group().unwrap();
    let names = group_member_names(&root).unwrap();
    let soft_idx = names.iter().position(|name| name == "soft_link").unwrap();
    let ext_idx = names.iter().position(|name| name == "ext_link").unwrap();
    let hard_idx = names.iter().position(|name| name == "alias_data").unwrap();

    let mut link_name = String::new();
    root.link_name_by_idx_into(soft_idx, &mut link_name)
        .unwrap();
    assert_eq!(link_name, "soft_link");
    let soft_info = root.link_info("soft_link").unwrap();
    assert_eq!(soft_info.link_type, LinkType::Soft);
    assert_eq!(root.link_info_by_idx(soft_idx).unwrap(), soft_info);
    root.link_value_by_idx_with(soft_idx, |value| {
        assert_eq!(value, Some(LinkValueRef::Soft("/a/b/c/data")));
        Ok(())
    })
    .unwrap();

    root.link_value_by_idx_with(ext_idx, |value| {
        assert_eq!(
            value,
            Some(LinkValueRef::External {
                filename: "other_file.h5",
                object_path: "/some/path"
            })
        );
        Ok(())
    })
    .unwrap();

    let hard_info = root.link_info_by_idx(hard_idx).unwrap();
    assert_eq!(hard_info.link_type, LinkType::Hard);
    assert!(hard_info.hard_link_addr.is_some());
    root.link_value_by_idx_with(hard_idx, |value| {
        assert_eq!(value, None);
        Ok(())
    })
    .unwrap();

    assert_eq!(root.object_comment().unwrap(), None);
    assert_eq!(root.object_comment_by_name("alias_data").unwrap(), None);
    let root_info = root.native_info().unwrap();
    assert_eq!(root_info.addr, root.addr());
    assert!(root_info.message_count > 0);
    let object_info = root.object_info_by_idx(hard_idx).unwrap();
    assert_eq!(object_info.addr, hard_info.hard_link_addr.unwrap());
    assert_eq!(root.native_info_by_idx(hard_idx).unwrap(), object_info);

    assert!(root
        .link_name_by_idx_into(names.len(), &mut link_name)
        .is_err());
    assert!(root.link_info_by_idx(names.len()).is_err());
    assert!(root
        .link_value_by_idx_with(names.len(), |value| {
            assert_eq!(value, None);
            Ok(())
        })
        .is_err());
    assert!(root.object_info_by_idx(names.len()).is_err());
}

// T6c: Many groups (may trigger dense storage)

#[test]
fn t6c_many_groups() {
    let f = open();
    let names = file_member_names(&f).unwrap();
    for i in 0..15 {
        let expected = format!("dense_{i:02}");
        assert!(names.contains(&expected), "missing {expected}");
    }
}

// T6d: Empty groups

#[test]
fn t6d_empty_group() {
    let g = open().group("a/b/d").unwrap();
    assert!(g.is_empty().unwrap());
}
