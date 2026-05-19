use hdf5_pure_rust::format::messages::link::LinkType;
use hdf5_pure_rust::{File, Group, Result};

fn file_member_count(file: &File) -> Result<usize> {
    let mut count = 0;
    file.visit_member_names(|name| {
        println!("  {name}");
        count += 1;
        Ok(())
    })?;
    Ok(count)
}

fn file_has_member(file: &File, expected: &str) -> Result<bool> {
    let mut found = false;
    file.visit_member_names(|name| {
        found |= name == expected;
        Ok(())
    })?;
    Ok(found)
}

fn group_member_count(group: &Group) -> Result<usize> {
    let mut count = 0;
    group.visit_member_names(|_| {
        count += 1;
        Ok(())
    })?;
    Ok(count)
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

#[test]
fn test_read_dense_links() {
    let f = File::open("tests/data/dense_links.h5").expect("failed to open dense links file");
    println!("Dense link members:");

    assert_eq!(file_member_count(&f).expect("failed to list members"), 20);
    let root = f.root_group().unwrap();
    assert_eq!(group_member_count(&root).unwrap(), 20);
    for i in 0..20 {
        let expected = format!("group_{i:02}");
        assert!(
            file_has_member(&f, &expected).unwrap(),
            "missing {expected}"
        );
        assert!(group_has_link(&root, &expected, LinkType::Hard).unwrap());
    }
}

#[test]
fn test_read_dense_links_open_group() {
    let f = File::open("tests/data/dense_links.h5").expect("failed to open dense links file");

    let g = f.group("group_05").expect("failed to open group_05");
    assert_eq!(g.name(), "/group_05");
    assert!(g.is_empty().unwrap());
}

#[test]
fn test_read_dense_attrs_file() {
    // The dense_attrs.h5 file has 20 attributes on the root group
    // and a "data" dataset child
    let f = File::open("tests/data/dense_attrs.h5").expect("failed to open dense attrs file");

    assert!(file_has_member(&f, "data").expect("failed to list members"));
}

#[test]
fn test_dense_group_multiple_v2_btree_levels_name_index() {
    let f = File::open("tests/data/hdf5_ref/dense_group_cases.h5").unwrap();
    let group = f.group("name_index_deep").unwrap();

    assert_eq!(group_member_count(&group).unwrap(), 4096);
    for idx in [0, 1, 1023, 2048, 4095] {
        let name = format!("link_{idx:04}");
        assert!(group_has_member(&group, &name).unwrap(), "missing {name}");
        assert_eq!(
            group.member_type(&name).unwrap(),
            hdf5_pure_rust::hl::file::ObjectType::Dataset
        );
    }
}

#[test]
fn test_dense_group_creation_order_indexing_enabled_and_disabled() {
    let f = File::open("tests/data/hdf5_ref/dense_group_cases.h5").unwrap();

    let tracked = f.group("creation_order_tracked").unwrap();
    assert_eq!(group_member_count(&tracked).unwrap(), 64);
    assert!(group_has_member(&tracked, "tracked_00").unwrap());
    assert!(group_has_member(&tracked, "tracked_63").unwrap());
    let mut tracked_creation_orders = Vec::new();
    let mut has_tracked_00 = false;
    let mut has_tracked_63 = false;
    tracked
        .visit_links_by_creation_order(|link| {
            tracked_creation_orders.push(link.creation_order.unwrap());
            has_tracked_00 |= link.name == "tracked_00";
            has_tracked_63 |= link.name == "tracked_63";
            Ok(())
        })
        .unwrap();
    assert_eq!(tracked_creation_orders, (0..64).collect::<Vec<_>>());
    assert!(has_tracked_00);
    assert!(has_tracked_63);
    assert_eq!(
        tracked.member_type("tracked_42").unwrap(),
        hdf5_pure_rust::hl::file::ObjectType::Dataset
    );

    let untracked = f.group("creation_order_untracked").unwrap();
    assert_eq!(group_member_count(&untracked).unwrap(), 64);
    assert!(group_has_member(&untracked, "untracked_00").unwrap());
    assert!(group_has_member(&untracked, "untracked_63").unwrap());
    assert!(untracked.visit_links_by_creation_order(|_| Ok(())).is_err());
    assert_eq!(
        untracked.member_type("untracked_42").unwrap(),
        hdf5_pure_rust::hl::file::ObjectType::Dataset
    );
}
