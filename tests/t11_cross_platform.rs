//! Phase T11: Cross-platform compatibility tests.

use hdf5_pure_rust::File;

const REF_DIR: &str = "tests/data/hdf5_ref";

fn member_count(file: &File) -> hdf5_pure_rust::Result<usize> {
    let mut count = 0;
    file.visit_member_names(|_| {
        count += 1;
        Ok(())
    })?;
    Ok(count)
}

// T11a: Big-endian files on little-endian host

#[test]
fn t11a_be_data_read() {
    let f = File::open(&format!("{REF_DIR}/be_data.h5")).unwrap();
    // Verify we can navigate the file
    assert_ne!(member_count(&f).unwrap(), 0);
}

#[test]
fn t11a_be_extlinks() {
    // Big-endian external link files
    let f1 = File::open(&format!("{REF_DIR}/be_extlink1.h5")).unwrap();
    assert_ne!(member_count(&f1).unwrap(), 0);

    let f2 = File::open(&format!("{REF_DIR}/be_extlink2.h5")).unwrap();
    assert_ne!(member_count(&f2).unwrap(), 0);
}

#[test]
fn t11a_be_filters() {
    let f = File::open(&format!("{REF_DIR}/test_filters_be.h5")).unwrap();
    assert_ne!(member_count(&f).unwrap(), 0);
}

#[test]
fn t11a_typed_read_be() {
    // Read big-endian data with byte-swap
    let f = File::open("tests/data/bigendian.h5").unwrap();
    let ds = f.dataset("be_float").unwrap();
    let mut vals = vec![0.0; ds.size().unwrap() as usize];
    ds.read_into(&mut vals).unwrap();
    assert_eq!(vals, vec![1.0, 2.0, 3.0]);
}

// T11b: Old format files

#[test]
fn t11b_old_group_format() {
    let f = File::open(&format!("{REF_DIR}/group_old.h5")).unwrap();
    assert_ne!(member_count(&f).unwrap(), 0);
}

#[test]
fn t11b_old_fill_values() {
    let f = File::open(&format!("{REF_DIR}/fill_old.h5")).unwrap();
    let mut names = Vec::new();
    f.member_names_into(&mut names).unwrap();
    assert_eq!(names, vec!["dset1".to_string(), "dset2".to_string()]);

    let ds = f.dataset("dset2").unwrap();
    let mut vals = vec![0; ds.size().unwrap() as usize];
    ds.read_into(&mut vals).unwrap();
    assert_eq!(vals, vec![4444; 8 * 8]);
}

#[test]
fn t11b_old_fill_value_create_plist() {
    let f = File::open(&format!("{REF_DIR}/fill_old.h5")).unwrap();
    let plist = f.dataset("dset2").unwrap().create_plist().unwrap();

    assert!(plist.fill_value_defined);
    assert_eq!(plist.fill_value, Some(4444i32.to_be_bytes().to_vec()));
}

#[test]
fn t11b_chunked_fill18_values() {
    let f = File::open(&format!("{REF_DIR}/fill18.h5")).unwrap();
    let ds = f.dataset("DS1").unwrap();
    let mut vals = vec![0; ds.size().unwrap() as usize];
    ds.read_into(&mut vals).unwrap();
    let expected = vec![
        0, -1, -2, -3, -4, -5, -6, 99, 99, 99, 0, 0, 0, 0, 0, 0, 0, 99, 99, 99, 0, 1, 2, 3, 4, 5,
        6, 99, 99, 99, 0, 2, 4, 6, 8, 10, 12, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    ];
    assert_eq!(vals, expected);
}

#[test]
fn t11b_old_layout() {
    let f = File::open(&format!("{REF_DIR}/tlayouto.h5")).unwrap();
    assert_ne!(member_count(&f).unwrap(), 0);
}

#[test]
fn t11b_old_mtime() {
    for name in ["tmtimen.h5", "tmtimeo.h5"] {
        let f = File::open(&format!("{REF_DIR}/{name}")).unwrap();
        assert_ne!(member_count(&f).unwrap(), 0, "{name} has no members");
    }
}

#[test]
fn t11b_old_array() {
    let f = File::open(&format!("{REF_DIR}/tarrold.h5")).unwrap();
    assert_ne!(member_count(&f).unwrap(), 0);
}

// T11c: Various file space strategies

#[test]
fn t11c_filespace_strategies() {
    for name in [
        "filespace_1_6.h5",
        "filespace_1_8.h5",
        "paged_nopersist.h5",
        "paged_persist.h5",
        "fsm_aggr_nopersist.h5",
        "fsm_aggr_persist.h5",
        "aggr.h5",
    ] {
        let f = File::open(&format!("{REF_DIR}/{name}")).unwrap();
        let sb = f.superblock();
        println!(
            "{name}: sb_version={}, sizeof_addr={}, sizeof_size={}",
            sb.version, sb.sizeof_addr, sb.sizeof_size
        );
    }
}

// T11d: Deflate filter files from C test suite

#[test]
fn t11d_deflate_reference() {
    let f = File::open(&format!("{REF_DIR}/deflate.h5")).unwrap();
    let mut names = Vec::new();
    f.member_names_into(&mut names).unwrap();
    let root = f.root_group().unwrap();
    let mut datasets = 0;
    // Try reading a dataset if available
    for name in &names {
        if let Ok(ds) = root.open_dataset(name) {
            let mut dims = Vec::new();
            ds.shape_into(&mut dims).unwrap();
            println!("  {name}: shape={dims:?}");
            datasets += 1;
        }
    }
    assert_ne!(datasets, 0);
}

// T11e: Charsets

#[test]
fn t11e_charsets() {
    let f = File::open(&format!("{REF_DIR}/charsets.h5")).unwrap();
    assert_ne!(member_count(&f).unwrap(), 0);
}
