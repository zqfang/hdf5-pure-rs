use hdf5_pure_rust::{DatasetAccess, File};

#[test]
fn test_external_raw_data_storage_reads_relative_file() {
    let f = File::open("tests/data/hdf5_ref/external_raw_storage.h5").unwrap();
    let ds = f.dataset("external_raw").unwrap();
    let plist = ds.create_plist().unwrap();

    assert_eq!(plist.external_count(), 1);
    let external = plist.external(0).unwrap();
    assert_eq!(external.name, "external_raw_storage.bin");
    assert_eq!(external.file_offset, 0);
    assert_eq!(external.size, u64::MAX);
    assert!(plist.external(1).is_none());

    let mut vals = [0; 4];
    ds.read_into(&mut vals).unwrap();
    assert_eq!(vals, [1, 2, 3, 4]);
}

#[test]
fn test_external_raw_data_storage_reads_multiple_files() {
    let f = File::open("tests/data/hdf5_ref/external_raw_multi.h5").unwrap();
    let ds = f.dataset("external_multi").unwrap();
    let plist = ds.create_plist().unwrap();

    assert_eq!(plist.external_count(), 2);
    assert_eq!(plist.external(0).unwrap().name, "external_raw_multi_a.bin");
    assert_eq!(plist.external(1).unwrap().name, "external_raw_multi_b.bin");

    let mut vals = [0; 8];
    ds.read_into(&mut vals).unwrap();
    assert_eq!(vals, [10, 11, 12, 13, 14, 15, 16, 17]);
}

#[test]
fn test_external_raw_data_storage_honors_efile_prefix() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("external_raw_storage.h5");
    std::fs::copy("tests/data/hdf5_ref/external_raw_storage.h5", &path).unwrap();

    let f = File::open(&path).unwrap();
    let ds = f.dataset("external_raw").unwrap();

    let mut vals = [0; 4];
    let err = ds
        .read_into(&mut vals)
        .expect_err("external raw file should not be adjacent to copied HDF5 file");
    assert!(
        err.to_string().contains("No such file or directory"),
        "unexpected error: {err}"
    );

    let mut access = DatasetAccess::new();
    access.set_efile_prefix(Some("tests/data/hdf5_ref"));
    ds.read_into_with_access(&access, &mut vals).unwrap();
    assert_eq!(vals, [1, 2, 3, 4]);
}

#[test]
fn test_external_raw_data_storage_preserves_output_when_later_file_is_missing() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("external_raw_multi.h5");
    std::fs::copy("tests/data/hdf5_ref/external_raw_multi.h5", &path).unwrap();
    std::fs::copy(
        "tests/data/hdf5_ref/external_raw_multi_a.bin",
        dir.path().join("external_raw_multi_a.bin"),
    )
    .unwrap();

    let f = File::open(&path).unwrap();
    let ds = f.dataset("external_multi").unwrap();
    let mut vals = [99u8; 8];
    let err = ds
        .read_into(&mut vals)
        .expect_err("second external raw file should be missing");
    assert!(
        err.to_string().contains("No such file or directory"),
        "unexpected error: {err}"
    );
    assert_eq!(vals, [99u8; 8]);
}
