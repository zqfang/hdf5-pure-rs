use hdf5_pure_rust::File;

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
