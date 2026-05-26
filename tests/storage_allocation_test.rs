use hdf5_pure_rust::{Dataset, File};

fn assert_shape(ds: &Dataset, expected: &[u64]) {
    let mut dims = Vec::new();
    ds.shape_into(&mut dims).unwrap();
    assert_eq!(dims, expected);
}

#[test]
fn test_contiguous_dataset_with_undefined_storage_address_reads_fill_value() {
    let f = File::open("tests/data/hdf5_ref/undefined_storage_address.h5").unwrap();
    let ds = f.dataset("late_fill").unwrap();

    assert_shape(&ds, &[4]);
    let mut vals = [0; 4];
    ds.read_into(&mut vals).unwrap();
    assert_eq!(vals, [-5, -5, -5, -5]);
}

#[test]
fn test_late_allocation_fill_time_never_reads_zeroes() {
    let f = File::open("tests/data/hdf5_ref/late_fill_time_never.h5").unwrap();
    let ds = f.dataset("late_never").unwrap();

    assert_shape(&ds, &[4]);
    let mut vals = [0; 4];
    ds.read_into(&mut vals).unwrap();
    assert_eq!(vals, [0, 0, 0, 0]);
}

#[test]
fn test_undefined_contiguous_storage_read_slice_uses_fill_policy() {
    let f = File::open("tests/data/hdf5_ref/undefined_storage_address.h5").unwrap();
    let ds = f.dataset("late_fill").unwrap();

    let vals = ds.read_slice::<i32, _>(1..3).unwrap();
    assert_eq!(vals, [-5, -5]);

    let mut into = [0; 2];
    ds.read_slice_into::<i32, _>(1..3, &mut into).unwrap();
    assert_eq!(into, [-5, -5]);

    let f = File::open("tests/data/hdf5_ref/late_fill_time_never.h5").unwrap();
    let ds = f.dataset("late_never").unwrap();

    let vals = ds.read_slice::<i32, _>(1..3).unwrap();
    assert_eq!(vals, [0, 0]);

    let mut into = [-1; 2];
    ds.read_slice_into::<i32, _>(1..3, &mut into).unwrap();
    assert_eq!(into, [0, 0]);
}
