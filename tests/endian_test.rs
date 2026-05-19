use hdf5_pure_rust::File;

#[test]
fn test_read_bigendian_float() {
    let f = File::open("tests/data/bigendian.h5").unwrap();
    let ds = f.dataset("be_float").unwrap();

    let dtype = ds.dtype().unwrap();
    assert_eq!(
        dtype.byte_order(),
        Some(hdf5_pure_rust::format::messages::datatype::ByteOrder::BigEndian)
    );

    let mut vals = [0.0f64; 3];
    ds.read_into(&mut vals).unwrap();
    assert_eq!(vals, [1.0, 2.0, 3.0]);
}

#[test]
fn test_read_bigendian_int() {
    let f = File::open("tests/data/bigendian.h5").unwrap();
    let ds = f.dataset("be_int").unwrap();

    let mut vals = [0i32; 3];
    ds.read_into(&mut vals).unwrap();
    assert_eq!(vals, [10, 20, 30]);
}

#[test]
fn test_read_littleendian_unchanged() {
    let f = File::open("tests/data/bigendian.h5").unwrap();
    let ds = f.dataset("le_float").unwrap();

    let mut vals = [0.0f64; 3];
    ds.read_into(&mut vals).unwrap();
    assert_eq!(vals, [4.0, 5.0, 6.0]);
}
