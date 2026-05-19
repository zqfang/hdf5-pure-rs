use hdf5_pure_rust::{Dataset, File, H5Type};

fn assert_shape(ds: &Dataset, expected: &[u64]) {
    let space = ds.space().unwrap();
    assert_eq!(space.shape(), expected);
}

fn read_dataset_into<T: H5Type + Default>(ds: &Dataset, values: &mut [T]) {
    ds.read_into(values).unwrap();
}

fn assert_dataset_values<T>(ds: &Dataset, expected: &[T])
where
    T: H5Type + Default + PartialEq + std::fmt::Debug,
{
    let mut values = (0..ds.size().unwrap())
        .map(|_| T::default())
        .collect::<Vec<_>>();
    read_dataset_into(ds, &mut values);
    assert_eq!(values, expected);
}

#[test]
fn test_read_float64_1d_v0() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("float64_1d").unwrap();

    assert_shape(&ds, &[5]);
    assert_eq!(ds.element_size().unwrap(), 8);

    assert_dataset_values::<f64>(&ds, &[1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_read_int32_1d_v0() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("int32_1d").unwrap();

    assert_shape(&ds, &[3]);
    assert_eq!(ds.element_size().unwrap(), 4);

    assert_dataset_values::<i32>(&ds, &[10, 20, 30]);
}

#[test]
fn test_read_scalar_v0() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("scalar").unwrap();

    assert_shape(&ds, &[]);
    assert_eq!(ds.size().unwrap(), 1);

    let mut value = 0.0;
    ds.read_scalar_into(&mut value).unwrap();
    assert_eq!(value, 42.0);
}

#[test]
fn test_read_int8_2d_v0() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("int8_2d").unwrap();

    assert_shape(&ds, &[2, 3]);
    assert_eq!(ds.element_size().unwrap(), 1);

    assert_dataset_values::<i8>(&ds, &[1, 2, 3, 4, 5, 6]);
}

#[test]
fn test_read_chunked_compressed_v0() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("chunked").unwrap();

    assert_shape(&ds, &[100]);
    assert_eq!(ds.element_size().unwrap(), 4);

    let mut values = vec![0.0f32; ds.size().unwrap() as usize];
    read_dataset_into(&ds, &mut values);
    assert_eq!(values.len(), 100);

    for (i, val) in values.iter().enumerate() {
        assert_eq!(*val, i as f32, "mismatch at index {i}");
    }
}

#[test]
fn test_read_float64_1d_v3() {
    let f = File::open("tests/data/datasets_v3.h5").unwrap();
    let ds = f.dataset("float64_1d").unwrap();

    assert_dataset_values::<f64>(&ds, &[1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_read_int32_1d_v3() {
    let f = File::open("tests/data/datasets_v3.h5").unwrap();
    let ds = f.dataset("int32_1d").unwrap();

    assert_dataset_values::<i32>(&ds, &[10, 20, 30]);
}

#[test]
fn test_read_scalar_v3() {
    let f = File::open("tests/data/datasets_v3.h5").unwrap();
    let ds = f.dataset("scalar").unwrap();

    let mut value = 0.0;
    ds.read_scalar_into(&mut value).unwrap();
    assert_eq!(value, 42.0);
}
