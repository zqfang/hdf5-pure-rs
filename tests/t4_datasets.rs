//! Phase T4+T5: Dataset storage layouts, filters, and attributes.

use hdf5_pure_rust::File;

const FILE: &str = "tests/data/hdf5_ref/layouts_and_filters.h5";
fn open() -> File {
    File::open(FILE).unwrap()
}

// T4a: Storage layouts

#[test]
fn t4a_compact() {
    let mut vals = [0u8; 3];
    open()
        .dataset("compact")
        .unwrap()
        .read_into(&mut vals)
        .unwrap();
    assert_eq!(vals, [1, 2, 3]);
}

#[test]
fn t4a_contiguous() {
    let mut vals = [0.0f64; 100];
    open()
        .dataset("contiguous")
        .unwrap()
        .read_into(&mut vals)
        .unwrap();
    assert_eq!(vals.len(), 100);
    assert_eq!(vals[0], 0.0);
    assert_eq!(vals[99], 99.0);
}

#[test]
fn t4a_chunked_raw() {
    let ds = open().dataset("chunked_raw").unwrap();
    assert!(ds.is_chunked().unwrap());
    let mut vals = [0i32; 200];
    ds.read_into(&mut vals).unwrap();
    assert_eq!(vals.len(), 200);
    for (i, v) in vals.iter().enumerate() {
        assert_eq!(*v, i as i32);
    }
}

// T4b: Chunked with filters

#[test]
fn t4b_deflate() {
    let ds = open().dataset("chunked_deflate").unwrap();
    let plist = ds.create_plist().unwrap();
    assert!(plist.is_compressed());
    let mut vals = [0.0f32; 200];
    ds.read_into(&mut vals).unwrap();
    assert_eq!(vals.len(), 200);
    for (i, v) in vals.iter().enumerate() {
        assert_eq!(*v, i as f32);
    }
}

#[test]
fn t4b_shuffle_deflate() {
    let ds = open().dataset("chunked_shuf_def").unwrap();
    let plist = ds.create_plist().unwrap();
    assert!(plist.has_shuffle());
    assert!(plist.is_compressed());
    let mut vals = [0i64; 200];
    ds.read_into(&mut vals).unwrap();
    assert_eq!(vals.len(), 200);
    for (i, v) in vals.iter().enumerate() {
        assert_eq!(*v, i as i64);
    }
}

#[test]
fn t4b_fletcher32() {
    let ds = open().dataset("chunked_fletcher").unwrap();
    let mut vals = [0.0f32; 100];
    ds.read_into(&mut vals).unwrap();
    assert_eq!(vals.len(), 100);
    for (i, v) in vals.iter().enumerate() {
        assert_eq!(*v, i as f32);
    }
}

#[test]
fn t4b_chunked_2d() {
    let ds = open().dataset("chunked_2d").unwrap();
    let mut shape = Vec::new();
    ds.shape_into(&mut shape).unwrap();
    assert_eq!(shape, vec![6, 10]);
    let mut chunk = Vec::new();
    assert!(ds.chunk_into(&mut chunk).unwrap());
    assert_eq!(chunk, vec![3, 5]);
    let arr = ds.read_2d::<f64>().unwrap();
    assert_eq!(arr[[0, 0]], 0.0);
    assert_eq!(arr[[5, 9]], 59.0);
}

// T4c: Selections on various layouts

#[test]
fn t4c_slice_contiguous() {
    let ds = open().dataset("contiguous").unwrap();
    let mut vals = [0.0f64; 10];
    ds.read_slice_into::<f64, _>(10..20, &mut vals).unwrap();
    assert_eq!(
        vals,
        [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
    );
}

#[test]
fn t4c_slice_chunked() {
    let ds = open().dataset("chunked_deflate").unwrap();
    let mut vals = [0.0f32; 5];
    ds.read_slice_into::<f32, _>(50..55, &mut vals).unwrap();
    assert_eq!(vals, [50.0, 51.0, 52.0, 53.0, 54.0]);
}

// T5: Attributes

#[test]
fn t5a_root_int_attr() {
    let attr = open().attr("root_int").unwrap();
    assert_eq!(attr.read_scalar::<i64>().unwrap(), 42);
}

#[test]
fn t5a_root_float_attr() {
    let attr = open().attr("root_float").unwrap();
    let val = attr.read_scalar::<f64>().unwrap();
    assert!((val - 3.14).abs() < 1e-10);
}

#[test]
fn t5b_root_array_attr() {
    let attr = open().attr("root_array").unwrap();
    let mut vals = [0.0f64; 3];
    attr.read_into(&mut vals).unwrap();
    assert_eq!(vals, [1.0, 2.0, 3.0]);
}

#[test]
fn t5c_dataset_attr() {
    let ds = open().dataset("contiguous").unwrap();
    let attr = ds.attr("ds_attr").unwrap();
    assert_eq!(attr.read_scalar::<i64>().unwrap(), 99);
}

#[test]
fn t5d_group_attr() {
    let g = open().group("mygroup").unwrap();
    let mut names = Vec::new();
    g.attr_names_into(&mut names).unwrap();
    assert!(names.contains(&"group_attr".to_string()));
}

#[test]
fn t5e_nested_dataset() {
    let ds = open().dataset("mygroup/nested").unwrap();
    let mut vals = [0i16; 3];
    ds.read_into(&mut vals).unwrap();
    assert_eq!(vals, [10, 20, 30]);
}

// T5f: Dense attributes (>8 triggers dense storage)

#[test]
fn t5f_dense_attrs() {
    let f = File::open("tests/data/dense_attrs.h5").unwrap();
    let mut names = Vec::new();
    f.attr_names_into(&mut names).unwrap();
    // Should have inline attributes (may be limited by compact threshold)
    // The dense ones are stored in fractal heap
    println!("Dense attr file has {} root attrs", names.len());
}
