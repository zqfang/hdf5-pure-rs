use hdf5_pure_rust::{Attribute, Dataset, File};

fn assert_dataset_strings<const N: usize>(ds: &Dataset, expected: [&str; N]) {
    let mut index = 0;
    ds.visit_strings(|value| {
        assert_eq!(value, expected[index]);
        index += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(index, N);
}

fn assert_attribute_strings<const N: usize>(attr: &Attribute, expected: [&str; N]) {
    let mut index = 0;
    attr.visit_strings(|value| {
        assert_eq!(value, expected[index]);
        index += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(index, N);
}

#[test]
fn test_read_fixed_strings() {
    let f = File::open("tests/data/strings.h5").unwrap();
    let ds = f.dataset("fixed_str").unwrap();

    let dtype = ds.dtype().unwrap();
    assert!(dtype.is_string());
    assert_eq!(dtype.size(), 10);

    assert_dataset_strings(&ds, ["hello", "world"]);
    let mut strings = vec!["stale".to_string()];
    ds.read_strings_into(&mut strings).unwrap();
    assert_eq!(strings, ["hello", "world"]);
}

#[test]
fn test_read_vlen_string_attr() {
    let f = File::open("tests/data/strings.h5").unwrap();
    let attr = f.attr("vlen_str").unwrap();

    assert!(attr.dtype().is_vlen());
    assert_attribute_strings(&attr, ["hello world"]);
    let mut value = String::new();
    attr.read_string_into(&mut value).unwrap();
    assert_eq!(value, "hello world");
}

#[test]
fn test_read_vlen_string_dataset() {
    let f = File::open("tests/data/strings.h5").unwrap();
    let ds = f.dataset("vlen_ds").unwrap();

    let dtype = ds.dtype().unwrap();
    assert!(dtype.is_vlen());

    assert_dataset_strings(&ds, ["alpha", "beta", "gamma"]);
    let mut strings = vec!["stale".to_string()];
    ds.read_strings_into(&mut strings).unwrap();
    assert_eq!(strings, ["alpha", "beta", "gamma"]);

    let mut first = String::from("stale");
    ds.read_string_into(&mut first).unwrap();
    assert_eq!(first, "alpha");
}
