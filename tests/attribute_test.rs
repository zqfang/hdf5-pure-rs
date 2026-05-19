use hdf5_pure_rust::format::messages::datatype::DatatypeClass;
use hdf5_pure_rust::{Attribute, Dataset, File, H5Type, Location, Result};

fn attribute_values<T>(attr: &Attribute) -> Result<Vec<T>>
where
    T: H5Type + Default + Clone,
{
    let len = attr.shape().iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim as usize)
            .ok_or_else(|| hdf5_pure_rust::Error::InvalidFormat("attribute shape overflows".into()))
    })?;
    let mut values = vec![T::default(); len];
    attr.read_into(&mut values)?;
    Ok(values)
}

fn attribute_strings(attr: &Attribute) -> Result<Vec<String>> {
    let mut strings = Vec::new();
    attr.visit_strings(|value| {
        strings.push(value.to_string());
        Ok(())
    })?;
    Ok(strings)
}

fn attribute_raw_data(attr: &Attribute) -> Result<Vec<u8>> {
    let mut data = vec![0; attr.info().data_size];
    attr.read_raw_into(&mut data)?;
    Ok(data)
}

fn dataset_attr_count_and_contains(ds: &Dataset, expected_name: &str) -> Result<(usize, bool)> {
    let mut count = 0usize;
    let mut found = false;
    ds.visit_attrs(|attr| {
        count += 1;
        found |= attr.name() == expected_name;
        Ok(())
    })?;
    Ok((count, found))
}

fn location_attr_names<L: Location>(location: &L) -> Result<Vec<String>> {
    let mut names = Vec::new();
    location.visit_attr_names(|name| {
        names.push(name.to_string());
        Ok(())
    })?;
    Ok(names)
}

fn location_attr_count_and_contains<L: Location>(
    location: &L,
    expected_name: &str,
) -> Result<(usize, bool)> {
    let mut count = 0usize;
    let mut found = false;
    location.visit_attrs(|attr| {
        count += 1;
        found |= attr.name() == expected_name;
        Ok(())
    })?;
    Ok((count, found))
}

fn location_attr_count_by_creation_order<L: Location>(location: &L) -> Result<usize> {
    let mut count = 0usize;
    location.visit_attrs_by_creation_order(|_| {
        count += 1;
        Ok(())
    })?;
    Ok(count)
}

fn location_attr_creation_orders_by_creation_order<L: Location>(location: &L) -> Result<Vec<u64>> {
    let mut creation_orders = Vec::new();
    location.visit_attrs_by_creation_order(|attr| {
        creation_orders.push(attr.creation_order().unwrap());
        Ok(())
    })?;
    Ok(creation_orders)
}

#[test]
fn test_list_root_attrs_v0() {
    let f = File::open("tests/data/attrs.h5").unwrap();
    let names = location_attr_names(&f).unwrap();
    let (attr_count, has_int_attr) = location_attr_count_and_contains(&f, "int_attr").unwrap();
    println!("v0 root attrs: {names:?}");

    assert!(names.contains(&"string_attr".to_string()));
    assert!(names.contains(&"int_attr".to_string()));
    assert!(names.contains(&"float_attr".to_string()));
    assert!(names.contains(&"array_attr".to_string()));
    assert_eq!(attr_count, names.len());
    assert!(has_int_attr);
    assert_eq!(f.attr_count().unwrap(), names.len());
    let mut first_name = String::new();
    f.attr_name_by_idx_into(0, &mut first_name).unwrap();
    assert_eq!(first_name, names[0]);
    assert_eq!(
        f.attr_info_by_idx(0).unwrap(),
        f.attr(&first_name).unwrap().info()
    );
    assert!(f
        .attr_name_by_idx_into(names.len(), &mut first_name)
        .is_err());
    assert!(f.attr_info_by_idx(names.len()).is_err());
}

#[test]
fn test_read_int_attr_v0() {
    let f = File::open("tests/data/attrs.h5").unwrap();
    let attr = f.attr("int_attr").unwrap();
    let val = attr.read_scalar::<i64>().unwrap();
    assert_eq!(val, 42);
}

#[test]
fn test_read_float_attr_v0() {
    let f = File::open("tests/data/attrs.h5").unwrap();
    let attr = f.attr("float_attr").unwrap();
    let val = attr.read_scalar::<f64>().unwrap();
    assert!((val - 3.14).abs() < 1e-10);
}

#[test]
fn test_read_array_attr_v0() {
    let f = File::open("tests/data/attrs.h5").unwrap();
    let attr = f.attr("array_attr").unwrap();
    assert_eq!(attr.shape(), &[3]);
    assert_eq!(attr.element_size(), 8);
    assert!(attr.dtype().is_float());
    assert_eq!(attr.dtype().class(), DatatypeClass::FloatingPoint);
    assert_eq!(attr.raw_datatype_message_ref().size, 8);
    assert!(attr.space().is_simple());
    assert_eq!(attr.space().shape(), &[3]);
    assert_eq!(attr.raw_dataspace_message_ref().dims.as_slice(), &[3]);
    let info = attr.info();
    assert!(!info.creation_order_valid);
    assert_eq!(info.creation_order, 0);
    assert_eq!(info.char_encoding, 0);
    assert_eq!(info.data_size, 24);
    assert_eq!(attr.create_plist().char_encoding(), info.char_encoding);

    let data = attribute_raw_data(&attr).unwrap();
    let values: Vec<f64> = data
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(values, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_dataset_attr_v0() {
    let f = File::open("tests/data/attrs.h5").unwrap();
    let ds = f.dataset("data").unwrap();
    let (attrs_len, has_ds_attr) = dataset_attr_count_and_contains(&ds, "ds_attr").unwrap();
    let attr = ds.attr("ds_attr").unwrap();
    let val = attr.read_scalar::<i64>().unwrap();
    assert_eq!(ds.attr_count().unwrap(), 1);
    let mut attr_name = String::new();
    ds.attr_name_by_idx_into(0, &mut attr_name).unwrap();
    assert_eq!(attr_name, "ds_attr");
    assert_eq!(ds.attr_info_by_idx(0).unwrap(), attr.info());
    assert_eq!(attrs_len, 1);
    assert!(has_ds_attr);
    assert_eq!(val, 100);
}

#[test]
fn test_attr_exists_on_file_group_and_dataset() {
    let f = File::open("tests/data/attrs.h5").unwrap();
    assert!(f.attr_exists("int_attr").unwrap());
    assert!(!f.attr_exists("missing_attr").unwrap());

    let ds = f.dataset("data").unwrap();
    assert!(ds.attr_exists("ds_attr").unwrap());
    assert!(!ds.attr_exists("missing_attr").unwrap());

    let f = File::open("tests/data/simple_v0.h5").unwrap();
    let group = f.group("group1").unwrap();
    assert!(!group.attr_exists("missing_attr").unwrap());
}

#[test]
fn test_list_root_attrs_v3() {
    let f = File::open("tests/data/attrs_v3.h5").unwrap();
    let names = location_attr_names(&f).unwrap();
    println!("v3 root attrs: {names:?}");

    assert!(names.contains(&"string_attr".to_string()));
    assert!(names.contains(&"int_attr".to_string()));
    assert!(names.contains(&"float_attr".to_string()));
}

#[test]
fn test_read_int_attr_v3() {
    let f = File::open("tests/data/attrs_v3.h5").unwrap();
    let attr = f.attr("int_attr").unwrap();
    let val = attr.read_scalar::<i64>().unwrap();
    assert_eq!(val, 42);
}

#[test]
fn test_dataset_attr_v3() {
    let f = File::open("tests/data/attrs_v3.h5").unwrap();
    let ds = f.dataset("data").unwrap();
    let attr = ds.attr("ds_attr").unwrap();
    let val = attr.read_scalar::<i64>().unwrap();
    assert_eq!(val, 100);
}

#[test]
fn test_large_compact_attribute_read() {
    let f = File::open("tests/data/hdf5_ref/attribute_cases.h5").unwrap();
    let group = f.group("large_compact_attrs").unwrap();
    let attr = group.attr("large_i32").unwrap();
    let values = attribute_values::<i32>(&attr).unwrap();

    assert_eq!(attr.shape(), &[256]);
    assert_eq!(values.len(), 256);
    assert_eq!(values[0], 0);
    assert_eq!(values[255], 255);
}

#[test]
fn test_dense_attributes_creation_order_indexing_enabled_and_disabled() {
    let f = File::open("tests/data/hdf5_ref/attribute_cases.h5").unwrap();

    let tracked = f.group("dense_attrs_tracked").unwrap();
    let tracked_names = location_attr_names(&tracked).unwrap();
    let (tracked_attr_count, has_attr_07) =
        location_attr_count_and_contains(&tracked, "attr_07").unwrap();
    assert_eq!(tracked_names.len(), 32);
    assert_eq!(tracked_attr_count, 32);
    assert!(tracked_names.contains(&"attr_00".to_string()));
    assert!(tracked_names.contains(&"attr_31".to_string()));
    assert!(has_attr_07);
    let tracked_creation_orders =
        location_attr_creation_orders_by_creation_order(&tracked).unwrap();
    assert_eq!(tracked_creation_orders.len(), 32);
    assert_eq!(tracked_creation_orders, (0..32).collect::<Vec<_>>());
    assert_eq!(
        attribute_values::<i32>(&tracked.attr("attr_07").unwrap()).unwrap(),
        vec![7, 107]
    );

    let untracked = f.group("dense_attrs_untracked").unwrap();
    let untracked_names = location_attr_names(&untracked).unwrap();
    assert_eq!(untracked_names.len(), 32);
    assert!(untracked_names.contains(&"attr_00".to_string()));
    assert!(untracked_names.contains(&"attr_31".to_string()));
    assert_eq!(
        location_attr_count_by_creation_order(&untracked).unwrap(),
        32
    );
    assert_eq!(
        attribute_values::<i32>(&untracked.attr("attr_07").unwrap()).unwrap(),
        vec![7, 207]
    );

    let old_file = File::open("tests/data/attrs.h5").unwrap();
    assert!(location_attr_count_by_creation_order(&old_file).is_err());
}

#[test]
fn test_variable_length_attribute_payload_raw_read() {
    let f = File::open("tests/data/hdf5_ref/attribute_cases.h5").unwrap();
    let group = f.group("vlen_attr_holder").unwrap();
    let attr = group.attr("vlen_strings").unwrap();

    assert_eq!(attr.shape(), &[3]);
    assert_eq!(attr.element_size(), 16);
    let raw = attribute_raw_data(&attr).unwrap();
    assert_eq!(raw.len(), 48);
    assert!(raw.iter().any(|&b| b != 0));
    assert_eq!(attribute_strings(&attr).unwrap(), vec!["", "alpha", "猫"]);
}
