use hdf5_pure_rust::engine::writer::{CompoundFieldSpec, DtypeSpec};
use hdf5_pure_rust::format::messages::data_layout::ChunkIndexType;
use hdf5_pure_rust::hl::types::{FieldDescriptor, H5Type, TypeClass};
use hdf5_pure_rust::{Attribute, Dataset, H5Value, Location, Result, WritableFile};

fn file_has_member(file: &hdf5_pure_rust::File, expected: &str) -> hdf5_pure_rust::Result<bool> {
    let mut found = false;
    file.visit_member_names(|name| {
        if name == expected {
            found = true;
        }
        Ok(())
    })?;
    Ok(found)
}

fn group_member_summary_into(
    group: &hdf5_pure_rust::Group,
    expected: &[&str],
    found: &mut [bool],
) -> hdf5_pure_rust::Result<usize> {
    assert_eq!(expected.len(), found.len());
    found.fill(false);
    let mut count = 0;
    group.visit_member_names(|name| {
        count += 1;
        for (idx, expected_name) in expected.iter().enumerate() {
            if name == *expected_name {
                found[idx] = true;
            }
        }
        Ok(())
    })?;
    Ok(count)
}

fn location_attr_summary_into<L: Location>(
    location: &L,
    expected: &[&str],
    found: &mut [bool],
) -> Result<usize> {
    assert_eq!(expected.len(), found.len());
    found.fill(false);
    let mut count = 0;
    location.visit_attr_names(|name| {
        count += 1;
        for (idx, expected_name) in expected.iter().enumerate() {
            if name == *expected_name {
                found[idx] = true;
            }
        }
        Ok(())
    })?;
    Ok(count)
}

fn assert_dataset_values<T>(ds: &Dataset, expected: &[T]) -> Result<()>
where
    T: H5Type + Default + Clone + PartialEq + std::fmt::Debug,
{
    let mut values = vec![T::default(); ds.size()? as usize];
    ds.read_into(&mut values)?;
    assert_eq!(values, expected);
    Ok(())
}

fn assert_dataset_raw(ds: &Dataset, expected: &[u8]) -> Result<()> {
    let mut raw = vec![0; ds.size()? as usize * ds.element_size()?];
    ds.read_raw_into(&mut raw)?;
    assert_eq!(raw, expected);
    Ok(())
}

fn dataset_scalar<T>(ds: &Dataset) -> Result<T>
where
    T: H5Type + Default,
{
    let mut value = T::default();
    ds.read_scalar_into(&mut value)?;
    Ok(value)
}

fn assert_dataset_strings(ds: &Dataset, expected: &[&str]) -> Result<()> {
    let mut strings = Vec::new();
    ds.read_strings_into(&mut strings)?;
    assert_eq!(strings.len(), expected.len());
    for (actual, expected) in strings.iter().zip(expected) {
        assert_eq!(actual, expected);
    }
    Ok(())
}

fn assert_dataset_field<T>(ds: &Dataset, field_name: &str, expected: &[T]) -> Result<()>
where
    T: H5Type + Default + Clone + PartialEq + std::fmt::Debug,
{
    let mut values = vec![T::default(); ds.size()? as usize];
    ds.read_field_into(field_name, &mut values)?;
    assert_eq!(values, expected);
    Ok(())
}

fn assert_dataset_field_values(ds: &Dataset, field_name: &str, expected: &[H5Value]) -> Result<()> {
    let mut values = Vec::new();
    ds.read_field_values_into(field_name, &mut values)?;
    assert_eq!(values, expected);
    Ok(())
}

fn dataset_field_value_count(ds: &Dataset, field_name: &str) -> Result<usize> {
    let mut count = 0;
    ds.visit_field_values(field_name, |_| {
        count += 1;
        Ok(())
    })?;
    Ok(count)
}

fn dataset_has_filter_id(ds: &Dataset, id: u16) -> Result<bool> {
    let mut found = false;
    ds.visit_filters(|filter| {
        if filter.id == id {
            found = true;
        }
        Ok(())
    })?;
    Ok(found)
}

fn assert_attribute_values<T>(attr: &Attribute, expected: &[T]) -> Result<()>
where
    T: H5Type + Default + Clone + PartialEq + std::fmt::Debug,
{
    let len = attr.shape().iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim as usize)
            .ok_or_else(|| hdf5_pure_rust::Error::InvalidFormat("attribute shape overflows".into()))
    })?;
    let mut values = vec![T::default(); len];
    attr.read_into(&mut values)?;
    assert_eq!(values, expected);
    Ok(())
}

fn attribute_string(attr: &Attribute) -> Result<String> {
    let mut value = String::new();
    attr.read_string_into(&mut value)?;
    Ok(value)
}

fn assert_attribute_strings(attr: &Attribute, expected: &[&str]) -> Result<()> {
    let mut index = 0;
    attr.visit_strings(|value| {
        assert_eq!(Some(value), expected.get(index).copied());
        index += 1;
        Ok(())
    })?;
    assert_eq!(index, expected.len());
    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Point {
    x: f64,
    label: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct WidePair {
    signed: i128,
    unsigned: u128,
}

unsafe impl H5Type for Point {
    fn type_size() -> usize {
        std::mem::size_of::<Point>()
    }

    fn compound_fields_into(out: &mut Vec<FieldDescriptor>) -> Option<()> {
        out.clear();
        out.extend([
            FieldDescriptor {
                name: "x".to_string(),
                offset: std::mem::offset_of!(Point, x),
                size: std::mem::size_of::<f64>(),
                type_class: TypeClass::Float,
            },
            FieldDescriptor {
                name: "label".to_string(),
                offset: std::mem::offset_of!(Point, label),
                size: std::mem::size_of::<i32>(),
                type_class: TypeClass::Integer { signed: true },
            },
        ]);
        Some(())
    }
}

unsafe impl H5Type for WidePair {
    fn type_size() -> usize {
        std::mem::size_of::<WidePair>()
    }

    fn compound_fields_into(out: &mut Vec<FieldDescriptor>) -> Option<()> {
        out.clear();
        out.extend([
            FieldDescriptor {
                name: "signed".to_string(),
                offset: std::mem::offset_of!(WidePair, signed),
                size: std::mem::size_of::<i128>(),
                type_class: TypeClass::Integer { signed: true },
            },
            FieldDescriptor {
                name: "unsigned".to_string(),
                offset: std::mem::offset_of!(WidePair, unsigned),
                size: std::mem::size_of::<u128>(),
                type_class: TypeClass::Integer { signed: false },
            },
        ]);
        Some(())
    }
}

#[test]
fn test_writable_file_simple() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_simple.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("temperatures")
            .shape(&[5])
            .write::<f64>(&[20.0, 21.5, 22.0, 19.8, 23.1])
            .unwrap();
        let f = wf.close().unwrap();
        let ds = f.dataset("temperatures").unwrap();
        assert_dataset_values::<f64>(&ds, &[20.0, 21.5, 22.0, 19.8, 23.1]).unwrap();
    }
}

#[test]
fn test_writable_file_i128_u128_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_i128_u128.h5");
    let signed = [i128::MIN, -42, 0, i128::MAX];
    let unsigned = [0u128, 42, 1u128 << 96, u128::MAX];

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("signed")
            .shape(&[signed.len() as u64])
            .write::<i128>(&signed)
            .unwrap();
        wf.new_dataset_builder("unsigned")
            .shape(&[unsigned.len() as u64])
            .write::<u128>(&unsigned)
            .unwrap();
        wf.add_attr("wide_attr", i128::MIN + 7).unwrap();
        wf.flush().unwrap();
    }

    let f = hdf5_pure_rust::File::open(&path).unwrap();
    assert_dataset_values::<i128>(&f.dataset("signed").unwrap(), &signed).unwrap();
    assert_dataset_values::<u128>(&f.dataset("unsigned").unwrap(), &unsigned).unwrap();
    assert_eq!(
        f.attr("wide_attr").unwrap().read_scalar::<i128>().unwrap(),
        i128::MIN + 7
    );
}

#[test]
fn test_writable_file_compound_i128_u128_fields_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_compound_i128_u128.h5");
    let data = [
        WidePair {
            signed: -123456789012345678901234567890i128,
            unsigned: 123456789012345678901234567890u128,
        },
        WidePair {
            signed: i128::MAX,
            unsigned: u128::MAX,
        },
    ];

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("wide_pairs")
            .shape(&[data.len() as u64])
            .write::<WidePair>(&data)
            .unwrap();
        wf.flush().unwrap();
    }

    let f = hdf5_pure_rust::File::open(&path).unwrap();
    let ds = f.dataset("wide_pairs").unwrap();
    let expected_signed = data.iter().map(|value| value.signed).collect::<Vec<_>>();
    let expected_unsigned = data.iter().map(|value| value.unsigned).collect::<Vec<_>>();
    let expected_signed_values = data
        .iter()
        .map(|value| H5Value::Int(value.signed))
        .collect::<Vec<_>>();
    let expected_unsigned_values = data
        .iter()
        .map(|value| H5Value::UInt(value.unsigned))
        .collect::<Vec<_>>();
    assert_dataset_field::<i128>(&ds, "signed", &expected_signed).unwrap();
    assert_dataset_field::<u128>(&ds, "unsigned", &expected_unsigned).unwrap();
    assert_dataset_field_values(&ds, "signed", &expected_signed_values).unwrap();
    assert_dataset_field_values(&ds, "unsigned", &expected_unsigned_values).unwrap();
}

#[test]
fn test_dataset_builder_rejects_shape_data_length_mismatch() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_shape_mismatch.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        let err = wf
            .new_dataset_builder("bad")
            .shape(&[2, 3])
            .write::<i32>(&[1, 2, 3, 4, 5])
            .expect_err("shape/data element count mismatch should be rejected");
        assert!(err
            .to_string()
            .contains("does not match shape element count"));
    }
}

#[test]
fn test_dataset_builder_rejects_excessive_dataspace_rank() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_rank_too_large.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        let err = wf
            .new_dataset_builder("bad_rank")
            .shape(&vec![1; 33])
            .write::<u8>(&[0])
            .expect_err("dataspace rank above supported maximum should be rejected");
        assert!(err.to_string().contains("dataspace rank"));
    }
}

#[test]
fn test_writable_rejects_duplicate_child_names_and_invalid_link_names() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_duplicate_child_names.h5");

    let mut wf = WritableFile::create(&path).unwrap();
    wf.new_dataset_builder("data")
        .write::<i32>(&[1, 2, 3])
        .unwrap();

    let err = wf
        .new_dataset_builder("data")
        .write::<i32>(&[4, 5, 6])
        .expect_err("duplicate dataset name should be rejected");
    assert!(err.to_string().contains("already exists"));

    let err = match wf.create_group("bad/name") {
        Ok(_) => panic!("slash-containing group name should be rejected"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("must not contain '/'"));

    let err = wf
        .link_soft("bad/link", "/data")
        .expect_err("slash-containing soft-link name should be rejected");
    assert!(err.to_string().contains("must not contain '/'"));

    let err = wf
        .link_soft("data", "/data")
        .expect_err("soft link must not shadow an existing dataset");
    assert!(err.to_string().contains("already exists"));

    wf.link_soft("alias", "/data").unwrap();
    let err = match wf.create_group("alias") {
        Ok(_) => panic!("group must not shadow a pending soft link"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("already exists"));
}

#[test]
fn test_writable_file_with_groups() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_groups.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        let mut g = wf.create_group("sensors").unwrap();
        g.new_dataset_builder("pressure")
            .write::<f32>(&[1013.25, 1012.0, 1011.5])
            .unwrap();
        let f = wf.close().unwrap();

        assert!(file_has_member(&f, "sensors").unwrap());

        let g = f.group("sensors").unwrap();
        let ds = g.open_dataset("pressure").unwrap();
        assert_dataset_values::<f32>(&ds, &[1013.25, 1012.0, 1011.5]).unwrap();
    }
}

#[test]
fn test_writable_group_attr() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_group_attr.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        let mut g = wf.create_group("sensors").unwrap();
        g.add_attr("site_id", 17i64).unwrap();
        g.new_dataset_builder("pressure")
            .write::<f32>(&[1013.25, 1012.0])
            .unwrap();
        let f = wf.close().unwrap();

        let g = f.group("sensors").unwrap();
        let mut found = [false];
        location_attr_summary_into(&g, &["site_id"], &mut found).unwrap();
        assert!(found[0]);
        let value: i64 = g.attr("site_id").unwrap().read_scalar::<i64>().unwrap();
        assert_eq!(value, 17);
    }
}

#[test]
fn test_writable_rejects_duplicate_attribute_names() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_duplicate_attrs.h5");

    let mut wf = WritableFile::create(&path).unwrap();
    wf.add_attr("version", 1i32).unwrap();
    let err = wf
        .add_attr("version", 2i32)
        .expect_err("duplicate root attribute should be rejected");
    assert!(err.to_string().contains("already exists"));

    let mut group = wf.create_group("metadata").unwrap();
    group.add_attr("version", 1i32).unwrap();
    let err = group
        .add_attr_array("version", &[2i32, 3])
        .expect_err("duplicate group attribute should be rejected");
    assert!(err.to_string().contains("already exists"));

    let err = match wf
        .new_dataset_builder("values")
        .attr("units", 1i32)
        .unwrap()
        .fixed_ascii_attr("units", "ms", 8)
    {
        Ok(_) => panic!("duplicate dataset builder attribute should be rejected"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("already exists"));
}

#[test]
fn test_writable_group_dense_attrs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_group_dense_attrs.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        let mut g = wf.create_group("annotated").unwrap();
        for idx in 0..16 {
            g.add_attr(&format!("attr_{idx:02}"), idx as i64).unwrap();
        }
        let f = wf.close().unwrap();

        let g = f.group("annotated").unwrap();
        let mut found = [false, false];
        let count = location_attr_summary_into(&g, &["attr_00", "attr_15"], &mut found).unwrap();
        assert_eq!(count, 16);
        assert!(found[0]);
        assert!(found[1]);
        assert_eq!(g.attr("attr_12").unwrap().read_scalar_i64(), Some(12));
    }
}

#[test]
fn test_writable_root_and_group_array_attrs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_array_attrs.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.add_attr_array("calibration", &[1.0f64, 2.5, 4.0])
            .unwrap();
        let mut g = wf.create_group("sensors").unwrap();
        g.add_attr_array("ids", &[10i32, 20, 30, 40]).unwrap();
        let f = wf.close().unwrap();

        let root_attr = f.attr("calibration").unwrap();
        assert_eq!(root_attr.shape(), &[3]);
        assert_attribute_values::<f64>(&root_attr, &[1.0, 2.5, 4.0]).unwrap();

        let group = f.group("sensors").unwrap();
        let group_attr = group.attr("ids").unwrap();
        assert_eq!(group_attr.shape(), &[4]);
        assert_attribute_values::<i32>(&group_attr, &[10, 20, 30, 40]).unwrap();
    }
}

#[test]
fn test_writable_fixed_string_attrs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_string_attrs.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.add_fixed_ascii_attr("project", "hdf", 8).unwrap();
        let mut g = wf.create_group("metadata").unwrap();
        g.add_fixed_utf8_attr("species", "猫", 8).unwrap();
        let f = wf.close().unwrap();

        assert_eq!(
            attribute_string(&f.attr("project").unwrap()).unwrap(),
            "hdf"
        );
        let group = f.group("metadata").unwrap();
        assert_eq!(
            attribute_string(&group.attr("species").unwrap()).unwrap(),
            "猫"
        );
    }
}

#[test]
#[cfg(target_pointer_width = "64")]
fn test_writable_rejects_unrepresentable_fixed_string_lengths() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_string_len_overflow.h5");
    let too_long = u32::MAX as usize + 1;

    let mut wf = WritableFile::create(&path).unwrap();
    let err = wf
        .add_fixed_ascii_attr("root", "x", too_long)
        .expect_err("root fixed string length should fit encoded u32");
    assert!(err.to_string().contains("fixed string length"));

    let err = wf
        .create_group("metadata")
        .unwrap()
        .add_fixed_utf8_attr("group", "x", too_long)
        .expect_err("group fixed string length should fit encoded u32");
    assert!(err.to_string().contains("fixed string length"));

    let err = match wf
        .new_dataset_builder("values")
        .fixed_ascii_attr("units", "x", too_long)
    {
        Ok(_) => panic!("dataset attribute fixed string length should fit encoded u32"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("fixed string length"));

    let err = wf
        .new_dataset_builder("names")
        .write_fixed_ascii_strings(&["x"], too_long)
        .expect_err("dataset fixed string length should fit encoded u32");
    assert!(err.to_string().contains("fixed string length"));
}

#[test]
fn test_writable_routes_oversized_attrs_to_dense_storage() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_oversized_dense_attrs.h5");
    let large_len = u16::MAX as usize + 1;

    {
        let mut wf = WritableFile::create(&path).unwrap();
        let mut group = wf.create_group("metadata").unwrap();
        group
            .add_fixed_ascii_attr("large_attr", "x", large_len)
            .unwrap();
        wf.new_dataset_builder("data")
            .fixed_ascii_attr("large_ds_attr", "y", large_len)
            .unwrap()
            .write::<i32>(&[1, 2, 3])
            .unwrap();
        let f = wf.close().unwrap();

        let group = f.group("metadata").unwrap();
        assert_eq!(
            attribute_string(&group.attr("large_attr").unwrap()).unwrap(),
            "x"
        );
        let ds = f.dataset("data").unwrap();
        assert_dataset_values::<i32>(&ds, &[1, 2, 3]).unwrap();
        assert_eq!(
            attribute_string(&ds.attr("large_ds_attr").unwrap()).unwrap(),
            "y"
        );
    }
}

#[test]
fn test_writable_dense_attrs_support_wider_heap_ids() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_dense_attr_wide_heap_id.h5");
    let large_len = (1usize << 24) + 1;

    {
        let mut wf = WritableFile::create(&path).unwrap();
        let mut group = wf.create_group("metadata").unwrap();
        group
            .add_fixed_ascii_attr("very_large_attr", "z", large_len)
            .unwrap();
        for idx in 0..8 {
            group.add_attr(&format!("attr_{idx}"), idx as i32).unwrap();
        }
        let f = wf.close().unwrap();

        let group = f.group("metadata").unwrap();
        assert_eq!(
            attribute_string(&group.attr("very_large_attr").unwrap()).unwrap(),
            "z"
        );
        assert_attribute_values::<i32>(&group.attr("attr_7").unwrap(), &[7]).unwrap();
    }

    assert!(std::fs::read(&path)
        .unwrap()
        .windows(4)
        .any(|window| window == b"BTHD"));
}

#[test]
fn test_writable_rejects_oversized_attribute_name_field() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_oversized_attr_name.h5");

    let mut wf = WritableFile::create(&path).unwrap();
    let name = "a".repeat(u16::MAX as usize);
    wf.add_attr(&name, 1i32).unwrap();
    let err = wf
        .flush()
        .expect_err("attribute name length should fit u16 field");
    assert!(err.to_string().contains("attribute name"));
}

#[test]
fn test_writable_fixed_string_array_attrs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_string_array_attrs.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.add_fixed_ascii_attr_array("stages", &["raw", "qc", "done"], 8)
            .unwrap();
        let mut g = wf.create_group("metadata").unwrap();
        g.add_fixed_utf8_attr_array("labels", &["猫", "å"], 8)
            .unwrap();
        let f = wf.close().unwrap();

        let stages = f.attr("stages").unwrap();
        assert_eq!(stages.shape(), &[3]);
        assert_attribute_strings(&stages, &["raw", "qc", "done"]).unwrap();

        let labels = f.group("metadata").unwrap().attr("labels").unwrap();
        assert_eq!(labels.shape(), &[2]);
        assert_attribute_strings(&labels, &["猫", "å"]).unwrap();
    }
}

#[test]
fn test_dataset_builder_scalar_and_array_attrs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_dataset_builder_attrs.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("values")
            .attr("version", 2i64)
            .unwrap()
            .attr_array("scale", &[1.0f64, 10.0, 100.0])
            .unwrap()
            .write::<i32>(&[4, 5, 6])
            .unwrap();
        let f = wf.close().unwrap();

        let ds = f.dataset("values").unwrap();
        assert_eq!(ds.attr("version").unwrap().read_scalar_i64(), Some(2));
        let scale = ds.attr("scale").unwrap();
        assert_eq!(scale.shape(), &[3]);
        assert_attribute_values::<f64>(&scale, &[1.0, 10.0, 100.0]).unwrap();
    }
}

#[test]
fn test_dataset_builder_fixed_string_attrs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_dataset_string_attrs.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("values")
            .fixed_ascii_attr("units", "ms", 8)
            .unwrap()
            .fixed_utf8_attr("label", "猫", 8)
            .unwrap()
            .write::<i32>(&[1, 2, 3])
            .unwrap();
        let f = wf.close().unwrap();

        let ds = f.dataset("values").unwrap();
        assert_eq!(attribute_string(&ds.attr("units").unwrap()).unwrap(), "ms");
        assert_eq!(attribute_string(&ds.attr("label").unwrap()).unwrap(), "猫");
    }
}

#[test]
fn test_dataset_builder_fixed_string_array_attrs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_dataset_string_array_attrs.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("values")
            .fixed_ascii_attr_array("units", &["ms", "s"], 8)
            .unwrap()
            .fixed_utf8_attr_array("labels", &["猫", "å"], 8)
            .unwrap()
            .write::<i32>(&[1, 2, 3])
            .unwrap();
        let f = wf.close().unwrap();

        let ds = f.dataset("values").unwrap();
        assert_attribute_strings(&ds.attr("units").unwrap(), &["ms", "s"]).unwrap();
        assert_attribute_strings(&ds.attr("labels").unwrap(), &["猫", "å"]).unwrap();
    }
}

#[test]
fn test_dataset_builder_compact_attrs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_compact_dataset_attrs.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("values")
            .compact()
            .attr("version", 3i64)
            .unwrap()
            .write::<i16>(&[7, 8, 9])
            .unwrap();
        let f = wf.close().unwrap();

        let ds = f.dataset("values").unwrap();
        assert_dataset_values::<i16>(&ds, &[7, 8, 9]).unwrap();
        assert_eq!(ds.attr("version").unwrap().read_scalar_i64(), Some(3));
    }
}

#[test]
fn test_dataset_builder_attrs_with_explicit_fill_values() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir
        .path()
        .join("api_write_dataset_attrs_with_fill_values.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("contiguous")
            .fill_value::<i32>(-7)
            .attr("version", 4i64)
            .unwrap()
            .write::<i32>(&[1, 2, 3])
            .unwrap();
        wf.new_dataset_builder("compact")
            .compact()
            .fill_value::<i16>(-2)
            .attr("version", 5i64)
            .unwrap()
            .write::<i16>(&[7, 8])
            .unwrap();
        wf.new_dataset_builder("scalar")
            .fill_value::<u64>(99)
            .attr("version", 6i64)
            .unwrap()
            .write_scalar::<u64>(42)
            .unwrap();
        let f = wf.close().unwrap();

        let contiguous = f.dataset("contiguous").unwrap();
        let plist = contiguous.create_plist().unwrap();
        assert_eq!(plist.fill_value, Some((-7i32).to_le_bytes().to_vec()));
        assert_eq!(
            contiguous.attr("version").unwrap().read_scalar_i64(),
            Some(4)
        );
        assert_dataset_values::<i32>(&contiguous, &[1, 2, 3]).unwrap();

        let compact = f.dataset("compact").unwrap();
        let plist = compact.create_plist().unwrap();
        assert_eq!(plist.fill_value, Some((-2i16).to_le_bytes().to_vec()));
        assert_eq!(compact.attr("version").unwrap().read_scalar_i64(), Some(5));
        assert_dataset_values::<i16>(&compact, &[7, 8]).unwrap();

        let scalar = f.dataset("scalar").unwrap();
        let plist = scalar.create_plist().unwrap();
        assert_eq!(plist.fill_value, Some(99u64.to_le_bytes().to_vec()));
        assert_eq!(scalar.attr("version").unwrap().read_scalar_i64(), Some(6));
        assert_eq!(dataset_scalar::<u64>(&scalar).unwrap(), 42);
    }
}

#[test]
fn test_writable_file_group_with_many_compact_links() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_many_links.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        let mut g = wf.create_group("many").unwrap();
        for idx in 0..40 {
            let name = format!("value_{idx:02}");
            g.new_dataset_builder(&name).write::<i32>(&[idx]).unwrap();
        }
        let f = wf.close().unwrap();

        let group = f.group("many").unwrap();
        let mut found = [false, false];
        let count =
            group_member_summary_into(&group, &["value_00", "value_39"], &mut found).unwrap();
        assert_eq!(count, 40);
        assert!(found[0]);
        assert!(found[1]);
        let value_37 = group.open_dataset("value_37").unwrap();
        assert_dataset_values::<i32>(&value_37, &[37]).unwrap();
    }

    let out = std::process::Command::new("h5dump")
        .arg("-H")
        .arg(&path)
        .output();
    if let Ok(out) = out {
        assert!(
            out.status.success(),
            "h5dump failed on many-link writer fixture: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

#[test]
fn test_writable_dense_links_support_wider_heap_ids() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_dense_link_wide_heap_id.h5");
    let long_name = format!("value_{}", "x".repeat(u16::MAX as usize + 64));

    {
        let mut wf = WritableFile::create(&path).unwrap();
        let mut g = wf.create_group("many").unwrap();
        for idx in 0..8 {
            let name = format!("value_{idx:02}");
            g.new_dataset_builder(&name).write::<i32>(&[idx]).unwrap();
        }
        g.new_dataset_builder(&long_name)
            .write::<i32>(&[99])
            .unwrap();
        let f = wf.close().unwrap();

        let group = f.group("many").unwrap();
        let expected = ["value_00", long_name.as_str()];
        let mut found = [false, false];
        let count = group_member_summary_into(&group, &expected, &mut found).unwrap();
        assert_eq!(count, 9);
        assert!(found[0]);
        assert!(found[1]);
    }

    assert!(std::fs::read(&path)
        .unwrap()
        .windows(4)
        .any(|window| window == b"BTHD"));
}

#[test]
fn test_writable_file_chunked_compressed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_chunked.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        let data: Vec<i32> = (0..100).collect();
        wf.new_dataset_builder("data")
            .shape(&[100])
            .chunk(&[25])
            .deflate(4)
            .shuffle()
            .write::<i32>(&data)
            .unwrap();
        let f = wf.close().unwrap();

        let ds = f.dataset("data").unwrap();
        assert!(ds.is_chunked().unwrap());
        let mut vals = vec![0i32; ds.size().unwrap() as usize];
        ds.read_into(&mut vals).unwrap();
        assert_eq!(vals.len(), 100);
        for (i, v) in vals.iter().enumerate() {
            assert_eq!(*v, i as i32);
        }
    }
}

#[test]
fn test_writable_file_chunked_v1_btree_beyond_two_levels() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_chunked_deep_btree.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        let data: Vec<i32> = (0..4097).collect();
        wf.new_dataset_builder("data")
            .shape(&[4097])
            .chunk(&[1])
            .write::<i32>(&data)
            .unwrap();
        let f = wf.close().unwrap();

        let ds = f.dataset("data").unwrap();
        assert!(ds.is_chunked().unwrap());
        assert_dataset_values::<i32>(&ds, &data).unwrap();
    }
}

#[test]
fn test_writable_file_chunked_fletcher32() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_chunked_fletcher32.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        let data: Vec<i32> = (0..64).collect();
        wf.new_dataset_builder("data")
            .shape(&[64])
            .chunk(&[16])
            .fletcher32()
            .write::<i32>(&data)
            .unwrap();
        let f = wf.close().unwrap();

        let ds = f.dataset("data").unwrap();
        assert!(ds.is_chunked().unwrap());
        assert_dataset_values::<i32>(&ds, &data).unwrap();
        assert!(dataset_has_filter_id(&ds, 3).unwrap());
    }
}

#[test]
fn test_dataset_builder_writes_max_shape_dataspace() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_max_shape.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("finite")
            .shape(&[4])
            .max_shape(&[10])
            .chunk(&[2])
            .write::<i32>(&[1, 2, 3, 4])
            .unwrap();
        wf.new_dataset_builder("unlimited_2d")
            .shape(&[2, 3])
            .resizable()
            .chunk(&[1, 3])
            .write::<i32>(&[1, 2, 3, 4, 5, 6])
            .unwrap();
        let f = wf.close().unwrap();

        let finite = f.dataset("finite").unwrap();
        let finite_space = finite.space().unwrap();
        assert_eq!(finite_space.shape(), &[4]);
        assert_eq!(finite_space.maxdims().unwrap(), &[10]);
        let finite_info = finite.info().unwrap();
        assert_eq!(finite_info.layout.version, 4);
        assert_eq!(
            finite_info.layout.chunk_index_type,
            Some(ChunkIndexType::ExtensibleArray)
        );
        assert_dataset_values::<i32>(&finite, &[1, 2, 3, 4]).unwrap();

        let unlimited = f.dataset("unlimited_2d").unwrap();
        let unlimited_space = unlimited.space().unwrap();
        assert_eq!(unlimited_space.shape(), &[2, 3]);
        assert_eq!(unlimited_space.maxdims().unwrap(), &[u64::MAX, u64::MAX]);
        let unlimited_info = unlimited.info().unwrap();
        assert_eq!(unlimited_info.layout.version, 4);
        assert_eq!(
            unlimited_info.layout.chunk_index_type,
            Some(ChunkIndexType::ExtensibleArray)
        );
        assert_dataset_values::<i32>(&unlimited, &[1, 2, 3, 4, 5, 6]).unwrap();
    }
}

#[test]
fn test_dataset_builder_chunked_attrs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_chunked_attrs.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        let data: Vec<i32> = (0..16).collect();
        wf.new_dataset_builder("data")
            .shape(&[16])
            .chunk(&[4])
            .attr("version", 7i64)
            .unwrap()
            .attr_array("scale", &[1.0f64, 2.0])
            .unwrap()
            .write::<i32>(&data)
            .unwrap();
        let f = wf.close().unwrap();

        let ds = f.dataset("data").unwrap();
        assert!(ds.is_chunked().unwrap());
        assert_dataset_values::<i32>(&ds, &data).unwrap();
        assert_eq!(ds.attr("version").unwrap().read_scalar_i64(), Some(7));
        assert_attribute_values::<f64>(&ds.attr("scale").unwrap(), &[1.0, 2.0]).unwrap();
    }
}

#[test]
fn test_dataset_builder_rejects_invalid_chunk_dimensions() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_invalid_chunk_dims.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        let zero_err = wf
            .new_dataset_builder("zero_chunk")
            .shape(&[4])
            .chunk(&[0])
            .write::<i32>(&[1, 2, 3, 4])
            .expect_err("zero chunk dimension should be rejected");
        assert!(zero_err.to_string().contains("chunk dimension 0 is zero"));

        let rank_err = wf
            .new_dataset_builder("rank_mismatch")
            .shape(&[2, 2])
            .chunk(&[2])
            .write::<i32>(&[1, 2, 3, 4])
            .expect_err("chunk rank mismatch should be rejected");
        assert!(rank_err.to_string().contains("chunk dimension rank"));

        let huge_chunk_err = wf
            .new_dataset_builder("huge_chunk")
            .shape(&[1])
            .chunk(&[u32::MAX as u64 + 1])
            .write::<u8>(&[0])
            .expect_err("chunk dimensions must fit v3 layout's 32-bit fields");
        assert!(huge_chunk_err.to_string().contains("32-bit layout field"));
    }
}

#[test]
fn test_dataset_builder_rejects_invalid_deflate_level() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_invalid_deflate_level.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        let err = wf
            .new_dataset_builder("bad_deflate")
            .shape(&[4])
            .chunk(&[2])
            .deflate(10)
            .write::<i32>(&[1, 2, 3, 4])
            .expect_err("deflate level above 9 should be rejected");
        assert!(err.to_string().contains("deflate compression level 10"));
    }
}

#[test]
fn test_dataset_builder_compressed_chunked_attrs_with_fill_value() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir
        .path()
        .join("api_write_compressed_chunked_attrs_with_fill.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        let data: Vec<i16> = (0..30).collect();
        wf.new_dataset_builder("data")
            .shape(&[30])
            .chunk(&[8])
            .shuffle()
            .deflate(3)
            .fill_value::<i16>(-9)
            .attr("version", 8i64)
            .unwrap()
            .write::<i16>(&data)
            .unwrap();
        let f = wf.close().unwrap();

        let ds = f.dataset("data").unwrap();
        assert!(ds.is_chunked().unwrap());
        assert_dataset_values::<i16>(&ds, &data).unwrap();
        assert_eq!(ds.attr("version").unwrap().read_scalar_i64(), Some(8));
        let plist = ds.create_plist().unwrap();
        assert_eq!(plist.fill_value, Some((-9i16).to_le_bytes().to_vec()));
    }
}

#[test]
fn test_dataset_builder_sparse_chunked_fill_only_dataset() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_sparse_chunked_fill_only.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("data")
            .shape(&[1_000])
            .chunk(&[128])
            .fill_value::<i32>(-11)
            .attr("version", 14i64)
            .unwrap()
            .write_fill::<i32>()
            .unwrap();
        let f = wf.close().unwrap();

        let ds = f.dataset("data").unwrap();
        assert!(ds.is_chunked().unwrap());
        assert_eq!(ds.attr("version").unwrap().read_scalar_i64(), Some(14));
        let values: Vec<i32> = ds.read().unwrap();
        assert_eq!(values.len(), 1_000);
        assert!(values.iter().all(|&value| value == -11));
        let plist = ds.create_plist().unwrap();
        assert_eq!(plist.fill_value, Some((-11i32).to_le_bytes().to_vec()));
    }
}

#[test]
fn test_dataset_builder_sparse_chunked_explicit_chunks() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir
        .path()
        .join("api_write_sparse_chunked_explicit_chunks.h5");
    let first = [1i32; 128];
    let middle = [5i32; 128];

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("data")
            .shape(&[1_000])
            .chunk(&[128])
            .fill_value::<i32>(-7)
            .deflate(1)
            .write_chunks::<i32, _, _>([([0u64], &first[..]), ([512u64], &middle[..])])
            .unwrap();
        let f = wf.close().unwrap();

        let ds = f.dataset("data").unwrap();
        assert!(ds.is_chunked().unwrap());
        let values: Vec<i32> = ds.read().unwrap();
        assert_eq!(values.len(), 1_000);
        assert_eq!(&values[..128], &first);
        assert_eq!(&values[512..640], &middle);
        assert!(values[128..512].iter().all(|&value| value == -7));
        assert!(values[640..].iter().all(|&value| value == -7));
        assert!(dataset_has_filter_id(&ds, 1).unwrap());
    }
}

#[test]
fn test_writable_file_scalar() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_scalar.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("pi")
            .write_scalar::<f64>(std::f64::consts::PI)
            .unwrap();
        let f = wf.close().unwrap();

        let ds = f.dataset("pi").unwrap();
        let val = dataset_scalar::<f64>(&ds).unwrap();
        assert!((val - std::f64::consts::PI).abs() < 1e-15);
    }
}

#[test]
fn test_dataset_builder_compact_scalar_and_rejected_scalar_options() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_scalar_options.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("compact_scalar")
            .compact()
            .attr("version", 13i64)
            .unwrap()
            .write_scalar::<i32>(99)
            .unwrap();

        let chunked_err = wf
            .new_dataset_builder("bad_chunked_scalar")
            .chunk(&[1])
            .write_scalar::<i32>(1)
            .expect_err("scalar chunked storage should be rejected");
        assert!(chunked_err.to_string().contains("scalar dataset writer"));

        let max_shape_err = wf
            .new_dataset_builder("bad_max_scalar")
            .max_shape(&[1])
            .write_scalar::<i32>(1)
            .expect_err("scalar max dimensions should be rejected");
        assert!(max_shape_err.to_string().contains("max dimensions"));

        let f = wf.close().unwrap();
        let ds = f.dataset("compact_scalar").unwrap();
        assert_eq!(dataset_scalar::<i32>(&ds).unwrap(), 99);
        assert_eq!(ds.attr("version").unwrap().read_scalar_i64(), Some(13));
    }
}

#[test]
fn test_writable_file_compact() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_compact.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("tiny")
            .compact()
            .write::<u8>(&[1, 2, 3, 4, 5])
            .unwrap();
        let f = wf.close().unwrap();

        let ds = f.dataset("tiny").unwrap();
        assert_dataset_values::<u8>(&ds, &[1, 2, 3, 4, 5]).unwrap();
    }
}

#[test]
fn test_dataset_builder_rejects_invalid_compact_options() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_compact_option_errors.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        let chunked_err = wf
            .new_dataset_builder("bad_chunked_compact")
            .compact()
            .chunk(&[1])
            .write::<u8>(&[1, 2, 3])
            .expect_err("compact plus chunked storage should be rejected");
        assert!(chunked_err.to_string().contains("compact dataset writer"));

        let max_shape_err = wf
            .new_dataset_builder("bad_max_compact")
            .compact()
            .max_shape(&[8])
            .write::<u8>(&[1, 2, 3])
            .expect_err("compact plus max dimensions should be rejected");
        assert!(max_shape_err.to_string().contains("max dimensions"));
    }
}

#[test]
fn test_dataset_builder_rejects_oversized_compact_payload() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_compact_too_large.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        let data = vec![0u8; u16::MAX as usize + 1];
        let err = wf
            .new_dataset_builder("too_large")
            .compact()
            .write::<u8>(&data)
            .expect_err("compact payload larger than u16 should be rejected");
        assert!(err.to_string().contains("compact dataset payload"));
    }
}

#[test]
fn test_writable_file_explicit_fill_value_properties() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_fill_value.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("with_fill")
            .fill_properties(1, 2)
            .fill_value::<i32>(-7)
            .write::<i32>(&[1, 2, 3])
            .unwrap();
        let f = wf.close().unwrap();

        let ds = f.dataset("with_fill").unwrap();
        let plist = ds.create_plist().unwrap();
        assert_eq!(plist.fill_alloc_time, Some(1));
        assert_eq!(plist.fill_time, Some(2));
        assert!(plist.fill_value_defined);
        assert_eq!(plist.fill_value, Some((-7i32).to_le_bytes().to_vec()));
        assert_dataset_values::<i32>(&ds, &[1, 2, 3]).unwrap();
    }
}

#[test]
fn test_writable_file_compact_fixed_strings() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_compact_strings.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("names")
            .compact()
            .write_fixed_ascii_strings(&["red", "green", "blue"], 8)
            .unwrap();
        let f = wf.close().unwrap();

        let ds = f.dataset("names").unwrap();
        assert_dataset_strings(&ds, &["red", "green", "blue"]).unwrap();
    }
}

#[test]
fn test_dataset_builder_fixed_string_datasets_with_attrs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_fixed_string_dataset_attrs.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("contiguous")
            .attr("version", 10i64)
            .unwrap()
            .write_fixed_ascii_strings(&["red", "green"], 8)
            .unwrap();
        wf.new_dataset_builder("compact")
            .compact()
            .attr("version", 11i64)
            .unwrap()
            .write_fixed_utf8_strings(&["猫", "å"], 8)
            .unwrap();
        let f = wf.close().unwrap();

        let contiguous = f.dataset("contiguous").unwrap();
        assert_dataset_strings(&contiguous, &["red", "green"]).unwrap();
        assert_eq!(
            contiguous.attr("version").unwrap().read_scalar_i64(),
            Some(10)
        );

        let compact = f.dataset("compact").unwrap();
        assert_dataset_strings(&compact, &["猫", "å"]).unwrap();
        assert_eq!(compact.attr("version").unwrap().read_scalar_i64(), Some(11));
    }
}

#[test]
fn test_dataset_builder_chunked_fixed_string_dataset() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_chunked_fixed_strings.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("names")
            .shape(&[4])
            .chunk(&[2])
            .shuffle()
            .deflate(3)
            .attr("version", 12i64)
            .unwrap()
            .write_fixed_ascii_strings(&["red", "green", "blue", "gold"], 8)
            .unwrap();
        let f = wf.close().unwrap();

        let ds = f.dataset("names").unwrap();
        assert!(ds.is_chunked().unwrap());
        assert_dataset_strings(&ds, &["red", "green", "blue", "gold"]).unwrap();
        assert_eq!(ds.attr("version").unwrap().read_scalar_i64(), Some(12));
    }
}

#[test]
fn test_writable_file_vlen_utf8_strings() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_vlen_strings.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("names")
            .write_vlen_utf8_strings(&["", "猫", "alpha"])
            .unwrap();
        let f = wf.close().unwrap();

        let ds = f.dataset("names").unwrap();
        assert!(ds.dtype().unwrap().is_vlen());
        assert_dataset_strings(&ds, &["", "猫", "alpha"]).unwrap();
    }

    let out = std::process::Command::new("h5dump")
        .arg("-H")
        .arg(&path)
        .output();
    if let Ok(out) = out {
        assert!(
            out.status.success(),
            "h5dump failed on vlen string writer fixture: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    let out = std::process::Command::new("timeout")
        .arg("10")
        .arg("h5dump")
        .arg("-d")
        .arg("names")
        .arg(&path)
        .output();
    if let Ok(out) = out {
        assert!(
            out.status.success(),
            "h5dump -d failed or timed out on vlen string writer fixture: status={:?}, stderr={}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr)
        );
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(stdout.contains("alpha"));
        assert!(stdout.contains("STRSIZE H5T_VARIABLE"));
    }
}

#[test]
fn test_writable_file_chunked_filtered_vlen_utf8_strings() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_chunked_vlen_strings.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("names")
            .shape(&[5])
            .chunk(&[2])
            .shuffle()
            .deflate(3)
            .attr("version", 13i64)
            .unwrap()
            .write_vlen_utf8_strings(&["", "猫", "alpha", "beta", "delta"])
            .unwrap();
        let f = wf.close().unwrap();

        let ds = f.dataset("names").unwrap();
        assert!(ds.dtype().unwrap().is_vlen());
        assert!(ds.is_chunked().unwrap());
        assert!(dataset_has_filter_id(&ds, 1).unwrap());
        assert!(dataset_has_filter_id(&ds, 2).unwrap());
        assert_dataset_strings(&ds, &["", "猫", "alpha", "beta", "delta"]).unwrap();
        assert_eq!(ds.attr("version").unwrap().read_scalar_i64(), Some(13));
    }

    let out = std::process::Command::new("timeout")
        .arg("10")
        .arg("h5dump")
        .arg("-pH")
        .arg("-d")
        .arg("names")
        .arg(&path)
        .output();
    if let Ok(out) = out {
        assert!(
            out.status.success(),
            "h5dump -pH failed or timed out on chunked vlen string fixture: status={:?}, stderr={}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr)
        );
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(stdout.contains("STRSIZE H5T_VARIABLE"));
        assert!(stdout.contains("CHUNKED"));
        assert!(stdout.contains("DEFLATE"));
        assert!(stdout.contains("SHUFFLE"));
    }
}

#[test]
fn test_writable_file_chunked_vlen_utf8_strings_with_fletcher32() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir
        .path()
        .join("api_write_chunked_vlen_strings_fletcher32.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("names")
            .shape(&[3])
            .chunk(&[1])
            .fletcher32()
            .write_vlen_utf8_strings(&["alpha", "", "猫"])
            .unwrap();
        let f = wf.close().unwrap();

        let ds = f.dataset("names").unwrap();
        assert!(ds.dtype().unwrap().is_vlen());
        assert!(ds.is_chunked().unwrap());
        assert!(dataset_has_filter_id(&ds, 3).unwrap());
        assert_dataset_strings(&ds, &["alpha", "", "猫"]).unwrap();
    }
}

#[test]
fn test_writable_file_vlen_utf8_strings_accept_fill_values() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_vlen_string_fill.h5");
    let empty_vlen_descriptor = 0u128.to_le_bytes().to_vec();

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("contiguous")
            .shape(&[2])
            .fill_value::<u128>(0)
            .write_vlen_utf8_strings(&["alpha", "beta"])
            .unwrap();
        wf.new_dataset_builder("chunked")
            .shape(&[3])
            .chunk(&[1])
            .fill_value::<u128>(0)
            .write_vlen_utf8_strings(&["", "猫", "gamma"])
            .unwrap();
        let f = wf.close().unwrap();

        let ds = f.dataset("contiguous").unwrap();
        assert!(ds.dtype().unwrap().is_vlen());
        assert_dataset_strings(&ds, &["alpha", "beta"]).unwrap();
        let plist = ds.create_plist().unwrap();
        assert!(plist.fill_value_defined);
        assert_eq!(plist.fill_value, Some(empty_vlen_descriptor.clone()));

        let ds = f.dataset("chunked").unwrap();
        assert!(ds.dtype().unwrap().is_vlen());
        assert!(ds.is_chunked().unwrap());
        assert_dataset_strings(&ds, &["", "猫", "gamma"]).unwrap();
        let plist = ds.create_plist().unwrap();
        assert!(plist.fill_value_defined);
        assert_eq!(plist.fill_value, Some(empty_vlen_descriptor));
    }
}

#[test]
fn test_writable_file_vlen_utf8_strings_encode_string_fill_value() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_vlen_string_text_fill.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("contiguous")
            .shape(&[2])
            .vlen_utf8_fill_value("fallback")
            .write_vlen_utf8_strings(&["alpha", "beta"])
            .unwrap();
        wf.new_dataset_builder("chunked")
            .shape(&[3])
            .chunk(&[1])
            .vlen_utf8_fill_value("missing")
            .write_vlen_utf8_strings(&["", "猫", "gamma"])
            .unwrap();
        let f = wf.close().unwrap();

        let ds = f.dataset("contiguous").unwrap();
        assert_dataset_strings(&ds, &["alpha", "beta"]).unwrap();
        let plist = ds.create_plist().unwrap();
        let fill = plist
            .fill_value
            .expect("vlen string fill should be encoded");
        assert_eq!(fill.len(), 16);
        assert_eq!(u32::from_le_bytes(fill[..4].try_into().unwrap()), 8);
        assert_ne!(&fill[4..12], &[0; 8]);
        assert_eq!(u32::from_le_bytes(fill[12..16].try_into().unwrap()), 1);

        let ds = f.dataset("chunked").unwrap();
        assert!(ds.is_chunked().unwrap());
        assert_dataset_strings(&ds, &["", "猫", "gamma"]).unwrap();
        let plist = ds.create_plist().unwrap();
        let fill = plist
            .fill_value
            .expect("chunked vlen string fill should be encoded");
        assert_eq!(fill.len(), 16);
        assert_eq!(u32::from_le_bytes(fill[..4].try_into().unwrap()), 7);
        assert_ne!(&fill[4..12], &[0; 8]);
        assert_eq!(u32::from_le_bytes(fill[12..16].try_into().unwrap()), 1);
    }
}

#[test]
fn test_writable_file_vlen_utf8_strings_split_global_heap_collections() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_many_vlen_strings.h5");
    let values: Vec<String> = (0..=u16::MAX).map(|i| format!("kmer-{i}")).collect();
    let refs: Vec<&str> = values.iter().map(String::as_str).collect();

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("kmers")
            .write_vlen_utf8_strings(&refs)
            .unwrap();
        let f = wf.close().unwrap();

        let ds = f.dataset("kmers").unwrap();
        assert!(ds.dtype().unwrap().is_vlen());
        let strings = ds.read_strings().unwrap();
        assert_eq!(strings.len(), refs.len());
        assert_eq!(strings.first().map(String::as_str), Some("kmer-0"));
        assert_eq!(
            strings.get(u16::MAX as usize).map(String::as_str),
            Some("kmer-65535")
        );
    }
}

#[test]
fn test_dataset_builder_vlen_utf8_strings_with_attrs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_vlen_string_attrs.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("names")
            .attr("version", 9i64)
            .unwrap()
            .fixed_utf8_attr("label", "猫", 8)
            .unwrap()
            .write_vlen_utf8_strings(&["", "alpha", "猫"])
            .unwrap();
        let f = wf.close().unwrap();

        let ds = f.dataset("names").unwrap();
        assert!(ds.dtype().unwrap().is_vlen());
        assert_dataset_strings(&ds, &["", "alpha", "猫"]).unwrap();
        assert_eq!(ds.attr("version").unwrap().read_scalar_i64(), Some(9));
        assert_eq!(attribute_string(&ds.attr("label").unwrap()).unwrap(), "猫");
    }
}

#[test]
fn test_writable_file_compact_compound() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_compact_compound.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("points")
            .compact()
            .write::<Point>(&[Point { x: 1.5, label: 10 }, Point { x: 2.5, label: 20 }])
            .unwrap();
        let f = wf.close().unwrap();

        let ds = f.dataset("points").unwrap();
        assert_dataset_field::<f64>(&ds, "x", &[1.5, 2.5]).unwrap();
        assert_dataset_field::<i32>(&ds, "label", &[10, 20]).unwrap();
    }
}

#[test]
fn test_dataset_builder_write_raw_with_explicit_complex_dtype() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_raw_complex_dtype.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        let enum_data: Vec<u8> = [0u16, 1, 2]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect();
        wf.new_dataset_builder("status")
            .write_raw_with_dtype(
                DtypeSpec::Enum {
                    base: Box::new(DtypeSpec::U16),
                    members: vec![
                        ("zero".to_string(), 0),
                        ("one".to_string(), 1),
                        ("two".to_string(), 2),
                    ],
                },
                &enum_data,
            )
            .unwrap();

        wf.new_dataset_builder("opaque")
            .shape(&[2])
            .write_raw_with_dtype(
                DtypeSpec::Opaque {
                    size: 4,
                    tag: "payload".to_string(),
                },
                b"abcdwxyz",
            )
            .unwrap();

        let array_data: Vec<u8> = (0i16..12).flat_map(i16::to_le_bytes).collect();
        wf.new_dataset_builder("cells")
            .shape(&[2])
            .write_raw_with_dtype(
                DtypeSpec::Array {
                    dims: vec![2, 3],
                    base: Box::new(DtypeSpec::I16),
                },
                &array_data,
            )
            .unwrap();

        let nested = DtypeSpec::Compound {
            size: 8,
            fields: vec![
                CompoundFieldSpec {
                    name: "a".to_string(),
                    offset: 0,
                    dtype: DtypeSpec::I32,
                },
                CompoundFieldSpec {
                    name: "b".to_string(),
                    offset: 4,
                    dtype: DtypeSpec::F32,
                },
            ],
        };
        let nested_compound = DtypeSpec::Compound {
            size: 16,
            fields: vec![
                CompoundFieldSpec {
                    name: "id".to_string(),
                    offset: 0,
                    dtype: DtypeSpec::I32,
                },
                CompoundFieldSpec {
                    name: "nested".to_string(),
                    offset: 8,
                    dtype: nested,
                },
            ],
        };
        let mut compound_data = Vec::new();
        for (id, a, b) in [(7i32, 10i32, 1.25f32), (8, 20, 2.5)] {
            compound_data.extend_from_slice(&id.to_le_bytes());
            compound_data.extend_from_slice(&[0; 4]);
            compound_data.extend_from_slice(&a.to_le_bytes());
            compound_data.extend_from_slice(&b.to_le_bytes());
        }
        wf.new_dataset_builder("nested")
            .shape(&[2])
            .write_raw_with_dtype(nested_compound, &compound_data)
            .unwrap();

        let f = wf.close().unwrap();

        let status = f.dataset("status").unwrap();
        assert!(status.dtype().unwrap().is_enum());
        assert_dataset_values::<u16>(&status, &[0, 1, 2]).unwrap();

        let opaque = f.dataset("opaque").unwrap();
        assert_eq!(opaque.dtype().unwrap().opaque_tag_str(), Some("payload"));
        assert_dataset_raw(&opaque, b"abcdwxyz").unwrap();

        let cells = f.dataset("cells").unwrap();
        let cells_dtype = cells.dtype().unwrap();
        let dims = cells_dtype
            .array_dims_iter()
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        let base = cells_dtype.array_base().unwrap();
        assert_eq!(dims, vec![2, 3]);
        assert_eq!(base.size(), 2);
        assert_dataset_raw(&cells, &array_data).unwrap();

        let nested = f.dataset("nested").unwrap();
        assert_eq!(dataset_field_value_count(&nested, "nested").unwrap(), 2);
    }
}

#[test]
fn test_dataset_builder_rejects_unsupported_enum_base_types() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_bad_enum_base.h5");
    let mut wf = WritableFile::create(&path).unwrap();

    let err = wf
        .new_dataset_builder("wide_enum")
        .write_raw_with_dtype(
            DtypeSpec::Enum {
                base: Box::new(DtypeSpec::U128),
                members: vec![("wide".to_string(), 1)],
            },
            &1u128.to_le_bytes(),
        )
        .expect_err("16-byte enum base should be rejected until member values are widened");
    assert!(err
        .to_string()
        .contains("enum writer supports only integer base datatypes up to 8 bytes"));

    let err = wf
        .new_dataset_builder("float_enum")
        .write_raw_with_dtype(
            DtypeSpec::Enum {
                base: Box::new(DtypeSpec::F32),
                members: vec![("bad".to_string(), 1)],
            },
            &1f32.to_le_bytes(),
        )
        .expect_err("non-integer enum base should be rejected");
    assert!(err
        .to_string()
        .contains("enum writer supports only integer base datatypes up to 8 bytes"));
}

#[test]
fn test_dataset_builder_rejects_unencodable_explicit_dtype() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_bad_explicit_dtype.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        let array_err = wf
            .new_dataset_builder("bad_array")
            .write_raw_with_dtype(
                DtypeSpec::Array {
                    dims: vec![1; u8::MAX as usize + 1],
                    base: Box::new(DtypeSpec::U8),
                },
                &[0],
            )
            .expect_err("array datatype rank should fit in the encoded rank byte");
        assert!(array_err.to_string().contains("array datatype rank"));

        let opaque_err = wf
            .new_dataset_builder("bad_opaque")
            .write_raw_with_dtype(
                DtypeSpec::Opaque {
                    size: 1,
                    tag: "x".repeat(u8::MAX as usize),
                },
                &[0],
            )
            .expect_err("opaque tag length should fit in the encoded tag-length byte");
        assert!(opaque_err.to_string().contains("opaque tag"));
    }
}

#[test]
fn test_writable_file_h5dump_interop() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_h5dump.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("values")
            .write::<f64>(&[1.0, 2.0, 3.0])
            .unwrap();
        wf.flush().unwrap();
    }

    let out = std::process::Command::new("h5dump").arg(&path).output();
    if let Ok(out) = out {
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            out.status.success(),
            "h5dump failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(stdout.contains("1, 2, 3"));
    }
}

#[test]
fn test_writable_file_root_attr() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_attr.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.add_attr("version", 42i64).unwrap();
        wf.new_dataset_builder("data")
            .write::<f32>(&[1.0, 2.0])
            .unwrap();
        let f = wf.close().unwrap();

        let attr = f.attr("version").unwrap();
        let val: i64 = attr.read_scalar::<i64>().unwrap();
        assert_eq!(val, 42);
    }
}

#[test]
fn test_writable_file_oversized_object_header_uses_continuation_chunk() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("api_write_header_continuation.h5");
    let payload = vec![7u8; 60_000];
    let attr_len = 60_000;

    {
        let mut wf = WritableFile::create(&path).unwrap();
        let mut builder = wf.new_dataset_builder("compact");
        for idx in 0..8 {
            builder = builder
                .fixed_ascii_attr(&format!("attr_{idx}"), "x", attr_len)
                .unwrap();
        }
        builder.compact().write::<u8>(&payload).unwrap();
        let f = wf.close().unwrap();

        let ds = f.dataset("compact").unwrap();
        assert_dataset_values::<u8>(&ds, &payload).unwrap();
        let expected = [
            "attr_0", "attr_1", "attr_2", "attr_3", "attr_4", "attr_5", "attr_6", "attr_7",
        ];
        let mut found = [false; 8];
        let count = location_attr_summary_into(&ds, &expected, &mut found).unwrap();
        assert_eq!(count, expected.len());
        assert!(found.iter().all(|present| *present));
        assert_eq!(attribute_string(&ds.attr("attr_7").unwrap()).unwrap(), "x");
    }

    let image = std::fs::read(&path).unwrap();
    let continuation_chunks = image.windows(4).filter(|window| *window == b"OCHK").count();
    assert!(
        continuation_chunks >= 2,
        "oversized compact root header should be split into chained v2 continuation chunks, found {continuation_chunks}"
    );
}
