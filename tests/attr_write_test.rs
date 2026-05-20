use std::fs;

use hdf5_pure_rust::engine::writer::{AttrSpec, DatasetSpec, DtypeSpec, HdfFileWriter};
use hdf5_pure_rust::{Dataset, File, Location, Result};

fn dataset_attr_names(ds: &Dataset) -> Result<Vec<String>> {
    let mut names = Vec::new();
    ds.visit_attr_names(|name| {
        names.push(name.to_string());
        Ok(())
    })?;
    Ok(names)
}

fn location_attr_names<L: Location>(location: &L) -> Result<Vec<String>> {
    let mut names = Vec::new();
    location.visit_attr_names(|name| {
        names.push(name.to_string());
        Ok(())
    })?;
    Ok(names)
}

#[test]
fn test_write_dataset_with_attrs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("written_with_attrs.h5");

    {
        let f = fs::File::create(&path).unwrap();
        let mut w = HdfFileWriter::new(f);
        w.begin().unwrap();
        w.create_root_group().unwrap();

        let data: Vec<u8> = vec![1.0f64, 2.0, 3.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let attr_data = 42i64.to_le_bytes().to_vec();

        w.create_dataset_with_attrs(
            "/",
            &DatasetSpec {
                name: "data",
                shape: &[3],
                max_shape: None,
                dtype: DtypeSpec::F64,
                data: &data,
            },
            &[AttrSpec {
                name: "count",
                shape: &[],
                dtype: DtypeSpec::I64,
                data: &attr_data,
            }],
        )
        .unwrap();

        w.finalize().unwrap();
    }

    {
        let f = File::open(&path).unwrap();
        let ds = f.dataset("data").unwrap();

        // Read dataset
        let mut values = vec![0.0; ds.size().unwrap() as usize];
        ds.read_into(&mut values).unwrap();
        assert_eq!(values, vec![1.0, 2.0, 3.0]);

        // Read attribute
        let attr_names = dataset_attr_names(&ds).unwrap();
        assert!(attr_names.contains(&"count".to_string()));

        let attr = ds.attr("count").unwrap();
        let mut value = 0i64;
        attr.read_into(std::slice::from_mut(&mut value)).unwrap();
        assert_eq!(value, 42);
    }

    // Verify with h5dump
    {
        let out = std::process::Command::new("h5dump").arg(&path).output();
        if let Ok(out) = out {
            let stdout = String::from_utf8_lossy(&out.stdout);
            println!("h5dump:\n{stdout}");
            assert!(out.status.success());
            assert!(stdout.contains("count"));
        }
    }
}

#[test]
fn test_write_dataset_rejects_attribute_data_length_mismatch() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("attr_bad_len.h5");

    let f = fs::File::create(&path).unwrap();
    let mut w = HdfFileWriter::new(f);
    w.begin().unwrap();
    w.create_root_group().unwrap();

    let data = 1i32.to_le_bytes().to_vec();
    let attr_data = 7i32.to_le_bytes().to_vec();
    let err = w
        .create_dataset_with_attrs(
            "/",
            &DatasetSpec {
                name: "data",
                shape: &[1],
                max_shape: None,
                dtype: DtypeSpec::I32,
                data: &data,
            },
            &[AttrSpec {
                name: "bad_attr",
                shape: &[2],
                dtype: DtypeSpec::I32,
                data: &attr_data,
            }],
        )
        .expect_err("attribute byte length should match shape * dtype size");
    assert!(err.to_string().contains("attribute byte length"));
}

#[test]
fn test_write_root_attrs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("written_root_attrs.h5");

    {
        let f = fs::File::create(&path).unwrap();
        let mut w = HdfFileWriter::new(f);
        w.begin().unwrap();
        w.create_root_group().unwrap();

        let val_data = 3.14f64.to_le_bytes().to_vec();
        w.add_root_attr(&AttrSpec {
            name: "pi",
            shape: &[],
            dtype: DtypeSpec::F64,
            data: &val_data,
        })
        .unwrap();

        w.finalize().unwrap();
    }

    {
        let f = File::open(&path).unwrap();
        let attr_names = location_attr_names(&f).unwrap();
        assert!(attr_names.contains(&"pi".to_string()));

        let attr = f.attr("pi").unwrap();
        let mut val = 0.0f64;
        attr.read_into(std::slice::from_mut(&mut val)).unwrap();
        assert!((val - 3.14).abs() < 1e-10);
    }
}

#[test]
fn test_read_false_true_enum_attr_with_u64_base() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("written_bool_enum_u64_attr.h5");

    {
        let f = fs::File::create(&path).unwrap();
        let mut w = HdfFileWriter::new(f);
        w.begin().unwrap();
        w.create_root_group().unwrap();

        let value = 1u64.to_le_bytes();
        w.add_root_attr(&AttrSpec {
            name: "ordered",
            shape: &[],
            dtype: DtypeSpec::Enum {
                base: Box::new(DtypeSpec::U64),
                members: vec![("FALSE".to_string(), 0), ("TRUE".to_string(), 1)],
            },
            data: &value,
        })
        .unwrap();

        w.finalize().unwrap();
    }

    {
        let f = File::open(&path).unwrap();
        let attr = f.attr("ordered").unwrap();

        assert!(attr.dtype().is_enum());
        assert_eq!(attr.element_size(), 8);
        assert_eq!(attr.read_scalar::<u8>().unwrap(), 1);
        assert!(attr.read_scalar_bool().unwrap());
    }
}

#[test]
fn test_write_dataset_with_many_compact_attrs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("written_many_attrs.h5");

    {
        let f = fs::File::create(&path).unwrap();
        let mut w = HdfFileWriter::new(f);
        w.begin().unwrap();
        w.create_root_group().unwrap();

        let data = 123i32.to_le_bytes().to_vec();
        let mut attr_names = Vec::new();
        let mut attr_payloads = Vec::new();
        for idx in 0..40 {
            attr_names.push(format!("attr_{idx:02}"));
            attr_payloads.push((idx as i64).to_le_bytes().to_vec());
        }
        let attrs: Vec<AttrSpec<'_>> = attr_names
            .iter()
            .zip(&attr_payloads)
            .map(|(name, payload)| AttrSpec {
                name,
                shape: &[],
                dtype: DtypeSpec::I64,
                data: payload,
            })
            .collect();

        w.create_dataset_with_attrs(
            "/",
            &DatasetSpec {
                name: "data",
                shape: &[1],
                max_shape: None,
                dtype: DtypeSpec::I32,
                data: &data,
            },
            &attrs,
        )
        .unwrap();

        w.finalize().unwrap();
    }

    {
        let f = File::open(&path).unwrap();
        let ds = f.dataset("data").unwrap();
        let names = dataset_attr_names(&ds).unwrap();
        assert_eq!(names.len(), 40);
        assert!(names.contains(&"attr_00".to_string()));
        assert!(names.contains(&"attr_39".to_string()));
        let mut value = 0i64;
        ds.attr("attr_37")
            .unwrap()
            .read_into(std::slice::from_mut(&mut value))
            .unwrap();
        assert_eq!(value, 37);
    }

    let out = std::process::Command::new("h5dump")
        .arg("-H")
        .arg(&path)
        .output();
    if let Ok(out) = out {
        assert!(
            out.status.success(),
            "h5dump failed on many-attribute writer fixture: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}
