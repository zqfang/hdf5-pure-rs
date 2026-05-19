use std::fs;

use hdf5_pure_rust::engine::writer::{DatasetSpec, DtypeSpec, HdfFileWriter};
use hdf5_pure_rust::{Dataset, File};

fn raw_len(ds: &Dataset) -> usize {
    let nbytes = ds.size().unwrap() as usize * ds.element_size().unwrap() as usize;
    nbytes
}

#[test]
fn test_write_compact_dataset() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("written_compact.h5");

    {
        let f = fs::File::create(&path).unwrap();
        let mut w = HdfFileWriter::new(f);
        w.begin().unwrap();
        w.create_root_group().unwrap();

        let data: Vec<u8> = vec![1u8, 2, 3, 4, 5];

        w.create_compact_dataset(
            "/",
            &DatasetSpec {
                name: "small",
                shape: &[5],
                max_shape: None,
                dtype: DtypeSpec::U8,
                data: &data,
            },
        )
        .unwrap();

        w.finalize().unwrap();
    }

    // Read back with pure-Rust
    {
        let f = File::open(&path).unwrap();
        let ds = f.dataset("small").unwrap();
        assert_eq!(ds.space().unwrap().shape(), &[5]);
        let mut values = [0u8; 5];
        ds.read_into(&mut values).unwrap();
        assert_eq!(values, [1, 2, 3, 4, 5]);
    }

    // Verify with h5dump
    {
        let out = std::process::Command::new("h5dump").arg(&path).output();
        if let Ok(out) = out {
            let stdout = String::from_utf8_lossy(&out.stdout);
            println!("h5dump compact:\n{stdout}");
            assert!(
                out.status.success(),
                "h5dump failed: {}",
                String::from_utf8_lossy(&out.stderr)
            );
            assert!(stdout.contains("1, 2, 3, 4, 5"));
        }
    }
}

#[test]
fn test_engine_writer_rejects_compact_data_length_mismatch() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("written_compact_bad_len.h5");

    let f = fs::File::create(&path).unwrap();
    let mut w = HdfFileWriter::new(f);
    w.begin().unwrap();
    w.create_root_group().unwrap();

    let err = w
        .create_compact_dataset(
            "/",
            &DatasetSpec {
                name: "bad",
                shape: &[2],
                max_shape: None,
                dtype: DtypeSpec::I16,
                data: &[1, 2],
            },
        )
        .expect_err("compact data byte length should match shape * dtype size");
    assert!(err.to_string().contains("dataset byte length"));
}

#[test]
fn test_compact_zero_sized_dataset_read() {
    let f = File::open("tests/data/hdf5_ref/compact_read_cases.h5").unwrap();
    let ds = f.dataset("compact_zero").unwrap();

    assert_eq!(ds.space().unwrap().shape(), &[0]);
    assert_eq!(ds.size().unwrap(), 0);
    assert_eq!(raw_len(&ds), 0);
    let mut raw = [];
    ds.read_raw_into(&mut raw).unwrap();

    let mut vals = Vec::<i32>::new();
    ds.read_into(&mut vals).unwrap();
    assert!(vals.is_empty());
}

#[test]
fn test_compact_scalar_compound_payload_read() {
    let f = File::open("tests/data/hdf5_ref/compact_read_cases.h5").unwrap();
    let ds = f.dataset("compact_compound_scalar").unwrap();

    assert_eq!(ds.space().unwrap().shape(), &[]);
    assert_eq!(ds.size().unwrap(), 1);
    let mut raw = [0u8; 12];
    ds.read_raw_into(&mut raw).unwrap();

    let dtype = ds.dtype().unwrap();
    let fields = dtype
        .compound_fields_iter()
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(fields.len(), 2);
    assert_eq!(fields[0].name, "x");
    assert_eq!(fields[1].name, "label");

    let mut x_vals = vec![0.0; ds.size().unwrap() as usize];
    ds.read_field_into("x", &mut x_vals).unwrap();
    assert_eq!(x_vals, vec![1.5]);

    let mut labels = vec![0; ds.size().unwrap() as usize];
    ds.read_field_into("label", &mut labels).unwrap();
    assert_eq!(labels, vec![7]);
}
