use std::fs;

use hdf5_pure_rust::engine::writer::{DatasetSpec, DtypeSpec, HdfFileWriter};
use hdf5_pure_rust::format::messages::data_layout::ChunkIndexType;
use hdf5_pure_rust::{Dataset, File, H5Type, Result};

fn assert_dataset_shape(ds: &Dataset, expected: &[u64]) -> Result<()> {
    let space = ds.space()?;
    assert_eq!(space.shape(), expected);
    Ok(())
}

fn read_dataset_into<T: H5Type + Default>(ds: &Dataset, values: &mut [T]) -> Result<()> {
    ds.read_into(values)
}

#[test]
fn test_write_chunked_no_compression() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("written_chunked_nocomp.h5");

    {
        let f = fs::File::create(&path).unwrap();
        let mut w = HdfFileWriter::new(f);
        w.begin().unwrap();
        w.create_root_group().unwrap();

        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let data_bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();

        w.create_chunked_dataset(
            "/",
            &DatasetSpec {
                name: "chunked",
                shape: &[100],
                max_shape: None,
                dtype: DtypeSpec::F32,
                data: &data_bytes,
            },
            &[10], // chunk dims
            None,  // no compression
            false, // no shuffle
        )
        .unwrap();

        w.finalize().unwrap();
    }

    // Read back
    {
        let f = File::open(&path).unwrap();
        let ds = f.dataset("chunked").unwrap();
        assert_dataset_shape(&ds, &[100]).unwrap();

        let mut values = vec![0.0f32; ds.size().unwrap() as usize];
        read_dataset_into(&ds, &mut values).unwrap();
        assert_eq!(values.len(), 100);
        for (i, v) in values.iter().enumerate() {
            assert_eq!(*v, i as f32, "mismatch at index {i}");
        }
    }
}

#[test]
fn test_write_single_chunk_uses_v4_single_chunk_index() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("written_single_chunk.h5");

    {
        let f = fs::File::create(&path).unwrap();
        let mut w = HdfFileWriter::new(f);
        w.begin().unwrap();
        w.create_root_group().unwrap();

        let data: Vec<i32> = (0..12).collect();
        let data_bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();

        w.create_chunked_dataset(
            "/",
            &DatasetSpec {
                name: "single",
                shape: &[12],
                max_shape: None,
                dtype: DtypeSpec::I32,
                data: &data_bytes,
            },
            &[16],
            None,
            false,
        )
        .unwrap();

        w.finalize().unwrap();
    }

    let f = File::open(&path).unwrap();
    let ds = f.dataset("single").unwrap();
    let info = ds.info().unwrap();
    assert_eq!(info.layout.version, 4);
    assert_eq!(
        info.layout.chunk_index_type,
        Some(ChunkIndexType::SingleChunk)
    );

    let mut values = vec![0i32; ds.size().unwrap() as usize];
    read_dataset_into(&ds, &mut values).unwrap();
    assert_eq!(values, (0..12).collect::<Vec<_>>());
}

#[test]
fn test_write_filtered_single_chunk_uses_v4_single_chunk_index() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("written_filtered_single_chunk.h5");

    {
        let f = fs::File::create(&path).unwrap();
        let mut w = HdfFileWriter::new(f);
        w.begin().unwrap();
        w.create_root_group().unwrap();

        let data: Vec<i32> = (0..64).map(|i| i % 7).collect();
        let data_bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();

        w.create_chunked_dataset(
            "/",
            &DatasetSpec {
                name: "single_deflate",
                shape: &[64],
                max_shape: None,
                dtype: DtypeSpec::I32,
                data: &data_bytes,
            },
            &[64],
            Some(6),
            true,
        )
        .unwrap();

        w.finalize().unwrap();
    }

    let f = File::open(&path).unwrap();
    let ds = f.dataset("single_deflate").unwrap();
    let info = ds.info().unwrap();
    assert_eq!(info.layout.version, 4);
    assert_eq!(
        info.layout.chunk_index_type,
        Some(ChunkIndexType::SingleChunk)
    );
    assert!(info.layout.single_chunk_filtered_size.is_some());
    assert_eq!(info.layout.single_chunk_filter_mask, Some(0));

    let mut values = vec![0i32; ds.size().unwrap() as usize];
    read_dataset_into(&ds, &mut values).unwrap();
    assert_eq!(values, (0..64).map(|i| i % 7).collect::<Vec<_>>());
}

#[test]
fn test_engine_writer_rejects_chunked_data_length_mismatch() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("written_chunked_bad_len.h5");

    let f = fs::File::create(&path).unwrap();
    let mut w = HdfFileWriter::new(f);
    w.begin().unwrap();
    w.create_root_group().unwrap();

    let err = w
        .create_chunked_dataset(
            "/",
            &DatasetSpec {
                name: "bad",
                shape: &[4],
                max_shape: None,
                dtype: DtypeSpec::I32,
                data: &[1, 2, 3, 4],
            },
            &[2],
            None,
            false,
        )
        .expect_err("chunked data byte length should match shape * dtype size");
    assert!(err.to_string().contains("dataset byte length"));
}

#[test]
fn test_write_chunked_with_deflate() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("written_chunked_deflate.h5");

    {
        let f = fs::File::create(&path).unwrap();
        let mut w = HdfFileWriter::new(f);
        w.begin().unwrap();
        w.create_root_group().unwrap();

        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let data_bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();

        w.create_chunked_dataset(
            "/",
            &DatasetSpec {
                name: "compressed",
                shape: &[100],
                max_shape: None,
                dtype: DtypeSpec::F32,
                data: &data_bytes,
            },
            &[25],   // chunk dims
            Some(6), // deflate level 6
            false,   // no shuffle
        )
        .unwrap();

        w.finalize().unwrap();
    }

    // Read back with pure-Rust
    {
        let f = File::open(&path).unwrap();
        let ds = f.dataset("compressed").unwrap();
        let mut values = vec![0.0f32; ds.size().unwrap() as usize];
        read_dataset_into(&ds, &mut values).unwrap();
        assert_eq!(values.len(), 100);
        for (i, v) in values.iter().enumerate() {
            assert_eq!(*v, i as f32, "mismatch at index {i}");
        }
    }

    // Verify with h5dump
    {
        let out = std::process::Command::new("h5dump")
            .arg("-d")
            .arg("compressed")
            .arg(&path)
            .output();
        if let Ok(out) = out {
            let stdout = String::from_utf8_lossy(&out.stdout);
            println!("h5dump output:\n{stdout}");
            assert!(
                out.status.success(),
                "h5dump failed: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
    }
}

#[test]
fn test_write_chunked_with_shuffle_and_deflate() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("written_chunked_shuf_def.h5");

    {
        let f = fs::File::create(&path).unwrap();
        let mut w = HdfFileWriter::new(f);
        w.begin().unwrap();
        w.create_root_group().unwrap();

        let data: Vec<i32> = (0..50).collect();
        let data_bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();

        w.create_chunked_dataset(
            "/",
            &DatasetSpec {
                name: "shuf_def",
                shape: &[50],
                max_shape: None,
                dtype: DtypeSpec::I32,
                data: &data_bytes,
            },
            &[10],
            Some(4), // deflate level 4
            true,    // shuffle
        )
        .unwrap();

        w.finalize().unwrap();
    }

    // Read back
    {
        let f = File::open(&path).unwrap();
        let ds = f.dataset("shuf_def").unwrap();
        let mut values = vec![0i32; ds.size().unwrap() as usize];
        read_dataset_into(&ds, &mut values).unwrap();
        assert_eq!(values.len(), 50);
        for (i, v) in values.iter().enumerate() {
            assert_eq!(*v, i as i32, "mismatch at {i}");
        }
    }

    // h5dump verification (structure only, data may differ in edge cases)
    {
        let out = std::process::Command::new("h5dump")
            .arg("-pH") // properties + header only, no data
            .arg(&path)
            .output();
        if let Ok(out) = out {
            let stdout = String::from_utf8_lossy(&out.stdout);
            println!("h5dump shuffle+deflate structure:\n{stdout}");
            assert!(stdout.contains("SHUFFLE"), "should detect shuffle filter");
            assert!(stdout.contains("DEFLATE"), "should detect deflate filter");
        }
    }
}
