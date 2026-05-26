use std::fs;

use hdf5_pure_rust::engine::writer::{DatasetSpec, DtypeSpec, FillValueSpec, HdfFileWriter};
use hdf5_pure_rust::format::messages::data_layout::ChunkIndexType;
use hdf5_pure_rust::io::reader::UNDEF_ADDR;
use hdf5_pure_rust::{Dataset, File, H5Type, Result};

fn assert_dataset_shape(ds: &Dataset, expected: &[u64]) -> Result<()> {
    let space = ds.space()?;
    assert_eq!(space.shape(), expected);
    Ok(())
}

fn read_dataset_into<T: H5Type + Default>(ds: &Dataset, values: &mut [T]) -> Result<()> {
    ds.read_into(values)
}

fn assert_h5dump_dataset_read(path: &std::path::Path, dataset: &str, context: &str) {
    let out = std::process::Command::new("timeout")
        .arg("10")
        .arg("h5dump")
        .arg("-d")
        .arg(dataset)
        .arg(path)
        .output();
    if let Ok(out) = out {
        assert!(
            out.status.success(),
            "h5dump -d failed or timed out on {context}: status={:?}, stderr={}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

fn assert_h5py_script(path: &std::path::Path, script: &str, context: &str) {
    let out = std::process::Command::new("python3")
        .arg("-c")
        .arg(format!(
            "import sys, h5py\n\
             f = h5py.File(sys.argv[1], 'r')\n\
             {script}\n\
             f.close()\n\
             print('OK')"
        ))
        .arg(path)
        .output();
    if let Ok(out) = out {
        assert!(
            out.status.success() && String::from_utf8_lossy(&out.stdout).contains("OK"),
            "h5py verification failed on {context}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
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

    assert_h5dump_dataset_read(
        &path,
        "chunked",
        "uncompressed fixed-array chunked writer fixture",
    );
    assert_h5py_script(
        &path,
        "d = f['chunked']\n\
         assert d.shape == (100,)\n\
         assert d.chunks == (10,)\n\
         assert d.compression is None\n\
         assert d.dtype.kind == 'f'\n\
         assert d.dtype.itemsize == 4\n\
         assert d[:].tolist() == [float(i) for i in range(100)]",
        "uncompressed fixed-array chunked writer fixture",
    );
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

    assert_h5dump_dataset_read(&path, "single", "single-chunk writer fixture");
    assert_h5py_script(
        &path,
        "d = f['single']\n\
         assert d.shape == (12,)\n\
         assert d.chunks == (16,)\n\
         assert d.compression is None\n\
         assert d[:].tolist() == list(range(12))",
        "single-chunk writer fixture",
    );
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

    assert_h5dump_dataset_read(
        &path,
        "single_deflate",
        "filtered single-chunk writer fixture",
    );
    assert_h5py_script(
        &path,
        "d = f['single_deflate']\n\
         assert d.shape == (64,)\n\
         assert d.chunks == (64,)\n\
         assert d.compression == 'gzip'\n\
         assert d.compression_opts == 6\n\
         assert d.shuffle\n\
         assert d[:].tolist() == [i % 7 for i in range(64)]",
        "filtered single-chunk writer fixture",
    );
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
        let info = ds.info().unwrap();
        assert_eq!(info.layout.version, 4);
        assert_eq!(
            info.layout.chunk_index_type,
            Some(ChunkIndexType::FixedArray)
        );
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

    let out = std::process::Command::new("python3")
        .arg("-c")
        .arg(
            "import sys, h5py\n\
             f = h5py.File(sys.argv[1], 'r')\n\
             d = f['compressed']\n\
             assert d.shape == (100,)\n\
             assert d.chunks == (25,)\n\
             assert d.compression == 'gzip'\n\
             assert d.compression_opts == 6\n\
             assert list(d[:]) == [float(i) for i in range(100)]\n\
             f.close()\n\
             print('OK')",
        )
        .arg(&path)
        .output();
    if let Ok(out) = out {
        assert!(
            out.status.success() && String::from_utf8_lossy(&out.stdout).contains("OK"),
            "h5py: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

#[test]
fn test_write_chunked_fixed_array_beyond_one_page() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("written_chunked_fixed_array_paged.h5");

    {
        let f = fs::File::create(&path).unwrap();
        let mut w = HdfFileWriter::new(f);
        w.begin().unwrap();
        w.create_root_group().unwrap();

        let data: Vec<u8> = (0..4_105).map(|i| (i % 251) as u8).collect();

        w.create_chunked_dataset(
            "/",
            &DatasetSpec {
                name: "paged_fixed_array",
                shape: &[4_105],
                max_shape: None,
                dtype: DtypeSpec::U8,
                data: &data,
            },
            &[1],
            None,
            false,
        )
        .unwrap();

        w.finalize().unwrap();
    }

    let f = File::open(&path).unwrap();
    let ds = f.dataset("paged_fixed_array").unwrap();
    let info = ds.info().unwrap();
    assert_eq!(info.layout.version, 4);
    assert_eq!(
        info.layout.chunk_index_type,
        Some(ChunkIndexType::FixedArray)
    );

    let mut values = vec![0u8; ds.size().unwrap() as usize];
    read_dataset_into(&ds, &mut values).unwrap();
    let expected: Vec<u8> = (0..4_105).map(|i| (i % 251) as u8).collect();
    assert_eq!(values, expected);

    assert_h5dump_dataset_read(
        &path,
        "paged_fixed_array",
        "direct writer paged fixed-array partial final page fixture",
    );
    assert_h5py_script(
        &path,
        "d = f['paged_fixed_array']\n\
         assert d.shape == (4105,)\n\
         assert d.chunks == (1,)\n\
         assert d.compression is None\n\
         assert d[:5].tolist() == [0, 1, 2, 3, 4]\n\
         assert d[4094:4099].tolist() == [78, 79, 80, 81, 82]\n\
         assert d[-3:].tolist() == [86, 87, 88]",
        "direct writer paged fixed-array partial final page fixture",
    );
}

#[test]
fn test_write_chunked_max_shape_uses_extensible_array_index() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("written_chunked_max_shape_ea.h5");

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
                name: "max_shape_ea",
                shape: &[12],
                max_shape: Some(&[u64::MAX]),
                dtype: DtypeSpec::I32,
                data: &data_bytes,
            },
            &[3],
            None,
            false,
        )
        .unwrap();

        w.finalize().unwrap();
    }

    let f = File::open(&path).unwrap();
    let ds = f.dataset("max_shape_ea").unwrap();
    let info = ds.info().unwrap();
    assert_eq!(info.layout.version, 4);
    assert_eq!(
        info.layout.chunk_index_type,
        Some(ChunkIndexType::ExtensibleArray)
    );

    let mut values = vec![0i32; ds.size().unwrap() as usize];
    read_dataset_into(&ds, &mut values).unwrap();
    assert_eq!(values, (0..12).collect::<Vec<_>>());

    assert_h5dump_dataset_read(
        &path,
        "max_shape_ea",
        "extensible-array max-shape chunked writer fixture",
    );
    let out = std::process::Command::new("python3")
        .arg("-c")
        .arg(
            "import sys, h5py\n\
             f = h5py.File(sys.argv[1], 'r')\n\
             d = f['max_shape_ea']\n\
             assert d.shape == (12,)\n\
             assert d.chunks == (3,)\n\
             assert d.maxshape == (None,)\n\
             assert d.dtype.kind == 'i'\n\
             assert d.dtype.itemsize == 4\n\
             assert list(d[:]) == list(range(12))\n\
             f.close()\n\
             print('OK')",
        )
        .arg(&path)
        .output();
    if let Ok(out) = out {
        assert!(
            out.status.success() && String::from_utf8_lossy(&out.stdout).contains("OK"),
            "h5py: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

#[test]
fn test_write_extensible_array_partial_edge_chunk_h5py_read() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("written_ea_partial_edge_chunk.h5");

    {
        let f = fs::File::create(&path).unwrap();
        let mut w = HdfFileWriter::new(f);
        w.begin().unwrap();
        w.create_root_group().unwrap();

        let data: Vec<i32> = (0..10).map(|value| value * 3 - 5).collect();
        let data_bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();

        w.create_chunked_dataset(
            "/",
            &DatasetSpec {
                name: "ea_partial_edge",
                shape: &[10],
                max_shape: Some(&[u64::MAX]),
                dtype: DtypeSpec::I32,
                data: &data_bytes,
            },
            &[4],
            None,
            false,
        )
        .unwrap();

        w.finalize().unwrap();
    }

    let f = File::open(&path).unwrap();
    let ds = f.dataset("ea_partial_edge").unwrap();
    let info = ds.info().unwrap();
    assert_eq!(info.layout.version, 4);
    assert_eq!(
        info.layout.chunk_index_type,
        Some(ChunkIndexType::ExtensibleArray)
    );

    let expected: Vec<i32> = (0..10).map(|value| value * 3 - 5).collect();
    let mut values = vec![0i32; ds.size().unwrap() as usize];
    read_dataset_into(&ds, &mut values).unwrap();
    assert_eq!(values, expected);

    assert_h5dump_dataset_read(
        &path,
        "ea_partial_edge",
        "extensible-array partial edge chunk file",
    );
    let out = std::process::Command::new("python3")
        .arg("-c")
        .arg(
            "import sys, h5py\n\
             f = h5py.File(sys.argv[1], 'r')\n\
             d = f['ea_partial_edge']\n\
             assert d.shape == (10,)\n\
             assert d.chunks == (4,)\n\
             assert d.maxshape == (None,)\n\
             assert d.dtype.kind == 'i'\n\
             assert d.dtype.itemsize == 4\n\
             assert list(d[:]) == [-5, -2, 1, 4, 7, 10, 13, 16, 19, 22]\n\
             f.close()\n\
             print('OK')",
        )
        .arg(&path)
        .output();
    if let Ok(out) = out {
        assert!(
            out.status.success() && String::from_utf8_lossy(&out.stdout).contains("OK"),
            "h5py: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

#[test]
fn test_write_chunked_large_max_shape_uses_extensible_array_index() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir
        .path()
        .join("written_chunked_max_shape_large_boundary.h5");

    let f = fs::File::create(&path).unwrap();
    let mut w = HdfFileWriter::new(f);
    w.begin().unwrap();
    w.create_root_group().unwrap();

    let data: Vec<u8> = (0..=255).collect();
    w.create_chunked_dataset(
        "/",
        &DatasetSpec {
            name: "large_ea_grid",
            shape: &[256],
            max_shape: Some(&[u64::MAX]),
            dtype: DtypeSpec::U8,
            data: &data,
        },
        &[1],
        None,
        false,
    )
    .unwrap();
    w.finalize().unwrap();

    let f = File::open(&path).unwrap();
    let ds = f.dataset("large_ea_grid").unwrap();
    let info = ds.info().unwrap();
    assert_eq!(info.layout.version, 4);
    assert_eq!(
        info.layout.chunk_index_type,
        Some(ChunkIndexType::ExtensibleArray)
    );

    let mut values = vec![0u8; ds.size().unwrap() as usize];
    read_dataset_into(&ds, &mut values).unwrap();
    assert_eq!(values, data);

    assert_h5dump_dataset_read(
        &path,
        "large_ea_grid",
        "large extensible-array chunk-1 writer fixture",
    );
    let out = std::process::Command::new("python3")
        .arg("-c")
        .arg(
            "import sys, h5py\n\
             f = h5py.File(sys.argv[1], 'r')\n\
             d = f['large_ea_grid']\n\
             assert d.shape == (256,)\n\
             assert d.chunks == (1,)\n\
             assert d.maxshape == (None,)\n\
             assert d.dtype.kind == 'u'\n\
             assert d.dtype.itemsize == 1\n\
             assert d[0] == 0\n\
             assert d[255] == 255\n\
             assert list(d[:]) == list(range(256))\n\
             f.close()\n\
             print('OK')",
        )
        .arg(&path)
        .output();
    if let Ok(out) = out {
        assert!(
            out.status.success() && String::from_utf8_lossy(&out.stdout).contains("OK"),
            "h5py: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

#[test]
fn test_write_chunked_finite_max_shape_uses_btree_v2_index() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir
        .path()
        .join("written_chunked_finite_max_shape_btree_v2.h5");

    {
        let f = fs::File::create(&path).unwrap();
        let mut w = HdfFileWriter::new(f);
        w.begin().unwrap();
        w.create_root_group().unwrap();

        let data: Vec<i32> = (0..12).map(|value| value * 7 - 3).collect();
        let data_bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();

        w.create_chunked_dataset(
            "/",
            &DatasetSpec {
                name: "finite_max_shape",
                shape: &[12],
                max_shape: Some(&[24]),
                dtype: DtypeSpec::I32,
                data: &data_bytes,
            },
            &[3],
            None,
            false,
        )
        .unwrap();

        w.finalize().unwrap();
    }

    let f = File::open(&path).unwrap();
    let ds = f.dataset("finite_max_shape").unwrap();
    let info = ds.info().unwrap();
    assert_eq!(info.layout.version, 4);
    assert_eq!(info.layout.chunk_index_type, Some(ChunkIndexType::BTreeV2));

    let mut values = vec![0i32; ds.size().unwrap() as usize];
    read_dataset_into(&ds, &mut values).unwrap();
    assert_eq!(
        values,
        (0..12).map(|value| value * 7 - 3).collect::<Vec<_>>()
    );

    assert_h5dump_dataset_read(&path, "finite_max_shape", "finite max-shape B-tree v2 file");
    assert_h5py_script(
        &path,
        "d = f['finite_max_shape']\n\
         assert d.shape == (12,)\n\
         assert d.chunks == (3,)\n\
         assert d.maxshape == (24,)\n\
         assert d.dtype.kind == 'i'\n\
         assert d.dtype.itemsize == 4\n\
         assert d.compression is None\n\
         assert d[:].tolist() == [-3, 4, 11, 18, 25, 32, 39, 46, 53, 60, 67, 74]",
        "finite max-shape B-tree v2 file",
    );
}

#[test]
fn test_write_sparse_fill_only_max_shape_uses_btree_v2_undefined_index() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir
        .path()
        .join("written_sparse_fill_only_max_shape_btree_v2.h5");
    let fill = (-6i16).to_le_bytes();

    {
        let f = fs::File::create(&path).unwrap();
        let mut w = HdfFileWriter::new(f);
        w.begin().unwrap();
        w.create_root_group().unwrap();

        w.create_sparse_chunked_dataset_with_attrs_and_fill(
            "/",
            &DatasetSpec {
                name: "fill_only_btree_v2",
                shape: &[12],
                max_shape: Some(&[24]),
                dtype: DtypeSpec::I16,
                data: &[],
            },
            &[4],
            None,
            false,
            false,
            Some(FillValueSpec::with_value(1, 2, &fill)),
            &[],
        )
        .unwrap();

        w.finalize().unwrap();
    }

    let f = File::open(&path).unwrap();
    let ds = f.dataset("fill_only_btree_v2").unwrap();
    let info = ds.info().unwrap();
    assert_eq!(info.layout.version, 4);
    assert_eq!(info.layout.chunk_index_type, Some(ChunkIndexType::BTreeV2));
    assert_eq!(info.layout.chunk_index_addr, Some(UNDEF_ADDR));

    let mut values = vec![0i16; ds.size().unwrap() as usize];
    read_dataset_into(&ds, &mut values).unwrap();
    assert_eq!(values, vec![-6; 12]);

    assert_h5dump_dataset_read(
        &path,
        "fill_only_btree_v2",
        "fill-only max-shape B-tree v2 undefined-index writer fixture",
    );
    assert_h5py_script(
        &path,
        "d = f['fill_only_btree_v2']\n\
         assert d.shape == (12,)\n\
         assert d.maxshape == (24,)\n\
         assert d.chunks == (4,)\n\
         assert d.dtype.kind == 'i'\n\
         assert d.dtype.itemsize == 2\n\
         assert d.fillvalue == -6\n\
         assert d[:].tolist() == [-6] * 12",
        "fill-only max-shape B-tree v2 undefined-index writer fixture",
    );
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

    assert_h5dump_dataset_read(&path, "shuf_def", "shuffle+deflate chunked file");
    assert_h5py_script(
        &path,
        "d = f['shuf_def']\n\
         assert d.shape == (50,)\n\
         assert d.chunks == (10,)\n\
         assert d.compression == 'gzip'\n\
         assert d.compression_opts == 4\n\
         assert d.shuffle\n\
         assert d[:].tolist() == list(range(50))",
        "shuffle+deflate chunked writer fixture",
    );
}
