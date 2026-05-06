use hdf5_pure_rust::{File, MutableFile, WritableFile};

#[test]
fn test_resize_chunked_dataset() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("resize_test.h5");

    // Create a chunked dataset with unlimited max dims
    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("data")
            .shape(&[10])
            .chunk(&[5])
            .resizable()
            .write::<f64>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
            .unwrap();
        wf.flush().unwrap();
    }

    // Verify initial shape
    {
        let f = File::open(&path).unwrap();
        let ds = f.dataset("data").unwrap();
        assert_eq!(ds.shape().unwrap(), vec![10]);
        let vals: Vec<f64> = ds.read::<f64>().unwrap();
        assert_eq!(vals.len(), 10);
        assert_eq!(vals[0], 1.0);
        assert_eq!(vals[9], 10.0);
    }

    // Resize to smaller (shrink)
    {
        let mut mf = MutableFile::open_rw(&path).unwrap();
        mf.resize_dataset("data", &[7]).unwrap();
    }

    // Verify shrunk shape
    {
        let f = File::open(&path).unwrap();
        let ds = f.dataset("data").unwrap();
        assert_eq!(ds.shape().unwrap(), vec![7]);
        let vals: Vec<f64> = ds.read::<f64>().unwrap();
        assert_eq!(vals.len(), 7);
        assert_eq!(vals[0], 1.0);
        assert_eq!(vals[6], 7.0);
    }

    // Resize to larger (extend -- new region reads as zeros)
    {
        let mut mf = MutableFile::open_rw(&path).unwrap();
        mf.resize_dataset("data", &[15]).unwrap();
    }

    // Verify extended shape
    {
        let f = File::open(&path).unwrap();
        let ds = f.dataset("data").unwrap();
        assert_eq!(ds.shape().unwrap(), vec![15]);
        let vals: Vec<f64> = ds.read::<f64>().unwrap();
        assert_eq!(vals.len(), 15);
        assert_eq!(vals[0], 1.0);
        // Original data preserved in existing chunks
        assert_eq!(vals[4], 5.0);
    }
}

#[test]
fn test_resize_respects_writer_max_shape() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("resize_max_shape.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("data")
            .shape(&[4])
            .max_shape(&[6])
            .chunk(&[2])
            .write::<i32>(&[1, 2, 3, 4])
            .unwrap();
        wf.flush().unwrap();
    }

    {
        let f = File::open(&path).unwrap();
        let ds = f.dataset("data").unwrap();
        assert_eq!(ds.space().unwrap().maxdims().unwrap(), &[6]);
    }

    {
        let mut mf = MutableFile::open_rw(&path).unwrap();
        mf.resize_dataset("data", &[6]).unwrap();
        let err = mf
            .resize_dataset("data", &[7])
            .expect_err("resize beyond writer max_shape should fail");
        assert!(err.to_string().contains("exceeds max 6"));
    }
}

#[test]
fn test_mutable_file_deletes_compact_attributes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("delete_compact_attrs.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.add_attr("root_keep", 1i32).unwrap();
        wf.add_attr("root_drop", 2i32).unwrap();
        let mut group = wf.create_group("metadata").unwrap();
        group.add_attr("group_keep", 3i32).unwrap();
        group.add_attr("group_drop", 4i32).unwrap();
        group
            .new_dataset_builder("values")
            .attr("dataset_keep", 5i32)
            .unwrap()
            .attr("dataset_drop", 6i32)
            .unwrap()
            .write::<i32>(&[10, 20])
            .unwrap();
        wf.flush().unwrap();
    }

    {
        let mut mf = MutableFile::open_rw(&path).unwrap();
        mf.delete_root_attr("root_drop").unwrap();
        mf.delete_group_attr("metadata", "group_drop").unwrap();
        mf.delete_dataset_attr("metadata/values", "dataset_drop")
            .unwrap();
        let err = mf
            .delete_root_attr("missing")
            .expect_err("missing compact attribute should fail");
        assert!(err.to_string().contains("attribute 'missing' not found"));
    }

    {
        let f = File::open(&path).unwrap();
        assert!(f.attr_exists("root_keep").unwrap());
        assert!(!f.attr_exists("root_drop").unwrap());
        let group = f.group("metadata").unwrap();
        assert!(group.attr_exists("group_keep").unwrap());
        assert!(!group.attr_exists("group_drop").unwrap());
        let ds = f.dataset("metadata/values").unwrap();
        assert!(ds.attr_exists("dataset_keep").unwrap());
        assert!(!ds.attr_exists("dataset_drop").unwrap());
        assert_eq!(ds.read::<i32>().unwrap(), vec![10, 20]);
    }
}

#[test]
fn test_mutable_file_deletes_dense_attribute() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("delete_dense_attr.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        for idx in 0..12 {
            wf.add_attr(&format!("attr_{idx:02}"), idx as i32).unwrap();
        }
        wf.flush().unwrap();
    }

    let mut mf = MutableFile::open_rw(&path).unwrap();
    mf.delete_root_attr("attr_03").unwrap();
    drop(mf);

    let f = File::open(&path).unwrap();
    assert!(!f.attr_exists("attr_03").unwrap());
    assert_eq!(f.attr("attr_04").unwrap().read_scalar::<i32>().unwrap(), 4);
}

#[test]
fn test_mutable_file_renames_dense_attribute_same_length() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rename_dense_attr.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        for idx in 0..12 {
            wf.add_attr(&format!("attr_{idx:02}"), idx as i32).unwrap();
        }
        wf.flush().unwrap();
    }

    {
        let mut mf = MutableFile::open_rw(&path).unwrap();
        mf.rename_root_attr("attr_03", "renm_03").unwrap();
    }

    let f = File::open(&path).unwrap();
    assert!(!f.attr_exists("attr_03").unwrap());
    assert!(f.attr_exists("renm_03").unwrap());
    assert_eq!(f.attr("renm_03").unwrap().read_scalar::<i32>().unwrap(), 3);
}

#[test]
fn test_mutable_file_mutates_group_and_dataset_dense_attributes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mutate_nested_dense_attrs.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        let mut group = wf.create_group("metadata").unwrap();
        for idx in 0..12 {
            group
                .add_attr(&format!("gattr_{idx:02}"), idx as i32)
                .unwrap();
        }
        let mut builder = group.new_dataset_builder("values");
        for idx in 0..12 {
            builder = builder
                .attr(&format!("dattr_{idx:02}"), (idx as i32) * 10)
                .unwrap();
        }
        builder.write::<i32>(&[1, 2, 3]).unwrap();
        wf.flush().unwrap();
    }

    {
        let mut mf = MutableFile::open_rw(&path).unwrap();
        mf.delete_group_attr("metadata", "gattr_03").unwrap();
        mf.rename_group_attr("metadata", "gattr_04", "grnam_04")
            .unwrap();
        mf.delete_dataset_attr("metadata/values", "dattr_03")
            .unwrap();
        mf.rename_dataset_attr("metadata/values", "dattr_04", "drnam_04")
            .unwrap();
    }

    let f = File::open(&path).unwrap();
    let group = f.group("metadata").unwrap();
    assert!(!group.attr_exists("gattr_03").unwrap());
    assert!(!group.attr_exists("gattr_04").unwrap());
    assert_eq!(
        group
            .attr("grnam_04")
            .unwrap()
            .read_scalar::<i32>()
            .unwrap(),
        4
    );
    let dataset = f.dataset("metadata/values").unwrap();
    assert!(!dataset.attr_exists("dattr_03").unwrap());
    assert!(!dataset.attr_exists("dattr_04").unwrap());
    assert_eq!(
        dataset
            .attr("drnam_04")
            .unwrap()
            .read_scalar::<i32>()
            .unwrap(),
        40
    );
    assert_eq!(dataset.read::<i32>().unwrap(), vec![1, 2, 3]);
}

#[test]
fn test_mutable_file_renames_compact_attributes_same_length() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rename_compact_attrs.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.add_attr("alpha", 11i32).unwrap();
        wf.add_attr("taken", 12i32).unwrap();
        let mut group = wf.create_group("metadata").unwrap();
        group.add_attr("g_old", 13i32).unwrap();
        group
            .new_dataset_builder("values")
            .attr("dsold", 14i32)
            .unwrap()
            .write::<i32>(&[10, 20])
            .unwrap();
        wf.flush().unwrap();
    }

    {
        let mut mf = MutableFile::open_rw(&path).unwrap();
        mf.rename_root_attr("alpha", "omega").unwrap();
        mf.rename_group_attr("metadata", "g_old", "g_new").unwrap();
        mf.rename_dataset_attr("metadata/values", "dsold", "dsnew")
            .unwrap();

        let err = mf
            .rename_root_attr("omega", "om")
            .expect_err("shrinking compact rename should be rejected");
        assert!(err.to_string().contains("cannot grow"));

        let err = mf
            .rename_root_attr("omega", "taken")
            .expect_err("duplicate compact rename target should be rejected");
        assert!(err.to_string().contains("already exists"));
    }

    {
        let f = File::open(&path).unwrap();
        assert!(!f.attr_exists("alpha").unwrap());
        assert!(f.attr_exists("omega").unwrap());
        assert_eq!(f.attr("omega").unwrap().read_scalar::<i32>().unwrap(), 11);
        assert!(f.attr_exists("taken").unwrap());

        let group = f.group("metadata").unwrap();
        assert!(!group.attr_exists("g_old").unwrap());
        assert!(group.attr_exists("g_new").unwrap());
        assert_eq!(
            group.attr("g_new").unwrap().read_scalar::<i32>().unwrap(),
            13
        );

        let ds = f.dataset("metadata/values").unwrap();
        assert!(!ds.attr_exists("dsold").unwrap());
        assert!(ds.attr_exists("dsnew").unwrap());
        assert_eq!(ds.attr("dsnew").unwrap().read_scalar::<i32>().unwrap(), 14);
        assert_eq!(ds.read::<i32>().unwrap(), vec![10, 20]);
    }
}

#[test]
fn test_resize_then_write_appended_chunk() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("resize_write_chunk.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("data")
            .shape(&[10])
            .chunk(&[5])
            .resizable()
            .write::<i32>(&(0..10).collect::<Vec<_>>())
            .unwrap();
        wf.flush().unwrap();
    }

    {
        let mut mf = MutableFile::open_rw(&path).unwrap();
        mf.resize_dataset("data", &[15]).unwrap();
        let chunk: Vec<i32> = (10..15).collect();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                chunk.as_ptr() as *const u8,
                chunk.len() * std::mem::size_of::<i32>(),
            )
        };
        mf.write_chunk("data", &[10], bytes).unwrap();
    }

    {
        let f = File::open(&path).unwrap();
        let vals: Vec<i32> = f.dataset("data").unwrap().read::<i32>().unwrap();
        assert_eq!(vals, (0..15).collect::<Vec<_>>());
    }
}

#[test]
fn test_resize_shrink_hides_removed_chunks() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("resize_shrink_hides_chunks.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("data")
            .shape(&[15])
            .chunk(&[5])
            .resizable()
            .write::<i32>(&(0..15).collect::<Vec<_>>())
            .unwrap();
        wf.flush().unwrap();
    }

    {
        let mut mf = MutableFile::open_rw(&path).unwrap();
        mf.resize_dataset("data", &[6]).unwrap();
    }

    {
        let f = File::open(&path).unwrap();
        let vals: Vec<i32> = f.dataset("data").unwrap().read::<i32>().unwrap();
        assert_eq!(vals, vec![0, 1, 2, 3, 4, 5]);
    }
}

#[test]
fn test_resize_grow_uses_chunked_fill_value() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("resize_grow_fill.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("data")
            .shape(&[5])
            .chunk(&[5])
            .resizable()
            .fill_properties(1, 2)
            .fill_value::<i32>(-7)
            .write::<i32>(&(0..5).collect::<Vec<_>>())
            .unwrap();
        wf.flush().unwrap();
    }

    {
        let mut mf = MutableFile::open_rw(&path).unwrap();
        mf.resize_dataset("data", &[10]).unwrap();
    }

    {
        let f = File::open(&path).unwrap();
        let vals: Vec<i32> = f.dataset("data").unwrap().read::<i32>().unwrap();
        assert_eq!(vals, vec![0, 1, 2, 3, 4, -7, -7, -7, -7, -7]);
    }
}

#[test]
fn test_write_chunk_replaces_existing_chunk() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("replace_chunk.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("data")
            .shape(&[10])
            .chunk(&[5])
            .resizable()
            .write::<i32>(&(0..10).collect::<Vec<_>>())
            .unwrap();
        wf.flush().unwrap();
    }

    {
        let mut mf = MutableFile::open_rw(&path).unwrap();
        let chunk: Vec<i32> = (100..105).collect();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                chunk.as_ptr() as *const u8,
                chunk.len() * std::mem::size_of::<i32>(),
            )
        };
        mf.write_chunk("data", &[5], bytes).unwrap();
    }

    {
        let f = File::open(&path).unwrap();
        let vals: Vec<i32> = f.dataset("data").unwrap().read::<i32>().unwrap();
        assert_eq!(vals, vec![0, 1, 2, 3, 4, 100, 101, 102, 103, 104]);
    }
}

#[test]
fn test_write_chunk_splits_full_v1_btree_leaf() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("split_full_chunk_btree.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("data")
            .shape(&[320])
            .chunk(&[5])
            .resizable()
            .write::<i32>(&(0..320).collect::<Vec<_>>())
            .unwrap();
        wf.flush().unwrap();
    }

    {
        let mut mf = MutableFile::open_rw(&path).unwrap();
        mf.resize_dataset("data", &[325]).unwrap();
        let chunk: Vec<i32> = (320..325).collect();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                chunk.as_ptr() as *const u8,
                chunk.len() * std::mem::size_of::<i32>(),
            )
        };
        mf.write_chunk("data", &[320], bytes).unwrap();
    }

    {
        let mut mf = MutableFile::open_rw(&path).unwrap();
        mf.resize_dataset("data", &[330]).unwrap();
        let chunk: Vec<i32> = (325..330).collect();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                chunk.as_ptr() as *const u8,
                chunk.len() * std::mem::size_of::<i32>(),
            )
        };
        mf.write_chunk("data", &[325], bytes).unwrap();
    }

    {
        let f = File::open(&path).unwrap();
        let vals: Vec<i32> = f.dataset("data").unwrap().read::<i32>().unwrap();
        assert_eq!(vals, (0..330).collect::<Vec<_>>());
    }
}

#[test]
fn test_write_chunk_replaces_existing_v4_fixed_array_chunk() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("replace_v4_fixed_array.h5");
    std::fs::copy("tests/data/hdf5_ref/v4_fixed_array_chunks.h5", &path).unwrap();

    {
        let mut mf = MutableFile::open_rw(&path).unwrap();
        let chunk: Vec<i32> = (1000..1010).collect();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                chunk.as_ptr() as *const u8,
                chunk.len() * std::mem::size_of::<i32>(),
            )
        };
        mf.write_chunk("fixed_array", &[0], bytes).unwrap();
    }

    {
        let f = File::open(&path).unwrap();
        let vals: Vec<i32> = f.dataset("fixed_array").unwrap().read::<i32>().unwrap();
        let mut expected: Vec<i32> = (0..100).collect();
        expected[..10].copy_from_slice(&(1000..1010).collect::<Vec<_>>());
        assert_eq!(vals, expected);
    }
}

#[test]
fn test_resize_then_write_appended_v4_btree2_chunk() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("append_v4_btree2.h5");
    std::fs::copy("tests/data/hdf5_ref/v4_btree2_chunks.h5", &path).unwrap();

    {
        let mut mf = MutableFile::open_rw(&path).unwrap();
        mf.resize_dataset("btree_v2", &[12, 8]).unwrap();
        let chunk: Vec<i32> = (64..80).collect();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                chunk.as_ptr() as *const u8,
                chunk.len() * std::mem::size_of::<i32>(),
            )
        };
        mf.write_chunk("btree_v2", &[8, 0], bytes).unwrap();
    }

    {
        let f = File::open(&path).unwrap();
        let vals: Vec<i32> = f.dataset("btree_v2").unwrap().read::<i32>().unwrap();
        assert_eq!(vals.len(), 96);
        assert_eq!(&vals[..64], &(0..64).collect::<Vec<_>>());
        assert_eq!(&vals[64..68], &[64, 65, 66, 67]);
        assert_eq!(&vals[72..76], &[68, 69, 70, 71]);
        assert_eq!(&vals[80..84], &[72, 73, 74, 75]);
        assert_eq!(&vals[88..92], &[76, 77, 78, 79]);
        for idx in (68..72).chain(76..80).chain(84..88).chain(92..96) {
            assert_eq!(vals[idx], 0);
        }
    }

    let out = std::process::Command::new("h5dump")
        .arg("-H")
        .arg(&path)
        .output();
    if let Ok(out) = out {
        assert!(
            out.status.success(),
            "h5dump -H failed on appended v2 B-tree file: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

#[test]
fn test_resize_then_write_appended_v4_extensible_array_chunk() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("append_v4_extensible_array.h5");
    std::fs::copy("tests/data/hdf5_ref/v4_extensible_array_chunks.h5", &path).unwrap();

    {
        let mut mf = MutableFile::open_rw(&path).unwrap();
        mf.resize_dataset("extensible_array", &[100]).unwrap();
        let chunk: Vec<f64> = (80..100).map(|value| value as f64).collect();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                chunk.as_ptr() as *const u8,
                chunk.len() * std::mem::size_of::<f64>(),
            )
        };
        mf.write_chunk("extensible_array", &[80], bytes).unwrap();
    }

    {
        let f = File::open(&path).unwrap();
        let vals: Vec<f64> = f
            .dataset("extensible_array")
            .unwrap()
            .read::<f64>()
            .unwrap();
        let expected: Vec<f64> = (0..100).map(|value| value as f64).collect();
        assert_eq!(vals, expected);
    }

    let out = std::process::Command::new("h5dump")
        .arg("-H")
        .arg(&path)
        .output();
    if let Ok(out) = out {
        assert!(
            out.status.success(),
            "h5dump -H failed on appended extensible-array file: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

#[test]
fn test_resize_then_write_v4_extensible_array_into_super_block() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("append_v4_extensible_array_super_block.h5");
    std::fs::copy("tests/data/hdf5_ref/v4_extensible_array_chunks.h5", &path).unwrap();

    {
        let mut mf = MutableFile::open_rw(&path).unwrap();
        mf.resize_dataset("extensible_array", &[4_900]).unwrap();
        for chunk_start in (80..4_900).step_by(20) {
            let chunk: Vec<f64> = (chunk_start..chunk_start + 20)
                .map(|value| value as f64)
                .collect();
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    chunk.as_ptr() as *const u8,
                    chunk.len() * std::mem::size_of::<f64>(),
                )
            };
            mf.write_chunk("extensible_array", &[chunk_start as u64], bytes)
                .unwrap();
        }
    }

    {
        let f = File::open(&path).unwrap();
        let vals: Vec<f64> = f
            .dataset("extensible_array")
            .unwrap()
            .read::<f64>()
            .unwrap();
        let expected: Vec<f64> = (0..4_900).map(|value| value as f64).collect();
        assert_eq!(vals, expected);
    }

    let out = std::process::Command::new("h5dump")
        .arg("-H")
        .arg(&path)
        .output();
    if let Ok(out) = out {
        assert!(
            out.status.success(),
            "h5dump -H failed on super-block extensible-array append file: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

#[test]
fn test_write_chunk_replaces_filtered_chunk() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("replace_filtered_chunk.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("data")
            .shape(&[20])
            .chunk(&[5])
            .shuffle()
            .deflate(4)
            .write::<i32>(&(0..20).collect::<Vec<_>>())
            .unwrap();
        wf.flush().unwrap();
    }

    {
        let mut mf = MutableFile::open_rw(&path).unwrap();
        let chunk: Vec<i32> = (1000..1005).collect();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                chunk.as_ptr() as *const u8,
                chunk.len() * std::mem::size_of::<i32>(),
            )
        };
        mf.write_chunk("data", &[5], bytes).unwrap();
    }

    {
        let f = File::open(&path).unwrap();
        let vals: Vec<i32> = f.dataset("data").unwrap().read::<i32>().unwrap();
        let mut expected: Vec<i32> = (0..20).collect();
        expected[5..10].copy_from_slice(&(1000..1005).collect::<Vec<_>>());
        assert_eq!(vals, expected);
    }
}

#[test]
fn test_write_chunk_replaces_fletcher32_chunk() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("replace_fletcher32_chunk.h5");
    std::fs::copy("tests/data/hdf5_ref/layouts_and_filters.h5", &path).unwrap();

    {
        let mut mf = MutableFile::open_rw(&path).unwrap();
        let chunk: Vec<f32> = (1000..1025).map(|value| value as f32).collect();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                chunk.as_ptr() as *const u8,
                chunk.len() * std::mem::size_of::<f32>(),
            )
        };
        mf.write_chunk("chunked_fletcher", &[25], bytes).unwrap();
    }

    {
        let f = File::open(&path).unwrap();
        let vals: Vec<f32> = f
            .dataset("chunked_fletcher")
            .unwrap()
            .read::<f32>()
            .unwrap();
        let mut expected: Vec<f32> = (0..100).map(|value| value as f32).collect();
        expected[25..50]
            .copy_from_slice(&(1000..1025).map(|value| value as f32).collect::<Vec<_>>());
        assert_eq!(vals, expected);
    }
}

#[test]
fn test_resize_non_chunked_fails() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("resize_nonchunked.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("contiguous")
            .write::<f64>(&[1.0, 2.0, 3.0])
            .unwrap();
        wf.flush().unwrap();
    }

    {
        let mut mf = MutableFile::open_rw(&path).unwrap();
        let result = mf.resize_dataset("contiguous", &[5]);
        assert!(result.is_err());
    }
}

#[test]
fn test_resize_wrong_ndims_fails() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("resize_wrongdims.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("data")
            .shape(&[10])
            .chunk(&[5])
            .resizable()
            .write::<i32>(&(0..10).collect::<Vec<_>>())
            .unwrap();
        wf.flush().unwrap();
    }

    {
        let mut mf = MutableFile::open_rw(&path).unwrap();
        // Try 2D resize on 1D dataset
        let result = mf.resize_dataset("data", &[5, 2]);
        assert!(result.is_err());
    }
}

#[test]
fn test_mutable_file_read() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mutable_read.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("values")
            .write::<f32>(&[1.0, 2.0, 3.0])
            .unwrap();
        wf.flush().unwrap();
    }

    {
        let mf = MutableFile::open_rw(&path).unwrap();
        let names = mf.member_names().unwrap();
        assert!(names.contains(&"values".to_string()));

        let ds = mf.dataset("values").unwrap();
        let vals: Vec<f32> = ds.read::<f32>().unwrap();
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    }
}
