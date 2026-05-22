use hdf5_pure_rust::format::checksum::checksum_metadata;
use hdf5_pure_rust::format::messages::data_layout::ChunkIndexType;
use hdf5_pure_rust::{Dataset, File, H5Type, MutableFile, Result, WritableFile};

const V0_BASE_ADDR_OFFSET: usize = 24;
const V2_BASE_ADDR_OFFSET: usize = 12;
const USERBLOCK_SIZE: usize = 512;

fn dataset_shape_into(ds: &Dataset, shape: &mut Vec<u64>) -> Result<()> {
    ds.shape_into(shape)
}

fn dataset_read_into<T>(ds: &Dataset, values: &mut [T]) -> Result<()>
where
    T: H5Type,
{
    ds.read_into(values)
}

fn assert_h5dump_dataset_read(path: &std::path::Path, dataset: &str, context: &str) {
    let out = std::process::Command::new("h5dump")
        .arg("-d")
        .arg(dataset)
        .arg(path)
        .output();
    if let Ok(out) = out {
        assert!(
            out.status.success(),
            "h5dump data read failed on {context}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

fn assert_h5py_script(path: &std::path::Path, script: &str, context: &str) {
    let code = format!(
        "import sys, h5py\n\
         f = h5py.File(sys.argv[1], 'r')\n\
         {script}\n\
         f.close()\n\
         print('OK')"
    );
    let out = std::process::Command::new("python3")
        .arg("-c")
        .arg(code)
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

fn assert_logical_eoa_matches_file_len(path: &std::path::Path) {
    let f = File::open(path).unwrap();
    let physical_len = std::fs::metadata(path).unwrap().len();
    assert_eq!(f.eoa(), physical_len - f.userblock());
}

fn copy_v0_fixture_with_userblock(src: &str, dst: &std::path::Path) {
    let mut original = std::fs::read(src).unwrap();
    original[V0_BASE_ADDR_OFFSET..V0_BASE_ADDR_OFFSET + 8]
        .copy_from_slice(&(USERBLOCK_SIZE as u64).to_le_bytes());
    let mut with_userblock = vec![0u8; USERBLOCK_SIZE];
    with_userblock[..b"resize test userblock\0".len()].copy_from_slice(b"resize test userblock\0");
    with_userblock.extend_from_slice(&original);
    std::fs::write(dst, with_userblock).unwrap();
}

fn copy_v2_v3_fixture_with_userblock(src: &str, dst: &std::path::Path) {
    let mut original = std::fs::read(src).unwrap();
    assert!(original[8] >= 2, "expected a v2/v3 superblock fixture");
    let sizeof_addr = usize::from(original[9]);
    let checksum_offset = V2_BASE_ADDR_OFFSET + 4 * sizeof_addr;
    let encoded_base = (USERBLOCK_SIZE as u64).to_le_bytes();
    original[V2_BASE_ADDR_OFFSET..V2_BASE_ADDR_OFFSET + sizeof_addr]
        .copy_from_slice(&encoded_base[..sizeof_addr]);
    let checksum = checksum_metadata(&original[..checksum_offset]);
    original[checksum_offset..checksum_offset + 4].copy_from_slice(&checksum.to_le_bytes());

    let mut with_userblock = vec![0u8; USERBLOCK_SIZE];
    with_userblock[..b"resize test modern userblock\0".len()]
        .copy_from_slice(b"resize test modern userblock\0");
    with_userblock.extend_from_slice(&original);
    std::fs::write(dst, with_userblock).unwrap();
}

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
        let mut shape = Vec::new();
        dataset_shape_into(&ds, &mut shape).unwrap();
        assert_eq!(shape, vec![10]);
        let mut vals = vec![0.0; ds.size().unwrap() as usize];
        dataset_read_into(&ds, &mut vals).unwrap();
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
        let mut shape = Vec::new();
        dataset_shape_into(&ds, &mut shape).unwrap();
        assert_eq!(shape, vec![7]);
        let mut vals = vec![0.0; ds.size().unwrap() as usize];
        dataset_read_into(&ds, &mut vals).unwrap();
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
        let mut shape = Vec::new();
        dataset_shape_into(&ds, &mut shape).unwrap();
        assert_eq!(shape, vec![15]);
        let mut vals = vec![0.0; ds.size().unwrap() as usize];
        dataset_read_into(&ds, &mut vals).unwrap();
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
        let mut values = vec![0; ds.size().unwrap() as usize];
        dataset_read_into(&ds, &mut values).unwrap();
        assert_eq!(values, vec![10, 20]);
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
    let mut values = vec![0; dataset.size().unwrap() as usize];
    dataset_read_into(&dataset, &mut values).unwrap();
    assert_eq!(values, vec![1, 2, 3]);
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
        let mut values = vec![0; ds.size().unwrap() as usize];
        dataset_read_into(&ds, &mut values).unwrap();
        assert_eq!(values, vec![10, 20]);
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
        let ds = f.dataset("data").unwrap();
        let mut vals = vec![0; ds.size().unwrap() as usize];
        dataset_read_into(&ds, &mut vals).unwrap();
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
            .max_shape(&[20])
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
        let ds = f.dataset("data").unwrap();
        let mut vals = vec![0; ds.size().unwrap() as usize];
        dataset_read_into(&ds, &mut vals).unwrap();
        assert_eq!(vals, vec![0, 1, 2, 3, 4, 5]);
    }
}

#[test]
fn test_resize_btree_v2_shrink_then_grow_does_not_reexpose_pruned_chunks() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("resize_btree_v2_shrink_grow.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("data")
            .shape(&[15])
            .max_shape(&[20])
            .chunk(&[5])
            .fill_value::<i32>(-7)
            .write::<i32>(&(0..15).collect::<Vec<_>>())
            .unwrap();
        wf.flush().unwrap();
    }

    {
        let mut mf = MutableFile::open_rw(&path).unwrap();
        mf.resize_dataset("data", &[5]).unwrap();
        mf.resize_dataset("data", &[15]).unwrap();
    }

    {
        let f = File::open(&path).unwrap();
        let ds = f.dataset("data").unwrap();
        let info = ds.info().unwrap();
        assert_eq!(info.layout.chunk_index_type, Some(ChunkIndexType::BTreeV2));
        let mut vals = vec![0; ds.size().unwrap() as usize];
        dataset_read_into(&ds, &mut vals).unwrap();
        assert_eq!(
            vals,
            vec![0, 1, 2, 3, 4, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7]
        );
    }

    assert_logical_eoa_matches_file_len(&path);
    assert_h5dump_dataset_read(&path, "data", "B-tree v2 shrink-grow pruned file");
}

#[test]
fn test_resize_btree_v2_partial_shrink_then_grow_scrubs_boundary_chunk_tail() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("resize_btree_v2_partial_shrink_grow.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("data")
            .shape(&[15])
            .max_shape(&[20])
            .chunk(&[5])
            .fill_value::<i32>(-7)
            .write::<i32>(&(0..15).collect::<Vec<_>>())
            .unwrap();
        wf.flush().unwrap();
    }

    {
        let mut mf = MutableFile::open_rw(&path).unwrap();
        mf.resize_dataset("data", &[7]).unwrap();
        mf.resize_dataset("data", &[15]).unwrap();
    }

    {
        let f = File::open(&path).unwrap();
        let ds = f.dataset("data").unwrap();
        assert_eq!(
            ds.info().unwrap().layout.chunk_index_type,
            Some(ChunkIndexType::BTreeV2)
        );
        let mut vals = vec![0; ds.size().unwrap() as usize];
        dataset_read_into(&ds, &mut vals).unwrap();
        assert_eq!(
            vals,
            vec![0, 1, 2, 3, 4, 5, 6, -7, -7, -7, -7, -7, -7, -7, -7]
        );
    }

    assert_logical_eoa_matches_file_len(&path);
    assert_h5dump_dataset_read(&path, "data", "B-tree v2 partial shrink-grow scrubbed file");
}

#[test]
fn test_resize_2d_btree_v2_partial_shrink_then_grow_scrubs_boundary_chunk_tail() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("resize_2d_btree_v2_partial_shrink_grow.h5");
    std::fs::copy("tests/data/hdf5_ref/v4_btree2_chunks.h5", &path).unwrap();

    {
        let mut mf = MutableFile::open_rw(&path).unwrap();
        mf.resize_dataset("btree_v2", &[6, 8]).unwrap();
        mf.resize_dataset("btree_v2", &[8, 8]).unwrap();
    }

    {
        let f = File::open(&path).unwrap();
        let ds = f.dataset("btree_v2").unwrap();
        assert_eq!(
            ds.info().unwrap().layout.chunk_index_type,
            Some(ChunkIndexType::BTreeV2)
        );
        let mut vals = vec![0; ds.size().unwrap() as usize];
        dataset_read_into(&ds, &mut vals).unwrap();
        let mut expected: Vec<i32> = (0..64).collect();
        expected[48..64].fill(0);
        assert_eq!(vals, expected);
    }

    assert_logical_eoa_matches_file_len(&path);
    assert_h5dump_dataset_read(
        &path,
        "btree_v2",
        "2D B-tree v2 partial shrink-grow scrubbed file",
    );
}

#[test]
fn test_resize_fixed_array_partial_shrink_then_grow_scrubs_boundary_chunk_tail() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("resize_fixed_array_partial_shrink_grow.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("data")
            .shape(&[15])
            .chunk(&[5])
            .fill_value::<i32>(-7)
            .write::<i32>(&(0..15).collect::<Vec<_>>())
            .unwrap();
        wf.flush().unwrap();
    }

    {
        let mut mf = MutableFile::open_rw(&path).unwrap();
        mf.resize_dataset("data", &[7]).unwrap();
        mf.resize_dataset("data", &[10]).unwrap();
    }

    {
        let f = File::open(&path).unwrap();
        let ds = f.dataset("data").unwrap();
        assert_eq!(
            ds.info().unwrap().layout.chunk_index_type,
            Some(ChunkIndexType::FixedArray)
        );
        let mut vals = vec![0; ds.size().unwrap() as usize];
        dataset_read_into(&ds, &mut vals).unwrap();
        assert_eq!(vals, vec![0, 1, 2, 3, 4, 5, 6, -7, -7, -7]);
    }

    assert_logical_eoa_matches_file_len(&path);
    assert_h5dump_dataset_read(
        &path,
        "data",
        "fixed-array partial shrink-grow scrubbed file",
    );
}

#[test]
fn test_resize_extensible_array_partial_shrink_then_grow_scrubs_boundary_chunk_tail() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir
        .path()
        .join("resize_extensible_array_partial_shrink_grow.h5");

    {
        let mut wf = WritableFile::create(&path).unwrap();
        wf.new_dataset_builder("data")
            .shape(&[9])
            .chunk(&[5])
            .fill_value::<i32>(-7)
            .resizable()
            .write::<i32>(&(0..9).collect::<Vec<_>>())
            .unwrap();
        wf.flush().unwrap();
    }

    {
        let mut mf = MutableFile::open_rw(&path).unwrap();
        mf.resize_dataset("data", &[7]).unwrap();
        mf.resize_dataset("data", &[9]).unwrap();
    }

    {
        let f = File::open(&path).unwrap();
        let ds = f.dataset("data").unwrap();
        assert_eq!(
            ds.info().unwrap().layout.chunk_index_type,
            Some(ChunkIndexType::ExtensibleArray)
        );
        let mut vals = vec![0; ds.size().unwrap() as usize];
        dataset_read_into(&ds, &mut vals).unwrap();
        assert_eq!(vals, vec![0, 1, 2, 3, 4, 5, 6, -7, -7]);
    }

    assert_logical_eoa_matches_file_len(&path);
    assert_h5dump_dataset_read(
        &path,
        "data",
        "extensible-array partial shrink-grow scrubbed file",
    );
}

#[test]
fn test_resize_v1_btree_partial_shrink_then_grow_scrubs_boundary_chunk_tail() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("resize_v1_btree_partial_shrink_grow.h5");
    std::fs::copy("tests/data/hdf5_ref/v1_btree_full_leaf_gap.h5", &path).unwrap();

    {
        let mut mf = MutableFile::open_rw(&path).unwrap();
        let chunk: Vec<i32> = (320..325).collect();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                chunk.as_ptr() as *const u8,
                chunk.len() * std::mem::size_of::<i32>(),
            )
        };
        mf.write_chunk("btree_v1_full_leaf_gap", &[320], bytes)
            .unwrap();
        mf.resize_dataset("btree_v1_full_leaf_gap", &[322]).unwrap();
        mf.resize_dataset("btree_v1_full_leaf_gap", &[325]).unwrap();
    }

    {
        let f = File::open(&path).unwrap();
        let ds = f.dataset("btree_v1_full_leaf_gap").unwrap();
        assert_eq!(ds.info().unwrap().layout.chunk_index_type, None);
        let mut vals = vec![0; ds.size().unwrap() as usize];
        dataset_read_into(&ds, &mut vals).unwrap();
        assert_eq!(&vals[..320], &(0..320).collect::<Vec<_>>());
        assert_eq!(&vals[320..325], &[320, 321, -1, -1, -1]);
    }

    assert_logical_eoa_matches_file_len(&path);
    assert_h5dump_dataset_read(
        &path,
        "btree_v1_full_leaf_gap",
        "v1 B-tree partial shrink-grow scrubbed file",
    );
    assert_h5py_script(
        &path,
        "x = f['btree_v1_full_leaf_gap'][:]\n\
         assert len(x) == 325\n\
         assert list(x[:3]) == [0, 1, 2]\n\
         assert list(x[319:325]) == [319, 320, 321, -1, -1, -1]",
        "v1 B-tree partial shrink-grow scrubbed file",
    );
}

#[test]
fn test_resize_v1_btree_userblock_partial_shrink_scrubs_without_corrupting_prefix() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("resize_v1_btree_userblock_partial.h5");
    copy_v0_fixture_with_userblock("tests/data/hdf5_ref/v1_btree_full_leaf_gap.h5", &path);
    let before = std::fs::read(&path).unwrap();
    let prefix = before[..USERBLOCK_SIZE].to_vec();

    {
        let mut mf = MutableFile::open_rw(&path).unwrap();
        let chunk: Vec<i32> = (320..325).collect();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                chunk.as_ptr() as *const u8,
                chunk.len() * std::mem::size_of::<i32>(),
            )
        };
        mf.write_chunk("btree_v1_full_leaf_gap", &[320], bytes)
            .unwrap();
        mf.resize_dataset("btree_v1_full_leaf_gap", &[322]).unwrap();
        mf.resize_dataset("btree_v1_full_leaf_gap", &[325]).unwrap();
    }

    let after = std::fs::read(&path).unwrap();
    assert_eq!(&after[..USERBLOCK_SIZE], prefix.as_slice());
    let f = File::open(&path).unwrap();
    assert_eq!(f.userblock(), USERBLOCK_SIZE as u64);
    let ds = f.dataset("btree_v1_full_leaf_gap").unwrap();
    let mut vals = vec![0; ds.size().unwrap() as usize];
    dataset_read_into(&ds, &mut vals).unwrap();
    assert_eq!(&vals[..320], &(0..320).collect::<Vec<_>>());
    assert_eq!(&vals[320..325], &[320, 321, -1, -1, -1]);
    assert_logical_eoa_matches_file_len(&path);
}

#[test]
fn test_resize_v1_btree_userblock_chunk_boundary_updates_dataspace_only() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("resize_v1_btree_userblock_boundary.h5");
    copy_v0_fixture_with_userblock("tests/data/hdf5_ref/v1_btree_full_leaf_gap.h5", &path);
    let before = std::fs::read(&path).unwrap();
    let prefix = before[..USERBLOCK_SIZE].to_vec();

    {
        let mut mf = MutableFile::open_rw(&path).unwrap();
        mf.resize_dataset("btree_v1_full_leaf_gap", &[320]).unwrap();
    }

    let after = std::fs::read(&path).unwrap();
    assert_eq!(&after[..USERBLOCK_SIZE], prefix.as_slice());
    let f = File::open(&path).unwrap();
    assert_eq!(f.userblock(), USERBLOCK_SIZE as u64);
    let ds = f.dataset("btree_v1_full_leaf_gap").unwrap();
    let mut vals = vec![0; ds.size().unwrap() as usize];
    dataset_read_into(&ds, &mut vals).unwrap();
    assert_eq!(vals, (0..320).collect::<Vec<_>>());

    assert_logical_eoa_matches_file_len(&path);
    // This synthetic userblock fixture is made by shifting the v0 superblock
    // only. libhdf5 can inspect its headers but cannot read its raw chunk data,
    // so this test verifies the in-place dataspace rewrite with the pure reader.
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
        let ds = f.dataset("data").unwrap();
        let mut vals = vec![0; ds.size().unwrap() as usize];
        dataset_read_into(&ds, &mut vals).unwrap();
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
        let ds = f.dataset("data").unwrap();
        let mut vals = vec![0; ds.size().unwrap() as usize];
        dataset_read_into(&ds, &mut vals).unwrap();
        assert_eq!(vals, vec![0, 1, 2, 3, 4, 100, 101, 102, 103, 104]);
    }
}

#[test]
fn test_write_chunk_splits_full_v1_btree_leaf() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("split_full_v1_btree_leaf.h5");
    std::fs::copy("tests/data/hdf5_ref/v1_btree_full_leaf_gap.h5", &path).unwrap();

    {
        let f = File::open(&path).unwrap();
        let ds = f.dataset("btree_v1_full_leaf_gap").unwrap();
        assert_eq!(ds.info().unwrap().layout.chunk_index_type, None);
        let mut vals = vec![0; ds.size().unwrap() as usize];
        dataset_read_into(&ds, &mut vals).unwrap();
        assert_eq!(&vals[..320], &(0..320).collect::<Vec<_>>());
        assert_eq!(&vals[320..325], &[-1; 5]);
    }

    {
        let mut mf = MutableFile::open_rw(&path).unwrap();
        let chunk: Vec<i32> = (320..325).collect();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                chunk.as_ptr() as *const u8,
                chunk.len() * std::mem::size_of::<i32>(),
            )
        };
        mf.write_chunk("btree_v1_full_leaf_gap", &[320], bytes)
            .unwrap();
    }

    {
        let f = File::open(&path).unwrap();
        let ds = f.dataset("btree_v1_full_leaf_gap").unwrap();
        let mut vals = vec![0; ds.size().unwrap() as usize];
        dataset_read_into(&ds, &mut vals).unwrap();
        assert_eq!(vals, (0..325).collect::<Vec<_>>());
    }

    assert_logical_eoa_matches_file_len(&path);
    assert_h5dump_dataset_read(&path, "btree_v1_full_leaf_gap", "split v1 B-tree file");
}

#[test]
fn test_write_chunk_updates_v0_userblock_v1_btree_without_corrupting_prefix() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("userblock_split_full_v1_btree_leaf.h5");
    copy_v0_fixture_with_userblock("tests/data/hdf5_ref/v1_btree_full_leaf_gap.h5", &path);
    let before = std::fs::read(&path).unwrap();
    let prefix = before[..USERBLOCK_SIZE].to_vec();

    {
        let f = File::open(&path).unwrap();
        assert_eq!(f.userblock(), 512);
        let ds = f.dataset("btree_v1_full_leaf_gap").unwrap();
        let mut vals = vec![0; ds.size().unwrap() as usize];
        dataset_read_into(&ds, &mut vals).unwrap();
        assert_eq!(&vals[..320], &(0..320).collect::<Vec<_>>());
        assert_eq!(&vals[320..325], &[-1; 5]);
    }

    {
        let mut mf = MutableFile::open_rw(&path).unwrap();
        let chunk: Vec<i32> = (320..325).collect();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                chunk.as_ptr() as *const u8,
                chunk.len() * std::mem::size_of::<i32>(),
            )
        };
        mf.write_chunk("btree_v1_full_leaf_gap", &[320], bytes)
            .unwrap();
    }

    let after = std::fs::read(&path).unwrap();
    assert_eq!(&after[..USERBLOCK_SIZE], prefix.as_slice());
    {
        let f = File::open(&path).unwrap();
        assert_eq!(f.userblock(), 512);
        let ds = f.dataset("btree_v1_full_leaf_gap").unwrap();
        let mut vals = vec![0; ds.size().unwrap() as usize];
        dataset_read_into(&ds, &mut vals).unwrap();
        assert_eq!(vals, (0..325).collect::<Vec<_>>());
    }
    assert_logical_eoa_matches_file_len(&path);
}

#[test]
fn test_write_chunk_rejects_modern_userblock_file_without_corrupting_it() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("userblock_v4_fixed_array.h5");
    copy_v2_v3_fixture_with_userblock("tests/data/hdf5_ref/v4_fixed_array_chunks.h5", &path);
    let before = std::fs::read(&path).unwrap();

    {
        let f = File::open(&path).unwrap();
        assert_eq!(f.userblock(), USERBLOCK_SIZE as u64);
        assert_logical_eoa_matches_file_len(&path);
        let ds = f.dataset("fixed_array").unwrap();
        let mut vals = vec![0; ds.size().unwrap() as usize];
        dataset_read_into(&ds, &mut vals).unwrap();
        assert_eq!(vals, (0..100).collect::<Vec<_>>());
    }

    {
        let mut mf = MutableFile::open_rw(&path).unwrap();
        let chunk: Vec<i32> = (1000..1010).collect();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                chunk.as_ptr() as *const u8,
                chunk.len() * std::mem::size_of::<i32>(),
            )
        };
        let err = mf
            .write_chunk("fixed_array", &[0], bytes)
            .expect_err("write_chunk should reject modern userblock files before appending data");
        assert!(err.to_string().contains("userblock"));
    }

    assert_eq!(std::fs::read(&path).unwrap(), before);
}

#[test]
fn test_resize_then_write_appended_extensible_array_chunk() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("append_extensible_array_chunks.h5");

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
        let ds = f.dataset("data").unwrap();
        assert_eq!(
            ds.info().unwrap().layout.chunk_index_type,
            Some(ChunkIndexType::ExtensibleArray)
        );
        let mut vals = vec![0; ds.size().unwrap() as usize];
        dataset_read_into(&ds, &mut vals).unwrap();
        assert_eq!(vals, (0..330).collect::<Vec<_>>());
    }

    assert_logical_eoa_matches_file_len(&path);
    assert_h5dump_dataset_read(&path, "data", "writer-created extensible-array append file");
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
        let ds = f.dataset("fixed_array").unwrap();
        let mut vals = vec![0; ds.size().unwrap() as usize];
        dataset_read_into(&ds, &mut vals).unwrap();
        let mut expected: Vec<i32> = (0..100).collect();
        expected[..10].copy_from_slice(&(1000..1010).collect::<Vec<_>>());
        assert_eq!(vals, expected);
    }
    assert_logical_eoa_matches_file_len(&path);
    assert_h5dump_dataset_read(&path, "fixed_array", "replaced fixed-array file");
    assert_h5py_script(
        &path,
        "x = f['fixed_array'][:]\n\
         assert len(x) == 100\n\
         assert list(x[:10]) == list(range(1000, 1010))\n\
         assert list(x[10:13]) == [10, 11, 12]\n\
         assert int(x[-1]) == 99",
        "replaced fixed-array file",
    );
}

#[test]
fn test_write_chunk_replaces_paged_v4_fixed_array_chunk() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("replace_paged_v4_fixed_array.h5");
    std::fs::copy("tests/data/hdf5_ref/v4_fixed_array_paged_chunks.h5", &path).unwrap();

    {
        let mut mf = MutableFile::open_rw(&path).unwrap();
        let chunk = [900_123i32];
        let bytes = unsafe {
            std::slice::from_raw_parts(chunk.as_ptr() as *const u8, std::mem::size_of_val(&chunk))
        };
        mf.write_chunk("fixed_array_paged", &[2048], bytes).unwrap();
    }

    {
        let f = File::open(&path).unwrap();
        let ds = f.dataset("fixed_array_paged").unwrap();
        let mut vals = vec![0; ds.size().unwrap() as usize];
        dataset_read_into(&ds, &mut vals).unwrap();
        assert_eq!(vals[2047], 2047);
        assert_eq!(vals[2048], 900_123);
        assert_eq!(vals[2049], 2049);
    }

    assert_logical_eoa_matches_file_len(&path);
    assert_h5dump_dataset_read(
        &path,
        "fixed_array_paged",
        "replaced paged fixed-array file",
    );
    assert_h5py_script(
        &path,
        "x = f['fixed_array_paged']\n\
         assert int(x[2047]) == 2047\n\
         assert int(x[2048]) == 900123\n\
         assert int(x[2049]) == 2049",
        "replaced paged fixed-array file",
    );
}

#[test]
fn test_write_chunk_replaces_existing_v4_extensible_array_chunk() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("replace_v4_extensible_array.h5");
    std::fs::copy("tests/data/hdf5_ref/v4_extensible_array_chunks.h5", &path).unwrap();

    {
        let mut mf = MutableFile::open_rw(&path).unwrap();
        let chunk: Vec<f64> = (1000..1020).map(|value| value as f64).collect();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                chunk.as_ptr() as *const u8,
                chunk.len() * std::mem::size_of::<f64>(),
            )
        };
        mf.write_chunk("extensible_array", &[20], bytes).unwrap();
    }

    {
        let f = File::open(&path).unwrap();
        let ds = f.dataset("extensible_array").unwrap();
        let mut vals = vec![0.0; ds.size().unwrap() as usize];
        dataset_read_into(&ds, &mut vals).unwrap();
        let mut expected: Vec<f64> = (0..80).map(|value| value as f64).collect();
        expected[20..40]
            .copy_from_slice(&(1000..1020).map(|value| value as f64).collect::<Vec<_>>());
        assert_eq!(vals, expected);
    }

    assert_logical_eoa_matches_file_len(&path);
    assert_h5dump_dataset_read(&path, "extensible_array", "replaced extensible-array file");
    assert_h5py_script(
        &path,
        "x = f['extensible_array']\n\
         assert x.shape == (80,)\n\
         assert abs(float(x[19]) - 19.0) < 1e-12\n\
         assert abs(float(x[20]) - 1000.0) < 1e-12\n\
         assert abs(float(x[39]) - 1019.0) < 1e-12\n\
         assert abs(float(x[40]) - 40.0) < 1e-12",
        "replaced extensible-array file",
    );
}

#[test]
fn test_write_chunk_replaces_existing_v4_btree2_chunk() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("replace_v4_btree2.h5");
    std::fs::copy("tests/data/hdf5_ref/v4_btree2_chunks.h5", &path).unwrap();

    {
        let mut mf = MutableFile::open_rw(&path).unwrap();
        let chunk: Vec<i32> = (1000..1016).collect();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                chunk.as_ptr() as *const u8,
                chunk.len() * std::mem::size_of::<i32>(),
            )
        };
        mf.write_chunk("btree_v2", &[4, 0], bytes).unwrap();
    }

    {
        let f = File::open(&path).unwrap();
        let ds = f.dataset("btree_v2").unwrap();
        let mut vals = vec![0; ds.size().unwrap() as usize];
        dataset_read_into(&ds, &mut vals).unwrap();
        let mut expected: Vec<i32> = (0..64).collect();
        for row in 0..4 {
            let src = row * 4;
            let dst = (4 + row) * 8;
            expected[dst..dst + 4]
                .copy_from_slice(&(1000 + src as i32..1004 + src as i32).collect::<Vec<_>>());
        }
        assert_eq!(vals, expected);
    }

    assert_logical_eoa_matches_file_len(&path);
    assert_h5dump_dataset_read(&path, "btree_v2", "replaced v2 B-tree file");
    assert_h5py_script(
        &path,
        "x = f['btree_v2']\n\
         assert x.shape == (8, 8)\n\
         assert list(x[4, 0:4]) == [1000, 1001, 1002, 1003]\n\
         assert list(x[7, 0:4]) == [1012, 1013, 1014, 1015]\n\
         assert int(x[7, 7]) == 63",
        "replaced v2 B-tree file",
    );
}

#[test]
fn test_write_chunk_replaces_existing_deep_v4_btree2_chunk() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("replace_deep_v4_btree2.h5");
    std::fs::copy(
        "tests/data/hdf5_ref/v4_btree2_deep_internal_chunks.h5",
        &path,
    )
    .unwrap();

    {
        let mut mf = MutableFile::open_rw(&path).unwrap();
        let chunk = [900_123i32];
        let bytes = unsafe {
            std::slice::from_raw_parts(chunk.as_ptr() as *const u8, std::mem::size_of_val(&chunk))
        };
        mf.write_chunk("btree_v2_deep_internal", &[0, 0], bytes)
            .unwrap();
    }

    {
        let f = File::open(&path).unwrap();
        let ds = f.dataset("btree_v2_deep_internal").unwrap();
        let mut vals = vec![0; ds.size().unwrap() as usize];
        dataset_read_into(&ds, &mut vals).unwrap();
        assert_eq!(vals.len(), 160 * 160);
        assert_eq!(vals[0], 900_123);
        assert_eq!(vals[1], 1);
        assert_eq!(vals[159], 159);
        assert_eq!(vals[160], 160);
        assert_eq!(vals[25_599], 25_599);
    }

    assert_logical_eoa_matches_file_len(&path);
    assert_h5dump_dataset_read(
        &path,
        "btree_v2_deep_internal",
        "replaced deep v2 B-tree file",
    );
    assert_h5py_script(
        &path,
        "x = f['btree_v2_deep_internal']\n\
         assert x.shape == (160, 160)\n\
         assert int(x[0, 0]) == 900123\n\
         assert int(x[0, 1]) == 1\n\
         assert int(x[1, 0]) == 160\n\
         assert int(x[159, 159]) == 25599",
        "replaced deep v2 B-tree file",
    );
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
        let ds = f.dataset("btree_v2").unwrap();
        let mut vals = vec![0; ds.size().unwrap() as usize];
        dataset_read_into(&ds, &mut vals).unwrap();
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

    assert_logical_eoa_matches_file_len(&path);
    assert_h5dump_dataset_read(&path, "btree_v2", "appended v2 B-tree file");
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
        let ds = f.dataset("extensible_array").unwrap();
        let mut vals = vec![0.0; ds.size().unwrap() as usize];
        dataset_read_into(&ds, &mut vals).unwrap();
        let expected: Vec<f64> = (0..100).map(|value| value as f64).collect();
        assert_eq!(vals, expected);
    }

    assert_logical_eoa_matches_file_len(&path);
    assert_h5dump_dataset_read(&path, "extensible_array", "appended extensible-array file");
}

#[test]
fn test_resize_then_write_non_next_v4_extensible_array_data_block_chunk() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir
        .path()
        .join("non_next_v4_extensible_array_data_block.h5");
    std::fs::copy("tests/data/hdf5_ref/v4_extensible_array_chunks.h5", &path).unwrap();

    {
        let mut mf = MutableFile::open_rw(&path).unwrap();
        mf.resize_dataset("extensible_array", &[120]).unwrap();
        let chunk: Vec<f64> = (1000..1020).map(|value| value as f64).collect();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                chunk.as_ptr() as *const u8,
                chunk.len() * std::mem::size_of::<f64>(),
            )
        };
        mf.write_chunk("extensible_array", &[100], bytes).unwrap();
    }

    {
        let f = File::open(&path).unwrap();
        let ds = f.dataset("extensible_array").unwrap();
        let mut vals = vec![0.0; ds.size().unwrap() as usize];
        dataset_read_into(&ds, &mut vals).unwrap();
        assert_eq!(
            &vals[..80],
            &(0..80).map(|value| value as f64).collect::<Vec<_>>()
        );
        assert_eq!(&vals[80..100], &[0.0; 20]);
        assert_eq!(
            &vals[100..120],
            &(1000..1020).map(|value| value as f64).collect::<Vec<_>>()
        );
    }

    assert_logical_eoa_matches_file_len(&path);
    assert_h5dump_dataset_read(
        &path,
        "extensible_array",
        "non-next extensible-array data-block file",
    );
}

#[test]
fn test_write_chunk_replaces_appended_v4_extensible_array_data_block_chunk() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir
        .path()
        .join("replace_appended_v4_extensible_array_data_block.h5");
    std::fs::copy("tests/data/hdf5_ref/v4_extensible_array_chunks.h5", &path).unwrap();

    {
        let mut mf = MutableFile::open_rw(&path).unwrap();
        mf.resize_dataset("extensible_array", &[120]).unwrap();
        for chunk_start in [80, 100] {
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
        let mut mf = MutableFile::open_rw(&path).unwrap();
        let chunk: Vec<f64> = (2000..2020).map(|value| value as f64).collect();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                chunk.as_ptr() as *const u8,
                chunk.len() * std::mem::size_of::<f64>(),
            )
        };
        mf.write_chunk("extensible_array", &[100], bytes).unwrap();
    }

    {
        let f = File::open(&path).unwrap();
        let ds = f.dataset("extensible_array").unwrap();
        let mut vals = vec![0.0; ds.size().unwrap() as usize];
        dataset_read_into(&ds, &mut vals).unwrap();
        let mut expected: Vec<f64> = (0..120).map(|value| value as f64).collect();
        expected[100..120]
            .copy_from_slice(&(2000..2020).map(|value| value as f64).collect::<Vec<_>>());
        assert_eq!(vals, expected);
    }

    assert_logical_eoa_matches_file_len(&path);
    assert_h5dump_dataset_read(
        &path,
        "extensible_array",
        "replaced appended extensible-array file",
    );
}

#[test]
fn test_write_chunk_replaces_later_v4_extensible_array_index_data_block_chunk() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir
        .path()
        .join("replace_later_v4_extensible_array_data_block.h5");
    std::fs::copy("tests/data/hdf5_ref/v4_extensible_array_chunks.h5", &path).unwrap();

    {
        let mut mf = MutableFile::open_rw(&path).unwrap();
        mf.resize_dataset("extensible_array", &[220]).unwrap();
        for chunk_start in (80..220).step_by(20) {
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
        let mut mf = MutableFile::open_rw(&path).unwrap();
        let chunk: Vec<f64> = (3000..3020).map(|value| value as f64).collect();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                chunk.as_ptr() as *const u8,
                chunk.len() * std::mem::size_of::<f64>(),
            )
        };
        mf.write_chunk("extensible_array", &[180], bytes).unwrap();
    }

    {
        let f = File::open(&path).unwrap();
        let ds = f.dataset("extensible_array").unwrap();
        let mut vals = vec![0.0; ds.size().unwrap() as usize];
        dataset_read_into(&ds, &mut vals).unwrap();
        let mut expected: Vec<f64> = (0..220).map(|value| value as f64).collect();
        expected[180..200]
            .copy_from_slice(&(3000..3020).map(|value| value as f64).collect::<Vec<_>>());
        assert_eq!(vals, expected);
    }

    assert_logical_eoa_matches_file_len(&path);
    assert_h5dump_dataset_read(
        &path,
        "extensible_array",
        "replaced later extensible-array data-block file",
    );
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
        let ds = f.dataset("extensible_array").unwrap();
        let mut vals = vec![0.0; ds.size().unwrap() as usize];
        dataset_read_into(&ds, &mut vals).unwrap();
        let expected: Vec<f64> = (0..4_900).map(|value| value as f64).collect();
        assert_eq!(vals, expected);
    }

    assert_logical_eoa_matches_file_len(&path);
    assert_h5dump_dataset_read(
        &path,
        "extensible_array",
        "super-block extensible-array append file",
    );
}

#[test]
fn test_resize_then_write_sparse_v4_extensible_array_super_block_chunk() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sparse_v4_extensible_array_super_block.h5");
    std::fs::copy("tests/data/hdf5_ref/v4_extensible_array_chunks.h5", &path).unwrap();

    {
        let mut mf = MutableFile::open_rw(&path).unwrap();
        mf.resize_dataset("extensible_array", &[4_920]).unwrap();
        let chunk: Vec<f64> = (9_000..9_020).map(|value| value as f64).collect();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                chunk.as_ptr() as *const u8,
                chunk.len() * std::mem::size_of::<f64>(),
            )
        };
        mf.write_chunk("extensible_array", &[4_900], bytes).unwrap();
    }

    {
        let f = File::open(&path).unwrap();
        let ds = f.dataset("extensible_array").unwrap();
        let mut vals = vec![0.0; ds.size().unwrap() as usize];
        dataset_read_into(&ds, &mut vals).unwrap();
        assert_eq!(
            &vals[..80],
            &(0..80).map(|value| value as f64).collect::<Vec<_>>()
        );
        assert!(vals[80..4_900].iter().all(|&value| value == 0.0));
        assert_eq!(
            &vals[4_900..4_920],
            &(9_000..9_020).map(|value| value as f64).collect::<Vec<_>>()
        );
    }

    assert_logical_eoa_matches_file_len(&path);
    assert_h5dump_dataset_read(
        &path,
        "extensible_array",
        "sparse super-block extensible-array file",
    );
    assert_h5py_script(
        &path,
        "x = f['extensible_array'][:]\n\
         assert len(x) == 4920\n\
         assert list(x[:3]) == [0.0, 1.0, 2.0]\n\
         assert (x[80:4900] == 0.0).all()\n\
         assert list(x[4900:4920]) == [float(v) for v in range(9000, 9020)]",
        "sparse super-block extensible-array file",
    );
}

#[test]
fn test_write_chunk_replaces_paged_v4_extensible_array_super_block_chunk() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir
        .path()
        .join("replace_paged_v4_extensible_array_super_block.h5");
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
        let mut mf = MutableFile::open_rw(&path).unwrap();
        let chunk: Vec<f64> = (9_000..9_020).map(|value| value as f64).collect();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                chunk.as_ptr() as *const u8,
                chunk.len() * std::mem::size_of::<f64>(),
            )
        };
        mf.write_chunk("extensible_array", &[4_880], bytes).unwrap();
    }

    {
        let f = File::open(&path).unwrap();
        let ds = f.dataset("extensible_array").unwrap();
        let mut vals = vec![0.0; ds.size().unwrap() as usize];
        dataset_read_into(&ds, &mut vals).unwrap();
        let mut expected: Vec<f64> = (0..4_900).map(|value| value as f64).collect();
        expected[4_880..4_900]
            .copy_from_slice(&(9_000..9_020).map(|value| value as f64).collect::<Vec<_>>());
        assert_eq!(vals, expected);
    }

    assert_logical_eoa_matches_file_len(&path);
    assert_h5dump_dataset_read(
        &path,
        "extensible_array",
        "replaced paged extensible-array super-block file",
    );
    assert_h5py_script(
        &path,
        "x = f['extensible_array']\n\
         assert x.shape == (4900,)\n\
         assert abs(float(x[4879]) - 4879.0) < 1e-9\n\
         assert list(x[4880:4900]) == [float(v) for v in range(9000, 9020)]",
        "replaced paged extensible-array super-block file",
    );
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
        let ds = f.dataset("data").unwrap();
        let mut vals = vec![0; ds.size().unwrap() as usize];
        dataset_read_into(&ds, &mut vals).unwrap();
        let mut expected: Vec<i32> = (0..20).collect();
        expected[5..10].copy_from_slice(&(1000..1005).collect::<Vec<_>>());
        assert_eq!(vals, expected);
    }

    assert_logical_eoa_matches_file_len(&path);
    assert_h5dump_dataset_read(&path, "data", "replaced shuffle-deflate chunk file");
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
        let ds = f.dataset("chunked_fletcher").unwrap();
        let mut vals = vec![0.0; ds.size().unwrap() as usize];
        dataset_read_into(&ds, &mut vals).unwrap();
        let mut expected: Vec<f32> = (0..100).map(|value| value as f32).collect();
        expected[25..50]
            .copy_from_slice(&(1000..1025).map(|value| value as f32).collect::<Vec<_>>());
        assert_eq!(vals, expected);
    }

    assert_logical_eoa_matches_file_len(&path);
    assert_h5dump_dataset_read(&path, "chunked_fletcher", "replaced fletcher32 chunk file");
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
        let mut names = Vec::new();
        mf.visit_member_names(|name| {
            names.push(name.to_string());
            Ok(())
        })
        .unwrap();
        assert!(names.contains(&"values".to_string()));

        let ds = mf.dataset("values").unwrap();
        let mut vals = vec![0.0; ds.size().unwrap() as usize];
        dataset_read_into(&ds, &mut vals).unwrap();
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    }
}
