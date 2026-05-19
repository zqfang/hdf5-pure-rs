//! Phase T9: Heap and index structure tests.

use std::io::Cursor;

use hdf5_pure_rust::format::checksum::checksum_metadata;
use hdf5_pure_rust::format::fractal_heap::FractalHeapHeader;
use hdf5_pure_rust::format::global_heap::{
    read_global_heap_object_into, read_global_heap_objects_batched, GlobalHeapCollection,
    GlobalHeapRef,
};
use hdf5_pure_rust::io::reader::HdfReader;
use hdf5_pure_rust::{Dataset, File, H5Type, Result};

fn file_member_summary<const N: usize>(
    file: &File,
    expected: [&str; N],
) -> hdf5_pure_rust::Result<(usize, [bool; N])> {
    let mut count = 0;
    let mut found = [false; N];
    file.visit_member_names(|name| {
        count += 1;
        for (idx, expected_name) in expected.iter().enumerate() {
            if name == *expected_name {
                found[idx] = true;
            }
        }
        Ok(())
    })?;
    Ok((count, found))
}

fn group_member_summary<const N: usize>(
    group: &hdf5_pure_rust::Group,
    expected: [&str; N],
) -> hdf5_pure_rust::Result<(usize, [bool; N])> {
    let mut count = 0;
    let mut found = [false; N];
    group.visit_member_names(|name| {
        count += 1;
        for (idx, expected_name) in expected.iter().enumerate() {
            if name == *expected_name {
                found[idx] = true;
            }
        }
        Ok(())
    })?;
    Ok((count, found))
}

fn assert_dataset_shape(ds: &Dataset, expected: &[u64]) {
    let space = ds.space().unwrap();
    assert_eq!(space.shape(), expected);
}

fn dataset_values<T>(ds: &Dataset) -> Result<Vec<T>>
where
    T: H5Type + Default + Clone,
{
    let mut values = vec![T::default(); ds.size()? as usize];
    ds.read_into(&mut values)?;
    Ok(values)
}

fn dataset_strings(ds: &Dataset) -> Result<Vec<String>> {
    let mut strings = Vec::new();
    ds.read_strings_into(&mut strings)?;
    Ok(strings)
}

fn append_serialized_global_heap(
    file: &mut Vec<u8>,
    addr: usize,
    objects: &[(u32, &[u8])],
) -> Result<()> {
    let mut collection = GlobalHeapCollection::create();
    for (index, data) in objects {
        collection.insert(*index, data.to_vec())?;
    }

    let mut image = Vec::new();
    collection.cache_heap_serialize_into(8, &mut image)?;
    file.resize(addr, 0);
    file.extend_from_slice(&image);
    Ok(())
}

// T9a: Global heap (variable-length data)

#[test]
fn t9a_global_heap_vlen_strings() {
    let f = File::open("tests/data/strings.h5").unwrap();
    let ds = f.dataset("vlen_ds").unwrap();
    let strings = dataset_strings(&ds).unwrap();
    assert_eq!(strings, vec!["alpha", "beta", "gamma"]);
}

#[test]
fn t9a_global_heap_batched_reads_preserve_order_across_collections() {
    let first_addr = 32usize;
    let second_addr = 160usize;
    let mut file = Vec::new();
    append_serialized_global_heap(&mut file, first_addr, &[(1, b"one-a"), (3, b"one-c")]).unwrap();
    append_serialized_global_heap(&mut file, second_addr, &[(1, b"two-a"), (2, b"two-b")]).unwrap();

    let refs = [
        GlobalHeapRef {
            collection_addr: second_addr as u64,
            object_index: 2,
        },
        GlobalHeapRef {
            collection_addr: first_addr as u64,
            object_index: 3,
        },
        GlobalHeapRef {
            collection_addr: first_addr as u64,
            object_index: 1,
        },
        GlobalHeapRef {
            collection_addr: second_addr as u64,
            object_index: 1,
        },
        GlobalHeapRef {
            collection_addr: first_addr as u64,
            object_index: 3,
        },
    ];

    let mut reader = HdfReader::new(Cursor::new(file));
    let data = read_global_heap_objects_batched(&mut reader, &refs).unwrap();
    assert_eq!(
        data,
        vec![
            b"two-b".to_vec(),
            b"one-c".to_vec(),
            b"one-a".to_vec(),
            b"two-a".to_vec(),
            b"one-c".to_vec(),
        ]
    );
}

#[test]
fn t9a_global_heap_batched_missing_object_reports_reference() {
    let heap_addr = 32usize;
    let mut file = Vec::new();
    append_serialized_global_heap(&mut file, heap_addr, &[(1, b"present")]).unwrap();

    let refs = [GlobalHeapRef {
        collection_addr: heap_addr as u64,
        object_index: 9,
    }];

    let mut reader = HdfReader::new(Cursor::new(file));
    let err = read_global_heap_objects_batched(&mut reader, &refs).unwrap_err();
    assert!(
        err.to_string()
            .contains("global heap object 9 not found in collection at 0x20"),
        "unexpected error: {err}"
    );
}

#[test]
fn t9a_global_heap_vlen_strings_reuse_and_truncate_output_buffer() {
    let f = File::open("tests/data/hdf5_ref/vlen_string_cases.h5").unwrap();
    let ds = f.dataset("vlen_global_heap_edges").unwrap();

    let mut strings = vec![
        String::from("stale-first"),
        String::from("stale-second"),
        String::from("stale-third"),
        String::from("stale-extra"),
    ];
    ds.read_strings_into(&mut strings).unwrap();

    let long = format!("long-{}", "x".repeat(96));
    assert_eq!(strings, vec!["dup", "dup", long.as_str()]);
}

#[test]
fn t9a_global_heap_vlen_attr() {
    // The simple_v2.h5 has a vlen string attribute "test_attr"
    let f = File::open("tests/data/simple_v2.h5").unwrap();
    assert!(f.attr_exists("test_attr").unwrap());
}

#[test]
fn t9a_global_heap_deleted_objects_duplicate_ids_and_padding() {
    fn push_object(heap: &mut Vec<u8>, index: u16, data: &[u8]) {
        heap.extend_from_slice(&index.to_le_bytes());
        heap.extend_from_slice(&1u16.to_le_bytes());
        heap.extend_from_slice(&[0; 4]);
        heap.extend_from_slice(&(data.len() as u64).to_le_bytes());
        heap.extend_from_slice(data);
        let padded = (data.len() + 7) & !7;
        heap.extend(std::iter::repeat(0xa5).take(padded - data.len()));
    }
    fn push_free_object(heap: &mut Vec<u8>, body_len: usize) {
        let padded = (body_len + 7) & !7;
        let object_size = 16 + padded;
        heap.extend_from_slice(&0u16.to_le_bytes());
        heap.extend_from_slice(&0u16.to_le_bytes());
        heap.extend_from_slice(&[0; 4]);
        heap.extend_from_slice(&(object_size as u64).to_le_bytes());
        heap.extend(std::iter::repeat(0xa5).take(padded));
    }

    let mut heap = b"GCOL".to_vec();
    heap.push(1);
    heap.extend_from_slice(&[0; 3]);
    heap.extend_from_slice(&0u64.to_le_bytes());
    push_object(&mut heap, 2, b"abc");
    push_object(&mut heap, 3, b"padded!!");
    push_free_object(&mut heap, 8);
    let collection_size = heap.len() as u64;
    heap[8..16].copy_from_slice(&collection_size.to_le_bytes());

    let mut reader = HdfReader::new(Cursor::new(heap));
    let collection = GlobalHeapCollection::read_at(&mut reader, 0).unwrap();
    assert_eq!(collection.objects.len(), 2);
    assert_eq!(collection.get_object(0), None);
    assert_eq!(collection.get_object(2), Some(&b"abc"[..]));
    assert_eq!(collection.get_object(3), Some(&b"padded!!"[..]));

    let mut heap = b"GCOL".to_vec();
    heap.push(1);
    heap.extend_from_slice(&[0; 3]);
    heap.extend_from_slice(&0u64.to_le_bytes());
    push_object(&mut heap, 5, b"first");
    push_object(&mut heap, 5, b"second");
    let collection_size = heap.len() as u64;
    heap[8..16].copy_from_slice(&collection_size.to_le_bytes());
    let mut reader = HdfReader::new(Cursor::new(heap));
    let collection = GlobalHeapCollection::read_at(&mut reader, 0).unwrap();
    assert_eq!(collection.objects.len(), 2);
    assert_eq!(collection.get_object(5), Some(&b"first"[..]));

    let mut heap = b"GCOL".to_vec();
    heap.push(1);
    heap.extend_from_slice(&[0; 3]);
    heap.extend_from_slice(&0u64.to_le_bytes());
    push_free_object(&mut heap, 8);
    push_object(&mut heap, 9, b"after-free");
    let collection_size = heap.len() as u64;
    heap[8..16].copy_from_slice(&collection_size.to_le_bytes());
    let mut reader = HdfReader::new(Cursor::new(heap));
    let collection = GlobalHeapCollection::read_at(&mut reader, 0).unwrap();
    assert_eq!(collection.objects.len(), 1);
    assert_eq!(collection.get_object(9), Some(&b"after-free"[..]));
}

#[test]
fn t9a_global_heap_read_object_skips_deleted_and_padding() {
    fn push_object(heap: &mut Vec<u8>, index: u16, data: &[u8]) {
        heap.extend_from_slice(&index.to_le_bytes());
        heap.extend_from_slice(&1u16.to_le_bytes());
        heap.extend_from_slice(&[0; 4]);
        heap.extend_from_slice(&(data.len() as u64).to_le_bytes());
        heap.extend_from_slice(data);
        let padded = (data.len() + 7) & !7;
        heap.extend(std::iter::repeat(0).take(padded - data.len()));
    }
    fn push_free_object(heap: &mut Vec<u8>, body_len: usize) {
        let padded = (body_len + 7) & !7;
        let object_size = 16 + padded;
        heap.extend_from_slice(&0u16.to_le_bytes());
        heap.extend_from_slice(&0u16.to_le_bytes());
        heap.extend_from_slice(&[0; 4]);
        heap.extend_from_slice(&(object_size as u64).to_le_bytes());
        heap.extend(std::iter::repeat(0).take(padded));
    }

    let mut heap = b"GCOL".to_vec();
    heap.push(1);
    heap.extend_from_slice(&[0; 3]);
    heap.extend_from_slice(&0u64.to_le_bytes());
    push_object(&mut heap, 7, b"target");
    push_free_object(&mut heap, 4);
    let collection_size = heap.len() as u64;
    heap[8..16].copy_from_slice(&collection_size.to_le_bytes());

    let mut reader = HdfReader::new(Cursor::new(heap));
    let mut data = Vec::new();
    read_global_heap_object_into(
        &mut reader,
        &GlobalHeapRef {
            collection_addr: 0,
            object_index: 7,
        },
        &mut data,
    )
    .unwrap();
    assert_eq!(data, b"target");
}

// T9b: Local heap (v1 group name storage)

#[test]
fn t9b_local_heap_names() {
    let f = File::open("tests/data/simple_v0.h5").unwrap();
    let (_count, [has_data, has_group1]) = file_member_summary(&f, ["data", "group1"]).unwrap();
    // Names come from the local heap
    assert!(has_data);
    assert!(has_group1);
}

#[test]
fn t9b_local_heap_large_group() {
    // datasets_v0.h5 has more members
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let (count, []) = file_member_summary(&f, []).unwrap();
    assert!(count >= 4); // float64_1d, int32_1d, scalar, int8_2d, chunked
}

// T9c: Fractal heap (dense link/attr storage)

#[test]
fn t9c_fractal_heap_dense_links() {
    let f = File::open("tests/data/dense_links.h5").unwrap();
    let (count, []) = file_member_summary(&f, []).unwrap();
    assert_eq!(count, 20);
}

#[test]
fn t9c_fractal_heap_modern_dense_links() {
    let f = File::open("tests/data/hdf5_ref/fractal_heap_modern.h5").unwrap();
    let group = f.group("many_links").unwrap();
    let (count, [has_first, has_last]) =
        group_member_summary(&group, ["link_000", "link_079"]).unwrap();
    assert_eq!(count, 80);
    assert!(has_first);
    assert!(has_last);
}

#[test]
fn t9c_fractal_heap_indirect_growth_beyond_one_level() {
    let f = File::open("tests/data/hdf5_ref/dense_group_cases.h5").unwrap();
    let group = f.group("name_index_deep").unwrap();
    let (count, [has_first, has_middle, has_last]) =
        group_member_summary(&group, ["link_0000", "link_2048", "link_4095"]).unwrap();

    assert_eq!(count, 4096);
    assert!(has_first);
    assert!(has_middle);
    assert!(has_last);
    assert_eq!(
        group.member_type("link_4095").unwrap(),
        hdf5_pure_rust::hl::file::ObjectType::Dataset
    );
}

#[test]
fn t9c_fractal_heap_direct_and_indirect_checksum_corruption_fails() {
    fn heap() -> FractalHeapHeader {
        FractalHeapHeader {
            heap_addr: 0,
            heap_id_len: 3,
            io_filter_len: 0,
            flags: 0x02,
            max_managed_obj_size: 1024,
            table_width: 1,
            start_block_size: 32,
            max_direct_block_size: 32,
            max_heap_size: 8,
            start_root_rows: 0,
            root_block_addr: 0,
            current_root_rows: 0,
            num_managed_objects: 1,
            has_checksum: true,
            sizeof_addr: 8,
            sizeof_size: 8,
            huge_btree_addr: u64::MAX,
            root_direct_filtered_size: None,
            root_direct_filter_mask: 0,
            filter_pipeline: None,
        }
    }

    let mut direct_block = b"FHDB".to_vec();
    direct_block.push(0);
    direct_block.extend_from_slice(&0u64.to_le_bytes());
    direct_block.push(0);
    let checksum = checksum_metadata(&direct_block) ^ 0xffff_ffff;
    direct_block.extend_from_slice(&checksum.to_le_bytes());
    direct_block.extend_from_slice(b"payload");
    direct_block.resize(18 + 32, 0);
    let mut reader = HdfReader::new(Cursor::new(direct_block));
    let err = heap()
        .read_managed_object(&mut reader, &[0x00, 18, 7])
        .expect_err("direct block checksum corruption should fail");
    assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));

    let mut indirect_block = b"FHIB".to_vec();
    indirect_block.push(0);
    indirect_block.extend_from_slice(&0u64.to_le_bytes());
    indirect_block.push(0);
    indirect_block.extend_from_slice(&64u64.to_le_bytes());
    let checksum = checksum_metadata(&indirect_block) ^ 0xffff_ffff;
    indirect_block.extend_from_slice(&checksum.to_le_bytes());

    let mut file_bytes = indirect_block;
    file_bytes.resize(64, 0);
    let mut direct_block = b"FHDB".to_vec();
    direct_block.push(0);
    direct_block.extend_from_slice(&0u64.to_le_bytes());
    direct_block.push(0);
    let mut checksum_input = direct_block.clone();
    checksum_input.extend_from_slice(&0u32.to_le_bytes());
    checksum_input.extend_from_slice(b"payload");
    checksum_input.resize(18 + 32, 0);
    let checksum = checksum_metadata(&checksum_input);
    direct_block.extend_from_slice(&checksum.to_le_bytes());
    direct_block.extend_from_slice(b"payload");
    direct_block.resize(18 + 32, 0);
    file_bytes.extend_from_slice(&direct_block);

    let mut heap = heap();
    heap.current_root_rows = 1;
    heap.root_block_addr = 0;
    let mut reader = HdfReader::new(Cursor::new(file_bytes));
    let err = heap
        .read_managed_object(&mut reader, &[0x00, 18, 7])
        .expect_err("indirect block checksum corruption should fail");
    assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
}

#[test]
fn t9c_fractal_heap_dense_attrs() {
    // dense_attrs.h5 has the "data" dataset via inline link
    let f = File::open("tests/data/dense_attrs.h5").unwrap();
    let (_count, [has_data]) = file_member_summary(&f, ["data"]).unwrap();
    assert!(has_data);
}

// T9d: V2 B-tree (used for dense link name index)

#[test]
fn t9d_v2_btree_link_lookup() {
    let f = File::open("tests/data/dense_links.h5").unwrap();
    // The links are indexed via v2 B-tree + fractal heap
    let root = f.root_group().unwrap();
    // Can find specific groups by name
    let g = root.open_group("group_10").unwrap();
    assert!(g.is_empty().unwrap());
}

// T9e/f: Chunk index structures (tested via dataset reads)

#[test]
fn t9ef_btree_v1_chunk_index() {
    // btree_idx_1_6 and btree_idx_1_8 from C test suite
    let f = File::open("tests/data/hdf5_ref/btree_idx_1_6.h5").unwrap();
    let (count, []) = file_member_summary(&f, []).unwrap();
    println!("btree_idx_1_6 member count: {count}");
    // Just verify it opens and lists without error
    assert!(count > 0);
}

#[test]
fn t9ef_btree_v1_chunk_index_18() {
    let f = File::open("tests/data/hdf5_ref/btree_idx_1_8.h5").unwrap();
    let (count, []) = file_member_summary(&f, []).unwrap();
    println!("btree_idx_1_8 member count: {count}");
    assert!(count > 0);
}

#[test]
fn t9ef_btree_v1_chunk_index_3d_coordinates() {
    let f = File::open("tests/data/hdf5_ref/v1_btree_3d_chunks.h5").unwrap();
    let ds = f.dataset("btree_v1_3d").unwrap();
    assert_dataset_shape(&ds, &[4, 5, 6]);

    let vals: Vec<i32> = dataset_values(&ds).unwrap();
    assert_eq!(vals.len(), 4 * 5 * 6);
    assert_eq!(vals[0], 0);
    assert_eq!(vals[5], 5);
    assert_eq!(vals[6], 6);
    assert_eq!(vals[37], 37);
    assert_eq!(vals[119], 119);
}

#[test]
fn t9ef_btree_v1_sparse_nonmonotonic_chunks() {
    let f = File::open("tests/data/hdf5_ref/v1_btree_sparse_nonmonotonic.h5").unwrap();
    let ds = f.dataset("btree_v1_sparse_nonmonotonic").unwrap();
    assert_dataset_shape(&ds, &[6, 6]);

    let vals: Vec<i32> = dataset_values(&ds).unwrap();
    let mut expected = vec![-9; 6 * 6];
    expected[0] = 0;
    expected[1] = 1;
    expected[6] = 10;
    expected[7] = 11;
    expected[14] = 22;
    expected[15] = 23;
    expected[20] = 32;
    expected[21] = 33;
    expected[28] = 44;
    expected[29] = 45;
    expected[34] = 54;
    expected[35] = 55;
    assert_eq!(vals, expected);
}

#[test]
fn t9ef_non_default_heap_sizes() {
    let f = File::open("tests/data/hdf5_ref/tsizeslheap.h5").unwrap();
    let (count, []) = file_member_summary(&f, []).unwrap();
    println!("tsizeslheap member count: {count}");
}
