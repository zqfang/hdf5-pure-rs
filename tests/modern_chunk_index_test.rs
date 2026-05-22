use std::fs;
use std::path::{Path, PathBuf};

use hdf5_pure_rust::format::checksum::checksum_metadata;
use hdf5_pure_rust::{Dataset, File, H5Type, IntoSelection};

fn dataset_len(ds: &Dataset) -> hdf5_pure_rust::Result<usize> {
    Ok(usize::try_from(ds.size()?).unwrap())
}

fn read_dataset_into<T: H5Type>(ds: &Dataset, values: &mut [T]) -> hdf5_pure_rust::Result<()> {
    ds.read_into(values)
}

fn assert_dataset_values<T>(ds: &Dataset, expected: &[T]) -> hdf5_pure_rust::Result<()>
where
    T: H5Type + Default + PartialEq + std::fmt::Debug,
{
    let mut values = (0..dataset_len(ds)?)
        .map(|_| T::default())
        .collect::<Vec<_>>();
    read_dataset_into(ds, &mut values)?;
    assert_eq!(values, expected);
    Ok(())
}

fn assert_shape(ds: &Dataset, expected: &[u64]) {
    let space = ds.space().unwrap();
    assert_eq!(space.shape(), expected);
}

fn read_dataset_slice_into<T: H5Type, S: IntoSelection>(
    ds: &Dataset,
    sel: S,
    values: &mut [T],
) -> hdf5_pure_rust::Result<()> {
    ds.read_slice_into(sel, values)
}

fn read_dataset_field_into<T: H5Type>(
    ds: &Dataset,
    field_name: &str,
    values: &mut [T],
) -> hdf5_pure_rust::Result<()> {
    ds.read_field_into(field_name, values)
}

fn corrupt_metadata_checksum(src: impl AsRef<Path>, magic: &[u8]) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let dst = dir.path().join("corrupt_checksum.h5");
    let mut bytes = fs::read(src).unwrap();
    let magic_pos = bytes
        .windows(magic.len())
        .position(|window| window == magic)
        .expect("metadata magic should be present in fixture");

    let search_end = (magic_pos + 256).min(bytes.len().saturating_sub(4));
    for checksum_pos in magic_pos + magic.len()..search_end {
        let stored = u32::from_le_bytes(bytes[checksum_pos..checksum_pos + 4].try_into().unwrap());
        let computed = checksum_metadata(&bytes[magic_pos..checksum_pos]);
        if stored == computed {
            bytes[checksum_pos] ^= 0x01;
            fs::write(&dst, bytes).unwrap();
            return (dir, dst);
        }
    }

    panic!("metadata checksum field should be discoverable");
}

fn patch_filtered_fixed_array_to_implicit(src: impl AsRef<Path>) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let dst = dir.path().join("filtered_implicit_chunk_index.h5");
    let mut bytes = fs::read(src).unwrap();

    // Data layout v4 chunked payload for the filtered fixed-array fixture:
    // version=4, class=chunked, flags=0, rank=2, dim-bytes=1,
    // chunk dims=[16, 2], index type=3 (fixed array). Rewriting the index type
    // to 2 creates the malformed filtered implicit-index case.
    let layout_pattern = [4u8, 2, 0, 2, 1, 16, 2, 3];
    let layout_pos = bytes
        .windows(layout_pattern.len())
        .position(|window| window == layout_pattern)
        .expect("filtered fixed-array layout should be present");
    let index_type_pos = layout_pos + layout_pattern.len() - 1;

    let ohdr_pos = bytes[..layout_pos]
        .windows(4)
        .rposition(|window| window == b"OHDR")
        .expect("object header should precede layout message");
    let checksum_search_end = (layout_pos + 512).min(bytes.len().saturating_sub(4));
    let checksum_pos = (layout_pos..checksum_search_end)
        .find(|&pos| {
            let stored = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
            stored == checksum_metadata(&bytes[ohdr_pos..pos])
        })
        .expect("object-header checksum should be discoverable");

    bytes[index_type_pos] = 2;
    let checksum = checksum_metadata(&bytes[ohdr_pos..checksum_pos]);
    bytes[checksum_pos..checksum_pos + 4].copy_from_slice(&checksum.to_le_bytes());
    fs::write(&dst, bytes).unwrap();
    (dir, dst)
}

fn assert_chunk_index_checksum_error(path: &Path, dataset: &str, expected: &str) {
    let f = File::open(path).unwrap();
    let ds = f.dataset(dataset).unwrap();
    let mut values = vec![0i32; dataset_len(&ds).unwrap()];
    let err =
        read_dataset_into(&ds, &mut values).expect_err("corrupt chunk index checksum should fail");
    assert!(
        err.to_string().contains(expected),
        "unexpected error: {err}"
    );
}

#[test]
fn test_v4_fixed_array_chunks_read() {
    let f = File::open("tests/data/hdf5_ref/v4_fixed_array_chunks.h5").unwrap();
    assert_dataset_values::<i32>(
        &f.dataset("fixed_array").unwrap(),
        &(0..100).collect::<Vec<_>>(),
    )
    .unwrap();
}

#[test]
fn test_v4_fixed_array_deflate_parallel_threshold_tail_read() {
    const CHUNK: usize = 2048;
    const LEN: usize = CHUNK * 8 + 17;

    let f = File::open("tests/data/hdf5_ref/v4_fixed_array_deflate_parallel_threshold_tail.h5")
        .unwrap();
    let ds = f
        .dataset("fixed_array_deflate_parallel_threshold_tail")
        .unwrap();
    assert_shape(&ds, &[LEN as u64]);

    let expected: Vec<i32> = (0..LEN).map(|value| value as i32 * 5 - 11).collect();
    assert_dataset_values::<i32>(&ds, &expected).unwrap();

    let mut boundary = vec![0i32; 7];
    read_dataset_slice_into(&ds, CHUNK - 3..CHUNK + 4, &mut boundary).unwrap();
    assert_eq!(boundary, expected[CHUNK - 3..CHUNK + 4]);

    let mut tail = vec![0i32; 20];
    read_dataset_slice_into(&ds, LEN - 20..LEN, &mut tail).unwrap();
    assert_eq!(tail, expected[LEN - 20..LEN]);
}

#[test]
fn test_v4_fixed_array_deflate_mask_parallel_fallback_read() {
    const CHUNK: usize = 2048;
    const LEN: usize = CHUNK * 8;

    let f =
        File::open("tests/data/hdf5_ref/v4_fixed_array_deflate_mask_parallel_fallback.h5").unwrap();
    let ds = f
        .dataset("fixed_array_deflate_mask_parallel_fallback")
        .unwrap();
    assert_shape(&ds, &[LEN as u64]);

    let expected: Vec<i32> = (0..LEN).map(|value| value as i32 * 2 - 31).collect();
    assert_dataset_values::<i32>(&ds, &expected).unwrap();

    let mut boundary = vec![0i32; 7];
    read_dataset_slice_into(&ds, CHUNK - 3..CHUNK + 4, &mut boundary).unwrap();
    assert_eq!(boundary, expected[CHUNK - 3..CHUNK + 4]);
}

#[test]
fn test_v4_fixed_array_3d_edge_chunks_read() {
    let f = File::open("tests/data/hdf5_ref/v4_fixed_array_3d_edges.h5").unwrap();
    let ds = f.dataset("fixed_array_3d_edges").unwrap();
    assert_shape(&ds, &[5, 7, 4]);

    let mut vals = vec![0i32; dataset_len(&ds).unwrap()];
    read_dataset_into(&ds, &mut vals).unwrap();
    assert_eq!(vals.len(), 5 * 7 * 4);
    assert_eq!(vals[0], 0);
    assert_eq!(vals[3], 3);
    assert_eq!(vals[4], 4);
    assert_eq!(vals[27], 27);
    assert_eq!(vals[28], 28);
    assert_eq!(vals[139], 139);
}

#[test]
fn test_v4_fixed_array_header_checksum_corruption_fails() {
    let (_dir, path) =
        corrupt_metadata_checksum("tests/data/hdf5_ref/v4_fixed_array_chunks.h5", b"FAHD");
    assert_chunk_index_checksum_error(&path, "fixed_array", "fixed array header checksum mismatch");
}

#[test]
fn test_v4_paged_fixed_array_chunks_read() {
    let f = File::open("tests/data/hdf5_ref/v4_fixed_array_paged_chunks.h5").unwrap();
    let ds = f.dataset("fixed_array_paged").unwrap();
    let mut vals = vec![0i32; dataset_len(&ds).unwrap()];
    read_dataset_into(&ds, &mut vals).unwrap();
    assert_eq!(vals.len(), 4096);
    assert_eq!(vals[0], 0);
    assert_eq!(vals[1024], 1024);
    assert_eq!(vals[4095], 4095);
}

#[test]
fn test_v4_paged_fixed_array_absent_pages_use_fill_value() {
    let f = File::open("tests/data/hdf5_ref/v4_fixed_array_paged_sparse.h5").unwrap();
    let ds = f.dataset("fixed_array_paged_sparse").unwrap();
    let mut vals = vec![0i32; dataset_len(&ds).unwrap()];
    read_dataset_into(&ds, &mut vals).unwrap();

    assert_eq!(vals.len(), 4096);
    assert_eq!(vals[0], 11);
    assert_eq!(vals[1], -3);
    assert_eq!(vals[2047], -3);
    assert_eq!(vals[2048], 22);
    assert_eq!(vals[4094], -3);
    assert_eq!(vals[4095], 33);
}

#[test]
fn test_v4_filtered_fixed_array_chunks_read() {
    let f = File::open("tests/data/hdf5_ref/v4_filtered_chunked.h5").unwrap();
    assert_dataset_values::<i16>(
        &f.dataset("filtered_chunked").unwrap(),
        &(0..64).collect::<Vec<_>>(),
    )
    .unwrap();
}

#[test]
fn test_filtered_implicit_chunk_index_fixture_is_rejected() {
    let (_dir, path) =
        patch_filtered_fixed_array_to_implicit("tests/data/hdf5_ref/v4_filtered_chunked.h5");
    let f = File::open(&path).unwrap();
    let ds = f.dataset("filtered_chunked").unwrap();
    let mut values = vec![0i16; dataset_len(&ds).unwrap()];
    let err = read_dataset_into(&ds, &mut values)
        .expect_err("filtered implicit chunk index should be rejected");
    assert!(
        err.to_string()
            .contains("v4 implicit chunk index with filters"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_filtered_implicit_chunk_index_slice_read_is_rejected() {
    let (_dir, path) =
        patch_filtered_fixed_array_to_implicit("tests/data/hdf5_ref/v4_filtered_chunked.h5");
    let f = File::open(&path).unwrap();
    let ds = f.dataset("filtered_chunked").unwrap();
    let mut values = vec![0i16; 4];
    let err = read_dataset_slice_into(&ds, 0..4, &mut values)
        .expect_err("filtered implicit chunk index slice read should be rejected");
    assert!(
        err.to_string()
            .contains("v4 implicit chunk index with filters"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_v4_implicit_2d_edge_chunks_read() {
    let f = File::open("tests/data/hdf5_ref/v4_implicit_2d_edge_chunks.h5").unwrap();
    let ds = f.dataset("implicit_2d_edge").unwrap();
    assert_shape(&ds, &[5, 7]);

    let mut vals = vec![0i32; dataset_len(&ds).unwrap()];
    read_dataset_into(&ds, &mut vals).unwrap();
    assert_eq!(vals.len(), 5 * 7);
    assert_eq!(vals[0], 0);
    assert_eq!(vals[6], 6);
    assert_eq!(vals[7], 7);
    assert_eq!(vals[27], 27);
    assert_eq!(vals[34], 34);
}

#[test]
fn test_sparse_chunked_fill_value_read() {
    let f = File::open("tests/data/hdf5_ref/sparse_chunked_fill_value.h5").unwrap();
    let ds = f.dataset("sparse_chunked_fill").unwrap();
    let mut vals = vec![0i32; dataset_len(&ds).unwrap()];
    read_dataset_into(&ds, &mut vals).unwrap();

    let mut expected = vec![-7; 4 * 6];
    for row in 0..2 {
        for col in 0..3 {
            expected[row * 6 + col] = (row * 3 + col) as i32;
        }
    }
    assert_eq!(vals, expected);
}

#[test]
fn test_sparse_chunked_partial_read_combines_present_chunks_and_fill_value() {
    let f = File::open("tests/data/hdf5_ref/sparse_chunked_fill_value.h5").unwrap();
    let ds = f.dataset("sparse_chunked_fill").unwrap();
    let mut vals = vec![0i32; 12];
    read_dataset_slice_into(&ds, (1..4, 2..6), &mut vals).unwrap();

    assert_eq!(
        vals,
        vec![
            5, -7, -7, -7, //
            -7, -7, -7, -7, //
            -7, -7, -7, -7,
        ]
    );
}

#[test]
fn test_filtered_chunk_mask_skips_unapplied_filters() {
    let f = File::open("tests/data/hdf5_ref/filtered_chunk_filter_mask.h5").unwrap();
    let ds = f.dataset("filtered_chunk_filter_mask").unwrap();
    let mut vals = vec![0i32; dataset_len(&ds).unwrap()];
    read_dataset_into(&ds, &mut vals).unwrap();

    let mut expected = vec![-7; 4 * 6];
    for row in 0..2 {
        for col in 0..3 {
            expected[row * 6 + col] = (row * 3 + col) as i32;
            expected[(row + 2) * 6 + col + 3] = (row * 3 + col + 100) as i32;
        }
    }
    assert_eq!(vals, expected);
}

#[test]
fn test_filtered_single_chunk_mask_skips_unapplied_filters() {
    let f = File::open("tests/data/hdf5_ref/filtered_single_chunk_filter_mask.h5").unwrap();
    let ds = f.dataset("filtered_single_chunk_filter_mask").unwrap();
    assert_shape(&ds, &[2, 3]);

    assert_dataset_values::<i32>(&ds, &(0..6).collect::<Vec<_>>()).unwrap();
}

#[test]
fn test_filtered_chunk_mask_skips_middle_filter_in_pipeline() {
    let f = File::open("tests/data/hdf5_ref/filtered_middle_filter_mask.h5").unwrap();
    let ds = f.dataset("filtered_middle_filter_mask").unwrap();
    assert_shape(&ds, &[6]);

    let mut vals = vec![0i32; 3];
    read_dataset_slice_into(&ds, 2..5, &mut vals).unwrap();
    assert_eq!(vals, vec![2, 3, 4]);
}

#[test]
fn test_multi_filter_scaleoffset_shuffle_deflate_order() {
    let f = File::open("tests/data/hdf5_ref/multi_filter_orders.h5").unwrap();
    assert_dataset_values::<i32>(
        &f.dataset("scaleoffset_shuffle_deflate").unwrap(),
        &(0..32).collect::<Vec<_>>(),
    )
    .unwrap();
}

#[test]
fn test_multi_filter_shuffle_deflate_fletcher_order() {
    let f = File::open("tests/data/hdf5_ref/multi_filter_orders.h5").unwrap();
    assert_dataset_values::<i32>(
        &f.dataset("shuffle_deflate_fletcher").unwrap(),
        &(0..32).collect::<Vec<_>>(),
    )
    .unwrap();
}

#[test]
fn test_multi_filter_nbit_deflate_order() {
    let f = File::open("tests/data/hdf5_ref/multi_filter_orders.h5").unwrap();
    assert_dataset_values::<i32>(
        &f.dataset("nbit_deflate").unwrap(),
        &(0..64).collect::<Vec<_>>(),
    )
    .unwrap();
}

#[test]
fn test_fletcher32_corruption_fails_for_uncompressed_chunk() {
    let f = File::open("tests/data/hdf5_ref/fletcher32_corrupt.h5").unwrap();
    let ds = f.dataset("fletcher32_corrupt").unwrap();
    let mut values = vec![0i32; dataset_len(&ds).unwrap()];
    let err = read_dataset_into(&ds, &mut values).expect_err("bad Fletcher32 checksum should fail");

    assert!(
        err.to_string().contains("fletcher32 checksum mismatch"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_fletcher32_corruption_fails_for_deflate_chunk() {
    let f = File::open("tests/data/hdf5_ref/fletcher32_corrupt.h5").unwrap();
    let ds = f.dataset("deflate_fletcher32_corrupt").unwrap();
    let mut values = vec![0i32; dataset_len(&ds).unwrap()];
    let err = read_dataset_into(&ds, &mut values)
        .expect_err("bad Fletcher32 checksum should fail before deflate");

    assert!(
        err.to_string().contains("fletcher32 checksum mismatch"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_v4_extensible_array_chunks_read() {
    let f = File::open("tests/data/hdf5_ref/v4_extensible_array_chunks.h5").unwrap();
    assert_dataset_values::<f64>(
        &f.dataset("extensible_array").unwrap(),
        &(0..80).map(|v| v as f64).collect::<Vec<_>>(),
    )
    .unwrap();
}

#[test]
fn test_v4_extensible_array_deflate_parallel_threshold_tail_read() {
    const CHUNK: usize = 2048;
    const LEN: usize = CHUNK * 8 + 17;

    let f =
        File::open("tests/data/hdf5_ref/v4_extensible_array_deflate_parallel_threshold_tail.h5")
            .unwrap();
    let ds = f
        .dataset("extensible_array_deflate_parallel_threshold_tail")
        .unwrap();
    assert_shape(&ds, &[LEN as u64]);
    assert!(ds.space().unwrap().is_resizable());

    let expected: Vec<i32> = (0..LEN).map(|value| value as i32 * 7 - 13).collect();
    assert_dataset_values::<i32>(&ds, &expected).unwrap();

    let mut boundary = vec![0i32; 7];
    read_dataset_slice_into(&ds, CHUNK - 3..CHUNK + 4, &mut boundary).unwrap();
    assert_eq!(boundary, expected[CHUNK - 3..CHUNK + 4]);

    let mut tail = vec![0i32; 20];
    read_dataset_slice_into(&ds, LEN - 20..LEN, &mut tail).unwrap();
    assert_eq!(tail, expected[LEN - 20..LEN]);
}

#[test]
fn test_v4_extensible_array_deflate_mask_parallel_fallback_read() {
    const CHUNK: usize = 2048;
    const LEN: usize = CHUNK * 8;

    let f = File::open("tests/data/hdf5_ref/v4_extensible_array_deflate_mask_parallel_fallback.h5")
        .unwrap();
    let ds = f
        .dataset("extensible_array_deflate_mask_parallel_fallback")
        .unwrap();
    assert_shape(&ds, &[LEN as u64]);
    assert!(ds.space().unwrap().is_resizable());

    let expected: Vec<i32> = (0..LEN).map(|value| value as i32 * 11 - 19).collect();
    assert_dataset_values::<i32>(&ds, &expected).unwrap();

    let mut boundary = vec![0i32; 7];
    read_dataset_slice_into(&ds, CHUNK - 3..CHUNK + 4, &mut boundary).unwrap();
    assert_eq!(boundary, expected[CHUNK - 3..CHUNK + 4]);
}

#[test]
fn test_v4_extensible_array_2d_unlimited_edge_chunks_read() {
    let f = File::open("tests/data/hdf5_ref/v4_extensible_array_2d_unlimited_edges.h5").unwrap();
    let ds = f.dataset("extensible_array_2d_unlimited_edges").unwrap();
    assert_shape(&ds, &[5, 7]);
    assert!(ds.space().unwrap().is_resizable());

    let mut vals = vec![0i32; dataset_len(&ds).unwrap()];
    read_dataset_into(&ds, &mut vals).unwrap();
    assert_eq!(vals.len(), 5 * 7);
    assert_eq!(vals[0], 0);
    assert_eq!(vals[6], 6);
    assert_eq!(vals[7], 7);
    assert_eq!(vals[27], 27);
    assert_eq!(vals[34], 34);
}

#[test]
fn test_v4_extensible_array_header_checksum_corruption_fails() {
    let (_dir, path) =
        corrupt_metadata_checksum("tests/data/hdf5_ref/v4_extensible_array_chunks.h5", b"EAHD");
    let f = File::open(&path).unwrap();
    let ds = f.dataset("extensible_array").unwrap();
    let mut values = vec![0.0f64; dataset_len(&ds).unwrap()];
    let err =
        read_dataset_into(&ds, &mut values).expect_err("corrupt chunk index checksum should fail");
    assert!(
        err.to_string()
            .contains("extensible array header checksum mismatch"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_v4_extensible_array_spillover_chunks_read() {
    let f = File::open("tests/data/hdf5_ref/v4_extensible_array_spillover.h5").unwrap();
    let ds = f.dataset("extensible_array_spillover").unwrap();
    let mut vals = vec![0.0f64; dataset_len(&ds).unwrap()];
    read_dataset_into(&ds, &mut vals).unwrap();
    assert_eq!(vals.len(), 4096);
    assert_eq!(vals[0], 0.0);
    assert_eq!(vals[8], 8.0);
    assert_eq!(vals[4095], 4095.0);
}

#[test]
fn test_v4_extensible_array_sparse_transition_chunks_read() {
    let f = File::open("tests/data/hdf5_ref/v4_extensible_array_sparse_transitions.h5").unwrap();
    let ds = f.dataset("extensible_array_sparse_transitions").unwrap();
    let mut vals = vec![0i32; dataset_len(&ds).unwrap()];
    read_dataset_into(&ds, &mut vals).unwrap();

    assert_eq!(vals.len(), 4096);
    for idx in [
        0usize, 1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 511, 512, 1023, 1024, 2047,
        2048, 4095,
    ] {
        assert_eq!(vals[idx], idx as i32);
    }
    for idx in [2usize, 9, 33, 65, 129, 257, 513, 1025, 2049, 4094] {
        assert_eq!(vals[idx], -4);
    }
}

#[test]
fn test_v4_btree2_chunks_read() {
    let f = File::open("tests/data/hdf5_ref/v4_btree2_chunks.h5").unwrap();
    assert_dataset_values::<i32>(
        &f.dataset("btree_v2").unwrap(),
        &(0..64).collect::<Vec<_>>(),
    )
    .unwrap();
}

#[test]
fn test_v4_btree2_header_checksum_corruption_fails() {
    let (_dir, path) =
        corrupt_metadata_checksum("tests/data/hdf5_ref/v4_btree2_chunks.h5", b"BTHD");
    assert_chunk_index_checksum_error(&path, "btree_v2", "v2 B-tree header checksum mismatch");
}

#[test]
fn test_v4_btree2_internal_chunks_read() {
    let f = File::open("tests/data/hdf5_ref/v4_btree2_internal_chunks.h5").unwrap();
    let ds = f.dataset("btree_v2_internal").unwrap();
    let mut vals = vec![0i32; dataset_len(&ds).unwrap()];
    read_dataset_into(&ds, &mut vals).unwrap();
    assert_eq!(vals.len(), 80 * 80);
    assert_eq!(vals[0], 0);
    assert_eq!(vals[79], 79);
    assert_eq!(vals[80], 80);
    assert_eq!(vals[6399], 6399);
}

#[test]
fn test_v4_btree2_deep_internal_chunks_read() {
    let f = File::open("tests/data/hdf5_ref/v4_btree2_deep_internal_chunks.h5").unwrap();
    let ds = f.dataset("btree_v2_deep_internal").unwrap();
    let mut vals = vec![0i32; dataset_len(&ds).unwrap()];
    read_dataset_into(&ds, &mut vals).unwrap();
    assert_eq!(vals.len(), 160 * 160);
    assert_eq!(vals[0], 0);
    assert_eq!(vals[159], 159);
    assert_eq!(vals[160], 160);
    assert_eq!(vals[25_599], 25_599);
}

#[test]
fn test_v4_btree2_filtered_chunk_mask_read() {
    let f = File::open("tests/data/hdf5_ref/v4_btree2_filtered_mask.h5").unwrap();
    let ds = f.dataset("btree_v2_filtered_mask").unwrap();
    let mut vals = vec![0i32; dataset_len(&ds).unwrap()];
    read_dataset_into(&ds, &mut vals).unwrap();

    assert_eq!(
        vals,
        vec![
            1, 2, -1, -1, //
            5, 6, -1, -1, //
            -1, -1, 11, 12, //
            -1, -1, 15, 16,
        ]
    );
}

#[test]
fn test_nbit_filter_i32_read() {
    let f = File::open("tests/data/hdf5_ref/nbit_filter_i32.h5").unwrap();
    assert_dataset_values::<i32>(
        &f.dataset("nbit_i32").unwrap(),
        &(0..100).collect::<Vec<_>>(),
    )
    .unwrap();
}

#[test]
fn test_nbit_filter_big_endian_i32_read() {
    let f = File::open("tests/data/hdf5_ref/nbit_filter_be_i32.h5").unwrap();
    assert_dataset_values::<i32>(
        &f.dataset("nbit_be_i32").unwrap(),
        &(0..100).collect::<Vec<_>>(),
    )
    .unwrap();
}

#[test]
fn test_nbit_filter_signed_unsigned_and_float_parity_vectors() {
    let f = File::open("tests/data/hdf5_ref/nbit_parity_vectors.h5").unwrap();

    assert_dataset_values::<i16>(
        &f.dataset("nbit_i16_signed").unwrap(),
        &[-32768, -257, -1, 0, 1, 255, 1024, 32767],
    )
    .unwrap();

    assert_dataset_values::<u16>(
        &f.dataset("nbit_u16_unsigned").unwrap(),
        &[0, 1, 255, 256, 1024, 32768, 65535],
    )
    .unwrap();

    let ds = f.dataset("nbit_f32").unwrap();
    let mut floats = vec![0.0f32; dataset_len(&ds).unwrap()];
    read_dataset_into(&ds, &mut floats).unwrap();
    let expected = [-0.0f32, 1.5, -2.25, 123.5, f32::INFINITY, f32::NEG_INFINITY];
    assert_eq!(
        floats
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
}

#[test]
fn test_nbit_filter_compound_member_parity_vectors() {
    let f = File::open("tests/data/hdf5_ref/nbit_parity_vectors.h5").unwrap();
    let ds = f.dataset("nbit_compound_members").unwrap();

    let mut codes = vec![0i16; dataset_len(&ds).unwrap()];
    read_dataset_field_into(&ds, "code", &mut codes).unwrap();
    assert_eq!(codes, vec![-7, 12, -1024]);

    let mut counts = vec![0u16; dataset_len(&ds).unwrap()];
    read_dataset_field_into(&ds, "count", &mut counts).unwrap();
    assert_eq!(counts, vec![3, 65530, 42]);

    let mut scores = vec![0.0f32; dataset_len(&ds).unwrap()];
    read_dataset_field_into(&ds, "score", &mut scores).unwrap();
    assert_eq!(
        scores
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        [1.25f32, -4.5, 32.0]
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
}

#[test]
fn test_scaleoffset_filter_i32_read() {
    let f = File::open("tests/data/hdf5_ref/scaleoffset_filter_i32.h5").unwrap();
    assert_dataset_values::<i32>(
        &f.dataset("scaleoffset_i32").unwrap(),
        &(0..100).collect::<Vec<_>>(),
    )
    .unwrap();
}

#[test]
fn test_scaleoffset_filter_big_endian_i32_read() {
    let f = File::open("tests/data/hdf5_ref/scaleoffset_filter_be_i32.h5").unwrap();
    assert_dataset_values::<i32>(
        &f.dataset("scaleoffset_be_i32").unwrap(),
        &(0..100).collect::<Vec<_>>(),
    )
    .unwrap();
}

#[test]
fn test_scaleoffset_filter_f32_read() {
    let f = File::open("tests/data/hdf5_ref/scaleoffset_filter_i32.h5").unwrap();
    let ds = f.dataset("scaleoffset_f32").unwrap();
    let mut vals = vec![0.0f32; dataset_len(&ds).unwrap()];
    read_dataset_into(&ds, &mut vals).unwrap();
    let expected: Vec<f32> = (0..40).map(|v| v as f32 / 10.0 + 1.25).collect();
    for (actual, expected) in vals.iter().zip(expected) {
        assert!((*actual - expected).abs() < 0.011);
    }
}

#[test]
fn test_scaleoffset_integer_parity_vectors() {
    let f = File::open("tests/data/hdf5_ref/scaleoffset_parity_vectors.h5").unwrap();

    assert_dataset_values::<i16>(
        &f.dataset("scaleoffset_i16_signed").unwrap(),
        &[-120, -17, -1, 0, 5, 63, 127],
    )
    .unwrap();

    assert_dataset_values::<u16>(
        &f.dataset("scaleoffset_u16_minbits").unwrap(),
        &[1000, 1001, 1003, 1007, 1015, 1023],
    )
    .unwrap();

    assert_dataset_values::<i32>(
        &f.dataset("scaleoffset_i32_zero_minbits").unwrap(),
        &[-42; 8],
    )
    .unwrap();
}

#[test]
fn test_scaleoffset_float_parity_vectors() {
    let f = File::open("tests/data/hdf5_ref/scaleoffset_parity_vectors.h5").unwrap();

    let ds = f.dataset("scaleoffset_f32_dscale").unwrap();
    let mut f32_vals = vec![0.0f32; dataset_len(&ds).unwrap()];
    read_dataset_into(&ds, &mut f32_vals).unwrap();
    for (actual, expected) in f32_vals.iter().zip([-1.25, -0.5, 0.0, 1.25, 3.5]) {
        assert!((*actual - expected).abs() < 0.011);
    }

    let ds = f.dataset("scaleoffset_f64_dscale").unwrap();
    let mut f64_vals = vec![0.0f64; dataset_len(&ds).unwrap()];
    read_dataset_into(&ds, &mut f64_vals).unwrap();
    for (actual, expected) in f64_vals.iter().zip([-100.125, -1.5, 0.25, 12.75, 2048.5]) {
        assert!((*actual - expected).abs() < 0.0011);
    }
}
