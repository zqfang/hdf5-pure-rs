use hdf5_pure_rust::hl::plist::dataset_create::VirtualSelectionInfo;
use hdf5_pure_rust::hl::selection::{HyperslabDim, Selection, SliceInfo};
use hdf5_pure_rust::{
    Dataset, DatasetAccess, Error, File, H5Type, VdsMissingSourcePolicy, VdsView,
};
use std::sync::{Mutex, MutexGuard, OnceLock};

#[derive(Clone, Copy, Debug, Default)]
#[allow(dead_code)]
struct ThreeBytes([u8; 3]);

unsafe impl hdf5_pure_rust::H5Type for ThreeBytes {
    fn type_size() -> usize {
        3
    }
}

fn vds_env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn fixture_vds_env() -> (MutexGuard<'static, ()>, EnvVarGuard) {
    let guard = vds_env_lock().lock().unwrap();
    let env_guard = EnvVarGuard::remove("HDF5_VDS_PREFIX");
    (guard, env_guard)
}

struct EnvVarGuard {
    name: &'static str,
    original: Option<std::ffi::OsString>,
}

impl EnvVarGuard {
    fn remove(name: &'static str) -> Self {
        let original = std::env::var_os(name);
        std::env::remove_var(name);
        Self { name, original }
    }

    fn set(name: &'static str, value: impl AsRef<std::ffi::OsStr>) -> Self {
        let original = std::env::var_os(name);
        std::env::set_var(name, value);
        Self { name, original }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        if let Some(value) = &self.original {
            std::env::set_var(self.name, value);
        } else {
            std::env::remove_var(self.name);
        }
    }
}

fn shape_into(ds: &Dataset, dims: &mut Vec<u64>) {
    ds.shape_into(dims).unwrap();
}

fn shape_with_access_into(ds: &Dataset, access: &DatasetAccess, dims: &mut Vec<u64>) {
    ds.shape_with_access_into(access, dims).unwrap();
}

fn read_into<T>(ds: &Dataset, vals: &mut [T])
where
    T: H5Type,
{
    ds.read_into(vals).unwrap();
}

fn read_into_with_access<T>(ds: &Dataset, access: &DatasetAccess, vals: &mut [T])
where
    T: H5Type,
{
    ds.read_into_with_access(access, vals).unwrap();
}

#[test]
fn test_reference_virtual_dataset_regular_hyperslabs_read() {
    let (_guard, _env_guard) = fixture_vds_env();
    let f = File::open("hdf5/tools/test/testfiles/vds/1_vds.h5").unwrap();
    let ds = f.dataset("vds_dset").unwrap();
    assert!(ds.is_virtual().unwrap());
    let mut dims = Vec::new();
    shape_into(&ds, &mut dims);
    assert_eq!(dims, vec![5, 18, 8]);

    let mut vals = vec![0; ds.size().unwrap() as usize];
    read_into(&ds, &mut vals);
    assert_eq!(vals.len(), 5 * 18 * 8);

    let row = |plane: usize, y: usize| -> &[i32] {
        let start = (plane * 18 * 8) + (y * 8);
        &vals[start..start + 8]
    };
    assert_eq!(row(0, 0), &[10; 8]);
    assert_eq!(row(0, 2), &[20; 8]);
    assert_eq!(row(0, 14), &[60; 8]);
    assert_eq!(row(4, 0), &[14; 8]);
    assert_eq!(row(4, 14), &[64; 8]);
}

#[test]
fn test_reference_virtual_dataset_cross_axis_3d_mosaic_read() {
    let (_guard, _env_guard) = fixture_vds_env();
    let f = File::open("hdf5/tools/test/testfiles/vds/2_vds.h5").unwrap();
    let ds = f.dataset("vds_dset").unwrap();
    assert!(ds.is_virtual().unwrap());
    let mut dims = Vec::new();
    shape_into(&ds, &mut dims);
    assert_eq!(dims, vec![6, 8, 14]);
    assert_eq!(ds.create_plist().unwrap().virtual_count(), 5);

    let mut vals = vec![0; ds.size().unwrap() as usize];
    read_into(&ds, &mut vals);

    let row = |plane: usize, y: usize| -> &[i32] {
        let start = (plane * 8 * 14) + (y * 14);
        &vals[start..start + 14]
    };
    assert_eq!(
        row(0, 0),
        &[10, 10, 10, 10, 10, 10, 10, 40, 40, 40, 40, 40, 40, 40]
    );
    assert_eq!(
        row(0, 5),
        &[20, 20, 20, 20, 20, 20, 20, 50, 50, 50, 50, 50, 50, 50]
    );
    assert_eq!(
        row(5, 2),
        &[25, 25, 25, 25, 25, 25, 25, 45, 45, 45, 45, 45, 45, 45]
    );
    assert_eq!(
        row(5, 7),
        &[35, 35, 35, 35, 35, 35, 35, 55, 55, 55, 55, 55, 55, 55]
    );
}

#[test]
fn test_reference_virtual_dataset_fill_gap_3d_mosaic_read() {
    let (_guard, _env_guard) = fixture_vds_env();
    let f = File::open("hdf5/tools/test/testfiles/vds/3_1_vds.h5").unwrap();
    let ds = f.dataset("vds_dset").unwrap();
    assert!(ds.is_virtual().unwrap());
    let mut dims = Vec::new();
    shape_into(&ds, &mut dims);
    assert_eq!(dims, vec![5, 25, 8]);
    assert_eq!(ds.create_plist().unwrap().virtual_count(), 6);

    let mut vals = vec![0; ds.size().unwrap() as usize];
    read_into(&ds, &mut vals);

    let row = |plane: usize, y: usize| -> &[i32] {
        let start = (plane * 25 * 8) + (y * 8);
        &vals[start..start + 8]
    };
    assert_eq!(row(0, 0), &[-9; 8]);
    assert_eq!(row(0, 1), &[10; 8]);
    assert_eq!(row(0, 3), &[-9; 8]);
    assert_eq!(row(0, 4), &[20; 8]);
    assert_eq!(row(0, 8), &[-9; 8]);
    assert_eq!(row(4, 17), &[54; 8]);
    assert_eq!(row(4, 24), &[-9; 8]);
}

#[test]
fn test_reference_virtual_dataset_fill_gap_slice_crosses_mapped_rows() {
    let (_guard, _env_guard) = fixture_vds_env();
    let f = File::open("hdf5/tools/test/testfiles/vds/3_1_vds.h5").unwrap();
    let ds = f.dataset("vds_dset").unwrap();
    assert!(ds.is_virtual().unwrap());

    let mut vals = vec![0; 5 * 2];
    ds.read_slice_into::<i32, _>((0..1, 0..5, 0..2), &mut vals)
        .unwrap();

    assert_eq!(
        vals,
        vec![
            -9, -9, //
            10, 10, //
            10, 10, //
            -9, -9, //
            20, 20,
        ]
    );
}

#[test]
fn test_reference_virtual_dataset_printf_source_missing_pattern_fills() {
    let (_guard, _env_guard) = fixture_vds_env();
    let f = File::open("hdf5/tools/test/testfiles/vds/vds-eiger.h5").unwrap();
    let ds = f.dataset("VDS-Eiger").unwrap();
    assert!(ds.is_virtual().unwrap());
    let access = DatasetAccess::new()
        .with_virtual_missing_source_policy(VdsMissingSourcePolicy::Fill)
        .with_virtual_view(VdsView::LastAvailable);
    let mut dims = Vec::new();
    shape_with_access_into(&ds, &access, &mut dims);
    assert_eq!(dims, vec![20, 10, 10]);

    let plist = ds.create_plist().unwrap();
    assert_eq!(plist.virtual_count(), 1);
    assert_eq!(plist.virtual_filename(0), Some("f-%b.h5"));
    assert_eq!(plist.virtual_dsetname(0), Some("A"));
    assert!(matches!(
        plist.virtual_srcspace(0),
        Some(VirtualSelectionInfo::All)
    ));
    assert!(matches!(
        plist.virtual_vspace(0),
        Some(VirtualSelectionInfo::Regular { start, stride, count, block })
            if start.as_slice() == [0, 0, 0]
                && stride.as_slice() == [5, 1, 1]
                && count.as_slice() == [u64::MAX, 1, 1]
                && block.as_slice() == [5, 10, 10]
    ));

    let mut vals = vec![0; ds.size_with_access(&access).unwrap() as usize];
    read_into_with_access(&ds, &access, &mut vals);
    assert_eq!(vals.len(), 20 * 10 * 10);
    assert!(vals[..5 * 10 * 10].iter().all(|&value| value == 0));
    assert!(vals[5 * 10 * 10..15 * 10 * 10]
        .iter()
        .all(|&value| value == 0));
    assert!(vals[15 * 10 * 10..].iter().all(|&value| value == 3));

    let first_missing = DatasetAccess::new()
        .with_virtual_missing_source_policy(VdsMissingSourcePolicy::Fill)
        .with_virtual_view(VdsView::FirstMissing);
    shape_with_access_into(&ds, &first_missing, &mut dims);
    assert_eq!(dims, vec![5, 10, 10]);
    let first_vals = ds.read_with_access::<i32>(&first_missing).unwrap();
    assert_eq!(first_vals, vec![0; 5 * 10 * 10]);
}

#[test]
fn test_virtual_dataset_all_selection_read() {
    let (_guard, _env_guard) = fixture_vds_env();
    let f = File::open("tests/data/hdf5_ref/vds_all.h5").unwrap();
    let ds = f.dataset("vds_all").unwrap();
    assert!(ds.is_virtual().unwrap());
    let mut dims = Vec::new();
    shape_into(&ds, &mut dims);
    assert_eq!(dims, vec![4, 6]);
    let plist = ds.create_plist().unwrap();
    assert_eq!(plist.virtual_count(), 1);
    assert_eq!(plist.virtual_filename(0), Some("vds_all_source.h5"));
    assert_eq!(plist.virtual_dsetname(0), Some("/source"));
    assert!(matches!(
        plist.virtual_srcspace(0),
        Some(VirtualSelectionInfo::All)
    ));
    assert!(matches!(
        plist.virtual_vspace(0),
        Some(VirtualSelectionInfo::All)
    ));
    assert!(!plist.virtual_spatial_tree());

    let mut vals = vec![0; ds.size().unwrap() as usize];
    read_into(&ds, &mut vals);
    assert_eq!(vals, (0..24).collect::<Vec<_>>());
}

#[test]
fn test_virtual_dataset_all_selection_read_slice_subregion() {
    let (_guard, _env_guard) = fixture_vds_env();
    let f = File::open("tests/data/hdf5_ref/vds_all.h5").unwrap();
    let ds = f.dataset("vds_all").unwrap();
    assert!(ds.is_virtual().unwrap());

    let mut vals = vec![0; 2 * 3];
    ds.read_slice_into::<i32, _>((1..3, 2..5), &mut vals)
        .unwrap();

    assert_eq!(vals, vec![8, 9, 10, 14, 15, 16]);
}

#[test]
fn test_virtual_dataset_all_selection_strided_slice_read() {
    let (_guard, _env_guard) = fixture_vds_env();
    let f = File::open("tests/data/hdf5_ref/vds_all.h5").unwrap();
    let ds = f.dataset("vds_all").unwrap();
    assert!(ds.is_virtual().unwrap());

    let selection = Selection::Slice(vec![
        SliceInfo::with_step(0, 4, 2),
        SliceInfo::with_step(1, 6, 2),
    ]);
    let mut vals = vec![0; 2 * 3];
    ds.read_slice_into::<i32, _>(selection, &mut vals).unwrap();

    assert_eq!(vals, vec![1, 3, 5, 13, 15, 17]);
}

#[test]
fn test_virtual_dataset_all_selection_raw_read_with_view() {
    let (_guard, _env_guard) = fixture_vds_env();
    let f = File::open("tests/data/hdf5_ref/vds_all.h5").unwrap();
    let ds = f.dataset("vds_all").unwrap();
    assert!(ds.is_virtual().unwrap());

    let mut raw = vec![0; ds.size().unwrap() as usize * i32::type_size()];
    ds.read_raw_into_with_vds_view(VdsView::LastAvailable, &mut raw)
        .unwrap();
    let vals = raw
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();

    assert_eq!(vals, (0..24).collect::<Vec<_>>());
}

#[test]
fn test_virtual_dataset_same_file_source_read() {
    let (_guard, _env_guard) = fixture_vds_env();
    let f = File::open("tests/data/hdf5_ref/vds_same_file.h5").unwrap();
    let ds = f.dataset("vds_same_file").unwrap();
    assert!(ds.is_virtual().unwrap());
    let mut dims = Vec::new();
    shape_into(&ds, &mut dims);
    assert_eq!(dims, vec![3, 4]);

    let mut vals = vec![0; ds.size().unwrap() as usize];
    read_into(&ds, &mut vals);
    assert_eq!(vals, (0..12).collect::<Vec<_>>());
}

#[test]
fn test_virtual_dataset_mixed_all_and_regular_selection_read() {
    let (_guard, _env_guard) = fixture_vds_env();
    let f = File::open("tests/data/hdf5_ref/vds_mixed_all_regular.h5").unwrap();
    let ds = f.dataset("vds_mixed_all_regular").unwrap();
    assert!(ds.is_virtual().unwrap());
    let mut dims = Vec::new();
    shape_into(&ds, &mut dims);
    assert_eq!(dims, vec![4, 6]);

    let mut vals = vec![0; ds.size().unwrap() as usize];
    read_into(&ds, &mut vals);
    let mut expected = vec![0; 4 * 6];
    for row in 0..2 {
        for col in 0..3 {
            expected[(row + 1) * 6 + col + 2] = (row * 3 + col) as i32;
        }
    }
    assert_eq!(vals, expected);
}

#[test]
fn test_virtual_dataset_fill_value_for_unmapped_regions() {
    let (_guard, _env_guard) = fixture_vds_env();
    let f = File::open("tests/data/hdf5_ref/vds_fill_value.h5").unwrap();
    let ds = f.dataset("vds_fill_value").unwrap();
    assert!(ds.is_virtual().unwrap());
    let mut dims = Vec::new();
    shape_into(&ds, &mut dims);
    assert_eq!(dims, vec![4, 6]);

    let mut vals = vec![0; ds.size().unwrap() as usize];
    read_into(&ds, &mut vals);
    let mut expected = vec![-7; 4 * 6];
    for row in 0..2 {
        for col in 0..3 {
            expected[(row + 1) * 6 + col + 2] = (row * 3 + col) as i32;
        }
    }
    assert_eq!(vals, expected);
}

#[test]
fn test_virtual_dataset_fill_value_create_plist_metadata() {
    let (_guard, _env_guard) = fixture_vds_env();
    let f = File::open("tests/data/hdf5_ref/vds_fill_value.h5").unwrap();
    let ds = f.dataset("vds_fill_value").unwrap();
    assert!(ds.is_virtual().unwrap());

    let plist = ds.create_plist().unwrap();
    assert_eq!(plist.virtual_count(), 1);
    assert!(plist.fill_value_defined);
    let fill = (-7i32).to_le_bytes();
    assert_eq!(plist.fill_value(), Some(fill.as_slice()));
}

#[test]
fn test_virtual_dataset_fill_value_slice_crosses_mapped_and_unmapped_regions() {
    let (_guard, _env_guard) = fixture_vds_env();
    let f = File::open("tests/data/hdf5_ref/vds_fill_value.h5").unwrap();
    let ds = f.dataset("vds_fill_value").unwrap();
    assert!(ds.is_virtual().unwrap());

    let mut vals = vec![0; 4 * 4];
    ds.read_slice_into::<i32, _>((0..4, 1..5), &mut vals)
        .unwrap();

    assert_eq!(
        vals,
        vec![
            -7, -7, -7, -7, //
            -7, 0, 1, 2, //
            -7, 3, 4, 5, //
            -7, -7, -7, -7,
        ]
    );
}

#[test]
fn test_virtual_dataset_fill_value_strided_slice_keeps_unmapped_fill() {
    let (_guard, _env_guard) = fixture_vds_env();
    let f = File::open("tests/data/hdf5_ref/vds_fill_value.h5").unwrap();
    let ds = f.dataset("vds_fill_value").unwrap();
    assert!(ds.is_virtual().unwrap());

    let selection = Selection::Slice(vec![
        SliceInfo::with_step(0, 4, 2),
        SliceInfo::with_step(0, 6, 2),
    ]);
    let mut vals = vec![0; 2 * 3];
    ds.read_slice_into::<i32, _>(selection, &mut vals).unwrap();

    assert_eq!(vals, vec![-7, -7, -7, -7, 3, 5]);
}

#[test]
fn test_virtual_dataset_fill_value_point_and_hyperslab_selections() {
    let (_guard, _env_guard) = fixture_vds_env();
    let f = File::open("tests/data/hdf5_ref/vds_fill_value.h5").unwrap();
    let ds = f.dataset("vds_fill_value").unwrap();
    assert!(ds.is_virtual().unwrap());

    let points = Selection::Points(vec![vec![0, 0], vec![1, 2], vec![2, 4], vec![3, 5]]);
    let mut point_vals = vec![0; 4];
    ds.read_selection_into::<i32>(&[4, 6], &points, &mut point_vals)
        .unwrap();
    assert_eq!(point_vals, vec![-7, 0, 5, -7]);

    let hyperslab = Selection::Hyperslab(vec![
        HyperslabDim::new(0, 2, 2, 1),
        HyperslabDim::new(1, 2, 3, 1),
    ]);
    let mut hyperslab_vals = vec![0; 2 * 3];
    ds.read_selection_into::<i32>(&[4, 6], &hyperslab, &mut hyperslab_vals)
        .unwrap();
    assert_eq!(hyperslab_vals, vec![-7, -7, -7, -7, 4, -7]);
}

#[test]
fn test_virtual_dataset_fill_value_point_selection_preserves_order_and_duplicates() {
    let (_guard, _env_guard) = fixture_vds_env();
    let f = File::open("tests/data/hdf5_ref/vds_fill_value.h5").unwrap();
    let ds = f.dataset("vds_fill_value").unwrap();
    assert!(ds.is_virtual().unwrap());

    let points = Selection::Points(vec![
        vec![2, 4],
        vec![0, 0],
        vec![2, 4],
        vec![1, 3],
        vec![3, 5],
    ]);
    let mut vals = vec![0; 5];
    ds.read_selection_into::<i32>(&[4, 6], &points, &mut vals)
        .unwrap();

    assert_eq!(vals, vec![5, -7, 5, 1, -7]);
}

#[test]
fn test_virtual_dataset_read_cell_covers_mapped_and_fill_regions() {
    let (_guard, _env_guard) = fixture_vds_env();
    let f = File::open("tests/data/hdf5_ref/vds_fill_value.h5").unwrap();
    let ds = f.dataset("vds_fill_value").unwrap();
    assert!(ds.is_virtual().unwrap());

    assert_eq!(ds.read_cell::<i32>(&[1, 2]).unwrap(), 0);
    assert_eq!(ds.read_cell::<i32>(&[2, 4]).unwrap(), 5);
    assert_eq!(ds.read_cell::<i32>(&[0, 0]).unwrap(), -7);
    assert_eq!(ds.read_cell::<i32>(&[3, 5]).unwrap(), -7);

    let mut value = 1234;
    ds.read_cell_into::<i32>(&[1, 3], &mut value).unwrap();
    assert_eq!(value, 1);
    ds.read_cell_into::<i32>(&[3, 0], &mut value).unwrap();
    assert_eq!(value, -7);

    let err = ds
        .read_cell_into::<i32>(&[4, 0], &mut value)
        .expect_err("out-of-bounds VDS cell read should fail");
    assert!(
        err.to_string().contains("exceeds extent"),
        "unexpected error: {err}"
    );
    assert_eq!(value, -7);
}

#[test]
fn test_virtual_dataset_f64_read() {
    let (_guard, _env_guard) = fixture_vds_env();
    let f = File::open("tests/data/hdf5_ref/vds_f64.h5").unwrap();
    let ds = f.dataset("vds_f64").unwrap();
    assert!(ds.is_virtual().unwrap());
    let mut dims = Vec::new();
    shape_into(&ds, &mut dims);
    assert_eq!(dims, vec![3, 4]);

    let mut vals = vec![0.0; ds.size().unwrap() as usize];
    read_into(&ds, &mut vals);
    let expected = (0..12)
        .map(|value| (value as f64 / 2.0) + 0.25)
        .collect::<Vec<_>>();
    assert_eq!(vals, expected);
}

#[test]
fn test_virtual_dataset_converts_source_datatype_to_destination() {
    let (_guard, _env_guard) = fixture_vds_env();
    let f = File::open("tests/data/hdf5_ref/vds_i32_to_f64.h5").unwrap();
    let ds = f.dataset("vds_f64_from_i32").unwrap();
    assert!(ds.is_virtual().unwrap());

    let mut vals = vec![0.0; ds.size().unwrap() as usize];
    read_into(&ds, &mut vals);
    assert_eq!(vals, vec![1.0, -2.0, 300.0, 4000.0]);
}

#[test]
fn test_virtual_dataset_converts_source_datatype_to_destination_slice() {
    let (_guard, _env_guard) = fixture_vds_env();
    let f = File::open("tests/data/hdf5_ref/vds_i32_to_f64.h5").unwrap();
    let ds = f.dataset("vds_f64_from_i32").unwrap();
    assert!(ds.is_virtual().unwrap());

    let mut vals = [0.0f64; 3];
    ds.read_slice_into::<f64, _>(1..4, &mut vals).unwrap();
    assert_eq!(vals, [-2.0, 300.0, 4000.0]);

    let sliced = ds.read_slice::<f64, _>(2..4).unwrap();
    assert_eq!(sliced, [300.0, 4000.0]);

    let mut narrowed = [0i16; 3];
    ds.read_slice_into::<i16, _>(0..3, &mut narrowed).unwrap();
    assert_eq!(narrowed, [1, -2, 300]);
}

#[test]
fn test_virtual_dataset_converted_read_cell_preserves_output_on_bad_coord() {
    let (_guard, _env_guard) = fixture_vds_env();
    let f = File::open("tests/data/hdf5_ref/vds_i32_to_f64.h5").unwrap();
    let ds = f.dataset("vds_f64_from_i32").unwrap();
    assert!(ds.is_virtual().unwrap());

    assert_eq!(ds.read_cell::<i16>(&[2]).unwrap(), 300);

    let mut value = -123i16;
    ds.read_cell_into::<i16>(&[3], &mut value).unwrap();
    assert_eq!(value, 4000);

    let err = ds
        .read_cell_into::<i16>(&[4], &mut value)
        .expect_err("out-of-bounds converted VDS cell read should fail");
    assert!(
        err.to_string().contains("exceeds extent"),
        "unexpected error: {err}"
    );
    assert_eq!(value, 4000);
}

#[test]
fn test_virtual_dataset_converted_point_selection_preserves_output_on_wrong_length() {
    let (_guard, _env_guard) = fixture_vds_env();
    let f = File::open("tests/data/hdf5_ref/vds_i32_to_f64.h5").unwrap();
    let ds = f.dataset("vds_f64_from_i32").unwrap();
    assert!(ds.is_virtual().unwrap());

    let selection = Selection::Points(vec![vec![0], vec![2], vec![3]]);
    let mut narrowed = [-1i16; 3];
    ds.read_selection_into::<i16>(&[4], &selection, &mut narrowed)
        .unwrap();
    assert_eq!(narrowed, [1, 300, 4000]);

    let mut stale = [-5i16, -6];
    let err = ds
        .read_selection_into::<i16>(&[4], &selection, &mut stale)
        .expect_err("converted VDS point selection should reject wrong output length");
    assert!(
        err.to_string().contains("slice output buffer"),
        "unexpected error: {err}"
    );
    assert_eq!(stale, [-5, -6]);
}

#[test]
fn test_virtual_dataset_converted_hyperslab_selection_preserves_output_on_wrong_length() {
    let (_guard, _env_guard) = fixture_vds_env();
    let f = File::open("tests/data/hdf5_ref/vds_i32_to_f64.h5").unwrap();
    let ds = f.dataset("vds_f64_from_i32").unwrap();
    assert!(ds.is_virtual().unwrap());

    let selection = Selection::Hyperslab(vec![HyperslabDim::new(1, 1, 3, 1)]);
    let mut narrowed = [-1i16; 3];
    ds.read_selection_into::<i16>(&[4], &selection, &mut narrowed)
        .unwrap();
    assert_eq!(narrowed, [-2, 300, 4000]);

    let mut stale = [-5i16, -6];
    let err = ds
        .read_selection_into::<i16>(&[4], &selection, &mut stale)
        .expect_err("converted VDS hyperslab selection should reject wrong output length");
    assert!(
        err.to_string().contains("slice output buffer"),
        "unexpected error: {err}"
    );
    assert_eq!(stale, [-5, -6]);
}

#[test]
fn test_virtual_dataset_rejects_mismatched_read_element_size() {
    let (_guard, _env_guard) = fixture_vds_env();
    let f = File::open("tests/data/hdf5_ref/vds_f64.h5").unwrap();
    let ds = f.dataset("vds_f64").unwrap();
    let mut vals = vec![ThreeBytes::default(); ds.size().unwrap() as usize];
    vals.fill(ThreeBytes([0xaa, 0xbb, 0xcc]));
    let err = ds
        .read_into(&mut vals)
        .expect_err("VDS read should reject mismatched destination element sizes");

    assert!(matches!(err, Error::InvalidFormat(_)));
    assert!(vals.iter().all(|value| value.0 == [0xaa, 0xbb, 0xcc]));
}

#[test]
fn test_virtual_dataset_scalar_mapping_read() {
    let (_guard, _env_guard) = fixture_vds_env();
    let f = File::open("tests/data/hdf5_ref/vds_scalar.h5").unwrap();
    let ds = f.dataset("vds_scalar").unwrap();
    assert!(ds.is_virtual().unwrap());
    let mut dims = Vec::new();
    shape_into(&ds, &mut dims);
    assert_eq!(dims, Vec::<u64>::new());
    assert_eq!(ds.size().unwrap(), 1);

    let val = ds.read_scalar::<i32>().unwrap();
    assert_eq!(val, 42);

    let mut out = 0;
    ds.read_scalar_into_with_vds_view(VdsView::LastAvailable, &mut out)
        .unwrap();
    assert_eq!(out, 42);
}

#[test]
fn test_virtual_dataset_zero_sized_mapping_read() {
    let (_guard, _env_guard) = fixture_vds_env();
    let f = File::open("tests/data/hdf5_ref/vds_zero_sized.h5").unwrap();
    let ds = f.dataset("vds_zero_sized").unwrap();
    assert!(ds.is_virtual().unwrap());
    let mut dims = Vec::new();
    shape_into(&ds, &mut dims);
    assert_eq!(dims, vec![0, 4]);
    assert_eq!(ds.size().unwrap(), 0);

    let mut vals = vec![0; ds.size().unwrap() as usize];
    read_into(&ds, &mut vals);
    assert!(vals.is_empty());
}

#[test]
fn test_virtual_dataset_null_mapping_read() {
    let (_guard, _env_guard) = fixture_vds_env();
    let f = File::open("tests/data/hdf5_ref/vds_null.h5").unwrap();
    let ds = f.dataset("vds_null").unwrap();
    assert!(ds.is_virtual().unwrap());
    assert!(ds.space().unwrap().is_null());
    let mut dims = Vec::new();
    shape_into(&ds, &mut dims);
    assert_eq!(dims, Vec::<u64>::new());
    assert_eq!(ds.size().unwrap(), 0);

    let mut vals = vec![0; ds.size().unwrap() as usize];
    read_into(&ds, &mut vals);
    assert!(vals.is_empty());
}

#[test]
fn test_virtual_dataset_rank_mismatch_mapping_read() {
    let (_guard, _env_guard) = fixture_vds_env();
    let f = File::open("tests/data/hdf5_ref/vds_rank_mismatch.h5").unwrap();
    let ds = f.dataset("vds_rank_mismatch").unwrap();
    assert!(ds.is_virtual().unwrap());
    let mut dims = Vec::new();
    shape_into(&ds, &mut dims);
    assert_eq!(dims, vec![2, 3]);

    let mut vals = vec![0; ds.size().unwrap() as usize];
    read_into(&ds, &mut vals);
    assert_eq!(vals, (0..6).collect::<Vec<_>>());
}

#[test]
fn test_virtual_dataset_overlapping_mappings_later_mapping_wins() {
    let (_guard, _env_guard) = fixture_vds_env();
    let f = File::open("tests/data/hdf5_ref/vds_overlap.h5").unwrap();
    let ds = f.dataset("vds_overlap").unwrap();
    assert!(ds.is_virtual().unwrap());
    let mut dims = Vec::new();
    shape_into(&ds, &mut dims);
    assert_eq!(dims, vec![4]);

    let mut vals = vec![0; ds.size().unwrap() as usize];
    read_into(&ds, &mut vals);
    assert_eq!(vals, vec![1, 9, 8, 4]);
}

#[test]
fn test_virtual_dataset_overlap_slice_uses_later_mapping() {
    let (_guard, _env_guard) = fixture_vds_env();
    let f = File::open("tests/data/hdf5_ref/vds_overlap.h5").unwrap();
    let ds = f.dataset("vds_overlap").unwrap();
    assert!(ds.is_virtual().unwrap());

    let mut vals = vec![0; 3];
    ds.read_slice_into::<i32, _>(1..4, &mut vals).unwrap();

    assert_eq!(vals, vec![9, 8, 4]);
}

#[test]
fn test_virtual_dataset_irregular_hyperslab_read() {
    let (_guard, _env_guard) = fixture_vds_env();
    let f = File::open("tests/data/hdf5_ref/vds_irregular_hyperslab.h5").unwrap();
    let ds = f.dataset("vds_irregular_hyperslab").unwrap();
    assert!(ds.is_virtual().unwrap());
    let mut dims = Vec::new();
    shape_into(&ds, &mut dims);
    assert_eq!(dims, vec![4, 4]);
    let plist = ds.create_plist().unwrap();
    assert_eq!(plist.virtual_count(), 1);
    assert!(matches!(
        plist.virtual_srcspace(0),
        Some(VirtualSelectionInfo::Irregular(blocks)) if blocks.len() == 2
    ));
    assert!(matches!(
        plist.virtual_vspace(0),
        Some(VirtualSelectionInfo::Irregular(blocks)) if blocks.len() == 2
    ));

    let mut vals = vec![0; ds.size().unwrap() as usize];
    read_into(&ds, &mut vals);
    let mut expected = vec![-2; 4 * 4];
    expected[1] = 1;
    expected[2] = 2;
    expected[2 * 4] = 8;
    expected[2 * 4 + 1] = 9;
    assert_eq!(vals, expected);
}

#[test]
fn test_virtual_dataset_point_selection_read() {
    let (_guard, _env_guard) = fixture_vds_env();
    let f = File::open("tests/data/hdf5_ref/vds_point_selection.h5").unwrap();
    let ds = f.dataset("vds_point_selection").unwrap();
    assert!(ds.is_virtual().unwrap());
    let mut dims = Vec::new();
    shape_into(&ds, &mut dims);
    assert_eq!(dims, vec![4, 4]);
    let plist = ds.create_plist().unwrap();
    assert_eq!(plist.virtual_count(), 1);
    assert!(matches!(
        plist.virtual_srcspace(0),
        Some(VirtualSelectionInfo::Points(points)) if points == &vec![vec![2, 1]]
    ));
    assert!(matches!(
        plist.virtual_vspace(0),
        Some(VirtualSelectionInfo::Points(points)) if points == &vec![vec![0, 3]]
    ));

    let mut vals = vec![0; ds.size().unwrap() as usize];
    read_into(&ds, &mut vals);
    let mut expected = vec![-2; 4 * 4];
    expected[3] = 9;
    assert_eq!(vals, expected);
}

#[test]
fn test_virtual_dataset_missing_source_file_fails_without_access_property_policy() {
    let _guard = vds_env_lock().lock().unwrap();
    let _env_guard = EnvVarGuard::remove("HDF5_VDS_PREFIX");
    let dir = tempfile::tempdir().unwrap();
    let vds_path = dir.path().join("vds_all.h5");
    std::fs::copy("tests/data/hdf5_ref/vds_all.h5", &vds_path).unwrap();

    let f = File::open(&vds_path).unwrap();
    let ds = f.dataset("vds_all").unwrap();
    let mut vals = vec![0; ds.size().unwrap() as usize];
    let err = ds
        .read_into(&mut vals)
        .expect_err("missing VDS source should fail without a VDS access policy");

    assert!(
        matches!(err, Error::Io(_)),
        "missing source should surface as file I/O without VDS access-property behavior: {err}"
    );
}

#[test]
fn test_virtual_dataset_missing_source_file_can_read_fill_values() {
    let _guard = vds_env_lock().lock().unwrap();
    let _env_guard = EnvVarGuard::remove("HDF5_VDS_PREFIX");
    let dir = tempfile::tempdir().unwrap();
    let vds_path = dir.path().join("vds_all.h5");
    std::fs::copy("tests/data/hdf5_ref/vds_all.h5", &vds_path).unwrap();

    let access =
        DatasetAccess::new().with_virtual_missing_source_policy(VdsMissingSourcePolicy::Fill);
    let f = File::open(&vds_path).unwrap();
    let ds = f.dataset("vds_all").unwrap();

    let mut dims = Vec::new();
    shape_with_access_into(&ds, &access, &mut dims);
    assert_eq!(dims, vec![4, 6]);
    let mut vals = vec![0; ds.size().unwrap() as usize];
    read_into_with_access(&ds, &access, &mut vals);
    assert_eq!(vals, vec![0; 24]);
}

#[test]
fn test_virtual_dataset_missing_source_file_read_with_access_converts_fill_values() {
    let _guard = vds_env_lock().lock().unwrap();
    let _env_guard = EnvVarGuard::remove("HDF5_VDS_PREFIX");
    let dir = tempfile::tempdir().unwrap();
    let vds_path = dir.path().join("vds_all.h5");
    std::fs::copy("tests/data/hdf5_ref/vds_all.h5", &vds_path).unwrap();

    let access =
        DatasetAccess::new().with_virtual_missing_source_policy(VdsMissingSourcePolicy::Fill);
    let f = File::open(&vds_path).unwrap();
    let ds = f.dataset("vds_all").unwrap();

    let vals = ds.read_with_access::<f64>(&access).unwrap();
    assert_eq!(vals, vec![0.0; 24]);
}

#[test]
fn test_virtual_dataset_missing_source_file_read_into_converts_fill_values() {
    let _guard = vds_env_lock().lock().unwrap();
    let _env_guard = EnvVarGuard::remove("HDF5_VDS_PREFIX");
    let dir = tempfile::tempdir().unwrap();
    let vds_path = dir.path().join("vds_all.h5");
    std::fs::copy("tests/data/hdf5_ref/vds_all.h5", &vds_path).unwrap();

    let access =
        DatasetAccess::new().with_virtual_missing_source_policy(VdsMissingSourcePolicy::Fill);
    let f = File::open(&vds_path).unwrap();
    let ds = f.dataset("vds_all").unwrap();

    let mut vals = vec![99.0f64; ds.size_with_access(&access).unwrap() as usize];
    ds.read_into_with_access(&access, &mut vals).unwrap();
    assert_eq!(vals, vec![0.0; 24]);
}

#[test]
fn test_virtual_dataset_missing_source_file_read_into_preserves_output_on_wrong_length() {
    let _guard = vds_env_lock().lock().unwrap();
    let _env_guard = EnvVarGuard::remove("HDF5_VDS_PREFIX");
    let dir = tempfile::tempdir().unwrap();
    let vds_path = dir.path().join("vds_all.h5");
    std::fs::copy("tests/data/hdf5_ref/vds_all.h5", &vds_path).unwrap();

    let access =
        DatasetAccess::new().with_virtual_missing_source_policy(VdsMissingSourcePolicy::Fill);
    let f = File::open(&vds_path).unwrap();
    let ds = f.dataset("vds_all").unwrap();

    let mut stale = vec![99.0f64; 23];
    let err = ds
        .read_into_with_access(&access, &mut stale)
        .expect_err("wrong-length fill-policy VDS read should fail before replacing output");
    assert!(
        err.to_string().contains("typed output buffer"),
        "unexpected error: {err}"
    );
    assert_eq!(stale, vec![99.0; 23]);
}

#[test]
fn test_virtual_dataset_missing_source_file_raw_read_uses_fill_policy() {
    let _guard = vds_env_lock().lock().unwrap();
    let _env_guard = EnvVarGuard::remove("HDF5_VDS_PREFIX");
    let dir = tempfile::tempdir().unwrap();
    let vds_path = dir.path().join("vds_all.h5");
    std::fs::copy("tests/data/hdf5_ref/vds_all.h5", &vds_path).unwrap();

    let access =
        DatasetAccess::new().with_virtual_missing_source_policy(VdsMissingSourcePolicy::Fill);
    let f = File::open(&vds_path).unwrap();
    let ds = f.dataset("vds_all").unwrap();

    assert_eq!(ds.size_with_access(&access).unwrap(), 24);
    let raw = ds.read_raw_with_access(&access).unwrap();
    assert_eq!(raw, vec![0; 24 * i32::type_size()]);
}

#[test]
fn test_virtual_dataset_missing_source_file_raw_read_into_uses_fill_policy() {
    let _guard = vds_env_lock().lock().unwrap();
    let _env_guard = EnvVarGuard::remove("HDF5_VDS_PREFIX");
    let dir = tempfile::tempdir().unwrap();
    let vds_path = dir.path().join("vds_all.h5");
    std::fs::copy("tests/data/hdf5_ref/vds_all.h5", &vds_path).unwrap();

    let access =
        DatasetAccess::new().with_virtual_missing_source_policy(VdsMissingSourcePolicy::Fill);
    let f = File::open(&vds_path).unwrap();
    let ds = f.dataset("vds_all").unwrap();

    let mut raw = vec![0xaa; 24 * i32::type_size()];
    ds.read_raw_into_with_access(&access, &mut raw).unwrap();
    assert_eq!(raw, vec![0; 24 * i32::type_size()]);
}

#[test]
fn test_virtual_dataset_missing_source_file_raw_read_into_preserves_output_on_wrong_length() {
    let _guard = vds_env_lock().lock().unwrap();
    let _env_guard = EnvVarGuard::remove("HDF5_VDS_PREFIX");
    let dir = tempfile::tempdir().unwrap();
    let vds_path = dir.path().join("vds_all.h5");
    std::fs::copy("tests/data/hdf5_ref/vds_all.h5", &vds_path).unwrap();

    let access =
        DatasetAccess::new().with_virtual_missing_source_policy(VdsMissingSourcePolicy::Fill);
    let f = File::open(&vds_path).unwrap();
    let ds = f.dataset("vds_all").unwrap();

    let mut stale = vec![0xaa; 24 * i32::type_size() - 1];
    let err = ds
        .read_raw_into_with_access(&access, &mut stale)
        .expect_err("wrong-length fill-policy raw VDS read should fail before replacing output");
    assert!(
        err.to_string().contains("raw output buffer"),
        "unexpected error: {err}"
    );
    assert_eq!(stale, vec![0xaa; 24 * i32::type_size() - 1]);
}

#[test]
fn test_virtual_dataset_missing_source_dataset_can_read_fill_values() {
    let _guard = vds_env_lock().lock().unwrap();
    let _env_guard = EnvVarGuard::remove("HDF5_VDS_PREFIX");
    let dir = tempfile::tempdir().unwrap();
    let vds_path = dir.path().join("vds_all.h5");
    std::fs::copy("tests/data/hdf5_ref/vds_all.h5", &vds_path).unwrap();
    std::fs::copy(
        "tests/data/hdf5_ref/vds_all.h5",
        dir.path().join("vds_all_source.h5"),
    )
    .unwrap();

    let access =
        DatasetAccess::new().with_virtual_missing_source_policy(VdsMissingSourcePolicy::Fill);
    let f = File::open(&vds_path).unwrap();
    let ds = f.dataset("vds_all").unwrap();
    let mut vals = vec![99; ds.size().unwrap() as usize];
    ds.read_into_with_access(&access, &mut vals).unwrap();
    assert_eq!(vals, vec![0; ds.size().unwrap() as usize]);

    let converted = ds.read_with_access::<f64>(&access).unwrap();
    assert_eq!(converted, vec![0.0; ds.size().unwrap() as usize]);
}

#[test]
fn test_virtual_dataset_missing_source_dataset_raw_reads_write_fill_values() {
    let _guard = vds_env_lock().lock().unwrap();
    let _env_guard = EnvVarGuard::remove("HDF5_VDS_PREFIX");
    let dir = tempfile::tempdir().unwrap();
    let vds_path = dir.path().join("vds_all.h5");
    std::fs::copy("tests/data/hdf5_ref/vds_all.h5", &vds_path).unwrap();
    std::fs::copy(
        "tests/data/hdf5_ref/vds_all.h5",
        dir.path().join("vds_all_source.h5"),
    )
    .unwrap();

    let access =
        DatasetAccess::new().with_virtual_missing_source_policy(VdsMissingSourcePolicy::Fill);
    let f = File::open(&vds_path).unwrap();
    let ds = f.dataset("vds_all").unwrap();

    let mut raw = vec![0xaa; ds.size().unwrap() as usize * i32::type_size()];
    ds.read_raw_into_with_access(&access, &mut raw).unwrap();
    assert_eq!(raw, vec![0; ds.size().unwrap() as usize * i32::type_size()]);

    let raw_vec = ds.read_raw_with_access(&access).unwrap();
    assert_eq!(
        raw_vec,
        vec![0; ds.size().unwrap() as usize * i32::type_size()]
    );
}

#[test]
fn test_virtual_dataset_uses_hdf5_vds_prefix_directory() {
    let _guard = vds_env_lock().lock().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let vds_path = dir.path().join("vds_all.h5");
    let prefixed_dir = dir.path().join("prefixed");
    std::fs::create_dir(&prefixed_dir).unwrap();

    std::fs::copy("tests/data/hdf5_ref/vds_all.h5", &vds_path).unwrap();
    std::fs::copy(
        "tests/data/hdf5_ref/vds_all_source.h5",
        prefixed_dir.join("vds_all_source.h5"),
    )
    .unwrap();

    let _env_guard = EnvVarGuard::set("HDF5_VDS_PREFIX", &prefixed_dir);
    let f = File::open(&vds_path).unwrap();
    let ds = f.dataset("vds_all").unwrap();
    let mut vals = vec![0; ds.size().unwrap() as usize];
    read_into(&ds, &mut vals);

    assert_eq!(vals, (0..24).collect::<Vec<_>>());
}

#[test]
fn test_virtual_dataset_hdf5_vds_prefix_searches_multiple_directories() {
    let _guard = vds_env_lock().lock().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let vds_path = dir.path().join("vds_all.h5");
    let missing_dir = dir.path().join("missing");
    let prefixed_dir = dir.path().join("prefixed");
    std::fs::create_dir(&missing_dir).unwrap();
    std::fs::create_dir(&prefixed_dir).unwrap();

    std::fs::copy("tests/data/hdf5_ref/vds_all.h5", &vds_path).unwrap();
    std::fs::copy(
        "tests/data/hdf5_ref/vds_all_source.h5",
        prefixed_dir.join("vds_all_source.h5"),
    )
    .unwrap();

    let prefixes = format!("{}:{}", missing_dir.display(), prefixed_dir.display());
    let _env_guard = EnvVarGuard::set("HDF5_VDS_PREFIX", prefixes);
    let f = File::open(&vds_path).unwrap();
    let ds = f.dataset("vds_all").unwrap();
    let mut vals = vec![0; ds.size().unwrap() as usize];
    read_into(&ds, &mut vals);

    assert_eq!(vals, (0..24).collect::<Vec<_>>());
}

#[test]
fn test_virtual_dataset_uses_explicit_access_prefix_directory() {
    let _guard = vds_env_lock().lock().unwrap();
    let _env_guard = EnvVarGuard::remove("HDF5_VDS_PREFIX");
    let dir = tempfile::tempdir().unwrap();
    let vds_path = dir.path().join("vds_all.h5");
    let prefixed_dir = dir.path().join("prefixed");
    std::fs::create_dir(&prefixed_dir).unwrap();

    std::fs::copy("tests/data/hdf5_ref/vds_all.h5", &vds_path).unwrap();
    std::fs::copy(
        "tests/data/hdf5_ref/vds_all_source.h5",
        prefixed_dir.join("vds_all_source.h5"),
    )
    .unwrap();

    let access = DatasetAccess::new().with_virtual_prefix(&prefixed_dir);
    let f = File::open(&vds_path).unwrap();
    let ds = f.dataset("vds_all").unwrap();
    let mut dims = Vec::new();
    shape_with_access_into(&ds, &access, &mut dims);
    assert_eq!(dims, vec![4, 6]);
    let mut vals = vec![0; ds.size().unwrap() as usize];
    read_into_with_access(&ds, &access, &mut vals);

    assert_eq!(vals, (0..24).collect::<Vec<_>>());
}

#[test]
fn test_virtual_dataset_uses_hdf5_vds_prefix_origin_expansion() {
    let _guard = vds_env_lock().lock().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let vds_path = dir.path().join("vds_all.h5");
    let prefixed_dir = dir.path().join("prefixed");
    std::fs::create_dir(&prefixed_dir).unwrap();

    std::fs::copy("tests/data/hdf5_ref/vds_all.h5", &vds_path).unwrap();
    std::fs::copy(
        "tests/data/hdf5_ref/vds_all_source.h5",
        prefixed_dir.join("vds_all_source.h5"),
    )
    .unwrap();

    let _env_guard = EnvVarGuard::set("HDF5_VDS_PREFIX", "${ORIGIN}/prefixed");
    let f = File::open(&vds_path).unwrap();
    let ds = f.dataset("vds_all").unwrap();
    let mut vals = vec![0; ds.size().unwrap() as usize];
    read_into(&ds, &mut vals);

    assert_eq!(vals, (0..24).collect::<Vec<_>>());
}

#[test]
fn test_virtual_dataset_access_prefix_expands_origin() {
    let _guard = vds_env_lock().lock().unwrap();
    let _env_guard = EnvVarGuard::remove("HDF5_VDS_PREFIX");
    let dir = tempfile::tempdir().unwrap();
    let vds_path = dir.path().join("vds_all.h5");
    let prefixed_dir = dir.path().join("prefixed");
    std::fs::create_dir(&prefixed_dir).unwrap();

    std::fs::copy("tests/data/hdf5_ref/vds_all.h5", &vds_path).unwrap();
    std::fs::copy(
        "tests/data/hdf5_ref/vds_all_source.h5",
        prefixed_dir.join("vds_all_source.h5"),
    )
    .unwrap();

    let access = DatasetAccess::new()
        .with_virtual_prefix("${ORIGIN}/prefixed")
        .with_virtual_view(VdsView::LastAvailable);
    let f = File::open(&vds_path).unwrap();
    let ds = f.dataset("vds_all").unwrap();
    let mut vals = vec![0; ds.size().unwrap() as usize];
    read_into_with_access(&ds, &access, &mut vals);

    assert_eq!(vals, (0..24).collect::<Vec<_>>());
}
