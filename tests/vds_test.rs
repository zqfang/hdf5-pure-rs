use hdf5_pure_rust::hl::plist::dataset_create::VirtualSelectionInfo;
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
fn test_virtual_dataset_rejects_mismatched_read_element_size() {
    let (_guard, _env_guard) = fixture_vds_env();
    let f = File::open("tests/data/hdf5_ref/vds_f64.h5").unwrap();
    let ds = f.dataset("vds_f64").unwrap();
    let mut vals = vec![ThreeBytes::default(); ds.size().unwrap() as usize];
    let err = ds
        .read_into(&mut vals)
        .expect_err("VDS read should reject mismatched destination element sizes");

    assert!(matches!(err, Error::InvalidFormat(_)));
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
fn test_virtual_dataset_missing_source_dataset_still_fails_with_fill_policy() {
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
    let mut vals = vec![0; ds.size().unwrap() as usize];
    let err = ds
        .read_into_with_access(&access, &mut vals)
        .expect_err("fill policy should not mask a missing source dataset path");

    assert!(
        !matches!(err, Error::Io(ref io_err) if io_err.kind() == std::io::ErrorKind::NotFound),
        "missing source dataset should not be handled as a missing source file: {err}"
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
