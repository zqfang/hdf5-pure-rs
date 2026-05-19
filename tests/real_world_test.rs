use hdf5_pure_rust::hl::types::H5Type;
use hdf5_pure_rust::{Dataset, File};

fn read_vec<T: H5Type + Default>(dataset: &Dataset) -> hdf5_pure_rust::Result<Vec<T>> {
    let mut values = vec![T::default(); dataset.size()? as usize];
    dataset.read_into(&mut values)?;
    Ok(values)
}

fn read_strings_vec(dataset: &Dataset) -> hdf5_pure_rust::Result<Vec<String>> {
    let mut values = Vec::new();
    dataset.read_strings_into(&mut values)?;
    Ok(values)
}

fn read_field_vec<T: H5Type + Default>(
    dataset: &Dataset,
    field_name: &str,
) -> hdf5_pure_rust::Result<Vec<T>> {
    let mut values = vec![T::default(); dataset.size()? as usize];
    dataset.read_field_into(field_name, &mut values)?;
    Ok(values)
}

fn shape_vec(dataset: &Dataset) -> hdf5_pure_rust::Result<Vec<u64>> {
    let mut shape = Vec::new();
    dataset.shape_into(&mut shape)?;
    Ok(shape)
}

fn compound_field_names(dataset: &Dataset) -> hdf5_pure_rust::Result<Vec<String>> {
    dataset
        .dtype()?
        .compound_fields_iter()?
        .map(|field| field.map(|field| field.name.into_owned()))
        .collect()
}

fn file_member_names(file: &File) -> hdf5_pure_rust::Result<Vec<String>> {
    let mut names = Vec::new();
    file.visit_member_names(|name| {
        names.push(name.to_string());
        Ok(())
    })?;
    Ok(names)
}

fn group_member_names(group: &hdf5_pure_rust::Group) -> hdf5_pure_rust::Result<Vec<String>> {
    let mut names = Vec::new();
    group.visit_member_names(|name| {
        names.push(name.to_string());
        Ok(())
    })?;
    Ok(names)
}

fn open_real_world_fixture(path: &str) -> Option<File> {
    match File::open(path) {
        Ok(file) => Some(file),
        Err(err) => {
            eprintln!(
                "skipping optional real-world fixture {path}: {err}; run scripts/download-real-world-fixtures.py"
            );
            None
        }
    }
}

#[test]
fn test_real_world_anndata_h5ad_smoke() {
    let Some(f) = open_real_world_fixture("tests/data/real_world/anndataR_example.h5ad") else {
        return;
    };
    let members = file_member_names(&f).unwrap();
    for expected in ["X", "layers", "obs", "obsm", "obsp", "uns", "var"] {
        assert!(
            members.contains(&expected.to_string()),
            "missing {expected}"
        );
    }

    let x_data: Vec<f32> = read_vec(&f.dataset("X/data").unwrap()).unwrap();
    let x_indices: Vec<i32> = read_vec(&f.dataset("X/indices").unwrap()).unwrap();
    let x_indptr: Vec<i32> = read_vec(&f.dataset("X/indptr").unwrap()).unwrap();
    assert_eq!(x_data.len(), 4317);
    assert_eq!(x_indices.len(), 4317);
    assert_eq!(x_indptr.len(), 51);
    assert_eq!(x_indptr[0], 0);
    assert_eq!(*x_indptr.last().unwrap(), 4317);

    let dense_x: Vec<f32> = read_vec(&f.dataset("layers/dense_X").unwrap()).unwrap();
    assert_eq!(dense_x.len(), 50 * 100);

    let obs_index = read_strings_vec(&f.dataset("obs/_index").unwrap()).unwrap();
    let var_index = read_strings_vec(&f.dataset("var/_index").unwrap()).unwrap();
    assert_eq!(obs_index.len(), 50);
    assert_eq!(var_index.len(), 100);

    let pca: Vec<f32> = read_vec(&f.dataset("obsm/X_pca").unwrap()).unwrap();
    assert_eq!(pca.len(), 50 * 38);
}

#[test]
fn test_real_world_keras_h5_model_smoke() {
    let Some(f) = open_real_world_fixture("tests/data/real_world/keras_conv_mnist_tf_model.h5")
    else {
        return;
    };
    let members = file_member_names(&f).unwrap();
    assert!(members.contains(&"model_weights".to_string()));
    assert!(members.contains(&"optimizer_weights".to_string()));

    let conv_kernel: Vec<f32> = read_vec(
        &f.dataset("model_weights/conv2d_2/conv2d_2/kernel:0")
            .unwrap(),
    )
    .unwrap();
    let conv_bias: Vec<f32> =
        read_vec(&f.dataset("model_weights/conv2d_2/conv2d_2/bias:0").unwrap()).unwrap();
    let dense_kernel: Vec<f32> =
        read_vec(&f.dataset("model_weights/dense_1/dense_1/kernel:0").unwrap()).unwrap();
    let dense_bias: Vec<f32> =
        read_vec(&f.dataset("model_weights/dense_1/dense_1/bias:0").unwrap()).unwrap();

    assert_eq!(conv_kernel.len(), 3 * 3 * 1 * 32);
    assert_eq!(conv_bias.len(), 32);
    assert_eq!(dense_kernel.len(), 1600 * 10);
    assert_eq!(dense_bias.len(), 10);
    assert!(conv_kernel.iter().any(|v| *v != 0.0));
    assert!(dense_kernel.iter().any(|v| *v != 0.0));
}

#[test]
fn test_real_world_h5py_smoke() {
    let Some(f) = open_real_world_fixture("tests/data/real_world/h5py_3_12_smoke.h5") else {
        return;
    };
    let run = f.group("experiment/run_001").unwrap();
    assert_eq!(
        run.attr("temperature_c").unwrap().read_scalar_f64(),
        Some(21.5)
    );

    let image_stack: Vec<u16> =
        read_vec(&f.dataset("experiment/run_001/image_stack").unwrap()).unwrap();
    assert_eq!(image_stack, (0u16..24).collect::<Vec<_>>());

    let signal: Vec<f64> = read_vec(&f.dataset("experiment/run_001/signal").unwrap()).unwrap();
    assert_eq!(signal.len(), 25);
    assert!((signal[0] - 0.0).abs() < 1e-12);
    assert!((signal[24] - 1.0).abs() < 1e-12);

    let labels = read_strings_vec(&f.dataset("experiment/run_001/labels").unwrap()).unwrap();
    assert_eq!(labels, vec!["alpha", "βeta", "猫"]);

    let table = f.dataset("experiment/run_001/compound_table").unwrap();
    let fields = compound_field_names(&table).unwrap();
    assert_eq!(
        fields.iter().map(String::as_str).collect::<Vec<_>>(),
        vec!["id", "score"]
    );
    assert_eq!(read_field_vec::<i32>(&table, "id").unwrap(), vec![1, 2, 3]);
    assert_eq!(
        read_field_vec::<f64>(&table, "score").unwrap(),
        vec![0.5, 0.75, 1.25]
    );
}

#[test]
fn test_real_world_10x_feature_barcode_matrix_smoke() {
    let Some(f) = open_real_world_fixture(
        "tests/data/real_world/10x_pbmc_1k_v3_filtered_feature_bc_matrix.h5",
    ) else {
        return;
    };

    let members = file_member_names(&f).unwrap();
    assert!(members.contains(&"matrix".to_string()));

    let data: Vec<i32> = read_vec(&f.dataset("matrix/data").unwrap()).unwrap();
    let indices: Vec<i32> = read_vec(&f.dataset("matrix/indices").unwrap()).unwrap();
    let indptr: Vec<i32> = read_vec(&f.dataset("matrix/indptr").unwrap()).unwrap();
    let shape: Vec<i32> = read_vec(&f.dataset("matrix/shape").unwrap()).unwrap();
    let barcodes = read_strings_vec(&f.dataset("matrix/barcodes").unwrap()).unwrap();
    let feature_ids = read_strings_vec(&f.dataset("matrix/features/id").unwrap()).unwrap();

    assert_eq!(data.len(), indices.len());
    assert_eq!(shape.len(), 2);
    assert_eq!(barcodes.len(), shape[1] as usize);
    assert_eq!(feature_ids.len(), shape[0] as usize);
    assert_eq!(indptr.len(), barcodes.len() + 1);
    assert_eq!(indptr[0], 0);
    assert_eq!(*indptr.last().unwrap(), data.len() as i32);
}

#[test]
#[ignore = "reproduces vlen string read hang for files written by the current pure Rust writer"]
fn test_real_world_counthovd_sparse_matrix_strings() {
    let Some(f) = open_real_world_fixture("tests/data/real_world/counthovd.10.h5") else {
        return;
    };

    let data: Vec<u32> = read_vec(&f.dataset("X/data").unwrap()).unwrap();
    let indices: Vec<u64> = read_vec(&f.dataset("X/indices").unwrap()).unwrap();
    let indptr: Vec<u64> = read_vec(&f.dataset("X/indptr").unwrap()).unwrap();
    let shape: Vec<u32> = read_vec(&f.dataset("X/shape").unwrap()).unwrap();
    let obs_index = read_strings_vec(&f.dataset("obs/_index").unwrap()).unwrap();
    let var_index = read_strings_vec(&f.dataset("var/_index").unwrap()).unwrap();
    let unmapped: Vec<u32> = read_vec(&f.dataset("obs/_unmapped").unwrap()).unwrap();

    assert_eq!(shape, vec![665, 4537]);
    assert_eq!(data.len(), 17823);
    assert_eq!(indices.len(), data.len());
    assert_eq!(indptr.len(), shape[0] as usize + 1);
    assert_eq!(obs_index.len(), shape[0] as usize);
    assert_eq!(var_index.len(), shape[1] as usize);
    assert_eq!(unmapped.len(), shape[0] as usize);
    assert_eq!(indptr[0], 0);
    assert_eq!(*indptr.last().unwrap(), data.len() as u64);
    assert!(data.iter().any(|count| *count > 0));
    assert!(indices.iter().all(|index| *index < shape[1] as u64));
    assert!(obs_index.iter().all(|name| !name.is_empty()));
    assert!(var_index.iter().all(|name| !name.is_empty()));
}

#[test]
fn test_real_world_netcdf4_like_smoke() {
    let Some(f) = open_real_world_fixture("tests/data/real_world/netcdf4_like_climate.nc") else {
        return;
    };

    let lat: Vec<f32> = read_vec(&f.dataset("lat").unwrap()).unwrap();
    let lon: Vec<f32> = read_vec(&f.dataset("lon").unwrap()).unwrap();
    let temperature = f.dataset("temperature").unwrap();
    let values: Vec<f32> = read_vec(&temperature).unwrap();

    assert_eq!(lat, vec![-45.0, 0.0, 45.0]);
    assert_eq!(lon, vec![0.0, 90.0, 180.0, 270.0]);
    assert_eq!(shape_vec(&temperature).unwrap(), vec![3, 4]);
    assert_eq!(values.len(), 12);
    assert!((values[0] - 273.15).abs() < 1e-4);
}

#[test]
fn test_real_world_netcdf4_grouped_smoke() {
    let Some(f) = open_real_world_fixture("tests/data/real_world/netcdf4_grouped_ocean.nc") else {
        return;
    };

    let lat: Vec<f32> = read_vec(&f.dataset("coordinates/lat").unwrap()).unwrap();
    let lon: Vec<f32> = read_vec(&f.dataset("coordinates/lon").unwrap()).unwrap();
    let depth: Vec<f32> = read_vec(&f.dataset("coordinates/depth").unwrap()).unwrap();
    let time: Vec<i32> = read_vec(&f.dataset("coordinates/time").unwrap()).unwrap();
    let temperature = f.dataset("ocean/temperature").unwrap();
    let salinity = f.dataset("ocean/salinity").unwrap();
    let profile = f.dataset("ocean/profile").unwrap();

    let temperature_values: Vec<f32> = read_vec(&temperature).unwrap();
    let salinity_values: Vec<f32> = read_vec(&salinity).unwrap();
    let profile_values: Vec<f32> = read_vec(&profile).unwrap();

    assert_eq!(lat, vec![58.0, 59.5]);
    assert_eq!(lon, vec![18.0, 19.0, 20.0]);
    assert_eq!(depth, vec![0.0, 10.0, 25.0]);
    assert_eq!(time, vec![0, 6]);
    assert_eq!(shape_vec(&temperature).unwrap(), vec![2, 2, 3]);
    assert_eq!(shape_vec(&salinity).unwrap(), vec![2, 2, 3]);
    assert_eq!(shape_vec(&profile).unwrap(), vec![2, 3]);
    assert!((temperature_values[0] - 280.0).abs() < 1e-4);
    assert!((salinity_values[0] - 35.0).abs() < 1e-4);
    assert_eq!(
        profile_values,
        vec![280.0, 279.5, 279.0, 281.0, 280.4, 279.8]
    );
}

#[test]
fn test_real_world_matlab_v73_like_smoke() {
    let Some(f) = open_real_world_fixture("tests/data/real_world/matlab_v73_like.mat") else {
        return;
    };

    let a: Vec<f64> = read_vec(&f.dataset("A").unwrap()).unwrap();
    let name: Vec<u16> = read_vec(&f.dataset("name").unwrap()).unwrap();
    let cell_refs = f.dataset("cell").unwrap();

    assert_eq!(a, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(name, "hello".encode_utf16().collect::<Vec<_>>());
    assert_eq!(shape_vec(&cell_refs).unwrap(), vec![1]);
}

#[test]
fn test_real_world_nexus_smoke() {
    let Some(f) = open_real_world_fixture("tests/data/real_world/nexus_simple.nxs") else {
        return;
    };

    let members = file_member_names(&f).unwrap();
    assert!(members.contains(&"entry".to_string()));
    let counts: Vec<i32> =
        read_vec(&f.dataset("entry/instrument/detector/counts").unwrap()).unwrap();
    assert_eq!(counts, (0..12).collect::<Vec<_>>());
}

#[test]
fn test_real_world_nexus_rich_smoke() {
    let Some(f) = open_real_world_fixture("tests/data/real_world/nexus_rich_scan.nxs") else {
        return;
    };

    let entry = f.group("entry").unwrap();
    let members = group_member_names(&entry).unwrap();
    assert!(members.contains(&"data".to_string()));
    assert!(members.contains(&"instrument".to_string()));
    assert!(members.contains(&"sample".to_string()));

    let counts: Vec<i32> =
        read_vec(&f.dataset("entry/instrument/detector/counts").unwrap()).unwrap();
    let linked_counts: Vec<i32> = read_vec(&f.dataset("entry/data/counts").unwrap()).unwrap();
    let two_theta: Vec<f32> = read_vec(&f.dataset("entry/data/two_theta").unwrap()).unwrap();
    let frame: Vec<i32> = read_vec(&f.dataset("entry/data/frame").unwrap()).unwrap();
    let temperature: Vec<f32> = read_vec(&f.dataset("entry/sample/temperature").unwrap()).unwrap();

    assert_eq!(counts, (0..24).collect::<Vec<_>>());
    assert_eq!(linked_counts, counts);
    assert_eq!(two_theta, vec![10.0, 16.0, 22.0, 28.0, 34.0, 40.0]);
    assert_eq!(frame, vec![0, 1, 2, 3]);
    assert_eq!(temperature, vec![295.0]);
}

#[test]
fn test_real_world_pandas_hdfstore_smoke() {
    let Some(f) = open_real_world_fixture("tests/data/real_world/pandas_hdfstore_table.h5") else {
        return;
    };

    let observations = f.group("observations").unwrap();
    let members = group_member_names(&observations).unwrap();
    assert!(members.contains(&"table".to_string()));
    let table = f.dataset("observations/table").unwrap();
    assert_eq!(shape_vec(&table).unwrap()[0], 4);
}

#[test]
fn test_real_world_pandas_hdfstore_fixed_smoke() {
    let Some(f) = open_real_world_fixture("tests/data/real_world/pandas_hdfstore_fixed.h5") else {
        return;
    };

    let frame = f.group("fixed_frame").unwrap();
    let members = group_member_names(&frame).unwrap();
    for expected in [
        "axis0",
        "axis1",
        "block1_items",
        "block1_values",
        "block2_items",
        "block2_values",
    ] {
        assert!(
            members.contains(&expected.to_string()),
            "missing {expected}"
        );
    }

    let axis0 = read_strings_vec(&f.dataset("fixed_frame/axis0").unwrap()).unwrap();
    let axis1 = read_strings_vec(&f.dataset("fixed_frame/axis1").unwrap()).unwrap();
    let block1_items = read_strings_vec(&f.dataset("fixed_frame/block1_items").unwrap()).unwrap();
    let block1_values: Vec<i64> =
        read_vec(&f.dataset("fixed_frame/block1_values").unwrap()).unwrap();
    let block2_items = read_strings_vec(&f.dataset("fixed_frame/block2_items").unwrap()).unwrap();
    let block2_values: Vec<f64> =
        read_vec(&f.dataset("fixed_frame/block2_values").unwrap()).unwrap();

    assert_eq!(axis0, vec!["sample", "count", "score"]);
    assert_eq!(axis1, vec!["r0", "r1", "r2", "r3"]);
    assert_eq!(block1_items, vec!["count"]);
    assert_eq!(block1_values, vec![1, 3, 5, 7]);
    assert_eq!(block2_items, vec!["score"]);
    assert_eq!(block2_values, vec![0.25, 0.5, 1.0, 2.0]);
}

#[test]
fn test_real_world_pytables_native_smoke() {
    let Some(f) = open_real_world_fixture("tests/data/real_world/pytables_native_layout.h5") else {
        return;
    };

    let image_stack: Vec<u16> = read_vec(&f.dataset("measurements/image_stack").unwrap()).unwrap();
    let trace: Vec<f32> = read_vec(&f.dataset("measurements/trace").unwrap()).unwrap();
    let labels = read_strings_vec(&f.dataset("metadata/labels").unwrap()).unwrap();
    let events = f.dataset("measurements/events").unwrap();

    assert_eq!(image_stack, (0u16..24).collect::<Vec<_>>());
    assert_eq!(trace, vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5]);
    assert_eq!(labels, vec!["alpha", "beta", "gamma"]);

    let fields = compound_field_names(&events).unwrap();
    assert_eq!(
        fields.iter().map(String::as_str).collect::<Vec<_>>(),
        vec!["sample_id", "value", "quality"]
    );
    assert_eq!(
        read_field_vec::<i32>(&events, "sample_id").unwrap(),
        vec![1, 2, 3]
    );
    assert_eq!(
        read_field_vec::<f64>(&events, "value").unwrap(),
        vec![0.25, 0.5, 0.75]
    );
}

#[test]
fn test_real_world_pytables_nested_smoke() {
    let Some(f) = open_real_world_fixture("tests/data/real_world/pytables_nested_layout.h5") else {
        return;
    };

    let waveform: Vec<f64> = read_vec(&f.dataset("run_001/sensors/waveform").unwrap()).unwrap();
    let names = read_strings_vec(&f.dataset("run_001/metadata/names").unwrap()).unwrap();
    let active: Vec<u8> = read_vec(&f.dataset("run_001/metadata/active").unwrap()).unwrap();
    let summary = f.dataset("run_001/sensors/summary").unwrap();

    assert_eq!(
        waveform,
        vec![0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3]
    );
    assert_eq!(names, vec!["s0", "s1", "s2"]);
    assert_eq!(active, vec![1, 0, 1]);

    let fields = compound_field_names(&summary).unwrap();
    assert_eq!(
        fields.iter().map(String::as_str).collect::<Vec<_>>(),
        vec!["sensor_id", "mean", "status"]
    );
    assert_eq!(
        read_field_vec::<i32>(&summary, "sensor_id").unwrap(),
        vec![10, 11, 12]
    );
    assert_eq!(
        read_field_vec::<f32>(&summary, "mean").unwrap(),
        vec![1.5, 2.5, 3.5]
    );
}
