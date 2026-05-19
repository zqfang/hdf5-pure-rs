use std::time::Instant;

use hdf5_pure_rust::{hl::types::H5Type, File};

const DEFAULT_PATH: &str = ".tmp/h5ad_bench/ahlers_2022_young_human.h5";

fn best(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::INFINITY, f64::min)
}

fn avg(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn time_op<F>(iterations: usize, mut op: F) -> hdf5_pure_rust::Result<Vec<f64>>
where
    F: FnMut() -> hdf5_pure_rust::Result<usize>,
{
    let mut times = Vec::with_capacity(iterations);
    let mut len = 0usize;
    for _ in 0..iterations {
        let start = Instant::now();
        len = op()?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        times.push(elapsed);
    }
    println!(
        "    len={len} best={:.3}ms avg={:.3}ms",
        best(&times),
        avg(&times)
    );
    Ok(times)
}

fn bench_numeric<T>(
    file: &File,
    dataset_name: &str,
    iterations: usize,
) -> hdf5_pure_rust::Result<()>
where
    T: H5Type + Clone + Default,
{
    let ds = file.dataset(dataset_name)?;
    let info = ds.info()?;
    let elements = info.dataspace.dims.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(usize::try_from(dim).unwrap_or(usize::MAX))
            .ok_or_else(|| hdf5_pure_rust::Error::InvalidFormat("shape overflow".into()))
    })?;
    let raw_len = info
        .datatype
        .size
        .try_into()
        .ok()
        .and_then(|size: usize| size.checked_mul(elements))
        .ok_or_else(|| hdf5_pure_rust::Error::InvalidFormat("raw length overflow".into()))?;

    println!("{dataset_name} numeric full read:");
    println!("  read_raw()");
    time_op(iterations, || ds.read_raw().map(|v| v.len()))?;

    println!("  read_raw_into()");
    let mut raw_out = vec![0u8; raw_len];
    time_op(iterations, || {
        ds.read_raw_into(&mut raw_out)?;
        Ok(raw_out.len())
    })?;

    println!("  read::<T>()");
    time_op(iterations, || ds.read::<T>().map(|v| v.len()))?;

    println!("  read_into::<T>()");
    let mut typed_out = vec![T::default(); elements];
    time_op(iterations, || {
        ds.read_into(&mut typed_out)?;
        Ok(typed_out.len())
    })?;

    println!("  read_1d::<T>()");
    time_op(iterations, || ds.read_1d::<T>().map(|v| v.len()))?;

    let slice_len = elements.min(100_000);
    println!("  read_slice::<T>(0..{slice_len})");
    time_op(iterations, || {
        ds.read_slice::<T, _>(0..slice_len).map(|v| v.len())
    })?;

    Ok(())
}

fn bench_strings(file: &File, dataset_name: &str, iterations: usize) -> hdf5_pure_rust::Result<()> {
    let ds = file.dataset(dataset_name)?;
    println!("{dataset_name} strings:");

    println!("  read_strings()");
    time_op(iterations, || ds.read_strings().map(|v| v.len()))?;

    println!("  read_strings_into()");
    let mut out = Vec::new();
    time_op(iterations, || {
        ds.read_strings_into(&mut out)?;
        Ok(out.len())
    })?;

    println!("  visit_strings()");
    time_op(iterations, || {
        let mut count = 0usize;
        ds.visit_strings(|_| {
            count += 1;
            Ok(())
        })?;
        Ok(count)
    })?;

    Ok(())
}

fn main() -> hdf5_pure_rust::Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_PATH.to_string());
    let iterations = std::env::args()
        .nth(2)
        .and_then(|value| value.parse().ok())
        .unwrap_or(3usize)
        .max(1);

    println!("{path}: {iterations} iterations");
    let file = File::open(&path)?;

    bench_numeric::<f32>(&file, "X/data", iterations)?;
    bench_numeric::<i32>(&file, "X/indices", iterations)?;
    bench_numeric::<i32>(&file, "X/indptr", iterations)?;
    bench_numeric::<u16>(&file, "layers/matrix/data", iterations)?;
    bench_numeric::<u64>(&file, "obs/TotalUMIs", iterations)?;
    bench_numeric::<i8>(&file, "obs/Age", iterations)?;
    bench_numeric::<i32>(&file, "var/Accession", iterations)?;

    bench_strings(&file, "obs/_index", iterations)?;
    bench_strings(&file, "var/_index", iterations)?;
    bench_strings(&file, "var/AccessionVersion", iterations)?;
    bench_strings(&file, "var/__categories/FullName", iterations)?;
    bench_strings(&file, "obs/__categories/Age", iterations)?;

    Ok(())
}
