use std::path::PathBuf;
use std::time::Instant;

use hdf5_pure_rust::{File, H5Type, WritableFile};

fn parse_usize(name: &str, value: &str) -> usize {
    value
        .parse::<usize>()
        .unwrap_or_else(|_| panic!("invalid {name}: {value}"))
}

fn parse_f64(name: &str, value: &str) -> f64 {
    value
        .parse::<f64>()
        .unwrap_or_else(|_| panic!("invalid {name}: {value}"))
}

fn fill_data(len: usize) -> Vec<f64> {
    // Use a repeating but nontrivial pattern so the benchmark is not dominated
    // by a single degenerate compression case.
    (0..len)
        .map(|i| {
            let x = (i % 1024) as f64;
            x * 0.25 + ((i / 1024) % 17) as f64
        })
        .collect()
}

fn write_dataset(
    path: &PathBuf,
    len: usize,
    chunk: usize,
    deflate: Option<u32>,
) -> hdf5_pure_rust::Result<()> {
    let data = fill_data(len);
    let mut file = WritableFile::create(path)?;
    let builder = file
        .new_dataset_builder("data")
        .shape(&[len as u64])
        .chunk(&[chunk as u64]);
    if let Some(level) = deflate {
        builder.deflate(level).write::<f64>(&data)?;
    } else {
        builder.write::<f64>(&data)?;
    }
    file.flush()?;
    Ok(())
}

trait ChecksumValue: Copy + Default {
    fn checksum_value(self) -> f64;
}

impl ChecksumValue for f64 {
    fn checksum_value(self) -> f64 {
        self
    }
}

impl ChecksumValue for i32 {
    fn checksum_value(self) -> f64 {
        f64::from(self)
    }
}

impl ChecksumValue for i64 {
    fn checksum_value(self) -> f64 {
        self as f64
    }
}

fn read_dataset_typed<T: ChecksumValue + H5Type>(
    path: &PathBuf,
    dataset_name: &str,
) -> hdf5_pure_rust::Result<f64> {
    let file = File::open(path)?;
    let dataset = file.dataset(dataset_name)?;
    let mut values = vec![T::default(); dataset.size()? as usize];
    dataset.read_into(&mut values)?;
    Ok(values.iter().copied().map(T::checksum_value).sum())
}

fn read_dataset(path: &PathBuf, dataset_name: &str) -> hdf5_pure_rust::Result<f64> {
    read_dataset_typed::<f64>(path, dataset_name)
}

fn read_dataset_raw(path: &PathBuf, dataset_name: &str) -> hdf5_pure_rust::Result<f64> {
    let file = File::open(path)?;
    let dataset = file.dataset(dataset_name)?;
    let mut raw = vec![0; dataset.size()? as usize * dataset.element_size()?];
    dataset.read_raw_into(&mut raw)?;
    Ok(raw.iter().map(|&b| b as f64).sum())
}

fn benchmark_reads<T: ChecksumValue + H5Type>(
    path: &PathBuf,
    dataset_name: &str,
    target_seconds: f64,
) -> hdf5_pure_rust::Result<()> {
    let benchmark_start = Instant::now();
    let mut iterations = 0usize;
    let mut last_checksum = 0.0;
    let mut best_ms = f64::INFINITY;
    let mut total_ms = 0.0;

    while benchmark_start.elapsed().as_secs_f64() < target_seconds {
        let start = Instant::now();
        last_checksum = read_dataset_typed::<T>(path, dataset_name)?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        best_ms = best_ms.min(elapsed_ms);
        total_ms += elapsed_ms;
        iterations += 1;
    }

    let wall_ms = benchmark_start.elapsed().as_secs_f64() * 1000.0;
    let avg_ms = total_ms / iterations as f64;
    println!(
        "benchmark_read iterations={iterations} total_ms={total_ms:.3} wall_ms={wall_ms:.3} avg_ms={avg_ms:.3} best_ms={best_ms:.3} checksum={last_checksum:.1}"
    );
    Ok(())
}

fn benchmark_raw_reads(
    path: &PathBuf,
    dataset_name: &str,
    target_seconds: f64,
) -> hdf5_pure_rust::Result<()> {
    let benchmark_start = Instant::now();
    let mut iterations = 0usize;
    let mut last_checksum = 0.0;
    let mut best_ms = f64::INFINITY;
    let mut total_ms = 0.0;

    while benchmark_start.elapsed().as_secs_f64() < target_seconds {
        let start = Instant::now();
        last_checksum = read_dataset_raw(path, dataset_name)?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        best_ms = best_ms.min(elapsed_ms);
        total_ms += elapsed_ms;
        iterations += 1;
    }

    let wall_ms = benchmark_start.elapsed().as_secs_f64() * 1000.0;
    let avg_ms = total_ms / iterations as f64;
    println!(
        "benchmark_read iterations={iterations} total_ms={total_ms:.3} wall_ms={wall_ms:.3} avg_ms={avg_ms:.3} best_ms={best_ms:.3} checksum={last_checksum:.1}"
    );
    Ok(())
}

fn main() -> hdf5_pure_rust::Result<()> {
    let mut args = std::env::args().skip(1);
    let mode = args
        .next()
        .unwrap_or_else(|| {
            "usage: perf_compare <write|read|read-f64|read-i32|read-i64|bench-read|bench-read-f64|bench-read-i32|bench-read-i64|read-raw|bench-read-raw> <path> [dataset|len] [chunk|seconds] [deflate]"
                .into()
        });
    let path = PathBuf::from(
        args.next()
            .unwrap_or_else(|| panic!("missing path argument for mode {mode}")),
    );

    match mode.as_str() {
        "write" => {
            let len = parse_usize("len", &args.next().expect("missing len"));
            let chunk = parse_usize("chunk", &args.next().expect("missing chunk"));
            let deflate = args.next().map(|s| {
                s.parse::<u32>()
                    .unwrap_or_else(|_| panic!("invalid deflate level: {s}"))
            });
            let start = Instant::now();
            write_dataset(&path, len, chunk, deflate)?;
            println!("write_ms={:.3}", start.elapsed().as_secs_f64() * 1000.0);
        }
        "read" => {
            let dataset_name = args.next().unwrap_or_else(|| "data".to_string());
            let start = Instant::now();
            let checksum = read_dataset(&path, &dataset_name)?;
            println!(
                "read_ms={:.3} checksum={:.1}",
                start.elapsed().as_secs_f64() * 1000.0,
                checksum
            );
        }
        "read-f64" => {
            let dataset_name = args.next().unwrap_or_else(|| "data".to_string());
            let start = Instant::now();
            let checksum = read_dataset_typed::<f64>(&path, &dataset_name)?;
            println!(
                "read_ms={:.3} checksum={:.1}",
                start.elapsed().as_secs_f64() * 1000.0,
                checksum
            );
        }
        "read-i32" => {
            let dataset_name = args.next().unwrap_or_else(|| "data".to_string());
            let start = Instant::now();
            let checksum = read_dataset_typed::<i32>(&path, &dataset_name)?;
            println!(
                "read_ms={:.3} checksum={:.1}",
                start.elapsed().as_secs_f64() * 1000.0,
                checksum
            );
        }
        "read-i64" => {
            let dataset_name = args.next().unwrap_or_else(|| "data".to_string());
            let start = Instant::now();
            let checksum = read_dataset_typed::<i64>(&path, &dataset_name)?;
            println!(
                "read_ms={:.3} checksum={:.1}",
                start.elapsed().as_secs_f64() * 1000.0,
                checksum
            );
        }
        "bench-read" => {
            let dataset_name = args.next().unwrap_or_else(|| "data".to_string());
            let target_seconds = args
                .next()
                .map(|s| parse_f64("target_seconds", &s))
                .unwrap_or(5.0);
            benchmark_reads::<f64>(&path, &dataset_name, target_seconds)?;
        }
        "bench-read-f64" => {
            let dataset_name = args.next().unwrap_or_else(|| "data".to_string());
            let target_seconds = args
                .next()
                .map(|s| parse_f64("target_seconds", &s))
                .unwrap_or(5.0);
            benchmark_reads::<f64>(&path, &dataset_name, target_seconds)?;
        }
        "bench-read-i32" => {
            let dataset_name = args.next().unwrap_or_else(|| "data".to_string());
            let target_seconds = args
                .next()
                .map(|s| parse_f64("target_seconds", &s))
                .unwrap_or(5.0);
            benchmark_reads::<i32>(&path, &dataset_name, target_seconds)?;
        }
        "bench-read-i64" => {
            let dataset_name = args.next().unwrap_or_else(|| "data".to_string());
            let target_seconds = args
                .next()
                .map(|s| parse_f64("target_seconds", &s))
                .unwrap_or(5.0);
            benchmark_reads::<i64>(&path, &dataset_name, target_seconds)?;
        }
        "read-raw" => {
            let dataset_name = args.next().unwrap_or_else(|| "data".to_string());
            let start = Instant::now();
            let checksum = read_dataset_raw(&path, &dataset_name)?;
            println!(
                "read_ms={:.3} checksum={:.1}",
                start.elapsed().as_secs_f64() * 1000.0,
                checksum
            );
        }
        "bench-read-raw" => {
            let dataset_name = args.next().unwrap_or_else(|| "data".to_string());
            let target_seconds = args
                .next()
                .map(|s| parse_f64("target_seconds", &s))
                .unwrap_or(5.0);
            benchmark_raw_reads(&path, &dataset_name, target_seconds)?;
        }
        other => panic!("unknown mode: {other}"),
    }

    Ok(())
}
