use std::time::Instant;

use hdf5_pure_rust::File;

#[derive(Default)]
struct Timings {
    raw_ms: Vec<f64>,
    strings_ms: Vec<f64>,
    raw_len: usize,
    strings_len: usize,
}

impl Timings {
    fn record(&mut self, raw_len: usize, raw_ms: f64, strings_len: usize, strings_ms: f64) {
        self.raw_len = raw_len;
        self.strings_len = strings_len;
        self.raw_ms.push(raw_ms);
        self.strings_ms.push(strings_ms);
    }
}

fn best(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::INFINITY, f64::min)
}

fn avg(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn time_strings(path: &str, dataset_name: &str, iterations: usize) -> hdf5_pure_rust::Result<()> {
    let file = File::open(path)?;
    let dataset = file.dataset(dataset_name)?;
    let mut timings = Timings::default();

    for _ in 0..iterations {
        let start = Instant::now();
        let raw = dataset.read_raw()?;
        let raw_ms = start.elapsed().as_secs_f64() * 1000.0;

        let start = Instant::now();
        let strings = dataset.read_strings()?;
        let strings_ms = start.elapsed().as_secs_f64() * 1000.0;

        timings.record(raw.len(), raw_ms, strings.len(), strings_ms);
    }

    println!(
        "{dataset_name}: raw={} bytes best={:.3}ms avg={:.3}ms, strings={} entries best={:.3}ms avg={:.3}ms",
        timings.raw_len,
        best(&timings.raw_ms),
        avg(&timings.raw_ms),
        timings.strings_len,
        best(&timings.strings_ms),
        avg(&timings.strings_ms)
    );
    Ok(())
}

fn main() -> hdf5_pure_rust::Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "tests/data/real_world/anndataR_example.h5ad".to_string());
    let iterations = std::env::args()
        .nth(2)
        .and_then(|value| value.parse().ok())
        .unwrap_or(5)
        .max(1);

    println!("{path}: {iterations} iterations");
    time_strings(&path, "obs/_index", iterations)?;
    time_strings(&path, "var/_index", iterations)?;
    Ok(())
}
