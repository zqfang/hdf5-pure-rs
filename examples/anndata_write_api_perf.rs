use std::path::{Path, PathBuf};
use std::time::Instant;

use hdf5_pure_rust::{File, Result, WritableFile};

const DEFAULT_INPUT: &str = ".tmp/h5ad_bench/ahlers_2022_young_human.h5";
const DEFAULT_OUTPUT_PREFIX: &str = ".tmp/h5ad_bench/rust_write_copy";

struct FixtureData {
    x_data: Vec<f32>,
    x_indices: Vec<i32>,
    x_indptr: Vec<i32>,
    layer_data: Vec<u16>,
    total_umis: Vec<u64>,
    age: Vec<i8>,
    accession: Vec<i32>,
    obs_index: Vec<String>,
    var_index: Vec<String>,
    accession_version: Vec<String>,
    full_name: Vec<String>,
    age_categories: Vec<String>,
}

impl FixtureData {
    fn load(input: &Path) -> Result<Self> {
        let file = File::open(input)?;
        Ok(Self {
            x_data: file.dataset("X/data")?.read::<f32>()?,
            x_indices: file.dataset("X/indices")?.read::<i32>()?,
            x_indptr: file.dataset("X/indptr")?.read::<i32>()?,
            layer_data: file.dataset("layers/matrix/data")?.read::<u16>()?,
            total_umis: file.dataset("obs/TotalUMIs")?.read::<u64>()?,
            age: file.dataset("obs/Age")?.read::<i8>()?,
            accession: file.dataset("var/Accession")?.read::<i32>()?,
            obs_index: file.dataset("obs/_index")?.read_strings()?,
            var_index: file.dataset("var/_index")?.read_strings()?,
            accession_version: file.dataset("var/AccessionVersion")?.read_strings()?,
            full_name: file.dataset("var/__categories/FullName")?.read_strings()?,
            age_categories: file.dataset("obs/__categories/Age")?.read_strings()?,
        })
    }

    fn write_copy(&self, output: &Path) -> Result<()> {
        if output.exists() {
            std::fs::remove_file(output)?;
        }
        if let Some(parent) = output.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut wf = WritableFile::create(output)?;
        {
            let mut x = wf.create_group("X")?;
            x.new_dataset_builder("data")
                .shape(&[self.x_data.len() as u64])
                .chunk(&[21109])
                .write::<f32>(&self.x_data)?;
            x.new_dataset_builder("indices")
                .shape(&[self.x_indices.len() as u64])
                .chunk(&[21109])
                .write::<i32>(&self.x_indices)?;
            x.new_dataset_builder("indptr")
                .shape(&[self.x_indptr.len() as u64])
                .write::<i32>(&self.x_indptr)?;
        }
        {
            let mut layers = wf.create_group("layers")?;
            let mut matrix = layers.create_group("matrix")?;
            matrix
                .new_dataset_builder("data")
                .shape(&[self.layer_data.len() as u64])
                .chunk(&[28688])
                .write::<u16>(&self.layer_data)?;
        }
        {
            let mut obs = wf.create_group("obs")?;
            obs.new_dataset_builder("TotalUMIs")
                .shape(&[self.total_umis.len() as u64])
                .write::<u64>(&self.total_umis)?;
            obs.new_dataset_builder("Age")
                .shape(&[self.age.len() as u64])
                .write::<i8>(&self.age)?;
            write_vlen_strings(&mut obs, "_index", &self.obs_index)?;
            let mut categories = obs.create_group("__categories")?;
            write_vlen_strings(&mut categories, "Age", &self.age_categories)?;
        }
        {
            let mut var = wf.create_group("var")?;
            var.new_dataset_builder("Accession")
                .shape(&[self.accession.len() as u64])
                .write::<i32>(&self.accession)?;
            write_vlen_strings(&mut var, "_index", &self.var_index)?;
            write_vlen_strings(&mut var, "AccessionVersion", &self.accession_version)?;
            let mut categories = var.create_group("__categories")?;
            write_vlen_strings(&mut categories, "FullName", &self.full_name)?;
        }

        wf.close()?;
        Ok(())
    }
}

fn write_vlen_strings(
    group: &mut hdf5_pure_rust::hl::writable_file::WritableGroup<'_>,
    name: &str,
    values: &[String],
) -> Result<()> {
    let refs: Vec<&str> = values.iter().map(String::as_str).collect();
    group
        .new_dataset_builder(name)
        .shape(&[refs.len() as u64])
        .write_vlen_utf8_strings(&refs)
}

fn output_for(prefix: &Path, iteration: usize) -> PathBuf {
    let mut name = prefix.as_os_str().to_owned();
    name.push(format!("_{iteration}.h5"));
    PathBuf::from(name)
}

fn main() -> Result<()> {
    let input = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_INPUT));
    let output_prefix = std::env::args()
        .nth(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_OUTPUT_PREFIX));
    let iterations = std::env::args()
        .nth(3)
        .and_then(|value| value.parse().ok())
        .unwrap_or(3usize)
        .max(1);

    println!("input={}", input.display());
    println!("output_prefix={}", output_prefix.display());
    println!("iterations={iterations}");

    let preload_start = Instant::now();
    let data = FixtureData::load(&input)?;
    println!(
        "preload_ms={:.3}",
        preload_start.elapsed().as_secs_f64() * 1000.0
    );

    let mut times = Vec::with_capacity(iterations);
    for iteration in 0..iterations {
        let output = output_for(&output_prefix, iteration);
        let start = Instant::now();
        data.write_copy(&output)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        let size = std::fs::metadata(&output)?.len();
        println!(
            "write iteration={} ms={:.3} bytes={} path={}",
            iteration,
            elapsed,
            size,
            output.display()
        );
        times.push(elapsed);
    }
    let best = times.iter().copied().fold(f64::INFINITY, f64::min);
    let avg = times.iter().sum::<f64>() / times.len() as f64;
    println!("write_best_ms={best:.3} write_avg_ms={avg:.3}");

    Ok(())
}
