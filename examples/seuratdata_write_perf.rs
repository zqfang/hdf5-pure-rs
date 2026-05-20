use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use hdf5_pure_rust::engine::writer::{DatasetSpec, DtypeSpec, HdfFileWriter};
use hdf5_pure_rust::hl::types::slice_as_bytes;
use hdf5_pure_rust::{File, Result};

const DEFAULT_INPUT: &str = ".tmp/seurat_bench/pbmc3k_source.h5";
const DEFAULT_OUTPUT_PREFIX: &str = ".tmp/seurat_bench/rust_seurat_copy";
const H5PY_CHUNK_BASE_BYTES: f64 = 16.0 * 1024.0;
const H5PY_CHUNK_MIN_BYTES: f64 = 8.0 * 1024.0;
const H5PY_CHUNK_MAX_BYTES: f64 = 1024.0 * 1024.0;

struct ObsDataset {
    name: String,
    numeric: Option<Vec<f64>>,
    strings: Option<Vec<String>>,
}

struct SeuratDataFixture {
    data: Vec<f64>,
    indices: Vec<i32>,
    indptr: Vec<i32>,
    shape: Vec<i32>,
    obs_names: Vec<String>,
    var_names: Vec<String>,
    obs: Vec<ObsDataset>,
}

impl SeuratDataFixture {
    fn load(input: &Path) -> Result<Self> {
        let file = File::open(input)?;
        let obs_group = file.group("obs")?;
        let mut obs = Vec::new();
        for name in obs_group.member_names()? {
            let dataset = obs_group.dataset(&name)?;
            if let Ok(numeric) = dataset.read::<f64>() {
                obs.push(ObsDataset {
                    name,
                    numeric: Some(numeric),
                    strings: None,
                });
            } else {
                obs.push(ObsDataset {
                    name,
                    numeric: None,
                    strings: Some(dataset.read_strings()?),
                });
            }
        }
        obs.sort_by(|a, b| a.name.cmp(&b.name));

        Ok(Self {
            data: file.dataset("rna/data")?.read::<f64>()?,
            indices: file.dataset("rna/indices")?.read::<i32>()?,
            indptr: file.dataset("rna/indptr")?.read::<i32>()?,
            shape: file.dataset("rna/shape")?.read::<i32>()?,
            obs_names: file.dataset("rna/obs_names")?.read_strings()?,
            var_names: file.dataset("rna/var_names")?.read_strings()?,
            obs,
        })
    }

    fn write_copy(&self, output: &Path) -> Result<()> {
        if output.exists() {
            fs::remove_file(output)?;
        }
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent)?;
        }

        let file = fs::File::create(output)?;
        let mut writer = HdfFileWriter::new(file);
        writer.begin()?;
        writer.create_root_group()?;
        writer.create_group("/", "rna")?;
        writer.create_group("/", "obs")?;

        let data_chunk = [h5py_auto_chunk_len_1d(
            self.data.len(),
            std::mem::size_of::<f64>(),
        )];
        let indices_chunk = [h5py_auto_chunk_len_1d(
            self.indices.len(),
            std::mem::size_of::<i32>(),
        )];

        write_numeric(
            &mut writer,
            "/rna",
            "data",
            DtypeSpec::F64,
            slice_as_bytes(&self.data),
            &[u64_from_len(self.data.len())],
            Some(&data_chunk),
        )?;
        write_numeric(
            &mut writer,
            "/rna",
            "indices",
            DtypeSpec::I32,
            slice_as_bytes(&self.indices),
            &[u64_from_len(self.indices.len())],
            Some(&indices_chunk),
        )?;
        write_numeric(
            &mut writer,
            "/rna",
            "indptr",
            DtypeSpec::I32,
            slice_as_bytes(&self.indptr),
            &[u64_from_len(self.indptr.len())],
            None,
        )?;
        write_numeric(
            &mut writer,
            "/rna",
            "shape",
            DtypeSpec::I32,
            slice_as_bytes(&self.shape),
            &[u64_from_len(self.shape.len())],
            None,
        )?;
        write_strings(&mut writer, "/rna", "obs_names", &self.obs_names)?;
        write_strings(&mut writer, "/rna", "var_names", &self.var_names)?;

        let mut used = HashSet::new();
        for dataset in &self.obs {
            let name = unique_hdf5_name(&dataset.name, &mut used);
            if let Some(values) = &dataset.numeric {
                write_numeric(
                    &mut writer,
                    "/obs",
                    &name,
                    DtypeSpec::F64,
                    slice_as_bytes(values),
                    &[u64_from_len(values.len())],
                    None,
                )?;
            } else if let Some(values) = &dataset.strings {
                write_strings(&mut writer, "/obs", &name, values)?;
            }
        }

        writer.finalize()
    }
}

fn write_numeric(
    writer: &mut HdfFileWriter<fs::File>,
    parent: &str,
    name: &str,
    dtype: DtypeSpec,
    data: &[u8],
    shape: &[u64],
    chunk: Option<&[u64]>,
) -> Result<()> {
    let spec = DatasetSpec {
        name,
        shape,
        max_shape: None,
        dtype,
        data,
    };
    if let Some(chunk) = chunk {
        writer.create_chunked_dataset_with_attrs_and_fill(
            parent,
            &spec,
            chunk,
            None,
            false,
            false,
            None,
            &[],
        )?;
    } else {
        writer.create_dataset_with_attrs(parent, &spec, &[])?;
    }
    Ok(())
}

fn write_strings(
    writer: &mut HdfFileWriter<fs::File>,
    parent: &str,
    name: &str,
    values: &[String],
) -> Result<()> {
    let refs: Vec<&str> = values.iter().map(String::as_str).collect();
    writer.create_vlen_utf8_string_dataset(parent, name, &[u64_from_len(values.len())], &refs)?;
    Ok(())
}

fn unique_hdf5_name(name: &str, used: &mut HashSet<String>) -> String {
    let mut candidate: String = name
        .chars()
        .map(|ch| if ch == '/' || ch == '\0' { '_' } else { ch })
        .collect();
    if candidate.is_empty() {
        candidate.push_str("unnamed");
    }
    if used.insert(candidate.clone()) {
        return candidate;
    }
    for suffix in 1usize.. {
        let next = format!("{candidate}_{suffix}");
        if used.insert(next.clone()) {
            return next;
        }
    }
    unreachable!()
}

fn u64_from_len(len: usize) -> u64 {
    u64::try_from(len).expect("dataset length exceeds u64")
}

fn h5py_auto_chunk_len_1d(len: usize, element_size: usize) -> u64 {
    let element_size = element_size.max(1);
    let mut chunk_len = len.max(1) as f64;
    let dataset_bytes = len.max(1).saturating_mul(element_size) as f64;
    let mut target_bytes =
        H5PY_CHUNK_BASE_BYTES * 2.0_f64.powf((dataset_bytes / (1024.0 * 1024.0)).log10());
    target_bytes = target_bytes.clamp(H5PY_CHUNK_MIN_BYTES, H5PY_CHUNK_MAX_BYTES);

    loop {
        let chunk_bytes = chunk_len * element_size as f64;
        let close_enough = ((chunk_bytes - target_bytes) / target_bytes).abs() < 0.5;
        if (chunk_bytes < target_bytes || close_enough) && chunk_bytes < H5PY_CHUNK_MAX_BYTES {
            return chunk_len.max(1.0) as u64;
        }
        if chunk_len <= 1.0 {
            return 1;
        }
        chunk_len = (chunk_len / 2.0).ceil();
    }
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
    let data = SeuratDataFixture::load(&input)?;
    println!(
        "preload_ms={:.3} nnz={} cells={} genes={} obs_columns={}",
        preload_start.elapsed().as_secs_f64() * 1000.0,
        data.data.len(),
        data.obs_names.len(),
        data.var_names.len(),
        data.obs.len()
    );

    let mut times = Vec::with_capacity(iterations);
    for iteration in 0..iterations {
        let output = output_for(&output_prefix, iteration);
        let start = Instant::now();
        data.write_copy(&output)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        let size = fs::metadata(&output)?.len();
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
