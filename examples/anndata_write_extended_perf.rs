use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use hdf5_pure_rust::engine::writer::{DatasetSpec, DtypeSpec, HdfFileWriter};
use hdf5_pure_rust::hl::types::slice_as_bytes;
use hdf5_pure_rust::{File, Result};

const DEFAULT_INPUT: &str = ".tmp/h5ad_bench/ahlers_2022_young_human.h5";
const DEFAULT_OUTPUT_PREFIX: &str = ".tmp/h5ad_bench/rust_write_extended_copy";

const F32_DATASETS: &[(&str, Option<&[u64]>)] = &[("X/data", Some(&[21109]))];

const F64_DATASETS: &[(&str, Option<&[u64]>)] = &[
    ("obs/Age (mean)", None),
    ("obs/Condition (other)", None),
    ("obs/Exclusion reason", None),
];

const I64_DATASETS: &[(&str, Option<&[u64]>)] = &[("obs/Year", None)];

const U64_DATASETS: &[(&str, Option<&[u64]>)] = &[("obs/TotalUMIs", None)];

const U16_DATASETS: &[(&str, Option<&[u64]>)] = &[
    ("layers/matrix/data", Some(&[28688])),
    ("layers/spliced/data", Some(&[42218])),
    ("layers/unspliced/data", Some(&[21525])),
];

const I32_DATASETS: &[(&str, Option<&[u64]>)] = &[
    ("X/indices", Some(&[21109])),
    ("X/indptr", Some(&[2757])),
    ("layers/matrix/indices", Some(&[28688])),
    ("layers/matrix/indptr", Some(&[2757])),
    ("layers/spliced/indices", Some(&[21109])),
    ("layers/spliced/indptr", Some(&[2757])),
    ("layers/unspliced/indices", Some(&[21525])),
    ("layers/unspliced/indptr", Some(&[2757])),
    ("var/Accession", None),
    ("var/ChromosomeEnd", None),
    ("var/ChromosomeStart", None),
    ("var/FullName", None),
    ("var/HgncID", None),
    ("var/RefseqID", None),
];

const I16_DATASETS: &[(&str, Option<&[u64]>)] = &[
    ("var/Aliases", None),
    ("var/CcdsID", None),
    ("var/CosmicID", None),
    ("var/Location", None),
    ("var/LocationSortable", None),
    ("var/MgdID", None),
    ("var/MirBaseID", None),
    ("var/OmimID", None),
    ("var/PubmedID", None),
    ("var/RgdID", None),
    ("var/UcscID", None),
    ("var/UniprotID", None),
    ("var/VegaID", None),
];

const I8_DATASETS: &[(&str, Option<&[u64]>)] = &[
    ("obs/Accession (General)", None),
    ("obs/Accession (Sample)", None),
    ("obs/Acession (SRR)", None),
    ("obs/Age", None),
    ("obs/Age format (y/m)", None),
    ("obs/Aligner", None),
    ("obs/Analysed", None),
    ("obs/Author", None),
    ("obs/Condition", None),
    ("obs/DOI", None),
    ("obs/Data origin", None),
    ("obs/Donor identifier", None),
    ("obs/Downloaded", None),
    ("obs/Ethnicity", None),
    ("obs/Gender", None),
    ("obs/Genome", None),
    ("obs/Internal sample identifier", None),
    ("obs/Library preparation", None),
    ("obs/Organism", None),
    ("obs/Race", None),
    ("obs/Sample identifier", None),
    ("obs/Sample location", None),
    ("obs/Sequencer", None),
    ("var/Chromosome", None),
    ("var/DnaBindingDomain", None),
    ("var/GeneType", None),
    ("var/IsTF", None),
    ("var/LocusGroup", None),
    ("var/LocusType", None),
];

const STRING_DATASETS: &[&str] = &[
    "obs/__categories/Accession (General)",
    "obs/__categories/Accession (Sample)",
    "obs/__categories/Acession (SRR)",
    "obs/__categories/Age",
    "obs/__categories/Age format (y/m)",
    "obs/__categories/Aligner",
    "obs/__categories/Analysed",
    "obs/__categories/Author",
    "obs/__categories/Condition",
    "obs/__categories/DOI",
    "obs/__categories/Data origin",
    "obs/__categories/Donor identifier",
    "obs/__categories/Downloaded",
    "obs/__categories/Ethnicity",
    "obs/__categories/Gender",
    "obs/__categories/Genome",
    "obs/__categories/Internal sample identifier",
    "obs/__categories/Library preparation",
    "obs/__categories/Organism",
    "obs/__categories/Race",
    "obs/__categories/Sample identifier",
    "obs/__categories/Sample location",
    "obs/__categories/Sequencer",
    "obs/_index",
    "var/AccessionVersion",
    "var/__categories/Accession",
    "var/__categories/Aliases",
    "var/__categories/CcdsID",
    "var/__categories/Chromosome",
    "var/__categories/ChromosomeEnd",
    "var/__categories/ChromosomeStart",
    "var/__categories/CosmicID",
    "var/__categories/DnaBindingDomain",
    "var/__categories/FullName",
    "var/__categories/GeneType",
    "var/__categories/HgncID",
    "var/__categories/IsTF",
    "var/__categories/Location",
    "var/__categories/LocationSortable",
    "var/__categories/LocusGroup",
    "var/__categories/LocusType",
    "var/__categories/MgdID",
    "var/__categories/MirBaseID",
    "var/__categories/OmimID",
    "var/__categories/PubmedID",
    "var/__categories/RefseqID",
    "var/__categories/RgdID",
    "var/__categories/UcscID",
    "var/__categories/UniprotID",
    "var/__categories/VegaID",
    "var/_index",
];

struct NumericDataset<T> {
    path: &'static str,
    chunk: Option<&'static [u64]>,
    data: Vec<T>,
}

struct StringDataset {
    path: &'static str,
    data: Vec<String>,
}

struct ExtendedData {
    f32s: Vec<NumericDataset<f32>>,
    f64s: Vec<NumericDataset<f64>>,
    i64s: Vec<NumericDataset<i64>>,
    u64s: Vec<NumericDataset<u64>>,
    u16s: Vec<NumericDataset<u16>>,
    i32s: Vec<NumericDataset<i32>>,
    i16s: Vec<NumericDataset<i16>>,
    i8s: Vec<NumericDataset<i8>>,
    strings: Vec<StringDataset>,
}

impl ExtendedData {
    fn load(input: &Path) -> Result<Self> {
        let file = File::open(input)?;
        Ok(Self {
            f32s: load_numeric(&file, F32_DATASETS)?,
            f64s: load_numeric(&file, F64_DATASETS)?,
            i64s: load_numeric(&file, I64_DATASETS)?,
            u64s: load_numeric(&file, U64_DATASETS)?,
            u16s: load_numeric(&file, U16_DATASETS)?,
            i32s: load_numeric(&file, I32_DATASETS)?,
            i16s: load_numeric(&file, I16_DATASETS)?,
            i8s: load_numeric(&file, I8_DATASETS)?,
            strings: load_strings(&file)?,
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
        ensure_groups(&mut writer)?;

        write_numeric_group(&mut writer, &self.f32s, DtypeSpec::F32)?;
        write_numeric_group(&mut writer, &self.f64s, DtypeSpec::F64)?;
        write_numeric_group(&mut writer, &self.i64s, DtypeSpec::I64)?;
        write_numeric_group(&mut writer, &self.u64s, DtypeSpec::U64)?;
        write_numeric_group(&mut writer, &self.u16s, DtypeSpec::U16)?;
        write_numeric_group(&mut writer, &self.i32s, DtypeSpec::I32)?;
        write_numeric_group(&mut writer, &self.i16s, DtypeSpec::I16)?;
        write_numeric_group(&mut writer, &self.i8s, DtypeSpec::I8)?;
        for dataset in &self.strings {
            let (parent, name) = split_parent_name(dataset.path);
            let refs: Vec<&str> = dataset.data.iter().map(String::as_str).collect();
            writer.create_vlen_utf8_string_dataset(
                &parent,
                name,
                &[u64_from_len(refs.len())],
                &refs,
            )?;
        }

        writer.finalize()
    }
}

fn load_numeric<T: hdf5_pure_rust::hl::types::H5Type>(
    file: &File,
    paths: &[(&'static str, Option<&'static [u64]>)],
) -> Result<Vec<NumericDataset<T>>> {
    paths
        .iter()
        .map(|&(path, chunk)| {
            Ok(NumericDataset {
                path,
                chunk,
                data: file.dataset(path)?.read::<T>()?,
            })
        })
        .collect()
}

fn load_strings(file: &File) -> Result<Vec<StringDataset>> {
    STRING_DATASETS
        .iter()
        .map(|&path| {
            Ok(StringDataset {
                path,
                data: file.dataset(path)?.read_strings()?,
            })
        })
        .collect()
}

fn ensure_groups(writer: &mut HdfFileWriter<fs::File>) -> Result<()> {
    let mut groups = HashSet::new();
    for path in all_dataset_paths() {
        let mut current = String::from("/");
        for component in path
            .rsplit_once('/')
            .map(|(parent, _)| parent)
            .unwrap_or("")
            .split('/')
        {
            if component.is_empty() {
                continue;
            }
            let parent = current.clone();
            current = if parent == "/" {
                format!("/{component}")
            } else {
                format!("{parent}/{component}")
            };
            if groups.insert(current.clone()) {
                writer.create_group(&parent, component)?;
            }
        }
    }
    Ok(())
}

fn all_dataset_paths() -> impl Iterator<Item = &'static str> {
    F32_DATASETS
        .iter()
        .chain(F64_DATASETS)
        .chain(I64_DATASETS)
        .chain(U64_DATASETS)
        .chain(U16_DATASETS)
        .chain(I32_DATASETS)
        .chain(I16_DATASETS)
        .chain(I8_DATASETS)
        .map(|(path, _)| *path)
        .chain(STRING_DATASETS.iter().copied())
}

fn write_numeric_group<T: hdf5_pure_rust::hl::types::H5Type>(
    writer: &mut HdfFileWriter<fs::File>,
    datasets: &[NumericDataset<T>],
    dtype: DtypeSpec,
) -> Result<()> {
    for dataset in datasets {
        let (parent, name) = split_parent_name(dataset.path);
        let shape = [u64_from_len(dataset.data.len())];
        let bytes = slice_as_bytes(&dataset.data);
        let spec = DatasetSpec {
            name,
            shape: &shape,
            max_shape: None,
            dtype: dtype.clone(),
            data: bytes,
        };
        if let Some(chunk) = dataset.chunk {
            writer.create_chunked_dataset_with_attrs_and_fill(
                &parent,
                &spec,
                chunk,
                None,
                false,
                false,
                None,
                &[],
            )?;
        } else {
            writer.create_dataset_with_attrs(&parent, &spec, &[])?;
        }
    }
    Ok(())
}

fn split_parent_name(path: &str) -> (String, &str) {
    match path.rsplit_once('/') {
        Some((parent, name)) => {
            let parent = if parent.is_empty() {
                "/".to_string()
            } else {
                format!("/{parent}")
            };
            (parent, name)
        }
        None => ("/".to_string(), path),
    }
}

fn u64_from_len(len: usize) -> u64 {
    u64::try_from(len).expect("dataset length exceeds u64")
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
    println!("numeric_datasets=65 string_datasets=51 attrs=skipped");

    let preload_start = Instant::now();
    let data = ExtendedData::load(&input)?;
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
