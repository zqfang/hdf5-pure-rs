use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hdf5_pure_rust::{File, WritableFile};

fn unique_h5_path(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "hdf5-pure-rust-{name}-{}-{}.h5",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

fn create_chunked_fixture(path: &std::path::Path, filtered: bool) {
    let data: Vec<f64> = (0..100_000).map(|i| i as f64).collect();
    let mut file = WritableFile::create(path).unwrap();
    let builder = file
        .new_dataset_builder("data")
        .shape(&[data.len() as u64])
        .chunk(&[10_000]);
    if filtered {
        builder.deflate(1).write::<f64>(&data).unwrap();
    } else {
        builder.write::<f64>(&data).unwrap();
    }
    file.flush().unwrap();
}

fn bench_chunked_reads(c: &mut Criterion) {
    let path = unique_h5_path("chunked");
    create_chunked_fixture(&path, false);
    c.bench_function("chunked_read_f64", |b| {
        b.iter(|| {
            let file = File::open(&path).unwrap();
            let dataset = file.dataset("data").unwrap();
            let mut values = vec![0.0; dataset.size().unwrap() as usize];
            dataset.read_into(&mut values).unwrap();
            black_box(values);
        })
    });
    std::fs::remove_file(path).ok();
}

fn bench_filtered_reads(c: &mut Criterion) {
    let path = unique_h5_path("filtered");
    create_chunked_fixture(&path, true);
    c.bench_function("filtered_chunked_read_f64", |b| {
        b.iter(|| {
            let file = File::open(&path).unwrap();
            let dataset = file.dataset("data").unwrap();
            let mut values = vec![0.0; dataset.size().unwrap() as usize];
            dataset.read_into(&mut values).unwrap();
            black_box(values);
        })
    });
    std::fs::remove_file(path).ok();
}

fn bench_dense_group_traversal(c: &mut Criterion) {
    c.bench_function("dense_group_member_names", |b| {
        b.iter(|| {
            let file = File::open("tests/data/hdf5_ref/dense_group_cases.h5").unwrap();
            let group = file.group("name_index_deep").unwrap();
            let mut names = Vec::new();
            group
                .visit_member_names(|name| {
                    names.push(name.to_string());
                    Ok(())
                })
                .unwrap();
            black_box(names);
        })
    });
}

fn bench_vds_read(c: &mut Criterion) {
    c.bench_function("vds_all_read_i32", |b| {
        b.iter(|| {
            let file = File::open("tests/data/hdf5_ref/vds_all.h5").unwrap();
            let dataset = file.dataset("vds_all").unwrap();
            let mut values = vec![0; dataset.size().unwrap() as usize];
            dataset.read_into(&mut values).unwrap();
            black_box(values);
        })
    });
}

criterion_group!(
    benches,
    bench_chunked_reads,
    bench_filtered_reads,
    bench_dense_group_traversal,
    bench_vds_read
);
criterion_main!(benches);
