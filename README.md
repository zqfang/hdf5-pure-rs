# hdf5-pure-rust

Pure Rust implementation of the HDF5 file format. 

Based on HDF5 C library commit [`62701c4`](https://github.com/HDFGroup/hdf5/commit/62701c4c79775d267deedd15ed14d4c09571e792) (2026-04-10, v1.14.x branch). The machine-readable source pin is `hdf5-source.json`.

* 2026-05-06: **Available as early release for testers** -- Features are still missing, and more testing is needed. **Due to the risk of data corruption, be especially vigilant if you use this crate. **


## This is an LLM-mediated faithful (hopefully) translation, not the original code! 

Most users should probably first see if the existing original code works for them, unless they have reason otherwise. The original source
may have newer features and it has had more love in terms of fixing bugs. In fact, we aim to replicate bugs if they are present, for the
sake of reproducibility! (but then we might have added a few more in the process)

There are however cases when you might prefer this Rust version. We generally agree with [this manifesto](https://rewrites.bio/) but more specifically:
* We have had many issues with ensuring that our software works using existing containers (Docker, PodMan, Singularity). One size does not fit all and it eats our resources trying to keep up with every way of delivering software
* Common package managers do not work well. It was great when we had a few Linux distributions with stable procedures, but now there are just too many ecosystems (Homebrew, Conda). Conda has an NP-complete resolver which does not scale. Homebrew is only so-stable. And our dependencies in Python still break. These can no longer be considered professional serious options. Meanwhile, Cargo enables multiple versions of packages to be available, even within the same program(!)
* The future is the web. We deploy software in the web browser, and until now that has meant Javascript. This is a language where even the == operator is broken. Typescript is one step up, but a game changer is the ability to compile Rust code into webassembly, enabling performance and sharing of code with the backend. Translating code to Rust enables new ways of deployment and running code in the browser has especial benefits for science - researchers do not have deep pockets to run servers, so pushing compute to the user enables deployment that otherwise would be impossible
* Old CLI-based utilities are bad for the environment(!). A large amount of compute resources are spent creating and communicating via small files, which we can bypass by using code as libraries. Even better, we can avoid frequent reloading of databases by hoisting this stage, with up to 100x speedups in some cases. Less compute means faster compute and less electricity wasted
* LLM-mediated translations may actually be safer to use than the original code. This article shows that [running the same code on different operating systems can give somewhat different answers](https://doi.org/10.1038/nbt.3820). This is a gap that Rust+Cargo can reduce. Typesafe interfaces also reduce coding mistakes and error handling, as opposed to typical command-line scripting

But:

* **This approach should still be considered experimental**. The LLM technology is immature and has sharp corners. But there are opportunities to reap, and the genie is not going back into the bottle. This translation is as much aimed to learn how to improve the technology and get feedback on the results.
* Translations are not endorsed by the original authors unless otherwise noted. **Do not send bug reports to the original developers**. Use our Github issues page instead.
* **Do not trust the benchmarks on this page**. They are used to help evaluate the translation. If you want improved performance, you generally have to use this code as a library, and use the additional tricks it offers. We generally accept performance losses in order to reduce our dependency issues
* **Check the original Github pages for information about the package**. This README is kept sparse on purpose. It is not meant to be the primary source of information
* **If you are the author of the original code and wish to move to Rust, you can obtain ownership of this repository and crate**. Until then, our commitment is to offer an as-faithful-as-possible translation of a snapshot of your code. If we find serious bugs, we will report them to you. Otherwise we will just replicate them, to ensure comparability across studies that claim to use package XYZ v.666. Think of this like a fancy Ubuntu .deb-package of your software - that is how we treat it

This blurb might be out of date. Go to [this page](https://github.com/henriksson-lab/rustification) for the latest information and further information about how we approach translation



## Installation

```toml
[dependencies]
hdf5-pure-rust = "0.3"
```

## Quick Start

```rust
use hdf5_pure_rust::{File, WritableFile};

// Write
let mut wf = WritableFile::create("data.h5")?;
wf.new_dataset_builder("temperatures")
    .shape(&[1000])
    .chunk(&[100])
    .deflate(4)
    .write::<f64>(&values)?;
wf.flush()?;

// Read
let f = File::open("data.h5")?;
let ds = f.dataset("temperatures")?;
let values: Vec<f64> = ds.read::<f64>()?;

// Typed reads with ndarray
let arr = ds.read_1d::<f64>()?;        // Array1<f64>
let mat = ds.read_2d::<i32>()?;        // Array2<i32>

// Slicing
let subset: Vec<f64> = ds.read_slice::<f64, _>(10..20)?;

// Strings
let strings = ds.read_strings()?;       // Vec<String>

// Compound types
let x_vals: Vec<f64> = ds.read_field::<f64>("x")?;
```

## Features

| Area | Supported | Explicitly Unsupported |
|------|-----------|------------------------|
| Superblocks and object headers | Superblock v0-v3; object header v1/v2 with checksums | Full C-library metadata-cache behavior |
| Dataset storage | Compact, contiguous including external raw data files, chunked with v1 B-tree, v4 single-chunk datasets, unfiltered v4 implicit chunk indexes, v4 fixed-array chunk indexes, v4 extensible-array chunk indexes including data/super-block spillover, v4 v2-B-tree chunk indexes, and virtual datasets with all-selection, point, regular hyperslab, or irregular hyperslab mappings; VDS numeric source-to-destination datatype conversion; VDS view/prefix, fixed-shape missing-source fill behavior, and unlimited-dimension view sizing through `DatasetAccess` | Full libhdf5 datatype conversion parity |
| Filters | Deflate, shuffle, fletcher32, LZF, NBit, ScaleOffset, optional Blosc | SZip, unknown filters |
| Datatypes | Primitive numeric types including 128-bit integers, enum metadata, fixed/vlen strings, compound metadata, primitive compound field reads, raw compound member extraction, recursive compound field values for nested compound/array/vlen/reference members | Full HDF5 datatype conversion parity |
| Groups and links | v1 symbol tables, v2 link messages, dense link/attribute storage, filtered direct fractal heap reads, filtered and unfiltered huge direct/indirect fractal heap reads, soft/external links | Full coverage of every HDF5 index/storage variant |
| Writing | v2 superblock, compact/contiguous/chunked primitive numeric datasets including 128-bit integers, compact fixed-string, flat/nested compound, enum, opaque, array, and contiguous vlen UTF-8 string datasets, explicit fill-value messages, scalar/array/fixed-string/fixed-string-array compact and dense attributes on root groups, groups, and datasets, deflate/shuffle/Fletcher32 chunk filters, hard/soft/external links, limited `MutableFile::resize_dataset`, compact attribute delete/rename, writer-created dense attribute delete/same-size rename, v1 chunk B-tree append/replace/rebuild, selected v4 extensible-array and v2-B-tree chunk updates, deflate/shuffle/Fletcher32 filtered chunk replacement, and existing v4 fixed-array chunk replacement | Variable-length writer allocation beyond contiguous strings, modern chunk-index creation beyond v1 B-tree, full modern chunk-index mutation/creation matrix, dense attribute mutation for indirect/filtered heaps or creation-order indexes, free-space reuse, and general-purpose HDF5 writer parity with the C library |

**Reading:**
- Superblock v0-v3
- Object header v1 and v2 (with checksums)
- All storage layouts: compact, contiguous including external raw storage, chunked
- Chunk indices: v1 B-tree, single chunk, unfiltered v4 implicit, v4 fixed array, v4 extensible array including data/super-block spillover, and v4 v2-B-tree including internal nodes.
- Virtual datasets with serialized all-selection, point, regular hyperslab, or irregular hyperslab source and destination selections.
- Filters: deflate, shuffle, fletcher32, LZF, NBit, ScaleOffset, and optional Blosc. SZip and unknown filters return `Unsupported` for reads.
- All primitive types (i8-i128, u8-u128, f32, f64) with automatic big-endian byte-swap
- Compound and enum datatypes, including compound member index/offset/class/type queries
- Raw compound field extraction and recursive compound field values for non-primitive member payloads
- Fixed-length and variable-length dataset/attribute strings (via global heap)
- Attribute listing/info, index-based attribute name/info queries, attribute creation character encoding, tracked creation-order iteration, typed reads with numeric conversion, dataset layout/offset inspection, and datatype/dataspace inspection including numeric precision/offset and floating-point field metadata
- Groups with v1 symbol tables, v2 link messages, public link iteration, and tracked creation-order link iteration
- Dense link/attribute storage (fractal heap + v2 B-tree)
- Hard, soft, and external links
- File inspection with `File::file_size()` and `File::path()`
- Hyperslab selections: `ds.read_slice::<f64>(10..20)`
- ndarray integration: `ds.read_1d()`, `ds.read_2d()`

**Writing:**
- v2 superblock with Jenkins lookup3 checksums
- Compact groups and nested groups
- Primitive numeric datasets, including 128-bit integers, in contiguous, compact, and chunked storage
- Compact fixed-length string, compound, enum, opaque, and array datasets
- Explicit dataset fill-value messages with raw allocation-time and fill-time properties
- Compact primitive numeric attributes
- Deflate, shuffle, and Fletcher32 filters for chunked datasets
- Soft and external links
- New chunked datasets use v1 B-tree chunk indexes. `MutableFile::open_rw()` supports limited in-place dataset resizing, v1 chunk B-tree append/replace/rebuild, selected v4 extensible-array and v2-B-tree chunk updates, deflate/shuffle/Fletcher32 filtered chunk replacement, and replacement of existing v4 fixed-array chunks. Creating new fixed-array/extensible-array/v2-B-tree chunk indexes from scratch is not implemented.
- Writes append new metadata/raw data and do not implement libhdf5 free-space manager reuse.
- Verified readable by h5dump and h5py

**Other:**
- `#[derive(H5Type)]` for user-defined structs and enums
- Property list queries (`ds.create_plist()`, `attr.create_plist()`, file creation sizes/K-values)
- Most checked-in C-library reference files parse successfully; the exact count is enforced by tests rather than treated as a general compatibility guarantee.
- Zero panics on corrupt/malformed files (CVE regression tested)

## Benchmark

These numbers are for local development only. They are intended to guide
translation work and performance regressions, not to make broad claims about
general HDF5 performance.

Current local read baselines after the recent chunked-read optimizations:

1. Large 1D `f64` dataset, `32,000,000` elements, chunked `(1,000,000)`,
   gzip/deflate level `1`, no shuffle:

| Reader | Average Read Time | Best Read Time |
|--------|------------------:|---------------:|
| h5py/libhdf5 | 275.8 ms | 268.5 ms |
| hdf5-pure-rust | 338.2 ms | 326.1 ms |

The remaining gap on this workload is now mostly in the deflate backend rather
than in the HDF5 chunk-copy path. Profiling currently shows
`zlib_rs::inflate::inflate_fast_help_avx2` as the largest single hot symbol.

2. Large 1D `f64` dataset, `16,000,000` elements, chunked `(1,000,000)`,
   gzip/deflate level `1`, shuffle enabled:

| Reader | Average Read Time | Best Read Time |
|--------|------------------:|---------------:|
| h5py/libhdf5 | 172.6 ms | 167.0 ms |
| hdf5-pure-rust | 166.4 ms | 160.6 ms |

This second case improved substantially after specializing the shuffle reversal
path for common numeric element sizes and routing full 1D chunk reads directly
into the final output buffer.

For reproducible local timing, use:

```bash
scripts/run-read-benchmark.sh
```

For arbitrary fixture datasets by name, use the benchmark example directly:

```bash
cargo run --release --example perf_compare -- bench-read-raw tests/data/hdf5_ref/v4_extensible_array_spillover.h5 extensible_array_spillover 3
```

**These benchmarks must be taken with a huge grain of salt. HDF5 is a large, complex library with many features, so these measurements are primarily intended to guide further development and track regressions.**

## Derive Macro

```rust
use hdf5_pure_rust::DeriveH5Type;

#[derive(Copy, Clone, DeriveH5Type)]
#[repr(C)]
struct Measurement {
    time: f64,
    value: f32,
    #[hdf5(rename = "error_margin")]
    error: f32,
}
```

## Optional Features

| Feature | Default | Description |
|---------|---------|-------------|
| `derive` | yes | `#[derive(H5Type)]` proc macro |
| `blosc`  | no  | Blosc decompression via [`blosc2-pure-rs`](https://crates.io/crates/blosc2-pure-rs). Manually verified with `cargo test --features blosc blosc`. |
| `tracehash` | no | Internal trace probe hooks for local debugging. The old Rust-vs-HDF5-C corpus runner has been retired because it no longer matched the current crate surface. Not needed for normal builds. |

## Test Suite

The checked test count changes frequently; use `cargo test -- --list` for the current number. Coverage includes:
- Selected C library reference files and generated fixtures
- All primitive types, compound, enum, strings
- All storage layouts and filter combinations
- Corrupt file handling (zero panics, CVE regressions)
- Write round-trips verified by h5dump and h5py
- Cross-platform: big-endian, old formats, various file space strategies
- Optional real-world smoke tests for AnnData `.h5ad`, 10x Genomics feature-barcode matrices, Keras/TensorFlow `.h5`, h5py files, netCDF4-like files, MATLAB v7.3-like files, NeXus files, and pandas/PyTables HDFStore files

Real-world fixture payloads are intentionally not checked in. To populate them locally:

```bash
scripts/download-real-world-fixtures.py
cargo test --test real_world_test -- --nocapture
```

Use `scripts/download-real-world-fixtures.py --no-download` to regenerate only local producer fixtures without fetching public files. The pandas/PyTables fixture requires the Python `tables` package.

**This test suite needs to be expanded before any claims of general compatibility.**

Unsupported HDF5 features are tracked in `analysis/unsupported_features.md`.


## How to Cite HDF5

If you use HDF5 in your research, please cite it. See the original [original code](https://github.com/HDFGroup/hdf5) for details

**Quick DOI:** [10.5281/zenodo.17808558](https://doi.org/10.5281/zenodo.17808558)


## License

This is [derived work](https://github.com/HDFGroup/hdf5) and the license follows from the original HDF5 (BSD-3-Clause).
See the LICENSE file
