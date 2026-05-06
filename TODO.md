# TODO: hdf5-pure-rust

## Current status: 692 tests, 0 failures, 0 warnings.

## Audit Backlog (generated 2026-04-15)

### P0: Parser Robustness And No-Panic Guarantees
- [x] Reject datatype messages with truncated fixed-size properties: fixed-point, floating-point, and bitfield classes must have their required class property bytes.
- [x] Make compound datatype field parsing return a structured error instead of silently returning partial fields through `Option`.
- [x] Make enum datatype member parsing return a structured error instead of silently returning partial names or values through `Option`.
- [x] Make array datatype parsing reject truncated dimension tables and missing base datatypes with an error path that callers can surface.
- [x] Make variable-length datatype parsing distinguish HDF5 vlen string metadata from vlen sequence base datatype metadata without permissive fallback ambiguity.
- [x] Add malformed datatype regression vectors for truncated compound names, truncated v1/v2 compound dimension blocks, truncated member offsets, and truncated nested member datatypes.
- [x] Add malformed datatype regression vectors for truncated enum base datatype, truncated enum names, and truncated enum value payloads.
- [x] Add malformed datatype regression vectors for array datatypes with too many dimensions, overflowing dimension byte counts, and missing base datatypes.
- [x] Audit every object-header message decoder for unchecked indexing, partial zero defaults, and permissive trailing truncation.
- [x] Reject object-header continuation messages whose target range overflows file size or overlaps invalid metadata regions.
- [x] Reject malformed shared-message payloads explicitly instead of treating them as unknown or empty messages.
- [x] Reject invalid message version/class combinations with `InvalidFormat` rather than later high-level failures.
- [x] Add a focused fuzz-style test that truncates every byte prefix of representative datatype, dataspace, data layout, link, attribute, fill value, and filter pipeline messages.
- [x] Add a focused corrupt-file test that opens a synthetic file with every object-header message truncated at each byte boundary and asserts no panic.
- [x] Replace remaining production `unwrap()`/`expect()` reachable from file input with checked error propagation.
- [x] Audit arithmetic on dimensions, chunk counts, element sizes, and file offsets for overflow before allocation or seek.
- [x] Add allocation caps or overflow checks for declared rank, number of members, number of filters, number of chunks, heap object sizes, and VDS mapping counts.

### P0: HDF5-C Faithfulness And Divergence Tracking
- [x] Run the local tracehash Rust corpus and patched HDF5 C corpus after the new TODO backlog is introduced, then commit the current divergence report.
- [x] Add tracehash coverage for datatype property parsing details, not just high-level datatype class decode.
- [x] Add tracehash coverage for dataspace extent decode, selection decode, and serialized VDS selection decode.
- [x] Add tracehash coverage for fill-value message version/allocation/write-time transitions.
- [x] Add tracehash coverage for B-tree v1 chunk lookup decisions, including chunk coordinate key comparison.
- [x] Add tracehash coverage for v2 B-tree internal-node traversal and record decode decisions.
- [x] Add tracehash coverage for fixed-array and extensible-array chunk index address resolution.
- [x] Add tracehash coverage for fractal heap direct block, indirect block, managed object, huge object, and filtered object reads.
- [x] Add tracehash coverage for global heap object dereference and variable-length string reads.
- [x] Add tracehash coverage for external link resolution and same-file VDS source resolution.
- [x] Record known intentional divergences from libhdf5 in `analysis/unsupported_features.md` with test names for each.
- [x] Add a small script that fails CI when Rust-vs-C tracehash output diverges for the supported corpus.
- [x] Add a script target that regenerates all local HDF5 fixture files and emits the HDF5 C library version used.
- [x] Pin the HDF5 source commit used for fixture generation in one machine-readable file, not just prose.

### P1: Dataset Read Semantics
- [x] Implement point-selection virtual dataset mappings or explicitly reject them at decode time with a regression test.
- [x] Implement irregular hyperslab virtual dataset mappings or explicitly reject them at decode time with a regression test.
- [x] Implement VDS access-property behavior for missing source files, prefix substitution, and view policy, or document each unsupported mode with tests.
- [x] Extend VDS reads to non-`i32` primitive datatypes with conversion parity tests.
- [x] Add VDS tests for scalar dataspace mappings, null dataspace mappings, and zero-sized mappings.
- [x] Add VDS tests where source and destination rank differ but the mapping is valid under HDF5 rules.
- [x] Add VDS tests for overlapping mappings and verify libhdf5-compatible precedence.
- [x] Add a real on-disk VDS point-selection fixture. `h5py` / libhdf5 still reject authoring point mappings through `set_virtual()`, so the fixture is generated by creating a legal VDS first and then rewriting its single global-heap mapping object to a valid point-selection serialization. Covered by `test_virtual_dataset_point_selection_read`.
- [x] Add h5py-generated end-to-end VDS fixtures for irregular hyperslab block-list selections to complement the current decode/materialization unit coverage.
- [x] Add chunked partial-read tests that combine hyperslab selections with missing chunks and fill values.
- [x] Add chunked partial-read tests for filtered chunks where the filter mask skips one middle filter in a multi-filter pipeline.
- [x] Add Fletcher32 verification failure tests for corrupted filtered and unfiltered chunks.
- [x] Add big-endian filtered chunk read coverage for NBit and ScaleOffset.
- [x] Add compact dataset read tests for zero-sized dataspaces and scalar compound payloads.
- [x] Add contiguous dataset read tests for external storage files or explicitly mark external raw data storage unsupported.
- [x] Add tests for reading datasets whose declared storage address is the undefined HDF5 address.
- [x] Add tests for datasets with allocation-time-late storage and fill-time-never semantics.

### P1: Chunk Index Coverage
- [x] Add v1 B-tree chunk lookup tests for multidimensional chunk coordinates beyond 2D.
- [x] Add v1 B-tree tests for sparse chunks with non-monotonic insertion order.
- [x] Add v1 B-tree tests for large chunk offsets that require full address-size handling.
- [x] Add v2 B-tree chunk index tests with multiple internal levels, not just one internal root path.
- [x] Add v2 B-tree tests for filtered chunks with nonzero per-record filter masks.
  v2 B-tree chunk-record decoding now uses checked reader-position arithmetic
  for address, filtered-size, filter-mask, and scaled-coordinate fields.
- [x] Add fixed-array tests for all page initialization states, including absent pages and fill-value fallback.
- [x] Add extensible-array tests for secondary block addressing across index-block, data-block, and super-block transitions.
- [x] Add implicit chunk index tests for multidimensional datasets and partial edge chunks.
- [x] Reject filtered implicit chunk indexes with a targeted fixture and error assertion.
- [x] Add chunk index checksum corruption tests for fixed array, extensible array, and v2 B-tree metadata.
- [x] Audit chunk coordinate linearization against libhdf5 for rank, chunk-dim, and unlimited-dimension edge cases.

### P1: Filters
- [x] Add exact libhdf5 parity vectors for NBit signed integers, unsigned integers, floating point, and compound members.
- [x] Add exact libhdf5 parity vectors for ScaleOffset integer minbits, signed values, zero minbits, and floating point scale types.
- [x] Add malformed NBit parameter tests for impossible precision/offset combinations.
- [x] Add malformed ScaleOffset parameter tests for invalid scale type, missing client data, and output-size mismatch.
- [x] Make all filter decoders verify that output length exactly matches the expected logical chunk size unless the HDF5 filter semantics allow otherwise.
- [x] Add multi-filter pipeline tests for every supported filter order that libhdf5 can emit.
- [x] Add Blosc feature tests to CI or mark the feature as manually verified only.
- [x] Add tests that unknown optional filters are skipped only when HDF5 semantics allow skipping, and required unknown filters fail.
- [x] Keep SZip unsupported unless a pure-Rust decoder is added, but add a fixture asserting the exact error surface.

### P1: Datatype And Conversion Semantics
- [x] Replace ad hoc high-level datatype conversion with a central conversion table modeled after libhdf5 conversion classes.
- [x] Add integer conversion tests for signed/unsigned widening, narrowing, and overflow behavior.
- [x] Add float conversion tests for f32/f64, integer-to-float, float-to-integer, NaN, infinity, and endian-swapped payloads.
- [x] Add fixed-length string tests for null padding, space padding, null termination, and UTF-8 character set flags.
- [x] Add variable-length string tests for empty strings, null strings, UTF-8 strings, and global heap edge cases.
- [x] Add opaque datatype tests including tag decode and raw payload reads.
- [x] Add reference datatype tests for object references and dataset region references.
- [x] Add time datatype tests or explicitly reject HDF5 time datatype reads.
- [x] Add enum conversion tests where the enum base type is wider than one byte and big-endian.
- [x] Add compound tests for member padding, overlapping members, reordered members, and nested variable-length members.
- [x] Add array datatype tests for v1/v2/v3/v4 encodings and multidimensional array fields.

### P1: Groups, Links, Attributes, And Heaps
- [x] Add dense group tests with multiple v2 B-tree levels for link-name indexing.
- [x] Add dense group tests with creation-order indexing enabled and disabled.
- [x] Add dense attribute tests with creation-order indexing enabled and disabled.
- [x] Add attribute read tests for large compact attributes, dense attributes, and variable-length attribute payloads.
- [x] Add link tests for UTF-8 names, non-ASCII external filenames, and invalid character-set flags.
- [x] Add soft-link cycle detection tests with a bounded traversal limit.
- [x] Add external-link tests for missing files, relative paths, absolute paths, and same-directory resolution.
- [x] Add global heap tests with deleted objects, duplicate object IDs, and collection padding.
- [x] Add fractal heap tests for indirect block growth beyond one level.
- [x] Add fractal heap checksum corruption tests for direct and indirect blocks.
- [x] Add filtered fractal heap tests using non-deflate filters where libhdf5 can generate them.

### P1: Writer And Mutable File Semantics
- [x] Update README write-support claims to mention current `MutableFile` v4 fixed-array replacement support and remaining writer gaps precisely.
- [x] Add writer support for creating datasets with fill-value messages and allocation-time/fill-time properties.
- [x] Add writer support for compact datasets beyond primitive numeric payloads, including fixed strings and compound values.
- [x] Add writer support for creating dense groups after compact link thresholds are exceeded.
- [x] Add writer support for dense attributes after compact attribute thresholds are exceeded.
- [x] Add writer support for variable-length strings via global heap allocation.
- [x] Add writer support for enum, opaque, array, and nested compound datatype messages.
- [x] Add writer support for chunked datasets with fixed-array or extensible-array indexes, or explicitly keep v1-only writer indexes documented.
- [x] Add mutable append/replacement support for simple depth-0 v4 v2-B-tree chunk indexes.
- [x] Add mutable append support for v4 extensible-array chunk indexes through the first data-block allocation.
- [x] Add mutable append support for multi-level v2-B-tree rebuilds after chunk-index appends.
- [x] Add mutable append support for v4 extensible-array super-block/page growth.
- [x] Add mutable replacement support for filtered chunks while preserving or recomputing per-chunk filter masks.
- [x] Add shrink tests that ensure removed chunks are no longer returned after `MutableFile::resize_dataset`.
- [x] Add grow tests that ensure newly exposed regions use the correct fill value for contiguous and chunked datasets.
- [x] Add writer round-trip tests validated by both h5dump and h5py for each newly supported writer feature.
- [x] Decide whether free-space managers are intentionally unsupported for writes and document the resulting file-growth behavior.

### P1: Public API And Error Surface
- [x] Replace parser helper APIs that return `Option` for malformed file data with `Result` carrying `InvalidFormat` or `Unsupported`.
- [x] Ensure all public read APIs distinguish unsupported HDF5 features from corrupt file data.
- [x] Add stable error-message tests only where callers are likely to branch on the error class.
- [x] Add `Dataset::read_dyn` or equivalent for N-dimensional ndarray reads beyond 1D/2D.
- [x] Add safe APIs for inspecting raw datatype messages, raw dataspace messages, and creation properties for unsupported features.
- [x] Add API docs warning that `read::<T>()` is not full libhdf5 conversion parity yet.
- [x] Add API docs for `read_field_raw()` and recursive compound `read_field_values()` limitations.
- [x] Audit exported types for accidental exposure of internal layout structs that may need semver stability.
- [x] Add examples for read-only use, write use, VDS use, and tracehash divergence checks.

### P2: Test Corpus, CI, And Documentation
- [x] Update README test count from 294 to the current count, or generate the count dynamically in release notes instead of hard-coding it.
- [x] Move generated or vendored HDF5 reference artifacts behind documented regeneration scripts and avoid committing local-only scratch output.
- [x] Decide whether the vendored `hdf5/` source tree belongs in the repository; document if it is required for tracehash and fixture generation.
- [x] Add CI jobs for default features, `--no-default-features`, `--features derive`, `--features blosc`, and `--features tracehash` where practical.
- [x] Add a CI job that runs tests under Miri or sanitizers for parser-only units where dependencies permit it.
- [x] Add cargo-deny or equivalent dependency/license checks.
- [x] Add cargo-semver-checks before releases.
- [x] Add criterion benchmarks that use generated fixture files and do not write fixed `/tmp` paths.
- [x] Replace benchmark hard-coded `/tmp/bench_rust.h5` with a unique temporary path.
- [x] Add performance baselines for chunked reads, filtered reads, dense group traversal, and VDS reads.
- [x] Add a compatibility matrix that maps each supported feature to at least one fixture and one test.
- [x] Add release checklist documenting fixture regeneration, tracehash comparison, test matrix, README count update, and crates.io packaging.
- [x] Audit untracked local files before any commit and classify each as source, generated fixture, vendored dependency, scratch output, or ignore rule.
- [x] Add `.gitignore` entries for local tracehash outputs, temporary HDF5 files, and generated reports that should never be committed.

## Next Fixes
- [x] Implement readable extensible-array chunk index data/super-block spillover.
- [x] Implement virtual dataset reads from parsed v3/v4 virtual layout mappings for regular hyperslab selections.
- [x] Extend `MutableFile::write_chunk` to replace existing chunks in leaf v1 B-tree indexes.
- [x] Extend `MutableFile::write_chunk` to handle full v1 B-tree leaf rebalancing.
- [x] Add `MutableFile` support for updating v4 fixed-array chunk indexes when replacing existing chunks.
- [x] Implement paged fixed-array chunk index data blocks.
- [x] Keep filtered v4 implicit chunk indexes explicitly unsupported; HDF5 does not normally choose implicit indexes for filtered datasets. Covered by an on-disk regression that patches a filtered fixed-array fixture into a filtered implicit-index layout while repairing the object-header checksum.
- [x] Implement filtered directly addressed huge fractal-heap object reads.
- [x] Keep documenting SZip as permanently unsupported unless a pure-Rust decoder is added later.
- [x] Broaden the tracehash corpus with fixtures for extensible-array and paged fixed-array spillover.
- [x] Triage tracehash corpus expansion for virtual datasets and writer-side chunk-index updates; VDS and writer-side behavior now have regression tests, while default tracehash parity intentionally waits for dedicated patched-C VDS/writer probes.
- [x] Resolve supported virtual dataset shapes from regular hyperslab mappings instead of returning stored placeholder extents.
- [x] Implement virtual dataset reads for serialized `H5S_SEL_ALL` source and destination mappings.
- [x] Add virtual dataset same-file source (`"."`) coverage.
- [x] Add virtual dataset mixed `H5S_SEL_ALL` and regular hyperslab selection coverage.
- [x] Honor defined virtual dataset fill values for unmapped regions.
- [x] Honor defined chunked dataset fill values for missing/unallocated chunks.
- [x] Honor per-chunk filter masks when reversing filtered chunk pipelines.
- [x] Reject per-chunk filter masks that reference filters outside the pipeline.
- [x] Reject filter pipelines longer than the 32-bit per-chunk filter mask can represent.
- [x] Add v4 filtered single-chunk filter-mask coverage.
- [x] Add dataset creation property coverage for parsed fill values.
- [x] Add old-format fill-value read and property-list coverage.
- [x] Add reference `fill18.h5` chunked fill-value read coverage.
- [x] Reject truncated defined fill-value message payloads.
- [x] Reject truncated data layout message payloads without panics or silent partial reads.
- [x] Reject truncated dense link/attribute info message addresses instead of decoding partial zeros.
- [x] Reject truncated filter pipeline message payloads instead of returning partial decoded pipelines.
- [x] Reject dataspace messages whose declared rank dimensions or max dimensions are truncated.
- [x] Reject truncated attribute message name/datatype/dataspace metadata sections before slicing.
- [x] Reject truncated link message optional fields and variable-width lengths before reading.
- [x] Make write tests hermetic by replacing fixed `tests/data/*.h5` output paths with `tempfile::tempdir()` or unique per-test paths.
- [x] Remove or isolate the absolute local `tracehash` path dependency so normal Cargo metadata is portable.
- [x] Add a concise README supported/unsupported feature table to prevent users from inferring full HDF5 compatibility.
- [x] Ensure generated test artifacts never persist in `tests/data`, including after failed tests.
- [x] Extend compound datatype support beyond primitive `read_field::<T>()` with `read_field_raw()` for nested compound, array, variable-length, and reference member payloads; recursive typed conversion remains explicitly unsupported.
- [x] Implement readable v4 implicit chunk indexes for unfiltered contiguous chunk storage; fixed array, extensible array, v2 B-tree, and filtered implicit indexes remain explicitly unsupported.
- [x] Vendor tracehash locally and add a patched-C runner that emits `/tmp/c.tsv`; matching HDF5 C-side probe targets remain documented for patched C builds.

## Next Improvements
- [x] Implement readable v4 fixed-array chunk indexes.
- [x] Implement readable v4 extensible-array chunk indexes for direct index-block entries and data/super-block spillover.
- [x] Implement readable v4 v2-B-tree chunk indexes for leaf-root chunk trees.
- [x] Implement filtered v4 fixed-array chunk indexes; HDF5 does not select implicit chunk indexes for filtered datasets.
- [x] Implement recursive typed compound conversion for nested compound members.
- [x] Implement recursive typed compound conversion for array members.
- [x] Implement recursive typed compound conversion for variable-length members.
- [x] Implement recursive typed compound conversion for reference members.
- [x] Add concrete HDF5 C-side tracehash probes in datatype decode paths (`H5T*`).
- [x] Add concrete HDF5 C-side tracehash probes in object-header decode paths (`H5O*`).
- [x] Add concrete HDF5 C-side tracehash probes in filter pipeline decode/application paths (`H5Z*`).
- [x] Add concrete HDF5 C-side tracehash probes for chunk index resolution.
- [x] Add concrete HDF5 C-side tracehash probes for fractal heap lookup and dense link/attribute traversal.
- [x] Generate Rust-vs-C tracehash divergence report: Rust emits `/tmp/rust.tsv`, patched HDF5 C emits `/tmp/c.tsv`, and the current instrumented corpus matches row-for-row with no output mismatches.
- [x] Implement v2 B-tree internal-node traversal.
- [x] Implement filtered fractal heap object lookup.
- [x] Implement huge fractal heap object lookup.
- [x] Implement datatype-aware NBit filter decoding.
- [x] Implement datatype-aware ScaleOffset filter decoding.
- [x] Extend `MutableFile` resize support to update chunk indexes when new chunks are appended.
- [x] Add C-generated fixture files for v4 fixed-array chunk indexes.
- [x] Add C-generated fixture files for paged v4 fixed-array chunk indexes.
- [x] Add C-generated fixture files for v4 extensible-array chunk indexes.
- [x] Add C-generated fixture files for v4 v2-B-tree chunk indexes.
- [x] Add C-generated fixture files for filtered chunk indexes.
- [x] Add C-generated fixture files for modern dense fractal heap coverage.
- [x] Audit untracked repo artifacts before commit, especially `.codex`, vendored `hdf5/`, `analysis/`, `scripts/`, and `tools/`.
- [x] Refresh tracehash documentation after the Rust-vs-C corpus reaches an exact row-for-row match.

### Core Format Engine (Phases 1-8)
- [x] Binary I/O primitives, superblock v0-v3, Jenkins lookup3 checksum
- [x] Object header v1/v2 parsing, v1 B-tree, local heap, symbol table
- [x] Dataset reading (contiguous/compact/chunked/compressed)
- [x] Attribute reading (v1/v2/v3)
- [x] File writing (superblock, groups, contiguous datasets, C library verified)
- [x] Chunked writing with deflate/shuffle compression
- [x] Compatibility fixes for supported write paths (float datatype encoding, B-tree padding)
- [x] Fractal heap + leaf-root v2 B-tree support for dense link/attr storage, global heap

### High-Level API (Phases A-E)
- [x] `H5Type` trait + generic reads: `ds.read::<f64>()`, `read_scalar`, `read_1d`, `read_2d`
- [x] Datatype/Dataspace public API: `ds.dtype()`, `ds.space()`, `ds.is_chunked()`, etc.
- [x] Write-through-API: `WritableFile::create()`, `DatasetBuilder`, `WritableGroup`
- [x] Selection/Hyperslab: `ds.read_slice::<f64>(10..20)`, 1D/2D/chunked
- [x] Property lists: `DatasetCreate`, `FileCreate`, `ds.create_plist()`

### Extended Features (Phases F-H + extras)
- [x] `Location` trait on File/Group/Dataset
- [x] Soft/external link read/write
- [x] String reading (fixed-length + variable-length via global heap)
- [x] Big-endian type conversion
- [x] Compound/enum datatype reading
- [x] LZF, NBit, and ScaleOffset filters; SZip now fails explicitly as unsupported on reads
- [x] Blosc filter (feature-gated `blosc2-rs`)
- [x] H5Type derive macro (separate proc-macro crate)
- [x] Limited in-place dataset resizing via `MutableFile`
- [x] Virtual dataset layout parsing (v3/v4); virtual dataset reads fail explicitly as unsupported

### HDF5 C Test Suite Ported (Phases T1-T11)
- [x] T1: Reference file smoke tests for checked-in corpus, 32 tests
- [x] T2: Corrupt file handling -- no panics on checked-in corpus + CVE regressions, 9 tests
- [x] T3: All datatypes -- i8-i64, u8-u64, f32, f64, BE, compound, enum, strings, N-D, 22 tests
- [x] T4: Dataset layouts -- compact/contiguous/chunked, deflate/shuffle/fletcher32, selections, 16 tests
- [x] T5: Attributes -- scalar, array, string, group/dataset attrs, dense storage
- [x] T6: Groups & links -- nested, hard/soft/external, dense, link_exists, 8 tests
- [x] T7: Dataspace & selections -- scalar/simple/null, maxdims, 1D/2D slices, chunked slices, 16 tests
- [x] T8: Object headers -- v1/v2, timestamps, continuation chunks, all message types, 7 tests
- [x] T9: Heaps & indices -- global heap, local heap, fractal heap, v2 B-tree, chunk indices, 10 tests
- [x] T10: Write round-trips -- h5dump, h5py, all types/layouts/filters, resize, 9 tests
- [x] T11: Cross-platform -- big-endian, old formats, file space strategies, charsets, 12 tests

### Faithfulness Audit vs HDF5 C Library
- [x] Replace broad "bitwise compatible" wording with a precise supported-feature compatibility statement.
- [x] Reconcile license metadata: Cargo.toml now uses BSD-3-Clause to match README/LICENSE.
- [x] Update README reference-file claims so checked-in corpus changes do not imply broad compatibility.
- [x] Add negative tests for explicitly unsupported paths: unsupported filters, virtual layout metadata-only parsing, huge fractal-heap objects, and filtered fractal heaps.
- [x] Clearly document NBit and ScaleOffset status; generic filter pipeline now decodes datatype-aware NBit/ScaleOffset parameters instead of pass-through.
- [x] Clearly document unsupported chunk index types; reads now return `Unsupported` instead of falling back to v1 B-tree for implicit, fixed array, extensible array, and v2 B-tree chunk indexes.
- [x] Remove speculative v2 B-tree internal-node parsing; internal v2 B-trees now return `Unsupported`.
- [x] Implement direct filtered managed fractal heap object lookup and direct/indirect huge object lookup.
- [x] Preserve embedded compound member datatypes and implement big-endian primitive member byte swapping; recursive high-level conversion for nested/array/vlen/reference fields is documented as unsupported.
- [x] Extend `MutableFile` from resize-only metadata updates to append chunks into leaf v1 chunk indexes written by this crate.
- [x] Keep v4 chunk indexes explicitly unsupported except for single-chunk datasets; no fallback to incorrect readers.
- [x] Keep virtual dataset reads explicitly unsupported while preserving layout metadata parsing.
- [x] Keep datatype-aware NBit and ScaleOffset decoding in the generic filter pipeline.
- [x] Keep SZip permanently unsupported unless a pure-Rust decoder is added later.

### Tracehash Divergence Tracking
- [x] Document the vendored tracehash path: `tools/tracehash`.
- [x] Add an optional `tracehash` feature for Rust-side probes without enabling it in normal builds.
- [x] Switch the Rust-side `tracehash` dependency to the published
  [`tracehash-rs`](https://crates.io/crates/tracehash-rs) 0.1 crate
  (`package = "tracehash-rs"` rename so call sites keep importing
  `tracehash::...`). Updated `output_bool`/`output_bytes` call sites to
  the new `output_value(&T)` API since the published crate dropped the
  explicit helpers in favor of the generic `TraceHash`-based path.
  `scripts/tracehash-compare.sh` now prefers a `tracehash-compare`
  binary on `$PATH` (installed via `cargo install tracehash-rs`) and
  falls back to `cargo run --package tracehash-rs`.

### Format-layer 1:1 mapping refactor (2026-04-18)

- [x] Split fused decode+traverse functions across `src/format/` so each
  half maps cleanly to its libhdf5 counterpart. Pure prefix-deserialize
  helpers now live as `decode_*` (mirroring `H5*_cache_*_deserialize`),
  with the existing `read_*` entry points either retained as thin
  compose wrappers (where backward compatibility matters) or replaced
  by separate traversal halves. Files touched:
  - `format/local_heap.rs`: `decode_prefix` + `load_data_segment`.
  - `format/global_heap.rs`: `decode_header` + `walk_objects`.
  - `format/btree_v2.rs`: `decode_internal_node` (was inlined into
    `read_internal_records`).
  - `format/fixed_array.rs`: `decode_data_block_prefix` +
    `collect_data_block_elements`.
  - `format/extensible_array.rs`: `decode_data_block_prefix`,
    `decode_super_block`, `decode_index_block`.
  - `format/fractal_heap.rs`: `decode_indirect_block` +
    `lookup_in_indirect_block`, `decode_filtered_indirect_block` +
    `lookup_in_filtered_indirect_block` (the originating case from the
    `read_from_indirect_block_rows` analysis).
  No public API change; tests stay green at 468. `ccc_mapping.toml`
  refreshed to point the canonical `H5*_cache_*_deserialize` targets at
  the new `decode_*` halves.
- [x] Mirror libhdf5's file/module split for the `format/` tree
  (2026-04-18). Four files moved into directories whose layout mirrors
  the matching `H5*.c` files in libhdf5:
  - `format/fixed_array/{mod,hdr,dblock}.rs` ← `H5FA{,hdr,dblock,dblkpage}.c`
    (dblkpage folded into dblock — Rust port has no separate page-cache).
  - `format/extensible_array/{mod,hdr,iblock,sblock,dblock}.rs` ←
    `H5EA{,hdr,iblock,sblock,dblock,dblkpage}.c`.
  - `format/fractal_heap/{mod,hdr,iblock,dblock,man,huge,tiny,dtable}.rs` ←
    `H5HF{,hdr,iblock,dblock,man,huge,tiny,dtable}.c`.
  - `format/object_header/{mod,cache,chunk,msg}.rs` ← `H5O{cache,chunk,
    message,pkg}.c`.
  Tests stayed at 471 throughout. The smaller files
  (`btree_v1.rs`, `btree_v2.rs`, `checksum.rs`, `global_heap.rs`,
  `local_heap.rs`, `superblock.rs`, `symbol_table.rs`) didn't need
  splitting — each fits on a single screen and already maps cleanly to
  one C file.

- [x] Mirror libhdf5's file/module split for the `hl/` tree. The previously
  oversized `hl` roots are now small module fronts:
  - `src/hl/mutable_file.rs` → split by subsystem (chunk-btree update,
    extensible-array update, object-header rewrite, dense-storage
    update, allocator/io). Initial slices extracted:
    `src/hl/mutable_file/attr_mutation.rs` now owns compact attribute
    object-header delete/rename mutation helpers and public wrappers;
    `src/hl/mutable_file/resize.rs` now owns dataset resize and dataspace
    object-header message rewrite lookup; `src/hl/mutable_file/write_chunk.rs`
    now owns the public chunk-write front-end plus chunk validation and
    filter encoding dispatch; `src/hl/mutable_file/chunk_btree_v1.rs` now
    owns raw-data v1 chunk B-tree update, traversal, rebuild, and node
    encoding helpers; `src/hl/mutable_file/chunk_fixed_array.rs` now owns
    fixed-array chunk-index element rewrites;
    `src/hl/mutable_file/chunk_btree_v2.rs` now owns v2 B-tree chunk-index
    record replacement, rebuild, and leaf/internal node encoding helpers;
    `src/hl/mutable_file/chunk_extensible_array.rs` now owns
    the full extensible-array chunk-index mutation path: mutation state
    structs, chunk-write planning, header decoding, direct/spillover append
    dispatch, super-block creation/updates, page-init bitmap helpers, sizing
    helpers, index/data-block verification, element address lookup,
    data-block/page encoding, element writes, and all EA checksum rewrites;
    `src/hl/mutable_file/support.rs` now owns shared checksum, reopen,
    allocation, integer-write, and chunk-index geometry helpers used by the
    sibling mutation modules.
  - `src/hl/dataset.rs` (~3300 LOC) → split read paths (chunked /
    contiguous / virtual) into siblings under `src/hl/dataset/`. Initial
    slice extracted: `src/hl/dataset/access.rs` now owns `DatasetAccess`
    and VDS access policy types while `dataset.rs` re-exports the original
    public API; `src/hl/dataset/chunk_read.rs` now owns chunk-read private
    state structs plus the chunked-read dispatcher, single-chunk path,
    chunk geometry helpers, full-coverage/prefill detection, linear chunk
    coordinate helpers, fast-path filter classification, chunk copy
    planning, generic N-D chunk materialization, and direct/decompress-into
    full 1-D chunk fast paths, plus implicit chunk-index read dispatch and
    shared chunk geometry helpers;
    `src/hl/dataset/chunk_btree_v1.rs` now owns the v1 B-tree chunk-index
    reader, recursive node walk, node/key decoding, payload filtering, and v1
    trace hook; `src/hl/dataset/chunk_btree_v2.rs` now owns the v2 B-tree
    chunk-index read path, v2 record decoding, and v2 trace hooks;
    `src/hl/dataset/chunk_copy.rs` now owns chunk copy planning,
    generic N-D chunk materialization, direct/decompress-into full 1-D chunk
    fast paths, and fast-path filter classification;
    `src/hl/dataset/chunk_linear_index.rs` now owns fixed-array and
    extensible-array chunk-index read paths, linear-index full-coverage
    checks, and linear lookup trace hooks; `src/hl/dataset/info.rs`
    now owns `DatasetInfo` /
    external-file-list metadata structs, object-header metadata parsing,
    external-file-list message decoding, attribute wrappers, and dataset
    property/shape accessors; `src/hl/dataset/storage.rs` now owns external
    contiguous raw-data reads and external-file path resolution;
    `src/hl/dataset/support.rs` now owns shared integer-size conversion and
    little-endian byte decoding helpers used by the sibling modules;
    `src/hl/dataset/read.rs` now owns raw storage-class dispatch, fill-value
    materialization, typed/scalar read wrappers, dataspace element counts, and
    ndarray read wrappers;
    `src/hl/dataset/selection.rs` now owns `read_slice`, typed point /
    hyperslab / slice extraction helpers, selection validation, the
    contiguous 1-D slice fast path, and shared row-major index helpers;
    `src/hl/dataset/value_read.rs` now owns fixed/variable-length string
    reads, compound field readers, recursive high-level value decoding,
    vlen heap tracing, field byte-swapping, and local endian/string decode
    helpers;
    `src/hl/dataset/virtual_dataset.rs` now owns the private VDS mapping,
    selection, and materialized-hyperslab data types plus VDS raw
    read/source-copy execution, dynamic output-shape resolution,
    point/hyperslab materialization, VDS selection span helpers, and
    selection coordinate validation; `src/hl/dataset/virtual_decode.rs` now
    owns VDS heap/mapping decode, encoded selection decode,
    decoded-selection helper structs, and VDS decoder trace hooks;
    `src/hl/dataset/virtual_source.rs` now owns VDS source-file, prefix,
    environment-prefix, and `${ORIGIN}` path resolution;
    `src/hl/dataset/tests.rs` now owns the dataset-module unit tests, leaving
    `src/hl/dataset.rs` as the 79-line module root, public re-export surface,
    and minimal `Dataset` shell.
  Bigger refactor than the `format/` tree because libhdf5 doesn't have
  one-to-one analogs; we'd be partitioning by Rust-internal subsystem
  rather than mirroring C exactly.

- [x] Closer-to-1:1 audit (no-fusion + naming-drift, 2026-04-18). Two
  more fusion candidates found and split:
  - `hl/conversion.rs::for_dataset` extracted into per-source-class
    helpers (`kind_for_integer_source`, `kind_for_float_source`,
    `kind_for_passthrough`) mirroring libhdf5's `H5T__conv_*` family.
    The dispatcher itself now matches on `DatatypeClass` and delegates,
    instead of nesting 30+ lines of per-class branching.
  - `hl/dataset.rs::collect_btree_v1_chunks` extracted a pure
    `decode_chunk_btree_node` (returns `ChunkBTreeNode::Leaf|Internal`)
    that mirrors `H5B__cache_deserialize` for the chunk-index node
    type. The recursive driver becomes a thin `match` over the
    decoded node.
  Naming-drift audit: 451 mapped pairs have name divergence from the C
  side, but inspection shows almost all are deliberate — Rust uses
  descriptive names where C uses abbreviations (`compound_fields` vs
  `H5O__dtype_decode_helper`, `read_at` vs `H5*_protect`,
  `datatype_encoded_len` vs `H5O__dtype_size`). Renaming Rust to match
  C's abbreviations would degrade readability for marginal TUI
  benefit; the mapping file already bridges the two.

- [x] Translation-gap audit driven by the now-comprehensive
  `ccc_mapping.toml` (2026-04-18). Three concrete C-side validation
  checks were missing on the Rust side and have been added:
  - `format/fractal_heap.rs::read_managed_object` now validates the
    heap-ID version bits (top 2 bits of byte 0). Mirrors libhdf5's
    `H5HF_get_obj_len` "incorrect heap ID version" check.
  - `format/fractal_heap.rs::read_managed` now bounds the decoded
    object offset against `2^max_heap_size` and the object length
    against `max_managed_obj_size`. Mirrors `H5HF__man_op_real`.
  - `format/messages/link_info.rs::decode` now rejects
    `max_creation_index` values whose encoded int64 representation would be
    negative, matching upstream `H5O__linfo_decode`. Covered by two tests in
    `tests/robustness_test.rs`.
  - `format/fractal_heap/huge.rs` now decodes direct/filtered huge heap
    IDs and huge-object v2 B-tree records through checked layout
    helpers before slicing. Malformed files now get explicit
    `InvalidFormat` diagnostics for overflowing field offsets or record
    size formulas instead of relying on unchecked `usize` arithmetic.
  - Fractal-heap doubling-table geometry now uses checked row-block
    sizes, child indirect-row derivation, span multiplication, and
    managed-object offset advancement. Filtered indirect-block decode
    uses the same checked row-size helper, so extreme row counts no
    longer reach unchecked shifts or block-size multiplication.
  Managed and tiny heap-ID payload decoding now uses checked byte windows
  and rejects offset widths beyond the crate's `u64` representation before
    shifting; dense-link heap-ID extraction also uses a checked record window.
    Filtered huge heap-ID filter masks now decode through a checked `u32`
    helper instead of direct fixed-index arrays.
  - Fractal-heap header decode now validates doubling-table geometry at
    the boundary: nonzero table width, power-of-two start/max direct
    block sizes, max direct block size >= start block size, and
    max-heap-size values that fit the crate's current 64-bit offset
    representation.
  - The shared `HdfReader::skip` helper now rejects distances above
    `i64::MAX` before calling `SeekFrom::Current`, and global-heap
    object walking checks the minimum object-entry end offset before
    comparing it with the declared collection bound.
  - Object-header v1/v2 message loops now compute message-header,
    payload-start, and creation-order bounds through checked helpers.
    Shared-message reference/table offset arithmetic and nested
    continuation chunk-index advancement also reject overflow
    explicitly.
  - Attribute, filter-pipeline, data-layout, dataspace, and link message
    decoders now use checked cursor advancement and checked slice-end
    calculations for file-controlled names, compact payloads, datatype /
    dataspace submessages, filter names, link names, soft-link targets,
    and external-link buffers.
  - Link-info, attribute-info, fill-value, and symbol-table message
    decoders now use checked cursor advancement and checked payload-end
    calculations for address fields and fill-value byte spans instead of
    unchecked `pos += size` / `base + size` arithmetic.
  - V2 B-tree node geometry now range-checks pointer-size sums,
    internal record-slot sizes, cumulative record-count formulas,
    child-count allocation, overfull internal nodes, and zero leaf
    record sizes. Superblock decode now rejects 16/32-byte address or
    length widths explicitly as unsupported by the crate's current
    `u64` address/length representation.
  - V1 group B-tree traversal now rejects undefined child pointers,
    detects recursive traversal cycles, and caps recursion depth before
    descending through symbol-table nodes. Symbol-table group scratch-pad
    sizing now checks address-width multiplication and rejects cached
    group payloads that exceed the fixed 16-byte scratch area.
  - V1 chunk-index B-tree traversal now mirrors the group-tree guards:
    undefined node/child addresses are rejected, recursive traversal
    cycles are detected, and recursion depth is capped before descending
    through internal chunk-index nodes.
  - Implicit fixed-array/extensible-array chunk coordinate calculation
    now validates rank agreement, rejects zero chunks-per-dimension, and
    checks `chunk_index * chunk_dim` coordinate multiplication before
    copying chunk payloads.
  - Variable-length dataset string/value descriptor decoding now uses a
    shared checked parser for descriptor size, address width, heap
    address bytes, and object-index offsets before dereferencing global
    heap objects.
  - Datatype compound and enum decoders now use checked member-name
    padding, cursor advancement, member-offset bounds, inline-array
    dimension offsets, enum value spans, and encoded-length arithmetic
    instead of raw `pos + len` / padded-name calculations.
  Other C-only error strings were investigated and cleared as
  non-actionable: most are runtime identifier checks that don't apply
  to a typed Rust API, validation that already lives in a different
  function, or cross-checks against state we don't have at decode time
  (datatype-vs-fill-value-size).

- [x] Bucket CCC "unsupported subsystem" misses into explicit roadmap
  items instead of leaving them as raw compare noise. Current
  `ccc-rs missing` output is still dominated by large libhdf5 surfaces
  that are intentionally out of scope or not yet translated, especially:
  - VOL / async / plugin / connector infrastructure (`H5VL*`,
    `H5ES*`, `H5PL*`)
  - MPI / parallel I/O / distributed datatype-selection paths
    (`H5_mpi*`, `H5S__mpio*`, parallel `H5D*`) — out of scope; we will
    not use MPI/parallel-HDF5. If CPU parallelism is added later, use
    Rayon and keep it Rust-side rather than chasing libhdf5's MPI stack.
  - Alternative VFDs and cloud/network drivers (`H5FD__hdfs*`,
    `H5FD__ros3*`, direct/core/stdio driver parity, and broad `H5FD_*`
    runtime entry points like alloc/read/write/open/free)
  - Large write-side object-header / message-management families
    (`H5O_msg_*`, shared-message machinery, free-space managers)
  - Remaining unported dataspace selector families (`all`/`none`
    iterators, projection helpers, full selection iterator parity)
  Action for a later pass:
  - classify each family as `won't implement`, `reader-only not needed`,
    or `planned parity work`
  - keep the translation rule explicit: when a function is brought over,
    translate it completely on the first pass where feasible, instead of
    landing partial/stubbed behavior and planning to fill semantics in
    later. Use follow-up passes for auditability/refactoring, not for
    basic missing branches.
  - record global policy: do parallelization last. Finish
    single-threaded faithful translation/audit first, then consider
    Rayon-based acceleration only after behavior is pinned.
  - mirror that classification in `analysis/unsupported_features.md`
  - trim obvious false-positive mappings like parser artifacts (`if`,
    `while`, `FAIL`, `NULL`) from the CCC follow-up workflow so the
    missing report is decision-relevant.
  Completed 2026-04-22 in `analysis/ccc_missing_roadmap.md`, with the
  same classification mirrored in `analysis/unsupported_features.md`.

- [x] `ccc_mapping.toml` was expanded enough to audit the whole tree,
  then tightened back down to a strict 1:1 mapping set.
  Current policy is no longer "every Rust function gets some closest C
  counterpart"; current policy is "only keep explicit mappings when the
  Rust function is a defensible owner of that C body under a faithful
  translation audit." Categories covered during that audit, with the C
  target families involved:
  - High-level public API (`hl/file.rs`, `hl/group.rs`, `hl/dataset.rs`,
    `hl/attribute.rs`, `hl/datatype.rs`, `hl/dataspace.rs`,
    `hl/writable_file.rs`, `hl/mutable_file.rs`, `hl/dataset_builder.rs`,
    `hl/types.rs`, `hl/selection.rs`, `hl/conversion.rs`,

## Remaining Original-Feature Parity

Unchecked items in this section are features or feature families present in the
original libhdf5 codebase that are not yet fully mirrored here. This is the
forward-looking backlog for "add all features of the original" within this
crate's intended scope.

### Planned Parity Work
- [x] Add bounded real functionality for the easiest unsupported-subsystem
  surfaces exposed by the explicit low-level stubs:
  - Metadata/no-op query APIs around existing state: cache hit-rate counters,
    page-buffer stats, file intent/access flags, and basic SWMR/logging flags.
    Implemented as file-level query structs over existing counters and flags.
  - Multi/family/splitter/log VFD config serialization: property/config
    encode/decode and validation without real I/O. Implemented deterministic
    round-trip codecs and validation helpers while keeping actual driver I/O
    explicitly unsupported. Onion, subfiling IOC, and subfiling superblock
    payloads now use the same fallible structured decode pattern and reject
    truncated, trailing, or zero-valued invalid config images instead of
    collapsing malformed payloads to `None`.
  - External file cache bookkeeping: track opened external files in a small
    map, max count, release/close behavior. Implemented bounded cache state
    without arbitrary external-link traversal.
  - VOL registry/introspection improvements: connector lookup, capability
    flags, optional-op registry, connector value/name APIs. Implemented
    registry introspection and optional-operation listing without plugin
    dispatch.
  - HDFS/S3/ROS3 explicit unsupported config parsing: parse/store config
    values and fail only on open/read/write. Implemented HDFS and ROS3 FAPL
    config storage plus existing S3 URL parsing; network/open/read/write
    operations remain explicitly unsupported.
- [x] Add fuller dataspace selector parity beyond the current decode and
  selected materialization paths:
  current support includes `all`, `none`, points, stepped slices, regular
  block hyperslabs, combine operators via bounded point materialization, and
  count/bounds/regularity/contiguity/linear-span helpers, iterator-style point
  traversal, and dimension projection helpers. Bounds helpers now avoid
  coordinate overflow on extreme slices/hyperslabs; N-D slice extraction now
  rejects coordinate/index overflow instead of relying on unchecked `u64`
  arithmetic; hyperslab `selected_count` now reports overflow instead of
  saturating per-dimension count/block products; dataset selection/VDS byte
  readers now use checked range arithmetic for malformed offsets.
- [x] Add fuller soft-link traversal parity:
  public link iteration and tracked creation-order link iteration are
  implemented; file-level and group-relative opens now share soft-link
  traversal, relative target normalization handles `.`/`..`, and repeated
  soft-link resolution paths report a direct cycle diagnostic before the
  traversal-limit fallback.
- [x] Finish virtual-dataset dataset-access property-list parity:
  `H5Pset_virtual_view` / `H5Pset_virtual_prefix` equivalents are exposed via
  `DatasetAccess`; fixed-shape missing-source fill behavior is supported, and
  unlimited-dimension missing-source fill behavior now uses declared virtual
  selection extents where they are encoded so `FirstMissing` and
  `LastAvailable` sizing remain distinct. VDS source string decoding now
  rejects out-of-range starts and unterminated strings explicitly. VDS output
  extent and hyperslab materialization paths now use checked coordinate/span
  arithmetic for extreme selections, including irregular hyperslab block
  coordinate materialization. VDS source-to-destination raw materialization
  now uses the shared integer/float datatype conversion path instead of
  requiring matching source and destination element sizes; covered by a real
  h5py-generated i32-source to f64-destination VDS fixture.
- [ ] Add broader writer-side chunk-index parity if writer scope expands:
  creation/growth paths for fixed-array, extensible-array, and deeper v2
  B-tree chunk indexes beyond the currently supported subset. Incremental
  fixed-array update robustness added: helper chunk-grid indexing now rejects
  zero chunk dimensions, chunk-count overflow, and usize conversion overflow
  explicitly; aligned append helpers now reject zero alignment and checked
  padding offset overflow. Incremental v2 B-tree update robustness added:
  chunk-record sizing, leaf/internal capacity arithmetic, record distribution,
  and scaled-coordinate decoding now fail explicitly on malformed overflow
  cases. Incremental extensible-array robustness added: read/write
  super-block geometry planning now rejects start-index span overflow instead
  of relying on unchecked multiplication; read-side extensible-array
  data/super-block walks now range-check page sizes, page offsets, skip spans,
  page-init slices, fill counts, and spillover address-table indices.
  Writer-side extensible-array data/super-block allocation and update helpers
  now range-check block sizes, page-init byte spans, address offsets, and
  checksum spans before allocating, slicing, or seeking. Additional
  writer-side extensible-array element-location, data-block page write, and
  data/page checksum helpers now range-check element offsets and checksum
  addresses before file I/O. Spillover append and metadata-rewrite helpers now
  range-check super-block spans, block offsets, index-block address-table
  offsets, header counters, and header checksum field offsets before patching
  mutable metadata. Incremental v1 chunk B-tree writer robustness added: leaf
  layout, entry-position, entry-count, and node-size helpers now use checked
  arithmetic before reading, appending, rebuilding, or encoding B-tree nodes.
  v1 chunk B-tree collection now rejects oversized node entry counts,
  unsupported depth, child-key coordinate-count overflow, and overlarge
  collected two-level trees before recursing or extending buffers. Fixed-array
  locate/read paths now range-check data-block prefix sizes, page sizes, page
  offsets, element offsets, and page addresses before seeking. Shared mutable
  object-header checksum rewrites and fixed-array/extensible-array/v2-B-tree
  checksum verifiers now check checksum end offsets before seeking.
- [ ] Add broader filter parity where practical:
  pure-Rust SZip support if it becomes available, plus fuller NBit and
  ScaleOffset parameter-space parity. Incremental ScaleOffset parity added:
  integer sign, minimum-bit bounds, and the 16-byte arithmetic limit are now
  validated before element decoding, including empty chunks; unsupported
  datatype classes, unsupported floating-point sizes, and unsupported float
  scale types are rejected before chunk-header parsing; header and output
  offsets now use checked arithmetic for malformed chunks. Incremental NBit
  parity added: zero datatype sizes, invalid atomic byte order, zero nested
  array base sizes, and non-divisible array/base sizes now fail explicitly;
  datatype precision/offset and output-copy helpers now use checked arithmetic
  for malformed parameter streams.
  Incremental LZF parity added: literal and back-reference runs now fail
  immediately when they would exceed the expected output size, and malformed
  literal/back-reference input offsets are range-checked explicitly.
  Incremental Shuffle parity added: the decoder now honors the filter's
  encoded element byte size when present and rejects a zero-sized encoded
  element; shuffle/unshuffle index arithmetic now fails explicitly on
  overflow. `MutableFile::write_chunk` can now encode Fletcher32 checksums
  when replacing chunks in Fletcher32-filtered datasets, matching the
  existing read-side verification path. `DatasetBuilder::fletcher32()` now
  writes Fletcher32-filtered chunked datasets using the same forward filter
  ordering as chunk mutation.
- [ ] Add broader datatype-conversion parity if user-facing conversion scope
  expands beyond the current exact-size / limited-recursive model:
  more of the `H5T__conv_*` engine families, packing helpers, and
  conversion-path selection behavior. Incremental parity added: typed reads
  now expose `i128` / `u128`, handle 128-bit integer clamp/sign-extension
  without shift overflow, and route same-size signed/unsigned integer reads
  through conversion instead of raw reinterpretation. Writer dtype inference
  now emits 16-byte fixed-point datatypes for `i128` / `u128`, including
  direct datasets, scalar attributes, and compound fields; recursive
  `H5Value` field decoding also handles signed 128-bit integer payloads
  safely. Datatype array dimension decoding now advances through dimension
  entries with checked cursor arithmetic, and string/VLEN readers reject
  non-record-aligned raw buffers instead of silently dropping trailing bytes;
  attribute VLEN string reads now reject address widths beyond 64-bit support
  before descriptor shift arithmetic. High-level numeric conversion now uses
  checked output-size and per-record output-window helpers instead of raw
  `idx * dst_size..(idx + 1) * dst_size` slice arithmetic, and float decoding
  no longer depends on unreachable `unwrap()` conversions. VDS raw reads now
  convert source dataset numeric bytes into the virtual destination datatype
  before selection placement. VDS regular
  hyperslabs with unlimited counts/blocks now reject starts past the dataspace
  extent instead of hiding them behind saturating subtraction. Mutable v2
  B-tree chunk-index header checksum rewrites now compute relative field
  offsets and header mutation windows through checked helpers instead of
  unchecked conversions and slice ranges. Writable fixed-string attribute
  payload construction now rejects capacity overflow, and extensible-array
  fill/unread-element helpers now reject overfilled state or impossible read
  counts instead of saturating them to zero. Chunked writer splitting now uses
  checked chunk-count, total-chunk, compressed-size, chunk-extraction, and
  VLEN descriptor payload arithmetic instead of unchecked products, casts, and
  offset ranges. Dataset-builder fixed-string dataset and attribute paths now
  use checked shape/product and payload-capacity arithmetic, and `Dataspace::size`
  no longer wraps on huge dimension products. Superblock v2 checksum spans and
  extensible-array index-block geometry now use checked arithmetic instead of
  raw `12 + 4 * sizeof_addr` / doubled pointer-count expressions. Selection
  hyperslab dimensions now expose a checked output-count helper, while the
  legacy infallible count saturates explicitly; contiguous-selection checking
  no longer uses an unreachable `expect()`. Writer chunk B-tree node sizing,
  minimal fractal-heap header sizing, managed-heap block growth, and the
  initial superblock placeholder now use checked arithmetic. Attribute VLEN
  string descriptors now decode through a checked helper, VDS point-selection
  span calculation avoids unreachable unwraps, and Fletcher32 word processing
  no longer uses manual `pos + 1` indexing. `Superblock::checked_size()` now
  exposes fallible checked size computation, while the legacy infallible
  `size()` helper saturates explicitly on overflow. Attribute-info,
  data-layout, and filter-pipeline message integer helpers now read from
  checked slices and checked variable-width spans instead of direct `pos + n`
  indexing. Dense writer B-tree record hash extraction and external-link object
  payload slicing now use checked windows instead of direct offset arithmetic.
  Datatype array and legacy compound inline-array dimension reads now use a
  checked little-endian helper, and compound member offset-width calculation
  now rejects underflow explicitly instead of using saturating subtraction.
  NBit nested array/compound filter decoding now computes recursive output
  offsets with checked multiplication and addition instead of unchecked
  `base + idx * size` arithmetic; top-level NBit atomic/array/compound element
  offsets now use the same checked helper, and NBit byte-copy writes through a
  checked output window. Chunk copy and virtual-dataset source-copy
  byte windows now use checked slice helpers at the final copy boundary instead
  of repeating unchecked `offset..offset + len` ranges. External raw-storage
  reads now use the same checked output-window pattern before `read_exact`.
  ScaleOffset and Fletcher32 filter headers now read checksum/minimum-bit and
  minimum-value windows through checked helpers, and Fletcher32 batching no
  longer constructs manual `pos..pos + byte_count` slices.
  Fill-value and attribute message decoders now read little-endian size fields
  and value/name/datatype/dataspace payload windows through checked helpers
  instead of direct fixed-index arrays.
  Symbol-table, link-info, dataspace, and link message variable-width integer
  readers now decode from checked windows instead of indexing `pos + i` or
  `offset + i` in their decode loops.
  Attribute-info, data-layout, filter-pipeline, Fletcher32, and ScaleOffset
  fixed-width integer helpers now convert checked windows with `try_into`
  instead of indexing helper-local byte arrays.
  Datatype, fill-value, attribute, attribute VLEN, dataset VLEN, compact
  attribute mutation, filtered huge-heap, and dense-writer record decoders now
  use the same checked-window plus `try_into` pattern.
  Filter-pipeline v1/v2 filter-name decoding now slices both padded name
  fields and null-terminated text through checked windows.
  Attribute and dataset VLEN descriptor readers now decode sequence length,
  heap address, and heap index fields through checked windows; compact
  attribute mutation now reads encoded name sizes through the same checked
  helper pattern.
  Mutable extensible-array header checksum recomputation now patches count
  fields through a checked helper instead of direct `offset..offset + ss`
  slices.
  Datatype message header/property validation now reads class bits, size,
  fixed-point precision/offset, and floating-point precision/offset through
  checked helpers, including the tracehash instrumentation path.
  The enum writer now rejects non-integer and
  wider-than-8-byte base datatypes explicitly instead of silently truncating
  member values to the existing `u64` representation. Malformed zero-sized
  float and integer-to-float conversion sources are now rejected explicitly
  instead of reaching modulo-by-zero conversion paths. Compound field value
  extraction now uses checked field-offset and array-payload arithmetic for
  malformed nested datatypes.
- [ ] Add broader attribute mutation/iteration parity if the public writer/API
  surface expands:
  current public APIs expose `attrs()`, `attr_names()`, and `attr(name)` for
  files, groups, and datasets across compact and dense storage, plus
  `attrs_by_creation_order()` and `attr_exists(name)`; writer paths now reject
  duplicate attribute creates by name. `MutableFile` can delete compact
  attributes on root, group, and dataset objects by marking the compact
  object-header message as NIL and recomputing the header checksum, and can
  rename compact attributes in-place when the new UTF-8 name fits in the
  existing encoded name field and does not collide with another attribute.
  `MutableFile` can also delete and same-size rename dense attributes for the
  crate's own writer-created layout: an unfiltered root-direct fractal heap
  plus a depth-0 v2 name B-tree. The mutation rewrites the B-tree leaf/header
  record counts and checksums, updates dense-name hashes on rename, and
  recomputes the direct-block checksum after heap payload edits. Remaining
  work is dense mutation for indirect/filtered heaps, non-leaf name indexes,
  or creation-order indexes, growing compact/dense rename that requires metadata
  repacking, and broader mutation APIs.

### Explicitly Out Of Scope
- [x] Do not chase MPI / parallel-HDF5 / distributed selection paths
  (`H5_mpi*`, `H5S__mpio*`, parallel `H5D*`). If parallelism is added
  later, do it last with Rayon in Rust code.
- [x] Do not chase VOL / async / plugin / connector infrastructure
  (`H5VL*`, `H5ES*`, `H5PL*`) in this crate.
- [x] Do not chase alternative VFD / cloud / network driver parity
  (`H5FD__hdfs*`, `H5FD__ros3*`, `H5FD__direct*`, broad core/stdio-only
  driver parity, and the broad `H5FD_*` runtime API surface).
- [x] Do not chase Map API parity (`H5M*`) in this crate.
- [x] Do not chase broad package-init / free-list / thread-runtime
  machinery (`H5FL*`, broad `H5TS*`) unless this crate grows a direct need
  for those surfaces.
    `hl/plist/*`) → `H5{F,G,D,A,T,S,L,P,R,I,Z}*` API + `H5*__cache_*`.
  - Format-layer decoders / encoders / lookups (`format/*`,
    `format/messages/*`) → `H5O__*_decode/encode/size`,
    `H5{B,B2,EA,FA,HF,HG,HL,F}__cache_*`, `H5*_iterate`, `H5*__man_op_real`.
  - Engine layer (`engine/{writer,handle,allocator}.rs`) → `H5{I,F,MF}*`.
  - I/O primitives (`io/reader.rs`/`io/writer.rs`) →
    `H5F__{en,de}code_uint{8,16,32,64}` / `H5F_addr_{en,de}code` /
    `H5F_{EN,DE}CODE_LENGTH` / `H5FD_{read,write,seek}`.
  - Filter pipeline (`filters/*`) → `H5Z_pipeline` + `H5Z__filter_{deflate,
    shuffle,fletcher32,scaleoffset,nbit,szip,blosc,lzf}` and helpers.
  - Pure utility helpers (`ensure_available`, `read_le_u64`, `read_u8`,
    `bit_is_set`, `log2_*`, `bytes_needed`, `usize_from_u64`) →
    `UINT*DECODE` macro family / `H5_IS_BUFFER_OVERFLOW` /
    `H5VM_{bit_get,log2_*}` / `H5_ASSIGN_OVERFLOW`.
  - Trait impls (`Display::fmt`, `Error::source`, `From::from`,
    `Default::default`) → `H5E*` / `H5I_init_interface` / `H5F__super_init`.
  - Constructor `new` methods → `H5*_create` / `H5*_init`.
  - `inner` / `inner_mut` accessors → `H5F_get_intent`.
  - Test functions → mapped to the C function under test
    (e.g. `test_lzf_*` → `H5Z__filter_lzf`).
  - Tracehash probe `#[cfg]` companion bodies → mapped to the C
    function whose behavior the probe captures.

- [x] Write-side fusion audit (no-fusion rule, 2026-04-18). Audited
  `hl/mutable_file.rs` and `engine/writer.rs`. Of the four originally
  flagged candidates, two were genuine fusions and have been split
  (encode-half extracted into a pure `encode_*` returning `Vec<u8>`,
  with the wrapper composing alloc + encode + write):
  - `encode_extensible_array_data_block_prefix` ↔ `H5EA__cache_dblock_serialize`
  - `encode_extensible_array_data_block_page` ↔ `H5EA__cache_dblk_page_serialize`
  - `encode_chunk_btree_node` ↔ `H5B__cache_serialize` (extracted from
    `write_chunk_btree_node`).
  Two were false positives: `rewrite_extensible_array_super_block` is
  read-modify-write (patches existing on-disk bytes, no encode-from-scratch
  step to extract); `rewrite_leaf_chunk_btree` is an orchestrator over
  already-separate primitives (`write_btree_entry`, `write_btree_final_key`,
  `rebuild_chunk_btree_from_entries`). The split is allocation-neutral —
  the original code was already building the same `Vec<u8>` internally.

- [x] Keep `ccc-rs` explicit mappings intentionally 1:1. When two Rust
  halves correspond to one fused-on-the-C-side function (for example
  `decode_indirect_block` and `decode_filtered_indirect_block` both
  relating to `H5HF__cache_iblock_deserialize`, or `decode_header` and
  `walk_objects` both relating to `H5HG__cache_heap_deserialize`), only
  one side gets the explicit mapping and the other falls through to
  fingerprint matching. This is expected tool behavior, not a bug to
  "fix". Residual compare noise from such fused C functions is accepted.

### blosc dependency

- [ ] Publish `blosc2-pure-rs` 0.3.0 to crates.io. The dependency in
  `Cargo.toml` now requires `^0.3` (was `^0.2`); a `[patch.crates-io]`
  entry overrides resolution to the in-tree sibling checkout at
  `../blosc2-rs` so local development continues to work, but
  `cargo publish` will fail until 0.3.0 is on crates.io. Drop the
  `[patch.crates-io]` entry once published.
- [x] Instrument Rust-side probes for datatype message decode (`DatatypeMessage::decode`).
- [x] Instrument Rust-side probes for object header message decode (`ObjectHeader::read_at`).
- [x] Instrument Rust-side probes for data layout and filter pipeline decode.
- [x] Add a Rust corpus runner that emits `/tmp/rust.tsv`.
- [x] Add a documented comparator command using `tracehash-compare`.
- [x] Document C-side probe targets for datatype message decode (`H5T*` decode path).
- [x] Document C-side probe targets for object header message decode (`H5O*` decode path).
- [x] Document C-side probe targets for filter pipeline decode and application (`H5Z*`).
- [x] Document C-side probe targets for chunk index resolution: v1 B-tree, v2 B-tree, fixed array, extensible array, and single chunk.
- [x] Document C-side probe targets for fractal heap object lookup and dense link/attribute traversal.
- [x] Defer representative divergence reports until a patched HDF5 C build emits `/tmp/c.tsv`.

## Concerns from ccc-rs cross-language scan (2026-04-17)

Scanned with `ccc-rs compare` and `ccc-rs constants-diff` against
`hdf5/src` using the project's `ccc_mapping.toml`. After clearing false positives
(named constants, optimization unrolling, tracehash gates, scratch-pad
layout literals, and an analyzer hex-parse bug since fixed upstream),
three items remain:

- [x] `format/btree_v2.rs::BTreeV2Header::read_at` now rejects
  `split_pct > 100`, `merge_pct > 100`, and `merge_pct >= split_pct` with
  `InvalidFormat` (matching upstream `H5B2__hdr_init`). Covered by four
  tests in `tests/robustness_test.rs`.
- [x] `hl/file.rs::resolve_path` enforces a 1024-byte per-component cap
  (`MAX_PATH_COMPONENT_LEN`, matching `H5G_TRAVERSE_PATH_MAX`). Covered by
  two tests in `tests/group_test.rs`.
- [x] `engine/writer.rs::build_v2_object_header` dead `8`/`_` match arms
  removed; `chunk0_bytes` is now `match`ed exhaustively over the only
  values it can take (1, 2, 4) with a single `unreachable!()` fallback.
- [x] `format/messages/datatype.rs::DatatypeMessage::decode` now validates
  FixedPoint and BitField `bit_offset` and `precision` against the byte
  size: rejects `precision == 0`, `bit_offset > size*8`, and
  `bit_offset + precision > size*8`, matching upstream
  `H5O__dtype_decode_helper`. Six tests in `tests/robustness_test.rs`
  cover the rejection paths and the canonical-width acceptance paths.
  This validation also caught six pre-existing fixtures in
  `tests/robustness_test.rs` whose `precision` bytes were encoded
  big-endian instead of little-endian per spec; those have been
  corrected.
- [x] `format/messages/datatype.rs::DatatypeMessage::decode` now validates
  FloatingPoint properties: rejects `precision == 0`, `exp_size == 0`,
  `mant_size == 0`, sign bit position outside precision, and
  exp/mantissa location+size overflowing precision. Six tests cover the
  rejection paths plus a canonical IEEE-754 binary32 acceptance test.
- [x] `format/messages/datatype.rs::DatatypeMessage::decode` now rejects
  the reserved FloatingPoint mantissa-normalization code `3`, matching
  upstream `H5O__dtype_decode_helper`.
- [x] `format/messages/datatype.rs::DatatypeMessage::decode` now validates
  array datatype byte size against `base.size * product(dims)` and rejects
  truncated array metadata at decode time. This matches
  `H5O__dtype_decode_helper`'s array element-count and size checks.
- [x] `format/messages/datatype.rs` now treats opaque datatype tag length
  as the padded, 8-byte-aligned length encoded in the class bit field,
  matching `H5O__dtype_decode_helper`; this fixes embedded opaque datatypes
  so following compound members are advanced from the right offset.
- [x] `format/messages/filter_pipeline.rs` v1 decoder now rejects
  filter `name_length` values that are not a multiple of eight, matching
  upstream `H5O__pline_decode`. Covered by one focused test in
  `tests/robustness_test.rs`.
- [x] `format/object_header.rs::read_v1` now cross-checks the declared
  `num_messages` field against the actual decoded count: a v1 object
  header that decodes more non-NIL/non-continuation messages than its
  prefix declares is rejected. Per spec the stored count is an upper
  bound, so the check is `decoded ≤ declared`.
- [x] `format/messages/dataspace.rs::decode_v2` now rejects Scalar and
  Null dataspaces with a non-zero rank, matching upstream
  `H5O__sdspace_decode`'s "invalid rank for scalar or NULL dataspace"
  check. Three tests cover the rejection paths and the canonical
  scalar acceptance path.
- [x] `format/messages/attribute.rs` (v1/v2/v3) now rejects messages
  with `name_size == 0`, matching upstream `H5O__attr_decode`. Covered
  by one test.
- [x] `format/messages/attribute.rs` now rejects unknown v2/v3 attribute
  flags, `name_size <= 1`, and embedded NULs before the stored name length,
  matching `H5O__attr_decode`'s flag and corrupted-name checks.
- [x] `format/messages/data_layout.rs` (v1/v2/v3/v4) now rejects chunk
  layouts with any chunk dimension equal to zero (matches
  `H5O__layout_decode`'s "chunk dimension must be positive"). Covered
  by one focused v3 test.
- [x] `format/messages/data_layout.rs` now rejects virtual layout messages
  before version 4, matching `H5O__layout_decode`.
- [x] `format/messages/data_layout.rs` now rejects v4 B-tree2 chunk-index
  layout messages with a zero node size, matching the positive creation
  parameter requirement used alongside the split/merge percentage checks.
- [x] `format/messages/dataspace.rs` now rejects current dimensions that
  exceed stored maximum dimensions, matching `H5O__sdspace_decode`.
- [x] `format/messages/fill_value.rs` now rejects unknown v3 fill-value
  flags, matching `H5O__fill_new_decode`.
- [x] `format/messages/fill_value.rs` now decodes nonzero v2 fill-defined
  states with the following size-prefixed payload, matching `H5O__fill_decode`
  and older files that encode explicit fill bytes with state `1`.
- [x] `engine/object_api.rs::H5O__mtime_new_decode` now decodes the new
  object modification-time message as version/reserved/u32 seconds instead of
  aliasing the old raw-u64 helper, matching `H5O__mtime_new_decode`.
- [x] v2 object headers now validate and apply `MSG_OBJ_REF_COUNT` payloads:
  the decoder requires the four-byte refcount field, tolerates trailing bytes,
  and reports the decoded value instead of always defaulting v2 refcount to 1.
- [x] `engine/object_api.rs` now exposes a faithful `H5O__refcount_decode`
  helper for the four-byte object refcount message image, matching the
  existing encode/size helpers and the v2 object-header path.
- [x] `engine/object_api.rs::{H5O__ginfo_decode,H5O__btreek_decode}` now
  decode structured group-info and B-tree-K message payloads instead of raw
  byte passthroughs, with version/flag/truncation checks and trailing-byte
  tolerance after known fields.
- [x] Real object-header message validation now applies those same group-info
  and B-tree-K checks during v1/v2 header reads, so malformed fixed-layout
  payloads fail before high-level code sees the header.
- [x] `engine/object_api.rs::H5O__fsinfo_decode` now preserves the optional
  file-space page-size field after the strategy/persist/threshold prefix and
  sizes/encodes the message accordingly.
- [x] `engine/object_api.rs::H5O__name_decode` now stops at the first NUL
  byte and `H5O__name_encode` writes the terminating NUL, matching the
  C-string object-name message convention.
- [x] `engine/object_api.rs::{H5O__fill_new_decode,H5O__pline_decode}` now
  return the existing parsed fill-value and filter-pipeline message structs
  instead of raw byte passthroughs, with size/reset/copy helpers operating on
  the decoded message state.
- [x] Real object-header message validation now dispatches fill-value,
  old-fill-value, and filter-pipeline payloads through their faithful message
  decoders during header reads, so malformed payloads fail at the same boundary
  as group-info, B-tree-K, shared-message, and refcount messages.
- [x] `engine/object_api.rs::H5O__dtype_decode_helper` now returns the
  parsed datatype message and its encode/size/copy/share/debug helpers operate
  on that decoded state while preserving the raw class-property bytes.
- [x] Real object-header message validation now also dispatches unshared
  datatype and attribute payloads through their faithful decoders during header
  reads, while still treating shared messages as shared-message references.
- [x] Object-header validation now also dispatches unshared dataspace, data
  layout, link, link-info, attribute-info, and symbol-table payloads through
  their real decoders while the reader still has the superblock address/size
  widths needed to interpret those messages faithfully.
- [x] `engine/object_api.rs::{H5O__shmesg_decode,H5O_SHARED_DECODE}` now
  expose parsed shared-message table and shared-reference payloads instead of
  byte passthroughs, with validation matching the object-header shared-message
  checks and trailing-byte tolerance after the decoded fields.
- [x] `engine/object_api.rs::H5O__efl_decode` now exposes a parsed external
  file list message using address/size-width-aware decoding instead of raw
  byte passthroughs, and object-header validation now applies the same EFL
  checks while the reader still has the required superblock widths.
- [x] `engine/object_api.rs::H5O__attr_decode` now exposes the parsed
  attribute message instead of a raw byte clone while retaining the original
  raw byte length for size/copy/debug helpers, because the parsed attribute
  representation intentionally does not preserve all padding bytes needed for
  exact re-encoding.
- [x] `engine/object_api.rs::{H5O__link_decode,H5O__linfo_decode,H5O__ainfo_decode}`
  now expose parsed link, link-info, and attribute-info object-message
  wrappers instead of raw byte helpers, preserving raw payload length for size
  callbacks while copy/delete/debug operate on the decoded state.
- [x] `engine/object_api.rs::H5O__fill_old_decode` now routes old fill-value
  messages through the existing `FillValueMessage::decode_old` parser, and
  old-fill encode/size helpers now operate on the parsed fill-value state.
- [x] `engine/vfd.rs::{H5FD__onion_revision_record_decode,H5FD__onion_sb_decode,H5FD__ioc_sb_decode,H5FD__subfiling_sb_decode}`
  now return structured `Result` errors for invalid/truncated superblock
  payloads, and onion revision history ingestion rejects partial trailing
  records instead of silently dropping them.
- [x] `engine/vfd.rs` mirror transmit decoders now return structured errors
  for truncated reply, set-EOA, and write payloads instead of mapping bad
  messages to sentinel values or `None`.
- [x] `engine/dataset_api.rs` chunk-index count-image decoders now parse and
  preserve the declared entry count separately from materialized records and
  reject malformed count images. They still do not fabricate chunk records
  because the local helper image only encodes the count.
- [x] `engine/reference.rs` reference-token compatibility decoders now return
  `Result` and validate the length-prefixed object/region token payloads
  instead of reporting truncation as `None`.
- [x] `engine/shared_message.rs::H5SM__cache_list_deserialize` now parses the
  shared-message cache-list image back into `SharedMessageStore` entries
  instead of returning a byte clone, using the existing
  `SharedMessage::encode` entry layout as its inverse and rejecting truncated
  cache-list records.
- [x] `format/local_heap.rs::LocalHeap::fl_deserialize` now returns a
  `Result` and rejects partial trailing free-list records instead of silently
  ignoring them, matching the fixed 16-byte free-list entry layout.
- [x] `engine/property.rs` HDFS/ROS3 FAPL config decoders now return
  structured `Result` errors for truncated strings, invalid presence flags,
  invalid UTF-8, and trailing config bytes instead of collapsing malformed
  stored property bytes to `None`.
- [x] `engine/group_api.rs` group cache-node and dense-link record decoders
  now reject invalid UTF-8 and malformed creation-order records instead of
  lossy-decoding names or accepting trailing bytes in fixed-size records.
- [x] `engine/vfd.rs` mirror transmit lock/open decoders now return
  structured errors for unexpected lock payload bytes and invalid UTF-8 open
  paths, matching the fallible reply/set-EOA/write transmit decoders.
- [x] Metadata string helpers for object names, local-heap strings, link
  message text fields, and filter-pipeline names now reject invalid UTF-8
  instead of lossy-decoding replacement characters. High-level raw value
  presentation still keeps lossy string display where it is intentionally
  user-facing rather than metadata validation.
- [x] Dataset/attribute fallible string reads now reject invalid UTF-8 for
  fixed-length and variable-length string payloads instead of silently
  substituting replacement characters. The legacy non-fallible attribute
  scalar string convenience method keeps its compatibility fallback.
- [x] File-level helper validation now rejects invalid cached superblock
  signatures and invalid configured file-locking environment values instead
  of treating malformed inputs as valid/absent state.
- [x] Dataset scatter/select wrappers and implicit chunk-index iteration now
  return `Result` and propagate span/geometry overflow errors instead of
  collapsing checked failures to empty vectors.
- [x] Attribute message names, object comments, enum member names, and opaque
  datatype tags now validate UTF-8 through their existing `Result` decode
  paths instead of lossy-decoding metadata text.
- [x] `engine/datatype_api.rs::H5T__vlen_mem_str_write` now returns
  `Result<()>` and rejects invalid UTF-8 instead of lossy-decoding into a
  Rust `String`.
- [x] `format/messages/filter_pipeline.rs` now rejects non-NUL-terminated
  stored filter names, matching `H5O__pline_decode`.
- [x] `format/object_header/cache.rs` now rejects reserved v2 object-header
  flag bits before decoding optional fields, matching the defined
  `H5O_HDR_*` flag mask. The mutable-file object-header scanners now share
  the same mask and use checked chunk/message range arithmetic.
- [x] Compact/dense in-place attribute rename now rejects both growing and
  shrinking the encoded name field. Shrinking would leave an embedded NUL
  in the fixed attribute name field, which the faithful `H5O__attr_decode`
  corrupted-name check now rejects.
- [x] `format/symbol_table.rs::SymbolTableNode::read_entry` now rejects
  unknown symbol-table entry cache types instead of silently skipping the
  scratch pad, matching the fixed `H5G_ent_decode` cache-type union.
- [x] `format/superblock.rs` now applies the same cache-type validation to
  the v0/v1 root symbol-table entry embedded in the superblock.
- [x] `format/fractal_heap/iblock.rs` now validates indirect-block version
  bytes and verifies checksums for filtered indirect blocks, matching the
  same metadata checks already applied to unfiltered indirect blocks.
- [x] `format/messages/{link_info,attribute_info}.rs` no longer rejects a
  creation-order index flag when creation-order tracking is absent, matching
  `H5O__linfo_decode` / `H5O__ainfo_decode` which decode the optional B-tree
  address without enforcing that implication.
- [x] `format/messages/{link_info,attribute_info}.rs` also no longer rejects
  trailing payload bytes after the decoded fields; upstream decodes the fields
  it knows about and returns without an end-of-message equality check. This
  fixes the big-endian external-link fixture.
- [x] v1 reserved-byte handling in dataspace, attribute, filter-pipeline, and
  v1/v2 layout message decoders now follows the upstream decoders: check the
  buffer is long enough, then skip the reserved bytes rather than requiring
  them to be zero.
- [x] Serialized VDS selection headers now skip reserved bytes for all, point,
  and hyperslab selections, matching `H5S__all_deserialize`,
  `H5S__point_deserialize`, and `H5S__hyper_deserialize`.
- [x] Global heap header/object reserved-byte handling now mirrors
  `H5HG__hdr_deserialize` / `H5HG__cache_heap_deserialize`: check the fields
  are present, then skip them. Index-0 free objects use their stored size as
  the movement amount even when it is smaller than a normal object header,
  while zero-size free entries terminate the walk to guarantee parser progress
  on patched VDS global-heap fixtures.
- [x] Link, symbol-table-message, symbol-table-node, local-heap, and
  external-file-list metadata parsing now follows the same upstream pattern:
  require the decoded fields to be present, skip reserved fields without
  zero-validation where libhdf5 does, and ignore trailing payload bytes where
  the corresponding C decoder has no consumed-all-buffer check. External file
  list entries also use `sizeof_size` for name offset, file offset, and size,
  matching `H5F_DECODE_LENGTH`.
- [x] v1 object-header prefix/message reserved bytes are now skipped rather
  than zero-validated, matching `H5O__prefix_deserialize` and
  `H5O__chunk_deserialize`. v2 object-header creation-order indexing no
  longer requires tracking during prefix decode, and object-header message
  flags now use libhdf5's full defined flag mask plus the same bad-combination
  checks.
- [x] Shared-message table/reference validation now mirrors
  `H5O__shmesg_decode` and `H5O__shared_decode`: validate the required fields
  and addresses, but tolerate trailing payload bytes after the decoded shared
  message reference.
- [x] Filter-pipeline v1 odd client-data padding is now skipped instead of
  zero-validated, matching `H5O__pline_decode`.
- [x] Array datatype decoding now uses libhdf5's version boundary for the
  compact array layout: version 2 keeps the reserved bytes plus dimension
  permutation block, while version 3 and later drop both. This matches
  `H5O__dtype_decode_helper`.
- [x] String datatype decode now preserves padding and character-set bitfields
  without range-checking them, matching `H5O__dtype_decode_helper`'s decode
  path.
- [x] `engine/object_api.rs::H5O__ginfo_encode` is now fallible and rejects
  unsupported group-info versions plus half-present paired fields instead of
  silently encoding missing compact/dense limits or estimates as zero.
- [x] `engine/object_api.rs::H5O__prefix_deserialize` now validates cached
  object-header prefix images before returning the raw bytes: v1 images must
  have a complete declared chunk, and v2 images must have valid version/flags,
  phase-change ordering, declared chunk length, and checksum.
- [x] `engine/object_api.rs::{H5O__chunk_deserialize,H5O__cache_chk_deserialize,H5O__cache_chk_serialize}`
  now validate v2 `OCHK` continuation-chunk checksums before returning raw
  bytes, while leaving v1 continuation chunks as raw message streams because
  they require object-header context to parse faithfully.
- [x] `engine/object_api.rs::H5O__layout_decode` now rejects impossible data
  layout message versions even though it still preserves the raw payload; full
  layout parsing remains in `format/messages/data_layout.rs` where
  address/size widths are available.
- [x] `engine/free_space.rs::FreeSpaceManager::cache_hdr_deserialize` now
  parses free-space header images as alignment/threshold/checksum records
  instead of routing them through the section-info decoder, rejects zero
  alignment, and verifies section-info checksums even for empty section lists.
- [x] `engine/group_api.rs::H5G__cache_node_deserialize` now enforces the
  serializer's NUL-terminated UTF-8 name stream instead of accepting an
  unterminated trailing name.
- [x] `engine/shared_message.rs` shared-message cache/table/list encoders now
  return `Result` and reject payload lengths that cannot be represented in
  the encoded u32 length field instead of narrowing silently.
- [x] `format/fractal_heap/mod.rs` indirect/direct block cache serializers
  now return `Result` and use checked image-length arithmetic before building
  checksum-protected metadata images; indirect-block serialization also
  rejects row counts that cannot be represented in the encoded u64 field.
- [x] Fixed-array and extensible-array data/index block cache serializers now
  return `Result` and use checked image-length arithmetic before appending
  checksum bytes, matching the checked image-length helpers used by their
  cache deserializers.
- [x] Local-heap free-list serialization and free-space section-info cache
  serialization now return `Result` and use checked image-length arithmetic
  instead of relying on unchecked `usize` multiplication/summing.
- [x] Group cache-node and fractal-heap row/indirect section serializers now
  return `Result`; cache-node serialization checks encoded name-stream length,
  and section serializers reject the wrong section class instead of encoding
  zeroed offset/size fields.
- [x] Object-header message/cache serializers now return `Result` and use
  checked encoded-image sizing, and free-space header serialization now
  rejects zero alignment to match the decoder's validated contract.
- [x] Additional object-message encoders (`layout`, `fsinfo`, new mtime,
  shared-message table info, B-tree K, external-file-list, and object name)
  now return `Result` and reject invalid versions, narrowing/truncation, bad
  slot counts, undefined addresses, or encoded-size overflow where applicable;
  the EFL size helper now reflects the full 16-byte fixed header.
- [x] Dataset chunk-index count-image encoders now return `Result` across the
  earray/farray/btree2/btree wrapper surfaces and validate that the entry
  count can be represented in the exact 8-byte image accepted by the decoders.
- [x] VFD family, multi, splitter, log, IOC, and subfiling superblock/config
  encoders now return `Result`, reuse decode-side validity rules, and reject
  string-length/count/usize narrowing before writing binary images.
- [x] Onion revision history encoding now returns `Result` and checks record
  count image length overflow; old-fill and datatype object-message encoders
  now return `Result` and reject old-fill length narrowing plus datatype image
  length overflow before emitting bytes.
- [x] Writer datatype encoding now returns `Result` and rejects compound/enum
  member-count narrowing, array rank/dimension/byte-size overflow, NUL-bearing
  metadata names, and opaque tag padded-length narrowing before writing bytes.
  HDFS/ROS3 property-list config encoders now likewise return `Result` and
  reject optional-string lengths that cannot be represented in the u32 image.
- [x] Writer metadata helpers for dataspaces, link messages, contiguous and
  chunked layouts, filter pipelines, fill-value messages, chunk B-tree nodes,
  dense B-tree headers, and managed-heap IDs now use fallible width/count
  checks instead of relying on unchecked `as` narrowing at the emit point.
- [x] Reference region/token encoding and aggregate property-list chunk
  encoding now return `Result` and check their u64 length prefixes before
  writing images; `H5P__encode_size_t` also rejects values wider than u64.
- [x] High-level writable-file and dataset-builder paths now reject
  unrepresentable attribute element counts, fixed-string lengths, compound
  field offsets, and compound type sizes before constructing writer specs.
- [x] Mutable-file resize/write-chunk paths now check dataspace rank,
  max-shape rank, chunk element counts, filtered chunk sizes, datatype sizes,
  and v2 B-tree root/child record counts before rewriting on-disk metadata.
- [x] Mutable fixed-array and extensible-array chunk-index updates now encode
  address, offset, chunk-size, and aggregate count/byte-size fields through
  checked width helpers instead of slicing little-endian integers directly.
- [x] Mutable v1/v2 chunk B-tree rebuilds now encode sibling/root/child
  addresses, node entry counts, child record counts, total record counts, and
  filtered chunk sizes through checked width helpers before writing metadata.
- [x] Object-header message and continuation readers now check chunk-local
  `u64 -> usize` spans before allocating message/padding/checksum buffers.
- [x] Dense attribute mutation now emits B-tree root/count fields through
  checked width helpers and validates fractal-heap ID offset widths; shared
  message and reference decode paths now check length-prefix conversion before
  allocating or slicing payloads.
- [x] Selection point/hyperslab serialization now checks rank/count conversion
  and validates decoded image lengths before allocating vectors; high-level
  attribute convenience creation and vlen string-reference decode now check
  length-prefix conversion explicitly.
- [x] File API EOF/accumulator offsets, metadata-cache image headers/entries,
  object-header checksum spans, and v2 superblock address emission now use
  explicit checked conversions instead of direct truncating casts or dynamic
  little-endian slices.
- [x] Dataset variable-length value/string descriptor decoding now converts
  sequence-length fields through checked helpers before slicing, allocating,
  or computing payload byte counts.
- [x] Dataset contiguous/chunked read paths and chunk-info enumeration now
  convert fallback byte sizes, implicit chunk indexes, and chunk coordinates
  with checked helpers; fixed/extensible array test-image helpers now reject
  unrepresentable usize fields instead of truncating them into u64 images.
- [x] V1 symbol-table group member collection now checks local-heap name
  offsets before converting them for heap-string lookup.
- [x] Typed conversion and dataset value-read paths now convert decoded
  datatype sizes and array dimensions through checked helpers before using
  them as record sizes, strides, or payload byte counts.
- [x] Global-heap trace/test helpers no longer truncate object data lengths or
  synthetic collection sizes when crossing usize/u64 boundaries.
- [x] Object-header v1 cache validation, external-file-list slot decoding,
  symbol-table node counts, and v1 B-tree child/key loops now use explicit
  checked or lossless conversions instead of direct integer casts.
- [x] Variable-length dataset trace instrumentation now accepts native usize
  lengths and performs a saturating trace-only conversion to u64 internally.
- [x] N-bit filter parameter parsing now routes decoded counts, datatype
  sizes, precisions, offsets, array sizes, and compound member metadata
  through checked `u32 -> usize` conversions before allocating or iterating.
- [x] Scale-offset filter parameter parsing now checks decoded element
  counts, datatype sizes, and minimum-bit-count fields before using them as
  host sizes or serialized header values.
- [x] Fill-value, compact data-layout, and attribute message decoding now
  checks length/datatype-size fields before slicing payload windows or
  computing expected attribute byte counts.
- [x] Datatype compound/enum/array decoding now checks encoded datatype sizes
  before computing member offsets, record bounds, enum value widths, and array
  byte counts; remaining small-width message decoder counts now use lossless
  `usize::from` conversions.
- [x] VFD superblock and property-list config decoders now check serialized
  `u32` string/count fields before using them as host lengths.
- [x] Dataspace, link-info, symbol-table, attribute-info, and object-header
  shared/external-message decoders now use lossless address/rank width
  conversions instead of direct casts.
- [x] Fractal heap tiny/managed/huge/header helpers now use lossless or
  checked conversions for heap-ID lengths, table widths, root row counts,
  filtered payload sizes, and in-memory object IDs.
- [x] V2 B-tree header/node parsing and cache image sizing now checks node
  and record sizes, depth/count indexes, child record counts, and
  record/image byte totals before indexing, allocating, or serializing.
- [x] Mutable and high-level chunk-index B-tree paths now use lossless or
  checked conversions for address/size widths, entry counts, root/child
  record counts, chunk sizes, filter masks, and header patch offsets.
- [x] Fixed-array and extensible-array chunk-index decoders and mutable
  patch paths now use lossless or checked conversions for raw element sizes,
  address widths, array-offset widths, page/block shifts, decoded element
  counts, and checksum/address byte spans.
- [x] High-level dataset, attribute, virtual-dataset, and dense-group read
  paths now use lossless or checked conversions for datatype sizes,
  address/length widths, external-file-list slot counts, vlen descriptors,
  dense heap-ID lengths, reference addresses, and ndarray dimensions.
- [x] Object-header and superblock parsing now uses lossless or checked
  conversions for message sizes/types, continuation widths, chunk data sizes,
  checksum spans, trace metadata lengths, and superblock address/size widths.
- [x] Mutable-file resize and attribute mutation paths now use lossless or
  checked conversions for v2 object-header message scanners, compact/dense
  attribute name offsets, dense heap IDs/record sizes, address/size widths,
  direct-block checksum positions, and append/checksum seek offsets.
- [x] Fractal-heap indirect-block, local-heap undefined value, symbol-table
  scratch-pad, and v2 B-tree validation paths now use lossless or checked
  conversions for heap offset widths, checksum spans, entry/address widths,
  undefined address/length bit counts, and metadata prefix comparisons.
- [x] LZF, Fletcher32, and filter-pipeline paths now use lossless byte and
  mask conversions, checked trace length widening, and checked LZF fallback
  expected-size arithmetic instead of truncating through `u32`.
- [x] N-bit and scale-offset bitstream paths now use explicit widening and
  checked bit-count conversions for packed byte runs, parameter byte widths,
  integer min-bit calculations, and low-bit float reconstruction helpers.
- [x] Filter-pipeline, dataspace, fill-value, attribute-info, data-layout,
  and link-info message helpers now use lossless or checked conversions for
  trace metadata lengths, enum trace values, address/size byte folds, chunk
  element sizes, and signed max-creation-index bounds.
- [x] Datatype message decoding and related high-level accessors now use
  lossless or checked conversions for trace fields, bit precision/offset
  metadata, compound/enum member counts, inline array dimensions, variable
  offset byte folds, legacy array sizes, dataspace ranks, chunk size bit
  counts, scalar i32 widening, and VDS selection encoded widths.
- [x] Engine file/dataset/group/shared-message/VFD/object helpers and global
  heap parsing now use lossless or checked conversions for byte checksums,
  cache IDs, chunk table addresses, bit shifts, group cache indexes, driver
  IDs, write spans, signature offsets, refcount deltas, message creation
  indexes, address/size widths, and global heap object indexes.
- [x] Writer datatype, object-header, layout, dense-storage, global-heap,
  fractal-heap, chunk-index, and allocation paths now use explicit lossless
  widening or checked narrowing for enum base sizes, fixed-point precision,
  name length classes, address/size slices, message sizes, heap object/free
  sizes, dataset payload lengths, chunk coordinates, B-tree node sizes,
  managed heap IDs, and dense B-tree record counts.

Cleared on inspection (recorded so a future scan doesn't re-flag them):

- `format/checksum.rs::fletcher32` — `360` batch size is present (line 47);
  the prior diff was an analyzer artifact, since fixed in ccc-rs.
- `format/superblock.rs::read_v0_v1` — literal `32`/`16`/`16`/`16` are
  `HDF5_BTREE_CHUNK_IK_DEF` and the spec-fixed scratch-pad size.
- `format/btree_v2.rs::read_internal_records` — literals `10` and `11` are
  the chunk-no-filter / chunk-with-filter B-tree types used to gate the
  `tracehash` probe; not magic numbers.
- `format/symbol_table.rs::read_entry` — literal `16`/`12` are the
  spec-fixed `H5G_SIZEOF_SCRATCH = 16` scratch-pad and its remainder for
  `cache_type==2`.
- `filters/shuffle.rs` — does not unroll per-element-size like
  `H5Z__filter_shuffle`; performance trade-off, not correctness.
