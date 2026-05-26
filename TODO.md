# TODO: Remaining Feature Work

This file tracks active feature gaps for `hdf5-pure-rust`. Historical audit
notes and completed migration details should live outside this active backlog.

## Next Implementation Checklist

- [ ] Existing-file chunk mutation
- [ ] Existing-file group/link/dataset mutation
  - [ ] Extend creation/relink/unlink through dense, creation-order-indexed,
    and v1 symbol-table parent chains.
  - [ ] Finish persistent hard-link refcount maintenance for the remaining
    dense/v1/non-compact deletion cases.
- [ ] Writer/read compatibility coverage
  - [ ] Keep h5py round-trip coverage alongside h5dump coverage for writer
    features added beyond the current dense attr/link and modern chunk-index
    fixtures.
  - [ ] Broaden NBit, ScaleOffset, VDS, and datatype-conversion fixtures toward
    practical libhdf5 parity.
  - [ ] Audit remaining low-level libhdf5 APIs and translate real behavior
    where it maps to this backend; avoid adding unsupported-only stubs.

## Drop-In API Gaps

- [ ] Continue replacing placeholder-level FCPL/FAPL behavior with meaningful
  property-list effects where they map to this pure-Rust backend. Remaining
  shared-message work is actual object-header message sharing into SOHM
  records/heaps; the writer now persists configured empty SOHM index headers
  through the superblock extension and `FileCreate::from_file` reports them.

## Existing-File Mutation

- [ ] Extend hard-link deletion beyond the current safe compact cases by
  emitting and maintaining persistent object refcount messages.
  Dense cross-parent deletion with creation-order indexes, non-leaf dense name
  indexes, filtered/indirect heaps, or v1 parent groups remains explicit.
- [ ] Generalize link-name size changes to dense layouts beyond direct
  unfiltered single-leaf name-index shrink-in-place, same-size creation-order
  indexed dense relink, and root/nested dense destination rebuilds. V1 and
  non-compact parent chains remain explicit. Dense layouts that require
  retaining dense storage with a new heap object or larger direct block remain
  explicit.
- [ ] Generalize cross-group relink beyond the currently supported compact and
  direct dense single-leaf shapes. Creation-order indexes, non-leaf dense
  indexes, indirect heaps, and v1 symbol-table groups remain explicit
  unsupported boundaries.
- [ ] Add v1 symbol-table group mutation only after SNOD, B-tree, local-heap
  write-location tracking, refcounts, and free-space semantics are modeled.
- [ ] Generalize existing-file creation to all non-compact parent chains.
  Indirect heaps, filtered heaps, creation-order indexes, and v1 symbol-table
  groups remain explicit unsupported boundaries.

## Reader And Format Coverage

- [ ] Broaden ScaleOffset and NBit coverage to the full practical libhdf5
  parameter space. ScaleOffset full-precision chunks now follow libhdf5's raw
  payload passthrough behavior for read and in-memory encode paths.
- [ ] Broaden datatype conversion behavior beyond the current supported numeric,
  string, enum, compound, reference, VDS, and same-shape array-base conversion
  paths.
- [ ] Broaden VDS mapping, selection, missing-source, and conversion coverage.
  Fill-policy reads now cover missing source files and missing source dataset
  paths for typed and caller-owned raw reads, and `%b` source-file mappings now
  expand available block files for unlimited regular hyperslab VDS reads;
  broader mapping and conversion combinations remain.
  - [ ] Broaden chunk-index mutation support by index type:
  - [ ] Fixed-array: 1D append/growth after resize is supported by rebuilding
    and relinking the fixed-array metadata with preserved records and checksums.
    General multidimensional growth still needs old-grid reconstruction for
    safe record reindexing after the dataspace message has already changed.
  - [ ] Extensible-array: sparse super-block allocation is implemented and
    covered by h5dump/h5py; broader super-block updates remain.
  - [ ] v2 B-tree: deeper-tree append and other record layouts remain.

## Writer Coverage

- [ ] Broaden writer-side modern chunk-index creation/selection parity beyond
  the current fixed-array, one-unlimited-dimension extensible-array,
  finite/growable max-shape v2-B-tree, fill-only max-shape undefined-index
  layouts, and multi-growable or large-grid v2 B-tree creation cases.
- [ ] Generalize dense link and dense attribute storage beyond the current
  writer-side deflated root-indirect fractal-heap managed direct blocks.
  Remaining dense-writer gaps include direct-root filtered heaps, non-deflate
  heap pipelines, and wider/deeper filtered indirect-table layouts.
- [ ] Keep h5dump/h5py round-trip coverage in step with new writer-created
  modern chunk-index layouts beyond the current fill-only B-tree v2
  undefined-index coverage.

## Compatibility And Test Coverage

- [ ] Audit low-level libhdf5 entry points that do not yet map to this crate's
  high-level API and translate actual behavior where the backend can support it.
  Public `H5F*` surfaces have been audited against the current `H5Fpublic.h`;
  public `H5ES*` and `H5VL*` surfaces are audited in the engine compatibility
  wrappers; continue with any remaining `H5FD*` surfaces beyond public
  unsupported VFD init/term boundaries. Avoid adding unsupported-only stubs or
  Rust-only pass-through aliases.
  Oversized shared-message payload encode-error output preservation remains
  covered by prevalidation rather than a cheap regression fixture because it
  requires a payload longer than `u32::MAX`.

## Explicitly Out Of Scope For Now

These are not blocked by Rust; they are outside this crate's current runtime
scope unless a real use case appears.

- MPI, parallel HDF5, and distributed dataset/selection I/O.
- VOL, async, plugin, and connector infrastructure.
- Alternative VFD runtime parity for cloud/network/distributed drivers beyond
  deterministic config parsing and explicit unsupported errors.
- Map API parity.
- Broad package-init, free-list, and thread-runtime machinery that has no
  high-level pure-Rust API use.
- Full C-library metadata-cache runtime behavior, file mounting, and public APIs
  that do not map to this crate's high-level API.
