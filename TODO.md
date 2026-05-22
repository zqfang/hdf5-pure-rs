# TODO: Remaining Feature Work

This file tracks active feature gaps for `hdf5-pure-rust`. Historical audit
notes and completed migration details should live outside this active backlog.

## Next Implementation Checklist

- [ ] Existing-file chunk mutation
  - [x] Prune whole v2-B-tree chunk records outside a shrunken extent.
  - [x] Scrub retained partial-edge chunks on shrink for v1 B-tree,
    fixed-array, 1D extensible-array, and multidimensional v2-B-tree indexes.
    Synthetic v1 userblock partial-edge shrink is supported for v1 B-tree
    indexes; modern userblock chunk-index mutation remains a separate
    base-address conversion task.
  - [x] Support sparse/non-next `write_chunk` for extensible-array indexes,
    including index-data-block and sparse super-block allocation.
  - [x] Replace existing records in deep v2-B-tree indexes in place instead of
    rebuilding the tree.
  - [x] Add paged fixed-array and paged extensible-array mutation coverage.
- [ ] Existing-file group/link/dataset mutation
  - [x] Expose compact-parent `create_group`, `link_*`, and `new_dataset*`
    through `Group` handles opened from `File::open_rw`.
  - [ ] Extend creation/relink/unlink through dense, creation-order-indexed,
    and v1 symbol-table parent chains.
  - [ ] Finish persistent hard-link refcount maintenance for the remaining
    dense/v1/non-compact deletion cases.
- [ ] Writer/read compatibility coverage
  - [ ] Add h5py round-trip coverage alongside current h5dump coverage for
    newly expanded writer features.
  - [ ] Broaden NBit, ScaleOffset, VDS, and datatype-conversion fixtures toward
    practical libhdf5 parity.
  - [ ] Add explicit unsupported stubs for remaining low-level libhdf5 APIs
    that are in scope for compatibility but not for this backend.

## Drop-In API Gaps

- [x] Implement `Group::create_group` for supported existing-file parent
  chains opened through `File::open_rw`.
- [x] Implement `Group::link_soft`, `Group::link_hard`, and
  `Group::link_external` for supported existing-file parent chains opened
  through `File::open_rw`.
- [x] Implement `Group::new_dataset` and `Group::new_dataset_builder` for
  supported existing-file parent chains opened through `File::open_rw`.
- [x] Generalize `Group::{create_group,link_*,new_dataset*}` to supported
  direct dense destinations.
- [ ] Generalize `Group::relink` beyond same-group, same-size compact link
  renames.
- [x] Implement `Dataspace::encode`.
- [x] Decide whether `File::start_swmr` and `OpenMode::ReadSWMR` should remain
  explicit unsupported boundaries or grow real SWMR behavior.
- [ ] Replace placeholder-level FCPL/FAPL behavior with meaningful property-list
  effects where they map to this pure-Rust backend.

## Existing-File Mutation

- [x] Add dense-link same-size rename support.
- [x] Add dense-link non-hard unlink support.
- [x] Add open_rw same-group hard-link creation for direct dense parents,
  including persistent refcount materialization and dense target address
  patching when the target object header is rewritten.
- [ ] Extend hard-link deletion beyond the current safe compact cases by
  emitting and maintaining persistent object refcount messages. Compact
  deletion can now handle compact and direct dense no-explicit-refcount cases
  when a root reachability walk proves exactly one other hard link remains;
  compact deletion can also materialize a persistent refcount and patch
  remaining compact hard links when more than one link remains and all
  reachable links to the target are in compact v2 parents. Direct dense
  same-parent deletion can now materialize a persistent refcount and patch the
  remaining dense links when all reachable links to the target are in that
  dense parent. Dense cross-parent deletion can now materialize a persistent
  refcount and patch remaining compact and direct dense single-leaf links.
  Dense cross-parent deletion with creation-order indexes, non-leaf dense name
  indexes, filtered/indirect heaps, or v1 parent groups remains explicit.
  Existing v2 persistent refcount messages are removed again when a deletion
  leaves one hard link, matching the original object-header refcount-message
  lifecycle.
- [x] Emit or increment persistent object refcount messages for compact
  same-group hard-link aliases and decrement them on compact unlink.
- [x] Add compact root link-name size changes by rebuilding root metadata.
- [x] Generalize compact same-group link-name size changes beyond root relinks
  by rebuilding compact hard-link parent chains.
- [ ] Generalize link-name size changes to dense layouts beyond direct
  unfiltered single-leaf name-index shrink-in-place and root/nested dense
  destination rebuilds, creation-order indexed, v1, and non-compact parent
  chains. Dense layouts that require retaining dense storage with a new heap
  object or larger direct block remain explicit.
- [x] Add compact v2/v3 cross-group relink support with parent propagation.
- [ ] Generalize cross-group relink to dense, creation-order indexed, v1, and
  non-compact parent chains. Compact root-source relink into a direct,
  unfiltered dense destination with a single-leaf name index is supported;
  nested compact sources into that same dense destination shape are supported.
  Dense sources in compact parent chains can move to compact or direct dense
  destinations for that same direct, unfiltered single-leaf name-index shape;
  direct dense sources can also move into direct dense destinations by
  rebuilding both endpoints as compact metadata. Creation-order indexes,
  non-leaf dense indexes, indirect heaps, and v1 symbol-table groups remain
  explicit unsupported boundaries.
- [ ] Add v1 symbol-table group mutation only after SNOD, B-tree, local-heap
  write-location tracking, refcounts, and free-space semantics are modeled.
- [x] Add root-only existing-file `create_group` for v2/v3 compact root groups
  as the first creation slice.
- [x] Add root-only existing-file soft/external link creation.
- [x] Add root-only existing-file simple dataset creation.
- [x] Generalize existing-file creation from root-only to direct compact
  root-child groups by propagating rewritten parent addresses up to the root
  and updating the superblock.
- [x] Generalize existing-file creation to deeper nested compact hard-link
  parent chains by propagating rewritten addresses up to the root and updating
  the superblock.
- [ ] Generalize existing-file creation to all non-compact parent chains.
  Direct, unfiltered dense parents with a leaf name index can now grow groups,
  datasets, and links by rebuilding the destination as compact metadata.
  Cross-group hard-link creation can now bridge
  compact destinations with direct dense target groups and direct dense
  destinations with compact target groups when the endpoint parent paths are
  compact-reachable; indirect heaps, filtered heaps, creation-order indexes,
  dense parent-chain propagation for rebuilt compact target parents, and v1
  symbol-table groups remain explicit unsupported boundaries.

## Reader And Format Coverage

- [x] Keep SZIP explicitly unsupported and pinned by tests.
- [x] Add BLOSC support if a suitable pure-Rust implementation is acceptable;
  otherwise keep the explicit unsupported boundary pinned by tests.
- [ ] Broaden ScaleOffset and NBit coverage to the full practical libhdf5
  parameter space.
- [x] Match ScaleOffset signed integer compression for two's-complement
  datatypes by using signed minima and signed deltas.
- [x] Match ScaleOffset integer compression for explicit fixed retained-bit
  counts from `client_data[1]`, including oversized counts capped to datatype
  width.
- [x] Pin HDF5 datatype message v5 as invalid because libhdf5's latest
  datatype message version is v4.
- [x] Add typed read support for HDF5 time datatypes if needed by real fixtures.
- [ ] Broaden datatype conversion behavior beyond the current supported numeric,
  string, enum, compound, reference, and VDS conversion paths.
- [ ] Broaden VDS mapping, selection, missing-source, and conversion coverage.
  Upstream libhdf5 multi-file 3D mosaic, fill-gap, and 3D printf-source VDS
  fixtures are now pinned by integration tests, along with an unlimited
  source-block VDS fixture.
  - [ ] Broaden chunk-index mutation support by index type:
  - [ ] Fixed-array: replacement is supported; append/growth after resize
    remains. General growth needs fixed-array metadata rebuild/relink, preserved
    records, checksum updates, and multidimensional reindexing rather than just
    allowing out-of-range element writes.
  - [ ] Extensible-array: selected replacement, paged replacement,
    index-data-block sparse/non-next allocation, and append paths are
    supported; sparse super-block allocation and broader super-block updates
    remain.
  - [ ] v2 B-tree: simple type 10/11 update, deep-tree existing-record
    replacement, append, and shrink pruning are supported; deeper-tree append
    and other record layouts remain.

## Writer Coverage

- [ ] Broaden writer-side modern chunk-index creation/selection parity beyond
  the current fixed-array, one-unlimited-dimension extensible-array,
  finite/growable max-shape v2-B-tree, fill-only max-shape undefined-index
  layouts, and multi-growable or large-grid v2 B-tree creation cases.
- [x] Stop always using v1 B-tree indexes for new chunked datasets by selecting
  fixed-array, extensible-array, or v2 B-tree indexes for supported writer
  layouts.
- [x] Create fixed-shape datasets with v4 fixed-array indexes, including
  filtered and paged/deeper fixed-array layouts.
- [x] Add sparse/fill-only and explicit sparse chunk-list dataset creation,
  including filtered chunks, so huge logical datasets do not require a full
  in-memory payload or full chunk enumeration.
- [x] Allow vlen UTF-8 datasets to use chunked storage and filters by writing
  chunked 16-byte heap descriptors while keeping string payloads in heap
  storage.
- [x] Add vlen UTF-8 fill values by encoding a null or heap-backed vlen
  descriptor in the fill-value message.
- [x] Make global-heap collection encoding width-aware instead of assuming
  8-byte size fields, if superblock size-width configurability is added.
- [x] Remove arbitrary 4 GiB caps from global-heap object and reference-region
  helpers where the on-disk size fields can represent larger payloads; keep
  allocation/streaming checks at the real boundary.
- [x] Add object-header continuation chunk writing and/or repacking so large
  metadata messages and long compact-message lists do not fail at the current
  single-chunk boundary.
- [x] Route oversized attributes to dense attribute storage even when attribute
  count is below the compact-to-dense threshold.
- [ ] Generalize dense link and dense attribute storage beyond one root direct
  fractal-heap block; support indirect/filtered heap blocks. Internal name
  B-tree nodes, wider heap offsets, and larger heap IDs are covered, and the
  root-direct overflow boundary now fails explicitly as the next heap-growth
  slice.
- [x] Widen dense link and dense attribute heap IDs when payload lengths exceed
  the previous fixed-width length fields.
- [x] Build dense link storage from one unified link list so soft, external,
  hard-link aliases, and object links can all densify together instead of only
  densifying ordinary object links.
- [x] Support 8-byte v2 link-name length encoding for link names larger than
  the current 4-byte length-field writer path.
- [x] Fix `H5T__set_size` for large fixed-length string datatypes so string
  sizes are not constrained by the fixed-point precision `u16` path.
- [ ] Add h5dump/h5py round-trip coverage for writer-created modern chunk
  indexes before declaring each layout fully supported. Finite max-shape v2
  B-tree writer output now has h5py coverage.
- [ ] Add h5dump/h5py round-trip coverage for mutable resize/write_chunk cases
  before declaring each mutation path fully supported. v1 B-tree partial
  shrink/write, fixed-array replacement, paged fixed-array replacement,
  extensible-array replacement, v2 B-tree replacement, deep v2 B-tree
  replacement, sparse extensible-array super-block write, and paged
  extensible-array super-block replacement now have h5py coverage.

## Compatibility And Test Coverage

- [x] Keep hdf5-metno public functions present and not deprecated.
- [ ] Add explicit unsupported API stubs for low-level libhdf5 entry points that
  do not map to this crate's high-level API, especially additional `H5F*`,
  `H5FD*`, `H5VL*`, `H5ES*`, `H5PL*`, and `H5M*` surfaces. Keep this limited
  to real libhdf5 symbols or true unsupported runtime boundaries; avoid adding
  Rust-only pass-through aliases. `H5Fget_metadata_read_retry_info` is now
  covered by the public file-API compatibility state.
- [x] Extend deterministic property/config parsing for unsupported VFDs:
  family, multi, splitter, log, onion, subfiling, HDFS, ROS3/S3, and related
  malformed-buffer tests.
- [x] Add generated malformed fixtures for remaining NBit and ScaleOffset edge
  cases.
- [ ] Add deeper soft-link path-normalization coverage if link-resolution parity
  becomes a public compatibility goal.
- [ ] Add deeper fractal-heap growth and checksum-corruption coverage if dense
  storage mutation expands beyond the current layouts.

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
