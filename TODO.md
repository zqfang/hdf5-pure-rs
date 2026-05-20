# TODO: Remaining Feature Work

This file tracks active feature gaps for `hdf5-pure-rust`. Historical audit
notes and completed migration details should live outside this active backlog.

## Drop-In API Gaps

- [ ] Implement `Group::create_group` for existing files opened through
  `File::open_rw`.
- [ ] Implement `Group::link_soft`, `Group::link_hard`, and
  `Group::link_external` for existing files opened through `File::open_rw`.
- [ ] Implement `Group::new_dataset` and `Group::new_dataset_builder` for
  existing files opened through `File::open_rw`.
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
  unfiltered single-leaf name-index shrink-in-place, creation-order indexed,
  v1, and non-compact parent chains. Dense link growth that needs a new heap
  object or larger direct block remains explicit.
- [x] Add compact v2/v3 cross-group relink support with parent propagation.
- [ ] Generalize cross-group relink to dense, creation-order indexed, v1, and
  non-compact parent chains. Compact root-source relink into a direct,
  unfiltered dense destination with a single-leaf name index is supported;
  nested compact sources into that same dense destination shape are supported.
  Dense sources in compact parent chains can move to compact destinations for
  that same direct, unfiltered single-leaf name-index shape; dense sources can
  move to dense destinations through dense hard-link parent paths when neither
  endpoint group object header must be rebuilt. Creation-order indexes,
  non-leaf dense indexes, and v1 symbol-table groups remain explicit
  unsupported boundaries.
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
  datasets, and links within the existing heap direct block. Cross-group
  hard-link creation can now bridge
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
- [ ] Broaden chunk-index mutation support beyond the current append/replace
  cases for fixed-array, extensible-array, and v2 B-tree indexes.

## Writer Coverage

- [ ] Broaden writer-side modern chunk-index parity beyond the current
  fixed-array, fixed max-shape, at-most-one-growable-dimension extensible-array,
  fill-only max-shape undefined-index layouts, and multi-growable or large-grid
  v2 B-tree creation cases.
- [ ] Create new chunked datasets with fixed-array, extensible-array, or v2
  B-tree indexes where appropriate instead of always using v1 B-tree indexes.
- [x] Create unfiltered fixed-size fully materialized multi-chunk datasets with
  a v4 fixed-array chunk index when they fit in one fixed-array page.
- [x] Create filtered fixed-size fully materialized multi-chunk datasets with a
  v4 fixed-array chunk index when they fit in one fixed-array page.
- [ ] Add growth support for modern chunk indexes.
- [x] Remove writer limits that libhdf5 does not impose by implementing
  multi-level v1 chunk B-tree creation instead of the current two-level writer
  cap.
- [x] Add sparse/fill-only chunked dataset creation and streaming chunk
  insertion so huge logical datasets do not require a full in-memory payload or
  full chunk enumeration.
- [x] Create explicit sparse chunk-list datasets, including filtered chunks,
  with v4 fixed-array indexes for fixed shapes and inline v4 extensible-array
  indexes for small max-shape grids.
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
- [ ] Add broader h5dump/h5py round-trip coverage for each newly expanded writer
  feature before declaring it supported.

## Compatibility And Test Coverage

- [x] Keep hdf5-metno public functions present and not deprecated.
- [ ] Add explicit unsupported API stubs for low-level libhdf5 entry points that
  do not map to this crate's high-level API, especially additional `H5F*`,
  `H5FD*`, `H5VL*`, `H5ES*`, `H5PL*`, and `H5M*` surfaces. Keep this limited
  to real libhdf5 symbols or true unsupported runtime boundaries; avoid adding
  Rust-only pass-through aliases.
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
