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
- [ ] Implement `Dataspace::encode`.
- [ ] Decide whether `File::start_swmr` and `OpenMode::ReadSWMR` should remain
  explicit unsupported boundaries or grow real SWMR behavior.
- [ ] Replace placeholder-level FCPL/FAPL behavior with meaningful property-list
  effects where they map to this pure-Rust backend.

## Existing-File Mutation

- [ ] Add dense-link same-size rename support.
- [ ] Add dense-link non-hard unlink support.
- [ ] Extend hard-link deletion beyond the current safe compact cases by
  emitting and maintaining persistent object refcount messages.
- [ ] Add support for link-name size changes by rewriting or repacking metadata.
- [ ] Add cross-group relink support with parent propagation.
- [ ] Add v1 symbol-table group mutation only after SNOD, B-tree, local-heap
  write-location tracking, refcounts, and free-space semantics are modeled.
- [ ] Add root-only existing-file `create_group` for v2/v3 compact root groups
  as the first creation slice.
- [ ] Add root-only existing-file soft/external link creation.
- [ ] Add root-only existing-file simple dataset creation.
- [ ] Generalize existing-file creation from root-only to nested groups by
  propagating rewritten parent addresses up to the root and updating the
  superblock.

## Reader And Format Coverage

- [ ] Add SZIP support if a suitable pure-Rust decoder becomes available;
  otherwise keep the explicit unsupported boundary pinned by tests.
- [ ] Add BLOSC support if a suitable pure-Rust implementation is acceptable;
  otherwise keep the explicit unsupported boundary pinned by tests.
- [ ] Broaden ScaleOffset and NBit coverage to the full practical libhdf5
  parameter space.
- [ ] Add HDF5 datatype message v5 support.
- [ ] Add typed read support for HDF5 time datatypes if needed by real fixtures.
- [ ] Broaden datatype conversion behavior beyond the current supported numeric,
  string, enum, compound, reference, and VDS conversion paths.
- [ ] Broaden VDS mapping, selection, missing-source, and conversion coverage.
- [ ] Broaden chunk-index mutation support beyond the current append/replace
  cases for fixed-array, extensible-array, and v2 B-tree indexes.

## Writer Coverage

- [ ] Broaden writer-side modern chunk-index parity.
- [ ] Create new chunked datasets with fixed-array, extensible-array, or v2
  B-tree indexes where appropriate instead of always using v1 B-tree indexes.
- [ ] Add growth support for modern chunk indexes.
- [ ] Remove writer limits that libhdf5 does not impose by implementing
  multi-level v1 chunk B-tree creation instead of the current two-level writer
  cap.
- [ ] Add sparse/fill-only chunked dataset creation and streaming chunk
  insertion so huge logical datasets do not require a full in-memory payload or
  full chunk enumeration.
- [ ] Allow vlen UTF-8 datasets to use chunked storage and filters by writing
  chunked 16-byte heap descriptors while keeping string payloads in heap
  storage.
- [ ] Add vlen UTF-8 fill values by encoding a null or heap-backed vlen
  descriptor in the fill-value message.
- [ ] Make global-heap collection encoding width-aware instead of assuming
  8-byte size fields, if superblock size-width configurability is added.
- [ ] Remove arbitrary 4 GiB caps from global-heap object and reference-region
  helpers where the on-disk size fields can represent larger payloads; keep
  allocation/streaming checks at the real boundary.
- [ ] Add object-header continuation chunk writing and/or repacking so large
  metadata messages and long compact-message lists do not fail at the current
  single-chunk boundary.
- [ ] Route oversized attributes to dense attribute storage even when attribute
  count is below the compact-to-dense threshold.
- [ ] Generalize dense link and dense attribute storage beyond one leaf B-tree
  node and one direct fractal-heap block; support internal nodes, wider heap
  offsets, larger heap IDs, and indirect/filtered heap blocks.
- [ ] Build dense link storage from one unified link list so soft, external,
  hard-link aliases, and object links can all densify together instead of only
  densifying ordinary object links.
- [ ] Support 8-byte v2 link-name length encoding for link names larger than
  the current 4-byte length-field writer path.
- [ ] Fix `H5T__set_size` for large fixed-length string datatypes so string
  sizes are not constrained by the fixed-point precision `u16` path.
- [ ] Add broader h5dump/h5py round-trip coverage for each newly expanded writer
  feature before declaring it supported.

## Compatibility And Test Coverage

- [ ] Keep hdf5-metno public functions present and not deprecated.
- [ ] Add explicit unsupported API stubs for low-level libhdf5 entry points that
  do not map to this crate's high-level API, especially additional `H5F*`,
  `H5FD*`, `H5VL*`, `H5ES*`, `H5PL*`, and `H5M*` surfaces.
- [ ] Extend deterministic property/config parsing for unsupported VFDs:
  family, multi, splitter, log, onion, subfiling, HDFS, ROS3/S3, and related
  malformed-buffer tests.
- [ ] Add generated malformed fixtures for remaining NBit and ScaleOffset edge
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
