# TODO: hdf5-pure-rust

Current status: 939 tests, 0 failures, 0 warnings.

This file tracks the remaining translation and release work. Completed audit
history has been moved out of the active backlog; unsupported-feature details
live in `analysis/unsupported_features.md`, and raw `ccc-rs missing`
classification lives in `analysis/ccc_missing_roadmap.md`.

## P0: Release Blockers

- [ ] Run the release checklist before publishing:
  fixture regeneration, default and feature test matrix, README feature
  review, crate packaging, and dependency/license checks.

## P1: Remaining Translation Parity

These items are the remaining libhdf5 feature families that are still plausible
translation work within this crate's intended scope.

- [x] Close the small raw CCC gaps before broad subsystem work.
  The raw `H5A*`, `H5T*`, and `H5O*` missing-symbol entries are closed in the
  current workspace. `H5Tget_*` datatype accessors and the public
  `H5Oget_info*` / `H5Oget_comment*` object queries now have exact Rust entry
  points for audit mapping; shared-message `H5O__shared_*` entry points also
  use the existing table/reference model, including configured address-width
  decode. The first object-message and object-header-cache pass expands
  append/write/remove/iterate/share/checksum behavior on the existing
  `ObjectHeaderState` model. The next H5O pass expanded object-table
  refcount/traversal/token/debug behavior and moved filter-pipeline message
  decode/copy/reset/debug into the exact `H5O__pline_*` entry points. A follow-up
  H5O pass tightened compact attribute name matching and mutation, expanded
  `H5O__attr_*` decode/copy/debug behavior, and gave chunk/allocation helpers
  real null-gap merging, creation-index maintenance, and release/unprotect
  effects. Later H5O passes expanded refcount/mtime/fsinfo/stab/sdspace message
  routines plus object-header copy/flush/refresh behavior on the current
  in-memory model, then object lifecycle/info, link/link-info/attribute-info,
  external-file-list copy/encode/decode, and fill-value decode/copy/reset/free
  paths. Follow-up message/shared/free passes expanded owned
  free/copy/remove/delete/flush normalization, direct shared-message default
  decode/encode/size/link/copy callbacks, object-message free callbacks,
  object-info/copy wrappers, continuation/group/refcount/free callbacks, and
  dataset/datatype/name callbacks. The latest object pass corrected group
  storage accounting to follow the C link-info versus symbol-table branch,
  expanded shared-read/message allocation/iteration/flush behavior, and moved
  small refcount/shared-message-table/filter-pipeline/driver-info callbacks
  away from one-line aliases into direct field-level translations. The newest
  pass inlined object-header prefix validation, tightened v2 continuation chunk
  message scanning, and expanded data-layout decode/encode/copy-file checks
  from decoded fields while preserving exact raw round-trips for decoded
  messages. The follow-up H5O pass replaced boolean/debug placeholders with
  object-header invariant checks, added address-deduplicating visit behavior,
  expanded committed-datatype copy tracking, packed object-header messages
  forward across null gaps, rebuilt object-header copy/allocation routines
  from normalized message streams, and expanded datatype decode/encode/debug
  callbacks with class-specific validation and C-style flag/property reporting.
  The next parity pass added dense-group storage deletion and the remaining
  property-list raw entries (`H5P__free_prop`, `H5P__dcrt_layout_del`, and
  `H5Pset_dataset_io_hyperslab_selection`) with direct Rust property-list
  behavior. The latest cleanup pass added the remaining raw cache/VFD/dataset
  callback entries for clean-list marking, mirror uint8 transmit encoding,
  earray index dumping, and the explicit MPI/VOL unsupported boundaries that
  remain outside this crate's runtime scope. The datatype follow-up expanded
  `H5T__set_precision`, `H5T__set_offset`, and `H5T__set_size` to adjust
  atomic datatype storage size, bit offsets, precision, fixed/string variable
  sizing, and compound shrink validation with C-style invalid-class/read-only
  checks; it also added enum-aware `H5Tget_member_index` behavior, direct
  variable-length string query coverage, exact little-endian bit-region
  behavior for `H5T__bit_copy`, `H5T__bit_set`, and `H5T__bit_find`, C-style
  class/read-only validation for `H5Tset_cset`, `H5Tset_strpad`, and
  `H5Tset_tag`, fixed/vlen string cset and padding flag accessors, and
  validated numeric setter behavior for `H5Tset_fields`, `H5Tset_norm`,
  `H5Tset_inpad`, `H5Tset_sign`, and `H5Tset_pad`; it replaced the
  `H5Tinsert` unsupported boundary with validated compound member encoding,
  added bit-range carry/borrow/negation behavior for `H5T__bit_inc`,
  `H5T__bit_dec`, and `H5T__bit_neg`, replaced native-float
  `H5T__bit_cmp`/`H5T__fix_order` byte-copy stand-ins with the C
  permutation/masked-bit logic, expanded disk-backed reference and vlen
  callbacks to encode/decode reference headers, blob sizes, legacy object
  tokens, dataset-region token/space payloads, and vlen length-prefixed
  blobs, translated `H5T__reverse_order`/`H5T__conv_order_opt` byte-order
  handling for little-endian, big-endian atomic, big-endian complex, and VAX
  pair ordering, added C-style `H5T__conv_float_find_special`
  zero/infinity/NaN/sign classification, translated `H5T_set_version` file
  low/high-bound datatype encoding checks, expanded `H5T_is_vl_storage` to
  recurse through compound/array/enum datatype messages and reference members,
  aligned `H5T_is_sensible` with the C empty-compound/empty-enum checks,
  expanded `H5T__enum_nameof`, `H5T__enum_valueof`, and `H5T__enum_insert`
  with sorted lookup and duplicate name/value rejection,
  expanded `H5T_noop_conv` over the datatype registry's no-op path checks,
  replaced the `H5T__sort_value` and `H5T__sort_name` byte-copy placeholders
  with in-place compound/enum member sorting and map reordering,
  tightened `H5T_nameof`, `H5T_own_vol_obj`, and `H5T_path_match` around the
  runtime datatype state and owned VOL-object marker,
  and removed the duplicate unsupported precision placeholder from the support
  module. Current CCC status: raw
  `missing_in_rust` is empty; `H5A*`/`H5T*`/`H5O*`/`H5G*`/`H5P*` missing
  entries all remain at zero, total partial findings are 2457, and `H5O*`
  partial/stub findings are also at zero after the latest object pass. The
  remaining partial flags from this datatype batch are wrapper/runtime-scale
  mismatches rather than raw absence, including `H5Tis_variable_str` delegating
  to the exact internal variable-string predicate while the C public wrapper
  carries registration and identifier plumbing not mirrored by the crate
  runtime. Treat any future one-line helper aliases in these families as
  incomplete unless the underlying behavior is represented by a real Rust
  implementation or an explicit, documented unsupported boundary.
- [ ] Broaden writer-side modern chunk-index parity.
  New chunked datasets are still created with v1 B-tree indexes. Remaining
  work is creation and growth for fixed-array, extensible-array, and deeper
  v2-B-tree chunk indexes, plus the full modern-index mutation matrix.
- [ ] Broaden filter parity where practical.
  SZip remains unsupported unless a pure-Rust decoder is added. NBit and
  ScaleOffset have broad fixture coverage, but not the full libhdf5 parameter
  space.
- [ ] Broaden datatype-conversion parity if user-facing conversion scope
  expands.
  Current reads support the crate's shared numeric conversion path and
  recursive compound value extraction, but not the full libhdf5 `H5T__conv_*`
  engine, packing helpers, or conversion-path selection behavior.
- [ ] Broaden attribute mutation and iteration parity if the public writer/API
  surface expands.
  Compact attributes can be deleted and renamed in place. Writer-created dense
  attributes can be deleted and same-size renamed for the supported
  root-direct, depth-0 name-index layout. Remaining work includes indirect or
  filtered dense heaps, non-leaf name indexes, creation-order indexes, growing
  renames that require metadata repacking, and broader mutation APIs.

## P2: Compatibility Coverage

- [ ] Add broader explicit unsupported API stubs for low-level libhdf5 entry
  points that do not map to this crate's high-level API.
  Good candidates are additional `H5F*`, `H5FD*`, `H5VL*`, `H5ES*`, `H5PL*`,
  and `H5M*` surfaces where the useful behavior is a stable
  `Error::Unsupported` instead of silent absence or accidental fallback.
- [ ] Extend deterministic property/config parsing for unsupported VFDs.
  Add or broaden encode/decode, validation, and malformed-buffer tests for
  family, multi, splitter, log, onion, subfiling, HDFS, and ROS3/S3 settings
  while keeping actual driver I/O explicitly unsupported.
- [ ] Add a real SZip fixture if a pure-Rust decoder becomes available, or keep
  the explicit unsupported error surface pinned.
- [ ] Add additional generated malformed fixtures for NBit and ScaleOffset
  edge-parameter combinations beyond the current unit and parity coverage.
- [ ] Add broader h5dump/h5py round-trip coverage for every newly expanded
  writer feature before declaring it supported.
- [ ] Add deeper libhdf5 path-normalization coverage for soft-link traversal if
  link-resolution parity becomes a public compatibility goal.
- [ ] Add deeper fractal-heap growth and checksum-corruption coverage if dense
  storage mutation support expands beyond the current layouts.

## Explicitly Out Of Scope

These libhdf5 families still appear in raw missing-symbol reports, but they are
not translation targets for this crate:

- MPI / parallel HDF5 / distributed selection paths (`H5_mpi*`, `H5S__mpio*`,
  parallel `H5D*`). If local parallelism is added later, use Rayon in Rust.
- VOL, async, plugin, and connector infrastructure (`H5VL*`, `H5ES*`,
  `H5PL*`).
- Alternative VFD and cloud/network driver runtime parity, including HDFS,
  ROS3/S3, Direct, broad core/stdio behavior, onion, subfiling, family, multi,
  splitter, and log drivers beyond deterministic config parsing and explicit
  unsupported errors.
- Map API parity (`H5M*`).
- Broad package-init, free-list, and thread-runtime machinery (`H5FL*`,
  broad `H5TS*`) unless this crate grows a direct need for those surfaces.
- Full C-library metadata-cache runtime behavior, SWMR runtime behavior, file
  mounting, and public APIs that do not map to this crate's high-level API.
