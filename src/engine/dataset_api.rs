use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;

use crate::error::{Error, Result};
use crate::hl::selection::{Selection, SelectionType};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VirtualMapping {
    pub source_file: String,
    pub source_dataset: String,
    pub min_dims: Vec<u64>,
    pub max_dims: Vec<Option<u64>>,
    pub open: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VirtualLayout {
    pub mappings: Vec<VirtualMapping>,
    pub unlimited: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VirtualSpaceStatus {
    Valid,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VirtualMappingValidation<'a> {
    pub virtual_selection: &'a Selection,
    pub virtual_shape: &'a [u64],
    pub virtual_max_dims: &'a [u64],
    pub source_selection: &'a Selection,
    pub source_shape: &'a [u64],
    pub source_max_dims: &'a [u64],
    pub source_file_printf_substitutions: usize,
    pub source_dataset_printf_substitutions: usize,
    pub source_space_status: VirtualSpaceStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VirtualParsedName {
    pub segments: Vec<String>,
    pub static_strlen: usize,
    pub substitutions: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SingleChunkIndex {
    pub open: bool,
    pub space_allocated: bool,
    pub chunk_addr: Option<u64>,
    pub metadata_loaded: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CompactStorage {
    pub data: Vec<u8>,
    pub space_allocated: bool,
    pub dirty: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DatasetApi {
    pub name: Option<String>,
    pub extent: Vec<u64>,
    pub raw: Vec<u8>,
    pub virtual_layout: Option<VirtualLayout>,
}

/// Dataset operation: virtual check mapping pre.
#[allow(non_snake_case)]
pub fn H5D_virtual_check_mapping_pre(args: &VirtualMappingValidation<'_>) -> Result<()> {
    if args.virtual_selection.selection_type() == SelectionType::Points
        || args.source_selection.selection_type() == SelectionType::Points
    {
        return Err(Error::Unsupported(
            "point selections are not supported by libhdf5 VDS mapping creation checks".into(),
        ));
    }

    let virtual_unlimited = args
        .virtual_selection
        .select_unlim_dim(args.virtual_max_dims)
        .is_some();
    let source_unlimited = args
        .source_selection
        .select_unlim_dim(args.source_max_dims)
        .is_some();

    if virtual_unlimited {
        if source_unlimited {
            let virtual_non_unlim = args
                .virtual_selection
                .select_num_elem_non_unlim(args.virtual_shape, args.virtual_max_dims)?;
            let source_non_unlim = args
                .source_selection
                .select_num_elem_non_unlim(args.source_shape, args.source_max_dims)?;
            if virtual_non_unlim != source_non_unlim {
                return Err(Error::InvalidFormat(
                    "VDS virtual/source non-unlimited element counts differ".into(),
                ));
            }
        }
    } else if args.source_space_status != VirtualSpaceStatus::Invalid {
        let virtual_count = args
            .virtual_selection
            .selected_count(args.virtual_shape)
            .ok_or_else(|| Error::InvalidFormat("VDS virtual selection count overflow".into()))?;
        let source_count = args
            .source_selection
            .selected_count(args.source_shape)
            .ok_or_else(|| Error::InvalidFormat("VDS source selection count overflow".into()))?;
        if virtual_count != source_count {
            return Err(Error::InvalidFormat(
                "VDS virtual/source selected element counts differ".into(),
            ));
        }
    }

    Ok(())
}

/// Dataset operation: virtual check mapping post.
#[allow(non_snake_case)]
pub fn H5D_virtual_check_mapping_post(args: &VirtualMappingValidation<'_>) -> Result<()> {
    let virtual_unlimited = args
        .virtual_selection
        .select_unlim_dim(args.virtual_max_dims)
        .is_some();
    let source_unlimited = args
        .source_selection
        .select_unlim_dim(args.source_max_dims)
        .is_some();
    let printf_subs =
        args.source_file_printf_substitutions + args.source_dataset_printf_substitutions;

    if virtual_unlimited && !source_unlimited {
        if printf_subs == 0 {
            return Err(Error::InvalidFormat(
                "VDS unlimited virtual selection with limited source needs printf substitutions"
                    .into(),
            ));
        }
        if args.virtual_selection.selection_type() != SelectionType::Hyperslab {
            return Err(Error::InvalidFormat(
                "VDS printf mapping virtual selection must be hyperslab".into(),
            ));
        }
        if args.source_space_status != VirtualSpaceStatus::Invalid {
            let virtual_block_count = virtual_first_block_element_count(args.virtual_selection)?;
            let source_count = args
                .source_selection
                .selected_count(args.source_shape)
                .ok_or_else(|| {
                    Error::InvalidFormat("VDS source selection count overflow".into())
                })?;
            if virtual_block_count != source_count {
                return Err(Error::InvalidFormat(
                    "VDS virtual single-block/source selected element counts differ".into(),
                ));
            }
        }
    } else if printf_subs > 0 {
        return Err(Error::InvalidFormat(
            "VDS printf substitutions require unlimited virtual selection and limited source selection"
                .into(),
        ));
    }

    Ok(())
}

/// Dataset operation: virtual check min dims.
#[allow(non_snake_case)]
pub fn H5D_virtual_check_min_dims(mapping: &VirtualMapping, dims: &[u64]) -> bool {
    H5D_virtual_check_min_dims_checked(mapping, dims).is_ok()
}

/// Dataset operation: virtual check min dims checked.
#[allow(non_snake_case)]
pub fn H5D_virtual_check_min_dims_checked(mapping: &VirtualMapping, dims: &[u64]) -> Result<()> {
    if dims.len() != mapping.min_dims.len() {
        return Err(Error::InvalidFormat(format!(
            "VDS rank {} does not match min-dims rank {}",
            dims.len(),
            mapping.min_dims.len()
        )));
    }
    mapping
        .min_dims
        .iter()
        .zip(dims)
        .enumerate()
        .try_for_each(|(idx, (minimum, actual))| {
            if actual < minimum {
                Err(Error::InvalidFormat(format!(
                    "VDS dimension {idx} is smaller than required minimum"
                )))
            } else {
                Ok(())
            }
        })
}

/// Dataset operation: virtual parse source name.
#[allow(non_snake_case)]
pub fn H5D_virtual_parse_source_name(source_name: &str) -> Result<Option<VirtualParsedName>> {
    let bytes = source_name.as_bytes();
    let mut pos = 0usize;
    let mut segment = String::new();
    let mut segments = Vec::new();
    let mut substitutions = 0usize;
    let mut static_strlen = source_name.len();

    while let Some(relative_pct) = bytes[pos..].iter().position(|&byte| byte == b'%') {
        let pct = pos
            .checked_add(relative_pct)
            .ok_or_else(|| Error::InvalidFormat("VDS source-name offset overflow".into()))?;
        let spec = bytes.get(pct + 1).copied().ok_or_else(|| {
            Error::InvalidFormat("VDS source name has truncated format specifier".into())
        })?;
        if !source_name.is_char_boundary(pct) || !source_name.is_char_boundary(pct + 2) {
            return Err(Error::InvalidFormat(
                "VDS source name format specifier is not UTF-8 aligned".into(),
            ));
        }

        match spec {
            b'b' => {
                segment.push_str(&source_name[pos..pct]);
                segments.push(std::mem::take(&mut segment));
                substitutions = substitutions.checked_add(1).ok_or_else(|| {
                    Error::InvalidFormat("VDS source-name substitution count overflow".into())
                })?;
                static_strlen = static_strlen.checked_sub(2).ok_or_else(|| {
                    Error::InvalidFormat("VDS source-name static length underflow".into())
                })?;
            }
            b'%' => {
                segment.push_str(&source_name[pos..pct]);
                segment.push('%');
                static_strlen = static_strlen.checked_sub(1).ok_or_else(|| {
                    Error::InvalidFormat("VDS source-name static length underflow".into())
                })?;
            }
            other => {
                return Err(Error::InvalidFormat(format!(
                    "invalid VDS source-name format specifier %{specifier}",
                    specifier = other as char
                )));
            }
        }
        pos = pct
            .checked_add(2)
            .ok_or_else(|| Error::InvalidFormat("VDS source-name offset overflow".into()))?;
    }

    if substitutions == 0 && segment.is_empty() {
        return Ok(None);
    }
    segment.push_str(&source_name[pos..]);
    segments.push(segment);
    Ok(Some(VirtualParsedName {
        segments,
        static_strlen,
        substitutions,
    }))
}

/// Free a dataset's in-memory resources.
#[allow(non_snake_case)]
pub fn H5D_virtual_free_parsed_name(_parsed_name: Option<VirtualParsedName>) {}

/// Dataset operation: virtual build source name.
#[allow(non_snake_case)]
pub fn H5D__virtual_build_source_name_cow<'a>(
    source_name: &'a str,
    parsed_name: Option<&'a VirtualParsedName>,
    blockno: u64,
) -> Result<Cow<'a, str>> {
    let Some(parsed_name) = parsed_name else {
        return Ok(Cow::Borrowed(source_name));
    };
    if parsed_name.substitutions == 0 {
        return Ok(parsed_name
            .segments
            .first()
            .map(String::as_str)
            .map(Cow::Borrowed)
            .unwrap_or(Cow::Borrowed(source_name)));
    }

    let mut out = String::new();
    H5D__virtual_build_source_name_into(source_name, Some(parsed_name), blockno, &mut out)?;
    Ok(Cow::Owned(out))
}

/// Dataset operation: virtual build source name into caller-provided storage.
#[allow(non_snake_case)]
pub fn H5D__virtual_build_source_name_into(
    source_name: &str,
    parsed_name: Option<&VirtualParsedName>,
    blockno: u64,
    out: &mut String,
) -> Result<()> {
    let Some(parsed_name) = parsed_name else {
        out.clear();
        out.push_str(source_name);
        return Ok(());
    };
    if parsed_name.substitutions == 0 {
        out.clear();
        out.push_str(
            parsed_name
                .segments
                .first()
                .map(String::as_str)
                .unwrap_or(source_name),
        );
        return Ok(());
    }
    if parsed_name.segments.len() != parsed_name.substitutions + 1 {
        return Err(Error::InvalidFormat(
            "VDS parsed source-name segment count does not match substitutions".into(),
        ));
    }
    let block = blockno.to_string();
    let substitution_bytes = parsed_name
        .substitutions
        .checked_mul(block.len())
        .ok_or_else(|| Error::InvalidFormat("VDS built source-name size overflow".into()))?;
    let capacity = parsed_name
        .static_strlen
        .checked_add(substitution_bytes)
        .ok_or_else(|| Error::InvalidFormat("VDS built source-name size overflow".into()))?;
    out.clear();
    out.reserve(capacity.saturating_sub(out.capacity()));
    for (idx, segment) in parsed_name.segments.iter().enumerate() {
        out.push_str(segment);
        if idx < parsed_name.substitutions {
            out.push_str(&block);
        }
    }
    Ok(())
}

/// Dataset operation: virtual build source name.
#[allow(non_snake_case)]
#[deprecated(
    note = "use H5D__virtual_build_source_name_cow or H5D__virtual_build_source_name_into"
)]
pub fn H5D__virtual_build_source_name(
    source_name: &str,
    parsed_name: Option<&VirtualParsedName>,
    blockno: u64,
) -> Result<String> {
    H5D__virtual_build_source_name_cow(source_name, parsed_name, blockno).map(Cow::into_owned)
}

/// Dataset operation: virtual store layout.
#[allow(non_snake_case)]
pub fn H5D__virtual_store_layout(dataset: &mut DatasetApi, layout: VirtualLayout) {
    dataset.virtual_layout = Some(layout);
}

/// Dataset operation: virtual load layout.
#[allow(non_snake_case)]
pub fn H5D__virtual_load_layout_ref(dataset: &DatasetApi) -> Option<&VirtualLayout> {
    dataset.virtual_layout.as_ref()
}

/// Dataset operation: virtual load layout.
#[allow(non_snake_case)]
#[deprecated(note = "use H5D__virtual_load_layout_ref to borrow the stored layout")]
pub fn H5D__virtual_load_layout(dataset: &DatasetApi) -> Option<VirtualLayout> {
    H5D__virtual_load_layout_ref(dataset).cloned()
}

/// Return a deep copy of a dataset.
#[allow(non_snake_case)]
pub fn H5D__virtual_copy_layout(layout: &VirtualLayout) -> VirtualLayout {
    layout.clone()
}

/// Free a dataset's in-memory resources.
#[allow(non_snake_case)]
pub fn H5D__virtual_free_layout_mappings(layout: &mut VirtualLayout) {
    layout.mappings.clear();
}

/// Reset a dataset to its default state.
#[allow(non_snake_case)]
pub fn H5D__virtual_reset_layout(layout: &mut VirtualLayout) {
    layout.mappings.clear();
    layout.unlimited = false;
}

/// Open a dataset.
#[allow(non_snake_case)]
pub fn H5D__virtual_open_source_dset(mapping: &mut VirtualMapping) -> Result<()> {
    if mapping.source_file.is_empty() || mapping.source_dataset.is_empty() {
        return Err(Error::InvalidFormat(
            "virtual source mapping is incomplete".into(),
        ));
    }
    mapping.open = true;
    Ok(())
}

/// Dataset operation: virtual set extent unlim.
#[allow(non_snake_case)]
pub fn H5D__virtual_set_extent_unlim(layout: &mut VirtualLayout) {
    layout.unlimited = true;
}

/// Initialize the dataset subsystem.
#[allow(non_snake_case)]
pub fn H5D__virtual_init_all(layout: &mut VirtualLayout) -> Result<()> {
    for mapping in &mut layout.mappings {
        H5D__virtual_open_source_dset(mapping)?;
    }
    Ok(())
}

/// Dataset operation: virtual pre io process mapping.
#[allow(non_snake_case)]
pub fn H5D__virtual_pre_io_process_mapping(mapping: &VirtualMapping, dims: &[u64]) -> bool {
    mapping.open && H5D_virtual_check_min_dims(mapping, dims)
}

/// Flush the dataset to storage.
#[allow(non_snake_case)]
pub fn H5D__virtual_flush(_layout: &mut VirtualLayout) {}

/// Refresh the dataset from storage.
#[allow(non_snake_case)]
pub fn H5D__virtual_refresh_source_dset(mapping: &mut VirtualMapping) -> Result<()> {
    H5D__virtual_open_source_dset(mapping)
}

/// Refresh the dataset from storage.
#[allow(non_snake_case)]
pub fn H5D__virtual_refresh_source_dsets(layout: &mut VirtualLayout) -> Result<()> {
    H5D__virtual_init_all(layout)
}

/// Dataset operation: virtual release source dset files.
#[allow(non_snake_case)]
pub fn H5D__virtual_release_source_dset_files(layout: &mut VirtualLayout) {
    for mapping in &mut layout.mappings {
        mapping.open = false;
    }
}

/// Dataset operation: mappings to leaves.
#[allow(non_snake_case)]
pub fn H5D__mappings_to_leaves_iter(
    layout: &VirtualLayout,
) -> impl Iterator<Item = &VirtualMapping> + '_ {
    layout.mappings.iter()
}

/// Dataset operation: mappings to leaves.
#[allow(non_snake_case)]
#[deprecated(note = "use H5D__mappings_to_leaves_iter to avoid cloning mappings")]
pub fn H5D__mappings_to_leaves(layout: &VirtualLayout) -> Vec<VirtualMapping> {
    H5D__mappings_to_leaves_iter(layout).cloned().collect()
}

/// Dataset operation: virtual not in tree grow.
#[allow(non_snake_case)]
pub fn H5D__virtual_not_in_tree_grow(layout: &mut VirtualLayout, mapping: VirtualMapping) {
    if !layout.mappings.contains(&mapping) {
        layout.mappings.push(mapping);
    }
}

/// Dataset operation: should build tree.
#[allow(non_snake_case)]
pub fn H5D__should_build_tree(layout: &VirtualLayout) -> bool {
    layout.mappings.len() > 1
}

/// Close a dataset.
#[allow(non_snake_case)]
pub fn H5D__virtual_close_mapping(mapping: &mut VirtualMapping) {
    mapping.open = false;
}

/// Initialize the dataset subsystem.
#[allow(non_snake_case)]
pub fn H5D__single_idx_init() -> SingleChunkIndex {
    SingleChunkIndex::default()
}

/// Create a new dataset.
#[allow(non_snake_case)]
pub fn H5D__single_idx_create(index: &mut SingleChunkIndex, addr: u64) {
    index.open = true;
    index.space_allocated = true;
    index.chunk_addr = Some(addr);
}

/// Close a dataset.
#[allow(non_snake_case)]
pub fn H5D__single_idx_close(index: &mut SingleChunkIndex) {
    index.open = false;
}

/// Open a dataset.
#[allow(non_snake_case)]
pub fn H5D__single_idx_is_open(index: &SingleChunkIndex) -> bool {
    index.open
}

/// Allocate storage for a dataset.
#[allow(non_snake_case)]
pub fn H5D__single_idx_is_space_alloc(index: &SingleChunkIndex) -> bool {
    index.space_allocated
}

/// Insert an entry into a dataset.
#[allow(non_snake_case)]
pub fn H5D__single_idx_insert(index: &mut SingleChunkIndex, addr: u64) {
    H5D__single_idx_create(index, addr);
}

/// Dataset operation: single idx get addr.
#[allow(non_snake_case)]
pub fn H5D__single_idx_get_addr(index: &SingleChunkIndex) -> Option<u64> {
    index.chunk_addr
}

/// Dataset operation: single idx load metadata.
#[allow(non_snake_case)]
pub fn H5D__single_idx_load_metadata(index: &mut SingleChunkIndex) {
    index.metadata_loaded = true;
}

/// Iterate over the entries of a dataset.
#[allow(non_snake_case)]
pub fn H5D__single_idx_iterate(index: &SingleChunkIndex) -> impl Iterator<Item = u64> + '_ {
    index.chunk_addr.into_iter()
}

/// Remove an entry from a dataset.
#[allow(non_snake_case)]
pub fn H5D__single_idx_remove(index: &mut SingleChunkIndex) -> Option<u64> {
    index.space_allocated = false;
    index.chunk_addr.take()
}

/// Delete a dataset.
#[allow(non_snake_case)]
pub fn H5D__single_idx_delete(index: &mut SingleChunkIndex) {
    *index = SingleChunkIndex::default();
}

/// Return a deep copy of a dataset.
#[allow(non_snake_case)]
#[deprecated(note = "use H5D__single_idx_copy_setup_into to reuse caller-provided storage")]
pub fn H5D__single_idx_copy_setup(index: &SingleChunkIndex) -> SingleChunkIndex {
    index.clone()
}

/// Copy a single-chunk index into caller-provided storage.
#[allow(non_snake_case)]
pub fn H5D__single_idx_copy_setup_into(index: &SingleChunkIndex, out: &mut SingleChunkIndex) {
    out.clone_from(index);
}

/// Reset a dataset to its default state.
#[allow(non_snake_case)]
pub fn H5D__single_idx_reset(index: &mut SingleChunkIndex) {
    H5D__single_idx_delete(index);
}

/// Render a single-chunk index for debug output into a formatter.
#[allow(non_snake_case)]
pub fn H5D__single_idx_dump_to<W: fmt::Write>(
    index: &SingleChunkIndex,
    out: &mut W,
) -> fmt::Result {
    write!(
        out,
        "single_idx(open={}, allocated={}, addr={:?})",
        index.open, index.space_allocated, index.chunk_addr
    )
}

/// Render a dataset for debug output.
#[allow(non_snake_case)]
#[deprecated(note = "use H5D__single_idx_dump_to to write into caller-provided output")]
pub fn H5D__single_idx_dump(index: &SingleChunkIndex) -> String {
    let mut out = String::new();
    let _ = H5D__single_idx_dump_to(index, &mut out);
    out
}

/// Create a new dataset.
#[allow(non_snake_case)]
pub fn H5D__create_api_common(name: Option<String>, extent: Vec<u64>) -> DatasetApi {
    DatasetApi {
        name,
        extent,
        raw: Vec::new(),
        virtual_layout: None,
    }
}

/// Create a new dataset.
#[allow(non_snake_case)]
pub fn H5Dcreate_anon(extent: Vec<u64>) -> DatasetApi {
    H5D__create_api_common(None, extent)
}

/// Open a dataset.
#[allow(non_snake_case)]
pub fn H5D__open_api_common(dataset: &DatasetApi) -> DatasetApi {
    dataset.clone()
}

/// Close a dataset.
#[allow(non_snake_case)]
pub fn H5Dclose(_dataset: DatasetApi) {}

/// Dataset operation: get space api common.
#[allow(non_snake_case)]
pub fn H5D__get_space_api_common(dataset: &DatasetApi) -> &[u64] {
    &dataset.extent
}

/// Read from a dataset.
#[allow(non_snake_case)]
#[deprecated(note = "use H5Dread_multi_with or H5Dread_multi_into to avoid cloning raw buffers")]
pub fn H5Dread_multi(datasets: &[DatasetApi]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    H5Dread_multi_into(datasets, &mut out);
    out
}

/// Visit each dataset raw buffer without cloning it.
#[allow(non_snake_case)]
pub fn H5Dread_multi_with<F>(datasets: &[DatasetApi], mut visitor: F)
where
    F: FnMut(&[u8]),
{
    for dataset in datasets {
        visitor(dataset.raw.as_slice());
    }
}

/// Copy dataset raw buffers into caller-provided storage.
#[allow(non_snake_case)]
pub fn H5Dread_multi_into(datasets: &[DatasetApi], out: &mut Vec<Vec<u8>>) {
    out.truncate(datasets.len());
    out.resize_with(datasets.len(), Vec::new);
    for (dst, dataset) in out.iter_mut().zip(datasets) {
        dst.clear();
        dst.extend_from_slice(&dataset.raw);
    }
}

/// Read from a dataset.
#[allow(non_snake_case)]
pub fn H5Dread_multi_async(_datasets: &[DatasetApi]) -> Result<()> {
    Err(Error::Unsupported(
        "async dataset reads require event-set infrastructure".into(),
    ))
}

/// Read from a dataset.
#[allow(non_snake_case)]
#[deprecated(note = "use H5Dread_chunk2_slice or H5Dread_chunk2_into to avoid allocating")]
pub fn H5Dread_chunk2(dataset: &DatasetApi, offset: usize, len: usize) -> Result<Vec<u8>> {
    Ok(H5Dread_chunk2_slice(dataset, offset, len)?.to_vec())
}

/// Borrow a raw chunk byte range from a dataset.
#[allow(non_snake_case)]
pub fn H5Dread_chunk2_slice(dataset: &DatasetApi, offset: usize, len: usize) -> Result<&[u8]> {
    let end = offset
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat("dataset chunk read overflow".into()))?;
    dataset
        .raw
        .get(offset..end)
        .ok_or_else(|| Error::InvalidFormat("dataset chunk read out of bounds".into()))
}

/// Copy a raw chunk byte range into caller-provided storage.
#[allow(non_snake_case)]
pub fn H5Dread_chunk2_into(dataset: &DatasetApi, offset: usize, out: &mut [u8]) -> Result<()> {
    out.copy_from_slice(H5Dread_chunk2_slice(dataset, offset, out.len())?);
    Ok(())
}

/// Write to a dataset.
#[allow(non_snake_case)]
pub fn H5D__write_api_common(dataset: &mut DatasetApi, data: &[u8]) {
    dataset.raw.clear();
    dataset.raw.extend_from_slice(data);
}

/// Write to a dataset.
#[allow(non_snake_case)]
pub fn H5Dwrite_multi(datasets: &mut [DatasetApi], payloads: &[Vec<u8>]) {
    H5Dwrite_multi_slices(datasets, payloads.iter().map(Vec::as_slice));
}

/// Write borrowed payloads to multiple datasets.
#[allow(non_snake_case)]
pub fn H5Dwrite_multi_slices<'a, I>(datasets: &mut [DatasetApi], payloads: I)
where
    I: IntoIterator<Item = &'a [u8]>,
{
    for (dataset, payload) in datasets.iter_mut().zip(payloads) {
        H5D__write_api_common(dataset, payload);
    }
}

/// Write to a dataset.
#[allow(non_snake_case)]
pub fn H5Dwrite_multi_async(_datasets: &mut [DatasetApi], _payloads: &[Vec<u8>]) -> Result<()> {
    Err(Error::Unsupported(
        "async dataset writes require event-set infrastructure".into(),
    ))
}

/// Flush the dataset to storage.
#[allow(non_snake_case)]
pub fn H5Dflush(_dataset: &mut DatasetApi) {}

/// Refresh the dataset from storage.
#[allow(non_snake_case)]
pub fn H5Drefresh(_dataset: &mut DatasetApi) {}

/// Convert a dataset.
#[allow(non_snake_case)]
pub fn H5Dformat_convert(dataset: &mut DatasetApi) {
    dataset.raw.shrink_to_fit();
}

/// Dataset operation: chunk iter.
#[allow(non_snake_case)]
pub fn H5Dchunk_iter(dataset: &DatasetApi, chunk_size: usize) -> impl Iterator<Item = &[u8]> {
    dataset.raw.chunks(chunk_size.max(1))
}

/// Dataset operation: compact construct.
#[allow(non_snake_case)]
pub fn H5D__compact_construct(data: Vec<u8>) -> CompactStorage {
    CompactStorage {
        space_allocated: true,
        dirty: false,
        data,
    }
}

/// Allocate storage for a dataset.
#[allow(non_snake_case)]
pub fn H5D__compact_is_space_alloc(storage: &CompactStorage) -> bool {
    storage.space_allocated
}

/// Initialize the dataset subsystem.
#[allow(non_snake_case)]
pub fn H5D__compact_io_init(storage: &mut CompactStorage) {
    storage.space_allocated = true;
}

/// Dataset operation: compact iovv memmanage cb.
#[allow(non_snake_case)]
pub fn H5D__compact_iovv_memmanage_cb(storage: &CompactStorage) -> usize {
    storage.data.len()
}

/// Dataset operation: compact readvv.
#[allow(non_snake_case)]
#[deprecated(note = "use H5D__compact_readvv_slice or H5D__compact_readvv_into")]
pub fn H5D__compact_readvv(storage: &CompactStorage, offset: usize, len: usize) -> Result<Vec<u8>> {
    Ok(H5D__compact_readvv_slice(storage, offset, len)?.to_vec())
}

/// Dataset operation: compact readvv borrowed slice.
#[allow(non_snake_case)]
pub fn H5D__compact_readvv_slice(
    storage: &CompactStorage,
    offset: usize,
    len: usize,
) -> Result<&[u8]> {
    let end = offset
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat("compact read overflow".into()))?;
    storage
        .data
        .get(offset..end)
        .ok_or_else(|| Error::InvalidFormat("compact read out of bounds".into()))
}

/// Dataset operation: compact readvv into caller storage.
#[allow(non_snake_case)]
pub fn H5D__compact_readvv_into(
    storage: &CompactStorage,
    offset: usize,
    out: &mut [u8],
) -> Result<()> {
    out.copy_from_slice(H5D__compact_readvv_slice(storage, offset, out.len())?);
    Ok(())
}

/// Dataset operation: compact writevv.
#[allow(non_snake_case)]
pub fn H5D__compact_writevv(
    storage: &mut CompactStorage,
    offset: usize,
    data: &[u8],
) -> Result<()> {
    let end = offset
        .checked_add(data.len())
        .ok_or_else(|| Error::InvalidFormat("compact write overflow".into()))?;
    if storage.data.len() < end {
        storage.data.resize(end, 0);
    }
    storage.data[offset..end].copy_from_slice(data);
    storage.space_allocated = true;
    storage.dirty = true;
    Ok(())
}

/// Flush the dataset to storage.
#[allow(non_snake_case)]
pub fn H5D__compact_flush(storage: &mut CompactStorage) {
    storage.dirty = false;
}

/// Dataset operation: compact dest.
#[allow(non_snake_case)]
pub fn H5D__compact_dest(storage: &mut CompactStorage) {
    storage.data.clear();
    storage.space_allocated = false;
    storage.dirty = false;
}

/// Return a deep copy of a dataset.
#[allow(non_snake_case)]
#[deprecated(note = "use H5D__compact_copy_into to reuse caller-provided storage")]
pub fn H5D__compact_copy(storage: &CompactStorage) -> CompactStorage {
    storage.clone()
}

/// Copy compact storage into caller-provided storage.
#[allow(non_snake_case)]
pub fn H5D__compact_copy_into(storage: &CompactStorage, out: &mut CompactStorage) {
    out.clone_from(storage);
}

/// Dataset operation: get chunk storage size.
#[allow(non_snake_case)]
pub fn H5D__get_chunk_storage_size(
    chunks: &HashMap<Vec<u64>, Vec<u8>>,
    coord: &[u64],
) -> Option<usize> {
    chunks.get(coord).map(Vec::len)
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChunkInfo {
    pub coord: Vec<u64>,
    pub addr: u64,
    pub size: usize,
    pub filter_mask: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChunkTable {
    pub chunk_dims: Vec<u64>,
    pub chunks: HashMap<Vec<u64>, ChunkInfo>,
    pub data: HashMap<Vec<u64>, Vec<u8>>,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub locked: bool,
}

/// Dataset operation: chunk set info real.
#[allow(non_snake_case)]
pub fn H5D__chunk_set_info_real(info: &mut ChunkInfo, addr: u64, size: usize, filter_mask: u32) {
    info.addr = addr;
    info.size = size;
    info.filter_mask = filter_mask;
}

/// Dataset operation: chunk set info.
#[allow(non_snake_case)]
pub fn H5D__chunk_set_info(info: &mut ChunkInfo, addr: u64, size: usize) {
    H5D__chunk_set_info_real(info, addr, size, info.filter_mask);
}

/// Dataset operation: chunk set sizes.
#[allow(non_snake_case)]
pub fn H5D__chunk_set_sizes(table: &mut ChunkTable, chunk_dims: Vec<u64>) {
    table.chunk_dims = chunk_dims;
}

/// Dataset operation: chunk construct.
#[allow(non_snake_case)]
pub fn H5D__chunk_construct(chunk_dims: Vec<u64>) -> ChunkTable {
    ChunkTable {
        chunk_dims,
        ..ChunkTable::default()
    }
}

/// Initialize the dataset subsystem.
#[allow(non_snake_case)]
pub fn H5D__chunk_io_init(_table: &mut ChunkTable) {}

/// Initialize the dataset subsystem.
#[allow(non_snake_case)]
pub fn H5D__chunk_io_init_selections_iter(
    coords: &[Vec<u64>],
) -> impl Iterator<Item = &[u64]> + '_ {
    coords.iter().map(Vec::as_slice)
}

/// Initialize the dataset subsystem.
#[allow(non_snake_case)]
#[deprecated(note = "use H5D__chunk_io_init_selections_iter to borrow coordinates")]
pub fn H5D__chunk_io_init_selections(coords: &[Vec<u64>]) -> Vec<Vec<u64>> {
    H5D__chunk_io_init_selections_iter(coords)
        .map(<[u64]>::to_vec)
        .collect()
}

/// Allocate storage for a dataset.
#[allow(non_snake_case)]
pub fn H5D__chunk_mem_alloc(size: usize) -> Vec<u8> {
    vec![0; size]
}

/// Dataset operation: chunk mem realloc.
#[allow(non_snake_case)]
pub fn H5D__chunk_mem_realloc(mut buf: Vec<u8>, size: usize) -> Vec<u8> {
    buf.resize(size, 0);
    buf
}

/// Free a dataset's in-memory resources.
#[allow(non_snake_case)]
pub fn H5D__free_piece_info(info: &mut ChunkInfo) {
    *info = ChunkInfo::default();
}

/// Create a new dataset.
#[allow(non_snake_case)]
pub fn H5D__create_piece_map_single(coord: Vec<u64>, data: Vec<u8>) -> HashMap<Vec<u64>, Vec<u8>> {
    HashMap::from([(coord, data)])
}

/// Create a new dataset.
#[allow(non_snake_case)]
pub fn H5D__create_piece_file_map_all_with<F>(table: &ChunkTable, mut visitor: F)
where
    F: FnMut(&[u64], u64),
{
    for (coord, info) in &table.chunks {
        visitor(coord, info.addr);
    }
}

/// Create a new dataset.
#[allow(non_snake_case)]
#[deprecated(note = "use H5D__create_piece_file_map_all_with to avoid cloning coordinates")]
pub fn H5D__create_piece_file_map_all(table: &ChunkTable) -> HashMap<Vec<u64>, u64> {
    let mut out = HashMap::with_capacity(table.chunks.len());
    H5D__create_piece_file_map_all_with(table, |coord, addr| {
        out.insert(coord.to_vec(), addr);
    });
    out
}

/// Create a new dataset.
#[allow(non_snake_case)]
pub fn H5D__create_piece_file_map_hyper_with<F>(
    table: &ChunkTable,
    coords: &[Vec<u64>],
    mut visitor: F,
) where
    F: FnMut(&[u64], u64),
{
    H5D__create_piece_file_map_hyper_slices_with(
        table,
        coords.iter().map(Vec::as_slice),
        |coord, addr| {
            visitor(coord, addr);
        },
    );
}

/// Create a new dataset from borrowed chunk coordinate slices.
#[allow(non_snake_case)]
pub fn H5D__create_piece_file_map_hyper_slices_with<'a, I, F>(
    table: &ChunkTable,
    coords: I,
    mut visitor: F,
) where
    I: IntoIterator<Item = &'a [u64]>,
    F: FnMut(&[u64], u64),
{
    for coord in coords {
        if let Some(info) = table.chunks.get(coord) {
            visitor(coord, info.addr);
        }
    }
}

/// Create a new dataset.
#[allow(non_snake_case)]
#[deprecated(note = "use H5D__create_piece_file_map_hyper_with to avoid cloning coordinates")]
pub fn H5D__create_piece_file_map_hyper(
    table: &ChunkTable,
    coords: &[Vec<u64>],
) -> HashMap<Vec<u64>, u64> {
    let mut out = HashMap::with_capacity(coords.len());
    H5D__create_piece_file_map_hyper_with(table, coords, |coord, addr| {
        out.insert(coord.to_vec(), addr);
    });
    out
}

/// Create a new dataset.
#[allow(non_snake_case)]
pub fn H5D__create_piece_mem_map_hyper_with<F>(
    table: &ChunkTable,
    coords: &[Vec<u64>],
    mut visitor: F,
) where
    F: FnMut(&[u64], &[u8]),
{
    H5D__create_piece_mem_map_hyper_slices_with(
        table,
        coords.iter().map(Vec::as_slice),
        |coord, data| {
            visitor(coord, data);
        },
    );
}

/// Create a new dataset from borrowed chunk coordinate slices.
#[allow(non_snake_case)]
pub fn H5D__create_piece_mem_map_hyper_slices_with<'a, I, F>(
    table: &ChunkTable,
    coords: I,
    mut visitor: F,
) where
    I: IntoIterator<Item = &'a [u64]>,
    F: FnMut(&[u64], &[u8]),
{
    for coord in coords {
        if let Some(data) = table.data.get(coord) {
            visitor(coord, data);
        }
    }
}

/// Create a new dataset.
#[allow(non_snake_case)]
#[deprecated(note = "use H5D__create_piece_mem_map_hyper_with to avoid cloning chunks")]
pub fn H5D__create_piece_mem_map_hyper(
    table: &ChunkTable,
    coords: &[Vec<u64>],
) -> HashMap<Vec<u64>, Vec<u8>> {
    let mut out = HashMap::with_capacity(coords.len());
    H5D__create_piece_mem_map_hyper_with(table, coords, |coord, data| {
        out.insert(coord.to_vec(), data.to_vec());
    });
    out
}

/// Dataset operation: piece file cb.
#[allow(non_snake_case)]
pub fn H5D__piece_file_cb(info: &ChunkInfo) -> u64 {
    info.addr
}

/// Dataset operation: piece mem cb.
#[allow(non_snake_case)]
pub fn H5D__piece_mem_cb(data: &[u8]) -> usize {
    data.len()
}

/// Initialize the dataset subsystem.
#[allow(non_snake_case)]
pub fn H5D__chunk_mdio_init(_table: &mut ChunkTable) {}

/// Dataset operation: chunk may use select io.
#[allow(non_snake_case)]
pub fn H5D__chunk_may_use_select_io(coords: &[Vec<u64>]) -> bool {
    coords.len() > 1
}

/// Read from a dataset.
#[allow(non_snake_case)]
pub fn H5D__chunk_read_slice<'a>(table: &'a mut ChunkTable, coord: &[u64]) -> Result<&'a [u8]> {
    if let Some(data) = table.data.get(coord) {
        table.cache_hits += 1;
        return Ok(data);
    }
    table.cache_misses += 1;
    Err(Error::InvalidFormat("chunk not found".into()))
}

/// Read from a dataset.
#[allow(non_snake_case)]
pub fn H5D__chunk_read_into(table: &mut ChunkTable, coord: &[u64], out: &mut [u8]) -> Result<()> {
    let data = H5D__chunk_read_slice(table, coord)?;
    if out.len() != data.len() {
        return Err(Error::InvalidFormat(
            "chunk read output size mismatch".into(),
        ));
    }
    out.copy_from_slice(data);
    Ok(())
}

/// Read from a dataset.
#[allow(non_snake_case)]
#[deprecated(note = "use H5D__chunk_read_slice or H5D__chunk_read_into to avoid cloning chunks")]
pub fn H5D__chunk_read(table: &mut ChunkTable, coord: &[u64]) -> Result<Vec<u8>> {
    Ok(H5D__chunk_read_slice(table, coord)?.to_vec())
}

/// Write to a dataset.
#[allow(non_snake_case)]
pub fn H5D__chunk_write(table: &mut ChunkTable, coord: Vec<u64>, data: Vec<u8>) {
    let _ = H5D__chunk_write_checked(table, coord, data);
}

/// Write to a dataset.
#[allow(non_snake_case)]
pub fn H5D__chunk_write_checked(
    table: &mut ChunkTable,
    coord: Vec<u64>,
    data: Vec<u8>,
) -> Result<()> {
    let addr = u64::try_from(table.chunks.len())
        .map_err(|_| Error::InvalidFormat("chunk table address exceeds u64".into()))?;
    let info = ChunkInfo {
        coord: coord.clone(),
        addr,
        size: data.len(),
        filter_mask: 0,
    };
    table.chunks.insert(coord.clone(), info);
    table.data.insert(coord, data);
    Ok(())
}

/// Flush the dataset to storage.
#[allow(non_snake_case)]
pub fn H5D__chunk_flush(_table: &mut ChunkTable) {}

/// Dataset operation: chunk io term.
#[allow(non_snake_case)]
pub fn H5D__chunk_io_term(_table: &mut ChunkTable) {}

/// Dataset operation: chunk dest.
#[allow(non_snake_case)]
pub fn H5D__chunk_dest(table: &mut ChunkTable) {
    table.chunks.clear();
    table.data.clear();
}

/// Reset a dataset to its default state.
#[allow(non_snake_case)]
pub fn H5D_chunk_idx_reset(table: &mut ChunkTable) {
    H5D__chunk_dest(table);
}

/// Reset a dataset to its default state.
#[allow(non_snake_case)]
pub fn H5D__chunk_cinfo_cache_reset(table: &mut ChunkTable) {
    table.cache_hits = 0;
    table.cache_misses = 0;
}

/// Update a dataset.
#[allow(non_snake_case)]
pub fn H5D__chunk_cinfo_cache_update(table: &mut ChunkTable, hit: bool) {
    if hit {
        table.cache_hits += 1;
    } else {
        table.cache_misses += 1;
    }
}

/// Dataset operation: chunk cinfo cache found.
#[allow(non_snake_case)]
pub fn H5D__chunk_cinfo_cache_found(table: &mut ChunkTable) -> bool {
    let found = table.cache_hits > 0;
    H5D__chunk_cinfo_cache_update(table, found);
    found
}

/// Create a new dataset.
#[allow(non_snake_case)]
pub fn H5D__chunk_create(chunk_dims: Vec<u64>) -> ChunkTable {
    H5D__chunk_construct(chunk_dims)
}

/// Dataset operation: chunk hash val.
#[allow(non_snake_case)]
pub fn H5D__chunk_hash_val(coord: &[u64]) -> u64 {
    coord.iter().fold(1469598103934665603, |hash, value| {
        (hash ^ value).wrapping_mul(1099511628211)
    })
}

/// Look up a dataset entry.
#[allow(non_snake_case)]
pub fn H5D__chunk_lookup<'a>(table: &'a ChunkTable, coord: &[u64]) -> Option<&'a ChunkInfo> {
    table.chunks.get(coord)
}

/// Flush the dataset to storage.
#[allow(non_snake_case)]
pub fn H5D__chunk_flush_entry(_table: &mut ChunkTable, _coord: &[u64]) {}

/// Dataset operation: chunk cache evict.
#[allow(non_snake_case)]
pub fn H5D__chunk_cache_evict(table: &mut ChunkTable, coord: &[u64]) -> Option<Vec<u8>> {
    table.data.remove(coord)
}

/// Dataset operation: chunk cache prune.
#[allow(non_snake_case)]
pub fn H5D__chunk_cache_prune(table: &mut ChunkTable, max_chunks: usize) {
    if table.data.len() <= max_chunks {
        return;
    }

    let mut keep = max_chunks;
    table.data.retain(|_, _| {
        let retain = keep > 0;
        keep = keep.saturating_sub(1);
        retain
    });
}

/// Lock a dataset against further modification.
#[allow(non_snake_case)]
pub fn H5D__chunk_lock(table: &mut ChunkTable) {
    table.locked = true;
}

/// Unlock a dataset.
#[allow(non_snake_case)]
pub fn H5D__chunk_unlock(table: &mut ChunkTable) {
    table.locked = false;
}

/// Dataset operation: chunk allocated.
#[allow(non_snake_case)]
pub fn H5D__chunk_allocated(table: &ChunkTable, coord: &[u64]) -> bool {
    table.chunks.contains_key(coord)
}

/// Dataset operation: chunk allocate.
#[allow(non_snake_case)]
pub fn H5D__chunk_allocate(table: &mut ChunkTable, coord: Vec<u64>, size: usize) {
    H5D__chunk_write(table, coord, vec![0; size]);
}

/// Dataset operation: chunk allocate checked.
#[allow(non_snake_case)]
pub fn H5D__chunk_allocate_checked(
    table: &mut ChunkTable,
    coord: Vec<u64>,
    size: usize,
) -> Result<()> {
    H5D__chunk_write_checked(table, coord, vec![0; size])
}

/// Update a dataset.
#[allow(non_snake_case)]
pub fn H5D__chunk_update_old_edge_chunks(_table: &mut ChunkTable) {}

/// Dataset operation: chunk cmp coll fill info.
#[allow(non_snake_case)]
pub fn H5D__chunk_cmp_coll_fill_info(left: &ChunkInfo, right: &ChunkInfo) -> std::cmp::Ordering {
    left.coord
        .cmp(&right.coord)
        .then_with(|| left.addr.cmp(&right.addr))
}

/// Dataset operation: chunk prune fill.
#[allow(non_snake_case)]
pub fn H5D__chunk_prune_fill(table: &mut ChunkTable) {
    table
        .data
        .retain(|_, data| data.iter().any(|byte| *byte != 0));
}

/// Dataset operation: chunk prune by extent.
#[allow(non_snake_case)]
pub fn H5D__chunk_prune_by_extent(table: &mut ChunkTable, extent: &[u64]) {
    table
        .data
        .retain(|coord, _| coord.iter().zip(extent).all(|(c, e)| c < e));
    table
        .chunks
        .retain(|coord, _| coord.iter().zip(extent).all(|(c, e)| c < e));
}

/// Dataset operation: chunk addrmap cb.
#[allow(non_snake_case)]
pub fn H5D__chunk_addrmap_cb_ref(info: &ChunkInfo) -> (&[u64], u64) {
    (&info.coord, info.addr)
}

/// Dataset operation: chunk addrmap cb.
#[allow(non_snake_case)]
#[deprecated(note = "use H5D__chunk_addrmap_cb_ref to borrow the coordinate")]
pub fn H5D__chunk_addrmap_cb(info: &ChunkInfo) -> (Vec<u64>, u64) {
    (info.coord.clone(), info.addr)
}

/// Dataset operation: chunk addrmap.
#[allow(non_snake_case)]
pub fn H5D__chunk_addrmap_with<F>(table: &ChunkTable, visitor: F)
where
    F: FnMut(&[u64], u64),
{
    H5D__create_piece_file_map_all_with(table, visitor);
}

/// Dataset operation: chunk addrmap.
#[allow(non_snake_case)]
#[deprecated(note = "use H5D__chunk_addrmap_with to avoid cloning coordinates")]
pub fn H5D__chunk_addrmap(table: &ChunkTable) -> HashMap<Vec<u64>, u64> {
    let mut out = HashMap::with_capacity(table.chunks.len());
    H5D__chunk_addrmap_with(table, |coord, addr| {
        out.insert(coord.to_vec(), addr);
    });
    out
}

/// Delete a dataset.
#[allow(non_snake_case)]
pub fn H5D__chunk_delete(table: &mut ChunkTable, coord: &[u64]) {
    table.chunks.remove(coord);
    table.data.remove(coord);
}

/// Update a dataset.
#[allow(non_snake_case)]
pub fn H5D__chunk_update_cache(table: &mut ChunkTable, coord: Vec<u64>, data: Vec<u8>) {
    table.data.insert(coord, data);
}

/// Return a deep copy of a dataset.
#[allow(non_snake_case)]
#[deprecated(note = "use H5D__chunk_copy_cb_ref to borrow the chunk info")]
pub fn H5D__chunk_copy_cb(info: &ChunkInfo) -> ChunkInfo {
    info.clone()
}

/// Borrow chunk information during copy callbacks.
#[allow(non_snake_case)]
pub fn H5D__chunk_copy_cb_ref(info: &ChunkInfo) -> &ChunkInfo {
    info
}

/// Return a deep copy of a dataset.
#[allow(non_snake_case)]
#[deprecated(note = "use H5D__chunk_copy_into to reuse caller-provided storage")]
pub fn H5D__chunk_copy(table: &ChunkTable) -> ChunkTable {
    table.clone()
}

/// Copy chunk table metadata into caller-provided storage.
#[allow(non_snake_case)]
pub fn H5D__chunk_copy_into(table: &ChunkTable, out: &mut ChunkTable) {
    out.clone_from(table);
}

/// Dataset operation: chunk stats.
#[allow(non_snake_case)]
pub fn H5D__chunk_stats(table: &ChunkTable) -> (usize, u64, u64) {
    (table.chunks.len(), table.cache_hits, table.cache_misses)
}

/// Fill caller-provided storage for a nonexistent chunk read callback.
#[allow(non_snake_case)]
pub fn H5D__nonexistent_readvv_cb_into(out: &mut [u8]) {
    out.fill(0);
}

/// Dataset operation: nonexistent readvv cb.
#[allow(non_snake_case)]
#[deprecated(note = "use H5D__nonexistent_readvv_cb_into to fill caller-provided buffers")]
pub fn H5D__nonexistent_readvv_cb(len: usize) -> Vec<u8> {
    vec![0; len]
}

/// Fill caller-provided storage for a nonexistent chunk read.
#[allow(non_snake_case)]
pub fn H5D__nonexistent_readvv_into(out: &mut [u8]) {
    H5D__nonexistent_readvv_cb_into(out);
}

/// Dataset operation: nonexistent readvv.
#[allow(non_snake_case)]
#[deprecated(note = "use H5D__nonexistent_readvv_into to fill caller-provided buffers")]
pub fn H5D__nonexistent_readvv(len: usize) -> Vec<u8> {
    vec![0; len]
}

/// Allocate storage for a dataset.
#[allow(non_snake_case)]
pub fn H5D__chunk_file_alloc(table: &mut ChunkTable, coord: Vec<u64>, size: usize) -> u64 {
    H5D__chunk_file_alloc_checked(table, coord, size).unwrap_or(u64::MAX)
}

/// Allocate storage for a dataset.
#[allow(non_snake_case)]
pub fn H5D__chunk_file_alloc_checked(
    table: &mut ChunkTable,
    coord: Vec<u64>,
    size: usize,
) -> Result<u64> {
    let addr = u64::try_from(table.chunks.len())
        .map_err(|_| Error::InvalidFormat("chunk table address exceeds u64".into()))?;
    let info = ChunkInfo {
        coord: coord.clone(),
        addr,
        size,
        filter_mask: 0,
    };
    table.chunks.insert(coord, info);
    Ok(addr)
}

/// Convert a dataset.
#[allow(non_snake_case)]
pub fn H5D__chunk_format_convert_cb_slice(data: &[u8]) -> &[u8] {
    data
}

/// Convert a dataset.
#[allow(non_snake_case)]
#[deprecated(note = "use H5D__chunk_format_convert_cb_slice to borrow converted data")]
pub fn H5D__chunk_format_convert_cb(data: &[u8]) -> Vec<u8> {
    H5D__chunk_format_convert_cb_slice(data).to_vec()
}

/// Convert a dataset.
#[allow(non_snake_case)]
pub fn H5D__chunk_format_convert(table: &mut ChunkTable) {
    for data in table.data.values_mut() {
        data.shrink_to_fit();
    }
}

/// Dataset operation: chunk index empty cb.
#[allow(non_snake_case)]
pub fn H5D__chunk_index_empty_cb(table: &ChunkTable) -> bool {
    table.chunks.is_empty()
}

/// Dataset operation: get num chunks cb.
#[allow(non_snake_case)]
pub fn H5D__get_num_chunks_cb(table: &ChunkTable) -> usize {
    table.chunks.len()
}

/// Dataset operation: get num chunks.
#[allow(non_snake_case)]
pub fn H5D__get_num_chunks(table: &ChunkTable) -> usize {
    H5D__get_num_chunks_cb(table)
}

/// Dataset operation: get chunk info cb.
#[allow(non_snake_case)]
pub fn H5D__get_chunk_info_cb<'a>(table: &'a ChunkTable, coord: &[u64]) -> Option<&'a ChunkInfo> {
    table.chunks.get(coord)
}

/// Dataset operation: get chunk info.
#[allow(non_snake_case)]
pub fn H5D__get_chunk_info<'a>(table: &'a ChunkTable, coord: &[u64]) -> Option<&'a ChunkInfo> {
    H5D__get_chunk_info_cb(table, coord)
}

/// Dataset operation: get chunk info by coord.
#[allow(non_snake_case)]
pub fn H5D__get_chunk_info_by_coord<'a>(
    table: &'a ChunkTable,
    coord: &[u64],
) -> Option<&'a ChunkInfo> {
    H5D__get_chunk_info(table, coord)
}

/// Return the offset of a dataset.
#[allow(non_snake_case)]
pub fn H5D__chunk_get_offset_copy_slice(coord: &[u64]) -> &[u64] {
    coord
}

/// Return the offset of a dataset.
#[allow(non_snake_case)]
#[deprecated(note = "use H5D__chunk_get_offset_copy_slice to borrow the coordinate")]
pub fn H5D__chunk_get_offset_copy(coord: &[u64]) -> Vec<u64> {
    H5D__chunk_get_offset_copy_slice(coord).to_vec()
}

/// Read from a dataset.
#[allow(non_snake_case)]
pub fn H5D__read(dataset: &DatasetApi) -> &[u8] {
    &dataset.raw
}

/// Write to a dataset.
#[allow(non_snake_case)]
pub fn H5D__write(dataset: &mut DatasetApi, data: &[u8]) {
    H5D__write_api_common(dataset, data);
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChunkIndexState {
    pub open: bool,
    pub space_allocated: bool,
    pub entries: HashMap<Vec<u64>, ChunkInfo>,
    pub metadata_loaded: bool,
    pub declared_entry_count: u64,
    pub none_base_addr: Option<u64>,
    pub none_chunk_size: usize,
    pub none_chunks_per_dim: Vec<u64>,
}

/// Internal helper `chunk_index_insert`.
fn chunk_index_insert(index: &mut ChunkIndexState, coord: Vec<u64>, addr: u64, size: usize) {
    index.open = true;
    index.space_allocated = true;
    index.entries.insert(
        coord.clone(),
        ChunkInfo {
            coord,
            addr,
            size,
            filter_mask: 0,
        },
    );
}

/// Internal helper `chunk_index_get_addr`.
fn chunk_index_get_addr(index: &ChunkIndexState, coord: &[u64]) -> Option<u64> {
    index.entries.get(coord).map(|info| info.addr)
}

/// Internal helper `chunk_index_remove`.
fn chunk_index_remove(index: &mut ChunkIndexState, coord: &[u64]) -> Option<ChunkInfo> {
    let removed = index.entries.remove(coord);
    index.space_allocated = !index.entries.is_empty();
    removed
}

/// Internal helper `chunk_index_dump`.
fn chunk_index_dump_to<W: fmt::Write>(
    kind: &str,
    index: &ChunkIndexState,
    out: &mut W,
) -> fmt::Result {
    write!(
        out,
        "{kind}(open={}, allocated={}, entries={}, declared={})",
        index.open,
        index.space_allocated,
        index.entries.len(),
        index.declared_entry_count
    )
}

/// Iterate over a chunk index without cloning entries.
#[allow(non_snake_case)]
pub fn H5D__chunk_index_iterate(index: &ChunkIndexState) -> impl Iterator<Item = &ChunkInfo> + '_ {
    index.entries.values()
}

/// Render a chunk index debug string into caller-provided output.
#[allow(non_snake_case)]
pub fn H5D__chunk_index_dump_to<W: fmt::Write>(
    kind: &str,
    index: &ChunkIndexState,
    out: &mut W,
) -> fmt::Result {
    chunk_index_dump_to(kind, index, out)
}

/// Copy a chunk index into caller-provided storage.
#[allow(non_snake_case)]
pub fn H5D__chunk_index_copy_setup_into(index: &ChunkIndexState, out: &mut ChunkIndexState) {
    out.clone_from(index);
}

/// Internal helper `chunk_index_encode_count`.
fn chunk_index_encode_count_array(index: &ChunkIndexState) -> Result<[u8; 8]> {
    let count = u64::try_from(index.entries.len()).map_err(|_| {
        Error::InvalidFormat("chunk-index entry count cannot be represented as u64".into())
    })?;
    Ok(count.to_le_bytes())
}

/// Encode a chunk-index count into caller-provided storage.
#[allow(non_snake_case)]
pub fn H5D__chunk_index_encode_count_into(
    index: &ChunkIndexState,
    out: &mut Vec<u8>,
) -> Result<()> {
    out.clear();
    out.extend_from_slice(&chunk_index_encode_count_array(index)?);
    Ok(())
}

/// Internal helper `chunk_index_encode_count`.
fn chunk_index_encode_count(index: &ChunkIndexState) -> Result<Vec<u8>> {
    Ok(chunk_index_encode_count_array(index)?.into())
}

/// Internal helper `chunk_index_decode_count_image`.
fn chunk_index_decode_count_image(bytes: &[u8], context: &str) -> Result<ChunkIndexState> {
    if bytes.len() != 8 {
        return Err(Error::InvalidFormat(format!(
            "{context} chunk-index count image has invalid length"
        )));
    }
    let raw: [u8; 8] = bytes
        .try_into()
        .map_err(|_| Error::InvalidFormat(format!("{context} chunk-index count is truncated")))?;
    let declared_entry_count = u64::from_le_bytes(raw);
    Ok(ChunkIndexState {
        metadata_loaded: true,
        space_allocated: declared_entry_count != 0,
        declared_entry_count,
        ..ChunkIndexState::default()
    })
}

/// Internal helper `none_total_chunks`.
fn none_total_chunks(chunks_per_dim: &[u64]) -> Result<u64> {
    if chunks_per_dim.is_empty() {
        return Err(Error::InvalidFormat("none chunk index rank is zero".into()));
    }
    chunks_per_dim.iter().try_fold(1u64, |acc, &dim| {
        if dim == 0 {
            return Err(Error::InvalidFormat(
                "none chunk index dimension is zero".into(),
            ));
        }
        acc.checked_mul(dim)
            .ok_or_else(|| Error::InvalidFormat("none chunk count overflow".into()))
    })
}

/// Internal helper `none_array_offset_pre`.
fn none_array_offset_pre(chunks_per_dim: &[u64], coord: &[u64]) -> Result<u64> {
    if chunks_per_dim.len() != coord.len() {
        return Err(Error::InvalidFormat(
            "none chunk rank does not match coordinate rank".into(),
        ));
    }
    let mut stride = 1u64;
    let mut index = 0u64;
    for (&dim, &coord) in chunks_per_dim.iter().zip(coord).rev() {
        if dim == 0 {
            return Err(Error::InvalidFormat(
                "none chunk index dimension is zero".into(),
            ));
        }
        if coord >= dim {
            return Err(Error::InvalidFormat(
                "none chunk coordinate is outside chunk grid".into(),
            ));
        }
        index = index
            .checked_add(coord.checked_mul(stride).ok_or_else(|| {
                Error::InvalidFormat("none chunk coordinate offset overflow".into())
            })?)
            .ok_or_else(|| Error::InvalidFormat("none chunk coordinate offset overflow".into()))?;
        stride = stride
            .checked_mul(dim)
            .ok_or_else(|| Error::InvalidFormat("none chunk stride overflow".into()))?;
    }
    Ok(index)
}

/// Internal helper `none_increment_scaled_coord`.
fn none_increment_scaled_coord(coord: &mut [u64], chunks_per_dim: &[u64]) {
    for dim in (0..coord.len()).rev() {
        coord[dim] += 1;
        if coord[dim] >= chunks_per_dim[dim] {
            coord[dim] = 0;
        } else {
            break;
        }
    }
}

/// Internal helper `virtual_first_block_element_count`.
fn virtual_first_block_element_count(selection: &Selection) -> Result<u64> {
    match selection {
        Selection::Hyperslab(dims) => {
            if dims.is_empty() {
                return Ok(1);
            }
            dims.iter().try_fold(1u64, |acc, dim| {
                acc.checked_mul(dim.block).ok_or_else(|| {
                    Error::InvalidFormat("VDS virtual block element count overflow".into())
                })
            })
        }
        Selection::Slice(slices) => {
            if slices.is_empty() {
                return Ok(1);
            }
            slices.iter().try_fold(1u64, |acc, slice| {
                let block = if slice.count() == 0 { 0 } else { 1 };
                acc.checked_mul(block).ok_or_else(|| {
                    Error::InvalidFormat("VDS virtual block element count overflow".into())
                })
            })
        }
        _ => Err(Error::InvalidFormat(
            "VDS printf mapping virtual selection must be hyperslab".into(),
        )),
    }
}

macro_rules! chunk_index_family {
    (
        $crt_context:ident,
        $dst_context:ident,
        $fill:ident,
        $encode:ident,
        $decode:ident,
        $debug:ident,
        $idx_depend:ident,
        $idx_init:ident,
        $idx_create:ident,
        $idx_open:ident,
        $idx_close:ident,
        $idx_is_open:ident,
        $idx_is_space_alloc:ident,
        $idx_insert:ident,
        $idx_get_addr:ident,
        $idx_load_metadata:ident,
        $idx_iterate_cb:ident,
        $idx_iterate:ident,
        $idx_remove:ident,
        $idx_delete_cb:ident,
        $idx_delete:ident,
        $idx_copy_setup:ident,
        $idx_size:ident,
        $idx_reset:ident,
        $idx_dump:ident,
        $idx_dest:ident,
        $kind:literal
    ) => {
        #[allow(non_snake_case)]
        pub fn $crt_context() -> ChunkIndexState {
            ChunkIndexState::default()
        }

        #[allow(non_snake_case)]
        pub fn $dst_context(index: &mut ChunkIndexState) {
            index.entries.clear();
        }

        #[allow(non_snake_case)]
        pub fn $fill(size: usize) -> Vec<u8> {
            vec![0; size]
        }

        #[allow(non_snake_case)]
        pub fn $encode(index: &ChunkIndexState) -> Result<Vec<u8>> {
            chunk_index_encode_count(index)
        }

        #[allow(non_snake_case)]
        pub fn $decode(bytes: &[u8]) -> Result<ChunkIndexState> {
            chunk_index_decode_count_image(bytes, $kind)
        }

        #[allow(non_snake_case)]
        #[deprecated(note = "use H5D__chunk_index_dump_to to write into caller-provided output")]
        pub fn $debug(index: &ChunkIndexState) -> String {
            let mut out = String::new();
            let _ = H5D__chunk_index_dump_to($kind, index, &mut out);
            out
        }

        #[allow(non_snake_case)]
        pub fn $idx_depend(index: &ChunkIndexState) -> usize {
            index.entries.len()
        }

        #[allow(non_snake_case)]
        pub fn $idx_init() -> ChunkIndexState {
            ChunkIndexState::default()
        }

        #[allow(non_snake_case)]
        pub fn $idx_create(index: &mut ChunkIndexState) {
            index.open = true;
        }

        #[allow(non_snake_case)]
        pub fn $idx_open(index: &mut ChunkIndexState) {
            index.open = true;
        }

        #[allow(non_snake_case)]
        pub fn $idx_close(index: &mut ChunkIndexState) {
            index.open = false;
        }

        #[allow(non_snake_case)]
        pub fn $idx_is_open(index: &ChunkIndexState) -> bool {
            index.open
        }

        #[allow(non_snake_case)]
        pub fn $idx_is_space_alloc(index: &ChunkIndexState) -> bool {
            index.space_allocated
        }

        #[allow(non_snake_case)]
        pub fn $idx_insert(index: &mut ChunkIndexState, coord: Vec<u64>, addr: u64, size: usize) {
            chunk_index_insert(index, coord, addr, size);
        }

        #[allow(non_snake_case)]
        pub fn $idx_get_addr(index: &ChunkIndexState, coord: &[u64]) -> Option<u64> {
            chunk_index_get_addr(index, coord)
        }

        #[allow(non_snake_case)]
        pub fn $idx_load_metadata(index: &mut ChunkIndexState) {
            index.metadata_loaded = true;
        }

        #[allow(non_snake_case)]
        #[deprecated(note = "use H5D__chunk_index_iterate to borrow chunk info entries")]
        pub fn $idx_iterate_cb(info: &ChunkInfo) -> ChunkInfo {
            info.clone()
        }

        #[allow(non_snake_case)]
        #[deprecated(note = "use H5D__chunk_index_iterate to avoid cloning chunk info entries")]
        pub fn $idx_iterate(index: &ChunkIndexState) -> Vec<ChunkInfo> {
            H5D__chunk_index_iterate(index).cloned().collect()
        }

        #[allow(non_snake_case)]
        pub fn $idx_remove(index: &mut ChunkIndexState, coord: &[u64]) -> Option<ChunkInfo> {
            chunk_index_remove(index, coord)
        }

        #[allow(non_snake_case)]
        #[deprecated(note = "use H5D__chunk_index_iterate to borrow chunk info entries")]
        pub fn $idx_delete_cb(info: &ChunkInfo) -> ChunkInfo {
            info.clone()
        }

        #[allow(non_snake_case)]
        pub fn $idx_delete(index: &mut ChunkIndexState) {
            index.entries.clear();
            index.space_allocated = false;
        }

        #[allow(non_snake_case)]
        #[deprecated(
            note = "use H5D__chunk_index_copy_setup_into to reuse caller-provided storage"
        )]
        pub fn $idx_copy_setup(index: &ChunkIndexState) -> ChunkIndexState {
            index.clone()
        }

        #[allow(non_snake_case)]
        pub fn $idx_size(index: &ChunkIndexState) -> usize {
            index.entries.len()
        }

        #[allow(non_snake_case)]
        pub fn $idx_reset(index: &mut ChunkIndexState) {
            *index = ChunkIndexState::default();
        }

        #[allow(non_snake_case)]
        #[deprecated(note = "use H5D__chunk_index_dump_to to write into caller-provided output")]
        pub fn $idx_dump(index: &ChunkIndexState) -> String {
            let mut out = String::new();
            let _ = H5D__chunk_index_dump_to($kind, index, &mut out);
            out
        }

        #[allow(non_snake_case)]
        pub fn $idx_dest(index: &mut ChunkIndexState) {
            index.entries.clear();
            index.open = false;
            index.space_allocated = false;
        }
    };
}

chunk_index_family!(
    H5D__earray_crt_context,
    H5D__earray_dst_context,
    H5D__earray_fill,
    H5D__earray_encode,
    H5D__earray_filt_decode,
    H5D__earray_debug,
    H5D__earray_idx_depend,
    H5D__earray_idx_init,
    H5D__earray_idx_create,
    H5D__earray_idx_open,
    H5D__earray_idx_close,
    H5D__earray_idx_is_open,
    H5D__earray_idx_is_space_alloc,
    H5D__earray_idx_insert,
    H5D__earray_idx_get_addr,
    H5D__earray_idx_load_metadata,
    H5D__earray_idx_iterate_cb,
    H5D__earray_idx_iterate,
    H5D__earray_idx_remove,
    H5D__earray_idx_delete_cb,
    H5D__earray_idx_delete,
    H5D__earray_idx_copy_setup,
    H5D__earray_idx_resize,
    H5D__earray_idx_reset,
    H5D__earray_idx_dump,
    H5D__earray_idx_dest,
    "earray"
);

/// Dataset operation: earray filt fill.
#[allow(non_snake_case)]
pub fn H5D__earray_filt_fill(size: usize) -> Vec<u8> {
    H5D__earray_fill(size)
}

/// Encode a dataset to its on-disk representation.
#[allow(non_snake_case)]
pub fn H5D__earray_filt_encode(index: &ChunkIndexState) -> Result<Vec<u8>> {
    H5D__earray_encode(index)
}

/// Return a debug-friendly representation of a dataset.
#[allow(non_snake_case)]
#[deprecated(note = "use H5D__chunk_index_dump_to to write into caller-provided output")]
pub fn H5D__earray_filt_debug(index: &ChunkIndexState) -> String {
    let mut out = String::new();
    let _ = H5D__chunk_index_dump_to("earray", index, &mut out);
    out
}

/// Dataset operation: earray crt dbg context.
#[allow(non_snake_case)]
pub fn H5D__earray_crt_dbg_context() -> ChunkIndexState {
    H5D__earray_crt_context()
}

/// Dataset operation: earray filt crt dbg context.
#[allow(non_snake_case)]
pub fn H5D__earray_filt_crt_dbg_context() -> ChunkIndexState {
    H5D__earray_crt_context()
}

/// Dataset operation: earray dst dbg context.
#[allow(non_snake_case)]
pub fn H5D__earray_dst_dbg_context(index: &mut ChunkIndexState) {
    H5D__earray_dst_context(index);
}

chunk_index_family!(
    H5D__farray_crt_context,
    H5D__farray_dst_context,
    H5D__farray_fill,
    H5D__farray_encode,
    H5D__farray_decode,
    H5D__farray_debug,
    H5D__farray_idx_depend,
    H5D__farray_idx_init,
    H5D__farray_idx_create,
    H5D__farray_idx_open,
    H5D__farray_idx_close,
    H5D__farray_idx_is_open,
    H5D__farray_idx_is_space_alloc,
    H5D__farray_idx_insert,
    H5D__farray_idx_get_addr,
    H5D__farray_idx_load_metadata,
    H5D__farray_idx_iterate_cb,
    H5D__farray_idx_iterate,
    H5D__farray_idx_remove,
    H5D__farray_idx_delete_cb,
    H5D__farray_idx_delete,
    H5D__farray_idx_copy_setup,
    H5D__farray_idx_size,
    H5D__farray_idx_reset,
    H5D__farray_idx_dump,
    H5D__farray_idx_dest,
    "farray"
);

/// Dataset operation: farray crt dbg context.
#[allow(non_snake_case)]
pub fn H5D__farray_crt_dbg_context() -> ChunkIndexState {
    H5D__farray_crt_context()
}

/// Dataset operation: farray dst dbg context.
#[allow(non_snake_case)]
pub fn H5D__farray_dst_dbg_context(index: &mut ChunkIndexState) {
    H5D__farray_dst_context(index);
}

/// Dataset operation: farray filt fill.
#[allow(non_snake_case)]
pub fn H5D__farray_filt_fill(size: usize) -> Vec<u8> {
    H5D__farray_fill(size)
}

/// Encode a dataset to its on-disk representation.
#[allow(non_snake_case)]
pub fn H5D__farray_filt_encode(index: &ChunkIndexState) -> Result<Vec<u8>> {
    H5D__farray_encode(index)
}

/// Decode a dataset from its on-disk representation.
#[allow(non_snake_case)]
pub fn H5D__farray_filt_decode(bytes: &[u8]) -> Result<ChunkIndexState> {
    H5D__farray_decode(bytes)
}

/// Return a debug-friendly representation of a dataset.
#[allow(non_snake_case)]
#[deprecated(note = "use H5D__chunk_index_dump_to to write into caller-provided output")]
pub fn H5D__farray_filt_debug(index: &ChunkIndexState) -> String {
    let mut out = String::new();
    let _ = H5D__chunk_index_dump_to("farray", index, &mut out);
    out
}

/// Dataset operation: farray filt crt dbg context.
#[allow(non_snake_case)]
pub fn H5D__farray_filt_crt_dbg_context() -> ChunkIndexState {
    H5D__farray_crt_context()
}

chunk_index_family!(
    H5D__bt2_crt_context,
    H5D__bt2_dst_context,
    H5D__bt2_store,
    H5D__bt2_unfilt_encode,
    H5D__bt2_unfilt_decode,
    H5D__bt2_unfilt_debug,
    H5D__btree2_idx_depend,
    H5D__bt2_idx_init,
    H5D__bt2_idx_create,
    H5D__bt2_idx_open,
    H5D__bt2_idx_close,
    H5D__bt2_idx_is_open,
    H5D__bt2_idx_is_space_alloc,
    H5D__bt2_idx_insert,
    H5D__bt2_idx_get_addr,
    H5D__bt2_idx_load_metadata,
    H5D__bt2_idx_iterate_cb,
    H5D__bt2_idx_iterate,
    H5D__bt2_idx_remove,
    H5D__bt2_remove_cb,
    H5D__bt2_idx_delete,
    H5D__bt2_idx_copy_setup,
    H5D__bt2_idx_size,
    H5D__bt2_idx_reset,
    H5D__bt2_idx_dump,
    H5D__bt2_idx_dest,
    "bt2"
);

/// Dataset operation: bt2 compare.
#[allow(non_snake_case)]
pub fn H5D__bt2_compare(left: &ChunkInfo, right: &ChunkInfo) -> std::cmp::Ordering {
    H5D__chunk_cmp_coll_fill_info(left, right)
}

/// Encode a dataset to its on-disk representation.
#[allow(non_snake_case)]
pub fn H5D__bt2_filt_encode(index: &ChunkIndexState) -> Result<Vec<u8>> {
    H5D__bt2_unfilt_encode(index)
}

/// Decode a dataset from its on-disk representation.
#[allow(non_snake_case)]
pub fn H5D__bt2_filt_decode(bytes: &[u8]) -> Result<ChunkIndexState> {
    H5D__bt2_unfilt_decode(bytes)
}

/// Return a debug-friendly representation of a dataset.
#[allow(non_snake_case)]
#[deprecated(note = "use H5D__chunk_index_dump_to to write into caller-provided output")]
pub fn H5D__bt2_filt_debug(index: &ChunkIndexState) -> String {
    let mut out = String::new();
    let _ = H5D__chunk_index_dump_to("bt2", index, &mut out);
    out
}

/// Dataset operation: bt2 mod cb.
#[allow(non_snake_case)]
pub fn H5D__bt2_mod_cb(info: &mut ChunkInfo, addr: u64, size: usize) {
    H5D__chunk_set_info(info, addr, size);
}

/// Dataset operation: bt2 found cb.
#[allow(non_snake_case)]
#[deprecated(note = "use H5D__bt2_found_cb_ref to borrow the chunk info")]
pub fn H5D__bt2_found_cb(info: &ChunkInfo) -> ChunkInfo {
    info.clone()
}

/// Dataset operation: bt2 found cb.
#[allow(non_snake_case)]
pub fn H5D__bt2_found_cb_ref(info: &ChunkInfo) -> &ChunkInfo {
    info
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DatasetIoInfo {
    pub element_size: usize,
    pub file_selection: Vec<u64>,
    pub memory_selection: Vec<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DatasetTypeInfo {
    pub src_size: usize,
    pub dst_size: usize,
    pub conversion_needed: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ContiguousStorage {
    pub addr: Option<u64>,
    pub data: Vec<u8>,
    pub cached: bool,
    pub allocated: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExternalFileList {
    pub files: Vec<String>,
    pub allocated: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FillState {
    pub value: Vec<u8>,
    pub initialized: bool,
}

/// Initialize the dataset subsystem.
#[allow(non_snake_case)]
pub fn H5D__ioinfo_init(element_size: usize) -> DatasetIoInfo {
    DatasetIoInfo {
        element_size,
        file_selection: Vec::new(),
        memory_selection: Vec::new(),
    }
}

/// Initialize the dataset subsystem.
#[allow(non_snake_case)]
pub fn H5D__dset_ioinfo_init(dataset: &DatasetApi, element_size: usize) -> DatasetIoInfo {
    let mut info = DatasetIoInfo {
        element_size,
        file_selection: Vec::new(),
        memory_selection: Vec::new(),
    };
    H5D__ioinfo_adjust(&mut info, H5D__get_space(dataset));
    info
}

/// Initialize the dataset subsystem.
#[allow(non_snake_case)]
pub fn H5D__typeinfo_init(src_size: usize, dst_size: usize) -> DatasetTypeInfo {
    DatasetTypeInfo {
        src_size,
        dst_size,
        conversion_needed: src_size != dst_size,
    }
}

/// Initialize the dataset subsystem.
#[allow(non_snake_case)]
pub fn H5D__typeinfo_init_phase2(info: &mut DatasetTypeInfo) {
    info.conversion_needed = info.src_size != info.dst_size;
}

/// Dataset operation: ioinfo adjust.
#[allow(non_snake_case)]
pub fn H5D__ioinfo_adjust(info: &mut DatasetIoInfo, extent: &[u64]) {
    info.file_selection.clear();
    info.file_selection.extend_from_slice(extent);
    info.memory_selection.clear();
    info.memory_selection.extend_from_slice(extent);
}

/// Initialize the dataset subsystem.
#[allow(non_snake_case)]
pub fn H5D__typeinfo_init_phase3(info: &mut DatasetTypeInfo) {
    H5D__typeinfo_init_phase2(info);
}

/// Dataset operation: typeinfo term.
#[allow(non_snake_case)]
pub fn H5D__typeinfo_term(info: &mut DatasetTypeInfo) {
    *info = DatasetTypeInfo::default();
}

/// Dataset operation: layout version test.
#[allow(non_snake_case)]
pub fn H5D__layout_version_test(version: u8) -> bool {
    version <= 4
}

/// Dataset operation: layout contig size test.
#[allow(non_snake_case)]
pub fn H5D__layout_contig_size_test(size: u64) -> bool {
    size > 0
}

/// Return a debug-friendly representation of a dataset.
#[allow(non_snake_case)]
pub fn H5Ddebug_to<W: fmt::Write>(dataset: &DatasetApi, out: &mut W) -> fmt::Result {
    write!(
        out,
        "dataset(name={:?}, extent={:?}, bytes={})",
        dataset.name,
        dataset.extent,
        dataset.raw.len()
    )
}

/// Return a debug-friendly representation of a dataset.
#[allow(non_snake_case)]
#[deprecated(note = "use H5Ddebug_to to write into caller-provided output")]
pub fn H5Ddebug(dataset: &DatasetApi) -> String {
    let mut out = String::new();
    let _ = H5Ddebug_to(dataset, &mut out);
    out
}

/// Return a debug-friendly representation of a dataset.
#[allow(non_snake_case)]
pub fn H5D__mpio_debug_init() -> Result<()> {
    Err(Error::Unsupported(
        "MPI dataset I/O is intentionally unsupported".into(),
    ))
}

/// Dataset operation: mpio opt possible.
#[allow(non_snake_case)]
pub fn H5D__mpio_opt_possible() -> bool {
    false
}

/// Write to a dataset.
#[allow(non_snake_case)]
pub fn H5D__mpio_select_write() -> Result<()> {
    Err(Error::Unsupported(
        "MPI collective dataset writes are intentionally unsupported".into(),
    ))
}

/// Dataset operation: mpio get sum chunk.
#[allow(non_snake_case)]
pub fn H5D__mpio_get_sum_chunk(chunks: &[ChunkInfo]) -> usize {
    chunks.iter().map(|chunk| chunk.size).sum()
}

/// Dataset operation: mpio get sum chunk dset.
#[allow(non_snake_case)]
pub fn H5D__mpio_get_sum_chunk_dset(table: &ChunkTable) -> usize {
    table.chunks.values().map(|chunk| chunk.size).sum()
}

/// Dataset operation: piece io.
#[allow(non_snake_case)]
pub fn H5D__piece_io() -> Result<()> {
    Err(Error::Unsupported(
        "piece I/O is only used by MPI collective dataset paths".into(),
    ))
}

/// Link a dataset.
#[allow(non_snake_case)]
pub fn H5D__link_chunk_filtered_collective_io() -> Result<()> {
    Err(Error::Unsupported(
        "MPI collective filtered chunk I/O is unsupported".into(),
    ))
}

/// Dataset operation: multi chunk collective io.
#[allow(non_snake_case)]
pub fn H5D__multi_chunk_collective_io() -> Result<()> {
    Err(Error::Unsupported(
        "MPI collective chunk I/O is unsupported".into(),
    ))
}

/// Dataset operation: multi chunk filtered collective io.
#[allow(non_snake_case)]
pub fn H5D__multi_chunk_filtered_collective_io() -> Result<()> {
    Err(Error::Unsupported(
        "MPI collective filtered chunk I/O is unsupported".into(),
    ))
}

/// Dataset operation: inter collective io.
#[allow(non_snake_case)]
pub fn H5D__inter_collective_io() -> Result<()> {
    Err(Error::Unsupported(
        "MPI inter-collective dataset I/O is unsupported".into(),
    ))
}

/// Dataset operation: final collective io.
#[allow(non_snake_case)]
pub fn H5D__final_collective_io() -> Result<()> {
    Err(Error::Unsupported(
        "MPI collective dataset I/O is unsupported".into(),
    ))
}

/// Dataset operation: cmp piece addr.
#[allow(non_snake_case)]
pub fn H5D__cmp_piece_addr(left: &ChunkInfo, right: &ChunkInfo) -> std::cmp::Ordering {
    left.addr.cmp(&right.addr)
}

/// Dataset operation: cmp filtered collective io info entry.
#[allow(non_snake_case)]
pub fn H5D__cmp_filtered_collective_io_info_entry(
    left: &ChunkInfo,
    right: &ChunkInfo,
) -> std::cmp::Ordering {
    H5D__chunk_cmp_coll_fill_info(left, right)
}

/// Dataset operation: cmp chunk redistribute info.
#[allow(non_snake_case)]
pub fn H5D__cmp_chunk_redistribute_info(left: &ChunkInfo, right: &ChunkInfo) -> std::cmp::Ordering {
    left.coord.cmp(&right.coord)
}

/// Dataset operation: cmp chunk redistribute info orig owner.
#[allow(non_snake_case)]
pub fn H5D__cmp_chunk_redistribute_info_orig_owner(
    left: &ChunkInfo,
    right: &ChunkInfo,
) -> std::cmp::Ordering {
    left.addr
        .cmp(&right.addr)
        .then_with(|| left.coord.cmp(&right.coord))
}

/// Dataset operation: obtain mpio mode.
#[allow(non_snake_case)]
pub fn H5D__obtain_mpio_mode() -> Result<()> {
    Err(Error::Unsupported(
        "MPI dataset transfer mode is unavailable".into(),
    ))
}

/// Dataset operation: mpio collective filtered chunk io setup.
#[allow(non_snake_case)]
pub fn H5D__mpio_collective_filtered_chunk_io_setup() -> Result<()> {
    Err(Error::Unsupported(
        "MPI collective filtered chunk I/O is unsupported".into(),
    ))
}

/// Dataset operation: mpio collective filtered vec io.
#[allow(non_snake_case)]
pub fn H5D__mpio_collective_filtered_vec_io() -> Result<()> {
    Err(Error::Unsupported(
        "MPI collective filtered vector I/O is unsupported".into(),
    ))
}

/// Dataset operation: mpio redistribute shared chunks int.
#[allow(non_snake_case)]
pub fn H5D__mpio_redistribute_shared_chunks_int() -> Result<()> {
    Err(Error::Unsupported(
        "MPI shared chunk redistribution is unsupported".into(),
    ))
}

/// Dataset operation: mpio share chunk modification data.
#[allow(non_snake_case)]
pub fn H5D__mpio_share_chunk_modification_data() -> Result<()> {
    Err(Error::Unsupported(
        "MPI chunk modification sharing is unsupported".into(),
    ))
}

/// Read from a dataset.
#[allow(non_snake_case)]
pub fn H5D__mpio_collective_filtered_chunk_read() -> Result<()> {
    Err(Error::Unsupported(
        "MPI collective filtered chunk read is unsupported".into(),
    ))
}

/// Update a dataset.
#[allow(non_snake_case)]
pub fn H5D__mpio_collective_filtered_chunk_update() -> Result<()> {
    Err(Error::Unsupported(
        "MPI collective filtered chunk update is unsupported".into(),
    ))
}

/// Dataset operation: mpio collective filtered chunk reallocate.
#[allow(non_snake_case)]
pub fn H5D__mpio_collective_filtered_chunk_reallocate() -> Result<()> {
    Err(Error::Unsupported(
        "MPI collective filtered chunk reallocate is unsupported".into(),
    ))
}

/// Dataset operation: mpio collective filtered chunk reinsert.
#[allow(non_snake_case)]
pub fn H5D__mpio_collective_filtered_chunk_reinsert() -> Result<()> {
    Err(Error::Unsupported(
        "MPI collective filtered chunk reinsert is unsupported".into(),
    ))
}

/// Dataset operation: mpio get chunk redistribute info types.
#[allow(non_snake_case)]
pub fn H5D__mpio_get_chunk_redistribute_info_types() -> Result<()> {
    Err(Error::Unsupported(
        "MPI datatype construction is unsupported".into(),
    ))
}

/// Allocate storage for a dataset.
#[allow(non_snake_case)]
pub fn H5D__mpio_get_chunk_alloc_info_types() -> Result<()> {
    Err(Error::Unsupported(
        "MPI datatype construction is unsupported".into(),
    ))
}

/// Insert an entry into a dataset.
#[allow(non_snake_case)]
pub fn H5D__mpio_get_chunk_insert_info_types() -> Result<()> {
    Err(Error::Unsupported(
        "MPI datatype construction is unsupported".into(),
    ))
}

/// Render a dataset for debug output.
#[allow(non_snake_case)]
#[deprecated(
    note = "use H5D__mpio_dump_collective_filtered_chunk_list_to to write into caller-provided output"
)]
pub fn H5D__mpio_dump_collective_filtered_chunk_list(chunks: &[ChunkInfo]) -> String {
    let mut out = String::new();
    let _ = H5D__mpio_dump_collective_filtered_chunk_list_to(chunks, &mut out);
    out
}

/// Render a dataset for debug output into caller-provided output.
#[allow(non_snake_case)]
pub fn H5D__mpio_dump_collective_filtered_chunk_list_to<W: fmt::Write>(
    chunks: &[ChunkInfo],
    out: &mut W,
) -> fmt::Result {
    write!(out, "{} collective chunks", chunks.len())
}

/// Dataset operation: scatter file.
#[allow(non_snake_case)]
#[deprecated(
    since = "0.4.0",
    note = "use H5D__scatter_file_into to fill caller-provided buffers"
)]
pub fn H5D__scatter_file(src: &[u8], spans: &[(usize, usize)]) -> Result<Vec<Vec<u8>>> {
    let mut out: Vec<Vec<u8>> = spans.iter().map(|&(_, len)| vec![0; len]).collect();
    H5D__scatter_file_into(src, spans, &mut out)?;
    Ok(out)
}

fn validate_scatter_output_shapes<'a, I>(spans: &[(usize, usize)], out: I) -> Result<()>
where
    I: IntoIterator<Item = &'a [u8]>,
{
    let mut out = out.into_iter();
    for &(_, len) in spans {
        let Some(dst) = out.next() else {
            return Err(Error::InvalidFormat(
                "dataset scatter output span count mismatch".into(),
            ));
        };
        if dst.len() != len {
            return Err(Error::InvalidFormat(
                "dataset scatter output span size mismatch".into(),
            ));
        }
    }
    if out.next().is_some() {
        return Err(Error::InvalidFormat(
            "dataset scatter output span count mismatch".into(),
        ));
    }
    Ok(())
}

fn validate_scatter_output_shapes_mut(spans: &[(usize, usize)], out: &[&mut [u8]]) -> Result<()> {
    if out.len() != spans.len() {
        return Err(Error::InvalidFormat(
            "dataset scatter output span count mismatch".into(),
        ));
    }

    spans.iter().zip(out).try_for_each(|(&(_, len), dst)| {
        if dst.len() != len {
            return Err(Error::InvalidFormat(
                "dataset scatter output span size mismatch".into(),
            ));
        }
        Ok(())
    })
}

/// Dataset operation: scatter file borrowed visitor.
#[allow(non_snake_case)]
pub fn H5D__scatter_file_with<F>(src: &[u8], spans: &[(usize, usize)], mut visitor: F) -> Result<()>
where
    F: FnMut(&[u8]),
{
    for &(offset, len) in spans {
        let end = offset
            .checked_add(len)
            .ok_or_else(|| Error::InvalidFormat("dataset scatter span overflow".into()))?;
        let span = src
            .get(offset..end)
            .ok_or_else(|| Error::InvalidFormat("dataset scatter span out of bounds".into()))?;
        visitor(span);
    }
    Ok(())
}

/// Dataset operation: scatter file into caller-provided buffers.
#[allow(non_snake_case)]
pub fn H5D__scatter_file_into(
    src: &[u8],
    spans: &[(usize, usize)],
    out: &mut [Vec<u8>],
) -> Result<()> {
    validate_scatter_output_shapes(spans, out.iter().map(Vec::as_slice))?;

    let mut out = out.iter_mut();
    H5D__scatter_file_with(src, spans, |span| {
        if let Some(dst) = out.next() {
            dst.copy_from_slice(span);
        }
    })?;

    Ok(())
}

/// Dataset operation: scatter file into caller-provided borrowed buffers.
#[allow(non_snake_case)]
pub fn H5D__scatter_file_into_slices(
    src: &[u8],
    spans: &[(usize, usize)],
    out: &mut [&mut [u8]],
) -> Result<()> {
    validate_scatter_output_shapes_mut(spans, out)?;

    let mut out = out.iter_mut();
    H5D__scatter_file_with(src, spans, |span| {
        if let Some(dst) = out.next() {
            dst.copy_from_slice(span);
        }
    })?;

    Ok(())
}

/// Dataset operation: scatter file checked.
#[allow(non_snake_case)]
#[deprecated(
    since = "0.4.0",
    note = "use H5D__scatter_file_into to fill caller-provided buffers"
)]
pub fn H5D__scatter_file_checked(src: &[u8], spans: &[(usize, usize)]) -> Result<Vec<Vec<u8>>> {
    let mut out: Vec<Vec<u8>> = spans.iter().map(|&(_, len)| vec![0; len]).collect();
    H5D__scatter_file_into(src, spans, &mut out)?;
    Ok(out)
}

/// Dataset operation: gather file.
#[allow(non_snake_case)]
pub fn H5D__gather_file_into<'a, I>(parts: I, out: &mut Vec<u8>)
where
    I: IntoIterator<Item = &'a [u8]>,
{
    out.clear();
    for part in parts {
        out.extend_from_slice(part);
    }
}

/// Dataset operation: gather file.
#[allow(non_snake_case)]
#[deprecated(note = "use H5D__gather_file_into to reuse caller-provided storage")]
pub fn H5D__gather_file(parts: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    H5D__gather_file_into(parts.iter().map(Vec::as_slice), &mut out);
    out
}

/// Dataset operation: scatter mem.
#[allow(non_snake_case)]
#[deprecated(
    since = "0.4.0",
    note = "use H5D__scatter_mem_into to fill caller-provided buffers"
)]
pub fn H5D__scatter_mem(src: &[u8], spans: &[(usize, usize)]) -> Result<Vec<Vec<u8>>> {
    let mut out: Vec<Vec<u8>> = spans.iter().map(|&(_, len)| vec![0; len]).collect();
    H5D__scatter_mem_into(src, spans, &mut out)?;
    Ok(out)
}

/// Dataset operation: scatter mem into caller-provided buffers.
#[allow(non_snake_case)]
pub fn H5D__scatter_mem_into(
    src: &[u8],
    spans: &[(usize, usize)],
    out: &mut [Vec<u8>],
) -> Result<()> {
    H5D__scatter_file_into(src, spans, out)
}

/// Dataset operation: scatter mem into caller-provided borrowed buffers.
#[allow(non_snake_case)]
pub fn H5D__scatter_mem_into_slices(
    src: &[u8],
    spans: &[(usize, usize)],
    out: &mut [&mut [u8]],
) -> Result<()> {
    H5D__scatter_file_into_slices(src, spans, out)
}

/// Dataset operation: scatter mem borrowed visitor.
#[allow(non_snake_case)]
pub fn H5D__scatter_mem_with<F>(src: &[u8], spans: &[(usize, usize)], visitor: F) -> Result<()>
where
    F: FnMut(&[u8]),
{
    H5D__scatter_file_with(src, spans, visitor)
}

/// Dataset operation: gather mem.
#[allow(non_snake_case)]
pub fn H5D__gather_mem_into<'a, I>(parts: I, out: &mut Vec<u8>)
where
    I: IntoIterator<Item = &'a [u8]>,
{
    H5D__gather_file_into(parts, out);
}

/// Dataset operation: gather mem.
#[allow(non_snake_case)]
#[deprecated(note = "use H5D__gather_mem_into to reuse caller-provided storage")]
pub fn H5D__gather_mem(parts: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    H5D__gather_mem_into(parts.iter().map(Vec::as_slice), &mut out);
    out
}

/// Read from a dataset.
#[allow(non_snake_case)]
pub fn H5D__scatgath_read_select_into(
    src: &[u8],
    spans: &[(usize, usize)],
    out: &mut [Vec<u8>],
) -> Result<()> {
    H5D__scatter_file_into(src, spans, out)
}

/// Read from a dataset into caller-provided borrowed buffers.
#[allow(non_snake_case)]
pub fn H5D__scatgath_read_select_into_slices(
    src: &[u8],
    spans: &[(usize, usize)],
    out: &mut [&mut [u8]],
) -> Result<()> {
    H5D__scatter_file_into_slices(src, spans, out)
}

/// Read selected spans through a borrowed visitor.
#[allow(non_snake_case)]
pub fn H5D__scatgath_read_select_with<F>(
    src: &[u8],
    spans: &[(usize, usize)],
    visitor: F,
) -> Result<()>
where
    F: FnMut(&[u8]),
{
    H5D__scatter_file_with(src, spans, visitor)
}

/// Read from a dataset.
#[allow(non_snake_case)]
#[deprecated(note = "use H5D__scatgath_read_select_into to fill caller-provided buffers")]
pub fn H5D__scatgath_read_select(src: &[u8], spans: &[(usize, usize)]) -> Result<Vec<Vec<u8>>> {
    let mut out: Vec<Vec<u8>> = spans.iter().map(|&(_, len)| vec![0; len]).collect();
    H5D__scatgath_read_select_into(src, spans, &mut out)?;
    Ok(out)
}

/// Write to a dataset.
#[allow(non_snake_case)]
pub fn H5D__scatgath_write_select_into<'a, I>(parts: I, out: &mut Vec<u8>)
where
    I: IntoIterator<Item = &'a [u8]>,
{
    H5D__gather_file_into(parts, out);
}

/// Write to a dataset.
#[allow(non_snake_case)]
#[deprecated(note = "use H5D__scatgath_write_select_into to reuse caller-provided storage")]
pub fn H5D__scatgath_write_select(parts: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    H5D__scatgath_write_select_into(parts.iter().map(Vec::as_slice), &mut out);
    out
}

/// Read from a dataset.
#[allow(non_snake_case)]
pub fn H5D__compound_opt_read_slice(src: &[u8]) -> &[u8] {
    src
}

/// Read from a dataset.
#[allow(non_snake_case)]
#[deprecated(note = "use H5D__compound_opt_read_slice to borrow optimized compound bytes")]
pub fn H5D__compound_opt_read(src: &[u8]) -> Vec<u8> {
    H5D__compound_opt_read_slice(src).to_vec()
}

/// Dataset operation: efl construct.
#[allow(non_snake_case)]
pub fn H5D__efl_construct(files: Vec<String>) -> ExternalFileList {
    ExternalFileList {
        files,
        allocated: true,
    }
}

/// Initialize the dataset subsystem.
#[allow(non_snake_case)]
pub fn H5D__efl_init(efl: &mut ExternalFileList) {
    efl.allocated = true;
}

/// Allocate storage for a dataset.
#[allow(non_snake_case)]
pub fn H5D__efl_is_space_alloc(efl: &ExternalFileList) -> bool {
    efl.allocated
}

/// Initialize the dataset subsystem.
#[allow(non_snake_case)]
pub fn H5D__efl_io_init(_efl: &mut ExternalFileList) {}

/// Read from a dataset.
#[allow(non_snake_case)]
pub fn H5D__efl_read(_efl: &ExternalFileList, _offset: u64, _buf: &mut [u8]) -> Result<()> {
    Err(Error::Unsupported(
        "external raw dataset file I/O is handled by high-level dataset storage".into(),
    ))
}

/// Write to a dataset.
#[allow(non_snake_case)]
pub fn H5D__efl_write(_efl: &ExternalFileList, _offset: u64, _data: &[u8]) -> Result<()> {
    Err(Error::Unsupported(
        "external raw dataset writes are handled by high-level dataset storage".into(),
    ))
}

/// Dataset operation: efl readvv cb.
#[allow(non_snake_case)]
pub fn H5D__efl_readvv_cb_into(out: &mut [u8]) {
    out.fill(0);
}

/// Dataset operation: efl readvv cb.
#[allow(non_snake_case)]
#[deprecated(note = "use H5D__efl_readvv_cb_into to fill caller-provided buffers")]
pub fn H5D__efl_readvv_cb(len: usize) -> Vec<u8> {
    let mut out = vec![0; len];
    H5D__efl_readvv_cb_into(&mut out);
    out
}

/// Dataset operation: efl readvv.
#[allow(non_snake_case)]
pub fn H5D__efl_readvv_into(efl: &ExternalFileList, offset: u64, out: &mut [u8]) -> Result<()> {
    H5D__efl_read(efl, offset, out)
}

/// Dataset operation: efl readvv.
#[allow(non_snake_case)]
#[deprecated(note = "use H5D__efl_readvv_into to fill caller-provided buffers")]
pub fn H5D__efl_readvv(efl: &ExternalFileList, offset: u64, len: usize) -> Result<Vec<u8>> {
    let mut buf = vec![0; len];
    H5D__efl_readvv_into(efl, offset, &mut buf)?;
    Ok(buf)
}

/// Dataset operation: efl writevv cb.
#[allow(non_snake_case)]
pub fn H5D__efl_writevv_cb(data: &[u8]) -> usize {
    data.len()
}

/// Dataset operation: efl writevv.
#[allow(non_snake_case)]
pub fn H5D__efl_writevv(efl: &ExternalFileList, offset: u64, data: &[u8]) -> Result<()> {
    H5D__efl_write(efl, offset, data)
}

/// Dataset operation: efl bh info.
#[allow(non_snake_case)]
pub fn H5D__efl_bh_info(efl: &ExternalFileList) -> usize {
    efl.files.len()
}

/// Dataset operation: layout set io ops.
#[allow(non_snake_case)]
pub fn H5D__layout_set_io_ops(_dataset: &mut DatasetApi) {}

/// Dataset operation: layout meta size.
#[allow(non_snake_case)]
pub fn H5D__layout_meta_size(dataset: &DatasetApi) -> usize {
    H5D__layout_meta_size_checked(dataset).unwrap_or(usize::MAX)
}

/// Dataset operation: layout meta size checked.
#[allow(non_snake_case)]
pub fn H5D__layout_meta_size_checked(dataset: &DatasetApi) -> Result<usize> {
    dataset
        .extent
        .len()
        .checked_mul(std::mem::size_of::<u64>())
        .ok_or_else(|| Error::InvalidFormat("dataset layout metadata size overflow".into()))
}

/// Create a new dataset.
#[allow(non_snake_case)]
pub fn H5D__layout_oh_create_into(dataset: &DatasetApi, out: &mut Vec<u8>) -> Result<()> {
    let size = H5D__layout_meta_size_checked(dataset)?;
    out.clear();
    out.reserve(size);
    for dim in &dataset.extent {
        out.extend_from_slice(&dim.to_le_bytes());
    }
    Ok(())
}

/// Create a new dataset.
#[allow(non_snake_case)]
#[deprecated(note = "use H5D__layout_oh_create_into to reuse caller-provided storage")]
pub fn H5D__layout_oh_create(dataset: &DatasetApi) -> Vec<u8> {
    let mut out = Vec::new();
    let _ = H5D__layout_oh_create_into(dataset, &mut out);
    out
}

/// Write to a dataset.
#[allow(non_snake_case)]
pub fn H5D__layout_oh_write_into(dataset: &DatasetApi, out: &mut Vec<u8>) -> Result<()> {
    H5D__layout_oh_create_into(dataset, out)
}

/// Write to a dataset.
#[allow(non_snake_case)]
#[deprecated(note = "use H5D__layout_oh_write_into to reuse caller-provided storage")]
pub fn H5D__layout_oh_write(dataset: &DatasetApi) -> Vec<u8> {
    let mut out = Vec::new();
    let _ = H5D__layout_oh_write_into(dataset, &mut out);
    out
}

/// Create a new dataset.
#[allow(non_snake_case)]
pub fn H5D__none_idx_create() -> ChunkIndexState {
    ChunkIndexState::default()
}

/// Dataset operation: none idx configure.
#[allow(non_snake_case)]
pub fn H5D__none_idx_configure(
    index: &mut ChunkIndexState,
    base_addr: u64,
    chunk_size: usize,
    chunks_per_dim: Vec<u64>,
) -> Result<()> {
    if chunk_size == 0 {
        return Err(Error::InvalidFormat(
            "none chunk index chunk size is zero".into(),
        ));
    }
    if chunks_per_dim.is_empty() || chunks_per_dim.iter().any(|&dim| dim == 0) {
        return Err(Error::InvalidFormat(
            "none chunk index dimensions must be nonzero".into(),
        ));
    }
    let _ = none_total_chunks(&chunks_per_dim)?;
    index.open = true;
    index.space_allocated = true;
    index.none_base_addr = Some(base_addr);
    index.none_chunk_size = chunk_size;
    index.none_chunks_per_dim = chunks_per_dim;
    Ok(())
}

/// Close a dataset.
#[allow(non_snake_case)]
pub fn H5D__none_idx_close(_index: &mut ChunkIndexState) {}

/// Open a dataset.
#[allow(non_snake_case)]
pub fn H5D__none_idx_is_open(_index: &ChunkIndexState) -> bool {
    true
}

/// Allocate storage for a dataset.
#[allow(non_snake_case)]
pub fn H5D__none_idx_is_space_alloc(_index: &ChunkIndexState) -> bool {
    _index.none_base_addr.is_some()
}

/// Dataset operation: none idx get addr.
#[allow(non_snake_case)]
pub fn H5D__none_idx_get_addr(index: &ChunkIndexState, coord: &[u64]) -> Option<u64> {
    H5D__none_idx_get_addr_checked(index, coord).ok()
}

/// Dataset operation: none idx get addr checked.
#[allow(non_snake_case)]
pub fn H5D__none_idx_get_addr_checked(index: &ChunkIndexState, coord: &[u64]) -> Result<u64> {
    let base_addr = index
        .none_base_addr
        .ok_or_else(|| Error::InvalidFormat("none chunk index base address is undefined".into()))?;
    let chunk_size = u64::try_from(index.none_chunk_size)
        .map_err(|_| Error::InvalidFormat("none chunk size exceeds u64".into()))?;
    if coord.len() != index.none_chunks_per_dim.len() {
        return Err(Error::InvalidFormat(format!(
            "none chunk coordinate rank {} does not match chunk rank {}",
            coord.len(),
            index.none_chunks_per_dim.len()
        )));
    }
    if coord
        .iter()
        .zip(&index.none_chunks_per_dim)
        .any(|(&coord, &dim)| coord >= dim)
    {
        return Err(Error::InvalidFormat(
            "none chunk coordinate is outside chunk grid".into(),
        ));
    }
    let chunk_idx = none_array_offset_pre(&index.none_chunks_per_dim, coord)?;
    base_addr
        .checked_add(
            chunk_idx
                .checked_mul(chunk_size)
                .ok_or_else(|| Error::InvalidFormat("none chunk byte offset overflow".into()))?,
        )
        .ok_or_else(|| Error::InvalidFormat("none chunk address overflow".into()))
}

/// Dataset operation: none idx load metadata.
#[allow(non_snake_case)]
pub fn H5D__none_idx_load_metadata(_index: &mut ChunkIndexState) {}

/// Iterate over the entries of a dataset.
#[allow(non_snake_case)]
pub fn H5D__none_idx_iterate_with<F>(index: &ChunkIndexState, mut visitor: F) -> Result<()>
where
    F: FnMut(&[u64], u64, usize),
{
    let total = none_total_chunks(&index.none_chunks_per_dim)?;
    let mut coord = vec![0u64; index.none_chunks_per_dim.len()];
    for _ in 0..total {
        let addr = H5D__none_idx_get_addr_checked(index, &coord)?;
        visitor(&coord, addr, index.none_chunk_size);
        none_increment_scaled_coord(&mut coord, &index.none_chunks_per_dim);
    }
    Ok(())
}

/// Iterate over the entries of a dataset.
#[allow(non_snake_case)]
#[deprecated(note = "use H5D__none_idx_iterate_with to visit implicit chunks without collecting")]
pub fn H5D__none_idx_iterate(index: &ChunkIndexState) -> Result<Vec<ChunkInfo>> {
    let total = none_total_chunks(&index.none_chunks_per_dim)?;
    let total_usize = usize::try_from(total)
        .map_err(|_| Error::InvalidFormat("none chunk count exceeds usize".into()))?;
    let mut out = Vec::with_capacity(total_usize);
    H5D__none_idx_iterate_with(index, |coord, addr, size| {
        out.push(ChunkInfo {
            coord: coord.to_vec(),
            addr,
            size,
            filter_mask: 0,
        });
    })?;
    Ok(out)
}

/// Iterate over the entries of a dataset.
#[allow(non_snake_case)]
#[deprecated(note = "use H5D__none_idx_iterate_with to visit implicit chunks without collecting")]
pub fn H5D__none_idx_iterate_checked(index: &ChunkIndexState) -> Result<Vec<ChunkInfo>> {
    let total = none_total_chunks(&index.none_chunks_per_dim)?;
    let total_usize = usize::try_from(total)
        .map_err(|_| Error::InvalidFormat("none chunk count exceeds usize".into()))?;
    let mut out = Vec::with_capacity(total_usize);
    H5D__none_idx_iterate_with(index, |coord, addr, size| {
        out.push(ChunkInfo {
            coord: coord.to_vec(),
            addr,
            size,
            filter_mask: 0,
        });
    })?;
    Ok(out)
}

/// Remove an entry from a dataset.
#[allow(non_snake_case)]
pub fn H5D__none_idx_remove(_index: &mut ChunkIndexState, _coord: &[u64]) -> Option<ChunkInfo> {
    None
}

/// Delete a dataset.
#[allow(non_snake_case)]
pub fn H5D__none_idx_delete(_index: &mut ChunkIndexState) {}

/// Return a deep copy of a dataset.
#[allow(non_snake_case)]
#[deprecated(note = "use H5D__chunk_index_copy_setup_into to reuse caller-provided storage")]
pub fn H5D__none_idx_copy_setup(index: &ChunkIndexState) -> ChunkIndexState {
    index.clone()
}

/// Reset a dataset to its default state.
#[allow(non_snake_case)]
pub fn H5D__none_idx_reset(index: &mut ChunkIndexState) {
    *index = ChunkIndexState::default();
}

/// Render a dataset for debug output.
#[allow(non_snake_case)]
pub fn H5D__none_idx_dump_to<W: fmt::Write>(index: &ChunkIndexState, out: &mut W) -> fmt::Result {
    write!(
        out,
        "none_idx(open={}, allocated={}, base={:?}, chunk_size={}, chunks_per_dim={:?})",
        index.open,
        H5D__none_idx_is_space_alloc(index),
        index.none_base_addr,
        index.none_chunk_size,
        index.none_chunks_per_dim
    )
}

/// Render a dataset for debug output.
#[allow(non_snake_case)]
#[deprecated(note = "use H5D__none_idx_dump_to to write into caller-provided output")]
pub fn H5D__none_idx_dump(index: &ChunkIndexState) -> String {
    let mut out = String::new();
    let _ = H5D__none_idx_dump_to(index, &mut out);
    out
}

/// Initialize the dataset subsystem.
#[allow(non_snake_case)]
pub fn H5D_init() -> bool {
    H5D__init_package()
}

/// Initialize the dataset package.
#[allow(non_snake_case)]
pub fn H5D__init_package() -> bool {
    true
}

/// Terminate the dataset package and release its resources.
#[allow(non_snake_case)]
pub fn H5D_top_term_package() {}

/// Terminate the dataset package and release its resources.
#[allow(non_snake_case)]
pub fn H5D_term_package() {}

/// Close callback for dataset objects.
#[allow(non_snake_case)]
pub fn H5D__close_cb(_dataset: DatasetApi) {}

/// Dataset operation: get space status.
#[allow(non_snake_case)]
pub fn H5D__get_space_status(dataset: &DatasetApi) -> bool {
    !dataset.raw.is_empty()
}

/// Dataset operation: new.
#[allow(non_snake_case)]
pub fn H5D__new(name: Option<String>, extent: Vec<u64>) -> DatasetApi {
    H5D__create_api_common(name, extent)
}

/// Initialize the dataset subsystem.
#[allow(non_snake_case)]
pub fn H5D__init_type(element_size: usize) -> DatasetTypeInfo {
    H5D__typeinfo_init(element_size, element_size)
}

/// Dataset operation: cache dataspace info.
#[allow(non_snake_case)]
pub fn H5D__cache_dataspace_info_slice(dataset: &DatasetApi) -> &[u64] {
    &dataset.extent
}

/// Dataset operation: cache dataspace info.
#[allow(non_snake_case)]
#[deprecated(note = "use H5D__cache_dataspace_info_slice to borrow the dataspace extent")]
pub fn H5D__cache_dataspace_info(dataset: &DatasetApi) -> Vec<u64> {
    H5D__cache_dataspace_info_slice(dataset).to_vec()
}

/// Initialize the dataset subsystem.
#[allow(non_snake_case)]
pub fn H5D__init_space(dataset: &mut DatasetApi, extent: Vec<u64>) {
    dataset.extent = extent;
}

/// Dataset operation: use minimized dset headers.
#[allow(non_snake_case)]
pub fn H5D__use_minimized_dset_headers(dataset: &DatasetApi) -> bool {
    dataset.raw.len() < 64 * 1024
}

/// Dataset operation: calculate minimum header size.
#[allow(non_snake_case)]
pub fn H5D__calculate_minimum_header_size(dataset: &DatasetApi) -> usize {
    H5D__layout_meta_size(dataset)
}

/// Dataset operation: calculate minimum header size checked.
#[allow(non_snake_case)]
pub fn H5D__calculate_minimum_header_size_checked(dataset: &DatasetApi) -> Result<usize> {
    H5D__layout_meta_size_checked(dataset)
}

/// Dataset operation: prepare minimized oh.
#[allow(non_snake_case)]
pub fn H5D__prepare_minimized_oh_into(dataset: &DatasetApi, out: &mut Vec<u8>) -> Result<()> {
    H5D__layout_oh_create_into(dataset, out)
}

/// Dataset operation: prepare minimized oh.
#[allow(non_snake_case)]
#[deprecated(note = "use H5D__prepare_minimized_oh_into to reuse caller-provided storage")]
pub fn H5D__prepare_minimized_oh(dataset: &DatasetApi) -> Vec<u8> {
    let mut out = Vec::new();
    let _ = H5D__prepare_minimized_oh_into(dataset, &mut out);
    out
}

/// Update a dataset.
#[allow(non_snake_case)]
pub fn H5D__update_oh_info(dataset: &mut DatasetApi, extent: Vec<u64>) {
    dataset.extent = extent;
}

/// Dataset operation: build file prefix.
#[allow(non_snake_case)]
pub fn H5D__build_file_prefix_slice(path: &str) -> &str {
    path.rsplit_once('/').map_or("", |(prefix, _)| prefix)
}

/// Dataset operation: build file prefix.
#[allow(non_snake_case)]
#[deprecated(note = "use H5D__build_file_prefix_slice to borrow the prefix")]
pub fn H5D__build_file_prefix(path: &str) -> String {
    H5D__build_file_prefix_slice(path).to_owned()
}

/// Create a new dataset.
#[allow(non_snake_case)]
pub fn H5D__create(name: Option<String>, extent: Vec<u64>) -> DatasetApi {
    H5D__create_api_common(name, extent)
}

/// Open a dataset.
#[allow(non_snake_case)]
pub fn H5D_open(dataset: &DatasetApi) -> DatasetApi {
    dataset.clone()
}

/// Flush the dataset to storage.
#[allow(non_snake_case)]
pub fn H5D__append_flush_setup(_dataset: &mut DatasetApi) {}

/// Open a dataset.
#[allow(non_snake_case)]
pub fn H5D__open_oid(dataset: &DatasetApi) -> DatasetApi {
    dataset.clone()
}

/// Close a dataset.
#[allow(non_snake_case)]
pub fn H5D_close(dataset: DatasetApi) {
    H5Dclose(dataset);
}

/// Refresh the dataset from storage.
#[allow(non_snake_case)]
pub fn H5D_mult_refresh_close(_datasets: &mut [DatasetApi]) {}

/// Refresh the dataset from storage.
#[allow(non_snake_case)]
pub fn H5D_mult_refresh_reopen_into(datasets: &[DatasetApi], out: &mut Vec<DatasetApi>) {
    out.clear();
    out.extend_from_slice(datasets);
}

/// Refresh the dataset from storage.
#[allow(non_snake_case)]
#[deprecated(note = "use H5D_mult_refresh_reopen_into to reuse caller-provided storage")]
pub fn H5D_mult_refresh_reopen(datasets: &[DatasetApi]) -> Vec<DatasetApi> {
    let mut out = Vec::new();
    H5D_mult_refresh_reopen_into(datasets, &mut out);
    out
}

/// Dataset operation: oloc.
#[allow(non_snake_case)]
pub fn H5D_oloc(dataset: &DatasetApi) -> Option<&str> {
    dataset.name.as_deref()
}

/// Dataset operation: nameof.
#[allow(non_snake_case)]
pub fn H5D_nameof(dataset: &DatasetApi) -> Option<&str> {
    dataset.name.as_deref()
}

/// Allocate storage for a dataset.
#[allow(non_snake_case)]
pub fn H5D__alloc_storage(dataset: &mut DatasetApi, size: usize) {
    dataset.raw.resize(size, 0);
}

/// Initialize the dataset subsystem.
#[allow(non_snake_case)]
pub fn H5D__init_storage(dataset: &mut DatasetApi) {
    dataset.raw.clear();
}

/// Dataset operation: get storage size.
#[allow(non_snake_case)]
pub fn H5D__get_storage_size(dataset: &DatasetApi) -> usize {
    dataset.raw.len()
}

/// Return the offset of a dataset.
#[allow(non_snake_case)]
pub fn H5D__get_offset(_dataset: &DatasetApi) -> Option<u64> {
    Some(0)
}

/// Allocate storage for a dataset.
#[allow(non_snake_case)]
pub fn H5D__vlen_get_buf_size_alloc(size: usize) -> usize {
    size
}

/// Dataset operation: vlen get buf size cb.
#[allow(non_snake_case)]
pub fn H5D__vlen_get_buf_size_cb(value: &[u8]) -> usize {
    value.len()
}

/// Dataset operation: vlen get buf size.
#[allow(non_snake_case)]
pub fn H5D__vlen_get_buf_size(values: &[Vec<u8>]) -> usize {
    H5D__vlen_get_buf_size_slices(values.iter().map(Vec::as_slice))
}

/// Dataset operation: vlen get buf size checked.
#[allow(non_snake_case)]
pub fn H5D__vlen_get_buf_size_checked(values: &[Vec<u8>]) -> Result<usize> {
    H5D__vlen_get_buf_size_slices_checked(values.iter().map(Vec::as_slice))
}

/// Dataset operation: vlen get buf size for borrowed values.
#[allow(non_snake_case)]
pub fn H5D__vlen_get_buf_size_slices<'a, I>(values: I) -> usize
where
    I: IntoIterator<Item = &'a [u8]>,
{
    H5D__vlen_get_buf_size_slices_checked(values).unwrap_or(usize::MAX)
}

/// Dataset operation: vlen get buf size checked for borrowed values.
#[allow(non_snake_case)]
pub fn H5D__vlen_get_buf_size_slices_checked<'a, I>(values: I) -> Result<usize>
where
    I: IntoIterator<Item = &'a [u8]>,
{
    values.into_iter().try_fold(0usize, |acc, value| {
        acc.checked_add(value.len()).ok_or_else(|| {
            Error::InvalidFormat("variable-length dataset buffer size overflow".into())
        })
    })
}

/// Dataset operation: vlen get buf size gen cb.
#[allow(non_snake_case)]
pub fn H5D__vlen_get_buf_size_gen_cb(value: &[u8]) -> usize {
    H5D__vlen_get_buf_size_cb(value)
}

/// Dataset operation: vlen get buf size gen.
#[allow(non_snake_case)]
pub fn H5D__vlen_get_buf_size_gen(values: &[Vec<u8>]) -> usize {
    H5D__vlen_get_buf_size_slices(values.iter().map(Vec::as_slice))
}

/// Dataset operation: vlen get buf size gen checked.
#[allow(non_snake_case)]
pub fn H5D__vlen_get_buf_size_gen_checked(values: &[Vec<u8>]) -> Result<usize> {
    H5D__vlen_get_buf_size_slices_checked(values.iter().map(Vec::as_slice))
}

/// Flush the dataset to storage.
#[allow(non_snake_case)]
pub fn H5D__flush_sieve_buf(_dataset: &mut DatasetApi) {}

/// Flush the dataset to storage.
#[allow(non_snake_case)]
pub fn H5D__flush_real(_dataset: &mut DatasetApi) {}

/// Flush the dataset to storage.
#[allow(non_snake_case)]
pub fn H5D__flush(dataset: &mut DatasetApi) {
    H5D__flush_real(dataset);
}

/// Convert a dataset.
#[allow(non_snake_case)]
pub fn H5D__format_convert(dataset: &mut DatasetApi) {
    H5Dformat_convert(dataset);
}

/// Dataset operation: mark.
#[allow(non_snake_case)]
pub fn H5D__mark(dataset: &mut DatasetApi, marked: bool) {
    if marked {
        dataset.raw.shrink_to_fit();
    }
}

/// Flush the dataset to storage.
#[allow(non_snake_case)]
pub fn H5D__flush_all_cb(dataset: &mut DatasetApi) {
    H5D__flush(dataset);
}

/// Flush the dataset to storage.
#[allow(non_snake_case)]
pub fn H5D_flush_all(datasets: &mut [DatasetApi]) {
    for dataset in datasets {
        H5D__flush_all_cb(dataset);
    }
}

/// Return the creation property list for a dataset.
#[allow(non_snake_case)]
pub fn H5D_get_create_plist(_dataset: &DatasetApi) -> HashMap<String, String> {
    HashMap::new()
}

/// Return the access property list for a dataset.
#[allow(non_snake_case)]
pub fn H5D_get_access_plist(_dataset: &DatasetApi) -> HashMap<String, String> {
    HashMap::new()
}

/// Dataset operation: get space.
#[allow(non_snake_case)]
pub fn H5D__get_space(dataset: &DatasetApi) -> &[u64] {
    &dataset.extent
}

/// Dataset operation: get type.
#[allow(non_snake_case)]
pub fn H5D__get_type(element_size: usize) -> DatasetTypeInfo {
    H5D__init_type(element_size)
}

/// Allocate storage for a dataset.
#[allow(non_snake_case)]
pub fn H5D__contig_alloc(storage: &mut ContiguousStorage, size: usize) {
    storage.data.resize(size, 0);
    storage.allocated = true;
}

/// Delete a dataset.
#[allow(non_snake_case)]
pub fn H5D__contig_delete(storage: &mut ContiguousStorage) {
    storage.data.clear();
    storage.allocated = false;
}

/// Dataset operation: contig check.
#[allow(non_snake_case)]
pub fn H5D__contig_check(storage: &ContiguousStorage) -> bool {
    storage.allocated
}

/// Dataset operation: contig construct.
#[allow(non_snake_case)]
pub fn H5D__contig_construct(data: Vec<u8>) -> ContiguousStorage {
    ContiguousStorage {
        addr: Some(0),
        data,
        cached: false,
        allocated: true,
    }
}

/// Initialize the dataset subsystem.
#[allow(non_snake_case)]
pub fn H5D__contig_init(storage: &mut ContiguousStorage) {
    storage.allocated = true;
}

/// Allocate storage for a dataset.
#[allow(non_snake_case)]
pub fn H5D__contig_is_space_alloc(storage: &ContiguousStorage) -> bool {
    storage.allocated
}

/// Dataset operation: contig is data cached.
#[allow(non_snake_case)]
pub fn H5D__contig_is_data_cached(storage: &ContiguousStorage) -> bool {
    storage.cached
}

/// Initialize the dataset subsystem.
#[allow(non_snake_case)]
pub fn H5D__contig_io_init(storage: &mut ContiguousStorage) {
    storage.cached = true;
}

/// Initialize the dataset subsystem.
#[allow(non_snake_case)]
pub fn H5D__contig_mdio_init(_storage: &mut ContiguousStorage) {}

/// Dataset operation: contig may use select io.
#[allow(non_snake_case)]
pub fn H5D__contig_may_use_select_io(spans: &[(usize, usize)]) -> bool {
    spans.len() > 1
}

/// Write to a dataset.
#[allow(non_snake_case)]
pub fn H5D__contig_write_one(
    storage: &mut ContiguousStorage,
    offset: usize,
    data: &[u8],
) -> Result<()> {
    H5D__contig_writevv_slices(storage, &[(offset, data)])
}

/// Dataset operation: contig readvv sieve cb.
#[allow(non_snake_case)]
pub fn H5D__contig_readvv_sieve_cb_slice(data: &[u8]) -> &[u8] {
    data
}

/// Dataset operation: contig readvv sieve cb.
#[allow(non_snake_case)]
#[deprecated(note = "use H5D__contig_readvv_sieve_cb_slice to borrow contiguous data")]
pub fn H5D__contig_readvv_sieve_cb(data: &[u8]) -> Vec<u8> {
    H5D__contig_readvv_sieve_cb_slice(data).to_vec()
}

/// Dataset operation: contig readvv cb.
#[allow(non_snake_case)]
pub fn H5D__contig_readvv_cb_slice(data: &[u8]) -> &[u8] {
    data
}

/// Dataset operation: contig readvv cb.
#[allow(non_snake_case)]
#[deprecated(note = "use H5D__contig_readvv_cb_slice to borrow contiguous data")]
pub fn H5D__contig_readvv_cb(data: &[u8]) -> Vec<u8> {
    H5D__contig_readvv_cb_slice(data).to_vec()
}

/// Dataset operation: contig readvv.
#[allow(non_snake_case)]
#[deprecated(note = "use H5D__contig_readvv_into to fill caller-provided buffers")]
pub fn H5D__contig_readvv(
    storage: &ContiguousStorage,
    spans: &[(usize, usize)],
) -> Result<Vec<Vec<u8>>> {
    let mut out: Vec<Vec<u8>> = spans.iter().map(|&(_, len)| vec![0; len]).collect();
    H5D__contig_readvv_into(storage, spans, &mut out)?;
    Ok(out)
}

/// Dataset operation: contig readvv into caller-provided buffers.
#[allow(non_snake_case)]
pub fn H5D__contig_readvv_into(
    storage: &ContiguousStorage,
    spans: &[(usize, usize)],
    out: &mut [Vec<u8>],
) -> Result<()> {
    H5D__scatter_file_into(&storage.data, spans, out)
}

/// Dataset operation: contig readvv into caller-provided borrowed buffers.
#[allow(non_snake_case)]
pub fn H5D__contig_readvv_into_slices(
    storage: &ContiguousStorage,
    spans: &[(usize, usize)],
    out: &mut [&mut [u8]],
) -> Result<()> {
    H5D__scatter_file_into_slices(&storage.data, spans, out)
}

/// Dataset operation: contig readvv borrowed visitor.
#[allow(non_snake_case)]
pub fn H5D__contig_readvv_with<F>(
    storage: &ContiguousStorage,
    spans: &[(usize, usize)],
    visitor: F,
) -> Result<()>
where
    F: FnMut(&[u8]),
{
    H5D__scatter_file_with(&storage.data, spans, visitor)
}

/// Dataset operation: contig writevv sieve cb.
#[allow(non_snake_case)]
pub fn H5D__contig_writevv_sieve_cb(data: &[u8]) -> usize {
    data.len()
}

/// Dataset operation: contig writevv cb.
#[allow(non_snake_case)]
pub fn H5D__contig_writevv_cb(data: &[u8]) -> usize {
    data.len()
}

/// Dataset operation: contig writevv.
#[allow(non_snake_case)]
#[deprecated(note = "use H5D__contig_writevv_slices to write borrowed spans")]
pub fn H5D__contig_writevv(
    storage: &mut ContiguousStorage,
    spans: &[(usize, Vec<u8>)],
) -> Result<()> {
    for (offset, data) in spans {
        let end = offset
            .checked_add(data.len())
            .ok_or_else(|| Error::InvalidFormat("contiguous write overflow".into()))?;
        if storage.data.len() < end {
            storage.data.resize(end, 0);
        }
        storage.data[*offset..end].copy_from_slice(data);
    }
    storage.allocated = true;
    Ok(())
}

/// Dataset operation: contig writevv from borrowed spans.
#[allow(non_snake_case)]
pub fn H5D__contig_writevv_slices(
    storage: &mut ContiguousStorage,
    spans: &[(usize, &[u8])],
) -> Result<()> {
    for (offset, data) in spans {
        let end = offset
            .checked_add(data.len())
            .ok_or_else(|| Error::InvalidFormat("contiguous write overflow".into()))?;
        if storage.data.len() < end {
            storage.data.resize(end, 0);
        }
        storage.data[*offset..end].copy_from_slice(data);
    }
    storage.allocated = true;
    Ok(())
}

/// Flush the dataset to storage.
#[allow(non_snake_case)]
pub fn H5D__contig_flush(_storage: &mut ContiguousStorage) {}

/// Dataset operation: contig io term.
#[allow(non_snake_case)]
pub fn H5D__contig_io_term(storage: &mut ContiguousStorage) {
    storage.cached = false;
}

/// Return a deep copy of a dataset.
#[allow(non_snake_case)]
#[deprecated(note = "use H5D__contig_copy_into to reuse caller-provided storage")]
pub fn H5D__contig_copy(storage: &ContiguousStorage) -> ContiguousStorage {
    storage.clone()
}

/// Copy contiguous storage into caller-provided storage.
#[allow(non_snake_case)]
pub fn H5D__contig_copy_into(storage: &ContiguousStorage, out: &mut ContiguousStorage) {
    out.clone_from(storage);
}

/// Dataset operation: create1.
#[allow(non_snake_case)]
pub fn H5Dcreate1(extent: Vec<u64>) -> DatasetApi {
    H5Dcreate_anon(extent)
}

/// Dataset operation: open1.
#[allow(non_snake_case)]
pub fn H5Dopen1(dataset: &DatasetApi) -> DatasetApi {
    H5D_open(dataset)
}

/// Dataset operation: extend.
#[allow(non_snake_case)]
pub fn H5Dextend(dataset: &mut DatasetApi, extent: Vec<u64>) {
    dataset.extent = extent;
}

/// Dataset operation: vlen reclaim.
#[allow(non_snake_case)]
pub fn H5Dvlen_reclaim(values: &mut Vec<Vec<u8>>) {
    values.clear();
}

/// Read from a dataset.
#[allow(non_snake_case)]
#[deprecated(note = "use H5Dread_chunk2_slice or H5Dread_chunk2_into to avoid allocating")]
pub fn H5Dread_chunk1(dataset: &DatasetApi, offset: usize, len: usize) -> Result<Vec<u8>> {
    Ok(H5Dread_chunk2_slice(dataset, offset, len)?.to_vec())
}

/// Dataset operation: fill.
#[allow(non_snake_case)]
pub fn H5D__fill(dst: &mut [u8], fill: &[u8]) {
    if fill.is_empty() {
        return;
    }
    for chunk in dst.chunks_mut(fill.len()) {
        let n = chunk.len();
        chunk.copy_from_slice(&fill[..n]);
    }
}

/// Initialize the dataset subsystem.
#[allow(non_snake_case)]
pub fn H5D__fill_init(value: Vec<u8>) -> FillState {
    FillState {
        value,
        initialized: true,
    }
}

/// Dataset operation: fill refill vl.
#[allow(non_snake_case)]
pub fn H5D__fill_refill_vl(state: &FillState, dst: &mut Vec<u8>) {
    dst.clear();
    dst.extend_from_slice(&state.value);
}

/// Dataset operation: fill release.
#[allow(non_snake_case)]
pub fn H5D__fill_release(state: &mut FillState) {
    state.value.clear();
    state.initialized = false;
}

/// Dataset operation: fill term.
#[allow(non_snake_case)]
pub fn H5D__fill_term(state: &mut FillState) {
    H5D__fill_release(state);
}

pub mod explicit_index_wrappers {
    use super::*;

    /// Dataset operation: earray crt context.
    #[allow(non_snake_case)]
    pub fn H5D__earray_crt_context() -> ChunkIndexState {
        ChunkIndexState::default()
    }

    /// Dataset operation: earray dst context.
    #[allow(non_snake_case)]
    pub fn H5D__earray_dst_context(index: &mut ChunkIndexState) {
        index.open = false;
    }

    /// Dataset operation: earray fill.
    #[allow(non_snake_case)]
    pub fn H5D__earray_fill(size: usize) -> Vec<u8> {
        vec![0; size]
    }

    /// Encode a dataset to its on-disk representation.
    #[allow(non_snake_case)]
    pub fn H5D__earray_encode(index: &ChunkIndexState) -> Result<Vec<u8>> {
        chunk_index_encode_count(index)
    }

    /// Return a debug-friendly representation of a dataset.
    #[allow(non_snake_case)]
    #[deprecated(note = "use H5D__chunk_index_dump_to to write into caller-provided output")]
    pub fn H5D__earray_debug(index: &ChunkIndexState) -> String {
        let mut out = String::new();
        let _ = H5D__chunk_index_dump_to("earray", index, &mut out);
        out
    }

    /// Decode a dataset from its on-disk representation.
    #[allow(non_snake_case)]
    pub fn H5D__earray_filt_decode(bytes: &[u8]) -> Result<ChunkIndexState> {
        chunk_index_decode_count_image(bytes, "earray")
    }

    /// Dataset operation: earray idx depend.
    #[allow(non_snake_case)]
    pub fn H5D__earray_idx_depend(index: &ChunkIndexState) -> usize {
        index.entries.len()
    }

    /// Initialize the dataset subsystem.
    #[allow(non_snake_case)]
    pub fn H5D__earray_idx_init() -> ChunkIndexState {
        ChunkIndexState::default()
    }

    /// Create a new dataset.
    #[allow(non_snake_case)]
    pub fn H5D__earray_idx_create(index: &mut ChunkIndexState) {
        index.open = true;
    }

    /// Open a dataset.
    #[allow(non_snake_case)]
    pub fn H5D__earray_idx_open(index: &mut ChunkIndexState) {
        index.open = true;
    }

    /// Close a dataset.
    #[allow(non_snake_case)]
    pub fn H5D__earray_idx_close(index: &mut ChunkIndexState) {
        index.open = false;
    }

    /// Open a dataset.
    #[allow(non_snake_case)]
    pub fn H5D__earray_idx_is_open(index: &ChunkIndexState) -> bool {
        index.open
    }

    /// Allocate storage for a dataset.
    #[allow(non_snake_case)]
    pub fn H5D__earray_idx_is_space_alloc(index: &ChunkIndexState) -> bool {
        index.space_allocated
    }

    /// Insert an entry into a dataset.
    #[allow(non_snake_case)]
    pub fn H5D__earray_idx_insert(
        index: &mut ChunkIndexState,
        coord: Vec<u64>,
        addr: u64,
        size: usize,
    ) {
        chunk_index_insert(index, coord, addr, size);
    }

    /// Dataset operation: earray idx get addr.
    #[allow(non_snake_case)]
    pub fn H5D__earray_idx_get_addr(index: &ChunkIndexState, coord: &[u64]) -> Option<u64> {
        chunk_index_get_addr(index, coord)
    }

    /// Dataset operation: earray idx load metadata.
    #[allow(non_snake_case)]
    pub fn H5D__earray_idx_load_metadata(index: &mut ChunkIndexState) {
        index.metadata_loaded = true;
    }

    /// Dataset operation: earray idx resize.
    #[allow(non_snake_case)]
    pub fn H5D__earray_idx_resize(index: &mut ChunkIndexState, additional: usize) {
        index.entries.reserve(additional);
    }

    /// Iterate over the entries of a dataset.
    #[allow(non_snake_case)]
    #[deprecated(note = "use H5D__chunk_index_iterate to borrow chunk info entries")]
    pub fn H5D__earray_idx_iterate_cb(info: &ChunkInfo) -> ChunkInfo {
        info.clone()
    }

    /// Iterate over the entries of a dataset.
    #[allow(non_snake_case)]
    #[deprecated(note = "use H5D__chunk_index_iterate to avoid cloning chunk info entries")]
    pub fn H5D__earray_idx_iterate(index: &ChunkIndexState) -> Vec<ChunkInfo> {
        H5D__chunk_index_iterate(index).cloned().collect()
    }

    /// Remove an entry from a dataset.
    #[allow(non_snake_case)]
    pub fn H5D__earray_idx_remove(index: &mut ChunkIndexState, coord: &[u64]) -> Option<ChunkInfo> {
        chunk_index_remove(index, coord)
    }

    /// Delete a dataset.
    #[allow(non_snake_case)]
    #[deprecated(note = "use H5D__chunk_index_iterate to borrow chunk info entries")]
    pub fn H5D__earray_idx_delete_cb(info: &ChunkInfo) -> ChunkInfo {
        info.clone()
    }

    /// Delete a dataset.
    #[allow(non_snake_case)]
    pub fn H5D__earray_idx_delete(index: &mut ChunkIndexState) {
        index.entries.clear();
        index.space_allocated = false;
    }

    /// Return a deep copy of a dataset.
    #[allow(non_snake_case)]
    #[deprecated(note = "use H5D__chunk_index_copy_setup_into to reuse caller-provided storage")]
    pub fn H5D__earray_idx_copy_setup(index: &ChunkIndexState) -> ChunkIndexState {
        index.clone()
    }

    /// Reset a dataset to its default state.
    #[allow(non_snake_case)]
    pub fn H5D__earray_idx_reset(index: &mut ChunkIndexState) {
        *index = ChunkIndexState::default();
    }

    /// Render a dataset for debug output into caller-provided output.
    #[allow(non_snake_case)]
    pub fn H5D__earray_idx_dump_to<W: fmt::Write>(
        index: &ChunkIndexState,
        out: &mut W,
    ) -> fmt::Result {
        let addr = index
            .none_base_addr
            .or_else(|| index.entries.values().map(|chunk| chunk.addr).min());
        match addr {
            Some(addr) => writeln!(out, "    Address: {addr}"),
            None => writeln!(out, "    Address: undefined"),
        }
    }

    /// Render a dataset for debug output.
    #[allow(non_snake_case)]
    #[deprecated(note = "use H5D__earray_idx_dump_to to write into caller-provided output")]
    pub fn H5D__earray_idx_dump(index: &ChunkIndexState) -> String {
        let mut out = String::new();
        let _ = H5D__earray_idx_dump_to(index, &mut out);
        out
    }

    /// Dataset operation: earray idx dest.
    #[allow(non_snake_case)]
    pub fn H5D__earray_idx_dest(index: &mut ChunkIndexState) {
        index.entries.clear();
        index.open = false;
    }

    /// Dataset operation: bt2 crt context.
    #[allow(non_snake_case)]
    pub fn H5D__bt2_crt_context() -> ChunkIndexState {
        ChunkIndexState::default()
    }

    /// Dataset operation: bt2 dst context.
    #[allow(non_snake_case)]
    pub fn H5D__bt2_dst_context(index: &mut ChunkIndexState) {
        index.entries.clear();
    }

    /// Dataset operation: bt2 store.
    #[allow(non_snake_case)]
    pub fn H5D__bt2_store(index: &ChunkIndexState) -> Result<Vec<u8>> {
        chunk_index_encode_count(index)
    }

    /// Encode a dataset to its on-disk representation.
    #[allow(non_snake_case)]
    pub fn H5D__bt2_unfilt_encode(index: &ChunkIndexState) -> Result<Vec<u8>> {
        H5D__bt2_store(index)
    }

    /// Decode a dataset from its on-disk representation.
    #[allow(non_snake_case)]
    pub fn H5D__bt2_unfilt_decode(bytes: &[u8]) -> Result<ChunkIndexState> {
        chunk_index_decode_count_image(bytes, "bt2")
    }

    /// Return a debug-friendly representation of a dataset.
    #[allow(non_snake_case)]
    #[deprecated(note = "use H5D__chunk_index_dump_to to write into caller-provided output")]
    pub fn H5D__bt2_unfilt_debug(index: &ChunkIndexState) -> String {
        let mut out = String::new();
        let _ = H5D__chunk_index_dump_to("bt2", index, &mut out);
        out
    }

    /// Initialize the dataset subsystem.
    #[allow(non_snake_case)]
    pub fn H5D__bt2_idx_init() -> ChunkIndexState {
        ChunkIndexState::default()
    }

    /// Dataset operation: btree2 idx depend.
    #[allow(non_snake_case)]
    pub fn H5D__btree2_idx_depend(index: &ChunkIndexState) -> usize {
        index.entries.len()
    }

    /// Create a new dataset.
    #[allow(non_snake_case)]
    pub fn H5D__bt2_idx_create(index: &mut ChunkIndexState) {
        index.open = true;
    }

    /// Open a dataset.
    #[allow(non_snake_case)]
    pub fn H5D__bt2_idx_open(index: &mut ChunkIndexState) {
        index.open = true;
    }

    /// Close a dataset.
    #[allow(non_snake_case)]
    pub fn H5D__bt2_idx_close(index: &mut ChunkIndexState) {
        index.open = false;
    }

    /// Open a dataset.
    #[allow(non_snake_case)]
    pub fn H5D__bt2_idx_is_open(index: &ChunkIndexState) -> bool {
        index.open
    }

    /// Allocate storage for a dataset.
    #[allow(non_snake_case)]
    pub fn H5D__bt2_idx_is_space_alloc(index: &ChunkIndexState) -> bool {
        index.space_allocated
    }

    /// Insert an entry into a dataset.
    #[allow(non_snake_case)]
    pub fn H5D__bt2_idx_insert(
        index: &mut ChunkIndexState,
        coord: Vec<u64>,
        addr: u64,
        size: usize,
    ) {
        chunk_index_insert(index, coord, addr, size);
    }

    /// Dataset operation: bt2 idx get addr.
    #[allow(non_snake_case)]
    pub fn H5D__bt2_idx_get_addr(index: &ChunkIndexState, coord: &[u64]) -> Option<u64> {
        chunk_index_get_addr(index, coord)
    }

    /// Dataset operation: bt2 idx load metadata.
    #[allow(non_snake_case)]
    pub fn H5D__bt2_idx_load_metadata(index: &mut ChunkIndexState) {
        index.metadata_loaded = true;
    }

    /// Iterate over the entries of a dataset.
    #[allow(non_snake_case)]
    #[deprecated(note = "use H5D__chunk_index_iterate to borrow chunk info entries")]
    pub fn H5D__bt2_idx_iterate_cb(info: &ChunkInfo) -> ChunkInfo {
        info.clone()
    }

    /// Iterate over the entries of a dataset.
    #[allow(non_snake_case)]
    #[deprecated(note = "use H5D__chunk_index_iterate to avoid cloning chunk info entries")]
    pub fn H5D__bt2_idx_iterate(index: &ChunkIndexState) -> Vec<ChunkInfo> {
        H5D__chunk_index_iterate(index).cloned().collect()
    }

    /// Remove an entry from a dataset.
    #[allow(non_snake_case)]
    #[deprecated(note = "use H5D__chunk_index_iterate to borrow chunk info entries")]
    pub fn H5D__bt2_remove_cb(info: &ChunkInfo) -> ChunkInfo {
        info.clone()
    }

    /// Remove an entry from a dataset.
    #[allow(non_snake_case)]
    pub fn H5D__bt2_idx_remove(index: &mut ChunkIndexState, coord: &[u64]) -> Option<ChunkInfo> {
        chunk_index_remove(index, coord)
    }

    /// Delete a dataset.
    #[allow(non_snake_case)]
    pub fn H5D__bt2_idx_delete(index: &mut ChunkIndexState) {
        index.entries.clear();
        index.space_allocated = false;
    }

    /// Return a deep copy of a dataset.
    #[allow(non_snake_case)]
    #[deprecated(note = "use H5D__chunk_index_copy_setup_into to reuse caller-provided storage")]
    pub fn H5D__bt2_idx_copy_setup(index: &ChunkIndexState) -> ChunkIndexState {
        index.clone()
    }

    /// Dataset operation: bt2 idx size.
    #[allow(non_snake_case)]
    pub fn H5D__bt2_idx_size(index: &ChunkIndexState) -> usize {
        index.entries.len()
    }

    /// Reset a dataset to its default state.
    #[allow(non_snake_case)]
    pub fn H5D__bt2_idx_reset(index: &mut ChunkIndexState) {
        *index = ChunkIndexState::default();
    }

    /// Render a dataset for debug output.
    #[allow(non_snake_case)]
    #[deprecated(note = "use H5D__chunk_index_dump_to to write into caller-provided output")]
    pub fn H5D__bt2_idx_dump(index: &ChunkIndexState) -> String {
        let mut out = String::new();
        let _ = H5D__chunk_index_dump_to("bt2", index, &mut out);
        out
    }

    /// Dataset operation: bt2 idx dest.
    #[allow(non_snake_case)]
    pub fn H5D__bt2_idx_dest(index: &mut ChunkIndexState) {
        index.entries.clear();
        index.open = false;
    }

    /// Dataset operation: farray crt context.
    #[allow(non_snake_case)]
    pub fn H5D__farray_crt_context() -> ChunkIndexState {
        ChunkIndexState::default()
    }

    /// Dataset operation: farray dst context.
    #[allow(non_snake_case)]
    pub fn H5D__farray_dst_context(index: &mut ChunkIndexState) {
        index.entries.clear();
    }

    /// Dataset operation: farray fill.
    #[allow(non_snake_case)]
    pub fn H5D__farray_fill(size: usize) -> Vec<u8> {
        vec![0; size]
    }

    /// Encode a dataset to its on-disk representation.
    #[allow(non_snake_case)]
    pub fn H5D__farray_encode(index: &ChunkIndexState) -> Result<Vec<u8>> {
        chunk_index_encode_count(index)
    }

    /// Decode a dataset from its on-disk representation.
    #[allow(non_snake_case)]
    pub fn H5D__farray_decode(bytes: &[u8]) -> Result<ChunkIndexState> {
        chunk_index_decode_count_image(bytes, "farray")
    }

    /// Return a debug-friendly representation of a dataset.
    #[allow(non_snake_case)]
    #[deprecated(note = "use H5D__chunk_index_dump_to to write into caller-provided output")]
    pub fn H5D__farray_debug(index: &ChunkIndexState) -> String {
        let mut out = String::new();
        let _ = H5D__chunk_index_dump_to("farray", index, &mut out);
        out
    }

    /// Dataset operation: farray idx depend.
    #[allow(non_snake_case)]
    pub fn H5D__farray_idx_depend(index: &ChunkIndexState) -> usize {
        index.entries.len()
    }

    /// Initialize the dataset subsystem.
    #[allow(non_snake_case)]
    pub fn H5D__farray_idx_init() -> ChunkIndexState {
        ChunkIndexState::default()
    }

    /// Create a new dataset.
    #[allow(non_snake_case)]
    pub fn H5D__farray_idx_create(index: &mut ChunkIndexState) {
        index.open = true;
    }

    /// Open a dataset.
    #[allow(non_snake_case)]
    pub fn H5D__farray_idx_open(index: &mut ChunkIndexState) {
        index.open = true;
    }

    /// Close a dataset.
    #[allow(non_snake_case)]
    pub fn H5D__farray_idx_close(index: &mut ChunkIndexState) {
        index.open = false;
    }

    /// Open a dataset.
    #[allow(non_snake_case)]
    pub fn H5D__farray_idx_is_open(index: &ChunkIndexState) -> bool {
        index.open
    }

    /// Allocate storage for a dataset.
    #[allow(non_snake_case)]
    pub fn H5D__farray_idx_is_space_alloc(index: &ChunkIndexState) -> bool {
        index.space_allocated
    }

    /// Insert an entry into a dataset.
    #[allow(non_snake_case)]
    pub fn H5D__farray_idx_insert(
        index: &mut ChunkIndexState,
        coord: Vec<u64>,
        addr: u64,
        size: usize,
    ) {
        chunk_index_insert(index, coord, addr, size);
    }

    /// Dataset operation: farray idx get addr.
    #[allow(non_snake_case)]
    pub fn H5D__farray_idx_get_addr(index: &ChunkIndexState, coord: &[u64]) -> Option<u64> {
        chunk_index_get_addr(index, coord)
    }

    /// Dataset operation: farray idx load metadata.
    #[allow(non_snake_case)]
    pub fn H5D__farray_idx_load_metadata(index: &mut ChunkIndexState) {
        index.metadata_loaded = true;
    }

    /// Iterate over the entries of a dataset.
    #[allow(non_snake_case)]
    #[deprecated(note = "use H5D__chunk_index_iterate to borrow chunk info entries")]
    pub fn H5D__farray_idx_iterate_cb(info: &ChunkInfo) -> ChunkInfo {
        info.clone()
    }

    /// Iterate over the entries of a dataset.
    #[allow(non_snake_case)]
    #[deprecated(note = "use H5D__chunk_index_iterate to avoid cloning chunk info entries")]
    pub fn H5D__farray_idx_iterate(index: &ChunkIndexState) -> Vec<ChunkInfo> {
        H5D__chunk_index_iterate(index).cloned().collect()
    }

    /// Remove an entry from a dataset.
    #[allow(non_snake_case)]
    pub fn H5D__farray_idx_remove(index: &mut ChunkIndexState, coord: &[u64]) -> Option<ChunkInfo> {
        chunk_index_remove(index, coord)
    }

    /// Delete a dataset.
    #[allow(non_snake_case)]
    #[deprecated(note = "use H5D__chunk_index_iterate to borrow chunk info entries")]
    pub fn H5D__farray_idx_delete_cb(info: &ChunkInfo) -> ChunkInfo {
        info.clone()
    }

    /// Delete a dataset.
    #[allow(non_snake_case)]
    pub fn H5D__farray_idx_delete(index: &mut ChunkIndexState) {
        index.entries.clear();
        index.space_allocated = false;
    }

    /// Return a deep copy of a dataset.
    #[allow(non_snake_case)]
    #[deprecated(note = "use H5D__chunk_index_copy_setup_into to reuse caller-provided storage")]
    pub fn H5D__farray_idx_copy_setup(index: &ChunkIndexState) -> ChunkIndexState {
        index.clone()
    }

    /// Dataset operation: farray idx size.
    #[allow(non_snake_case)]
    pub fn H5D__farray_idx_size(index: &ChunkIndexState) -> usize {
        index.entries.len()
    }

    /// Reset a dataset to its default state.
    #[allow(non_snake_case)]
    pub fn H5D__farray_idx_reset(index: &mut ChunkIndexState) {
        *index = ChunkIndexState::default();
    }

    /// Render a dataset for debug output.
    #[allow(non_snake_case)]
    #[deprecated(note = "use H5D__chunk_index_dump_to to write into caller-provided output")]
    pub fn H5D__farray_idx_dump(index: &ChunkIndexState) -> String {
        let mut out = String::new();
        let _ = H5D__chunk_index_dump_to("farray", index, &mut out);
        out
    }

    /// Dataset operation: farray idx dest.
    #[allow(non_snake_case)]
    pub fn H5D__farray_idx_dest(index: &mut ChunkIndexState) {
        index.entries.clear();
        index.open = false;
    }

    /// Dataset operation: btree get shared.
    #[allow(non_snake_case)]
    pub fn H5D__btree_get_shared() -> ChunkIndexState {
        ChunkIndexState::default()
    }

    /// Dataset operation: btree cmp2.
    #[allow(non_snake_case)]
    pub fn H5D__btree_cmp2(left: &ChunkInfo, right: &ChunkInfo) -> std::cmp::Ordering {
        H5D__chunk_cmp_coll_fill_info(left, right)
    }

    /// Dataset operation: btree cmp3.
    #[allow(non_snake_case)]
    pub fn H5D__btree_cmp3(left: &ChunkInfo, right: &ChunkInfo) -> std::cmp::Ordering {
        H5D__chunk_cmp_coll_fill_info(left, right)
    }

    /// Remove an entry from a dataset.
    #[allow(non_snake_case)]
    pub fn H5D__btree_remove(index: &mut ChunkIndexState, coord: &[u64]) -> Option<ChunkInfo> {
        chunk_index_remove(index, coord)
    }

    /// Decode a dataset from its on-disk representation.
    #[allow(non_snake_case)]
    pub fn H5D__btree_decode_key(bytes: &[u8]) -> Result<ChunkIndexState> {
        chunk_index_decode_count_image(bytes, "btree")
    }

    /// Encode a dataset to its on-disk representation.
    #[allow(non_snake_case)]
    pub fn H5D__btree_encode_key(index: &ChunkIndexState) -> Result<Vec<u8>> {
        chunk_index_encode_count(index)
    }

    /// Return a debug-friendly representation of a dataset.
    #[allow(non_snake_case)]
    #[deprecated(note = "use H5D__chunk_index_dump_to to write into caller-provided output")]
    pub fn H5D__btree_debug_key(index: &ChunkIndexState) -> String {
        let mut out = String::new();
        let _ = H5D__chunk_index_dump_to("btree", index, &mut out);
        out
    }

    /// Free a dataset's in-memory resources.
    #[allow(non_snake_case)]
    pub fn H5D__btree_shared_free(index: &mut ChunkIndexState) {
        index.entries.clear();
    }

    /// Create a new dataset.
    #[allow(non_snake_case)]
    pub fn H5D__btree_shared_create() -> ChunkIndexState {
        ChunkIndexState::default()
    }

    /// Initialize the dataset subsystem.
    #[allow(non_snake_case)]
    pub fn H5D__btree_idx_init() -> ChunkIndexState {
        ChunkIndexState::default()
    }

    /// Close a dataset.
    #[allow(non_snake_case)]
    pub fn H5D__btree_idx_close(index: &mut ChunkIndexState) {
        index.open = false;
    }

    /// Open a dataset.
    #[allow(non_snake_case)]
    pub fn H5D__btree_idx_is_open(index: &ChunkIndexState) -> bool {
        index.open
    }

    /// Allocate storage for a dataset.
    #[allow(non_snake_case)]
    pub fn H5D__btree_idx_is_space_alloc(index: &ChunkIndexState) -> bool {
        index.space_allocated
    }

    /// Dataset operation: btree idx get addr.
    #[allow(non_snake_case)]
    pub fn H5D__btree_idx_get_addr(index: &ChunkIndexState, coord: &[u64]) -> Option<u64> {
        chunk_index_get_addr(index, coord)
    }

    /// Dataset operation: btree idx load metadata.
    #[allow(non_snake_case)]
    pub fn H5D__btree_idx_load_metadata(index: &mut ChunkIndexState) {
        index.metadata_loaded = true;
    }

    /// Iterate over the entries of a dataset.
    #[allow(non_snake_case)]
    #[deprecated(note = "use H5D__chunk_index_iterate to borrow chunk info entries")]
    pub fn H5D__btree_idx_iterate_cb(info: &ChunkInfo) -> ChunkInfo {
        info.clone()
    }

    /// Iterate over the entries of a dataset.
    #[allow(non_snake_case)]
    #[deprecated(note = "use H5D__chunk_index_iterate to avoid cloning chunk info entries")]
    pub fn H5D__btree_idx_iterate(index: &ChunkIndexState) -> Vec<ChunkInfo> {
        H5D__chunk_index_iterate(index).cloned().collect()
    }

    /// Remove an entry from a dataset.
    #[allow(non_snake_case)]
    pub fn H5D__btree_idx_remove(index: &mut ChunkIndexState, coord: &[u64]) -> Option<ChunkInfo> {
        chunk_index_remove(index, coord)
    }

    /// Delete a dataset.
    #[allow(non_snake_case)]
    pub fn H5D__btree_idx_delete(index: &mut ChunkIndexState) {
        index.entries.clear();
        index.space_allocated = false;
    }

    /// Return a deep copy of a dataset.
    #[allow(non_snake_case)]
    #[deprecated(note = "use H5D__chunk_index_copy_setup_into to reuse caller-provided storage")]
    pub fn H5D__btree_idx_copy_setup(index: &ChunkIndexState) -> ChunkIndexState {
        index.clone()
    }

    /// Return a deep copy of a dataset.
    #[allow(non_snake_case)]
    pub fn H5D__btree_idx_copy_shutdown(index: &mut ChunkIndexState) {
        index.open = false;
    }

    /// Dataset operation: btree idx size.
    #[allow(non_snake_case)]
    pub fn H5D__btree_idx_size(index: &ChunkIndexState) -> usize {
        index.entries.len()
    }

    /// Reset a dataset to its default state.
    #[allow(non_snake_case)]
    pub fn H5D__btree_idx_reset(index: &mut ChunkIndexState) {
        *index = ChunkIndexState::default();
    }

    /// Render a dataset for debug output.
    #[allow(non_snake_case)]
    #[deprecated(note = "use H5D__chunk_index_dump_to to write into caller-provided output")]
    pub fn H5D__btree_idx_dump(index: &ChunkIndexState) -> String {
        let mut out = String::new();
        let _ = H5D__chunk_index_dump_to("btree", index, &mut out);
        out
    }

    /// Dataset operation: btree idx dest.
    #[allow(non_snake_case)]
    pub fn H5D__btree_idx_dest(index: &mut ChunkIndexState) {
        index.entries.clear();
        index.open = false;
    }
}

/// Dataset operation: chunk disjoint.
#[allow(non_snake_case)]
pub fn H5D__chunk_disjoint(left: &ChunkInfo, right: &ChunkInfo) -> bool {
    left.coord != right.coord
}

/// Return a debug-friendly representation of a dataset.
#[allow(non_snake_case)]
#[deprecated(note = "use H5D__chunk_index_dump_to to write into caller-provided output")]
pub fn H5D_btree_debug(index: &ChunkIndexState) -> String {
    let mut out = String::new();
    let _ = H5D__chunk_index_dump_to("btree", index, &mut out);
    out
}

/// Dataset operation: select io.
#[allow(non_snake_case)]
#[deprecated(
    since = "0.4.0",
    note = "use H5D__select_io_into to fill caller-provided buffers"
)]
pub fn H5D__select_io(dataset: &DatasetApi, spans: &[(usize, usize)]) -> Result<Vec<Vec<u8>>> {
    let mut out: Vec<Vec<u8>> = spans.iter().map(|&(_, len)| vec![0; len]).collect();
    H5D__select_io_into(dataset, spans, &mut out)?;
    Ok(out)
}

/// Dataset operation: select io into caller-provided buffers.
#[allow(non_snake_case)]
pub fn H5D__select_io_into(
    dataset: &DatasetApi,
    spans: &[(usize, usize)],
    out: &mut [Vec<u8>],
) -> Result<()> {
    H5D__scatter_file_into(&dataset.raw, spans, out)
}

/// Dataset operation: select io into caller-provided borrowed buffers.
#[allow(non_snake_case)]
pub fn H5D__select_io_into_slices(
    dataset: &DatasetApi,
    spans: &[(usize, usize)],
    out: &mut [&mut [u8]],
) -> Result<()> {
    H5D__scatter_file_into_slices(&dataset.raw, spans, out)
}

/// Dataset operation: select io borrowed visitor.
#[allow(non_snake_case)]
pub fn H5D__select_io_with<F>(
    dataset: &DatasetApi,
    spans: &[(usize, usize)],
    visitor: F,
) -> Result<()>
where
    F: FnMut(&[u8]),
{
    H5D__scatter_file_with(&dataset.raw, spans, visitor)
}

/// Dataset operation: select io mem.
#[allow(non_snake_case)]
#[deprecated(
    since = "0.4.0",
    note = "use H5D_select_io_mem_into to fill caller-provided buffers"
)]
pub fn H5D_select_io_mem(src: &[u8], spans: &[(usize, usize)]) -> Result<Vec<Vec<u8>>> {
    let mut out: Vec<Vec<u8>> = spans.iter().map(|&(_, len)| vec![0; len]).collect();
    H5D_select_io_mem_into(src, spans, &mut out)?;
    Ok(out)
}

/// Dataset operation: select io mem into caller-provided buffers.
#[allow(non_snake_case)]
pub fn H5D_select_io_mem_into(
    src: &[u8],
    spans: &[(usize, usize)],
    out: &mut [Vec<u8>],
) -> Result<()> {
    H5D__scatter_mem_into(src, spans, out)
}

/// Dataset operation: select io mem into caller-provided borrowed buffers.
#[allow(non_snake_case)]
pub fn H5D_select_io_mem_into_slices(
    src: &[u8],
    spans: &[(usize, usize)],
    out: &mut [&mut [u8]],
) -> Result<()> {
    H5D__scatter_mem_into_slices(src, spans, out)
}

/// Dataset operation: select io mem borrowed visitor.
#[allow(non_snake_case)]
pub fn H5D_select_io_mem_with<F>(src: &[u8], spans: &[(usize, usize)], visitor: F) -> Result<()>
where
    F: FnMut(&[u8]),
{
    H5D__scatter_mem_with(src, spans, visitor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_storage_reads_and_writes() {
        let mut storage = H5D__compact_construct(b"abcd".to_vec());
        H5D__compact_writevv(&mut storage, 2, b"XY").unwrap();
        let mut data = [0; 4];
        H5D__compact_readvv_into(&storage, 0, &mut data).unwrap();
        assert_eq!(&data, b"abXY");
        assert!(storage.dirty);
        H5D__compact_flush(&mut storage);
        assert!(!storage.dirty);
    }

    #[test]
    fn none_chunk_index_computes_implicit_addresses() {
        let mut index = H5D__none_idx_create();
        H5D__none_idx_configure(&mut index, 1000, 16, vec![2, 3]).unwrap();

        assert!(H5D__none_idx_is_open(&index));
        assert!(H5D__none_idx_is_space_alloc(&index));
        assert_eq!(
            H5D__none_idx_get_addr_checked(&index, &[0, 0]).unwrap(),
            1000
        );
        assert_eq!(
            H5D__none_idx_get_addr_checked(&index, &[0, 2]).unwrap(),
            1032
        );
        assert_eq!(
            H5D__none_idx_get_addr_checked(&index, &[1, 0]).unwrap(),
            1048
        );

        let mut chunks = Vec::new();
        H5D__none_idx_iterate_with(&index, |coord, addr, size| {
            chunks.push(ChunkInfo {
                coord: coord.to_vec(),
                addr,
                size,
                filter_mask: 0,
            });
        })
        .unwrap();
        assert_eq!(chunks.len(), 6);
        assert_eq!(chunks[0].coord, vec![0, 0]);
        assert_eq!(chunks[0].addr, 1000);
        assert_eq!(chunks[5].coord, vec![1, 2]);
        assert_eq!(chunks[5].addr, 1080);
        assert!(chunks.iter().all(|chunk| chunk.filter_mask == 0));
    }

    #[test]
    fn none_chunk_index_rejects_malformed_geometry() {
        let mut index = H5D__none_idx_create();
        assert!(H5D__none_idx_configure(&mut index, 0, 0, vec![1]).is_err());
        assert!(H5D__none_idx_configure(&mut index, 0, 1, vec![0]).is_err());

        H5D__none_idx_configure(&mut index, u64::MAX - 7, 8, vec![2]).unwrap();
        assert!(H5D__none_idx_get_addr_checked(&index, &[1]).is_err());
        assert!(H5D__none_idx_get_addr_checked(&index, &[2]).is_err());
        assert!(H5D__none_idx_get_addr_checked(&index, &[0, 0]).is_err());
    }

    #[test]
    fn chunk_index_count_images_roundtrip_and_reject_bad_lengths() {
        let mut index = ChunkIndexState::default();
        H5D__farray_idx_insert(&mut index, vec![0], 64, 8);
        H5D__farray_idx_insert(&mut index, vec![1], 72, 8);

        let mut farray_image = Vec::with_capacity(8);
        H5D__chunk_index_encode_count_into(&index, &mut farray_image).unwrap();
        let farray = H5D__farray_decode(&farray_image).unwrap();
        assert!(farray.metadata_loaded);
        assert!(farray.space_allocated);
        assert_eq!(farray.declared_entry_count, 2);
        assert!(farray.entries.is_empty());
        assert_eq!(
            H5D__farray_filt_decode(&farray_image)
                .unwrap()
                .declared_entry_count,
            2
        );

        let earray = H5D__earray_filt_decode(&H5D__earray_encode(&index).unwrap()).unwrap();
        assert_eq!(earray.declared_entry_count, 2);
        let mut earray_dump = String::new();
        explicit_index_wrappers::H5D__earray_idx_dump_to(&index, &mut earray_dump).unwrap();
        assert_eq!(earray_dump, "    Address: 64\n");

        let bt2 = H5D__bt2_filt_decode(&H5D__bt2_filt_encode(&index).unwrap()).unwrap();
        assert_eq!(bt2.declared_entry_count, 2);

        let explicit = explicit_index_wrappers::H5D__bt2_unfilt_decode(
            &explicit_index_wrappers::H5D__bt2_unfilt_encode(&index).unwrap(),
        )
        .unwrap();
        assert_eq!(explicit.declared_entry_count, 2);

        let btree = explicit_index_wrappers::H5D__btree_decode_key(
            &explicit_index_wrappers::H5D__btree_encode_key(&index).unwrap(),
        )
        .unwrap();
        assert_eq!(btree.declared_entry_count, 2);

        assert!(H5D__farray_decode(&[0; 7]).is_err());
        assert!(H5D__bt2_filt_decode(&[0; 9]).is_err());
        assert!(explicit_index_wrappers::H5D__earray_filt_decode(&[0; 7]).is_err());
        assert!(explicit_index_wrappers::H5D__btree_decode_key(&[0; 9]).is_err());
    }

    #[test]
    fn virtual_mapping_pre_rejects_points_and_mismatched_limited_counts() {
        let point = Selection::Points(vec![vec![0]]);
        let all = Selection::All;
        let args = VirtualMappingValidation {
            virtual_selection: &point,
            virtual_shape: &[1],
            virtual_max_dims: &[1],
            source_selection: &all,
            source_shape: &[1],
            source_max_dims: &[1],
            source_file_printf_substitutions: 0,
            source_dataset_printf_substitutions: 0,
            source_space_status: VirtualSpaceStatus::Valid,
        };
        assert!(matches!(
            H5D_virtual_check_mapping_pre(&args),
            Err(Error::Unsupported(_))
        ));

        let virtual_selection = Selection::All;
        let source_selection = Selection::All;
        let args = VirtualMappingValidation {
            virtual_selection: &virtual_selection,
            virtual_shape: &[2],
            virtual_max_dims: &[2],
            source_selection: &source_selection,
            source_shape: &[3],
            source_max_dims: &[3],
            source_file_printf_substitutions: 0,
            source_dataset_printf_substitutions: 0,
            source_space_status: VirtualSpaceStatus::Valid,
        };
        assert!(matches!(
            H5D_virtual_check_mapping_pre(&args),
            Err(Error::InvalidFormat(_))
        ));
    }

    #[test]
    fn virtual_mapping_post_enforces_printf_rules() {
        let virtual_selection = Selection::Hyperslab(vec![crate::hl::selection::HyperslabDim {
            start: 0,
            stride: 10,
            count: 2,
            block: 3,
        }]);
        let source_selection = Selection::All;
        let ok = VirtualMappingValidation {
            virtual_selection: &virtual_selection,
            virtual_shape: &[20],
            virtual_max_dims: &[u64::MAX],
            source_selection: &source_selection,
            source_shape: &[3],
            source_max_dims: &[3],
            source_file_printf_substitutions: 1,
            source_dataset_printf_substitutions: 0,
            source_space_status: VirtualSpaceStatus::Valid,
        };
        H5D_virtual_check_mapping_post(&ok).unwrap();

        let missing_printf = VirtualMappingValidation {
            source_file_printf_substitutions: 0,
            ..ok.clone()
        };
        assert!(matches!(
            H5D_virtual_check_mapping_post(&missing_printf),
            Err(Error::InvalidFormat(_))
        ));

        let limited_with_printf = VirtualMappingValidation {
            virtual_max_dims: &[20],
            source_file_printf_substitutions: 1,
            ..ok
        };
        assert!(matches!(
            H5D_virtual_check_mapping_post(&limited_with_printf),
            Err(Error::InvalidFormat(_))
        ));
    }

    #[test]
    fn virtual_min_dims_check_rejects_rank_and_extent_mismatch() {
        let mapping = VirtualMapping {
            min_dims: vec![2, 3],
            ..VirtualMapping::default()
        };

        H5D_virtual_check_min_dims_checked(&mapping, &[2, 3]).unwrap();
        assert!(H5D_virtual_check_min_dims(&mapping, &[4, 5]));
        assert!(matches!(
            H5D_virtual_check_min_dims_checked(&mapping, &[2]),
            Err(Error::InvalidFormat(_))
        ));
        assert!(matches!(
            H5D_virtual_check_min_dims_checked(&mapping, &[2, 2]),
            Err(Error::InvalidFormat(_))
        ));
        assert!(!H5D_virtual_check_min_dims(&mapping, &[2, 2]));
    }

    #[test]
    fn virtual_source_name_parser_and_builder_follow_printf_rules() {
        assert!(H5D_virtual_parse_source_name("plain.h5").unwrap().is_none());

        let parsed = H5D_virtual_parse_source_name("run_%b/part_%%_%b.h5")
            .unwrap()
            .expect("printf source name should parse");
        assert_eq!(parsed.substitutions, 2);
        assert_eq!(parsed.static_strlen, "run_/part_%_.h5".len());
        assert_eq!(
            H5D__virtual_build_source_name_cow("ignored", Some(&parsed), 42).unwrap(),
            "run_42/part_%_42.h5"
        );
        let mut built = String::new();
        H5D__virtual_build_source_name_into("ignored", Some(&parsed), 42, &mut built).unwrap();
        assert_eq!(built, "run_42/part_%_42.h5");

        let escaped = H5D_virtual_parse_source_name("literal_%%.h5")
            .unwrap()
            .expect("escaped percent should allocate parsed name");
        assert_eq!(escaped.substitutions, 0);
        assert_eq!(
            H5D__virtual_build_source_name_cow("literal_%%.h5", Some(&escaped), 7).unwrap(),
            "literal_%.h5"
        );
        assert!(matches!(
            H5D__virtual_build_source_name_cow("plain.h5", None, 7).unwrap(),
            Cow::Borrowed("plain.h5")
        ));
        H5D_virtual_free_parsed_name(Some(escaped));
    }

    #[test]
    fn virtual_source_name_parser_rejects_bad_format_specifiers() {
        assert!(matches!(
            H5D_virtual_parse_source_name("bad_%x.h5"),
            Err(Error::InvalidFormat(_))
        ));
        assert!(matches!(
            H5D_virtual_parse_source_name("bad_%"),
            Err(Error::InvalidFormat(_))
        ));

        let malformed = VirtualParsedName {
            segments: vec!["a".to_string()],
            static_strlen: 1,
            substitutions: 1,
        };
        assert!(matches!(
            H5D__virtual_build_source_name_cow("ignored", Some(&malformed), 0),
            Err(Error::InvalidFormat(_))
        ));
    }

    #[test]
    fn scatter_and_contiguous_reads_reject_bad_spans() {
        let mut out = vec![vec![0; 2], vec![0; 2]];
        H5D__scatter_file_into(b"abcdef", &[(1, 2), (4, 2)], &mut out).unwrap();
        assert_eq!(out, vec![b"bc".to_vec(), b"ef".to_vec()]);

        let mut out = vec![vec![0; 1]];
        assert!(H5D__scatter_file_into(b"abc", &[(usize::MAX, 1)], &mut out).is_err());
        let mut out = vec![vec![0; 2]];
        assert!(H5D__scatter_file_into(b"abc", &[(2, 2)], &mut out).is_err());
        let mut out = vec![vec![0; 2]];
        assert!(H5D__scatgath_read_select_into(b"abc", &[(2, 2)], &mut out).is_err());

        let storage = ContiguousStorage {
            addr: Some(0),
            data: b"abcdef".to_vec(),
            cached: false,
            allocated: true,
        };
        let mut out = vec![vec![0; 3]];
        H5D__contig_readvv_into(&storage, &[(0, 3)], &mut out).unwrap();
        assert_eq!(out, vec![b"abc".to_vec()]);
        let mut out = vec![vec![0; 2]];
        assert!(H5D__contig_readvv_into(&storage, &[(5, 2)], &mut out).is_err());
    }

    #[test]
    fn scatter_into_fills_original_shaped_output() {
        let mut out = vec![vec![0; 2], vec![0; 2]];
        H5D__scatter_file_into(b"abcdef", &[(1, 2), (4, 2)], &mut out).unwrap();
        assert_eq!(out, vec![b"bc".to_vec(), b"ef".to_vec()]);

        let mut mem_out = vec![vec![0; 3]];
        H5D__scatter_mem_into(b"abcdef", &[(2, 3)], &mut mem_out).unwrap();
        assert_eq!(mem_out, vec![b"cde".to_vec()]);
    }

    #[test]
    fn select_io_into_fills_original_shaped_output() {
        let dataset = DatasetApi {
            name: Some("d".into()),
            extent: vec![6],
            raw: b"abcdef".to_vec(),
            virtual_layout: None,
        };
        let mut out = vec![vec![0; 1], vec![0; 3]];
        H5D__select_io_into(&dataset, &[(0, 1), (3, 3)], &mut out).unwrap();
        assert_eq!(out, vec![b"a".to_vec(), b"def".to_vec()]);

        let mut mem_out = vec![vec![0; 2]];
        H5D_select_io_mem_into(b"abcdef", &[(4, 2)], &mut mem_out).unwrap();
        assert_eq!(mem_out, vec![b"ef".to_vec()]);
    }

    #[test]
    fn scatter_into_rejects_output_size_mismatch() {
        let mut missing = vec![vec![0; 2]];
        assert!(matches!(
            H5D__scatter_file_into(b"abcdef", &[(1, 2), (4, 2)], &mut missing),
            Err(Error::InvalidFormat(_))
        ));
        assert_eq!(missing, vec![vec![0; 2]]);

        let mut wrong_span_len = vec![vec![0; 3]];
        assert!(matches!(
            H5D__scatter_mem_into(b"abcdef", &[(1, 2)], &mut wrong_span_len),
            Err(Error::InvalidFormat(_))
        ));
        assert_eq!(wrong_span_len, vec![vec![0; 3]]);
    }

    #[test]
    fn select_io_into_rejects_output_size_mismatch() {
        let dataset = DatasetApi {
            name: None,
            extent: vec![3],
            raw: b"abc".to_vec(),
            virtual_layout: None,
        };
        let mut out = vec![vec![0; 2]];
        assert!(matches!(
            H5D__select_io_into(&dataset, &[(0, 1)], &mut out),
            Err(Error::InvalidFormat(_))
        ));
        assert_eq!(out, vec![vec![0; 2]]);
    }

    #[test]
    fn scatter_into_rejects_span_bounds_errors() {
        let mut out = vec![vec![0; 1]];
        assert!(matches!(
            H5D__scatter_file_into(b"abc", &[(usize::MAX, 1)], &mut out),
            Err(Error::InvalidFormat(_))
        ));
        assert_eq!(out, vec![vec![0; 1]]);

        let mut out = vec![vec![0; 2]];
        assert!(matches!(
            H5D_select_io_mem_into(b"abc", &[(2, 2)], &mut out),
            Err(Error::InvalidFormat(_))
        ));
        assert_eq!(out, vec![vec![0; 2]]);
    }

    #[test]
    fn layout_metadata_size_checked_matches_legacy_wrapper() {
        let dataset = H5D__create_api_common(Some("d".into()), vec![2, 3, 4]);
        assert_eq!(H5D__layout_meta_size_checked(&dataset).unwrap(), 24);
        assert_eq!(H5D__layout_meta_size(&dataset), 24);
        assert_eq!(
            H5D__calculate_minimum_header_size_checked(&dataset).unwrap(),
            24
        );
    }

    #[test]
    fn vlen_buffer_size_checked_sums_payloads() {
        let values = vec![b"aa".to_vec(), b"bbb".to_vec()];
        assert_eq!(H5D__vlen_get_buf_size_checked(&values).unwrap(), 5);
        assert_eq!(H5D__vlen_get_buf_size(&values), 5);
        assert_eq!(H5D__vlen_get_buf_size_gen_checked(&values).unwrap(), 5);
        assert_eq!(
            H5D__vlen_get_buf_size_slices_checked([b"aa".as_slice(), b"bbb".as_slice()]).unwrap(),
            5
        );
    }

    #[test]
    fn chunk_table_checked_writers_assign_addresses() {
        let mut table = H5D__chunk_construct(vec![2, 2]);
        H5D__chunk_write_checked(&mut table, vec![0, 0], vec![1, 2]).unwrap();
        assert_eq!(H5D__chunk_lookup(&table, &[0, 0]).unwrap().addr, 0);

        H5D__chunk_allocate_checked(&mut table, vec![1, 0], 4).unwrap();
        assert_eq!(H5D__chunk_lookup(&table, &[1, 0]).unwrap().addr, 1);

        let addr = H5D__chunk_file_alloc_checked(&mut table, vec![2, 0], 8).unwrap();
        assert_eq!(addr, 2);
        assert_eq!(H5D__chunk_file_alloc(&mut table, vec![3, 0], 8), 3);
    }

    #[test]
    fn allocation_aware_views_borrow_layout_and_chunk_data() {
        let mut dataset = H5D__create_api_common(Some("vds".into()), vec![4]);
        let layout = VirtualLayout {
            mappings: vec![VirtualMapping {
                source_file: "src.h5".into(),
                source_dataset: "/d".into(),
                min_dims: vec![4],
                max_dims: vec![Some(4)],
                open: true,
            }],
            unlimited: false,
        };
        H5D__virtual_store_layout(&mut dataset, layout);

        let borrowed = H5D__virtual_load_layout_ref(&dataset).unwrap();
        assert_eq!(borrowed.mappings.len(), 1);
        assert_eq!(
            H5D__mappings_to_leaves_iter(borrowed)
                .next()
                .unwrap()
                .source_dataset,
            "/d"
        );
        assert_eq!(H5D__cache_dataspace_info_slice(&dataset), &[4]);

        let mut image = Vec::with_capacity(64);
        H5D__layout_oh_create_into(&dataset, &mut image).unwrap();
        assert_eq!(image, 4u64.to_le_bytes());
        H5D__prepare_minimized_oh_into(&dataset, &mut image).unwrap();
        assert_eq!(image, 4u64.to_le_bytes());
    }

    #[test]
    fn allocation_aware_chunk_table_helpers_visit_and_copy_into() {
        let mut table = H5D__chunk_construct(vec![2]);
        H5D__chunk_write_checked(&mut table, vec![1], b"xy".to_vec()).unwrap();

        assert_eq!(H5D__chunk_read_slice(&mut table, &[1]).unwrap(), b"xy");
        assert_eq!(table.cache_hits, 1);

        let mut buf = [0; 2];
        H5D__chunk_read_into(&mut table, &[1], &mut buf).unwrap();
        assert_eq!(&buf, b"xy");
        assert_eq!(table.cache_hits, 2);

        let mut addrmap = Vec::new();
        H5D__chunk_addrmap_with(&table, |coord, addr| addrmap.push((coord.to_vec(), addr)));
        assert_eq!(addrmap, vec![(vec![1], 0)]);

        let coords = vec![vec![1], vec![2]];
        let borrowed_coords: Vec<&[u64]> = H5D__chunk_io_init_selections_iter(&coords).collect();
        assert_eq!(borrowed_coords, vec![&[1][..], &[2][..]]);

        let mut chunks = Vec::new();
        H5D__create_piece_mem_map_hyper_with(&table, &coords, |coord, data| {
            chunks.push((coord.to_vec(), data.to_vec()));
        });
        assert_eq!(chunks, vec![(vec![1], b"xy".to_vec())]);

        let borrowed_coords = [&[1][..], &[2][..]];
        let mut borrowed_file_map = Vec::new();
        H5D__create_piece_file_map_hyper_slices_with(&table, borrowed_coords, |coord, addr| {
            borrowed_file_map.push((coord.to_vec(), addr))
        });
        assert_eq!(borrowed_file_map, vec![(vec![1], 0)]);

        let mut borrowed_mem_map = Vec::new();
        H5D__create_piece_mem_map_hyper_slices_with(&table, borrowed_coords, |coord, data| {
            borrowed_mem_map.push((coord.to_vec(), data.to_vec()))
        });
        assert_eq!(borrowed_mem_map, vec![(vec![1], b"xy".to_vec())]);
    }

    #[test]
    fn allocation_aware_contiguous_and_select_reads_fill_existing_buffers() {
        let storage = ContiguousStorage {
            addr: Some(0),
            data: b"abcdef".to_vec(),
            cached: false,
            allocated: true,
        };
        let mut out = vec![vec![0; 2], vec![0; 3]];
        H5D__contig_readvv_into(&storage, &[(1, 2), (3, 3)], &mut out).unwrap();
        assert_eq!(out, vec![b"bc".to_vec(), b"def".to_vec()]);

        let mut scat = vec![vec![0; 2]];
        H5D__scatgath_read_select_into(b"abcdef", &[(2, 2)], &mut scat).unwrap();
        assert_eq!(scat, vec![b"cd".to_vec()]);
        assert_eq!(H5D__compound_opt_read_slice(b"abc"), b"abc");
        assert_eq!(H5D__contig_readvv_cb_slice(b"abc"), b"abc");
        assert_eq!(H5D__contig_readvv_sieve_cb_slice(b"abc"), b"abc");

        let mut visited = Vec::new();
        H5D__contig_readvv_with(&storage, &[(0, 2), (4, 2)], |span| {
            visited.push(span.to_vec());
        })
        .unwrap();
        assert_eq!(visited, vec![b"ab".to_vec(), b"ef".to_vec()]);

        let dataset = DatasetApi {
            name: None,
            extent: vec![6],
            raw: b"abcdef".to_vec(),
            virtual_layout: None,
        };
        let mut selected_len = 0usize;
        H5D__select_io_with(&dataset, &[(1, 3)], |span| {
            selected_len += span.len();
            assert_eq!(span, b"bcd");
        })
        .unwrap();
        assert_eq!(selected_len, 3);
    }

    #[test]
    fn allocation_aware_scatter_gather_dump_and_copy_helpers_reuse_storage() {
        let mut first = [0; 2];
        let mut second = [0; 3];
        let mut borrowed_out: [&mut [u8]; 2] = [&mut first, &mut second];
        H5D__scatter_file_into_slices(b"abcdef", &[(1, 2), (3, 3)], &mut borrowed_out).unwrap();
        assert_eq!(&first, b"bc");
        assert_eq!(&second, b"def");

        let parts = [b"ab".as_slice(), b"cde".as_slice()];
        let mut gathered = Vec::with_capacity(16);
        H5D__gather_file_into(parts, &mut gathered);
        assert_eq!(gathered, b"abcde");
        H5D__scatgath_write_select_into([b"x".as_slice(), b"yz".as_slice()], &mut gathered);
        assert_eq!(gathered, b"xyz");

        let mut index = ChunkIndexState::default();
        H5D__bt2_idx_insert(&mut index, vec![1], 42, 3);
        assert_eq!(H5D__chunk_index_iterate(&index).next().unwrap().addr, 42);

        let mut index_copy = ChunkIndexState::default();
        H5D__chunk_index_copy_setup_into(&index, &mut index_copy);
        assert_eq!(index_copy, index);

        let mut dump = String::new();
        H5D__chunk_index_dump_to("bt2", &index, &mut dump).unwrap();
        assert_eq!(
            dump,
            "bt2(open=true, allocated=true, entries=1, declared=0)"
        );

        let single = SingleChunkIndex {
            open: true,
            space_allocated: true,
            chunk_addr: Some(7),
            metadata_loaded: false,
        };
        let mut single_copy = SingleChunkIndex::default();
        H5D__single_idx_copy_setup_into(&single, &mut single_copy);
        assert_eq!(single_copy, single);

        let mut single_dump = String::new();
        H5D__single_idx_dump_to(&single, &mut single_dump).unwrap();
        assert_eq!(
            single_dump,
            "single_idx(open=true, allocated=true, addr=Some(7))"
        );

        let mut zeros = [1, 2, 3];
        H5D__nonexistent_readvv_into(&mut zeros);
        assert_eq!(zeros, [0, 0, 0]);
    }

    #[test]
    fn allocation_aware_misc_helpers_borrow_or_reuse_storage() {
        assert_eq!(
            H5D__build_file_prefix_slice("/tmp/data/file.h5"),
            "/tmp/data"
        );
        assert_eq!(H5D__build_file_prefix_slice("file.h5"), "");

        let datasets = vec![H5D__create_api_common(Some("d".into()), vec![1])];
        let mut reopened = Vec::with_capacity(4);
        H5D_mult_refresh_reopen_into(&datasets, &mut reopened);
        assert_eq!(reopened, datasets);

        let mut efl_buf = [9; 4];
        H5D__efl_readvv_cb_into(&mut efl_buf);
        assert_eq!(efl_buf, [0; 4]);

        let mut storage = H5D__contig_construct(Vec::new());
        H5D__contig_writevv_slices(&mut storage, &[(1, b"ab".as_slice())]).unwrap();
        assert_eq!(storage.data, vec![0, b'a', b'b']);

        let mut datasets = vec![
            H5D__create_api_common(None, vec![1]),
            H5Dcreate_anon(vec![2]),
        ];
        H5Dwrite_multi_slices(
            &mut datasets,
            [b"x".as_slice(), b"yz".as_slice()].into_iter(),
        );
        assert_eq!(datasets[0].raw, b"x");
        assert_eq!(datasets[1].raw, b"yz");

        let before_capacity = datasets[1].raw.capacity();
        let mut read_multi_out = vec![Vec::new(), Vec::with_capacity(before_capacity)];
        H5Dread_multi_into(&datasets, &mut read_multi_out);
        assert_eq!(read_multi_out, vec![b"x".to_vec(), b"yz".to_vec()]);
        assert_eq!(read_multi_out[1].capacity(), before_capacity);

        let mut visited = Vec::new();
        H5D__scatter_file_with(b"abcdef", &[(0, 1), (3, 2)], |span| {
            visited.push(span.to_vec());
        })
        .unwrap();
        assert_eq!(visited, vec![b"a".to_vec(), b"de".to_vec()]);
    }
}
