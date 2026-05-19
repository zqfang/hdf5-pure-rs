use crate::hl::plist::file_access::FileAccess;

/// Lightweight API-context state mirroring libhdf5's H5CX bookkeeping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApiContext {
    dxpl: FileAccess,
    dcpl: Option<String>,
    lcpl: Option<String>,
    lapl: FileAccess,
    apl: FileAccess,
    vol_wrap_ctx: Option<String>,
    vol_connector_prop: Option<String>,
    tag: Option<String>,
    ring: u8,
    mpi_coll_datatypes: Option<String>,
    mpi_file_flushing: bool,
    mpio_rank0_bcast: bool,
    btree_split_ratios: (u8, u8, u8),
    max_temp_buf: usize,
    tconv_buf: Vec<u8>,
    bkgr_buf: Vec<u8>,
    bkgr_buf_type: BackgroundBufferType,
    vec_size: usize,
    io_xfer_mode: IoTransferMode,
    mpio_coll_opt: MpioCollectiveOpt,
    mpio_local_no_coll_cause: u32,
    mpio_global_no_coll_cause: u32,
    mpio_chunk_opt_num: u32,
    mpio_chunk_opt_ratio: u32,
    err_detect: bool,
    filter_cb: Option<String>,
    data_transform: Option<String>,
    dt_conv_cb: Option<String>,
    selection_io_mode: SelectionIoMode,
    no_selection_io_cause: u32,
    actual_selection_io_mode: SelectionIoMode,
    modify_write_buf: bool,
    encoding: Option<String>,
    intermediate_group: bool,
    nlinks: usize,
    dset_min_ohdr_flag: bool,
    ext_file_prefix: Option<String>,
    vds_prefix: Option<String>,
    vlen_alloc_info: Option<String>,
    mpio_actual_io_mode: SelectionIoMode,
    ohdr_flags: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackgroundBufferType {
    None,
    Temporary,
    Application,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoTransferMode {
    Independent,
    Collective,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MpioCollectiveOpt {
    None,
    OneLinkChunk,
    MultiChunk,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionIoMode {
    None,
    Scalar,
    Vector,
    Chunk,
}

impl Default for ApiContext {
    fn default() -> Self {
        Self {
            dxpl: FileAccess::default(),
            dcpl: None,
            lcpl: None,
            lapl: FileAccess::default(),
            apl: FileAccess::default(),
            vol_wrap_ctx: None,
            vol_connector_prop: None,
            tag: None,
            ring: 0,
            mpi_coll_datatypes: None,
            mpi_file_flushing: false,
            mpio_rank0_bcast: false,
            btree_split_ratios: (100, 40, 80),
            max_temp_buf: 1024 * 1024,
            tconv_buf: Vec::new(),
            bkgr_buf: Vec::new(),
            bkgr_buf_type: BackgroundBufferType::None,
            vec_size: 1024,
            io_xfer_mode: IoTransferMode::Independent,
            mpio_coll_opt: MpioCollectiveOpt::None,
            mpio_local_no_coll_cause: 0,
            mpio_global_no_coll_cause: 0,
            mpio_chunk_opt_num: 0,
            mpio_chunk_opt_ratio: 0,
            err_detect: true,
            filter_cb: None,
            data_transform: None,
            dt_conv_cb: None,
            selection_io_mode: SelectionIoMode::None,
            no_selection_io_cause: 0,
            actual_selection_io_mode: SelectionIoMode::None,
            modify_write_buf: false,
            encoding: None,
            intermediate_group: false,
            nlinks: 16,
            dset_min_ohdr_flag: false,
            ext_file_prefix: None,
            vds_prefix: None,
            vlen_alloc_info: None,
            mpio_actual_io_mode: SelectionIoMode::None,
            ohdr_flags: 0,
        }
    }
}

impl ApiContext {
    pub fn init_package() -> Self {
        Self::default()
    }

    pub fn term_package(self) {}

    pub fn pushed(stack: &[Self]) -> bool {
        !stack.is_empty()
    }

    pub fn push(stack: &mut Vec<Self>, context: Self) {
        stack.push(context);
    }

    pub fn retrieve_state(&self) -> Self {
        let mut state = Self::default();
        self.retrieve_state_into(&mut state);
        state
    }

    pub fn retrieve_state_into(&self, out: &mut Self) {
        out.dxpl.clone_from(&self.dxpl);
        out.dcpl.clone_from(&self.dcpl);
        out.lcpl.clone_from(&self.lcpl);
        out.lapl.clone_from(&self.lapl);
        out.apl.clone_from(&self.apl);
        out.vol_wrap_ctx.clone_from(&self.vol_wrap_ctx);
        out.vol_connector_prop.clone_from(&self.vol_connector_prop);
        out.tag.clone_from(&self.tag);
        out.ring = self.ring;
        out.mpi_coll_datatypes.clone_from(&self.mpi_coll_datatypes);
        out.mpi_file_flushing = self.mpi_file_flushing;
        out.mpio_rank0_bcast = self.mpio_rank0_bcast;
        out.btree_split_ratios = self.btree_split_ratios;
        out.max_temp_buf = self.max_temp_buf;
        out.tconv_buf.clone_from(&self.tconv_buf);
        out.bkgr_buf.clone_from(&self.bkgr_buf);
        out.bkgr_buf_type = self.bkgr_buf_type;
        out.vec_size = self.vec_size;
        out.io_xfer_mode = self.io_xfer_mode;
        out.mpio_coll_opt = self.mpio_coll_opt;
        out.mpio_local_no_coll_cause = self.mpio_local_no_coll_cause;
        out.mpio_global_no_coll_cause = self.mpio_global_no_coll_cause;
        out.mpio_chunk_opt_num = self.mpio_chunk_opt_num;
        out.mpio_chunk_opt_ratio = self.mpio_chunk_opt_ratio;
        out.err_detect = self.err_detect;
        out.filter_cb.clone_from(&self.filter_cb);
        out.data_transform.clone_from(&self.data_transform);
        out.dt_conv_cb.clone_from(&self.dt_conv_cb);
        out.selection_io_mode = self.selection_io_mode;
        out.no_selection_io_cause = self.no_selection_io_cause;
        out.actual_selection_io_mode = self.actual_selection_io_mode;
        out.modify_write_buf = self.modify_write_buf;
        out.encoding.clone_from(&self.encoding);
        out.intermediate_group = self.intermediate_group;
        out.nlinks = self.nlinks;
        out.dset_min_ohdr_flag = self.dset_min_ohdr_flag;
        out.ext_file_prefix.clone_from(&self.ext_file_prefix);
        out.vds_prefix.clone_from(&self.vds_prefix);
        out.vlen_alloc_info.clone_from(&self.vlen_alloc_info);
        out.mpio_actual_io_mode = self.mpio_actual_io_mode;
        out.ohdr_flags = self.ohdr_flags;
    }

    pub fn restore_state(&mut self, state: Self) {
        *self = state;
    }

    pub fn restore_state_from(&mut self, state: &Self) {
        state.retrieve_state_into(self);
    }

    pub fn free_state(self) {}

    pub fn is_def_dxpl(&self) -> bool {
        self.dxpl == FileAccess::default()
    }

    pub fn set_dxpl(&mut self, dxpl: FileAccess) {
        self.dxpl = dxpl;
    }

    pub fn set_dcpl<S: Into<String>>(&mut self, dcpl: Option<S>) {
        self.dcpl = dcpl.map(Into::into);
    }

    pub fn set_dcpl_str(&mut self, dcpl: Option<&str>) {
        set_option_string_from_str(&mut self.dcpl, dcpl);
    }

    pub fn set_lcpl<S: Into<String>>(&mut self, lcpl: Option<S>) {
        self.lcpl = lcpl.map(Into::into);
    }

    pub fn set_lcpl_str(&mut self, lcpl: Option<&str>) {
        set_option_string_from_str(&mut self.lcpl, lcpl);
    }

    pub fn set_lapl(&mut self, lapl: FileAccess) {
        self.lapl = lapl;
    }

    pub fn set_apl(&mut self, apl: FileAccess) {
        self.apl = apl;
    }

    pub fn set_vol_wrap_ctx<S: Into<String>>(&mut self, ctx: Option<S>) {
        self.vol_wrap_ctx = ctx.map(Into::into);
    }

    pub fn set_vol_wrap_ctx_str(&mut self, ctx: Option<&str>) {
        set_option_string_from_str(&mut self.vol_wrap_ctx, ctx);
    }

    pub fn set_vol_connector_prop<S: Into<String>>(&mut self, prop: Option<S>) {
        self.vol_connector_prop = prop.map(Into::into);
    }

    pub fn set_vol_connector_prop_str(&mut self, prop: Option<&str>) {
        set_option_string_from_str(&mut self.vol_connector_prop, prop);
    }

    pub fn get_dxpl(&self) -> &FileAccess {
        &self.dxpl
    }

    pub fn get_lapl(&self) -> &FileAccess {
        &self.lapl
    }

    pub fn get_vol_wrap_ctx(&self) -> Option<&str> {
        self.vol_wrap_ctx.as_deref()
    }

    pub fn get_vol_connector_prop(&self) -> Option<&str> {
        self.vol_connector_prop.as_deref()
    }

    pub fn get_tag(&self) -> Option<&str> {
        self.tag.as_deref()
    }

    pub fn get_ring(&self) -> u8 {
        self.ring
    }

    pub fn get_mpi_coll_datatypes(&self) -> Option<&str> {
        self.mpi_coll_datatypes.as_deref()
    }

    pub fn get_mpi_file_flushing(&self) -> bool {
        self.mpi_file_flushing
    }

    pub fn get_mpio_rank0_bcast(&self) -> bool {
        self.mpio_rank0_bcast
    }

    pub fn get_btree_split_ratios(&self) -> (u8, u8, u8) {
        self.btree_split_ratios
    }

    pub fn get_max_temp_buf(&self) -> usize {
        self.max_temp_buf
    }

    pub fn get_tconv_buf(&self) -> &[u8] {
        &self.tconv_buf
    }

    pub fn get_bkgr_buf(&self) -> &[u8] {
        &self.bkgr_buf
    }

    pub fn get_bkgr_buf_type(&self) -> BackgroundBufferType {
        self.bkgr_buf_type
    }

    pub fn get_vec_size(&self) -> usize {
        self.vec_size
    }

    pub fn get_io_xfer_mode(&self) -> IoTransferMode {
        self.io_xfer_mode
    }

    pub fn get_mpio_coll_opt(&self) -> MpioCollectiveOpt {
        self.mpio_coll_opt
    }

    pub fn get_mpio_local_no_coll_cause(&self) -> u32 {
        self.mpio_local_no_coll_cause
    }

    pub fn get_mpio_global_no_coll_cause(&self) -> u32 {
        self.mpio_global_no_coll_cause
    }

    pub fn get_mpio_chunk_opt_num(&self) -> u32 {
        self.mpio_chunk_opt_num
    }

    pub fn get_mpio_chunk_opt_ratio(&self) -> u32 {
        self.mpio_chunk_opt_ratio
    }

    pub fn get_err_detect(&self) -> bool {
        self.err_detect
    }

    pub fn get_filter_cb(&self) -> Option<&str> {
        self.filter_cb.as_deref()
    }

    pub fn get_data_transform(&self) -> Option<&str> {
        self.data_transform.as_deref()
    }

    pub fn get_dt_conv_cb(&self) -> Option<&str> {
        self.dt_conv_cb.as_deref()
    }

    pub fn get_selection_io_mode(&self) -> SelectionIoMode {
        self.selection_io_mode
    }

    pub fn get_no_selection_io_cause(&self) -> u32 {
        self.no_selection_io_cause
    }

    pub fn get_actual_selection_io_mode(&self) -> SelectionIoMode {
        self.actual_selection_io_mode
    }

    pub fn get_modify_write_buf(&self) -> bool {
        self.modify_write_buf
    }

    pub fn get_encoding(&self) -> Option<&str> {
        self.encoding.as_deref()
    }

    pub fn get_intermediate_group(&self) -> bool {
        self.intermediate_group
    }

    pub fn get_nlinks(&self) -> usize {
        self.nlinks
    }

    pub fn get_dset_min_ohdr_flag(&self) -> bool {
        self.dset_min_ohdr_flag
    }

    pub fn get_ext_file_prefix(&self) -> Option<&str> {
        self.ext_file_prefix.as_deref()
    }

    pub fn get_vds_prefix(&self) -> Option<&str> {
        self.vds_prefix.as_deref()
    }

    pub fn set_tag<S: Into<String>>(&mut self, tag: Option<S>) {
        self.tag = tag.map(Into::into);
    }

    pub fn set_tag_str(&mut self, tag: Option<&str>) {
        set_option_string_from_str(&mut self.tag, tag);
    }

    pub fn set_ring(&mut self, ring: u8) {
        self.ring = ring;
    }

    pub fn set_mpi_coll_datatypes<S: Into<String>>(&mut self, datatypes: Option<S>) {
        self.mpi_coll_datatypes = datatypes.map(Into::into);
    }

    pub fn set_mpi_coll_datatypes_str(&mut self, datatypes: Option<&str>) {
        set_option_string_from_str(&mut self.mpi_coll_datatypes, datatypes);
    }

    pub fn set_io_xfer_mode(&mut self, mode: IoTransferMode) {
        self.io_xfer_mode = mode;
    }

    pub fn set_mpio_coll_opt(&mut self, opt: MpioCollectiveOpt) {
        self.mpio_coll_opt = opt;
    }

    pub fn set_mpi_file_flushing(&mut self, enabled: bool) {
        self.mpi_file_flushing = enabled;
    }

    pub fn set_mpio_rank0_bcast(&mut self, enabled: bool) {
        self.mpio_rank0_bcast = enabled;
    }

    pub fn set_vlen_alloc_info<S: Into<String>>(&mut self, info: Option<S>) {
        self.vlen_alloc_info = info.map(Into::into);
    }

    pub fn set_vlen_alloc_info_str(&mut self, info: Option<&str>) {
        set_option_string_from_str(&mut self.vlen_alloc_info, info);
    }

    pub fn set_nlinks(&mut self, nlinks: usize) {
        self.nlinks = nlinks;
    }

    pub fn set_mpio_actual_io_mode(&mut self, mode: SelectionIoMode) {
        self.mpio_actual_io_mode = mode;
        self.actual_selection_io_mode = mode;
    }

    pub fn set_mpio_local_no_coll_cause(&mut self, cause: u32) {
        self.mpio_local_no_coll_cause = cause;
    }

    pub fn set_mpio_global_no_coll_cause(&mut self, cause: u32) {
        self.mpio_global_no_coll_cause = cause;
    }

    pub fn test_set_mpio_coll_chunk_link_hard(&mut self) {
        self.mpio_coll_opt = MpioCollectiveOpt::OneLinkChunk;
        self.mpio_chunk_opt_num = 1;
    }

    pub fn test_set_mpio_coll_chunk_multi_hard(&mut self) {
        self.mpio_coll_opt = MpioCollectiveOpt::MultiChunk;
        self.mpio_chunk_opt_num = 2;
    }

    pub fn test_set_mpio_coll_chunk_link_num_true(&mut self) {
        self.mpio_chunk_opt_num = 1;
    }

    pub fn test_set_mpio_coll_chunk_link_num_false(&mut self) {
        self.mpio_chunk_opt_num = 0;
    }

    pub fn test_set_mpio_coll_chunk_multi_ratio_coll(&mut self) {
        self.mpio_chunk_opt_ratio = 100;
    }

    pub fn test_set_mpio_coll_chunk_multi_ratio_ind(&mut self) {
        self.mpio_chunk_opt_ratio = 0;
    }

    pub fn test_set_mpio_coll_rank0_bcast(&mut self) {
        self.mpio_rank0_bcast = true;
    }

    pub fn set_no_selection_io_cause(&mut self, cause: u32) {
        self.no_selection_io_cause = cause;
    }

    pub fn set_actual_selection_io_mode(&mut self, mode: SelectionIoMode) {
        self.actual_selection_io_mode = mode;
    }

    pub fn get_ohdr_flags(&self) -> u8 {
        self.ohdr_flags
    }

    pub fn pop(stack: &mut Vec<Self>) -> Option<Self> {
        stack.pop()
    }
}

fn set_option_string_from_str(slot: &mut Option<String>, value: Option<&str>) {
    if let Some(value) = value {
        if let Some(existing) = slot {
            existing.clear();
            existing.push_str(value);
        } else {
            *slot = Some(value.to_string());
        }
    } else {
        *slot = None;
    }
}

#[cfg(test)]
mod tests {
    use super::{ApiContext, IoTransferMode, SelectionIoMode};

    #[test]
    fn api_context_push_restore_and_getters_round_trip() {
        let mut stack = Vec::new();
        let mut ctx = ApiContext::init_package();
        ctx.set_tag(Some("dataset read"));
        ctx.set_io_xfer_mode(IoTransferMode::Collective);
        ctx.set_actual_selection_io_mode(SelectionIoMode::Vector);
        ApiContext::push(&mut stack, ctx.clone());

        assert!(ApiContext::pushed(&stack));
        assert_eq!(stack[0].get_tag(), Some("dataset read"));
        assert_eq!(stack[0].get_io_xfer_mode(), IoTransferMode::Collective);
        assert_eq!(
            stack[0].get_actual_selection_io_mode(),
            SelectionIoMode::Vector
        );
        assert_eq!(ApiContext::pop(&mut stack), Some(ctx));
        assert!(!ApiContext::pushed(&stack));
    }
}
