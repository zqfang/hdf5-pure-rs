/// File creation properties.
#[derive(Debug, Clone)]
pub struct FileCreate {
    /// Superblock version.
    pub superblock_version: u8,
    /// Size of file addresses in bytes.
    pub sizeof_addr: u8,
    /// Size of file lengths in bytes.
    pub sizeof_size: u8,
    /// Symbol table leaf node K value (v0/v1 only).
    pub sym_leaf_k: u16,
    /// B-tree internal node K value (v0/v1 only).
    pub btree_k: u16,
    /// Chunk B-tree K value.
    pub chunk_btree_k: u16,
    file_space: (FileSpaceStrategy, bool, u64),
    file_space_page_size: u64,
    shared_mesg_indexes: Vec<SharedMessageIndex>,
    shared_mesg_phase_change: (u32, u32),
}

/// File-space management strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileSpaceStrategy {
    None,
    Aggregate,
    FreeSpaceManager,
    Page,
}

/// Shared object-header-message index configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SharedMessageIndex {
    pub message_type_flags: u32,
    pub minimum_message_size: u32,
}

impl FileCreate {
    /// Extract file creation properties from a File.
    pub fn from_file(f: &crate::hl::file::File) -> Self {
        let sb = f.superblock();
        Self {
            superblock_version: sb.version,
            sizeof_addr: sb.sizeof_addr,
            sizeof_size: sb.sizeof_size,
            sym_leaf_k: sb.sym_leaf_k,
            btree_k: sb.snode_btree_k,
            chunk_btree_k: sb.chunk_btree_k,
            file_space: (FileSpaceStrategy::Aggregate, false, 1),
            file_space_page_size: 4096,
            shared_mesg_indexes: Vec::new(),
            shared_mesg_phase_change: (50, 40),
        }
    }

    /// File address and length field sizes in bytes.
    ///
    /// Mirrors the useful read-side subset of `H5Pget_sizes`.
    pub fn sizes(&self) -> (u8, u8) {
        (self.sizeof_addr, self.sizeof_size)
    }

    /// Set file address and length field sizes in bytes.
    pub fn set_sizes(&mut self, sizeof_addr: u8, sizeof_size: u8) {
        self.sizeof_addr = sizeof_addr;
        self.sizeof_size = sizeof_size;
    }

    /// Symbol-table internal-node and leaf-node K values.
    ///
    /// Mirrors `H5Pget_sym_k` ordering: `(ik, lk)`.
    pub fn sym_k(&self) -> (u16, u16) {
        (self.btree_k, self.sym_leaf_k)
    }

    /// Set symbol-table internal-node and leaf-node K values.
    pub fn set_sym_k(&mut self, ik: u16, lk: u16) {
        self.btree_k = ik;
        self.sym_leaf_k = lk;
    }

    /// Indexed-storage B-tree internal-node K value.
    ///
    /// Mirrors `H5Pget_istore_k`.
    pub fn istore_k(&self) -> u16 {
        self.chunk_btree_k
    }

    /// Set indexed-storage B-tree internal-node K value.
    pub fn set_istore_k(&mut self, ik: u16) {
        self.chunk_btree_k = ik;
    }

    /// File-space strategy, persistence flag, and free-space threshold.
    pub fn file_space(&self) -> (FileSpaceStrategy, bool, u64) {
        self.file_space
    }

    /// Set file-space strategy, persistence flag, and free-space threshold.
    pub fn set_file_space(&mut self, strategy: FileSpaceStrategy, persist: bool, threshold: u64) {
        self.file_space = (strategy, persist, threshold);
    }

    /// File-space page size in bytes.
    pub fn file_space_page_size(&self) -> u64 {
        self.file_space_page_size
    }

    /// Set file-space page size in bytes.
    pub fn set_file_space_page_size(&mut self, page_size: u64) {
        self.file_space_page_size = page_size;
    }

    /// Number of shared object-header-message indexes.
    pub fn shared_mesg_nindexes(&self) -> usize {
        self.shared_mesg_indexes.len()
    }

    /// Set the number of shared object-header-message indexes.
    pub fn set_shared_mesg_nindexes(&mut self, count: usize) {
        self.shared_mesg_indexes.resize(
            count,
            SharedMessageIndex {
                message_type_flags: 0,
                minimum_message_size: 0,
            },
        );
    }

    /// Shared object-header-message index configuration by index.
    pub fn shared_mesg_index(&self, _index: usize) -> Option<SharedMessageIndex> {
        self.shared_mesg_indexes.get(_index).copied()
    }

    /// Borrow shared object-header-message index configuration by index.
    pub fn shared_mesg_index_ref(&self, index: usize) -> Option<&SharedMessageIndex> {
        self.shared_mesg_indexes.get(index)
    }

    /// Borrow all shared object-header-message index configurations.
    pub fn shared_mesg_indexes(&self) -> &[SharedMessageIndex] {
        self.shared_mesg_indexes.as_slice()
    }

    /// Set one shared object-header-message index configuration.
    pub fn set_shared_mesg_index(&mut self, index: usize, config: SharedMessageIndex) -> bool {
        if let Some(slot) = self.shared_mesg_indexes.get_mut(index) {
            *slot = config;
            true
        } else {
            false
        }
    }

    /// Shared-message list-to-B-tree and B-tree-to-list phase-change thresholds.
    pub fn shared_mesg_phase_change(&self) -> (u32, u32) {
        self.shared_mesg_phase_change
    }

    /// Set shared-message list/tree phase-change thresholds.
    pub fn set_shared_mesg_phase_change(&mut self, max_list: u32, min_btree: u32) {
        self.shared_mesg_phase_change = (max_list, min_btree);
    }
}
