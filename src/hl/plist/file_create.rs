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
    userblock: u64,
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
    const MAX_FILE_SPACE_PAGE_SIZE: u64 = 1024 * 1024 * 1024;

    /// Default file creation properties used by the pure-Rust writer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Extract file creation properties from a File.
    pub fn from_file(f: &crate::hl::file::File) -> Self {
        let sb = f.superblock();
        let (file_space, file_space_page_size) = read_file_space_info_from_extension(f)
            .unwrap_or(((FileSpaceStrategy::Aggregate, false, 1), 4096));
        let (shared_mesg_indexes, shared_mesg_phase_change) =
            read_shared_message_info_from_extension(f).unwrap_or((Vec::new(), (50, 40)));
        Self {
            superblock_version: sb.version,
            sizeof_addr: sb.sizeof_addr,
            sizeof_size: sb.sizeof_size,
            sym_leaf_k: sb.sym_leaf_k,
            btree_k: sb.snode_btree_k,
            chunk_btree_k: sb.chunk_btree_k,
            userblock: f.userblock(),
            file_space,
            file_space_page_size,
            shared_mesg_indexes,
            shared_mesg_phase_change,
        }
    }

    /// File address and length field sizes in bytes.
    ///
    /// Mirrors the useful read-side subset of `H5Pget_sizes`.
    pub fn sizes(&self) -> (u8, u8) {
        (self.sizeof_addr, self.sizeof_size)
    }

    /// Set the superblock version for newly created files.
    ///
    /// The pure-Rust writer emits the v2/v3 superblock layout. Invalid or
    /// legacy versions are rejected and leave the previous value unchanged.
    pub fn set_superblock_version(&mut self, version: u8) -> bool {
        if !matches!(version, 2 | 3) {
            return false;
        }
        self.superblock_version = version;
        true
    }

    /// Set file address and length field sizes in bytes.
    ///
    /// The writer can emit 2-, 4-, or 8-byte address/length fields. Invalid
    /// widths are rejected and leave the previous values unchanged.
    pub fn set_sizes(&mut self, sizeof_addr: u8, sizeof_size: u8) -> bool {
        if !matches!(sizeof_addr, 2 | 4 | 8) || !matches!(sizeof_size, 2 | 4 | 8) {
            return false;
        }
        self.sizeof_addr = sizeof_addr;
        self.sizeof_size = sizeof_size;
        true
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

    /// Userblock size in bytes.
    pub fn userblock(&self) -> u64 {
        self.userblock
    }

    /// Set userblock size in bytes.
    ///
    /// HDF5 userblocks are either disabled (`0`) or powers of two at least
    /// 512 bytes. Invalid sizes are rejected and leave the current value
    /// unchanged.
    pub fn set_userblock(&mut self, userblock: u64) -> bool {
        if !Self::valid_userblock_size(userblock) {
            return false;
        }
        self.userblock = userblock;
        true
    }

    /// Return whether a userblock size can be represented by this FCPL.
    pub fn valid_userblock_size(userblock: u64) -> bool {
        userblock == 0 || (userblock >= 512 && userblock.is_power_of_two())
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
    ///
    /// The pure-Rust reader accepts the same bounded page-size range when it
    /// parses file-space info messages. Invalid sizes are rejected and leave
    /// the current value unchanged.
    pub fn set_file_space_page_size(&mut self, page_size: u64) -> bool {
        if page_size == 0 || page_size > Self::MAX_FILE_SPACE_PAGE_SIZE {
            return false;
        }
        self.file_space_page_size = page_size;
        true
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

fn read_file_space_info_from_extension(
    f: &crate::hl::file::File,
) -> crate::error::Result<((FileSpaceStrategy, bool, u64), u64)> {
    let sb = f.superblock();
    if sb.ext_addr == crate::io::reader::UNDEF_ADDR {
        return Ok(((FileSpaceStrategy::Aggregate, false, 1), 4096));
    }

    let oh = f.object_header_at(sb.ext_addr)?;
    let Some(msg) = oh
        .messages
        .iter()
        .find(|msg| msg.msg_type == crate::format::object_header::MSG_FILE_SPACE_INFO)
    else {
        return Ok(((FileSpaceStrategy::Aggregate, false, 1), 4096));
    };
    decode_file_space_info_message(&msg.data, sb.sizeof_size)
}

fn read_shared_message_info_from_extension(
    f: &crate::hl::file::File,
) -> crate::error::Result<(Vec<SharedMessageIndex>, (u32, u32))> {
    let sb = f.superblock();
    if sb.ext_addr == crate::io::reader::UNDEF_ADDR {
        return Ok((Vec::new(), (50, 40)));
    }

    let oh = f.object_header_at(sb.ext_addr)?;
    let Some(msg) = oh
        .messages
        .iter()
        .find(|msg| msg.msg_type == crate::format::object_header::MSG_SHARED_MSG_TABLE)
    else {
        return Ok((Vec::new(), (50, 40)));
    };
    let (table_addr, nindexes) = decode_shared_message_table_message(&msg.data, sb.sizeof_addr)?;
    decode_shared_message_table(f, table_addr, nindexes, sb.sizeof_addr)
}

fn decode_shared_message_table_message(
    data: &[u8],
    sizeof_addr: u8,
) -> crate::error::Result<(u64, usize)> {
    let mut pos = 0usize;
    let version = read_u8_at(data, &mut pos, "shared-message table message version")?;
    if version != 0 {
        return Err(crate::error::Error::InvalidFormat(format!(
            "shared-message table message version {version}"
        )));
    }
    let table_addr = read_le_uint_at(
        data,
        &mut pos,
        usize::from(sizeof_addr),
        "shared-message table address",
    )?;
    if table_addr == encoded_undefined_value(sizeof_addr)? {
        return Err(crate::error::Error::InvalidFormat(
            "shared-message table address is undefined".into(),
        ));
    }
    let nindexes = usize::from(read_u8_at(
        data,
        &mut pos,
        "shared-message table index count",
    )?);
    if nindexes == 0 || nindexes > 8 {
        return Err(crate::error::Error::InvalidFormat(
            "shared-message table index count is invalid".into(),
        ));
    }
    Ok((table_addr, nindexes))
}

fn decode_shared_message_table(
    f: &crate::hl::file::File,
    table_addr: u64,
    nindexes: usize,
    sizeof_addr: u8,
) -> crate::error::Result<(Vec<SharedMessageIndex>, (u32, u32))> {
    let per_index = 16usize
        .checked_add(usize::from(sizeof_addr).checked_mul(2).ok_or_else(|| {
            crate::error::Error::InvalidFormat("shared-message table index size overflow".into())
        })?)
        .ok_or_else(|| {
            crate::error::Error::InvalidFormat("shared-message table index size overflow".into())
        })?;
    let table_len = 4usize
        .checked_add(nindexes.checked_mul(per_index).ok_or_else(|| {
            crate::error::Error::InvalidFormat("shared-message table size overflow".into())
        })?)
        .and_then(|len| len.checked_add(4))
        .ok_or_else(|| {
            crate::error::Error::InvalidFormat("shared-message table size overflow".into())
        })?;
    let data = f.read_at(table_addr, table_len)?;
    if data.get(0..4) != Some(b"SMTB") {
        return Err(crate::error::Error::InvalidFormat(
            "shared-message table signature is invalid".into(),
        ));
    }
    let checksum_pos = table_len - 4;
    let stored_checksum = u32::from_le_bytes(
        data[checksum_pos..table_len]
            .try_into()
            .map_err(|_| crate::error::Error::InvalidFormat("checksum slice".into()))?,
    );
    let actual_checksum = crate::format::checksum::checksum_metadata(&data[..checksum_pos]);
    if stored_checksum != actual_checksum {
        return Err(crate::error::Error::InvalidFormat(
            "shared-message table checksum mismatch".into(),
        ));
    }

    let mut pos = 4usize;
    let mut indexes = Vec::with_capacity(nindexes);
    let mut phase_change = (50, 40);
    for index in 0..nindexes {
        let version = read_u8_at(&data, &mut pos, "shared-message index version")?;
        if version != 0 {
            return Err(crate::error::Error::InvalidFormat(format!(
                "shared-message index {index} version {version}"
            )));
        }
        let index_type = read_u8_at(&data, &mut pos, "shared-message index type")?;
        if index_type > 1 {
            return Err(crate::error::Error::InvalidFormat(format!(
                "shared-message index {index} type {index_type} is invalid"
            )));
        }
        let message_type_flags = read_u16_le_at(&data, &mut pos, "shared-message type flags")?;
        let minimum_message_size = read_u32_le_at(&data, &mut pos, "shared-message minimum size")?;
        let max_list = u32::from(read_u16_le_at(
            &data,
            &mut pos,
            "shared-message list cutoff",
        )?);
        let min_btree = u32::from(read_u16_le_at(
            &data,
            &mut pos,
            "shared-message B-tree cutoff",
        )?);
        let _nmessages = read_u32_le_at(&data, &mut pos, "shared-message count")?;
        let _index_addr = read_le_uint_at(
            &data,
            &mut pos,
            usize::from(sizeof_addr),
            "shared-message index address",
        )?;
        let _heap_addr = read_le_uint_at(
            &data,
            &mut pos,
            usize::from(sizeof_addr),
            "shared-message heap address",
        )?;
        if index == 0 {
            phase_change = (max_list, min_btree);
        }
        indexes.push(SharedMessageIndex {
            message_type_flags: u32::from(message_type_flags),
            minimum_message_size,
        });
    }
    Ok((indexes, phase_change))
}

fn decode_file_space_info_message(
    data: &[u8],
    sizeof_size: u8,
) -> crate::error::Result<((FileSpaceStrategy, bool, u64), u64)> {
    let mut pos = 0usize;
    let version = read_u8_at(data, &mut pos, "file-space info version")?;
    if version != 1 {
        return Err(crate::error::Error::InvalidFormat(format!(
            "file-space info message version {version}"
        )));
    }
    let strategy = match read_u8_at(data, &mut pos, "file-space info strategy")? {
        0 => FileSpaceStrategy::FreeSpaceManager,
        1 => FileSpaceStrategy::Page,
        2 => FileSpaceStrategy::Aggregate,
        3 => FileSpaceStrategy::None,
        other => {
            return Err(crate::error::Error::InvalidFormat(format!(
                "file-space info strategy {other} is invalid"
            )));
        }
    };
    let persist = read_u8_at(data, &mut pos, "file-space info persist")? != 0;
    let threshold = read_le_uint_at(
        data,
        &mut pos,
        usize::from(sizeof_size),
        "file-space info threshold",
    )?;
    let page_size = read_le_uint_at(
        data,
        &mut pos,
        usize::from(sizeof_size),
        "file-space info page size",
    )?;
    Ok(((strategy, persist, threshold), page_size))
}

fn read_u8_at(data: &[u8], pos: &mut usize, context: &str) -> crate::error::Result<u8> {
    let byte = data
        .get(*pos)
        .copied()
        .ok_or_else(|| crate::error::Error::InvalidFormat(format!("{context} is truncated")))?;
    *pos += 1;
    Ok(byte)
}

fn read_le_uint_at(
    data: &[u8],
    pos: &mut usize,
    width: usize,
    context: &str,
) -> crate::error::Result<u64> {
    if !matches!(width, 1 | 2 | 4 | 8) {
        return Err(crate::error::Error::InvalidFormat(format!(
            "{context} encoded width {width} is invalid"
        )));
    }
    let end = pos
        .checked_add(width)
        .ok_or_else(|| crate::error::Error::InvalidFormat(format!("{context} offset overflow")))?;
    let bytes = data
        .get(*pos..end)
        .ok_or_else(|| crate::error::Error::InvalidFormat(format!("{context} is truncated")))?;
    let mut value = 0u64;
    for (idx, byte) in bytes.iter().enumerate() {
        value |= u64::from(*byte) << (idx * 8);
    }
    *pos = end;
    Ok(value)
}

fn read_u16_le_at(data: &[u8], pos: &mut usize, context: &str) -> crate::error::Result<u16> {
    let value = read_le_uint_at(data, pos, 2, context)?;
    u16::try_from(value)
        .map_err(|_| crate::error::Error::InvalidFormat(format!("{context} does not fit in u16")))
}

fn read_u32_le_at(data: &[u8], pos: &mut usize, context: &str) -> crate::error::Result<u32> {
    let value = read_le_uint_at(data, pos, 4, context)?;
    u32::try_from(value)
        .map_err(|_| crate::error::Error::InvalidFormat(format!("{context} does not fit in u32")))
}

fn encoded_undefined_value(width: u8) -> crate::error::Result<u64> {
    if !matches!(width, 1 | 2 | 4 | 8) {
        return Err(crate::error::Error::InvalidFormat(format!(
            "address width {width} is invalid"
        )));
    }
    if width == 8 {
        Ok(u64::MAX)
    } else {
        Ok((1u64 << (u32::from(width) * 8)) - 1)
    }
}

impl Default for FileCreate {
    fn default() -> Self {
        Self {
            superblock_version: 3,
            sizeof_addr: 8,
            sizeof_size: 8,
            sym_leaf_k: 4,
            btree_k: 16,
            chunk_btree_k: 32,
            userblock: 0,
            file_space: (FileSpaceStrategy::Aggregate, false, 1),
            file_space_page_size: 4096,
            shared_mesg_indexes: Vec::new(),
            shared_mesg_phase_change: (50, 40),
        }
    }
}
