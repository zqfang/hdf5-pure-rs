use std::io::{Read, Seek, SeekFrom, Write};

use crate::error::{Error, Result};
use crate::format::checksum::checksum_metadata;
use crate::format::extensible_array::hdr::read_header_core as read_extensible_array_header_core;
use crate::io::reader::HdfReader;

use super::MutableFile;

#[derive(Debug, Clone)]
pub(super) struct MutableExtensibleArrayHeader {
    pub(super) class_id: u8,
    pub(super) raw_element_size: usize,
    pub(super) index_block_elements: u8,
    pub(super) data_block_min_elements: usize,
    pub(super) max_data_block_page_elements: usize,
    pub(super) max_index_set: u64,
    pub(super) realized_elements: u64,
    pub(super) index_block_addr: u64,
    pub(super) array_offset_size: u8,
    pub(super) index_block_super_blocks: usize,
    pub(super) index_block_data_block_addrs: usize,
    pub(super) index_block_super_block_addrs: usize,
    pub(super) super_block_info: Vec<MutableExtensibleArraySuperBlockInfo>,
    pub(super) super_block_count: u64,
    pub(super) super_block_size: u64,
    pub(super) data_block_count: u64,
    pub(super) data_block_size: u64,
    pub(super) checksum_pos: u64,
    pub(super) super_block_count_pos: u64,
    pub(super) super_block_size_pos: u64,
    pub(super) data_block_count_pos: u64,
    pub(super) data_block_size_pos: u64,
    pub(super) max_index_set_pos: u64,
    pub(super) realized_elements_pos: u64,
}

#[derive(Debug, Clone)]
pub(super) struct MutableExtensibleArraySuperBlockInfo {
    pub(super) data_blocks: usize,
    pub(super) data_block_elements: usize,
    pub(super) start_index: u64,
    pub(super) start_data_block: u64,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ExtensibleArrayChunkWrite {
    pub(super) element_index: usize,
    pub(super) filtered: bool,
    pub(super) chunk_size_len: usize,
}

impl MutableFile {
    pub(super) fn rewrite_extensible_array_chunk(
        &mut self,
        index_addr: u64,
        info: &crate::hl::dataset::DatasetInfo,
        chunk_coords: &[u64],
        chunk_dims: &[u64],
        chunk_size: u64,
        chunk_addr: u64,
        unfiltered_chunk_bytes: usize,
    ) -> Result<()> {
        let write = self.plan_extensible_array_chunk_write(
            info,
            chunk_coords,
            chunk_dims,
            unfiltered_chunk_bytes,
        )?;

        let mut guard = self.inner.lock();
        let header = Self::read_extensible_array_header(
            &mut guard.reader,
            index_addr,
            write.filtered,
            write.chunk_size_len,
        )?;
        let element_count = usize::try_from(header.max_index_set).map_err(|_| {
            Error::InvalidFormat("extensible array element count does not fit usize".into())
        })?;
        let direct_count = usize::from(header.index_block_elements);
        if write.element_index < element_count {
            let element_pos = Self::locate_extensible_array_element(
                &mut guard.reader,
                index_addr,
                &header,
                write.element_index,
            )?;
            drop(guard);
            self.rewrite_existing_extensible_array_element(
                element_pos,
                chunk_addr,
                chunk_size,
                write.filtered,
                write.chunk_size_len,
            )?;
            return Ok(());
        }
        if write.element_index != element_count {
            return Err(Error::Unsupported(
                "write_chunk can append only the next extensible-array chunk index".into(),
            ));
        }
        if write.element_index < direct_count {
            let element_pos = Self::locate_extensible_array_element(
                &mut guard.reader,
                index_addr,
                &header,
                write.element_index,
            )?;
            drop(guard);
            self.append_direct_extensible_array_element(
                index_addr,
                &header,
                element_pos,
                chunk_addr,
                chunk_size,
                write.filtered,
                write.chunk_size_len,
            )
        } else {
            drop(guard);
            self.append_extensible_array_spillover_element(
                index_addr,
                &header,
                write.element_index,
                chunk_addr,
                chunk_size,
                write.filtered,
                write.chunk_size_len,
            )
        }
    }

    fn rewrite_existing_extensible_array_element(
        &mut self,
        element_pos: u64,
        chunk_addr: u64,
        chunk_size: u64,
        filtered: bool,
        chunk_size_len: usize,
    ) -> Result<()> {
        self.write_extensible_array_element(
            element_pos,
            chunk_addr,
            chunk_size,
            filtered,
            chunk_size_len,
        )
    }

    fn append_direct_extensible_array_element(
        &mut self,
        header_addr: u64,
        header: &MutableExtensibleArrayHeader,
        element_pos: u64,
        chunk_addr: u64,
        chunk_size: u64,
        filtered: bool,
        chunk_size_len: usize,
    ) -> Result<()> {
        self.write_extensible_array_element(
            element_pos,
            chunk_addr,
            chunk_size,
            filtered,
            chunk_size_len,
        )?;
        self.rewrite_extensible_array_header_counts(
            header_addr,
            header,
            Self::checked_u64_add(header.max_index_set, 1, "extensible array max index count")?,
            header.realized_elements.max(Self::checked_u64_add(
                header.max_index_set,
                1,
                "extensible array realized element count",
            )?),
            None,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn append_extensible_array_spillover_element(
        &mut self,
        header_addr: u64,
        header: &MutableExtensibleArrayHeader,
        element_index: usize,
        chunk_addr: u64,
        chunk_size: u64,
        filtered: bool,
        chunk_size_len: usize,
    ) -> Result<()> {
        if crate::io::reader::is_undef_addr(header.index_block_addr) {
            return Err(Error::Unsupported(
                "write_chunk cannot create a missing extensible-array index block yet".into(),
            ));
        }

        let direct_count = usize::from(header.index_block_elements);
        if element_index < direct_count {
            return Err(Error::InvalidFormat(
                "extensible-array spillover append called for index-block element".into(),
            ));
        }
        let spillover_index = element_index - direct_count;
        let Some((super_block_index, super_info)) = header
            .super_block_info
            .iter()
            .enumerate()
            .find(|(_, info)| {
                let Ok(start) = usize::try_from(info.start_index).map_err(|_| ()) else {
                    return false;
                };
                let Ok(span) = info
                    .data_blocks
                    .checked_mul(info.data_block_elements)
                    .ok_or(())
                else {
                    return false;
                };
                let Ok(end) = start.checked_add(span).ok_or(()) else {
                    return false;
                };
                spillover_index >= start && spillover_index < end
            })
        else {
            return Err(Error::Unsupported(
                "extensible-array append index exceeds supported array geometry".into(),
            ));
        };

        let super_start = usize::try_from(super_info.start_index).map_err(|_| {
            Error::InvalidFormat(
                "extensible array super block start index does not fit usize".into(),
            )
        })?;
        let index_in_super = spillover_index.checked_sub(super_start).ok_or_else(|| {
            Error::InvalidFormat("extensible array super block index underflow".into())
        })?;
        let local_data_block_index = index_in_super / super_info.data_block_elements;
        let element_in_block = index_in_super % super_info.data_block_elements;
        let local_block_span = Self::u64_from_usize(
            local_data_block_index,
            "extensible array local data block index",
        )?
        .checked_mul(Self::u64_from_usize(
            super_info.data_block_elements,
            "extensible array data block elements",
        )?)
        .ok_or_else(|| Error::InvalidFormat("extensible array block offset overflow".into()))?;
        let block_offset = Self::checked_u64_add(
            Self::u64_from_usize(direct_count, "extensible array direct element count")?,
            super_info.start_index,
            "extensible array block offset",
        )
        .and_then(|value| {
            Self::checked_u64_add(value, local_block_span, "extensible array block offset")
        })?;

        if super_block_index < header.index_block_super_blocks {
            self.append_extensible_array_index_data_block_element(
                header_addr,
                header,
                super_info,
                local_data_block_index,
                element_in_block,
                block_offset,
                chunk_addr,
                chunk_size,
                filtered,
                chunk_size_len,
            )
        } else {
            self.append_extensible_array_super_block_element(
                header_addr,
                header,
                super_block_index,
                super_info,
                local_data_block_index,
                element_in_block,
                block_offset,
                chunk_addr,
                chunk_size,
                filtered,
                chunk_size_len,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn append_extensible_array_index_data_block_element(
        &mut self,
        header_addr: u64,
        header: &MutableExtensibleArrayHeader,
        super_info: &MutableExtensibleArraySuperBlockInfo,
        local_data_block_index: usize,
        element_in_block: usize,
        block_offset: u64,
        chunk_addr: u64,
        chunk_size: u64,
        filtered: bool,
        chunk_size_len: usize,
    ) -> Result<()> {
        let global_data_block_index = usize::try_from(super_info.start_data_block)
            .map_err(|_| {
                Error::InvalidFormat(
                    "extensible-array data-block start index does not fit usize".into(),
                )
            })
            .and_then(|start| {
                Self::checked_usize_add(
                    start,
                    local_data_block_index,
                    "extensible-array index data-block address index",
                )
            })?;
        if global_data_block_index >= header.index_block_data_block_addrs {
            return Err(Error::InvalidFormat(
                "extensible-array index data-block address is out of bounds".into(),
            ));
        }
        let sa = usize::from(self.superblock.sizeof_addr);
        let data_block_addr_pos = Self::extensible_array_index_data_block_addr_pos(
            header.index_block_addr,
            usize::from(header.index_block_elements),
            header.raw_element_size,
            sa,
            global_data_block_index,
        )?;

        let mut guard = self.inner.lock();
        guard.reader.seek(data_block_addr_pos)?;
        let data_block_addr = guard.reader.read_addr()?;
        drop(guard);

        let data_block_size =
            self.extensible_array_data_block_size(header, super_info.data_block_elements)?;
        if crate::io::reader::is_undef_addr(data_block_addr) {
            if element_in_block != 0 {
                return Err(Error::Unsupported(
                    "write_chunk cannot allocate a sparse extensible-array data block".into(),
                ));
            }
            let new_addr = self.create_extensible_array_data_block(
                header_addr,
                header,
                block_offset,
                super_info.data_block_elements,
                Some((
                    element_in_block,
                    chunk_addr,
                    chunk_size,
                    filtered,
                    chunk_size_len,
                )),
                None,
            )?;
            self.write_handle
                .seek(SeekFrom::Start(data_block_addr_pos))?;
            self.write_handle.write_all(&Self::encode_uint_le(
                new_addr,
                sa,
                "extensible array data block address",
            )?)?;
            self.rewrite_extensible_array_index_block_checksum(
                header,
                Some((global_data_block_index, new_addr)),
                None,
            )?;
            self.rewrite_extensible_array_header_counts(
                header_addr,
                header,
                Self::checked_u64_add(header.max_index_set, 1, "extensible array max index count")?,
                header.realized_elements.max(Self::checked_u64_add(
                    block_offset,
                    Self::u64_from_usize(
                        super_info.data_block_elements,
                        "extensible array data block elements",
                    )?,
                    "extensible array realized element count",
                )?),
                Some((
                    Self::checked_u64_add(
                        header.data_block_count,
                        1,
                        "extensible array data block count",
                    )?,
                    Self::checked_u64_add(
                        header.data_block_size,
                        Self::u64_from_usize(
                            data_block_size,
                            "extensible array data block byte size",
                        )?,
                        "extensible array data block byte size",
                    )?,
                )),
                None,
            )?;
            return Ok(());
        }

        self.write_extensible_array_data_block_element(
            data_block_addr,
            header,
            super_info.data_block_elements,
            element_in_block,
            chunk_addr,
            chunk_size,
            filtered,
            chunk_size_len,
            None,
        )?;
        self.rewrite_extensible_array_header_counts(
            header_addr,
            header,
            Self::checked_u64_add(header.max_index_set, 1, "extensible array max index count")?,
            header.realized_elements,
            None,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn append_extensible_array_super_block_element(
        &mut self,
        header_addr: u64,
        header: &MutableExtensibleArrayHeader,
        super_block_index: usize,
        super_info: &MutableExtensibleArraySuperBlockInfo,
        local_data_block_index: usize,
        element_in_block: usize,
        block_offset: u64,
        chunk_addr: u64,
        chunk_size: u64,
        filtered: bool,
        chunk_size_len: usize,
    ) -> Result<()> {
        let super_block_addr_index = super_block_index - header.index_block_super_blocks;
        if super_block_addr_index >= header.index_block_super_block_addrs {
            return Err(Error::InvalidFormat(
                "extensible-array super-block address is out of bounds".into(),
            ));
        }
        let sa = usize::from(self.superblock.sizeof_addr);
        let super_block_addr_pos = Self::extensible_array_index_super_block_addr_pos(
            header.index_block_addr,
            usize::from(header.index_block_elements),
            header.raw_element_size,
            header.index_block_data_block_addrs,
            sa,
            super_block_addr_index,
        )?;

        let mut guard = self.inner.lock();
        guard.reader.seek(super_block_addr_pos)?;
        let super_block_addr = guard.reader.read_addr()?;
        drop(guard);

        let super_block_size = self.extensible_array_super_block_size(header, super_info)?;
        let data_block_size =
            self.extensible_array_data_block_size(header, super_info.data_block_elements)?;
        if crate::io::reader::is_undef_addr(super_block_addr) {
            if local_data_block_index != 0 || element_in_block != 0 {
                return Err(Error::Unsupported(
                    "write_chunk cannot allocate a sparse extensible-array super block".into(),
                ));
            }
            let (new_super_addr, new_data_addr) = self.create_extensible_array_super_block(
                header_addr,
                header,
                super_info,
                super_block_index,
                local_data_block_index,
                element_in_block,
                block_offset,
                chunk_addr,
                chunk_size,
                filtered,
                chunk_size_len,
            )?;
            self.write_handle
                .seek(SeekFrom::Start(super_block_addr_pos))?;
            self.write_handle.write_all(&Self::encode_uint_le(
                new_super_addr,
                sa,
                "extensible array super block address",
            )?)?;
            self.rewrite_extensible_array_index_block_checksum(
                header,
                None,
                Some((super_block_addr_index, new_super_addr)),
            )?;
            let _ = new_data_addr;
            self.rewrite_extensible_array_header_counts(
                header_addr,
                header,
                Self::checked_u64_add(header.max_index_set, 1, "extensible array max index count")?,
                header.realized_elements.max(Self::checked_u64_add(
                    block_offset,
                    Self::u64_from_usize(
                        super_info.data_block_elements,
                        "extensible array data block elements",
                    )?,
                    "extensible array realized element count",
                )?),
                Some((
                    Self::checked_u64_add(
                        header.data_block_count,
                        1,
                        "extensible array data block count",
                    )?,
                    Self::checked_u64_add(
                        header.data_block_size,
                        Self::u64_from_usize(
                            data_block_size,
                            "extensible array data block byte size",
                        )?,
                        "extensible array data block byte size",
                    )?,
                )),
                Some((
                    Self::checked_u64_add(
                        header.super_block_count,
                        1,
                        "extensible array super block count",
                    )?,
                    Self::checked_u64_add(
                        header.super_block_size,
                        Self::u64_from_usize(
                            super_block_size,
                            "extensible array super block byte size",
                        )?,
                        "extensible array super block byte size",
                    )?,
                )),
            )?;
            return Ok(());
        }

        let data_block_addr = self.read_extensible_array_super_block_data_addr(
            super_block_addr,
            header,
            super_info,
            local_data_block_index,
        )?;
        if crate::io::reader::is_undef_addr(data_block_addr) {
            if element_in_block != 0 {
                return Err(Error::Unsupported(
                    "write_chunk cannot allocate a sparse extensible-array super-block data block"
                        .into(),
                ));
            }
            let page_index = self.extensible_array_page_index(
                header,
                super_info.data_block_elements,
                element_in_block,
            );
            let new_data_addr = self.create_extensible_array_data_block(
                header_addr,
                header,
                block_offset,
                super_info.data_block_elements,
                Some((
                    element_in_block,
                    chunk_addr,
                    chunk_size,
                    filtered,
                    chunk_size_len,
                )),
                page_index,
            )?;
            self.rewrite_extensible_array_super_block(
                super_block_addr,
                header,
                super_info,
                Some((local_data_block_index, new_data_addr)),
                page_index.map(|idx| (local_data_block_index, idx)),
            )?;
            self.rewrite_extensible_array_header_counts(
                header_addr,
                header,
                Self::checked_u64_add(header.max_index_set, 1, "extensible array max index count")?,
                header.realized_elements.max(Self::checked_u64_add(
                    block_offset,
                    Self::u64_from_usize(
                        super_info.data_block_elements,
                        "extensible array data block elements",
                    )?,
                    "extensible array realized element count",
                )?),
                Some((
                    Self::checked_u64_add(
                        header.data_block_count,
                        1,
                        "extensible array data block count",
                    )?,
                    Self::checked_u64_add(
                        header.data_block_size,
                        Self::u64_from_usize(
                            data_block_size,
                            "extensible array data block byte size",
                        )?,
                        "extensible array data block byte size",
                    )?,
                )),
                None,
            )?;
            return Ok(());
        }

        let page_index = self.extensible_array_page_index(
            header,
            super_info.data_block_elements,
            element_in_block,
        );
        self.write_extensible_array_data_block_element(
            data_block_addr,
            header,
            super_info.data_block_elements,
            element_in_block,
            chunk_addr,
            chunk_size,
            filtered,
            chunk_size_len,
            page_index,
        )?;
        if let Some(page_index) = page_index {
            self.rewrite_extensible_array_super_block(
                super_block_addr,
                header,
                super_info,
                None,
                Some((local_data_block_index, page_index)),
            )?;
        }
        self.rewrite_extensible_array_header_counts(
            header_addr,
            header,
            Self::checked_u64_add(header.max_index_set, 1, "extensible array max index count")?,
            header.realized_elements,
            None,
            None,
        )
    }

    fn extensible_array_super_block_size(
        &self,
        header: &MutableExtensibleArrayHeader,
        super_info: &MutableExtensibleArraySuperBlockInfo,
    ) -> Result<usize> {
        let page_init_size =
            Self::extensible_array_page_init_size(header, super_info.data_block_elements);
        let prefix = Self::checked_usize_add(
            4 + 1 + 1,
            usize::from(self.superblock.sizeof_addr),
            "extensible array super block size",
        )
        .and_then(|value| {
            Self::checked_usize_add(
                value,
                usize::from(header.array_offset_size),
                "extensible array super block size",
            )
        })?;
        let page_init_bytes = Self::checked_usize_mul(
            super_info.data_blocks,
            page_init_size,
            "extensible array super block size",
        )?;
        let addr_bytes = Self::checked_usize_mul(
            super_info.data_blocks,
            usize::from(self.superblock.sizeof_addr),
            "extensible array super block size",
        )?;
        Self::checked_usize_add(prefix, page_init_bytes, "extensible array super block size")
            .and_then(|value| {
                Self::checked_usize_add(value, addr_bytes, "extensible array super block size")
            })
            .and_then(|value| {
                Self::checked_usize_add(value, 4, "extensible array super block size")
            })
    }

    fn extensible_array_page_index(
        &self,
        header: &MutableExtensibleArrayHeader,
        data_block_elements: usize,
        element_in_block: usize,
    ) -> Option<usize> {
        if Self::extensible_array_data_block_pages(header, data_block_elements) == 0 {
            None
        } else {
            Some(element_in_block / header.max_data_block_page_elements)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn create_extensible_array_super_block(
        &mut self,
        header_addr: u64,
        header: &MutableExtensibleArrayHeader,
        super_info: &MutableExtensibleArraySuperBlockInfo,
        _super_block_index: usize,
        local_data_block_index: usize,
        element_in_block: usize,
        block_offset: u64,
        chunk_addr: u64,
        chunk_size: u64,
        filtered: bool,
        chunk_size_len: usize,
    ) -> Result<(u64, u64)> {
        let page_index = self.extensible_array_page_index(
            header,
            super_info.data_block_elements,
            element_in_block,
        );
        let data_block_addr = self.create_extensible_array_data_block(
            header_addr,
            header,
            block_offset,
            super_info.data_block_elements,
            Some((
                element_in_block,
                chunk_addr,
                chunk_size,
                filtered,
                chunk_size_len,
            )),
            page_index,
        )?;

        let super_block_size = self.extensible_array_super_block_size(header, super_info)?;
        let super_block_addr = self.append_aligned_zeros(super_block_size, 8)?;
        let block = self.encode_extensible_array_super_block(
            header_addr,
            header,
            super_info,
            local_data_block_index,
            page_index,
            data_block_addr,
            super_block_size,
        )?;
        self.write_handle.seek(SeekFrom::Start(super_block_addr))?;
        self.write_handle.write_all(&block)?;
        Ok((super_block_addr, data_block_addr))
    }

    /// Pure encoder for an extensible-array super block (EASB magic +
    /// header-addr + offset + page-init bitmap + data-block addr table +
    /// checksum). Mirrors the serialize half of libhdf5's
    /// `H5EA__cache_sblock_serialize`.
    #[allow(clippy::too_many_arguments)]
    fn encode_extensible_array_super_block(
        &self,
        header_addr: u64,
        header: &MutableExtensibleArrayHeader,
        super_info: &MutableExtensibleArraySuperBlockInfo,
        local_data_block_index: usize,
        page_index: Option<usize>,
        data_block_addr: u64,
        super_block_size: usize,
    ) -> Result<Vec<u8>> {
        let page_init_size =
            Self::extensible_array_page_init_size(header, super_info.data_block_elements);
        let sa = usize::from(self.superblock.sizeof_addr);
        let array_offset_size = usize::from(header.array_offset_size);
        let block_offset = Self::checked_u64_add(
            u64::from(header.index_block_elements),
            super_info.start_index,
            "extensible array super block offset",
        )?;
        let mut block = Vec::with_capacity(super_block_size);
        block.extend_from_slice(b"EASB");
        block.push(0);
        block.push(header.class_id);
        block.extend_from_slice(&Self::encode_uint_le(
            header_addr,
            sa,
            "extensible array super block header address",
        )?);
        block.extend_from_slice(&Self::encode_uint_le(
            block_offset,
            array_offset_size,
            "extensible array super block offset",
        )?);
        let page_init_len = Self::checked_usize_mul(
            super_info.data_blocks,
            page_init_size,
            "extensible array super block page-init size",
        )?;
        let mut page_init = vec![0u8; page_init_len];
        if let Some(page_index) = page_index {
            let start = Self::checked_usize_mul(
                local_data_block_index,
                page_init_size,
                "extensible array super block page-init offset",
            )?;
            let end = Self::checked_usize_add(
                start,
                page_init_size,
                "extensible array super block page-init offset",
            )?;
            Self::set_extensible_array_page_init_bit(
                page_init.get_mut(start..end).ok_or_else(|| {
                    Error::InvalidFormat("extensible array page-init slice out of bounds".into())
                })?,
                page_index,
            )?;
        }
        block.extend_from_slice(&page_init);
        let fill_addr =
            Self::undefined_addr_bytes(sa, "extensible array super block data address")?;
        for idx in 0..super_info.data_blocks {
            if idx == local_data_block_index {
                block.extend_from_slice(&Self::encode_uint_le(
                    data_block_addr,
                    sa,
                    "extensible array super block data address",
                )?);
            } else {
                block.extend_from_slice(&fill_addr);
            }
        }
        let checksum = checksum_metadata(&block);
        block.extend_from_slice(&checksum.to_le_bytes());
        Ok(block)
    }

    fn read_extensible_array_super_block_data_addr(
        &mut self,
        super_block_addr: u64,
        header: &MutableExtensibleArrayHeader,
        super_info: &MutableExtensibleArraySuperBlockInfo,
        local_data_block_index: usize,
    ) -> Result<u64> {
        let page_init_size =
            Self::extensible_array_page_init_size(header, super_info.data_block_elements);
        let prefix_size = Self::checked_usize_add(
            4 + 1 + 1,
            usize::from(self.superblock.sizeof_addr),
            "extensible array super block address offset",
        )
        .and_then(|value| {
            Self::checked_usize_add(
                value,
                usize::from(header.array_offset_size),
                "extensible array super block address offset",
            )
        })?;
        let page_init_bytes = Self::checked_usize_mul(
            super_info.data_blocks,
            page_init_size,
            "extensible array super block address offset",
        )?;
        let addr_index_bytes = Self::checked_usize_mul(
            local_data_block_index,
            usize::from(self.superblock.sizeof_addr),
            "extensible array super block address offset",
        )?;
        let addr_offset = Self::checked_usize_add(
            prefix_size,
            page_init_bytes,
            "extensible array super block address offset",
        )
        .and_then(|value| {
            Self::checked_usize_add(
                value,
                addr_index_bytes,
                "extensible array super block address offset",
            )
        })?;
        let addr_pos = Self::checked_u64_add(
            super_block_addr,
            Self::u64_from_usize(addr_offset, "extensible array super block address offset")?,
            "extensible array super block address",
        )?;
        let mut guard = self.inner.lock();
        guard.reader.seek(addr_pos)?;
        guard.reader.read_addr()
    }

    fn rewrite_extensible_array_super_block(
        &mut self,
        super_block_addr: u64,
        header: &MutableExtensibleArrayHeader,
        super_info: &MutableExtensibleArraySuperBlockInfo,
        data_block_addr: Option<(usize, u64)>,
        page_init_bit: Option<(usize, usize)>,
    ) -> Result<()> {
        let block_size = self.extensible_array_super_block_size(header, super_info)?;
        let check_len = block_size.checked_sub(4).ok_or_else(|| {
            Error::InvalidFormat("extensible array super block checksum span overflow".into())
        })?;
        self.write_handle.flush()?;
        let mut block = self.read_fresh_bytes(super_block_addr, check_len)?;

        let page_init_size =
            Self::extensible_array_page_init_size(header, super_info.data_block_elements);
        let page_init_start = Self::checked_usize_add(
            4 + 1 + 1,
            usize::from(self.superblock.sizeof_addr),
            "extensible array super block page-init offset",
        )
        .and_then(|value| {
            Self::checked_usize_add(
                value,
                usize::from(header.array_offset_size),
                "extensible array super block page-init offset",
            )
        })?;
        if let Some((data_block_index, page_index)) = page_init_bit {
            let start = Self::checked_usize_add(
                page_init_start,
                Self::checked_usize_mul(
                    data_block_index,
                    page_init_size,
                    "extensible array super block page-init offset",
                )?,
                "extensible array super block page-init offset",
            )?;
            let end = Self::checked_usize_add(
                start,
                page_init_size,
                "extensible array super block page-init offset",
            )?;
            Self::set_extensible_array_page_init_bit(
                block.get_mut(start..end).ok_or_else(|| {
                    Error::InvalidFormat("extensible array page-init slice out of bounds".into())
                })?,
                page_index,
            )?;
        }
        if let Some((data_block_index, addr)) = data_block_addr {
            let addr_start = Self::checked_usize_add(
                page_init_start,
                Self::checked_usize_mul(
                    super_info.data_blocks,
                    page_init_size,
                    "extensible array super block address offset",
                )?,
                "extensible array super block address offset",
            )?;
            let pos = Self::checked_usize_add(
                addr_start,
                Self::checked_usize_mul(
                    data_block_index,
                    usize::from(self.superblock.sizeof_addr),
                    "extensible array super block address offset",
                )?,
                "extensible array super block address offset",
            )?;
            let end = Self::checked_usize_add(
                pos,
                usize::from(self.superblock.sizeof_addr),
                "extensible array super block address offset",
            )?;
            block
                .get_mut(pos..end)
                .ok_or_else(|| {
                    Error::InvalidFormat(
                        "extensible array super block address slice out of bounds".into(),
                    )
                })?
                .copy_from_slice(&Self::encode_uint_le(
                    addr,
                    usize::from(self.superblock.sizeof_addr),
                    "extensible array super block address",
                )?);
        }
        let checksum = checksum_metadata(&block);
        self.write_handle.seek(SeekFrom::Start(super_block_addr))?;
        self.write_handle.write_all(&block)?;
        self.write_handle.write_all(&checksum.to_le_bytes())?;
        Ok(())
    }

    fn rewrite_extensible_array_index_block_checksum(
        &mut self,
        header: &MutableExtensibleArrayHeader,
        data_block_addr: Option<(usize, u64)>,
        super_block_addr: Option<(usize, u64)>,
    ) -> Result<()> {
        let sa = usize::from(self.superblock.sizeof_addr);
        let index_prefix_size = Self::extensible_array_index_prefix_size(sa)?;
        let inline_bytes = Self::extensible_array_index_inline_bytes(
            usize::from(header.index_block_elements),
            header.raw_element_size,
        )?;
        let data_block_addr_bytes = Self::checked_usize_mul(
            header.index_block_data_block_addrs,
            sa,
            "extensible array index block checksum span",
        )?;
        let super_block_addr_bytes = Self::checked_usize_mul(
            header.index_block_super_block_addrs,
            sa,
            "extensible array index block checksum span",
        )?;
        let check_len = Self::checked_usize_add(
            index_prefix_size,
            inline_bytes,
            "extensible array index block checksum span",
        )
        .and_then(|value| {
            Self::checked_usize_add(
                value,
                data_block_addr_bytes,
                "extensible array index block checksum span",
            )
        })
        .and_then(|value| {
            Self::checked_usize_add(
                value,
                super_block_addr_bytes,
                "extensible array index block checksum span",
            )
        })?;
        self.write_handle.flush()?;
        let mut index_bytes = self.read_fresh_bytes(header.index_block_addr, check_len)?;
        if let Some((data_block_index, data_block_addr)) = data_block_addr {
            let data_block_addr_offset = Self::checked_usize_add(
                index_prefix_size,
                inline_bytes,
                "extensible array data block address offset",
            )
            .and_then(|value| {
                Self::checked_usize_add(
                    value,
                    Self::checked_usize_mul(
                        data_block_index,
                        sa,
                        "extensible array data block address offset",
                    )?,
                    "extensible array data block address offset",
                )
            })?;
            let end = Self::checked_usize_add(
                data_block_addr_offset,
                sa,
                "extensible array data block address offset",
            )?;
            index_bytes
                .get_mut(data_block_addr_offset..end)
                .ok_or_else(|| {
                    Error::InvalidFormat(
                        "extensible array data block address slice out of bounds".into(),
                    )
                })?
                .copy_from_slice(&Self::encode_uint_le(
                    data_block_addr,
                    sa,
                    "extensible array data block address",
                )?);
        }
        if let Some((super_block_index, super_block_addr)) = super_block_addr {
            let super_block_addr_offset = Self::checked_usize_add(
                index_prefix_size,
                inline_bytes,
                "extensible array super block address offset",
            )
            .and_then(|value| {
                Self::checked_usize_add(
                    value,
                    data_block_addr_bytes,
                    "extensible array super block address offset",
                )
            })
            .and_then(|value| {
                Self::checked_usize_add(
                    value,
                    Self::checked_usize_mul(
                        super_block_index,
                        sa,
                        "extensible array super block address offset",
                    )?,
                    "extensible array super block address offset",
                )
            })?;
            let end = Self::checked_usize_add(
                super_block_addr_offset,
                sa,
                "extensible array super block address offset",
            )?;
            index_bytes
                .get_mut(super_block_addr_offset..end)
                .ok_or_else(|| {
                    Error::InvalidFormat(
                        "extensible array super block address slice out of bounds".into(),
                    )
                })?
                .copy_from_slice(&Self::encode_uint_le(
                    super_block_addr,
                    sa,
                    "extensible array super block address",
                )?);
        }
        let checksum = checksum_metadata(&index_bytes);
        let checksum_addr = Self::checked_u64_add(
            header.index_block_addr,
            Self::u64_from_usize(check_len, "extensible array index block checksum span")?,
            "extensible array index block checksum address",
        )?;
        self.write_handle.seek(SeekFrom::Start(checksum_addr))?;
        self.write_handle.write_all(&checksum.to_le_bytes())?;
        Ok(())
    }

    fn rewrite_extensible_array_header_counts(
        &mut self,
        header_addr: u64,
        header: &MutableExtensibleArrayHeader,
        new_max_index_set: u64,
        new_realized_elements: u64,
        data_block_counts: Option<(u64, u64)>,
        super_block_counts: Option<(u64, u64)>,
    ) -> Result<()> {
        let ss = usize::from(self.superblock.sizeof_size);
        if let Some((super_block_count, super_block_size)) = super_block_counts {
            self.write_handle
                .seek(SeekFrom::Start(header.super_block_count_pos))?;
            self.write_uint_le(super_block_count, ss)?;
            self.write_handle
                .seek(SeekFrom::Start(header.super_block_size_pos))?;
            self.write_uint_le(super_block_size, ss)?;
        }
        if let Some((data_block_count, data_block_size)) = data_block_counts {
            self.write_handle
                .seek(SeekFrom::Start(header.data_block_count_pos))?;
            self.write_uint_le(data_block_count, ss)?;
            self.write_handle
                .seek(SeekFrom::Start(header.data_block_size_pos))?;
            self.write_uint_le(data_block_size, ss)?;
        }
        self.write_handle
            .seek(SeekFrom::Start(header.max_index_set_pos))?;
        self.write_uint_le(new_max_index_set, ss)?;
        self.write_handle
            .seek(SeekFrom::Start(header.realized_elements_pos))?;
        self.write_uint_le(new_realized_elements, ss)?;

        let check_len = header
            .checksum_pos
            .checked_sub(header_addr)
            .and_then(|span| usize::try_from(span).ok())
            .ok_or_else(|| {
                Error::InvalidFormat("extensible array header checksum span is too large".into())
            })?;
        self.write_handle.flush()?;
        let mut header_bytes = self.read_fresh_bytes(header_addr, check_len)?;
        if let Some((super_block_count, super_block_size)) = super_block_counts {
            let offset = Self::header_relative_offset(
                header.super_block_count_pos,
                header_addr,
                ss,
                header_bytes.len(),
                "extensible array super block count",
            )?;
            Self::patch_header_uint(
                &mut header_bytes,
                offset,
                ss,
                super_block_count,
                "extensible array super block count",
            )?;
            let offset = Self::header_relative_offset(
                header.super_block_size_pos,
                header_addr,
                ss,
                header_bytes.len(),
                "extensible array super block size",
            )?;
            Self::patch_header_uint(
                &mut header_bytes,
                offset,
                ss,
                super_block_size,
                "extensible array super block size",
            )?;
        }
        if let Some((data_block_count, data_block_size)) = data_block_counts {
            let offset = Self::header_relative_offset(
                header.data_block_count_pos,
                header_addr,
                ss,
                header_bytes.len(),
                "extensible array data block count",
            )?;
            Self::patch_header_uint(
                &mut header_bytes,
                offset,
                ss,
                data_block_count,
                "extensible array data block count",
            )?;
            let offset = Self::header_relative_offset(
                header.data_block_size_pos,
                header_addr,
                ss,
                header_bytes.len(),
                "extensible array data block size",
            )?;
            Self::patch_header_uint(
                &mut header_bytes,
                offset,
                ss,
                data_block_size,
                "extensible array data block size",
            )?;
        }
        let offset = Self::header_relative_offset(
            header.max_index_set_pos,
            header_addr,
            ss,
            header_bytes.len(),
            "extensible array max index",
        )?;
        Self::patch_header_uint(
            &mut header_bytes,
            offset,
            ss,
            new_max_index_set,
            "extensible array max index",
        )?;
        let offset = Self::header_relative_offset(
            header.realized_elements_pos,
            header_addr,
            ss,
            header_bytes.len(),
            "extensible array realized elements",
        )?;
        Self::patch_header_uint(
            &mut header_bytes,
            offset,
            ss,
            new_realized_elements,
            "extensible array realized elements",
        )?;
        let checksum = checksum_metadata(&header_bytes);
        self.write_handle
            .seek(SeekFrom::Start(header.checksum_pos))?;
        self.write_handle.write_all(&checksum.to_le_bytes())?;
        Ok(())
    }

    pub(super) fn read_extensible_array_header<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        addr: u64,
        filtered: bool,
        chunk_size_len: usize,
    ) -> Result<MutableExtensibleArrayHeader> {
        let parsed = read_extensible_array_header_core(reader, addr)?;
        let class_id = parsed.class_id;
        let expected_class = if filtered { 1 } else { 0 };
        if class_id != expected_class {
            return Err(Error::InvalidFormat(format!(
                "extensible array class {class_id} does not match filtered={filtered}"
            )));
        }
        let raw_element_size = parsed.raw_element_size;
        let expected_element_size = if filtered {
            Self::checked_usize_add(
                Self::checked_usize_add(
                    usize::from(reader.sizeof_addr()),
                    chunk_size_len,
                    "extensible array raw element size",
                )?,
                4,
                "extensible array raw element size",
            )?
        } else {
            usize::from(reader.sizeof_addr())
        };
        if raw_element_size != expected_element_size {
            return Err(Error::InvalidFormat(format!(
                "extensible array raw element size {raw_element_size} does not match expected {expected_element_size}"
            )));
        }
        let super_block_info = Self::build_mutable_extensible_array_super_block_info(
            parsed.derived_super_block_count,
            parsed.data_block_min_elements,
        )?;

        Ok(MutableExtensibleArrayHeader {
            class_id,
            raw_element_size,
            index_block_elements: parsed.index_block_elements,
            data_block_min_elements: parsed.data_block_min_elements,
            max_data_block_page_elements: parsed.data_block_page_elements,
            max_index_set: parsed.max_index_set,
            realized_elements: parsed.realized_elements,
            index_block_addr: parsed.index_block_addr,
            array_offset_size: parsed.array_offset_size,
            index_block_super_blocks: parsed.index_block_super_blocks,
            index_block_data_block_addrs: parsed.index_block_data_block_addrs,
            index_block_super_block_addrs: parsed.index_block_super_block_addrs,
            super_block_info,
            super_block_count: parsed.super_block_count,
            super_block_size: parsed.super_block_size,
            data_block_count: parsed.data_block_count,
            data_block_size: parsed.data_block_size,
            checksum_pos: parsed.checksum_pos,
            super_block_count_pos: parsed.super_block_count_pos,
            super_block_size_pos: parsed.super_block_size_pos,
            data_block_count_pos: parsed.data_block_count_pos,
            data_block_size_pos: parsed.data_block_size_pos,
            max_index_set_pos: parsed.max_index_set_pos,
            realized_elements_pos: parsed.realized_elements_pos,
        })
    }

    fn build_mutable_extensible_array_super_block_info(
        count: usize,
        min_data_block_elements: usize,
    ) -> Result<Vec<MutableExtensibleArraySuperBlockInfo>> {
        let mut infos = Vec::with_capacity(count);
        let mut start_index = 0u64;
        let mut start_data_block = 0u64;
        for index in 0..count {
            let data_blocks = 1usize
                .checked_shl(u32::try_from(index / 2).map_err(|_| {
                    Error::InvalidFormat("extensible array data block shift overflow".into())
                })?)
                .ok_or_else(|| {
                    Error::InvalidFormat("extensible array data block count overflow".into())
                })?;
            let data_block_elements = min_data_block_elements
                .checked_mul(
                    1usize
                        .checked_shl(u32::try_from(index.div_ceil(2)).map_err(|_| {
                            Error::InvalidFormat(
                                "extensible array data block element shift overflow".into(),
                            )
                        })?)
                        .ok_or_else(|| {
                            Error::InvalidFormat(
                                "extensible array data block element count overflow".into(),
                            )
                        })?,
                )
                .ok_or_else(|| {
                    Error::InvalidFormat("extensible array data block size overflow".into())
                })?;
            infos.push(MutableExtensibleArraySuperBlockInfo {
                data_blocks,
                data_block_elements,
                start_index,
                start_data_block,
            });
            let index_span = Self::u64_from_usize(data_blocks, "extensible array data blocks")?
                .checked_mul(Self::u64_from_usize(
                    data_block_elements,
                    "extensible array data block elements",
                )?)
                .ok_or_else(|| {
                    Error::InvalidFormat("extensible array start index overflow".into())
                })?;
            start_index = start_index.checked_add(index_span).ok_or_else(|| {
                Error::InvalidFormat("extensible array start index overflow".into())
            })?;
            start_data_block = start_data_block
                .checked_add(Self::u64_from_usize(
                    data_blocks,
                    "extensible array data blocks",
                )?)
                .ok_or_else(|| {
                    Error::InvalidFormat("extensible array data block index overflow".into())
                })?;
        }
        Ok(infos)
    }

    pub(super) fn extensible_array_data_block_pages(
        header: &MutableExtensibleArrayHeader,
        data_block_elements: usize,
    ) -> usize {
        if data_block_elements > header.max_data_block_page_elements {
            data_block_elements / header.max_data_block_page_elements
        } else {
            0
        }
    }

    pub(super) fn extensible_array_page_init_size(
        header: &MutableExtensibleArrayHeader,
        data_block_elements: usize,
    ) -> usize {
        let pages = Self::extensible_array_data_block_pages(header, data_block_elements);
        if pages > 0 {
            pages.div_ceil(8)
        } else {
            0
        }
    }

    pub(super) fn set_extensible_array_page_init_bit(bytes: &mut [u8], bit: usize) -> Result<()> {
        let Some(byte) = bytes.get_mut(bit / 8) else {
            return Err(Error::InvalidFormat(
                "extensible array page-init bit index out of bounds".into(),
            ));
        };
        *byte |= 0x80 >> (bit % 8);
        Ok(())
    }

    fn checked_usize_add(lhs: usize, rhs: usize, context: &str) -> Result<usize> {
        lhs.checked_add(rhs)
            .ok_or_else(|| Error::InvalidFormat(format!("{context} overflow")))
    }

    fn checked_usize_mul(lhs: usize, rhs: usize, context: &str) -> Result<usize> {
        lhs.checked_mul(rhs)
            .ok_or_else(|| Error::InvalidFormat(format!("{context} overflow")))
    }

    fn checked_u64_add(lhs: u64, rhs: u64, context: &str) -> Result<u64> {
        lhs.checked_add(rhs)
            .ok_or_else(|| Error::InvalidFormat(format!("{context} overflow")))
    }

    fn u64_from_usize(value: usize, context: &str) -> Result<u64> {
        u64::try_from(value)
            .map_err(|_| Error::InvalidFormat(format!("{context} does not fit in u64")))
    }

    fn header_relative_offset(
        field_pos: u64,
        header_addr: u64,
        field_len: usize,
        buffer_len: usize,
        context: &str,
    ) -> Result<usize> {
        let offset = field_pos
            .checked_sub(header_addr)
            .and_then(|span| usize::try_from(span).ok())
            .ok_or_else(|| Error::InvalidFormat(format!("{context} offset is out of range")))?;
        let end = Self::checked_usize_add(offset, field_len, context)?;
        if end > buffer_len {
            return Err(Error::InvalidFormat(format!(
                "{context} field exceeds extensible array header"
            )));
        }
        Ok(offset)
    }

    fn patch_header_uint(
        header_bytes: &mut [u8],
        offset: usize,
        len: usize,
        value: u64,
        context: &str,
    ) -> Result<()> {
        let end = Self::checked_usize_add(offset, len, context)?;
        let window = header_bytes
            .get_mut(offset..end)
            .ok_or_else(|| Error::InvalidFormat(format!("{context} patch window is truncated")))?;
        let raw = value.to_le_bytes();
        let src = raw
            .get(..len)
            .ok_or_else(|| Error::InvalidFormat(format!("{context} integer width is invalid")))?;
        window.copy_from_slice(src);
        Ok(())
    }

    fn extensible_array_index_prefix_size(sizeof_addr: usize) -> Result<usize> {
        Self::checked_usize_add(
            4 + 1 + 1,
            sizeof_addr,
            "extensible array index block prefix size",
        )
    }

    fn extensible_array_index_inline_bytes(
        index_block_elements: usize,
        raw_element_size: usize,
    ) -> Result<usize> {
        Self::checked_usize_mul(
            index_block_elements,
            raw_element_size,
            "extensible array index block inline element span",
        )
    }

    fn extensible_array_index_data_block_addr_pos(
        index_block_addr: u64,
        index_block_elements: usize,
        raw_element_size: usize,
        sizeof_addr: usize,
        data_block_index: usize,
    ) -> Result<u64> {
        let offset = Self::checked_usize_add(
            Self::extensible_array_index_prefix_size(sizeof_addr)?,
            Self::extensible_array_index_inline_bytes(index_block_elements, raw_element_size)?,
            "extensible array data block address offset",
        )
        .and_then(|value| {
            Self::checked_usize_add(
                value,
                Self::checked_usize_mul(
                    data_block_index,
                    sizeof_addr,
                    "extensible array data block address offset",
                )?,
                "extensible array data block address offset",
            )
        })?;
        Self::checked_u64_add(
            index_block_addr,
            Self::u64_from_usize(offset, "extensible array data block address offset")?,
            "extensible array data block address",
        )
    }

    fn extensible_array_index_super_block_addr_pos(
        index_block_addr: u64,
        index_block_elements: usize,
        raw_element_size: usize,
        index_block_data_block_addrs: usize,
        sizeof_addr: usize,
        super_block_index: usize,
    ) -> Result<u64> {
        let offset = Self::checked_usize_add(
            Self::extensible_array_index_prefix_size(sizeof_addr)?,
            Self::extensible_array_index_inline_bytes(index_block_elements, raw_element_size)?,
            "extensible array super block address offset",
        )
        .and_then(|value| {
            Self::checked_usize_add(
                value,
                Self::checked_usize_mul(
                    index_block_data_block_addrs,
                    sizeof_addr,
                    "extensible array super block address offset",
                )?,
                "extensible array super block address offset",
            )
        })
        .and_then(|value| {
            Self::checked_usize_add(
                value,
                Self::checked_usize_mul(
                    super_block_index,
                    sizeof_addr,
                    "extensible array super block address offset",
                )?,
                "extensible array super block address offset",
            )
        })?;
        Self::checked_u64_add(
            index_block_addr,
            Self::u64_from_usize(offset, "extensible array super block address offset")?,
            "extensible array super block address",
        )
    }

    pub(super) fn locate_extensible_array_element<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        header_addr: u64,
        header: &MutableExtensibleArrayHeader,
        element_index: usize,
    ) -> Result<u64> {
        if crate::io::reader::is_undef_addr(header.index_block_addr) {
            return Err(Error::Unsupported(
                "cannot update extensible-array chunk entry without an index block".into(),
            ));
        }
        Self::verify_extensible_array_index_block(reader, header_addr, header)?;
        let direct_count = usize::from(header.index_block_elements);
        let index_prefix_size = Self::checked_usize_add(
            4 + 1 + 1,
            usize::from(reader.sizeof_addr()),
            "extensible array index block offset",
        )?;
        if element_index < direct_count {
            let element_offset = Self::checked_usize_add(
                index_prefix_size,
                Self::checked_usize_mul(
                    element_index,
                    header.raw_element_size,
                    "extensible array index block element offset",
                )?,
                "extensible array index block element offset",
            )?;
            return Self::checked_u64_add(
                header.index_block_addr,
                Self::u64_from_usize(
                    element_offset,
                    "extensible array index block element offset",
                )?,
                "extensible array index block element address",
            );
        }

        let data_block_index = element_index - direct_count;
        if data_block_index < header.data_block_min_elements {
            let data_block_addr_offset = Self::checked_usize_add(
                index_prefix_size,
                Self::checked_usize_mul(
                    direct_count,
                    header.raw_element_size,
                    "extensible array data block address offset",
                )?,
                "extensible array data block address offset",
            )?;
            let data_block_addr_pos = Self::checked_u64_add(
                header.index_block_addr,
                Self::u64_from_usize(
                    data_block_addr_offset,
                    "extensible array data block address offset",
                )?,
                "extensible array data block address",
            )?;
            reader.seek(data_block_addr_pos)?;
            let data_block_addr = reader.read_addr()?;
            if crate::io::reader::is_undef_addr(data_block_addr) {
                return Err(Error::Unsupported(
                    "cannot update unallocated extensible-array data block".into(),
                ));
            }
            return Self::locate_extensible_array_data_block_element(
                reader,
                header_addr,
                header,
                data_block_addr,
                Self::u64_from_usize(direct_count, "extensible array direct element count")?,
                data_block_index,
            );
        }

        Err(Error::Unsupported(
            "write_chunk cannot update extensible-array super-block entries yet".into(),
        ))
    }

    fn locate_extensible_array_data_block_element<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        header_addr: u64,
        header: &MutableExtensibleArrayHeader,
        data_block_addr: u64,
        block_offset: u64,
        element_index: usize,
    ) -> Result<u64> {
        reader.seek(data_block_addr)?;
        if reader.read_bytes(4)? != b"EADB" {
            return Err(Error::InvalidFormat(
                "invalid extensible array data block magic".into(),
            ));
        }
        let version = reader.read_u8()?;
        let class_id = reader.read_u8()?;
        let owner = reader.read_addr()?;
        let stored_offset = reader.read_uint(header.array_offset_size)?;
        if version != 0
            || class_id != header.class_id
            || owner != header_addr
            || stored_offset != block_offset
        {
            return Err(Error::InvalidFormat(
                "extensible array data block header does not match index".into(),
            ));
        }
        if header.data_block_min_elements > header.max_data_block_page_elements {
            return Err(Error::Unsupported(
                "write_chunk cannot update paged extensible-array data blocks yet".into(),
            ));
        }
        let prefix_size = Self::checked_usize_add(
            4 + 1 + 1,
            usize::from(reader.sizeof_addr()),
            "extensible array data block element offset",
        )
        .and_then(|value| {
            Self::checked_usize_add(
                value,
                usize::from(header.array_offset_size),
                "extensible array data block element offset",
            )
        })?;
        let element_offset = Self::checked_usize_add(
            prefix_size,
            Self::checked_usize_mul(
                element_index,
                header.raw_element_size,
                "extensible array data block element offset",
            )?,
            "extensible array data block element offset",
        )?;
        Self::checked_u64_add(
            data_block_addr,
            Self::u64_from_usize(element_offset, "extensible array data block element offset")?,
            "extensible array data block element address",
        )
    }

    fn verify_extensible_array_index_block<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        header_addr: u64,
        header: &MutableExtensibleArrayHeader,
    ) -> Result<()> {
        reader.seek(header.index_block_addr)?;
        if reader.read_bytes(4)? != b"EAIB" {
            return Err(Error::InvalidFormat(
                "invalid extensible array index block magic".into(),
            ));
        }
        let version = reader.read_u8()?;
        let class_id = reader.read_u8()?;
        let owner = reader.read_addr()?;
        if version != 0 || class_id != header.class_id || owner != header_addr {
            return Err(Error::InvalidFormat(
                "extensible array index block header does not match array header".into(),
            ));
        }
        Ok(())
    }

    pub(super) fn plan_extensible_array_chunk_write(
        &self,
        info: &crate::hl::dataset::DatasetInfo,
        chunk_coords: &[u64],
        chunk_dims: &[u64],
        unfiltered_chunk_bytes: usize,
    ) -> Result<ExtensibleArrayChunkWrite> {
        let element_index =
            Self::linear_chunk_index(chunk_coords, &info.dataspace.dims, chunk_dims)?;
        let filtered = info
            .filter_pipeline
            .as_ref()
            .map(|pipeline| !pipeline.filters.is_empty())
            .unwrap_or(false);
        let chunk_size_len = if filtered {
            Self::filtered_chunk_size_len(
                info.layout.version,
                unfiltered_chunk_bytes,
                self.superblock.sizeof_size,
            )
        } else {
            0
        };
        Ok(ExtensibleArrayChunkWrite {
            element_index,
            filtered,
            chunk_size_len,
        })
    }

    pub(super) fn extensible_array_data_block_size(
        &self,
        header: &MutableExtensibleArrayHeader,
        data_block_elements: usize,
    ) -> Result<usize> {
        let pages = Self::extensible_array_data_block_pages(header, data_block_elements);
        let prefix_size = Self::checked_usize_add(
            4 + 1 + 1,
            usize::from(self.superblock.sizeof_addr),
            "extensible array data block size",
        )
        .and_then(|value| {
            Self::checked_usize_add(
                value,
                usize::from(header.array_offset_size),
                "extensible array data block size",
            )
        })
        .and_then(|value| Self::checked_usize_add(value, 4, "extensible array data block size"))?;
        if pages == 0 {
            let element_bytes = Self::checked_usize_mul(
                data_block_elements,
                header.raw_element_size,
                "extensible array data block size",
            )?;
            Self::checked_usize_add(
                prefix_size,
                element_bytes,
                "extensible array data block size",
            )
        } else {
            let page_payload = Self::checked_usize_mul(
                header.max_data_block_page_elements,
                header.raw_element_size,
                "extensible array data block size",
            )?;
            let page_size =
                Self::checked_usize_add(page_payload, 4, "extensible array data block size")?;
            let page_bytes =
                Self::checked_usize_mul(pages, page_size, "extensible array data block size")?;
            Self::checked_usize_add(prefix_size, page_bytes, "extensible array data block size")
        }
    }

    #[allow(clippy::too_many_arguments)]
    /// Pure encoder for the EADB prefix (+ inline elements when the block
    /// is not paginated). Mirrors the serialize half of libhdf5's
    /// `H5EA__cache_dblock_serialize`: no I/O, bytes out.
    fn encode_extensible_array_data_block_prefix(
        &self,
        header_addr: u64,
        header: &MutableExtensibleArrayHeader,
        block_offset: u64,
        data_block_elements: usize,
        initial: Option<(usize, u64, u64, bool, usize)>,
    ) -> Result<Vec<u8>> {
        let pages = Self::extensible_array_data_block_pages(header, data_block_elements);
        let prefix_size = Self::checked_usize_add(
            4 + 1 + 1,
            usize::from(self.superblock.sizeof_addr),
            "extensible array data block prefix size",
        )
        .and_then(|value| {
            Self::checked_usize_add(
                value,
                usize::from(header.array_offset_size),
                "extensible array data block prefix size",
            )
        })?;
        let inline_bytes = if pages == 0 {
            Self::checked_usize_mul(
                data_block_elements,
                header.raw_element_size,
                "extensible array data block prefix size",
            )?
        } else {
            0
        };
        let capacity = Self::checked_usize_add(
            prefix_size,
            inline_bytes,
            "extensible array data block prefix size",
        )?;
        let sa = usize::from(self.superblock.sizeof_addr);
        let array_offset_size = usize::from(header.array_offset_size);
        let mut prefix = Vec::with_capacity(capacity);
        prefix.extend_from_slice(b"EADB");
        prefix.push(0);
        prefix.push(header.class_id);
        prefix.extend_from_slice(&Self::encode_uint_le(
            header_addr,
            sa,
            "extensible array data block header address",
        )?);
        prefix.extend_from_slice(&Self::encode_uint_le(
            block_offset,
            array_offset_size,
            "extensible array data block offset",
        )?);

        if pages == 0 {
            let fill_addr = Self::undefined_addr_bytes(sa, "extensible array data block address")?;
            for idx in 0..data_block_elements {
                if let Some((initial_idx, chunk_addr, chunk_size, filtered, chunk_size_len)) =
                    initial
                {
                    if idx == initial_idx {
                        prefix.extend_from_slice(&Self::encode_uint_le(
                            chunk_addr,
                            sa,
                            "extensible array data block chunk address",
                        )?);
                        if filtered {
                            prefix.extend_from_slice(&Self::encode_uint_le(
                                chunk_size,
                                chunk_size_len,
                                "extensible array data block chunk size",
                            )?);
                            prefix.extend_from_slice(&0u32.to_le_bytes());
                        }
                        continue;
                    }
                }
                prefix.extend_from_slice(&fill_addr);
                if initial
                    .map(|(_, _, _, filtered, _)| filtered)
                    .unwrap_or(false)
                {
                    let chunk_size_len = initial.map(|(_, _, _, _, len)| len).unwrap_or(0);
                    prefix.extend_from_slice(&vec![0u8; chunk_size_len]);
                    prefix.extend_from_slice(&0u32.to_le_bytes());
                }
            }
        }
        let checksum = checksum_metadata(&prefix);
        prefix.extend_from_slice(&checksum.to_le_bytes());
        Ok(prefix)
    }

    /// Allocate + encode + write an extensible-array data block. Composes
    /// `encode_extensible_array_data_block_prefix` with file I/O. C-side
    /// analogue: `H5EA__dblock_create`.
    pub(super) fn create_extensible_array_data_block(
        &mut self,
        header_addr: u64,
        header: &MutableExtensibleArrayHeader,
        block_offset: u64,
        data_block_elements: usize,
        initial: Option<(usize, u64, u64, bool, usize)>,
        initialized_page: Option<usize>,
    ) -> Result<u64> {
        let data_block_size = self.extensible_array_data_block_size(header, data_block_elements)?;
        let data_block_addr = self.append_aligned_zeros(data_block_size, 8)?;

        let prefix = self.encode_extensible_array_data_block_prefix(
            header_addr,
            header,
            block_offset,
            data_block_elements,
            initial,
        )?;
        self.write_handle.seek(SeekFrom::Start(data_block_addr))?;
        self.write_handle.write_all(&prefix)?;

        let pages = Self::extensible_array_data_block_pages(header, data_block_elements);
        if pages > 0 {
            if let Some(page_index) = initialized_page {
                let Some((initial_idx, chunk_addr, chunk_size, filtered, chunk_size_len)) = initial
                else {
                    return Err(Error::InvalidFormat(
                        "initialized extensible-array page requires an initial element".into(),
                    ));
                };
                self.write_extensible_array_page(
                    data_block_addr,
                    header,
                    page_index,
                    Some((
                        initial_idx % header.max_data_block_page_elements,
                        chunk_addr,
                        chunk_size,
                        filtered,
                        chunk_size_len,
                    )),
                )?;
            }
        }

        Ok(data_block_addr)
    }

    /// Pure encoder for one extensible-array data-block page (element
    /// records + trailing checksum). Mirrors the per-page serialize path
    /// in libhdf5's `H5EA__cache_dblk_page_serialize`.
    fn encode_extensible_array_data_block_page(
        &self,
        header: &MutableExtensibleArrayHeader,
        initial: Option<(usize, u64, u64, bool, usize)>,
    ) -> Result<Vec<u8>> {
        let page_payload = Self::checked_usize_mul(
            header.max_data_block_page_elements,
            header.raw_element_size,
            "extensible array data block page size",
        )?;
        let page_size =
            Self::checked_usize_add(page_payload, 4, "extensible array data block page size")?;
        let mut page = Vec::with_capacity(page_size);
        let sa = usize::from(self.superblock.sizeof_addr);
        let fill_addr = Self::undefined_addr_bytes(sa, "extensible array data block page address")?;
        for idx in 0..header.max_data_block_page_elements {
            if let Some((initial_idx, chunk_addr, chunk_size, filtered, chunk_size_len)) = initial {
                if idx == initial_idx {
                    page.extend_from_slice(&Self::encode_uint_le(
                        chunk_addr,
                        sa,
                        "extensible array data block page chunk address",
                    )?);
                    if filtered {
                        page.extend_from_slice(&Self::encode_uint_le(
                            chunk_size,
                            chunk_size_len,
                            "extensible array data block page chunk size",
                        )?);
                        page.extend_from_slice(&0u32.to_le_bytes());
                    }
                    continue;
                }
            }
            page.extend_from_slice(&fill_addr);
            if initial
                .map(|(_, _, _, filtered, _)| filtered)
                .unwrap_or(false)
            {
                let chunk_size_len = initial.map(|(_, _, _, _, len)| len).unwrap_or(0);
                page.extend_from_slice(&vec![0u8; chunk_size_len]);
                page.extend_from_slice(&0u32.to_le_bytes());
            }
        }
        let checksum = checksum_metadata(&page);
        page.extend_from_slice(&checksum.to_le_bytes());
        Ok(page)
    }

    /// Encode + write one extensible-array data-block page.
    fn write_extensible_array_page(
        &mut self,
        data_block_addr: u64,
        header: &MutableExtensibleArrayHeader,
        page_index: usize,
        initial: Option<(usize, u64, u64, bool, usize)>,
    ) -> Result<()> {
        let page_payload = Self::checked_usize_mul(
            header.max_data_block_page_elements,
            header.raw_element_size,
            "extensible array data block page size",
        )?;
        let page_size =
            Self::checked_usize_add(page_payload, 4, "extensible array data block page size")?;
        let prefix_size = Self::checked_usize_add(
            4 + 1 + 1,
            usize::from(self.superblock.sizeof_addr),
            "extensible array data block page address",
        )
        .and_then(|value| {
            Self::checked_usize_add(
                value,
                usize::from(header.array_offset_size),
                "extensible array data block page address",
            )
        })
        .and_then(|value| {
            Self::checked_usize_add(value, 4, "extensible array data block page address")
        })?;
        let page_offset = Self::checked_usize_add(
            prefix_size,
            Self::checked_usize_mul(
                page_index,
                page_size,
                "extensible array data block page address",
            )?,
            "extensible array data block page address",
        )?;
        let page_addr = Self::checked_u64_add(
            data_block_addr,
            Self::u64_from_usize(page_offset, "extensible array data block page address")?,
            "extensible array data block page address",
        )?;
        let page = self.encode_extensible_array_data_block_page(header, initial)?;
        self.write_handle.seek(SeekFrom::Start(page_addr))?;
        self.write_handle.write_all(&page)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn write_extensible_array_data_block_element(
        &mut self,
        data_block_addr: u64,
        header: &MutableExtensibleArrayHeader,
        data_block_elements: usize,
        element_in_block: usize,
        chunk_addr: u64,
        chunk_size: u64,
        filtered: bool,
        chunk_size_len: usize,
        page_index: Option<usize>,
    ) -> Result<()> {
        if let Some(page_index) = page_index {
            let page_payload = Self::checked_usize_mul(
                header.max_data_block_page_elements,
                header.raw_element_size,
                "extensible array data block page size",
            )?;
            let page_size =
                Self::checked_usize_add(page_payload, 4, "extensible array data block page size")?;
            let page_prefix_size = Self::checked_usize_add(
                4 + 1 + 1,
                usize::from(self.superblock.sizeof_addr),
                "extensible array data block page address",
            )
            .and_then(|value| {
                Self::checked_usize_add(
                    value,
                    usize::from(header.array_offset_size),
                    "extensible array data block page address",
                )
            })
            .and_then(|value| {
                Self::checked_usize_add(value, 4, "extensible array data block page address")
            })?;
            let page_offset = Self::checked_usize_add(
                page_prefix_size,
                Self::checked_usize_mul(
                    page_index,
                    page_size,
                    "extensible array data block page address",
                )?,
                "extensible array data block page address",
            )?;
            let page_addr = Self::checked_u64_add(
                data_block_addr,
                Self::u64_from_usize(page_offset, "extensible array data block page address")?,
                "extensible array data block page address",
            )?;
            let local_index = element_in_block % header.max_data_block_page_elements;
            let element_pos = Self::checked_u64_add(
                page_addr,
                Self::u64_from_usize(
                    Self::checked_usize_mul(
                        local_index,
                        header.raw_element_size,
                        "extensible array data block page element offset",
                    )?,
                    "extensible array data block page element offset",
                )?,
                "extensible array data block page element address",
            )?;
            self.write_extensible_array_element(
                element_pos,
                chunk_addr,
                chunk_size,
                filtered,
                chunk_size_len,
            )?;
            self.rewrite_extensible_array_page_checksum(page_addr, header)?;
            return Ok(());
        }

        if data_block_elements > header.max_data_block_page_elements {
            return Err(Error::Unsupported(
                "write_chunk cannot update an unpaged view of a paged extensible-array data block"
                    .into(),
            ));
        }
        let prefix_size = Self::checked_usize_add(
            4 + 1 + 1,
            usize::from(self.superblock.sizeof_addr),
            "extensible array data block element offset",
        )
        .and_then(|value| {
            Self::checked_usize_add(
                value,
                usize::from(header.array_offset_size),
                "extensible array data block element offset",
            )
        })?;
        let element_pos = Self::checked_u64_add(
            data_block_addr,
            Self::u64_from_usize(
                Self::checked_usize_add(
                    prefix_size,
                    Self::checked_usize_mul(
                        element_in_block,
                        header.raw_element_size,
                        "extensible array data block element offset",
                    )?,
                    "extensible array data block element offset",
                )?,
                "extensible array data block element offset",
            )?,
            "extensible array data block element address",
        )?;
        self.write_extensible_array_element(
            element_pos,
            chunk_addr,
            chunk_size,
            filtered,
            chunk_size_len,
        )?;
        self.rewrite_extensible_array_data_block_checksum(
            data_block_addr,
            header,
            data_block_elements,
        )
    }

    fn rewrite_extensible_array_data_block_checksum(
        &mut self,
        data_block_addr: u64,
        header: &MutableExtensibleArrayHeader,
        data_block_elements: usize,
    ) -> Result<()> {
        let prefix_size = Self::checked_usize_add(
            4 + 1 + 1,
            usize::from(self.superblock.sizeof_addr),
            "extensible array data block checksum span",
        )
        .and_then(|value| {
            Self::checked_usize_add(
                value,
                usize::from(header.array_offset_size),
                "extensible array data block checksum span",
            )
        })?;
        let check_len = Self::checked_usize_add(
            prefix_size,
            Self::checked_usize_mul(
                data_block_elements,
                header.raw_element_size,
                "extensible array data block checksum span",
            )?,
            "extensible array data block checksum span",
        )?;
        self.write_handle.flush()?;
        let mut bytes = self.read_fresh_bytes(data_block_addr, check_len)?;
        let checksum = checksum_metadata(&bytes);
        let checksum_addr = Self::checked_u64_add(
            data_block_addr,
            Self::u64_from_usize(check_len, "extensible array data block checksum span")?,
            "extensible array data block checksum address",
        )?;
        self.write_handle.seek(SeekFrom::Start(checksum_addr))?;
        self.write_handle.write_all(&checksum.to_le_bytes())?;
        bytes.clear();
        Ok(())
    }

    fn rewrite_extensible_array_page_checksum(
        &mut self,
        page_addr: u64,
        header: &MutableExtensibleArrayHeader,
    ) -> Result<()> {
        let check_len = Self::checked_usize_mul(
            header.max_data_block_page_elements,
            header.raw_element_size,
            "extensible array data block page checksum span",
        )?;
        self.write_handle.flush()?;
        let bytes = self.read_fresh_bytes(page_addr, check_len)?;
        let checksum = checksum_metadata(&bytes);
        let checksum_addr = Self::checked_u64_add(
            page_addr,
            Self::u64_from_usize(check_len, "extensible array data block page checksum span")?,
            "extensible array data block page checksum address",
        )?;
        self.write_handle.seek(SeekFrom::Start(checksum_addr))?;
        self.write_handle.write_all(&checksum.to_le_bytes())?;
        Ok(())
    }

    pub(super) fn write_extensible_array_element(
        &mut self,
        element_pos: u64,
        chunk_addr: u64,
        chunk_size: u64,
        filtered: bool,
        chunk_size_len: usize,
    ) -> Result<()> {
        self.write_handle.seek(SeekFrom::Start(element_pos))?;
        let chunk_addr = Self::encode_uint_le(
            chunk_addr,
            usize::from(self.superblock.sizeof_addr),
            "extensible array chunk address",
        )?;
        self.write_handle.write_all(&chunk_addr)?;
        if filtered {
            self.write_uint_le(chunk_size, chunk_size_len)?;
            self.write_handle.write_all(&0u32.to_le_bytes())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::MutableFile;

    #[test]
    fn mutable_super_block_info_rejects_start_index_product_overflow() {
        let err =
            MutableFile::build_mutable_extensible_array_super_block_info(3, usize::MAX / 4 + 1)
                .unwrap_err();
        assert!(err.to_string().contains("start index overflow"));
    }

    #[test]
    fn mutable_checked_helpers_reject_overflow() {
        assert!(MutableFile::checked_usize_add(usize::MAX, 1, "ea add").is_err());
        assert!(MutableFile::checked_usize_mul(usize::MAX, 2, "ea mul").is_err());
        assert!(MutableFile::checked_u64_add(u64::MAX, 1, "ea addr").is_err());
    }

    #[test]
    fn mutable_index_address_helpers_reject_overflow() {
        assert!(
            MutableFile::extensible_array_index_data_block_addr_pos(0, usize::MAX, 8, 8, 0)
                .is_err()
        );
        assert!(MutableFile::extensible_array_index_super_block_addr_pos(
            0,
            4,
            8,
            usize::MAX,
            8,
            0
        )
        .is_err());
    }

    #[test]
    fn header_relative_offset_rejects_out_of_bounds_field() {
        let err = MutableFile::header_relative_offset(20, 10, 8, 12, "test ea header").unwrap_err();
        assert!(err.to_string().contains("field exceeds"));
    }

    #[test]
    fn patch_header_uint_rejects_invalid_integer_width() {
        let mut header = [0u8; 16];
        let err =
            MutableFile::patch_header_uint(&mut header, 0, 9, 1, "test ea header").unwrap_err();
        assert!(err.to_string().contains("integer width is invalid"));
    }
}
