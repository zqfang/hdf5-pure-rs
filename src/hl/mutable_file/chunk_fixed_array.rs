use std::io::{Read, Seek, SeekFrom, Write};

use crate::error::{Error, Result};
use crate::format::checksum::checksum_metadata;
use crate::format::fixed_array::FixedArrayElement;
use crate::format::object_header;
use crate::io::reader::UNDEF_ADDR;

use super::MutableFile;

impl MutableFile {
    pub(super) fn rewrite_fixed_array_chunk(
        &mut self,
        index_addr: u64,
        dataset_addr: u64,
        info: &crate::hl::dataset::DatasetInfo,
        chunk_coords: &[u64],
        chunk_dims: &[u64],
        chunk_size: u64,
        chunk_addr: u64,
        unfiltered_chunk_bytes: usize,
    ) -> Result<()> {
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

        let mut guard = self.inner.lock();
        if info.dataspace.dims.len() == 1 {
            let mut elements = Vec::new();
            crate::format::fixed_array::read_fixed_array_chunks_into(
                &mut guard.reader,
                index_addr,
                filtered,
                chunk_size_len,
                &mut elements,
            )?;
            if element_index >= elements.len() {
                drop(guard);
                return self.rebuild_grown_1d_fixed_array_chunk_index(
                    index_addr,
                    info,
                    chunk_coords,
                    chunk_dims,
                    chunk_size,
                    chunk_addr,
                    filtered,
                    chunk_size_len,
                    element_index,
                    dataset_addr,
                    elements,
                );
            }
        }
        let element_location =
            crate::format::fixed_array::locate_fixed_array_element_with_checksum(
                &mut guard.reader,
                index_addr,
                filtered,
                chunk_size_len,
                element_index,
            )?;
        drop(guard);

        let sa = usize::from(self.superblock.sizeof_addr);
        self.write_handle
            .seek(SeekFrom::Start(element_location.element_addr))?;
        let mut encoded_addr = [0u8; 8];
        let chunk_addr = Self::encode_uint_le_into(
            chunk_addr,
            &mut encoded_addr,
            sa,
            "fixed array chunk address",
        )?;
        self.write_handle.write_all(&chunk_addr)?;
        if filtered {
            self.write_uint_le(chunk_size, chunk_size_len)?;
            self.write_handle.write_all(&0u32.to_le_bytes())?;
        }
        self.rewrite_fixed_array_element_checksum(
            element_location.checksum_start,
            element_location.checksum_len,
            element_location.checksum_addr,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn rebuild_grown_1d_fixed_array_chunk_index(
        &mut self,
        old_index_addr: u64,
        info: &crate::hl::dataset::DatasetInfo,
        chunk_coords: &[u64],
        chunk_dims: &[u64],
        chunk_size: u64,
        chunk_addr: u64,
        filtered: bool,
        chunk_size_len: usize,
        element_index: usize,
        dataset_addr: u64,
        mut elements: Vec<FixedArrayElement>,
    ) -> Result<()> {
        if chunk_coords.len() != 1 || chunk_dims.len() != 1 || info.dataspace.dims.len() != 1 {
            return Err(Error::Unsupported(
                "fixed-array growth after resize is currently limited to 1D chunk grids".into(),
            ));
        }
        let new_element_count =
            Self::fixed_array_1d_chunk_count(info.dataspace.dims[0], chunk_dims[0])?;
        if element_index >= new_element_count {
            return Err(Error::InvalidFormat(
                "fixed-array chunk coordinate is outside resized extent".into(),
            ));
        }
        if new_element_count <= elements.len() {
            return Err(Error::InvalidFormat(
                "fixed-array rebuild called without growth".into(),
            ));
        }
        elements.resize(
            new_element_count,
            FixedArrayElement {
                addr: UNDEF_ADDR,
                nbytes: if filtered { Some(0) } else { None },
                filter_mask: 0,
            },
        );
        elements[element_index] = FixedArrayElement {
            addr: chunk_addr,
            nbytes: if filtered { Some(chunk_size) } else { None },
            filter_mask: 0,
        };

        let page_bits = self.fixed_array_page_bits_from_layout(dataset_addr)?;
        let new_index_addr =
            self.append_fixed_array_chunk_index(&elements, page_bits, filtered, chunk_size_len)?;
        self.rewrite_fixed_array_layout_index_addr(dataset_addr, old_index_addr, new_index_addr)?;
        Ok(())
    }

    fn fixed_array_1d_chunk_count(dim: u64, chunk: u64) -> Result<usize> {
        if chunk == 0 {
            return Err(Error::InvalidFormat("zero chunk dimension".into()));
        }
        let count = dim
            .checked_add(chunk - 1)
            .ok_or_else(|| Error::InvalidFormat("fixed-array chunk count overflow".into()))?
            / chunk;
        Self::u64_to_usize(count, "fixed-array chunk count")
    }

    fn fixed_array_page_bits_from_layout(&self, dataset_addr: u64) -> Result<u8> {
        let location = self.find_message_in_oh(dataset_addr, object_header::MSG_LAYOUT)?;
        let mut layout = vec![0; location.msg_data_len];
        let layout_pos = self.physical_addr(location.msg_data_offset, "data layout message")?;
        self.read_fresh_bytes_into(layout_pos, &mut layout)?;
        let page_bits_offset = Self::fixed_array_layout_page_bits_offset(&layout)?;
        Ok(layout[page_bits_offset])
    }

    fn fixed_array_layout_page_bits_offset(layout: &[u8]) -> Result<usize> {
        if layout.len() < 7 || layout[0] != 4 || layout[1] != 2 {
            return Err(Error::InvalidFormat(
                "expected v4 chunked fixed-array layout message".into(),
            ));
        }
        let ndims = usize::from(layout[3]);
        let dim_width = usize::from(layout[4]);
        if ndims == 0 || dim_width == 0 || dim_width > 8 {
            return Err(Error::InvalidFormat(
                "invalid fixed-array layout dimension encoding".into(),
            ));
        }
        let index_type_offset = 5usize
            .checked_add(ndims.checked_mul(dim_width).ok_or_else(|| {
                Error::InvalidFormat("fixed-array layout dimension span overflow".into())
            })?)
            .ok_or_else(|| Error::InvalidFormat("fixed-array layout offset overflow".into()))?;
        if layout.get(index_type_offset).copied() != Some(3) {
            return Err(Error::InvalidFormat(
                "layout message is not a fixed-array chunk index".into(),
            ));
        }
        let page_bits_offset = index_type_offset
            .checked_add(1)
            .ok_or_else(|| Error::InvalidFormat("fixed-array layout offset overflow".into()))?;
        if page_bits_offset >= layout.len() {
            return Err(Error::InvalidFormat(
                "fixed-array layout message is truncated".into(),
            ));
        }
        Ok(page_bits_offset)
    }

    fn rewrite_fixed_array_layout_index_addr(
        &mut self,
        dataset_addr: u64,
        old_index_addr: u64,
        new_index_addr: u64,
    ) -> Result<()> {
        let location = self.find_message_in_oh(dataset_addr, object_header::MSG_LAYOUT)?;
        let mut layout = vec![0; location.msg_data_len];
        let layout_pos = self.physical_addr(location.msg_data_offset, "data layout message")?;
        self.read_fresh_bytes_into(layout_pos, &mut layout)?;
        let page_bits_offset = Self::fixed_array_layout_page_bits_offset(&layout)?;
        let addr_offset = page_bits_offset.checked_add(1).ok_or_else(|| {
            Error::InvalidFormat("fixed-array layout address offset overflow".into())
        })?;
        let sa = usize::from(self.superblock.sizeof_addr);
        let addr_end = addr_offset.checked_add(sa).ok_or_else(|| {
            Error::InvalidFormat("fixed-array layout address offset overflow".into())
        })?;
        if addr_end > layout.len() {
            return Err(Error::InvalidFormat(
                "fixed-array layout address field is truncated".into(),
            ));
        }
        let stored_old = Self::decode_uint_le(
            &layout[addr_offset..addr_end],
            "fixed-array layout old address",
        )?;
        if stored_old != old_index_addr {
            return Err(Error::InvalidFormat(
                "fixed-array layout address does not match current chunk index".into(),
            ));
        }

        let mut encoded = [0u8; 8];
        let encoded = Self::encode_uint_le_into(
            new_index_addr,
            &mut encoded,
            sa,
            "fixed-array layout address",
        )?;
        self.write_handle.seek(SeekFrom::Start(
            layout_pos
                .checked_add(Self::usize_to_u64(
                    addr_offset,
                    "fixed-array layout address offset",
                )?)
                .ok_or_else(|| {
                    Error::InvalidFormat("fixed-array layout address position overflow".into())
                })?,
        ))?;
        self.write_handle.write_all(encoded)?;
        if let Some((oh_start, oh_check_len)) = location.v2_checksum {
            self.rewrite_oh_checksum(oh_start, oh_check_len)?;
        }
        Ok(())
    }

    fn append_fixed_array_chunk_index(
        &mut self,
        elements: &[FixedArrayElement],
        page_bits: u8,
        filtered: bool,
        chunk_size_len: usize,
    ) -> Result<u64> {
        if page_bits == 0 {
            return Err(Error::InvalidFormat(
                "fixed-array chunk page bits must be positive".into(),
            ));
        }
        let sa = usize::from(self.superblock.sizeof_addr);
        let ss = usize::from(self.superblock.sizeof_size);
        let page_elements = 1usize.checked_shl(u32::from(page_bits)).ok_or_else(|| {
            Error::InvalidFormat("fixed-array page element count overflow".into())
        })?;
        let raw_element_size = if filtered {
            sa.checked_add(chunk_size_len)
                .and_then(|value| value.checked_add(4))
                .ok_or_else(|| {
                    Error::InvalidFormat("fixed-array raw element size overflow".into())
                })?
        } else {
            sa
        };
        let class_id = if filtered { 1 } else { 0 };
        let header_len = 4usize
            .checked_add(1)
            .and_then(|value| value.checked_add(1))
            .and_then(|value| value.checked_add(1))
            .and_then(|value| value.checked_add(1))
            .and_then(|value| value.checked_add(ss))
            .and_then(|value| value.checked_add(sa))
            .and_then(|value| value.checked_add(4))
            .ok_or_else(|| Error::InvalidFormat("fixed-array header size overflow".into()))?;
        let data_block = self.encode_fixed_array_data_block(
            elements,
            page_bits,
            page_elements,
            raw_element_size,
            class_id,
            filtered,
            chunk_size_len,
            0,
        )?;
        let header_physical = self.append_aligned_zeros(header_len, 8)?;
        let header_addr =
            self.logical_addr_from_physical(header_physical, "fixed-array header address")?;
        let data_physical = self.append_aligned_zeros(data_block.len(), 8)?;
        let data_addr =
            self.logical_addr_from_physical(data_physical, "fixed-array data block address")?;

        let data_block = self.encode_fixed_array_data_block(
            elements,
            page_bits,
            page_elements,
            raw_element_size,
            class_id,
            filtered,
            chunk_size_len,
            header_addr,
        )?;
        self.write_handle.seek(SeekFrom::Start(data_physical))?;
        self.write_handle.write_all(&data_block)?;

        let mut header = Vec::with_capacity(header_len);
        header.extend_from_slice(b"FAHD");
        header.push(0);
        header.push(class_id);
        header.push(
            u8::try_from(raw_element_size).map_err(|_| {
                Error::InvalidFormat("fixed-array raw element size exceeds u8".into())
            })?,
        );
        header.push(page_bits);
        self.append_uint_le_to_vec(
            &mut header,
            Self::usize_to_u64(elements.len(), "fixed-array element count")?,
            ss,
            "fixed-array element count",
        )?;
        self.append_uint_le_to_vec(&mut header, data_addr, sa, "fixed-array data block address")?;
        let checksum = checksum_metadata(&header);
        header.extend_from_slice(&checksum.to_le_bytes());
        self.write_handle.seek(SeekFrom::Start(header_physical))?;
        self.write_handle.write_all(&header)?;
        Ok(header_addr)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_fixed_array_data_block(
        &self,
        elements: &[FixedArrayElement],
        page_bits: u8,
        page_elements: usize,
        raw_element_size: usize,
        class_id: u8,
        filtered: bool,
        chunk_size_len: usize,
        header_addr: u64,
    ) -> Result<Vec<u8>> {
        let sa = usize::from(self.superblock.sizeof_addr);
        let prefix_payload_len = 4usize
            .checked_add(1)
            .and_then(|value| value.checked_add(1))
            .and_then(|value| value.checked_add(sa))
            .ok_or_else(|| Error::InvalidFormat("fixed-array data block prefix overflow".into()))?;
        let paginated = elements.len() > page_elements;
        let mut data = Vec::new();
        data.extend_from_slice(b"FADB");
        data.push(0);
        data.push(class_id);
        self.append_uint_le_to_vec(
            &mut data,
            header_addr,
            sa,
            "fixed-array data block owner address",
        )?;
        if paginated {
            let page_count = elements.len().div_ceil(page_elements);
            let page_init_len = page_count.div_ceil(8);
            let page_init_start = data.len();
            data.resize(page_init_start + page_init_len, 0);
            for page_index in 0..page_count {
                data[page_init_start + page_index / 8] |= 0x80 >> (page_index % 8);
            }
            let prefix_checksum = checksum_metadata(&data);
            data.extend_from_slice(&prefix_checksum.to_le_bytes());
            for page in elements.chunks(page_elements) {
                let page_start = data.len();
                for element in page {
                    self.append_fixed_array_element(&mut data, element, filtered, chunk_size_len)?;
                }
                let checksum = checksum_metadata(&data[page_start..]);
                data.extend_from_slice(&checksum.to_le_bytes());
            }
        } else {
            debug_assert_eq!(data.len(), prefix_payload_len);
            for element in elements {
                self.append_fixed_array_element(&mut data, element, filtered, chunk_size_len)?;
            }
            let checksum = checksum_metadata(&data);
            data.extend_from_slice(&checksum.to_le_bytes());
        }
        let expected_min = prefix_payload_len
            .checked_add(
                elements
                    .len()
                    .checked_mul(raw_element_size)
                    .ok_or_else(|| {
                        Error::InvalidFormat("fixed-array data block payload overflow".into())
                    })?,
            )
            .ok_or_else(|| Error::InvalidFormat("fixed-array data block size overflow".into()))?;
        if !paginated && data.len() != expected_min + 4 {
            return Err(Error::InvalidFormat(
                "fixed-array data block size accounting mismatch".into(),
            ));
        }
        let _ = page_bits;
        Ok(data)
    }

    fn append_fixed_array_element(
        &self,
        out: &mut Vec<u8>,
        element: &FixedArrayElement,
        filtered: bool,
        chunk_size_len: usize,
    ) -> Result<()> {
        let sa = usize::from(self.superblock.sizeof_addr);
        self.append_uint_le_to_vec(out, element.addr, sa, "fixed-array chunk address")?;
        if filtered {
            self.append_uint_le_to_vec(
                out,
                element.nbytes.unwrap_or(0),
                chunk_size_len,
                "fixed-array filtered chunk size",
            )?;
            out.extend_from_slice(&element.filter_mask.to_le_bytes());
        }
        Ok(())
    }

    fn append_uint_le_to_vec(
        &self,
        out: &mut Vec<u8>,
        value: u64,
        size: usize,
        context: &str,
    ) -> Result<()> {
        let mut bytes = [0u8; 8];
        let bytes = Self::encode_uint_le_into(value, &mut bytes, size, context)?;
        out.extend_from_slice(bytes);
        Ok(())
    }

    fn decode_uint_le(bytes: &[u8], context: &str) -> Result<u64> {
        if bytes.is_empty() || bytes.len() > 8 {
            return Err(Error::InvalidFormat(format!(
                "{context} integer width is invalid"
            )));
        }
        let mut buf = [0u8; 8];
        buf[..bytes.len()].copy_from_slice(bytes);
        Ok(u64::from_le_bytes(buf))
    }

    fn rewrite_fixed_array_element_checksum(
        &mut self,
        checksum_start: u64,
        checksum_len: usize,
        checksum_addr: u64,
    ) -> Result<()> {
        self.write_handle.flush()?;
        let mut reader = std::fs::File::open(&self.path)?;
        reader.seek(SeekFrom::Start(checksum_start))?;
        let mut bytes = vec![0; checksum_len];
        reader.read_exact(&mut bytes)?;
        let checksum = checksum_metadata(&bytes);
        let checksum_end = checksum_addr
            .checked_add(4)
            .ok_or_else(|| Error::InvalidFormat("fixed array checksum address overflow".into()))?;
        let file_len = reader.metadata()?.len();
        if checksum_end > file_len {
            return Err(Error::InvalidFormat(
                "fixed array checksum address exceeds file size".into(),
            ));
        }
        self.write_handle.seek(SeekFrom::Start(checksum_addr))?;
        self.write_handle.write_all(&checksum.to_le_bytes())?;
        Ok(())
    }
}
