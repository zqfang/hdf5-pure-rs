use crate::error::{Error, Result};

use super::virtual_dataset::{
    IrregularHyperslabBlock, RegularHyperslab, VirtualMapping, VirtualSelection,
};
use super::{read_le_u32_at, read_le_uint_at, read_u8_at, usize_from_u64, Dataset};

const MAX_VDS_MAPPINGS: usize = 65_536;
const MAX_VDS_SELECTION_RANK: usize = 32;

struct VirtualHyperslabHeader {
    flags: u8,
    enc_size: usize,
}

struct DecodedRegularHyperslabDim {
    start: u64,
    stride: u64,
    count: u64,
    block: u64,
}

struct DecodedIrregularHyperslabBlock {
    start: Vec<u64>,
    end: Vec<u64>,
}

struct DecodedVirtualSourceNames {
    file_name: String,
    dataset_name: String,
}

impl Dataset {
    pub(super) fn decode_virtual_mappings(
        heap_data: &[u8],
        sizeof_size: usize,
    ) -> Result<Vec<VirtualMapping>> {
        let mut pos = 0usize;
        let version = *heap_data
            .get(pos)
            .ok_or_else(|| Error::InvalidFormat("empty virtual dataset heap object".into()))?;
        pos += 1;
        if version > 1 {
            return Err(Error::Unsupported(format!(
                "virtual dataset heap encoding version {version}"
            )));
        }
        let count = usize_from_u64(
            read_le_uint_at(heap_data, &mut pos, sizeof_size)?,
            "virtual dataset mapping count",
        )?;
        if count > MAX_VDS_MAPPINGS {
            return Err(Error::InvalidFormat(format!(
                "virtual dataset mapping count {count} exceeds supported maximum {MAX_VDS_MAPPINGS}"
            )));
        }
        let mut mappings = Vec::with_capacity(count);
        let mut file_names: Vec<String> = Vec::with_capacity(count);
        let mut dataset_names: Vec<String> = Vec::with_capacity(count);

        for _ in 0..count {
            mappings.push(Self::decode_virtual_mapping(
                heap_data,
                &mut pos,
                version,
                sizeof_size,
                &mut file_names,
                &mut dataset_names,
            )?);
        }

        Ok(mappings)
    }

    fn decode_virtual_mapping(
        heap_data: &[u8],
        pos: &mut usize,
        version: u8,
        sizeof_size: usize,
        file_names: &mut Vec<String>,
        dataset_names: &mut Vec<String>,
    ) -> Result<VirtualMapping> {
        let flags = Self::decode_virtual_mapping_flags(heap_data, pos, version)?;
        let names = Self::decode_virtual_source_names(
            heap_data,
            pos,
            sizeof_size,
            flags,
            file_names,
            dataset_names,
        )?;
        let source_select = Self::decode_virtual_selection(heap_data, pos)?;
        let virtual_select = Self::decode_virtual_selection(heap_data, pos)?;
        trace_vds_source_resolve(&names.file_name, &names.dataset_name);
        file_names.push(names.file_name.clone());
        dataset_names.push(names.dataset_name.clone());
        Ok(VirtualMapping {
            file_name: names.file_name,
            dataset_name: names.dataset_name,
            source_select,
            virtual_select,
        })
    }

    fn decode_virtual_mapping_flags(data: &[u8], pos: &mut usize, version: u8) -> Result<u8> {
        const VDS_MAPPING_KNOWN_FLAGS: u8 = 0x07;

        if version == 0 {
            return Ok(0);
        }
        let flags = *data.get(*pos).ok_or_else(|| {
            Error::InvalidFormat("truncated virtual dataset mapping flags".into())
        })?;
        *pos += 1;
        if flags & !VDS_MAPPING_KNOWN_FLAGS != 0 {
            return Err(Error::InvalidFormat(format!(
                "virtual dataset mapping flags contain unknown bits 0x{flags:02x}"
            )));
        }
        if flags & 0x04 != 0 && flags & 0x01 != 0 {
            return Err(Error::InvalidFormat(
                "virtual dataset mapping cannot use both same-file and shared file-name flags"
                    .into(),
            ));
        }
        Ok(flags)
    }

    fn decode_virtual_source_names(
        heap_data: &[u8],
        pos: &mut usize,
        sizeof_size: usize,
        flags: u8,
        file_names: &[String],
        dataset_names: &[String],
    ) -> Result<DecodedVirtualSourceNames> {
        Ok(DecodedVirtualSourceNames {
            file_name: Self::decode_virtual_source_file_name(
                heap_data,
                pos,
                sizeof_size,
                flags,
                file_names,
            )?,
            dataset_name: Self::decode_virtual_source_dataset_name(
                heap_data,
                pos,
                sizeof_size,
                flags,
                dataset_names,
            )?,
        })
    }

    fn decode_virtual_source_file_name(
        heap_data: &[u8],
        pos: &mut usize,
        sizeof_size: usize,
        flags: u8,
        file_names: &[String],
    ) -> Result<String> {
        if flags & 0x04 != 0 {
            return Ok(".".to_string());
        }
        if flags & 0x01 != 0 {
            return Self::decode_virtual_shared_name_ref(
                heap_data,
                pos,
                sizeof_size,
                file_names,
                "virtual dataset shared file-name index",
                "invalid shared VDS source file reference",
            );
        }
        read_c_string(heap_data, pos)
    }

    fn decode_virtual_source_dataset_name(
        heap_data: &[u8],
        pos: &mut usize,
        sizeof_size: usize,
        flags: u8,
        dataset_names: &[String],
    ) -> Result<String> {
        if flags & 0x02 != 0 {
            return Self::decode_virtual_shared_name_ref(
                heap_data,
                pos,
                sizeof_size,
                dataset_names,
                "virtual dataset shared dataset-name index",
                "invalid shared VDS source dataset reference",
            );
        }
        read_c_string(heap_data, pos)
    }

    fn decode_virtual_shared_name_ref(
        heap_data: &[u8],
        pos: &mut usize,
        sizeof_size: usize,
        names: &[String],
        index_context: &'static str,
        invalid_context: &'static str,
    ) -> Result<String> {
        let origin = usize_from_u64(read_le_uint_at(heap_data, pos, sizeof_size)?, index_context)?;
        names
            .get(origin)
            .cloned()
            .ok_or_else(|| Error::InvalidFormat(invalid_context.into()))
    }

    pub(super) fn decode_virtual_selection(
        data: &[u8],
        pos: &mut usize,
    ) -> Result<VirtualSelection> {
        const H5S_SEL_POINTS: u32 = 1;
        const H5S_SEL_HYPERSLABS: u32 = 2;
        const H5S_SEL_ALL: u32 = 3;

        let start_pos = *pos;
        let sel_type = read_le_u32_at(data, pos)?;
        let selection = match sel_type {
            H5S_SEL_ALL => Self::decode_virtual_all_selection(data, pos)?,
            H5S_SEL_POINTS => Self::decode_virtual_point_selection(data, pos)?,
            H5S_SEL_HYPERSLABS => Self::decode_virtual_hyperslab_selection(data, pos)?,
            _ => {
                return Err(Error::Unsupported(format!(
                    "virtual dataset selection type {sel_type}"
                )))
            }
        };

        trace_selection_deserialize(&data[start_pos..*pos], sel_type);
        Ok(selection)
    }

    fn decode_virtual_all_selection(data: &[u8], pos: &mut usize) -> Result<VirtualSelection> {
        let version = read_le_u32_at(data, pos)?;
        if version != 1 {
            return Err(Error::Unsupported(format!(
                "virtual all-selection version {version}"
            )));
        }
        skip_reserved(data, pos, 8, "virtual all-selection reserved bytes")?;
        Ok(VirtualSelection::All)
    }

    fn decode_virtual_point_selection(data: &[u8], pos: &mut usize) -> Result<VirtualSelection> {
        let version = read_le_u32_at(data, pos)?;
        let enc_size = Self::decode_virtual_point_enc_size(data, pos, version)?;
        let rank = Self::decode_virtual_selection_rank(data, pos)?;
        let point_count = usize_from_u64(
            read_le_uint_at(data, pos, enc_size)?,
            "virtual point selection count",
        )?;
        let coordinate_count = point_count.checked_mul(rank).ok_or_else(|| {
            Error::InvalidFormat("virtual point selection coordinate count overflow".into())
        })?;
        let mut points = Vec::with_capacity(point_count);
        for _ in 0..point_count {
            let mut point = Vec::with_capacity(rank);
            for _ in 0..rank {
                point.push(read_le_uint_at(data, pos, enc_size)?);
            }
            points.push(point);
        }
        debug_assert_eq!(coordinate_count, points.iter().map(Vec::len).sum::<usize>());
        Ok(VirtualSelection::Points(points))
    }

    fn decode_virtual_point_enc_size(data: &[u8], pos: &mut usize, version: u32) -> Result<usize> {
        let enc_size = if version >= 2 {
            usize::from(read_u8_at(data, pos)?)
        } else if version == 1 {
            skip_reserved(data, pos, 8, "virtual point-selection reserved bytes")?;
            4
        } else {
            return Err(Error::Unsupported(format!(
                "virtual point selection version {version}"
            )));
        };
        validate_vds_selection_enc_size(enc_size, "point")?;
        Ok(enc_size)
    }

    fn decode_virtual_hyperslab_selection(
        data: &[u8],
        pos: &mut usize,
    ) -> Result<VirtualSelection> {
        const H5S_HYPER_REGULAR: u8 = 0x01;

        let version = read_le_u32_at(data, pos)?;
        let header = Self::decode_virtual_hyperslab_header(data, pos, version)?;
        let rank = Self::decode_virtual_selection_rank(data, pos)?;

        if header.flags & H5S_HYPER_REGULAR != 0 {
            return Self::decode_virtual_regular_hyperslab_selection(
                data,
                pos,
                rank,
                header.enc_size,
            );
        }
        Self::decode_virtual_irregular_hyperslab_selection(data, pos, rank, header.enc_size)
    }

    fn decode_virtual_hyperslab_header(
        data: &[u8],
        pos: &mut usize,
        version: u32,
    ) -> Result<VirtualHyperslabHeader> {
        const H5S_SELECT_FLAG_BITS: u8 = 0x01;

        let (flags, enc_size) = if version >= 3 {
            let flags = read_u8_at(data, pos)?;
            let enc_size = usize::from(read_u8_at(data, pos)?);
            (flags, enc_size)
        } else if version == 2 {
            let flags = read_u8_at(data, pos)?;
            skip_reserved(data, pos, 4, "virtual hyperslab selection reserved bytes")?;
            (flags, 8)
        } else if version == 1 {
            skip_reserved(data, pos, 8, "virtual hyperslab selection reserved bytes")?;
            (0, 4)
        } else {
            return Err(Error::Unsupported(format!(
                "virtual hyperslab selection version {version}"
            )));
        };

        if flags & !H5S_SELECT_FLAG_BITS != 0 {
            return Err(Error::InvalidFormat(format!(
                "virtual hyperslab selection has unknown flags 0x{flags:02x}"
            )));
        }
        validate_vds_selection_enc_size(enc_size, "hyperslab")?;
        Ok(VirtualHyperslabHeader { flags, enc_size })
    }

    fn decode_virtual_selection_rank(data: &[u8], pos: &mut usize) -> Result<usize> {
        let rank = usize_from_u64(
            u64::from(read_le_u32_at(data, pos)?),
            "virtual selection rank",
        )?;
        if rank == 0 || rank > MAX_VDS_SELECTION_RANK {
            return Err(Error::InvalidFormat(format!(
                "virtual selection rank {rank} exceeds supported maximum {MAX_VDS_SELECTION_RANK}"
            )));
        }
        Ok(rank)
    }

    fn decode_virtual_regular_hyperslab_selection(
        data: &[u8],
        pos: &mut usize,
        rank: usize,
        enc_size: usize,
    ) -> Result<VirtualSelection> {
        let mut start = Vec::with_capacity(rank);
        let mut stride = Vec::with_capacity(rank);
        let mut count = Vec::with_capacity(rank);
        let mut block = Vec::with_capacity(rank);

        for _ in 0..rank {
            let dim = Self::decode_virtual_regular_hyperslab_dim(data, pos, enc_size)?;
            start.push(dim.start);
            stride.push(dim.stride);
            count.push(dim.count);
            block.push(dim.block);
        }

        Ok(VirtualSelection::Regular(RegularHyperslab {
            start,
            stride,
            count,
            block,
        }))
    }

    fn decode_virtual_irregular_hyperslab_selection(
        data: &[u8],
        pos: &mut usize,
        rank: usize,
        enc_size: usize,
    ) -> Result<VirtualSelection> {
        let block_count = usize_from_u64(
            read_le_uint_at(data, pos, enc_size)?,
            "virtual hyperslab block count",
        )?;
        let mut blocks = Vec::with_capacity(block_count);
        for _ in 0..block_count {
            let block = Self::decode_virtual_irregular_hyperslab_block(data, pos, rank, enc_size)?;
            blocks.push(Self::materialize_virtual_irregular_hyperslab_block(block)?);
        }
        Ok(VirtualSelection::Irregular(blocks))
    }

    fn decode_virtual_hyperslab_vector(
        data: &[u8],
        pos: &mut usize,
        rank: usize,
        enc_size: usize,
        decode_extent: bool,
    ) -> Result<Vec<u64>> {
        let mut values = Vec::with_capacity(rank);
        for _ in 0..rank {
            let value = read_le_uint_at(data, pos, enc_size)?;
            values.push(if decode_extent {
                decode_hyperslab_extent(value, enc_size)
            } else {
                value
            });
        }
        Ok(values)
    }

    fn decode_virtual_regular_hyperslab_dim(
        data: &[u8],
        pos: &mut usize,
        enc_size: usize,
    ) -> Result<DecodedRegularHyperslabDim> {
        Ok(DecodedRegularHyperslabDim {
            start: read_le_uint_at(data, pos, enc_size)?,
            stride: read_le_uint_at(data, pos, enc_size)?,
            count: decode_hyperslab_extent(read_le_uint_at(data, pos, enc_size)?, enc_size),
            block: decode_hyperslab_extent(read_le_uint_at(data, pos, enc_size)?, enc_size),
        })
    }

    fn decode_virtual_irregular_hyperslab_block(
        data: &[u8],
        pos: &mut usize,
        rank: usize,
        enc_size: usize,
    ) -> Result<DecodedIrregularHyperslabBlock> {
        Ok(DecodedIrregularHyperslabBlock {
            start: Self::decode_virtual_hyperslab_vector(data, pos, rank, enc_size, false)?,
            end: Self::decode_virtual_hyperslab_vector(data, pos, rank, enc_size, false)?,
        })
    }

    fn materialize_virtual_irregular_hyperslab_block(
        block: DecodedIrregularHyperslabBlock,
    ) -> Result<IrregularHyperslabBlock> {
        let mut extents = Vec::with_capacity(block.start.len());
        for (start_coord, end_coord) in block.start.iter().zip(&block.end) {
            if end_coord < start_coord {
                return Err(Error::InvalidFormat(
                    "virtual irregular hyperslab end precedes start".into(),
                ));
            }
            extents.push(end_coord - start_coord + 1);
        }

        Ok(IrregularHyperslabBlock {
            start: block.start,
            block: extents,
        })
    }
}

#[cfg(feature = "tracehash")]
fn trace_selection_deserialize(data: &[u8], sel_type: u32) {
    let mut th = tracehash::th_call!("hdf5.selection.deserialize");
    th.input_bytes(data);
    th.output_value(&(true));
    th.output_u64(u64::from(sel_type));
    th.finish();
}

#[cfg(not(feature = "tracehash"))]
fn trace_selection_deserialize(_data: &[u8], _sel_type: u32) {}

#[cfg(feature = "tracehash")]
fn trace_vds_source_resolve(file_name: &str, dataset_name: &str) {
    let mut th = tracehash::th_call!("hdf5.vds.source.resolve");
    th.input_bytes(file_name.as_bytes());
    th.input_bytes(dataset_name.as_bytes());
    th.output_value(&(file_name == "."));
    th.output_value(file_name.as_bytes());
    th.output_value(dataset_name.as_bytes());
    th.finish();
}

#[cfg(not(feature = "tracehash"))]
fn trace_vds_source_resolve(_file_name: &str, _dataset_name: &str) {}

fn read_c_string(bytes: &[u8], pos: &mut usize) -> Result<String> {
    let tail = bytes
        .get(*pos..)
        .ok_or_else(|| Error::InvalidFormat("truncated string field".into()))?;
    let rel_end = tail
        .iter()
        .position(|&byte| byte == 0)
        .ok_or_else(|| Error::InvalidFormat("unterminated string field".into()))?;
    let end = pos
        .checked_add(rel_end)
        .ok_or_else(|| Error::InvalidFormat("string field offset overflow".into()))?;
    let value = std::str::from_utf8(&bytes[*pos..end])
        .map_err(|err| Error::InvalidFormat(format!("invalid UTF-8 string field: {err}")))?
        .to_string();
    *pos = end
        .checked_add(1)
        .ok_or_else(|| Error::InvalidFormat("string field offset overflow".into()))?;
    Ok(value)
}

fn decode_hyperslab_extent(value: u64, enc_size: usize) -> u64 {
    match enc_size {
        2 if value == u64::from(u16::MAX) => u64::MAX,
        4 if value == u64::from(u32::MAX) => u64::MAX,
        8 if value == u64::MAX => u64::MAX,
        _ => value,
    }
}

fn validate_vds_selection_enc_size(enc_size: usize, kind: &str) -> Result<()> {
    match enc_size {
        2 | 4 | 8 => Ok(()),
        _ => Err(Error::InvalidFormat(format!(
            "virtual {kind} selection uses unsupported encoded integer size {enc_size}"
        ))),
    }
}

fn skip_reserved(data: &[u8], pos: &mut usize, len: usize, context: &str) -> Result<()> {
    let end = pos
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))?;
    if end > data.len() {
        return Err(Error::InvalidFormat(format!("{context} are truncated")));
    }
    *pos = end;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c_string_rejects_start_past_buffer() {
        let mut pos = 2;
        let err = read_c_string(b"a", &mut pos).unwrap_err();
        assert!(
            err.to_string().contains("truncated string field"),
            "unexpected error: {err}"
        );
        assert_eq!(pos, 2);
    }

    #[test]
    fn c_string_rejects_unterminated_buffer() {
        let mut pos = 0;
        let err = read_c_string(b"abc", &mut pos).unwrap_err();
        assert!(
            err.to_string().contains("unterminated string field"),
            "unexpected error: {err}"
        );
        assert_eq!(pos, 0);
    }
}
