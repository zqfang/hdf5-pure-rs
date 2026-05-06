//! Fractal heap huge-object access — mirrors libhdf5's `H5HFhuge.c` plus
//! `H5HFbtree2.c` (the v2 B-tree record decode for huge objects).

use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::io::reader::HdfReader;

use super::{heap_object_len, read_le_uint, FractalHeapHeader};

fn checked_huge_len(parts: &[usize], context: &str) -> Result<usize> {
    let mut len = 0usize;
    for part in parts {
        len = len.checked_add(*part).ok_or_else(|| {
            Error::InvalidFormat(format!("{context} length overflows address space"))
        })?;
    }
    Ok(len)
}

fn take_huge_field<'a>(
    bytes: &'a [u8],
    pos: &mut usize,
    len: usize,
    context: &str,
) -> Result<&'a [u8]> {
    let end = pos
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} field offset overflows")))?;
    let field = bytes.get(*pos..end).ok_or_else(|| {
        Error::InvalidFormat(format!(
            "{context} is truncated at byte range {}..{}",
            *pos, end
        ))
    })?;
    *pos = end;
    Ok(field)
}

#[derive(Debug, Clone, Copy)]
pub(super) struct HugeRecord {
    pub(super) addr: u64,
    pub(super) len: u64,
    pub(super) filtered: bool,
    pub(super) obj_size: Option<u64>,
    pub(super) id: Option<u64>,
}

impl FractalHeapHeader {
    pub(super) fn read_huge<R: Read + Seek>(
        &self,
        reader: &mut HdfReader<R>,
        heap_id: &[u8],
    ) -> Result<Vec<u8>> {
        let addr_size = usize::from(self.sizeof_addr);
        let len_size = usize::from(self.sizeof_size);

        let direct_id_len = checked_huge_len(&[1, addr_size, len_size], "direct huge heap ID")?;
        if self.io_filter_len == 0 && heap_id.len() >= direct_id_len {
            let mut p = 1usize;
            let addr = read_le_uint(take_huge_field(
                heap_id,
                &mut p,
                addr_size,
                "direct huge heap ID address",
            )?);
            let len = read_le_uint(take_huge_field(
                heap_id,
                &mut p,
                len_size,
                "direct huge heap ID length",
            )?);
            if crate::io::reader::is_undef_addr(addr) {
                return Err(Error::InvalidFormat(
                    "huge heap object has undefined address".into(),
                ));
            }
            reader.seek(addr)?;
            let data = reader.read_bytes(heap_object_len(len, "huge heap object length")?)?;
            self.trace_huge_object(heap_id, addr, len, len, 0, false);
            return Ok(data);
        }

        let filtered_id_len = checked_huge_len(
            &[1, addr_size, len_size, 4, len_size],
            "filtered huge heap ID",
        )?;
        if self.io_filter_len > 0 && heap_id.len() >= filtered_id_len {
            let mut p = 1usize;
            let addr = read_le_uint(take_huge_field(
                heap_id,
                &mut p,
                addr_size,
                "filtered huge heap ID address",
            )?);
            let len = read_le_uint(take_huge_field(
                heap_id,
                &mut p,
                len_size,
                "filtered huge heap ID length",
            )?);
            let filter_mask = read_u32_le(take_huge_field(
                heap_id,
                &mut p,
                4,
                "filtered huge heap ID filter mask",
            )?)?;
            let obj_size = read_le_uint(take_huge_field(
                heap_id,
                &mut p,
                len_size,
                "filtered huge heap ID object size",
            )?);
            if crate::io::reader::is_undef_addr(addr) {
                return Err(Error::InvalidFormat(
                    "huge heap object has undefined address".into(),
                ));
            }
            let pipeline = self.filter_pipeline.as_ref().ok_or_else(|| {
                Error::InvalidFormat("filtered huge object missing filter pipeline".into())
            })?;
            reader.seek(addr)?;
            let filtered =
                reader.read_bytes(heap_object_len(len, "filtered huge heap object length")?)?;
            let mut data = crate::filters::apply_pipeline_reverse(&filtered, pipeline, 1)?;
            data.truncate(heap_object_len(obj_size, "filtered huge heap object size")?);
            self.trace_huge_object(heap_id, addr, len, obj_size, filter_mask, true);
            return Ok(data);
        }

        if crate::io::reader::is_undef_addr(self.huge_btree_addr) {
            return Err(Error::InvalidFormat(
                "huge heap object ID is indirect but heap has no huge-object B-tree".into(),
            ));
        }

        let id = read_le_uint(&heap_id[1..]);
        let records = crate::format::btree_v2::collect_all_records(reader, self.huge_btree_addr)?;
        for record in records {
            let huge = self.decode_huge_record(&record)?;
            if huge.id == Some(id) {
                reader.seek(huge.addr)?;
                let mut data =
                    reader.read_bytes(heap_object_len(huge.len, "huge heap object length")?)?;
                if huge.filtered {
                    let pipeline = self.filter_pipeline.as_ref().ok_or_else(|| {
                        Error::InvalidFormat("filtered huge object missing filter pipeline".into())
                    })?;
                    data = crate::filters::apply_pipeline_reverse(&data, pipeline, 1)?;
                    let decoded_len = u64::try_from(data.len()).map_err(|_| {
                        Error::InvalidFormat("filtered huge heap object length overflow".into())
                    })?;
                    data.truncate(heap_object_len(
                        huge.obj_size.unwrap_or(decoded_len),
                        "filtered huge heap object size",
                    )?);
                }
                self.trace_huge_object(
                    heap_id,
                    huge.addr,
                    huge.len,
                    huge.obj_size.unwrap_or(huge.len),
                    0,
                    huge.filtered,
                );
                return Ok(data);
            }
        }

        Err(Error::InvalidFormat(format!(
            "huge fractal heap object id {id} not found"
        )))
    }

    pub(super) fn decode_huge_record(&self, record: &[u8]) -> Result<HugeRecord> {
        let sa = usize::from(self.sizeof_addr);
        let ss = usize::from(self.sizeof_size);

        let unfiltered_direct = checked_huge_len(&[sa, ss], "unfiltered direct huge record")?;
        if record.len() == unfiltered_direct {
            let mut p = 0usize;
            let addr = read_le_uint(take_huge_field(
                record,
                &mut p,
                sa,
                "unfiltered direct huge record address",
            )?);
            let len = read_le_uint(take_huge_field(
                record,
                &mut p,
                ss,
                "unfiltered direct huge record length",
            )?);
            return Ok(HugeRecord {
                addr,
                len,
                filtered: false,
                obj_size: None,
                id: None,
            });
        }
        let unfiltered_indirect =
            checked_huge_len(&[sa, ss, ss], "unfiltered indirect huge record")?;
        if record.len() == unfiltered_indirect {
            let mut p = 0usize;
            let addr = read_le_uint(take_huge_field(
                record,
                &mut p,
                sa,
                "unfiltered indirect huge record address",
            )?);
            let len = read_le_uint(take_huge_field(
                record,
                &mut p,
                ss,
                "unfiltered indirect huge record length",
            )?);
            let id = read_le_uint(take_huge_field(
                record,
                &mut p,
                ss,
                "unfiltered indirect huge record ID",
            )?);
            return Ok(HugeRecord {
                addr,
                len,
                filtered: false,
                obj_size: None,
                id: Some(id),
            });
        }
        let filtered_direct = checked_huge_len(&[sa, ss, 4, ss], "filtered direct huge record")?;
        if record.len() == filtered_direct {
            let mut p = 0usize;
            let addr = read_le_uint(take_huge_field(
                record,
                &mut p,
                sa,
                "filtered direct huge record address",
            )?);
            let len = read_le_uint(take_huge_field(
                record,
                &mut p,
                ss,
                "filtered direct huge record length",
            )?);
            let _filter_mask =
                take_huge_field(record, &mut p, 4, "filtered direct huge record filter mask")?;
            let obj_size = read_le_uint(take_huge_field(
                record,
                &mut p,
                ss,
                "filtered direct huge record object size",
            )?);
            return Ok(HugeRecord {
                addr,
                len,
                filtered: true,
                obj_size: Some(obj_size),
                id: None,
            });
        }
        let filtered_indirect =
            checked_huge_len(&[sa, ss, 4, ss, ss], "filtered indirect huge record")?;
        if record.len() == filtered_indirect {
            let mut p = 0usize;
            let addr = read_le_uint(take_huge_field(
                record,
                &mut p,
                sa,
                "filtered indirect huge record address",
            )?);
            let len = read_le_uint(take_huge_field(
                record,
                &mut p,
                ss,
                "filtered indirect huge record length",
            )?);
            let _filter_mask = take_huge_field(
                record,
                &mut p,
                4,
                "filtered indirect huge record filter mask",
            )?;
            let obj_size = read_le_uint(take_huge_field(
                record,
                &mut p,
                ss,
                "filtered indirect huge record object size",
            )?);
            let id = read_le_uint(take_huge_field(
                record,
                &mut p,
                ss,
                "filtered indirect huge record ID",
            )?);
            return Ok(HugeRecord {
                addr,
                len,
                filtered: true,
                obj_size: Some(obj_size),
                id: Some(id),
            });
        }

        Err(Error::Unsupported(format!(
            "unsupported huge fractal heap B-tree record size {}",
            record.len()
        )))
    }
}

fn read_u32_le(bytes: &[u8]) -> Result<u32> {
    let bytes = bytes
        .get(..4)
        .ok_or_else(|| Error::InvalidFormat("u32 field is truncated".into()))?;
    let bytes: [u8; 4] = bytes
        .try_into()
        .map_err(|_| Error::InvalidFormat("u32 field is truncated".into()))?;
    Ok(u32::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::{checked_huge_len, read_u32_le, take_huge_field};

    #[test]
    fn huge_record_length_rejects_overflow() {
        let err = checked_huge_len(&[usize::MAX, 1], "huge record").unwrap_err();
        assert!(err.to_string().contains("overflows"));
    }

    #[test]
    fn huge_field_take_rejects_offset_overflow() {
        let bytes = [0u8; 4];
        let mut pos = usize::MAX;
        let err = take_huge_field(&bytes, &mut pos, 1, "huge field").unwrap_err();
        assert!(err.to_string().contains("overflows"));
    }

    #[test]
    fn huge_u32_reader_rejects_truncated_field() {
        let err = read_u32_le(&[0; 3]).unwrap_err();
        assert!(err.to_string().contains("u32 field is truncated"));
    }
}
