//! Fractal heap huge-object access — mirrors libhdf5's `H5HFhuge.c` plus
//! `H5HFbtree2.c` (the v2 B-tree record decode for huge objects).

use std::{
    cmp::Ordering,
    io::{Read, Seek},
};

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
    pub(super) filter_mask: u32,
    pub(super) id: Option<u64>,
}

impl FractalHeapHeader {
    pub(super) fn read_huge<R: Read + Seek>(
        &self,
        reader: &mut HdfReader<R>,
        heap_id: &[u8],
    ) -> Result<Vec<u8>> {
        self.visit_huge(reader, heap_id, |data| Ok(data.to_vec()))
    }

    pub(super) fn read_huge_into<R: Read + Seek>(
        &self,
        reader: &mut HdfReader<R>,
        heap_id: &[u8],
        out: &mut Vec<u8>,
    ) -> Result<()> {
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
            out.clear();
            out.resize(heap_object_len(len, "huge heap object length")?, 0);
            reader.read_bytes_into(out)?;
            self.trace_huge_object(heap_id, addr, len, len, 0, false);
            return Ok(());
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
            out.clear();
            out.resize(heap_object_len(len, "filtered huge heap object length")?, 0);
            reader.read_bytes_into(out)?;
            let filtered = std::mem::take(out);
            crate::filters::apply_pipeline_reverse_with_mask_expected_into(
                &filtered,
                pipeline,
                1,
                filter_mask,
                heap_object_len(obj_size, "filtered huge heap object size")?,
                out,
            )?;
            self.trace_huge_object(heap_id, addr, len, obj_size, filter_mask, true);
            return Ok(());
        }

        if crate::io::reader::is_undef_addr(self.huge_btree_addr) {
            return Err(Error::InvalidFormat(
                "huge heap object ID is indirect but heap has no huge-object B-tree".into(),
            ));
        }

        let id = read_le_uint(&heap_id[1..]);
        let mut matching_huge = None;
        crate::format::btree_v2::visit_matching_records(
            reader,
            self.huge_btree_addr,
            |record| compare_huge_indirect_record_id(record, id, addr_size, len_size),
            |record| {
                let huge = self.decode_huge_record(record)?;
                if huge.id == Some(id) {
                    matching_huge = Some(huge);
                }
                Ok(())
            },
        )?;

        if let Some(huge) = matching_huge {
            reader.seek(huge.addr)?;
            out.clear();
            out.resize(heap_object_len(huge.len, "huge heap object length")?, 0);
            reader.read_bytes_into(out)?;
            if huge.filtered {
                let pipeline = self.filter_pipeline.as_ref().ok_or_else(|| {
                    Error::InvalidFormat("filtered huge object missing filter pipeline".into())
                })?;
                let expected_len = huge.obj_size.ok_or_else(|| {
                    Error::InvalidFormat("filtered huge record missing object size".into())
                })?;
                let filtered = std::mem::take(out);
                crate::filters::apply_pipeline_reverse_with_mask_expected_into(
                    &filtered,
                    pipeline,
                    1,
                    huge.filter_mask,
                    heap_object_len(expected_len, "filtered huge heap object size")?,
                    out,
                )?;
            }
            self.trace_huge_object(
                heap_id,
                huge.addr,
                huge.len,
                huge.obj_size.unwrap_or(huge.len),
                huge.filter_mask,
                huge.filtered,
            );
            return Ok(());
        }

        Err(Error::InvalidFormat(format!(
            "huge fractal heap object id {id} not found"
        )))
    }

    pub(super) fn visit_huge<R, T, F>(
        &self,
        reader: &mut HdfReader<R>,
        heap_id: &[u8],
        op: F,
    ) -> Result<T>
    where
        R: Read + Seek,
        F: FnOnce(&[u8]) -> Result<T>,
    {
        let mut out = Vec::new();
        self.read_huge_into(reader, heap_id, &mut out)?;
        op(&out)
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
                filter_mask: 0,
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
                filter_mask: 0,
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
            let filter_mask = read_u32_le(take_huge_field(
                record,
                &mut p,
                4,
                "filtered direct huge record filter mask",
            )?)?;
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
                filter_mask,
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
            let filter_mask = read_u32_le(take_huge_field(
                record,
                &mut p,
                4,
                "filtered indirect huge record filter mask",
            )?)?;
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
                filter_mask,
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

fn compare_huge_indirect_record_id(
    record: &[u8],
    target_id: u64,
    addr_size: usize,
    len_size: usize,
) -> Ordering {
    let Some(record_id) = huge_indirect_record_id(record, addr_size, len_size) else {
        return Ordering::Equal;
    };
    record_id.cmp(&target_id)
}

fn huge_indirect_record_id(record: &[u8], addr_size: usize, len_size: usize) -> Option<u64> {
    let unfiltered_id_start = addr_size.checked_add(len_size)?;
    let unfiltered_len = unfiltered_id_start.checked_add(len_size)?;
    if record.len() == unfiltered_len {
        return record
            .get(unfiltered_id_start..unfiltered_len)
            .map(read_le_uint);
    }

    let filtered_id_start = addr_size
        .checked_add(len_size)?
        .checked_add(4)?
        .checked_add(len_size)?;
    let filtered_len = filtered_id_start.checked_add(len_size)?;
    if record.len() == filtered_len {
        return record
            .get(filtered_id_start..filtered_len)
            .map(read_le_uint);
    }

    None
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use super::{
        checked_huge_len, compare_huge_indirect_record_id, huge_indirect_record_id, read_u32_le,
        take_huge_field,
    };

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

    #[test]
    fn huge_indirect_record_id_borrows_id_field() {
        let mut record = Vec::new();
        record.extend_from_slice(&0x10u64.to_le_bytes());
        record.extend_from_slice(&5u64.to_le_bytes());
        record.extend_from_slice(&42u64.to_le_bytes());

        assert_eq!(huge_indirect_record_id(&record, 8, 8), Some(42));
        assert_eq!(
            compare_huge_indirect_record_id(&record, 40, 8, 8),
            Ordering::Greater
        );
    }
}
