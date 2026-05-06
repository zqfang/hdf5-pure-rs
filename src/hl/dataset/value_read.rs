use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::format::messages::datatype::DatatypeMessage;
use crate::hl::value::H5Value;
use crate::io::reader::HdfReader;

use super::{usize_from_u64, Dataset};

impl Dataset {
    /// Read fixed-length strings from the dataset.
    /// Each element is `element_size` bytes, null-padded or space-padded.
    pub fn read_strings(&self) -> Result<Vec<String>> {
        let info = self.info()?;
        let elem_size = usize_from_u64(u64::from(info.datatype.size), "datatype size")?;
        let raw = self.read_raw()?;

        if info.datatype.is_variable_length() {
            // Variable-length data: each element is stored as:
            // sequence_length(4) + global_heap_collection_addr(sizeof_addr) + heap_object_index(4)
            let mut guard = self.inner.lock();
            let sizeof_addr = usize::from(guard.superblock.sizeof_addr);
            let ref_size = vlen_descriptor_size(sizeof_addr)?;
            validate_record_aligned(raw.len(), ref_size, "variable-length string descriptors")?;
            let mut strings = Vec::new();

            for chunk in raw.chunks_exact(ref_size) {
                let (seq_len, addr, index) = decode_vlen_descriptor(chunk, sizeof_addr)?;

                if seq_len == 0 && (addr == 0 || crate::io::reader::is_undef_addr(addr)) {
                    strings.push(String::new());
                } else {
                    if addr == 0 || crate::io::reader::is_undef_addr(addr) {
                        return Err(Error::InvalidFormat(
                            "variable-length string descriptor has length but no heap address"
                                .into(),
                        ));
                    }
                    let gh_ref = crate::format::global_heap::GlobalHeapRef {
                        collection_addr: addr,
                        object_index: index,
                    };
                    let data = crate::format::global_heap::read_global_heap_object(
                        &mut guard.reader,
                        &gh_ref,
                    )?;
                    if data.len() < seq_len {
                        return Err(Error::InvalidFormat(format!(
                            "variable-length string payload too short: expected {seq_len} bytes, got {}",
                            data.len()
                        )));
                    }
                    let data = &data[..seq_len];
                    trace_vlen_read(seq_len, data);
                    strings.push(decode_utf8_string(data, "variable-length string payload")?);
                }
            }
            return Ok(strings);
        }

        // Fixed-length strings
        validate_record_aligned(raw.len(), elem_size, "fixed-length string data")?;
        let padding = info.datatype.string_padding().unwrap_or(1);
        let mut strings = Vec::new();
        for chunk in raw.chunks_exact(elem_size) {
            strings.push(decode_fixed_string_with_padding(chunk, padding)?);
        }
        Ok(strings)
    }

    /// Read a single string (for scalar string datasets/attributes).
    pub fn read_string(&self) -> Result<String> {
        let strings = self.read_strings()?;
        strings
            .into_iter()
            .next()
            .ok_or_else(|| Error::InvalidFormat("no string data".into()))
    }

    /// Read compound type field info. Returns field names, offsets, and sizes.
    pub fn compound_fields(&self) -> Result<Vec<crate::format::messages::datatype::CompoundField>> {
        let info = self.info()?;
        info.datatype.compound_fields()
    }

    /// Read a single field from a compound dataset as typed values.
    /// Example: `ds.read_field::<f64>("x")` reads the "x" field from all records.
    pub fn read_field<T: crate::hl::types::H5Type>(&self, field_name: &str) -> Result<Vec<T>> {
        let fields = self.compound_fields()?;
        let field = fields
            .iter()
            .find(|f| f.name == field_name)
            .ok_or_else(|| Error::InvalidFormat(format!("field '{field_name}' not found")))?;

        if field.size != T::type_size() {
            return Err(Error::InvalidFormat(format!(
                "field '{}' has size {} but requested type has size {}",
                field_name,
                field.size,
                T::type_size()
            )));
        }

        let mut raw = self.read_raw()?;
        self.maybe_byte_swap_field(&mut raw, field)?;

        let info = self.info()?;
        let record_size = usize_from_u64(u64::from(info.datatype.size), "datatype size")?;
        let offset = field.byte_offset;
        let elem_size = field.size;
        let n_records = raw.len() / record_size;

        let mut result = Vec::with_capacity(n_records);
        for i in 0..n_records {
            let start = i
                .checked_mul(record_size)
                .and_then(|value| value.checked_add(offset))
                .ok_or_else(|| Error::InvalidFormat("compound field offset overflow".into()))?;
            let end = start
                .checked_add(elem_size)
                .ok_or_else(|| Error::InvalidFormat("compound field offset overflow".into()))?;
            if end > raw.len() {
                return Err(Error::InvalidFormat(format!(
                    "compound field '{field_name}' exceeds record bounds"
                )));
            }
            let bytes = &raw[start..end];
            // Copy to aligned buffer
            let val = unsafe {
                let mut v = std::mem::MaybeUninit::<T>::uninit();
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), v.as_mut_ptr() as *mut u8, elem_size);
                v.assume_init()
            };
            result.push(val);
        }
        Ok(result)
    }

    /// Read a single compound field as raw per-record byte slices.
    ///
    /// This is useful for compound members whose HDF5 datatype is not directly
    /// representable as a primitive Rust `H5Type`, such as nested compound,
    /// array, variable-length, or reference members. No recursive typed
    /// conversion is performed; callers must interpret each returned byte
    /// vector using the field datatype from [`Dataset::compound_fields`].
    pub fn read_field_raw(&self, field_name: &str) -> Result<Vec<Vec<u8>>> {
        let fields = self.compound_fields()?;
        let field = fields
            .iter()
            .find(|f| f.name == field_name)
            .ok_or_else(|| Error::InvalidFormat(format!("field '{field_name}' not found")))?;

        let raw = self.read_raw()?;
        let info = self.info()?;
        let record_size = usize_from_u64(u64::from(info.datatype.size), "datatype size")?;
        let offset = field.byte_offset;
        let elem_size = field.size;
        let n_records = raw.len() / record_size;

        let mut result = Vec::with_capacity(n_records);
        for i in 0..n_records {
            let start = i
                .checked_mul(record_size)
                .and_then(|value| value.checked_add(offset))
                .ok_or_else(|| Error::InvalidFormat("compound field offset overflow".into()))?;
            let end = start
                .checked_add(elem_size)
                .ok_or_else(|| Error::InvalidFormat("compound field offset overflow".into()))?;
            if end > raw.len() {
                return Err(Error::InvalidFormat(format!(
                    "compound field '{field_name}' exceeds record bounds"
                )));
            }
            result.push(raw[start..end].to_vec());
        }

        Ok(result)
    }

    /// Read a compound field as recursively decoded high-level values.
    ///
    /// This handles nested compound, array, variable-length, and reference
    /// members. Datatype classes without a richer public representation are
    /// returned as `H5Value::Raw`. This API is intended for inspection and
    /// simple extraction, not full libhdf5 typed conversion parity.
    pub fn read_field_values(&self, field_name: &str) -> Result<Vec<H5Value>> {
        let fields = self.compound_fields()?;
        let field = fields
            .iter()
            .find(|f| f.name == field_name)
            .ok_or_else(|| Error::InvalidFormat(format!("field '{field_name}' not found")))?;

        let raw = self.read_raw()?;
        let info = self.info()?;
        let record_size = usize_from_u64(u64::from(info.datatype.size), "datatype size")?;
        let field_end = compound_field_end(field.byte_offset, field.size)?;
        if record_size == 0 || field_end > record_size {
            return Err(Error::InvalidFormat(format!(
                "compound field '{field_name}' exceeds record bounds"
            )));
        }

        let mut guard = self.inner.lock();
        let sizeof_addr = usize::from(guard.superblock.sizeof_addr);
        let n_records = raw.len() / record_size;
        let mut result = Vec::with_capacity(n_records);

        for record in raw.chunks_exact(record_size) {
            let bytes = &record[field.byte_offset..field_end];
            result.push(Self::decode_value(
                &field.datatype,
                bytes,
                sizeof_addr,
                &mut guard.reader,
            )?);
        }

        Ok(result)
    }

    fn decode_value<R: Read + Seek>(
        dtype: &DatatypeMessage,
        bytes: &[u8],
        sizeof_addr: usize,
        reader: &mut HdfReader<R>,
    ) -> Result<H5Value> {
        use crate::format::messages::datatype::{ByteOrder, DatatypeClass};

        match dtype.class {
            DatatypeClass::FixedPoint | DatatypeClass::BitField => {
                let le = matches!(dtype.byte_order(), Some(ByteOrder::LittleEndian) | None);
                if dtype.is_signed().unwrap_or(false) {
                    Ok(H5Value::Int(read_signed_int(bytes, le)))
                } else {
                    Ok(H5Value::UInt(read_unsigned_int(bytes, le)))
                }
            }
            DatatypeClass::FloatingPoint => match dtype.size {
                4 => {
                    let arr = endian_array::<4>(bytes, dtype.byte_order())?;
                    Ok(H5Value::Float(f32::from_le_bytes(arr) as f64))
                }
                8 => {
                    let arr = endian_array::<8>(bytes, dtype.byte_order())?;
                    Ok(H5Value::Float(f64::from_le_bytes(arr)))
                }
                _ => Ok(H5Value::Raw(bytes.to_vec())),
            },
            DatatypeClass::String => Ok(H5Value::String(decode_fixed_string(bytes)?)),
            DatatypeClass::Compound => {
                let fields = dtype.compound_fields()?;
                let mut values = Vec::with_capacity(fields.len());
                for field in fields {
                    let end = field.byte_offset.checked_add(field.size).ok_or_else(|| {
                        Error::InvalidFormat("nested compound field offset overflow".into())
                    })?;
                    if end > bytes.len() {
                        return Err(Error::InvalidFormat(format!(
                            "nested compound field '{}' exceeds record bounds",
                            field.name
                        )));
                    }
                    values.push((
                        field.name.clone(),
                        Self::decode_value(
                            &field.datatype,
                            &bytes[field.byte_offset..end],
                            sizeof_addr,
                            reader,
                        )?,
                    ));
                }
                Ok(H5Value::Compound(values))
            }
            DatatypeClass::Array => {
                let (dims, base) = dtype.array_dims_base()?;
                let count = dims.iter().try_fold(1usize, |acc, &dim| {
                    acc.checked_mul(usize_from_u64(dim, "array dimension")?)
                        .ok_or_else(|| Error::InvalidFormat("array element count overflow".into()))
                })?;
                let elem_size = usize_from_u64(u64::from(base.size), "array base datatype size")?;
                let byte_len = count.checked_mul(elem_size).ok_or_else(|| {
                    Error::InvalidFormat("array field payload size overflow".into())
                })?;
                if elem_size == 0 || bytes.len() < byte_len {
                    return Err(Error::InvalidFormat("array field payload too short".into()));
                }
                let mut values = Vec::with_capacity(count);
                for chunk in bytes[..byte_len].chunks_exact(elem_size) {
                    values.push(Self::decode_value(&base, chunk, sizeof_addr, reader)?);
                }
                Ok(H5Value::Array(values))
            }
            DatatypeClass::VarLen => {
                let base = dtype.vlen_base()?;
                Self::decode_vlen_value(base.as_ref(), bytes, sizeof_addr, reader)
            }
            DatatypeClass::Reference => {
                let n = bytes.len().min(sizeof_addr).min(8);
                let mut addr = 0u64;
                for (i, byte) in bytes.iter().take(n).enumerate() {
                    addr |= u64::from(*byte) << (i * 8);
                }
                Ok(H5Value::Reference(addr))
            }
            DatatypeClass::Enum | DatatypeClass::Opaque | DatatypeClass::Time => {
                Ok(H5Value::Raw(bytes.to_vec()))
            }
        }
    }

    pub(super) fn decode_vlen_value<R: Read + Seek>(
        base: Option<&DatatypeMessage>,
        bytes: &[u8],
        sizeof_addr: usize,
        reader: &mut HdfReader<R>,
    ) -> Result<H5Value> {
        let (seq_len, addr, index) = decode_vlen_descriptor(bytes, sizeof_addr)?;

        if seq_len == 0 || addr == 0 || crate::io::reader::is_undef_addr(addr) {
            return Ok(H5Value::VarLen(Vec::new()));
        }

        let data = crate::format::global_heap::read_global_heap_object(
            reader,
            &crate::format::global_heap::GlobalHeapRef {
                collection_addr: addr,
                object_index: index,
            },
        )?;
        let Some(base) = base else {
            trace_vlen_read(seq_len, &data[..data.len().min(seq_len)]);
            return Ok(H5Value::Raw(data[..data.len().min(seq_len)].to_vec()));
        };

        if base.class == crate::format::messages::datatype::DatatypeClass::String {
            if data.len() < seq_len {
                return Err(Error::InvalidFormat(format!(
                    "variable-length string payload too short: expected {seq_len} bytes, got {}",
                    data.len()
                )));
            }
            let data = &data[..seq_len];
            trace_vlen_read(seq_len, data);
            return Ok(H5Value::String(decode_utf8_string(
                data,
                "variable-length string payload",
            )?));
        }

        let elem_size = usize_from_u64(u64::from(base.size), "vlen base datatype size")?;
        if elem_size == 0 {
            let data = &data[..data.len().min(seq_len)];
            trace_vlen_read(seq_len, data);
            return Ok(H5Value::Raw(data.to_vec()));
        }
        let expected_len = seq_len
            .checked_mul(elem_size)
            .ok_or_else(|| Error::InvalidFormat("variable-length payload size overflow".into()))?;
        if data.len() < expected_len {
            return Err(Error::InvalidFormat(format!(
                "variable-length payload too short: expected {expected_len} bytes, got {}",
                data.len()
            )));
        }
        let data = &data[..expected_len];
        trace_vlen_read(expected_len, data);

        let mut values = Vec::with_capacity(seq_len);
        for chunk in data.chunks_exact(elem_size) {
            values.push(Self::decode_value(base, chunk, sizeof_addr, reader)?);
        }

        Ok(H5Value::VarLen(values))
    }

    /// Byte-swap a specific compound field in the raw data buffer.
    fn maybe_byte_swap_field(
        &self,
        data: &mut [u8],
        field: &crate::format::messages::datatype::CompoundField,
    ) -> Result<()> {
        use crate::format::messages::datatype::{ByteOrder, DatatypeClass};

        if field.size <= 1 {
            return Ok(());
        }

        match field.class {
            DatatypeClass::FixedPoint | DatatypeClass::FloatingPoint | DatatypeClass::BitField => {}
            _ => return Ok(()),
        }

        let need_swap = match field.byte_order {
            Some(ByteOrder::BigEndian) => cfg!(target_endian = "little"),
            Some(ByteOrder::LittleEndian) => cfg!(target_endian = "big"),
            None => false,
        };

        if !need_swap {
            return Ok(());
        }

        let info = self.info()?;
        let record_size = usize_from_u64(u64::from(info.datatype.size), "datatype size")?;
        let field_end = compound_field_end(field.byte_offset, field.size)?;
        if record_size == 0 || field_end > record_size {
            return Err(Error::InvalidFormat(format!(
                "compound field '{}' exceeds record bounds",
                field.name
            )));
        }

        for record in data.chunks_exact_mut(record_size) {
            record[field.byte_offset..field_end].reverse();
        }

        Ok(())
    }
}

fn compound_field_end(offset: usize, size: usize) -> Result<usize> {
    offset
        .checked_add(size)
        .ok_or_else(|| Error::InvalidFormat("compound field offset overflow".into()))
}

fn vlen_descriptor_size(sizeof_addr: usize) -> Result<usize> {
    if sizeof_addr > 8 {
        return Err(Error::Unsupported(format!(
            "variable-length descriptor address width {sizeof_addr} exceeds 64-bit support"
        )));
    }
    4usize
        .checked_add(sizeof_addr)
        .and_then(|value| value.checked_add(4))
        .ok_or_else(|| Error::InvalidFormat("variable-length descriptor size overflow".into()))
}

fn validate_record_aligned(total_len: usize, record_size: usize, context: &str) -> Result<()> {
    if record_size == 0 {
        return Err(Error::InvalidFormat(format!("{context} size is zero")));
    }
    if total_len % record_size != 0 {
        return Err(Error::InvalidFormat(format!(
            "{context} length {total_len} is not a multiple of record size {record_size}"
        )));
    }
    Ok(())
}

fn decode_vlen_descriptor(bytes: &[u8], sizeof_addr: usize) -> Result<(usize, u64, u32)> {
    let descriptor_size = vlen_descriptor_size(sizeof_addr)?;
    if bytes.len() < descriptor_size {
        return Err(Error::InvalidFormat(
            "variable-length descriptor too short".into(),
        ));
    }

    let seq_len_u32 = read_u32_le_at(bytes, 0, "variable-length sequence length")?;
    let seq_len = usize::try_from(seq_len_u32).map_err(|_| {
        Error::InvalidFormat("variable-length sequence length exceeds usize".into())
    })?;
    let mut addr = 0u64;
    let addr_start = 4usize;
    let addr_end = addr_start
        .checked_add(sizeof_addr)
        .ok_or_else(|| Error::InvalidFormat("variable-length address offset overflow".into()))?;
    for (i, byte) in checked_window(bytes, addr_start, sizeof_addr, "variable-length address")?
        .iter()
        .enumerate()
    {
        addr |= u64::from(*byte) << (i * 8);
    }
    let index_pos = addr_end;
    let index = read_u32_le_at(bytes, index_pos, "variable-length heap index")?;

    Ok((seq_len, addr, index))
}

fn checked_window<'a>(data: &'a [u8], pos: usize, len: usize, context: &str) -> Result<&'a [u8]> {
    let end = pos
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))?;
    data.get(pos..end)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))
}

fn read_u32_le_at(data: &[u8], pos: usize, context: &str) -> Result<u32> {
    let bytes = checked_window(data, pos, 4, context)?;
    let bytes: [u8; 4] = bytes
        .try_into()
        .map_err(|_| Error::InvalidFormat(format!("{context} is truncated")))?;
    Ok(u32::from_le_bytes(bytes))
}

#[cfg(feature = "tracehash")]
fn trace_vlen_read(len: usize, data: &[u8]) {
    let mut th = tracehash::th_call!("hdf5.vlen.read");
    th.input_u64(u64::try_from(len).unwrap_or(u64::MAX));
    th.output_value(&(true));
    th.output_value(data);
    th.finish();
}

#[cfg(not(feature = "tracehash"))]
fn trace_vlen_read(_len: usize, _data: &[u8]) {}

fn read_unsigned_int(bytes: &[u8], little_endian: bool) -> u128 {
    let mut value = 0u128;
    let n = bytes.len().min(16);
    if little_endian {
        for (idx, byte) in bytes.iter().take(n).enumerate() {
            value |= (*byte as u128) << (idx * 8);
        }
    } else {
        for byte in bytes.iter().take(n) {
            value = (value << 8) | (*byte as u128);
        }
    }
    value
}

fn read_signed_int(bytes: &[u8], little_endian: bool) -> i128 {
    let unsigned = read_unsigned_int(bytes, little_endian);
    let bits = bytes.len().min(16) * 8;
    if bits == 0 {
        return 0;
    }
    let sign_bit = 1u128 << (bits - 1);
    if unsigned & sign_bit == 0 {
        unsigned as i128
    } else if bits == 128 {
        unsigned as i128
    } else {
        (unsigned as i128) - ((1u128 << bits) as i128)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compound_field_end_rejects_offset_overflow() {
        let err = compound_field_end(usize::MAX, 1).unwrap_err();
        assert!(
            err.to_string().contains("compound field offset overflow"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn validate_record_aligned_rejects_zero_and_trailing_bytes() {
        let err = validate_record_aligned(8, 0, "test records").unwrap_err();
        assert!(
            err.to_string().contains("test records size is zero"),
            "unexpected error: {err}"
        );

        let err = validate_record_aligned(9, 4, "test records").unwrap_err();
        assert!(
            err.to_string()
                .contains("test records length 9 is not a multiple of record size 4"),
            "unexpected error: {err}"
        );

        validate_record_aligned(8, 4, "test records").unwrap();
    }

    #[test]
    fn checked_window_rejects_offset_overflow() {
        let err = checked_window(&[], usize::MAX, 1, "vlen descriptor test").unwrap_err();
        assert!(
            err.to_string()
                .contains("vlen descriptor test offset overflow"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn decode_vlen_descriptor_checks_sequence_length_conversion() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0x1234u64.to_le_bytes());
        bytes.extend_from_slice(&7u32.to_le_bytes());
        let (seq_len, addr, index) = decode_vlen_descriptor(&bytes, 8).unwrap();
        assert_eq!(seq_len, 3);
        assert_eq!(addr, 0x1234);
        assert_eq!(index, 7);
    }

    #[test]
    fn string_decoders_reject_invalid_utf8() {
        assert_eq!(
            decode_fixed_string_with_padding(b"alpha\0tail", 1).unwrap(),
            "alpha"
        );
        assert_eq!(
            decode_fixed_string_with_padding(b"alpha   ", 2).unwrap(),
            "alpha"
        );
        assert!(decode_fixed_string_with_padding(&[0xff, 0], 1).is_err());
        assert!(decode_utf8_string(&[0xff, 0], "vlen test").is_err());
    }
}

fn endian_array<const N: usize>(
    bytes: &[u8],
    order: Option<crate::format::messages::datatype::ByteOrder>,
) -> Result<[u8; N]> {
    if bytes.len() < N {
        return Err(Error::InvalidFormat(
            "floating point payload too short".into(),
        ));
    }
    let mut arr = [0u8; N];
    arr.copy_from_slice(&bytes[..N]);
    match order {
        Some(crate::format::messages::datatype::ByteOrder::BigEndian) => {
            if cfg!(target_endian = "little") {
                arr.reverse();
            }
        }
        Some(crate::format::messages::datatype::ByteOrder::LittleEndian) | None => {
            if cfg!(target_endian = "big") {
                arr.reverse();
            }
        }
    }
    Ok(arr)
}

fn decode_fixed_string(bytes: &[u8]) -> Result<String> {
    decode_fixed_string_with_padding(bytes, 1)
}

fn decode_fixed_string_with_padding(bytes: &[u8], padding: u8) -> Result<String> {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    let bytes = &bytes[..end];
    let s = std::str::from_utf8(bytes)
        .map_err(|_| Error::InvalidFormat("fixed-length string payload is not UTF-8".into()))?;
    Ok(if padding == 2 {
        s.trim_end().to_string()
    } else {
        s.to_string()
    })
}

fn decode_utf8_string(bytes: &[u8], context: &str) -> Result<String> {
    Ok(std::str::from_utf8(bytes)
        .map_err(|_| Error::InvalidFormat(format!("{context} is not UTF-8")))?
        .trim_end_matches('\0')
        .to_string())
}
