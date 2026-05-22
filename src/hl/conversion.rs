use std::any::TypeId;

use crate::error::{Error, Result};
use crate::format::messages::datatype::{ByteOrder, DatatypeClass, DatatypeMessage};
use crate::hl::types::{self, H5Type};

#[derive(Debug, Clone, Copy)]
pub(crate) struct ReadConversion {
    element_size: usize,
    byte_order: Option<ByteOrder>,
    kind: ConversionKind,
}

#[derive(Debug, Clone, Copy)]
enum ConversionKind {
    SameSizeBytes,
    Integer {
        src_size: usize,
        src_signed: bool,
        dst_size: usize,
        dst_signed: bool,
    },
    FloatToFloat {
        src_size: usize,
        dst_size: usize,
    },
    IntegerToFloat {
        src_size: usize,
        src_signed: bool,
        dst_size: usize,
    },
    FloatToInteger {
        src_size: usize,
        dst_size: usize,
        dst_signed: bool,
    },
}

impl ReadConversion {
    pub(crate) fn for_dataset<T: H5Type>(datatype: &DatatypeMessage) -> Result<Self> {
        let requested = T::type_size();
        let stored = usize::try_from(datatype.size)
            .map_err(|_| Error::InvalidFormat("datatype size does not fit in usize".into()))?;
        let byte_order = datatype.byte_order();

        // Dispatch on source class — each per-class helper picks the
        // matching `ConversionKind` (mirroring how libhdf5's
        // `H5T_path_find` registers per-class converters in
        // `H5T__conv_*`). Same-size byte copies are handled in the
        // dispatcher's fallthrough so each helper can stay focused on
        // the type-class-specific decisions.
        let kind = match datatype.class {
            class if is_integer_like_class(class) => {
                Self::kind_for_integer_source::<T>(datatype, requested, stored)?
            }
            DatatypeClass::FloatingPoint => Self::kind_for_float_source::<T>(requested, stored)?,
            _ => Self::kind_for_passthrough(requested, stored)?,
        };

        Ok(Self {
            element_size: stored,
            byte_order,
            kind,
        })
    }

    /// Source class is FixedPoint / Enum / BitField / Time. Mirrors libhdf5's
    /// `H5T__conv_i_i` / `H5T__conv_i_f` selection.
    fn kind_for_integer_source<T: H5Type>(
        datatype: &DatatypeMessage,
        requested: usize,
        stored: usize,
    ) -> Result<ConversionKind> {
        if let Some((dst_signed, dst_size)) = target_integer::<T>() {
            let src_signed = datatype.is_signed().unwrap_or(false);
            Ok(if requested == stored && src_signed == dst_signed {
                ConversionKind::SameSizeBytes
            } else {
                ConversionKind::Integer {
                    src_size: stored,
                    src_signed,
                    dst_size,
                    dst_signed,
                }
            })
        } else if let Some(dst_size) = target_float::<T>() {
            Ok(ConversionKind::IntegerToFloat {
                src_size: stored,
                src_signed: datatype.is_signed().unwrap_or(false),
                dst_size,
            })
        } else {
            Err(Error::InvalidFormat(format!(
                "requested element size {requested} does not match dataset element size {stored}"
            )))
        }
    }

    /// Source class is FloatingPoint. Mirrors libhdf5's
    /// `H5T__conv_f_f` / `H5T__conv_f_i` selection.
    fn kind_for_float_source<T: H5Type>(requested: usize, stored: usize) -> Result<ConversionKind> {
        if let Some(dst_size) = target_float::<T>() {
            Ok(if requested == stored {
                ConversionKind::SameSizeBytes
            } else {
                ConversionKind::FloatToFloat {
                    src_size: stored,
                    dst_size,
                }
            })
        } else if let Some((dst_signed, dst_size)) = target_integer::<T>() {
            Ok(ConversionKind::FloatToInteger {
                src_size: stored,
                dst_size,
                dst_signed,
            })
        } else {
            Err(Error::InvalidFormat(format!(
                "requested element size {requested} does not match dataset element size {stored}"
            )))
        }
    }

    /// Source classes that fall through to a same-size byte copy
    /// (String / Opaque / Reference / Compound / Array / VarLen). The
    /// caller must pre-validate that a typed read is meaningful for the
    /// given source class.
    fn kind_for_passthrough(requested: usize, stored: usize) -> Result<ConversionKind> {
        if requested == stored {
            Ok(ConversionKind::SameSizeBytes)
        } else {
            Err(Error::InvalidFormat(format!(
                "requested element size {requested} does not match dataset element size {stored}"
            )))
        }
    }

    pub(crate) fn bytes_into_vec<T: H5Type>(&self, mut bytes: Vec<u8>) -> Result<Vec<T>> {
        match self.kind {
            ConversionKind::SameSizeBytes => {
                self.convert_bytes_in_place(&mut bytes);
                types::bytes_to_vec(bytes)
            }
            _ => self.converted_bytes_into_vec(&bytes),
        }
    }

    pub(crate) fn bytes_to_vec<T: H5Type>(&self, bytes: Vec<u8>) -> Result<Vec<T>> {
        self.bytes_into_vec(bytes)
    }

    pub(crate) fn bytes_into_slice<T: H5Type>(&self, bytes: &[u8], out: &mut [T]) -> Result<()> {
        let raw_out = types::slice_as_bytes_mut(out);
        self.bytes_into_raw_out(bytes, raw_out)
    }

    fn bytes_into_raw_out(&self, bytes: &[u8], raw_out: &mut [u8]) -> Result<()> {
        match self.kind {
            ConversionKind::SameSizeBytes => {
                if raw_out.len() != bytes.len() {
                    return Err(Error::InvalidFormat(format!(
                        "typed output buffer has {} bytes, expected {}",
                        raw_out.len(),
                        bytes.len()
                    )));
                }
                raw_out.copy_from_slice(bytes);
                self.convert_bytes_in_place(raw_out);
            }
            ConversionKind::Integer {
                src_size,
                src_signed,
                dst_size,
                dst_signed,
            } => {
                convert_integer_bytes_into(
                    bytes,
                    src_size,
                    src_signed,
                    self.byte_order,
                    raw_out,
                    dst_size,
                    dst_signed,
                )?;
            }
            ConversionKind::FloatToFloat { src_size, dst_size } => {
                convert_float_bytes_into(bytes, src_size, self.byte_order, raw_out, dst_size)?;
            }
            ConversionKind::IntegerToFloat {
                src_size,
                src_signed,
                dst_size,
            } => {
                convert_integer_to_float_bytes_into(
                    bytes,
                    src_size,
                    src_signed,
                    self.byte_order,
                    raw_out,
                    dst_size,
                )?;
            }
            ConversionKind::FloatToInteger {
                src_size,
                dst_size,
                dst_signed,
            } => {
                convert_float_to_integer_bytes_into(
                    bytes,
                    src_size,
                    self.byte_order,
                    raw_out,
                    dst_size,
                    dst_signed,
                )?;
            }
        }
        Ok(())
    }

    fn converted_bytes_into_vec<T: H5Type>(&self, bytes: &[u8]) -> Result<Vec<T>> {
        let elem_size = T::type_size();
        if elem_size == 0 {
            return Err(Error::Other("zero-size type".into()));
        }
        let count = converted_element_count(bytes.len(), self.source_element_size()?)?;
        let out_len = count
            .checked_mul(elem_size)
            .ok_or_else(|| Error::InvalidFormat("typed conversion output size overflow".into()))?;

        let mut values = Vec::<T>::with_capacity(count);
        // SAFETY: The vector has capacity for `count` values, `out_len` is
        // exactly `count * size_of::<T>()`, and the conversion routines fully
        // initialize the byte range before `set_len`.
        let raw_out =
            unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut u8, out_len) };
        self.bytes_into_raw_out(bytes, raw_out)?;
        // SAFETY: `bytes_into_raw_out` initialized every byte of each element
        // and `T: H5Type` guarantees a byte-addressable `Copy` representation.
        unsafe {
            values.set_len(count);
        }
        Ok(values)
    }

    fn source_element_size(&self) -> Result<usize> {
        match self.kind {
            ConversionKind::SameSizeBytes => Ok(self.element_size),
            ConversionKind::Integer { src_size, .. }
            | ConversionKind::FloatToFloat { src_size, .. }
            | ConversionKind::IntegerToFloat { src_size, .. }
            | ConversionKind::FloatToInteger { src_size, .. } => {
                if src_size == 0 {
                    Err(Error::InvalidFormat(
                        "conversion source element size is zero".into(),
                    ))
                } else {
                    Ok(src_size)
                }
            }
        }
    }

    pub(crate) fn bytes_to_scalar_from_slice<T: H5Type>(&self, bytes: &[u8]) -> Result<T> {
        if bytes.len() != self.element_size {
            return Err(Error::InvalidFormat(format!(
                "scalar read has {} bytes, expected {}",
                bytes.len(),
                self.element_size
            )));
        }

        let mut value = std::mem::MaybeUninit::<T>::uninit();
        // SAFETY: The raw byte view covers one uninitialized `T`; conversion
        // writes exactly `T::type_size()` bytes before `assume_init`.
        let raw_out = unsafe {
            std::slice::from_raw_parts_mut(value.as_mut_ptr() as *mut u8, T::type_size())
        };
        self.bytes_into_raw_out(bytes, raw_out)?;
        // SAFETY: `bytes_into_raw_out` initialized the complete object bytes.
        Ok(unsafe { value.assume_init() })
    }

    pub(crate) fn is_same_size_bytes(&self) -> bool {
        matches!(self.kind, ConversionKind::SameSizeBytes)
    }

    pub(crate) fn convert_bytes_in_place(&self, bytes: &mut [u8]) {
        maybe_swap_elements(bytes, self.element_size, self.byte_order);
    }
}

pub(crate) fn convert_between_datatypes_into(
    bytes: &[u8],
    source: &DatatypeMessage,
    destination: &DatatypeMessage,
    out: &mut Vec<u8>,
) -> Result<()> {
    let src_size = usize::try_from(source.size)
        .map_err(|_| Error::InvalidFormat("source datatype size does not fit in usize".into()))?;
    let dst_size = usize::try_from(destination.size).map_err(|_| {
        Error::InvalidFormat("destination datatype size does not fit in usize".into())
    })?;
    if is_integer_like_class(source.class) {
        if is_integer_like_class(destination.class) {
            return convert_integer_bytes_to_order_into(
                bytes,
                src_size,
                source.is_signed().unwrap_or(false),
                source.byte_order(),
                dst_size,
                destination.is_signed().unwrap_or(false),
                destination.byte_order(),
                out,
            );
        }
        if destination.class == DatatypeClass::FloatingPoint {
            return convert_integer_to_float_bytes_to_order_into(
                bytes,
                src_size,
                source.is_signed().unwrap_or(false),
                source.byte_order(),
                dst_size,
                destination.byte_order(),
                out,
            );
        }
    }
    if source.class == DatatypeClass::FloatingPoint && is_integer_like_class(destination.class) {
        return convert_float_to_integer_bytes_to_order_into(
            bytes,
            src_size,
            source.byte_order(),
            dst_size,
            destination.is_signed().unwrap_or(false),
            destination.byte_order(),
            out,
        );
    }
    match (source.class, destination.class) {
        (DatatypeClass::FloatingPoint, DatatypeClass::FloatingPoint) => {
            convert_float_bytes_to_order_into(
                bytes,
                src_size,
                source.byte_order(),
                dst_size,
                destination.byte_order(),
                out,
            )
        }
        _ if source.class == destination.class && src_size == dst_size => {
            out.clear();
            out.extend_from_slice(bytes);
            if source.byte_order() != destination.byte_order() {
                maybe_swap_elements(out, src_size, source.byte_order());
                maybe_swap_elements(out, dst_size, destination.byte_order());
            }
            Ok(())
        }
        _ => Err(Error::Unsupported(format!(
            "virtual dataset datatype conversion from {:?} size {} to {:?} size {} is not supported",
            source.class, source.size, destination.class, destination.size
        ))),
    }
}

fn is_integer_like_class(class: DatatypeClass) -> bool {
    matches!(
        class,
        DatatypeClass::FixedPoint
            | DatatypeClass::Enum
            | DatatypeClass::BitField
            | DatatypeClass::Time
    )
}

pub(crate) fn convert_between_datatypes(
    bytes: &[u8],
    source: &DatatypeMessage,
    destination: &DatatypeMessage,
) -> Result<Vec<u8>> {
    let dst_size = usize::try_from(destination.size).map_err(|_| {
        Error::InvalidFormat("destination datatype size does not fit in usize".into())
    })?;
    let capacity = if source.size == 0 {
        0
    } else {
        bytes
            .len()
            .checked_div(source.size as usize)
            .and_then(|len| len.checked_mul(dst_size))
            .unwrap_or(0)
    };
    let mut out = Vec::with_capacity(capacity);
    convert_between_datatypes_into(bytes, source, destination, &mut out)?;
    Ok(out)
}

fn target_integer<T: H5Type>() -> Option<(bool, usize)> {
    let type_id = TypeId::of::<T>();
    if type_id == TypeId::of::<i8>() {
        Some((true, 1))
    } else if type_id == TypeId::of::<i16>() {
        Some((true, 2))
    } else if type_id == TypeId::of::<i32>() {
        Some((true, 4))
    } else if type_id == TypeId::of::<i64>() {
        Some((true, 8))
    } else if type_id == TypeId::of::<i128>() {
        Some((true, 16))
    } else if type_id == TypeId::of::<u8>() {
        Some((false, 1))
    } else if type_id == TypeId::of::<u16>() {
        Some((false, 2))
    } else if type_id == TypeId::of::<u32>() {
        Some((false, 4))
    } else if type_id == TypeId::of::<u64>() {
        Some((false, 8))
    } else if type_id == TypeId::of::<u128>() {
        Some((false, 16))
    } else {
        None
    }
}

fn target_float<T: H5Type>() -> Option<usize> {
    let type_id = TypeId::of::<T>();
    if type_id == TypeId::of::<f32>() {
        Some(4)
    } else if type_id == TypeId::of::<f64>() {
        Some(8)
    } else {
        None
    }
}

fn convert_integer_bytes_into(
    bytes: &[u8],
    src_size: usize,
    src_signed: bool,
    src_order: Option<ByteOrder>,
    out: &mut [u8],
    dst_size: usize,
    dst_signed: bool,
) -> Result<()> {
    validate_integer_conversion_buffers(bytes, src_size, out, dst_size)?;
    for (idx, chunk) in bytes.chunks_exact(src_size).enumerate() {
        let value = if src_signed {
            IntegerValue::Signed(read_signed(chunk, src_order))
        } else {
            IntegerValue::Unsigned(read_unsigned(chunk, src_order))
        };
        let raw = clamp_integer(value, dst_size, dst_signed);
        let dst = conversion_output_window(out, idx, dst_size)?;
        write_uint_ordered(dst, raw, None);
    }
    Ok(())
}

fn convert_integer_bytes_to_order_into(
    bytes: &[u8],
    src_size: usize,
    src_signed: bool,
    src_order: Option<ByteOrder>,
    dst_size: usize,
    dst_signed: bool,
    dst_order: Option<ByteOrder>,
    out: &mut Vec<u8>,
) -> Result<()> {
    if src_size == 0 || dst_size == 0 || src_size > 16 || dst_size > 16 {
        return Err(Error::Unsupported(
            "integer conversion supports 1..=16 byte integer payloads".into(),
        ));
    }
    if bytes.len() % src_size != 0 {
        return Err(Error::InvalidFormat(format!(
            "byte count {} is not a multiple of source integer size {src_size}",
            bytes.len()
        )));
    }

    out.clear();
    out.resize(conversion_output_len(bytes.len(), src_size, dst_size)?, 0);
    for (idx, chunk) in bytes.chunks_exact(src_size).enumerate() {
        let value = if src_signed {
            IntegerValue::Signed(read_signed(chunk, src_order))
        } else {
            IntegerValue::Unsigned(read_unsigned(chunk, src_order))
        };
        let raw = clamp_integer(value, dst_size, dst_signed);
        let dst = conversion_output_window(out.as_mut_slice(), idx, dst_size)?;
        write_uint_ordered(dst, raw, dst_order);
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
enum IntegerValue {
    Signed(i128),
    Unsigned(u128),
}

fn clamp_integer(value: IntegerValue, dst_size: usize, dst_signed: bool) -> u128 {
    let bits = dst_size * 8;
    if dst_signed {
        let (min, max) = signed_bounds(bits);
        let clamped = match value {
            IntegerValue::Signed(value) => value.clamp(min, max),
            IntegerValue::Unsigned(value) => {
                if value > max as u128 {
                    max
                } else {
                    value as i128
                }
            }
        };
        signed_to_raw(clamped, bits)
    } else {
        let max = unsigned_max(bits);
        match value {
            IntegerValue::Signed(value) => {
                if value <= 0 {
                    0
                } else {
                    (value as u128).min(max)
                }
            }
            IntegerValue::Unsigned(value) => value.min(max),
        }
    }
}

fn signed_bounds(bits: usize) -> (i128, i128) {
    if bits == 128 {
        (i128::MIN, i128::MAX)
    } else {
        let max = (1i128 << (bits - 1)) - 1;
        (-1i128 << (bits - 1), max)
    }
}

fn unsigned_max(bits: usize) -> u128 {
    if bits == 128 {
        u128::MAX
    } else {
        (1u128 << bits) - 1
    }
}

fn signed_to_raw(value: i128, bits: usize) -> u128 {
    if bits == 128 {
        return value as u128;
    }
    if value >= 0 {
        value as u128
    } else {
        (1u128 << bits).wrapping_add(value as u128)
    }
}

fn read_unsigned(bytes: &[u8], byte_order: Option<ByteOrder>) -> u128 {
    let little = matches!(byte_order, Some(ByteOrder::LittleEndian) | None);
    let mut value = 0u128;
    if little {
        for (idx, byte) in bytes.iter().take(16).enumerate() {
            value |= (*byte as u128) << (idx * 8);
        }
    } else {
        for byte in bytes.iter().take(16) {
            value = (value << 8) | (*byte as u128);
        }
    }
    value
}

fn read_signed(bytes: &[u8], byte_order: Option<ByteOrder>) -> i128 {
    let n = bytes.len().min(16);
    let unsigned = read_unsigned(bytes, byte_order);
    let bits = n * 8;
    let sign_bit = 1u128 << (bits - 1);
    if unsigned & sign_bit == 0 {
        unsigned as i128
    } else if bits == 128 {
        unsigned as i128
    } else {
        (unsigned as i128) - (1i128 << bits)
    }
}

fn write_uint_ordered(bytes: &mut [u8], value: u128, byte_order: Option<ByteOrder>) {
    if cfg!(target_endian = "little") {
        for (idx, byte) in bytes.iter_mut().enumerate() {
            *byte = (value >> (idx * 8)) as u8;
        }
    } else {
        let n = bytes.len();
        for (idx, byte) in bytes.iter_mut().enumerate() {
            *byte = (value >> ((n - idx - 1) * 8)) as u8;
        }
    }
    maybe_swap_elements(bytes, bytes.len(), byte_order);
}

#[cfg(test)]
fn convert_float_bytes(
    bytes: &[u8],
    src_size: usize,
    src_order: Option<ByteOrder>,
    dst_size: usize,
) -> Result<Vec<u8>> {
    convert_float_bytes_to_order(bytes, src_size, src_order, dst_size, None)
}

fn convert_float_bytes_into(
    bytes: &[u8],
    src_size: usize,
    src_order: Option<ByteOrder>,
    out: &mut [u8],
    dst_size: usize,
) -> Result<()> {
    validate_float_size(src_size, "source")?;
    validate_float_size(dst_size, "target")?;
    validate_conversion_buffers(bytes, src_size, out, dst_size, "source float")?;
    for (idx, chunk) in bytes.chunks_exact(src_size).enumerate() {
        let value = read_float(chunk, src_size, src_order)?;
        let dst = conversion_output_window(out, idx, dst_size)?;
        write_float_ordered(dst, value, None)?;
    }
    Ok(())
}

#[cfg(test)]
fn convert_float_bytes_to_order(
    bytes: &[u8],
    src_size: usize,
    src_order: Option<ByteOrder>,
    dst_size: usize,
    dst_order: Option<ByteOrder>,
) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    convert_float_bytes_to_order_into(bytes, src_size, src_order, dst_size, dst_order, &mut out)?;
    Ok(out)
}

fn convert_float_bytes_to_order_into(
    bytes: &[u8],
    src_size: usize,
    src_order: Option<ByteOrder>,
    dst_size: usize,
    dst_order: Option<ByteOrder>,
    out: &mut Vec<u8>,
) -> Result<()> {
    validate_float_size(src_size, "source")?;
    validate_float_size(dst_size, "target")?;
    if bytes.len() % src_size != 0 {
        return Err(Error::InvalidFormat(format!(
            "byte count {} is not a multiple of source float size {src_size}",
            bytes.len()
        )));
    }
    out.clear();
    out.resize(conversion_output_len(bytes.len(), src_size, dst_size)?, 0);
    for (idx, chunk) in bytes.chunks_exact(src_size).enumerate() {
        let value = read_float(chunk, src_size, src_order)?;
        let dst = conversion_output_window(out.as_mut_slice(), idx, dst_size)?;
        write_float_ordered(dst, value, dst_order)?;
    }
    Ok(())
}

#[cfg(test)]
fn convert_integer_to_float_bytes(
    bytes: &[u8],
    src_size: usize,
    src_signed: bool,
    src_order: Option<ByteOrder>,
    dst_size: usize,
) -> Result<Vec<u8>> {
    convert_integer_to_float_bytes_to_order(bytes, src_size, src_signed, src_order, dst_size, None)
}

fn convert_integer_to_float_bytes_into(
    bytes: &[u8],
    src_size: usize,
    src_signed: bool,
    src_order: Option<ByteOrder>,
    out: &mut [u8],
    dst_size: usize,
) -> Result<()> {
    if src_size == 0 || src_size > 16 {
        return Err(Error::Unsupported(
            "integer-to-float conversion supports 1..=16 byte integer payloads".into(),
        ));
    }
    validate_float_size(dst_size, "target")?;
    validate_conversion_buffers(bytes, src_size, out, dst_size, "source integer")?;
    for (idx, chunk) in bytes.chunks_exact(src_size).enumerate() {
        let value = if src_signed {
            read_signed(chunk, src_order) as f64
        } else {
            read_unsigned(chunk, src_order) as f64
        };
        let dst = conversion_output_window(out, idx, dst_size)?;
        write_float_ordered(dst, value, None)?;
    }
    Ok(())
}

#[cfg(test)]
fn convert_integer_to_float_bytes_to_order(
    bytes: &[u8],
    src_size: usize,
    src_signed: bool,
    src_order: Option<ByteOrder>,
    dst_size: usize,
    dst_order: Option<ByteOrder>,
) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    convert_integer_to_float_bytes_to_order_into(
        bytes, src_size, src_signed, src_order, dst_size, dst_order, &mut out,
    )?;
    Ok(out)
}

fn convert_integer_to_float_bytes_to_order_into(
    bytes: &[u8],
    src_size: usize,
    src_signed: bool,
    src_order: Option<ByteOrder>,
    dst_size: usize,
    dst_order: Option<ByteOrder>,
    out: &mut Vec<u8>,
) -> Result<()> {
    if src_size == 0 || src_size > 16 {
        return Err(Error::Unsupported(
            "integer-to-float conversion supports 1..=16 byte integer payloads".into(),
        ));
    }
    validate_float_size(dst_size, "target")?;
    if bytes.len() % src_size != 0 {
        return Err(Error::InvalidFormat(format!(
            "byte count {} is not a multiple of source integer size {src_size}",
            bytes.len()
        )));
    }
    out.clear();
    out.resize(conversion_output_len(bytes.len(), src_size, dst_size)?, 0);
    for (idx, chunk) in bytes.chunks_exact(src_size).enumerate() {
        let value = if src_signed {
            read_signed(chunk, src_order) as f64
        } else {
            read_unsigned(chunk, src_order) as f64
        };
        let dst = conversion_output_window(out.as_mut_slice(), idx, dst_size)?;
        write_float_ordered(dst, value, dst_order)?;
    }
    Ok(())
}

#[cfg(test)]
fn convert_float_to_integer_bytes(
    bytes: &[u8],
    src_size: usize,
    src_order: Option<ByteOrder>,
    dst_size: usize,
    dst_signed: bool,
) -> Result<Vec<u8>> {
    convert_float_to_integer_bytes_to_order(bytes, src_size, src_order, dst_size, dst_signed, None)
}

fn convert_float_to_integer_bytes_into(
    bytes: &[u8],
    src_size: usize,
    src_order: Option<ByteOrder>,
    out: &mut [u8],
    dst_size: usize,
    dst_signed: bool,
) -> Result<()> {
    validate_float_size(src_size, "source")?;
    if dst_size == 0 || dst_size > 16 {
        return Err(Error::Unsupported(
            "float-to-integer conversion supports 1..=16 byte integer targets".into(),
        ));
    }
    validate_conversion_buffers(bytes, src_size, out, dst_size, "source float")?;
    for (idx, chunk) in bytes.chunks_exact(src_size).enumerate() {
        let value = read_float(chunk, src_size, src_order)?;
        let raw = clamp_float_to_integer(value, dst_size, dst_signed);
        let dst = conversion_output_window(out, idx, dst_size)?;
        write_uint_ordered(dst, raw, None);
    }
    Ok(())
}

#[cfg(test)]
fn convert_float_to_integer_bytes_to_order(
    bytes: &[u8],
    src_size: usize,
    src_order: Option<ByteOrder>,
    dst_size: usize,
    dst_signed: bool,
    dst_order: Option<ByteOrder>,
) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    convert_float_to_integer_bytes_to_order_into(
        bytes, src_size, src_order, dst_size, dst_signed, dst_order, &mut out,
    )?;
    Ok(out)
}

fn convert_float_to_integer_bytes_to_order_into(
    bytes: &[u8],
    src_size: usize,
    src_order: Option<ByteOrder>,
    dst_size: usize,
    dst_signed: bool,
    dst_order: Option<ByteOrder>,
    out: &mut Vec<u8>,
) -> Result<()> {
    validate_float_size(src_size, "source")?;
    if dst_size == 0 || dst_size > 16 {
        return Err(Error::Unsupported(
            "float-to-integer conversion supports 1..=16 byte integer targets".into(),
        ));
    }
    if bytes.len() % src_size != 0 {
        return Err(Error::InvalidFormat(format!(
            "byte count {} is not a multiple of source float size {src_size}",
            bytes.len()
        )));
    }
    out.clear();
    out.resize(conversion_output_len(bytes.len(), src_size, dst_size)?, 0);
    for (idx, chunk) in bytes.chunks_exact(src_size).enumerate() {
        let value = read_float(chunk, src_size, src_order)?;
        let raw = clamp_float_to_integer(value, dst_size, dst_signed);
        let dst = conversion_output_window(out.as_mut_slice(), idx, dst_size)?;
        write_uint_ordered(dst, raw, dst_order);
    }
    Ok(())
}

fn conversion_output_len(byte_len: usize, src_size: usize, dst_size: usize) -> Result<usize> {
    (byte_len / src_size)
        .checked_mul(dst_size)
        .ok_or_else(|| Error::InvalidFormat("conversion output size overflow".into()))
}

fn converted_element_count(byte_len: usize, src_size: usize) -> Result<usize> {
    if src_size == 0 {
        return Err(Error::InvalidFormat(
            "conversion source element size is zero".into(),
        ));
    }
    if byte_len % src_size != 0 {
        return Err(Error::InvalidFormat(format!(
            "byte count {byte_len} is not a multiple of source element size {src_size}"
        )));
    }
    Ok(byte_len / src_size)
}

fn validate_integer_conversion_buffers(
    bytes: &[u8],
    src_size: usize,
    out: &[u8],
    dst_size: usize,
) -> Result<()> {
    if src_size == 0 || dst_size == 0 || src_size > 16 || dst_size > 16 {
        return Err(Error::Unsupported(
            "integer conversion supports 1..=16 byte integer payloads".into(),
        ));
    }
    validate_conversion_buffers(bytes, src_size, out, dst_size, "source integer")
}

fn validate_conversion_buffers(
    bytes: &[u8],
    src_size: usize,
    out: &[u8],
    dst_size: usize,
    source_name: &str,
) -> Result<()> {
    if bytes.len() % src_size != 0 {
        return Err(Error::InvalidFormat(format!(
            "byte count {} is not a multiple of {source_name} size {src_size}",
            bytes.len()
        )));
    }
    let expected = conversion_output_len(bytes.len(), src_size, dst_size)?;
    if out.len() != expected {
        return Err(Error::InvalidFormat(format!(
            "conversion output buffer has {} bytes, expected {expected}",
            out.len()
        )));
    }
    Ok(())
}

fn conversion_output_window(out: &mut [u8], idx: usize, dst_size: usize) -> Result<&mut [u8]> {
    let start = idx
        .checked_mul(dst_size)
        .ok_or_else(|| Error::InvalidFormat("conversion output offset overflow".into()))?;
    let end = start
        .checked_add(dst_size)
        .ok_or_else(|| Error::InvalidFormat("conversion output offset overflow".into()))?;
    out.get_mut(start..end)
        .ok_or_else(|| Error::InvalidFormat("conversion output offset out of bounds".into()))
}

fn validate_float_size(size: usize, role: &str) -> Result<()> {
    if matches!(size, 4 | 8) {
        Ok(())
    } else {
        Err(Error::Unsupported(format!(
            "floating-point conversion supports 4- and 8-byte {role} payloads, got {size}"
        )))
    }
}

fn read_float(bytes: &[u8], size: usize, byte_order: Option<ByteOrder>) -> Result<f64> {
    let input = bytes
        .get(..size)
        .ok_or_else(|| Error::InvalidFormat("floating-point payload is truncated".into()))?;
    let mut raw = [0u8; 8];
    raw[..size].copy_from_slice(input);
    maybe_swap_elements(&mut raw[..size], size, byte_order);
    match size {
        4 => {
            let arr: [u8; 4] = raw[..4]
                .try_into()
                .map_err(|_| Error::InvalidFormat("float32 payload size mismatch".into()))?;
            Ok(f32::from_ne_bytes(arr) as f64)
        }
        8 => {
            let arr: [u8; 8] = raw;
            Ok(f64::from_ne_bytes(arr))
        }
        _ => Err(Error::Unsupported(format!(
            "floating-point conversion supports 4- and 8-byte payloads, got {size}"
        ))),
    }
}

fn write_float_ordered(bytes: &mut [u8], value: f64, byte_order: Option<ByteOrder>) -> Result<()> {
    match bytes.len() {
        4 => bytes.copy_from_slice(&(value as f32).to_ne_bytes()),
        8 => bytes.copy_from_slice(&value.to_ne_bytes()),
        size => {
            return Err(Error::Unsupported(format!(
                "floating-point conversion supports 4- and 8-byte targets, got {size}"
            )));
        }
    }
    maybe_swap_elements(bytes, bytes.len(), byte_order);
    Ok(())
}

fn clamp_float_to_integer(value: f64, dst_size: usize, dst_signed: bool) -> u128 {
    let bits = dst_size * 8;
    if dst_signed {
        let (min, max) = signed_bounds(bits);
        if value.is_nan() {
            return 0;
        }
        let clamped = if value.is_infinite() && value.is_sign_negative() {
            min
        } else if value.is_infinite() {
            max
        } else if value <= min as f64 {
            min
        } else if value >= max as f64 {
            max
        } else {
            value.trunc() as i128
        };
        signed_to_raw(clamped, bits)
    } else {
        let max = unsigned_max(bits);
        if value.is_nan() || value <= 0.0 {
            0
        } else if value.is_infinite() || value >= max as f64 {
            max
        } else {
            value.trunc() as u128
        }
    }
}

pub(crate) fn maybe_swap_elements(
    bytes: &mut [u8],
    element_size: usize,
    byte_order: Option<ByteOrder>,
) {
    if element_size <= 1 {
        return;
    }

    let need_swap = match byte_order {
        Some(ByteOrder::BigEndian) => cfg!(target_endian = "little"),
        Some(ByteOrder::LittleEndian) => cfg!(target_endian = "big"),
        None => false,
    };

    if need_swap {
        for chunk in bytes.chunks_exact_mut(element_size) {
            chunk.reverse();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixed_type(size: u32, signed: bool, order: ByteOrder) -> DatatypeMessage {
        let mut class_bits = [0u8; 3];
        if matches!(order, ByteOrder::BigEndian) {
            class_bits[0] |= 0x01;
        }
        if signed {
            class_bits[0] |= 0x08;
        }
        DatatypeMessage {
            version: 1,
            class: DatatypeClass::FixedPoint,
            class_bits,
            size,
            properties: Vec::new(),
        }
    }

    fn float_type(size: u32, order: ByteOrder) -> DatatypeMessage {
        let mut class_bits = [0u8; 3];
        if matches!(order, ByteOrder::BigEndian) {
            class_bits[0] |= 0x01;
        }
        DatatypeMessage {
            version: 1,
            class: DatatypeClass::FloatingPoint,
            class_bits,
            size,
            properties: Vec::new(),
        }
    }

    fn time_type(size: u32, order: ByteOrder) -> DatatypeMessage {
        let mut class_bits = [0u8; 3];
        if matches!(order, ByteOrder::BigEndian) {
            class_bits[0] |= 0x01;
        }
        DatatypeMessage {
            version: 1,
            class: DatatypeClass::Time,
            class_bits,
            size,
            properties: Vec::new(),
        }
    }

    fn bitfield_type(size: u32, order: ByteOrder) -> DatatypeMessage {
        let mut class_bits = [0u8; 3];
        if matches!(order, ByteOrder::BigEndian) {
            class_bits[0] |= 0x01;
        }
        DatatypeMessage {
            version: 1,
            class: DatatypeClass::BitField,
            class_bits,
            size,
            properties: vec![0, 0, (size * 8) as u8, 0],
        }
    }

    #[test]
    fn reads_big_endian_u128_same_size() {
        let datatype = fixed_type(16, false, ByteOrder::BigEndian);
        let conversion = ReadConversion::for_dataset::<u128>(&datatype).unwrap();
        let raw = 0x0102_0304_0506_0708_1112_1314_1516_1718u128.to_be_bytes();
        let values = conversion.bytes_into_vec::<u128>(raw.to_vec()).unwrap();
        assert_eq!(values, vec![0x0102_0304_0506_0708_1112_1314_1516_1718u128]);
    }

    #[test]
    fn sign_extends_i64_to_i128() {
        let datatype = fixed_type(8, true, ByteOrder::LittleEndian);
        let conversion = ReadConversion::for_dataset::<i128>(&datatype).unwrap();
        let raw = (-42i64).to_le_bytes();
        let values = conversion.bytes_into_vec::<i128>(raw.to_vec()).unwrap();
        assert_eq!(values, vec![-42i128]);
    }

    #[test]
    fn converted_numeric_vec_writes_final_typed_storage() {
        let datatype = fixed_type(2, true, ByteOrder::LittleEndian);
        let conversion = ReadConversion::for_dataset::<i32>(&datatype).unwrap();
        let mut raw = Vec::new();
        raw.extend_from_slice(&(-7i16).to_le_bytes());
        raw.extend_from_slice(&42i16.to_le_bytes());
        let values = conversion.bytes_into_vec::<i32>(raw).unwrap();
        assert_eq!(values, vec![-7, 42]);
    }

    #[test]
    fn converted_scalar_writes_stack_storage() {
        let datatype = fixed_type(2, false, ByteOrder::LittleEndian);
        let conversion = ReadConversion::for_dataset::<u32>(&datatype).unwrap();
        let raw = 513u16.to_le_bytes();
        let value = conversion.bytes_to_scalar_from_slice::<u32>(&raw).unwrap();
        assert_eq!(value, 513);
    }

    #[test]
    fn clamps_float_to_u128_without_128_bit_shift_overflow() {
        let datatype = float_type(8, ByteOrder::LittleEndian);
        let conversion = ReadConversion::for_dataset::<u128>(&datatype).unwrap();
        let raw = f64::INFINITY.to_le_bytes();
        let values = conversion.bytes_into_vec::<u128>(raw.to_vec()).unwrap();
        assert_eq!(values, vec![u128::MAX]);
    }

    #[test]
    fn clamps_unsigned_to_i128_without_128_bit_shift_overflow() {
        let datatype = fixed_type(16, false, ByteOrder::LittleEndian);
        let conversion = ReadConversion::for_dataset::<i128>(&datatype).unwrap();
        let raw = u128::MAX.to_le_bytes();
        let values = conversion.bytes_into_vec::<i128>(raw.to_vec()).unwrap();
        assert_eq!(values, vec![i128::MAX]);
    }

    #[test]
    fn converts_time_datatype_between_integer_payloads() {
        let source = time_type(8, ByteOrder::BigEndian);
        let destination = fixed_type(4, false, ByteOrder::LittleEndian);
        let mut raw = Vec::new();
        raw.extend_from_slice(&1u64.to_be_bytes());
        raw.extend_from_slice(&5_000_000_000u64.to_be_bytes());

        let converted = convert_between_datatypes(&raw, &source, &destination).unwrap();
        let values = converted
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(values, vec![1, u32::MAX]);
    }

    #[test]
    fn converts_time_datatype_to_float_payloads() {
        let source = time_type(4, ByteOrder::LittleEndian);
        let destination = float_type(8, ByteOrder::BigEndian);
        let mut raw = Vec::new();
        raw.extend_from_slice(&0u32.to_le_bytes());
        raw.extend_from_slice(&1_700_000_000u32.to_le_bytes());

        let converted = convert_between_datatypes(&raw, &source, &destination).unwrap();
        let values = converted
            .chunks_exact(8)
            .map(|chunk| f64::from_be_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(values, vec![0.0, 1_700_000_000.0]);
    }

    #[test]
    fn converts_enum_and_bitfield_integer_like_payloads() {
        let enum_source =
            DatatypeMessage::enum_create(fixed_type(2, false, ByteOrder::LittleEndian)).unwrap();
        let unsigned_enum_dest = fixed_type(4, false, ByteOrder::LittleEndian);
        let mut enum_raw = Vec::new();
        enum_raw.extend_from_slice(&2u16.to_le_bytes());
        enum_raw.extend_from_slice(&300u16.to_le_bytes());

        let enum_converted =
            convert_between_datatypes(&enum_raw, &enum_source, &unsigned_enum_dest)
                .expect("enum base integers should use the integer conversion path");
        let enum_values = enum_converted
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(enum_values, vec![2, 300]);

        let bitfield_source = bitfield_type(2, ByteOrder::BigEndian);
        let unsigned_dest = fixed_type(1, false, ByteOrder::LittleEndian);
        let mut bitfield_raw = Vec::new();
        bitfield_raw.extend_from_slice(&0x00abu16.to_be_bytes());
        bitfield_raw.extend_from_slice(&0x01ffu16.to_be_bytes());

        let bitfield_converted =
            convert_between_datatypes(&bitfield_raw, &bitfield_source, &unsigned_dest)
                .expect("bitfield payloads should convert as unsigned integer-like data");
        assert_eq!(bitfield_converted, vec![0xab, 0xff]);
    }

    #[test]
    fn rejects_zero_sized_float_source_without_panicking() {
        let err = convert_float_bytes(&[], 0, Some(ByteOrder::LittleEndian), 4).unwrap_err();
        assert!(err
            .to_string()
            .contains("floating-point conversion supports 4- and 8-byte source payloads"));

        let err = convert_float_to_integer_bytes(&[], 0, Some(ByteOrder::LittleEndian), 4, true)
            .unwrap_err();
        assert!(err
            .to_string()
            .contains("floating-point conversion supports 4- and 8-byte source payloads"));
    }

    #[test]
    fn rejects_zero_sized_integer_source_for_integer_to_float_without_panicking() {
        let err = convert_integer_to_float_bytes(&[], 0, true, Some(ByteOrder::LittleEndian), 4)
            .unwrap_err();
        assert!(err
            .to_string()
            .contains("integer-to-float conversion supports 1..=16 byte integer payloads"));
    }

    #[test]
    fn conversion_output_window_rejects_overflow_and_out_of_bounds() {
        let mut out = vec![0u8; 4];

        let err = conversion_output_window(&mut out, usize::MAX, 2).unwrap_err();
        assert!(
            err.to_string()
                .contains("conversion output offset overflow"),
            "unexpected error: {err}"
        );

        let err = conversion_output_window(&mut out, 2, 4).unwrap_err();
        assert!(
            err.to_string()
                .contains("conversion output offset out of bounds"),
            "unexpected error: {err}"
        );

        assert_eq!(conversion_output_window(&mut out, 1, 2).unwrap().len(), 2);
    }
}
