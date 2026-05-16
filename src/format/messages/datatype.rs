use crate::error::{Error, Result};

/// A field in a compound datatype.
#[derive(Debug, Clone)]
pub struct CompoundField {
    pub name: String,
    pub byte_offset: usize,
    pub size: usize,
    pub class: DatatypeClass,
    pub byte_order: Option<ByteOrder>,
    pub datatype: Box<DatatypeMessage>,
}

/// Floating-point bit-field layout inside the significant precision region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FloatFields {
    pub sign_position: u8,
    pub exponent_position: u8,
    pub exponent_size: u8,
    pub mantissa_position: u8,
    pub mantissa_size: u8,
}

struct DecodedCompoundMember {
    raw_name: Vec<u8>,
    name: String,
    byte_offset: usize,
    datatype: DatatypeMessage,
}

/// HDF5 datatype class values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatatypeClass {
    FixedPoint,    // 0 - integers
    FloatingPoint, // 1
    Time,          // 2
    String,        // 3
    BitField,      // 4
    Opaque,        // 5
    Compound,      // 6
    Reference,     // 7
    Enum,          // 8
    VarLen,        // 9
    Array,         // 10
}

impl DatatypeClass {
    /// Convert a raw datatype-class byte (low nibble of the version/class byte
    /// in a datatype message) into the corresponding `DatatypeClass`.
    pub fn from_u8(val: u8) -> Result<Self> {
        match val {
            0 => Ok(Self::FixedPoint),
            1 => Ok(Self::FloatingPoint),
            2 => Ok(Self::Time),
            3 => Ok(Self::String),
            4 => Ok(Self::BitField),
            5 => Ok(Self::Opaque),
            6 => Ok(Self::Compound),
            7 => Ok(Self::Reference),
            8 => Ok(Self::Enum),
            9 => Ok(Self::VarLen),
            10 => Ok(Self::Array),
            _ => Err(Error::InvalidFormat(format!(
                "unknown datatype class {val}"
            ))),
        }
    }
}

/// Byte order for numeric types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteOrder {
    LittleEndian,
    BigEndian,
}

/// Parsed Datatype message (type 0x0003).
/// This is a partial parse -- full type info depends on class.
#[derive(Debug, Clone)]
pub struct DatatypeMessage {
    /// Class + version packed byte: version in top 4 bits, class in bottom 4.
    pub version: u8,
    pub class: DatatypeClass,
    /// Class-specific bit fields (3 bytes).
    pub class_bits: [u8; 3],
    /// Total size of the datatype in bytes.
    pub size: u32,
    /// Raw class-specific properties (variable length).
    pub properties: Vec<u8>,
}

const MAX_DATATYPE_ARRAY_DIMS: usize = 32;
const MAX_DATATYPE_MEMBERS: usize = 4096;

impl DatatypeMessage {
    /// Decode the raw disk form of a Datatype (type 0x0003) message into an
    /// in-memory `DatatypeMessage`. Wraps `decode_impl` and emits a tracehash
    /// record when that feature is enabled.
    pub fn decode(data: &[u8]) -> Result<Self> {
        let message = Self::decode_impl(data)?;

        #[cfg(feature = "tracehash")]
        {
            let class_val = data[0] & 0x0F;
            let version = (data[0] >> 4) & 0x0F;
            let class_bits = datatype_class_bits(data)?;
            let size = read_u32_le_at(data, 4, "datatype size")?;

            let mut th = tracehash::th_call!("hdf5.datatype.decode");
            th.input_bytes(data);
            th.output_value(&(true));
            th.output_u64(u64::from(class_val));
            th.finish();

            let mut th = tracehash::th_call!("hdf5.datatype.properties");
            th.input_bytes(data);
            th.output_value(&(true));
            th.output_u64(u64::from(version));
            th.output_u64(u64::from(class_val));
            th.output_value(&class_bits[..]);
            th.output_u64(u64::from(size));
            th.output_value(&message.properties);
            th.finish();
        }

        Ok(message)
    }

    /// Core decoder for a Datatype message. Mirrors libhdf5's
    /// `H5O__dtype_decode` / `H5O__dtype_decode_helper`: parses the common
    /// header (version+class, class bits, byte size), copies the
    /// class-specific property bytes, and runs the per-class validity checks
    /// (fixed-point/bitfield bit ranges, floating-point sign/exponent/mantissa
    /// positions, opaque tag, reference type, vlen subtype, array sizing).
    fn decode_impl(data: &[u8]) -> Result<Self> {
        if data.len() < 8 {
            return Err(Error::InvalidFormat("datatype message too short".into()));
        }

        let class_and_version = data[0];
        let version = (class_and_version >> 4) & 0x0F;
        let class_val = class_and_version & 0x0F;
        if version == 0 || version > 5 {
            return Err(Error::InvalidFormat(format!(
                "invalid datatype message version {version}"
            )));
        }
        if version == 5 {
            return Err(Error::Unsupported(
                "datatype message version 5 is not supported".into(),
            ));
        }
        let class = DatatypeClass::from_u8(class_val)?;

        let class_bits = datatype_class_bits(data)?;
        let size = read_u32_le_at(data, 4, "datatype size")?;
        if size == 0 {
            return Err(Error::InvalidFormat("datatype size is zero".into()));
        }

        let properties = if data.len() > 8 {
            data[8..].to_vec()
        } else {
            Vec::new()
        };

        match class {
            DatatypeClass::FixedPoint | DatatypeClass::BitField if properties.len() < 4 => {
                return Err(Error::InvalidFormat(
                    "datatype message truncated fixed-size properties".into(),
                ));
            }
            DatatypeClass::FloatingPoint if properties.len() < 12 => {
                return Err(Error::InvalidFormat(
                    "datatype message truncated fixed-size properties".into(),
                ));
            }
            DatatypeClass::Array if version < 2 => {
                return Err(Error::InvalidFormat(
                    "array datatype cannot use datatype message version 1".into(),
                ));
            }
            DatatypeClass::Opaque => {
                validate_opaque_properties(class_bits, &properties)?;
            }
            DatatypeClass::Reference => {
                validate_reference_properties(class_bits)?;
            }
            DatatypeClass::VarLen if class_bits[0] & 0x0f > 1 => {
                return Err(Error::InvalidFormat(
                    "variable-length datatype has invalid class type".into(),
                ));
            }
            _ => {}
        }

        // Validate FixedPoint / BitField bit_offset and precision against
        // the byte size, matching the upstream `H5O__dtype_decode_helper`
        // checks ("precision is zero" / "integer offset out of bounds" /
        // "integer offset+precision out of bounds"). The properties layout
        // is bit_offset(u16 LE) + precision(u16 LE).
        if matches!(class, DatatypeClass::FixedPoint | DatatypeClass::BitField) {
            let bit_offset = u64::from(read_u16_le_at(&properties, 0, "datatype bit offset")?);
            let precision = u64::from(read_u16_le_at(&properties, 2, "datatype precision")?);
            let size_bits = u64::from(size)
                .checked_mul(8)
                .ok_or_else(|| Error::InvalidFormat("datatype bit size overflow".into()))?;
            if precision == 0 {
                return Err(Error::InvalidFormat("datatype precision is zero".into()));
            }
            if bit_offset > size_bits {
                return Err(Error::InvalidFormat(format!(
                    "datatype bit offset {bit_offset} exceeds size {size_bits} bits"
                )));
            }
            if bit_offset + precision > size_bits {
                return Err(Error::InvalidFormat(format!(
                    "datatype bit offset+precision ({}) exceeds size {size_bits} bits",
                    bit_offset + precision
                )));
            }
        }

        // Validate FloatingPoint properties against the byte size, matching
        // upstream `H5O__dtype_decode_helper` ("sign bit position out of
        // bounds" / "exponent size can't be zero" / "exponent starting
        // position out of bounds" / "mantissa starting position out of
        // bounds"). Property layout: bit_offset(u16) + precision(u16) +
        // exp_loc(u8) + exp_size(u8) + mant_loc(u8) + mant_size(u8) +
        // exp_bias(u32). Sign bit position lives in class_bits[1].
        if class == DatatypeClass::FloatingPoint {
            let normalization = (class_bits[0] >> 4) & 0x03;
            if normalization == 3 {
                return Err(Error::InvalidFormat(
                    "floating-point mantissa normalization code is invalid".into(),
                ));
            }

            let bit_offset =
                u64::from(read_u16_le_at(&properties, 0, "floating-point bit offset")?);
            let precision = u64::from(read_u16_le_at(&properties, 2, "floating-point precision")?);
            let exp_loc = u64::from(properties[4]);
            let exp_size = u64::from(properties[5]);
            let mant_loc = u64::from(properties[6]);
            let mant_size = u64::from(properties[7]);
            let sign_loc = u64::from(class_bits[1]);
            let size_bits = u64::from(size)
                .checked_mul(8)
                .ok_or_else(|| Error::InvalidFormat("floating-point bit size overflow".into()))?;
            if precision == 0 {
                return Err(Error::InvalidFormat(
                    "floating-point precision is zero".into(),
                ));
            }
            if bit_offset + precision > size_bits {
                return Err(Error::InvalidFormat(format!(
                    "floating-point bit offset+precision ({}) exceeds size {size_bits} bits",
                    bit_offset + precision
                )));
            }
            if exp_size == 0 {
                return Err(Error::InvalidFormat(
                    "floating-point exponent size is zero".into(),
                ));
            }
            if mant_size == 0 {
                return Err(Error::InvalidFormat(
                    "floating-point mantissa size is zero".into(),
                ));
            }
            if sign_loc >= precision {
                return Err(Error::InvalidFormat(format!(
                    "floating-point sign bit position {sign_loc} is outside precision {precision}"
                )));
            }
            if exp_loc + exp_size > precision {
                return Err(Error::InvalidFormat(format!(
                    "floating-point exponent location+size ({}) exceeds precision {precision}",
                    exp_loc + exp_size
                )));
            }
            if mant_loc + mant_size > precision {
                return Err(Error::InvalidFormat(format!(
                    "floating-point mantissa location+size ({}) exceeds precision {precision}",
                    mant_loc + mant_size
                )));
            }
        }

        let message = Self {
            version,
            class,
            class_bits,
            size,
            properties,
        };

        if class == DatatypeClass::Array {
            message.validate_array_size_matches_base()?;
        }

        Ok(message)
    }

    /// Get byte order for numeric types.
    pub fn byte_order(&self) -> Option<ByteOrder> {
        match self.class {
            DatatypeClass::FixedPoint | DatatypeClass::FloatingPoint | DatatypeClass::BitField => {
                if self.class_bits[0] & 0x01 == 0 {
                    Some(ByteOrder::LittleEndian)
                } else {
                    Some(ByteOrder::BigEndian)
                }
            }
            DatatypeClass::Enum => self.enum_base().ok().and_then(|base| base.byte_order()),
            _ => None,
        }
    }

    /// Whether a fixed-point type is signed.
    pub fn is_signed(&self) -> Option<bool> {
        match self.class {
            DatatypeClass::FixedPoint => Some(self.class_bits[0] & 0x08 != 0),
            DatatypeClass::Enum => self.enum_base().ok().and_then(|base| base.is_signed()),
            _ => None,
        }
    }

    /// Bit offset of the significant payload for fixed-point, bitfield,
    /// floating-point, or enum-base datatypes.
    pub fn bit_offset(&self) -> Option<u16> {
        match self.class {
            DatatypeClass::FixedPoint | DatatypeClass::BitField | DatatypeClass::FloatingPoint => {
                read_u16_le_at(&self.properties, 0, "datatype bit offset").ok()
            }
            DatatypeClass::Enum => self.enum_base().ok().and_then(|base| base.bit_offset()),
            _ => None,
        }
    }

    /// Number of significant bits for fixed-point, bitfield, floating-point,
    /// or enum-base datatypes.
    pub fn precision(&self) -> Option<u16> {
        match self.class {
            DatatypeClass::FixedPoint | DatatypeClass::BitField | DatatypeClass::FloatingPoint => {
                read_u16_le_at(&self.properties, 2, "datatype precision").ok()
            }
            DatatypeClass::Enum => self.enum_base().ok().and_then(|base| base.precision()),
            _ => None,
        }
    }

    /// Floating-point sign/exponent/mantissa field locations and sizes.
    pub fn float_fields(&self) -> Option<FloatFields> {
        if self.class != DatatypeClass::FloatingPoint || self.properties.len() < 8 {
            return None;
        }
        Some(FloatFields {
            sign_position: self.class_bits[1],
            exponent_position: self.properties[4],
            exponent_size: self.properties[5],
            mantissa_position: self.properties[6],
            mantissa_size: self.properties[7],
        })
    }

    /// Floating-point exponent bias.
    pub fn exponent_bias(&self) -> Option<u32> {
        if self.class != DatatypeClass::FloatingPoint {
            return None;
        }
        let bytes = self.properties.get(8..12)?;
        Some(u32::from_le_bytes(bytes.try_into().ok()?))
    }

    /// Floating-point mantissa normalization code:
    /// 0=none, 1=MSB-set, 2=implied.
    pub fn mantissa_normalization(&self) -> Option<u8> {
        (self.class == DatatypeClass::FloatingPoint).then_some((self.class_bits[0] >> 4) & 0x03)
    }

    /// Floating-point internal padding code: 0=zero, 1=one.
    pub fn internal_padding(&self) -> Option<u8> {
        (self.class == DatatypeClass::FloatingPoint).then_some((self.class_bits[0] >> 3) & 0x01)
    }

    /// Whether this is a fixed-length string type.
    pub fn is_fixed_string(&self) -> bool {
        self.class == DatatypeClass::String
    }

    /// Whether this is a variable-length type (including vlen strings).
    pub fn is_variable_length(&self) -> bool {
        self.class == DatatypeClass::VarLen
    }

    /// Whether this is a variable-length string datatype.
    pub fn is_variable_string(&self) -> bool {
        self.class == DatatypeClass::VarLen && (self.class_bits[0] & 0x0f) == 1
    }

    /// Get the number of members for compound types.
    pub fn compound_nmembers(&self) -> Option<u16> {
        if self.class == DatatypeClass::Compound {
            Some(u16::from(self.class_bits[0]) | (u16::from(self.class_bits[1]) << 8))
        } else {
            None
        }
    }

    /// Parse compound type member fields.
    /// Returns field names, byte offsets, member sizes, and member datatypes.
    pub fn compound_fields(&self) -> Result<Vec<CompoundField>> {
        self.decode_compound_fields_impl()
    }

    /// Walk this compound datatype's properties and produce the per-member
    /// `CompoundField` table, rejecting empty/oversized member counts.
    fn decode_compound_fields_impl(&self) -> Result<Vec<CompoundField>> {
        let nmembers = usize::from(
            self.compound_nmembers()
                .ok_or_else(|| Error::InvalidFormat("not a compound datatype".into()))?,
        );
        if nmembers == 0 {
            return Err(Error::InvalidFormat(
                "invalid number of compound datatype members: 0".into(),
            ));
        }
        if nmembers > MAX_DATATYPE_MEMBERS {
            return Err(Error::InvalidFormat(format!(
                "compound datatype member count {nmembers} exceeds supported maximum {MAX_DATATYPE_MEMBERS}"
            )));
        }
        let data = &self.properties;
        let members = self.decode_compound_members(data, nmembers)?;
        self.build_compound_fields(&members)
    }

    /// Decode `nmembers` consecutive compound members from `data`.
    fn decode_compound_members(
        &self,
        data: &[u8],
        nmembers: usize,
    ) -> Result<Vec<DecodedCompoundMember>> {
        let mut members = Vec::with_capacity(nmembers);
        let mut p = 0;

        for _ in 0..nmembers {
            if p >= data.len() {
                return Err(Error::InvalidFormat(
                    "compound datatype truncated before member".into(),
                ));
            }
            members.push(self.decode_compound_member(data, &mut p)?);
        }

        Ok(members)
    }

    /// Decode one compound member at the cursor: name string, byte offset
    /// within the record, and the embedded member datatype message.
    fn decode_compound_member(
        &self,
        data: &[u8],
        pos: &mut usize,
    ) -> Result<DecodedCompoundMember> {
        let (raw_name, name) = self.decode_compound_member_name(data, pos)?;
        let byte_offset = self.decode_compound_member_offset(data, pos)?;
        let datatype = self.decode_compound_member_datatype(data, pos)?;
        Ok(DecodedCompoundMember {
            raw_name,
            name,
            byte_offset,
            datatype,
        })
    }

    /// Decode a NUL-terminated compound member name and advance the cursor
    /// past it. Versions 1/2 pad the name to an 8-byte boundary; version 3
    /// stores it tightly packed.
    fn decode_compound_member_name(
        &self,
        data: &[u8],
        pos: &mut usize,
    ) -> Result<(Vec<u8>, String)> {
        let name_start = *pos;
        let name_end = data[*pos..].iter().position(|&b| b == 0).ok_or_else(|| {
            Error::InvalidFormat("compound datatype member name is not terminated".into())
        })?;
        let raw_name_end = checked_usize_add(*pos, name_end, "compound datatype member name")?;
        let raw_name = data[*pos..raw_name_end].to_vec();
        let name = String::from_utf8_lossy(&raw_name).to_string();

        if self.version < 3 {
            let name_with_null = checked_usize_add(name_end, 1, "compound datatype member name")?;
            let padded = align8(name_with_null, "compound datatype member name")?;
            *pos = checked_usize_add(name_start, padded, "compound datatype member name")?;
        } else {
            let advanced = checked_usize_add(name_end, 1, "compound datatype member name")?;
            *pos = checked_usize_add(*pos, advanced, "compound datatype member name")?;
        }

        Ok((raw_name, name))
    }

    /// Decode the byte offset of a compound member within the record. Pre-v3
    /// messages use a fixed 4-byte offset; v3 uses a variable-width offset
    /// sized to the compound type's overall byte size.
    fn decode_compound_member_offset(&self, data: &[u8], pos: &mut usize) -> Result<usize> {
        let offset_size =
            compound_member_offset_size(self.version, self.size_usize("compound datatype size")?)?;
        let offset_end = checked_usize_add(*pos, offset_size, "compound datatype member offset")?;
        if offset_end > data.len() {
            return Err(Error::InvalidFormat(
                "compound datatype member offset is truncated".into(),
            ));
        }
        let byte_offset = read_le_var_usize(&data[*pos..offset_end]);
        *pos = offset_end;
        Ok(byte_offset)
    }

    /// Decode the embedded datatype message for a compound member. For v1
    /// messages this also consumes the legacy inline-array dimension block,
    /// synthesizing an array datatype when any positive dimensions are set.
    fn decode_compound_member_datatype(
        &self,
        data: &[u8],
        pos: &mut usize,
    ) -> Result<DatatypeMessage> {
        let legacy_array_dims = if self.version == 1 {
            Some(Self::decode_legacy_compound_array_dims(data, pos)?)
        } else {
            None
        };

        let header_end = checked_usize_add(*pos, 8, "compound datatype member datatype")?;
        if header_end > data.len() {
            return Err(Error::InvalidFormat(
                "compound datatype member datatype is truncated".into(),
            ));
        }
        let encoded_len = datatype_encoded_len(&data[*pos..])?;
        let encoded_end =
            checked_usize_add(*pos, encoded_len, "compound datatype member datatype")?;
        let base_dt = DatatypeMessage::decode(&data[*pos..encoded_end])?;
        *pos = encoded_end;

        match legacy_array_dims {
            Some(dims) if !dims.is_empty() => create_legacy_compound_array_member(base_dt, dims),
            _ => Ok(base_dt),
        }
    }

    /// Decode the 28-byte legacy compound-member dimension block found in
    /// version 1 messages: rank, reserved bytes, four 4-byte dimensions.
    fn decode_legacy_compound_array_dims(data: &[u8], pos: &mut usize) -> Result<Vec<u64>> {
        let block_end = checked_usize_add(*pos, 28, "compound datatype member dimension block")?;
        if block_end > data.len() {
            return Err(Error::InvalidFormat(
                "compound datatype member dimension block is truncated".into(),
            ));
        }

        let ndims = usize::from(data[*pos]);
        if ndims > 4 {
            return Err(Error::InvalidFormat(
                "compound datatype inline array rank exceeds supported maximum 4".into(),
            ));
        }
        let dims_start = checked_usize_add(*pos, 12, "compound datatype member dimension table")?;
        let mut dims = Vec::with_capacity(ndims);
        for idx in 0usize..4 {
            let elem_offset = idx.checked_mul(4).ok_or_else(|| {
                Error::InvalidFormat("compound datatype dimension offset overflow".into())
            })?;
            let base = checked_usize_add(
                dims_start,
                elem_offset,
                "compound datatype member dimension",
            )?;
            let end = checked_usize_add(base, 4, "compound datatype member dimension")?;
            if end > data.len() {
                return Err(Error::InvalidFormat(
                    "compound datatype member dimension block is truncated".into(),
                ));
            }
            let dim = u64::from(read_u32_le_at(
                data,
                base,
                "compound datatype member dimension",
            )?);
            if idx < ndims {
                if dim == 0 {
                    return Err(Error::InvalidFormat(
                        "compound datatype inline array dimension must be positive".into(),
                    ));
                }
                dims.push(dim);
            }
        }
        *pos = block_end;
        Ok(dims)
    }

    /// Turn a list of decoded compound members into the public
    /// `CompoundField` vector, validating uniqueness and bounds along the way.
    fn build_compound_fields(
        &self,
        members: &[DecodedCompoundMember],
    ) -> Result<Vec<CompoundField>> {
        let mut fields = Vec::with_capacity(members.len());

        for member in members {
            self.validate_compound_member_name(member, members)?;
            fields.push(self.build_compound_field(member, &fields)?);
        }

        Ok(fields)
    }

    /// Reject duplicate compound-member names (matched on the raw NUL-free
    /// byte string, not the UTF-8 lossy form).
    fn validate_compound_member_name(
        &self,
        member: &DecodedCompoundMember,
        members: &[DecodedCompoundMember],
    ) -> Result<()> {
        if members.iter().any(|other| {
            !std::ptr::eq(other, member) && other.raw_name.as_slice() == member.raw_name.as_slice()
        }) {
            return Err(Error::InvalidFormat(format!(
                "duplicated compound field name '{}'",
                member.name
            )));
        }

        Ok(())
    }

    /// Build a single `CompoundField` for `member`, checking that the member
    /// fits inside the compound record and does not overlap any previously
    /// emitted field.
    fn build_compound_field(
        &self,
        member: &DecodedCompoundMember,
        fields: &[CompoundField],
    ) -> Result<CompoundField> {
        let member_type_size = member
            .datatype
            .size_usize("compound datatype member size")?;
        let member_end = member
            .byte_offset
            .checked_add(member_type_size)
            .ok_or_else(|| {
                Error::InvalidFormat("compound datatype member offset overflow".into())
            })?;
        let compound_size = self.size_usize("compound datatype size")?;
        if member_end > compound_size {
            return Err(Error::InvalidFormat(format!(
                "compound datatype member '{}' exceeds record bounds",
                member.name
            )));
        }
        if fields.iter().any(|field| {
            let field_end = field.byte_offset + field.size;
            member.byte_offset < field_end && field.byte_offset < member_end
        }) {
            return Err(Error::InvalidFormat(format!(
                "compound datatype member '{}' overlaps another member",
                member.name
            )));
        }

        Ok(CompoundField {
            name: member.name.clone(),
            byte_offset: member.byte_offset,
            size: member_type_size,
            class: member.datatype.class,
            byte_order: member.datatype.byte_order(),
            datatype: Box::new(member.datatype.clone()),
        })
    }

    /// Get the number of enum members.
    pub fn enum_nmembers(&self) -> Option<u16> {
        if self.class == DatatypeClass::Enum {
            Some(u16::from(self.class_bits[0]) | (u16::from(self.class_bits[1]) << 8))
        } else {
            None
        }
    }

    /// Parse the integer base datatype for enum types.
    pub fn enum_base(&self) -> Result<DatatypeMessage> {
        if self.class != DatatypeClass::Enum {
            return Err(Error::InvalidFormat("not an enum datatype".into()));
        }
        if self.properties.len() < 8 {
            return Err(Error::InvalidFormat(
                "enum datatype base datatype is truncated".into(),
            ));
        }
        let base_len = datatype_encoded_len(&self.properties)?;
        DatatypeMessage::decode(&self.properties[..base_len])
    }

    /// Parse enum type members. Returns Vec of (name, integer_value).
    pub fn enum_members(&self) -> Result<Vec<(String, u64)>> {
        let nmembers = usize::from(
            self.enum_nmembers()
                .ok_or_else(|| Error::InvalidFormat("not an enum datatype".into()))?,
        );
        if nmembers > MAX_DATATYPE_MEMBERS {
            return Err(Error::InvalidFormat(format!(
                "enum datatype member count {nmembers} exceeds supported maximum {MAX_DATATYPE_MEMBERS}"
            )));
        }
        let data = &self.properties;
        if data.len() < 8 {
            return Err(Error::InvalidFormat(
                "enum datatype base datatype is truncated".into(),
            ));
        }

        // Base type (embedded datatype)
        let base_len = datatype_encoded_len(data)?;
        let base_dt = DatatypeMessage::decode(&data[..base_len])?;
        let base_size = base_dt.size_usize("enum datatype base size")?;
        let base_le = !matches!(base_dt.byte_order(), Some(ByteOrder::BigEndian));
        let mut p = base_len;

        // Member names (null-terminated, padded to 8 in v1/v2)
        let mut names = Vec::with_capacity(nmembers);
        for _ in 0..nmembers {
            if p >= data.len() {
                return Err(Error::InvalidFormat(
                    "enum datatype member name is truncated".into(),
                ));
            }
            let name_end = data[p..].iter().position(|&b| b == 0).ok_or_else(|| {
                Error::InvalidFormat("enum datatype member name is not terminated".into())
            })?;
            if name_end == 0 {
                return Err(Error::InvalidFormat(
                    "enum datatype member name must not be empty".into(),
                ));
            }
            let name_slice_end = checked_usize_add(p, name_end, "enum datatype member name")?;
            let name = std::str::from_utf8(&data[p..name_slice_end])
                .map(str::to_string)
                .map_err(|_| {
                    Error::InvalidFormat("enum datatype member name is not UTF-8".into())
                })?;
            if self.version < 3 {
                let name_with_null = checked_usize_add(name_end, 1, "enum datatype member name")?;
                let padded = align8(name_with_null, "enum datatype member name")?;
                let padded_end = checked_usize_add(p, padded, "enum datatype member name")?;
                if padded_end > data.len() {
                    return Err(Error::InvalidFormat(
                        "enum datatype member name padding is truncated".into(),
                    ));
                }
                p = padded_end;
            } else {
                let advance = checked_usize_add(name_end, 1, "enum datatype member name")?;
                p = checked_usize_add(p, advance, "enum datatype member name")?;
            }
            names.push(name);
        }

        // Member values (each base_size bytes)
        let mut members = Vec::with_capacity(nmembers);
        for name in names {
            let value_end = checked_usize_add(p, base_size, "enum datatype member value")?;
            if value_end > data.len() {
                return Err(Error::InvalidFormat(
                    "enum datatype member value is truncated".into(),
                ));
            }
            let val = read_unsigned_value(&data[p..value_end], base_le);
            p = value_end;
            members.push((name, val));
        }

        Ok(members)
    }

    /// Create a new enumeration datatype based on the supplied integer-like
    /// base type. Equivalent to libhdf5's `H5Tenum_create`.
    pub fn enum_create(base: DatatypeMessage) -> Result<Self> {
        if !matches!(
            base.class,
            DatatypeClass::FixedPoint | DatatypeClass::BitField
        ) {
            return Err(Error::InvalidFormat(
                "enum base datatype must be integer-like".into(),
            ));
        }
        let mut properties = Vec::new();
        properties.extend_from_slice(&encode_embedded_datatype_message(&base)?);
        Ok(Self {
            version: 1,
            class: DatatypeClass::Enum,
            class_bits: [0, 0, 0],
            size: base.size,
            properties,
        })
    }

    /// Insert a new (name, value) member into this enumeration datatype.
    /// Both name and value must be unique within the enum. Mirrors
    /// `H5Tenum_insert`.
    pub fn enum_insert(&mut self, name: &str, value: u64) -> Result<()> {
        if self.class != DatatypeClass::Enum {
            return Err(Error::InvalidFormat("not an enum datatype".into()));
        }
        if name.is_empty() {
            return Err(Error::InvalidFormat(
                "enum datatype member name must not be empty".into(),
            ));
        }
        if name.as_bytes().contains(&0) {
            return Err(Error::InvalidFormat(
                "enum datatype member name contains NUL".into(),
            ));
        }
        if self
            .enum_members()?
            .iter()
            .any(|(member, _)| member == name)
        {
            return Err(Error::InvalidFormat(format!(
                "enum datatype member '{name}' already exists"
            )));
        }

        let nmembers = self.enum_nmembers().unwrap_or(0);
        let new_nmembers = nmembers
            .checked_add(1)
            .ok_or_else(|| Error::InvalidFormat("enum datatype member count overflow".into()))?;
        let base = self.enum_base()?;
        let base_len = datatype_encoded_len(&self.properties)?;
        let base_size = base.size_usize("enum datatype base size")?;
        let names_end = enum_member_names_end(self, base_len)?;
        let values_end = self.properties.len();
        let value_bytes = value.to_le_bytes();

        let mut member_name = Vec::new();
        member_name.extend_from_slice(name.as_bytes());
        member_name.push(0);
        if self.version < 3 {
            while member_name.len() % 8 != 0 {
                member_name.push(0);
            }
        }

        let capacity = checked_usize_sum(
            &[self.properties.len(), member_name.len(), base_size],
            "enum datatype properties",
        )?;
        let mut new_properties = Vec::with_capacity(capacity);
        new_properties.extend_from_slice(&self.properties[..names_end]);
        new_properties.extend_from_slice(&member_name);
        new_properties.extend_from_slice(&self.properties[names_end..values_end]);
        new_properties.extend_from_slice(&value_bytes[..base_size.min(value_bytes.len())]);
        if base_size > value_bytes.len() {
            let padded_len = checked_usize_add(
                new_properties.len(),
                base_size - value_bytes.len(),
                "enum datatype value padding",
            )?;
            new_properties.resize(padded_len, 0);
        }

        self.properties = new_properties;
        self.class_bits[0] = (new_nmembers & 0xff) as u8;
        self.class_bits[1] = (new_nmembers >> 8) as u8;
        Ok(())
    }

    /// Find the symbol name corresponding to an enumeration value, returning
    /// `Ok(None)` when no member has that value. Mirrors `H5Tenum_nameof`.
    pub fn enum_nameof(&self, value: u64) -> Result<Option<String>> {
        Ok(self
            .enum_members()?
            .into_iter()
            .find_map(|(name, member_value)| (member_value == value).then_some(name)))
    }

    /// Find the integer value corresponding to a named enumeration member,
    /// returning `Ok(None)` when the name is unknown. Mirrors
    /// `H5Tenum_valueof`.
    pub fn enum_valueof(&self, name: &str) -> Result<Option<u64>> {
        Ok(self
            .enum_members()?
            .into_iter()
            .find_map(|(member_name, value)| (member_name == name).then_some(value)))
    }

    /// Get the character set for string types (0=ASCII, 1=UTF-8).
    pub fn char_set(&self) -> Option<u8> {
        if self.class == DatatypeClass::String {
            Some((self.class_bits[0] >> 4) & 0x0F)
        } else {
            None
        }
    }

    /// Get the string padding type for fixed-length strings.
    ///
    /// Values follow HDF5: 0=null-terminated, 1=null-padded, 2=space-padded.
    pub fn string_padding(&self) -> Option<u8> {
        if self.class == DatatypeClass::String {
            Some(self.class_bits[0] & 0x0F)
        } else {
            None
        }
    }

    /// Get the tag for opaque datatypes.
    pub fn opaque_tag(&self) -> Option<String> {
        if self.class != DatatypeClass::Opaque {
            return None;
        }
        let tag_len = opaque_tag_len_from_class_bits(self.class_bits)
            .ok()?
            .min(self.properties.len());
        let tag = &self.properties[..tag_len];
        let tag_end = tag.iter().position(|&b| b == 0).unwrap_or(tag.len());
        std::str::from_utf8(&tag[..tag_end])
            .map(str::to_string)
            .ok()
    }

    /// Get the reference type for HDF5 reference datatypes.
    ///
    /// Values follow HDF5's datatype class bit field: 0=object reference,
    /// 1=dataset region reference.
    pub fn reference_type(&self) -> Option<u8> {
        if self.class == DatatypeClass::Reference {
            Some(self.class_bits[0] & 0x0f)
        } else {
            None
        }
    }

    /// Get array dimensions and base datatype for array datatypes.
    pub fn array_dims_base(&self) -> Result<(Vec<u64>, DatatypeMessage)> {
        if self.class != DatatypeClass::Array {
            return Err(Error::InvalidFormat("not an array datatype".into()));
        }
        if self.properties.is_empty() {
            return Err(Error::InvalidFormat(
                "array datatype properties are truncated".into(),
            ));
        }
        let ndims = usize::from(self.properties[0]);
        if ndims == 0 {
            return Err(Error::InvalidFormat(
                "array datatype rank must be positive".into(),
            ));
        }
        if ndims > MAX_DATATYPE_ARRAY_DIMS {
            return Err(Error::InvalidFormat(format!(
                "array datatype has too many dimensions: {ndims}"
            )));
        }
        let mut p = if self.version >= 3 { 1usize } else { 4usize };
        if self.properties.len() < p {
            return Err(Error::InvalidFormat(
                "array datatype header is truncated".into(),
            ));
        }
        let dims_len = ndims.checked_mul(4).ok_or_else(|| {
            Error::InvalidFormat("array datatype dimension table overflow".into())
        })?;
        let dims_end = p.checked_add(dims_len).ok_or_else(|| {
            Error::InvalidFormat("array datatype dimension table overflow".into())
        })?;
        if self.properties.len() < dims_end {
            return Err(Error::InvalidFormat(
                "array datatype dimension table is truncated".into(),
            ));
        }

        let mut dims = Vec::with_capacity(ndims);
        for _ in 0..ndims {
            let dim_end = checked_usize_add(p, 4, "array datatype dimension")?;
            if dim_end > self.properties.len() {
                return Err(Error::InvalidFormat(
                    "array datatype dimension table is truncated".into(),
                ));
            }
            let dim = read_u32_le_at(&self.properties, p, "array datatype dimension")?;
            if dim == 0 {
                return Err(Error::InvalidFormat(
                    "array datatype dimension must be positive".into(),
                ));
            }
            dims.push(u64::from(dim));
            p = dim_end;
        }

        if self.version < 3 {
            let permutation_len = ndims.checked_mul(4).ok_or_else(|| {
                Error::InvalidFormat("array datatype permutation table overflow".into())
            })?;
            p = checked_usize_add(p, permutation_len, "array datatype permutation table")?;
            if p > self.properties.len() {
                return Err(Error::InvalidFormat(
                    "array datatype permutation table is truncated".into(),
                ));
            }
        }

        if p >= self.properties.len() {
            return Err(Error::InvalidFormat(
                "array datatype base datatype is missing".into(),
            ));
        }
        let base = DatatypeMessage::decode(&self.properties[p..])?;
        datatype_encoded_len(&self.properties[p..])?;
        Ok((dims, base))
    }

    /// Cross-check that an array datatype's declared byte size equals
    /// `nelem * base_size`, where `nelem` is the product of its dimensions.
    fn validate_array_size_matches_base(&self) -> Result<()> {
        let (dims, base) = self.array_dims_base()?;
        let nelem = dims.iter().try_fold(1usize, |acc, &dim| {
            let dim = usize::try_from(dim)
                .map_err(|_| Error::InvalidFormat("array datatype dimension overflow".into()))?;
            acc.checked_mul(dim)
                .ok_or_else(|| Error::InvalidFormat("array datatype element count overflow".into()))
        })?;
        let base_size = base.size_usize("array datatype base size")?;
        let expected = base_size
            .checked_mul(nelem)
            .ok_or_else(|| Error::InvalidFormat("array datatype byte size overflow".into()))?;
        let actual = self.size_usize("array datatype size")?;
        if actual != expected {
            return Err(Error::InvalidFormat(format!(
                "array datatype size {actual} does not match base size {} times element count {nelem}",
                base.size
            )));
        }
        Ok(())
    }

    /// Get the base datatype for variable-length sequence/string datatypes.
    pub fn vlen_base(&self) -> Result<Option<DatatypeMessage>> {
        if self.class != DatatypeClass::VarLen {
            return Err(Error::InvalidFormat(
                "not a variable-length datatype".into(),
            ));
        }
        if self.properties.is_empty() {
            return Err(Error::InvalidFormat(
                "variable-length datatype properties are truncated".into(),
            ));
        }

        if let Ok(base_len) = datatype_encoded_len(&self.properties) {
            if base_len == self.properties.len() {
                return DatatypeMessage::decode(&self.properties).map(Some);
            }
        }

        if self.properties.len() < 4 {
            return Err(Error::InvalidFormat(
                "variable-length datatype metadata is truncated".into(),
            ));
        }
        if self.properties.len() == 4 {
            return Ok(None);
        }

        let base = &self.properties[4..];
        let base_len = datatype_encoded_len(base)?;
        if base_len != base.len() {
            return Err(Error::InvalidFormat(
                "variable-length datatype base datatype has trailing bytes".into(),
            ));
        }
        DatatypeMessage::decode(base).map(Some)
    }

    /// Convert this datatype's `u32` byte size to `usize`, with a contextual
    /// error message if the value would not fit.
    fn size_usize(&self, context: &'static str) -> Result<usize> {
        usize::try_from(self.size)
            .map_err(|_| Error::InvalidFormat(format!("{context} does not fit in usize")))
    }
}

/// Compute the encoded length of the datatype message at the head of
/// `data`, dispatching to class-specific helpers for the variable-length
/// classes (compound, enum, vlen, array, opaque). Mirrors the size half of
/// libhdf5's `H5O__dtype_size` family.
fn datatype_encoded_len(data: &[u8]) -> Result<usize> {
    if data.len() < 8 {
        return Err(Error::InvalidFormat("datatype message too short".into()));
    }

    let class_and_version = data[0];
    let version = (class_and_version >> 4) & 0x0F;
    let class_val = class_and_version & 0x0F;
    let class = DatatypeClass::from_u8(class_val)?;
    let class_bits = datatype_class_bits(data)?;
    let size = read_u32_len_at(data, 4, "datatype encoded size")?;

    let prop_len = match class {
        DatatypeClass::FixedPoint | DatatypeClass::BitField => 4,
        DatatypeClass::FloatingPoint => 12,
        DatatypeClass::Time | DatatypeClass::String | DatatypeClass::Reference => 0,
        DatatypeClass::Opaque => datatype_opaque_prop_len(data, class_bits)?,
        DatatypeClass::Enum => return datatype_enum_encoded_len(data, version, class_bits),
        DatatypeClass::Compound => return datatype_compound_encoded_len(data, version, size),
        DatatypeClass::VarLen => return datatype_vlen_encoded_len(data),
        DatatypeClass::Array => return datatype_array_encoded_len(data, version),
    };

    let len = checked_usize_add(8, prop_len, "datatype message size")?;
    if len > data.len() {
        return Err(Error::InvalidFormat(
            "datatype message properties are truncated".into(),
        ));
    }
    Ok(len)
}

/// Length of an opaque datatype's properties (just its NUL-terminated tag,
/// padded to a multiple of 8 bytes as recorded in `class_bits[0]`).
fn datatype_opaque_prop_len(data: &[u8], class_bits: [u8; 3]) -> Result<usize> {
    let tag_len = opaque_tag_len_from_class_bits(class_bits)?;
    let end = checked_usize_add(8, tag_len, "opaque datatype tag")?;
    if data.len() < end {
        return Err(Error::InvalidFormat(
            "opaque datatype tag is truncated".into(),
        ));
    }
    Ok(tag_len)
}

/// Validate an opaque datatype's tag: properties must hold the full padded
/// tag and the tag (up to the first NUL) must be valid UTF-8.
fn validate_opaque_properties(class_bits: [u8; 3], properties: &[u8]) -> Result<()> {
    let tag_len = opaque_tag_len_from_class_bits(class_bits)?;
    if properties.len() < tag_len {
        return Err(Error::InvalidFormat(
            "opaque datatype tag is truncated".into(),
        ));
    }
    let tag = &properties[..tag_len];
    let tag_end = tag.iter().position(|&b| b == 0).unwrap_or(tag.len());
    std::str::from_utf8(&tag[..tag_end])
        .map_err(|_| Error::InvalidFormat("opaque datatype tag is not UTF-8".into()))?;
    Ok(())
}

/// Reject reference datatypes whose reference type (low nibble of
/// `class_bits[0]`) is neither object (0) nor region (1).
fn validate_reference_properties(class_bits: [u8; 3]) -> Result<()> {
    let reference_type = class_bits[0] & 0x0f;
    if reference_type > 1 {
        return Err(Error::InvalidFormat(format!(
            "reference datatype type {reference_type} is invalid"
        )));
    }
    Ok(())
}

/// Recover the padded opaque tag length stored in the first class-bit byte
/// of the datatype header (must be a multiple of 8).
fn opaque_tag_len_from_class_bits(class_bits: [u8; 3]) -> Result<usize> {
    let tag_len = usize::from(class_bits[0]);
    if tag_len & 0x07 != 0 {
        return Err(Error::InvalidFormat(
            "opaque datatype tag length is not aligned".into(),
        ));
    }
    Ok(tag_len)
}

/// Compute the encoded byte length of an enum datatype message: header,
/// embedded base datatype, padded member names, then per-member values.
fn datatype_enum_encoded_len(data: &[u8], version: u8, class_bits: [u8; 3]) -> Result<usize> {
    let base_len = datatype_encoded_len(&data[8..])?;
    let base_end = checked_usize_add(8, base_len, "enum datatype base datatype")?;
    let base = DatatypeMessage::decode(&data[8..base_end])?;
    let nmembers = usize::from(class_bits[0]) | (usize::from(class_bits[1]) << 8);
    let mut p = base_end;

    for _ in 0..nmembers {
        p = datatype_advance_enum_member_name(data, p, version)?;
    }

    p = p
        .checked_add(
            nmembers
                .checked_mul(base.size_usize("enum datatype base size")?)
                .ok_or_else(|| {
                    Error::InvalidFormat("enum datatype member value size overflow".into())
                })?,
        )
        .ok_or_else(|| Error::InvalidFormat("enum datatype size overflow".into()))?;
    if p > data.len() {
        return Err(Error::InvalidFormat(
            "enum datatype member value is truncated".into(),
        ));
    }
    Ok(p)
}

/// Advance `pos` past one enum member name, applying 8-byte padding for
/// pre-v3 messages and rejecting empty or unterminated names.
fn datatype_advance_enum_member_name(data: &[u8], pos: usize, version: u8) -> Result<usize> {
    if pos >= data.len() {
        return Err(Error::InvalidFormat(
            "enum datatype member name is truncated".into(),
        ));
    }
    let name_len = checked_usize_add(
        data[pos..].iter().position(|&b| b == 0).ok_or_else(|| {
            Error::InvalidFormat("enum datatype member name is not terminated".into())
        })?,
        1,
        "enum datatype member name",
    )?;
    if name_len == 1 {
        return Err(Error::InvalidFormat(
            "enum datatype member name must not be empty".into(),
        ));
    }
    let advance = if version < 3 {
        align8(name_len, "enum datatype member name")?
    } else {
        name_len
    };
    let next = checked_usize_add(pos, advance, "enum datatype member name")?;
    if next > data.len() {
        return Err(Error::InvalidFormat(
            "enum datatype member name padding is truncated".into(),
        ));
    }
    Ok(next)
}

/// Locate the byte offset within `message.properties` where the enum
/// member-name table ends and the value table begins.
fn enum_member_names_end(message: &DatatypeMessage, base_len: usize) -> Result<usize> {
    let nmembers = usize::from(
        message
            .enum_nmembers()
            .ok_or_else(|| Error::InvalidFormat("not an enum datatype".into()))?,
    );
    let mut pos = base_len;
    for _ in 0..nmembers {
        pos = datatype_advance_enum_member_name(&message.properties, pos, message.version)?;
    }
    Ok(pos)
}

/// Compute the encoded byte length of a compound datatype message by
/// advancing past each of its `nmembers` member descriptors.
fn datatype_compound_encoded_len(data: &[u8], version: u8, size: usize) -> Result<usize> {
    let msg = DatatypeMessage::decode(data)?;
    let nmembers = usize::from(
        msg.compound_nmembers()
            .ok_or_else(|| Error::InvalidFormat("not a compound datatype".into()))?,
    );
    if nmembers == 0 {
        return Err(Error::InvalidFormat(
            "invalid number of compound datatype members: 0".into(),
        ));
    }
    let mut p = 8;

    for _ in 0..nmembers {
        p = datatype_advance_compound_member(data, p, version, size)?;
    }
    if p > data.len() {
        return Err(Error::InvalidFormat(
            "compound datatype member datatype is truncated".into(),
        ));
    }
    Ok(p)
}

/// Advance `pos` past one compound member descriptor: padded name, member
/// offset (fixed 4 bytes pre-v3, variable width in v3), optional v1 legacy
/// dimension block, and the embedded member datatype message.
fn datatype_advance_compound_member(
    data: &[u8],
    pos: usize,
    version: u8,
    size: usize,
) -> Result<usize> {
    let name_start = pos;
    if pos >= data.len() {
        return Err(Error::InvalidFormat(
            "compound datatype member name is truncated".into(),
        ));
    }
    let name_len = data[pos..].iter().position(|&b| b == 0).ok_or_else(|| {
        Error::InvalidFormat("compound datatype member name is not terminated".into())
    })? + 1;
    let mut next = if version < 3 {
        checked_usize_add(
            name_start,
            align8(name_len, "compound datatype member name")?,
            "compound datatype member name",
        )?
    } else {
        checked_usize_add(pos, name_len, "compound datatype member name")?
    };
    next = next
        .checked_add(compound_member_offset_size(version, size)?)
        .ok_or_else(|| Error::InvalidFormat("compound datatype size overflow".into()))?;
    if version == 1 {
        next = next
            .checked_add(28)
            .ok_or_else(|| Error::InvalidFormat("compound datatype size overflow".into()))?;
    }
    if next > data.len() {
        return Err(Error::InvalidFormat(
            "compound datatype member metadata is truncated".into(),
        ));
    }
    let member_len = datatype_encoded_len(&data[next..])?;
    next.checked_add(member_len)
        .ok_or_else(|| Error::InvalidFormat("compound datatype size overflow".into()))
}

/// Compute the encoded byte length of a variable-length datatype message,
/// handling both the tight layout (header + base) and the legacy layout
/// with a 4-byte metadata block before the base datatype.
fn datatype_vlen_encoded_len(data: &[u8]) -> Result<usize> {
    if let Ok(base_len) = datatype_encoded_len(&data[8..]) {
        return checked_usize_add(8, base_len, "vlen datatype size");
    }
    if data.len() < 12 {
        return Err(Error::InvalidFormat(
            "variable-length datatype metadata is truncated".into(),
        ));
    }
    let base_len = datatype_encoded_len(&data[12..])?;
    checked_usize_add(12, base_len, "vlen datatype size")
}

/// Compute the encoded byte length of an array datatype message: rank +
/// dimension table + (pre-v3 only) permutation table + base datatype.
fn datatype_array_encoded_len(data: &[u8], version: u8) -> Result<usize> {
    if data.len() < 9 {
        return Err(Error::InvalidFormat(
            "array datatype properties are truncated".into(),
        ));
    }
    let ndims = usize::from(data[8]);
    let mut p = if version >= 3 { 9usize } else { 12usize };
    if p > data.len() {
        return Err(Error::InvalidFormat(
            "array datatype header is truncated".into(),
        ));
    }
    p = checked_usize_add(
        p,
        ndims.checked_mul(4).ok_or_else(|| {
            Error::InvalidFormat("array datatype dimension table overflow".into())
        })?,
        "array datatype dimension table",
    )?;
    if p > data.len() {
        return Err(Error::InvalidFormat(
            "array datatype dimension table is truncated".into(),
        ));
    }
    if version < 3 {
        p = checked_usize_add(
            p,
            ndims.checked_mul(4).ok_or_else(|| {
                Error::InvalidFormat("array datatype permutation table overflow".into())
            })?,
            "array datatype permutation table",
        )?;
        if p > data.len() {
            return Err(Error::InvalidFormat(
                "array datatype permutation table is truncated".into(),
            ));
        }
    }
    let base_len = datatype_encoded_len(&data[p..])?;
    p = checked_usize_add(p, base_len, "array datatype size")?;
    if p > data.len() {
        return Err(Error::InvalidFormat(
            "array datatype base datatype is truncated".into(),
        ));
    }
    Ok(p)
}

/// Return the byte width used to encode a compound member's offset: 4 for
/// pre-v3 messages, otherwise the minimum width that can represent
/// `compound_size - 1`.
fn compound_member_offset_size(version: u8, compound_size: usize) -> Result<usize> {
    if version < 3 {
        return Ok(4);
    }

    let max_offset = compound_size.checked_sub(1).ok_or_else(|| {
        Error::InvalidFormat("compound datatype member offset size underflow".into())
    })?;
    Ok(bytes_needed(max_offset.max(1)))
}

/// Number of bytes required to encode `value` (minimum 1).
fn bytes_needed(mut value: usize) -> usize {
    let mut bytes = 1;
    while value > 0xff {
        value >>= 8;
        bytes += 1;
    }
    bytes
}

/// Read a little-endian u32 at `data[pos..pos+4]` with a contextual error.
fn read_u32_le_at(data: &[u8], pos: usize, context: &str) -> Result<u32> {
    let end = checked_usize_add(pos, 4, context)?;
    let bytes = data
        .get(pos..end)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))?;
    let bytes: [u8; 4] = bytes
        .try_into()
        .map_err(|_| Error::InvalidFormat(format!("{context} is truncated")))?;
    Ok(u32::from_le_bytes(bytes))
}

/// Read a little-endian u32 length and convert it to `usize`.
fn read_u32_len_at(data: &[u8], pos: usize, context: &'static str) -> Result<usize> {
    usize::try_from(read_u32_le_at(data, pos, context)?)
        .map_err(|_| Error::InvalidFormat(format!("{context} does not fit in usize")))
}

/// Read a little-endian u16 at `data[pos..pos+2]` with a contextual error.
fn read_u16_le_at(data: &[u8], pos: usize, context: &str) -> Result<u16> {
    let end = checked_usize_add(pos, 2, context)?;
    let bytes = data
        .get(pos..end)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))?;
    let bytes: [u8; 2] = bytes
        .try_into()
        .map_err(|_| Error::InvalidFormat(format!("{context} is truncated")))?;
    Ok(u16::from_le_bytes(bytes))
}

/// Extract the three class-specific bit-field bytes (`data[1..4]`) from a
/// datatype message header.
fn datatype_class_bits(data: &[u8]) -> Result<[u8; 3]> {
    let bytes = data
        .get(1..4)
        .ok_or_else(|| Error::InvalidFormat("datatype class bits are truncated".into()))?;
    Ok([bytes[0], bytes[1], bytes[2]])
}

/// Decode a variable-width little-endian unsigned integer into `usize`.
/// Mirrors libhdf5's `H5F_DECODE_LENGTH`-style decoding for fields whose
/// width is configured at file level.
fn read_le_var_usize(bytes: &[u8]) -> usize {
    let mut value = 0usize;
    for (idx, byte) in bytes.iter().enumerate() {
        value |= usize::from(*byte) << (idx * 8);
    }
    value
}

/// Add two `usize` values, mapping overflow to a context-annotated error.
fn checked_usize_add(lhs: usize, rhs: usize, context: &str) -> Result<usize> {
    lhs.checked_add(rhs)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} overflow")))
}

/// Sum a slice of `usize` values, mapping overflow to a context-annotated
/// error.
fn checked_usize_sum(parts: &[usize], context: &str) -> Result<usize> {
    parts.iter().try_fold(0usize, |acc, &part| {
        acc.checked_add(part)
            .ok_or_else(|| Error::InvalidFormat(format!("{context} overflow")))
    })
}

/// Round `len` up to the next multiple of 8, mapping overflow to a
/// context-annotated error.
fn align8(len: usize, context: &str) -> Result<usize> {
    len.checked_add(7)
        .map(|value| value & !7)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} padding overflow")))
}

/// Decode up to 8 bytes as an unsigned integer in the requested endianness.
/// Used to interpret enum member value bytes whose byte order is dictated
/// by the enum's base integer datatype.
fn read_unsigned_value(bytes: &[u8], little_endian: bool) -> u64 {
    let mut value = 0u64;
    if little_endian {
        for (idx, byte) in bytes.iter().take(8).enumerate() {
            value |= u64::from(*byte) << (idx * 8);
        }
    } else {
        for byte in bytes.iter().take(8) {
            value = (value << 8) | u64::from(*byte);
        }
    }
    value
}

/// Synthesize a v2 array `DatatypeMessage` from a base datatype and the
/// dimension list found in a v1 compound member's legacy inline-array
/// block, so the rest of the decoder can treat it as a normal array type.
fn create_legacy_compound_array_member(
    base_dt: DatatypeMessage,
    dims: Vec<u64>,
) -> Result<DatatypeMessage> {
    let nelem = dims.iter().try_fold(1u64, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| Error::InvalidFormat("array datatype size overflow".into()))
    })?;
    let total_size = nelem
        .checked_mul(u64::from(base_dt.size))
        .ok_or_else(|| Error::InvalidFormat("array datatype size overflow".into()))?;
    let size = u32::try_from(total_size)
        .map_err(|_| Error::InvalidFormat("array datatype size overflow".into()))?;

    let ndims = u8::try_from(dims.len())
        .map_err(|_| Error::InvalidFormat("array datatype rank exceeds u8".into()))?;
    let mut properties = vec![ndims];
    properties.extend_from_slice(&[0; 3]);
    for dim in &dims {
        let dim = u32::try_from(*dim)
            .map_err(|_| Error::InvalidFormat("array datatype dimension exceeds u32".into()))?;
        properties.extend_from_slice(&dim.to_le_bytes());
    }
    for _ in &dims {
        properties.extend_from_slice(&0u32.to_le_bytes());
    }
    properties.extend_from_slice(&encode_embedded_datatype_message(&base_dt)?);

    Ok(DatatypeMessage {
        version: 2,
        class: DatatypeClass::Array,
        class_bits: [0, 0, 0],
        size,
        properties,
    })
}

/// Re-encode a `DatatypeMessage` into its raw 8-byte header plus property
/// bytes for embedding inside enum/array/compound parent datatypes.
fn encode_embedded_datatype_message(message: &DatatypeMessage) -> Result<Vec<u8>> {
    let class = match message.class {
        DatatypeClass::FixedPoint => 0u8,
        DatatypeClass::FloatingPoint => 1,
        DatatypeClass::Time => 2,
        DatatypeClass::String => 3,
        DatatypeClass::BitField => 4,
        DatatypeClass::Opaque => 5,
        DatatypeClass::Compound => 6,
        DatatypeClass::Reference => 7,
        DatatypeClass::Enum => 8,
        DatatypeClass::VarLen => 9,
        DatatypeClass::Array => 10,
    };
    if message.version == 0 || message.version > 0x0f {
        return Err(Error::InvalidFormat(format!(
            "datatype message version {} cannot be re-encoded",
            message.version
        )));
    }

    let image_len = checked_usize_add(8, message.properties.len(), "datatype message image")?;
    let mut buf = Vec::with_capacity(image_len);
    buf.push((message.version << 4) | class);
    buf.extend_from_slice(&message.class_bits);
    buf.extend_from_slice(&message.size.to_le_bytes());
    buf.extend_from_slice(&message.properties);
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v1_fixed_point(size: u32, precision: u16) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.push(0x10);
        buf.extend_from_slice(&[0, 0, 0]);
        buf.extend_from_slice(&size.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&precision.to_le_bytes());
        buf
    }

    #[test]
    fn compound_v2_member_does_not_require_legacy_dimension_block() {
        let member = v1_fixed_point(4, 32);
        let mut data = Vec::new();
        data.push(0x26);
        data.extend_from_slice(&[1, 0, 0]);
        data.extend_from_slice(&4u32.to_le_bytes());
        data.extend_from_slice(b"x\0");
        data.extend_from_slice(&[0; 6]);
        data.extend_from_slice(&0u32.to_le_bytes());
        data.extend_from_slice(&member);

        let dtype = DatatypeMessage::decode(&data).expect("compound v2 datatype should decode");
        let fields = dtype
            .compound_fields()
            .expect("compound v2 member should decode without legacy array metadata");
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].name, "x");
        assert_eq!(fields[0].byte_offset, 0);
        assert_eq!(fields[0].size, 4);
    }

    #[test]
    fn compound_v3_uses_variable_width_member_offsets() {
        let member = v1_fixed_point(1, 8);
        let mut data = Vec::new();
        data.push(0x36);
        data.extend_from_slice(&[1, 0, 0]);
        data.extend_from_slice(&0x1234u32.to_le_bytes());
        data.extend_from_slice(b"x\0");
        data.extend_from_slice(&0x1233u16.to_le_bytes());
        data.extend_from_slice(&member);

        let dtype = DatatypeMessage::decode(&data).expect("compound v3 datatype should decode");
        let fields = dtype
            .compound_fields()
            .expect("compound v3 member should use variable-width offsets");
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].byte_offset, 0x1233);
        assert_eq!(fields[0].size, 1);
    }

    #[test]
    fn compound_duplicate_member_names_are_rejected() {
        let member = v1_fixed_point(1, 8);
        let mut data = Vec::new();
        data.push(0x26);
        data.extend_from_slice(&[2, 0, 0]);
        data.extend_from_slice(&2u32.to_le_bytes());

        for offset in 0u32..2 {
            data.extend_from_slice(b"a\0");
            data.extend_from_slice(&[0; 6]);
            data.extend_from_slice(&offset.to_le_bytes());
            data.extend_from_slice(&member);
        }

        let dtype = DatatypeMessage::decode(&data).expect("compound datatype should decode");
        let err = dtype
            .compound_fields()
            .expect_err("duplicate compound member names should be rejected");
        assert!(
            err.to_string().contains("duplicated compound field name"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn compound_duplicate_raw_member_names_are_rejected_even_if_utf8_is_lossy() {
        let member = v1_fixed_point(1, 8);
        let mut data = Vec::new();
        data.push(0x26);
        data.extend_from_slice(&[2, 0, 0]);
        data.extend_from_slice(&2u32.to_le_bytes());

        for offset in 0u32..2 {
            data.extend_from_slice(&[0xff, 0x00]);
            data.extend_from_slice(&[0; 6]);
            data.extend_from_slice(&offset.to_le_bytes());
            data.extend_from_slice(&member);
        }

        let dtype = DatatypeMessage::decode(&data).expect("compound datatype should decode");
        let err = dtype
            .compound_fields()
            .expect_err("duplicate raw compound member names should be rejected");
        assert!(
            err.to_string().contains("duplicated compound field name"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn datatype_u16_reader_rejects_offset_overflow() {
        let err = read_u16_le_at(&[], usize::MAX, "datatype test u16").unwrap_err();
        assert!(
            err.to_string().contains("datatype test u16 overflow"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn datatype_image_sizing_helpers_reject_overflow() {
        assert!(checked_usize_sum(&[usize::MAX, 1], "datatype test sum").is_err());
        assert!(checked_usize_add(usize::MAX, 1, "datatype test add").is_err());
    }
}
