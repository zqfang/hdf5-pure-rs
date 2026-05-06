use crate::format::messages::datatype::{
    ByteOrder, CompoundField, DatatypeClass, DatatypeMessage, FloatFields,
};

/// High-level datatype descriptor.
#[derive(Debug, Clone)]
pub struct Datatype {
    msg: DatatypeMessage,
}

impl Datatype {
    pub(crate) fn from_message(msg: DatatypeMessage) -> Self {
        Self { msg }
    }

    pub fn enum_create(base: &Datatype) -> crate::Result<Self> {
        Ok(Self {
            msg: DatatypeMessage::enum_create(base.msg.clone())?,
        })
    }

    pub fn enum_insert(&mut self, name: &str, value: u64) -> crate::Result<()> {
        self.msg.enum_insert(name, value)
    }

    /// Return the parsed low-level datatype message.
    pub fn raw_message(&self) -> DatatypeMessage {
        self.msg.clone()
    }

    /// Get datatype creation properties.
    pub fn create_plist(&self) -> crate::hl::plist::datatype_create::DatatypeCreate {
        crate::hl::plist::datatype_create::DatatypeCreate::from_datatype(self)
    }

    /// Return the native-memory representation of this datatype.
    ///
    /// This pure-Rust layer already normalizes reads into Rust-native values,
    /// so the metadata-level native type is represented by the same datatype
    /// descriptor.
    pub fn native_type(&self) -> Datatype {
        self.clone()
    }

    /// Return low/high bit padding policies for numeric datatypes.
    pub fn pad(&self) -> Option<(u8, u8)> {
        match self.msg.class {
            DatatypeClass::FixedPoint | DatatypeClass::BitField | DatatypeClass::FloatingPoint => {
                let plist = self.create_plist();
                Some((plist.low_pad(), plist.high_pad()))
            }
            _ => None,
        }
    }

    /// Total size of one element in bytes.
    pub fn size(&self) -> usize {
        usize::try_from(self.msg.size).unwrap_or(usize::MAX)
    }

    /// Datatype class (FixedPoint, FloatingPoint, String, Compound, etc.).
    pub fn class(&self) -> DatatypeClass {
        self.msg.class
    }

    /// Byte order for numeric types.
    pub fn byte_order(&self) -> Option<ByteOrder> {
        self.msg.byte_order()
    }

    /// Whether a fixed-point type is signed.
    pub fn is_signed(&self) -> Option<bool> {
        self.msg.is_signed()
    }

    /// Bit offset of the significant payload for integer, bitfield,
    /// floating-point, or enum-base datatypes.
    pub fn bit_offset(&self) -> Option<u16> {
        self.msg.bit_offset()
    }

    /// Number of significant bits for integer, bitfield, floating-point,
    /// or enum-base datatypes.
    pub fn precision(&self) -> Option<u16> {
        self.msg.precision()
    }

    /// Whether this is a floating-point type.
    pub fn is_float(&self) -> bool {
        self.msg.class == DatatypeClass::FloatingPoint
    }

    /// Floating-point sign/exponent/mantissa field locations and sizes.
    pub fn float_fields(&self) -> Option<FloatFields> {
        self.msg.float_fields()
    }

    /// Floating-point exponent bias.
    pub fn exponent_bias(&self) -> Option<u32> {
        self.msg.exponent_bias()
    }

    /// Floating-point mantissa normalization code:
    /// 0=none, 1=MSB-set, 2=implied.
    pub fn mantissa_normalization(&self) -> Option<u8> {
        self.msg.mantissa_normalization()
    }

    /// Floating-point internal padding code: 0=zero, 1=one.
    pub fn internal_padding(&self) -> Option<u8> {
        self.msg.internal_padding()
    }

    /// Whether this is an integer type.
    pub fn is_integer(&self) -> bool {
        self.msg.class == DatatypeClass::FixedPoint
    }

    /// Whether this is a string type.
    pub fn is_string(&self) -> bool {
        self.msg.class == DatatypeClass::String
    }

    /// String character set for HDF5 string datatypes: 0=ASCII, 1=UTF-8.
    pub fn char_set(&self) -> Option<u8> {
        self.msg.char_set()
    }

    /// Fixed-length string padding type: 0=null-terminated, 1=null-padded, 2=space-padded.
    pub fn string_padding(&self) -> Option<u8> {
        self.msg.string_padding()
    }

    /// Whether this is a compound type.
    pub fn is_compound(&self) -> bool {
        self.msg.class == DatatypeClass::Compound
    }

    /// Whether this is an enum type.
    pub fn is_enum(&self) -> bool {
        self.msg.class == DatatypeClass::Enum
    }

    /// Whether this is a variable-length type.
    pub fn is_vlen(&self) -> bool {
        self.msg.class == DatatypeClass::VarLen
    }

    /// Opaque datatype tag, if this is an opaque datatype.
    pub fn opaque_tag(&self) -> Option<String> {
        self.msg.opaque_tag()
    }

    /// Reference datatype kind: 0=object reference, 1=dataset region reference.
    pub fn reference_type(&self) -> Option<u8> {
        self.msg.reference_type()
    }

    /// Get compound type fields (returns None if not compound).
    pub fn compound_fields(&self) -> Option<Vec<CompoundField>> {
        self.msg.compound_fields().ok()
    }

    /// Get the number of compound members.
    pub fn compound_nmembers(&self) -> Option<u16> {
        self.msg.compound_nmembers()
    }

    /// Return the zero-based index of a named compound member.
    pub fn member_index(&self, name: &str) -> Option<usize> {
        self.compound_fields()
            .and_then(|fields| fields.iter().position(|field| field.name == name))
    }

    /// Return the byte offset of a compound member by index.
    pub fn member_offset(&self, index: usize) -> Option<usize> {
        self.compound_fields()
            .and_then(|fields| fields.get(index).map(|field| field.byte_offset))
    }

    /// Return the datatype class of a compound member by index.
    pub fn member_class(&self, index: usize) -> Option<DatatypeClass> {
        self.compound_fields()
            .and_then(|fields| fields.get(index).map(|field| field.class))
    }

    /// Return the datatype of a compound member by index.
    pub fn member_type(&self, index: usize) -> Option<Datatype> {
        self.compound_fields().and_then(|fields| {
            fields
                .get(index)
                .map(|field| Datatype::from_message((*field.datatype).clone()))
        })
    }

    /// Get enum members as (name, value) pairs.
    pub fn enum_members(&self) -> Option<Vec<(String, u64)>> {
        self.msg.enum_members().ok()
    }

    pub fn enum_nameof(&self, value: u64) -> crate::Result<Option<String>> {
        self.msg.enum_nameof(value)
    }

    pub fn enum_valueof(&self, name: &str) -> crate::Result<Option<u64>> {
        self.msg.enum_valueof(name)
    }

    /// Get array dimensions and base datatype for array types.
    pub fn array_dims_base(&self) -> Option<(Vec<u64>, Datatype)> {
        self.msg
            .array_dims_base()
            .ok()
            .map(|(dims, base)| (dims, Datatype::from_message(base)))
    }

    /// Get the base datatype for variable-length sequence/string types.
    pub fn vlen_base(&self) -> Option<Datatype> {
        self.msg
            .vlen_base()
            .ok()
            .flatten()
            .map(Datatype::from_message)
    }
}
