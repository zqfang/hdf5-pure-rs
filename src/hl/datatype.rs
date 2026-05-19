use crate::format::messages::datatype::{
    ByteOrder, CompoundField, CompoundFields as RawCompoundFields, DatatypeClass, DatatypeMessage,
    EnumMembers as RawEnumMembers, FloatFields,
};
use std::borrow::Cow;

/// hdf5-metno compatibility descriptor alias backed by this crate's parsed datatype message.
pub type TypeDescriptor = crate::format::messages::datatype::DatatypeMessage;

/// High-level datatype descriptor.
#[derive(Debug, Clone)]
pub struct Datatype {
    msg: DatatypeMessage,
}

/// Borrowed view of a compound member's metadata.
#[derive(Debug, Clone)]
pub struct CompoundFieldView<'a> {
    pub raw_name: &'a [u8],
    pub name: Cow<'a, str>,
    pub byte_offset: usize,
    pub size: usize,
    pub class: DatatypeClass,
    pub byte_order: Option<ByteOrder>,
    pub datatype: Datatype,
}

/// Borrowed view of an enum member's metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnumMemberView<'a> {
    pub name: &'a str,
    pub value: u64,
}

/// Iterator over compound datatype fields.
pub struct CompoundFields<'a> {
    inner: RawCompoundFields<'a>,
}

/// Iterator over enum datatype members.
pub struct EnumMembers<'a> {
    inner: RawEnumMembers<'a>,
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
    pub fn raw_message_ref(&self) -> &DatatypeMessage {
        &self.msg
    }

    /// Return the parsed low-level datatype message.
    pub fn raw_message(&self) -> DatatypeMessage {
        self.msg.clone()
    }

    /// hdf5-metno compatibility layer: construct from this crate's descriptor alias; do not remove.
    pub fn from_descriptor(desc: &TypeDescriptor) -> crate::Result<Self> {
        Ok(Self::from_message(desc.clone()))
    }

    /// Construct from an owned descriptor without cloning it first.
    pub fn from_descriptor_owned(desc: TypeDescriptor) -> crate::Result<Self> {
        Ok(Self::from_message(desc))
    }

    /// hdf5-metno compatibility layer: return this crate's descriptor alias; do not remove.
    pub fn to_descriptor(&self) -> crate::Result<TypeDescriptor> {
        Ok(self.raw_message())
    }

    /// Borrow this crate's descriptor alias without cloning it.
    pub fn to_descriptor_ref(&self) -> &TypeDescriptor {
        self.raw_message_ref()
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

    /// Borrow the opaque datatype tag, if this is an opaque datatype.
    pub fn opaque_tag_str(&self) -> Option<&str> {
        self.msg.opaque_tag_str()
    }

    /// Opaque datatype tag, if this is an opaque datatype.
    pub fn opaque_tag(&self) -> Option<String> {
        self.opaque_tag_str().map(str::to_string)
    }

    /// Reference datatype kind: 0=object reference, 1=dataset region reference.
    pub fn reference_type(&self) -> Option<u8> {
        self.msg.reference_type()
    }

    /// Iterate compound type fields (returns an error if not compound).
    pub fn compound_fields_iter(&self) -> crate::Result<CompoundFields<'_>> {
        Ok(CompoundFields {
            inner: self.msg.compound_fields_iter()?,
        })
    }

    /// Get compound type fields (returns `None` if not compound).
    pub fn compound_fields(&self) -> Option<Vec<CompoundField>> {
        let mut fields = Vec::new();
        self.compound_fields_into(&mut fields).ok()?;
        Some(fields)
    }

    /// Store compound type fields in caller-provided storage.
    pub fn compound_fields_into(&self, out: &mut Vec<CompoundField>) -> crate::Result<()> {
        let fields = self.msg.compound_fields_iter()?;
        out.clear();
        out.reserve(fields.len());
        for field in fields {
            let field = field?;
            out.push(CompoundField {
                name: field.name.into_owned(),
                byte_offset: field.byte_offset,
                size: field.size,
                class: field.class,
                byte_order: field.byte_order,
                datatype: Box::new(field.datatype),
            });
        }
        Ok(())
    }

    /// Get the number of compound members.
    pub fn compound_nmembers(&self) -> Option<u16> {
        self.msg.compound_nmembers()
    }

    /// Return the zero-based index of a named compound member.
    pub fn member_index(&self, name: &str) -> Option<usize> {
        self.compound_fields_iter().ok().and_then(|fields| {
            fields.enumerate().find_map(|(index, field)| match field {
                Ok(field) if field.name == name => Some(index),
                _ => None,
            })
        })
    }

    /// Return the byte offset of a compound member by index.
    pub fn member_offset(&self, index: usize) -> Option<usize> {
        self.compound_fields_iter()
            .ok()?
            .nth(index)?
            .ok()
            .map(|field| field.byte_offset)
    }

    /// Return the datatype class of a compound member by index.
    pub fn member_class(&self, index: usize) -> Option<DatatypeClass> {
        self.compound_fields_iter()
            .ok()?
            .nth(index)?
            .ok()
            .map(|field| field.class)
    }

    /// Return the datatype of a compound member by index.
    pub fn member_type(&self, index: usize) -> Option<Datatype> {
        self.compound_fields_iter()
            .ok()?
            .nth(index)?
            .ok()
            .map(|field| field.datatype)
    }

    /// Iterate enum members as borrowed `(name, value)` views.
    pub fn enum_members_iter(&self) -> crate::Result<EnumMembers<'_>> {
        Ok(EnumMembers {
            inner: self.msg.enum_members_iter()?,
        })
    }

    /// Get enum members as owned `(name, value)` pairs.
    pub fn enum_members(&self) -> Option<Vec<(String, u64)>> {
        let mut members = Vec::new();
        self.enum_members_into(&mut members).ok()?;
        Some(members)
    }

    /// Store enum members as owned `(name, value)` pairs in caller-provided storage.
    pub fn enum_members_into(&self, out: &mut Vec<(String, u64)>) -> crate::Result<()> {
        let members = self.msg.enum_members_iter()?;
        out.clear();
        out.reserve(members.len());
        for member in members {
            let member = member?;
            out.push((member.name.to_string(), member.value));
        }
        Ok(())
    }

    /// Find the symbol name corresponding to an enumeration value without allocating.
    pub fn enum_nameof_ref(&self, value: u64) -> crate::Result<Option<&str>> {
        for member in self.msg.enum_members_iter()? {
            let member = member?;
            if member.value == value {
                return Ok(Some(member.name));
            }
        }
        Ok(None)
    }

    pub fn enum_nameof(&self, value: u64) -> crate::Result<Option<String>> {
        self.enum_nameof_ref(value)
            .map(|name| name.map(str::to_string))
    }

    pub fn enum_valueof(&self, name: &str) -> crate::Result<Option<u64>> {
        self.msg.enum_valueof(name)
    }

    /// Iterate array dimensions without allocating a dimension vector.
    pub fn array_dims_iter(
        &self,
    ) -> crate::Result<impl ExactSizeIterator<Item = crate::Result<u64>> + '_> {
        self.msg.array_dims_iter()
    }

    /// Get the base datatype for array types.
    pub fn array_base(&self) -> Option<Datatype> {
        self.msg.array_base().ok().map(Datatype::from_message)
    }

    /// Get array dimensions and base datatype for array types.
    pub fn array_dims_base(&self) -> Option<(Vec<u64>, Datatype)> {
        let mut dims = Vec::new();
        self.array_dims_into(&mut dims).ok()?;
        let base = self.array_base()?;
        Some((dims, base))
    }

    /// Store array dimensions in caller-provided storage.
    pub fn array_dims_into(&self, out: &mut Vec<u64>) -> crate::Result<()> {
        let dims = self.array_dims_iter()?;
        out.clear();
        out.reserve(dims.len());
        for dim in dims {
            out.push(dim?);
        }
        Ok(())
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

impl<'a> Iterator for CompoundFields<'a> {
    type Item = crate::Result<CompoundFieldView<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|field| {
            field.map(|field| CompoundFieldView {
                raw_name: field.raw_name,
                name: field.name,
                byte_offset: field.byte_offset,
                size: field.size,
                class: field.class,
                byte_order: field.byte_order,
                datatype: Datatype::from_message(field.datatype),
            })
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl ExactSizeIterator for CompoundFields<'_> {}

impl<'a> Iterator for EnumMembers<'a> {
    type Item = crate::Result<EnumMemberView<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|member| {
            member.map(|member| EnumMemberView {
                name: member.name,
                value: member.value,
            })
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl ExactSizeIterator for EnumMembers<'_> {}
