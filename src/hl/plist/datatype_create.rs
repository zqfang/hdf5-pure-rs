/// Datatype creation properties read from an existing datatype.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DatatypeCreate {
    low_pad: u8,
    high_pad: u8,
}

impl DatatypeCreate {
    pub(crate) fn from_datatype(dtype: &crate::hl::datatype::Datatype) -> Self {
        let raw = dtype.raw_message_ref();
        let (low_pad, high_pad) = match raw.class {
            crate::format::messages::datatype::DatatypeClass::FixedPoint
            | crate::format::messages::datatype::DatatypeClass::BitField => {
                (raw.class_bits[1] & 0x01, (raw.class_bits[1] >> 1) & 0x01)
            }
            crate::format::messages::datatype::DatatypeClass::FloatingPoint => {
                (raw.class_bits[2] & 0x01, (raw.class_bits[2] >> 1) & 0x01)
            }
            _ => (0, 0),
        };
        Self { low_pad, high_pad }
    }

    /// Padding policy below the significant bit range: 0=zero, 1=one.
    pub fn low_pad(&self) -> u8 {
        self.low_pad
    }

    /// Padding policy above the significant bit range: 0=zero, 1=one.
    pub fn high_pad(&self) -> u8 {
        self.high_pad
    }

    /// Set low/high padding policy.
    pub fn set_pad(&mut self, low_pad: u8, high_pad: u8) {
        self.low_pad = low_pad;
        self.high_pad = high_pad;
    }
}
