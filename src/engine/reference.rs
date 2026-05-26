use crate::error::{Error, Result};
use std::fmt;

/// Pure Rust object/region reference token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reference {
    object_token: u64,
    region: Option<Vec<u8>>,
    file_name: Option<String>,
    loc_id: Option<u64>,
}

/// Borrowed view of a pure Rust object/region reference token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReferenceRef<'a> {
    object_token: u64,
    region: Option<&'a [u8]>,
    file_name: Option<&'a str>,
    loc_id: Option<u64>,
}

impl Reference {
    /// Render an object token into an existing formatter.
    pub fn print_token_into<W: fmt::Write + ?Sized>(token: u64, out: &mut W) -> fmt::Result {
        out.write_fmt(format_args!("{token:#x}"))
    }

    /// Initialize reference package support.
    pub fn init_package() -> bool {
        true
    }

    /// Create an object reference.
    pub fn create_object(object_token: u64, file_name: Option<String>) -> Self {
        Self {
            object_token,
            region: None,
            file_name,
            loc_id: None,
        }
    }

    /// Create a region reference.
    pub fn create_region(object_token: u64, region: Vec<u8>, file_name: Option<String>) -> Self {
        Self {
            object_token,
            region: Some(region),
            file_name,
            loc_id: None,
        }
    }

    /// Destroy a reference. The pure Rust value is consumed.
    pub fn destroy(self) {}

    /// Set the associated location id.
    pub fn set_loc_id(&mut self, loc_id: u64) {
        self.loc_id = Some(loc_id);
    }

    /// Return the associated location id.
    pub fn get_loc_id(&self) -> Option<u64> {
        self.loc_id
    }

    /// Reopen the referenced file, represented here by returning its name.
    pub fn reopen_file(&self) -> Option<&str> {
        self.file_name.as_deref()
    }

    /// Return reference equality.
    pub fn equal(&self, other: &Self) -> bool {
        self == other
    }

    /// Return a borrowed view of this reference.
    pub fn as_ref(&self) -> ReferenceRef<'_> {
        ReferenceRef {
            object_token: self.object_token,
            region: self.region.as_deref(),
            file_name: self.file_name.as_deref(),
            loc_id: self.loc_id,
        }
    }

    /// Return object token.
    pub fn get_obj_token(&self) -> u64 {
        self.object_token
    }

    /// Set object token.
    pub fn set_obj_token(&mut self, token: u64) {
        self.object_token = token;
    }

    /// Return region bytes.
    pub fn get_region(&self) -> Option<&[u8]> {
        self.region_slice()
    }

    /// Return region bytes.
    pub fn region_slice(&self) -> Option<&[u8]> {
        self.region.as_deref()
    }

    /// Return file name.
    pub fn get_file_name(&self) -> Option<&str> {
        self.file_name.as_deref()
    }

    /// Return the encoded reference length.
    pub fn encoded_len(&self) -> Result<usize> {
        Self::encoded_len_for(self.region.as_deref())
    }

    /// Encode a reference into a caller-provided buffer.
    pub fn encode_into(&self, out: &mut Vec<u8>) -> Result<()> {
        Self::reserve_reference_image(out, self.region.as_deref())?;
        Self::encode_obj_token_into(self.object_token, out);
        Self::encode_region_into(self.region.as_deref(), out)
    }

    /// Encode a reference into a caller-provided fixed buffer.
    pub fn encode_into_slice(&self, out: &mut [u8]) -> Result<usize> {
        self.as_ref().encode_into_slice(out)
    }

    /// Decode a full encoded reference image as a borrowed view.
    pub fn decode_ref(bytes: &[u8]) -> Result<ReferenceRef<'_>> {
        if bytes.len() < 16 {
            return Err(Error::InvalidFormat(
                "reference image is shorter than object token and region length".into(),
            ));
        }
        let object_token =
            Self::decode_token_obj_compat(checked_window(bytes, 0, 8, "reference object token")?)?;
        let region = Self::decode_region_slice(checked_window(
            bytes,
            8,
            bytes.len() - 8,
            "reference region payload",
        )?)?;
        Ok(ReferenceRef {
            object_token,
            region,
            file_name: None,
            loc_id: None,
        })
    }

    /// Decode a region payload as borrowed bytes.
    pub fn decode_region_slice(bytes: &[u8]) -> Result<Option<&[u8]>> {
        if bytes.is_empty() {
            Ok(None)
        } else {
            let len_u64 = read_u64_le_at(bytes, 0, "reference region length")?;
            let len = usize::try_from(len_u64).map_err(|_| {
                Error::InvalidFormat("reference region length exceeds usize".into())
            })?;
            let end = 8usize
                .checked_add(len)
                .ok_or_else(|| Error::InvalidFormat("reference region length overflow".into()))?;
            if bytes.len() != end {
                return Err(Error::InvalidFormat(
                    "reference region payload has invalid length".into(),
                ));
            }
            Ok(Some(&bytes[8..end]))
        }
    }

    /// Encode a heap reference payload into a caller-provided buffer.
    pub fn encode_heap_into(&self, out: &mut Vec<u8>) -> Result<()> {
        self.encode_into(out)
    }

    /// Decode an object-token compatibility payload.
    pub fn decode_token_obj_compat(bytes: &[u8]) -> Result<u64> {
        if bytes.len() != 8 {
            return Err(Error::InvalidFormat(
                "object reference token payload has invalid length".into(),
            ));
        }
        read_u64_le_at(bytes, 0, "object reference token")
    }

    /// Decode a region-token compatibility payload as borrowed bytes.
    pub fn decode_token_region_compat_slice(bytes: &[u8]) -> Result<&[u8]> {
        Self::decode_region_slice(bytes)?
            .ok_or_else(|| Error::InvalidFormat("region reference token payload is empty".into()))
    }

    /// Public object-reference constructor alias.
    pub fn create_object_api(object_token: u64, file_name: Option<String>) -> Self {
        Self::create_object(object_token, file_name)
    }

    /// Public region-reference constructor alias.
    pub fn create_region_api(
        object_token: u64,
        region: Vec<u8>,
        file_name: Option<String>,
    ) -> Self {
        Self::create_region(object_token, region, file_name)
    }

    /// Open-attribute helper returns the target token.
    pub fn open_attr_api_common(&self) -> u64 {
        self.object_token
    }

    /// Encode a region-token compatibility payload into a caller-provided buffer.
    pub fn encode_token_region_compat_into(&self, out: &mut Vec<u8>) -> Result<()> {
        Self::encode_region_into(self.region.as_deref(), out)
    }

    /// Public get-region alias.
    pub fn get_region_api(&self) -> Option<&[u8]> {
        self.get_region()
    }

    /// Encode an object token into a caller-provided buffer.
    pub fn encode_obj_token_into(token: u64, out: &mut Vec<u8>) {
        out.extend_from_slice(&Self::encode_obj_token_array(token));
    }

    /// Encode an object token into a caller-provided fixed buffer.
    pub fn encode_obj_token_slice(token: u64, out: &mut [u8]) -> Result<usize> {
        let dst = out.get_mut(..8).ok_or_else(|| {
            Error::InvalidFormat("object reference token destination is too small".into())
        })?;
        dst.copy_from_slice(&Self::encode_obj_token_array(token));
        Ok(8)
    }

    /// Encode an object token as fixed-size bytes.
    pub fn encode_obj_token_array(token: u64) -> [u8; 8] {
        token.to_le_bytes()
    }

    /// Encode a region payload into a caller-provided buffer.
    pub fn encode_region_into(region: Option<&[u8]>, out: &mut Vec<u8>) -> Result<()> {
        let len = region.map_or(Ok(0u64), |region| {
            u64::try_from(region.len())
                .map_err(|_| Error::InvalidFormat("reference region length exceeds u64".into()))
        })?;
        out.try_reserve_exact(Self::encoded_region_len_for(region)?)
            .map_err(|err| {
                Error::InvalidFormat(format!("reference region allocation failed: {err}"))
            })?;
        out.extend_from_slice(&len.to_le_bytes());
        if let Some(region) = region {
            out.extend_from_slice(region);
        }
        Ok(())
    }

    /// Encode a region payload into a caller-provided fixed buffer.
    pub fn encode_region_slice(region: Option<&[u8]>, out: &mut [u8]) -> Result<usize> {
        let len = region.map_or(Ok(0u64), |region| {
            u64::try_from(region.len())
                .map_err(|_| Error::InvalidFormat("reference region length exceeds u64".into()))
        })?;
        let total_len = Self::encoded_region_len_for(region)?;
        let dst = out.get_mut(..total_len).ok_or_else(|| {
            Error::InvalidFormat("reference region destination is too small".into())
        })?;
        dst[..8].copy_from_slice(&len.to_le_bytes());
        if let Some(region) = region {
            dst[8..].copy_from_slice(region);
        }
        Ok(total_len)
    }

    fn encoded_len_for(region: Option<&[u8]>) -> Result<usize> {
        8usize
            .checked_add(Self::encoded_region_len_for(region)?)
            .ok_or_else(|| Error::InvalidFormat("reference image length overflow".into()))
    }

    fn encoded_region_len_for(region: Option<&[u8]>) -> Result<usize> {
        8usize
            .checked_add(region.map_or(0, <[u8]>::len))
            .ok_or_else(|| Error::InvalidFormat("reference region length overflow".into()))
    }

    fn reserve_reference_image(out: &mut Vec<u8>, region: Option<&[u8]>) -> Result<()> {
        let encoded_len = Self::encoded_len_for(region)?;
        out.try_reserve_exact(encoded_len).map_err(|err| {
            Error::InvalidFormat(format!("reference image allocation failed: {err}"))
        })
    }
}

impl<'a> ReferenceRef<'a> {
    /// Create a borrowed object-reference view.
    pub fn object(object_token: u64, file_name: Option<&'a str>) -> Self {
        Self {
            object_token,
            region: None,
            file_name,
            loc_id: None,
        }
    }

    /// Create a borrowed region-reference view.
    pub fn region(object_token: u64, region: &'a [u8], file_name: Option<&'a str>) -> Self {
        Self {
            object_token,
            region: Some(region),
            file_name,
            loc_id: None,
        }
    }

    /// Return this view with an associated location id.
    pub fn with_loc_id(mut self, loc_id: u64) -> Self {
        self.loc_id = Some(loc_id);
        self
    }

    /// Return object token.
    pub fn object_token(&self) -> u64 {
        self.object_token
    }

    /// Return region bytes.
    pub fn region_slice(&self) -> Option<&'a [u8]> {
        self.region
    }

    /// Return file name.
    pub fn file_name(&self) -> Option<&'a str> {
        self.file_name
    }

    /// Return associated location id.
    pub fn loc_id(&self) -> Option<u64> {
        self.loc_id
    }

    /// Return the encoded reference length.
    pub fn encoded_len(&self) -> Result<usize> {
        Reference::encoded_len_for(self.region)
    }

    /// Encode this reference view into a caller-provided buffer.
    pub fn encode_into(&self, out: &mut Vec<u8>) -> Result<()> {
        Reference::reserve_reference_image(out, self.region)?;
        Reference::encode_obj_token_into(self.object_token, out);
        Reference::encode_region_into(self.region, out)
    }

    /// Encode this reference view into a caller-provided fixed buffer.
    pub fn encode_into_slice(&self, out: &mut [u8]) -> Result<usize> {
        let encoded_len = self.encoded_len()?;
        if out.len() < encoded_len {
            return Err(Error::InvalidFormat(
                "reference image destination is too small".into(),
            ));
        }
        let mut written = Reference::encode_obj_token_slice(self.object_token, out)?;
        written += Reference::encode_region_slice(self.region, &mut out[written..])?;
        Ok(written)
    }
}

/// Bounds-checked subslice `data[pos..pos+len]`, surfacing `context` in errors.
fn checked_window<'a>(data: &'a [u8], pos: usize, len: usize, context: &str) -> Result<&'a [u8]> {
    let end = pos
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))?;
    data.get(pos..end)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))
}

/// Read a little-endian `u64` from `data` at `pos`, surfacing `context` in errors.
fn read_u64_le_at(data: &[u8], pos: usize, context: &str) -> Result<u64> {
    let bytes: [u8; 8] = checked_window(data, pos, 8, context)?
        .try_into()
        .map_err(|_| Error::InvalidFormat(format!("{context} is truncated")))?;
    Ok(u64::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::Reference;

    #[test]
    fn reference_aliases_roundtrip() {
        assert!(Reference::init_package());
        let mut token = String::new();
        Reference::print_token_into(42, &mut token).unwrap();
        assert_eq!(token, "0x2a");
        let mut r = Reference::create_region(7, vec![1, 2, 3], Some("a.h5".into()));
        assert_eq!(r.get_obj_token(), 7);
        r.set_obj_token(8);
        r.set_loc_id(9);
        assert_eq!(r.get_loc_id(), Some(9));
        assert_eq!(r.reopen_file(), Some("a.h5"));
        assert_eq!(r.get_file_name(), Some("a.h5"));
        assert_eq!(r.get_region(), Some([1, 2, 3].as_slice()));
        assert_eq!(r.as_ref().object_token(), 8);
        assert_eq!(r.as_ref().region_slice(), Some([1, 2, 3].as_slice()));
        assert_eq!(r.as_ref().file_name(), Some("a.h5"));
        assert_eq!(r.as_ref().loc_id(), Some(9));
        assert!(r.equal(&r.clone()));

        let mut encoded = Vec::new();
        r.encode_into(&mut encoded).unwrap();
        assert_eq!(encoded.len(), r.encoded_len().unwrap());
        let decoded = Reference::decode_ref(&encoded).unwrap();
        assert_eq!(decoded.object_token(), 8);
        assert_eq!(decoded.region_slice(), Some([1, 2, 3].as_slice()));

        let borrowed = Reference::as_ref(&r);
        let mut fixed = [0; 19];
        assert_eq!(
            borrowed.encode_into_slice(&mut fixed).unwrap(),
            encoded.len()
        );
        assert_eq!(fixed.as_slice(), encoded.as_slice());
        let mut stale_fixed = [0x5a; 18];
        assert!(borrowed.encode_into_slice(&mut stale_fixed).is_err());
        assert_eq!(stale_fixed, [0x5a; 18]);

        let borrowed_region =
            super::ReferenceRef::region(8, &[1, 2, 3], Some("a.h5")).with_loc_id(9);
        assert_eq!(borrowed_region.file_name(), Some("a.h5"));
        assert_eq!(borrowed_region.loc_id(), Some(9));
        let mut fixed_from_constructor = [0; 19];
        assert_eq!(
            borrowed_region
                .encode_into_slice(&mut fixed_from_constructor)
                .unwrap(),
            encoded.len()
        );
        assert_eq!(fixed_from_constructor.as_slice(), encoded.as_slice());
        assert_eq!(super::ReferenceRef::object(8, None).region_slice(), None);

        let mut encoded_from_ref = Vec::new();
        decoded.encode_into(&mut encoded_from_ref).unwrap();
        assert_eq!(encoded_from_ref, encoded);
        let mut appended_from_ref = b"prefix".to_vec();
        decoded.encode_into(&mut appended_from_ref).unwrap();
        assert_eq!(&appended_from_ref[..6], b"prefix");
        assert_eq!(&appended_from_ref[6..], encoded.as_slice());

        let mut token_payload = Vec::new();
        Reference::encode_obj_token_into(r.get_obj_token(), &mut token_payload);
        assert_eq!(
            Reference::encode_obj_token_array(r.get_obj_token()),
            [8, 0, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(
            Reference::decode_token_obj_compat(&token_payload).unwrap(),
            8
        );

        let mut region_payload = Vec::new();
        r.encode_token_region_compat_into(&mut region_payload)
            .unwrap();
        assert_eq!(
            Reference::decode_token_region_compat_slice(&region_payload).unwrap(),
            [1, 2, 3].as_slice()
        );
        assert_eq!(
            Reference::decode_region_slice(&region_payload).unwrap(),
            Some([1, 2, 3].as_slice())
        );
        let mut stale_region = [0x3c; 10];
        assert!(Reference::encode_region_slice(Some(&[1, 2, 3]), &mut stale_region[..9]).is_err());
        assert_eq!(stale_region, [0x3c; 10]);

        let mut heap_payload = Vec::new();
        r.encode_heap_into(&mut heap_payload).unwrap();
        assert_eq!(heap_payload, encoded);

        let mut appended = b"old".to_vec();
        r.encode_into(&mut appended).unwrap();
        assert_eq!(&appended[..3], b"old");
        assert_eq!(&appended[3..], heap_payload.as_slice());

        assert!(Reference::decode_token_obj_compat(&[0; 7]).is_err());
        assert!(Reference::decode_ref(&[0; 15]).is_err());
        let mut trailing = encoded;
        trailing.push(0);
        assert!(Reference::decode_ref(&trailing).is_err());
        assert!(Reference::decode_token_region_compat_slice(&[2, 0, 0, 0, 0, 0, 0, 0, 1]).is_err());
        assert!(Reference::decode_region_slice(&u64::MAX.to_le_bytes()).is_err());
        let mut declared_over_4g = (4u64 * 1024 * 1024 * 1024 + 1).to_le_bytes().to_vec();
        declared_over_4g.extend_from_slice(&[1, 2, 3]);
        let err = Reference::decode_region_slice(&declared_over_4g).unwrap_err();
        assert!(
            err.to_string()
                .contains("reference region payload has invalid length"),
            "unexpected error: {err}"
        );
        assert_eq!(r.open_attr_api_common(), 8);
        assert_eq!(r.get_region_api(), Some([1, 2, 3].as_slice()));
        Reference::create_object_api(1, None).destroy();
        Reference::create_region_api(1, vec![4], None).destroy();
    }
}
