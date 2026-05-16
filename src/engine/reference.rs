use crate::error::{Error, Result};

/// Pure Rust object/region reference token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reference {
    object_token: u64,
    region: Option<Vec<u8>>,
    file_name: Option<String>,
    loc_id: Option<u64>,
}

const MAX_REFERENCE_REGION_BYTES: usize = 4 * 1024 * 1024 * 1024;

impl Reference {
    /// Render an object token.
    pub fn print_token(token: u64) -> String {
        format!("{token:#x}")
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

    /// Copy a reference.
    pub fn copy(&self) -> Self {
        self.clone()
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
        self.region.as_deref()
    }

    /// Return file name.
    pub fn get_file_name(&self) -> Option<&str> {
        self.file_name.as_deref()
    }

    /// Encode a reference.
    pub fn encode(&self) -> Result<Vec<u8>> {
        let capacity = 16usize
            .checked_add(self.region.as_ref().map_or(0, Vec::len))
            .ok_or_else(|| Error::InvalidFormat("reference image length overflow".into()))?;
        let mut out = Vec::with_capacity(capacity);
        Self::encode_obj_token_into(self.object_token, &mut out);
        Self::encode_region_into(self.region.as_deref(), &mut out)?;
        Ok(out)
    }

    /// Decode a full encoded reference image.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 16 {
            return Err(Error::InvalidFormat(
                "reference image is shorter than object token and region length".into(),
            ));
        }
        let object_token =
            Self::decode_token_obj_compat(checked_window(bytes, 0, 8, "reference object token")?)?;
        let region = Self::decode_region(checked_window(
            bytes,
            8,
            bytes.len() - 8,
            "reference region payload",
        )?)?;
        Ok(Self {
            object_token,
            region,
            file_name: None,
            loc_id: None,
        })
    }

    /// Encode an object token.
    pub fn encode_obj_token(&self) -> Vec<u8> {
        let mut out = Vec::new();
        Self::encode_obj_token_into(self.object_token, &mut out);
        out
    }

    /// Encode a region payload.
    pub fn encode_region(&self) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        Self::encode_region_into(self.region.as_deref(), &mut out)?;
        Ok(out)
    }

    /// Decode a region payload.
    pub fn decode_region(bytes: &[u8]) -> Result<Option<Vec<u8>>> {
        if bytes.is_empty() {
            return Ok(None);
        } else {
            let len_u64 = read_u64_le_at(bytes, 0, "reference region length")?;
            let len = usize::try_from(len_u64).map_err(|_| {
                Error::InvalidFormat("reference region length exceeds usize".into())
            })?;
            if len > MAX_REFERENCE_REGION_BYTES {
                return Err(Error::InvalidFormat(format!(
                    "reference region length {len} exceeds supported maximum {MAX_REFERENCE_REGION_BYTES}"
                )));
            }
            let end = 8usize
                .checked_add(len)
                .ok_or_else(|| Error::InvalidFormat("reference region length overflow".into()))?;
            if bytes.len() != end {
                return Err(Error::InvalidFormat(
                    "reference region payload has invalid length".into(),
                ));
            }
            Ok(Some(bytes[8..end].to_vec()))
        }
    }

    /// Encode a heap reference payload.
    pub fn encode_heap(&self) -> Result<Vec<u8>> {
        self.encode()
    }

    /// Encode an object-token compatibility payload.
    pub fn encode_token_obj_compat(&self) -> Vec<u8> {
        self.encode_obj_token()
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

    /// Decode a region-token compatibility payload.
    pub fn decode_token_region_compat(bytes: &[u8]) -> Result<Vec<u8>> {
        Self::decode_region(bytes)?
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

    /// Encode a region-token compatibility payload.
    pub fn encode_token_region_compat(&self) -> Result<Vec<u8>> {
        self.encode_region()
    }

    /// Public get-region alias.
    pub fn get_region_api(&self) -> Option<&[u8]> {
        self.get_region()
    }

    fn encode_obj_token_into(token: u64, out: &mut Vec<u8>) {
        out.extend_from_slice(&token.to_le_bytes());
    }

    fn encode_region_into(region: Option<&[u8]>, out: &mut Vec<u8>) -> Result<()> {
        let len = region.map_or(Ok(0u64), |region| {
            if region.len() > MAX_REFERENCE_REGION_BYTES {
                return Err(Error::InvalidFormat(format!(
                    "reference region length {} exceeds supported maximum {MAX_REFERENCE_REGION_BYTES}",
                    region.len()
                )));
            }
            u64::try_from(region.len())
                .map_err(|_| Error::InvalidFormat("reference region length exceeds u64".into()))
        })?;
        out.extend_from_slice(&len.to_le_bytes());
        if let Some(region) = region {
            out.extend_from_slice(region);
        }
        Ok(())
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
        assert_eq!(Reference::print_token(42), "0x2a");
        let mut r = Reference::create_region(7, vec![1, 2, 3], Some("a.h5".into()));
        assert_eq!(r.get_obj_token(), 7);
        r.set_obj_token(8);
        r.set_loc_id(9);
        assert_eq!(r.get_loc_id(), Some(9));
        assert_eq!(r.reopen_file(), Some("a.h5"));
        assert_eq!(r.get_file_name(), Some("a.h5"));
        assert_eq!(r.get_region(), Some([1, 2, 3].as_slice()));
        assert!(r.equal(&r.copy()));
        let decoded = Reference::decode(&r.encode().unwrap()).unwrap();
        assert_eq!(decoded.get_obj_token(), 8);
        assert_eq!(decoded.get_region(), Some([1, 2, 3].as_slice()));
        assert_eq!(
            Reference::decode_token_obj_compat(&r.encode_obj_token()).unwrap(),
            8
        );
        assert_eq!(
            Reference::decode_token_region_compat(&r.encode_token_region_compat().unwrap())
                .unwrap(),
            vec![1, 2, 3]
        );
        assert_eq!(
            Reference::decode_region(&r.encode_region().unwrap()).unwrap(),
            Some(vec![1, 2, 3])
        );
        assert!(Reference::decode_token_obj_compat(&[0; 7]).is_err());
        assert!(Reference::decode(&[0; 15]).is_err());
        let mut trailing = r.encode().unwrap();
        trailing.push(0);
        assert!(Reference::decode(&trailing).is_err());
        assert!(Reference::decode_token_region_compat(&[2, 0, 0, 0, 0, 0, 0, 0, 1]).is_err());
        assert!(Reference::decode_region(&u64::MAX.to_le_bytes()).is_err());
        assert_eq!(r.open_attr_api_common(), 8);
        assert_eq!(r.get_region_api(), Some([1, 2, 3].as_slice()));
        Reference::create_object_api(1, None).destroy();
        Reference::create_region_api(1, vec![4], None).destroy();
    }
}
