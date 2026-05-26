use crate::error::{Error, Result};
use crate::io::reader::UNDEF_ADDR;

/// Parsed Link Info message (type 0x0002).
/// Contains pointers to dense link storage structures.
#[derive(Debug, Clone)]
pub struct LinkInfoMessage {
    /// Version of the link info message.
    pub version: u8,
    /// Flags.
    pub flags: u8,
    /// Maximum creation order index (if tracked).
    pub max_creation_index: Option<u64>,
    /// Address of fractal heap for storing link data.
    pub fractal_heap_addr: u64,
    /// Address of v2 B-tree for name index.
    pub name_btree_addr: u64,
    /// Address of v2 B-tree for creation order index (if indexed).
    pub corder_btree_addr: Option<u64>,
}

impl LinkInfoMessage {
    pub fn decode(data: &[u8], sizeof_addr: u8) -> Result<Self> {
        Self::decode_impl(data, sizeof_addr)
    }

    fn decode_impl(data: &[u8], sizeof_addr: u8) -> Result<Self> {
        let mut pos = 0;
        let sa = usize::from(sizeof_addr);

        let version = read_u8(data, &mut pos, "link info message version")?;
        if version != 0 {
            return Err(Error::InvalidFormat(format!(
                "link info message version {version}"
            )));
        }

        let flags = read_u8(data, &mut pos, "link info message flags")?;
        if flags & !0x03 != 0 {
            return Err(Error::InvalidFormat(format!(
                "link info message flags {flags:#x} are invalid"
            )));
        }

        let has_max_crt_order = flags & 0x01 != 0;
        let has_corder_btree = flags & 0x02 != 0;

        let max_creation_index = if has_max_crt_order {
            let val = read_le_u64(data, &mut pos, 8, "link info max creation index")?;
            // libhdf5 decodes this as int64_t and rejects negative values.
            let max_i64 = u64::try_from(i64::MAX).map_err(|_| {
                Error::InvalidFormat("link info max creation index bound is invalid".into())
            })?;
            if val > max_i64 {
                return Err(Error::InvalidFormat(format!(
                    "link info max creation index {val} exceeds supported maximum {}",
                    i64::MAX
                )));
            }
            Some(val)
        } else {
            None
        };

        let fractal_heap_addr = read_addr(data, &mut pos, sa, "link info fractal heap address")?;

        let name_btree_addr = read_addr(data, &mut pos, sa, "link info name btree address")?;

        let corder_btree_addr = if has_corder_btree {
            let addr = read_addr(data, &mut pos, sa, "link info creation order btree address")?;
            Some(addr)
        } else {
            None
        };

        Ok(Self {
            version,
            flags,
            max_creation_index,
            fractal_heap_addr,
            name_btree_addr,
            corder_btree_addr,
        })
    }

    /// Whether this group has dense link storage (fractal heap).
    pub fn has_dense_storage(&self) -> bool {
        self.fractal_heap_addr != UNDEF_ADDR
    }
}

fn ensure_available(data: &[u8], pos: usize, len: usize, context: &str) -> Result<()> {
    let end = pos
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} length overflow")))?;
    if end > data.len() {
        return Err(Error::InvalidFormat(format!("{context} is truncated")));
    }
    Ok(())
}

fn read_u8(data: &[u8], pos: &mut usize, context: &str) -> Result<u8> {
    ensure_available(data, *pos, 1, context)?;
    let value = data[*pos];
    advance_pos(pos, 1, context)?;
    Ok(value)
}

fn read_le_u64(data: &[u8], pos: &mut usize, size: usize, context: &str) -> Result<u64> {
    if !(1..=8).contains(&size) {
        return Err(Error::InvalidFormat(format!(
            "{context} has invalid byte width {size}"
        )));
    }
    let bytes = checked_window(data, *pos, size, context)?;
    let mut val = 0u64;
    for (i, byte) in bytes.iter().enumerate() {
        val |= u64::from(*byte) << (i * 8);
    }
    advance_pos(pos, size, context)?;
    Ok(val)
}

fn read_addr(data: &[u8], pos: &mut usize, size: usize, context: &str) -> Result<u64> {
    let value = read_le_u64(data, pos, size, context)?;
    if size < 8 && value == ((1u64 << (size * 8)) - 1) {
        Ok(UNDEF_ADDR)
    } else {
        Ok(value)
    }
}

fn checked_window<'a>(data: &'a [u8], pos: usize, len: usize, context: &str) -> Result<&'a [u8]> {
    let end = pos
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} length overflow")))?;
    data.get(pos..end)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))
}

fn advance_pos(pos: &mut usize, len: usize, context: &str) -> Result<()> {
    *pos = pos
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn link_info_decode_normalizes_width_specific_undefined_addresses() {
        let mut data = vec![0, 0];
        data.extend_from_slice(&u32::MAX.to_le_bytes());
        data.extend_from_slice(&u32::MAX.to_le_bytes());

        let message = LinkInfoMessage::decode(&data, 4).unwrap();

        assert_eq!(message.fractal_heap_addr, UNDEF_ADDR);
        assert_eq!(message.name_btree_addr, UNDEF_ADDR);
        assert!(!message.has_dense_storage());
    }

    #[test]
    fn link_info_decode_normalizes_creation_order_btree_address() {
        let mut data = vec![0, 0x03];
        data.extend_from_slice(&7u64.to_le_bytes());
        data.extend_from_slice(&u32::MAX.to_le_bytes());
        data.extend_from_slice(&8u32.to_le_bytes());
        data.extend_from_slice(&u32::MAX.to_le_bytes());

        let message = LinkInfoMessage::decode(&data, 4).unwrap();

        assert_eq!(message.max_creation_index, Some(7));
        assert_eq!(message.fractal_heap_addr, UNDEF_ADDR);
        assert_eq!(message.name_btree_addr, 8);
        assert_eq!(message.corder_btree_addr, Some(UNDEF_ADDR));
    }
}
