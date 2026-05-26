use crate::error::{Error, Result};
use crate::io::reader::UNDEF_ADDR;

/// Parsed Attribute Info message (type 0x0015).
#[derive(Debug, Clone)]
pub struct AttributeInfoMessage {
    pub version: u8,
    pub flags: u8,
    pub max_creation_index: Option<u16>,
    pub fractal_heap_addr: u64,
    pub name_btree_addr: u64,
    pub corder_btree_addr: Option<u64>,
}

impl AttributeInfoMessage {
    pub fn decode(data: &[u8], sizeof_addr: u8) -> Result<Self> {
        if data.len() < 2 {
            return Err(Error::InvalidFormat(
                "attribute info message too short".into(),
            ));
        }

        let sa = usize::from(sizeof_addr);
        let mut pos = 0;

        let version = read_u8(data, &mut pos, "attribute info message version")?;
        if version != 0 {
            return Err(Error::InvalidFormat(format!(
                "attribute info version {version}"
            )));
        }

        let flags = read_u8(data, &mut pos, "attribute info message flags")?;
        if flags & !0x03 != 0 {
            return Err(Error::InvalidFormat(format!(
                "attribute info message flags {flags:#x} are invalid"
            )));
        }

        let has_max_crt_order = flags & 0x01 != 0;
        let has_corder_btree = flags & 0x02 != 0;

        let max_creation_index = if has_max_crt_order {
            let val = read_u16_le(data, &mut pos, "attribute info max creation index")?;
            Some(val)
        } else {
            None
        };

        let fractal_heap_addr =
            read_addr(data, &mut pos, sa, "attribute info fractal heap address")?;

        let name_btree_addr = read_addr(data, &mut pos, sa, "attribute info name btree address")?;

        let corder_btree_addr = if has_corder_btree {
            let addr = read_addr(
                data,
                &mut pos,
                sa,
                "attribute info creation order btree address",
            )?;
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

fn read_u16_le(data: &[u8], pos: &mut usize, context: &str) -> Result<u16> {
    ensure_available(data, *pos, 2, context)?;
    let end = checked_add_pos(*pos, 2, context)?;
    let bytes = data
        .get(*pos..end)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))?;
    let value = u16::from_le_bytes(
        bytes
            .try_into()
            .map_err(|_| Error::InvalidFormat(format!("{context} is truncated")))?,
    );
    advance_pos(pos, 2, context)?;
    Ok(value)
}

fn read_le_u64(data: &[u8], pos: &mut usize, size: usize, context: &str) -> Result<u64> {
    if !(1..=8).contains(&size) {
        return Err(Error::InvalidFormat(format!(
            "{context} has invalid byte width {size}"
        )));
    }
    ensure_available(data, *pos, size, context)?;
    let end = checked_add_pos(*pos, size, context)?;
    let mut val = 0u64;
    for (i, byte) in data[*pos..end].iter().enumerate() {
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

fn checked_add_pos(pos: usize, len: usize, context: &str) -> Result<usize> {
    pos.checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))
}

fn advance_pos(pos: &mut usize, len: usize, context: &str) -> Result<()> {
    *pos = checked_add_pos(*pos, len, context)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attribute_info_decode_normalizes_width_specific_undefined_addresses() {
        let mut data = vec![0, 0];
        data.extend_from_slice(&u32::MAX.to_le_bytes());
        data.extend_from_slice(&u32::MAX.to_le_bytes());

        let message = AttributeInfoMessage::decode(&data, 4).unwrap();

        assert_eq!(message.fractal_heap_addr, UNDEF_ADDR);
        assert_eq!(message.name_btree_addr, UNDEF_ADDR);
        assert!(!message.has_dense_storage());
    }

    #[test]
    fn attribute_info_decode_normalizes_creation_order_btree_address() {
        let mut data = vec![0, 0x03];
        data.extend_from_slice(&7u16.to_le_bytes());
        data.extend_from_slice(&u32::MAX.to_le_bytes());
        data.extend_from_slice(&8u32.to_le_bytes());
        data.extend_from_slice(&u32::MAX.to_le_bytes());

        let message = AttributeInfoMessage::decode(&data, 4).unwrap();

        assert_eq!(message.max_creation_index, Some(7));
        assert_eq!(message.fractal_heap_addr, UNDEF_ADDR);
        assert_eq!(message.name_btree_addr, 8);
        assert_eq!(message.corder_btree_addr, Some(UNDEF_ADDR));
    }
}
