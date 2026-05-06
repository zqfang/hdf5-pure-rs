use crate::error::{Error, Result};
use crate::io::reader::UNDEF_ADDR;

/// Parsed Symbol Table message (type 0x0011).
/// Points to a v1 B-tree and local heap for group membership.
#[derive(Debug, Clone)]
pub struct SymbolTableMessage {
    /// Address of the v1 B-tree for group nodes.
    pub btree_addr: u64,
    /// Address of the local heap for link names.
    pub name_heap_addr: u64,
}

impl SymbolTableMessage {
    /// Decode from raw message bytes. `sizeof_addr` determines address width.
    pub fn decode(data: &[u8], sizeof_addr: u8) -> Result<Self> {
        let sa = usize::from(sizeof_addr);
        if !(1..=8).contains(&sa) {
            return Err(Error::InvalidFormat(format!(
                "symbol table address size {sa} is invalid"
            )));
        }
        let expected_len = sa.checked_mul(2).ok_or_else(|| {
            Error::InvalidFormat("symbol table message address size overflow".into())
        })?;
        if data.len() < expected_len {
            return Err(Error::InvalidFormat(
                "symbol table message too short".into(),
            ));
        }
        let btree_addr = read_addr(data, 0, sa)?;
        let name_heap_addr = read_addr(data, sa, sa)?;
        if is_undefined_addr(btree_addr, sa) {
            return Err(Error::InvalidFormat(
                "symbol table B-tree address is undefined".into(),
            ));
        }
        if is_undefined_addr(name_heap_addr, sa) {
            return Err(Error::InvalidFormat(
                "symbol table local heap address is undefined".into(),
            ));
        }

        Ok(Self {
            btree_addr,
            name_heap_addr,
        })
    }
}

fn read_addr(data: &[u8], offset: usize, size: usize) -> Result<u64> {
    if size > 8 {
        return Err(Error::InvalidFormat(
            "symbol table address payload is wider than u64".into(),
        ));
    }
    let bytes = checked_window(data, offset, size, "symbol table address")?;
    let mut val = 0u64;
    for (i, byte) in bytes.iter().enumerate() {
        val |= u64::from(*byte) << (i * 8);
    }
    Ok(val)
}

fn is_undefined_addr(addr: u64, size: usize) -> bool {
    if size >= 8 {
        addr == UNDEF_ADDR
    } else {
        let mask = (1u64 << (size * 8)) - 1;
        addr == mask
    }
}

fn checked_window<'a>(
    data: &'a [u8],
    offset: usize,
    len: usize,
    context: &str,
) -> Result<&'a [u8]> {
    let end = offset
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))?;
    data.get(offset..end)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbol_table_window_rejects_offset_overflow() {
        let err = checked_window(&[], usize::MAX, 1, "symbol table test").unwrap_err();
        assert!(
            err.to_string()
                .contains("symbol table test offset overflow"),
            "unexpected error: {err}"
        );
    }
}
