use std::io::{Read, Seek, SeekFrom};

use crate::error::{Error, Result};

/// Low-level binary reader for HDF5 files.
///
/// Handles variable-width little-endian encoding of addresses and lengths,
/// sized according to the superblock's `sizeof_addr` and `sizeof_size` fields.
pub struct HdfReader<R: Read + Seek> {
    inner: R,
    /// Size of file addresses in bytes (from superblock, typically 8).
    sizeof_addr: u8,
    /// Size of file lengths in bytes (from superblock, typically 8).
    sizeof_size: u8,
    /// Physical file offset of logical HDF5 address zero.
    base_addr: u64,
}

impl<R: Read + Seek> HdfReader<R> {
    /// Create a new reader with default address/length sizes (8 bytes each).
    /// These will be updated after reading the superblock.
    pub fn new(inner: R) -> Self {
        Self {
            inner,
            sizeof_addr: 8,
            sizeof_size: 8,
            base_addr: 0,
        }
    }

    /// Set the address size (from superblock).
    pub fn set_sizeof_addr(&mut self, size: u8) {
        self.sizeof_addr = size;
    }

    /// Set the length size (from superblock).
    pub fn set_sizeof_size(&mut self, size: u8) {
        self.sizeof_size = size;
    }

    pub fn sizeof_addr(&self) -> u8 {
        self.sizeof_addr
    }

    pub fn sizeof_size(&self) -> u8 {
        self.sizeof_size
    }

    /// Set the physical file offset corresponding to logical HDF5 address zero.
    pub fn set_base_addr(&mut self, base_addr: u64) {
        self.base_addr = base_addr;
    }

    /// Return the physical file offset corresponding to logical HDF5 address zero.
    pub fn base_addr(&self) -> u64 {
        self.base_addr
    }

    /// Borrow the wrapped reader.
    pub fn get_ref(&self) -> &R {
        &self.inner
    }

    /// Get the current logical HDF5 position in the stream.
    pub fn position(&mut self) -> Result<u64> {
        let physical = self.inner.stream_position()?;
        physical.checked_sub(self.base_addr).ok_or_else(|| {
            Error::InvalidFormat("physical stream position is before HDF5 base address".into())
        })
    }

    /// Get the current physical byte position in the underlying stream.
    pub fn position_physical(&mut self) -> Result<u64> {
        Ok(self.inner.stream_position()?)
    }

    /// Get the logical HDF5 stream length while preserving the current position.
    pub fn len(&mut self) -> Result<u64> {
        let pos = self.inner.stream_position()?;
        let len = self.inner.seek(SeekFrom::End(0))?;
        self.inner.seek(SeekFrom::Start(pos))?;
        len.checked_sub(self.base_addr).ok_or_else(|| {
            Error::InvalidFormat("file length is smaller than HDF5 base address".into())
        })
    }

    /// Get the physical stream length while preserving the current position.
    pub fn len_physical(&mut self) -> Result<u64> {
        let pos = self.inner.stream_position()?;
        let len = self.inner.seek(SeekFrom::End(0))?;
        self.inner.seek(SeekFrom::Start(pos))?;
        Ok(len)
    }

    /// Seek to a logical HDF5 address.
    pub fn seek(&mut self, pos: u64) -> Result<u64> {
        let physical = self
            .base_addr
            .checked_add(pos)
            .ok_or_else(|| Error::InvalidFormat("logical seek address overflow".into()))?;
        self.inner.seek(SeekFrom::Start(physical))?;
        Ok(pos)
    }

    /// Seek to a physical byte offset in the underlying stream.
    pub fn seek_physical(&mut self, pos: u64) -> Result<u64> {
        Ok(self.inner.seek(SeekFrom::Start(pos))?)
    }

    /// Seek relative to current position.
    pub fn seek_relative(&mut self, offset: i64) -> Result<u64> {
        self.inner.seek(SeekFrom::Current(offset))?;
        self.position()
    }

    /// Read bytes into a provided buffer.
    pub fn read_bytes_into(&mut self, buf: &mut [u8]) -> Result<()> {
        let mut scratch = vec![0; buf.len()];
        self.inner.read_exact(&mut scratch)?;
        buf.copy_from_slice(&scratch);
        Ok(())
    }

    /// Read exactly `n` bytes into a provided buffer.
    pub fn read_exact(&mut self, buf: &mut [u8]) -> Result<()> {
        self.read_bytes_into(buf)
    }

    /// Read a single byte.
    pub fn read_u8(&mut self) -> Result<u8> {
        let mut buf = [0u8; 1];
        self.inner.read_exact(&mut buf)?;
        Ok(buf[0])
    }

    /// Read a little-endian u16.
    pub fn read_u16(&mut self) -> Result<u16> {
        let mut buf = [0u8; 2];
        self.inner.read_exact(&mut buf)?;
        Ok(u16::from_le_bytes(buf))
    }

    /// Read a little-endian u32.
    pub fn read_u32(&mut self) -> Result<u32> {
        let mut buf = [0u8; 4];
        self.inner.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    /// Read a little-endian u64.
    pub fn read_u64(&mut self) -> Result<u64> {
        let mut buf = [0u8; 8];
        self.inner.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    /// Read a little-endian i32.
    pub fn read_i32(&mut self) -> Result<i32> {
        let mut buf = [0u8; 4];
        self.inner.read_exact(&mut buf)?;
        Ok(i32::from_le_bytes(buf))
    }

    /// Read a variable-width unsigned integer (1-8 bytes, little-endian).
    pub fn read_uint(&mut self, size: u8) -> Result<u64> {
        match size {
            1 => self.read_u8().map(u64::from),
            2 => self.read_u16().map(u64::from),
            4 => self.read_u32().map(u64::from),
            8 => self.read_u64(),
            3 | 5..=7 => self.read_var_uint(size),
            0 | 9..=u8::MAX => Err(Error::InvalidFormat(format!(
                "unsupported integer size: {size}"
            ))),
        }
    }

    /// Read a file address (variable width based on sizeof_addr).
    pub fn read_addr(&mut self) -> Result<u64> {
        let addr = self.read_uint(self.sizeof_addr)?;
        if self.sizeof_addr < 8 && addr == max_uint_for_size(self.sizeof_addr)? {
            Ok(UNDEF_ADDR)
        } else {
            Ok(addr)
        }
    }

    /// Read a file length (variable width based on sizeof_size).
    pub fn read_length(&mut self) -> Result<u64> {
        self.read_uint(self.sizeof_size)
    }

    /// Read a variable-length encoded integer (1-8 bytes).
    /// Used for chunk dimensions in v4+ layout messages.
    pub fn read_var_uint(&mut self, nbytes: u8) -> Result<u64> {
        if nbytes == 0 || nbytes > 8 {
            return Err(Error::InvalidFormat(format!(
                "invalid variable uint size: {nbytes}"
            )));
        }
        let mut val: u64 = 0;
        for i in 0..nbytes {
            let byte = self.read_u8()? as u64;
            val |= byte << (i * 8);
        }
        Ok(val)
    }

    /// Skip `n` bytes.
    pub fn skip(&mut self, n: u64) -> Result<()> {
        let delta = i64::try_from(n)
            .map_err(|_| Error::InvalidFormat("skip distance exceeds i64::MAX".into()))?;
        self.inner.seek(SeekFrom::Current(delta))?;
        Ok(())
    }

    /// Read a checksum (4-byte little-endian).
    pub fn read_checksum(&mut self) -> Result<u32> {
        self.read_u32()
    }

    /// Get a mutable reference to the underlying reader.
    pub fn inner_mut(&mut self) -> &mut R {
        &mut self.inner
    }

    /// Get a reference to the underlying reader.
    pub fn inner(&self) -> &R {
        &self.inner
    }
}

/// Undefined address sentinel value.
pub const UNDEF_ADDR: u64 = u64::MAX;

/// Check if an address is the undefined sentinel.
pub fn is_undef_addr(addr: u64) -> bool {
    addr == UNDEF_ADDR
}

fn max_uint_for_size(size: u8) -> Result<u64> {
    match size {
        1..=7 => Ok((1u64 << (u32::from(size) * 8)) - 1),
        8 => Ok(u64::MAX),
        _ => Err(Error::InvalidFormat(format!(
            "unsupported integer size: {size}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_read_integers() {
        let data: Vec<u8> = vec![
            0x42, // u8: 0x42
            0x34, 0x12, // u16: 0x1234
            0x78, 0x56, 0x34, 0x12, // u32: 0x12345678
            0xEF, 0xCD, 0xAB, 0x90, 0x78, 0x56, 0x34, 0x12, // u64
        ];
        let mut r = HdfReader::new(Cursor::new(data));

        assert_eq!(r.read_u8().unwrap(), 0x42);
        assert_eq!(r.read_u16().unwrap(), 0x1234);
        assert_eq!(r.read_u32().unwrap(), 0x12345678);
        assert_eq!(r.read_u64().unwrap(), 0x1234567890ABCDEF);
    }

    #[test]
    fn test_read_var_uint() {
        // 3-byte little-endian: 0x03 0x02 0x01 = 0x010203
        let data = vec![0x03, 0x02, 0x01];
        let mut r = HdfReader::new(Cursor::new(data));
        assert_eq!(r.read_var_uint(3).unwrap(), 0x010203);
    }

    #[test]
    fn test_read_uint_accepts_three_byte_width() {
        let data = vec![0xef, 0xcd, 0xab];
        let mut r = HdfReader::new(Cursor::new(data));
        assert_eq!(r.read_uint(3).unwrap(), 0xabcdef);
    }

    #[test]
    fn test_read_bytes_into() {
        let data = vec![1, 2, 3, 4];
        let mut r = HdfReader::new(Cursor::new(data));
        let mut buf = [0u8; 4];

        r.read_bytes_into(&mut buf).unwrap();

        assert_eq!(buf, [1, 2, 3, 4]);
    }

    #[test]
    fn read_bytes_into_preserves_output_on_short_read() {
        let data = vec![1, 2];
        let mut r = HdfReader::new(Cursor::new(data));
        let mut buf = [9u8; 4];

        assert!(r.read_bytes_into(&mut buf).is_err());

        assert_eq!(buf, [9, 9, 9, 9]);
    }

    #[test]
    fn test_read_exact() {
        let data = vec![5, 6, 7];
        let mut r = HdfReader::new(Cursor::new(data));
        let mut buf = [0u8; 2];

        r.read_exact(&mut buf).unwrap();

        assert_eq!(buf, [5, 6]);
        assert_eq!(r.read_u8().unwrap(), 7);
    }

    #[test]
    fn test_read_addr_4byte() {
        let data = vec![0x78, 0x56, 0x34, 0x12];
        let mut r = HdfReader::new(Cursor::new(data));
        r.set_sizeof_addr(4);
        assert_eq!(r.read_addr().unwrap(), 0x12345678);
    }

    #[test]
    fn read_addr_normalizes_width_specific_undefined_sentinel() {
        let data = vec![0xff, 0xff, 0xff, 0xff];
        let mut r = HdfReader::new(Cursor::new(data));
        r.set_sizeof_addr(4);

        assert_eq!(r.read_addr().unwrap(), UNDEF_ADDR);
        assert!(is_undef_addr(UNDEF_ADDR));
    }

    #[test]
    fn skip_rejects_i64_overflow() {
        let data = vec![0u8; 8];
        let mut r = HdfReader::new(Cursor::new(data));
        let err = r.skip(i64::MAX as u64 + 1).unwrap_err();
        assert!(err.to_string().contains("skip distance"));
    }

    #[test]
    fn logical_seek_and_position_are_relative_to_base_addr() {
        let data: Vec<u8> = (0..16).collect();
        let mut r = HdfReader::new(Cursor::new(data));
        r.set_base_addr(8);

        assert_eq!(r.len().unwrap(), 8);
        assert_eq!(r.len_physical().unwrap(), 16);
        assert_eq!(r.seek(2).unwrap(), 2);
        assert_eq!(r.position().unwrap(), 2);
        assert_eq!(r.position_physical().unwrap(), 10);
        assert_eq!(r.read_u8().unwrap(), 10);
        assert_eq!(r.position().unwrap(), 3);

        r.seek_physical(1).unwrap();
        assert!(r.position().is_err());
    }
}
