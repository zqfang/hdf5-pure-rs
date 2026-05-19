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
}

impl<R: Read + Seek> HdfReader<R> {
    /// Create a new reader with default address/length sizes (8 bytes each).
    /// These will be updated after reading the superblock.
    pub fn new(inner: R) -> Self {
        Self {
            inner,
            sizeof_addr: 8,
            sizeof_size: 8,
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

    /// Borrow the wrapped reader.
    pub fn get_ref(&self) -> &R {
        &self.inner
    }

    /// Get the current position in the stream.
    pub fn position(&mut self) -> Result<u64> {
        Ok(self.inner.stream_position()?)
    }

    /// Get the stream length while preserving the current position.
    pub fn len(&mut self) -> Result<u64> {
        let pos = self.inner.stream_position()?;
        let len = self.inner.seek(SeekFrom::End(0))?;
        self.inner.seek(SeekFrom::Start(pos))?;
        Ok(len)
    }

    /// Seek to an absolute position.
    pub fn seek(&mut self, pos: u64) -> Result<u64> {
        Ok(self.inner.seek(SeekFrom::Start(pos))?)
    }

    /// Seek relative to current position.
    pub fn seek_relative(&mut self, offset: i64) -> Result<u64> {
        Ok(self.inner.seek(SeekFrom::Current(offset))?)
    }

    /// Read bytes into a provided buffer.
    pub fn read_bytes_into(&mut self, buf: &mut [u8]) -> Result<()> {
        self.inner.read_exact(buf)?;
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

    /// Read a variable-width unsigned integer (1, 2, 4, or 8 bytes, little-endian).
    pub fn read_uint(&mut self, size: u8) -> Result<u64> {
        match size {
            1 => self.read_u8().map(u64::from),
            2 => self.read_u16().map(u64::from),
            4 => self.read_u32().map(u64::from),
            8 => self.read_u64(),
            _ => Err(Error::InvalidFormat(format!(
                "unsupported integer size: {size}"
            ))),
        }
    }

    /// Read a file address (variable width based on sizeof_addr).
    pub fn read_addr(&mut self) -> Result<u64> {
        self.read_uint(self.sizeof_addr)
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
    fn test_read_bytes_into() {
        let data = vec![1, 2, 3, 4];
        let mut r = HdfReader::new(Cursor::new(data));
        let mut buf = [0u8; 4];

        r.read_bytes_into(&mut buf).unwrap();

        assert_eq!(buf, [1, 2, 3, 4]);
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
    fn skip_rejects_i64_overflow() {
        let data = vec![0u8; 8];
        let mut r = HdfReader::new(Cursor::new(data));
        let err = r.skip(i64::MAX as u64 + 1).unwrap_err();
        assert!(err.to_string().contains("skip distance"));
    }
}
