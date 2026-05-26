use std::io::{Seek, SeekFrom, Write};

use crate::error::{Error, Result};

/// Low-level binary writer for HDF5 files.
///
/// Handles variable-width little-endian encoding of addresses and lengths,
/// sized according to the superblock's `sizeof_addr` and `sizeof_size` fields.
pub struct HdfWriter<W: Write + Seek> {
    inner: W,
    /// Size of file addresses in bytes (from superblock, typically 8).
    sizeof_addr: u8,
    /// Size of file lengths in bytes (from superblock, typically 8).
    sizeof_size: u8,
}

impl<W: Write + Seek> HdfWriter<W> {
    /// Create a new writer with default address/length sizes (8 bytes each).
    pub fn new(inner: W) -> Self {
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

    /// Get the current position in the stream.
    pub fn position(&mut self) -> Result<u64> {
        Ok(self.inner.stream_position()?)
    }

    /// Seek to an absolute position.
    pub fn seek(&mut self, pos: u64) -> Result<u64> {
        Ok(self.inner.seek(SeekFrom::Start(pos))?)
    }

    /// Write raw bytes.
    pub fn write_bytes(&mut self, data: &[u8]) -> Result<()> {
        self.inner.write_all(data)?;
        Ok(())
    }

    /// Write a single byte.
    pub fn write_u8(&mut self, val: u8) -> Result<()> {
        self.inner.write_all(&[val])?;
        Ok(())
    }

    /// Write a little-endian u16.
    pub fn write_u16(&mut self, val: u16) -> Result<()> {
        self.inner.write_all(&val.to_le_bytes())?;
        Ok(())
    }

    /// Write a little-endian u32.
    pub fn write_u32(&mut self, val: u32) -> Result<()> {
        self.inner.write_all(&val.to_le_bytes())?;
        Ok(())
    }

    /// Write a little-endian u64.
    pub fn write_u64(&mut self, val: u64) -> Result<()> {
        self.inner.write_all(&val.to_le_bytes())?;
        Ok(())
    }

    /// Write a little-endian i32.
    pub fn write_i32(&mut self, val: i32) -> Result<()> {
        self.inner.write_all(&val.to_le_bytes())?;
        Ok(())
    }

    /// Write a variable-width unsigned integer (1-8 bytes, little-endian).
    pub fn write_uint(&mut self, val: u64, size: u8) -> Result<()> {
        ensure_uint_fits(val, size)?;
        match size {
            1 => self.write_u8(val as u8),
            2 => self.write_u16(val as u16),
            4 => self.write_u32(val as u32),
            8 => self.write_u64(val),
            3 | 5..=7 => self.write_var_uint(val, size),
            0 | 9..=u8::MAX => Err(Error::InvalidFormat(format!(
                "unsupported integer size: {size}"
            ))),
        }
    }

    /// Write a file address (variable width based on sizeof_addr).
    pub fn write_addr(&mut self, val: u64) -> Result<()> {
        if val == crate::io::reader::UNDEF_ADDR {
            return self.write_undefined_addr();
        }
        self.write_uint(val, self.sizeof_addr)
    }

    /// Write a file length (variable width based on sizeof_size).
    pub fn write_length(&mut self, val: u64) -> Result<()> {
        self.write_uint(val, self.sizeof_size)
    }

    /// Write a variable-length encoded integer (1-8 bytes).
    pub fn write_var_uint(&mut self, val: u64, nbytes: u8) -> Result<()> {
        ensure_uint_fits(val, nbytes)?;
        for i in 0..nbytes {
            self.write_u8((val >> (i * 8)) as u8)?;
        }
        Ok(())
    }

    fn write_undefined_addr(&mut self) -> Result<()> {
        if self.sizeof_addr == 0 || self.sizeof_addr > 8 {
            return Err(Error::InvalidFormat(format!(
                "unsupported integer size: {}",
                self.sizeof_addr
            )));
        }
        for _ in 0..self.sizeof_addr {
            self.write_u8(0xff)?;
        }
        Ok(())
    }

    /// Write `n` zero bytes (padding).
    pub fn write_zeros(&mut self, n: usize) -> Result<()> {
        const ZERO_BLOCK: [u8; 8192] = [0; 8192];
        let mut remaining = n;
        while remaining > 0 {
            let chunk = remaining.min(ZERO_BLOCK.len());
            self.inner.write_all(&ZERO_BLOCK[..chunk])?;
            remaining -= chunk;
        }
        Ok(())
    }

    /// Write a checksum (4-byte little-endian).
    pub fn write_checksum(&mut self, val: u32) -> Result<()> {
        self.write_u32(val)
    }

    /// Flush the underlying writer.
    pub fn flush(&mut self) -> Result<()> {
        self.inner.flush()?;
        Ok(())
    }

    /// Get a mutable reference to the underlying writer.
    pub fn inner_mut(&mut self) -> &mut W {
        &mut self.inner
    }

    /// Get a reference to the underlying writer.
    pub fn inner(&self) -> &W {
        &self.inner
    }
}

fn ensure_uint_fits(val: u64, size: u8) -> Result<()> {
    let max = match size {
        1..=7 => (1u64 << (u32::from(size) * 8)) - 1,
        8 => u64::MAX,
        0 | 9..=u8::MAX => {
            return Err(Error::InvalidFormat(format!(
                "unsupported integer size: {size}"
            )));
        }
    };
    if val > max {
        return Err(Error::InvalidFormat(format!(
            "integer value {val} does not fit in {size} bytes"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_write_integers() {
        let mut buf = Cursor::new(Vec::new());
        let mut w = HdfWriter::new(&mut buf);

        w.write_u8(0x42).unwrap();
        w.write_u16(0x1234).unwrap();
        w.write_u32(0x12345678).unwrap();
        w.write_u64(0x1234567890ABCDEF).unwrap();

        let data = buf.into_inner();
        assert_eq!(
            data,
            vec![
                0x42, 0x34, 0x12, 0x78, 0x56, 0x34, 0x12, 0xEF, 0xCD, 0xAB, 0x90, 0x78, 0x56, 0x34,
                0x12,
            ]
        );
    }

    #[test]
    fn test_write_addr_4byte() {
        let mut buf = Cursor::new(Vec::new());
        let mut w = HdfWriter::new(&mut buf);
        w.set_sizeof_addr(4);
        w.write_addr(0x12345678).unwrap();

        let data = buf.into_inner();
        assert_eq!(data, vec![0x78, 0x56, 0x34, 0x12]);
    }

    #[test]
    fn test_roundtrip_var_uint() {
        use crate::io::reader::HdfReader;

        let mut buf = Cursor::new(Vec::new());
        {
            let mut w = HdfWriter::new(&mut buf);
            w.write_var_uint(0xABCDEF, 5).unwrap();
        }
        buf.set_position(0);
        let mut r = HdfReader::new(&mut buf);
        assert_eq!(r.read_var_uint(5).unwrap(), 0xABCDEF);
    }

    #[test]
    fn test_write_uint_accepts_three_byte_width() {
        let mut buf = Cursor::new(Vec::new());
        let mut w = HdfWriter::new(&mut buf);
        w.write_uint(0xabcdef, 3).unwrap();

        assert_eq!(buf.into_inner(), vec![0xef, 0xcd, 0xab]);
    }

    #[test]
    fn write_uint_rejects_truncating_values_without_partial_output() {
        let mut buf = Cursor::new(vec![0xaa]);
        buf.set_position(1);
        let mut w = HdfWriter::new(&mut buf);

        let err = w.write_uint(0x1_000000, 3).unwrap_err();

        assert!(err.to_string().contains("does not fit in 3 bytes"));
        assert_eq!(buf.into_inner(), vec![0xaa]);
    }

    #[test]
    fn write_var_uint_rejects_truncating_values_without_partial_output() {
        let mut buf = Cursor::new(vec![0xaa]);
        buf.set_position(1);
        let mut w = HdfWriter::new(&mut buf);

        let err = w.write_var_uint(0x1_0000, 2).unwrap_err();

        assert!(err.to_string().contains("does not fit in 2 bytes"));
        assert_eq!(buf.into_inner(), vec![0xaa]);
    }

    #[test]
    fn write_addr_encodes_undefined_sentinel_at_configured_width() {
        let mut buf = Cursor::new(Vec::new());
        let mut w = HdfWriter::new(&mut buf);
        w.set_sizeof_addr(4);

        w.write_addr(crate::io::reader::UNDEF_ADDR).unwrap();

        assert_eq!(buf.into_inner(), vec![0xff, 0xff, 0xff, 0xff]);
    }

    #[test]
    fn test_write_zeros_large_padding() {
        let mut buf = Cursor::new(Vec::new());
        let mut w = HdfWriter::new(&mut buf);
        w.write_u8(0xaa).unwrap();
        w.write_zeros(9000).unwrap();
        w.write_u8(0xbb).unwrap();

        let data = buf.into_inner();
        assert_eq!(data.len(), 9002);
        assert_eq!(data[0], 0xaa);
        assert!(data[1..9001].iter().all(|&byte| byte == 0));
        assert_eq!(data[9001], 0xbb);
    }
}
