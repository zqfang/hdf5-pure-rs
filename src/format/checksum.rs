/// Jenkins lookup3 hash function, as used by HDF5 for metadata checksums.
///
/// This is a direct translation of Bob Jenkins' lookup3 hashlittle() function
/// from the HDF5 C source (H5checksum.c). It must produce bit-identical output.
#[inline]
fn rot(x: u32, k: u32) -> u32 {
    (x << k) ^ (x >> (32 - k))
}

#[inline]
#[allow(clippy::many_single_char_names)]
fn mix(a: &mut u32, b: &mut u32, c: &mut u32) {
    *a = a.wrapping_sub(*c);
    *a ^= rot(*c, 4);
    *c = c.wrapping_add(*b);
    *b = b.wrapping_sub(*a);
    *b ^= rot(*a, 6);
    *a = a.wrapping_add(*c);
    *c = c.wrapping_sub(*b);
    *c ^= rot(*b, 8);
    *b = b.wrapping_add(*a);
    *a = a.wrapping_sub(*c);
    *a ^= rot(*c, 16);
    *c = c.wrapping_add(*b);
    *b = b.wrapping_sub(*a);
    *b ^= rot(*a, 19);
    *a = a.wrapping_add(*c);
    *c = c.wrapping_sub(*b);
    *c ^= rot(*b, 4);
    *b = b.wrapping_add(*a);
}

#[inline]
#[allow(clippy::many_single_char_names)]
fn final_mix(a: &mut u32, b: &mut u32, c: &mut u32) {
    *c ^= *b;
    *c = c.wrapping_sub(rot(*b, 14));
    *a ^= *c;
    *a = a.wrapping_sub(rot(*c, 11));
    *b ^= *a;
    *b = b.wrapping_sub(rot(*a, 25));
    *c ^= *b;
    *c = c.wrapping_sub(rot(*b, 16));
    *a ^= *c;
    *a = a.wrapping_sub(rot(*c, 4));
    *b ^= *a;
    *b = b.wrapping_sub(rot(*a, 14));
    *c ^= *b;
    *c = c.wrapping_sub(rot(*b, 24));
}

/// Compute the Jenkins lookup3 checksum used by HDF5 for metadata.
///
/// This matches `H5_checksum_lookup3()` / `H5_checksum_metadata()` from the C library.
#[allow(clippy::many_single_char_names)]
pub fn checksum_lookup3(key: &[u8], initval: u32) -> u32 {
    let mut length = key.len();
    let mut k = key;

    let mut a: u32 = 0xdeadbeef_u32
        .wrapping_add(length as u32)
        .wrapping_add(initval);
    let mut b: u32 = a;
    let mut c: u32 = a;

    // Process all but the last block (12 bytes at a time)
    while length > 12 {
        a = a.wrapping_add(k[0] as u32);
        a = a.wrapping_add((k[1] as u32) << 8);
        a = a.wrapping_add((k[2] as u32) << 16);
        a = a.wrapping_add((k[3] as u32) << 24);
        b = b.wrapping_add(k[4] as u32);
        b = b.wrapping_add((k[5] as u32) << 8);
        b = b.wrapping_add((k[6] as u32) << 16);
        b = b.wrapping_add((k[7] as u32) << 24);
        c = c.wrapping_add(k[8] as u32);
        c = c.wrapping_add((k[9] as u32) << 8);
        c = c.wrapping_add((k[10] as u32) << 16);
        c = c.wrapping_add((k[11] as u32) << 24);
        mix(&mut a, &mut b, &mut c);
        length -= 12;
        k = &k[12..];
    }

    // Last block: affect all 32 bits of (c)
    match length {
        12 => {
            c = c.wrapping_add((k[11] as u32) << 24);
            c = c.wrapping_add((k[10] as u32) << 16);
            c = c.wrapping_add((k[9] as u32) << 8);
            c = c.wrapping_add(k[8] as u32);
            b = b.wrapping_add((k[7] as u32) << 24);
            b = b.wrapping_add((k[6] as u32) << 16);
            b = b.wrapping_add((k[5] as u32) << 8);
            b = b.wrapping_add(k[4] as u32);
            a = a.wrapping_add((k[3] as u32) << 24);
            a = a.wrapping_add((k[2] as u32) << 16);
            a = a.wrapping_add((k[1] as u32) << 8);
            a = a.wrapping_add(k[0] as u32);
        }
        11 => {
            c = c.wrapping_add((k[10] as u32) << 16);
            c = c.wrapping_add((k[9] as u32) << 8);
            c = c.wrapping_add(k[8] as u32);
            b = b.wrapping_add((k[7] as u32) << 24);
            b = b.wrapping_add((k[6] as u32) << 16);
            b = b.wrapping_add((k[5] as u32) << 8);
            b = b.wrapping_add(k[4] as u32);
            a = a.wrapping_add((k[3] as u32) << 24);
            a = a.wrapping_add((k[2] as u32) << 16);
            a = a.wrapping_add((k[1] as u32) << 8);
            a = a.wrapping_add(k[0] as u32);
        }
        10 => {
            c = c.wrapping_add((k[9] as u32) << 8);
            c = c.wrapping_add(k[8] as u32);
            b = b.wrapping_add((k[7] as u32) << 24);
            b = b.wrapping_add((k[6] as u32) << 16);
            b = b.wrapping_add((k[5] as u32) << 8);
            b = b.wrapping_add(k[4] as u32);
            a = a.wrapping_add((k[3] as u32) << 24);
            a = a.wrapping_add((k[2] as u32) << 16);
            a = a.wrapping_add((k[1] as u32) << 8);
            a = a.wrapping_add(k[0] as u32);
        }
        9 => {
            c = c.wrapping_add(k[8] as u32);
            b = b.wrapping_add((k[7] as u32) << 24);
            b = b.wrapping_add((k[6] as u32) << 16);
            b = b.wrapping_add((k[5] as u32) << 8);
            b = b.wrapping_add(k[4] as u32);
            a = a.wrapping_add((k[3] as u32) << 24);
            a = a.wrapping_add((k[2] as u32) << 16);
            a = a.wrapping_add((k[1] as u32) << 8);
            a = a.wrapping_add(k[0] as u32);
        }
        8 => {
            b = b.wrapping_add((k[7] as u32) << 24);
            b = b.wrapping_add((k[6] as u32) << 16);
            b = b.wrapping_add((k[5] as u32) << 8);
            b = b.wrapping_add(k[4] as u32);
            a = a.wrapping_add((k[3] as u32) << 24);
            a = a.wrapping_add((k[2] as u32) << 16);
            a = a.wrapping_add((k[1] as u32) << 8);
            a = a.wrapping_add(k[0] as u32);
        }
        7 => {
            b = b.wrapping_add((k[6] as u32) << 16);
            b = b.wrapping_add((k[5] as u32) << 8);
            b = b.wrapping_add(k[4] as u32);
            a = a.wrapping_add((k[3] as u32) << 24);
            a = a.wrapping_add((k[2] as u32) << 16);
            a = a.wrapping_add((k[1] as u32) << 8);
            a = a.wrapping_add(k[0] as u32);
        }
        6 => {
            b = b.wrapping_add((k[5] as u32) << 8);
            b = b.wrapping_add(k[4] as u32);
            a = a.wrapping_add((k[3] as u32) << 24);
            a = a.wrapping_add((k[2] as u32) << 16);
            a = a.wrapping_add((k[1] as u32) << 8);
            a = a.wrapping_add(k[0] as u32);
        }
        5 => {
            b = b.wrapping_add(k[4] as u32);
            a = a.wrapping_add((k[3] as u32) << 24);
            a = a.wrapping_add((k[2] as u32) << 16);
            a = a.wrapping_add((k[1] as u32) << 8);
            a = a.wrapping_add(k[0] as u32);
        }
        4 => {
            a = a.wrapping_add((k[3] as u32) << 24);
            a = a.wrapping_add((k[2] as u32) << 16);
            a = a.wrapping_add((k[1] as u32) << 8);
            a = a.wrapping_add(k[0] as u32);
        }
        3 => {
            a = a.wrapping_add((k[2] as u32) << 16);
            a = a.wrapping_add((k[1] as u32) << 8);
            a = a.wrapping_add(k[0] as u32);
        }
        2 => {
            a = a.wrapping_add((k[1] as u32) << 8);
            a = a.wrapping_add(k[0] as u32);
        }
        1 => {
            a = a.wrapping_add(k[0] as u32);
        }
        0 => return c,
        _ => unreachable!(),
    }

    final_mix(&mut a, &mut b, &mut c);
    c
}

/// Compute HDF5 metadata checksum (convenience wrapper).
pub fn checksum_metadata(data: &[u8]) -> u32 {
    checksum_lookup3(data, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_ish() {
        // Single byte
        let hash = checksum_lookup3(&[0x42], 0);
        // Just verify it doesn't panic and returns a deterministic value
        assert_eq!(hash, checksum_lookup3(&[0x42], 0));
    }

    #[test]
    fn test_deterministic() {
        let data = b"Hello, HDF5!";
        let h1 = checksum_lookup3(data, 0);
        let h2 = checksum_lookup3(data, 0);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_initval_differs() {
        let data = b"test";
        let h1 = checksum_lookup3(data, 0);
        let h2 = checksum_lookup3(data, 1);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_longer_than_12() {
        // Test the mix path with > 12 bytes
        let data = b"This is a test string that is longer than twelve bytes";
        let h = checksum_lookup3(data, 0);
        assert_eq!(h, checksum_lookup3(data, 0));
    }
}
