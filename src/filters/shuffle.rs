use crate::error::{Error, Result};

/// Unshuffle bytes into a provided output buffer.
///
/// Shuffle rearranges bytes so that byte 0 of all elements comes first,
/// then byte 1 of all elements, etc. Unshuffle reverses this.
pub fn unshuffle_into(data: &[u8], element_size: usize, out: &mut [u8]) -> Result<()> {
    if out.len() != data.len() {
        return Err(crate::error::Error::InvalidFormat(
            "shuffle output length mismatch".into(),
        ));
    }
    if element_size <= 1 || data.is_empty() {
        out.copy_from_slice(data);
        return Ok(());
    }

    let n_elements = data.len() / element_size;
    let grouped = n_elements * element_size;
    match element_size {
        8 => {
            let p0 = &data[0..n_elements];
            let p1 = &data[n_elements..2 * n_elements];
            let p2 = &data[2 * n_elements..3 * n_elements];
            let p3 = &data[3 * n_elements..4 * n_elements];
            let p4 = &data[4 * n_elements..5 * n_elements];
            let p5 = &data[5 * n_elements..6 * n_elements];
            let p6 = &data[6 * n_elements..7 * n_elements];
            let p7 = &data[7 * n_elements..8 * n_elements];
            for (i, elem) in out[..grouped].chunks_exact_mut(8).enumerate() {
                elem[0] = p0[i];
                elem[1] = p1[i];
                elem[2] = p2[i];
                elem[3] = p3[i];
                elem[4] = p4[i];
                elem[5] = p5[i];
                elem[6] = p6[i];
                elem[7] = p7[i];
            }
        }
        4 => {
            let p0 = &data[0..n_elements];
            let p1 = &data[n_elements..2 * n_elements];
            let p2 = &data[2 * n_elements..3 * n_elements];
            let p3 = &data[3 * n_elements..4 * n_elements];
            for (i, elem) in out[..grouped].chunks_exact_mut(4).enumerate() {
                elem[0] = p0[i];
                elem[1] = p1[i];
                elem[2] = p2[i];
                elem[3] = p3[i];
            }
        }
        2 => {
            let p0 = &data[0..n_elements];
            let p1 = &data[n_elements..2 * n_elements];
            for (i, elem) in out[..grouped].chunks_exact_mut(2).enumerate() {
                elem[0] = p0[i];
                elem[1] = p1[i];
            }
        }
        _ => {
            for i in 0..n_elements {
                for j in 0..element_size {
                    let dst = shuffle_index(i, element_size, j)?;
                    let src = shuffle_index(j, n_elements, i)?;
                    out[dst] = data[src];
                }
            }
        }
    }
    out[grouped..].copy_from_slice(&data[grouped..]);
    Ok(())
}

/// Shuffle bytes for compression into a provided output buffer.
pub fn shuffle_into(data: &[u8], element_size: usize, out: &mut [u8]) -> Result<()> {
    if out.len() != data.len() {
        return Err(crate::error::Error::InvalidFormat(
            "shuffle output length mismatch".into(),
        ));
    }
    if element_size <= 1 || data.is_empty() {
        out.copy_from_slice(data);
        return Ok(());
    }

    let n_elements = data.len() / element_size;
    for i in 0..n_elements {
        for j in 0..element_size {
            let dst = shuffle_index(j, n_elements, i)?;
            let src = shuffle_index(i, element_size, j)?;
            out[dst] = data[src];
        }
    }
    let grouped = n_elements * element_size;
    out[grouped..].copy_from_slice(&data[grouped..]);
    Ok(())
}

/// Return true when shuffle/unshuffle would leave the byte stream unchanged.
pub fn is_noop(data_len: usize, element_size: usize) -> bool {
    element_size <= 1 || data_len / element_size <= 1
}

/// Validate and return the shuffle element size for local filter setup.
pub fn set_local_shuffle(element_size: usize) -> Result<usize> {
    if element_size == 0 {
        return Err(Error::InvalidFormat(
            "shuffle filter element size is zero".into(),
        ));
    }
    Ok(element_size)
}

/// HDF5 shuffle filter entry point: reverse unshuffles, forward shuffles,
/// writing into a caller-provided buffer.
pub fn filter_shuffle_into(
    data: &[u8],
    element_size: usize,
    reverse: bool,
    out: &mut [u8],
) -> Result<()> {
    let element_size = set_local_shuffle(element_size)?;
    if reverse {
        unshuffle_into(data, element_size, out)
    } else {
        shuffle_into(data, element_size, out)
    }
}

fn shuffle_index(base: usize, stride: usize, offset: usize) -> Result<usize> {
    base.checked_mul(stride)
        .and_then(|value| value.checked_add(offset))
        .ok_or_else(|| Error::InvalidFormat("shuffle index overflow".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shuffle_roundtrip() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8]; // 2 elements of 4 bytes each
        let mut shuffled = vec![0; data.len()];
        shuffle_into(&data, 4, &mut shuffled).unwrap();
        let mut unshuffled = vec![0; shuffled.len()];
        unshuffle_into(&shuffled, 4, &mut unshuffled).unwrap();
        assert_eq!(unshuffled, data);
    }

    #[test]
    fn test_shuffle_roundtrip_preserves_trailing_bytes() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let mut shuffled = vec![0; data.len()];
        shuffle_into(&data, 4, &mut shuffled).unwrap();
        let mut unshuffled = vec![0; shuffled.len()];
        unshuffle_into(&shuffled, 4, &mut unshuffled).unwrap();
        assert_eq!(unshuffled, data);
    }

    #[test]
    fn shuffle_index_rejects_overflow() {
        let err = shuffle_index(usize::MAX, 2, 0).unwrap_err();
        assert!(
            err.to_string().contains("shuffle index overflow"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn shuffle_noop_detects_single_or_smaller_element_payloads() {
        assert!(is_noop(0, 4));
        assert!(is_noop(1, 1));
        assert!(is_noop(3, 4));
        assert!(is_noop(4, 4));
        assert!(!is_noop(8, 4));
    }

    #[test]
    fn filter_shuffle_preserves_output_on_validation_errors() {
        let data = [1, 2, 3, 4];
        let mut out = *b"stale";

        let err = filter_shuffle_into(&data, 0, false, &mut out).unwrap_err();
        assert!(
            err.to_string()
                .contains("shuffle filter element size is zero"),
            "unexpected error: {err}"
        );
        assert_eq!(&out, b"stale");

        let err = filter_shuffle_into(&data, 2, true, &mut out[..3]).unwrap_err();
        assert!(
            err.to_string().contains("shuffle output length mismatch"),
            "unexpected error: {err}"
        );
        assert_eq!(&out, b"stale");
    }
}
