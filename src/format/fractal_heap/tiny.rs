//! Fractal heap tiny-object access — mirrors libhdf5's `H5HFtiny.c`.
//! Tiny objects live entirely inside the heap-ID byte string; no I/O.

use crate::error::{Error, Result};

use super::FractalHeapHeader;

impl FractalHeapHeader {
    pub(super) fn read_tiny_payload<'a>(&self, heap_id: &'a [u8]) -> Result<&'a [u8]> {
        let length = usize::from(heap_id[0] & 0x0f) + 1;
        let data = tiny_heap_payload(heap_id, length)?;
        self.trace_tiny_object(heap_id, u64::try_from(length).unwrap_or(u64::MAX));
        Ok(data)
    }
}

fn tiny_heap_payload(heap_id: &[u8], length: usize) -> Result<&[u8]> {
    let end = 1usize
        .checked_add(length)
        .ok_or_else(|| Error::InvalidFormat("tiny heap ID length overflow".into()))?;
    heap_id
        .get(1..end)
        .ok_or_else(|| Error::InvalidFormat("tiny heap ID too short".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_heap_payload_rejects_length_overflow() {
        let err = tiny_heap_payload(&[0], usize::MAX).unwrap_err();
        assert!(
            err.to_string().contains("tiny heap ID length overflow"),
            "unexpected error: {err}"
        );
    }
}
