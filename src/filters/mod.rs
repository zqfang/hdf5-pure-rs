pub mod blosc;
pub mod deflate;
pub mod fletcher32;
pub mod lzf;
pub mod nbit;
pub mod registry;
pub mod scaleoffset;
pub mod shuffle;
pub mod szip;

use std::borrow::Cow;

use crate::error::{Error, Result};
use crate::format::messages::filter_pipeline::{
    FilterDesc, FilterPipelineMessage, FILTER_DEFLATE, FILTER_FLETCHER32, FILTER_NBIT,
    FILTER_SCALEOFFSET, FILTER_SHUFFLE, FILTER_SZIP,
};

/// Apply filter pipeline in reverse (for reading/decompression).
/// Filters are applied in reverse order of their definition.
pub fn apply_pipeline_reverse(
    data: &[u8],
    pipeline: &FilterPipelineMessage,
    element_size: usize,
) -> Result<Vec<u8>> {
    apply_pipeline_reverse_with_mask(data, pipeline, element_size, 0)
}

/// Apply filter pipeline in reverse while honoring an HDF5 per-chunk filter mask.
/// Bit `i` set means filter `i` in the stored pipeline was not applied to the chunk.
pub fn apply_pipeline_reverse_with_mask(
    data: &[u8],
    pipeline: &FilterPipelineMessage,
    element_size: usize,
    filter_mask: u32,
) -> Result<Vec<u8>> {
    apply_pipeline_reverse_with_mask_and_expected(data, pipeline, element_size, filter_mask, None)
}

pub fn apply_pipeline_reverse_with_mask_into(
    data: &[u8],
    pipeline: &FilterPipelineMessage,
    element_size: usize,
    filter_mask: u32,
    out: &mut Vec<u8>,
) -> Result<()> {
    apply_pipeline_reverse_with_mask_into_inner(
        data,
        pipeline,
        element_size,
        filter_mask,
        None,
        out,
    )
}

pub fn apply_pipeline_reverse_with_mask_expected(
    data: &[u8],
    pipeline: &FilterPipelineMessage,
    element_size: usize,
    filter_mask: u32,
    expected_len: usize,
) -> Result<Vec<u8>> {
    apply_pipeline_reverse_with_mask_and_expected(
        data,
        pipeline,
        element_size,
        filter_mask,
        Some(expected_len),
    )
}

pub fn apply_pipeline_reverse_with_mask_expected_into(
    data: &[u8],
    pipeline: &FilterPipelineMessage,
    element_size: usize,
    filter_mask: u32,
    expected_len: usize,
    out: &mut Vec<u8>,
) -> Result<()> {
    apply_pipeline_reverse_with_mask_into_inner(
        data,
        pipeline,
        element_size,
        filter_mask,
        Some(expected_len),
        out,
    )
}

fn apply_pipeline_reverse_with_mask_into_inner(
    data: &[u8],
    pipeline: &FilterPipelineMessage,
    element_size: usize,
    filter_mask: u32,
    expected_len: Option<usize>,
    out: &mut Vec<u8>,
) -> Result<()> {
    let mut staged = Vec::new();
    apply_pipeline_reverse_with_mask_into_inner_unchecked(
        data,
        pipeline,
        element_size,
        filter_mask,
        expected_len,
        &mut staged,
    )?;
    *out = staged;
    Ok(())
}

fn apply_pipeline_reverse_with_mask_into_inner_unchecked(
    data: &[u8],
    pipeline: &FilterPipelineMessage,
    element_size: usize,
    filter_mask: u32,
    expected_len: Option<usize>,
    out: &mut Vec<u8>,
) -> Result<()> {
    validate_filter_mask(pipeline, filter_mask)?;

    let mut active_filters = pipeline
        .filters
        .iter()
        .enumerate()
        .filter(|(index, _)| filter_mask & (1u32 << index) == 0);

    let Some((index, filter)) = active_filters.next_back() else {
        out.clear();
        out.extend_from_slice(data);
        validate_expected_len(out.len(), expected_len)?;
        return Ok(());
    };

    if active_filters.next_back().is_none() {
        let deflate_exact_len = deflate_exact_len_hint(pipeline, filter_mask, index, expected_len);
        apply_filter_reverse_into(
            data,
            filter,
            element_size,
            expected_len,
            deflate_exact_len,
            out,
        )?;
        validate_expected_len(out.len(), expected_len)?;
        return Ok(());
    }

    apply_pipeline_reverse_multi_into(
        data,
        pipeline,
        element_size,
        filter_mask,
        expected_len,
        out,
    )?;
    validate_expected_len(out.len(), expected_len)?;
    Ok(())
}

enum PipelineBuffer {
    Input,
    Out,
    Scratch,
}

fn apply_pipeline_reverse_multi_into(
    data: &[u8],
    pipeline: &FilterPipelineMessage,
    element_size: usize,
    filter_mask: u32,
    expected_len: Option<usize>,
    out: &mut Vec<u8>,
) -> Result<()> {
    let mut current = PipelineBuffer::Input;
    let mut scratch = Vec::new();

    for (index, filter) in pipeline.filters.iter().enumerate().rev() {
        if filter_mask & (1u32 << index) != 0 {
            continue;
        }

        let deflate_exact_len = deflate_exact_len_hint(pipeline, filter_mask, index, expected_len);
        current = match current {
            PipelineBuffer::Input => {
                apply_filter_reverse_into(
                    data,
                    filter,
                    element_size,
                    expected_len,
                    deflate_exact_len,
                    out,
                )?;
                PipelineBuffer::Out
            }
            PipelineBuffer::Out => {
                apply_filter_reverse_into(
                    out,
                    filter,
                    element_size,
                    expected_len,
                    deflate_exact_len,
                    &mut scratch,
                )?;
                PipelineBuffer::Scratch
            }
            PipelineBuffer::Scratch => {
                apply_filter_reverse_into(
                    &scratch,
                    filter,
                    element_size,
                    expected_len,
                    deflate_exact_len,
                    out,
                )?;
                PipelineBuffer::Out
            }
        };
    }

    if let PipelineBuffer::Scratch = current {
        std::mem::swap(out, &mut scratch);
    }
    Ok(())
}

fn apply_pipeline_reverse_with_mask_and_expected(
    data: &[u8],
    pipeline: &FilterPipelineMessage,
    element_size: usize,
    filter_mask: u32,
    expected_len: Option<usize>,
) -> Result<Vec<u8>> {
    apply_pipeline_reverse_with_mask_and_expected_cow(
        data,
        pipeline,
        element_size,
        filter_mask,
        expected_len,
    )
    .map(Cow::into_owned)
}

fn apply_pipeline_reverse_with_mask_and_expected_cow<'a>(
    data: &'a [u8],
    pipeline: &FilterPipelineMessage,
    element_size: usize,
    filter_mask: u32,
    expected_len: Option<usize>,
) -> Result<Cow<'a, [u8]>> {
    validate_filter_mask(pipeline, filter_mask)?;

    #[cfg(feature = "tracehash")]
    let mut th = {
        let mut th = tracehash::th_call!("hdf5.filter_pipeline.apply");
        th.input_u64(usize_to_u64(
            pipeline.filters.len(),
            "filter pipeline length",
        )?);
        th.input_u64(0x0100);
        th.input_u64(u64::from(filter_mask));
        th.input_u64(usize_to_u64(data.len(), "filter input length")?);
        th
    };

    let mut buf: Cow<'a, [u8]> = Cow::Borrowed(data);

    // Apply filters in reverse order
    for (index, filter) in pipeline.filters.iter().enumerate().rev() {
        if filter_mask & (1u32 << index) != 0 {
            continue;
        }
        let deflate_exact_len = deflate_exact_len_hint(pipeline, filter_mask, index, expected_len);
        buf = apply_filter_reverse(buf, filter, element_size, expected_len, deflate_exact_len)?;
    }

    validate_expected_len(buf.len(), expected_len)?;

    #[cfg(feature = "tracehash")]
    {
        th.output_value(&(true));
        th.output_u64(0);
        th.output_u64(usize_to_u64(buf.len(), "filter output length")?);
        th.finish();
    }

    Ok(buf)
}

fn validate_filter_mask(pipeline: &FilterPipelineMessage, filter_mask: u32) -> Result<()> {
    if pipeline.filters.len() > 32 {
        return Err(Error::InvalidFormat(format!(
            "filter pipeline length {} exceeds 32-bit chunk filter mask",
            pipeline.filters.len()
        )));
    }

    let valid_mask = if pipeline.filters.len() >= 32 {
        u32::MAX
    } else {
        (1u32 << pipeline.filters.len()) - 1
    };
    if filter_mask & !valid_mask != 0 {
        return Err(Error::InvalidFormat(format!(
            "filter mask {filter_mask:#x} references filters outside pipeline length {}",
            pipeline.filters.len()
        )));
    }
    Ok(())
}

fn validate_expected_len(actual_len: usize, expected_len: Option<usize>) -> Result<()> {
    if let Some(expected_len) = expected_len {
        if actual_len != expected_len {
            return Err(Error::InvalidFormat(format!(
                "filter pipeline output length mismatch: expected {expected_len}, got {actual_len}"
            )));
        }
    }
    Ok(())
}

fn apply_filter_reverse<'a>(
    data: Cow<'a, [u8]>,
    filter: &FilterDesc,
    element_size: usize,
    expected_len: Option<usize>,
    deflate_exact_len: Option<usize>,
) -> Result<Cow<'a, [u8]>> {
    let bytes = data.as_ref();
    match filter.id {
        FILTER_DEFLATE => {
            let mut out = Vec::new();
            if let Some(expected_len) = deflate_exact_len {
                out.resize(expected_len, 0);
                deflate::decompress_exact_into(bytes, &mut out)?;
            } else {
                deflate::decompress_with_hint_into(bytes, expected_len, &mut out)?;
            }
            Ok(Cow::Owned(out))
        }
        FILTER_SHUFFLE => {
            let shuffle_element_size =
                shuffle_element_size(filter, element_size).ok_or_else(|| {
                    Error::InvalidFormat("shuffle filter element size is zero".into())
                })?;
            if shuffle::is_noop(data.len(), shuffle_element_size) {
                return Ok(data);
            }
            let mut out = vec![0u8; bytes.len()];
            shuffle::unshuffle_into(bytes, shuffle_element_size, &mut out)?;
            Ok(Cow::Owned(out))
        }
        FILTER_FLETCHER32 => match data {
            Cow::Borrowed(bytes) => {
                let payload = fletcher32::verify_and_strip_view(bytes)?;
                Ok(Cow::Borrowed(payload))
            }
            Cow::Owned(mut bytes) => {
                let payload_len = fletcher32::verify_and_strip_view(&bytes)?.len();
                bytes.truncate(payload_len);
                Ok(Cow::Owned(bytes))
            }
        },
        FILTER_NBIT => match data {
            Cow::Borrowed(bytes) => {
                if let Some(payload) = nbit::decompress_view_if_noop(bytes, &filter.client_data)? {
                    return Ok(Cow::Borrowed(payload));
                }
                let mut out = Vec::new();
                nbit::decompress_into(bytes, &filter.client_data, &mut out)?;
                Ok(Cow::Owned(out))
            }
            Cow::Owned(bytes) => {
                if nbit::decompress_view_if_noop(&bytes, &filter.client_data)?.is_some() {
                    return Ok(Cow::Owned(bytes));
                }
                let mut out = Vec::new();
                nbit::decompress_into(&bytes, &filter.client_data, &mut out)?;
                Ok(Cow::Owned(out))
            }
        },
        FILTER_SCALEOFFSET => {
            let mut out = Vec::new();
            scaleoffset::decompress_into(bytes, &filter.client_data, &mut out)?;
            Ok(Cow::Owned(out))
        }
        FILTER_SZIP => szip::decompress(bytes).map(Cow::Owned),
        32001 => blosc::decompress(bytes).map(Cow::Owned), // HDF5 Blosc filter ID
        32000 => {
            // LZF filter -- need the uncompressed size
            // LZF stores the original size in the first client_data parameter
            let expected = if let Some(&encoded) = filter.client_data.first() {
                usize::try_from(encoded)
                    .map_err(|_| Error::InvalidFormat("lzf expected size exceeds usize".into()))?
            } else {
                bytes
                    .len()
                    .checked_mul(2)
                    .ok_or_else(|| Error::InvalidFormat("lzf expected size hint overflow".into()))?
            };
            let mut out = vec![0u8; expected];
            lzf::decompress_into(bytes, &mut out)?;
            Ok(Cow::Owned(out))
        }
        _ => Err(Error::Unsupported(format!(
            "filter {} not implemented",
            filter.id
        ))),
    }
}

fn apply_filter_reverse_into(
    data: &[u8],
    filter: &FilterDesc,
    element_size: usize,
    expected_len: Option<usize>,
    deflate_exact_len: Option<usize>,
    out: &mut Vec<u8>,
) -> Result<()> {
    out.clear();
    match filter.id {
        FILTER_DEFLATE => {
            if let Some(expected_len) = deflate_exact_len {
                out.resize(expected_len, 0);
                deflate::decompress_exact_into(data, out)?;
            } else {
                deflate::decompress_with_hint_into(data, expected_len, out)?;
            }
        }
        FILTER_SHUFFLE => {
            let shuffle_element_size =
                shuffle_element_size(filter, element_size).ok_or_else(|| {
                    Error::InvalidFormat("shuffle filter element size is zero".into())
                })?;
            if shuffle::is_noop(data.len(), shuffle_element_size) {
                out.extend_from_slice(data);
            } else {
                out.resize(data.len(), 0);
                shuffle::unshuffle_into(data, shuffle_element_size, out)?;
            }
        }
        FILTER_FLETCHER32 => {
            out.extend_from_slice(fletcher32::verify_and_strip_view(data)?);
        }
        FILTER_NBIT => {
            if let Some(payload) = nbit::decompress_view_if_noop(data, &filter.client_data)? {
                out.extend_from_slice(payload);
            } else {
                nbit::decompress_into(data, &filter.client_data, out)?;
            }
        }
        FILTER_SCALEOFFSET => {
            scaleoffset::decompress_into(data, &filter.client_data, out)?;
        }
        FILTER_SZIP => {
            *out = szip::decompress(data)?;
        }
        32001 => {
            *out = blosc::decompress(data)?;
        }
        32000 => {
            let expected = if let Some(&encoded) = filter.client_data.first() {
                usize::try_from(encoded)
                    .map_err(|_| Error::InvalidFormat("lzf expected size exceeds usize".into()))?
            } else {
                data.len()
                    .checked_mul(2)
                    .ok_or_else(|| Error::InvalidFormat("lzf expected size hint overflow".into()))?
            };
            out.resize(expected, 0);
            lzf::decompress_into(data, out)?;
        }
        _ => {
            return Err(Error::Unsupported(format!(
                "filter {} not implemented",
                filter.id
            )));
        }
    }
    Ok(())
}

#[cfg(feature = "tracehash")]
fn usize_to_u64(value: usize, context: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| Error::InvalidFormat(format!("{context} exceeds u64")))
}

fn shuffle_element_size(filter: &FilterDesc, dataset_element_size: usize) -> Option<usize> {
    let Some(&encoded_size) = filter.client_data.first() else {
        return (dataset_element_size != 0).then_some(dataset_element_size);
    };
    if encoded_size == 0 {
        return None;
    }
    usize::try_from(encoded_size).ok()
}

fn deflate_exact_len_hint(
    pipeline: &FilterPipelineMessage,
    filter_mask: u32,
    current_index: usize,
    expected_len: Option<usize>,
) -> Option<usize> {
    let expected_len = expected_len?;
    if pipeline.filters[current_index].id != FILTER_DEFLATE {
        return None;
    }
    for (index, filter) in pipeline.filters[..current_index].iter().enumerate() {
        if filter_mask & (1u32 << index) != 0 {
            continue;
        }
        let preserves_len = matches!(filter.id, FILTER_SHUFFLE | FILTER_FLETCHER32);
        if !preserves_len {
            return None;
        }
    }
    Some(expected_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deflate_pipeline() -> FilterPipelineMessage {
        FilterPipelineMessage {
            version: 2,
            filters: vec![FilterDesc {
                id: FILTER_DEFLATE,
                name: Some("deflate".into()),
                flags: 0,
                client_data: vec![4],
            }],
        }
    }

    fn unknown_pipeline(flags: u16) -> FilterPipelineMessage {
        FilterPipelineMessage {
            version: 2,
            filters: vec![FilterDesc {
                id: 32_099,
                name: Some("unknown".into()),
                flags,
                client_data: Vec::new(),
            }],
        }
    }

    fn shuffle_pipeline(element_size: u32) -> FilterPipelineMessage {
        FilterPipelineMessage {
            version: 2,
            filters: vec![FilterDesc {
                id: FILTER_SHUFFLE,
                name: Some("shuffle".into()),
                flags: 0,
                client_data: vec![element_size],
            }],
        }
    }

    fn nbit_noop_pipeline() -> FilterPipelineMessage {
        FilterPipelineMessage {
            version: 2,
            filters: vec![FilterDesc {
                id: FILTER_NBIT,
                name: Some("nbit".into()),
                flags: 0,
                client_data: vec![5, 1, 0, 0, 1],
            }],
        }
    }

    fn deflate_compress(data: &[u8], level: u32) -> Vec<u8> {
        let mut out = Vec::new();
        deflate::compress_into(data, level, &mut out).unwrap();
        out
    }

    #[test]
    fn no_filter_reverse_borrows_input_internally() {
        let pipeline = FilterPipelineMessage {
            version: 2,
            filters: Vec::new(),
        };
        let out = apply_pipeline_reverse_with_mask_and_expected_cow(b"abcd", &pipeline, 1, 0, None)
            .unwrap();
        assert!(matches!(out, Cow::Borrowed(b"abcd")));
    }

    #[test]
    fn fully_masked_reverse_borrows_input_internally() {
        let out = apply_pipeline_reverse_with_mask_and_expected_cow(
            b"abcd",
            &unknown_pipeline(1),
            1,
            0b1,
            Some(4),
        )
        .unwrap();
        assert!(matches!(out, Cow::Borrowed(b"abcd")));
    }

    #[test]
    fn shuffle_noop_reverse_borrows_input_internally() {
        let out = apply_pipeline_reverse_with_mask_and_expected_cow(
            b"abcd",
            &shuffle_pipeline(4),
            1,
            0,
            Some(4),
        )
        .unwrap();
        assert!(matches!(out, Cow::Borrowed(b"abcd")));
    }

    #[test]
    fn nbit_noop_reverse_borrows_input_internally() {
        let out = apply_pipeline_reverse_with_mask_and_expected_cow(
            b"abcd",
            &nbit_noop_pipeline(),
            1,
            0,
            Some(4),
        )
        .unwrap();
        assert!(matches!(out, Cow::Borrowed(b"abcd")));
    }

    #[test]
    fn expected_length_accepts_exact_filter_output() {
        let compressed = deflate_compress(b"abcd", 4);
        let out =
            apply_pipeline_reverse_with_mask_expected(&compressed, &deflate_pipeline(), 1, 0, 4)
                .unwrap();
        assert_eq!(out, b"abcd");
    }

    #[test]
    fn expected_length_rejects_filter_output_mismatch() {
        let compressed = deflate_compress(b"abcd", 4);
        let err =
            apply_pipeline_reverse_with_mask_expected(&compressed, &deflate_pipeline(), 1, 0, 3)
                .unwrap_err();
        assert!(
            err.to_string()
                .contains("filter pipeline output length mismatch")
                || err
                    .to_string()
                    .contains("deflate decompression produced more bytes than expected"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn reverse_pipeline_into_preserves_output_on_validation_errors() {
        let pipeline = deflate_pipeline();
        let compressed = deflate_compress(b"abcd", 4);
        let mut out = b"stale".to_vec();

        let err = apply_pipeline_reverse_with_mask_expected_into(
            &compressed,
            &pipeline,
            1,
            0,
            3,
            &mut out,
        )
        .expect_err("expected-length mismatch should fail");
        assert!(
            err.to_string()
                .contains("filter pipeline output length mismatch")
                || err
                    .to_string()
                    .contains("deflate decompression produced more bytes than expected"),
            "unexpected error: {err}"
        );
        assert_eq!(out, b"stale");

        let err = apply_pipeline_reverse_with_mask_into(b"abcd", &pipeline, 1, 0b10, &mut out)
            .expect_err("filter mask outside pipeline should fail");
        assert!(err.to_string().contains("filter mask"));
        assert_eq!(out, b"stale");

        let err =
            apply_pipeline_reverse_with_mask_into(b"abcd", &shuffle_pipeline(0), 4, 0, &mut out)
                .expect_err("invalid shuffle client data should fail");
        assert!(err
            .to_string()
            .contains("shuffle filter element size is zero"));
        assert_eq!(out, b"stale");
    }

    #[test]
    fn masked_unknown_optional_filter_is_skipped() {
        let out = apply_pipeline_reverse_with_mask(b"abcd", &unknown_pipeline(1), 1, 0b1).unwrap();
        assert_eq!(out, b"abcd");
    }

    #[test]
    fn unmasked_unknown_optional_filter_fails() {
        let err =
            apply_pipeline_reverse_with_mask(b"abcd", &unknown_pipeline(1), 1, 0).unwrap_err();
        assert!(
            err.to_string().contains("filter 32099 not implemented"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn unmasked_unknown_required_filter_fails() {
        let err =
            apply_pipeline_reverse_with_mask(b"abcd", &unknown_pipeline(0), 1, 0).unwrap_err();
        assert!(
            err.to_string().contains("filter 32099 not implemented"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn shuffle_uses_filter_client_element_size() {
        let data = [1u8, 5, 2, 6, 3, 7, 4, 8];
        let out = apply_pipeline_reverse_with_mask(&data, &shuffle_pipeline(4), 1, 0).unwrap();
        assert_eq!(out, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn multi_filter_reverse_into_reuses_caller_output() {
        let data = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let mut shuffled = vec![0; data.len()];
        shuffle::shuffle_into(&data, 4, &mut shuffled).unwrap();
        let mut encoded = Vec::new();
        fletcher32::append_checksum_into(&shuffled, &mut encoded).unwrap();

        let pipeline = FilterPipelineMessage {
            version: 2,
            filters: vec![
                FilterDesc {
                    id: FILTER_SHUFFLE,
                    name: Some("shuffle".into()),
                    flags: 0,
                    client_data: vec![4],
                },
                FilterDesc {
                    id: FILTER_FLETCHER32,
                    name: Some("fletcher32".into()),
                    flags: 0,
                    client_data: Vec::new(),
                },
            ],
        };
        let mut out = Vec::with_capacity(data.len());
        apply_pipeline_reverse_with_mask_expected_into(
            &encoded,
            &pipeline,
            1,
            0,
            data.len(),
            &mut out,
        )
        .unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn shuffle_rejects_zero_client_element_size() {
        let err = apply_pipeline_reverse_with_mask(b"abcd", &shuffle_pipeline(0), 4, 0)
            .expect_err("zero-sized shuffle parameter should fail");
        assert!(err
            .to_string()
            .contains("shuffle filter element size is zero"));
    }
}
