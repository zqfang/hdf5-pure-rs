use crate::error::{Error, Result};

pub fn vector_reduce_product(values: &[u64]) -> Result<u64> {
    values.iter().try_fold(1u64, |acc, &value| {
        acc.checked_mul(value)
            .ok_or_else(|| Error::InvalidFormat("vector product overflow".into()))
    })
}

pub fn vector_zerop_u(values: &[u64]) -> bool {
    values.iter().any(|&value| value == 0)
}

pub fn vector_zerop_s(values: &[i64]) -> bool {
    values.iter().any(|&value| value == 0)
}

pub fn vector_cmp_u(lhs: &[u64], rhs: &[u64]) -> std::cmp::Ordering {
    lhs.cmp(rhs)
}

pub fn vector_cmp_s(lhs: &[i64], rhs: &[i64]) -> std::cmp::Ordering {
    lhs.cmp(rhs)
}

pub fn vector_inc(values: &mut [u64], dims: &[u64]) -> bool {
    if values.len() != dims.len() {
        return false;
    }
    for index in (0..values.len()).rev() {
        values[index] += 1;
        if values[index] < dims[index] {
            return true;
        }
        values[index] = 0;
    }
    false
}

pub fn power2up(value: u64) -> Option<u64> {
    if value <= 1 {
        Some(1)
    } else {
        value.checked_next_power_of_two()
    }
}

pub fn bit_set(bytes: &mut [u8], bit: usize, value: bool) -> Result<()> {
    let byte = bytes
        .get_mut(bit / 8)
        .ok_or_else(|| Error::InvalidFormat("bit index out of bounds".into()))?;
    let mask = 0x80 >> (bit % 8);
    if value {
        *byte |= mask;
    } else {
        *byte &= !mask;
    }
    Ok(())
}

pub fn bit_get(bytes: &[u8], bit: usize) -> Result<bool> {
    let byte = bytes
        .get(bit / 8)
        .ok_or_else(|| Error::InvalidFormat("bit index out of bounds".into()))?;
    Ok((*byte & (0x80 >> (bit % 8))) != 0)
}

pub fn log2_of2(value: u64) -> Result<u32> {
    if value == 0 || !value.is_power_of_two() {
        return Err(Error::InvalidFormat("value is not a power of two".into()));
    }
    Ok(value.trailing_zeros())
}

pub fn log2_gen(value: u64) -> Result<u32> {
    if value == 0 {
        return Err(Error::InvalidFormat("log2 input is zero".into()));
    }
    Ok(u64::BITS - 1 - value.leading_zeros())
}

pub fn stride_optimize1(stride: usize, elem_size: usize) -> Result<usize> {
    stride
        .checked_mul(elem_size)
        .ok_or_else(|| Error::InvalidFormat("stride optimization overflow".into()))
}

pub fn stride_optimize2(stride: usize, count: usize, elem_size: usize) -> Result<usize> {
    stride
        .checked_mul(count)
        .and_then(|value| value.checked_mul(elem_size))
        .ok_or_else(|| Error::InvalidFormat("stride optimization overflow".into()))
}

pub fn hyper_stride_into(
    start: &[u64],
    stride: &[u64],
    index: &[u64],
    out: &mut [u64],
) -> Result<()> {
    if start.len() != stride.len() || start.len() != index.len() {
        return Err(Error::InvalidFormat("hyperslab rank mismatch".into()));
    }
    if out.len() != start.len() {
        return Err(Error::InvalidFormat(
            "hyperslab output rank mismatch".into(),
        ));
    }
    start
        .iter()
        .zip(stride)
        .zip(index)
        .zip(out.iter_mut())
        .try_for_each(|(((&start, &stride), &index), out)| {
            *out = index
                .checked_mul(stride)
                .and_then(|delta| start.checked_add(delta))
                .ok_or_else(|| Error::InvalidFormat("hyperslab coordinate overflow".into()))?;
            Ok(())
        })
}

pub fn hyper_eq(start_a: &[u64], stride_a: &[u64], start_b: &[u64], stride_b: &[u64]) -> bool {
    start_a == start_b && stride_a == stride_b
}

pub fn hyper_fill<T: Clone>(out: &mut [T], value: T) {
    out.fill(value);
}

pub fn hyper_copy<T: Clone>(src: &[T], dst: &mut [T]) -> Result<()> {
    if dst.len() < src.len() {
        return Err(Error::InvalidFormat(
            "hyperslab destination too small".into(),
        ));
    }
    dst[..src.len()].clone_from_slice(src);
    Ok(())
}

pub fn stride_fill<T: Clone>(
    out: &mut [T],
    offset: usize,
    stride: usize,
    count: usize,
    value: T,
) -> Result<()> {
    let mut pos = offset;
    for _ in 0..count {
        let slot = out
            .get_mut(pos)
            .ok_or_else(|| Error::InvalidFormat("stride fill index out of bounds".into()))?;
        *slot = value.clone();
        pos = pos
            .checked_add(stride)
            .ok_or_else(|| Error::InvalidFormat("stride fill index overflow".into()))?;
    }
    Ok(())
}

pub fn stride_copy<T: Clone>(
    src: &[T],
    dst: &mut [T],
    src_stride: usize,
    dst_stride: usize,
    count: usize,
) -> Result<()> {
    stride_copy_s(src, dst, 0, 0, src_stride, dst_stride, count)
}

pub fn stride_copy_s<T: Clone>(
    src: &[T],
    dst: &mut [T],
    mut src_pos: usize,
    mut dst_pos: usize,
    src_stride: usize,
    dst_stride: usize,
    count: usize,
) -> Result<()> {
    for _ in 0..count {
        let value = src
            .get(src_pos)
            .ok_or_else(|| Error::InvalidFormat("stride copy source out of bounds".into()))?
            .clone();
        let slot = dst
            .get_mut(dst_pos)
            .ok_or_else(|| Error::InvalidFormat("stride copy destination out of bounds".into()))?;
        *slot = value;
        src_pos = src_pos
            .checked_add(src_stride)
            .ok_or_else(|| Error::InvalidFormat("stride copy source overflow".into()))?;
        dst_pos = dst_pos
            .checked_add(dst_stride)
            .ok_or_else(|| Error::InvalidFormat("stride copy destination overflow".into()))?;
    }
    Ok(())
}

pub fn array_down_into(coords: &[u64], dims: &[u64], out: &mut [u64]) -> Result<()> {
    if coords.len() != dims.len() {
        return Err(Error::InvalidFormat("array rank mismatch".into()));
    }
    if out.len() != coords.len() {
        return Err(Error::InvalidFormat("array output rank mismatch".into()));
    }
    coords
        .iter()
        .zip(dims)
        .zip(out.iter_mut())
        .try_for_each(|((&coord, &dim), out)| {
            if dim == 0 {
                Err(Error::InvalidFormat("array dimension is zero".into()))
            } else {
                *out = coord % dim;
                Ok(())
            }
        })
}

pub fn array_offset_pre_into(dims: &[u64], out: &mut [u64]) -> Result<()> {
    if out.len() != dims.len() {
        return Err(Error::InvalidFormat("array stride rank mismatch".into()));
    }
    out.fill(1);
    if dims.len() > 1 {
        for index in (0..dims.len() - 1).rev() {
            out[index] = out[index + 1]
                .checked_mul(dims[index + 1])
                .ok_or_else(|| Error::InvalidFormat("array stride overflow".into()))?;
        }
    }
    Ok(())
}

pub fn visit_array_offset_pre<F>(dims: &[u64], mut visitor: F) -> Result<()>
where
    F: FnMut(u64) -> Result<()>,
{
    for index in 0..dims.len() {
        let stride = dims[index + 1..].iter().try_fold(1u64, |acc, &dim| {
            acc.checked_mul(dim)
                .ok_or_else(|| Error::InvalidFormat("array stride overflow".into()))
        })?;
        visitor(stride)?;
    }
    Ok(())
}

pub fn array_offset_with_strides(coords: &[u64], dims: &[u64], strides: &[u64]) -> Result<u64> {
    if coords.len() != dims.len() {
        return Err(Error::InvalidFormat("array rank mismatch".into()));
    }
    if strides.len() != coords.len() {
        return Err(Error::InvalidFormat("array stride rank mismatch".into()));
    }
    coords
        .iter()
        .zip(dims)
        .zip(strides.iter())
        .try_fold(0u64, |acc, ((&coord, &dim), &stride)| {
            if coord >= dim {
                return Err(Error::InvalidFormat(
                    "array coordinate out of bounds".into(),
                ));
            }
            let term = coord
                .checked_mul(stride)
                .ok_or_else(|| Error::InvalidFormat("array offset overflow".into()))?;
            acc.checked_add(term)
                .ok_or_else(|| Error::InvalidFormat("array offset overflow".into()))
        })
}

pub fn array_offset(coords: &[u64], dims: &[u64]) -> Result<u64> {
    if coords.len() != dims.len() {
        return Err(Error::InvalidFormat("array rank mismatch".into()));
    }
    let mut stride = 1u64;
    let mut offset = 0u64;
    for index in (0..coords.len()).rev() {
        let coord = coords[index];
        let dim = dims[index];
        if coord >= dim {
            return Err(Error::InvalidFormat(
                "array coordinate out of bounds".into(),
            ));
        }
        let term = coord
            .checked_mul(stride)
            .ok_or_else(|| Error::InvalidFormat("array offset overflow".into()))?;
        offset = offset
            .checked_add(term)
            .ok_or_else(|| Error::InvalidFormat("array offset overflow".into()))?;
        if index > 0 {
            stride = stride
                .checked_mul(dim)
                .ok_or_else(|| Error::InvalidFormat("array stride overflow".into()))?;
        }
    }
    Ok(offset)
}

pub fn chunk_index(coords: &[u64], dims: &[u64]) -> Result<u64> {
    array_offset(coords, dims)
}

pub fn chunk_scaled_into(coords: &[u64], chunk_dims: &[u64], out: &mut [u64]) -> Result<()> {
    if coords.len() != chunk_dims.len() {
        return Err(Error::InvalidFormat("chunk rank mismatch".into()));
    }
    if out.len() != coords.len() {
        return Err(Error::InvalidFormat("chunk output rank mismatch".into()));
    }
    coords
        .iter()
        .zip(chunk_dims)
        .zip(out.iter_mut())
        .try_for_each(|((&coord, &chunk), out)| {
            if chunk == 0 {
                Err(Error::InvalidFormat("chunk dimension is zero".into()))
            } else {
                *out = coord / chunk;
                Ok(())
            }
        })
}

pub fn chunk_index_scaled_into(
    coords: &[u64],
    dims: &[u64],
    chunk_dims: &[u64],
    scaled: &mut [u64],
    scaled_dims: &mut [u64],
) -> Result<u64> {
    chunk_scaled_into(coords, chunk_dims, scaled)?;
    chunk_scaled_into(dims, chunk_dims, scaled_dims)?;
    chunk_index(scaled, scaled_dims)
}

pub fn opvv<T: Copy, F: Fn(T, T) -> T>(lhs: &[T], rhs: &[T], out: &mut [T], op: F) -> Result<()> {
    if lhs.len() != rhs.len() || lhs.len() > out.len() {
        return Err(Error::InvalidFormat(
            "vector operation length mismatch".into(),
        ));
    }
    for ((out, &lhs), &rhs) in out.iter_mut().zip(lhs).zip(rhs) {
        *out = op(lhs, rhs);
    }
    Ok(())
}

pub fn memcpyvv<T: Copy>(
    dst: &mut [T],
    dst_offsets: &[usize],
    src: &[T],
    src_offsets: &[usize],
    lens: &[usize],
) -> Result<()> {
    if dst_offsets.len() != src_offsets.len() || dst_offsets.len() != lens.len() {
        return Err(Error::InvalidFormat(
            "vector copy segment length mismatch".into(),
        ));
    }
    for ((&dst_offset, &src_offset), &len) in dst_offsets.iter().zip(src_offsets).zip(lens) {
        let src_end = src_offset
            .checked_add(len)
            .ok_or_else(|| Error::InvalidFormat("vector copy source overflow".into()))?;
        let dst_end = dst_offset
            .checked_add(len)
            .ok_or_else(|| Error::InvalidFormat("vector copy destination overflow".into()))?;
        let src_window = src
            .get(src_offset..src_end)
            .ok_or_else(|| Error::InvalidFormat("vector copy source out of bounds".into()))?;
        let dst_window = dst
            .get_mut(dst_offset..dst_end)
            .ok_or_else(|| Error::InvalidFormat("vector copy destination out of bounds".into()))?;
        dst_window.copy_from_slice(src_window);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_scaling_and_indexing_work() {
        let mut scaled = [0, 0];
        chunk_scaled_into(&[9, 5], &[4, 2], &mut scaled).unwrap();
        assert_eq!(scaled, [2, 2]);
        assert_eq!(chunk_index(&[2, 1], &[4, 4]).unwrap(), 9);
    }

    #[test]
    fn coordinate_helpers_fill_caller_buffers() {
        let mut coords = [0, 0];
        hyper_stride_into(&[10, 20], &[2, 3], &[4, 5], &mut coords).unwrap();
        assert_eq!(coords, [18, 35]);

        array_down_into(&[9, 5], &[4, 2], &mut coords).unwrap();
        assert_eq!(coords, [1, 1]);

        array_offset_pre_into(&[3, 4], &mut coords).unwrap();
        assert_eq!(coords, [4, 1]);

        let mut visited = Vec::new();
        visit_array_offset_pre(&[3, 4], |stride| {
            visited.push(stride);
            Ok(())
        })
        .unwrap();
        assert_eq!(visited, [4, 1]);
        assert_eq!(
            array_offset_with_strides(&[2, 1], &[3, 4], &coords).unwrap(),
            9
        );

        chunk_scaled_into(&[9, 5], &[4, 2], &mut coords).unwrap();
        assert_eq!(coords, [2, 2]);

        let mut scaled = [0, 0];
        let mut scaled_dims = [0, 0];
        assert_eq!(
            chunk_index_scaled_into(&[9, 5], &[16, 8], &[4, 2], &mut scaled, &mut scaled_dims)
                .unwrap(),
            10
        );
        assert_eq!(scaled, [2, 2]);
        assert_eq!(scaled_dims, [4, 4]);
    }

    #[test]
    fn vector_copy_segments_work() {
        let src = [1, 2, 3, 4, 5];
        let mut dst = [0; 5];
        memcpyvv(&mut dst, &[0, 3], &src, &[1, 4], &[2, 1]).unwrap();
        assert_eq!(dst, [2, 3, 0, 5, 0]);
    }
}
