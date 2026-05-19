use std::borrow::Cow;

use crate::error::{Error, Result};

pub fn memcpy(dst: &mut [u8], src: &[u8]) -> Result<()> {
    if dst.len() < src.len() {
        return Err(Error::InvalidFormat("memcpy destination too small".into()));
    }
    dst[..src.len()].copy_from_slice(src);
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5MM_memcpy(dst: &mut [u8], src: &[u8]) -> Result<()> {
    memcpy(dst, src)
}

pub fn realloc(mut buf: Vec<u8>, new_size: usize) -> Vec<u8> {
    buf.resize(new_size, 0);
    buf
}

pub fn xstrdup_ref(value: &str) -> &str {
    value
}

pub fn xstrdup_into(value: &str, out: &mut String) {
    out.clear();
    out.push_str(value);
}

pub fn strdup_ref(value: &str) -> &str {
    value
}

pub fn strdup_into(value: &str, out: &mut String) {
    xstrdup_into(value, out);
}

pub fn strndup_cow(value: &str, max_len: usize) -> Cow<'_, str> {
    let mut char_indices = value.char_indices();
    match char_indices.nth(max_len) {
        Some((end, _)) => Cow::Borrowed(&value[..end]),
        None => Cow::Borrowed(value),
    }
}

pub fn strndup_into(value: &str, max_len: usize, out: &mut String) {
    out.clear();
    out.push_str(&strndup_cow(value, max_len));
}

pub fn xfree_const<T>(_value: T) {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WrappedBuffer {
    actual: Vec<u8>,
}

impl WrappedBuffer {
    pub fn wrap(data: impl Into<Vec<u8>>) -> Self {
        Self {
            actual: data.into(),
        }
    }

    pub fn actual(&self) -> &[u8] {
        &self.actual
    }

    pub fn actual_clear(&mut self) {
        self.actual.clear();
    }

    pub fn into_inner(self) -> Vec<u8> {
        self.actual
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrapped_buffer_roundtrips() {
        let mut wb = WrappedBuffer::wrap(b"abc".to_vec());
        assert_eq!(wb.actual(), b"abc");
        wb.actual_clear();
        assert!(wb.actual().is_empty());
    }

    #[test]
    fn string_dup_helpers_can_borrow_or_reuse_storage() {
        assert!(matches!(strndup_cow("abc", 3), Cow::Borrowed("abc")));
        assert!(matches!(strndup_cow("abc", 2), Cow::Borrowed("ab")));
        assert_eq!(strndup_cow("abc", 2).as_ref(), "ab");

        let mut out = String::from("old");
        xstrdup_into("new", &mut out);
        assert_eq!(out, "new");
        strndup_into("abcdef", 4, &mut out);
        assert_eq!(out, "abcd");
    }
}
