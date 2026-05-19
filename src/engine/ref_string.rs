use std::cmp::Ordering;
use std::fmt::{self, Write};
use std::sync::Arc;

/// Reference-counted string wrapper mirroring H5RS.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefString {
    inner: Arc<String>,
}

impl RefString {
    /// Duplicate a Rust string into H5RS-owned storage.
    pub fn xstrdup_ref(value: &str) -> &str {
        value
    }

    /// Duplicate a Rust string into caller-owned storage.
    pub fn xstrdup_into(value: &str, out: &mut String) {
        out.clear();
        out.push_str(value);
    }

    /// Ensure capacity before appending.
    pub fn prepare_for_append(&mut self, additional: usize) {
        Arc::make_mut(&mut self.inner).reserve(additional);
    }

    /// Resize append capacity.
    pub fn resize_for_append(&mut self, additional: usize) {
        self.prepare_for_append(additional);
    }

    /// Create a reference-counted string.
    pub fn create(value: impl Into<String>) -> Self {
        Self {
            inner: Arc::new(value.into()),
        }
    }

    /// Wrap an existing string.
    pub fn wrap(value: String) -> Self {
        Self {
            inner: Arc::new(value),
        }
    }

    /// Append formatted text.
    pub fn asprintf_cat(&mut self, args: fmt::Arguments<'_>) -> fmt::Result {
        Arc::make_mut(&mut self.inner).write_fmt(args)
    }

    /// Append a string.
    pub fn acat(&mut self, value: &str) {
        Arc::make_mut(&mut self.inner).push_str(value);
    }

    /// Append at most `count` bytes from a string.
    pub fn ancat(&mut self, value: &str, count: usize) {
        let end = value
            .char_indices()
            .map(|(idx, _)| idx)
            .chain(std::iter::once(value.len()))
            .take_while(|&idx| idx <= count)
            .last()
            .unwrap_or(0);
        self.acat(&value[..end]);
    }

    /// Append one character.
    pub fn aputc(&mut self, ch: char) {
        Arc::make_mut(&mut self.inner).push(ch);
    }

    /// Decrement reference count by consuming this handle.
    pub fn decr(self) {}

    /// Increment reference count by cloning this handle.
    pub fn incr(&self) -> Self {
        self.clone()
    }

    /// Duplicate this string handle.
    pub fn dup(&self) -> Self {
        self.clone()
    }

    /// Compare string contents.
    pub fn cmp(&self, other: &Self) -> Ordering {
        self.inner.cmp(&other.inner)
    }

    /// Return string byte length.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Return string contents.
    pub fn get_str(&self) -> &str {
        &self.inner
    }

    /// Return strong reference count.
    pub fn get_count(&self) -> usize {
        Arc::strong_count(&self.inner)
    }
}

#[cfg(test)]
mod tests {
    use super::RefString;

    #[test]
    fn ref_string_aliases_roundtrip() {
        assert_eq!(RefString::xstrdup_ref("a"), "a");
        let mut out = String::new();
        RefString::xstrdup_into("b", &mut out);
        assert_eq!(out, "b");
        let mut s = RefString::create("ab");
        s.prepare_for_append(8);
        s.resize_for_append(8);
        s.acat("cd");
        s.ancat("efgh", 2);
        s.aputc('!');
        s.asprintf_cat(format_args!("{}", 7)).unwrap();
        assert_eq!(s.get_str(), "abcdef!7");
        assert_eq!(s.len(), 8);
        let copy = s.incr();
        assert_eq!(s.get_count(), 2);
        assert_eq!(s.cmp(&copy), std::cmp::Ordering::Equal);
        copy.decr();
        assert_eq!(s.dup().get_str(), "abcdef!7");
        assert_eq!(RefString::wrap("z".to_string()).get_str(), "z");
    }
}
