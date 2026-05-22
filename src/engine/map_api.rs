use std::collections::BTreeMap;
use std::ops::ControlFlow;

use crate::error::{Error, Result};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct H5Map {
    key_type: String,
    value_type: String,
    create_plist: BTreeMap<String, Vec<u8>>,
    access_plist: BTreeMap<String, Vec<u8>>,
    entries: BTreeMap<Vec<u8>, Vec<u8>>,
    open: bool,
}

impl H5Map {
    pub fn new(key_type: impl Into<String>, value_type: impl Into<String>) -> Self {
        Self {
            key_type: key_type.into(),
            value_type: value_type.into(),
            create_plist: BTreeMap::new(),
            access_plist: BTreeMap::new(),
            entries: BTreeMap::new(),
            open: true,
        }
    }

    fn ensure_open(&self) -> Result<()> {
        if self.open {
            Ok(())
        } else {
            Err(Error::InvalidFormat("map handle is closed".into()))
        }
    }
}

pub trait H5MIterCallbackResult {
    fn should_stop(self) -> bool;
}

impl H5MIterCallbackResult for () {
    fn should_stop(self) -> bool {
        false
    }
}

impl H5MIterCallbackResult for ControlFlow<()> {
    fn should_stop(self) -> bool {
        matches!(self, ControlFlow::Break(()))
    }
}

#[allow(non_snake_case)]
pub fn H5M_init() -> bool {
    true
}

#[allow(non_snake_case)]
pub fn H5M__init_package() -> bool {
    H5M_init()
}

#[allow(non_snake_case)]
pub fn H5M_top_term_package() {}

#[allow(non_snake_case)]
pub fn H5M_term_package() {}

#[allow(non_snake_case)]
pub fn H5M__close_cb(map: &mut H5Map) {
    map.open = false;
}

#[allow(non_snake_case)]
pub fn H5M__create_api_common(key_type: impl Into<String>, value_type: impl Into<String>) -> H5Map {
    H5Map::new(key_type, value_type)
}

#[allow(non_snake_case)]
pub fn H5Mcreate(
    _loc: impl Into<String>,
    _name: impl Into<String>,
    key_type: impl Into<String>,
    value_type: impl Into<String>,
) -> H5Map {
    H5M__create_api_common(key_type, value_type)
}

#[allow(non_snake_case)]
pub fn H5Mcreate_async(
    loc: impl Into<String>,
    name: impl Into<String>,
    key_type: impl Into<String>,
    value_type: impl Into<String>,
) -> H5Map {
    H5Mcreate(loc, name, key_type, value_type)
}

#[allow(non_snake_case)]
pub fn H5Mcreate_anon(key_type: impl Into<String>, value_type: impl Into<String>) -> H5Map {
    H5M__create_api_common(key_type, value_type)
}

#[allow(non_snake_case)]
pub fn H5Mcreate_anon_async(key_type: impl Into<String>, value_type: impl Into<String>) -> H5Map {
    H5Mcreate_anon(key_type, value_type)
}

#[allow(non_snake_case)]
pub fn H5M__open_api_common_ref(map: &H5Map) -> Result<&H5Map> {
    map.ensure_open()?;
    Ok(map)
}

#[allow(non_snake_case)]
pub fn H5Mopen_ref<'a>(
    map: &'a H5Map,
    _loc: impl Into<String>,
    _name: impl Into<String>,
) -> Result<&'a H5Map> {
    H5M__open_api_common_ref(map)
}

#[allow(non_snake_case)]
pub fn H5Mopen_async_ref<'a>(
    map: &'a H5Map,
    loc: impl Into<String>,
    name: impl Into<String>,
) -> Result<&'a H5Map> {
    H5Mopen_ref(map, loc, name)
}

#[deprecated(note = "use H5M__open_api_common_ref to borrow the map without cloning")]
#[allow(non_snake_case)]
pub fn H5M__open_api_common(map: &H5Map) -> Result<H5Map> {
    Ok(H5M__open_api_common_ref(map)?.clone())
}

#[deprecated(note = "use H5Mopen_ref to borrow the map without cloning")]
#[allow(non_snake_case)]
#[allow(deprecated)]
pub fn H5Mopen(map: &H5Map, _loc: impl Into<String>, _name: impl Into<String>) -> Result<H5Map> {
    H5M__open_api_common(map)
}

#[deprecated(note = "use H5Mopen_async_ref to borrow the map without cloning")]
#[allow(non_snake_case)]
#[allow(deprecated)]
pub fn H5Mopen_async(
    map: &H5Map,
    loc: impl Into<String>,
    name: impl Into<String>,
) -> Result<H5Map> {
    H5Mopen(map, loc, name)
}

#[allow(non_snake_case)]
pub fn H5Mclose(map: &mut H5Map) {
    H5M__close_cb(map);
}

#[allow(non_snake_case)]
pub fn H5Mclose_async(map: &mut H5Map) {
    H5Mclose(map);
}

#[allow(non_snake_case)]
pub fn H5Mget_key_type(map: &H5Map) -> Result<&str> {
    map.ensure_open()?;
    Ok(&map.key_type)
}

#[allow(non_snake_case)]
pub fn H5Mget_val_type(map: &H5Map) -> Result<&str> {
    map.ensure_open()?;
    Ok(&map.value_type)
}

#[allow(non_snake_case)]
pub fn H5Mget_create_plist_ref(map: &H5Map) -> Result<&BTreeMap<String, Vec<u8>>> {
    map.ensure_open()?;
    Ok(&map.create_plist)
}

#[allow(non_snake_case)]
pub fn H5Mget_access_plist_ref(map: &H5Map) -> Result<&BTreeMap<String, Vec<u8>>> {
    map.ensure_open()?;
    Ok(&map.access_plist)
}

#[deprecated(note = "use H5Mget_create_plist_ref to borrow the property list without cloning")]
#[allow(non_snake_case)]
pub fn H5Mget_create_plist(map: &H5Map) -> Result<BTreeMap<String, Vec<u8>>> {
    Ok(H5Mget_create_plist_ref(map)?.clone())
}

#[deprecated(note = "use H5Mget_access_plist_ref to borrow the property list without cloning")]
#[allow(non_snake_case)]
pub fn H5Mget_access_plist(map: &H5Map) -> Result<BTreeMap<String, Vec<u8>>> {
    Ok(H5Mget_access_plist_ref(map)?.clone())
}

#[allow(non_snake_case)]
pub fn H5Mget_count(map: &H5Map) -> Result<usize> {
    map.ensure_open()?;
    Ok(map.entries.len())
}

#[allow(non_snake_case)]
pub fn H5M__put_api_common(map: &mut H5Map, key: Vec<u8>, value: Vec<u8>) -> Result<()> {
    map.ensure_open()?;
    map.entries.insert(key, value);
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5Mput(map: &mut H5Map, key: Vec<u8>, value: Vec<u8>) -> Result<()> {
    H5M__put_api_common(map, key, value)
}

#[allow(non_snake_case)]
pub fn H5Mput_async(map: &mut H5Map, key: Vec<u8>, value: Vec<u8>) -> Result<()> {
    H5Mput(map, key, value)
}

#[allow(non_snake_case)]
pub fn H5M__get_api_common_ref<'a>(map: &'a H5Map, key: &[u8]) -> Result<Option<&'a [u8]>> {
    map.ensure_open()?;
    Ok(map.entries.get(key).map(Vec::as_slice))
}

#[allow(non_snake_case)]
pub fn H5Mget_ref<'a>(map: &'a H5Map, key: &[u8]) -> Result<Option<&'a [u8]>> {
    H5M__get_api_common_ref(map, key)
}

#[allow(non_snake_case)]
pub fn H5Mget_async_ref<'a>(map: &'a H5Map, key: &[u8]) -> Result<Option<&'a [u8]>> {
    H5Mget_ref(map, key)
}

#[deprecated(note = "use H5M__get_api_common_ref to borrow the value without cloning")]
#[allow(non_snake_case)]
pub fn H5M__get_api_common(map: &H5Map, key: &[u8]) -> Result<Option<Vec<u8>>> {
    Ok(H5M__get_api_common_ref(map, key)?.map(<[u8]>::to_vec))
}

#[deprecated(note = "use H5Mget_ref to borrow the value without cloning")]
#[allow(non_snake_case)]
pub fn H5Mget(map: &H5Map, key: &[u8]) -> Result<Option<Vec<u8>>> {
    Ok(H5Mget_ref(map, key)?.map(<[u8]>::to_vec))
}

#[deprecated(note = "use H5Mget_async_ref to borrow the value without cloning")]
#[allow(non_snake_case)]
pub fn H5Mget_async(map: &H5Map, key: &[u8]) -> Result<Option<Vec<u8>>> {
    Ok(H5Mget_async_ref(map, key)?.map(<[u8]>::to_vec))
}

#[allow(non_snake_case)]
pub fn H5Mexists(map: &H5Map, key: &[u8]) -> Result<bool> {
    map.ensure_open()?;
    Ok(map.entries.contains_key(key))
}

#[allow(non_snake_case)]
pub fn H5M_iter_entries(map: &H5Map) -> Result<impl Iterator<Item = (&[u8], &[u8])> + '_> {
    map.ensure_open()?;
    Ok(map
        .entries
        .iter()
        .map(|(key, value)| (key.as_slice(), value.as_slice())))
}

#[allow(non_snake_case)]
pub fn H5M_iterate_with<F, R>(map: &H5Map, mut callback: F) -> Result<()>
where
    F: FnMut(&[u8], &[u8]) -> Result<R>,
    R: H5MIterCallbackResult,
{
    map.ensure_open()?;
    for (key, value) in &map.entries {
        if callback(key, value)?.should_stop() {
            break;
        }
    }
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5Miterate_with<F, R>(map: &H5Map, callback: F) -> Result<()>
where
    F: FnMut(&[u8], &[u8]) -> Result<R>,
    R: H5MIterCallbackResult,
{
    H5M_iterate_with(map, callback)
}

#[allow(non_snake_case)]
pub fn H5M_iterate_into(map: &H5Map, entries: &mut Vec<(Vec<u8>, Vec<u8>)>) -> Result<()> {
    map.ensure_open()?;
    entries.reserve(map.entries.len());
    H5M_iterate_with(map, |key, value| {
        entries.push((key.to_vec(), value.to_vec()));
        Ok(())
    })
}

#[allow(non_snake_case)]
pub fn H5Miterate_into(map: &H5Map, entries: &mut Vec<(Vec<u8>, Vec<u8>)>) -> Result<()> {
    H5M_iterate_into(map, entries)
}

#[deprecated(
    note = "use H5M_iter_entries, H5M_iterate_with, or H5M_iterate_into to avoid returning an allocated Vec"
)]
#[allow(non_snake_case)]
pub fn H5M_iterate(map: &H5Map) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
    let mut entries = Vec::new();
    H5M_iterate_into(map, &mut entries)?;
    Ok(entries)
}

#[deprecated(
    note = "use H5M_iter_entries, H5Miterate_with, or H5Miterate_into to avoid returning an allocated Vec"
)]
#[allow(non_snake_case)]
pub fn H5Miterate(map: &H5Map) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
    let mut entries = Vec::new();
    H5Miterate_into(map, &mut entries)?;
    Ok(entries)
}

#[deprecated(
    note = "use H5M_iter_entries, H5Miterate_with, or H5Miterate_into to avoid returning an allocated Vec"
)]
#[allow(non_snake_case)]
pub fn H5Miterate_by_name(map: &H5Map) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
    let mut entries = Vec::new();
    H5Miterate_into(map, &mut entries)?;
    Ok(entries)
}

#[allow(non_snake_case)]
pub fn H5Mdelete(map: &mut H5Map, key: &[u8]) -> Result<Option<Vec<u8>>> {
    map.ensure_open()?;
    Ok(map.entries.remove(key))
}

#[allow(non_snake_case)]
pub fn H5Mdelete_async(map: &mut H5Map, key: &[u8]) -> Result<Option<Vec<u8>>> {
    H5Mdelete(map, key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_api_put_get_iterate_delete() {
        let mut map = H5Mcreate("file", "map", "u8", "bytes");
        assert_eq!(H5Mget_key_type(&map).unwrap(), "u8");
        assert_eq!(H5Mget_val_type(&map).unwrap(), "bytes");
        H5Mput(&mut map, b"k".to_vec(), b"v".to_vec()).unwrap();
        assert_eq!(H5Mget_count(&map).unwrap(), 1);
        assert!(H5Mexists(&map, b"k").unwrap());
        assert!(std::ptr::eq(H5M__open_api_common_ref(&map).unwrap(), &map));
        assert!(std::ptr::eq(
            H5Mopen_ref(&map, "file", "map").unwrap(),
            &map
        ));
        assert!(std::ptr::eq(
            H5Mopen_async_ref(&map, "file", "map").unwrap(),
            &map
        ));
        assert_eq!(H5Mget_ref(&map, b"k").unwrap(), Some(b"v".as_slice()));
        let entries: Vec<_> = H5M_iter_entries(&map).unwrap().collect();
        assert_eq!(entries, vec![(b"k".as_slice(), b"v".as_slice())]);

        let mut visited = Vec::new();
        H5Miterate_with(&map, |key, value| {
            visited.push((key.to_vec(), value.to_vec()));
            Ok(())
        })
        .unwrap();
        assert_eq!(visited, vec![(b"k".to_vec(), b"v".to_vec())]);

        let mut copied = Vec::new();
        H5M_iterate_into(&map, &mut copied).unwrap();
        assert_eq!(copied, vec![(b"k".to_vec(), b"v".to_vec())]);

        H5Mput(&mut map, b"k2".to_vec(), b"v2".to_vec()).unwrap();
        let mut stopped = Vec::new();
        H5Miterate_with(&map, |key, value| {
            stopped.push((key.to_vec(), value.to_vec()));
            Ok(ControlFlow::Break(()))
        })
        .unwrap();
        assert_eq!(stopped, vec![(b"k".to_vec(), b"v".to_vec())]);
        assert_eq!(H5Mdelete(&mut map, b"k2").unwrap(), Some(b"v2".to_vec()));

        assert_eq!(H5Mdelete(&mut map, b"k").unwrap(), Some(b"v".to_vec()));
        H5Mput_async(&mut map, b"k2".to_vec(), b"v2".to_vec()).unwrap();
        assert_eq!(
            H5Mdelete_async(&mut map, b"k2").unwrap(),
            Some(b"v2".to_vec())
        );
        H5Mclose_async(&mut map);
        assert!(H5Mget_count(&map).is_err());

        let anon = H5Mcreate_anon_async("u8", "bytes");
        assert_eq!(H5Mget_count(&anon).unwrap(), 0);
    }
}
