use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PluginCache {
    plugins: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PluginPathTable {
    paths: Vec<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginRegistry {
    cache: PluginCache,
    paths: PluginPathTable,
    control_mask: u64,
    open_plugins: BTreeMap<String, usize>,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum H5PLPluginType {
    Error = -1,
    Filter = 0,
    Vol = 1,
    Vfd = 2,
    None = 3,
}

pub const H5PL_FILTER_PLUGIN: u64 = 0x0001;
pub const H5PL_VOL_PLUGIN: u64 = 0x0002;
pub const H5PL_VFD_PLUGIN: u64 = 0x0004;
pub const H5PL_ALL_PLUGIN: u64 = 0xffff;
pub const H5PL_NO_PLUGIN: &str = "::";
pub const H5PL_TYPE_ERROR: H5PLPluginType = H5PLPluginType::Error;
pub const H5PL_TYPE_FILTER: H5PLPluginType = H5PLPluginType::Filter;
pub const H5PL_TYPE_VOL: H5PLPluginType = H5PLPluginType::Vol;
pub const H5PL_TYPE_VFD: H5PLPluginType = H5PLPluginType::Vfd;
pub const H5PL_TYPE_NONE: H5PLPluginType = H5PLPluginType::None;

impl Default for PluginRegistry {
    fn default() -> Self {
        Self {
            cache: PluginCache::default(),
            paths: PluginPathTable::default(),
            control_mask: H5PL_ALL_PLUGIN,
            open_plugins: BTreeMap::new(),
        }
    }
}

#[allow(non_snake_case)]
pub fn H5PL__create_plugin_cache() -> PluginCache {
    PluginCache::default()
}

#[allow(non_snake_case)]
pub fn H5PL__close_plugin_cache(cache: &mut PluginCache) {
    cache.plugins.clear();
}

#[allow(non_snake_case)]
pub fn H5PL__expand_cache(cache: &mut PluginCache, additional: usize) {
    cache.plugins.reserve(additional);
}

#[allow(non_snake_case)]
pub fn H5PL__add_plugin_ref(cache: &mut PluginCache, name: &str) {
    if !cache.plugins.iter().any(|plugin| plugin == name) {
        cache.plugins.push(name.to_owned());
    }
}

#[allow(non_snake_case)]
pub fn H5PL__add_plugin_owned(cache: &mut PluginCache, name: String) {
    if !cache.plugins.contains(&name) {
        cache.plugins.push(name);
    }
}

#[allow(non_snake_case)]
pub fn H5PL__plugin_cache_iter(cache: &PluginCache) -> impl Iterator<Item = &str> {
    cache.plugins.iter().map(String::as_str)
}

#[allow(non_snake_case)]
pub fn H5PL__plugin_cache_iterate_with<F>(cache: &PluginCache, mut callback: F)
where
    F: FnMut(&str),
{
    for plugin in H5PL__plugin_cache_iter(cache) {
        callback(plugin);
    }
}

#[allow(non_snake_case)]
pub fn H5PL__get_plugin_control_mask(registry: &PluginRegistry) -> u64 {
    registry.control_mask
}

#[allow(non_snake_case)]
pub fn H5PL__set_plugin_control_mask(registry: &mut PluginRegistry, mask: u64) {
    registry.control_mask = mask;
}

#[allow(non_snake_case)]
pub fn H5PL__init_package() -> PluginRegistry {
    PluginRegistry::default()
}

#[allow(non_snake_case)]
pub fn H5PL_init() -> PluginRegistry {
    H5PL__init_package()
}

#[allow(non_snake_case)]
pub fn H5PL_term_package(registry: &mut PluginRegistry) {
    H5PL__close_plugin_cache(&mut registry.cache);
    H5PL__close_path_table(&mut registry.paths);
    registry.open_plugins.clear();
}

fn load_cached_plugin(registry: &mut PluginRegistry, name: &str) -> Result<()> {
    if registry.control_mask == 0 {
        return Err(Error::Unsupported("plugin loading is disabled".into()));
    }
    if registry.cache.plugins.iter().any(|plugin| plugin == name) {
        if let Some(count) = registry.open_plugins.get_mut(name) {
            *count += 1;
        } else {
            registry.open_plugins.insert(name.to_owned(), 1);
        }
        Ok(())
    } else {
        Err(Error::Unsupported(format!(
            "dynamic plugin loading is not supported for '{name}'"
        )))
    }
}

/// `H5PL_load`: package-level libhdf5 dynamic plugin lookup by plugin type and
/// identifier. The pure-Rust backend does not `dlopen` external plugin
/// libraries.
#[allow(non_snake_case)]
pub fn H5PL_load(plugin_type: H5PLPluginType, plugin_id: i32) -> Result<()> {
    Err(Error::Unsupported(format!(
        "dynamic HDF5 plugin lookup via H5PL_load is unsupported in pure-Rust mode (type {plugin_type:?}, id {plugin_id})"
    )))
}

#[allow(non_snake_case)]
pub fn H5PL_load_owned(registry: &mut PluginRegistry, name: String) -> Result<()> {
    load_cached_plugin(registry, &name)
}

/// `H5PLget_plugin_type`: external dynamic-plugin entry point, not a
/// host-library query in this pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5PLget_plugin_type() -> Result<H5PLPluginType> {
    Err(Error::Unsupported(
        "external HDF5 plugin entry point H5PLget_plugin_type is unsupported in pure-Rust mode"
            .into(),
    ))
}

#[allow(non_snake_case)]
pub fn H5PLget_plugin_type_into(plugin_type: &mut H5PLPluginType) -> Result<()> {
    *plugin_type = H5PLget_plugin_type()?;
    Ok(())
}

/// `H5PLget_plugin_info`: external dynamic-plugin entry point, not a
/// host-library query in this pure-Rust backend.
#[allow(non_snake_case)]
pub fn H5PLget_plugin_info() -> Result<()> {
    Err(Error::Unsupported(
        "external HDF5 plugin entry point H5PLget_plugin_info is unsupported in pure-Rust mode"
            .into(),
    ))
}

#[allow(non_snake_case)]
pub fn H5PL__open(registry: &mut PluginRegistry, name: &str) -> Result<()> {
    load_cached_plugin(registry, name)
}

#[allow(non_snake_case)]
pub fn H5PL__close(registry: &mut PluginRegistry, name: &str) -> Result<()> {
    let Some(count) = registry.open_plugins.get_mut(name) else {
        return Err(Error::InvalidFormat(format!("plugin '{name}' is not open")));
    };
    *count = count.saturating_sub(1);
    if *count == 0 {
        registry.open_plugins.remove(name);
    }
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5PL_iterate_into(registry: &PluginRegistry, plugins: &mut Vec<String>) {
    plugins.clear();
    plugins.extend(H5PL_iter_plugins(registry).map(str::to_owned));
}

#[allow(non_snake_case)]
pub fn H5PL_iter_plugins(registry: &PluginRegistry) -> impl Iterator<Item = &str> {
    H5PL__plugin_cache_iter(&registry.cache)
}

#[allow(non_snake_case)]
pub fn H5PL_iterate_with<F>(registry: &PluginRegistry, callback: F)
where
    F: FnMut(&str),
{
    H5PL__plugin_cache_iterate_with(&registry.cache, callback);
}

#[allow(non_snake_case)]
pub fn H5PL__insert_at(
    cache: &mut PluginCache,
    index: usize,
    name: impl Into<String>,
) -> Result<()> {
    H5PL__make_space_at(cache, index)?;
    cache.plugins[index] = name.into();
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5PL__make_space_at(cache: &mut PluginCache, index: usize) -> Result<()> {
    if index > cache.plugins.len() {
        return Err(Error::InvalidFormat(format!(
            "plugin cache index {index} out of range"
        )));
    }
    cache.plugins.insert(index, String::new());
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5PL__replace_at(
    cache: &mut PluginCache,
    index: usize,
    name: impl Into<String>,
) -> Result<()> {
    let Some(slot) = cache.plugins.get_mut(index) else {
        return Err(Error::InvalidFormat(format!(
            "plugin cache index {index} out of range"
        )));
    };
    *slot = name.into();
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5PL__create_path_table() -> PluginPathTable {
    PluginPathTable::default()
}

#[allow(non_snake_case)]
pub fn H5PL__close_path_table(table: &mut PluginPathTable) {
    table.paths.clear();
}

#[allow(non_snake_case)]
pub fn H5PL__get_num_paths(table: &PluginPathTable) -> usize {
    table.paths.len()
}

#[allow(non_snake_case)]
pub fn H5PL__expand_path_table(table: &mut PluginPathTable, additional: usize) {
    table.paths.reserve(additional);
}

#[allow(non_snake_case)]
pub fn H5PL__append_path(table: &mut PluginPathTable, path: impl Into<PathBuf>) {
    table.paths.push(path.into());
}

#[allow(non_snake_case)]
pub fn H5PL__prepend_path(table: &mut PluginPathTable, path: impl Into<PathBuf>) {
    table.paths.insert(0, path.into());
}

#[allow(non_snake_case)]
pub fn H5PL__replace_path(
    table: &mut PluginPathTable,
    index: usize,
    path: impl Into<PathBuf>,
) -> Result<()> {
    let Some(slot) = table.paths.get_mut(index) else {
        return Err(Error::InvalidFormat(format!(
            "plugin path index {index} out of range"
        )));
    };
    *slot = path.into();
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5PL__insert_path(
    table: &mut PluginPathTable,
    index: usize,
    path: impl Into<PathBuf>,
) -> Result<()> {
    if index > table.paths.len() {
        return Err(Error::InvalidFormat(format!(
            "plugin path index {index} out of range"
        )));
    }
    table.paths.insert(index, path.into());
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5PL__remove_path_into(
    table: &mut PluginPathTable,
    index: usize,
    removed: &mut PathBuf,
) -> Result<()> {
    if index >= table.paths.len() {
        return Err(Error::InvalidFormat(format!(
            "plugin path index {index} out of range"
        )));
    }
    *removed = table.paths.remove(index);
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5PL__get_path(table: &PluginPathTable, index: usize) -> Result<&Path> {
    table
        .paths
        .get(index)
        .map(PathBuf::as_path)
        .ok_or_else(|| Error::InvalidFormat(format!("plugin path index {index} out of range")))
}

#[allow(non_snake_case)]
pub fn H5PL__path_table_paths(table: &PluginPathTable) -> impl Iterator<Item = &Path> {
    table.paths.iter().map(PathBuf::as_path)
}

#[allow(non_snake_case)]
pub fn H5PL__path_table_iterate_with<F>(table: &PluginPathTable, mut callback: F)
where
    F: FnMut(&Path),
{
    for path in H5PL__path_table_paths(table) {
        callback(path);
    }
}

#[allow(non_snake_case)]
pub fn H5PL_iterate<F>(registry: &PluginRegistry, callback: F)
where
    F: FnMut(&Path),
{
    H5PL__path_table_iterate_with(&registry.paths, callback);
}

#[allow(non_snake_case)]
pub fn H5PL__path_table_iterate_into(table: &PluginPathTable, paths: &mut Vec<PathBuf>) {
    paths.clear();
    paths.extend(H5PL__path_table_paths(table).map(Path::to_path_buf));
}

#[allow(non_snake_case)]
pub fn H5PL__find_plugin_in_path_table_into(
    table: &PluginPathTable,
    name: &str,
    candidate: &mut PathBuf,
) -> bool {
    H5PL__path_table_paths(table).any(|path| H5PL__find_plugin_in_path_into(path, name, candidate))
}

#[allow(non_snake_case)]
pub fn H5PL__find_plugin_in_path_table(table: &PluginPathTable, name: &str) -> Option<PathBuf> {
    let mut candidate = PathBuf::new();
    H5PL__find_plugin_in_path_table_into(table, name, &mut candidate).then_some(candidate)
}

#[allow(non_snake_case)]
pub fn H5PL__find_plugin_in_path_into(path: &Path, name: &str, candidate: &mut PathBuf) -> bool {
    candidate.clear();
    candidate.push(path);
    candidate.push(name);
    candidate.exists()
}

#[allow(non_snake_case)]
pub fn H5PL__find_plugin_in_path(path: &Path, name: &str) -> Option<PathBuf> {
    let mut candidate = PathBuf::new();
    H5PL__find_plugin_in_path_into(path, name, &mut candidate).then_some(candidate)
}

#[allow(non_snake_case)]
pub fn H5PLset_loading_state(registry: &mut PluginRegistry, plugin_control_mask: u64) {
    H5PL__set_plugin_control_mask(registry, plugin_control_mask);
}

#[allow(non_snake_case)]
pub fn H5PLset_loading_state_i32(registry: &mut PluginRegistry, plugin_flags: i32) -> Result<()> {
    let mask = if plugin_flags < 0 {
        H5PL_ALL_PLUGIN
    } else {
        plugin_flags as u64
    };
    H5PLset_loading_state(registry, mask);
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5PLget_loading_state(registry: &PluginRegistry) -> u64 {
    H5PL__get_plugin_control_mask(registry)
}

#[allow(non_snake_case)]
pub fn H5PLget_loading_state_i32_into(
    registry: &PluginRegistry,
    plugin_flags: &mut i32,
) -> Result<()> {
    let mask = H5PLget_loading_state(registry);
    if mask > i32::MAX as u64 {
        return Err(Error::InvalidFormat(format!(
            "plugin loading state mask {mask} does not fit in i32"
        )));
    }
    *plugin_flags = mask as i32;
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5PLget_loading_state_u32_into(
    registry: &PluginRegistry,
    plugin_control_mask: &mut u32,
) -> Result<()> {
    let mask = H5PLget_loading_state(registry);
    if mask > u32::MAX as u64 {
        return Err(Error::InvalidFormat(format!(
            "plugin loading state mask {mask} does not fit in u32"
        )));
    }
    *plugin_control_mask = mask as u32;
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5PLget_loading_state_into(
    registry: &PluginRegistry,
    plugin_control_mask: &mut u64,
) -> Result<()> {
    *plugin_control_mask = H5PLget_loading_state(registry);
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5PLappend(registry: &mut PluginRegistry, path: impl Into<PathBuf>) {
    H5PL__append_path(&mut registry.paths, path);
}

#[allow(non_snake_case)]
pub fn H5PLprepend(registry: &mut PluginRegistry, path: impl Into<PathBuf>) {
    H5PL__prepend_path(&mut registry.paths, path);
}

#[allow(non_snake_case)]
pub fn H5PLreplace(
    registry: &mut PluginRegistry,
    index: usize,
    path: impl Into<PathBuf>,
) -> Result<()> {
    H5PL__replace_path(&mut registry.paths, index, path)
}

#[allow(non_snake_case)]
pub fn H5PLinsert(
    registry: &mut PluginRegistry,
    index: usize,
    path: impl Into<PathBuf>,
) -> Result<()> {
    H5PL__insert_path(&mut registry.paths, index, path)
}

#[allow(non_snake_case)]
#[deprecated(note = "use H5PLremove_into with caller-provided PathBuf storage")]
pub fn H5PLremove(registry: &mut PluginRegistry, index: usize) -> Result<PathBuf> {
    let mut removed = PathBuf::new();
    H5PLremove_into(registry, index, &mut removed)?;
    Ok(removed)
}

#[allow(non_snake_case)]
pub fn H5PLremove_into(
    registry: &mut PluginRegistry,
    index: usize,
    removed: &mut PathBuf,
) -> Result<()> {
    H5PL__remove_path_into(&mut registry.paths, index, removed)
}

#[allow(non_snake_case)]
pub fn H5PLget(registry: &PluginRegistry, index: usize) -> Result<&Path> {
    H5PL__get_path(&registry.paths, index)
}

#[allow(non_snake_case)]
pub fn H5PLget_into(registry: &PluginRegistry, index: usize, path: &mut PathBuf) -> Result<()> {
    *path = H5PLget(registry, index)?.to_path_buf();
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5PLget_str_into(
    registry: &PluginRegistry,
    index: usize,
    out: &mut String,
) -> Result<usize> {
    let path = H5PLget(registry, index)?;
    let Some(path) = path.to_str() else {
        return Err(Error::InvalidFormat(
            "plugin path is not valid UTF-8".into(),
        ));
    };
    out.clear();
    out.push_str(path);
    Ok(out.len())
}

/// C-style `H5PLget` compatibility helper: writes at most `buf_size` bytes
/// into caller-owned storage and returns the full path length.
#[allow(non_snake_case)]
pub fn H5PLget_buf_into(
    registry: &PluginRegistry,
    index: usize,
    out: &mut Vec<u8>,
    buf_size: usize,
) -> Result<usize> {
    let path = H5PLget(registry, index)?;
    let Some(path) = path.to_str() else {
        return Err(Error::InvalidFormat(
            "plugin path is not valid UTF-8".into(),
        ));
    };
    let bytes = path.as_bytes();
    out.clear();
    out.extend_from_slice(&bytes[..bytes.len().min(buf_size)]);
    Ok(bytes.len())
}

#[allow(non_snake_case)]
pub fn H5PLsize(registry: &PluginRegistry) -> usize {
    H5PL__get_num_paths(&registry.paths)
}

#[allow(non_snake_case)]
pub fn H5PLsize_into(registry: &PluginRegistry, num_paths: &mut usize) -> Result<()> {
    *num_paths = H5PLsize(registry);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plugin_path_table_mutates_order() {
        let mut registry = H5PL__init_package();
        H5PLappend(&mut registry, "/b");
        H5PLprepend(&mut registry, "/a");
        H5PLinsert(&mut registry, 1, "/mid").unwrap();
        assert_eq!(H5PLget_loading_state(&registry), H5PL_ALL_PLUGIN);
        H5PLset_loading_state(&mut registry, H5PL_FILTER_PLUGIN);
        assert_eq!(H5PLget_loading_state(&registry), H5PL_FILTER_PLUGIN);
        let mut loading_state = H5PL_NO_PLUGIN.len() as u64;
        H5PLget_loading_state_into(&registry, &mut loading_state).unwrap();
        assert_eq!(loading_state, H5PL_FILTER_PLUGIN);
        assert_eq!(H5PLsize(&registry), 3);
        assert_eq!(H5PLget(&registry, 1).unwrap(), Path::new("/mid"));
        let mut path = PathBuf::new();
        H5PLget_into(&registry, 1, &mut path).unwrap();
        assert_eq!(path, PathBuf::from("/mid"));
        let mut path_str = String::from("stale");
        assert_eq!(H5PLget_str_into(&registry, 1, &mut path_str).unwrap(), 4);
        assert_eq!(path_str, "/mid");
        let mut path_buf = b"stale".to_vec();
        assert_eq!(H5PLget_buf_into(&registry, 1, &mut path_buf, 2).unwrap(), 4);
        assert_eq!(path_buf, b"/m");
        let mut num_paths = 0;
        H5PLsize_into(&registry, &mut num_paths).unwrap();
        assert_eq!(num_paths, 3);
        assert_eq!(
            H5PL__path_table_paths(&registry.paths).collect::<Vec<_>>(),
            vec![Path::new("/a"), Path::new("/mid"), Path::new("/b")]
        );
        let mut removed = PathBuf::new();
        H5PLremove_into(&mut registry, 1, &mut removed).unwrap();
        assert_eq!(removed, PathBuf::from("/mid"));
    }

    #[test]
    fn plugin_load_is_explicitly_unsupported_without_cached_plugin() {
        let mut registry = H5PL__init_package();
        let err = H5PL__open(&mut registry, "missing").unwrap_err();
        assert_eq!(
            err.to_string(),
            "Unsupported: dynamic plugin loading is not supported for 'missing'"
        );

        H5PL__add_plugin_ref(&mut registry.cache, "known");
        H5PL__open(&mut registry, "known").unwrap();
        H5PL_load_owned(&mut registry, "known".to_owned()).unwrap();
        assert_eq!(registry.open_plugins.get("known"), Some(&2));
        H5PL__close(&mut registry, "known").unwrap();
        H5PL__close(&mut registry, "known").unwrap();

        H5PLset_loading_state(&mut registry, 0);
        let err = H5PL__open(&mut registry, "known").unwrap_err();
        assert_eq!(err.to_string(), "Unsupported: plugin loading is disabled");
        let err = H5PL_load_owned(&mut registry, "known".to_owned()).unwrap_err();
        assert_eq!(err.to_string(), "Unsupported: plugin loading is disabled");
    }

    #[test]
    fn external_plugin_entry_points_are_explicitly_unsupported() {
        assert_eq!(H5PL_FILTER_PLUGIN, 0x0001);
        assert_eq!(H5PL_VOL_PLUGIN, 0x0002);
        assert_eq!(H5PL_VFD_PLUGIN, 0x0004);
        assert_eq!(H5PL_ALL_PLUGIN, 0xffff);
        assert_eq!(H5PL_NO_PLUGIN, "::");
        assert_eq!(H5PLPluginType::Error as isize, -1);
        assert_eq!(H5PLPluginType::Filter as isize, 0);
        assert_eq!(H5PLPluginType::Vol as isize, 1);
        assert_eq!(H5PLPluginType::Vfd as isize, 2);
        assert_eq!(H5PLPluginType::None as isize, 3);
        assert_eq!(H5PL_TYPE_ERROR as isize, -1);
        assert_eq!(H5PL_TYPE_FILTER as isize, 0);
        assert_eq!(H5PL_TYPE_VOL as isize, 1);
        assert_eq!(H5PL_TYPE_VFD as isize, 2);
        assert_eq!(H5PL_TYPE_NONE as isize, 3);

        let err = H5PLget_plugin_type().unwrap_err();
        assert_eq!(
            err.to_string(),
            "Unsupported: external HDF5 plugin entry point H5PLget_plugin_type is unsupported in pure-Rust mode"
        );
        let mut plugin_type = H5PL_TYPE_VOL;
        let err = H5PLget_plugin_type_into(&mut plugin_type).unwrap_err();
        assert_eq!(
            err.to_string(),
            "Unsupported: external HDF5 plugin entry point H5PLget_plugin_type is unsupported in pure-Rust mode"
        );
        assert_eq!(plugin_type, H5PL_TYPE_VOL);

        let err = H5PLget_plugin_info().unwrap_err();
        assert_eq!(
            err.to_string(),
            "Unsupported: external HDF5 plugin entry point H5PLget_plugin_info is unsupported in pure-Rust mode"
        );

        let err = H5PL_load(H5PL_TYPE_FILTER, 32_015).unwrap_err();
        assert_eq!(
            err.to_string(),
            "Unsupported: dynamic HDF5 plugin lookup via H5PL_load is unsupported in pure-Rust mode (type Filter, id 32015)"
        );
    }

    #[test]
    fn plugin_iterators_and_path_search_use_borrowed_or_caller_storage() {
        let mut registry = H5PL__init_package();
        H5PL__add_plugin_ref(&mut registry.cache, "known");
        H5PL__add_plugin_owned(&mut registry.cache, "owned".to_owned());
        assert_eq!(
            H5PL_iter_plugins(&registry).collect::<Vec<_>>(),
            vec!["known", "owned"]
        );

        let mut plugin_names = vec!["stale".to_string()];
        H5PL_iterate_into(&registry, &mut plugin_names);
        assert_eq!(plugin_names, vec!["known", "owned"]);

        let mut visited = Vec::new();
        H5PL_iterate_with(&registry, |plugin| visited.push(plugin.to_owned()));
        assert_eq!(visited, vec!["known", "owned"]);

        H5PLappend(&mut registry, "/tmp");
        let mut iterated_paths = Vec::new();
        H5PL_iterate(&registry, |path| iterated_paths.push(path.to_path_buf()));
        assert_eq!(iterated_paths, vec![PathBuf::from("/tmp")]);

        let mut path_count = 0;
        H5PL__path_table_iterate_with(&registry.paths, |path| {
            assert_eq!(path, Path::new("/tmp"));
            path_count += 1;
        });
        assert_eq!(path_count, 1);

        let mut paths = vec![PathBuf::from("stale")];
        H5PL__path_table_iterate_into(&registry.paths, &mut paths);
        assert_eq!(paths, vec![PathBuf::from("/tmp")]);

        let mut candidate = PathBuf::from("stale");
        assert!(!H5PL__find_plugin_in_path_table_into(
            &registry.paths,
            "definitely-missing-hdf5-plugin",
            &mut candidate
        ));
        assert_eq!(
            candidate,
            PathBuf::from("/tmp/definitely-missing-hdf5-plugin")
        );
    }

    #[test]
    fn signed_loading_state_helpers_match_c_flag_semantics() {
        let mut registry = H5PL__init_package();
        H5PLset_loading_state_i32(&mut registry, H5PL_FILTER_PLUGIN as i32).unwrap();
        assert_eq!(H5PLget_loading_state(&registry), H5PL_FILTER_PLUGIN);

        let mut flags = -99;
        H5PLget_loading_state_i32_into(&registry, &mut flags).unwrap();
        assert_eq!(flags, H5PL_FILTER_PLUGIN as i32);

        H5PLset_loading_state_i32(&mut registry, -1).unwrap();
        assert_eq!(H5PLget_loading_state(&registry), H5PL_ALL_PLUGIN);
        H5PLget_loading_state_i32_into(&registry, &mut flags).unwrap();
        assert_eq!(flags, H5PL_ALL_PLUGIN as i32);

        H5PLset_loading_state(&mut registry, i32::MAX as u64 + 1);
        let err = H5PLget_loading_state_i32_into(&registry, &mut flags).unwrap_err();
        assert_eq!(
            err.to_string(),
            "Invalid HDF5 format: plugin loading state mask 2147483648 does not fit in i32"
        );
    }

    #[test]
    fn unsigned_loading_state_helper_matches_c_output_parameter() {
        let mut registry = H5PL__init_package();
        H5PLset_loading_state(&mut registry, H5PL_FILTER_PLUGIN | H5PL_VFD_PLUGIN);

        let mut mask = 0;
        H5PLget_loading_state_u32_into(&registry, &mut mask).unwrap();
        assert_eq!(mask, (H5PL_FILTER_PLUGIN | H5PL_VFD_PLUGIN) as u32);

        H5PLset_loading_state(&mut registry, u32::MAX as u64 + 1);
        let err = H5PLget_loading_state_u32_into(&registry, &mut mask).unwrap_err();
        assert_eq!(
            err.to_string(),
            "Invalid HDF5 format: plugin loading state mask 4294967296 does not fit in u32"
        );
        assert_eq!(mask, (H5PL_FILTER_PLUGIN | H5PL_VFD_PLUGIN) as u32);
    }

    #[test]
    fn allocating_plugin_path_search_wrappers_return_existing_candidates() {
        let mut table = H5PL__create_path_table();
        let dir =
            std::env::temp_dir().join(format!("hdf5-pure-rust-plugin-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let plugin_name = "libh5z_test_plugin.so";
        let plugin_path = dir.join(plugin_name);
        std::fs::write(&plugin_path, b"placeholder").unwrap();

        H5PL__append_path(&mut table, dir.clone());
        assert_eq!(
            H5PL__find_plugin_in_path(&dir, plugin_name),
            Some(plugin_path.clone())
        );
        assert_eq!(
            H5PL__find_plugin_in_path_table(&table, plugin_name),
            Some(plugin_path.clone())
        );
        assert_eq!(H5PL__find_plugin_in_path(&dir, "missing-plugin"), None);
        assert_eq!(
            H5PL__find_plugin_in_path_table(&table, "missing-plugin"),
            None
        );

        let _ = std::fs::remove_file(plugin_path);
        let _ = std::fs::remove_dir(dir);
    }

    #[test]
    fn public_plugin_init_wrapper_creates_default_registry() {
        let registry = H5PL_init();
        assert_eq!(H5PLget_loading_state(&registry), H5PL_ALL_PLUGIN);
        assert_eq!(H5PLsize(&registry), 0);
        assert_eq!(H5PL_iter_plugins(&registry).count(), 0);
    }

    #[test]
    #[allow(deprecated)]
    fn allocating_public_plugin_remove_wrapper_remains_callable() {
        let mut registry = H5PL__init_package();
        H5PL__add_plugin_ref(&mut registry.cache, "known");
        let mut names = Vec::new();
        H5PL_iterate_into(&registry, &mut names);
        assert_eq!(names, vec!["known"]);

        H5PLappend(&mut registry, "/tmp");
        assert_eq!(H5PLremove(&mut registry, 0).unwrap(), PathBuf::from("/tmp"));
    }
}
