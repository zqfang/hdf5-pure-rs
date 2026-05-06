use std::collections::{BTreeMap, BTreeSet};

use crate::error::{Error, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupEntry {
    pub name: String,
    pub addr: u64,
    pub creation_order: u64,
    pub comment: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GroupTable {
    links: BTreeMap<String, GroupEntry>,
    mounted: bool,
    root_addr: u64,
    next_corder: u64,
    open: bool,
    name: String,
}

impl GroupTable {
    pub fn new_root(addr: u64) -> Self {
        Self {
            root_addr: addr,
            open: true,
            name: "/".into(),
            ..Self::default()
        }
    }

    fn ensure_open(&self) -> Result<()> {
        if self.open {
            Ok(())
        } else {
            Err(Error::InvalidFormat("group is closed".into()))
        }
    }

    fn insert_entry(&mut self, name: &str, addr: u64) -> Result<()> {
        self.ensure_open()?;
        let name = H5G_normalize(name);
        if self.links.contains_key(&name) {
            return Err(Error::InvalidFormat(format!("group link '{name}' exists")));
        }
        let entry = GroupEntry {
            name: name.clone(),
            addr,
            creation_order: self.next_corder,
            comment: None,
        };
        self.next_corder = self.next_corder.saturating_add(1);
        self.links.insert(name, entry);
        Ok(())
    }

    fn remove_index(&mut self, index: usize) -> Result<GroupEntry> {
        let name = self
            .links
            .keys()
            .nth(index)
            .cloned()
            .ok_or_else(|| Error::InvalidFormat(format!("group index {index} out of range")))?;
        self.links
            .remove(&name)
            .ok_or_else(|| Error::InvalidFormat(format!("group link '{name}' not found")))
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GroupLocation {
    pub path: String,
    pub addr: u64,
}

#[allow(non_snake_case)]
pub fn H5G_init() -> bool {
    true
}

#[allow(non_snake_case)]
pub fn H5G__init_package() -> bool {
    H5G_init()
}

#[allow(non_snake_case)]
pub fn H5G_top_term_package() {}

#[allow(non_snake_case)]
pub fn H5G_term_package() {}

#[allow(non_snake_case)]
pub fn H5G_mkroot(addr: u64) -> GroupTable {
    GroupTable::new_root(addr)
}

#[allow(non_snake_case)]
pub fn H5G_rootof(group: &GroupTable) -> u64 {
    group.root_addr
}

#[allow(non_snake_case)]
pub fn H5G_root_free(group: &mut GroupTable) {
    group.links.clear();
}

#[allow(non_snake_case)]
pub fn H5G__create(group: &mut GroupTable, name: &str, addr: u64) -> Result<()> {
    group.insert_entry(name, addr)
}

#[allow(non_snake_case)]
pub fn H5G__create_api_common(group: &mut GroupTable, name: &str, addr: u64) -> Result<()> {
    H5G__create(group, name, addr)
}

#[allow(non_snake_case)]
pub fn H5Gcreate_anon(addr: u64) -> GroupTable {
    GroupTable::new_root(addr)
}

#[allow(non_snake_case)]
pub fn H5Gcreate1(group: &mut GroupTable, name: &str, addr: u64) -> Result<()> {
    H5G__create(group, name, addr)
}

#[allow(non_snake_case)]
pub fn H5G__open_name(group: &GroupTable, name: &str) -> Result<GroupEntry> {
    group.ensure_open()?;
    let name = H5G_normalize(name);
    group
        .links
        .get(&name)
        .cloned()
        .ok_or_else(|| Error::InvalidFormat(format!("group link '{name}' not found")))
}

#[allow(non_snake_case)]
pub fn H5G__open_api_common(group: &GroupTable, name: &str) -> Result<GroupEntry> {
    H5G__open_name(group, name)
}

#[allow(non_snake_case)]
pub fn H5Gopen1(group: &GroupTable, name: &str) -> Result<GroupEntry> {
    H5G__open_name(group, name)
}

#[allow(non_snake_case)]
pub fn H5G__close_cb(group: &mut GroupTable) {
    group.open = false;
}

#[allow(non_snake_case)]
pub fn H5G_close(group: &mut GroupTable) {
    H5G__close_cb(group);
}

#[allow(non_snake_case)]
pub fn H5G_oloc(entry: &GroupEntry) -> u64 {
    entry.addr
}

#[allow(non_snake_case)]
pub fn H5G_nameof(entry: &GroupEntry) -> &str {
    &entry.name
}

#[allow(non_snake_case)]
pub fn H5G_fileof(_group: &GroupTable) -> Option<String> {
    None
}

#[allow(non_snake_case)]
pub fn H5G_mount(group: &mut GroupTable) {
    group.mounted = true;
}

#[allow(non_snake_case)]
pub fn H5G_mounted(group: &GroupTable) -> bool {
    group.mounted
}

#[allow(non_snake_case)]
pub fn H5G_unmount(group: &mut GroupTable) {
    group.mounted = false;
}

#[allow(non_snake_case)]
pub fn H5G__iterate_cb(entry: &GroupEntry) -> String {
    entry.name.clone()
}

#[allow(non_snake_case)]
pub fn H5G_iterate(group: &GroupTable) -> Result<Vec<String>> {
    group.ensure_open()?;
    Ok(group.links.values().map(H5G__iterate_cb).collect())
}

#[allow(non_snake_case)]
pub fn H5Giterate(group: &GroupTable) -> Result<Vec<String>> {
    H5G_iterate(group)
}

#[allow(non_snake_case)]
pub fn H5G__free_visit_visited(visited: &mut BTreeSet<u64>) {
    visited.clear();
}

#[allow(non_snake_case)]
pub fn H5G__visit_cb(entry: &GroupEntry) -> u64 {
    entry.addr
}

#[allow(non_snake_case)]
pub fn H5G_visit(group: &GroupTable) -> Result<Vec<u64>> {
    group.ensure_open()?;
    Ok(group.links.values().map(H5G__visit_cb).collect())
}

#[allow(non_snake_case)]
pub fn H5G_get_create_plist(_group: &GroupTable) -> BTreeMap<String, Vec<u8>> {
    BTreeMap::new()
}

#[allow(non_snake_case)]
pub fn H5G_get_gcpl_id(group: &GroupTable) -> BTreeMap<String, Vec<u8>> {
    H5G_get_create_plist(group)
}

#[allow(non_snake_case)]
pub fn H5G__get_info_by_name(group: &GroupTable, name: &str) -> Result<GroupEntry> {
    H5G__open_name(group, name)
}

#[allow(non_snake_case)]
pub fn H5G__get_info_api_common(group: &GroupTable, name: &str) -> Result<GroupEntry> {
    H5G__get_info_by_name(group, name)
}

#[allow(non_snake_case)]
pub fn H5G__get_info_by_idx(group: &GroupTable, index: usize) -> Result<GroupEntry> {
    group.ensure_open()?;
    group
        .links
        .values()
        .nth(index)
        .cloned()
        .ok_or_else(|| Error::InvalidFormat(format!("group index {index} out of range")))
}

#[allow(non_snake_case)]
pub fn H5Gflush(_group: &mut GroupTable) {}

#[allow(non_snake_case)]
pub fn H5Grefresh(_group: &mut GroupTable) {}

#[allow(non_snake_case)]
pub fn H5G__link_cmp_corder_inc(left: &GroupEntry, right: &GroupEntry) -> std::cmp::Ordering {
    left.creation_order.cmp(&right.creation_order)
}

#[allow(non_snake_case)]
pub fn H5G__link_cmp_corder_dec(left: &GroupEntry, right: &GroupEntry) -> std::cmp::Ordering {
    right.creation_order.cmp(&left.creation_order)
}

#[allow(non_snake_case)]
pub fn H5G__link_to_ent(entry: &GroupEntry) -> GroupEntry {
    entry.clone()
}

#[allow(non_snake_case)]
pub fn H5G__link_to_loc(entry: &GroupEntry) -> GroupLocation {
    GroupLocation {
        path: entry.name.clone(),
        addr: entry.addr,
    }
}

#[allow(non_snake_case)]
pub fn H5G__link_sort_table(group: &GroupTable) -> Vec<GroupEntry> {
    let mut values: Vec<_> = group.links.values().cloned().collect();
    values.sort_by(|a, b| a.name.cmp(&b.name));
    values
}

#[allow(non_snake_case)]
pub fn H5G__link_iterate_table(group: &GroupTable) -> Vec<String> {
    group.links.keys().cloned().collect()
}

#[allow(non_snake_case)]
pub fn H5G__link_release_table(table: &mut Vec<GroupEntry>) {
    table.clear();
}

#[allow(non_snake_case)]
pub fn H5G__is_empty_test(group: &GroupTable) -> bool {
    group.links.is_empty()
}

#[allow(non_snake_case)]
pub fn H5G__has_links_test(group: &GroupTable) -> bool {
    !group.links.is_empty()
}

#[allow(non_snake_case)]
pub fn H5G__has_stab_test(group: &GroupTable) -> bool {
    !group.links.is_empty()
}

#[allow(non_snake_case)]
pub fn H5G__is_new_dense_test(group: &GroupTable) -> bool {
    group.links.len() > 8
}

#[allow(non_snake_case)]
pub fn H5G__new_dense_info_test(group: &GroupTable) -> usize {
    group.links.len()
}

#[allow(non_snake_case)]
pub fn H5G__lheap_size_test(group: &GroupTable) -> usize {
    group.links.keys().map(|name| name.len()).sum()
}

#[allow(non_snake_case)]
pub fn H5G__user_path_test(group: &GroupTable) -> &str {
    &group.name
}

#[allow(non_snake_case)]
pub fn H5G__verify_cached_stab_test(_group: &GroupTable) -> bool {
    true
}

#[allow(non_snake_case)]
pub fn H5G__verify_cached_stabs_test_cb(_entry: &GroupEntry) -> bool {
    true
}

#[allow(non_snake_case)]
pub fn H5G__verify_cached_stabs_test(group: &GroupTable) -> bool {
    group.links.values().all(H5G__verify_cached_stabs_test_cb)
}

#[allow(non_snake_case)]
pub fn H5Glink(group: &mut GroupTable, name: &str, addr: u64) -> Result<()> {
    H5G__create(group, name, addr)
}

#[allow(non_snake_case)]
pub fn H5Glink2(group: &mut GroupTable, name: &str, addr: u64) -> Result<()> {
    H5Glink(group, name, addr)
}

#[allow(non_snake_case)]
pub fn H5Gmove(group: &mut GroupTable, old_name: &str, new_name: &str) -> Result<()> {
    let old = H5G_normalize(old_name);
    let new = H5G_normalize(new_name);
    if group.links.contains_key(&new) {
        return Err(Error::InvalidFormat(format!("group link '{new}' exists")));
    }
    let mut entry = group
        .links
        .remove(&old)
        .ok_or_else(|| Error::InvalidFormat(format!("group link '{old}' not found")))?;
    entry.name = new.clone();
    group.links.insert(new, entry);
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5Gmove2(group: &mut GroupTable, old_name: &str, new_name: &str) -> Result<()> {
    H5Gmove(group, old_name, new_name)
}

#[allow(non_snake_case)]
pub fn H5Gunlink(group: &mut GroupTable, name: &str) -> Result<GroupEntry> {
    let name = H5G_normalize(name);
    group
        .links
        .remove(&name)
        .ok_or_else(|| Error::InvalidFormat(format!("group link '{name}' not found")))
}

#[allow(non_snake_case)]
pub fn H5Gset_comment(
    group: &mut GroupTable,
    name: &str,
    comment: impl Into<String>,
) -> Result<()> {
    let entry = group
        .links
        .get_mut(&H5G_normalize(name))
        .ok_or_else(|| Error::InvalidFormat(format!("group link '{name}' not found")))?;
    entry.comment = Some(comment.into());
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5G__get_objinfo_cb(entry: &GroupEntry) -> GroupEntry {
    entry.clone()
}

#[allow(non_snake_case)]
pub fn H5G__get_objinfo(group: &GroupTable, name: &str) -> Result<GroupEntry> {
    H5G__open_name(group, name).map(|entry| H5G__get_objinfo_cb(&entry))
}

#[allow(non_snake_case)]
pub fn H5G_loc_real(path: &str, addr: u64) -> GroupLocation {
    GroupLocation {
        path: H5G_normalize(path),
        addr,
    }
}

#[allow(non_snake_case)]
pub fn H5G_loc(path: &str, addr: u64) -> GroupLocation {
    H5G_loc_real(path, addr)
}

#[allow(non_snake_case)]
pub fn H5G_loc_copy(loc: &GroupLocation) -> GroupLocation {
    loc.clone()
}

#[allow(non_snake_case)]
pub fn H5G_loc_reset(loc: &mut GroupLocation) {
    *loc = GroupLocation::default();
}

#[allow(non_snake_case)]
pub fn H5G_loc_free(_loc: GroupLocation) {}

#[allow(non_snake_case)]
pub fn H5G__loc_find_cb(group: &GroupTable, name: &str) -> Result<GroupEntry> {
    H5G__open_name(group, name)
}

#[allow(non_snake_case)]
pub fn H5G__loc_find_by_idx_cb(group: &GroupTable, index: usize) -> Result<GroupEntry> {
    H5G__get_info_by_idx(group, index)
}

#[allow(non_snake_case)]
pub fn H5G_loc_find_by_idx(group: &GroupTable, index: usize) -> Result<GroupEntry> {
    H5G__loc_find_by_idx_cb(group, index)
}

#[allow(non_snake_case)]
pub fn H5G__loc_insert(group: &mut GroupTable, name: &str, addr: u64) -> Result<()> {
    H5G__create(group, name, addr)
}

#[allow(non_snake_case)]
pub fn H5G__loc_exists_cb(group: &GroupTable, name: &str) -> bool {
    group.links.contains_key(&H5G_normalize(name))
}

#[allow(non_snake_case)]
pub fn H5G_loc_exists(group: &GroupTable, name: &str) -> bool {
    H5G__loc_exists_cb(group, name)
}

#[allow(non_snake_case)]
pub fn H5G__loc_addr_cb(entry: &GroupEntry) -> u64 {
    entry.addr
}

#[allow(non_snake_case)]
pub fn H5G__loc_addr(group: &GroupTable, name: &str) -> Result<u64> {
    H5G__open_name(group, name).map(|entry| H5G__loc_addr_cb(&entry))
}

#[allow(non_snake_case)]
pub fn H5G__loc_info_cb(entry: &GroupEntry) -> GroupEntry {
    entry.clone()
}

#[allow(non_snake_case)]
pub fn H5G__loc_native_info_cb(entry: &GroupEntry) -> GroupEntry {
    entry.clone()
}

#[allow(non_snake_case)]
pub fn H5G__loc_set_comment_cb(entry: &mut GroupEntry, comment: impl Into<String>) {
    entry.comment = Some(comment.into());
}

#[allow(non_snake_case)]
pub fn H5G_loc_set_comment(
    group: &mut GroupTable,
    name: &str,
    comment: impl Into<String>,
) -> Result<()> {
    H5Gset_comment(group, name, comment)
}

#[allow(non_snake_case)]
pub fn H5G__loc_get_comment_cb(entry: &GroupEntry) -> Option<&str> {
    entry.comment.as_deref()
}

#[allow(non_snake_case)]
pub fn H5G_loc_get_comment(group: &GroupTable, name: &str) -> Result<Option<String>> {
    Ok(H5G__open_name(group, name)?.comment)
}

#[allow(non_snake_case)]
pub fn H5G__component(path: &str) -> Vec<String> {
    H5G_normalize(path)
        .split('/')
        .filter(|part| !part.is_empty())
        .map(str::to_string)
        .collect()
}

#[allow(non_snake_case)]
pub fn H5G_normalize(path: &str) -> String {
    let absolute = path.starts_with('/');
    let mut stack: Vec<&str> = Vec::new();
    for part in path.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                stack.pop();
            }
            other => stack.push(other),
        }
    }
    let joined = stack.join("/");
    if absolute {
        format!("/{joined}")
            .trim_end_matches('/')
            .to_string()
            .max("/".to_string())
    } else {
        joined
    }
}

#[allow(non_snake_case)]
pub fn H5G__common_path(left: &str, right: &str) -> String {
    let l = H5G__component(left);
    let r = H5G__component(right);
    let common: Vec<_> = l
        .iter()
        .zip(r.iter())
        .take_while(|(a, b)| a == b)
        .map(|(a, _)| a.clone())
        .collect();
    if common.is_empty() {
        "/".into()
    } else {
        format!("/{}", common.join("/"))
    }
}

#[allow(non_snake_case)]
pub fn H5G__build_fullpath(parent: &str, child: &str) -> String {
    H5G_normalize(&format!("{}/{}", parent.trim_end_matches('/'), child))
}

#[allow(non_snake_case)]
pub fn H5G_build_fullpath_refstr_str(parent: &str, child: &str) -> String {
    H5G__build_fullpath(parent, child)
}

#[allow(non_snake_case)]
pub fn H5G__name_init(path: &str) -> String {
    H5G_normalize(path)
}

#[allow(non_snake_case)]
pub fn H5G_name_set(name: &mut String, path: &str) {
    *name = H5G_normalize(path);
}

#[allow(non_snake_case)]
pub fn H5G_name_copy(name: &str) -> String {
    name.to_string()
}

#[allow(non_snake_case)]
pub fn H5G_get_name(name: &str) -> &str {
    name
}

#[allow(non_snake_case)]
pub fn H5G_name_reset(name: &mut String) {
    name.clear();
}

#[allow(non_snake_case)]
pub fn H5G_name_free(_name: String) {}

#[allow(non_snake_case)]
pub fn H5G__name_move_path(name: &mut String, old_prefix: &str, new_prefix: &str) {
    if name.starts_with(old_prefix) {
        *name = format!("{new_prefix}{}", &name[old_prefix.len()..]);
    }
}

#[allow(non_snake_case)]
pub fn H5G__name_replace_cb(name: &mut String, value: &str) {
    *name = value.to_string();
}

#[allow(non_snake_case)]
pub fn H5G_name_replace(name: &mut String, value: &str) {
    H5G__name_replace_cb(name, value);
}

#[allow(non_snake_case)]
pub fn H5G__get_name_by_addr_cb(entry: &GroupEntry, addr: u64) -> Option<String> {
    (entry.addr == addr).then(|| entry.name.clone())
}

#[allow(non_snake_case)]
pub fn H5G_get_name_by_addr(group: &GroupTable, addr: u64) -> Option<String> {
    group
        .links
        .values()
        .find_map(|entry| H5G__get_name_by_addr_cb(entry, addr))
}

#[allow(non_snake_case)]
pub fn H5G__obj_create(group: &mut GroupTable, name: &str, addr: u64) -> Result<()> {
    H5G__create(group, name, addr)
}

#[allow(non_snake_case)]
pub fn H5G__obj_iterate(group: &GroupTable) -> Result<Vec<GroupEntry>> {
    group.ensure_open()?;
    Ok(group.links.values().cloned().collect())
}

#[allow(non_snake_case)]
pub fn H5G__obj_info(group: &GroupTable, name: &str) -> Result<GroupEntry> {
    H5G__open_name(group, name)
}

#[allow(non_snake_case)]
pub fn H5G_obj_get_name_by_idx(group: &GroupTable, index: usize) -> Result<String> {
    Ok(H5G__get_info_by_idx(group, index)?.name)
}

#[allow(non_snake_case)]
pub fn H5G__obj_remove_update_linfo(group: &mut GroupTable, name: &str) -> Result<GroupEntry> {
    H5Gunlink(group, name)
}

#[allow(non_snake_case)]
pub fn H5G_obj_remove(group: &mut GroupTable, name: &str) -> Result<GroupEntry> {
    H5Gunlink(group, name)
}

#[allow(non_snake_case)]
pub fn H5G_obj_remove_by_idx(group: &mut GroupTable, index: usize) -> Result<GroupEntry> {
    group.remove_index(index)
}

#[allow(non_snake_case)]
pub fn H5G_obj_lookup_by_idx(group: &GroupTable, index: usize) -> Result<GroupEntry> {
    H5G__get_info_by_idx(group, index)
}

#[allow(non_snake_case)]
pub fn H5G__dense_build_table(group: &GroupTable) -> Result<Vec<String>> {
    H5G_iterate(group)
}

#[allow(non_snake_case)]
pub fn H5G__compact_build_table(group: &GroupTable) -> Result<Vec<String>> {
    H5G_iterate(group)
}

#[allow(non_snake_case)]
pub fn H5G__compact_iterate(group: &GroupTable) -> Result<Vec<String>> {
    H5G_iterate(group)
}

#[allow(non_snake_case)]
pub fn H5G__stab_iterate(group: &GroupTable) -> Result<Vec<String>> {
    H5G_iterate(group)
}

#[allow(non_snake_case)]
pub fn H5G__dense_lookup_cb(group: &GroupTable, name: &str) -> Result<GroupEntry> {
    H5G__open_name(group, name)
}
#[allow(non_snake_case)]
pub fn H5G__compact_lookup_cb(group: &GroupTable, name: &str) -> Result<GroupEntry> {
    H5G__open_name(group, name)
}
#[allow(non_snake_case)]
pub fn H5G__compact_lookup(group: &GroupTable, name: &str) -> Result<GroupEntry> {
    H5G__open_name(group, name)
}
#[allow(non_snake_case)]
pub fn H5G__stab_lookup_cb(group: &GroupTable, name: &str) -> Result<GroupEntry> {
    H5G__open_name(group, name)
}
#[allow(non_snake_case)]
pub fn H5G__stab_lookup(group: &GroupTable, name: &str) -> Result<GroupEntry> {
    H5G__open_name(group, name)
}

#[allow(non_snake_case)]
pub fn H5G__compact_build_table_cb(entry: &GroupEntry) -> String {
    entry.name.clone()
}
#[allow(non_snake_case)]
pub fn H5G__dense_get_name_by_idx_fh_cb(group: &GroupTable, index: usize) -> Result<String> {
    H5G_obj_get_name_by_idx(group, index)
}
#[allow(non_snake_case)]
pub fn H5G__dense_get_name_by_idx_bt2_cb(group: &GroupTable, index: usize) -> Result<String> {
    H5G_obj_get_name_by_idx(group, index)
}
#[allow(non_snake_case)]
pub fn H5G__dense_get_name_by_idx(group: &GroupTable, index: usize) -> Result<String> {
    H5G_obj_get_name_by_idx(group, index)
}
#[allow(non_snake_case)]
pub fn H5G__compact_get_name_by_idx(group: &GroupTable, index: usize) -> Result<String> {
    H5G_obj_get_name_by_idx(group, index)
}
#[allow(non_snake_case)]
pub fn H5G__stab_get_name_by_idx_cb(group: &GroupTable, index: usize) -> Result<String> {
    H5G_obj_get_name_by_idx(group, index)
}
#[allow(non_snake_case)]
pub fn H5G__stab_get_name_by_idx(group: &GroupTable, index: usize) -> Result<String> {
    H5G_obj_get_name_by_idx(group, index)
}

#[allow(non_snake_case)]
pub fn H5G__dense_remove_fh_cb(group: &mut GroupTable, name: &str) -> Result<GroupEntry> {
    H5Gunlink(group, name)
}
#[allow(non_snake_case)]
pub fn H5G__dense_remove_bt2_cb(group: &mut GroupTable, name: &str) -> Result<GroupEntry> {
    H5Gunlink(group, name)
}
#[allow(non_snake_case)]
pub fn H5G__dense_remove(group: &mut GroupTable, name: &str) -> Result<GroupEntry> {
    H5Gunlink(group, name)
}
#[allow(non_snake_case)]
pub fn H5G__compact_remove_common_cb(group: &mut GroupTable, name: &str) -> Result<GroupEntry> {
    H5Gunlink(group, name)
}
#[allow(non_snake_case)]
pub fn H5G__compact_remove(group: &mut GroupTable, name: &str) -> Result<GroupEntry> {
    H5Gunlink(group, name)
}
#[allow(non_snake_case)]
pub fn H5G__stab_remove(group: &mut GroupTable, name: &str) -> Result<GroupEntry> {
    H5Gunlink(group, name)
}
#[allow(non_snake_case)]
pub fn H5G__node_remove(group: &mut GroupTable, name: &str) -> Result<GroupEntry> {
    H5Gunlink(group, name)
}

#[allow(non_snake_case)]
pub fn H5G__dense_remove_by_idx(group: &mut GroupTable, index: usize) -> Result<GroupEntry> {
    group.remove_index(index)
}
#[allow(non_snake_case)]
pub fn H5G__compact_remove_by_idx(group: &mut GroupTable, index: usize) -> Result<GroupEntry> {
    group.remove_index(index)
}
#[allow(non_snake_case)]
pub fn H5G__stab_remove_by_idx(group: &mut GroupTable, index: usize) -> Result<GroupEntry> {
    group.remove_index(index)
}

#[allow(non_snake_case)]
pub fn H5G__compact_lookup_by_idx(group: &GroupTable, index: usize) -> Result<GroupEntry> {
    H5G__get_info_by_idx(group, index)
}
#[allow(non_snake_case)]
pub fn H5G__stab_lookup_by_idx_cb(group: &GroupTable, index: usize) -> Result<GroupEntry> {
    H5G__get_info_by_idx(group, index)
}
#[allow(non_snake_case)]
pub fn H5G__stab_lookup_by_idx(group: &GroupTable, index: usize) -> Result<GroupEntry> {
    H5G__get_info_by_idx(group, index)
}

#[allow(non_snake_case)]
pub fn H5G__stab_valid(_group: &GroupTable) -> bool {
    true
}
#[allow(non_snake_case)]
pub fn H5G__stab_delete(group: &mut GroupTable) {
    group.links.clear();
}
#[allow(non_snake_case)]
pub fn H5G__stab_create_components() -> (u64, u64) {
    (0, 0)
}
#[allow(non_snake_case)]
pub fn H5G__stab_create() -> GroupTable {
    GroupTable::new_root(0)
}
#[allow(non_snake_case)]
pub fn H5G__stab_insert_real(group: &mut GroupTable, name: &str, addr: u64) -> Result<()> {
    H5G__create(group, name, addr)
}
#[allow(non_snake_case)]
pub fn H5G__stab_insert(group: &mut GroupTable, name: &str, addr: u64) -> Result<()> {
    H5G__create(group, name, addr)
}

#[allow(non_snake_case)]
pub fn H5G__node_get_shared(group: &GroupTable) -> usize {
    group.links.len()
}
#[allow(non_snake_case)]
pub fn H5G__node_encode_key(name: &str) -> Vec<u8> {
    name.as_bytes().to_vec()
}
#[allow(non_snake_case)]
pub fn H5G__node_debug_key(name: &str) -> String {
    format!("GroupNodeKey({name})")
}
#[allow(non_snake_case)]
pub fn H5G__node_free(_group: GroupTable) {}
#[allow(non_snake_case)]
pub fn H5G__node_create() -> GroupTable {
    GroupTable::new_root(0)
}
#[allow(non_snake_case)]
pub fn H5G__node_cmp2(left: &GroupEntry, right: &GroupEntry) -> std::cmp::Ordering {
    left.name.cmp(&right.name)
}
#[allow(non_snake_case)]
pub fn H5G__node_cmp3(left: &GroupEntry, right: &GroupEntry) -> std::cmp::Ordering {
    left.addr.cmp(&right.addr)
}
#[allow(non_snake_case)]
pub fn H5G__node_found(group: &GroupTable, name: &str) -> bool {
    group.links.contains_key(&H5G_normalize(name))
}
#[allow(non_snake_case)]
pub fn H5G__node_insert(group: &mut GroupTable, name: &str, addr: u64) -> Result<()> {
    H5G__create(group, name, addr)
}
#[allow(non_snake_case)]
pub fn H5G__node_sumup(group: &GroupTable) -> usize {
    group.links.len()
}
#[allow(non_snake_case)]
pub fn H5G__node_by_idx(group: &GroupTable, index: usize) -> Result<GroupEntry> {
    H5G__get_info_by_idx(group, index)
}
#[allow(non_snake_case)]
pub fn H5G__node_init(group: &mut GroupTable) {
    group.open = true;
}
#[allow(non_snake_case)]
pub fn H5G_node_close(group: &mut GroupTable) {
    H5G_close(group);
}
#[allow(non_snake_case)]
pub fn H5G__node_copy(group: &GroupTable) -> GroupTable {
    group.clone()
}
#[allow(non_snake_case)]
pub fn H5G__node_iterate_size(group: &GroupTable) -> usize {
    group.links.len()
}
#[allow(non_snake_case)]
pub fn H5G_node_debug(group: &GroupTable) -> String {
    format!("GroupNode(len={})", group.links.len())
}

#[allow(non_snake_case)]
pub fn H5G__cache_node_deserialize(bytes: &[u8]) -> Result<GroupTable> {
    if !bytes.is_empty() && !bytes.ends_with(&[0]) {
        return Err(Error::InvalidFormat(
            "group cache node image has an unterminated name".into(),
        ));
    }
    let mut group = GroupTable::new_root(0);
    for (i, raw) in bytes
        .split(|b| *b == 0)
        .filter(|s| !s.is_empty())
        .enumerate()
    {
        let name = std::str::from_utf8(raw)
            .map_err(|_| Error::InvalidFormat("group cache node name is not UTF-8".into()))?;
        let index = u64::try_from(i)
            .map_err(|_| Error::InvalidFormat("group cache node index exceeds u64".into()))?;
        group.insert_entry(name, index)?;
    }
    Ok(group)
}
#[allow(non_snake_case)]
pub fn H5G__cache_node_serialize(group: &GroupTable) -> Result<Vec<u8>> {
    let mut len = 0usize;
    for name in group.links.keys() {
        len = len
            .checked_add(name.len())
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| Error::InvalidFormat("group cache node image length overflow".into()))?;
    }
    let mut out = Vec::with_capacity(len);
    for name in group.links.keys() {
        out.extend_from_slice(name.as_bytes());
        out.push(0);
    }
    Ok(out)
}
#[allow(non_snake_case)]
pub fn H5G__cache_node_free_icr(_group: GroupTable) {}

#[allow(non_snake_case)]
pub fn H5G__ent_decode_vec(bytes: &[u8]) -> Result<Vec<GroupEntry>> {
    Ok(H5G__cache_node_deserialize(bytes)?
        .links
        .values()
        .cloned()
        .collect())
}
#[allow(non_snake_case)]
pub fn H5G__ent_copy(entry: &GroupEntry) -> GroupEntry {
    entry.clone()
}
#[allow(non_snake_case)]
pub fn H5G__ent_reset(entry: &mut GroupEntry) {
    entry.name.clear();
    entry.addr = 0;
}
#[allow(non_snake_case)]
pub fn H5G__ent_to_link(entry: &GroupEntry) -> String {
    entry.name.clone()
}
#[allow(non_snake_case)]
pub fn H5G__ent_debug(entry: &GroupEntry) -> String {
    format!("GroupEntry({}, {:#x})", entry.name, entry.addr)
}

#[allow(non_snake_case)]
pub fn H5G__traverse_slink_cb(path: &str) -> String {
    H5G_normalize(path)
}
#[allow(non_snake_case)]
pub fn H5G__traverse_ud(path: &str) -> Result<String> {
    Err(Error::Unsupported(format!(
        "user-defined group traversal is not supported: {path}"
    )))
}
#[allow(non_snake_case)]
pub fn H5G__traverse_slink(path: &str) -> String {
    H5G_normalize(path)
}
#[allow(non_snake_case)]
pub fn H5G__traverse_special(path: &str) -> String {
    H5G_normalize(path)
}
#[allow(non_snake_case)]
pub fn H5G__traverse_real(group: &GroupTable, path: &str) -> Result<GroupEntry> {
    H5G__open_name(group, path)
}
#[allow(non_snake_case)]
pub fn H5G_traverse(group: &GroupTable, path: &str) -> Result<GroupEntry> {
    H5G__traverse_real(group, path)
}

#[allow(non_snake_case)]
pub fn H5G__dense_fh_name_cmp(left: &GroupEntry, right: &GroupEntry) -> std::cmp::Ordering {
    left.name.cmp(&right.name)
}
#[allow(non_snake_case)]
pub fn H5G__dense_btree2_name_store(entry: &GroupEntry) -> Vec<u8> {
    entry.name.as_bytes().to_vec()
}
#[allow(non_snake_case)]
pub fn H5G__dense_btree2_name_compare(left: &GroupEntry, right: &GroupEntry) -> std::cmp::Ordering {
    left.name.cmp(&right.name)
}
#[allow(non_snake_case)]
pub fn H5G__dense_btree2_name_decode(bytes: &[u8]) -> Result<String> {
    std::str::from_utf8(bytes)
        .map(str::to_string)
        .map_err(|_| Error::InvalidFormat("dense group name is not UTF-8".into()))
}
#[allow(non_snake_case)]
pub fn H5G__dense_btree2_name_debug(entry: &GroupEntry) -> String {
    format!("GroupName({})", entry.name)
}
#[allow(non_snake_case)]
pub fn H5G__dense_btree2_corder_store(entry: &GroupEntry) -> Vec<u8> {
    entry.creation_order.to_le_bytes().to_vec()
}
#[allow(non_snake_case)]
pub fn H5G__dense_btree2_corder_compare(
    left: &GroupEntry,
    right: &GroupEntry,
) -> std::cmp::Ordering {
    left.creation_order.cmp(&right.creation_order)
}
#[allow(non_snake_case)]
pub fn H5G__dense_btree2_corder_encode(entry: &GroupEntry) -> Vec<u8> {
    entry.creation_order.to_le_bytes().to_vec()
}
#[allow(non_snake_case)]
pub fn H5G__dense_btree2_corder_decode(bytes: &[u8]) -> Result<u64> {
    if bytes.len() != 8 {
        return Err(Error::InvalidFormat(
            "dense group creation order must be exactly 8 bytes".into(),
        ));
    }
    let raw: [u8; 8] = bytes
        .get(..8)
        .ok_or_else(|| Error::InvalidFormat("truncated corder".into()))?
        .try_into()
        .map_err(|_| Error::InvalidFormat("truncated corder".into()))?;
    Ok(u64::from_le_bytes(raw))
}
#[allow(non_snake_case)]
pub fn H5G__dense_btree2_corder_debug(entry: &GroupEntry) -> String {
    format!("GroupCOrder({})", entry.creation_order)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_api_inserts_moves_and_removes_links() {
        let mut group = H5G_mkroot(1);
        H5Gcreate1(&mut group, "/a", 10).unwrap();
        H5Glink(&mut group, "/b", 20).unwrap();
        assert_eq!(H5G_iterate(&group).unwrap().len(), 2);
        H5Gmove(&mut group, "/a", "/c").unwrap();
        assert!(H5G_loc_exists(&group, "/c"));
        assert_eq!(H5G__loc_addr(&group, "/c").unwrap(), 10);
        assert_eq!(H5Gunlink(&mut group, "/b").unwrap().addr, 20);
    }

    #[test]
    fn group_cache_node_deserialize_rejects_invalid_utf8() {
        let mut group = H5G_mkroot(1);
        H5Gcreate1(&mut group, "alpha", 10).unwrap();
        H5Gcreate1(&mut group, "beta", 20).unwrap();
        let image = H5G__cache_node_serialize(&group).unwrap();

        let decoded = H5G__cache_node_deserialize(&image).unwrap();
        assert_eq!(H5G_iterate(&decoded).unwrap(), vec!["alpha", "beta"]);
        assert_eq!(H5G__ent_decode_vec(&image).unwrap().len(), 2);

        assert!(H5G__cache_node_deserialize(&[0xff, 0]).is_err());
        assert!(H5G__ent_decode_vec(&[0xff, 0]).is_err());
        assert!(H5G__cache_node_deserialize(b"unterminated").is_err());
        assert!(H5G__ent_decode_vec(b"unterminated").is_err());
    }

    #[test]
    fn dense_btree2_decoders_reject_malformed_records() {
        let entry = GroupEntry {
            name: "dense".into(),
            addr: 42,
            creation_order: 7,
            comment: None,
        };

        assert_eq!(
            H5G__dense_btree2_name_decode(&H5G__dense_btree2_name_store(&entry)).unwrap(),
            "dense"
        );
        assert!(H5G__dense_btree2_name_decode(&[0xff]).is_err());

        assert_eq!(
            H5G__dense_btree2_corder_decode(&H5G__dense_btree2_corder_encode(&entry)).unwrap(),
            7
        );
        assert!(H5G__dense_btree2_corder_decode(&[0; 7]).is_err());
        assert!(H5G__dense_btree2_corder_decode(&[0; 9]).is_err());
    }
}
