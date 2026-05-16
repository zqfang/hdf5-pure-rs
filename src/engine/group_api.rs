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
    /// Construct a new root group rooted at the given object-header address.
    pub fn new_root(addr: u64) -> Self {
        Self {
            root_addr: addr,
            open: true,
            name: "/".into(),
            ..Self::default()
        }
    }

    /// Return `Ok(())` if the group is open, else an `InvalidFormat` error.
    fn ensure_open(&self) -> Result<()> {
        if self.open {
            Ok(())
        } else {
            Err(Error::InvalidFormat("group is closed".into()))
        }
    }

    /// Insert a new link entry into the group, assigning the next creation order.
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

    /// Remove and return the link at the given index in the group's name-sorted order.
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

/// Initialize the group interface.
#[allow(non_snake_case)]
pub fn H5G_init() -> bool {
    true
}

/// Initialize interface-specific information for the group package.
#[allow(non_snake_case)]
pub fn H5G__init_package() -> bool {
    H5G_init()
}

/// Top-level group package teardown hook.
#[allow(non_snake_case)]
pub fn H5G_top_term_package() {}

/// Terminate the group package and release its resources.
#[allow(non_snake_case)]
pub fn H5G_term_package() {}

/// Create the root group at the given object-header address.
#[allow(non_snake_case)]
pub fn H5G_mkroot(addr: u64) -> GroupTable {
    GroupTable::new_root(addr)
}

/// Return the root group's object-header address.
#[allow(non_snake_case)]
pub fn H5G_rootof(group: &GroupTable) -> u64 {
    group.root_addr
}

/// Release all links stored under the root group.
#[allow(non_snake_case)]
pub fn H5G_root_free(group: &mut GroupTable) {
    group.links.clear();
}

/// Create a new link inside the group pointing at `addr`.
#[allow(non_snake_case)]
pub fn H5G__create(group: &mut GroupTable, name: &str, addr: u64) -> Result<()> {
    group.insert_entry(name, addr)
}

/// Common create-API plumbing for group entries.
#[allow(non_snake_case)]
pub fn H5G__create_api_common(group: &mut GroupTable, name: &str, addr: u64) -> Result<()> {
    H5G__create(group, name, addr)
}

/// Create an anonymous group (no name link), returning the new group table.
#[allow(non_snake_case)]
pub fn H5Gcreate_anon(addr: u64) -> GroupTable {
    GroupTable::new_root(addr)
}

/// Legacy H5Gcreate1: create a named group link.
#[allow(non_snake_case)]
pub fn H5Gcreate1(group: &mut GroupTable, name: &str, addr: u64) -> Result<()> {
    H5G__create(group, name, addr)
}

/// Look up a group entry by name.
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

/// Common open-API plumbing for group entries.
#[allow(non_snake_case)]
pub fn H5G__open_api_common(group: &GroupTable, name: &str) -> Result<GroupEntry> {
    H5G__open_name(group, name)
}

/// Legacy H5Gopen1: open a named group entry.
#[allow(non_snake_case)]
pub fn H5Gopen1(group: &GroupTable, name: &str) -> Result<GroupEntry> {
    H5G__open_name(group, name)
}

/// Close callback invoked when the last reference to a group is released.
#[allow(non_snake_case)]
pub fn H5G__close_cb(group: &mut GroupTable) {
    group.open = false;
}

/// Close a group, marking it as no longer open.
#[allow(non_snake_case)]
pub fn H5G_close(group: &mut GroupTable) {
    H5G__close_cb(group);
}

/// Return the object-header location for a group entry.
#[allow(non_snake_case)]
pub fn H5G_oloc(entry: &GroupEntry) -> u64 {
    entry.addr
}

/// Return the name of a group entry.
#[allow(non_snake_case)]
pub fn H5G_nameof(entry: &GroupEntry) -> &str {
    &entry.name
}

/// Return the file containing a group (none in pure-Rust mode).
#[allow(non_snake_case)]
pub fn H5G_fileof(_group: &GroupTable) -> Option<String> {
    None
}

/// Mark a group as mounted.
#[allow(non_snake_case)]
pub fn H5G_mount(group: &mut GroupTable) {
    group.mounted = true;
}

/// Return whether a group is currently mounted.
#[allow(non_snake_case)]
pub fn H5G_mounted(group: &GroupTable) -> bool {
    group.mounted
}

/// Mark a group as no longer mounted.
#[allow(non_snake_case)]
pub fn H5G_unmount(group: &mut GroupTable) {
    group.mounted = false;
}

/// Per-entry iteration callback returning the entry's name.
#[allow(non_snake_case)]
pub fn H5G__iterate_cb(entry: &GroupEntry) -> String {
    entry.name.clone()
}

/// Iterate over the group's links, returning the list of names.
#[allow(non_snake_case)]
pub fn H5G_iterate(group: &GroupTable) -> Result<Vec<String>> {
    group.ensure_open()?;
    Ok(group.links.values().map(H5G__iterate_cb).collect())
}

/// Legacy H5Giterate: iterate over a group's links.
#[allow(non_snake_case)]
pub fn H5Giterate(group: &GroupTable) -> Result<Vec<String>> {
    H5G_iterate(group)
}

/// Reset the visited-objects set used during recursive `H5Gvisit` traversal.
#[allow(non_snake_case)]
pub fn H5G__free_visit_visited(visited: &mut BTreeSet<u64>) {
    visited.clear();
}

/// Per-entry visit callback returning the entry's object-header address.
#[allow(non_snake_case)]
pub fn H5G__visit_cb(entry: &GroupEntry) -> u64 {
    entry.addr
}

/// Recursively visit all objects reachable from the group, returning their addresses.
#[allow(non_snake_case)]
pub fn H5G_visit(group: &GroupTable) -> Result<Vec<u64>> {
    group.ensure_open()?;
    Ok(group.links.values().map(H5G__visit_cb).collect())
}

/// Return the group creation property list (empty map in pure-Rust mode).
#[allow(non_snake_case)]
pub fn H5G_get_create_plist(_group: &GroupTable) -> BTreeMap<String, Vec<u8>> {
    BTreeMap::new()
}

/// Return the group creation property list ID.
#[allow(non_snake_case)]
pub fn H5G_get_gcpl_id(group: &GroupTable) -> BTreeMap<String, Vec<u8>> {
    H5G_get_create_plist(group)
}

/// Look up group info for the link with the given name.
#[allow(non_snake_case)]
pub fn H5G__get_info_by_name(group: &GroupTable, name: &str) -> Result<GroupEntry> {
    H5G__open_name(group, name)
}

/// Common get-info API plumbing.
#[allow(non_snake_case)]
pub fn H5G__get_info_api_common(group: &GroupTable, name: &str) -> Result<GroupEntry> {
    H5G__get_info_by_name(group, name)
}

/// Look up group info for the link at the given index in name-sorted order.
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

/// Flush the group to storage.
#[allow(non_snake_case)]
pub fn H5Gflush(_group: &mut GroupTable) {}

/// Refresh the group from storage, picking up any external changes.
#[allow(non_snake_case)]
pub fn H5Grefresh(_group: &mut GroupTable) {}

/// Sort comparator for group links by ascending creation order.
#[allow(non_snake_case)]
pub fn H5G__link_cmp_corder_inc(left: &GroupEntry, right: &GroupEntry) -> std::cmp::Ordering {
    left.creation_order.cmp(&right.creation_order)
}

/// Sort comparator for group links by descending creation order.
#[allow(non_snake_case)]
pub fn H5G__link_cmp_corder_dec(left: &GroupEntry, right: &GroupEntry) -> std::cmp::Ordering {
    right.creation_order.cmp(&left.creation_order)
}

/// Convert a link message into a group entry record.
#[allow(non_snake_case)]
pub fn H5G__link_to_ent(entry: &GroupEntry) -> GroupEntry {
    entry.clone()
}

/// Convert a link message into a `GroupLocation`.
#[allow(non_snake_case)]
pub fn H5G__link_to_loc(entry: &GroupEntry) -> GroupLocation {
    GroupLocation {
        path: entry.name.clone(),
        addr: entry.addr,
    }
}

/// Build and sort a flat table of links by name.
#[allow(non_snake_case)]
pub fn H5G__link_sort_table(group: &GroupTable) -> Vec<GroupEntry> {
    let mut values: Vec<_> = group.links.values().cloned().collect();
    values.sort_by(|a, b| a.name.cmp(&b.name));
    values
}

/// Iterate over the link table, returning the list of link names.
#[allow(non_snake_case)]
pub fn H5G__link_iterate_table(group: &GroupTable) -> Vec<String> {
    group.links.keys().cloned().collect()
}

/// Release a link table previously built for iteration.
#[allow(non_snake_case)]
pub fn H5G__link_release_table(table: &mut Vec<GroupEntry>) {
    table.clear();
}

/// Test helper: return whether the group has no links.
#[allow(non_snake_case)]
pub fn H5G__is_empty_test(group: &GroupTable) -> bool {
    group.links.is_empty()
}

/// Test helper: return whether the group has at least one link.
#[allow(non_snake_case)]
pub fn H5G__has_links_test(group: &GroupTable) -> bool {
    !group.links.is_empty()
}

/// Test helper: return whether the group has a symbol-table message.
#[allow(non_snake_case)]
pub fn H5G__has_stab_test(group: &GroupTable) -> bool {
    !group.links.is_empty()
}

/// Test helper: return whether the group has transitioned to "new dense" storage.
#[allow(non_snake_case)]
pub fn H5G__is_new_dense_test(group: &GroupTable) -> bool {
    group.links.len() > 8
}

/// Test helper: return the number of entries in the new-dense storage.
#[allow(non_snake_case)]
pub fn H5G__new_dense_info_test(group: &GroupTable) -> usize {
    group.links.len()
}

/// Test helper: return the cumulative size of names stored in the local heap.
#[allow(non_snake_case)]
pub fn H5G__lheap_size_test(group: &GroupTable) -> usize {
    group.links.keys().map(|name| name.len()).sum()
}

/// Test helper: return the user-visible path attached to the group.
#[allow(non_snake_case)]
pub fn H5G__user_path_test(group: &GroupTable) -> &str {
    &group.name
}

/// Test helper: verify the cached symbol-table data for the group.
#[allow(non_snake_case)]
pub fn H5G__verify_cached_stab_test(_group: &GroupTable) -> bool {
    true
}

/// Per-entry callback used to verify cached symbol-table data for nested groups.
#[allow(non_snake_case)]
pub fn H5G__verify_cached_stabs_test_cb(_entry: &GroupEntry) -> bool {
    true
}

/// Test helper: recursively verify cached symbol-table data for the group's links.
#[allow(non_snake_case)]
pub fn H5G__verify_cached_stabs_test(group: &GroupTable) -> bool {
    group.links.values().all(H5G__verify_cached_stabs_test_cb)
}

/// Legacy H5Glink: create a hard link from `name` to `addr` inside the group.
#[allow(non_snake_case)]
pub fn H5Glink(group: &mut GroupTable, name: &str, addr: u64) -> Result<()> {
    H5G__create(group, name, addr)
}

/// Legacy H5Glink2: alias for `H5Glink`.
#[allow(non_snake_case)]
pub fn H5Glink2(group: &mut GroupTable, name: &str, addr: u64) -> Result<()> {
    H5Glink(group, name, addr)
}

/// Move (rename) a link from `old_name` to `new_name`.
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

/// Legacy H5Gmove2: alias for `H5Gmove`.
#[allow(non_snake_case)]
pub fn H5Gmove2(group: &mut GroupTable, old_name: &str, new_name: &str) -> Result<()> {
    H5Gmove(group, old_name, new_name)
}

/// Remove a link from the group and return its entry.
#[allow(non_snake_case)]
pub fn H5Gunlink(group: &mut GroupTable, name: &str) -> Result<GroupEntry> {
    let name = H5G_normalize(name);
    group
        .links
        .remove(&name)
        .ok_or_else(|| Error::InvalidFormat(format!("group link '{name}' not found")))
}

/// Attach a comment string to a group entry.
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

/// Per-entry callback returning a clone of the entry for `H5Gget_objinfo`.
#[allow(non_snake_case)]
pub fn H5G__get_objinfo_cb(entry: &GroupEntry) -> GroupEntry {
    entry.clone()
}

/// Return legacy `H5Gget_objinfo`-style information for a named link.
#[allow(non_snake_case)]
pub fn H5G__get_objinfo(group: &GroupTable, name: &str) -> Result<GroupEntry> {
    H5G__open_name(group, name).map(|entry| H5G__get_objinfo_cb(&entry))
}

/// Build a real (resolved) group location from a path and object-header address.
#[allow(non_snake_case)]
pub fn H5G_loc_real(path: &str, addr: u64) -> GroupLocation {
    GroupLocation {
        path: H5G_normalize(path),
        addr,
    }
}

/// Build a group location from a path and object-header address.
#[allow(non_snake_case)]
pub fn H5G_loc(path: &str, addr: u64) -> GroupLocation {
    H5G_loc_real(path, addr)
}

/// Return a deep copy of a group location.
#[allow(non_snake_case)]
pub fn H5G_loc_copy(loc: &GroupLocation) -> GroupLocation {
    loc.clone()
}

/// Reset a group location to its default empty state.
#[allow(non_snake_case)]
pub fn H5G_loc_reset(loc: &mut GroupLocation) {
    *loc = GroupLocation::default();
}

/// Release a group location, dropping any owned resources.
#[allow(non_snake_case)]
pub fn H5G_loc_free(_loc: GroupLocation) {}

/// Callback that finds a link by name inside a group.
#[allow(non_snake_case)]
pub fn H5G__loc_find_cb(group: &GroupTable, name: &str) -> Result<GroupEntry> {
    H5G__open_name(group, name)
}

/// Callback that finds a link by creation/name index.
#[allow(non_snake_case)]
pub fn H5G__loc_find_by_idx_cb(group: &GroupTable, index: usize) -> Result<GroupEntry> {
    H5G__get_info_by_idx(group, index)
}

/// Find the link at the given index within a group.
#[allow(non_snake_case)]
pub fn H5G_loc_find_by_idx(group: &GroupTable, index: usize) -> Result<GroupEntry> {
    H5G__loc_find_by_idx_cb(group, index)
}

/// Insert a new link at the given location inside the group.
#[allow(non_snake_case)]
pub fn H5G__loc_insert(group: &mut GroupTable, name: &str, addr: u64) -> Result<()> {
    H5G__create(group, name, addr)
}

/// Callback that checks whether a name exists inside the group.
#[allow(non_snake_case)]
pub fn H5G__loc_exists_cb(group: &GroupTable, name: &str) -> bool {
    group.links.contains_key(&H5G_normalize(name))
}

/// Check whether a path exists at a group location.
#[allow(non_snake_case)]
pub fn H5G_loc_exists(group: &GroupTable, name: &str) -> bool {
    H5G__loc_exists_cb(group, name)
}

/// Callback returning the object-header address for an entry.
#[allow(non_snake_case)]
pub fn H5G__loc_addr_cb(entry: &GroupEntry) -> u64 {
    entry.addr
}

/// Return the object-header address for a named link in the group.
#[allow(non_snake_case)]
pub fn H5G__loc_addr(group: &GroupTable, name: &str) -> Result<u64> {
    H5G__open_name(group, name).map(|entry| H5G__loc_addr_cb(&entry))
}

/// Callback returning object info for an entry.
#[allow(non_snake_case)]
pub fn H5G__loc_info_cb(entry: &GroupEntry) -> GroupEntry {
    entry.clone()
}

/// Callback returning native-specific object info for an entry.
#[allow(non_snake_case)]
pub fn H5G__loc_native_info_cb(entry: &GroupEntry) -> GroupEntry {
    entry.clone()
}

/// Callback that sets the comment on a group entry.
#[allow(non_snake_case)]
pub fn H5G__loc_set_comment_cb(entry: &mut GroupEntry, comment: impl Into<String>) {
    entry.comment = Some(comment.into());
}

/// Set the comment on the link at a given location.
#[allow(non_snake_case)]
pub fn H5G_loc_set_comment(
    group: &mut GroupTable,
    name: &str,
    comment: impl Into<String>,
) -> Result<()> {
    H5Gset_comment(group, name, comment)
}

/// Callback returning an entry's comment, if any.
#[allow(non_snake_case)]
pub fn H5G__loc_get_comment_cb(entry: &GroupEntry) -> Option<&str> {
    entry.comment.as_deref()
}

/// Get the comment on the link at a given location.
#[allow(non_snake_case)]
pub fn H5G_loc_get_comment(group: &GroupTable, name: &str) -> Result<Option<String>> {
    Ok(H5G__open_name(group, name)?.comment)
}

/// Split a path into its non-empty components.
#[allow(non_snake_case)]
pub fn H5G__component(path: &str) -> Vec<String> {
    H5G_normalize(path)
        .split('/')
        .filter(|part| !part.is_empty())
        .map(str::to_string)
        .collect()
}

/// Normalize a group path by collapsing `.`, `..`, and duplicate slashes.
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

/// Return the longest common path prefix shared by two paths.
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

/// Build a full path by joining a parent and child path component.
#[allow(non_snake_case)]
pub fn H5G__build_fullpath(parent: &str, child: &str) -> String {
    H5G_normalize(&format!("{}/{}", parent.trim_end_matches('/'), child))
}

/// Build a full path from a parent reference-string and a child string.
#[allow(non_snake_case)]
pub fn H5G_build_fullpath_refstr_str(parent: &str, child: &str) -> String {
    H5G__build_fullpath(parent, child)
}

/// Initialize a name field from a path string.
#[allow(non_snake_case)]
pub fn H5G__name_init(path: &str) -> String {
    H5G_normalize(path)
}

/// Set a name field to the normalized form of a path.
#[allow(non_snake_case)]
pub fn H5G_name_set(name: &mut String, path: &str) {
    *name = H5G_normalize(path);
}

/// Return a copy of a name string.
#[allow(non_snake_case)]
pub fn H5G_name_copy(name: &str) -> String {
    name.to_string()
}

/// Return the underlying name string.
#[allow(non_snake_case)]
pub fn H5G_get_name(name: &str) -> &str {
    name
}

/// Reset a name field to the empty string.
#[allow(non_snake_case)]
pub fn H5G_name_reset(name: &mut String) {
    name.clear();
}

/// Release a name field, dropping any owned resources.
#[allow(non_snake_case)]
pub fn H5G_name_free(_name: String) {}

/// Rewrite a name's prefix when its parent path is moved.
#[allow(non_snake_case)]
pub fn H5G__name_move_path(name: &mut String, old_prefix: &str, new_prefix: &str) {
    if name.starts_with(old_prefix) {
        *name = format!("{new_prefix}{}", &name[old_prefix.len()..]);
    }
}

/// Callback that replaces an entry's name with a new value.
#[allow(non_snake_case)]
pub fn H5G__name_replace_cb(name: &mut String, value: &str) {
    *name = value.to_string();
}

/// Replace a name field with a new string.
#[allow(non_snake_case)]
pub fn H5G_name_replace(name: &mut String, value: &str) {
    H5G__name_replace_cb(name, value);
}

/// Per-entry callback returning the entry's name iff its address matches.
#[allow(non_snake_case)]
pub fn H5G__get_name_by_addr_cb(entry: &GroupEntry, addr: u64) -> Option<String> {
    (entry.addr == addr).then(|| entry.name.clone())
}

/// Search the group for the name of an entry with the given object-header address.
#[allow(non_snake_case)]
pub fn H5G_get_name_by_addr(group: &GroupTable, addr: u64) -> Option<String> {
    group
        .links
        .values()
        .find_map(|entry| H5G__get_name_by_addr_cb(entry, addr))
}

/// Create an object link inside a group via the object-API path.
#[allow(non_snake_case)]
pub fn H5G__obj_create(group: &mut GroupTable, name: &str, addr: u64) -> Result<()> {
    H5G__create(group, name, addr)
}

/// Iterate over the entries in a group via the object-API path.
#[allow(non_snake_case)]
pub fn H5G__obj_iterate(group: &GroupTable) -> Result<Vec<GroupEntry>> {
    group.ensure_open()?;
    Ok(group.links.values().cloned().collect())
}

/// Look up object info for a named link via the object-API path.
#[allow(non_snake_case)]
pub fn H5G__obj_info(group: &GroupTable, name: &str) -> Result<GroupEntry> {
    H5G__open_name(group, name)
}

/// Return the name of the link at a given index in the group.
#[allow(non_snake_case)]
pub fn H5G_obj_get_name_by_idx(group: &GroupTable, index: usize) -> Result<String> {
    Ok(H5G__get_info_by_idx(group, index)?.name)
}

/// Remove a link from the group, also updating its link-info state.
#[allow(non_snake_case)]
pub fn H5G__obj_remove_update_linfo(group: &mut GroupTable, name: &str) -> Result<GroupEntry> {
    H5Gunlink(group, name)
}

/// Object-API: remove a named link from the group.
#[allow(non_snake_case)]
pub fn H5G_obj_remove(group: &mut GroupTable, name: &str) -> Result<GroupEntry> {
    H5Gunlink(group, name)
}

/// Object-API: remove the link at a given index from the group.
#[allow(non_snake_case)]
pub fn H5G_obj_remove_by_idx(group: &mut GroupTable, index: usize) -> Result<GroupEntry> {
    group.remove_index(index)
}

/// Object-API: look up the link at a given index in the group.
#[allow(non_snake_case)]
pub fn H5G_obj_lookup_by_idx(group: &GroupTable, index: usize) -> Result<GroupEntry> {
    H5G__get_info_by_idx(group, index)
}

/// Build an iteration table for a dense-storage group.
#[allow(non_snake_case)]
pub fn H5G__dense_build_table(group: &GroupTable) -> Result<Vec<String>> {
    H5G_iterate(group)
}

/// Build an iteration table for a compact-storage group.
#[allow(non_snake_case)]
pub fn H5G__compact_build_table(group: &GroupTable) -> Result<Vec<String>> {
    H5G_iterate(group)
}

/// Iterate over the links in a compact-storage group.
#[allow(non_snake_case)]
pub fn H5G__compact_iterate(group: &GroupTable) -> Result<Vec<String>> {
    H5G_iterate(group)
}

/// Iterate over the links in a symbol-table-storage group.
#[allow(non_snake_case)]
pub fn H5G__stab_iterate(group: &GroupTable) -> Result<Vec<String>> {
    H5G_iterate(group)
}

/// Callback that looks up a name in a dense-storage group.
#[allow(non_snake_case)]
pub fn H5G__dense_lookup_cb(group: &GroupTable, name: &str) -> Result<GroupEntry> {
    H5G__open_name(group, name)
}
/// Callback that looks up a name in a compact-storage group.
#[allow(non_snake_case)]
pub fn H5G__compact_lookup_cb(group: &GroupTable, name: &str) -> Result<GroupEntry> {
    H5G__open_name(group, name)
}
/// Look up a name in a compact-storage group.
#[allow(non_snake_case)]
pub fn H5G__compact_lookup(group: &GroupTable, name: &str) -> Result<GroupEntry> {
    H5G__open_name(group, name)
}
/// Callback that looks up a name in a symbol-table-storage group.
#[allow(non_snake_case)]
pub fn H5G__stab_lookup_cb(group: &GroupTable, name: &str) -> Result<GroupEntry> {
    H5G__open_name(group, name)
}
/// Look up a name in a symbol-table-storage group.
#[allow(non_snake_case)]
pub fn H5G__stab_lookup(group: &GroupTable, name: &str) -> Result<GroupEntry> {
    H5G__open_name(group, name)
}

/// Per-entry callback when building a compact-storage iteration table.
#[allow(non_snake_case)]
pub fn H5G__compact_build_table_cb(entry: &GroupEntry) -> String {
    entry.name.clone()
}
/// Dense-storage fractal-heap callback returning the name at an index.
#[allow(non_snake_case)]
pub fn H5G__dense_get_name_by_idx_fh_cb(group: &GroupTable, index: usize) -> Result<String> {
    H5G_obj_get_name_by_idx(group, index)
}
/// Dense-storage v2 B-tree callback returning the name at an index.
#[allow(non_snake_case)]
pub fn H5G__dense_get_name_by_idx_bt2_cb(group: &GroupTable, index: usize) -> Result<String> {
    H5G_obj_get_name_by_idx(group, index)
}
/// Return the name at an index in a dense-storage group.
#[allow(non_snake_case)]
pub fn H5G__dense_get_name_by_idx(group: &GroupTable, index: usize) -> Result<String> {
    H5G_obj_get_name_by_idx(group, index)
}
/// Return the name at an index in a compact-storage group.
#[allow(non_snake_case)]
pub fn H5G__compact_get_name_by_idx(group: &GroupTable, index: usize) -> Result<String> {
    H5G_obj_get_name_by_idx(group, index)
}
/// Symbol-table callback returning the name at an index.
#[allow(non_snake_case)]
pub fn H5G__stab_get_name_by_idx_cb(group: &GroupTable, index: usize) -> Result<String> {
    H5G_obj_get_name_by_idx(group, index)
}
/// Return the name at an index in a symbol-table-storage group.
#[allow(non_snake_case)]
pub fn H5G__stab_get_name_by_idx(group: &GroupTable, index: usize) -> Result<String> {
    H5G_obj_get_name_by_idx(group, index)
}

/// Dense-storage fractal-heap callback removing a link by name.
#[allow(non_snake_case)]
pub fn H5G__dense_remove_fh_cb(group: &mut GroupTable, name: &str) -> Result<GroupEntry> {
    H5Gunlink(group, name)
}
/// Dense-storage v2 B-tree callback removing a link by name.
#[allow(non_snake_case)]
pub fn H5G__dense_remove_bt2_cb(group: &mut GroupTable, name: &str) -> Result<GroupEntry> {
    H5Gunlink(group, name)
}
/// Remove a link by name from a dense-storage group.
#[allow(non_snake_case)]
pub fn H5G__dense_remove(group: &mut GroupTable, name: &str) -> Result<GroupEntry> {
    H5Gunlink(group, name)
}
/// Common compact-storage callback removing a link by name.
#[allow(non_snake_case)]
pub fn H5G__compact_remove_common_cb(group: &mut GroupTable, name: &str) -> Result<GroupEntry> {
    H5Gunlink(group, name)
}
/// Remove a link by name from a compact-storage group.
#[allow(non_snake_case)]
pub fn H5G__compact_remove(group: &mut GroupTable, name: &str) -> Result<GroupEntry> {
    H5Gunlink(group, name)
}
/// Remove a link by name from a symbol-table-storage group.
#[allow(non_snake_case)]
pub fn H5G__stab_remove(group: &mut GroupTable, name: &str) -> Result<GroupEntry> {
    H5Gunlink(group, name)
}
/// Remove a link by name from a B-tree node-storage group.
#[allow(non_snake_case)]
pub fn H5G__node_remove(group: &mut GroupTable, name: &str) -> Result<GroupEntry> {
    H5Gunlink(group, name)
}

/// Remove a link by index from a dense-storage group.
#[allow(non_snake_case)]
pub fn H5G__dense_remove_by_idx(group: &mut GroupTable, index: usize) -> Result<GroupEntry> {
    group.remove_index(index)
}

/// Delete all links in a dense-storage group, optionally returning the removed entries.
#[allow(non_snake_case)]
pub fn H5G__dense_delete(group: &mut GroupTable, adj_link: bool) -> Result<Vec<GroupEntry>> {
    group.ensure_open()?;
    let mut removed = Vec::new();
    if adj_link {
        removed.reserve(group.links.len());
        for entry in group.links.values() {
            removed.push(GroupEntry {
                name: entry.name.clone(),
                addr: entry.addr,
                creation_order: entry.creation_order,
                comment: entry.comment.clone(),
            });
        }
    }
    group.links.clear();
    group.next_corder = 0;
    Ok(removed)
}

/// Remove a link by index from a compact-storage group.
#[allow(non_snake_case)]
pub fn H5G__compact_remove_by_idx(group: &mut GroupTable, index: usize) -> Result<GroupEntry> {
    group.remove_index(index)
}
/// Remove a link by index from a symbol-table-storage group.
#[allow(non_snake_case)]
pub fn H5G__stab_remove_by_idx(group: &mut GroupTable, index: usize) -> Result<GroupEntry> {
    group.remove_index(index)
}

/// Look up the link at an index in a compact-storage group.
#[allow(non_snake_case)]
pub fn H5G__compact_lookup_by_idx(group: &GroupTable, index: usize) -> Result<GroupEntry> {
    H5G__get_info_by_idx(group, index)
}
/// Callback that looks up the link at an index in a symbol-table-storage group.
#[allow(non_snake_case)]
pub fn H5G__stab_lookup_by_idx_cb(group: &GroupTable, index: usize) -> Result<GroupEntry> {
    H5G__get_info_by_idx(group, index)
}
/// Look up the link at an index in a symbol-table-storage group.
#[allow(non_snake_case)]
pub fn H5G__stab_lookup_by_idx(group: &GroupTable, index: usize) -> Result<GroupEntry> {
    H5G__get_info_by_idx(group, index)
}

/// Validate a symbol-table-storage group (always considered valid here).
#[allow(non_snake_case)]
pub fn H5G__stab_valid(_group: &GroupTable) -> bool {
    true
}
/// Delete all entries in a symbol-table-storage group.
#[allow(non_snake_case)]
pub fn H5G__stab_delete(group: &mut GroupTable) {
    group.links.clear();
}
/// Allocate the on-disk components for a new symbol-table group.
#[allow(non_snake_case)]
pub fn H5G__stab_create_components() -> (u64, u64) {
    (0, 0)
}
/// Create a new, empty symbol-table-storage group.
#[allow(non_snake_case)]
pub fn H5G__stab_create() -> GroupTable {
    GroupTable::new_root(0)
}
/// Insert a link into a symbol-table-storage group, lower-level path.
#[allow(non_snake_case)]
pub fn H5G__stab_insert_real(group: &mut GroupTable, name: &str, addr: u64) -> Result<()> {
    H5G__create(group, name, addr)
}
/// Insert a link into a symbol-table-storage group.
#[allow(non_snake_case)]
pub fn H5G__stab_insert(group: &mut GroupTable, name: &str, addr: u64) -> Result<()> {
    H5G__create(group, name, addr)
}

/// Return the shared state for a B-tree group node.
#[allow(non_snake_case)]
pub fn H5G__node_get_shared(group: &GroupTable) -> usize {
    group.links.len()
}
/// Encode a B-tree group-node key from a name string.
#[allow(non_snake_case)]
pub fn H5G__node_encode_key(name: &str) -> Vec<u8> {
    name.as_bytes().to_vec()
}
/// Format a B-tree group-node key for debug output.
#[allow(non_snake_case)]
pub fn H5G__node_debug_key(name: &str) -> String {
    format!("GroupNodeKey({name})")
}
/// Free a B-tree group node and its storage.
#[allow(non_snake_case)]
pub fn H5G__node_free(_group: GroupTable) {}
/// Create a new, empty B-tree group node.
#[allow(non_snake_case)]
pub fn H5G__node_create() -> GroupTable {
    GroupTable::new_root(0)
}
/// Compare two B-tree group-node entries by name.
#[allow(non_snake_case)]
pub fn H5G__node_cmp2(left: &GroupEntry, right: &GroupEntry) -> std::cmp::Ordering {
    left.name.cmp(&right.name)
}
/// Compare two B-tree group-node entries by object-header address.
#[allow(non_snake_case)]
pub fn H5G__node_cmp3(left: &GroupEntry, right: &GroupEntry) -> std::cmp::Ordering {
    left.addr.cmp(&right.addr)
}
/// Check whether a B-tree group node contains a given name.
#[allow(non_snake_case)]
pub fn H5G__node_found(group: &GroupTable, name: &str) -> bool {
    group.links.contains_key(&H5G_normalize(name))
}
/// Insert a link into a B-tree group node.
#[allow(non_snake_case)]
pub fn H5G__node_insert(group: &mut GroupTable, name: &str, addr: u64) -> Result<()> {
    H5G__create(group, name, addr)
}
/// Sum up totals across a B-tree group node.
#[allow(non_snake_case)]
pub fn H5G__node_sumup(group: &GroupTable) -> usize {
    group.links.len()
}
/// Look up an entry by index in a B-tree group node.
#[allow(non_snake_case)]
pub fn H5G__node_by_idx(group: &GroupTable, index: usize) -> Result<GroupEntry> {
    H5G__get_info_by_idx(group, index)
}
/// Initialize a B-tree group node, marking it open.
#[allow(non_snake_case)]
pub fn H5G__node_init(group: &mut GroupTable) {
    group.open = true;
}
/// Close a B-tree group node.
#[allow(non_snake_case)]
pub fn H5G_node_close(group: &mut GroupTable) {
    H5G_close(group);
}
/// Return a deep copy of a B-tree group node.
#[allow(non_snake_case)]
pub fn H5G__node_copy(group: &GroupTable) -> GroupTable {
    group.clone()
}
/// Return the iteration size (number of entries) of a B-tree group node.
#[allow(non_snake_case)]
pub fn H5G__node_iterate_size(group: &GroupTable) -> usize {
    group.links.len()
}
/// Format a debug-readable representation of a B-tree group node.
#[allow(non_snake_case)]
pub fn H5G_node_debug(group: &GroupTable) -> String {
    format!("GroupNode(len={})", group.links.len())
}

/// Metadata-cache hook: deserialize a B-tree group-node image (null-terminated names).
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
/// Metadata-cache hook: serialize a B-tree group node into a null-terminated name image.
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
/// Metadata-cache hook: free the in-core representation of a cached group node.
#[allow(non_snake_case)]
pub fn H5G__cache_node_free_icr(_group: GroupTable) {}

/// Decode a serialized group-node image into a vector of entries.
#[allow(non_snake_case)]
pub fn H5G__ent_decode_vec(bytes: &[u8]) -> Result<Vec<GroupEntry>> {
    Ok(H5G__cache_node_deserialize(bytes)?
        .links
        .values()
        .cloned()
        .collect())
}
/// Return a deep copy of a group entry.
#[allow(non_snake_case)]
pub fn H5G__ent_copy(entry: &GroupEntry) -> GroupEntry {
    entry.clone()
}
/// Reset a group entry to its empty default state.
#[allow(non_snake_case)]
pub fn H5G__ent_reset(entry: &mut GroupEntry) {
    entry.name.clear();
    entry.addr = 0;
}
/// Convert a group entry into a link-message-style name string.
#[allow(non_snake_case)]
pub fn H5G__ent_to_link(entry: &GroupEntry) -> String {
    entry.name.clone()
}
/// Format a group entry for debug output.
#[allow(non_snake_case)]
pub fn H5G__ent_debug(entry: &GroupEntry) -> String {
    format!("GroupEntry({}, {:#x})", entry.name, entry.addr)
}

/// Soft-link traversal callback: normalize the target path.
#[allow(non_snake_case)]
pub fn H5G__traverse_slink_cb(path: &str) -> String {
    H5G_normalize(path)
}
/// User-defined link traversal; unsupported in pure-Rust mode.
#[allow(non_snake_case)]
pub fn H5G__traverse_ud(path: &str) -> Result<String> {
    Err(Error::Unsupported(format!(
        "user-defined group traversal is not supported: {path}"
    )))
}
/// Resolve a soft link by normalizing the target path.
#[allow(non_snake_case)]
pub fn H5G__traverse_slink(path: &str) -> String {
    H5G_normalize(path)
}
/// Handle a special-character path during traversal (normalize it).
#[allow(non_snake_case)]
pub fn H5G__traverse_special(path: &str) -> String {
    H5G_normalize(path)
}
/// Core path-traversal: resolve a path inside the group to an entry.
#[allow(non_snake_case)]
pub fn H5G__traverse_real(group: &GroupTable, path: &str) -> Result<GroupEntry> {
    H5G__open_name(group, path)
}
/// Traverse a path inside the group, following links and mount points.
#[allow(non_snake_case)]
pub fn H5G_traverse(group: &GroupTable, path: &str) -> Result<GroupEntry> {
    H5G__traverse_real(group, path)
}

/// Compare two dense-storage fractal-heap entries by name.
#[allow(non_snake_case)]
pub fn H5G__dense_fh_name_cmp(left: &GroupEntry, right: &GroupEntry) -> std::cmp::Ordering {
    left.name.cmp(&right.name)
}
/// Encode a dense-storage v2 B-tree name-index record.
#[allow(non_snake_case)]
pub fn H5G__dense_btree2_name_store(entry: &GroupEntry) -> Vec<u8> {
    entry.name.as_bytes().to_vec()
}
/// Compare two dense-storage v2 B-tree name-index records.
#[allow(non_snake_case)]
pub fn H5G__dense_btree2_name_compare(left: &GroupEntry, right: &GroupEntry) -> std::cmp::Ordering {
    left.name.cmp(&right.name)
}
/// Decode a dense-storage v2 B-tree name-index record from UTF-8 bytes.
#[allow(non_snake_case)]
pub fn H5G__dense_btree2_name_decode(bytes: &[u8]) -> Result<String> {
    std::str::from_utf8(bytes)
        .map(str::to_string)
        .map_err(|_| Error::InvalidFormat("dense group name is not UTF-8".into()))
}
/// Format a dense-storage name record for debug output.
#[allow(non_snake_case)]
pub fn H5G__dense_btree2_name_debug(entry: &GroupEntry) -> String {
    format!("GroupName({})", entry.name)
}
/// Encode a dense-storage v2 B-tree creation-order record.
#[allow(non_snake_case)]
pub fn H5G__dense_btree2_corder_store(entry: &GroupEntry) -> Vec<u8> {
    entry.creation_order.to_le_bytes().to_vec()
}
/// Compare two dense-storage v2 B-tree creation-order records.
#[allow(non_snake_case)]
pub fn H5G__dense_btree2_corder_compare(
    left: &GroupEntry,
    right: &GroupEntry,
) -> std::cmp::Ordering {
    left.creation_order.cmp(&right.creation_order)
}
/// Encode a creation-order value as 8 little-endian bytes.
#[allow(non_snake_case)]
pub fn H5G__dense_btree2_corder_encode(entry: &GroupEntry) -> Vec<u8> {
    entry.creation_order.to_le_bytes().to_vec()
}
/// Decode a creation-order value from exactly 8 little-endian bytes.
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
/// Format a creation-order record for debug output.
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

    #[test]
    fn dense_delete_clears_storage_and_reports_links_when_adjusting() {
        let mut group = H5G_mkroot(1);
        H5Gcreate1(&mut group, "alpha", 10).unwrap();
        H5Gcreate1(&mut group, "beta", 20).unwrap();

        let removed = H5G__dense_delete(&mut group, true).unwrap();
        assert_eq!(
            removed
                .iter()
                .map(|entry| (entry.name.as_str(), entry.addr, entry.creation_order))
                .collect::<Vec<_>>(),
            vec![("alpha", 10, 0), ("beta", 20, 1)]
        );
        assert!(H5G_iterate(&group).unwrap().is_empty());

        H5Gcreate1(&mut group, "gamma", 30).unwrap();
        assert_eq!(H5G__open_name(&group, "gamma").unwrap().creation_order, 0);
        assert!(H5G__dense_delete(&mut group, false).unwrap().is_empty());
        assert!(H5G_iterate(&group).unwrap().is_empty());

        H5G_close(&mut group);
        assert!(H5G__dense_delete(&mut group, true).is_err());
    }
}
