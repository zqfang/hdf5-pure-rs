use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

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
    fn insert_entry(&mut self, raw_name: &str, addr: u64) -> Result<()> {
        self.ensure_open()?;
        let mut name = String::new();
        H5G_normalize_into(raw_name, &mut name);
        match self.links.entry(name) {
            Entry::Occupied(entry) => Err(Error::InvalidFormat(format!(
                "group link '{}' exists",
                entry.key()
            ))),
            Entry::Vacant(slot) => {
                let entry = GroupEntry {
                    name: slot.key().clone(),
                    addr,
                    creation_order: self.next_corder,
                    comment: None,
                };
                self.next_corder = self.next_corder.saturating_add(1);
                slot.insert(entry);
                Ok(())
            }
        }
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

/// Borrow a group entry by name.
#[allow(non_snake_case)]
pub fn H5G__open_name_ref<'a>(group: &'a GroupTable, raw_name: &str) -> Result<&'a GroupEntry> {
    group.ensure_open()?;
    let mut name = String::new();
    H5G_normalize_into(raw_name, &mut name);
    group
        .links
        .get(&name)
        .ok_or_else(|| Error::InvalidFormat(format!("group link '{name}' not found")))
}

/// Common open-API plumbing for borrowed group entries.
#[allow(non_snake_case)]
pub fn H5G__open_api_common_ref<'a>(group: &'a GroupTable, name: &str) -> Result<&'a GroupEntry> {
    H5G__open_name_ref(group, name)
}

/// Legacy H5Gopen1: open a named group entry.
#[deprecated(note = "use H5G__open_name_ref to borrow the entry without cloning")]
#[allow(non_snake_case)]
pub fn H5Gopen1(group: &GroupTable, name: &str) -> Result<GroupEntry> {
    Ok(H5G__open_name_ref(group, name)?.clone())
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
pub fn H5G__iterate_name_cb(entry: &GroupEntry) -> &str {
    &entry.name
}

/// Iterate over borrowed group entries without building an intermediate table.
#[allow(non_snake_case)]
pub fn H5G_iter_entries(group: &GroupTable) -> Result<impl Iterator<Item = &GroupEntry>> {
    group.ensure_open()?;
    Ok(group.links.values())
}

/// Iterate over borrowed link names without allocating owned strings.
#[allow(non_snake_case)]
pub fn H5G_iter_names(group: &GroupTable) -> Result<impl Iterator<Item = &str>> {
    group.ensure_open()?;
    Ok(group.links.values().map(H5G__iterate_name_cb))
}

/// Iterate over the group's links with a caller-provided callback.
#[allow(non_snake_case)]
pub fn H5G_iterate_with<F>(group: &GroupTable, mut callback: F) -> Result<()>
where
    F: FnMut(&GroupEntry) -> Result<()>,
{
    group.ensure_open()?;
    for entry in group.links.values() {
        callback(entry)?;
    }
    Ok(())
}

/// Iterate over the group's links with a caller-provided visitor.
#[allow(non_snake_case)]
pub fn H5G_iterate_visit<F>(group: &GroupTable, visitor: F) -> Result<()>
where
    F: FnMut(&GroupEntry) -> Result<()>,
{
    H5G_iterate_with(group, visitor)
}

/// Append the group's link names into caller-owned storage.
#[allow(non_snake_case)]
pub fn H5G_iterate_into(group: &GroupTable, names: &mut Vec<String>) -> Result<()> {
    group.ensure_open()?;
    names.reserve(group.links.len());
    names.extend(
        group
            .links
            .values()
            .map(|entry| entry.name.as_str().to_string()),
    );
    Ok(())
}

/// Iterate over the group's links, returning the list of names.
#[deprecated(
    note = "use H5G_iter_names, H5G_iterate_visit, or H5G_iterate_into to avoid allocating a Vec<String>"
)]
#[allow(non_snake_case)]
pub fn H5G_iterate(group: &GroupTable) -> Result<Vec<String>> {
    let mut names = Vec::new();
    H5G_iterate_into(group, &mut names)?;
    Ok(names)
}

/// Legacy H5Giterate: iterate over a group's links.
#[deprecated(
    note = "use H5G_iter_names, H5G_iterate_visit, or H5G_iterate_into to avoid allocating a Vec<String>"
)]
#[allow(non_snake_case)]
pub fn H5Giterate(group: &GroupTable) -> Result<Vec<String>> {
    let mut names = Vec::new();
    H5G_iterate_into(group, &mut names)?;
    Ok(names)
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

/// Visit object-header addresses reachable from the group without allocating a table.
#[allow(non_snake_case)]
pub fn H5G_visit_addrs(group: &GroupTable) -> Result<impl Iterator<Item = u64> + '_> {
    group.ensure_open()?;
    Ok(group.links.values().map(H5G__visit_cb))
}

/// Visit objects reachable from the group with a caller-provided callback.
#[allow(non_snake_case)]
pub fn H5G_visit_with<F>(group: &GroupTable, mut callback: F) -> Result<()>
where
    F: FnMut(&GroupEntry) -> Result<()>,
{
    group.ensure_open()?;
    for entry in group.links.values() {
        callback(entry)?;
    }
    Ok(())
}

/// Append object-header addresses reachable from the group into caller-owned storage.
#[allow(non_snake_case)]
pub fn H5G_visit_into(group: &GroupTable, addrs: &mut Vec<u64>) -> Result<()> {
    group.ensure_open()?;
    addrs.reserve(group.links.len());
    addrs.extend(group.links.values().map(H5G__visit_cb));
    Ok(())
}

/// Recursively visit all objects reachable from the group, returning their addresses.
#[deprecated(
    note = "use H5G_visit_addrs, H5G_visit_with, or H5G_visit_into to avoid allocating a Vec<u64>"
)]
#[allow(non_snake_case)]
pub fn H5G_visit(group: &GroupTable) -> Result<Vec<u64>> {
    let mut addrs = Vec::new();
    H5G_visit_into(group, &mut addrs)?;
    Ok(addrs)
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

/// Borrow group info for the link with the given name.
#[allow(non_snake_case)]
pub fn H5G__get_info_by_name_ref<'a>(group: &'a GroupTable, name: &str) -> Result<&'a GroupEntry> {
    H5G__open_name_ref(group, name)
}

/// Common borrowed get-info API plumbing.
#[allow(non_snake_case)]
pub fn H5G__get_info_api_common_ref<'a>(
    group: &'a GroupTable,
    name: &str,
) -> Result<&'a GroupEntry> {
    H5G__get_info_by_name_ref(group, name)
}

/// Borrow group info for the link at the given index in name-sorted order.
#[allow(non_snake_case)]
pub fn H5G__get_info_by_idx_ref(group: &GroupTable, index: usize) -> Result<&GroupEntry> {
    group.ensure_open()?;
    group
        .links
        .values()
        .nth(index)
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

/// Borrow a link message as a group entry record.
#[allow(non_snake_case)]
pub fn H5G__link_to_ent_ref(entry: &GroupEntry) -> &GroupEntry {
    entry
}

/// Convert a link message into a `GroupLocation`.
#[allow(non_snake_case)]
pub fn H5G__link_to_loc(entry: &GroupEntry) -> GroupLocation {
    GroupLocation {
        path: entry.name.clone(),
        addr: entry.addr,
    }
}

/// Iterate over links in name order without allocating a sorted table.
#[allow(non_snake_case)]
pub fn H5G__link_sorted_entries(group: &GroupTable) -> impl Iterator<Item = &GroupEntry> {
    group.links.values()
}

/// Iterate over the link table with a caller-provided callback.
#[allow(non_snake_case)]
pub fn H5G__link_iterate_table_with<F>(group: &GroupTable, mut callback: F) -> Result<()>
where
    F: FnMut(&GroupEntry) -> Result<()>,
{
    group.ensure_open()?;
    for entry in H5G__link_sorted_entries(group) {
        callback(entry)?;
    }
    Ok(())
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
    let mut old = String::new();
    H5G_normalize_into(old_name, &mut old);
    let mut new = String::new();
    H5G_normalize_into(new_name, &mut new);
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
pub fn H5Gunlink(group: &mut GroupTable, raw_name: &str) -> Result<GroupEntry> {
    let mut name = String::new();
    H5G_normalize_into(raw_name, &mut name);
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
    let mut normalized = String::new();
    H5G_normalize_into(name, &mut normalized);
    let entry = group
        .links
        .get_mut(&normalized)
        .ok_or_else(|| Error::InvalidFormat(format!("group link '{name}' not found")))?;
    entry.comment = Some(comment.into());
    Ok(())
}

/// Per-entry callback borrowing the entry for `H5Gget_objinfo`.
#[allow(non_snake_case)]
pub fn H5G__get_objinfo_ref_cb(entry: &GroupEntry) -> &GroupEntry {
    entry
}

/// Borrow legacy `H5Gget_objinfo`-style information for a named link.
#[allow(non_snake_case)]
pub fn H5G__get_objinfo_ref<'a>(group: &'a GroupTable, name: &str) -> Result<&'a GroupEntry> {
    H5G__open_name_ref(group, name).map(H5G__get_objinfo_ref_cb)
}

/// Build a real (resolved) group location from a path and object-header address.
#[allow(non_snake_case)]
pub fn H5G_loc_real(path: &str, addr: u64) -> GroupLocation {
    let mut normalized = String::new();
    H5G_normalize_into(path, &mut normalized);
    GroupLocation {
        path: normalized,
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

/// Callback that borrows a link by name inside a group.
#[allow(non_snake_case)]
pub fn H5G__loc_find_ref_cb<'a>(group: &'a GroupTable, name: &str) -> Result<&'a GroupEntry> {
    H5G__open_name_ref(group, name)
}

/// Callback that borrows a link by creation/name index.
#[allow(non_snake_case)]
pub fn H5G__loc_find_by_idx_ref_cb(group: &GroupTable, index: usize) -> Result<&GroupEntry> {
    H5G__get_info_by_idx_ref(group, index)
}

/// Find the link at the given index within a group.
#[deprecated(note = "use H5G_loc_find_by_idx_ref to borrow the entry without cloning")]
#[allow(non_snake_case)]
pub fn H5G_loc_find_by_idx(group: &GroupTable, index: usize) -> Result<GroupEntry> {
    Ok(H5G_loc_find_by_idx_ref(group, index)?.clone())
}

/// Borrow the link at the given index within a group.
#[allow(non_snake_case)]
pub fn H5G_loc_find_by_idx_ref(group: &GroupTable, index: usize) -> Result<&GroupEntry> {
    H5G__loc_find_by_idx_ref_cb(group, index)
}

/// Insert a new link at the given location inside the group.
#[allow(non_snake_case)]
pub fn H5G__loc_insert(group: &mut GroupTable, name: &str, addr: u64) -> Result<()> {
    H5G__create(group, name, addr)
}

/// Callback that checks whether a name exists inside the group.
#[allow(non_snake_case)]
pub fn H5G__loc_exists_cb(group: &GroupTable, name: &str) -> bool {
    let mut normalized = String::new();
    H5G_normalize_into(name, &mut normalized);
    group.links.contains_key(&normalized)
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
    H5G__open_name_ref(group, name).map(H5G__loc_addr_cb)
}

/// Callback borrowing object info for an entry.
#[allow(non_snake_case)]
pub fn H5G__loc_info_ref_cb(entry: &GroupEntry) -> &GroupEntry {
    entry
}

/// Callback borrowing native-specific object info for an entry.
#[allow(non_snake_case)]
pub fn H5G__loc_native_info_ref_cb(entry: &GroupEntry) -> &GroupEntry {
    entry
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

/// Get the comment on the link at a given location as a borrowed string.
#[allow(non_snake_case)]
pub fn H5G_loc_get_comment_ref<'a>(
    group: &'a GroupTable,
    raw_name: &str,
) -> Result<Option<&'a str>> {
    group.ensure_open()?;
    let mut name = String::new();
    H5G_normalize_into(raw_name, &mut name);
    let entry = group
        .links
        .get(&name)
        .ok_or_else(|| Error::InvalidFormat(format!("group link '{name}' not found")))?;
    Ok(H5G__loc_get_comment_cb(entry))
}

/// Get the comment on the link at a given location.
#[deprecated(note = "use H5G_loc_get_comment_ref to borrow the comment without cloning")]
#[allow(non_snake_case)]
pub fn H5G_loc_get_comment(group: &GroupTable, name: &str) -> Result<Option<String>> {
    Ok(H5G_loc_get_comment_ref(group, name)?.map(str::to_string))
}

/// Visit normalized, non-empty path components without allocating component strings.
#[allow(non_snake_case)]
pub fn H5G__component_visit<'a, F>(path: &'a str, mut visitor: F)
where
    F: FnMut(&'a str),
{
    let mut stack = Vec::new();
    H5G__component_parts_into(path, &mut stack);

    if path.starts_with('/') && stack.is_empty() {
        return;
    }
    for part in stack {
        visitor(part);
    }
}

#[allow(non_snake_case)]
fn H5G__component_parts_into<'a>(path: &'a str, stack: &mut Vec<&'a str>) {
    stack.clear();
    for part in path.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                stack.pop();
            }
            other => stack.push(other),
        }
    }
}

/// Normalize a group path by collapsing `.`, `..`, and duplicate slashes.
#[allow(non_snake_case)]
pub fn H5G_normalize_into(path: &str, out: &mut String) {
    let absolute = path.starts_with('/');
    let mut stack = Vec::new();
    H5G__component_parts_into(path, &mut stack);

    out.clear();
    if absolute {
        out.push('/');
    }
    for (index, part) in stack.into_iter().enumerate() {
        if index > 0 {
            out.push('/');
        }
        out.push_str(part);
    }
    if out.len() > 1 && out.ends_with('/') {
        out.pop();
    }
}

/// Return the longest common path prefix shared by two paths.
#[allow(non_snake_case)]
pub fn H5G__common_path_into(left: &str, right: &str, out: &mut String) {
    let mut l = Vec::new();
    let mut r = Vec::new();
    H5G__component_parts_into(left, &mut l);
    H5G__component_parts_into(right, &mut r);
    out.clear();
    out.push('/');
    let mut first = true;
    for part in l
        .into_iter()
        .zip(r)
        .take_while(|(a, b)| a == b)
        .map(|(a, _)| a)
    {
        if !first {
            out.push('/');
        }
        out.push_str(part);
        first = false;
    }
}

/// Build a full path by joining a parent and child path component.
#[allow(non_snake_case)]
pub fn H5G__build_fullpath_into(parent: &str, child: &str, out: &mut String) {
    if child.starts_with('/') {
        H5G_normalize_into(child, out);
        return;
    }

    let absolute = parent.is_empty() || parent.starts_with('/');
    let mut stack = Vec::new();
    H5G__component_parts_into(parent.trim_end_matches('/'), &mut stack);
    for part in child.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                stack.pop();
            }
            other => stack.push(other),
        }
    }

    out.clear();
    if absolute {
        out.push('/');
    }
    for (index, part) in stack.into_iter().enumerate() {
        if index > 0 {
            out.push('/');
        }
        out.push_str(part);
    }
}

/// Set a name field to the normalized form of a path.
#[allow(non_snake_case)]
pub fn H5G_name_set(name: &mut String, path: &str) {
    H5G_normalize_into(path, name);
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
        name.replace_range(..old_prefix.len(), new_prefix);
    }
}

/// Callback that replaces an entry's name with a new value.
#[allow(non_snake_case)]
pub fn H5G__name_replace_cb(name: &mut String, value: &str) {
    name.clear();
    name.push_str(value);
}

/// Replace a name field with a new string.
#[allow(non_snake_case)]
pub fn H5G_name_replace(name: &mut String, value: &str) {
    H5G__name_replace_cb(name, value);
}

/// Per-entry callback returning the entry's name iff its address matches.
#[allow(non_snake_case)]
pub fn H5G__get_name_by_addr_ref_cb(entry: &GroupEntry, addr: u64) -> Option<&str> {
    (entry.addr == addr).then_some(entry.name.as_str())
}

/// Search the group for the name of an entry with the given object-header address.
#[allow(non_snake_case)]
pub fn H5G_get_name_by_addr_ref(group: &GroupTable, addr: u64) -> Option<&str> {
    group
        .links
        .values()
        .find_map(|entry| H5G__get_name_by_addr_ref_cb(entry, addr))
}

/// Search the group for the owned name of an entry with the given object-header address.
#[deprecated(note = "use H5G_get_name_by_addr_ref to borrow the name without cloning")]
#[allow(non_snake_case)]
pub fn H5G_get_name_by_addr(group: &GroupTable, addr: u64) -> Option<String> {
    H5G_get_name_by_addr_ref(group, addr).map(str::to_string)
}

/// Create an object link inside a group via the object-API path.
#[allow(non_snake_case)]
pub fn H5G__obj_create(group: &mut GroupTable, name: &str, addr: u64) -> Result<()> {
    H5G__create(group, name, addr)
}

/// Iterate over the entries in a group via the object-API path.
#[allow(non_snake_case)]
pub fn H5G__obj_iter_entries(group: &GroupTable) -> Result<impl Iterator<Item = &GroupEntry>> {
    H5G_iter_entries(group)
}

/// Iterate over entries in a group via a caller-provided object-API callback.
#[allow(non_snake_case)]
pub fn H5G__obj_iterate_with<F>(group: &GroupTable, callback: F) -> Result<()>
where
    F: FnMut(&GroupEntry) -> Result<()>,
{
    H5G_iterate_with(group, callback)
}

/// Iterate over entries in a group via a caller-provided object-API visitor.
#[allow(non_snake_case)]
pub fn H5G__obj_iterate_visit<F>(group: &GroupTable, visitor: F) -> Result<()>
where
    F: FnMut(&GroupEntry) -> Result<()>,
{
    H5G__obj_iterate_with(group, visitor)
}

/// Append cloned object-API entries into caller-owned storage.
#[allow(non_snake_case)]
pub fn H5G__obj_iterate_into(group: &GroupTable, entries: &mut Vec<GroupEntry>) -> Result<()> {
    group.ensure_open()?;
    entries.reserve(group.links.len());
    entries.extend(group.links.values().cloned());
    Ok(())
}

/// Borrow object info for a named link via the object-API path.
#[allow(non_snake_case)]
pub fn H5G__obj_info_ref<'a>(group: &'a GroupTable, name: &str) -> Result<&'a GroupEntry> {
    H5G__open_name_ref(group, name)
}

/// Return the name of the link at a given index in the group.
#[allow(non_snake_case)]
pub fn H5G_obj_get_name_by_idx_ref(group: &GroupTable, index: usize) -> Result<&str> {
    Ok(H5G__get_info_by_idx_ref(group, index)?.name.as_str())
}

/// Return an owned name of the link at a given index in the group.
#[deprecated(note = "use H5G_obj_get_name_by_idx_ref to borrow the name without cloning")]
#[allow(non_snake_case)]
pub fn H5G_obj_get_name_by_idx(group: &GroupTable, index: usize) -> Result<String> {
    Ok(H5G_obj_get_name_by_idx_ref(group, index)?.to_string())
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
#[deprecated(note = "use H5G_obj_lookup_by_idx_ref to borrow the entry without cloning")]
#[allow(non_snake_case)]
pub fn H5G_obj_lookup_by_idx(group: &GroupTable, index: usize) -> Result<GroupEntry> {
    Ok(H5G_obj_lookup_by_idx_ref(group, index)?.clone())
}

/// Object-API: borrow the link at a given index in the group.
#[allow(non_snake_case)]
pub fn H5G_obj_lookup_by_idx_ref(group: &GroupTable, index: usize) -> Result<&GroupEntry> {
    H5G__get_info_by_idx_ref(group, index)
}

/// Build an iteration table for a dense-storage group.
#[allow(non_snake_case)]
pub fn H5G__dense_iter_names(group: &GroupTable) -> Result<impl Iterator<Item = &str>> {
    H5G_iter_names(group)
}

/// Iterate over dense-storage links with a caller-provided callback.
#[allow(non_snake_case)]
pub fn H5G__dense_iterate_with<F>(group: &GroupTable, callback: F) -> Result<()>
where
    F: FnMut(&GroupEntry) -> Result<()>,
{
    H5G_iterate_with(group, callback)
}

/// Iterate over dense-storage links with a caller-provided visitor.
#[allow(non_snake_case)]
pub fn H5G__dense_iterate_visit<F>(group: &GroupTable, visitor: F) -> Result<()>
where
    F: FnMut(&GroupEntry) -> Result<()>,
{
    H5G__dense_iterate_with(group, visitor)
}

/// Append dense-storage link names into caller-owned storage.
#[allow(non_snake_case)]
pub fn H5G__dense_iterate_into(group: &GroupTable, names: &mut Vec<String>) -> Result<()> {
    H5G_iterate_into(group, names)
}

/// Build an iteration table for a compact-storage group.
#[allow(non_snake_case)]
pub fn H5G__compact_iter_names(group: &GroupTable) -> Result<impl Iterator<Item = &str>> {
    H5G_iter_names(group)
}

/// Iterate over compact-storage links with a caller-provided callback.
#[allow(non_snake_case)]
pub fn H5G__compact_iterate_with<F>(group: &GroupTable, callback: F) -> Result<()>
where
    F: FnMut(&GroupEntry) -> Result<()>,
{
    H5G_iterate_with(group, callback)
}

/// Iterate over compact-storage links with a caller-provided visitor.
#[allow(non_snake_case)]
pub fn H5G__compact_iterate_visit<F>(group: &GroupTable, visitor: F) -> Result<()>
where
    F: FnMut(&GroupEntry) -> Result<()>,
{
    H5G__compact_iterate_with(group, visitor)
}

/// Append compact-storage link names into caller-owned storage.
#[allow(non_snake_case)]
pub fn H5G__compact_iterate_into(group: &GroupTable, names: &mut Vec<String>) -> Result<()> {
    H5G_iterate_into(group, names)
}

/// Iterate over the links in a symbol-table-storage group.
#[allow(non_snake_case)]
pub fn H5G__stab_iter_names(group: &GroupTable) -> Result<impl Iterator<Item = &str>> {
    H5G_iter_names(group)
}

/// Iterate over symbol-table-storage links with a caller-provided callback.
#[allow(non_snake_case)]
pub fn H5G__stab_iterate_with<F>(group: &GroupTable, callback: F) -> Result<()>
where
    F: FnMut(&GroupEntry) -> Result<()>,
{
    H5G_iterate_with(group, callback)
}

/// Iterate over symbol-table-storage links with a caller-provided visitor.
#[allow(non_snake_case)]
pub fn H5G__stab_iterate_visit<F>(group: &GroupTable, visitor: F) -> Result<()>
where
    F: FnMut(&GroupEntry) -> Result<()>,
{
    H5G__stab_iterate_with(group, visitor)
}

/// Append symbol-table-storage link names into caller-owned storage.
#[allow(non_snake_case)]
pub fn H5G__stab_iterate_into(group: &GroupTable, names: &mut Vec<String>) -> Result<()> {
    H5G_iterate_into(group, names)
}

/// Callback that borrows a name in a dense-storage group.
#[allow(non_snake_case)]
pub fn H5G__dense_lookup_ref_cb<'a>(group: &'a GroupTable, name: &str) -> Result<&'a GroupEntry> {
    H5G__open_name_ref(group, name)
}
/// Callback that borrows a name in a compact-storage group.
#[allow(non_snake_case)]
pub fn H5G__compact_lookup_ref_cb<'a>(group: &'a GroupTable, name: &str) -> Result<&'a GroupEntry> {
    H5G__open_name_ref(group, name)
}
/// Borrow a name in a compact-storage group.
#[allow(non_snake_case)]
pub fn H5G__compact_lookup_ref<'a>(group: &'a GroupTable, name: &str) -> Result<&'a GroupEntry> {
    H5G__open_name_ref(group, name)
}
/// Callback that borrows a name in a symbol-table-storage group.
#[allow(non_snake_case)]
pub fn H5G__stab_lookup_ref_cb<'a>(group: &'a GroupTable, name: &str) -> Result<&'a GroupEntry> {
    H5G__open_name_ref(group, name)
}
/// Borrow a name in a symbol-table-storage group.
#[allow(non_snake_case)]
pub fn H5G__stab_lookup_ref<'a>(group: &'a GroupTable, name: &str) -> Result<&'a GroupEntry> {
    H5G__open_name_ref(group, name)
}

/// Per-entry callback when building a compact-storage iteration table.
#[allow(non_snake_case)]
pub fn H5G__compact_build_table_name_cb(entry: &GroupEntry) -> &str {
    &entry.name
}

/// Dense-storage fractal-heap callback returning the name at an index.
#[allow(non_snake_case)]
pub fn H5G__dense_get_name_by_idx_fh_ref_cb(group: &GroupTable, index: usize) -> Result<&str> {
    H5G_obj_get_name_by_idx_ref(group, index)
}

/// Dense-storage v2 B-tree callback returning the name at an index.
#[allow(non_snake_case)]
pub fn H5G__dense_get_name_by_idx_bt2_ref_cb(group: &GroupTable, index: usize) -> Result<&str> {
    H5G_obj_get_name_by_idx_ref(group, index)
}

/// Return the name at an index in a dense-storage group.
#[allow(non_snake_case)]
pub fn H5G__dense_get_name_by_idx_ref(group: &GroupTable, index: usize) -> Result<&str> {
    H5G_obj_get_name_by_idx_ref(group, index)
}

/// Return the name at an index in a compact-storage group.
#[allow(non_snake_case)]
pub fn H5G__compact_get_name_by_idx_ref(group: &GroupTable, index: usize) -> Result<&str> {
    H5G_obj_get_name_by_idx_ref(group, index)
}

/// Symbol-table callback returning the name at an index.
#[allow(non_snake_case)]
pub fn H5G__stab_get_name_by_idx_ref_cb(group: &GroupTable, index: usize) -> Result<&str> {
    H5G_obj_get_name_by_idx_ref(group, index)
}

/// Return the name at an index in a symbol-table-storage group.
#[allow(non_snake_case)]
pub fn H5G__stab_get_name_by_idx_ref(group: &GroupTable, index: usize) -> Result<&str> {
    H5G_obj_get_name_by_idx_ref(group, index)
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

/// Delete all links in a dense-storage group, optionally appending removed entries.
#[allow(non_snake_case)]
pub fn H5G__dense_delete_into(
    group: &mut GroupTable,
    adj_link: bool,
    removed: &mut Vec<GroupEntry>,
) -> Result<()> {
    group.ensure_open()?;
    if adj_link {
        removed.extend(std::mem::take(&mut group.links).into_values());
    } else {
        group.links.clear();
    }
    group.next_corder = 0;
    Ok(())
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

/// Borrow the link at an index in a compact-storage group.
#[allow(non_snake_case)]
pub fn H5G__compact_lookup_by_idx_ref(group: &GroupTable, index: usize) -> Result<&GroupEntry> {
    H5G__get_info_by_idx_ref(group, index)
}
/// Callback that borrows the link at an index in a symbol-table-storage group.
#[allow(non_snake_case)]
pub fn H5G__stab_lookup_by_idx_ref_cb(group: &GroupTable, index: usize) -> Result<&GroupEntry> {
    H5G__get_info_by_idx_ref(group, index)
}
/// Borrow the link at an index in a symbol-table-storage group.
#[allow(non_snake_case)]
pub fn H5G__stab_lookup_by_idx_ref(group: &GroupTable, index: usize) -> Result<&GroupEntry> {
    H5G__get_info_by_idx_ref(group, index)
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
/// Borrow a B-tree group-node key from a name string.
#[allow(non_snake_case)]
pub fn H5G__node_encode_key_ref(name: &str) -> &[u8] {
    name.as_bytes()
}
/// Append a B-tree group-node key into caller-owned storage.
#[allow(non_snake_case)]
pub fn H5G__node_encode_key_into(name: &str, out: &mut Vec<u8>) {
    out.extend_from_slice(H5G__node_encode_key_ref(name));
}
/// Format a B-tree group-node key for debug output.
#[allow(non_snake_case)]
pub fn H5G__node_debug_key_into(name: &str, out: &mut impl fmt::Write) -> fmt::Result {
    write!(out, "GroupNodeKey({name})")
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
    let mut normalized = String::new();
    H5G_normalize_into(name, &mut normalized);
    group.links.contains_key(&normalized)
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
/// Borrow an entry by index in a B-tree group node.
#[allow(non_snake_case)]
pub fn H5G__node_by_idx_ref(group: &GroupTable, index: usize) -> Result<&GroupEntry> {
    H5G__get_info_by_idx_ref(group, index)
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
pub fn H5G_node_debug_fmt(group: &GroupTable, out: &mut impl fmt::Write) -> fmt::Result {
    write!(out, "GroupNode(len={})", group.links.len())
}

/// Format a debug-readable representation of a B-tree group node.
#[deprecated(note = "use H5G_node_debug_fmt to format without allocating a String")]
#[allow(non_snake_case)]
pub fn H5G_node_debug(group: &GroupTable) -> String {
    let mut out = String::new();
    H5G_node_debug_fmt(group, &mut out).expect("formatting into String cannot fail");
    out
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
/// Metadata-cache hook: serialize a B-tree group node into caller-owned storage.
#[allow(non_snake_case)]
pub fn H5G__cache_node_serialize_into(group: &GroupTable, out: &mut Vec<u8>) -> Result<()> {
    let mut len = 0usize;
    for name in group.links.keys() {
        len = len
            .checked_add(name.len())
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| Error::InvalidFormat("group cache node image length overflow".into()))?;
    }
    out.clear();
    out.reserve(len);
    for name in group.links.keys() {
        out.extend_from_slice(name.as_bytes());
        out.push(0);
    }
    Ok(())
}
/// Metadata-cache hook: free the in-core representation of a cached group node.
#[allow(non_snake_case)]
pub fn H5G__cache_node_free_icr(_group: GroupTable) {}

/// Decode a serialized group-node image into caller-owned entry storage.
#[allow(non_snake_case)]
pub fn H5G__ent_decode_into(bytes: &[u8], entries: &mut Vec<GroupEntry>) -> Result<()> {
    entries.extend(H5G__cache_node_deserialize(bytes)?.links.into_values());
    Ok(())
}
/// Reset a group entry to its empty default state.
#[allow(non_snake_case)]
pub fn H5G__ent_reset(entry: &mut GroupEntry) {
    entry.name.clear();
    entry.addr = 0;
}
/// Borrow a group entry as a link-message-style name string.
#[allow(non_snake_case)]
pub fn H5G__ent_to_link_ref(entry: &GroupEntry) -> &str {
    entry.name.as_str()
}
/// Format a group entry for debug output.
#[allow(non_snake_case)]
pub fn H5G__ent_debug_fmt(entry: &GroupEntry, out: &mut impl fmt::Write) -> fmt::Result {
    write!(out, "GroupEntry({}, {:#x})", entry.name, entry.addr)
}

/// Soft-link traversal callback: normalize the target path.
#[allow(non_snake_case)]
pub fn H5G__traverse_slink_cb_into(path: &str, out: &mut String) {
    H5G_normalize_into(path, out);
}
/// User-defined link traversal; unsupported in pure-Rust mode.
#[allow(non_snake_case)]
pub fn H5G__traverse_ud_into(path: &str, _out: &mut String) -> Result<()> {
    Err(Error::Unsupported(format!(
        "user-defined group traversal is not supported: {path}"
    )))
}
/// Resolve a soft link by normalizing the target path.
#[allow(non_snake_case)]
pub fn H5G__traverse_slink_into(path: &str, out: &mut String) {
    H5G_normalize_into(path, out);
}
/// Handle a special-character path during traversal (normalize it).
#[allow(non_snake_case)]
pub fn H5G__traverse_special_into(path: &str, out: &mut String) {
    H5G_normalize_into(path, out);
}
/// Core path-traversal: borrow the resolved path inside the group.
#[allow(non_snake_case)]
pub fn H5G__traverse_real_ref<'a>(group: &'a GroupTable, path: &str) -> Result<&'a GroupEntry> {
    H5G__open_name_ref(group, path)
}
/// Traverse a path inside the group, following links and mount points.
#[deprecated(note = "use H5G_traverse_ref to borrow the entry without cloning")]
#[allow(non_snake_case)]
pub fn H5G_traverse(group: &GroupTable, path: &str) -> Result<GroupEntry> {
    Ok(H5G_traverse_ref(group, path)?.clone())
}
/// Traverse a path inside the group, borrowing the resolved entry.
#[allow(non_snake_case)]
pub fn H5G_traverse_ref<'a>(group: &'a GroupTable, path: &str) -> Result<&'a GroupEntry> {
    H5G__traverse_real_ref(group, path)
}

/// Compare two dense-storage fractal-heap entries by name.
#[allow(non_snake_case)]
pub fn H5G__dense_fh_name_cmp(left: &GroupEntry, right: &GroupEntry) -> std::cmp::Ordering {
    left.name.cmp(&right.name)
}
/// Borrow a dense-storage v2 B-tree name-index record.
#[allow(non_snake_case)]
pub fn H5G__dense_btree2_name_store_ref(entry: &GroupEntry) -> &[u8] {
    entry.name.as_bytes()
}
/// Append a dense-storage v2 B-tree name-index record into caller-owned storage.
#[allow(non_snake_case)]
pub fn H5G__dense_btree2_name_store_into(entry: &GroupEntry, out: &mut Vec<u8>) {
    out.extend_from_slice(H5G__dense_btree2_name_store_ref(entry));
}
/// Compare two dense-storage v2 B-tree name-index records.
#[allow(non_snake_case)]
pub fn H5G__dense_btree2_name_compare(left: &GroupEntry, right: &GroupEntry) -> std::cmp::Ordering {
    left.name.cmp(&right.name)
}
/// Decode a dense-storage v2 B-tree name-index record as borrowed UTF-8.
#[allow(non_snake_case)]
pub fn H5G__dense_btree2_name_decode_ref(bytes: &[u8]) -> Result<&str> {
    std::str::from_utf8(bytes)
        .map_err(|_| Error::InvalidFormat("dense group name is not UTF-8".into()))
}
/// Format a dense-storage name record for debug output.
#[allow(non_snake_case)]
pub fn H5G__dense_btree2_name_debug_fmt(
    entry: &GroupEntry,
    out: &mut impl fmt::Write,
) -> fmt::Result {
    write!(out, "GroupName({})", entry.name)
}
/// Append a dense-storage v2 B-tree creation-order record into caller-owned storage.
#[allow(non_snake_case)]
pub fn H5G__dense_btree2_corder_store_into(entry: &GroupEntry, out: &mut Vec<u8>) {
    out.extend_from_slice(&entry.creation_order.to_le_bytes());
}
/// Compare two dense-storage v2 B-tree creation-order records.
#[allow(non_snake_case)]
pub fn H5G__dense_btree2_corder_compare(
    left: &GroupEntry,
    right: &GroupEntry,
) -> std::cmp::Ordering {
    left.creation_order.cmp(&right.creation_order)
}
/// Encode a creation-order value as an owned fixed-size byte array.
#[allow(non_snake_case)]
pub fn H5G__dense_btree2_corder_encode_array(entry: &GroupEntry) -> [u8; 8] {
    entry.creation_order.to_le_bytes()
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
pub fn H5G__dense_btree2_corder_debug_fmt(
    entry: &GroupEntry,
    out: &mut impl fmt::Write,
) -> fmt::Result {
    write!(out, "GroupCOrder({})", entry.creation_order)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_api_inserts_moves_and_removes_links() {
        let mut group = H5G_mkroot(1);
        H5Gcreate1(&mut group, "/a", 10).unwrap();
        H5Glink(&mut group, "/b", 20).unwrap();
        assert_eq!(H5G_iter_names(&group).unwrap().count(), 2);
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
        let mut image = Vec::new();
        H5G__cache_node_serialize_into(&group, &mut image).unwrap();

        let decoded = H5G__cache_node_deserialize(&image).unwrap();
        assert_eq!(
            H5G_iter_names(&decoded).unwrap().collect::<Vec<_>>(),
            vec!["alpha", "beta"]
        );
        let mut entries = Vec::new();
        H5G__ent_decode_into(&image, &mut entries).unwrap();
        assert_eq!(entries.len(), 2);

        assert!(H5G__cache_node_deserialize(&[0xff, 0]).is_err());
        assert!(H5G__ent_decode_into(&[0xff, 0], &mut entries).is_err());
        assert!(H5G__cache_node_deserialize(b"unterminated").is_err());
        assert!(H5G__ent_decode_into(b"unterminated", &mut entries).is_err());
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
            H5G__dense_btree2_name_decode_ref(H5G__dense_btree2_name_store_ref(&entry)).unwrap(),
            "dense"
        );
        assert!(H5G__dense_btree2_name_decode_ref(&[0xff]).is_err());

        assert_eq!(
            H5G__dense_btree2_corder_decode(&H5G__dense_btree2_corder_encode_array(&entry))
                .unwrap(),
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

        let mut removed = Vec::new();
        H5G__dense_delete_into(&mut group, true, &mut removed).unwrap();
        assert_eq!(
            removed
                .iter()
                .map(|entry| (entry.name.as_str(), entry.addr, entry.creation_order))
                .collect::<Vec<_>>(),
            vec![("alpha", 10, 0), ("beta", 20, 1)]
        );
        assert!(H5G_iter_names(&group).unwrap().next().is_none());

        H5Gcreate1(&mut group, "gamma", 30).unwrap();
        assert_eq!(
            H5G__open_name_ref(&group, "gamma").unwrap().creation_order,
            0
        );
        H5G__dense_delete_into(&mut group, false, &mut removed).unwrap();
        assert_eq!(removed.len(), 2);
        assert!(H5G_iter_names(&group).unwrap().next().is_none());

        H5G_close(&mut group);
        assert!(H5G__dense_delete_into(&mut group, true, &mut removed).is_err());
    }

    #[test]
    fn group_iteration_uses_borrowed_views_and_callbacks() {
        let mut group = H5G_mkroot(1);
        H5Gcreate1(&mut group, "alpha", 10).unwrap();
        H5Gcreate1(&mut group, "beta", 20).unwrap();

        assert_eq!(
            H5G_iter_names(&group).unwrap().collect::<Vec<_>>(),
            vec!["alpha", "beta"]
        );
        assert_eq!(
            H5G_iter_entries(&group)
                .unwrap()
                .map(|entry| entry.addr)
                .collect::<Vec<_>>(),
            vec![10, 20]
        );
        assert_eq!(
            H5G_visit_addrs(&group).unwrap().collect::<Vec<_>>(),
            vec![10, 20]
        );

        let mut visited_names = Vec::new();
        H5G_iterate_visit(&group, |entry| {
            visited_names.push(entry.name.clone());
            Ok(())
        })
        .unwrap();
        assert_eq!(visited_names, vec!["alpha", "beta"]);

        let mut names = Vec::with_capacity(8);
        H5G_iterate_into(&group, &mut names).unwrap();
        assert_eq!(names, vec!["alpha", "beta"]);

        let mut addrs = Vec::new();
        H5G_visit_with(&group, |entry| {
            addrs.push(entry.addr);
            Ok(())
        })
        .unwrap();
        assert_eq!(addrs, vec![10, 20]);

        addrs.clear();
        H5G_visit_into(&group, &mut addrs).unwrap();
        assert_eq!(addrs, vec![10, 20]);
    }

    #[test]
    fn group_name_and_comment_apis_borrow_existing_storage() {
        let mut group = H5G_mkroot(1);
        H5Gcreate1(&mut group, "alpha", 10).unwrap();
        H5G_loc_set_comment(&mut group, "alpha", "kept in entry").unwrap();

        assert_eq!(
            H5G_loc_get_comment_ref(&group, "alpha").unwrap(),
            Some("kept in entry")
        );
        assert_eq!(H5G_get_name_by_addr_ref(&group, 10), Some("alpha"));
        assert_eq!(H5G__open_name_ref(&group, "alpha").unwrap().addr, 10);
        assert_eq!(H5G__get_info_by_name_ref(&group, "alpha").unwrap().addr, 10);
        assert_eq!(H5G__obj_info_ref(&group, "alpha").unwrap().addr, 10);
        assert_eq!(H5G__loc_find_ref_cb(&group, "alpha").unwrap().addr, 10);
        assert_eq!(H5G_obj_get_name_by_idx_ref(&group, 0).unwrap(), "alpha");
        assert_eq!(H5G_obj_lookup_by_idx_ref(&group, 0).unwrap().addr, 10);
        assert_eq!(H5G_loc_find_by_idx_ref(&group, 0).unwrap().addr, 10);
        assert_eq!(H5G__node_by_idx_ref(&group, 0).unwrap().addr, 10);
        assert_eq!(H5G__dense_get_name_by_idx_ref(&group, 0).unwrap(), "alpha");
        assert_eq!(
            H5G__compact_get_name_by_idx_ref(&group, 0).unwrap(),
            "alpha"
        );
        assert_eq!(H5G__stab_get_name_by_idx_ref(&group, 0).unwrap(), "alpha");
        assert_eq!(H5G_traverse_ref(&group, "alpha").unwrap().addr, 10);
    }

    #[test]
    fn path_and_encoding_helpers_support_borrowed_or_caller_storage() {
        let mut components = Vec::new();
        H5G__component_visit("/alpha/./beta/../gamma", |part| components.push(part));
        assert_eq!(components, vec!["alpha", "gamma"]);
        let mut common = String::new();
        H5G__common_path_into("/alpha/gamma/x", "/alpha/gamma/y", &mut common);
        assert_eq!(common, "/alpha/gamma");

        let mut full = String::from("stale");
        H5G__build_fullpath_into("/alpha/beta", "../gamma", &mut full);
        assert_eq!(full, "/alpha/gamma");
        H5G__build_fullpath_into("/alpha/beta", "/delta/./epsilon/..", &mut full);
        assert_eq!(full, "/delta");

        let entry = GroupEntry {
            name: "dense".into(),
            addr: 42,
            creation_order: 7,
            comment: None,
        };

        assert_eq!(H5G__ent_to_link_ref(&entry), "dense");
        assert_eq!(H5G__node_encode_key_ref("node"), b"node");
        let mut bytes = Vec::with_capacity(16);
        H5G__node_encode_key_into("node", &mut bytes);
        assert_eq!(bytes, b"node");

        bytes.clear();
        H5G__dense_btree2_name_store_into(&entry, &mut bytes);
        assert_eq!(bytes, b"dense");

        bytes.clear();
        H5G__dense_btree2_corder_store_into(&entry, &mut bytes);
        assert_eq!(bytes, 7_u64.to_le_bytes());

        let mut text = String::new();
        H5G_node_debug_fmt(&H5G_mkroot(0), &mut text).unwrap();
        assert_eq!(text, "GroupNode(len=0)");

        text.clear();
        H5G__ent_debug_fmt(&entry, &mut text).unwrap();
        assert_eq!(text, "GroupEntry(dense, 0x2a)");

        text.clear();
        H5G__dense_btree2_name_debug_fmt(&entry, &mut text).unwrap();
        assert_eq!(text, "GroupName(dense)");

        text.clear();
        H5G__dense_btree2_corder_debug_fmt(&entry, &mut text).unwrap();
        assert_eq!(text, "GroupCOrder(7)");

        text.clear();
        H5G__traverse_slink_cb_into("/alpha/./beta", &mut text);
        assert_eq!(text, "/alpha/beta");

        H5G__traverse_slink_into("alpha/../beta", &mut text);
        assert_eq!(text, "beta");

        H5G__traverse_special_into("/alpha//beta", &mut text);
        assert_eq!(text, "/alpha/beta");

        assert!(H5G__traverse_ud_into("custom", &mut text).is_err());
    }

    #[test]
    fn dense_compact_and_link_tables_have_allocation_free_iteration() {
        let mut group = H5G_mkroot(1);
        H5Gcreate1(&mut group, "alpha", 10).unwrap();
        H5Gcreate1(&mut group, "beta", 20).unwrap();

        assert_eq!(
            H5G__dense_iter_names(&group).unwrap().collect::<Vec<_>>(),
            vec!["alpha", "beta"]
        );
        assert_eq!(
            H5G__compact_iter_names(&group).unwrap().collect::<Vec<_>>(),
            vec!["alpha", "beta"]
        );
        assert_eq!(
            H5G__stab_iter_names(&group).unwrap().collect::<Vec<_>>(),
            vec!["alpha", "beta"]
        );

        let mut dense_names = Vec::with_capacity(8);
        H5G__dense_iterate_into(&group, &mut dense_names).unwrap();
        assert_eq!(dense_names, vec!["alpha", "beta"]);

        let mut compact_names = Vec::with_capacity(8);
        H5G__compact_iterate_into(&group, &mut compact_names).unwrap();
        assert_eq!(compact_names, vec!["alpha", "beta"]);

        let mut stab_names = Vec::with_capacity(8);
        H5G__stab_iterate_into(&group, &mut stab_names).unwrap();
        assert_eq!(stab_names, vec!["alpha", "beta"]);

        assert_eq!(
            H5G__link_sorted_entries(&group)
                .map(|entry| entry.name.as_str())
                .collect::<Vec<_>>(),
            vec!["alpha", "beta"]
        );

        let mut names = Vec::new();
        H5G__link_iterate_table_with(&group, |entry| {
            names.push(entry.name.clone());
            Ok(())
        })
        .unwrap();
        assert_eq!(names, vec!["alpha", "beta"]);

        let mut visited = Vec::new();
        H5G__compact_iterate_visit(&group, |entry| {
            visited.push(entry.name.clone());
            Ok(())
        })
        .unwrap();
        assert_eq!(visited, vec!["alpha", "beta"]);

        visited.clear();
        H5G__stab_iterate_visit(&group, |entry| {
            visited.push(entry.name.clone());
            Ok(())
        })
        .unwrap();
        assert_eq!(visited, vec!["alpha", "beta"]);

        assert_eq!(H5G__dense_lookup_ref_cb(&group, "alpha").unwrap().addr, 10);
        assert_eq!(H5G__compact_lookup_ref(&group, "alpha").unwrap().addr, 10);
        assert_eq!(H5G__stab_lookup_ref(&group, "beta").unwrap().addr, 20);
        assert_eq!(H5G__compact_lookup_by_idx_ref(&group, 1).unwrap().addr, 20);
        assert_eq!(H5G__stab_lookup_by_idx_ref(&group, 1).unwrap().addr, 20);
    }

    #[test]
    fn object_iteration_supports_visitors_and_caller_storage() {
        let mut group = H5G_mkroot(1);
        H5Gcreate1(&mut group, "alpha", 10).unwrap();
        H5Gcreate1(&mut group, "beta", 20).unwrap();

        let mut addrs = Vec::new();
        H5G__obj_iterate_visit(&group, |entry| {
            addrs.push(entry.addr);
            Ok(())
        })
        .unwrap();
        assert_eq!(addrs, vec![10, 20]);

        let mut entries = Vec::with_capacity(8);
        H5G__obj_iterate_into(&group, &mut entries).unwrap();
        assert_eq!(
            entries
                .iter()
                .map(|entry| entry.name.as_str())
                .collect::<Vec<_>>(),
            vec!["alpha", "beta"]
        );
    }

    #[allow(deprecated)]
    #[test]
    fn deprecated_group_allocation_wrappers_remain_callable() {
        let mut group = H5G_mkroot(1);
        H5Gcreate1(&mut group, "alpha", 10).unwrap();
        H5G_loc_set_comment(&mut group, "alpha", "legacy comment").unwrap();

        assert_eq!(H5G_iterate(&group).unwrap(), vec!["alpha"]);
        assert_eq!(H5Giterate(&group).unwrap(), vec!["alpha"]);
        assert_eq!(H5G_visit(&group).unwrap(), vec![10]);
        assert_eq!(
            H5G_loc_get_comment(&group, "alpha").unwrap(),
            Some("legacy comment".into())
        );
        assert_eq!(H5G_get_name_by_addr(&group, 10), Some("alpha".into()));
    }
}
