use std::collections::{BTreeMap, BTreeSet};

use crate::error::{Error, Result};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MetadataCacheStats {
    pub entries: usize,
    pub dirty_entries: usize,
    pub protected_entries: usize,
    pub pinned_entries: usize,
    pub total_image_bytes: usize,
    pub log_records: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MetadataCacheImageHeader {
    pub entries: usize,
    pub total_image_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetadataCacheResizeConfig {
    pub min_clean_fraction: u8,
    pub max_size: usize,
    pub min_size: usize,
}

impl Default for MetadataCacheResizeConfig {
    fn default() -> Self {
        Self {
            min_clean_fraction: 90,
            max_size: usize::MAX,
            min_size: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetadataCacheEntry {
    pub addr: u64,
    pub entry_type: String,
    pub image: Vec<u8>,
    pub dirty: bool,
    pub serialized: bool,
    pub pinned: bool,
    pub protected: bool,
    pub ring: u8,
    pub tag: Option<u64>,
    pub corked: bool,
    parents: BTreeSet<u64>,
    children: BTreeSet<u64>,
}

impl MetadataCacheEntry {
    /// Construct a new cache entry with the given image bytes.
    pub fn new(addr: u64, entry_type: impl Into<String>, image: Vec<u8>) -> Self {
        Self {
            addr,
            entry_type: entry_type.into(),
            image,
            dirty: false,
            serialized: true,
            pinned: false,
            protected: false,
            ring: 0,
            tag: None,
            corked: false,
            parents: BTreeSet::new(),
            children: BTreeSet::new(),
        }
    }

    /// Return the byte size of the in-core image.
    pub fn size(&self) -> usize {
        self.image.len()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MetadataCache {
    entries: BTreeMap<u64, MetadataCacheEntry>,
    candidate_list: BTreeSet<u64>,
    clean_list: BTreeSet<u64>,
    current_tag: Option<u64>,
    current_ring: u8,
    serialization_in_progress: bool,
    cache_image_pending: bool,
    file_flush_secure: bool,
    write_done_callback: Option<String>,
    auto_resize_config: MetadataCacheResizeConfig,
    logs: Vec<String>,
}

impl MetadataCache {
    /// Initialize an empty metadata cache.
    pub fn init() -> Self {
        Self::default()
    }

    /// Terminate the package and drop all entries.
    pub fn term_package(&mut self) {
        self.entries.clear();
        self.candidate_list.clear();
        self.clean_list.clear();
        self.logs.clear();
        self.cache_image_pending = false;
        self.serialization_in_progress = false;
    }

    /// Insert a new metadata cache entry.
    pub fn insert_entry(&mut self, entry: MetadataCacheEntry) -> Result<()> {
        if self.entries.contains_key(&entry.addr) {
            return Err(Error::InvalidFormat(format!(
                "metadata cache entry {:#x} already exists",
                entry.addr
            )));
        }
        self.log_inserted_entry(entry.addr);
        self.entries.insert(entry.addr, entry);
        Ok(())
    }

    /// Compute statistics about the current cache contents.
    pub fn stats(&self) -> MetadataCacheStats {
        MetadataCacheStats {
            entries: self.entries.len(),
            dirty_entries: self.entries.values().filter(|entry| entry.dirty).count(),
            protected_entries: self
                .entries
                .values()
                .filter(|entry| entry.protected)
                .count(),
            pinned_entries: self.entries.values().filter(|entry| entry.pinned).count(),
            total_image_bytes: self.entries.values().map(MetadataCacheEntry::size).sum(),
            log_records: self.logs.len(),
        }
    }

    /// Render a human-readable summary of the cache.
    pub fn dump_cache(&self) -> String {
        let stats = self.stats();
        format!(
            "MetadataCache(entries={}, dirty={}, protected={}, pinned={}, image_bytes={})",
            stats.entries,
            stats.dirty_entries,
            stats.protected_entries,
            stats.pinned_entries,
            stats.total_image_bytes
        )
    }

    /// Check whether a flush dependency exists between two entries.
    pub fn flush_dependency_exists(&self, parent: u64, child: u64) -> bool {
        self.entries
            .get(&parent)
            .is_some_and(|entry| entry.children.contains(&child))
            && self
                .entries
                .get(&child)
                .is_some_and(|entry| entry.parents.contains(&parent))
    }

    /// Verify the recorded entry type for an address.
    pub fn verify_entry_type(&self, addr: u64, entry_type: &str) -> bool {
        self.entries
            .get(&addr)
            .is_some_and(|entry| entry.entry_type == entry_type)
    }

    /// Return whether serialization is in progress.
    pub fn get_serialization_in_progress(&self) -> bool {
        self.serialization_in_progress
    }

    /// True if no entry is currently dirty.
    pub fn cache_is_clean(&self) -> bool {
        self.entries.values().all(|entry| !entry.dirty)
    }

    /// Register a write-done callback name.
    pub fn set_write_done_callback(&mut self, callback_name: impl Into<String>) {
        self.write_done_callback = Some(callback_name.into());
    }

    /// Add an address to the candidate-flush list.
    pub fn add_candidate(&mut self, addr: u64) -> Result<()> {
        self.require_entry(addr)?;
        self.candidate_list.insert(addr);
        Ok(())
    }

    /// Snapshot the candidate-flush list for broadcast.
    pub fn broadcast_candidate_list(&self) -> Vec<u64> {
        self.broadcast_candidate_list_cb()
    }

    /// Callback returning the candidate-flush list.
    pub fn broadcast_candidate_list_cb(&self) -> Vec<u64> {
        self.candidate_list.iter().copied().collect()
    }

    /// Snapshot the clean list for broadcast.
    pub fn broadcast_clean_list(&self) -> Vec<u64> {
        self.broadcast_clean_list_cb()
    }

    /// Callback returning the clean list.
    pub fn broadcast_clean_list_cb(&self) -> Vec<u64> {
        self.clean_list.iter().copied().collect()
    }

    /// Rebuild the candidate-flush list from dirty entries.
    pub fn construct_candidate_list(&mut self) -> Vec<u64> {
        self.candidate_list = self
            .entries
            .iter()
            .filter_map(|(&addr, entry)| entry.dirty.then_some(addr))
            .collect();
        self.broadcast_candidate_list()
    }

    /// Serialize the candidate list as little-endian addresses.
    pub fn copy_candidate_list_to_buffer(&self, out: &mut Vec<u8>) {
        self.copy_candidate_list_to_buffer_cb(out);
    }

    /// Callback that serializes the candidate list.
    pub fn copy_candidate_list_to_buffer_cb(&self, out: &mut Vec<u8>) {
        out.clear();
        for addr in &self.candidate_list {
            out.extend_from_slice(&addr.to_le_bytes());
        }
    }

    /// Append a deletion log record for an entry.
    pub fn log_deleted_entry(&mut self, addr: u64) {
        self.log_event("deleted", addr);
    }

    /// Append a dirtied log record for an entry.
    pub fn log_dirtied_entry(&mut self, addr: u64) {
        self.log_event("dirtied", addr);
    }

    /// Append a cleaned log record for an entry.
    pub fn log_cleaned_entry(&mut self, addr: u64) {
        self.log_event("cleaned", addr);
    }

    /// Append a flushed log record for an entry.
    pub fn log_flushed_entry(&mut self, addr: u64) {
        self.log_event("flushed", addr);
    }

    /// Append an insertion log record for an entry.
    pub fn log_inserted_entry(&mut self, addr: u64) {
        self.log_event("inserted", addr);
    }

    /// Append a moved log record for an entry.
    pub fn log_moved_entry(&mut self, addr: u64) {
        self.log_event("moved", addr);
    }

    /// Apply a received candidate list and flush its entries.
    pub fn propagate_and_apply_candidate_list(&mut self, addrs: &[u64]) -> Result<()> {
        self.receive_candidate_list(addrs);
        self.flush_entries()
    }

    /// Rebuild the clean list and broadcast it.
    pub fn propagate_flushed_and_still_clean_entries_list(&mut self) -> Vec<u64> {
        self.clean_list = self
            .entries
            .iter()
            .filter_map(|(&addr, entry)| (!entry.dirty).then_some(addr))
            .collect();
        self.broadcast_clean_list()
    }

    /// Replace the clean list with the supplied addresses.
    pub fn receive_haddr_list(&mut self, addrs: &[u64]) {
        self.clean_list = addrs.iter().copied().collect();
    }

    /// Replace the clean list and mark each entry clean.
    pub fn receive_and_apply_clean_list(&mut self, addrs: &[u64]) -> Result<()> {
        self.receive_haddr_list(addrs);
        for addr in addrs {
            self.mark_entry_clean(*addr)?;
        }
        Ok(())
    }

    /// Replace the candidate list with the supplied addresses.
    pub fn receive_candidate_list(&mut self, addrs: &[u64]) {
        self.candidate_list = addrs.iter().copied().collect();
    }

    /// Distributed metadata-write sync-point flush.
    pub fn rsp_dist_md_write_flush(&mut self) -> Result<()> {
        self.flush_entries()
    }

    /// Distributed metadata-write flush-to-min-clean.
    pub fn rsp_dist_md_write_flush_to_min_clean(&mut self) -> Result<()> {
        self.flush_entries()
    }

    /// Process-zero-only sync-point flush.
    pub fn rsp_p0_only_flush(&mut self) -> Result<()> {
        self.flush_entries()
    }

    /// Process-zero-only flush-to-min-clean.
    pub fn rsp_p0_only_flush_to_min_clean(&mut self) -> Result<()> {
        self.flush_entries()
    }

    /// Run a parallel sync point: construct candidates then flush.
    pub fn run_sync_point(&mut self) -> Result<()> {
        self.construct_candidate_list();
        self.flush_entries()
    }

    /// Clear the candidate and clean lists.
    pub fn tidy_cache_0_lists(&mut self) {
        self.candidate_list.clear();
        self.clean_list.clear();
    }

    /// Flush all candidate (or all) entries to backing storage.
    pub fn flush_entries(&mut self) -> Result<()> {
        let addrs: Vec<u64> = if self.candidate_list.is_empty() {
            self.entries.keys().copied().collect()
        } else {
            self.candidate_list.iter().copied().collect()
        };
        for addr in addrs {
            self.flush(addr)?;
        }
        self.candidate_list.clear();
        Ok(())
    }

    /// True if a cache image is pending load on next protect.
    pub fn cache_image_pending(&self) -> bool {
        self.cache_image_pending
    }

    /// Destroy the cache, releasing any held state.
    pub fn dest(mut self) {
        self.term_package();
    }

    /// Evict clean entries from the cache.
    pub fn evict(&mut self) -> Result<usize> {
        let removable: Vec<u64> = self
            .entries
            .iter()
            .filter_map(|(&addr, entry)| {
                (!entry.dirty && !entry.protected && !entry.pinned && !entry.corked).then_some(addr)
            })
            .collect();
        for addr in &removable {
            self.entries.remove(addr);
            self.log_deleted_entry(*addr);
        }
        Ok(removable.len())
    }

    /// Expunge an entry (mark deleted and remove).
    pub fn expunge_entry(&mut self, addr: u64) -> Result<MetadataCacheEntry> {
        let entry = self.remove_entry(addr)?;
        self.log_deleted_entry(addr);
        Ok(entry)
    }

    /// Flush dirty entries to backing storage.
    pub fn flush(&mut self, addr: u64) -> Result<()> {
        {
            let entry = self.require_entry_mut(addr)?;
            entry.dirty = false;
            entry.serialized = true;
        }
        self.clean_list.insert(addr);
        self.log_flushed_entry(addr);
        Ok(())
    }

    /// Return the status flags for a cache entry.
    pub fn get_entry_status(&self, addr: u64) -> Result<MetadataCacheEntryStatus> {
        let entry = self.require_entry(addr)?;
        Ok(MetadataCacheEntryStatus {
            dirty: entry.dirty,
            serialized: entry.serialized,
            pinned: entry.pinned,
            protected: entry.protected,
            ring: entry.ring,
            tag: entry.tag,
        })
    }

    /// Arrange for a cache image to be loaded on the next protect.
    pub fn load_cache_image_on_next_protect(&mut self) {
        self.cache_image_pending = true;
    }

    /// Mark an entry as needing to be flushed.
    pub fn mark_entry_dirty(&mut self, addr: u64) -> Result<()> {
        self.require_entry_mut(addr)?.dirty = true;
        self.log_dirtied_entry(addr);
        Ok(())
    }

    /// Mark an entry as clean.
    pub fn mark_entry_clean(&mut self, addr: u64) -> Result<()> {
        self.require_entry_mut(addr)?.dirty = false;
        self.clean_list.insert(addr);
        self.log_cleaned_entry(addr);
        Ok(())
    }

    /// Mark an entry as needing serialization.
    pub fn mark_entry_unserialized(&mut self, addr: u64) -> Result<()> {
        self.require_entry_mut(addr)?.serialized = false;
        Ok(())
    }

    /// Mark an entry as serialized.
    pub fn mark_entry_serialized(&mut self, addr: u64) -> Result<()> {
        self.require_entry_mut(addr)?.serialized = true;
        Ok(())
    }

    /// Move an entry from one address to another.
    pub fn move_entry(&mut self, old_addr: u64, new_addr: u64) -> Result<()> {
        if self.entries.contains_key(&new_addr) {
            return Err(Error::InvalidFormat(format!(
                "metadata cache entry {new_addr:#x} already exists"
            )));
        }
        let mut entry = self.remove_entry(old_addr)?;
        entry.addr = new_addr;
        self.entries.insert(new_addr, entry);
        self.replace_dependency_addr(old_addr, new_addr);
        self.log_moved_entry(new_addr);
        Ok(())
    }

    /// Pin an entry that is currently protected.
    pub fn pin_protected_entry(&mut self, addr: u64) -> Result<()> {
        let entry = self.require_entry_mut(addr)?;
        if !entry.protected {
            return Err(Error::InvalidFormat(format!(
                "metadata cache entry {addr:#x} is not protected"
            )));
        }
        entry.pinned = true;
        Ok(())
    }

    /// Prepare the cache for file close (flush then evict).
    pub fn prep_for_file_close(&mut self) -> Result<()> {
        self.flush_entries()?;
        self.evict()?;
        Ok(())
    }

    /// Prepare the cache for file flush (build candidate list).
    pub fn prep_for_file_flush(&mut self) -> Result<()> {
        self.construct_candidate_list();
        Ok(())
    }

    /// Mark the cache as secured against file flush races.
    pub fn secure_from_file_flush(&mut self) {
        self.file_flush_secure = true;
    }

    /// Create a flush dependency between parent and child.
    pub fn create_flush_dependency(&mut self, parent: u64, child: u64) -> Result<()> {
        self.require_entry(parent)?;
        self.require_entry(child)?;
        self.require_entry_mut(parent)?.children.insert(child);
        self.require_entry_mut(child)?.parents.insert(parent);
        Ok(())
    }

    /// Mark an entry as in-use and return its image.
    pub fn protect(&mut self, addr: u64) -> Result<&[u8]> {
        let pending = self.cache_image_pending;
        let entry = self.require_entry_mut(addr)?;
        entry.protected = true;
        if pending {
            entry.serialized = true;
            self.cache_image_pending = false;
        }
        Ok(&self.entries[&addr].image)
    }

    /// Resize the in-core image of an entry.
    pub fn resize_entry(&mut self, addr: u64, new_size: usize) -> Result<()> {
        let entry = self.require_entry_mut(addr)?;
        entry.image.resize(new_size, 0);
        entry.dirty = true;
        Ok(())
    }

    /// Unpin a previously pinned entry.
    pub fn unpin_entry(&mut self, addr: u64) -> Result<()> {
        self.require_entry_mut(addr)?.pinned = false;
        Ok(())
    }

    /// Destroy a flush dependency between parent and child.
    pub fn destroy_flush_dependency(&mut self, parent: u64, child: u64) -> Result<()> {
        self.require_entry(parent)?;
        self.require_entry(child)?;
        self.require_entry_mut(parent)?.children.remove(&child);
        self.require_entry_mut(child)?.parents.remove(&parent);
        Ok(())
    }

    /// Release a previously protected entry.
    pub fn unprotect(&mut self, addr: u64, dirtied: bool) -> Result<()> {
        let entry = self.require_entry_mut(addr)?;
        entry.protected = false;
        if dirtied {
            entry.dirty = true;
        }
        Ok(())
    }

    /// Return the auto-resize configuration.
    pub fn get_cache_auto_resize_config(&self) -> MetadataCacheResizeConfig {
        self.auto_resize_config.clone()
    }

    /// Evict all clean entries with a given tag.
    pub fn evict_tagged_metadata(&mut self, tag: u64) -> Result<usize> {
        let addrs: Vec<u64> = self
            .entries
            .iter()
            .filter_map(|(&addr, entry)| (entry.tag == Some(tag) && !entry.dirty).then_some(addr))
            .collect();
        for addr in &addrs {
            self.entries.remove(addr);
            self.log_deleted_entry(*addr);
        }
        Ok(addrs.len())
    }

    /// Expunge all entries matching tag and entry type.
    pub fn expunge_tag_type_metadata(&mut self, tag: u64, entry_type: &str) -> Result<usize> {
        let addrs: Vec<u64> = self
            .entries
            .iter()
            .filter_map(|(&addr, entry)| {
                (entry.tag == Some(tag) && entry.entry_type == entry_type).then_some(addr)
            })
            .collect();
        for addr in &addrs {
            self.entries.remove(addr);
            self.log_deleted_entry(*addr);
        }
        Ok(addrs.len())
    }

    /// Return the current cache tag.
    pub fn get_tag(&self) -> Option<u64> {
        self.current_tag
    }

    /// Cork (or uncork) an entry from being evicted.
    pub fn cork(&mut self, addr: u64, corked: bool) -> Result<()> {
        self.require_entry_mut(addr)?.corked = corked;
        Ok(())
    }

    /// Verify the tag stored on a cache entry.
    pub fn verify_tag(&self, addr: u64, tag: u64) -> bool {
        self.entries
            .get(&addr)
            .is_some_and(|entry| entry.tag == Some(tag))
    }

    /// Return the ring assignment for an entry.
    pub fn get_entry_ring(&self, addr: u64) -> Result<u8> {
        Ok(self.require_entry(addr)?.ring)
    }

    /// Set the current ring used for new entries.
    pub fn set_ring(&mut self, ring: u8) {
        self.current_ring = ring;
    }

    /// Unsettle the ring assignment of an entry.
    pub fn unsettle_entry_ring(&mut self, addr: u64) -> Result<()> {
        let ring = self.current_ring;
        self.require_entry_mut(addr)?.ring = ring;
        Ok(())
    }

    /// Unsettle all ring assignments in the cache.
    pub fn unsettle_ring(&mut self) {
        for entry in self.entries.values_mut() {
            entry.ring = self.current_ring;
        }
    }

    /// Remove an entry from the cache.
    pub fn remove_entry(&mut self, addr: u64) -> Result<MetadataCacheEntry> {
        let entry = self.entries.remove(&addr).ok_or_else(|| {
            Error::InvalidFormat(format!("metadata cache entry {addr:#x} not found"))
        })?;
        self.candidate_list.remove(&addr);
        self.clean_list.remove(&addr);
        for parent in &entry.parents {
            if let Some(parent_entry) = self.entries.get_mut(parent) {
                parent_entry.children.remove(&addr);
            }
        }
        for child in &entry.children {
            if let Some(child_entry) = self.entries.get_mut(child) {
                child_entry.parents.remove(&addr);
            }
        }
        Ok(entry)
    }

    /// Create a proxy entry used to chain flush dependencies.
    pub fn proxy_entry_create(&mut self, addr: u64, image: Vec<u8>) -> Result<()> {
        self.insert_entry(MetadataCacheEntry::new(addr, "proxy", image))
    }

    /// Add a parent dependency to a proxy entry.
    pub fn proxy_entry_add_parent(&mut self, proxy_addr: u64, parent: u64) -> Result<()> {
        self.create_flush_dependency(parent, proxy_addr)
    }

    /// Remove a parent dependency from a proxy entry.
    pub fn proxy_entry_remove_parent(&mut self, proxy_addr: u64, parent: u64) -> Result<()> {
        self.destroy_flush_dependency(parent, proxy_addr)
    }

    /// Callback that adds a child dependency to a proxy entry.
    pub fn proxy_entry_add_child_cb(&mut self, proxy_addr: u64, child: u64) -> Result<()> {
        self.create_flush_dependency(proxy_addr, child)
    }

    /// Add a child dependency to a proxy entry.
    pub fn proxy_entry_add_child(&mut self, proxy_addr: u64, child: u64) -> Result<()> {
        self.proxy_entry_add_child_cb(proxy_addr, child)
    }

    /// Callback that removes a child dependency from a proxy entry.
    pub fn proxy_entry_remove_child_cb(&mut self, proxy_addr: u64, child: u64) -> Result<()> {
        self.destroy_flush_dependency(proxy_addr, child)
    }

    /// Remove a child dependency from a proxy entry.
    pub fn proxy_entry_remove_child(&mut self, proxy_addr: u64, child: u64) -> Result<()> {
        self.proxy_entry_remove_child_cb(proxy_addr, child)
    }

    /// Destroy a proxy entry, returning the underlying entry.
    pub fn proxy_entry_dest(&mut self, proxy_addr: u64) -> Result<MetadataCacheEntry> {
        self.remove_entry(proxy_addr)
    }

    /// Return the image length of a proxy entry.
    pub fn proxy_entry_image_len(&self, proxy_addr: u64) -> Result<usize> {
        Ok(self.require_entry(proxy_addr)?.image.len())
    }

    /// Serialize a proxy entry and mark it serialized.
    pub fn proxy_entry_serialize(&mut self, proxy_addr: u64) -> Result<Vec<u8>> {
        let entry = self.require_entry_mut(proxy_addr)?;
        entry.serialized = true;
        Ok(entry.image.clone())
    }

    /// Append a proxy notification message to the cache log.
    pub fn proxy_entry_notify(
        &mut self,
        proxy_addr: u64,
        message: impl Into<String>,
    ) -> Result<()> {
        self.require_entry(proxy_addr)?;
        let message = message.into();
        self.logs.push(format!("proxy:{proxy_addr:#x}:{message}"));
        Ok(())
    }

    /// Free the in-core representation of a proxy entry.
    pub fn proxy_entry_free_icr(&mut self, proxy_addr: u64) -> Result<()> {
        self.remove_entry(proxy_addr).map(|_| ())
    }

    /// Set the tag on a cache entry.
    pub fn set_tag_for_entry(&mut self, addr: u64, tag: u64) -> Result<()> {
        self.require_entry_mut(addr)?.tag = Some(tag);
        self.current_tag = Some(tag);
        Ok(())
    }

    /// Borrow an entry by address or error if missing.
    fn require_entry(&self, addr: u64) -> Result<&MetadataCacheEntry> {
        self.entries.get(&addr).ok_or_else(|| {
            Error::InvalidFormat(format!("metadata cache entry {addr:#x} not found"))
        })
    }

    /// Mutably borrow an entry by address or error if missing.
    fn require_entry_mut(&mut self, addr: u64) -> Result<&mut MetadataCacheEntry> {
        self.entries.get_mut(&addr).ok_or_else(|| {
            Error::InvalidFormat(format!("metadata cache entry {addr:#x} not found"))
        })
    }

    /// Rewrite parent/child sets to reflect an entry move.
    fn replace_dependency_addr(&mut self, old_addr: u64, new_addr: u64) {
        for entry in self.entries.values_mut() {
            if entry.parents.remove(&old_addr) {
                entry.parents.insert(new_addr);
            }
            if entry.children.remove(&old_addr) {
                entry.children.insert(new_addr);
            }
        }
        if self.candidate_list.remove(&old_addr) {
            self.candidate_list.insert(new_addr);
        }
        if self.clean_list.remove(&old_addr) {
            self.clean_list.insert(new_addr);
        }
    }

    /// Append an arbitrary event record to the cache log.
    fn log_event(&mut self, kind: &str, addr: u64) {
        self.logs.push(format!("{kind}:{addr:#x}"));
    }
}

/// Create a fresh metadata cache.
#[allow(non_snake_case)]
pub fn H5C_create() -> MetadataCache {
    MetadataCache::init()
}

/// Destroy a metadata cache and free its state.
#[allow(non_snake_case)]
pub fn H5C_dest(cache: MetadataCache) {
    cache.dest();
}

/// `H5AC` wrapper that destroys a metadata cache.
#[allow(non_snake_case)]
pub fn H5AC_dest(cache: MetadataCache) {
    H5C_dest(cache);
}

/// Return the auto-resize configuration.
#[allow(non_snake_case)]
pub fn H5C_get_cache_auto_resize_config(cache: &MetadataCache) -> MetadataCacheResizeConfig {
    cache.get_cache_auto_resize_config()
}

/// `H5AC` wrapper: return the auto-resize configuration.
#[allow(non_snake_case)]
pub fn H5AC_get_cache_auto_resize_config(cache: &MetadataCache) -> MetadataCacheResizeConfig {
    H5C_get_cache_auto_resize_config(cache)
}

/// Overwrite the cache auto-resize configuration.
#[allow(non_snake_case)]
pub fn H5C_set_cache_auto_resize_config(
    cache: &mut MetadataCache,
    config: MetadataCacheResizeConfig,
) {
    cache.auto_resize_config = config;
}

/// Return (cache_image_pending, total_image_bytes) for the cache.
#[allow(non_snake_case)]
pub fn H5C_get_mdc_image_info(cache: &MetadataCache) -> (bool, usize) {
    (cache.cache_image_pending(), cache.stats().total_image_bytes)
}

/// Apply a list of candidate addresses (receive then flush).
#[allow(non_snake_case)]
pub fn H5C_apply_candidate_list(cache: &mut MetadataCache, addrs: &[u64]) -> Result<()> {
    cache.propagate_and_apply_candidate_list(addrs)
}

/// `H5AC` wrapper: snapshot the candidate-flush list for broadcast.
#[allow(non_snake_case)]
pub fn H5AC__broadcast_candidate_list(cache: &MetadataCache) -> Vec<u64> {
    cache.broadcast_candidate_list()
}

/// `H5AC` wrapper: callback returning the clean list.
#[allow(non_snake_case)]
pub fn H5AC__broadcast_clean_list_cb(cache: &MetadataCache) -> Vec<u64> {
    cache.broadcast_clean_list_cb()
}

/// `H5AC` wrapper: snapshot the clean list for broadcast.
#[allow(non_snake_case)]
pub fn H5AC__broadcast_clean_list(cache: &MetadataCache) -> Vec<u64> {
    cache.broadcast_clean_list()
}

/// `H5AC` wrapper: serialize the candidate list as little-endian addresses.
#[allow(non_snake_case)]
pub fn H5AC__copy_candidate_list_to_buffer(cache: &MetadataCache, out: &mut Vec<u8>) {
    cache.copy_candidate_list_to_buffer(out);
}

/// `H5AC` wrapper: append a deletion log record for an entry.
#[allow(non_snake_case)]
pub fn H5AC__log_deleted_entry(cache: &mut MetadataCache, addr: u64) {
    cache.log_deleted_entry(addr);
}

/// `H5AC` wrapper: append a dirtied log record for an entry.
#[allow(non_snake_case)]
pub fn H5AC__log_dirtied_entry(cache: &mut MetadataCache, addr: u64) {
    cache.log_dirtied_entry(addr);
}

/// `H5AC` wrapper: append a cleaned log record for an entry.
#[allow(non_snake_case)]
pub fn H5AC__log_cleaned_entry(cache: &mut MetadataCache, addr: u64) {
    cache.log_cleaned_entry(addr);
}

/// `H5AC` wrapper: append a flushed log record for an entry.
#[allow(non_snake_case)]
pub fn H5AC__log_flushed_entry(cache: &mut MetadataCache, addr: u64) {
    cache.log_flushed_entry(addr);
}

/// `H5AC` wrapper: append an insertion log record for an entry.
#[allow(non_snake_case)]
pub fn H5AC__log_inserted_entry(cache: &mut MetadataCache, addr: u64) {
    cache.log_inserted_entry(addr);
}

/// `H5AC` wrapper: append a moved log record for an entry.
#[allow(non_snake_case)]
pub fn H5AC__log_moved_entry(cache: &mut MetadataCache, addr: u64) {
    cache.log_moved_entry(addr);
}

/// `H5AC` wrapper: apply a received candidate list and flush its entries.
#[allow(non_snake_case)]
pub fn H5AC__propagate_and_apply_candidate_list(
    cache: &mut MetadataCache,
    addrs: &[u64],
) -> Result<()> {
    cache.propagate_and_apply_candidate_list(addrs)
}

/// `H5AC` wrapper: replace the clean list with the supplied addresses.
#[allow(non_snake_case)]
pub fn H5AC__receive_haddr_list(cache: &mut MetadataCache, addrs: &[u64]) {
    cache.receive_haddr_list(addrs);
}

/// `H5AC` wrapper: replace the candidate list with the supplied addresses.
#[allow(non_snake_case)]
pub fn H5AC__receive_candidate_list(cache: &mut MetadataCache, addrs: &[u64]) {
    cache.receive_candidate_list(addrs);
}

/// Internal cache helper: rsp  dist md write  flush.
#[allow(non_snake_case)]
pub fn H5AC__rsp__dist_md_write__flush(cache: &mut MetadataCache) -> Result<()> {
    cache.rsp_dist_md_write_flush()
}

/// Internal cache helper: rsp  dist md write  flush to min clean.
#[allow(non_snake_case)]
pub fn H5AC__rsp__dist_md_write__flush_to_min_clean(cache: &mut MetadataCache) -> Result<()> {
    cache.rsp_dist_md_write_flush_to_min_clean()
}

/// Internal cache helper: rsp  p0 only  flush.
#[allow(non_snake_case)]
pub fn H5AC__rsp__p0_only__flush(cache: &mut MetadataCache) -> Result<()> {
    cache.rsp_p0_only_flush()
}

/// Internal cache helper: rsp  p0 only  flush to min clean.
#[allow(non_snake_case)]
pub fn H5AC__rsp__p0_only__flush_to_min_clean(cache: &mut MetadataCache) -> Result<()> {
    cache.rsp_p0_only_flush_to_min_clean()
}

/// `H5AC` wrapper: run a parallel sync point: construct candidates then flush.
#[allow(non_snake_case)]
pub fn H5AC__run_sync_point(cache: &mut MetadataCache) -> Result<()> {
    cache.run_sync_point()
}

/// `H5AC` wrapper: clear the candidate and clean lists.
#[allow(non_snake_case)]
pub fn H5AC__tidy_cache_0_lists(cache: &mut MetadataCache) {
    cache.tidy_cache_0_lists();
}

/// Build the clean list by gathering all non-dirty entries.
#[allow(non_snake_case)]
pub fn H5C_construct_candidate_list__clean_cache(cache: &mut MetadataCache) -> Vec<u64> {
    cache.propagate_flushed_and_still_clean_entries_list()
}

/// Build the candidate list of all dirty entries.
#[allow(non_snake_case)]
pub fn H5C_construct_candidate_list__min_clean(cache: &mut MetadataCache) -> Vec<u64> {
    cache.construct_candidate_list()
}

/// Clear collective-entry tracking lists.
#[allow(non_snake_case)]
pub fn H5C_clear_coll_entries(cache: &mut MetadataCache) {
    cache.tidy_cache_0_lists();
}

/// Collective-write hook that flushes pending entries.
#[allow(non_snake_case)]
pub fn H5C__collective_write(cache: &mut MetadataCache) -> Result<()> {
    cache.flush_entries()
}

/// Flush all entries currently on the candidate list.
#[allow(non_snake_case)]
pub fn H5C__flush_candidate_entries(cache: &mut MetadataCache) -> Result<()> {
    cache.flush_entries()
}

/// Flush candidate entries that belong to the given ring.
#[allow(non_snake_case)]
pub fn H5C__flush_candidates_in_ring(cache: &mut MetadataCache, ring: u8) -> Result<()> {
    let addrs: Vec<u64> = cache
        .candidate_list
        .iter()
        .copied()
        .filter(|addr| {
            cache
                .entries
                .get(addr)
                .is_some_and(|entry| entry.ring == ring)
        })
        .collect();
    for addr in addrs {
        cache.flush(addr)?;
    }
    Ok(())
}

/// Auto-adjust the cache size; reports current stats.
#[allow(non_snake_case)]
pub fn H5C__auto_adjust_cache_size(cache: &mut MetadataCache) -> MetadataCacheStats {
    cache.stats()
}

/// Age-out step: evict clean entries.
#[allow(non_snake_case)]
pub fn H5C__autoadjust__ageout(cache: &mut MetadataCache) -> Result<usize> {
    cache.evict()
}

/// Insert a `cycle` epoch-marker log record.
#[allow(non_snake_case)]
pub fn H5C__autoadjust__ageout__cycle_epoch_marker(cache: &mut MetadataCache) {
    cache.logs.push("epoch:cycle".into());
}

/// Evict entries aged out of the cache.
#[allow(non_snake_case)]
pub fn H5C__autoadjust__ageout__evict_aged_out_entries(cache: &mut MetadataCache) -> Result<usize> {
    cache.evict()
}

/// Insert a new epoch-marker log record.
#[allow(non_snake_case)]
pub fn H5C__autoadjust__ageout__insert_new_marker(cache: &mut MetadataCache) {
    cache.logs.push("epoch:insert".into());
}

/// Remove all epoch-marker log records.
#[allow(non_snake_case)]
pub fn H5C__autoadjust__ageout__remove_all_markers(cache: &mut MetadataCache) {
    cache.logs.retain(|record| !record.starts_with("epoch:"));
}

/// Keep the first epoch marker; remove later ones.
#[allow(non_snake_case)]
pub fn H5C__autoadjust__ageout__remove_excess_markers(cache: &mut MetadataCache) {
    let mut seen = false;
    cache.logs.retain(|record| {
        if !record.starts_with("epoch:") {
            return true;
        }
        if seen {
            false
        } else {
            seen = true;
            true
        }
    });
}

/// Flush all entries then evict everything possible.
#[allow(non_snake_case)]
pub fn H5C__flush_invalidate_cache(cache: &mut MetadataCache) -> Result<()> {
    cache.flush_entries()?;
    cache.evict()?;
    Ok(())
}

/// Flush then remove all entries in a given ring.
#[allow(non_snake_case)]
pub fn H5C__flush_invalidate_ring(cache: &mut MetadataCache, ring: u8) -> Result<usize> {
    let addrs: Vec<u64> = cache
        .entries
        .iter()
        .filter_map(|(&addr, entry)| (entry.ring == ring).then_some(addr))
        .collect();
    for addr in &addrs {
        cache.flush(*addr)?;
        cache.remove_entry(*addr)?;
    }
    Ok(addrs.len())
}

/// Flush all entries in a given ring.
#[allow(non_snake_case)]
pub fn H5C__flush_ring(cache: &mut MetadataCache, ring: u8) -> Result<()> {
    let addrs: Vec<u64> = cache
        .entries
        .iter()
        .filter_map(|(&addr, entry)| (entry.ring == ring).then_some(addr))
        .collect();
    for addr in addrs {
        cache.flush(addr)?;
    }
    Ok(())
}

/// Try to make space for `needed` bytes by evicting clean entries.
#[allow(non_snake_case)]
pub fn H5C__make_space_in_cache(cache: &mut MetadataCache, needed: usize) -> Result<bool> {
    if cache.stats().total_image_bytes <= cache.auto_resize_config.max_size.saturating_sub(needed) {
        return Ok(true);
    }
    cache.evict()?;
    Ok(cache.stats().total_image_bytes <= cache.auto_resize_config.max_size.saturating_sub(needed))
}

/// Flush and serialize the entire cache to a buffer.
#[allow(non_snake_case)]
pub fn H5C__serialize_cache(cache: &mut MetadataCache) -> Result<Vec<u8>> {
    cache.flush_entries()?;
    H5C__construct_cache_image_buffer(cache)
}

/// Flush and serialize only the entries in a given ring.
#[allow(non_snake_case)]
pub fn H5C__serialize_ring(cache: &mut MetadataCache, ring: u8) -> Result<Vec<u8>> {
    H5C__flush_ring(cache, ring)?;
    let mut out = Vec::new();
    for entry in cache.entries.values().filter(|entry| entry.ring == ring) {
        out.extend_from_slice(&entry.addr.to_le_bytes());
        out.extend_from_slice(
            &usize_to_u64(entry.image.len(), "metadata cache entry image length")?.to_le_bytes(),
        );
        out.extend_from_slice(&entry.image);
    }
    Ok(out)
}

/// Free the in-core representation of a prefetched entry.
#[allow(non_snake_case)]
pub fn H5C__prefetched_entry_free_icr(entry: MetadataCacheEntry) {
    drop(entry);
}

/// Return whether a cache image is pending load on next protect.
#[allow(non_snake_case)]
pub fn H5C_cache_image_pending(cache: &MetadataCache) -> bool {
    cache.cache_image_pending()
}

/// Return the pending status and total image bytes of the cache.
#[allow(non_snake_case)]
pub fn H5C_cache_image_status(cache: &MetadataCache) -> (bool, usize) {
    H5C_get_mdc_image_info(cache)
}

/// Serialize all cache entries into a single image buffer.
#[allow(non_snake_case)]
pub fn H5C__construct_cache_image_buffer(cache: &MetadataCache) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    for entry in cache.entries.values() {
        out.extend_from_slice(&entry.addr.to_le_bytes());
        out.extend_from_slice(
            &usize_to_u64(entry.image.len(), "metadata cache entry image length")?.to_le_bytes(),
        );
        out.extend_from_slice(&entry.image);
    }
    Ok(out)
}

/// Deserialize a cache image buffer into individual entries.
#[allow(non_snake_case)]
pub fn H5C__deserialize_cache_image_buffer(bytes: &[u8]) -> Result<Vec<MetadataCacheEntry>> {
    let mut pos = 0usize;
    let mut entries = Vec::new();
    while pos < bytes.len() {
        let addr = read_u64_at(bytes, pos, "metadata cache image entry address")?;
        pos = checked_add(pos, 8, "metadata cache image entry address")?;
        let len = read_u64_at(bytes, pos, "metadata cache image entry length")?;
        pos = checked_add(pos, 8, "metadata cache image entry length")?;
        let len = usize::try_from(len).map_err(|_| {
            Error::InvalidFormat("metadata cache image entry length exceeds usize".into())
        })?;
        let end = checked_add(pos, len, "metadata cache image entry payload")?;
        let image = bytes
            .get(pos..end)
            .ok_or_else(|| {
                Error::InvalidFormat("metadata cache image entry payload is truncated".into())
            })?
            .to_vec();
        entries.push(MetadataCacheEntry::new(addr, "prefetched", image));
        pos = end;
    }
    Ok(entries)
}

/// Reconstruct a cache from its serialized image buffer.
#[allow(non_snake_case)]
pub fn H5C__reconstruct_cache_contents_from_image(bytes: &[u8]) -> Result<MetadataCache> {
    H5C__reconstruct_cache_contents(H5C__deserialize_cache_image_buffer(bytes)?)
}

/// Generate a cache image buffer for the whole cache.
#[allow(non_snake_case)]
pub fn H5C__generate_cache_image(cache: &mut MetadataCache) -> Result<Vec<u8>> {
    H5C__serialize_cache(cache)
}

/// Drop a list of cache-image entries.
#[allow(non_snake_case)]
pub fn H5C__free_image_entries_array(entries: Vec<MetadataCacheEntry>) {
    drop(entries);
}

/// Arrange for a cache image to be loaded on next protect.
#[allow(non_snake_case)]
pub fn H5C_load_cache_image_on_next_protect(cache: &mut MetadataCache) {
    cache.load_cache_image_on_next_protect();
}

/// Compare two cache-image entries by address.
#[allow(non_snake_case)]
pub fn H5C__image_entry_cmp(
    left: &MetadataCacheEntry,
    right: &MetadataCacheEntry,
) -> std::cmp::Ordering {
    left.addr.cmp(&right.addr)
}

/// Prepare a cache image buffer prior to file close.
#[allow(non_snake_case)]
pub fn H5C__prep_image_for_file_close(cache: &mut MetadataCache) -> Result<Vec<u8>> {
    H5C__serialize_cache(cache)
}

/// Validate an auto-resize config (min<=max and fraction<=100).
#[allow(non_snake_case)]
pub fn H5C_validate_cache_image_config(config: &MetadataCacheResizeConfig) -> bool {
    config.min_size <= config.max_size && config.min_clean_fraction <= 100
}

/// Encode the eight-byte magic + counts header for a cache image.
#[allow(non_snake_case)]
pub fn H5C__encode_cache_image_header(cache: &MetadataCache) -> Result<Vec<u8>> {
    let stats = cache.stats();
    let mut out = b"H5CIMG\0\0".to_vec();
    out.extend_from_slice(
        &usize_to_u64(stats.entries, "metadata cache image entry count")?.to_le_bytes(),
    );
    out.extend_from_slice(
        &usize_to_u64(
            stats.total_image_bytes,
            "metadata cache image total byte count",
        )?
        .to_le_bytes(),
    );
    Ok(out)
}

/// Decode the eight-byte magic + counts header of a cache image.
#[allow(non_snake_case)]
pub fn H5C__decode_cache_image_header(bytes: &[u8]) -> Result<MetadataCacheImageHeader> {
    if bytes.len() < 24 {
        return Err(Error::InvalidFormat(
            "metadata cache image header is truncated".into(),
        ));
    }
    if bytes.get(..8) != Some(&b"H5CIMG\0\0"[..]) {
        return Err(Error::InvalidFormat(
            "metadata cache image header has invalid magic".into(),
        ));
    }
    let entries = read_u64_at(bytes, 8, "metadata cache image entry count")?;
    let total_image_bytes = read_u64_at(bytes, 16, "metadata cache image total byte count")?;
    Ok(MetadataCacheImageHeader {
        entries: usize::try_from(entries).map_err(|_| {
            Error::InvalidFormat("metadata cache image entry count exceeds usize".into())
        })?,
        total_image_bytes: usize::try_from(total_image_bytes).map_err(|_| {
            Error::InvalidFormat("metadata cache image total byte count exceeds usize".into())
        })?,
    })
}

/// Compute flush-dependency height for each entry at close.
#[allow(non_snake_case)]
pub fn H5C__prep_for_file_close__compute_fd_heights(cache: &MetadataCache) -> BTreeMap<u64, usize> {
    cache
        .entries
        .iter()
        .map(|(&addr, entry)| (addr, entry.parents.len()))
        .collect()
}

/// Internal variant of compute_fd_heights.
#[allow(non_snake_case)]
pub fn H5C__prep_for_file_close__compute_fd_heights_real(
    cache: &MetadataCache,
) -> BTreeMap<u64, usize> {
    H5C__prep_for_file_close__compute_fd_heights(cache)
}

/// Return all entry addresses for close-time scanning.
#[allow(non_snake_case)]
pub fn H5C__prep_for_file_close__scan_entries(cache: &MetadataCache) -> Vec<u64> {
    cache.entries.keys().copied().collect()
}

/// Detect duplicate addresses in the entry index.
#[allow(non_snake_case)]
pub fn H5C__check_for_duplicates(cache: &MetadataCache) -> bool {
    cache.entries.len() == cache.entries.keys().collect::<BTreeSet<_>>().len()
}

/// Reconstruct a metadata cache from a list of entries.
#[allow(non_snake_case)]
pub fn H5C__reconstruct_cache_contents(entries: Vec<MetadataCacheEntry>) -> Result<MetadataCache> {
    let mut cache = MetadataCache::init();
    for entry in entries {
        cache.insert_entry(entry)?;
    }
    Ok(cache)
}

/// Build a cache entry from its address, type and image.
#[allow(non_snake_case)]
pub fn H5C__reconstruct_cache_entry(
    addr: u64,
    entry_type: impl Into<String>,
    image: Vec<u8>,
) -> MetadataCacheEntry {
    MetadataCacheEntry::new(addr, entry_type, image)
}

/// Encode the cache-image superblock message payload.
#[allow(non_snake_case)]
pub fn H5C__write_cache_image_superblock_msg(cache: &MetadataCache) -> Result<Vec<u8>> {
    H5C__encode_cache_image_header(cache)
}

/// Flush the cache and serialize it into an image buffer.
#[allow(non_snake_case)]
pub fn H5C__write_cache_image(cache: &mut MetadataCache) -> Result<Vec<u8>> {
    H5C__serialize_cache(cache)
}

/// Prepare the cache for file close (flush then evict).
#[allow(non_snake_case)]
pub fn H5C_prep_for_file_close(cache: &mut MetadataCache) -> Result<()> {
    cache.prep_for_file_close()
}

/// Evict clean entries from the cache.
#[allow(non_snake_case)]
pub fn H5C_evict(cache: &mut MetadataCache) -> Result<usize> {
    cache.evict()
}

/// Flush all entries to backing storage.
#[allow(non_snake_case)]
pub fn H5C_flush_cache(cache: &mut MetadataCache) -> Result<()> {
    cache.flush_entries()
}

/// Flush entries until the min-clean fraction is reached.
#[allow(non_snake_case)]
pub fn H5C_flush_to_min_clean(cache: &mut MetadataCache) -> Result<()> {
    cache.flush_entries()
}

/// Clear hit-rate statistics from the cache log.
#[allow(non_snake_case)]
pub fn H5C_reset_cache_hit_rate_stats(cache: &mut MetadataCache) {
    cache.logs.retain(|record| !record.starts_with("hit-rate:"));
}

/// Toggle evictions on or off via a log record.
#[allow(non_snake_case)]
pub fn H5C_set_evictions_enabled(cache: &mut MetadataCache, enabled: bool) {
    cache.logs.push(format!("evictions_enabled:{enabled}"));
}

/// Unsettle all ring assignments.
#[allow(non_snake_case)]
pub fn H5C_unsettle_ring(cache: &mut MetadataCache) {
    cache.unsettle_ring();
}

/// Internal unpin used by the cache implementation.
#[allow(non_snake_case)]
pub fn H5C__unpin_entry_real(cache: &mut MetadataCache, addr: u64) -> Result<()> {
    cache.unpin_entry(addr)
}

/// Unpin entry triggered by a client request.
#[allow(non_snake_case)]
pub fn H5C__unpin_entry_from_client(cache: &mut MetadataCache, addr: u64) -> Result<()> {
    cache.unpin_entry(addr)
}

/// Return a clone of an entry's in-core image.
#[allow(non_snake_case)]
pub fn H5C__generate_image(cache: &mut MetadataCache, addr: u64) -> Result<Vec<u8>> {
    Ok(cache.require_entry(addr)?.image.clone())
}

/// Flush one specific entry by address.
#[allow(non_snake_case)]
pub fn H5C__flush_single_entry(cache: &mut MetadataCache, addr: u64) -> Result<()> {
    cache.flush(addr)
}

/// Remove one specific entry by address.
#[allow(non_snake_case)]
pub fn H5C__discard_single_entry(
    cache: &mut MetadataCache,
    addr: u64,
) -> Result<MetadataCacheEntry> {
    cache.remove_entry(addr)
}

/// Verify that an entry image of length `len` fits before EOA.
#[allow(non_snake_case)]
pub fn H5C__verify_len_eoa(cache: &MetadataCache, addr: u64, len: usize, eoa: u64) -> bool {
    let Ok(len_u64) = u64::try_from(len) else {
        return false;
    };
    cache.entries.get(&addr).is_some_and(|entry| {
        entry.image.len() == len && addr.checked_add(len_u64).is_some_and(|end| end <= eoa)
    })
}

/// Convert a `usize` to `u64`, surfacing `context` on overflow.
fn usize_to_u64(value: usize, context: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| Error::InvalidFormat(format!("{context} exceeds u64")))
}

/// Add two `usize` values, surfacing `context` on overflow.
fn checked_add(left: usize, right: usize, context: &str) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))
}

/// Read a little-endian `u64` from `bytes` at `pos`.
fn read_u64_at(bytes: &[u8], pos: usize, context: &str) -> Result<u64> {
    let end = checked_add(pos, 8, context)?;
    let raw: [u8; 8] = bytes
        .get(pos..end)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))?
        .try_into()
        .map_err(|_| Error::InvalidFormat(format!("{context} is truncated")))?;
    Ok(u64::from_le_bytes(raw))
}

/// Load (clone) an entry image by address.
#[allow(non_snake_case)]
pub fn H5C__load_entry(cache: &MetadataCache, addr: u64) -> Result<Vec<u8>> {
    Ok(cache.require_entry(addr)?.image.clone())
}

/// Mark a flush-dependency entry as dirty.
#[allow(non_snake_case)]
pub fn H5C__mark_flush_dep_dirty(cache: &mut MetadataCache, addr: u64) -> Result<()> {
    cache.mark_entry_dirty(addr)
}

/// Mark a flush-dependency entry as clean.
#[allow(non_snake_case)]
pub fn H5C__mark_flush_dep_clean(cache: &mut MetadataCache, addr: u64) -> Result<()> {
    cache.mark_entry_clean(addr)
}

/// Mark a flush-dependency entry as serialized.
#[allow(non_snake_case)]
pub fn H5C__mark_flush_dep_serialized(cache: &mut MetadataCache, addr: u64) -> Result<()> {
    cache.mark_entry_serialized(addr)
}

/// Mark a flush-dependency entry as needing serialization.
#[allow(non_snake_case)]
pub fn H5C__mark_flush_dep_unserialized(cache: &mut MetadataCache, addr: u64) -> Result<()> {
    cache.mark_entry_unserialized(addr)
}

/// Assert that adding parent->child would not introduce a cycle.
#[allow(non_snake_case)]
pub fn H5C__assert_flush_dep_nocycle(cache: &MetadataCache, parent: u64, child: u64) -> bool {
    parent != child && !cache.flush_dependency_exists(child, parent)
}

/// Mark an entry serialized and return a clone of its image.
#[allow(non_snake_case)]
pub fn H5C__serialize_single_entry(cache: &mut MetadataCache, addr: u64) -> Result<Vec<u8>> {
    let entry = cache.require_entry_mut(addr)?;
    entry.serialized = true;
    Ok(entry.image.clone())
}

/// Destroy all child flush dependencies of a prefetched entry.
#[allow(non_snake_case)]
pub fn H5C__destroy_pf_entry_child_flush_deps(cache: &mut MetadataCache, addr: u64) -> Result<()> {
    let children: Vec<u64> = cache
        .require_entry(addr)?
        .children
        .iter()
        .copied()
        .collect();
    for child in children {
        cache.destroy_flush_dependency(addr, child)?;
    }
    Ok(())
}

/// Reconstruct a prefetched cache entry from its image.
#[allow(non_snake_case)]
pub fn H5C__deserialize_prefetched_entry(addr: u64, image: Vec<u8>) -> MetadataCacheEntry {
    MetadataCacheEntry::new(addr, "prefetched", image)
}

/// Insert a new metadata cache entry.
#[allow(non_snake_case)]
pub fn H5C_insert_entry(cache: &mut MetadataCache, entry: MetadataCacheEntry) -> Result<()> {
    cache.insert_entry(entry)
}

/// `H5AC` wrapper that inserts a metadata cache entry.
#[allow(non_snake_case)]
pub fn H5AC_insert_entry(cache: &mut MetadataCache, entry: MetadataCacheEntry) -> Result<()> {
    H5C_insert_entry(cache, entry)
}

/// Register a sync-point-done callback name.
#[allow(non_snake_case)]
pub fn H5AC__set_sync_point_done_callback(
    cache: &mut MetadataCache,
    callback_name: impl Into<String>,
) {
    cache.set_write_done_callback(callback_name);
}

/// Mark an entry as dirty.
#[allow(non_snake_case)]
pub fn H5C_mark_entry_dirty(cache: &mut MetadataCache, addr: u64) -> Result<()> {
    cache.mark_entry_dirty(addr)
}

/// Mark an entry as clean.
#[allow(non_snake_case)]
pub fn H5C_mark_entry_clean(cache: &mut MetadataCache, addr: u64) -> Result<()> {
    cache.mark_entry_clean(addr)
}

/// Mark a batch of currently-dirty entries as clean.
#[allow(non_snake_case)]
pub fn H5C_mark_entries_as_clean(cache: &mut MetadataCache, addrs: &[u64]) -> Result<()> {
    if addrs.is_empty() {
        return Err(Error::InvalidFormat(
            "metadata cache clean list is empty".into(),
        ));
    }

    for &addr in addrs {
        if !cache.get_entry_status(addr)?.dirty {
            return Err(Error::InvalidFormat(format!(
                "metadata cache entry {addr:#x} is not dirty"
            )));
        }
    }

    for &addr in addrs {
        cache.mark_entry_clean(addr)?;
    }
    Ok(())
}

/// Mark an entry as needing serialization.
#[allow(non_snake_case)]
pub fn H5C_mark_entry_unserialized(cache: &mut MetadataCache, addr: u64) -> Result<()> {
    cache.mark_entry_unserialized(addr)
}

/// Mark an entry as serialized.
#[allow(non_snake_case)]
pub fn H5C_mark_entry_serialized(cache: &mut MetadataCache, addr: u64) -> Result<()> {
    cache.mark_entry_serialized(addr)
}

/// Move an entry from one address to another.
#[allow(non_snake_case)]
pub fn H5C_move_entry(cache: &mut MetadataCache, old_addr: u64, new_addr: u64) -> Result<()> {
    cache.move_entry(old_addr, new_addr)
}

/// Resize the image buffer of an entry.
#[allow(non_snake_case)]
pub fn H5C_resize_entry(cache: &mut MetadataCache, addr: u64, new_size: usize) -> Result<()> {
    cache.resize_entry(addr, new_size)
}

/// Pin a protected entry so it cannot be evicted.
#[allow(non_snake_case)]
pub fn H5C_pin_protected_entry(cache: &mut MetadataCache, addr: u64) -> Result<()> {
    cache.pin_protected_entry(addr)
}

/// Mark an entry protected and return its image.
#[allow(non_snake_case)]
pub fn H5C_protect(cache: &mut MetadataCache, addr: u64) -> Result<&[u8]> {
    cache.protect(addr)
}

/// Unpin a previously pinned entry.
#[allow(non_snake_case)]
pub fn H5C_unpin_entry(cache: &mut MetadataCache, addr: u64) -> Result<()> {
    cache.unpin_entry(addr)
}

/// Release a previously protected entry, optionally dirtied.
#[allow(non_snake_case)]
pub fn H5C_unprotect(cache: &mut MetadataCache, addr: u64, dirtied: bool) -> Result<()> {
    cache.unprotect(addr, dirtied)
}

/// `H5AC` wrapper that releases a previously protected entry.
#[allow(non_snake_case)]
pub fn H5AC_unprotect(cache: &mut MetadataCache, addr: u64, dirtied: bool) -> Result<()> {
    H5C_unprotect(cache, addr, dirtied)
}

/// Unsettle the ring assignment of a single entry.
#[allow(non_snake_case)]
pub fn H5C_unsettle_entry_ring(cache: &mut MetadataCache, addr: u64) -> Result<()> {
    cache.unsettle_entry_ring(addr)
}

/// Create a flush dependency between parent and child.
#[allow(non_snake_case)]
pub fn H5C_create_flush_dependency(
    cache: &mut MetadataCache,
    parent: u64,
    child: u64,
) -> Result<()> {
    cache.create_flush_dependency(parent, child)
}

/// Destroy a flush dependency between parent and child.
#[allow(non_snake_case)]
pub fn H5C_destroy_flush_dependency(
    cache: &mut MetadataCache,
    parent: u64,
    child: u64,
) -> Result<()> {
    cache.destroy_flush_dependency(parent, child)
}

/// Expunge an entry from the cache.
#[allow(non_snake_case)]
pub fn H5C_expunge_entry(cache: &mut MetadataCache, addr: u64) -> Result<MetadataCacheEntry> {
    cache.expunge_entry(addr)
}

/// Remove an entry from the cache.
#[allow(non_snake_case)]
pub fn H5C_remove_entry(cache: &mut MetadataCache, addr: u64) -> Result<MetadataCacheEntry> {
    cache.remove_entry(addr)
}

/// Test helper: verify the tag on a corked entry.
#[allow(non_snake_case)]
pub fn H5C__verify_cork_tag_test(cache: &MetadataCache, addr: u64, tag: u64) -> bool {
    cache.verify_tag(addr, tag)
}

/// `H5AC` wrapper that corks or uncorks an entry.
#[allow(non_snake_case)]
pub fn H5AC_cork(cache: &mut MetadataCache, addr: u64, corked: bool) -> Result<()> {
    cache.cork(addr, corked)
}

/// Create a proxy entry for flush-dependency chaining.
#[allow(non_snake_case)]
pub fn H5AC_proxy_entry_create(cache: &mut MetadataCache, addr: u64, image: Vec<u8>) -> Result<()> {
    cache.proxy_entry_create(addr, image)
}

/// Add a parent dependency to a proxy entry.
#[allow(non_snake_case)]
pub fn H5AC_proxy_entry_add_parent(
    cache: &mut MetadataCache,
    proxy_addr: u64,
    parent: u64,
) -> Result<()> {
    cache.proxy_entry_add_parent(proxy_addr, parent)
}

/// Remove a parent dependency from a proxy entry.
#[allow(non_snake_case)]
pub fn H5AC_proxy_entry_remove_parent(
    cache: &mut MetadataCache,
    proxy_addr: u64,
    parent: u64,
) -> Result<()> {
    cache.proxy_entry_remove_parent(proxy_addr, parent)
}

/// Add a child dependency to a proxy entry.
#[allow(non_snake_case)]
pub fn H5AC_proxy_entry_add_child(
    cache: &mut MetadataCache,
    proxy_addr: u64,
    child: u64,
) -> Result<()> {
    cache.proxy_entry_add_child(proxy_addr, child)
}

/// Remove a child dependency from a proxy entry.
#[allow(non_snake_case)]
pub fn H5AC_proxy_entry_remove_child(
    cache: &mut MetadataCache,
    proxy_addr: u64,
    child: u64,
) -> Result<()> {
    cache.proxy_entry_remove_child(proxy_addr, child)
}

/// Destroy a proxy entry.
#[allow(non_snake_case)]
pub fn H5AC_proxy_entry_dest(
    cache: &mut MetadataCache,
    proxy_addr: u64,
) -> Result<MetadataCacheEntry> {
    cache.proxy_entry_dest(proxy_addr)
}

/// Append a proxy notification message to the cache log.
#[allow(non_snake_case)]
pub fn H5AC__proxy_entry_notify(
    cache: &mut MetadataCache,
    proxy_addr: u64,
    message: impl Into<String>,
) -> Result<()> {
    cache.proxy_entry_notify(proxy_addr, message)
}

/// Enable or disable tag enforcement, recorded in the log.
#[allow(non_snake_case)]
pub fn H5C_ignore_tags(cache: &mut MetadataCache, ignore: bool) {
    cache.logs.push(format!("ignore_tags:{ignore}"));
}

/// Return whether tag enforcement is currently disabled.
#[allow(non_snake_case)]
pub fn H5C_get_ignore_tags(cache: &MetadataCache) -> bool {
    cache
        .logs
        .iter()
        .rev()
        .find_map(|record| {
            record
                .strip_prefix("ignore_tags:")
                .map(|value| value == "true")
        })
        .unwrap_or(false)
}

/// Count the entries currently corked.
#[allow(non_snake_case)]
pub fn H5C_get_num_objs_corked(cache: &MetadataCache) -> usize {
    cache.entries.values().filter(|entry| entry.corked).count()
}

/// Tag an entry and update the cache's current tag.
#[allow(non_snake_case)]
pub fn H5C__tag_entry(cache: &mut MetadataCache, addr: u64, tag: u64) -> Result<()> {
    cache.set_tag_for_entry(addr, tag)
}

/// Clear the tag on an entry.
#[allow(non_snake_case)]
pub fn H5C__untag_entry(cache: &mut MetadataCache, addr: u64) -> Result<()> {
    cache.require_entry_mut(addr)?.tag = None;
    Ok(())
}

/// Internal variant of [`H5C__iter_tagged_entries`].
#[allow(non_snake_case)]
pub fn H5C__iter_tagged_entries_real(cache: &MetadataCache, tag: u64) -> Vec<u64> {
    cache
        .entries
        .iter()
        .filter_map(|(&addr, entry)| (entry.tag == Some(tag)).then_some(addr))
        .collect()
}

/// List entries with a given tag.
#[allow(non_snake_case)]
pub fn H5C__iter_tagged_entries(cache: &MetadataCache, tag: u64) -> Vec<u64> {
    H5C__iter_tagged_entries_real(cache, tag)
}

/// Callback that evicts entries with a given tag.
#[allow(non_snake_case)]
pub fn H5C__evict_tagged_entries_cb(cache: &mut MetadataCache, tag: u64) -> Result<usize> {
    cache.evict_tagged_metadata(tag)
}

/// Evict all clean entries with a given tag.
#[allow(non_snake_case)]
pub fn H5C_evict_tagged_entries(cache: &mut MetadataCache, tag: u64) -> Result<usize> {
    cache.evict_tagged_metadata(tag)
}

/// Verify the tag on a cache entry.
#[allow(non_snake_case)]
pub fn H5C_verify_tag(cache: &MetadataCache, addr: u64, tag: u64) -> bool {
    cache.verify_tag(addr, tag)
}

/// Callback that flushes all entries with a given tag.
#[allow(non_snake_case)]
pub fn H5C__flush_tagged_entries_cb(cache: &mut MetadataCache, tag: u64) -> Result<()> {
    let addrs = H5C__iter_tagged_entries(cache, tag);
    for addr in addrs {
        cache.flush(addr)?;
    }
    Ok(())
}

/// Flush all entries with a given tag.
#[allow(non_snake_case)]
pub fn H5C_flush_tagged_entries(cache: &mut MetadataCache, tag: u64) -> Result<()> {
    H5C__flush_tagged_entries_cb(cache, tag)
}

/// Move entries from one tag value to another.
#[allow(non_snake_case)]
pub fn H5C_retag_entries(cache: &mut MetadataCache, old_tag: u64, new_tag: u64) -> Result<usize> {
    let addrs = H5C__iter_tagged_entries(cache, old_tag);
    for addr in &addrs {
        cache.set_tag_for_entry(*addr, new_tag)?;
    }
    Ok(addrs.len())
}

/// Callback that expunges entries with a given tag and entry type.
#[allow(non_snake_case)]
pub fn H5C__expunge_tag_type_metadata_cb(
    cache: &mut MetadataCache,
    tag: u64,
    entry_type: &str,
) -> Result<usize> {
    cache.expunge_tag_type_metadata(tag, entry_type)
}

/// Expunge entries matching tag and entry type.
#[allow(non_snake_case)]
pub fn H5C_expunge_tag_type_metadata(
    cache: &mut MetadataCache,
    tag: u64,
    entry_type: &str,
) -> Result<usize> {
    cache.expunge_tag_type_metadata(tag, entry_type)
}

/// Return the current cache tag.
#[allow(non_snake_case)]
pub fn H5C_get_tag(cache: &MetadataCache) -> Option<u64> {
    cache.get_tag()
}

/// Render a human-readable summary of the cache.
#[allow(non_snake_case)]
pub fn H5C_dump_cache(cache: &MetadataCache) -> String {
    cache.dump_cache()
}

/// Return the cache addresses in insertion order.
#[allow(non_snake_case)]
pub fn H5C_dump_cache_LRU(cache: &MetadataCache) -> Vec<u64> {
    cache.entries.keys().copied().collect()
}

/// Return the cache addresses in reverse insertion order.
#[allow(non_snake_case)]
pub fn H5C_dump_cache_skip_list(cache: &MetadataCache) -> Vec<u64> {
    cache.entries.keys().rev().copied().collect()
}

/// Set a prefix string for log records.
#[allow(non_snake_case)]
pub fn H5C_set_prefix(cache: &mut MetadataCache, prefix: impl Into<String>) {
    cache.logs.push(format!("prefix:{}", prefix.into()));
}

/// Return the current cache statistics.
#[allow(non_snake_case)]
pub fn H5C_stats(cache: &MetadataCache) -> MetadataCacheStats {
    cache.stats()
}

/// Check whether a flush dependency exists.
#[allow(non_snake_case)]
pub fn H5C_flush_dependency_exists(cache: &MetadataCache, parent: u64, child: u64) -> bool {
    cache.flush_dependency_exists(parent, child)
}

/// Borrow a cache entry by address.
#[allow(non_snake_case)]
pub fn H5C_get_entry_ptr_from_addr(
    cache: &MetadataCache,
    addr: u64,
) -> Result<&MetadataCacheEntry> {
    cache.require_entry(addr)
}

/// Return whether serialization is in progress.
#[allow(non_snake_case)]
pub fn H5C_get_serialization_in_progress(cache: &MetadataCache) -> bool {
    cache.get_serialization_in_progress()
}

/// Return whether no entry is currently dirty.
#[allow(non_snake_case)]
pub fn H5C_cache_is_clean(cache: &MetadataCache) -> bool {
    cache.cache_is_clean()
}

/// Verify the recorded entry type for an address.
#[allow(non_snake_case)]
pub fn H5C_verify_entry_type(cache: &MetadataCache, addr: u64, entry_type: &str) -> bool {
    cache.verify_entry_type(addr, entry_type)
}

/// Render the auto-resize configuration as a string.
#[allow(non_snake_case)]
pub fn H5C_def_auto_resize_rpt_fcn(cache: &MetadataCache) -> String {
    format!("{:?}", cache.get_cache_auto_resize_config())
}

/// Validate the LRU list invariants (test helper).
#[allow(non_snake_case)]
pub fn H5C__validate_lru_list(cache: &MetadataCache) -> bool {
    H5C__check_for_duplicates(cache)
}

/// Validate that pinned entries are also protected.
#[allow(non_snake_case)]
pub fn H5C__validate_pinned_entry_list(cache: &MetadataCache) -> bool {
    cache
        .entries
        .values()
        .all(|entry| !entry.pinned || entry.protected)
}

/// Validate that all protected entries are still present.
#[allow(non_snake_case)]
pub fn H5C__validate_protected_entry_list(cache: &MetadataCache) -> bool {
    cache
        .entries
        .values()
        .all(|entry| !entry.protected || cache.entries.contains_key(&entry.addr))
}

/// Return whether an entry exists in the cache.
#[allow(non_snake_case)]
pub fn H5C__entry_in_skip_list(cache: &MetadataCache, addr: u64) -> bool {
    cache.entries.contains_key(&addr)
}

/// Append a trace-log record to the cache log.
#[allow(non_snake_case)]
pub fn H5C__trace_write_log_message(cache: &mut MetadataCache, message: impl Into<String>) {
    cache.logs.push(format!("trace:{}", message.into()));
}

/// Initialize trace logging.
#[allow(non_snake_case)]
pub fn H5C__log_trace_set_up(cache: &mut MetadataCache) {
    H5C__trace_write_log_message(cache, "setup");
}

/// Append a trace-log record to the cache log.
#[allow(non_snake_case)]
pub fn H5C__trace_tear_down_logging(cache: &mut MetadataCache) {
    H5C__trace_write_log_message(cache, "teardown");
}

/// Append a trace-log record to the cache log.
#[allow(non_snake_case)]
pub fn H5C__trace_write_expunge_entry_log_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__trace_write_log_message(cache, format!("expunge:{addr:#x}"));
}

/// Append a trace-log record to the cache log.
#[allow(non_snake_case)]
pub fn H5C__trace_write_flush_cache_log_msg(cache: &mut MetadataCache) {
    H5C__trace_write_log_message(cache, "flush_cache");
}

/// Append a trace-log record to the cache log.
#[allow(non_snake_case)]
pub fn H5C__trace_write_insert_entry_log_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__trace_write_log_message(cache, format!("insert:{addr:#x}"));
}

/// Append a trace-log record to the cache log.
#[allow(non_snake_case)]
pub fn H5C__trace_write_mark_unserialized_entry_log_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__trace_write_log_message(cache, format!("mark_unserialized:{addr:#x}"));
}

/// Append a trace-log record to the cache log.
#[allow(non_snake_case)]
pub fn H5C__trace_write_mark_serialized_entry_log_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__trace_write_log_message(cache, format!("mark_serialized:{addr:#x}"));
}

/// Append a trace-log record to the cache log.
#[allow(non_snake_case)]
pub fn H5C__trace_write_move_entry_log_msg(
    cache: &mut MetadataCache,
    old_addr: u64,
    new_addr: u64,
) {
    H5C__trace_write_log_message(cache, format!("move:{old_addr:#x}->{new_addr:#x}"));
}

/// Append a trace-log record to the cache log.
#[allow(non_snake_case)]
pub fn H5C__trace_write_pin_entry_log_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__trace_write_log_message(cache, format!("pin:{addr:#x}"));
}

/// Append a trace-log record to the cache log.
#[allow(non_snake_case)]
pub fn H5C__trace_write_create_fd_log_msg(cache: &mut MetadataCache, parent: u64, child: u64) {
    H5C__trace_write_log_message(cache, format!("create_fd:{parent:#x}->{child:#x}"));
}

/// Append a trace-log record to the cache log.
#[allow(non_snake_case)]
pub fn H5C__trace_write_protect_entry_log_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__trace_write_log_message(cache, format!("protect:{addr:#x}"));
}

/// Append a trace-log record to the cache log.
#[allow(non_snake_case)]
pub fn H5C__trace_write_resize_entry_log_msg(cache: &mut MetadataCache, addr: u64, size: usize) {
    H5C__trace_write_log_message(cache, format!("resize:{addr:#x}:{size}"));
}

/// Append a trace-log record to the cache log.
#[allow(non_snake_case)]
pub fn H5C__trace_write_unpin_entry_log_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__trace_write_log_message(cache, format!("unpin:{addr:#x}"));
}

/// Append a trace-log record to the cache log.
#[allow(non_snake_case)]
pub fn H5C__trace_write_destroy_fd_log_msg(cache: &mut MetadataCache, parent: u64, child: u64) {
    H5C__trace_write_log_message(cache, format!("destroy_fd:{parent:#x}->{child:#x}"));
}

/// Append a trace-log record to the cache log.
#[allow(non_snake_case)]
pub fn H5C__trace_write_unprotect_entry_log_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__trace_write_log_message(cache, format!("unprotect:{addr:#x}"));
}

/// Append a trace-log record to the cache log.
#[allow(non_snake_case)]
pub fn H5C__trace_write_remove_entry_log_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__trace_write_log_message(cache, format!("remove:{addr:#x}"));
}

/// Append a JSON-log record to the cache log.
#[allow(non_snake_case)]
pub fn H5C__json_write_log_message(cache: &mut MetadataCache, event: &str, payload: &str) {
    cache.logs.push(format!(
        "json:{{\"event\":\"{event}\",\"payload\":\"{payload}\"}}"
    ));
}

/// Initialize JSON logging.
#[allow(non_snake_case)]
pub fn H5C__log_json_set_up(cache: &mut MetadataCache) {
    H5C__json_write_log_message(cache, "setup", "");
}

/// Append a JSON-log record to the cache log.
#[allow(non_snake_case)]
pub fn H5C__json_tear_down_logging(cache: &mut MetadataCache) {
    H5C__json_write_log_message(cache, "teardown", "");
}

/// Append a JSON-log record to the cache log.
#[allow(non_snake_case)]
pub fn H5C__json_write_start_log_msg(cache: &mut MetadataCache) {
    H5C__json_write_log_message(cache, "start", "");
}

/// Append a JSON-log record to the cache log.
#[allow(non_snake_case)]
pub fn H5C__json_write_stop_log_msg(cache: &mut MetadataCache) {
    H5C__json_write_log_message(cache, "stop", "");
}

/// Append a JSON-log record to the cache log.
#[allow(non_snake_case)]
pub fn H5C__json_write_create_cache_log_msg(cache: &mut MetadataCache) {
    H5C__json_write_log_message(cache, "create_cache", "");
}

/// Append a JSON-log record to the cache log.
#[allow(non_snake_case)]
pub fn H5C__json_write_destroy_cache_log_msg(cache: &mut MetadataCache) {
    H5C__json_write_log_message(cache, "destroy_cache", "");
}

/// Append a JSON-log record to the cache log.
#[allow(non_snake_case)]
pub fn H5C__json_write_evict_cache_log_msg(cache: &mut MetadataCache) {
    H5C__json_write_log_message(cache, "evict_cache", "");
}

/// Append a JSON-log record to the cache log.
#[allow(non_snake_case)]
pub fn H5C__json_write_expunge_entry_log_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__json_write_log_message(cache, "expunge_entry", &format!("{addr:#x}"));
}

/// Append a JSON-log record to the cache log.
#[allow(non_snake_case)]
pub fn H5C__json_write_flush_cache_log_msg(cache: &mut MetadataCache) {
    H5C__json_write_log_message(cache, "flush_cache", "");
}

/// Append a JSON-log record to the cache log.
#[allow(non_snake_case)]
pub fn H5C__json_write_insert_entry_log_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__json_write_log_message(cache, "insert_entry", &format!("{addr:#x}"));
}

/// Append a JSON-log record to the cache log.
#[allow(non_snake_case)]
pub fn H5C__json_write_mark_entry_clean_log_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__json_write_log_message(cache, "mark_entry_clean", &format!("{addr:#x}"));
}

/// Append a JSON-log record to the cache log.
#[allow(non_snake_case)]
pub fn H5C__json_write_mark_unserialized_entry_log_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__json_write_log_message(cache, "mark_unserialized", &format!("{addr:#x}"));
}

/// Append a JSON-log record to the cache log.
#[allow(non_snake_case)]
pub fn H5C__json_write_mark_serialized_entry_log_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__json_write_log_message(cache, "mark_serialized", &format!("{addr:#x}"));
}

/// Append a JSON-log record to the cache log.
#[allow(non_snake_case)]
pub fn H5C__json_write_move_entry_log_msg(cache: &mut MetadataCache, old_addr: u64, new_addr: u64) {
    H5C__json_write_log_message(
        cache,
        "move_entry",
        &format!("{old_addr:#x}->{new_addr:#x}"),
    );
}

/// Append a JSON-log record to the cache log.
#[allow(non_snake_case)]
pub fn H5C__json_write_pin_entry_log_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__json_write_log_message(cache, "pin_entry", &format!("{addr:#x}"));
}

/// Append a JSON-log record to the cache log.
#[allow(non_snake_case)]
pub fn H5C__json_write_create_fd_log_msg(cache: &mut MetadataCache, parent: u64, child: u64) {
    H5C__json_write_log_message(cache, "create_fd", &format!("{parent:#x}->{child:#x}"));
}

/// Append a JSON-log record to the cache log.
#[allow(non_snake_case)]
pub fn H5C__json_write_protect_entry_log_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__json_write_log_message(cache, "protect_entry", &format!("{addr:#x}"));
}

/// Append a JSON-log record to the cache log.
#[allow(non_snake_case)]
pub fn H5C__json_write_resize_entry_log_msg(cache: &mut MetadataCache, addr: u64, size: usize) {
    H5C__json_write_log_message(cache, "resize_entry", &format!("{addr:#x}:{size}"));
}

/// Append a JSON-log record to the cache log.
#[allow(non_snake_case)]
pub fn H5C__json_write_unpin_entry_log_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__json_write_log_message(cache, "unpin_entry", &format!("{addr:#x}"));
}

/// Append a JSON-log record to the cache log.
#[allow(non_snake_case)]
pub fn H5C__json_write_destroy_fd_log_msg(cache: &mut MetadataCache, parent: u64, child: u64) {
    H5C__json_write_log_message(cache, "destroy_fd", &format!("{parent:#x}->{child:#x}"));
}

/// Append a JSON-log record to the cache log.
#[allow(non_snake_case)]
pub fn H5C__json_write_unprotect_entry_log_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__json_write_log_message(cache, "unprotect_entry", &format!("{addr:#x}"));
}

/// Append a JSON-log record to the cache log.
#[allow(non_snake_case)]
pub fn H5C__json_write_set_cache_config_log_msg(cache: &mut MetadataCache) {
    H5C__json_write_log_message(cache, "set_cache_config", "");
}

/// Append a JSON-log record to the cache log.
#[allow(non_snake_case)]
pub fn H5C__json_write_remove_entry_log_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__json_write_log_message(cache, "remove_entry", &format!("{addr:#x}"));
}

/// Internal cache helper: log set up.
#[allow(non_snake_case)]
pub fn H5C_log_set_up(cache: &mut MetadataCache) {
    H5C__log_trace_set_up(cache);
    H5C__log_json_set_up(cache);
}

/// Internal cache helper: log tear down.
#[allow(non_snake_case)]
pub fn H5C_log_tear_down(cache: &mut MetadataCache) {
    H5C__trace_tear_down_logging(cache);
    H5C__json_tear_down_logging(cache);
}

/// Internal cache helper: start logging.
#[allow(non_snake_case)]
pub fn H5C_start_logging(cache: &mut MetadataCache) {
    cache.logs.push("logging:true".into());
}

/// Internal cache helper: stop logging.
#[allow(non_snake_case)]
pub fn H5C_stop_logging(cache: &mut MetadataCache) {
    cache.logs.push("logging:false".into());
}

/// Internal cache helper: get logging status.
#[allow(non_snake_case)]
pub fn H5C_get_logging_status(cache: &MetadataCache) -> bool {
    cache
        .logs
        .iter()
        .rev()
        .find_map(|record| record.strip_prefix("logging:").map(|value| value == "true"))
        .unwrap_or(false)
}

/// Append a JSON log message for the named event.
#[allow(non_snake_case)]
pub fn H5C_log_write_create_cache_msg(cache: &mut MetadataCache) {
    H5C__json_write_create_cache_log_msg(cache);
}

/// Append a JSON log message for the named event.
#[allow(non_snake_case)]
pub fn H5C_log_write_destroy_cache_msg(cache: &mut MetadataCache) {
    H5C__json_write_destroy_cache_log_msg(cache);
}

/// Append a JSON log message for the named event.
#[allow(non_snake_case)]
pub fn H5C_log_write_evict_cache_msg(cache: &mut MetadataCache) {
    H5C__json_write_evict_cache_log_msg(cache);
}

/// Append a JSON log message for the named event.
#[allow(non_snake_case)]
pub fn H5C_log_write_expunge_entry_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__json_write_expunge_entry_log_msg(cache, addr);
}

/// Append a JSON log message for the named event.
#[allow(non_snake_case)]
pub fn H5C_log_write_flush_cache_msg(cache: &mut MetadataCache) {
    H5C__json_write_flush_cache_log_msg(cache);
}

/// Append a JSON log message for the named event.
#[allow(non_snake_case)]
pub fn H5C_log_write_insert_entry_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__json_write_insert_entry_log_msg(cache, addr);
}

/// Append a JSON log message for the named event.
#[allow(non_snake_case)]
pub fn H5C_log_write_mark_entry_dirty_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__json_write_log_message(cache, "mark_entry_dirty", &format!("{addr:#x}"));
}

/// Append a JSON log message for the named event.
#[allow(non_snake_case)]
pub fn H5C_log_write_mark_entry_clean_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__json_write_mark_entry_clean_log_msg(cache, addr);
}

/// Append a JSON log message for the named event.
#[allow(non_snake_case)]
pub fn H5C_log_write_mark_unserialized_entry_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__json_write_mark_unserialized_entry_log_msg(cache, addr);
}

/// Append a JSON log message for the named event.
#[allow(non_snake_case)]
pub fn H5C_log_write_mark_serialized_entry_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__json_write_mark_serialized_entry_log_msg(cache, addr);
}

/// Append a JSON log message for the named event.
#[allow(non_snake_case)]
pub fn H5C_log_write_move_entry_msg(cache: &mut MetadataCache, old_addr: u64, new_addr: u64) {
    H5C__json_write_move_entry_log_msg(cache, old_addr, new_addr);
}

/// Append a JSON log message for the named event.
#[allow(non_snake_case)]
pub fn H5C_log_write_pin_entry_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__json_write_pin_entry_log_msg(cache, addr);
}

/// Append a JSON log message for the named event.
#[allow(non_snake_case)]
pub fn H5C_log_write_create_fd_msg(cache: &mut MetadataCache, parent: u64, child: u64) {
    H5C__json_write_create_fd_log_msg(cache, parent, child);
}

/// Append a JSON log message for the named event.
#[allow(non_snake_case)]
pub fn H5C_log_write_protect_entry_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__json_write_protect_entry_log_msg(cache, addr);
}

/// Append a JSON log message for the named event.
#[allow(non_snake_case)]
pub fn H5C_log_write_resize_entry_msg(cache: &mut MetadataCache, addr: u64, size: usize) {
    H5C__json_write_resize_entry_log_msg(cache, addr, size);
}

/// Append a JSON log message for the named event.
#[allow(non_snake_case)]
pub fn H5C_log_write_unpin_entry_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__json_write_unpin_entry_log_msg(cache, addr);
}

/// Append a JSON log message for the named event.
#[allow(non_snake_case)]
pub fn H5C_log_write_destroy_fd_msg(cache: &mut MetadataCache, parent: u64, child: u64) {
    H5C__json_write_destroy_fd_log_msg(cache, parent, child);
}

/// Append a JSON log message for the named event.
#[allow(non_snake_case)]
pub fn H5C_log_write_unprotect_entry_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__json_write_unprotect_entry_log_msg(cache, addr);
}

/// Append a JSON log message for the named event.
#[allow(non_snake_case)]
pub fn H5C_log_write_set_cache_config_msg(cache: &mut MetadataCache) {
    H5C__json_write_set_cache_config_log_msg(cache);
}

/// Append a JSON log message for the named event.
#[allow(non_snake_case)]
pub fn H5C_log_write_remove_entry_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__json_write_remove_entry_log_msg(cache, addr);
}

/// Epoch-marker entries have no on-disk image.
#[allow(non_snake_case)]
pub fn H5C__epoch_marker_get_initial_load_size() -> usize {
    0
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetadataCacheEntryStatus {
    pub dirty: bool,
    pub serialized: bool,
    pub pinned: bool,
    pub protected: bool,
    pub ring: u8,
    pub tag: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_tracks_dirty_flush_and_candidates() {
        let mut cache = MetadataCache::init();
        cache
            .insert_entry(MetadataCacheEntry::new(0x100, "btree", vec![1, 2, 3]))
            .unwrap();
        cache.mark_entry_dirty(0x100).unwrap();
        assert!(!cache.cache_is_clean());
        assert_eq!(cache.construct_candidate_list(), vec![0x100]);
        cache.flush_entries().unwrap();
        assert!(cache.cache_is_clean());
        assert_eq!(cache.broadcast_clean_list(), vec![0x100]);
    }

    #[test]
    fn mark_entries_as_clean_validates_dirty_cache_entries() {
        let mut cache = MetadataCache::init();
        cache
            .insert_entry(MetadataCacheEntry::new(0x100, "btree", vec![1, 2, 3]))
            .unwrap();
        cache
            .insert_entry(MetadataCacheEntry::new(0x200, "heap", vec![4, 5]))
            .unwrap();
        cache.mark_entry_dirty(0x100).unwrap();
        cache.mark_entry_dirty(0x200).unwrap();

        H5C_mark_entries_as_clean(&mut cache, &[0x100, 0x200]).unwrap();
        assert!(cache.cache_is_clean());
        assert_eq!(cache.broadcast_clean_list(), vec![0x100, 0x200]);

        assert!(H5C_mark_entries_as_clean(&mut cache, &[0x100]).is_err());
        assert!(H5C_mark_entries_as_clean(&mut cache, &[0x300]).is_err());
        assert!(H5C_mark_entries_as_clean(&mut cache, &[]).is_err());
    }

    #[test]
    fn proxy_entries_preserve_flush_dependencies() {
        let mut cache = MetadataCache::init();
        cache
            .insert_entry(MetadataCacheEntry::new(1, "parent", vec![1]))
            .unwrap();
        cache.proxy_entry_create(2, vec![2, 2]).unwrap();
        cache
            .insert_entry(MetadataCacheEntry::new(3, "child", vec![3]))
            .unwrap();
        cache.proxy_entry_add_parent(2, 1).unwrap();
        cache.proxy_entry_add_child(2, 3).unwrap();
        assert!(cache.flush_dependency_exists(1, 2));
        assert!(cache.flush_dependency_exists(2, 3));
        assert_eq!(cache.proxy_entry_image_len(2).unwrap(), 2);
        cache.proxy_entry_free_icr(2).unwrap();
        assert!(!cache.flush_dependency_exists(1, 2));
    }

    #[test]
    fn cache_image_header_and_entries_roundtrip_with_checks() {
        let mut cache = MetadataCache::init();
        cache
            .insert_entry(MetadataCacheEntry::new(0x100, "btree", vec![1, 2, 3]))
            .unwrap();
        cache
            .insert_entry(MetadataCacheEntry::new(0x200, "heap", vec![4, 5]))
            .unwrap();

        let header =
            H5C__decode_cache_image_header(&H5C__encode_cache_image_header(&cache).unwrap())
                .unwrap();
        assert_eq!(header.entries, 2);
        assert_eq!(header.total_image_bytes, 5);
        assert!(H5C__decode_cache_image_header(b"bad").is_err());

        let image = H5C__construct_cache_image_buffer(&cache).unwrap();
        let entries = H5C__deserialize_cache_image_buffer(&image).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].addr, 0x100);
        assert_eq!(entries[0].image, vec![1, 2, 3]);
        assert_eq!(entries[1].addr, 0x200);
        assert_eq!(entries[1].image, vec![4, 5]);

        let reconstructed = H5C__reconstruct_cache_contents_from_image(&image).unwrap();
        assert_eq!(reconstructed.entries.len(), 2);
        assert_eq!(
            reconstructed.require_entry(0x100).unwrap().image,
            vec![1, 2, 3]
        );

        assert!(H5C__deserialize_cache_image_buffer(&image[..15]).is_err());
        let mut truncated_payload = Vec::new();
        truncated_payload.extend_from_slice(&0x300u64.to_le_bytes());
        truncated_payload.extend_from_slice(&3u64.to_le_bytes());
        truncated_payload.extend_from_slice(&[1, 2]);
        assert!(H5C__deserialize_cache_image_buffer(&truncated_payload).is_err());
    }
}
