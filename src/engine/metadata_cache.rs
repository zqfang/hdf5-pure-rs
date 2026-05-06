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
    pub fn init() -> Self {
        Self::default()
    }

    pub fn term_package(&mut self) {
        self.entries.clear();
        self.candidate_list.clear();
        self.clean_list.clear();
        self.logs.clear();
        self.cache_image_pending = false;
        self.serialization_in_progress = false;
    }

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

    pub fn flush_dependency_exists(&self, parent: u64, child: u64) -> bool {
        self.entries
            .get(&parent)
            .is_some_and(|entry| entry.children.contains(&child))
            && self
                .entries
                .get(&child)
                .is_some_and(|entry| entry.parents.contains(&parent))
    }

    pub fn verify_entry_type(&self, addr: u64, entry_type: &str) -> bool {
        self.entries
            .get(&addr)
            .is_some_and(|entry| entry.entry_type == entry_type)
    }

    pub fn get_serialization_in_progress(&self) -> bool {
        self.serialization_in_progress
    }

    pub fn cache_is_clean(&self) -> bool {
        self.entries.values().all(|entry| !entry.dirty)
    }

    pub fn set_write_done_callback(&mut self, callback_name: impl Into<String>) {
        self.write_done_callback = Some(callback_name.into());
    }

    pub fn add_candidate(&mut self, addr: u64) -> Result<()> {
        self.require_entry(addr)?;
        self.candidate_list.insert(addr);
        Ok(())
    }

    pub fn broadcast_candidate_list(&self) -> Vec<u64> {
        self.broadcast_candidate_list_cb()
    }

    pub fn broadcast_candidate_list_cb(&self) -> Vec<u64> {
        self.candidate_list.iter().copied().collect()
    }

    pub fn broadcast_clean_list(&self) -> Vec<u64> {
        self.broadcast_clean_list_cb()
    }

    pub fn broadcast_clean_list_cb(&self) -> Vec<u64> {
        self.clean_list.iter().copied().collect()
    }

    pub fn construct_candidate_list(&mut self) -> Vec<u64> {
        self.candidate_list = self
            .entries
            .iter()
            .filter_map(|(&addr, entry)| entry.dirty.then_some(addr))
            .collect();
        self.broadcast_candidate_list()
    }

    pub fn copy_candidate_list_to_buffer(&self, out: &mut Vec<u8>) {
        self.copy_candidate_list_to_buffer_cb(out);
    }

    pub fn copy_candidate_list_to_buffer_cb(&self, out: &mut Vec<u8>) {
        out.clear();
        for addr in &self.candidate_list {
            out.extend_from_slice(&addr.to_le_bytes());
        }
    }

    pub fn log_deleted_entry(&mut self, addr: u64) {
        self.log_event("deleted", addr);
    }

    pub fn log_dirtied_entry(&mut self, addr: u64) {
        self.log_event("dirtied", addr);
    }

    pub fn log_cleaned_entry(&mut self, addr: u64) {
        self.log_event("cleaned", addr);
    }

    pub fn log_flushed_entry(&mut self, addr: u64) {
        self.log_event("flushed", addr);
    }

    pub fn log_inserted_entry(&mut self, addr: u64) {
        self.log_event("inserted", addr);
    }

    pub fn log_moved_entry(&mut self, addr: u64) {
        self.log_event("moved", addr);
    }

    pub fn propagate_and_apply_candidate_list(&mut self, addrs: &[u64]) -> Result<()> {
        self.receive_candidate_list(addrs);
        self.flush_entries()
    }

    pub fn propagate_flushed_and_still_clean_entries_list(&mut self) -> Vec<u64> {
        self.clean_list = self
            .entries
            .iter()
            .filter_map(|(&addr, entry)| (!entry.dirty).then_some(addr))
            .collect();
        self.broadcast_clean_list()
    }

    pub fn receive_haddr_list(&mut self, addrs: &[u64]) {
        self.clean_list = addrs.iter().copied().collect();
    }

    pub fn receive_and_apply_clean_list(&mut self, addrs: &[u64]) -> Result<()> {
        self.receive_haddr_list(addrs);
        for addr in addrs {
            self.mark_entry_clean(*addr)?;
        }
        Ok(())
    }

    pub fn receive_candidate_list(&mut self, addrs: &[u64]) {
        self.candidate_list = addrs.iter().copied().collect();
    }

    pub fn rsp_dist_md_write_flush(&mut self) -> Result<()> {
        self.flush_entries()
    }

    pub fn rsp_dist_md_write_flush_to_min_clean(&mut self) -> Result<()> {
        self.flush_entries()
    }

    pub fn rsp_p0_only_flush(&mut self) -> Result<()> {
        self.flush_entries()
    }

    pub fn rsp_p0_only_flush_to_min_clean(&mut self) -> Result<()> {
        self.flush_entries()
    }

    pub fn run_sync_point(&mut self) -> Result<()> {
        self.construct_candidate_list();
        self.flush_entries()
    }

    pub fn tidy_cache_0_lists(&mut self) {
        self.candidate_list.clear();
        self.clean_list.clear();
    }

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

    pub fn cache_image_pending(&self) -> bool {
        self.cache_image_pending
    }

    pub fn dest(mut self) {
        self.term_package();
    }

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

    pub fn expunge_entry(&mut self, addr: u64) -> Result<MetadataCacheEntry> {
        let entry = self.remove_entry(addr)?;
        self.log_deleted_entry(addr);
        Ok(entry)
    }

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

    pub fn load_cache_image_on_next_protect(&mut self) {
        self.cache_image_pending = true;
    }

    pub fn mark_entry_dirty(&mut self, addr: u64) -> Result<()> {
        self.require_entry_mut(addr)?.dirty = true;
        self.log_dirtied_entry(addr);
        Ok(())
    }

    pub fn mark_entry_clean(&mut self, addr: u64) -> Result<()> {
        self.require_entry_mut(addr)?.dirty = false;
        self.clean_list.insert(addr);
        self.log_cleaned_entry(addr);
        Ok(())
    }

    pub fn mark_entry_unserialized(&mut self, addr: u64) -> Result<()> {
        self.require_entry_mut(addr)?.serialized = false;
        Ok(())
    }

    pub fn mark_entry_serialized(&mut self, addr: u64) -> Result<()> {
        self.require_entry_mut(addr)?.serialized = true;
        Ok(())
    }

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

    pub fn prep_for_file_close(&mut self) -> Result<()> {
        self.flush_entries()?;
        self.evict()?;
        Ok(())
    }

    pub fn prep_for_file_flush(&mut self) -> Result<()> {
        self.construct_candidate_list();
        Ok(())
    }

    pub fn secure_from_file_flush(&mut self) {
        self.file_flush_secure = true;
    }

    pub fn create_flush_dependency(&mut self, parent: u64, child: u64) -> Result<()> {
        self.require_entry(parent)?;
        self.require_entry(child)?;
        self.require_entry_mut(parent)?.children.insert(child);
        self.require_entry_mut(child)?.parents.insert(parent);
        Ok(())
    }

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

    pub fn resize_entry(&mut self, addr: u64, new_size: usize) -> Result<()> {
        let entry = self.require_entry_mut(addr)?;
        entry.image.resize(new_size, 0);
        entry.dirty = true;
        Ok(())
    }

    pub fn unpin_entry(&mut self, addr: u64) -> Result<()> {
        self.require_entry_mut(addr)?.pinned = false;
        Ok(())
    }

    pub fn destroy_flush_dependency(&mut self, parent: u64, child: u64) -> Result<()> {
        self.require_entry(parent)?;
        self.require_entry(child)?;
        self.require_entry_mut(parent)?.children.remove(&child);
        self.require_entry_mut(child)?.parents.remove(&parent);
        Ok(())
    }

    pub fn unprotect(&mut self, addr: u64, dirtied: bool) -> Result<()> {
        let entry = self.require_entry_mut(addr)?;
        entry.protected = false;
        if dirtied {
            entry.dirty = true;
        }
        Ok(())
    }

    pub fn get_cache_auto_resize_config(&self) -> MetadataCacheResizeConfig {
        self.auto_resize_config.clone()
    }

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

    pub fn get_tag(&self) -> Option<u64> {
        self.current_tag
    }

    pub fn cork(&mut self, addr: u64, corked: bool) -> Result<()> {
        self.require_entry_mut(addr)?.corked = corked;
        Ok(())
    }

    pub fn verify_tag(&self, addr: u64, tag: u64) -> bool {
        self.entries
            .get(&addr)
            .is_some_and(|entry| entry.tag == Some(tag))
    }

    pub fn get_entry_ring(&self, addr: u64) -> Result<u8> {
        Ok(self.require_entry(addr)?.ring)
    }

    pub fn set_ring(&mut self, ring: u8) {
        self.current_ring = ring;
    }

    pub fn unsettle_entry_ring(&mut self, addr: u64) -> Result<()> {
        let ring = self.current_ring;
        self.require_entry_mut(addr)?.ring = ring;
        Ok(())
    }

    pub fn unsettle_ring(&mut self) {
        for entry in self.entries.values_mut() {
            entry.ring = self.current_ring;
        }
    }

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

    pub fn proxy_entry_create(&mut self, addr: u64, image: Vec<u8>) -> Result<()> {
        self.insert_entry(MetadataCacheEntry::new(addr, "proxy", image))
    }

    pub fn proxy_entry_add_parent(&mut self, proxy_addr: u64, parent: u64) -> Result<()> {
        self.create_flush_dependency(parent, proxy_addr)
    }

    pub fn proxy_entry_remove_parent(&mut self, proxy_addr: u64, parent: u64) -> Result<()> {
        self.destroy_flush_dependency(parent, proxy_addr)
    }

    pub fn proxy_entry_add_child_cb(&mut self, proxy_addr: u64, child: u64) -> Result<()> {
        self.create_flush_dependency(proxy_addr, child)
    }

    pub fn proxy_entry_add_child(&mut self, proxy_addr: u64, child: u64) -> Result<()> {
        self.proxy_entry_add_child_cb(proxy_addr, child)
    }

    pub fn proxy_entry_remove_child_cb(&mut self, proxy_addr: u64, child: u64) -> Result<()> {
        self.destroy_flush_dependency(proxy_addr, child)
    }

    pub fn proxy_entry_remove_child(&mut self, proxy_addr: u64, child: u64) -> Result<()> {
        self.proxy_entry_remove_child_cb(proxy_addr, child)
    }

    pub fn proxy_entry_dest(&mut self, proxy_addr: u64) -> Result<MetadataCacheEntry> {
        self.remove_entry(proxy_addr)
    }

    pub fn proxy_entry_image_len(&self, proxy_addr: u64) -> Result<usize> {
        Ok(self.require_entry(proxy_addr)?.image.len())
    }

    pub fn proxy_entry_serialize(&mut self, proxy_addr: u64) -> Result<Vec<u8>> {
        let entry = self.require_entry_mut(proxy_addr)?;
        entry.serialized = true;
        Ok(entry.image.clone())
    }

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

    pub fn proxy_entry_free_icr(&mut self, proxy_addr: u64) -> Result<()> {
        self.remove_entry(proxy_addr).map(|_| ())
    }

    pub fn set_tag_for_entry(&mut self, addr: u64, tag: u64) -> Result<()> {
        self.require_entry_mut(addr)?.tag = Some(tag);
        self.current_tag = Some(tag);
        Ok(())
    }

    fn require_entry(&self, addr: u64) -> Result<&MetadataCacheEntry> {
        self.entries.get(&addr).ok_or_else(|| {
            Error::InvalidFormat(format!("metadata cache entry {addr:#x} not found"))
        })
    }

    fn require_entry_mut(&mut self, addr: u64) -> Result<&mut MetadataCacheEntry> {
        self.entries.get_mut(&addr).ok_or_else(|| {
            Error::InvalidFormat(format!("metadata cache entry {addr:#x} not found"))
        })
    }

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

    fn log_event(&mut self, kind: &str, addr: u64) {
        self.logs.push(format!("{kind}:{addr:#x}"));
    }
}

#[allow(non_snake_case)]
pub fn H5C_create() -> MetadataCache {
    MetadataCache::init()
}

#[allow(non_snake_case)]
pub fn H5C_dest(cache: MetadataCache) {
    cache.dest();
}

#[allow(non_snake_case)]
pub fn H5C_get_cache_auto_resize_config(cache: &MetadataCache) -> MetadataCacheResizeConfig {
    cache.get_cache_auto_resize_config()
}

#[allow(non_snake_case)]
pub fn H5C_set_cache_auto_resize_config(
    cache: &mut MetadataCache,
    config: MetadataCacheResizeConfig,
) {
    cache.auto_resize_config = config;
}

#[allow(non_snake_case)]
pub fn H5C_get_mdc_image_info(cache: &MetadataCache) -> (bool, usize) {
    (cache.cache_image_pending(), cache.stats().total_image_bytes)
}

#[allow(non_snake_case)]
pub fn H5C_apply_candidate_list(cache: &mut MetadataCache, addrs: &[u64]) -> Result<()> {
    cache.propagate_and_apply_candidate_list(addrs)
}

#[allow(non_snake_case)]
pub fn H5C_construct_candidate_list__clean_cache(cache: &mut MetadataCache) -> Vec<u64> {
    cache.propagate_flushed_and_still_clean_entries_list()
}

#[allow(non_snake_case)]
pub fn H5C_construct_candidate_list__min_clean(cache: &mut MetadataCache) -> Vec<u64> {
    cache.construct_candidate_list()
}

#[allow(non_snake_case)]
pub fn H5C_clear_coll_entries(cache: &mut MetadataCache) {
    cache.tidy_cache_0_lists();
}

#[allow(non_snake_case)]
pub fn H5C__collective_write(cache: &mut MetadataCache) -> Result<()> {
    cache.flush_entries()
}

#[allow(non_snake_case)]
pub fn H5C__flush_candidate_entries(cache: &mut MetadataCache) -> Result<()> {
    cache.flush_entries()
}

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

#[allow(non_snake_case)]
pub fn H5C__auto_adjust_cache_size(cache: &mut MetadataCache) -> MetadataCacheStats {
    cache.stats()
}

#[allow(non_snake_case)]
pub fn H5C__autoadjust__ageout(cache: &mut MetadataCache) -> Result<usize> {
    cache.evict()
}

#[allow(non_snake_case)]
pub fn H5C__autoadjust__ageout__cycle_epoch_marker(cache: &mut MetadataCache) {
    cache.logs.push("epoch:cycle".into());
}

#[allow(non_snake_case)]
pub fn H5C__autoadjust__ageout__evict_aged_out_entries(cache: &mut MetadataCache) -> Result<usize> {
    cache.evict()
}

#[allow(non_snake_case)]
pub fn H5C__autoadjust__ageout__insert_new_marker(cache: &mut MetadataCache) {
    cache.logs.push("epoch:insert".into());
}

#[allow(non_snake_case)]
pub fn H5C__autoadjust__ageout__remove_all_markers(cache: &mut MetadataCache) {
    cache.logs.retain(|record| !record.starts_with("epoch:"));
}

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

#[allow(non_snake_case)]
pub fn H5C__flush_invalidate_cache(cache: &mut MetadataCache) -> Result<()> {
    cache.flush_entries()?;
    cache.evict()?;
    Ok(())
}

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

#[allow(non_snake_case)]
pub fn H5C__make_space_in_cache(cache: &mut MetadataCache, needed: usize) -> Result<bool> {
    if cache.stats().total_image_bytes <= cache.auto_resize_config.max_size.saturating_sub(needed) {
        return Ok(true);
    }
    cache.evict()?;
    Ok(cache.stats().total_image_bytes <= cache.auto_resize_config.max_size.saturating_sub(needed))
}

#[allow(non_snake_case)]
pub fn H5C__serialize_cache(cache: &mut MetadataCache) -> Result<Vec<u8>> {
    cache.flush_entries()?;
    H5C__construct_cache_image_buffer(cache)
}

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

#[allow(non_snake_case)]
pub fn H5C__prefetched_entry_free_icr(entry: MetadataCacheEntry) {
    drop(entry);
}

#[allow(non_snake_case)]
pub fn H5C_cache_image_pending(cache: &MetadataCache) -> bool {
    cache.cache_image_pending()
}

#[allow(non_snake_case)]
pub fn H5C_cache_image_status(cache: &MetadataCache) -> (bool, usize) {
    H5C_get_mdc_image_info(cache)
}

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

#[allow(non_snake_case)]
pub fn H5C__generate_cache_image(cache: &mut MetadataCache) -> Result<Vec<u8>> {
    H5C__serialize_cache(cache)
}

#[allow(non_snake_case)]
pub fn H5C__free_image_entries_array(entries: Vec<MetadataCacheEntry>) {
    drop(entries);
}

#[allow(non_snake_case)]
pub fn H5C_load_cache_image_on_next_protect(cache: &mut MetadataCache) {
    cache.load_cache_image_on_next_protect();
}

#[allow(non_snake_case)]
pub fn H5C__image_entry_cmp(
    left: &MetadataCacheEntry,
    right: &MetadataCacheEntry,
) -> std::cmp::Ordering {
    left.addr.cmp(&right.addr)
}

#[allow(non_snake_case)]
pub fn H5C__prep_image_for_file_close(cache: &mut MetadataCache) -> Result<Vec<u8>> {
    H5C__serialize_cache(cache)
}

#[allow(non_snake_case)]
pub fn H5C_validate_cache_image_config(config: &MetadataCacheResizeConfig) -> bool {
    config.min_size <= config.max_size && config.min_clean_fraction <= 100
}

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

#[allow(non_snake_case)]
pub fn H5C__prep_for_file_close__compute_fd_heights(cache: &MetadataCache) -> BTreeMap<u64, usize> {
    cache
        .entries
        .iter()
        .map(|(&addr, entry)| (addr, entry.parents.len()))
        .collect()
}

#[allow(non_snake_case)]
pub fn H5C__prep_for_file_close__compute_fd_heights_real(
    cache: &MetadataCache,
) -> BTreeMap<u64, usize> {
    H5C__prep_for_file_close__compute_fd_heights(cache)
}

#[allow(non_snake_case)]
pub fn H5C__prep_for_file_close__scan_entries(cache: &MetadataCache) -> Vec<u64> {
    cache.entries.keys().copied().collect()
}

#[allow(non_snake_case)]
pub fn H5C__check_for_duplicates(cache: &MetadataCache) -> bool {
    cache.entries.len() == cache.entries.keys().collect::<BTreeSet<_>>().len()
}

#[allow(non_snake_case)]
pub fn H5C__reconstruct_cache_contents(entries: Vec<MetadataCacheEntry>) -> Result<MetadataCache> {
    let mut cache = MetadataCache::init();
    for entry in entries {
        cache.insert_entry(entry)?;
    }
    Ok(cache)
}

#[allow(non_snake_case)]
pub fn H5C__reconstruct_cache_entry(
    addr: u64,
    entry_type: impl Into<String>,
    image: Vec<u8>,
) -> MetadataCacheEntry {
    MetadataCacheEntry::new(addr, entry_type, image)
}

#[allow(non_snake_case)]
pub fn H5C__write_cache_image_superblock_msg(cache: &MetadataCache) -> Result<Vec<u8>> {
    H5C__encode_cache_image_header(cache)
}

#[allow(non_snake_case)]
pub fn H5C__write_cache_image(cache: &mut MetadataCache) -> Result<Vec<u8>> {
    H5C__serialize_cache(cache)
}

#[allow(non_snake_case)]
pub fn H5C_prep_for_file_close(cache: &mut MetadataCache) -> Result<()> {
    cache.prep_for_file_close()
}

#[allow(non_snake_case)]
pub fn H5C_evict(cache: &mut MetadataCache) -> Result<usize> {
    cache.evict()
}

#[allow(non_snake_case)]
pub fn H5C_flush_cache(cache: &mut MetadataCache) -> Result<()> {
    cache.flush_entries()
}

#[allow(non_snake_case)]
pub fn H5C_flush_to_min_clean(cache: &mut MetadataCache) -> Result<()> {
    cache.flush_entries()
}

#[allow(non_snake_case)]
pub fn H5C_reset_cache_hit_rate_stats(cache: &mut MetadataCache) {
    cache.logs.retain(|record| !record.starts_with("hit-rate:"));
}

#[allow(non_snake_case)]
pub fn H5C_set_evictions_enabled(cache: &mut MetadataCache, enabled: bool) {
    cache.logs.push(format!("evictions_enabled:{enabled}"));
}

#[allow(non_snake_case)]
pub fn H5C_unsettle_ring(cache: &mut MetadataCache) {
    cache.unsettle_ring();
}

#[allow(non_snake_case)]
pub fn H5C__unpin_entry_real(cache: &mut MetadataCache, addr: u64) -> Result<()> {
    cache.unpin_entry(addr)
}

#[allow(non_snake_case)]
pub fn H5C__unpin_entry_from_client(cache: &mut MetadataCache, addr: u64) -> Result<()> {
    cache.unpin_entry(addr)
}

#[allow(non_snake_case)]
pub fn H5C__generate_image(cache: &mut MetadataCache, addr: u64) -> Result<Vec<u8>> {
    Ok(cache.require_entry(addr)?.image.clone())
}

#[allow(non_snake_case)]
pub fn H5C__flush_single_entry(cache: &mut MetadataCache, addr: u64) -> Result<()> {
    cache.flush(addr)
}

#[allow(non_snake_case)]
pub fn H5C__discard_single_entry(
    cache: &mut MetadataCache,
    addr: u64,
) -> Result<MetadataCacheEntry> {
    cache.remove_entry(addr)
}

#[allow(non_snake_case)]
pub fn H5C__verify_len_eoa(cache: &MetadataCache, addr: u64, len: usize, eoa: u64) -> bool {
    let Ok(len_u64) = u64::try_from(len) else {
        return false;
    };
    cache.entries.get(&addr).is_some_and(|entry| {
        entry.image.len() == len && addr.checked_add(len_u64).is_some_and(|end| end <= eoa)
    })
}

fn usize_to_u64(value: usize, context: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| Error::InvalidFormat(format!("{context} exceeds u64")))
}

#[allow(non_snake_case)]
pub fn H5C__load_entry(cache: &MetadataCache, addr: u64) -> Result<Vec<u8>> {
    Ok(cache.require_entry(addr)?.image.clone())
}

#[allow(non_snake_case)]
pub fn H5C__mark_flush_dep_dirty(cache: &mut MetadataCache, addr: u64) -> Result<()> {
    cache.mark_entry_dirty(addr)
}

#[allow(non_snake_case)]
pub fn H5C__mark_flush_dep_clean(cache: &mut MetadataCache, addr: u64) -> Result<()> {
    cache.mark_entry_clean(addr)
}

#[allow(non_snake_case)]
pub fn H5C__mark_flush_dep_serialized(cache: &mut MetadataCache, addr: u64) -> Result<()> {
    cache.mark_entry_serialized(addr)
}

#[allow(non_snake_case)]
pub fn H5C__mark_flush_dep_unserialized(cache: &mut MetadataCache, addr: u64) -> Result<()> {
    cache.mark_entry_unserialized(addr)
}

#[allow(non_snake_case)]
pub fn H5C__assert_flush_dep_nocycle(cache: &MetadataCache, parent: u64, child: u64) -> bool {
    parent != child && !cache.flush_dependency_exists(child, parent)
}

#[allow(non_snake_case)]
pub fn H5C__serialize_single_entry(cache: &mut MetadataCache, addr: u64) -> Result<Vec<u8>> {
    let entry = cache.require_entry_mut(addr)?;
    entry.serialized = true;
    Ok(entry.image.clone())
}

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

#[allow(non_snake_case)]
pub fn H5C__deserialize_prefetched_entry(addr: u64, image: Vec<u8>) -> MetadataCacheEntry {
    MetadataCacheEntry::new(addr, "prefetched", image)
}

#[allow(non_snake_case)]
pub fn H5C_insert_entry(cache: &mut MetadataCache, entry: MetadataCacheEntry) -> Result<()> {
    cache.insert_entry(entry)
}

#[allow(non_snake_case)]
pub fn H5C_mark_entry_dirty(cache: &mut MetadataCache, addr: u64) -> Result<()> {
    cache.mark_entry_dirty(addr)
}

#[allow(non_snake_case)]
pub fn H5C_mark_entry_clean(cache: &mut MetadataCache, addr: u64) -> Result<()> {
    cache.mark_entry_clean(addr)
}

#[allow(non_snake_case)]
pub fn H5C_mark_entry_unserialized(cache: &mut MetadataCache, addr: u64) -> Result<()> {
    cache.mark_entry_unserialized(addr)
}

#[allow(non_snake_case)]
pub fn H5C_mark_entry_serialized(cache: &mut MetadataCache, addr: u64) -> Result<()> {
    cache.mark_entry_serialized(addr)
}

#[allow(non_snake_case)]
pub fn H5C_move_entry(cache: &mut MetadataCache, old_addr: u64, new_addr: u64) -> Result<()> {
    cache.move_entry(old_addr, new_addr)
}

#[allow(non_snake_case)]
pub fn H5C_resize_entry(cache: &mut MetadataCache, addr: u64, new_size: usize) -> Result<()> {
    cache.resize_entry(addr, new_size)
}

#[allow(non_snake_case)]
pub fn H5C_pin_protected_entry(cache: &mut MetadataCache, addr: u64) -> Result<()> {
    cache.pin_protected_entry(addr)
}

#[allow(non_snake_case)]
pub fn H5C_protect(cache: &mut MetadataCache, addr: u64) -> Result<&[u8]> {
    cache.protect(addr)
}

#[allow(non_snake_case)]
pub fn H5C_unpin_entry(cache: &mut MetadataCache, addr: u64) -> Result<()> {
    cache.unpin_entry(addr)
}

#[allow(non_snake_case)]
pub fn H5C_unprotect(cache: &mut MetadataCache, addr: u64, dirtied: bool) -> Result<()> {
    cache.unprotect(addr, dirtied)
}

#[allow(non_snake_case)]
pub fn H5C_unsettle_entry_ring(cache: &mut MetadataCache, addr: u64) -> Result<()> {
    cache.unsettle_entry_ring(addr)
}

#[allow(non_snake_case)]
pub fn H5C_create_flush_dependency(
    cache: &mut MetadataCache,
    parent: u64,
    child: u64,
) -> Result<()> {
    cache.create_flush_dependency(parent, child)
}

#[allow(non_snake_case)]
pub fn H5C_destroy_flush_dependency(
    cache: &mut MetadataCache,
    parent: u64,
    child: u64,
) -> Result<()> {
    cache.destroy_flush_dependency(parent, child)
}

#[allow(non_snake_case)]
pub fn H5C_expunge_entry(cache: &mut MetadataCache, addr: u64) -> Result<MetadataCacheEntry> {
    cache.expunge_entry(addr)
}

#[allow(non_snake_case)]
pub fn H5C_remove_entry(cache: &mut MetadataCache, addr: u64) -> Result<MetadataCacheEntry> {
    cache.remove_entry(addr)
}

#[allow(non_snake_case)]
pub fn H5C__verify_cork_tag_test(cache: &MetadataCache, addr: u64, tag: u64) -> bool {
    cache.verify_tag(addr, tag)
}

#[allow(non_snake_case)]
pub fn H5C_ignore_tags(cache: &mut MetadataCache, ignore: bool) {
    cache.logs.push(format!("ignore_tags:{ignore}"));
}

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

#[allow(non_snake_case)]
pub fn H5C_get_num_objs_corked(cache: &MetadataCache) -> usize {
    cache.entries.values().filter(|entry| entry.corked).count()
}

#[allow(non_snake_case)]
pub fn H5C__tag_entry(cache: &mut MetadataCache, addr: u64, tag: u64) -> Result<()> {
    cache.set_tag_for_entry(addr, tag)
}

#[allow(non_snake_case)]
pub fn H5C__untag_entry(cache: &mut MetadataCache, addr: u64) -> Result<()> {
    cache.require_entry_mut(addr)?.tag = None;
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5C__iter_tagged_entries_real(cache: &MetadataCache, tag: u64) -> Vec<u64> {
    cache
        .entries
        .iter()
        .filter_map(|(&addr, entry)| (entry.tag == Some(tag)).then_some(addr))
        .collect()
}

#[allow(non_snake_case)]
pub fn H5C__iter_tagged_entries(cache: &MetadataCache, tag: u64) -> Vec<u64> {
    H5C__iter_tagged_entries_real(cache, tag)
}

#[allow(non_snake_case)]
pub fn H5C__evict_tagged_entries_cb(cache: &mut MetadataCache, tag: u64) -> Result<usize> {
    cache.evict_tagged_metadata(tag)
}

#[allow(non_snake_case)]
pub fn H5C_evict_tagged_entries(cache: &mut MetadataCache, tag: u64) -> Result<usize> {
    cache.evict_tagged_metadata(tag)
}

#[allow(non_snake_case)]
pub fn H5C_verify_tag(cache: &MetadataCache, addr: u64, tag: u64) -> bool {
    cache.verify_tag(addr, tag)
}

#[allow(non_snake_case)]
pub fn H5C__flush_tagged_entries_cb(cache: &mut MetadataCache, tag: u64) -> Result<()> {
    let addrs = H5C__iter_tagged_entries(cache, tag);
    for addr in addrs {
        cache.flush(addr)?;
    }
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5C_flush_tagged_entries(cache: &mut MetadataCache, tag: u64) -> Result<()> {
    H5C__flush_tagged_entries_cb(cache, tag)
}

#[allow(non_snake_case)]
pub fn H5C_retag_entries(cache: &mut MetadataCache, old_tag: u64, new_tag: u64) -> Result<usize> {
    let addrs = H5C__iter_tagged_entries(cache, old_tag);
    for addr in &addrs {
        cache.set_tag_for_entry(*addr, new_tag)?;
    }
    Ok(addrs.len())
}

#[allow(non_snake_case)]
pub fn H5C__expunge_tag_type_metadata_cb(
    cache: &mut MetadataCache,
    tag: u64,
    entry_type: &str,
) -> Result<usize> {
    cache.expunge_tag_type_metadata(tag, entry_type)
}

#[allow(non_snake_case)]
pub fn H5C_expunge_tag_type_metadata(
    cache: &mut MetadataCache,
    tag: u64,
    entry_type: &str,
) -> Result<usize> {
    cache.expunge_tag_type_metadata(tag, entry_type)
}

#[allow(non_snake_case)]
pub fn H5C_get_tag(cache: &MetadataCache) -> Option<u64> {
    cache.get_tag()
}

#[allow(non_snake_case)]
pub fn H5C_dump_cache(cache: &MetadataCache) -> String {
    cache.dump_cache()
}

#[allow(non_snake_case)]
pub fn H5C_dump_cache_LRU(cache: &MetadataCache) -> Vec<u64> {
    cache.entries.keys().copied().collect()
}

#[allow(non_snake_case)]
pub fn H5C_dump_cache_skip_list(cache: &MetadataCache) -> Vec<u64> {
    cache.entries.keys().rev().copied().collect()
}

#[allow(non_snake_case)]
pub fn H5C_set_prefix(cache: &mut MetadataCache, prefix: impl Into<String>) {
    cache.logs.push(format!("prefix:{}", prefix.into()));
}

#[allow(non_snake_case)]
pub fn H5C_stats(cache: &MetadataCache) -> MetadataCacheStats {
    cache.stats()
}

#[allow(non_snake_case)]
pub fn H5C_flush_dependency_exists(cache: &MetadataCache, parent: u64, child: u64) -> bool {
    cache.flush_dependency_exists(parent, child)
}

#[allow(non_snake_case)]
pub fn H5C_get_entry_ptr_from_addr(
    cache: &MetadataCache,
    addr: u64,
) -> Result<&MetadataCacheEntry> {
    cache.require_entry(addr)
}

#[allow(non_snake_case)]
pub fn H5C_get_serialization_in_progress(cache: &MetadataCache) -> bool {
    cache.get_serialization_in_progress()
}

#[allow(non_snake_case)]
pub fn H5C_cache_is_clean(cache: &MetadataCache) -> bool {
    cache.cache_is_clean()
}

#[allow(non_snake_case)]
pub fn H5C_verify_entry_type(cache: &MetadataCache, addr: u64, entry_type: &str) -> bool {
    cache.verify_entry_type(addr, entry_type)
}

#[allow(non_snake_case)]
pub fn H5C_def_auto_resize_rpt_fcn(cache: &MetadataCache) -> String {
    format!("{:?}", cache.get_cache_auto_resize_config())
}

#[allow(non_snake_case)]
pub fn H5C__validate_lru_list(cache: &MetadataCache) -> bool {
    H5C__check_for_duplicates(cache)
}

#[allow(non_snake_case)]
pub fn H5C__validate_pinned_entry_list(cache: &MetadataCache) -> bool {
    cache
        .entries
        .values()
        .all(|entry| !entry.pinned || entry.protected)
}

#[allow(non_snake_case)]
pub fn H5C__validate_protected_entry_list(cache: &MetadataCache) -> bool {
    cache
        .entries
        .values()
        .all(|entry| !entry.protected || cache.entries.contains_key(&entry.addr))
}

#[allow(non_snake_case)]
pub fn H5C__entry_in_skip_list(cache: &MetadataCache, addr: u64) -> bool {
    cache.entries.contains_key(&addr)
}

#[allow(non_snake_case)]
pub fn H5C__trace_write_log_message(cache: &mut MetadataCache, message: impl Into<String>) {
    cache.logs.push(format!("trace:{}", message.into()));
}

#[allow(non_snake_case)]
pub fn H5C__log_trace_set_up(cache: &mut MetadataCache) {
    H5C__trace_write_log_message(cache, "setup");
}

#[allow(non_snake_case)]
pub fn H5C__trace_tear_down_logging(cache: &mut MetadataCache) {
    H5C__trace_write_log_message(cache, "teardown");
}

#[allow(non_snake_case)]
pub fn H5C__trace_write_expunge_entry_log_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__trace_write_log_message(cache, format!("expunge:{addr:#x}"));
}

#[allow(non_snake_case)]
pub fn H5C__trace_write_flush_cache_log_msg(cache: &mut MetadataCache) {
    H5C__trace_write_log_message(cache, "flush_cache");
}

#[allow(non_snake_case)]
pub fn H5C__trace_write_insert_entry_log_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__trace_write_log_message(cache, format!("insert:{addr:#x}"));
}

#[allow(non_snake_case)]
pub fn H5C__trace_write_mark_unserialized_entry_log_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__trace_write_log_message(cache, format!("mark_unserialized:{addr:#x}"));
}

#[allow(non_snake_case)]
pub fn H5C__trace_write_mark_serialized_entry_log_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__trace_write_log_message(cache, format!("mark_serialized:{addr:#x}"));
}

#[allow(non_snake_case)]
pub fn H5C__trace_write_move_entry_log_msg(
    cache: &mut MetadataCache,
    old_addr: u64,
    new_addr: u64,
) {
    H5C__trace_write_log_message(cache, format!("move:{old_addr:#x}->{new_addr:#x}"));
}

#[allow(non_snake_case)]
pub fn H5C__trace_write_pin_entry_log_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__trace_write_log_message(cache, format!("pin:{addr:#x}"));
}

#[allow(non_snake_case)]
pub fn H5C__trace_write_create_fd_log_msg(cache: &mut MetadataCache, parent: u64, child: u64) {
    H5C__trace_write_log_message(cache, format!("create_fd:{parent:#x}->{child:#x}"));
}

#[allow(non_snake_case)]
pub fn H5C__trace_write_protect_entry_log_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__trace_write_log_message(cache, format!("protect:{addr:#x}"));
}

#[allow(non_snake_case)]
pub fn H5C__trace_write_resize_entry_log_msg(cache: &mut MetadataCache, addr: u64, size: usize) {
    H5C__trace_write_log_message(cache, format!("resize:{addr:#x}:{size}"));
}

#[allow(non_snake_case)]
pub fn H5C__trace_write_unpin_entry_log_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__trace_write_log_message(cache, format!("unpin:{addr:#x}"));
}

#[allow(non_snake_case)]
pub fn H5C__trace_write_destroy_fd_log_msg(cache: &mut MetadataCache, parent: u64, child: u64) {
    H5C__trace_write_log_message(cache, format!("destroy_fd:{parent:#x}->{child:#x}"));
}

#[allow(non_snake_case)]
pub fn H5C__trace_write_unprotect_entry_log_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__trace_write_log_message(cache, format!("unprotect:{addr:#x}"));
}

#[allow(non_snake_case)]
pub fn H5C__trace_write_remove_entry_log_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__trace_write_log_message(cache, format!("remove:{addr:#x}"));
}

#[allow(non_snake_case)]
pub fn H5C__json_write_log_message(cache: &mut MetadataCache, event: &str, payload: &str) {
    cache.logs.push(format!(
        "json:{{\"event\":\"{event}\",\"payload\":\"{payload}\"}}"
    ));
}

#[allow(non_snake_case)]
pub fn H5C__log_json_set_up(cache: &mut MetadataCache) {
    H5C__json_write_log_message(cache, "setup", "");
}

#[allow(non_snake_case)]
pub fn H5C__json_tear_down_logging(cache: &mut MetadataCache) {
    H5C__json_write_log_message(cache, "teardown", "");
}

#[allow(non_snake_case)]
pub fn H5C__json_write_start_log_msg(cache: &mut MetadataCache) {
    H5C__json_write_log_message(cache, "start", "");
}

#[allow(non_snake_case)]
pub fn H5C__json_write_stop_log_msg(cache: &mut MetadataCache) {
    H5C__json_write_log_message(cache, "stop", "");
}

#[allow(non_snake_case)]
pub fn H5C__json_write_create_cache_log_msg(cache: &mut MetadataCache) {
    H5C__json_write_log_message(cache, "create_cache", "");
}

#[allow(non_snake_case)]
pub fn H5C__json_write_destroy_cache_log_msg(cache: &mut MetadataCache) {
    H5C__json_write_log_message(cache, "destroy_cache", "");
}

#[allow(non_snake_case)]
pub fn H5C__json_write_evict_cache_log_msg(cache: &mut MetadataCache) {
    H5C__json_write_log_message(cache, "evict_cache", "");
}

#[allow(non_snake_case)]
pub fn H5C__json_write_expunge_entry_log_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__json_write_log_message(cache, "expunge_entry", &format!("{addr:#x}"));
}

#[allow(non_snake_case)]
pub fn H5C__json_write_flush_cache_log_msg(cache: &mut MetadataCache) {
    H5C__json_write_log_message(cache, "flush_cache", "");
}

#[allow(non_snake_case)]
pub fn H5C__json_write_insert_entry_log_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__json_write_log_message(cache, "insert_entry", &format!("{addr:#x}"));
}

#[allow(non_snake_case)]
pub fn H5C__json_write_mark_entry_clean_log_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__json_write_log_message(cache, "mark_entry_clean", &format!("{addr:#x}"));
}

#[allow(non_snake_case)]
pub fn H5C__json_write_mark_unserialized_entry_log_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__json_write_log_message(cache, "mark_unserialized", &format!("{addr:#x}"));
}

#[allow(non_snake_case)]
pub fn H5C__json_write_mark_serialized_entry_log_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__json_write_log_message(cache, "mark_serialized", &format!("{addr:#x}"));
}

#[allow(non_snake_case)]
pub fn H5C__json_write_move_entry_log_msg(cache: &mut MetadataCache, old_addr: u64, new_addr: u64) {
    H5C__json_write_log_message(
        cache,
        "move_entry",
        &format!("{old_addr:#x}->{new_addr:#x}"),
    );
}

#[allow(non_snake_case)]
pub fn H5C__json_write_pin_entry_log_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__json_write_log_message(cache, "pin_entry", &format!("{addr:#x}"));
}

#[allow(non_snake_case)]
pub fn H5C__json_write_create_fd_log_msg(cache: &mut MetadataCache, parent: u64, child: u64) {
    H5C__json_write_log_message(cache, "create_fd", &format!("{parent:#x}->{child:#x}"));
}

#[allow(non_snake_case)]
pub fn H5C__json_write_protect_entry_log_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__json_write_log_message(cache, "protect_entry", &format!("{addr:#x}"));
}

#[allow(non_snake_case)]
pub fn H5C__json_write_resize_entry_log_msg(cache: &mut MetadataCache, addr: u64, size: usize) {
    H5C__json_write_log_message(cache, "resize_entry", &format!("{addr:#x}:{size}"));
}

#[allow(non_snake_case)]
pub fn H5C__json_write_unpin_entry_log_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__json_write_log_message(cache, "unpin_entry", &format!("{addr:#x}"));
}

#[allow(non_snake_case)]
pub fn H5C__json_write_destroy_fd_log_msg(cache: &mut MetadataCache, parent: u64, child: u64) {
    H5C__json_write_log_message(cache, "destroy_fd", &format!("{parent:#x}->{child:#x}"));
}

#[allow(non_snake_case)]
pub fn H5C__json_write_unprotect_entry_log_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__json_write_log_message(cache, "unprotect_entry", &format!("{addr:#x}"));
}

#[allow(non_snake_case)]
pub fn H5C__json_write_set_cache_config_log_msg(cache: &mut MetadataCache) {
    H5C__json_write_log_message(cache, "set_cache_config", "");
}

#[allow(non_snake_case)]
pub fn H5C__json_write_remove_entry_log_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__json_write_log_message(cache, "remove_entry", &format!("{addr:#x}"));
}

#[allow(non_snake_case)]
pub fn H5C_log_set_up(cache: &mut MetadataCache) {
    H5C__log_trace_set_up(cache);
    H5C__log_json_set_up(cache);
}

#[allow(non_snake_case)]
pub fn H5C_log_tear_down(cache: &mut MetadataCache) {
    H5C__trace_tear_down_logging(cache);
    H5C__json_tear_down_logging(cache);
}

#[allow(non_snake_case)]
pub fn H5C_start_logging(cache: &mut MetadataCache) {
    cache.logs.push("logging:true".into());
}

#[allow(non_snake_case)]
pub fn H5C_stop_logging(cache: &mut MetadataCache) {
    cache.logs.push("logging:false".into());
}

#[allow(non_snake_case)]
pub fn H5C_get_logging_status(cache: &MetadataCache) -> bool {
    cache
        .logs
        .iter()
        .rev()
        .find_map(|record| record.strip_prefix("logging:").map(|value| value == "true"))
        .unwrap_or(false)
}

#[allow(non_snake_case)]
pub fn H5C_log_write_create_cache_msg(cache: &mut MetadataCache) {
    H5C__json_write_create_cache_log_msg(cache);
}

#[allow(non_snake_case)]
pub fn H5C_log_write_destroy_cache_msg(cache: &mut MetadataCache) {
    H5C__json_write_destroy_cache_log_msg(cache);
}

#[allow(non_snake_case)]
pub fn H5C_log_write_evict_cache_msg(cache: &mut MetadataCache) {
    H5C__json_write_evict_cache_log_msg(cache);
}

#[allow(non_snake_case)]
pub fn H5C_log_write_expunge_entry_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__json_write_expunge_entry_log_msg(cache, addr);
}

#[allow(non_snake_case)]
pub fn H5C_log_write_flush_cache_msg(cache: &mut MetadataCache) {
    H5C__json_write_flush_cache_log_msg(cache);
}

#[allow(non_snake_case)]
pub fn H5C_log_write_insert_entry_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__json_write_insert_entry_log_msg(cache, addr);
}

#[allow(non_snake_case)]
pub fn H5C_log_write_mark_entry_dirty_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__json_write_log_message(cache, "mark_entry_dirty", &format!("{addr:#x}"));
}

#[allow(non_snake_case)]
pub fn H5C_log_write_mark_entry_clean_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__json_write_mark_entry_clean_log_msg(cache, addr);
}

#[allow(non_snake_case)]
pub fn H5C_log_write_mark_unserialized_entry_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__json_write_mark_unserialized_entry_log_msg(cache, addr);
}

#[allow(non_snake_case)]
pub fn H5C_log_write_mark_serialized_entry_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__json_write_mark_serialized_entry_log_msg(cache, addr);
}

#[allow(non_snake_case)]
pub fn H5C_log_write_move_entry_msg(cache: &mut MetadataCache, old_addr: u64, new_addr: u64) {
    H5C__json_write_move_entry_log_msg(cache, old_addr, new_addr);
}

#[allow(non_snake_case)]
pub fn H5C_log_write_pin_entry_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__json_write_pin_entry_log_msg(cache, addr);
}

#[allow(non_snake_case)]
pub fn H5C_log_write_create_fd_msg(cache: &mut MetadataCache, parent: u64, child: u64) {
    H5C__json_write_create_fd_log_msg(cache, parent, child);
}

#[allow(non_snake_case)]
pub fn H5C_log_write_protect_entry_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__json_write_protect_entry_log_msg(cache, addr);
}

#[allow(non_snake_case)]
pub fn H5C_log_write_resize_entry_msg(cache: &mut MetadataCache, addr: u64, size: usize) {
    H5C__json_write_resize_entry_log_msg(cache, addr, size);
}

#[allow(non_snake_case)]
pub fn H5C_log_write_unpin_entry_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__json_write_unpin_entry_log_msg(cache, addr);
}

#[allow(non_snake_case)]
pub fn H5C_log_write_destroy_fd_msg(cache: &mut MetadataCache, parent: u64, child: u64) {
    H5C__json_write_destroy_fd_log_msg(cache, parent, child);
}

#[allow(non_snake_case)]
pub fn H5C_log_write_unprotect_entry_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__json_write_unprotect_entry_log_msg(cache, addr);
}

#[allow(non_snake_case)]
pub fn H5C_log_write_set_cache_config_msg(cache: &mut MetadataCache) {
    H5C__json_write_set_cache_config_log_msg(cache);
}

#[allow(non_snake_case)]
pub fn H5C_log_write_remove_entry_msg(cache: &mut MetadataCache, addr: u64) {
    H5C__json_write_remove_entry_log_msg(cache, addr);
}

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
}
