use std::collections::BTreeMap;

use crate::error::{Error, Result};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PageBufferStats {
    pub accesses: u64,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub writes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageEntry {
    pub addr: u64,
    pub data: Vec<u8>,
    pub dirty: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageBuffer {
    page_size: usize,
    capacity_pages: usize,
    entries: BTreeMap<u64, PageEntry>,
    stats: PageBufferStats,
    enabled: bool,
}

impl PageBuffer {
    pub fn create(page_size: usize, capacity_pages: usize) -> Result<Self> {
        if page_size == 0 {
            return Err(Error::InvalidFormat("page buffer page size is zero".into()));
        }
        Ok(Self {
            page_size,
            capacity_pages,
            entries: BTreeMap::new(),
            stats: PageBufferStats::default(),
            enabled: capacity_pages > 0,
        })
    }

    pub fn reset_stats(&mut self) {
        self.stats = PageBufferStats::default();
    }

    pub fn get_stats(&self) -> PageBufferStats {
        self.stats.clone()
    }

    pub fn print_stats(&self) -> String {
        format!(
            "PageBufferStats(accesses={}, hits={}, misses={}, evictions={}, writes={})",
            self.stats.accesses,
            self.stats.hits,
            self.stats.misses,
            self.stats.evictions,
            self.stats.writes
        )
    }

    pub fn flush_cb(entry: &mut PageEntry) {
        entry.dirty = false;
    }

    pub fn flush(&mut self) {
        for entry in self.entries.values_mut() {
            Self::flush_cb(entry);
        }
    }

    pub fn dest_cb(_entry: PageEntry) {}

    pub fn dest(mut self) {
        self.entries.clear();
        self.enabled = false;
    }

    pub fn add_new_page(&mut self, addr: u64, data: Vec<u8>) -> Result<()> {
        if data.len() > self.page_size {
            return Err(Error::InvalidFormat(
                "page buffer entry exceeds page size".into(),
            ));
        }
        self.insert_entry(PageEntry {
            addr,
            data,
            dirty: false,
        })
    }

    pub fn add_new_page_from_slice(&mut self, addr: u64, data: &[u8]) -> Result<()> {
        self.add_new_page(addr, data.to_vec())
    }

    pub fn update_entry(&mut self, addr: u64, data: Vec<u8>) -> Result<()> {
        if data.len() > self.page_size {
            return Err(Error::InvalidFormat(
                "page buffer entry exceeds page size".into(),
            ));
        }
        let entry = self.entries.get_mut(&addr).ok_or_else(|| {
            Error::InvalidFormat(format!("page buffer entry {addr:#x} not found"))
        })?;
        entry.data = data;
        entry.dirty = true;
        self.stats.writes = self.stats.writes.saturating_add(1);
        Ok(())
    }

    pub fn update_entry_from_slice(&mut self, addr: u64, data: &[u8]) -> Result<()> {
        if data.len() > self.page_size {
            return Err(Error::InvalidFormat(
                "page buffer entry exceeds page size".into(),
            ));
        }
        let entry = self.entries.get_mut(&addr).ok_or_else(|| {
            Error::InvalidFormat(format!("page buffer entry {addr:#x} not found"))
        })?;
        entry.data.resize(data.len(), 0);
        entry.data.copy_from_slice(data);
        entry.dirty = true;
        self.stats.writes = self.stats.writes.saturating_add(1);
        Ok(())
    }

    pub fn remove_entry(&mut self, addr: u64) -> Result<PageEntry> {
        self.entries
            .remove(&addr)
            .ok_or_else(|| Error::InvalidFormat(format!("page buffer entry {addr:#x} not found")))
    }

    pub fn read(&mut self, addr: u64) -> Option<Vec<u8>> {
        self.stats.accesses = self.stats.accesses.saturating_add(1);
        if let Some(entry) = self.entries.get(&addr) {
            self.stats.hits = self.stats.hits.saturating_add(1);
            Some(entry.data.clone())
        } else {
            self.stats.misses = self.stats.misses.saturating_add(1);
            None
        }
    }

    pub fn read_view(&mut self, addr: u64) -> Option<&[u8]> {
        self.stats.accesses = self.stats.accesses.saturating_add(1);
        if let Some(entry) = self.entries.get(&addr) {
            self.stats.hits = self.stats.hits.saturating_add(1);
            Some(entry.data.as_slice())
        } else {
            self.stats.misses = self.stats.misses.saturating_add(1);
            None
        }
    }

    pub fn read_into(&mut self, addr: u64, out: &mut [u8]) -> Result<bool> {
        self.stats.accesses = self.stats.accesses.saturating_add(1);
        if let Some(entry) = self.entries.get(&addr) {
            self.stats.hits = self.stats.hits.saturating_add(1);
            if out.len() != entry.data.len() {
                return Err(Error::InvalidFormat(
                    "page buffer output length mismatch".into(),
                ));
            }
            out.copy_from_slice(&entry.data);
            Ok(true)
        } else {
            self.stats.misses = self.stats.misses.saturating_add(1);
            Ok(false)
        }
    }

    pub fn write(&mut self, addr: u64, data: Vec<u8>) -> Result<()> {
        if self.entries.contains_key(&addr) {
            self.update_entry(addr, data)
        } else {
            self.insert_entry(PageEntry {
                addr,
                data,
                dirty: true,
            })?;
            self.stats.writes = self.stats.writes.saturating_add(1);
            Ok(())
        }
    }

    pub fn write_slice(&mut self, addr: u64, data: &[u8]) -> Result<()> {
        if self.entries.contains_key(&addr) {
            self.update_entry_from_slice(addr, data)
        } else {
            self.insert_entry(PageEntry {
                addr,
                data: data.to_vec(),
                dirty: true,
            })?;
            self.stats.writes = self.stats.writes.saturating_add(1);
            Ok(())
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn insert_entry(&mut self, entry: PageEntry) -> Result<()> {
        if entry.data.len() > self.page_size {
            return Err(Error::InvalidFormat(
                "page buffer entry exceeds page size".into(),
            ));
        }
        if self.capacity_pages == 0 {
            return Ok(());
        }
        while self.entries.len() >= self.capacity_pages {
            let Some((&addr, _)) = self.entries.iter().next() else {
                break;
            };
            self.entries.remove(&addr);
            self.stats.evictions = self.stats.evictions.saturating_add(1);
        }
        self.entries.insert(entry.addr, entry);
        Ok(())
    }

    pub fn write_entry(entry: &mut PageEntry, data: Vec<u8>) {
        entry.data = data;
        entry.dirty = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_buffer_tracks_hits_misses_and_evictions() {
        let mut pb = PageBuffer::create(8, 1).unwrap();
        pb.add_new_page(0, b"abc".to_vec()).unwrap();
        assert_eq!(pb.read(0).unwrap(), b"abc");
        assert!(pb.read(8).is_none());
        pb.add_new_page(8, b"def".to_vec()).unwrap();
        assert!(pb.read(0).is_none());
        let stats = pb.get_stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 2);
        assert_eq!(stats.evictions, 1);
    }

    #[test]
    fn page_buffer_supports_borrowed_and_caller_buffer_reads() {
        let mut pb = PageBuffer::create(8, 2).unwrap();
        pb.write_slice(0, b"abc").unwrap();
        assert_eq!(pb.read_view(0).unwrap(), b"abc");

        let mut out = [0; 3];
        assert!(pb.read_into(0, &mut out).unwrap());
        assert_eq!(&out, b"abc");

        pb.update_entry_from_slice(0, b"defg").unwrap();
        assert!(pb.read_into(0, &mut out).is_err());
        assert_eq!(pb.read(0).unwrap(), b"defg");
    }
}
