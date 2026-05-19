use std::collections::BTreeMap;
use std::fmt::{self, Write};

use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FreeSpaceClass {
    Simple,
    Small,
    Large,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FreeSpaceSection {
    pub addr: u64,
    pub size: u64,
    pub class: FreeSpaceClass,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FreeSpaceStats {
    pub section_count: usize,
    pub total_space: u64,
    pub largest_section: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FreeSpaceCreateParams {
    pub alignment: u64,
    pub threshold: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FreeSpaceManager {
    sections: BTreeMap<u64, FreeSpaceSection>,
    ref_count: usize,
    dirty: bool,
    locked: bool,
    params: FreeSpaceCreateParams,
    flush_dependencies: usize,
}

impl Default for FreeSpaceManager {
    fn default() -> Self {
        Self::new()
    }
}

impl FreeSpaceSection {
    const SERIALIZED_SIZE: usize = 17;

    /// Construct a new free-space section, rejecting a zero size.
    pub fn new(addr: u64, size: u64, class: FreeSpaceClass) -> Result<Self> {
        if size == 0 {
            return Err(Error::InvalidFormat(
                "free-space section size is zero".into(),
            ));
        }
        Ok(Self { addr, size, class })
    }

    /// One past the last byte covered by this section, with overflow check.
    pub fn end(&self) -> Result<u64> {
        self.addr
            .checked_add(self.size)
            .ok_or_else(|| Error::InvalidFormat("free-space section end overflow".into()))
    }

    /// Return the serialized size in bytes of one section record.
    pub fn serialize_size(&self) -> usize {
        Self::SERIALIZED_SIZE
    }

    /// Append the serialized representation of this section to `out`.
    pub fn serialize(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.addr.to_le_bytes());
        out.extend_from_slice(&self.size.to_le_bytes());
        out.push(match self.class {
            FreeSpaceClass::Simple => 0,
            FreeSpaceClass::Small => 1,
            FreeSpaceClass::Large => 2,
        });
    }

    /// Serialize, returning an error if the section fails validation.
    pub fn serialize_checked(&self, out: &mut Vec<u8>) -> Result<()> {
        if !self.valid() {
            return Err(Error::InvalidFormat(
                "invalid free-space section cannot be serialized".into(),
            ));
        }
        self.serialize(out);
        Ok(())
    }

    /// Deserialize a section record from a 17-byte image.
    pub fn deserialize(data: &[u8]) -> Result<Self> {
        if data.len() < 17 {
            return Err(Error::InvalidFormat(
                "free-space section image is truncated".into(),
            ));
        }
        let addr = read_u64_le_at(data, 0, "free-space section address")?;
        let size = read_u64_le_at(data, 8, "free-space section size")?;
        let class = match data[16] {
            0 => FreeSpaceClass::Simple,
            1 => FreeSpaceClass::Small,
            2 => FreeSpaceClass::Large,
            other => {
                return Err(Error::InvalidFormat(format!(
                    "invalid free-space section class {other}"
                )));
            }
        };
        Self::new(addr, size, class)
    }

    /// Return true when this section has a non-zero size and a non-overflowing end.
    pub fn valid(&self) -> bool {
        self.size != 0 && self.end().is_ok()
    }

    /// Split off the first `size` bytes as a new section, shrinking `self`.
    pub fn split(&mut self, size: u64) -> Result<Self> {
        if size == 0 || size > self.size {
            return Err(Error::InvalidFormat(
                "invalid free-space section split size".into(),
            ));
        }
        let allocated = Self::new(self.addr, size, self.class)?;
        self.addr = self
            .addr
            .checked_add(size)
            .ok_or_else(|| Error::InvalidFormat("free-space section split overflow".into()))?;
        self.size -= size;
        Ok(allocated)
    }

    /// Return true if `self` and `other` are adjacent sections of the same class.
    pub fn can_merge(&self, other: &Self) -> bool {
        self.class == other.class
            && (self.end().ok() == Some(other.addr) || other.end().ok() == Some(self.addr))
    }

    /// Merge `other` into `self`; returns an error if the sections are not adjacent.
    pub fn merge(&mut self, other: Self) -> Result<()> {
        if !self.can_merge(&other) {
            return Err(Error::InvalidFormat(
                "free-space sections are not mergeable".into(),
            ));
        }
        let start = self.addr.min(other.addr);
        let end = self.end()?.max(other.end()?);
        self.addr = start;
        self.size = end
            .checked_sub(start)
            .ok_or_else(|| Error::InvalidFormat("free-space section merge underflow".into()))?;
        Ok(())
    }

    /// True if this section terminates exactly at the end-of-allocation boundary.
    pub fn can_shrink(&self, eoa: u64) -> bool {
        self.end().ok() == Some(eoa)
    }
}

/// Bounds-checked subslice `data[pos..pos+len]`, surfacing `context` in errors.
fn checked_window<'a>(data: &'a [u8], pos: usize, len: usize, context: &str) -> Result<&'a [u8]> {
    let end = pos
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))?;
    data.get(pos..end)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))
}

/// Read a little-endian `u64` from `data` at `pos`, surfacing `context` in errors.
fn read_u64_le_at(data: &[u8], pos: usize, context: &str) -> Result<u64> {
    let bytes = checked_window(data, pos, 8, context)?;
    Ok(u64::from_le_bytes(bytes.try_into().map_err(|_| {
        Error::InvalidFormat(format!("{context} is truncated"))
    })?))
}

impl FreeSpaceManager {
    /// Initialize the free-space interface eagerly.
    pub fn init() -> Self {
        Self::new()
    }

    /// Allocate and initialize a file free-space info structure.
    pub fn create() -> Self {
        Self::new()
    }

    /// Open an existing file free-space info structure from section records.
    pub fn open_from_iter<I>(sections: I) -> Result<Self>
    where
        I: IntoIterator<Item = FreeSpaceSection>,
    {
        let mut manager = Self::new();
        for section in sections {
            manager.sect_add(section)?;
        }
        Ok(manager)
    }

    /// Open an existing file free-space info structure from a list of sections.
    #[deprecated(note = "use open_from_iter to avoid requiring Vec storage")]
    pub fn open(sections: Vec<FreeSpaceSection>) -> Result<Self> {
        Self::open_from_iter(sections)
    }

    /// Delete the free-space manager state on disk.
    pub fn delete(&mut self) {
        self.sections.clear();
        self.dirty = true;
    }

    /// Destroy and deallocate the free-list structure.
    pub fn close(self) {}

    /// Create a new in-memory free-space manager with default parameters.
    pub fn new() -> Self {
        Self {
            sections: BTreeMap::new(),
            ref_count: 1,
            dirty: false,
            locked: false,
            params: FreeSpaceCreateParams {
                alignment: 1,
                threshold: 0,
            },
            flush_dependencies: 0,
        }
    }

    /// Increment the reference count on the free-space header.
    pub fn incr(&mut self) {
        self.ref_count = self.ref_count.saturating_add(1);
    }

    /// Decrement the reference count on the free-space header.
    pub fn decr(&mut self) -> Result<()> {
        if self.ref_count == 0 {
            return Err(Error::InvalidFormat(
                "free-space manager reference underflow".into(),
            ));
        }
        self.ref_count -= 1;
        Ok(())
    }

    /// Mark the free-space header as dirty.
    pub fn dirty(&mut self) {
        self.dirty = true;
    }

    /// Borrow the create parameters used for the free-space manager header.
    pub fn alloc_hdr_ref(&self) -> &FreeSpaceCreateParams {
        &self.params
    }

    /// Allocate a new free-space section record.
    pub fn alloc_sect(addr: u64, size: u64, class: FreeSpaceClass) -> Result<FreeSpaceSection> {
        FreeSpaceSection::new(addr, size, class)
    }

    /// Free a region by adding it as a simple section to the manager.
    pub fn free(&mut self, addr: u64, size: u64) -> Result<()> {
        self.sect_add(FreeSpaceSection::new(addr, size, FreeSpaceClass::Simple)?)
    }

    /// Destroy the in-memory free-space header.
    pub fn hdr_dest(self) {}

    /// Free a size-tracking section node (no-op in Rust ownership model).
    pub fn sinfo_free_sect_cb(_section: FreeSpaceSection) {}

    /// Free a size-tracking bin node (no-op in Rust ownership model).
    pub fn sinfo_free_node_cb(_addr: u64) {}

    /// Destroy the in-memory section-info structure.
    pub fn sinfo_dest(&mut self) {
        self.sections.clear();
    }

    /// Return the number of tracked sections.
    pub fn get_sect_count(&self) -> usize {
        self.sections.len()
    }

    /// Verify that all tracked sections are well-formed.
    pub fn assert_valid(&self) -> Result<()> {
        for section in self.sections.values() {
            if !section.valid() {
                return Err(Error::InvalidFormat(
                    "invalid free-space section in manager".into(),
                ));
            }
        }
        Ok(())
    }

    /// Verify the trailing checksum of a serialized free-space header image.
    pub fn cache_hdr_verify_chksum(data: &[u8]) -> Result<()> {
        verify_trailing_checksum(data, "free-space header")
    }

    /// Allocate a free-space manager and populate it from a serialized header image.
    pub fn cache_hdr_deserialize(data: &[u8]) -> Result<Self> {
        if data.len() != 20 {
            return Err(Error::InvalidFormat(
                "free-space header image has invalid length".into(),
            ));
        }
        Self::cache_hdr_verify_chksum(data)?;
        let alignment = read_u64_le_at(data, 0, "free-space header alignment")?;
        let threshold = read_u64_le_at(data, 8, "free-space header threshold")?;
        if alignment == 0 {
            return Err(Error::InvalidFormat(
                "free-space header alignment is zero".into(),
            ));
        }
        Ok(Self {
            params: FreeSpaceCreateParams {
                alignment,
                threshold,
            },
            ..Self::new()
        })
    }

    /// Encoded on-disk size of the free-space header image in bytes.
    pub fn cache_hdr_image_len(&self) -> usize {
        8 + 8 + 4
    }

    /// Pre-serialize hook for the free-space header; clears the dirty flag.
    pub fn cache_hdr_pre_serialize(&mut self) {
        self.dirty = false;
    }

    /// Append the serialized free-space header image with trailing checksum to `out`.
    pub fn cache_hdr_serialize_into(&self, out: &mut Vec<u8>) -> Result<()> {
        if self.params.alignment == 0 {
            return Err(Error::InvalidFormat(
                "free-space header alignment is zero".into(),
            ));
        }
        out.reserve(self.cache_hdr_image_len());
        let start = out.len();
        out.extend_from_slice(&self.params.alignment.to_le_bytes());
        out.extend_from_slice(&self.params.threshold.to_le_bytes());
        let checksum = crate::format::checksum::checksum_metadata(&out[start..]);
        out.extend_from_slice(&checksum.to_le_bytes());
        Ok(())
    }

    /// Cache action notification hook for the header (no-op).
    pub fn cache_hdr_notify(&self) {}

    /// Free the in-core representation of the free-space header.
    pub fn cache_hdr_free_icr(self) {}

    /// Initial on-disk image size of the section info (just the trailing checksum).
    pub fn cache_sinfo_get_initial_load_size() -> usize {
        4
    }

    /// Verify the trailing checksum of a serialized section-info image.
    pub fn cache_sinfo_verify_chksum(data: &[u8]) -> Result<()> {
        verify_trailing_checksum(data, "free-space section info")
    }

    /// Allocate a manager and populate it from a serialized section-info image.
    pub fn cache_sinfo_deserialize(data: &[u8]) -> Result<Self> {
        if data.len() < 4 {
            return Err(Error::InvalidFormat(
                "free-space section info is truncated".into(),
            ));
        }
        let payload_end = data.len().saturating_sub(4);
        let section_size = FreeSpaceSection::SERIALIZED_SIZE;
        if payload_end % section_size != 0 {
            return Err(Error::InvalidFormat(
                "free-space section info has a partial section record".into(),
            ));
        }
        Self::cache_sinfo_verify_chksum(data)?;
        let mut manager = Self::new();
        let mut pos = 0usize;
        while pos < payload_end {
            let end = pos
                .checked_add(section_size)
                .ok_or_else(|| Error::InvalidFormat("free-space section offset overflow".into()))?;
            let section = data.get(pos..end).ok_or_else(|| {
                Error::InvalidFormat("free-space section info is truncated".into())
            })?;
            manager.sect_add(FreeSpaceSection::deserialize(section)?)?;
            pos = end;
        }
        Ok(manager)
    }

    /// Compute the on-disk size of the section-info image including its checksum.
    pub fn cache_sinfo_image_len(&self) -> Result<usize> {
        let mut len = 4usize;
        for section in self.sections.values() {
            len = len.checked_add(section.serialize_size()).ok_or_else(|| {
                Error::InvalidFormat("free-space section info image length overflow".into())
            })?;
        }
        Ok(len)
    }

    /// Pre-serialize hook for the section info; clears the dirty flag.
    pub fn cache_sinfo_pre_serialize(&mut self) {
        self.dirty = false;
    }

    /// Append the serialized section-info image with trailing checksum to `out`.
    pub fn cache_sinfo_serialize_into(&self, out: &mut Vec<u8>) -> Result<()> {
        out.reserve(self.cache_sinfo_image_len()?);
        let start = out.len();
        for section in self.sections.values() {
            section.serialize_checked(out)?;
        }
        let checksum = crate::format::checksum::checksum_metadata(&out[start..]);
        out.extend_from_slice(&checksum.to_le_bytes());
        Ok(())
    }

    /// Cache action notification hook for the section info (no-op).
    pub fn cache_sinfo_notify(&self) {}

    /// Free the in-core representation of the section info.
    pub fn cache_sinfo_free_icr(self) {}

    /// Skip-list iterator callback that serializes one free-space section.
    pub fn sinfo_serialize_sect_cb(section: &FreeSpaceSection, out: &mut Vec<u8>) -> Result<()> {
        section.serialize_checked(out)
    }

    /// Skip-list iterator callback that serializes all sections in a bin.
    pub fn sinfo_serialize_node_cb(sections: &[FreeSpaceSection], out: &mut Vec<u8>) -> Result<()> {
        for section in sections {
            section.serialize_checked(out)?;
        }
        Ok(())
    }

    /// Borrow the common section-class table.
    pub fn sect_classes() -> &'static [FreeSpaceClass] {
        &[
            FreeSpaceClass::Simple,
            FreeSpaceClass::Small,
            FreeSpaceClass::Large,
        ]
    }

    /// Allocate a free-space section node of a particular type.
    pub fn sect_node_new(section: FreeSpaceSection) -> FreeSpaceSection {
        section
    }

    /// Retrieve metadata statistics for the free-space manager.
    pub fn stat_info(&self) -> FreeSpaceStats {
        self.stat_info_checked().unwrap_or_else(|_| {
            let mut stats = FreeSpaceStats::default();
            stats.section_count = self.sections.len();
            stats.total_space = u64::MAX;
            for section in self.sections.values() {
                stats.largest_section = stats.largest_section.max(section.size);
            }
            stats
        })
    }

    /// Checked variant of [`stat_info`] that errors on total-size overflow.
    pub fn stat_info_checked(&self) -> Result<FreeSpaceStats> {
        let mut stats = FreeSpaceStats::default();
        stats.section_count = self.sections.len();
        for section in self.sections.values() {
            stats.total_space = stats
                .total_space
                .checked_add(section.size)
                .ok_or_else(|| Error::InvalidFormat("free-space total size overflow".into()))?;
            stats.largest_section = stats.largest_section.max(section.size);
        }
        Ok(stats)
    }

    /// Write debugging info about the free-space manager into `out`.
    pub fn write_debug<W: Write>(&self, out: &mut W) -> fmt::Result {
        let stats = self.stat_info();
        write!(
            out,
            "FreeSpaceManager(sections={}, total_space={}, largest_section={})",
            stats.section_count, stats.total_space, stats.largest_section
        )
    }

    /// Write debugging info about a single free-space section into `out`.
    pub fn write_sect_debug<W: Write>(section: &FreeSpaceSection, out: &mut W) -> fmt::Result {
        write!(
            out,
            "FreeSpaceSection(addr={:#x}, size={}, class={:?})",
            section.addr, section.size, section.class
        )
    }

    /// Write debugging info for all tracked sections into `out`.
    pub fn write_sects_debug<W: Write>(&self, out: &mut W) -> fmt::Result {
        for (idx, section) in self.sections.values().enumerate() {
            if idx != 0 {
                out.write_char('\n')?;
            }
            Self::write_sect_debug(section, out)?;
        }
        Ok(())
    }

    /// Create a fresh section-info structure.
    pub fn sinfo_new() -> Self {
        Self::new()
    }

    /// Make certain the section info is loaded; marks it locked.
    pub fn sinfo_lock(&mut self) {
        self.locked = true;
    }

    /// Release the section info back to the cache; marks it unlocked.
    pub fn sinfo_unlock(&mut self) {
        self.locked = false;
    }

    /// Serialized size of all section records in the manager.
    pub fn sect_serialize_size(section: &FreeSpaceSection) -> usize {
        section.serialize_size()
    }

    /// Grow a section's size by `amount`, returning an error on overflow.
    pub fn sect_increase(section: &mut FreeSpaceSection, amount: u64) -> Result<()> {
        section.size = section
            .size
            .checked_add(amount)
            .ok_or_else(|| Error::InvalidFormat("free-space section size overflow".into()))?;
        Ok(())
    }

    /// Shrink a section's size by `amount`, returning an error on underflow.
    pub fn sect_decrease(section: &mut FreeSpaceSection, amount: u64) -> Result<()> {
        if amount > section.size {
            return Err(Error::InvalidFormat(
                "free-space section decrease underflow".into(),
            ));
        }
        section.size -= amount;
        Ok(())
    }

    /// Decrement the count of sections of a particular size by removing one.
    pub fn size_node_decr(&mut self, addr: u64) -> Result<()> {
        self.sections.remove(&addr).ok_or_else(|| {
            Error::InvalidFormat(format!("free-space section {addr:#x} not found"))
        })?;
        Ok(())
    }

    /// Remove a section from the size-tracking data structures.
    pub fn sect_unlink_size(&mut self, addr: u64) -> Result<FreeSpaceSection> {
        self.sect_remove(addr)
    }

    /// Remove a section from the rest of the manager after unlinking from size tracking.
    pub fn sect_unlink_rest(&mut self, addr: u64) -> Result<FreeSpaceSection> {
        self.sect_remove(addr)
    }

    /// Remove a section from the manager (real removal, not a hold).
    pub fn sect_remove_real(&mut self, addr: u64) -> Result<FreeSpaceSection> {
        self.sect_remove(addr)
    }

    /// Remove a section from the manager and mark the manager dirty.
    pub fn sect_remove(&mut self, addr: u64) -> Result<FreeSpaceSection> {
        self.dirty = true;
        self.sections
            .remove(&addr)
            .ok_or_else(|| Error::InvalidFormat(format!("free-space section {addr:#x} not found")))
    }

    /// Add a section to the size-tracking bins.
    pub fn sect_link_size(&mut self, section: FreeSpaceSection) -> Result<()> {
        self.sect_add(section)
    }

    /// Link a section into the non-size tracking data structures.
    pub fn sect_link_rest(&mut self, section: FreeSpaceSection) -> Result<()> {
        self.sect_add(section)
    }

    /// Link a section into the internal data structures of the manager.
    pub fn sect_link(&mut self, section: FreeSpaceSection) -> Result<()> {
        self.sect_add(section)
    }

    /// Merge `rhs` into `lhs`; both sections must be adjacent and of the same class.
    pub fn sect_merge(lhs: &mut FreeSpaceSection, rhs: FreeSpaceSection) -> Result<()> {
        lhs.merge(rhs)
    }

    /// Add a section of free space to the free list, merging adjacent sections.
    pub fn sect_add(&mut self, mut section: FreeSpaceSection) -> Result<()> {
        section = self.sect_try_merge(section)?;
        self.sections.insert(section.addr, section);
        self.dirty = true;
        Ok(())
    }

    /// Try to extend a block using space from a section on the free list.
    pub fn sect_try_extend(&mut self, addr: u64, amount: u64) -> Result<bool> {
        if let Some(section) = self.sections.get_mut(&addr) {
            Self::sect_increase(section, amount)?;
            self.dirty = true;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Attempt to merge a returned section with existing adjacent free space.
    pub fn sect_try_merge(&mut self, mut section: FreeSpaceSection) -> Result<FreeSpaceSection> {
        if let Some((&prev_addr, prev)) = self.sections.range(..section.addr).next_back() {
            if prev.can_merge(&section) {
                let prev = self.sections.remove(&prev_addr).unwrap();
                section.merge(prev)?;
            }
        }
        if let Some((&next_addr, next)) = self.sections.range(section.addr..).next() {
            if section.addr != next_addr && section.can_merge(next) {
                let next = self.sections.remove(&next_addr).unwrap();
                section.merge(next)?;
            }
        }
        Ok(section)
    }

    /// Locate the first existing section large enough to satisfy `size`.
    pub fn sect_find_node(&self, size: u64) -> Option<&FreeSpaceSection> {
        self.sections.values().find(|section| section.size >= size)
    }

    /// Allocate space from the free list, splitting the section if necessary.
    pub fn sect_find(&mut self, size: u64) -> Result<Option<FreeSpaceSection>> {
        self.sect_find_matching(size, |_| true)
    }

    /// Allocate space from a matching free-list section, splitting it if necessary.
    pub fn sect_find_matching<F>(
        &mut self,
        size: u64,
        mut predicate: F,
    ) -> Result<Option<FreeSpaceSection>>
    where
        F: FnMut(&FreeSpaceSection) -> bool,
    {
        let Some((&addr, _)) = self
            .sections
            .iter()
            .find(|(_, section)| section.size >= size && predicate(section))
        else {
            return Ok(None);
        };
        let mut section = self.sections.remove(&addr).ok_or_else(|| {
            Error::InvalidFormat(format!("free-space section {addr:#x} not found"))
        })?;
        if section.size == size {
            self.dirty = true;
            Ok(Some(section))
        } else {
            let allocated = section.split(size)?;
            self.sections.insert(section.addr, section);
            self.dirty = true;
            Ok(Some(allocated))
        }
    }

    /// Allocate space from a free-list section of `class`, splitting it if necessary.
    pub fn sect_find_by_class(
        &mut self,
        class: FreeSpaceClass,
        size: u64,
    ) -> Result<Option<FreeSpaceSection>> {
        self.sect_find_matching(size, |section| section.class == class)
    }

    /// Borrow all sections managed by the free-space header in address order.
    pub fn sections(&self) -> impl Iterator<Item = &FreeSpaceSection> {
        self.sections.values()
    }

    /// Skip-list iterator callback that invokes `f` for each section.
    pub fn iterate_sect_cb<F: FnMut(&FreeSpaceSection)>(&self, mut f: F) {
        for section in self.sections.values() {
            f(section);
        }
    }

    /// Skip-list iterator callback that visits sections per bin.
    pub fn iterate_node_cb<F: FnMut(&FreeSpaceSection)>(&self, f: F) {
        self.iterate_sect_cb(f);
    }

    /// Iterate over all sections managed by the free-space header.
    pub fn sect_iterate<F: FnMut(&FreeSpaceSection)>(&self, f: F) {
        self.iterate_sect_cb(f);
    }

    /// Retrieve aggregate info about the managed sections.
    pub fn sect_stats(&self) -> FreeSpaceStats {
        self.stat_info()
    }

    /// Checked variant of [`sect_stats`] that errors on overflow.
    pub fn sect_stats_checked(&self) -> Result<FreeSpaceStats> {
        self.stat_info_checked()
    }

    /// Change a section's class, updating internal categorization.
    pub fn sect_change_class(section: &mut FreeSpaceSection, class: FreeSpaceClass) {
        section.class = class;
    }

    /// Verify a single section is sane.
    pub fn sect_assert(section: &FreeSpaceSection) -> Result<()> {
        if section.valid() {
            Ok(())
        } else {
            Err(Error::InvalidFormat("invalid free-space section".into()))
        }
    }

    /// Shrink the last section if it sits at the end-of-allocation boundary.
    pub fn sect_try_shrink_eoa(&mut self, eoa: u64) -> Result<Option<FreeSpaceSection>> {
        let Some((&addr, _section)) = self
            .sections
            .iter()
            .find(|(_, section)| section.can_shrink(eoa))
        else {
            return Ok(None);
        };
        self.dirty = true;
        Ok(self.sections.remove(&addr))
    }

    /// Borrow the create parameters used by this manager (test helper).
    pub fn get_cparam_test_ref(&self) -> &FreeSpaceCreateParams {
        &self.params
    }

    /// Compare two sets of create parameters for equality (test helper).
    pub fn cmp_cparam_test(lhs: &FreeSpaceCreateParams, rhs: &FreeSpaceCreateParams) -> bool {
        lhs == rhs
    }

    /// Create a flush dependency between two data-structure components.
    pub fn create_flush_depend(&mut self) {
        let _ = self.create_flush_depend_checked();
    }

    /// Checked variant of [`create_flush_depend`] that errors on overflow.
    pub fn create_flush_depend_checked(&mut self) -> Result<()> {
        self.flush_dependencies = self
            .flush_dependencies
            .checked_add(1)
            .ok_or_else(|| Error::InvalidFormat("free-space flush dependency overflow".into()))?;
        Ok(())
    }

    /// Destroy a flush dependency between two data-structure components.
    pub fn destroy_flush_depend(&mut self) -> Result<()> {
        if self.flush_dependencies == 0 {
            return Err(Error::InvalidFormat(
                "free-space flush dependency underflow".into(),
            ));
        }
        self.flush_dependencies -= 1;
        Ok(())
    }
}

/// Verify that the last four bytes of `data` form a valid metadata checksum over the rest.
fn verify_trailing_checksum(data: &[u8], context: &str) -> Result<()> {
    if data.len() < 4 {
        return Err(Error::InvalidFormat(format!("{context} image too short")));
    }
    let split = data.len() - 4;
    let stored_bytes = checked_window(data, split, 4, &format!("{context} checksum"))?;
    let stored = u32::from_le_bytes(
        stored_bytes
            .try_into()
            .map_err(|_| Error::InvalidFormat(format!("{context} checksum is truncated")))?,
    );
    let computed = crate::format::checksum::checksum_metadata(checked_window(
        data,
        0,
        split,
        &format!("{context} payload"),
    )?);
    if stored != computed {
        return Err(Error::InvalidFormat(format!(
            "{context} checksum mismatch: stored={stored:#010x}, computed={computed:#010x}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sections_merge_and_allocate_first_fit() {
        let mut fs = FreeSpaceManager::new();
        fs.sect_add(FreeSpaceSection::new(100, 10, FreeSpaceClass::Simple).unwrap())
            .unwrap();
        fs.sect_add(FreeSpaceSection::new(110, 10, FreeSpaceClass::Simple).unwrap())
            .unwrap();
        assert_eq!(fs.get_sect_count(), 1);
        let allocated = fs.sect_find(8).unwrap().unwrap();
        assert_eq!(allocated.addr, 100);
        assert_eq!(allocated.size, 8);
        assert_eq!(fs.sect_find_node(12).unwrap().addr, 108);
    }

    #[test]
    fn section_info_roundtrips_with_checksum() {
        let mut fs = FreeSpaceManager::new();
        fs.free(64, 32).unwrap();
        let mut image = Vec::new();
        fs.cache_sinfo_serialize_into(&mut image).unwrap();
        let decoded = FreeSpaceManager::cache_sinfo_deserialize(&image).unwrap();
        assert_eq!(decoded.get_sect_count(), 1);
    }

    #[test]
    fn free_space_header_roundtrips_and_rejects_malformed_images() {
        let mut fs = FreeSpaceManager::new();
        fs.params.alignment = 8;
        fs.params.threshold = 4096;

        let mut image = Vec::new();
        fs.cache_hdr_serialize_into(&mut image).unwrap();
        let decoded = FreeSpaceManager::cache_hdr_deserialize(&image).unwrap();
        assert_eq!(decoded.params.alignment, 8);
        assert_eq!(decoded.params.threshold, 4096);
        assert_eq!(decoded.get_sect_count(), 0);

        let err = FreeSpaceManager::cache_hdr_deserialize(&image[..19]).unwrap_err();
        assert!(matches!(err, Error::InvalidFormat(_)));

        let mut bad_checksum = image.clone();
        *bad_checksum.last_mut().unwrap() ^= 0x80;
        let err = FreeSpaceManager::cache_hdr_deserialize(&bad_checksum).unwrap_err();
        assert!(matches!(err, Error::InvalidFormat(_)));

        let mut zero_alignment = image;
        zero_alignment[..8].copy_from_slice(&0u64.to_le_bytes());
        let checksum = crate::format::checksum::checksum_metadata(&zero_alignment[..16]);
        zero_alignment[16..20].copy_from_slice(&checksum.to_le_bytes());
        let err = FreeSpaceManager::cache_hdr_deserialize(&zero_alignment).unwrap_err();
        assert!(matches!(err, Error::InvalidFormat(_)));

        fs.params.alignment = 0;
        let err = fs.cache_hdr_serialize_into(&mut Vec::new()).unwrap_err();
        assert!(matches!(err, Error::InvalidFormat(_)));
    }

    #[test]
    fn section_info_rejects_partial_section_payload() {
        let mut image = vec![
            0u8;
            FreeSpaceSection::new(1, 1, FreeSpaceClass::Simple)
                .unwrap()
                .serialize_size()
                - 1
        ];
        let checksum = crate::format::checksum::checksum_metadata(&image);
        image.extend_from_slice(&checksum.to_le_bytes());

        let err = FreeSpaceManager::cache_sinfo_deserialize(&image).unwrap_err();
        assert!(matches!(err, Error::InvalidFormat(_)));
    }

    #[test]
    fn section_info_rejects_bad_checksum() {
        let mut fs = FreeSpaceManager::new();
        fs.free(64, 32).unwrap();
        let mut image = Vec::new();
        fs.cache_sinfo_serialize_into(&mut image).unwrap();
        *image.last_mut().unwrap() ^= 0x80;

        let err = FreeSpaceManager::cache_sinfo_deserialize(&image).unwrap_err();
        assert!(matches!(err, Error::InvalidFormat(_)));

        let mut empty_image = Vec::new();
        FreeSpaceManager::new()
            .cache_sinfo_serialize_into(&mut empty_image)
            .unwrap();
        assert_eq!(empty_image.len(), 4);
        *empty_image.last_mut().unwrap() ^= 0x80;
        let err = FreeSpaceManager::cache_sinfo_deserialize(&empty_image).unwrap_err();
        assert!(matches!(err, Error::InvalidFormat(_)));
    }

    #[test]
    fn section_info_serialize_rejects_invalid_sections() {
        let invalid = FreeSpaceSection {
            addr: u64::MAX,
            size: 1,
            class: FreeSpaceClass::Simple,
        };
        let mut out = Vec::new();
        assert!(FreeSpaceManager::sinfo_serialize_sect_cb(&invalid, &mut out).is_err());

        let mut fs = FreeSpaceManager::new();
        fs.sections.insert(invalid.addr, invalid);
        assert!(fs.cache_sinfo_serialize_into(&mut Vec::new()).is_err());

        let valid = FreeSpaceSection::new(4, 8, FreeSpaceClass::Small).unwrap();
        assert!(FreeSpaceManager::sinfo_serialize_node_cb(&[valid], &mut out).is_ok());
    }

    #[test]
    fn free_space_stats_checked_rejects_total_overflow() {
        let mut fs = FreeSpaceManager::new();
        fs.sections.insert(
            0,
            FreeSpaceSection {
                addr: 0,
                size: u64::MAX,
                class: FreeSpaceClass::Simple,
            },
        );
        fs.sections.insert(
            u64::MAX,
            FreeSpaceSection {
                addr: u64::MAX,
                size: 1,
                class: FreeSpaceClass::Small,
            },
        );

        assert!(fs.stat_info_checked().is_err());
        assert!(fs.sect_stats_checked().is_err());
        assert_eq!(fs.stat_info().total_space, u64::MAX);
    }

    #[test]
    fn matching_allocation_moves_section_without_staging_clone() {
        let mut fs = FreeSpaceManager::new();
        fs.sect_add(FreeSpaceSection::new(100, 8, FreeSpaceClass::Small).unwrap())
            .unwrap();
        fs.sect_add(FreeSpaceSection::new(200, 16, FreeSpaceClass::Large).unwrap())
            .unwrap();

        let allocated = fs
            .sect_find_matching(4, |section| section.class == FreeSpaceClass::Large)
            .unwrap()
            .unwrap();

        assert_eq!(allocated.addr, 200);
        assert_eq!(allocated.size, 4);
        assert_eq!(fs.sect_find_node(12).unwrap().addr, 204);
        assert!(fs.sections().any(|section| section.addr == 100));
    }

    #[test]
    fn checked_window_rejects_offset_overflow() {
        let err = checked_window(&[], usize::MAX, 1, "free-space test window").unwrap_err();
        assert!(
            err.to_string()
                .contains("free-space test window offset overflow"),
            "unexpected error: {err}"
        );
    }
}
