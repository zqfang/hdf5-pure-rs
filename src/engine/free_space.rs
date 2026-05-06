use std::collections::BTreeMap;

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

    pub fn new(addr: u64, size: u64, class: FreeSpaceClass) -> Result<Self> {
        if size == 0 {
            return Err(Error::InvalidFormat(
                "free-space section size is zero".into(),
            ));
        }
        Ok(Self { addr, size, class })
    }

    pub fn end(&self) -> Result<u64> {
        self.addr
            .checked_add(self.size)
            .ok_or_else(|| Error::InvalidFormat("free-space section end overflow".into()))
    }

    pub fn serialize_size(&self) -> usize {
        Self::SERIALIZED_SIZE
    }

    pub fn serialize(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.addr.to_le_bytes());
        out.extend_from_slice(&self.size.to_le_bytes());
        out.push(match self.class {
            FreeSpaceClass::Simple => 0,
            FreeSpaceClass::Small => 1,
            FreeSpaceClass::Large => 2,
        });
    }

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

    pub fn valid(&self) -> bool {
        self.size != 0 && self.end().is_ok()
    }

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

    pub fn can_merge(&self, other: &Self) -> bool {
        self.class == other.class
            && (self.end().ok() == Some(other.addr) || other.end().ok() == Some(self.addr))
    }

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

    pub fn can_shrink(&self, eoa: u64) -> bool {
        self.end().ok() == Some(eoa)
    }
}

fn checked_window<'a>(data: &'a [u8], pos: usize, len: usize, context: &str) -> Result<&'a [u8]> {
    let end = pos
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))?;
    data.get(pos..end)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))
}

fn read_u64_le_at(data: &[u8], pos: usize, context: &str) -> Result<u64> {
    let bytes = checked_window(data, pos, 8, context)?;
    Ok(u64::from_le_bytes(bytes.try_into().map_err(|_| {
        Error::InvalidFormat(format!("{context} is truncated"))
    })?))
}

impl FreeSpaceManager {
    pub fn init() -> Self {
        Self::new()
    }

    pub fn create() -> Self {
        Self::new()
    }

    pub fn open(sections: Vec<FreeSpaceSection>) -> Result<Self> {
        let mut manager = Self::new();
        for section in sections {
            manager.sect_add(section)?;
        }
        Ok(manager)
    }

    pub fn delete(&mut self) {
        self.sections.clear();
        self.dirty = true;
    }

    pub fn close(self) {}

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

    pub fn incr(&mut self) {
        self.ref_count = self.ref_count.saturating_add(1);
    }

    pub fn decr(&mut self) -> Result<()> {
        if self.ref_count == 0 {
            return Err(Error::InvalidFormat(
                "free-space manager reference underflow".into(),
            ));
        }
        self.ref_count -= 1;
        Ok(())
    }

    pub fn dirty(&mut self) {
        self.dirty = true;
    }

    pub fn alloc_hdr(&self) -> FreeSpaceCreateParams {
        self.params.clone()
    }

    pub fn alloc_sect(addr: u64, size: u64, class: FreeSpaceClass) -> Result<FreeSpaceSection> {
        FreeSpaceSection::new(addr, size, class)
    }

    pub fn free(&mut self, addr: u64, size: u64) -> Result<()> {
        self.sect_add(FreeSpaceSection::new(addr, size, FreeSpaceClass::Simple)?)
    }

    pub fn hdr_dest(self) {}

    pub fn sinfo_free_sect_cb(_section: FreeSpaceSection) {}

    pub fn sinfo_free_node_cb(_addr: u64) {}

    pub fn sinfo_dest(&mut self) {
        self.sections.clear();
    }

    pub fn get_sect_count(&self) -> usize {
        self.sections.len()
    }

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

    pub fn cache_hdr_verify_chksum(data: &[u8]) -> Result<()> {
        verify_trailing_checksum(data, "free-space header")
    }

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

    pub fn cache_hdr_image_len(&self) -> usize {
        8 + 8 + 4
    }

    pub fn cache_hdr_pre_serialize(&mut self) {
        self.dirty = false;
    }

    pub fn cache_hdr_serialize(&self) -> Result<Vec<u8>> {
        if self.params.alignment == 0 {
            return Err(Error::InvalidFormat(
                "free-space header alignment is zero".into(),
            ));
        }
        let mut out = Vec::with_capacity(self.cache_hdr_image_len());
        out.extend_from_slice(&self.params.alignment.to_le_bytes());
        out.extend_from_slice(&self.params.threshold.to_le_bytes());
        let checksum = crate::format::checksum::checksum_metadata(&out);
        out.extend_from_slice(&checksum.to_le_bytes());
        Ok(out)
    }

    pub fn cache_hdr_notify(&self) {}

    pub fn cache_hdr_free_icr(self) {}

    pub fn cache_sinfo_get_initial_load_size() -> usize {
        4
    }

    pub fn cache_sinfo_verify_chksum(data: &[u8]) -> Result<()> {
        verify_trailing_checksum(data, "free-space section info")
    }

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
        let mut pos = 0usize;
        let mut sections = Vec::new();
        while pos < payload_end {
            let end = pos
                .checked_add(section_size)
                .ok_or_else(|| Error::InvalidFormat("free-space section offset overflow".into()))?;
            let section = data.get(pos..end).ok_or_else(|| {
                Error::InvalidFormat("free-space section info is truncated".into())
            })?;
            sections.push(FreeSpaceSection::deserialize(section)?);
            pos = end;
        }
        Self::open(sections)
    }

    pub fn cache_sinfo_image_len(&self) -> Result<usize> {
        let mut len = 4usize;
        for section in self.sections.values() {
            len = len.checked_add(section.serialize_size()).ok_or_else(|| {
                Error::InvalidFormat("free-space section info image length overflow".into())
            })?;
        }
        Ok(len)
    }

    pub fn cache_sinfo_pre_serialize(&mut self) {
        self.dirty = false;
    }

    pub fn cache_sinfo_serialize(&self) -> Result<Vec<u8>> {
        let mut out = Vec::with_capacity(self.cache_sinfo_image_len()?);
        for section in self.sections.values() {
            section.serialize(&mut out);
        }
        let checksum = crate::format::checksum::checksum_metadata(&out);
        out.extend_from_slice(&checksum.to_le_bytes());
        Ok(out)
    }

    pub fn cache_sinfo_notify(&self) {}

    pub fn cache_sinfo_free_icr(self) {}

    pub fn sinfo_serialize_sect_cb(section: &FreeSpaceSection, out: &mut Vec<u8>) {
        section.serialize(out);
    }

    pub fn sinfo_serialize_node_cb(sections: &[FreeSpaceSection], out: &mut Vec<u8>) {
        for section in sections {
            section.serialize(out);
        }
    }

    pub fn sect_init_cls() -> Vec<FreeSpaceClass> {
        vec![
            FreeSpaceClass::Simple,
            FreeSpaceClass::Small,
            FreeSpaceClass::Large,
        ]
    }

    pub fn sect_term_cls(_classes: Vec<FreeSpaceClass>) {}

    pub fn sect_node_new(section: FreeSpaceSection) -> FreeSpaceSection {
        section
    }

    pub fn stat_info(&self) -> FreeSpaceStats {
        let mut stats = FreeSpaceStats::default();
        stats.section_count = self.sections.len();
        for section in self.sections.values() {
            stats.total_space = stats.total_space.saturating_add(section.size);
            stats.largest_section = stats.largest_section.max(section.size);
        }
        stats
    }

    pub fn debug(&self) -> String {
        let stats = self.stat_info();
        format!(
            "FreeSpaceManager(sections={}, total_space={}, largest_section={})",
            stats.section_count, stats.total_space, stats.largest_section
        )
    }

    pub fn sect_debug(section: &FreeSpaceSection) -> String {
        format!(
            "FreeSpaceSection(addr={:#x}, size={}, class={:?})",
            section.addr, section.size, section.class
        )
    }

    pub fn sects_debug(&self) -> Vec<String> {
        self.sections.values().map(Self::sect_debug).collect()
    }

    pub fn sinfo_new() -> Self {
        Self::new()
    }

    pub fn sinfo_lock(&mut self) {
        self.locked = true;
    }

    pub fn sinfo_unlock(&mut self) {
        self.locked = false;
    }

    pub fn sect_serialize_size(section: &FreeSpaceSection) -> usize {
        section.serialize_size()
    }

    pub fn sect_increase(section: &mut FreeSpaceSection, amount: u64) -> Result<()> {
        section.size = section
            .size
            .checked_add(amount)
            .ok_or_else(|| Error::InvalidFormat("free-space section size overflow".into()))?;
        Ok(())
    }

    pub fn sect_decrease(section: &mut FreeSpaceSection, amount: u64) -> Result<()> {
        if amount > section.size {
            return Err(Error::InvalidFormat(
                "free-space section decrease underflow".into(),
            ));
        }
        section.size -= amount;
        Ok(())
    }

    pub fn size_node_decr(&mut self, addr: u64) -> Result<()> {
        self.sections.remove(&addr).ok_or_else(|| {
            Error::InvalidFormat(format!("free-space section {addr:#x} not found"))
        })?;
        Ok(())
    }

    pub fn sect_unlink_size(&mut self, addr: u64) -> Result<FreeSpaceSection> {
        self.sect_remove(addr)
    }

    pub fn sect_unlink_rest(&mut self, addr: u64) -> Result<FreeSpaceSection> {
        self.sect_remove(addr)
    }

    pub fn sect_remove_real(&mut self, addr: u64) -> Result<FreeSpaceSection> {
        self.sect_remove(addr)
    }

    pub fn sect_remove(&mut self, addr: u64) -> Result<FreeSpaceSection> {
        self.dirty = true;
        self.sections
            .remove(&addr)
            .ok_or_else(|| Error::InvalidFormat(format!("free-space section {addr:#x} not found")))
    }

    pub fn sect_link_size(&mut self, section: FreeSpaceSection) -> Result<()> {
        self.sect_add(section)
    }

    pub fn sect_link_rest(&mut self, section: FreeSpaceSection) -> Result<()> {
        self.sect_add(section)
    }

    pub fn sect_link(&mut self, section: FreeSpaceSection) -> Result<()> {
        self.sect_add(section)
    }

    pub fn sect_merge(lhs: &mut FreeSpaceSection, rhs: FreeSpaceSection) -> Result<()> {
        lhs.merge(rhs)
    }

    pub fn sect_add(&mut self, mut section: FreeSpaceSection) -> Result<()> {
        section = self.sect_try_merge(section)?;
        self.sections.insert(section.addr, section);
        self.dirty = true;
        Ok(())
    }

    pub fn sect_try_extend(&mut self, addr: u64, amount: u64) -> Result<bool> {
        if let Some(section) = self.sections.get_mut(&addr) {
            Self::sect_increase(section, amount)?;
            self.dirty = true;
            Ok(true)
        } else {
            Ok(false)
        }
    }

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

    pub fn sect_find_node(&self, size: u64) -> Option<&FreeSpaceSection> {
        self.sections.values().find(|section| section.size >= size)
    }

    pub fn sect_find(&mut self, size: u64) -> Result<Option<FreeSpaceSection>> {
        let Some((&addr, section)) = self
            .sections
            .iter()
            .find(|(_, section)| section.size >= size)
        else {
            return Ok(None);
        };
        let mut section = section.clone();
        if section.size == size {
            self.sections.remove(&addr);
            self.dirty = true;
            Ok(Some(section))
        } else {
            let allocated = section.split(size)?;
            self.sections.remove(&addr);
            self.sections.insert(section.addr, section);
            self.dirty = true;
            Ok(Some(allocated))
        }
    }

    pub fn iterate_sect_cb<F: FnMut(&FreeSpaceSection)>(&self, mut f: F) {
        for section in self.sections.values() {
            f(section);
        }
    }

    pub fn iterate_node_cb<F: FnMut(&FreeSpaceSection)>(&self, f: F) {
        self.iterate_sect_cb(f);
    }

    pub fn sect_iterate<F: FnMut(&FreeSpaceSection)>(&self, f: F) {
        self.iterate_sect_cb(f);
    }

    pub fn sect_stats(&self) -> FreeSpaceStats {
        self.stat_info()
    }

    pub fn sect_change_class(section: &mut FreeSpaceSection, class: FreeSpaceClass) {
        section.class = class;
    }

    pub fn sect_assert(section: &FreeSpaceSection) -> Result<()> {
        if section.valid() {
            Ok(())
        } else {
            Err(Error::InvalidFormat("invalid free-space section".into()))
        }
    }

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

    pub fn get_cparam_test(&self) -> FreeSpaceCreateParams {
        self.params.clone()
    }

    pub fn cmp_cparam_test(lhs: &FreeSpaceCreateParams, rhs: &FreeSpaceCreateParams) -> bool {
        lhs == rhs
    }

    pub fn create_flush_depend(&mut self) {
        self.flush_dependencies = self.flush_dependencies.saturating_add(1);
    }

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
        let image = fs.cache_sinfo_serialize().unwrap();
        let decoded = FreeSpaceManager::cache_sinfo_deserialize(&image).unwrap();
        assert_eq!(decoded.get_sect_count(), 1);
    }

    #[test]
    fn free_space_header_roundtrips_and_rejects_malformed_images() {
        let mut fs = FreeSpaceManager::new();
        fs.params.alignment = 8;
        fs.params.threshold = 4096;

        let image = fs.cache_hdr_serialize().unwrap();
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
        let err = fs.cache_hdr_serialize().unwrap_err();
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
        let mut image = fs.cache_sinfo_serialize().unwrap();
        *image.last_mut().unwrap() ^= 0x80;

        let err = FreeSpaceManager::cache_sinfo_deserialize(&image).unwrap_err();
        assert!(matches!(err, Error::InvalidFormat(_)));

        let mut empty_image = FreeSpaceManager::new().cache_sinfo_serialize().unwrap();
        assert_eq!(empty_image.len(), 4);
        *empty_image.last_mut().unwrap() ^= 0x80;
        let err = FreeSpaceManager::cache_sinfo_deserialize(&empty_image).unwrap_err();
        assert!(matches!(err, Error::InvalidFormat(_)));
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
