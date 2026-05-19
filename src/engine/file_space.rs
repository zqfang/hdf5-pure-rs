use std::fmt::{self, Write};

use crate::engine::allocator::FileAllocator;
use crate::engine::free_space::{
    FreeSpaceClass, FreeSpaceManager, FreeSpaceSection, FreeSpaceStats,
};
use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileSpaceType {
    RawData,
    MetaData,
    Page,
    Temporary,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileSpaceAggregator {
    pub addr: u64,
    pub size: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FileSpaceMergeFlags {
    pub simple: bool,
    pub small: bool,
    pub large: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileSpaceManager {
    allocator: FileAllocator,
    free_space: FreeSpaceManager,
    raw_aggr: FileSpaceAggregator,
    meta_aggr: FileSpaceAggregator,
    page_size: u64,
    closed: bool,
}

impl FileSpaceAggregator {
    pub fn new(addr: u64, size: u64) -> Self {
        Self { addr, size }
    }

    pub fn end(&self) -> Result<u64> {
        self.addr
            .checked_add(self.size)
            .ok_or_else(|| Error::InvalidFormat("file-space aggregator end overflow".into()))
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
}

impl FileSpaceManager {
    pub fn new(eoa: u64) -> Self {
        Self {
            allocator: FileAllocator::new(eoa),
            free_space: FreeSpaceManager::new(),
            raw_aggr: FileSpaceAggregator::new(0, 0),
            meta_aggr: FileSpaceAggregator::new(0, 0),
            page_size: 4096,
            closed: false,
        }
    }

    pub fn aggr_vfd_alloc(&mut self, size: u64, align: u64) -> Result<FileSpaceAggregator> {
        let addr = self.alloc_from_vfd(size, align)?;
        Ok(FileSpaceAggregator::new(addr, size))
    }

    pub fn aggr_try_extend(aggr: &mut FileSpaceAggregator, addr: u64, size: u64) -> Result<bool> {
        if size == 0 {
            return Ok(true);
        }
        if aggr.end()? == addr {
            aggr.size = aggr.size.checked_add(size).ok_or_else(|| {
                Error::InvalidFormat("file-space aggregator size overflow".into())
            })?;
            return Ok(true);
        }
        Ok(false)
    }

    pub fn aggr_can_absorb(aggr: &FileSpaceAggregator, section: &FreeSpaceSection) -> bool {
        !aggr.is_empty() && aggr.end().ok() == Some(section.addr)
    }

    pub fn aggr_absorb(aggr: &mut FileSpaceAggregator, section: FreeSpaceSection) -> Result<bool> {
        if !Self::aggr_can_absorb(aggr, &section) {
            return Ok(false);
        }
        aggr.size = aggr
            .size
            .checked_add(section.size)
            .ok_or_else(|| Error::InvalidFormat("file-space aggregator absorb overflow".into()))?;
        Ok(true)
    }

    pub fn aggr_query(aggr: &FileSpaceAggregator) -> (u64, u64) {
        (aggr.addr, aggr.size)
    }

    pub fn aggr_reset(aggr: &mut FileSpaceAggregator) {
        aggr.addr = 0;
        aggr.size = 0;
    }

    pub fn free_aggrs(&mut self) -> Result<()> {
        Self::aggr_free(
            &mut self.free_space,
            &mut self.raw_aggr,
            FreeSpaceClass::Large,
        )?;
        Self::aggr_free(
            &mut self.free_space,
            &mut self.meta_aggr,
            FreeSpaceClass::Small,
        )
    }

    pub fn aggr_can_shrink_eoa(aggr: &FileSpaceAggregator, eoa: u64) -> bool {
        !aggr.is_empty() && aggr.end().ok() == Some(eoa)
    }

    pub fn aggr_free(
        free_space: &mut FreeSpaceManager,
        aggr: &mut FileSpaceAggregator,
        class: FreeSpaceClass,
    ) -> Result<()> {
        if aggr.is_empty() {
            return Ok(());
        }
        free_space.sect_add(FreeSpaceSection::new(aggr.addr, aggr.size, class)?)?;
        Self::aggr_reset(aggr);
        Ok(())
    }

    pub fn aggrs_try_shrink_eoa(&mut self, eoa: u64) -> Result<Option<FileSpaceAggregator>> {
        if Self::aggr_can_shrink_eoa(&self.raw_aggr, eoa) {
            let aggr = self.raw_aggr.clone();
            Self::aggr_reset(&mut self.raw_aggr);
            return Ok(Some(aggr));
        }
        if Self::aggr_can_shrink_eoa(&self.meta_aggr, eoa) {
            let aggr = self.meta_aggr.clone();
            Self::aggr_reset(&mut self.meta_aggr);
            return Ok(Some(aggr));
        }
        Ok(None)
    }

    pub fn sect_new(addr: u64, size: u64, class: FreeSpaceClass) -> Result<FreeSpaceSection> {
        FreeSpaceSection::new(addr, size, class)
    }

    pub fn sect_free(_section: FreeSpaceSection) {}

    pub fn sect_deserialize(data: &[u8]) -> Result<FreeSpaceSection> {
        FreeSpaceSection::deserialize(data)
    }

    pub fn sect_valid(section: &FreeSpaceSection) -> bool {
        section.valid()
    }

    pub fn sect_simple_can_merge(lhs: &FreeSpaceSection, rhs: &FreeSpaceSection) -> bool {
        lhs.class == FreeSpaceClass::Simple && lhs.can_merge(rhs)
    }

    pub fn sect_simple_merge(lhs: &mut FreeSpaceSection, rhs: FreeSpaceSection) -> Result<()> {
        if lhs.class != FreeSpaceClass::Simple || rhs.class != FreeSpaceClass::Simple {
            return Err(Error::InvalidFormat("simple section class mismatch".into()));
        }
        lhs.merge(rhs)
    }

    pub fn sect_simple_can_shrink(section: &FreeSpaceSection, eoa: u64) -> bool {
        section.class == FreeSpaceClass::Simple && section.can_shrink(eoa)
    }

    pub fn sect_small_add(&mut self, addr: u64, size: u64) -> Result<()> {
        self.add_sect(FreeSpaceSection::new(addr, size, FreeSpaceClass::Small)?)
    }

    pub fn sect_small_can_merge(lhs: &FreeSpaceSection, rhs: &FreeSpaceSection) -> bool {
        lhs.class == FreeSpaceClass::Small && lhs.can_merge(rhs)
    }

    pub fn sect_small_merge(lhs: &mut FreeSpaceSection, rhs: FreeSpaceSection) -> Result<()> {
        if lhs.class != FreeSpaceClass::Small || rhs.class != FreeSpaceClass::Small {
            return Err(Error::InvalidFormat("small section class mismatch".into()));
        }
        lhs.merge(rhs)
    }

    pub fn sect_large_can_merge(lhs: &FreeSpaceSection, rhs: &FreeSpaceSection) -> bool {
        lhs.class == FreeSpaceClass::Large && lhs.can_merge(rhs)
    }

    pub fn sect_large_merge(lhs: &mut FreeSpaceSection, rhs: FreeSpaceSection) -> Result<()> {
        if lhs.class != FreeSpaceClass::Large || rhs.class != FreeSpaceClass::Large {
            return Err(Error::InvalidFormat("large section class mismatch".into()));
        }
        lhs.merge(rhs)
    }

    pub fn sect_large_can_shrink(section: &FreeSpaceSection, eoa: u64) -> bool {
        section.class == FreeSpaceClass::Large && section.can_shrink(eoa)
    }

    pub fn sect_large_shrink(section: FreeSpaceSection, eoa: u64) -> Result<u64> {
        if !Self::sect_large_can_shrink(&section, eoa) {
            return Err(Error::InvalidFormat(
                "large section does not reach EOA".into(),
            ));
        }
        Ok(section.addr)
    }

    pub fn init_merge_flags() -> FileSpaceMergeFlags {
        FileSpaceMergeFlags {
            simple: true,
            small: true,
            large: true,
        }
    }

    pub fn alloc_to_fs_type(raw: bool, temporary: bool) -> FileSpaceType {
        if temporary {
            FileSpaceType::Temporary
        } else if raw {
            FileSpaceType::RawData
        } else {
            FileSpaceType::MetaData
        }
    }

    pub fn open_fstype_from_iter<I>(sections: I, eoa: u64) -> Result<Self>
    where
        I: IntoIterator<Item = FreeSpaceSection>,
    {
        let mut manager = Self::new(eoa);
        manager.free_space = FreeSpaceManager::open_from_iter(sections)?;
        Ok(manager)
    }

    #[deprecated(note = "use open_fstype_from_iter to avoid requiring Vec storage")]
    pub fn open_fstype(sections: Vec<FreeSpaceSection>, eoa: u64) -> Result<Self> {
        Self::open_fstype_from_iter(sections, eoa)
    }

    pub fn create_fstype(eoa: u64) -> Self {
        Self::new(eoa)
    }

    pub fn start_fstype(&mut self) {
        self.closed = false;
    }

    pub fn delete_fstype(&mut self) {
        self.free_space.delete();
    }

    pub fn close_fstype(&mut self) -> Result<()> {
        self.free_aggrs()?;
        self.closed = true;
        Ok(())
    }

    pub fn add_sect(&mut self, section: FreeSpaceSection) -> Result<()> {
        self.free_space.sect_add(section)
    }

    pub fn find_sect(&mut self, size: u64) -> Result<Option<FreeSpaceSection>> {
        self.free_space.sect_find(size)
    }

    pub fn alloc(&mut self, ty: FileSpaceType, size: u64, align: u64) -> Result<u64> {
        if self.closed {
            return Err(Error::InvalidFormat("file-space manager is closed".into()));
        }
        if size == 0 {
            return Err(Error::InvalidFormat(
                "zero-byte file-space allocation".into(),
            ));
        }
        if ty != FileSpaceType::Temporary {
            if let Some(section) = self.find_typed_sect(ty, size)? {
                return Ok(section.addr);
            }
        }
        match ty {
            FileSpaceType::Page => self.alloc_pagefs(size),
            FileSpaceType::Temporary => self.alloc_tmp(size),
            FileSpaceType::RawData | FileSpaceType::MetaData => self.alloc_from_vfd(size, align),
        }
    }

    pub fn alloc_pagefs(&mut self, size: u64) -> Result<u64> {
        let page_size = self.page_size.max(1);
        self.alloc_from_vfd(size, page_size)
    }

    pub fn alloc_tmp(&mut self, size: u64) -> Result<u64> {
        self.alloc_from_vfd(size, 1)
    }

    pub fn xfree(&mut self, ty: FileSpaceType, addr: u64, size: u64) -> Result<()> {
        let class = Self::class_for_type(ty);
        self.add_sect(FreeSpaceSection::new(addr, size, class)?)
    }

    pub fn try_extend(&mut self, addr: u64, size: u64, extra: u64) -> Result<bool> {
        if extra == 0 {
            return Ok(true);
        }
        if self.free_space.sect_try_extend(addr + size, extra)? {
            return Ok(true);
        }
        if self.allocator.eof() == addr + size {
            let _ = self.alloc_from_vfd(extra, 1)?;
            return Ok(true);
        }
        Ok(false)
    }

    pub fn try_shrink(&mut self, eoa: u64) -> Result<Option<FreeSpaceSection>> {
        self.free_space.sect_try_shrink_eoa(eoa)
    }

    pub fn close(&mut self) -> Result<()> {
        self.close_fstype()
    }

    pub fn close_delete_fstype(&mut self) {
        self.delete_fstype();
        self.closed = true;
    }

    pub fn try_close(&mut self) -> Result<bool> {
        if self.free_space.get_sect_count() == 0
            && self.raw_aggr.is_empty()
            && self.meta_aggr.is_empty()
        {
            self.closed = true;
            return Ok(true);
        }
        Ok(false)
    }

    pub fn close_aggrfs(&mut self) -> Result<()> {
        self.free_aggrs()
    }

    pub fn close_pagefs(&mut self) -> Result<()> {
        self.free_space.assert_valid()
    }

    pub fn close_shrink_eoa(&mut self, eoa: u64) -> Result<Option<FreeSpaceSection>> {
        self.try_shrink(eoa)
    }

    pub fn get_freespace(&self) -> FreeSpaceStats {
        self.free_space.stat_info()
    }

    #[deprecated(note = "use free_sections or sects_cb to borrow sections without cloning")]
    pub fn get_free_sections(&self) -> Vec<FreeSpaceSection> {
        self.free_sections().cloned().collect()
    }

    pub fn free_sections(&self) -> impl Iterator<Item = &FreeSpaceSection> {
        self.free_space.sections()
    }

    pub fn sects_cb<F: FnMut(&FreeSpaceSection)>(&self, f: F) {
        self.free_space.iterate_sect_cb(f);
    }

    #[deprecated(note = "use free_sections or sects_cb to borrow sections without cloning")]
    pub fn get_free_sects(&self) -> Vec<FreeSpaceSection> {
        self.free_sections().cloned().collect()
    }

    pub fn settle_raw_data_fsm(&mut self) -> Result<()> {
        Self::aggr_free(
            &mut self.free_space,
            &mut self.raw_aggr,
            FreeSpaceClass::Large,
        )
    }

    pub fn settle_meta_data_fsm(&mut self) -> Result<()> {
        Self::aggr_free(
            &mut self.free_space,
            &mut self.meta_aggr,
            FreeSpaceClass::Small,
        )
    }

    pub fn continue_alloc_fsm(&mut self, ty: FileSpaceType, size: u64) -> Result<u64> {
        self.alloc(ty, size, 1)
    }

    pub fn fsm_type_is_self_referential(ty: FileSpaceType) -> bool {
        matches!(ty, FileSpaceType::MetaData | FileSpaceType::Page)
    }

    pub fn fsm_is_self_referential(&self) -> bool {
        self.free_space.get_sect_count() != 0
            && (Self::fsm_type_is_self_referential(FileSpaceType::MetaData)
                || !self.meta_aggr.is_empty())
    }

    pub fn write_sects_debug_cb<W: Write>(section: &FreeSpaceSection, out: &mut W) -> fmt::Result {
        FreeSpaceManager::write_sect_debug(section, out)
    }

    pub fn write_sects_debug<W: Write>(&self, out: &mut W) -> fmt::Result {
        self.free_space.write_sects_debug(out)
    }

    pub fn write_sects_dump<W: Write>(&self, out: &mut W) -> fmt::Result {
        self.write_sects_debug(out)
    }

    fn alloc_from_vfd(&mut self, size: u64, align: u64) -> Result<u64> {
        if size == 0 {
            return Err(Error::InvalidFormat("zero-byte VFD allocation".into()));
        }
        Ok(self.allocator.allocate(size, align))
    }

    fn class_for_type(ty: FileSpaceType) -> FreeSpaceClass {
        match ty {
            FileSpaceType::RawData => FreeSpaceClass::Large,
            FileSpaceType::MetaData => FreeSpaceClass::Small,
            FileSpaceType::Page | FileSpaceType::Temporary => FreeSpaceClass::Simple,
        }
    }

    fn find_typed_sect(
        &mut self,
        ty: FileSpaceType,
        size: u64,
    ) -> Result<Option<FreeSpaceSection>> {
        let class = Self::class_for_type(ty);
        self.free_space.sect_find_by_class(class, size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_space_reuses_freed_sections_before_vfd_growth() {
        let mut manager = FileSpaceManager::new(128);
        manager.xfree(FileSpaceType::RawData, 32, 16).unwrap();
        let addr = manager.alloc(FileSpaceType::RawData, 8, 1).unwrap();
        assert_eq!(addr, 32);
        assert_eq!(manager.get_freespace().total_space, 8);
        assert_eq!(manager.alloc(FileSpaceType::MetaData, 4, 1).unwrap(), 128);
    }

    #[test]
    fn aggregators_absorb_and_settle_to_free_space() {
        let mut manager = FileSpaceManager::new(256);
        manager.raw_aggr = FileSpaceAggregator::new(64, 16);
        let section = FreeSpaceSection::new(80, 8, FreeSpaceClass::Large).unwrap();
        assert!(FileSpaceManager::aggr_absorb(&mut manager.raw_aggr, section).unwrap());
        assert_eq!(manager.raw_aggr.size, 24);
        manager.settle_raw_data_fsm().unwrap();
        assert!(manager.raw_aggr.is_empty());
        assert_eq!(manager.get_freespace().total_space, 24);
    }

    #[test]
    fn open_fstype_accepts_section_iterators_and_visits_sections() {
        let sections = [
            FreeSpaceSection::new(8, 4, FreeSpaceClass::Simple).unwrap(),
            FreeSpaceSection::new(32, 16, FreeSpaceClass::Large).unwrap(),
        ];
        let manager =
            FileSpaceManager::open_fstype_from_iter(sections.iter().cloned(), 128).unwrap();

        assert_eq!(manager.get_freespace().section_count, 2);
        assert_eq!(manager.free_sections().count(), 2);

        let mut total = 0;
        manager.sects_cb(|section| total += section.size);
        assert_eq!(total, 20);
    }
}
