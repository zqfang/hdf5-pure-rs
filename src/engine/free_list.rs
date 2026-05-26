use crate::error::{Error, Result};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FreeListStats {
    pub regular_bytes: usize,
    pub block_bytes: usize,
    pub array_bytes: usize,
    pub factory_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FreeListManager {
    regular: Vec<Vec<u8>>,
    blocks: Vec<Vec<u8>>,
    arrays: Vec<Vec<u8>>,
    factories: Vec<Vec<u8>>,
    max_regular: usize,
    max_block: usize,
    max_array: usize,
    max_factory: usize,
}

impl Default for FreeListManager {
    fn default() -> Self {
        Self::new()
    }
}

impl FreeListManager {
    pub fn new() -> Self {
        Self {
            regular: Vec::new(),
            blocks: Vec::new(),
            arrays: Vec::new(),
            factories: Vec::new(),
            max_regular: usize::MAX,
            max_block: usize::MAX,
            max_array: usize::MAX,
            max_factory: usize::MAX,
        }
    }

    pub fn term_package(&mut self) {
        self.garbage_coll();
    }

    fn malloc_vec(size: usize) -> Result<Vec<u8>> {
        if size == 0 {
            return Err(Error::InvalidFormat(
                "free-list allocation size is zero".into(),
            ));
        }
        let mut buf = Vec::new();
        buf.try_reserve_exact(size)
            .map_err(|err| Error::InvalidFormat(format!("free-list allocation failed: {err}")))?;
        buf.resize(size, 0);
        Ok(buf)
    }

    fn checked_array_size(count: usize, elem_size: usize) -> Result<usize> {
        count
            .checked_mul(elem_size)
            .ok_or_else(|| Error::InvalidFormat("free-list array size overflow".into()))
    }

    fn resize_allocation_in_place(buf: &mut Vec<u8>, new_size: usize) -> Result<()> {
        if new_size == 0 {
            return Err(Error::InvalidFormat(
                "free-list allocation size is zero".into(),
            ));
        }
        if buf.capacity() < new_size {
            buf.try_reserve_exact(new_size - buf.len()).map_err(|err| {
                Error::InvalidFormat(format!("free-list allocation failed: {err}"))
            })?;
        }
        buf.resize(new_size, 0);
        Ok(())
    }

    fn take_reusable_buffer(list: &mut Vec<Vec<u8>>, size: usize) -> Option<Vec<u8>> {
        let pos = list.iter().position(|buf| buf.len() >= size)?;
        Some(list.remove(pos))
    }

    pub fn malloc_into(size: usize, out: &mut Vec<u8>) -> Result<()> {
        if size == 0 {
            return Err(Error::InvalidFormat(
                "free-list allocation size is zero".into(),
            ));
        }
        if out.capacity() < size {
            out.try_reserve_exact(size - out.len()).map_err(|err| {
                Error::InvalidFormat(format!("free-list allocation failed: {err}"))
            })?;
        }
        out.clear();
        out.resize(size, 0);
        Ok(())
    }

    fn reg_malloc_vec(&mut self, size: usize) -> Result<Vec<u8>> {
        Self::take_reusable_buffer(&mut self.regular, size).map_or_else(
            || Self::malloc_vec(size),
            |mut buf| {
                buf.resize(size, 0);
                Ok(buf)
            },
        )
    }

    pub fn reg_malloc_into(&mut self, size: usize, out: &mut Vec<u8>) -> Result<()> {
        if out.capacity() >= size {
            Self::malloc_into(size, out)
        } else {
            *out = self.reg_malloc_vec(size)?;
            Ok(())
        }
    }

    pub fn reg_calloc_into(&mut self, size: usize, out: &mut Vec<u8>) -> Result<()> {
        self.reg_malloc_into(size, out)?;
        out.fill(0);
        Ok(())
    }

    pub fn reg_gc_list(&mut self) {
        self.regular.clear();
    }

    pub fn reg_gc(&mut self) {
        self.reg_gc_list();
    }

    pub fn reg_term(&mut self) {
        self.reg_gc();
    }

    fn blk_find_list_vec(&mut self, size: usize) -> Option<Vec<u8>> {
        Self::take_reusable_buffer(&mut self.blocks, size)
    }

    pub fn blk_find_list_into(&mut self, size: usize, out: &mut Vec<u8>) -> bool {
        match self.blk_find_list_vec(size) {
            Some(buf) => {
                *out = buf;
                true
            }
            None => false,
        }
    }

    pub fn blk_create_list(&mut self) {
        self.blocks.clear();
    }

    pub fn blk_init(&mut self) {
        self.blk_create_list();
    }

    pub fn blk_free_block_avail(&self, size: usize) -> bool {
        self.blocks.iter().any(|buf| buf.len() >= size)
    }

    fn blk_malloc_vec(&mut self, size: usize) -> Result<Vec<u8>> {
        self.blk_find_list_vec(size).map_or_else(
            || Self::malloc_vec(size),
            |mut buf| {
                buf.resize(size, 0);
                Ok(buf)
            },
        )
    }

    pub fn blk_malloc_into(&mut self, size: usize, out: &mut Vec<u8>) -> Result<()> {
        if out.capacity() >= size {
            Self::malloc_into(size, out)
        } else {
            *out = self.blk_malloc_vec(size)?;
            Ok(())
        }
    }

    pub fn blk_calloc_into(&mut self, size: usize, out: &mut Vec<u8>) -> Result<()> {
        self.blk_malloc_into(size, out)?;
        out.fill(0);
        Ok(())
    }

    pub fn blk_free(&mut self, mut buf: Vec<u8>) {
        if self.blocks.len() < self.max_block {
            buf.fill(0);
            self.blocks.push(buf);
        }
    }

    pub fn blk_realloc_in_place(&mut self, buf: &mut Vec<u8>, new_size: usize) -> Result<()> {
        Self::resize_allocation_in_place(buf, new_size)
    }

    pub fn blk_gc_list(&mut self) {
        self.blocks.clear();
    }

    pub fn blk_gc(&mut self) {
        self.blk_gc_list();
    }

    pub fn blk_term(&mut self) {
        self.blk_gc();
    }

    pub fn arr_init(&mut self) {
        self.arrays.clear();
    }

    pub fn arr_free(&mut self, mut buf: Vec<u8>) {
        if self.arrays.len() < self.max_array {
            buf.fill(0);
            self.arrays.push(buf);
        }
    }

    pub fn arr_malloc_into(
        &mut self,
        count: usize,
        elem_size: usize,
        out: &mut Vec<u8>,
    ) -> Result<()> {
        let size = Self::checked_array_size(count, elem_size)?;
        if out.capacity() >= size {
            Self::malloc_into(size, out)
        } else {
            *out = Self::take_reusable_buffer(&mut self.arrays, size).map_or_else(
                || Self::malloc_vec(size),
                |mut buf| {
                    buf.resize(size, 0);
                    Ok(buf)
                },
            )?;
            Ok(())
        }
    }

    pub fn arr_calloc_into(
        &mut self,
        count: usize,
        elem_size: usize,
        out: &mut Vec<u8>,
    ) -> Result<()> {
        self.arr_malloc_into(count, elem_size, out)
    }

    pub fn arr_realloc_in_place(
        &mut self,
        buf: &mut Vec<u8>,
        count: usize,
        elem_size: usize,
    ) -> Result<()> {
        let size = Self::checked_array_size(count, elem_size)?;
        Self::resize_allocation_in_place(buf, size)
    }

    pub fn arr_gc_list(&mut self) {
        self.arrays.clear();
    }

    pub fn arr_gc(&mut self) {
        self.arr_gc_list();
    }

    pub fn arr_term(&mut self) {
        self.arr_gc();
    }

    pub fn seq_free(&mut self, buf: Vec<u8>) {
        self.arr_free(buf);
    }

    pub fn seq_malloc_into(
        &mut self,
        count: usize,
        elem_size: usize,
        out: &mut Vec<u8>,
    ) -> Result<()> {
        self.arr_malloc_into(count, elem_size, out)
    }

    pub fn seq_calloc_into(
        &mut self,
        count: usize,
        elem_size: usize,
        out: &mut Vec<u8>,
    ) -> Result<()> {
        self.arr_calloc_into(count, elem_size, out)
    }

    pub fn seq_realloc_in_place(
        &mut self,
        buf: &mut Vec<u8>,
        count: usize,
        elem_size: usize,
    ) -> Result<()> {
        self.arr_realloc_in_place(buf, count, elem_size)
    }

    pub fn fac_init(&mut self) {
        self.factories.clear();
    }

    pub fn fac_free(&mut self, mut buf: Vec<u8>) {
        if self.factories.len() < self.max_factory {
            buf.fill(0);
            self.factories.push(buf);
        }
    }

    fn fac_malloc_vec(&mut self, size: usize) -> Result<Vec<u8>> {
        Self::take_reusable_buffer(&mut self.factories, size).map_or_else(
            || Self::malloc_vec(size),
            |mut buf| {
                buf.resize(size, 0);
                Ok(buf)
            },
        )
    }

    pub fn fac_malloc_into(&mut self, size: usize, out: &mut Vec<u8>) -> Result<()> {
        if out.capacity() >= size {
            Self::malloc_into(size, out)
        } else {
            *out = self.fac_malloc_vec(size)?;
            Ok(())
        }
    }

    pub fn fac_calloc_into(&mut self, size: usize, out: &mut Vec<u8>) -> Result<()> {
        self.fac_malloc_into(size, out)?;
        out.fill(0);
        Ok(())
    }

    pub fn fac_gc_list(&mut self) {
        self.factories.clear();
    }

    pub fn fac_gc(&mut self) {
        self.fac_gc_list();
    }

    pub fn fac_term(&mut self) {
        self.fac_gc();
    }

    pub fn fac_term_all(&mut self) {
        self.fac_term();
    }

    pub fn garbage_coll(&mut self) {
        self.reg_gc();
        self.blk_gc();
        self.arr_gc();
        self.fac_gc();
    }

    pub fn set_free_list_limits(
        &mut self,
        regular: usize,
        block: usize,
        array: usize,
        factory: usize,
    ) {
        self.max_regular = regular;
        self.max_block = block;
        self.max_array = array;
        self.max_factory = factory;
        self.regular.truncate(regular);
        self.blocks.truncate(block);
        self.arrays.truncate(array);
        self.factories.truncate(factory);
    }

    pub fn get_free_list_sizes(&self) -> FreeListStats {
        FreeListStats {
            regular_bytes: self.regular.iter().map(Vec::len).sum(),
            block_bytes: self.blocks.iter().map(Vec::len).sum(),
            array_bytes: self.arrays.iter().map(Vec::len).sum(),
            factory_bytes: self.factories.iter().map(Vec::len).sum(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_invalid_format(result: Result<()>) {
        assert!(matches!(result, Err(Error::InvalidFormat(_))));
    }

    #[test]
    fn block_free_list_reuses_buffers() {
        let mut lists = FreeListManager::new();
        let mut buf = Vec::new();
        lists.blk_malloc_into(8, &mut buf).unwrap();
        lists.blk_free(buf);
        assert!(lists.blk_free_block_avail(8));
        let mut reused = Vec::new();
        lists.blk_malloc_into(4, &mut reused).unwrap();
        assert_eq!(reused.len(), 4);
    }

    #[test]
    fn array_free_list_reuses_buffers() {
        let mut lists = FreeListManager::new();
        let buf = vec![7; 16];
        let original_capacity = buf.capacity();
        lists.arr_free(buf);

        let mut reused = Vec::new();
        lists.arr_malloc_into(4, 2, &mut reused).unwrap();

        assert_eq!(reused.len(), 8);
        assert!(reused.capacity() >= original_capacity);
        assert_eq!(lists.get_free_list_sizes().array_bytes, 0);
    }

    #[test]
    fn malloc_into_replaces_stale_on_success_and_preserves_on_errors() {
        let mut out = b"stale".to_vec();
        FreeListManager::malloc_into(3, &mut out).unwrap();
        assert_eq!(out, vec![0; 3]);

        out = b"stale".to_vec();
        assert_invalid_format(FreeListManager::malloc_into(0, &mut out));
        assert_eq!(out, b"stale");

        assert_invalid_format(FreeListManager::malloc_into(usize::MAX, &mut out));
        assert_eq!(out, b"stale");
    }

    #[test]
    fn typed_malloc_into_helpers_replace_stale_on_success_and_preserve_on_errors() {
        let mut lists = FreeListManager::new();

        let mut regular = b"stale".to_vec();
        lists.reg_malloc_into(3, &mut regular).unwrap();
        assert_eq!(regular, vec![0; 3]);
        regular = b"stale".to_vec();
        assert_invalid_format(lists.reg_malloc_into(0, &mut regular));
        assert_eq!(regular, b"stale");
        assert_invalid_format(lists.reg_malloc_into(usize::MAX, &mut regular));
        assert_eq!(regular, b"stale");

        let mut block = b"stale".to_vec();
        lists.blk_malloc_into(3, &mut block).unwrap();
        assert_eq!(block, vec![0; 3]);
        block = b"stale".to_vec();
        assert_invalid_format(lists.blk_malloc_into(0, &mut block));
        assert_eq!(block, b"stale");
        assert_invalid_format(lists.blk_malloc_into(usize::MAX, &mut block));
        assert_eq!(block, b"stale");

        let mut factory = b"stale".to_vec();
        lists.fac_malloc_into(3, &mut factory).unwrap();
        assert_eq!(factory, vec![0; 3]);
        factory = b"stale".to_vec();
        assert_invalid_format(lists.fac_malloc_into(0, &mut factory));
        assert_eq!(factory, b"stale");
        assert_invalid_format(lists.fac_malloc_into(usize::MAX, &mut factory));
        assert_eq!(factory, b"stale");
    }

    #[test]
    fn array_and_sequence_into_helpers_preserve_stale_after_validation_errors() {
        let mut lists = FreeListManager::new();

        let mut array = b"stale".to_vec();
        lists.arr_malloc_into(2, 2, &mut array).unwrap();
        assert_eq!(array, vec![0; 4]);
        array = b"stale".to_vec();
        assert_invalid_format(lists.arr_malloc_into(0, 2, &mut array));
        assert_eq!(array, b"stale");
        assert_invalid_format(lists.arr_malloc_into(usize::MAX, 2, &mut array));
        assert_eq!(array, b"stale");
        assert_invalid_format(lists.arr_malloc_into(usize::MAX, 1, &mut array));
        assert_eq!(array, b"stale");

        let mut sequence = b"stale".to_vec();
        lists.seq_malloc_into(2, 2, &mut sequence).unwrap();
        assert_eq!(sequence, vec![0; 4]);
        sequence = b"stale".to_vec();
        assert_invalid_format(lists.seq_malloc_into(0, 2, &mut sequence));
        assert_eq!(sequence, b"stale");
        assert_invalid_format(lists.seq_malloc_into(usize::MAX, 2, &mut sequence));
        assert_eq!(sequence, b"stale");
        assert_invalid_format(lists.seq_malloc_into(usize::MAX, 1, &mut sequence));
        assert_eq!(sequence, b"stale");
    }

    #[test]
    fn calloc_into_helpers_replace_stale_on_success_and_preserve_on_errors() {
        let mut lists = FreeListManager::new();

        let mut regular = b"stale".to_vec();
        lists.reg_calloc_into(3, &mut regular).unwrap();
        assert_eq!(regular, vec![0; 3]);
        regular = b"stale".to_vec();
        assert_invalid_format(lists.reg_calloc_into(0, &mut regular));
        assert_eq!(regular, b"stale");

        let mut block = b"stale".to_vec();
        lists.blk_calloc_into(3, &mut block).unwrap();
        assert_eq!(block, vec![0; 3]);
        block = b"stale".to_vec();
        assert_invalid_format(lists.blk_calloc_into(0, &mut block));
        assert_eq!(block, b"stale");

        let mut array = b"stale".to_vec();
        lists.arr_calloc_into(2, 2, &mut array).unwrap();
        assert_eq!(array, vec![0; 4]);
        array = b"stale".to_vec();
        assert_invalid_format(lists.arr_calloc_into(usize::MAX, 2, &mut array));
        assert_eq!(array, b"stale");

        let mut sequence = b"stale".to_vec();
        lists.seq_calloc_into(2, 2, &mut sequence).unwrap();
        assert_eq!(sequence, vec![0; 4]);
        sequence = b"stale".to_vec();
        assert_invalid_format(lists.seq_calloc_into(usize::MAX, 2, &mut sequence));
        assert_eq!(sequence, b"stale");

        let mut factory = b"stale".to_vec();
        lists.fac_calloc_into(3, &mut factory).unwrap();
        assert_eq!(factory, vec![0; 3]);
        factory = b"stale".to_vec();
        assert_invalid_format(lists.fac_calloc_into(0, &mut factory));
        assert_eq!(factory, b"stale");
    }

    #[test]
    fn realloc_in_place_helpers_preserve_stale_after_validation_errors() {
        let mut lists = FreeListManager::new();

        let mut block = b"stale".to_vec();
        lists.blk_realloc_in_place(&mut block, 3).unwrap();
        assert_eq!(block, b"sta");
        assert_invalid_format(lists.blk_realloc_in_place(&mut block, 0));
        assert_eq!(block, b"sta");
        assert_invalid_format(lists.blk_realloc_in_place(&mut block, usize::MAX));
        assert_eq!(block, b"sta");

        let mut array = b"stale".to_vec();
        lists.arr_realloc_in_place(&mut array, 2, 2).unwrap();
        assert_eq!(array, b"stal");
        assert_invalid_format(lists.arr_realloc_in_place(&mut array, 0, 2));
        assert_eq!(array, b"stal");
        assert_invalid_format(lists.arr_realloc_in_place(&mut array, usize::MAX, 2));
        assert_eq!(array, b"stal");
        assert_invalid_format(lists.arr_realloc_in_place(&mut array, usize::MAX, 1));
        assert_eq!(array, b"stal");

        let mut sequence = b"stale".to_vec();
        lists.seq_realloc_in_place(&mut sequence, 2, 2).unwrap();
        assert_eq!(sequence, b"stal");
        assert_invalid_format(lists.seq_realloc_in_place(&mut sequence, 0, 2));
        assert_eq!(sequence, b"stal");
        assert_invalid_format(lists.seq_realloc_in_place(&mut sequence, usize::MAX, 2));
        assert_eq!(sequence, b"stal");
        assert_invalid_format(lists.seq_realloc_in_place(&mut sequence, usize::MAX, 1));
        assert_eq!(sequence, b"stal");
    }
}
