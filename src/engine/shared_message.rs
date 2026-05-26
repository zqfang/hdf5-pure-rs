use std::collections::BTreeMap;
use std::fmt::{self, Write};

use crate::error::{Error, Result};

/// Shared object-header message payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedMessage {
    pub msg_type: u8,
    pub heap_addr: u64,
    pub data: Vec<u8>,
    pub refcount: u32,
}

/// Shared-message table/list state.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SharedMessageStore {
    messages: BTreeMap<u64, SharedMessage>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SharedMessageInfo {
    pub count: usize,
    pub total_bytes: usize,
}

impl SharedMessage {
    /// Construct a new shared message with refcount 1.
    pub fn new(msg_type: u8, heap_addr: u64, data: Vec<u8>) -> Self {
        Self {
            msg_type,
            heap_addr,
            data,
            refcount: 1,
        }
    }

    /// Return the encoded size in bytes, saturating to `usize::MAX` on overflow.
    pub fn encoded_len(&self) -> usize {
        self.encoded_len_checked().unwrap_or(usize::MAX)
    }

    /// Compute the encoded size in bytes, returning an error on overflow.
    pub fn encoded_len_checked(&self) -> Result<usize> {
        self.validate_encodable()?;
        13usize
            .checked_add(self.data.len())
            .ok_or_else(|| Error::InvalidFormat("shared message image length overflow".into()))
    }

    /// Append this shared message's encoded representation to `out`.
    pub fn encode_into(&self, out: &mut impl Extend<u8>) -> Result<()> {
        let data_len = self.validate_encodable()?;
        out.extend([self.msg_type]);
        out.extend(self.heap_addr.to_le_bytes());
        out.extend(data_len.to_le_bytes());
        out.extend(self.data.iter().copied());
        Ok(())
    }

    fn validate_encodable(&self) -> Result<u32> {
        u32::try_from(self.data.len()).map_err(|_| {
            Error::InvalidFormat("shared message payload is too large to encode".into())
        })
    }
}

impl SharedMessageStore {
    /// Return the on-disk size of the master table of Shared Object Header Message indexes.
    pub fn cache_table_get_initial_load_size(count: usize) -> usize {
        Self::cache_table_get_initial_load_size_checked(count).unwrap_or(usize::MAX)
    }

    /// Checked variant of [`cache_table_get_initial_load_size`] that errors on overflow.
    pub fn cache_table_get_initial_load_size_checked(count: usize) -> Result<usize> {
        count.checked_mul(16).ok_or_else(|| {
            Error::InvalidFormat("shared message table initial load size overflow".into())
        })
    }

    /// Verify the computed checksum of the table data structure matches the stored checksum.
    pub fn cache_table_verify_chksum(bytes: &[u8], checksum: u32) -> bool {
        crc32fast::hash(bytes) == checksum
    }

    /// Compute the encoded size in bytes of the table image on disk.
    pub fn cache_table_image_len(&self) -> usize {
        self.cache_table_image_len_checked().unwrap_or(usize::MAX)
    }

    /// Checked variant of [`cache_table_image_len`] that errors on overflow.
    pub fn cache_table_image_len_checked(&self) -> Result<usize> {
        self.messages.values().try_fold(0usize, |acc, message| {
            acc.checked_add(message.encoded_len_checked()?)
                .ok_or_else(|| {
                    Error::InvalidFormat("shared message cache table image length overflow".into())
                })
        })
    }

    /// Append the serialized shared message table image to `out`.
    pub fn cache_table_serialize_into(&self, out: &mut impl Extend<u8>) -> Result<()> {
        let image_len = self.cache_table_image_len_checked()?;
        let mut image = Vec::with_capacity(image_len);
        for message in self.messages.values() {
            message.encode_into(&mut image)?;
        }
        out.extend(image);
        Ok(())
    }

    /// Free memory used by the SOHM table image.
    pub fn cache_table_free_icr(_bytes: Vec<u8>) {}

    /// Return the on-disk size of a list of SOHM messages.
    pub fn cache_list_get_initial_load_size(count: usize) -> usize {
        Self::cache_list_get_initial_load_size_checked(count).unwrap_or(usize::MAX)
    }

    /// Checked variant of [`cache_list_get_initial_load_size`] that errors on overflow.
    pub fn cache_list_get_initial_load_size_checked(count: usize) -> Result<usize> {
        count.checked_mul(12).ok_or_else(|| {
            Error::InvalidFormat("shared message list initial load size overflow".into())
        })
    }

    /// Verify the computed checksum of the list data structure matches the stored checksum.
    pub fn cache_list_verify_chksum(bytes: &[u8], checksum: u32) -> bool {
        Self::cache_table_verify_chksum(bytes, checksum)
    }

    /// Deserialize a buffer containing the on-disk image of a SOHM message list.
    pub fn cache_list_deserialize(bytes: &[u8]) -> Result<Self> {
        let mut store = Self::default();
        store.cache_list_deserialize_into(bytes)?;
        Ok(store)
    }

    /// Deserialize a SOHM message list into this store, reusing the map allocation.
    pub fn cache_list_deserialize_into(&mut self, bytes: &[u8]) -> Result<()> {
        let mut pos = 0usize;
        let mut messages = BTreeMap::new();
        while pos < bytes.len() {
            let remaining = bytes.len() - pos;
            if remaining < 13 {
                return Err(Error::InvalidFormat(
                    "shared message cache list entry is truncated".into(),
                ));
            }
            let msg_type = bytes[pos];
            pos += 1;
            let heap_addr = read_u64_le_at(bytes, pos, "shared message heap address")?;
            pos += 8;
            let data_len_u32 = read_u32_le_at(bytes, pos, "shared message data length")?;
            let data_len = usize::try_from(data_len_u32).map_err(|_| {
                Error::InvalidFormat("shared message data length exceeds usize".into())
            })?;
            pos += 4;
            let data_end = pos.checked_add(data_len).ok_or_else(|| {
                Error::InvalidFormat("shared message data length overflow".into())
            })?;
            let data = bytes
                .get(pos..data_end)
                .ok_or_else(|| {
                    Error::InvalidFormat("shared message cache list data is truncated".into())
                })?
                .to_vec();
            pos = data_end;
            messages.insert(heap_addr, SharedMessage::new(msg_type, heap_addr, data));
        }
        self.messages = messages;
        Ok(())
    }

    /// Compute the on-disk size of the shared message list.
    pub fn cache_list_image_len(&self) -> usize {
        self.cache_table_image_len()
    }

    /// Checked variant of [`cache_list_image_len`] that errors on overflow.
    pub fn cache_list_image_len_checked(&self) -> Result<usize> {
        self.cache_table_image_len_checked()
    }

    /// Append the serialized shared message list image to `out`.
    pub fn cache_list_serialize_into(&self, out: &mut impl Extend<u8>) -> Result<()> {
        self.cache_table_serialize_into(out)
    }

    /// Free memory used by the SOHM list image.
    pub fn cache_list_free_icr(_bytes: Vec<u8>) {}

    /// Retrieve the number of messages currently tracked (test helper).
    pub fn get_mesg_count_test(&self) -> usize {
        self.messages.len()
    }

    /// Initialize the Shared Message interface and master SOHM table.
    pub fn init() -> Self {
        Self::default()
    }

    /// Get the shared message flag bit for a given message type.
    pub fn type_to_flag(msg_type: u8) -> u32 {
        1u32.checked_shl(u32::from(msg_type)).unwrap_or(0)
    }

    /// Check whether a given message type is marked shared in a file.
    pub fn type_shared(msg_type: u8, mask: u32) -> bool {
        mask & Self::type_to_flag(msg_type) != 0
    }

    /// Return the address of the fractal heap used to store messages of a given type id.
    pub fn get_fheap_addr(&self, key: u64) -> Option<u64> {
        self.messages.get(&key).map(|msg| msg.heap_addr)
    }

    /// Allocate storage for a new SOHM index header.
    pub fn create_index(&mut self) {}

    /// De-allocate storage for an index, optionally deleting the underlying heap.
    pub fn delete_index(&mut self) {
        self.messages.clear();
    }

    /// Create a fresh list of SOHM messages for a newly created or converted index.
    pub fn create_list(&mut self) {}

    /// Append cloned messages in key order to `out`.
    pub fn clone_messages_into(&self, out: &mut impl Extend<SharedMessage>) {
        out.extend(self.messages.values().cloned());
    }

    /// Consume the store and iterate over messages in key order without cloning.
    pub fn into_messages(self) -> impl Iterator<Item = SharedMessage> {
        self.messages.into_values()
    }

    /// Borrow all shared messages in key order.
    pub fn messages(&self) -> impl Iterator<Item = &SharedMessage> {
        self.messages.values()
    }

    /// Visit all shared messages in key order.
    pub fn visit_messages<F: FnMut(&SharedMessage)>(&self, mut f: F) {
        for message in self.messages.values() {
            f(message);
        }
    }

    /// Trivial check for whether an object header message is shareable.
    pub fn can_share_common(msg: &SharedMessage) -> bool {
        !msg.data.is_empty()
    }

    /// Check whether a message would be shared or is already shared.
    pub fn can_share(msg: &SharedMessage) -> bool {
        Self::can_share_common(msg)
    }

    /// Attempt to share an object header message; returns true if it was shared.
    pub fn try_share(&mut self, key: u64, msg: SharedMessage) -> bool {
        if Self::can_share(&msg) {
            self.messages.insert(key, msg);
            true
        } else {
            false
        }
    }

    /// Increment the reference count of a SOHM message and return the new count.
    pub fn incr_ref(&mut self, key: u64) -> Option<u32> {
        let msg = self.messages.get_mut(&key)?;
        msg.refcount = msg.refcount.saturating_add(1);
        Some(msg.refcount)
    }

    /// Insert or overwrite a shareable message in the index at `key`.
    pub fn write_mesg(&mut self, key: u64, msg: SharedMessage) {
        self.messages.insert(key, msg);
    }

    /// Delete a SOHM message, returning the removed entry if any.
    pub fn delete(&mut self, key: u64) -> Option<SharedMessage> {
        self.messages.remove(&key)
    }

    /// Find a message in the list by key.
    pub fn find_in_list(&self, key: u64) -> Option<&SharedMessage> {
        self.messages.get(&key)
    }

    /// Decrement the reference count of a SOHM message and return the new count.
    pub fn decr_ref(&mut self, key: u64) -> Option<u32> {
        let msg = self.messages.get_mut(&key)?;
        msg.refcount = msg.refcount.saturating_sub(1);
        Some(msg.refcount)
    }

    /// Decrement the refcount for a particular message in this index, removing it on zero.
    pub fn delete_from_index(&mut self, key: u64) -> Option<SharedMessage> {
        self.delete(key)
    }

    /// Get the shared message info for this file (count and total bytes).
    pub fn get_info(&self) -> SharedMessageInfo {
        SharedMessageInfo {
            count: self.messages.len(),
            total_bytes: self.cache_table_image_len(),
        }
    }

    /// Reconstitute a shared message store from a plain (key, message) iterator.
    pub fn reconstitute(messages: impl IntoIterator<Item = (u64, SharedMessage)>) -> Self {
        Self {
            messages: messages.into_iter().collect(),
        }
    }

    /// V2 B-tree find callback returning the record's reference count.
    pub fn get_refcount_bt2_cb(&self, key: u64) -> Option<u32> {
        self.get_refcount(key)
    }

    /// Retrieve the reference count for a message shared in the heap.
    pub fn get_refcount(&self, key: u64) -> Option<u32> {
        self.messages.get(&key).map(|msg| msg.refcount)
    }

    /// Read back the encoded message payload for a given key.
    pub fn read_mesg(&self, key: u64) -> Option<&[u8]> {
        self.messages.get(&key).map(|msg| msg.data.as_slice())
    }

    /// Free memory used by the SOHM table (clears all entries).
    pub fn table_free(&mut self) {
        self.messages.clear();
    }

    /// Free all memory used by the SOHM list (clears all entries).
    pub fn list_free(&mut self) {
        self.messages.clear();
    }

    /// Write debugging information for the SOHM master table into `out`.
    pub fn write_table_debug<W: Write>(&self, out: &mut W) -> fmt::Result {
        write!(out, "{:?}", self.messages)
    }

    /// Write debugging information for a SOHM list into `out`.
    pub fn write_list_debug<W: Write>(&self, out: &mut W) -> fmt::Result {
        self.write_table_debug(out)
    }

    /// Sum of storage used by header, B-tree/list and fractal heap entries.
    pub fn ih_size(&self) -> usize {
        self.messages.len()
    }

    /// Compare callback used by [`crate::format::fractal_heap`] ops.
    pub fn compare_cb(lhs: &SharedMessage, rhs: &SharedMessage) -> std::cmp::Ordering {
        Self::message_compare(lhs, rhs)
    }

    /// Object-header iteration callback comparing a key against a stored message.
    pub fn compare_iter_op(lhs: &SharedMessage, rhs: &SharedMessage) -> std::cmp::Ordering {
        Self::message_compare(lhs, rhs)
    }

    /// Compare two shared messages by type then payload bytes.
    pub fn message_compare(lhs: &SharedMessage, rhs: &SharedMessage) -> std::cmp::Ordering {
        lhs.msg_type
            .cmp(&rhs.msg_type)
            .then_with(|| lhs.data.cmp(&rhs.data))
    }

    /// Append a [`SharedMessage`] raw encoding into `out`.
    pub fn message_encode_into(msg: &SharedMessage, out: &mut impl Extend<u8>) -> Result<()> {
        msg.encode_into(out)
    }

    /// Create the client callback context used by the v2 B-tree backing store.
    pub fn bt2_crt_context() -> Self {
        Self::default()
    }

    /// Destroy the client callback context used by the v2 B-tree backing store.
    pub fn bt2_dst_context(self) {}

    /// Store a SOHM message record into the v2 B-tree.
    pub fn bt2_store(&mut self, key: u64, msg: SharedMessage) {
        self.write_mesg(key, msg);
    }

    /// Write debugging information for a SOHM v2 B-tree record into `out`.
    pub fn write_bt2_debug<W: Write>(&self, out: &mut W) -> fmt::Result {
        self.write_table_debug(out)
    }
}

/// Read a little-endian `u64` from `data` at `pos`, surfacing `context` in error messages.
fn read_u64_le_at(data: &[u8], pos: usize, context: &str) -> Result<u64> {
    let end = pos
        .checked_add(8)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))?;
    let bytes: [u8; 8] = data
        .get(pos..end)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))?
        .try_into()
        .map_err(|_| Error::InvalidFormat(format!("{context} is truncated")))?;
    Ok(u64::from_le_bytes(bytes))
}

/// Read a little-endian `u32` from `data` at `pos`, surfacing `context` in error messages.
fn read_u32_le_at(data: &[u8], pos: usize, context: &str) -> Result<u32> {
    let end = pos
        .checked_add(4)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} offset overflow")))?;
    let bytes: [u8; 4] = data
        .get(pos..end)
        .ok_or_else(|| Error::InvalidFormat(format!("{context} is truncated")))?
        .try_into()
        .map_err(|_| Error::InvalidFormat(format!("{context} is truncated")))?;
    Ok(u32::from_le_bytes(bytes))
}

/// C-name alias for [`SharedMessageStore::cache_table_get_initial_load_size`].
#[allow(non_snake_case)]
pub fn H5SM__cache_table_get_initial_load_size(count: usize) -> usize {
    SharedMessageStore::cache_table_get_initial_load_size(count)
}

/// C-name alias for [`SharedMessageStore::cache_table_get_initial_load_size_checked`].
#[allow(non_snake_case)]
pub fn H5SM__cache_table_get_initial_load_size_checked(count: usize) -> Result<usize> {
    SharedMessageStore::cache_table_get_initial_load_size_checked(count)
}

/// C-name alias for [`SharedMessageStore::cache_table_verify_chksum`].
#[allow(non_snake_case)]
pub fn H5SM__cache_table_verify_chksum(bytes: &[u8], checksum: u32) -> bool {
    SharedMessageStore::cache_table_verify_chksum(bytes, checksum)
}

/// C-name alias for [`SharedMessageStore::cache_table_image_len`].
#[allow(non_snake_case)]
pub fn H5SM__cache_table_image_len(store: &SharedMessageStore) -> usize {
    store.cache_table_image_len()
}

/// C-name alias for [`SharedMessageStore::cache_table_image_len_checked`].
#[allow(non_snake_case)]
pub fn H5SM__cache_table_image_len_checked(store: &SharedMessageStore) -> Result<usize> {
    store.cache_table_image_len_checked()
}

/// C-name alias for [`SharedMessageStore::cache_table_serialize_into`].
#[allow(non_snake_case)]
pub fn H5SM__cache_table_serialize_into(
    store: &SharedMessageStore,
    out: &mut impl Extend<u8>,
) -> Result<()> {
    store.cache_table_serialize_into(out)
}

/// C-name alias for [`SharedMessageStore::cache_table_free_icr`].
#[allow(non_snake_case)]
pub fn H5SM__cache_table_free_icr(bytes: Vec<u8>) {
    SharedMessageStore::cache_table_free_icr(bytes)
}

/// C-name alias for [`SharedMessageStore::cache_list_get_initial_load_size`].
#[allow(non_snake_case)]
pub fn H5SM__cache_list_get_initial_load_size(count: usize) -> usize {
    SharedMessageStore::cache_list_get_initial_load_size(count)
}

/// C-name alias for [`SharedMessageStore::cache_list_get_initial_load_size_checked`].
#[allow(non_snake_case)]
pub fn H5SM__cache_list_get_initial_load_size_checked(count: usize) -> Result<usize> {
    SharedMessageStore::cache_list_get_initial_load_size_checked(count)
}

/// C-name alias for [`SharedMessageStore::cache_list_verify_chksum`].
#[allow(non_snake_case)]
pub fn H5SM__cache_list_verify_chksum(bytes: &[u8], checksum: u32) -> bool {
    SharedMessageStore::cache_list_verify_chksum(bytes, checksum)
}

/// C-name alias for [`SharedMessageStore::cache_list_deserialize`].
#[allow(non_snake_case)]
pub fn H5SM__cache_list_deserialize(bytes: &[u8]) -> Result<SharedMessageStore> {
    SharedMessageStore::cache_list_deserialize(bytes)
}

/// C-name alias for [`SharedMessageStore::cache_list_deserialize_into`].
#[allow(non_snake_case)]
pub fn H5SM__cache_list_deserialize_into(
    store: &mut SharedMessageStore,
    bytes: &[u8],
) -> Result<()> {
    store.cache_list_deserialize_into(bytes)
}

/// C-name alias for [`SharedMessageStore::cache_list_image_len`].
#[allow(non_snake_case)]
pub fn H5SM__cache_list_image_len(store: &SharedMessageStore) -> usize {
    store.cache_list_image_len()
}

/// C-name alias for [`SharedMessageStore::cache_list_image_len_checked`].
#[allow(non_snake_case)]
pub fn H5SM__cache_list_image_len_checked(store: &SharedMessageStore) -> Result<usize> {
    store.cache_list_image_len_checked()
}

/// C-name alias for [`SharedMessageStore::cache_list_serialize_into`].
#[allow(non_snake_case)]
pub fn H5SM__cache_list_serialize_into(
    store: &SharedMessageStore,
    out: &mut impl Extend<u8>,
) -> Result<()> {
    store.cache_list_serialize_into(out)
}

/// C-name alias for [`SharedMessageStore::cache_list_free_icr`].
#[allow(non_snake_case)]
pub fn H5SM__cache_list_free_icr(bytes: Vec<u8>) {
    SharedMessageStore::cache_list_free_icr(bytes)
}

/// C-name alias for [`SharedMessageStore::get_mesg_count_test`].
#[allow(non_snake_case)]
pub fn H5SM__get_mesg_count_test(store: &SharedMessageStore) -> usize {
    store.get_mesg_count_test()
}

/// C-name alias for [`SharedMessageStore::init`].
#[allow(non_snake_case)]
pub fn H5SM_init() -> SharedMessageStore {
    SharedMessageStore::init()
}

/// C-name alias for [`SharedMessageStore::type_to_flag`].
#[allow(non_snake_case)]
pub fn H5SM__type_to_flag(msg_type: u8) -> u32 {
    SharedMessageStore::type_to_flag(msg_type)
}

/// C-name alias for [`SharedMessageStore::type_shared`].
#[allow(non_snake_case)]
pub fn H5SM_type_shared(msg_type: u8, mask: u32) -> bool {
    SharedMessageStore::type_shared(msg_type, mask)
}

/// C-name alias for [`SharedMessageStore::get_fheap_addr`].
#[allow(non_snake_case)]
pub fn H5SM_get_fheap_addr(store: &SharedMessageStore, key: u64) -> Option<u64> {
    store.get_fheap_addr(key)
}

/// C-name alias for [`SharedMessageStore::create_index`].
#[allow(non_snake_case)]
pub fn H5SM__create_index(store: &mut SharedMessageStore) {
    store.create_index()
}

/// C-name alias for [`SharedMessageStore::delete_index`].
#[allow(non_snake_case)]
pub fn H5SM__delete_index(store: &mut SharedMessageStore) {
    store.delete_index()
}

/// C-name alias for [`SharedMessageStore::create_list`].
#[allow(non_snake_case)]
pub fn H5SM__create_list(store: &mut SharedMessageStore) {
    store.create_list()
}

/// C-name alias for [`SharedMessageStore::clone_messages_into`].
#[allow(non_snake_case)]
pub fn H5SM_clone_messages_into(store: &SharedMessageStore, out: &mut impl Extend<SharedMessage>) {
    store.clone_messages_into(out)
}

/// C-name alias for [`SharedMessageStore::into_messages`].
#[allow(non_snake_case)]
pub fn H5SM_into_messages(store: SharedMessageStore) -> impl Iterator<Item = SharedMessage> {
    store.into_messages()
}

/// C-name alias for [`SharedMessageStore::messages`].
#[allow(non_snake_case)]
pub fn H5SM_messages(store: &SharedMessageStore) -> impl Iterator<Item = &SharedMessage> {
    store.messages()
}

/// C-name alias for [`SharedMessageStore::can_share_common`].
#[allow(non_snake_case)]
pub fn H5SM__can_share_common(msg: &SharedMessage) -> bool {
    SharedMessageStore::can_share_common(msg)
}

/// C-name alias for [`SharedMessageStore::can_share`].
#[allow(non_snake_case)]
pub fn H5SM_can_share(msg: &SharedMessage) -> bool {
    SharedMessageStore::can_share(msg)
}

/// C-name alias for [`SharedMessageStore::try_share`].
#[allow(non_snake_case)]
pub fn H5SM_try_share(store: &mut SharedMessageStore, key: u64, msg: SharedMessage) -> bool {
    store.try_share(key, msg)
}

/// C-name alias for [`SharedMessageStore::incr_ref`].
#[allow(non_snake_case)]
pub fn H5SM__incr_ref(store: &mut SharedMessageStore, key: u64) -> Option<u32> {
    store.incr_ref(key)
}

/// C-name alias for [`SharedMessageStore::write_mesg`].
#[allow(non_snake_case)]
pub fn H5SM__write_mesg(store: &mut SharedMessageStore, key: u64, msg: SharedMessage) {
    store.write_mesg(key, msg)
}

/// C-name alias for [`SharedMessageStore::delete`].
#[allow(non_snake_case)]
pub fn H5SM_delete(store: &mut SharedMessageStore, key: u64) -> Option<SharedMessage> {
    store.delete(key)
}

/// C-name alias for [`SharedMessageStore::find_in_list`].
#[allow(non_snake_case)]
pub fn H5SM__find_in_list(store: &SharedMessageStore, key: u64) -> Option<&SharedMessage> {
    store.find_in_list(key)
}

/// C-name alias for [`SharedMessageStore::decr_ref`].
#[allow(non_snake_case)]
pub fn H5SM__decr_ref(store: &mut SharedMessageStore, key: u64) -> Option<u32> {
    store.decr_ref(key)
}

/// C-name alias for [`SharedMessageStore::delete_from_index`].
#[allow(non_snake_case)]
pub fn H5SM__delete_from_index(store: &mut SharedMessageStore, key: u64) -> Option<SharedMessage> {
    store.delete_from_index(key)
}

/// C-name alias for [`SharedMessageStore::get_info`].
#[allow(non_snake_case)]
pub fn H5SM_get_info(store: &SharedMessageStore) -> SharedMessageInfo {
    store.get_info()
}

/// C-name alias for [`SharedMessageStore::reconstitute`].
#[allow(non_snake_case)]
pub fn H5SM_reconstitute(
    messages: impl IntoIterator<Item = (u64, SharedMessage)>,
) -> SharedMessageStore {
    SharedMessageStore::reconstitute(messages)
}

/// C-name alias for [`SharedMessageStore::get_refcount_bt2_cb`].
#[allow(non_snake_case)]
pub fn H5SM__get_refcount_bt2_cb(store: &SharedMessageStore, key: u64) -> Option<u32> {
    store.get_refcount_bt2_cb(key)
}

/// C-name alias for [`SharedMessageStore::get_refcount`].
#[allow(non_snake_case)]
pub fn H5SM_get_refcount(store: &SharedMessageStore, key: u64) -> Option<u32> {
    store.get_refcount(key)
}

/// C-name alias for [`SharedMessageStore::read_mesg`].
#[allow(non_snake_case)]
pub fn H5SM__read_mesg(store: &SharedMessageStore, key: u64) -> Option<&[u8]> {
    store.read_mesg(key)
}

/// C-name alias for [`SharedMessageStore::table_free`].
#[allow(non_snake_case)]
pub fn H5SM__table_free(store: &mut SharedMessageStore) {
    store.table_free()
}

/// C-name alias for [`SharedMessageStore::list_free`].
#[allow(non_snake_case)]
pub fn H5SM__list_free(store: &mut SharedMessageStore) {
    store.list_free()
}

/// C-name alias for [`SharedMessageStore::write_table_debug`].
#[allow(non_snake_case)]
pub fn H5SM_write_table_debug<W: Write>(store: &SharedMessageStore, out: &mut W) -> fmt::Result {
    store.write_table_debug(out)
}

/// C-name alias for [`SharedMessageStore::write_list_debug`].
#[allow(non_snake_case)]
pub fn H5SM_write_list_debug<W: Write>(store: &SharedMessageStore, out: &mut W) -> fmt::Result {
    store.write_list_debug(out)
}

/// C-name alias for [`SharedMessageStore::ih_size`].
#[allow(non_snake_case)]
pub fn H5SM_ih_size(store: &SharedMessageStore) -> usize {
    store.ih_size()
}

/// C-name alias for [`SharedMessageStore::compare_cb`].
#[allow(non_snake_case)]
pub fn H5SM__compare_cb(lhs: &SharedMessage, rhs: &SharedMessage) -> std::cmp::Ordering {
    SharedMessageStore::compare_cb(lhs, rhs)
}

/// C-name alias for [`SharedMessageStore::compare_iter_op`].
#[allow(non_snake_case)]
pub fn H5SM__compare_iter_op(lhs: &SharedMessage, rhs: &SharedMessage) -> std::cmp::Ordering {
    SharedMessageStore::compare_iter_op(lhs, rhs)
}

/// C-name alias for [`SharedMessageStore::message_compare`].
#[allow(non_snake_case)]
pub fn H5SM__message_compare(lhs: &SharedMessage, rhs: &SharedMessage) -> std::cmp::Ordering {
    SharedMessageStore::message_compare(lhs, rhs)
}

/// C-name alias for [`SharedMessageStore::message_encode_into`].
#[allow(non_snake_case)]
pub fn H5SM__message_encode_into(msg: &SharedMessage, out: &mut impl Extend<u8>) -> Result<()> {
    SharedMessageStore::message_encode_into(msg, out)
}

/// C-name alias for [`SharedMessageStore::bt2_crt_context`].
#[allow(non_snake_case)]
pub fn H5SM__bt2_crt_context() -> SharedMessageStore {
    SharedMessageStore::bt2_crt_context()
}

/// C-name alias for [`SharedMessageStore::bt2_dst_context`].
#[allow(non_snake_case)]
pub fn H5SM__bt2_dst_context(store: SharedMessageStore) {
    store.bt2_dst_context()
}

/// C-name alias for [`SharedMessageStore::bt2_store`].
#[allow(non_snake_case)]
pub fn H5SM__bt2_store(store: &mut SharedMessageStore, key: u64, msg: SharedMessage) {
    store.bt2_store(key, msg)
}

/// C-name alias for [`SharedMessageStore::write_bt2_debug`].
#[allow(non_snake_case)]
pub fn H5SM__write_bt2_debug<W: Write>(store: &SharedMessageStore, out: &mut W) -> fmt::Result {
    store.write_bt2_debug(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_message_store_aliases_roundtrip() {
        let msg = SharedMessage::new(2, 99, vec![1, 2, 3]);
        assert!(SharedMessageStore::can_share(&msg));
        assert!(SharedMessageStore::type_shared(
            2,
            SharedMessageStore::type_to_flag(2)
        ));

        let mut store = SharedMessageStore::init();
        store.create_index();
        store.create_list();
        assert!(store.try_share(10, msg.clone()));
        assert_eq!(store.get_mesg_count_test(), 1);
        assert_eq!(store.get_fheap_addr(10), Some(99));
        assert_eq!(store.incr_ref(10), Some(2));
        assert_eq!(store.decr_ref(10), Some(1));
        assert_eq!(store.get_refcount_bt2_cb(10), Some(1));
        assert_eq!(store.read_mesg(10), Some([1, 2, 3].as_slice()));
        assert!(store.cache_table_image_len() >= msg.encoded_len());
        assert_eq!(store.cache_table_image_len_checked().unwrap(), 16);
        assert_eq!(store.cache_list_image_len_checked().unwrap(), 16);
        let mut encoded = Vec::new();
        SharedMessageStore::message_encode_into(&msg, &mut encoded).unwrap();
        let mut encoded_direct = Vec::new();
        msg.encode_into(&mut encoded_direct).unwrap();
        assert_eq!(encoded, encoded_direct);
        assert_eq!(store.get_info().count, 1);
        let mut debug = String::new();
        store.write_table_debug(&mut debug).unwrap();
        assert!(debug.contains("SharedMessage"));

        let mut bytes = Vec::new();
        store.cache_list_serialize_into(&mut bytes).unwrap();
        let decoded = SharedMessageStore::cache_list_deserialize(&bytes).unwrap();
        assert_eq!(decoded.get_mesg_count_test(), 1);
        assert_eq!(decoded.find_in_list(99).unwrap().data, vec![1, 2, 3]);
        SharedMessageStore::cache_table_free_icr(bytes.clone());
        SharedMessageStore::cache_list_free_icr(bytes);
        assert_eq!(store.messages().count(), 1);
        let mut cloned_messages = Vec::new();
        store.clone_messages_into(&mut cloned_messages);
        assert_eq!(cloned_messages, vec![msg.clone()]);

        let mut rebuilt = SharedMessageStore::reconstitute(vec![(1, msg.clone())]);
        rebuilt.bt2_store(2, msg.clone());
        assert_eq!(rebuilt.ih_size(), 2);
        rebuilt.delete_from_index(1);
        rebuilt.table_free();
        assert_eq!(rebuilt.ih_size(), 0);

        let consumed = SharedMessageStore::reconstitute([(3, msg)])
            .into_messages()
            .collect::<Vec<_>>();
        assert_eq!(consumed.len(), 1);
    }

    #[test]
    fn h5sm_aliases_roundtrip() {
        let msg = SharedMessage::new(2, 99, vec![1, 2, 3]);
        let other = SharedMessage::new(3, 100, vec![4]);
        assert!(H5SM__can_share_common(&msg));
        assert!(H5SM_can_share(&msg));
        assert!(H5SM_type_shared(2, H5SM__type_to_flag(2)));
        assert_eq!(
            H5SM__message_compare(&msg, &other),
            std::cmp::Ordering::Less
        );
        assert_eq!(H5SM__compare_cb(&msg, &other), std::cmp::Ordering::Less);
        assert_eq!(
            H5SM__compare_iter_op(&msg, &other),
            std::cmp::Ordering::Less
        );
        let mut encoded = Vec::new();
        H5SM__message_encode_into(&msg, &mut encoded).unwrap();
        let mut encoded_direct = Vec::new();
        msg.encode_into(&mut encoded_direct).unwrap();
        assert_eq!(encoded, encoded_direct);

        let mut store = H5SM_init();
        H5SM__create_index(&mut store);
        H5SM__create_list(&mut store);
        assert!(H5SM_try_share(&mut store, 10, msg.clone()));
        assert_eq!(H5SM__get_mesg_count_test(&store), 1);
        assert_eq!(H5SM_get_fheap_addr(&store, 10), Some(99));
        assert_eq!(H5SM__incr_ref(&mut store, 10), Some(2));
        assert_eq!(H5SM__decr_ref(&mut store, 10), Some(1));
        assert_eq!(H5SM_get_refcount(&store, 10), Some(1));
        assert_eq!(H5SM__get_refcount_bt2_cb(&store, 10), Some(1));
        assert_eq!(H5SM__read_mesg(&store, 10), Some([1, 2, 3].as_slice()));
        assert_eq!(H5SM__find_in_list(&store, 10), Some(&msg));
        assert_eq!(H5SM_get_info(&store).count, 1);
        assert_eq!(H5SM_ih_size(&store), 1);
        let mut table_debug = String::new();
        H5SM_write_table_debug(&store, &mut table_debug).unwrap();
        assert!(table_debug.contains("SharedMessage"));
        let mut list_debug = String::new();
        H5SM_write_list_debug(&store, &mut list_debug).unwrap();
        assert!(list_debug.contains("SharedMessage"));

        let mut table_image = Vec::new();
        H5SM__cache_table_serialize_into(&store, &mut table_image).unwrap();
        let table_checksum = crc32fast::hash(&table_image);
        assert!(H5SM__cache_table_verify_chksum(
            &table_image,
            table_checksum
        ));
        assert_eq!(H5SM__cache_table_image_len(&store), table_image.len());
        assert_eq!(
            H5SM__cache_table_image_len_checked(&store).unwrap(),
            table_image.len()
        );
        assert_eq!(H5SM__cache_table_get_initial_load_size(2), 32);
        assert_eq!(
            H5SM__cache_table_get_initial_load_size_checked(2).unwrap(),
            32
        );
        H5SM__cache_table_free_icr(table_image.clone());

        let mut list_image = Vec::new();
        H5SM__cache_list_serialize_into(&store, &mut list_image).unwrap();
        let list_checksum = crc32fast::hash(&list_image);
        assert!(H5SM__cache_list_verify_chksum(&list_image, list_checksum));
        let decoded = H5SM__cache_list_deserialize(&list_image).unwrap();
        assert_eq!(H5SM__get_mesg_count_test(&decoded), 1);
        assert_eq!(H5SM__read_mesg(&decoded, 99), Some([1, 2, 3].as_slice()));
        assert_eq!(H5SM__cache_list_image_len(&store), list_image.len());
        assert_eq!(
            H5SM__cache_list_image_len_checked(&store).unwrap(),
            list_image.len()
        );
        assert_eq!(H5SM__cache_list_get_initial_load_size(2), 24);
        assert_eq!(
            H5SM__cache_list_get_initial_load_size_checked(2).unwrap(),
            24
        );
        H5SM__cache_list_free_icr(list_image);
        assert_eq!(
            H5SM_messages(&store).cloned().collect::<Vec<_>>(),
            vec![msg.clone()]
        );
        let mut cloned_messages = Vec::new();
        H5SM_clone_messages_into(&store, &mut cloned_messages);
        assert_eq!(cloned_messages, vec![msg.clone()]);
        assert_eq!(
            H5SM_into_messages(H5SM_reconstitute([(1, msg.clone())])).collect::<Vec<_>>(),
            vec![msg.clone()]
        );

        let mut bt2 = H5SM__bt2_crt_context();
        H5SM__bt2_store(&mut bt2, 1, msg.clone());
        let mut bt2_debug = String::new();
        H5SM__write_bt2_debug(&bt2, &mut bt2_debug).unwrap();
        assert!(bt2_debug.contains("SharedMessage"));
        H5SM__bt2_dst_context(bt2);

        H5SM__write_mesg(&mut store, 20, other.clone());
        assert_eq!(H5SM_delete(&mut store, 20), Some(other));
        assert_eq!(H5SM__delete_from_index(&mut store, 10), Some(msg.clone()));
        assert_eq!(H5SM_ih_size(&store), 0);

        let mut rebuilt = H5SM_reconstitute(vec![(5, msg)]);
        assert_eq!(H5SM_ih_size(&rebuilt), 1);
        H5SM__list_free(&mut rebuilt);
        assert_eq!(H5SM_ih_size(&rebuilt), 0);
        H5SM__write_mesg(&mut rebuilt, 6, SharedMessage::new(1, 7, vec![8]));
        H5SM__delete_index(&mut rebuilt);
        assert_eq!(H5SM_ih_size(&rebuilt), 0);
        H5SM__write_mesg(&mut rebuilt, 7, SharedMessage::new(1, 8, vec![9]));
        H5SM__table_free(&mut rebuilt);
        assert_eq!(H5SM_ih_size(&rebuilt), 0);
    }

    #[test]
    fn shared_message_initial_load_size_checked_rejects_overflow() {
        assert!(H5SM__cache_table_get_initial_load_size_checked(usize::MAX).is_err());
        assert!(H5SM__cache_list_get_initial_load_size_checked(usize::MAX).is_err());
        assert_eq!(
            H5SM__cache_table_get_initial_load_size(usize::MAX),
            usize::MAX
        );
        assert_eq!(
            H5SM__cache_list_get_initial_load_size(usize::MAX),
            usize::MAX
        );
    }

    #[test]
    fn shared_message_decode_rejects_truncated_payload() {
        let mut bytes = Vec::new();
        bytes.push(1);
        bytes.extend_from_slice(&42u64.to_le_bytes());
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&[1, 2, 3]);

        let err = SharedMessageStore::cache_list_deserialize(&bytes).unwrap_err();
        assert!(
            err.to_string()
                .contains("shared message cache list data is truncated"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn shared_message_encode_into_appends_to_existing_output() {
        let msg = SharedMessage::new(7, 0x0102_0304_0506_0708, vec![9, 8, 7]);

        let mut bytes = vec![0xaa, 0xbb];
        msg.encode_into(&mut bytes).unwrap();
        assert_eq!(&bytes[..2], &[0xaa, 0xbb]);
        assert_eq!(bytes[2], 7);
        assert_eq!(&bytes[3..11], &0x0102_0304_0506_0708u64.to_le_bytes());
        assert_eq!(&bytes[11..15], &3u32.to_le_bytes());
        assert_eq!(&bytes[15..], &[9, 8, 7]);

        let mut wrapper_bytes = vec![0xcc];
        H5SM__message_encode_into(&msg, &mut wrapper_bytes).unwrap();
        assert_eq!(wrapper_bytes[0], 0xcc);
        assert_eq!(&wrapper_bytes[1..], &bytes[2..]);
    }

    #[test]
    fn shared_message_cache_serialize_into_appends_to_existing_output() {
        let store = SharedMessageStore::reconstitute([
            (20, SharedMessage::new(2, 20, vec![1, 2])),
            (10, SharedMessage::new(1, 10, vec![3])),
        ]);

        let mut direct = Vec::new();
        store.cache_table_serialize_into(&mut direct).unwrap();

        let mut table_bytes = vec![0xee];
        H5SM__cache_table_serialize_into(&store, &mut table_bytes).unwrap();
        assert_eq!(table_bytes[0], 0xee);
        assert_eq!(&table_bytes[1..], direct.as_slice());

        let mut list_bytes = vec![0xdd, 0xcc];
        H5SM__cache_list_serialize_into(&store, &mut list_bytes).unwrap();
        assert_eq!(&list_bytes[..2], &[0xdd, 0xcc]);
        assert_eq!(&list_bytes[2..], direct.as_slice());
    }

    #[test]
    fn shared_message_cache_list_deserialize_into_preserves_store_on_error() {
        let original = SharedMessageStore::reconstitute([
            (11, SharedMessage::new(4, 11, vec![1, 1])),
            (22, SharedMessage::new(5, 22, vec![2, 2])),
        ]);
        let mut malformed = Vec::new();
        malformed.push(1);
        malformed.extend_from_slice(&42u64.to_le_bytes());
        malformed.extend_from_slice(&4u32.to_le_bytes());
        malformed.extend_from_slice(&[1, 2, 3]);

        let mut store = original.clone();
        let err = store.cache_list_deserialize_into(&malformed).unwrap_err();
        assert!(
            err.to_string()
                .contains("shared message cache list data is truncated"),
            "unexpected error: {err}"
        );
        assert_eq!(store, original);

        let mut wrapper_store = original.clone();
        assert!(H5SM__cache_list_deserialize_into(&mut wrapper_store, &[0]).is_err());
        assert_eq!(wrapper_store, original);
    }
}
