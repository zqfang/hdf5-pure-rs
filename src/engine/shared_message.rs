use std::collections::BTreeMap;

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
    pub fn new(msg_type: u8, heap_addr: u64, data: Vec<u8>) -> Self {
        Self {
            msg_type,
            heap_addr,
            data,
            refcount: 1,
        }
    }

    pub fn encoded_len(&self) -> usize {
        1 + 8 + 4 + self.data.len()
    }

    pub fn encode(&self) -> Result<Vec<u8>> {
        let data_len = u32::try_from(self.data.len()).map_err(|_| {
            Error::InvalidFormat("shared message payload is too large to encode".into())
        })?;
        let mut out = Vec::with_capacity(self.encoded_len());
        out.push(self.msg_type);
        out.extend_from_slice(&self.heap_addr.to_le_bytes());
        out.extend_from_slice(&data_len.to_le_bytes());
        out.extend_from_slice(&self.data);
        Ok(out)
    }
}

impl SharedMessageStore {
    pub fn cache_table_get_initial_load_size(count: usize) -> usize {
        count.saturating_mul(16)
    }

    pub fn cache_table_verify_chksum(bytes: &[u8], checksum: u32) -> bool {
        crc32fast::hash(bytes) == checksum
    }

    pub fn cache_table_image_len(&self) -> usize {
        self.messages.values().map(SharedMessage::encoded_len).sum()
    }

    pub fn cache_table_serialize(&self) -> Result<Vec<u8>> {
        let mut out = Vec::with_capacity(self.cache_table_image_len());
        for message in self.messages.values() {
            out.extend_from_slice(&message.encode()?);
        }
        Ok(out)
    }

    pub fn cache_table_free_icr(_bytes: Vec<u8>) {}

    pub fn cache_list_get_initial_load_size(count: usize) -> usize {
        count.saturating_mul(12)
    }

    pub fn cache_list_verify_chksum(bytes: &[u8], checksum: u32) -> bool {
        Self::cache_table_verify_chksum(bytes, checksum)
    }

    pub fn cache_list_deserialize(bytes: &[u8]) -> Result<Self> {
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
        Ok(Self { messages })
    }

    pub fn cache_list_image_len(&self) -> usize {
        self.cache_table_image_len()
    }

    pub fn cache_list_serialize(&self) -> Result<Vec<u8>> {
        self.cache_table_serialize()
    }

    pub fn cache_list_free_icr(_bytes: Vec<u8>) {}

    pub fn get_mesg_count_test(&self) -> usize {
        self.messages.len()
    }

    pub fn init() -> Self {
        Self::default()
    }

    pub fn type_to_flag(msg_type: u8) -> u32 {
        1u32.checked_shl(u32::from(msg_type)).unwrap_or(0)
    }

    pub fn type_shared(msg_type: u8, mask: u32) -> bool {
        mask & Self::type_to_flag(msg_type) != 0
    }

    pub fn get_fheap_addr(&self, key: u64) -> Option<u64> {
        self.messages.get(&key).map(|msg| msg.heap_addr)
    }

    pub fn create_index(&mut self) {}

    pub fn delete_index(&mut self) {
        self.messages.clear();
    }

    pub fn create_list(&mut self) {}

    pub fn bt2_convert_to_list_op(&self) -> Vec<SharedMessage> {
        self.messages.values().cloned().collect()
    }

    pub fn can_share_common(msg: &SharedMessage) -> bool {
        !msg.data.is_empty()
    }

    pub fn can_share(msg: &SharedMessage) -> bool {
        Self::can_share_common(msg)
    }

    pub fn try_share(&mut self, key: u64, msg: SharedMessage) -> bool {
        if Self::can_share(&msg) {
            self.messages.insert(key, msg);
            true
        } else {
            false
        }
    }

    pub fn incr_ref(&mut self, key: u64) -> Option<u32> {
        let msg = self.messages.get_mut(&key)?;
        msg.refcount = msg.refcount.saturating_add(1);
        Some(msg.refcount)
    }

    pub fn write_mesg(&mut self, key: u64, msg: SharedMessage) {
        self.messages.insert(key, msg);
    }

    pub fn delete(&mut self, key: u64) -> Option<SharedMessage> {
        self.messages.remove(&key)
    }

    pub fn find_in_list(&self, key: u64) -> Option<&SharedMessage> {
        self.messages.get(&key)
    }

    pub fn decr_ref(&mut self, key: u64) -> Option<u32> {
        let msg = self.messages.get_mut(&key)?;
        msg.refcount = msg.refcount.saturating_sub(1);
        Some(msg.refcount)
    }

    pub fn delete_from_index(&mut self, key: u64) -> Option<SharedMessage> {
        self.delete(key)
    }

    pub fn get_info(&self) -> SharedMessageInfo {
        SharedMessageInfo {
            count: self.messages.len(),
            total_bytes: self.cache_table_image_len(),
        }
    }

    pub fn reconstitute(messages: Vec<(u64, SharedMessage)>) -> Self {
        Self {
            messages: messages.into_iter().collect(),
        }
    }

    pub fn get_refcount_bt2_cb(&self, key: u64) -> Option<u32> {
        self.get_refcount(key)
    }

    pub fn get_refcount(&self, key: u64) -> Option<u32> {
        self.messages.get(&key).map(|msg| msg.refcount)
    }

    pub fn read_mesg(&self, key: u64) -> Option<&[u8]> {
        self.messages.get(&key).map(|msg| msg.data.as_slice())
    }

    pub fn table_free(&mut self) {
        self.messages.clear();
    }

    pub fn list_free(&mut self) {
        self.messages.clear();
    }

    pub fn table_debug(&self) -> String {
        format!("{:?}", self.messages)
    }

    pub fn list_debug(&self) -> String {
        self.table_debug()
    }

    pub fn ih_size(&self) -> usize {
        self.messages.len()
    }

    pub fn compare_cb(lhs: &SharedMessage, rhs: &SharedMessage) -> std::cmp::Ordering {
        Self::message_compare(lhs, rhs)
    }

    pub fn compare_iter_op(lhs: &SharedMessage, rhs: &SharedMessage) -> std::cmp::Ordering {
        Self::message_compare(lhs, rhs)
    }

    pub fn message_compare(lhs: &SharedMessage, rhs: &SharedMessage) -> std::cmp::Ordering {
        lhs.msg_type
            .cmp(&rhs.msg_type)
            .then_with(|| lhs.data.cmp(&rhs.data))
    }

    pub fn message_encode(msg: &SharedMessage) -> Result<Vec<u8>> {
        msg.encode()
    }

    pub fn bt2_crt_context() -> Self {
        Self::default()
    }

    pub fn bt2_dst_context(self) {}

    pub fn bt2_store(&mut self, key: u64, msg: SharedMessage) {
        self.write_mesg(key, msg);
    }

    pub fn bt2_debug(&self) -> String {
        self.table_debug()
    }
}

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

#[allow(non_snake_case)]
pub fn H5SM__cache_table_get_initial_load_size(count: usize) -> usize {
    SharedMessageStore::cache_table_get_initial_load_size(count)
}

#[allow(non_snake_case)]
pub fn H5SM__cache_table_verify_chksum(bytes: &[u8], checksum: u32) -> bool {
    SharedMessageStore::cache_table_verify_chksum(bytes, checksum)
}

#[allow(non_snake_case)]
pub fn H5SM__cache_table_image_len(store: &SharedMessageStore) -> usize {
    store.cache_table_image_len()
}

#[allow(non_snake_case)]
pub fn H5SM__cache_table_serialize(store: &SharedMessageStore) -> Result<Vec<u8>> {
    store.cache_table_serialize()
}

#[allow(non_snake_case)]
pub fn H5SM__cache_table_free_icr(bytes: Vec<u8>) {
    SharedMessageStore::cache_table_free_icr(bytes)
}

#[allow(non_snake_case)]
pub fn H5SM__cache_list_get_initial_load_size(count: usize) -> usize {
    SharedMessageStore::cache_list_get_initial_load_size(count)
}

#[allow(non_snake_case)]
pub fn H5SM__cache_list_verify_chksum(bytes: &[u8], checksum: u32) -> bool {
    SharedMessageStore::cache_list_verify_chksum(bytes, checksum)
}

#[allow(non_snake_case)]
pub fn H5SM__cache_list_deserialize(bytes: &[u8]) -> Result<SharedMessageStore> {
    SharedMessageStore::cache_list_deserialize(bytes)
}

#[allow(non_snake_case)]
pub fn H5SM__cache_list_image_len(store: &SharedMessageStore) -> usize {
    store.cache_list_image_len()
}

#[allow(non_snake_case)]
pub fn H5SM__cache_list_serialize(store: &SharedMessageStore) -> Result<Vec<u8>> {
    store.cache_list_serialize()
}

#[allow(non_snake_case)]
pub fn H5SM__cache_list_free_icr(bytes: Vec<u8>) {
    SharedMessageStore::cache_list_free_icr(bytes)
}

#[allow(non_snake_case)]
pub fn H5SM__get_mesg_count_test(store: &SharedMessageStore) -> usize {
    store.get_mesg_count_test()
}

#[allow(non_snake_case)]
pub fn H5SM_init() -> SharedMessageStore {
    SharedMessageStore::init()
}

#[allow(non_snake_case)]
pub fn H5SM__type_to_flag(msg_type: u8) -> u32 {
    SharedMessageStore::type_to_flag(msg_type)
}

#[allow(non_snake_case)]
pub fn H5SM_type_shared(msg_type: u8, mask: u32) -> bool {
    SharedMessageStore::type_shared(msg_type, mask)
}

#[allow(non_snake_case)]
pub fn H5SM_get_fheap_addr(store: &SharedMessageStore, key: u64) -> Option<u64> {
    store.get_fheap_addr(key)
}

#[allow(non_snake_case)]
pub fn H5SM__create_index(store: &mut SharedMessageStore) {
    store.create_index()
}

#[allow(non_snake_case)]
pub fn H5SM__delete_index(store: &mut SharedMessageStore) {
    store.delete_index()
}

#[allow(non_snake_case)]
pub fn H5SM__create_list(store: &mut SharedMessageStore) {
    store.create_list()
}

#[allow(non_snake_case)]
pub fn H5SM__bt2_convert_to_list_op(store: &SharedMessageStore) -> Vec<SharedMessage> {
    store.bt2_convert_to_list_op()
}

#[allow(non_snake_case)]
pub fn H5SM__can_share_common(msg: &SharedMessage) -> bool {
    SharedMessageStore::can_share_common(msg)
}

#[allow(non_snake_case)]
pub fn H5SM_can_share(msg: &SharedMessage) -> bool {
    SharedMessageStore::can_share(msg)
}

#[allow(non_snake_case)]
pub fn H5SM_try_share(store: &mut SharedMessageStore, key: u64, msg: SharedMessage) -> bool {
    store.try_share(key, msg)
}

#[allow(non_snake_case)]
pub fn H5SM__incr_ref(store: &mut SharedMessageStore, key: u64) -> Option<u32> {
    store.incr_ref(key)
}

#[allow(non_snake_case)]
pub fn H5SM__write_mesg(store: &mut SharedMessageStore, key: u64, msg: SharedMessage) {
    store.write_mesg(key, msg)
}

#[allow(non_snake_case)]
pub fn H5SM_delete(store: &mut SharedMessageStore, key: u64) -> Option<SharedMessage> {
    store.delete(key)
}

#[allow(non_snake_case)]
pub fn H5SM__find_in_list(store: &SharedMessageStore, key: u64) -> Option<&SharedMessage> {
    store.find_in_list(key)
}

#[allow(non_snake_case)]
pub fn H5SM__decr_ref(store: &mut SharedMessageStore, key: u64) -> Option<u32> {
    store.decr_ref(key)
}

#[allow(non_snake_case)]
pub fn H5SM__delete_from_index(store: &mut SharedMessageStore, key: u64) -> Option<SharedMessage> {
    store.delete_from_index(key)
}

#[allow(non_snake_case)]
pub fn H5SM_get_info(store: &SharedMessageStore) -> SharedMessageInfo {
    store.get_info()
}

#[allow(non_snake_case)]
pub fn H5SM_reconstitute(messages: Vec<(u64, SharedMessage)>) -> SharedMessageStore {
    SharedMessageStore::reconstitute(messages)
}

#[allow(non_snake_case)]
pub fn H5SM__get_refcount_bt2_cb(store: &SharedMessageStore, key: u64) -> Option<u32> {
    store.get_refcount_bt2_cb(key)
}

#[allow(non_snake_case)]
pub fn H5SM_get_refcount(store: &SharedMessageStore, key: u64) -> Option<u32> {
    store.get_refcount(key)
}

#[allow(non_snake_case)]
pub fn H5SM__read_mesg(store: &SharedMessageStore, key: u64) -> Option<&[u8]> {
    store.read_mesg(key)
}

#[allow(non_snake_case)]
pub fn H5SM__table_free(store: &mut SharedMessageStore) {
    store.table_free()
}

#[allow(non_snake_case)]
pub fn H5SM__list_free(store: &mut SharedMessageStore) {
    store.list_free()
}

#[allow(non_snake_case)]
pub fn H5SM_table_debug(store: &SharedMessageStore) -> String {
    store.table_debug()
}

#[allow(non_snake_case)]
pub fn H5SM_list_debug(store: &SharedMessageStore) -> String {
    store.list_debug()
}

#[allow(non_snake_case)]
pub fn H5SM_ih_size(store: &SharedMessageStore) -> usize {
    store.ih_size()
}

#[allow(non_snake_case)]
pub fn H5SM__compare_cb(lhs: &SharedMessage, rhs: &SharedMessage) -> std::cmp::Ordering {
    SharedMessageStore::compare_cb(lhs, rhs)
}

#[allow(non_snake_case)]
pub fn H5SM__compare_iter_op(lhs: &SharedMessage, rhs: &SharedMessage) -> std::cmp::Ordering {
    SharedMessageStore::compare_iter_op(lhs, rhs)
}

#[allow(non_snake_case)]
pub fn H5SM__message_compare(lhs: &SharedMessage, rhs: &SharedMessage) -> std::cmp::Ordering {
    SharedMessageStore::message_compare(lhs, rhs)
}

#[allow(non_snake_case)]
pub fn H5SM__message_encode(msg: &SharedMessage) -> Result<Vec<u8>> {
    SharedMessageStore::message_encode(msg)
}

#[allow(non_snake_case)]
pub fn H5SM__bt2_crt_context() -> SharedMessageStore {
    SharedMessageStore::bt2_crt_context()
}

#[allow(non_snake_case)]
pub fn H5SM__bt2_dst_context(store: SharedMessageStore) {
    store.bt2_dst_context()
}

#[allow(non_snake_case)]
pub fn H5SM__bt2_store(store: &mut SharedMessageStore, key: u64, msg: SharedMessage) {
    store.bt2_store(key, msg)
}

#[allow(non_snake_case)]
pub fn H5SM__bt2_debug(store: &SharedMessageStore) -> String {
    store.bt2_debug()
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
        assert_eq!(
            SharedMessageStore::message_encode(&msg).unwrap(),
            msg.encode().unwrap()
        );
        assert_eq!(store.get_info().count, 1);
        assert!(store.table_debug().contains("SharedMessage"));

        let bytes = store.cache_list_serialize().unwrap();
        let decoded = SharedMessageStore::cache_list_deserialize(&bytes).unwrap();
        assert_eq!(decoded.get_mesg_count_test(), 1);
        assert_eq!(decoded.find_in_list(99).unwrap().data, vec![1, 2, 3]);
        SharedMessageStore::cache_table_free_icr(bytes.clone());
        SharedMessageStore::cache_list_free_icr(bytes);
        assert_eq!(store.bt2_convert_to_list_op().len(), 1);

        let mut rebuilt = SharedMessageStore::reconstitute(vec![(1, msg.clone())]);
        rebuilt.bt2_store(2, msg);
        assert_eq!(rebuilt.ih_size(), 2);
        rebuilt.delete_from_index(1);
        rebuilt.table_free();
        assert_eq!(rebuilt.ih_size(), 0);
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
        assert_eq!(H5SM__message_encode(&msg).unwrap(), msg.encode().unwrap());

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
        assert!(H5SM_table_debug(&store).contains("SharedMessage"));
        assert!(H5SM_list_debug(&store).contains("SharedMessage"));

        let table_image = H5SM__cache_table_serialize(&store).unwrap();
        let table_checksum = crc32fast::hash(&table_image);
        assert!(H5SM__cache_table_verify_chksum(
            &table_image,
            table_checksum
        ));
        assert_eq!(H5SM__cache_table_image_len(&store), table_image.len());
        assert_eq!(H5SM__cache_table_get_initial_load_size(2), 32);
        H5SM__cache_table_free_icr(table_image.clone());

        let list_image = H5SM__cache_list_serialize(&store).unwrap();
        let list_checksum = crc32fast::hash(&list_image);
        assert!(H5SM__cache_list_verify_chksum(&list_image, list_checksum));
        let decoded = H5SM__cache_list_deserialize(&list_image).unwrap();
        assert_eq!(H5SM__get_mesg_count_test(&decoded), 1);
        assert_eq!(H5SM__read_mesg(&decoded, 99), Some([1, 2, 3].as_slice()));
        assert_eq!(H5SM__cache_list_image_len(&store), list_image.len());
        assert_eq!(H5SM__cache_list_get_initial_load_size(2), 24);
        H5SM__cache_list_free_icr(list_image);
        assert_eq!(H5SM__bt2_convert_to_list_op(&store), vec![msg.clone()]);

        let mut bt2 = H5SM__bt2_crt_context();
        H5SM__bt2_store(&mut bt2, 1, msg.clone());
        assert!(H5SM__bt2_debug(&bt2).contains("SharedMessage"));
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
}
