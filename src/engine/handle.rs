use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;

/// The type used for internal object identifiers, analogous to HDF5's `hid_t`.
pub type Hid = i64;

/// Invalid handle ID sentinel.
pub const INVALID_HID: Hid = -1;

pub fn invalid_hid() -> Hid {
    INVALID_HID
}

/// Types of internal objects, analogous to HDF5's H5I_type_t.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HandleType {
    File,
    Group,
    Dataset,
    Attribute,
    Datatype,
    Dataspace,
    PropertyList,
}

/// Metadata stored for each registered handle.
struct HandleEntry {
    handle_type: HandleType,
    refcount: i32,
    /// Opaque data associated with this handle.
    data: Arc<dyn std::any::Any + Send + Sync>,
}

/// Global registry of internal handles, replacing HDF5's C-level hid_t system.
pub struct HandleRegistry {
    next_id: AtomicI64,
    entries: RwLock<HashMap<Hid, HandleEntry>>,
    type_refcounts: RwLock<HashMap<HandleType, i32>>,
}

impl HandleRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            next_id: AtomicI64::new(1), // Start at 1, 0 and negative are reserved
            entries: RwLock::new(HashMap::new()),
            type_refcounts: RwLock::new(HashMap::new()),
        }
    }

    /// Invalid handle sentinel.
    pub fn invalid_hid() -> Hid {
        INVALID_HID
    }

    /// Terminate this registry package by clearing all registered handles.
    pub fn term_package(&self) {
        self.entries.write().clear();
        self.type_refcounts.write().clear();
    }

    /// Register a handle type in the local registry metadata.
    pub fn register_type_common(&self, handle_type: HandleType) -> HandleType {
        self.type_refcounts.write().entry(handle_type).or_insert(1);
        handle_type
    }

    /// Register a handle type.
    pub fn register_type(&self, handle_type: HandleType) -> HandleType {
        self.register_type_common(handle_type)
    }

    /// Legacy type-registration alias.
    pub fn register_type_v1(&self, handle_type: HandleType) -> HandleType {
        self.register_type_common(handle_type)
    }

    /// Current type-registration alias.
    pub fn register_type_v2(&self, handle_type: HandleType) -> HandleType {
        self.register_type_common(handle_type)
    }

    /// Return whether a type currently exists in this registry.
    pub fn type_exists(&self, handle_type: HandleType) -> bool {
        self.type_refcounts.read().contains_key(&handle_type)
            || self
                .entries
                .read()
                .values()
                .any(|entry| entry.handle_type == handle_type)
    }

    /// Register a new object and return its handle ID.
    pub fn register<T: Send + Sync + 'static>(&self, handle_type: HandleType, data: T) -> Hid {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let entry = HandleEntry {
            handle_type,
            refcount: 1,
            data: Arc::new(data),
        };
        self.entries.write().insert(id, entry);
        self.register_type_common(handle_type);
        id
    }

    /// Internal register alias.
    pub fn register_internal<T: Send + Sync + 'static>(
        &self,
        handle_type: HandleType,
        data: T,
    ) -> Hid {
        self.register(handle_type, data)
    }

    /// Register an object using an existing id.
    pub fn register_using_existing_id<T: Send + Sync + 'static>(
        &self,
        id: Hid,
        handle_type: HandleType,
        data: T,
    ) -> Option<Hid> {
        if id <= 0 {
            return None;
        }
        let entry = HandleEntry {
            handle_type,
            refcount: 1,
            data: Arc::new(data),
        };
        self.entries.write().insert(id, entry);
        self.register_type_common(handle_type);
        Some(id)
    }

    /// Public existing-id registration alias.
    pub fn register_using_existing_id_api<T: Send + Sync + 'static>(
        &self,
        id: Hid,
        handle_type: HandleType,
        data: T,
    ) -> Option<Hid> {
        self.register_using_existing_id(id, handle_type, data)
    }

    /// Public register alias.
    pub fn register_api<T: Send + Sync + 'static>(&self, handle_type: HandleType, data: T) -> Hid {
        self.register(handle_type, data)
    }

    /// Future register alias. Async execution is not used here.
    pub fn register_future<T: Send + Sync + 'static>(
        &self,
        handle_type: HandleType,
        data: T,
    ) -> Hid {
        self.register(handle_type, data)
    }

    /// Replace the object associated with an id.
    pub fn subst<T: Send + Sync + 'static>(&self, id: Hid, data: T) -> bool {
        let mut entries = self.entries.write();
        if let Some(entry) = entries.get_mut(&id) {
            entry.data = Arc::new(data);
            true
        } else {
            false
        }
    }

    /// Return true if this id is a file object.
    pub fn is_file_object(&self, id: Hid) -> bool {
        self.handle_type(id) == Some(HandleType::File)
    }

    /// Internal unwrap helper returning the id when valid.
    pub fn unwrap_id(&self, id: Hid) -> Option<Hid> {
        self.is_valid(id).then_some(id)
    }

    /// Mark-node helper. This compact registry has no separate mark bit.
    pub fn mark_node(&self, id: Hid) -> bool {
        self.is_valid(id)
    }

    /// Increment the reference count for a handle.
    pub fn incref(&self, id: Hid) -> Option<i32> {
        let mut entries = self.entries.write();
        if let Some(entry) = entries.get_mut(&id) {
            entry.refcount += 1;
            Some(entry.refcount)
        } else {
            None
        }
    }

    /// Public increment-ref alias.
    pub fn inc_ref(&self, id: Hid) -> Option<i32> {
        self.incref(id)
    }

    /// Decrement the reference count for a handle.
    /// Returns the new refcount, or None if the handle was invalid.
    /// When refcount reaches 0, the entry is removed.
    pub fn decref(&self, id: Hid) -> Option<i32> {
        let mut entries = self.entries.write();
        if let Some(entry) = entries.get_mut(&id) {
            entry.refcount -= 1;
            let rc = entry.refcount;
            if rc <= 0 {
                entries.remove(&id);
            }
            Some(rc)
        } else {
            None
        }
    }

    /// Internal decrement-ref alias.
    pub fn dec_ref_internal(&self, id: Hid) -> Option<i32> {
        self.decref(id)
    }

    /// Internal decrement application-ref alias.
    pub fn dec_app_ref_internal(&self, id: Hid) -> Option<i32> {
        self.decref(id)
    }

    /// Public decrement application-ref alias.
    pub fn dec_app_ref(&self, id: Hid) -> Option<i32> {
        self.decref(id)
    }

    /// Async decrement application-ref alias. Async execution is not used here.
    pub fn dec_app_ref_async(&self, id: Hid) -> Option<i32> {
        self.decref(id)
    }

    /// Always-close decrement application-ref alias.
    pub fn dec_app_ref_always_close(&self, id: Hid) -> Option<i32> {
        self.decref(id)
    }

    /// Internal always-close decrement application-ref alias.
    pub fn dec_app_ref_always_close_internal(&self, id: Hid) -> Option<i32> {
        self.decref(id)
    }

    /// Async always-close decrement application-ref alias.
    pub fn dec_app_ref_always_close_async(&self, id: Hid) -> Option<i32> {
        self.decref(id)
    }

    /// Public decrement-ref alias.
    pub fn dec_ref(&self, id: Hid) -> Option<i32> {
        self.decref(id)
    }

    /// Get the current reference count.
    pub fn refcount(&self, id: Hid) -> Option<i32> {
        self.entries.read().get(&id).map(|e| e.refcount)
    }

    /// Public refcount alias.
    pub fn get_ref(&self, id: Hid) -> Option<i32> {
        self.refcount(id)
    }

    /// Get the handle type.
    pub fn handle_type(&self, id: Hid) -> Option<HandleType> {
        self.entries.read().get(&id).map(|e| e.handle_type)
    }

    /// Public handle-type alias.
    pub fn get_type(&self, id: Hid) -> Option<HandleType> {
        self.handle_type(id)
    }

    /// Verify an object by id and expected type.
    pub fn object_verify(&self, id: Hid, expected: HandleType) -> bool {
        self.handle_type(id) == Some(expected)
    }

    /// Check if a handle is valid (exists and has refcount > 0).
    pub fn is_valid(&self, id: Hid) -> bool {
        self.entries
            .read()
            .get(&id)
            .map_or(false, |e| e.refcount > 0)
    }

    /// Public validity alias.
    pub fn is_valid_api(&self, id: Hid) -> bool {
        self.is_valid(id)
    }

    /// Get the data associated with a handle, downcasted to the expected type.
    pub fn get<T: Send + Sync + 'static>(&self, id: Hid) -> Option<Arc<T>> {
        let entries = self.entries.read();
        entries
            .get(&id)
            .and_then(|e| e.data.clone().downcast::<T>().ok())
    }

    /// Remove an id and return whether it existed.
    pub fn remove(&self, id: Hid) -> bool {
        self.entries.write().remove(&id).is_some()
    }

    /// Internal remove-and-verify helper.
    pub fn remove_verify_internal(&self, id: Hid, expected: HandleType) -> bool {
        if self.object_verify(id, expected) {
            self.remove(id)
        } else {
            false
        }
    }

    /// Internal common remove helper.
    pub fn remove_common(&self, id: Hid) -> bool {
        self.remove(id)
    }

    /// Public remove-and-verify helper.
    pub fn remove_verify(&self, id: Hid, expected: HandleType) -> bool {
        self.remove_verify_internal(id, expected)
    }

    /// Clear all handles of a type. Returns the number removed.
    pub fn clear_type(&self, handle_type: HandleType) -> usize {
        let mut entries = self.entries.write();
        let before = entries.len();
        entries.retain(|_, entry| entry.handle_type != handle_type);
        before - entries.len()
    }

    /// Public clear-type alias.
    pub fn clear_type_api(&self, handle_type: HandleType) -> usize {
        self.clear_type(handle_type)
    }

    /// Destroy all handles of a type and remove its type metadata.
    pub fn destroy_type(&self, handle_type: HandleType) -> usize {
        self.type_refcounts.write().remove(&handle_type);
        self.clear_type(handle_type)
    }

    /// Internal destroy-type alias.
    pub fn destroy_type_internal(&self, handle_type: HandleType) -> usize {
        self.destroy_type(handle_type)
    }

    /// Public destroy-type alias.
    pub fn destroy_type_api(&self, handle_type: HandleType) -> usize {
        self.destroy_type(handle_type)
    }

    /// Number of members of a handle type.
    pub fn nmembers(&self, handle_type: HandleType) -> usize {
        self.entries
            .read()
            .values()
            .filter(|entry| entry.handle_type == handle_type)
            .count()
    }

    /// Public number-of-members alias.
    pub fn nmembers_api(&self, handle_type: HandleType) -> usize {
        self.nmembers(handle_type)
    }

    /// Increment a type reference count.
    pub fn inc_type_ref(&self, handle_type: HandleType) -> i32 {
        let mut refs = self.type_refcounts.write();
        let value = refs.entry(handle_type).or_insert(0);
        *value += 1;
        *value
    }

    /// Internal increment type-ref alias.
    pub fn inc_type_ref_internal(&self, handle_type: HandleType) -> i32 {
        self.inc_type_ref(handle_type)
    }

    /// Decrement a type reference count.
    pub fn dec_type_ref(&self, handle_type: HandleType) -> Option<i32> {
        let mut refs = self.type_refcounts.write();
        let value = refs.get_mut(&handle_type)?;
        *value -= 1;
        let out = *value;
        if out <= 0 {
            refs.remove(&handle_type);
        }
        Some(out)
    }

    /// Public no-underscore type-ref decrement alias.
    pub fn dec_type_ref_api(&self, handle_type: HandleType) -> Option<i32> {
        self.dec_type_ref(handle_type)
    }

    /// Get a type reference count.
    pub fn get_type_ref(&self, handle_type: HandleType) -> i32 {
        self.type_refcounts
            .read()
            .get(&handle_type)
            .copied()
            .unwrap_or(0)
    }

    /// Internal get-type-ref alias.
    pub fn get_type_ref_internal(&self, handle_type: HandleType) -> i32 {
        self.get_type_ref(handle_type)
    }

    /// Iterate over ids of a type.
    pub fn iterate<F>(&self, handle_type: HandleType, mut callback: F)
    where
        F: FnMut(Hid),
    {
        let ids: Vec<_> = self
            .entries
            .read()
            .iter()
            .filter_map(|(&id, entry)| (entry.handle_type == handle_type).then_some(id))
            .collect();
        for id in ids {
            callback(id);
        }
    }

    /// Internal iterate callback adapter.
    pub fn iterate_cb<F>(&self, handle_type: HandleType, callback: F)
    where
        F: FnMut(Hid),
    {
        self.iterate(handle_type, callback);
    }

    /// Public iterate callback adapter.
    pub fn iterate_pub_cb<F>(&self, handle_type: HandleType, callback: F)
    where
        F: FnMut(Hid),
    {
        self.iterate(handle_type, callback);
    }

    /// Public iterate alias.
    pub fn iterate_api<F>(&self, handle_type: HandleType, callback: F)
    where
        F: FnMut(Hid),
    {
        self.iterate(handle_type, callback);
    }

    /// Find an id of a type matching a predicate.
    pub fn find_id<F>(&self, handle_type: HandleType, mut predicate: F) -> Option<Hid>
    where
        F: FnMut(Hid) -> bool,
    {
        self.entries.read().iter().find_map(|(&id, entry)| {
            (entry.handle_type == handle_type && predicate(id)).then_some(id)
        })
    }

    /// Internal search callback adapter.
    pub fn search_cb<F>(&self, handle_type: HandleType, predicate: F) -> Option<Hid>
    where
        F: FnMut(Hid) -> bool,
    {
        self.find_id(handle_type, predicate)
    }

    /// Public search alias.
    pub fn search<F>(&self, handle_type: HandleType, predicate: F) -> Option<Hid>
    where
        F: FnMut(Hid) -> bool,
    {
        self.find_id(handle_type, predicate)
    }

    /// Return a file id for file objects.
    pub fn get_file_id(&self, id: Hid) -> Option<Hid> {
        self.is_file_object(id).then_some(id)
    }

    /// Return an implementation-defined object name.
    pub fn get_name(&self, id: Hid) -> Option<String> {
        self.handle_type(id)
            .map(|handle_type| format!("{handle_type:?}:{id}"))
    }

    /// Internal name-test helper.
    pub fn get_name_test(&self, id: Hid) -> Option<String> {
        self.get_name(id)
    }

    /// Dump ids for a type.
    pub fn dump_ids_for_type(&self, handle_type: HandleType) -> Vec<Hid> {
        let mut ids = Vec::new();
        self.iterate(handle_type, |id| ids.push(id));
        ids
    }

    /// Internal dump callback helper.
    pub fn id_dump_cb(&self, handle_type: HandleType) -> Vec<Hid> {
        self.dump_ids_for_type(handle_type)
    }

    /// Number of currently registered handles.
    pub fn len(&self) -> usize {
        self.entries.read().len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.read().is_empty()
    }
}

impl Default for HandleRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Global handle registry instance.
static REGISTRY: parking_lot::Mutex<Option<HandleRegistry>> = parking_lot::Mutex::new(None);

/// Get or initialize the global handle registry.
pub fn global_registry() -> &'static parking_lot::Mutex<Option<HandleRegistry>> {
    // Ensure registry is initialized
    {
        let mut guard = REGISTRY.lock();
        if guard.is_none() {
            *guard = Some(HandleRegistry::new());
        }
    }
    &REGISTRY
}

#[allow(non_snake_case)]
pub fn H5I_INVALID_HID() -> Hid {
    invalid_hid()
}

#[allow(non_snake_case)]
pub fn H5I_init_interface() -> HandleRegistry {
    HandleRegistry::new()
}

#[allow(non_snake_case)]
pub fn H5I_term_package(registry: &HandleRegistry) {
    registry.term_package()
}

#[allow(non_snake_case)]
pub fn H5I__register_type_common(registry: &HandleRegistry, handle_type: HandleType) -> HandleType {
    registry.register_type_common(handle_type)
}

#[allow(non_snake_case)]
pub fn H5I_register_type(registry: &HandleRegistry, handle_type: HandleType) -> HandleType {
    registry.register_type(handle_type)
}

#[allow(non_snake_case)]
pub fn H5Iregister_type1(registry: &HandleRegistry, handle_type: HandleType) -> HandleType {
    registry.register_type_v1(handle_type)
}

#[allow(non_snake_case)]
pub fn H5Iregister_type2(registry: &HandleRegistry, handle_type: HandleType) -> HandleType {
    registry.register_type_v2(handle_type)
}

#[allow(non_snake_case)]
pub fn H5Itype_exists(registry: &HandleRegistry, handle_type: HandleType) -> bool {
    registry.type_exists(handle_type)
}

#[allow(non_snake_case)]
pub fn H5I__register<T: Send + Sync + 'static>(
    registry: &HandleRegistry,
    handle_type: HandleType,
    data: T,
) -> Hid {
    registry.register_internal(handle_type, data)
}

#[allow(non_snake_case)]
pub fn H5I_register<T: Send + Sync + 'static>(
    registry: &HandleRegistry,
    handle_type: HandleType,
    data: T,
) -> Hid {
    registry.register(handle_type, data)
}

#[allow(non_snake_case)]
pub fn H5Iregister<T: Send + Sync + 'static>(
    registry: &HandleRegistry,
    handle_type: HandleType,
    data: T,
) -> Hid {
    registry.register_api(handle_type, data)
}

#[allow(non_snake_case)]
pub fn H5Iregister_future<T: Send + Sync + 'static>(
    registry: &HandleRegistry,
    handle_type: HandleType,
    data: T,
) -> Hid {
    registry.register_future(handle_type, data)
}

#[allow(non_snake_case)]
pub fn H5I_register_using_existing_id<T: Send + Sync + 'static>(
    registry: &HandleRegistry,
    id: Hid,
    handle_type: HandleType,
    data: T,
) -> Option<Hid> {
    registry.register_using_existing_id_api(id, handle_type, data)
}

#[allow(non_snake_case)]
pub fn H5I_subst<T: Send + Sync + 'static>(registry: &HandleRegistry, id: Hid, data: T) -> bool {
    registry.subst(id, data)
}

#[allow(non_snake_case)]
pub fn H5I_is_file_object(registry: &HandleRegistry, id: Hid) -> bool {
    registry.is_file_object(id)
}

#[allow(non_snake_case)]
pub fn H5I__unwrap(registry: &HandleRegistry, id: Hid) -> Option<Hid> {
    registry.unwrap_id(id)
}

#[allow(non_snake_case)]
pub fn H5I__mark_node(registry: &HandleRegistry, id: Hid) -> bool {
    registry.mark_node(id)
}

#[allow(non_snake_case)]
pub fn H5I_inc_ref(registry: &HandleRegistry, id: Hid) -> Option<i32> {
    registry.inc_ref(id)
}

#[allow(non_snake_case)]
pub fn H5Iinc_ref(registry: &HandleRegistry, id: Hid) -> Option<i32> {
    registry.inc_ref(id)
}

#[allow(non_snake_case)]
pub fn H5I__dec_ref(registry: &HandleRegistry, id: Hid) -> Option<i32> {
    registry.dec_ref_internal(id)
}

#[allow(non_snake_case)]
pub fn H5I_dec_ref(registry: &HandleRegistry, id: Hid) -> Option<i32> {
    registry.dec_ref(id)
}

#[allow(non_snake_case)]
pub fn H5Idec_ref(registry: &HandleRegistry, id: Hid) -> Option<i32> {
    registry.dec_ref(id)
}

#[allow(non_snake_case)]
pub fn H5I__dec_app_ref(registry: &HandleRegistry, id: Hid) -> Option<i32> {
    registry.dec_app_ref_internal(id)
}

#[allow(non_snake_case)]
pub fn H5I_dec_app_ref(registry: &HandleRegistry, id: Hid) -> Option<i32> {
    registry.dec_app_ref(id)
}

#[allow(non_snake_case)]
pub fn H5I_dec_app_ref_async(registry: &HandleRegistry, id: Hid) -> Option<i32> {
    registry.dec_app_ref_async(id)
}

#[allow(non_snake_case)]
pub fn H5I__dec_app_ref_always_close(registry: &HandleRegistry, id: Hid) -> Option<i32> {
    registry.dec_app_ref_always_close_internal(id)
}

#[allow(non_snake_case)]
pub fn H5I_dec_app_ref_always_close(registry: &HandleRegistry, id: Hid) -> Option<i32> {
    registry.dec_app_ref_always_close(id)
}

#[allow(non_snake_case)]
pub fn H5I_dec_app_ref_always_close_async(registry: &HandleRegistry, id: Hid) -> Option<i32> {
    registry.dec_app_ref_always_close_async(id)
}

#[allow(non_snake_case)]
pub fn H5I_get_ref(registry: &HandleRegistry, id: Hid) -> Option<i32> {
    registry.get_ref(id)
}

#[allow(non_snake_case)]
pub fn H5Iget_ref(registry: &HandleRegistry, id: Hid) -> Option<i32> {
    registry.get_ref(id)
}

#[allow(non_snake_case)]
pub fn H5I_get_type(registry: &HandleRegistry, id: Hid) -> Option<HandleType> {
    registry.get_type(id)
}

#[allow(non_snake_case)]
pub fn H5Iget_type(registry: &HandleRegistry, id: Hid) -> Option<HandleType> {
    registry.get_type(id)
}

#[allow(non_snake_case)]
pub fn H5Iobject_verify(registry: &HandleRegistry, id: Hid, expected: HandleType) -> bool {
    registry.object_verify(id, expected)
}

#[allow(non_snake_case)]
pub fn H5I_is_valid(registry: &HandleRegistry, id: Hid) -> bool {
    registry.is_valid_api(id)
}

#[allow(non_snake_case)]
pub fn H5Iis_valid(registry: &HandleRegistry, id: Hid) -> bool {
    registry.is_valid_api(id)
}

#[allow(non_snake_case)]
pub fn H5I_object<T: Send + Sync + 'static>(registry: &HandleRegistry, id: Hid) -> Option<Arc<T>> {
    registry.get(id)
}

#[allow(non_snake_case)]
pub fn H5I_remove(registry: &HandleRegistry, id: Hid) -> bool {
    registry.remove(id)
}

#[allow(non_snake_case)]
pub fn H5I__remove_common(registry: &HandleRegistry, id: Hid) -> bool {
    registry.remove_common(id)
}

#[allow(non_snake_case)]
pub fn H5I__remove_verify(registry: &HandleRegistry, id: Hid, expected: HandleType) -> bool {
    registry.remove_verify_internal(id, expected)
}

#[allow(non_snake_case)]
pub fn H5I_remove_verify(registry: &HandleRegistry, id: Hid, expected: HandleType) -> bool {
    registry.remove_verify(id, expected)
}

#[allow(non_snake_case)]
pub fn H5Iremove_verify(registry: &HandleRegistry, id: Hid, expected: HandleType) -> bool {
    registry.remove_verify(id, expected)
}

#[allow(non_snake_case)]
pub fn H5I_clear_type(registry: &HandleRegistry, handle_type: HandleType) -> usize {
    registry.clear_type(handle_type)
}

#[allow(non_snake_case)]
pub fn H5Iclear_type(registry: &HandleRegistry, handle_type: HandleType) -> usize {
    registry.clear_type_api(handle_type)
}

#[allow(non_snake_case)]
pub fn H5I__destroy_type(registry: &HandleRegistry, handle_type: HandleType) -> usize {
    registry.destroy_type_internal(handle_type)
}

#[allow(non_snake_case)]
pub fn H5Idestroy_type(registry: &HandleRegistry, handle_type: HandleType) -> usize {
    registry.destroy_type_api(handle_type)
}

#[allow(non_snake_case)]
pub fn H5I_nmembers(registry: &HandleRegistry, handle_type: HandleType) -> usize {
    registry.nmembers(handle_type)
}

#[allow(non_snake_case)]
pub fn H5Inmembers(registry: &HandleRegistry, handle_type: HandleType) -> usize {
    registry.nmembers_api(handle_type)
}

#[allow(non_snake_case)]
pub fn H5I__inc_type_ref(registry: &HandleRegistry, handle_type: HandleType) -> i32 {
    registry.inc_type_ref_internal(handle_type)
}

#[allow(non_snake_case)]
pub fn H5Iinc_type_ref(registry: &HandleRegistry, handle_type: HandleType) -> i32 {
    registry.inc_type_ref(handle_type)
}

#[allow(non_snake_case)]
pub fn H5I_dec_type_ref(registry: &HandleRegistry, handle_type: HandleType) -> Option<i32> {
    registry.dec_type_ref(handle_type)
}

#[allow(non_snake_case)]
pub fn H5Idec_type_ref(registry: &HandleRegistry, handle_type: HandleType) -> Option<i32> {
    registry.dec_type_ref_api(handle_type)
}

#[allow(non_snake_case)]
pub fn H5I__get_type_ref(registry: &HandleRegistry, handle_type: HandleType) -> i32 {
    registry.get_type_ref_internal(handle_type)
}

#[allow(non_snake_case)]
pub fn H5I_get_type_ref(registry: &HandleRegistry, handle_type: HandleType) -> i32 {
    registry.get_type_ref(handle_type)
}

#[allow(non_snake_case)]
pub fn H5Iget_type_ref(registry: &HandleRegistry, handle_type: HandleType) -> i32 {
    registry.get_type_ref(handle_type)
}

#[allow(non_snake_case)]
pub fn H5I__iterate_cb<F>(registry: &HandleRegistry, handle_type: HandleType, callback: F)
where
    F: FnMut(Hid),
{
    registry.iterate_cb(handle_type, callback)
}

#[allow(non_snake_case)]
pub fn H5I__iterate_pub_cb<F>(registry: &HandleRegistry, handle_type: HandleType, callback: F)
where
    F: FnMut(Hid),
{
    registry.iterate_pub_cb(handle_type, callback)
}

#[allow(non_snake_case)]
pub fn H5I_iterate<F>(registry: &HandleRegistry, handle_type: HandleType, callback: F)
where
    F: FnMut(Hid),
{
    registry.iterate_api(handle_type, callback)
}

#[allow(non_snake_case)]
pub fn H5Iiterate<F>(registry: &HandleRegistry, handle_type: HandleType, callback: F)
where
    F: FnMut(Hid),
{
    registry.iterate(handle_type, callback)
}

#[allow(non_snake_case)]
pub fn H5I_find_id<F>(
    registry: &HandleRegistry,
    handle_type: HandleType,
    predicate: F,
) -> Option<Hid>
where
    F: FnMut(Hid) -> bool,
{
    registry.find_id(handle_type, predicate)
}

#[allow(non_snake_case)]
pub fn H5I__search_cb<F>(
    registry: &HandleRegistry,
    handle_type: HandleType,
    predicate: F,
) -> Option<Hid>
where
    F: FnMut(Hid) -> bool,
{
    registry.search_cb(handle_type, predicate)
}

#[allow(non_snake_case)]
pub fn H5Isearch<F>(registry: &HandleRegistry, handle_type: HandleType, predicate: F) -> Option<Hid>
where
    F: FnMut(Hid) -> bool,
{
    registry.search(handle_type, predicate)
}

#[allow(non_snake_case)]
pub fn H5Iget_file_id(registry: &HandleRegistry, id: Hid) -> Option<Hid> {
    registry.get_file_id(id)
}

#[allow(non_snake_case)]
pub fn H5Iget_name(registry: &HandleRegistry, id: Hid) -> Option<String> {
    registry.get_name(id)
}

#[allow(non_snake_case)]
pub fn H5I__get_name_test(registry: &HandleRegistry, id: Hid) -> Option<String> {
    registry.get_name_test(id)
}

#[allow(non_snake_case)]
pub fn H5I_dump_ids_for_type(registry: &HandleRegistry, handle_type: HandleType) -> Vec<Hid> {
    registry.dump_ids_for_type(handle_type)
}

#[allow(non_snake_case)]
pub fn H5I__id_dump_cb(registry: &HandleRegistry, handle_type: HandleType) -> Vec<Hid> {
    registry.id_dump_cb(handle_type)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_get() {
        let reg = HandleRegistry::new();
        let id = reg.register(HandleType::File, "test_data".to_string());

        assert!(reg.is_valid(id));
        assert_eq!(reg.handle_type(id), Some(HandleType::File));
        assert_eq!(reg.refcount(id), Some(1));

        let data = reg.get::<String>(id).unwrap();
        assert_eq!(&*data, "test_data");
    }

    #[test]
    fn test_refcount() {
        let reg = HandleRegistry::new();
        let id = reg.register(HandleType::Group, 42u32);

        assert_eq!(reg.incref(id), Some(2));
        assert_eq!(reg.incref(id), Some(3));
        assert_eq!(reg.decref(id), Some(2));
        assert_eq!(reg.decref(id), Some(1));
        assert!(reg.is_valid(id));

        // Drop to 0 -- entry removed
        assert_eq!(reg.decref(id), Some(0));
        assert!(!reg.is_valid(id));
    }

    #[test]
    fn test_invalid_handle() {
        let reg = HandleRegistry::new();
        assert!(!reg.is_valid(999));
        assert_eq!(reg.refcount(999), None);
        assert_eq!(reg.incref(999), None);
    }

    #[test]
    fn test_type_registry_and_iteration_aliases() {
        let reg = HandleRegistry::new();
        assert_eq!(HandleRegistry::invalid_hid(), INVALID_HID);
        assert_eq!(reg.register_type(HandleType::Dataset), HandleType::Dataset);
        assert!(reg.type_exists(HandleType::Dataset));
        assert_eq!(reg.get_type_ref(HandleType::Dataset), 1);
        assert_eq!(reg.inc_type_ref(HandleType::Dataset), 2);
        assert_eq!(reg.dec_type_ref_api(HandleType::Dataset), Some(1));

        let a = reg.register_api(HandleType::Dataset, "a");
        let b = reg.register_future(HandleType::Dataset, "b");
        assert_eq!(reg.nmembers(HandleType::Dataset), 2);
        assert_eq!(reg.find_id(HandleType::Dataset, |id| id == b), Some(b));
        assert_eq!(reg.search(HandleType::Dataset, |id| id == a), Some(a));

        let mut seen = Vec::new();
        reg.iterate_api(HandleType::Dataset, |id| seen.push(id));
        seen.sort_unstable();
        assert_eq!(seen, vec![a, b]);
        assert_eq!(reg.dump_ids_for_type(HandleType::Dataset).len(), 2);
        assert_eq!(reg.get_name(a), Some(format!("Dataset:{a}")));

        assert!(reg.subst(a, "replacement"));
        assert_eq!(&*reg.get::<&str>(a).unwrap(), &"replacement");
        assert!(reg.remove_verify(a, HandleType::Dataset));
        assert!(!reg.is_valid(a));
        assert_eq!(reg.clear_type_api(HandleType::Dataset), 1);
        assert_eq!(reg.nmembers_api(HandleType::Dataset), 0);
        reg.term_package();
        assert!(reg.is_empty());
    }

    #[test]
    fn test_existing_id_and_ref_aliases() {
        let reg = HandleRegistry::new();
        assert_eq!(
            reg.register_using_existing_id_api(44, HandleType::File, "file"),
            Some(44)
        );
        assert_eq!(reg.unwrap_id(44), Some(44));
        assert!(reg.mark_node(44));
        assert!(reg.is_file_object(44));
        assert_eq!(reg.get_file_id(44), Some(44));
        assert!(reg.object_verify(44, HandleType::File));
        assert_eq!(reg.get_type(44), Some(HandleType::File));
        assert_eq!(reg.inc_ref(44), Some(2));
        assert_eq!(reg.get_ref(44), Some(2));
        assert_eq!(reg.dec_app_ref(44), Some(1));
        assert_eq!(reg.dec_app_ref_always_close(44), Some(0));
        assert!(!reg.is_valid_api(44));
    }

    #[test]
    fn h5i_aliases_cover_registry_surface() {
        let reg = H5I_init_interface();
        assert_eq!(H5I_INVALID_HID(), INVALID_HID);
        assert_eq!(
            H5I__register_type_common(&reg, HandleType::Dataset),
            HandleType::Dataset
        );
        assert_eq!(
            H5I_register_type(&reg, HandleType::Group),
            HandleType::Group
        );
        assert_eq!(
            H5Iregister_type1(&reg, HandleType::Attribute),
            HandleType::Attribute
        );
        assert_eq!(
            H5Iregister_type2(&reg, HandleType::Datatype),
            HandleType::Datatype
        );
        assert!(H5Itype_exists(&reg, HandleType::Dataset));
        assert_eq!(H5I__inc_type_ref(&reg, HandleType::Dataset), 2);
        assert_eq!(H5Iinc_type_ref(&reg, HandleType::Dataset), 3);
        assert_eq!(H5I__get_type_ref(&reg, HandleType::Dataset), 3);
        assert_eq!(H5I_get_type_ref(&reg, HandleType::Dataset), 3);
        assert_eq!(H5Iget_type_ref(&reg, HandleType::Dataset), 3);
        assert_eq!(H5I_dec_type_ref(&reg, HandleType::Dataset), Some(2));
        assert_eq!(H5Idec_type_ref(&reg, HandleType::Dataset), Some(1));

        let file = H5I_register(&reg, HandleType::File, "file");
        let group = H5Iregister(&reg, HandleType::Group, "group");
        let dset = H5Iregister_future(&reg, HandleType::Dataset, "future");
        let attr = H5I__register(&reg, HandleType::Attribute, "attr");
        assert_eq!(
            H5I_register_using_existing_id(&reg, 44, HandleType::File, "existing"),
            Some(44)
        );
        assert!(H5I_subst(&reg, dset, "replacement"));
        assert_eq!(&*H5I_object::<&str>(&reg, dset).unwrap(), &"replacement");

        assert!(H5I_is_file_object(&reg, file));
        assert_eq!(H5I__unwrap(&reg, file), Some(file));
        assert!(H5I__mark_node(&reg, file));
        assert_eq!(H5Iget_file_id(&reg, file), Some(file));
        assert_eq!(H5I_get_type(&reg, group), Some(HandleType::Group));
        assert_eq!(H5Iget_type(&reg, group), Some(HandleType::Group));
        assert!(H5Iobject_verify(&reg, attr, HandleType::Attribute));
        assert!(H5I_is_valid(&reg, attr));
        assert!(H5Iis_valid(&reg, attr));

        assert_eq!(H5I_inc_ref(&reg, file), Some(2));
        assert_eq!(H5Iinc_ref(&reg, file), Some(3));
        assert_eq!(H5I_get_ref(&reg, file), Some(3));
        assert_eq!(H5Iget_ref(&reg, file), Some(3));
        assert_eq!(H5I__dec_ref(&reg, file), Some(2));
        assert_eq!(H5I_dec_ref(&reg, file), Some(1));

        let close_async = H5I_register(&reg, HandleType::File, "close_async");
        assert_eq!(H5I_dec_app_ref_async(&reg, close_async), Some(0));
        let close = H5I_register(&reg, HandleType::File, "close");
        assert_eq!(H5I__dec_app_ref(&reg, close), Some(0));
        let close_always = H5I_register(&reg, HandleType::File, "close_always");
        assert_eq!(H5I_dec_app_ref_always_close(&reg, close_always), Some(0));
        let close_always_internal = H5I_register(&reg, HandleType::File, "close_always_internal");
        assert_eq!(
            H5I__dec_app_ref_always_close(&reg, close_always_internal),
            Some(0)
        );
        let close_always_async = H5I_register(&reg, HandleType::File, "close_always_async");
        assert_eq!(
            H5I_dec_app_ref_always_close_async(&reg, close_always_async),
            Some(0)
        );

        assert_eq!(H5I_nmembers(&reg, HandleType::Group), 1);
        assert_eq!(H5Inmembers(&reg, HandleType::Group), 1);
        assert_eq!(
            H5I_find_id(&reg, HandleType::Group, |id| id == group),
            Some(group)
        );
        assert_eq!(
            H5I__search_cb(&reg, HandleType::Group, |id| id == group),
            Some(group)
        );
        assert_eq!(
            H5Isearch(&reg, HandleType::Group, |id| id == group),
            Some(group)
        );

        let mut iterated = Vec::new();
        H5I__iterate_cb(&reg, HandleType::Group, |id| iterated.push(id));
        H5I__iterate_pub_cb(&reg, HandleType::Group, |id| iterated.push(id));
        H5I_iterate(&reg, HandleType::Group, |id| iterated.push(id));
        H5Iiterate(&reg, HandleType::Group, |id| iterated.push(id));
        assert_eq!(iterated, vec![group, group, group, group]);
        assert_eq!(H5I_dump_ids_for_type(&reg, HandleType::Group), vec![group]);
        assert_eq!(H5I__id_dump_cb(&reg, HandleType::Group), vec![group]);
        assert_eq!(H5Iget_name(&reg, group), Some(format!("Group:{group}")));
        assert_eq!(
            H5I__get_name_test(&reg, group),
            Some(format!("Group:{group}"))
        );

        assert!(H5I__remove_verify(&reg, attr, HandleType::Attribute));
        assert!(H5I_remove_verify(&reg, dset, HandleType::Dataset));
        assert!(H5Iremove_verify(&reg, 44, HandleType::File));
        let common = H5I_register(&reg, HandleType::Datatype, "common");
        assert!(H5I__remove_common(&reg, common));
        let removed = H5I_register(&reg, HandleType::Datatype, "removed");
        assert!(H5I_remove(&reg, removed));
        assert_eq!(H5I_clear_type(&reg, HandleType::File), 1);
        assert_eq!(H5Iclear_type(&reg, HandleType::Group), 1);
        assert_eq!(H5I__destroy_type(&reg, HandleType::Datatype), 0);
        assert_eq!(H5Idestroy_type(&reg, HandleType::Dataset), 0);
        H5I_term_package(&reg);
        assert!(reg.is_empty());
    }
}
