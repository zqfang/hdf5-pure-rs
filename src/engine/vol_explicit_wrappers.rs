#![allow(dead_code, non_snake_case)]

use super::{
    H5VL__conn_free, H5VL__get_connector_by_name, H5VL__get_connector_by_value,
    H5VL__native_attr_create, H5VL__native_attr_open, H5VL__native_blob_put,
    H5VL__native_blob_read_into, H5VL__native_blob_specific, H5VL__native_blob_view,
    H5VL__native_blob_visit_chunks, H5VL__register_connector_by_name,
    H5VL__register_connector_by_value, H5VL__set_def_conn, H5VL__wrap_obj, H5VL_conn_dec_rc,
    H5VL_conn_inc_rc, H5VL_new_vol_obj, H5VL_object_unwrap, H5VL_pass_through_get_wrap_ctx,
    H5VL_restore_lib_state, H5VL_retrieve_lib_state, H5VL_wrap_register, VolConnector, VolLibState,
    VolObject, VolRegistry,
};
use crate::error::{Error, Result};
use std::fmt;

/// Build an [`Error::Unsupported`] for a VOL operation not implemented by the pure-Rust backend.
fn unsupported_vol(name: &str) -> Error {
    Error::Unsupported(format!(
        "{name} requires a concrete VOL connector operation that is not implemented"
    ))
}

/// Passthrough VOL wrapper: create a attr.
pub fn H5VL_pass_through_attr_create(parent: &VolObject, name: &str) -> VolObject {
    H5VL__native_attr_create(parent, name)
}

/// Passthrough VOL wrapper: open a attr.
pub fn H5VL_pass_through_attr_open(parent: &VolObject, name: &str) -> VolObject {
    H5VL__native_attr_open(parent, name)
}

/// Passthrough VOL wrapper: write to a attr.
pub fn H5VL_pass_through_attr_write(object: &mut VolObject, data: &[u8]) {
    H5VL__native_blob_put(object, data);
}

/// Passthrough VOL wrapper: borrow attr bytes.
pub fn H5VL_pass_through_attr_view(object: &VolObject) -> &[u8] {
    H5VL__native_blob_view(object)
}

/// Passthrough VOL wrapper: copy attr bytes into a caller-owned buffer.
pub fn H5VL_pass_through_attr_read_into(object: &VolObject, dst: &mut [u8]) -> Result<usize> {
    H5VL__native_blob_read_into(object, dst)
}

/// Passthrough VOL wrapper: visit attr bytes without allocating.
pub fn H5VL_pass_through_attr_visit_chunks(
    object: &VolObject,
    chunk_size: usize,
    visit: impl FnMut(&[u8]) -> Result<()>,
) -> Result<()> {
    H5VL__native_blob_visit_chunks(object, chunk_size, visit)
}

/// Passthrough VOL wrapper: invoke a connector-specific op on a attr.
pub fn H5VL_pass_through_attr_specific() -> Result<()> {
    Err(unsupported_vol("H5VL_pass_through_attr_specific"))
}

/// Passthrough VOL wrapper: create a dataset.
pub fn H5VL_pass_through_dataset_create(parent: &VolObject, name: &str) -> VolObject {
    H5VL_new_vol_obj(parent.connector_id, name)
}

/// Passthrough VOL wrapper: borrow dataset bytes.
pub fn H5VL_pass_through_dataset_view(object: &VolObject) -> &[u8] {
    H5VL__native_blob_view(object)
}

/// Passthrough VOL wrapper: copy dataset bytes into a caller-owned buffer.
pub fn H5VL_pass_through_dataset_read_into(object: &VolObject, dst: &mut [u8]) -> Result<usize> {
    H5VL__native_blob_read_into(object, dst)
}

/// Passthrough VOL wrapper: visit dataset bytes without allocating.
pub fn H5VL_pass_through_dataset_visit_chunks(
    object: &VolObject,
    chunk_size: usize,
    visit: impl FnMut(&[u8]) -> Result<()>,
) -> Result<()> {
    H5VL__native_blob_visit_chunks(object, chunk_size, visit)
}

/// Passthrough VOL wrapper: write to a dataset.
pub fn H5VL_pass_through_dataset_write(object: &mut VolObject, data: &[u8]) {
    H5VL__native_blob_put(object, data);
}

/// Passthrough VOL wrapper: get information about a dataset.
pub fn H5VL_pass_through_dataset_get(object: &VolObject) -> usize {
    object.payload.len()
}

/// Passthrough VOL wrapper: invoke a connector-specific op on a dataset.
pub fn H5VL_pass_through_dataset_specific() -> Result<()> {
    Err(unsupported_vol("H5VL_pass_through_dataset_specific"))
}

/// Passthrough VOL wrapper: invoke an optional op on a dataset.
pub fn H5VL_pass_through_dataset_optional() -> Result<()> {
    Err(unsupported_vol("H5VL_pass_through_dataset_optional"))
}

/// Passthrough VOL wrapper: close a dataset.
pub fn H5VL_pass_through_dataset_close(_object: VolObject) {}

/// Passthrough VOL wrapper: commit a datatype.
pub fn H5VL_pass_through_datatype_commit(parent: &VolObject, name: &str) -> VolObject {
    H5VL_new_vol_obj(parent.connector_id, name)
}

/// Passthrough VOL wrapper: open a datatype.
pub fn H5VL_pass_through_datatype_open(parent: &VolObject, name: &str) -> VolObject {
    H5VL_new_vol_obj(parent.connector_id, name)
}

/// Passthrough VOL wrapper: get information about a datatype.
pub fn H5VL_pass_through_datatype_get(object: &VolObject) -> usize {
    object.payload.len()
}

/// Passthrough VOL wrapper: invoke a connector-specific op on a datatype.
pub fn H5VL_pass_through_datatype_specific() -> Result<()> {
    Err(unsupported_vol("H5VL_pass_through_datatype_specific"))
}

/// Passthrough VOL wrapper: invoke an optional op on a datatype.
pub fn H5VL_pass_through_datatype_optional() -> Result<()> {
    Err(unsupported_vol("H5VL_pass_through_datatype_optional"))
}

/// Passthrough VOL wrapper: close a datatype.
pub fn H5VL_pass_through_datatype_close(_object: VolObject) {}

/// Passthrough VOL wrapper: create a file.
pub fn H5VL_pass_through_file_create(connector_id: u64, name: &str) -> VolObject {
    H5VL_new_vol_obj(connector_id, name)
}

/// Passthrough VOL wrapper: open a file.
pub fn H5VL_pass_through_file_open(connector_id: u64, name: &str) -> VolObject {
    H5VL_new_vol_obj(connector_id, name)
}

/// Passthrough VOL wrapper: get information about a file.
pub fn H5VL_pass_through_file_get(object: &VolObject) -> &str {
    &object.name
}

/// Passthrough VOL wrapper: invoke a connector-specific op on a file.
pub fn H5VL_pass_through_file_specific() -> Result<()> {
    Err(unsupported_vol("H5VL_pass_through_file_specific"))
}

/// Passthrough VOL wrapper: invoke an optional op on a file.
pub fn H5VL_pass_through_file_optional() -> Result<()> {
    Err(unsupported_vol("H5VL_pass_through_file_optional"))
}

/// Passthrough VOL wrapper: close a file.
pub fn H5VL_pass_through_file_close(_object: VolObject) {}

/// Passthrough VOL wrapper: create a group.
pub fn H5VL_pass_through_group_create(parent: &VolObject, name: &str) -> VolObject {
    H5VL_new_vol_obj(parent.connector_id, name)
}

/// Passthrough VOL wrapper: open a group.
pub fn H5VL_pass_through_group_open(parent: &VolObject, name: &str) -> VolObject {
    H5VL_new_vol_obj(parent.connector_id, name)
}

/// Passthrough VOL wrapper: get information about a group.
pub fn H5VL_pass_through_group_get(object: &VolObject) -> &str {
    &object.name
}

/// Passthrough VOL wrapper: invoke a connector-specific op on a group.
pub fn H5VL_pass_through_group_specific() -> Result<()> {
    Err(unsupported_vol("H5VL_pass_through_group_specific"))
}

/// Passthrough VOL wrapper: invoke an optional op on a group.
pub fn H5VL_pass_through_group_optional() -> Result<()> {
    Err(unsupported_vol("H5VL_pass_through_group_optional"))
}

/// Passthrough VOL wrapper: close a group.
pub fn H5VL_pass_through_group_close(object: VolObject) -> Result<()> {
    H5VLgroup_close(object);
    Ok(())
}

/// Passthrough VOL wrapper: copy a link.
pub fn H5VL_pass_through_link_copy(_src: &VolObject, _dst: &mut VolObject) -> Result<()> {
    Err(unsupported_vol("H5VL_pass_through_link_copy"))
}

/// Passthrough VOL wrapper: move a link.
pub fn H5VL_pass_through_link_move(_src: &mut VolObject, _dst: &mut VolObject) -> Result<()> {
    Err(unsupported_vol("H5VL_pass_through_link_move"))
}

/// Passthrough VOL wrapper: get information about a link.
pub fn H5VL_pass_through_link_get(object: &VolObject) -> &str {
    &object.name
}

/// Passthrough VOL wrapper: invoke a connector-specific op on a link.
pub fn H5VL_pass_through_link_specific() -> Result<()> {
    Err(unsupported_vol("H5VL_pass_through_link_specific"))
}

/// Passthrough VOL wrapper: invoke an optional op on a link.
pub fn H5VL_pass_through_link_optional() -> Result<()> {
    Err(unsupported_vol("H5VL_pass_through_link_optional"))
}

/// Passthrough VOL wrapper: open a object.
pub fn H5VL_pass_through_object_open(parent: &VolObject, name: &str) -> VolObject {
    H5VL_new_vol_obj(parent.connector_id, name)
}

/// Passthrough VOL wrapper: copy a object.
pub fn H5VL_pass_through_object_copy_ref(object: &VolObject) -> &VolObject {
    object
}

/// Passthrough VOL wrapper: copy a object into caller-owned storage.
pub fn H5VL_pass_through_object_copy_into(object: &VolObject, dst: &mut VolObject) {
    dst.clone_from(H5VL_pass_through_object_copy_ref(object));
}

/// Passthrough VOL wrapper: copy a object.
pub fn H5VL_pass_through_object_copy(object: &VolObject) -> VolObject {
    let mut copied = VolObject::default();
    H5VL_pass_through_object_copy_into(object, &mut copied);
    copied
}

/// Passthrough VOL wrapper: get information about a object.
pub fn H5VL_pass_through_object_get(object: &VolObject) -> &str {
    &object.name
}

/// Passthrough VOL wrapper: invoke a connector-specific op on a object.
pub fn H5VL_pass_through_object_specific() -> Result<()> {
    Err(unsupported_vol("H5VL_pass_through_object_specific"))
}

/// Passthrough VOL wrapper: invoke an optional op on a object.
pub fn H5VL_pass_through_object_optional() -> Result<()> {
    Err(unsupported_vol("H5VL_pass_through_object_optional"))
}

/// Return the connector class via the passthrough.
pub fn H5VL_pass_through_introspect_get_conn_cls_ref(connector: &VolConnector) -> &VolConnector {
    connector
}

/// Return the connector class via the passthrough into caller-owned storage.
pub fn H5VL_pass_through_introspect_get_conn_cls_into(
    connector: &VolConnector,
    dst: &mut VolConnector,
) {
    dst.clone_from(H5VL_pass_through_introspect_get_conn_cls_ref(connector));
}

/// Return the connector class via the passthrough.
pub fn H5VL_pass_through_introspect_get_conn_cls(connector: &VolConnector) -> VolConnector {
    let mut copied = VolConnector::default();
    H5VL_pass_through_introspect_get_conn_cls_into(connector, &mut copied);
    copied
}

/// Return the connector capability flags via the passthrough.
pub fn H5VL_pass_through_introspect_get_cap_flags(connector: &VolConnector) -> u64 {
    connector.cap_flags
}

/// Passthrough VOL wrapper: opt query a introspect.
pub fn H5VL_pass_through_introspect_opt_query() -> Result<()> {
    Err(unsupported_vol("H5VL_pass_through_introspect_opt_query"))
}

/// Passthrough VOL wrapper: wait on a request.
pub fn H5VL_pass_through_request_wait() -> Result<()> {
    Err(unsupported_vol("H5VL_pass_through_request_wait"))
}

/// Passthrough VOL wrapper: notify a request.
pub fn H5VL_pass_through_request_notify() -> Result<()> {
    Err(unsupported_vol("H5VL_pass_through_request_notify"))
}

/// Passthrough VOL wrapper: cancel a request.
pub fn H5VL_pass_through_request_cancel() -> Result<()> {
    Err(unsupported_vol("H5VL_pass_through_request_cancel"))
}

/// Passthrough VOL wrapper: invoke a connector-specific op on a request.
pub fn H5VL_pass_through_request_specific() -> Result<()> {
    Err(unsupported_vol("H5VL_pass_through_request_specific"))
}

/// Passthrough VOL wrapper: invoke an optional op on a request.
pub fn H5VL_pass_through_request_optional() -> Result<()> {
    Err(unsupported_vol("H5VL_pass_through_request_optional"))
}

/// Passthrough VOL wrapper: free a request.
pub fn H5VL_pass_through_request_free() {}

/// Passthrough VOL wrapper: put data into a blob.
pub fn H5VL_pass_through_blob_put(object: &mut VolObject, data: &[u8]) {
    H5VL__native_blob_put(object, data);
}

/// Passthrough VOL wrapper: borrow blob bytes.
pub fn H5VL_pass_through_blob_view(object: &VolObject) -> &[u8] {
    H5VL__native_blob_view(object)
}

/// Passthrough VOL wrapper: copy blob bytes into a caller-owned buffer.
pub fn H5VL_pass_through_blob_read_into(object: &VolObject, dst: &mut [u8]) -> Result<usize> {
    H5VL__native_blob_read_into(object, dst)
}

/// Passthrough VOL wrapper: visit blob bytes without allocating.
pub fn H5VL_pass_through_blob_visit_chunks(
    object: &VolObject,
    chunk_size: usize,
    visit: impl FnMut(&[u8]) -> Result<()>,
) -> Result<()> {
    H5VL__native_blob_visit_chunks(object, chunk_size, visit)
}

/// Passthrough VOL wrapper: invoke a connector-specific op on a blob.
pub fn H5VL_pass_through_blob_specific(object: &VolObject) -> usize {
    H5VL__native_blob_specific(object)
}

/// Passthrough VOL wrapper: invoke an optional op on a blob.
pub fn H5VL_pass_through_blob_optional() -> Result<()> {
    Err(unsupported_vol("H5VL_pass_through_blob_optional"))
}

/// Compare two VOL tokens.
pub fn H5VL_pass_through_token_cmp(left: u64, right: u64) -> std::cmp::Ordering {
    left.cmp(&right)
}

/// Render a VOL token into a caller-owned formatter.
pub fn H5VL_pass_through_token_fmt(token: u64, dst: &mut impl fmt::Write) -> fmt::Result {
    write!(dst, "{token}")
}

/// Parse a VOL token from its string representation.
pub fn H5VL_pass_through_token_from_str(token: &str) -> Result<u64> {
    token
        .parse()
        .map_err(|_| Error::InvalidFormat("invalid VOL token string".into()))
}

/// VOL passthrough wrapper for `optional`.
pub fn H5VL_pass_through_optional() -> Result<()> {
    Err(unsupported_vol("H5VL_pass_through_optional"))
}

/// Register a VOL connector by name in the registry.
pub fn H5VLregister_connector(registry: &mut VolRegistry, name: &str, value: u64) -> u64 {
    H5VL__register_connector_by_name(registry, name, value)
}

/// Register a VOL connector by name in the registry.
pub fn H5VLregister_connector_by_name(registry: &mut VolRegistry, name: &str, value: u64) -> u64 {
    H5VL__register_connector_by_name(registry, name, value)
}

/// Register a VOL connector at the given id and value.
pub fn H5VLregister_connector_by_value(
    registry: &mut VolRegistry,
    id: u64,
    name: &str,
    value: u64,
) -> u64 {
    H5VL__register_connector_by_value(registry, id, name, value)
}

/// Look up a connector id by name.
pub fn H5VLget_connector_id_by_name(registry: &VolRegistry, name: &str) -> Option<u64> {
    H5VL__get_connector_by_name(registry, name).map(|connector| connector.id)
}

/// Look up a connector id by value.
pub fn H5VLget_connector_id_by_value(registry: &VolRegistry, value: u64) -> Option<u64> {
    H5VL__get_connector_by_value(registry, value).map(|connector| connector.id)
}

/// Borrow the name of a connector by id.
pub fn H5VLget_connector_name_view(registry: &VolRegistry, id: u64) -> Option<&str> {
    registry
        .connectors
        .get(&id)
        .map(|connector| connector.name.as_str())
}

/// Write the name of a connector by id into a caller-owned formatter.
pub fn H5VLget_connector_name_fmt(
    registry: &VolRegistry,
    id: u64,
    dst: &mut impl fmt::Write,
) -> Result<bool> {
    let Some(name) = H5VLget_connector_name_view(registry, id) else {
        return Ok(false);
    };
    dst.write_str(name)
        .map_err(|_| Error::InvalidFormat("failed to format VOL connector name".into()))?;
    Ok(true)
}

/// Look up the name of a connector by id.
#[deprecated(note = "use H5VLget_connector_name_view or H5VLget_connector_name_fmt")]
pub fn H5VLget_connector_name(registry: &VolRegistry, id: u64) -> Option<String> {
    H5VLget_connector_name_view(registry, id).map(str::to_owned)
}

/// Close (release) a VOL connector by id.
pub fn H5VLclose(registry: &mut VolRegistry, id: u64) {
    H5VL__conn_free(registry, id);
}

/// Unregister a VOL connector from the registry.
pub fn H5VLunregister_connector(registry: &mut VolRegistry, id: u64) {
    H5VL__conn_free(registry, id);
}

/// Compare two connector classes by name then value.
pub fn H5VLcmp_connector_cls(left: &VolConnector, right: &VolConnector) -> std::cmp::Ordering {
    left.name
        .cmp(&right.name)
        .then(left.value.cmp(&right.value))
}

/// Mark a VOL object as wrap-registered.
pub fn H5VLwrap_register(object: &mut VolObject) {
    H5VL_wrap_register(object);
}

/// Snapshot the library state for cross-thread use.
pub fn H5VLretrieve_lib_state(registry: &VolRegistry) -> VolLibState {
    H5VL_retrieve_lib_state(registry)
}

/// Restore a previously snapshotted library state.
pub fn H5VLrestore_lib_state(registry: &mut VolRegistry, state: &VolLibState) {
    H5VL_restore_lib_state(registry, state);
}

/// Free a previously snapshotted library state.
pub fn H5VLfree_lib_state(_state: VolLibState) {}

/// Native VOL wrapper: create a link.
pub fn H5VL__native_link_create() -> Result<()> {
    Err(unsupported_vol("H5VL__native_link_create"))
}

/// Native VOL wrapper: copy a link.
pub fn H5VL__native_link_copy() -> Result<()> {
    Err(unsupported_vol("H5VL__native_link_copy"))
}

/// Native VOL wrapper: move a link.
pub fn H5VL__native_link_move() -> Result<()> {
    Err(unsupported_vol("H5VL__native_link_move"))
}

/// Native VOL wrapper: get information about a link.
pub fn H5VL__native_link_get() -> Result<()> {
    Err(unsupported_vol("H5VL__native_link_get"))
}

/// Native VOL wrapper: invoke a connector-specific op on a link.
pub fn H5VL__native_link_specific() -> Result<()> {
    Err(unsupported_vol("H5VL__native_link_specific"))
}

/// Return the connector class for the native VOL.
pub fn H5VL__native_introspect_get_conn_cls_ref(connector: &VolConnector) -> &VolConnector {
    connector
}

/// Return the connector class for the native VOL into caller-owned storage.
pub fn H5VL__native_introspect_get_conn_cls_into(connector: &VolConnector, dst: &mut VolConnector) {
    dst.clone_from(H5VL__native_introspect_get_conn_cls_ref(connector));
}

/// Return the connector class for the native VOL.
pub fn H5VL__native_introspect_get_conn_cls(connector: &VolConnector) -> VolConnector {
    let mut copied = VolConnector::default();
    H5VL__native_introspect_get_conn_cls_into(connector, &mut copied);
    copied
}

/// Return the capability flags for the native VOL.
pub fn H5VL__native_introspect_get_cap_flags(connector: &VolConnector) -> u64 {
    connector.cap_flags
}

/// Return the address length used by the native VOL (8 bytes).
pub fn H5VL_native_get_file_addr_len() -> usize {
    std::mem::size_of::<u64>()
}

/// Return the address length used by the native VOL (8 bytes).
pub fn H5VL__native_get_file_addr_len() -> usize {
    H5VL_native_get_file_addr_len()
}

/// Native VOL wrapper: open a object.
pub fn H5VL__native_object_open(parent: &VolObject, name: &str) -> VolObject {
    H5VL_new_vol_obj(parent.connector_id, name)
}

/// Native VOL wrapper: copy a object.
pub fn H5VL__native_object_copy_ref(object: &VolObject) -> &VolObject {
    object
}

/// Native VOL wrapper: copy a object into caller-owned storage.
pub fn H5VL__native_object_copy_into(object: &VolObject, dst: &mut VolObject) {
    dst.clone_from(H5VL__native_object_copy_ref(object));
}

/// Native VOL wrapper: copy a object.
pub fn H5VL__native_object_copy(object: &VolObject) -> VolObject {
    let mut copied = VolObject::default();
    H5VL__native_object_copy_into(object, &mut copied);
    copied
}

/// Native VOL wrapper: get information about a object.
pub fn H5VL__native_object_get(object: &VolObject) -> &str {
    &object.name
}

/// Native VOL wrapper: invoke a connector-specific op on a object.
pub fn H5VL__native_object_specific() -> Result<()> {
    Err(unsupported_vol("H5VL__native_object_specific"))
}

/// Native VOL wrapper: invoke an optional op on a object.
pub fn H5VL__native_object_optional() -> Result<()> {
    Err(unsupported_vol("H5VL__native_object_optional"))
}

/// Native VOL wrapper: create a group.
pub fn H5VL__native_group_create(parent: &VolObject, name: &str) -> VolObject {
    H5VL_new_vol_obj(parent.connector_id, name)
}

/// Native VOL wrapper: open a group.
pub fn H5VL__native_group_open(parent: &VolObject, name: &str) -> VolObject {
    H5VL_new_vol_obj(parent.connector_id, name)
}

/// Native VOL wrapper: get information about a group.
pub fn H5VL__native_group_get(object: &VolObject) -> &str {
    &object.name
}

/// Native VOL wrapper: invoke a connector-specific op on a group.
pub fn H5VL__native_group_specific() -> Result<()> {
    Err(unsupported_vol("H5VL__native_group_specific"))
}

/// Native VOL wrapper: invoke an optional op on a group.
pub fn H5VL__native_group_optional() -> Result<()> {
    Err(unsupported_vol("H5VL__native_group_optional"))
}

/// Native VOL wrapper: close a group.
pub fn H5VL__native_group_close(_object: VolObject) {}

/// Compare two native VOL tokens.
pub fn H5VL__native_token_cmp(left: u64, right: u64) -> std::cmp::Ordering {
    left.cmp(&right)
}

/// Render a native VOL token into a caller-owned formatter.
pub fn H5VL__native_token_fmt(token: u64, dst: &mut impl fmt::Write) -> fmt::Result {
    H5VL_pass_through_token_fmt(token, dst)
}

/// Parse a native VOL token from its string representation.
pub fn H5VL__native_str_to_token(token: &str) -> Result<u64> {
    H5VL_pass_through_token_from_str(token)
}

/// Return the capability flags of a VOL connector.
pub fn H5VLget_cap_flags(connector: &VolConnector) -> u64 {
    connector.cap_flags
}

/// Return the numeric value of a VOL connector.
pub fn H5VLget_value(connector: &VolConnector) -> u64 {
    connector.value
}

/// Free a wrap context (no-op in the Rust backend).
pub fn H5VLfree_wrap_ctx(_wrapped: bool) {}

/// VOL wrapper: create a attr.
pub fn H5VLattr_create(parent: &VolObject, name: &str) -> VolObject {
    H5VL__native_attr_create(parent, name)
}

/// VOL wrapper: open a attr.
pub fn H5VLattr_open(parent: &VolObject, name: &str) -> VolObject {
    H5VL__native_attr_open(parent, name)
}

/// VOL wrapper: write to a attr.
pub fn H5VLattr_write(object: &mut VolObject, data: &[u8]) {
    H5VL__native_blob_put(object, data);
}

/// VOL wrapper: borrow attr bytes.
pub fn H5VLattr_view(object: &VolObject) -> &[u8] {
    H5VL__native_blob_view(object)
}

/// VOL wrapper: copy attr bytes into a caller-owned buffer.
pub fn H5VLattr_read_into(object: &VolObject, dst: &mut [u8]) -> Result<usize> {
    H5VL__native_blob_read_into(object, dst)
}

/// VOL wrapper: visit attr bytes without allocating.
pub fn H5VLattr_visit_chunks(
    object: &VolObject,
    chunk_size: usize,
    visit: impl FnMut(&[u8]) -> Result<()>,
) -> Result<()> {
    H5VL__native_blob_visit_chunks(object, chunk_size, visit)
}

/// Internal VOL wrapper: invoke a connector-specific op on a attr.
pub fn H5VL__attr_specific() -> Result<()> {
    Err(unsupported_vol("H5VL__attr_specific"))
}

/// VOL wrapper: invoke a connector-specific op on a attr.
pub fn H5VL_attr_specific() -> Result<()> {
    H5VL__attr_specific()
}

/// VOL wrapper: invoke a connector-specific op on a attr.
pub fn H5VLattr_specific() -> Result<()> {
    H5VL__attr_specific()
}

/// Internal VOL wrapper: invoke an optional op on a attr.
pub fn H5VL__attr_optional() -> Result<()> {
    Err(unsupported_vol("H5VL__attr_optional"))
}

/// VOL wrapper: invoke an optional op on a attr.
pub fn H5VLattr_optional() -> Result<()> {
    H5VL__attr_optional()
}

/// VOL wrapper: invoke an optional op on a attr.
pub fn H5VLattr_optional_op() -> Result<()> {
    H5VL__attr_optional()
}

/// VOL wrapper: close a attr.
pub fn H5VLattr_close(_object: VolObject) {}

/// VOL wrapper: create a dataset.
pub fn H5VLdataset_create(parent: &VolObject, name: &str) -> VolObject {
    H5VL_new_vol_obj(parent.connector_id, name)
}

/// VOL wrapper: open a dataset.
pub fn H5VLdataset_open(parent: &VolObject, name: &str) -> VolObject {
    H5VL_new_vol_obj(parent.connector_id, name)
}

/// VOL wrapper: borrow dataset bytes.
pub fn H5VLdataset_view(object: &VolObject) -> &[u8] {
    H5VL__native_blob_view(object)
}

/// VOL wrapper: copy dataset bytes into a caller-owned buffer.
pub fn H5VLdataset_read_into(object: &VolObject, dst: &mut [u8]) -> Result<usize> {
    H5VL__native_blob_read_into(object, dst)
}

/// VOL wrapper: visit dataset bytes without allocating.
pub fn H5VLdataset_visit_chunks(
    object: &VolObject,
    chunk_size: usize,
    visit: impl FnMut(&[u8]) -> Result<()>,
) -> Result<()> {
    H5VL__native_blob_visit_chunks(object, chunk_size, visit)
}

/// VOL wrapper: write to a dataset.
pub fn H5VLdataset_write(object: &mut VolObject, data: &[u8]) {
    H5VL__native_blob_put(object, data);
}

/// VOL wrapper: get information about a dataset.
pub fn H5VLdataset_get(object: &VolObject) -> usize {
    object.payload.len()
}

/// Internal VOL wrapper: invoke a connector-specific op on a dataset.
pub fn H5VL__dataset_specific() -> Result<()> {
    Err(unsupported_vol("H5VL__dataset_specific"))
}

/// VOL wrapper: invoke a connector-specific op on a dataset.
pub fn H5VL_dataset_specific() -> Result<()> {
    H5VL__dataset_specific()
}

/// VOL wrapper: invoke a connector-specific op on a dataset.
pub fn H5VLdataset_specific() -> Result<()> {
    H5VL__dataset_specific()
}

/// Internal VOL wrapper: invoke an optional op on a dataset.
pub fn H5VL__dataset_optional() -> Result<()> {
    Err(unsupported_vol("H5VL__dataset_optional"))
}

/// VOL wrapper: invoke an optional op on a dataset.
pub fn H5VL_dataset_optional() -> Result<()> {
    H5VL__dataset_optional()
}

/// VOL wrapper: invoke an optional op on a dataset.
pub fn H5VLdataset_optional() -> Result<()> {
    H5VL__dataset_optional()
}

/// VOL wrapper: invoke an optional op on a dataset.
pub fn H5VLdataset_optional_op() -> Result<()> {
    H5VL__dataset_optional()
}

/// VOL wrapper: close a dataset.
pub fn H5VLdataset_close(_object: VolObject) {}

/// Internal VOL wrapper: commit a datatype.
pub fn H5VL__datatype_commit(parent: &VolObject, name: &str) -> VolObject {
    H5VL_new_vol_obj(parent.connector_id, name)
}

/// VOL wrapper: commit a datatype.
pub fn H5VL_datatype_commit(parent: &VolObject, name: &str) -> VolObject {
    H5VL__datatype_commit(parent, name)
}

/// VOL wrapper: commit a datatype.
pub fn H5VLdatatype_commit(parent: &VolObject, name: &str) -> VolObject {
    H5VL__datatype_commit(parent, name)
}

/// VOL wrapper: open a datatype.
pub fn H5VLdatatype_open(parent: &VolObject, name: &str) -> VolObject {
    H5VL_new_vol_obj(parent.connector_id, name)
}

/// Internal VOL wrapper: get information about a datatype.
pub fn H5VL__datatype_get(object: &VolObject) -> usize {
    object.payload.len()
}

/// VOL wrapper: get information about a datatype.
pub fn H5VLdatatype_get(object: &VolObject) -> usize {
    H5VL__datatype_get(object)
}

/// Internal VOL wrapper: invoke a connector-specific op on a datatype.
pub fn H5VL__datatype_specific() -> Result<()> {
    Err(unsupported_vol("H5VL__datatype_specific"))
}

/// VOL wrapper: invoke a connector-specific op on a datatype.
pub fn H5VL_datatype_specific() -> Result<()> {
    H5VL__datatype_specific()
}

/// VOL wrapper: invoke a connector-specific op on a datatype.
pub fn H5VLdatatype_specific() -> Result<()> {
    H5VL__datatype_specific()
}

/// Internal VOL wrapper: invoke an optional op on a datatype.
pub fn H5VL__datatype_optional() -> Result<()> {
    Err(unsupported_vol("H5VL__datatype_optional"))
}

/// VOL wrapper: invoke an optional op on a datatype.
pub fn H5VL_datatype_optional() -> Result<()> {
    H5VL__datatype_optional()
}

/// VOL wrapper: invoke an optional op on a datatype.
pub fn H5VL_datatype_optional_op() -> Result<()> {
    H5VL__datatype_optional()
}

/// VOL wrapper: invoke an optional op on a datatype.
pub fn H5VLdatatype_optional() -> Result<()> {
    H5VL__datatype_optional()
}

/// VOL wrapper: invoke an optional op on a datatype.
pub fn H5VLdatatype_optional_op() -> Result<()> {
    H5VL__datatype_optional()
}

/// Internal VOL wrapper: close a datatype.
pub fn H5VL__datatype_close(_object: VolObject) {}

/// VOL wrapper: close a datatype.
pub fn H5VLdatatype_close(object: VolObject) {
    H5VL__datatype_close(object);
}

/// Internal VOL wrapper: create a file.
pub fn H5VL__file_create(connector_id: u64, name: &str) -> VolObject {
    H5VL_new_vol_obj(connector_id, name)
}

/// VOL wrapper: create a file.
pub fn H5VL_file_create(connector_id: u64, name: &str) -> VolObject {
    H5VL__file_create(connector_id, name)
}

/// VOL wrapper: create a file.
pub fn H5VLfile_create(connector_id: u64, name: &str) -> VolObject {
    H5VL__file_create(connector_id, name)
}

/// Internal VOL wrapper: open a file.
pub fn H5VL__file_open(connector_id: u64, name: &str) -> VolObject {
    H5VL_new_vol_obj(connector_id, name)
}

/// File-open callback that looks up a connector by name.
pub fn H5VL__file_open_find_connector_cb<'a>(
    registry: &'a VolRegistry,
    name: &str,
) -> Option<&'a VolConnector> {
    H5VL__get_connector_by_name(registry, name)
}

/// VOL wrapper: open a file.
pub fn H5VL_file_open(connector_id: u64, name: &str) -> VolObject {
    H5VL__file_open(connector_id, name)
}

/// VOL wrapper: open a file.
pub fn H5VLfile_open(connector_id: u64, name: &str) -> VolObject {
    H5VL__file_open(connector_id, name)
}

/// VOL wrapper: get information about a file.
pub fn H5VLfile_get(object: &VolObject) -> &str {
    &object.name
}

/// Internal VOL wrapper: invoke a connector-specific op on a file.
pub fn H5VL__file_specific() -> Result<()> {
    Err(unsupported_vol("H5VL__file_specific"))
}

/// VOL wrapper: invoke a connector-specific op on a file.
pub fn H5VLfile_specific() -> Result<()> {
    H5VL__file_specific()
}

/// VOL wrapper: invoke an optional op on a file.
pub fn H5VLfile_optional() -> Result<()> {
    Err(unsupported_vol("H5VLfile_optional"))
}

/// VOL wrapper: invoke an optional op on a file.
pub fn H5VLfile_optional_op() -> Result<()> {
    H5VLfile_optional()
}

/// VOL wrapper: close a file.
pub fn H5VLfile_close(_object: VolObject) {}

/// VOL wrapper: create a group.
pub fn H5VLgroup_create(parent: &VolObject, name: &str) -> VolObject {
    H5VL_new_vol_obj(parent.connector_id, name)
}

/// VOL wrapper: open a group.
pub fn H5VLgroup_open(parent: &VolObject, name: &str) -> VolObject {
    H5VL_new_vol_obj(parent.connector_id, name)
}

/// VOL wrapper: get information about a group.
pub fn H5VLgroup_get(object: &VolObject) -> &str {
    &object.name
}

/// Internal VOL wrapper: invoke a connector-specific op on a group.
pub fn H5VL__group_specific() -> Result<()> {
    Err(unsupported_vol("H5VL__group_specific"))
}

/// VOL wrapper: invoke a connector-specific op on a group.
pub fn H5VL_group_specific() -> Result<()> {
    H5VL__group_specific()
}

/// VOL wrapper: invoke a connector-specific op on a group.
pub fn H5VLgroup_specific() -> Result<()> {
    H5VL__group_specific()
}

/// Internal VOL wrapper: invoke an optional op on a group.
pub fn H5VL__group_optional() -> Result<()> {
    Err(unsupported_vol("H5VL__group_optional"))
}

/// VOL wrapper: invoke an optional op on a group.
pub fn H5VL_group_optional() -> Result<()> {
    H5VL__group_optional()
}

/// VOL wrapper: invoke an optional op on a group.
pub fn H5VLgroup_optional() -> Result<()> {
    H5VL__group_optional()
}

/// VOL wrapper: invoke an optional op on a group.
pub fn H5VLgroup_optional_op() -> Result<()> {
    H5VL__group_optional()
}

/// VOL wrapper: close a group.
pub fn H5VLgroup_close(_object: VolObject) {}

/// VOL wrapper: create a link.
pub fn H5VLlink_create() -> Result<()> {
    Err(unsupported_vol("H5VLlink_create"))
}

/// VOL wrapper: copy a link.
pub fn H5VLlink_copy() -> Result<()> {
    Err(unsupported_vol("H5VLlink_copy"))
}

/// VOL wrapper: move a link.
pub fn H5VLlink_move() -> Result<()> {
    Err(unsupported_vol("H5VLlink_move"))
}

/// VOL wrapper: get information about a link.
pub fn H5VLlink_get(object: &VolObject) -> &str {
    &object.name
}

/// Internal VOL wrapper: invoke a connector-specific op on a link.
pub fn H5VL__link_specific() -> Result<()> {
    Err(unsupported_vol("H5VL__link_specific"))
}

/// VOL wrapper: invoke a connector-specific op on a link.
pub fn H5VL_link_specific() -> Result<()> {
    H5VL__link_specific()
}

/// VOL wrapper: invoke a connector-specific op on a link.
pub fn H5VLlink_specific() -> Result<()> {
    H5VL__link_specific()
}

/// Internal VOL wrapper: invoke an optional op on a link.
pub fn H5VL__link_optional() -> Result<()> {
    Err(unsupported_vol("H5VL__link_optional"))
}

/// VOL wrapper: invoke an optional op on a link.
pub fn H5VL_link_optional() -> Result<()> {
    H5VL__link_optional()
}

/// VOL wrapper: invoke an optional op on a link.
pub fn H5VLlink_optional() -> Result<()> {
    H5VL__link_optional()
}

/// VOL wrapper: invoke an optional op on a link.
pub fn H5VLlink_optional_op() -> Result<()> {
    H5VL__link_optional()
}

/// VOL wrapper: open a object.
pub fn H5VLobject_open(parent: &VolObject, name: &str) -> VolObject {
    H5VL_new_vol_obj(parent.connector_id, name)
}

/// VOL wrapper: copy a object.
pub fn H5VLobject_copy_ref(object: &VolObject) -> &VolObject {
    object
}

/// VOL wrapper: copy a object into caller-owned storage.
pub fn H5VLobject_copy_into(object: &VolObject, dst: &mut VolObject) {
    dst.clone_from(H5VLobject_copy_ref(object));
}

/// VOL wrapper: copy a object.
pub fn H5VLobject_copy(object: &VolObject) -> VolObject {
    let mut copied = VolObject::default();
    H5VLobject_copy_into(object, &mut copied);
    copied
}

/// VOL wrapper: get information about a object.
pub fn H5VLobject_get(object: &VolObject) -> &str {
    &object.name
}

/// VOL wrapper: invoke a connector-specific op on a object.
pub fn H5VLobject_specific() -> Result<()> {
    Err(unsupported_vol("H5VLobject_specific"))
}

/// Internal VOL wrapper: invoke an optional op on a object.
pub fn H5VL__object_optional() -> Result<()> {
    Err(unsupported_vol("H5VL__object_optional"))
}

/// VOL wrapper: invoke an optional op on a object.
pub fn H5VL_object_optional() -> Result<()> {
    H5VL__object_optional()
}

/// VOL wrapper: invoke an optional op on a object.
pub fn H5VLobject_optional() -> Result<()> {
    H5VL__object_optional()
}

/// VOL wrapper: invoke an optional op on a object.
pub fn H5VLobject_optional_op() -> Result<()> {
    H5VL__object_optional()
}

/// VOL wrapper: opt query a introspect.
pub fn H5VLintrospect_opt_query() -> Result<()> {
    Err(unsupported_vol("H5VLintrospect_opt_query"))
}

/// VOL wrapper: wait on a request.
pub fn H5VLrequest_wait() -> Result<()> {
    Err(unsupported_vol("H5VLrequest_wait"))
}

/// VOL wrapper: notify a request.
pub fn H5VLrequest_notify() -> Result<()> {
    Err(unsupported_vol("H5VLrequest_notify"))
}

/// VOL wrapper: cancel a request.
pub fn H5VLrequest_cancel() -> Result<()> {
    Err(unsupported_vol("H5VLrequest_cancel"))
}

/// Internal VOL wrapper: invoke a connector-specific op on a request.
pub fn H5VL__request_specific() -> Result<()> {
    Err(unsupported_vol("H5VL__request_specific"))
}

/// VOL wrapper: invoke a connector-specific op on a request.
pub fn H5VL_request_specific() -> Result<()> {
    H5VL__request_specific()
}

/// VOL wrapper: invoke a connector-specific op on a request.
pub fn H5VLrequest_specific() -> Result<()> {
    H5VL__request_specific()
}

/// Internal VOL wrapper: invoke an optional op on a request.
pub fn H5VL__request_optional() -> Result<()> {
    Err(unsupported_vol("H5VL__request_optional"))
}

/// VOL wrapper: invoke an optional op on a request.
pub fn H5VL_request_optional() -> Result<()> {
    H5VL__request_optional()
}

/// VOL wrapper: invoke an optional op on a request.
pub fn H5VLrequest_optional() -> Result<()> {
    H5VL__request_optional()
}

/// VOL wrapper: invoke an optional op on a request.
pub fn H5VLrequest_optional_op() -> Result<()> {
    H5VL__request_optional()
}

/// VOL wrapper: free a request.
pub fn H5VLrequest_free() {}

/// VOL wrapper: put data into a blob.
pub fn H5VLblob_put(object: &mut VolObject, data: &[u8]) {
    H5VL__native_blob_put(object, data);
}

/// VOL wrapper: borrow blob bytes.
pub fn H5VLblob_view(object: &VolObject) -> &[u8] {
    H5VL__native_blob_view(object)
}

/// VOL wrapper: copy blob bytes into a caller-owned buffer.
pub fn H5VLblob_read_into(object: &VolObject, dst: &mut [u8]) -> Result<usize> {
    H5VL__native_blob_read_into(object, dst)
}

/// VOL wrapper: visit blob bytes in chunks without allocating.
pub fn H5VLblob_visit_chunks(
    object: &VolObject,
    chunk_size: usize,
    visit: impl FnMut(&[u8]) -> Result<()>,
) -> Result<()> {
    H5VL__native_blob_visit_chunks(object, chunk_size, visit)
}

/// Internal VOL wrapper: invoke a connector-specific op on a blob.
pub fn H5VL__blob_specific(object: &VolObject) -> usize {
    H5VL__native_blob_specific(object)
}

/// VOL wrapper: invoke a connector-specific op on a blob.
pub fn H5VL_blob_specific(object: &VolObject) -> usize {
    H5VL__blob_specific(object)
}

/// VOL wrapper: invoke a connector-specific op on a blob.
pub fn H5VLblob_specific(object: &VolObject) -> usize {
    H5VL__blob_specific(object)
}

/// Internal VOL wrapper: invoke an optional op on a blob.
pub fn H5VL__blob_optional() -> Result<()> {
    Err(unsupported_vol("H5VL__blob_optional"))
}

/// VOL wrapper: invoke an optional op on a blob.
pub fn H5VL_blob_optional() -> Result<()> {
    H5VL__blob_optional()
}

/// VOL wrapper: invoke an optional op on a blob.
pub fn H5VLblob_optional() -> Result<()> {
    H5VL__blob_optional()
}

/// Public token comparison.
pub fn H5VLtoken_cmp(left: u64, right: u64) -> std::cmp::Ordering {
    left.cmp(&right)
}

/// Public token-to-formatter conversion.
pub fn H5VLtoken_fmt(token: u64, dst: &mut impl fmt::Write) -> fmt::Result {
    H5VL_pass_through_token_fmt(token, dst)
}

/// Public token-to-string conversion.
#[deprecated(note = "use H5VLtoken_fmt")]
pub fn H5VLtoken_to_str(token: u64) -> String {
    token.to_string()
}

/// Native VOL wrapper: create a file.
pub fn H5VL__native_file_create(connector_id: u64, name: &str) -> VolObject {
    H5VL_new_vol_obj(connector_id, name)
}

/// Native VOL wrapper: open a file.
pub fn H5VL__native_file_open(connector_id: u64, name: &str) -> VolObject {
    H5VL_new_vol_obj(connector_id, name)
}

/// Native VOL wrapper: get information about a file.
pub fn H5VL__native_file_get(object: &VolObject) -> &str {
    &object.name
}

/// Native VOL wrapper: invoke a connector-specific op on a file.
pub fn H5VL__native_file_specific() -> Result<()> {
    Err(unsupported_vol("H5VL__native_file_specific"))
}

/// Native VOL wrapper: invoke an optional op on a file.
pub fn H5VL__native_file_optional() -> Result<()> {
    Err(unsupported_vol("H5VL__native_file_optional"))
}

/// Native VOL wrapper: close a file.
pub fn H5VL__native_file_close(_object: VolObject) {}

/// Test helper to reparse the default VOL connector environment variable.
pub fn H5VL__reparse_def_vol_conn_variable_test(registry: &mut VolRegistry, id: u64) -> Result<()> {
    H5VL__set_def_conn(registry, id)
}

/// Test helper that returns true if the connector is the native VOL.
pub fn H5VL__is_native_connector_test(connector: &VolConnector) -> bool {
    connector.name == "native" || connector.value == 0
}

/// Test helper to register a connector using a specific VOL id.
pub fn H5VL__register_using_vol_id_test(
    registry: &mut VolRegistry,
    id: u64,
    name: &str,
    value: u64,
) -> u64 {
    H5VL__register_connector_by_value(registry, id, name, value)
}

/// Get the reference count of a registered VOL connector.
pub fn H5VL_obj_get_rc(registry: &VolRegistry, id: u64) -> Option<usize> {
    registry
        .connectors
        .get(&id)
        .map(|connector| connector.refcount)
}

/// Look up the connector backing a VOL object.
pub fn H5VL_obj_get_connector<'a>(
    registry: &'a VolRegistry,
    object: &VolObject,
) -> Option<&'a VolConnector> {
    registry.connectors.get(&object.connector_id)
}

/// Borrow the payload bytes stored in a VOL object.
pub fn H5VL_obj_get_data(object: &VolObject) -> &[u8] {
    &object.payload
}

/// Clear the payload bytes of a VOL object.
pub fn H5VL_obj_reset_data(object: &mut VolObject) {
    object.payload.clear();
}

/// Native VOL wrapper: commit a datatype.
pub fn H5VL__native_datatype_commit(parent: &VolObject, name: &str) -> VolObject {
    H5VL_new_vol_obj(parent.connector_id, name)
}

/// Native VOL wrapper: open a datatype.
pub fn H5VL__native_datatype_open(parent: &VolObject, name: &str) -> VolObject {
    H5VL_new_vol_obj(parent.connector_id, name)
}

/// Native VOL wrapper: get information about a datatype.
pub fn H5VL__native_datatype_get(object: &VolObject) -> usize {
    object.payload.len()
}

/// Native VOL wrapper: invoke a connector-specific op on a datatype.
pub fn H5VL__native_datatype_specific() -> Result<()> {
    Err(unsupported_vol("H5VL__native_datatype_specific"))
}

/// Native VOL wrapper: close a datatype.
pub fn H5VL__native_datatype_close(_object: VolObject) {}

/// Native VOL wrapper: opt query a introspect.
pub fn H5VL__native_introspect_opt_query() -> Result<()> {
    Err(unsupported_vol("H5VL__native_introspect_opt_query"))
}

/// Native VOL: clone a dataset for I/O setup.
pub fn H5VL__native_dataset_io_setup_ref(object: &VolObject) -> &VolObject {
    object
}

/// Native VOL: clone a dataset for I/O setup into caller-owned storage.
pub fn H5VL__native_dataset_io_setup_into(object: &VolObject, dst: &mut VolObject) {
    dst.clone_from(H5VL__native_dataset_io_setup_ref(object));
}

/// Native VOL: clone a dataset for I/O setup.
pub fn H5VL__native_dataset_io_setup(object: &VolObject) -> VolObject {
    let mut copied = VolObject::default();
    H5VL__native_dataset_io_setup_into(object, &mut copied);
    copied
}

/// Native VOL: cleanup after a dataset I/O.
pub fn H5VL__native_dataset_io_cleanup(_object: VolObject) {}

/// Native VOL wrapper: create a dataset.
pub fn H5VL__native_dataset_create(parent: &VolObject, name: &str) -> VolObject {
    H5VL_new_vol_obj(parent.connector_id, name)
}

/// Native VOL wrapper: open a dataset.
pub fn H5VL__native_dataset_open(parent: &VolObject, name: &str) -> VolObject {
    H5VL_new_vol_obj(parent.connector_id, name)
}

/// Native VOL wrapper: borrow dataset bytes.
pub fn H5VL__native_dataset_view(object: &VolObject) -> &[u8] {
    H5VL__native_blob_view(object)
}

/// Native VOL wrapper: copy dataset bytes into a caller-owned buffer.
pub fn H5VL__native_dataset_read_into(object: &VolObject, dst: &mut [u8]) -> Result<usize> {
    H5VL__native_blob_read_into(object, dst)
}

/// Native VOL wrapper: visit dataset bytes without allocating.
pub fn H5VL__native_dataset_visit_chunks(
    object: &VolObject,
    chunk_size: usize,
    visit: impl FnMut(&[u8]) -> Result<()>,
) -> Result<()> {
    H5VL__native_blob_visit_chunks(object, chunk_size, visit)
}

/// Native VOL wrapper: write to a dataset.
pub fn H5VL__native_dataset_write(object: &mut VolObject, data: &[u8]) {
    H5VL__native_blob_put(object, data);
}

/// Native VOL wrapper: invoke an optional op on a dataset.
pub fn H5VL__native_dataset_optional() -> Result<()> {
    Err(unsupported_vol("H5VL__native_dataset_optional"))
}

/// Native VOL wrapper: close a dataset.
pub fn H5VL__native_dataset_close(_object: VolObject) {}

/// Increment the reference count on a connector.
pub fn H5VL_conn_inc_rc_public(registry: &mut VolRegistry, id: u64) -> Result<usize> {
    H5VL_conn_inc_rc(registry, id)
}

/// Decrement the reference count on a connector.
pub fn H5VL_conn_dec_rc_public(registry: &mut VolRegistry, id: u64) -> Result<usize> {
    H5VL_conn_dec_rc(registry, id)
}

/// Wrap a VOL object and return whether a wrap context was retrieved.
pub fn H5VL_object_wrap_state(object: VolObject) -> (VolObject, bool) {
    let wrapped = H5VL_pass_through_get_wrap_ctx(&object);
    (H5VL__wrap_obj(object), wrapped)
}

/// Unwrap a VOL object, returning its inner representation.
pub fn H5VL_object_unwrap_public(object: VolObject) -> VolObject {
    H5VL_object_unwrap(object)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Write as _;

    struct FixedStrBuf<const N: usize> {
        bytes: [u8; N],
        len: usize,
    }

    impl<const N: usize> FixedStrBuf<N> {
        fn new(prefix: &str) -> Self {
            let mut buf = Self {
                bytes: [0; N],
                len: 0,
            };
            buf.write_str(prefix).unwrap();
            buf
        }

        fn as_str(&self) -> &str {
            std::str::from_utf8(&self.bytes[..self.len]).unwrap()
        }
    }

    impl<const N: usize> fmt::Write for FixedStrBuf<N> {
        fn write_str(&mut self, value: &str) -> fmt::Result {
            let end = self.len.checked_add(value.len()).ok_or(fmt::Error)?;
            let dst = self.bytes.get_mut(self.len..end).ok_or(fmt::Error)?;
            dst.copy_from_slice(value.as_bytes());
            self.len = end;
            Ok(())
        }
    }

    #[test]
    fn vol_wrappers_expose_allocation_aware_byte_access() {
        let parent = H5VL_new_vol_obj(0, "file");

        let mut attr = H5VLattr_create(&parent, "units");
        H5VLattr_write(&mut attr, b"meters");
        assert_eq!(H5VLattr_view(&attr), b"meters");
        let mut attr_dst = [0; 6];
        assert_eq!(H5VLattr_read_into(&attr, &mut attr_dst).unwrap(), 6);
        assert_eq!(&attr_dst, b"meters");
        let attr_expected: [&[u8]; 2] = [b"mete", b"rs"];
        let mut attr_chunk_count = 0;
        H5VLattr_visit_chunks(&attr, 4, |chunk| {
            assert_eq!(chunk, attr_expected[attr_chunk_count]);
            attr_chunk_count += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(attr_chunk_count, attr_expected.len());

        let mut dataset = H5VLdataset_create(&parent, "values");
        H5VLdataset_write(&mut dataset, b"\x01\x02\x03\x04");
        assert_eq!(H5VLdataset_view(&dataset), b"\x01\x02\x03\x04");
        let mut dataset_dst = [0; 4];
        assert_eq!(
            H5VLdataset_read_into(&dataset, &mut dataset_dst).unwrap(),
            4
        );
        assert_eq!(&dataset_dst, b"\x01\x02\x03\x04");
        let dataset_expected: [&[u8]; 2] = [b"\x01\x02\x03", b"\x04"];
        let mut dataset_chunk_count = 0;
        H5VLdataset_visit_chunks(&dataset, 3, |chunk| {
            assert_eq!(chunk, dataset_expected[dataset_chunk_count]);
            dataset_chunk_count += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(dataset_chunk_count, dataset_expected.len());

        let mut blob = H5VL_new_vol_obj(0, "blob");
        H5VLblob_put(&mut blob, b"abcdef");
        assert_eq!(H5VLblob_view(&blob), b"abcdef");
        assert!(H5VLblob_read_into(&blob, &mut [0; 2]).is_err());

        let blob_expected: [&[u8]; 2] = [b"abc", b"def"];
        let mut blob_chunk_count = 0;
        H5VLblob_visit_chunks(&blob, 3, |chunk| {
            assert_eq!(chunk, blob_expected[blob_chunk_count]);
            blob_chunk_count += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(blob_chunk_count, blob_expected.len());
    }

    #[test]
    fn passthrough_wrappers_expose_allocation_aware_byte_access() {
        let parent = H5VL_new_vol_obj(0, "file");

        let mut attr = H5VL_pass_through_attr_create(&parent, "units");
        H5VL_pass_through_attr_write(&mut attr, b"kelvin");
        assert_eq!(H5VL_pass_through_attr_view(&attr), b"kelvin");
        let mut attr_dst = [0; 6];
        assert_eq!(
            H5VL_pass_through_attr_read_into(&attr, &mut attr_dst).unwrap(),
            6
        );
        assert_eq!(&attr_dst, b"kelvin");
        let attr_expected: [&[u8]; 3] = [b"ke", b"lv", b"in"];
        let mut attr_chunk_count = 0;
        H5VL_pass_through_attr_visit_chunks(&attr, 2, |chunk| {
            assert_eq!(chunk, attr_expected[attr_chunk_count]);
            attr_chunk_count += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(attr_chunk_count, attr_expected.len());

        let mut dataset = H5VL_pass_through_dataset_create(&parent, "values");
        H5VL_pass_through_dataset_write(&mut dataset, b"data");
        assert_eq!(H5VL_pass_through_dataset_view(&dataset), b"data");
        let mut dataset_dst = [0; 4];
        assert_eq!(
            H5VL_pass_through_dataset_read_into(&dataset, &mut dataset_dst).unwrap(),
            4
        );
        assert_eq!(&dataset_dst, b"data");
        let dataset_expected: [&[u8]; 2] = [b"dat", b"a"];
        let mut dataset_chunk_count = 0;
        H5VL_pass_through_dataset_visit_chunks(&dataset, 3, |chunk| {
            assert_eq!(chunk, dataset_expected[dataset_chunk_count]);
            dataset_chunk_count += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(dataset_chunk_count, dataset_expected.len());

        let mut blob = H5VL_new_vol_obj(0, "blob");
        H5VL_pass_through_blob_put(&mut blob, b"abcdef");
        assert_eq!(H5VL_pass_through_blob_view(&blob), b"abcdef");
        let mut blob_dst = [0; 6];
        assert_eq!(
            H5VL_pass_through_blob_read_into(&blob, &mut blob_dst).unwrap(),
            6
        );
        assert_eq!(&blob_dst, b"abcdef");

        let blob_expected: [&[u8]; 2] = [b"abcd", b"ef"];
        let mut blob_chunk_count = 0;
        H5VL_pass_through_blob_visit_chunks(&blob, 4, |chunk| {
            assert_eq!(chunk, blob_expected[blob_chunk_count]);
            blob_chunk_count += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(blob_chunk_count, blob_expected.len());
    }

    #[test]
    fn native_dataset_wrappers_expose_allocation_aware_byte_access() {
        let parent = H5VL_new_vol_obj(0, "file");
        let mut dataset = H5VL__native_dataset_create(&parent, "values");

        H5VL__native_dataset_write(&mut dataset, b"native");
        assert_eq!(H5VL__native_dataset_view(&dataset), b"native");
        let mut dst = [0; 6];
        assert_eq!(
            H5VL__native_dataset_read_into(&dataset, &mut dst).unwrap(),
            6
        );
        assert_eq!(&dst, b"native");

        let expected: [&[u8]; 2] = [b"nativ", b"e"];
        let mut chunk_count = 0;
        H5VL__native_dataset_visit_chunks(&dataset, 5, |chunk| {
            assert_eq!(chunk, expected[chunk_count]);
            chunk_count += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(chunk_count, expected.len());
    }

    #[test]
    fn token_and_connector_names_have_allocation_aware_accessors() {
        let mut registry = VolRegistry::default();
        let id = H5VLregister_connector(&mut registry, "native", 0);

        assert_eq!(H5VLget_connector_name_view(&registry, id), Some("native"));
        let mut connector_name = FixedStrBuf::<32>::new("connector=");
        assert!(H5VLget_connector_name_fmt(&registry, id, &mut connector_name).unwrap());
        assert_eq!(connector_name.as_str(), "connector=native");

        let mut token = FixedStrBuf::<16>::new("token=");
        H5VLtoken_fmt(42, &mut token).unwrap();
        assert_eq!(token.as_str(), "token=42");

        let mut passthrough_token = FixedStrBuf::<16>::new("");
        H5VL_pass_through_token_fmt(7, &mut passthrough_token).unwrap();
        assert_eq!(passthrough_token.as_str(), "7");

        let mut native_token = FixedStrBuf::<16>::new("");
        H5VL__native_token_fmt(9, &mut native_token).unwrap();
        assert_eq!(native_token.as_str(), "9");
    }

    #[test]
    fn object_and_connector_copy_wrappers_have_borrowed_and_into_paths() {
        let mut object = H5VL_new_vol_obj(3, "object");
        H5VL__native_blob_put(&mut object, b"payload");
        let connector = VolConnector {
            id: 3,
            name: "native".to_string(),
            value: 0,
            refcount: 1,
            cap_flags: 0x20,
        };

        assert!(std::ptr::eq(
            H5VL_pass_through_object_copy_ref(&object),
            &object
        ));
        assert!(std::ptr::eq(H5VL__native_object_copy_ref(&object), &object));
        assert!(std::ptr::eq(H5VLobject_copy_ref(&object), &object));
        assert!(std::ptr::eq(
            H5VL__native_dataset_io_setup_ref(&object),
            &object
        ));
        assert!(std::ptr::eq(
            H5VL_pass_through_introspect_get_conn_cls_ref(&connector),
            &connector
        ));
        assert!(std::ptr::eq(
            H5VL__native_introspect_get_conn_cls_ref(&connector),
            &connector
        ));

        let mut copied_object = VolObject {
            name: "reuse-capacity".repeat(4),
            payload: vec![0; 32],
            ..VolObject::default()
        };
        H5VLobject_copy_into(&object, &mut copied_object);
        assert_eq!(copied_object, object);

        H5VL__native_dataset_io_setup_into(&object, &mut copied_object);
        assert_eq!(copied_object, object);

        let mut copied_connector = VolConnector {
            name: "reuse-capacity".repeat(4),
            ..VolConnector::default()
        };
        H5VL_pass_through_introspect_get_conn_cls_into(&connector, &mut copied_connector);
        assert_eq!(copied_connector, connector);

        H5VL__native_introspect_get_conn_cls_into(&connector, &mut copied_connector);
        assert_eq!(copied_connector, connector);
    }
}
