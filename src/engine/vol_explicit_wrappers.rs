#![allow(dead_code, non_snake_case)]

use super::{
    H5VL__conn_free, H5VL__get_connector_by_name, H5VL__get_connector_by_value,
    H5VL__native_attr_create, H5VL__native_attr_open, H5VL__native_blob_get, H5VL__native_blob_put,
    H5VL__native_blob_specific, H5VL__register_connector_by_name,
    H5VL__register_connector_by_value, H5VL__set_def_conn, H5VL__wrap_obj, H5VL_conn_dec_rc,
    H5VL_conn_inc_rc, H5VL_new_vol_obj, H5VL_object_unwrap, H5VL_pass_through_get_wrap_ctx,
    H5VL_restore_lib_state, H5VL_retrieve_lib_state, H5VL_wrap_register, VolConnector, VolLibState,
    VolObject, VolRegistry,
};
use crate::error::{Error, Result};

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

/// Passthrough VOL wrapper: invoke a connector-specific op on a attr.
pub fn H5VL_pass_through_attr_specific() -> Result<()> {
    Err(unsupported_vol("H5VL_pass_through_attr_specific"))
}

/// Passthrough VOL wrapper: create a dataset.
pub fn H5VL_pass_through_dataset_create(parent: &VolObject, name: &str) -> VolObject {
    H5VL_new_vol_obj(parent.connector_id, name)
}

/// Passthrough VOL wrapper: read from a dataset.
pub fn H5VL_pass_through_dataset_read(object: &VolObject) -> Vec<u8> {
    H5VL__native_blob_get(object)
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
pub fn H5VL_pass_through_object_copy(object: &VolObject) -> VolObject {
    object.clone()
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
pub fn H5VL_pass_through_introspect_get_conn_cls(connector: &VolConnector) -> VolConnector {
    connector.clone()
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

/// Passthrough VOL wrapper: get information about a blob.
pub fn H5VL_pass_through_blob_get(object: &VolObject) -> Vec<u8> {
    H5VL__native_blob_get(object)
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

/// Render a VOL token as a decimal string.
pub fn H5VL_pass_through_token_to_str(token: u64) -> String {
    token.to_string()
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

/// Look up the name of a connector by id.
pub fn H5VLget_connector_name(registry: &VolRegistry, id: u64) -> Option<String> {
    registry
        .connectors
        .get(&id)
        .map(|connector| connector.name.clone())
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
pub fn H5VL__native_introspect_get_conn_cls(connector: &VolConnector) -> VolConnector {
    connector.clone()
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
pub fn H5VL__native_object_copy(object: &VolObject) -> VolObject {
    object.clone()
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

/// Render a native VOL token as a decimal string.
pub fn H5VL__native_token_to_str(token: u64) -> String {
    token.to_string()
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

/// VOL wrapper: get information about a attr.
pub fn H5VLattr_get(object: &VolObject) -> Vec<u8> {
    H5VL__native_blob_get(object)
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

/// VOL wrapper: read from a dataset.
pub fn H5VLdataset_read(object: &VolObject) -> Vec<u8> {
    H5VL__native_blob_get(object)
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
pub fn H5VLobject_copy(object: &VolObject) -> VolObject {
    object.clone()
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

/// VOL wrapper: get information about a blob.
pub fn H5VLblob_get(object: &VolObject) -> Vec<u8> {
    H5VL__native_blob_get(object)
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

/// Public token-to-string conversion.
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
pub fn H5VL__native_dataset_io_setup(object: &VolObject) -> VolObject {
    object.clone()
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

/// Native VOL wrapper: read from a dataset.
pub fn H5VL__native_dataset_read(object: &VolObject) -> Vec<u8> {
    H5VL__native_blob_get(object)
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
