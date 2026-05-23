use std::ffi::c_void;

use crate::error::{Error, ErrorEvent, ErrorEventSet, Result};

pub const H5ES_NONE: u64 = 0;
pub const H5ES_WAIT_FOREVER: u64 = u64::MAX;
pub const H5ES_WAIT_NONE: u64 = 0;

#[allow(non_camel_case_types)]
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum H5ES_status_t {
    H5ES_STATUS_IN_PROGRESS = 0,
    H5ES_STATUS_SUCCEED = 1,
    H5ES_STATUS_CANCELED = 2,
    H5ES_STATUS_FAIL = 3,
}

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct H5ES_op_info_t {
    pub api_name: String,
    pub api_args: String,
    pub app_file_name: String,
    pub app_func_name: String,
    pub app_line_num: u32,
    pub op_ins_count: u64,
    pub op_ins_ts: u64,
    pub op_exec_ts: u64,
    pub op_exec_time: u64,
}

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct H5ES_err_info_t {
    pub api_name: String,
    pub api_args: String,
    pub app_file_name: String,
    pub app_func_name: String,
    pub app_line_num: u32,
    pub op_ins_count: u64,
    pub op_ins_ts: u64,
    pub op_exec_ts: u64,
    pub op_exec_time: u64,
    pub err_stack_id: u64,
}

#[allow(non_camel_case_types)]
pub type H5ES_event_insert_func_t =
    Option<unsafe extern "C" fn(*const H5ES_op_info_t, *mut c_void) -> i32>;

#[allow(non_camel_case_types)]
pub type H5ES_event_complete_func_t =
    Option<unsafe extern "C" fn(*mut H5ES_op_info_t, H5ES_status_t, u64, *mut c_void) -> i32>;

fn unsupported_event_set(name: &str) -> Error {
    Error::Unsupported(format!(
        "{name} requires libhdf5 asynchronous VOL event-set behavior not implemented in pure-Rust mode"
    ))
}

#[allow(non_snake_case)]
pub fn H5ES__init_package() -> ErrorEventSet {
    ErrorEventSet::init_package()
}

#[allow(non_snake_case)]
pub fn H5ES_term_package(event_set: &mut ErrorEventSet) {
    event_set.term_package();
}

#[allow(non_snake_case)]
pub fn H5EScreate() -> ErrorEventSet {
    ErrorEventSet::create()
}

#[allow(non_snake_case)]
pub fn H5ES__create_api_common() -> ErrorEventSet {
    ErrorEventSet::create_internal()
}

#[allow(non_snake_case)]
pub fn H5ESclose(event_set: ErrorEventSet) {
    event_set.close_cb();
}

#[allow(non_snake_case)]
pub fn H5ES__close_cb(event_set: ErrorEventSet) {
    event_set.close_cb();
}

#[allow(non_snake_case)]
pub fn H5ES__insert_request(event_set: &mut ErrorEventSet, request: impl Into<String>) {
    event_set.insert_request_internal(request);
}

#[allow(non_snake_case)]
pub fn H5ESinsert_request(event_set: &mut ErrorEventSet, request: impl Into<String>) {
    event_set.insert_request(request);
}

#[allow(non_snake_case)]
pub fn H5ES_insert(event_set: &mut ErrorEventSet, request: impl Into<String>) {
    event_set.insert(request);
}

#[allow(non_snake_case)]
pub fn H5ES__list_append(event_set: &mut ErrorEventSet, event: ErrorEvent) {
    event_set.list_append(event);
}

#[allow(non_snake_case)]
pub fn H5ES__list_count(event_set: &ErrorEventSet) -> usize {
    event_set.list_count()
}

#[allow(non_snake_case)]
pub fn H5ESget_count(event_set: &ErrorEventSet) -> usize {
    event_set.get_count()
}

#[allow(non_snake_case)]
pub fn H5ESget_count_into(event_set: &ErrorEventSet, count: &mut usize) -> Result<()> {
    *count = event_set.get_count();
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5ESget_op_counter(event_set: &ErrorEventSet) -> u64 {
    event_set.get_op_counter()
}

#[allow(non_snake_case)]
pub fn H5ESget_op_counter_into(event_set: &ErrorEventSet, counter: &mut u64) -> Result<()> {
    *counter = event_set.get_op_counter();
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5ESget_op_info(
    _event_set: &ErrorEventSet,
    _op_counter: u64,
    _op_info: &mut H5ES_op_info_t,
) -> Result<()> {
    Err(unsupported_event_set("H5ESget_op_info"))
}

#[allow(non_snake_case)]
pub fn H5ESwait(
    event_set: &mut ErrorEventSet,
    _timeout: u64,
    num_in_progress: &mut usize,
    err_occurred: &mut bool,
) -> Result<()> {
    *num_in_progress = event_set.wait();
    *err_occurred = event_set.get_err_status();
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5EScancel(
    event_set: &mut ErrorEventSet,
    num_not_canceled: &mut usize,
    err_occurred: &mut bool,
) -> Result<()> {
    *num_not_canceled = event_set.cancel();
    *err_occurred = event_set.get_err_status();
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5ESget_err_status(event_set: &ErrorEventSet) -> bool {
    event_set.get_err_status()
}

#[allow(non_snake_case)]
pub fn H5ESget_err_status_into(event_set: &ErrorEventSet, err_occurred: &mut bool) -> Result<()> {
    *err_occurred = event_set.get_err_status();
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5ESget_err_count(event_set: &ErrorEventSet) -> usize {
    event_set.get_err_count()
}

#[allow(non_snake_case)]
pub fn H5ESget_err_count_into(event_set: &ErrorEventSet, err_count: &mut usize) -> Result<()> {
    *err_count = event_set.get_err_count();
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5ESget_requests_with<F>(event_set: &ErrorEventSet, callback: F)
where
    F: FnMut(&str),
{
    event_set.get_requests_with(callback);
}

#[allow(non_snake_case)]
pub fn H5ESget_requests_into(event_set: &ErrorEventSet, out: &mut Vec<String>) {
    out.clear();
    out.extend(event_set.requests().map(str::to_owned));
}

#[deprecated(note = "use H5ESget_requests_with or H5ESget_requests_into")]
#[allow(non_snake_case)]
pub fn H5ESget_requests(event_set: &ErrorEventSet) -> Vec<String> {
    let mut out = Vec::new();
    H5ESget_requests_into(event_set, &mut out);
    out
}

#[allow(non_snake_case)]
pub fn H5ESget_err_info_with<F>(event_set: &ErrorEventSet, callback: F)
where
    F: FnMut(&str),
{
    event_set.get_err_info_with(callback);
}

#[allow(non_snake_case)]
pub fn H5ESget_err_info_into(event_set: &ErrorEventSet, out: &mut Vec<String>) {
    out.clear();
    out.extend(event_set.err_info().map(str::to_owned));
}

#[allow(non_snake_case)]
pub fn H5ESfree_err_info(num_err_info: usize, err_info: &mut [H5ES_err_info_t]) -> Result<()> {
    let available = err_info.len();
    let infos = err_info.get_mut(..num_err_info).ok_or_else(|| {
        Error::Other(format!(
            "H5ESfree_err_info requested {num_err_info} records from {available}-record buffer"
        ))
    })?;

    for info in infos {
        info.api_name.clear();
        info.api_args.clear();
        info.app_file_name.clear();
        info.app_func_name.clear();
        info.app_line_num = 0;
        info.op_ins_count = 0;
        info.op_ins_ts = 0;
        info.op_exec_ts = 0;
        info.op_exec_time = 0;
        info.err_stack_id = 0;
    }
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5ESget_err_info(
    _event_set: &mut ErrorEventSet,
    _num_err_info: usize,
    _err_info: &mut [H5ES_err_info_t],
    _err_cleared: &mut usize,
) -> Result<()> {
    Err(unsupported_event_set("H5ESget_err_info"))
}

#[allow(non_snake_case)]
pub fn H5ES__get_err_info(
    _event_set: &mut ErrorEventSet,
    _num_err_info: usize,
    _err_info: &mut [H5ES_err_info_t],
    _err_cleared: &mut usize,
) -> Result<()> {
    Err(unsupported_event_set("H5ES__get_err_info"))
}

#[allow(non_snake_case)]
pub fn H5ES__close_failed_cb(event_set: &mut ErrorEventSet) {
    event_set.close_failed_cb();
}

#[allow(non_snake_case)]
pub fn H5ESclear(event_set: &mut ErrorEventSet) {
    event_set.term_package();
}

#[allow(non_snake_case)]
pub fn H5ESfail_request(
    event_set: &mut ErrorEventSet,
    request: impl Into<String>,
    error: impl Into<String>,
) {
    let mut event = ErrorEvent::event_new(request);
    event.handle_fail(error);
    event_set.list_append(event);
}

#[allow(non_snake_case)]
pub fn H5EScomplete_request(event_set: &mut ErrorEventSet, request: impl Into<String>) {
    let mut event = ErrorEvent::event_new(request);
    event.event_completed();
    event_set.list_append(event);
}

#[allow(non_snake_case)]
pub fn H5ESregister_insert_func(
    _event_set: &mut ErrorEventSet,
    _func: H5ES_event_insert_func_t,
    _ctx: *mut c_void,
) -> Result<()> {
    Err(unsupported_event_set("H5ESregister_insert_func"))
}

#[allow(non_snake_case)]
pub fn H5ESregister_complete_func(
    _event_set: &mut ErrorEventSet,
    _func: H5ES_event_complete_func_t,
    _ctx: *mut c_void,
) -> Result<()> {
    Err(unsupported_event_set("H5ESregister_complete_func"))
}

#[allow(non_snake_case)]
pub fn H5EShas_insert_func(event_set: &ErrorEventSet) -> bool {
    event_set.has_insert_callback()
}

#[allow(non_snake_case)]
pub fn H5EShas_complete_func(event_set: &ErrorEventSet) -> bool {
    event_set.has_complete_callback()
}

#[allow(non_snake_case)]
pub fn H5ESnoop() -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    #[test]
    fn h5es_aliases_preserve_event_set_semantics() {
        let mut set = H5EScreate();
        H5ESinsert_request(&mut set, "read");
        H5ES__insert_request(&mut set, "write");
        H5EScomplete_request(&mut set, "flush");
        H5ESfail_request(&mut set, "close", "failed");

        assert_eq!(H5ESget_count(&set), 4);
        assert_eq!(H5ES__list_count(&set), 4);
        assert_eq!(H5ESget_op_counter(&set), 4);
        assert!(H5ESget_err_status(&set));
        assert_eq!(H5ESget_err_count(&set), 1);

        let mut requests = Vec::new();
        H5ESget_requests_into(&set, &mut requests);
        assert_eq!(requests, ["read", "write", "flush", "close"]);

        let mut visited = Vec::new();
        H5ESget_requests_with(&set, |request| visited.push(request.to_owned()));
        assert_eq!(visited, requests);

        let mut errors = Vec::new();
        H5ESget_err_info_into(&set, &mut errors);
        assert_eq!(errors, ["failed"]);
        errors.clear();

        assert!(!H5EShas_insert_func(&set));
        assert!(!H5EShas_complete_func(&set));

        let mut num_in_progress = usize::MAX;
        let mut err_occurred = false;
        H5ESwait(
            &mut set,
            H5ES_WAIT_FOREVER,
            &mut num_in_progress,
            &mut err_occurred,
        )
        .unwrap();
        assert_eq!(num_in_progress, 2);
        assert!(err_occurred);

        let mut num_not_canceled = usize::MAX;
        err_occurred = false;
        H5EScancel(&mut set, &mut num_not_canceled, &mut err_occurred).unwrap();
        assert_eq!(num_not_canceled, 0);
        assert!(err_occurred);
        H5ES__close_failed_cb(&mut set);
        assert_eq!(H5ESget_err_count(&set), 0);
        H5ESclear(&mut set);
        assert_eq!(H5ESget_count(&set), 0);
    }

    #[test]
    fn h5es_wait_and_cancel_use_public_output_parameters() {
        let mut set = H5EScreate();
        H5ESinsert_request(&mut set, "read");
        H5ESfail_request(&mut set, "write", "failed");

        let mut num_in_progress = 99;
        let mut err_occurred = false;
        H5ESwait(
            &mut set,
            H5ES_WAIT_NONE,
            &mut num_in_progress,
            &mut err_occurred,
        )
        .unwrap();
        assert_eq!(num_in_progress, 1);
        assert!(err_occurred);

        let mut num_not_canceled = 99;
        err_occurred = false;
        H5EScancel(&mut set, &mut num_not_canceled, &mut err_occurred).unwrap();
        assert_eq!(num_not_canceled, 0);
        assert!(err_occurred);
    }

    #[test]
    fn h5es_insert_entry_point_preserves_request_semantics() {
        let mut set = H5EScreate();

        H5ES_insert(&mut set, "read");
        H5ESinsert_request(&mut set, "write");

        let mut requests = Vec::new();
        H5ESget_requests_into(&set, &mut requests);
        assert_eq!(requests, ["read", "write"]);
        assert_eq!(H5ESget_count(&set), 2);
        assert_eq!(H5ESget_op_counter(&set), 2);
    }

    #[test]
    fn h5es_count_and_counter_use_public_output_parameters() {
        let mut set = H5EScreate();
        H5ESinsert_request(&mut set, "read");
        H5EScomplete_request(&mut set, "flush");
        H5ESfail_request(&mut set, "close", "failed");

        let mut count = usize::MAX;
        H5ESget_count_into(&set, &mut count).unwrap();
        assert_eq!(count, 3);

        let mut counter = u64::MAX;
        H5ESget_op_counter_into(&set, &mut counter).unwrap();
        assert_eq!(counter, 3);

        H5ESclear(&mut set);
        H5ESget_count_into(&set, &mut count).unwrap();
        assert_eq!(count, 0);
        H5ESget_op_counter_into(&set, &mut counter).unwrap();
        assert_eq!(counter, 3);
    }

    #[test]
    fn h5es_op_info_query_is_explicit_unsupported_boundary() {
        let mut set = H5EScreate();
        H5ESinsert_request(&mut set, "read");
        let mut op_info = H5ES_op_info_t {
            api_name: "sentinel".into(),
            api_args: "args".into(),
            app_file_name: "app.rs".into(),
            app_func_name: "caller".into(),
            app_line_num: 7,
            op_ins_count: 9,
            op_ins_ts: 11,
            op_exec_ts: 13,
            op_exec_time: 17,
        };
        let unchanged = op_info.clone();

        assert!(matches!(
            H5ESget_op_info(&set, 1, &mut op_info),
            Err(Error::Unsupported(_))
        ));
        assert_eq!(op_info, unchanged);
    }

    #[test]
    fn h5es_error_status_and_count_use_public_output_parameters() {
        let mut set = H5EScreate();

        let mut err_occurred = true;
        H5ESget_err_status_into(&set, &mut err_occurred).unwrap();
        assert!(!err_occurred);

        let mut err_count = usize::MAX;
        H5ESget_err_count_into(&set, &mut err_count).unwrap();
        assert_eq!(err_count, 0);

        H5ESfail_request(&mut set, "write", "disk");
        H5EScomplete_request(&mut set, "flush");
        H5ESfail_request(&mut set, "close", "metadata");

        H5ESget_err_status_into(&set, &mut err_occurred).unwrap();
        assert!(err_occurred);
        H5ESget_err_count_into(&set, &mut err_count).unwrap();
        assert_eq!(err_count, 2);

        H5ES__close_failed_cb(&mut set);
        H5ESget_err_status_into(&set, &mut err_occurred).unwrap();
        assert!(!err_occurred);
        H5ESget_err_count_into(&set, &mut err_count).unwrap();
        assert_eq!(err_count, 0);
    }

    #[test]
    fn h5es_public_constants_and_statuses_match_libhdf5_names() {
        assert_eq!(H5ES_NONE, 0);
        assert_eq!(H5ES_WAIT_NONE, 0);
        assert_eq!(H5ES_WAIT_FOREVER, u64::MAX);

        assert_eq!(
            H5ES_status_t::H5ES_STATUS_IN_PROGRESS,
            H5ES_status_t::H5ES_STATUS_IN_PROGRESS
        );
        assert_eq!(H5ES_status_t::H5ES_STATUS_IN_PROGRESS as i32, 0);
        assert_eq!(H5ES_status_t::H5ES_STATUS_SUCCEED as i32, 1);
        assert_eq!(H5ES_status_t::H5ES_STATUS_CANCELED as i32, 2);
        assert_eq!(H5ES_status_t::H5ES_STATUS_FAIL as i32, 3);
        assert_ne!(
            H5ES_status_t::H5ES_STATUS_SUCCEED,
            H5ES_status_t::H5ES_STATUS_FAIL
        );
        assert_ne!(
            H5ES_status_t::H5ES_STATUS_CANCELED,
            H5ES_status_t::H5ES_STATUS_FAIL
        );
    }

    #[test]
    fn h5es_error_info_records_are_explicit_unsupported_boundary() {
        let mut set = H5EScreate();
        H5ESfail_request(&mut set, "close", "failed");
        let mut records = vec![H5ES_err_info_t::default()];
        let mut err_cleared = usize::MAX;

        assert!(matches!(
            H5ESget_err_info(&mut set, records.len(), &mut records, &mut err_cleared),
            Err(Error::Unsupported(_))
        ));
        assert_eq!(err_cleared, usize::MAX);
        assert!(matches!(
            H5ES__get_err_info(&mut set, records.len(), &mut records, &mut err_cleared),
            Err(Error::Unsupported(_))
        ));
        assert_eq!(err_cleared, usize::MAX);

        records[0].api_name = "H5Dwrite_async".into();
        records[0].err_stack_id = 42;
        H5ESfree_err_info(records.len(), &mut records).unwrap();

        assert_eq!(records, [H5ES_err_info_t::default()]);
    }

    #[test]
    fn h5es_free_err_info_uses_public_count_parameter() {
        let mut records = vec![H5ES_err_info_t::default(); 2];
        records[0].api_name = "H5Dwrite_async".into();
        records[0].api_args = "first".into();
        records[0].err_stack_id = 1;
        records[1].api_name = "H5Dflush_async".into();
        records[1].api_args = "second".into();
        records[1].err_stack_id = 2;
        let second = records[1].clone();

        H5ESfree_err_info(1, &mut records).unwrap();

        assert_eq!(records[0], H5ES_err_info_t::default());
        assert_eq!(records[1], second);
        assert!(matches!(
            H5ESfree_err_info(3, &mut records),
            Err(Error::Other(_))
        ));
    }

    #[test]
    fn h5es_callback_registration_is_explicit_unsupported_boundary() {
        unsafe extern "C" fn insert_cb(_op_info: *const H5ES_op_info_t, _ctx: *mut c_void) -> i32 {
            0
        }

        unsafe extern "C" fn complete_cb(
            _op_info: *mut H5ES_op_info_t,
            _status: H5ES_status_t,
            _err_stack_id: u64,
            _ctx: *mut c_void,
        ) -> i32 {
            0
        }

        let mut set = H5EScreate();

        assert!(matches!(
            H5ESregister_insert_func(&mut set, Some(insert_cb), ptr::null_mut()),
            Err(Error::Unsupported(_))
        ));
        assert!(!H5EShas_insert_func(&set));

        assert!(matches!(
            H5ESregister_complete_func(&mut set, Some(complete_cb), ptr::null_mut()),
            Err(Error::Unsupported(_))
        ));
        assert!(!H5EShas_complete_func(&set));
    }
}
