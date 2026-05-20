use crate::error::{ErrorEvent, ErrorEventSet, Result};

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
pub fn H5ESget_op_counter(event_set: &ErrorEventSet) -> u64 {
    event_set.get_op_counter()
}

#[allow(non_snake_case)]
pub fn H5ESwait(event_set: &mut ErrorEventSet) -> usize {
    event_set.wait()
}

#[allow(non_snake_case)]
pub fn H5EScancel(event_set: &mut ErrorEventSet) -> usize {
    event_set.cancel()
}

#[allow(non_snake_case)]
pub fn H5ESget_err_status(event_set: &ErrorEventSet) -> bool {
    event_set.get_err_status()
}

#[allow(non_snake_case)]
pub fn H5ESget_err_count(event_set: &ErrorEventSet) -> usize {
    event_set.get_err_count()
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
pub fn H5ESfree_err_info(out: &mut Vec<String>) {
    out.clear();
}

#[deprecated(note = "use H5ESget_err_info_with or H5ESget_err_info_into")]
#[allow(non_snake_case)]
pub fn H5ESget_err_info(event_set: &ErrorEventSet) -> Vec<String> {
    let mut out = Vec::new();
    H5ESget_err_info_into(event_set, &mut out);
    out
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
pub fn H5ESregister_insert_func(event_set: &mut ErrorEventSet, registered: bool) -> Result<()> {
    event_set.register_insert_func(registered);
    Ok(())
}

#[allow(non_snake_case)]
pub fn H5ESregister_complete_func(event_set: &mut ErrorEventSet, registered: bool) -> Result<()> {
    event_set.register_complete_func(registered);
    Ok(())
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
        H5ESfree_err_info(&mut errors);
        assert!(errors.is_empty());

        H5ESregister_insert_func(&mut set, true).unwrap();
        H5ESregister_complete_func(&mut set, true).unwrap();
        assert!(H5EShas_insert_func(&set));
        assert!(H5EShas_complete_func(&set));

        assert_eq!(H5ESwait(&mut set), 2);
        assert_eq!(H5EScancel(&mut set), 0);
        H5ES__close_failed_cb(&mut set);
        assert_eq!(H5ESget_err_count(&set), 0);
        H5ESclear(&mut set);
        assert_eq!(H5ESget_count(&set), 0);
    }
}
