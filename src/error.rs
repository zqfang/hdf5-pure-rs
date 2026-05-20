use std::fmt;
use std::io;

/// The main error type for hdf5-pure-rust.
#[derive(Debug)]
pub enum Error {
    /// An I/O error occurred.
    Io(io::Error),
    /// Invalid HDF5 file format.
    InvalidFormat(String),
    /// Unsupported HDF5 feature or version.
    Unsupported(String),
    /// Other error.
    Other(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::InvalidFormat(msg) => write!(f, "Invalid HDF5 format: {msg}"),
            Error::Unsupported(msg) => write!(f, "Unsupported: {msg}"),
            Error::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::Io(e)
    }
}

/// Result type alias for hdf5-pure-rust.
pub type Result<T> = std::result::Result<T, Error>;

/// Registered error class metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorClass {
    pub library: String,
    pub class_name: String,
    pub version: String,
}

/// Registered error message metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorMessage {
    pub message: String,
}

/// One asynchronous event-set entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorEvent {
    pub request: String,
    pub completed: bool,
    pub error: Option<String>,
}

/// Lightweight HDF5 event-set analogue used for non-MPI async bookkeeping.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ErrorEventSet {
    events: Vec<ErrorEvent>,
    op_counter: u64,
    insert_callback_registered: bool,
    complete_callback_registered: bool,
}

/// One error stack entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorStackEntry {
    pub class_name: String,
    pub major: String,
    pub minor: String,
    pub description: String,
}

/// Lightweight HDF5-style error stack.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ErrorStack {
    entries: Vec<ErrorStackEntry>,
    auto_enabled: bool,
}

impl ErrorClass {
    /// Register an error class.
    pub fn register_class(
        library: impl Into<String>,
        class_name: impl Into<String>,
        version: impl Into<String>,
    ) -> Self {
        Self {
            library: library.into(),
            class_name: class_name.into(),
            version: version.into(),
        }
    }

    /// Internal register-class alias.
    pub fn register_class_internal(
        library: impl Into<String>,
        class_name: impl Into<String>,
        version: impl Into<String>,
    ) -> Self {
        Self::register_class(library, class_name, version)
    }

    /// Unregister an error class. The pure Rust value is consumed.
    pub fn unregister_class(self) {}

    /// Internal unregister-class alias.
    pub fn unregister_class_internal(self) {}

    /// Return this class name.
    pub fn class_name(&self) -> &str {
        &self.class_name
    }
}

impl ErrorMessage {
    /// Create an error message.
    pub fn create_msg(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// Internal create-message alias.
    pub fn create_msg_internal(message: impl Into<String>) -> Self {
        Self::create_msg(message)
    }

    /// Return the message text.
    pub fn get_msg(&self) -> &str {
        &self.message
    }

    /// Internal message getter alias.
    pub fn get_msg_internal(&self) -> &str {
        self.get_msg()
    }

    /// Close an error message. The pure Rust value is consumed.
    pub fn close_msg(self) {}

    /// Internal close-message callback alias.
    pub fn close_msg_cb(self) {}

    /// Internal close-message alias.
    pub fn close_msg_internal(self) {}

    /// Internal free-message alias.
    pub fn free_msg(self) {}
}

impl ErrorEvent {
    /// Create an event-set entry.
    pub fn event_new(request: impl Into<String>) -> Self {
        Self {
            request: request.into(),
            completed: false,
            error: None,
        }
    }

    /// Free an event entry. The pure Rust value is consumed.
    pub fn event_free(self) {}

    /// Mark this event complete.
    pub fn event_completed(&mut self) {
        self.completed = true;
    }

    /// Record an event failure.
    pub fn handle_fail(&mut self, error: impl Into<String>) {
        self.completed = true;
        self.error = Some(error.into());
    }
}

impl ErrorEventSet {
    /// Initialize event-set package support.
    pub fn init_package() -> Self {
        Self::default()
    }

    /// Terminate event-set package support.
    pub fn term_package(&mut self) {
        self.events.clear();
    }

    /// Create an event set.
    pub fn create() -> Self {
        Self::default()
    }

    /// Internal create alias.
    pub fn create_internal() -> Self {
        Self::create()
    }

    /// Internal close callback alias.
    pub fn close_cb(self) {}

    /// Internal close alias.
    pub fn close_internal(self) {}

    /// Append an event to the internal list.
    pub fn list_append(&mut self, event: ErrorEvent) {
        self.events.push(event);
        self.op_counter = self.op_counter.saturating_add(1);
    }

    /// Return the event-list length.
    pub fn list_count(&self) -> usize {
        self.events.len()
    }

    /// Iterate over queued events.
    pub fn list_iterate<F>(&self, mut callback: F)
    where
        F: FnMut(&ErrorEvent),
    {
        for event in &self.events {
            callback(event);
        }
    }

    /// Borrow queued events in insertion order.
    pub fn events(&self) -> impl Iterator<Item = &ErrorEvent> {
        self.events.iter()
    }

    /// Remove an event by index.
    pub fn list_remove(&mut self, index: usize) -> Option<ErrorEvent> {
        if index < self.events.len() {
            Some(self.events.remove(index))
        } else {
            None
        }
    }

    /// Insert a request.
    pub fn insert(&mut self, request: impl Into<String>) {
        self.insert_request(request);
    }

    /// Public request-insert alias.
    pub fn insert_request(&mut self, request: impl Into<String>) {
        self.insert_request_internal(request);
    }

    /// Internal request-insert alias.
    pub fn insert_request_internal(&mut self, request: impl Into<String>) {
        self.list_append(ErrorEvent::event_new(request));
    }

    /// Iterate over request names in insertion order.
    pub fn requests(&self) -> impl Iterator<Item = &str> {
        self.events.iter().map(|event| event.request.as_str())
    }

    /// Visit request names in insertion order.
    pub fn get_requests_with<F>(&self, mut callback: F)
    where
        F: FnMut(&str),
    {
        for request in self.requests() {
            callback(request);
        }
    }

    /// Return request names in insertion order.
    #[deprecated(note = "use requests() or get_requests_with() to avoid allocating a Vec<String>")]
    pub fn get_requests(&self) -> Vec<String> {
        self.requests().map(str::to_owned).collect()
    }

    /// Return number of events.
    pub fn get_count(&self) -> usize {
        self.events.len()
    }

    /// Return the total insert operation counter.
    pub fn get_op_counter(&self) -> u64 {
        self.op_counter
    }

    /// Complete all pending events and return the number completed.
    pub fn wait(&mut self) -> usize {
        let mut completed = 0;
        for event in &mut self.events {
            if !event.completed {
                event.event_completed();
                completed += 1;
            }
        }
        completed
    }

    /// Cancel incomplete events and return the number canceled.
    pub fn cancel(&mut self) -> usize {
        let mut canceled = 0;
        for event in &mut self.events {
            if !event.completed {
                event.completed = true;
                event.error = Some("canceled".into());
                canceled += 1;
            }
        }
        canceled
    }

    /// Return whether any event has failed.
    pub fn get_err_status(&self) -> bool {
        self.events.iter().any(|event| event.error.is_some())
    }

    /// Return failed-event count.
    pub fn get_err_count(&self) -> usize {
        self.events
            .iter()
            .filter(|event| event.error.is_some())
            .count()
    }

    /// Iterate over failed-event messages.
    pub fn err_info(&self) -> impl Iterator<Item = &str> {
        self.events
            .iter()
            .filter_map(|event| event.error.as_deref())
    }

    /// Visit failed-event messages.
    pub fn get_err_info_with<F>(&self, mut callback: F)
    where
        F: FnMut(&str),
    {
        for error in self.err_info() {
            callback(error);
        }
    }

    /// Return failed-event messages.
    #[deprecated(note = "use err_info() or get_err_info_with() to avoid allocating a Vec<String>")]
    pub fn get_err_info(&self) -> Vec<String> {
        self.err_info().map(str::to_owned).collect()
    }

    /// Internal close-failed callback alias.
    pub fn close_failed_cb(&mut self) {
        self.events.retain(|event| event.error.is_none());
    }

    /// Register whether an insert callback is installed on this event set.
    pub fn register_insert_func(&mut self, registered: bool) {
        self.insert_callback_registered = registered;
    }

    /// Register whether a completion callback is installed on this event set.
    pub fn register_complete_func(&mut self, registered: bool) {
        self.complete_callback_registered = registered;
    }

    /// Return whether an insert callback is installed.
    pub fn has_insert_callback(&self) -> bool {
        self.insert_callback_registered
    }

    /// Return whether a completion callback is installed.
    pub fn has_complete_callback(&self) -> bool {
        self.complete_callback_registered
    }
}

impl ErrorStackEntry {
    pub fn new(
        class_name: impl Into<String>,
        major: impl Into<String>,
        minor: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            class_name: class_name.into(),
            major: major.into(),
            minor: minor.into(),
            description: description.into(),
        }
    }

    /// Copy a stack entry.
    pub fn copy_stack_entry(&self) -> Self {
        self.clone()
    }

    /// Set stack-entry fields.
    pub fn set_stack_entry(
        &mut self,
        class_name: impl Into<String>,
        major: impl Into<String>,
        minor: impl Into<String>,
        description: impl Into<String>,
    ) {
        *self = Self::new(class_name, major, minor, description);
    }
}

impl ErrorStack {
    /// Initialize error-stack support.
    pub fn init() -> Self {
        Self::default()
    }

    /// Internal package-initialization alias.
    pub fn init_package() -> Self {
        Self::init()
    }

    /// Terminate error-stack support by clearing this stack.
    pub fn term_package(&mut self) {
        self.entries.clear();
    }

    /// Prepare user callback state. This stack has no hidden callback payload.
    pub fn user_cb_prepare(&self) -> bool {
        true
    }

    /// Return a copy of the current stack.
    pub fn get_current_stack(&self) -> Self {
        self.clone()
    }

    /// Internal current-stack getter alias.
    pub fn get_current_stack_internal(&self) -> Self {
        self.get_current_stack()
    }

    /// Replace the current stack.
    pub fn set_current_stack(&mut self, other: Self) {
        *self = other;
    }

    /// Internal current-stack setter alias.
    pub fn set_current_stack_internal(&mut self, other: Self) {
        self.set_current_stack(other);
    }

    /// Close a stack. The pure Rust value is consumed.
    pub fn close_stack(self) {}

    /// Internal close-stack alias.
    pub fn close_stack_internal(self) {}

    /// Return number of entries.
    pub fn get_num(&self) -> usize {
        self.entries.len()
    }

    /// Internal number-of-entries alias.
    pub fn get_num_internal(&self) -> usize {
        self.get_num()
    }

    /// Push an entry.
    pub fn push_stack(&mut self, entry: ErrorStackEntry) {
        self.entries.push(entry);
    }

    /// Old public push alias.
    pub fn push1(&mut self, entry: ErrorStackEntry) {
        self.push_stack(entry);
    }

    /// Formatted push alias.
    pub fn printf_stack(&mut self, entry: ErrorStackEntry) {
        self.push_stack(entry);
    }

    /// Append another stack.
    pub fn append_stack(&mut self, other: &Self) {
        self.entries.extend(other.entries.iter().cloned());
    }

    /// Pop up to `count` entries.
    pub fn pop(&mut self, count: usize) {
        let keep = self.entries.len().saturating_sub(count);
        self.entries.truncate(keep);
    }

    /// Internal pop alias.
    pub fn pop_internal(&mut self, count: usize) {
        self.pop(count);
    }

    /// Clear stack entries.
    pub fn clear2(&mut self) {
        self.entries.clear();
    }

    /// Old clear alias.
    pub fn clear1(&mut self) {
        self.clear2();
    }

    /// Clear-stack alias.
    pub fn clear_stack(&mut self) {
        self.clear2();
    }

    /// Internal clear-entries alias.
    pub fn clear_entries(&mut self) {
        self.clear2();
    }

    /// Destroy stack by clearing entries.
    pub fn destroy_stack(&mut self) {
        self.clear2();
    }

    /// Borrow stack entries from oldest to newest.
    pub fn entries(&self) -> impl Iterator<Item = &ErrorStackEntry> {
        self.entries.iter()
    }

    /// Write stack descriptions to a caller-provided formatter.
    pub fn print_into<W>(&self, writer: &mut W) -> fmt::Result
    where
        W: fmt::Write + ?Sized,
    {
        for (index, entry) in self.entries.iter().enumerate() {
            if index > 0 {
                writer.write_char('\n')?;
            }
            writer.write_str(&entry.description)?;
        }
        Ok(())
    }

    /// Visit printable stack descriptions from oldest to newest.
    pub fn print_with<F>(&self, mut callback: F)
    where
        F: FnMut(&str),
    {
        for entry in &self.entries {
            callback(&entry.description);
        }
    }

    /// Print stack to a string.
    #[deprecated(note = "use print_into() or print_with() to avoid allocating a String")]
    pub fn print(&self) -> String {
        let mut output = String::new();
        self.print_into(&mut output)
            .expect("writing to String cannot fail");
        output
    }

    /// Version-2 print alias.
    #[deprecated(note = "use print_into() or print_with() to avoid allocating a String")]
    pub fn print2(&self) -> String {
        let mut output = String::new();
        self.print_into(&mut output)
            .expect("writing to String cannot fail");
        output
    }

    /// Walk entries from oldest to newest.
    pub fn walk<F>(&self, mut callback: F)
    where
        F: FnMut(&ErrorStackEntry),
    {
        for entry in &self.entries {
            callback(entry);
        }
    }

    /// Version-1 walk alias.
    pub fn walk1<F>(&self, callback: F)
    where
        F: FnMut(&ErrorStackEntry),
    {
        self.walk(callback);
    }

    /// Internal walk alias.
    pub fn walk_internal<F>(&self, callback: F)
    where
        F: FnMut(&ErrorStackEntry),
    {
        self.walk(callback);
    }

    /// Internal v1 walk-callback adapter.
    pub fn walk1_cb<F>(&self, callback: F)
    where
        F: FnMut(&ErrorStackEntry),
    {
        self.walk(callback);
    }

    /// Internal v2 walk-callback adapter.
    pub fn walk2_cb<F>(&self, callback: F)
    where
        F: FnMut(&ErrorStackEntry),
    {
        self.walk(callback);
    }

    /// Enable/disable default automatic error printing.
    pub fn set_default_auto(&mut self, enabled: bool) {
        self.auto_enabled = enabled;
    }

    /// Set automatic error printing.
    pub fn set_auto(&mut self, enabled: bool) {
        self.auto_enabled = enabled;
    }

    /// Version-1 automatic error printing setter.
    pub fn set_auto1(&mut self, enabled: bool) {
        self.set_auto(enabled);
    }

    /// Return automatic error printing state.
    pub fn get_auto(&self) -> bool {
        self.auto_enabled
    }

    /// Return default automatic error printing state.
    pub fn get_default_auto_func(&self) -> bool {
        self.auto_enabled
    }

    /// Version-1 automatic error printing getter.
    pub fn get_auto1(&self) -> bool {
        self.auto_enabled
    }

    /// Pause automatic error printing.
    pub fn pause_stack(&mut self) {
        self.auto_enabled = false;
    }

    /// Resume automatic error printing.
    pub fn resume_stack(&mut self) {
        self.auto_enabled = true;
    }

    /// Return the major message for the top entry.
    pub fn get_major(&self) -> Option<&str> {
        self.entries.last().map(|entry| entry.major.as_str())
    }

    /// Return the minor message for the top entry.
    pub fn get_minor(&self) -> Option<&str> {
        self.entries.last().map(|entry| entry.minor.as_str())
    }
}

#[cfg(test)]
mod error_stack_tests {
    use super::*;

    #[test]
    fn error_stack_aliases_roundtrip() {
        let class = ErrorClass::register_class("hdf5", "major", "1.0");
        assert_eq!(class.class_name(), "major");
        let msg = ErrorMessage::create_msg("minor");
        assert_eq!(msg.get_msg(), "minor");
        assert_eq!(msg.get_msg_internal(), "minor");

        let mut stack = ErrorStack::init();
        assert!(stack.user_cb_prepare());
        stack.push_stack(ErrorStackEntry::new("major", "io", "open", "failed"));
        assert_eq!(stack.get_num(), 1);
        assert_eq!(stack.get_major(), Some("io"));
        assert_eq!(stack.get_minor(), Some("open"));
        let mut printed = String::new();
        stack.print_into(&mut printed).unwrap();
        assert_eq!(printed, "failed");

        let mut walked = Vec::new();
        stack.print_with(|description| walked.push(description.to_owned()));
        assert_eq!(walked, vec!["failed"]);

        let copy = stack.get_current_stack();
        stack.append_stack(&copy);
        assert_eq!(stack.get_num_internal(), 2);
        stack.pop_internal(1);
        assert_eq!(stack.get_num(), 1);
        stack.set_auto1(true);
        assert!(stack.get_auto1());
        stack.pause_stack();
        assert!(!stack.get_auto());
        stack.resume_stack();
        assert!(stack.get_default_auto_func());
        stack.clear_stack();
        assert_eq!(stack.get_num(), 0);
    }

    #[test]
    fn error_event_set_aliases_roundtrip() {
        let mut event = ErrorEvent::event_new("read");
        assert!(!event.completed);
        event.event_completed();
        assert!(event.completed);

        let mut failed = ErrorEvent::event_new("write");
        failed.handle_fail("disk");

        let mut set = ErrorEventSet::create();
        set.list_append(event);
        set.list_append(failed);
        set.insert_request("flush");
        assert_eq!(set.list_count(), 3);
        assert_eq!(set.get_count(), 3);
        assert_eq!(set.get_op_counter(), 3);
        assert_eq!(
            set.requests().collect::<Vec<_>>(),
            vec!["read", "write", "flush"]
        );
        assert!(set.get_err_status());
        assert_eq!(set.get_err_count(), 1);
        assert_eq!(set.err_info().collect::<Vec<_>>(), vec!["disk"]);

        let mut visited = Vec::new();
        set.list_iterate(|event| visited.push(event.request.clone()));
        assert_eq!(visited, vec!["read", "write", "flush"]);

        assert_eq!(set.wait(), 1);
        set.close_failed_cb();
        let mut requests = Vec::new();
        set.get_requests_with(|request| requests.push(request.to_owned()));
        assert_eq!(requests, vec!["read", "flush"]);
        assert_eq!(set.cancel(), 0);
        assert_eq!(
            set.list_remove(1).map(|event| event.request),
            Some("flush".into())
        );
        set.term_package();
        assert_eq!(set.requests().count(), 0);
    }

    #[test]
    #[allow(deprecated)]
    fn deprecated_allocating_error_wrappers_remain_callable() {
        let mut set = ErrorEventSet::create();
        set.insert_request("read");
        assert_eq!(set.get_requests(), vec!["read"]);

        let mut failed = ErrorEvent::event_new("write");
        failed.handle_fail("disk");
        set.list_append(failed);
        assert_eq!(set.get_err_info(), vec!["disk"]);

        let mut stack = ErrorStack::init();
        stack.push_stack(ErrorStackEntry::new("major", "io", "open", "failed"));
        assert_eq!(stack.print(), "failed");
        assert_eq!(stack.print2(), "failed");
    }
}
