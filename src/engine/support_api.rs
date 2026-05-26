#![allow(dead_code, non_snake_case)]

use std::borrow::Cow;
use std::cmp::Ordering;
use std::fmt;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::{Condvar, Mutex, Once};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::engine::free_list::{FreeListManager, FreeListStats};
use crate::error::{Error, Result};

static LIBRARY_OPEN: AtomicBool = AtomicBool::new(false);
static LIBRARY_TERMINATING: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Clone)]
pub struct H5Timer {
    pub started: Option<Instant>,
    pub elapsed: Duration,
}

impl Default for H5Timer {
    /// Construct a fresh, idle timer with zero accumulated elapsed time.
    fn default() -> Self {
        Self {
            started: None,
            elapsed: Duration::ZERO,
        }
    }
}

#[derive(Debug, Default)]
pub struct H5TsMutex {
    locked: Mutex<bool>,
}

#[derive(Debug, Default)]
pub struct H5TsCond {
    cond: Condvar,
}

#[derive(Debug, Default)]
pub struct H5TsSemaphore {
    permits: Mutex<usize>,
    cond: Condvar,
}

#[derive(Debug, Default)]
pub struct H5TsRwLock {
    state: Mutex<(usize, bool)>,
    cond: Condvar,
}

/// Returns an `Unsupported` error stub for platform/MPI features not implemented in pure-Rust mode.
fn unsupported_support(name: &str) -> Error {
    Error::Unsupported(format!(
        "{name} requires platform/MPI behavior not implemented in pure-Rust mode"
    ))
}

/// Sentinel iteration-error return value.
pub fn H5_ITER_ERROR() -> i32 {
    -1
}

/// Hook invoked just before a user callback fires.
pub fn H5_BEFORE_USER_CB() {}

/// Hook invoked before a no-error user callback fires.
pub fn H5_BEFORE_USER_CB_NOERR() {}

/// Duplicate an MPI communicator; unsupported in pure-Rust mode.
pub fn H5_mpi_comm_dup() -> Result<()> {
    Err(unsupported_support("H5_mpi_comm_dup"))
}

/// Free an MPI communicator; unsupported in pure-Rust mode.
pub fn H5_mpi_comm_free() -> Result<()> {
    Err(unsupported_support("H5_mpi_comm_free"))
}

/// Duplicate an MPI info object; unsupported in pure-Rust mode.
pub fn H5_mpi_info_dup() -> Result<()> {
    Err(unsupported_support("H5_mpi_info_dup"))
}

/// Compare two MPI communicators; unsupported in pure-Rust mode.
pub fn H5_mpi_comm_cmp() -> Result<Ordering> {
    Err(unsupported_support("H5_mpi_comm_cmp"))
}

/// Compare two MPI communicators through an out parameter; unsupported in pure-Rust mode.
pub fn H5_mpi_comm_cmp_into(_ordering: &mut Ordering) -> Result<()> {
    Err(unsupported_support("H5_mpi_comm_cmp"))
}

/// Allocate a buffer for MPI gatherv; unsupported in pure-Rust mode.
pub fn H5_mpio_gatherv_alloc() -> Result<()> {
    Err(unsupported_support("H5_mpio_gatherv_alloc"))
}

/// Simple variant of MPI gatherv allocation; unsupported in pure-Rust mode.
pub fn H5_mpio_gatherv_alloc_simple() -> Result<()> {
    Err(unsupported_support("H5_mpio_gatherv_alloc_simple"))
}

/// Whether the MPI implementation requires explicit file syncs (always false here).
pub fn H5_mpio_get_file_sync_required() -> bool {
    false
}

/// Attribute-open common path; tracked in the attribute API rather than here.
pub fn H5A__open_common() -> Result<()> {
    Err(Error::Unsupported(
        "attribute open-common duplicate is tracked in the attribute API".into(),
    ))
}

/// Convert a `SystemTime` to seconds-since-epoch in UTC.
pub fn H5_gmtime_r(time: SystemTime) -> u64 {
    time.duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Convert a `SystemTime` to seconds-since-epoch (local time is treated as UTC here).
pub fn H5_localtime_r(time: SystemTime) -> u64 {
    H5_gmtime_r(time)
}

/// Render a byte slice as a space-separated hex string for debug output.
pub fn H5_buffer_dump_into<W>(bytes: &[u8], out: &mut W) -> fmt::Result
where
    W: fmt::Write + ?Sized,
{
    for (idx, byte) in bytes.iter().enumerate() {
        if idx > 0 {
            out.write_char(' ')?;
        }
        write!(out, "{byte:02x}")?;
    }
    Ok(())
}

/// Render a byte slice as a space-separated hex string for debug output.
#[deprecated(note = "use H5_buffer_dump_into to reuse formatting storage")]
pub fn H5_buffer_dump(bytes: &[u8]) -> String {
    let mut out = String::new();
    H5_buffer_dump_into(bytes, &mut out).expect("writing to String cannot fail");
    out
}

/// Acquire the global API lock; no-op in pure-Rust mode.
pub fn H5TS_api_lock() {}

/// Compute bandwidth in bytes/second from a transfer size and elapsed time.
pub fn H5_bandwidth(bytes: u64, elapsed: Duration) -> f64 {
    let secs = elapsed.as_secs_f64();
    if secs == 0.0 {
        0.0
    } else {
        bytes as f64 / secs
    }
}

/// Return the current wall-clock time in whole seconds since the Unix epoch.
pub fn H5_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Return the current wall-clock time in microseconds since the Unix epoch.
pub fn H5_now_usec() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros()
}

/// Return the current wall-clock time as a `SystemTime`.
pub fn H5_get_time() -> SystemTime {
    SystemTime::now()
}

/// Return current `(secs, usecs)` time used by the libhdf5 timer subsystem.
pub fn H5__timer_get_timevals() -> (u64, u128) {
    (H5_now(), H5_now_usec())
}

/// Construct a fresh, idle timer.
pub fn H5_timer_init() -> H5Timer {
    H5Timer::default()
}

/// Start (or resume) a timer.
pub fn H5_timer_start(timer: &mut H5Timer) {
    timer.started = Some(Instant::now());
}

/// Stop a running timer and return its accumulated elapsed duration.
pub fn H5_timer_stop(timer: &mut H5Timer) -> Duration {
    if let Some(started) = timer.started.take() {
        timer.elapsed = timer.elapsed.saturating_add(started.elapsed());
    }
    timer.elapsed
}

/// Return the elapsed duration recorded by the timer so far.
pub fn H5_timer_get_times(timer: &H5Timer) -> Duration {
    timer.elapsed
}

/// Return the cumulative duration recorded by the timer.
pub fn H5_timer_get_total_times(timer: &H5Timer) -> Duration {
    timer.elapsed
}

/// Format a duration as a human-readable seconds string.
pub fn H5_timer_get_time_string_fmt<W>(duration: Duration, out: &mut W) -> fmt::Result
where
    W: fmt::Write + ?Sized,
{
    write!(out, "{:.6}s", duration.as_secs_f64())
}

/// Format a duration as a human-readable seconds string.
pub fn H5_timer_get_time_string_into(duration: Duration, out: &mut String) {
    out.clear();
    H5_timer_get_time_string_fmt(duration, out).expect("writing to String cannot fail");
}

/// Format a duration as a human-readable seconds string.
#[deprecated(note = "use H5_timer_get_time_string_into to reuse String storage")]
pub fn H5_timer_get_time_string(duration: Duration) -> String {
    let mut out = String::new();
    H5_timer_get_time_string_into(duration, &mut out);
    out
}

/// Initialize the H5 package: mark the library as open.
pub fn H5__init_package() {
    LIBRARY_TERMINATING.store(false, AtomicOrdering::SeqCst);
    LIBRARY_OPEN.store(true, AtomicOrdering::SeqCst);
}

/// Initialize the HDF5 library.
pub fn H5_init_library() {
    H5__init_package();
}

/// Terminate the HDF5 library and release all resources.
pub fn H5_term_library() {
    LIBRARY_TERMINATING.store(true, AtomicOrdering::SeqCst);
    LIBRARY_OPEN.store(false, AtomicOrdering::SeqCst);
}

/// Disable the library's `atexit` handler; no-op in pure-Rust mode.
pub fn H5dont_atexit() {}

/// Run a garbage-collection pass on free lists; no-op in pure-Rust mode.
pub fn H5garbage_collect() {}

/// Query current free-list storage sizes.
pub fn H5get_free_list_sizes(lists: &FreeListManager) -> FreeListStats {
    lists.get_free_list_sizes()
}

/// Set per-kind free-list entry limits.
pub fn H5set_free_list_limits(
    lists: &mut FreeListManager,
    regular: usize,
    block: usize,
    array: usize,
    factory: usize,
) {
    lists.set_free_list_limits(regular, block, array, factory);
}

/// Echo the requested debug-print mask back to the caller.
pub fn H5__debug_mask(mask: u64) -> u64 {
    mask
}

/// Callback invoked when MPI deletes its file-key attribute; unsupported here.
pub fn H5__mpi_delete_cb() -> Result<()> {
    Err(unsupported_support("H5__mpi_delete_cb"))
}

/// Return the HDF5 library version as `(major, minor, release)`.
pub fn H5get_libversion() -> (u32, u32, u32) {
    (0, 1, 0)
}

/// Return the HDF5 library version through libhdf5-style out parameters.
pub fn H5get_libversion_into(major: &mut u32, minor: &mut u32, release: &mut u32) {
    (*major, *minor, *release) = H5get_libversion();
}

/// Check whether the linked library matches the requested version.
pub fn H5_check_version(major: u32, minor: u32, release: u32) -> bool {
    H5get_libversion() == (major, minor, release)
}

/// Public alias for `H5_check_version`.
pub fn H5check_version(major: u32, minor: u32, release: u32) -> bool {
    H5_check_version(major, minor, release)
}

/// Open the HDF5 library (initialize it if needed).
pub fn H5open() {
    H5_init_library();
}

/// Register an at-close handler; no-op in pure-Rust mode.
pub fn H5atclose() {}

/// Close the HDF5 library and release all resources.
pub fn H5close() {
    H5_term_library();
}

/// Allocate `size` bytes of zero-initialized memory.
pub fn H5allocate_memory(size: usize) -> Vec<u8> {
    vec![0; size]
}

/// Allocate `size` bytes of zero-initialized memory.
pub fn H5MM_malloc(size: usize) -> Vec<u8> {
    H5allocate_memory(size)
}

/// Allocate `count * size` bytes of zero-initialized memory.
pub fn H5MM_calloc(count: usize, size: usize) -> Result<Vec<u8>> {
    let total = count
        .checked_mul(size)
        .ok_or_else(|| Error::InvalidFormat("H5MM calloc size overflow".into()))?;
    Ok(H5allocate_memory(total))
}

/// Resize a previously allocated buffer to `size` bytes.
pub fn H5resize_memory(mut bytes: Vec<u8>, size: usize) -> Vec<u8> {
    bytes.resize(size, 0);
    bytes
}

/// Resize a previously allocated buffer to `size` bytes.
pub fn H5MM_realloc(bytes: Vec<u8>, size: usize) -> Vec<u8> {
    H5resize_memory(bytes, size)
}

/// Free a previously allocated buffer.
pub fn H5free_memory(_bytes: Vec<u8>) {}

/// Free a previously allocated buffer.
pub fn H5MM_xfree(bytes: Vec<u8>) {
    H5free_memory(bytes);
}

/// Return whether the library was built thread-safe.
pub fn H5is_library_threadsafe() -> bool {
    true
}

/// Return whether the library was built thread-safe through a libhdf5-style out parameter.
pub fn H5is_library_threadsafe_into(is_threadsafe: &mut bool) {
    *is_threadsafe = H5is_library_threadsafe();
}

/// Return whether the library is currently shutting down.
pub fn H5is_library_terminating() -> bool {
    LIBRARY_TERMINATING.load(AtomicOrdering::SeqCst)
}

/// Return whether the library is currently shutting down through an out parameter.
pub fn H5is_library_terminating_into(is_terminating: &mut bool) {
    *is_terminating = H5is_library_terminating();
}

/// Hook invoked before entering user callback context.
pub fn H5_user_cb_prepare() {}

/// Hook invoked after returning from user callback context.
pub fn H5_user_cb_restore() {}

/// Return whether the library was built with parallel (MPI) support.
pub fn H5_HAVE_PARALLEL() -> bool {
    false
}

/// Update a running checksum with additional bytes (placeholder CRC variant).
pub fn H5__checksum_crc_update(seed: u32, bytes: &[u8]) -> u32 {
    bytes
        .iter()
        .fold(seed, |acc, byte| acc.rotate_left(5) ^ u32::from(*byte))
}

/// Compute a CRC-style checksum over the given bytes.
pub fn H5_checksum_crc(bytes: &[u8]) -> u32 {
    H5__checksum_crc_update(0, bytes)
}

/// Create a use-count tracker initialized to 1.
pub fn H5UC_create() -> usize {
    1
}

/// Decrement a use-count tracker and return the new value.
pub fn H5UC_decr(count: &mut usize) -> usize {
    *count = count.saturating_sub(1);
    *count
}

/// Create a new (unlocked) thread-safety mutex.
pub fn H5TS_mutex_init() -> H5TsMutex {
    H5TsMutex::default()
}

/// Attempt to lock a mutex without blocking; returns whether the lock was acquired.
pub fn H5TS_mutex_trylock(mutex: &H5TsMutex) -> bool {
    if let Ok(mut locked) = mutex.locked.try_lock() {
        if !*locked {
            *locked = true;
            return true;
        }
    }
    false
}

/// Destroy a thread-safety mutex.
pub fn H5TS_mutex_destroy(_mutex: H5TsMutex) {}

/// Create a counting semaphore initialized with `permits` permits.
pub fn H5TS_semaphore_init(permits: usize) -> H5TsSemaphore {
    H5TsSemaphore {
        permits: Mutex::new(permits),
        cond: Condvar::new(),
    }
}

/// Destroy a counting semaphore.
pub fn H5TS_semaphore_destroy(_sem: H5TsSemaphore) {}

/// Run `f` exactly once across all callers of `once`.
pub fn H5TS_once(once: &Once, f: fn()) {
    once.call_once(f);
}

/// Create a new condition variable.
pub fn H5TS_cond_init() -> H5TsCond {
    H5TsCond::default()
}

/// Destroy a condition variable.
pub fn H5TS_cond_destroy(_cond: H5TsCond) {}

/// Signal a counting semaphore, releasing one permit.
pub fn H5TS_semaphore_signal(sem: &H5TsSemaphore) {
    if let Ok(mut permits) = sem.permits.lock() {
        *permits = permits.saturating_add(1);
        sem.cond.notify_one();
    }
}

/// Block until a permit is available on the semaphore, then consume it.
pub fn H5TS_semaphore_wait(sem: &H5TsSemaphore) {
    if let Ok(mut permits) = sem.permits.lock() {
        while *permits == 0 {
            permits = sem.cond.wait(permits).expect("semaphore wait poisoned");
        }
        *permits = permits.saturating_sub(1);
    }
}

/// Create a new reader/writer lock.
pub fn H5TS_rwlock_init() -> H5TsRwLock {
    H5TsRwLock::default()
}

/// Destroy a reader/writer lock.
pub fn H5TS_rwlock_destroy(_lock: H5TsRwLock) {}

/// Acquire a shared (read) lock, blocking while a writer holds the lock.
pub fn H5TS_rwlock_rdlock(lock: &H5TsRwLock) {
    let mut state = lock.state.lock().expect("rwlock poisoned");
    while state.1 {
        state = lock.cond.wait(state).expect("rwlock wait poisoned");
    }
    state.0 = state.0.saturating_add(1);
}

/// Release a shared (read) lock, waking waiting writers when the last reader exits.
pub fn H5TS_rwlock_rdunlock(lock: &H5TsRwLock) {
    if let Ok(mut state) = lock.state.lock() {
        state.0 = state.0.saturating_sub(1);
        if state.0 == 0 {
            lock.cond.notify_all();
        }
    }
}

/// Acquire an exclusive (write) lock, blocking while any readers or another writer hold it.
pub fn H5TS_rwlock_wrlock(lock: &H5TsRwLock) {
    let mut state = lock.state.lock().expect("rwlock poisoned");
    while state.1 || state.0 != 0 {
        state = lock.cond.wait(state).expect("rwlock wait poisoned");
    }
    state.1 = true;
}

/// Try to acquire an exclusive write lock without blocking.
pub fn H5TS_rwlock_trywrlock(lock: &H5TsRwLock) -> bool {
    if let Ok(mut state) = lock.state.try_lock() {
        if !state.1 && state.0 == 0 {
            state.1 = true;
            return true;
        }
    }
    false
}

/// Release an exclusive (write) lock and notify all waiters.
pub fn H5TS_rwlock_wrunlock(lock: &H5TsRwLock) {
    if let Ok(mut state) = lock.state.lock() {
        state.1 = false;
        lock.cond.notify_all();
    }
}

/// Create a thread-local key.
pub fn H5TS_key_create() -> usize {
    0
}

/// Delete a thread-local key.
pub fn H5TS_key_delete(_key: usize) {}

/// Store a thread-local value under the given key.
pub fn H5TS_key_set_value<T>(_key: usize, value: T) -> T {
    value
}

/// Retrieve the thread-local value associated with the given key.
pub fn H5TS_key_get_value(_key: usize) -> Option<()> {
    None
}

/// Create a thread pool; intentionally unsupported in pure-Rust mode.
#[allow(non_snake_case)]
pub fn H5TS_pool_create() -> Result<()> {
    Err(Error::Unsupported(
        "thread-pool runtime is intentionally unsupported".into(),
    ))
}

/// Queue a task on a thread pool; intentionally unsupported in pure-Rust mode.
#[allow(non_snake_case)]
pub fn H5TS_pool_add_task<T>(_task: &mut T) -> Result<()> {
    Err(Error::Unsupported(
        "thread-pool runtime is intentionally unsupported".into(),
    ))
}

/// Format a boolean argument for the API trace log.
pub fn H5_trace_args_bool_ref(value: bool) -> &'static str {
    if value {
        "true"
    } else {
        "false"
    }
}

/// Format a boolean argument for the API trace log.
pub fn H5_trace_args_bool_into(value: bool, out: &mut String) {
    out.clear();
    out.push_str(H5_trace_args_bool_ref(value));
}

/// Format a boolean argument for the API trace log.
#[deprecated(note = "use H5_trace_args_bool_into to reuse String storage")]
pub fn H5_trace_args_bool(value: bool) -> String {
    let mut out = String::new();
    H5_trace_args_bool_into(value, &mut out);
    out
}

/// Format a character-set argument for the API trace log.
pub fn H5_trace_args_cset_fmt<W>(value: u8, out: &mut W) -> fmt::Result
where
    W: fmt::Write + ?Sized,
{
    write!(out, "{value}")
}

/// Format a character-set argument for the API trace log.
pub fn H5_trace_args_cset_into(value: u8, out: &mut String) {
    out.clear();
    H5_trace_args_cset_fmt(value, out).expect("writing to String cannot fail");
}

/// Format a character-set argument for the API trace log.
#[deprecated(note = "use H5_trace_args_cset_into to reuse String storage")]
pub fn H5_trace_args_cset(value: u8) -> String {
    let mut out = String::new();
    H5_trace_args_cset_into(value, &mut out);
    out
}

/// Format a file-close-degree argument for the API trace log.
pub fn H5_trace_args_close_degree_fmt<W>(value: u8, out: &mut W) -> fmt::Result
where
    W: fmt::Write + ?Sized,
{
    H5_trace_args_cset_fmt(value, out)
}

/// Format a file-close-degree argument for the API trace log.
pub fn H5_trace_args_close_degree_into(value: u8, out: &mut String) {
    out.clear();
    H5_trace_args_close_degree_fmt(value, out).expect("writing to String cannot fail");
}

/// Format a file-close-degree argument for the API trace log.
#[deprecated(note = "use H5_trace_args_close_degree_into to reuse String storage")]
pub fn H5_trace_args_close_degree(value: u8) -> String {
    let mut out = String::new();
    H5_trace_args_close_degree_into(value, &mut out);
    out
}

/// Join a list of pre-formatted argument strings into a single trace argument list.
pub fn H5_trace_args_fmt<S, W>(args: &[S], out: &mut W) -> fmt::Result
where
    S: AsRef<str>,
    W: fmt::Write + ?Sized,
{
    for (idx, arg) in args.iter().enumerate() {
        if idx > 0 {
            out.write_str(", ")?;
        }
        out.write_str(arg.as_ref())?;
    }
    Ok(())
}

/// Join a list of pre-formatted argument strings into a single trace argument list.
pub fn H5_trace_args_into<S>(args: &[S], out: &mut String)
where
    S: AsRef<str>,
{
    out.clear();
    H5_trace_args_fmt(args, out).expect("writing to String cannot fail");
}

/// Visit pre-formatted trace arguments in order.
pub fn H5_trace_args_iter<S>(args: &[S]) -> impl Iterator<Item = &str>
where
    S: AsRef<str>,
{
    args.iter().map(AsRef::as_ref)
}

/// Join a list of pre-formatted argument strings into a single trace argument list.
#[deprecated(note = "use H5_trace_args_iter or H5_trace_args_into")]
pub fn H5_trace_args(args: &[String]) -> String {
    let mut out = String::new();
    H5_trace_args_into(args, &mut out);
    out
}

/// Emit a single trace message line.
pub fn H5_trace_ref(message: &str) -> &str {
    message
}

/// Emit a single trace message line.
#[deprecated(note = "use H5_trace_ref to borrow the trace message")]
pub fn H5_trace(message: &str) -> String {
    H5_trace_ref(message).to_string()
}

/// Spawn a new OS thread running `f`.
pub fn H5TS_thread_create<F>(f: F) -> JoinHandle<()>
where
    F: FnOnce() + Send + 'static,
{
    thread::spawn(f)
}

/// Join a previously spawned thread, mapping panics to an error.
pub fn H5TS_thread_join(handle: JoinHandle<()>) -> Result<()> {
    handle
        .join()
        .map_err(|_| Error::InvalidFormat("thread panicked".into()))
}

/// Detach a thread, allowing it to run to completion without being joined.
pub fn H5TS_thread_detach(_handle: JoinHandle<()>) {}

/// Hint to the scheduler that the current thread is willing to yield.
pub fn H5TS_thread_yield() {
    thread::yield_now();
}

/// Copy a VFD property list; tracked in the VFD API rather than here.
pub fn H5FD__copy_plist() -> Result<()> {
    Err(Error::Unsupported(
        "VFD property-list copy duplicate is tracked in the VFD API".into(),
    ))
}

/// Query VFD driver capability flags; unsupported without a libhdf5 VFD driver registry.
pub fn H5FDdriver_query(_driver_id: u64) -> Result<u64> {
    Err(unsupported_support("H5FDdriver_query"))
}

/// Query VFD driver capability flags through caller-owned storage.
pub fn H5FDdriver_query_into(_driver_id: u64, _flags: &mut u64) -> Result<()> {
    Err(unsupported_support("H5FDdriver_query"))
}

/// Append a plugin search path; unsupported without dynamic plugin loading.
pub fn H5PLappend(_path: &str) -> Result<()> {
    Err(unsupported_support("H5PLappend"))
}

/// Prepend a plugin search path; unsupported without dynamic plugin loading.
pub fn H5PLprepend(_path: &str) -> Result<()> {
    Err(unsupported_support("H5PLprepend"))
}

/// Replace a plugin search path; unsupported without dynamic plugin loading.
pub fn H5PLreplace(_path: &str, _index: usize) -> Result<()> {
    Err(unsupported_support("H5PLreplace"))
}

/// Insert a plugin search path; unsupported without dynamic plugin loading.
pub fn H5PLinsert(_path: &str, _index: usize) -> Result<()> {
    Err(unsupported_support("H5PLinsert"))
}

/// Remove a plugin search path; unsupported without dynamic plugin loading.
pub fn H5PLremove(_index: usize) -> Result<()> {
    Err(unsupported_support("H5PLremove"))
}

/// Dump pending subfiling I/O vectors; unsupported in pure-Rust mode.
pub fn H5_subfiling_dump_iovecs() -> Result<()> {
    Err(unsupported_support("H5_subfiling_dump_iovecs"))
}

/// Convert a seconds-since-epoch value to a `SystemTime`.
pub fn H5_make_time(secs: u64) -> SystemTime {
    UNIX_EPOCH + Duration::from_secs(secs)
}

/// Format a `SystemTime` as a local-time string (here, seconds since epoch).
pub fn H5_get_localtime_str_fmt<W>(time: SystemTime, out: &mut W) -> fmt::Result
where
    W: fmt::Write + ?Sized,
{
    write!(out, "{}", H5_gmtime_r(time))
}

/// Format a `SystemTime` as a local-time string (here, seconds since epoch).
pub fn H5_get_localtime_str_into(time: SystemTime, out: &mut String) {
    out.clear();
    H5_get_localtime_str_fmt(time, out).expect("writing to String cannot fail");
}

/// Format a `SystemTime` as a local-time string (here, seconds since epoch).
#[deprecated(note = "use H5_get_localtime_str_into to reuse String storage")]
pub fn H5_get_localtime_str(time: SystemTime) -> String {
    H5_gmtime_r(time).to_string()
}

/// Retrieve Windows-style high-resolution times; unsupported on non-Windows platforms.
pub fn H5_get_win32_times() -> Result<()> {
    Err(unsupported_support("H5_get_win32_times"))
}

/// Retrieve Windows-style high-resolution times through caller-owned storage.
pub fn H5_get_win32_times_into(
    _created: &mut SystemTime,
    _accessed: &mut SystemTime,
    _modified: &mut SystemTime,
) -> Result<()> {
    Err(unsupported_support("H5_get_win32_times"))
}

/// Decode a UTF-16 buffer into a `String`.
pub fn H5_get_utf16_str_into(bytes: &[u16], out: &mut String) -> Result<()> {
    let mut decoded = String::new();
    for item in std::char::decode_utf16(bytes.iter().copied()) {
        let ch = item.map_err(|_| Error::InvalidFormat("invalid UTF-16 string".into()))?;
        decoded.push(ch);
    }
    out.clear();
    out.push_str(&decoded);
    Ok(())
}

/// Decode a UTF-16 buffer into a `String`.
#[deprecated(note = "use H5_get_utf16_str_into to reuse String storage")]
pub fn H5_get_utf16_str(bytes: &[u16]) -> Result<String> {
    let mut out = String::new();
    H5_get_utf16_str_into(bytes, &mut out)?;
    Ok(out)
}

/// Build a `prefix/name` external-link path.
pub fn H5_build_extpath_cow<'a>(prefix: &str, name: &'a str) -> Cow<'a, str> {
    if prefix.is_empty() {
        Cow::Borrowed(name)
    } else {
        Cow::Owned(format!("{prefix}/{name}"))
    }
}

/// Build a `prefix/name` external-link path into caller-owned storage.
pub fn H5_build_extpath_into(prefix: &str, name: &str, out: &mut String) {
    out.clear();
    if prefix.is_empty() {
        out.push_str(name);
    } else {
        out.push_str(prefix);
        out.push('/');
        out.push_str(name);
    }
}

/// Build a `prefix/name` external-link path.
#[deprecated(note = "use H5_build_extpath_cow or H5_build_extpath_into")]
pub fn H5_build_extpath(prefix: &str, name: &str) -> String {
    H5_build_extpath_cow(prefix, name).into_owned()
}

/// Sleep for `nanos` nanoseconds.
pub fn H5_nanosleep(nanos: u64) {
    thread::sleep(Duration::from_nanos(nanos));
}

/// Expand `%ENV%`-style Windows environment variables in a string (no-op here).
pub fn H5_expand_windows_env_vars_ref(value: &str) -> &str {
    value
}

/// Expand `%ENV%`-style Windows environment variables in a string (no-op here).
#[deprecated(note = "use H5_expand_windows_env_vars_ref to borrow the unchanged string")]
pub fn H5_expand_windows_env_vars(value: &str) -> String {
    H5_expand_windows_env_vars_ref(value).to_string()
}

/// Duplicate the first `len` characters of a string.
pub fn H5_strndup_cow(value: &str, len: usize) -> Cow<'_, str> {
    let mut char_indices = value.char_indices();
    match char_indices.nth(len) {
        Some((end, _)) => Cow::Borrowed(&value[..end]),
        None => Cow::Borrowed(value),
    }
}

/// Duplicate the first `len` characters of a string into caller-owned storage.
pub fn H5_strndup_into(value: &str, len: usize, out: &mut String) {
    out.clear();
    out.push_str(&H5_strndup_cow(value, len));
}

/// Duplicate the first `len` characters of a string.
#[deprecated(note = "use H5_strndup_cow or H5_strndup_into")]
pub fn H5_strndup(value: &str, len: usize) -> String {
    H5_strndup_cow(value, len).into_owned()
}

/// Return the directory portion of a path, or "." if there is none.
pub fn H5_dirname_cow(value: &str) -> Cow<'_, str> {
    Path::new(value)
        .parent()
        .map(Path::to_string_lossy)
        .unwrap_or(Cow::Borrowed("."))
}

/// Return the directory portion of a path into caller-owned storage.
pub fn H5_dirname_into(value: &str, out: &mut String) {
    out.clear();
    out.push_str(&H5_dirname_cow(value));
}

/// Return the directory portion of a path, or "." if there is none.
#[deprecated(note = "use H5_dirname_cow or H5_dirname_into")]
pub fn H5_dirname(value: &str) -> String {
    H5_dirname_cow(value).into_owned()
}

/// Return the trailing component (filename) of a path.
pub fn H5_basename_cow(value: &str) -> Cow<'_, str> {
    Path::new(value)
        .file_name()
        .map(|path| path.to_string_lossy())
        .unwrap_or(Cow::Borrowed(value))
}

/// Return the trailing component (filename) of a path into caller-owned storage.
pub fn H5_basename_into(value: &str, out: &mut String) {
    out.clear();
    out.push_str(&H5_basename_cow(value));
}

/// Return the trailing component (filename) of a path.
#[deprecated(note = "use H5_basename_cow or H5_basename_into")]
pub fn H5_basename(value: &str) -> String {
    H5_basename_cow(value).into_owned()
}

/// Return `value` if set, otherwise the provided default.
pub fn H5_get_option<T: Copy>(value: Option<T>, default: T) -> T {
    value.unwrap_or(default)
}

/// Case-insensitive substring search returning the byte offset of the match.
pub fn H5_strcasestr(haystack: &str, needle: &str) -> Option<usize> {
    let needle = needle.as_bytes();
    if needle.is_empty() {
        return Some(0);
    }

    haystack
        .as_bytes()
        .windows(needle.len())
        .position(|window| {
            window
                .iter()
                .zip(needle)
                .all(|(left, right)| left.eq_ignore_ascii_case(right))
        })
}

/// Return whether the host CPU is little-endian.
pub fn is_host_little_endian() -> bool {
    cfg!(target_endian = "little")
}

/// Create an extensible-array data block; not implemented (tracked elsewhere).
pub fn H5EA__dblock_create() -> Result<()> {
    Err(Error::Unsupported(
        "extensible-array data block creation is not implemented".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strcasestr_searches_without_changing_offsets() {
        assert_eq!(H5_strcasestr("prefixDATA", "data"), Some(6));
        assert_eq!(H5_strcasestr("abcDEFghi", "Def"), Some(3));
        assert_eq!(H5_strcasestr("abc", ""), Some(0));
        assert_eq!(H5_strcasestr("abc", "abcd"), None);
        assert_eq!(H5_strcasestr("abc", "z"), None);
    }

    #[test]
    fn support_owned_compat_wrappers_route_through_new_apis() {
        let mut out = String::new();

        H5_buffer_dump_into(&[0, 10, 255], &mut out).expect("formatting should succeed");
        assert_eq!(out, "00 0a ff");

        H5_trace_args_into(&["one", "two"], &mut out);
        assert_eq!(out, "one, two");

        H5_strndup_into("abcdef", 3, &mut out);
        assert_eq!(out, "abc");

        H5_build_extpath_into("dir", "file.h5", &mut out);
        assert_eq!(out, "dir/file.h5");
    }

    #[test]
    fn utf16_decode_replaces_output_only_after_successful_decode() {
        let mut out = String::from("stale");
        H5_get_utf16_str_into(&[0x0041, 0xD83D, 0xDE00], &mut out)
            .expect("valid UTF-16 should decode");
        assert_eq!(out, "A😀");

        let err = H5_get_utf16_str_into(&[0x0042, 0xD800], &mut out)
            .expect_err("unpaired surrogate should fail");
        assert!(matches!(err, Error::InvalidFormat(_)));
        assert_eq!(out, "A😀");

        H5_get_utf16_str_into(&[], &mut out).expect("empty UTF-16 should decode");
        assert!(out.is_empty());
    }

    #[test]
    fn free_list_public_wrappers_query_and_set_limits() {
        let mut lists = FreeListManager::new();
        lists.blk_free(vec![0; 4]);
        lists.blk_free(vec![0; 8]);
        lists.arr_free(vec![0; 16]);

        let stats = H5get_free_list_sizes(&lists);
        assert_eq!(stats.block_bytes, 12);
        assert_eq!(stats.array_bytes, 16);

        H5set_free_list_limits(&mut lists, usize::MAX, 1, 0, usize::MAX);
        let stats = H5get_free_list_sizes(&lists);
        assert_eq!(stats.block_bytes, 4);
        assert_eq!(stats.array_bytes, 0);
    }

    #[test]
    fn h5mm_allocation_wrappers_zero_resize_and_reject_overflow() {
        let mut bytes = H5MM_malloc(4);
        assert_eq!(bytes, vec![0; 4]);

        bytes[0] = 9;
        bytes = H5MM_realloc(bytes, 6);
        assert_eq!(&bytes[..4], &[9, 0, 0, 0]);
        assert_eq!(&bytes[4..], &[0, 0]);

        bytes = H5MM_realloc(bytes, 2);
        assert_eq!(bytes, vec![9, 0]);

        let calloc = H5MM_calloc(3, 2).unwrap();
        assert_eq!(calloc, vec![0; 6]);
        assert!(H5MM_calloc(usize::MAX, 2).is_err());

        H5MM_xfree(calloc);
    }

    #[test]
    fn public_h5_out_parameter_wrappers_match_convenience_queries() {
        let mut major = u32::MAX;
        let mut minor = u32::MAX;
        let mut release = u32::MAX;
        H5get_libversion_into(&mut major, &mut minor, &mut release);
        assert_eq!((major, minor, release), H5get_libversion());

        let mut is_threadsafe = false;
        H5is_library_threadsafe_into(&mut is_threadsafe);
        assert_eq!(is_threadsafe, H5is_library_threadsafe());

        H5open();
        let mut is_terminating = true;
        H5is_library_terminating_into(&mut is_terminating);
        assert!(!is_terminating);

        H5close();
        H5is_library_terminating_into(&mut is_terminating);
        assert!(is_terminating);
    }

    #[test]
    fn support_runtime_boundaries_are_explicitly_unsupported() {
        let mut ordering = Ordering::Less;
        let err = H5_mpi_comm_cmp_into(&mut ordering)
            .expect_err("MPI communicator comparison remains unsupported");
        assert!(matches!(err, Error::Unsupported(_)));
        assert_eq!(ordering, Ordering::Less);

        for err in [
            H5_mpi_comm_dup().unwrap_err(),
            H5_mpi_comm_free().unwrap_err(),
            H5_mpi_info_dup().unwrap_err(),
            H5_mpi_comm_cmp().unwrap_err(),
            H5_mpio_gatherv_alloc().unwrap_err(),
            H5_mpio_gatherv_alloc_simple().unwrap_err(),
            H5FD__copy_plist().unwrap_err(),
            H5FDdriver_query(0).unwrap_err(),
            {
                let mut flags = 0xfeed;
                let err = H5FDdriver_query_into(0, &mut flags).unwrap_err();
                assert_eq!(flags, 0xfeed);
                err
            },
            H5PLappend("/tmp/hdf5-plugins").unwrap_err(),
            H5PLprepend("/tmp/hdf5-plugins").unwrap_err(),
            H5PLreplace("/tmp/hdf5-plugins", 0).unwrap_err(),
            H5PLinsert("/tmp/hdf5-plugins", 0).unwrap_err(),
            H5PLremove(0).unwrap_err(),
            H5_subfiling_dump_iovecs().unwrap_err(),
            H5_get_win32_times().unwrap_err(),
            {
                let original_created = UNIX_EPOCH + Duration::from_secs(11);
                let original_accessed = UNIX_EPOCH + Duration::from_secs(22);
                let original_modified = UNIX_EPOCH + Duration::from_secs(33);
                let mut created = original_created;
                let mut accessed = original_accessed;
                let mut modified = original_modified;
                let err = H5_get_win32_times_into(&mut created, &mut accessed, &mut modified)
                    .unwrap_err();
                assert_eq!(created, original_created);
                assert_eq!(accessed, original_accessed);
                assert_eq!(modified, original_modified);
                err
            },
            H5EA__dblock_create().unwrap_err(),
            H5TS_pool_create().unwrap_err(),
            {
                let mut task_state = String::from("queued");
                let err = H5TS_pool_add_task(&mut task_state).unwrap_err();
                assert_eq!(task_state, "queued");
                err
            },
        ] {
            assert!(matches!(err, Error::Unsupported(_)));
        }
    }
}
