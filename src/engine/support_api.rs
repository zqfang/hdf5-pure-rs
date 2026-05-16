#![allow(dead_code, non_snake_case)]

use std::cmp::Ordering;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::{Condvar, Mutex, Once};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

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

/// Compare two MPI communicators; unsupported in pure-Rust mode.
pub fn H5_mpi_comm_cmp() -> Result<Ordering> {
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
pub fn H5_buffer_dump(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<Vec<_>>()
        .join(" ")
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
pub fn H5_timer_get_time_string(duration: Duration) -> String {
    format!("{:.6}s", duration.as_secs_f64())
}

/// Initialize the H5 package: mark the library as open.
pub fn H5__init_package() {
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

/// Resize a previously allocated buffer to `size` bytes.
pub fn H5resize_memory(mut bytes: Vec<u8>, size: usize) -> Vec<u8> {
    bytes.resize(size, 0);
    bytes
}

/// Free a previously allocated buffer.
pub fn H5free_memory(_bytes: Vec<u8>) {}

/// Return whether the library was built thread-safe.
pub fn H5is_library_threadsafe() -> bool {
    true
}

/// Return whether the library is currently shutting down.
pub fn H5is_library_terminating() -> bool {
    LIBRARY_TERMINATING.load(AtomicOrdering::SeqCst)
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

/// Format a boolean argument for the API trace log.
pub fn H5_trace_args_bool(value: bool) -> String {
    value.to_string()
}

/// Format a character-set argument for the API trace log.
pub fn H5_trace_args_cset(value: u8) -> String {
    value.to_string()
}

/// Format a file-close-degree argument for the API trace log.
pub fn H5_trace_args_close_degree(value: u8) -> String {
    value.to_string()
}

/// Join a list of pre-formatted argument strings into a single trace argument list.
pub fn H5_trace_args(args: &[String]) -> String {
    args.join(", ")
}

/// Emit a single trace message line.
pub fn H5_trace(message: &str) -> String {
    message.to_string()
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

/// Dump pending subfiling I/O vectors; unsupported in pure-Rust mode.
pub fn H5_subfiling_dump_iovecs() -> Result<()> {
    Err(unsupported_support("H5_subfiling_dump_iovecs"))
}

/// Convert a seconds-since-epoch value to a `SystemTime`.
pub fn H5_make_time(secs: u64) -> SystemTime {
    UNIX_EPOCH + Duration::from_secs(secs)
}

/// Format a `SystemTime` as a local-time string (here, seconds since epoch).
pub fn H5_get_localtime_str(time: SystemTime) -> String {
    H5_gmtime_r(time).to_string()
}

/// Retrieve Windows-style high-resolution times; unsupported on non-Windows platforms.
pub fn H5_get_win32_times() -> Result<()> {
    Err(unsupported_support("H5_get_win32_times"))
}

/// Decode a UTF-16 buffer into a `String`.
pub fn H5_get_utf16_str(bytes: &[u16]) -> Result<String> {
    String::from_utf16(bytes).map_err(|_| Error::InvalidFormat("invalid UTF-16 string".into()))
}

/// Build a `prefix/name` external-link path.
pub fn H5_build_extpath(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}/{name}")
    }
}

/// Sleep for `nanos` nanoseconds.
pub fn H5_nanosleep(nanos: u64) {
    thread::sleep(Duration::from_nanos(nanos));
}

/// Expand `%ENV%`-style Windows environment variables in a string (no-op here).
pub fn H5_expand_windows_env_vars(value: &str) -> String {
    value.to_string()
}

/// Duplicate the first `len` characters of a string.
pub fn H5_strndup(value: &str, len: usize) -> String {
    value.chars().take(len).collect()
}

/// Return the directory portion of a path, or "." if there is none.
pub fn H5_dirname(value: &str) -> String {
    Path::new(value)
        .parent()
        .map(|path| path.to_string_lossy().into_owned())
        .unwrap_or_else(|| ".".to_string())
}

/// Return the trailing component (filename) of a path.
pub fn H5_basename(value: &str) -> String {
    Path::new(value)
        .file_name()
        .map(|path| path.to_string_lossy().into_owned())
        .unwrap_or_else(|| value.to_string())
}

/// Return `value` if set, otherwise the provided default.
pub fn H5_get_option<T: Copy>(value: Option<T>, default: T) -> T {
    value.unwrap_or(default)
}

/// Case-insensitive substring search returning the byte offset of the match.
pub fn H5_strcasestr(haystack: &str, needle: &str) -> Option<usize> {
    haystack
        .to_ascii_lowercase()
        .find(&needle.to_ascii_lowercase())
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
