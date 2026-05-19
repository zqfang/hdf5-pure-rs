//! Phase T2: Corrupt/malformed file tests.
//! Verify graceful error handling -- no panics, no UB.

use hdf5_pure_rust::{Dataset, File};

const REF_DIR: &str = "tests/data/hdf5_ref";

fn try_raw_read(ds: &Dataset) {
    const MAX_PROBE_RAW_BYTES: usize = 1024 * 1024;

    if let Ok(size) = ds.size() {
        if let Ok(element_size) = ds.element_size() {
            if let Some(raw_len) = (size as usize).checked_mul(element_size) {
                if raw_len > MAX_PROBE_RAW_BYTES {
                    return;
                }
                let mut raw = vec![0; raw_len];
                let _ = ds.read_raw_into(&mut raw);
            }
        }
    }
}

/// Try to open and fully explore a file. Returns Ok if no panics.
fn try_full_explore(filename: &str) -> Result<(), String> {
    let path = format!("{REF_DIR}/{filename}");
    let f = match File::open(&path) {
        Ok(f) => f,
        Err(e) => return Err(format!("open: {e}")),
    };

    // Try to list members (may fail gracefully)
    let mut names = Vec::new();
    let _ = f.visit_member_names(|name| {
        names.push(name.to_string());
        Ok(())
    });

    // Try to list attributes
    let mut attr_names = Vec::new();
    let _ = f.attr_names_into(&mut attr_names);

    // Try to navigate into groups/datasets
    if !names.is_empty() {
        for name in &names {
            let root = f.root_group().unwrap();
            // Try opening as group
            let _ = root.open_group(name);
            // Try opening as dataset
            if let Ok(ds) = root.open_dataset(name) {
                let mut shape = Vec::new();
                let _ = ds.shape_into(&mut shape);
                let _ = ds.dtype();
                try_raw_read(&ds);
            }
        }
    }
    Ok(())
}

// T2a: Corrupted structures -- should error, not panic

#[test]
fn t2a_corrupt_stab_msg() {
    // Should not panic
    let _ = try_full_explore("corrupt_stab_msg.h5");
}

#[test]
fn t2a_tbad_msg_count() {
    let _ = try_full_explore("tbad_msg_count.h5");
}

#[test]
fn t2a_tbogus() {
    let _ = try_full_explore("tbogus.h5");
}

#[test]
fn t2a_bad_compound() {
    let _ = try_full_explore("bad_compound.h5");
}

#[test]
fn t2a_bad_offset() {
    let _ = try_full_explore("bad_offset.h5");
}

// T2b: CVE regression tests -- must not panic

#[test]
fn t2b_cve_2020_10810() {
    let _ = try_full_explore("cve_2020_10810.h5");
}

#[test]
fn t2b_cve_2020_10812() {
    let _ = try_full_explore("cve_2020_10812.h5");
}

#[test]
fn t2b_memleak_dtype() {
    let _ = try_full_explore("memleak_H5O_dtype_decode_helper_H5Odtype.h5");
}

// T2c: Verify ALL reference files don't panic even when fully explored

#[test]
fn t2c_explore_all_no_panic() {
    let dir = std::fs::read_dir(REF_DIR).unwrap();
    let mut panicked = Vec::new();

    for entry in dir {
        let entry = entry.unwrap();
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.ends_with(".h5") {
            continue;
        }
        // Use catch_unwind to detect panics
        let result = std::panic::catch_unwind(|| {
            let _ = try_full_explore(&name);
        });
        if result.is_err() {
            panicked.push(name);
        }
    }

    if !panicked.is_empty() {
        panic!("Files that caused panics: {:?}", panicked);
    }
}
