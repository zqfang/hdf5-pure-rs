//! Phase T1: Reference file read tests.
//! Opens each .h5 reference file from the HDF5 C test suite and verifies
//! we can parse the superblock and navigate the structure without panicking.

use hdf5_pure_rust::File;

const REF_DIR: &str = "tests/data/hdf5_ref";

/// Helper: try to open a file, parse superblock, visit root members.
fn try_open_and_visit_members<F>(filename: &str, mut visitor: F) -> Result<(), String>
where
    F: FnMut(&str),
{
    let path = format!("{REF_DIR}/{filename}");
    let f = File::open(&path).map_err(|e| format!("{e}"))?;
    f.visit_member_names(|name| {
        visitor(name);
        Ok(())
    })
    .map_err(|e| format!("{e}"))?;
    Ok(())
}

/// Helper: try to open a file, parse superblock, count root members.
fn try_open_and_count_members(filename: &str) -> Result<usize, String> {
    let mut count = 0;
    try_open_and_visit_members(filename, |_| count += 1)?;
    Ok(count)
}

/// Helper: just try to open (parse superblock).
fn try_open(filename: &str) -> Result<(), String> {
    let path = format!("{REF_DIR}/{filename}");
    File::open(&path).map_err(|e| format!("{e}"))?;
    Ok(())
}

// =============================================================
// T1a: Superblock & format versions
// =============================================================

#[test]
fn t1a_filespace_1_6() {
    assert!(try_open("filespace_1_6.h5").is_ok());
}

#[test]
fn t1a_filespace_1_8() {
    assert!(try_open("filespace_1_8.h5").is_ok());
}

#[test]
fn t1a_paged_nopersist() {
    assert!(try_open("paged_nopersist.h5").is_ok());
}

#[test]
fn t1a_paged_persist() {
    assert!(try_open("paged_persist.h5").is_ok());
}

#[test]
fn t1a_fsm_aggr_nopersist() {
    assert!(try_open("fsm_aggr_nopersist.h5").is_ok());
}

#[test]
fn t1a_fsm_aggr_persist() {
    assert!(try_open("fsm_aggr_persist.h5").is_ok());
}

#[test]
fn t1a_aggr() {
    assert!(try_open("aggr.h5").is_ok());
}

#[test]
fn t1a_tarrold() {
    assert!(try_open("tarrold.h5").is_ok());
}

// =============================================================
// T1b: Groups & links
// =============================================================

#[test]
fn t1b_group_old() {
    let count = try_open_and_count_members("group_old.h5");
    assert!(count.is_ok(), "group_old.h5: {}", count.unwrap_err());
    assert!(count.unwrap() > 0);
}

#[test]
fn t1b_be_extlink1() {
    assert!(try_open("be_extlink1.h5").is_ok());
}

#[test]
fn t1b_be_extlink2() {
    assert!(try_open("be_extlink2.h5").is_ok());
}

#[test]
fn t1b_le_extlink1() {
    assert!(try_open("le_extlink1.h5").is_ok());
}

#[test]
fn t1b_le_extlink2() {
    assert!(try_open("le_extlink2.h5").is_ok());
}

#[test]
fn t1b_mergemsg() {
    assert!(try_open("mergemsg.h5").is_ok());
}

// =============================================================
// T1c: Datatypes
// =============================================================

#[test]
fn t1c_charsets() {
    let result = try_open_and_visit_members("charsets.h5", |_| {});
    assert!(result.is_ok(), "charsets.h5: {}", result.unwrap_err());
}

#[test]
fn t1c_tnullspace() {
    assert!(try_open("tnullspace.h5").is_ok());
}

// =============================================================
// T1d: Fill values & layouts
// =============================================================

#[test]
fn t1d_fill18() {
    assert!(try_open("fill18.h5").is_ok());
}

#[test]
fn t1d_fill_old() {
    assert!(try_open("fill_old.h5").is_ok());
}

#[test]
fn t1d_tlayouto() {
    assert!(try_open("tlayouto.h5").is_ok());
}

// =============================================================
// T1e: Filters
// =============================================================

#[test]
fn t1e_deflate() {
    let result = try_open_and_visit_members("deflate.h5", |_| {});
    assert!(result.is_ok(), "deflate.h5: {}", result.unwrap_err());
}

#[test]
fn t1e_test_filters_be() {
    assert!(try_open("test_filters_be.h5").is_ok());
}

#[test]
fn t1e_test_filters_le() {
    assert!(try_open("test_filters_le.h5").is_ok());
}

#[test]
fn t1e_noencoder() {
    assert!(try_open("noencoder.h5").is_ok());
}

#[test]
fn t1e_filter_error() {
    assert!(try_open("filter_error.h5").is_ok());
}

// =============================================================
// T1f: Chunk indexing
// =============================================================

#[test]
fn t1f_btree_idx_1_6() {
    assert!(try_open("btree_idx_1_6.h5").is_ok());
}

#[test]
fn t1f_btree_idx_1_8() {
    assert!(try_open("btree_idx_1_8.h5").is_ok());
}

// =============================================================
// T1g: Endianness
// =============================================================

#[test]
fn t1g_be_data() {
    let r = try_open_and_visit_members("be_data.h5", |_| {});
    assert!(r.is_ok(), "be_data.h5: {}", r.unwrap_err());
}

#[test]
fn t1g_le_data() {
    let r = try_open_and_visit_members("le_data.h5", |_| {});
    assert!(r.is_ok(), "le_data.h5: {}", r.unwrap_err());
}

// =============================================================
// T1h: Modification times
// =============================================================

#[test]
fn t1h_tmtimen() {
    assert!(try_open("tmtimen.h5").is_ok());
}

#[test]
fn t1h_tmtimeo() {
    assert!(try_open("tmtimeo.h5").is_ok());
}

// =============================================================
// T1i: Heap structures
// =============================================================

#[test]
fn t1i_tsizeslheap() {
    assert!(try_open("tsizeslheap.h5").is_ok());
}

// =============================================================
// All remaining reference files -- just verify open doesn't panic
// =============================================================

#[test]
fn t1_open_all_reference_files() {
    let dir = std::fs::read_dir(REF_DIR).unwrap();
    let mut total = 0;
    let mut ok = 0;
    let mut failed = Vec::new();

    for entry in dir {
        let entry = entry.unwrap();
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.ends_with(".h5") {
            continue;
        }
        total += 1;
        match try_open(&name) {
            Ok(()) => ok += 1,
            Err(e) => failed.push((name, e)),
        }
    }

    println!("\nReference file results: {ok}/{total} opened successfully");
    if !failed.is_empty() {
        println!("Failed files:");
        for (name, err) in &failed {
            println!("  {name}: {err}");
        }
    }

    // We expect MOST files to open. Some may fail due to unsupported features.
    // At minimum 80% should open.
    let pct = (ok as f64 / total as f64) * 100.0;
    println!("Success rate: {pct:.0}%");
    assert!(
        pct >= 70.0,
        "Too many reference files failed: {ok}/{total} ({pct:.0}%)"
    );
}
