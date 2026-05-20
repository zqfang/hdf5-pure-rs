use std::fs::File;
use std::io::BufReader;
use std::io::Write;

use hdf5_pure_rust::format::superblock::Superblock;
use hdf5_pure_rust::io::HdfReader;
use hdf5_pure_rust::File as Hdf5File;

const V0_BASE_ADDR_OFFSET: usize = 24;

fn userblock_v0_fixture() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("userblock_v0.h5");
    let mut original = std::fs::read("tests/data/simple_v0.h5").unwrap();
    original[V0_BASE_ADDR_OFFSET..V0_BASE_ADDR_OFFSET + 8].copy_from_slice(&512u64.to_le_bytes());

    let mut with_userblock = vec![0u8; 512];
    with_userblock[..b"custom header\0".len()].copy_from_slice(b"custom header\0");
    with_userblock.extend_from_slice(&original);

    let mut file = File::create(&path).unwrap();
    file.write_all(&with_userblock).unwrap();
    dir
}

#[test]
fn test_parse_v0_superblock() {
    let f = File::open("tests/data/simple_v0.h5").expect("failed to open test file");
    let mut reader = HdfReader::new(BufReader::new(f));
    let sb = Superblock::read(&mut reader).expect("failed to parse superblock");

    assert_eq!(sb.version, 0);
    assert_eq!(sb.sizeof_addr, 8);
    assert_eq!(sb.sizeof_size, 8);
    assert_eq!(sb.sym_leaf_k, 4);
    assert_eq!(sb.snode_btree_k, 16);
    assert_eq!(sb.base_addr, 0);
    assert_eq!(sb.status_flags, 0);
    // Root group object header address should be valid
    assert_ne!(sb.root_addr, u64::MAX);
    println!("v0 superblock: {sb:#?}");
}

#[test]
fn test_parse_v3_superblock() {
    let f = File::open("tests/data/simple_v2.h5").expect("failed to open test file");
    let mut reader = HdfReader::new(BufReader::new(f));
    let sb = Superblock::read(&mut reader).expect("failed to parse superblock");

    assert_eq!(sb.version, 3);
    assert_eq!(sb.sizeof_addr, 8);
    assert_eq!(sb.sizeof_size, 8);
    assert_eq!(sb.base_addr, 0);
    assert_eq!(sb.status_flags, 0);
    // Root group object header address should be valid
    assert_ne!(sb.root_addr, u64::MAX);
    println!("v3 superblock: {sb:#?}");
}

#[test]
fn test_parse_v0_superblock_after_userblock() {
    let dir = userblock_v0_fixture();
    let path = dir.path().join("userblock_v0.h5");
    let f = File::open(&path).expect("failed to open test file");
    let mut reader = HdfReader::new(BufReader::new(f));
    let sb = Superblock::read(&mut reader).expect("failed to parse superblock");

    assert_eq!(sb.version, 0);
    assert_eq!(sb.base_addr, 512);
    assert_eq!(reader.base_addr(), 512);
    assert_eq!(
        reader.position_physical().unwrap(),
        reader.position().unwrap() + 512
    );
    assert_ne!(sb.root_addr, u64::MAX);
}

#[test]
fn test_open_v0_file_after_userblock_reads_root_members() {
    let dir = userblock_v0_fixture();
    let path = dir.path().join("userblock_v0.h5");
    let f = Hdf5File::open(&path).expect("failed to open userblock fixture");

    assert_eq!(f.userblock(), 512);
    assert!(f.member_names().expect("failed to list members").len() > 0);
    assert_eq!(
        f.file_image().unwrap().len() as u64,
        std::fs::metadata(path).unwrap().len()
    );
}
