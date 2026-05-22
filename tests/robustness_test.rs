use hdf5_pure_rust::format::btree_v2::BTreeV2Header;
use hdf5_pure_rust::format::checksum::checksum_metadata;
use hdf5_pure_rust::format::fractal_heap::FractalHeapHeader;
use hdf5_pure_rust::format::global_heap::GlobalHeapCollection;
use hdf5_pure_rust::format::local_heap::LocalHeap;
use hdf5_pure_rust::format::messages::attribute::AttributeMessage;
use hdf5_pure_rust::format::messages::attribute_info::AttributeInfoMessage;
use hdf5_pure_rust::format::messages::data_layout::DataLayoutMessage;
use hdf5_pure_rust::format::messages::dataspace::DataspaceMessage;
use hdf5_pure_rust::format::messages::datatype::{ByteOrder, DatatypeClass, DatatypeMessage};
use hdf5_pure_rust::format::messages::fill_value::FillValueMessage;
use hdf5_pure_rust::format::messages::filter_pipeline::{
    FilterDesc, FilterPipelineMessage, FILTER_DEFLATE, FILTER_FLETCHER32, FILTER_NBIT,
    FILTER_SCALEOFFSET, FILTER_SZIP,
};
use hdf5_pure_rust::format::messages::link::LinkMessage;
use hdf5_pure_rust::format::messages::link_info::LinkInfoMessage;
use hdf5_pure_rust::format::messages::symbol_table::SymbolTableMessage;
use hdf5_pure_rust::format::object_header::{
    ObjectHeader, HDR_ATTR_CRT_ORDER_INDEXED, HDR_ATTR_STORE_PHASE_CHANGE, MSG_ATTRIBUTE,
    MSG_ATTR_INFO, MSG_BTREE_K, MSG_DATASPACE, MSG_DATATYPE, MSG_DRIVER_INFO,
    MSG_EXTERNAL_FILE_LIST, MSG_FILE_SPACE_INFO, MSG_FILL_VALUE, MSG_FILL_VALUE_OLD,
    MSG_FILTER_PIPELINE, MSG_GROUP_INFO, MSG_HEADER_CONTINUATION, MSG_LAYOUT, MSG_LINK,
    MSG_LINK_INFO, MSG_MDCI, MSG_OBJ_REF_COUNT, MSG_SHARED_MSG_TABLE, MSG_SYMBOL_TABLE,
};
use hdf5_pure_rust::format::superblock::Superblock;
use hdf5_pure_rust::format::symbol_table::SymbolTableNode;
use hdf5_pure_rust::format::{extensible_array, fixed_array};
use hdf5_pure_rust::io::reader::UNDEF_ADDR;
use hdf5_pure_rust::io::HdfReader;

use std::io::Cursor;

#[test]
fn test_invalid_signature() {
    let data = vec![0u8; 64];
    let mut reader = HdfReader::new(Cursor::new(data));
    assert!(Superblock::read(&mut reader).is_err());
}

#[test]
fn test_truncated_superblock() {
    // Valid signature but truncated
    let mut data = vec![0x89, 0x48, 0x44, 0x46, 0x0D, 0x0A, 0x1A, 0x0A];
    data.push(2); // version 2
                  // Missing sizeof_addr, sizeof_size, etc.
    let mut reader = HdfReader::new(Cursor::new(data));
    assert!(Superblock::read(&mut reader).is_err());
}

#[test]
fn test_superblock_rejects_invalid_root_symbol_cache_type() {
    let mut data = Vec::new();
    data.extend_from_slice(&hdf5_pure_rust::format::superblock::HDF5_SIGNATURE);
    data.push(0); // superblock version
    data.push(0); // free-space version
    data.push(0); // root group version
    data.push(0); // reserved
    data.push(0); // shared header version
    data.push(8); // sizeof_addr
    data.push(8); // sizeof_size
    data.push(0); // reserved
    data.extend_from_slice(&4u16.to_le_bytes()); // sym_leaf_k
    data.extend_from_slice(&16u16.to_le_bytes()); // snode_btree_k
    data.extend_from_slice(&0u32.to_le_bytes()); // status flags
    data.extend_from_slice(&0u64.to_le_bytes()); // base addr
    data.extend_from_slice(&UNDEF_ADDR.to_le_bytes()); // ext addr
    data.extend_from_slice(&128u64.to_le_bytes()); // eof addr
    data.extend_from_slice(&UNDEF_ADDR.to_le_bytes()); // driver addr
    data.extend_from_slice(&0u64.to_le_bytes()); // root name offset
    data.extend_from_slice(&64u64.to_le_bytes()); // root object header addr
    data.extend_from_slice(&3u32.to_le_bytes()); // invalid root cache type
    data.extend_from_slice(&0u32.to_le_bytes()); // reserved
    data.extend_from_slice(&[0u8; 16]); // scratch pad

    let mut reader = HdfReader::new(Cursor::new(data));
    let err = Superblock::read(&mut reader)
        .expect_err("invalid root symbol-table cache type should fail");
    assert!(
        format!("{err}").contains("cache type"),
        "expected cache-type error, got: {err}"
    );
}

#[test]
fn test_superblock_v0_v1_rejects_nonzero_reserved_bytes() {
    fn v0_superblock(first_reserved: u8, second_reserved: u8, root_reserved: u32) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(&hdf5_pure_rust::format::superblock::HDF5_SIGNATURE);
        data.push(0); // superblock version
        data.push(0); // free-space version
        data.push(0); // root group version
        data.push(first_reserved);
        data.push(0); // shared header version
        data.push(8); // sizeof_addr
        data.push(8); // sizeof_size
        data.push(second_reserved);
        data.extend_from_slice(&4u16.to_le_bytes()); // sym_leaf_k
        data.extend_from_slice(&16u16.to_le_bytes()); // snode_btree_k
        data.extend_from_slice(&0u32.to_le_bytes()); // status flags
        data.extend_from_slice(&0u64.to_le_bytes()); // base addr
        data.extend_from_slice(&UNDEF_ADDR.to_le_bytes()); // ext addr
        data.extend_from_slice(&128u64.to_le_bytes()); // eof addr
        data.extend_from_slice(&UNDEF_ADDR.to_le_bytes()); // driver addr
        data.extend_from_slice(&0u64.to_le_bytes()); // root name offset
        data.extend_from_slice(&64u64.to_le_bytes()); // root object header addr
        data.extend_from_slice(&0u32.to_le_bytes()); // root cache type
        data.extend_from_slice(&root_reserved.to_le_bytes());
        data.extend_from_slice(&[0u8; 16]); // scratch pad
        data
    }

    fn v1_superblock(chunk_reserved: u16) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(&hdf5_pure_rust::format::superblock::HDF5_SIGNATURE);
        data.push(1); // superblock version
        data.push(0); // free-space version
        data.push(0); // root group version
        data.push(0); // reserved
        data.push(0); // shared header version
        data.push(8); // sizeof_addr
        data.push(8); // sizeof_size
        data.push(0); // reserved
        data.extend_from_slice(&4u16.to_le_bytes()); // sym_leaf_k
        data.extend_from_slice(&16u16.to_le_bytes()); // snode_btree_k
        data.extend_from_slice(&0u32.to_le_bytes()); // status flags
        data.extend_from_slice(&32u16.to_le_bytes()); // chunk_btree_k
        data.extend_from_slice(&chunk_reserved.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes()); // base addr
        data.extend_from_slice(&UNDEF_ADDR.to_le_bytes()); // ext addr
        data.extend_from_slice(&128u64.to_le_bytes()); // eof addr
        data.extend_from_slice(&UNDEF_ADDR.to_le_bytes()); // driver addr
        data.extend_from_slice(&0u64.to_le_bytes()); // root name offset
        data.extend_from_slice(&64u64.to_le_bytes()); // root object header addr
        data.extend_from_slice(&0u32.to_le_bytes()); // root cache type
        data.extend_from_slice(&0u32.to_le_bytes()); // reserved
        data.extend_from_slice(&[0u8; 16]); // scratch pad
        data
    }

    for data in [
        v0_superblock(1, 0, 0),
        v0_superblock(0, 1, 0),
        v0_superblock(0, 0, 1),
        v1_superblock(1),
    ] {
        let mut reader = HdfReader::new(Cursor::new(data));
        let err = Superblock::read(&mut reader)
            .expect_err("nonzero superblock reserved bytes should fail");
        assert!(
            format!("{err}").contains("reserved"),
            "expected reserved-byte error, got: {err}"
        );
    }
}

#[test]
fn test_object_header_rejects_truncated_v1_message_containers() {
    fn v1_header(chunk: &[u8]) -> Vec<u8> {
        let mut data = vec![1, 0];
        data.extend_from_slice(&1u16.to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&(chunk.len() as u32).to_le_bytes());
        data.extend_from_slice(&[0; 4]);
        data.extend_from_slice(chunk);
        data
    }

    let mut payload_exceeds_chunk = Vec::new();
    payload_exceeds_chunk.extend_from_slice(&MSG_DATATYPE.to_le_bytes());
    payload_exceeds_chunk.extend_from_slice(&8u16.to_le_bytes());
    payload_exceeds_chunk.push(0);
    payload_exceeds_chunk.extend_from_slice(&[0; 3]);

    let mut truncated_continuation = Vec::new();
    truncated_continuation.extend_from_slice(&MSG_HEADER_CONTINUATION.to_le_bytes());
    truncated_continuation.extend_from_slice(&4u16.to_le_bytes());
    truncated_continuation.push(0);
    truncated_continuation.extend_from_slice(&[0; 3]);

    for data in [
        v1_header(&[1; 7]),
        v1_header(&payload_exceeds_chunk),
        v1_header(&truncated_continuation),
    ] {
        let mut reader = HdfReader::new(Cursor::new(data));
        let err = ObjectHeader::read_at(&mut reader, 0)
            .expect_err("truncated object-header message container should fail");
        assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
    }
}

#[test]
fn test_object_header_rejects_invalid_v1_continuation_ranges() {
    fn v1_header(chunk: &[u8]) -> Vec<u8> {
        let mut data = vec![1, 0];
        data.extend_from_slice(&1u16.to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&(chunk.len() as u32).to_le_bytes());
        data.extend_from_slice(&[0; 4]);
        data.extend_from_slice(chunk);
        data
    }

    fn continuation_message(addr: u64, length: u64) -> Vec<u8> {
        let mut chunk = Vec::new();
        chunk.extend_from_slice(&MSG_HEADER_CONTINUATION.to_le_bytes());
        chunk.extend_from_slice(&16u16.to_le_bytes());
        chunk.push(0);
        chunk.extend_from_slice(&[0; 3]);
        chunk.extend_from_slice(&addr.to_le_bytes());
        chunk.extend_from_slice(&length.to_le_bytes());
        chunk
    }

    for data in [
        v1_header(&continuation_message(u64::MAX - 4, 16)),
        v1_header(&continuation_message(64, 16)),
        v1_header(&continuation_message(16, 16)),
        v1_header(&continuation_message(64, 0)),
    ] {
        let mut reader = HdfReader::new(Cursor::new(data));
        let err = ObjectHeader::read_at(&mut reader, 0)
            .expect_err("invalid object-header continuation range should fail");
        assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
    }
}

#[test]
fn test_object_header_rejects_malformed_shared_message_payloads() {
    fn v1_header(msg_type: u16, flags: u8, payload: &[u8]) -> Vec<u8> {
        let aligned_size = (payload.len() + 7) & !7;
        let mut chunk = Vec::new();
        chunk.extend_from_slice(&msg_type.to_le_bytes());
        chunk.extend_from_slice(&(payload.len() as u16).to_le_bytes());
        chunk.push(flags);
        chunk.extend_from_slice(&[0; 3]);
        chunk.extend_from_slice(payload);
        chunk.resize(8 + aligned_size, 0);

        let mut data = vec![1, 0];
        data.extend_from_slice(&1u16.to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&(chunk.len() as u32).to_le_bytes());
        data.extend_from_slice(&[0; 4]);
        data.extend_from_slice(&chunk);
        data
    }

    let shared_flag = 0x02;
    let mut valid_table = vec![0];
    valid_table.extend_from_slice(&64u64.to_le_bytes());
    valid_table.push(1);
    let mut trailing_table = valid_table.clone();
    trailing_table.push(0xaa);
    let mut shared_v3_sohm = vec![3, 1];
    shared_v3_sohm.extend_from_slice(&[0x5a; 8]);
    let mut trailing_shared_v3_sohm = shared_v3_sohm.clone();
    trailing_shared_v3_sohm.push(0xaa);

    for data in [
        v1_header(MSG_SHARED_MSG_TABLE, 0, &[]),
        v1_header(MSG_SHARED_MSG_TABLE, 0, &[1, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
        v1_header(
            MSG_SHARED_MSG_TABLE,
            0,
            &[0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 1],
        ),
        v1_header(MSG_SHARED_MSG_TABLE, 0, &[0, 64, 0, 0, 0, 0, 0, 0, 0, 0]),
        v1_header(MSG_DATATYPE, shared_flag, &[4, 2, 64, 0, 0, 0, 0, 0, 0, 0]),
        v1_header(MSG_DATATYPE, shared_flag, &[3, 3, 64, 0, 0, 0, 0, 0, 0, 0]),
        v1_header(MSG_DATATYPE, shared_flag, &[3, 2, 64, 0, 0]),
        v1_header(
            MSG_DATATYPE,
            shared_flag,
            &[2, 2, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
        ),
    ] {
        let mut reader = HdfReader::new(Cursor::new(data));
        let err = ObjectHeader::read_at(&mut reader, 0)
            .expect_err("malformed shared-message payload should fail");
        assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
    }

    for data in [
        v1_header(MSG_SHARED_MSG_TABLE, 0, &valid_table),
        v1_header(MSG_SHARED_MSG_TABLE, 0, &trailing_table),
        v1_header(MSG_DATATYPE, shared_flag, &shared_v3_sohm),
        v1_header(MSG_DATATYPE, shared_flag, &trailing_shared_v3_sohm),
    ] {
        let mut reader = HdfReader::new(Cursor::new(data));
        ObjectHeader::read_at(&mut reader, 0)
            .expect("H5O__shmesg_decode/H5O__shared_decode tolerate trailing payload bytes");
    }
}

#[test]
fn test_object_header_v2_rejects_reserved_header_flags() {
    let mut data = b"OHDR".to_vec();
    data.push(2);
    data.push(0x40);
    data.push(0);
    let checksum = checksum_metadata(&data);
    data.extend_from_slice(&checksum.to_le_bytes());

    let mut reader = HdfReader::new(Cursor::new(data));
    let err = ObjectHeader::read_at(&mut reader, 0)
        .expect_err("reserved v2 object-header flags should fail");
    assert!(
        format!("{err}").contains("reserved bits"),
        "expected reserved-bits error, got: {err}"
    );
}

#[test]
fn test_object_header_v2_accepts_indexed_creation_order_without_tracking_like_libhdf5() {
    let mut data = b"OHDR".to_vec();
    data.push(2);
    data.push(HDR_ATTR_CRT_ORDER_INDEXED);
    data.push(0);
    let checksum = checksum_metadata(&data);
    data.extend_from_slice(&checksum.to_le_bytes());

    let mut reader = HdfReader::new(Cursor::new(data));
    ObjectHeader::read_at(&mut reader, 0)
        .expect("H5O__prefix_deserialize accepts indexed creation order without tracking");
}

#[test]
fn test_object_header_v2_rejects_invalid_attribute_phase_change() {
    let mut data = b"OHDR".to_vec();
    data.push(2);
    data.push(HDR_ATTR_STORE_PHASE_CHANGE);
    data.extend_from_slice(&2u16.to_le_bytes()); // max compact attrs
    data.extend_from_slice(&3u16.to_le_bytes()); // min dense attrs
    data.push(0); // chunk 0 size, one byte because size-width flags are 0
    let checksum = checksum_metadata(&data);
    data.extend_from_slice(&checksum.to_le_bytes());

    let mut reader = HdfReader::new(Cursor::new(data));
    let err = ObjectHeader::read_at(&mut reader, 0)
        .expect_err("invalid attribute phase-change values should fail");
    assert!(
        format!("{err}").contains("phase change"),
        "expected phase-change error, got: {err}"
    );
}

#[test]
fn test_object_header_v2_decodes_refcount_message_like_libhdf5() {
    let mut data = b"OHDR".to_vec();
    data.push(2);
    data.push(0);
    data.push(9); // chunk 0 size
    data.push(MSG_OBJ_REF_COUNT as u8);
    data.extend_from_slice(&5u16.to_le_bytes());
    data.push(0);
    data.extend_from_slice(&7u32.to_le_bytes());
    data.push(0xaa); // trailing byte tolerated by H5O__refcount_decode
    let checksum = checksum_metadata(&data);
    data.extend_from_slice(&checksum.to_le_bytes());

    let mut reader = HdfReader::new(Cursor::new(data));
    let oh = ObjectHeader::read_at(&mut reader, 0)
        .expect("v2 object-header refcount message should decode");
    assert_eq!(oh.refcount, 7);
}

#[test]
fn test_object_header_v2_rejects_truncated_refcount_message() {
    let mut data = b"OHDR".to_vec();
    data.push(2);
    data.push(0);
    data.push(7); // chunk 0 size
    data.push(MSG_OBJ_REF_COUNT as u8);
    data.extend_from_slice(&3u16.to_le_bytes());
    data.push(0);
    data.extend_from_slice(&[1, 2, 3]);
    let checksum = checksum_metadata(&data);
    data.extend_from_slice(&checksum.to_le_bytes());

    let mut reader = HdfReader::new(Cursor::new(data));
    let err = ObjectHeader::read_at(&mut reader, 0)
        .expect_err("truncated object-header refcount message should fail");
    assert!(
        format!("{err}").contains("refcount"),
        "expected refcount truncation error, got: {err}"
    );
}

#[test]
fn test_object_header_validates_group_info_and_btree_k_messages_like_libhdf5() {
    fn v2_header(msg_type: u16, payload: &[u8]) -> Vec<u8> {
        let mut data = b"OHDR".to_vec();
        data.push(2);
        data.push(0);
        data.push((4 + payload.len()) as u8);
        data.push(msg_type as u8);
        data.extend_from_slice(&(payload.len() as u16).to_le_bytes());
        data.push(0);
        data.extend_from_slice(payload);
        let checksum = checksum_metadata(&data);
        data.extend_from_slice(&checksum.to_le_bytes());
        data
    }

    for (msg_type, payload, expected) in [
        (MSG_GROUP_INFO, vec![0, 0x80], "group info"),
        (MSG_GROUP_INFO, vec![0, 0x01, 8, 0, 6], "group info"),
        (MSG_BTREE_K, vec![0, 32, 0, 16, 0, 4], "B-tree K"),
        (MSG_BTREE_K, vec![1, 32, 0, 16, 0, 4, 0], "B-tree K"),
    ] {
        let mut reader = HdfReader::new(Cursor::new(v2_header(msg_type, &payload)));
        let err = ObjectHeader::read_at(&mut reader, 0)
            .expect_err("malformed object-header message should fail");
        assert!(
            format!("{err}").contains(expected),
            "expected {expected} error, got: {err}"
        );
    }

    for (msg_type, payload) in [
        (MSG_GROUP_INFO, vec![0, 0x03, 8, 0, 6, 0, 5, 0, 12, 0, 0xaa]),
        (MSG_BTREE_K, vec![0, 32, 0, 16, 0, 4, 0, 0xaa]),
    ] {
        let mut reader = HdfReader::new(Cursor::new(v2_header(msg_type, &payload)));
        ObjectHeader::read_at(&mut reader, 0)
            .expect("known group-info/B-tree-K fields should decode with trailing bytes");
    }
}

#[test]
fn test_object_header_validates_fill_value_and_filter_pipeline_messages_like_libhdf5() {
    fn v2_header(msg_type: u16, payload: &[u8]) -> Vec<u8> {
        let mut data = b"OHDR".to_vec();
        data.push(2);
        data.push(0);
        data.push((4 + payload.len()) as u8);
        data.push(msg_type as u8);
        data.extend_from_slice(&(payload.len() as u16).to_le_bytes());
        data.push(0);
        data.extend_from_slice(payload);
        let checksum = checksum_metadata(&data);
        data.extend_from_slice(&checksum.to_le_bytes());
        data
    }

    for (msg_type, payload, expected) in [
        (MSG_FILL_VALUE, vec![9], "fill value"),
        (MSG_FILL_VALUE, vec![3, 0x40], "fill value"),
        (MSG_FILL_VALUE_OLD, vec![4, 0, 0, 0, 1, 2], "old fill value"),
        (MSG_FILTER_PIPELINE, vec![9, 0], "filter pipeline"),
        (MSG_FILTER_PIPELINE, vec![2, 1], "filter pipeline"),
    ] {
        let mut reader = HdfReader::new(Cursor::new(v2_header(msg_type, &payload)));
        let err = ObjectHeader::read_at(&mut reader, 0)
            .expect_err("malformed object-header message should fail");
        assert!(
            format!("{err}").contains(expected),
            "expected {expected} error, got: {err}"
        );
    }

    for (msg_type, payload) in [
        (MSG_FILL_VALUE, vec![3, 0, 0xaa]),
        (MSG_FILL_VALUE_OLD, vec![0, 0, 0, 0, 0xaa]),
        (MSG_FILTER_PIPELINE, vec![2, 0, 0xaa]),
    ] {
        let mut reader = HdfReader::new(Cursor::new(v2_header(msg_type, &payload)));
        ObjectHeader::read_at(&mut reader, 0)
            .expect("known fill-value/filter-pipeline fields should decode with trailing bytes");
    }
}

#[test]
fn test_object_header_validates_datatype_and_attribute_messages_like_libhdf5() {
    fn v2_header(msg_type: u16, payload: &[u8]) -> Vec<u8> {
        let mut data = b"OHDR".to_vec();
        data.push(2);
        data.push(0);
        data.push((4 + payload.len()) as u8);
        data.push(msg_type as u8);
        data.extend_from_slice(&(payload.len() as u16).to_le_bytes());
        data.push(0);
        data.extend_from_slice(payload);
        let checksum = checksum_metadata(&data);
        data.extend_from_slice(&checksum.to_le_bytes());
        data
    }

    for (msg_type, payload, expected) in [
        (MSG_DATATYPE, vec![0x10, 0, 0, 0, 0, 0, 0, 0], "datatype"),
        (MSG_DATATYPE, vec![0x1f, 0, 0, 0, 1, 0, 0, 0], "datatype"),
        (MSG_ATTRIBUTE, vec![4, 0, 0, 0, 0, 0], "attribute"),
        (MSG_ATTRIBUTE, vec![2, 0x80, 2, 0, 8, 0, 4, 0], "attribute"),
    ] {
        let mut reader = HdfReader::new(Cursor::new(v2_header(msg_type, &payload)));
        let err = ObjectHeader::read_at(&mut reader, 0)
            .expect_err("malformed object-header message should fail");
        assert!(
            format!("{err}").contains(expected),
            "expected {expected} error, got: {err}"
        );
    }

    let dtype = vec![0x10, 0, 0, 0, 4, 0, 0, 0, 0, 0, 32, 0, 0xaa];
    let mut reader = HdfReader::new(Cursor::new(v2_header(MSG_DATATYPE, &dtype)));
    ObjectHeader::read_at(&mut reader, 0)
        .expect("datatype messages should decode while preserving trailing properties");

    let attr_dtype = vec![0x10, 0, 0, 0, 4, 0, 0, 0, 0, 0, 32, 0];
    let attr_dataspace = vec![2, 0, 0, 0];
    let mut attr = vec![3, 0];
    attr.extend_from_slice(&2u16.to_le_bytes());
    attr.extend_from_slice(&(attr_dtype.len() as u16).to_le_bytes());
    attr.extend_from_slice(&(attr_dataspace.len() as u16).to_le_bytes());
    attr.push(0);
    attr.extend_from_slice(b"x\0");
    attr.extend_from_slice(&attr_dtype);
    attr.extend_from_slice(&attr_dataspace);
    attr.extend_from_slice(&[1, 2, 3, 4]);
    let mut reader = HdfReader::new(Cursor::new(v2_header(MSG_ATTRIBUTE, &attr)));
    ObjectHeader::read_at(&mut reader, 0)
        .expect("valid compact attribute message should pass header validation");
}

#[test]
fn test_object_header_validates_core_message_decoders_like_libhdf5() {
    fn v2_header(msg_type: u16, payload: &[u8]) -> Vec<u8> {
        let mut data = b"OHDR".to_vec();
        data.push(2);
        data.push(0);
        data.push((4 + payload.len()) as u8);
        data.push(msg_type as u8);
        data.extend_from_slice(&(payload.len() as u16).to_le_bytes());
        data.push(0);
        data.extend_from_slice(payload);
        let checksum = checksum_metadata(&data);
        data.extend_from_slice(&checksum.to_le_bytes());
        data
    }

    for (msg_type, payload, expected) in [
        (MSG_DATASPACE, vec![9, 0, 0, 0], "dataspace"),
        (MSG_LINK_INFO, vec![9, 0], "link info"),
        (MSG_LINK, vec![9, 0], "link message"),
        (MSG_LAYOUT, Vec::new(), "data layout"),
        (
            MSG_EXTERNAL_FILE_LIST,
            vec![1, 0, 0, 0, 0, 0, 0, 0],
            "external file list",
        ),
        (MSG_SYMBOL_TABLE, Vec::new(), "symbol table"),
        (MSG_ATTR_INFO, vec![9, 0], "attribute info"),
        (MSG_DRIVER_INFO, vec![0, b's', b'e', b'c'], "driver info"),
        (MSG_MDCI, vec![9, 0, 0, 0], "metadata cache image"),
        (MSG_FILE_SPACE_INFO, vec![1, 0, 0, 0], "file-space info"),
    ] {
        let mut reader = HdfReader::new(Cursor::new(v2_header(msg_type, &payload)));
        let err = ObjectHeader::read_at(&mut reader, 0)
            .expect_err("malformed object-header message should fail");
        assert!(
            format!("{err}").contains(expected),
            "expected {expected} error, got: {err}"
        );
    }

    for (msg_type, payload) in [
        (MSG_DATASPACE, vec![2, 0, 0, 0]),
        (MSG_LAYOUT, vec![3, 0, 0, 0, 0xaa]),
        (MSG_DRIVER_INFO, {
            let mut bytes = vec![0, b's', b'e', b'c', b'2', 0, 0, 0, 0];
            bytes.extend_from_slice(&3u16.to_le_bytes());
            bytes.extend_from_slice(b"abc");
            bytes.push(0xaa);
            bytes
        }),
        (
            MSG_MDCI,
            vec![0, 16, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
        ),
        (MSG_FILE_SPACE_INFO, {
            let mut bytes = vec![1, 0, 0];
            bytes.extend_from_slice(&1u64.to_le_bytes());
            bytes.extend_from_slice(&4096u64.to_le_bytes());
            bytes.extend_from_slice(&0u16.to_le_bytes());
            bytes.extend_from_slice(&u64::MAX.to_le_bytes());
            bytes.push(0xaa);
            bytes
        }),
        (MSG_EXTERNAL_FILE_LIST, {
            let mut bytes = vec![1, 1, 0, 0];
            bytes.extend_from_slice(&1u16.to_le_bytes());
            bytes.extend_from_slice(&0u16.to_le_bytes());
            bytes.extend_from_slice(&0u64.to_le_bytes());
            bytes.push(0xaa);
            bytes
        }),
    ] {
        let mut reader = HdfReader::new(Cursor::new(v2_header(msg_type, &payload)));
        ObjectHeader::read_at(&mut reader, 0)
            .expect("known core message fields should decode with trailing bytes");
    }
}

#[test]
fn test_object_header_v1_skips_reserved_bytes_like_libhdf5() {
    fn v1_header(reserved_byte: u8, prefix_padding: [u8; 4], message_reserved: [u8; 3]) -> Vec<u8> {
        let payload = [0x10, 0, 0, 0, 4, 0, 0, 0, 0, 0, 32, 0];
        let aligned_size = (payload.len() + 7) & !7;
        let mut chunk = Vec::new();
        chunk.extend_from_slice(&MSG_DATATYPE.to_le_bytes());
        chunk.extend_from_slice(&(payload.len() as u16).to_le_bytes());
        chunk.push(0);
        chunk.extend_from_slice(&message_reserved);
        chunk.extend_from_slice(&payload);
        chunk.resize(8 + aligned_size, 0);

        let mut data = vec![1, reserved_byte];
        data.extend_from_slice(&1u16.to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&(chunk.len() as u32).to_le_bytes());
        data.extend_from_slice(&prefix_padding);
        data.extend_from_slice(&chunk);
        data
    }

    for data in [
        v1_header(1, [0; 4], [0; 3]),
        v1_header(0, [0, 1, 0, 0], [0; 3]),
        v1_header(0, [0; 4], [0, 1, 0]),
    ] {
        let mut reader = HdfReader::new(Cursor::new(data));
        ObjectHeader::read_at(&mut reader, 0)
            .expect("H5O__prefix_deserialize/H5O__chunk_deserialize skip v1 reserved bytes");
    }
}

#[test]
fn test_object_header_truncated_synthetic_files_do_not_panic() {
    fn v1_header_with_tail(chunk: &[u8], tail: &[u8]) -> Vec<u8> {
        let mut data = vec![1, 0];
        data.extend_from_slice(&1u16.to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&(chunk.len() as u32).to_le_bytes());
        data.extend_from_slice(&[0; 4]);
        data.extend_from_slice(chunk);
        data.extend_from_slice(tail);
        data
    }

    fn v1_message(msg_type: u16, flags: u8, payload: &[u8]) -> Vec<u8> {
        let aligned_size = (payload.len() + 7) & !7;
        let mut chunk = Vec::new();
        chunk.extend_from_slice(&msg_type.to_le_bytes());
        chunk.extend_from_slice(&(payload.len() as u16).to_le_bytes());
        chunk.push(flags);
        chunk.extend_from_slice(&[0; 3]);
        chunk.extend_from_slice(payload);
        chunk.resize(8 + aligned_size, 0);
        chunk
    }

    let datatype_payload = [0x10, 0, 0, 0, 4, 0, 0, 0, 0, 0, 32, 0];
    let normal_message_file =
        v1_header_with_tail(&v1_message(MSG_DATATYPE, 0, &datatype_payload), &[]);

    let mut shared_table_payload = vec![0];
    shared_table_payload.extend_from_slice(&64u64.to_le_bytes());
    shared_table_payload.push(1);
    let shared_table_file = v1_header_with_tail(
        &v1_message(MSG_SHARED_MSG_TABLE, 0, &shared_table_payload),
        &[],
    );

    let mut shared_ref_payload = vec![3, 1];
    shared_ref_payload.extend_from_slice(&[0x5a; 8]);
    let shared_ref_file =
        v1_header_with_tail(&v1_message(MSG_DATATYPE, 0x02, &shared_ref_payload), &[]);

    let continuation_chunk = {
        let mut payload = Vec::new();
        payload.extend_from_slice(&40u64.to_le_bytes());
        payload.extend_from_slice(&8u64.to_le_bytes());
        v1_message(MSG_HEADER_CONTINUATION, 0, &payload)
    };
    let continuation_file = v1_header_with_tail(&continuation_chunk, &[0; 8]);

    for (name, data) in [
        ("normal", normal_message_file),
        ("shared table", shared_table_file),
        ("shared reference", shared_ref_file),
        ("continuation", continuation_file),
    ] {
        for len in 0..=data.len() {
            let prefix = data[..len].to_vec();
            let result = std::panic::catch_unwind(|| {
                let mut reader = HdfReader::new(Cursor::new(prefix));
                let _ = ObjectHeader::read_at(&mut reader, 0);
            });
            assert!(
                result.is_ok(),
                "{name} object-header prefix length {len} panicked"
            );
        }
    }
}

#[test]
fn test_link_message_empty() {
    assert!(LinkMessage::decode(&[], 8).is_err());
}

#[test]
fn test_link_message_truncated() {
    // Valid version + flags but truncated
    let data = vec![1, 0];
    assert!(LinkMessage::decode(&data, 8).is_err());
}

#[test]
fn test_link_message_rejects_truncated_optional_fields() {
    for data in [
        vec![1],
        vec![1, 0x08],
        vec![1, 0x04, 0, 0, 0],
        vec![1, 0x10],
        vec![1, 0x03, 1, 2],
        vec![1, 0x08, 1, 1, b's', 1],
        vec![1, 0x08, 64, 1, b'e', 3, 0, 0],
    ] {
        let err = LinkMessage::decode(&data, 8).expect_err("truncated link message should fail");
        assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
    }
}

#[test]
fn test_link_message_bad_name_length() {
    // Version 1, flags=0 (hard link, 1-byte name len), name_len=255 but only 4 bytes of data
    let data = vec![1, 0x00, 0xFF, 0x41]; // version=1, flags=0, name_len=255, 'A'
    assert!(LinkMessage::decode(&data, 8).is_err());
}

#[test]
fn test_link_message_rejects_bad_flags_and_zero_lengths() {
    let err = LinkMessage::decode(&[1, 0x80], 8).unwrap_err();
    assert!(format!("{err}").contains("flags"));

    let err = LinkMessage::decode(&[1, 0, 0], 8).unwrap_err();
    assert!(format!("{err}").contains("name length"));

    let err = LinkMessage::decode(&[1, 0x08, 2], 8).unwrap_err();
    assert!(format!("{err}").contains("link type"));

    let data = [1, 0x08, 1, 1, b's', 0, 0];
    let err = LinkMessage::decode(&data, 8).unwrap_err();
    assert!(format!("{err}").contains("soft link length"));
}

#[test]
fn test_link_message_skips_trailing_bytes_after_fixed_payloads_like_libhdf5() {
    let mut hard = vec![1, 0, 1, b'h'];
    hard.extend_from_slice(&0x1234u64.to_le_bytes());
    hard.push(0);
    LinkMessage::decode(&hard, 8).expect("H5O__link_decode tolerates trailing hard-link bytes");

    let mut soft = vec![1, 0x08, 1, 1, b's'];
    soft.extend_from_slice(&1u16.to_le_bytes());
    soft.push(b'/');
    soft.push(0);
    LinkMessage::decode(&soft, 8).expect("H5O__link_decode tolerates trailing soft-link bytes");

    let user_defined = [1, 0x08, 65, 1, b'u', 1, 2, 3];
    LinkMessage::decode(&user_defined, 8).expect("user-defined link payload should remain opaque");
}

#[test]
fn test_external_link_rejects_invalid_header_and_path_encoding() {
    fn external_link_message(header: u8, payload: &[u8]) -> Vec<u8> {
        let mut data = vec![1u8, 0x08, 64, 1, b'e'];
        let info_len = 1 + payload.len();
        data.extend_from_slice(&(info_len as u16).to_le_bytes());
        data.push(header);
        data.extend_from_slice(payload);
        data
    }

    for data in [
        external_link_message(0x10, b"file\0path\0"),
        external_link_message(0x01, b"file\0path\0"),
        external_link_message(0x00, b"file\0path"),
        external_link_message(0x00, b"file\0"),
        external_link_message(0x00, b"\xff\0path\0"),
        external_link_message(0x00, b"file\0\xff\0"),
    ] {
        let err =
            LinkMessage::decode(&data, 8).expect_err("malformed external link buffer should fail");
        assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
    }
}

#[test]
fn test_link_message_rejects_invalid_utf8_text() {
    let mut hard = vec![1, 0, 1, 0xff];
    hard.extend_from_slice(&0u64.to_le_bytes());
    let err = LinkMessage::decode(&hard, 8).expect_err("invalid UTF-8 link name should fail");
    assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));

    let mut soft = vec![1, 0x08, 1, 4, b's', b'o', b'f', b't'];
    soft.extend_from_slice(&2u16.to_le_bytes());
    soft.extend_from_slice(&[0xff, 0]);
    let err = LinkMessage::decode(&soft, 8).expect_err("invalid UTF-8 soft target should fail");
    assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
}

#[test]
fn test_info_messages_reject_truncated_addresses() {
    for data in [
        vec![0, 0, 1, 2, 3],
        vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 2],
        vec![0, 0x03, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2],
    ] {
        let err =
            LinkInfoMessage::decode(&data, 8).expect_err("truncated link info message should fail");
        assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
    }

    for data in [
        vec![0, 0, 1, 2, 3],
        vec![0, 0x01, 1],
        vec![0, 0x02, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2],
    ] {
        let err = AttributeInfoMessage::decode(&data, 8)
            .expect_err("truncated attribute info message should fail");
        assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
    }
}

#[test]
fn test_info_messages_reject_bad_flags_and_invalid_address_width() {
    let symbol = [0; 2];
    assert!(SymbolTableMessage::decode(&symbol, 0).is_err());
    assert!(SymbolTableMessage::decode(&symbol, 9).is_err());
    let mut undefined_btree = Vec::new();
    undefined_btree.extend_from_slice(&UNDEF_ADDR.to_le_bytes());
    undefined_btree.extend_from_slice(&1u64.to_le_bytes());
    let err = SymbolTableMessage::decode(&undefined_btree, 8)
        .expect_err("undefined symbol-table B-tree address should fail");
    assert!(format!("{err}").contains("undefined"));
    let mut undefined_heap = Vec::new();
    undefined_heap.extend_from_slice(&1u64.to_le_bytes());
    undefined_heap.extend_from_slice(&UNDEF_ADDR.to_le_bytes());
    let err = SymbolTableMessage::decode(&undefined_heap, 8)
        .expect_err("undefined symbol-table local heap address should fail");
    assert!(format!("{err}").contains("undefined"));

    let mut link_info = vec![0, 0];
    link_info.extend_from_slice(&1u64.to_le_bytes());
    link_info.extend_from_slice(&2u64.to_le_bytes());
    let mut bad_link_flags = link_info.clone();
    bad_link_flags[1] = 0x80;
    assert!(LinkInfoMessage::decode(&bad_link_flags, 8).is_err());

    let mut attr_info = vec![0, 0];
    attr_info.extend_from_slice(&1u64.to_le_bytes());
    attr_info.extend_from_slice(&2u64.to_le_bytes());
    let mut bad_attr_flags = attr_info.clone();
    bad_attr_flags[1] = 0x80;
    assert!(AttributeInfoMessage::decode(&bad_attr_flags, 8).is_err());

    let mut indexed_link_without_tracking = vec![0, 0x02];
    indexed_link_without_tracking.extend_from_slice(&1u64.to_le_bytes());
    indexed_link_without_tracking.extend_from_slice(&2u64.to_le_bytes());
    indexed_link_without_tracking.extend_from_slice(&3u64.to_le_bytes());
    let decoded = LinkInfoMessage::decode(&indexed_link_without_tracking, 8)
        .expect("H5O__linfo_decode permits indexed creation order without tracking");
    assert_eq!(decoded.max_creation_index, None);
    assert_eq!(decoded.corder_btree_addr, Some(3));

    let mut indexed_attr_without_tracking = vec![0, 0x02];
    indexed_attr_without_tracking.extend_from_slice(&1u64.to_le_bytes());
    indexed_attr_without_tracking.extend_from_slice(&2u64.to_le_bytes());
    indexed_attr_without_tracking.extend_from_slice(&3u64.to_le_bytes());
    let decoded = AttributeInfoMessage::decode(&indexed_attr_without_tracking, 8)
        .expect("H5O__ainfo_decode permits indexed creation order without tracking");
    assert_eq!(decoded.max_creation_index, None);
    assert_eq!(decoded.corder_btree_addr, Some(3));

    let mut trailing_symbol = Vec::new();
    trailing_symbol.extend_from_slice(&1u64.to_le_bytes());
    trailing_symbol.extend_from_slice(&2u64.to_le_bytes());
    trailing_symbol.push(0);
    SymbolTableMessage::decode(&trailing_symbol, 8)
        .expect("H5O__stab_decode tolerates trailing payload bytes");

    let mut trailing_link_info = link_info;
    trailing_link_info.push(0);
    LinkInfoMessage::decode(&trailing_link_info, 8)
        .expect("H5O__linfo_decode tolerates trailing payload bytes");

    let mut trailing_attr_info = attr_info;
    trailing_attr_info.push(0);
    AttributeInfoMessage::decode(&trailing_attr_info, 8)
        .expect("H5O__ainfo_decode tolerates trailing payload bytes");
}

#[test]
fn test_v1_messages_skip_reserved_header_bytes_like_libhdf5() {
    let mut dataspace = vec![1, 0, 0, 0, 0, 0, 0, 0];
    dataspace[3] = 1;
    DataspaceMessage::decode(&dataspace).expect("H5O__sdspace_decode skips v1 reserved bytes");

    let datatype = fixed_point_datatype_bytes(4, 0, 32);
    let attr_dataspace = dataspace;
    let mut attribute = vec![1, 1];
    attribute.extend_from_slice(&2u16.to_le_bytes());
    attribute.extend_from_slice(&(datatype.len() as u16).to_le_bytes());
    attribute.extend_from_slice(&(attr_dataspace.len() as u16).to_le_bytes());
    attribute.extend_from_slice(b"a\0");
    attribute.extend_from_slice(&[0; 6]);
    attribute.extend_from_slice(&datatype);
    attribute.extend_from_slice(&[0; 4]);
    attribute.extend_from_slice(&attr_dataspace);
    attribute.extend_from_slice(&123i32.to_le_bytes());
    AttributeMessage::decode(&attribute).expect("H5O__attr_decode skips the v1 reserved byte");

    let mut filters = vec![1, 0, 0, 0, 0, 0, 0, 0];
    filters[2] = 1;
    FilterPipelineMessage::decode(&filters).expect("H5O__pline_decode skips v1 reserved bytes");
}

#[test]
fn test_symbol_table_node_rejects_invalid_entry_cache_type() {
    let mut data = b"SNOD".to_vec();
    data.push(1); // version
    data.push(0); // reserved
    data.extend_from_slice(&1u16.to_le_bytes()); // one symbol
    data.extend_from_slice(&0u64.to_le_bytes()); // name offset
    data.extend_from_slice(&32u64.to_le_bytes()); // object header address
    data.extend_from_slice(&3u32.to_le_bytes()); // invalid cache type
    data.extend_from_slice(&0u32.to_le_bytes()); // reserved
    data.extend_from_slice(&[0u8; 16]); // scratch pad

    let mut reader = HdfReader::new(Cursor::new(data));
    let err = SymbolTableNode::read_at(&mut reader, 0)
        .expect_err("invalid symbol table cache type should fail");
    assert!(
        format!("{err}").contains("cache type"),
        "expected cache-type error, got: {err}"
    );
}

#[test]
fn test_symbol_table_node_skips_reserved_bytes_like_libhdf5() {
    fn snod(node_reserved: u8, entry_reserved: u32) -> Vec<u8> {
        let mut data = b"SNOD".to_vec();
        data.push(1); // version
        data.push(node_reserved);
        data.extend_from_slice(&1u16.to_le_bytes()); // one symbol
        data.extend_from_slice(&0u64.to_le_bytes()); // name offset
        data.extend_from_slice(&32u64.to_le_bytes()); // object header address
        data.extend_from_slice(&0u32.to_le_bytes()); // no cached info
        data.extend_from_slice(&entry_reserved.to_le_bytes());
        data.extend_from_slice(&[0u8; 16]); // scratch pad
        data
    }

    for data in [snod(1, 0), snod(0, 1)] {
        let mut reader = HdfReader::new(Cursor::new(data));
        SymbolTableNode::read_at(&mut reader, 0)
            .expect("H5G_cache_load/H5G_ent_decode skip symbol-table reserved bytes");
    }
}

#[test]
fn test_heap_prefixes_skip_nonzero_reserved_bytes_like_libhdf5() {
    let mut local_heap = b"HEAP".to_vec();
    local_heap.push(0);
    local_heap.extend_from_slice(&[0, 1, 0]);
    local_heap.extend_from_slice(&0u64.to_le_bytes());
    local_heap.extend_from_slice(&UNDEF_ADDR.to_le_bytes());
    local_heap.extend_from_slice(&UNDEF_ADDR.to_le_bytes());
    let mut reader = HdfReader::new(Cursor::new(local_heap));
    LocalHeap::decode_prefix(&mut reader, 0)
        .expect("H5HL__hdr_deserialize skips local-heap reserved bytes");

    let mut global_heap = b"GCOL".to_vec();
    global_heap.push(1);
    global_heap.extend_from_slice(&[0, 1, 0]);
    global_heap.extend_from_slice(&16u64.to_le_bytes());
    let mut reader = HdfReader::new(Cursor::new(global_heap));
    GlobalHeapCollection::decode_header(&mut reader, 0)
        .expect("H5HG__hdr_deserialize skips reserved bytes");
}

#[test]
fn test_global_heap_object_skips_reserved_bytes_like_libhdf5() {
    let mut global_heap = b"GCOL".to_vec();
    global_heap.push(1);
    global_heap.extend_from_slice(&[0; 3]);
    global_heap.extend_from_slice(&32u64.to_le_bytes());
    global_heap.extend_from_slice(&1u16.to_le_bytes());
    global_heap.extend_from_slice(&1u16.to_le_bytes());
    global_heap.extend_from_slice(&1u32.to_le_bytes());
    global_heap.extend_from_slice(&0u64.to_le_bytes());

    let mut reader = HdfReader::new(Cursor::new(global_heap));
    GlobalHeapCollection::read_at(&mut reader, 0)
        .expect("H5HG__cache_heap_deserialize skips object reserved bytes");
}

#[test]
fn test_dataspace_empty() {
    assert!(DataspaceMessage::decode(&[]).is_err());
}

#[test]
fn test_dataspace_truncated() {
    let data = vec![2, 3, 0, 1]; // version 2, ndims=3, flags=0, type=simple -- missing dim data
    assert!(DataspaceMessage::decode(&data).is_err());
}

#[test]
fn test_dataspace_rejects_truncated_declared_dims() {
    for data in [
        vec![1, 1, 0, 0],
        vec![1, 1, 0, 0, 0, 0, 0, 0],
        vec![2, 1, 0, 1, 1, 2, 3],
        vec![2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    ] {
        let err = DataspaceMessage::decode(&data)
            .expect_err("truncated dataspace dimensions should fail");
        assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
    }
}

#[test]
fn test_dataspace_rejects_unsupported_or_invalid_flags() {
    let err = DataspaceMessage::decode(&[1, 0, 0x02, 0, 0, 0, 0, 0])
        .expect_err("dataspace v1 permutation flag is unsupported");
    assert!(
        format!("{err}").contains("flags"),
        "expected flags error, got: {err}"
    );

    let err = DataspaceMessage::decode(&[2, 0, 0x80, 0])
        .expect_err("dataspace v2 unknown flags should fail");
    assert!(
        format!("{err}").contains("flags"),
        "expected flags error, got: {err}"
    );
}

#[test]
fn test_dataspace_rejects_current_dimension_greater_than_max() {
    let mut v1 = vec![1, 1, 1, 0, 0, 0, 0, 0];
    v1.extend_from_slice(&5u64.to_le_bytes());
    v1.extend_from_slice(&4u64.to_le_bytes());
    let err = DataspaceMessage::decode(&v1).unwrap_err();
    assert!(format!("{err}").contains("exceeds maximum"));

    let mut v2 = vec![2, 1, 1, 1];
    v2.extend_from_slice(&5u64.to_le_bytes());
    v2.extend_from_slice(&4u64.to_le_bytes());
    let err = DataspaceMessage::decode(&v2).unwrap_err();
    assert!(format!("{err}").contains("exceeds maximum"));
}

#[test]
fn test_datatype_empty() {
    assert!(DatatypeMessage::decode(&[]).is_err());
}

#[test]
fn test_datatype_truncated() {
    let data = vec![0x10, 0, 0, 0]; // class 0, version 1, 4 bytes -- missing size
    assert!(DatatypeMessage::decode(&data).is_err());
}

#[test]
fn test_datatype_rejects_truncated_fixed_size_properties() {
    for data in [
        vec![0x10, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0],
        vec![0x11, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        vec![0x14, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0],
    ] {
        let err =
            DatatypeMessage::decode(&data).expect_err("truncated datatype properties should fail");
        assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
    }
}

#[test]
fn test_message_decoders_reject_invalid_versions_and_classes_as_format_errors() {
    let datatype_v5 = vec![0x50, 0, 0, 0, 4, 0, 0, 0, 0, 0, 32, 0];
    let datatype_cases = [
        vec![0x00, 0, 0, 0, 1, 0, 0, 0, 0, 8, 0, 8],
        vec![0x10, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 8],
        datatype_v5.clone(),
        vec![0x1a, 1, 0, 0, 4, 0, 0, 0, 1, 4, 0, 0, 0],
        vec![0x19, 2, 0, 0, 8, 0, 0, 0, 1, 2, 3, 4],
    ];
    for data in datatype_cases {
        let err = DatatypeMessage::decode(&data)
            .expect_err("invalid datatype version/class combination should fail");
        assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
    }

    let err = hdf5_pure_rust::engine::object_api::H5O__dtype_decode_helper(&datatype_v5)
        .expect_err("datatype message v5 should be outside the supported libhdf5 bound");
    assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));

    for err in [
        DataspaceMessage::decode(&[3, 0, 0, 0]).expect_err("invalid dataspace version"),
        DataLayoutMessage::decode(&[9, 0], 8, 8).expect_err("invalid layout version"),
        LinkMessage::decode(&[2, 0], 8).expect_err("invalid link version"),
        AttributeMessage::decode(&[4, 0, 0, 0, 0, 0]).expect_err("invalid attribute version"),
        FillValueMessage::decode(&[9]).expect_err("invalid fill value version"),
        FilterPipelineMessage::decode(&[9, 0]).expect_err("invalid filter pipeline version"),
        LinkInfoMessage::decode(&[1, 0], 8).expect_err("invalid link info version"),
        AttributeInfoMessage::decode(&[1, 0], 8).expect_err("invalid attribute info version"),
    ] {
        assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
    }
}

#[test]
fn test_message_decoders_reject_every_truncated_prefix() {
    fn assert_prefixes_fail<F>(name: &str, data: &[u8], decode: F)
    where
        F: Fn(&[u8]) -> hdf5_pure_rust::Result<()>,
    {
        decode(data).unwrap_or_else(|err| panic!("{name} full payload should decode: {err}"));
        for len in 0..data.len() {
            let prefix = &data[..len];
            assert!(
                decode(prefix).is_err(),
                "{name} prefix length {len} decoded unexpectedly"
            );
        }
    }

    let datatype = vec![0x10, 0, 0, 0, 4, 0, 0, 0, 0, 0, 32, 0];
    assert_prefixes_fail("datatype", &datatype, |data| {
        DatatypeMessage::decode(data).map(|_| ())
    });

    let mut dataspace = vec![2, 1, 0, 1];
    dataspace.extend_from_slice(&3u64.to_le_bytes());
    assert_prefixes_fail("dataspace", &dataspace, |data| {
        DataspaceMessage::decode(data).map(|_| ())
    });

    let mut layout = vec![3, 1];
    layout.extend_from_slice(&64u64.to_le_bytes());
    layout.extend_from_slice(&12u64.to_le_bytes());
    assert_prefixes_fail("data layout", &layout, |data| {
        DataLayoutMessage::decode(data, 8, 8).map(|_| ())
    });

    let mut link = vec![1, 0, 1, b'x'];
    link.extend_from_slice(&64u64.to_le_bytes());
    assert_prefixes_fail("link", &link, |data| {
        LinkMessage::decode(data, 8).map(|_| ())
    });

    let mut attribute = vec![3, 0];
    attribute.extend_from_slice(&2u16.to_le_bytes());
    attribute.extend_from_slice(&(datatype.len() as u16).to_le_bytes());
    attribute.extend_from_slice(&4u16.to_le_bytes());
    attribute.push(0);
    attribute.extend_from_slice(b"a\0");
    attribute.extend_from_slice(&datatype);
    attribute.extend_from_slice(&[2, 0, 0, 0]);
    attribute.extend_from_slice(&[0; 4]);
    assert_prefixes_fail("attribute", &attribute, |data| {
        AttributeMessage::decode(data).map(|_| ())
    });

    let fill_value = vec![3, 0x20, 4, 0, 0, 0, 1, 2, 3, 4];
    assert_prefixes_fail("fill value", &fill_value, |data| {
        FillValueMessage::decode(data).map(|_| ())
    });

    let mut filter_pipeline = vec![2, 1];
    filter_pipeline.extend_from_slice(&FILTER_DEFLATE.to_le_bytes());
    filter_pipeline.extend_from_slice(&0u16.to_le_bytes());
    filter_pipeline.extend_from_slice(&1u16.to_le_bytes());
    filter_pipeline.extend_from_slice(&6u32.to_le_bytes());
    assert_prefixes_fail("filter pipeline", &filter_pipeline, |data| {
        FilterPipelineMessage::decode(data).map(|_| ())
    });
}

#[test]
fn test_attribute_null_dataspace_reports_zero_data_size() {
    let attribute = AttributeMessage {
        version: 3,
        name: "null".to_string(),
        char_encoding: 0,
        datatype: DatatypeMessage {
            version: 1,
            class: DatatypeClass::FixedPoint,
            class_bits: [0; 3],
            size: 4,
            properties: vec![0, 0, 32, 0],
        },
        dataspace: DataspaceMessage {
            version: 2,
            space_type: hdf5_pure_rust::format::messages::dataspace::DataspaceType::Null,
            ndims: 0,
            dims: Vec::new(),
            max_dims: None,
        },
        data: Vec::new(),
    };

    assert_eq!(attribute.num_elements().unwrap(), 0);
    assert_eq!(attribute.data_size().unwrap(), 0);
}

#[test]
fn test_message_size_arithmetic_overflow_returns_format_error() {
    let mut overflowing_layout = vec![1, 3, 1, 0, 0, 0, 0, 0];
    overflowing_layout.extend_from_slice(&64u64.to_le_bytes());
    overflowing_layout.extend_from_slice(&u32::MAX.to_le_bytes());
    overflowing_layout.extend_from_slice(&u32::MAX.to_le_bytes());
    overflowing_layout.extend_from_slice(&u32::MAX.to_le_bytes());
    let layout = DataLayoutMessage::decode(&overflowing_layout, 8, 8)
        .expect("v1 contiguous layout should parse without inferring a truncated size");
    assert_eq!(layout.contiguous_addr, Some(64));
    assert_eq!(layout.contiguous_size, None);

    let attribute = AttributeMessage {
        version: 3,
        name: "overflow".to_string(),
        char_encoding: 0,
        datatype: DatatypeMessage {
            version: 1,
            class: DatatypeClass::FixedPoint,
            class_bits: [0; 3],
            size: 8,
            properties: vec![0, 0, 64, 0],
        },
        dataspace: DataspaceMessage {
            version: 2,
            space_type: hdf5_pure_rust::format::messages::dataspace::DataspaceType::Simple,
            ndims: 2,
            dims: vec![u64::MAX, 2],
            max_dims: None,
        },
        data: Vec::new(),
    };
    let err = attribute
        .data_size()
        .expect_err("overflowing attribute data size should fail");
    assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
}

#[test]
fn test_declared_allocation_counts_are_capped() {
    let err =
        DataspaceMessage::decode(&[2, 33, 0, 1]).expect_err("dataspace rank above cap should fail");
    assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));

    let err =
        FilterPipelineMessage::decode(&[2, 33]).expect_err("filter count above cap should fail");
    assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));

    let err = DataLayoutMessage::decode(&[1, 33, 1, 0, 0, 0, 0, 0], 8, 8)
        .expect_err("layout rank above cap should fail");
    assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));

    let mut local_heap = b"HEAP".to_vec();
    local_heap.push(0);
    local_heap.extend_from_slice(&[0; 3]);
    local_heap.extend_from_slice(&(4u64 * 1024 * 1024 * 1024 + 1).to_le_bytes());
    local_heap.extend_from_slice(&0u64.to_le_bytes());
    local_heap.extend_from_slice(&32u64.to_le_bytes());
    let mut reader = HdfReader::new(Cursor::new(local_heap));
    let err = LocalHeap::read_at(&mut reader, 0).expect_err("oversized local heap should fail");
    assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));

    let heap = LocalHeap {
        data: b"valid\0unterminated".to_vec(),
    };
    assert_eq!(heap.get_str(0).unwrap(), "valid");
    let err = heap
        .get_str(6)
        .expect_err("unterminated local heap string should fail");
    assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
    let err = LocalHeap {
        data: vec![0xff, 0],
    }
    .get_str(0)
    .expect_err("invalid UTF-8 local heap string should fail");
    assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
    let err = heap
        .get_str(heap.data.len())
        .expect_err("out-of-bounds local heap string should fail");
    assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));

    let mut global_heap = b"GCOL".to_vec();
    global_heap.push(1);
    global_heap.extend_from_slice(&[0; 3]);
    global_heap.extend_from_slice(&64u64.to_le_bytes());
    global_heap.extend_from_slice(&1u16.to_le_bytes());
    global_heap.extend_from_slice(&1u16.to_le_bytes());
    global_heap.extend_from_slice(&[0; 4]);
    global_heap.extend_from_slice(&(4u64 * 1024 * 1024 * 1024 + 1).to_le_bytes());
    let mut reader = HdfReader::new(Cursor::new(global_heap));
    let err = GlobalHeapCollection::read_at(&mut reader, 0)
        .expect_err("oversized global heap object should fail");
    assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));

    let mut fixed = b"FAHD".to_vec();
    fixed.push(0);
    fixed.push(0);
    fixed.push(8);
    fixed.push(0);
    fixed.extend_from_slice(&1_000_001u64.to_le_bytes());
    fixed.extend_from_slice(&UNDEF_ADDR.to_le_bytes());
    fixed.extend_from_slice(&0u32.to_le_bytes());
    let mut reader = HdfReader::new(Cursor::new(fixed));
    let mut elements = Vec::new();
    let err = fixed_array::read_fixed_array_chunks_into(&mut reader, 0, false, 0, &mut elements)
        .expect_err("oversized fixed array should fail");
    assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));

    let mut extensible = b"EAHD".to_vec();
    extensible.extend_from_slice(&[0, 0, 8, 1, 1, 1, 1, 1]);
    extensible.extend_from_slice(&0u64.to_le_bytes());
    extensible.extend_from_slice(&0u64.to_le_bytes());
    extensible.extend_from_slice(&0u64.to_le_bytes());
    extensible.extend_from_slice(&0u64.to_le_bytes());
    extensible.extend_from_slice(&1_000_001u64.to_le_bytes());
    extensible.extend_from_slice(&0u64.to_le_bytes());
    extensible.extend_from_slice(&UNDEF_ADDR.to_le_bytes());
    extensible.extend_from_slice(&0u32.to_le_bytes());
    let mut reader = HdfReader::new(Cursor::new(extensible));
    elements.clear();
    let err = extensible_array::read_extensible_array_chunks_into(
        &mut reader,
        0,
        false,
        0,
        &mut elements,
    )
    .expect_err("oversized extensible array should fail");
    assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
}

#[test]
fn test_compound_field_preserves_member_byte_order() {
    let mut data = Vec::new();
    data.push(0x36); // version 3, compound class
    data.extend_from_slice(&[1, 0, 0]); // one member
    data.extend_from_slice(&4u32.to_le_bytes()); // record size
    data.extend_from_slice(b"x\0");
    data.push(0); // member offset
    data.push(0x10); // version 1, fixed-point class
    data.extend_from_slice(&[1, 0, 0]); // big-endian member
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&0u16.to_le_bytes()); // bit offset
    data.extend_from_slice(&32u16.to_le_bytes()); // bit precision

    let dtype = DatatypeMessage::decode(&data).unwrap();
    let fields: Vec<_> = dtype
        .compound_fields_iter()
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(fields.len(), 1);
    assert_eq!(fields[0].name, "x");
    assert_eq!(fields[0].byte_order, Some(ByteOrder::BigEndian));
    assert_eq!(fields[0].datatype.size, 4);
}

#[test]
fn test_compound_fields_advance_padded_opaque_member_datatype() {
    let mut data = Vec::new();
    data.push(0x36); // version 3, compound class
    data.extend_from_slice(&[2, 0, 0]); // two members
    data.extend_from_slice(&5u32.to_le_bytes()); // record size

    data.extend_from_slice(b"o\0");
    data.push(0); // member offset
    data.push(0x15); // version 1, opaque class
    data.extend_from_slice(&[8, 0, 0]); // padded opaque tag length
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(b"abc\0\0\0\0\0");

    data.extend_from_slice(b"i\0");
    data.push(1); // member offset
    data.push(0x10); // version 1, fixed-point class
    data.extend_from_slice(&[0, 0, 0]);
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&0u16.to_le_bytes());
    data.extend_from_slice(&32u16.to_le_bytes());

    let dtype = DatatypeMessage::decode(&data).unwrap();
    let fields: Vec<_> = dtype
        .compound_fields_iter()
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(fields.len(), 2);
    assert_eq!(fields[0].datatype.opaque_tag_str(), Some("abc"));
    assert_eq!(fields[1].name, "i");
    assert_eq!(fields[1].datatype.size, 4);
}

fn validate_compound_fields(dtype: &DatatypeMessage) -> hdf5_pure_rust::Result<()> {
    for field in dtype.compound_fields_iter()? {
        field?;
    }
    Ok(())
}

#[test]
fn test_compound_fields_reject_overlapping_members() {
    let mut data = Vec::new();
    data.push(0x36); // version 3, compound class
    data.extend_from_slice(&[2, 0, 0]); // two members
    data.extend_from_slice(&4u32.to_le_bytes()); // record size

    data.extend_from_slice(b"lo\0");
    data.push(0); // member offset
    data.push(0x10); // version 1, fixed-point class
    data.extend_from_slice(&[0, 0, 0]);
    data.extend_from_slice(&2u32.to_le_bytes());
    data.extend_from_slice(&0u16.to_le_bytes());
    data.extend_from_slice(&16u16.to_le_bytes());

    data.extend_from_slice(b"word\0");
    data.push(0); // overlapping offset
    data.push(0x10); // version 1, fixed-point class
    data.extend_from_slice(&[0, 0, 0]);
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&0u16.to_le_bytes());
    data.extend_from_slice(&32u16.to_le_bytes());

    let dtype = DatatypeMessage::decode(&data).unwrap();
    let err =
        validate_compound_fields(&dtype).expect_err("overlapping compound members should fail");
    assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
}

#[test]
fn test_opaque_datatype_rejects_unaligned_tag_length() {
    let data = [0x15, 3, 0, 0, 1, 0, 0, 0, b'a', b'b', b'c'];
    let err = DatatypeMessage::decode(&data).unwrap_err();
    assert!(
        format!("{err}").contains("not aligned"),
        "expected opaque tag alignment error, got: {err}"
    );
}

#[test]
fn test_opaque_datatype_rejects_invalid_utf8_tag() {
    let data = [0x15, 8, 0, 0, 1, 0, 0, 0, 0xff, 0, 0, 0, 0, 0, 0, 0];
    let err = DatatypeMessage::decode(&data).unwrap_err();
    assert!(
        format!("{err}").contains("not UTF-8"),
        "expected opaque tag UTF-8 error, got: {err}"
    );
}

#[test]
fn test_string_datatype_preserves_unknown_padding_and_charset_like_libhdf5() {
    let invalid_padding = [0x13, 3, 0, 0, 1, 0, 0, 0];
    let decoded = DatatypeMessage::decode(&invalid_padding)
        .expect("H5O__dtype_decode_helper stores string padding bits without range checks");
    assert_eq!(decoded.string_padding(), Some(3));

    let invalid_charset = [0x13, 0x20, 0, 0, 1, 0, 0, 0];
    let decoded = DatatypeMessage::decode(&invalid_charset)
        .expect("H5O__dtype_decode_helper stores string charset bits without range checks");
    assert_eq!(decoded.char_set(), Some(2));
}

#[test]
fn test_reference_datatype_rejects_invalid_reference_type() {
    let invalid_ref_type = [0x17, 2, 0, 0, 8, 0, 0, 0];
    let err = DatatypeMessage::decode(&invalid_ref_type).unwrap_err();
    assert!(
        format!("{err}").contains("reference datatype type"),
        "expected reference type error, got: {err}"
    );
}

#[test]
fn test_compound_fields_reject_truncated_member_metadata() {
    let cases = [
        vec![0x36, 1, 0, 0, 4, 0, 0, 0],
        vec![0x36, 1, 0, 0, 4, 0, 0, 0, b'x'],
        vec![0x36, 1, 0, 0, 4, 0, 0, 0, b'x', 0, 1, 2],
        {
            let mut data = vec![0x36, 2, 0, 0, 4, 0, 0, 0];
            data.extend_from_slice(b"x\0");
            data.push(0);
            data.extend_from_slice(&[0x10, 0, 0, 0, 4, 0, 0, 0, 0, 0, 32, 0]);
            data
        },
        {
            let mut data = vec![0x16, 1, 0, 0, 4, 0, 0, 0];
            data.extend_from_slice(b"x\0");
            data.extend_from_slice(&[0; 6]);
            data.extend_from_slice(&0u32.to_le_bytes());
            data.extend_from_slice(&[1, 2, 3]);
            data
        },
        {
            let mut data = vec![0x26, 1, 0, 0, 4, 0, 0, 0];
            data.extend_from_slice(b"x\0");
            data.extend_from_slice(&[0; 6]);
            data.extend_from_slice(&0u32.to_le_bytes());
            data.extend_from_slice(&[1, 2, 3, 4]);
            data
        },
        {
            let mut data = vec![0x46, 1, 0, 0, 0, 1, 0, 0];
            data.extend_from_slice(b"x\0");
            data.push(0);
            data
        },
        {
            let mut data = vec![0x36, 1, 0, 0, 4, 0, 0, 0];
            data.extend_from_slice(b"x\0");
            data.push(0);
            data.extend_from_slice(&[0x10, 0, 0]);
            data
        },
        {
            let mut data = vec![0x36, 1, 0, 0, 4, 0, 0, 0];
            data.extend_from_slice(b"x\0");
            data.push(0);
            data.extend_from_slice(&[0x10, 0, 0, 0, 4, 0, 0, 0, 1]);
            data
        },
    ];

    for data in cases {
        let dtype = DatatypeMessage::decode(&data).expect("compound header should decode");
        let err = dtype
            .compound_fields_iter()
            .and_then(|fields| fields.collect::<Result<Vec<_>, _>>())
            .expect_err("truncated compound member metadata should fail");
        assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
    }
}

#[test]
fn test_enum_members_reject_truncated_metadata() {
    let enum_header = |version: u8| vec![(version << 4) | 8, 1, 0, 0, 1, 0, 0, 0];
    let enum_header_n =
        |version: u8, nmembers: u8| vec![(version << 4) | 8, nmembers, 0, 0, 1, 0, 0, 0];
    let base_u8 = [0x10, 0, 0, 0, 1, 0, 0, 0, 0, 0, 8, 0];
    let base_u16 = [0x10, 0, 0, 0, 2, 0, 0, 0, 0, 0, 16, 0];
    let cases = [
        {
            let mut data = enum_header(3);
            data.extend_from_slice(&[0x10, 0, 0]);
            data
        },
        {
            let mut data = enum_header(3);
            data.extend_from_slice(&[0x10, 0, 0, 0, 1, 0, 0, 0, 0]);
            data
        },
        {
            let mut data = enum_header(3);
            data.extend_from_slice(&base_u8);
            data
        },
        {
            let mut data = enum_header(3);
            data.extend_from_slice(&base_u8);
            data.extend_from_slice(b"A");
            data
        },
        {
            let mut data = enum_header(1);
            data.extend_from_slice(&base_u8);
            data.extend_from_slice(b"A\0");
            data
        },
        {
            let mut data = enum_header(3);
            data.extend_from_slice(&base_u8);
            data.extend_from_slice(b"A\0");
            data
        },
        {
            let mut data = enum_header(3);
            data.extend_from_slice(&base_u16);
            data.extend_from_slice(b"A\0");
            data.push(1);
            data
        },
        {
            let mut data = enum_header_n(3, 2);
            data.extend_from_slice(&base_u8);
            data.extend_from_slice(b"A\0B\0");
            data.push(1);
            data
        },
    ];

    for data in cases {
        let dtype = DatatypeMessage::decode(&data).expect("enum header should decode");
        let err = dtype
            .enum_members_iter()
            .and_then(|members| members.collect::<Result<Vec<_>, _>>())
            .expect_err("truncated enum member metadata should fail");
        assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
    }
}

#[test]
fn test_enum_members_reject_invalid_utf8_names() {
    let mut data = vec![(3 << 4) | 8, 1, 0, 0, 1, 0, 0, 0];
    data.extend_from_slice(&[0x10, 0, 0, 0, 1, 0, 0, 0, 0, 0, 8, 0]);
    data.extend_from_slice(&[0xff, 0]);
    data.push(1);
    let dtype = DatatypeMessage::decode(&data).expect("enum header should decode");
    let err = dtype
        .enum_members_iter()
        .and_then(|members| members.collect::<Result<Vec<_>, _>>())
        .expect_err("invalid enum member name should fail");
    assert!(format!("{err}").contains("not UTF-8"));
}

#[test]
fn test_array_datatype_decode_rejects_truncated_metadata() {
    let base_i32 = [0x10, 0, 0, 0, 4, 0, 0, 0, 0, 0, 32, 0];
    let cases = [
        vec![0x3a, 0, 0, 0, 4, 0, 0, 0],
        vec![0x4a, 0, 0, 0, 4, 0, 0, 0, 33],
        vec![0x4a, 0, 0, 0, 4, 0, 0, 0, 255],
        vec![0x3a, 0, 0, 0, 4, 0, 0, 0, 1],
        vec![0x4a, 0, 0, 0, 4, 0, 0, 0, 2, 4, 0, 0, 0],
        vec![0x4a, 0, 0, 0, 4, 0, 0, 0, 1, 4, 0, 0, 0],
        {
            let mut data = vec![0x4a, 0, 0, 0, 4, 0, 0, 0, 1, 4, 0, 0, 0];
            data.extend_from_slice(&[0x10, 0, 0]);
            data
        },
        {
            let mut data = vec![0x4a, 0, 0, 0, 4, 0, 0, 0, 1, 4, 0, 0, 0];
            data.extend_from_slice(&base_i32[..9]);
            data
        },
    ];

    for data in cases {
        let err = DatatypeMessage::decode(&data)
            .expect_err("truncated array datatype metadata should fail during decode");
        assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
    }
}

#[test]
fn test_array_datatype_decode_rejects_zero_rank_or_dimension() {
    let base_i16 = [0x10, 0, 0, 0, 2, 0, 0, 0, 0, 0, 16, 0];

    let mut zero_rank = vec![0x4a, 0, 0, 0, 2, 0, 0, 0, 0];
    zero_rank.extend_from_slice(&base_i16);
    let err = DatatypeMessage::decode(&zero_rank).unwrap_err();
    assert!(format!("{err}").contains("rank"));

    let mut zero_dim = vec![0x4a, 0, 0, 0, 2, 0, 0, 0, 1];
    zero_dim.extend_from_slice(&0u32.to_le_bytes());
    zero_dim.extend_from_slice(&base_i16);
    let err = DatatypeMessage::decode(&zero_dim).unwrap_err();
    assert!(format!("{err}").contains("dimension"));
}

#[test]
fn test_array_datatype_decode_rejects_size_mismatch() {
    let base_i16 = [0x10, 0, 0, 0, 2, 0, 0, 0, 0, 0, 16, 0];
    let mut data = vec![0x4a, 0, 0, 0, 10, 0, 0, 0, 2];
    data.extend_from_slice(&2u32.to_le_bytes());
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&base_i16);

    let err = DatatypeMessage::decode(&data).unwrap_err();
    assert!(
        format!("{err}").contains("array datatype size"),
        "expected array size mismatch error, got: {err}"
    );
}

#[test]
fn test_array_dims_base_handles_v2_v3_v4_and_rejects_v1() {
    let base_i16 = [0x10, 0, 0, 0, 2, 0, 0, 0, 0, 0, 16, 0];

    let mut v1 = vec![0x1a, 0, 0, 0, 12, 0, 0, 0, 2, 0, 0, 0];
    v1.extend_from_slice(&2u32.to_le_bytes());
    v1.extend_from_slice(&3u32.to_le_bytes());
    v1.extend_from_slice(&base_i16);
    assert!(DatatypeMessage::decode(&v1).is_err());

    let mut v2 = vec![0x2a, 0, 0, 0, 12, 0, 0, 0, 2, 0, 0, 0];
    v2.extend_from_slice(&2u32.to_le_bytes());
    v2.extend_from_slice(&3u32.to_le_bytes());
    v2.extend_from_slice(&0u32.to_le_bytes());
    v2.extend_from_slice(&1u32.to_le_bytes());
    v2.extend_from_slice(&base_i16);
    let dtype = DatatypeMessage::decode(&v2).unwrap();
    let dims = dtype
        .array_dims_iter()
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let base = dtype.array_base().unwrap();
    assert_eq!(dims, vec![2, 3]);
    assert_eq!(base.size, 2);

    let mut v3 = vec![0x3a, 0, 0, 0, 12, 0, 0, 0, 2];
    v3.extend_from_slice(&2u32.to_le_bytes());
    v3.extend_from_slice(&3u32.to_le_bytes());
    v3.extend_from_slice(&base_i16);
    let dtype = DatatypeMessage::decode(&v3).unwrap();
    let dims = dtype
        .array_dims_iter()
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let base = dtype.array_base().unwrap();
    assert_eq!(dims, vec![2, 3]);
    assert_eq!(base.size, 2);

    let mut v4 = vec![0x4a, 0, 0, 0, 12, 0, 0, 0, 2];
    v4.extend_from_slice(&2u32.to_le_bytes());
    v4.extend_from_slice(&3u32.to_le_bytes());
    v4.extend_from_slice(&base_i16);
    let dtype = DatatypeMessage::decode(&v4).unwrap();
    let dims = dtype
        .array_dims_iter()
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let base = dtype.array_base().unwrap();
    assert_eq!(dims, vec![2, 3]);
    assert_eq!(base.size, 2);
}

#[test]
fn test_vlen_base_distinguishes_sequence_and_string_metadata() {
    let base_i32 = [0x10, 0, 0, 0, 4, 0, 0, 0, 0, 0, 32, 0];

    let mut direct_sequence = vec![0x39, 0, 0, 0, 16, 0, 0, 0];
    direct_sequence.extend_from_slice(&base_i32);
    let dtype = DatatypeMessage::decode(&direct_sequence).expect("vlen header should decode");
    assert_eq!(dtype.vlen_base().unwrap().unwrap().size, 4);

    let mut string_metadata = vec![0x39, 0, 0, 0, 16, 0, 0, 0];
    string_metadata.extend_from_slice(&[1, 0, 0, 0]);
    let dtype =
        DatatypeMessage::decode(&string_metadata).expect("vlen string header should decode");
    assert!(dtype.vlen_base().unwrap().is_none());

    let mut metadata_sequence = vec![0x39, 0, 0, 0, 16, 0, 0, 0];
    metadata_sequence.extend_from_slice(&[0, 0, 0, 0]);
    metadata_sequence.extend_from_slice(&base_i32);
    let dtype =
        DatatypeMessage::decode(&metadata_sequence).expect("vlen sequence header should decode");
    assert_eq!(dtype.vlen_base().unwrap().unwrap().size, 4);
}

#[test]
fn test_vlen_base_rejects_truncated_or_ambiguous_metadata() {
    let cases = [
        vec![0x39, 0, 0, 0, 16, 0, 0, 0],
        vec![0x39, 0, 0, 0, 16, 0, 0, 0, 1, 2],
        vec![0x39, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0x10, 0, 0],
        vec![
            0x39, 0, 0, 0, 16, 0, 0, 0, 0x10, 0, 0, 0, 4, 0, 0, 0, 0, 0, 32, 0, 99,
        ],
    ];

    for data in cases {
        let dtype = DatatypeMessage::decode(&data).expect("vlen header should decode");
        let err = dtype
            .vlen_base()
            .expect_err("truncated vlen metadata should fail");
        assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
    }
}

#[test]
fn test_layout_empty() {
    assert!(DataLayoutMessage::decode(&[], 8, 8).is_err());
}

#[test]
fn test_layout_bad_version() {
    let data = vec![99, 0]; // version 99
    assert!(DataLayoutMessage::decode(&data, 8, 8).is_err());
}

#[test]
fn test_layout_rejects_truncated_payloads() {
    for data in [
        vec![1, 1, 1, 0, 0, 0, 0, 0],
        vec![3, 0, 4, 0, 1],
        vec![3, 1, 0, 0],
        vec![4, 2, 0, 1, 8],
        vec![4, 3],
    ] {
        let err = DataLayoutMessage::decode(&data, 8, 8)
            .expect_err("truncated data layout message should fail without panicking");
        assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
    }
}

#[test]
fn test_filter_pipeline_empty() {
    assert!(FilterPipelineMessage::decode(&[]).is_err());
}

#[test]
fn test_filter_pipeline_rejects_truncated_decode_payloads() {
    for data in [
        vec![1, 1, 0, 0, 0, 0, 0, 0],
        vec![1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0, 0, 0, 0, 0],
        vec![2, 1],
        vec![2, 1, 0, 1],
        vec![2, 1, 1, 0, 0, 0, 1, 0],
    ] {
        let err = FilterPipelineMessage::decode(&data)
            .expect_err("truncated filter pipeline message should fail");
        assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
    }
}

#[test]
fn test_fill_value_rejects_truncated_defined_values() {
    assert!(FillValueMessage::decode(&[2, 0, 0, 1]).is_err());
    assert!(FillValueMessage::decode(&[3, 0x20, 4, 0, 0, 0, 1, 2]).is_err());
    assert!(FillValueMessage::decode_old(&[4, 0, 0, 0, 1, 2]).is_err());
}

#[test]
fn test_fill_value_v3_rejects_unknown_flags() {
    let err = FillValueMessage::decode(&[3, 0x40]).unwrap_err();
    assert!(format!("{err}").contains("flags"));
}

#[test]
fn test_fill_value_rejects_invalid_enum_values() {
    for data in [
        vec![2, 4, 0, 0], // invalid allocation time
        vec![2, 0, 3, 0], // invalid write time
        vec![2, 0, 0, 3], // invalid defined-state enum
        vec![3, 0x0c],    // invalid write time
        vec![3, 0x30],    // conflicting undefined and value-present flags
    ] {
        let err = FillValueMessage::decode(&data).expect_err("invalid fill-value enum should fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("invalid") || msg.contains("undefined and value-present"),
            "expected invalid enum error, got: {err}"
        );
    }
}

#[test]
fn test_fill_value_v2_nonzero_defined_state_has_payload_like_libhdf5() {
    let v2 = FillValueMessage::decode(&[2, 0, 0, 1, 4, 0, 0, 0, 0xf7, 0xff, 0xff, 0xff])
        .expect("H5O__fill_decode reads a payload for nonzero v2 fill-defined state");
    assert!(v2.defined);
    assert_eq!(v2.value.as_deref(), Some(&[0xf7, 0xff, 0xff, 0xff][..]));

    let undefined = FillValueMessage::decode(&[3, 0x00])
        .expect("H5O__fill_decode accepts v3 fill state without payload");
    assert!(undefined.defined);
    assert!(undefined.value.is_none());
}

#[test]
fn test_layout_rejects_invalid_ranks_flags_and_btree2_percents() {
    let err = DataLayoutMessage::decode(&[1, 0, 2, 0, 0, 0, 0, 0], 8, 8)
        .expect_err("v1/v2 zero rank must fail");
    assert!(format!("{err}").contains("rank"));

    let mut nonzero_reserved = vec![1u8, 1, 1, 0, 0, 1, 0, 0];
    nonzero_reserved.extend_from_slice(&0u64.to_le_bytes());
    nonzero_reserved.extend_from_slice(&4u32.to_le_bytes());
    DataLayoutMessage::decode(&nonzero_reserved, 8, 8)
        .expect("H5O__layout_decode skips v1/v2 reserved bytes");

    let mut v3_rank1 = vec![3u8, 2, 1];
    v3_rank1.extend_from_slice(&[0u8; 8]);
    v3_rank1.extend_from_slice(&4u32.to_le_bytes());
    let err = DataLayoutMessage::decode(&v3_rank1, 8, 8).expect_err("v3 chunk rank < 2 must fail");
    assert!(format!("{err}").contains("rank"));

    let mut bad_flags = vec![4u8, 2, 0x01, 1, 4];
    bad_flags.extend_from_slice(&4u32.to_le_bytes());
    bad_flags.push(2);
    bad_flags.extend_from_slice(&0u64.to_le_bytes());
    let err =
        DataLayoutMessage::decode(&bad_flags, 8, 8).expect_err("invalid chunk flags must fail");
    assert!(format!("{err}").contains("flags"));

    let mut bad_split = vec![4u8, 2, 0, 1, 4];
    bad_split.extend_from_slice(&4u32.to_le_bytes());
    bad_split.push(5);
    bad_split.extend_from_slice(&512u32.to_le_bytes());
    bad_split.push(101);
    bad_split.push(40);
    bad_split.extend_from_slice(&0u64.to_le_bytes());
    let err =
        DataLayoutMessage::decode(&bad_split, 8, 8).expect_err("split percent > 100 must fail");
    assert!(format!("{err}").contains("split percent"));

    let mut zero_node_size = vec![4u8, 2, 0, 1, 4];
    zero_node_size.extend_from_slice(&4u32.to_le_bytes());
    zero_node_size.push(5);
    zero_node_size.extend_from_slice(&0u32.to_le_bytes());
    zero_node_size.push(60);
    zero_node_size.push(40);
    zero_node_size.extend_from_slice(&0u64.to_le_bytes());
    let err = DataLayoutMessage::decode(&zero_node_size, 8, 8)
        .expect_err("B-tree2 node size 0 must fail");
    assert!(format!("{err}").contains("node size"));

    let mut zero_merge = vec![4u8, 2, 0, 1, 4];
    zero_merge.extend_from_slice(&4u32.to_le_bytes());
    zero_merge.push(5);
    zero_merge.extend_from_slice(&512u32.to_le_bytes());
    zero_merge.push(60);
    zero_merge.push(0);
    zero_merge.extend_from_slice(&0u64.to_le_bytes());
    let err = DataLayoutMessage::decode(&zero_merge, 8, 8).expect_err("merge percent 0 must fail");
    assert!(format!("{err}").contains("merge percent"));

    let mut forbidden_v1_index = vec![4u8, 2, 0, 1, 4];
    forbidden_v1_index.extend_from_slice(&4u32.to_le_bytes());
    forbidden_v1_index.push(0);
    let err = DataLayoutMessage::decode(&forbidden_v1_index, 8, 8)
        .expect_err("v4 B-tree v1 index type must fail");
    assert!(format!("{err}").contains("B-tree v1"));

    let mut v3_virtual = vec![3u8, 3];
    v3_virtual.extend_from_slice(&0u64.to_le_bytes());
    v3_virtual.extend_from_slice(&0u32.to_le_bytes());
    let err =
        DataLayoutMessage::decode(&v3_virtual, 8, 8).expect_err("v3 virtual layout must fail");
    assert!(format!("{err}").contains("version 4"));

    let mut bad_fixed_array = vec![4u8, 2, 0, 1, 4];
    bad_fixed_array.extend_from_slice(&4u32.to_le_bytes());
    bad_fixed_array.push(3);
    bad_fixed_array.push(0);
    bad_fixed_array.extend_from_slice(&0u64.to_le_bytes());
    let err = DataLayoutMessage::decode(&bad_fixed_array, 8, 8)
        .expect_err("fixed-array page bits 0 must fail");
    assert!(format!("{err}").contains("page bits"));

    let mut bad_extensible_array = vec![4u8, 2, 0, 1, 4];
    bad_extensible_array.extend_from_slice(&4u32.to_le_bytes());
    bad_extensible_array.push(4);
    bad_extensible_array.extend_from_slice(&[1, 0, 1, 1, 1]);
    bad_extensible_array.extend_from_slice(&0u64.to_le_bytes());
    let err = DataLayoutMessage::decode(&bad_extensible_array, 8, 8)
        .expect_err("extensible-array zero creation parameter must fail");
    assert!(format!("{err}").contains("index block elements"));

    let mut zero_elem_size_v3 = vec![3u8, 2, 2];
    zero_elem_size_v3.extend_from_slice(&0u64.to_le_bytes());
    zero_elem_size_v3.extend_from_slice(&4u32.to_le_bytes());
    zero_elem_size_v3.extend_from_slice(&0u32.to_le_bytes());
    let err = DataLayoutMessage::decode(&zero_elem_size_v3, 8, 8)
        .expect_err("v3 zero element size must fail");
    assert!(format!("{err}").contains("element size"));

    let mut zero_elem_size_v1 = vec![1u8, 2, 2, 0, 0, 0, 0, 0];
    zero_elem_size_v1.extend_from_slice(&0u64.to_le_bytes());
    zero_elem_size_v1.extend_from_slice(&4u32.to_le_bytes());
    zero_elem_size_v1.extend_from_slice(&0u32.to_le_bytes());
    let err = DataLayoutMessage::decode(&zero_elem_size_v1, 8, 8)
        .expect_err("v1 zero element size must fail");
    assert!(format!("{err}").contains("element size"));
}

#[test]
fn test_layout_accepts_equal_nonzero_btree2_percents() {
    let mut equal_percents = vec![4u8, 2, 0, 1, 4];
    equal_percents.extend_from_slice(&4u32.to_le_bytes());
    equal_percents.push(5);
    equal_percents.extend_from_slice(&512u32.to_le_bytes());
    equal_percents.push(60);
    equal_percents.push(60);
    equal_percents.extend_from_slice(&0u64.to_le_bytes());
    let layout = DataLayoutMessage::decode(&equal_percents, 8, 8)
        .expect("equal nonzero v4 B-tree2 percents should parse");
    assert_eq!(
        layout.chunk_index_type,
        Some(hdf5_pure_rust::format::messages::data_layout::ChunkIndexType::BTreeV2)
    );
}

#[test]
fn test_layout_v1_contiguous_does_not_infer_truncated_size() {
    let mut layout = vec![1u8, 1, 1, 0, 0, 0, 0, 0];
    layout.extend_from_slice(&0x1122334455667788u64.to_le_bytes());
    layout.extend_from_slice(&123u32.to_le_bytes());

    let decoded =
        DataLayoutMessage::decode(&layout, 8, 8).expect("v1 contiguous layout should parse");
    assert_eq!(decoded.contiguous_addr, Some(0x1122334455667788));
    assert_eq!(decoded.contiguous_size, None);
}

#[test]
fn test_layout_v4_preserves_encoded_chunk_dims() {
    let mut layout = vec![4u8, 2, 0, 2, 2];
    layout.extend_from_slice(&3u16.to_le_bytes());
    layout.extend_from_slice(&5u16.to_le_bytes());
    layout.push(2);
    layout.extend_from_slice(&0x8877665544332211u64.to_le_bytes());

    let decoded =
        DataLayoutMessage::decode(&layout, 8, 8).expect("v4 implicit chunk layout should parse");
    assert_eq!(decoded.chunk_dims, Some(vec![3, 5]));
    assert_eq!(decoded.chunk_encoded_dims, Some(vec![3, 5]));
    assert_eq!(
        decoded.chunk_index_type,
        Some(hdf5_pure_rust::format::messages::data_layout::ChunkIndexType::Implicit)
    );
    assert_eq!(decoded.chunk_index_addr, Some(0x8877665544332211));
}

#[test]
fn test_layout_rejects_unknown_chunk_index_type_as_invalid_format() {
    let mut layout = vec![4u8, 2, 0, 1, 4];
    layout.extend_from_slice(&4u32.to_le_bytes());
    layout.push(99);
    layout.extend_from_slice(&0u64.to_le_bytes());

    let err = DataLayoutMessage::decode(&layout, 8, 8)
        .expect_err("unknown v4 chunk index type must fail as invalid format");
    assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
    assert!(format!("{err}").contains("invalid chunk index type"));
}

#[test]
fn test_layout_rejects_zero_btree2_split_percent() {
    let mut bad_split = vec![4u8, 2, 0, 1, 4];
    bad_split.extend_from_slice(&4u32.to_le_bytes());
    bad_split.push(5);
    bad_split.extend_from_slice(&512u32.to_le_bytes());
    bad_split.push(0);
    bad_split.push(40);
    bad_split.extend_from_slice(&0u64.to_le_bytes());
    let err = DataLayoutMessage::decode(&bad_split, 8, 8).expect_err("split percent 0 must fail");
    assert!(format!("{err}").contains("split percent"));
}

#[test]
fn test_compound_distinct_non_utf8_member_names_do_not_false_duplicate() {
    let mut data = vec![0x36, 2, 0, 0];
    data.extend_from_slice(&2u32.to_le_bytes());

    data.extend_from_slice(&[0xff, 0]);
    data.push(0);
    data.push(0x10);
    data.extend_from_slice(&[0, 0, 0]);
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&0u16.to_le_bytes());
    data.extend_from_slice(&8u16.to_le_bytes());

    data.extend_from_slice(&[0xfe, 0]);
    data.push(1);
    data.push(0x10);
    data.extend_from_slice(&[0, 0, 0]);
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&0u16.to_le_bytes());
    data.extend_from_slice(&8u16.to_le_bytes());

    let datatype = DatatypeMessage::decode(&data).expect("compound datatype should parse");
    let fields = datatype
        .compound_fields_iter()
        .and_then(|fields| fields.collect::<Result<Vec<_>, _>>())
        .expect("distinct non-UTF8 names must not be treated as duplicates");
    assert_eq!(fields.len(), 2);
    assert_eq!(fields[0].byte_offset, 0);
    assert_eq!(fields[1].byte_offset, 1);
}

#[test]
fn test_enum_rejects_empty_member_name() {
    let mut data = vec![0x18, 1, 0, 0];
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&[0x10, 0, 0, 0]);
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&0u16.to_le_bytes());
    data.extend_from_slice(&8u16.to_le_bytes());
    data.push(0);
    data.push(7);

    let datatype = DatatypeMessage::decode(&data).expect("enum datatype should decode");
    let err = datatype
        .enum_members_iter()
        .and_then(|members| members.collect::<Result<Vec<_>, _>>())
        .expect_err("empty enum member names must be rejected");
    assert!(format!("{err}").contains("must not be empty"));
}

#[test]
fn test_unsupported_filters_fail_explicitly() {
    let szip_pipeline = FilterPipelineMessage {
        version: 2,
        filters: vec![FilterDesc {
            id: FILTER_SZIP,
            name: None,
            flags: 0,
            client_data: Vec::new(),
        }],
    };
    let err = hdf5_pure_rust::filters::apply_pipeline_reverse(&[1, 2, 3, 4], &szip_pipeline, 4)
        .expect_err("SZip should return an unsupported error");
    assert_eq!(
        err.to_string(),
        "Unsupported: SZip decompression not available in pure-Rust mode. Re-save the dataset with deflate compression, or use the C HDF5 library."
    );

    let unknown_pipeline = FilterPipelineMessage {
        version: 2,
        filters: vec![FilterDesc {
            id: 65535,
            name: None,
            flags: 0,
            client_data: Vec::new(),
        }],
    };
    let err = hdf5_pure_rust::filters::apply_pipeline_reverse(&[1, 2, 3, 4], &unknown_pipeline, 4)
        .expect_err("unknown filter should return an unsupported error");
    assert!(
        err.to_string().contains("filter 65535 not implemented"),
        "unexpected error: {err}"
    );
}

#[cfg(not(feature = "blosc"))]
#[test]
fn test_blosc_filter_fails_explicitly_without_feature() {
    let pipeline = FilterPipelineMessage {
        version: 2,
        filters: vec![FilterDesc {
            id: 32_001,
            name: None,
            flags: 0,
            client_data: Vec::new(),
        }],
    };
    let err = hdf5_pure_rust::filters::apply_pipeline_reverse(&[1, 2, 3, 4], &pipeline, 4)
        .expect_err("blosc filter should require the blosc feature");
    match err {
        hdf5_pure_rust::Error::Unsupported(message) => {
            assert!(message.contains("Blosc decompression requires the 'blosc' feature"));
        }
        other => panic!("expected unsupported Blosc error, got {other:?}"),
    }
}

#[test]
fn test_branchable_error_messages_are_stable() {
    let pipeline = FilterPipelineMessage {
        version: 2,
        filters: vec![FilterDesc {
            id: 65535,
            name: None,
            flags: 0,
            client_data: Vec::new(),
        }],
    };
    let err = hdf5_pure_rust::filters::apply_pipeline_reverse(&[1, 2, 3, 4], &pipeline, 4)
        .expect_err("unknown required filter should fail");
    match err {
        hdf5_pure_rust::Error::Unsupported(message) => {
            assert!(message.contains("filter 65535 not implemented"));
        }
        other => panic!("expected Unsupported, got {other:?}"),
    }

    let err = hdf5_pure_rust::filters::apply_pipeline_reverse(
        &[1, 2, 3, 4],
        &FilterPipelineMessage {
            version: 2,
            filters: vec![FilterDesc {
                id: 65535,
                name: None,
                flags: 1,
                client_data: Vec::new(),
            }],
        },
        4,
    )
    .expect_err("unknown optional filter should fail on read when not masked out");
    assert!(matches!(err, hdf5_pure_rust::Error::Unsupported(_)));

    let err = hdf5_pure_rust::filters::apply_pipeline_reverse_with_mask(
        &[1, 2, 3, 4],
        &FilterPipelineMessage {
            version: 2,
            filters: vec![FilterDesc {
                id: FILTER_DEFLATE,
                name: None,
                flags: 0,
                client_data: Vec::new(),
            }],
        },
        4,
        0b10,
    )
    .expect_err("invalid filter mask should fail");
    match err {
        hdf5_pure_rust::Error::InvalidFormat(message) => {
            assert!(message.contains("references filters outside pipeline length"));
        }
        other => panic!("expected InvalidFormat, got {other:?}"),
    }
}

#[test]
fn test_datatype_aware_filters_reject_missing_parameters() {
    for id in [FILTER_NBIT, FILTER_SCALEOFFSET] {
        let pipeline = FilterPipelineMessage {
            version: 2,
            filters: vec![FilterDesc {
                id,
                name: None,
                flags: 0,
                client_data: Vec::new(),
            }],
        };
        let err = hdf5_pure_rust::filters::apply_pipeline_reverse(&[1, 2, 3, 4], &pipeline, 4)
            .expect_err("datatype-aware filter should reject missing parameters");
        assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
    }
}

#[test]
fn test_nbit_top_level_nooptype_filter_copies_payload() {
    let pipeline = FilterPipelineMessage {
        version: 2,
        filters: vec![FilterDesc {
            id: FILTER_NBIT,
            name: Some("nbit".into()),
            flags: 0,
            client_data: vec![5, 0, 2, 4, 2],
        }],
    };
    let decoded =
        hdf5_pure_rust::filters::apply_pipeline_reverse(&[0x12, 0x34, 0xab, 0xcd], &pipeline, 2)
            .expect("top-level NBit NOOPTYPE should copy each element");
    assert_eq!(&*decoded, &[0x12, 0x34, 0xab, 0xcd]);
}

#[test]
fn test_scaleoffset_constant_integer_filter_expands_minimum_value() {
    let pipeline = FilterPipelineMessage {
        version: 2,
        filters: vec![FilterDesc {
            id: FILTER_SCALEOFFSET,
            name: Some("scaleoffset".into()),
            flags: 0,
            client_data: vec![2, 0, 3, 0, 1, 0, 0],
        }],
    };
    let mut chunk = vec![0u8; 21];
    chunk[4] = 1;
    chunk[5] = 42;

    let decoded = hdf5_pure_rust::filters::apply_pipeline_reverse(&chunk, &pipeline, 1)
        .expect("constant ScaleOffset integer chunks should expand from the minimum value");
    assert_eq!(&*decoded, &[42, 42, 42]);
}

#[test]
fn test_filter_pipeline_rejects_out_of_range_filter_mask() {
    let pipeline = FilterPipelineMessage {
        version: 2,
        filters: vec![FilterDesc {
            id: FILTER_DEFLATE,
            name: None,
            flags: 0,
            client_data: Vec::new(),
        }],
    };
    let err = hdf5_pure_rust::filters::apply_pipeline_reverse_with_mask(
        &[1, 2, 3, 4],
        &pipeline,
        4,
        0b10,
    )
    .expect_err("out-of-range filter mask should return an error");
    assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
}

#[test]
fn test_filter_pipeline_rejects_more_than_32_filters() {
    let filter = FilterDesc {
        id: FILTER_DEFLATE,
        name: None,
        flags: 0,
        client_data: Vec::new(),
    };
    let pipeline = FilterPipelineMessage {
        version: 2,
        filters: vec![filter; 33],
    };
    let err =
        hdf5_pure_rust::filters::apply_pipeline_reverse_with_mask(&[1, 2, 3, 4], &pipeline, 4, 0)
            .expect_err("pipeline longer than the 32-bit filter mask should return an error");
    assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
}

#[test]
fn test_virtual_layout_parses_as_metadata_only() {
    let mut data = Vec::new();
    data.push(4); // layout message version
    data.push(3); // virtual layout class
    data.extend_from_slice(&0x1234u64.to_le_bytes());
    data.extend_from_slice(&7u32.to_le_bytes());

    let layout = DataLayoutMessage::decode(&data, 8, 8).unwrap();
    assert_eq!(
        layout.layout_class,
        hdf5_pure_rust::format::messages::data_layout::LayoutClass::Virtual
    );
    assert_eq!(layout.virtual_heap_addr, Some(0x1234));
    assert_eq!(layout.virtual_heap_index, Some(7));
}

#[test]
fn test_huge_fractal_heap_indirect_object_is_unsupported() {
    let heap = test_fractal_heap(0);
    let mut reader = HdfReader::new(Cursor::new(Vec::<u8>::new()));
    let err = heap
        .read_managed_object(&mut reader, &[0x10])
        .expect_err("indirect huge fractal-heap IDs should fail explicitly");
    assert!(matches!(
        err,
        hdf5_pure_rust::Error::Unsupported(_) | hdf5_pure_rust::Error::InvalidFormat(_)
    ));
}

#[test]
fn test_huge_fractal_heap_direct_object_read() {
    let heap = test_fractal_heap(0);
    let payload = b"huge object".to_vec();
    let mut file_bytes = vec![0u8; 32];
    file_bytes.extend_from_slice(&payload);
    let addr = 32u64;
    let len = payload.len() as u64;

    let mut id = vec![0x10];
    id.extend_from_slice(&addr.to_le_bytes());
    id.extend_from_slice(&len.to_le_bytes());

    let mut reader = HdfReader::new(Cursor::new(file_bytes));
    let read = heap.read_managed_object(&mut reader, &id).unwrap();
    assert_eq!(read, payload);
}

#[test]
fn test_filtered_huge_fractal_heap_direct_object_read() {
    let mut heap = test_fractal_heap(8);
    let payload = b"filtered huge heap object".to_vec();
    let mut filtered = Vec::new();
    hdf5_pure_rust::filters::deflate::compress_into(&payload, 6, &mut filtered).unwrap();
    let mut file_bytes = vec![0u8; 48];
    file_bytes.extend_from_slice(&filtered);
    let addr = 48u64;

    heap.heap_id_len = 1 + 8 + 8 + 4 + 8;
    heap.filter_pipeline = Some(FilterPipelineMessage {
        version: 2,
        filters: vec![FilterDesc {
            id: FILTER_DEFLATE,
            name: None,
            flags: 0,
            client_data: vec![6],
        }],
    });

    let mut id = vec![0x10];
    id.extend_from_slice(&addr.to_le_bytes());
    id.extend_from_slice(&(filtered.len() as u64).to_le_bytes());
    id.extend_from_slice(&0u32.to_le_bytes());
    id.extend_from_slice(&(payload.len() as u64).to_le_bytes());

    let mut reader = HdfReader::new(Cursor::new(file_bytes));
    let read = heap.read_managed_object(&mut reader, &id).unwrap();
    assert_eq!(read, payload);
}

#[test]
fn test_filtered_fractal_heap_direct_object_read() {
    let heap = test_fractal_heap(8);
    let payload = b"filtered heap object".to_vec();
    let mut filtered = Vec::new();
    hdf5_pure_rust::filters::deflate::compress_into(&payload, 6, &mut filtered).unwrap();
    let mut file_bytes = vec![0u8; 64];
    file_bytes.extend_from_slice(&filtered);

    let mut heap = heap;
    heap.root_block_addr = 64;
    heap.root_direct_filtered_size = Some(filtered.len() as u64);
    heap.filter_pipeline = Some(FilterPipelineMessage {
        version: 2,
        filters: vec![FilterDesc {
            id: FILTER_DEFLATE,
            name: None,
            flags: 0,
            client_data: vec![6],
        }],
    });

    let mut id = vec![0x00];
    id.extend_from_slice(&0u32.to_le_bytes());
    id.extend_from_slice(&(payload.len() as u64).to_le_bytes());

    let mut reader = HdfReader::new(Cursor::new(file_bytes));
    let read = heap.read_managed_object(&mut reader, &id).unwrap();
    assert_eq!(read, payload);
}

#[test]
fn test_filtered_fractal_heap_direct_object_read_fletcher32() {
    let heap = test_fractal_heap(8);
    let payload = b"fletcher filtered heap object".to_vec();
    let mut filtered = payload.clone();
    filtered.extend_from_slice(&hdf5_fletcher32(&payload).to_le_bytes());
    let mut file_bytes = vec![0u8; 64];
    file_bytes.extend_from_slice(&filtered);

    let mut heap = heap;
    heap.root_block_addr = 64;
    heap.root_direct_filtered_size = Some(filtered.len() as u64);
    heap.filter_pipeline = Some(FilterPipelineMessage {
        version: 2,
        filters: vec![FilterDesc {
            id: FILTER_FLETCHER32,
            name: None,
            flags: 0,
            client_data: Vec::new(),
        }],
    });

    let mut id = vec![0x00];
    id.extend_from_slice(&0u32.to_le_bytes());
    id.extend_from_slice(&(payload.len() as u64).to_le_bytes());

    let mut reader = HdfReader::new(Cursor::new(file_bytes));
    let read = heap.read_managed_object(&mut reader, &id).unwrap();
    assert_eq!(read, payload);
}

fn hdf5_fletcher32(data: &[u8]) -> u32 {
    let mut sum1: u32 = 0;
    let mut sum2: u32 = 0;
    let mut pos = 0;
    let mut remaining = data.len() / 2;

    while remaining > 0 {
        let tlen = remaining.min(360);
        remaining -= tlen;

        for _ in 0..tlen {
            let value = ((data[pos] as u32) << 8) | data[pos + 1] as u32;
            sum1 += value;
            sum2 += sum1;
            pos += 2;
        }

        sum1 = (sum1 & 0xffff) + (sum1 >> 16);
        sum2 = (sum2 & 0xffff) + (sum2 >> 16);
    }

    if data.len() % 2 != 0 {
        sum1 += (data[pos] as u32) << 8;
        sum2 += sum1;
        sum1 = (sum1 & 0xffff) + (sum1 >> 16);
        sum2 = (sum2 & 0xffff) + (sum2 >> 16);
    }

    sum1 = (sum1 & 0xffff) + (sum1 >> 16);
    sum2 = (sum2 & 0xffff) + (sum2 >> 16);

    (sum2 << 16) | sum1
}

fn test_fractal_heap(io_filter_len: u16) -> FractalHeapHeader {
    FractalHeapHeader {
        heap_addr: 0,
        heap_id_len: 8,
        io_filter_len,
        flags: 0,
        max_managed_obj_size: 1024,
        table_width: 4,
        start_block_size: 512,
        max_direct_block_size: 4096,
        max_heap_size: 32,
        start_root_rows: 1,
        root_block_addr: 0,
        current_root_rows: 0,
        num_managed_objects: 0,
        has_checksum: false,
        sizeof_addr: 8,
        sizeof_size: 8,
        huge_btree_addr: hdf5_pure_rust::io::reader::UNDEF_ADDR,
        root_direct_filtered_size: None,
        root_direct_filter_mask: 0,
        filter_pipeline: None,
    }
}

#[test]
fn test_attribute_empty() {
    assert!(AttributeMessage::decode(&[]).is_err());
}

#[test]
fn test_attribute_rejects_truncated_metadata_sections() {
    for data in [
        vec![1, 0, 1, 0, 8, 0, 4, 0],
        vec![2, 0, 4, 0, 8, 0, 4, 0, b'a'],
        vec![2, 0, 1, 0, 8, 0, 4, 0, b'a', 0x10, 0, 0],
        vec![3, 0, 1, 0, 8, 0, 4, 0],
        vec![3, 0, 1, 0, 8, 0, 4, 0, 0, b'a', 0x10],
    ] {
        let err =
            AttributeMessage::decode(&data).expect_err("truncated attribute message should fail");
        assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
    }
}

#[test]
fn test_attribute_bad_version() {
    let data = vec![99, 0, 0, 0, 0, 0, 0, 0]; // version 99
    assert!(AttributeMessage::decode(&data).is_err());
}

#[test]
fn test_attribute_rejects_bad_flags_and_corrupt_name_length() {
    let bad_v2_flags = vec![2, 0x80, 2, 0, 8, 0, 4, 0];
    let err = AttributeMessage::decode(&bad_v2_flags).unwrap_err();
    assert!(format!("{err}").contains("flags"));

    let bad_v3_flags = vec![3, 0x80, 2, 0, 8, 0, 4, 0, 0];
    let err = AttributeMessage::decode(&bad_v3_flags).unwrap_err();
    assert!(format!("{err}").contains("flags"));

    let bad_v3_encoding = vec![3, 0, 2, 0, 8, 0, 4, 0, 2];
    let err = AttributeMessage::decode(&bad_v3_encoding).unwrap_err();
    assert!(format!("{err}").contains("encoding"));

    let mut name_size_one = vec![1u8, 0];
    name_size_one.extend_from_slice(&1u16.to_le_bytes());
    name_size_one.extend_from_slice(&8u16.to_le_bytes());
    name_size_one.extend_from_slice(&8u16.to_le_bytes());
    name_size_one.push(0);
    let err = AttributeMessage::decode(&name_size_one).unwrap_err();
    assert!(format!("{err}").contains("name length"));

    let mut embedded_null = vec![2u8, 0];
    embedded_null.extend_from_slice(&4u16.to_le_bytes());
    embedded_null.extend_from_slice(&8u16.to_le_bytes());
    embedded_null.extend_from_slice(&8u16.to_le_bytes());
    embedded_null.extend_from_slice(b"a\0b\0");
    let err = AttributeMessage::decode(&embedded_null).unwrap_err();
    assert!(format!("{err}").contains("different length"));

    let mut missing_null = vec![2u8, 0];
    missing_null.extend_from_slice(&2u16.to_le_bytes());
    missing_null.extend_from_slice(&8u16.to_le_bytes());
    missing_null.extend_from_slice(&8u16.to_le_bytes());
    missing_null.extend_from_slice(b"ab");
    let err = AttributeMessage::decode(&missing_null).unwrap_err();
    assert!(format!("{err}").contains("different length"));

    let mut invalid_utf8 = vec![2u8, 0];
    invalid_utf8.extend_from_slice(&2u16.to_le_bytes());
    invalid_utf8.extend_from_slice(&8u16.to_le_bytes());
    invalid_utf8.extend_from_slice(&8u16.to_le_bytes());
    invalid_utf8.extend_from_slice(&[0xff, 0]);
    let err = AttributeMessage::decode(&invalid_utf8).unwrap_err();
    assert!(format!("{err}").contains("not UTF-8"));
}

#[test]
fn test_attribute_rejects_truncated_value_payload() {
    let datatype = vec![0x10, 0, 0, 0, 4, 0, 0, 0, 0, 0, 32, 0];
    let dataspace = vec![2, 0, 0, 0];
    let mut attr = vec![3, 0];
    attr.extend_from_slice(&2u16.to_le_bytes());
    attr.extend_from_slice(&(datatype.len() as u16).to_le_bytes());
    attr.extend_from_slice(&(dataspace.len() as u16).to_le_bytes());
    attr.push(0);
    attr.extend_from_slice(b"x\0");
    attr.extend_from_slice(&datatype);
    attr.extend_from_slice(&dataspace);
    attr.extend_from_slice(&[1, 2, 3]);

    let err = AttributeMessage::decode(&attr)
        .expect_err("attribute value payload shorter than datatype size should fail");
    assert!(
        format!("{err}").contains("attribute data is truncated"),
        "expected attribute data truncation error, got: {err}"
    );
}

#[test]
fn test_open_nonexistent_file() {
    let result = hdf5_pure_rust::File::open("/nonexistent/path.h5");
    assert!(result.is_err());
}

#[test]
fn test_open_non_hdf5_file() {
    // Create a temp file with random data
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("not_hdf5.bin");
    std::fs::write(&path, b"This is not an HDF5 file").unwrap();
    let result = hdf5_pure_rust::File::open(&path);
    assert!(result.is_err());
}

/// Build a syntactically valid v2 B-tree header with specified split/merge
/// percent bytes, including a correct metadata checksum so that the
/// validation under test fires on the percent fields, not the checksum.
fn btree_v2_header_bytes(split_pct: u8, merge_pct: u8) -> Vec<u8> {
    let mut buf = Vec::with_capacity(28);
    buf.extend_from_slice(b"BTHD"); // magic
    buf.push(0); // version
    buf.push(5); // tree_type (chunked dataset, no filter)
    buf.extend_from_slice(&512u32.to_le_bytes()); // node_size
    buf.extend_from_slice(&8u16.to_le_bytes()); // record_size
    buf.extend_from_slice(&0u16.to_le_bytes()); // depth
    buf.push(split_pct);
    buf.push(merge_pct);
    buf.extend_from_slice(&0u64.to_le_bytes()); // root_addr (sizeof_addr=8)
    buf.extend_from_slice(&0u16.to_le_bytes()); // root_nrecords
    buf.extend_from_slice(&0u64.to_le_bytes()); // total_records (sizeof_size=8)
    let checksum = checksum_metadata(&buf);
    buf.extend_from_slice(&checksum.to_le_bytes());
    buf
}

fn btree_v2_header_bytes_with_sizes(node_size: u32, record_size: u16) -> Vec<u8> {
    let mut buf = Vec::with_capacity(28);
    buf.extend_from_slice(b"BTHD"); // magic
    buf.push(0); // version
    buf.push(5); // tree_type (chunked dataset, no filter)
    buf.extend_from_slice(&node_size.to_le_bytes());
    buf.extend_from_slice(&record_size.to_le_bytes());
    buf.extend_from_slice(&0u16.to_le_bytes()); // depth
    buf.push(100); // split percent
    buf.push(40); // merge percent
    buf.extend_from_slice(&0u64.to_le_bytes()); // root_addr (sizeof_addr=8)
    buf.extend_from_slice(&0u16.to_le_bytes()); // root_nrecords
    buf.extend_from_slice(&0u64.to_le_bytes()); // total_records (sizeof_size=8)
    let checksum = checksum_metadata(&buf);
    buf.extend_from_slice(&checksum.to_le_bytes());
    buf
}

fn btree_v2_header_bytes_with_root(
    root_addr: u64,
    root_nrecords: u16,
    total_records: u64,
) -> Vec<u8> {
    let mut buf = Vec::with_capacity(28);
    buf.extend_from_slice(b"BTHD"); // magic
    buf.push(0); // version
    buf.push(5); // tree_type (chunked dataset, no filter)
    buf.extend_from_slice(&512u32.to_le_bytes());
    buf.extend_from_slice(&8u16.to_le_bytes());
    buf.extend_from_slice(&0u16.to_le_bytes()); // depth
    buf.push(100); // split percent
    buf.push(40); // merge percent
    buf.extend_from_slice(&root_addr.to_le_bytes());
    buf.extend_from_slice(&root_nrecords.to_le_bytes());
    buf.extend_from_slice(&total_records.to_le_bytes());
    let checksum = checksum_metadata(&buf);
    buf.extend_from_slice(&checksum.to_le_bytes());
    buf
}

#[test]
fn test_btree_v2_rejects_split_percent_over_100() {
    let bytes = btree_v2_header_bytes(101, 40);
    let mut reader = HdfReader::new(Cursor::new(bytes));
    let err = BTreeV2Header::read_at(&mut reader, 0).unwrap_err();
    assert!(
        format!("{err}").contains("split percent"),
        "expected split-percent error, got: {err}"
    );
}

#[test]
fn test_btree_v2_rejects_merge_percent_over_100() {
    let bytes = btree_v2_header_bytes(100, 200);
    let mut reader = HdfReader::new(Cursor::new(bytes));
    let err = BTreeV2Header::read_at(&mut reader, 0).unwrap_err();
    assert!(
        format!("{err}").contains("merge percent"),
        "expected merge-percent error, got: {err}"
    );
}

#[test]
fn test_btree_v2_rejects_zero_split_percent() {
    let bytes = btree_v2_header_bytes(0, 40);
    let mut reader = HdfReader::new(Cursor::new(bytes));
    let err = BTreeV2Header::read_at(&mut reader, 0).unwrap_err();
    assert!(
        format!("{err}").contains("split percent"),
        "expected split-percent error, got: {err}"
    );
}

#[test]
fn test_btree_v2_rejects_zero_merge_percent() {
    let bytes = btree_v2_header_bytes(100, 0);
    let mut reader = HdfReader::new(Cursor::new(bytes));
    let err = BTreeV2Header::read_at(&mut reader, 0).unwrap_err();
    assert!(
        format!("{err}").contains("merge percent"),
        "expected merge-percent error, got: {err}"
    );
}

#[test]
fn test_btree_v2_rejects_invalid_node_or_record_size() {
    for bytes in [
        btree_v2_header_bytes_with_sizes(10, 8),
        btree_v2_header_bytes_with_sizes(512, 0),
    ] {
        let mut reader = HdfReader::new(Cursor::new(bytes));
        let err = BTreeV2Header::read_at(&mut reader, 0)
            .expect_err("invalid v2 B-tree sizing should fail");
        assert!(matches!(err, hdf5_pure_rust::Error::InvalidFormat(_)));
    }
}

#[test]
fn test_btree_v2_rejects_nonempty_tree_with_undefined_root() {
    let bytes = btree_v2_header_bytes_with_root(UNDEF_ADDR, 1, 1);
    let mut reader = HdfReader::new(Cursor::new(bytes));
    let err = BTreeV2Header::read_at(&mut reader, 0)
        .expect_err("non-empty v2 B-tree with undefined root should fail");
    assert!(
        format!("{err}").contains("root address"),
        "expected root-address error, got: {err}"
    );
}

#[test]
fn test_btree_v2_accepts_equal_nonzero_percents() {
    let bytes = btree_v2_header_bytes(60, 60);
    let mut reader = HdfReader::new(Cursor::new(bytes));
    let hdr = BTreeV2Header::read_at(&mut reader, 0).expect("equal nonzero percents must parse");
    assert_eq!(hdr.split_pct, 60);
    assert_eq!(hdr.merge_pct, 60);
}

#[test]
fn test_btree_v2_accepts_canonical_percents() {
    // 100 / 40 are HDF5's defaults; must still parse.
    let bytes = btree_v2_header_bytes(100, 40);
    let mut reader = HdfReader::new(Cursor::new(bytes));
    let hdr = BTreeV2Header::read_at(&mut reader, 0).expect("canonical 100/40 must parse");
    assert_eq!(hdr.split_pct, 100);
    assert_eq!(hdr.merge_pct, 40);
}

/// Build a minimal v1 FixedPoint datatype message with explicit bit_offset
/// and bit_precision fields (each u16 little-endian).
fn fixed_point_datatype_bytes(size_bytes: u32, bit_offset: u16, precision: u16) -> Vec<u8> {
    let mut buf = Vec::with_capacity(12);
    buf.push(0x10); // version=1, class=0 (FixedPoint)
    buf.extend_from_slice(&[0, 0, 0]); // class_bits
    buf.extend_from_slice(&size_bytes.to_le_bytes());
    buf.extend_from_slice(&bit_offset.to_le_bytes());
    buf.extend_from_slice(&precision.to_le_bytes());
    buf
}

fn time_datatype_bytes(size_bytes: u32, big_endian: bool, precision: u16) -> Vec<u8> {
    let mut buf = Vec::with_capacity(10);
    buf.push(0x12); // version=1, class=2 (Time)
    buf.push(u8::from(big_endian));
    buf.extend_from_slice(&[0, 0]);
    buf.extend_from_slice(&size_bytes.to_le_bytes());
    buf.extend_from_slice(&precision.to_le_bytes());
    buf
}

#[test]
fn test_datatype_rejects_zero_precision() {
    let bytes = fixed_point_datatype_bytes(4, 0, 0);
    let err = DatatypeMessage::decode(&bytes).unwrap_err();
    assert!(
        format!("{err}").contains("precision is zero"),
        "expected precision-is-zero error, got: {err}"
    );
}

#[test]
fn test_datatype_rejects_bit_offset_over_size() {
    // size = 2 bytes = 16 bits; offset of 17 is out of bounds.
    let bytes = fixed_point_datatype_bytes(2, 17, 1);
    let err = DatatypeMessage::decode(&bytes).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("bit offset"),
        "expected bit-offset error, got: {msg}"
    );
}

#[test]
fn test_datatype_rejects_offset_plus_precision_over_size() {
    // size = 4 bytes = 32 bits; offset+precision = 16+24 = 40 > 32.
    let bytes = fixed_point_datatype_bytes(4, 16, 24);
    let err = DatatypeMessage::decode(&bytes).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("bit offset+precision"),
        "expected offset+precision error, got: {msg}"
    );
}

#[test]
fn test_datatype_accepts_canonical_integer_widths() {
    // i8/i16/i32/i64 with offset=0 and precision = size*8 must all parse.
    for size_bytes in [1u32, 2, 4, 8] {
        let bytes = fixed_point_datatype_bytes(size_bytes, 0, (size_bytes * 8) as u16);
        let dt = DatatypeMessage::decode(&bytes)
            .unwrap_or_else(|e| panic!("size={size_bytes} should parse: {e}"));
        assert_eq!(dt.size, size_bytes);
    }
}

#[test]
fn test_time_datatype_validates_precision_and_byte_order_flags() {
    let le = DatatypeMessage::decode(&time_datatype_bytes(4, false, 32)).unwrap();
    assert_eq!(le.class, DatatypeClass::Time);
    assert_eq!(le.byte_order(), Some(ByteOrder::LittleEndian));

    let be = DatatypeMessage::decode(&time_datatype_bytes(8, true, 64)).unwrap();
    assert_eq!(be.byte_order(), Some(ByteOrder::BigEndian));

    let err = DatatypeMessage::decode(&time_datatype_bytes(4, false, 0)).unwrap_err();
    assert!(
        format!("{err}").contains("time datatype precision"),
        "expected time precision error, got: {err}"
    );

    let err = DatatypeMessage::decode(&time_datatype_bytes(4, false, 33)).unwrap_err();
    assert!(
        format!("{err}").contains("time datatype precision"),
        "expected time precision error, got: {err}"
    );

    let mut unsupported_flags = time_datatype_bytes(4, false, 32);
    unsupported_flags[1] = 0x02;
    let err = DatatypeMessage::decode(&unsupported_flags).unwrap_err();
    assert!(
        format!("{err}").contains("time datatype has unsupported class flags"),
        "expected time flag error, got: {err}"
    );

    let truncated = time_datatype_bytes(4, false, 32);
    let err = DatatypeMessage::decode(&truncated[..9]).unwrap_err();
    assert!(
        format!("{err}").contains("time datatype precision")
            || format!("{err}").contains("properties are truncated"),
        "expected time truncation error, got: {err}"
    );
}

#[test]
fn test_datatype_accepts_subbyte_field_within_size() {
    // 4-byte payload, 12 bits of meaningful precision starting at bit 4 — well-formed.
    let bytes = fixed_point_datatype_bytes(4, 4, 12);
    let dt = DatatypeMessage::decode(&bytes).expect("sub-field within size must parse");
    assert_eq!(dt.size, 4);
}

#[test]
fn test_datatype_bitfield_validates_precision_too() {
    // BitField shares the layout — same checks apply.
    let mut bytes = fixed_point_datatype_bytes(2, 0, 0);
    bytes[0] = 0x14; // version=1, class=4 (BitField)
    let err = DatatypeMessage::decode(&bytes).unwrap_err();
    assert!(
        format!("{err}").contains("precision is zero"),
        "expected BitField precision-is-zero error, got: {err}"
    );
}

/// Build a v1 FloatingPoint datatype message with explicit field positions.
/// Class bits layout: byte 0 default 0; byte 1 = sign location.
fn float_datatype_bytes(
    size_bytes: u32,
    sign_loc: u8,
    bit_offset: u16,
    precision: u16,
    exp_loc: u8,
    exp_size: u8,
    mant_loc: u8,
    mant_size: u8,
) -> Vec<u8> {
    let mut buf = Vec::with_capacity(20);
    buf.push(0x11); // version=1, class=1 (FloatingPoint)
    buf.push(0); // class_bits[0]
    buf.push(sign_loc); // class_bits[1] = sign location
    buf.push(0); // class_bits[2]
    buf.extend_from_slice(&size_bytes.to_le_bytes());
    buf.extend_from_slice(&bit_offset.to_le_bytes());
    buf.extend_from_slice(&precision.to_le_bytes());
    buf.push(exp_loc);
    buf.push(exp_size);
    buf.push(mant_loc);
    buf.push(mant_size);
    buf.extend_from_slice(&127u32.to_le_bytes()); // exp_bias (irrelevant)
    buf
}

#[test]
fn test_float_datatype_accepts_canonical_f32_layout() {
    // IEEE 754 binary32: size=4, sign@31, exp@23 (8 bits), mant@0 (23 bits),
    // precision=32. Must parse cleanly.
    let bytes = float_datatype_bytes(4, 31, 0, 32, 23, 8, 0, 23);
    let dt = DatatypeMessage::decode(&bytes).expect("canonical f32 must parse");
    assert_eq!(dt.size, 4);
}

#[test]
fn test_float_datatype_rejects_invalid_mantissa_normalization() {
    let mut bytes = float_datatype_bytes(4, 31, 0, 32, 23, 8, 0, 23);
    bytes[1] = 0x30; // class_bits[0] normalization bits set to reserved code 3.
    let err = DatatypeMessage::decode(&bytes).unwrap_err();
    assert!(
        format!("{err}").contains("normalization"),
        "expected normalization error, got: {err}"
    );
}

#[test]
fn test_float_datatype_rejects_zero_exp_size() {
    let bytes = float_datatype_bytes(4, 31, 0, 32, 0, 0, 0, 23);
    let err = DatatypeMessage::decode(&bytes).unwrap_err();
    assert!(
        format!("{err}").contains("exponent size is zero"),
        "got: {err}"
    );
}

#[test]
fn test_float_datatype_rejects_zero_mant_size() {
    let bytes = float_datatype_bytes(4, 31, 0, 32, 23, 8, 0, 0);
    let err = DatatypeMessage::decode(&bytes).unwrap_err();
    assert!(
        format!("{err}").contains("mantissa size is zero"),
        "got: {err}"
    );
}

#[test]
fn test_float_datatype_rejects_sign_outside_precision() {
    // sign_loc must be < precision.
    let bytes = float_datatype_bytes(4, 32, 0, 32, 23, 8, 0, 23);
    let err = DatatypeMessage::decode(&bytes).unwrap_err();
    assert!(format!("{err}").contains("sign bit position"), "got: {err}");
}

#[test]
fn test_float_datatype_rejects_exp_overflow_precision() {
    // exp_loc + exp_size > precision.
    let bytes = float_datatype_bytes(4, 31, 0, 32, 25, 8, 0, 23);
    let err = DatatypeMessage::decode(&bytes).unwrap_err();
    assert!(
        format!("{err}").contains("exponent location+size"),
        "got: {err}"
    );
}

#[test]
fn test_float_datatype_rejects_mant_overflow_precision() {
    // mant_loc + mant_size > precision.
    let bytes = float_datatype_bytes(4, 31, 0, 32, 23, 8, 23, 10);
    let err = DatatypeMessage::decode(&bytes).unwrap_err();
    assert!(
        format!("{err}").contains("mantissa location+size"),
        "got: {err}"
    );
}

#[test]
fn test_dataspace_v2_rejects_scalar_with_nonzero_rank() {
    // Build a v2 dataspace header with space_type=Scalar (0) and ndims=3.
    // Bytes: version(1)=2, ndims(1)=3, flags(1)=0, space_type(1)=0,
    // followed by three u64 dims so the truncation check doesn't fire first.
    let mut bytes = vec![2u8, 3, 0, 0];
    bytes.extend_from_slice(&[0u8; 24]); // 3 dims of u64 = 24 bytes
    let err = DataspaceMessage::decode(&bytes).unwrap_err();
    assert!(
        format!("{err}").contains("Scalar") && format!("{err}").contains("rank 3"),
        "expected scalar-rank error, got: {err}"
    );
}

#[test]
fn test_dataspace_v2_rejects_null_with_nonzero_rank() {
    let mut bytes = vec![2u8, 1, 0, 2];
    bytes.extend_from_slice(&[0u8; 8]);
    let err = DataspaceMessage::decode(&bytes).unwrap_err();
    assert!(
        format!("{err}").contains("Null") && format!("{err}").contains("rank 1"),
        "expected null-rank error, got: {err}"
    );
}

#[test]
fn test_dataspace_v2_rejects_simple_with_zero_rank() {
    let bytes = vec![2u8, 0, 0, 1];
    let err = DataspaceMessage::decode(&bytes).unwrap_err();
    assert!(
        format!("{err}").contains("nonzero rank"),
        "expected simple-rank error, got: {err}"
    );
}

#[test]
fn test_dataspace_v2_accepts_scalar_with_zero_rank() {
    // Canonical scalar: ndims=0, space_type=Scalar.
    let bytes = vec![2u8, 0, 0, 0];
    let ds = DataspaceMessage::decode(&bytes).expect("scalar with rank 0 must parse");
    assert_eq!(ds.ndims, 0);
}

#[test]
fn test_attribute_message_rejects_zero_name_length() {
    // v1 attribute with name_size=0 (everything else zeroed) — must reject.
    let mut bytes = vec![1u8, 0]; // version=1, reserved
    bytes.extend_from_slice(&0u16.to_le_bytes()); // name_size=0 (BAD)
    bytes.extend_from_slice(&8u16.to_le_bytes()); // dt_size
    bytes.extend_from_slice(&8u16.to_le_bytes()); // ds_size
    let err = AttributeMessage::decode(&bytes).unwrap_err();
    assert!(format!("{err}").contains("name length"), "got: {err}");
}

#[test]
fn test_layout_v3_rejects_zero_chunk_dimension() {
    // Build a v3 chunked layout with chunk_dims = [4, 0] (the zero is bad)
    // and element_size = 4. Wire format: version(1)=3, class(1)=2 (Chunked),
    // ndims(1)=3 (= 2 chunk dims + 1 element-size dim), addr(8 bytes for
    // sizeof_addr=8), then 3 × u32 dims.
    let mut bytes = vec![3u8, 2, 3];
    bytes.extend_from_slice(&[0u8; 8]); // chunk index addr
    bytes.extend_from_slice(&4u32.to_le_bytes()); // chunk dim 0
    bytes.extend_from_slice(&0u32.to_le_bytes()); // chunk dim 1 = 0 (BAD)
    bytes.extend_from_slice(&4u32.to_le_bytes()); // element size
    let err = DataLayoutMessage::decode(&bytes, 8, 8).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("must be positive"),
        "expected zero-chunk-dim error, got: {msg}"
    );
}

#[test]
fn test_fractal_heap_id_rejects_unsupported_version() {
    let heap = test_fractal_heap(0);
    let mut reader = HdfReader::new(Cursor::new(Vec::<u8>::new()));
    let err = heap
        .read_managed_object(&mut reader, &[0x40])
        .expect_err("unsupported fractal heap ID version should fail");
    assert!(
        format!("{err}").contains("version"),
        "expected version error, got: {err}"
    );
}

#[test]
fn test_fractal_heap_id_rejects_managed_and_huge_reserved_bits() {
    let heap = test_fractal_heap(0);
    for id in [&[0x01u8][..], &[0x11u8][..]] {
        let mut reader = HdfReader::new(Cursor::new(Vec::<u8>::new()));
        let err = heap
            .read_managed_object(&mut reader, id)
            .expect_err("nonzero managed/huge heap ID reserved bits should fail");
        assert!(
            format!("{err}").contains("reserved"),
            "expected reserved-bit error, got: {err}"
        );
    }
}

#[test]
fn test_link_info_rejects_negative_encoded_max_creation_index() {
    use hdf5_pure_rust::format::messages::link_info::LinkInfoMessage;
    // v0 link_info: version(1) + flags(1) | 0x01 (has_max_crt_order)
    // + max_creation_index(8) + heap_addr(8) + name_btree_addr(8)
    let mut bytes = vec![0u8, 0x01];
    bytes.extend_from_slice(&(i64::MAX as u64 + 1).to_le_bytes());
    bytes.extend_from_slice(&[0u8; 8]);
    bytes.extend_from_slice(&[0u8; 8]);
    let err = LinkInfoMessage::decode(&bytes, 8).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("max creation index") && msg.contains("exceeds"),
        "expected oversized-max-creation-index error, got: {msg}"
    );
}

#[test]
fn test_link_info_accepts_max_creation_index_at_limit() {
    use hdf5_pure_rust::format::messages::link_info::LinkInfoMessage;
    let mut bytes = vec![0u8, 0x01];
    bytes.extend_from_slice(&(i64::MAX as u64).to_le_bytes());
    bytes.extend_from_slice(&[0u8; 8]);
    bytes.extend_from_slice(&[0u8; 8]);
    let li = LinkInfoMessage::decode(&bytes, 8).expect("max_creation_index == i64::MAX must parse");
    assert_eq!(li.max_creation_index, Some(i64::MAX as u64));
}

#[test]
fn test_filter_pipeline_v1_rejects_non_multiple_of_eight_name_length() {
    // Build a minimal v1 filter pipeline with a single filter whose
    // declared name_length is 7 — must be rejected.
    let mut buf = Vec::new();
    buf.push(1); // version
    buf.push(1); // nfilters
    buf.extend_from_slice(&[0, 0, 0, 0, 0, 0]); // 6 reserved bytes (header is 8 bytes total)
    buf.extend_from_slice(&1u16.to_le_bytes()); // filter id (deflate)
    buf.extend_from_slice(&7u16.to_le_bytes()); // name_length = 7 (BAD)
    buf.extend_from_slice(&0u16.to_le_bytes()); // flags
    buf.extend_from_slice(&0u16.to_le_bytes()); // cd_nelmts
    let err =
        hdf5_pure_rust::format::messages::filter_pipeline::FilterPipelineMessage::decode(&buf)
            .unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("not a multiple of eight"), "got: {msg}");
}

#[test]
fn test_filter_pipeline_rejects_unterminated_names() {
    let mut v1 = vec![1, 1, 0, 0, 0, 0, 0, 0];
    v1.extend_from_slice(&256u16.to_le_bytes());
    v1.extend_from_slice(&8u16.to_le_bytes());
    v1.extend_from_slice(&0u16.to_le_bytes());
    v1.extend_from_slice(&0u16.to_le_bytes());
    v1.extend_from_slice(b"nonulpad");
    let err = hdf5_pure_rust::format::messages::filter_pipeline::FilterPipelineMessage::decode(&v1)
        .unwrap_err();
    assert!(format!("{err}").contains("not null-terminated"));

    let mut v2 = vec![2, 1];
    v2.extend_from_slice(&256u16.to_le_bytes());
    v2.extend_from_slice(&4u16.to_le_bytes());
    v2.extend_from_slice(b"name");
    v2.extend_from_slice(&0u16.to_le_bytes());
    v2.extend_from_slice(&0u16.to_le_bytes());
    let err = hdf5_pure_rust::format::messages::filter_pipeline::FilterPipelineMessage::decode(&v2)
        .unwrap_err();
    assert!(format!("{err}").contains("not null-terminated"));
}

#[test]
fn test_filter_pipeline_rejects_invalid_utf8_names() {
    let mut v1 = vec![1, 1, 0, 0, 0, 0, 0, 0];
    v1.extend_from_slice(&256u16.to_le_bytes());
    v1.extend_from_slice(&8u16.to_le_bytes());
    v1.extend_from_slice(&0u16.to_le_bytes());
    v1.extend_from_slice(&0u16.to_le_bytes());
    v1.extend_from_slice(&[0xff, 0, 0, 0, 0, 0, 0, 0]);
    let err = hdf5_pure_rust::format::messages::filter_pipeline::FilterPipelineMessage::decode(&v1)
        .unwrap_err();
    assert!(format!("{err}").contains("not UTF-8"));

    let mut v2 = vec![2, 1];
    v2.extend_from_slice(&256u16.to_le_bytes());
    v2.extend_from_slice(&2u16.to_le_bytes());
    v2.extend_from_slice(&[0xff, 0]);
    v2.extend_from_slice(&0u16.to_le_bytes());
    v2.extend_from_slice(&0u16.to_le_bytes());
    let err = hdf5_pure_rust::format::messages::filter_pipeline::FilterPipelineMessage::decode(&v2)
        .unwrap_err();
    assert!(format!("{err}").contains("not UTF-8"));
}

#[test]
fn test_filter_pipeline_v1_skips_client_data_padding_like_libhdf5() {
    let mut buf = vec![1, 1, 0, 0, 0, 0, 0, 0];
    buf.extend_from_slice(&1u16.to_le_bytes()); // filter id
    buf.extend_from_slice(&0u16.to_le_bytes()); // name length
    buf.extend_from_slice(&0u16.to_le_bytes()); // flags
    buf.extend_from_slice(&1u16.to_le_bytes()); // odd client data count
    buf.extend_from_slice(&6u32.to_le_bytes()); // one client data value
    buf.extend_from_slice(&1u32.to_le_bytes()); // nonzero v1 padding

    hdf5_pure_rust::format::messages::filter_pipeline::FilterPipelineMessage::decode(&buf)
        .expect("H5O__pline_decode skips v1 client-data padding bytes");
}
