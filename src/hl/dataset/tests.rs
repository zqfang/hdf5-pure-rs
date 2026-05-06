use super::chunk_read::ChunkReadContext;
use super::virtual_dataset::{
    IrregularHyperslabBlock, RegularHyperslab, VirtualMapping, VirtualSelection,
};
use super::*;
use crate::error::Error;
use crate::format::messages::data_layout::{ChunkIndexType, DataLayoutMessage, LayoutClass};
use crate::format::messages::dataspace::{DataspaceMessage, DataspaceType};
use crate::format::messages::datatype::DatatypeMessage;
use crate::format::messages::filter_pipeline::FilterPipelineMessage;
use crate::format::object_header::{self, RawMessage};
use crate::hl::value::H5Value;
use crate::io::reader::HdfReader;
use std::io::Cursor;
use tempfile::tempdir;

fn le_u32(value: u32, out: &mut Vec<u8>) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn le_u64(value: u64, out: &mut Vec<u8>) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn build_global_heap_collection(
    collection_addr: usize,
    object_index: u16,
    payload: &[u8],
) -> Vec<u8> {
    let mut bytes = vec![0; collection_addr];
    bytes.extend_from_slice(b"GCOL");
    bytes.push(1);
    bytes.extend_from_slice(&[0; 3]);

    let aligned = (payload.len() as u64 + 7) & !7;
    let total_size = 16u64 + 16u64 + aligned + 16u64;
    bytes.extend_from_slice(&total_size.to_le_bytes());

    bytes.extend_from_slice(&object_index.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&[0; 4]);
    bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    bytes.extend_from_slice(payload);
    bytes.resize(collection_addr + 16 + 16 + aligned as usize, 0);

    bytes.extend_from_slice(&0u16.to_le_bytes());
    bytes.extend_from_slice(&0u16.to_le_bytes());
    bytes.extend_from_slice(&[0; 4]);
    bytes.extend_from_slice(&16u64.to_le_bytes());
    bytes
}

#[test]
fn virtual_point_selection_decodes_and_materializes() {
    let mut encoded = Vec::new();
    le_u32(1, &mut encoded); // H5S_SEL_POINTS
    le_u32(2, &mut encoded); // point selection version
    encoded.push(8); // encoded integer size
    le_u32(2, &mut encoded); // rank
    le_u64(3, &mut encoded); // number of points
    le_u64(0, &mut encoded);
    le_u64(1, &mut encoded);
    le_u64(2, &mut encoded);
    le_u64(3, &mut encoded);
    le_u64(1, &mut encoded);
    le_u64(4, &mut encoded);
    let mut pos = 0;

    let selection = Dataset::decode_virtual_selection(&encoded, &mut pos)
        .expect("point VDS selections should decode");

    let points = Dataset::materialize_virtual_selection_points(&selection, &[3, 5])
        .expect("point VDS selections should materialize");
    assert_eq!(points, vec![vec![0, 1], vec![2, 3], vec![1, 4]]);
}

#[test]
fn virtual_irregular_hyperslab_selection_decodes_and_materializes() {
    let mut encoded = Vec::new();
    le_u32(2, &mut encoded); // H5S_SEL_HYPERSLABS
    le_u32(3, &mut encoded); // hyperslab selection version
    encoded.push(0); // flags without H5S_HYPER_REGULAR
    encoded.push(8); // encoded integer size
    le_u32(2, &mut encoded); // rank
    le_u64(2, &mut encoded); // block count
    le_u64(0, &mut encoded);
    le_u64(1, &mut encoded);
    le_u64(0, &mut encoded);
    le_u64(2, &mut encoded);
    le_u64(2, &mut encoded);
    le_u64(0, &mut encoded);
    le_u64(2, &mut encoded);
    le_u64(1, &mut encoded);

    let mut pos = 0;
    let selection = Dataset::decode_virtual_selection(&encoded, &mut pos)
        .expect("irregular VDS hyperslabs should decode");

    let points = Dataset::materialize_virtual_selection_points(&selection, &[4, 4])
        .expect("irregular VDS hyperslabs should materialize");
    assert_eq!(
        points,
        vec![vec![0, 1], vec![0, 2], vec![2, 0], vec![2, 1],]
    );
}

#[test]
fn virtual_hyperslab_selection_rejects_unknown_flags() {
    let mut encoded = Vec::new();
    le_u32(2, &mut encoded); // H5S_SEL_HYPERSLABS
    le_u32(3, &mut encoded); // hyperslab selection version
    encoded.push(0x80); // unknown flag
    encoded.push(8); // encoded integer size
    le_u32(1, &mut encoded); // rank
    le_u64(1, &mut encoded); // block count
    le_u64(0, &mut encoded); // start
    le_u64(0, &mut encoded); // end

    let mut pos = 0;
    let err = Dataset::decode_virtual_selection(&encoded, &mut pos)
        .expect_err("unknown hyperslab flags should be rejected");

    assert!(matches!(err, Error::InvalidFormat(_)));
    assert!(err.to_string().contains("unknown flags"));
}

#[test]
fn virtual_selection_skips_reserved_header_bytes_like_libhdf5() {
    let mut all = Vec::new();
    le_u32(3, &mut all); // H5S_SEL_ALL
    le_u32(1, &mut all); // all selection version
    all.extend_from_slice(&[0, 0, 1, 0, 0, 0, 0, 0]);
    let mut pos = 0;
    Dataset::decode_virtual_selection(&all, &mut pos)
        .expect("H5S__all_deserialize skips reserved header bytes");

    let mut points = Vec::new();
    le_u32(1, &mut points); // H5S_SEL_POINTS
    le_u32(1, &mut points); // point selection version
    points.extend_from_slice(&[0, 0, 1, 0, 0, 0, 0, 0]);
    le_u32(1, &mut points); // rank
    le_u32(0, &mut points); // point count
    let mut pos = 0;
    Dataset::decode_virtual_selection(&points, &mut pos)
        .expect("H5S__point_deserialize skips reserved header bytes");

    let mut hyperslab = Vec::new();
    le_u32(2, &mut hyperslab); // H5S_SEL_HYPERSLABS
    le_u32(2, &mut hyperslab); // hyperslab selection version
    hyperslab.push(0); // flags
    hyperslab.extend_from_slice(&[0, 1, 0, 0]);
    le_u32(1, &mut hyperslab); // rank
    le_u64(0, &mut hyperslab); // block count
    let mut pos = 0;
    Dataset::decode_virtual_selection(&hyperslab, &mut pos)
        .expect("H5S__hyper_deserialize skips reserved header bytes");
}

#[test]
fn virtual_mapping_rejects_unknown_flags() {
    let mut encoded = Vec::new();
    encoded.push(1); // VDS heap encoding version
    le_u64(1, &mut encoded); // mapping count, sizeof_size=8
    encoded.push(0x80); // unknown mapping flag

    let err = Dataset::decode_virtual_mappings(&encoded, 8)
        .expect_err("unknown VDS mapping flags should be rejected");
    assert!(matches!(err, Error::InvalidFormat(_)));
    assert!(err.to_string().contains("mapping flags"));
}

#[test]
fn virtual_mapping_rejects_conflicting_file_name_flags() {
    let mut encoded = Vec::new();
    encoded.push(1); // VDS heap encoding version
    le_u64(1, &mut encoded); // mapping count, sizeof_size=8
    encoded.push(0x05); // same-file plus shared file-name flags

    let err = Dataset::decode_virtual_mappings(&encoded, 8)
        .expect_err("conflicting VDS file-name flags should be rejected");
    assert!(matches!(err, Error::InvalidFormat(_)));
    assert!(err.to_string().contains("same-file"));
}

#[test]
fn virtual_output_extent_uses_point_and_irregular_destination_bounds() {
    let point_extent = Dataset::virtual_mapping_output_extent(
        &VirtualSelection::All,
        &VirtualSelection::Points(vec![vec![1, 4], vec![3, 2]]),
        &[10, 10],
        1,
    )
    .expect("point VDS extent should derive from destination bounds");
    assert_eq!(point_extent, 5);

    let irregular_extent = Dataset::virtual_mapping_output_extent(
        &VirtualSelection::All,
        &VirtualSelection::Irregular(vec![
            IrregularHyperslabBlock {
                start: vec![0, 1],
                block: vec![1, 2],
            },
            IrregularHyperslabBlock {
                start: vec![4, 0],
                block: vec![2, 1],
            },
        ]),
        &[10, 10],
        0,
    )
    .expect("irregular VDS extent should derive from destination block bounds");
    assert_eq!(irregular_extent, 6);
}

#[test]
fn virtual_output_extent_rejects_overflow() {
    let point_err = Dataset::virtual_mapping_output_extent(
        &VirtualSelection::All,
        &VirtualSelection::Points(vec![vec![u64::MAX]]),
        &[10],
        0,
    )
    .expect_err("point extent should reject u64 overflow");
    assert!(point_err.to_string().contains("point-selection extent"));

    let regular_err = Dataset::virtual_mapping_output_extent(
        &VirtualSelection::Regular(RegularHyperslab {
            start: vec![0],
            stride: vec![u64::MAX],
            count: vec![2],
            block: vec![2],
        }),
        &VirtualSelection::Regular(RegularHyperslab {
            start: vec![u64::MAX],
            stride: vec![1],
            count: vec![1],
            block: vec![1],
        }),
        &[u64::MAX],
        0,
    )
    .expect_err("regular extent should reject u64 overflow");
    assert!(regular_err.to_string().contains("hyperslab span overflow"));

    let irregular_err = Dataset::virtual_mapping_output_extent(
        &VirtualSelection::All,
        &VirtualSelection::Irregular(vec![IrregularHyperslabBlock {
            start: vec![u64::MAX],
            block: vec![1],
        }]),
        &[10],
        0,
    )
    .expect_err("irregular extent should reject u64 overflow");
    assert!(irregular_err
        .to_string()
        .contains("irregular-selection extent"));
}

#[test]
fn virtual_irregular_hyperslab_materialization_rejects_coordinate_overflow() {
    let selection = VirtualSelection::Irregular(vec![IrregularHyperslabBlock {
        start: vec![u64::MAX],
        block: vec![2],
    }]);
    let err = Dataset::materialize_virtual_selection_points(&selection, &[u64::MAX]).unwrap_err();
    assert!(
        err.to_string()
            .contains("irregular hyperslab coordinate overflow"),
        "unexpected error: {err}"
    );
}

#[test]
fn virtual_regular_unlimited_hyperslab_rejects_start_past_extent() {
    let selection = VirtualSelection::Regular(RegularHyperslab {
        start: vec![6],
        stride: vec![1],
        count: vec![u64::MAX],
        block: vec![u64::MAX],
    });

    let err = Dataset::materialize_virtual_selection_points(&selection, &[5]).unwrap_err();
    assert!(
        err.to_string()
            .contains("virtual regular hyperslab start exceeds dataspace extent"),
        "unexpected error: {err}"
    );

    let err = Dataset::virtual_selection_span(&selection, &[5], 0).unwrap_err();
    assert!(
        err.to_string()
            .contains("virtual regular hyperslab start exceeds dataspace extent"),
        "unexpected error: {err}"
    );
}

#[test]
fn virtual_source_resolution_requires_base_path_for_relative_and_same_file_sources() {
    let access = DatasetAccess::new();
    for source in [".", "relative-source.h5"] {
        let err = Dataset::resolve_virtual_source_path(None, source, &access)
            .expect_err("VDS source resolution without a base file should fail");
        assert!(matches!(err, Error::Unsupported(_)));
    }
}

#[test]
fn btree_v1_chunk_records_preserve_8_byte_chunk_addresses() {
    let mut node = Vec::new();
    node.extend_from_slice(b"TREE");
    node.push(1); // raw data B-tree
    node.push(0); // leaf
    node.extend_from_slice(&1u16.to_le_bytes()); // entries used
    node.extend_from_slice(&u64::MAX.to_le_bytes()); // left sibling
    node.extend_from_slice(&u64::MAX.to_le_bytes()); // right sibling

    node.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    node.extend_from_slice(&0u32.to_le_bytes()); // filter mask
    node.extend_from_slice(&4u64.to_le_bytes()); // dim 0 chunk offset
    node.extend_from_slice(&8u64.to_le_bytes()); // dim 1 chunk offset
    node.extend_from_slice(&0u64.to_le_bytes()); // extra element-size dimension
    let large_chunk_addr = 0x1_0000_0040u64;
    node.extend_from_slice(&large_chunk_addr.to_le_bytes());

    node.extend_from_slice(&0u32.to_le_bytes()); // final key chunk size
    node.extend_from_slice(&0u32.to_le_bytes()); // final key filter mask
    node.extend_from_slice(&0u64.to_le_bytes()); // final dim 0 key
    node.extend_from_slice(&0u64.to_le_bytes()); // final dim 1 key
    node.extend_from_slice(&0u64.to_le_bytes()); // final extra key

    let mut reader = HdfReader::new(Cursor::new(node));
    reader.set_sizeof_addr(8);
    let chunks = Dataset::collect_btree_v1_chunks(&mut reader, 0, 2).unwrap();

    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].coords, vec![4, 8]);
    assert_eq!(chunks[0].chunk_addr, large_chunk_addr);
    assert_eq!(chunks[0].chunk_size, 16);
    assert_eq!(chunks[0].filter_mask, 0);
}

#[test]
fn btree_v1_chunk_traversal_rejects_child_level_mismatch() {
    fn leaf_node(level: u8, addr: u64) -> Vec<u8> {
        let mut node = Vec::new();
        node.extend_from_slice(b"TREE");
        node.push(1);
        node.push(level);
        node.extend_from_slice(&1u16.to_le_bytes());
        node.extend_from_slice(&u64::MAX.to_le_bytes());
        node.extend_from_slice(&u64::MAX.to_le_bytes());
        node.extend_from_slice(&16u32.to_le_bytes());
        node.extend_from_slice(&0u32.to_le_bytes());
        node.extend_from_slice(&0u64.to_le_bytes());
        node.extend_from_slice(&0u64.to_le_bytes());
        node.extend_from_slice(&addr.to_le_bytes());
        node.extend_from_slice(&0u32.to_le_bytes());
        node.extend_from_slice(&0u32.to_le_bytes());
        node.extend_from_slice(&0u64.to_le_bytes());
        node.extend_from_slice(&0u64.to_le_bytes());
        node
    }

    let mut root = Vec::new();
    root.extend_from_slice(b"TREE");
    root.push(1);
    root.push(1);
    root.extend_from_slice(&1u16.to_le_bytes());
    root.extend_from_slice(&u64::MAX.to_le_bytes());
    root.extend_from_slice(&u64::MAX.to_le_bytes());
    root.extend_from_slice(&0u32.to_le_bytes());
    root.extend_from_slice(&0u32.to_le_bytes());
    root.extend_from_slice(&0u64.to_le_bytes());
    root.extend_from_slice(&0u64.to_le_bytes());
    root.extend_from_slice(&128u64.to_le_bytes());
    root.extend_from_slice(&0u32.to_le_bytes());
    root.extend_from_slice(&0u32.to_le_bytes());
    root.extend_from_slice(&0u64.to_le_bytes());
    root.extend_from_slice(&0u64.to_le_bytes());

    let mut file = root;
    file.resize(128, 0);
    file.extend_from_slice(&leaf_node(1, 256)); // should be level 0 under root level 1.

    let mut reader = HdfReader::new(Cursor::new(file));
    reader.set_sizeof_addr(8);
    let err = Dataset::collect_btree_v1_chunks(&mut reader, 0, 1)
        .expect_err("child level mismatch should fail");
    assert!(err.to_string().contains("child level"));
}

#[test]
fn external_file_list_skips_reserved_and_trailing_bytes_like_libhdf5() {
    let mut reserved = vec![1, 1, 0, 0];
    reserved.extend_from_slice(&1u16.to_le_bytes()); // allocated slots
    reserved.extend_from_slice(&0u16.to_le_bytes()); // used slots
    reserved.extend_from_slice(&0u64.to_le_bytes()); // heap address
    Dataset::decode_external_file_list(&reserved, 8, 8)
        .expect("H5O__efl_decode skips external-file-list reserved bytes");

    let mut trailing = vec![1, 0, 0, 0];
    trailing.extend_from_slice(&1u16.to_le_bytes()); // allocated slots
    trailing.extend_from_slice(&0u16.to_le_bytes()); // used slots
    trailing.extend_from_slice(&0u64.to_le_bytes()); // heap address
    trailing.push(0);
    Dataset::decode_external_file_list(&trailing, 8, 8)
        .expect("H5O__efl_decode tolerates trailing payload bytes");
}

#[test]
fn external_file_list_rejects_invalid_slot_count_and_undefined_heap() {
    let mut no_alloc = vec![1, 0, 0, 0];
    no_alloc.extend_from_slice(&0u16.to_le_bytes()); // allocated slots
    no_alloc.extend_from_slice(&0u16.to_le_bytes()); // used slots
    no_alloc.extend_from_slice(&0u64.to_le_bytes()); // heap address
    let err = Dataset::decode_external_file_list(&no_alloc, 8, 8)
        .expect_err("external file list with no allocated slots should fail");
    assert!(err.to_string().contains("allocated slots"));

    let mut undefined_heap = vec![1, 0, 0, 0];
    undefined_heap.extend_from_slice(&1u16.to_le_bytes()); // allocated slots
    undefined_heap.extend_from_slice(&1u16.to_le_bytes()); // used slots
    undefined_heap.extend_from_slice(&u64::MAX.to_le_bytes()); // undefined heap address
    undefined_heap.extend_from_slice(&0u64.to_le_bytes()); // name offset
    undefined_heap.extend_from_slice(&0u64.to_le_bytes()); // file offset
    undefined_heap.extend_from_slice(&u64::MAX.to_le_bytes()); // size
    let err = Dataset::decode_external_file_list(&undefined_heap, 8, 8)
        .expect_err("used external file list with undefined heap should fail");
    assert!(err.to_string().contains("heap address"));
}

#[test]
fn external_file_list_entry_fields_use_file_size_width() {
    let mut bytes = vec![1, 0, 0, 0];
    bytes.extend_from_slice(&1u16.to_le_bytes()); // allocated slots
    bytes.extend_from_slice(&1u16.to_le_bytes()); // used slots
    bytes.extend_from_slice(&16u32.to_le_bytes()); // heap address
    bytes.extend_from_slice(&4u32.to_le_bytes()); // name offset
    bytes.extend_from_slice(&8u32.to_le_bytes()); // file offset
    bytes.extend_from_slice(&12u32.to_le_bytes()); // size
    let decoded = Dataset::decode_external_file_list(&bytes, 4, 4)
        .expect("H5O__efl_decode uses sizeof_size for all entry length fields");
    assert_eq!(decoded.heap_addr, 16);
    assert_eq!(decoded.entries.len(), 1);
    assert_eq!(decoded.entries[0].name_offset, 4);
    assert_eq!(decoded.entries[0].file_offset, 8);
    assert_eq!(decoded.entries[0].size, 12);
}

#[test]
fn filtered_implicit_chunk_index_is_rejected() {
    let info = DatasetInfo {
        dataspace: DataspaceMessage {
            version: 2,
            space_type: DataspaceType::Simple,
            ndims: 1,
            dims: vec![4],
            max_dims: None,
        },
        datatype: DatatypeMessage {
            version: 1,
            class: crate::format::messages::datatype::DatatypeClass::FixedPoint,
            class_bits: [0, 0, 0],
            size: 4,
            properties: Vec::new(),
        },
        layout: DataLayoutMessage {
            version: 4,
            layout_class: LayoutClass::Chunked,
            compact_data: None,
            contiguous_addr: None,
            contiguous_size: None,
            chunk_dims: Some(vec![2]),
            chunk_index_addr: Some(0),
            chunk_index_type: Some(ChunkIndexType::Implicit),
            chunk_element_size: None,
            chunk_flags: None,
            chunk_encoded_dims: None,
            single_chunk_filtered_size: None,
            single_chunk_filter_mask: None,
            data_addr: Some(0),
            virtual_heap_addr: None,
            virtual_heap_index: None,
        },
        filter_pipeline: Some(FilterPipelineMessage {
            version: 2,
            filters: vec![crate::format::messages::filter_pipeline::FilterDesc {
                id: crate::format::messages::filter_pipeline::FILTER_DEFLATE,
                name: None,
                flags: 0,
                client_data: vec![4],
            }],
        }),
        fill_value: None,
        external_file_list: None,
    };
    let mut reader = HdfReader::new(Cursor::new(Vec::<u8>::new()));
    let chunk_ctx = ChunkReadContext {
        idx_addr: 0,
        data_dims: &[4],
        chunk_dims: &[2],
        chunk_bytes: 8,
        element_size: 4,
        total_bytes: 16,
    };
    let err = Dataset::read_chunked_implicit(&mut reader, &info, &chunk_ctx)
        .expect_err("filtered implicit chunk indexes should be rejected");

    assert!(matches!(err, Error::Unsupported(_)));
    assert!(
        err.to_string()
            .contains("v4 implicit chunk index with filters"),
        "unexpected error: {err}"
    );
}

#[test]
fn old_fill_value_must_match_datatype_size_during_dataset_parse() {
    let messages = vec![
        RawMessage {
            msg_type: object_header::MSG_DATASPACE,
            flags: 0,
            creation_index: None,
            chunk_index: 0,
            data: vec![2, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        },
        RawMessage {
            msg_type: object_header::MSG_DATATYPE,
            flags: 0,
            creation_index: None,
            chunk_index: 0,
            data: vec![0x10, 0, 0, 0, 4, 0, 0, 0, 0, 0, 32, 0],
        },
        RawMessage {
            msg_type: object_header::MSG_LAYOUT,
            flags: 0,
            creation_index: None,
            chunk_index: 0,
            data: vec![3, 0, 0, 0],
        },
        RawMessage {
            msg_type: object_header::MSG_FILL_VALUE_OLD,
            flags: 0,
            creation_index: None,
            chunk_index: 0,
            data: vec![2, 0, 0, 0, 1, 2],
        },
    ];

    let err = Dataset::parse_info(&messages, 8, 8)
        .expect_err("old fill value with mismatched datatype size should fail");
    assert!(
        err.to_string().contains("does not match datatype size"),
        "unexpected error: {err}"
    );
}

#[test]
fn vlen_sequence_reads_only_requested_bytes() {
    let heap_addr = 32u64;
    let heap = build_global_heap_collection(heap_addr as usize, 1, &[1, 2, 3, 4, 99, 100]);
    let mut reader = HdfReader::new(Cursor::new(heap));
    reader.set_sizeof_size(8);

    let base = DatatypeMessage {
        version: 1,
        class: crate::format::messages::datatype::DatatypeClass::FixedPoint,
        class_bits: [0, 0, 0],
        size: 2,
        properties: vec![0, 0, 16, 0],
    };

    let mut descriptor = Vec::new();
    descriptor.extend_from_slice(&2u32.to_le_bytes());
    descriptor.extend_from_slice(&heap_addr.to_le_bytes());
    descriptor.extend_from_slice(&1u32.to_le_bytes());

    let value = Dataset::decode_vlen_value(Some(&base), &descriptor, 8, &mut reader)
        .expect("exact-length vlen read should succeed");
    match value {
        H5Value::VarLen(values) => {
            assert_eq!(values.len(), 2);
            assert!(matches!(values[0], H5Value::UInt(513)));
            assert!(matches!(values[1], H5Value::UInt(1027)));
        }
        other => panic!("expected VarLen, got {other:?}"),
    }
}

#[test]
fn vlen_string_rejects_short_heap_payload() {
    let heap_addr = 32u64;
    let heap = build_global_heap_collection(heap_addr as usize, 1, b"abc");
    let mut reader = HdfReader::new(Cursor::new(heap));
    reader.set_sizeof_size(8);

    let base = DatatypeMessage {
        version: 1,
        class: crate::format::messages::datatype::DatatypeClass::String,
        class_bits: [0, 0, 0],
        size: 1,
        properties: Vec::new(),
    };

    let mut descriptor = Vec::new();
    descriptor.extend_from_slice(&4u32.to_le_bytes());
    descriptor.extend_from_slice(&heap_addr.to_le_bytes());
    descriptor.extend_from_slice(&1u32.to_le_bytes());

    let err = Dataset::decode_vlen_value(Some(&base), &descriptor, 8, &mut reader)
        .expect_err("short vlen string payload should fail");
    assert!(
        err.to_string().contains("payload too short"),
        "unexpected error: {err}"
    );
}

#[test]
fn virtual_output_dims_respects_vds_view_for_unlimited_dimensions() {
    let dir = tempdir().unwrap();
    let short_path = dir.path().join("short.h5");
    let long_path = dir.path().join("long.h5");

    {
        let mut wf = crate::hl::writable_file::WritableFile::create(&short_path).unwrap();
        wf.new_dataset_builder("data")
            .write::<i32>(&[1, 2, 3])
            .unwrap();
        wf.close().unwrap();
    }

    {
        let mut wf = crate::hl::writable_file::WritableFile::create(&long_path).unwrap();
        wf.new_dataset_builder("data")
            .write::<i32>(&[1, 2, 3, 4, 5])
            .unwrap();
        wf.close().unwrap();
    }

    let mappings = vec![
        VirtualMapping {
            file_name: short_path.to_string_lossy().into_owned(),
            dataset_name: "data".to_string(),
            source_select: VirtualSelection::All,
            virtual_select: VirtualSelection::All,
        },
        VirtualMapping {
            file_name: long_path.to_string_lossy().into_owned(),
            dataset_name: "data".to_string(),
            source_select: VirtualSelection::All,
            virtual_select: VirtualSelection::All,
        },
    ];
    let info = DatasetInfo {
        dataspace: DataspaceMessage {
            version: 2,
            space_type: DataspaceType::Simple,
            ndims: 1,
            dims: vec![0],
            max_dims: Some(vec![u64::MAX]),
        },
        datatype: DatatypeMessage {
            version: 1,
            class: crate::format::messages::datatype::DatatypeClass::FixedPoint,
            class_bits: [0, 0, 0],
            size: 4,
            properties: vec![0, 0, 32, 0],
        },
        layout: DataLayoutMessage {
            version: 4,
            layout_class: LayoutClass::Virtual,
            compact_data: None,
            contiguous_addr: None,
            contiguous_size: None,
            chunk_dims: None,
            chunk_index_addr: None,
            chunk_index_type: None,
            chunk_element_size: None,
            chunk_flags: None,
            chunk_encoded_dims: None,
            single_chunk_filtered_size: None,
            single_chunk_filter_mask: None,
            data_addr: None,
            virtual_heap_addr: None,
            virtual_heap_index: None,
        },
        filter_pipeline: None,
        fill_value: None,
        external_file_list: None,
    };

    let last_available = Dataset::virtual_output_dims(
        &mappings,
        None,
        &info,
        &DatasetAccess::new().with_virtual_view(VdsView::LastAvailable),
    )
    .unwrap();
    let first_missing = Dataset::virtual_output_dims(
        &mappings,
        None,
        &info,
        &DatasetAccess::new().with_virtual_view(VdsView::FirstMissing),
    )
    .unwrap();

    assert_eq!(last_available, vec![5]);
    assert_eq!(first_missing, vec![3]);
}

#[test]
fn virtual_output_dims_uses_declared_extents_for_missing_unlimited_sources() {
    let dir = tempdir().unwrap();
    let missing_path = dir.path().join("missing.h5");
    let long_path = dir.path().join("long.h5");

    {
        let mut wf = crate::hl::writable_file::WritableFile::create(&long_path).unwrap();
        wf.new_dataset_builder("data")
            .write::<i32>(&[1, 2])
            .unwrap();
        wf.close().unwrap();
    }

    let mappings = vec![
        VirtualMapping {
            file_name: missing_path.to_string_lossy().into_owned(),
            dataset_name: "data".to_string(),
            source_select: VirtualSelection::All,
            virtual_select: VirtualSelection::Regular(RegularHyperslab {
                start: vec![0],
                stride: vec![1],
                count: vec![3],
                block: vec![1],
            }),
        },
        VirtualMapping {
            file_name: long_path.to_string_lossy().into_owned(),
            dataset_name: "data".to_string(),
            source_select: VirtualSelection::All,
            virtual_select: VirtualSelection::Regular(RegularHyperslab {
                start: vec![3],
                stride: vec![1],
                count: vec![2],
                block: vec![1],
            }),
        },
    ];
    let info = DatasetInfo {
        dataspace: DataspaceMessage {
            version: 2,
            space_type: DataspaceType::Simple,
            ndims: 1,
            dims: vec![0],
            max_dims: Some(vec![u64::MAX]),
        },
        datatype: DatatypeMessage {
            version: 1,
            class: crate::format::messages::datatype::DatatypeClass::FixedPoint,
            class_bits: [0, 0, 0],
            size: 4,
            properties: vec![0, 0, 32, 0],
        },
        layout: DataLayoutMessage {
            version: 4,
            layout_class: LayoutClass::Virtual,
            compact_data: None,
            contiguous_addr: None,
            contiguous_size: None,
            chunk_dims: None,
            chunk_index_addr: None,
            chunk_index_type: None,
            chunk_element_size: None,
            chunk_flags: None,
            chunk_encoded_dims: None,
            single_chunk_filtered_size: None,
            single_chunk_filter_mask: None,
            data_addr: None,
            virtual_heap_addr: None,
            virtual_heap_index: None,
        },
        filter_pipeline: None,
        fill_value: None,
        external_file_list: None,
    };
    let access =
        DatasetAccess::new().with_virtual_missing_source_policy(VdsMissingSourcePolicy::Fill);

    let last_available = Dataset::virtual_output_dims(
        &mappings,
        None,
        &info,
        &access.clone().with_virtual_view(VdsView::LastAvailable),
    )
    .unwrap();
    let first_missing = Dataset::virtual_output_dims(
        &mappings,
        None,
        &info,
        &access.with_virtual_view(VdsView::FirstMissing),
    )
    .unwrap();

    assert_eq!(last_available, vec![5]);
    assert_eq!(first_missing, vec![3]);
}

#[test]
fn virtual_declared_output_extent_rejects_overflow() {
    let point_err = Dataset::virtual_mapping_declared_output_extent(
        &VirtualSelection::Points(vec![vec![u64::MAX]]),
        0,
    )
    .expect_err("point extent should reject u64 overflow");
    assert!(point_err.to_string().contains("point-selection extent"));

    let irregular_err = Dataset::virtual_mapping_declared_output_extent(
        &VirtualSelection::Irregular(vec![IrregularHyperslabBlock {
            start: vec![u64::MAX],
            block: vec![1],
        }]),
        0,
    )
    .expect_err("irregular block extent should reject u64 overflow");
    assert!(irregular_err
        .to_string()
        .contains("irregular-selection extent"));
}
