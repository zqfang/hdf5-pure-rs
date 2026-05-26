//! Phase T7: Dataspace and selection tests.

use hdf5_pure_rust::{
    hl::selection::{
        H5S__point_serialize_into, H5S_select_serialize_into, H5Sget_select_bounds_into,
        H5Sget_select_hyper_blocklist_into,
    },
    Dataspace, DataspaceType, File, HyperslabDim, Selection, SelectionType,
};

const FILE: &str = "tests/data/hdf5_ref/selections_test.h5";
fn open() -> File {
    File::open(FILE).unwrap()
}

// T7a: Dataspace types

#[test]
fn t7a_simple_dataspace() {
    let ds = open().dataset("seq100").unwrap();
    let space = ds.space().unwrap();
    assert!(space.is_simple());
    assert!(!space.is_scalar());
    assert!(!space.is_null());
    assert_eq!(space.ndim(), 1);
    assert_eq!(space.shape(), &[100]);
    assert_eq!(space.size(), 100);
}

#[test]
fn t7a_dataspace_create_copy_and_extent_mutation() {
    let scalar = Dataspace::scalar();
    assert!(scalar.is_scalar());
    assert_eq!(scalar.extent_type(), DataspaceType::Scalar);
    assert!(scalar.has_extent());
    assert_eq!(scalar.npoints_max(), 1);

    let null = Dataspace::null();
    assert!(null.is_null());
    assert_eq!(null.extent_nelem(), 0);

    let mut simple = Dataspace::simple(vec![2, 3], Some(vec![4, u64::MAX])).unwrap();
    assert!(simple.is_simple());
    assert_eq!(
        simple.extent_dims(),
        (&[2, 3][..], Some(&[4, u64::MAX][..]))
    );
    assert_eq!(simple.npoints_max(), u64::MAX);
    assert!(simple.extent_equal(&simple.copy()));
    simple.set_extent_simple(vec![5], None).unwrap();
    assert_eq!(simple.shape(), &[5]);
    assert_eq!(simple.extent_nelem(), 5);
    simple.set_version(1).unwrap();
    assert_eq!(simple.raw_message_ref().version, 1);
    assert!(simple.set_version(3).is_err());
}

#[test]
fn t7a_scalar_dataspace() {
    let ds = open().dataset("scalar_val").unwrap();
    let space = ds.space().unwrap();
    assert!(space.is_scalar());
    assert_eq!(space.ndim(), 0);
    assert_eq!(space.size(), 1);
}

#[test]
fn t7a_null_dataspace() {
    let ds = open().dataset("null_ds").unwrap();
    let space = ds.space().unwrap();
    assert!(space.is_null() || space.is_scalar());
    assert_eq!(space.ndim(), 0);
}

// T7b: Dimension queries

#[test]
fn t7b_2d_shape() {
    let ds = open().dataset("matrix").unwrap();
    let space = ds.space().unwrap();
    assert_eq!(space.shape(), &[6, 10]);
    assert_eq!(space.ndim(), 2);
    assert_eq!(space.size(), 60);
}

#[test]
fn t7b_3d_shape() {
    let ds = open().dataset("cube").unwrap();
    let space = ds.space().unwrap();
    assert_eq!(space.shape(), &[4, 5, 6]);
}

#[test]
fn t7b_resizable_maxdims() {
    let ds = open().dataset("resizable").unwrap();
    let space = ds.space().unwrap();
    assert!(space.is_resizable());
    let maxdims = space.maxdims().unwrap();
    assert_eq!(maxdims[0], u64::MAX); // unlimited
}

#[test]
fn t7b_non_resizable() {
    let ds = open().dataset("seq100").unwrap();
    assert!(!ds.space().unwrap().is_resizable());
}

// T7c-e: Selection / read-slice tests

#[test]
fn t7c_slice_1d_range() {
    let ds = open().dataset("seq100").unwrap();
    let mut vals = vec![0.0; 10];
    ds.read_slice_into::<f64, _>(10..20, &mut vals).unwrap();
    assert_eq!(vals.len(), 10);
    for (i, v) in vals.iter().enumerate() {
        assert_eq!(*v, (10 + i) as f64);
    }
}

#[test]
fn t7c_selection_constructor_and_internal_query_aliases() {
    let mut start = Vec::new();
    let mut end = Vec::new();
    assert!(!Selection::None.bounds_into(&[2, 2], &mut start, &mut end));
    assert!(start.is_empty());
    assert!(end.is_empty());
    assert_eq!(Selection::None.selection_type(), SelectionType::None);
    assert_eq!(Selection::None.selected_count(&[2, 2]), Some(0));
    assert!(Selection::None.select_valid(&[2, 2]));
    let mut encoded_none = Vec::new();
    Selection::None.encode1_into(&mut encoded_none).unwrap();
    assert_eq!(encoded_none, [0]);
    assert!(!Selection::None.bounds_into(&[2, 2], &mut start, &mut end));
    assert!(Selection::None.is_contiguous(&[2, 2]).unwrap());
    assert!(!Selection::None.select_is_single(&[2, 2]));
    assert!(Selection::None.is_regular());
    assert_eq!(
        Selection::None.select_adjust_signed(&[1, -1]).unwrap(),
        Selection::None
    );

    let mut all_iter = Selection::All.select_iter_init(&[2, 2]).unwrap();
    let mut coord = [0, 0];
    assert!(Selection::All
        .select_iter_init(&[2, 2])
        .unwrap()
        .select_iter_next_into(&mut coord)
        .unwrap());
    assert_eq!(coord, [0, 0]);
    start.clear();
    end.clear();
    assert!(Selection::All.bounds_into(&[2, 2], &mut start, &mut end));
    assert_eq!(start, [0, 0]);
    assert_eq!(end, [1, 1]);
    assert_eq!(Selection::All.selected_count(&[2, 2]), Some(4));
    assert_eq!(Selection::All.hyperslab_block_count(&[2, 2]), Some(1));
    assert_eq!(all_iter.select_iter_next_ref(), Some(&[0, 0][..]));
    start.clear();
    end.clear();
    assert!(Selection::All.bounds_into(&[2, 2], &mut start, &mut end));
    assert_eq!(start, [0, 0]);
    assert_eq!(end, [1, 1]);
    let mut seq = [0; 6];
    assert_eq!(
        Selection::All
            .select_iter_init(&[2, 2])
            .unwrap()
            .select_iter_get_seq_list_into(3, &mut seq)
            .unwrap(),
        3
    );
    assert_eq!(seq, [0, 0, 0, 1, 1, 0]);
    let mut encoded_all = Vec::new();
    Selection::All.encode1_into(&mut encoded_all).unwrap();
    assert_eq!(encoded_all, [1]);
    start.clear();
    end.clear();
    assert!(Selection::All.bounds_into(&[2, 2], &mut start, &mut end));
    assert_eq!(start, [0, 0]);
    assert_eq!(end, [1, 1]);
    assert_eq!(
        Selection::All.select_adjust_signed(&[0, 0]).unwrap(),
        Selection::All
    );
    assert_eq!(Selection::All.select_unlim_dim(&[4, u64::MAX]), Some(1));
    assert!(Selection::All.is_contiguous(&[2, 2]).unwrap());
    assert!(Selection::All.select_is_single(&[]));
    assert!(Selection::All.is_regular());
    assert_eq!(
        Selection::All.select_adjust_unsigned(&[0]).unwrap(),
        Selection::All
    );
    assert_eq!(
        Selection::All.project(&[2, 2], &[1]).unwrap(),
        Selection::Points(vec![vec![0], vec![1]])
    );

    let points = Selection::Points(vec![vec![1, 2]]);
    assert_eq!(points.selected_count(&[3, 4]), Some(1));
    assert_eq!(points.selection_type(), SelectionType::Points);
    assert!(points.select_valid(&[3, 4]));
    assert!(!points.select_valid(&[1, 1]));
    assert_eq!(points.clone(), points);
    assert!(points.select_is_single(&[3, 4]));
    start.clear();
    end.clear();
    assert!(points.bounds_into(&[3, 4], &mut start, &mut end));
    assert_eq!(start, [1, 2]);
    assert_eq!(end, [1, 2]);
    assert_eq!(
        Selection::Points(vec![vec![0]]).element_point_count(),
        Some(1)
    );

    let hyper = Selection::Hyperslab(vec![HyperslabDim::new(1, 1, 2, 1)]);
    assert_eq!(hyper.selected_count(&[5]), Some(2));
    assert!(hyper.select_is_contiguous(&[5]).unwrap());
    assert!(hyper.select_shape_same(
        &Selection::Hyperslab(vec![HyperslabDim::new(0, 1, 2, 1)]),
        &[5]
    ));
    assert_eq!(
        Selection::Hyperslab(vec![HyperslabDim::new(0, 1, 1, 1)]).selection_type(),
        SelectionType::Hyperslab
    );
}

#[test]
fn t7c_selection_encoding_replaces_stale_output_transactionally() {
    let mut encoded = vec![9, 8, 7];
    Selection::All.encode1_into(&mut encoded).unwrap();
    assert_eq!(encoded, [1]);

    encoded = vec![6, 5, 4];
    H5S_select_serialize_into(&Selection::None, &mut encoded).unwrap();
    assert_eq!(encoded, [0]);

    let malformed = Selection::Points(vec![vec![1, 2], vec![3]]);
    encoded = vec![3, 2, 1];
    let err = malformed
        .encode1_into(&mut encoded)
        .expect_err("mixed-rank point selections should fail to encode");
    assert!(err.to_string().contains("mixed ranks"));
    assert_eq!(encoded, [3, 2, 1]);

    encoded = vec![7, 7, 7];
    let err = H5S__point_serialize_into(&malformed, &mut encoded)
        .expect_err("direct point serialization should fail transactionally");
    assert!(err.to_string().contains("mixed ranks"));
    assert_eq!(encoded, [7, 7, 7]);
}

#[test]
fn t7c_selection_bounds_clear_stale_output_on_empty_and_error() {
    let mut start = vec![9, 8, 7];
    let mut end = vec![6, 5, 4];
    assert!(!H5Sget_select_bounds_into(
        &Selection::None,
        &[2, 2],
        &mut start,
        &mut end
    ));
    assert!(start.is_empty());
    assert!(end.is_empty());

    start = vec![4, 3, 2];
    end = vec![1, 0, 9];
    let malformed = Selection::Points(vec![vec![1, 2], vec![0]]);
    assert!(!H5Sget_select_bounds_into(
        &malformed,
        &[3, 3],
        &mut start,
        &mut end
    ));
    assert!(start.is_empty());
    assert!(end.is_empty());

    start = vec![7, 7, 7];
    end = vec![8, 8, 8];
    assert!(H5Sget_select_bounds_into(
        &Selection::All,
        &[2, 3],
        &mut start,
        &mut end
    ));
    assert_eq!(start, [0, 0]);
    assert_eq!(end, [1, 2]);
}

#[test]
fn t7c_hyperslab_blocklist_overwrites_copied_prefix_only() {
    let selection = Selection::Hyperslab(vec![
        HyperslabDim::new(0, 1, 2, 1),
        HyperslabDim::new(1, 1, 2, 1),
    ]);
    let mut starts = vec![9; 10];
    let mut ends = vec![8; 10];

    assert_eq!(
        H5Sget_select_hyper_blocklist_into(&selection, &[3, 3], &mut starts, &mut ends).unwrap(),
        Some(4)
    );
    assert_eq!(&starts[..8], &[0, 1, 0, 2, 1, 1, 1, 2]);
    assert_eq!(&ends[..8], &[0, 1, 0, 2, 1, 1, 1, 2]);
    assert_eq!(&starts[8..], &[9, 9]);
    assert_eq!(&ends[8..], &[8, 8]);

    let points = Selection::Points(vec![vec![0, 0]]);
    assert_eq!(
        H5Sget_select_hyper_blocklist_into(&points, &[3, 3], &mut starts, &mut ends).unwrap(),
        None
    );
    assert_eq!(&starts[..8], &[0, 1, 0, 2, 1, 1, 1, 2]);
    assert_eq!(&ends[..8], &[0, 1, 0, 2, 1, 1, 1, 2]);
}

#[test]
fn t7c_selection_iterator_and_adjust_aliases() {
    let selection = Selection::Hyperslab(vec![
        HyperslabDim::new(0, 2, 2, 1),
        HyperslabDim::new(1, 1, 2, 1),
    ]);

    let mut iter = selection.select_iter_init(&[4, 4]).unwrap();
    assert_eq!(iter.select_iter_nelmts(), 4);
    assert_eq!(iter.select_iter_next_ref(), Some(&[0, 1][..]));
    let mut seq = [0; 4];
    assert_eq!(iter.select_iter_get_seq_list_into(2, &mut seq).unwrap(), 2);
    assert_eq!(seq, [0, 2, 2, 1]);
    assert_eq!(iter.select_iter_nelmts(), 1);
    iter.select_iter_release();

    let mut visited = Vec::new();
    selection
        .visit_points(&[4, 4], |point| {
            visited.push([point[0], point[1]]);
            Ok(())
        })
        .unwrap();
    assert_eq!(visited, [[0, 1], [0, 2], [2, 1], [2, 2]]);

    assert_eq!(
        Selection::Points(vec![vec![1, 2]])
            .select_adjust_signed(&[2, -1])
            .unwrap(),
        Selection::Points(vec![vec![3, 1]])
    );
    assert!(Selection::Points(vec![vec![0]])
        .select_adjust_signed(&[-1])
        .is_err());
    assert_eq!(
        Selection::Slice(vec![hdf5_pure_rust::SliceInfo::new(1, 3)])
            .select_adjust_unsigned(&[2])
            .unwrap(),
        Selection::Slice(vec![hdf5_pure_rust::SliceInfo::new(3, 5)])
    );
    assert_eq!(
        Selection::All.select_adjust_signed(&[0, 0]).unwrap(),
        Selection::All
    );
}

#[test]
fn t7c_selection_projection_unlimited_and_fill_aliases() {
    let lhs = Selection::Points(vec![vec![0, 1, 2], vec![3, 1, 4], vec![3, 2, 4]]);
    let rhs = Selection::Points(vec![vec![3, 1, 4], vec![9, 9, 9]]);

    assert_eq!(lhs.select_unlim_dim(&[5, u64::MAX, 6]), Some(1));
    assert_eq!(
        lhs.select_num_elem_non_unlim(&[4, 3, 5], &[4, u64::MAX, 5])
            .unwrap(),
        2
    );
    assert_eq!(
        lhs.project(&[4, 3, 5], &[0, 2]).unwrap(),
        Selection::Points(vec![vec![0, 2], vec![3, 4]])
    );
    assert_eq!(
        lhs.combine_and(&rhs, &[4, 3, 5])
            .unwrap()
            .project(&[4, 3, 5], &[0, 2])
            .unwrap(),
        Selection::Points(vec![vec![3, 4]])
    );
    assert_eq!(
        lhs.combine_and(&rhs, &[4, 3, 5])
            .unwrap()
            .project(&[4, 3, 5], &[1])
            .unwrap(),
        Selection::Points(vec![vec![1]])
    );

    let mut values = vec![0i32; 6];
    Selection::Points(vec![vec![0, 1], vec![1, 2]])
        .select_fill(&[2, 3], &mut values, 7)
        .unwrap();
    assert_eq!(values, vec![0, 7, 0, 0, 0, 7]);
}

#[test]
fn t7c_selection_class_specific_internal_aliases() {
    let hyper = Selection::Hyperslab(vec![HyperslabDim::new(0, 2, 2, 1)]);
    assert!(hyper.select_valid(&[4]));
    assert_eq!(hyper.hyperslab_block_count(&[4]), Some(2));
    let mut starts = [0; 2];
    let mut ends = [0; 2];
    assert_eq!(
        hyper
            .hyperslab_blocklist_into(&[4], &mut starts, &mut ends)
            .unwrap(),
        Some(2)
    );
    assert_eq!(starts, [0, 2]);
    assert_eq!(ends, [0, 2]);
    let mut start = Vec::new();
    let mut end = Vec::new();
    assert!(hyper.bounds_into(&[4], &mut start, &mut end));
    assert_eq!(start, [0]);
    assert_eq!(end, [2]);
    assert!(hyper.is_regular());
    assert!(hyper.select_shape_same(
        &Selection::Hyperslab(vec![HyperslabDim::new(1, 1, 2, 1)]),
        &[4]
    ));
    assert_eq!(
        hyper.select_adjust_unsigned(&[1]).unwrap(),
        Selection::Hyperslab(vec![HyperslabDim::new(1, 2, 2, 1)])
    );
    assert_eq!(
        hyper.project(&[4], &[0]).unwrap(),
        Selection::Points(vec![vec![0], vec![2]])
    );

    let points = Selection::Points(vec![vec![0, 1], vec![1, 1]]);
    let mut flat_points = [0; 4];
    assert_eq!(
        points.element_pointlist_into(&mut flat_points).unwrap(),
        Some(2)
    );
    assert_eq!(flat_points, [0, 1, 1, 1]);
    let mut grown_points = points.clone();
    if let Selection::Points(ref mut coords) = grown_points {
        coords.push(vec![1, 0]);
    }
    assert_eq!(
        grown_points,
        Selection::Points(vec![vec![0, 1], vec![1, 1], vec![1, 0]])
    );
    let mut iter = points.select_iter_init(&[2, 2]).unwrap();
    assert_eq!(iter.select_iter_next_ref(), Some(&[0, 1][..]));
    let mut seq = [0; 2];
    assert_eq!(iter.select_iter_get_seq_list_into(2, &mut seq).unwrap(), 1);
    assert_eq!(seq, [1, 1]);
    assert_eq!(
        points.element_points().unwrap().collect::<Vec<_>>(),
        vec![&[0, 1][..], &[1, 1][..]]
    );
    assert!(points.select_valid(&[2, 2]));
    assert!(!points.is_regular());
    assert!(points.select_shape_same(&Selection::Points(vec![vec![9, 9], vec![8, 8]]), &[2, 2]));
    let mut intersects_first_row = false;
    points
        .visit_points(&[2, 2], |point| {
            intersects_first_row |= point[0] == 0 && point[1] <= 1;
            Ok(())
        })
        .unwrap();
    assert!(intersects_first_row);
    assert_eq!(
        points.select_adjust_signed(&[1, -1]).unwrap(),
        Selection::Points(vec![vec![1, 0], vec![2, 0]])
    );
}

#[test]
fn t7c_slice_1d_from_start() {
    let ds = open().dataset("seq100").unwrap();
    let mut vals = vec![0.0; 5];
    ds.read_slice_into::<f64, _>(..5, &mut vals).unwrap();
    assert_eq!(vals, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn t7c_slice_1d_to_end() {
    let ds = open().dataset("seq100").unwrap();
    let mut vals = vec![0.0; 5];
    ds.read_slice_into::<f64, _>(95.., &mut vals).unwrap();
    assert_eq!(vals, vec![95.0, 96.0, 97.0, 98.0, 99.0]);
}

#[test]
fn t7c_slice_all() {
    let ds = open().dataset("seq100").unwrap();
    let mut vals = vec![0.0; 100];
    ds.read_slice_into::<f64, _>(.., &mut vals).unwrap();
    assert_eq!(vals.len(), 100);
}

#[test]
fn t7d_slice_2d() {
    let ds = open().dataset("matrix").unwrap();
    // Read rows 1-3, cols 2-5
    let mut vals = vec![0; 6];
    ds.read_slice_into::<i32, _>((1..3, 2..5), &mut vals)
        .unwrap();
    // row 1: [10,11,12,13,14,15,16,17,18,19], cols 2-4 = [12,13,14]
    // row 2: [20,21,22,23,24,25,26,27,28,29], cols 2-4 = [22,23,24]
    assert_eq!(vals, vec![12, 13, 14, 22, 23, 24]);
}

#[test]
fn t7d_slice_2d_single_row() {
    let ds = open().dataset("matrix").unwrap();
    let mut vals = vec![0; 10];
    ds.read_slice_into::<i32, _>((0..1, 0..10), &mut vals)
        .unwrap();
    assert_eq!(vals, (0..10).collect::<Vec<i32>>());
}

#[test]
fn t7d_slice_3d_subregion() {
    let ds = open().dataset("cube").unwrap();
    let mut vals = vec![0; 12];
    ds.read_slice_into::<i32, _>((1..3, 2..4, 1..4), &mut vals)
        .unwrap();
    assert_eq!(vals, vec![43, 44, 45, 49, 50, 51, 73, 74, 75, 79, 80, 81]);
}

#[test]
fn t7d_slice_3d_with_full_dimension() {
    let ds = open().dataset("cube").unwrap();
    let mut vals = vec![0; 10];
    ds.read_slice_into::<i32, _>((2..3, .., 4..), &mut vals)
        .unwrap();
    assert_eq!(vals, vec![64, 65, 70, 71, 76, 77, 82, 83, 88, 89]);
}

#[test]
fn t7d_point_selection_3d() {
    let ds = open().dataset("cube").unwrap();
    let mut vals = vec![0; 3];
    ds.read_slice_into::<i32, _>(
        Selection::Points(vec![vec![3, 4, 5], vec![0, 0, 0], vec![1, 2, 3]]),
        &mut vals,
    )
    .unwrap();
    assert_eq!(vals, vec![119, 0, 45]);
}

#[test]
fn t7e_slice_chunked() {
    let ds = open().dataset("chunked_seq").unwrap();
    // Slice across chunk boundary (chunks are 25 elements)
    let mut vals = vec![0; 10];
    ds.read_slice_into::<i64, _>(20..30, &mut vals).unwrap();
    assert_eq!(vals.len(), 10);
    for (i, v) in vals.iter().enumerate() {
        assert_eq!(*v, (20 + i) as i64);
    }
}

#[test]
fn t7e_slice_chunked_all() {
    let ds = open().dataset("chunked_seq").unwrap();
    let mut vals = vec![0; 200];
    ds.read_slice_into::<i64, _>(.., &mut vals).unwrap();
    assert_eq!(vals.len(), 200);
    assert_eq!(vals[0], 0);
    assert_eq!(vals[199], 199);
}

#[test]
fn t7e_read_scalar() {
    let ds = open().dataset("scalar_val").unwrap();
    let mut vals = [0.0];
    ds.read_into(&mut vals).unwrap();
    assert_eq!(vals[0], 42.0);
}
