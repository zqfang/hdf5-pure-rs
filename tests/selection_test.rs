use hdf5_pure_rust::{
    hl::selection::{Selection, SliceInfo},
    File, HyperslabDim, SelectionType,
};

fn assert_hyperslab_blocks_eq_into(
    selection: &Selection,
    shape: &[u64],
    starts: &mut Vec<u64>,
    ends: &mut Vec<u64>,
    expected: Option<(&[u64], &[u64])>,
) -> hdf5_pure_rust::Result<()> {
    let Some(block_count) = selection.hyperslab_block_count(shape) else {
        assert!(expected.is_none());
        return Ok(());
    };
    let rank = match selection {
        Selection::None => 0,
        Selection::All => shape.len(),
        Selection::Hyperslab(dims) => dims.len(),
        Selection::Slice(slices) => slices.len(),
        Selection::Points(_) => unreachable!("point selection block count is None"),
    };
    starts.resize(block_count as usize * rank, 0);
    ends.resize(block_count as usize * rank, 0);
    assert_eq!(
        selection.hyperslab_blocklist_into(shape, starts, ends)?,
        Some(block_count as usize)
    );
    let Some((expected_starts, expected_ends)) = expected else {
        panic!("expected hyperslab blocklist for non-point selection");
    };
    assert_eq!(starts, expected_starts);
    assert_eq!(ends, expected_ends);
    Ok(())
}

fn assert_bounds_eq(
    selection: &Selection,
    shape: &[u64],
    start: &mut Vec<u64>,
    end: &mut Vec<u64>,
    expected: Option<(&[u64], &[u64])>,
) {
    assert_eq!(selection.bounds_into(shape, start, end), expected.is_some());
    if let Some((expected_start, expected_end)) = expected {
        assert_eq!(start, expected_start);
        assert_eq!(end, expected_end);
    } else {
        assert!(start.is_empty());
        assert!(end.is_empty());
    }
}

#[test]
fn test_read_slice_1d_range() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("float64_1d").unwrap();
    // Read elements 1..4 (indices 1, 2, 3)
    let mut vals = [0.0; 3];
    ds.read_slice_into::<f64, _>(1..4, &mut vals).unwrap();
    assert_eq!(vals, [2.0, 3.0, 4.0]);
}

#[test]
fn test_read_slice_into_1d_range() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("float64_1d").unwrap();
    let mut vals = [0.0; 3];
    ds.read_slice_into::<f64, _>(1..4, &mut vals).unwrap();
    assert_eq!(vals, [2.0, 3.0, 4.0]);
}

#[test]
fn test_read_slice_into_rejects_wrong_output_length() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("float64_1d").unwrap();
    let mut vals = [0.0; 2];
    let err = ds
        .read_slice_into::<f64, _>(1..4, &mut vals)
        .expect_err("output length should be validated");
    assert!(err.to_string().contains("expected 3"));
}

#[test]
fn test_read_slice_into_rejects_nonempty_output_for_empty_selection() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("float64_1d").unwrap();
    let mut vals = [42.0];
    let err = ds
        .read_slice_into::<f64, _>(Selection::None, &mut vals)
        .expect_err("empty selections should require an empty output buffer");
    assert!(err.to_string().contains("expected 0"));
    assert_eq!(vals, [42.0]);
}

#[test]
fn test_read_slice_1d_from_start() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("float64_1d").unwrap();
    let mut vals = [0.0; 3];
    ds.read_slice_into::<f64, _>(..3, &mut vals).unwrap();
    assert_eq!(vals, [1.0, 2.0, 3.0]);
}

#[test]
fn test_read_slice_1d_to_end() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("float64_1d").unwrap();
    let mut vals = [0.0; 2];
    ds.read_slice_into::<f64, _>(3.., &mut vals).unwrap();
    assert_eq!(vals, [4.0, 5.0]);
}

#[test]
fn test_read_slice_all() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("float64_1d").unwrap();
    let mut vals = [0.0; 5];
    ds.read_slice_into::<f64, _>(.., &mut vals).unwrap();
    assert_eq!(vals, [1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_read_slice_none() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("float64_1d").unwrap();
    let mut vals: Vec<f64> = Vec::new();
    ds.read_slice_into::<f64, _>(Selection::None, &mut vals)
        .unwrap();
    assert!(vals.is_empty());
}

#[test]
fn test_read_slice_zero_length_range() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("float64_1d").unwrap();
    let mut vals: Vec<f64> = Vec::new();
    ds.read_slice_into::<f64, _>(2..2, &mut vals).unwrap();
    assert!(vals.is_empty());
}

#[test]
fn test_read_slice_1d_stepped_selection() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("float64_1d").unwrap();
    let selection = Selection::Slice(vec![SliceInfo::with_step(0, 5, 2)]);
    let mut vals = [0.0; 3];
    ds.read_slice_into::<f64, _>(selection, &mut vals).unwrap();
    assert_eq!(vals, [1.0, 3.0, 5.0]);
}

#[test]
fn test_read_slice_1d_point_selection() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("float64_1d").unwrap();
    let selection = Selection::Points(vec![vec![4], vec![0], vec![2]]);
    let mut vals = [0.0; 3];
    ds.read_slice_into::<f64, _>(selection, &mut vals).unwrap();
    assert_eq!(vals, [5.0, 1.0, 3.0]);
}

#[test]
fn test_read_slice_1d_block_hyperslab_selection() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("float64_1d").unwrap();
    let selection = Selection::Hyperslab(vec![HyperslabDim::new(0, 3, 2, 2)]);
    let mut vals = [0.0; 4];
    ds.read_slice_into::<f64, _>(selection, &mut vals).unwrap();
    assert_eq!(vals, [1.0, 2.0, 4.0, 5.0]);
}

#[test]
fn test_read_slice_2d() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("int8_2d").unwrap();
    // Read row 1 only (row 1, all columns)
    let mut vals = [0; 3];
    ds.read_slice_into::<i8, _>((1..2, 0..3), &mut vals)
        .unwrap();
    assert_eq!(vals, [4, 5, 6]);
}

#[test]
fn test_read_slice_2d_range_full_column() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("int8_2d").unwrap();
    let mut vals = [0; 4];
    ds.read_slice_into::<i8, _>((.., 1..3), &mut vals).unwrap();
    assert_eq!(vals, [2, 3, 5, 6]);
}

#[test]
fn test_read_slice_2d_range_full_row() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("int8_2d").unwrap();
    let mut vals = [0; 3];
    ds.read_slice_into::<i8, _>((1..2, ..), &mut vals).unwrap();
    assert_eq!(vals, [4, 5, 6]);
}

#[test]
fn test_read_slice_2d_full_tuple() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("int8_2d").unwrap();
    let mut vals = [0; 6];
    ds.read_slice_into::<i8, _>((.., ..), &mut vals).unwrap();
    assert_eq!(vals, [1, 2, 3, 4, 5, 6]);
}

#[test]
fn test_read_slice_2d_open_ended_tuple_ranges() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("int8_2d").unwrap();
    let mut vals = [0; 2];
    ds.read_slice_into::<i8, _>((1.., ..2), &mut vals).unwrap();
    assert_eq!(vals, [4, 5]);
}

#[test]
fn test_read_slice_2d_subregion() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("int8_2d").unwrap();
    // Read rows 0-1, cols 1-2
    let mut vals = [0; 4];
    ds.read_slice_into::<i8, _>((0..2, 1..3), &mut vals)
        .unwrap();
    assert_eq!(vals, [2, 3, 5, 6]);
}

#[test]
fn test_read_slice_2d_stepped_selection() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("int8_2d").unwrap();
    let selection = Selection::Slice(vec![
        SliceInfo::with_step(0, 2, 1),
        SliceInfo::with_step(0, 3, 2),
    ]);
    let mut vals = [0; 4];
    ds.read_slice_into::<i8, _>(selection, &mut vals).unwrap();
    assert_eq!(vals, [1, 3, 4, 6]);
}

#[test]
fn test_read_slice_2d_point_selection() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("int8_2d").unwrap();
    let selection = Selection::Points(vec![vec![1, 2], vec![0, 0], vec![1, 1]]);
    let mut vals = [0; 3];
    ds.read_slice_into::<i8, _>(selection, &mut vals).unwrap();
    assert_eq!(vals, [6, 1, 5]);
}

#[test]
fn test_read_slice_2d_point_selection_preserves_order_and_duplicates() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("int8_2d").unwrap();
    let selection = Selection::Points(vec![vec![1, 2], vec![1, 2], vec![0, 1], vec![1, 0]]);
    let mut vals = [0; 4];
    ds.read_slice_into::<i8, _>(selection, &mut vals).unwrap();
    assert_eq!(vals, [6, 6, 2, 4]);
}

#[test]
fn test_read_cell_uses_point_selection_and_preserves_stale_output_on_error() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("int8_2d").unwrap();

    assert_eq!(ds.read_cell::<i8>(&[1, 2]).unwrap(), 6);

    let mut value = -1;
    ds.read_cell_into::<i8>(&[0, 1], &mut value).unwrap();
    assert_eq!(value, 2);

    let err = ds
        .read_cell_into::<i8>(&[2, 0], &mut value)
        .expect_err("out-of-bounds cell read should fail");
    assert!(
        err.to_string().contains("exceeds extent"),
        "unexpected error: {err}"
    );
    assert_eq!(value, 2);
}

#[test]
fn test_read_slice_2d_block_hyperslab_selection() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("int8_2d").unwrap();
    let selection = Selection::Hyperslab(vec![
        HyperslabDim::new(0, 1, 2, 1),
        HyperslabDim::new(0, 2, 2, 1),
    ]);
    let mut vals = [0; 4];
    ds.read_slice_into::<i8, _>(selection, &mut vals).unwrap();
    assert_eq!(vals, [1, 3, 4, 6]);
}

#[test]
fn test_selection_bounds_count_and_regularity_helpers() {
    let shape = [2, 3];
    let mut start = Vec::new();
    let mut end = Vec::new();
    let mut block_starts = Vec::new();
    let mut block_ends = Vec::new();

    let all = Selection::All;
    assert!(all.is_all());
    assert_eq!(all.selection_type(), SelectionType::All);
    assert_eq!(all.selected_count(&shape), Some(6));
    assert_bounds_eq(&all, &shape, &mut start, &mut end, Some((&[0, 0], &[1, 2])));
    assert_eq!(all.hyperslab_block_count(&shape), Some(1));
    assert_hyperslab_blocks_eq_into(
        &all,
        &shape,
        &mut block_starts,
        &mut block_ends,
        Some((&[0, 0], &[1, 2])),
    )
    .unwrap();
    assert!(all.is_regular());

    let none = Selection::None;
    assert!(none.is_none());
    assert_eq!(none.selection_type(), SelectionType::None);
    assert_eq!(none.selected_count(&shape), Some(0));
    assert_bounds_eq(&none, &shape, &mut start, &mut end, None);
    assert_eq!(none.hyperslab_block_count(&shape), Some(0));
    assert_hyperslab_blocks_eq_into(
        &none,
        &shape,
        &mut block_starts,
        &mut block_ends,
        Some((&[], &[])),
    )
    .unwrap();
    assert!(none.is_regular());

    let points = Selection::Points(vec![vec![1, 2], vec![0, 1], vec![1, 0]]);
    assert_eq!(points.selection_type(), SelectionType::Points);
    assert_eq!(points.selected_count(&shape), Some(3));
    assert_bounds_eq(
        &points,
        &shape,
        &mut start,
        &mut end,
        Some((&[0, 0], &[1, 2])),
    );
    assert_eq!(points.hyperslab_block_count(&shape), None);
    assert_eq!(
        points
            .hyperslab_blocklist_into(&shape, &mut block_starts, &mut block_ends)
            .unwrap(),
        None
    );
    assert_eq!(points.element_point_count(), Some(3));
    assert_eq!(
        points.element_points().unwrap().collect::<Vec<_>>(),
        vec![&[1, 2][..], &[0, 1][..], &[1, 0][..]]
    );
    let mut pointlist = [0; 6];
    assert_eq!(
        points.element_pointlist_into(&mut pointlist).unwrap(),
        Some(3)
    );
    assert_eq!(pointlist, [1, 2, 0, 1, 1, 0]);
    assert!(!points.is_regular());

    let slices = Selection::Slice(vec![
        SliceInfo::with_step(0, 2, 1),
        SliceInfo::with_step(0, 3, 2),
    ]);
    assert_eq!(slices.selection_type(), SelectionType::Hyperslab);
    assert_eq!(slices.selected_count(&shape), Some(4));
    assert_bounds_eq(
        &slices,
        &shape,
        &mut start,
        &mut end,
        Some((&[0, 0], &[1, 2])),
    );
    assert_eq!(slices.hyperslab_block_count(&shape), Some(2));
    assert_hyperslab_blocks_eq_into(
        &slices,
        &shape,
        &mut block_starts,
        &mut block_ends,
        Some((&[0, 0, 0, 2], &[1, 0, 1, 2])),
    )
    .unwrap();
    assert_eq!(slices.element_point_count(), None);
    assert!(slices.is_regular());

    let hyperslab = Selection::Hyperslab(vec![
        HyperslabDim::new(0, 1, 2, 1),
        HyperslabDim::new(0, 2, 2, 1),
    ]);
    assert_eq!(hyperslab.selection_type(), SelectionType::Hyperslab);
    assert_eq!(hyperslab.selected_count(&shape), Some(4));
    assert_bounds_eq(
        &hyperslab,
        &shape,
        &mut start,
        &mut end,
        Some((&[0, 0], &[1, 2])),
    );
    assert_eq!(hyperslab.hyperslab_block_count(&shape), Some(4));
    assert_hyperslab_blocks_eq_into(
        &hyperslab,
        &shape,
        &mut block_starts,
        &mut block_ends,
        Some((&[0, 0, 0, 2, 1, 0, 1, 2], &[0, 0, 0, 2, 1, 0, 1, 2])),
    )
    .unwrap();
    assert!(hyperslab.is_regular());
}

#[test]
fn test_selection_materialize_and_combine_helpers() {
    let shape = [2, 3];
    let left = Selection::Slice(vec![SliceInfo::new(0, 2), SliceInfo::with_step(0, 3, 2)]);
    let right = Selection::Points(vec![vec![0, 1], vec![1, 2]]);

    let mut visited = Vec::new();
    left.visit_points(&shape, |point| {
        visited.push(point.to_vec());
        Ok(())
    })
    .unwrap();
    assert_eq!(
        visited,
        vec![vec![0, 0], vec![0, 2], vec![1, 0], vec![1, 2]]
    );

    assert_eq!(
        left.combine_or(&right, &shape).unwrap(),
        Selection::Points(vec![
            vec![0, 0],
            vec![0, 1],
            vec![0, 2],
            vec![1, 0],
            vec![1, 2],
        ])
    );
    assert_eq!(
        left.combine_and(&right, &shape).unwrap(),
        Selection::Points(vec![vec![1, 2]])
    );
    assert_eq!(
        left.combine_xor(&right, &shape).unwrap(),
        Selection::Points(vec![vec![0, 0], vec![0, 1], vec![0, 2], vec![1, 0]])
    );
    assert_eq!(
        left.combine_and_not(&right, &shape).unwrap(),
        Selection::Points(vec![vec![0, 0], vec![0, 2], vec![1, 0]])
    );
}

#[test]
fn test_selection_linear_bounds_and_contiguity_helpers() {
    let shape = [3, 4];

    let all = Selection::All;
    assert_eq!(all.linear_bounds(&shape).unwrap(), Some((0, 11)));
    assert!(all.is_contiguous(&shape).unwrap());

    let none = Selection::None;
    assert_eq!(none.linear_bounds(&shape).unwrap(), None);
    assert!(none.is_contiguous(&shape).unwrap());

    let row = Selection::Slice(vec![SliceInfo::new(1, 2), SliceInfo::new(0, 4)]);
    assert_eq!(row.linear_bounds(&shape).unwrap(), Some((4, 7)));
    assert!(row.is_contiguous(&shape).unwrap());

    let column = Selection::Slice(vec![SliceInfo::new(0, 3), SliceInfo::new(1, 2)]);
    assert_eq!(column.linear_bounds(&shape).unwrap(), Some((1, 9)));
    assert!(!column.is_contiguous(&shape).unwrap());

    let explicit = Selection::Points(vec![vec![0, 2], vec![0, 1], vec![0, 3]]);
    assert_eq!(explicit.linear_bounds(&shape).unwrap(), Some((1, 3)));
    assert!(explicit.is_contiguous(&shape).unwrap());

    let duplicate = Selection::Points(vec![vec![0, 1], vec![0, 1]]);
    assert!(!duplicate.is_contiguous(&shape).unwrap());
}

#[test]
fn test_selection_point_iterator_and_projection_helpers() {
    let shape = [2, 3];
    let selection = Selection::Slice(vec![SliceInfo::new(0, 2), SliceInfo::with_step(0, 3, 2)]);

    let mut iter = selection.select_iter_init(&shape).unwrap();
    assert_eq!(iter.len(), 4);
    assert_eq!(iter.select_iter_next_ref(), Some(&[0, 0][..]));
    let mut coord = [0, 0];
    assert!(iter.select_iter_next_into(&mut coord).unwrap());
    assert_eq!(coord, [0, 2]);
    assert_eq!(iter.len(), 2);
    assert_eq!(iter.collect::<Vec<_>>(), vec![vec![1, 0], vec![1, 2]]);

    assert_eq!(
        selection.project(&shape, &[0]).unwrap(),
        Selection::Points(vec![vec![0], vec![1]])
    );
    assert_eq!(
        selection.project(&shape, &[1]).unwrap(),
        Selection::Points(vec![vec![0], vec![2]])
    );
    assert_eq!(
        selection.project(&shape, &[1, 0]).unwrap(),
        Selection::Points(vec![vec![0, 0], vec![0, 1], vec![2, 0], vec![2, 1]])
    );

    let err = selection
        .project(&shape, &[2])
        .expect_err("projection should reject out-of-rank dimensions");
    assert!(err.to_string().contains("out of bounds"));
}

#[test]
fn test_selection_bounds_reject_coordinate_overflow() {
    let mut start = Vec::new();
    let mut end = Vec::new();

    let slice = Selection::Slice(vec![SliceInfo::with_step(u64::MAX - 1, u64::MAX, 2)]);
    assert_bounds_eq(
        &slice,
        &[u64::MAX],
        &mut start,
        &mut end,
        Some((&[u64::MAX - 1], &[u64::MAX - 1])),
    );

    let zero_step_slice = Selection::Slice(vec![SliceInfo::with_step(1, u64::MAX, 0)]);
    assert_bounds_eq(&zero_step_slice, &[u64::MAX], &mut start, &mut end, None);

    let overflowing_hyperslab =
        Selection::Hyperslab(vec![HyperslabDim::new(u64::MAX - 1, 2, 2, 1)]);
    assert_bounds_eq(
        &overflowing_hyperslab,
        &[u64::MAX],
        &mut start,
        &mut end,
        None,
    );
}

#[test]
fn test_read_slice_chunked() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("chunked").unwrap();
    // Read elements 50..55 from a chunked dataset
    let mut vals = [0.0; 5];
    ds.read_slice_into::<f32, _>(50..55, &mut vals).unwrap();
    assert_eq!(vals, [50.0, 51.0, 52.0, 53.0, 54.0]);
}
