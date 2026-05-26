use hdf5_pure_rust::hl::types::H5Type;
use hdf5_pure_rust::File;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
struct I16ArrayCell {
    values: [i16; 6],
}

unsafe impl H5Type for I16ArrayCell {
    fn type_size() -> usize {
        std::mem::size_of::<Self>()
    }
}

#[test]
fn test_read_typed_f64() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("float64_1d").unwrap();
    let mut values = [0.0f64; 5];
    ds.read_into(&mut values).unwrap();
    assert_eq!(values, [1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_read_typed_i32() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("int32_1d").unwrap();
    let mut values = [0i32; 3];
    ds.read_into(&mut values).unwrap();
    assert_eq!(values, [10, 20, 30]);
}

#[test]
fn test_dataset_read_into_uses_caller_buffers() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("int32_1d").unwrap();

    let mut raw = [0u8; 12];
    ds.read_raw_into(&mut raw).unwrap();
    assert_eq!(raw, [10, 0, 0, 0, 20, 0, 0, 0, 30, 0, 0, 0]);

    let mut values = [0i32; 3];
    ds.read_into(&mut values).unwrap();
    assert_eq!(values, [10, 20, 30]);

    let mut narrowed = [i16::MIN; 3];
    ds.read_into(&mut narrowed).unwrap();
    assert_eq!(narrowed, [10, 20, 30]);

    let mut widened = [-1.0f64; 3];
    ds.read_into(&mut widened).unwrap();
    assert_eq!(widened, [10.0, 20.0, 30.0]);

    let mut too_short = [-7i32, -8];
    assert!(ds.read_into(&mut too_short).is_err());
    assert_eq!(too_short, [-7, -8]);
}

#[test]
fn test_dataset_read_allocating_applies_typed_conversion() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("int32_1d").unwrap();

    let narrowed = ds.read::<i16>().unwrap();
    assert_eq!(narrowed, vec![10, 20, 30]);

    let as_float = ds.read::<f32>().unwrap();
    assert_eq!(as_float, vec![10.0, 20.0, 30.0]);
}

#[test]
fn test_converted_dataset_read_into_preserves_output_on_wrong_length() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("int32_1d").unwrap();
    let mut stale = [-7i16, -8];

    let err = ds
        .read_into(&mut stale)
        .expect_err("converted full-dataset reads should reject the wrong output length");
    assert!(
        err.to_string().contains("output buffer"),
        "unexpected error: {err}"
    );
    assert_eq!(stale, [-7, -8]);
}

#[test]
fn test_unsigned_integer_conversion_read_slice_into() {
    let f = File::open("tests/data/hdf5_ref/integer_conversion_vectors.h5").unwrap();
    let ds = f.dataset("u16_conversion").unwrap();

    let mut to_signed = [0i8; 7];
    ds.read_slice_into::<i8, _>(1..8, &mut to_signed).unwrap();
    assert_eq!(to_signed, [1, 127, 127, 127, 127, 127, 127]);

    let mut narrowed = [0u8; 6];
    ds.read_slice_into::<u8, _>(3..9, &mut narrowed).unwrap();
    assert_eq!(narrowed, [128, 255, 255, 255, 255, 255]);
}

#[test]
fn test_converted_read_slice_into_preserves_output_on_wrong_length() {
    let f = File::open("tests/data/hdf5_ref/integer_conversion_vectors.h5").unwrap();
    let ds = f.dataset("u16_conversion").unwrap();
    let mut stale = [-5i8, -6, -7];

    let err = ds
        .read_slice_into::<i8, _>(1..5, &mut stale)
        .expect_err("converted selected reads should reject the wrong output length");
    assert!(
        err.to_string().contains("slice output buffer"),
        "unexpected error: {err}"
    );
    assert_eq!(stale, [-5, -6, -7]);
}

#[test]
fn test_float_conversion_read_slice_into() {
    let f = File::open("tests/data/hdf5_ref/float_conversion_vectors.h5").unwrap();
    let ds = f.dataset("f64_conversion").unwrap();

    let mut narrowed = [0.0f32; 8];
    ds.read_slice_into::<f32, _>(1..9, &mut narrowed).unwrap();
    assert_eq!(
        narrowed.map(f32::to_bits),
        [
            (-129.75f32).to_bits(),
            (-1.5f32).to_bits(),
            (-0.0f32).to_bits(),
            0.0f32.to_bits(),
            1.5f32.to_bits(),
            127.25f32.to_bits(),
            128.75f32.to_bits(),
            f32::INFINITY.to_bits(),
        ]
    );

    let mut to_integer = [0i16; 8];
    ds.read_slice_into::<i16, _>(1..9, &mut to_integer).unwrap();
    assert_eq!(to_integer, [-129, -1, 0, 0, 1, 127, 128, i16::MAX]);

    let ds = f.dataset("i16_to_float_conversion").unwrap();
    let mut to_float = [0.0f64; 5];
    ds.read_slice_into::<f64, _>(1..6, &mut to_float).unwrap();
    assert_eq!(to_float, [-1.0, 0.0, 1.0, 127.0, 128.0]);
}

#[test]
fn test_big_endian_float_conversion_read_slice_into() {
    let f = File::open("tests/data/hdf5_ref/float_conversion_vectors.h5").unwrap();
    let ds = f.dataset("be_f32_conversion").unwrap();

    let mut widened = [0.0f64; 6];
    ds.read_slice_into::<f64, _>(1..7, &mut widened).unwrap();
    assert_eq!(
        widened
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        [-129.75, -1.5, -0.0, 0.0, 1.5, 127.25]
            .iter()
            .map(|value: &f64| value.to_bits())
            .collect::<Vec<_>>()
    );
}

#[test]
fn test_read_scalar_typed() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("scalar").unwrap();
    let val: f64 = ds.read_scalar::<f64>().unwrap();
    assert_eq!(val, 42.0);

    let mut val32 = 0.0f32;
    ds.read_scalar_into(&mut val32).unwrap();
    assert_eq!(val32, 42.0);

    let narrowed: i32 = ds.read_scalar().unwrap();
    assert_eq!(narrowed, 42);
}

#[test]
fn test_read_1d_ndarray() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("float64_1d").unwrap();
    let arr = ds.read_1d::<f64>().unwrap();
    assert_eq!(arr.len(), 5);
    assert_eq!(arr[0], 1.0);
    assert_eq!(arr[4], 5.0);
}

#[test]
fn test_read_2d_ndarray() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("int8_2d").unwrap();
    let arr = ds.read_2d::<i8>().unwrap();
    assert_eq!(arr.shape(), &[2, 3]);
    assert_eq!(arr[[0, 0]], 1);
    assert_eq!(arr[[1, 2]], 6);
}

#[test]
fn test_read_dyn_ndarray_3d() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("read_dyn_3d.h5");

    {
        let mut wf = hdf5_pure_rust::WritableFile::create(&path).unwrap();
        let data: Vec<i32> = (0..24).collect();
        wf.new_dataset_builder("cube")
            .shape(&[2, 3, 4])
            .write::<i32>(&data)
            .unwrap();
        wf.flush().unwrap();
    }

    let f = File::open(&path).unwrap();
    let arr = f.dataset("cube").unwrap().read_dyn::<i32>().unwrap();
    assert_eq!(arr.shape(), &[2, 3, 4]);
    assert_eq!(arr[[0, 0, 0]], 0);
    assert_eq!(arr[[1, 2, 3]], 23);
}

#[test]
fn test_raw_message_inspection_apis() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("float64_1d").unwrap();

    let dtype = ds.raw_datatype_message().unwrap();
    assert_eq!(dtype.size, 8);
    assert_eq!(ds.dtype().unwrap().raw_message_ref().size, 8);

    let space = ds.raw_dataspace_message().unwrap();
    assert_eq!(space.dims, vec![5]);
    assert_eq!(ds.space().unwrap().raw_message_ref().dims, vec![5]);

    let plist = ds.create_plist().unwrap();
    assert!(plist.filters.is_empty());
    assert_eq!(plist.external_count(), 0);
    assert!(plist.external(0).is_none());
    assert!(plist.filter(0).is_none());
    assert!(plist.filter_by_id(1).is_none());

    let access = ds.access_plist();
    assert_eq!(
        access.virtual_view(),
        hdf5_pure_rust::VdsView::LastAvailable
    );
    assert_eq!(access.virtual_prefix(), None);
    assert_eq!(
        access.virtual_missing_source_policy(),
        hdf5_pure_rust::VdsMissingSourcePolicy::Error
    );
}

#[test]
fn test_h5type_output_helpers_clear_when_unsupported() {
    let mut fields = vec![hdf5_pure_rust::hl::types::FieldDescriptor {
        name: "stale".to_string(),
        offset: 1,
        size: 1,
        type_class: hdf5_pure_rust::hl::types::TypeClass::Integer { signed: false },
    }];
    assert!(u8::compound_fields_into(&mut fields).is_none());
    assert!(fields.is_empty());

    let mut members = vec![("stale".to_string(), 7)];
    assert!(u8::enum_members_into(&mut members).is_none());
    assert!(members.is_empty());
}

#[test]
fn test_read_chunked_typed() {
    let f = File::open("tests/data/datasets_v0.h5").unwrap();
    let ds = f.dataset("chunked").unwrap();
    let plist = ds.create_plist().unwrap();
    assert!(plist.chunk_opts().is_none() || plist.chunk_opts() == Some(0));

    let mut values = [0.0f32; 100];
    ds.read_into(&mut values).unwrap();
    assert_eq!(values.len(), 100);
    for (i, v) in values.iter().enumerate() {
        assert_eq!(*v, i as f32);
    }
}

#[test]
fn test_attr_read_typed() {
    let f = File::open("tests/data/attrs.h5").unwrap();
    let attr = f.attr("int_attr").unwrap();
    let val: i64 = attr.read_scalar::<i64>().unwrap();
    assert_eq!(val, 42);
    let narrowed: i8 = attr.read_scalar::<i8>().unwrap();
    assert_eq!(narrowed, 42);
    let as_float: f64 = attr.read_scalar::<f64>().unwrap();
    assert_eq!(as_float, 42.0);
}

#[test]
fn test_attr_read_array_typed() {
    let f = File::open("tests/data/attrs.h5").unwrap();
    let attr = f.attr("array_attr").unwrap();
    let mut values = [0.0f64; 3];
    attr.read_into(&mut values).unwrap();
    assert_eq!(values, [1.0, 2.0, 3.0]);
    let mut values32 = [0.0f32; 3];
    attr.read_into(&mut values32).unwrap();
    assert_eq!(values32, [1.0, 2.0, 3.0]);
}

#[test]
fn test_attribute_read_into_uses_caller_buffers() {
    let f = File::open("tests/data/attrs.h5").unwrap();
    let attr = f.attr("array_attr").unwrap();

    let mut raw = vec![0; attr.raw_data().len()];
    attr.read_raw_into(&mut raw).unwrap();
    assert_eq!(raw, attr.raw_data());

    let mut values = [0.0f64; 3];
    attr.read_into(&mut values).unwrap();
    assert_eq!(values, [1.0, 2.0, 3.0]);

    let mut values32 = [0.0f32; 3];
    attr.read_into(&mut values32).unwrap();
    assert_eq!(values32, [1.0, 2.0, 3.0]);

    let mut too_short = [-7.0f32, -8.0];
    let err = attr
        .read_into(&mut too_short)
        .expect_err("converted attribute reads should reject the wrong output length");
    assert!(
        err.to_string().contains("attribute typed output buffer"),
        "unexpected error: {err}"
    );
    assert_eq!(too_short, [-7.0, -8.0]);
}

#[test]
fn test_read_wrong_type_size() {
    let f = File::open("tests/data/strings.h5").unwrap();
    let ds = f.dataset("fixed_str").unwrap();
    // Fixed strings are not part of the numeric conversion table.
    let mut value = [0u64; 1];
    let result = ds.read_into(&mut value);
    assert!(result.is_err());
}

#[test]
fn test_read_array_datatype_into_matching_typed_cells() {
    let f = File::open("tests/data/hdf5_ref/array_datatype_cases.h5").unwrap();
    let ds = f.dataset("array_i16_2x3").unwrap();

    let mut cells = [I16ArrayCell { values: [0; 6] }; 2];
    ds.read_into(&mut cells).unwrap();
    assert_eq!(
        cells,
        [
            I16ArrayCell {
                values: [0, 1, 2, 3, 4, 5]
            },
            I16ArrayCell {
                values: [6, 7, 8, 9, 10, 11]
            }
        ]
    );

    let mut flattened = [-1i16; 12];
    let err = ds
        .read_into(&mut flattened)
        .expect_err("array datatype reads should preserve dataspace element boundaries");
    assert!(
        err.to_string().contains("requested element size 2"),
        "unexpected error: {err}"
    );
    assert_eq!(flattened, [-1; 12]);
}
