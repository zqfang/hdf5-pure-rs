//! Phase T3: Datatype tests -- read all primitive, compound, enum, string types.

use hdf5_pure_rust::format::messages::datatype::{DatatypeClass, FloatFields};
use hdf5_pure_rust::hl::types::H5Type;
use hdf5_pure_rust::{Dataset, File};

const FILE: &str = "tests/data/hdf5_ref/all_dtypes.h5";

fn open() -> File {
    File::open(FILE).expect("failed to open all_dtypes.h5")
}

fn read_array<T: H5Type + Default, const N: usize>(ds: &Dataset) -> hdf5_pure_rust::Result<[T; N]> {
    let mut values = [T::default(); N];
    ds.read_into(&mut values)?;
    Ok(values)
}

fn read_raw_array<const N: usize>(ds: &Dataset) -> hdf5_pure_rust::Result<[u8; N]> {
    let mut raw = [0; N];
    ds.read_raw_into(&mut raw)?;
    Ok(raw)
}

fn assert_strings<const N: usize>(ds: &Dataset, expected: [&str; N]) {
    let mut index = 0;
    ds.visit_strings(|value| {
        assert_eq!(value, expected[index]);
        index += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(index, N);
}

fn read_field_array<T: H5Type + Default, const N: usize>(
    ds: &Dataset,
    field_name: &str,
) -> hdf5_pure_rust::Result<[T; N]> {
    let mut values = [T::default(); N];
    ds.read_field_into(field_name, &mut values)?;
    Ok(values)
}

// T3a: Primitive integer types

#[test]
fn t3a_int8() {
    let f = open();
    let vals: [i8; 5] = read_array(&f.dataset("int8").unwrap()).unwrap();
    assert_eq!(vals, [0, 1, 2, 3, 4]);
}

#[test]
fn t3a_int16() {
    let vals: [i16; 5] = read_array(&open().dataset("int16").unwrap()).unwrap();
    assert_eq!(vals, [0, 1, 2, 3, 4]);
}

#[test]
fn t3a_int32() {
    let vals: [i32; 5] = read_array(&open().dataset("int32").unwrap()).unwrap();
    assert_eq!(vals, [0, 1, 2, 3, 4]);
}

#[test]
fn t3a_int64() {
    let vals: [i64; 5] = read_array(&open().dataset("int64").unwrap()).unwrap();
    assert_eq!(vals, [0, 1, 2, 3, 4]);
}

#[test]
fn t3a_uint8() {
    let vals: [u8; 5] = read_array(&open().dataset("uint8").unwrap()).unwrap();
    assert_eq!(vals, [0, 1, 2, 3, 4]);
}

#[test]
fn t3a_uint16() {
    let vals: [u16; 5] = read_array(&open().dataset("uint16").unwrap()).unwrap();
    assert_eq!(vals, [0, 1, 2, 3, 4]);
}

#[test]
fn t3a_uint32() {
    let vals: [u32; 5] = read_array(&open().dataset("uint32").unwrap()).unwrap();
    assert_eq!(vals, [0, 1, 2, 3, 4]);
}

#[test]
fn t3a_uint64() {
    let vals: [u64; 5] = read_array(&open().dataset("uint64").unwrap()).unwrap();
    assert_eq!(vals, [0, 1, 2, 3, 4]);
}

#[test]
fn t3a_integer_datatype_precision_and_offset() {
    let dtype = open().dataset("int32").unwrap().dtype().unwrap();

    assert_eq!(dtype.bit_offset(), Some(0));
    assert_eq!(dtype.precision(), Some(32));
    assert_eq!(dtype.create_plist().low_pad(), 0);
    assert_eq!(dtype.create_plist().high_pad(), 0);
    assert_eq!(dtype.pad(), Some((0, 0)));
    assert_eq!(dtype.native_type().size(), dtype.size());
}

#[test]
fn t3a_integer_signed_widening_and_narrowing() {
    let f = File::open("tests/data/hdf5_ref/integer_conversion_vectors.h5").unwrap();
    let ds = f.dataset("i16_conversion").unwrap();

    let widened: [i32; 9] = read_array(&ds).unwrap();
    assert_eq!(widened, [-129, -1, 0, 1, 127, 128, 255, 256, 32767]);

    let narrowed: [i8; 9] = read_array(&ds).unwrap();
    assert_eq!(narrowed, [-128, -1, 0, 1, 127, 127, 127, 127, 127]);

    let to_unsigned: [u8; 9] = read_array(&ds).unwrap();
    assert_eq!(to_unsigned, [0, 0, 0, 1, 127, 128, 255, 255, 255]);
}

#[test]
fn t3a_integer_unsigned_widening_and_narrowing() {
    let f = File::open("tests/data/hdf5_ref/integer_conversion_vectors.h5").unwrap();
    let ds = f.dataset("u16_conversion").unwrap();

    let widened: [u32; 9] = read_array(&ds).unwrap();
    assert_eq!(widened, [0, 1, 127, 128, 255, 256, 32767, 32768, 65535]);

    let to_signed: [i8; 9] = read_array(&ds).unwrap();
    assert_eq!(to_signed, [0, 1, 127, 127, 127, 127, 127, 127, 127]);

    let narrowed: [u8; 9] = read_array(&ds).unwrap();
    assert_eq!(narrowed, [0, 1, 127, 128, 255, 255, 255, 255, 255]);
}

#[test]
fn t3a_float32() {
    let vals: [f32; 3] = read_array(&open().dataset("float32").unwrap()).unwrap();
    assert_eq!(vals, [1.5, 2.5, 3.5]);
}

#[test]
fn t3a_float32_field_metadata() {
    let dtype = open().dataset("float32").unwrap().dtype().unwrap();

    assert_eq!(dtype.bit_offset(), Some(0));
    assert_eq!(dtype.precision(), Some(32));
    assert_eq!(
        dtype.float_fields(),
        Some(FloatFields {
            sign_position: 31,
            exponent_position: 23,
            exponent_size: 8,
            mantissa_position: 0,
            mantissa_size: 23,
        })
    );
    assert_eq!(dtype.exponent_bias(), Some(127));
    assert_eq!(dtype.mantissa_normalization(), Some(2));
    assert_eq!(dtype.internal_padding(), Some(0));
    assert_eq!(dtype.pad(), Some((0, 0)));
}

#[test]
fn t3a_float64() {
    let vals: [f64; 3] = read_array(&open().dataset("float64").unwrap()).unwrap();
    assert_eq!(vals, [1.5, 2.5, 3.5]);
}

#[test]
fn t3a_float_widening_narrowing_nan_inf_and_endian() {
    let f = File::open("tests/data/hdf5_ref/float_conversion_vectors.h5").unwrap();

    let f32_to_f64: [f64; 10] = read_array(&f.dataset("f32_conversion").unwrap()).unwrap();
    let expected_f64 = [
        f64::NEG_INFINITY,
        -129.75,
        -1.5,
        -0.0,
        0.0,
        1.5,
        127.25,
        128.75,
        f64::INFINITY,
        f64::NAN,
    ];
    assert_eq!(
        f32_to_f64
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected_f64
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );

    let f64_to_f32: [f32; 10] = read_array(&f.dataset("f64_conversion").unwrap()).unwrap();
    let expected_f32 = [
        f32::NEG_INFINITY,
        -129.75,
        -1.5,
        -0.0,
        0.0,
        1.5,
        127.25,
        128.75,
        f32::INFINITY,
        f32::NAN,
    ];
    assert_eq!(
        f64_to_f32
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected_f32
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );

    let be_f32_to_f64: [f64; 10] = read_array(&f.dataset("be_f32_conversion").unwrap()).unwrap();
    assert_eq!(
        be_f32_to_f64
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected_f64
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
}

#[test]
fn t3a_integer_to_float_conversions() {
    let f = File::open("tests/data/hdf5_ref/float_conversion_vectors.h5").unwrap();

    let signed: [f32; 8] = read_array(&f.dataset("i16_to_float_conversion").unwrap()).unwrap();
    assert_eq!(
        signed,
        [-129.0, -1.0, 0.0, 1.0, 127.0, 128.0, 255.0, 32767.0]
    );

    let unsigned: [f64; 8] = read_array(&f.dataset("u16_to_float_conversion").unwrap()).unwrap();
    assert_eq!(
        unsigned,
        [0.0, 1.0, 127.0, 128.0, 255.0, 256.0, 32767.0, 65535.0]
    );
}

#[test]
fn t3a_float_to_integer_conversions() {
    let f = File::open("tests/data/hdf5_ref/float_conversion_vectors.h5").unwrap();
    let ds = f.dataset("f64_conversion").unwrap();

    let signed: [i8; 10] = read_array(&ds).unwrap();
    assert_eq!(signed, [-128, -128, -1, 0, 0, 1, 127, 127, 127, 0]);

    let unsigned: [u8; 10] = read_array(&ds).unwrap();
    assert_eq!(unsigned, [0, 0, 0, 0, 0, 1, 127, 128, 255, 0]);
}

// T3a: Big-endian variants

#[test]
fn t3a_be_float64() {
    let vals: [f64; 3] = read_array(&open().dataset("be_f8").unwrap()).unwrap();
    assert_eq!(vals, [10.0, 20.0, 30.0]);
}

#[test]
fn t3a_be_int32() {
    let vals: [i32; 3] = read_array(&open().dataset("be_i4").unwrap()).unwrap();
    assert_eq!(vals, [10, 20, 30]);
}

#[test]
fn t3a_be_uint16() {
    let vals: [u16; 3] = read_array(&open().dataset("be_u2").unwrap()).unwrap();
    assert_eq!(vals, [10, 20, 30]);
}

// T3b: Compound type

#[test]
fn t3b_compound_fields() {
    let ds = open().dataset("compound").unwrap();
    let dtype = ds.dtype().unwrap();
    let field_views = dtype
        .compound_fields_iter()
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(field_views.len(), 3);
    assert_eq!(field_views[0].name, "x");
    assert_eq!(field_views[1].raw_name, b"y");
    assert_eq!(field_views[2].name, "flag");
    let owned_fields = dtype.compound_fields().unwrap();
    assert_eq!(owned_fields.len(), field_views.len());
    assert_eq!(owned_fields[0].name, "x");
    assert_eq!(owned_fields[1].name, "y");
    assert_eq!(owned_fields[2].name, "flag");
    assert_eq!(dtype.member_index("x"), Some(0));
    assert_eq!(dtype.member_index("y"), Some(1));
    assert_eq!(dtype.member_index("flag"), Some(2));
    assert_eq!(dtype.member_index("missing"), None);
    for (index, field) in field_views.iter().enumerate() {
        assert_eq!(dtype.member_offset(index), Some(field.byte_offset));
        assert_eq!(dtype.member_class(index), Some(field.class));
        assert_eq!(
            dtype.member_type(index).map(|member| member.size()),
            Some(field.size)
        );
    }
    assert_eq!(dtype.member_offset(field_views.len()), None);
    assert_eq!(dtype.member_class(field_views.len()), None);
    assert!(dtype.member_type(field_views.len()).is_none());
}

#[test]
fn t3b_compound_read_field() {
    let ds = open().dataset("compound").unwrap();
    let x: [f64; 2] = read_field_array(&ds, "x").unwrap();
    assert_eq!(x, [1.0, 3.0]);
}

// T3c: Enum type

#[test]
fn t3c_enum_members() {
    let ds = open().dataset("enum").unwrap();
    let dt = ds.dtype().unwrap();
    let member_views = dt
        .enum_members_iter()
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert!(member_views
        .iter()
        .any(|member| member.name == "OFF" && member.value == 0));
    assert!(member_views
        .iter()
        .any(|member| member.name == "ON" && member.value == 1));
    assert!(member_views
        .iter()
        .any(|member| member.name == "AUTO" && member.value == 2));
    let owned_members = dt.enum_members().unwrap();
    assert!(owned_members
        .iter()
        .any(|member| member.0 == "OFF" && member.1 == 0));
    assert!(owned_members
        .iter()
        .any(|member| member.0 == "ON" && member.1 == 1));
    assert!(owned_members
        .iter()
        .any(|member| member.0 == "AUTO" && member.1 == 2));
}

#[test]
fn t3c_enum_values() {
    let vals: [u8; 3] = read_array(&open().dataset("enum").unwrap()).unwrap();
    assert_eq!(vals, [0, 1, 2]);
}

#[test]
fn t3c_enum_u16_big_endian_members_and_values() {
    let f = File::open("tests/data/hdf5_ref/enum_conversion_cases.h5").unwrap();
    let ds = f.dataset("enum_u16be").unwrap();
    let dt = ds.dtype().unwrap();
    let members = dt
        .enum_members_iter()
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert!(members
        .iter()
        .any(|member| member.name == "LOW" && member.value == 1));
    assert!(members
        .iter()
        .any(|member| member.name == "MID" && member.value == 258));
    assert!(members
        .iter()
        .any(|member| member.name == "HIGH" && member.value == 4095));

    let vals: [u16; 3] = read_array(&ds).unwrap();
    assert_eq!(vals, [1, 258, 4095]);
}

// T3d: String types

#[test]
fn t3d_fixed_string() {
    let ds = open().dataset("fstring").unwrap();
    assert_strings(&ds, ["hello", "world"]);
}

#[test]
fn t3d_fixed_string_padding_and_charset_flags() {
    let f = File::open("tests/data/hdf5_ref/fixed_string_cases.h5").unwrap();

    let null_padded = f.dataset("null_padded").unwrap();
    assert_eq!(null_padded.dtype().unwrap().string_padding(), Some(1));
    assert_eq!(null_padded.dtype().unwrap().char_set(), Some(0));
    assert_strings(&null_padded, ["hi", "a b", "trail "]);

    let space_padded = f.dataset("space_padded").unwrap();
    assert_eq!(space_padded.dtype().unwrap().string_padding(), Some(2));
    assert_strings(&space_padded, ["hi", "a b", "trail"]);

    let null_terminated = f.dataset("null_terminated").unwrap();
    assert_eq!(null_terminated.dtype().unwrap().string_padding(), Some(0));
    assert_strings(&null_terminated, ["hi", "a b", "trail "]);

    let utf8 = f.dataset("utf8_fixed").unwrap();
    assert_eq!(utf8.dtype().unwrap().char_set(), Some(1));
    assert_strings(&utf8, ["å", "猫", "hi"]);
}

#[test]
fn t3d_vlen_string() {
    let ds = open().dataset("vstring").unwrap();
    assert_strings(&ds, ["alpha", "beta", "gamma"]);
}

#[test]
fn t3d_vlen_string_empty_null_utf8_and_heap_edges() {
    let f = File::open("tests/data/hdf5_ref/vlen_string_cases.h5").unwrap();

    let utf8 = f.dataset("vlen_utf8_strings").unwrap();
    assert_strings(&utf8, ["", "猫", "å", "alpha"]);

    let heap_edges = f.dataset("vlen_global_heap_edges").unwrap();
    let long = format!("long-{}", "x".repeat(96));
    assert_strings(&heap_edges, ["dup", "dup", long.as_str()]);

    let null_descriptor = f.dataset("vlen_null_descriptor").unwrap();
    assert_strings(&null_descriptor, ["", "kept"]);
}

#[test]
fn t3d_opaque_tag_and_raw_payload() {
    let f = File::open("tests/data/hdf5_ref/opaque_cases.h5").unwrap();
    let ds = f.dataset("opaque_tagged").unwrap();
    let dtype = ds.dtype().unwrap();

    assert_eq!(dtype.class(), DatatypeClass::Opaque);
    assert_eq!(dtype.size(), 4);
    assert_eq!(
        dtype.opaque_tag_str(),
        Some("hdf5-pure-rust opaque fixture")
    );
    assert_eq!(
        read_raw_array::<12>(&ds).unwrap().as_slice(),
        b"abcd\x00\x01\x02\x03wxyz"
    );
}

#[test]
fn t3d_reference_object_and_region_payloads() {
    let f = File::open("tests/data/hdf5_ref/reference_cases.h5").unwrap();

    let object_refs = f.dataset("object_refs").unwrap();
    let object_dtype = object_refs.dtype().unwrap();
    assert_eq!(object_dtype.class(), DatatypeClass::Reference);
    assert_eq!(object_dtype.size(), 8);
    assert_eq!(object_dtype.reference_type(), Some(0));
    let object_raw = read_raw_array::<24>(&object_refs).unwrap();
    assert_eq!(object_raw.len(), 24);
    assert!(object_raw[0..8].iter().any(|&b| b != 0));
    assert!(object_raw[8..16].iter().any(|&b| b != 0));
    assert!(object_raw[16..24].iter().all(|&b| b == 0));
    assert_ne!(&object_raw[0..8], &object_raw[8..16]);

    let region_refs = f.dataset("region_refs").unwrap();
    let region_dtype = region_refs.dtype().unwrap();
    assert_eq!(region_dtype.class(), DatatypeClass::Reference);
    assert_eq!(region_dtype.size(), 12);
    assert_eq!(region_dtype.reference_type(), Some(1));
    let region_raw = read_raw_array::<24>(&region_refs).unwrap();
    assert_eq!(region_raw.len(), 24);
    assert!(region_raw[0..12].iter().any(|&b| b != 0));
    assert!(region_raw[12..24].iter().all(|&b| b == 0));
}

#[test]
fn t3d_time_datatype_raw_and_typed_reads() {
    let f = File::open("tests/data/hdf5_ref/time_cases.h5").unwrap();

    let d32 = f.dataset("unix_d32le").unwrap();
    let dtype32 = d32.dtype().unwrap();
    assert_eq!(dtype32.class(), DatatypeClass::Time);
    assert_eq!(dtype32.size(), 4);
    assert_eq!(
        read_raw_array::<12>(&d32).unwrap().as_slice(),
        [0_u32, 1, 2_147_483_647]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>()
            .as_slice()
    );
    assert_eq!(read_array::<u32, 3>(&d32).unwrap(), [0, 1, 2_147_483_647]);

    let d64 = f.dataset("unix_d64be").unwrap();
    let dtype64 = d64.dtype().unwrap();
    assert_eq!(dtype64.class(), DatatypeClass::Time);
    assert_eq!(dtype64.size(), 8);
    assert_eq!(
        read_raw_array::<24>(&d64).unwrap().as_slice(),
        [0_u64, 1, 4_102_444_800]
            .into_iter()
            .flat_map(u64::to_be_bytes)
            .collect::<Vec<_>>()
            .as_slice()
    );
    assert_eq!(read_array::<u64, 3>(&d64).unwrap(), [0, 1, 4_102_444_800]);
}

#[test]
fn t3d_array_datatype_multidimensional_dims_and_raw_payload() {
    let f = File::open("tests/data/hdf5_ref/array_datatype_cases.h5").unwrap();
    let ds = f.dataset("array_i16_2x3").unwrap();
    let dtype = ds.dtype().unwrap();
    let dims = dtype
        .array_dims_iter()
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let base = dtype.array_base().unwrap();

    assert_eq!(dtype.class(), DatatypeClass::Array);
    assert_eq!(dtype.size(), 12);
    assert_eq!(dims, [2, 3]);
    assert_eq!(base.class(), DatatypeClass::FixedPoint);
    assert_eq!(base.size(), 2);
    assert_eq!(dtype.array_base().unwrap().size(), 2);
    assert_eq!(
        read_raw_array::<24>(&ds).unwrap().as_slice(),
        (0_i16..12)
            .flat_map(i16::to_le_bytes)
            .collect::<Vec<_>>()
            .as_slice()
    );
}

// T3e: Multi-dimensional

#[test]
fn t3e_2d_matrix() {
    let ds = open().dataset("matrix").unwrap();
    let space = ds.space().unwrap();
    assert_eq!(space.shape(), &[3, 4]);
    let arr = ds.read_2d::<i32>().unwrap();
    assert_eq!(arr[[0, 0]], 0);
    assert_eq!(arr[[2, 3]], 11);
}

#[test]
fn t3e_3d_cube() {
    let ds = open().dataset("cube").unwrap();
    let space = ds.space().unwrap();
    assert_eq!(space.shape(), &[2, 3, 4]);
    let vals: [f32; 24] = read_array(&ds).unwrap();
    assert_eq!(vals.len(), 24);
    assert_eq!(vals[0], 0.0);
    assert_eq!(vals[23], 23.0);
}

// T3f: Null dataspace (empty dataset)

#[test]
fn t3f_null_dataspace() {
    let ds = open().dataset("null").unwrap();
    let space = ds.space().unwrap();
    // h5py creates this as null dataspace (no data), not scalar
    assert!(space.is_null() || space.is_scalar());
    assert_eq!(space.ndim(), 0);
}
