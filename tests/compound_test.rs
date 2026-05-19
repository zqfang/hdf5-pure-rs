use hdf5_pure_rust::{Dataset, File, H5Type, H5Value};

fn read_field_array<T: H5Type + Default + Copy, const N: usize>(
    ds: &Dataset,
    field: &str,
) -> [T; N] {
    let mut values = [T::default(); N];
    ds.read_field_into(field, &mut values).unwrap();
    values
}

fn field_values(ds: &Dataset, field: &str) -> Vec<H5Value> {
    let mut values = Vec::new();
    ds.visit_field_values(field, |value| {
        values.push(value);
        Ok(())
    })
    .unwrap();
    values
}

#[test]
fn test_compound_dtype_info() {
    let f = File::open("tests/data/compound.h5").unwrap();
    let ds = f.dataset("points").unwrap();

    let dtype = ds.dtype().unwrap();
    assert!(dtype.is_compound());
    assert_eq!(dtype.size(), 20); // f64 + f64 + i32

    let mut fields = dtype.compound_fields_iter().unwrap();
    assert_eq!(fields.len(), 3);

    let field = fields.next().unwrap().unwrap();
    assert_eq!(field.name, "x");
    assert_eq!(field.byte_offset, 0);
    assert_eq!(field.size, 8);

    let field = fields.next().unwrap().unwrap();
    assert_eq!(field.name, "y");
    assert_eq!(field.byte_offset, 8);
    assert_eq!(field.size, 8);

    let field = fields.next().unwrap().unwrap();
    assert_eq!(field.name, "label");
    assert_eq!(field.byte_offset, 16);
    assert_eq!(field.size, 4);
}

#[test]
fn test_compound_read_field_f64() {
    let f = File::open("tests/data/compound.h5").unwrap();
    let ds = f.dataset("points").unwrap();

    let mut x_into = [0.0; 3];
    ds.read_field_into::<f64>("x", &mut x_into).unwrap();
    assert_eq!(x_into, [1.0, 3.0, 5.0]);

    let x_vals: [f64; 3] = read_field_array(&ds, "x");
    assert_eq!(x_vals, [1.0, 3.0, 5.0]);

    let y_vals: [f64; 3] = read_field_array(&ds, "y");
    assert_eq!(y_vals, [2.0, 4.0, 6.0]);
}

#[test]
fn test_compound_read_field_i32() {
    let f = File::open("tests/data/compound.h5").unwrap();
    let ds = f.dataset("points").unwrap();

    let labels: [i32; 3] = read_field_array(&ds, "label");
    assert_eq!(labels, [10, 20, 30]);
}

#[test]
fn test_compound_read_field_raw() {
    let f = File::open("tests/data/compound.h5").unwrap();
    let ds = f.dataset("points").unwrap();

    let mut raw_into = [0u8; 12];
    ds.read_field_raw_into("label", &mut raw_into).unwrap();
    let labels_into: Vec<i32> = raw_into
        .chunks_exact(4)
        .map(|bytes| i32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    assert_eq!(labels_into, vec![10, 20, 30]);

    let mut raw = [0u8; 12];
    ds.read_field_raw_into("label", &mut raw).unwrap();
    let labels: Vec<i32> = raw
        .chunks_exact(4)
        .map(|bytes| i32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    assert_eq!(labels, vec![10, 20, 30]);
}

#[test]
fn test_compound_read_field_wrong_size() {
    let f = File::open("tests/data/compound.h5").unwrap();
    let ds = f.dataset("points").unwrap();

    // Try to read f64 field as i32 (wrong size)
    let mut values = [0i32; 3];
    let result = ds.read_field_into("x", &mut values);
    assert!(result.is_err());
}

#[test]
fn test_compound_fields_api() {
    let f = File::open("tests/data/compound.h5").unwrap();
    let ds = f.dataset("points").unwrap();

    let dtype = ds.dtype().unwrap();
    let mut fields = dtype.compound_fields_iter().unwrap();
    assert_eq!(fields.next().unwrap().unwrap().name, "x");
    assert_eq!(fields.next().unwrap().unwrap().name, "y");
    assert_eq!(fields.next().unwrap().unwrap().name, "label");
    assert!(fields.next().is_none());
}

#[test]
fn test_recursive_compound_nested_member_values() {
    let f = File::open("tests/data/hdf5_ref/compound_complex.h5").unwrap();
    let ds = f.dataset("compound_complex").unwrap();

    let nested = field_values(&ds, "nested");
    assert_eq!(
        nested[0],
        H5Value::Compound(vec![
            ("a".to_string(), H5Value::Int(7)),
            ("b".to_string(), H5Value::Float(1.5)),
        ])
    );
    assert_eq!(
        nested[1],
        H5Value::Compound(vec![
            ("a".to_string(), H5Value::Int(8)),
            ("b".to_string(), H5Value::Float(2.5)),
        ])
    );
}

#[test]
fn test_recursive_compound_array_member_values() {
    let f = File::open("tests/data/hdf5_ref/compound_complex.h5").unwrap();
    let ds = f.dataset("compound_complex").unwrap();

    let arrays = field_values(&ds, "arr");
    assert_eq!(
        arrays[0],
        H5Value::Array(vec![H5Value::Int(1), H5Value::Int(2), H5Value::Int(3),])
    );
    assert_eq!(
        arrays[1],
        H5Value::Array(vec![H5Value::Int(4), H5Value::Int(5), H5Value::Int(6),])
    );
}

#[test]
fn test_recursive_compound_vlen_member_values() {
    let f = File::open("tests/data/hdf5_ref/compound_complex.h5").unwrap();
    let ds = f.dataset("compound_complex").unwrap();

    let values = field_values(&ds, "vlen");
    assert_eq!(
        values[0],
        H5Value::VarLen(vec![H5Value::Int(10), H5Value::Int(11)])
    );
    assert_eq!(
        values[1],
        H5Value::VarLen(vec![H5Value::Int(20), H5Value::Int(21), H5Value::Int(22),])
    );
}

#[test]
fn test_recursive_compound_reference_member_values() {
    let f = File::open("tests/data/hdf5_ref/compound_complex.h5").unwrap();
    let ds = f.dataset("compound_complex").unwrap();

    let refs = field_values(&ds, "ref");
    match (&refs[0], &refs[1]) {
        (H5Value::Reference(a), H5Value::Reference(b)) => {
            assert_ne!(*a, 0);
            assert_eq!(a, b);
        }
        other => panic!("unexpected reference values: {other:?}"),
    }
}

#[test]
fn test_compound_padded_reordered_members() {
    let f = File::open("tests/data/hdf5_ref/compound_layout_cases.h5").unwrap();
    let ds = f.dataset("padded_reordered").unwrap();
    let dtype = ds.dtype().unwrap();
    let fields = dtype
        .compound_fields_iter()
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    assert_eq!(fields.len(), 3);
    assert_eq!(fields[0].name, "second");
    assert_eq!(fields[0].byte_offset, 4);
    assert_eq!(fields[1].name, "first");
    assert_eq!(fields[1].byte_offset, 0);
    assert_eq!(fields[2].name, "last");
    assert_eq!(fields[2].byte_offset, 8);
    assert_eq!(ds.dtype().unwrap().size(), 12);

    assert_eq!(read_field_array::<i32, 2>(&ds, "first"), [1000, 2000]);
    assert_eq!(read_field_array::<i16, 2>(&ds, "second"), [10, 20]);
    assert_eq!(read_field_array::<u8, 2>(&ds, "last"), [7, 8]);
}

#[test]
fn test_recursive_compound_nested_vlen_member_values() {
    let f = File::open("tests/data/hdf5_ref/compound_layout_cases.h5").unwrap();
    let ds = f.dataset("nested_vlen").unwrap();

    let values = field_values(&ds, "nested_vlen");
    assert_eq!(
        values[0],
        H5Value::Compound(vec![
            ("tag".to_string(), H5Value::Int(3)),
            (
                "seq".to_string(),
                H5Value::VarLen(vec![H5Value::Int(1), H5Value::Int(2)])
            ),
        ])
    );
    assert_eq!(
        values[1],
        H5Value::Compound(vec![
            ("tag".to_string(), H5Value::Int(4)),
            (
                "seq".to_string(),
                H5Value::VarLen(vec![H5Value::Int(5), H5Value::Int(6), H5Value::Int(7)])
            ),
        ])
    );
}

#[test]
fn test_compound_multidimensional_array_member_values() {
    let f = File::open("tests/data/hdf5_ref/array_datatype_cases.h5").unwrap();
    let ds = f.dataset("compound_array2d").unwrap();
    let dtype = ds.dtype().unwrap();
    let grid = dtype
        .compound_fields_iter()
        .unwrap()
        .find_map(|field| {
            let field = field.unwrap();
            (field.name == "grid").then_some(field)
        })
        .unwrap();
    let dims = grid
        .datatype
        .array_dims_iter()
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let base = grid.datatype.array_base().unwrap();

    assert_eq!(dims, vec![2, 3]);
    assert_eq!(base.size(), 2);
    let values = field_values(&ds, "grid");
    assert_eq!(
        values[0],
        H5Value::Array(vec![
            H5Value::Int(1),
            H5Value::Int(2),
            H5Value::Int(3),
            H5Value::Int(4),
            H5Value::Int(5),
            H5Value::Int(6),
        ])
    );
    assert_eq!(
        values[1],
        H5Value::Array(vec![
            H5Value::Int(7),
            H5Value::Int(8),
            H5Value::Int(9),
            H5Value::Int(10),
            H5Value::Int(11),
            H5Value::Int(12),
        ])
    );
}
