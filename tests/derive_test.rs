use hdf5_pure_rust::DeriveH5Type;
use hdf5_pure_rust::H5Type;

#[derive(Copy, Clone, DeriveH5Type)]
#[repr(C)]
struct Point {
    x: f64,
    y: f64,
    label: i32,
}

#[derive(Copy, Clone, DeriveH5Type)]
#[repr(u8)]
#[allow(dead_code)]
enum Color {
    Red = 0,
    Green = 1,
    Blue = 2,
}

#[derive(Copy, Clone, DeriveH5Type)]
#[repr(C)]
struct Measurement {
    value: f32,
    #[hdf5(rename = "error_margin")]
    error: f32,
}

#[test]
fn test_derive_struct_size() {
    assert_eq!(Point::type_size(), std::mem::size_of::<Point>());
    // f64(8) + f64(8) + i32(4) + padding(4) = 24 on most platforms, or 20 with packed
    assert!(Point::type_size() >= 20);
}

#[test]
fn test_derive_struct_fields() {
    let mut index = 0;
    Point::visit_compound_fields(|field| {
        match index {
            0 => {
                assert_eq!(field.name, "x");
                assert_eq!(field.offset, 0);
                assert_eq!(field.size, 8);
            }
            1 => {
                assert_eq!(field.name, "y");
                assert_eq!(field.offset, 8);
                assert_eq!(field.size, 8);
            }
            2 => {
                assert_eq!(field.name, "label");
                assert_eq!(field.offset, 16);
                assert_eq!(field.size, 4);
            }
            _ => panic!("unexpected extra field: {field:?}"),
        }
        index += 1;
    })
    .unwrap();
    assert_eq!(index, 3);
}

#[test]
fn test_derive_enum_size() {
    assert_eq!(Color::type_size(), 1);
}

#[test]
fn test_derive_enum_members() {
    let expected = [("Red", 0), ("Green", 1), ("Blue", 2)];
    let mut index = 0;
    Color::visit_enum_members(|name, value| {
        assert_eq!((name, value), expected[index]);
        index += 1;
    })
    .unwrap();
    assert_eq!(index, expected.len());
}

#[test]
fn test_derive_with_rename() {
    let expected = ["value", "error_margin"];
    let mut index = 0;
    Measurement::visit_compound_fields(|field| {
        assert_eq!(field.name, expected[index]);
        index += 1;
    })
    .unwrap();
    assert_eq!(index, expected.len());
}

#[test]
fn test_derive_struct_can_read() {
    // Verify the derived type works with read operations
    // (uses type_size for byte reinterpretation)
    let mut bytes = [0u8; 24];
    bytes[0..8].copy_from_slice(&1.0f64.to_le_bytes());
    bytes[8..16].copy_from_slice(&2.0f64.to_le_bytes());
    bytes[16..20].copy_from_slice(&42i32.to_le_bytes());

    let points: Vec<Point> =
        hdf5_pure_rust::hl::types::bytes_to_vec::<Point>(bytes.to_vec()).unwrap();
    assert_eq!(points.len(), 1);
    assert_eq!(points[0].x, 1.0);
    assert_eq!(points[0].y, 2.0);
    assert_eq!(points[0].label, 42);
}
