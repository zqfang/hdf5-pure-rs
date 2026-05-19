use hdf5_pure_rust::File;

#[test]
fn test_enum_dtype_info() {
    let f = File::open("tests/data/enum.h5").unwrap();
    let ds = f.dataset("colors").unwrap();

    let dtype = ds.dtype().unwrap();
    assert!(dtype.is_enum());
    assert_eq!(dtype.size(), 1); // u8 base type

    let members: Vec<_> = dtype
        .enum_members_iter()
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    println!("Enum members: {members:?}");
    assert_eq!(members.len(), 3);

    // Find members by name
    let red = members.iter().find(|member| member.name == "RED").unwrap();
    let green = members
        .iter()
        .find(|member| member.name == "GREEN")
        .unwrap();
    let blue = members.iter().find(|member| member.name == "BLUE").unwrap();
    assert_eq!(red.value, 0);
    assert_eq!(green.value, 1);
    assert_eq!(blue.value, 2);
}

#[test]
fn test_enum_read_raw_values() {
    let f = File::open("tests/data/enum.h5").unwrap();
    let ds = f.dataset("colors").unwrap();

    let mut values = vec![0; ds.size().unwrap() as usize];
    ds.read_into(&mut values).unwrap();
    assert_eq!(values, vec![0, 1, 2, 1]); // RED, GREEN, BLUE, GREEN
}
