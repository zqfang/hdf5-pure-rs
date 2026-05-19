use hdf5_pure_rust::File;

fn main() -> hdf5_pure_rust::Result<()> {
    let file = File::open("tests/data/datasets_v0.h5")?;
    let dataset = file.dataset("float64_1d")?;
    let mut values = vec![0.0; dataset.size()? as usize];
    dataset.read_into(&mut values)?;
    println!("{values:?}");
    Ok(())
}
