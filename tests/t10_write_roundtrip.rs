//! Phase T10: Write and round-trip tests.
//! Write with pure-Rust, verify with our reader, h5dump, and h5py.

use std::path::PathBuf;

use hdf5_pure_rust::hl::types::{FieldDescriptor, H5Type, TypeClass};
use hdf5_pure_rust::{Dataset, File, MutableFile, WritableFile};

#[repr(C)]
#[derive(Clone, Copy)]
struct Point {
    x: f64,
    label: i32,
}

unsafe impl H5Type for Point {
    fn type_size() -> usize {
        std::mem::size_of::<Point>()
    }

    fn compound_fields_into(out: &mut Vec<FieldDescriptor>) -> Option<()> {
        out.clear();
        out.extend([
            FieldDescriptor {
                name: "x".to_string(),
                offset: std::mem::offset_of!(Point, x),
                size: std::mem::size_of::<f64>(),
                type_class: TypeClass::Float,
            },
            FieldDescriptor {
                name: "label".to_string(),
                offset: std::mem::offset_of!(Point, label),
                size: std::mem::size_of::<i32>(),
                type_class: TypeClass::Integer { signed: true },
            },
        ]);
        Some(())
    }
}

fn tmp(name: &str) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(format!("t10_{name}.h5"));
    (dir, path)
}

fn assert_shape(ds: &Dataset, expected: &[u64]) -> hdf5_pure_rust::Result<()> {
    let mut dims = Vec::new();
    ds.shape_into(&mut dims)?;
    assert_eq!(dims, expected);
    Ok(())
}

fn assert_h5py_script(path: &std::path::Path, script: &str, context: &str) {
    let out = std::process::Command::new("python3")
        .arg("-c")
        .arg(format!(
            "import sys, h5py\n\
             f = h5py.File(sys.argv[1], 'r')\n\
             {script}\n\
             f.close()\n\
             print('OK')"
        ))
        .arg(path)
        .output();
    if let Ok(out) = out {
        assert!(
            out.status.success() && String::from_utf8_lossy(&out.stdout).contains("OK"),
            "h5py verification failed on {context}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

// T10a: Write + h5dump verify -- all layout types

#[test]
fn t10a_contiguous_h5dump() {
    let (_dir, p) = tmp("contiguous");
    {
        let mut wf = WritableFile::create(&p).unwrap();
        wf.new_dataset_builder("data")
            .write::<f64>(&[1.0, 2.0, 3.0])
            .unwrap();
        wf.flush().unwrap();
    }
    let out = std::process::Command::new("h5dump").arg(&p).output();
    if let Ok(out) = out {
        assert!(
            out.status.success(),
            "h5dump: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let s = String::from_utf8_lossy(&out.stdout);
        assert!(s.contains("1, 2, 3"));
    }

    let out = std::process::Command::new("python3")
        .arg("-c")
        .arg(format!(
            "import h5py; f=h5py.File('{}','r'); \
             assert f['data'].shape == (3,); \
             assert list(f['data'][:]) == [1.0, 2.0, 3.0]; \
             print('OK'); f.close()",
            p.display()
        ))
        .output();
    if let Ok(out) = out {
        assert!(
            String::from_utf8_lossy(&out.stdout).contains("OK"),
            "h5py: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

#[test]
fn t10a_compact_h5dump() {
    let (_dir, p) = tmp("compact");
    {
        let mut wf = WritableFile::create(&p).unwrap();
        wf.new_dataset_builder("tiny")
            .compact()
            .write::<u8>(&[10, 20, 30])
            .unwrap();
        wf.flush().unwrap();
    }
    let out = std::process::Command::new("h5dump").arg(&p).output();
    if let Ok(out) = out {
        assert!(out.status.success());
        assert!(String::from_utf8_lossy(&out.stdout).contains("10, 20, 30"));
    }

    assert_h5py_script(
        &p,
        "d = f['tiny']\n\
         assert d.shape == (3,)\n\
         assert d.chunks is None\n\
         assert d[:].tolist() == [10, 20, 30]",
        "compact dataset round-trip fixture",
    );
}

#[test]
fn t10a_chunked_h5dump() {
    let (_dir, p) = tmp("chunked");
    {
        let mut wf = WritableFile::create(&p).unwrap();
        let data: Vec<i32> = (0..50).collect();
        wf.new_dataset_builder("chunked")
            .shape(&[50])
            .chunk(&[10])
            .deflate(4)
            .write::<i32>(&data)
            .unwrap();
        wf.flush().unwrap();
    }
    let out = std::process::Command::new("h5dump")
        .arg("-d")
        .arg("chunked")
        .arg(&p)
        .output();
    if let Ok(out) = out {
        assert!(
            out.status.success(),
            "h5dump: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let s = String::from_utf8_lossy(&out.stdout);
        assert!(s.contains("49"));
    }

    assert_h5py_script(
        &p,
        "d = f['chunked']\n\
         assert d.shape == (50,)\n\
         assert d.chunks == (10,)\n\
         assert d.compression == 'gzip'\n\
         assert d.compression_opts == 4\n\
         assert d[:].tolist() == list(range(50))",
        "chunked h5dump fixture h5py parity",
    );
}

// T10b: Write + h5py verify

#[test]
fn t10b_h5py_verify() {
    let (_dir, p) = tmp("h5py_verify");
    {
        let mut wf = WritableFile::create(&p).unwrap();
        wf.new_dataset_builder("values")
            .write::<f64>(&[1.5, 2.5, 3.5])
            .unwrap();
        wf.add_attr("version", 1i64).unwrap();
        wf.flush().unwrap();
    }
    let out = std::process::Command::new("python3")
        .arg("-c")
        .arg(format!(
            "import h5py; f=h5py.File('{}','r'); \
             assert list(f['values'][:])==[1.5,2.5,3.5]; \
             assert f.attrs['version']==1; \
             print('OK'); f.close()",
            p.display()
        ))
        .output();
    if let Ok(out) = out {
        let s = String::from_utf8_lossy(&out.stdout);
        assert!(
            s.contains("OK"),
            "h5py: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

#[test]
fn t10b_h5py_verify_fill_string_and_compound() {
    let (_dir, p) = tmp("h5py_new_writer_features");
    {
        let mut wf = WritableFile::create(&p).unwrap();
        wf.new_dataset_builder("filled")
            .fill_properties(1, 2)
            .fill_value::<i32>(-7)
            .write::<i32>(&[1, 2, 3])
            .unwrap();
        wf.new_dataset_builder("names")
            .compact()
            .write_fixed_ascii_strings(&["red", "green"], 8)
            .unwrap();
        wf.new_dataset_builder("points")
            .compact()
            .write::<Point>(&[Point { x: 1.5, label: 10 }, Point { x: 2.5, label: 20 }])
            .unwrap();
        wf.flush().unwrap();
    }

    let dump = std::process::Command::new("h5dump").arg(&p).output();
    if let Ok(out) = dump {
        assert!(
            out.status.success(),
            "h5dump: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let s = String::from_utf8_lossy(&out.stdout);
        assert!(s.contains("red"));
        assert!(s.contains("green"));
    }

    let out = std::process::Command::new("python3")
        .arg("-c")
        .arg(format!(
            "import h5py; f=h5py.File('{}','r'); \
             assert list(f['filled'][:])==[1,2,3]; \
             assert [x.decode().rstrip('\\x00') for x in f['names'][:]]==['red','green']; \
             assert list(f['points']['label'])==[10,20]; \
             assert list(f['points']['x'])==[1.5,2.5]; \
             print('OK'); f.close()",
            p.display()
        ))
        .output();
    if let Ok(out) = out {
        let s = String::from_utf8_lossy(&out.stdout);
        assert!(
            s.contains("OK"),
            "h5py: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

// T10c: Write + read round-trip

#[test]
fn t10c_roundtrip_all_types() {
    let (_dir, p) = tmp("roundtrip_types");
    {
        let mut wf = WritableFile::create(&p).unwrap();
        wf.new_dataset_builder("f64")
            .write::<f64>(&[1.0, 2.0])
            .unwrap();
        wf.new_dataset_builder("f32")
            .write::<f32>(&[3.0, 4.0])
            .unwrap();
        wf.new_dataset_builder("i32").write::<i32>(&[5, 6]).unwrap();
        wf.new_dataset_builder("i64").write::<i64>(&[7, 8]).unwrap();
        wf.new_dataset_builder("u8").write::<u8>(&[9, 10]).unwrap();
        wf.flush().unwrap();
    }
    {
        let f = File::open(&p).unwrap();
        let mut f64_values = [0.0f64; 2];
        f.dataset("f64")
            .unwrap()
            .read_into(&mut f64_values)
            .unwrap();
        assert_eq!(f64_values, [1.0, 2.0]);

        let mut f32_values = [0.0f32; 2];
        f.dataset("f32")
            .unwrap()
            .read_into(&mut f32_values)
            .unwrap();
        assert_eq!(f32_values, [3.0, 4.0]);

        let mut i32_values = [0i32; 2];
        f.dataset("i32")
            .unwrap()
            .read_into(&mut i32_values)
            .unwrap();
        assert_eq!(i32_values, [5, 6]);

        let mut i64_values = [0i64; 2];
        f.dataset("i64")
            .unwrap()
            .read_into(&mut i64_values)
            .unwrap();
        assert_eq!(i64_values, [7, 8]);

        let mut u8_values = [0u8; 2];
        f.dataset("u8").unwrap().read_into(&mut u8_values).unwrap();
        assert_eq!(u8_values, [9, 10]);
    }
}

// T10d: Chunked write with all filter combos

#[test]
fn t10d_deflate_only() {
    let (_dir, p) = tmp("deflate_only");
    {
        let mut wf = WritableFile::create(&p).unwrap();
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        wf.new_dataset_builder("d")
            .chunk(&[25])
            .deflate(6)
            .write::<f32>(&data)
            .unwrap();
        wf.flush().unwrap();
    }
    let f = File::open(&p).unwrap();
    let mut vals = vec![0.0f32; 100];
    f.dataset("d").unwrap().read_into(&mut vals).unwrap();
    for (i, v) in vals.iter().enumerate() {
        assert_eq!(*v, i as f32);
    }

    assert_h5py_script(
        &p,
        "d = f['d']\n\
         assert d.shape == (100,)\n\
         assert d.chunks == (25,)\n\
         assert d.compression == 'gzip'\n\
         assert d.compression_opts == 6\n\
         assert d[:].tolist() == [float(i) for i in range(100)]",
        "deflate-only chunked writer fixture",
    );
}

#[test]
fn t10d_shuffle_deflate() {
    let (_dir, p) = tmp("shuffle_deflate");
    {
        let mut wf = WritableFile::create(&p).unwrap();
        let data: Vec<i32> = (0..100).collect();
        wf.new_dataset_builder("d")
            .chunk(&[20])
            .shuffle()
            .deflate(4)
            .write::<i32>(&data)
            .unwrap();
        wf.flush().unwrap();
    }
    let f = File::open(&p).unwrap();
    let mut vals = vec![0i32; 100];
    f.dataset("d").unwrap().read_into(&mut vals).unwrap();
    for (i, v) in vals.iter().enumerate() {
        assert_eq!(*v, i as i32);
    }

    assert_h5py_script(
        &p,
        "d = f['d']\n\
         assert d.shape == (100,)\n\
         assert d.chunks == (20,)\n\
         assert d.compression == 'gzip'\n\
         assert d.compression_opts == 4\n\
         assert d.shuffle\n\
         assert d[:].tolist() == list(range(100))",
        "shuffle+deflate chunked writer fixture",
    );
}

#[test]
fn t10d_many_chunk_deflate_two_level_btree() {
    let (_dir, p) = tmp("many_chunk_deflate_two_level_btree");
    let data: Vec<i32> = (0..5000).collect();
    {
        let mut wf = WritableFile::create(&p).unwrap();
        wf.new_dataset_builder("d")
            .shape(&[data.len() as u64])
            .chunk(&[50])
            .deflate(1)
            .write::<i32>(&data)
            .unwrap();
        wf.flush().unwrap();
    }

    let f = File::open(&p).unwrap();
    let mut vals = vec![0i32; data.len()];
    f.dataset("d").unwrap().read_into(&mut vals).unwrap();
    assert_eq!(vals, data);

    let out = std::process::Command::new("python3")
        .arg("-c")
        .arg(format!(
            "import h5py; f=h5py.File('{}','r'); \
             x=f['d'][:]; \
             assert len(x)==5000; \
             assert int(x[0])==0 and int(x[-1])==4999; \
             print('OK'); f.close()",
            p.display()
        ))
        .output();
    if let Ok(out) = out {
        let s = String::from_utf8_lossy(&out.stdout);
        assert!(
            s.contains("OK"),
            "h5py: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

// T10e: Attribute write round-trip

#[test]
fn t10e_scalar_attrs() {
    let (_dir, p) = tmp("scalar_attrs");
    {
        let mut wf = WritableFile::create(&p).unwrap();
        wf.add_attr("count", 42i64).unwrap();
        wf.add_attr("pi", std::f64::consts::PI).unwrap();
        wf.new_dataset_builder("x").write::<f64>(&[1.0]).unwrap();
        wf.flush().unwrap();
    }
    let f = File::open(&p).unwrap();
    let mut count = [0i64; 1];
    f.attr("count").unwrap().read_into(&mut count).unwrap();
    assert_eq!(count[0], 42);

    let mut pi = [0.0f64; 1];
    f.attr("pi").unwrap().read_into(&mut pi).unwrap();
    assert!((pi[0] - std::f64::consts::PI).abs() < 1e-15);
}

// T10f: MutableFile round-trip

#[test]
fn t10f_resize_roundtrip() {
    let (_dir, p) = tmp("resize_rt");
    {
        let mut wf = WritableFile::create(&p).unwrap();
        wf.new_dataset_builder("d")
            .shape(&[10])
            .chunk(&[5])
            .resizable()
            .write::<f64>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
            .unwrap();
        wf.flush().unwrap();
    }
    {
        let mut mf = MutableFile::open_rw(&p).unwrap();
        mf.resize_dataset("d", &[5]).unwrap();
    }
    {
        let f = File::open(&p).unwrap();
        let ds = f.dataset("d").unwrap();
        assert_shape(&ds, &[5]).unwrap();
        let mut vals = [0.0f64; 5];
        ds.read_into(&mut vals).unwrap();
        assert_eq!(vals, [1.0, 2.0, 3.0, 4.0, 5.0]);
    }
}
