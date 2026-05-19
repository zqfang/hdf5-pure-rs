use std::fs;
use std::io::{Read, Seek};
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::io::reader::HdfReader;

use super::{usize_from_u64, Dataset, DatasetInfo};

impl Dataset {
    pub(super) fn read_external_raw_data_into<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        hdf5_path: Option<&Path>,
        info: &DatasetInfo,
        output: &mut [u8],
    ) -> Result<()> {
        let external = info.external_file_list.as_ref().ok_or_else(|| {
            Error::InvalidFormat("contiguous dataset has no external file list".into())
        })?;
        let heap = crate::format::local_heap::LocalHeap::read_at(reader, external.heap_addr)?;
        let total_bytes = output.len();
        let mut output_offset = 0usize;
        for entry in &external.entries {
            if output_offset >= total_bytes {
                break;
            }
            let name_offset = usize_from_u64(entry.name_offset, "external file name offset")?;
            let file_name = heap.get_str(name_offset)?;
            let path = Self::resolve_external_raw_file_path(hdf5_path, file_name)?;
            let remaining = total_bytes - output_offset;
            let reserved = if entry.size == u64::MAX {
                remaining
            } else {
                usize_from_u64(entry.size, "external file reserved size")?.min(remaining)
            };
            let mut file = fs::File::open(&path)?;
            file.seek(std::io::SeekFrom::Start(entry.file_offset))?;
            let dst = external_output_window(output, output_offset, reserved)?;
            file.read_exact(dst)?;
            output_offset += reserved;
        }
        if output_offset < total_bytes {
            return Err(Error::InvalidFormat(format!(
                "external raw data storage covers {output_offset} of {total_bytes} bytes"
            )));
        }
        Ok(())
    }

    fn resolve_external_raw_file_path(
        hdf5_path: Option<&Path>,
        file_name: &str,
    ) -> Result<PathBuf> {
        let path = Path::new(file_name);
        if path.is_absolute() {
            return Ok(path.to_path_buf());
        }
        let base = hdf5_path.and_then(Path::parent).ok_or_else(|| {
            Error::Unsupported("relative external raw data file has no base file path".into())
        })?;
        Ok(base.join(path))
    }
}

fn external_output_window(output: &mut [u8], offset: usize, len: usize) -> Result<&mut [u8]> {
    let end = offset
        .checked_add(len)
        .ok_or_else(|| Error::InvalidFormat("external raw data output range overflow".into()))?;
    output
        .get_mut(offset..end)
        .ok_or_else(|| Error::InvalidFormat("external raw data output range exceeds buffer".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn external_output_window_rejects_offset_overflow() {
        let mut output = [];
        let err = external_output_window(&mut output, usize::MAX, 1).unwrap_err();
        assert!(
            err.to_string()
                .contains("external raw data output range overflow"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn external_output_window_rejects_out_of_range_window() {
        let mut output = [0u8; 1];
        let err = external_output_window(&mut output, 1, 1).unwrap_err();
        assert!(
            err.to_string()
                .contains("external raw data output range exceeds buffer"),
            "unexpected error: {err}"
        );
    }
}
