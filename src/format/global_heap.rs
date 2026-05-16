use std::io::{Read, Seek};

use crate::error::{Error, Result};
use crate::io::reader::HdfReader;

/// Global heap collection magic: "GCOL"
const GCOL_MAGIC: [u8; 4] = [b'G', b'C', b'O', b'L'];
const MAX_GLOBAL_HEAP_OBJECT_BYTES: usize = 4 * 1024 * 1024 * 1024;

/// A global heap object reference (collection address + object index).
#[derive(Debug, Clone)]
pub struct GlobalHeapRef {
    pub collection_addr: u64,
    pub object_index: u32,
}

/// Decoded global-heap header — the 16-byte (or so) prefix that precedes
/// the object table. Mirrors the work `H5HG__cache_heap_deserialize` does
/// in libhdf5: magic + version + collection size, no object iteration.
#[derive(Debug, Clone, Copy)]
pub struct GlobalHeapHeader {
    /// Address the collection lives at (anchor for object-table walking).
    pub addr: u64,
    /// Total collection size (includes the header itself).
    pub collection_size: u64,
}

/// A global heap collection containing objects.
#[derive(Debug)]
pub struct GlobalHeapCollection {
    pub objects: Vec<(u32, Vec<u8>)>, // (index, data)
}

impl GlobalHeapCollection {
    /// Creates a new, empty global heap collection. Mirrors libhdf5's
    /// `H5HG__create`.
    pub fn create() -> Self {
        Self {
            objects: Vec::new(),
        }
    }

    /// Pin a collection for use (wrapper around libhdf5's
    /// `H5HG__protect`/`H5AC_protect`). The Rust port owns the value
    /// outright, so this is an identity passthrough.
    pub fn protect(collection: Self) -> Self {
        collection
    }

    /// Allocate a new object in the collection and return its assigned
    /// index. Mirrors libhdf5's `H5HG__alloc`: splits free space, makes
    /// room for the object header, returns the heap object ID.
    pub fn alloc(&mut self, data: Vec<u8>) -> Result<u32> {
        let next = self
            .objects
            .iter()
            .map(|(idx, _)| *idx)
            .max()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or_else(|| Error::InvalidFormat("global heap object index overflow".into()))?;
        self.objects.push((next, data));
        Ok(next)
    }

    /// Extend the collection's capacity to make room for additional
    /// objects. Mirrors libhdf5's `H5HG_extend` (round-up-for-alignment
    /// semantics aside).
    pub fn extend(&mut self, additional: usize) {
        self.objects.reserve(additional);
    }

    /// Insert an object at an explicit index. Mirrors libhdf5's
    /// `H5HG_insert`: a new object is placed in the collection;
    /// index 0 is reserved for free space.
    pub fn insert(&mut self, index: u32, data: Vec<u8>) -> Result<()> {
        if index == 0 {
            return Err(Error::InvalidFormat(
                "global heap object index zero is reserved".into(),
            ));
        }
        if self.objects.iter().any(|(idx, _)| *idx == index) {
            return Err(Error::InvalidFormat(format!(
                "global heap object {index} already exists"
            )));
        }
        self.objects.push((index, data));
        Ok(())
    }

    /// Adjust the link count for a global heap object. Mirrors libhdf5's
    /// `H5HG_link` (we only verify existence; the Rust port doesn't
    /// track refcounts on disk).
    pub fn link(&mut self, index: u32) -> Result<()> {
        if self.get_object(index).is_some() {
            Ok(())
        } else {
            Err(Error::InvalidFormat(format!(
                "global heap object {index} not found"
            )))
        }
    }

    /// Remove the specified object from the global heap collection.
    /// Mirrors libhdf5's `H5HG_remove`.
    pub fn remove(&mut self, index: u32) -> Result<Vec<u8>> {
        let pos = self
            .objects
            .iter()
            .position(|(idx, _)| *idx == index)
            .ok_or_else(|| Error::InvalidFormat(format!("global heap object {index} not found")))?;
        Ok(self.objects.remove(pos).1)
    }

    /// Destroy the in-memory representation of the collection.
    /// Mirrors libhdf5's `H5HG__free`.
    pub fn free(&mut self) {
        self.objects.clear();
    }

    /// Decode the global heap header at `addr`. Mirrors libhdf5's
    /// `H5HG__hdr_deserialize`.
    pub fn hdr_deserialize<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        addr: u64,
    ) -> Result<GlobalHeapHeader> {
        Self::decode_header(reader, addr)
    }

    /// Return the final read size for a speculatively read heap to the
    /// metadata cache. Mirrors `H5HG__cache_heap_get_final_load_size`.
    pub fn cache_heap_get_final_load_size(header: &GlobalHeapHeader) -> Result<usize> {
        heap_object_len(header.collection_size, "global heap collection size")
    }

    /// Return the on-disk image size of this collection assuming an 8-byte
    /// size field. Mirrors `H5HG__cache_heap_image_len`.
    pub fn cache_heap_image_len(&self) -> Result<usize> {
        self.cache_heap_image_len_with_size(8)
    }

    /// On-disk image size of this collection, parametrized by the file's
    /// configured size field width.
    pub fn cache_heap_image_len_with_size(&self, sizeof_size: u8) -> Result<usize> {
        if sizeof_size == 0 || sizeof_size > 8 {
            return Err(Error::InvalidFormat(
                "global heap size field width is invalid".into(),
            ));
        }
        let header_len = 8usize
            .checked_add(usize::from(sizeof_size))
            .ok_or_else(|| Error::InvalidFormat("global heap header length overflow".into()))?;
        let object_header_len = header_len;
        let mut len = header_len;
        for (_, data) in &self.objects {
            validate_global_heap_object_size(data.len())?;
            let padded = data
                .len()
                .checked_add(7)
                .map(|size| size & !7)
                .ok_or_else(|| Error::InvalidFormat("global heap object image overflow".into()))?;
            len = len
                .checked_add(object_header_len)
                .and_then(|value| value.checked_add(padded))
                .ok_or_else(|| Error::InvalidFormat("global heap image length overflow".into()))?;
        }
        len.checked_add(7)
            .map(|value| value & !7)
            .ok_or_else(|| Error::InvalidFormat("global heap image length overflow".into()))
    }

    /// Serialize this collection to its on-disk image, using the supplied
    /// size-field width. Counterpart of the libhdf5 cache-serialize hook.
    pub fn cache_heap_serialize(&self, sizeof_size: u8) -> Result<Vec<u8>> {
        if sizeof_size == 0 || sizeof_size > 8 {
            return Err(Error::InvalidFormat(
                "global heap size field width is invalid".into(),
            ));
        }
        let mut out = Vec::with_capacity(self.cache_heap_image_len_with_size(sizeof_size)?);
        out.extend_from_slice(&GCOL_MAGIC);
        out.push(1);
        out.extend_from_slice(&[0; 3]);
        encode_heap_size(&mut out, 0, sizeof_size, "global heap collection size")?;

        for (index, data) in &self.objects {
            if *index == 0 {
                return Err(Error::InvalidFormat(
                    "global heap object index zero is reserved".into(),
                ));
            }
            validate_global_heap_object_size(data.len())?;
            let data_size = u64::try_from(data.len())
                .map_err(|_| Error::InvalidFormat("global heap object size exceeds u64".into()))?;
            let padded = data
                .len()
                .checked_add(7)
                .map(|size| size & !7)
                .ok_or_else(|| Error::InvalidFormat("global heap object image overflow".into()))?;

            out.extend_from_slice(
                &u16::try_from(*index)
                    .map_err(|_| {
                        Error::InvalidFormat("global heap object index exceeds u16".into())
                    })?
                    .to_le_bytes(),
            );
            out.extend_from_slice(&0u16.to_le_bytes());
            out.extend_from_slice(&[0; 4]);
            encode_heap_size(&mut out, data_size, sizeof_size, "global heap object size")?;
            out.extend_from_slice(data);
            let padded_end = out
                .len()
                .checked_add(padded - data.len())
                .ok_or_else(|| Error::InvalidFormat("global heap object image overflow".into()))?;
            out.resize(padded_end, 0);
        }

        let padded_collection_len = out
            .len()
            .checked_add(7)
            .map(|value| value & !7)
            .ok_or_else(|| Error::InvalidFormat("global heap collection size overflow".into()))?;
        out.resize(padded_collection_len, 0);

        let collection_size = u64::try_from(out.len())
            .map_err(|_| Error::InvalidFormat("global heap collection size exceeds u64".into()))?;
        let mut encoded_size = Vec::new();
        encode_heap_size(
            &mut encoded_size,
            collection_size,
            sizeof_size,
            "global heap collection size",
        )?;
        out[8..8 + usize::from(sizeof_size)].copy_from_slice(&encoded_size);
        Ok(out)
    }

    /// Free the in-core image (no-op in Rust where buffers are owned).
    pub fn cache_heap_free_icr(_image: Vec<u8>) {}

    /// Address of the heap collection that a reference points into.
    /// Mirrors `H5HG_get_addr`.
    pub fn get_addr(reference: &GlobalHeapRef) -> u64 {
        reference.collection_addr
    }

    /// Size in bytes of the object at `index`, or `None` if absent.
    /// Mirrors `H5HG_get_size` / `H5HG_get_obj_size`.
    pub fn get_size(&self, index: u32) -> Option<usize> {
        self.get_object(index).map(|data| data.len())
    }

    /// Free space remaining in the collection's allocation buffer.
    /// Mirrors `H5HG_get_free_size`.
    pub fn get_free_size(&self) -> usize {
        self.objects.capacity().saturating_sub(self.objects.len())
    }

    /// Render debugging information about a global heap collection.
    /// Mirrors `H5HG_debug`.
    pub fn debug(&self) -> String {
        format!("GlobalHeapCollection(objects={})", self.objects.len())
    }

    /// Read a global heap collection at the given address.
    ///
    /// Thin wrapper around the full deserialize path.
    pub fn read_at<R: Read + Seek>(reader: &mut HdfReader<R>, addr: u64) -> Result<Self> {
        Self::deserialize_collection(reader, addr)
    }

    /// Full collection deserialize. This is the closest Rust analog to
    /// libhdf5's `H5HG__cache_heap_deserialize`: parse the fixed prefix,
    /// then walk the variable-length object table into a materialized
    /// collection value.
    pub fn deserialize_collection<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        addr: u64,
    ) -> Result<Self> {
        let header = Self::decode_header(reader, addr)?;
        Self::walk_objects(reader, &header)
    }

    /// Pure header decode: validate magic+version, return `(addr,
    /// collection_size)`. Leaves the reader positioned at the first object
    /// entry so that callers don't have to reseek.
    pub fn decode_header<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        addr: u64,
    ) -> Result<GlobalHeapHeader> {
        reader.seek(addr)?;

        let magic = reader.read_bytes(4)?;
        if magic != GCOL_MAGIC {
            return Err(Error::InvalidFormat(
                "invalid global heap collection magic".into(),
            ));
        }

        let version = reader.read_u8()?;
        if version != 1 {
            return Err(Error::Unsupported(format!("global heap version {version}")));
        }

        reader.read_bytes(3)?;

        // Collection size (includes header)
        let collection_size = reader.read_length()?;
        let header_len = 8u64
            .checked_add(u64::from(reader.sizeof_size()))
            .ok_or_else(|| Error::InvalidFormat("global heap header length overflow".into()))?;
        if collection_size < header_len {
            return Err(Error::InvalidFormat(
                "global heap collection is smaller than its header".into(),
            ));
        }
        if collection_size % 8 != 0 {
            return Err(Error::InvalidFormat(
                "global heap collection size is not 8-byte aligned".into(),
            ));
        }

        Ok(GlobalHeapHeader {
            addr,
            collection_size,
        })
    }

    /// Walk the object table from the reader's current position (which the
    /// header decoder already advanced to). Index-0 records are free-space
    /// objects whose size includes their 16-byte header.
    pub fn walk_objects<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        header: &GlobalHeapHeader,
    ) -> Result<Self> {
        let mut objects: Vec<(u32, Vec<u8>)> = Vec::new();
        let data_end = header
            .addr
            .checked_add(header.collection_size)
            .ok_or_else(|| Error::InvalidFormat("global heap collection size overflow".into()))?;

        while reader.position()? < data_end {
            let pos = reader.position()?;
            let object_header_len = 8u64
                .checked_add(u64::from(reader.sizeof_size()))
                .ok_or_else(|| {
                    Error::InvalidFormat("global heap object header size overflow".into())
                })?;
            let min_entry_end = pos.checked_add(object_header_len).ok_or_else(|| {
                Error::InvalidFormat("global heap object header offset overflow".into())
            })?;
            if min_entry_end > data_end {
                break;
            }

            let index = u32::from(reader.read_u16()?);
            let _reference_count = reader.read_u16()?;
            reader.read_u32()?;
            let obj_size = reader.read_length()?;

            if index == 0 {
                if obj_size == 0 {
                    break;
                }
                let next_pos = pos.checked_add(obj_size).ok_or_else(|| {
                    Error::InvalidFormat("global heap free object offset overflow".into())
                })?;
                if next_pos > data_end {
                    return Err(Error::InvalidFormat(
                        "global heap free object exceeds collection bounds".into(),
                    ));
                }
                reader.seek(next_pos)?;
                continue;
            }

            let obj_len = heap_object_len(obj_size, "global heap object size")?;
            let padded = obj_size
                .checked_add(7)
                .map(|size| size & !7)
                .ok_or_else(|| Error::InvalidFormat("global heap object size overflow".into()))?;
            let next_pos = reader
                .position()?
                .checked_add(padded)
                .ok_or_else(|| Error::InvalidFormat("global heap object offset overflow".into()))?;
            if next_pos > data_end {
                return Err(Error::InvalidFormat(
                    "global heap object exceeds collection bounds".into(),
                ));
            }

            let data = reader.read_bytes(obj_len)?;
            objects.push((index, data));

            // Pad to 8-byte boundary
            let padding = padded - obj_size;
            if padding > 0 {
                reader.skip(padding)?;
            }
        }

        Ok(Self { objects })
    }

    /// Get an object by index from this collection.
    pub fn get_object(&self, index: u32) -> Option<&[u8]> {
        self.objects
            .iter()
            .find(|(idx, _)| *idx == index)
            .map(|(_, data)| data.as_slice())
    }
}

/// Convert a heap-encoded object size into a `usize`, rejecting values
/// that overflow or exceed the supported per-object cap.
fn heap_object_len(value: u64, context: &str) -> Result<usize> {
    let len = usize::try_from(value)
        .map_err(|_| Error::InvalidFormat(format!("{context} does not fit in usize")))?;
    if len > MAX_GLOBAL_HEAP_OBJECT_BYTES {
        return Err(Error::InvalidFormat(format!(
            "{context} {len} exceeds supported maximum {MAX_GLOBAL_HEAP_OBJECT_BYTES}"
        )));
    }
    Ok(len)
}

fn validate_global_heap_object_size(len: usize) -> Result<()> {
    if len > MAX_GLOBAL_HEAP_OBJECT_BYTES {
        return Err(Error::InvalidFormat(format!(
            "global heap object size {len} exceeds supported maximum {MAX_GLOBAL_HEAP_OBJECT_BYTES}"
        )));
    }
    Ok(())
}

fn encode_heap_size(out: &mut Vec<u8>, value: u64, width: u8, context: &str) -> Result<()> {
    let width = usize::from(width);
    if width == 0 || width > 8 {
        return Err(Error::InvalidFormat(format!("{context} width is invalid")));
    }
    if width < 8 && value >= (1u64 << (width * 8)) {
        return Err(Error::InvalidFormat(format!(
            "{context} value {value:#x} does not fit in {width} bytes"
        )));
    }
    out.extend_from_slice(&value.to_le_bytes()[..width]);
    Ok(())
}

/// Read a global heap object by its reference.
pub fn read_global_heap_object<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    gh_ref: &GlobalHeapRef,
) -> Result<Vec<u8>> {
    let collection = GlobalHeapCollection::read_at(reader, gh_ref.collection_addr)?;
    let data = collection
        .get_object(gh_ref.object_index)
        .map(|d| d.to_vec())
        .ok_or_else(|| {
            Error::InvalidFormat(format!(
                "global heap object {} not found in collection at {:#x}",
                gh_ref.object_index, gh_ref.collection_addr
            ))
        })?;
    trace_global_heap_deref(gh_ref, &data);
    Ok(data)
}

#[cfg(feature = "tracehash")]
fn trace_global_heap_deref(gh_ref: &GlobalHeapRef, data: &[u8]) {
    let mut th = tracehash::th_call!("hdf5.global_heap.deref");
    th.input_u64(gh_ref.collection_addr);
    th.input_u64(u64::from(gh_ref.object_index));
    th.output_value(&(true));
    th.output_u64(u64::try_from(data.len()).unwrap_or(u64::MAX));
    th.output_value(data);
    th.finish();
}

#[cfg(not(feature = "tracehash"))]
fn trace_global_heap_deref(_gh_ref: &GlobalHeapRef, _data: &[u8]) {}

#[cfg(test)]
mod tests {
    use super::GlobalHeapCollection;
    use crate::error::Error;
    use crate::io::reader::HdfReader;
    use std::io::Cursor;

    fn heap_with_size(collection_size: u64) -> Vec<u8> {
        let mut heap = b"GCOL".to_vec();
        heap.push(1);
        heap.extend_from_slice(&[0; 3]);
        heap.extend_from_slice(&collection_size.to_le_bytes());
        let collection_size =
            usize::try_from(collection_size).expect("test collection size should fit in usize");
        heap.resize(collection_size.max(16), 0);
        heap
    }

    #[test]
    fn global_heap_rejects_invalid_collection_sizes() {
        let mut reader = HdfReader::new(Cursor::new(heap_with_size(8)));
        let err = GlobalHeapCollection::read_at(&mut reader, 0).unwrap_err();
        assert!(matches!(err, Error::InvalidFormat(_)));

        let mut reader = HdfReader::new(Cursor::new(heap_with_size(17)));
        let err = GlobalHeapCollection::read_at(&mut reader, 0).unwrap_err();
        assert!(matches!(err, Error::InvalidFormat(_)));
    }

    #[test]
    fn global_heap_ignores_trailing_fragment_smaller_than_object_header() {
        let mut heap = b"GCOL".to_vec();
        heap.push(1);
        heap.extend_from_slice(&[0; 3]);
        heap.extend_from_slice(&24u64.to_le_bytes());
        heap.extend_from_slice(&1u16.to_le_bytes());
        heap.extend_from_slice(&0u16.to_le_bytes());
        heap.extend_from_slice(&[0; 4]);

        let mut reader = HdfReader::new(Cursor::new(heap));
        let collection = GlobalHeapCollection::read_at(&mut reader, 0).unwrap();
        assert!(collection.objects.is_empty());
    }

    #[test]
    fn global_heap_cache_serialize_roundtrips_and_checks_size_width() {
        let mut collection = GlobalHeapCollection::create();
        collection.insert(1, b"alpha".to_vec()).unwrap();
        collection.insert(2, b"beta".to_vec()).unwrap();

        let image = collection.cache_heap_serialize(8).unwrap();
        assert_eq!(image.len(), collection.cache_heap_image_len().unwrap());
        let mut reader = HdfReader::new(Cursor::new(image));
        let decoded = GlobalHeapCollection::read_at(&mut reader, 0).unwrap();
        assert_eq!(decoded.get_object(1), Some(&b"alpha"[..]));
        assert_eq!(decoded.get_object(2), Some(&b"beta"[..]));

        let image_4 = collection.cache_heap_serialize(4).unwrap();
        assert_eq!(
            image_4.len(),
            collection.cache_heap_image_len_with_size(4).unwrap()
        );
        let mut reader = HdfReader::new(Cursor::new(image_4));
        reader.set_sizeof_size(4);
        let decoded = GlobalHeapCollection::read_at(&mut reader, 0).unwrap();
        assert_eq!(decoded.get_object(1), Some(&b"alpha"[..]));
        assert_eq!(decoded.get_object(2), Some(&b"beta"[..]));

        let mut too_large_index = GlobalHeapCollection::create();
        too_large_index
            .insert(u32::from(u16::MAX) + 1, Vec::new())
            .unwrap();
        assert!(too_large_index.cache_heap_serialize(8).is_err());
        assert!(collection.cache_heap_serialize(0).is_err());
    }
}
