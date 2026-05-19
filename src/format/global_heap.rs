use std::{
    collections::HashMap,
    fmt,
    io::{Read, Seek},
};

use crate::error::{Error, Result};
use crate::io::reader::HdfReader;

/// Global heap collection magic: "GCOL"
const GCOL_MAGIC: [u8; 4] = [b'G', b'C', b'O', b'L'];

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

    /// Serialize this collection to `out`, using the supplied size-field
    /// width.
    ///
    /// The output buffer is cleared before the image is written.
    pub fn cache_heap_serialize_into(&self, sizeof_size: u8, out: &mut Vec<u8>) -> Result<()> {
        if sizeof_size == 0 || sizeof_size > 8 {
            return Err(Error::InvalidFormat(
                "global heap size field width is invalid".into(),
            ));
        }
        let image_len = self.cache_heap_image_len_with_size(sizeof_size)?;
        out.clear();
        out.try_reserve_exact(image_len).map_err(|err| {
            Error::InvalidFormat(format!("global heap image allocation failed: {err}"))
        })?;
        out.extend_from_slice(&GCOL_MAGIC);
        out.push(1);
        out.extend_from_slice(&[0; 3]);
        encode_heap_size(out, 0, sizeof_size, "global heap collection size")?;

        for (index, data) in &self.objects {
            if *index == 0 {
                return Err(Error::InvalidFormat(
                    "global heap object index zero is reserved".into(),
                ));
            }
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
            encode_heap_size(out, data_size, sizeof_size, "global heap object size")?;
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
        let encoded_size =
            encode_heap_size_bytes(collection_size, sizeof_size, "global heap collection size")?;
        out[8..8 + usize::from(sizeof_size)]
            .copy_from_slice(&encoded_size[..usize::from(sizeof_size)]);
        Ok(())
    }

    /// Serialize this collection to its on-disk image, using the supplied
    /// size-field width.
    pub fn cache_heap_serialize(&self, sizeof_size: u8) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        self.cache_heap_serialize_into(sizeof_size, &mut out)?;
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

    /// Render debugging information about a global heap collection into `out`.
    /// Mirrors `H5HG_debug`.
    pub fn write_debug(&self, out: &mut impl fmt::Write) -> fmt::Result {
        write!(out, "GlobalHeapCollection(objects={})", self.objects.len())
    }

    /// Render debugging information about a global heap collection.
    pub fn debug(&self) -> String {
        let mut out = String::new();
        self.write_debug(&mut out)
            .expect("writing GlobalHeapCollection debug output to String cannot fail");
        out
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
        Self::walk_objects(reader, &header).map_err(|err| match err {
            Error::Io(io_err) if io_err.kind() == std::io::ErrorKind::UnexpectedEof => {
                Error::InvalidFormat("global heap collection body is truncated".into())
            }
            err => err,
        })
    }

    /// Pure header decode: validate magic+version, return `(addr,
    /// collection_size)`. Leaves the reader positioned at the first object
    /// entry so that callers don't have to reseek.
    pub fn decode_header<R: Read + Seek>(
        reader: &mut HdfReader<R>,
        addr: u64,
    ) -> Result<GlobalHeapHeader> {
        reader.seek(addr)?;

        let mut magic = [0u8; 4];
        reader.read_bytes_into(&mut magic)?;
        if magic != GCOL_MAGIC {
            return Err(Error::InvalidFormat(
                "invalid global heap collection magic".into(),
            ));
        }

        let version = reader.read_u8()?;
        if version != 1 {
            return Err(Error::Unsupported(format!("global heap version {version}")));
        }

        let mut reserved = [0u8; 3];
        reader.read_bytes_into(&mut reserved)?;

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
        let object_header_len = 8u64
            .checked_add(u64::from(reader.sizeof_size()))
            .ok_or_else(|| {
                Error::InvalidFormat("global heap object header size overflow".into())
            })?;
        let file_len = reader.len()?;
        let mut pos = reader.position()?;

        while pos < data_end {
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
                pos = next_pos;
                continue;
            }

            let obj_len = heap_object_len(obj_size, "global heap object size")?;
            let padded = obj_size
                .checked_add(7)
                .map(|size| size & !7)
                .ok_or_else(|| Error::InvalidFormat("global heap object size overflow".into()))?;
            let data_pos = min_entry_end;
            let data_bytes_end = data_pos
                .checked_add(obj_size)
                .ok_or_else(|| Error::InvalidFormat("global heap object offset overflow".into()))?;
            let next_pos = data_pos
                .checked_add(padded)
                .ok_or_else(|| Error::InvalidFormat("global heap object offset overflow".into()))?;
            if next_pos > data_end {
                return Err(Error::InvalidFormat(
                    "global heap object exceeds collection bounds".into(),
                ));
            }
            if data_bytes_end > file_len {
                return Err(Error::InvalidFormat(
                    "global heap object data extends past end of file".into(),
                ));
            }

            let mut data = Vec::new();
            resize_heap_object_buffer(&mut data, obj_len)?;
            reader.read_bytes_into(&mut data)?;
            objects.push((index, data));

            // Pad to 8-byte boundary
            let padding = padded - obj_size;
            if padding > 0 {
                reader.skip(padding)?;
            }
            pos = next_pos;
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

    /// Iterate over the objects in this collection without cloning payloads.
    pub fn iter_objects(&self) -> impl Iterator<Item = (u32, &[u8])> {
        self.objects
            .iter()
            .map(|(index, data)| (*index, data.as_slice()))
    }
}

/// Convert a heap-encoded object size into a `usize`, rejecting values
/// that cannot be represented on this platform.
fn heap_object_len(value: u64, context: &str) -> Result<usize> {
    usize::try_from(value)
        .map_err(|_| Error::InvalidFormat(format!("{context} does not fit in usize")))
}

fn resize_heap_object_buffer(out: &mut Vec<u8>, len: usize) -> Result<()> {
    out.clear();
    out.try_reserve_exact(len).map_err(|err| {
        Error::InvalidFormat(format!("global heap object allocation failed: {err}"))
    })?;
    out.resize(len, 0);
    Ok(())
}

fn encode_heap_size_bytes(value: u64, width: u8, context: &str) -> Result<[u8; 8]> {
    let width = usize::from(width);
    if width == 0 || width > 8 {
        return Err(Error::InvalidFormat(format!("{context} width is invalid")));
    }
    if width < 8 && value >= (1u64 << (width * 8)) {
        return Err(Error::InvalidFormat(format!(
            "{context} value {value:#x} does not fit in {width} bytes"
        )));
    }
    Ok(value.to_le_bytes())
}

fn encode_heap_size(out: &mut Vec<u8>, value: u64, width: u8, context: &str) -> Result<()> {
    let bytes = encode_heap_size_bytes(value, width, context)?;
    out.extend_from_slice(&bytes[..usize::from(width)]);
    Ok(())
}

/// Read a global heap object into `out`.
///
/// The output buffer is cleared before the object bytes are written.
pub fn read_global_heap_object_into<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    gh_ref: &GlobalHeapRef,
    out: &mut Vec<u8>,
) -> Result<()> {
    let header = GlobalHeapCollection::decode_header(reader, gh_ref.collection_addr)?;
    read_global_heap_object_from_decoded_header_into(reader, &header, gh_ref.object_index, out)?;
    trace_global_heap_deref(gh_ref, out);
    Ok(())
}

/// Read a global heap object by its reference.
pub fn read_global_heap_object<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    gh_ref: &GlobalHeapRef,
) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    read_global_heap_object_into(reader, gh_ref, &mut out)?;
    Ok(out)
}

/// Read multiple global heap objects, preserving input order in `out`.
///
/// Heap references are batched by collection address so each global heap
/// collection is deserialized at most once for this call.
pub fn read_global_heap_objects_batched<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    refs: &[GlobalHeapRef],
) -> Result<Vec<Vec<u8>>> {
    let mut out = Vec::new();
    read_global_heap_objects_batched_into(reader, refs, &mut out)?;
    Ok(out)
}

/// Read multiple global heap objects into `out`, preserving input order.
///
/// All referenced objects are validated before `out` is modified. Existing
/// per-object buffers are reused where possible.
pub fn read_global_heap_objects_batched_into<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    refs: &[GlobalHeapRef],
    out: &mut Vec<Vec<u8>>,
) -> Result<()> {
    let mut cache = GlobalHeapObjectCache::new();

    for gh_ref in refs {
        cache.object_data(reader, gh_ref)?;
    }

    out.truncate(refs.len());
    while out.len() < refs.len() {
        out.push(Vec::new());
    }

    for (slot, gh_ref) in out.iter_mut().zip(refs) {
        let data = cache.cached_object_data(gh_ref);
        trace_global_heap_deref(gh_ref, data);
        slot.clear();
        slot.extend_from_slice(data);
    }

    Ok(())
}

/// Per-operation global heap collection cache.
///
/// This keeps recursive or generic vlen decoders from repeatedly seeking to
/// and walking the same collection for each object reference.
#[derive(Debug, Default)]
pub struct GlobalHeapObjectCache {
    collections: HashMap<u64, CachedGlobalHeapCollection>,
}

#[derive(Debug)]
struct CachedGlobalHeapCollection {
    collection: GlobalHeapCollection,
    object_positions: HashMap<u32, usize>,
}

impl CachedGlobalHeapCollection {
    fn new(collection: GlobalHeapCollection) -> Self {
        let mut object_positions = HashMap::with_capacity(collection.objects.len());
        for (position, (index, _)) in collection.objects.iter().enumerate() {
            object_positions.entry(*index).or_insert(position);
        }
        Self {
            collection,
            object_positions,
        }
    }

    fn get_object(&self, index: u32) -> Option<&[u8]> {
        self.object_positions
            .get(&index)
            .and_then(|position| self.collection.objects.get(*position))
            .map(|(_, data)| data.as_slice())
    }
}

impl GlobalHeapObjectCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn read_object<R: Read + Seek>(
        &mut self,
        reader: &mut HdfReader<R>,
        gh_ref: &GlobalHeapRef,
    ) -> Result<Vec<u8>> {
        self.visit_object(reader, gh_ref, |data| Ok(data.to_vec()))
    }

    fn object_data<R: Read + Seek>(
        &mut self,
        reader: &mut HdfReader<R>,
        gh_ref: &GlobalHeapRef,
    ) -> Result<&[u8]> {
        let collection = match self.collections.entry(gh_ref.collection_addr) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(CachedGlobalHeapCollection::new(
                    GlobalHeapCollection::read_at(reader, gh_ref.collection_addr)?,
                ))
            }
        };
        collection.get_object(gh_ref.object_index).ok_or_else(|| {
            Error::InvalidFormat(format!(
                "global heap object {} not found in collection at {:#x}",
                gh_ref.object_index, gh_ref.collection_addr
            ))
        })
    }

    fn cached_object_data(&self, gh_ref: &GlobalHeapRef) -> &[u8] {
        self.collections
            .get(&gh_ref.collection_addr)
            .and_then(|collection| collection.get_object(gh_ref.object_index))
            .expect("global heap object was validated before cloning output")
    }

    pub fn visit_object<R: Read + Seek, T>(
        &mut self,
        reader: &mut HdfReader<R>,
        gh_ref: &GlobalHeapRef,
        visitor: impl FnOnce(&[u8]) -> Result<T>,
    ) -> Result<T> {
        let data = self.object_data(reader, gh_ref)?;
        trace_global_heap_deref(gh_ref, data);
        visitor(data)
    }
}

/// Visit a global heap object by reference without cloning its bytes.
pub fn visit_global_heap_object<R: Read + Seek, T>(
    reader: &mut HdfReader<R>,
    gh_ref: &GlobalHeapRef,
    visitor: impl FnOnce(&[u8]) -> Result<T>,
) -> Result<T> {
    let mut data = Vec::new();
    let header = GlobalHeapCollection::decode_header(reader, gh_ref.collection_addr)?;
    read_global_heap_object_from_decoded_header_into(
        reader,
        &header,
        gh_ref.object_index,
        &mut data,
    )?;
    trace_global_heap_deref(gh_ref, &data);
    visitor(&data)
}

fn read_global_heap_object_from_decoded_header_into<R: Read + Seek>(
    reader: &mut HdfReader<R>,
    header: &GlobalHeapHeader,
    object_index: u32,
    out: &mut Vec<u8>,
) -> Result<()> {
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
        let data_pos = reader.position()?;
        let next_pos = data_pos
            .checked_add(padded)
            .ok_or_else(|| Error::InvalidFormat("global heap object offset overflow".into()))?;
        if next_pos > data_end {
            return Err(Error::InvalidFormat(
                "global heap object exceeds collection bounds".into(),
            ));
        }

        if index == object_index {
            let data_bytes_end = data_pos
                .checked_add(obj_size)
                .ok_or_else(|| Error::InvalidFormat("global heap object offset overflow".into()))?;
            if data_bytes_end > reader.len()? {
                return Err(Error::InvalidFormat(
                    "global heap object data extends past end of file".into(),
                ));
            }
            resize_heap_object_buffer(out, obj_len)?;
            reader.read_bytes_into(out)?;
            return Ok(());
        }

        reader.seek(next_pos)?;
    }

    Err(Error::InvalidFormat(format!(
        "global heap object {object_index} not found in collection at {:#x}",
        header.addr
    )))
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
    use super::{
        read_global_heap_object_into, read_global_heap_objects_batched,
        read_global_heap_objects_batched_into, visit_global_heap_object, GlobalHeapCollection,
        GlobalHeapRef,
    };
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

    fn heap_with_declared_object(collection_size: u64, object_size: u64) -> Vec<u8> {
        let mut heap = b"GCOL".to_vec();
        heap.push(1);
        heap.extend_from_slice(&[0; 3]);
        heap.extend_from_slice(&collection_size.to_le_bytes());
        heap.extend_from_slice(&1u16.to_le_bytes());
        heap.extend_from_slice(&1u16.to_le_bytes());
        heap.extend_from_slice(&[0; 4]);
        heap.extend_from_slice(&object_size.to_le_bytes());
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

        let mut image = vec![99];
        collection.cache_heap_serialize_into(8, &mut image).unwrap();
        assert_eq!(collection.cache_heap_serialize(8).unwrap(), image);
        assert_eq!(image.len(), collection.cache_heap_image_len().unwrap());
        let mut reader = HdfReader::new(Cursor::new(image));
        let decoded = GlobalHeapCollection::read_at(&mut reader, 0).unwrap();
        assert_eq!(decoded.get_object(1), Some(&b"alpha"[..]));
        assert_eq!(decoded.get_object(2), Some(&b"beta"[..]));
        assert_eq!(decoded.debug(), "GlobalHeapCollection(objects=2)");

        let mut image_4 = Vec::new();
        collection
            .cache_heap_serialize_into(4, &mut image_4)
            .unwrap();
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
        assert!(too_large_index
            .cache_heap_serialize_into(8, &mut Vec::new())
            .is_err());
        assert!(collection
            .cache_heap_serialize_into(0, &mut Vec::new())
            .is_err());
    }

    #[test]
    fn global_heap_declared_over_4g_object_uses_bounds_not_helper_cap() {
        let object_size = 4u64 * 1024 * 1024 * 1024 + 8;
        let collection_size = 16 + 16 + object_size;
        let image = heap_with_declared_object(collection_size, object_size);

        let mut reader = HdfReader::new(Cursor::new(image.clone()));
        let err = GlobalHeapCollection::read_at(&mut reader, 0).unwrap_err();
        assert!(
            err.to_string()
                .contains("global heap object data extends past end of file"),
            "unexpected error: {err}"
        );

        let mut reader = HdfReader::new(Cursor::new(image));
        let mut out = b"unchanged".to_vec();
        let err = read_global_heap_object_into(
            &mut reader,
            &GlobalHeapRef {
                collection_addr: 0,
                object_index: 1,
            },
            &mut out,
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("global heap object data extends past end of file"),
            "unexpected error: {err}"
        );
        assert_eq!(out, b"unchanged");
    }

    #[test]
    fn global_heap_object_read_fills_caller_buffer() {
        let mut collection = GlobalHeapCollection::create();
        collection.insert(1, b"alpha".to_vec()).unwrap();
        collection.insert(2, b"beta".to_vec()).unwrap();

        let mut image = Vec::new();
        collection.cache_heap_serialize_into(8, &mut image).unwrap();

        let gh_ref = GlobalHeapRef {
            collection_addr: 0,
            object_index: 2,
        };
        let mut out = vec![99, 99, 99];
        let mut reader = HdfReader::new(Cursor::new(image.clone()));
        read_global_heap_object_into(&mut reader, &gh_ref, &mut out).unwrap();
        assert_eq!(out, b"beta");

        let mut reader = HdfReader::new(Cursor::new(image));
        let len = visit_global_heap_object(&mut reader, &gh_ref, |data| Ok(data.len())).unwrap();
        assert_eq!(len, 4);
    }

    #[test]
    fn global_heap_batched_read_preserves_order_and_duplicates() {
        let mut collection = GlobalHeapCollection::create();
        collection.insert(1, b"alpha".to_vec()).unwrap();
        collection.insert(2, b"beta".to_vec()).unwrap();
        collection.insert(3, b"gamma".to_vec()).unwrap();

        let mut image = Vec::new();
        collection.cache_heap_serialize_into(8, &mut image).unwrap();

        let refs = [
            GlobalHeapRef {
                collection_addr: 0,
                object_index: 3,
            },
            GlobalHeapRef {
                collection_addr: 0,
                object_index: 1,
            },
            GlobalHeapRef {
                collection_addr: 0,
                object_index: 3,
            },
            GlobalHeapRef {
                collection_addr: 0,
                object_index: 2,
            },
        ];
        let mut reader = HdfReader::new(Cursor::new(image));
        let out = read_global_heap_objects_batched(&mut reader, &refs).unwrap();

        assert_eq!(
            out,
            vec![
                b"gamma".to_vec(),
                b"alpha".to_vec(),
                b"gamma".to_vec(),
                b"beta".to_vec()
            ]
        );

        let mut image = Vec::new();
        collection.cache_heap_serialize_into(8, &mut image).unwrap();
        let mut reader = HdfReader::new(Cursor::new(image));
        let mut reused = vec![Vec::with_capacity(16), b"stale".to_vec(), b"extra".to_vec()];
        read_global_heap_objects_batched_into(&mut reader, &refs[..2], &mut reused).unwrap();
        assert_eq!(reused, vec![b"gamma".to_vec(), b"alpha".to_vec()]);
    }

    #[test]
    fn global_heap_batched_read_reports_missing_object_in_input_order() {
        let mut collection = GlobalHeapCollection::create();
        collection.insert(1, b"alpha".to_vec()).unwrap();

        let mut image = Vec::new();
        collection.cache_heap_serialize_into(8, &mut image).unwrap();

        let refs = [
            GlobalHeapRef {
                collection_addr: 0,
                object_index: 1,
            },
            GlobalHeapRef {
                collection_addr: 0,
                object_index: 9,
            },
        ];
        let mut reader = HdfReader::new(Cursor::new(image));
        let err = read_global_heap_objects_batched(&mut reader, &refs).unwrap_err();

        assert!(
            err.to_string()
                .contains("global heap object 9 not found in collection at 0x0"),
            "unexpected error: {err}"
        );

        let mut image = Vec::new();
        collection.cache_heap_serialize_into(8, &mut image).unwrap();
        let mut reader = HdfReader::new(Cursor::new(image));
        let mut out = b"keep me".to_vec();
        let missing = GlobalHeapRef {
            collection_addr: 0,
            object_index: 9,
        };
        let err = read_global_heap_object_into(&mut reader, &missing, &mut out).unwrap_err();
        assert!(
            err.to_string()
                .contains("global heap object 9 not found in collection at 0x0"),
            "unexpected error: {err}"
        );
        assert_eq!(out, b"keep me");
    }
}
