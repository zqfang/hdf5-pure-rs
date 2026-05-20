use std::fmt::{self, Write};

use crate::format::messages::dataspace::{DataspaceMessage, DataspaceType};
use crate::hl::selection::Selection;

/// hdf5-metno compatibility extents alias using this crate's current and maximum dimensions.
pub type Extents = (Vec<u64>, Option<Vec<u64>>);

/// hdf5-metno compatibility raw-selection alias backed by this crate's selection type.
pub type RawSelection = Selection;

/// High-level dataspace descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dataspace {
    msg: DataspaceMessage,
}

impl Dataspace {
    /// Initialize dataspace package support.
    pub fn init() -> bool {
        true
    }

    /// Internal dataspace package initialization alias.
    pub fn init_package() -> bool {
        Self::init()
    }

    /// Top-level dataspace package termination alias.
    pub fn top_term_package() {}

    /// Dataspace package termination alias.
    pub fn term_package() {}

    /// Create a scalar, simple, or null dataspace with no simple dimensions.
    pub fn create(space_type: DataspaceType) -> Self {
        let dims = Vec::new();
        Self {
            msg: DataspaceMessage {
                version: 2,
                space_type,
                ndims: 0,
                dims,
                max_dims: None,
            },
        }
    }

    /// Create a scalar dataspace.
    pub fn scalar() -> Self {
        Self::create(DataspaceType::Scalar)
    }

    /// Create a null dataspace.
    pub fn null() -> Self {
        Self::create(DataspaceType::Null)
    }

    /// Create a simple dataspace from current and optional maximum dimensions.
    pub fn simple(dims: Vec<u64>, max_dims: Option<Vec<u64>>) -> crate::Result<Self> {
        let ndims = u8::try_from(dims.len())
            .map_err(|_| crate::Error::InvalidFormat("dataspace rank exceeds u8::MAX".into()))?;
        if let Some(max_dims) = max_dims.as_ref() {
            if max_dims.len() != dims.len() {
                return Err(crate::Error::InvalidFormat(
                    "dataspace max dimensions rank does not match current dimensions".into(),
                ));
            }
        }
        Ok(Self {
            msg: DataspaceMessage {
                version: 2,
                space_type: DataspaceType::Simple,
                ndims,
                dims,
                max_dims,
            },
        })
    }

    pub(crate) fn from_message(msg: DataspaceMessage) -> Self {
        Self { msg }
    }

    /// Close callback alias. The pure Rust dataspace is consumed.
    pub fn close_cb(self) {}

    /// Return the parsed low-level dataspace message.
    pub fn raw_message_ref(&self) -> &DataspaceMessage {
        &self.msg
    }

    /// Return the parsed low-level dataspace message.
    pub fn raw_message(&self) -> DataspaceMessage {
        self.raw_message_ref().clone()
    }

    /// Return the parsed simple extent metadata.
    pub fn simple_extent_ref(&self) -> &DataspaceMessage {
        self.raw_message_ref()
    }

    /// Return the parsed simple extent metadata.
    pub fn simple_extent(&self) -> DataspaceMessage {
        self.simple_extent_ref().clone()
    }

    /// Explicit dataspace copy operation.
    pub fn copy(&self) -> Self {
        self.clone()
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        usize::from(self.msg.ndims)
    }

    /// Internal simple-extent rank helper.
    pub fn simple_extent_ndims(&self) -> usize {
        self.ndim()
    }

    /// Current dimension sizes.
    pub fn shape(&self) -> &[u64] {
        &self.msg.dims
    }

    /// Maximum dimension sizes (None if same as current).
    pub fn maxdims(&self) -> Option<&[u64]> {
        self.msg.max_dims.as_deref()
    }

    /// Return `(current_dims, max_dims)` extent vectors.
    pub fn extent_dims(&self) -> (&[u64], Option<&[u64]>) {
        (self.shape(), self.maxdims())
    }

    /// Simple-extent dimension helper.
    pub fn simple_extent_dims(&self) -> (&[u64], Option<&[u64]>) {
        self.extent_dims()
    }

    /// Return the dataspace extent type.
    pub fn extent_type(&self) -> DataspaceType {
        self.msg.space_type
    }

    /// Simple-extent type helper.
    pub fn simple_extent_type(&self) -> DataspaceType {
        self.extent_type()
    }

    /// Whether this dataspace has an extent.
    pub fn has_extent(&self) -> bool {
        true
    }

    /// hdf5-metno compatibility layer: validate the in-memory extent metadata; do not remove.
    pub fn is_valid(&self) -> bool {
        if !matches!(self.msg.version, 1 | 2) {
            return false;
        }

        if usize::from(self.msg.ndims) != self.msg.dims.len() {
            return false;
        }

        if matches!(
            self.msg.space_type,
            DataspaceType::Scalar | DataspaceType::Null
        ) && self.msg.ndims != 0
        {
            return false;
        }

        if let Some(max_dims) = &self.msg.max_dims {
            max_dims.len() == self.msg.dims.len()
                && self
                    .msg
                    .dims
                    .iter()
                    .zip(max_dims)
                    .all(|(&dim, &max_dim)| dim <= max_dim)
        } else {
            true
        }
    }

    /// hdf5-metno compatibility layer: encode this dataspace extent message.
    pub fn encode(&self) -> crate::Result<Vec<u8>> {
        self.msg.encode()
    }

    /// hdf5-metno compatibility layer: return current and maximum dimensions; do not remove.
    pub fn extents(&self) -> crate::Result<Extents> {
        let mut dims = Vec::with_capacity(self.msg.dims.len());
        let mut max_dims = None;
        self.extents_into(&mut dims, &mut max_dims);
        Ok((dims, max_dims))
    }

    /// Store current and maximum dimensions in caller-provided buffers.
    ///
    /// This keeps repeated metadata access from allocating new dimension
    /// vectors while the hdf5-metno compatibility `extents` wrapper still
    /// returns owned values.
    pub fn extents_into(&self, dims: &mut Vec<u64>, max_dims: &mut Option<Vec<u64>>) {
        dims.clear();
        dims.extend_from_slice(&self.msg.dims);

        match (&self.msg.max_dims, max_dims) {
            (Some(src), Some(dst)) => {
                dst.clear();
                dst.extend_from_slice(src);
            }
            (Some(src), slot @ None) => {
                *slot = Some(src.clone());
            }
            (None, slot) => {
                *slot = None;
            }
        }
    }

    /// Validate and return a simple dataspace offset vector.
    pub fn offset_simple(&self, offsets: &[i64]) -> crate::Result<Vec<i64>> {
        let mut out = Vec::with_capacity(offsets.len());
        self.offset_simple_into(offsets, &mut out)?;
        Ok(out)
    }

    /// Validate and store a simple dataspace offset vector in caller storage.
    pub fn offset_simple_into(&self, offsets: &[i64], out: &mut Vec<i64>) -> crate::Result<()> {
        if offsets.len() != self.ndim() {
            return Err(crate::Error::InvalidFormat(format!(
                "dataspace offset rank {} does not match dataspace rank {}",
                offsets.len(),
                self.ndim()
            )));
        }
        out.clear();
        out.extend_from_slice(offsets);
        Ok(())
    }

    /// Replace this dataspace with a simple extent.
    pub fn set_extent_simple(
        &mut self,
        dims: Vec<u64>,
        max_dims: Option<Vec<u64>>,
    ) -> crate::Result<()> {
        *self = Self::simple(dims, max_dims)?;
        Ok(())
    }

    /// Internal real extent mutation helper.
    pub fn set_extent_real(
        &mut self,
        dims: Vec<u64>,
        max_dims: Option<Vec<u64>>,
    ) -> crate::Result<()> {
        self.set_extent_simple(dims, max_dims)
    }

    /// Total number of elements.
    ///
    /// If the dimension product exceeds `u64::MAX`, this returns `u64::MAX`.
    /// Fallible read/write paths validate shape products and return errors
    /// instead of relying on this display-oriented helper.
    pub fn size(&self) -> u64 {
        if self.msg.dims.is_empty() {
            if self.msg.space_type == DataspaceType::Scalar {
                1
            } else {
                0
            }
        } else {
            self.msg
                .dims
                .iter()
                .try_fold(1u64, |acc, &dim| acc.checked_mul(dim))
                .unwrap_or(u64::MAX)
        }
    }

    /// Return the extent element count.
    pub fn extent_nelem(&self) -> u64 {
        self.size()
    }

    /// hdf5-metno compatibility layer: return the current selection element count; do not remove.
    pub fn selection_size(&self) -> usize {
        usize::try_from(self.size()).unwrap_or(usize::MAX)
    }

    /// hdf5-metno compatibility layer: return the implicit all-selection; do not remove.
    pub fn get_raw_selection(&self) -> crate::Result<RawSelection> {
        Ok(Selection::All)
    }

    /// hdf5-metno compatibility layer: return the implicit all-selection; do not remove.
    pub fn get_selection(&self) -> crate::Result<Selection> {
        self.get_raw_selection()
    }

    /// Return the maximum possible element count if all max dimensions are
    /// finite; returns `u64::MAX` when any max dimension is unlimited or the
    /// product overflows.
    pub fn npoints_max(&self) -> u64 {
        let dims = self.msg.max_dims.as_ref().unwrap_or(&self.msg.dims);
        if dims.iter().any(|&dim| dim == u64::MAX) {
            return u64::MAX;
        }
        if dims.is_empty() {
            return if self.is_scalar() { 1 } else { 0 };
        }
        dims.iter()
            .try_fold(1u64, |acc, &dim| acc.checked_mul(dim))
            .unwrap_or(u64::MAX)
    }

    /// Whether this is a scalar dataspace.
    pub fn is_scalar(&self) -> bool {
        self.msg.space_type == DataspaceType::Scalar
    }

    /// Whether this is a null dataspace (no data).
    pub fn is_null(&self) -> bool {
        self.msg.space_type == DataspaceType::Null
    }

    /// Whether this is a simple (N-dimensional) dataspace.
    pub fn is_simple(&self) -> bool {
        self.msg.space_type == DataspaceType::Simple
    }

    /// Internal simple-dataspace predicate alias.
    pub fn is_simple_internal(&self) -> bool {
        self.is_simple()
    }

    /// Return the local dataspace category used before MPI-specific I/O.
    pub fn mpio_space_type(&self) -> DataspaceType {
        self.msg.space_type
    }

    /// Return the dataspace category used when obtaining transfer datatype
    /// context in the C selection I/O path.
    pub fn obtain_datatype(&self) -> DataspaceType {
        self.msg.space_type
    }

    /// Debug representation for dataspace diagnostics.
    pub fn write_debug<W: Write + ?Sized>(&self, out: &mut W) -> fmt::Result {
        write!(out, "{:?}", self.msg)
    }

    /// Debug representation for dataspace diagnostics.
    pub fn debug(&self) -> String {
        let mut out = String::new();
        let _ = self.write_debug(&mut out);
        out
    }

    /// Whether any dimension is resizable (has unlimited max dim).
    pub fn is_resizable(&self) -> bool {
        if let Some(maxdims) = &self.msg.max_dims {
            maxdims.iter().any(|&d| d == u64::MAX)
        } else {
            false
        }
    }

    /// Compare two dataspaces' extents.
    pub fn extent_equal(&self, other: &Self) -> bool {
        self.msg.space_type == other.msg.space_type
            && self.msg.dims == other.msg.dims
            && self.msg.max_dims == other.msg.max_dims
    }

    /// Internal extent equality helper.
    pub fn extent_equal_internal(&self, other: &Self) -> bool {
        self.extent_equal(other)
    }

    /// Set the serialized dataspace message version used for writer-side
    /// metadata.
    pub fn set_version(&mut self, version: u8) -> crate::Result<()> {
        if !matches!(version, 1 | 2) {
            return Err(crate::Error::InvalidFormat(format!(
                "dataspace message version {version}"
            )));
        }
        self.msg.version = version;
        Ok(())
    }
}

#[allow(non_snake_case)]
pub fn H5S_init() -> bool {
    Dataspace::init()
}

#[allow(non_snake_case)]
pub fn H5S__init_package() -> bool {
    Dataspace::init_package()
}

#[allow(non_snake_case)]
pub fn H5S_top_term_package() {
    Dataspace::top_term_package()
}

#[allow(non_snake_case)]
pub fn H5S_term_package() {
    Dataspace::term_package()
}

#[allow(non_snake_case)]
pub fn H5S__close_cb(space: Dataspace) {
    space.close_cb()
}

#[allow(non_snake_case)]
pub fn H5S_create(space_type: DataspaceType) -> Dataspace {
    Dataspace::create(space_type)
}

#[allow(non_snake_case)]
pub fn H5Screate_simple(dims: Vec<u64>, max_dims: Option<Vec<u64>>) -> crate::Result<Dataspace> {
    Dataspace::simple(dims, max_dims)
}

#[allow(non_snake_case)]
pub fn H5S_copy(space: &Dataspace) -> Dataspace {
    space.copy()
}

#[allow(non_snake_case)]
pub fn H5S_get_npoints_max(space: &Dataspace) -> u64 {
    space.npoints_max()
}

#[allow(non_snake_case)]
pub fn H5S_get_simple_extent_ndims(space: &Dataspace) -> usize {
    space.simple_extent_ndims()
}

#[allow(non_snake_case)]
pub fn H5Sget_simple_extent_ndims(space: &Dataspace) -> usize {
    H5S_get_simple_extent_ndims(space)
}

#[allow(non_snake_case)]
pub fn H5S_extent_get_dims_ref(space: &Dataspace) -> (&[u64], Option<&[u64]>) {
    space.extent_dims()
}

#[allow(non_snake_case)]
pub fn H5S_extent_get_dims(space: &Dataspace) -> (Vec<u64>, Option<Vec<u64>>) {
    space
        .extents()
        .expect("dataspace extents are already validated in memory")
}

#[allow(non_snake_case)]
pub fn H5S_get_simple_extent_dims(space: &Dataspace) -> (Vec<u64>, Option<Vec<u64>>) {
    H5S_extent_get_dims(space)
}

#[allow(non_snake_case)]
pub fn H5Sget_simple_extent_dims(space: &Dataspace) -> (Vec<u64>, Option<Vec<u64>>) {
    H5S_get_simple_extent_dims(space)
}

#[allow(non_snake_case)]
pub fn H5S__is_simple(space: &Dataspace) -> bool {
    space.is_simple_internal()
}

#[allow(non_snake_case)]
pub fn H5Sis_simple(space: &Dataspace) -> bool {
    space.is_simple()
}

#[allow(non_snake_case)]
pub fn H5S_get_simple_extent(space: &Dataspace) -> Dataspace {
    Dataspace::from_message(space.simple_extent_ref().clone())
}

#[allow(non_snake_case)]
pub fn H5S_get_simple_extent_type(space: &Dataspace) -> DataspaceType {
    space.simple_extent_type()
}

#[allow(non_snake_case)]
pub fn H5Sget_simple_extent_type(space: &Dataspace) -> DataspaceType {
    H5S_get_simple_extent_type(space)
}

#[allow(non_snake_case)]
pub fn H5S_get_simple_extent_npoints(space: &Dataspace) -> u64 {
    space.extent_nelem()
}

#[allow(non_snake_case)]
pub fn H5Sget_simple_extent_npoints(space: &Dataspace) -> u64 {
    H5S_get_simple_extent_npoints(space)
}

#[allow(non_snake_case)]
pub fn H5S_has_extent(space: &Dataspace) -> bool {
    space.has_extent()
}

#[allow(non_snake_case)]
pub fn H5S_set_extent_real(
    space: &mut Dataspace,
    dims: Vec<u64>,
    max_dims: Option<Vec<u64>>,
) -> crate::Result<()> {
    space.set_extent_real(dims, max_dims)
}

#[allow(non_snake_case)]
pub fn H5Sset_extent_simple(
    space: &mut Dataspace,
    dims: Vec<u64>,
    max_dims: Option<Vec<u64>>,
) -> crate::Result<()> {
    space.set_extent_simple(dims, max_dims)
}

#[allow(non_snake_case)]
pub fn H5S_select_offset(space: &Dataspace, offsets: &[i64]) -> crate::Result<Vec<i64>> {
    space.offset_simple(offsets)
}

#[allow(non_snake_case)]
pub fn H5S_select_offset_into(
    space: &Dataspace,
    offsets: &[i64],
    out: &mut Vec<i64>,
) -> crate::Result<()> {
    space.offset_simple_into(offsets, out)
}

#[allow(non_snake_case)]
pub fn H5Soffset_simple(space: &Dataspace, offsets: &[i64]) -> crate::Result<Vec<i64>> {
    H5S_select_offset(space, offsets)
}

#[allow(non_snake_case)]
pub fn H5Soffset_simple_into(
    space: &Dataspace,
    offsets: &[i64],
    out: &mut Vec<i64>,
) -> crate::Result<()> {
    H5S_select_offset_into(space, offsets, out)
}

#[allow(non_snake_case)]
pub fn H5Sextent_equal(left: &Dataspace, right: &Dataspace) -> bool {
    left.extent_equal(right)
}

#[allow(non_snake_case)]
pub fn H5S_extent_equal(left: &Dataspace, right: &Dataspace) -> bool {
    H5Sextent_equal(left, right)
}

#[allow(non_snake_case)]
pub fn H5S_extent_nelem(space: &Dataspace) -> u64 {
    space.extent_nelem()
}

#[allow(non_snake_case)]
pub fn H5S_write_debug<W: Write + ?Sized>(space: &Dataspace, out: &mut W) -> fmt::Result {
    space.write_debug(out)
}

#[allow(non_snake_case)]
pub fn H5S_debug(space: &Dataspace) -> String {
    space.debug()
}

#[allow(non_snake_case)]
pub fn H5S_mpio_space_type(space: &Dataspace) -> DataspaceType {
    space.mpio_space_type()
}

#[allow(non_snake_case)]
pub fn H5S__obtain_datatype(space: &Dataspace) -> DataspaceType {
    space.obtain_datatype()
}

#[allow(non_snake_case)]
pub fn H5S_set_version(space: &mut Dataspace, version: u8) -> crate::Result<()> {
    space.set_version(version)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataspace_package_aliases_roundtrip() {
        assert!(Dataspace::init());
        assert!(Dataspace::init_package());
        Dataspace::top_term_package();
        Dataspace::term_package();

        let space = Dataspace::simple(vec![2, 3], None).unwrap();
        assert!(space.is_simple_internal());
        assert_eq!(space.mpio_space_type(), DataspaceType::Simple);
        assert_eq!(space.obtain_datatype(), DataspaceType::Simple);
        let mut debug = String::new();
        space.write_debug(&mut debug).unwrap();
        assert!(debug.contains("Simple"));
        assert_eq!(space.offset_simple(&[1, -1]).unwrap(), vec![1, -1]);
        let mut offsets = vec![99];
        space.offset_simple_into(&[1, -1], &mut offsets).unwrap();
        assert_eq!(offsets, vec![1, -1]);
        assert!(space.offset_simple(&[0]).is_err());
        space.close_cb();
    }

    #[test]
    fn h5s_extent_aliases_roundtrip() {
        assert!(H5S_init());
        assert!(H5S__init_package());
        H5S_top_term_package();
        H5S_term_package();

        let mut space = Dataspace::simple(vec![2, 3], Some(vec![4, u64::MAX])).unwrap();
        assert_eq!(H5S_get_simple_extent_ndims(&space), 2);
        assert_eq!(H5Sget_simple_extent_ndims(&space), 2);
        assert_eq!(
            H5Screate_simple(vec![2, 3], Some(vec![4, u64::MAX])).unwrap(),
            space
        );
        assert_eq!(
            H5S_extent_get_dims_ref(&space),
            (&[2, 3][..], Some(&[4, u64::MAX][..]))
        );
        assert_eq!(H5S_get_npoints_max(&space), u64::MAX);
        assert_eq!(H5S_extent_nelem(&space), 6);
        assert_eq!(H5S_get_simple_extent_npoints(&space), 6);
        assert_eq!(H5Sget_simple_extent_npoints(&space), 6);
        assert_eq!(H5S_get_simple_extent_type(&space), DataspaceType::Simple);
        assert_eq!(H5S_select_offset(&space, &[1, -1]).unwrap(), vec![1, -1]);
        let mut offsets = vec![99];
        H5Soffset_simple_into(&space, &[1, -1], &mut offsets).unwrap();
        assert_eq!(offsets, vec![1, -1]);
        assert!(H5Soffset_simple(&space, &[0]).is_err());
        assert!(H5S_has_extent(&space));
        assert!(H5S__is_simple(&space));
        assert!(H5Sis_simple(&space));
        let mut debug = String::new();
        H5S_write_debug(&space, &mut debug).unwrap();
        assert!(debug.contains("Simple"));
        assert_eq!(H5S_mpio_space_type(&space), DataspaceType::Simple);
        assert_eq!(H5S__obtain_datatype(&space), DataspaceType::Simple);

        let copied = H5S_copy(&space);
        assert!(H5S_extent_equal(&space, &copied));
        H5S_set_extent_real(&mut space, vec![5], Some(vec![10])).unwrap();
        assert!(!H5Sextent_equal(&space, &copied));
        H5Sset_extent_simple(&mut space, vec![2, 3], Some(vec![4, 5])).unwrap();
        assert_eq!(
            H5S_extent_get_dims_ref(&space),
            (&[2, 3][..], Some(&[4, 5][..]))
        );
        H5S_set_version(&mut space, 1).unwrap();
        assert_eq!(space.raw_message_ref().version, 1);
        H5S__close_cb(H5S_create(DataspaceType::Scalar));
    }

    #[test]
    fn dataspace_encode_roundtrips_current_message() {
        let space = Dataspace::simple(vec![2, 3], Some(vec![4, u64::MAX])).unwrap();
        let encoded = space.encode().unwrap();
        assert_eq!(encoded[..4], [2, 2, 1, 1]);
        assert_eq!(
            DataspaceMessage::decode(&encoded).unwrap(),
            space.raw_message()
        );

        let scalar = Dataspace::scalar();
        assert_eq!(scalar.encode().unwrap(), vec![2, 0, 0, 0]);

        let null = Dataspace::null();
        assert_eq!(null.encode().unwrap(), vec![2, 0, 0, 2]);
    }

    #[test]
    fn dataspace_encode_honors_v1_layout() {
        let mut space = Dataspace::simple(vec![6], None).unwrap();
        space.set_version(1).unwrap();

        let encoded = space.encode().unwrap();
        let mut expected = vec![1, 1, 0, 0, 0, 0, 0, 0];
        expected.extend_from_slice(&6u64.to_le_bytes());
        assert_eq!(encoded, expected);
        assert_eq!(
            DataspaceMessage::decode(&encoded).unwrap(),
            space.raw_message()
        );
    }
}
