use crate::format::messages::dataspace::{DataspaceMessage, DataspaceType};

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
    pub fn raw_message(&self) -> DataspaceMessage {
        self.msg.clone()
    }

    /// Return the parsed simple extent metadata.
    pub fn simple_extent(&self) -> DataspaceMessage {
        self.raw_message()
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

    /// Validate and return a simple dataspace offset vector.
    pub fn offset_simple(&self, offsets: &[i64]) -> crate::Result<Vec<i64>> {
        if offsets.len() != self.ndim() {
            return Err(crate::Error::InvalidFormat(format!(
                "dataspace offset rank {} does not match dataspace rank {}",
                offsets.len(),
                self.ndim()
            )));
        }
        Ok(offsets.to_vec())
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
    pub fn debug(&self) -> String {
        format!("{:?}", self.msg)
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
pub fn H5S_extent_get_dims(space: &Dataspace) -> (Vec<u64>, Option<Vec<u64>>) {
    let (dims, max_dims) = space.extent_dims();
    (dims.to_vec(), max_dims.map(|dims| dims.to_vec()))
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
    Dataspace::from_message(space.simple_extent())
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
pub fn H5Soffset_simple(space: &Dataspace, offsets: &[i64]) -> crate::Result<Vec<i64>> {
    H5S_select_offset(space, offsets)
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
        assert!(space.debug().contains("Simple"));
        assert_eq!(space.offset_simple(&[1, -1]).unwrap(), vec![1, -1]);
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
            H5S_extent_get_dims(&space),
            (vec![2, 3], Some(vec![4, u64::MAX]))
        );
        assert_eq!(H5S_get_npoints_max(&space), u64::MAX);
        assert_eq!(H5S_extent_nelem(&space), 6);
        assert_eq!(H5S_get_simple_extent_npoints(&space), 6);
        assert_eq!(H5Sget_simple_extent_npoints(&space), 6);
        assert_eq!(H5S_get_simple_extent_type(&space), DataspaceType::Simple);
        assert_eq!(H5S_select_offset(&space, &[1, -1]).unwrap(), vec![1, -1]);
        assert!(H5Soffset_simple(&space, &[0]).is_err());
        assert!(H5S_has_extent(&space));
        assert!(H5S__is_simple(&space));
        assert!(H5Sis_simple(&space));
        assert!(H5S_debug(&space).contains("Simple"));
        assert_eq!(H5S_mpio_space_type(&space), DataspaceType::Simple);
        assert_eq!(H5S__obtain_datatype(&space), DataspaceType::Simple);

        let copied = H5S_copy(&space);
        assert!(H5S_extent_equal(&space, &copied));
        H5S_set_extent_real(&mut space, vec![5], Some(vec![10])).unwrap();
        assert!(!H5Sextent_equal(&space, &copied));
        H5Sset_extent_simple(&mut space, vec![2, 3], Some(vec![4, 5])).unwrap();
        assert_eq!(
            H5S_get_simple_extent_dims(&space),
            (vec![2, 3], Some(vec![4, 5]))
        );
        H5S_set_version(&mut space, 1).unwrap();
        assert_eq!(space.raw_message().version, 1);
        H5S__close_cb(H5S_create(DataspaceType::Scalar));
    }
}
