pub mod engine;
pub mod error;
pub mod filters;
pub mod format;
pub mod hl;
pub mod io;

pub use error::{Error, Result};
pub use format::messages::dataspace::DataspaceType;
pub use hl::attribute::{Attribute, AttributeInfo};
pub use hl::context::{
    ApiContext, BackgroundBufferType, IoTransferMode, MpioCollectiveOpt, SelectionIoMode,
};
pub use hl::dataset::{
    ChunkInfo, Dataset, DatasetAccess, DatasetSpaceStatus, VdsMissingSourcePolicy, VdsView,
};
pub use hl::dataset_builder::DatasetBuilder;
pub use hl::dataspace::{Dataspace, Extents};
pub use hl::datatype::{Datatype, TypeDescriptor};
pub use hl::file::{
    File, FileBuilder, FileCreateBuilder, FileInfo, FileIntent, FreeSpaceInfo,
    MetadataCacheImageInfo, MetadataCacheSize, OpenMode, PageBufferingStats, SharedMessageInfo,
    SuperblockInfo,
};
pub use hl::group::{Group, GroupInfo, LinkInfo, LinkValue, ObjectInfo};
pub use hl::location::Location;
pub use hl::mutable_file::MutableFile;
pub use hl::plist::data_transfer::DataTransfer;
pub use hl::plist::dataset_create::{
    IrregularHyperslabBlockInfo, VirtualMappingInfo, VirtualSelectionInfo,
};
pub use hl::plist::file_access::{
    FileCloseDegree, LibverBound, MetadataCacheConfig, MetadataCacheImageConfig,
    MetadataCacheLogOptions,
};
pub use hl::plist::file_create::{FileSpaceStrategy, SharedMessageIndex};
pub use hl::plist::link_access::LinkAccess;
pub use hl::plist::object_copy::ObjectCopy;
pub use hl::plist::object_create::ObjectCreate;
pub use hl::selection::{
    HyperslabDim, IntoSelection, IntoSliceDim, RawSelection, Selection, SelectionType, SliceInfo,
};
pub use hl::types::H5Type;
pub use hl::value::H5Value;
pub use hl::writable_file::WritableFile;

/// Re-export the derive macro when the `derive` feature is enabled.
#[cfg(feature = "derive")]
pub use hdf5_pure_rust_derive::H5Type as DeriveH5Type;
