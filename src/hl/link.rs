use std::collections::BTreeMap;

use crate::error::{Error, Result};
use crate::format::messages::link::LinkMessage;
use crate::hl::group::{Group, LinkInfo, LinkMessageRef, LinkValue};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinkClass {
    pub id: u8,
    pub name: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LinkClassRegistry {
    classes: BTreeMap<u8, LinkClass>,
    external_registered: bool,
}

impl LinkClassRegistry {
    pub fn init() -> Self {
        let mut registry = Self::default();
        registry.register(LinkClass {
            id: 0,
            name: "hard".to_string(),
        });
        registry.register(LinkClass {
            id: 1,
            name: "soft".to_string(),
        });
        registry.register_external();
        registry
    }

    pub fn init_package() -> Self {
        Self::init()
    }

    pub fn term_package(&mut self) {
        self.classes.clear();
        self.external_registered = false;
    }

    pub fn register_external(&mut self) {
        self.external_registered = true;
        self.register(LinkClass {
            id: 64,
            name: "external".to_string(),
        });
    }

    pub fn find_class_idx(&self, id: u8) -> Option<usize> {
        self.classes.keys().position(|class_id| *class_id == id)
    }

    pub fn find_class(&self, id: u8) -> Option<&LinkClass> {
        self.classes.get(&id)
    }

    pub fn register(&mut self, class: LinkClass) {
        self.classes.insert(class.id, class);
    }

    pub fn unregister(&mut self, id: u8) -> Option<LinkClass> {
        if id == 64 {
            self.external_registered = false;
        }
        self.classes.remove(&id)
    }

    pub fn is_registered(&self, id: u8) -> bool {
        self.classes.contains_key(&id)
    }

    pub fn register_api(&mut self, class: LinkClass) {
        self.register(class);
    }

    pub fn unregister_api(&mut self, id: u8) -> Option<LinkClass> {
        self.unregister(id)
    }

    pub fn is_registered_api(&self, id: u8) -> bool {
        self.is_registered(id)
    }
}

pub fn extern_traverse_with<R, F>(group: &Group, name: &str, visitor: F) -> Result<R>
where
    F: FnOnce(Option<(&str, &str)>) -> Result<R>,
{
    group.with_link_by_name(name, |link| {
        visitor(
            link.external_link
                .as_ref()
                .map(|(filename, object_path)| (filename.as_str(), object_path.as_str())),
        )
    })
}

pub fn link(group: &Group, name: &str) -> Result<LinkMessage> {
    link_with(group, name, |link| Ok(link.clone()))
}

pub fn link_with<R, F>(group: &Group, name: &str, visitor: F) -> Result<R>
where
    F: FnOnce(&LinkMessage) -> Result<R>,
{
    group.with_link_by_name(name, visitor)
}

pub fn link_into(group: &Group, name: &str, out: &mut LinkMessage) -> Result<()> {
    link_with(group, name, |link| {
        out.clone_from(link);
        Ok(())
    })
}

pub fn link_object(group: &Group, name: &str) -> Result<u64> {
    group.with_link_by_name(name, |link| {
        link.hard_link_addr.ok_or_else(|| {
            Error::InvalidFormat(format!("link '{name}' does not reference an object header"))
        })
    })
}

pub fn create_soft_api_common(
    writer: &mut crate::hl::writable_file::WritableFile,
    name: &str,
    target: &str,
) -> Result<()> {
    writer.link_soft(name, target)
}

pub fn create_hard_api_common(
    writer: &mut crate::hl::writable_file::WritableFile,
    name: &str,
    target: &str,
) -> Result<()> {
    writer.link_hard(name, target)
}

pub fn create_real(group: &Group, name: &str) -> Result<u64> {
    link_object(group, name)
}

pub fn create_soft(
    writer: &mut crate::hl::writable_file::WritableFile,
    name: &str,
    target: &str,
) -> Result<()> {
    writer.link_soft(name, target)
}

pub fn create_soft_async(
    writer: &mut crate::hl::writable_file::WritableFile,
    name: &str,
    target: &str,
) -> Result<()> {
    create_soft(writer, name, target)
}

pub fn create_hard(
    writer: &mut crate::hl::writable_file::WritableFile,
    name: &str,
    target: &str,
) -> Result<()> {
    writer.link_hard(name, target)
}

pub fn create_hard_async(
    writer: &mut crate::hl::writable_file::WritableFile,
    name: &str,
    target: &str,
) -> Result<()> {
    create_hard(writer, name, target)
}

pub fn create_external(
    writer: &mut crate::hl::writable_file::WritableFile,
    name: &str,
    filename: &str,
    object_path: &str,
) -> Result<()> {
    writer.link_external(name, filename, object_path)
}

pub fn create_ud(
    writer: &mut crate::hl::writable_file::WritableFile,
    name: &str,
    filename: &str,
    object_path: &str,
) -> Result<()> {
    writer.link_external(name, filename, object_path)
}

pub fn create_ud_api(
    writer: &mut crate::hl::writable_file::WritableFile,
    name: &str,
    filename: &str,
    object_path: &str,
) -> Result<()> {
    create_ud(writer, name, filename, object_path)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkValueRef<'a> {
    Soft(&'a str),
    External {
        filename: &'a str,
        object_path: &'a str,
    },
}

impl LinkValueRef<'_> {
    pub fn to_owned(self) -> LinkValue {
        match self {
            LinkValueRef::Soft(target) => LinkValue::Soft(target.to_string()),
            LinkValueRef::External {
                filename,
                object_path,
            } => LinkValue::External {
                filename: filename.to_string(),
                object_path: object_path.to_string(),
            },
        }
    }
}

pub(crate) fn get_val_ref_borrowed(link: LinkMessageRef<'_>) -> Option<LinkValueRef<'_>> {
    if let Some(target) = link.soft_link_target {
        return Some(LinkValueRef::Soft(target));
    }
    link.external_link
        .map(|(filename, object_path)| LinkValueRef::External {
            filename,
            object_path,
        })
}

pub fn get_val_cb_borrowed(link: &LinkMessage) -> Option<LinkValueRef<'_>> {
    if let Some(target) = &link.soft_link_target {
        return Some(LinkValueRef::Soft(target));
    }
    link.external_link
        .as_ref()
        .map(|(filename, object_path)| LinkValueRef::External {
            filename,
            object_path,
        })
}

pub fn get_val_with<R, F>(group: &Group, name: &str, visitor: F) -> Result<R>
where
    F: FnOnce(Option<LinkValueRef<'_>>) -> Result<R>,
{
    group.with_link_ref_by_name(name, |link| visitor(get_val_ref_borrowed(link)))
}

pub fn get_val(group: &Group, name: &str) -> Result<Option<LinkValue>> {
    get_val_with(group, name, |value| Ok(value.map(LinkValueRef::to_owned)))
}

pub fn get_val_into(group: &Group, name: &str, out: &mut Option<LinkValue>) -> Result<()> {
    let value = get_val(group, name)?;
    *out = value;
    Ok(())
}

pub fn get_val_by_idx_cb_borrowed(link: &LinkMessage) -> Option<LinkValueRef<'_>> {
    get_val_cb_borrowed(link)
}

pub fn get_val_by_idx_with<R, F>(group: &Group, index: usize, visitor: F) -> Result<R>
where
    F: FnOnce(Option<LinkValueRef<'_>>) -> Result<R>,
{
    let mut visitor = Some(visitor);
    let mut result = None;
    let mut pos = 0usize;
    group.visit_link_refs_for_link_access(|link| {
        if pos == index {
            let visit = visitor
                .take()
                .expect("link index visitor called more than once");
            result = Some(visit(get_val_ref_borrowed(link))?);
        }
        pos += 1;
        Ok(())
    })?;
    result.ok_or_else(|| Error::InvalidFormat(format!("link index {index} is out of bounds")))
}

pub fn get_val_by_idx(group: &Group, index: usize) -> Result<Option<LinkValue>> {
    get_val_by_idx_with(group, index, |value| Ok(value.map(LinkValueRef::to_owned)))
}

pub fn get_val_by_idx_into(group: &Group, index: usize, out: &mut Option<LinkValue>) -> Result<()> {
    let value = get_val_by_idx(group, index)?;
    *out = value;
    Ok(())
}

pub fn exists_final_cb(group: &Group, name: &str) -> Result<bool> {
    group.link_exists(name)
}

pub fn exists_inter_cb(group: &Group, name: &str) -> Result<bool> {
    group.link_exists(name)
}

pub fn exists_tolerant(group: &Group, name: &str) -> bool {
    group.link_exists(name).unwrap_or(false)
}

pub fn exists(group: &Group, name: &str) -> Result<bool> {
    group.link_exists(name)
}

pub fn exists_api_common(group: &Group, name: &str) -> Result<bool> {
    exists(group, name)
}

pub fn get_info_by_idx_cb(link: &LinkMessage) -> Result<LinkInfo> {
    super::group::link_info_from_message(link)
}

pub fn get_info_by_idx(group: &Group, index: usize) -> Result<LinkInfo> {
    group.link_info_by_idx(index)
}

pub fn get_name_by_idx(group: &Group, index: usize) -> Result<String> {
    get_name_by_idx_with(group, index, |name| Ok(name.to_string()))
}

pub fn get_name_by_idx_with<R, F>(group: &Group, index: usize, visitor: F) -> Result<R>
where
    F: FnOnce(&str) -> Result<R>,
{
    let mut visitor = Some(visitor);
    let mut result = None;
    let mut pos = 0usize;
    group.visit_link_refs_for_link_access(|link| {
        if pos == index {
            let visit = visitor
                .take()
                .expect("link name index visitor called more than once");
            result = Some(visit(link.name)?);
        }
        pos += 1;
        Ok(())
    })?;
    result.ok_or_else(|| Error::InvalidFormat(format!("link index {index} is out of bounds")))
}

pub fn get_name_by_idx_into(group: &Group, index: usize, out: &mut String) -> Result<()> {
    get_name_by_idx_with(group, index, |name| {
        out.clear();
        out.push_str(name);
        Ok(())
    })
}

pub fn link_copy_file_into(link: &LinkMessage, out: &mut LinkMessage) {
    out.clone_from(link);
}

pub fn copy_into(link: &LinkMessage, out: &mut LinkMessage) {
    link_copy_file_into(link, out)
}

pub fn iterate(group: &Group) -> Result<Vec<LinkMessage>> {
    collect_links(group)
}

pub fn iterate_with<F>(group: &Group, visitor: F) -> Result<()>
where
    F: FnMut(&LinkMessage) -> Result<()>,
{
    group.visit_links(visitor)
}

pub fn iterate_by_name2(group: &Group) -> Result<Vec<LinkMessage>> {
    collect_links(group)
}

pub fn visit2(group: &Group) -> Result<Vec<LinkMessage>> {
    collect_links(group)
}

pub fn visit_with<F>(group: &Group, visitor: F) -> Result<()>
where
    F: FnMut(&LinkMessage) -> Result<()>,
{
    group.visit_links(visitor)
}

pub fn visit_by_name2(group: &Group) -> Result<Vec<LinkMessage>> {
    collect_links(group)
}

pub fn iterate1(group: &Group) -> Result<Vec<LinkMessage>> {
    collect_links(group)
}

pub fn iterate_by_name1(group: &Group) -> Result<Vec<LinkMessage>> {
    collect_links(group)
}

pub fn visit1(group: &Group) -> Result<Vec<LinkMessage>> {
    collect_links(group)
}

pub fn visit_by_name1(group: &Group) -> Result<Vec<LinkMessage>> {
    collect_links(group)
}

pub fn get_ocrt_info(link: &LinkMessage) -> Option<u64> {
    link.creation_order
}

fn collect_links(group: &Group) -> Result<Vec<LinkMessage>> {
    group.links()
}

#[derive(Debug, Clone, Default)]
pub struct LinkTable {
    links: BTreeMap<String, LinkMessage>,
}

impl LinkTable {
    pub fn from_links(links: impl IntoIterator<Item = LinkMessage>) -> Self {
        let links = links
            .into_iter()
            .map(|link| (link.name.clone(), link))
            .collect();
        Self { links }
    }

    pub fn iter(&self) -> impl Iterator<Item = &LinkMessage> {
        self.links.values()
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.links.keys().map(String::as_str)
    }

    pub fn get(&self, name: &str) -> Option<&LinkMessage> {
        self.links.get(name)
    }

    pub fn visit<F>(&self, mut visitor: F) -> Result<()>
    where
        F: FnMut(&LinkMessage) -> Result<()>,
    {
        for link in self.links.values() {
            visitor(link)?;
        }
        Ok(())
    }

    pub fn insert(&mut self, link: LinkMessage) -> Option<LinkMessage> {
        self.links.insert(link.name.clone(), link)
    }

    pub fn delete(&mut self, name: &str) -> Result<LinkMessage> {
        self.links
            .remove(name)
            .ok_or_else(|| Error::InvalidFormat(format!("link '{name}' not found")))
    }

    pub fn delete_by_idx(&mut self, index: usize) -> Result<LinkMessage> {
        let name = self
            .links
            .keys()
            .nth(index)
            .cloned()
            .ok_or_else(|| Error::InvalidFormat(format!("link index {index} out of range")))?;
        self.delete(&name)
    }

    pub fn move_link(&mut self, old_name: &str, new_name: &str) -> Result<()> {
        if self.links.contains_key(new_name) {
            return Err(Error::InvalidFormat(format!(
                "destination link '{new_name}' already exists"
            )));
        }
        let mut link = self.delete(old_name)?;
        link.name = new_name.to_string();
        self.insert(link);
        Ok(())
    }
}

#[allow(non_snake_case)]
pub fn H5L__delete_cb(table: &mut LinkTable, name: &str) -> Result<LinkMessage> {
    table.delete(name)
}

#[allow(non_snake_case)]
pub fn H5L__delete(table: &mut LinkTable, name: &str) -> Result<LinkMessage> {
    H5L__delete_cb(table, name)
}

#[allow(non_snake_case)]
pub fn H5L__delete_by_idx_cb(table: &mut LinkTable, index: usize) -> Result<LinkMessage> {
    table.delete_by_idx(index)
}

#[allow(non_snake_case)]
pub fn H5L__delete_by_idx(table: &mut LinkTable, index: usize) -> Result<LinkMessage> {
    H5L__delete_by_idx_cb(table, index)
}

#[allow(non_snake_case)]
pub fn H5L__delete_by_idx_api_common(table: &mut LinkTable, index: usize) -> Result<LinkMessage> {
    H5L__delete_by_idx(table, index)
}

#[allow(non_snake_case)]
pub fn H5L__move(table: &mut LinkTable, old_name: &str, new_name: &str) -> Result<()> {
    table.move_link(old_name, new_name)
}

#[allow(non_snake_case)]
pub fn H5Lmove(table: &mut LinkTable, old_name: &str, new_name: &str) -> Result<()> {
    H5L__move(table, old_name, new_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn link_registry_tracks_builtin_classes() {
        let mut registry = LinkClassRegistry::init();
        assert!(registry.is_registered(0));
        assert!(registry.is_registered(1));
        assert!(registry.is_registered(64));
        assert_eq!(registry.find_class_idx(64), Some(2));
        registry.unregister(64);
        assert!(!registry.is_registered(64));
    }

    #[test]
    fn link_table_deletes_and_moves_links() {
        let mut table = LinkTable::default();
        table.insert(LinkMessage {
            link_type: crate::format::messages::link::LinkType::Soft,
            creation_order: None,
            char_encoding: 0,
            name: "old".into(),
            hard_link_addr: None,
            soft_link_target: Some("/target".into()),
            external_link: None,
        });
        H5Lmove(&mut table, "old", "new").unwrap();
        {
            let mut names = table.names();
            assert_eq!(names.next(), Some("new"));
            assert_eq!(names.next(), None);
        }
        assert_eq!(table.get("new").map(|link| link.name.as_str()), Some("new"));
        let mut visited = false;
        table
            .visit(|link| {
                assert_eq!(link.name.as_str(), "new");
                visited = true;
                Ok(())
            })
            .unwrap();
        assert!(visited);
        assert!(H5L__delete(&mut table, "new").is_ok());
        assert!(H5L__delete(&mut table, "new").is_err());
    }

    #[test]
    fn link_value_ref_borrows_soft_targets() {
        let link = LinkMessage {
            link_type: crate::format::messages::link::LinkType::Soft,
            creation_order: None,
            char_encoding: 0,
            name: "soft".into(),
            hard_link_addr: None,
            soft_link_target: Some("/target".into()),
            external_link: None,
        };

        assert_eq!(
            get_val_cb_borrowed(&link),
            Some(LinkValueRef::Soft("/target"))
        );
        assert_eq!(
            get_val_by_idx_cb_borrowed(&link),
            Some(LinkValueRef::Soft("/target"))
        );
    }

    #[test]
    fn link_value_ref_borrows_external_targets() {
        let link = LinkMessage {
            link_type: crate::format::messages::link::LinkType::External,
            creation_order: None,
            char_encoding: 0,
            name: "external".into(),
            hard_link_addr: None,
            soft_link_target: None,
            external_link: Some(("file.h5".into(), "/object".into())),
        };

        assert_eq!(
            get_val_cb_borrowed(&link),
            Some(LinkValueRef::External {
                filename: "file.h5",
                object_path: "/object"
            })
        );
    }

    #[test]
    fn link_copy_file_into_reuses_output_storage() {
        let link = LinkMessage {
            link_type: crate::format::messages::link::LinkType::Soft,
            creation_order: None,
            char_encoding: 0,
            name: "soft".into(),
            hard_link_addr: None,
            soft_link_target: Some("/target".into()),
            external_link: None,
        };
        let mut out = LinkMessage {
            link_type: crate::format::messages::link::LinkType::Hard,
            creation_order: None,
            char_encoding: 0,
            name: "hard".into(),
            hard_link_addr: Some(42),
            soft_link_target: None,
            external_link: None,
        };

        link_copy_file_into(&link, &mut out);

        assert_eq!(out.name, link.name);
        assert_eq!(out.link_type, link.link_type);
        assert_eq!(out.hard_link_addr, link.hard_link_addr);
        assert_eq!(out.soft_link_target, link.soft_link_target);
        assert_eq!(out.external_link, link.external_link);
    }

    #[test]
    fn link_value_into_replaces_on_success_and_preserves_on_missing_name() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("link_value_into.h5");

        {
            let mut writer = crate::hl::writable_file::WritableFile::create(&path).unwrap();
            writer
                .new_dataset_builder("hard")
                .write::<i32>(&[1])
                .unwrap();
            writer.link_soft("soft", "/hard").unwrap();
            writer
                .link_external("external", "missing.h5", "/remote")
                .unwrap();
            writer.flush().unwrap();
        }

        let file = crate::hl::file::File::open(&path).unwrap();
        let root = file.root_group().unwrap();
        let mut value = Some(LinkValue::External {
            filename: "stale.h5".into(),
            object_path: "/stale".into(),
        });

        get_val_into(&root, "soft", &mut value).unwrap();
        assert_eq!(value, Some(LinkValue::Soft("/hard".into())));

        get_val_into(&root, "hard", &mut value).unwrap();
        assert_eq!(value, None);

        value = Some(LinkValue::Soft("stale".into()));
        let err = get_val_into(&root, "missing", &mut value)
            .expect_err("missing link should fail without touching caller output");
        assert!(err.to_string().contains("not found"));
        assert_eq!(value, Some(LinkValue::Soft("stale".into())));
    }

    #[test]
    fn link_value_by_idx_into_replaces_on_success_and_preserves_on_missing_index() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("link_value_by_idx_into.h5");

        {
            let mut writer = crate::hl::writable_file::WritableFile::create(&path).unwrap();
            writer.link_soft("soft", "/target").unwrap();
            writer.flush().unwrap();
        }

        let file = crate::hl::file::File::open(&path).unwrap();
        let root = file.root_group().unwrap();
        let mut value = Some(LinkValue::External {
            filename: "stale.h5".into(),
            object_path: "/stale".into(),
        });

        get_val_by_idx_into(&root, 0, &mut value).unwrap();
        assert_eq!(value, Some(LinkValue::Soft("/target".into())));

        value = Some(LinkValue::Soft("stale".into()));
        let err = get_val_by_idx_into(&root, 1, &mut value)
            .expect_err("missing link index should fail without touching caller output");
        assert!(err.to_string().contains("out of bounds"));
        assert_eq!(value, Some(LinkValue::Soft("stale".into())));
    }
}
