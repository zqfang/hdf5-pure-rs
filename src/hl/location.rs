/// Common interface for named HDF5 objects (File, Group, Dataset).
pub trait Location {
    /// Get the object's name/path within the file.
    fn name(&self) -> &str;

    /// List attribute names on this object.
    fn attr_names(&self) -> crate::Result<Vec<String>>;

    /// List attributes on this object.
    fn attrs(&self) -> crate::Result<Vec<crate::hl::attribute::Attribute>> {
        self.attr_names()?
            .into_iter()
            .map(|name| self.attr(&name))
            .collect()
    }

    /// Return the number of attributes on this object.
    fn attr_count(&self) -> crate::Result<usize> {
        Ok(self.attr_names()?.len())
    }

    /// Return an attribute name by zero-based storage-order index.
    fn attr_name_by_idx(&self, index: usize) -> crate::Result<String> {
        self.attr_names()?.get(index).cloned().ok_or_else(|| {
            crate::Error::InvalidFormat(format!("attribute index {index} is out of bounds"))
        })
    }

    /// Return attribute metadata by zero-based storage-order index.
    fn attr_info_by_idx(&self, index: usize) -> crate::Result<crate::hl::attribute::AttributeInfo> {
        self.attrs()?
            .get(index)
            .map(|attr| attr.info())
            .ok_or_else(|| {
                crate::Error::InvalidFormat(format!("attribute index {index} is out of bounds"))
            })
    }

    /// List attributes sorted by tracked creation order.
    fn attrs_by_creation_order(&self) -> crate::Result<Vec<crate::hl::attribute::Attribute>> {
        let mut attrs = self.attrs()?;
        if attrs.iter().any(|attr| attr.creation_order().is_none()) {
            return Err(crate::Error::Unsupported(
                "object does not track attribute creation order".into(),
            ));
        }
        attrs.sort_by_key(|attr| attr.creation_order().unwrap_or(u64::MAX));
        Ok(attrs)
    }

    /// Get an attribute by name.
    fn attr(&self, name: &str) -> crate::Result<crate::hl::attribute::Attribute>;

    /// Check whether an attribute exists on this object.
    fn attr_exists(&self, name: &str) -> crate::Result<bool> {
        let names = self.attr_names()?;
        Ok(names.iter().any(|attr_name| attr_name == name))
    }
}

impl Location for crate::hl::file::File {
    fn name(&self) -> &str {
        "/"
    }
    fn attr_names(&self) -> crate::Result<Vec<String>> {
        self.attr_names()
    }
    fn attrs(&self) -> crate::Result<Vec<crate::hl::attribute::Attribute>> {
        self.attrs()
    }
    fn attrs_by_creation_order(&self) -> crate::Result<Vec<crate::hl::attribute::Attribute>> {
        self.attrs_by_creation_order()
    }
    fn attr(&self, name: &str) -> crate::Result<crate::hl::attribute::Attribute> {
        self.attr(name)
    }
    fn attr_exists(&self, name: &str) -> crate::Result<bool> {
        self.attr_exists(name)
    }
}

impl Location for crate::hl::group::Group {
    fn name(&self) -> &str {
        self.name()
    }
    fn attr_names(&self) -> crate::Result<Vec<String>> {
        self.attr_names()
    }
    fn attrs(&self) -> crate::Result<Vec<crate::hl::attribute::Attribute>> {
        self.attrs()
    }
    fn attrs_by_creation_order(&self) -> crate::Result<Vec<crate::hl::attribute::Attribute>> {
        self.attrs_by_creation_order()
    }
    fn attr(&self, name: &str) -> crate::Result<crate::hl::attribute::Attribute> {
        self.attr(name)
    }
    fn attr_exists(&self, name: &str) -> crate::Result<bool> {
        self.attr_exists(name)
    }
}

impl Location for crate::hl::dataset::Dataset {
    fn name(&self) -> &str {
        self.name()
    }
    fn attr_names(&self) -> crate::Result<Vec<String>> {
        self.attr_names()
    }
    fn attrs(&self) -> crate::Result<Vec<crate::hl::attribute::Attribute>> {
        self.attrs()
    }
    fn attrs_by_creation_order(&self) -> crate::Result<Vec<crate::hl::attribute::Attribute>> {
        self.attrs_by_creation_order()
    }
    fn attr(&self, name: &str) -> crate::Result<crate::hl::attribute::Attribute> {
        self.attr(name)
    }
    fn attr_exists(&self, name: &str) -> crate::Result<bool> {
        self.attr_exists(name)
    }
}

/// Check if a named member exists in a group.
pub fn link_exists(group: &crate::hl::group::Group, name: &str) -> crate::Result<bool> {
    match group.find_link_by_name(name) {
        Ok(_) => Ok(true),
        Err(crate::Error::InvalidFormat(msg)) if msg.contains("not found") => {
            let members = group.members()?;
            Ok(members.iter().any(|(member_name, _)| member_name == name))
        }
        Err(err) => Err(err),
    }
}
