/// Common interface for named HDF5 objects (File, Group, Dataset).
pub trait Location {
    /// Get the object's name/path within the file.
    fn name(&self) -> &str;

    /// List attribute names on this object.
    fn attr_names(&self) -> crate::Result<Vec<String>> {
        let mut names = Vec::new();
        self.attr_names_into(&mut names)?;
        Ok(names)
    }

    /// Visit attribute names on this object in storage order.
    fn visit_attr_names<F>(&self, f: F) -> crate::Result<()>
    where
        F: FnMut(&str) -> crate::Result<()>;

    /// Append attribute names on this object into caller-provided storage.
    fn attr_names_into(&self, out: &mut Vec<String>) -> crate::Result<()> {
        out.clear();
        self.visit_attr_names(|name| {
            out.push(name.to_string());
            Ok(())
        })
    }

    /// List attributes on this object.
    fn attrs(&self) -> crate::Result<Vec<crate::hl::attribute::Attribute>> {
        let mut attrs = Vec::new();
        self.attrs_into(&mut attrs)?;
        Ok(attrs)
    }

    /// Visit attributes on this object in storage order.
    fn visit_attrs<F>(&self, f: F) -> crate::Result<()>
    where
        F: FnMut(&crate::hl::attribute::Attribute) -> crate::Result<()>;

    /// Store attributes on this object in caller-provided storage.
    fn attrs_into(&self, out: &mut Vec<crate::hl::attribute::Attribute>) -> crate::Result<()> {
        out.clear();
        self.visit_attrs(|attr| {
            out.push(attr.clone());
            Ok(())
        })
    }

    /// Return the number of attributes on this object.
    fn attr_count(&self) -> crate::Result<usize> {
        let mut count = 0usize;
        self.visit_attr_names(|_| {
            count += 1;
            Ok(())
        })?;
        Ok(count)
    }

    /// Return an attribute name by zero-based storage-order index.
    fn attr_name_by_idx(&self, index: usize) -> crate::Result<String> {
        let mut name = String::new();
        self.attr_name_by_idx_into(index, &mut name)?;
        Ok(name)
    }

    /// Store an attribute name by zero-based storage-order index in caller-provided storage.
    fn attr_name_by_idx_into(&self, index: usize, out: &mut String) -> crate::Result<()> {
        let mut found = None;
        let mut pos = 0usize;
        self.visit_attr_names(|name| {
            if pos == index {
                found = Some(name.to_string());
            }
            pos += 1;
            Ok(())
        })?;
        match found {
            Some(name) => {
                *out = name;
                Ok(())
            }
            None => Err(crate::Error::InvalidFormat(format!(
                "attribute index {index} is out of bounds"
            ))),
        }
    }

    /// Return attribute metadata by zero-based storage-order index.
    fn attr_info_by_idx(&self, index: usize) -> crate::Result<crate::hl::attribute::AttributeInfo> {
        let mut found = None;
        let mut pos = 0usize;
        self.visit_attrs(|attr| {
            if pos == index {
                found = Some(attr.info());
            }
            pos += 1;
            Ok(())
        })?;
        found.ok_or_else(|| {
            crate::Error::InvalidFormat(format!("attribute index {index} is out of bounds"))
        })
    }

    /// List attributes sorted by tracked creation order.
    fn attrs_by_creation_order(&self) -> crate::Result<Vec<crate::hl::attribute::Attribute>> {
        let mut attrs = Vec::new();
        self.attrs_by_creation_order_into(&mut attrs)?;
        Ok(attrs)
    }

    /// Visit attributes sorted by tracked creation order.
    fn visit_attrs_by_creation_order<F>(&self, mut f: F) -> crate::Result<()>
    where
        F: FnMut(&crate::hl::attribute::Attribute) -> crate::Result<()>,
    {
        let mut attrs = Vec::new();
        self.attrs_by_creation_order_into(&mut attrs)?;
        for attr in attrs.iter() {
            f(attr)?;
        }
        Ok(())
    }

    /// Store attributes sorted by tracked creation order in caller-provided storage.
    fn attrs_by_creation_order_into(
        &self,
        out: &mut Vec<crate::hl::attribute::Attribute>,
    ) -> crate::Result<()> {
        self.attrs_into(out)?;
        sort_attrs_by_creation_order(out)
    }

    /// Get an attribute by name.
    fn attr(&self, name: &str) -> crate::Result<crate::hl::attribute::Attribute>;

    /// Check whether an attribute exists on this object.
    fn attr_exists(&self, name: &str) -> crate::Result<bool> {
        let mut found = false;
        self.visit_attr_names(|attr_name| {
            if attr_name == name {
                found = true;
            }
            Ok(())
        })?;
        Ok(found)
    }
}

impl Location for crate::hl::file::File {
    fn name(&self) -> &str {
        "/"
    }
    fn visit_attr_names<F>(&self, f: F) -> crate::Result<()>
    where
        F: FnMut(&str) -> crate::Result<()>,
    {
        crate::hl::file::File::visit_attr_names(self, f)
    }
    fn attr_names_into(&self, out: &mut Vec<String>) -> crate::Result<()> {
        crate::hl::file::File::attr_names_into(self, out)
    }
    fn visit_attrs<F>(&self, f: F) -> crate::Result<()>
    where
        F: FnMut(&crate::hl::attribute::Attribute) -> crate::Result<()>,
    {
        crate::hl::file::File::visit_attrs(self, f)
    }
    fn attrs_into(&self, out: &mut Vec<crate::hl::attribute::Attribute>) -> crate::Result<()> {
        crate::hl::file::File::attrs_into(self, out)
    }
    fn visit_attrs_by_creation_order<F>(&self, f: F) -> crate::Result<()>
    where
        F: FnMut(&crate::hl::attribute::Attribute) -> crate::Result<()>,
    {
        crate::hl::file::File::visit_attrs_by_creation_order(self, f)
    }
    fn attrs_by_creation_order_into(
        &self,
        out: &mut Vec<crate::hl::attribute::Attribute>,
    ) -> crate::Result<()> {
        crate::hl::file::File::attrs_by_creation_order_into(self, out)
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
    fn visit_attr_names<F>(&self, f: F) -> crate::Result<()>
    where
        F: FnMut(&str) -> crate::Result<()>,
    {
        crate::hl::group::Group::visit_attr_names(self, f)
    }
    fn attr_names_into(&self, out: &mut Vec<String>) -> crate::Result<()> {
        crate::hl::group::Group::attr_names_into(self, out)
    }
    fn visit_attrs<F>(&self, f: F) -> crate::Result<()>
    where
        F: FnMut(&crate::hl::attribute::Attribute) -> crate::Result<()>,
    {
        crate::hl::group::Group::visit_attrs(self, f)
    }
    fn attrs_into(&self, out: &mut Vec<crate::hl::attribute::Attribute>) -> crate::Result<()> {
        crate::hl::group::Group::attrs_into(self, out)
    }
    fn visit_attrs_by_creation_order<F>(&self, f: F) -> crate::Result<()>
    where
        F: FnMut(&crate::hl::attribute::Attribute) -> crate::Result<()>,
    {
        crate::hl::group::Group::visit_attrs_by_creation_order(self, f)
    }
    fn attrs_by_creation_order_into(
        &self,
        out: &mut Vec<crate::hl::attribute::Attribute>,
    ) -> crate::Result<()> {
        crate::hl::group::Group::attrs_by_creation_order_into(self, out)
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
    fn visit_attr_names<F>(&self, f: F) -> crate::Result<()>
    where
        F: FnMut(&str) -> crate::Result<()>,
    {
        crate::hl::dataset::Dataset::visit_attr_names(self, f)
    }
    fn attr_names_into(&self, out: &mut Vec<String>) -> crate::Result<()> {
        crate::hl::dataset::Dataset::attr_names_into(self, out)
    }
    fn visit_attrs<F>(&self, f: F) -> crate::Result<()>
    where
        F: FnMut(&crate::hl::attribute::Attribute) -> crate::Result<()>,
    {
        crate::hl::dataset::Dataset::visit_attrs(self, f)
    }
    fn attrs_into(&self, out: &mut Vec<crate::hl::attribute::Attribute>) -> crate::Result<()> {
        crate::hl::dataset::Dataset::attrs_into(self, out)
    }
    fn visit_attrs_by_creation_order<F>(&self, f: F) -> crate::Result<()>
    where
        F: FnMut(&crate::hl::attribute::Attribute) -> crate::Result<()>,
    {
        crate::hl::dataset::Dataset::visit_attrs_by_creation_order(self, f)
    }
    fn attrs_by_creation_order_into(
        &self,
        out: &mut Vec<crate::hl::attribute::Attribute>,
    ) -> crate::Result<()> {
        crate::hl::dataset::Dataset::attrs_by_creation_order_into(self, out)
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
        Err(crate::Error::InvalidFormat(msg)) if msg.contains("not found") => Ok(false),
        Err(err) => Err(err),
    }
}

fn sort_attrs_by_creation_order(
    attrs: &mut Vec<crate::hl::attribute::Attribute>,
) -> crate::Result<()> {
    if attrs.iter().any(|attr| attr.creation_order().is_none()) {
        attrs.clear();
        return Err(crate::Error::Unsupported(
            "object does not track attribute creation order".into(),
        ));
    }
    attrs.sort_by_key(|attr| attr.creation_order().unwrap_or(u64::MAX));
    Ok(())
}
