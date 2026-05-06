use std::collections::BTreeMap;

/// Ordered key/value container mirroring the externally visible H5SL API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkipList<K, V> {
    items: BTreeMap<K, V>,
}

impl<K: Ord, V> Default for SkipList<K, V> {
    fn default() -> Self {
        Self {
            items: BTreeMap::new(),
        }
    }
}

impl<K: Ord + Clone, V> SkipList<K, V> {
    /// Initialize skip-list package support.
    pub fn init_package() -> bool {
        true
    }

    /// Terminate skip-list package support.
    pub fn term_package() {}

    /// Create a new list node representation.
    pub fn new_node(key: K, value: V) -> (K, V) {
        (key, value)
    }

    /// Insert a node into a list, replacing any existing value for the key.
    pub fn insert_common(&mut self, key: K, value: V) -> Option<V> {
        self.items.insert(key, value)
    }

    /// Common close hook. The pure Rust list is consumed.
    pub fn close_common(self) {}

    /// Create a skip list.
    pub fn create() -> Self {
        Self::default()
    }

    /// Return number of items.
    pub fn count(&self) -> usize {
        self.items.len()
    }

    /// Insert a key/value pair.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.insert_common(key, value)
    }

    /// Add a key/value pair.
    pub fn add(&mut self, key: K, value: V) -> Option<V> {
        self.insert_common(key, value)
    }

    /// Remove a key/value pair.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        self.items.remove(key)
    }

    /// Return the item after `key`, or the first item when `key` is absent.
    pub fn next(&self, key: Option<&K>) -> Option<(&K, &V)> {
        match key {
            Some(key) => self
                .items
                .range((std::ops::Bound::Excluded(key), std::ops::Bound::Unbounded))
                .next(),
            None => self.items.iter().next(),
        }
    }

    /// Return the item before `key`, or the last item when `key` is absent.
    pub fn prev(&self, key: Option<&K>) -> Option<(&K, &V)> {
        match key {
            Some(key) => self
                .items
                .range((std::ops::Bound::Unbounded, std::ops::Bound::Excluded(key)))
                .next_back(),
            None => self.items.iter().next_back(),
        }
    }

    /// Return the last item.
    pub fn last(&self) -> Option<(&K, &V)> {
        self.items.iter().next_back()
    }

    /// Return an item by key.
    pub fn item(&self, key: &K) -> Option<&V> {
        self.items.get(key)
    }

    /// Iterate in sorted key order.
    pub fn iterate<F>(&self, mut callback: F)
    where
        F: FnMut(&K, &V),
    {
        for (key, value) in &self.items {
            callback(key, value);
        }
    }

    /// Release transient iterator/node state.
    pub fn release(&mut self) {}

    /// Free all nodes while keeping the list reusable.
    pub fn free(&mut self) {
        self.items.clear();
    }

    /// Destroy the list. The pure Rust list is consumed.
    pub fn destroy(self) {}

    /// Close the list. The pure Rust list is consumed.
    pub fn close(self) {}
}

#[allow(non_snake_case)]
pub fn H5SL__init_package<K: Ord + Clone, V>() -> bool {
    SkipList::<K, V>::init_package()
}

#[allow(non_snake_case)]
pub fn H5SL_term_package<K: Ord + Clone, V>() {
    SkipList::<K, V>::term_package()
}

#[allow(non_snake_case)]
pub fn H5SL__new_node<K: Ord + Clone, V>(key: K, value: V) -> (K, V) {
    SkipList::<K, V>::new_node(key, value)
}

#[allow(non_snake_case)]
pub fn H5SL__insert_common<K: Ord + Clone, V>(
    list: &mut SkipList<K, V>,
    key: K,
    value: V,
) -> Option<V> {
    list.insert_common(key, value)
}

#[allow(non_snake_case)]
pub fn H5SL__close_common<K: Ord + Clone, V>(list: SkipList<K, V>) {
    list.close_common()
}

#[allow(non_snake_case)]
pub fn H5SL_create<K: Ord + Clone, V>() -> SkipList<K, V> {
    SkipList::create()
}

#[allow(non_snake_case)]
pub fn H5SL_count<K: Ord + Clone, V>(list: &SkipList<K, V>) -> usize {
    list.count()
}

#[allow(non_snake_case)]
pub fn H5SL_insert<K: Ord + Clone, V>(list: &mut SkipList<K, V>, key: K, value: V) -> Option<V> {
    list.insert(key, value)
}

#[allow(non_snake_case)]
pub fn H5SL_add<K: Ord + Clone, V>(list: &mut SkipList<K, V>, key: K, value: V) -> Option<V> {
    list.add(key, value)
}

#[allow(non_snake_case)]
pub fn H5SL_remove<K: Ord + Clone, V>(list: &mut SkipList<K, V>, key: &K) -> Option<V> {
    list.remove(key)
}

#[allow(non_snake_case)]
pub fn H5SL_next<'a, K: Ord + Clone, V>(
    list: &'a SkipList<K, V>,
    key: Option<&K>,
) -> Option<(&'a K, &'a V)> {
    list.next(key)
}

#[allow(non_snake_case)]
pub fn H5SL_prev<'a, K: Ord + Clone, V>(
    list: &'a SkipList<K, V>,
    key: Option<&K>,
) -> Option<(&'a K, &'a V)> {
    list.prev(key)
}

#[allow(non_snake_case)]
pub fn H5SL_last<K: Ord + Clone, V>(list: &SkipList<K, V>) -> Option<(&K, &V)> {
    list.last()
}

#[allow(non_snake_case)]
pub fn H5SL_item<'a, K: Ord + Clone, V>(list: &'a SkipList<K, V>, key: &K) -> Option<&'a V> {
    list.item(key)
}

#[allow(non_snake_case)]
pub fn H5SL_iterate<K: Ord + Clone, V, F>(list: &SkipList<K, V>, callback: F)
where
    F: FnMut(&K, &V),
{
    list.iterate(callback)
}

#[allow(non_snake_case)]
pub fn H5SL_release<K: Ord + Clone, V>(list: &mut SkipList<K, V>) {
    list.release()
}

#[allow(non_snake_case)]
pub fn H5SL_free<K: Ord + Clone, V>(list: &mut SkipList<K, V>) {
    list.free()
}

#[allow(non_snake_case)]
pub fn H5SL_destroy<K: Ord + Clone, V>(list: SkipList<K, V>) {
    list.destroy()
}

#[allow(non_snake_case)]
pub fn H5SL_close<K: Ord + Clone, V>(list: SkipList<K, V>) {
    list.close()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skip_list_aliases_preserve_sorted_order() {
        assert!(SkipList::<i32, &str>::init_package());
        SkipList::<i32, &str>::term_package();
        assert_eq!(SkipList::new_node(3, "c"), (3, "c"));

        let mut list = SkipList::create();
        assert_eq!(list.insert(2, "b"), None);
        assert_eq!(list.add(1, "a"), None);
        assert_eq!(list.insert_common(3, "c"), None);
        assert_eq!(list.count(), 3);
        assert_eq!(list.item(&2), Some(&"b"));
        assert_eq!(list.next(Some(&1)), Some((&2, &"b")));
        assert_eq!(list.prev(Some(&3)), Some((&2, &"b")));
        assert_eq!(list.last(), Some((&3, &"c")));

        let mut seen = Vec::new();
        list.iterate(|key, value| seen.push((*key, *value)));
        assert_eq!(seen, vec![(1, "a"), (2, "b"), (3, "c")]);
        assert_eq!(list.remove(&2), Some("b"));
        list.release();
        list.free();
        assert_eq!(list.count(), 0);
        list.close_common();

        SkipList::<i32, &str>::create().destroy();
        SkipList::<i32, &str>::create().close();
    }

    #[test]
    fn h5sl_aliases_preserve_sorted_order() {
        assert!(H5SL__init_package::<i32, &str>());
        H5SL_term_package::<i32, &str>();
        assert_eq!(H5SL__new_node(3, "c"), (3, "c"));

        let mut list = H5SL_create();
        assert_eq!(H5SL_insert(&mut list, 2, "b"), None);
        assert_eq!(H5SL_add(&mut list, 1, "a"), None);
        assert_eq!(H5SL__insert_common(&mut list, 3, "c"), None);
        assert_eq!(H5SL_count(&list), 3);
        assert_eq!(H5SL_item(&list, &2), Some(&"b"));
        assert_eq!(H5SL_next(&list, Some(&1)), Some((&2, &"b")));
        assert_eq!(H5SL_prev(&list, Some(&3)), Some((&2, &"b")));
        assert_eq!(H5SL_last(&list), Some((&3, &"c")));

        let mut seen = Vec::new();
        H5SL_iterate(&list, |key, value| seen.push((*key, *value)));
        assert_eq!(seen, vec![(1, "a"), (2, "b"), (3, "c")]);
        assert_eq!(H5SL_remove(&mut list, &2), Some("b"));
        H5SL_release(&mut list);
        H5SL_free(&mut list);
        assert_eq!(H5SL_count(&list), 0);
        H5SL__close_common(list);

        H5SL_destroy(H5SL_create::<i32, &str>());
        H5SL_close(H5SL_create::<i32, &str>());
    }
}
