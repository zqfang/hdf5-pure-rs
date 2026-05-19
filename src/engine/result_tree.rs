/// Inclusive integer range result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResultRange {
    pub start: u64,
    pub end: u64,
}

/// Ordered range result tree analogue.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResultTree {
    ranges: Vec<ResultRange>,
}

impl ResultRange {
    pub fn intersects(self, other: Self) -> bool {
        self.start <= other.end && other.start <= self.end
    }
}

impl ResultTree {
    /// Initialize leaf storage.
    pub fn leaf_init(range: ResultRange) -> ResultRange {
        range
    }

    /// Add a range to a result set.
    pub fn result_set_add(&mut self, range: ResultRange) {
        let key = (range.start, range.end);
        let index = self
            .ranges
            .partition_point(|existing| (existing.start, existing.end) <= key);
        self.ranges.insert(index, range);
    }

    /// Bulk-load ranges from any iterator.
    pub fn bulk_load_iter(ranges: impl IntoIterator<Item = ResultRange>) -> Self {
        let mut ranges: Vec<_> = ranges.into_iter().collect();
        ranges.sort_by_key(|range| (range.start, range.end));
        Self { ranges }
    }

    /// Create an empty result tree.
    pub fn create() -> Self {
        Self::default()
    }

    /// Iterate over all ranges in sorted order.
    pub fn iter(&self) -> impl Iterator<Item = ResultRange> + '_ {
        self.ranges.iter().copied()
    }

    /// Iterate over ranges intersecting `query`.
    pub fn search_iter(&self, query: ResultRange) -> impl Iterator<Item = ResultRange> + '_ {
        self.iter().filter(move |range| range.intersects(query))
    }

    /// Append ranges intersecting `query` to caller-provided storage.
    pub fn search_into(&self, query: ResultRange, out: &mut Vec<ResultRange>) -> usize {
        let start_len = out.len();
        out.extend(self.search_iter(query));
        out.len() - start_len
    }

    /// Visit ranges intersecting `query`.
    pub fn search_visit<F>(&self, query: ResultRange, mut visitor: F)
    where
        F: FnMut(ResultRange),
    {
        for range in self.search_iter(query) {
            visitor(range);
        }
    }

    /// Copy a node/range.
    pub fn node_copy(range: ResultRange) -> ResultRange {
        range
    }

    /// Recursive free hook.
    pub fn free_recurse(&mut self) {
        self.ranges.clear();
    }

    /// Free this tree.
    pub fn free(mut self) {
        self.free_recurse();
    }

    /// Copy this tree.
    pub fn copy(&self) -> Self {
        self.clone()
    }

    /// Return whether two leaves intersect.
    pub fn leaves_intersect(lhs: ResultRange, rhs: ResultRange) -> bool {
        lhs.intersects(rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::{ResultRange, ResultTree};

    #[test]
    fn result_tree_aliases_find_intersections() {
        let a = ResultTree::leaf_init(ResultRange { start: 0, end: 3 });
        let b = ResultRange { start: 10, end: 12 };
        assert!(ResultTree::leaves_intersect(
            a,
            ResultRange { start: 2, end: 5 }
        ));
        assert!(!ResultTree::leaves_intersect(a, b));

        let mut tree = ResultTree::create();
        tree.result_set_add(a);
        tree.result_set_add(b);
        assert_eq!(tree.iter().collect::<Vec<_>>(), vec![a, b]);
        assert_eq!(
            tree.search_iter(ResultRange { start: 1, end: 1 })
                .collect::<Vec<_>>(),
            vec![a]
        );

        let mut found = Vec::new();
        assert_eq!(
            tree.search_into(ResultRange { start: 11, end: 11 }, &mut found),
            1
        );
        assert_eq!(found, vec![b]);

        found.clear();
        tree.search_visit(ResultRange { start: 0, end: 20 }, |range| found.push(range));
        assert_eq!(found, vec![a, b]);

        let copy = tree.copy();
        assert_eq!(copy, tree);
        ResultTree::bulk_load_iter([a, b]).free();
        assert_eq!(ResultTree::node_copy(a), a);
    }
}
