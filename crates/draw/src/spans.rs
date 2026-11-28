//! A map of entries that each cover a span of guest addresses, indexed by
//! where the spans start, so a write visits only the entries it can meet
//! instead of every one.

use std::collections::BTreeSet;
use std::ops::Deref;

type HashMap<K, V> =
    std::collections::HashMap<K, V, std::hash::BuildHasherDefault<rustc_hash::FxHasher>>;

/// The key of a [`SpanMap`]: a texture fetch constant.
pub(crate) type Key = [u32; 6];

/// Spans longer than this are not indexed by their start but visited by
/// every [`SpanMap::meeting`]: one long span would otherwise widen the
/// window each query looks back over.
const LONG_SPAN: u64 = 4 << 20;

/// An entry of a [`SpanMap`]: the guest bytes `[lo, hi)` it covers.
pub(crate) trait Spanned {
    fn span(&self) -> (u64, u64);
}

/// A map from [`Key`]s to [`Spanned`] values. Reads go through the map
/// itself ([`Deref`]); changes go through the methods here, which keep the
/// index in step. A value's span must not change while it is in the map.
pub(crate) struct SpanMap<V> {
    map: HashMap<Key, V>,
    /// `(lo, key)` of every entry spanning at most [`LONG_SPAN`].
    starts: BTreeSet<(u64, Key)>,
    /// The longest span in `starts` since the last clear.
    reach: u64,
    /// The entries spanning over [`LONG_SPAN`].
    long: BTreeSet<Key>,
}

impl<V> Default for SpanMap<V> {
    fn default() -> Self {
        Self {
            map: HashMap::default(),
            starts: BTreeSet::new(),
            reach: 0,
            long: BTreeSet::new(),
        }
    }
}

impl<V> Deref for SpanMap<V> {
    type Target = HashMap<Key, V>;
    fn deref(&self) -> &Self::Target {
        &self.map
    }
}

impl<V: Spanned> SpanMap<V> {
    fn index(&mut self, k: Key, (lo, hi): (u64, u64)) {
        if hi - lo > LONG_SPAN {
            self.long.insert(k);
        } else {
            self.starts.insert((lo, k));
            self.reach = self.reach.max(hi - lo);
        }
    }

    fn unindex(&mut self, k: &Key, (lo, hi): (u64, u64)) {
        if hi - lo > LONG_SPAN {
            self.long.remove(k);
        } else {
            self.starts.remove(&(lo, *k));
        }
    }

    pub(crate) fn insert(&mut self, k: Key, v: V) -> Option<V> {
        let span = v.span();
        let old = self.map.insert(k, v);
        if let Some(o) = &old {
            self.unindex(&k, o.span());
        }
        self.index(k, span);
        old
    }

    pub(crate) fn remove(&mut self, k: &Key) -> Option<V> {
        let old = self.map.remove(k)?;
        self.unindex(k, old.span());
        Some(old)
    }

    /// For changes that keep the span (the last use, say).
    pub(crate) fn get_mut(&mut self, k: &Key) -> Option<&mut V> {
        self.map.get_mut(k)
    }

    pub(crate) fn clear(&mut self) {
        self.map.clear();
        self.starts.clear();
        self.long.clear();
        self.reach = 0;
    }

    pub(crate) fn retain(&mut self, mut keep: impl FnMut(&Key, &mut V) -> bool) {
        let (starts, long) = (&mut self.starts, &mut self.long);
        self.map.retain(|k, v| {
            let kept = keep(k, v);
            if !kept {
                let (lo, hi) = v.span();
                if hi - lo > LONG_SPAN {
                    long.remove(k);
                } else {
                    starts.remove(&(lo, *k));
                }
            }
            kept
        });
    }

    /// Pushes to `out` once each the keys of the entries whose spans may
    /// meet one of `ranges`, which are sorted and do not overlap (as
    /// `merge_ranges` makes them): all that do, and some that end before
    /// the range they lie near. In index order, not map order.
    pub(crate) fn meeting(&self, ranges: &[(u64, u64)], out: &mut Vec<Key>) {
        out.extend(self.long.iter().copied());
        // Each range looks back `reach` for spans starting before it; the
        // windows of consecutive ranges are cut so none is visited twice.
        let mut from = 0;
        for &(lo, hi) in ranges {
            let a = lo.saturating_sub(self.reach).max(from);
            if a < hi {
                let found = self.starts.range((a, [0; 6])..(hi, [0; 6]));
                out.extend(found.map(|&(_, k)| k));
            }
            from = from.max(hi);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct S(u64, u64);
    impl Spanned for S {
        fn span(&self) -> (u64, u64) {
            (self.0, self.1)
        }
    }

    fn key(n: u32) -> Key {
        [n, 0, 0, 0, 0, 0]
    }

    fn meets(ranges: &[(u64, u64)], (a, b): (u64, u64)) -> bool {
        ranges.iter().any(|&(lo, hi)| lo < b && a < hi)
    }

    /// Every entry meeting the ranges is found, each once.
    fn check(m: &SpanMap<S>, ranges: &[(u64, u64)]) {
        let mut out = Vec::new();
        m.meeting(ranges, &mut out);
        let mut sorted = out.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), out.len(), "a key found twice for {ranges:?}");
        for (k, v) in m.iter() {
            if meets(ranges, v.span()) {
                assert!(out.contains(k), "{k:?} missed for {ranges:?}");
            }
        }
        for k in &out {
            assert!(m.contains_key(k), "{k:?} found but not in the map");
        }
    }

    #[test]
    fn finds_every_entry_a_write_meets() {
        let mut m = SpanMap::default();
        // Short spans, one shared start, and one long span.
        m.insert(key(1), S(0x1000, 0x2000));
        m.insert(key(2), S(0x1000, 0x1800));
        m.insert(key(3), S(0x8000, 0x10000));
        m.insert(key(4), S(0x20_0000, 0x20_0000 + LONG_SPAN + 1));
        m.insert(key(5), S(0x9_0000, 0x9_0100));
        for ranges in [
            vec![(0x1fff, 0x2000)],
            vec![(0, 0x1000)],
            vec![(0x1800, 0x8001)],
            vec![(0x300_0000, 0x300_0001)],
            vec![(0x1000, 0x1001), (0x1100, 0x1200), (0xffff, 0x9_0001)],
            vec![(0, u64::MAX)],
        ] {
            check(&m, &ranges);
        }
    }

    #[test]
    fn the_index_follows_changes() {
        let mut m = SpanMap::default();
        m.insert(key(1), S(0x1000, 0x2000));
        // Replaced with another span: the old one no longer finds it.
        m.insert(key(1), S(0x5000, 0x6000));
        let mut out = Vec::new();
        m.meeting(&[(0x1000, 0x2000)], &mut out);
        assert!(out.is_empty());
        m.insert(key(2), S(0x7000, 0x7000 + LONG_SPAN + 1));
        m.retain(|k, _| *k != key(2));
        assert!(m.long.is_empty());
        m.remove(&key(1));
        assert!(m.starts.is_empty());
        m.insert(key(3), S(0x10, 0x20));
        check(&m, &[(0x15, 0x16)]);
        m.clear();
        assert!(m.is_empty() && m.starts.is_empty() && m.reach == 0);
    }
}
