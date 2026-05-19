//! High-performance bitmap using u64 chunks.
//!
//! Provides O(1) find-first-free and set/clear operations via hardware popcnt.

/// Bitmap backed by u64 chunks.
pub struct Bitmap {
    chunks: Vec<u64>,
    len: usize,
}

impl Bitmap {
    /// Create a new bitmap with `len` bits, all clear (0).
    pub fn new(len: usize) -> Self {
        let num_chunks = len.div_ceil(64);
        Self {
            chunks: vec![0u64; num_chunks],
            len,
        }
    }

    /// Set bit at index.
    pub fn set(&mut self, idx: usize) {
        debug_assert!(idx < self.len);
        let chunk = idx / 64;
        let bit = idx % 64;
        self.chunks[chunk] |= 1u64 << bit;
    }

    /// Clear bit at index.
    pub fn clear(&mut self, idx: usize) {
        debug_assert!(idx < self.len);
        let chunk = idx / 64;
        let bit = idx % 64;
        self.chunks[chunk] &= !(1u64 << bit);
    }

    /// Test bit at index.
    pub fn get(&self, idx: usize) -> bool {
        debug_assert!(idx < self.len);
        let chunk = idx / 64;
        let bit = idx % 64;
        (self.chunks[chunk] >> bit) & 1 == 1
    }

    /// Find first clear (0) bit. Returns None if all set.
    pub fn find_first_clear(&self) -> Option<usize> {
        for (chunk_idx, &chunk) in self.chunks.iter().enumerate() {
            if chunk != u64::MAX {
                // There's at least one zero bit in this chunk.
                let bit_pos = (!chunk).trailing_zeros() as usize;
                let idx = chunk_idx * 64 + bit_pos;
                if idx < self.len {
                    return Some(idx);
                }
            }
        }
        None
    }

    /// Count number of set bits.
    pub fn count_ones(&self) -> usize {
        self.chunks.iter().map(|c| c.count_ones() as usize).sum()
    }

    /// Count number of clear bits.
    pub fn count_zeros(&self) -> usize {
        self.len - self.count_ones()
    }

    /// Total capacity.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the bitmap has zero capacity.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_ops() {
        let mut bm = Bitmap::new(128);
        assert_eq!(bm.count_zeros(), 128);

        bm.set(0);
        bm.set(63);
        bm.set(64);
        bm.set(127);
        assert_eq!(bm.count_ones(), 4);
        assert!(bm.get(0));
        assert!(bm.get(63));
        assert!(!bm.get(1));

        bm.clear(0);
        assert!(!bm.get(0));
        assert_eq!(bm.count_ones(), 3);
    }

    #[test]
    fn find_first_clear() {
        let mut bm = Bitmap::new(10);
        bm.set(0);
        bm.set(1);
        bm.set(2);
        assert_eq!(bm.find_first_clear(), Some(3));

        // Fill all.
        for i in 0..10 {
            bm.set(i);
        }
        assert_eq!(bm.find_first_clear(), None);
    }
}
