//! A small, fast, non-cryptographic hasher (the algorithm rustc itself uses).
//!
//! The search hashes a `State` on every transposition-table probe, and a state
//! is a whole bitplane of box positions.  The default SipHash is built to
//! resist hash-flooding from untrusted input, a property worth nothing here and
//! paid for on every node.

use std::hash::{BuildHasherDefault, Hasher};

const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

#[derive(Default, Clone, Copy)]
pub struct FxHasher {
    hash: u64,
}

impl FxHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(SEED);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for chunk in &mut chunks {
            self.add(u64::from_le_bytes(chunk.try_into().unwrap()));
        }
        let rest = chunks.remainder();
        if !rest.is_empty() {
            let mut buf = [0u8; 8];
            buf[..rest.len()].copy_from_slice(rest);
            self.add(u64::from_le_bytes(buf));
        }
    }
    #[inline] fn write_u8(&mut self, i: u8)       { self.add(i as u64); }
    #[inline] fn write_u16(&mut self, i: u16)     { self.add(i as u64); }
    #[inline] fn write_u32(&mut self, i: u32)     { self.add(i as u64); }
    #[inline] fn write_u64(&mut self, i: u64)     { self.add(i); }
    #[inline] fn write_usize(&mut self, i: usize) { self.add(i as u64); }
    #[inline] fn finish(&self) -> u64 { self.hash }
}

pub type FxBuildHasher = BuildHasherDefault<FxHasher>;
pub type FxHashMap<K, V> = std::collections::HashMap<K, V, FxBuildHasher>;
pub type FxHashSet<K> = std::collections::HashSet<K, FxBuildHasher>;
