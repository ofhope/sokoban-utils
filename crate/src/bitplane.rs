/// A packed bitplane for a W×H grid.
/// Bits are stored MSB-first: bit 0 of byte 0 is cell (0,0).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BitPlane {
    pub width: u8,
    pub height: u8,
    data: Vec<u8>,
}

impl BitPlane {
    pub fn new(width: u8, height: u8) -> Self {
        let cells = width as usize * height as usize;
        Self {
            width,
            height,
            data: vec![0u8; cells.div_ceil(8)],
        }
    }

    #[inline]
    fn idx(width: u8, x: u8, y: u8) -> usize {
        y as usize * width as usize + x as usize
    }

    #[inline]
    pub fn set(&mut self, x: u8, y: u8) {
        let i = Self::idx(self.width, x, y);
        self.data[i / 8] |= 1 << (7 - i % 8);
    }

    #[inline]
    pub fn clear(&mut self, x: u8, y: u8) {
        let i = Self::idx(self.width, x, y);
        self.data[i / 8] &= !(1 << (7 - i % 8));
    }

    #[inline]
    pub fn get(&self, x: u8, y: u8) -> bool {
        let i = Self::idx(self.width, x, y);
        self.get_flat(i)
    }

    #[inline]
    pub fn get_flat(&self, i: usize) -> bool {
        self.data[i / 8] & (1 << (7 - i % 8)) != 0
    }

    pub fn len(&self) -> usize {
        self.width as usize * self.height as usize
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    pub fn from_bytes(width: u8, height: u8, bytes: &[u8]) -> Option<Self> {
        let cells = width as usize * height as usize;
        let expected = cells.div_ceil(8);
        if bytes.len() < expected {
            return None;
        }
        Some(Self {
            width,
            height,
            data: bytes[..expected].to_vec(),
        })
    }

    /// Iterate over (x, y) of every set bit.
    pub fn iter_set_bits(&self) -> impl Iterator<Item = (u8, u8)> + '_ {
        let w = self.width as usize;
        let total = self.len();
        (0..total)
            .filter(move |&i| self.get_flat(i))
            .map(move |i| ((i % w) as u8, (i / w) as u8))
    }

    /// RLE encode: alternating runs of 0-bits then 1-bits.
    /// Each run length uses 4-bit nibbles; 0xF means "add next nibble".
    pub fn rle_encode(&self) -> Vec<u8> {
        let total = self.len();
        let mut out = Vec::new();
        let mut current = false; // first run is 0s
        let mut run: u32 = 0;

        for i in 0..total {
            if self.get_flat(i) == current {
                run += 1;
            } else {
                encode_run(&mut out, run);
                current = !current;
                run = 1;
            }
        }
        encode_run(&mut out, run);
        out
    }

    pub fn rle_decode(width: u8, height: u8, data: &[u8]) -> Option<Self> {
        let total = width as usize * height as usize;
        let mut plane = Self::new(width, height);
        let mut current = false;
        let mut flat_idx = 0usize;
        let mut nibble_idx = 0usize;

        loop {
            if flat_idx >= total {
                break;
            }
            let (run, consumed) = decode_run(data, nibble_idx)?;
            nibble_idx = consumed;
            for _ in 0..run {
                if flat_idx >= total {
                    break;
                }
                if current {
                    let x = (flat_idx % width as usize) as u8;
                    let y = (flat_idx / width as usize) as u8;
                    plane.set(x, y);
                }
                flat_idx += 1;
            }
            current = !current;
        }
        Some(plane)
    }
}

fn encode_run(out: &mut Vec<u8>, mut run: u32) {
    // Encode as sequence of 4-bit nibbles packed into bytes.
    // We buffer nibbles and flush pairs.
    // A nibble value of 15 means "another nibble follows (add 15 and continue)".
    // This is simpler: just emit nibbles and pack them.
    let mut nibbles: Vec<u8> = Vec::new();
    loop {
        if run < 15 {
            nibbles.push(run as u8);
            break;
        } else {
            nibbles.push(15);
            run -= 15;
        }
    }
    for chunk in nibbles.chunks(2) {
        if chunk.len() == 2 {
            out.push((chunk[0] << 4) | chunk[1]);
        } else {
            out.push(chunk[0] << 4);
        }
    }
}

fn decode_run(data: &[u8], mut nibble_idx: usize) -> Option<(u32, usize)> {
    let mut run: u32 = 0;
    loop {
        let byte_idx = nibble_idx / 2;
        if byte_idx >= data.len() {
            return Some((run, nibble_idx));
        }
        let nibble = if nibble_idx % 2 == 0 {
            (data[byte_idx] >> 4) & 0xF
        } else {
            data[byte_idx] & 0xF
        };
        nibble_idx += 1;
        run += nibble as u32;
        if nibble < 15 {
            break;
        }
    }
    Some((run, nibble_idx))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_set_get() {
        let mut bp = BitPlane::new(8, 4);
        bp.set(0, 0);
        bp.set(3, 1);
        bp.set(7, 3);
        assert!(bp.get(0, 0));
        assert!(bp.get(3, 1));
        assert!(bp.get(7, 3));
        assert!(!bp.get(1, 0));
    }

    #[test]
    fn rle_round_trip() {
        let mut bp = BitPlane::new(7, 7);
        bp.set(0, 0); bp.set(2, 3); bp.set(6, 6);
        let enc = bp.rle_encode();
        let dec = BitPlane::rle_decode(7, 7, &enc).unwrap();
        assert_eq!(bp, dec);
    }
}
