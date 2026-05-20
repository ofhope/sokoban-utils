use wasm_bindgen::prelude::*;
use crate::bitplane::BitPlane;

/// XSB character set:
/// `#` wall  `$` box  `.` goal  `*` box-on-goal
/// `@` player  `+` player-on-goal  ` ` floor
#[wasm_bindgen]
pub struct SokobanLevel {
    pub(crate) width: u8,
    pub(crate) height: u8,
    pub(crate) walls: BitPlane,
    pub(crate) boxes: BitPlane,
    pub(crate) goals: BitPlane,
    pub(crate) player_pos: u16, // flat index
}

#[wasm_bindgen]
impl SokobanLevel {
    // ── Constructors ──────────────────────────────────────────────────────

    /// Parse a level from XSB string format.
    pub fn from_xsb(xsb: &str) -> Result<SokobanLevel, JsValue> {
        let lines: Vec<&str> = xsb.lines().collect();
        let height = lines.len() as u8;
        let width = lines
            .iter()
            .map(|l| l.chars().count())
            .max()
            .ok_or_else(|| JsValue::from_str("empty level"))? as u8;

        let mut walls = BitPlane::new(width, height);
        let mut boxes = BitPlane::new(width, height);
        let mut goals = BitPlane::new(width, height);
        let mut player_pos = 0u16;
        let mut found_player = false;

        for (y, line) in lines.iter().enumerate() {
            for (x, ch) in line.chars().enumerate() {
                let (x, y) = (x as u8, y as u8);
                match ch {
                    '#' => walls.set(x, y),
                    '$' => boxes.set(x, y),
                    '.' => goals.set(x, y),
                    '*' => {
                        boxes.set(x, y);
                        goals.set(x, y);
                    }
                    '@' => {
                        player_pos = y as u16 * width as u16 + x as u16;
                        found_player = true;
                    }
                    '+' => {
                        goals.set(x, y);
                        player_pos = y as u16 * width as u16 + x as u16;
                        found_player = true;
                    }
                    _ => {}
                }
            }
        }

        if !found_player {
            return Err(JsValue::from_str("no player found in level"));
        }

        Ok(SokobanLevel { width, height, walls, boxes, goals, player_pos })
    }

    /// Reconstruct from the compact binary wire format produced by `to_bytes`.
    pub fn from_bytes(data: &[u8]) -> Result<SokobanLevel, JsValue> {
        if data.len() < 5 {
            return Err(JsValue::from_str("data too short"));
        }

        // Verify CRC-8 (last byte)
        let (body, crc_slice) = data.split_at(data.len() - 1);
        if crc8(body) != crc_slice[0] {
            return Err(JsValue::from_str("CRC mismatch — data may be corrupted"));
        }

        let width = body[0];
        let height = body[1];
        let player_pos = ((body[2] as u16) << 8) | body[3] as u16;
        let plane_bytes = (width as usize * height as usize).div_ceil(8);

        let needed = 4 + plane_bytes * 3;
        if body.len() < needed {
            return Err(JsValue::from_str("data truncated"));
        }

        let walls = BitPlane::from_bytes(width, height, &body[4..4 + plane_bytes])
            .ok_or_else(|| JsValue::from_str("bad wall plane"))?;
        let boxes = BitPlane::from_bytes(width, height, &body[4 + plane_bytes..4 + plane_bytes * 2])
            .ok_or_else(|| JsValue::from_str("bad box plane"))?;
        let goals = BitPlane::from_bytes(width, height, &body[4 + plane_bytes * 2..needed])
            .ok_or_else(|| JsValue::from_str("bad goal plane"))?;

        Ok(SokobanLevel { width, height, walls, boxes, goals, player_pos })
    }

    // ── Serialisation ─────────────────────────────────────────────────────

    /// Serialise to XSB string.
    pub fn to_xsb(&self) -> String {
        let w = self.width as usize;
        let h = self.height as usize;
        let mut rows = Vec::with_capacity(h);

        for y in 0..h {
            let mut row = String::with_capacity(w);
            for x in 0..w {
                let (x8, y8) = (x as u8, y as u8);
                let wall   = self.walls.get(x8, y8);
                let box_   = self.boxes.get(x8, y8);
                let goal   = self.goals.get(x8, y8);
                let player = self.player_pos as usize == y * w + x;

                row.push(match (wall, box_, goal, player) {
                    (true, ..)                    => '#',
                    (_, true,  true,  _)          => '*',
                    (_, true,  false, _)          => '$',
                    (_, false, true,  true)       => '+',
                    (_, false, true,  false)      => '.',
                    (_, false, false, true)       => '@',
                    _                             => ' ',
                });
            }
            rows.push(row);
        }
        rows.join("\n")
    }

    /// Serialise to compact binary wire format (with CRC-8 trailer).
    ///
    /// Layout: `[width, height, player_hi, player_lo, wall_plane..., box_plane..., goal_plane..., crc8]`
    pub fn to_bytes(&self) -> Vec<u8> {
        let plane_bytes = (self.width as usize * self.height as usize).div_ceil(8);
        let mut out = Vec::with_capacity(4 + plane_bytes * 3 + 1);

        out.push(self.width);
        out.push(self.height);
        out.push((self.player_pos >> 8) as u8);
        out.push(self.player_pos as u8);
        out.extend_from_slice(self.walls.as_bytes());
        out.extend_from_slice(self.boxes.as_bytes());
        out.extend_from_slice(self.goals.as_bytes());

        let crc = crc8(&out);
        out.push(crc);
        out
    }

    // ── Accessors ─────────────────────────────────────────────────────────

    pub fn width(&self)  -> u8  { self.width }
    pub fn height(&self) -> u8  { self.height }
    pub fn player_pos(&self) -> u16 { self.player_pos }

    pub fn is_wall(&self,  x: u8, y: u8) -> bool { self.walls.get(x, y) }
    pub fn is_box(&self,   x: u8, y: u8) -> bool { self.boxes.get(x, y) }
    pub fn is_goal(&self,  x: u8, y: u8) -> bool { self.goals.get(x, y) }

    pub fn is_solved(&self) -> bool {
        // Every goal cell has a box on it
        self.goals
            .iter_set_bits()
            .all(|(x, y)| self.boxes.get(x, y))
    }
}

/// CRC-8/SMBUS — tiny and sufficient for URL corruption detection.
pub(crate) fn crc8(data: &[u8]) -> u8 {
    data.iter().fold(0u8, |acc, &b| {
        (0..8).fold(acc ^ b, |crc, _| {
            if crc & 0x80 != 0 { (crc << 1) ^ 0x07 } else { crc << 1 }
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIMPLE: &str = "  #####\n\
                          ###   #\n\
                          # $@. #\n\
                          ###   #\n\
                          ##  ###";

    #[test]
    fn xsb_round_trip() {
        let level = SokobanLevel::from_xsb(SIMPLE).expect("parse failed");
        assert_eq!(level.width, 7);
        assert_eq!(level.height, 5);
        assert!(level.is_wall(2, 0));
        assert!(level.is_box(3, 2));
        assert!(level.is_goal(5, 2));
        let back = level.to_xsb();
        let re = SokobanLevel::from_xsb(&back).expect("re-parse failed");
        assert_eq!(re.player_pos, level.player_pos);
    }

    #[test]
    fn bytes_round_trip() {
        let level = SokobanLevel::from_xsb(SIMPLE).expect("parse");
        let bytes = level.to_bytes();
        let restored = SokobanLevel::from_bytes(&bytes).expect("restore");
        assert_eq!(restored.width, level.width);
        assert_eq!(restored.height, level.height);
        assert_eq!(restored.player_pos, level.player_pos);
    }

    #[test]
    fn crc_detects_corruption() {
        let level = SokobanLevel::from_xsb(SIMPLE).expect("parse");
        let mut bytes = level.to_bytes();
        bytes[4] ^= 0xFF; // corrupt wall plane
        assert!(SokobanLevel::from_bytes(&bytes).is_err());
    }
}
