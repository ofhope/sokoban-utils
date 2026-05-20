use wasm_bindgen::prelude::*;
use crate::level::SokobanLevel;

// Base64url alphabet (RFC 4648 §5, no padding)
const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

fn to_base64url(bytes: &[u8]) -> String {
    let mut out = String::with_capacity((bytes.len() * 4).div_ceil(3));
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHABET[(n >> 18 & 0x3F) as usize] as char);
        out.push(ALPHABET[(n >> 12 & 0x3F) as usize] as char);
        if chunk.len() > 1 { out.push(ALPHABET[(n >> 6 & 0x3F) as usize] as char); }
        if chunk.len() > 2 { out.push(ALPHABET[(n       & 0x3F) as usize] as char); }
    }
    out
}

fn from_base64url_clean(s: &str) -> Option<Vec<u8>> {
    let mut table = [0xFFu8; 128];
    for (i, &c) in ALPHABET.iter().enumerate() {
        if (c as usize) < 128 {
            table[c as usize] = i as u8;
        }
    }

    let chars: Vec<u8> = s.bytes().collect();
    let mut out = Vec::with_capacity(chars.len() * 3 / 4 + 1);
    let mut buf = 0u32;
    let mut bits = 0u8;

    for &c in &chars {
        if c as usize >= 128 { return None; }
        let v = table[c as usize];
        if v == 0xFF { return None; }
        buf = (buf << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push(((buf >> bits) & 0xFF) as u8);
        }
    }
    Some(out)
}

// ── Public WASM API ────────────────────────────────────────────────────────

/// Encode a level to a URL fragment string (e.g. `v1-AQcH...`).
///
/// Optionally attach a title and author for display in the editor.
#[wasm_bindgen]
pub fn level_to_fragment(
    level: &SokobanLevel,
    title: Option<String>,
    author: Option<String>,
) -> String {
    let binary = level.to_bytes(); // already includes CRC-8

    // Versioned envelope: we embed the binary (with its CRC) directly.
    // Future versions can change the wrapper without breaking v1 decoders.
    let mut payload = Vec::with_capacity(binary.len() + 64);

    // v1 flags byte
    let flags: u8 = if title.is_some() { 0b01 } else { 0 }
                  | if author.is_some() { 0b10 } else { 0 };
    payload.push(1u8);   // version
    payload.push(flags);
    payload.extend_from_slice(&binary);

    if let Some(t) = title {
        payload.extend_from_slice(t.as_bytes());
        payload.push(0x00);
    }
    if let Some(a) = author {
        payload.extend_from_slice(a.as_bytes());
        payload.push(0x00);
    }

    format!("v1-{}", to_base64url(&payload))
}

/// Decode a URL fragment back to a level.
///
/// Accepts the fragment with or without the leading `#`.
/// Returns the level; title/author metadata (if present) is currently discarded
/// — use `fragment_metadata` to retrieve them separately.
#[wasm_bindgen]
pub fn fragment_to_level(fragment: &str) -> Result<SokobanLevel, JsValue> {
    let s = fragment.trim_start_matches('#');

    if let Some(encoded) = s.strip_prefix("v1-") {
        let bytes = from_base64url_clean(encoded)
            .ok_or_else(|| JsValue::from_str("invalid base64url"))?;

        if bytes.len() < 2 {
            return Err(JsValue::from_str("payload too short"));
        }

        // bytes[0] = version (1), bytes[1] = flags, bytes[2..] = to_bytes() output + optional metadata
        let _version = bytes[0];
        let _flags   = bytes[1];
        let rest     = &bytes[2..];

        // The level binary ends after the known fixed + plane bytes.
        // We find the end by reading width/height from the binary header.
        if rest.len() < 4 {
            return Err(JsValue::from_str("payload missing level header"));
        }
        let width  = rest[0] as usize;
        let height = rest[1] as usize;
        let plane_bytes = (width * height).div_ceil(8);
        let level_end = 4 + plane_bytes * 3 + 1; // header + 3 planes + CRC

        if rest.len() < level_end {
            return Err(JsValue::from_str("payload truncated"));
        }

        SokobanLevel::from_bytes(&rest[..level_end])
    } else {
        Err(JsValue::from_str("unsupported fragment version"))
    }
}

/// Extract optional metadata from a fragment without fully decoding the level.
/// Returns a JS object `{ title?: string, author?: string }`.
#[wasm_bindgen]
pub fn fragment_metadata(fragment: &str) -> Result<js_sys::Object, JsValue> {
    let obj = js_sys::Object::new();

    let s = fragment.trim_start_matches('#');
    let encoded = s.strip_prefix("v1-")
        .ok_or_else(|| JsValue::from_str("unsupported version"))?;

    let bytes = from_base64url_clean(encoded)
        .ok_or_else(|| JsValue::from_str("invalid base64url"))?;

    if bytes.len() < 2 { return Ok(obj); }

    let flags  = bytes[1];
    let rest   = &bytes[2..];
    if rest.len() < 4 { return Ok(obj); }

    let width  = rest[0] as usize;
    let height = rest[1] as usize;
    let plane_bytes = (width * height).div_ceil(8);
    let level_end = 4 + plane_bytes * 3 + 1;

    if rest.len() <= level_end { return Ok(obj); }

    let mut cursor = &rest[level_end..];

    if flags & 0b01 != 0 {
        if let Some(nul) = cursor.iter().position(|&b| b == 0) {
            let title = String::from_utf8_lossy(&cursor[..nul]).to_string();
            js_sys::Reflect::set(&obj, &"title".into(), &JsValue::from_str(&title))?;
            cursor = &cursor[nul + 1..];
        }
    }

    if flags & 0b10 != 0 {
        if let Some(nul) = cursor.iter().position(|&b| b == 0) {
            let author = String::from_utf8_lossy(&cursor[..nul]).to_string();
            js_sys::Reflect::set(&obj, &"author".into(), &JsValue::from_str(&author))?;
        }
    }

    Ok(obj)
}
