"""
Normalize a folder of XSB Sokoban puzzle files into a single
canonical XSB file with consistent metadata per level.

Normalized output format (one blank line between levels):

    <grid lines>
    Title: <title or "Level N">
    Author: <author or collection author or "Unknown">
    Collection: <collection name>

"""

import re
import os
from pathlib import Path

# Valid sokoban grid characters
GRID_CHARS = re.compile(r'^[ \t_\-|#@+$*.]+$')
MUST_HAVE = re.compile(r'[#@+$*.]')  # a real grid line must have at least one of these

def is_grid_line(line):
    s = line.rstrip('\r\n')
    if not s:
        return False
    if not GRID_CHARS.match(s):
        return False
    if not MUST_HAVE.search(s):
        return False
    return True

def strip_comment(line):
    """Strip leading ; or // comment markers."""
    s = line.strip()
    s = re.sub(r'^;+\s*', '', s)
    s = re.sub(r'^//+\s*', '', s)
    return s.strip()

def parse_metadata_line(line):
    """Return (key, value) if line matches 'Key: value' or 'Key:value', else (None,None)."""
    m = re.match(r'^([A-Za-z][\w\s\-]*?)\s*:\s*(.*)$', line.strip())
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None, None

# Known collection-level metadata keys (ignore per-level)
COLLECTION_KEYS = {'set', 'copyright', 'email', 'homepage', 'e-mail', 'website',
                   'date', 'difficulty', 'description'}


def read_metadata_block(lines, i):
    """
    Read metadata lines (Title:, Author:, Comment:..Comment-End:, etc.)
    after a grid block, starting at index i.
    Returns (title, author, new_i) where new_i points past the blank separator.
    Stops at a blank line, a new grid line, a comment marker, or a bare number.
    """
    title = None
    author = None
    in_comment = False

    while i < len(lines):
        ml = lines[i].strip()
        raw = lines[i].rstrip()

        # Blank line → level separator, consume it and stop
        if not ml:
            i += 1
            break

        # Comment-End: resets comment mode and continues
        if re.match(r'^Comment-End\s*:?', ml, re.IGNORECASE):
            in_comment = False
            i += 1
            continue

        # Inside a comment block – skip content lines
        if in_comment:
            i += 1
            continue

        # Start of comment block
        if re.match(r'^Comment\s*:', ml, re.IGNORECASE):
            in_comment = True
            i += 1
            continue

        # Stop conditions: new grid line, or start of next level's comment/number
        if is_grid_line(raw):
            break
        if ml.startswith(';') or ml.startswith('//'):
            break
        if re.match(r'^\d+$', ml):
            break

        # Try to parse as metadata
        key, val = parse_metadata_line(ml)
        if key:
            kl = key.lower()
            if kl == 'title':
                title = val
            elif kl == 'author':
                author = val
            # All other keys (color, solution, moves, collection-level info) are ignored
        # Non-metadata non-grid line → skip silently
        i += 1

    return title, author, i


def parse_xsb_file(filepath):
    """
    Parse an XSB file.
    Returns:
      coll_info: dict {name, author}
      levels: list of {grid, title, author, index}
    """
    raw = Path(filepath).read_text(encoding='utf-8', errors='replace')
    lines = raw.splitlines()

    # Derive collection name from filename, strip trailing _NNN (level count)
    fname = Path(filepath).stem
    coll_name = re.sub(r'_\d+$', '', fname).replace('_', ' ')

    coll_info = {'name': coll_name, 'author': None}
    levels = []

    # ---- Collect collection-level metadata from the file header ----
    for line in lines:
        s = line.strip()
        if is_grid_line(line):
            break  # first grid line ends the header
        key, val = parse_metadata_line(line)
        if key:
            kl = key.lower()
            if kl in ('set', 'collection', 'name') and val:
                coll_info['name'] = val
            elif kl == 'author' and val and not coll_info['author']:
                coll_info['author'] = val
        # Bare "author:value" with no space (SokEvo style)
        m = re.match(r'^author\s*:\s*(.+)$', s, re.IGNORECASE)
        if m and not coll_info['author']:
            coll_info['author'] = m.group(1).strip()

    # ---- Main parsing loop ----
    pending_label = None   # label/comment seen just before a grid
    level_index = 0

    i = 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        raw = line.rstrip()

        if is_grid_line(raw):
            # Collect all consecutive grid lines
            grid = [raw]
            i += 1
            while i < len(lines) and is_grid_line(lines[i].rstrip()):
                grid.append(lines[i].rstrip())
                i += 1

            # Read trailing metadata
            title, author, i = read_metadata_block(lines, i)

            # Resolve title: trailing metadata > pending comment > fallback
            if not title:
                title = pending_label

            if not author:
                author = coll_info['author'] or 'Unknown'

            # Only save if the grid has at least one wall
            if any('#' in row for row in grid):
                level_index += 1
                if not title:
                    title = f"Level {level_index}"
                levels.append({
                    'grid': grid,
                    'title': title,
                    'author': author,
                    'index': level_index,
                })

            pending_label = None

        elif s.startswith(';') or s.startswith('//'):
            label = strip_comment(line)
            # Classify the comment
            if re.match(r'^\d+$', label):
                # Pure number → ignore (level counter)
                pending_label = None
            elif re.match(r'^screen\.\d+$', label, re.IGNORECASE):
                # XSokoban ";screen.01"
                num = label.split('.')[1].lstrip('0') or '0'
                pending_label = f"Screen {num}"
            elif re.match(r'^GrigrSpecial\d+$', label, re.IGNORECASE):
                # GrigrSpecial uses this as an identifier; Title: follows in metadata
                pending_label = None
            elif re.match(r'^YM_M(\d+)\.XSB$', label, re.IGNORECASE):
                m = re.match(r'^YM_M(\d+)\.XSB$', label, re.IGNORECASE)
                pending_label = f"Handmade {int(m.group(1))}" if m else label
            elif label:
                # Skip labels that look like copyright notices, URLs, or long prose
                is_noise = bool(re.match(r'^[=\-]+$', label) or re.search(r'copyright|©|\bhttp\b|@.*\.|e-mail', label, re.IGNORECASE) or len(label) > 80)
                if is_noise:
                    i += 1
                    continue
                pending_label = label
            i += 1

        elif re.match(r'^\d+$', s):
            # Bare number (Sven / Yoshio-Auto format) – level counter, not a title
            pending_label = None
            i += 1

        else:
            # Collection-level metadata or prose – check for author/name
            key, val = parse_metadata_line(line)
            if key and val:
                kl = key.lower()
                if kl in ('set', 'collection', 'name') and not coll_info['name']:
                    coll_info['name'] = val
                elif kl == 'author' and not coll_info['author']:
                    coll_info['author'] = val
            i += 1

    return coll_info, levels


def normalize_grid(grid_lines):
    """Strip trailing whitespace from each row (leading spaces must be preserved)."""
    return [row.rstrip() for row in grid_lines]


def write_normalized(output_path, all_collections):
    total = sum(len(lvls) for _, lvls in all_collections)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('; normalised-levels.xsb\n')
        f.write(f'; Total levels: {total}\n')
        f.write('; Generated by normalize_xsb.py\n')
        f.write(';\n')
        f.write('; Format: grid rows, then Title:/Author:/Collection: metadata.\n')
        f.write('; Levels are separated by a single blank line.\n')
        f.write('\n')

        for coll_info, levels in all_collections:
            cname = coll_info['name']
            cauthor = coll_info['author'] or 'Unknown'
            f.write('; ' + '=' * 58 + '\n')
            f.write(f'; Collection: {cname}\n')
            f.write(f'; Author:     {cauthor}\n')
            f.write(f'; Levels:     {len(levels)}\n')
            f.write('; ' + '=' * 58 + '\n')
            f.write('\n')

            for lvl in levels:
                grid = normalize_grid(lvl['grid'])
                for row in grid:
                    f.write(row + '\n')
                f.write(f'Title: {lvl["title"]}\n')
                f.write(f'Author: {lvl["author"]}\n')
                f.write(f'Collection: {cname}\n')
                f.write('\n')


def main():
    levels_dir = Path('/sessions/epic-laughing-ritchie/mnt/sokoban-utils/test-levels')
    output_path = Path('/sessions/epic-laughing-ritchie/mnt/outputs/normalised-levels.xsb')

    xsb_files = sorted(levels_dir.glob('*.xsb'))
    print(f"Found {len(xsb_files)} XSB files\n")

    all_collections = []
    expected = {
        'Aymeric_Du_Peloux_282.xsb': 282,
        'Grigr2001_100.xsb': 100,
        'Grigr2002_40.xsb': 40,
        'GrigrSpecial_40.xsb': 40,
        'Holland_81.xsb': 81,
        'Microban II_135.xsb': 135,
        'Microban_155.xsb': 155,
        'Sasquatch II_50.xsb': 50,
        'Sasquatch_50.xsb': 50,
        'Sasquatch_III_50.xsb': 50,
        'Sasquatch_IV_50.xsb': 50,
        'Sasquatch_VII_50.xsb': 50,
        'Sasquatch_VI_50.xsb': 50,
        'Sasquatch_V_50.xsb': 50,
        'SokEvo_107.xsb': 107,
        'SokHard_163.xsb': 163,
        'Sven_1623.xsb': 1623,
        'XSokoban_90.xsb': 90,
        'Yoshio Murase_Handmade_54.xsb': 54,
        'Yoshio_Murase_Autogenerated_52.xsb': 52,
    }

    for xf in xsb_files:
        coll_info, levels = parse_xsb_file(xf)
        all_collections.append((coll_info, levels))
        exp = expected.get(xf.name, '?')
        ok = '✓' if len(levels) == exp else f'✗ (expected {exp})'
        print(f"  {xf.name:42s} → {len(levels):4d} levels  {ok}")

    total = sum(len(lvls) for _, lvls in all_collections)
    print(f"\nTotal levels parsed: {total}")

    write_normalized(output_path, all_collections)
    print(f"\nWrote: {output_path}")
    print(f"File size: {output_path.stat().st_size:,} bytes")


if __name__ == '__main__':
    main()
