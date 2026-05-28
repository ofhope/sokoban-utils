//! Unified deadlock-set registry.
//!
//! YASS folds every deadlock-set family — closed-edge, controller, freeze,
//! diagonal-center, and dynamically-discovered — into one shared structure
//! (`TDeadlockSets` in YASS.pas:751).  Each set carries a `capacity` and a
//! `flags` set distinguishing the family.  The A* loop only needs to ask one
//! question at every push: "does the box landing here overflow any set this
//! cell belongs to?"  That uniform query is what makes per-cell membership
//! indexing pay for itself.
//!
//! This module is the Rust counterpart.  At the time of writing it holds
//! only the closed-edge and general-closed families (the ones already
//! implemented in this crate), but the data model is intentionally shaped
//! to absorb controller / freeze / diagonal-center / dynamic sets without
//! further refactoring.

use std::collections::HashSet;

use crate::bitplane::BitPlane;

// ── Directions ────────────────────────────────────────────────────────────
//
// Duplicated from solver.rs intentionally: the deadlock-set computations are
// self-contained, and re-exporting DIRS just to dodge four lines of code adds
// cross-module coupling we'd rather avoid.

const DIRS: [(i8, i8); 4] = [(0, -1), (0, 1), (-1, 0), (1, 0)];

fn in_bounds(width: u8, height: u8, x: i8, y: i8) -> bool {
    x >= 0 && y >= 0 && (x as u8) < width && (y as u8) < height
}

// ── Public types ──────────────────────────────────────────────────────────

/// Which YASS family a `DeadlockSet` came from.
///
/// Mirrors a subset of `TDeadlockSetFlag` from YASS.pas:227.  We add variants
/// here as the corresponding families are ported.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum DeadlockSetKind {
    /// Closed-edge: contiguous wall-edge run with sealed lateral ends.
    /// YASS does not have a dedicated flag for this — they're emitted by
    /// the same `CalculateDeadlockSets` pass that produces the general
    /// closed sets, with no special flag set.
    ClosedEdge,
    /// General closed superset of a seed cell (L-/T-/irregular shapes).
    /// Found by forced-expansion BFS; produces a superset of the
    /// closed-edge family.
    GeneralClosed,
    /// `dsfControllerSet` — placed for completeness; not yet emitted.
    #[allow(dead_code)]
    Controller,
    /// `dsfFreezeSet` — placed for completeness; not yet emitted.
    #[allow(dead_code)]
    Freeze,
    /// `dsfDiagonalCenterSquares` — placed for completeness; not yet emitted.
    #[allow(dead_code)]
    DiagonalCenter,
    /// Dynamically discovered no-progress pattern (`dsfIsANoProgressPattern`
    /// and friends).  Emitted by the in-search `TTAddNoPushesDeadlock` /
    /// `CommitDeadlockSet` engine — not yet implemented.
    #[allow(dead_code)]
    Dynamic,
}

/// A single deadlock set: a fixed group of cells with a capacity constraint.
///
/// Invariant tested at every push: when a box lands on a cell that belongs
/// to a set, count the boxes currently in that set's `cells`; if that count
/// exceeds `capacity`, the state is a deadlock.
///
/// For closed-edge and general-closed sets, `capacity == goal_count`
/// (the set can hold at most one box per goal it contains).  Other YASS
/// families use lower capacities — e.g. a controller set may carry
/// `capacity == 0` even when its cells contain goals — which is why we keep
/// `capacity` and `goal_count` as distinct fields rather than collapsing them.
pub(crate) struct DeadlockSet {
    pub cells: Vec<(u8, u8)>,
    pub goal_count: u32,
    pub capacity: u32,
    pub kind: DeadlockSetKind,
}

/// All deadlock sets for a level plus a per-cell index.
///
/// `membership[flat]` lists the indices of every set the cell at `flat`
/// belongs to, supporting the O(memberships-of-pushed-cell) overflow check
/// run on every push.
pub(crate) struct DeadlockSetRegistry {
    pub sets: Vec<DeadlockSet>,
    pub membership: Vec<Vec<u32>>,
    width: u8,
}

impl DeadlockSetRegistry {
    pub fn new(width: u8, height: u8) -> Self {
        Self {
            sets: Vec::new(),
            membership: vec![Vec::new(); width as usize * height as usize],
            width,
        }
    }

    /// Append a new set, indexing its cells in `membership`.
    pub fn add(&mut self, set: DeadlockSet) -> u32 {
        let idx = self.sets.len() as u32;
        for &(x, y) in &set.cells {
            self.membership[y as usize * self.width as usize + x as usize].push(idx);
        }
        self.sets.push(set);
        idx
    }

    /// Returns `true` when placing a box at `(x, y)` would push any set this
    /// cell belongs to over its capacity.  `boxes` must already reflect the
    /// placement.
    pub fn has_overflow_deadlock(&self, x: u8, y: u8, boxes: &BitPlane) -> bool {
        let flat = y as usize * self.width as usize + x as usize;
        self.membership[flat].iter().any(|&idx| {
            let set = &self.sets[idx as usize];
            let count = set.cells.iter()
                .filter(|&&(cx, cy)| boxes.get(cx, cy))
                .count() as u32;
            count > set.capacity
        })
    }

    /// Render a diagnostic histogram for `Solver::set_diagnostics`.
    pub fn diagnostics(&self) -> String {
        let total_sets = self.sets.len();
        let counts: Vec<usize> = self.membership.iter().map(|v| v.len()).collect();
        let max_count = counts.iter().copied().max().unwrap_or(0);

        let cap = max_count.min(10);
        let mut hist = vec![0usize; cap + 2];
        for &c in &counts {
            if c <= cap { hist[c] += 1; } else { hist[cap + 1] += 1; }
        }

        let mut out = format!(
            "deadlock sets: {}  |  max memberships per cell: {}\n  histogram (memberships → cell count):\n",
            total_sets, max_count,
        );
        for (i, &n) in hist.iter().enumerate() {
            if i == cap + 1 {
                out.push_str(&format!("    >{}  → {}\n", cap, n));
            } else {
                out.push_str(&format!("    {}  → {}\n", i, n));
            }
        }
        out
    }
}

// ── Closed-edge precomputation ────────────────────────────────────────────
//
// A "closed-edge" is a contiguous run of floor cells all sharing the same
// sealed wall on one side (top / bottom / left / right), whose run endpoints
// are also blocked by walls or the level boundary on the lateral axis.
//
// Because boxes on such a segment can never leave it, the maximum number of
// boxes that can be satisfied is goal_count.  Any state with more boxes than
// goals in the set is a deadlock.
//
// Dead squares act as run-breakers (boxes can never land there).
// Sets where goal_count >= run_length are skipped (boxes can always fit).
//
// This is the Rust equivalent of YASS's simplest deadlock-set class.

pub(crate) fn compute_closed_edge_sets(
    registry: &mut DeadlockSetRegistry,
    walls: &BitPlane,
    goals: &BitPlane,
    dead: &BitPlane,
    width: u8,
    height: u8,
) {
    // Horizontal scans — top-wall and bottom-wall edges.
    for y in 0..height {
        // Top-wall edge: each cell has wall or OOB directly above.
        {
            let mut xs: Option<u8> = None;
            for xi in 0..=(width as u16) {
                let in_run = xi < width as u16 && {
                    let x = xi as u8;
                    !walls.get(x, y) && !dead.get(x, y)
                        && (y == 0 || walls.get(x, y - 1))
                };
                if in_run {
                    if xs.is_none() { xs = Some(xi as u8); }
                } else if let Some(x_start) = xs.take() {
                    let x_end = xi as u8 - 1;
                    try_record_h_run(x_start, x_end, y, walls, goals, dead, width, registry);
                }
            }
        }
        // Bottom-wall edge: each cell has wall or OOB directly below.
        {
            let mut xs: Option<u8> = None;
            for xi in 0..=(width as u16) {
                let in_run = xi < width as u16 && {
                    let x = xi as u8;
                    !walls.get(x, y) && !dead.get(x, y)
                        && (y + 1 >= height || walls.get(x, y + 1))
                };
                if in_run {
                    if xs.is_none() { xs = Some(xi as u8); }
                } else if let Some(x_start) = xs.take() {
                    let x_end = xi as u8 - 1;
                    try_record_h_run(x_start, x_end, y, walls, goals, dead, width, registry);
                }
            }
        }
    }

    // Vertical scans — left-wall and right-wall edges.
    for x in 0..width {
        // Left-wall edge.
        {
            let mut ys: Option<u8> = None;
            for yi in 0..=(height as u16) {
                let in_run = yi < height as u16 && {
                    let y = yi as u8;
                    !walls.get(x, y) && !dead.get(x, y)
                        && (x == 0 || walls.get(x - 1, y))
                };
                if in_run {
                    if ys.is_none() { ys = Some(yi as u8); }
                } else if let Some(y_start) = ys.take() {
                    let y_end = yi as u8 - 1;
                    try_record_v_run(x, y_start, y_end, walls, goals, dead, width, height, registry);
                }
            }
        }
        // Right-wall edge.
        {
            let mut ys: Option<u8> = None;
            for yi in 0..=(height as u16) {
                let in_run = yi < height as u16 && {
                    let y = yi as u8;
                    !walls.get(x, y) && !dead.get(x, y)
                        && (x + 1 >= width || walls.get(x + 1, y))
                };
                if in_run {
                    if ys.is_none() { ys = Some(yi as u8); }
                } else if let Some(y_start) = ys.take() {
                    let y_end = yi as u8 - 1;
                    try_record_v_run(x, y_start, y_end, walls, goals, dead, width, height, registry);
                }
            }
        }
    }
}

fn try_record_h_run(
    xs: u8, xe: u8, y: u8,
    walls: &BitPlane,
    goals: &BitPlane,
    dead: &BitPlane,
    width: u8,
    registry: &mut DeadlockSetRegistry,
) {
    let left_bound  = xs == 0 || walls.get(xs - 1, y) || dead.get(xs - 1, y);
    let right_bound = xe >= width - 1 || walls.get(xe + 1, y) || dead.get(xe + 1, y);
    if !left_bound || !right_bound { return; }

    let cells: Vec<(u8, u8)> = (xs..=xe).map(|x| (x, y)).collect();
    let goal_count = cells.iter().filter(|&&(x, y)| goals.get(x, y)).count() as u32;
    if goal_count >= cells.len() as u32 { return; }

    registry.add(DeadlockSet {
        cells,
        goal_count,
        capacity: goal_count,
        kind: DeadlockSetKind::ClosedEdge,
    });
}

fn try_record_v_run(
    x: u8, ys: u8, ye: u8,
    walls: &BitPlane,
    goals: &BitPlane,
    dead: &BitPlane,
    width: u8,
    height: u8,
    registry: &mut DeadlockSetRegistry,
) {
    let _ = width; // currently unused, kept for symmetry with try_record_h_run
    let top_bound = ys == 0 || walls.get(x, ys - 1) || dead.get(x, ys - 1);
    let bot_bound = ye >= height - 1 || walls.get(x, ye + 1) || dead.get(x, ye + 1);
    if !top_bound || !bot_bound { return; }

    let cells: Vec<(u8, u8)> = (ys..=ye).map(|y| (x, y)).collect();
    let goal_count = cells.iter().filter(|&&(x, y)| goals.get(x, y)).count() as u32;
    if goal_count >= cells.len() as u32 { return; }

    registry.add(DeadlockSet {
        cells,
        goal_count,
        capacity: goal_count,
        kind: DeadlockSetKind::ClosedEdge,
    });
}

// ── General closed-set detection ──────────────────────────────────────────
//
// For each non-dead, non-wall floor cell (the "seed") we compute its
// minimum closed superset via forced expansion: any escape route from the
// current set S must be closed by pulling the destination cell into S.  The
// process terminates when S is stable (truly closed) or exceeds MAX_SIZE
// (the region is too open to be a useful constraint).
//
// SOUNDNESS — no false positives are possible:
//   After expansion, for every cell p ∈ S and every direction d, the push
//   "box at p in direction d" is either impossible (dest is OOB/wall/dead,
//   or from is OOB/wall) or the dest is already in S.  So S is closed under
//   *any* player position, not just the current one.  If box_count > goal_count
//   in S, the surplus box can never reach a goal — permanent deadlock.
//
// COMPLETENESS — some real deadlocks are missed (acceptable):
//   We treat the player as omnipresent (able to reach any non-wall cell).
//   States where the player is blocked from an escape route by boxes are not
//   detected here, but may be caught by freeze detection or corral pruning.
//
// General-closed subsumes closed-edge (it finds all linear closed-edge runs
// plus L-, T-, and irregular shapes) but we keep closed-edge for the fast
// linear scan.  Duplicates between the two pre-computations result in
// redundant (but harmless) membership entries; the `seen` HashSet prevents
// duplicates within general-closed.

const MAX_CLOSED_SET_SIZE: usize = 8;

pub(crate) fn compute_general_closed_sets(
    registry: &mut DeadlockSetRegistry,
    walls: &BitPlane,
    goals: &BitPlane,
    dead: &BitPlane,
    width: u8,
    height: u8,
) {
    let mut seen: HashSet<Vec<(u8, u8)>> = HashSet::new();

    for y in 0..height {
        for x in 0..width {
            if walls.get(x, y) || dead.get(x, y) { continue; }

            let Some(mut s) = find_min_closure(x, y, walls, dead, width, height) else { continue };

            s.sort_unstable();
            if seen.contains(&s) { continue; }

            let goal_count = s.iter().filter(|&&(cx, cy)| goals.get(cx, cy)).count() as u32;

            seen.insert(s.clone());

            if goal_count < s.len() as u32 {
                registry.add(DeadlockSet {
                    cells: s,
                    goal_count,
                    capacity: goal_count,
                    kind: DeadlockSetKind::GeneralClosed,
                });
            }
        }
    }
}

fn find_min_closure(
    sx: u8, sy: u8,
    walls: &BitPlane,
    dead: &BitPlane,
    width: u8, height: u8,
) -> Option<Vec<(u8, u8)>> {
    let mut cells: Vec<(u8, u8)> = vec![(sx, sy)];
    let mut head = 0usize;

    while head < cells.len() {
        let (px, py) = cells[head];
        head += 1;

        for &(dx, dy) in &DIRS {
            let dest_x = px as i8 + dx;
            let dest_y = py as i8 + dy;
            let from_x = px as i8 - dx;
            let from_y = py as i8 - dy;

            if !in_bounds(width, height, dest_x, dest_y) { continue; }
            let (dest_x, dest_y) = (dest_x as u8, dest_y as u8);
            if walls.get(dest_x, dest_y) || dead.get(dest_x, dest_y) { continue; }

            if cells.contains(&(dest_x, dest_y)) { continue; }

            if !in_bounds(width, height, from_x, from_y) { continue; }
            if walls.get(from_x as u8, from_y as u8) { continue; }

            cells.push((dest_x, dest_y));
            if cells.len() > MAX_CLOSED_SET_SIZE { return None; }
        }
    }

    Some(cells)
}
