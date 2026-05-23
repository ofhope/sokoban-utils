use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::cmp::Reverse;
use wasm_bindgen::prelude::*;
use crate::bitplane::BitPlane;
use crate::level::SokobanLevel;

// ── Public WASM types ──────────────────────────────────────────────────────

#[wasm_bindgen]
pub struct SolverResult {
    solved: bool,
    push_count: u32,
    /// Flat push sequence: [from_x, from_y, to_x, to_y, ...] pairs
    pushes: Vec<u8>,
    /// Full solution in LURD format: lowercase = player walk, uppercase = box push.
    /// Empty string when unsolved.
    moves: String,
}

#[wasm_bindgen]
impl SolverResult {
    pub fn solved(&self) -> bool { self.solved }
    pub fn push_count(&self) -> u32 { self.push_count }
    /// Returns the push sequence as [from_x, from_y, to_x, to_y, ...].
    pub fn pushes(&self) -> Vec<u8> { self.pushes.clone() }
    /// Returns the full move sequence in LURD format.
    /// Lowercase letters are player walks, uppercase letters are box pushes.
    /// u/U = up, d/D = down, l/L = left, r/R = right.
    /// Empty string when unsolved.
    pub fn moves(&self) -> String { self.moves.clone() }
}

#[wasm_bindgen]
pub struct Solver {
    width: u8,
    height: u8,
    walls: BitPlane,
    goals: BitPlane,
    dead: BitPlane,
    goal_distances: Vec<Vec<u32>>,
    /// `tunnel_data[flat][d]` is `true` when pushing a box onto cell `flat`
    /// arriving from DIRS[d] should be extended further in that same direction.
    /// A cell qualifies when it is a non-goal corridor: walled on both sides
    /// perpendicular to the push direction, not a dead square, and the next
    /// cell in the push direction is also a legal, non-dead, non-goal square.
    tunnel_data: Vec<[bool; 4]>,
    /// Precomputed closed-edge deadlock sets (Phase 2).
    /// Each set covers a contiguous wall-edge segment with sealed lateral ends.
    deadlock_sets: Vec<ClosedEdgeSet>,
    /// `set_membership[flat]` lists every set index that cell belongs to.
    /// Used to quickly find which sets to check when a box lands on that cell.
    set_membership: Vec<Vec<u32>>,
}

#[wasm_bindgen]
impl Solver {
    /// Build a solver from a level. Precomputation happens here (once per level load).
    pub fn new(level: &SokobanLevel) -> Solver {
        let dead = compute_dead_squares(&level.walls, &level.goals, level.width, level.height);
        let goal_distances = precompute_goal_distances(
            &level.walls, &level.goals, level.width, level.height,
        );
        let tunnel_data = compute_tunnels(&level.walls, &level.goals, &dead, level.width, level.height);
        let (mut deadlock_sets, mut set_membership) = compute_closed_edge_sets(
            &level.walls, &level.goals, &dead, level.width, level.height,
        );
        // Phase 3: extend with general closed sets (L-shapes, T-shapes, etc.)
        compute_general_closed_sets(
            &level.walls, &level.goals, &dead, level.width, level.height,
            &mut deadlock_sets, &mut set_membership,
        );
        Solver {
            width: level.width,
            height: level.height,
            walls: level.walls.clone(),
            goals: level.goals.clone(),
            dead,
            goal_distances,
            tunnel_data,
            deadlock_sets,
            set_membership,
        }
    }

    /// Run A* over push states. Returns None (unsolvable or limit hit) via solved=false.
    pub fn solve(&self, level: &SokobanLevel, max_nodes: u32) -> SolverResult {
        let initial_boxes = level.boxes.clone();
        let initial_player = self.normalise_player(
            level.player_pos, &initial_boxes,
        );

        let initial_state = State { boxes: initial_boxes.clone(), player_norm: initial_player };

        if self.is_solved(&initial_state.boxes) {
            return SolverResult { solved: true, push_count: 0, pushes: vec![], moves: String::new() };
        }

        let mut visited: HashMap<State, u32> = HashMap::new();
        let mut heap: BinaryHeap<Reverse<Node>> = BinaryHeap::new();
        let h0 = self.heuristic(&initial_state.boxes);

        heap.push(Reverse(Node {
            f: h0,
            cost: 0,
            state: initial_state,
            path: vec![],
        }));

        let mut nodes_explored = 0u32;

        while let Some(Reverse(node)) = heap.pop() {
            nodes_explored += 1;
            if nodes_explored > max_nodes {
                break;
            }

            if self.is_solved(&node.state.boxes) {
                let mut pushes = Vec::with_capacity(node.path.len() * 4);
                for p in &node.path {
                    pushes.push(p.from_x);
                    pushes.push(p.from_y);
                    pushes.push(p.to_x);
                    pushes.push(p.to_y);
                }
                let moves = reconstruct_moves(
                    level.player_pos, &initial_boxes, &node.path,
                    self.width, &self.walls,
                );
                return SolverResult { solved: true, push_count: node.cost, pushes, moves };
            }

            if let Some(&best) = visited.get(&node.state) {
                if best <= node.f { continue; }
            }
            visited.insert(node.state.clone(), node.f);

            for push in self.generate_pushes(&node.state) {
                let mut new_boxes = node.state.boxes.clone();
                new_boxes.clear(push.from_x, push.from_y);

                // Prune: dead square
                if self.dead.get(push.to_x, push.to_y) { continue; }

                new_boxes.set(push.to_x, push.to_y);

                // Prune: closed-edge set overflow — cheap O(|set|) check before
                // the more expensive freeze check.  Catches cases where the
                // pushed box lands in the interior of a sealed wall-edge segment
                // that already holds as many boxes as it has goals, even when
                // the pushed box is not adjacent to any wall (so freeze would
                // miss it).
                if self.has_set_deadlock(push.to_x, push.to_y, &new_boxes) { continue; }

                // Prune: freeze deadlock (biaxial chain detection)
                if is_freeze_deadlock(push.to_x, push.to_y, &new_boxes, &self.walls, &self.goals, &self.dead) {
                    continue;
                }

                let new_player_norm = self.normalise_player(
                    push.from_flat as u16, &new_boxes,
                );

                let new_state = State { boxes: new_boxes, player_norm: new_player_norm };
                let g = node.cost + push.steps;
                let h = self.heuristic(&new_state.boxes);
                let mut new_path = node.path.clone();
                new_path.push(push);

                heap.push(Reverse(Node { f: g + h, cost: g, state: new_state, path: new_path }));
            }
        }

        SolverResult { solved: false, push_count: 0, pushes: vec![], moves: String::new() }
    }
}

// ── Internal types ─────────────────────────────────────────────────────────

/// A closed-edge deadlock set: a contiguous wall-edge segment whose boxes
/// can never escape.  If `boxes_in_set > goal_count`, the position is a deadlock.
///
/// Corresponds to the simplest class of YASS deadlock sets — a straight
/// wall-edge with both lateral ends sealed by walls or the level boundary.
struct ClosedEdgeSet {
    /// All (x, y) cells in this set (no walls, no dead squares).
    cells: Vec<(u8, u8)>,
    /// Number of goal cells within the set.
    goal_count: u32,
}

#[derive(Clone, Eq, PartialEq, Hash)]
struct State {
    boxes: BitPlane,

    player_norm: u16,
}

#[derive(Clone, Eq, PartialEq)]
struct Push {
    from_x: u8,
    from_y: u8,
    to_x: u8,
    to_y: u8,
    /// Flat index of the cell the player must stand on to make this push.
    from_flat: usize,
    /// Number of individual box-push steps (≥1; >1 for tunnel macro-pushes).
    steps: u32,
}

#[derive(Eq, PartialEq)]
struct Node {
    f: u32,
    cost: u32,
    state: State,
    path: Vec<Push>,
}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.f.cmp(&other.f)
    }
}
impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

// ── Directions ─────────────────────────────────────────────────────────────

const DIRS: [(i8, i8); 4] = [(0, -1), (0, 1), (-1, 0), (1, 0)];

fn in_bounds(width: u8, height: u8, x: i8, y: i8) -> bool {
    x >= 0 && y >= 0 && (x as u8) < width && (y as u8) < height
}

// ── Solver internals ───────────────────────────────────────────────────────

impl Solver {
    fn is_solved(&self, boxes: &BitPlane) -> bool {
        self.goals.iter_set_bits().all(|(x, y)| boxes.get(x, y))
    }

    /// A* heuristic: sum of each box's minimum push-distance to nearest goal.
    fn heuristic(&self, boxes: &BitPlane) -> u32 {
        boxes.iter_set_bits().map(|(bx, by)| {
            let flat = by as usize * self.width as usize + bx as usize;
            self.goal_distances
                .iter()
                .map(|gdist| gdist[flat])
                .min()
                .unwrap_or(u32::MAX / 2)
        }).sum()
    }

    /// Normalise player position to the top-left-most reachable cell.
    fn normalise_player(&self, player_pos: u16, boxes: &BitPlane) -> u16 {
        let px = (player_pos as usize % self.width as usize) as u8;
        let py = (player_pos as usize / self.width as usize) as u8;
        let reachable = flood_fill(px, py, &self.walls, boxes, self.width, self.height);
        let flat = reachable
            .iter_set_bits()
            .next()
            .map(|(x, y)| y as u16 * self.width as u16 + x as u16)
            .unwrap_or(player_pos);
        flat
    }

    /// Check whether placing a box at `(to_x, to_y)` causes any closed-edge
    /// set to overflow its goal capacity.
    ///
    /// `boxes` must already include the new box at `(to_x, to_y)`.
    fn has_set_deadlock(&self, to_x: u8, to_y: u8, boxes: &BitPlane) -> bool {
        let flat = to_y as usize * self.width as usize + to_x as usize;
        self.set_membership[flat].iter().any(|&set_idx| {
            let set = &self.deadlock_sets[set_idx as usize];
            let box_count = set.cells.iter()
                .filter(|&&(x, y)| boxes.get(x, y))
                .count() as u32;
            box_count > set.goal_count
        })
    }

    /// Generate all valid pushes from a search state.
    /// Pushes that enter a tunnel square are extended to the far end of the
    /// tunnel as a single macro-push (the box passes through intermediate
    /// squares without stopping).  This reduces the branching factor on
    /// corridor-heavy levels without affecting correctness.
    fn generate_pushes(&self, state: &State) -> Vec<Push> {
        let px = (state.player_norm as usize % self.width as usize) as u8;
        let py = (state.player_norm as usize / self.width as usize) as u8;
        let reachable = flood_fill(px, py, &self.walls, &state.boxes, self.width, self.height);
        let mut pushes = Vec::new();

        for (bx, by) in state.boxes.iter_set_bits() {
            for (dir_idx, &(dx, dy)) in DIRS.iter().enumerate() {
                let push_from_x = bx as i8 - dx;
                let push_from_y = by as i8 - dy;
                let push_to_x   = bx as i8 + dx;
                let push_to_y   = by as i8 + dy;

                if !in_bounds(self.width, self.height, push_from_x, push_from_y) { continue; }
                if !in_bounds(self.width, self.height, push_to_x,   push_to_y)   { continue; }

                let (pfx, pfy) = (push_from_x as u8, push_from_y as u8);
                let (mut ptx, mut pty) = (push_to_x as u8, push_to_y as u8);

                if !reachable.get(pfx, pfy)    { continue; }
                if self.walls.get(ptx, pty)    { continue; }
                if state.boxes.get(ptx, pty)   { continue; }

                // Extend through tunnel squares: keep advancing in (dx,dy) while
                // the current landing square is a tunnel for this direction AND
                // the next square is free (not a wall, not a box).
                let mut steps = 1u32;
                loop {
                    let flat = pty as usize * self.width as usize + ptx as usize;
                    if !self.tunnel_data[flat][dir_idx] { break; }
                    let nx = ptx as i8 + dx;
                    let ny = pty as i8 + dy;
                    if !in_bounds(self.width, self.height, nx, ny) { break; }
                    let (nx, ny) = (nx as u8, ny as u8);
                    if self.walls.get(nx, ny) || state.boxes.get(nx, ny) { break; }
                    ptx = nx;
                    pty = ny;
                    steps += 1;
                }

                let from_flat = pfy as usize * self.width as usize + pfx as usize;
                pushes.push(Push { from_x: bx, from_y: by, to_x: ptx, to_y: pty, from_flat, steps });
            }
        }
        pushes
    }
}

// ── Flood fill ─────────────────────────────────────────────────────────────

fn flood_fill(
    start_x: u8, start_y: u8,
    walls: &BitPlane, boxes: &BitPlane,
    width: u8, height: u8,
) -> BitPlane {
    let mut visited = BitPlane::new(width, height);
    let mut queue = VecDeque::new();
    if !walls.get(start_x, start_y) && !boxes.get(start_x, start_y) {
        visited.set(start_x, start_y);
        queue.push_back((start_x, start_y));
    }
    while let Some((x, y)) = queue.pop_front() {
        for (dx, dy) in DIRS {
            let nx = x as i8 + dx;
            let ny = y as i8 + dy;
            if !in_bounds(width, height, nx, ny) { continue; }
            let (nx, ny) = (nx as u8, ny as u8);
            if visited.get(nx, ny) || walls.get(nx, ny) || boxes.get(nx, ny) { continue; }
            visited.set(nx, ny);
            queue.push_back((nx, ny));
        }
    }
    visited
}

// ── Dead square precomputation ─────────────────────────────────────────────

fn compute_dead_squares(walls: &BitPlane, goals: &BitPlane, width: u8, height: u8) -> BitPlane {
    let mut dead = BitPlane::new(width, height);

    for y in 0..height {
        for x in 0..width {
            if walls.get(x, y) || goals.get(x, y) { continue; }
            let blocked_h = (x == 0 || walls.get(x - 1, y)) || (x + 1 >= width || walls.get(x + 1, y));
            let blocked_v = (y == 0 || walls.get(x, y - 1)) || (y + 1 >= height || walls.get(x, y + 1));
            if blocked_h && blocked_v {
                dead.set(x, y);
            }
        }
    }

    // Propagate: a cell next to a dead cell along a wall is also dead.
    loop {
        let mut changed = false;
        for y in 0..height {
            for x in 0..width {
                if walls.get(x, y) || goals.get(x, y) || dead.get(x, y) { continue; }
                // Horizontal dead propagation along a horizontal wall
                for dx in [-1i8, 1] {
                    let nx = x as i8 + dx;
                    if nx < 0 || nx >= width as i8 { continue; }
                    if dead.get(nx as u8, y)
                        && (y == 0 || walls.get(x, y - 1))
                        && (y + 1 >= height || walls.get(x, y + 1))
                    {
                        dead.set(x, y);
                        changed = true;
                    }
                }
                // Vertical dead propagation along a vertical wall
                for dy in [-1i8, 1] {
                    let ny = y as i8 + dy;
                    if ny < 0 || ny >= height as i8 { continue; }
                    if dead.get(x, ny as u8)
                        && (x == 0 || walls.get(x - 1, y))
                        && (x + 1 >= width || walls.get(x + 1, y))
                    {
                        dead.set(x, y);
                        changed = true;
                    }
                }
            }
        }
        if !changed { break; }
    }

    dead
}

// ── Tunnel precomputation ──────────────────────────────────────────────────

/// For each cell and each of the 4 push directions, return whether arriving
/// at that cell from that direction puts the box in a "tunnel" that should be
/// extended further.
///
/// A cell `c` is a tunnel for direction `d` when ALL of:
///   1. `c` is not a wall, not a dead square, and not a goal.
///   2. The two cells perpendicular to `d` at `c` are both walls or out-of-bounds.
///   3. The next cell `c + d` is reachable (not a wall, not dead, not out-of-bounds).
///
/// Condition 2 is the corridor test: walls on both sides perpendicular to
/// the push mean there is no reason to stop here — the box must continue.
/// Goals are excluded because the player may intentionally stop a box there.
fn compute_tunnels(
    walls: &BitPlane,
    goals: &BitPlane,
    dead: &BitPlane,
    width: u8, height: u8,
) -> Vec<[bool; 4]> {
    let cells = width as usize * height as usize;
    let mut out = vec![[false; 4]; cells];

    // Perpendicular direction pairs: for DIRS[i] = (dx,dy), the perpendicular
    // cells along the other axis are at offsets (±perp_dx, ±perp_dy).
    // DIRS = [(0,-1),(0,1),(-1,0),(1,0)]
    // perps[i] = the two perpendicular unit vectors for DIRS[i]
    let perps: [[(i8, i8); 2]; 4] = [
        [(-1, 0), (1, 0)], // for (0,-1) up   → left and right
        [(-1, 0), (1, 0)], // for (0, 1) down → left and right
        [(0, -1), (0, 1)], // for (-1,0) left → up and down
        [(0, -1), (0, 1)], // for (1, 0) right→ up and down
    ];

    for y in 0..height {
        for x in 0..width {
            if walls.get(x, y) || dead.get(x, y) || goals.get(x, y) { continue; }
            let flat = y as usize * width as usize + x as usize;

            for (dir_idx, &(dx, dy)) in DIRS.iter().enumerate() {
                // Next cell in push direction must be legal.
                let nx = x as i8 + dx;
                let ny = y as i8 + dy;
                if !in_bounds(width, height, nx, ny) { continue; }
                let (nx, ny) = (nx as u8, ny as u8);
                if walls.get(nx, ny) || dead.get(nx, ny) { continue; }

                // Both perpendicular neighbours at the current cell must be walls
                // (or out-of-bounds — treated as walls for this purpose).
                let [(pdx0, pdy0), (pdx1, pdy1)] = perps[dir_idx];
                let p0x = x as i8 + pdx0;
                let p0y = y as i8 + pdy0;
                let p1x = x as i8 + pdx1;
                let p1y = y as i8 + pdy1;

                let p0_wall = !in_bounds(width, height, p0x, p0y)
                    || walls.get(p0x as u8, p0y as u8);
                let p1_wall = !in_bounds(width, height, p1x, p1y)
                    || walls.get(p1x as u8, p1y as u8);

                if p0_wall && p1_wall {
                    out[flat][dir_idx] = true;
                }
            }
        }
    }
    out
}

// ── Closed-edge deadlock sets ──────────────────────────────────────────────
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

fn compute_closed_edge_sets(
    walls: &BitPlane,
    goals: &BitPlane,
    dead: &BitPlane,
    width: u8,
    height: u8,
) -> (Vec<ClosedEdgeSet>, Vec<Vec<u32>>) {
    let cells_count = width as usize * height as usize;
    let mut sets: Vec<ClosedEdgeSet> = Vec::new();
    let mut membership: Vec<Vec<u32>> = vec![Vec::new(); cells_count];

    // Horizontal scans — top-wall and bottom-wall edges.
    for y in 0..height {
        // Top-wall edge: each cell has wall or OOB directly above.
        // A box there cannot be pushed up (wall) or down (player needs the wall cell).
        {
            let mut xs: Option<u8> = None;
            // Iterate one past the end to flush any in-progress run.
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
                    try_record_h_run(x_start, x_end, y, walls, goals, dead, width,
                                     &mut sets, &mut membership);
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
                    try_record_h_run(x_start, x_end, y, walls, goals, dead, width,
                                     &mut sets, &mut membership);
                }
            }
        }
    }

    // Vertical scans — left-wall and right-wall edges.
    for x in 0..width {
        // Left-wall edge: each cell has wall or OOB directly to the left.
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
                    try_record_v_run(x, y_start, y_end, walls, goals, dead, width, height,
                                     &mut sets, &mut membership);
                }
            }
        }
        // Right-wall edge: each cell has wall or OOB directly to the right.
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
                    try_record_v_run(x, y_start, y_end, walls, goals, dead, width, height,
                                     &mut sets, &mut membership);
                }
            }
        }
    }

    (sets, membership)
}

/// Attempt to record a horizontal closed-edge run [xs..=xe] at row `y`.
///
/// The run is only recorded if it is laterally bounded (wall or OOB on both
/// the left of `xs` and the right of `xe`) and has fewer goals than cells
/// (otherwise the capacity can never be exceeded).
fn try_record_h_run(
    xs: u8, xe: u8, y: u8,
    walls: &BitPlane,
    goals: &BitPlane,
    dead: &BitPlane,
    width: u8,
    sets: &mut Vec<ClosedEdgeSet>,
    membership: &mut Vec<Vec<u32>>,
) {
    // Left bound: wall, OOB, or dead square to the left of the run start.
    // Dead squares count as walls here — boxes are never pushed there, so a
    // box at `xs` cannot escape left (the solver would prune that push).
    let left_bound  = xs == 0 || walls.get(xs - 1, y) || dead.get(xs - 1, y);
    // Right bound: same logic on the right side.
    let right_bound = xe >= width - 1 || walls.get(xe + 1, y) || dead.get(xe + 1, y);
    if !left_bound || !right_bound { return; }

    let cells: Vec<(u8, u8)> = (xs..=xe).map(|x| (x, y)).collect();
    let goal_count = cells.iter().filter(|&&(x, y)| goals.get(x, y)).count() as u32;
    if goal_count >= cells.len() as u32 { return; } // can never overflow

    let set_idx = sets.len() as u32;
    for &(x, y) in &cells {
        membership[y as usize * width as usize + x as usize].push(set_idx);
    }
    sets.push(ClosedEdgeSet { cells, goal_count });
}

/// Attempt to record a vertical closed-edge run [ys..=ye] at column `x`.
fn try_record_v_run(
    x: u8, ys: u8, ye: u8,
    walls: &BitPlane,
    goals: &BitPlane,
    dead: &BitPlane,
    width: u8,
    height: u8,
    sets: &mut Vec<ClosedEdgeSet>,
    membership: &mut Vec<Vec<u32>>,
) {
    let top_bound = ys == 0 || walls.get(x, ys - 1) || dead.get(x, ys - 1);
    let bot_bound = ye >= height - 1 || walls.get(x, ye + 1) || dead.get(x, ye + 1);
    if !top_bound || !bot_bound { return; }

    let cells: Vec<(u8, u8)> = (ys..=ye).map(|y| (x, y)).collect();
    let goal_count = cells.iter().filter(|&&(x, y)| goals.get(x, y)).count() as u32;
    if goal_count >= cells.len() as u32 { return; }

    let set_idx = sets.len() as u32;
    for &(x, y) in &cells {
        membership[y as usize * width as usize + x as usize].push(set_idx);
    }
    sets.push(ClosedEdgeSet { cells, goal_count });
}

// ── General closed-set detection (Phase 3) ────────────────────────────────
//
// For each non-dead, non-wall floor cell (the "seed") we compute its
// *minimum closed superset* via forced expansion: any escape route from the
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
// Phase 3 subsumes Phase 2 (it finds all linear closed-edge runs plus L-,
// T-, and irregular shapes) but we keep Phase 2 for the fast linear scan.
// Duplicates between Phase 2 and Phase 3 result in redundant (but harmless)
// membership entries; the `seen` HashSet prevents duplicates within Phase 3.

/// Maximum cell count for a general closed set.
/// Larger sets are too broad to fire often enough to be useful.
const MAX_CLOSED_SET_SIZE: usize = 8;

/// Append general closed sets to an existing `sets`/`membership` pair
/// (built by Phase 2).  Called once per level load.
fn compute_general_closed_sets(
    walls: &BitPlane,
    goals: &BitPlane,
    dead: &BitPlane,
    width: u8,
    height: u8,
    sets: &mut Vec<ClosedEdgeSet>,
    membership: &mut Vec<Vec<u32>>,
) {
    // Dedup within Phase 3; Phase 2 duplicates are accepted as harmless.
    let mut seen: HashSet<Vec<(u8, u8)>> = HashSet::new();

    for y in 0..height {
        for x in 0..width {
            if walls.get(x, y) || dead.get(x, y) { continue; }

            let Some(mut s) = find_min_closure(x, y, walls, dead, width, height) else { continue };

            s.sort_unstable(); // canonical form for dedup

            if seen.contains(&s) { continue; }

            let goal_count = s.iter().filter(|&&(cx, cy)| goals.get(cx, cy)).count() as u32;

            // Always track in `seen` even if not a deadlock set (avoids
            // re-examining the same closure from a different seed).
            seen.insert(s.clone());

            if goal_count < s.len() as u32 {
                let set_idx = sets.len() as u32;
                for &(cx, cy) in &s {
                    membership[cy as usize * width as usize + cx as usize].push(set_idx);
                }
                sets.push(ClosedEdgeSet { cells: s, goal_count });
            }
        }
    }
}

/// Compute the minimum closed superset of `(sx, sy)` via forced expansion.
///
/// For every cell `p` currently in the set and every direction `d`:
///   * `dest = p + d`   — where the box would land
///   * `from = p - d`   — where the player must stand to make the push
///
/// If `dest` is a valid, box-occupiable floor cell outside the set, *and*
/// the player could stand at `from` (not OOB, not a wall), then `dest` must
/// be added to the set — otherwise a box at `p` could escape via this push.
///
/// Returns `None` when the set grows beyond `MAX_CLOSED_SET_SIZE`, meaning
/// the region is too open to be a useful deadlock constraint.
fn find_min_closure(
    sx: u8, sy: u8,
    walls: &BitPlane,
    dead: &BitPlane,
    width: u8, height: u8,
) -> Option<Vec<(u8, u8)>> {
    // Small Vec is faster than a HashSet for n ≤ MAX_CLOSED_SET_SIZE.
    let mut cells: Vec<(u8, u8)> = vec![(sx, sy)];
    let mut head = 0usize; // BFS frontier pointer

    while head < cells.len() {
        let (px, py) = cells[head];
        head += 1;

        for &(dx, dy) in &DIRS {
            let dest_x = px as i8 + dx;
            let dest_y = py as i8 + dy;
            let from_x = px as i8 - dx;
            let from_y = py as i8 - dy;

            // Destination must be an in-bounds, box-occupiable floor cell.
            if !in_bounds(width, height, dest_x, dest_y) { continue; }
            let (dest_x, dest_y) = (dest_x as u8, dest_y as u8);
            if walls.get(dest_x, dest_y) || dead.get(dest_x, dest_y) { continue; }

            // Already in S — box stays inside, no escape.
            if cells.contains(&(dest_x, dest_y)) { continue; }

            // Player-from must be reachable (not OOB, not a wall).
            // Note: players CAN stand on dead squares — dead restricts box
            // placement only, so no `dead.get` check here.
            if !in_bounds(width, height, from_x, from_y) { continue; }
            if walls.get(from_x as u8, from_y as u8) { continue; }

            // Forced expansion: this is an unclosed escape route; pull dest in.
            cells.push((dest_x, dest_y));
            if cells.len() > MAX_CLOSED_SET_SIZE { return None; }
        }
    }

    Some(cells)
}

// ── Goal distance precomputation ───────────────────────────────────────────

fn precompute_goal_distances(
    walls: &BitPlane, goals: &BitPlane,
    width: u8, height: u8,
) -> Vec<Vec<u32>> {
    let cells = width as usize * height as usize;
    goals.iter_set_bits().map(|(gx, gy)| {
        let mut dist = vec![u32::MAX; cells];
        let mut queue = VecDeque::new();
        let gi = gy as usize * width as usize + gx as usize;
        dist[gi] = 0;
        queue.push_back((gx, gy));

        while let Some((x, y)) = queue.pop_front() {
            let d = dist[y as usize * width as usize + x as usize];
            for (dx, dy) in DIRS {
                // Reverse push: box was at adjacent cell, player was opposite
                let bx = x as i8 + dx;
                let by = y as i8 + dy;
                let px = x as i8 - dx;
                let py = y as i8 - dy;
                if !in_bounds(width, height, bx, by) { continue; }
                if !in_bounds(width, height, px, py) { continue; }
                let (bx, by) = (bx as u8, by as u8);
                let (px, py) = (px as u8, py as u8);
                if walls.get(bx, by) || walls.get(px, py) { continue; }
                let bi = by as usize * width as usize + bx as usize;
                if dist[bi] == u32::MAX {
                    dist[bi] = d + 1;
                    queue.push_back((bx, by));
                }
            }
        }
        dist
    }).collect()
}

// ── Move reconstruction ────────────────────────────────────────────────────

/// BFS from (sx, sy) to (tx, ty) avoiding walls and boxes.
/// Returns the sequence of (dx, dy) steps taken, or an empty vec if already there
/// or unreachable (shouldn't be unreachable when called correctly).
fn bfs_path(
    sx: u8, sy: u8, tx: u8, ty: u8,
    walls: &BitPlane, boxes: &BitPlane,
    width: u8, height: u8,
) -> Vec<(i8, i8)> {
    if sx == tx && sy == ty { return vec![]; }

    let size = width as usize * height as usize;
    let si = sy as usize * width as usize + sx as usize;
    let ti = ty as usize * width as usize + tx as usize;

    // parent[i] = flat index of the cell we came from; usize::MAX = unvisited.
    let mut parent = vec![usize::MAX; size];
    let mut from_dir: Vec<(i8, i8)> = vec![(0, 0); size];
    parent[si] = si;

    let mut queue = VecDeque::new();
    queue.push_back((sx, sy));

    'bfs: while let Some((x, y)) = queue.pop_front() {
        let ci = y as usize * width as usize + x as usize;
        for &(dx, dy) in &DIRS {
            let nx = x as i8 + dx;
            let ny = y as i8 + dy;
            if !in_bounds(width, height, nx, ny) { continue; }
            let (nx, ny) = (nx as u8, ny as u8);
            let ni = ny as usize * width as usize + nx as usize;
            if parent[ni] != usize::MAX { continue; }
            if walls.get(nx, ny) || boxes.get(nx, ny) { continue; }
            parent[ni] = ci;
            from_dir[ni] = (dx, dy);
            if ni == ti { break 'bfs; }
            queue.push_back((nx, ny));
        }
    }

    if parent[ti] == usize::MAX { return vec![]; }

    // Walk back from target to source and reverse.
    let mut path = Vec::new();
    let mut cur = ti;
    while cur != si {
        path.push(from_dir[cur]);
        cur = parent[cur];
    }
    path.reverse();
    path
}

/// Reconstruct the full LURD move string from the push sequence.
/// Lowercase = player walk step, uppercase = box push.
fn reconstruct_moves(
    start_player: u16,
    initial_boxes: &BitPlane,
    path: &[Push],
    width: u8,
    walls: &BitPlane,
) -> String {
    let height = walls.height;
    let mut player_pos = start_player;
    let mut boxes = initial_boxes.clone();
    let mut moves = String::with_capacity(path.len() * 4);

    for push in path {
        // Where the player needs to stand to make this push.
        let target_x = (push.from_flat % width as usize) as u8;
        let target_y = (push.from_flat / width as usize) as u8;
        let px = (player_pos as usize % width as usize) as u8;
        let py = (player_pos as usize / width as usize) as u8;

        // Walk player from current position to the push cell.
        for (dx, dy) in bfs_path(px, py, target_x, target_y, walls, &boxes, width, height) {
            moves.push(match (dx, dy) {
                ( 0, -1) => 'u',
                ( 0,  1) => 'd',
                (-1,  0) => 'l',
                ( 1,  0) => 'r',
                _ => '?',
            });
        }

        // The push itself.  For a tunnel macro-push the box may have moved
        // multiple squares in one step, so we output one uppercase LURD
        // character per square travelled.
        let total_dx = push.to_x as i32 - push.from_x as i32;
        let total_dy = push.to_y as i32 - push.from_y as i32;
        let dx_unit  = total_dx.signum();
        let dy_unit  = total_dy.signum();
        let steps    = (total_dx.abs() + total_dy.abs()) as usize;
        let push_ch  = match (dx_unit, dy_unit) {
            ( 0, -1) => 'U',
            ( 0,  1) => 'D',
            (-1,  0) => 'L',
            ( 1,  0) => 'R',
            _ => '?',
        };
        for _ in 0..steps { moves.push(push_ch); }

        // After the push: box at (to_x, to_y); player is one step behind the
        // box's final position (they followed the box through the tunnel).
        boxes.clear(push.from_x, push.from_y);
        boxes.set(push.to_x, push.to_y);
        let final_px = (push.to_x as i32 - dx_unit) as u16;
        let final_py = (push.to_y as i32 - dy_unit) as u16;
        player_pos = final_py * width as u16 + final_px;
    }

    moves
}

// ── Freeze deadlock detection ──────────────────────────────────────────────
//
// A box is frozen if it cannot be moved: blocked along both the horizontal
// and the vertical axis simultaneously.  A box is blocked along an axis when
// *either* neighbour along that axis is a wall (which prevents both push
// directions on that axis — a wall on one side blocks the player from
// reaching the other side).  The check recurses through chains of adjacent
// boxes; cycles are treated as frozen (mutually-blocking groups are frozen).
//
// This is the Rust equivalent of YASS's `IsAFreezingMove` / `BoxIsBlockedAlongOneAxis`.

/// Returns `true` if placing a box at `(to_x, to_y)` creates a frozen group
/// that contains at least one box not sitting on a goal.
///
/// `boxes` must already reflect the push (box at `to_x,to_y`).
fn is_freeze_deadlock(
    to_x: u8, to_y: u8,
    boxes: &BitPlane,
    walls: &BitPlane,
    goals: &BitPlane,
    dead: &BitPlane,
) -> bool {
    let w = walls.width;
    let h = walls.height;
    let cells = w as usize * h as usize;

    // Per-cell state:  0 = unchecked  1 = in-progress  2 = not-blocked  3 = blocked
    let mut ch = vec![0u8; cells]; // horizontal axis cache
    let mut cv = vec![0u8; cells]; // vertical axis cache

    if !blocked_axis(to_x, to_y, true,  boxes, walls, dead, w, h, &mut ch) { return false; }
    if !blocked_axis(to_x, to_y, false, boxes, walls, dead, w, h, &mut cv) { return false; }

    // There must be at least one non-goal box in the frozen group, otherwise
    // all frozen boxes are already on their goals and that is fine.
    for flat in 0..cells {
        if ch[flat] == 3 && cv[flat] == 3 {
            let bx = (flat % w as usize) as u8;
            let by = (flat / w as usize) as u8;
            if boxes.get(bx, by) && !goals.get(bx, by) {
                return true;
            }
        }
    }
    false
}

/// Returns `true` if the box at `(x, y)` cannot escape along the given axis.
///
/// `horizontal = true`  → checks the left/right axis.
/// `horizontal = false` → checks the up/down axis.
///
/// A box is blocked along an axis when any of the following hold:
///   - A wall (or OOB) exists on either side along that axis.
///   - Both neighbours along that axis are dead squares (YASS `FLAG_ILLEGAL_BOX_SQUARE`):
///     pushing left would land on a dead square; same for right — so there is no
///     useful push in either direction.
///   - A neighbouring box exists that is itself blocked along the same axis
///     (biaxial chain recursion).
///
/// The in-progress marker (1) breaks cycles: if A depends on B and B on A,
/// both are treated as frozen, which is correct — they mutually block each other.
fn blocked_axis(
    x: u8, y: u8,
    horizontal: bool,
    boxes: &BitPlane,
    walls: &BitPlane,
    dead: &BitPlane,
    width: u8, height: u8,
    cache: &mut [u8],
) -> bool {
    let idx = y as usize * width as usize + x as usize;
    match cache[idx] {
        3 => return true,
        2 => return false,
        1 => return true, // in-progress → cycle → treat as frozen
        _ => {}
    }
    cache[idx] = 1; // mark in-progress

    let result = if horizontal {
        let left_wall  = x == 0         || walls.get(x - 1, y);
        let right_wall = x + 1 >= width || walls.get(x + 1, y);

        // YASS FLAG_ILLEGAL_BOX_SQUARE: if both neighbours are dead squares, there
        // is no useful push in either direction along this axis.
        // Guard: `!left_wall && !right_wall` also guarantees x >= 1 and x+1 < width,
        // so the dead.get() calls are safe.
        let both_dead = !left_wall && !right_wall
            && dead.get(x - 1, y) && dead.get(x + 1, y);

        // Only check for a frozen adjacent box when there is no wall on that side
        // (a wall already blocks that direction; no need to recurse).
        let left_frozen = !left_wall
            && boxes.get(x - 1, y)
            && blocked_axis(x - 1, y, true, boxes, walls, dead, width, height, cache);
        let right_frozen = !right_wall
            && boxes.get(x + 1, y)
            && blocked_axis(x + 1, y, true, boxes, walls, dead, width, height, cache);

        left_wall || right_wall || both_dead || left_frozen || right_frozen
    } else {
        let up_wall   = y == 0          || walls.get(x, y - 1);
        let down_wall = y + 1 >= height || walls.get(x, y + 1);

        let both_dead = !up_wall && !down_wall
            && dead.get(x, y - 1) && dead.get(x, y + 1);

        let up_frozen = !up_wall
            && boxes.get(x, y - 1)
            && blocked_axis(x, y - 1, false, boxes, walls, dead, width, height, cache);
        let down_frozen = !down_wall
            && boxes.get(x, y + 1)
            && blocked_axis(x, y + 1, false, boxes, walls, dead, width, height, cache);

        up_wall || down_wall || both_dead || up_frozen || down_frozen
    };

    cache[idx] = if result { 3 } else { 2 };
    result
}
