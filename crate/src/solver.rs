use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::cmp::Reverse;
use wasm_bindgen::prelude::*;
use crate::bitplane::BitPlane;
use crate::deadlock_sets::{
    compute_closed_edge_sets, compute_general_closed_sets, DeadlockSetRegistry,
};
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
    /// Unified deadlock-set registry: closed-edge, general-closed, and
    /// (eventually) controller / freeze / diagonal-center / dynamic sets,
    /// indexed by cell for O(memberships) overflow checks on every push.
    deadlock_sets: DeadlockSetRegistry,
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

        // Build the unified deadlock-set registry.  Closed-edge first (fast
        // linear scan), then general-closed (forced-expansion closures of
        // L-/T-/irregular shapes).  Closed-edge sets are technically a subset
        // of what general-closed finds, but we keep the linear scan because
        // its sets are produced more cheaply and the overlap is harmless —
        // duplicate memberships only cost one extra check per push.
        let mut deadlock_sets = DeadlockSetRegistry::new(level.width, level.height);
        compute_closed_edge_sets(
            &mut deadlock_sets,
            &level.walls, &level.goals, &dead, level.width, level.height,
        );
        compute_general_closed_sets(
            &mut deadlock_sets,
            &level.walls, &level.goals, &dead, level.width, level.height,
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
        }
    }

    /// Return a human-readable diagnostic string about the precomputed deadlock sets.
    ///
    /// Useful for tuning: call this from TypeScript right after `Solver.new(level)`
    /// and log the result to the console.
    ///
    /// Reports:
    ///   - Total number of deadlock sets registered
    ///   - Max number of set memberships any single cell has accumulated
    ///   - Histogram of membership counts (how many cells have 0, 1, 2, … memberships)
    pub fn set_diagnostics(&self) -> String {
        self.deadlock_sets.diagnostics()
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

                // Prune: corral with deadlocked fence box.
                // Pass the player's position after the push (= old box cell).
                if self.corral_prune(push.from_flat as u16, &new_boxes) { continue; }

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

    /// A* heuristic: minimum-cost bipartite matching between boxes and goals.
    ///
    /// Uses the Hungarian algorithm (O(n³)) to find the optimal box→goal
    /// assignment.  This is strictly tighter than the greedy "sum of nearest
    /// goals" approach: the greedy sum treats each box independently and may
    /// implicitly assign two boxes to the same goal, whereas the matching
    /// enforces uniqueness.
    ///
    /// Both are admissible lower bounds — the greedy sum ≤ optimal matching
    /// cost ≤ true remaining cost — but the matching heuristic dominates the
    /// greedy one and therefore prunes more branches from the A* heap.
    fn heuristic(&self, boxes: &BitPlane) -> u32 {
        let box_flats: Vec<usize> = boxes
            .iter_set_bits()
            .map(|(x, y)| y as usize * self.width as usize + x as usize)
            .collect();

        let n_boxes = box_flats.len();
        let n_goals = self.goal_distances.len();

        if n_boxes == 0 { return 0; }

        hungarian_matching(n_boxes, n_goals, |bi, gi| {
            self.goal_distances[gi][box_flats[bi]]
        })
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

    /// Check whether placing a box at `(to_x, to_y)` causes any registered
    /// deadlock set to overflow its capacity.
    ///
    /// `boxes` must already include the new box at `(to_x, to_y)`.
    fn has_set_deadlock(&self, to_x: u8, to_y: u8, boxes: &BitPlane) -> bool {
        self.deadlock_sets.has_overflow_deadlock(to_x, to_y, boxes)
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

// Closed-edge and general-closed deadlock-set precomputations live in
// `crate::deadlock_sets`.  Kept out of here so the registry, the families,
// and (eventually) the dynamic discovery engine share one home.

// ── Bipartite matching heuristic ──────────────────────────────────────────
//
// The greedy "sum of nearest goals" heuristic picks the closest goal for each
// box independently.  Because two boxes can silently share the same goal in
// this estimate, the sum can lie strictly below the true push cost.  The
// optimal box→goal assignment (computed once per A* node) is always ≥ the
// greedy sum, so it is still an admissible lower bound while being strictly
// tighter — it prunes more of the A* search tree.
//
// Algorithm: classic potential-based shortest-path augmentation (Kuhn–Munkres /
// Jonker–Volgenant style), O(n²·m) with n workers and m jobs.  For the small
// n typical in Sokoban (≤ ~20 boxes) this is negligible per node.
//
// Correctness note: the algorithm maintains complementary slackness throughout
// augmentation, so the final assignment is provably optimal (exact minimum
// cost, not just ε-optimal).  Unreachable box→goal pairs are encoded as
// INFEASIBLE (u32::MAX / 2); if any box cannot reach any goal the heuristic
// returns INFEASIBLE, signalling a deadlock to the caller.

/// Solve the minimum-cost assignment problem.
///
/// Assigns each of the `n_workers` boxes to a **distinct** goal drawn from
/// `n_jobs` goals (`n_workers ≤ n_jobs`).  `cost_fn(worker, job)` returns the
/// push-distance; `u32::MAX` (or any value ≥ `u32::MAX / 2`) marks a pair as
/// infeasible (that goal is unreachable from that box position).
///
/// Returns the total cost of the optimal assignment, or `u32::MAX / 2` when
/// no perfect matching of boxes to distinct goals exists.
fn hungarian_matching(
    n_workers: usize,
    n_jobs: usize,
    cost_fn: impl Fn(usize, usize) -> u32,
) -> u32 {
    const INFEASIBLE: u32 = u32::MAX / 2;
    const INF: i64 = i64::MAX / 4;

    if n_workers == 0 { return 0; }
    if n_workers > n_jobs { return INFEASIBLE; }

    // Dual variables (potentials): u[i] for workers (1-indexed), v[j] for jobs.
    // These enforce complementary slackness throughout the augmentation loop.
    let mut u = vec![0i64; n_workers + 1];
    let mut v = vec![0i64; n_jobs + 1];

    // p[j] = worker currently assigned to job j  (0 = unassigned).
    // Index 0 is a virtual "free" job used as the augmentation source.
    let mut p = vec![0usize; n_jobs + 1];

    // way[j] = predecessor job on the shortest augmenting path reaching j.
    let mut way = vec![0usize; n_jobs + 1];

    for i in 1..=n_workers {
        // Augment for worker i: find the cheapest augmenting path starting from
        // the virtual source (job 0, which "holds" worker i) to any unmatched
        // real job, then flip the path to extend the matching by one edge.
        p[0] = i;
        let mut j0 = 0usize; // current job on the path

        // min_val[j]: cheapest reduced cost to reach job j from the current path.
        let mut min_val = vec![INF; n_jobs + 1];
        let mut used = vec![false; n_jobs + 1]; // jobs already on the path

        loop {
            used[j0] = true;
            let i0 = p[j0]; // the worker sitting at job j0
            let mut delta = INF;
            let mut j1 = 0usize; // next job to extend the path to

            for j in 1..=n_jobs {
                if !used[j] {
                    let raw = cost_fn(i0 - 1, j - 1);
                    let c: i64 = if raw >= INFEASIBLE { INF } else { raw as i64 };
                    // Reduced cost: actual_cost - worker_potential - job_potential.
                    let val = c - u[i0] - v[j];
                    if val < min_val[j] {
                        min_val[j] = val;
                        way[j] = j0;
                    }
                    if min_val[j] < delta {
                        delta = min_val[j];
                        j1 = j;
                    }
                }
            }

            if delta >= INF / 2 {
                return INFEASIBLE; // no perfect matching exists
            }

            // Update potentials to keep complementary slackness intact.
            for j in 0..=n_jobs {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    min_val[j] -= delta;
                }
            }

            j0 = j1;
            if p[j0] == 0 { break; } // j1 was unmatched — augmenting path found
        }

        // Flip the augmenting path to extend the matching.
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 { break; }
        }
    }

    // Sum the actual (un-reduced) costs of the optimal assignment.
    let mut total = 0u32;
    for j in 1..=n_jobs {
        if p[j] > 0 {
            let c = cost_fn(p[j] - 1, j - 1);
            if c >= INFEASIBLE { return INFEASIBLE; }
            total = total.saturating_add(c);
        }
    }
    total
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
    // Both caches are threaded through every recursive call because deciding
    // whether a *neighbour* box blocks us now requires knowing whether that
    // neighbour is frozen on BOTH axes (see `blocked_axis`).
    let mut ch = vec![0u8; cells]; // horizontal axis cache
    let mut cv = vec![0u8; cells]; // vertical axis cache

    if !blocked_axis(to_x, to_y, true,  boxes, walls, dead, w, h, &mut ch, &mut cv) { return false; }
    if !blocked_axis(to_x, to_y, false, boxes, walls, dead, w, h, &mut ch, &mut cv) { return false; }

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
///   - A neighbouring box exists that is itself **frozen** — i.e. blocked along
///     *both* axes.
///
/// ## Why the neighbour must be frozen on BOTH axes
///
/// An adjacent box only *permanently* blocks us if it can never get out of the
/// way.  A box that is blocked on this axis but free on the *other* axis can
/// simply be pushed aside along that other axis, after which we are free to
/// move.  Requiring only same-axis blocking (the previous behaviour) therefore
/// produced **false freeze deadlocks**: e.g. a box with a wall on one side and
/// such a one-axis-blocked neighbour on the other was wrongly reported frozen
/// even though the neighbour could slide away.  This is the standard YASS /
/// Sokoban-wiki freeze rule, and it is sound — it can only ever *miss* a
/// deadlock, never invent one.
///
/// The in-progress marker (1) breaks cycles: if A depends on B and B on A,
/// both are treated as frozen, which is correct — they mutually block each other.
/// Both axis caches (`ch`, `cv`) are threaded through so the two-axis neighbour
/// test can reuse partial results and share the cycle markers.
fn blocked_axis(
    x: u8, y: u8,
    horizontal: bool,
    boxes: &BitPlane,
    walls: &BitPlane,
    dead: &BitPlane,
    width: u8, height: u8,
    ch: &mut [u8],
    cv: &mut [u8],
) -> bool {
    let idx = y as usize * width as usize + x as usize;
    {
        let state = if horizontal { ch[idx] } else { cv[idx] };
        match state {
            3 => return true,
            2 => return false,
            1 => return true, // in-progress → cycle → treat as frozen
            _ => {}
        }
    }
    if horizontal { ch[idx] = 1 } else { cv[idx] = 1 } // mark in-progress

    // A neighbouring box blocks us only if it is fully frozen: blocked on both
    // the horizontal AND the vertical axis.  We evaluate the two axes with `&`
    // (not `&&`) so the recursion populates both caches, then combine.
    let result = if horizontal {
        let left_wall  = x == 0         || walls.get(x - 1, y);
        let right_wall = x + 1 >= width || walls.get(x + 1, y);

        // YASS FLAG_ILLEGAL_BOX_SQUARE: if both neighbours are dead squares, there
        // is no useful push in either direction along this axis.
        // Guard: `!left_wall && !right_wall` also guarantees x >= 1 and x+1 < width,
        // so the dead.get() calls are safe.
        let both_dead = !left_wall && !right_wall
            && dead.get(x - 1, y) && dead.get(x + 1, y);

        // Only recurse into an adjacent box when there is no wall on that side
        // (a wall already blocks that direction; no need to recurse).
        let left_frozen = !left_wall && boxes.get(x - 1, y) && {
            let h_blk = blocked_axis(x - 1, y, true,  boxes, walls, dead, width, height, ch, cv);
            let v_blk = blocked_axis(x - 1, y, false, boxes, walls, dead, width, height, ch, cv);
            h_blk && v_blk
        };
        let right_frozen = !right_wall && boxes.get(x + 1, y) && {
            let h_blk = blocked_axis(x + 1, y, true,  boxes, walls, dead, width, height, ch, cv);
            let v_blk = blocked_axis(x + 1, y, false, boxes, walls, dead, width, height, ch, cv);
            h_blk && v_blk
        };

        left_wall || right_wall || both_dead || left_frozen || right_frozen
    } else {
        let up_wall   = y == 0          || walls.get(x, y - 1);
        let down_wall = y + 1 >= height || walls.get(x, y + 1);

        let both_dead = !up_wall && !down_wall
            && dead.get(x, y - 1) && dead.get(x, y + 1);

        let up_frozen = !up_wall && boxes.get(x, y - 1) && {
            let h_blk = blocked_axis(x, y - 1, true,  boxes, walls, dead, width, height, ch, cv);
            let v_blk = blocked_axis(x, y - 1, false, boxes, walls, dead, width, height, ch, cv);
            h_blk && v_blk
        };
        let down_frozen = !down_wall && boxes.get(x, y + 1) && {
            let h_blk = blocked_axis(x, y + 1, true,  boxes, walls, dead, width, height, ch, cv);
            let v_blk = blocked_axis(x, y + 1, false, boxes, walls, dead, width, height, ch, cv);
            h_blk && v_blk
        };

        up_wall || down_wall || both_dead || up_frozen || down_frozen
    };

    if horizontal { ch[idx] = if result { 3 } else { 2 } }
    else          { cv[idx] = if result { 3 } else { 2 } }
    result
}

// ── Corral pruning ─────────────────────────────────────────────────────────
//
// A "corral" is a connected region of floor cells the player cannot reach in
// the current state, bounded by boxes (the "fence") and walls.  Because no
// fence box can be pushed outward — there is no player access on the outside
// of those boxes — any box that ends up inside the corral is effectively
// trapped there.
//
// Pruning criterion (conservative, no false positives):
//   1. Identify each connected pocket of floor cells unreachable by the player.
//   2. Safety guard: if the pocket contains a goal with no box on it, skip it —
//      a box might still need to be pushed in, so we cannot conclude deadlock.
//   3. Collect the "fence": every box adjacent to at least one pocket cell.
//   4. If any fence box is already deadlocked (freeze or closed-set overflow),
//      the state is a deadlock: the fence can never be removed.
//
// Reference: YASS `CorralPruning`, approximately line 17361.

impl Solver {
    /// Returns `true` if the position is provably a deadlock via corral analysis.
    ///
    /// `player_flat` — flat index of the player's position immediately after the
    ///                 push (`push.from_flat as u16` — the old box cell).
    /// `boxes`        — box bitplane already reflecting the completed push.
    fn corral_prune(&self, player_flat: u16, boxes: &BitPlane) -> bool {
        let px = (player_flat as usize % self.width as usize) as u8;
        let py = (player_flat as usize / self.width as usize) as u8;

        // Player's reachable floor region in the new state.
        let reachable = flood_fill(px, py, &self.walls, boxes, self.width, self.height);

        // Track interior cells already assigned to a corral pocket so we do
        // not re-seed from them and process the same pocket twice.
        let mut interior_seen = BitPlane::new(self.width, self.height);

        for y in 0..self.height {
            for x in 0..self.width {
                // Candidate seed: floor cell, not a box, not player-reachable,
                // not already part of a processed corral.
                if self.walls.get(x, y) { continue; }
                if boxes.get(x, y)      { continue; }
                if reachable.get(x, y)  { continue; }
                if interior_seen.get(x, y) { continue; }

                // Flood-fill from this seed (obstacles = walls + boxes) to find
                // the full corral pocket.
                let interior = flood_fill(x, y, &self.walls, boxes, self.width, self.height);

                // Mark all pocket cells so future seeds skip them.
                for (ix, iy) in interior.iter_set_bits() {
                    interior_seen.set(ix, iy);
                }

                // Safety guard: bare goal inside the pocket → skip.
                // We may still need to push a box there, so we cannot safely
                // declare a deadlock without deeper analysis.
                let has_bare_goal = interior
                    .iter_set_bits()
                    .any(|(ix, iy)| self.goals.get(ix, iy) && !boxes.get(ix, iy));
                if has_bare_goal { continue; }

                // Collect fence boxes: boxes adjacent to at least one pocket cell.
                let mut fence: Vec<(u8, u8)> = Vec::new();
                for (ix, iy) in interior.iter_set_bits() {
                    for &(dx, dy) in &DIRS {
                        let nx = ix as i8 + dx;
                        let ny = iy as i8 + dy;
                        if !in_bounds(self.width, self.height, nx, ny) { continue; }
                        let (nx, ny) = (nx as u8, ny as u8);
                        if boxes.get(nx, ny) && !fence.contains(&(nx, ny)) {
                            fence.push((nx, ny));
                        }
                    }
                }

                // If any fence box is already deadlocked, this state is unsolvable.
                for &(fx, fy) in &fence {
                    if self.has_set_deadlock(fx, fy, boxes) {
                        return true;
                    }
                    if is_freeze_deadlock(
                        fx, fy, boxes, &self.walls, &self.goals, &self.dead,
                    ) {
                        return true;
                    }
                }
            }
        }

        false
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── hungarian_matching ─────────────────────────────────────────────────

    #[test]
    fn hungarian_zero_workers() {
        assert_eq!(hungarian_matching(0, 0, |_, _| 0u32), 0);
        assert_eq!(hungarian_matching(0, 3, |_, _| 5u32), 0);
    }

    #[test]
    fn hungarian_single_pair() {
        assert_eq!(hungarian_matching(1, 1, |_, _| 7u32), 7);
    }

    #[test]
    fn hungarian_single_infeasible() {
        // The only goal is unreachable.
        assert_eq!(hungarian_matching(1, 1, |_, _| u32::MAX), u32::MAX / 2);
    }

    #[test]
    fn hungarian_more_goals_than_boxes() {
        // 1 box, 3 goals: cheapest is goal 1 (cost 2).
        let costs = [10u32, 2, 8];
        assert_eq!(hungarian_matching(1, 3, |_, gi| costs[gi]), 2);
    }

    /// Key correctness test: show that the matching gives the true optimum,
    /// which is HIGHER than the greedy "sum of individual minimums".
    ///
    /// Cost matrix:
    ///              goal 0   goal 1
    ///   box 0:       1       10
    ///   box 1:       2        3
    ///
    /// Greedy sum-of-min = min(1,10) + min(2,3) = 1 + 2 = 3
    ///
    /// Optimal matching:
    ///   box0→goal0 (1) + box1→goal1 (3) = 4   ← chosen
    ///   box0→goal1 (10) + box1→goal0 (2) = 12
    ///
    /// The matching heuristic returns 4, which is a strictly tighter lower
    /// bound than the greedy 3.
    #[test]
    fn hungarian_tighter_than_greedy() {
        let costs = [[1u32, 10], [2u32, 3]];
        let result = hungarian_matching(2, 2, |bi, gi| costs[bi][gi]);
        assert_eq!(result, 4);

        // Confirm greedy underestimates (it would return 3).
        let greedy: u32 = costs.iter().map(|row| *row.iter().min().unwrap()).sum();
        assert_eq!(greedy, 3);
        assert!(result > greedy);
    }

    #[test]
    fn hungarian_3x3_known_optimum() {
        // Cost matrix — exhaustive enumeration shown below.
        //          goal 0  goal 1  goal 2
        // box 0:     3       4       5
        // box 1:     7       8       1
        // box 2:     2       6       9
        //
        // All 3! = 6 perfect matchings:
        //   0→0, 1→1, 2→2:  3+8+9 = 20
        //   0→0, 1→2, 2→1:  3+1+6 = 10
        //   0→1, 1→0, 2→2:  4+7+9 = 20
        //   0→1, 1→2, 2→0:  4+1+2 =  7  ← optimal
        //   0→2, 1→0, 2→1:  5+7+6 = 18
        //   0→2, 1→1, 2→0:  5+8+2 = 15
        let costs = [[3u32, 4, 5], [7u32, 8, 1], [2u32, 6, 9]];
        assert_eq!(hungarian_matching(3, 3, |bi, gi| costs[bi][gi]), 7);
    }

    #[test]
    fn hungarian_identity_matrix() {
        // n×n identity: optimal is the diagonal, cost = 0 * n.
        // Use costs: diagonal = 0, off-diagonal = 100.
        let n = 5usize;
        let result = hungarian_matching(n, n, |bi, gi| if bi == gi { 0 } else { 100 });
        assert_eq!(result, 0);
    }

    #[test]
    fn hungarian_all_equal() {
        // All costs equal → any matching is optimal; total = n * cost.
        let result = hungarian_matching(4, 4, |_, _| 6u32);
        assert_eq!(result, 24);
    }

    #[test]
    fn hungarian_workers_exceed_jobs_infeasible() {
        assert_eq!(hungarian_matching(3, 2, |_, _| 1u32), u32::MAX / 2);
    }

    // ── corral_prune ──────────────────────────────────────────────────────

    /// A 3-row horizontal corridor: player is behind a box, free space beyond.
    ///
    /// ```text
    /// #######
    /// #@$   #
    /// #######
    /// ```
    ///
    /// The space to the right of the box forms a corral (no goals, no bare-goal
    /// guard fires).  The fence is the single box at (2,1).  That box sits in a
    /// closed-edge set covering cells (2,1)–(4,1) with 0 goals, so
    /// `has_set_deadlock` returns true → `corral_prune` must return `true`.
    #[test]
    fn corral_prune_detects_deadlock() {
        let level = SokobanLevel::from_xsb(
            "#######\n#@$   #\n#######",
        ).expect("parse");
        let solver = Solver::new(&level);
        assert!(
            solver.corral_prune(level.player_pos, &level.boxes),
            "expected deadlocked corral to be pruned",
        );
    }

    /// Same topology but with a goal at (3,1) inside the corral — no box on it.
    ///
    /// ```text
    /// #######
    /// #@$.  #
    /// #######
    /// ```
    ///
    /// The bare-goal safety guard must fire and skip pruning this corral, so
    /// `corral_prune` must return `false` (no false prune).
    #[test]
    fn corral_prune_skips_bare_goal() {
        let level = SokobanLevel::from_xsb(
            "#######\n#@$.  #\n#######",
        ).expect("parse");
        let solver = Solver::new(&level);
        assert!(
            !solver.corral_prune(level.player_pos, &level.boxes),
            "expected bare-goal corral to be left unpruned",
        );
    }

    // ── is_freeze_deadlock ─────────────────────────────────────────────────
    //
    // These guard the freeze check directly.  It had no unit coverage before,
    // which is how a latent unsoundness (treating a box as blocked along an
    // axis when an adjacent box was blocked only on that *same* axis) survived
    // unnoticed — it only mattered once `corral_prune` started seeding the
    // check at fence boxes rather than just the freshly pushed box.

    /// Helper: run the freeze check using a solver's precomputed planes.
    fn freeze(solver: &Solver, boxes: &BitPlane, x: u8, y: u8) -> bool {
        is_freeze_deadlock(x, y, boxes, &solver.walls, &solver.goals, &solver.dead)
    }

    /// A box wedged into a corner (wall left + wall above), not on a goal, is a
    /// genuine freeze deadlock.
    ///
    /// ```text
    /// ####
    /// #$ #
    /// #  #
    /// ####
    /// ```
    #[test]
    fn freeze_corner_is_deadlock() {
        let level = SokobanLevel::from_xsb("####\n#$ #\n#  #\n####").expect("parse");
        let solver = Solver::new(&level);
        assert!(freeze(&solver, &level.boxes, 1, 1),
            "a non-goal box in a wall corner must be a freeze deadlock");
    }

    /// The same corner, but the box is already on a goal (`*`).  A frozen group
    /// whose every box sits on a goal is not a deadlock.
    ///
    /// ```text
    /// ####
    /// #* #
    /// #  #
    /// ####
    /// ```
    #[test]
    fn freeze_corner_on_goal_is_ok() {
        let level = SokobanLevel::from_xsb("####\n#* #\n#  #\n####").expect("parse");
        let solver = Solver::new(&level);
        assert!(!freeze(&solver, &level.boxes, 1, 1),
            "a frozen box that sits on a goal must NOT be reported as a deadlock");
    }

    /// Recursive (chain) freeze: box B at (2,1) has open floor to its right, so
    /// it is only blocked horizontally *because* its left neighbour A at (1,1)
    /// is itself fully frozen (wedged against walls on all relevant sides).
    /// The two-axis neighbour recursion must still detect this real deadlock.
    ///
    /// ```text
    /// #####
    /// #$$ #
    /// #####
    /// ```
    #[test]
    fn freeze_recursive_pair_is_deadlock() {
        let level = SokobanLevel::from_xsb("#####\n#$$ #\n#####").expect("parse");
        let solver = Solver::new(&level);
        assert!(freeze(&solver, &level.boxes, 2, 1),
            "box blocked via a fully-frozen neighbour must be a freeze deadlock");
    }

    /// Regression guard for the corral false positive (mirrors the level-15
    /// configuration that broke the solver).  Box B at (3,2) has a wall on its
    /// right and box A at (3,1) directly above it.  A has a wall above but is
    /// free to slide left/right — so A can vacate and B can later be pushed
    /// down.  B is therefore NOT frozen.
    ///
    /// The previous freeze check wrongly reported B as frozen because it
    /// considered A "vertically blocked" (wall above A) without checking that
    /// A could still escape horizontally.  The fix requires a neighbour box to
    /// be frozen on BOTH axes before it counts as a blocker.
    ///
    /// ```text
    /// ######
    /// #. $ #   A = box at (3,1)
    /// #. $##   B = box at (3,2), wall at (4,2)
    /// #  @ #
    /// ######
    /// ```
    #[test]
    fn freeze_not_fooled_by_slidable_neighbour() {
        let level = SokobanLevel::from_xsb(
            "######\n#. $ #\n#. $##\n#  @ #\n######",
        ).expect("parse");
        let solver = Solver::new(&level);
        // Sanity: the slidable neighbour A itself is not frozen.
        assert!(!freeze(&solver, &level.boxes, 3, 1),
            "neighbour A can slide horizontally, so it is not frozen");
        // The actual regression guard: B must not be a false freeze deadlock.
        assert!(!freeze(&solver, &level.boxes, 3, 2),
            "B must not be reported frozen — its neighbour can slide away");
    }
}
