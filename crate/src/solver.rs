use std::collections::{BinaryHeap, HashMap, VecDeque};
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
}

#[wasm_bindgen]
impl Solver {
    /// Build a solver from a level. Precomputation happens here (once per level load).
    pub fn new(level: &SokobanLevel) -> Solver {
        let dead = compute_dead_squares(&level.walls, &level.goals, level.width, level.height);
        let goal_distances = precompute_goal_distances(
            &level.walls, &level.goals, level.width, level.height,
        );
        Solver {
            width: level.width,
            height: level.height,
            walls: level.walls.clone(),
            goals: level.goals.clone(),
            dead,
            goal_distances,
        }
    }

    /// Run A* over push states. Returns None (unsolvable or limit hit) via solved=false.
    pub fn solve(&self, level: &SokobanLevel, max_nodes: u32) -> SolverResult {
        let initial_boxes = level.boxes.clone();
        let initial_player = self.normalise_player(
            level.player_pos, &initial_boxes,
        );

        let initial_state = State { boxes: initial_boxes, player_norm: initial_player };

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

                // Prune: 2×2 freeze deadlock
                if is_2x2_deadlock(&new_boxes, &self.walls, &self.goals, push.to_x, push.to_y) {
                    continue;
                }

                let new_player_norm = self.normalise_player(
                    push.from_flat as u16, &new_boxes,
                );

                let new_state = State { boxes: new_boxes, player_norm: new_player_norm };
                let g = node.cost + 1;
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

    /// Generate all valid pushes from a search state.
    fn generate_pushes(&self, state: &State) -> Vec<Push> {
        let px = (state.player_norm as usize % self.width as usize) as u8;
        let py = (state.player_norm as usize / self.width as usize) as u8;
        let reachable = flood_fill(px, py, &self.walls, &state.boxes, self.width, self.height);
        let mut pushes = Vec::new();

        for (bx, by) in state.boxes.iter_set_bits() {
            for (dx, dy) in DIRS {
                let push_from_x = bx as i8 - dx;
                let push_from_y = by as i8 - dy;
                let push_to_x   = bx as i8 + dx;
                let push_to_y   = by as i8 + dy;

                if !in_bounds(self.width, self.height, push_from_x, push_from_y) { continue; }
                if !in_bounds(self.width, self.height, push_to_x,   push_to_y)   { continue; }

                let (pfx, pfy) = (push_from_x as u8, push_from_y as u8);
                let (ptx, pty) = (push_to_x   as u8, push_to_y   as u8);

                if !reachable.get(pfx, pfy)      { continue; }
                if self.walls.get(ptx, pty)       { continue; }
                if state.boxes.get(ptx, pty)      { continue; }

                let from_flat = pfy as usize * self.width as usize + pfx as usize;
                pushes.push(Push { from_x: bx, from_y: by, to_x: ptx, to_y: pty, from_flat });
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

        // The push itself — direction is box movement (= player movement direction).
        let dx = push.to_x as i8 - push.from_x as i8;
        let dy = push.to_y as i8 - push.from_y as i8;
        moves.push(match (dx, dy) {
            ( 0, -1) => 'U',
            ( 0,  1) => 'D',
            (-1,  0) => 'L',
            ( 1,  0) => 'R',
            _ => '?',
        });

        // After the push: box moves to (to_x, to_y), player stands at old box cell.
        boxes.clear(push.from_x, push.from_y);
        boxes.set(push.to_x, push.to_y);
        player_pos = push.from_y as u16 * width as u16 + push.from_x as u16;
    }

    moves
}

// ── 2×2 deadlock detection ─────────────────────────────────────────────────

fn is_2x2_deadlock(
    boxes: &BitPlane, walls: &BitPlane, goals: &BitPlane,
    bx: u8, by: u8,
) -> bool {
    for (ox, oy) in [(0u8, 0u8), (1, 0), (0, 1), (1, 1)] {
        let cx = bx.saturating_sub(ox);
        let cy = by.saturating_sub(oy);
        let w = boxes.width;
        let h = boxes.height;
        if cx + 1 >= w || cy + 1 >= h { continue; }
        let all_blocked = (0..2u8).all(|i| (0..2u8).all(|j| {
            let nx = cx + i;
            let ny = cy + j;
            walls.get(nx, ny) || boxes.get(nx, ny)
        }));
        if all_blocked {
            let all_goals = (0..2u8).all(|i| (0..2u8).all(|j| {
                goals.get(cx + i, cy + j)
            }));
            if !all_goals { return true; }
        }
    }
    false
}
