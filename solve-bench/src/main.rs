//! solve-bench
//!
//! Parses `normalised-levels.xsb`, runs every puzzle through the sokoban-utils
//! solver in parallel (Rayon), and writes the results to a CSV file.
//!
//! # Usage
//!
//! ```
//! cd sokoban-utils/solve-bench
//! cargo run --release
//!
//! # override defaults:
//! cargo run --release -- \
//!     --levels  ../normalised-levels.xsb \
//!     --output  solver-results.csv       \
//!     --max-nodes 500000
//! ```
//!
//! # How the parallelism works
//!
//! `SokobanLevel` and `Solver` are plain Rust structs with no shared state —
//! each puzzle is fully self-contained.  Rayon's `par_iter()` hands each
//! puzzle to a worker in its work-stealing thread pool; no locking required.
//!
//! # CSV columns
//!
//! | column         | description                                            |
//! |----------------|--------------------------------------------------------|
//! | index          | 1-based position in the file                           |
//! | title          | from the `Title:` metadata line                        |
//! | collection     | from the `Collection:` metadata line                   |
//! | author         | from the `Author:` metadata line                       |
//! | solved         | `true` / `false` / `error`                             |
//! | push_count     | number of box pushes in the solution (0 if unsolved)   |
//! | moves          | full LURD move string (lowercase=walk, uppercase=push) |

use std::path::PathBuf;
use std::time::Instant;

use rayon::prelude::*;
use sokoban_utils::{Solver, SokobanLevel};

// ── CLI args ────────────────────────────────────────────────────────────────

struct Args {
    levels:    PathBuf,
    output:    PathBuf,
    max_nodes: u32,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut args      = std::env::args().skip(1).peekable();
        let mut levels    = PathBuf::from("../sample-10.xsb");
        let mut output    = PathBuf::from("solver-results.csv");
        let mut max_nodes = 500_000u32;

        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--levels"    => levels    = args.next().ok_or("--levels needs a value")?.into(),
                "--output"    => output    = args.next().ok_or("--output needs a value")?.into(),
                "--max-nodes" => max_nodes = args.next()
                                    .ok_or("--max-nodes needs a value")?
                                    .parse()
                                    .map_err(|_| "--max-nodes must be a u32")?,
                other => return Err(format!("Unknown flag: {other}")),
            }
        }
        Ok(Args { levels, output, max_nodes })
    }
}

// ── Puzzle ──────────────────────────────────────────────────────────────────

#[derive(Debug)]
struct Puzzle {
    global_index: usize,
    title:        String,
    author:       String,
    collection:   String,
    /// Raw XSB grid rows only (no Title/Author/Collection lines).
    grid:         String,
}

/// Parse the normalised XSB file produced by `normalize_xsb.py`.
///
/// Format (one blank line between levels):
/// ```text
/// <grid rows>
/// Title: …
/// Author: …
/// Collection: …
/// ```
/// Section header blocks (all lines start with `;`) are skipped.
fn parse_normalised_xsb(content: &str) -> Vec<Puzzle> {
    let mut puzzles = Vec::new();
    let mut index   = 0usize;

    for block in content.split("\n\n") {
        let block = block.trim();
        if block.is_empty() || block.lines().all(|l| l.trim_start().starts_with(';')) {
            continue;
        }

        let mut grid_lines = Vec::new();
        let mut title      = String::new();
        let mut author     = String::new();
        let mut collection = String::new();

        for line in block.lines() {
            let t = line.trim();
            if let Some(v)      = t.strip_prefix("Title:")      { title      = v.trim().to_owned(); }
            else if let Some(v) = t.strip_prefix("Author:")     { author     = v.trim().to_owned(); }
            else if let Some(v) = t.strip_prefix("Collection:") { collection = v.trim().to_owned(); }
            else if !t.starts_with(';') {
                grid_lines.push(line); // keep leading spaces — they matter in XSB
            }
        }

        let grid = grid_lines.join("\n");
        if !grid.contains('#') {
            continue; // header-only or empty block
        }

        index += 1;
        puzzles.push(Puzzle { global_index: index, title, author, collection, grid });
    }

    puzzles
}

// ── Solve one puzzle ────────────────────────────────────────────────────────

struct SolveResult {
    solved:     bool,
    push_count: u32,
    moves:      String,
}

fn solve_puzzle(puzzle: &Puzzle, max_nodes: u32) -> Result<SolveResult, String> {
    // Parse the XSB grid into a SokobanLevel.
    // from_xsb returns Result<SokobanLevel, JsValue>; on native targets JsValue
    // is a wasm_bindgen shim — map its error to a plain String.
    let level = SokobanLevel::from_xsb(&puzzle.grid)
        .map_err(|e| format!("{e:?}"))?;

    // Build the solver (runs dead-square / goal-distance precomputation).
    let solver = Solver::new(&level);

    // Run A* up to max_nodes expanded states.
    let result = solver.solve(&level, max_nodes);

    Ok(SolveResult {
        solved:     result.solved(),
        push_count: result.push_count(),
        moves:      result.moves(),
    })
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() {
    let args = match Args::parse() {
        Ok(a)  => a,
        Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
    };

    // Load puzzles.
    println!("Loading puzzles from {:?}…", args.levels);
    let content = std::fs::read_to_string(&args.levels)
        .unwrap_or_else(|e| { eprintln!("Cannot read {:?}: {e}", args.levels); std::process::exit(1); });
    let puzzles = parse_normalised_xsb(&content);
    println!("  {} puzzles loaded", puzzles.len());

    // Solve all puzzles in parallel.
    // Rayon distributes work across a thread pool sized to the number of
    // logical CPUs.  Each closure is independent — no shared mutable state.
    println!("Solving {} puzzles in parallel (max_nodes={})…", puzzles.len(), args.max_nodes);
    let t0 = Instant::now();

    let results: Vec<(&Puzzle, Result<SolveResult, String>)> = puzzles
        .par_iter()
        .map(|puzzle| (puzzle, solve_puzzle(puzzle, args.max_nodes)))
        .collect();

    let elapsed  = t0.elapsed();
    let solved   = results.iter().filter(|(_, r)| r.as_ref().map(|r| r.solved).unwrap_or(false)).count();
    let errored  = results.iter().filter(|(_, r)| r.is_err()).count();
    let unsolved = results.len() - solved - errored;

    println!("  Done in {:.2?}", elapsed);
    println!(
        "  {} solved  /  {} unsolved (node limit)  /  {} error",
        solved, unsolved, errored
    );

    // Write CSV.
    println!("Writing CSV to {:?}…", args.output);
    let mut wtr = csv::Writer::from_path(&args.output)
        .unwrap_or_else(|e| { eprintln!("Cannot create {:?}: {e}", args.output); std::process::exit(1); });

    wtr.write_record(["index", "title", "collection", "author", "solved", "push_count", "moves"])
        .unwrap();

    for (puzzle, result) in &results {
        match result {
            Ok(r) => wtr.write_record([
                &puzzle.global_index.to_string(),
                &puzzle.title,
                &puzzle.collection,
                &puzzle.author,
                &r.solved.to_string(),
                &r.push_count.to_string(),
                &r.moves,
            ]).unwrap(),
            Err(e) => wtr.write_record([
                &puzzle.global_index.to_string(),
                &puzzle.title,
                &puzzle.collection,
                &puzzle.author,
                "error",
                "0",
                e.as_str(),
            ]).unwrap(),
        }
    }
    wtr.flush().unwrap();

    println!("Done. Results written to {:?}", args.output);
}
