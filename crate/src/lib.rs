mod bitplane;
mod fxhash;
mod deadlock_sets;
mod level;
mod solver;
mod url;

// Re-export everything wasm-bindgen marked public.
// The JS/TS consumer sees: SokobanLevel, Solver, SolverResult,
// level_to_fragment, fragment_to_level, fragment_metadata.
pub use level::SokobanLevel;
pub use solver::{Solver, SolverResult};
pub use url::{level_to_fragment, fragment_to_level, fragment_metadata};
