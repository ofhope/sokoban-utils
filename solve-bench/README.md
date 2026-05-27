

# Step 1 — 10 puzzles, one from each collection (now the default)
cargo run --release

# Step 2 — 60 medium puzzles
cargo run --release -- --levels ../medium-levels-60.xsb --output results-60.csv

# Step 3 — 100 puzzles, still all small/medium grids
cargo run --release -- --levels ../sample-100.xsb --output results-100.csv

# Step 4 — everything, once you're confident in the node limit
cargo run --release -- --levels ../normalised-levels.xsb --output results-full.csv --max-nodes 100000