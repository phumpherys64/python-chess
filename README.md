# python-chess playground (Stockfish + Syzygy)

CLI tools to explore fun ideas with python-chess:
- Engine multi-PV analysis
- Blunder detection from PGN
- Best-move SVG arrow
- Endgame coach with Syzygy tablebases (if available), engine fallback otherwise
- Move heatmap (colored arrows by strength)
- Eval charts with blunder markers
- Annotate PGNs with evals, blunder tags, and TB info
- Simple Tk GUI to step through endgames with TB hints
  - Click-to-move, arrow keys (←/→), 'a' to analyse, 't' to probe TB
  - Status line shows TB WDL/DTZ/DTM if available, else engine eval
  - Flip board and adjust engine depth via GUI controls
  - Optional continuous analysis with independent auto-depth slider
  - Session resume: saves last position, orientation, depth, auto settings, and window size
  - Export current board to SVG/PNG from the GUI
  - Copy FEN and PV (SAN/UCI) to clipboard; insert current engine PV into a PGN (searches variations for matching position)
  - Controls for PV length when copying/inserting; option to replace previously inserted engine PVs; toggle to search only mainline
  - Tag field used to label inserted PVs; GUI can also remove all tagged PVs across a PGN
  
- Lichess integrations
  - Tablebase (cloud WDL/DTZ/DTM)
  - Cloud eval (Stockfish PVs)
  - Opening Explorer (Masters / Lichess DB)
  - Export user games (PGN/NDJSON)
  - Optional cache with TTL for explorer/cloud-eval/tablebase

## Setup

- Install dependencies:
  pip install -r requirements.txt

- Ensure Stockfish is installed and on your PATH, or set `STOCKFISH` env var to its path.
  - Common macOS path: `/opt/homebrew/bin/stockfish`
- Optional: point to local Syzygy tablebases via `SYZYGY_PATH` or pass `--tb-path` to the endgame coach.

### Quick Start

- Install Stockfish (macOS): `brew install stockfish`
- Install deps: `pip install -r requirements.txt`
- (Optional) Create `.chessrc.json` with your engine/TB/Lichess defaults (see below)
- Try it: `python -m scripts analyse --depth 12 --multipv 3`

## Usage

Call the CLI via module run:

  python -m scripts --help

Examples:

- Analyse current start position:

  python -m scripts analyse --depth 16 --multipv 3 --pv-len 6

- From a FEN:

  python -m scripts analyse --fen "8/8/8/8/8/8/5K2/6k1 w - - 0 1"

- Blunders from PGN (first game):

  python -m scripts blunders --pgn my_game.pgn --threshold 150 --depth 14

- Best-move SVG from a short line of moves (SAN or UCI):

  python -m scripts bestmove-svg --moves "e4 e5 Nf3 Nc6 Bb5" --out best.svg --png-out best.png  # optional PNG

- Endgame coach with tablebases (if you have them):

  export SYZYGY_PATH=/path/to/syzygy
  python -m scripts endgame-coach --fen "8/8/8/8/8/8/5K2/6k1 w - - 0 1" --top 5

If no tablebases are found or the position is unsupported, the coach falls back to the engine and shows multi-PV suggestions.

- Auto-detect endgame from a PGN and coach it:

  python -m scripts endgame-from-pgn --pgn my_game.pgn --top 5

- Annotate a PGN with evals, blunder NAGs, and TB info:

  python -m scripts annotate-pgn --input in.pgn --output out_annotated.pgn --depth 14 --threshold 150 --big-threshold 300 --tb-path "$SYZYGY_PATH"
  # Add engine PVs as variations only for blunders (default), or for all moves
  python -m scripts annotate-pgn --input in.pgn --output out_var.pgn --variations blunders --max-pv-len 6
  python -m scripts annotate-pgn --input in.pgn --output out_allvar.pgn --variations all --max-pv-len 6

- Simple GUI to step through a PGN or FEN with TB hints and engine suggestions (requires Tk):

  python -m scripts endgame-gui --pgn my_game.pgn --tb-path "$SYZYGY_PATH"
  # or
  python -m scripts endgame-gui --fen "8/8/8/8/8/8/5K2/6k1 w - - 0 1" --tb-path "$SYZYGY_PATH"

- Batch TB scan to tag first TB-covered position per game and emit a CSV:

  python -m scripts tb-scan --input many_games.pgn --output tagged.pgn --tb-path "$SYZYGY_PATH" --csv-out tb_index.csv

- Move heatmap (evaluate all legal moves and color arrows):

  python -m scripts move-heatmap-svg --moves "e4 e5 Nf3" --out heatmap.svg --png-out heatmap.png

- Eval chart with blunder markers (PNG), and optional CSV:

  python -m scripts eval-chart --pgn my_game.pgn --depth 12 --threshold 150 --out eval.png --csv-out eval.csv

- Lichess tablebase for position:

  python -m scripts lichess-tablebase --fen "8/8/8/8/8/8/5K2/6k1 w - - 0 1" --top 6 --cache-dir .cache/lichess --cache-ttl 3600

- Lichess cloud eval (multi-PV):

  python -m scripts lichess-cloud-eval --moves "e4 e5 Nf3 Nc6 Bb5" --multi-pv 3 --cache-dir .cache/lichess --cache-ttl 600

- Lichess opening explorer (Masters DB):

  python -m scripts lichess-explorer --db masters --moves "e4 e5 Nf3" --top-games 5 --cache-dir .cache/lichess --cache-ttl 86400

- Export Lichess user games to PGN (token optional):

  export LICHESS_TOKEN=\<your_token\>
  python -m scripts lichess-export --user YOUR_NAME --out your_games.pgn --max 200 --perf classical --rated true

## Command Index

- `analyse`: Multi-PV engine suggestions for a position
- `blunders`: Detect blunders in a PGN by eval drops
- `bestmove-svg`: Render best move arrow to SVG/PNG
- `endgame-coach`: Syzygy or engine (optional Lichess fallbacks)
- `endgame-from-pgn`: Detect endgame segment and coach it
- `endgame-gui`: Tk GUI with TB hints, analysis, export tools
- `move-heatmap-svg`: Evaluate all legal moves and color arrows
- `eval-chart`: Plot eval over game with blunder markers
- `annotate-pgn`: Add evals/NAGs and engine lines (tagged)
- `pgn-insert-pv`: Insert a chosen PV into PGN at a FEN/ply
- `tb-scan`: Tag first TB-covered ply and emit CSV
- `lichess-tablebase`: Query cloud TB for WDL/DTZ/DTM
- `lichess-cloud-eval`: Query cloud Stockfish PVs
- `lichess-explorer`: Opening stats (Masters/Lichess DB)
- `lichess-export`: Export user games to PGN/NDJSON
- `lichess-export-annotate`: Export + annotate + (optional) blunders CSV
- `lichess-cache clear|stats|fetch`: Manage and prewarm cache

- Export and annotate a user's games in one go:

  python -m scripts lichess-export-annotate --user YOUR_NAME --out annotated.pgn --max 50 \
    --variations blunders --max-pv-len 4 --depth 12 --threshold 150 --big-threshold 300
  # Also emit a blunders CSV with FENs and game IDs
  python -m scripts lichess-export-annotate --user YOUR_NAME --out annotated.pgn --max 50 \
    --blunders-csv blunders.csv --variations blunders --max-pv-len 4 --depth 12 --threshold 150 --big-threshold 300
  # Add a cloud-eval score column for each blunder position (uses cache defaults)
  python -m scripts lichess-export-annotate --user YOUR_NAME --out annotated.pgn --max 50 \
    --blunders-csv blunders.csv --cloud-eval-blunders
  # Customize which PGN headers appear in the CSV context columns
  python -m scripts lichess-export-annotate --user YOUR_NAME --out annotated.pgn \
    --blunders-csv blunders.csv --csv-headers "White,Black,ECO,Result,Event,Date"
  # If ECO header is missing, fill via Lichess Explorer
  python -m scripts lichess-export-annotate --user YOUR_NAME --out annotated.pgn \
    --blunders-csv blunders.csv --eco-from-explorer --cache-dir .cache/lichess --cache-ttl 86400

- Clear the Lichess JSON cache directory (uses config default if set):

  python -m scripts lichess-cache clear --cache-dir .cache/lichess
  python -m scripts lichess-cache stats --cache-dir .cache/lichess --top 10
  # Prewarm cache for a repertoire (SAN sequences per line)
  python -m scripts lichess-cache fetch --moves-file repertoire.txt --format san --types explorer,cloud,tablebase
  # Fetch in parallel (default 4 workers)
  python -m scripts lichess-cache fetch --moves-lines "e4 e5; d4 d5; c4 e5" --workers 8

- Insert an engine PV into a PGN at a specific position (headless):

  # Using UCI PV and a custom tag marker
  python -m scripts pgn-insert-pv --input in.pgn --output out.pgn \
    --fen "r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 2 3" \
    --pv-uci "g8f6 b1c3 f8b4" --pv-len 6 --replace --replace-if-first-matches --tag "[MY_PV]"

  # Using SAN PV and searching only the mainline
  python -m scripts pgn-insert-pv --input in.pgn --output out.pgn \
    --fen "r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 2 3" \
    --pv-san "Nf6 Nc3 Bb4" --mainline-only

  # Reading PV from a file (requires specifying format)
  python -m scripts pgn-insert-pv --input in.pgn --output out.pgn \
    --fen "8/8/8/8/8/8/5K2/6k1 w - - 0 1" --pv-file pv.txt --pv-format san

  # Auto-detect SAN/ UCI in PV file if --pv-format omitted
  python -m scripts pgn-insert-pv --input in.pgn --output out.pgn \
    --fen "8/8/8/8/8/8/5K2/6k1 w - - 0 1" --pv-file pv.txt

  # Building target position from moves instead of a FEN
  python -m scripts pgn-insert-pv --input in.pgn --output out.pgn \
    --from-moves-uci "e2e4 e7e5 g1f3" --pv-uci "b8c6 f1b5"

  # Or derive target position from PGN + ply index
  python -m scripts pgn-insert-pv --input in.pgn --output out.pgn \
    --from-pgn game.pgn --from-pgn-index 2 --ply 23 --pv-san "Nf6 Nc3 Bb4"

## Configuration

- `STOCKFISH`: path or name of the Stockfish binary (optional if in PATH).
- `SYZYGY_PATH` or `TB_PATH`: directory containing Syzygy tablebases.
  - `endgame-from-pgn` tries tablebases first; otherwise uses a 7-man piece-count heuristic.
- Engine options: `--threads` and `--hash` auto-detect sensible defaults if omitted (threads ≈ CPU cores up to 8; hash ≈ RAM/8 clamped 128–1024MB). You can override via flags.
- Additional engine knobs available on most commands: `--contempt`, `--move-overhead`.

### Optional .chessrc.json
Place a config file in the project root or your home directory to set defaults:

{
  "syzygy_path": "/path/to/syzygy",
  "stockfish_path": "/opt/homebrew/bin/stockfish",
  "engine": {
    "threads": 6,
    "hash": 512,
    "skill": 12,
    "contempt": 10,
    "move_overhead": 30
  }
  ,
  "lichess": {
    "token": "YOUR_LICHESS_TOKEN",
    "cache_dir": ".cache/lichess",
    "cache_ttl": 3600
  }
}

CLI flags still override these values.

## Notes

- The blunder detector reads the first game in a PGN. Extend as needed.
- SVG rendering uses `chess.svg` and writes an SVG file (no extra deps).
- Engine “strength” can be adjusted with `--skill 0..20` on supported commands.
- PNG export for SVGs requires CairoSVG.
- Eval chart plotting requires matplotlib; if missing, `eval-chart` can still write `--csv-out`.
- For Lichess endpoints, set `LICHESS_TOKEN` (optional) and consider using `--cache-dir` and `--cache-ttl`.

## Troubleshooting

- Stockfish not found: set `STOCKFISH` env var or add path in `.chessrc.json` as `stockfish_path`.
- Tkinter errors on macOS: install Python via the official installer (includes Tk) or `brew install python-tk` (varies by OS).
- Lichess rate limits: enable caching with `--cache-dir` and a reasonable `--cache-ttl`, and/or set `LICHESS_TOKEN`.
- GUI images not rendering: ensure `cairosvg` is installed (`pip install cairosvg`).
- Large exports: use `lichess-export-annotate` (streams to temp file), or export first then annotate.

## Contributing

Contributions are welcome! Please see `CONTRIBUTING.md` for guidelines on setup, workflow, and submitting PRs. By contributing, you agree that your contributions will be licensed under the MIT License.

## Code of Conduct

This project follows the Contributor Covenant. See `CODE_OF_CONDUCT.md`.

## License

MIT — see `LICENSE` for details.
