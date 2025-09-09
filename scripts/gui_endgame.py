import os
import tempfile
import tkinter as tk
from tkinter import simpledialog, filedialog as fd
from typing import List, Optional

import chess
import chess.svg
import threading

from .engine_utils import analyse_multipv, score_to_str, get_engine, build_engine_options
from .viz_utils import try_svg_to_png
from .tablebase_utils import open_tb, tb_summary


class EndgameGUI:
    SESSION_FILE = os.path.join(os.getcwd(), ".endgame_gui_session.json")

    def __init__(self, root: tk.Tk, board: chess.Board, tb_path: Optional[str] = None, engine_options: Optional[dict] = None):
        self.root = root
        self.root.title("Endgame Coach")
        self.tb_path = tb_path
        self.base_board = board.copy()
        self.positions: List[chess.Board] = [self.base_board]
        self.idx = 0
        self.tmp_png: Optional[str] = None
        self.best_move: Optional[chess.Move] = None
        self.size = 480  # px
        self.selected_square: Optional[chess.Square] = None
        self.orientation_white = True
        self.engine_options = engine_options or build_engine_options()
        self._busy = False
        self._status_busy = False
        self._auto_job = None
        self._last_auto_fen = None

        # UI
        self.canvas = tk.Label(self.root)
        self.canvas.grid(row=0, column=0, columnspan=4, padx=6, pady=6)
        self.canvas.bind("<Button-1>", self.on_click)

        self.info = tk.Text(self.root, width=60, height=8)
        self.info.grid(row=1, column=0, columnspan=4, padx=6, pady=6)
        self.info.configure(state=tk.DISABLED)

        self.btn_prev = tk.Button(self.root, text="Prev", command=self.on_prev)
        self.btn_next = tk.Button(self.root, text="Next", command=self.on_next)
        self.btn_tb = tk.Button(self.root, text="Probe TB", command=self.on_tb)
        self.btn_analyse = tk.Button(self.root, text="Analyse", command=self.on_analyse)
        self.btn_prev.grid(row=2, column=0, sticky="ew", padx=4, pady=4)
        self.btn_next.grid(row=2, column=1, sticky="ew", padx=4, pady=4)
        self.btn_tb.grid(row=2, column=2, sticky="ew", padx=4, pady=4)
        self.btn_analyse.grid(row=2, column=3, sticky="ew", padx=4, pady=4)

        # Depth slider, PV length, and orientation toggle
        self.depth_var = tk.IntVar(value=18)
        self.pv_len_var = tk.IntVar(value=6)
        depth_row = tk.Frame(self.root)
        depth_row.grid(row=3, column=0, columnspan=3, sticky="ew", padx=6, pady=4)
        tk.Label(depth_row, text="Depth:").pack(side=tk.LEFT)
        tk.Scale(depth_row, from_=6, to=30, orient=tk.HORIZONTAL, variable=self.depth_var, length=240).pack(side=tk.LEFT)
        tk.Label(depth_row, text="PV len:").pack(side=tk.LEFT, padx=(10, 4))
        tk.Spinbox(depth_row, from_=1, to=20, width=4, textvariable=self.pv_len_var).pack(side=tk.LEFT)
        self.btn_flip = tk.Button(self.root, text="Flip", command=self.flip_orientation)
        self.btn_flip.grid(row=3, column=3, sticky="ew", padx=6, pady=4)

        # Auto-analyse toggle and auto-depth
        auto_row = tk.Frame(self.root)
        auto_row.grid(row=4, column=0, columnspan=4, sticky="ew", padx=6, pady=4)
        self.auto_var = tk.BooleanVar(value=False)
        tk.Checkbutton(auto_row, text="Analyse continuously", variable=self.auto_var, command=self.on_toggle_auto).pack(side=tk.LEFT)
        tk.Label(auto_row, text="Auto depth:").pack(side=tk.LEFT, padx=(10, 4))
        self.auto_depth_var = tk.IntVar(value=12)
        tk.Scale(auto_row, from_=6, to=24, orient=tk.HORIZONTAL, variable=self.auto_depth_var, length=180).pack(side=tk.LEFT)

        self.status = tk.Label(self.root, anchor="w")
        self.status.grid(row=5, column=0, columnspan=4, sticky="ew", padx=6, pady=4)

        # Progress indicator (spinner)
        self.progress = tk.Label(self.root, anchor="w", fg="#666")
        self.progress.grid(row=6, column=0, columnspan=4, sticky="ew", padx=6, pady=(0,6))
        self._spin_job = None
        self._spin_idx = 0

        # Meta info for current analysis (PV length / nodes / nps)
        self.meta = tk.Label(self.root, anchor="w", fg="#444")
        self.meta.grid(row=7, column=0, columnspan=4, sticky="ew", padx=6, pady=(0,6))

        # Save/Copy/Insert buttons
        save_row = tk.Frame(self.root)
        save_row.grid(row=8, column=0, columnspan=4, sticky="ew", padx=6, pady=4)
        tk.Button(save_row, text="Save SVG", command=self.on_save_svg).pack(side=tk.LEFT)
        tk.Button(save_row, text="Save PNG", command=self.on_save_png).pack(side=tk.LEFT, padx=(8,0))
        tk.Button(save_row, text="Copy FEN", command=self.on_copy_fen).pack(side=tk.LEFT, padx=(16,0))
        tk.Button(save_row, text="Copy PV", command=self.on_copy_pv).pack(side=tk.LEFT, padx=(8,0))
        tk.Button(save_row, text="Copy UCI PV", command=self.on_copy_pv_uci).pack(side=tk.LEFT, padx=(8,0))
        tk.Button(save_row, text="Insert PV→PGN", command=self.on_insert_pv_into_pgn).pack(side=tk.LEFT, padx=(16,0))
        tk.Button(save_row, text="Remove Tagged PVs in PGN", command=self.on_remove_tagged_pvs_in_pgn).pack(side=tk.LEFT, padx=(16,0))

        # Insert options
        insert_opts = tk.Frame(self.root)
        insert_opts.grid(row=9, column=0, columnspan=4, sticky="ew", padx=6, pady=(0,6))
        self.replace_var = tk.BooleanVar(value=False)
        tk.Checkbutton(insert_opts, text="Replace existing engine-added variations", variable=self.replace_var).pack(side=tk.LEFT)
        self.mainline_only_var = tk.BooleanVar(value=False)
        tk.Checkbutton(insert_opts, text="Search mainline only", variable=self.mainline_only_var).pack(side=tk.LEFT, padx=(16,0))
        self.replace_if_match_var = tk.BooleanVar(value=True)
        tk.Checkbutton(insert_opts, text="Replace only if first move matches", variable=self.replace_if_match_var).pack(side=tk.LEFT, padx=(16,0))

        # Tag marker configuration for inserted PVs
        tag_row = tk.Frame(self.root)
        tag_row.grid(row=10, column=0, columnspan=4, sticky="ew", padx=6, pady=(0,6))
        tk.Label(tag_row, text="Tag:").pack(side=tk.LEFT)
        self.tag_var = tk.StringVar(value="[EG_PV]")
        tk.Entry(tag_row, textvariable=self.tag_var, width=16).pack(side=tk.LEFT)

        # Intercept close to save session
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Try to load session preferences (orientation/depth) if present
        self.load_session_preferences()

        # Key bindings
        self.root.bind("<Left>", lambda e: self.on_prev())
        self.root.bind("<Right>", lambda e: self.on_next())
        self.root.bind("a", lambda e: self.on_analyse())
        self.root.bind("t", lambda e: self.on_tb())
        self.root.bind("<Escape>", lambda e: self.clear_selection())

        self.render()

    def set_info(self, text: str):
        self.info.configure(state=tk.NORMAL)
        self.info.delete("1.0", tk.END)
        self.info.insert(tk.END, text)
        self.info.configure(state=tk.DISABLED)

    def current(self) -> chess.Board:
        return self.positions[self.idx]

    def render(self, arrows: Optional[list] = None):
        board = self.current()
        last_mv = board.peek() if board.move_stack else None
        arr = []
        if last_mv:
            arr.append(chess.svg.Arrow(last_mv.from_square, last_mv.to_square, color="#0984e3"))
        if self.best_move:
            arr.append(chess.svg.Arrow(self.best_move.from_square, self.best_move.to_square, color="#00b894"))
        if arrows:
            arr.extend(arrows)
        squares = []
        try:
            if self.selected_square is not None:
                squares.append(chess.svg.Square(self.selected_square, color="#ffeaa7"))
        except Exception:
            squares = []
        svg = chess.svg.board(board=board, arrows=arr, squares=squares, size=self.size, orientation=chess.WHITE if self.orientation_white else chess.BLACK)

        # Write to temporary PNG and display
        if self.tmp_png and os.path.exists(self.tmp_png):
            try:
                os.remove(self.tmp_png)
            except Exception:
                pass
        try:
            import cairosvg  # type: ignore
            fd, path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=path)
            self.tmp_png = path
            img = tk.PhotoImage(file=path)
            self.canvas.configure(image=img)
            self.canvas.image = img  # keep reference
        except Exception as e:
            self.set_info(f"Failed to render PNG: {e}\nYou can still use CLI commands.")
        # Update status line after render
        # Queue status update without blocking UI
        self.queue_status_update()
        self.maybe_schedule_auto()
        # Save session snapshot
        self.save_session()

    def clear_selection(self):
        self.selected_square = None
        self.render()

    def on_prev(self):
        if self.idx > 0:
            self.idx -= 1
            self.best_move = None
            self.set_info("")
            self.render()

    def on_next(self):
        if self.idx + 1 < len(self.positions):
            self.idx += 1
            self.best_move = None
            self.set_info("")
            self.render()

    def on_click(self, event):
        # Map click to board square
        sq = self._xy_to_square(event.x, event.y)
        if sq is None:
            return
        board = self.current()
        if self.selected_square is None:
            # Select if a piece of side-to-move is on that square
            piece = board.piece_at(sq)
            if piece and piece.color == board.turn:
                self.selected_square = sq
                self.render()
        else:
            from_sq = self.selected_square
            to_sq = sq
            # Try to find a legal move matching from->to (handle promotions)
            chosen = None
            candidates = [mv for mv in board.legal_moves if mv.from_square == from_sq and mv.to_square == to_sq]
            if len(candidates) == 1:
                chosen = candidates[0]
            elif len(candidates) > 1:
                # Likely promotion; ask user which piece
                promo_map = {
                    'q': chess.QUEEN,
                    'r': chess.ROOK,
                    'b': chess.BISHOP,
                    'n': chess.KNIGHT,
                }
                choice = self.ask_promotion()
                if choice:
                    for mv in candidates:
                        if mv.promotion == promo_map[choice]:
                            chosen = mv
                            break
            if chosen is None:
                # If clicked own piece, treat as reselect
                piece = board.piece_at(to_sq)
                if piece and piece.color == board.turn:
                    self.selected_square = to_sq
                    self.render()
                    return
                # Otherwise clear selection
                self.clear_selection()
                return
            # Apply move; truncate forward history if not at end
            new_board = board.copy()
            new_board.push(chosen)
            self.positions = self.positions[: self.idx + 1]
            self.positions.append(new_board)
            self.idx += 1
            self.best_move = None
            self.selected_square = None
            self.set_info("")
            self.render()

    def _xy_to_square(self, x: int, y: int) -> Optional[chess.Square]:
        # Convert pixel coordinates to 0..7 file/rank indices
        size = self.size
        tile = size / 8.0
        if x < 0 or y < 0 or x >= size or y >= size:
            return None
        file_idx = int(x // tile)
        rank_from_top = int(y // tile)
        if self.orientation_white:
            file = file_idx
            rank = 7 - rank_from_top
        else:
            file = 7 - file_idx
            rank = rank_from_top
        if not (0 <= file <= 7 and 0 <= rank <= 7):
            return None
        return chess.square(file, rank)

    def ask_promotion(self) -> Optional[str]:
        # Simple dialog: return one of 'q','r','b','n' or None if cancelled
        dlg = tk.Toplevel(self.root)
        dlg.title("Promote to…")
        dlg.transient(self.root)
        dlg.grab_set()
        choice: Optional[str] = None

        def set_choice(c: str):
            nonlocal choice
            choice = c
            dlg.destroy()

        row = tk.Frame(dlg)
        row.pack(padx=10, pady=10)
        for lbl, key in [("Queen", 'q'), ("Rook", 'r'), ("Bishop", 'b'), ("Knight", 'n')]:
            tk.Button(row, text=lbl, width=8, command=lambda k=key: set_choice(k)).pack(side=tk.LEFT, padx=5)
        dlg.wait_window()
        return choice

    def on_tb(self):
        with open_tb(self.tb_path) as tb:
            if not tb:
                self.set_info("No tablebases found. Set SYZYGY_PATH or pass --tb-path.")
                return
            s = tb_summary(tb, self.current())
            if not s:
                self.set_info("Position not covered by available tablebases.")
                return
            lines = []
            wdl_map = {2: "Win", 1: "Cursed Win", 0: "Draw", -1: "Blessed Loss", -2: "Loss"}
            lines.append(f"Syzygy: {wdl_map.get(int(s.get('wdl', 0)), '?')}")
            if "dtm" in s:
                lines.append(f"DTM: {s['dtm']}")
            if "dtz" in s:
                lines.append(f"DTZ: {s['dtz']}")
            roots = s.get("root", [])
            if roots:
                # Sort by WDL desc, DTZ abs asc
                roots = sorted(roots, key=lambda r: (-int(r.get("wdl", 0)), abs(int(r.get("dtz", 0)))))
                lines.append("Best TB moves:")
                for i, r in enumerate(roots[:5], 1):
                    mv = r["move"]
                    lines.append(f"  {i}. {self.current().san(mv)}  WDL={wdl_map.get(int(r['wdl']), '?')} DTZ={r.get('dtz','?')}")
            self.set_info("\n".join(lines))

    def on_analyse(self, depth: Optional[int] = None):
        if getattr(self, "_busy", False):
            return
        self._busy = True
        self.btn_analyse.configure(state=tk.DISABLED)
        self.btn_tb.configure(state=tk.DISABLED)
        cur_fen = self.current().fen()
        use_depth = int(depth if depth is not None else self.depth_var.get())
        self.start_spinner(prefix=f"Analysing (d={use_depth}) ")

        def task():
            board = chess.Board(cur_fen)
            infos = analyse_multipv(board, depth=use_depth, multipv=3, engine_options=self.engine_options)
            lines = []
            best = None
            meta_txt = ""
            self._last_pv_moves = []
            self._last_pv_san = ""
            if infos:
                tmp = board.copy()
                for i, info in enumerate(infos, 1):
                    pv = info.get("pv", [])
                    if i == 1 and pv:
                        best = pv[0]
                        # Limit PV length per UI setting
                        max_pv = max(1, int(self.pv_len_var.get()))
                        self._last_pv_moves = list(pv[:max_pv])
                    san_pv = []
                    tmp.reset()
                    tmp.set_fen(board.fen())
                    # Use the same PV length for rendering summary lines
                    max_line_pv = max(1, int(self.pv_len_var.get()))
                    for m in pv[:max_line_pv]:
                        san_pv.append(tmp.san(m))
                        tmp.push(m)
                    if i == 1:
                        self._last_pv_san = " ".join(san_pv)
                    score = info.get("score")
                    s = score_to_str(score.pov(board.turn)) if score else "?"
                    lines.append(f"{i}. {san_pv[0] if san_pv else '(no move)'}  eval={s}  pv={' '.join(san_pv)}")
                # Meta from PV 1
                first = infos[0]
                meta_parts = []
                if first.get("depth"):
                    meta_parts.append(f"depth {first.get('depth')}")
                if first.get("seldepth"):
                    meta_parts.append(f"sd {first.get('seldepth')}")
                if first.get("pv"):
                    meta_parts.append(f"pv {len(first.get('pv'))}")
                if first.get("nodes"):
                    meta_parts.append(f"nodes {self._humanize(first.get('nodes'))}")
                if first.get("nps"):
                    meta_parts.append(f"nps {self._humanize(first.get('nps'))}/s")
                # Append TB info if available
                try:
                    with open_tb(self.tb_path) as tb:
                        if tb:
                            s = tb_summary(tb, board)
                            if s:
                                wmap = {2: "Win", 1: "Cursed Win", 0: "Draw", -1: "Blessed Loss", -2: "Loss"}
                                meta_parts.append(f"TB {wmap.get(int(s.get('wdl',0)), '?')}")
                                if 'dtz' in s:
                                    meta_parts.append(f"DTZ {s['dtz']}")
                                if 'dtm' in s:
                                    meta_parts.append(f"DTM {s['dtm']}")
                except Exception:
                    pass
                meta_txt = " | ".join(meta_parts)
            def done():
                self.best_move = best
                self.set_info("\n".join(lines) if lines else "No analysis available.")
                self._busy = False
                self.btn_analyse.configure(state=tk.NORMAL)
                self.btn_tb.configure(state=tk.NORMAL)
                self.stop_spinner()
                self.meta.configure(text=meta_txt)
                self.render()
            self.root.after(0, done)

        threading.Thread(target=task, daemon=True).start()

    def update_status(self):
        """Show TB WDL or a quick engine eval for current position."""
        board = self.current()
        # Try tablebases first
        with open_tb(self.tb_path) as tb:
            if tb:
                s = tb_summary(tb, board)
                if s:
                    wmap = {2: "Win", 1: "Cursed Win", 0: "Draw", -1: "Blessed Loss", -2: "Loss"}
                    parts = [f"TB: {wmap.get(int(s.get('wdl', 0)), '?')}"]
                    if "dtz" in s:
                        parts.append(f"DTZ {s['dtz']}")
                    if "dtm" in s:
                        parts.append(f"DTM {s['dtm']}")
                    self.status.configure(text=" | ".join(parts))
                    return
        # Engine quick eval
        try:
            with get_engine(options=self.engine_options) as eng:
                info = eng.analyse(board, chess.engine.Limit(depth=10))
                score = info.get("score")
                if score:
                    s = score_to_str(score.pov(board.turn))
                    self.status.configure(text=f"Eval: {s}")
        except Exception:
            pass

    def queue_status_update(self):
        if getattr(self, "_status_busy", False):
            return
        self._status_busy = True
        fen = self.current().fen()

        def task():
            txt = ""
            b = chess.Board(fen)
            with open_tb(self.tb_path) as tb:
                if tb:
                    s = tb_summary(tb, b)
                    if s:
                        wmap = {2: "Win", 1: "Cursed Win", 0: "Draw", -1: "Blessed Loss", -2: "Loss"}
                        parts = [f"TB: {wmap.get(int(s.get('wdl', 0)), '?')}"]
                        if "dtz" in s:
                            parts.append(f"DTZ {s['dtz']}")
                        if "dtm" in s:
                            parts.append(f"DTM {s['dtm']}")
                        txt = " | ".join(parts)
            if not txt:
                try:
                    with get_engine(options=self.engine_options) as eng:
                        info = eng.analyse(b, chess.engine.Limit(depth=10))
                        score = info.get("score")
                        if score:
                            txt = f"Eval: {score_to_str(score.pov(b.turn))}"
                except Exception:
                    txt = ""

            def done():
                self.status.configure(text=txt)
                self._status_busy = False

            self.root.after(0, done)

        threading.Thread(target=task, daemon=True).start()

    def flip_orientation(self):
        self.orientation_white = not self.orientation_white
        self.render()

    def start_spinner(self, prefix: str = "Working "):
        self._spin_prefix = prefix
        self._spin_chars = "-|/\\"
        def tick():
            ch = self._spin_chars[self._spin_idx % len(self._spin_chars)]
            self._spin_idx += 1
            self.progress.configure(text=f"{self._spin_prefix}{ch}")
            self._spin_job = self.root.after(120, tick)
        self.stop_spinner()
        self._spin_idx = 0
        tick()

    def stop_spinner(self):
        if self._spin_job is not None:
            try:
                self.root.after_cancel(self._spin_job)
            except Exception:
                pass
            self._spin_job = None
        self.progress.configure(text="")

    def _humanize(self, n: int) -> str:
        try:
            n = int(n)
        except Exception:
            return str(n)
        for unit, div in (("B", 1), ("k", 1_000), ("M", 1_000_000), ("G", 1_000_000_000)):
            if n < div * 1000 or unit == "G":
                val = n / div
                if val >= 100:
                    return f"{val:.0f}{unit}"
                if val >= 10:
                    return f"{val:.1f}{unit}"
                return f"{val:.2f}{unit}"
        return str(n)

    def build_svg(self) -> str:
        board = self.current()
        last_mv = board.peek() if board.move_stack else None
        arrows = []
        if last_mv:
            arrows.append(chess.svg.Arrow(last_mv.from_square, last_mv.to_square, color="#0984e3"))
        if self.best_move:
            arrows.append(chess.svg.Arrow(self.best_move.from_square, self.best_move.to_square, color="#00b894"))
        squares = []
        try:
            if self.selected_square is not None:
                squares.append(chess.svg.Square(self.selected_square, color="#ffeaa7"))
        except Exception:
            squares = []
        return chess.svg.board(
            board=board,
            arrows=arrows,
            squares=squares,
            size=self.size,
            orientation=chess.WHITE if self.orientation_white else chess.BLACK,
        )

    def on_save_svg(self):
        try:
            path = fd.asksaveasfilename(defaultextension=".svg", filetypes=[("SVG", ".svg")], title="Save board as SVG")
            if not path:
                return
            svg = self.build_svg()
            with open(path, "w", encoding="utf-8") as f:
                f.write(svg)
        except Exception:
            pass

    def on_save_png(self):
        try:
            path = fd.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", ".png")], title="Save board as PNG")
            if not path:
                return
            svg = self.build_svg()
            ok, msg = try_svg_to_png(svg, path)
            # Optionally reflect status
            self.status.configure(text=msg)
        except Exception:
            pass

    def on_copy_fen(self):
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.current().fen())
            self.status.configure(text="FEN copied to clipboard")
            self.show_toast("FEN copied")
        except Exception:
            pass

    def on_copy_pv(self):
        try:
            if not getattr(self, "_last_pv_san", ""):
                self.status.configure(text="Run Analyse to get PV first")
                return
            self.root.clipboard_clear()
            self.root.clipboard_append(self._last_pv_san)
            self.status.configure(text="PV (SAN) copied to clipboard")
            self.show_toast("PV (SAN) copied")
        except Exception:
            pass

    def on_copy_pv_uci(self):
        try:
            if not getattr(self, "_last_pv_moves", []):
                self.status.configure(text="Run Analyse to get PV first")
                return
            uci = " ".join(m.uci() for m in self._last_pv_moves)
            self.root.clipboard_clear()
            self.root.clipboard_append(uci)
            self.status.configure(text="PV (UCI) copied to clipboard")
            self.show_toast("PV (UCI) copied")
        except Exception:
            pass

    def on_insert_pv_into_pgn(self):
        # Requires last PV from analysis
        try:
            if not getattr(self, "_last_pv_moves", []):
                self.status.configure(text="Run Analyse to get PV first")
                return
            src = fd.askopenfilename(filetypes=[("PGN", ".pgn")], title="Select PGN to modify")
            if not src:
                return
            dst = fd.asksaveasfilename(defaultextension=".pgn", filetypes=[("PGN", ".pgn")], title="Save modified PGN as")
            if not dst:
                return
            import chess.pgn
            count = 0
            inserted = 0
            with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
                while True:
                    game = chess.pgn.read_game(fin)
                    if not game:
                        break
                    count += 1
                    cur_fen = self.current().fen()
                    found_node = None

                    if self.mainline_only_var.get():
                        # Search only mainline
                        node = game
                        board = game.board()
                        if board.fen() == cur_fen:
                            found_node = node
                        else:
                            while node.variations:
                                next_node = node.variations[0]
                                board.push(next_node.move)
                                if board.fen() == cur_fen:
                                    found_node = node
                                    break
                                node = next_node
                    else:
                        # Depth-first search through all nodes including variations
                        def dfs(node, board):
                            nonlocal found_node
                            if found_node is not None:
                                return
                            if board.fen() == cur_fen:
                                found_node = node
                                return
                            for child in node.variations:
                                nb = board.copy()
                                nb.push(child.move)
                                dfs(child, nb)

                        dfs(game, game.board())
                    if found_node:
                        # Optionally remove previous engine-added variations at this node
                        if self.replace_var.get():
                            marker = self.tag_var.get().strip() or "[EG_PV]"
                            new_first = self._last_pv_moves[0] if self._last_pv_moves else None
                            def keep_var(var_node):
                                if marker not in (var_node.comment or ""):
                                    return True
                                if self.replace_if_match_var.get() and new_first is not None:
                                    # Compare first move of this variation to new_first
                                    first_child = var_node
                                    mv = first_child.move if hasattr(first_child, 'move') else None
                                    return mv != new_first
                                return False
                            try:
                                found_node.variations = [v for v in list(found_node.variations) if keep_var(v)]
                            except Exception:
                                kept = []
                                for v in list(found_node.variations):
                                    if keep_var(v):
                                        kept.append(v)
                                try:
                                    found_node.variations = kept
                                except Exception:
                                    pass

                        # Insert limited PV moves
                        var_node = found_node
                        first_child = None
                        for mv in self._last_pv_moves:
                            var_node = var_node.add_variation(mv)
                            if first_child is None:
                                first_child = var_node
                        # Mark the first variation node so we can identify later for replace
                        if first_child is not None:
                            c = (first_child.comment or "").strip()
                            marker = self.tag_var.get().strip() or "[EG_PV]"
                            first_child.comment = (c + (" " if c else "") + marker).strip()
                        inserted += 1
                    exporter = chess.pgn.FileExporter(fout)
                    game.accept(exporter)
            self.status.configure(text=f"Inserted PV in {inserted}/{count} game(s)")
            self.show_toast(f"Inserted PV in {inserted}/{count}")
        except Exception as e:
            self.status.configure(text=f"Insert failed: {e}")

    def show_toast(self, message: str, duration_ms: int = 1500):
        try:
            toast = tk.Toplevel(self.root)
            toast.overrideredirect(True)
            toast.attributes("-topmost", True)
            lbl = tk.Label(toast, text=message, bg="#333", fg="#fff", padx=10, pady=6)
            lbl.pack()
            # Position near bottom-right of root
            self.root.update_idletasks()
            rx = self.root.winfo_rootx()
            ry = self.root.winfo_rooty()
            rw = self.root.winfo_width()
            rh = self.root.winfo_height()
            tw = lbl.winfo_reqwidth()
            th = lbl.winfo_reqheight()
            x = rx + rw - tw - 20
            y = ry + rh - th - 40
            toast.geometry(f"{tw}x{th}+{x}+{y}")
            toast.after(duration_ms, toast.destroy)
        except Exception:
            pass

    def on_toggle_auto(self):
        if self.auto_var.get():
            self.maybe_schedule_auto(force=True)
        else:
            if self._auto_job is not None:
                try:
                    self.root.after_cancel(self._auto_job)
                except Exception:
                    pass
                self._auto_job = None

    def maybe_schedule_auto(self, force: bool = False):
        if not self.auto_var.get():
            return
        # Avoid tight loops; schedule a tick every ~1.0s
        if self._auto_job is not None and not force:
            return
        def tick():
            self._auto_job = None
            fen = self.current().fen()
            # Re-analyse if FEN changed or no best move yet
            if self._busy:
                # Backoff while busy
                delay = 1500
            elif fen != self._last_auto_fen or self.best_move is None:
                self._last_auto_fen = fen
                self.on_analyse(depth=int(self.auto_depth_var.get()))
                delay = 1000
            else:
                delay = 1000
            # reschedule
            if self.auto_var.get():
                self._auto_job = self.root.after(delay, tick)
        self._auto_job = self.root.after(1000, tick)

    def on_close(self):
        try:
            self.save_session()
        finally:
            self.root.destroy()

    def save_session(self):
        try:
            import json
            data = {
                "fen": self.current().fen(),
                "orientation_white": self.orientation_white,
                "depth": int(self.depth_var.get()) if hasattr(self, "depth_var") else 18,
                "auto": bool(self.auto_var.get()) if hasattr(self, "auto_var") else False,
                "auto_depth": int(self.auto_depth_var.get()) if hasattr(self, "auto_depth_var") else 12,
                "geometry": self.root.winfo_geometry(),
            }
            with open(self.SESSION_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception:
            pass

    def load_session_preferences(self):
        try:
            import json
            if not os.path.exists(self.SESSION_FILE):
                return
            with open(self.SESSION_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self.orientation_white = bool(data.get("orientation_white", True))
                if hasattr(self, "depth_var") and isinstance(data.get("depth"), int):
                    self.depth_var.set(int(data.get("depth")))
                if hasattr(self, "auto_var"):
                    self.auto_var.set(bool(data.get("auto", False)))
                if hasattr(self, "auto_depth_var") and isinstance(data.get("auto_depth"), int):
                    self.auto_depth_var.set(int(data.get("auto_depth")))
                geom = data.get("geometry")
                if isinstance(geom, str):
                    try:
                        self.root.geometry(geom)
                    except Exception:
                        pass
        except Exception:
            pass


def run(fen: Optional[str] = None, pgn: Optional[str] = None, tb_path: Optional[str] = None, engine_options: Optional[dict] = None):
    # If no FEN/PGN provided, try to resume last session FEN
    resume_fen = None
    if not fen and not pgn:
        try:
            import json
            sess_path = EndgameGUI.SESSION_FILE
            if os.path.exists(sess_path):
                with open(sess_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and data.get("fen"):
                    resume_fen = str(data.get("fen"))
        except Exception:
            resume_fen = None

    board = chess.Board(fen or resume_fen) if (fen or resume_fen) else chess.Board()
    if pgn:
        import chess.pgn
        with open(pgn, "r", encoding="utf-8") as f:
            game = chess.pgn.read_game(f)
        if game:
            board = game.board()
            # preload all positions for navigation
            positions = [board.copy()]
            for mv in game.mainline_moves():
                board.push(mv)
                positions.append(board.copy())
            board = positions[0]
            # Create GUI and attach positions
            root = tk.Tk()
            gui = EndgameGUI(root, board, tb_path=tb_path, engine_options=engine_options)
            gui.positions = positions
            gui.idx = 0
            gui.render()
            root.mainloop()
            return

    root = tk.Tk()
    EndgameGUI(root, board, tb_path=tb_path, engine_options=engine_options)
    root.mainloop()
    def on_remove_tagged_pvs_in_pgn(self):
        try:
            src = fd.askopenfilename(filetypes=[("PGN", ".pgn")], title="Select PGN to clean")
            if not src:
                return
            dst = fd.asksaveasfilename(defaultextension=".pgn", filetypes=[("PGN", ".pgn")], title="Save cleaned PGN as")
            if not dst:
                return
            import chess.pgn
            tag = self.tag_var.get().strip() or "[EG_PV]"
            removed = 0
            total = 0
            with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
                while True:
                    game = chess.pgn.read_game(fin)
                    if not game:
                        break
                    total += 1
                    # DFS and clean tagged variations at every node
                    def clean(node):
                        nonlocal removed
                        try:
                            before = len(node.variations)
                            node.variations = [v for v in list(node.variations) if tag not in (v.comment or "")]
                            removed += before - len(node.variations)
                        except Exception:
                            pass
                        for v in list(node.variations):
                            clean(v)
                    clean(game)
                    exporter = chess.pgn.FileExporter(fout)
                    game.accept(exporter)
            self.status.configure(text=f"Removed {removed} tagged variation(s) across {total} game(s)")
            self.show_toast(f"Removed {removed} variations")
        except Exception as e:
            self.status.configure(text=f"Clean failed: {e}")
