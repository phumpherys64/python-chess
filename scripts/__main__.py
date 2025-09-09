import argparse
import io
import sys
from typing import Optional

import chess
import chess.pgn
import chess.svg

from .engine_utils import analyse_multipv, score_to_str, set_strength
from .engine_utils import get_engine, build_engine_options
from .tablebase_utils import open_tb, tb_summary
from .viz_utils import red_green_hex, try_svg_to_png
from .gui_endgame import run as run_endgame_gui
from .config import load_config, config_value
from .lichess_api import tablebase_standard as lichess_tb, cloud_eval as lichess_cloud_eval, opening_explorer as lichess_opening_explorer, export_user_games as lichess_export_user


def load_board_from_args(args: argparse.Namespace) -> chess.Board:
    if args.fen:
        return chess.Board(args.fen)
    board = chess.Board()
    if args.moves:
        # support SAN or UCI list (space-separated)
        for token in args.moves.split():
            try:
                move = board.parse_san(token)
            except ValueError:
                move = chess.Move.from_uci(token)
                if move not in board.legal_moves:
                    raise
            board.push(move)
    return board


def _cfg_tb_path(args: argparse.Namespace) -> Optional[str]:
    if getattr(args, "tb_path", None):
        return args.tb_path
    return config_value(getattr(args, "_config", {}), "syzygy_path")


def _engine_opts_from_args(args: argparse.Namespace) -> dict:
    cfg = getattr(args, "_config", {})
    def pick(val, *keys, default=None):
        return val if val is not None else config_value(cfg, *keys, default=default)

    return build_engine_options(
        skill=pick(getattr(args, "skill", None), "engine", "skill"),
        threads=pick(getattr(args, "threads", None), "engine", "threads"),
        hash_mb=pick(getattr(args, "hash", None), "engine", "hash"),
        contempt=pick(getattr(args, "contempt", None), "engine", "contempt"),
        move_overhead=pick(getattr(args, "move_overhead", None), "engine", "move_overhead"),
        use_defaults=True,
    )


def _lichess_defaults(args: argparse.Namespace) -> dict:
    cfg = getattr(args, "_config", {})
    return {
        "token": getattr(args, "token", None) or config_value(cfg, "lichess", "token"),
        "cache_dir": getattr(args, "cache_dir", None) or config_value(cfg, "lichess", "cache_dir"),
        "cache_ttl": getattr(args, "cache_ttl", None) or config_value(cfg, "lichess", "cache_ttl"),
    }


def cmd_analyse(args: argparse.Namespace) -> int:
    board = load_board_from_args(args)
    opts = _engine_opts_from_args(args)
    infos = analyse_multipv(board, depth=args.depth, multipv=args.multipv, engine_options=opts)
    for i, info in enumerate(infos, 1):
        pv = info.get("pv", [])
        score = info.get("score")
        san_pv = []
        tmp = board.copy()
        for m in pv[:args.pv_len]:
            san_pv.append(tmp.san(m))
            tmp.push(m)
        score_str = score_to_str(score.pov(board.turn)) if score else "?"
        print(f"{i}. {san_pv[0] if san_pv else '(no move)'}  eval={score_str}  pv={' '.join(san_pv)}")
    return 0


def cmd_blunders(args: argparse.Namespace) -> int:
    if not args.pgn:
        print("--pgn is required for blunders", file=sys.stderr)
        return 2
    with open(args.pgn, "r", encoding="utf-8") as f:
        game = chess.pgn.read_game(f)
    if not game:
        print("No game found in PGN", file=sys.stderr)
        return 2

    board = game.board()
    prev_eval = None
    opts = _engine_opts_from_args(args)

    with get_engine(options=opts) as eng:
        for idx, move in enumerate(game.mainline_moves(), 1):
            info_before = eng.analyse(board, chess.engine.Limit(depth=args.depth))
            prev_eval = info_before["score"].pov(board.turn).score(mate_score=100000)
            board.push(move)
            info_after = eng.analyse(board, chess.engine.Limit(depth=args.depth))
            new_eval = info_after["score"].pov(board.turn).score(mate_score=100000)
            drop = (prev_eval or 0) - (new_eval or 0)
            if drop > args.threshold:
                print(f"Move {idx}: {board.peek().uci()} looks bad (Δ={drop}cp)")
    return 0


def cmd_bestmove_svg(args: argparse.Namespace) -> int:
    board = load_board_from_args(args)
    opts = _engine_opts_from_args(args)
    with get_engine(options=opts) as eng:
        info = eng.analyse(board, chess.engine.Limit(depth=args.depth))
    best = info.get("pv", [None])[0]
    arrows = []
    if best:
        arrows = [chess.svg.Arrow(best.from_square, best.to_square, color="#00b894")]
    svg = chess.svg.board(board=board, arrows=arrows)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"Wrote {args.out}")
    if args.png_out:
        ok, msg = try_svg_to_png(svg, args.png_out)
        print(msg)
    return 0


def wdl_to_text(wdl: int) -> str:
    mapping = {
        2: "Win",
        1: "Cursed Win",
        0: "Draw",
        -1: "Blessed Loss",
        -2: "Loss",
    }
    return mapping.get(wdl, "?")


def cmd_endgame_coach(args: argparse.Namespace) -> int:
    board = load_board_from_args(args)

    # First try tablebases
    with open_tb(_cfg_tb_path(args)) as tb:
        if tb:
            summary = tb_summary(tb, board)
            if summary:
                print(f"Syzygy: {wdl_to_text(summary.get('wdl'))}")
                if "dtm" in summary:
                    print(f"DTM: {summary['dtm']}")
                if "dtz" in summary:
                    print(f"DTZ: {summary['dtz']}")
                roots = summary.get("root", [])
                if roots:
                    # Show up to N best moves by WDL/DTZ ordering
                    def _key(item):
                        # Prefer higher WDL, then lower DTZ
                        return (-item.get("wdl", 0), abs(item.get("dtz", 0)))

                    roots = sorted(roots, key=_key)
                    print("Best TB moves:")
                    for i, r in enumerate(roots[: args.top], 1):
                        san = board.san(r["move"]) if isinstance(r["move"], chess.Move) else str(r["move"]) 
                        print(f"  {i}. {san}  WDL={wdl_to_text(r['wdl'])}  DTZ={r.get('dtz','?')}")
                return 0
            else:
                print("Position not covered by available tablebases; falling back to engine…")
        else:
            if args.tb_path:
                print("No tablebases found at provided path; falling back to engine…")

    # Optional online fallback to Lichess tablebase (if enabled)
    if getattr(args, "online_fallbacks", False):
        try:
            ld = _lichess_defaults(args)
            data = lichess_tb(board.fen(), token=ld["token"], cache_dir=ld["cache_dir"], cache_ttl=ld["cache_ttl"])
            if data and data.get("category"):
                print(f"Lichess TB: {data.get('category')} dtz={data.get('dtz')} dtm={data.get('dtm')}")
                for i, m in enumerate((data.get("moves") or [])[: args.top], 1):
                    print(f"  {i}. {m.get('uci')} wdl={m.get('wdl')} dtz={m.get('dtz')}")
                return 0
        except Exception as e:
            print(f"Lichess TB unavailable: {e}")

    # Engine/cloud fallback: multi-PV suggestions
    if getattr(args, "online_fallbacks", False):
        try:
            ld = _lichess_defaults(args)
            data = lichess_cloud_eval(board.fen(), multi_pv=args.multipv, token=ld["token"], cache_dir=ld["cache_dir"], cache_ttl=ld["cache_ttl"])
            pvs = data.get("pvs", [])
            if pvs:
                print("Lichess cloud suggestions:")
                for i, pv in enumerate(pvs, 1):
                    line = pv.get("moves", "").split()
                    score = pv.get("cp") if "cp" in pv else (f"M{pv.get('mate')}" if pv.get('mate') is not None else "?")
                    print(f"  {i}. score={score} pv={' '.join(line[:args.pv_len])}")
                return 0
        except Exception as e:
            print(f"Lichess cloud eval unavailable: {e}")

    # Local engine fallback: simple multi-PV suggestions with short PV
    opts = _engine_opts_from_args(args)
    infos = analyse_multipv(board, depth=args.depth, multipv=args.multipv, engine_options=opts)
    print("Engine suggestions:")
    for i, info in enumerate(infos, 1):
        pv = info.get("pv", [])
        score = info.get("score")
        tmp = board.copy()
        san_pv = []
        for m in pv[: args.pv_len]:
            san_pv.append(tmp.san(m))
            tmp.push(m)
        score_str = score_to_str(score.pov(board.turn)) if score else "?"
        print(f"  {i}. {san_pv[0] if san_pv else '(no move)'}  eval={score_str}  pv={' '.join(san_pv)}")
    return 0


def cmd_move_heatmap_svg(args: argparse.Namespace) -> int:
    board = load_board_from_args(args)
    side = board.turn
    legal = list(board.legal_moves)
    if not legal:
        print("No legal moves in position.")
        return 0

    scores = []
    opts = _engine_opts_from_args(args)
    with get_engine(options=opts) as eng:
        for mv in legal:
            board.push(mv)
            info = eng.analyse(board, chess.engine.Limit(depth=args.depth))
            sc = info["score"].pov(side).score(mate_score=100000)
            scores.append((mv, sc if sc is not None else -10_000_000))
            board.pop()

    # Normalize scores to 0..1 for coloring
    vals = [s for _, s in scores]
    mn, mx = min(vals), max(vals)
    span = (mx - mn) if mx != mn else 1
    arrows = []
    for mv, val in scores:
        t = (val - mn) / span
        color = red_green_hex(t)
        arrows.append(chess.svg.Arrow(mv.from_square, mv.to_square, color=color, size=8))

    svg = chess.svg.board(board=board, arrows=arrows)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"Wrote {args.out}")
    if args.png_out:
        ok, msg = try_svg_to_png(svg, args.png_out)
        print(msg)
    return 0


def cmd_eval_chart(args: argparse.Namespace) -> int:
    if not args.pgn:
        print("--pgn required", file=sys.stderr)
        return 2
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        print(f"matplotlib not available: {e}")
        if args.csv_out:
            # Fallback: write CSV data only
            with open(args.pgn, "r", encoding="utf-8") as f:
                game = chess.pgn.read_game(f)
            if not game:
                print("No game found in PGN", file=sys.stderr)
                return 2
            board = game.board()
            rows = [(0, 0)]
            opts = _engine_opts_from_args(args)
            with get_engine(options=opts) as eng:
                for idx, mv in enumerate(game.mainline_moves(), 1):
                    board.push(mv)
                    info = eng.analyse(board, chess.engine.Limit(depth=args.depth))
                    side = not board.turn  # eval after move from mover's POV
                    cp = info["score"].pov(side).score(mate_score=100000) or 0
                    rows.append((idx, cp))
            with open(args.csv_out, "w", encoding="utf-8") as f:
                f.write("ply,cp\n")
                for ply, cp in rows:
                    f.write(f"{ply},{cp}\n")
            print(f"Wrote {args.csv_out}")
            return 0
        return 2

    with open(args.pgn, "r", encoding="utf-8") as f:
        game = chess.pgn.read_game(f)
    if not game:
        print("No game found in PGN", file=sys.stderr)
        return 2

    board = game.board()
    evals = [0]
    blunders = []  # list of (ply, cp)
    prev_cp = 0
    opts = _engine_opts_from_args(args)
    with get_engine(options=opts) as eng:
        for idx, mv in enumerate(game.mainline_moves(), 1):
            info_before = eng.analyse(board, chess.engine.Limit(depth=args.depth))
            prev_cp = info_before["score"].pov(board.turn).score(mate_score=100000) or 0
            board.push(mv)
            info_after = eng.analyse(board, chess.engine.Limit(depth=args.depth))
            cp = info_after["score"].pov(not board.turn).score(mate_score=100000) or 0
            drop = prev_cp - cp
            if drop > args.threshold:
                blunders.append((idx, cp))
            evals.append(cp)

    xs = list(range(len(evals)))
    ys = [cp / 100.0 for cp in evals]
    plt.figure(figsize=(10, 3))
    plt.plot(xs, ys, label="eval (pawns)")
    if blunders:
        bx = [ply for ply, _ in blunders]
        by = [cp / 100.0 for _, cp in blunders]
        plt.scatter(bx, by, color="red", label="blunder")
    plt.axhline(0, color="#999", linewidth=0.8)
    plt.xlabel("Ply")
    plt.ylabel("Eval (pawns)")
    plt.title("Evaluation chart")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Wrote {args.out}")
    if args.csv_out:
        with open(args.csv_out, "w", encoding="utf-8") as f:
            f.write("ply,cp\n")
            for ply, cp in enumerate(evals):
                f.write(f"{ply},{cp}\n")
        print(f"Wrote {args.csv_out}")
    return 0


def cmd_endgame_from_pgn(args: argparse.Namespace) -> int:
    if not args.pgn:
        print("--pgn required", file=sys.stderr)
        return 2
    with open(args.pgn, "r", encoding="utf-8") as f:
        game = chess.pgn.read_game(f)
    if not game:
        print("No game found in PGN", file=sys.stderr)
        return 2

    board = game.board()
    found_pos = None

    with open_tb(_cfg_tb_path(args)) as tb:
        for mv in game.mainline_moves():
            board.push(mv)
            if tb:
                s = tb_summary(tb, board)
                if s:
                    found_pos = board.copy()
                    break
            else:
                # Heuristic: 7 or fewer pieces → endgame (7-man coverage)
                if len(board.piece_map()) <= 7:
                    found_pos = board.copy()
                    break

        if not found_pos:
            print("No endgame segment detected; try adjusting criteria.")
            return 0

        # Reuse endgame_coach logic on found_pos
        print("Detected endgame position:")
        print(found_pos.fen())
        if tb:
            summary = tb_summary(tb, found_pos)
            if summary:
                print(f"Syzygy: {wdl_to_text(summary.get('wdl'))}")
                if "dtm" in summary:
                    print(f"DTM: {summary['dtm']}")
                if "dtz" in summary:
                    print(f"DTZ: {summary['dtz']}")
                roots = summary.get("root", [])
                if roots:
                    def _key(item):
                        return (-item.get("wdl", 0), abs(item.get("dtz", 0)))
                    roots = sorted(roots, key=_key)
                    print("Best TB moves:")
                    for i, r in enumerate(roots[: args.top], 1):
                        san = found_pos.san(r["move"]) if isinstance(r["move"], chess.Move) else str(r["move"]) 
                        print(f"  {i}. {san}  WDL={wdl_to_text(r['wdl'])}  DTZ={r.get('dtz','?')}")
                return 0

    # Engine fallback
    opts = build_engine_options(args.skill, args.threads, args.hash)
    infos = analyse_multipv(found_pos, depth=args.depth, multipv=args.multipv, engine_options=opts)
    print("Engine suggestions:")
    tmp = found_pos.copy()
    for i, info in enumerate(infos, 1):
        pv = info.get("pv", [])
        tmp = found_pos.copy()
        san_pv = []
        for m in pv[: args.pv_len]:
            san_pv.append(tmp.san(m))
            tmp.push(m)
        score = info.get("score")
        score_str = score_to_str(score.pov(found_pos.turn)) if score else "?"
        print(f"  {i}. {san_pv[0] if san_pv else '(no move)'}  eval={score_str}  pv={' '.join(san_pv)}")
    return 0


def annotate_game(
    game: chess.pgn.Game,
    depth: int,
    threshold: int,
    big_threshold: int,
    tb_path: Optional[str],
    add_variations: str = "blunders",  # "none" | "blunders" | "all"
    max_pv_len: int = 6,
    keep_existing_variations: bool = True,
    engine_options: Optional[dict] = None,
    tag: Optional[str] = None,
    blunders_list: Optional[list] = None,
    game_id: Optional[str] = None,
) -> chess.pgn.Game:
    board = game.board()
    node = game
    with get_engine(options=engine_options) as eng:
        with open_tb(tb_path) as tb:
            while node.variations:
                next_node = node.variations[0]
                move = next_node.move
                # Before eval
                info_before = eng.analyse(board, chess.engine.Limit(depth=depth))
                prev_cp = info_before["score"].pov(board.turn).score(mate_score=100000) or 0
                fen_before = board.fen()
                board.push(move)
                # After eval
                info_after = eng.analyse(board, chess.engine.Limit(depth=depth))
                cp = info_after["score"].pov(not board.turn).score(mate_score=100000) or 0
                fen_after = board.fen()
                delta = prev_cp - cp
                # Compose comment
                parts = [f"eval={cp/100:.2f} (Δ={delta:+.2f} pawns)"]
                if tb:
                    s = tb_summary(tb, board)
                    if s:
                        wmap = {2: "Win", 1: "Cursed Win", 0: "Draw", -1: "Blessed Loss", -2: "Loss"}
                        parts.append(f"TB={wmap.get(int(s.get('wdl', 0)), '?')}")
                        if "dtz" in s:
                            parts.append(f"DTZ={s['dtz']}")
                        if "dtm" in s:
                            parts.append(f"DTM={s['dtm']}")
                # Add PV summary
                pv = info_after.get("pv", [])
                if pv:
                    tmp = board.copy()
                    san_pv = []
                    for m in pv[:max_pv_len]:
                        san_pv.append(tmp.san(m))
                        tmp.push(m)
                    parts.append("pv=" + " ".join(san_pv))

                # Write comment and NAGs
                comment = next_node.comment.strip()
                comment = (comment + " ").strip() if comment else ""
                next_node.comment = (comment + "{" + "; ".join(parts) + "}").strip()
                is_blunder = False
                if delta > big_threshold:
                    next_node.nags.add(chess.pgn.NAG_BLUNDER)  # ??
                    is_blunder = True
                elif delta > threshold:
                    next_node.nags.add(chess.pgn.NAG_MISTAKE)  # ?
                    is_blunder = True
                if blunders_list is not None and is_blunder:
                    try:
                        blunders_list.append({
                            "game_id": game_id or "",
                            "ply": node.ply(),
                            "move": move.uci(),
                            "san": next_node.san(),
                            "delta_cp": float(delta) / 100.0,
                            "prev_cp": float(prev_cp) / 100.0,
                            "new_cp": float(cp) / 100.0,
                            "fen_before": fen_before,
                            "fen_after": fen_after,
                        })
                    except Exception:
                        pass

                # Add engine line as a variation:
                # - For blunders: add best line from BEFORE the move to show the refutation or better move.
                # - For all: add from before-move regardless of delta.
                need_line = (add_variations == "all") or (add_variations == "blunders" and (delta > big_threshold))
                if need_line:
                    if keep_existing_variations and node.variations and len(node.variations) > 1:
                        pass  # keep them; still can add more
                    # Best line from before the move
                    best_line_info = info_before
                    pv_moves = best_line_info.get("pv", [])
                    if pv_moves:
                        # Create a copy board to safely push moves
                        pv_board = board.copy()
                        pv_board.pop()  # revert to pre-move
                        # Add a variation to node (pre-move)
                        var_node = node
                        first_added = None
                        added = 0
                        for m in pv_moves:
                            if added >= max_pv_len:
                                break
                            var_node = var_node.add_variation(m)
                            if first_added is None:
                                first_added = var_node
                            added += 1
                        if first_added is not None and tag:
                            c = (first_added.comment or "").strip()
                            first_added.comment = (c + (" " if c else "") + tag).strip()

                node = next_node
    return game


def cmd_annotate_pgn(args: argparse.Namespace) -> int:
    if not args.input or not args.output:
        print("--input and --output required", file=sys.stderr)
        return 2
    if args.all:
        with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
            while True:
                game = chess.pgn.read_game(fin)
                if not game:
                    break
                game = annotate_game(
                    game,
                    args.depth,
                    args.threshold,
                    args.big_threshold,
                    _cfg_tb_path(args),
                    add_variations=args.variations,
                    max_pv_len=args.max_pv_len,
                    keep_existing_variations=not args.prune_variations,
                    engine_options=_engine_opts_from_args(args),
                    tag=args.tag,
                )
                exporter = chess.pgn.FileExporter(fout)
                game.accept(exporter)
        print(f"Wrote {args.output}")
        return 0
    else:
        with open(args.input, "r", encoding="utf-8") as fin:
            game = chess.pgn.read_game(fin)
        if not game:
            print("No game found in input PGN", file=sys.stderr)
            return 2
        game = annotate_game(
            game,
            args.depth,
            args.threshold,
            args.big_threshold,
            _cfg_tb_path(args),
            add_variations=args.variations,
            max_pv_len=args.max_pv_len,
            keep_existing_variations=not args.prune_variations,
            engine_options=_engine_opts_from_args(args),
            tag=args.tag,
        )
        with open(args.output, "w", encoding="utf-8") as fout:
            exporter = chess.pgn.FileExporter(fout)
            game.accept(exporter)
        print(f"Wrote {args.output}")
        return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m scripts", description="Chess tools: engine analysis, blunders, endgame coach, SVG.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_positional_args(sp):
        sp.add_argument("--fen", help="FEN position; default startpos")
        sp.add_argument("--moves", help="Moves from startpos (SAN or UCI, space-separated)")

    # analyse
    sp = sub.add_parser("analyse", help="Multi-PV engine suggestions for a position")
    add_positional_args(sp)
    sp.add_argument("--depth", type=int, default=15)
    sp.add_argument("--multipv", type=int, default=3)
    sp.add_argument("--pv-len", type=int, default=5)
    sp.add_argument("--skill", type=int, help="Engine Skill Level 0..20")
    sp.add_argument("--threads", type=int, help="Engine Threads (auto-detected if omitted)")
    sp.add_argument("--hash", type=int, help="Engine Hash (MB) (auto-detected if omitted)")
    sp.add_argument("--contempt", type=int, help="Engine Contempt (centipawns)")
    sp.add_argument("--move-overhead", type=int, help="Engine Move Overhead (ms)")
    sp.set_defaults(func=cmd_analyse)

    # blunders
    sp = sub.add_parser("blunders", help="Detect blunders in a PGN (eval drop > threshold)")
    sp.add_argument("--pgn", required=True, help="Path to PGN file (reads first game)")
    sp.add_argument("--depth", type=int, default=14)
    sp.add_argument("--threshold", type=int, default=120)
    sp.add_argument("--skill", type=int, help="Engine Skill Level 0..20")
    sp.add_argument("--threads", type=int, help="Engine Threads (auto-detected if omitted)")
    sp.add_argument("--hash", type=int, help="Engine Hash (MB) (auto-detected if omitted)")
    sp.add_argument("--contempt", type=int, help="Engine Contempt (centipawns)")
    sp.add_argument("--move-overhead", type=int, help="Engine Move Overhead (ms)")
    sp.set_defaults(func=cmd_blunders)

    # bestmove-svg
    sp = sub.add_parser("bestmove-svg", help="Render a board with best move arrow as SVG")
    add_positional_args(sp)
    sp.add_argument("--depth", type=int, default=14)
    sp.add_argument("--out", default="best_move.svg")
    sp.add_argument("--png-out", help="Optional PNG output; requires CairoSVG")
    sp.add_argument("--skill", type=int, help="Engine Skill Level 0..20")
    sp.add_argument("--threads", type=int, help="Engine Threads (auto-detected if omitted)")
    sp.add_argument("--hash", type=int, help="Engine Hash (MB) (auto-detected if omitted)")
    sp.add_argument("--contempt", type=int, help="Engine Contempt (centipawns)")
    sp.add_argument("--move-overhead", type=int, help="Engine Move Overhead (ms)")
    sp.set_defaults(func=cmd_bestmove_svg)

    # endgame-coach
    sp = sub.add_parser("endgame-coach", help="Use Syzygy tablebases (if available) or engine to suggest endgame moves and outcomes")
    add_positional_args(sp)
    sp.add_argument("--tb-path", help="Path to Syzygy tablebases (overrides $SYZYGY_PATH)")
    sp.add_argument("--depth", type=int, default=18)
    sp.add_argument("--multipv", type=int, default=3)
    sp.add_argument("--pv-len", type=int, default=6)
    sp.add_argument("--top", type=int, default=3, help="Show top-N TB root moves")
    sp.add_argument("--skill", type=int, help="Engine Skill Level 0..20")
    sp.add_argument("--threads", type=int, help="Engine Threads (auto-detected if omitted)")
    sp.add_argument("--hash", type=int, help="Engine Hash (MB) (auto-detected if omitted)")
    sp.add_argument("--contempt", type=int, help="Engine Contempt (centipawns)")
    sp.add_argument("--move-overhead", type=int, help="Engine Move Overhead (ms)")
    sp.add_argument("--online-fallbacks", action="store_true", help="Use Lichess tablebase/cloud eval if no local TB or unsupported")
    sp.add_argument("--token", help="Lichess token (or set LICHESS_TOKEN)")
    sp.add_argument("--cache-dir", help="Cache directory for JSON responses")
    sp.add_argument("--cache-ttl", type=int, help="Cache TTL in seconds")
    sp.set_defaults(func=cmd_endgame_coach)

    # move-heatmap-svg
    sp = sub.add_parser("move-heatmap-svg", help="Evaluate all legal moves and render colored arrows by strength")
    add_positional_args(sp)
    sp.add_argument("--depth", type=int, default=12)
    sp.add_argument("--out", default="heatmap.svg")
    sp.add_argument("--png-out", help="Optional PNG output; requires CairoSVG")
    sp.add_argument("--skill", type=int, help="Engine Skill Level 0..20")
    sp.add_argument("--threads", type=int, help="Engine Threads (auto-detected if omitted)")
    sp.add_argument("--hash", type=int, help="Engine Hash (MB) (auto-detected if omitted)")
    sp.add_argument("--contempt", type=int, help="Engine Contempt (centipawns)")
    sp.add_argument("--move-overhead", type=int, help="Engine Move Overhead (ms)")
    sp.set_defaults(func=cmd_move_heatmap_svg)

    # eval-chart
    sp = sub.add_parser("eval-chart", help="Plot eval over game with blunder markers")
    sp.add_argument("--pgn", required=True)
    sp.add_argument("--depth", type=int, default=12)
    sp.add_argument("--threshold", type=int, default=120)
    sp.add_argument("--out", default="eval_chart.png")
    sp.add_argument("--csv-out", help="Optional CSV dump of (ply, cp)")
    sp.add_argument("--skill", type=int, help="Engine Skill Level 0..20")
    sp.add_argument("--threads", type=int, help="Engine Threads (auto-detected if omitted)")
    sp.add_argument("--hash", type=int, help="Engine Hash (MB) (auto-detected if omitted)")
    sp.add_argument("--contempt", type=int, help="Engine Contempt (centipawns)")
    sp.add_argument("--move-overhead", type=int, help="Engine Move Overhead (ms)")
    sp.set_defaults(func=cmd_eval_chart)

    # endgame-from-pgn
    sp = sub.add_parser("endgame-from-pgn", help="Detect endgame segment from a PGN and coach that position")
    sp.add_argument("--pgn", required=True)
    sp.add_argument("--tb-path", help="Path to Syzygy tablebases (overrides $SYZYGY_PATH)")
    sp.add_argument("--depth", type=int, default=18)
    sp.add_argument("--multipv", type=int, default=3)
    sp.add_argument("--pv-len", type=int, default=6)
    sp.add_argument("--top", type=int, default=3)
    sp.add_argument("--skill", type=int, help="Engine Skill Level 0..20")
    sp.add_argument("--threads", type=int, help="Engine Threads (auto-detected if omitted)")
    sp.add_argument("--hash", type=int, help="Engine Hash (MB) (auto-detected if omitted)")
    sp.add_argument("--contempt", type=int, help="Engine Contempt (centipawns)")
    sp.add_argument("--move-overhead", type=int, help="Engine Move Overhead (ms)")
    sp.set_defaults(func=cmd_endgame_from_pgn)

    # annotate-pgn
    sp = sub.add_parser("annotate-pgn", help="Annotate PGN with evals, blunder NAGs, and TB info")
    sp.add_argument("--input", required=True, help="Input PGN path")
    sp.add_argument("--output", required=True, help="Output PGN path")
    sp.add_argument("--depth", type=int, default=14)
    sp.add_argument("--threshold", type=int, default=120, help="Δcp for ?")
    sp.add_argument("--big-threshold", type=int, default=250, help="Δcp for ??")
    sp.add_argument("--tb-path", help="Path to Syzygy tablebases")
    sp.add_argument("--all", action="store_true", help="Process all games in the PGN")
    sp.add_argument("--variations", choices=["none", "blunders", "all"], default="blunders", help="Add engine PVs as PGN variations")
    sp.add_argument("--max-pv-len", type=int, default=6, help="Max moves in an engine PV variation")
    sp.add_argument("--prune-variations", action="store_true", help="Remove existing variations before adding engine lines")
    sp.add_argument("--tag", default="[EG_PV]", help="Tag to mark engine-inserted variations (default: [EG_PV])")
    sp.add_argument("--skill", type=int, help="Engine Skill Level 0..20")
    sp.add_argument("--threads", type=int, help="Engine Threads (auto-detected if omitted)")
    sp.add_argument("--hash", type=int, help="Engine Hash (MB) (auto-detected if omitted)")
    sp.add_argument("--contempt", type=int, help="Engine Contempt (centipawns)")
    sp.add_argument("--move-overhead", type=int, help="Engine Move Overhead (ms)")
    sp.set_defaults(func=cmd_annotate_pgn)

    # tb-scan
    sp = sub.add_parser("tb-scan", help="Scan PGN(s) to mark the first TB-covered position per game and output a CSV")
    sp.add_argument("--input", required=True, help="Input PGN path")
    sp.add_argument("--output", required=True, help="Output annotated PGN path")
    sp.add_argument("--tb-path", required=True, help="Path to Syzygy tablebases")
    sp.add_argument("--csv-out", help="Optional CSV summary output")
    def _tb_scan(args):
        import csv
        with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
            writer = None
            if args.csv_out:
                csvf = open(args.csv_out, "w", newline="", encoding="utf-8")
                writer = csv.writer(csvf)
                writer.writerow(["index", "White", "Black", "ply", "wdl", "dtz", "dtm"])  # header
            idx = 0
            with open_tb(args.tb_path) as tb:
                while True:
                    game = chess.pgn.read_game(fin)
                    if not game:
                        break
                    idx += 1
                    board = game.board()
                    node = game
                    found = None
                    ply_at = 0
                    if tb:
                        for ply_at, mv in enumerate(game.mainline_moves(), 1):
                            board.push(mv)
                            s = tb_summary(tb, board)
                            if s:
                                found = (node.variations[0] if node.variations else node).parent  # current node is before push
                                break
                            node = node.variations[0]
                    if found:
                        # Add a comment at this node indicating TB start
                        comment = found.comment.strip()
                        parts = ["TB_START"]
                        if s:
                            parts.append(f"WDL={int(s.get('wdl', 0))}")
                            if 'dtz' in s:
                                parts.append(f"DTZ={s['dtz']}")
                            if 'dtm' in s:
                                parts.append(f"DTM={s['dtm']}")
                        found.comment = (comment + (" " if comment else "") + "{" + "; ".join(parts) + "}").strip()
                        game.headers["TBStartPly"] = str(ply_at)
                        if writer:
                            writer.writerow([idx, game.headers.get("White", ""), game.headers.get("Black", ""), ply_at, int(s.get('wdl', 0)) if s else "", s.get('dtz', '' ) if s else "", s.get('dtm', '' ) if s else ""])
                    exporter = chess.pgn.FileExporter(fout)
                    game.accept(exporter)
            if args.csv_out and writer:
                csvf.close()
        print(f"Wrote {args.output}")
        if args.csv_out:
            print(f"Wrote {args.csv_out}")
        return 0
    sp.set_defaults(func=_tb_scan)

    # endgame-gui
    sp = sub.add_parser("endgame-gui", help="Simple Tkinter GUI to step through positions with TB hints and engine suggestions")
    sp.add_argument("--fen", help="FEN position to start from")
    sp.add_argument("--pgn", help="Load the first game from PGN and navigate its moves")
    sp.add_argument("--tb-path", help="Path to Syzygy tablebases (overrides $SYZYGY_PATH)")
    def _run_gui(args):
        eng_opts = _engine_opts_from_args(args)
        run_endgame_gui(fen=args.fen, pgn=args.pgn, tb_path=_cfg_tb_path(args), engine_options=eng_opts)
        return 0
    sp.set_defaults(func=_run_gui)
    sp.add_argument("--skill", type=int, help="Engine Skill Level 0..20")
    sp.add_argument("--threads", type=int, help="Engine Threads (auto-detected if omitted)")
    sp.add_argument("--hash", type=int, help="Engine Hash (MB) (auto-detected if omitted)")
    sp.add_argument("--contempt", type=int, help="Engine Contempt (centipawns)")
    sp.add_argument("--move-overhead", type=int, help="Engine Move Overhead (ms)")

    # lichess-tablebase
    sp = sub.add_parser("lichess-tablebase", help="Query Lichess tablebase (standard) for a FEN or moves")
    add_positional_args(sp)
    sp.add_argument("--fen", help="FEN position; default startpos")
    sp.add_argument("--top", type=int, default=8)
    sp.add_argument("--token", help="Lichess token (or set LICHESS_TOKEN)")
    sp.add_argument("--cache-dir", help="Cache directory for JSON responses")
    sp.add_argument("--cache-ttl", type=int, help="Cache TTL in seconds")
    def _ltb(args):
        board = load_board_from_args(args)
        ld = _lichess_defaults(args)
        data = lichess_tb(board.fen(), token=ld["token"], cache_dir=ld["cache_dir"], cache_ttl=ld["cache_ttl"])
        print(f"category: {data.get('category')}  dtz: {data.get('dtz')}  dtm: {data.get('dtm')}")
        moves = data.get("moves", [])
        for i, m in enumerate(moves[: args.top], 1):
            uci = m.get("uci")
            wdl = m.get("wdl")
            dtz = m.get("dtz")
            print(f"  {i}. {uci}  wdl={wdl} dtz={dtz}")
        return 0
    sp.set_defaults(func=_ltb)

    # lichess-cloud-eval
    sp = sub.add_parser("lichess-cloud-eval", help="Query Lichess cloud evaluation for a FEN or moves")
    add_positional_args(sp)
    sp.add_argument("--fen", help="FEN position; default startpos")
    sp.add_argument("--multi-pv", type=int, default=3)
    sp.add_argument("--token", help="Lichess token (or set LICHESS_TOKEN)")
    sp.add_argument("--cache-dir", help="Cache directory for JSON responses")
    sp.add_argument("--cache-ttl", type=int, help="Cache TTL in seconds")
    def _lce(args):
        board = load_board_from_args(args)
        ld = _lichess_defaults(args)
        data = lichess_cloud_eval(board.fen(), multi_pv=args.multi_pv, token=ld["token"], cache_dir=ld["cache_dir"], cache_ttl=ld["cache_ttl"])
        pvs = data.get("pvs", [])
        for i, pv in enumerate(pvs, 1):
            line = pv.get("moves", "").split()
            score = pv.get("cp") if "cp" in pv else (f"M{pv.get('mate')}" if pv.get('mate') is not None else "?")
            print(f"{i}. score={score} pv={' '.join(line[:12])}")
        return 0
    sp.set_defaults(func=_lce)

    # lichess-explorer
    sp = sub.add_parser("lichess-explorer", help="Query Lichess Opening Explorer (masters or lichess DB)")
    add_positional_args(sp)
    sp.add_argument("--db", choices=["masters", "lichess"], default="masters")
    sp.add_argument("--top-games", type=int, default=5)
    sp.add_argument("--recent-games", type=int, default=0)
    sp.add_argument("--speeds", help="Comma list: bullet,blitz,rapid,classical,correspondence (lichess db only)")
    sp.add_argument("--ratings", help="Comma list of ratings e.g. 2200,2400 (lichess db only)")
    sp.add_argument("--since", type=int, help="Unix millis (lichess db only)")
    sp.add_argument("--until", type=int, help="Unix millis (lichess db only)")
    sp.add_argument("--token", help="Lichess token (or set LICHESS_TOKEN)")
    sp.add_argument("--cache-dir", help="Cache directory for JSON responses")
    sp.add_argument("--cache-ttl", type=int, help="Cache TTL in seconds")
    def _lexp(args):
        board = chess.Board()
        play_tokens = []
        if args.moves:
            for tok in args.moves.split():
                try:
                    mv = board.parse_san(tok)
                except ValueError:
                    mv = chess.Move.from_uci(tok)
                    if mv not in board.legal_moves:
                        raise
                board.push(mv)
                play_tokens.append(mv.uci())
        play_csv = ",".join(play_tokens)
        ld = _lichess_defaults(args)
        data = lichess_opening_explorer(
            play_csv,
            db=args.db,
            top_games=args.top_games,
            recent_games=args.recent_games,
            speeds=args.speeds,
            ratings=args.ratings,
            since=args.since,
            until=args.until,
            token=ld["token"],
            cache_dir=ld["cache_dir"],
            cache_ttl=ld["cache_ttl"],
        )
        print(f"moves: {len(data.get('moves', []))}, topGames: {len(data.get('topGames', []))}")
        for i, mv in enumerate(data.get("moves", [])[:12], 1):
            print(f"  {i}. {mv.get('uci')} W:{mv.get('white')} D:{mv.get('draws')} B:{mv.get('black')} games:{mv.get('games')}")
        return 0
    sp.set_defaults(func=_lexp)

    # lichess-export-user
    sp = sub.add_parser("lichess-export", help="Export a user's games from Lichess to a file (PGN by default)")
    sp.add_argument("--user", required=True)
    sp.add_argument("--out", required=True, help="Output file path (.pgn or .ndjson)")
    sp.add_argument("--max", type=int, help="Max games")
    sp.add_argument("--perf", help="Perf type: blitz,rapid,classical,bullet,correspondence")
    sp.add_argument("--since", type=int, help="Unix millis since")
    sp.add_argument("--until", type=int, help="Unix millis until")
    sp.add_argument("--rated", choices=["true", "false"], help="Only rated or only casual")
    sp.add_argument("--ndjson", action="store_true", help="Write NDJSON instead of PGN")
    sp.add_argument("--token", help="Lichess token (or set LICHESS_TOKEN)")
    def _lexport(args):
        as_pgn = not args.ndjson
        rated = None
        if args.rated == "true":
            rated = True
        elif args.rated == "false":
            rated = False
        resp = lichess_export_user(
            username=args.user,
            max_games=args.max,
            perf_type=args.perf,
            since=args.since,
            until=args.until,
            as_pgn=as_pgn,
            rated=rated,
            token=args.token,
        )
        with open(args.out, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Wrote {args.out}")
        return 0
    sp.set_defaults(func=_lexport)

    # lichess-export-annotate
    sp = sub.add_parser("lichess-export-annotate", help="Export a user's games from Lichess and annotate them with evals/variations")
    sp.add_argument("--user", required=True)
    sp.add_argument("--out", required=True, help="Output annotated PGN path")
    sp.add_argument("--max", type=int, help="Max games")
    sp.add_argument("--perf", help="Perf type: blitz,rapid,classical,bullet,correspondence")
    sp.add_argument("--since", type=int, help="Unix millis since")
    sp.add_argument("--until", type=int, help="Unix millis until")
    sp.add_argument("--rated", choices=["true", "false"], help="Only rated or only casual")
    sp.add_argument("--variations", choices=["none", "blunders", "all"], default="blunders")
    sp.add_argument("--max-pv-len", type=int, default=4)
    sp.add_argument("--tag", default="[EG_PV]")
    sp.add_argument("--depth", type=int, default=12)
    sp.add_argument("--threshold", type=int, default=150)
    sp.add_argument("--big-threshold", type=int, default=300)
    sp.add_argument("--tb-path", help="Path to Syzygy tablebases")
    sp.add_argument("--token", help="Lichess token (or set LICHESS_TOKEN)")
    sp.add_argument("--blunders-csv", help="Optional CSV path to write detected blunders (from thresholds)")
    sp.add_argument("--cloud-eval-blunders", action="store_true", help="Add Lichess cloud eval score for each blunder position to the CSV (uses config defaults/token/cache)")
    sp.add_argument("--cache-dir", help="Cache directory for JSON responses (for cloud eval)")
    sp.add_argument("--cache-ttl", type=int, help="Cache TTL in seconds (for cloud eval)")
    sp.add_argument("--csv-headers", default="White,Black,ECO,Result,Date", help="Comma-separated PGN headers to include in CSV context columns")
    sp.add_argument("--eco-from-explorer", action="store_true", help="If ECO header missing, try Explorer to fill ECO for each blunder (uses cache/token defaults)")
    sp.add_argument("--skill", type=int, help="Engine Skill Level 0..20")
    sp.add_argument("--threads", type=int, help="Engine Threads (auto-detected if omitted)")
    sp.add_argument("--hash", type=int, help="Engine Hash (MB) (auto-detected if omitted)")
    sp.add_argument("--contempt", type=int, help="Engine Contempt (centipawns)")
    sp.add_argument("--move-overhead", type=int, help="Engine Move Overhead (ms)")
    def _lexport_annotate(args):
        as_pgn = True
        rated = None
        if args.rated == "true":
            rated = True
        elif args.rated == "false":
            rated = False
        resp = lichess_export_user(
            username=args.user,
            max_games=args.max,
            perf_type=args.perf,
            since=args.since,
            until=args.until,
            as_pgn=as_pgn,
            rated=rated,
            token=args.token,
        )
        # Stream response to a temp file to avoid loading whole content
        import tempfile as _tmp
        import chess.pgn as _pgn
        eng_opts = _engine_opts_from_args(args)
        bl_csv_path = getattr(args, "blunders_csv", None)
        bl_csv = None
        if bl_csv_path:
            bl_csv = open(bl_csv_path, "w", encoding="utf-8")
            # Build header row
            extra_cols = []
            if args.cloud_eval_blunders:
                extra_cols.append("cloud_score")
            # PGN headers requested
            hdr_keys = [h.strip() for h in (args.csv_headers or "").split(',') if h.strip()]
            extra_cols.extend(hdr_keys)
            bl_csv.write(
                "game_index,game_id,ply,move_uci,move_san,delta_pawns,prev_cp,new_cp,fen_before,fen_after"
                + ("," + ",".join(extra_cols) if extra_cols else "")
                + "\n"
            )
        fd, tmp_path = _tmp.mkstemp(suffix=".pgn")
        os.close(fd)
        try:
            with open(tmp_path, "wb") as _f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        _f.write(chunk)
            with open(args.out, "w", encoding="utf-8") as fout, open(tmp_path, "r", encoding="utf-8") as fin:
                gi = 0
                while True:
                    game = _pgn.read_game(fin)
                    if not game:
                        break
                    gi += 1
                    bl_list = []
                    site = game.headers.get("Site", "")
                    game_id = site
                    game = annotate_game(
                        game,
                        args.depth,
                        args.threshold,
                        args.big_threshold,
                        args.tb_path,
                        add_variations=args.variations,
                        max_pv_len=args.max_pv_len,
                        keep_existing_variations=True,
                        engine_options=eng_opts,
                        tag=args.tag,
                        blunders_list=bl_list,
                        game_id=game_id,
                    )
                    exporter = _pgn.FileExporter(fout)
                    game.accept(exporter)
                    if bl_csv is not None:
                        # Optionally compute cloud eval score for each blunder position
                        ld = _lichess_defaults(args)
                        hdr_keys = [h.strip() for h in (args.csv_headers or "").split(',') if h.strip()]
                        # Precompute UCI history for ECO lookup if requested
                        uci_hist = None
                        if args.eco_from_explorer and any(k.upper() == "ECO" for k in hdr_keys):
                            try:
                                # Build UCI history of the game mainline
                                tb = game.board()
                                uci_hist = []
                                for mv in game.mainline_moves():
                                    uci_hist.append(mv.uci())
                                    tb.push(mv)
                            except Exception:
                                uci_hist = None
                        for b in bl_list:
                            cols = []
                            cloud_col = ""
                            if args.cloud_eval_blunders:
                                try:
                                    data = lichess_cloud_eval(b['fen_after'], multi_pv=1, token=ld.get('token'), cache_dir=ld.get('cache_dir'), cache_ttl=ld.get('cache_ttl'))
                                    pvs = data.get("pvs", [])
                                    if pvs:
                                        pv0 = pvs[0]
                                        cloud_col = str(pv0.get("cp") if "cp" in pv0 else (f"M{pv0.get('mate')}" if pv0.get('mate') is not None else ""))
                                except Exception:
                                    cloud_col = ""
                            if args.cloud_eval_blunders:
                                cols.append(cloud_col)
                            # Headers for context
                            h = game.headers
                            for key in hdr_keys:
                                key_u = key.upper()
                                if key_u == "ECO" and (h.get("ECO") in (None, "")) and args.eco_from_explorer:
                                    # Attempt explorer lookup for this blunder ply
                                    try:
                                        if uci_hist:
                                            # Determine play CSV up to this blunder ply
                                            ply_idx = int(b["ply"])  # ply after applying current move index? We'll cap
                                            uci_seq = ",".join(uci_hist[: max(0, min(len(uci_hist), ply_idx))])
                                            edata = lichess_opening_explorer(uci_seq, db="masters", top_games=0, recent_games=0, token=ld.get('token'), cache_dir=ld.get('cache_dir'), cache_ttl=ld.get('cache_ttl'))
                                            eco_val = ""
                                            # Heuristics for eco/name in response
                                            if isinstance(edata, dict):
                                                if edata.get("opening") and isinstance(edata["opening"], dict):
                                                    eco_val = edata["opening"].get("eco") or ""
                                                elif "eco" in edata:
                                                    eco_val = edata.get("eco") or ""
                                            cols.append(eco_val)
                                        else:
                                            cols.append(h.get(key, ""))
                                    except Exception:
                                        cols.append(h.get(key, ""))
                                else:
                                    cols.append(h.get(key, ""))
                            bl_csv.write(
                                f"{gi},{b['game_id']},{b['ply']},{b['move']},{b['san']},{b['delta_cp']:.2f},{b['prev_cp']:.2f},{b['new_cp']:.2f},{b['fen_before']},{b['fen_after']}"
                                + ("," + ",".join(str(c) for c in cols) if cols else "")
                                + "\n"
                            )
        finally:
            if bl_csv is not None:
                bl_csv.close()
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        print(f"Wrote {args.out}")
        return 0
    sp.set_defaults(func=_lexport_annotate)

    # lichess-cache (manage cache directory)
    sp = sub.add_parser("lichess-cache", help="Manage Lichess JSON cache directory")
    sub2 = sp.add_subparsers(dest="action", required=True)
    spc = sub2.add_parser("clear", help="Clear cache directory contents")
    spc.add_argument("--cache-dir", help="Cache directory to clear (defaults to config lichess.cache_dir)")
    def _cache_clear(args):
        import shutil, os
        cfg = getattr(args, "_config", {})
        cache_dir = getattr(args, "cache_dir", None) or config_value(cfg, "lichess", "cache_dir")
        if not cache_dir:
            print("No cache directory provided or configured.")
            return 2
        if not os.path.isdir(cache_dir):
            print("Cache directory does not exist; nothing to clear.")
            return 0
        try:
            for name in os.listdir(cache_dir):
                p = os.path.join(cache_dir, name)
                if os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    try:
                        os.remove(p)
                    except Exception:
                        pass
            print(f"Cleared cache: {cache_dir}")
            return 0
        except Exception as e:
            print(f"Failed to clear cache: {e}")
            return 2
    spc.set_defaults(func=_cache_clear)
    sps = sub2.add_parser("stats", help="Show basic stats for cache directory")
    sps.add_argument("--cache-dir", help="Cache directory (defaults to config lichess.cache_dir)")
    sps.add_argument("--top", type=int, default=10, help="Show top-N largest entries")
    def _cache_stats(args):
        import os
        cfg = getattr(args, "_config", {})
        cache_dir = getattr(args, "cache_dir", None) or config_value(cfg, "lichess", "cache_dir")
        if not cache_dir or not os.path.isdir(cache_dir):
            print("No cache directory found.")
            return 2
        total = 0
        entries = []
        for name in os.listdir(cache_dir):
            p = os.path.join(cache_dir, name)
            try:
                sz = os.path.getsize(p)
                total += sz
                entries.append((sz, name))
            except Exception:
                pass
        entries.sort(reverse=True)
        print(f"Entries: {len(entries)}  Total: {total/1024:.1f} KiB")
        for sz, name in entries[: max(0, int(args.top))]:
            print(f"  {sz/1024:.1f} KiB  {name}")
        return 0
    sps.set_defaults(func=_cache_stats)

    spf = sub2.add_parser("fetch", help="Prewarm cache by fetching Lichess APIs for move sequences")
    spf.add_argument("--moves-file", help="Path to a text file; each line is a move sequence (SAN or UCI) from startpos")
    spf.add_argument("--moves-lines", help="Inline move sequences separated by ';' (e.g., 'e4 e5; d4 d5')")
    spf.add_argument("--format", choices=["san", "uci"], default="san")
    spf.add_argument("--types", default="explorer,cloud,tablebase", help="Comma list: explorer,cloud,tablebase")
    spf.add_argument("--workers", type=int, default=4, help="Parallel workers for prewarming")
    spf.add_argument("--db", choices=["masters", "lichess"], default="masters", help="Explorer DB")
    spf.add_argument("--multi-pv", type=int, default=1)
    spf.add_argument("--top-games", type=int, default=0)
    spf.add_argument("--recent-games", type=int, default=0)
    spf.add_argument("--speeds", help="Explorer speeds filter (lichess db only)")
    spf.add_argument("--ratings", help="Explorer ratings filter (lichess db only)")
    spf.add_argument("--since", type=int)
    spf.add_argument("--until", type=int)
    spf.add_argument("--cache-dir", help="Cache dir (defaults to config)")
    spf.add_argument("--cache-ttl", type=int, help="Cache TTL (defaults to config)")
    spf.add_argument("--token", help="Lichess token (defaults to config)")
    def _cache_fetch(args):
        ld = _lichess_defaults(args)
        cache_dir = ld.get("cache_dir")
        cache_ttl = ld.get("cache_ttl")
        token = ld.get("token")
        seqs = []
        if args.moves_file:
            try:
                with open(args.moves_file, "r", encoding="utf-8") as f:
                    seqs.extend([ln.strip() for ln in f if ln.strip()])
            except Exception as e:
                print(f"Failed to read moves file: {e}")
                return 2
        if args.moves_lines:
            seqs.extend([s.strip() for s in args.moves_lines.split(";") if s.strip()])
        if not seqs:
            print("No move sequences provided.")
            return 2
        types = {t.strip() for t in args.types.split(',') if t.strip()}
        from concurrent.futures import ThreadPoolExecutor, as_completed
        def work(line: str) -> bool:
            try:
                board = chess.Board()
                tokens = [t for t in line.split() if t]
                play_tokens = []
                for tok in tokens:
                    if args.format == "san":
                        mv = board.parse_san(tok)
                    else:
                        mv = chess.Move.from_uci(tok)
                        if mv not in board.legal_moves:
                            raise ValueError(f"Illegal move: {tok}")
                    play_tokens.append(mv.uci())
                    board.push(mv)
                fen = board.fen()
                uci_csv = ",".join(play_tokens)
                if "explorer" in types:
                    try:
                        _ = lichess_opening_explorer(
                            uci_csv, db=args.db, top_games=args.top_games, recent_games=args.recent_games,
                            speeds=args.speeds, ratings=args.ratings, since=args.since, until=args.until,
                            token=token, cache_dir=cache_dir, cache_ttl=cache_ttl,
                        )
                    except Exception:
                        pass
                if "cloud" in types:
                    try:
                        _ = lichess_cloud_eval(fen, multi_pv=args.multi_pv, token=token, cache_dir=cache_dir, cache_ttl=cache_ttl)
                    except Exception:
                        pass
                if "tablebase" in types:
                    try:
                        _ = lichess_tb(fen, token=token, cache_dir=cache_dir, cache_ttl=cache_ttl)
                    except Exception:
                        pass
                return True
            except Exception as e:
                print(f"Skip '{line}': {e}")
                return False
        ok = 0
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
            futs = [ex.submit(work, ln) for ln in seqs]
            for fut in as_completed(futs):
                ok += 1 if fut.result() else 0
        print(f"Warmed cache for {ok}/{len(seqs)} sequences")
        return 0
    spf.set_defaults(func=_cache_fetch)
    # pgn-insert-pv (headless)
    sp = sub.add_parser("pgn-insert-pv", help="Insert a PV as a variation at the node matching a FEN across games")
    sp.add_argument("--input", required=True, help="Input PGN path")
    sp.add_argument("--output", required=True, help="Output PGN path")
    target_group = sp.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--fen", help="Target FEN to match (position where to insert)")
    target_group.add_argument("--from-moves-uci", help="Build target position by applying UCI moves from startpos")
    target_group.add_argument("--from-moves-san", help="Build target position by applying SAN moves from startpos")
    pv_group = sp.add_mutually_exclusive_group(required=True)
    pv_group.add_argument("--pv-uci", help="PV in UCI format, space-separated")
    pv_group.add_argument("--pv-san", help="PV in SAN format, space-separated")
    pv_group.add_argument("--pv-file", help="Path to a text file containing PV tokens (whitespace-separated)")
    sp.add_argument("--pv-format", choices=["san", "uci"], help="Format for --pv-file contents (optional; auto-detected if omitted)")
    sp.add_argument("--pv-len", type=int, default=6, help="Max PV moves to insert")
    sp.add_argument("--replace", action="store_true", help="Replace existing GUI-inserted PVs at that node")
    sp.add_argument("--replace-if-first-matches", action="store_true", help="When replacing, only remove PVs whose first move matches new PV's first move")
    sp.add_argument("--mainline-only", action="store_true", help="Search only along mainline for the matching FEN")
    sp.add_argument("--tag", default="[EG_PV]", help="Marker tag to identify inserted variations (default: [EG_PV])")
    # Build target position from PGN + ply
    sp.add_argument("--from-pgn", help="PGN file to derive target position from")
    sp.add_argument("--from-pgn-index", type=int, default=1, help="1-based game index within --from-pgn (default: 1)")
    sp.add_argument("--ply", type=int, help="Ply index within the selected game to derive the target position")
    def _pgn_insert(args):
        import chess.pgn
        # Determine target FEN or derive from moves/PGN
        if args.fen:
            target_fen = args.fen
        else:
            target_fen = None

        if not target_fen:
            if args.from_moves_uci:
                b = chess.Board()
                try:
                    for tok in args.from_moves_uci.split():
                        mv = chess.Move.from_uci(tok)
                        if mv not in b.legal_moves:
                            raise ValueError(f"Illegal move in sequence: {tok}")
                        b.push(mv)
                except Exception as e:
                    print(f"Invalid --from-moves-uci: {e}", file=sys.stderr)
                    return 2
                target_fen = b.fen()
            elif args.from_moves_san:
                b = chess.Board()
                try:
                    for tok in args.from_moves_san.split():
                        mv = b.parse_san(tok)
                        b.push(mv)
                except Exception as e:
                    print(f"Invalid --from-moves-san: {e}", file=sys.stderr)
                    return 2
                target_fen = b.fen()
            elif args.from_pgn and args.ply is not None:
                try:
                    # Read the Nth game (1-based)
                    g = None
                    with open(args.from_pgn, "r", encoding="utf-8") as pf:
                        idx = 0
                        while True:
                            gg = chess.pgn.read_game(pf)
                            if not gg:
                                break
                            idx += 1
                            if idx == max(1, int(args.from_pgn_index)):
                                g = gg
                                break
                    if g is None:
                        print("--from-pgn index out of range or file empty", file=sys.stderr)
                        return 2
                    b = g.board()
                    cur_ply = 0
                    if args.ply == 0:
                        target_fen = b.fen()
                    else:
                        for mv in g.mainline_moves():
                            b.push(mv)
                            cur_ply += 1
                            if cur_ply == args.ply:
                                break
                        if cur_ply != args.ply:
                            print("--ply exceeds game length", file=sys.stderr)
                            return 2
                        target_fen = b.fen()
                except Exception as e:
                    print(f"Invalid --from-pgn/--ply: {e}", file=sys.stderr)
                    return 2
            else:
                print("Must provide --fen or --from-moves-uci/--from-moves-san", file=sys.stderr)
                return 2
        # Normalize fen (ignore halfmove/fullmove counters for matching)
        def norm_fen(f):
            try:
                parts = f.split()
                return " ".join(parts[:4])
            except Exception:
                return f

        target_core = norm_fen(target_fen)

        def parse_pv_san(board, san_str):
            moves = []
            b = board.copy()
            for tok in san_str.split():
                mv = b.parse_san(tok)
                moves.append(mv)
                b.push(mv)
            return moves

        def parse_pv_uci(board, uci_str):
            moves = []
            b = board.copy()
            for tok in uci_str.split():
                mv = chess.Move.from_uci(tok)
                if mv not in b.legal_moves:
                    raise ValueError(f"Illegal move in PV for this node: {tok}")
                moves.append(mv)
                b.push(mv)
            return moves

        inserted = 0
        total = 0
        with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
            while True:
                game = chess.pgn.read_game(fin)
                if not game:
                    break
                total += 1
                board = game.board()
                found = None

                if args.mainline_only:
                    node = game
                    if norm_fen(board.fen()) == target_core:
                        found = node
                    else:
                        while node.variations:
                            next_node = node.variations[0]
                            board.push(next_node.move)
                            if norm_fen(board.fen()) == target_core:
                                found = node
                                break
                            node = next_node
                else:
                    def dfs(n, b):
                        nonlocal found
                        if found is not None:
                            return
                        if norm_fen(b.fen()) == target_core:
                            found = n
                            return
                        for child in n.variations:
                            nb = b.copy()
                            nb.push(child.move)
                            dfs(child, nb)
                    dfs(game, game.board())

                if found:
                    # Build PV from the found node's board context
                    ctx_board = found.board()
                    try:
                        if args.pv_file:
                            with open(args.pv_file, "r", encoding="utf-8") as pf:
                                contents = pf.read()
                            fmt = args.pv_format
                            if not fmt:
                                # Auto-detect: try SAN, then UCI
                                try:
                                    pv_moves = parse_pv_san(ctx_board, contents)
                                    fmt = "san"
                                except Exception:
                                    pv_moves = parse_pv_uci(ctx_board, contents)
                                    fmt = "uci"
                            else:
                                if fmt == "san":
                                    pv_moves = parse_pv_san(ctx_board, contents)
                                else:
                                    pv_moves = parse_pv_uci(ctx_board, contents)
                        elif args.pv_san:
                            pv_moves = parse_pv_san(ctx_board, args.pv_san)
                        else:
                            pv_moves = parse_pv_uci(ctx_board, args.pv_uci)
                    except Exception as e:
                        # Skip this game if PV cannot be parsed in its context
                        exporter = chess.pgn.FileExporter(fout)
                        game.accept(exporter)
                        continue
                    pv_moves = pv_moves[: max(1, args.pv_len)]

                    if args.replace:
                        marker = args.tag
                        new_first = pv_moves[0]
                        def keep_var(var_node):
                            if marker not in (var_node.comment or ""):
                                return True
                            if args.replace_if_first_matches:
                                mv = getattr(var_node, 'move', None)
                                return mv != new_first
                            return False
                        try:
                            found.variations = [v for v in list(found.variations) if keep_var(v)]
                        except Exception:
                            kept = []
                            for v in list(found.variations):
                                if keep_var(v):
                                    kept.append(v)
                            try:
                                found.variations = kept
                            except Exception:
                                pass

                    # Insert variation and tag
                    var_node = found
                    first_child = None
                    for mv in pv_moves:
                        var_node = var_node.add_variation(mv)
                        if first_child is None:
                            first_child = var_node
                    if first_child is not None:
                        c = (first_child.comment or "").strip()
                        first_child.comment = (c + (" " if c else "") + args.tag).strip()
                    inserted += 1

                exporter = chess.pgn.FileExporter(fout)
                game.accept(exporter)

        print(f"Inserted into {inserted}/{total} game(s)")
        return 0
    sp.set_defaults(func=_pgn_insert)

    return p


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Attach config for defaults
    args._config = load_config()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
