import os
import shutil
from contextlib import contextmanager
from typing import Optional, List
import platform
import subprocess

import chess
import chess.engine

from .config import load_config, config_value


def resolve_stockfish_path() -> str:
    """Return a likely Stockfish path.

    Order:
    1) .chessrc.json: stockfish_path
    2) $STOCKFISH env
    3) PATH/common locations
    """
    try:
        cfg = load_config()
        cfg_path = config_value(cfg, "stockfish_path")
        if cfg_path and (shutil.which(cfg_path) or os.path.exists(cfg_path)):
            return cfg_path
    except Exception:
        pass

    env = os.getenv("STOCKFISH")
    if env and (shutil.which(env) or os.path.exists(env)):
        return env

    # Common macOS/Homebrew and Linux locations
    candidates = [
        "stockfish",
        "/opt/homebrew/bin/stockfish",
        "/usr/local/bin/stockfish",
        "/usr/bin/stockfish",
    ]
    for c in candidates:
        if shutil.which(c) or os.path.exists(c):
            return c
    # Fallback to plain name; engine open may fail with a clear error
    return "stockfish"


@contextmanager
def get_engine(path: Optional[str] = None, options: Optional[dict] = None):
    """Context manager for a configured UCI engine (Stockfish).

    Args:
        path: Optional path to the engine binary. Defaults to resolve_stockfish_path().
        options: Optional dict of UCI options, e.g. {"Skill Level": 10, "Threads": 2}.
    """
    engine_path = path or resolve_stockfish_path()
    eng = chess.engine.SimpleEngine.popen_uci(engine_path)
    try:
        if options:
            eng.configure(options)
        yield eng
    finally:
        try:
            eng.quit()
        except Exception:
            pass


def score_to_str(score: chess.engine.PovScore) -> str:
    """Format a PovScore as a compact string from the given POV.

    Pass score.pov(side) to express from side-to-move perspective.
    """
    s = score
    if s.is_mate():
        m = s.mate()
        if m is None:
            return "M?"
        return f"M{m}" if m > 0 else f"M{-m}"
    cp = s.score(mate_score=100000)
    if cp is None:
        return "?"
    return f"{cp/100:.2f}"


def analyse_multipv(
    board: chess.Board,
    depth: int = 15,
    multipv: int = 3,
    engine_options: Optional[dict] = None,
) -> List[dict]:
    """Run a multi-PV analysis and return list of infos.

    Each info includes keys like: 'pv' (list[moves]), 'score' (PovScore), 'nodes', etc.
    """
    with get_engine(options=engine_options) as eng:
        info_list = eng.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
    # Normalize to list (python-chess may return a dict when multipv=1)
    if isinstance(info_list, dict):
        info_list = [info_list]
    # sort by PV rank if present
    info_list = sorted(info_list, key=lambda i: i.get("multipv", 1))
    return info_list


def set_strength(level: int) -> dict:
    """Return Stockfish UCI options for a given strength level 0..20."""
    level = max(0, min(20, level))
    return {
        "Skill Level": level,
    }


def build_engine_options(
    skill: Optional[int] = None,
    threads: Optional[int] = None,
    hash_mb: Optional[int] = None,
    contempt: Optional[int] = None,
    move_overhead: Optional[int] = None,
    use_defaults: bool = True,
) -> dict:
    """Construct a dict of Stockfish UCI options from optional parameters."""
    opts: dict = {}
    if skill is not None:
        opts.update(set_strength(skill))
    if threads is None and use_defaults:
        threads = detect_default_threads()
    if hash_mb is None and use_defaults:
        hash_mb = detect_default_hash_mb()
    if threads is not None and threads > 0:
        opts["Threads"] = int(threads)
    if hash_mb is not None and hash_mb > 0:
        opts["Hash"] = int(hash_mb)
    if contempt is not None:
        opts["Contempt"] = int(contempt)
    if move_overhead is not None and move_overhead >= 0:
        opts["Move Overhead"] = int(move_overhead)
    return opts


def detect_default_threads() -> int:
    """Return a sensible default for threads (clamped)."""
    try:
        cpu = os.cpu_count() or 1
    except Exception:
        cpu = 1
    # Avoid oversubscription; clamp between 1 and 8
    return max(1, min(cpu, 8))


def _get_total_ram_bytes_unix() -> Optional[int]:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        return int(page_size) * int(phys_pages)
    except Exception:
        return None


def _get_total_ram_bytes_macos() -> Optional[int]:
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        return int(out)
    except Exception:
        return None


def _get_total_ram_bytes_windows() -> Optional[int]:
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        c_ulonglong = ctypes.c_ulonglong
        mem_kb = c_ulonglong(0)
        if hasattr(kernel32, 'GetPhysicallyInstalledSystemMemory') and kernel32.GetPhysicallyInstalledSystemMemory(ctypes.byref(mem_kb)):
            return int(mem_kb.value) * 1024
    except Exception:
        return None
    return None


def detect_total_ram_mb() -> Optional[int]:
    """Best-effort detection of total RAM in MB without extra deps."""
    try:
        system = platform.system().lower()
        if system == "darwin":
            v = _get_total_ram_bytes_macos() or _get_total_ram_bytes_unix()
        elif system == "linux":
            v = _get_total_ram_bytes_unix()
        elif system == "windows":
            v = _get_total_ram_bytes_windows()
        else:
            v = _get_total_ram_bytes_unix()
        if v is None:
            return None
        return max(256, int(v // (1024 * 1024)))
    except Exception:
        return None


def detect_default_hash_mb() -> int:
    """Pick a conservative default hash size in MB.

    Use roughly 1/8th of total RAM, clamped to [128, 1024].
    """
    total = detect_total_ram_mb() or 2048
    proposed = max(128, min(1024, int(total / 8)))
    return proposed
