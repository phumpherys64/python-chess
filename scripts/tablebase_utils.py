import os
from contextlib import contextmanager
from typing import Optional, Tuple, List

import chess
import chess.syzygy


def resolve_tb_path() -> Optional[str]:
    """Return a Syzygy tablebase folder path if available via env.

    Looks for $SYZYGY_PATH or $TB_PATH.
    """
    for name in ("SYZYGY_PATH", "TB_PATH"):
        p = os.getenv(name)
        if p and os.path.isdir(p):
            return p
    return None


@contextmanager
def open_tb(path: Optional[str] = None):
    """Context manager that yields a chess.syzygy.Tablebase or None."""
    resolved = path or resolve_tb_path()
    if not resolved:
        yield None
        return
    tb = chess.syzygy.open_tablebase(resolved)
    try:
        yield tb
    finally:
        tb.close()


def tb_summary(tb: chess.syzygy.Tablebase, board: chess.Board) -> Optional[dict]:
    """Return a summary dict with WDL/DTZ/DTM and best moves if supported.

    Returns None if position not supported by available tablebases.
    """
    try:
        wdl = tb.probe_wdl(board)
    except Exception:
        return None

    res = {
        "wdl": int(wdl),  # 2 win, 1 cursed win, 0 draw, -1 blessed loss, -2 loss
    }
    try:
        res["dtz"] = tb.probe_dtz(board)
    except Exception:
        pass
    try:
        res["dtm"] = tb.probe_dtm(board)
    except Exception:
        pass

    # Best root moves via WDL/DTZ probing
    try:
        root = tb.probe_root(board)
        # probe_root returns list of (move, wdl, dtz)
        res["root"] = [
            {
                "move": mv,
                "wdl": int(m_wdl),
                "dtz": m_dtz,
            }
            for mv, m_wdl, m_dtz in root
        ]
    except Exception:
        pass
    return res

