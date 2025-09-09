import json
import os
from typing import Any, Dict, Optional


DEFAULT_CONFIG_NAME = ".chessrc.json"


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_config(cwd: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from CWD and home.

    Search order:
      1) {cwd}/.chessrc.json
      2) ~/.chessrc.json
    Returns a dict with possible keys:
      - syzygy_path
      - stockfish_path
      - engine: { threads, hash, skill, contempt, move_overhead }
      - lichess: { token, cache_dir, cache_ttl }
    Missing keys are simply absent.
    """
    cfg: Dict[str, Any] = {}
    cwd = cwd or os.getcwd()
    for path in (
        os.path.join(cwd, DEFAULT_CONFIG_NAME),
        os.path.join(os.path.expanduser("~"), DEFAULT_CONFIG_NAME),
    ):
        data = _read_json(path)
        if not data:
            continue
        # Shallow merge; later files override earlier ones
        for k, v in data.items():
            cfg[k] = v
    return cfg


def config_value(cfg: Dict[str, Any], *keys: str, default=None):
    """Get nested value like config_value(cfg, 'engine', 'threads', default=None)."""
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur
