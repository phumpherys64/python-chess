import os
import time
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests


DEFAULT_TIMEOUT = 15


def _headers(token: Optional[str] = None) -> Dict[str, str]:
    tok = token or os.getenv("LICHESS_TOKEN")
    hdrs = {"Accept": "application/json"}
    if tok:
        hdrs["Authorization"] = f"Bearer {tok}"
    return hdrs


def _cache_key(url: str, params: Optional[dict]) -> str:
    raw = url + "?" + json.dumps(params or {}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _cache_paths(cache_dir: Optional[str], key: str) -> Tuple[Optional[Path], Optional[Path]]:
    if not cache_dir:
        return None, None
    base = Path(cache_dir)
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{key}.json", base / f"{key}.meta"


def _get(
    url: str,
    params: Optional[dict] = None,
    token: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
    cache_dir: Optional[str] = None,
    cache_ttl: Optional[int] = None,
) -> requests.Response:
    # Cache lookup
    key = _cache_key(url, params)
    cache_path, meta_path = _cache_paths(cache_dir, key)
    if cache_path and cache_path.exists() and cache_ttl and cache_ttl > 0:
        try:
            age = time.time() - cache_path.stat().st_mtime
            if age <= cache_ttl:
                # Return a synthetic Response
                class _R:
                    def __init__(self, text):
                        self._text = text
                        self.status_code = 200
                    def raise_for_status(self):
                        return None
                    def json(self):
                        return json.loads(self._text)
                with cache_path.open("r", encoding="utf-8") as f:
                    txt = f.read()
                return _R(txt)  # type: ignore
        except Exception:
            pass
    # Minimal retry/backoff for 429/5xx
    backoffs = [0.5, 1.0, 2.0]
    last_exc: Optional[Exception] = None
    for i in range(len(backoffs) + 1):
        try:
            r = requests.get(url, params=params, headers=_headers(token), timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504) and i < len(backoffs):
                time.sleep(backoffs[i])
                continue
            r.raise_for_status()
            # Write cache
            if cache_path:
                try:
                    with cache_path.open("w", encoding="utf-8") as f:
                        f.write(r.text)
                    if meta_path:
                        with meta_path.open("w", encoding="utf-8") as mf:
                            json.dump({"url": url, "params": params, "ts": time.time()}, mf)
                except Exception:
                    pass
            return r
        except Exception as e:
            last_exc = e
            if i < len(backoffs):
                time.sleep(backoffs[i])
            else:
                raise
    assert False, last_exc


def tablebase_standard(fen: str, token: Optional[str] = None, cache_dir: Optional[str] = None, cache_ttl: Optional[int] = None) -> Dict[str, Any]:
    url = "https://tablebase.lichess.ovh/standard"
    r = _get(url, {"fen": fen}, token, cache_dir=cache_dir, cache_ttl=cache_ttl)
    return r.json()


def cloud_eval(fen: str, multi_pv: int = 3, token: Optional[str] = None, cache_dir: Optional[str] = None, cache_ttl: Optional[int] = None) -> Dict[str, Any]:
    url = "https://lichess.org/api/cloud-eval"
    r = _get(url, {"fen": fen, "multiPv": multi_pv}, token, cache_dir=cache_dir, cache_ttl=cache_ttl)
    return r.json()


def opening_explorer(
    play_csv: str,
    db: str = "masters",  # masters | lichess
    top_games: int = 0,
    recent_games: int = 0,
    speeds: Optional[str] = None,  # e.g., "classical,rapid"
    ratings: Optional[str] = None,  # e.g., "2200,2400"
    since: Optional[int] = None,
    until: Optional[int] = None,
    token: Optional[str] = None,
    cache_dir: Optional[str] = None,
    cache_ttl: Optional[int] = None,
) -> Dict[str, Any]:
    base = "https://explorer.lichess.ovh"
    if db == "masters":
        url = f"{base}/masters"
    else:
        url = f"{base}/lichess"
    params: Dict[str, Any] = {
        "play": play_csv,
        "topGames": top_games,
        "recentGames": recent_games,
    }
    if speeds:
        params["speeds"] = speeds
    if ratings:
        params["ratings"] = ratings
    if since:
        params["since"] = since
    if until:
        params["until"] = until
    r = _get(url, params, token, cache_dir=cache_dir, cache_ttl=cache_ttl)
    return r.json()


def export_user_games(
    username: str,
    max_games: Optional[int] = None,
    perf_type: Optional[str] = None,
    since: Optional[int] = None,
    until: Optional[int] = None,
    as_pgn: bool = True,
    rated: Optional[bool] = None,
    token: Optional[str] = None,
) -> requests.Response:
    """Return the Response stream for user games export.

    The response may be PGN text or NDJSON depending on headers/params.
    Here we default to PGN text for simplicity.
    """
    url = f"https://lichess.org/api/games/user/{username}"
    params: Dict[str, Any] = {}
    if max_games:
        params["max"] = int(max_games)
    if perf_type:
        params["perfType"] = perf_type
    if since:
        params["since"] = since
    if until:
        params["until"] = until
    if rated is not None:
        params["rated"] = str(rated).lower()
    headers = _headers(token)
    headers["Accept"] = "application/x-chess-pgn" if as_pgn else "application/x-ndjson"
    # Stream to allow writing to file without loading all in memory
    r = requests.get(url, params=params, headers=headers, timeout=DEFAULT_TIMEOUT, stream=True)
    r.raise_for_status()
    return r
