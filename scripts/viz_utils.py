from typing import Tuple


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def red_green_hex(t: float) -> str:
    """Map 0..1 to a red→yellow→green gradient hex color."""
    t = clamp01(t)
    # Simple two-segment gradient: red->yellow (0..0.5), yellow->green (0.5..1)
    if t < 0.5:
        # red (255,0,0) to yellow (255,255,0)
        g = int(lerp(0, 255, t / 0.5))
        r = 255
        b = 0
    else:
        # yellow (255,255,0) to green (0,255,0)
        g = 255
        r = int(lerp(255, 0, (t - 0.5) / 0.5))
        b = 0
    return f"#{r:02x}{g:02x}{b:02x}"


def try_svg_to_png(svg_str: str, out_path: str) -> Tuple[bool, str]:
    """Attempt to write PNG from SVG using CairoSVG. Returns (ok, message)."""
    try:
        import cairosvg  # type: ignore
    except Exception as e:
        return False, f"CairoSVG not available: {e}"
    try:
        cairosvg.svg2png(bytestring=svg_str.encode("utf-8"), write_to=out_path)
        return True, f"Wrote {out_path}"
    except Exception as e:
        return False, f"PNG export failed: {e}"

