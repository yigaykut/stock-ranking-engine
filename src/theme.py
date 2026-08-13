"""Gorsel tema: karanlik vaporwave + kutsal geometri.

Arka plandaki egriler dekoratif clip-art degil — gercek parametrik egriler
(hipotrokoid / epitrokoid ve Lissajous figurleri) burada hesaplanip SVG
yoluna cevriliyor. Ayni parametreler her uretimde ayni cizimi verir.
"""
from __future__ import annotations

import math

# --- Palet -----------------------------------------------------------------
# Kategorik seriler validate_palette.js ile #120609 zemine karsi dogrulandi:
# aydinlik bandi, kroma tabani, CVD ayrimi, normal gorus tabani ve kontrast PASS.
SURFACE = "#120609"
PLANE = "#0a0406"
CRIMSON = "#f0483a"
STEEL = "#3d6b8a"

SERIES = [
    "#f0483a",  # crimson    — momentum
    "#1a9fb8",  # cyan       — analist
    "#b8871f",  # gold       — teknik
    "#dc4a8a",  # magenta    — kalite
    "#4a82d8",  # azure      — degerleme
    "#1ba372",  # jade       — buyume
    "#8f6ef0",  # violet     — risk
    "#d66c22",  # amber      — diger
]


def _hypotrochoid(R: float, r: float, d: float, turns: int = 1,
                  steps: int = 320, cx: float = 0, cy: float = 0,
                  scale: float = 1.0) -> str:
    """Spirograf egrisi -> SVG path 'd' dizesi."""
    g = math.gcd(int(R), int(r)) or 1
    period = 2 * math.pi * (r // g if r >= 1 else 1) * turns
    pts = []
    k = (R - r) / r if r else 1
    for i in range(steps + 1):
        t = period * i / steps
        x = (R - r) * math.cos(t) + d * math.cos(k * t)
        y = (R - r) * math.sin(t) - d * math.sin(k * t)
        pts.append(f"{cx + x * scale:.2f},{cy + y * scale:.2f}")
    return "M" + "L".join(pts)


def _lissajous(a: float, b: float, delta: float, amp_x: float, amp_y: float,
               cx: float, cy: float, steps: int = 280) -> str:
    pts = []
    for i in range(steps + 1):
        t = 2 * math.pi * i / steps
        x = amp_x * math.sin(a * t + delta)
        y = amp_y * math.sin(b * t)
        pts.append(f"{cx + x:.2f},{cy + y:.2f}")
    return "M" + "L".join(pts)


def hero_svg(w: int = 1600, h: int = 640) -> str:
    """Basliktaki tam genislik gorsel: kubbe kafesi + parcacik izleri + zemin izgarasi."""
    cx, horizon = w / 2, h * 0.66

    parts: list[str] = []

    # --- kubbe kafesi (eliptik yaylar) ---------------------------------------
    dome = []
    for i in range(1, 9):
        rx, ry = w * 0.085 * i, h * 0.100 * i
        dome.append(f'<ellipse cx="{cx}" cy="{horizon}" rx="{rx:.1f}" ry="{ry:.1f}"/>')
    for i in range(13):
        ang = math.pi * i / 12
        x = cx + w * 0.72 * math.cos(ang)
        y = horizon - h * 0.80 * math.sin(ang)
        dome.append(f'<path d="M{cx},{horizon} Q{(cx+x)/2:.1f},{y*0.55:.1f} {x:.1f},{y:.1f}"/>')
    parts.append(f'<g class="dome">{"".join(dome)}</g>')

    # --- parcacik izleri: spirograf demeti ------------------------------------
    ty = horizon - h * 0.44
    curves = [
        _hypotrochoid(220, 63, 148, cx=cx, cy=ty, scale=0.62),
        _hypotrochoid(180, 44, 120, cx=cx - w * 0.235, cy=ty - h * 0.03, scale=0.42),
        _hypotrochoid(150, 37, 105, cx=cx + w * 0.245, cy=ty - h * 0.02, scale=0.44),
        _hypotrochoid(96, 25, 88, cx=cx, cy=ty, scale=0.95),
        _lissajous(5, 4, math.pi / 2, w * 0.235, h * 0.15, cx, ty),
        _lissajous(7, 6, math.pi / 3, w * 0.145, h * 0.10, cx - w * 0.125, ty - h * 0.02),
        _lissajous(3, 2, math.pi / 4, w * 0.115, h * 0.08, cx + w * 0.135, ty + h * 0.01),
    ]
    parts.append('<g class="traces">' +
                 "".join(f'<path d="{d}"/>' for d in curves) + "</g>")

    # --- zemin izgarasi (perspektif) -----------------------------------------
    floor = []
    for i in range(-16, 17):
        x_far = cx + i * (w * 0.030)
        x_near = cx + i * (w * 0.135)
        floor.append(f'<line x1="{x_far:.1f}" y1="{horizon:.1f}" x2="{x_near:.1f}" y2="{h}"/>')
    for i in range(1, 15):
        t = (i / 14) ** 2.4
        y = horizon + (h - horizon) * t
        floor.append(f'<line x1="0" y1="{y:.1f}" x2="{w}" y2="{y:.1f}"/>')
    parts.append(f'<g class="floor">{"".join(floor)}</g>')

    # --- silüet siradaglar ----------------------------------------------------
    ridge = [f"M0,{horizon+2}"]
    peaks = [(0.00, .00), (0.06, .07), (0.11, .03), (0.17, .12), (0.23, .05),
             (0.29, .02), (0.36, .00), (0.44, .01), (0.50, .00), (0.57, .01),
             (0.64, .00), (0.71, .04), (0.78, .11), (0.85, .05), (0.92, .09),
             (1.00, .02)]
    for px, ph in peaks:
        ridge.append(f"L{px*w:.0f},{horizon - ph*h:.1f}")
    ridge.append(f"L{w},{h}L0,{h}Z")
    parts.append(f'<path class="ridge" d="{"".join(ridge)}"/>')

    return (f'<svg class="hero-svg" viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMax slice" '
            f'aria-hidden="true">{"".join(parts)}</svg>')


def sigil_svg(size: int = 120) -> str:
    """Kucuk kutsal-geometri muhru (bolum isaretcisi)."""
    c = size / 2
    r = size * 0.22
    circles = [f'<circle cx="{c}" cy="{c}" r="{r:.1f}"/>']
    for i in range(6):
        a = math.pi * i / 3
        circles.append(f'<circle cx="{c + r*math.cos(a):.1f}" cy="{c + r*math.sin(a):.1f}" r="{r:.1f}"/>')
    circles.append(f'<circle cx="{c}" cy="{c}" r="{r*2:.1f}"/>')
    return (f'<svg class="sigil" viewBox="0 0 {size} {size}" aria-hidden="true">'
            f'{"".join(circles)}</svg>')
