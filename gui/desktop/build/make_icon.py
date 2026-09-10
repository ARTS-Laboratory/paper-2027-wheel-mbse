"""Draw the app icon, from the shipped genome, with the project's own geometry kernel.

WHY GENERATE IT RATHER THAN DRAW ONE.  `gui/preview.py` already turns a genome into exact
polygons -- the same `generate_bezier_centerline -> thicken_3taper_curve -> place_sector`
chain the mesher and the STEP exporter build the real part from -- so the icon can be the
wheel this repo actually converged on instead of a picture of a wheel.  Re-run it after a
promotion and the icon follows the part.

    .venv-opt/bin/python gui/desktop/build/make_icon.py

STDLIB PNG, because the alternative is putting Pillow in an env whose two requirements
files are argued over at length, to write one file that then sits in git.  A PNG is a
zlib stream of filtered scanlines and `zlib` and `struct` are both stdlib; the whole
encoder below is nine lines.

ANTI-ALIASED BY SUPERSAMPLING at 3x and boxing down, which is the cheapest way to get a
clean edge out of a scanline fill.  At 512 px the spokes are ~4 px wide and an aliased
edge on them is the difference between a dock icon and a screenshot of one.
"""

import os
import struct
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
GUI = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, GUI)

import preview                                                     # noqa: E402

SIZE = 512
SS = 3                       # supersampling factor
N = SIZE * SS

# On palette with static/style.css: ink ground, paper part, one accent on the rim.
GROUND = (0x1a, 0x1a, 0x18)
PART = (0xfa, 0xf9, 0xf7)
ACCENT = (0x7f, 0xa8, 0xcf)     # the dark-theme accent: it has to hold against the ground
RADIUS_FRAC = 0.2237            # the macOS squircle corner, near enough on a circle arc


def fill(cover, polys, value):
    """Even-odd scanline fill of `polys` into `cover`, one byte a pixel.

    Even-odd rather than nonzero because an annulus is drawn here as two circles in the
    same shape, and parity is what makes the inner one a hole without caring which way
    round it was wound.
    """
    edges = []
    for poly in polys:
        for i in range(len(poly)):
            x0, y0 = poly[i]
            x1, y1 = poly[(i + 1) % len(poly)]
            if y0 != y1:
                edges.append((y0, y1, x0, x1))
    if not edges:
        return
    ylo = max(0, int(min(min(e[0], e[1]) for e in edges)))
    yhi = min(N - 1, int(max(max(e[0], e[1]) for e in edges)) + 1)
    for y in range(ylo, yhi + 1):
        yc = y + 0.5
        xs = []
        for y0, y1, x0, x1 in edges:
            if (y0 <= yc < y1) or (y1 <= yc < y0):
                xs.append(x0 + (yc - y0) * (x1 - x0) / (y1 - y0))
        xs.sort()
        row = y * N
        for i in range(0, len(xs) - 1, 2):
            a = max(0, int(xs[i] + 0.5))
            b = min(N, int(xs[i + 1] + 0.5))
            if b > a:
                cover[row + a:row + b] = bytes([value]) * (b - a)


def circle(cx, cy, r, n=512):
    from math import cos, sin, pi
    return [(cx + r * cos(2 * pi * i / n), cy + r * sin(2 * pi * i / n)) for i in range(n)]


def rounded_square(n, r, steps=64):
    from math import cos, sin, pi
    pts = []
    for cx, cy, a0 in ((n - r, n - r, 0.0), (r, n - r, pi / 2),
                       (r, r, pi), (n - r, r, 1.5 * pi)):
        for i in range(steps + 1):
            a = a0 + (pi / 2) * i / steps
            pts.append((cx + r * cos(a), cy + r * sin(a)))
    return pts


def write_png(path, rgba):
    """`rgba` is SIZE rows of SIZE (r, g, b, a). Filter type 0 on every scanline.

    RGBA and not RGB: the tile's rounded corners have to be genuinely transparent or the
    icon is a square with a slightly different square drawn inside it.  Colour type 6.
    """
    raw = b"".join(b"\x00" + bytes(v for px in row for v in px) for row in rgba)
    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", SIZE, SIZE, 8, 6, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9))
           + chunk(b"IEND", b""))
    with open(path, "wb") as fh:
        fh.write(png)


def main():
    from math import cos, sin, radians
    o = preview.outline(preview.load_genes())
    c = o["constants"]
    span = c["rim_outer_radius_mm"] * 2.06      # a hair of margin inside the tile
    scale = N / span

    def to_px(p):
        # y flips: model y is up, raster y is down.
        return (N / 2 + p[0] * scale, N / 2 - p[1] * scale)

    def rotated(poly, deg):
        a = radians(deg)
        ca, sa = cos(a), sin(a)
        return [to_px((x * ca - y * sa, x * sa + y * ca)) for x, y in poly]

    # 1 = part, 2 = accent. Painted in that order so the rim ring reads on top.
    cover = bytearray(N * N)
    step = 360.0 / c["n_spokes"]
    for i in range(c["n_spokes"]):
        fill(cover, [rotated(o["sector"], i * step)], 1)
    hub, rim = c["hub_radius_mm"] * scale, c["rim_radius_mm"] * scale
    band = max(2.0, 0.010 * N)
    # Both rings in the accent, NOT the part colour.  The spoke roots cross the hub
    # circle, so a hub ring in the same paper as the spokes shows only in the gaps
    # between them and reads as twelve broken spokes rather than as a ring.
    for r in (hub, rim):
        fill(cover, [circle(N / 2, N / 2, r + band / 2), circle(N / 2, N / 2, r - band / 2)], 2)

    tile = bytearray(N * N)
    fill(tile, [rounded_square(N, RADIUS_FRAC * N)], 1)

    # Box-filter down. Each output pixel averages SS*SS samples of ground/part/accent.
    out = []
    for y in range(SIZE):
        row = []
        for x in range(SIZE):
            acc = [0.0, 0.0, 0.0]
            inside = 0
            for dy in range(SS):
                base = (y * SS + dy) * N + x * SS
                for dx in range(SS):
                    if not tile[base + dx]:
                        continue                    # outside the tile: contributes alpha 0
                    inside += 1
                    k = cover[base + dx]
                    src = PART if k == 1 else ACCENT if k == 2 else GROUND
                    for j in range(3):
                        acc[j] += src[j]
            n = SS * SS
            if not inside:
                row.append((0, 0, 0, 0))
                continue
            # Colour is the average of the samples that are ON the tile; alpha carries
            # how many those were. Averaging in the off-tile samples instead would darken
            # the whole corner arc toward the ground colour and fringe it.
            row.append(tuple(int(v / inside) for v in acc) + (int(255 * inside / n),))
        out.append(row)

    path = os.path.join(HERE, "icon.png")
    write_png(path, out)
    print(f"{path}  {SIZE}x{SIZE}  genome {o['constants']['n_spokes']} spokes  "
          f"{os.path.getsize(path)} bytes")


if __name__ == "__main__":
    main()
