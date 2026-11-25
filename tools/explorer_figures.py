"""Animated README figures, rendered from the explorer data in docs/data (run tools/build_explorer.py first).

    python tools/explorer_figures.py

figures/explorer_section_sweep.gif
    external_06 at GrowPatch cutoff 128, which ranks 8th of 16 by gain (largest first). The CT cross-section sweeps through
    the 256-voxel cube with the manual labels, the official GrowPatch trace and the tiled-blend trace.
figures/explorer_surface_blink.gif
    CT sampled on the official GrowPatch mesh and on the tiled blend, alternating, for the regions ranked 13th, 8th
    and 4th of 16 by gain at cutoff 128, largest gain first (below average, near the middle, above average).
"""
import json
import subprocess
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "docs" / "data"
OUT = REPO / "figures"
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
BG = (24, 22, 19)
INK = (240, 236, 226)
INK2 = (184, 179, 168)
TILED = (85, 152, 231)
STOCK = (243, 239, 230)
TARGET = (0x19, 0x9E, 0x70)
OTHER = (0xD9, 0x59, 0x26)
SS = 2  # supersampling for line drawing


def font(size, bold=False):
    return ImageFont.truetype(str(FONT_DIR / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")), size)


def load_traces(rid):
    js = (DATA / rid / "traces.js").read_text(encoding="utf-8")
    return json.loads(js[js.index("(") + 1: js.rindex(")")])


def gains(cut="cut128"):
    js = (DATA / "manifest.js").read_text(encoding="utf-8")
    man = json.loads(js[js.index("=") + 1: js.rindex(";")])
    g = sorted((r["scores"][cut]["tiled"]["r128"] - r["scores"][cut]["stock"]["r128"], r["id"]) for r in man["regions"])
    return g, man


def dashed(draw, pts, fill, width, dash, gap):
    """Dashes that continue across polyline vertices."""
    on, left = True, dash
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        seg = float(np.hypot(x1 - x0, y1 - y0))
        t = 0.0
        while seg - t > 1e-6:
            step = min(left, seg - t)
            if on:
                a, b = t / seg, (t + step) / seg
                draw.line([(x0 + (x1 - x0) * a, y0 + (y1 - y0) * a), (x0 + (x1 - x0) * b, y0 + (y1 - y0) * b)], fill=fill, width=width)
            t += step
            left -= step
            if left <= 1e-6:
                on = not on
                left = dash if on else gap


def section_frame(rid, t, ct, lab, k, cut, s, show_v):
    H, W = t["height"], t["width"]
    sl = ct.crop((0, k * H, W, (k + 1) * H)).convert("RGBA")
    lb = lab.crop((0, k * H, W, (k + 1) * H))
    base = Image.new("RGBA", (W, H), BG + (255,))
    base.alpha_composite(sl)
    base.alpha_composite(lb)
    big = base.resize((int(W * s * SS), int(H * s * SS)), Image.BICUBIC)
    d = ImageDraw.Draw(big)
    v0 = t["slices"][k]
    half = np.sqrt(max(0, 128 ** 2 - v0 ** 2))
    shade = Image.new("RGBA", big.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shade)
    xl, xr = (128 - half + .5) * s * SS, (128 + half + .5) * s * SS
    sd.rectangle([0, 0, xl, big.height], fill=(14, 12, 10, 158))
    sd.rectangle([xr, 0, big.width, big.height], fill=(14, 12, 10, 158))
    big.alpha_composite(shade)
    pt = lambda p: [((p[j] / 10 + .5) * s * SS, (p[j + 1] / 10 + .5) * s * SS) for j in range(0, len(p), 2)]
    for p in t["traces"][cut]["tiled"][k]:
        d.line(pt(p), fill=(8, 10, 16), width=4 * SS, joint="curve")
    for p in t["traces"][cut]["tiled"][k]:
        d.line(pt(p), fill=TILED, width=int(2.4 * SS), joint="curve")
    for p in t["traces"][cut]["stock"][k]:
        dashed(d, pt(p), STOCK, int(1.6 * SS), 5 * SS, 4 * SS)
    if v0 == 0:
        cx, cy, r = 128.5 * s * SS, (t["wmax"] + .5) * s * SS, 12 * s * SS
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=STOCK, width=SS)
    img = big.resize((int(W * s), int(H * s)), Image.LANCZOS)
    return img


def legend(draw, x, y, f, right):
    items = [("line", TILED, "Tiled blend"), ("dash", STOCK, "Official GrowPatch"),
             ("box", TARGET, "Target sheet (manual label)"), ("box", OTHER, "Other labeled sheets")]
    x0 = x
    for kind, col, text in items:
        need = (34 if kind != "box" else 22) + draw.textlength(text, font=f)
        if x + need > right:
            x, y = x0, y + 22
        if kind == "line":
            draw.line([(x, y + 8), (x + 26, y + 8)], fill=col, width=3)
        elif kind == "dash":
            for a in range(0, 26, 8):
                draw.line([(x + a, y + 8), (x + min(a + 5, 26), y + 8)], fill=col, width=2)
        else:
            draw.rectangle([x, y + 2, x + 14, y + 14], fill=col)
        x += 34 if kind != "box" else 22
        draw.text((x, y), text, font=f, fill=INK2)
        x += draw.textlength(text, font=f) + 26


def sweep():
    rid, cut = "external_06", "cut128"
    t = load_traces(rid)
    ct, lab = Image.open(DATA / rid / "ct.webp"), Image.open(DATA / rid / "labels.png")
    s = 2.2
    W, H = int(t["width"] * s), int(t["height"] * s)
    pad, top, bottom = 16, 70, 64
    frames = []
    f1, f2, f3 = font(19, True), font(14), font(13)
    for k, v0 in enumerate(t["slices"]):
        canvas = Image.new("RGB", (W + 2 * pad, H + top + bottom), BG)
        canvas.paste(section_frame(rid, t, ct, lab, k, cut, s, True), (pad, top))
        d = ImageDraw.Draw(canvas)
        d.text((pad, 10), "Held-out region external_06", font=f1, fill=INK)
        d.text((pad, 36), "Ranks 8th of 16 by gain, close to the middle. Scroll 1, GrowPatch cutoff 128.", font=f3, fill=INK2)
        vs = "slice through the seed" if v0 == 0 else f"slice {abs(v0) * 7.91 / 1000:.2f} mm {'ahead of' if v0 > 0 else 'behind'} the seed"
        d.text((pad + W - d.textlength(vs, font=f2), 14), vs, font=f2, fill=INK)
        # progress through the cube
        d.line([(pad, top - 8), (pad + W, top - 8)], fill=(60, 57, 52), width=2)
        d.line([(pad, top - 8), (pad + W * (k / (len(t["slices"]) - 1)), top - 8)], fill=TILED, width=2)
        bar = 500 / 7.91 * s
        d.rectangle([pad + W - bar - 10, top + H - 16, pad + W - 10, top + H - 13], fill=INK)
        d.text((pad + W - 10 - d.textlength("0.5 mm", font=f3), top + H - 34), "0.5 mm", font=f3, fill=INK)
        d.text((pad + 10, top + 8), "↑ across the sheets", font=f3, fill=INK)
        legend(d, pad, top + H + 14, f3, pad + W)
        frames.append(canvas)
    write_gif(frames, OUT / "explorer_section_sweep.gif", fps=10, colors=48)
    write_mp4(frames, OUT / "explorer_section_sweep.mp4", fps=10)


def blink():
    g, man = gains()
    picks = [g[3][1], g[8][1], g[12][1]]
    size, pad, gap, top, bottom = 300, 18, 14, 64, 58
    f1, f2, f3 = font(19, True), font(14), font(13)
    states = []
    for mesh, label in (("stock", "Official GrowPatch"), ("tiled", "Tiled blend")):
        canvas = Image.new("RGB", (2 * pad + 3 * size + 2 * gap, top + size + bottom), BG)
        d = ImageDraw.Draw(canvas)
        d.text((pad, 14), "What each surface sees in the scan", font=f1, fill=INK)
        tag = f"Showing: {label}"
        d.text((pad + 3 * size + 2 * gap - d.textlength(tag, font=f2), 18), tag, font=f2, fill=TILED if mesh == "tiled" else STOCK)
        for n, rid in enumerate(picks):
            tex = Image.open(DATA / rid / f"tex_cut128_{mesh}.webp").convert("RGBA").resize((size, size), Image.BICUBIC)
            tile = Image.new("RGBA", (size, size), (18, 17, 16, 255))
            tile.alpha_composite(tex)
            x = pad + n * (size + gap)
            canvas.paste(tile.convert("RGB"), (x, top))
            r = next(r for r in man["regions"] if r["id"] == rid)["scores"]["cut128"]
            d.text((x, top - 22), f"{rid}: target sheet {r['stock']['r128']:.2f} → {r['tiled']['r128']:.2f} mm²", font=f3, fill=INK2)
        d.text((pad, top + size + 10), "Bright, fibrous texture is papyrus. Dark patches are likely gaps between sheets. Each square is 2 mm across.",
               font=f3, fill=INK2)
        d.text((pad, top + size + 30), "Left to right: ranked 13th, 8th and 4th of 16 by gain at cutoff 128 (largest gain first).",
               font=f3, fill=INK2)
        states.append(canvas)
    write_gif([states[0], states[1]], OUT / "explorer_surface_blink.gif", fps=1)


def write_mp4(frames, path, fps):
    """H.264 copy for places that play video inline (Discord, slides)."""
    with tempfile.TemporaryDirectory() as tmp:
        for i, f in enumerate(frames):
            f.save(Path(tmp) / f"f{i:04d}.png")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(Path(tmp) / "f%04d.png"),
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22",
                        "-movflags", "+faststart", str(path)], check=True)
    print(f"{path.relative_to(REPO)}: {path.stat().st_size / 1e6:.2f} MB")


def write_gif(frames, path, fps, colors=128):
    with tempfile.TemporaryDirectory() as tmp:
        for i, f in enumerate(frames):
            f.save(Path(tmp) / f"f{i:04d}.png")
        pattern = str(Path(tmp) / "f%04d.png")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", pattern,
                        "-filter_complex", f"[0:v]split[a][b];[a]palettegen=max_colors={colors}:stats_mode=full[p];[b][p]paletteuse=dither=none",
                        "-loop", "0", str(path)], check=True)
    print(f"{path.relative_to(REPO)}: {path.stat().st_size / 1e6:.2f} MB, {len(frames)} frames")


def preview_card():
    """1200 x 630 link-preview image for the explorer page (docs/og.png)."""
    rid, cut = "external_06", "cut128"
    t = load_traces(rid)
    ct, lab = Image.open(DATA / rid / "ct.webp"), Image.open(DATA / rid / "labels.png")
    k = t["slices"].index(0)
    s = min(520 / t["height"], 660 / t["width"])
    img = section_frame(rid, t, ct, lab, k, cut, s, True)
    card = Image.new("RGB", (1200, 630), BG)
    x = 1200 - img.width - 40
    card.paste(img, (x, (630 - img.height) // 2))
    d = ImageDraw.Draw(card)
    f0, f1, f2 = font(13), font(40), font(19)
    d.text((48, 64), "HELD-OUT REGION EXPLORER  /  VESUVIUS SCROLL 1", font=f0, fill=INK2)
    y = 104
    for line in ("Does the refined", "mesh stay on", "its sheet?"):
        d.text((48, y), line, font=f1, fill=INK)
        y += 52
    y += 30
    for line in ("16 hand-labeled test regions.", "Official GrowPatch (dashed) vs.", "the tiled blend (blue).", "", "Slide through every region."):
        d.text((48, y), line, font=f2, fill=INK2)
        y += 29
    card.save(REPO / "docs" / "og.png", optimize=True)
    print("docs/og.png written")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    sweep()
    blink()
    preview_card()
