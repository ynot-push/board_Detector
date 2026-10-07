#!/usr/bin/env python3
"""
Synthetic chess-screenshot generator, v7 (board-detector training data).

What is new compared with v6 (see the list at the bottom of this docstring for the reasons):

  BACKGROUNDS     flat / gradient / dark+bright+mid wallpapers (clouds, starfields) / bokeh / patterns,
                  with translucent panels over them (lichess/chess.com style)
  BOARD COLOURS   ~16 themes: very dark, very bright, wood, marble, glass; board brightness filter;
                  "low-contrast" boards that nearly match the page (dark-on-dark, white-on-white)
  BOARD DETAILS   legal-move dots and capture rings, check glow, last-move colours, selected square
                  (alpha-blended: v6 drew it opaque), arrows, circles, piece shadows, rounded corners,
                  frames, 4 coordinate styles (lichess / chess.com / v6 / outside / none)
  LAYOUTS         lichess-like, chess.com-like, centred, left, minimal; eval bar; FEN box under the board;
                  menus, opening tables, ads, cookie banners, dropdowns, mouse cursor, tooltips
  BROWSERS        Chrome/Firefox light+dark chrome, vector icons (no missing-glyph squares), bookmarks bar,
                  Windows taskbar, "page only" screenshots (no browser) and "tight" crops around the board
  DECOYS          mini boards in side panels (the label is always the MAIN board) and hard negatives for
                  the no-board class: video players, 6x6/10x10 checker grids, tables, calendars
  PHOTOMETRICS    gamma / brightness / contrast extremes (dark AND washed-out), grayscale, colour tint,
                  soft HiDPI-style resampling, noise, JPEG recompression
  EXTRA COLUMNS   layout, view, bg_style, chrome, board_kind, board_brightness, low_contrast, gray, gamma,
                  decoys -> so you can evaluate the detector per category and see WHICH kind fails.
                  (If your loader dislikes extra columns, delete them from CSV_FIELDS.)
  BUG FIXES       selected-square tint was opaque; move-list highlight and active tab were re-rolled
                  inside loops; full-canvas RGBA layers per highlight (slow); missing-glyph icons.

Usage
  python generate_data_v7.py --preview 24        # writes preview_v7.jpg (boxes drawn) and exits
  python generate_data_v7.py --n 10000           # full dataset into dataset_v7/
The label convention is unchanged: x1,y1,x2,y2 = the 8x8 PLAYING AREA (no frame, no outside coordinates).
"""
import argparse
import colorsys
import csv
import io
import math
import random
from functools import lru_cache
from pathlib import Path

import chess
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageStat

# ============================================================
# CONFIG
# ============================================================
NUM_IMAGES = 10000
OUTPUT_DIR = Path("dataset_v7")
PIECE_ROOT = Path("../pieces_styles")
USE_ALL_PIECE_STYLES = True          # every complete folder in PIECE_ROOT (more variety)
PIECE_STYLES = ["cburnett", "alpha", "merida", "companion", "neo_64"]   # used when the flag above is False
SAVE_JPEG = True
JPEG_QUALITY_RANGE = (82, 96)
SEED = 42

SAMPLE_KIND_WEIGHTS = {"normal": 0.88, "partial_crop": 0.0, "no_board": 0.12}
VIEW_WEIGHTS = {"full": 0.70, "page_only": 0.18, "tight": 0.12}
LAYOUT_WEIGHTS = {"lichess": 0.28, "chesscom": 0.26, "centered": 0.20, "left": 0.14, "minimal": 0.12}
SCREEN_SIZES = [((1920, 1080), 0.34), ((1366, 768), 0.10), ((1440, 900), 0.08), ((1536, 864), 0.10),
                ((1280, 720), 0.06), ((1600, 900), 0.08), ((2560, 1440), 0.10), ((1280, 800), 0.06),
                ((1680, 1050), 0.08)]
LOW_CONTRAST_PROB = 0.24             # board colours chosen to nearly match the page around the board

CSV_FIELDS = ["filename", "has_board", "sample_kind", "x1", "y1", "x2", "y2", "screen_width", "screen_height",
              "page_type", "website_theme", "board_theme", "piece_style", "coordinates", "border", "flipped", "fen",
              # extra, for per-category evaluation:
              "layout", "view", "bg_style", "chrome", "board_kind", "board_brightness", "low_contrast", "gray",
              "gamma", "decoys"]

# ============================================================
# THEMES
# ============================================================
BOARD_THEMES = [   # (name, light, dark, kind)
    ("brown", (240, 217, 181), (181, 136, 99), "flat"),
    ("blue", (222, 227, 230), (90, 120, 150), "flat"),
    ("green", (235, 236, 208), (119, 149, 86), "flat"),
    ("gray", (225, 225, 225), (120, 120, 120), "flat"),
    ("purple", (235, 225, 240), (135, 105, 150), "flat"),
    ("red", (240, 220, 215), (155, 90, 80), "flat"),
    ("dark", (190, 190, 190), (80, 80, 80), "flat"),
    ("midnight", (96, 104, 118), (54, 60, 72), "flat"),        # very dark boards (dark themes)
    ("charcoal", (118, 118, 120), (62, 62, 64), "flat"),
    ("slate", (112, 124, 138), (66, 78, 92), "glass"),
    ("white", (250, 250, 250), (208, 213, 220), "flat"),       # very bright boards
    ("ice", (238, 246, 252), (170, 192, 210), "glass"),
    ("cream", (253, 246, 230), (224, 207, 172), "flat"),
    ("maple", (240, 217, 181), (181, 136, 99), "wood"),
    ("walnut", (222, 184, 135), (128, 82, 42), "wood"),
    ("marble", (232, 232, 234), (124, 129, 140), "marble"),
]

WEBSITE_THEMES = {
    "dark": {"bg": (24, 26, 30), "panel": (34, 37, 43), "panel2": (43, 46, 53), "text": (235, 235, 235),
             "muted": (150, 155, 165), "accent": (80, 150, 240), "border": (60, 64, 72)},
    "light": {"bg": (238, 240, 243), "panel": (255, 255, 255), "panel2": (245, 246, 248), "text": (35, 38, 42),
              "muted": (110, 115, 125), "accent": (55, 110, 200), "border": (210, 212, 216)},
    "blue": {"bg": (20, 32, 48), "panel": (29, 46, 68), "panel2": (38, 58, 82), "text": (235, 242, 250),
             "muted": (155, 175, 195), "accent": (70, 150, 230), "border": (55, 75, 100)},
    "purple": {"bg": (31, 25, 40), "panel": (46, 36, 57), "panel2": (59, 47, 72), "text": (240, 235, 245),
               "muted": (170, 155, 180), "accent": (155, 105, 220), "border": (75, 62, 90)},
    "gray": {"bg": (55, 57, 60), "panel": (70, 72, 76), "panel2": (82, 84, 89), "text": (235, 235, 235),
             "muted": (170, 170, 175), "accent": (120, 150, 190), "border": (95, 97, 102)},
    "black": {"bg": (22, 21, 18), "panel": (38, 36, 33), "panel2": (50, 48, 44), "text": (186, 186, 186),
              "muted": (130, 130, 128), "accent": (120, 170, 60), "border": (58, 56, 52)},
    "white": {"bg": (252, 252, 252), "panel": (246, 246, 245), "panel2": (236, 236, 234), "text": (40, 40, 40),
              "muted": (120, 120, 120), "accent": (60, 120, 190), "border": (215, 215, 212)},
    "olive": {"bg": (49, 46, 43), "panel": (39, 37, 34), "panel2": (60, 57, 53), "text": (240, 240, 235),
              "muted": (160, 158, 150), "accent": (129, 182, 76), "border": (70, 67, 62)},
}

CHROMES = {
    "chrome_dark": {"bar": (32, 33, 36), "tab": (53, 54, 58), "url": (41, 42, 45), "text": (232, 234, 237), "muted": (154, 160, 166)},
    "chrome_light": {"bar": (222, 225, 230), "tab": (255, 255, 255), "url": (241, 243, 244), "text": (32, 33, 36), "muted": (95, 99, 104)},
    "ff_dark": {"bar": (28, 27, 34), "tab": (66, 65, 77), "url": (43, 42, 51), "text": (251, 251, 254), "muted": (190, 190, 200)},
    "ff_light": {"bar": (240, 240, 244), "tab": (255, 255, 255), "url": (255, 255, 255), "text": (21, 20, 26), "muted": (91, 91, 102)},
}

URLS = ["chess.example.com/game/8fj29s", "lichess.org/analysis", "www.chess.com/play/online", "chess.example.org/puzzle",
        "lichess.org/tv", "www.chess.com/game/live/1234567", "chessplay.net/watch", "lichess.org/study/aBcDeF"]
TAB_NAMES = ["Chess", "Game", "Play", "Analysis board", "YouTube", "New Tab", "Puzzles", "Inbox", "Docs", "Watch"]
PAGE_TYPES = ["game", "game", "analysis", "analysis", "replay", "tournament", "puzzle", "watch", "community"]


# ============================================================
# SMALL HELPERS
# ============================================================
def clamp(v, a, b):
    return max(a, min(v, b))


def lum(c):
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def mix(a, b, t):
    return tuple(int(round(a[i] * (1 - t) + b[i] * t)) for i in range(3))


def scale_c(c, k):
    return tuple(int(clamp(round(v * k), 0, 255)) for v in c[:3])


def rgba(c, a=255):
    return (int(c[0]), int(c[1]), int(c[2]), int(a))


def wchoice(d):
    return random.choices(list(d.keys()), weights=list(d.values()))[0]


def rect_intersects(a, b, margin=0):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 + margin <= bx1 or ax1 - margin >= bx2 or ay2 + margin <= by1 or ay1 - margin >= by2)


_FAMILY = [0]
FONT_FAMILIES = [
    (["C:/Windows/Fonts/segoeui.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
     ["C:/Windows/Fonts/segoeuib.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]),
    (["C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
     ["C:/Windows/Fonts/arialbd.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]),
    (["C:/Windows/Fonts/verdana.ttf", "C:/Windows/Fonts/tahoma.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
     ["C:/Windows/Fonts/verdanab.ttf", "C:/Windows/Fonts/tahomabd.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]),
    (["C:/Windows/Fonts/calibri.ttf", "/usr/share/fonts/truetype/freefont/FreeSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
     ["C:/Windows/Fonts/calibrib.ttf", "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]),
]


@lru_cache(maxsize=1024)
def _font(size, bold, family):
    for path in FONT_FAMILIES[family][1 if bold else 0]:
        if Path(path).exists():
            return ImageFont.truetype(path, int(size))
    return ImageFont.load_default()


def font(size, bold=False):
    return _font(int(max(6, size)), bool(bold), _FAMILY[0])


def rounded(draw, box, radius, fill, outline=None, width=1):
    x1, y1, x2, y2 = box
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    draw.rounded_rectangle([x1, y1, x2, y2], radius=min(radius, (x2 - x1) // 2, (y2 - y1) // 2), fill=fill, outline=outline, width=width)


def panel(draw, box, colors, radius=8, key="panel"):
    a = colors["palpha"]
    rounded(draw, box, radius, rgba(colors[key], a), rgba(colors["border"], min(255, a + 30)), 1)


def text_wh(draw, s, fnt):
    b = draw.textbbox((0, 0), s, font=fnt)
    return b[2] - b[0], b[3] - b[1]


def draw_text_center(draw, box, s, fnt, fill):
    tw, th = text_wh(draw, s, fnt)
    draw.text(((box[0] + box[2]) / 2 - tw / 2, (box[1] + box[3]) / 2 - th / 2 - 1), s, font=fnt, fill=fill)


def random_username():
    names = ["ChessMaster", "KnightStorm", "BlueKnight", "QueenHunter", "RookAttack", "ChessWizard", "TacticalFox",
             "EndgameKing", "RapidPlayer", "DarkKnight", "BoardVision", "AlphaChess", "GrandMaster", "ChessFan", "PuzzleKing"]
    return random.choice(names) + str(random.randint(10, 9999))


# ============================================================
# VECTOR ICONS (no font glyphs -> no missing-glyph squares)
# ============================================================
def icon(draw, name, cx, cy, s, color, w=2):
    h = s / 2.0
    if name == "back":
        draw.line([(cx + h * .5, cy - h * .7), (cx - h * .5, cy), (cx + h * .5, cy + h * .7)], fill=color, width=w)
        draw.line([(cx - h * .5, cy), (cx + h * .9, cy)], fill=color, width=w)
    elif name == "forward":
        draw.line([(cx - h * .5, cy - h * .7), (cx + h * .5, cy), (cx - h * .5, cy + h * .7)], fill=color, width=w)
        draw.line([(cx + h * .5, cy), (cx - h * .9, cy)], fill=color, width=w)
    elif name == "reload":
        draw.arc([cx - h * .7, cy - h * .7, cx + h * .7, cy + h * .7], 40, 330, fill=color, width=w)
        draw.polygon([(cx + h * .75, cy - h * .75), (cx + h * .75, cy - h * .05), (cx + h * .1, cy - h * .05)], fill=color)
    elif name == "lock":
        draw.rounded_rectangle([cx - h * .55, cy - h * .1, cx + h * .55, cy + h * .75], radius=2, fill=color)
        draw.arc([cx - h * .4, cy - h * .85, cx + h * .4, cy + h * .3], 180, 360, fill=color, width=max(1, w - 1))
    elif name == "star":
        pts = []
        for i in range(10):
            r = h * (0.85 if i % 2 == 0 else 0.38)
            a = -math.pi / 2 + i * math.pi / 5
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        draw.polygon(pts, outline=color, width=max(1, w - 1))
    elif name == "menu":
        for dy in (-h * .55, 0, h * .55):
            draw.line([(cx - h * .7, cy + dy), (cx + h * .7, cy + dy)], fill=color, width=w)
    elif name == "dots":
        for dy in (-h * .6, 0, h * .6):
            draw.ellipse([cx - 1.6, cy + dy - 1.6, cx + 1.6, cy + dy + 1.6], fill=color)
    elif name == "close":
        draw.line([(cx - h * .5, cy - h * .5), (cx + h * .5, cy + h * .5)], fill=color, width=w)
        draw.line([(cx - h * .5, cy + h * .5), (cx + h * .5, cy - h * .5)], fill=color, width=w)
    elif name == "min":
        draw.line([(cx - h * .5, cy), (cx + h * .5, cy)], fill=color, width=w)
    elif name == "max":
        draw.rectangle([cx - h * .45, cy - h * .45, cx + h * .45, cy + h * .45], outline=color, width=max(1, w - 1))
    elif name == "search":
        draw.ellipse([cx - h * .7, cy - h * .7, cx + h * .3, cy + h * .3], outline=color, width=w)
        draw.line([(cx + h * .2, cy + h * .2), (cx + h * .8, cy + h * .8)], fill=color, width=w)
    elif name == "gear":
        draw.ellipse([cx - h * .55, cy - h * .55, cx + h * .55, cy + h * .55], outline=color, width=w)
        for k in range(8):
            a = k * math.pi / 4
            draw.line([(cx + h * .55 * math.cos(a), cy + h * .55 * math.sin(a)), (cx + h * .85 * math.cos(a), cy + h * .85 * math.sin(a))], fill=color, width=w)
    elif name == "home":
        draw.polygon([(cx - h * .8, cy), (cx, cy - h * .8), (cx + h * .8, cy)], outline=color, width=w)
        draw.rectangle([cx - h * .5, cy, cx + h * .5, cy + h * .7], outline=color, width=max(1, w - 1))
    elif name == "play":
        draw.polygon([(cx - h * .5, cy - h * .7), (cx + h * .7, cy), (cx - h * .5, cy + h * .7)], fill=color)
    elif name == "pause":
        draw.rectangle([cx - h * .55, cy - h * .6, cx - h * .15, cy + h * .6], fill=color)
        draw.rectangle([cx + h * .15, cy - h * .6, cx + h * .55, cy + h * .6], fill=color)
    elif name == "first":
        draw.line([(cx - h * .7, cy - h * .6), (cx - h * .7, cy + h * .6)], fill=color, width=w)
        draw.polygon([(cx + h * .6, cy - h * .6), (cx - h * .4, cy), (cx + h * .6, cy + h * .6)], fill=color)
    elif name == "last":
        draw.line([(cx + h * .7, cy - h * .6), (cx + h * .7, cy + h * .6)], fill=color, width=w)
        draw.polygon([(cx - h * .6, cy - h * .6), (cx + h * .4, cy), (cx - h * .6, cy + h * .6)], fill=color)
    elif name == "knight":   # simple stand-in for a logo / piece icon
        draw.polygon([(cx - h * .5, cy + h * .8), (cx - h * .4, cy), (cx - h * .1, cy - h * .7), (cx + h * .5, cy - h * .5),
                      (cx + h * .6, cy - h * .1), (cx + h * .1, cy + h * .1), (cx + h * .5, cy + h * .8)], fill=color)
    elif name == "plus":
        draw.line([(cx - h * .5, cy), (cx + h * .5, cy)], fill=color, width=w)
        draw.line([(cx, cy - h * .5), (cx, cy + h * .5)], fill=color, width=w)


# ============================================================
# BACKGROUNDS
# ============================================================
def value_noise(w, h, scale, octaves=4, persistence=0.55):
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        gw = max(2, int(w / max(scale, 1) * (2 ** o)) + 2)
        gh = max(2, int(h / max(scale, 1) * (2 ** o)) + 2)
        small = (np.random.rand(gh, gw) * 255).astype(np.uint8)
        out += amp * np.asarray(Image.fromarray(small).resize((w, h), Image.BICUBIC), np.float32) / 255.0
        tot += amp
        amp *= persistence
    return out / tot


def hsv(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, clamp(s, 0, 1), clamp(v, 0, 1))
    return np.array([r, g, b], np.float32) * 255.0


def wallpaper(W, H, mode):
    w, h = max(16, W // 5), max(16, H // 5)
    n1 = value_noise(w, h, random.uniform(14, 50), 4)
    n2 = value_noise(w, h, random.uniform(30, 90), 3)
    h1 = random.random()
    h2 = h1 + random.uniform(-0.15, 0.3)
    if mode == "dark":
        c1, c2 = hsv(h1, random.uniform(.2, .8), random.uniform(.03, .18)), hsv(h2, random.uniform(.2, .8), random.uniform(.2, .55))
    elif mode == "bright":
        c1, c2 = hsv(h1, random.uniform(.03, .3), random.uniform(.88, .99)), hsv(h2, random.uniform(.1, .45), random.uniform(.62, .9))
    else:
        c1, c2 = hsv(h1, random.uniform(.2, .7), random.uniform(.3, .5)), hsv(h2, random.uniform(.2, .7), random.uniform(.55, .85))
    t = np.clip((n1 - 0.35) * 2.2, 0, 1)[..., None]
    img = c1 * (1 - t) + c2 * t
    img *= (0.82 + 0.36 * n2[..., None])
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = ((xx / w - 0.5) ** 2 + (yy / h - 0.5) ** 2)
    img *= (1 - (1.1 if mode == "dark" else 0.5) * d)[..., None]
    out = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)
    if mode == "dark" and random.random() < 0.6:           # starfield
        arr = np.asarray(out).copy()
        k = random.randint(80, 520)
        xs, ys = np.random.randint(0, W - 1, k), np.random.randint(0, H - 1, k)
        b = np.random.randint(110, 255, k)[:, None]
        arr[ys, xs] = np.minimum(255, arr[ys, xs].astype(np.int32) + b)
        arr[ys, xs + 1] = np.minimum(255, arr[ys, xs + 1].astype(np.int32) + b // 2)
        out = Image.fromarray(arr.astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6))
    return out


def make_background(W, H, colors):
    kind = random.choices(["flat", "gradient", "wall_dark", "wall_bright", "wall_mid", "bokeh", "pattern"],
                          weights=[0.34, 0.14, 0.14, 0.10, 0.08, 0.06, 0.08])[0]
    base = colors["bg"]
    if kind == "flat":
        img = Image.new("RGB", (W, H), base)
    elif kind == "gradient":
        other = mix(base, tuple(random.randint(0, 255) for _ in range(3)), random.uniform(0.15, 0.5))
        t = np.linspace(0, 1, H, dtype=np.float32)[:, None] if random.random() < 0.5 else (np.linspace(0, 1, W, dtype=np.float32)[None, :] * 0.7 + np.linspace(0, 1, H, dtype=np.float32)[:, None] * 0.3)
        arr = np.array(base, np.float32)[None, None, :] * (1 - t[..., None]) + np.array(other, np.float32)[None, None, :] * t[..., None]
        img = Image.fromarray(np.clip(np.broadcast_to(arr, (H, W, 3)), 0, 255).astype(np.uint8))
    elif kind.startswith("wall"):
        img = wallpaper(W, H, kind.split("_")[1])
    elif kind == "bokeh":
        img = Image.new("RGB", (W, H), base)
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(lay, "RGBA")
        for _ in range(random.randint(10, 30)):
            r = random.randint(30, 160)
            x, y = random.randint(0, W), random.randint(0, H)
            c = mix(base, tuple(random.randint(0, 255) for _ in range(3)), random.uniform(0.2, 0.7))
            d.ellipse([x - r, y - r, x + r, y + r], fill=rgba(c, random.randint(30, 90)))
        img = Image.alpha_composite(img.convert("RGBA"), lay.filter(ImageFilter.GaussianBlur(random.randint(8, 30)))).convert("RGB")
    else:
        img = Image.new("RGB", (W, H), base)
        d = ImageDraw.Draw(img, "RGBA")
        step = random.randint(18, 60)
        c = rgba(colors["text"], random.randint(6, 18))
        if random.random() < 0.5:
            for x in range(-H, W, step):
                d.line([(x, 0), (x + H, H)], fill=c, width=1)
        else:
            for x in range(0, W, step):
                for y in range(0, H, step):
                    d.ellipse([x, y, x + 2, y + 2], fill=c)
    return img.convert("RGB"), kind


# ============================================================
# BROWSER CHROME, TASKBAR, WEBSITE HEADER
# ============================================================
def draw_browser(draw, W, chrome):
    tab_h, nav_h = random.randint(34, 42), random.randint(38, 46)
    bm_h = random.choice([0, 0, 26])
    total = tab_h + nav_h + bm_h
    draw.rectangle([0, 0, W, total], fill=chrome["bar"])
    x, n = 8, random.randint(1, 6)
    active = random.randint(0, n - 1)
    for i in range(n):
        wt = random.randint(130, 230)
        if x + wt > W - 220:
            break
        box = [x, 6, x + wt, tab_h + 1]
        if i == active:
            try:
                draw.rounded_rectangle(box, radius=9, fill=chrome["tab"], corners=(True, True, False, False))
            except TypeError:
                draw.rounded_rectangle(box, radius=9, fill=chrome["tab"])
        fav = tuple(random.randint(60, 230) for _ in range(3))
        draw.ellipse([x + 10, 14, x + 24, 28], fill=fav)
        draw.text((x + 32, 12), random.choice(TAB_NAMES), font=font(13), fill=chrome["text"])
        icon(draw, "close", x + wt - 16, 21, 12, chrome["muted"], 1)
        x += wt + random.randint(0, 3)
    icon(draw, "plus", x + 16, 21, 14, chrome["muted"], 2)
    for k, nm in enumerate(["min", "max", "close"]):                   # window controls
        icon(draw, nm, W - 138 + k * 46, 20, 14, chrome["text"], 1)
    cy = tab_h + nav_h // 2
    icon(draw, "back", 22, cy, 18, chrome["text"], 2)
    icon(draw, "forward", 56, cy, 18, chrome["muted"], 2)
    icon(draw, "reload", 90, cy, 18, chrome["text"], 2)
    ux1, ux2 = 122, W - 175
    rounded(draw, [ux1, tab_h + 5, ux2, tab_h + nav_h - 5], (nav_h - 10) // 2, chrome["url"])
    icon(draw, "lock", ux1 + 20, cy, 13, chrome["muted"], 1)
    draw.text((ux1 + 38, cy - 9), random.choice(URLS), font=font(14), fill=chrome["text"])
    icon(draw, "star", ux2 - 24, cy, 16, chrome["muted"], 2)
    for k in range(random.randint(1, 3)):
        rounded(draw, [W - 160 + k * 30, cy - 10, W - 142 + k * 30, cy + 8], 4, rgba(chrome["muted"], 160))
    icon(draw, "dots", W - 40, cy, 18, chrome["text"], 2)
    if bm_h:
        by = tab_h + nav_h
        xx = 18
        for s in random.sample(["Chess", "News", "Games", "History", "Analysis", "Videos", "Mail", "Maps"], 6):
            draw.ellipse([xx, by + 7, xx + 12, by + 19], fill=tuple(random.randint(80, 220) for _ in range(3)))
            draw.text((xx + 18, by + 5), s, font=font(12), fill=chrome["text"])
            xx += random.randint(95, 130)
    return total


def draw_taskbar(draw, W, H):
    h = random.randint(40, 48)
    dark = random.random() < 0.6
    draw.rectangle([0, H - h, W, H], fill=(32, 32, 32, 240) if dark else (243, 243, 243, 240))
    n = random.randint(5, 9)
    x0 = W // 2 - n * 19
    for i in range(n):
        rounded(draw, [x0 + i * 38, H - h + 9, x0 + i * 38 + 24, H - h + 33], 6, tuple(random.randint(60, 230) for _ in range(3)))
    t = random.choice(["10:42 PM", "9:15 AM", "14:07", "23:58"])
    draw.text((W - 105, H - h + 6), t, font=font(12), fill=(235, 235, 235) if dark else (30, 30, 30))
    draw.text((W - 105, H - h + 24), "10/5/2026", font=font(11), fill=(200, 200, 200) if dark else (60, 60, 60))
    return h


def draw_website_header(draw, y, W, colors):
    h = random.randint(46, 80)
    draw.rectangle([0, y, W, y + h], fill=rgba(colors["panel"], min(255, colors["palpha"] + 20)))
    icon(draw, "knight", 32, y + h // 2, 22, colors["accent"], 2)
    draw.text((52, y + h // 2 - 13), random.choice(["CHESS", "chess", "BOARD", "PLAY"]), font=font(20, True), fill=colors["text"])
    x = 190
    for item in ["PLAY", "PUZZLES", "WATCH", "LEARN", "COMMUNITY", "TOOLS"][:random.randint(4, 6)]:
        draw.text((x, y + h // 2 - 8), item, font=font(12, True), fill=colors["muted"])
        x += random.randint(78, 108)
    sw = random.randint(150, 240)
    sx = W - sw - 190
    if sx > x:
        rounded(draw, [sx, y + 12, sx + sw, y + h - 12], 16, colors["panel2"])
        icon(draw, "search", sx + 20, y + h // 2, 14, colors["muted"], 2)
        draw.text((sx + 38, y + h // 2 - 8), "Search", font=font(12), fill=colors["muted"])
    icon(draw, "gear", W - 135, y + h // 2, 18, colors["text"], 2)
    icon(draw, "menu", W - 100, y + h // 2, 18, colors["text"], 2)
    draw.ellipse([W - 62, y + h // 2 - 18, W - 26, y + h // 2 + 18], fill=colors["accent"])
    return y + h


# ============================================================
# PIECES, POSITIONS
# ============================================================
PIECE_FILES = ["wK", "wQ", "wR", "wB", "wN", "wP", "bK", "bQ", "bR", "bB", "bN", "bP"]


def find_complete_styles():
    if not PIECE_ROOT.exists():
        raise FileNotFoundError(f"Piece directory not found: {PIECE_ROOT.resolve()}")
    names = sorted(p.name for p in PIECE_ROOT.iterdir() if p.is_dir()) if USE_ALL_PIECE_STYLES else PIECE_STYLES
    ok = [s for s in names if all((PIECE_ROOT / s / f"{p}.png").exists() for p in PIECE_FILES)]
    if not ok:
        raise RuntimeError(f"No complete piece style found in {PIECE_ROOT.resolve()}")
    return ok


COMPLETE_STYLES = find_complete_styles() if PIECE_ROOT.exists() else []
_PIECE_CACHE = {}


def get_piece(style, code, size):
    key = (style, code)
    if key not in _PIECE_CACHE:
        _PIECE_CACHE[key] = Image.open(PIECE_ROOT / style / f"{code}.png").convert("RGBA")
    img = _PIECE_CACHE[key].copy()
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    return img


def random_position():
    """Mix of openings, middlegames and endgames (v6 was uniform random play)."""
    board = chess.Board()
    r = random.random()
    plies = random.randint(0, 24) if r < 0.35 else random.randint(24, 70) if r < 0.80 else random.randint(60, 140)
    for _ in range(plies):
        legal = list(board.legal_moves)
        if not legal:
            break
        board.push(random.choice(legal))
    if random.random() < 0.04:                       # sparse / near-empty boards
        for sq in list(board.piece_map()):
            p = board.piece_at(sq)
            if p.piece_type != chess.KING and random.random() < 0.8:
                board.remove_piece_at(sq)
    return board


# ============================================================
# BOARD RENDERING
# ============================================================
def board_texture(bs, light, dark, kind, brightness):
    idx = (np.arange(bs) * 8) // bs
    rr, cc = np.meshgrid(idx, idx, indexing="ij")
    par = ((rr + cc) % 2 == 0)[..., None]
    arr = np.where(par, np.array(light, np.float32), np.array(dark, np.float32)).astype(np.float32)
    if kind == "wood":
        g = np.asarray(Image.fromarray((np.random.rand(bs // 3 + 2, 6) * 255).astype(np.uint8)).resize((bs, bs), Image.BICUBIC), np.float32) / 255
        fine = np.asarray(Image.fromarray((np.random.rand(bs // 2 + 2, 40) * 255).astype(np.uint8)).resize((bs, bs), Image.BICUBIC), np.float32) / 255
        arr *= (1 + (g - 0.5) * 0.24 + (fine - 0.5) * 0.10)[..., None]
    elif kind == "marble":
        n = value_noise(bs // 2, bs // 2, 40, 4)
        n = np.asarray(Image.fromarray((n * 255).astype(np.uint8)).resize((bs, bs), Image.BICUBIC), np.float32) / 255
        veins = np.abs(np.sin(n * 5.0 * np.pi))
        arr *= (1 - 0.07 * (1 - veins))[..., None]
    elif kind == "glass":
        yy = np.linspace(0, 1, bs, dtype=np.float32)[:, None]
        arr *= (1.08 - 0.16 * yy)[..., None]
    arr *= (1 + (np.random.rand(8, 8) - 0.5) * 0.03).repeat(max(1, bs // 8) + 1, 0).repeat(max(1, bs // 8) + 1, 1)[:bs, :bs, None]
    return np.clip(arr * brightness, 0, 255)


def arrow_poly(p0, p1, cell):
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy) or 1
    ux, uy = dx / L, dy / L
    nx, ny = -uy, ux
    sw, hw, hl = cell * 0.10, cell * 0.26, cell * 0.38
    sx0, sy0 = x0 + ux * cell * 0.25, y0 + uy * cell * 0.25
    bx, by = x1 - ux * hl, y1 - uy * hl
    return [(sx0 + nx * sw, sy0 + ny * sw), (bx + nx * sw, by + ny * sw), (bx + nx * hw, by + ny * hw), (x1, y1),
            (bx - nx * hw, by - ny * hw), (bx - nx * sw, by - ny * sw), (sx0 - nx * sw, sy0 - ny * sw)]


def render_board(board, bs, style, theme, flipped, coord_style, brightness, overlays=True):
    _, light, dark, kind = theme
    arr = board_texture(bs, light, dark, kind, brightness)
    cell = bs / 8.0

    def rc(sq):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        return (r, 7 - f) if flipped else (7 - r, f)

    def box(r, c):
        return int(round(c * cell)), int(round(r * cell)), int(round((c + 1) * cell)), int(round((r + 1) * cell))

    def tint(r, c, col, a):
        x1, y1, x2, y2 = box(r, c)
        arr[y1:y2, x1:x2] = arr[y1:y2, x1:x2] * (1 - a) + np.array(col, np.float32) * a

    sel_sq = None
    if overlays:
        if board.move_stack and random.random() < 0.75:
            col, a = random.choice([((155, 199, 0), 0.41), ((255, 255, 51), 0.5), ((20, 85, 255), 0.30), ((255, 170, 0), 0.45)])
            m = board.peek()
            for sq in (m.from_square, m.to_square):
                tint(*rc(sq), col, a)
        if random.random() < 0.28:
            mine = [s for s, p in board.piece_map().items() if p.color == board.turn]
            if mine:
                sel_sq = random.choice(mine)
                tint(*rc(sel_sq), random.choice([(20, 85, 30), (255, 255, 51), (80, 160, 240)]), 0.5)
        if board.is_check():
            ks = board.king(board.turn)
            if ks is not None:
                r, c = rc(ks)
                x1, y1, x2, y2 = box(r, c)
                yy, xx = np.mgrid[y1:y2, x1:x2].astype(np.float32)
                d = np.hypot(xx - (x1 + x2) / 2, yy - (y1 + y2) / 2) / (cell * 0.6)
                a = np.clip(1 - d, 0, 1)[..., None] * 0.9
                arr[y1:y2, x1:x2] = arr[y1:y2, x1:x2] * (1 - a) + np.array((255, 30, 30), np.float32) * a
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im, "RGBA")
    # inside coordinates (drawn under the pieces)
    if coord_style in ("lichess", "chesscom", "v6"):
        cf = font(max(9, int(cell * random.uniform(0.15, 0.2))), True)
        for i in range(8):
            letter = "abcdefgh"[7 - i if flipped else i]
            number = str(i + 1 if flipped else 8 - i)
            lc = light if i % 2 == 1 else dark
            if coord_style == "chesscom":
                d.text((3, int(i * cell) + 2), number, font=cf, fill=rgba(dark if i % 2 == 0 else light, 255))
                d.text((int((i + 1) * cell) - cf.size * 0.7 - 3, bs - cf.size - 4), letter, font=cf, fill=rgba(dark if i % 2 == 1 else light, 255))
            elif coord_style == "lichess":
                d.text((bs - cf.size * 0.6 - 3, int(i * cell) + 2), number, font=cf, fill=rgba(dark if i % 2 == 1 else light, 230))
                d.text((int(i * cell) + 3, bs - cf.size - 3), letter, font=cf, fill=rgba(dark if i % 2 == 0 else light, 230))
            else:
                d.text((3, int(i * cell) + 2), number, font=cf, fill=rgba(light if i % 2 == 0 else dark, 255))
                d.text((int(i * cell) + 3, bs - cf.size - 4), letter, font=cf, fill=rgba(dark if i % 2 == 0 else light, 255))
    # pieces
    psize = int(cell * random.uniform(0.82, 1.0))
    shadow_on = random.random() < 0.30
    for sq, p in board.piece_map().items():
        r, c = rc(sq)
        code = ("w" if p.color else "b") + p.symbol().upper()
        pi = get_piece(style, code, psize)
        px, py = int(c * cell + (cell - pi.width) / 2), int(r * cell + (cell - pi.height) / 2)
        if shadow_on:
            a = pi.split()[3].filter(ImageFilter.GaussianBlur(max(1.0, cell * 0.025))).point(lambda v: int(v * 0.38))
            im.paste(Image.new("RGB", pi.size, (0, 0, 0)),
                     (int(clamp(px + cell * 0.02, 0, bs - pi.width)), int(clamp(py + cell * 0.035, 0, bs - pi.height))), a)
        im.paste(pi, (int(clamp(px, 0, bs - pi.width)), int(clamp(py, 0, bs - pi.height))), pi)
    if overlays and sel_sq is not None:                                   # legal-move dots and capture rings
        for mv in board.legal_moves:
            if mv.from_square != sel_sq:
                continue
            r, c = rc(mv.to_square)
            cx, cy = (c + 0.5) * cell, (r + 0.5) * cell
            if board.piece_at(mv.to_square) is not None:
                d.ellipse([cx - cell * .46, cy - cell * .46, cx + cell * .46, cy + cell * .46], outline=(20, 85, 30, 130), width=max(2, int(cell * 0.09)))
            else:
                d.ellipse([cx - cell * .16, cy - cell * .16, cx + cell * .16, cy + cell * .16], fill=(20, 85, 30, 120))
    if overlays and random.random() < 0.25:                               # arrows and circles
        for _ in range(random.randint(1, 3)):
            col = random.choice([(21, 138, 0), (204, 0, 0), (0, 48, 136), (255, 170, 0)])
            a, b = random.sample(range(64), 2)
            ra, ca = rc(a)
            rb, cb = rc(b)
            d.polygon(arrow_poly(((ca + .5) * cell, (ra + .5) * cell), ((cb + .5) * cell, (rb + .5) * cell), cell), fill=rgba(col, 190))
        if random.random() < 0.4:
            r, c = rc(random.randrange(64))
            d.ellipse([(c + .08) * cell, (r + .08) * cell, (c + .92) * cell, (r + .92) * cell], outline=(21, 138, 0, 200), width=max(2, int(cell * 0.06)))
    return im


def board_shadow(img, x, y, bs):
    pad = 40
    lay = Image.new("L", (bs + 2 * pad, bs + 2 * pad), 0)
    ImageDraw.Draw(lay).rectangle([pad + 6, pad + 9, pad + bs + 6, pad + bs + 9], fill=random.randint(70, 130))
    lay = lay.filter(ImageFilter.GaussianBlur(random.uniform(8, 14)))
    img.paste(Image.new("RGB", lay.size, (0, 0, 0)), (x - pad, y - pad), lay)


def draw_frame(draw, x, y, bs, kind, colors):
    if kind == "thin":
        draw.rectangle([x - 1, y - 1, x + bs, y + bs], outline=(30, 30, 30, 255), width=2)
    elif kind == "thick":
        draw.rectangle([x - 5, y - 5, x + bs + 5, y + bs + 5], outline=(35, 35, 35, 255), width=5)
    elif kind == "wood":
        w = random.randint(8, 16)
        col = random.choice([(110, 72, 40), (150, 105, 60), (80, 55, 35)])
        draw.rectangle([x - w, y - w, x + bs + w, y + bs + w], outline=rgba(col, 255), width=w)
    elif kind == "light":
        draw.rectangle([x - 3, y - 3, x + bs + 3, y + bs + 3], outline=(245, 245, 245, 255), width=3)


def outside_coords(draw, x, y, bs, flipped, colors):
    cf = font(max(10, int(bs / 8 * 0.18)))
    for i in range(8):
        letter = "abcdefgh"[7 - i if flipped else i]
        number = str(i + 1 if flipped else 8 - i)
        draw.text((x - 16, y + (i + 0.5) * bs / 8 - 7), number, font=cf, fill=colors["muted"])
        draw.text((x + (i + 0.5) * bs / 8 - 4, y + bs + 4), letter, font=cf, fill=colors["muted"])


# ============================================================
# UI PANELS
# ============================================================
def draw_player_card(draw, box, colors):
    x1, y1, x2, y2 = box
    panel(draw, box, colors)
    av = min(40, y2 - y1 - 12)
    draw.ellipse([x1 + 10, y1 + 6, x1 + 10 + av, y1 + 6 + av], fill=colors["accent"])
    tx = x1 + av + 20
    title = random.choice(["", "", "", "GM", "IM", "FM", "CM", "NM"])
    if title:
        draw.text((tx, y1 + 7), title, font=font(11, True), fill=colors["accent"])
        tx += 30
    draw.text((tx, y1 + 7), random_username(), font=font(13, True), fill=colors["text"])
    draw.text((tx, y1 + 27), str(random.randint(1100, 2800)), font=font(11), fill=colors["muted"])
    if x2 - x1 > 260:
        clock = f"{random.randint(0, 19)}:{random.randint(0, 59):02d}"
        draw.text((x2 - 80, y1 + 10), clock, font=font(20, True), fill=colors["text"])


def draw_captured(draw, x, y, colors):
    for k in range(random.randint(0, 8)):
        sym = random.choice(["p", "n", "b", "r", "q"])
        cx = x + k * 13
        draw.ellipse([cx, y + 3, cx + 10, y + 13], fill=colors["muted"])
        if sym in "bq":
            draw.rectangle([cx + 3, y - 1, cx + 7, y + 3], fill=colors["muted"])
    if random.random() < 0.4:
        draw.text((x + 120, y), f"+{random.randint(1, 8)}", font=font(12, True), fill=colors["accent"])


def draw_move_panel(draw, box, colors):
    x1, y1, x2, y2 = box
    panel(draw, box, colors)
    draw.text((x1 + 14, y1 + 12), "MOVES", font=font(13, True), fill=colors["text"])
    names = ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7", "Re1", "b5", "Bb3", "d6", "c3", "O-O", "h3", "Nb8"]
    y, hl = y1 + 42, random.randint(0, 8)
    for i in range(min(14, (y2 - y - 30) // 28)):
        if i == hl:
            draw.rectangle([x1 + 8, y, x2 - 8, y + 24], fill=rgba(colors["accent"], 200))
        draw.text((x1 + 12, y + 4), f"{i + 1}.", font=font(11), fill=colors["muted"])
        draw.text((x1 + 45, y + 4), names[(i * 2) % len(names)], font=font(12), fill=colors["text"])
        draw.text((x1 + 100, y + 4), names[(i * 2 + 1) % len(names)], font=font(12), fill=colors["text"])
        y += 28
    draw.text((x1 + 12, y2 - 24), random.choice(["Sicilian Defense", "Ruy Lopez", "Queen's Gambit", "Italian Game"]), font=font(10), fill=colors["muted"])


def draw_engine_panel(draw, box, colors):
    x1, y1, x2, y2 = box
    panel(draw, box, colors)
    draw.text((x1 + 12, y1 + 10), "ENGINE", font=font(12, True), fill=colors["text"])
    draw.text((x1 + 12, y1 + 32), random.choice(["Stockfish 18", "Stockfish", "Cloud analysis"]), font=font(11), fill=colors["muted"])
    draw.text((x2 - 62, y1 + 28), random.choice(["+0.3", "+0.8", "-0.4", "+1.2", "0.0", "-1.1"]), font=font(16, True), fill=colors["text"])
    y = y1 + 62
    for line in ["Nf3 Nc6 Bb5 a6", "d4 exd4 Qxd4", "Re1 Be7 O-O", "Bb3 d6 Nbd2"]:
        if y + 18 > y2:
            break
        draw.text((x1 + 12, y), line, font=font(10), fill=colors["muted"])
        y += 20


def draw_opening_table(draw, box, colors):
    x1, y1, x2, y2 = box
    panel(draw, box, colors)
    draw.text((x1 + 12, y1 + 10), "OPENING EXPLORER", font=font(11, True), fill=colors["text"])
    y = y1 + 34
    for mv in ["e4", "d4", "Nf3", "c4", "g3", "b3"]:
        if y + 22 > y2:
            break
        draw.text((x1 + 12, y + 3), mv, font=font(11), fill=colors["text"])
        bw = x2 - x1 - 110
        a, b = random.randint(25, 55), random.randint(20, 40)
        draw.rectangle([x1 + 60, y + 4, x1 + 60 + bw * a // 100, y + 18], fill=(235, 235, 235))
        draw.rectangle([x1 + 60 + bw * a // 100, y + 4, x1 + 60 + bw * (a + b) // 100, y + 18], fill=(130, 130, 130))
        draw.rectangle([x1 + 60 + bw * (a + b) // 100, y + 4, x1 + 60 + bw, y + 18], fill=(40, 40, 40))
        y += 26


def draw_chat(draw, box, colors):
    x1, y1, x2, y2 = box
    panel(draw, box, colors)
    draw.text((x1 + 12, y1 + 10), "CHAT", font=font(12, True), fill=colors["text"])
    y = y1 + 40
    for _ in range(6):
        if y + 34 > y2 - 44:
            break
        draw.text((x1 + 12, y), random_username()[:12], font=font(9, True), fill=colors["accent"])
        draw.text((x1 + 12, y + 14), random.choice(["Good game!", "Nice move", "gg", "Hello", "Good luck!"]), font=font(10), fill=colors["text"])
        y += 36
    rounded(draw, [x1 + 10, y2 - 38, x2 - 10, y2 - 10], 6, colors["panel2"])
    draw.text((x1 + 20, y2 - 30), "Write a message...", font=font(10), fill=colors["muted"])


def draw_info_card(draw, box, colors, title):
    x1, y1, x2, y2 = box
    panel(draw, box, colors)
    draw.text((x1 + 12, y1 + 10), title, font=font(11, True), fill=colors["text"])
    y = y1 + 34
    for lab in ["Players", "Rated game", "Rapid", "Online", "Spectators", "Moves", "Opening"]:
        if y + 20 > y2:
            break
        draw.text((x1 + 12, y), lab, font=font(9), fill=colors["muted"])
        draw.text((x2 - 90, y), random.choice(["Yes", "1,842", "10+0", "Online", "247", "32"]), font=font(9), fill=colors["text"])
        y += 24


def draw_menu_list(draw, box, colors):
    x1, y1, x2, y2 = box
    panel(draw, box, colors)
    items = random.sample(["Standard", "Crazyhouse", "Chess960", "King of the Hill", "Three-Check", "Antichess", "Atomic",
                           "Horde", "Racing Kings", "Rapid", "Blitz", "Bullet"], random.randint(5, 9))
    rh, y = random.randint(44, 62), y1 + 4
    for i, it in enumerate(items):
        if y + rh > y2:
            break
        if i == 0:
            draw.rectangle([x1 + 2, y, x2 - 2, y + rh], fill=colors["panel2"])
        rounded(draw, [x1 + 14, y + rh // 2 - 11, x1 + 36, y + rh // 2 + 11], 5, rgba(colors["muted"], 200))
        draw.text((x1 + 50, y + rh // 2 - 11), it, font=font(random.choice([18, 20, 22]), False), fill=colors["text"])
        y += rh


def draw_nav_icons(draw, box, colors):
    x1, y1, x2, y2 = box
    panel(draw, box, colors)
    y = y1 + 24
    for nm in ["home", "knight", "play", "star", "search", "menu", "gear"]:
        if y + 40 > y2:
            break
        icon(draw, nm, x1 + 28, y + 14, 20, colors["muted"], 2)
        if x2 - x1 > 120:
            draw.text((x1 + 54, y + 4), random.choice(["Play", "Puzzles", "Learn", "Watch", "News", "Social", "More"]), font=font(15, True), fill=colors["text"])
        y += 52


def draw_controls_row(draw, x, y, w, colors):
    names = ["first", "back", "play", "forward", "last"]
    bw = max(40, min(80, w // 8))
    for i, nm in enumerate(names):
        bx = x + i * (bw + 6)
        if bx + bw > x + w:
            break
        rounded(draw, [bx, y, bx + bw, y + 34], 5, rgba(colors["panel"], colors["palpha"]), rgba(colors["border"], 255))
        icon(draw, nm, bx + bw / 2, y + 17, 14, colors["muted"], 2)
    for k, t in enumerate(["FLIP", "BOARD", "SETTINGS"]):
        bx = x + w - (3 - k) * (bw + 8) - 4
        if bx > x + 5 * (bw + 6) + 10:
            rounded(draw, [bx, y, bx + bw + 4, y + 34], 5, rgba(colors["panel"], colors["palpha"]), rgba(colors["border"], 255))
            draw_text_center(draw, [bx, y, bx + bw + 4, y + 34], t, font(9, True), colors["muted"])


def draw_fen_box(draw, x, y, w, fen, colors):
    draw.text((x, y + 8), "FEN", font=font(13, True), fill=colors["muted"])
    rounded(draw, [x + 44, y, x + w, y + 34], 4, colors["panel"], colors["border"])
    draw.text((x + 56, y + 8), fen[:max(10, (w - 70) // 8)], font=font(13), fill=colors["text"])


def draw_mini_board(img, x, y, size, theme):
    if not COMPLETE_STYLES:
        return
    b = random_position()
    im = render_board(b, size, random.choice(COMPLETE_STYLES), theme, False, "none", 1.0, overlays=False)
    img.paste(im, (int(x), int(y)))


def draw_hard_negative(img, draw, box, colors):
    x1, y1, x2, y2 = [int(v) for v in box]
    kind = random.choice(["video", "checker", "table", "calendar", "image"])
    if kind == "video":
        draw.rectangle([x1, y1, x2, y2], fill=(8, 8, 10, 255))
        icon(draw, "play", (x1 + x2) / 2, (y1 + y2) / 2, 60, (240, 240, 240), 2)
        draw.rectangle([x1, y2 - 10, x2, y2], fill=(70, 70, 75, 255))
        draw.rectangle([x1, y2 - 10, x1 + (x2 - x1) * random.randint(10, 80) // 100, y2], fill=(220, 40, 40, 255))
    elif kind == "checker":
        n = random.choice([5, 6, 7, 9, 10, 12])                   # wrong number of squares -> NOT a chess board
        c1 = tuple(random.randint(120, 250) for _ in range(3))
        c2 = scale_c(c1, random.uniform(0.35, 0.7))
        s = (x2 - x1) / n
        for r in range(n):
            for c in range(n):
                draw.rectangle([x1 + c * s, y1 + r * s, x1 + (c + 1) * s, y1 + (r + 1) * s], fill=c1 if (r + c) % 2 == 0 else c2)
    elif kind == "table":
        rows, cols = random.randint(6, 14), random.randint(4, 9)
        rh, cw = (y2 - y1) / rows, (x2 - x1) / cols
        draw.rectangle([x1, y1, x2, y2], fill=rgba(colors["panel"], 255), outline=colors["border"])
        for r in range(rows):
            for c in range(cols):
                draw.rectangle([x1 + c * cw, y1 + r * rh, x1 + (c + 1) * cw, y1 + (r + 1) * rh], outline=colors["border"])
                if random.random() < 0.6:
                    draw.text((x1 + c * cw + 6, y1 + r * rh + 4), str(random.randint(0, 999)), font=font(11), fill=colors["text"])
    elif kind == "calendar":
        s = (x2 - x1) / 7
        for r in range(6):
            for c in range(7):
                draw.rectangle([x1 + c * s, y1 + r * s * 0.8, x1 + (c + 1) * s, y1 + (r + 1) * s * 0.8], fill=rgba(colors["panel"], 255), outline=colors["border"])
                draw.text((x1 + c * s + 5, y1 + r * s * 0.8 + 3), str(r * 7 + c + 1), font=font(11), fill=colors["muted"])
    else:
        n = value_noise(max(8, (x2 - x1) // 4), max(8, (y2 - y1) // 4), 20, 4)
        a = np.stack([n * random.randint(80, 255), n * random.randint(80, 255), n * random.randint(80, 255)], -1)
        img.paste(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).resize((x2 - x1, y2 - y1), Image.BICUBIC).convert("RGBA"), (x1, y1))


def draw_cursor(draw, x, y, k=1.0):
    pts = [(0, 0), (0, 17), (4, 13), (7, 20), (10, 19), (7, 12), (12, 12)]
    pts = [(x + a * k, y + b * k) for a, b in pts]
    draw.polygon(pts, fill=(255, 255, 255, 255), outline=(0, 0, 0, 255))


def draw_tooltip(draw, x, y, colors):
    t = random.choice(["Flip board", "Analysis", "Resign", "Last move", "Copy FEN"])
    f = font(12)
    tw, th = text_wh(draw, t, f)
    rounded(draw, [x, y, x + tw + 16, y + th + 12], 4, (20, 20, 20, 235))
    draw.text((x + 8, y + 5), t, font=f, fill=(240, 240, 240))


# ============================================================
# LAYOUT PLANNING
# ============================================================
def plan_layout(layout, W, Hc, top):
    avail = Hc - top
    margin = random.randint(12, 36)
    cards = (random.random() < (0.8 if layout in ("chesscom", "centered") else 0.45)) and layout != "minimal"
    card_h = random.randint(44, 60) if cards else 0
    below = random.choice([0, 0, 48, 90])                   # controls / FEN under the board
    left_w = right_w = 0
    if layout == "lichess":
        left_w, right_w = int(W * random.uniform(0.15, 0.24)), int(W * random.uniform(0.2, 0.3))
    elif layout == "chesscom":
        left_w, right_w = random.choice([0, random.randint(58, 80), int(W * 0.13)]), int(W * random.uniform(0.2, 0.3))
    elif layout == "centered":
        left_w, right_w = int(W * random.uniform(0.0, 0.2)), int(W * random.uniform(0.15, 0.28))
    elif layout == "left":
        left_w, right_w = random.choice([0, 60]), int(W * random.uniform(0.3, 0.46))
    max_h = avail - 2 * margin - (2 * (card_h + 8) if cards else 0) - below
    max_w = W - left_w - right_w - 3 * margin
    bs = int(min(max_h, max_w) * random.uniform(0.84, 1.0))
    if bs < 300:
        left_w = right_w = 0
        bs = int(min(max_h, W - 2 * margin) * random.uniform(0.85, 1.0))
    bs = int(clamp(bs, 220, 1100))
    free_w = max(0, W - left_w - right_w)
    if layout == "centered":
        bx = left_w + (free_w - bs) // 2 + random.randint(-40, 40)
    elif layout == "minimal":
        bx = (W - bs) // 2 + random.randint(-80, 80)
    else:
        bx = left_w + margin + random.randint(0, max(0, free_w - bs - 2 * margin) // 3)
    bx = int(clamp(bx, margin, max(margin, W - bs - margin)))
    spare = max(0, avail - 2 * margin - (2 * (card_h + 8) if cards else 0) - below - bs)
    by = top + margin + (card_h + 8 if cards else 0) + random.randint(0, spare // 2)
    by = int(clamp(by, top + 6, max(top + 6, Hc - bs - (card_h + 8 if cards else 0) - 8)))
    return dict(bx=bx, by=by, bs=bs, cards=cards, card_h=card_h, left_w=left_w, right_w=right_w, margin=margin)


def stack_panels(draw, box, colors, board, img, decoys_ok=True):
    x1, y1, x2, y2 = box
    if x2 - x1 < 170 or y2 - y1 < 150:
        return 0
    y, n_mini = y1, 0
    kinds = random.sample(["moves", "engine", "opening", "chat", "info"], random.randint(1, 3))
    for k in kinds:
        h = min(random.randint(200, 430), y2 - y)
        if h < 130:
            break
        b = [x1, y, x2, y + h]
        {"moves": draw_move_panel, "engine": draw_engine_panel, "opening": draw_opening_table, "chat": draw_chat}.get(k, lambda d, bb, c: draw_info_card(d, bb, c, random.choice(["GAME INFO", "PLAYERS", "TOURNAMENT", "GAME DETAILS"])))(draw, b, colors)
        y += h + 12
    if decoys_ok and random.random() < 0.28 and y2 - y > 110 and x2 - x1 > 120:      # mini-board decoys
        for _ in range(random.randint(1, min(3, max(1, (x2 - x1) // 130)))):
            sz = random.randint(90, min(170, y2 - y - 8, x2 - x1 - 8))
            theme = random.choice(BOARD_THEMES)
            draw_mini_board(img, x1 + n_mini * (sz + 10), y + 4, sz, theme)
            n_mini += 1
            if x1 + n_mini * (sz + 10) + sz > x2:
                break
    return n_mini


# ============================================================
# PHOTOMETRICS
# ============================================================
def finalize(img):
    p = {"gray": False, "gamma": 1.0}
    if random.random() < 0.25:                                       # soft HiDPI-style resampling
        f = random.uniform(0.55, 0.92)
        w, h = img.size
        img = img.resize((max(8, int(w * f)), max(8, int(h * f))), Image.LANCZOS).resize((w, h), Image.BICUBIC)
    r = random.random()
    gamma = random.uniform(0.62, 0.82) if r < 0.09 else random.uniform(1.25, 1.7) if r < 0.19 else 1.0
    if gamma != 1.0:
        lut = [int(255 * ((i / 255.0) ** gamma)) for i in range(256)] * 3
        img = img.point(lut)
    p["gamma"] = round(gamma, 2)
    b = random.uniform(0.5, 0.78) if random.random() < 0.07 else random.uniform(1.2, 1.4) if random.random() < 0.06 else random.uniform(0.88, 1.12)
    img = ImageEnhance.Brightness(img).enhance(b)
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.62, 0.82) if random.random() < 0.06 else random.uniform(0.85, 1.18))
    if random.random() < 0.14:
        img = ImageEnhance.Color(img).enhance(0.0)
        p["gray"] = True
    else:
        img = ImageEnhance.Color(img).enhance(random.uniform(0.7, 1.2))
    if random.random() < 0.08:                                       # night-light / cool tint
        k = random.choice([(1.0, 0.93, 0.8), (0.86, 0.94, 1.0)])
        ch = img.split()
        img = Image.merge("RGB", [c.point(lambda v, kk=kk: int(clamp(v * kk, 0, 255))) for c, kk in zip(ch, k)])
    if random.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.4, 1.1)))
    if random.random() < 0.25:
        arr = np.asarray(img).astype(np.float32)
        arr = np.clip(arr + np.random.normal(0, random.uniform(2, 9), arr.shape), 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    if random.random() < 0.6:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(45, 95))
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img, p


# ============================================================
# ONE IMAGE
# ============================================================
def pick_screen():
    sizes, weights = zip(*SCREEN_SIZES)
    return random.choices(sizes, weights=weights)[0]


def choose_board_theme(low_contrast, bg_lum):
    if low_contrast:
        w = [math.exp(-abs((lum(a) + lum(b)) / 2 - bg_lum) / 28.0) + 0.02 for _, a, b, _ in BOARD_THEMES]
        return random.choices(BOARD_THEMES, weights=w)[0]
    return random.choice(BOARD_THEMES)


SEED_BASE = SEED


def seed_image(index):
    """Every image has its own seed: reproducible, independent of worker count, resumable."""
    v = (SEED_BASE * 1_000_003 + index) % (2 ** 32)
    random.seed(v)
    np.random.seed(v)


def generate_image(index):
    seed_image(index)
    _FAMILY[0] = random.randint(0, len(FONT_FAMILIES) - 1)
    W, H = pick_screen()
    sample_kind = wchoice(SAMPLE_KIND_WEIGHTS)
    has_board = sample_kind != "no_board"
    view = wchoice(VIEW_WEIGHTS)
    layout = wchoice(LAYOUT_WEIGHTS)
    website_theme = random.choice(list(WEBSITE_THEMES))
    colors = dict(WEBSITE_THEMES[website_theme])
    img, bg_style = make_background(W, H, colors)
    colors["palpha"] = 255 if bg_style in ("flat", "pattern") and random.random() < 0.8 else random.randint(175, 238)
    draw = ImageDraw.Draw(img, "RGBA")

    chrome_name, browser_h, taskbar_h = "", 0, 0
    if view == "full":
        chrome_name = random.choice(list(CHROMES))
        browser_h = draw_browser(draw, W, CHROMES[chrome_name])
        if random.random() < 0.25:
            taskbar_h = draw_taskbar(draw, W, H)
    Hc = H - taskbar_h
    top = browser_h
    if random.random() < 0.85:
        top = draw_website_header(draw, browser_h, W, colors)

    plan = plan_layout(layout, W, Hc, top)
    bx, by, bs = plan["bx"], plan["by"], plan["bs"]
    if sample_kind == "partial_crop":
        frac, edge = random.uniform(0.25, 0.55), random.choice(["left", "right", "top", "bottom"])
        if edge == "left": bx = -int(bs * frac)
        elif edge == "right": bx = W - bs + int(bs * frac)
        elif edge == "top": by = -int(bs * frac)
        else: by = H - bs + int(bs * frac)

    # ---- side panels
    card_pad = plan["card_h"] + 8 if plan["cards"] else 0
    ptop = max(top + 8, by - card_pad)
    n_decoys = 0
    if plan["left_w"] >= 150 and bx - 20 > 150 and random.random() < 0.9:
        lb = [10, ptop, min(bx - 20, plan["left_w"] - 6 if layout != "left" else bx - 20), Hc - 12]
        if layout == "chesscom" and plan["left_w"] < 100:
            draw_nav_icons(draw, [0, top, plan["left_w"], Hc], colors)
        elif lb[2] - lb[0] > 120:
            (draw_menu_list if layout == "lichess" or random.random() < 0.4 else draw_nav_icons)(draw, lb, colors)
    elif plan["left_w"] and plan["left_w"] < 100:
        draw_nav_icons(draw, [0, top, plan["left_w"], Hc], colors)
    eval_w = 0
    if layout in ("lichess", "chesscom", "centered") and random.random() < 0.18 and bx > 70:
        eval_w = random.randint(16, 26)
    rx1 = bx + bs + 18
    if random.random() < 0.9 and W - rx1 - 14 > 170:
        n_decoys += stack_panels(draw, [rx1, ptop, W - 14, Hc - 12], colors, None, img)

    # ---- board
    board = random_position()
    piece_style = random.choice(COMPLETE_STYLES)
    low_contrast = random.random() < LOW_CONTRAST_PROB
    reg = img.crop((max(0, bx), max(0, by), min(W, bx + bs), min(H, by + bs)))
    bg_lum = ImageStat.Stat(reg.convert("L")).mean[0] if reg.width > 4 and reg.height > 4 else lum(colors["bg"])
    theme = choose_board_theme(low_contrast, bg_lum)
    bright = random.choice([1.0, 1.0, 1.0, 1.0, random.uniform(0.9, 1.1)]) if random.random() < 0.82 else (random.uniform(0.35, 0.7) if random.random() < 0.65 else random.uniform(1.12, 1.3))
    coord_style = random.choices(["lichess", "chesscom", "v6", "outside", "none"], weights=[0.28, 0.28, 0.14, 0.10, 0.20])[0]
    flipped = random.random() < 0.15
    frame = random.choices(["none", "thin", "thick", "wood", "light"], weights=[0.34, 0.30, 0.12, 0.12, 0.12])[0]
    rcorner = random.choice([0, 0, 0, 4, 6, 8])
    border_label = {"none": "none", "thin": "thin", "thick": "thick", "wood": "thick", "light": "thin"}[frame]
    page_type = random.choice(PAGE_TYPES)

    if has_board:
        board_shadow(img, bx, by, bs)
        draw_frame(draw, bx, by, bs, frame, colors)
        bimg = render_board(board, bs, piece_style, theme, flipped, coord_style if coord_style != "outside" else "none", bright)
        if rcorner:
            m = Image.new("L", bimg.size, 0)
            ImageDraw.Draw(m).rounded_rectangle([0, 0, bs - 1, bs - 1], radius=rcorner, fill=255)
            img.paste(bimg, (bx, by), m)
        else:
            img.paste(bimg, (bx, by))
        if coord_style == "outside":
            outside_coords(draw, bx, by, bs, flipped, colors)
        if eval_w:
            ex = bx - eval_w - random.randint(6, 14)
            frac = random.uniform(0.15, 0.85)
            draw.rectangle([ex, by, ex + eval_w, by + bs], fill=(40, 40, 40, 255))
            draw.rectangle([ex, by + int(bs * (1 - frac)), ex + eval_w, by + bs], fill=(240, 240, 240, 255))
        if plan["cards"]:
            if by - plan["card_h"] - 8 > top:
                draw_player_card(draw, [bx, by - plan["card_h"] - 8, bx + bs, by - 8], colors)
            if by + bs + 8 + plan["card_h"] < Hc:
                draw_player_card(draw, [bx, by + bs + 8, bx + bs, by + bs + 8 + plan["card_h"]], colors)
        elif random.random() < 0.7:
            draw_captured(draw, bx + 6, max(top + 4, by - 26), colors)
        yb = by + bs + (plan["card_h"] + 16 if plan["cards"] else 14)
        if yb + 40 < Hc:
            if random.random() < 0.7:
                draw_controls_row(draw, bx, yb, bs, colors)
                yb += 46
            if yb + 40 < Hc and random.random() < 0.5:
                draw_fen_box(draw, bx, yb, bs, board.fen(), colors)
        ty = by - card_pad - 40
        if ty > top + 4 and random.random() < 0.6:
            titles = {"game": ["Rated Rapid Game", "Casual Chess", "Blitz Game"], "analysis": ["Game Analysis", "Analyze Position"],
                      "replay": ["Game Replay", "Replay"], "tournament": ["Live Tournament", "Arena"], "puzzle": ["Daily Puzzle", "Puzzle Rush"],
                      "watch": ["Watch Live", "Top Games"], "community": ["Chess Club", "Friends"]}
            draw.text((bx, ty), random.choice(titles[page_type]), font=font(18, True), fill=colors["text"])
    else:
        if random.random() < 0.65:
            draw_hard_negative(img, draw, [bx, by, bx + bs, by + bs], colors)
        page_type = random.choice(PAGE_TYPES)

    # ---- clutter
    if random.random() < 0.45:                                         # bottom banner / cookie bar
        h = random.randint(44, 74)
        box = [random.randint(10, max(10, W // 4)), Hc - h - random.randint(5, 24), random.randint(W // 2, W - 20), Hc - 8]
        if not rect_intersects(box, (bx, by, bx + bs, by + bs), 12):
            panel(draw, box, colors)
            draw.text((box[0] + 12, box[1] + 10), random.choice(["We use cookies", "Upgrade to Premium", "Tournament starts soon", "Game saved"]), font=font(12, True), fill=colors["text"])
    if random.random() < 0.15 and W - bx - bs > 330:                     # ad
        box = [W - 330, top + 20, W - 20, top + 270]
        if not rect_intersects(box, (bx, by, bx + bs, by + bs), 12):
            draw.rectangle(box, fill=tuple(random.randint(30, 230) for _ in range(3)) + (255,))
            draw_text_center(draw, box, "AD", font(28, True), (255, 255, 255))
    if random.random() < 0.22 and layout != "minimal":                   # dropdown menu
        dx, dy = random.randint(10, max(20, W // 3)), top + random.randint(0, 10)
        box = [dx, dy, dx + random.randint(170, 240), dy + random.randint(150, 260)]
        if not rect_intersects(box, (bx, by, bx + bs, by + bs), 12):
            panel(draw, box, colors, key="panel2")
            for k, s in enumerate(["Arena tournaments", "Swiss tournaments", "Simultaneous exhibitions", "Create game"]):
                if box[1] + 14 + k * 38 + 20 < box[3]:
                    draw.text((box[0] + 14, box[1] + 14 + k * 38), s, font=font(14), fill=colors["text"])
    if random.random() < 0.65:                                           # page scrollbar
        sw = random.choice([8, 10, 12, 15])
        draw.rectangle([W - sw, top, W, Hc], fill=rgba(colors["panel2"], 255))
        th = random.randint(80, max(100, H // 3))
        ty2 = random.randint(top, max(top, Hc - th))
        rounded(draw, [W - sw, ty2, W, ty2 + th], 3, rgba(colors["muted"], 255))
    if has_board and random.random() < 0.05:                              # modal dim over the whole page
        draw.rectangle([0, top, W, Hc], fill=(0, 0, 0, random.randint(60, 130)))
    if has_board and random.random() < 0.35:                              # mouse cursor
        draw_cursor(draw, random.randint(bx, bx + bs - 20) if random.random() < 0.6 else random.randint(0, W - 20),
                    random.randint(by, by + bs - 20) if random.random() < 0.6 else random.randint(top, Hc - 20), random.choice([1.0, 1.0, 1.4]))
    if has_board and random.random() < 0.08:
        draw_tooltip(draw, bx + random.randint(0, max(1, bs - 100)), by + random.randint(0, max(1, bs - 40)), colors)

    # ---- label (visible region) and view cropping
    if has_board:
        vx1, vy1, vx2, vy2 = clamp(bx, 0, W), clamp(by, 0, H), clamp(bx + bs, 0, W), clamp(by + bs, 0, H)
    else:
        vx1 = vy1 = vx2 = vy2 = 0
    out = img.convert("RGB")
    if view == "tight" and has_board:
        ml, mt, mr, mb = [random.randint(0, 70) for _ in range(4)]
        cb = (max(0, bx - ml), max(0, by - mt), min(W, bx + bs + mr), min(H, by + bs + mb))
        out = out.crop(cb)
        vx1, vy1, vx2, vy2 = vx1 - cb[0], vy1 - cb[1], vx2 - cb[0], vy2 - cb[1]
        W, H = out.size
    final, ph = finalize(out)
    return final, {
        "has_board": has_board, "sample_kind": sample_kind,
        "x1": round(vx1, 1), "y1": round(vy1, 1), "x2": round(vx2, 1), "y2": round(vy2, 1),
        "screen_width": W, "screen_height": H, "page_type": page_type, "website_theme": website_theme,
        "board_theme": theme[0] if has_board else "", "piece_style": piece_style if has_board else "",
        "coordinates": (coord_style != "none") if has_board else "", "border": border_label if has_board else "none",
        "flipped": flipped if has_board else "", "fen": board.fen() if has_board else "",
        "layout": layout, "view": view, "bg_style": bg_style, "chrome": chrome_name,
        "board_kind": theme[3] if has_board else "", "board_brightness": round(bright, 2) if has_board else "",
        "low_contrast": low_contrast if has_board else "", "gray": ph["gray"], "gamma": ph["gamma"], "decoys": n_decoys,
    }


def save_image(final, index):
    IMAGE_DIR = OUTPUT_DIR / "images"
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{index:06d}.jpg" if SAVE_JPEG else f"{index:06d}.png"
    if SAVE_JPEG:
        final.save(IMAGE_DIR / name, quality=random.randint(*JPEG_QUALITY_RANGE), optimize=True)
    else:
        final.save(IMAGE_DIR / name, optimize=True)
    return name


def _init_worker(seed, out_dir, piece_root):
    global SEED_BASE, OUTPUT_DIR, PIECE_ROOT, COMPLETE_STYLES
    SEED_BASE, OUTPUT_DIR, PIECE_ROOT = seed, Path(out_dir), Path(piece_root)
    COMPLETE_STYLES = find_complete_styles()


def _work(i):
    try:
        final, row = generate_image(i)
        row["filename"] = save_image(final, i)
        return i, row, None
    except Exception as e:                       # keep going, report at the end
        return i, None, repr(e)


def generate_dataset(n, workers=1):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nChess board detector dataset v7: {n} images, {workers} worker(s), piece styles: {COMPLETE_STYLES}\n")
    results = []
    if workers > 1:
        import multiprocessing as mp
        with mp.Pool(workers, initializer=_init_worker, initargs=(SEED_BASE, str(OUTPUT_DIR), str(PIECE_ROOT))) as pool:
            for k, res in enumerate(pool.imap_unordered(_work, range(n), chunksize=8), 1):
                results.append(res)
                if k % 200 == 0:
                    print(f"Generated {k}/{n}")
    else:
        for i in range(n):
            results.append(_work(i))
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{n}")
    rows = [r for _, r, e in sorted(results, key=lambda t: t[0]) if r is not None]
    for i, r, e in results:
        if e:
            print(f"[ERROR] image {i}: {e}")
    with open(OUTPUT_DIR / "labels.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nDONE: {len(rows)} images in {(OUTPUT_DIR / 'images').resolve()}\nLabels: {(OUTPUT_DIR / 'labels.csv').resolve()}")


def preview(n, path="preview_v7.jpg", cols=4):
    """Contact sheet with the label box drawn in green (no files are written to the dataset folder)."""
    cell_w = 560
    tiles = []
    for i in range(n):
        final, row = generate_image(i)
        im = final.convert("RGB")
        if row["has_board"]:
            ImageDraw.Draw(im).rectangle([row["x1"], row["y1"], row["x2"], row["y2"]], outline=(0, 255, 0), width=max(3, im.width // 300))
        k = cell_w / im.width
        im = im.resize((cell_w, int(im.height * k)), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        tag = f"{row['layout']} | {row['view']} | bg={row['bg_style']} | {row['board_theme']} x{row['board_brightness']} {'LOWC' if row['low_contrast'] is True else ''} {'GRAY' if row['gray'] else ''} g={row['gamma']}" if row["has_board"] else "NO BOARD"
        d.rectangle([0, 0, cell_w, 16], fill=(0, 0, 0))
        d.text((4, 2), tag, fill=(255, 255, 0), font=ImageFont.load_default())
        tiles.append(im)
    rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
    heights = [max(t.height for t in r) for r in rows]
    sheet = Image.new("RGB", (cols * cell_w, sum(heights)), (20, 20, 20))
    y = 0
    for r, h in zip(rows, heights):
        for j, t in enumerate(r):
            sheet.paste(t, (j * cell_w, y))
        y += h
    sheet.save(path, quality=90)
    print(f"preview written to {Path(path).resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=NUM_IMAGES)
    ap.add_argument("--out", default=str(OUTPUT_DIR))
    ap.add_argument("--piece-root", default=str(PIECE_ROOT))
    ap.add_argument("--preview", type=int, default=0, help="render N images into preview_v7.jpg and exit")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--workers", type=int, default=1, help="processes (use your CPU core count minus 1)")
    a = ap.parse_args()
    SEED_BASE = a.seed
    OUTPUT_DIR = Path(a.out)
    PIECE_ROOT = Path(a.piece_root)
    COMPLETE_STYLES = find_complete_styles()
    if a.preview:
        preview(a.preview)
    else:
        generate_dataset(a.n, a.workers)
