from pathlib import Path
import random
import csv
import math
import io

import chess
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np


# ============================================================
# CONFIG
# ============================================================

NUM_IMAGES = 10000          # TEST FIRST -> change to 100_000 later
OUTPUT_DIR = Path("dataset_v6")

PIECE_ROOT = Path("../pieces_styles")

IMAGE_DIR = OUTPUT_DIR / "images"
LABEL_FILE = OUTPUT_DIR / "labels.csv"

SAVE_JPEG = True
JPEG_QUALITY = 92

SEED = 42
random.seed(SEED)

# fraction of images that are: a normal full board / board partly off-canvas
# / no board at all (negative example)
SAMPLE_KIND_WEIGHTS = {
    "normal": 0.85,        
    "partial_crop": 0.0,  
    "no_board": 0.10,     
}


SCREEN_SIZES = [
    (1280, 720),
    (1366, 768),
    (1440, 900),
    (1920, 1080),
]


PIECE_STYLES = [
    "cburnett",
    "alpha",
    "merida",
    "companion",
    "neo_64",
]


BOARD_THEMES = [
    ("brown", (240, 217, 181), (181, 136, 99)),
    ("blue", (222, 227, 230), (90, 120, 150)),
    ("green", (235, 236, 208), (119, 149, 86)),
    ("gray", (225, 225, 225), (120, 120, 120)),
    ("purple", (235, 225, 240), (135, 105, 150)),
    ("red", (240, 220, 215), (155, 90, 80)),
    ("dark", (190, 190, 190), (80, 80, 80)),
]


WEBSITE_THEMES = {
    "dark": {
        "bg": (24, 26, 30),
        "panel": (34, 37, 43),
        "panel2": (43, 46, 53),
        "text": (235, 235, 235),
        "muted": (150, 155, 165),
        "accent": (80, 150, 240),
        "border": (60, 64, 72),
    },

    "light": {
        "bg": (238, 240, 243),
        "panel": (255, 255, 255),
        "panel2": (245, 246, 248),
        "text": (35, 38, 42),
        "muted": (110, 115, 125),
        "accent": (55, 110, 200),
        "border": (210, 212, 216),
    },

    "blue": {
        "bg": (20, 32, 48),
        "panel": (29, 46, 68),
        "panel2": (38, 58, 82),
        "text": (235, 242, 250),
        "muted": (155, 175, 195),
        "accent": (70, 150, 230),
        "border": (55, 75, 100),
    },

    "purple": {
        "bg": (31, 25, 40),
        "panel": (46, 36, 57),
        "panel2": (59, 47, 72),
        "text": (240, 235, 245),
        "muted": (170, 155, 180),
        "accent": (155, 105, 220),
        "border": (75, 62, 90),
    },

    "gray": {
        "bg": (55, 57, 60),
        "panel": (70, 72, 76),
        "panel2": (82, 84, 89),
        "text": (235, 235, 235),
        "muted": (170, 170, 175),
        "accent": (120, 150, 190),
        "border": (95, 97, 102),
    },
}


# ============================================================
# FONTS
# ============================================================

def font(size, bold=False):
    candidates = []

    if bold:
        candidates += [
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/segoeuib.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
    else:
        candidates += [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]

    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)

    return ImageFont.load_default()


# ============================================================
# HELPERS
# ============================================================

def clamp(v, a, b):
    return max(a, min(v, b))


def rect_intersects(a, b, margin=0):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ax1 -= margin
    ay1 -= margin
    ax2 += margin
    ay2 += margin

    return not (
        ax2 <= bx1 or
        ax1 >= bx2 or
        ay2 <= by1 or
        ay1 >= by2
    )


def draw_text_center(draw, box, text, fnt, fill):
    x1, y1, x2, y2 = box

    bbox = draw.textbbox((0, 0), text, font=fnt)

    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    draw.text(
        (
            (x1 + x2) / 2 - tw / 2,
            (y1 + y2) / 2 - th / 2,
        ),
        text,
        font=fnt,
        fill=fill,
    )


def rounded(draw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(
        box,
        radius=radius,
        fill=fill,
        outline=outline,
        width=width,
    )


def random_username():
    names = [
        "ChessMaster",
        "KnightStorm",
        "BlueKnight",
        "QueenHunter",
        "RookAttack",
        "ChessWizard",
        "TacticalFox",
        "EndgameKing",
        "RapidPlayer",
        "DarkKnight",
        "BoardVision",
        "AlphaChess",
        "GrandMaster",
        "ChessFan",
        "PuzzleKing",
    ]

    return random.choice(names) + str(random.randint(10, 9999))


def random_title():
    return random.choice([
        "",
        "",
        "",
        "GM",
        "IM",
        "FM",
        "CM",
        "NM",
    ])


# ============================================================
# PHOTOMETRIC AUGMENTATION (new)
# ============================================================

def photometric_augment(img_rgb):
    """img_rgb: PIL Image in RGB mode. Randomized brightness/contrast,
    occasional blur/noise, occasional lossy re-compression - mimics the
    variety real screenshots pick up (recompression, scaling, capture
    tool differences)."""
    img = ImageEnhance.Brightness(img_rgb).enhance(random.uniform(0.85, 1.15))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.15))

    if random.random() < 0.25:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.4, 1.2)))

    if random.random() < 0.25:
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, random.uniform(3, 10), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    if random.random() < 0.6:
        quality = random.randint(45, 95)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

    return img


# ============================================================
# PIECES
# ============================================================

PIECE_FILES = [
    "wK", "wQ", "wR", "wB", "wN", "wP",
    "bK", "bQ", "bR", "bB", "bN", "bP",
]


def find_complete_styles():
    complete = []

    if not PIECE_ROOT.exists():
        raise FileNotFoundError(
            f"Piece directory not found: {PIECE_ROOT.resolve()}"
        )

    for style in PIECE_STYLES:

        folder = PIECE_ROOT / style

        if not folder.exists():
            continue

        if all((folder / f"{p}.png").exists() for p in PIECE_FILES):
            complete.append(style)

    if not complete:
        raise RuntimeError(
            "No complete piece style found.\n"
            f"Checked: {PIECE_ROOT.resolve()}"
        )

    return complete


COMPLETE_STYLES = find_complete_styles()


def load_piece(style, piece, size):
    path = PIECE_ROOT / style / f"{piece}.png"

    img = Image.open(path).convert("RGBA")

    img.thumbnail((size, size), Image.Resampling.LANCZOS)

    return img


# ============================================================
# CHESS POSITION
# ============================================================

def random_position():
    board = chess.Board()

    moves = random.randint(0, 100)

    for _ in range(moves):
        legal = list(board.legal_moves)

        if not legal:
            break

        board.push(random.choice(legal))

    return board


# ============================================================
# BROWSER
# ============================================================

def draw_browser(draw, W, H, theme):
    browser_h = random.randint(80, 150)

    browser_bg = (
        random.randint(40, 70),
        random.randint(40, 70),
        random.randint(40, 70),
    )

    draw.rectangle(
        [0, 0, W, browser_h],
        fill=browser_bg,
    )

    # Tabs
    tab_y = 8
    tab_h = 32

    x = 10

    tab_names = [
        "Chess",
        "Game",
        "Play",
        "Analysis",
        "YouTube",
        "New Tab",
    ]

    num_tabs = random.randint(2, 6)
    active_tab_idx = random.randint(0, num_tabs - 1)  # fixed: was re-rolled per tab

    for i in range(num_tabs):

        width = random.randint(120, 220)

        active = i == active_tab_idx

        fill = (
            (70, 72, 78)
            if active
            else (48, 50, 55)
        )

        rounded(
            draw,
            [x, tab_y, x + width, tab_y + tab_h],
            7,
            fill,
        )

        draw.text(
            (x + 12, tab_y + 8),
            random.choice(tab_names),
            font=font(13),
            fill=(225, 225, 225),
        )

        draw.text(
            (x + width - 22, tab_y + 8),
            "×",
            font=font(13),
            fill=(180, 180, 180),
        )

        x += width + 5

        if x > W - 150:
            break

    # Navigation row
    nav_y = browser_h - 45

    draw.text(
        (15, nav_y + 7),
        "←",
        font=font(25),
        fill=(220, 220, 220),
    )

    draw.text(
        (50, nav_y + 7),
        "→",
        font=font(25),
        fill=(170, 170, 170),
    )

    draw.text(
        (85, nav_y + 8),
        "⟳",
        font=font(22),
        fill=(220, 220, 220),
    )

    # URL bar
    url_x = 125
    url_w = W - 270

    rounded(
        draw,
        [url_x, nav_y, url_x + url_w, nav_y + 35],
        17,
        (245, 245, 245),
    )

    draw.text(
        (url_x + 15, nav_y + 9),
        "🔒   chess.example.com/game/8fj29s",
        font=font(13),
        fill=(70, 70, 70),
    )

    # Browser icons
    draw.text(
        (W - 125, nav_y + 7),
        "☆",
        font=font(22),
        fill=(220, 220, 220),
    )

    draw.text(
        (W - 90, nav_y + 7),
        "◉",
        font=font(20),
        fill=(220, 220, 220),
    )

    draw.text(
        (W - 55, nav_y + 7),
        "⋮",
        font=font(22),
        fill=(220, 220, 220),
    )

    # bookmarks row sometimes
    if random.random() < 0.55:

        bookmark_y = browser_h

        draw.rectangle(
            [0, bookmark_y, W, bookmark_y + 25],
            fill=(
                browser_bg[0] + 10,
                browser_bg[1] + 10,
                browser_bg[2] + 10,
            ),
        )

        bookmarks = [
            "Chess",
            "News",
            "Games",
            "History",
            "Analysis",
            "Videos",
            "Favorites",
        ]

        x = 20

        for item in bookmarks:
            draw.text(
                (x, bookmark_y + 5),
                item,
                font=font(11),
                fill=(190, 190, 190),
            )

            x += random.randint(75, 120)

    return browser_h


# ============================================================
# WEBSITE HEADER
# ============================================================

def draw_website_header(draw, y, W, colors):
    h = random.randint(50, 85)

    draw.rectangle(
        [0, y, W, y + h],
        fill=colors["panel"],
    )

    # logo
    draw.text(
        (20, y + 16),
        random.choice([
            "♞ CHESS",
            "CHESS",
            "♟ PLAY",
            "BOARD",
        ]),
        font=font(21, True),
        fill=colors["text"],
    )

    nav = [
        "PLAY",
        "PUZZLES",
        "WATCH",
        "LEARN",
        "COMMUNITY",
    ]

    x = 180

    for item in nav:

        draw.text(
            (x, y + 22),
            item,
            font=font(12, True),
            fill=colors["muted"],
        )

        x += random.randint(75, 105)

    # search
    search_w = random.randint(150, 240)

    sx = W - search_w - 210

    rounded(
        draw,
        [sx, y + 13, sx + search_w, y + h - 13],
        15,
        colors["panel2"],
    )

    draw.text(
        (sx + 14, y + 20),
        "🔍 Search",
        font=font(12),
        fill=colors["muted"],
    )

    # icons
    draw.text(
        (W - 175, y + 20),
        "♟",
        font=font(20),
        fill=colors["text"],
    )

    draw.text(
        (W - 135, y + 20),
        "●",
        font=font(18),
        fill=colors["text"],
    )

    draw.text(
        (W - 100, y + 20),
        "☰",
        font=font(18),
        fill=colors["text"],
    )

    # avatar
    rounded(
        draw,
        [W - 55, y + 12, W - 15, y + 52],
        20,
        colors["accent"],
    )

    return y + h


# ============================================================
# BOARD POSITION
# ============================================================

def choose_board_position(W, H, browser_h, header_h):
    board_size = random.randint(
        min(400, H - 250),
        min(800, H - 120),
    )

    board_size = min(board_size, H - header_h - 60)

    # Reserve enough space for side panels
    layout = random.choice([
        "center",
        "left",
        "center",
        "center",
    ])

    if layout == "left":

        min_x = 50
        max_x = max(min_x, W // 2 - board_size - 30)

    else:

        min_x = max(30, (W - board_size) // 2 - 150)
        max_x = min(
            W - board_size - 30,
            (W - board_size) // 2 + 150
        )

    if max_x < min_x:
        x = max(20, (W - board_size) // 2)
    else:
        x = random.randint(min_x, max_x)

    y_min = header_h + 40
    y_max = H - board_size - 45

    if y_max < y_min:
        y = y_min
    else:
        y = random.randint(y_min, y_max)

    return x, y, board_size


# ============================================================
# BOARD
# ============================================================

def draw_board(
    img,
    draw,
    board,
    x,
    y,
    size,
    style,
    theme_name,
    coordinates,
    flipped,
):
    light, dark = next(
        (a, b)
        for name, a, b in BOARD_THEMES
        if name == theme_name
    )

    square = size / 8

    # Shadow
    shadow = Image.new(
        "RGBA",
        img.size,
        (0, 0, 0, 0),
    )

    sd = ImageDraw.Draw(shadow)

    sd.rounded_rectangle(
        [
            x + 10,
            y + 12,
            x + size + 10,
            y + size + 12,
        ],
        radius=8,
        fill=(0, 0, 0, 100),
    )

    shadow = shadow.filter(
        ImageFilter.GaussianBlur(12)
    )

    img.alpha_composite(shadow)

    # Board
    for row in range(8):

        for col in range(8):

            px = x + col * square
            py = y + row * square

            color = light if (row + col) % 2 == 0 else dark

            draw.rectangle(
                [
                    px,
                    py,
                    px + square + 1,
                    py + square + 1,
                ],
                fill=color,
            )

    # Last move
    if random.random() < 0.75 and board.move_stack:

        move = board.peek()

        for sq in [move.from_square, move.to_square]:

            file = chess.square_file(sq)
            rank = chess.square_rank(sq)

            if flipped:
                col = 7 - file
                row = rank
            else:
                col = file
                row = 7 - rank

            px = x + col * square
            py = y + row * square

            overlay = Image.new(
                "RGBA",
                img.size,
                (0, 0, 0, 0),
            )

            od = ImageDraw.Draw(overlay)

            od.rectangle(
                [
                    px,
                    py,
                    px + square,
                    py + square,
                ],
                fill=(245, 205, 60, 90),
            )

            img.alpha_composite(overlay)

    # Selected square
    if random.random() < 0.30:

        col = random.randint(0, 7)
        row = random.randint(0, 7)

        px = x + col * square
        py = y + row * square

        draw.rectangle(
            [
                px,
                py,
                px + square,
                py + square,
            ],
            fill=(80, 160, 240, 80),
        )

    # Pieces
    piece_size = int(square * 0.88)

    for row in range(8):

        for col in range(8):

            if flipped:
                file = 7 - col
                rank = row
            else:
                file = col
                rank = 7 - row

            sq = chess.square(file, rank)

            piece = board.piece_at(sq)

            if piece is None:
                continue

            code = (
                ("w" if piece.color else "b")
                + piece.symbol().upper()
            )

            piece_img = load_piece(
                style,
                code,
                piece_size,
            )

            px = int(
                x + col * square +
                (square - piece_img.width) / 2
            )

            py = int(
                y + row * square +
                (square - piece_img.height) / 2
            )

            img.alpha_composite(
                piece_img,
                (px, py),
            )

    # Coordinates
    if coordinates:

        coord_font = font(
            max(9, int(square * 0.16)),
            True,
        )

        for i in range(8):

            file_letter = "abcdefgh"[i]

            if flipped:
                file_letter = "abcdefgh"[7 - i]

            draw.text(
                (
                    x + i * square + 3,
                    y + size - 17,
                ),
                file_letter,
                font=coord_font,
                fill=dark if i % 2 == 0 else light,
            )

            rank_number = str(
                i + 1 if flipped else 8 - i
            )

            draw.text(
                (
                    x + 3,
                    y + i * square + 2,
                ),
                rank_number,
                font=coord_font,
                fill=light if i % 2 == 0 else dark,
            )

    # Border
    border_type = random.choice([
        "none",
        "thin",
        "thin",
        "thick",
    ])

    if border_type == "thin":

        draw.rectangle(
            [x, y, x + size, y + size],
            outline=(30, 30, 30),
            width=2,
        )

    elif border_type == "thick":

        draw.rectangle(
            [x - 5, y - 5, x + size + 5, y + size + 5],
            outline=(35, 35, 35),
            width=5,
        )

    return border_type


# ============================================================
# PLAYER CARD
# ============================================================

def draw_player_card(draw, box, colors, top=True):
    x1, y1, x2, y2 = box

    rounded(
        draw,
        box,
        8,
        colors["panel"],
        colors["border"],
        1,
    )

    avatar_size = min(42, y2 - y1 - 12)

    rounded(
        draw,
        [
            x1 + 10,
            y1 + 7,
            x1 + 10 + avatar_size,
            y1 + 7 + avatar_size,
        ],
        avatar_size // 2,
        colors["accent"],
    )

    title = random_title()
    name = random_username()
    rating = random.randint(1100, 2800)

    text_x = x1 + avatar_size + 20

    if title:
        draw.text(
            (text_x, y1 + 8),
            title,
            font=font(11, True),
            fill=colors["accent"],
        )

        text_x += 30

    draw.text(
        (text_x, y1 + 8),
        name,
        font=font(13, True),
        fill=colors["text"],
    )

    draw.text(
        (text_x, y1 + 28),
        str(rating),
        font=font(11),
        fill=colors["muted"],
    )

    # Clock
    clock = random.choice([
        f"{random.randint(0, 9)}:{random.randint(0, 59):02d}",
        f"{random.randint(5, 19)}:{random.randint(0, 59):02d}",
    ])

    draw.text(
        (x2 - 85, y1 + 12),
        clock,
        font=font(20, True),
        fill=colors["text"],
    )

    draw.text(
        (x2 - 85, y1 + 35),
        f"+{random.choice([0, 1, 2, 3, 5])}",
        font=font(10),
        fill=colors["muted"],
    )


# ============================================================
# CAPTURED PIECES
# ============================================================

def draw_captured(draw, x, y, colors):

    pieces = random.randint(0, 8)

    symbols = [
        "♟",
        "♞",
        "♝",
        "♜",
        "♛",
    ]

    text = ""

    for _ in range(pieces):
        text += random.choice(symbols)

    draw.text(
        (x, y),
        text,
        font=font(15),
        fill=colors["muted"],
    )

    if random.random() < 0.4:
        draw.text(
            (x + 120, y),
            f"+{random.randint(1, 8)}",
            font=font(12, True),
            fill=colors["accent"],
        )


# ============================================================
# MOVE LIST
# ============================================================

def draw_move_panel(draw, box, colors):

    x1, y1, x2, y2 = box

    rounded(
        draw,
        box,
        8,
        colors["panel"],
        colors["border"],
        1,
    )

    draw.text(
        (x1 + 15, y1 + 12),
        "MOVES",
        font=font(13, True),
        fill=colors["text"],
    )

    y = y1 + 42

    move_names = [
        "e4",
        "e5",
        "Nf3",
        "Nc6",
        "Bb5",
        "a6",
        "Ba4",
        "Nf6",
        "O-O",
        "Be7",
        "Re1",
        "b5",
        "Bb3",
        "d6",
        "c3",
        "O-O",
        "h3",
        "Nb8",
    ]

    move_number = 1

    for i in range(min(12, (y2 - y) // 28)):

        white = move_names[
            (i * 2) % len(move_names)
        ]

        black = move_names[
            (i * 2 + 1) % len(move_names)
        ]

        if i == random.randint(0, 8):

            draw.rectangle(
                [
                    x1 + 8,
                    y,
                    x2 - 8,
                    y + 24,
                ],
                fill=colors["accent"],
            )

        draw.text(
            (x1 + 12, y + 4),
            f"{move_number}.",
            font=font(11),
            fill=colors["muted"],
        )

        draw.text(
            (x1 + 45, y + 4),
            white,
            font=font(12),
            fill=colors["text"],
        )

        draw.text(
            (x1 + 90, y + 4),
            black,
            font=font(12),
            fill=colors["text"],
        )

        move_number += 1
        y += 28

    # Opening
    draw.text(
        (x1 + 12, y2 - 28),
        random.choice([
            "Sicilian Defense",
            "Ruy Lopez",
            "Queen's Gambit",
            "Italian Game",
            "French Defense",
            "King's Indian",
        ]),
        font=font(10),
        fill=colors["muted"],
    )


# ============================================================
# ENGINE PANEL
# ============================================================

def draw_engine_panel(draw, box, colors):

    x1, y1, x2, y2 = box

    rounded(
        draw,
        box,
        8,
        colors["panel"],
        colors["border"],
        1,
    )

    draw.text(
        (x1 + 12, y1 + 10),
        "ENGINE",
        font=font(12, True),
        fill=colors["text"],
    )

    draw.text(
        (x1 + 12, y1 + 35),
        random.choice([
            "Stockfish 18",
            "Stockfish",
            "Engine",
            "Cloud analysis",
        ]),
        font=font(11),
        fill=colors["muted"],
    )

    eval_value = random.choice([
        "+0.3",
        "+0.8",
        "-0.4",
        "+1.2",
        "0.0",
        "-1.1",
    ])

    draw.text(
        (x2 - 65, y1 + 30),
        eval_value,
        font=font(16, True),
        fill=colors["text"],
    )

    draw.text(
        (x1 + 12, y1 + 65),
        f"Depth {random.randint(18, 38)}",
        font=font(10),
        fill=colors["muted"],
    )

    # engine lines
    y = y1 + 90

    lines = [
        "Nf3 Nc6 Bb5 a6",
        "d4 exd4 Qxd4",
        "Re1 Be7 O-O",
        "Bb3 d6 Nbd2",
    ]

    for line in lines:

        draw.text(
            (x1 + 12, y),
            line,
            font=font(10),
            fill=colors["muted"],
        )

        y += 20


# ============================================================
# CHAT PANEL
# ============================================================

def draw_chat(draw, box, colors):

    x1, y1, x2, y2 = box

    rounded(
        draw,
        box,
        8,
        colors["panel"],
        colors["border"],
        1,
    )

    draw.text(
        (x1 + 12, y1 + 10),
        "CHAT",
        font=font(12, True),
        fill=colors["text"],
    )

    messages = [
        "Good game!",
        "Nice move",
        "Hello",
        "gg",
        "That was close",
        "Interesting opening",
        "Good luck!",
    ]

    y = y1 + 42

    for _ in range(5):

        name = random_username()[:12]

        draw.text(
            (x1 + 12, y),
            name,
            font=font(9, True),
            fill=colors["accent"],
        )

        draw.text(
            (x1 + 12, y + 15),
            random.choice(messages),
            font=font(10),
            fill=colors["text"],
        )

        y += 37

    rounded(
        draw,
        [x1 + 10, y2 - 38, x2 - 10, y2 - 10],
        6,
        colors["panel2"],
    )

    draw.text(
        (x1 + 20, y2 - 31),
        "Write a message...",
        font=font(10),
        fill=colors["muted"],
    )


# ============================================================
# CONTROLS
# ============================================================

def draw_controls(draw, x, y, w, colors):

    buttons = [
        "↶",
        "↷",
        "▶",
        "⏸",
        "FLIP",
        "BOARD",
        "SETTINGS",
        "FULL",
    ]

    bw = max(45, w // len(buttons) - 5)

    for i, text in enumerate(buttons):

        bx = x + i * (bw + 5)

        rounded(
            draw,
            [bx, y, bx + bw, y + 34],
            5,
            colors["panel"],
            colors["border"],
        )

        draw_text_center(
            draw,
            [bx, y, bx + bw, y + 34],
            text,
            font(9, True),
            colors["muted"],
        )


# ============================================================
# EXTRA CARDS
# ============================================================

def draw_info_card(draw, box, colors, title):

    x1, y1, x2, y2 = box

    rounded(
        draw,
        box,
        8,
        colors["panel"],
        colors["border"],
    )

    draw.text(
        (x1 + 12, y1 + 10),
        title,
        font=font(11, True),
        fill=colors["text"],
    )

    labels = [
        "Players",
        "Rated game",
        "Rapid",
        "Online",
        "Spectators",
        "Moves",
        "Opening",
    ]

    y = y1 + 35

    for label in labels[:max(1, (y2 - y) // 25)]:

        draw.text(
            (x1 + 12, y),
            label,
            font=font(9),
            fill=colors["muted"],
        )

        draw.text(
            (x2 - 90, y),
            random.choice([
                "Yes",
                "1,842",
                "10+0",
                "Online",
                "247",
                "32",
                "Sicilian",
            ]),
            font=font(9),
            fill=colors["text"],
        )

        y += 25


# ============================================================
# SIDE NAVIGATION
# ============================================================

def draw_side_nav(draw, W, H, browser_h, colors):

    if random.random() > 0.35:
        return 0

    width = random.randint(55, 90)

    y1 = browser_h

    draw.rectangle(
        [0, y1, width, H],
        fill=colors["panel"],
    )

    icons = [
        "♟",
        "♞",
        "♜",
        "♛",
        "⌂",
        "★",
        "☰",
        "⚙",
    ]

    y = y1 + 30

    for icon in icons:

        draw_text_center(
            draw,
            [0, y, width, y + 45],
            icon,
            font(18),
            colors["muted"],
        )

        y += 55

    return width


# ============================================================
# NOTIFICATIONS / ADS / BANNERS
# ============================================================

def draw_bottom_noise(draw, W, H, colors, board_box):

    candidates = []

    # bottom banner
    if random.random() < 0.45:

        h = random.randint(45, 75)

        box = [
            random.randint(10, max(10, W // 4)),
            H - h - random.randint(5, 30),
            random.randint(W // 2, W - 20),
            H - 10,
        ]

        candidates.append(box)

    # top notification
    if random.random() < 0.35:

        box = [
            random.randint(10, max(10, W // 3)),
            random.randint(100, 180),
            random.randint(W // 2, W - 20),
            random.randint(190, 260),
        ]

        candidates.append(box)

    for box in candidates:

        if rect_intersects(box, board_box, margin=15):
            continue

        rounded(
            draw,
            box,
            8,
            colors["panel"],
            colors["border"],
        )

        draw.text(
            (box[0] + 12, box[1] + 10),
            random.choice([
                "New challenge available",
                "Your opponent is online",
                "Game saved",
                "Upgrade to Premium",
                "Analysis complete",
                "Tournament starts soon",
            ]),
            font=font(11, True),
            fill=colors["text"],
        )


# ============================================================
# EXTRA RIGHT/LEFT CARDS
# ============================================================

def fill_side_area(
    draw,
    W,
    H,
    board_box,
    colors,
    side,
):
    bx1, by1, bx2, by2 = board_box

    if side == "right":

        x1 = bx2 + 20
        x2 = W - 20

    else:

        x1 = 20
        x2 = bx1 - 20

    if x2 - x1 < 170:
        return

    available_h = H - by1 - 20

    if available_h < 200:
        return

    layouts = [
        "moves",
        "engine",
        "chat",
        "info",
        "mixed",
    ]

    layout = random.choice(layouts)

    y = by1

    if layout in ["moves", "mixed"]:

        h = min(
            random.randint(300, 450),
            H - y - 20,
        )

        box = [x1, y, x2, y + h]

        if not rect_intersects(box, board_box, 5):
            draw_move_panel(
                draw,
                box,
                colors,
            )

        y += h + 12

    if layout in ["engine", "mixed"]:

        h = min(
            random.randint(180, 300),
            H - y - 20,
        )

        if h > 120:

            box = [x1, y, x2, y + h]

            if not rect_intersects(box, board_box, 5):
                draw_engine_panel(
                    draw,
                    box,
                    colors,
                )

            y += h + 12

    if layout == "chat":

        h = min(
            random.randint(280, 430),
            H - y - 20,
        )

        if h > 150:

            box = [x1, y, x2, y + h]

            if not rect_intersects(box, board_box, 5):
                draw_chat(
                    draw,
                    box,
                    colors,
                )

            y += h + 12

    if layout == "info":

        h = min(
            random.randint(220, 350),
            H - y - 20,
        )

        if h > 150:

            box = [x1, y, x2, y + h]

            if not rect_intersects(box, board_box, 5):
                draw_info_card(
                    draw,
                    box,
                    colors,
                    random.choice([
                        "GAME INFO",
                        "PLAYERS",
                        "OPENING",
                        "TOURNAMENT",
                        "GAME DETAILS",
                    ]),
                )


# ============================================================
# PAGE TYPE ELEMENTS
# ============================================================

def draw_page_specific(draw, W, H, board_box, colors):

    bx1, by1, bx2, by2 = board_box

    page_type = random.choice([
        "game",
        "game",
        "analysis",
        "analysis",
        "replay",
        "tournament",
        "puzzle",
        "watch",
        "community",
    ])

    # Top title outside board
    title_y = max(100, by1 - random.randint(45, 90))

    if title_y + 30 < by1:

        titles = {
            "game": [
                "Rated Rapid Game",
                "Casual Chess",
                "Blitz Game",
                "Live Game",
            ],

            "analysis": [
                "Game Analysis",
                "Computer Analysis",
                "Analyze Position",
            ],

            "replay": [
                "Game Replay",
                "Historical Game",
                "Replay",
            ],

            "tournament": [
                "Tournament",
                "Live Tournament",
                "Arena",
            ],

            "puzzle": [
                "Daily Puzzle",
                "Tactical Puzzle",
                "Puzzle Rush",
            ],

            "watch": [
                "Watch Live",
                "Featured Game",
                "Top Games",
            ],

            "community": [
                "Community",
                "Chess Club",
                "Friends",
            ],
        }

        draw.text(
            (bx1, title_y),
            random.choice(titles[page_type]),
            font=font(18, True),
            fill=colors["text"],
        )

    # Bottom controls
    controls_y = by2 + 15

    if controls_y + 40 < H:

        draw_controls(
            draw,
            bx1,
            controls_y,
            bx2 - bx1,
            colors,
        )

    # Additional information under board
    info_y = controls_y + 50

    if info_y + 100 < H:

        box = [
            bx1,
            info_y,
            bx2,
            info_y + 90,
        ]

        if not rect_intersects(box, board_box):

            draw_info_card(
                draw,
                box,
                colors,
                random.choice([
                    "GAME INFORMATION",
                    "ANALYSIS",
                    "OPENING EXPLORER",
                    "PLAYER INFORMATION",
                ]),
            )

    return page_type


# ============================================================
# MAIN IMAGE
# ============================================================

def generate_image(index):

    W, H = random.choice(SCREEN_SIZES)

    website_theme = random.choice(
        list(WEBSITE_THEMES.keys())
    )

    colors = WEBSITE_THEMES[website_theme]

    # background
    img = Image.new(
        "RGBA",
        (W, H),
        colors["bg"] + (255,),
    )

    draw = ImageDraw.Draw(img)

    # subtle background panels
    if random.random() < 0.6:

        for _ in range(random.randint(2, 6)):

            px = random.randint(0, W)
            py = random.randint(0, H)

            pw = random.randint(150, 500)
            ph = random.randint(80, 300)

            draw.rectangle(
                [
                    px,
                    py,
                    min(W, px + pw),
                    min(H, py + ph),
                ],
                fill=colors["panel2"],
            )

    # browser
    browser_h = draw_browser(
        draw,
        W,
        H,
        colors,
    )

    # website header
    header_bottom = draw_website_header(
        draw,
        browser_h,
        W,
        colors,
    )

    # side nav
    side_nav_width = draw_side_nav(
        draw,
        W,
        H,
        header_bottom,
        colors,
    )

    # ========================================================
    # SAMPLE KIND: normal / board partly off-canvas / no board
    # ========================================================

    sample_kind = random.choices(
        list(SAMPLE_KIND_WEIGHTS.keys()),
        weights=list(SAMPLE_KIND_WEIGHTS.values()),
    )[0]

    has_board = sample_kind != "no_board"

    # board position
    bx, by, board_size = choose_board_position(
        W,
        H,
        browser_h,
        header_bottom,
    )

    # shift if side navigation exists
    if side_nav_width:

        bx = max(
            bx,
            side_nav_width + 20,
        )

        if bx + board_size > W - 20:

            bx = W - board_size - 20

    if sample_kind == "partial_crop":
        # Force actual off-canvas overflow on one edge - overriding the
        # position outright (a relative shift from wherever the board
        # already was is NOT guaranteed to cross the canvas edge if
        # there was a lot of margin to begin with). PIL silently clips
        # drawing calls outside the canvas, so this is safe to draw;
        # the LABEL is clipped to the visible region below.
        frac = random.uniform(0.25, 0.55)
        edge = random.choice(["left", "right", "top", "bottom"])
        if edge == "left":
            bx = -int(board_size * frac)
        elif edge == "right":
            bx = W - board_size + int(board_size * frac)
        elif edge == "top":
            by = -int(board_size * frac)
        else:
            by = H - board_size + int(board_size * frac)

    board_box = (
        bx,
        by,
        bx + board_size,
        by + board_size,
    )

    # ========================================================
    # BOARD
    # ========================================================

    board = random_position()

    piece_style = random.choice(
        COMPLETE_STYLES
    )

    board_theme = random.choice(
        [x[0] for x in BOARD_THEMES]
    )

    coordinates = random.random() < 0.75

    flipped = random.random() < 0.15

    border = "none"

    if has_board:
        border = draw_board(
            img,
            draw,
            board,
            bx,
            by,
            board_size,
            piece_style,
            board_theme,
            coordinates,
            flipped,
        )

    # ========================================================
    # PLAYER CARDS
    # ========================================================

    card_h = 60

    if has_board and by - card_h - 10 > header_bottom:

        draw_player_card(
            draw,
            [
                bx,
                by - card_h - 8,
                bx + board_size,
                by - 8,
            ],
            colors,
            top=True,
        )

    if has_board and by + board_size + 8 + card_h < H:

        draw_player_card(
            draw,
            [
                bx,
                by + board_size + 8,
                bx + board_size,
                by + board_size + 8 + card_h,
            ],
            colors,
            top=False,
        )

    if has_board:
        # captured pieces near players
        draw_captured(
            draw,
            bx + 10,
            by - 28,
            colors,
        )

        draw_captured(
            draw,
            bx + 10,
            by + board_size + card_h + 18,
            colors,
        )

    # ========================================================
    # SIDE CONTENT
    # ========================================================

    if random.random() < 0.85:

        side = random.choice([
            "left",
            "right",
            "right",
        ])

        fill_side_area(
            draw,
            W,
            H,
            board_box,
            colors,
            side,
        )

    # Sometimes put content on opposite side too
    if random.random() < 0.25:

        side = "left"

        fill_side_area(
            draw,
            W,
            H,
            board_box,
            colors,
            side,
        )

    # ========================================================
    # PAGE-SPECIFIC CONTENT
    # ========================================================

    page_type = draw_page_specific(
        draw,
        W,
        H,
        board_box,
        colors,
    )

    # ========================================================
    # BANNERS / NOTIFICATIONS
    # ========================================================

    draw_bottom_noise(
        draw,
        W,
        H,
        colors,
        board_box,
    )

    # ========================================================
    # SCROLLBAR
    # ========================================================

    if random.random() < 0.65:

        scrollbar_x = W - 8

        draw.rectangle(
            [
                scrollbar_x,
                browser_h,
                W,
                H,
            ],
            fill=colors["panel2"],
        )

        thumb_h = random.randint(
            80,
            max(100, H // 3),
        )

        thumb_y = random.randint(
            browser_h,
            max(browser_h, H - thumb_h),
        )

        rounded(
            draw,
            [
                scrollbar_x,
                thumb_y,
                W,
                thumb_y + thumb_h,
            ],
            3,
            colors["muted"],
        )

    # ========================================================
    # LABEL BOX: clip to what's actually visible in the canvas
    # ========================================================

    if has_board:
        vis_x1 = clamp(board_box[0], 0, W)
        vis_y1 = clamp(board_box[1], 0, H)
        vis_x2 = clamp(board_box[2], 0, W)
        vis_y2 = clamp(board_box[3], 0, H)
    else:
        vis_x1 = vis_y1 = vis_x2 = vis_y2 = 0

    # ========================================================
    # SAVE
    # ========================================================

    IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if SAVE_JPEG:

        filename = f"{index:06d}.jpg"

        path = IMAGE_DIR / filename

        final_img = photometric_augment(img.convert("RGB"))

        final_img.save(
            path,
            quality=JPEG_QUALITY,
            optimize=True,
        )

    else:

        filename = f"{index:06d}.png"

        path = IMAGE_DIR / filename

        img.save(
            path,
            optimize=True,
        )

    # ========================================================
    # LABEL
    # ========================================================

    return {
        "filename": filename,

        "has_board": has_board,
        "sample_kind": sample_kind,

        "x1": round(vis_x1, 1),
        "y1": round(vis_y1, 1),
        "x2": round(vis_x2, 1),
        "y2": round(vis_y2, 1),

        "screen_width": W,
        "screen_height": H,

        "page_type": page_type,

        "website_theme": website_theme,
        "board_theme": board_theme if has_board else "",

        "piece_style": piece_style if has_board else "",

        "coordinates": coordinates if has_board else "",
        "border": border,
        "flipped": flipped if has_board else "",

        "fen": board.fen() if has_board else "",
    }


# ============================================================
# DATASET
# ============================================================

def generate_dataset():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    print()
    print("========================================")
    print("   CHESS BOARD DETECTOR DATASET V6")
    print("========================================")
    print()

    print(
        "Complete piece styles:",
        COMPLETE_STYLES,
    )

    print(
        "Images:",
        NUM_IMAGES,
    )

    print()

    for i in range(NUM_IMAGES):

        try:

            row = generate_image(i)

            rows.append(row)

        except Exception as e:

            print(
                f"[ERROR] image {i}: {e}"
            )

            continue

        if (i + 1) % 10 == 0:

            print(
                f"Generated {i + 1}/{NUM_IMAGES}"
            )

    # CSV
    fields = [
        "filename",
        "has_board",
        "sample_kind",
        "x1",
        "y1",
        "x2",
        "y2",
        "screen_width",
        "screen_height",
        "page_type",
        "website_theme",
        "board_theme",
        "piece_style",
        "coordinates",
        "border",
        "flipped",
        "fen",
    ]

    with open(
        LABEL_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(rows)

    print()
    print("========================================")
    print("DONE")
    print("========================================")
    print()
    print(
        "Images:",
        IMAGE_DIR.resolve(),
    )

    print(
        "Labels:",
        LABEL_FILE.resolve(),
    )

    print(
        "Generated:",
        len(rows),
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    generate_dataset()