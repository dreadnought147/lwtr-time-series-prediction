"""
Build the title-slide image: a newsroom composition — the presenter in a
"LIVE" inset on the left, and a big 'financial forecast' board on the right
covered in rand/dollar signs, a trend line, and trading bots.

Source assets (both free to use, downloaded from Wikimedia Commons):
  weatherman.png  — "File:Weatherman.png"   — Public domain
  robot.png       — "File:Robot-icon.png"   — CC0 (public-domain dedication)

Run:
    python takeoff/make_title_image.py

Output:
    takeoff/assets/title_financial_forecast.png
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"

W, H = 1600, 760

# Palette matched to the deck theme in make_pptx.html
GOLD = (251, 191, 36)
GREEN = (52, 211, 153)
WHITE = (255, 255, 255)
SLATE = (30, 41, 59)
DARK = (15, 23, 42)
RED = (220, 38, 38)
GRID = (71, 85, 105)


def load_font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    """Grab a Windows system font; fall back to PIL's default if missing."""
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def rounded_mask(size, radius):
    """Alpha mask with rounded corners — used to clip the presenter inset."""
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, size[0] - 1, size[1] - 1],
                                        radius=radius, fill=255)
    return m


def main():
    canvas = Image.new("RGBA", (W, H), DARK + (255,))
    draw = ImageDraw.Draw(canvas)

    # ===================================================================
    # LEFT — the presenter, clipped into a rounded "LIVE feed" inset
    # ===================================================================
    INSET = (40, 70, 470, 640)                      # x0, y0, x1, y1
    iw, ih = INSET[2] - INSET[0], INSET[3] - INSET[1]

    wm = Image.open(ASSETS / "weatherman.png").convert("RGBA")
    # upscale then centre-crop to the inset's aspect ratio
    scale = max(iw / wm.width, ih / wm.height) * 1.25
    wm = wm.resize((int(wm.width * scale), int(wm.height * scale)), Image.LANCZOS)
    left = (wm.width - iw) // 2
    top = int((wm.height - ih) * 0.42)               # bias upward: keep the face
    wm = wm.crop((left, top, left + iw, top + ih))

    canvas.paste(wm, (INSET[0], INSET[1]), rounded_mask((iw, ih), 20))
    draw.rounded_rectangle(INSET, radius=20, outline=GOLD + (255,), width=5)

    # LIVE badge over the inset
    draw.rounded_rectangle([INSET[0] + 18, INSET[1] + 18, INSET[0] + 150, INSET[1] + 74],
                           radius=9, fill=RED + (255,))
    draw.text((INSET[0] + 44, INSET[1] + 28), "LIVE", font=load_font(36), fill=WHITE + (255,))

    # caption under the inset
    draw.text((INSET[0] + 6, INSET[3] + 18), "YOUR AGENT, REPORTING LIVE",
              font=load_font(28), fill=GOLD + (255,))

    # ===================================================================
    # RIGHT — the financial forecast board
    # ===================================================================
    BOARD = (520, 70, W - 45, 640)
    bx0, by0, bx1, by1 = BOARD
    draw.rounded_rectangle(BOARD, radius=22, fill=SLATE + (255,),
                           outline=GOLD + (255,), width=5)

    # faint chart grid
    for i in range(1, 12):
        x = bx0 + (bx1 - bx0) * i / 12
        draw.line([(x, by0 + 12), (x, by1 - 12)], fill=GRID + (110,), width=1)
    for i in range(1, 8):
        y = by0 + (by1 - by0) * i / 8
        draw.line([(bx0 + 12, y), (bx1 - 12, y)], fill=GRID + (110,), width=1)

    # currency symbols — the "weather map" is now rand and dollars
    symbols = [
        ("R", 575, 100, 78, GOLD),
        ("$", 690, 132, 58, GREEN),
        ("R", 800, 96,  70, GREEN),
        ("$", 915, 136, 54, GOLD),
        ("R", 1020, 100, 74, GOLD),
        ("$", 1140, 136, 56, GREEN),
        ("R", 1255, 96,  72, GREEN),
        ("$", 1380, 134, 56, GOLD),
        ("R", 1470, 100, 68, GOLD),
        ("R", 600, 330, 66, GREEN),
        ("$", 735, 362, 52, GOLD),
        ("R", 1180, 330, 70, GOLD),
        ("$", 1430, 356, 54, GREEN),
    ]
    for ch, x, y, size, colour in symbols:
        draw.text((x, y), ch, font=load_font(size), fill=colour + (255,))

    # the forecast trend line
    trend = [(575, 540), (720, 482), (865, 500), (1010, 410),
             (1155, 432), (1300, 330), (1450, 288)]
    draw.line(trend, fill=GREEN + (255,), width=8, joint="curve")
    for px, py in trend:
        draw.ellipse([px - 10, py - 10, px + 10, py + 10], fill=WHITE + (255,))
    ax, ay = trend[-1]
    draw.polygon([(ax + 44, ay - 12), (ax - 4, ay - 36), (ax - 4, ay + 12)],
                 fill=GREEN + (255,))

    # trading bots on the board
    robot = Image.open(ASSETS / "robot.png").convert("RGBA")
    for (rx, ry, rs) in [(880, 210, 112), (1290, 180, 96), (660, 215, 84)]:
        canvas.alpha_composite(robot.resize((rs, rs), Image.LANCZOS), (rx, ry))

    # caption strip along the bottom of the board
    draw.rounded_rectangle([bx0 + 20, by1 - 84, bx1 - 20, by1 - 20],
                           radius=10, fill=(15, 23, 42, 240))
    draw.text((bx0 + 42, by1 - 72), "TODAY'S FORECAST:  MEMORY, REGIMES & POSITIONS",
              font=load_font(32), fill=GOLD + (255,))

    # ===================================================================
    # attribution footer (keeps the licences honest)
    # ===================================================================
    draw.text((40, H - 46),
              "Imagery: \"Weatherman\" (public domain) and \"Robot-icon\" (CC0), Wikimedia Commons.",
              font=load_font(20, bold=False), fill=(148, 163, 184, 255))

    out = ASSETS / "title_financial_forecast.png"
    canvas.convert("RGB").save(out, "PNG", optimize=True)
    print(f"wrote {out}  ({canvas.size[0]}x{canvas.size[1]})")


if __name__ == "__main__":
    main()
