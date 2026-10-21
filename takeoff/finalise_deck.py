"""
Post-process make_pptx.html so it works when opened as a file:// page.

Two jobs, in this order:
  1. Strip every em-dash. A colon replaces it when the dash separates a
     short label from its description, a comma otherwise. Done first, while
     the file is still small.
  2. Inline every referenced PNG as a base64 data URI. Browsers block XHR
     fetches of local files from a file:// page, so PptxGenJS cannot load
     `path:` references. Switching to `data:` removes the fetch entirely.

Run:
    python takeoff/finalise_deck.py

Edits make_pptx.html in place. Safe to re-run: already-inlined images are
skipped, and a file with no em-dashes is left alone.
"""

import base64
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
HTML = HERE / "make_pptx.html"
EM_DASH = "—"


def strip_em_dashes(src: str) -> tuple[str, int]:
    """Remove em-dashes line by line.

    A dash that follows a short run of words with no sentence punctuation is
    acting as a label separator, so a colon reads best there. Anywhere else
    it is parenthetical, so a comma reads best.
    """
    removed = 0
    out_lines = []

    for line in src.split("\n"):
        if EM_DASH not in line:
            out_lines.append(line)
            continue

        # Walk the dashes left to right, deciding each one from its own lead-in.
        parts = line.split(EM_DASH)
        rebuilt = parts[0]
        for piece in parts[1:]:
            removed += 1
            # Text since the last quote mark is what the dash is attached to.
            lead = re.split(r"['\"]", rebuilt)[-1].strip()
            label_like = len(lead) <= 38 and not re.search(r"[.,;:?!]", lead)
            sep = ": " if label_like else ", "
            rebuilt = rebuilt.rstrip() + sep + piece.lstrip()
        out_lines.append(rebuilt)

    return "\n".join(out_lines), removed


def inline_images(src: str) -> tuple[str, int]:
    """Replace `path: '<file>.png'` with an inline base64 data URI."""
    pattern = re.compile(r"path:\s*'([^']+\.png)'")
    count = 0

    def swap(m):
        nonlocal count
        rel = m.group(1)
        img = (HERE / rel).resolve()          # paths are relative to takeoff/
        if not img.exists():
            print(f"  MISSING, left as-is: {rel}")
            return m.group(0)
        b64 = base64.b64encode(img.read_bytes()).decode("ascii")
        count += 1
        print(f"  inlined {rel}  ({len(b64) // 1024} KB base64)")
        return f"data: 'data:image/png;base64,{b64}'"

    return pattern.sub(swap, src), count


def main():
    src = HTML.read_text(encoding="utf-8")

    print("Stripping em-dashes:")
    src, n_dash = strip_em_dashes(src)
    print(f"  removed {n_dash}")

    print("\nInlining images:")
    src, n_img = inline_images(src)

    HTML.write_text(src, encoding="utf-8")
    size_mb = HTML.stat().st_size / (1024 * 1024)
    print(f"\nwrote {HTML.name}  ({size_mb:.2f} MB, {n_img} images inlined)")


if __name__ == "__main__":
    main()
