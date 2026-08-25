"""
Walk-forward split generator.

Walk-forward = train on a past chunk, test on the chunk right after, slide
forward, repeat. Each window's test slice is fresh data the model never saw
during training for that window.

We do this because financial data shifts shape over time (2001 regime ≠ 2020
regime). A single train/test split hides that; walk-forward exposes it.

Sanity band (from published quant-finance practice, 2026):
    train:test ratio    2:1 to 5:1
    min windows         6-8 (12-20 preferred)
    step size           = test_size  (so test slices don't overlap)

Choices outside this band need a written reason.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class Window:
    """One walk-forward slice: a contiguous train chunk followed by a contiguous test chunk.

    # think of this as: [ train ][ test ] pinned to some starting position.
    Indices are into the flat (obs, returns) arrays.
    """
    index: int         # 0, 1, 2, ... which window this is
    train_start: int
    train_end: int     # exclusive
    test_start: int
    test_end: int      # exclusive

    def sizes(self) -> tuple[int, int]:
        # returns (train_length, test_length) in days
        return self.train_end - self.train_start, self.test_end - self.test_start


def walk_forward_windows(
    n_samples: int,
    train_size: int,
    test_size: int,
    step: int | None = None,
    max_windows: int | None = None,
) -> Iterator[Window]:
    """Yield walk-forward windows in chronological order.

    Args:
        n_samples: total length of the time series (e.g. 9188 for VIX+S&P).
        train_size: how many days go in each train chunk.
        test_size: how many days go in each test chunk.
        step: how far to slide between windows.
              Default = test_size (non-overlapping test slices).
        max_windows: optional cap on how many windows to yield. Useful for debug.
    """
    # by default, slide forward by exactly the test length so tests don't overlap
    if step is None:
        step = test_size

    assert train_size > 0 and test_size > 0, "sizes must be positive"
    assert train_size + test_size <= n_samples, (
        f"train_size ({train_size}) + test_size ({test_size}) "
        f"exceeds n_samples ({n_samples})"
    )

    i = 0
    n_yielded = 0
    while True:
        # a window is [train chunk][test chunk] starting at position i
        train_start = i
        train_end = i + train_size
        test_start = train_end
        test_end = test_start + test_size

        # stop once the window would run off the end of the data
        if test_end > n_samples:
            break

        yield Window(
            index=n_yielded,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )
        n_yielded += 1
        if max_windows is not None and n_yielded >= max_windows:
            break
        i += step


if __name__ == "__main__":
    # sanity check: 5-year train, 1-year test, non-overlapping, over ~24 years
    N = 9188
    TRAIN = 252 * 5   # ~5 trading years 5:1
    TEST = 252        # ~1 trading year

    print(f"Data length: {N} trading days (~{N // 252} years)")
    print(f"Train window: {TRAIN} days | Test window: {TEST} days\n")
    for w in walk_forward_windows(n_samples=N, train_size=TRAIN, test_size=TEST):
        tr, te = w.sizes()
        print(
            f"  W{w.index:2d}: train [{w.train_start:5d}:{w.train_end:5d}] ({tr}d)  "
            f"test [{w.test_start:5d}:{w.test_end:5d}] ({te}d)"
        )