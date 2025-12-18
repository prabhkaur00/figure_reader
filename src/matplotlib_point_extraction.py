from __future__ import annotations

"""
Placeholder for a Matplotlib-based point extraction pipeline.

This module is intended to house deterministic digitization logic powered by
Matplotlib (e.g., interactive point selection, image sampling, etc.). The
Figure Reader CLI currently exposes a stub tool (`MatplotlibImageExtractor`)
that references this module conceptually, but the actual implementation is
not yet provided.

TODO:
    - Implement helpers that accept an image path and return digitized (x, y)
      pairs using Matplotlib-based techniques.
    - Provide functions mirroring the OpenCV interface so the agent can call
      them interchangeably.
"""


def digitize_with_matplotlib(*args, **kwargs):
    raise NotImplementedError(
        "Matplotlib point extraction not implemented. "
        "Fill in src/matplotlib_point_extraction.py with real logic."
    )
