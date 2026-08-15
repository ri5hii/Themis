# Extract: clause-level classification into the lease clause taxonomy.
#
# Fast-lane regex triggers (fast_lane.py) are authoritative when they fire;
# the trained classifier (classifier.py) is a fallback for `unknown` sections.
# analyzeSections (analyze.py) drives the hybrid over a full document.
from __future__ import annotations

from .analyze import analyzeSections

__all__ = ["analyzeSections"]
