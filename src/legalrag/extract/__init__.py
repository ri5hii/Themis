# Extract: clause-level classification into the lease clause taxonomy.
#
# Fast-lane regex triggers (fast_lane.py) are authoritative when they fire;
# the trained classifier (classifier.py) is a fallback for `unknown` sections.
from __future__ import annotations
