#!/usr/bin/env python3
"""Backward-compatible wrapper for `themis analyze` (use the entry point)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from legalrag.cli.analyze import main

if __name__ == "__main__":
    raise SystemExit(main())
