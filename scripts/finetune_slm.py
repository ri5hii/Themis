#!/usr/bin/env python3
"""LoRA fine-tune the plain-language SLM (thin wrapper for legalrag.train)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from legalrag.cli.train import run_slm_cli

if __name__ == "__main__":
    raise SystemExit(run_slm_cli(sys.argv[1:]))