#!/usr/bin/env python3
"""Run from a checkout or installed skill without pip or environment changes."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from usual.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
