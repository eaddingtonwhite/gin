#!/usr/bin/env python3
"""Entry point for the vendored runner: python3 .mo-eval/runner/main.py suite …"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner.main import main  # noqa: E402

raise SystemExit(main())
