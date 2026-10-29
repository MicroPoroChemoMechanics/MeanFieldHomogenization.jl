"""The Julia project of the tests, unless the environment names another."""

import os
from pathlib import Path

os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", str(Path(__file__).parent / "julia"))
