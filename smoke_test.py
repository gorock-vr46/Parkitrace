"""Basic offline checks for the ParkiTrace project."""
import os
import py_compile
from pathlib import Path

ROOT = Path(__file__).parent
for filename in ["app.py", "train.py", "parkitrace_model.py", "make_demo_data.py"]:
    py_compile.compile(str(ROOT / filename), doraise=True)
for required in ["templates/index.html", "requirements.txt", "README.md"]:
    assert (ROOT / required).exists(), required
print("ParkiTrace smoke test passed: Python files compile and required files exist.")
