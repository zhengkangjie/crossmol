from pathlib import Path
import importlib

from .unimol import UniMolTask

# automatically import any Python files in the criterions/ directory
for file in sorted(Path(__file__).parent.glob("*.py")):
    if not file.name.startswith("_"):
        importlib.import_module("unimol.tasks." + file.name[:-3])
