"""Make scripts/ importable so tests can `import kg_loader` and `load_data`."""

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))
