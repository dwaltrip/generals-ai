from pathlib import Path


DATA_ROOT = Path(__file__).resolve().parents[2] / "tests" / "goldens"
FIXTURES_DIR = DATA_ROOT / "fixtures"
REFERENCES_DIR = DATA_ROOT / "references"
