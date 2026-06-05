from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

DB_PATH = PROJECT_ROOT / "replay-collector" / "data" / "generals.sqlite"
