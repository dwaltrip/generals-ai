from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

UA_CONTACT_EMAIL_KEY = "UA_CONTACT_EMAIL"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DOTENV_PATH = _PROJECT_ROOT / ".env"


@dataclass(frozen=True)
class Config:
    UA_CONTACT_EMAIL: str


def load_env(path: Path = _DOTENV_PATH) -> Config:
    """Parse a `.env` file and return a validated Config. Raises if the file
    is missing required keys. Missing files are treated as empty (validation
    will then surface the missing key with a clear message)."""
    values = dotenv_values(path)
    contact_email = (values.get(UA_CONTACT_EMAIL_KEY) or "").strip()
    if not contact_email:
        raise RuntimeError(
            f"{UA_CONTACT_EMAIL_KEY} is not set in {path}.\n"
            f"Add a line like:\n"
            f"  {UA_CONTACT_EMAIL_KEY}=you@example.com"
        )
    return Config(UA_CONTACT_EMAIL=contact_email)


config = load_env()
