import sys


def doc_summary(doc: str | None) -> str:
    """First line of a module docstring; warns to stderr if missing."""
    if not doc:
        print("warning: module docstring missing", file=sys.stderr)
        return ""
    return doc.splitlines()[0]
