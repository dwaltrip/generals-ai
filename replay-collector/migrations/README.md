# Migrations

Run-once schema-change scripts for `data/generals.sqlite`.

Conventions:

- Numbered sequentially: `NNN_short_description.py`.
- Idempotent: each script checks current schema state and no-ops if already applied. Safe to re-run.
- Standalone: invoked directly via `uv run python migrations/NNN_*.py` from the `replay-collector/` directory. Not imported by application code.
- Refuses to run if a precondition isn't met (e.g. depends on prior migration), with a clear error.
