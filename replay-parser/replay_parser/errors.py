class ArmyOverflowError(Exception):
    """A single-tile army stack exceeded int16 range during simulation.

    Per docs/replay-parser-design.md, such replays are skipped rather than
    supported — stacks >32k don't occur in competitive FFA play. The parser
    raises this; callers log and skip.
    """
