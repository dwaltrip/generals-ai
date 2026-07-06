"""Typed access to a checkpoint file's top-level layout.

`RawCheckpoint` wraps the dict that `torch.load` returns and derives layout
facts from its top-level keys. Nothing here does I/O or builds a model
(`storage.checkpoint` owns both). The stored config block passes through
unopened — `bc.config` owns its contents.
"""

from __future__ import annotations

from dataclasses import dataclass

from training.bc.config import StoredConfigBlock


@dataclass(frozen=True)
class RawCheckpoint:
    """The raw contents of one checkpoint file."""

    data: dict[str, object]

    @property
    def versioned(self) -> bool:
        """True if the checkpoint carries a stored config block (v1+)."""
        return "config" in self.data

    @property
    def combined(self) -> bool:
        """True if the checkpoint uses the combined layout that
        `TrainingState.save` writes. A bare `state_dict` is the only other layout.
        """
        return "model" in self.data

    @property
    def arch_bearing(self) -> bool:
        """True if the checkpoint records its own architecture.

        A versioned checkpoint records it inside the config block.
        A v0 combined checkpoint records it under a top-level `arch` key (if at all).
        """
        return self.combined and (self.versioned or "arch" in self.data)

    @property
    def config_block(self) -> StoredConfigBlock:
        assert self.versioned, "no config block: not a versioned checkpoint"
        block = self.data["config"]
        if not isinstance(block, dict):
            raise ValueError(
                f"malformed checkpoint: config block is {type(block).__name__}, "
                "expected a dict"
            )
        return block

    @property
    def in_ch(self) -> int:
        """The input-channel checksum that a versioned checkpoint records."""
        value = self.data["in_ch"]
        if not isinstance(value, int):
            raise ValueError(
                f"malformed checkpoint: in_ch is {type(value).__name__}, expected int"
            )
        return value

    @property
    def model_state(self) -> dict[str, object]:
        """The model's `state_dict`, regardless of layout."""
        if not self.combined:
            return self.data
        state = self.data["model"]
        if not isinstance(state, dict):
            raise ValueError(
                f"malformed checkpoint: model entry is {type(state).__name__}, "
                "expected a state_dict"
            )
        return state
