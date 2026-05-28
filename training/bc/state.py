"""Persistent training state — the bits that survive a resume.

`TrainingState` bundles the model, optimizer, GradScaler, and the
last-completed epoch, and owns their serialization to/from a combined
checkpoint dict. The runner mutates it in place (`state.epoch += 1` after
each completed epoch) and saves it at each epoch boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from bc.checkpoint import ckpt_name, load_bc_model
from bc.model import BCModel
from bc.train_config import TrainConfig
from shared.device import resolve_precision


def _build_optim(model: BCModel, config: TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
    )


def _build_scaler(config: TrainConfig, device: torch.device) -> torch.amp.GradScaler:
    # fp16 is the only AMP mode that needs loss scaling; the scaler is a
    # near-no-op when disabled, so the fp32 path stays unchanged.
    amp_enabled = resolve_precision(config.precision, device) == "fp16"
    return torch.amp.GradScaler(device.type, enabled=amp_enabled)


@dataclass
class TrainingState:
    """Model + optimizer + scaler + last-completed epoch.

    Unfrozen: PyTorch modules are inherently mutable, and `epoch` is
    advanced in place by the runner. `epoch` is the last *successfully
    completed* epoch — resume continues at `epoch + 1`, and a mid-epoch
    crash leaves it one behind (which is what resume expects).
    """

    model: BCModel
    optim: torch.optim.Optimizer
    scaler: torch.amp.GradScaler
    epoch: int

    @classmethod
    def fresh(cls, config: TrainConfig, device: torch.device) -> TrainingState:
        """Build a brand-new state: fresh model/optim/scaler, epoch 0."""
        model = BCModel(value_head_variant=config.value_head_variant).to(device)
        return cls(
            model=model,
            optim=_build_optim(model, config),
            scaler=_build_scaler(config, device),
            epoch=0,
        )

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, config: TrainConfig, device: torch.device
    ) -> TrainingState:
        """Restore a state from a checkpoint, for resuming a run.

        The model loads via `load_bc_model` (the single format-detection
        point). Optim/scaler state is restored only when present in the
        combined format — a legacy bare-state_dict checkpoint yields fresh
        optim/scaler shells (a cold optimizer restart) and `epoch` falls
        back to 0.

        Re-reads the file once for the optim/scaler/epoch payload after
        `load_bc_model` reads it for the weights — a one-time cost paid
        only at resume startup.
        """
        model = load_bc_model(path, device, config.value_head_variant)
        optim = _build_optim(model, config)
        scaler = _build_scaler(config, device)
        epoch = 0

        obj = torch.load(path, map_location=device, weights_only=True)
        if isinstance(obj, dict) and "model" in obj:  # combined format
            if "optim" in obj:
                optim.load_state_dict(obj["optim"])
            if "scaler" in obj:
                scaler.load_state_dict(obj["scaler"])
            epoch = int(obj.get("epoch", 0))

        return cls(model=model, optim=optim, scaler=scaler, epoch=epoch)

    def save(self, ckpt_dir: Path) -> Path:
        """Write the combined checkpoint for the current epoch; return its path.

        One `torch.save` — the load either returns the whole dict or raises;
        partial writes aren't handled (operator cleans up).
        """
        path = ckpt_dir / ckpt_name(self.epoch)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optim": self.optim.state_dict(),
                "scaler": self.scaler.state_dict(),
                "epoch": self.epoch,
            },
            path,
        )
        return path
