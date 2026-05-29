"""The domain training state: model, optimizer, GradScaler, epoch.

`TrainingState` bundles the pieces the training loop mutates and owns their
serialization to/from a combined checkpoint dict. The persistent four
(model + optim + scaler + epoch) round-trip through `save()` /
`from_checkpoint`; the runner mutates the state in place (`state.epoch += 1`
after each completed epoch) and saves it at each epoch boundary.

A legacy-checkpoint resume also attaches a transient `WarmupSchedule` (the
`warmup` field) — it drives the optimizer, so it lives with the optimizer, but
it is deliberately not part of the checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from shared.device import resolve_precision
import torch

from bc.checkpoint import ckpt_name, is_combined_checkpoint, load_bc_model
from bc.model import BCModel
from bc.resume_warmup import WarmupSchedule
from bc.train_config import TrainConfig


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
    # Transient: a legacy-checkpoint resume attaches a WarmupSchedule here to
    # ramp the LR while AdamW's variance estimate re-warms. NOT written by
    # `save()` (which hand-picks the four persistent keys) — the resume path
    # rebuilds it per-process. None on fresh runs and combined-format resumes.
    warmup: WarmupSchedule | None = None

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
        cls,
        path: str | Path,
        config: TrainConfig,
        device: torch.device,
        fallback_epoch: int = 0,
    ) -> TrainingState:
        """Restore a state from a checkpoint, for resuming a run.

        The model loads via `load_bc_model`. Optim/scaler state is restored
        only when present in the combined format — a legacy bare-state_dict
        checkpoint yields fresh optim/scaler shells (a cold optimizer restart),
        which is why legacy resume pairs with `--legacy-lr-warmup-batches`.

        `epoch` comes from the checkpoint in the combined format. A legacy
        checkpoint has no epoch of its own, so it falls back to `fallback_epoch`
        — the resume path passes the parent epoch parsed from the filename
        (`epoch_NNN.pt`), so numbering stays monotonic across the resume.

        Re-reads the file once for the optim/scaler/epoch payload after
        `load_bc_model` reads it for the weights — a one-time cost paid
        only at resume startup.
        """
        model = load_bc_model(path, device, config.value_head_variant)
        optim = _build_optim(model, config)
        scaler = _build_scaler(config, device)
        epoch = fallback_epoch

        obj = torch.load(path, map_location=device, weights_only=True)
        if is_combined_checkpoint(obj):
            if "optim" in obj:
                optim.load_state_dict(obj["optim"])
            if "scaler" in obj:
                scaler.load_state_dict(obj["scaler"])
            epoch = int(obj.get("epoch", fallback_epoch))

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
