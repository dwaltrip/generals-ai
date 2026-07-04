"""The domain training state — what the training loop mutates and checkpoints.

`TrainingState` bundles the model, optimizer, scaler, and epoch counter, and
owns their round trip through a checkpoint (`save` / `from_checkpoint`). The
runner mutates the state in place and saves it at each epoch boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from training.bc.checkpoint import ckpt_name, is_combined_checkpoint
from training.bc.model import BCModel
from training.bc.model_builder import build_model
from training.bc.resume_warmup import WarmupSchedule
from training.bc.storage.checkpoint import load_checkpoint, serialize_checkpoint
from training.bc.train_config import TrainConfig
from training.shared.device import resolve_precision


def _build_optim(model: BCModel, config: TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
    )


def _build_scaler(config: TrainConfig, device: torch.device) -> torch.amp.GradScaler:
    # fp16 is the only AMP mode that needs loss scaling. The scaler is a
    # near-no-op when disabled, so the fp32 path stays unchanged.
    amp_enabled = resolve_precision(config.precision, device) == "fp16"
    return torch.amp.GradScaler(device.type, enabled=amp_enabled)


@dataclass
class TrainingState:
    """A training run's full state — the runtime state it updates, and the
    fixed facts it started from.

    The runtime state is the core state held and updated during the run: the
    model, optimizer, scaler, and `epoch`. `epoch` is the last *successfully
    completed* epoch (resume continues at `epoch + 1`).

    The fixed facts are `config` and `code_sha`. They are set at init, never
    change, and are written into every checkpoint — which is what makes a
    checkpoint self-describing.

    Unfrozen: PyTorch modules are inherently mutable, and `epoch` is advanced in
    place by the runner.
    """

    model: BCModel
    optim: torch.optim.Optimizer
    scaler: torch.amp.GradScaler
    epoch: int
    config: TrainConfig
    code_sha: str
    # Transient: a legacy-checkpoint resume attaches a WarmupSchedule here to
    # ramp the LR while AdamW's variance estimate re-warms. It lives on the
    # state because it drives the optimizer. NOT written to the checkpoint
    # (`to_dict` hand-picks the runtime keys) — the resume path rebuilds it
    # per-process. None on fresh runs and combined-format resumes.
    warmup: WarmupSchedule | None = None

    @classmethod
    def fresh(
        cls, config: TrainConfig, device: torch.device, code_sha: str
    ) -> TrainingState:
        """Build a brand-new state: fresh model/optim/scaler, epoch 0."""
        model = build_model(config.arch).to(device)
        return cls(
            model=model,
            optim=_build_optim(model, config),
            scaler=_build_scaler(config, device),
            epoch=0,
            config=config,
            code_sha=code_sha,
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        config: TrainConfig,
        device: torch.device,
        code_sha: str,
        fallback_epoch: int = 0,
    ) -> TrainingState:
        """Restore a state from a checkpoint, for resuming a run.

        The model loads via `load_checkpoint`. Optim/scaler state is restored
        only when present in the combined format — a legacy bare-state_dict
        checkpoint yields fresh optim/scaler shells (a cold optimizer restart),
        which is why legacy resume pairs with `--legacy-lr-warmup-batches`.

        `epoch` comes from the checkpoint in the combined format. A legacy
        checkpoint has no epoch of its own, so it falls back to `fallback_epoch`
        — the resume path passes the parent epoch parsed from the filename
        (`epoch_NNN.pt`), so numbering stays monotonic across the resume.
        """
        # Arch comes from the checkpoint (you can't reshape restored weights);
        # `config.arch.value_head_variant` is only the legacy fallback, used
        # when the checkpoint has no `arch` key.
        model = load_checkpoint(path, device, config.arch.value_head_variant).model
        optim = _build_optim(model, config)
        scaler = _build_scaler(config, device)
        epoch = fallback_epoch

        # NOTE(ckpt-cfg-refactor-note): second full read — load_checkpoint already
        # read the file for the weights. Fold out with the resume rewiring.
        obj = torch.load(path, map_location=device, weights_only=True)
        if is_combined_checkpoint(obj):
            if "arch" in obj:
                # Redundant second guard — the real gate is run_dir.check_drift
                # (arch is checkpoint-owned).
                # NOTE(ckpt-cfg-refactor-note): keys on the v0 top-level `arch`, so
                # the assert never runs for a v1 checkpoint (arch lives at
                # config.arch). Extend or drop with the resume rewiring.
                assert model.cfg == config.arch, (
                    f"checkpoint arch {model.cfg} != resume config arch {config.arch}"
                )
            if "optim" in obj:
                optim.load_state_dict(obj["optim"])
            if "scaler" in obj:
                scaler.load_state_dict(obj["scaler"])
            epoch = int(obj.get("epoch", fallback_epoch))

        return cls(
            model=model,
            optim=optim,
            scaler=scaler,
            epoch=epoch,
            config=config,
            code_sha=code_sha,
        )

    def to_dict(self) -> dict[str, Any]:
        """The runtime-state portion of a checkpoint."""
        return {
            "model": self.model.state_dict(),
            "optim": self.optim.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": self.epoch,
        }

    def save(self, ckpt_dir: Path) -> Path:
        """Write the checkpoint for the current epoch and return its path.

        One `torch.save` — a load either returns the whole dict or raises.
        Partial writes aren't handled (the operator cleans up).
        """
        path = ckpt_dir / ckpt_name(self.epoch)
        torch.save(serialize_checkpoint(self.to_dict(), self.config, self.code_sha), path)
        return path
